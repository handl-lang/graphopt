+++
id = "PL-006"
title = "Plan 0010 — Search performance"
status = "open"
priority = "medium"
+++

---
title: Plan 0010 — Search performance
type: plan
date: 2026-10-14
status_note: accepted
---

# Plan 0010 — Search performance

Depends on **0006** (API seam), **0007** (backend abstraction),
**0008** (package split), **0009** (composable rule sets).

## Goal

Make the search fast **without changing what it finds** (except where
a change is explicit and opt-in). Three levers, biggest first:

1. **Search less** — compile patterns, target the scan, schedule
   rules, saturate lazily.
2. **Do less per match** — persistent bindings, streaming, an
   incremental congruence closure.
3. **Leave Python** — a native (Rust) core behind the same API.

## Non-goals

* **Rule *sets*** — which laws exist, how they compose, and which
  preset is the default — are **0009**'s job. This plan consumes a
  rule set; it does not define one.
* **Completeness.** Full equality saturation is exponential
  (EXPSPACE-hard in the worst case). Every lever here that trades
  completeness for speed is **opt-in** and documented.

## Lever 1 — search less

### 1a. Compiled matcher (biggest single win)

Today `_match` interprets the LHS pattern tree on every candidate
match (`core.py:500`). Compile each rule's LHS **once**, at rule
load, into a matcher:

* a **discrimination net** keyed by `(root op, arity, child ops)` so
  a class dispatches straight to the applicable rules;
* the compiled node is a closure/decision tree — no per-match tree
  walk, no `isinstance` ladder, no `_pattern_attrs` recomputation.

Preserves semantics exactly: attribute metavariables (`$attr:`),
shared-metavariable consistency, `max_results`.

### 1b. Targeted eligibility

`_candidate_classes` filters by the LHS head op via `_op_classes`
(`core.py:924`). Extend the eligibility key to **arity** and
**child-op constraints** from the compiled pattern, so a rule never
scans a class that cannot match its head shape.

### 1c. Rule scheduling

Order rules within an iteration: cheap/filtering rules first (they
shrink classes and enable early merges), closure-generating rules
last and budgeted. The scheduler is data — a priority per rule —
supplied by the rule set (0009). comm/assoc stay available but are
**opt-in, never default**: they are the Catalan blowup, and 0009's
default preset excludes them.

### 1d. Lazy / best-first saturation (opt-in)

Instead of running to the fixed point, extract **every iteration**
and stop when the cost stops improving for `patience` iterations
(default e.g. 3). Most wins land in the first few iterations.

* Exposed as `search(..., stop="fixed_point" | "improving",
  patience=3)`.
* `"improving"` is a **heuristic** — it can stop before the true
  optimum. The default stays **`"fixed_point"`**; `"improving"` is
  opt-in, and is only reconsidered as the default with corpus
  evidence (no quality loss on `bench/`) — a data decision, not a
  design one.

## Lever 2 — do less per match

### 2a. Persistent bindings / in-place backtracking

`_match` copies `dict(subst)` at every recursion level and every
branch (`core.py:540-602`). Replace with a single mutable binding
table plus a **trail** (bind on descend, undo on backtrack). Removes
the dominant allocation in the hot loop.

### 2b. Stream matches

`matches` materialises `list[dict]` per class per rule
(`core.py:486`). Make it a **generator**; `apply_rule` consumes the
stream and applies as it goes, so an over-large enumeration is never
built. `max_results` still caps it.

### 2c. Incremental congruence closure

`_close_congruence` scans **every** class and node each round,
building an `owner` dict of all enodes (`core.py:703`). Replace with
a **worklist**: only enodes whose children changed since the last
closure are re-keyed, so the pass is proportional to the delta, not
to `E`.

## Lever 3 — native core (Rust)

Port the **search core** to Rust, behind the same Python API:

* union-find, enode storage, matching (compiled), rebuild /
  congruence, and greedy extraction;
* PyO3 bindings.

**Scope boundary — search only.** The proof / certificate layer
(`_track`, `_merge_log`, `_applications`, `certificate`,
`verify_certificate`) and the carrier non-local passes (which mutate
the graph through `union(witness=...)`) stay in **Python**. A run
that needs a certificate uses the Python engine, or the native core
searches and the certificate is produced by the Python path. Porting
the proof machinery is explicitly out of scope.

**Not parallel (yet).** Rules within an iteration are applied
*sequentially* against a mutating graph — rule B matches what rule A
just changed — so naive parallel rule application changes semantics.
Parallelism is deferred; if pursued it needs a snapshot-per-rule +
deterministic-merge discipline, specified on its own.

Shape:

* `catopt_core` stays **pure-Python, zero-dependency** — it remains
  the reference implementation, the correctness oracle, and the
  default engine.
* A new optional distribution **`catopt-native`** (`catopt_native`)
  implements the same `EGraph` **search** surface behind an
  **`Engine`** protocol defined in `catopt_core`.
* **Engine selection is explicit.** `Optimizer(..., engine=...)` /
  `search(..., engine=...)`; the default is the pure-Python
  reference. The native engine is **never auto-detected** —
  installing the wheel must not silently switch engines.
  `stats["engine"]` records which ran.
* `tests/egglog_oracle.py` is the existing differential oracle; the
  native core is validated against the Python one on extracted
  **terms and costs**. Certificates stay a Python-engine concern.

This is the only **step change** (10–100×) — on the search phase
only; levers 1–2 are 2–10×.

## Correctness oracle

Every lever must be checked against a known-good path:

* compiled matcher vs interpreted matcher — identical substitution
  sets on the property-test corpus;
* streaming vs list — identical applications;
* incremental congruence vs full closure — identical canonical
  classes;
* lazy saturation — documented as heuristic, never asserted equal;
* native vs Python — identical extracted term and cost (certificates
  are produced by the Python engine, not the native core).

## Measurement

Profile before changing anything: `bench/suites/search/search_efficiency.py` plus
`cProfile` on a real model (the demo block and a stacked transformer
block). Record per-phase time (match / union / rebuild / extract) so
each lever is attributed. Add a scaling table (enodes vs seconds) to
the bench results.

## Execution stages

Each stage lands independently, suite green, oracle-verified.

1. **Compiled matcher** (1a) — pure Python.
2. **Matcher allocations + streaming** (2a, 2b).
3. **Incremental congruence** (2c).
4. **Targeted eligibility** (1b).
5. **Rule scheduling** (1c) — consumes 0009's priority data.
6. **Lazy saturation** (1d) — opt-in, off by default.
7. **Native core** (3) — `catopt-native`, PyO3, differential oracle.

## Risks

* **Semantic drift in the compiled matcher** — attribute metavars,
  shared-metavar consistency, and `max_results` are easy to get
  subtly wrong; the property corpus is the guard.
* **Lazy saturation changes results** — it must stay opt-in and be
  reported in `stats` (e.g. `stats["stop"]`, `stats["improved"]`).
* **Native build/CI** — wheels per platform (maturin), a
  `--torch-backend cpu`-style matrix entry, and keeping the
  pure-Python fallback working everywhere.
* **Native scope creep** — the temptation to port proofs,
  certificates and the non-local passes. Explicitly out of scope;
  the native core is the search loop only.
* **Two implementations to maintain** — mitigated by the oracle: the
  Python core is the spec, the native core the accelerator.

## Decisions (resolved)

1. **Native package** — `catopt-native` / `catopt_native`. Role-named,
   not language-named (a future rewrite in another language does not
   invalidate it).
2. **Engine selection** — an `Engine` protocol in `catopt_core`,
   passed explicitly (`Optimizer(..., engine=...)` /
   `search(..., engine=...)`). The pure-Python core is the default
   (the library's own reference, not an external assumption). The
   native engine is **never auto-detected**; `stats["engine"]`
   records which ran.
3. **Default `stop`** — `"fixed_point"`. `"improving"` is opt-in and
   revisited only with corpus evidence.

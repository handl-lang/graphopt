+++
id = "PL-025"
title = "Plan 0029 — the handl ↔ catopt boundary spike"
status = "open"
priority = "medium"
+++

---
title: Plan 0029 — the handl ↔ catopt boundary spike
type: plan
date: 2026-10-15
---

# Plan 0029 — the handl ↔ catopt boundary spike

Depends on: **ADR 0010** (handl *is* the kernel — one language, many
interpretations), ADR 0009 (the ladder; the calculus stage is gated on
0010), ADR 0008 (combinator presentation), and the existing bridge
`catopt_discovery.handl`.
External: `~/GitHub/handl-lang/host` (Peregrin) — `handl-language`,
`handl-kit-arrow-lower`, `plugin-engine-{exec,typecheck,transform-rewrite}`,
`handl-runtime/src/vm_primitives.rs` (host functions).

## Motivation

ADR 0010 decided that **handl is the kernel** rather than a Python
reimplementation, on the strength of three facts: handl already is the
arrow calculus, its language is data (ADR-0044), and it supports **host
functions** — so catopt's tensor primitives can live outside the kernel.
That is an *architectural* claim.  It becomes a fact only when measured,
and the three measurements are exactly ADR 0010's falsification #0.

**Spike, not implementation**: the goal is to answer three questions and
produce a go / split / no-go decision, not to build the integration.

## What already exists (don't rebuild)

- `catopt_discovery.handl` — `store_to_handl(store)` emits handl
  fragments: op cells → `AstShapeDef` rows, law/object cells →
  `TransformDef` rows, handle cells → `!Native` records; every row
  carries a `catopt:` extension; `cell_from_handl` round-trips
  (provenance `imported`).  **The seam exists and was built
  deliberately.**
- `lab/bridge/combinator.py` — the monoidal/trace prototype
  (`BASIS`, `dag_to_comb`/`comb_to_dag`), which is what a `Term`
  encoding would replace.
- `catopt-native` — the precedent for a Rust peer: optional, excluded
  from the workspace, maturin-built.
- handl's engines and its own suite (4,288 tests) — the kernel side is
  verified by handl, not by catopt.

## The spike — four questions, cheapest first

### Q0 · Can catopt's *language* be expressed as handl data?  (hours — decisive)

Encode **one** catopt law as a handl `TransformDef` (YAML) and **one**
catopt term shape as an `AstShapeDef`; load them into a `Language` via
handl's `merge_fragments`; then run handl's `RewriteEngine` over a small
term and see the law fire.

- **Pass**: the fragment merges (no `validate_transforms` error), the
  rewrite fires, and the result is the term catopt's own `apply_rule`
  produces for the same input (compare structurally + numerically).
- **Fail**: the vocabulary cannot express it — **name the missing
  piece** (a shape role, a `when` condition, a value form).  A named gap
  is a result; a vague "it doesn't fit" is not.
- Cheapest candidate for a first law: a *structural* one
  (`assoc`/`comm`-class, no attrs), because it exercises shapes and
  matching without touching guards or values.  Second: a law with a
  guard (`cond`) — that is where the data-only ratchet (ADR 0010 #2)
  gets its first evidence.

### Q1 · Can a catopt *tensor value* cross the boundary?  (a day)

Register **one** catopt tensor op (e.g. `matmul` or a pointwise op) as a
handl **host function**; evaluate a 3–5 node tensor term through handl's
exec; verify numerically against catopt's own `eval_term`.

- **Pass**: the value crosses intact — dtype, shape, device, and
  fp64-exact (or within the documented tolerance) agreement with
  catopt's evaluation.  The host function must be able to *read and
  write* a tensor (not only scalars).
- **Fail**: tensors cannot cross without a lossy encoding (e.g. only
  scalars/JSON) — **name the boundary** and record whether the fallback
  (catopt owns the tensor exec interpretation) is required.
- This is the question the ADR's "host functions" claim rests on, and it
  is *not* answerable by reading — the docs say host functions exist;
  whether they carry a tensor is the measurement.

### Q2 · What does the boundary cost?  (a day)

Time the two paths on a realistic term (~100 ops, a shape catopt's bench
uses): (a) catopt's existing executors (generic / batched), (b) the same
term through handl exec with host-fn dispatch.

- **Pass criterion is *not* "faster".**  The kernel serves **search and
  rewriting**; the delivered module runs the hot loop.  So the criterion
  is: **host-fn dispatch must not dominate rewrite throughput** —
  measure rewrites/second through handl's engine on a small law set and
  compare with catopt's e-graph on the same set.  Pass = same order of
  magnitude; the exec timing is **informational**.
- **Fail**: rewriting through the boundary is an order of magnitude
  slower than in-process — then the split in Q1's fallback applies, or
  the boundary needs a batched crossing.

### Q3 · Which direction is the integration?  (a day)

Prototype the *shape* of the dependency, not the whole thing:

- **(A) catopt embeds handl** — a PyO3 peer like `catopt-native`
  (`maturin`), catopt calls handl for term/rewrite/typecheck.
- **(B) handl drives catopt** — catopt's language is a handl language;
  tensor ops are host functions registered by a catopt plugin; catopt's
  discovery/RL become handl tools.
- **Pass**: one of them is buildable in catopt's CI story (wheel
  availability, no 8 GB cargo target in the default path) **and** has a
  named **seam test** (round-trip + semantics agreement — the new
  verification boundary handl's suite cannot cover).
- **Fail**: neither fits without weakening a catopt invariant (core
  zero-dependency, the coverage floors) — then record which invariant
  blocked it and what would have to change.

## Decision the spike produces

| outcome | meaning |
|---|---|
| **go (A)** | handl as a Rust peer; catopt keeps language fragments + host fns + adapters |
| **go (B)** | catopt becomes an application of the platform |
| **split** | handl owns language + rewrite + typecheck; **catopt owns the tensor exec interpretation** (the ADR's fallback) |
| **no-go** | the boundary cannot carry tensor semantics faithfully — the kernel stays a catopt prototype, and 0009 stage 4 is not justified |

## Order and budget

Q0 → Q1 → Q3 → Q2 (fail-fast: expression first, values second,
integration shape third, cost last — because cost only matters if the
first three pass).  **Bounded: ~3 days.**  Each question is a retro with
its measured answer, committed as it lands.

## Risks (measured, from this box and handl's own guide)

- **Disk/toolchain**: handl's debug target is ~8.5 GB
  (`CARGO_PROFILE_DEV_DEBUG=0` is the documented mitigation); this host
  was at 93% full today (32 GB free after cleanup).  Budget disk before
  starting.
- **Memory**: handl's guide says `cargo nextest -j 8` OOMs an 11 GB box;
  `-j 2` peaks ~5 GB.  The spike must respect that, and catopt's own
  suite must not run concurrently.
- **Second toolchain in catopt's CI** — a real cost that Q3 must price,
  not assume.
- **Duplication temptation**: if the spike drifts into "reimplement a
  little of it in Python for convenience", that is the drift ADR 0010
  forbids — record it and stop.

## Falsification of the spike itself

If Q0 passes but Q1 fails, the *right* answer is the split (handl owns
language/rewrite/typecheck), and that is a **success** for the spike: it
converted an architectural claim into a measured boundary.  The spike
fails only if it produces no named gap and no decision.

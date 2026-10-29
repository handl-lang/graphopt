---
title: Retro — the saturation, and the sharing-aware optimum, were not reproducible across processes
type: retro
date: 2026-10-15
---

# Retro — the saturation, and the sharing-aware optimum, were not reproducible across processes

**Commit**: `abf6114` (catopt `main`).
**Decides**: the staged saturation and the sharing-aware extraction are
now **order-independent** — the same inputs give identical enode counts
and identical extracted costs in every process, with or without
`PYTHONHASHSEED`.  The gate-3 retro's headline recovery
(*455,752 → 432,696, −5.06 %*) is **superseded**: 432,696 was a
hash-order artifact, and the reproducible recovery is **451,128
(−1.01 %)**.  The gate-3 *conclusion* (the fix does not close the gate)
is unaffected.

## The defect as reported

> the staged saturation is not reproducible across processes
> (set-iteration rule-budget order): 9733/9752/9756 enodes.  The
> winner's cost is stable (455,752) but the sharing-aware local optimum
> varies (451,128 unseeded vs 432,696 at `PYTHONHASHSEED=0`).

Reproduced and localised: **both halves were real, and they had
different causes** — the enode counts varied in the saturation
(`catopt_core`), the local optimum varied in the *extraction*
(`lab/bridge/sharing_extract.py`).  Fixing only the saturation leaves
the reported cost variation intact.

## The before-state (measured)

The gate-3 workload, staged saturation
(`om_shared_state_stages(4)` → `reach` then `reassoc` on one e-graph,
`iters=10`, `om_shared_state_budgets(4, kway=200)`,
`add_term(om_shared_state_probe._export(64))`), each run a **fresh
process**:

| `PYTHONHASHSEED` | enodes | classes | greedy | sharing-aware |
|---|---|---|---|---|
| unset | 9726 | 2324 | 455,752 | 432,696 |
| 0 | 9733 | 2324 | 455,752 | 432,696 |
| 1 | 9749 | 2328 | 455,752 | 432,696 |
| 2 | 9743 | 2327 | 455,752 | **451,128** |
| 3 | 9728 | 2324 | 455,752 | 432,696 |
| 4 | 9761 | 2331 | 455,752 | — |
| 5 | 9734 | 2326 | 455,752 | — |

(the sharing-aware column was measured in a second pass: seeds
0/1/3/unset gave 432,696 with 5 overrides and 1307 trials, seed 2 gave
451,128 with 3 overrides and 898 trials).  Seven distinct enode counts
from one input.

## The mechanism

`EClass.nodes` is a `set[ENode]` (`packages/catopt-core/src/catopt_core/egraph/types.py`),
and `ENode.__hash__` hashes `(op, children, tuple((k, repr(v))))` — the
`op` is a `str`, so the hash, and therefore the set's *iteration order*,
changes with `PYTHONHASHSEED`.  Four reads of that set reached a
decision:

1. **The matcher's member enumeration — the budget decision.**
   `core._nodes_of` (pre-fix `core.py:925`,
   `[n for n in eclass.nodes if n.op == op]`) and `core._mnodes`
   (pre-fix `core.py:706`), consumed by `_m_stream` / `_m_bounded`.
   `apply_rule` computes `match_cap = min(enode_budget - spent, _MATCH_CAP)`
   per candidate class and stops between classes once
   `spent >= enode_budget`; `_m_bounded` drops a member's substitutions
   **entirely** when its enumeration crosses the cap.  So which matches
   land before a rule is suspended — and therefore the truncation point
   of every budgeted rule, and the enode count — followed the set order.
2. **The congruence worklist — the class-id space.**
   `core._close_congruence` (pre-fix `core.py:1058`) drained
   `_cong_pend`, a `set[tuple[int, ENode]]`, with `set.pop`.  The
   batch's merge order decides which root wins each union, i.e. **which
   class id survives**; class ids order the dirty frontier
   (`run`: `search = sorted({self.find(e) for e in self._dirty})`,
   `core.py:1612`), which orders budget consumption *across* classes.
3. **The representative term — the side-condition binding.**
   `core.any_term` (pre-fix `core.py:1165/1173`) and `core._min_term`
   (pre-fix `core.py:1207/1220`).  Both take the first acceptable
   member, `_min_term` with a strict `<`, so equal-size members resolve
   by set order — and that term is exactly what a rule's `check` /
   `derive` sees.
4. **The extraction-side half.**  `lab/bridge/sharing_extract.py:142`
   scanned `eg.get_class(cls).nodes` and kept the *first* strictly
   improving member of each class.  The coordinate descent's local
   optimum therefore moved with the scan order — the
   432,696 / 451,128 split.

Leaks 1 and 2 are **independent and both necessary**.  Measured
pre-fix, on the same workload: sorting only the matcher order gave
9788 / 9787 / 9790 enodes (still varying); sorting only the congruence
order gave 9729 / 9750 / 9741 (still varying); both together gave
9784, stable.  Leak 3 is **latent** on this workload (1+2 already
suffice) but is the same class of defect — an equal-size tie in the
representative walk — and is pinned by a unit test rather than left to
bite the next workload.

### Minimal reproduction (leak 1)

A class holding 16 distinct `add` members, and a rule under
`enode_budget=1`: the bounded matcher admits exactly the first
enumerated member, so the single created `mul` is the RHS of whichever
member the scan yields first.  Pre-fix, seeds 0–5 created
`mul(2,0)` / `mul(20,0)` / `mul(4,0)` / `mul(4,0)` / `mul(10,0)` /
`mul(18,0)`; post-fix it is `mul(0,0)` in every process.  This is
`tests/test_egraph_determinism.py::test_budgeted_apply_rule_lands_on_the_canonical_member`.

## The fix

One canonical member order —
`catopt_core.egraph.types._member_key(node) = (node.op, node.children, repr(node.attrs))`
— applied at every decision-reaching read.  It is not a new
convention: it is the key `extract_best` already used for its own
tie-break, and the same key `lab/probes/coordination_probe._node_key`
had already duplicated.  Sites changed:

- `egraph/core.py` — `_nodes_of`, `_mnodes`, `any_term`, `_min_term`,
  and `_close_congruence` (the worklist batch is now
  `sorted(pend, key=lambda p: (_member_key(p[1]), p[0]))`).
- `egraph/extract.py` — `extract_best` and `extract_min_depth` (deduped
  to the named key), `extract_alternatives` (canonical scan ⇒ the
  stable cost sort breaks ties canonically), `diverse_classes`,
  `_paired_subset_term` (`min` keeps the first of an equal-score run),
  `_locate` ("first structural match" is lenient in attrs and children,
  so two members can match).
- `lab/bridge/sharing_extract.py` — the coordinate-descent scan.
- `lab/probes/gate3_state_spelling_probe.py` — `_class_members`, whose
  tie table is itself a reported measurement.

### What changed about the search

**Nothing about *what* it does.**  The same rules, the same
`rule_budgets`, the same stop condition, the same fixed-point
definition.  What changed is which of several *order-arbitrary*
choices is made: previously "whichever this process's hash order
produced", now one canonical order.

The consequence is that the numbers moved from an arbitrary hash-order
outcome to the canonical one — at seed 0 the staged saturation went
9733 → 9784 enodes, and the sharing-aware optimum 432,696 → 451,128.
Neither value is "the" answer for a *budget-truncated* saturation (the
budget makes the truncation point a choice); only the new one is
reproducible.  Two notes on the scope of that:

- For an **unbudgeted** rule set the closure is order-independent, so
  its result is unchanged by this fix.
- With `check` / `derive` hooks the search is **not** confluent — the
  binding the guard sees decides which rewrites fire — which is
  precisely why leak 3 matters and was fixed rather than declared
  harmless.

The 9784 enode figure is therefore the new baseline for this workload;
the gate-3 retro's 9733 should be read as a hash-order artifact.

## Verification (after)

Same staged saturation, **8 fresh processes** (seeds 0–6 and unset):

| `PYTHONHASHSEED` | enodes | classes | greedy | sharing-aware | overrides | trials |
|---|---|---|---|---|---|---|
| 0 | 9784 | 2336 | 455,752 | 451,128 | 3 | 898 |
| 1 | 9784 | 2336 | 455,752 | 451,128 | 3 | 898 |
| 2 | 9784 | 2336 | 455,752 | 451,128 | 3 | 898 |
| 3 | 9784 | 2336 | 455,752 | 451,128 | 3 | 898 |
| 4 | 9784 | 2336 | 455,752 | 451,128 | 3 | 898 |
| 5 | 9784 | 2336 | 455,752 | 451,128 | 3 | 898 |
| 6 | 9784 | 2336 | 455,752 | 451,128 | 3 | 898 |
| unset | 9784 | 2336 | 455,752 | 451,128 | 3 | 898 |

Identical in every process, and identical *across* the two passes
(the pre-sharing-fix saturation runs and the post-fix ones).  The
sharing-aware term is numerically equivalent to the greedy one —
`max|Δ| = 2.220e-16` through `TorchSink` on the same weights and feeds
— and the gate-3 abstraction verdict is unchanged (`irregular`, first
difference `block1/block2/add[1]/mul[1]: op om_decayed vs add`).

## Superseded numbers in the gate-3 retro

`project/retros/gate3-extraction-state-spelling.md` reports
455,752 → 432,696 (−5.06 %, 5 overrides) **and** a read-sequence change
(block 4 `reexp` → `shared`).  With the canonical order:

| | superseded | reproducible |
|---|---|---|
| e-graph | 9733 enodes | **9784 enodes** |
| sharing-aware cost | 432,696 | **451,128** |
| recovery | −23,056 (−5.06 %) | **−4,624 (−1.01 %)** |
| overrides | 5 | **3** (classes 355, 347 `split`; 93 `matmul`) |
| read sequence | `reexp\|reexp\|fold\|fold\|shared\|shared\|shared` | `reexp\|reexp\|fold\|fold\|reexp\|shared\|shared` (unchanged from greedy) |
| abstraction verdict | irregular | irregular (same first difference) |
| numeric max\|Δ\| | 2.2e-16 | 2.2e-16 |

The retro's **mechanism** is unaffected: the state-read spelling *is* a
cascade of per-class cost ties (indifferent to within 0.03 %), and the
greedy per-class pass *is* off the cost its own model assigns.  Those
are properties of the e-graph, not of the scan order.  Only the size of
the recovery, and the claim that the pass unified the large-chunk
family's read, were overstated.  A `Correction` note has been added to
`lab/bridge/sharing_extract.py`'s module docstring; the gate-3 retro
itself is left for its own author to amend (a concurrent session is
editing the gate-3 retros in this checkout).

## Tests

`tests/test_egraph_determinism.py` — 4 tests, <1 s:

- `test_member_reads_are_in_canonical_order` — `_nodes_of` returns the
  canonical order, and the cached bucket is rebuilt canonically after a
  mutation (pre-fix: `False`).
- `test_budgeted_apply_rule_lands_on_the_canonical_member` — the
  minimal reproduction above, with the exact enode count (33) and the
  exact created member (pre-fix: a different member per seed).
- `test_representative_choice_is_canonical` — `any_term` / `_min_term`
  break an 8-way equal-size tie canonically.
- `test_budgeted_saturation_is_hash_seed_independent` — runs the same
  small saturation in **fresh processes** under `PYTHONHASHSEED`
  0/1/2/3/unset and requires byte-identical output.  This is the
  seed-independent check; the first three are in-process and assert the
  canonical invariant (pre-fix they fail unless the process's hash
  order happens to *be* canonical).

## Latent sites deliberately not fixed here

The same pattern exists in other `lab/` probes that read `EClass.nodes`
in a decision-reaching way:
`lab/probes/om_decayed_kway_deliver_probe.py:150`
(`fused = [n for n in ... if n.op == ...]`),
`lab/probes/om_decayed_kway_probe.py:96`,
`lab/probes/coordination_probe.py:96/119/142` (partly canonical already,
via its own `_node_key`), `lab/train/train_graph_policy.py:132`.  Each
is a separate measurement with its own numbers; changing them blind
would silently move those numbers, so they are listed rather than
touched.  The `catopt_core` engine — which every measurement sits on —
is clean.

## Gates run

- `.venv/bin/ty check` — clean.
- `.venv/bin/ruff check` / `ruff format --check packages lab scripts
  examples` — **one pre-existing error, not from this change**:
  `lab/bridge/index_abstract.py:524 D401`, an uncommitted in-progress
  change by a concurrent session in this checkout (`ruff format` would
  also rewrap that file).  Every file in this commit is clean under
  both.
- `.venv/bin/vulture` — clean.  `.venv/bin/lint-imports` — 4 kept, 0
  broken.  `.venv/bin/bandit -c .bandit.yaml -r packages` — 0 findings.
  `.venv/bin/semgrep --config .semgrep.yml packages` — 0 findings.
  `scripts/radon_ratchet.py` — ok (4712 functions; the new
  `_member_key` is well under rank C, and `extract_alternatives` was
  kept at its recorded complexity by sorting the already-canonical scan
  rather than adding a comprehension).
- Scoped pytest slices — 311, 277, 406, 143, 118, 88 passed
  (`test_egraph*`, `test_property_*`, `test_extract_memo`,
  `test_fusion_extract`, `test_cert*`, `test_cost*`,
  `test_lemma_certificates`, `test_index_abstract`,
  `test_combinator_bridge`, `test_loop_propose`, `test_*laws*`,
  `test_optimize_*`, `test_e2e_pipeline`, `test_trace_pipeline`).
- **4 failures in `tests/test_laws_structure.py` are pre-existing** —
  they fail identically against unmodified `HEAD` code (the law-bank
  count assertions expect 58 axioms and see 59) and are unrelated to
  this change.
- The full suite was **not** run (memory/time budget, per AGENTS: this
  host has OOMed, and the suite is 6,341 tests / ~28 min).

## Cost

One 15 s staged saturation (~1 GB peak RSS) plus a ~46 s sharing-aware
pass per process; the 8-process verification is ~9 min serial.  The
regression test is <1 s (four short subprocesses).

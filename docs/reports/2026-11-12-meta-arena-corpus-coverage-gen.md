# 2026-11-12 — meta-arena corpus coverage generation (Stage-1 lever 1, corpus half)

Self-driving Stage 1, lever 1: **coverage-driven corpus generation**.
The corpus half only — law auto-admission is a separate later lever
and was not touched.  Prior rounds extended `gen_cases` *by hand*
(T-009 added `square`/`dual`/`tri` kinds); this round makes the
extension **mechanical**: enumerate the op/attr/handler/rule space
the arena machinery can express, diff it against the live corpus,
and emit real cases for the measured holes — re-runnable as the
vocabulary grows.

## What landed

* **`mlopt_discovery/corpus_cov.py`** (new, ~970 lines) — the
  machinery:
  * `vocabulary()` — the expressible space: `genkernel.ELEMENTWISE`
    (clamps included), `reduce_heads()` (`REDUCE_DIM_OPS` +
    `softmax`), every op named by a shipped law side (`DEFAULT` +
    `SCAN_LAWS`/`SCAN_DIAG_LAWS` — the `apply`/`aff` carriers), the
    `HANDLERS` patterns/kernels, and the `opmeta` view/pointwise
    classes; plus `ATTR_SCHEMA` `(op, attr)` pairs, handler tags,
    and the 153-name rule vocabulary.
  * `coverage_gaps(cases)` — the diff.  Ops/attrs counted over every
    subterm; a handler tag is *covered* when its concrete pattern
    occurs (the `claim` precondition); a rule is *armed* when its
    LHS matches a subterm **of a board whose ruleset carries it**
    (the `fire`/`saturate` payoff precondition — stricter and more
    honest than "matches somewhere").
  * `coverage_builders()` / `gen_cases_coverage(seed, n)` — 16
    gap-targeting kinds emitted as real `(name, term, rules, decls,
    supported)` probe tuples; `probe_case()` runs each through
    `MetaArena`'s own observe/legal/referee path.
  * CLI: `python -m mlopt_discovery.corpus_cov [--cases N --seed S
    --stream base|coverage|all]`.
* **`meta_player.gen_cases(seed, n, coverage=True)`** — the default
  stream now appends the 16 coverage kinds to the ten hand kinds;
  `coverage=False` reproduces the pre-extension stream exactly (the
  A/B control).  Kinds: `law_silu`, `law_softsign`, `law_square`,
  `law_softmax`, `law_glu`, `law_rsqrt`, `law_mm`, `law_split`,
  `law_unsq`, `law_idem`, `scan`, `ew_gap`, `reduce_gap`,
  `view_gap`, `kernel_ops`, `mixed_gap`.
* **Supported-bound fix (found by this work).**  Three coverage
  builders initially shared `_SWIGLU_SUP` — a bound derived from the
  silu/swiglu site that omitted the new ops, so those boards priced
  `+inf`/infeasible and oracle lines declined spuriously (310/400
  declines in the first bake).  `_sup(term, rules)` now computes each
  board's bound from its own term via `ma._supported_or`.
* **`meta_prior.META_ARENA_PLAYER_WEIGHTS` re-baked** — see below.

## The gap census (400-case streams, seed 0)

| metric | base (`coverage=False`) | extended (`coverage=True`) |
|---|---:|---:|
| ops covered / missing | **7** / 157 | **57** / 107 |
| attrs covered / missing | **0** / 91 | **46** / 45 |
| handlers covered | 3/3 | 3/3 |
| rules armed (of 153) | **1** | **38** |

The base corpus's seven ops were `abs, add, div, matmul, mul,
sigmoid, tanh` — no attributes at all.  The single armed rule is
`prim:synop_bb8c54c201` (the `control` boards' `DEFAULT` ruleset);
the paying folds (`silu_fold`, `softsign_fold`, `mul_square`) are
deliberately stripped from the handler-test boards, so the corpus
never exercised the `fire`/`saturate` apply-law payoff dimension.
The extended stream arms 38 rules including `aff_lift`,
`aff_unlift`, `affd_lift_*`, `double_neg`, `id_add`, `id_mul`,
`pow_to_rsqrt`, `select_mul`, `softmax_fold`, `split_scalar_add_*`,
`sub_to_add`, `transpose_noop`, `weight_factor_*`, `mul_unsq_*`,
`silu_expand`, `square_expand`, `glu_fold`, `assoc_*` — the apply-
law tier the meta-arena's `declare`/`handle`/`fire` moves exist for.

**Honest remainder**: the 107 still-missing ops are dominated by
`synop*` minted-law names and genuinely new classes — `conv2d`,
`sdpa`, `batch_norm`, `dropout`, `glu`, comparisons/logicals,
`masked_fill`, `narrow`, `permute`, etc.  The `ew_gap`/`reduce_gap`
cycles absorb new `ELEMENTWISE`/reduce members automatically; new
*classes* need a new template (the documented extension seam).

## Verification — every kind through the real arena

`probe_case` builds each board exactly as `meta_player._board`
(`count_cost`, the case's ruleset and per-board `supported`), then
plays `saturate → extract` under `run_episode`.  All 16 kinds:

| kind | legal | specs | sites | claimable | sat→extract reward | cert |
|---|---:|---:|---:|---:|---:|---|
| law_silu | 139 | 3 | 3 | 1 | +0.149 | ✓ |
| law_softsign | 138 | 4 | 4 | 0 | +0.299 | ✓ |
| law_square | 137 | 2 | 2 | 1 | −0.103 | ✓ |
| law_softmax | 138 | 4 | 4 | 0 | +0.299 | ✓ |
| law_glu | 139 | 5 | 5 | 0 | +0.149 | ✓ |
| law_rsqrt | 137 | 3 | 3 | 0 | +0.149 | ✓ |
| law_mm | 136 | 2 | 2 | 0 | +0.148 | ✓ |
| law_split | 138 | 4 | 4 | 0 | −0.100 | ✓ |
| law_unsq | 136 | 2 | 2 | 0 | +0.146 | ✓ |
| law_idem | 136 | 2 | 2 | 0 | +0.232 | ✓ |
| scan | 158 | 2 | 2 | 0 | −0.129 | ✓ |
| ew_gap | 133 | 1 | 1 | 0 | −0.100 | ✓ |
| reduce_gap | 134 | 2 | 2 | 0 | −0.100 | ✓ |
| view_gap | 133 | 1 | 1 | 0 | −0.100 | ✓ |
| kernel_ops | 135 | 1 | 1 | 0 | −0.100 | ✓ |
| mixed_gap | 140 | 4 | 4 | 1 | −0.100 | ✓ |

A deliberate mix of paying boards (`law_softsign`/`law_softmax`
+0.30), claimable boards (`law_silu`, `law_square`, `mixed_gap`),
and honest declines — the coverage kinds are boards the referee
actually prices, not padding.  Oracle-line spot checks (30 seeds
per kind): `ew_gap` +0.070 mean (16/30 paying), `view_gap` +0.116
(21/30), `mixed_gap` +0.194 (23/30), `reduce_gap` +0.033 (10/30) —
real payoff mass exists on the new kinds.

## Held-out meta-player impact

Bake recipe (unchanged): `meta_player.main --imitate-epochs 3
--epochs 5 --train-cases 400 --eval-cases 40 --budget 24
--temperature 1.2 --epsilon 0.25`.  `main` evals
`gen_cases(seed + 7919, 40)`, so seed 23418 lands held-out on
`gen_cases(31337, 40)` — the recipe's two offsets.

Artifacts: `project/reports/corpusgen_s0.json`,
`corpusgen_s23418.json` (pre-fix `s0` bake discarded — its eval
priced `cost=inf` under the shared-bound bug).

Mean reward, learned / learned-greedy (40 held-out cases each):

| weights | stream | @7919 | @31337 |
|---|---|---:|---:|
| shipped prior (pre-change) | extended | +0.086 / +0.101 | +0.115 / +0.131 |
| seed-0 bake | extended | +0.109 / +0.123 | +0.143 / +0.151 |
| **seed-23418 bake (SHIPPED)** | extended | **+0.104 / +0.123** | **+0.144 / +0.156** |
| shipped prior | base-only | +0.299 / +0.329 | +0.293 / +0.326 |
| seed-0 bake | base-only | +0.276 / +0.282 | +0.288 / +0.288 |
| seed-23418 bake | base-only | +0.290 / +0.311 | +0.288 / +0.308 |

**Decision: re-baked, and it earned.**  On the stream `gen_cases`
now actually produces, both bakes beat the shipped prior by
+0.02..+0.03 on *both* offsets and *both* eval arms — replicated,
not noise.  On the legacy base-only stream the cost is −0.01..−0.05
(s23418 retains better: −0.009..−0.018 sampled, −0.009..−0.018
greedy vs s0's −0.023..−0.047), so the 23418 table ships.  The base
stream reproduces the documented +0.30/+0.33 for the old weights —
the coverage kinds, not a policy regression, account for the lower
absolute extended-stream numbers (more decline boards in the mix).

Watch-item: `op:saturate` +3.36 and `a:unbounded` +5.32 are large
logits — real on both arms/offsets, but the coverage boards reward
saturate-then-extract lines the base corpus never priced; if a
future corpus shift flips them, re-measure before trusting.

Imitation: 1416/1416 (s23418) and 1431/1431 (s0) demonstrated moves
cloned; oracle plans ~113-114 claim-line / ~286-287 decline /
~49-52 spelled-pick of 400 — the decline share roughly doubled vs
the base corpus, as expected for gap-targeting boards.

## Files changed

* `packages/mlopt-discovery/src/mlopt_discovery/corpus_cov.py` — new.
* `packages/mlopt-discovery/src/mlopt_discovery/meta_player.py` —
  `gen_cases(..., coverage=True)` arm + docstring.
* `packages/mlopt-discovery/src/mlopt_discovery/meta_prior.py` —
  re-baked weight table + bake record.
* `tests/test_discovery_corpus_cov.py` — new (vocabulary keys,
  subterm traversal, baseline-vs-extended delta, real-arena
  execution, determinism, control-stream isolation, per-board
  supported bounds; `corpus_cov.py` at 100% scoped coverage).
* `tests/test_discovery_meta_player.py` — base-kind assertions pin
  `coverage=False`; `test_gen_cases_kinds` asserts both arms.
* `project/reports/corpusgen_{s0,s23418}.json` — bake artifacts.

Checks: `pytest -m "not slow"` on the meta/corpus slice green
(318 tests); ruff check/format and `ty check` clean on touched
files.  Code left uncommitted.

## Honest scope / next

This is the corpus-generation machinery plus the first generated
cases — not law auto-admission.  The remaining 107 ops / 45 attrs /
115 unarmed rules are the next rounds' input: `coverage_gaps`
re-reports them on every run, and `gen_cases_coverage` emits only
what the builders cover — widening vocabulary coverage further is
mechanical for ops/attrs already in the grammar, and a template
authoring task for new classes (`conv2d`, `sdpa`, comparisons).

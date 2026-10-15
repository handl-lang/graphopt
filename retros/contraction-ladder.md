# Contraction-ordering ladder — does greedy lose, and what closes it?

Plan 0016 stage 7 (ADR 0003, the EVALUATION dimension).  The
contraction-ordering precondition
(`contraction-precondition.md`) established that the one-step greedy
cost-model oracle is *provably* suboptimal on matrix chains and general
tensor networks.  This retro scores the players on a **hard-instance
set**, against the exact DP optimum, as a bench suite
(`bench/suites/evaluation/contraction_ladder.py`).

The question it answers:

> Does greedy lose on contraction ordering, and does anything close
> the gap to the exact optimum?

**Yes, greedy loses; yes, equality saturation closes it exactly.**
The consequence is the honest one: on this space a learned policy has
**no quality headroom** above the non-learned players — the residual
headroom is *search cost*, not quality.

Reproduce: `python -m bench run contraction_ladder` (~18 s,
CPU-only, seeded; `--quick` shrinks to ~7 s).  Direct:
`python -m bench.suites.evaluation.contraction_ladder`.

## The hard-instance set

Both families are **seeded random draws from the measured
distribution**, not hand-picked winners — ties are kept and counted.

* **Matrix chains** — 60 draws, 4–8 matrices, dims 1..40 (seed 0).
  This is the hard regime (2–3-matrix chains are trivially optimal).
* **General tensor networks** — 60 draws, 3–6 tensors, mixed ranks
  over 3–7 shared indices, sizes 2–12 (seed 1).

The subset DP is **re-validated inside the suite** against exhaustive
enumeration (40 random cases, 0 mismatches) so the "optimal" the ladder
scores against is trustworthy.

## The ladder

Players (chains — the priced e-graph rule space over `Var` leaves,
`flops_cost` units):

* `greedy-1step` — the one-step rule-delta oracle
  (`catopt_core.trajectories.rule_samples`): apply the largest-delta
  rule once.
* `greedy-episode` — the RL script's greedy player (horizon 6,
  patience 2).
* `saturate` — full equality saturation: the ceiling this rule set
  can reach.
* `optimal` — the exact `O(n³)` interval DP (`2 ×` the
  scalar-multiplication DP; `flops_cost` prices a matmul at `2·M·N·K`).

### Chain ladder (n=60)

| player | mean ratio | median | % at optimum | worst | mean rank |
|---|---|---|---|---|---|
| greedy-1step | 1.276 | 1.000 | 63 % | **4.10×** | 3.04 |
| greedy-episode | 1.000 | 1.000 | 98 % | 1.007× | 2.34 |
| saturate | 1.000 | 1.000 | **100 %** | 1.000× | 2.31 |
| optimal | 1.000 | 1.000 | 100 % | 1.000× | 2.31 |

Ties the optimum: `saturate` 60/60, `greedy-episode` 59/60,
`greedy-1step` 38/60.  Ranks are **shared on ties** (all four tie on
38 instances → each scores rank 2.5), which is why the top three share
a mean rank of 2.31 — the ratio / `%opt` / worst columns are the
meaningful ones.

### Network ladder (n=60)

There is no e-graph spelling of a general tensor network (the
precondition retro's documented limitation), so the priced players do
not apply; the net family measures the textbook baseline directly
(`greedy-classic` = contract the cheapest pair; `optimal` = the subset
DP).

| player | mean ratio | median | % at optimum | worst | mean rank |
|---|---|---|---|---|---|
| greedy-classic | 1.574 | 1.047 | 33 % | **5.50×** | 1.83 |
| optimal | 1.000 | 1.000 | 100 % | 1.000× | 1.17 |

Ties the optimum: `greedy-classic` 20/60.

## Headline finding

* **`saturate == optimal` on every chain (60/60).**  The priced e-graph
  rule space is *exact* here: `assoc_matmul` closes the full bracketing
  space, so saturation reaches the DP optimum with no gap.  This is a
  **WIN** for the rule set, and it is the answer to "does anything
  close the gap" — yes, exactly.
* **Greedy loses.**  The one-step oracle (`greedy-1step`) is at the
  optimum on only 38/60 chains (**loses 37 %**, worst **4.1×**); the
  multi-step episode (`greedy-episode`, h6/p2) nearly closes it —
  loses **1.7 %** (1/60), worst 1.007×; the classic network greedy
  loses **67 %** (40/60), worst **5.5×**.

So: greedy **does** lose, most sharply the one-step oracle and the
classic network greedy; and **saturation closes the gap exactly** on
every chain.

### Consequence for a learned policy — search cost, not quality

Because `saturate` already ties the DP optimum on every chain, a
learned `Policy` **cannot beat the non-learned players on quality** in
this space — there is nothing left to win.  Any remaining headroom is
**search cost**: reaching the optimum in fewer rule applications than
saturation's fixed point.

That headroom is *small at these sizes*, measured honestly
(best-of-3 wall time per instance, mean / max ms):

| player | chain mean | chain max | net mean |
|---|---|---|---|
| greedy-1step | 13.9 | 20.1 | — |
| greedy-episode | 44.6 | 83.7 | — |
| saturate | 13.5 | 52.2 | — |
| optimal | 0.01 | 0.02 | 0.12 |
| greedy-classic | — | — | 0.01 |

Saturation on a 4–8-matrix chain is already *cheaper* than the greedy
episode (it stops at the fixed point; the episode evaluates all 51
rules per step, up to 6 steps).  A policy could only pay off where
saturation **blows up** — the exact-saturation fragmentation at large
chain depth (`search_efficiency`).  That is the honest scope of the
learned-policy headroom, and it is a search-cost question.

## The seam for a learned `Policy`

The suite scores the non-learned players only and does not depend on
the RL reward (being reworked elsewhere).  A learned player drops into
`_chain_players(rules)` as one more entry:

```python
("learned", lambda dims: _learned_episode(dims, rules, policy))
```

It drives the same `catopt_core.search_env.SearchEnv` episode, so it
returns a `flops_cost` in the same units and the ratio/rank statistics
need no other change.

## Honest limitations

* **Ranks are shared on ties.**  The chain ladder's top three share a
  mean rank of 2.31 because they tie on most instances; read the ratio
  and `%opt` columns, not the rank.
* **The net family has no priced player.**  `greedy-classic` vs the
  subset DP is the only ladder there; there is no catopt e-graph
  spelling of a general tensor network today.
* **One cost model.**  All chain numbers use `flops_cost` (the
  `SearchEnv` default).  A different model could shift the exact
  ratios, though the bracketings differ in FLOPs, not kernel count.
* **Saturation is exact *here*.**  Only because `assoc_matmul` closes
  the whole bracketing space; under a budget-truncated saturation
  (`rule_budgets`, `stop="improving"`) the "optimum" is only the
  reachable one.
* **Greedy is *a* baseline.**  The one-step oracle is the plan's
  baseline; a stronger classical baseline (e.g. a beam over contraction
  moves) would shrink the measured headroom.

## Reproduce

```sh
python -m bench run contraction_ladder          # full ladder + findings
python -m bench run contraction_ladder --quick   # ~7 s
python -m bench.suites.evaluation.contraction_ladder --no-artifacts
```

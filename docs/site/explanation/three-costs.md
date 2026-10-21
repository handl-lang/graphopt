# The three costs — how the engine knows a form is *better*

<!-- provenance: graphopt:docs/cost-model.md@eea02f6 — ported; upstream is the source of truth -->

"How do you know the cost — and how do you know when it found a better
algorithm?" The answer is three different costs at three different
trust levels. Each level exists because the cheaper one can lie.

## 1. `dag_cost` — the *guess*

The e-graph holds thousands of equivalent forms; extraction picks one.
`dag_cost` walks the program graph and sums static weights — FLOPs per
op, parameter bytes, node counts. No tensors ever execute.

**Why it can lie:** it has no notion of memory traffic, kernel fusion,
launch overhead, or cache behavior. A fused `sdpa` call looks
expensive on paper but is fast under Inductor. Used for *ranking*,
never for *claims*.

## 2. Measured pay — the *experiment*

A minted law cannot ship on `dag_cost` alone. The admission
[gauntlet](./the-gauntlet) runs the candidate on **real tensors,
seeded fixtures** and measures wall time / peak memory. This is where
"better" becomes empirical — and where 212 bad accepts died: they
*looked* paying until measured at real sites.

**Why it can still lie:** fixtures are small, seeded, one GPU box —
hence "regime-bounded" in the honest scope.

## 3. Numerical verify — the *proof of equality*

After lowering, the produced module executes and its outputs are
compared against the source's, fp64-exact where stated. Verification
says **the output is identical** — nothing about speed.

## The composition

| level | answers | can be wrong because |
|---|---|---|
| `dag_cost` | "probably cheaper" | the model misses hardware reality |
| measured pay | "actually cheaper on these fixtures" | fixtures ≠ all regimes |
| verify | "computes the same thing" | says nothing about cost |

"Found a better algorithm" = all three: **verified-equal output +
measured cheaper + ranked by the model that guided the search.**

The referee separates the same way: [the game's](./the-game) value
spec (`ScoringSpec`) *declares* which axes matter; the gauntlet's
measurement supplies the numbers — the two sides meet only through a
plain `Mapping[str, float]`.

**One-liner:** *the model proposes, the fixtures measure, the verifier
certifies — a law ships only if all three agree.*

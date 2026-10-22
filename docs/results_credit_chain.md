# Credit chains — does RL learn enabling mints?

**Date:** 2026-10-21 · **Script:** `bench/drivers/policy/credit_chain.py` ·
**Seed:** 0 (main), 0–2 sanity · **Hardware:** the 11 GB CPU box (no
CUDA) · **Runner:**

```bash
.venv/bin/python bench/drivers/policy/credit_chain.py \
    --json /tmp/chain.json                       # main run, ~7 s
.venv/bin/python bench/drivers/policy/credit_chain.py \
    --k 16 --budget 8 --train-cases 200 --net-episodes 600   # stress
```

The Q3 arm: a curriculum where the bank is *deliberately
incomplete*.  Reaching the win needs a two-step enabling chain —
`mint` the premise the goal rewrite requires (immediate reward
−0.05: it buys a law that enables nothing by itself), then `apply`
the now-banked premise (the terminal +1.0).  The question is purely
temporal credit assignment: the mint only ever earns through the
*next* move's payout, and the learners are never told the chain.

## Board design

Synthetic by design (option (a) — no e-graph, no oracle, no torch
in the board).  `ChainArena` implements the `engine.Board`
contract; a case names which of the `k` mintable candidates is the
premise the goal needs.  Moves: `mint(c)` (−0.05, banks *c*),
`apply(law)` (the needed premise → +1.0 terminal win; `base` →
+0.2, the chain-free fallback always in the bank; any other minted
law → 0.0, a dead apply), `stop`.  Budget 6 — mint-everything
(`flood`) physically cannot buy the chain.  The state honestly
publishes `needed` (the missing-premise diagnostic — the same
thing `missing_hooks` reports); what it withholds is the strategy.

The measurement is the **premise delta**: the same held-out boards
played as drawn (`nopre`) and with the needed premise pre-minted
(`pre` — the win degenerates to a one-step, greedy-visible apply).
An arm that cannot assign credit to the mint scores high `pre` and
low `nopre` — a full-win delta.  Learned arms train on unpreminted
boards only.

## Arms table — main run (k = 8, budget 6; linear/advantage 400
episodes, neural 1200; 96 held-out boards, paired)

| player | reward nopre | win nopre | reward pre | win pre | Δreward | Δwin |
|---|---|---|---|---|---|---|
| random | 0.049 | 0.10 | 0.383 | 0.43 | +0.334 | +0.32 |
| flood | −0.050 | 0.00 | −0.050 | 0.00 | 0.000 | 0.00 |
| greedy (myopic) | 0.200 | 0.00 | 1.000 | 1.00 | +0.800 | +1.00 |
| scripted | 1.150 | 1.00 | 1.200 | 1.00 | +0.050 | 0.00 |
| linear-cold | 0.048 | 0.10 | 0.348 | 0.40 | +0.300 | +0.29 |
| linear (REINFORCE) | 0.414 | 0.40 | 0.958 | 0.92 | +0.544 | +0.52 |
| linear-greedy | 1.150 | 1.00 | 1.000 | 1.00 | −0.150 | 0.00 |
| advantage (V-baseline) | 0.754 | 0.71 | 1.033 | 0.98 | +0.279 | +0.27 |
| neural (PolicyNet) | 1.150 | 1.00 | 1.200 | 1.00 | +0.051 | 0.00 |
| neural-greedy | 1.150 | 1.00 | 1.200 | 1.00 | +0.050 | 0.00 |

Controls behaved as designed: `scripted` runs base → mint → apply
for the 1.15 optimum (+0.05 pre-minted = the mint cost saved);
`greedy` is the delta's type specimen — myopic argmax never mints
(its Δwin is exactly 1.00); `flood` proves the budget binds
(−0.05 both ways — minting 5 candidates never reaches an apply).

## Finding 1 — RL *does* learn the enabling chain at this density

`neural` converges to the scripted ceiling on both conditions
(train means/100: 0.31 → 1.13 by episode 200, then flat at 1.15);
the trained `linear` table wins **100%** of unpreminted boards
under greedy decode.  The credit reaches: reward-to-go REINFORCE
and discounted-returns Adam both assign the terminal +1.0 back to
the negative-reward mint.

The honest nuance is *decoding*: sampled `linear` wins only 40%
(the softmax keeps spreading over ~11 legal actions — the learned
preference is real but not sharp), while the per-step
`G_t − V(s_t)` baseline of `advantage` calibrates that to 71%.
`linear-greedy` pre-minted dips to 1.00 (applies the armed premise
before banking `base` — wins, skips the +0.2).

## Finding 2 — the binding constraint is chain *density*, not the
credit mechanism

Stress run (k = 16 candidates, budget 8; linear/advantage cut to
200 episodes, neural to 600):

| player | reward nopre | win nopre | reward pre | win pre | Δreward |
|---|---|---|---|---|---|
| greedy | 0.200 | 0.00 | 1.000 | 1.00 | +0.800 |
| scripted | 1.150 | 1.00 | 1.200 | 1.00 | +0.050 |
| linear | −0.059 | 0.06 | 0.409 | 0.48 | +0.469 |
| linear-greedy | 0.200 | 0.00 | 1.000 | 1.00 | +0.800 |
| advantage | 0.017 | 0.15 | 0.446 | 0.52 | +0.429 |
| neural | 1.150 | 1.00 | 1.167 | 1.00 | +0.017 |

At k = 16 a cold episode completes the chain by luck with
probability ~1/18 × 1/18 per step pair — too sparse for 200
REINFORCE episodes to lift `a:mint:needed` out of the noise.
`linear-greedy` collapses to the *myopic signature itself* (0.20 /
1.00, Δ = 0.80 — indistinguishable from `greedy`), and `advantage`'s
state-value baseline does not rescue it: V(s) can only lower the
variance of credit for chains the sampler already finds.  Neural —
same reward stream, Adam per episode — still squeezes its rare
chain hits to convergence (1.15/1.00 win both ways, seed 1
replicates).

## Verdict

**Yes — with a density caveat.**  On the enabling-chain board the
shipped learners *do* learn "mint intermediate → use it later":
the trained linear policy argmaxes the chain on every held-out
board and the PolicyNet port converges outright.  The failure mode
the arm was built to expose lives upstream of credit assignment —
in *discovery*: when spontaneous chains thin out (more decoys,
fewer episodes), linear REINFORCE never gathers enough (+chain)
episodes to learn from, and degrades to exactly the myopic arm's
behavior.  The premise pre-mint delta is the diagnostic that
separates the two questions: Δwin ≈ 1.0 marks "could not buy the
premise" (greedy, flood, stress-run linear), Δ ≈ the mint cost
marks "chain learned" (scripted, neural).

Files: `bench/drivers/policy/credit_chain.py` (board + arms +
protocol), `tests/test_credit_chain.py` (board contract, the
mint-gates-win property, arm-eval smoke).

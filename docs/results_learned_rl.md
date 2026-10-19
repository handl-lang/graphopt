# Dedicated RL policy arms

**Date:** 2026-10-19 · **Script:** `lab/policy_rl_arms.py` · **Seed:** 0
throughout · **Hardware:** the 11 GB CPU box (no CUDA) · **Runners:**

```bash
timeout 1800 .venv/bin/python lab/policy_rl_arms.py --only open \
    --json /tmp/rl_open.json          # ~6 min
timeout 1800 .venv/bin/python lab/policy_rl_arms.py --only contraction \
    --json /tmp/rl_contraction.json   # ~3.5 min
timeout 1800 .venv/bin/python lab/policy_rl_arms.py --only az \
    --json /tmp/rl_az.json            # ~5 min
```

The question: the controlled baseline/ablation runs
(`docs/results_learned_baseline.md`, `docs/results_learned_ablations.md`)
measured *linear* policies trained by the players' built-in REINFORCE
plus a supervised `RuleValueNet`. The repo also ships two dedicated
neural-RL training loops — `mlopt_torch.rl.train_reinforce`
(`PolicyNet`, the SearchEnv REINFORCE net) and the contraction-domain
`train_rl` curriculum / `contraction_az.train_search` (expert-iteration
PUCT). This run puts all of them under the same held-out protocol.

## Methodology

* **Open domain — identical protocol to the baseline.** Held-out odd
  pages `og.gen_cases(0, 36, page=16, start=1, step=2)`, budget 17;
  even pages train; every pool member is measured once under a
  per-candidate seed and `lp.evaluate` replays the memoized verdict
  for episodes *and* training — identical oracle draws everywhere.
  Pool: 576 held-out candidates, 7 useful (random AP 0.0122).
* **The port.** `train_rl_policy.py` only drives `SearchEnv`; the
  script ports the `PolicyNet` loop to `OpenArena` episodes verbatim —
  MLP `in→64→64→1`, Categorical sampling, γ = 0.95 returns-to-go, Adam
  3e-3, one update per episode — and varies two things the port is
  free to choose:
  * **input encoding** — `policynet`: the `PolicyNet` contract
    verbatim, `state_vector(compute_features(cand.lhs), progress) ⊕
    rule_vector(cand.as_rule())` per `evaluate` action (the same
    information `RuleValueNet` gets, plus progress), `stop` a zero
    row; `full`: the linear policy's own feature row
    (`og.featurize(state, action, hist)` over `OPEN_GAME_FEATURES`) —
    identical information to the trained prior.
  * **advantage baseline** — `std`: the shipped per-episode
    standardization (`rl._advantage`); `ema`: the `LinearPolicy`-level
    baseline, `G_t − ema` with `ema` a 0.2-EMA of episode totals.
* **Budgets.** 180 episodes = the linear recipe's 5 epochs over 36
  even pages; 1440 = 8× (the "does bigger RL help" arm).
* **Contraction.** `contraction_policy.train_rl` re-run at **full
  scale** — 1800 iterations, batch 24, `ns=(8,10,12)` — on CPU
  (141 s; no reduction needed despite the CUDA-targeted default).
  Evaluated with `_ladder` on `cs.random_network` hypergraphs at
  n = 20/30/40, 3 seeds each, alongside the two bundled artifacts
  (curriculum-RL and distilled).
* **AZ.** `contraction_az.train_search` at shipped defaults (200
  expert iterations, batch 24, sims 24 — 110 s on CPU) plus a
  bond-family `train_rl` (1800 iters) as `rl-bond`, mirroring
  `contraction_az.main`; the `contraction_einsum` equal-wall-clock
  board at n = 20/30, budgets 200/1000 ms, 3 instances, `puct` played
  by `puct_best_order` (eval sims 24). Requires `opt_einsum`
  (installed into `.venv` via `pip install opt_einsum`, 3.4.0). The
  two cost models coincide on these boards — pairwise contraction
  cost is the product of contracted dims in both — except a residual
  discrepancy for `our-greedy` (see notes).
* **Artifact caveat found during the run.** `load_contraction_policy()`
  is **broken on `main`**: the catopt→mlopt rename changed `_FORMAT`
  to `mlopt-contraction-policy/1` but the bundled `.pt` payloads still
  carry `catopt-contraction-policy/1`, so the tag check rejects both
  artifacts — including the shipped default. This script loads the
  state dicts directly (feature contract `scale-free-v1` still
  validated); the shipped loader needs a compat shim or regenerated
  artifacts.

## Open domain — dedicated PolicyNet REINFORCE vs the anchors

| arm | reward/ep | useful/ep | evals/ep | useful/eval | found | AP |
|---|---|---|---|---|---|---|
| random | 0.1339 | 0.083 | 8.67 | 0.0096 | 3 | — |
| heuristic drain | 0.3595 | 0.167 | 16.00 | 0.0104 | 6 | 0.0450 |
| cold-linear | 0.1699 | 0.111 | 9.31 | 0.0119 | 4 | 0.0123 |
| trained-linear (baked) | 0.3271 | 0.167 | 13.75 | **0.0121** | 6 | **0.1332** |
| trained-greedy | 0.3576 | 0.167 | 16.00 | 0.0104 | 6 | — |
| cold+rl (linear, 5 ep) | 0.3043 | 0.167 | 14.03 | 0.0119 | 6 | 0.1184 |
| cold+rl-greedy | 0.3553 | 0.167 | 16.00 | 0.0104 | 6 | 0.1184 |
| rl-net (policynet, std) | 0.0000 | 0.000 | 1.00 | 0.0000 | 0 | 0.0081 |
| rl-net-long (std) | 0.0000 | 0.000 | 1.00 | 0.0000 | 0 | 0.0081 |
| rl-net-ema | 0.0000 | 0.000 | 1.00 | 0.0000 | 0 | 0.0081 |
| rl-net-ema-long | 0.0000 | 0.000 | 1.00 | 0.0000 | 0 | 0.0081 |
| rl-net-full (std) | 0.0000 | 0.000 | 1.00 | 0.0000 | 0 | 0.0203 |
| rl-net-full-long (std) | 0.0000 | 0.000 | 1.00 | 0.0000 | 0 | 0.0125 |
| **rl-net-full-ema** | 0.3505 | 0.167 | 15.72 | 0.0106 | 6 | **0.1092** |
| rl-net-full-ema-greedy | 0.3611 | 0.167 | 16.00 | 0.0104 | 6 | |
| **rl-net-full-ema-long** | 0.3655 | 0.167 | 15.86 | 0.0105 | 6 | 0.0899 |
| rl-net-full-ema-long-greedy | **0.3784** | 0.167 | 15.72 | 0.0106 | 6 | |

**The shipped recipe collapses to the absorbing `stop`.** Every `std`
arm — and both `ema` arms on the `policynet` input — converges inside
one epoch to evaluating nothing (train curve flat at 0.0; a
single-action episode has a zero-spread return, so the standardized
advantage is identically 0 and the collapse is unrecoverable). The
mechanism is structural, not a tuning miss: per-episode
standardization discards the cross-episode reward *level*, and in the
~85% of pages with no `useful` find every `evaluate` has
`G_t ≈ −0.002·k` while `stop` has `G = 0` — above the episode mean,
pushed up every losing episode. The EMA baseline preserves the level
(stop's 0 loses to a positive expected total), but on the `policynet`
input the net still cannot keep early episodes net-positive — the
`compute_features(lhs) ⊕ rule_vector` encoding lacks the
discriminative `a:*` shape features (heuristic score, depth, metas,
form one-hots), so the baseline EMA still settles at ~0 and stop still
wins. Only `full` + `ema` escapes: it needs *both* the
level-preserving baseline *and* the linear policy's feature schema.

**The working arm is good but not better than the linear prior.**
`rl-net-full-ema` reaches AP 0.109 — above heuristic (0.045), within
noise of cold+rl-linear (0.118), below the shipped prior (0.133). Its
greedy-decode reward 0.3784 at 1440 episodes is the best of *any* arm
— but that edge is the drain confound the baseline doc isolated
(15.7 evals/ep ≈ full page); on the priced metric, useful/eval, it
sits at 0.0104-0.0106 vs trained-linear's 0.0121. **Bigger RL does not
help the ranking**: 8× episodes buys reward (0.3505→0.3655) while AP
*degrades* (0.109→0.090) — longer training drifts toward drain
behavior, not a better ordering. Training cost: ~5-7 s for 180
episodes, ~37-53 s for 1440, all CPU.

## Contraction — curriculum RL re-run at full scale on CPU

`train_rl` (REINFORCE, greedy-completion critic, entropy bonus,
batch 24, 1800 iterations) re-ran verbatim on CPU in **141 s** — the
GPU default is a convenience, not a requirement. Ratio to
best-found per board (mean of 3 seeds, `cs.random_network`
hypergraphs; `*-restart` = best of 64 sampled rollouts):

| player | n=20 | n=30 | n=40 |
|---|---|---|---|
| greedy | 2.653 | 1.153 | 2.339 |
| restart | 1.231 | 1.037 | 1.006 |
| one-step | 1.426 | — | — |
| search | 1.280 | 1.006 | — |
| **rl-fresh** | 1.363 | 1.015 | 1.001 |
| **rl-fresh-restart** | 1.033 | 1.005 | 1.001 |
| rl-bundled (curriculum artifact) | 2.207 | 1.003 | 1.000 |
| rl-bundled-restart | **1.008** | **1.001** | **1.000** |
| distilled (shipped) | 3.208 | 1.047 | 1.077 |
| distilled-restart | 1.384 | 1.018 | 1.001 |

Pairwise (our cost model): `rl-fresh/greedy` 0.62 / 0.89 / 0.73 —
beats deterministic greedy at every scale; `rl-fresh/restart` 1.10 /
0.98 / 1.00; `rl-fresh/search` 1.08 / 1.02.

Caveats: the bundled artifacts were trained on `random_bond_network`
(meta), so they are **off-family** on these hypergraph boards — which
is why the bundled/distilled *single-shot* rows look bad at n=20 while
their `-restart` rows are fine. The pinned "RL curriculum 0.96/0.94 at
n=20/30" numbers were measured on bond networks against
`opt_einsum`'s players — the AZ section below is that protocol.
On the hypergraph family the fresh CPU re-run reproduces the shape of
the original result: the learned policy beats greedy decisively,
reaches restart/search parity by n=30, and sits at the best-found
ceiling at n=40.

## AZ — PUCT + learned value head, equal wall-clock, CPU

`train_search` (expert iteration: PUCT visit distributions teach the
policy head, episode cost teaches the value head) ran 200 iterations
in **110 s** on CPU — no scale reduction needed. The board is the
`contraction_einsum` equal-wall-clock protocol on `random_bond_network`
instances: every player gets the same per-instance budget, orders
scored by `opt_einsum`'s `contract_path` (the `our` column; the `oe`
model is numerically identical on pairwise orders here).

Ratio to best-found (mean of 3 instances):

| player | 200ms n=20 | 200ms n=30 | 1000ms n=20 | 1000ms n=30 |
|---|---|---|---|---|
| our-greedy | 99.4 | 282.7 | 101.1 | 302.6 |
| our-restart | 6.22 | 78.8 | 3.86 | 31.5 |
| rl-bond (fresh, bond-trained) | 1.012 | 1.044 | 1.030 | 1.068 |
| distilled (shipped) | 1.075 | 1.043 | 1.015 | 1.038 |
| **puct (learned value)** | 1.388 | 2.548 | 1.410 | 2.685 |
| oe-greedy | 1.239 | 1.330 | 1.259 | 1.383 |
| oe-rand-greedy | 1.005 | 1.031 | 1.002 | 1.042 |
| oe-optimal | DNF | DNF | DNF | DNF |

Pairwise (oe cost): `rl-bond/oe-rand-greedy` 1.007 / 1.015 / 1.028 /
1.031; `distilled/oe-rand-greedy` 1.069 / 1.013 / 1.013 / 0.998;
`puct/oe-rand-greedy` 1.381 / 2.497 / 1.408 / 2.618; `puct/rl-bond`
1.372 / 2.395 / 1.370 / 2.437; `puct/oe-greedy` 1.125 / 2.176 (flat
across budgets at n=30).

**The AZ arm is a negative at this scale.** PUCT + learned value does
5.5k–7.6k simulation-work units at 1000 ms and still lands 1.4×–2.7×
off best-found — worse than every learned and oe player, and the gap
*grows* with n. Meanwhile the plain sampled-rollout RL arm
(`rl-bond`, trained on the same bond family) and the shipped
distilled player both sit at parity with `oe-rand-greedy` (~1.0–1.07),
consistent with the 0.89–1.01× pin at n=40. Two honest caveats: the
our-greedy rows show naive hypergraph greedy is ~100-300× off on
einsum-valid bond networks (it has no notion of oe's staged
memory-removed ordering — the context the whole oe comparison exists
for), and `oe-optimal` never finishes inside 1 s at n ≥ 20 (exact DP;
DNF = its `inf` cost).

## Verdict

**Dedicated RL does not beat the linear prior on the open domain —
but the failure is the recipe and the input, not the model class.**
The shipped `PolicyNet`/REINFORCE recipe collapses to the absorbing
`stop` inside one epoch: per-episode standardized advantage makes
`stop`'s zero return beat the episode mean on every find-less page,
and a single-action episode has zero advantage forever after — a
structural trap the linear player's cross-episode EMA baseline avoids
by construction. Given (a) the linear policy's *own feature rows* and
(b) a level-preserving baseline, the identical MLP trains to AP
0.109 — heuristic-beating, prior-matching-within-noise — and its
greedy reward (0.378) is the best measured. But nothing crosses the
0.133 AP line: the linear prior's lead holds for nets too, and 8×
training buys drain-reward while *costing* ranking. The feature
ceiling the ablation doc named binds the nets as well — but only
*after* the recipe and the input encoding stop self-sabotaging;
`RuleValueNet`'s BCE on the same structural input reached 0.041
because supervised BCE cannot fall into the stop trap.

Elsewhere: the contraction RL curriculum re-runs at **full scale** on
CPU in ~2.5 min and reproduces its qualitative result (beats greedy
everywhere, restart parity by n=30, best-found ceiling at n=40);
fresh bond-family RL and the shipped distilled artifact both hold
~parity with `opt_einsum`'s randomized greedy, while the PUCT +
learned-value arm loses to all of them at equal wall-clock — more
search in the loop did not beat more rollouts on this box.

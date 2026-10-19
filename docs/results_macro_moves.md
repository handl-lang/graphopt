# Macro-action minting — new *moves* the way laws are minted

**Date:** 2026-10-19 · **Scripts:** `lab/mint/macro_moves.py` (the
`"macro"` handler + offer seam), `lab/mint/macro_miner.py` (miner +
eval) · **Seed:** 0 throughout · **Runner:**

```bash
.venv/bin/python lab/mint/macro_miner.py --json /tmp/macro.json
```

The same move the discovery loop makes on laws, run on the action
space: collect `Trajectory` reports from played episodes, count
canonical action n-grams, rank by `count x payoff` (the mean
terminal-step reward of the episodes the gram rode in — "did the
subsequence precede a *paying* certified extract?"), and mint the
top grams as `Macro` templates. A minted macro is replayed as
`Action("macro", {"seq": [...]})` through the refereed `macro`
handler — each sub-action dispatches through `ACTIONS`, declares are
accepted two-phase, a declined sub-move declines the whole macro, a
terminal one propagates its certificate. Soundness is the antiunify
argument: a macro mints nothing a player could not already spell —
it compresses the line into **one** step (one `step` charge, one
budget tick).

## Mining — 206 trajectories over `gen_cases(0, 24)`, budget 24

Arms played per case: scripted, greedy, 2 seeded randoms, the frozen
shipped prior (sampled + greedy), `claim→extract` per claimable
handler tag, and one declare-line episode per mined candidate
(`declare→saturate→handle*→extract`). N-grams are counted over the
*applied* action stream, keyed by `(op, identity-params)` — declare
by its spelled spec's coverage skeleton, handle/claim by handler
tag, fire by rule — so the silu and swiglu lines mint as distinct
macros and fresh fold names merge.

| rank | seq | freq | cert% | term-rw | score |
|---|---|---|---|---|---|
| 1 | `claim(silu)→extract` | 15 | 1.00 | 0.289 | 4.333 |
| 2 | `declare(silu-spec)→saturate` | 13 | 1.00 | 0.296 | 3.850 |
| 3 | `declare(silu-spec)→saturate→handle(silu)` | 13 | 1.00 | 0.296 | 3.850 |
| 4 | **`declare(silu-spec)→saturate→handle(silu)→extract`** | 13 | 1.00 | 0.296 | 3.850 |
| 5 | `handle(silu)→extract` | 13 | 1.00 | 0.296 | 3.850 |
| 6 | `saturate→handle(silu)` | 13 | 1.00 | 0.296 | 3.850 |
| 7 | `saturate→handle(silu)→extract` | 13 | 1.00 | 0.296 | 3.850 |
| 8 | `saturate→handle(swiglu)` | 7 | 0.86 | 0.529 | 3.700 |

Just below the top-8 offer cap (score 3.70): `claim(swiglu)→extract`
(freq 6, term-rw 0.617) and the swiglu-spec `declare→saturate→
handle→extract` line. The base playbook `saturate→extract` itself
mines at freq 48 but **score 0** — its mean terminal reward is
−0.05, the correct ranking signal: the gram precedes certified
extracts that pay nothing.

**Rediscovery: YES** — the smoke test surfaces the known winning
line in *both* forms: the spelled `declare→saturate→handle→extract`
(rank 4) and its claim-fused form `claim→extract` (rank 1).

## Held-out eval — `mp.gen_cases(0 + 7919, 16)`, budget 12

Macro offers enter through the players' `legal=` seam
(`legal_with_macros`: ranked, `viable`-filtered — a `claim` needs
its tag in `state.claim_tags`, a `declare` needs its spec's skeleton
among the occurring sites — deduplicated, capped at 8). The trained
arms use the **frozen** shipped prior — weights untouched; a macro
featurizes to the generic broadcast row (no `op:macro` weight).

| arm | reward/ep | cost | steps | cert | decl |
|---|---|---|---|---|---|
| scripted | −0.100 | 2.94 | 2.00 | 16 | 0 |
| **scripted+macros** | **+0.088** | **2.50** | 1.56 | 16 | 7 |
| greedy | −0.600 | — | 12.00 | 0 | 0 |
| greedy+macros | −0.600 | — | 12.00 | 0 | 0 |
| trained-linear | −0.056 | 2.94 | 1.12 | 16 | 0 |
| trained+macros | −0.056 | 2.94 | 1.12 | 16 | 0 |

**scripted+macros vs scripted: 7 wins / 0 losses / 9 ties, all 7
cost wins.** Every case where the base playbook stalled at baseline
(a silu/swiglu site under the stripped ruleset) was won by the
minted `claim(silu)→extract` — one refereed step replacing a line
the scripted arm never plays:

| case | scripted → +macros | cost → | declared |
|---|---|---|---|
| gen1:silu_nested | −0.100 → +0.200 | 4.0 → 3.0 | `claim_silu_0` |
| gen2:swiglu | −0.100 → +0.283 | 3.0 → 2.0 | `claim_silu_0` |
| gen5:silu | −0.100 → +0.450 | 2.0 → 1.0 | `claim_silu_0` |
| gen6:silu | −0.100 → +0.450 | 2.0 → 1.0 | `claim_silu_0` |
| gen7:swiglu | −0.100 → +0.283 | 3.0 → 2.0 | `claim_silu_0` |
| gen12:silu_nested | −0.100 → +0.200 | 4.0 → 3.0 | `claim_silu_0` |
| gen14:silu | −0.100 → +0.450 | 2.0 → 1.0 | `claim_silu_0` |

The macro player is also the *first* arm in this protocol to beat
the trained prior on mean reward (+0.088 vs −0.056).

## Honest notes

* **`trained+macros` ≡ `trained-linear`.** 0 macro picks in 18
  sampled steps: the frozen policy scores a macro on the generic
  broadcast row (logit ≈ 0) while `op:extract` sits at +3.15 and
  ends episodes in ~1 step. Macro offers are invisible to a policy
  that was not trained on them — the inventory pays a player that
  already knows to look (scripted+macros), not a frozen one.
* **`greedy+macros` ≡ `greedy`.** 0/192 steps — greedy grinds the
  `-step`-priced fire list first; it never reaches the offers.
* **Rank order leaves value on the table.** On swiglu cases the
  rank-1 `claim(silu)` macro fires (the silu subterm occurs) and
  extracts at cost 2.0 where `claim(swiglu)` would reach 1.0 —
  the scripted arm plays the first *viable* macro, and viability is
  legality, not payoff. A case-aware picker is future work.
* **Off-case macros are declined, not wrong.** `viable` keeps the
  minted lines off boards they cannot bind; where it lets one
  through that mid-sequence declines (e.g. a `handle` whose object
  never got declared), the macro scores 0 — the referee's honest
  refusal, no rollback needed because a declined sub-move leaves the
  board exactly as its refereed effects dictated.

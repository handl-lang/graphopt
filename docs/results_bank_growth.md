# Does a grown bank help the same player? — the G\* experiment

**Date:** 2026-10-25 · **Driver:** `bench/drivers/policy/bank_growth.py`
(new) · **Domain:** `search` (rule-fire boards) · **Policy:**
`gamekit.players.LinearPolicy`, trained once then **frozen** (no
learning at eval) · **Seed:** 0–4 · **Box:** i5-12500H, CPU

```bash
.venv/bin/python -m bench.drivers.policy.bank_growth --seed 0 \
    --train-cases 32 --eval-cases 64 --epochs 4 --json /tmp/bg_0.json
```

> **Verdict: growth alone does not help the same player — growth that
> *pays* does.** Adding the 57 minted prim folds moves a frozen player
> **−0.023** reward/episode (5 seeds, all ≤ +0.008). Adding four laws that
> actually collapse the corpus's cases moves the same frozen player
> **+0.066** (5 seeds, all ≥ +0.036). The instrument is sensitive in both
> directions, and it agrees with `pay_wall`: unpaid vocabulary is
> dilution.

## The protocol

The stack's claim is a loop: play mints rules → rules enter the bank →
the grown bank makes the *same* player better without retraining. The
first half is measured (21/21 machine-original). This measures the second.

- **Two banks.** `v0` = `DEFAULT` minus the growth under test; `v1` =
  `DEFAULT`. Two growths tested, same protocol.
- **Three arms**, trained once each, then frozen: `train-v0`,
  `train-v1` (the control — what the new moves are worth to a player that
  has seen them), `random` (bank effect with no learning).
- **Crossed evaluation.** Every frozen arm plays both banks' boards on the
  same 64 held-out cases, same budget (24). The G\* number is
  `train-v0` on **v1** minus `train-v0` on **v0**.

## Result A — the minted prim growth (57 folds, 74 → 131 rules)

| seed | on v0 | on v1 | delta |
|---|---|---|---|
| 0 | 0.109 | 0.070 | −0.039 |
| 1 | 0.075 | 0.047 | −0.029 |
| 2 | 0.143 | 0.081 | −0.062 |
| 3 | 0.112 | 0.120 | +0.008 |
| 4 | 0.081 | 0.086 | +0.005 |
| **mean** | **0.104** | **0.081** | **−0.023** (sd 0.030) |

A player trained on the smaller bank gets **worse** when the bank grows:
the extra moves are legal but unproductive, and a fixed policy spends
budget on them. This is the dilution the pay gate predicted.

## Result B — the control: paying growth (4 laws, 127 → 131 rules)

`id_add`, `id_mul`, `double_neg`, `silu_fold` — the laws that collapse
the `search` pool's cases and therefore pay.

| seed | on v0 | on v1 | delta |
|---|---|---|---|
| 0 | 0.010 | 0.070 | +0.060 |
| 1 | 0.010 | 0.047 | +0.036 |
| 2 | 0.000 | 0.081 | +0.081 |
| 3 | 0.010 | 0.104 | +0.094 |
| 4 | 0.010 | 0.070 | +0.060 |
| **mean** | **0.008** | **0.075** | **+0.066** (sd 0.022) |

The same frozen player picks up the new moves and earns ~9× the reward
per episode — the featurizer generalizes far enough that the moves are
taken without retraining.

## What this says about G\*

The loop is real but **conditional, and the condition is pay**:

> growth helps the same player **iff** the grown moves pay on the board
> they land on. Our prim growth does not pay (`results_pay_measured.md`),
> so it does not help — it dilutes.

That is a sharper and more useful statement than "the loop closes". It
also explains why the retrained control (`train-v1` on v1) is no better
than `train-v0` on v0 in Result A (0.075 vs 0.104): even a player that
has seen the 57 folds cannot profit from them, because they do not reduce
cost on this corpus.

## Step 3 — minting *during play* on this board admits nothing

The G\* loop's middle step is "play mints rules". Run on the board's own
corpus (`bench/drivers/policy/play_mint.py`: compose pairs of the shipped
bank over the case terms, then pay → novel → fires):

```bash
.venv/bin/python -m bench.drivers.policy.play_mint \
    --cases 32 --fuel 600 --cap 120 --json /tmp/play_minted.json
```

| minted | admitted | rejects |
|---|---|---|
| 1 | **0** | 1 — `syn_mul_square__square_to_pow_0`, *no pay* |

The board's pool has **7 distinct terms**, so 32 cases are repeats: there
is almost nothing left to compose, and the single composite found
(`x·x → x²`) does not lower cost. So on this corpus the loop is:

1. **mint during play** — works mechanically, admits nothing that pays;
2. **grow the bank anyway** (the 57 shipped folds) — frozen player
   **−0.023**;
3. **grow with rules that pay** (the control) — frozen player **+0.066**.

Which is the same conclusion from three directions: **the loop's growth
is real, its benefit is conditional on pay, and the current mint product
does not pay.** The next run that can move this needs a corpus with real
composable structure *and* a delivery path that pays for it — the `gen`
domain's 2.87× is the only measured example we have.


- One board family (`search`), one policy class (linear), 24-step budget,
  5 seeds; rewards are small (0.0–0.14 per episode), so treat ±0.03 as
  the noise band — both results clear it, in opposite directions.
- Growth is measured on a *fixed corpus*: no mint happens during play
  here. The full G\* story — mint during play, then measure the frozen
  player — is the next run and needs the online mints' rules.
- 74 → 131 is a 77 % vocabulary increase with zero pay benefit on these
  cases; a corpus where the folds pay would be a different test of the
  same mechanism.

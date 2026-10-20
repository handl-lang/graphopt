# The stopping policy, round 8 — calibrate the predictor first

**Date:** 2026-10-20 · **Script:** `bench/policy/stop_calibrated.py`
(new — the `stop_valuelabel.py` harness with a calibration stage:
`bench/common/calibrate.py` `PlattCal` / `IsotonicCal`, fit on the
train pages' OOF predictions, judged on held-out) · **Seed:** 0 ·
**Runner:**

```bash
timeout 1800 .venv/bin/python bench/policy/stop_calibrated.py \
    --json /tmp/cal.json            # ~5.5 min (155 s memoization,
                                    #  5.7 s predictors, <0.1 s fits)
```

## Motivation

Round 7 (`docs/results_stop_valuelabel.md`) closed on a precise
diagnosis: the `open:vlabel` expected-yield columns carry real
signal, but the verdict-label nets are **miscalibrated in magnitude
held-out** — they under-predict (7/26 in-sample positives memorized,
held-out AP ≤0.10 on `useful`/`pays`), so the analytic break-even
gates `ev-margin`/`ev-pool` (κ = `|step|` = 0.002) quit at 3–9 evals
and lose 0.09–0.19 reward.  This round calibrates the predictor
*before* judging it: Platt scaling (logistic on the score's logit)
and isotonic regression (PAVA) on the raw scores → p̂, both fit on
**train pages' predictions only** — which are already out-of-fold
(each train page scored by a net that never saw it), so the
calibration sample is held-out-quality — and evaluated on the odd
pages.  Then the gates re-run on p̂ at κ = `|step|` plus a
0.5×–2× sweep, and round 7's earning arms (`head+vlabel`,
`ac+vlabel`) re-run under the calibrated injection.

## Port control — the harness is exact

The raw-κ=1× gate rows replicate round 7 **bit-identically**
(`ev-margin` 0.1263 @ 9.06 evals/33 stops; `ev-pool` 0.0937 @ 7.33;
`useful-pool` 0.0067 @ 3.22; `pays-margin` 0.0883 @ 7.25; `delta`
both 0.2821 @ 15.31/23); incumbent 0.2728 / AP 0.2101 and `ac`
0.2761 / 0.1098 likewise.  Predictor APs replicate (delta 0.9320,
useful 0.0167, pays 0.0643).

## Calibration quality — held-out odd pages (fit on train OOF)

| label | map | log-loss | Brier | ECE | slope | int | mean p̂ | pos rate | AP |
|---|---|---|---|---|---|---|---|---|---|
| `delta` | raw | 0.9277 | 0.1126 | 0.1123 | 0.19 | −0.22 | 0.453 | 0.443 | 0.932 |
| | platt (a=0.13, b=−0.11) | 0.3308 | 0.0965 | 0.0774 | 1.43 | −0.06 | 0.454 | | 0.931 |
| | isotonic (16 blocks) | 0.3293 | 0.0992 | 0.0498 | 1.26 | −0.08 | 0.449 | | 0.898 |
| `useful` | raw | 0.1759 | 0.0178 | 0.0190 | 0.02 | −4.21 | 0.010 | 0.0104 | 0.017 |
| | platt (a=0.12, b=−2.69) | 0.0590 | 0.0104 | 0.0027 | 0.20 | −3.67 | 0.013 | | 0.018 |
| | isotonic (96 blocks) | 0.0876 | 0.0105 | 0.0053 | −0.00 | −4.58 | 0.007 | | 0.017 |
| `pays` | raw | 0.8795 | 0.0828 | 0.0894 | 0.00 | −3.00 | 0.043 | 0.0469 | 0.064 |
| | platt (a=0.04, b=−2.51) | 0.1902 | 0.0449 | 0.0032 | 0.03 | −2.92 | 0.046 | | 0.063 |
| | isotonic (27 blocks) | 0.2076 | 0.0453 | 0.0312 | 0.21 | −2.10 | 0.026 | | 0.064 |

Both methods fix the *level* — held-out `mean p̂` lands on the base
rate and log-loss improves 3–5× — while AP is untouched (a monotone
map cannot change a ranking; the small isotonic AP dip on `delta`
is its block ties).  **Neither adds discrimination**: held-out
slope stays ≪1 on `useful`/`pays` because the OOF train scores
themselves carry almost no signal to rescale (isotonic `useful`
slope −0.004 — its held-out map is nearly the base-rate constant).
Calibration repaired the magnitude; it cannot mint signal.

## The gates — the round-7 verdict repriced

κ multipliers on `|step|`; all gate rows share the incumbent
ranker's AP 0.2101.

| gate | raw reward @ evals/stops | platt | isotonic |
|---|---|---|---|
| ev-margin @1× | 0.1263 @ 9.06/33 | 0.2850 @ 16.00/**0** | 0.2856 @ 15.89/**7** |
| ev-pool @1× | 0.0937 @ 7.33/34 | 0.2850 @ 16.00/**0** | 0.2856 @ 15.89/**7** |
| useful-margin @1× | 0.1279 @ 5.56/36 | 0.2850 @ 16.00/**0** | 0.2875 @ 15.39/**23** |
| useful-pool @1× | 0.0067 @ 3.22/36 | 0.2850 @ 16.00/**0** | **0.2899 @ 14.25/26** |
| pays-margin @1× | 0.0883 @ 7.25/33 | 0.2850 @ 16.00/**0** | 0.2858 @ 15.81/9 |
| pays-pool @1× | 0.0897 @ 6.56/34 | 0.2850 @ 16.00/**0** | 0.2823 @ 14.81/10 |
| delta-{margin,pool} @1× | 0.2821 @ 15.31/23 | 0.2850 @ 16.00/**0** | 0.2851 @ 16.00/2 |

The κ sweep under isotonic draws a real curve, not a step: e.g.
`useful-pool` 0.2881→0.2899→**0.2929**→0.2607 and `pays-pool`
0.2862→0.2823→0.2701→0.2191 at 0.5×/1×/1.5×/2× — reward peaks just
past break-even and falls as the tighter gate starts paying the
over-stop price; `useful-margin` rises monotonically
(0.2867→0.2883) and `delta` is flat (0.2851 — the calibrated
`delta` map sits above 2×κ everywhere).

## What calibration actually decided

* **The under-prediction diagnosis was correct, and Platt confirms
  it by collapsing to never-stop.**  Recalibrated to the true base
  rate, every `platt` gate — every mode — plays 16.00 evals, 0
  stops, and earns **0.2850** ≥ 0.273 (AP 0.2101, incumbent
  ranker).  This is not a bug: at the calibrated level the *next
  eval's* expected payoff is `p̂·tier ≈ 0.013–0.02`, which clears
  κ = 0.002 roughly 7–10× over.  **The asymmetric table prices even
  a 1% chance of +1.0 as worth a −0.002 eval** — the break-even rule
  at honest probabilities *rationally drains*.  Round 7's
  catastrophic early quits were the miscalibration artifact the
  diagnosis said they were; fix the level and the gates do what the
  head's reward-max τ did all along — drain — now for the measured,
  principled reason rather than the calibrated-threshold accident.
* **Isotonic's blocky tail buys real (modest) stopping.**  Where the
  OOF fit kept even coarse differentiation (96/27/16 blocks), the
  gate quits genuinely — `useful-pool`@1× **0.2899** with 26 stops
  at 14.25 evals, the best break-even gate ever posted, and the
  1.5× point **0.2929** — versus raw 0.0067.  But the stops are the
  familiar *tail* quits (14–16 evals/ep, not the oracle's 3.75):
  the isotonic map can certify "these pending candidates price at
  the bottom block" only near page end.  It cannot certify an empty
  pool early because there *is* no signal: `useful` held-out slope
  ≈ 0.
* **The trained arms re-run under calibrated features: same wall.**
  `ac+vlabel` posts the round's best rewards — `ev/platt` **0.3021**,
  `delta/isotonic` **0.3080** — every one on a collapsed ranking
  (AP 0.105–0.147); `vlabel`/rl flat 0.2594–0.2691 at AP ~0.06–0.10.
  The head gates improve where the calibrated columns carry
  structure: `head+vlabel:useful/platt` **0.2912** @ 22 stops /
  13.5 evals (best head arm yet, beats round 7's 0.2890), and its
  eval AP rises to **0.483** under isotonic (`useful`) vs the ~0.40
  raw ceiling — the calibrated *magnitudes* sharpen the supervised
  continue-label fit.  But head arms stay non-serializable (gate +
  τ), one still over-stops (`head+vlabel:ev/isotonic` 0.2245 @
  6.83 evals — τ calibrated on train doesn't transfer), and **every
  fold misses or never-stops**: best fold `head+vlabel:delta/platt`
  0.2727 — the recurring sub-0.273 knife-edge — and
  `head+vlabel:pays/isotonic-fold` "clears" at 0.2850 with **0
  stops**, the documented drain confound at a sixth feature family.

## Anchors

oracle-stop ceiling 0.3877 @ 3.75 evals · baked-prior 0.2758 /
AP 0.2101 · heuristic drain 0.3595 / AP 0.0450 · random 0.1339 /
AP 0.0123.

## Verdict (round 8): not earned — `OPEN_GAME_PLAYER_WEIGHTS` stays

Gate: held-out reward ≥ 0.273 **and** AP ≥ 0.210, honestly — a
never-stop clear or an AP collapse is an honest negative.  No
*serializable* arm clears both: the reward-clearing `ac+vlabel`
rows carry collapsed rankings (≤0.147), the folds all miss or
degenerate (the 0.2850 fold clears by never stopping).  The gated
arms earn honestly — `head+vlabel:useful/platt` 0.2912 @ 22 real
stops, and the *headless* break-even gate now earns under isotonic
(`useful-pool` 0.2899 @ 26 stops) where raw it starved (0.0067) —
but a gate remains a second scorer, not a weight table.  **No
bake.**

What the round adds to the arc: the last unpriced axis —
*magnitude* — is now priced, and it relocates the blocker rather
than removing it.  Calibration fixed the predictors' level
(log-loss 3–5× down, `mean p̂` = base rate, ECE ≤0.05) and the
gates stopped starving.  But at honest probabilities the
break-even rule *correctly* says drain — the cost asymmetry
(+1.0 find vs −0.002 eval) is so extreme that a calibrated 1–2%
chance clears the eval's price — so a bakeable stop policy needs
precisely what eight rounds keep re-deriving: per-candidate
*discrimination* that transfers held-out (the `useful`/`pays` nets'
held-out slope is ≈0 even after calibration), i.e. verdict-
conditioned or verifier-visible features — or a delivery outside
the shared softmax, which the fold arm still cannot serialize.

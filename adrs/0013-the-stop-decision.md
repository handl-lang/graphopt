# ADR 0013 — the stop decision: an eight-round characterization

Status: measured, open.  Series:
`results_stop_policy.md` (r1–2) → `results_stop_advantage.md` (r6,
self-posterior + actor-critic) → `results_stop_valuelabel.md` (r7,
expected-yield labels) → `results_stop_calibrated.md` (r8,
Platt/isotonic calibration) → `results_stop_isolated.md` (r9, the
delivery outside the softmax — `GatedPlayer` + the `forced_action`
engine hook).

## The problem

Should the open-game player keep evaluating pending candidates or
stop?  The oracle ceiling is measured: **+0.026 reward/ep and 4.3×
yield-per-eval** (3.75 evals vs the incumbent's ~15) — real, and
unreachable so far.

## What eight rounds eliminated

| blocker | round | verdict |
|---|---|---|
| features | r1–2 | shared-softmax stop logit drowns; supervised head degenerates to never-stop (Bayes-rational under asymmetric costs) |
| credit | r6 | per-step advantage (actor-critic `AdvantagePlayer`) rescues AP vs collapse — real mechanism gain |
| observability | r6 | policy's own posterior (`open:self`) ≈0.33 solo AP vs oracle labels — signal exists at zero predictor cost |
| value labels | r7 | `open:vlabel` expected-yield columns real, but predictors miscalibrate in magnitude held-out |
| calibration | r8 | Platt/isotonic fixes magnitude (log-loss 3–5×, ECE ≤0.05) — **best headless gate: isotonic `useful-pool` 0.2929 @ 12.8 evals, 26 real stops** |
| delivery | r9 | the isolated channel (`forced_action` + `GatedPlayer`) clears both bars honestly — `iso-head:isotonic` 0.2892/AP 0.2101 @ 26 real stops, seam-free; the fold twin never-stops.  Shared-softmax hypothesis **confirmed for delivery** — residual is evidence (all clears are tail-trims) |

## The residual blocker (stated exactly)

At calibrated probabilities the break-even rule *rationally drains*
(cost asymmetry: missed useful −1.0 vs wasted eval −0.002).  No
serializable arm clears both bake bars (reward ≥0.273 ∧ AP ≥0.210).
What remains missing is **held-out discrimination** — a predictor that
ranks pending candidates above chance out-of-distribution — or a stop
delivery outside the shared softmax (the recurring corruption channel:
five feature families all collapse AP the same way).

## Transfer

The lesson generalizes beyond stop: any gated decision in the game
needs (a) features that see the decision-relevant quantity, (b) credit
assigned per-step, (c) calibrated magnitudes, (d) a delivery channel
isolated from ranking logits.  Rounds 1–8 are the measured map of
which of those was actually the wall at each stage.

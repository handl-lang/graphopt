# The stopping policy, round 7 — candidate value *labels* priced as yield

**Date:** 2026-10-19 · **Script:** `bench/drivers/policy/stop_valuelabel.py`
(new — the `stop_advantage.py` harness with the label→payoff
injection, the `open:vlabel` part, and two analytic break-even
gates) · **Part:** `mlopt_discovery.open_vlabel` (registered as
`open:vlabel`, unshipped) · **Seed:** 0 · **Runner:**

```bash
timeout 1800 .venv/bin/python bench/drivers/policy/stop_valuelabel.py \
    --value-label ev --json /tmp/vl_ev.json        # ~3.5 min each mode
                                                 # (~150-175 s memoization)
# and --value-label useful|pays|delta
```

## Motivation

Round 6 (`docs/results_stop_advantage.md`) closed on the closest
serializable arm yet — `ac+nv` 0.2764 / AP 0.2074, missing the 0.210
bar by 0.003 — and re-named the blocker: every stop-row signal is a
*ranking* view; none is a *will-it-pay* one, and the gate's absolute
threshold still doesn't serialize.  This round prices the remaining
label-side lever: inject the verdict-label predictors' output as
**expected payoff** — `p(label)·tier`, or the two-net expected tier
payoff `p_useful·1.0 + max(p_pays − p_useful, 0)·0.2` (`ev`) — and let
the stop row read the pool-priced *decision* quantities the
asymmetric table actually scores:

| feature | emitted on | what it encodes |
|---|---|---|
| `x:stop:vl_best` | `stop` | max pending `v(c)` — the best remaining eval's expected payoff |
| `x:stop:vl_pool` | `stop` | Σ pending `v(c)` — **expected remaining yield** |
| `x:stop:vl_net` | `stop` | Σ (`v(c)` − 0.002) — the drain-the-rest EV |
| `x:stop:vl_margin` | `stop` | max `v(c)` − 0.002 — the next eval's break-even margin |
| `x:stop:vl_frac` | `stop` | `vl_pool` / page's starting value — value-drain progress |
| `x:vl:exp`, `x:vl:net` | `evaluate` | this candidate's expected payoff, raw and net |

Unfed, `vl_net`/`vl_margin` carry only their `−|step|` terms (with no
positive predictions, draining does lose — the honest emission, not a
zero).  The payoff-unit injection also makes the *break-even* κ
uniform: `|step|` = 0.002 in injected units under every label.

## The arms

Same protocol as rounds 5–6 (even pages train / odd held out, page
16, budget 17, OOF-6 predictor calibration, page-aware AP).  New
mechanisms:

* `vlabel` / `ac+vlabel` / `ac+self+vlabel` — the part under flat
  REINFORCE and under round 6's per-step advantage (the `ac` twin —
  does priced yield + isolated credit finally carry the timing?);
* `head+vlabel` / `head+stop+vlabel` — the corrected supervised gate
  over the new columns, with the serializable `-fold` variants;
* **`ev-margin` / `ev-pool` — the calibrated-break-even variant**: no
  learned head, no reward-max τ.  `ev-margin` continues while
  `max_pending v(c) > 0.002` (the best eval still clears its cost);
  `ev-pool` while `Σ_pending (v − 0.002) > 0` (draining the rest
  still nets positive).  They price *signal quality alone* — the
  threshold is the asymmetric table's own break-even, so any failure
  is the predictor's, not the calibration's.

Four runs, one per injection.  The `delta` run is the port control:
its `nv` / `ac+nv` rows replicate round 6 **bit-identically**
(0.3008/0.1042 and 0.2764/0.2074) — the harness and the unit-1.0
injection are exact.

## The predictors (identical nets in all four runs)

Held-out AP of each net's sigmoid vs each label (576 even-page
samples; OOF train preds):

| net | train pos | pos_w | AP vs `delta` | AP vs `useful` | AP vs `pays` |
|---|---|---|---|---|---|
| `delta` | 271 | 1.1 | **0.9320** | 0.0282 | 0.0407 |
| `useful` | 7 | 81.3 | 0.4647 | 0.0167 | 0.0959 |
| `pays` | 26 | 21.1 | 0.4103 | 0.0168 | 0.0643 |

Round 5's label-gap result replicates: `delta>0` is learnable (0.93),
the truth∧novelty conjunction is not (0.017) — the `useful`/`pays`
nets memorize their few positives in-sample and transfer almost
nothing held-out.

## Results (`--value-label ev` — the expected-tier payoff)

| arm | mechanism | reward/ep | greedy | use/eval | evals/ep | stops | AP |
|---|---|---|---|---|---|---|---|
| incumbent | rl 5ep | 0.2728 | 0.3590 | 0.0115 | 14.44 | 19 | 0.2101 |
| ac | actor-critic | 0.2761 | 0.3659 | 0.0115 | 14.44 | 20 | 0.1098 |
| nv | rl + `open:next-val` | 0.2767 | 0.2954 | 0.0094 | 14.81 | 18 | 0.0284 |
| ac+nv | ac + `open:next-val` | **0.2932** | 0.3586 | 0.0109 | 15.28 | 15 | 0.1163 |
| vlabel | rl + `open:vlabel` | 0.2709 | 0.2885 | 0.0094 | 14.81 | 18 | 0.0206 |
| ac+vlabel | ac + `open:vlabel` | 0.2768 | 0.3447 | 0.0091 | 15.28 | 15 | 0.1116 |
| ac+self+vlabel | ac + both | **0.2932** | 0.3447 | 0.0109 | 15.28 | 15 | 0.1005 |
| **head+vlabel** | gate over `open:vlabel` | **0.2890** | 0.3523 | 0.0114 | 14.67 | 24 | **0.2101** |
| head+vlabel-fold | weight table | 0.2850 | 0.3590 | 0.0104 | 16.00 | **0** | 0.2101 |
| head+stop+vlabel | gate over both | 0.2712 | 0.3614 | 0.0112 | 14.83 | 11 | 0.2101 |
| head+stop+vlabel-fold | weight table | 0.2582 | 0.3486 | 0.0107 | 15.56 | 3 | 0.2101 |
| **ev-margin** | break-even `max v > κ` | 0.1263 | 0.2488 | 0.0092 | **9.06** | 33 | 0.2101 |
| **ev-pool** | break-even `Σ(v−κ) > 0` | 0.0937 | 0.1863 | 0.0076 | **7.33** | 34 | 0.2101 |

## Results (`--value-label useful` — `p·1.0`)

| arm | reward/ep | evals/ep | stops | AP |
|---|---|---|---|---|
| vlabel | 0.2594 | 14.81 | 18 | 0.0653 |
| ac+vlabel | 0.2116 | 14.50 | 18 | 0.1879 |
| ac+self+vlabel | 0.2413 | 14.61 | 18 | 0.1131 |
| head+vlabel | **0.2850** | 16.00 | **1** | 0.2101 |
| head+vlabel-fold | 0.2396 | 13.17 | 25 | 0.2101 |
| head+stop+vlabel | 0.2425 | 9.22 | 31 | 0.2101 |
| ev-margin | 0.1279 | 5.56 | 36 | 0.2101 |
| ev-pool | 0.0067 | 3.22 | 36 | 0.2101 |

## Results (`--value-label pays` — `p·0.2` floor)

| arm | reward/ep | evals/ep | stops | AP |
|---|---|---|---|---|
| vlabel | 0.2668 | 14.61 | 16 | 0.1239 |
| ac+vlabel | **0.2806** | 15.31 | 14 | 0.1142 |
| head+vlabel | 0.2413 | 9.50 | 31 | 0.2101 |
| head+stop+vlabel | **0.2735** | 13.72 | 12 | 0.2101 |
| ev-margin | 0.0883 | 7.25 | 33 | 0.2101 |
| ev-pool | 0.0897 | 6.56 | 34 | 0.2101 |

## Results (`--value-label delta` — control; `nv`/`ac+nv` replicate round 6 exactly)

| arm | reward/ep | evals/ep | stops | AP |
|---|---|---|---|---|
| nv | 0.3008 | 15.31 | 16 | 0.1042 |
| **ac+nv** | **0.2764** | 15.36 | 12 | **0.2074** |
| vlabel | 0.2626 | 15.61 | 13 | 0.0156 |
| **ac+vlabel** | **0.3009** | 15.28 | 17 | 0.1069 |
| ac+self+vlabel | 0.2783 | 15.17 | 21 | 0.0646 |
| head+vlabel | **0.2866** | 15.67 | 18 | 0.2101 |
| head+vlabel-fold | 0.2726 | 14.67 | 22 | 0.2101 |
| **ev-margin** | **0.2821** | 15.31 | 23 | 0.2101 |
| **ev-pool** | **0.2821** | 15.31 | 23 | 0.2101 |

Anchors (all runs): oracle-stop ceiling 0.3877 @ 3.75 evals ·
baked-prior 0.2758 / AP 0.2101 · heuristic drain 0.3595 / AP 0.0450 ·
random 0.1339 / AP 0.0123.

## What the priced labels showed

* **The expected-yield columns are a real — but not better — decision
  signal.**  Held-out decision-row probe (`head+vlabel`, oracle
  continue-labels, base rate 0.155): under `useful` injection
  `x:stop:vl_pool` solo AP **0.3859**, `vl_best`/`vl_margin` 0.3818 —
  matching round 5's `next_value` peak (0.3818) and beating the
  progress counter (0.3287); under `ev` the same columns land at
  0.29/0.27, under `delta` 0.32/0.17.  Head eval AP tops at 0.4348
  (`head+stop+vlabel` under `useful`) — below round 3's 0.5494 peak.
  Pool aggregation still forgives the weak per-candidate APs, but the
  payoff pricing adds nothing the raw max/Σ didn't already carry —
  `vl_net`/`vl_margin` are affine in the same injected table, and the
  head spans them interchangeably (LOO deltas ≈ 0).

* **The break-even gate answers the calibrated-variant question,
  and the answer is "the predictor can't afford it".**  Under every
  *verdict* label the analytic gates over-stop brutally: `useful`
  `ev-pool` quits at 3.22 evals for 0.0067 reward — the memorized-7-
  positives net predicts ≈0 on held-out pages, so "expected yield
  covers cost" fails immediately; `pays` gates land 6.6–7.3 evals,
  ~0.09; `ev` 7.3–9.1, ~0.09–0.13.  Only under `delta` — the
  *wrong* label but the calibrated one — do the break-even gates land
  near the drain point (15.31 evals, 23 tail stops) and earn
  **0.2821** (+0.009 over incumbent, AP-preserving at 0.2101 by
  incumbent-ranker construction).  The asymmetry is now measured from
  both ends: a reward-calibrated τ mostly declines early quits; an
  evidence-calibrated κ executes them and starves — because the
  predictors *under-predict* held-out value (in-sample AP 1.0,
  held-out ≤0.10 vs the labels that matter).

* **The `ac` twin helps reward, not the ranking.**  `ac+vlabel` under
  `delta` posts 0.3009 — best arm reward in the round — at AP 0.1069
  (the shared-softmax corruption, fifth feature family).  Under `ev`
  it is 0.2768 (reward-clear) / 0.1116; under `pays` 0.2806 / 0.1142.
  The V fits are legible and signed correctly (`ac+vlabel`/`delta`:
  `x:stop:vl_frac` +0.30, `st:pending_frac` +0.32, `st:seen_frac`
  −0.22 — "remaining return tracks remaining predicted value"), but
  the stop timing still does not appear: every arm operates at
  14–16 evals/ep.

* **Gates earn, folds degenerate — again.**  `head+vlabel` clears
  both bars under `ev` (0.2890 @ 24 real stops) and `delta` (0.2866 @
  18), `head+stop+vlabel` under `pays` (0.2735 @ 12) — all
  non-serializable (second scorer + τ).  The folds either miss
  (`delta` 0.2726 — the recurring 0.0004-short knife-edge) or clear
  by never stopping (`ev` fold: 0.2850 @ **0 stops** — the documented
  confound, now at a fifth feature family).

## Verdict (round 7): not earned — `OPEN_GAME_PLAYER_WEIGHTS` stays

Gate: held-out reward ≥ 0.273 **and** AP ≥ 0.210, honestly — a
never-stop clear or an AP collapse is an honest negative.  No
*serializable* arm clears both: the reward-clearing weight-table arms
all carry collapsed rankings (`ac+vlabel`/`delta` 0.3009 / 0.1069;
`ac+nv`/`ev` 0.2932 / 0.1163; `ac+vlabel`/`pays` 0.2806 / 0.1142;
`ac+self+vlabel`/`ev` 0.2932 / 0.1005); the folds miss or
never-stop; `ac+nv` under `delta` re-misses AP by 0.003 (the
round-6 closest-ever, replicated).  The gated arms clear honestly —
`head+vlabel`/`ev` 0.2890 @ 24 stops is the best earning gate yet —
and `ev-margin`/`delta` 0.2821 @ 23 stops is the strongest *gate*
result posted without a learned head at all — but a gate is a second
scorer (here: the injected table + κ), not a weight table.  **No
bake.**

The seven-round arc, updated: pricing the label into payoff units
moved the columns' *meaning* but not their *information* — the same
injected table drives both `open:next-val` and `open:vlabel`, and
head APs confirm the nets' held-out ceiling (≤0.10 on the labels that
pay, 0.93 on the one that doesn't).  The break-even variant adds a
second failure mode to the diagnosis: it is not only that no feature
certifies "pending is duds" — the predictors' own *magnitude* is
miscalibrated enough that the cost-table's own threshold executes the
wrong operating point.  A bakeable stop policy still needs what
rounds 5–7 keep re-deriving: a per-candidate usefulness signal that
transfers held-out (verdict-conditioned or verifier-visible
features), or a stop decision that lives outside the shared softmax —
the gates keep earning precisely because they already live there.

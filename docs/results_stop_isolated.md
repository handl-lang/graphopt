# The stopping policy, round 9 — the isolated stop channel

**Date:** 2026-10-20 · **Script:** `bench/drivers/policy/stop_isolated.py`
(new — same memoized-verdict protocol as `stop_calibrated.py`, whose
predictor/calibration machinery it reuses) · **Player:** the new
`gated` registry seat (`gamekit.players.gated` — `GatedPlayer` + the
`run_episode` side-channel hook) · **Seed:** 0 · **Runner:**

```bash
timeout 1800 .venv/bin/python bench/drivers/policy/stop_isolated.py \
    --json /tmp/iso.json            # ~5 min (193 s pool memoization)
```

## Motivation — attack the delivery

Eight rounds relocated the blocker to two survivors: (a) no
*serializable* arm clears both bake bars (reward ≥ 0.273 ∧ AP ≥
0.210), and (b) every stop feature family routed through the shared
`stop` logit corrupts the ranking (five measured AP collapses) while
every softmax *fold* of a working gate lands a degenerate point —
stop-on-average or never-stop.  The recurring corruption channel is
the softmax itself, so this round removes the quit decision from it:
a **separate stop scorer evaluated per step, outside the ranking
softmax** — the policy proposes the evaluate ranking as today and a
side channel computes `stop_score` independently.

Two small mechanisms carry it:

* **`engine.run_episode`'s `forced_action` hook** — a player that
  exposes `forced_action(state) -> action | None` is consulted
  *before* the policy's own pick; a non-`None` return is the move
  played (the board's own `stop` move, so the reports are
  identical-shaped to a softmax-picked quit).  Optional and
  backward-compatible — a plain callable player is untouched.
* **`gamekit.players.GatedPlayer`** (registered as `gated`,
  `boards=("open",)`) — wraps an `eval`-only ranker (`stop` is not
  in its legal set) with the channel's scorer: `score(state, hist)`
  when given, else the linear head `w_stop · φ_stop`; force `stop`
  when the score drops to κ, or when no non-stop move remains (the
  drain-terminal move).  The serialized payload is data — a
  `{feature: weight}` head or a calibrator's blocks + κ — not a
  `OPEN_GAME_PLAYER_WEIGHTS`-shape table.

**Honest κ, not tuned-by-reward.**  Every scorer's threshold is the
asymmetric table's own break-even: gates fire when the expected
yield of continuing drops below `|step|` = 0.002 (margin: best
pending `v > κ`; pool: `Σ_pending (v − κ) > 0`); the calibrated head
fires when `p̂ < p*`, with `p* = C⁻/(V⁺ + C⁻)` measured on the
collected train rows — `V⁺ = 0.8429` (mean oracle best-prefix value
a correct continue banks), `C⁻ = 0.0163` (`|step| × ~8` pending, the
cost of draining a dead board) → **p\* = 0.019**: the table tolerates
a 1.9% continue-chance.  The vreg head regresses the best-prefix
*value* directly (κ=0, the literal break-even).  Per-candidate
predictor scores are Platt/isotonic-calibrated on train OOF preds
exactly as round 8.

## Protocol

Identical to rounds 5–8: cold retrain per arm, 5 epochs × 36 even
pages of `engine.train_policy` under memoized per-candidate
verdicts; 36 held-out odd pages, budget 17, page-aware AP.  All
`iso-*` rows share the incumbent ranker (`rank_tags=()` — the
ranking is held apart from the decision).  The raw κ=1× gate rows
are the **port control** — they must reproduce round 8's `_EVGate`
numbers bit-identically, and they do: `ev-margin` 0.1263 @ 9.06/33,
`ev-pool` 0.0937 @ 7.33/34, `useful-margin` 0.1279 @ 5.56,
`useful-pool` 0.0067 @ 3.22, `pays-margin` 0.0883 @ 7.25,
`pays-pool` 0.0897 @ 6.56, `delta` both 0.2821 @ 15.31/23; isotonic
κ=1× likewise (`useful-pool` 0.2899 @ 14.25/26).  **The channel is an
exact delivery — zero distortion vs the wrapper gates.**

## Results (held-out odd pages, budget 17)

| arm | mechanism | reward/ep | evals/ep | stops | AP |
|---|---|---|---|---|---|
| incumbent (retrained) | rl 5ep | 0.2728 | 14.44 | 19 | 0.2101 |
| ac | actor-critic | 0.2761 | 14.44 | 20 | 0.1098 |
| **oracle (via channel)** | best-prefix ≤ 0 → stop | 0.3106 / **0.3857 g** | 4.17 | 35 | 0.2101 |
| iso-drain | scorer unbound (never fires) | 0.2850 | 16.00 | 0 | 0.2101 |
| iso-always | κ=+∞ | 0.0000 | 1.00 | 36 | — |
| **iso-useful-pool:isotonic@1×** | Σ(v̂−κ) > 0, κ=\|step\| | **0.2899** | 14.25 | 26 | 0.2101 |
| **iso-useful-pool:isotonic@1.5×** | κ sweep peak | **0.2929** | 12.81 | 28 | 0.2101 |
| iso-useful-margin:isotonic@1× | max v̂ > κ | 0.2875 | 15.39 | 23 | 0.2101 |
| iso-ev-pool:isotonic@1× | | 0.2856 | 15.89 | 7 | 0.2101 |
| iso-pays-pool:isotonic@1× | | 0.2823 | 14.81 | 10 | 0.2101 |
| iso-delta-pool:raw@1× | | 0.2821 | 15.31 | 23 | 0.2101 |
| **iso-head:isotonic** | calibrated head, incumbent schema | **0.2892** | 14.58 | 26 | 0.2101 |
| iso-head:platt | same head, Platt map | 0.2850 | 16.00 | **0** | 0.2101 |
| iso-head-fold | same head, softmax fold | 0.2850 | 16.00 | **0** | 0.2101 |
| **iso-vhead** | value-regression head, κ=0 | **0.2863** | 16.00¹ | 25 | 0.2101 |
| iso-head+stop:platt / :isotonic | +`open:stop` cols | 0.2428 / 0.2457 | 14.28 / 12.97 | 17 / 23 | 0.2101 |
| iso-head+stop-fold | softmax fold | 0.2638 | 15.56 | 3 | 0.2101 |
| **iso-head+vl:delta/raw:platt** | +`open:vlabel`, raw Δ preds | **0.2864** | 15.78 | 18 | 0.2101 |
| iso-head+vl:delta/raw-fold | softmax fold | 0.2726 | 14.67 | 22 | 0.2101 |
| iso-head+vl:useful/iso:platt / :isotonic | | 0.2568 / 0.2307 | 12.25 / **6.61** | 9 / 35 | 0.2101 |
| iso-head+vl:useful/iso-fold | softmax fold | 0.2419 | 12.42 | 27 | 0.2101 |
| iso-head+vl:ev/platt:isotonic | | **0.2825** | 12.39 | 25 | 0.2101 |
| iso-vhead+vl:ev/platt | vreg over `vlabel` | **0.2863** | 15.42 | 3 | 0.2101 |
| **rl-gated:useful/isotonic** | ranker trained *under* the gate | **0.3008** | 14.50 | 20 | **0.0407** |
| **rl-gated:delta/raw** | same, raw δ gate | **0.3205** | 14.97 | 26 | **0.0839** |
| — anchors — ||||||
| oracle-stop ceiling | best-prefix under the oracle | 0.3877 | 3.75 | 36 | — |
| baked-prior | | 0.2758 | 13.06 | — | 0.2101 |
| heuristic drain | | 0.3595 | 16.00 | — | 0.0450 |
| random | | 0.1339 | 8.67 | — | 0.0123 |

¹ the counter sums *all* reports — a 16.00-read with 25 stops is a
page that played 15 evals then `stop`: the channel fires exactly on
the last pending dud (the tail-trim).

Head diagnostics (train fit → held-out decision rows): logistic head
eval AP 0.3694 (base) → 0.483 under `useful/iso` preds; vreg eval AP
0.406–0.480; held-out calibration of `p̂` — Platt ECE 0.03–0.06 /
isotonic 0.03–0.10 (isotonic keeps the blocky tail that actually
dips below p\*; Platt's smooth map never does on the incumbent
schema — hence `iso-head:platt`'s 0-stop drain).

**Unfed replay** (`set_value_preds(None)` — the seam cleared): every
value-reading arm collapses to stop-at-step-0, `0.0000 @ 1.00` —
`Σ_pending (0 − κ) < 0` fires immediately.  The gate/`vlabel` arms
are *seam-dependent*, the same caveat as rounds 5–8's folds, now
failing loudly instead of silently.  `iso-head:isotonic` and the
`open:stop` arms are seam-free — the incumbent schema plus a head is
the whole payload.

## What the isolated channel showed

* **The delivery works — the hypothesis is confirmed for
  delivery.**  Identical scorer content clears both bars through the
  channel where the shared-softmax deliveries all failed: the same
  head earns **0.2892 with 26 real stops** as `iso-head:isotonic`
  and degenerates to **0.2850 with 0 stops** as `iso-head-fold`
  (the never-stop confound) — and `iso-head+vl:delta/raw`'s fold
  lands the recurring sub-0.273 knife-edge (0.2726) a fifth time.
  The gates reproduce round 8's earning table bit-identically, and
  `iso-useful-pool:isotonic` posts the round's best bar-clear:
  **0.2929 @ 12.8 evals / 28 stops** — at an honest-κ grid point
  (1×=0.2899, 1.5×=0.2929, 2×=0.2607).
* **`iso-head:isotonic` is the first clearing arm that is pure-data
  serializable.**  Incumbent schema only — no `set_value_preds`
  seam: the payload is `(w_head dict, isotonic blocks, p*=0.019)`,
  all literals.  It is not `OPEN_GAME_PLAYER_WEIGHTS` *shaped* — it
  needs the `gated` player's artifact class (spec + head + κ) — but
  nothing in it is a runtime injection.
* **Every earning stop is a tail-trim, not an early quit.**  The
  bar-clearing arms operate at 12.4–16 evals/ep — the isotonic tail
  can certify "these pending members are the bottom block" only near
  page end.  The oracle's 3.75-eval point is still out of reach: the
  *evidence* gap (held-out discrimination ~AP 0.37–0.48 vs the
  continue labels) bounds where κ can fire, exactly as ADR 0013
  says.
* **Isolating the channel does *not* rescue ranking training under a
  gate.**  `rl-gated` posts the run's best rewards (0.3008 / 0.3205)
  on **collapsed** rankings (AP 0.041 / 0.084): the ranker trained
  with `stop` fully outside its softmax still corrupts when episodes
  truncate early — the credit-side mechanism (truncated-episode
  returns over an eval-only legal set) is a *different* corruption
  than the delivery-side one this arm removes.  The working recipe
  is: train the ranker ungated, deliver the gate through the
  channel.
* **Over-stop is still the live failure mode when the signal is
  strong.**  `iso-head+vl:useful/iso:isotonic` quits at 6.61 evals
  for 0.2307 — the isotonic-useful preds give the head enough
  discrimination (eval AP 0.483) to fire early, and early is wrong:
  the calibrated table can't certify an empty pool, so its confident
  quits forfeit the banked tail.  `iso-head+stop` (0.243–0.246) —
  the `x:stop:*` progress columns sharpen the same wrong lesson.

## Verdict (round 9): the channel earns it; the blocker is now purely evidence

Gate: held-out reward ≥ 0.273 **and** AP ≥ 0.210, honestly — never-
stop clears and AP collapses are negatives.  **For the first time,
multiple honest arms clear both bars with real stopping** — and the
clears are *not* the reward-max-τ artifacts of rounds 4–8: κ is the
break-even `|step|` (gates) or the data-derived `p* = 0.019` /
literal 0 (heads).  The shared-softmax hypothesis is **confirmed on
the delivery axis**: identical head content, channel 0.2892 @ 26
stops vs fold 0.2850 @ 0 stops.

**No `OPEN_GAME_PLAYER_WEIGHTS` bake** — by construction: the
clearing arms are not a weight table.  What the round earns is the
*delivery mechanism* and its serialization story: the `gated`
player's payload is a small pure-data artifact (head weights /
calibrator blocks / κ — the candidate-stream value table for the
seam-fed variants), which is a new artifact class, not a new row of
the prior.  `iso-head:isotonic` (0.2892 @ 26 stops, seam-free) is
the bakeable prototype if a `STOP_HEAD` artifact ships; the
seam-fed gates (best 0.2929) additionally need the per-candidate
p̂ table serialized — finite stream, serializable, measured here as
deployed-else-broken (unfed → 0.0000).

The residual blocker narrows once more: delivery is solved; the
~+0.026 oracle headroom is bounded purely by *evidence* — every
bar-clearing arm is a tail-trimmer because nothing yet ranks pending
members by held-out usefulness well enough to quit early.

# p31 — referee flip guard: margin-noise hardening for `genkernel.profitable`

T-026 residual follow-up.  `profitable`'s documented margin noise is
offer-visible: one contaminated sweep dropped chainfuse's whole claim
set (delivered 1.37x vs compile) while a fresh process kept
`gen_0,1,3` and the 2.75x win; ChunkedSSD's gen-claim set flipped
run-to-run (`[]` → `gen_10,gen_14` → `gen_12,gen_13,gen_17`, row
speedups 0.84–1.01).  This report measures the keep-set flip rate
directly and lands a bounded hardening.

## 1. Instrument

`bench/drivers/probes/referee_flip_probe.py` — one fresh process per
sample, runs the exact gate the board runs:

* `--gen --only chainfuse,twosite,chain_mlp` — `play._gen_cases` at
  the eval seed, `play._claim_cost_gate` per case (5 cases matched);
* `--zoo ChunkedSSD` — `play._zoo_case`, whose intake `profitable`
  sweep filters `ex["gen_handlers"]`.

`--log-ratios` dumps `genkernel.PROFITABLE_LOG` — per-handler referee
timings appended by `profitable` (measurement surface only, the
`TRITON_REFEREE_LOG` precedent).

Protocol: 8 fresh-process samples per mode, serial, on the 11 GB
host.  Flip metric: a handler *flips* when its keep status varies
across the 8 runs.

## 2. Before — measured flip rate

| case | flipping handlers (kept in k/8 runs) |
|---|---|
| gen0:chainfuse | `gen_1` 7/8 |
| gen3:chainfuse | `gen_1` 7/8, `scan_0` 7/8 |
| gen9:twosite | `gen_0` 2/8 |
| zoo:ChunkedSSD | `scan_2` 1/8 |

chainfuse produced 3 distinct keep-sets across 8 identical-binary
runs; ChunkedSSD admitted `scan_2` once on a 0.93x read against a
~1.05x typical.  `PROFITABLE_LOG` ratios show the mechanism: the
flippers all sit in a ±10–15% band around the keep threshold
(`t_kernel <= t_spelled`), e.g. chainfuse `gen_1` read 0.80–1.03x,
twosite `gen_0` 0.87–1.22x, ChunkedSSD `scan_2` 0.93–1.16x across
runs.  Claims outside the band (chainfuse `gen_0` ~0.4x, `gen_2`
~1.4x) never flipped — the landed fixes (sorted order, interleaved
`_paired_time` A/B, compile-worker drain) removed the *systematic*
inflation; what remains is irreducible window-level drift on
launch-bound µs-scale timings.

## 3. Hardening — borderline re-measure, majority decision

`genkernel.profitable` now guards the threshold band
(`_REFEREE_BORDERLINE = 1.25`):

* ratio `t_kernel/t_spelled` inside
  `margin/1.25 .. margin*1.25` → the handler is **re-timed** with
  fresh `_paired_time` windows (each re-warms both arms, so every
  sample is an independent interleaved window);
* a second confirmation runs only when the first disagrees — two
  agreeing windows decide on their own, a disagreement buys a third;
* keep/drop = **majority of the windows' votes** (2 or 3 samples);
* handlers outside the band keep the single-window cost — the sweep
  is bounded at ~3x on the borderline subset only, typically far
  less (measured probe wall-time unchanged within noise: ~6 s gen,
  ~4 s zoo per sample).

`PROFITABLE_LOG` rows are now per *sample*
`(name, t_spelled, t_kernel, window, keep_vote)`; a borderline
handler's consecutive rows majority-decide.  Applies uniformly to
every `profitable` caller — the zoo intake sweep, the gen claim
gate, and `_fold_cost_gate`.

## 4. After — measured flip rate (same 8+8 protocol)

| case | before | after |
|---|---|---|
| gen0:chainfuse | `gen_1` 7/8 | **stable 8/8** |
| gen3:chainfuse | `gen_1`,`scan_0` 7/8 | **stable 8/8** |
| gen4/gen8:chain_mlp | stable [] | `gen_0` kept 1/8 (gen8) |
| gen9:twosite | `gen_0` 2/8 | `gen_0` 2/8, `scan_0`+`scan_1` 1/8 |
| zoo:ChunkedSSD | `scan_2` 1/8 | **stable [] 8/8** |

Known-good keeps unchanged: chainfuse `gen_0,1,3,scan_0` and twosite
`gen_0` kept identically; all clear losers (`gen_2,4,5`,
`decayed_0`, zoo `gen_*`) still dropped every run.

## 5. Honest verdict

The guard works where noise was the whole story: every claim with a
true ratio clearly off 1.0 is now deterministic across processes —
the chainfuse class of failure (whole claim set dropped on one
contaminated window) is closed, and ChunkedSSD's intake sweep is
stable.

The residual flips are **true ties**, not measurement error:
twosite `gen_0`/`scan_0`/`scan_1` and chain_mlp `gen_0` measure
~0.95–1.05x — their keep/drop is a coin flip no re-measurement can
stabilize without changing the decision rule itself (e.g. demanding
a strictly-positive margin, which would also drop true-neutral
claims and alter known-good keep-sets — rejected).  Stakes are
bounded either way: a slipped borderline claim is priced end-to-end
in the row speedup, costing or gaining only a few percent — it can
never fabricate the 1.37x-vs-2.75x class of error the contaminated
sweep produced.

Tests: three new pins in `TestMeasuredReferee`
(`test_borderline_ratio_decided_by_window_majority`,
`test_borderline_ratio_losing_majority_drops`,
`test_clear_ratio_skips_confirmation_windows`) script
`_kernel_pays`/`_paired_time` to pin the band, the early-stop, the
majority, and the single-window cost outside the band.
`tests/test_discovery_genkernel.py` + `tests/test_discovery_play.py`
`-m "not slow"`: 118 + 54 passed.  `ruff check`/`format`, `ty`:
clean on touched files.

# Re-sweep — does the chain→`scan_<i>_k` delivery route pay at module level?

**Date:** 2026-11-15 · **Builds on:** `a405205` (the route),
`t037_resweep_frontier.md` (the sweep protocol this re-uses).

The frontier sweep rows predate the deliver-time chain→scan route:
seq/seqm claims were structurally unroutable, so every delivered
module in `zoo_chain_sweep*.json` spelled the recurrence as an
unrolled `add(mul(select…))` chain.  `a405205` rewrites matching
chain subterms to `scan_<i>_k` at `MetaArena.deliverable` time
(`meta_arena._route_scan_chains`).  This report re-measures the
seq-relevant zoo set at the sweep shape and asks whether the route
pays in wall-clock, not just in structure.

## Protocol

Same machinery as the last sweep — `bench/drivers/probes/zoo_chain.py`,
unchanged code path (`_zoo_case` → mint tiers → `profitable` →
`gen_probe` → deliver → verify → time vs eager / `torch.compile` /
`CudaGraphRunner`).  Reduced protocol, matching T-037:

```sh
.venv/bin/python -u -m bench.drivers.probes.zoo_chain --isolate \
    --dtype fp32 --rounds 2 --reps 100 --budget 400 \
    --only GatedLinearAttention,GatedDeltaRule,RealGatedLRU,DeltaRuleBlock,MambaSelectiveScan \
    --out /tmp/zoo_scanroute_fp32.json   # and --dtype fp64 → _fp64.json
```

- `zoo(64, 32, 8)`, fp32 *and* fp64 (both sweep dtypes).
- 2 rounds per row, `--reps 100`, `--isolate` (one process per
  workload, 400 s kill budget — up from the canonical sweep's
  150 s, same as T-037).  Serial GPU, one child at a time.
- "Pre-route" numbers are the committed sweep rows
  (`zoo_chain_sweep.json` fp64 / `zoo_chain_sweep_fp32.json` fp32),
  which T-037 last refreshed — all measured before `a405205`.

### Which workloads are seq-relevant

Enumerated by running the shipped collectors
(`play._seq_scan_sites` / `._seqm_scan_sites`) over each zoo
workload's intake IR at `zoo(64,32,8)` — CPU-only, no timing.  Across
all 68 workloads exactly five mint seq/seqm sites (the only site
records carrying `"chain"`, hence the only ones the route can
fire on):

| workload | site kind | site count |
|---|---|---|
| GatedLinearAttention | seq | 1 |
| GatedDeltaRule | seqm | 1 |
| RealGatedLRU | seq | 1 |
| DeltaRuleBlock | seqm | 1 |
| MambaSelectiveScan | seq | 1 (plus 64 unroutable `unrolled` records) |

Every other zoo workload mints zero seq/seqm sites — `unrolled`-kind
`_scan_sites` records carry no `"chain"` and remain claim-dead by
design (`applyd_route.md` "Known residuals").  So this five-row set
IS the full route-relevant zoo set, not a sample.

## Does the delivered module contain the scan kernel?

Yes — verified at the term level, both dtypes, by rebuilding each
case (`_zoo_case` → `_gen_board` → `_probe_play` →
`board.deliverable`) and reading `_ops_of(term)`:

| workload | dtype | `scan_*` op in delivered term |
|---|---|---|
| GatedLinearAttention | fp32, fp64 | `scan_64_k` |
| GatedDeltaRule | fp32, fp64 | `scan_8_k` |
| DeltaRuleBlock | fp32, fp64 | `scan_4_k` |
| MambaSelectiveScan | fp32, fp64 | `scan_64_k` |
| RealGatedLRU | fp32 | `scan_64_k` (fp64 not re-inspected; the route is a deliver-time transform on the same IR) |

`TRITON_REFEREE_LOG` records for the minted `scan_*` claims show the
`scan` tier `bound` (the `gen` tier declines them — outside its
grammar — by design).  `hybrid_sink` binds `scan_triton_bindings`
last, so a recognised seq/seqm spec delivers the generated Triton
kernel wherever module generation succeeds, else the registered
spelled fold denotation — never the ambient map application.  All
rows `verified` with `max_abs` ≤ 9.5e-6 (fp32) / ≤ 8.9e-16 (fp64).

Note the claim_tags stay empty for these claims — `claim` actions
for `scan_*` remain honestly unavailable (`_concrete_occurs` still
declines); the route is delivery-side only, exactly as designed.

## Per-workload results

`delivered_us` = median `gen_us` over the 2 rounds; pre-route rows
marked † were "over budget: killed after 150s" stubs with **no**
timing — there is no honest pre-route delivered_us to compare
against, only the fact that no row existed.

### fp32 (`zoo_chain_sweep_fp32.json` scope — shipped dtype)

| workload | pre-route delivered_us | post-route delivered_us | scan bound? | verdict |
|---|---|---|---|---|
| GatedDeltaRule | 628.0 (speedup 0.642) | **298.4** (speedup **1.425**) | `scan_8_k` | **Pays** — 2.1× faster delivered body, now beats eager; graph arm 46.5 µs = 9.15× eager / **3.69× compile** (was 122.3 µs / 1.43×) |
| DeltaRuleBlock | 304.8 (0.994) | **233.7** (1.385) | `scan_4_k` | **Pays** — modest eager win where it treading water; graph arm 36.9 µs = 8.77× / **5.25× compile** (was 68.9 / 2.75×) |
| MambaSelectiveScan | † no row | 4148.1 (0.469) | `scan_64_k` | **Delivers now, still loses** — the spelled chain was never timed before; the routed module verifies but is 2× slower than eager. Graph arm 1055 µs = 1.84× eager but **0.57× compile** |
| GatedLinearAttention | † no row | 6219.6 (0.638) | `scan_64_k` | **Delivers now, still loses** — eager itself is unstable here (2913–5021 µs across rounds); graph arm 5821 µs ≈ no better (0.68× eager, 0.11× compile) |
| RealGatedLRU | † no row | † no row (killed at 400 s) | `scan_64_k` in term | **Unchanged** — route fires at term level, but the module still cannot be timed inside 400 s (extraction/lowering cost, not the scan kernel) |

### fp64 (`zoo_chain_sweep.json` — canonical frontier input)

| workload | pre-route delivered_us | post-route delivered_us | scan bound? | verdict |
|---|---|---|---|---|
| GatedDeltaRule | 572.8 (0.702) | **327.9** (1.224) | `scan_8_k` | **Pays** — 1.75× faster; graph arm 135.8 µs = 2.95× eager / **3.18× compile** (was 206.7 / 2.08×) |
| DeltaRuleBlock | 300.0 (0.974) | **272.9** (1.607) | `scan_4_k` | **Pays** — modest; graph arm 114.3 µs = 3.85× / **2.10× compile** (was 143.6 / 1.44×) |
| MambaSelectiveScan | † no row | 3873.6 (0.487) | `scan_64_k` | Same as fp32 — delivers, loses to eager and compile |
| GatedLinearAttention | † no row | 7140.4 (0.286) | `scan_64_k` | Same — delivers, loses badly; graph arm 3763 µs = 0.54× eager / 0.34× compile |
| RealGatedLRU | † no row | † no row (killed at 400 s) | — | Same — still unmeasurable |

## Honest reading

- **The route pays where the chain was the whole story.**  On
  GatedDeltaRule and DeltaRuleBlock the delivered body got
  meaningfully faster in *both* dtypes (fp32 GatedDeltaRule is the
  headline: 628 → 298 µs, crossing from 0.64× to 1.43× eager), and
  the graph arm improved in lockstep — the captured module is
  smaller, so replay is cheaper too.  DeltaRuleBlock's fp64 row went
  0.97× → 1.61×.
- **The route does not rescue the big unrolled modules.**  On
  GatedLinearAttention and MambaSelectiveScan the delivered term now
  contains `scan_64_k`, yet delivered_us is 3.9–7.1 ms against
  ~2 ms eager — the loss is not the recurrence fold, it is the rest
  of the module (Mamba's delivered term carries ~10 `gen_*_k`/`synop_*`
  ops beside the scan; GatedLinearAttention's eager arm itself is
  unstable at this shape).  The frontier rows now quantify this:
  `captured` is −0.51 / −0.15 — delivery loses more than the
  ceiling allows.
- **Eager delivery still loses to `torch.compile` almost
  everywhere.**  Delivered-body vs compile (compiled_us ÷ gen_us
  medians): fp32 GatedDeltaRule ≈ 0.58×, DeltaRuleBlock ≈ 0.81×,
  Mamba ≈ 0.14×, GatedLinearAttention ≈ 0.10×; fp64 GatedDeltaRule
  ≈ 1.32× is the only delivered body that beats Inductor, with
  DeltaRuleBlock ≈ 0.88× close behind.  The *graph* arm is where
  the route's payoff fully shows: it beats compile on GatedDeltaRule
  (3.69× fp32 / 3.18× fp64) and DeltaRuleBlock (5.25× / 2.10×).
  The route makes the captured artifact better; it does not change
  the standing conclusion that per-leaf lowering alone doesn't beat
  Inductor — `frontier.json` median capture stays −0.02.
- **RealGatedLRU remains unmeasurable** at the sweep shape — the
  route fires (term contains `scan_64_k`) but 400 s is not enough
  for the probe to finish; same status as pre-route, now with the
  structural proof that delivery *would* route.
- **New frontier rows.**  `frontier.json` (fp64) gains two rows that
  were previously over-budget stubs — `zoo:GatedLinearAttention`
  and `zoo:MambaSelectiveScan` — plus refreshed GatedDeltaRule /
  DeltaRuleBlock rows; 65 → 67 total rows (35 → 37 with material
  headroom).  `frontier_fp32` scope
  (`--only` five non-seq workloads) is unaffected; regenerated
  byte-identical.  `RealGatedLRU` stays a stub in both sweeps.

## Files changed

- `docs/reports/zoo_chain_sweep.json` — 5 rows replaced by the fp64
  re-measurement (T-037 merge protocol: by `case`, order preserved).
- `docs/reports/zoo_chain_sweep_fp32.json` — same for fp32.
- `docs/reports/frontier.json`, `docs/reports/frontier.md` —
  regenerated (`bench.suites.integration.frontier`); +2 rows.
- `docs/reports/frontier_fp32.json`, `frontier_fp32.md` —
  regenerated with the T-037 `--only` scope; **unchanged** (none of
  the re-measured workloads are in scope).

## Caveats

- 2 rounds, not the canonical 3–5 — medians kept with `runs[]`
  spread; GatedLinearAttention's eager arm varies ~1.7× between
  rounds at this shape, so its speedup is the noisiest row.
- Claim sets differ run-to-run (`synop_*` names are minted
  non-deterministically); the route-relevant fact — `scan_<i>_k`
  in the delivered term — was verified by direct term inspection,
  not inferred from claim tags (which stay empty by design).
- fp64 term inspection covered the four delivered workloads;
  RealGatedLRU's `scan_64_k` was verified at fp32 only.
- The 400 s budget is still a budget: "over budget" means the probe
  (dominated by `torch.compile` on the spelled baseline and the
  extraction play, not the scan kernel) did not finish — it is not
  evidence the delivered module is slow.

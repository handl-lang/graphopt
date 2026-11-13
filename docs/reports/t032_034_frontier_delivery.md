# T-032/T-033/T-034 — are the frontier "uncaptured headroom" rows the seq/seqm opportunity, or a different gap?

Date: 2026-11-13.  Question behind the three agenda rows (all
`frontier_delivery_gap`, "Uncaptured headroom", mined by
`bench/drivers/agenda/agenda_miner.py::_sig_frontier` from
`docs/reports/frontier.json`): does a paying candidate exist that the
delivery path fails to route, or is the headroom something else?

**Short answer: the rows are stale.**  `frontier.json` and both of its
inputs (`zoo_chain_sweep.json`, `zoo_kcount.json`) were last committed
2026-10-26 — *before* the `seq` tier (`89ea185`), the seq view-lift
(`2d13865`), and the `seqm` matrix-state tier (`df2923c`) all landed on
2026-11-13, the same day the agenda miner ran.  The sweep has never been
re-measured since.  Per-row verdicts below; two of the three paying
candidates now exist **and are routed** through the exact path the
sweep measures, and one row (DecoupledMLA) is not a scan workload at
all.

## Measurement chain (read before interpreting)

- `agenda_miner._sig_frontier` (`bench/drivers/agenda/agenda_miner.py:431+`)
  reads `frontier*.json` rows with `headroom_us > ~5µs` and
  `captured` missing or < 0.5.
- `bench/suites/integration/frontier.py` builds `frontier.json`
  *without re-timing*: it joins `zoo_chain_sweep.json` run medians
  (`gen_us` = delivered, `compiled_us` = torch.compile arm) against
  `zoo_kcount.json` `device_share`; `ceiling = 1/(1-F)`,
  `captured = (speedup-1)/(ceiling-1)` (`frontier.py:125-163`).
- `zoo_chain.py` measures each zoo workload through the shipped gen
  path unchanged: `TorchSource.to_ir` → `_mint_tiers` (all
  `genkernel.TIERS` site collectors, incl. `scan`) →
  `gk.profitable` (measured-cost referee) → `gen_sink` → `gen_probe`
  (deliver, verify, time vs eager and `torch.compile`).
- git dates: `zoo_chain_sweep.json` last touched `999daec`
  (2026-10-26), `zoo_kcount.json` `8669bfb` (2026-10-26),
  `frontier.json` `468e495` (2026-10-26).  The scan tiers:
  `89ea185`, `2d13865`, `df2923c` — all 2026-11-13.  `agenda.jsonl`
  mined `9849982` 2026-11-13 01:04.
- `docs/reports/delivery.json`/`pay_measured.json` are also
  2026-10-26 vintage — same staleness applies to any frontier-adjacent
  signal mined from them.

## Live re-measurement (this session, fp32, zoo(64,32,8), CUDA, reps=50, one round each)

| case | seq/seqm sites minted | kept by `profitable` | Triton bound | eager µs | compile µs | delivered µs | speedup | vs_compile | verified |
|---|---|---|---|---|---|---|---|---|---|
| DeltaRuleBlock | 1 `seqm` | `scan_4` + `gen_1` | yes (`_gsm`) | 490.4 | 302.8 | 387.2 | **1.266** | 0.782 | yes (max_abs 0) |
| GatedDeltaRule | 1 `seqm` | `scan_8` + `gen_6` | yes (`_gsm`) | 423.9 | 180.4 | 554.0 | **0.765** | 0.326 | yes (4.8e-7) |
| DecoupledMLA | 0 | **none — `gen_handlers` empty** | n/a | 385.3 | 286.8 | 342.2 | **1.126** | 0.838 | yes (2.4e-7) |

Site-mint check (CPU, `play._seq_scan_sites`/`_seqm_scan_sites` on the
intake IR): DeltaRuleBlock → 1 seqm spec
(`applyd(aff_diag(1, mul(mul(Q1,Q2), unsqueeze(sub(Q3, squeeze(matmul(Q4,H)))))), "H", {kind:"seqm"})`);
GatedDeltaRule → 1 seqm spec (same skeleton + scalar gate `Q1` on `a`
and in-residual gate on the read); DecoupledMLA → **0 sites of either
form** (it is an MLA attention block — latent down/up projections,
decoupled rope, manual softmax — not a recurrence); RealGatedLRU → 1
`seq` site as a control.

## Per-row verdict

### T-033 zoo:DeltaRuleBlock — SAME opportunity, already priced AND routed; row is stale

The seqm tier mints exactly the site p37 predicted
(`docs/reports/p37_seqm_matrix_scan.md`: "seqm site minted … 12 spines,
all `deindex-a` under old grammar").  `profitable` keeps `scan_4` —
the generated `_gsm` Triton kernel binds through `scan_triton_bindings`
(`TRITON_REFEREE_LOG` outcome `bound`, tier `scan`) — and the delivered
module verifies equal.  The frontier row's `delivered_us=307.6,
speedup=1.006` predates the tier; live the same path delivers
`speedup=1.266`.  **A paying candidate exists and is being
delivered/routed.**  Residual gap is `vs_compile=0.78`: torch.compile
still wins because it fuses the three input projections *and* the
4-step recurrence into far fewer launches, while delivery pays one
launch per minted kernel plus the seqm kernel's own cost.  Note the
zoo fixes `steps=4` for this workload (`zoo.py:2435` —
`_DeltaRuleBlock(d, d, 4)`, feed `(b,4,d)`); the seqm margin grows
with T (p37: 0.50× at T=4 → 0.036× at T=64 synthetic), so at T=4 the
kernel's edge is smallest exactly where compile is strongest.

### T-034 zoo:GatedDeltaRule — candidate exists and is routed, but whole-module delivery LOSES; genuine residual

Same seqm site minted and kept (`scan_8`), kernel bound, delivered
verified — yet live `gen_us=554µs` vs eager 424µs (**0.765×**, worse
than the stale fp64 row's 0.951) and `vs_compile=0.326` (compile hits
180µs).  So for this row the answer is *both*: the priced opportunity
is delivered, AND a real delivery-quality gap remains — the assembled
module loses to eager outright.  `profitable` prices each claim
against its spelled denotation in isolation (fold-vs-kernel, p37:
346.7→173.1µs at fp64 16×8×2); it does not price the delivered module
as a whole, so a kept claim can still lose end-to-end when launch
overhead, operand materialisation, or interaction with the module's
other kernels dominates — and at `dk=dv=64, T=4, fp32` the `_gsm`
kernel is at its `dk ≤ 64` register-tile edge.  This is the honest
residual the agenda row points at, but the lever is delivery/module-
level costing (or a `torch.compile`-class fusion arm), not "route the
paying kernel" — it already is.

### T-032 zoo:DecoupledMLA — DIFFERENT gap: no paying candidate exists; not a scan workload

Zero seq/seqm sites mint (verified above).  After `profitable`,
`gen_handlers` is **empty** — every consulted claim (`gen_*`, `red_0`
softmax, `synop_bb8c54c201`) was declined by the measured-cost
referee, so delivery ships the plainly-lowered IR.  Live fp32:
1.126× eager / 0.838× compile, verified — better than the stale fp64
row (0.907/0.624) but still losing to Inductor.  The frontier
"headroom" is pure dispatch share (`F=0.28`, ceiling 1.39×) that no
minted claim addresses: capturing it needs a fused-attention/MLA
kernel tier or better generic lowering — work no existing collector
touches.  Dtype sensitivity is also real: the partial fp32 frontier
(`frontier_fp32.json`, 5 rows) shows this same case at 1.438×
`captured=1.543` — the fp64-era row flipped sign with dtype, per the
T-021 fp32/fp64 divergence documented in `zoo_chain.py`'s docstring.

## Bottom line for the agenda rows

- **T-033, T-034**: the "uncaptured headroom" *is* the seq/seqm
  opportunity already priced — and it is already routed: `profitable`
  keeps the claim, `scan_triton_bindings` binds the kernel, `gen_probe`
  delivers+verifies it.  What the stale rows actually show is that
  **the sweep inputs were never re-run after the tiers landed**
  (2026-10-26 artifacts vs 2026-11-13 tiers).  Action: re-run
  `zoo_chain --only DeltaRuleBlock,GatedDeltaRule --isolate` (fp32,
  3+ rounds, idle box) → regenerate `zoo_chain_sweep*.json` →
  `frontier` — the rows will move on their own.  T-034 additionally
  keeps a genuine module-level delivery gap (0.33× vs compile) worth
  a *separate* proposal aimed at assembled-module costing, not claim
  routing.
- **T-032**: unrelated to the scan tiers — no recurrence, no minted
  claim, delivery is generic lowering.  The headroom is dispatch
  share vs a `torch.compile` that fuses the whole attention block;
  closing it needs a new tier (fused attention / fused-MLA kernel) or
  lowering work.  Re-scoping the proposal target is warranted.

## Caveats

- Live numbers are single-round `gen_probe` (reps=50) on a shared
  4 GB box — directionally solid (verified deliveries, bound kernels
  in `TRITON_REFEREE_LOG`, kept-handler tables) but not sweep-grade;
  the committed sweep uses 3–5 rounds.
- `row["claims"]` in `gen_probe` output is the *board's* offered
  claim tags, not the kept-handler set — the authoritative kept set
  is `case[5]["gen_handlers"]` post-`profitable` (used above).
- `frontier_fp32.json`'s `inputs` block still names
  `zoo_chain_sweep.json`/`zoo_kcount.json` though its rows differ —
  it was generated against a different sweep payload; treat its 5
  rows as a partial fp32 sample, not the canonical frontier.

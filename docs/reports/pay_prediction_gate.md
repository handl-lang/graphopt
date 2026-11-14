# Pay-prediction gate — route budget by structure, not uniformly

**Date:** 2026-11-15 · **Code:** `mlopt_discovery.pay_gate` (new),
`play.py` (`--pay-gate`, `zoo_pay_verdict`, `_zoo_export`,
`gen_probe`/`zoo_probe` `pay_gate` params), `bench/drivers/probes/
zoo_chain.py` (`--pay-gate`) · **Tests:** `tests/test_discovery_pay_gate.py`
(22 tests, `pay_gate.py` at 100% coverage).

## 1. Where the budget was spent — and where the gate sits

The per-workload spend has three layers, all previously **uniform**
— every zoo workload paid all of them:

1. **Intake mint + referee** — `play._zoo_case` runs `_mint_tiers`
   (every codegen tier's site collector) then `gk.profitable`
   (`reps=20` per minted claim, Triton binding attempts + spelled
   timings).  This is the first real cost and it runs *before* any
   search.
2. **Board episode** — `gen_probe` → `_gen_board` →
   `_claim_cost_gate`/`_fold_cost_gate` (a second `profitable`
   consult) → claim/saturate/extract on the meta-arena.
3. **Delivery + timing** — lower, verify, `_time_module` vs eager
   *and* `torch.compile`, plus the CUDA-graph arm.  `zoo_chain`'s
   `--budget` (default 150 s/workload) exists precisely because some
   `torch.compile` calls never return.

The loops that spend it: `play.zoo_probe`/`_zoo_rows` (`--zoo`),
`bench/drivers/probes/zoo_chain.py::measure` (the committed sweeps),
and `play.gen_probe` over the synthetic `_gen_cases` corpus
(`--probe`).  The meta-arena corpus drivers (`mp.gen_cases`, the
`overnight`/`agenda` miners) select *claims*, not workloads — the
agenda miner already reads `pay_measured`/`frontier` JSONs; no
per-workload budget loop exists there to gate.

**The gate's decision point** is therefore *before* `_zoo_case`'s
mint: export the module to IR (build + `TorchSource.to_ir` —
milliseconds-to-seconds, no mint, no referee) and census the root
term.  `_zoo_export` was factored out of `_zoo_case` for exactly
this; the expensive half is untouched.

## 2. The predictor — `mlopt_discovery.pay_gate`

Torch-free, microseconds: an op census of the IR term plus derived
flags (`term_features`), then a three-way verdict (`predict`):

- **`pay`** — *measured* scan carriers only: `select ≥ 8` feeding
  `stack ≥ 1` (the `seqt` trajectory signature) or `select ≥ 8` with
  no stack feed, no einsum read-out and no cumsum (the `seqm`
  matrix-state signature).  Evidence: `seqt_readout.md` (GLA 1.65→2.06×,
  Mamba 1.61→6.29×, HeadedSelectiveScan 1.59→5.38×,
  BidirectionalScan 2.58→2.37×), `resweep_scan_route.md` /
  `module_loss_gla_mamba.md` (DeltaRule/GatedDelta seqm pay).
  `ChunkedSSD` is deliberately excluded (select=15 but einsum=7,
  cumsum=1 — chunked algebra, measured loss).
- **`skip`** — the elementwise-bound signature, ALL of: `n_ops ≤ 24`,
  `compute_ops ≤ 5` (GEMM/conv/attention family), zero
  reduction/route/quant ops, zero conv, `linear < 3` (deep spelled
  stacks exempt — the LoRA/deep-equilibrium payers), not the
  dequant-matmul shape (`matmul==1` + transpose), and no
  einsum/select/stack/cumsum.  On the measured zoo this is exactly
  the "Inductor already fuses it" class.
- **`unknown`** — everything else → **measure, exactly as today**.

The asymmetry is deliberate: a wrong `unknown` costs budget (status
quo); a wrong `skip` loses a payer.  The skip rule is strict.

## 3. Validation on the zoo — honest confusion

Fit/evaluated against the committed measurements:
`zoo_chain_sweep.json` (fp64) + `zoo_chain_sweep_fp32.json` (fp32),
67 workloads measured in both dtypes, label = median speedup vs
eager ≥ 1.02 (the suite's win tolerance); `union-pay` = paid in
*either* dtype (12 modules flip sign between dtypes — a backend
kernel fact, `zoo_chain.md`).

| verdict | n | outcome |
|---|---|---|
| `skip` | 9 | **8 true losers, 1 false-skip** |
| `pay` | 4 (+4 unmeasured-in-sweep scan workloads) | DeltaRuleBlock + GatedDeltaRule confirmed payers; MambaSelectiveScan + GatedLinearAttention read as losses in the *stale* committed sweeps (they predate the seqt carrier — `seqt_readout.md` measures them paying 2–6× under current machinery) |
| `unknown` | 54 | measured exactly as today |

Skipped set: FiLMHead, GEGLUBlock, HighwayGate, GRUFusedGate,
LearnedPosSlice, SoftcapLogitHead, SigmoidAttention,
SquaredReluFFN — all measured losers in **both** dtypes — plus
**DynamicTanhBlock**, the single false-skip (fp32 1.129×, fp64
1.019× — a borderline fp32-only payer the census cannot distinguish
from HighwayGate; documented, not hidden).

**Precision/recall, stated plainly:**

- skip precision = 8/9 ≈ **0.89**; skip recall = 8/37 union-losers
  ≈ **0.22** — the gate removes ~13% of the zoo's workloads, ~89%
  of them correctly.
- **Zero both-dtype payers skipped** (0/17); one fp32-only payer
  (DynamicTanhBlock) skipped.
- The `pay` class is *not* a pay predictor for the sweep table —
  it marks carrier structure with dedicated measured pay; it only
  prioritises, it never skips.

This is a heuristic lookup fit on the same data it is evaluated
on — in-sample numbers, reported as such.  It is not claimed to
generalise; `unknown` is the safe default everywhere else.

## 4. Wiring — optional, default off, scheduling only

- `play --pay-gate` — applies to `--probe` (per-case
  `gen_probe(pay_gate=True)`, early-returns a declined row before
  `_gen_board`) and `--zoo` (`zoo_probe(pay_gate=True)`, skipped
  rows recorded with the verdict).
- `zoo_chain --pay-gate` (also propagated under `--isolate`) —
  `play.zoo_pay_verdict(wl, dtype=...)` before `_measure_one`;
  skipped rows carry `"pay_gate": {class, reasons, features}`.
- `gen_probe(..., pay_gate=True)` / `zoo_probe(..., pay_gate=True)`
  for programmatic callers.
- Export failure → `None` verdict → the workload proceeds exactly
  as today (fail-open toward measuring, never toward skipping).

Verification is untouched: the gate decides *whether* a workload is
measured, never what a measurement means.  `pay_gate=False`
everywhere reproduces the previous behaviour bit-for-bit.

## 5. Whole-chain pointwise fusion — scoped, one probe run

Question: is there a measured case where a multi-op pointwise chain
fuses *more* profitably than Inductor?  Probe (this host, RTX 2050,
median of 200 reps, `(256,512)` fp32/fp64): a 10-op pointwise chain
(`x*relu(x) + sigmoid(x) - tanh(x)*|x| + x² - softplus(x) +
tanh(x²)*softplus(x*sigmoid(x))`):

| dtype | unfused eager | `torch.compile` | ratio |
|---|---|---|---|
| fp32 | 231.2 µs | 28.0 µs | **8.3×** |
| fp64 | 1333.5 µs | 618.6 µs | **2.2×** |

Inductor already collapses the whole chain to ~one kernel; the
unfused spelling is 2–8× slower *before* any search runs.  The
`gen_*` Triton kernels measured against the compiled spelled body
lost 1.1–1.7× (`attr_elementwise_probe*.json`, `pay_measured.md`).
**Answer: no measured case shows a profitable whole-chain pointwise
fusion carrier — Inductor is already optimal there.**  The paying
boundary is non-elementwise structure (scan carriers, conv gates,
quant-deploy spellings, spelled-norm reductions), which is exactly
what the gate's `pay`/`unknown` classes protect.

## 6. Caveats

- The select≥8 threshold was fit at `zoo(d=64,t=32,b=8)`; at the
  default `t=8` the scan workloads still census ≥12 selects, but a
  shorter unroll (`t<8`) could drop below it → `unknown` → measured
  anyway (fail-safe direction).
- `pay`/`skip` are workload-level verdicts on the *spelled* IR;
  per-claim keep-votes remain the honest arbiter inside a measured
  workload.
- Dtype is not an input: 12 zoo modules flip pay sign between fp32
  and fp64; the gate cannot and does not try to predict that —
  `DynamicTanhBlock` is the documented casualty.

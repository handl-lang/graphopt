# catopt — measured results

Generated from the pinned baselines in `bench/baselines/` —
regenerate with `python -m bench results --out docs/results.md`.
Each entry states what the suite asked, what it expected, what
it measured, and the machine it measured on.  The full
per-suite reports (timings, plots, raw metrics) are produced by
`python -m bench report <baseline.json>`.

## Certified bounded rewrites

*Do error-budget rewrites buy wall-time on a real checkpoint?*

**Expected** — WIN — error-budget rewrites deliver bounded members and a measured wall-time win vs Inductor on stories15M (see docs/results.md).

| verdict | finding | headline |
|---|---|---|
| **WIN** | bounded rewrites (error_budget) buy wall-time on a real checkpoint | 4/5 budgets delivered a bounded member; best 1.322× vs eager |
| **WIN** | every budget's optimized module stays within the fp32 gate | 5/5 budgets optimized cleanly |

Measured 2026-10-02T01:00:43+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `b63355d` (dirty)) — baseline [`bench/baselines/bounded_e2e.json`](../bench/baselines/bounded_e2e.json).

## MiniGPT end-to-end

*Does composition (pairing + fold) beat plain Inductor end-to-end?*

**Expected** — PARITY — pairing fires per block, verified fp64-exact; wall time is ~parity.

| verdict | finding | headline |
|---|---|---|
| **WIN** | the composed catopt+Inductor module beats plain Inductor end-to-end | best 1.03× vs Inductor on tiny_llama@4x256 |
| **WIN** | the optimized module verifies against eager | 5/5 cells verified |

Measured 2026-10-02T00:47:33+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `f1e5d1a` (dirty)) — baseline [`bench/baselines/e2e_model.json`](../bench/baselines/e2e_model.json).

## Compositional pairing demo

*Does per-model lowering autotune pick the measured-fastest variant?*

**Expected** — WIN — the reported pick is the measured winner, never a static choice.

| verdict | finding | headline |
|---|---|---|
| **WIN** | the Autotuned lowering beats eager on at least one model | best 2.55× vs eager on matrix_chain |
| **WIN** | every model's optimized lowering verifies | 4/4 models verified |

Measured 2026-10-02T00:19:09+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `ba38bd9` (dirty)) — baseline [`bench/baselines/killer_demo.json`](../bench/baselines/killer_demo.json).

## Rewrite-law effect bench

*Does each registered rewrite law fire, get picked by extraction, and lower to a verified term?*

**Expected** — WIN — every registered law fires/picks/verifies; non-firing laws report honestly.

| verdict | finding | headline |
|---|---|---|
| **WIN** | every registered law fires, is picked by extraction, and lowers to a verified term | 11 verified / 11 picked / 13 fired across 13 cells |

Measured 2026-10-02T00:18:07+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `ba38bd9` (dirty)) — baseline [`bench/baselines/law_bench.json`](../bench/baselines/law_bench.json).

## Morphism-window composition

*Do morphism windows convert term-flops into wall time?*

**Expected** — WIN — term-FLOP reduction converts to measured wall time at GEMM-bound sizes.

| verdict | finding | headline |
|---|---|---|
| **WIN** | morphism windows convert term-level flops headroom into measured wall time | best 14.83× vs eager on chain_x4@4096x128 |
| **WIN** | every morphism rewrite verifies equivalent | 14/14 cells verified |

Measured 2026-10-02T00:40:26+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `f1e5d1a` (dirty)) — baseline [`bench/baselines/morphism_e2e.json`](../bench/baselines/morphism_e2e.json).

## Weight-chain reassociation vs Inductor

*Can the e-graph find a form Inductor's post-grad graph cannot express?*

**Expected** — WIN — the e-graph reaches a weights-first form Inductor's post-grad graph cannot express (see docs/results.md).

| verdict | finding | headline |
|---|---|---|
| **WIN** | catopt reaches a weight-folded form Inductor's post-grad graph cannot express | 17.12× vs Inductor at (k,d,B·T)=(16,512,4096) (18.63× vs eager) |
| **WIN** | every timed cell passes the fp32 equivalence gate | 12/12 cells verified |

Measured 2026-10-02T00:22:06+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `ba38bd9` (dirty)) — baseline [`bench/baselines/reassoc_scale.json`](../bench/baselines/reassoc_scale.json).

## Tooling-level measurements (hand-maintained)

This file is appended verbatim to `docs/results.md` by
`bench results` / `render_results_doc` — edit it here, never in the
generated file.  These measurements live in `tools/` and are
recorded in the retros, not in `bench/baselines/`.

All measured on the dev box (RTX 2050 / i5-12500H, fp64 — same
hardware class as the baselines); magnitudes do not extrapolate to
datacenter hardware:

| measurement | result | tool | retro |
|---|---|---|---|
| `softmax_fold` on `ManualSoftmaxAttention` | **+13–27% eager, +29–48% CUDA-graph** vs raw | `tools/law_wallclock.py` | [law-wallclock-verification](../project/retros/law-wallclock-verification.md) |
| `select_mul` marginal (optimized vs law-ablated) | **faster in 19/20 measurements** | `tools/law_wallclock.py` | [law-wallclock-verification](../project/retros/law-wallclock-verification.md) |
| executor routing after measured pricing | **12/12 cases ship the measured-fastest member** (uncorrected picks ran 2–3× slower) | `tools/executor_cost_probe.py` | [executor-pricing-fix](../project/retros/executor-pricing-fix.md) |
| `silu_fold` on the bench case | **−53–55% modeled cost, 1.43–2.62× wall-clock** | `bench run law_bench` | [three-cell-mediator](../project/retros/three-cell-mediator.md) |
| contraction player, n=40 vs `opt_einsum` | **0.89–1.01× of randomised greedy at equal wall-clock** (0.91–0.98 at T≤1, three seeds, fed budgets); **0.51–0.90× deterministic greedy** | `tools/contraction_guided_restart.py` | [contraction-synthesis](../project/retros/contraction-synthesis.md), [contraction-player-shipped](../project/retros/contraction-player-shipped.md) |
| pipeline held-out rediscovery | **winner re-ranks #1, SHIP, every run** | `catopt_discovery.pipeline --holdout` | [law-proposer-extensions](../project/retros/law-proposer-extensions.md), [groupnorm-convnext-corpus](../project/retros/groupnorm-convnext-corpus.md) |
| coherence catalogue over `ALL_RULES` (82) | **47 axioms / 13 lemmas / 2 redundant; 9 no-instance; divergence 1** (`rms_norm_fold` × `_nogain` — measured, unresolved) | `catopt_discovery.coherence --emit-basis` | [axiom-lemma-split](../project/retros/axiom-lemma-split.md) |
| evidence store, second run | **50/50 verdicts cached; 128 s → 6.6 s (~19×)** | `catopt_discovery.pipeline --evidence-db` | [evidence-store](../project/retros/evidence-store.md) |
| `chainfuse` — laws + `claim(gen_0)`, Triton-bound | synthetic composite (`x@A@B@C` + 6-op pointwise tail, 256×512): **2.94× vs `torch.compile`** (86µs vs 253µs), verified 1.7e-05; laws alone reach 181µs — the minted kernel is required for the full win | `python -m catopt_discovery.play --domain gen --probe` | [gen-kernels](../project/retros/gen-kernels.md) |
| delivered module vs `torch.compile` (`gen` probe) | spelled matmul chain **3.96×** (22µs vs 87µs; Inductor keeps all three GEMMs even under `freezing`); swiglu **2.11×**; pointwise fusion **0.82–0.88× loss** | `python -m catopt_discovery.play --domain gen --probe` | [gen-kernels](../project/retros/gen-kernels.md) |
| held-out zoo probe | **11/18 delivered verified modules beat `torch.compile`** — LoRAAdapter 2.18×, WaveNetGate 1.51×, DeepEquilibrium 1.35×; every winner ran `claims=[]` (the `gk.profitable` measured referee filtered unprofitable minted claims) | `python -m catopt_discovery.play --zoo` | [gen-kernels](../project/retros/gen-kernels.md) |
| joint fwd+bwd step vs `torch.autograd` | **+13%** (68µs vs 78µs, verified 9.5e-07) — one shared memo across forward + derived VJP; unshared costs 197µs | `catopt_discovery.training.joint_step` | [joint-training-graph](../project/retros/joint-training-graph.md) |
| machine law pack | **16/16 store objects deployable** as an opt-in `RuleSet`; machine-only arm wins 3/22 zoo models the human library misses (`linear_chain_t` owns LoRAAdapter's −20%) | `catopt_discovery.machine_pack` | [human-bar](../project/retros/human-bar.md) |

Two reconciliation notes:

- **The contraction "~1.8×" claim is stale.**  Early numbers had
  the learned policy at 1.54–1.78× of `opt_einsum`'s randomised
  greedy at n=40; the RL curriculum regime narrowed the gap to
  ~1.04×, and the oe-all **distilled** player that now ships as
  `catopt_torch.load_contraction_policy()`'s default *beats* the
  teacher at equal wall-clock (the figure in the table above).  The
  curriculum-RL weights remain bundled as an alternate artifact.
- **The `bounded_e2e` baseline's headline is vs eager** (best
  1.322×, above).  Its per-cell `speedup_vs_inductor` spans
  0.94–1.25 (1.16–1.25 over the budgets that delivered a bounded
  member) — so the README's "1.15–1.32× vs Inductor" is not the
  pinned baseline's literal column: the 1.32 is vs eager and the
  vs-Inductor delivered range tops out at 1.25 on this baseline.

## Derived structure on a real export (2026-10-15)

*Does the machine derive a real architecture's recurrence, and does
it pay?*  Not a `bench/` suite — an ad-hoc probe run on the dev box
(RTX 2050, fp64); scripts in `tools/`.

**Expected** — the e-graph reaches the chunked recurrence and
extraction prefers it; delivery is measured, not asserted.

| verdict | finding | headline |
|---|---|---|
| **WIN** | the derived chunked carrier is preferred by extraction | 29.4M vs 39.1M FLOPs (T=512) |
| **WIN** | end-to-end, delivered + verified vs the fused baseline | **1.82–2.33×** (866–1113 µs vs 2018–2028 µs), max_abs ≈1.3e-8 |
| **WIN** (symbolic) | K-way chunk construct | **4.4×** fewer FLOPs (29.4M → 6.70M); fold depth 1 → 7 |
| **WIN** (symbolic) | shared-state reassociation | **10.8–24.5%** fewer FLOPs (T=512, K=8: 8.57M → 7.64M) |
| **PARTIAL** | the *folded* carrier is never cheaper than the spelled form | a fold is a cost-neutral relabelling — the gap is search, not pricing |

**Honest scope.**  The 1.82–2.33× pays through the decayed fused
kernel plus an elementwise generated kernel; the scan kernels are
declined as unloweable at these sites.  The K-way and shared-state
figures are FLOP counts of extracted programs, not wall time — the
timing probe for the K-way extraction is in flight.  All numbers
are fp64-verified against the source module.

**`gen` probe, re-run 2026-10-15 (CUDA):** 10/10 cases delivered
and verified; 9/10 faster than eager; all faster than
`torch.compile` — `chainfuse` 3.71×/2.90×, `swiglu_mlp`
1.17×/2.00×, `chain_mlp` 1.06×/2.52×, `mulrelu` 1.22×/1.39×.

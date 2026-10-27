# The integration — index of the measured programme

Four suites, one question: **is the substrate + tensor + kernel
combination worth running, and where exactly does it pay?** Each suite
re-reads committed measurements (it fabricates no timings) and writes a
JSON + Markdown twin under `docs/reports/`.

Run all four:

```bash
PYTHONPATH=../kernelopt/src .venv/bin/python -m bench run-all \
  --suites frontier,delivery,referee,vocabulary
```

## What is measured

| suite | report | the question | headline |
|---|---|---|---|
| `frontier` | [`frontier.md`](frontier.md) | what is the theoretical ceiling on a measured row, how much does delivery capture — and does it beat the *free* competitor? | compute-bound rows cap at **1.03×** (28 of 65); launch-bound rows carry **2.23× median, 5.95× max** (25 of 65); **capture −0.03** over the 35 rows with real headroom; **13/25 launch-bound rows beat `torch.compile`** (median **1.11×**, best 3.60×); **65/65 verified** |
| `delivery` | [`delivery.md`](delivery.md) | do the same minted laws lose as a step and win as a kernel? | **3/3** paired ops lose as a step (worst **2.66×** slower), **3/3** win as a kernel (**2.72×** median), **3/3 flip** — the delivery sets the sign |
| `referee` | [`referee.md`](referee.md) | do the tensor and kernel banks share a vocabulary? | Jaccard **0.17** over 121 ops; **28/49** kernel ops (**57%**) have no tensor spelling; only **21** ops are shared |
| `vocabulary` | [`vocabulary.md`](vocabulary.md) | is originality a search result or a vocabulary event? | fixed-bank **0/39** machine-original across 14 runs; vocabulary mint **260/262** across 8 runs |

## Live session — 26 Oct (P1 + P2 + P3 legs run)

**P1, bounded.** Five of the 46 losing rows with the most kernels, re-measured at module scale (`d64 t32 b8`, isolated, fp32, **1 round**):

| case | fp64 (sweep) | fp32 (live) | flip |
|---|---|---|---|
| `DecoupledMLA` | 0.931× | **1.438×** | ✅ |
| `ChunkedSSD` | 0.956× | **1.101×** | ✅ |
| `RWKVMixBlock` | 0.959× | **1.029×** | ✅ |
| `SharedSwiGLUMoE` | 0.884× | 0.813× | ❌ |
| `TopKRouterMoE` | 0.911× | 0.710× | ❌ |

**3 of 5 flip ≥1.02×** — the P1 gate was 1/3, so it passes *on this sample*. It is the **dtype contract**, not the kernel-body bridge, that moved them; all five verified equal (`max_abs ≤ 4.8e-7`).

**P2, proof gate — run.** `kern_fuse_silu`, `kern_fuse_softmax`, `kern_fuse_layernorm`: **3/3 derivable** from the 374-rule union, each with a prior-art witness (`pa_kf_*_expand`), review queue empty. The translation rows, not a shared vocabulary, are what make the union real.

**P3, module scale.** `frontier` re-run on the live sweep (`frontier_fp32.md`): 5/5 verified, **capture 0.80** over the 2 rows with material headroom, **r(F, speedup) = +0.41**, compute-bound cap 1.04×.

**P5, industry lane — correctness passed, speed claim withdrawn.**
vLLM 0.32.0 serves the mlopt-optimized checkpoint **token-for-token
identically** to eager in every configuration tested
(`token_agree = 1.000`). The **+8% decode** first reported did **not**
survive a controlled re-measurement: each arm is ~15% faster when it
runs *first*, and in a paired 12-round A/B (both engines resident,
orders alternated) mlopt led by 5.0% on the median but lost 5 of 12
rounds with ~10% within-arm sd — inside the noise floor of this 4 GB
card. Report: [`p5_vllm_paired.md`](p5_vllm_paired.md); raw first run
[`p5_vllm.md`](p5_vllm.md).

Torch-side (vLLM out of the loop, 6-layer stories15M, ctx 64, fp32):
eager 370.9 tok/s, mlopt 367.9, `torch.compile` 418.1,
mlopt+compile 396.2 — **parity at this size**, as expected for a
6-layer model.

**P5b — the delivery path, tested.** We replaced vLLM's hand-written
CUDA `silu_and_mul` with a **Triton kernel of ours** inside a live Llama,
toggling between them per round in one engine: **token-identical** and
**speed-parity** (ctx 64 +1.8%, ctx 256 −0.2%, 30 rounds each). Parity is
the ceiling for a one-op swap — our wins come from cross-op fusion.
Report: [`p5_vllm_kernel_bridge.md`](p5_vllm_kernel_bridge.md).

**P5c — the fusion ceiling: 1.29%.** Before building a fused
`silu+mul+down` kernel we measured what it could possibly return. The
activation costs 7.5 µs and a launch costs 9.5 µs, so on a 3,490 µs
token the whole op is **1.29%** — a fusion can return at most that,
realistically nothing. Report:
[`p5_vllm_fusion_ceiling.md`](p5_vllm_fusion_ceiling.md). vLLM's Llama
graph is already fused; our wins live in *exposed* graphs.

**P5 — token-identical, speed-neutral.** With locked clocks and 30
alternating rounds the paired difference is **−0.19%** (ctx 64) and
**+0.94%** (ctx 256): no measurable speed effect. Correctness holds at
**1.000 token agreement** in every configuration. Reports:
[`p5_vllm_paired.md`](p5_vllm_paired.md) (final),
[`p5_vllm_paired_locked1500_30r.json`](p5_vllm_paired_locked1500_30r.json),
raw first run [`p5_vllm.md`](p5_vllm.md). `vllm` 0.32.0 at
`~/vllm-probe`.

## The competitor that matters: `torch.compile`. Across the 65 zoo
rows, the delivered form beats eager on 22 — but beats **`torch.compile`
on 31**, 13 of them in the launch-bound regime where the median is
**1.11×** and the best is **3.60×** (`PerChannelQuantMatmul`), with
`GroupedQuantLinear` **3.56×**, `TernaryWeightLinear` **2.25×**,
`FNOBlock` **1.82×**. In the compute-bound regime the median is 0.99× —
parity, as the ceiling predicts. Full table:
[`frontier.md`](frontier.md).

## P6 — the win, shipped as a callable op

`GroupedQuantLinear`'s zoo win is now a real operator:
`torch.library.custom_op("mlopt::grouped_quant_linear")` — one Triton
kernel that dequantizes per-group weights **in registers** and never
materializes the `N×K` float weight. Callable from plain PyTorch,
`torch.compile`, and vLLM's `CustomOp`.

| d | `torch.compile` | ours (fp32-exact) | vs compile | max abs error |
|---|---|---|---|---|
| 64 | 203.6 µs | 117.0 µs | **1.74×** | 5.7e-06 |
| 128 | 198.6 | 118.4 | **1.68×** | 1.9e-05 |
| 256 | 151.9 | 99.1 | **1.53×** | 3.8e-05 |
| 768 | 176.2 | 200.1 | **0.88× (loss)** | 1.2e-04 |
| 2048 | 819.0 | 593.5 | 1.38× | 3.1e-04 |

Beats `torch.compile` **1.5–1.7× while fp32-exact** at small/medium
widths; loses honestly at 768; the 3.0× at 2048 is TF32, not fp32, and is
reported separately. Report:
[`p6_grouped_quant_op.md`](p6_grouped_quant_op.md).

**P6b — inside vLLM.** All 24 linears of the model were W4-quantized and
driven by that op, toggled against the torch dequant path in one engine:
**token-identical output**, speed **inconclusive** (±3.9 % with 46–54 %
within-arm sd — the 1500 MHz clock lock did not hold and the card
throttled at 73 °C). Report: [`p6_vllm_w4.md`](p6_vllm_w4.md).

## The gates

| gate | status | what decides it |
|---|---|---|
| **P0** instrument | **done** | the four reports above, green, `ruff` clean, 17 tests |
| **P1** delivery bridge | **passing on a bounded sample** — 3/5 flips ≥1.02× (gate 1/3); the other 41 rows and the kernel-body (not just dtype) lever are open | ≥1 in 3 of the **46 losing rows** flip; kill if <10% |
| **P2** shared referee | **done** — 3/3 shipped kernel laws derivable from the 374-rule union via denotation rows | a law minted in one domain, *paid* in the other (pay leg still open) |
| **P3** scale | **partial** — capture 0.80 on 2 material-headroom rows at module scale, 1 round; needs 3 rounds + the whole zoo | capture ≥0.5, median ≥1.2×, fp32 + CUDA graph, fp64 verify |
| **P4** new algorithms | **done (control)** | the 0/39-vs-260/262 split; the remaining run is held-out architectures |
| **P5** industry lane | **speed-neutral, token-exact** — paired difference −0.19% / +0.94% over 30 rounds; agreement 1.000 everywhere | correctness stands; no speed claim survives |

### Reproduce the live session

```bash
.venv/bin/python -m bench.drivers.probes.zoo_chain \
  --only 'RWKVMixBlock,DecoupledMLA,SharedSwiGLUMoE,TopKRouterMoE,ChunkedSSD' \
  --isolate --dtype fp32 --rounds 1 --out docs/reports/zoo_chain_probe_fp32.json
.venv/bin/python -m bench.drivers.probes.zoo_kcount --only '<same>' \
  --out docs/reports/zoo_kcount_probe.json
.venv/bin/python -m bench.suites.integration.frontier \
  --sweep docs/reports/zoo_chain_probe_fp32.json \
  --kcount docs/reports/zoo_kcount_probe.json --stem frontier_fp32

# P2 (the sieve lives in kernelopt; that repo's venv carries a stale gamekit)
cd ../kernelopt && PYTHONPATH=src:../graphopt/packages/mlopt-core/src:\
../graphopt/packages/mlopt-discovery/src:../graphopt/packages/mlopt-torch/src \
  ../graphopt/.venv/bin/python -m bench.drivers.screen.novelty_sieve \
  --names kern_fuse_silu,kern_fuse_softmax,kern_fuse_layernorm --no-measure \
  --json ../graphopt/docs/reports/referee_proof.json --out ../graphopt/docs/reports/referee_proof.md
```

## The one number to watch

`frontier`'s **capture**. Every other result is already strong; capture is
the difference between a paper and a product. It is **−0.03** today,
meaning delivery currently spends slightly more than the algebra saves,
while 2–6× of dispatch headroom sits unused on the rows that have it.

## What this programme does *not* claim

- not that whole dense models get faster from rewrites — they are capped
  near 1.0–1.05× by the equality argument, and `frontier` reports that as
  a **negative** on purpose;
- not that a player transfers across domains — `referee` shows the
  vocabularies are largely disjoint, which is the precondition for the
  transfer question, not an answer to it;
- not that originality pays — `vocabulary` measures originality only;
  payoff is the separate `bank_growth` result (+0.066 iff it pays).

# kernelopt × hand-Triton wiring — does a real Triton kernel beat the `kf_*` composites?

Date: 2026-11-14.  Hardware: RTX 2050 4 GB (shared), torch
2.14.0+cu130 / triton 3.8.0, fp32, CUDA.  Probe:
`bench/drivers/probes/kernelopt_triton_probe.py` (new file).  Builds
on `kernelopt_bridge_probe.md` (no pays for the `torch.compile`
composites) and `kernelopt_integration_design.md` Track A, whose
bottom line recommended revisiting "if hand-Triton epilogue kernels
get wired into `KernelSink` lowering (today they are a separate
bench arm, not the sink's codegen path)."  This is that measurement.

## What was wired

**Family:** the `_triton_mm` epilogue family in
`kernelopt:bench/limits/triton/__init__.py` — the tutorial-tile
tiled matmul with absorbed epilogues: `kernel_mm_bias`,
`kernel_mm_scale` (scale baked 0.125 = whisper's `1/√64`),
`kernel_mm_gelu`, `kernel_mm_scale_bias` (0.5).  Chosen over the
pointwise/row-reduction builders because the bridge probe showed the
pay question lives at the GEMM epilogues (`kf_mm_*`); the pointwise
fusions (`kf_silu` etc.) are exactly what Inductor already emits
well.

**The seam needed zero kernelopt source change.**  `KernelExecutor`
already takes a `bindings` override, and only `FUSED_KERNEL_OPS`
route through `torch.compile` — so the delivery path is: extract
with `KERNEL_RULES` + `extract_best(kernel_cost)` as usual, rewrite
`kf_mm_bias → triton_mm_bias` (etc.) in the term, lower with a
bindings table that swaps those names for the Triton callables.
Scale-bearing rewrites (`kf_mm_scale`, `kf_mm_sb`) only fire when
the scale leaf is a `Const` matching a baked builder — a live or
unmatched scale declines to the compiled composite and is counted.

**Honest-cost adaptations** (what a shipped kernel assumes anyway):

- `w` arrives as the evaluated `transpose(param)` — non-contiguous
  `(K,N)`, but a *static* subtree the executor caches once.  The
  wrapper packs it contiguous keyed by `data_ptr` — load-time weight
  packing, the cuBLAS precedent.
- Activations are reshaped `(...,K) → (∏,K)` and the output reshaped
  back — views, no copies; the kernel is the same 2-D tile.

## Results — whisper subterms, per site

Same modules/shapes as the bridge probe (whisper-base, T=512, fp32,
warmup 5 / reps 30, µs/call).  `sched` = kernelopt incumbent
(`torch.compile` composites); `triton` = same extracted term with
the mm ops rewired to hand Triton.

| subterm | fused ops in best term | rewired | eager | sched | triton | verdict |
|---|---|---|---|---|---|---|
| `o_proj` (`x+linear(x)`, 512→512) | `kf_mm_bias` ×1 | 1 | 146.7 | 177.8 | **124.8** | **pays** — 1.42× vs sched, 1.18× vs eager; ≈ties graphopt deliver (126.7) |
| `mlp_block` (LN→fc1→gelu→fc2→res) | `kf_mm_bias` ×2, `kf_layernorm`, `kf_addmul` | 2 | 931.3 | 1087.7 | **968.5** | **pays vs kernelopt path** (1.12×) but **loses to eager** (0.96×) and graphopt (905) |
| `attn_core` | `kf_sdpa` ×1 (atomic) | 0 | 738.2 | 550.8 | — | **infeasible for this family** — the whole-attention op is one fused unit; the 2-D mm builders can't decompose it, and `_triton_sdpa` expects the `(x,w)`-same-matrix spelling, not `(a, w_k, s, v)` |
| `ln_res` | `kf_layernorm`, `kf_addmul` | 0 | 63.3 | 85.2 | — | n/a — no mm site; pointwise/norm family not wired here |
| `conv_block` | — | 0 | 614.6 | lower fails (`conv1d` ∉ `kernel_ops`) | — | infeasible, unchanged from bridge |

All wired arms verified: `sink.verify` vs spelled ≤ 5.3e-7 max rel,
and vs the original module ≤ 4.4e-7.

*Run-to-run note:* absolute µs drift with the shared 4 GB card (a
second `o_proj` run read ~2–3× slower across all arms under
contention); the **ratios replicate** — sched/triton 1.42× then
1.48×, eager/triton 1.18× then 1.41×.

## Why the pay is narrow — the corpus already said so

`kernelopt:docs/results_kernel_check.md` §1 measured this exact
family on the 46-case bench corpus: at 4096×512 @ 512×512 the
hand tile **loses** to the Inductor composite (mm_bias 941 vs 763 µs;
mm_scale_bias 921 vs 851 µs) — "Inductor's mm path beats the hand
tile at this size."  The whisper sites that pay are the *small-M*
regime (M = B·T = 512 tokens, N = 512/2048): decode-shaped GEMMs
where cuBLAS/Inductor's tile heuristics are weakest and a fixed
128×64 tile with a fused epilogue wins a launch.  `o_proj` is the
clean case — one GEMM, bias absorbed, no surrounding pointwise
noise; `mlp_block` dilutes the win across LN + residual + an
unfused eager `gelu` (see below).

## Gaps the wiring exposed (not fixed — honest notes)

- **`gelu` survives as an eager pointwise.**  The extracted
  `mlp_block` term is `kf_mm_bias` ×2 with a *plain* `gelu` binding
  between them: `kern_fuse_mm_gelu` only matches `gelu(matmul)` —
  the bias-carrying site is `gelu(add(mm, b))`, which no law folds.
  A `gelu`-absorbing epilogue variant (or a `kf_mm_bias_gelu` law)
  is the obvious next arm; today the Triton GEMMs bracket an eager
  activation kernel.
- **`kf_sdpa` is atomic.**  Attention stays on the compiled
  composite (which already ties graphopt's SDPA carrier).  The
  `_triton_sdpa*` pipeline builders exist but speak `(x, w)` —
  wiring them needs a per-head 2-D adapter and a different operand
  order; not attempted.
- **Scale gates are Const-only.**  `kf_mm_scale`/`kf_mm_sb` decline
  on live scales; no whisper site produced a standalone scale op
  anyway (`attn_core` folded whole into `kf_sdpa`).

## Vendoring decision (gen_manifest precedent)

`gen_manifest.py`'s rule: *gate vs incumbent, vendor the winner.*
Applied here: the only paying site is `o_proj`'s `kf_mm_bias →
triton_mm_bias` (1.42× vs the kernelopt incumbent, ≈tie vs graphopt
delivery).  **Not vendored** — `lib-triton-qgemv`'s `ROUTES`/`kernel_for`
signature is AWQ-qgemv-specific (`qgemv(op, x, qw, qz, sc)`); a dense
fp32 mm+bias kernel doesn't fit that contract, and the kernel source
already ships in `kernelopt:bench/limits/triton/` as the registered
baseline arm.  If a paying regime justifies a home, the natural
artifact is a `KernelSink` triton-bindings option (the
`bindings=` seam this probe uses) plus a per-site route table —
small follow-up, not done here.

## Verdict

**One family wired, one honest pay.**  The hand-Triton mm-epilogue
family runs inside the real `KernelSink` delivery path (verify-first,
no source change) and beats the `torch.compile` composite incumbent
on the small-M GEMM sites — decisively on `o_proj` (1.42×), partially
on `mlp_block` (1.12× vs incumbent, still behind eager/graphopt).
At corpus sizes the same kernels lose to Inductor, so the win is a
*regime* result — small-M/decode-shaped projections — not a general
one.  Attention, norms and conv remain outside the family's reach.

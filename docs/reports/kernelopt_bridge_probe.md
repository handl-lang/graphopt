# kernelopt-bridge probe — can kernelopt-derived kernels beat graphopt delivery on whisper hot subterms?

Date: 2026-11-14.  Hardware: RTX 2050 4 GB, torch 2.14.1+cu130,
fp32, CUDA.  Probe:
`bench/drivers/probes/kernelopt_bridge_probe.py` (scratch driver,
uncommitted).  Sibling checkouts consumed read-only via `sys.path`
(`lib-kernel-ir/src`, `kernelopt/src`) — no installs, no repo
mutation, nothing committed.

## Question

kernelopt searches lib-kernel-ir terms / `mlopt_core.ir` terms for
fused GPU kernels (`KERNEL_RULES` → `KernelSink`, fused ops lowered
as `torch.compile` composites; hand-Triton baselines live separately
in `kernelopt:bench/limits/triton`).  graphopt delivers via
`Optimizer.search` + `TorchBackend` lower (carrier folds like
`sdpa`, executor routing, optional compile/CUDA-graph runners).
The two were never integrated.  Can the kernelopt path beat
current graphopt delivery on whisper encoder/decoder hot
subterms?

## The bridge turned out to be nearly free — the laws are the wall

Both repos pin eqsat `v0.1.0` and consume the **same**
`mlopt_core.ir` term IR, so no term mapping is needed: export once
through graphopt's `TorchSource`, feed `res.ir` to both sinks.  The
probe does exactly that, plus a ~50-line `_decompose` shim
(see below).  Integration *code* cost: hours, not weeks.  Integration
*semantic* cost is the finding.

## Subterms measured

Pattern-representative `nn.Module`s at whisper-base shapes
(d_model=512, ffn=2048, 8 heads × 64, T=512 tokens, conv frontend
80→512→512 k3 s1/s2), exported via `TorchSource`, searched with
`Optimizer(backend=TorchBackend())`, delivered with
`lower(verify=True)`.  kernelopt arm: same `res.ir` → `EGraph` +
all 28 `KERNEL_RULES` fired once at root → `extract_best(kernel_cost)`
→ `KernelSink().lower` → `sink.verify` **and** an external check
against the original module output (`vs_module`, all ≈1e-7 — the
shim is numerically correct).

Times are per-call µs (warmup 5 / reps 30, `torch.no_grad`).

| subterm | eager | graphopt deliver | kernelopt as-exported | kernelopt + decomp shim |
|---|---|---|---|---|
| `attn_core` — `softmax((q@kᵀ)·s)@v` | 738.6 | **549.8** (verified) | 739.8 — only `kern_fuse_mm_scale` fired; **loses 1.35×** | 551.0 — `kf_sdpa` fired; **ties 1.00×** |
| `mlp_block` — `x+fc2(gelu(fc1(ln(x))))` | 926.8 | **905.0** (verified) | infeasible — `layer_norm` not in `kernel_ops` | 1195.8 — ln+mm_bias laws fired; **loses 1.32×** |
| `conv_block` — conv1d→gelu→conv1d | 528.2 | 529.0 (verified) | infeasible — `conv1d` not in `kernel_ops` | infeasible — same |
| `o_proj` — `x+linear(x)` | 130.3 | **126.7** (verified) | 127.4 — no law fires (`linear` is atomic); ties | 158.7 — `kf_mm_bias` fired; **loses 1.25×** |
| `ln_res` — `x+layer_norm(x)` | 61.7 | **61.5** (verified) | infeasible — `layer_norm` not in `kernel_ops` | 84.1 — `kf_layernorm`+`kf_addmul`; **loses 1.37×** |

## Why

1. **Atomic vs decomposed spellings.** `torch.export` emits atomic
   `layer_norm`, `softmax`, `linear` ops.  The kernelopt bank
   consumes the *decomposed* forms (`c·rsqrt(mean(c²)+eps)`,
   `div(exp u, Σexp u)`, `matmul`+`add`) — the shapes the substrate
   law bank produces mid-rewrite, not what a frontend exports.  The
   `_decompose` shim (softmax→div/exp/sum, layer_norm→mean/pow/rsqrt
   chain with the `dim=(512,)` *shape* attr correctly mapped to the
   last-axis reduce, linear→`matmul(x, Wᵀ)`+add) makes the laws
   reachable — at the cost of a spelled program that is itself
   1.5–3× slower than the atomic ops (ln_res: 63→187 µs spelled).
   The fused composites recover part of that but never reach the
   atomic baseline: `kf_layernorm` is a `torch.compile` composite of
   ~8 eager ops, and native `layer_norm` is already one cuDNN kernel.

2. **graphopt already owns the attention win.** On `attn_core`,
   graphopt's carrier fold extracts `sdpa` (verified: the extracted
   term is `reshape(transpose(sdpa(...)))`) → torch SDPA backend —
   a real fused kernel.  kernelopt's best possible (`kf_sdpa`, a
   compiled mm·softmax·mm composite) only *ties* it, and only after
   the decomp shim.  As-exported it fires just `kern_fuse_mm_scale`
   and loses 1.35×.

3. **kernelopt's differentiated vocabulary doesn't cover whisper.**
   Its measured wins (per `kernelopt:bench/discovery/kernel.py` and
   AGENTS.md) are the decayed-scan/retnet carrier ops
   (`om_decayed`, `om_chunk_scan*`) and hand-Triton GEMM epilogues —
   none of which appear in whisper's gelu/LayerNorm/conv/attention
   subterm census (`docs/reports/decoder_law_coverage.md`).  The
   `kf_*` fused ops are Inductor-grade composites, not hand kernels;
   on a 4 GB card at these shapes they don't beat cuBLAS/cuDNN
   epilogues (`kf_mm_bias` loses 1.25× to plain `addmm`).

4. **conv is out of scope entirely** — `KernelSink.kernel_ops` has
   no `conv1d`/`conv2d` binding and no kernel law touches it.
   graphopt also just ties eager here (529 vs 528 µs).

## Verdict per subterm

- `attn_core` (softmax-attention fragment): **ties** — and only with
  the decomposition shim; graphopt's `sdpa` carrier already captures
  the same win natively.
- `mlp_block` (LN+GEMM+GELU+residual): **loses** (1.32×) even wired;
  infeasible as-exported.
- `conv_block` (conv frontend): **infeasible-to-wire** — no conv in
  the sink vocabulary.
- `o_proj` (residual projection): **loses** with shim (1.25×), ties
  without (no law reaches inside atomic `linear`).
- `ln_res` (norm+residual): **loses** (1.37×) with shim; infeasible
  as-exported.

## Bottom line

**No pays.**  The IR bridge is free (shared `mlopt_core.ir`), but the
law-bank spelling mismatch means a real integration needs a
decomposition front-end just to make the laws *fire* — and even then
every reachable subterm ties or loses to what graphopt already
delivers.  kernelopt's edge is the decayed-scan carrier vocabulary
(retnet-class models), not whisper's pointwise/norm/GEMM-epilogue
surface; `torch.compile`-grade `kf_*` composites don't beat cuDNN
kernels or graphopt's `sdpa`/carrier folds on this hardware.
Recommendation: don't integrate for whisper; revisit only if a
whisper-relevant carrier (e.g. chunked decode scan) lands in the
kernelopt bank, or if hand-Triton epilogue kernels get wired into
`KernelSink` lowering (today they are a separate bench arm, not the
sink's codegen path).

## Artifacts

- Probe: `bench/drivers/probes/kernelopt_bridge_probe.py`
  (scratch; imports `mlopt_kernel`/`lib_kernel_ir` from sibling
  checkouts via `sys.path`, runs entirely in graphopt's `.venv`).
- No repo changes outside this report and the probe file.

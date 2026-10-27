# P7 — real AWQ weights (Qwen2.5-0.5B-Instruct-AWQ)

The synthetic W4 test is now repeated on a **production quantized
checkpoint**: real `qweight` (int4-packed), `qzeros`, per-group `scales`
(group 128 along the input dim), unpacked once at load time — unpacking
is loading cost and is excluded from both arms.

The comparison is the honest one:

* **torch** — dequantize to a float weight, then `@` (what any portable
  path does: HF transformers, custom kernels, a `torch.compile`d model),
* **ours** — `mlopt::grouped_quant_linear`, dequant in registers, float
  weight never materialized, fp32 (IEEE) accumulate.

## Result

| layer | out × in | M | torch | ours | **vs torch** | max\|Δ\| |
|---|---|---|---|---|---|---|
| `q_proj` | 896 × 896 | 1 | 121.4 µs | 53.3 µs | **2.28×** | 1.2e-07 |
| `o_proj` | 896 × 896 | 1 | 121.1 | 53.4 | **2.27×** | 1.5e-08 |
| `down_proj` | 896 × 4864 | 1 | 596.2 | 207.5 | **2.87×** | 5.2e-08 |
| `gate_proj` | 4864 × 896 | 1 | 595.8 | 211.7 | **2.81×** | 2.2e-08 |
| `q_proj` | 896 × 896 | 32 | 141.7 | 133.8 | 1.06× | 3.8e-06 |
| `o_proj` | 896 × 896 | 32 | 141.6 | 134.2 | 1.06× | 2.1e-07 |
| `down_proj` | 896 × 4864 | 32 | 657.6 | 623.5 | 1.05× | 4.2e-07 |
| `gate_proj` | 4864 × 896 | 32 | 867.5 | 570.6 | **1.52×** | 3.7e-08 |

**2.27–2.87× on the decode path, exact to ~1e-7.** Batched (M=32) is
1.05–1.52×: the fusion still pays, but cuBLAS catches up because the
dequant traffic is amortized over more rows.

## The caveat that decides how much this is worth

The baseline here is **dequant-then-matmul**, not vLLM's `awq_gemm`.
vLLM's W4 kernel unpacks int4 *and* matmuls in one launch, so it does not
pay the float round-trip we remove — and our op takes already-unpacked
float weights, so it cannot compete with that kernel on its own turf.

What this result does say: **any stack that materializes a float weight
from a quantized checkpoint is leaving 2.3–2.9× on the decode path.** That
includes HF transformers, most research code, and every model that is
quantized but not served by a specialized W4 kernel.

## What it costs to be right

The first version of this kernel **lost 2.4×** on real shapes — a 32×32
tile wastes tensor rows at one token. A GEMV path for M=1 fixed it. The
numbers above are post-fix, and the pre-fix loss is kept in
`p6_grouped_quant_op.md` so the claim cannot drift.

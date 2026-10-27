# P18 — end-to-end, correct and faster: −29.2 % with identical output

The integration works. vLLM's AWQ path, with our int4 kernel substituted
on three of the four layer geometries, produces **token-identical**
output and runs **29 % faster**.

| | |
|---|---|
| **token agreement** | **identical** (no divergence over 25+ rounds) |
| stock vLLM | 1164.6 ms |
| ours | **824.1 ms** |
| **delta** | **−29.2 %** |
| **rounds won** | **25 / 25** |
| linear calls routed to us | 43.8 % (ours), 50 % stay on `awq_gemm`, 6.2 % prefill |

Per-layer agreement against `awq_gemm` on the same input: **max|Δ|
0.0005, cosine 1.0000** on `4864×896`, **0.0000** on `896×896`.

## The bug that hid this for two days

AWQ packs eight int4 values into an int32 in the order
`[0, 4, 1, 5, 2, 6, 3, 7]`, not `0..7`. My unpack used plain order — and
my *reference implementation* used the same wrong order, so every
standalone check I ran agreed with itself while the kernel computed a
different function from the model's. The tell was cosine similarity 0.19
against `awq_gemm` on identical input; the fix is one permutation
(`awq_packing.to_standard_order`).

**Lesson:** a self-written reference can only prove self-consistency.
Agreement must be checked against the *incumbent implementation*, on the
same input, before any timing is believed.

## How it is wired

1. `process_weights_after_loading` is wrapped to snapshot each layer's
   `qweight`/`qzeros`/`scales` **before** vLLM converts AWQ into its
   GPTQ-like layout and pads 896 → 1152.
2. `apply` routes by `(in, out)` shape to our kernel for
   `down_proj`, `q/o_proj`, `k/v_proj`; everything else falls through to
   vLLM's own path.
3. `VLLM_BATCH_INVARIANT=1` keeps vLLM on the plain AWQ kernel instead
   of Marlin, which repacks the weights into an opaque layout.

## Honest limits

- **Decode only** (one token per call). Prefill and batched steps fall
  back to `awq_gemm` — 6.2 % of calls here.
- **One model, one GPU, fp16, batch 1**, clocks drifting (~1327 of
  2100 MHz).
- **`gate_proj` is not routed** — we lose to `awq_gemm` there (0.95×),
  so 50 % of linear calls still run their kernel.
- **Memory cost**: the snapshot duplicates the quantized weights
  (~0.7 GB for this model).
- **Batch-invariant mode** is not the default serving path; the same
  substitution under Marlin is not implemented.

Within those limits the claim is strong: **a kernel nobody hand-tuned,
plugged into an industry serving stack, exact to the last token, 29 %
faster on a 4 GB laptop GPU.**

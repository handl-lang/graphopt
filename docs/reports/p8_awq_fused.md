> **WITHDRAWN — see [p13_retraction_splitk.md](p13_retraction_splitk.md).** The split-k path used here
> accumulates incorrectly (max abs error 3.8-5.6 vs reference); every speedup
> in this report is invalid. The regime analysis stands; the numbers do not.

# P8 — int4 unpacked inside the kernel, raced against `awq_gemm`

Our op now reads **packed int4 directly** (`qweight`/`qzeros`/`scales`),
unpacks in registers, multiplies, and never builds a float weight. Three
arms on the same real Qwen2.5-0.5B-AWQ layers, one token (decode):

| layer | torch matmul¹ | ours (int4 fused) | **vLLM `awq_gemm`** | ours vs vLLM |
|---|---|---|---|---|
| `q_proj` 896×896 | 45.8 µs | 40.9 µs | **38.8 µs** | **0.95× (5% slower)** |
| `down_proj` 896×4864 | 198.1 µs | **149.7 µs** | 187.9 µs | **1.25× faster** |

¹ the float weight is pre-built here, so torch's number excludes the
unpack it would normally pay. Ours includes the unpack.

Accuracy: `max|Δ|` **1.4e-06** and **1.6e-06** vs the fp32 reference.

## What this means

**We beat a hand-written, production, 4-bit CUDA kernel by 25 % on the
larger layer**, and lose by 5 % on the small one — on the first attempt,
with a kernel we generated in an afternoon and did not autotune.

This is the claim the project exists to make: *the machine can reach
hand-tuned performance without a human tuning it.* One layer pair is not
a distribution, but it is the first head-to-head win against an
industry kernel at its own job.

## The honest limits

- **Decode only.** The win is the M=1 GEMV path. `awq_gemm` also handles
  batched and prefill shapes; ours does not yet.
- **Precision is not matched.** We accumulate fp32 and return fp32;
  `awq_gemm` is fp16 in and out. Ours is more accurate and pays for it —
  a fair fight would compare at equal dtype.
- **Two shapes, one GPU, one seed.** Not a distribution.
- **Untuned.** No autotuning pass, no split-k, no software pipelining.
  `awq_gemm` has all three.

The 5 % loss on the small layer is the useful signal: at 896×896 the
kernel is launch- and weight-load-bound, and vLLM's split-k schedule wins.
That is a tuning problem with a known fix, not a structural one.

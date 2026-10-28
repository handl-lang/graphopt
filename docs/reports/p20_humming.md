# P20 — Humming exists, covers our exact format, and hits the same wall

**Humming** (Ant Group + vLLM, Oct 2026, Apache-2.0,
`github.com/vllm-project/humming`): a JIT-compiled CUDA GEMM library for
quantized inference. It is the closest thing in the wild to this project's
thesis — unified quantization description, per-workload configuration
selection, shared CUDA components, broad hardware coverage.

## Does it cover our case? Yes.

From the support matrix: activation **FP16 (e5m10), SM75+**, weights
**symmetric INT1-8** — i.e. exactly W4A16 group-scaled int4 decode. It
supports dense and MoE, CUDA only (requires `NVCC`), SM75+ NVIDIA.

## Does it beat Marlin? Conditionally — not in our regime.

- The blog claims "significantly outperforms vLLM Marlin, FlashInfer and
  DeepGEMM across **many representative workloads**", with up to 5×
  kernel speedups and 14-43 % e2e MoE gains.
- Those wins are **compute-bound / large-batch / high-concurrency MoE**,
  and several compare **W4A16 Marlin against W4A8 Humming** — a format
  change, not a same-format multiplier.

## In our regime it reproduces our number exactly

[Issue #72](https://github.com/vllm-project/humming/issues/72), M=1
single-request decode on SM80, dense `n8192 k8192`, INT3:

| M | effective BW | % of peak (2039 GB/s) |
|---|---|---|
| 1 | 692 GB/s | **34.0 %** |
| 8 | 708 GB/s | 34.7 % |
| 16 | 709 GB/s | 34.7 % |

And on the reporter's 170HX, `gate_up` (K=5120, N=34816) at M=1:
**≈691 GB/s of ≈1591 GB/s = 43 %**, tile `(16,256,128)`, 5 stages.

Three things follow:

1. **Our 42 % of roofline on `down_proj` is not a beginner's number.** A
   funded team with a year of arch-specific tuning lands in the same band.
2. The comparison inside that issue is the important one: **plain BF16 on
   the same GPU reaches 60.4 %**, and FP8 e2e reaches "above 90 %". Low-bit
   *decode at M=1* is the regime where the whole field gives up bandwidth.
3. The repo's own `benchmarks/results/a800/` files show it — not a user bug.

## Where the door is, and it is open

The report itself notes that `tune/sm8x.py` last changed for *interface*
reasons only, while `sm90/sm120/sm121` heuristics got "multiple dedicated
small-M/small-batch performance passes". And the blog concedes:

> "it is difficult for **heuristics** to select the optimal configuration
> in every case. Humming therefore **plans** to introduce **autotune**."

So Humming is **heuristic-tuned, with search still on the roadmap**, and
its weakest heuristic is exactly the M=1 SM80 path we care about. That is
the gap this project is aiming at, and it is now documented by someone
else's issue tracker rather than by our own optimism.

## Consequences

- Differentiator cannot be "unified quant representation + JIT + autotune".
  Humming has two of three and has announced the third.
- It *can* be: **search over the regime heuristics abandon** (M=1, odd
  shapes, stale arches), and non-CUDA/NVCC backends.
- Benchmark against **Humming**, not only Marlin, from M1 onward.
- Strongest available citation that the regime is unsolved.

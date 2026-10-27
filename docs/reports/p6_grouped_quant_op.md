# P6 — the win, shipped as a drop-in op

`GroupedQuantLinear` beat `torch.compile` by 3.56× in the zoo sweep. This
turns that win into something a stack can actually call: **one Triton
kernel behind `torch.library.custom_op("mlopt::grouped_quant_linear")`**.

The module, verbatim from the zoo:

```python
w = w_q * scale.unsqueeze(-1)          # per-group dequant, (N, K)
return x @ w.transpose(0, 1) + bias    # linear
```

Eager materializes an `N×K` float weight, then calls `addmm`. Our kernel
**never materializes it** — the dequant happens in registers inside the
matmul loop — so the saving grows with the weight.

## Measured (32 tokens, this box)

Two variants, because it matters:

* `tf32` — Triton's default dot precision (fast, 10-bit mantissa)
* `ieee` — full fp32 accumulate (a real drop-in)

| d | eager | `torch.compile` | ours (tf32) | ours (ieee) | **ieee vs compile** | ieee max\|Δ\| |
|---|---|---|---|---|---|---|
| 64 | 119.3 µs | 203.6 µs | 131.2 | 117.0 | **1.74×** | 5.7e-06 |
| 128 | 123.4 | 198.6 | 120.7 | 118.4 | **1.68×** | 1.9e-05 |
| 256 | 103.5 | 151.9 | 102.1 | 99.1 | **1.53×** | 3.8e-05 |
| 768 | 135.6 | 176.2 | 121.2 | 200.1 | **0.88× (loss)** | 1.2e-04 |
| 2048 | 630.5 | 819.0 | 271.8 | 593.5 | 1.38× | 3.1e-04 |

## What this says

- **At small/medium widths (64–256) the fused op beats `torch.compile` by
  1.5–1.7× while staying fp32-exact** (max abs error ≤ 3.8e-05). That is
  a genuine, shippable win: the dequant never hits memory.
- **At d=768 it loses** (0.88×) — honest regression; a single 32×32 tile
  set is a bad matmul for that shape, and cuBLAS/Inductor win.
- **The 3.0× at d=2048 is TF32**, not fp32. It is a real speed number but
  not a drop-in one; the fp32 version is 1.38×. Both are reported so the
  claim cannot drift.

**Caveat on absolutes:** eager read 28 µs at d=64 in an earlier run of the
same script and 119 µs here — the box's clocks move between sessions. Only
within-run ratios are comparable.

## Why this is the deliverable

`torch.library.custom_op` is callable from **any** stack: plain PyTorch,
`torch.compile` (it appears as an opaque op and survives), and vLLM's
`CustomOp` registry. That is the transfer path the vLLM experiment showed
we lacked: not "hand them weights", but "hand them an op".

**Next:** register this as a vLLM `CustomOp` and swap it into a quantized
linear layer of a real model — the same manoeuvre as the SwiGLU test, but
this time the op is one where we measurably win.

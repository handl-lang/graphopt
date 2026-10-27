# P5b — our kernel, running **inside** vLLM

**Question:** can a kernel we generate live in an industry serving stack,
or does our advantage only exist in our own runtime?

**Answer: it installs and runs, token-identical, at parity with vLLM's
own hand-written kernel.**

## Method

vLLM's `SiluAndMul` (the SwiGLU activation in every Llama MLP) calls a
hand-written CUDA op, `torch.ops._C.silu_and_mul`. We replaced its
`forward_cuda` with a Triton kernel — `silu(x[:d]) * x[d:]` in one
launch, index math instead of copies — and **toggled between the two
per round inside one engine**, so model, weights, memory and scheduler
are identical; only the kernel changes.

Kernel verified against the spelled reference: `max|Δ| = 1.43e-06`.

## Result — stories15M, RTX 2050, greedy, 30 alternating rounds

| ctx | vLLM CUDA kernel | our Triton kernel | delta | ours wins | tokens |
|---|---|---|---|---|---|
| 64 | 111.7 ms ±8.7 | 113.7 ms ±12.3 | +1.8% | 11/30 | **identical** |
| 256 | 118.3 ms ±10.7 | 118.1 ms ±11.0 | −0.2% | 18/30 | **identical** |

**Parity.** Our kernel is not faster than vLLM's — but it is
indistinguishable from it, and it is running in their server, in the
hot path of every layer.

## Why parity is the right result here

A one-operator swap cannot beat an already-optimal operator. vLLM's
`silu_and_mul` is a single fused CUDA kernel doing exactly what ours
does. There is no work to remove.

Our measured wins (2.6× kernel, 2.9× chain) come from **fusing across
operations**, not from beating one op. A single-op substitution is the
one test where parity is the ceiling.

## What this opens

- **The delivery path exists.** A generated Triton kernel can be
  injected into vLLM's model without forking it, and it preserves output
  exactly. That was the open question; it is now closed.
- **The next experiment writes itself:** substitute at *chain* level —
  fuse `silu_and_mul` + the down-projection epilogue, or the residual +
  norm + quant chain — where our 2.6–2.9× came from. A one-op swap
  cannot show it by construction.
- **Caveats:** fp32, 6-layer model, clocks drifted (1327 MHz observed,
  not the 1500 lock — the paired design cancels drift but absolute
  numbers are not comparable across sessions).

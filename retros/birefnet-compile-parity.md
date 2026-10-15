# BiRefNet compile-parity probe — verdict

Question: the 34607e0 trial's delivered module timed 0.90× vs eager /
0.80× vs `torch.compile` on a reconstructed Swin stage under a
FLOP-neutral extraction.  Is the loss *lowering quality* (plan-tape
interpreter overhead) rather than semantics — i.e. does
`torch.compile(delivered)` recover `torch.compile(source)`?

Probe: `lab/probes/birefnet_parity_probe.py` (2026-10-14, main checkout,
RTX 2050 4 GB, torch 2.14.0+cu130, `TORCHINDUCTOR_COMPILE_THREADS=2`).

Block: **real timm `SwinTransformerBlock` ×2** (timm 1.0.30), shaped
like BiRefNet encoder stage 1 — dim 192, heads 6, window 7,
alternating shift 0/3, input `(1, 56, 56, 192)` fp32, 892K params.

## Arm A — `--rules none` (FLOP-neutral; the regime under test)

No rewrites; delivered = the source term re-lowered through the
plan tape.  `sink.verify` max_abs = **0.0** (bitwise identical —
same ops, same order).  Two runs:

| arm                    | run 1 µs | run 2 µs | peak MB |
|------------------------|----------|----------|---------|
| source eager           | 5845.5   | 5837.8   | 107.4   |
| source compile         | 4982.5   | 4961.8   | 109.7   |
| delivered eager        | 5882.9   | 5881.1   | 174.1   |
| delivered compile      | 5000.5   | 5007.8   | 109.7   |

compiled(delivered) / compiled(source) = **1.004× / 1.009× →
PARITY**.  The tape overhead (~0.7% eager) is erased entirely by
inductor; peak memory identical.

## Arm B — `--rules default` (the composed pipeline set)

Only `sdpa_fold_add_drop` fires (timm's *manual* attention —
q·kᵀ·scale + bias + softmax ·v — folds to `sdpa`; note real
BiRefNet already calls `F.sdpa`, so this fold is reconstruction-
specific).  Extracted cost 0.406× of source; verify max_abs
9.5e-07.

| arm                    | µs      | peak MB |
|------------------------|---------|---------|
| source eager           | 5824.3  | 107.4   |
| source compile         | 4981.2  | 109.7   |
| delivered eager        | 5455.3  | 154.2   |
| delivered compile      | 4699.4  | 108.3   |

compiled(delivered) / compiled(source) = **0.943×** — the delivered
program is *faster* than compiled source: the FLOP win survives
compilation.

## Verdict

**Yes — compiling the delivered module recovers compiled parity.**
In the FLOP-neutral regime `compile(delivered) ≈ compile(source)`
(1.00–1.01×, identical peak memory): the earlier 0.90×/0.80× loss
was pure interpreter overhead, not lowering quality.  With the
standard rules the delivered module compiles to a *faster* program
here (sdpa fold, 0.94×).  Delivery via the tape is not a compiled-
deployment tax — but eager deployments of a FLOP-neutral
extraction still pay it (0.99×; worse in the 34607e0 trial's larger
reconstruction, 0.90×).

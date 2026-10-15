# Retro: RetNet carrier vs external baselines

Date: session of 2026-10-14.  Probe: `lab/probes/retnet_baseline_probe.py`
(committed `6eca509`).  Dev box: RTX 2050-class GPU, fp64 unless noted.

## Why this exists

The README's "7.30x less memory" row was measured against *our own*
spelled-quadratic `_ChunkedRetention` reference — not against the
RetNet paper's implementation or tuned third-party kernels.  This
probe re-measures the delivered K-way carrier against external
baselines.  Verdict below is honest: the carrier is the paper's
algorithm, and the hand-tuned world still beats our artifact.

## Arms

- `spelled-eager` / `spelled-compiled` — the optimize-input model.
- `carrier` (fp64) / `carrier32` (fp32) — extracted + lowered K-way
  carrier.
- `paper-eqn` — faithful in-script reimplementation of Sun et al.'s
  chunkwise recurrence, gamma matched (0.9), no stability rescale.
- `paper-code` — official `torchscale` `chunk_recurrent_forward`
  driven directly (`inner_mask` inlined verbatim from
  `RetNetRelPos.forward`, decay patched to log(0.9)).
- `fla-chunk` / `fla-rec` — fla-org `chunk_retention` /
  `fused_recurrent_retention`, fp32 (Triton).

## Numbers (T=2048, K=8, C=256)

    spelled-eager     169.92 MiB   21.73 ms   1.00x
    spelled-compiled   73.90 MiB   27.01 ms   2.30x
    carrier            30.01 MiB    3.14 ms   5.66x
    paper-eqn          26.91 MiB    2.12 ms   6.32x
    paper-code         27.44 MiB    3.00 ms   6.19x
    carrier32          19.45 MiB    2.38 ms   8.73x
    fla-chunk           9.96 MiB    0.28 ms  17.05x
    fla-rec            10.15 MiB    1.16 ms  16.74x

T=512 / T=1024 same ordering; carrier loses at 512, crosses ~1024.

## Numerics

- carrier vs spelled-eager: 9.8e-9 fp64 — the delivery is correct.
- carrier vs paper-eqn: 9.8e-9 fp64 — **the extracted program IS the
  paper's chunkwise algorithm**, not an approximation of it.
- carrier vs paper-code: ~0.5 — the official impl applies an
  `all_scale` row-normalization (fp16 stability) that is NOT an exact
  identity; its output differs from raw retention by construction.
  Worth knowing if anyone compares against torchscale outputs.

## What this means for the claims

1. "Beats its own input" — yes, decisively (5.66x memory, 6.9x speed
   at T=2048; 2.5x memory and ~8.6x speed vs the input under
   torch.compile — compile fixes fusion, not asymptotics).
2. "Beats the paper's chunkwise impl" — no: ~1.1x more memory, ~1.5x
   slower than a plain hand-written chunked loop.
3. "Beats tuned kernels" — no: FLA is ~3x less memory and ~11x
   faster.  That gap is *lowering quality* — our artifact is a
   general lowering of the extracted term (many small ops,
   materialized decay vectors), not a fused scan kernel.
4. "A new transformation" — no: it is the paper's published duality,
   machine-rederived and machine-verified.

The defensible claim stands at: certified mechanical reachability of
a published linearization + delivery as a verified runnable module.
Closing the FLA gap is a *lowering/kernel* problem (genkernel/fusion),
not a semantics problem — a separate dimension by ADR 0003.

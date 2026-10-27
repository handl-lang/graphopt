# Plan — from a hand-tuned kernel to a substrate-native search

**Date:** 2026-10-27
**Follows:** `2026-10-27-automatic-tuning-margin.md`
**State:** a hand-written int4 GEMV beats `awq_gemm` 3.32× on
`down_proj`, runs at 44 % of the measured 89 GB/s roofline, and does not
make the model faster end-to-end.

## The four things blocking a real result

1. **Kernel efficiency** — 8× redundant packed-word loads, 32-bit loads
   where int4 wants 128-bit, fp32 output where fp16 halves the traffic,
   split-k atomics adding traffic back. Roofline says 24.6 µs; we run
   55.9 µs.
2. **Model-level dilution** — fp32↔fp16 conversion per call plus only two
   of four geometries routed to us. End-to-end is +1.0 % (flat).
3. **No vocabulary for quantized ops** — the minting loop cannot propose
   a packed-int4 kernel because "load nibble / dequantize" is not a thing
   it can say.
4. **No template space** — the loop tunes configs, not structure
   (GEMV vs tiled vs split-k), because structure lives in my head.

## Phases and gates

| phase | work | gate | kill |
|---|---|---|---|
| **1. Close the gap by hand** | load-once/unpack-8, 128-bit loads, fp16 in/out with fp32 accumulate, split-k only where K is large | `down_proj` < 35 µs (≥70 % of roofline), `q_proj` < 20 µs, exact to 1e-3 (fp16) | hand-tuning cannot reach 70 % of roofline → the whole thread is dead |
| **2. End-to-end, dtype-clean** | rerun the routed AWQ test with the fp16 kernel | model ≥ 3 % faster, paired, locked clocks, tokens identical | flat → the product is a fast-op library for unattended shapes, not serving |
| **3. Make it searchable** | int4 as vocabulary; template space (GEMV / tiled / split-k × load width × tiles × dtype) scored by the calibrated roofline | the machine **rediscovers the phase-1 kernel without being told its structure** | the search cannot beat a trivial baseline → the loop is not yet a kernel designer |
| **4. Coverage** | sweep shapes, models, formats; emit the routing table automatically | a routing table produced with no hand-written entries | — |

## Methodology (non-negotiable)

- **Paired measurement** in one process, alternating order each round —
  sequential comparison produced three different answers (+8 %, +6.5 %,
  −1.8 %) for the same two models.
- **Locked clocks, reported spread.** Median + sd + win count. No
  best-of, no warmup asymmetry.
- **Exactness before speed.** `max|Δ|` against the reference, measured
  before any timer runs; a precision change is reported as such, never
  folded into a speed claim.
- **Roofline first.** Bytes ÷ measured bandwidth before building, and
  every result reported as a fraction of that ceiling.
- **Kill criteria decided before running**, as in the table above.
- **Configs by sweep, not taste** — and record which config won.
- **Negatives kept next to wins.** Corrections appended, never silently
  edited.

## Architecture requirements (the part the prototype fails)

The current kernel is a **prototype that proves the physics**, not a
component. To be worth keeping it must become:

- **substrate-native** — expressed as ops the kernel domain already
  knows (`KERNEL_RULES`, `KernelSink`, the cost model), so the mint,
  the sieve and the certificate apply to it;
- **composable** — a template instantiated by parameters, not one
  monolithic kernel per shape;
- **decoupled** — no monkeypatched vLLM internals in the shipped path;
  an op plus a routing policy that any host can call;
- **reusable and scalable** — one search harness over many shapes and
  formats, with the routing table emitted, not typed.

Phase 3 is where the prototype becomes a component. Anything shipped
before that is a demo and should be labelled as one.

## Risks

- **Phase 2 flat.** Cheapest thing to learn, do it first.
- **Referee noise at scale.** ±50 % drift was observed on this box;
  a search over thousands of candidates measured that way produces
  confident nonsense. Lock clocks, repeat, and gate on the referee
  before trusting the search.
- **Overfitting to one GPU.** Every number here is a 4 GB laptop card;
  the roofline model must be calibrated per machine
  (`scripts/calibrate_profile.py`) before any routing table ships.

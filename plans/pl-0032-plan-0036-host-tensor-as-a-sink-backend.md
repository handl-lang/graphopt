+++
id = "PL-032"
title = "Plan 0036 — host-tensor as a `Sink` backend"
status = "open"
priority = "med"
+++

---
title: 'Plan 0036 — host-tensor as a `Sink` backend'
type: plan
date: 2026-11-19
---

# Plan 0036 — host-tensor as a `Sink` backend

Origin: architecture review. host-tensor/lib-tensor implement the
same ~204-op ATen surface as the torch bridge, generated from the
same schemas and verified bit-exact under `verify.hnl` — but they are
a *parallel* stack: verified, shipped, and unused by the optimizer.
This plan makes them a delivery backend for the same IR.

## Phases

**Phase 0 — op coverage map.** Diff the mlopt IR vocabulary against
the host-tensor port list (204 ops). Output: the liftable subset
(elementwise/reduce/linalg first), the honest-decline set, and the
wire-format codec requirements.

**Phase 1 — `HostTensorSink`.** Implement `mlopt_core.ports.Sink`:
lower IR ops → host ops (generated bindings), decline the rest
loudly. Reuses the `TensorValue` ↔ host wire codec verify.hnl
already crosses.

**Phase 2 — verify through the existing pipeline.** Candidate runs
on host-tensor; compared against the source oracle (torch eager for
torch-sourced models) with the same dtype-aware tolerances and
seeded-RNG discipline. No second verify path.

**Phase 3 — measure + deliver.** Time vs eager/compile on real
subterms; deliverable = `.hnl` artifact or Rust-callable module.
Honest negative is a valid result.

## Hard gates

- Every delivered candidate passes the same verify floor as any
  other sink — no relaxed tolerances for the new backend.
- Decline coverage is explicit (which ops, why) in the report.
- First milestone = elementwise/reduce/linalg subset only; matmul/
  conv quality is measured, not assumed.

## Dependencies / adjacency

- Plan 0037 supplies the `BackendRegistry` this sink registers into.
- `verify.hnl` (now 204 verdicts, generated) is the codec + parity
  evidence Phase 0 builds on.
- math.* leaves landed (203 clean / 4 stubs) — the stub set is the
  decline boundary.

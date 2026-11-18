+++
id = "PL-033"
title = "Plan 0037 — backend-neutral model layer: torch as one plugin"
status = "open"
priority = "high"
+++

---
title: 'Plan 0037 — backend-neutral model layer: torch as one plugin'
type: plan
date: 2026-11-19
---

# Plan 0037 — backend-neutral model layer: torch as one plugin

Origin: architecture review + user direction (2026-11 session):
"torch should be just one impl or integration — we should be
decoupled from pytorch." The substrate is already neutral (eqsat is
torch-free; ONNX source/sink proves the port; kernelopt proves a
second domain seats in). The coupling lives in the model-facing
layer: `model_corpus` assumes torch export, verify defaults to torch
eager, and there is no registry where backends advertise themselves.

## The `BackendRegistry`

Each backend registers `{format: Source, targets: [Sink...],
oracle: VerifyOracle, cost_profile}` — the same `register()` seam
kernelopt uses for domains, applied to model formats:

- **torch**: `torch.export` → IR (exists), `IRModule`/CUDA-graph/
  ONNX sinks (exist), eager oracle (exists)
- **ONNX**: `OnnxSource`/`OnnxSink` (exist), onnxruntime oracle
- **host-tensor/.hnl**: plan 0036 sink + numeric oracle
- future: safetensors+graph-descriptor, JAX, TVM — the registry is
  the extension point, not a rewrite

## Phases

**Phase 0 — oracle port.** Define `VerifyOracle` in
`mlopt_core.ports`: `run(term/env) -> TensorValue` + comparison
contract (dtype-aware tolerances, seeded RNG). Torch-eager becomes
one implementation; onnxruntime and host-tensor executors already
exist and become others.

**Phase 1 — source port generalization.** `model_corpus`/probes take
a `Source`, not a torch module. Census/mint/arena already operate on
IR terms — the change is the entry point, not the machinery.

**Phase 2 — registry + cost profiles.** `BackendRegistry` in
gamekit-or-discoverykit (plan 0034 decides home); `DevicePolicy`
gains cost-profile data so launch-aware costs calibrate per backend.

**Phase 3 — second non-torch model end-to-end.** An ONNX or .hnl
model through lift → search → verify → deliver, measured against
its own oracle. This is the proof the layer is generic — torch
becomes one row in the registry, not the center.

## Hard gates

- Zero behavior change for torch-sourced models (regression corpus).
- Every backend's oracle is independent — no verify path trusts the
  source backend's own numerics.
- Decline coverage per backend is explicit.

## Dependencies / adjacency

- Plan 0034 (discoverykit) gives the mining machinery a
  domain-free home; plan 0036 supplies the host-tensor backend;
  plan 0035 supplies the typed surfaces the registry returns.
- `torch_port_gen` (in flight) reduces torch's hand-tables so the
  torch plugin is thin data, not parallel implementation.

+++
id = "ADR-0006"
title = "ADR 0006 — ONNX as a second backend, in two directions"
status = "open"
priority = "medium"
+++

---
title: ONNX as a second backend, in two directions
type: adr
date: 2026-10-15
status: accepted
status_note: |-
  **Accepted (2026-10-15) — direction 1 shipped, and the boundary
  held.**  `packages/catopt-onnx` implements `Sink` over 169 IR ops
  (opset 18), verified through `onnxruntime` (219 tests; ~150
  programs round-tripped).  Pipeline round-trip:
  `Linear→ReLU→Linear→softmax` 1.49e-08 (fp32), 3× stacked `Linear`
  **1.11e-16** (fp64, machine epsilon), a tuple-returning two-head
  module < 1e-6 on both outputs.

  **The falsification did not fire**: `git diff --name-only` against
  the package list shows **zero** changes under `catopt-core` (or any
  other package) — the claim *"a new backend implements `Sink`;
  nothing in core changes"* held on the first non-torch backend.

  **What the experiment actually found** (the valuable part):

  - Two port surfaces are thinner than a second backend needs, both
    worked around sink-side rather than by widening core: `Source`
    carries **no element type** for inputs (`Var.typ` is shape-only,
    so an index input is untypable without a dtype channel), and
    `Capabilities.ops` is spelled `torch_bindings` (cosmetic — the
    pipeline never reads it).
  - `supported_ops` is a *name* set, but ONNX feasibility is
    sometimes **attribute-dependent** (a `transpose` needs a static
    rank); those refuse at lowering rather than at pricing time, so
    extraction can in principle commit to a form the sink then
    declines — no occurrence in the tests, but a real seam.
  - Two attribute-spelling traps, exactly as predicted: ONNX
    `HardSigmoid` defaults `alpha=0.2` where ATen is `1/6` (silent
    20% slope error), and `Flatten` is always 2-D.
  - fp64 is exact only where onnxruntime ships a double kernel
    (measured gaps: `Tan`, `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`,
    `Erf`, `Conv`, `InstanceNormalization`).
  - Only `Monolithic` is exercised: `Compositional` needs a
    `Composer`, `Autotuned` a `Meter` — both torch-shaped ports with
    no ONNX equivalent.  **That is the datum for direction 2.**

  **Open follow-ups:** a staged coverage climb for `catopt-onnx`
  (not yet in `[tool.coverage.run] source`); direction 2
  (`OnnxSource`) remains unbuilt.
---

# ADR 0006 — ONNX as a second backend, in two directions

## Decision


**Adopt ONNX as a backend, staged: export first, import second.**
Core does not change; ONNX lands as its own package behind the
existing ports, with the op-mapping work made explicit rather than
discovered late.

### Direction 1 — export (IR → ONNX) — the first stage

`OnnxSink` implements `Sink`: IR → an ONNX graph (via `onnx` /
`onnxruntime` for verification).  This is the *inverse* of what
`torch_bridge` already does — a table from the IR op set to ONNX
ops plus the attribute spelling each op uses.

Value: an optimized program can be delivered to any ONNX runtime
— the hardware-vendor compilers, mobile, non-torch serving — which
is where the *measured* wins could actually reach users.
Cost: moderate.  The op table is finite and the IR is already the
intermediate form; the risk is attribute spelling (see the
positional-attr trap below), not architecture.

### Direction 2 — import (ONNX → IR) — the second stage

`OnnxSource` implements `Source`: an ONNX graph → IR.  This is
what lets the optimizer touch models that are not torch at all.

Cost: **the large one**, for three reasons measured on the torch
side:

1. **The op-mapping long tail.**  The torch backend needed ~260
   bindings, an `ATTR_SCHEMA` for positional→named attributes, and
   per-op shape rules.  ONNX needs the same three artifacts for
   its op set — mechanical, but the long tail is where the bugs
   live: the U²-Net trial found `max_pool2d` silently truncating
   because `ceil_mode` arrives *positionally*, and
   `upsample_bilinear2d` placing `align_corners` at position 2 so
   the scale lands at 3 (committed `ecb2fb7`).
2. **Shape inference.**  The IR's typing is explicit; ONNX graphs
   carry less of it, and several laws (the carrier/`_vshape`
   family) are shape-guarded.  Without a shape source the
   interesting half of the library is unreachable.
3. **Opset drift.**  ONNX ops change semantics across opsets;
   the mapping must pin the opset it claims to support.

### Prerequisite already met: multi-output

ONNX graphs are natively multi-output, and until this week the
torch path silently delivered **one** output (`export_to_ir` took
the first resolvable leaf and `break`ed; measured on U²-Net: 7
outputs, 1 delivered).  The `pack` op + `IR.roots` +
per-output verification (branch `feat/multi-output`) close that —
a second backend would otherwise have inherited the same bug.

### The interim: refuse, do not drop

Until `OnnxSink` exists, the honest behaviour is what ships
today: an explicit `unsupported export format` error.  A silent
single-output delivery is worse than a refusal, and the same rule
applies to any ONNX op the mapping cannot express — decline the
model, never drop a node.


## Why

Every model the optimizer has touched entered through `TorchSource` (`torch.export` → IR) and left through `TorchSink` (IR → `IRModule`).  The hexagonal boundary was built for more than one backend — `AGENTS.md` states it outright: *"A new backend implements `Sink`; nothing in `catopt-core` changes."* — and `import-linter` enforces it (core imports no torch, numpy, or GPU library; adapters push state into core through registered seams).

## Not doing

- none recorded — legacy record; the design doc lists what was deferred.

## Consequences

- **The boundary is validated by use.**  The first non-torch
- **Verification semantics must be pinned.**  The equivalence
- **Costing stays backend-aware.**  `backend_cost(cost_fn,
- **The staged order is a commitment.**  Export-first is not
- **What this is not.**  Not a rewrite, not a new IR, not a

## Enforcement

- Code and tests are named in the design doc.

## Design

The full record (context, tables, phases, trade-offs): [`design/0006-onnx-backend.md`](../design/0006-onnx-backend.md).

---
title: ONNX as a second backend, in two directions — design detail
type: design
date: 2026-10-15
refs: [ADR-0006]
updated: 2026-10-29
---

# ONNX as a second backend, in two directions — design detail

Moved verbatim from ADR-0006 (ADR-0161 size budget). The ADR carries the rules;
this page carries the context, tables, phases and trade-offs.

## Context

Every model the optimizer has touched entered through
`TorchSource` (`torch.export` → IR) and left through `TorchSink`
(IR → `IRModule`).  The hexagonal boundary was built for more than
one backend — `AGENTS.md` states it outright: *"A new backend
implements `Sink`; nothing in `catopt-core` changes."* — and
`import-linter` enforces it (core imports no torch, numpy, or GPU
library; adapters push state into core through registered seams).

ONNX is the obvious second backend: it is the interchange format
every other framework speaks (TF, JAX, Paddle, TFLite, the
hardware vendors' compilers), and the IR is already
backend-neutral — `catopt_core` has no tensor library in it.

**Today ONNX is explicitly unsupported**, and pinned as such:
`tests/test_export.py::test_unknown_format_error` asserts
`export_optimized(..., fmt="onnx")` raises `unsupported export
format`.  That is the honest current state, not an accident.

The pressure this ADR answers: *is ONNX a second backend we
adopt, and in which direction first?*  The two directions have
very different costs and different value.

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

## Consequences

- **The boundary is validated by use.**  The first non-torch
  backend is the real test of ADR 0003's claim that
  semantics/search/evaluation/execution are independent: if ONNX
  needs core changes, the claim was overstated and this ADR
  records the falsification.
- **Verification semantics must be pinned.**  The equivalence
  oracle is numeric on real tensors; for ONNX the comparison runs
  through `onnxruntime`, so tolerance and dtype policy need
  stating, and the certificate's meaning (a replayable derivation
  over IR terms) stays backend-independent — that part does not
  move.
- **Costing stays backend-aware.**  `backend_cost(cost_fn,
  sink.supported_ops)` already prices only what a backend can
  lower; an ONNX backend narrows or widens that set, and the
  search follows automatically.
- **The staged order is a commitment.**  Export-first is not
  timidity — it is where the value is (delivery to other runtimes)
  at a fraction of the cost, and it produces the op table the
  import direction then reuses in reverse.
- **What this is not.**  Not a rewrite, not a new IR, not a
  second semantics.  The IR stays the interchange form; ONNX is a
  pair of adapters behind ports that already exist.

## Direction 2 — `OnnxSource` (implemented)

`OnnxSource` implements `Source`: `onnx.ModelProto` (or a `.onnx`
path) → `(IR, leaves)`, over `catopt_onnx.liftmap`'s `ONNX_LIFTS`
table — **107 op types**, the inverse vocabulary of the sink's
`ONNX_BINDINGS`, sharing `errors.py`'s refusal vocabulary
(`UnsupportedOnnxOpError` for op-level refusals, new
`OnnxImportError` for structural ones: opsets outside the pinned
13–21 window, non-tensor inputs/outputs, missing producers).

**What the direction actually costs.**  The ADR's three predicted
costs all showed up, two smaller than feared, one exactly as
predicted:

1. *The op-mapping long tail* — real but bounded: ONNX moves its op
   parameters from attributes to *tensor inputs* (opset 13+: `Slice`
   bounds, `Reduce*` axes, `Clip` limits, `Reshape` shapes), so the
   importer's real work is **attr folding** — initializer/`Constant`
   inputs become IR attrs under `ATTR_SCHEMA`'s canonical spellings,
   and a computed (non-constant) parameter input is a refusal.  That
   is the direction-2 form of "ambiguous attribute": the value is
   data, not metadata.
2. *Shape inference* — `onnx.shape_inference` runs once and types
   every leaf; `dim_param` dims land as the IR's `None` extent
   (representable), while a value with no tensor type at all is a
   refusal.
3. *Opset drift* — pinned to a window (13–21) the way the sink pins
   18; outside is refused, not guessed.

**The port gap, worked around source-side.**  `Source` returns
`(IR, leaves)` and `Var.typ` is shape-only — ONNX inputs *do*
declare element types, and the IR has no channel for them.  The
workaround is the sink's, mirrored: `OnnxSource.input_dtypes`
publishes the declared per-input dtypes after each import, and the
caller wires it back (`OnnxSink(input_dtypes=source.input_dtypes)`)
so an `int64` index input stays an index input.  Zero core changes —
the *falsification did not fire a second time*: `git diff` shows no
edits under `catopt-core`.

**Second new seam found.**  `Op` has no name field, so ONNX
node-name provenance survives only on `Var`/`Param` leaves (which
keep the graph's own value names) and in refusal messages.  Also,
`Source.to_ir(model, example_inputs)`'s second argument has no role
for ONNX — the graph declares its own shapes — it is accepted and
ignored so the `Optimizer` call shape stays uniform.

**Multi-output held both ways.**  Graph outputs become `IR.roots`
via `pack` exactly as the torch source mints them; a `Split`'s
outputs mint one `split(index=i)` projection per live output name.
Verified end-to-end: torch → IR → ONNX → IR → optimizer →
onnxruntime — `Linear→ReLU→Linear→softmax` verified `passed` (max_abs
< 1e-5 vs the original torch module), and a shipped law
(`silu_fold`) fires on the lifted term and wins extraction.

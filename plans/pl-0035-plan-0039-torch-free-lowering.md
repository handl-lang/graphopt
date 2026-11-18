+++
id = "PL-035"
title = "Plan 0039 — torch-free lowering: IR → native CUDA via kernelopt"
status = "open"
priority = "high"
+++

---
title: 'Plan 0039 — torch-free lowering: IR → native CUDA via kernelopt'
type: plan
date: 2026-11-20
---

# Plan 0039 — torch-free lowering: IR → native CUDA via kernelopt

Origin: user direction (2026-11 session): "why would we need torch?
if we can just lift the model and then lower to cuda and also, it
would use opt-kernel to generate the optimized code." Verification
is the safety envelope for aggressive search — the product is
**speed, safely**. Torch is today's *incumbent executor* (the
baseline to beat and the measurement harness), not the foundation.

This plan is the delivery-side complement to plan 0037
(backend-neutral model layer, which owns the lift/source side).
0037 makes *inputs* backend-neutral; 0039 makes *outputs*
backend-neutral — a verified optimized program that runs without a
torch runtime at all.

## What already exists (inventory, verified)

- **ONNX backend** (`mlopt-onnx`): `OnnxSink` lowers IR → checked
  ONNX artifact, `OnnxSource` lifts back; verified through
  onnxruntime end-to-end with no torch import.
- **host-tensor** (`handl-host-tensor`): Rust PyO3 extension,
  238 tensor ops, shared f64 wire codec, 137-op Python Sink,
  installable wheel (`handl_host_tensor` via maturin). `.hnl`
  export from mlopt emits runnable terms over this surface.
- **kernel-ir bridge** (kernelopt `mlopt_kernel.search`): mlopt
  term → `lib-kernel-ir` Term, split-k verified, loud declines.
  Coverage today: **14 of ~95 mlopt ops** denote.
- **lib-kernel-ir**: extended 10 → 21 ops (transpose/permute views
  + 9 pointwise) with honest costs/grades; **ADR-0171 pending
  ratification**; compile-triton refuses the new ops until its
  lowering extends.
- **Delivery game** (mlopt `deliver_game.py`): runner configs as
  moves, measured-ms referee under mandatory verify — the natural
  seat for "which lowering target wins per module."

## The gap

1. **Op coverage**: ~14/95 mlopt ops bridge to kernel-ir. Decoder
   blocks need attention (SDPA/matmul+bias+softmax), layernorm,
   activations (silu/gelu), residual adds, embedding/gather, rope.
   GEMM coverage exists but attention fusion patterns do not.
2. **Lowering patterns**: kernel-ir is a weight-first GEMV/dequant
   vocabulary — it has no attention-block or layernorm *templates*.
   Term-to-term translation can't express "this whole decoder
   layer becomes one fused kernel" without pattern-level lowering.
3. **Emit path**: no mlopt IR → CUDA-source/Cubin artifact exists.
   kernelopt's search produces kernel-ir Terms + measurements; the
   Triton/CUDA emit lives in `compile-triton` which currently
   covers only the original 10 ops.
4. **Verification without torch**: `VerifyOracle` port (plan 0037
   phase 0) must land first — a native lowered artifact needs an
   oracle that isn't torch eager. host-tensor's numeric executor
   or onnxruntime are the candidates.

## Phases

**Phase 0 — ratify + retag substrate.** ADR-0171 ratification,
lib-kernel-ir `v0.1.1` tag, kernelopt repin. Unblocks everything
downstream. (Blocked on user sign-off.)

**Phase 1 — coverage map.** Census which mlopt ops appear in real
lifted modules (whisper decoder step, kokoro decoder, t5 decoder —
the r4 corpus modules are the natural domain). Rank by frequency ×
difficulty. Output: `docs/reports/kernel_ir_coverage.md` — the
honest denominator for every later claim.

**Phase 2 — extend lib-kernel-ir vocabulary.** Add the missing op
classes the census demands (attention primitives: `sdpa`/`mha`
block op or matmul+softmax decomposition; `layernorm`; `silu`/`gelu`;
`embedding`/`gather`; `rope` as a view+op). Each with honest cost +
grade + denotation, ADR-documented. This is substrate growth —
expect multiple ADRs, not one.

**Phase 3 — lowering patterns.** Pattern→template lowering in
kernelopt: matched subgraphs (attention block, norm+linear pairs)
lower to fused kernel templates, not term-to-term translation.
Reuse `mlopt_kernel.scan` (`build_scan_plan`/`is_chunk_scan`) as
the precedent — it already does schedule-pattern detection.

**Phase 4 — native emit.** Extend `compile-triton` (or a sibling
emitter) to cover the new ops → CUDA/Triton source artifact.
The delivery game gains a `native` config arm; the referee measures
it against eager/compile/graph like any other move.

**Phase 5 — torch-free end-to-end proof.** A real model (whisper
decoder step first — smallest verified unit) through:
lift → search → verify(oracles incl. non-torch) → kernel-ir →
Triton/CUDA artifact → measured vs torch incumbent. Success =
verified faster *without torch in the loop at runtime*.

## Hard gates

- Every phase lands with tests + honest declines; no coverage
  claims without the Phase-1 census numbers.
- The torch-free artifact must verify against a **non-torch
  oracle** before any speed claim — self-consistency is not proof.
- `DEFAULT` law-bank changes stay behind the existing gauntlet;
  this plan adds *targets*, not laws.
- Perf claims require measured numbers vs the torch incumbent on
  the same hardware; declines are recorded, not hidden.

## Risks

- **Vocabulary creep**: kernel-ir could grow toward a second full
  tensor IR — the census (Phase 1) is the guard; extend only what
  real modules need.
- **Attention is hard**: fused SDPA in Triton is a research problem
  itself; Phase 2/3 may need to ship GEMM+softmax decomposition
  before true fusion — honest partial coverage is fine.
- **Oracle gap**: without plan 0037 phase 0, native artifacts can
  only be verified against torch — which reintroduces the
  dependency we're removing. Sequence matters: oracle port early.
- **RTX 2050 4GB test bed**: large-model native runs may OOM where
  torch manages — record honestly, don't tune around it.

## Relationship to other plans

- **0037** (backend-neutral model layer): owns Source/lift side +
  `VerifyOracle` port — Phase 0 dependency here.
- **0036** (host-tensor sink): the `.hnl`/host-tensor runtime is a
  *complementary* torch-free target — lighter than native CUDA,
  useful as the first non-torch delivery arm before kernel-ir
  matures.
- **0038** (taxonomy rename): `opt-kernel` naming affects how this
  plan's artifacts are described, not the technical content.

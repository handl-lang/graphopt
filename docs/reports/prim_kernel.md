# Does a real kernel make the prim folds pay? — yes, 2.6×

**Date:** 2026-10-25 · **Probe:** `bench/drivers/probes/prim_kernel_probe.py`
(new) · **Device:** CUDA (RTX 2050), **fp32**, 1 048 576 elements ·
**Op set:** the shipped pointwise prims (the ones a kernel codegen can
model)

```bash
.venv/bin/python -m bench.drivers.probes.prim_kernel_probe \
    --json docs/reports/prim_kernel.json
```

> **Verdict: the fold was never the problem — the interpreter was.**
> Six shipped prims get a generated `triton.jit` body; each is verified
> against the shipped binding on the same inputs, then timed. The kernel
> runs at **0.371–0.392× the binding's cost: 2.55–2.70× faster**, every
> op, no exceptions.

## The two arms

- **binding** — what ships today: `shipped_primitives._binding`, a Python
  spec-walk over the definitional premise's RHS with the ambient op table
  (`docs/reports/pay_measured.md` measured this as the loss).
- **kernel** — a generated Triton body for the *same* spec
  (`genkernel.triton_bindings`), broadcast + flattened host-side, one
  program per 1024 elements. A kernel that disagrees with the binding is
  refused before timing.

| op | binding ms | kernel ms | kernel / binding |
|---|---|---|---|
| `synop2_35bbd03301` | 1.696 | 0.664 | **0.391** |
| `synop2_d46808a970` | 1.695 | 0.665 | **0.392** |
| `synop3_04df5fd446` | 2.394 | 0.896 | **0.374** |
| `synop3_083e9c241f` | 1.832 | 0.708 | **0.386** |
| `synop3_0ca31bf1e5` | 2.395 | 0.898 | **0.375** |
| `synop3_0fa19d1ff8` | 2.534 | 0.940 | **0.371** |

**fp64 — the dtype the family is verified in** (`--dtype float64`): the
win holds, at the same ratio.

| op | binding ms | kernel ms | kernel / binding |
|---|---|---|---|
| `synop2_35bbd03301` | 3.368 | 1.295 | **0.384** |
| `synop2_d46808a970` | 3.366 | 1.294 | **0.384** |
| `synop3_04df5fd446` | 4.768 | 1.750 | **0.367** |
| `synop3_083e9c241f` | 3.647 | 1.384 | **0.379** |
| `synop3_0ca31bf1e5` | 4.768 | 1.751 | **0.367** |
| `synop3_0fa19d1ff8` | 5.048 | 1.841 | **0.365** |

So the fp32 and fp64 legs agree: **2.60–2.74× vs the spec-walk, at the
precision the certification uses.** The earlier caveat ("a fp64 kernel may
not keep the 2.6×") is retired — it does.

Every kernel passed the fp64-argument / fp32-run equality check against
the binding (`rtol 1e-3`), and the codegen *declined* every spec it does
not model (non-pointwise bodies such as the `linear`-bearing prims) —
those are reported as declined, not as losses.

## What this changes

`pay_measured.md` concluded "the family does not pay" and named the spec
walk as the cause. This closes the loop on that diagnosis:

- **a fold pays iff its body is a kernel** — 2.6× here, and the `gen`
  domain's 6-op chain measured 2.87× over `torch.compile`;
- the shipped path is fp64 **on CPU**, where no Triton kernel can run, so
  the loss is structural in the *delivery*, not in the rule;
- therefore the honest summary is not "invented operations are slow" but
  **"invented operations are slow only when delivered by the
  interpreter"**.

## The module-level gap — the delivery path is CPU-pinned

The obvious next measurement is the same comparison one level up: a real
module, `spelled` vs `noprim` vs `synop`, with the kernel body bound.
`prim_kernel_module` is written for exactly that, and **it cannot run yet**:

```
RuntimeError: Expected all tensors to be on the same device,
              but found at least two devices, cuda:0 and cpu!
              (torch_bridge.Plan.run_xs → _run_tail → linear)
```

`measure_module` now carries `device`/`dtype` and moves the feeds and the
exported parameters — but `torch_bridge.Plan.run_xs` allocates its
intermediates from a **CPU `like` tensor**, so the plan is pinned to CPU
by construction. Kernels need CUDA; the delivery path refuses to leave the
CPU.

The diagnosis, narrowed by instrumenting every binding: with feeds,
`_param_values`, `_param_map` and built modules all moved to CUDA, one
tensor still arrives on CPU at a two-operand op — a parameter or baked
weight **minted inside the plan** in a CPU context (`_randn_param`'s
fallback and `_mint_const`'s weak-scalar mint both default to CPU by
design). Aligning at the probe boundary cannot fix that; the mint has to
become device-aware.

So: **the shipped delivery path cannot host a generated kernel at all.**
The fix is a `torch_bridge` change (a device-aware fallback/baked-weight
mint inside the plan). I attempted it and **reverted it rather than ship
it unverified** — it touches the fp64 verify path, and a half-tested
device change there is worse than the documented gap. Until then the
module-level number does not exist, while the op-level 2.6–2.7× does.


- op level, not module level: this times the denotation alone (one
  broadcast + one kernel launch vs one interpreter pass). The module-level
  probe cannot use these kernels at all — it times CPU fp64, where Triton
  is unavailable.
- fp32 on GPU; the verification leg the family ships under is fp64. A fp64
  kernel is a separate measurement and may not keep the 2.6×.
- 1 M elements, one device, one seed. Small tensors are launch-bound and
  will not show this ratio; `gen1` measured 1.92× vs compiled on a 2-op
  chain for exactly that reason.
- Six of 57 prims are pointwise enough to codegen; the rest need a body
  the generator does not model yet.

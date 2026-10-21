"""The kernelopt spike's bench cases — measured codegen pay.

Eleven kernel cases: a spelled program, a codegen-law candidate, and
the honest row — lower both through
:class:`mlopt_kernel.KernelSink`, verify numerically (verify-first),
then time spelled vs scheduled.  Where the case has a registered
hand-Triton kernel (``bench.limits.triton``), the row times and
verifies that arm too — the practitioner's kernel is the field
baseline the fused composite is measured against.  This is the
spike's measured half of the go/no-go: does the engine emit a
verified-equal kernel at lower measured cost than the spelled
baseline.

Run::

    .venv/bin/python -m bench.discovery.kernel
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from typing import Any

from mlopt_core.egraph import EGraph
from mlopt_core.ir import IR, Const, Op, TensorType, Var
from mlopt_kernel import KERNEL_RULES, KernelSink, kernel_cost

#: (name, category, spelled builder, rules to fire in order)
CASES: tuple[tuple[str, str, Any, tuple[str, ...]], ...] = ()


def _v(name, shape):
    return Var(name, TensorType(tuple(shape)))


def _mk():
    x = _v("x", (4096, 512))
    y = _v("y", (4096, 512))
    w = _v("w", (512, 512))
    b = _v("b", (512,))
    return x, y, w, b


def _cases():
    x, y, w, b = _mk()
    ln_center = Op.make(
        "sub", x, Op.make("mean", x, dim=-1, keepdim=True)
    )
    return (
        (
            "kernel_silu",
            "speed",
            Op.make("mul", x, Op.make("sigmoid", x)),
            ("kern_fuse_silu",),
        ),
        (
            "kernel_softsign",
            "speed",
            Op.make(
                "div", x, Op.make("add", Op.make("abs", x), Const(1))
            ),
            ("kern_fuse_softsign",),
        ),
        (
            "kernel_fma",
            "speed",
            Op.make("add", Op.make("mul", x, y), Op.make("tanh", y)),
            ("kern_fuse_addmul",),
        ),
        (
            "kernel_swiglu",
            "speed",
            Op.make(
                "mul", Op.make("mul", x, Op.make("sigmoid", x)), y
            ),
            ("kern_fuse_silu", "kern_fuse_swiglu"),
        ),
        (
            "kernel_softmax",
            "speed",
            Op.make(
                "div",
                Op.make("exp", x),
                Op.make(
                    "sum", Op.make("exp", x), dim=-1, keepdim=True
                ),
            ),
            ("kern_fuse_softmax",),
        ),
        (
            "kernel_layernorm",
            "speed",
            Op.make(
                "mul",
                ln_center,
                Op.make(
                    "rsqrt",
                    Op.make(
                        "add",
                        Op.make(
                            "mean",
                            Op.make("pow", ln_center, Const(2)),
                            dim=-1,
                            keepdim=True,
                        ),
                        Const(1e-5),
                    ),
                ),
            ),
            ("kern_fuse_layernorm",),
        ),
        (
            "kernel_mm_bias",
            "speed",
            Op.make("add", Op.make("matmul", x, w), b),
            ("kern_fuse_mm_bias",),
        ),
        (
            "kernel_mm_scale_bias",
            "speed",
            Op.make(
                "add",
                Op.make("mul", Op.make("matmul", x, w), Const(0.5)),
                b,
            ),
            ("kern_fuse_mm_scale_bias", "kern_fuse_addmul"),
        ),
        (
            "kernel_strength",
            "speed",
            Op.make(
                "add",
                Op.make("pow", x, Const(2)),
                Op.make("div", y, Const(4.0)),
            ),
            ("kern_square_strength", "kern_div_recip"),
        ),
        (
            "kernel_mixed",
            "memory",
            Op.make(
                "mul",
                Op.make("mul", x, Op.make("sigmoid", x)),
                Op.make("sigmoid", y),
            ),
            ("kern_fuse_silu",),
        ),
        (
            "kernel_flat",
            "speed",
            Op.make("add", Op.make("tanh", x), Op.make("cos", y)),
            (),
        ),
    )


@dataclass
class KernelRow:
    """One measured case row — spelled vs scheduled, verified."""

    name: str
    category: str
    scheduled: str = ""
    verified: bool | None = None
    max_abs: float | None = None
    cost_before: float = 0.0
    cost_after: float = 0.0
    spelled_us: float = 0.0
    scheduled_us: float = 0.0
    speedup: float | None = None
    triton_us: float | None = None
    triton_verified: bool | None = None
    triton_note: str = ""
    detail: dict = field(default_factory=dict)


def _time(fn, args, reps=50, device=None):
    import torch

    args = tuple(a.to(device) for a in args) if device else args
    for _ in range(10):
        fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def _vars_of(t):
    out = []

    def walk(u):
        if isinstance(u, Var):
            if u.name not in [v.name for v in out]:
                out.append(u)
        elif isinstance(u, Op):
            for a in u.args:
                walk(a)

    walk(t)
    return out


def _triton_arm(name):
    """Return ``(callable, None)`` for case *name*'s hand Triton kernel.

    The honest skip cases: triton not installed, no CUDA, or no
    kernel registered — the row records the reason, never a silent
    drop.
    """
    if importlib.util.find_spec("triton") is None:
        return None, "triton not installed"
    import torch

    if not torch.cuda.is_available():
        return None, "needs CUDA"
    from bench.limits.triton import kernel_builder

    mk = kernel_builder(name)
    if mk is None:
        return None, f"no triton kernel registered for {name!r}"
    return mk(), None


def run_case(
    name, category, term, rules_to_fire, *, device=None, reps=50
):
    """Board the spelled term, fire the laws, verify + time delivery."""
    import torch

    row = KernelRow(name=name, category=category)
    eg = EGraph()
    root = eg.add_term(term)
    for rn in rules_to_fire:
        r = next(r for r in KERNEL_RULES if r.name == rn)
        eg.apply_rule(r, root)
    best = eg.extract_best(root, kernel_cost) or term
    row.scheduled = repr(best)
    row.cost_before = kernel_cost(term)
    row.cost_after = kernel_cost(best)

    inputs = _vars_of(term)
    params = {}
    ir_s = IR(
        root=term,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params=params,
    )
    ir_b = IR(root=best, inputs=inputs, input_names=ir_s.input_names)
    sink = KernelSink()
    spelled = sink.lower(ir_s)
    sched = sink.lower(ir_b)
    feeds = tuple(
        torch.randn(tuple(v.typ.shape), dtype=torch.float32)
        for v in inputs
    )
    rep = sink.verify(spelled, sched, feeds)
    row.verified = bool(rep.passed)
    row.max_abs = rep.max_abs
    row.spelled_us = _time(spelled, feeds, reps, device)
    row.scheduled_us = _time(sched, feeds, reps, device)
    if row.scheduled_us:
        row.speedup = round(row.spelled_us / row.scheduled_us, 3)

    arm, why = _triton_arm(name)
    if arm is None:
        row.triton_note = why or "unavailable"
    else:
        try:
            feeds_dev = tuple(a.to(device) for a in feeds)
            rep_t = sink.verify(spelled, arm, feeds_dev)
            row.triton_verified = bool(rep_t.passed)
            row.triton_us = _time(arm, feeds_dev, reps, device)
        except Exception as e:  # kernel failed — honest row
            row.triton_note = f"triton arm: {e}"
    return row


def main() -> list[KernelRow]:
    """Run every kernel case; print the honest row."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for name, cat, term, fires in _cases():
        row = run_case(name, cat, term, fires, device=device)
        rows.append(row)
        tri = (
            f"triton {row.triton_us:.1f}us (verified={row.triton_verified})"
            if row.triton_us is not None
            else f"triton — ({row.triton_note})"
        )
        print(  # stdout-compat
            f"{name}: verified={row.verified} "
            f"cost {row.cost_before:.2f}->{row.cost_after:.2f} "
            f"{row.spelled_us:.1f}us -> {row.scheduled_us:.1f}us "
            f"({row.speedup}x), {tri} [{device}]"
        )
    return rows


if __name__ == "__main__":
    main()

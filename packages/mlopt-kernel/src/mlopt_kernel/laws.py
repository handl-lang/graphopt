# ruff: noqa: RUF001, RUF003 — math notation (≡, σ, −, ·) in docstrings.
"""The codegen law bank — schedule-level rewrites over the IR.

Not a language: five laws.  Three fold spelled op-compositions into
*named fused-kernel ops* (``kf_*`` — the kernel vocabulary the
:class:`~mlopt_kernel.sink.KernelSink` lowers as one compiled
composite, in the repo's ``silu``/``sdpa``/``rms_norm`` fold-op
tradition); two are arithmetic strength reductions — the codegen
move that replaces an expensive elementary op with the cheap one a
device kernel emits anyway.

A design note the e-graph forces: a *wrapper* marker
(``k_fused(X)``) merges into ``X``'s own e-class and points back at
it — a cyclic enode the extractor can never pick.  The fused ops
here take the region's *leaves* as children, so they are ordinary
non-cyclic members: extraction chooses between spelled and fused by
:func:`~mlopt_kernel.cost.kernel_cost` alone.

The bank deliberately stays out of ``mlopt_core.laws.DEFAULT``: a
change to a default rule changes extraction for every caller, and
these rules only pay under the kernel cost model.
"""

from __future__ import annotations

from typing import Any

from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Const, Op
from mlopt_core.laws.base import R

__all__ = [
    "FUSED_KERNEL_OPS",
    "KERNEL_RULES",
    "kern_div_recip",
    "kern_fuse_addmul",
    "kern_fuse_silu",
    "kern_fuse_softsign",
    "kern_square_strength",
]

#: The kernel-vocabulary ops the laws mint — every ``kf_*`` name is a
#: single kernel launch in the sink's cost model and a compiled
#: composite in its binding table.
FUSED_KERNEL_OPS = frozenset({"kf_silu", "kf_softsign", "kf_addmul"})

kern_fuse_silu = R(
    "kern_fuse_silu",
    Op.make("mul", "a", Op.make("sigmoid", "b")),
    Op.make("kf_silu", "a", "b"),
    law="Gate fusion: a·σ(b) lowers as one fused kernel — the "
    "elementwise product and the sigmoid in a single pass (SiLU is "
    "the a == b section).",
)

kern_fuse_softsign = R(
    "kern_fuse_softsign",
    Op.make("div", "a", Op.make("add", Op.make("abs", "b"), "c")),
    Op.make("kf_softsign", "a", "b", "c"),
    law="Softsign fusion: a/(|b|+c) lowers as one fused kernel — "
    "abs, add and div in a single pass (softsign(x) is a≡b, c≡1).",
)

kern_fuse_addmul = R(
    "kern_fuse_addmul",
    Op.make("add", Op.make("mul", "a", "b"), "c"),
    Op.make("kf_addmul", "a", "b", "c"),
    law="Fused multiply-add: a·b + c lowers as one fused kernel — "
    "the hardware FMA the elementwise pair would emit anyway.",
)

kern_square_strength = R(
    "kern_square_strength",
    Op.make("pow", "x", Const(2)),
    Op.make("mul", "x", "x"),
    law="Strength reduction: pow(x, 2) lowers as one multiply — "
    "pow_to_square + square_expand composed, at codegen level.",
    derivation=("pow_to_square", "square_expand"),
)


def _numeric_const(bound: dict[str, Any]) -> bool:
    """``c`` bound to a nonzero numeric literal (reciprocal folds)."""
    c = bound["c"]
    return (
        isinstance(c, Const)
        and isinstance(c.value, (int, float))
        and not isinstance(c.value, bool)
        and c.value != 0
    )


def _recip(bound: dict[str, Any]) -> dict[str, Any] | None:
    """Mint ``r = 1/c`` — the folded constant ``kern_div_recip`` needs."""
    return {"r": Const(1 / bound["c"].value)}


kern_div_recip = R(
    "kern_div_recip",
    Op.make("div", "x", "c"),
    Op.make("mul", "x", "r"),
    check=_numeric_const,
    derive=_recip,
    law="Reciprocal folding: x / c → x · (1/c) for constant c — "
    "division is the expensive elementary op; a device kernel emits "
    "the multiply.",
)

#: The bank, in budget order — the board's action set.
KERNEL_RULES: tuple[Rewrite, ...] = (
    kern_fuse_silu,
    kern_fuse_softsign,
    kern_fuse_addmul,
    kern_square_strength,
    kern_div_recip,
)

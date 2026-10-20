"""``eager`` — the spelled module, run as written.

The floor every other arm is quoted against: the case's "before"
fixture with no compilation, folding, or reordering — exactly what a
practitioner ships before reaching for a compiler.  On the symbolic
axis its cost is the case's ``cost_before``; measured, it is the
arm that prices the compiler's overhead honestly — the "0.51× vs
eager" rows in ``docs/results.md`` are eager embarrassing a carrier
below its launch-overhead crossover.
"""

# ruff: noqa: RUF001, RUF002 — math notation in prose.

from __future__ import annotations

from typing import Any

from bench.limits.registry import BaselineCtx, BaselineSpec

SPEC = BaselineSpec(
    name="eager",
    kind="eager",
    title="PyTorch eager — the spelled module, as written",
    note="no compilation — the unoptimized floor",
    record=(
        "launch-bound crossover: carriers lose to eager below ~T=512 "
        "(0.51×, docs/results.md)"
    ),
)


def build(ctx: BaselineCtx, *, spec: BaselineSpec) -> Any:
    """Return the spelled module itself — the eager floor."""
    del spec
    return ctx.model

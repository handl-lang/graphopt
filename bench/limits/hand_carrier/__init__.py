"""``hand_carrier`` — the carrier term under the generic evaluator.

When the extraction lands on a carrier term (``om_decayed``,
``applyd`` — the hand-written scan/carrier laws), the delivery path
routes it to the batched executor and the "mlopt" arm *is* that
schedule.  This baseline prices the same extracted term lowered
through ``sink.lower`` — the generic evaluator, leaves one-by-one —
the arm ``lab/probes/retnet_baseline_probe.py`` reports as
``carrier``.  The gap between this arm and the delivered one is
what the executor routing buys; the arm is also the honest loss
column — **0.51× vs eager at T=512**, below the launch-overhead
crossover (``docs/results.md``).

On a case whose extraction no carrier executor claims, the generic
lowering is the delivered module — the arm would be a duplicate, so
the baseline declines: ``_route_spec`` returning ``None`` is the
honest "no carrier here" signal.
"""

# ruff: noqa: RUF001, RUF002 — math notation in prose.

from __future__ import annotations

from typing import Any

from bench.limits.registry import (
    BaselineCtx,
    BaselineSpec,
    SkipBaseline,
)

SPEC = BaselineSpec(
    name="hand_carrier",
    kind="carrier",
    title="carrier term, generic lowering (no batched schedule)",
    note="same extracted term — executor routing is the delta",
    record=(
        "0.51× vs eager below the T=512 crossover — the carrier "
        "without its level-batched schedule (docs/results.md)"
    ),
)


def build(ctx: BaselineCtx, *, spec: BaselineSpec) -> Any:
    """Lower the extracted term through the generic evaluator."""
    del spec
    from mlopt_core.ir import IR
    from mlopt_orchestrator.optimize import _route_spec

    sr = ctx.sr
    if _route_spec(sr.term, ctx.sink) is None:
        raise SkipBaseline(
            "no carrier executor claims the extraction — "
            "generic lowering would duplicate the mlopt arm"
        )
    optimized_ir = IR(
        root=sr.term,
        inputs=sr.ir.inputs,
        input_names=sr.ir.input_names,
        params=sr.ir.params,
    )
    return ctx.sink.lower(optimized_ir, dict(sr.param_values))

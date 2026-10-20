"""``opt_einsum`` — the field's contraction-ordering baseline.

`opt_einsum` is what the tensor-network world ships for ordering:
its ``greedy`` path is the floor the contraction board prices every
player against (``bench/players``), and on the corpus it applies
where the workload *is* a contraction — the ``weight_chain_fold``
two-GEMM chain, spelled ``x @ W1ᵀ @ W2ᵀ`` as the einsum
``bi,ji,kj->bk``.  Greedy picks the weights-first order — the same
fold the engine's ``assoc_linear`` reaches — so the arm is the
honest "the ordering tool already knew this" row.

The arm is a ``oe.contract_expression``-compiled callable: not a
torch graph ``TorchSource`` can export, so it registers
measured-only (the ordering itself is the claim; the contraction
board prices order *quality* under the shared pair-cost model).

Requires the ``opt_einsum`` package (installed — the contraction
board's ``oe-greedy`` floor already uses it).
"""

from __future__ import annotations

from typing import Any

from bench.limits.registry import (
    BaselineCtx,
    BaselineSpec,
    SkipBaseline,
)

SPEC = BaselineSpec(
    name="opt_einsum",
    kind="order",
    title="opt_einsum greedy — the contraction-ordering baseline",
    cases=("weight_chain_fold",),
    requires=("opt_einsum",),
    symbolic=False,  # a contract_expression, not a torch graph
    note="ordering, not a graph — measured-only",
    record=(
        "oe-greedy is the contraction board's floor — the engine's "
        "ordering players tie its cost under the shared pair model "
        "(bench/players leaderboard)"
    ),
)

#: case name → (einsum equation, operand extractor).  A case joins
#: the roster by spelling its contraction as one equation.
_EQUATIONS = {
    "weight_chain_fold": "bi,ji,kj->bk",  # (x @ W1ᵀ) @ W2ᵀ
}


def build(ctx: BaselineCtx, *, spec: BaselineSpec) -> Any:
    """Return the oe-greedy contraction of the case's chain."""
    del spec
    import opt_einsum as oe

    eq = _EQUATIONS[ctx.case.name]
    model, feed = ctx.model, ctx.feed
    x = feed[0] if isinstance(feed, tuple) else feed
    try:
        w1, w2 = model[0].weight, model[1].weight
    except (TypeError, KeyError, IndexError, AttributeError) as e:
        raise SkipBaseline(
            f"case {ctx.case.name!r} does not expose a two-Linear "
            f"chain: {e}"
        ) from e
    expr = oe.contract_expression(
        eq, x.shape, w1.shape, w2.shape, optimize="greedy"
    )

    def run(x: Any, _expr=expr, _w1=w1, _w2=w2) -> Any:
        return _expr(x, _w1, _w2)

    return run

"""``paper_impl`` — the known-good impl the case's citation ships.

Every corpus case that names a ``ref`` baseline declares the
implementation a careful practitioner would build by hand: the
RetNet paper's own chunkwise recurrence (``retnet_chunked`` — the
measured **parity** arm, 2.08 vs 2.12 ms at T=2048), the hand-folded
conv, ``F.scaled_dot_product_attention``, the Blelloch closed-form
prefix, the hand-fused SwiGLU, the pre-folded Linear.  Those impls
live in :data:`bench.discovery.refs.REFS`; this adapter resolves the
case's registered ``ref`` entry to its builder, so the floor is
*code*, not a number copied from a results doc.
"""

from __future__ import annotations

from typing import Any

from bench.discovery.refs import REFS
from bench.limits.registry import (
    BaselineCtx,
    BaselineSpec,
    SkipBaseline,
)

SPEC = BaselineSpec(
    name="paper_impl",
    kind="impl",
    title="known-good impl — the case's registered ref",
    note="from bench.discovery.refs — the hand-built floor",
    record=(
        "paper chunkwise ties the fused carrier (2.08 vs 2.12 ms "
        "T=2048; faster under graph replay) — "
        "lab/probes/retnet_baseline_probe.py"
    ),
)


def build(ctx: BaselineCtx, *, spec: BaselineSpec) -> Any:
    """Return the case's registered ``ref`` impl, built and smoked."""
    del spec
    ref = next(
        (b.ref for b in ctx.case.baselines if b.kind == "ref"), None
    )
    if ref is None:
        raise SkipBaseline(
            f"case {ctx.case.name!r} registers no ref impl"
        )
    if ref not in REFS:
        raise SkipBaseline(f"ref {ref!r} not in REFS")
    mod = REFS[ref](ctx.model, ctx.feed)
    import torch

    with torch.no_grad():
        args = ctx.feed if isinstance(ctx.feed, tuple) else (ctx.feed,)
        mod(*args)  # a broken ref is no floor — smoke it here
    return mod

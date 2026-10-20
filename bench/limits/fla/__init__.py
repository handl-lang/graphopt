"""``fla`` — the FLA flash-linear-attention Triton kernels.

fla-org's ``chunk_retention`` is the kernel-level floor the RetNet
carrier is honestly measured against: **~7–11× ahead** of the fused
carrier in ``bench/probes/retnet_baseline_probe.py`` — the gap the
"limits" table exists to keep visible.  The kernels are fp32 Triton
(no fp64, per-head decay and ``1/√d`` scaling baked in), so this is
a *speed/memory* baseline — numerics intentionally differ from the
fp64 corpus fixtures and the row says so.  The arm runs the same
projections the probe times (q/k/v linears inside the call), so the
number is like-for-like with the published record.

Two scratch gates stand in front of the arm: ``fla-core`` (the
``fla`` package, installed ``--no-deps`` outside the lockfile) and
CUDA — Triton does not run on the CPU lane.  Absent either, the row
reports the reason; the fast lane never breaks on its absence.
"""

# ruff: noqa: RUF001, RUF002 — math notation in prose.

from __future__ import annotations

from typing import Any

from bench.limits.registry import BaselineCtx, BaselineSpec

SPEC = BaselineSpec(
    name="fla",
    kind="kernel",
    title="FLA chunk_retention — the Triton-kernel floor",
    cases=("retnet_chunkwise",),
    requires=("fla",),
    needs_cuda=True,
    symbolic=False,  # Triton kernels — no IR export
    note="fp32 kernel — speed/memory baseline; numerics differ",
    record=(
        "~7–11× ahead of the fused carrier — kernel-level: fp64 "
        "GEMMs + a materialized score tensor on our side "
        "(bench/probes/retnet_baseline_probe.py)"
    ),
)


def build(ctx: BaselineCtx, *, spec: BaselineSpec) -> Any:
    """Return the ``chunk_retention`` arm, projections inside the call."""
    del spec
    from fla.ops.retention import chunk_retention

    model = ctx.model

    def run(x: Any) -> Any:
        q = model.q(x).unsqueeze(2).float().contiguous()
        k = model.k(x).unsqueeze(2).float().contiguous()
        v = model.v(x).unsqueeze(2).float().contiguous()
        return chunk_retention(q, k, v)[0]

    return run

"""``torch_compile`` — ``torch.compile`` of the spelled module.

The compiler a practitioner actually reaches for: Inductor over the
case's "before" fixture.  It has no honest symbolic form — the
post-grad graph it runs is not the spelled term — so the fast lane
registers it measured-only and ``--full`` times it.

Measured record (``docs/results.md``): Inductor wins its home turf —
pure pointwise fusion is a **0.82–0.88× loss** for the engine — but
cannot express the weights-first form the e-graph reaches on the
matmul chain (**3.96× mlopt**, 22 µs vs 87 µs — no weight folding
without ``freezing``), and ties the fused SwiGLU at **2.00–2.11×**
against it.
"""

# ruff: noqa: RUF001, RUF002 — math notation in prose.

from __future__ import annotations

from typing import Any

from bench.limits.registry import BaselineCtx, BaselineSpec

SPEC = BaselineSpec(
    name="torch_compile",
    kind="compile",
    title="torch.compile — the Inductor pipeline",
    symbolic=False,  # the post-grad graph is not the spelled term
    note="measured-only — no symbolic form",
    record=(
        "pointwise fusion 0.82–0.88× loss (Inductor wins); weight "
        "chain 3.96× mlopt (no fold without freezing) — "
        "docs/results.md"
    ),
)


def build(ctx: BaselineCtx, *, spec: BaselineSpec) -> Any:
    """Return ``torch.compile`` of the spelled module."""
    del spec
    import torch

    return torch.compile(ctx.model)

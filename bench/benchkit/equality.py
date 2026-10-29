"""The equality gate: a candidate must match the *incumbent*, not itself.

The AWQ nibble-order bug survived a full round of "exactness" checks
because the kernel and its reference shared the same wrong assumption
(cosine 0.21 against `awq_gemm`, P21). A gate that can be passed by two
implementations agreeing on a mistake is not a gate.

So the gate has one rule, enforced by its signature: the comparison target
is a callable the candidate's author did not write — `awq_gemm`, Marlin,
the framework. Nothing in this module computes a reference; it only
compares. `graphopt/bench/drivers/probes/awq/kernel_v2.py` used to
satisfy this by computing its own `unpack_ref` and calling the result
"truth".

    from bench.benchkit.equality import cosine_gate, require

    verdict = cosine_gate("split_k=4", ours(), incumbent=cops.awq_gemm(...))
    require(verdict)  # raises if cosine < 0.9999

Tolerance is `0.9999` because that is the number the M1 build list gates
on; it is loose enough for fp16 accumulate order and tight enough that
0.21 cannot pass.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

__all__ = [
    "GATE_TOLERANCE",
    "GateVerdict",
    "cosine_gate",
    "gate_all",
    "require",
]

#: Minimum cosine similarity for a candidate to count as the same function.
GATE_TOLERANCE = 0.9999


@dataclass(frozen=True)
class GateVerdict:
    """One candidate, judged against the incumbent."""

    name: str
    cosine: float
    passed: bool
    tolerance: float
    #: max|candidate − incumbent| ÷ mean|incumbent| — scale-free, so it can
    #: be quoted across shapes. Cosine alone hides a uniform scaling error.
    relative_worst: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cosine": round(self.cosine, 7),
            "passed": self.passed,
            "tolerance": self.tolerance,
            "relative_worst": round(self.relative_worst, 6),
        }


def _flatten(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().reshape(-1)


def cosine_gate(
    name: str,
    out: torch.Tensor,
    *,
    incumbent: torch.Tensor,
    tolerance: float = GATE_TOLERANCE,
) -> GateVerdict:
    """Judge ``out`` against ``incumbent`` (an external implementation's)."""
    a, b = _flatten(out), _flatten(incumbent)
    if a.shape != b.shape:
        raise ValueError(
            f"{name}: candidate {tuple(out.shape)} vs incumbent "
            f"{tuple(incumbent.shape)} — shapes must match"
        )
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    scale = float(b.abs().mean()) or 1.0
    worst = float((a - b).abs().max()) / scale
    return GateVerdict(
        name=name,
        cosine=cosine,
        passed=cosine >= tolerance,
        tolerance=tolerance,
        relative_worst=worst,
    )


def gate_all(
    candidates: Mapping[str, Callable[[], torch.Tensor]],
    *,
    incumbent: torch.Tensor,
    tolerance: float = GATE_TOLERANCE,
) -> dict[str, GateVerdict]:
    """Run every candidate through :func:`cosine_gate`."""
    return {
        name: cosine_gate(
            name, fn(), incumbent=incumbent, tolerance=tolerance
        )
        for name, fn in candidates.items()
    }


def require(verdict: GateVerdict) -> None:
    """Raise unless the verdict passed.

    The message names the failure mode that motivated the gate, so the next
    reader does not "fix" it by loosening the tolerance.
    """
    if verdict.passed:
        return
    raise AssertionError(
        f"{verdict.name}: cosine {verdict.cosine:.6f} < "
        f"{verdict.tolerance} against the incumbent "
        f"(relative worst {verdict.relative_worst:.4f}).\n"
        "If the candidate matches your own reference but not the incumbent, "
        "the two share an assumption that the incumbent does not — that is "
        "the AWQ nibble-order bug (P21), not a tolerance problem."
    )

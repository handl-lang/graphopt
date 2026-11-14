"""Typed report layer and the uniform verification gate.

Plan 0001 phase 3a/3b: the optimizer's stats dicts stay plain dicts at
the public boundary — callers and tests index them — but every schema
is now code.  ``OptReport`` / ``CompositionalReport`` / ``BlockReport``
give typed access with an exact ``to_dict`` round-trip, and the three
near-duplicate diff+tolerance checks (optimize_model's verbose verify,
optimize_compositional's per-block and end-to-end gates, the bench
files' ``rel_diff`` helpers) collapse into ``verify_equiv`` /
``verify_module``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "BlockReport",
    "CompositionalReport",
    "OptReport",
    "SaturationStats",
    "VerifyReport",
    "rel_diff",
    "verify_equiv",
    "verify_module",
]

#: Denominator floor of the historical rel-diff metric — every call
#: site this module replaces used ``+ 1e-8``; it is part of the gate's
#: bit-for-bit semantics, not a tunable.
_REL_FLOOR = 1e-8

#: Headroom of the dtype-resolution rtol floor, in units of the
#: dtype's machine epsilon (``torch.finfo(dtype).eps``).
#:
#: ``verify_equiv``'s ``max_rel`` metric cannot resolve differences
#: below the compute dtype's own rounding: two evaluations of the
#: SAME program in bf16 already drift by reassociation at ~4 eps —
#: measured ``max_rel ≈ 0.031`` on whisper-turbo bf16 (bf16 eps is
#: 0.0078; see ``docs/reports/verify_bf16_mem.md``).  Demanding the
#: historical ``rtol=1e-4`` of a bf16 pair asks for ~13x more
#: precision than the dtype carries, so every honest bf16 delivery
#: fails on rounding noise, not semantics.  The floor is therefore
#: ``_RTOL_DTYPE_FACTOR * eps`` — a reassociation-drift budget, not a
#: semantic loosening: a wrong kernel produces O(0.1-1+) relative
#: errors, far above the bf16 floor (~6e-2), while sub-precision
#: noise is indistinguishable from a correct rewrite *in that dtype*.
#: The factor is a heuristic anchored on the measured whisper drift
#: (~4 eps) with ~2x headroom; it is deliberately NOT applied to
#: fp32/fp64 — their eps*factor sits below every shipped rtol and the
#: fp64 verify convention must stay bit-identical.
_RTOL_DTYPE_FACTOR = 8.0

_FP32_EPS = torch.finfo(torch.float32).eps


def _dtype_rtol_floor(*outputs: Any) -> float:
    """Return the dtype-resolution floor for ``verify_equiv``'s rtol.

    Returns ``_RTOL_DTYPE_FACTOR * eps`` for the coarsest floating
    output dtype whose precision is below fp32's (fp16, bf16, the
    fp8 family — dtypes whose machine epsilon exceeds
    ``_FP32_EPS``), else ``0.0``.  Non-floating and non-torch dtypes
    contribute nothing: integer outputs compare exactly and foreign
    array types keep the historical semantics.  The effective
    tolerance is ``max(rtol, floor)`` — an explicit *looser* rtol is
    honored, and an explicit *tighter* one cannot be satisfied by a
    dtype that cannot represent it anyway.
    """
    floor = 0.0
    for out in outputs:
        dtype = getattr(out, "dtype", None)
        if not isinstance(dtype, torch.dtype):
            continue
        try:
            eps = torch.finfo(dtype).eps
        except (TypeError, ValueError):
            continue
        if eps > _FP32_EPS:
            floor = max(floor, _RTOL_DTYPE_FACTOR * eps)
    return floor


# ---------------------------------------------------------------------------
#  Uniform verify gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyReport:
    """Result of one equivalence check between two outputs.

    For a multi-output module the check is over every output: the
    scalar fields are the worst case across the outputs and
    ``per_output`` carries each output's ``max|ref - out|`` in export
    order (``None`` for a single-output comparison).
    """

    max_abs: float  # max|ref - out|
    max_rel: float  # max_abs / (max|ref| + _REL_FLOOR)
    passed: bool
    per_output: tuple[float, ...] | None = None
    #: The effective rtol the gate applied — ``max(rtol,
    #: _dtype_rtol_floor(...))``; for a multi-output aggregate the
    #: loosest per-output value, ``None`` on the fail-closed
    #: arity-mismatch report.
    rtol: float | None = None


def rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the historical relative-diff metric.

    Identical everywhere it was inlined:
    ``max|a - b| / (max|a| + 1e-8)``.
    """
    return (a - b).abs().max().item() / (
        a.abs().max().item() + _REL_FLOOR
    )


def _flatten_outputs(x: Any) -> list:
    """Flatten a returned pytree to its tensor leaves, in order.

    ``torch.export`` flattens a model's return pytree, and the IR's
    ``pack`` root pairs those leaves in the same order — so the
    lowered module returns a *flat* tuple even when the source module
    returns a nested one (measured: ``nn.LSTM`` returns
    ``(output, (h_n, c_n))``).  Comparing the nested reference
    against the flat result pairs a tuple against a tensor and
    reports ``inf``; flattening both sides is what makes the
    comparison meaningful.
    """
    if isinstance(x, (tuple, list)):
        out: list = []
        for item in x:
            out.extend(_flatten_outputs(item))
        return out
    return [x]


def _verify_outputs(
    ref_out: tuple | list,
    out: Any,
    rtol: float,
    atol: float | None,
) -> VerifyReport:
    """Compare two multi-output results, element by element.

    Every output must verify: ``passed`` is the conjunction and the
    scalar fields are the worst case over the outputs, with each
    output's ``max_abs`` recorded in ``per_output``.  A shape mismatch
    (the lowered module returned a different number of outputs) fails
    closed rather than comparing a prefix.  Both sides are flattened
    first, since a nested return pytree compares against the flat
    ``pack`` result (:func:`_flatten_outputs`).
    """
    ref_flat = _flatten_outputs(ref_out)
    out_flat = _flatten_outputs(out)
    if not isinstance(out, (tuple, list)) or len(out_flat) != len(
        ref_flat
    ):
        return VerifyReport(
            max_abs=float("inf"),
            max_rel=float("inf"),
            passed=False,
        )
    reports = [
        verify_equiv(r, o, rtol=rtol, atol=atol)
        for r, o in zip(ref_flat, out_flat, strict=True)
    ]
    return VerifyReport(
        max_abs=max(r.max_abs for r in reports),
        max_rel=max(r.max_rel for r in reports),
        passed=all(r.passed for r in reports),
        per_output=tuple(r.max_abs for r in reports),
        rtol=max(
            (r.rtol for r in reports if r.rtol is not None),
            default=None,
        ),
    )


def _verify_pair(
    ref_out: Any,
    out: Any,
    rtol: float,
    atol: float | None,
) -> VerifyReport:
    """Compare two single-output results — the historical metric.

    The effective rtol is ``max(rtol, _dtype_rtol_floor(ref, out))``:
    sub-fp32 dtypes cannot resolve the historical ``1e-4`` and get a
    precision-derived floor instead (see ``_RTOL_DTYPE_FACTOR``);
    fp32/fp64 pairs are bit-identical to before.
    """
    max_abs = (ref_out - out).abs().max().item()
    max_rel = max_abs / (ref_out.abs().max().item() + _REL_FLOOR)
    rtol_eff = max(rtol, _dtype_rtol_floor(ref_out, out))
    passed = bool(max_rel < rtol_eff) and (
        atol is None or bool(max_abs <= atol)
    )
    return VerifyReport(
        max_abs=max_abs, max_rel=max_rel, passed=passed, rtol=rtol_eff
    )


def verify_equiv(
    ref_out: Any,
    out: Any,
    rtol: float = 1e-4,
    atol: float | None = None,
) -> VerifyReport:
    """Compare two already-computed outputs.

    Gate semantics (deliberately NOT ``torch.allclose``'s elementwise
    ``atol + rtol * |ref|`` — this is the metric every historical call
    site used):

    * ``max_rel = max|ref - out| / (max|ref| + 1e-8)``
    * ``passed = max_rel < rtol_eff`` and, when ``atol`` is given,
      additionally ``max_abs <= atol`` — where ``rtol_eff`` is
      ``max(rtol, _dtype_rtol_floor(ref, out))``: sub-fp32 dtypes
      (fp16/bf16/fp8) get a machine-epsilon-derived floor because
      they cannot resolve the historical ``1e-4``; fp32/fp64
      comparisons are bit-identical to before.  The applied value is
      recorded on the report's ``rtol`` field.

    A tuple/list ``ref_out`` is a multi-output result: every output is
    compared (see :func:`_verify_outputs`) — a module returning
    several tensors verifies on all of them, not just the first.
    """
    if isinstance(ref_out, (tuple, list)):
        return _verify_outputs(ref_out, out, rtol, atol)
    return _verify_pair(ref_out, out, rtol, atol)


def verify_module(
    ref_mod: torch.nn.Module,
    opt_mod: torch.nn.Module,
    inputs: torch.Tensor | tuple,
    rtol: float = 1e-4,
    atol: float | None = None,
) -> VerifyReport:
    """Run ``ref_mod`` and ``opt_mod`` on ``inputs`` and compare.

    ``inputs`` is a tensor or an args tuple; the run is under
    ``no_grad`` and the comparison goes through :func:`verify_equiv`.
    Tensor inputs are cloned so a forward that mutates its arguments
    cannot corrupt the comparison.

    A multi-output module (one returning a tuple) is verified on
    EVERY output: the report's scalars are the worst case and
    ``per_output`` lists each output's ``max|Δ|``.

    Low-memory verify: a module exposing ``_park_params`` (the
    :class:`~mlopt_torch.torch_bridge.IRModule` hook) releases its
    live weights to CPU around the OTHER side's forward, so the pair
    never holds two full weight sets on the compute device at once —
    the whisper-turbo 4 GB ``lower(verify=True)`` OOM.  Both sides
    park before the first forward (a delivered module may already be
    device-resident), the reference re-parks before the optimized
    forward, and each parked module re-aligns on its next forward.
    The forwards, input clones and comparison are unchanged — only
    device residency between them differs, so the check's strength
    is identical.  Modules without the hook keep the historical
    both-resident behaviour.

    The comparison itself goes through :func:`verify_equiv`, whose
    ``rtol`` gets the dtype-resolution floor (``_dtype_rtol_floor``)
    — sub-fp32 outputs (bf16/fp16/fp8) verify against
    ``max(rtol, 8*eps)`` because the dtype cannot resolve the
    historical ``1e-4``; fp32/fp64 are unchanged.
    """
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    ref_mod.eval()
    opt_mod.eval()
    ref_park = getattr(ref_mod, "_park_params", None)
    opt_park = getattr(opt_mod, "_park_params", None)
    with torch.no_grad():
        for park in (ref_park, opt_park):
            if park is not None:
                park()
        ref = ref_mod(
            *[
                a.clone() if isinstance(a, torch.Tensor) else a
                for a in args
            ]
        )
        if ref_park is not None:
            ref_park()
        out = opt_mod(
            *[
                a.clone() if isinstance(a, torch.Tensor) else a
                for a in args
            ]
        )
    return verify_equiv(ref, out, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
#  Typed stats reports — exact dict round-trip
# ---------------------------------------------------------------------------


def _plain(v: Any) -> Any:
    """Serialize a field value.

    Typed reports go back to dicts, anything else (dicts, lists, offer
    objects) passes through untouched.
    """
    if isinstance(v, (OptReport, BlockReport)):
        return v.to_dict()
    return v


@dataclass
class SaturationStats:
    """The ``EGraph.run`` payload.

    Always present on a stats dict that came out of equality saturation.
    Exposed on ``OptReport`` as the ``saturation_stats`` view; stored
    flat so ``to_dict`` reproduces the original keys exactly.
    """

    iterations: int | None = None
    n_enodes: int | None = None
    n_classes: int | None = None
    n_proof_edges: int | None = None
    truncation_level: int | None = None
    rule_budgets: dict[str, int] | None = None
    budget_suspended: list[str] | None = None


@dataclass
class OptReport:
    """Typed view of the ``stats`` dict ``optimize_model`` returns.

    Field names equal the dict keys.  Saturation keys are always
    populated on a real stats dict; the path-dependent extras
    (``pairing_groups`` only when the pairing pass found groups,
    ``nonlocal_lifts`` only when a lift fired,
    ``paired_extract``/``causal_specialized`` only when
    the corresponding specialization won) stay ``None`` when absent —
    and ``to_dict`` omits them, so
    ``OptReport.from_stats(stats).to_dict() == stats`` exactly.
    ``extra`` captures keys this schema does not model (forward-compat)
    and re-emits them verbatim.
    """

    # -- EGraph.run payload ------------------------------------------
    iterations: int | None = None
    n_enodes: int | None = None
    n_classes: int | None = None
    n_proof_edges: int | None = None
    truncation_level: int | None = None
    rule_budgets: dict[str, int] | None = None
    budget_suspended: list[str] | None = None
    # -- optimize_model additions ------------------------------------
    rule_fires: dict[str, int] | None = None
    pairing_groups: int | None = None
    nonlocal_lifts: int | None = None
    paired_extract: bool | None = None
    causal_specialized: bool | None = None
    # -- unmodelled keys, re-emitted verbatim -------------------------
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def saturation_stats(self) -> SaturationStats:
        """The ``EGraph.run`` keys grouped as one typed view."""
        return SaturationStats(
            iterations=self.iterations,
            n_enodes=self.n_enodes,
            n_classes=self.n_classes,
            n_proof_edges=self.n_proof_edges,
            truncation_level=self.truncation_level,
            rule_budgets=self.rule_budgets,
            budget_suspended=self.budget_suspended,
        )

    @classmethod
    def from_stats(cls, stats: dict[str, Any]) -> OptReport:
        """Build the typed report from a stats dict.

        Field names equal dict keys, so extraction is mechanical;
        ``blocks`` values (only on :class:`CompositionalReport`) are
        lifted to :class:`BlockReport`.
        """
        names = {f.name for f in dataclasses.fields(cls)} - {"extra"}
        kwargs: dict[str, Any] = {n: stats.get(n) for n in names}
        if kwargs.get("blocks") is not None:
            kwargs["blocks"] = {
                n: BlockReport.from_dict(e, name=n)
                for n, e in kwargs["blocks"].items()
            }
        kwargs["extra"] = {
            k: v for k, v in stats.items() if k not in names
        }
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to the loose stats dict.

        ``None`` fields are omitted, so the round-trip reproduces the
        input dict's exact key set.
        """
        out = dict(self.extra)
        for f in dataclasses.fields(self):
            if f.name == "extra":
                continue
            v = getattr(self, f.name)
            if v is None:
                continue
            if f.name == "blocks" and isinstance(v, dict):
                out[f.name] = {n: _plain(b) for n, b in v.items()}
            else:
                out[f.name] = _plain(v)
        return out


@dataclass
class BlockReport:
    """Typed view of one ``stats["blocks"][name]`` entry.

    ``status`` is one of ``"optimized"`` / ``"failed"`` / ``"skipped"``
    / ``"not_executed"``; which of the remaining keys an entry carries
    depends on it (today: ``not_executed`` has only ``status``;
    ``skipped`` adds ``reason``; ``failed`` adds ``error`` — plus
    ``reason == "resource_limit"`` and a ``rel_diff`` when it failed at
    verification — and ``time_s``; ``optimized`` adds ``rel_diff``,
    ``stats``, ``param_report``, ``time_s``).  ``None`` fields are
    omitted by :meth:`to_dict`, reproducing each shape exactly.
    ``name`` is the dotted block path — the dict key, not an entry
    key — so it is typed-only and never serialized into the entry.
    """

    name: str
    status: str
    rel_diff: float | None = None
    time_s: float | None = None
    reason: str | None = None
    error: str | None = None
    stats: OptReport | dict | None = None
    param_report: dict | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, entry: dict[str, Any], name: str = ""
    ) -> BlockReport:
        """Build a block report from a ``blocks`` entry dict."""
        names = {f.name for f in dataclasses.fields(cls)} - {
            "name",
            "extra",
        }
        kwargs: dict[str, Any] = {n: entry.get(n) for n in names}
        if isinstance(kwargs.get("stats"), dict):
            kwargs["stats"] = OptReport.from_stats(kwargs["stats"])
        kwargs["extra"] = {
            k: v for k, v in entry.items() if k not in names
        }
        return cls(name=name, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the block report back to an entry dict."""
        out: dict[str, Any] = {"status": self.status}
        for f in dataclasses.fields(self):
            if f.name in ("name", "status", "extra"):
                continue
            v = getattr(self, f.name)
            if v is not None:
                out[f.name] = _plain(v)
        out.update(self.extra)
        return out


@dataclass
class CompositionalReport(OptReport):
    """Typed view of the ``optimize_compositional`` stats dict.

    An ``OptReport`` extended with the driver keys.  The saturation
    fields stay ``None`` at top level (per-block saturation lives inside
    ``blocks[name].stats``) and are omitted on ``to_dict``, so the
    round-trip is exact.
    """

    compositional: bool | None = None
    n_blocks: int | None = None
    n_optimized: int | None = None
    n_failed: int | None = None
    n_skipped: int | None = None
    blocks: dict[str, BlockReport] | None = None
    in_place: bool | None = None
    #: True when the recomposed module shares the input's tensor
    #: storage (the param-sharing clone — the normal path); False when
    #: ``in_place`` fell back to returning the caller's module.
    shared_params: bool | None = None
    param_report: dict | None = None
    end_to_end: dict | None = None
    wall_time_s: float | None = None

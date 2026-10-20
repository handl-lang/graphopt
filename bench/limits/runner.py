"""The limits runner — registered baselines × corpus cases, one ratio.

For each :class:`~bench.discovery.cases.DiscoveryCase` the runner
builds the spelled fixture, runs the pipeline's search phase (same
knobs as :mod:`bench.discovery.runner`), extracts the certified
form — the "mlopt" arm — and then asks the *registry* for the
opponents: every baseline whose spec covers the case builds an arm
through :func:`bench.limits.registry.build`, or declines with a
reason the row reports.

Each row is one ``(case, baseline)`` pair and carries two ratios,
one per lane:

* ``ratio`` — ``baseline_cost / mlopt_cost`` on the case's symbolic
  axis (``dag_cost``).  >1 means the delivered form is priced below
  the baseline; <1 is an honest loss.
* ``measured_ratio`` — ``baseline_median / mlopt_median`` wall time
  under the benchkit ``Runner`` (``--full`` only), plus per-arm
  ``sink.verify`` against the spelled module and peak CUDA memory
  when the lane runs on a CUDA device.

The convention matches ``docs/results.md``: a ratio rendered
"0.82×" means the baseline runs at 82% of the delivered form's
speed — a loss.  Unavailable baselines keep their row with the
reason ("scratch dep", "requires CUDA"), never a silent drop.
"""

# ruff: noqa: RUF002 — math notation in prose.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mlopt_orchestrator.optimize import lower, search
from mlopt_torch.adapters import TorchSink, TorchSource

from bench.discovery.cases import DiscoveryCase
from bench.discovery.runner import (
    _RULE_PRESETS,
    COST_FNS,
    _build_fixture,
    _call,
    _module_cost,
)
from bench.limits import registry
from bench.limits.registry import (
    BaselineCtx,
    BaselineSpec,
    SkipBaseline,
)


@dataclass
class LimitRow:
    """One ``(case, baseline)`` row of the limits table."""

    case: str
    category: str
    baseline: str
    kind: str
    axis: str = "launch"
    available: bool = True
    mlopt_cost: float | None = None  # symbolic, case axis
    baseline_cost: float | None = None
    ratio: float | None = None  # baseline/mlopt — <1 is a loss
    mlopt_s: float | None = None  # --full lane medians
    baseline_s: float | None = None
    measured_ratio: float | None = None
    verify_passed: bool | None = None
    verify_max_abs: float | None = None
    peak_mib: float | None = None
    caveat: str = ""


@dataclass
class CaseLimits:
    """One case's outcome: the delivered arm plus its baseline rows."""

    case: DiscoveryCase
    cost_before: float | None = None
    cost_after: float | None = None
    mlopt_s: float | None = None
    verified: bool | None = None
    verify_max_abs: float | None = None
    rows: list[LimitRow] = field(default_factory=list)
    error: str = ""
    detail: dict = field(default_factory=dict)


def _peak_mib(fn: Callable, device: str) -> float | None:
    """Peak CUDA allocation (MiB) of one call; ``None`` off-CUDA."""
    if not device.startswith("cuda"):
        return None
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    return torch.cuda.max_memory_allocated() / 2**20


def _verdict(ratio: float | None) -> str:
    """The docs convention: >1 ahead, ~1 parity, <1 an honest loss."""
    if ratio is None:
        return "—"
    if ratio > 1.02:
        return "WIN"
    if ratio < 0.98:
        return "loss"
    return "parity"


def _spec_row(
    case: DiscoveryCase, s: BaselineSpec, ctx: BaselineCtx
) -> tuple[LimitRow, Any]:
    """Build one baseline's row (+ its arm) for *case*.

    Returns ``(row, module_or_None)`` — the module rides along so the
    full lane can time and verify it without rebuilding.
    """
    row = LimitRow(
        case=case.name,
        category=case.category,
        baseline=s.name,
        kind=s.kind,
        axis=case.cost,
        caveat=s.note,
    )
    reason = registry.availability(s.name, device=ctx.device)
    if reason is not None:
        row.available = False
        row.caveat = reason
        return row, None
    try:
        arm = registry.build(s.name, ctx)
    except SkipBaseline as e:
        row.available = False
        row.caveat = str(e)
        return row, None
    except Exception as e:  # build failed — honest row, not a crash
        row.available = False
        row.caveat = f"build: {e}"
        return row, None
    if not s.symbolic:
        row.caveat = s.note or "symbolic n/a — run --full"
        return row, arm
    try:
        row.baseline_cost = _module_cost(
            arm, ctx.feed, COST_FNS[case.cost], ctx.source
        )
    except Exception as e:
        row.caveat = f"cost: {e}"
    return row, arm


def run_case(
    case: DiscoveryCase,
    *,
    lane: str = "fast",
    baselines: list[str] | None = None,
    device: str = "cpu",
    min_run_time: float = 0.05,
    warmup: int = 3,
) -> CaseLimits:
    """Cross the registered baselines with one case; ``lane`` is
    ``"fast"`` (symbolic) or ``"full"`` (measured + verified)."""
    out = CaseLimits(case=case)
    if baselines is None:
        specs = registry.for_case(case.name)
    else:
        # An explicit request keeps its row even where the baseline
        # declines — build() reports the skip, never a silent drop.
        specs = [registry.spec(n) for n in baselines]
    try:
        model, feed = _build_fixture(case)
    except Exception as e:
        out.error = f"fixture: {e}"
        return out
    sink = TorchSink()
    source = TorchSource()
    cost_fn = COST_FNS[case.cost]
    rules = _RULE_PRESETS.get(case.rules, case.rules)
    try:
        sr = search(
            model,
            feed,
            source=source,
            capabilities=sink,
            rules=rules,
            max_iterations=case.max_iterations,
            max_enodes=case.max_enodes,
        )
    except Exception as e:
        out.error = f"search: {e}"
        return out
    from mlopt_core.cost import dag_cost

    out.cost_before = dag_cost(sr.ir.root, cost_fn)
    out.cost_after = dag_cost(sr.term, cost_fn)

    ctx = BaselineCtx(
        case=case,
        model=model,
        feed=feed,
        sr=sr,
        sink=sink,
        source=source,
        device=device,
        lane=lane,
    )
    arms: dict[str, Any] = {}
    for s in specs:
        row, arm = _spec_row(case, s, ctx)
        row.mlopt_cost = out.cost_after
        if (
            row.baseline_cost is not None
            and out.cost_after
            and out.cost_after > 0
        ):
            row.ratio = row.baseline_cost / out.cost_after
        out.rows.append(row)
        if arm is not None:
            arms[s.name] = arm

    if lane != "full":
        return out

    # -- measured pay ---------------------------------------------------
    from bench.benchkit import Case, Runner, Variant

    try:
        lr = lower(sr, feed, sink=sink, verify=True)
        delivered = lr.module
        out.verified = bool(lr.verified.passed) if lr.verified else None
        out.verify_max_abs = (
            float(lr.verified.max_abs) if lr.verified else None
        )
    except Exception as e:  # lowering declined — honest row
        out.error = f"lower: {e}"
        out.verified = False
        return out

    timed = {"mlopt": delivered, **arms}
    variants = [
        Variant(n, lambda m=mod: _call(m, feed))
        for n, mod in timed.items()
    ]
    runner = Runner(
        device=device, warmup=warmup, min_run_time=min_run_time
    )
    cells = runner.run(
        [
            Case(
                name=case.name,
                params={"size": case.size},
                variants=variants,
            )
        ]
    )
    medians = cells[0].medians
    iqrs = cells[0].iqr
    out.detail["median_s"] = dict(medians)
    out.detail["iqr_s"] = dict(iqrs)
    out.mlopt_s = medians.get("mlopt")
    for row in out.rows:
        mod = timed.get(row.baseline)
        if mod is None:
            continue
        row.baseline_s = medians.get(row.baseline)
        row.mlopt_s = out.mlopt_s
        if row.baseline_s and out.mlopt_s:
            row.measured_ratio = row.baseline_s / out.mlopt_s
        row.peak_mib = _peak_mib(lambda m=mod: _call(m, feed), device)
        try:
            rep = sink.verify(model, mod, feed)
            row.verify_passed = bool(rep.passed)
            row.verify_max_abs = float(rep.max_abs)
        except Exception as e:
            row.verify_passed = False
            row.caveat = f"verify: {e}"
    return out


def run_suite(
    cases: list[DiscoveryCase],
    *,
    lane: str = "fast",
    baselines: list[str] | None = None,
    device: str = "cpu",
    min_run_time: float = 0.05,
    warmup: int = 3,
    progress: Callable[[CaseLimits], None] | None = None,
) -> list[CaseLimits]:
    """Run every case in order; ``progress`` fires per case."""
    out = []
    for case in cases:
        res = run_case(
            case,
            lane=lane,
            baselines=baselines,
            device=device,
            min_run_time=min_run_time,
            warmup=warmup,
        )
        out.append(res)
        if progress is not None:
            progress(res)
    return out

"""The discovery-benchmark runner — per case, two questions.

For each :class:`~bench.discovery.cases.DiscoveryCase`:

1. build the spelled "before" module, export it through
   ``TorchSource`` and run the pipeline's search phase
   (``mlopt_orchestrator.optimize.search`` — saturation plus the
   non-local passes, priced through the sink's own cost model);
2. **derivability** — the ``expect`` spec: at least one signal
   observed (a law in ``rule_fires``, a non-local-pass stat, an op in
   the extracted term) AND the op contract holding on the extraction;
3. **certification** — ``eg.certificate(src, dst)`` replayed by
   ``verify_certificate``: the derivation stands alone, without the
   e-graph;
4. **cost** — ``dag_cost`` of the spelled and extracted terms on the
   case's axis, against every baseline that has a symbolic form;
5. **beat-best-registered** — the certified form is *strictly below*
   the best registered baseline's floor.  Tying the hand-known impl
   is a rediscovery, not a find — the flag says so.

Lanes: ``fast`` is steps 1-5 symbolically (seconds, gate-suitable);
``full`` additionally lowers the extracted term through the real
pipeline (``lower(verify=True)``) and times every available arm with
the benchkit ``Runner``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from mlopt_core.cost import (
    count_cost,
    dag_cost,
    executor_cost_for,
    flops_cost,
    launch_aware_cost,
    param_bytes_cost_for,
)
from mlopt_core.egraph import verify_certificate
from mlopt_core.ir import Op
from mlopt_orchestrator.optimize import lower, search
from mlopt_torch.adapters import TorchSink, TorchSource

from bench.discovery.cases import DiscoveryCase
from bench.discovery.fixtures import SOURCES
from bench.discovery.refs import REFS

#: The symbolic cost axes a case may name — ``dag_cost(term, fn)``.
#: ``param_bytes`` binds the ``*_for`` closure: it carries the
#: ``(term, memo)`` CostFn signature (the raw model's second
#: parameter is ``source_tensors``, which ``backend_cost``'s
#: positional ``cost_fn(term, memo)`` call would capture), so the
#: same table doubles as the search's extraction model below.
COST_FNS: dict[str, Callable] = {
    "launch": launch_aware_cost,
    "flops": flops_cost,
    "count": count_cost,
    "exec": executor_cost_for(lowering="generic"),
    "param_bytes": param_bytes_cost_for(),
}

#: ``search()`` rules presets the corpus uses — ``"default"`` is the
#: pipeline's composed ``DEFAULT_RULES`` (``None`` to ``search``).
_RULE_PRESETS = {"default": None, "carrier_search": "carrier_search"}


def _case_rules(name: str):
    """Resolve a case's ``rules`` field to a ``search()`` argument.

    ``"default_attention"`` is the composed opt-in set —
    ``default_rules() + ATTENTION`` — the decompose-then-refuse lane:
    the ``sdpa_expand`` family + the score-scale migration rides on
    top of the default bank (the refold lives in DEFAULT already).
    """
    if name == "default_attention":
        from mlopt_core.laws import preset as _preset
        from mlopt_orchestrator.optimize import default_rules

        return default_rules() + _preset("attention")
    return _RULE_PRESETS.get(name, name)


@dataclass
class ArmResult:
    """One baseline arm's outcome for a case."""

    name: str
    kind: str
    available: bool = True
    cost: float | None = (
        None  # symbolic, case axis — None: measured-only
    )
    median_s: float | None = None  # --full lane
    iqr_s: float | None = None
    verify_passed: bool | None = None  # vs the spelled module
    verify_max_abs: float | None = None
    note: str = ""


@dataclass
class CaseResult:
    """The scorecard row for one named optimization."""

    name: str
    title: str
    category: str
    citation: str
    found: bool = False
    picked: bool = False  # op contract held on the extraction
    certified: bool = False
    verified: bool | None = None  # sink.verify on the lowered form
    verify_max_abs: float | None = None
    axis: str = "launch"
    cost_before: float | None = None
    cost_after: float | None = None
    baselines: list[ArmResult] = field(default_factory=list)
    beat_best: bool | None = None  # symbolic floor
    beat_best_measured: bool | None = None  # --full lane
    signals_seen: list[str] = field(default_factory=list)
    error: str = ""
    detail: dict = field(default_factory=dict)


def _ops_of(term: Any) -> set[str]:
    """Every op name reachable in *term*."""
    return set(_op_counts(term))


def _op_counts(term: Any) -> dict[str, int]:
    """``{op: occurrences}`` over *term*'s reachable nodes."""
    acc: dict[str, int] = {}
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Op):
            acc[t.op] = acc.get(t.op, 0) + 1
            stack.extend(t.args)
    return acc


def _call(module: Any, feed: Any) -> Any:
    """Invoke *module* on *feed* (tensor or positional-args tuple)."""
    args = feed if isinstance(feed, tuple) else (feed,)
    return module(*args)


def _term_cost(term: Any, cost_fn: Callable) -> float:
    return dag_cost(term, cost_fn)


def _module_cost(
    module: Any, feed: Any, cost_fn: Callable, source: TorchSource
) -> float | None:
    """Price a baseline module by exporting and costing its term."""
    try:
        ir, _leaves = source.to_ir(module, feed)
    except Exception:
        return None
    return _term_cost(ir.root, cost_fn)


def _signals_observed(
    case: DiscoveryCase, term: Any, stats: dict, eg: Any
) -> list[str]:
    """The ``expect.signals`` the run actually produced."""
    fires = stats.get("rule_fires") or getattr(eg, "rule_fires", {})
    term_ops = _ops_of(term)
    seen = []
    for sig in case.expect.signals:
        kind, _, key = sig.partition(":")
        if (
            (kind == "law" and fires.get(key, 0) > 0)
            or (kind == "stat" and stats.get(key))
            or (kind == "op" and key in term_ops)
        ):
            seen.append(sig)
    return seen


def _measure_arms(
    case: DiscoveryCase,
    sr: Any,
    model: Any,
    feed: Any,
    arms: dict[str, Any],
    runner: Any,
    sink: TorchSink,
    result: CaseResult,
) -> None:
    """``--full``: lower the extracted form, verify + time every arm."""
    from bench.benchkit import Case, Variant

    # The delivered form — the pipeline's own lower+verify path.
    try:
        lr = lower(sr, feed, sink=sink, verify=True)
        delivered = lr.module
        result.verified = (
            bool(lr.verified.passed) if lr.verified else None
        )
        result.verify_max_abs = (
            float(lr.verified.max_abs) if lr.verified else None
        )
    except Exception as e:  # lowering declined — honest row
        result.error = f"lower: {e}"
        result.verified = False
        delivered = None

    variants = [
        Variant(
            "mlopt",
            lambda m=delivered: _call(m, feed),
            flops=result.cost_after,
            note="certified extraction",
        )
    ]
    for arm_name, mod in arms.items():
        variants.append(
            Variant(
                arm_name,
                lambda m=mod: _call(m, feed),
            )
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
    result.detail["median_s"] = {k: v for k, v in medians.items()}
    if delivered is not None and "mlopt" in medians:
        result.detail["mlopt_ms"] = medians["mlopt"] * 1e3

    # verify each baseline module against the spelled model, then
    # attach medians to the result arms
    for arm in result.baselines:
        mod = arms.get(arm.name)
        if mod is None:
            continue
        arm.median_s = medians.get(arm.name)
        arm.iqr_s = iqrs.get(arm.name)
        try:
            rep = sink.verify(model, mod, feed)
            arm.verify_passed = bool(rep.passed)
            arm.verify_max_abs = float(rep.max_abs)
        except Exception as e:
            arm.verify_passed = False
            arm.note = f"verify: {e}" if not arm.note else arm.note

    best_measured = min(
        (
            a.median_s
            for a in result.baselines
            if a.median_s is not None
        ),
        default=None,
    )
    if best_measured is not None and delivered is not None:
        result.beat_best_measured = medians["mlopt"] < best_measured


def run_case(
    case: DiscoveryCase,
    *,
    lane: str = "fast",
    device: str = "cpu",
    min_run_time: float = 0.05,
    warmup: int = 3,
) -> CaseResult:
    """Evaluate one case; ``lane`` is ``"fast"`` or ``"full"``."""
    res = CaseResult(
        name=case.name,
        title=case.title,
        category=case.category,
        citation=case.citation,
        axis=case.cost,
    )
    try:
        model, feed = _build_fixture(case)
    except Exception as e:
        res.error = f"fixture: {e}"
        return res
    sink = TorchSink()
    source = TorchSource()
    cost_fn = COST_FNS[case.cost]
    rules = _case_rules(case.rules)

    t0 = time.time()
    try:
        sr = search(
            model,
            feed,
            source=source,
            capabilities=sink,
            cost_fn=cost_fn,
            rules=rules,
            max_iterations=case.max_iterations,
            max_enodes=case.max_enodes,
        )
    except Exception as e:
        res.error = f"search: {e}"
        return res
    res.detail["search_s"] = round(time.time() - t0, 3)
    res.detail["n_enodes"] = sr.stats.get("n_enodes")
    res.detail["iterations"] = sr.stats.get("iterations")

    # -- derivability ------------------------------------------------------
    term_ops = _ops_of(sr.term)
    counts = _op_counts(sr.term)
    res.picked = (
        set(case.expect.ops) <= term_ops
        and not (set(case.expect.absent_ops) & term_ops)
        and all(
            counts.get(op, 0) <= cap
            for op, cap in case.expect.max_ops.items()
        )
    )
    res.signals_seen = _signals_observed(case, sr.term, sr.stats, sr.eg)
    res.found = bool(res.signals_seen) and res.picked
    res.detail["extracted_ops"] = sorted(term_ops)

    # -- certification ------------------------------------------------------
    try:
        cert = sr.eg.certificate(sr.ir.root, sr.term)
        replayed = verify_certificate(sr.ir.root, cert)
        res.certified = True
        res.detail["cert_steps"] = cert.stats.get("n_steps")
        res.detail["cert_egraph_dep"] = cert.stats.get(
            "n_egraph_dependent"
        )
        del replayed
    except Exception as e:
        res.certified = False
        res.detail["cert_error"] = str(e)

    # -- symbolic cost -------------------------------------------------------
    res.cost_before = _term_cost(sr.ir.root, cost_fn)
    res.cost_after = _term_cost(sr.term, cost_fn)

    # -- baselines -----------------------------------------------------------
    arms: dict[str, Any] = {}
    for b in case.baselines:
        arm = ArmResult(name=b.name, kind=b.kind, note=b.note)
        if b.kind == "self":
            arm.cost = res.cost_before
            arms[b.name] = model
        elif b.kind == "compile":
            arm.available = lane == "full"
            arm.note = b.note or "measured-only (no symbolic form)"
            if lane == "full":
                arms[b.name] = torch.compile(model)
        elif b.kind == "external":
            arm.available = False
            arm.note = b.note or "not in this repository"
        elif b.kind == "ref":
            try:
                mod = REFS[b.ref](model, feed)
                with torch.no_grad():
                    _call(mod, feed)  # smoke — a broken ref is no floor
                arms[b.name] = mod
                arm.cost = _module_cost(mod, feed, cost_fn, source)
            except Exception as e:
                arm.available = False
                arm.note = f"ref build: {e}"
        res.baselines.append(arm)

    floors = [a.cost for a in res.baselines if a.cost is not None]
    if floors:
        res.beat_best = res.cost_after < min(floors)

    # -- measured pay ---------------------------------------------------------
    if lane == "full":
        from bench.benchkit import Runner

        runner = Runner(
            device=device, warmup=warmup, min_run_time=min_run_time
        )
        _measure_arms(case, sr, model, feed, arms, runner, sink, res)
    return res


def _build_fixture(case: DiscoveryCase):
    """Build ``(model, feed)`` for *case* under its seed."""
    if case.source not in SOURCES:
        raise KeyError(
            f"unknown source {case.source!r} — known: {sorted(SOURCES)}"
        )
    torch.manual_seed(case.seed)
    return SOURCES[case.source](case.size)


def run_suite(
    cases: list[DiscoveryCase],
    *,
    lane: str = "fast",
    device: str = "cpu",
    min_run_time: float = 0.05,
    warmup: int = 3,
    progress: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    """Run every case in order; ``progress`` fires per row."""
    rows = []
    for case in cases:
        row = run_case(
            case,
            lane=lane,
            device=device,
            min_run_time=min_run_time,
            warmup=warmup,
        )
        rows.append(row)
        if progress is not None:
            progress(row)
    return rows

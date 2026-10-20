"""The referee registry — every strictness arm is a folder under ``referees/``.

Plan 0032 phase 2 lane B ("systematic eval"): a referee is a folder in
this package — one measured record in its docstring, its
:data:`RefereeSpec` row, and the component set it runs.  The registry
maps *name → spec* so the matrix runner (:mod:`bench.matrix`)
enumerates judges from one table — the same shape
:mod:`bench.limits.registry` gives the baseline roster.

A :class:`RefereeSpec` is *data*: which gauntlet components the arm
runs (a subset of :data:`COMPONENT_ORDER`, cumulative along the
numeric-oracle → typed-pay → adversarial-sweep → cert-replay dial),
the :class:`~mlopt_discovery.scoring.ScoringSpec` the measured axes
are gated against (``None`` = no value function — the component
verdicts alone decide), and the measured record that earned the arm
its seat.  Loading is lazy — :func:`spec` imports the folders on
first touch, so ``import bench.referees`` stays cheap.

The judging contract: :func:`judge` runs the spec's components in
canonical order through :mod:`bench.referees.components`, stops at
the first refusal (the gauntlet's own convention — a failing gate
ends the run), then scores the collected measurement against the
spec's :class:`ScoringSpec`.  A referee that cannot judge this claim
or environment raises :class:`SkipReferee` with the reason — the row
reports the skip, never a silent drop.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from mlopt_discovery.scoring import ScoringSpec, score

__all__ = [
    "COMPONENTS",
    "COMPONENT_ORDER",
    "ComponentVerdict",
    "RefereeCtx",
    "RefereeSpec",
    "RefereeVerdict",
    "SkipReferee",
    "availability",
    "judge",
    "names",
    "spec",
    "specs",
]


class SkipReferee(Exception):
    """A referee declining a claim — the row records *why*."""


#: The component vocabulary, in gauntlet order.  A spec's
#: ``components`` is a subset; :func:`judge` always runs them in this
#: order, so a spec cannot silently reorder the gauntlet.
COMPONENT_ORDER: tuple[str, ...] = ("oracle", "pay", "sweep", "cert")

#: The set form, for validation.
COMPONENTS: frozenset[str] = frozenset(COMPONENT_ORDER)


@dataclass(frozen=True)
class RefereeSpec:
    """One referee's registration — its seat on the roster.

    *components* is the strictness dial: the gauntlet legs the arm
    runs, a subset of :data:`COMPONENT_ORDER` — ``()`` is the trust
    arm (``naive``), the full tuple is ``full``.  *spec* is the value
    function: a :class:`ScoringSpec` gating the axes the ``pay``
    component measures (``None`` = no scoring — component verdicts
    decide alone).  *requires* names modules that must import for the
    arm to exist (a scratch dep reports honestly); *record* is the
    one-line measured record the folder docstring argues at length.
    """

    name: str
    components: tuple[str, ...] = ()
    spec: ScoringSpec | None = None
    title: str = ""
    note: str = ""
    record: str = ""
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the component subset at declaration."""
        object.__setattr__(self, "components", tuple(self.components))
        bad = [c for c in self.components if c not in COMPONENTS]
        if bad:
            raise ValueError(
                f"referee {self.name!r}: unknown component(s) "
                f"{bad} — expected a subset of {COMPONENT_ORDER}"
            )


@dataclass
class RefereeCtx:
    """Everything a component may read — the measurement context.

    The same inputs the gauntlet's corpus carries, shrunk to what a
    claim-level referee needs: ``probe`` (``TermCase`` list) is the
    firing/typed-pay domain, ``real_terms`` the real-match domain the
    oracle and sweep enumerate, ``base_rules`` the axiom set
    ``verify_law`` derives under (``None`` = the shipped bank),
    ``sink``/``cost_fn`` the lowering + pricing backend, ``axis_fns``
    the ``{axis: dag_cost fn}`` table the ``pay`` leg measures
    (``None`` = the shipped five-axis table).  ``oracle_limit`` /
    ``sweep_limit`` bound the synthesized binding windows — the
    oracle's 360-site window is the auto_cond-era measurement domain;
    the sweep's widened bank + deeper cap is the adversarial leg.
    ``sweep_envs`` is the env-draw count a *concrete* claim's sweep
    evaluates (no binding domain to enumerate).  ``probe_instances``
    caps the instance terms a pattern claim's pay leg probes.
    """

    probe: tuple = ()
    real_terms: tuple = ()
    base_rules: tuple | None = None
    sink: Any = None
    cost_fn: Any = None
    axis_fns: Mapping[str, Callable] | None = None
    oracle_limit: int = 360
    sweep_limit: int = 2_000
    sweep_envs: int = 8
    probe_instances: int = 4
    device: str = "cpu"


@dataclass(frozen=True)
class ComponentVerdict:
    """One component's verdict: the leg name, pass/fail, evidence."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class RefereeVerdict:
    """The referee's verdict on one claim.

    ``passed`` is the conjunction of the component stages *and* the
    scoring spec's gates; ``value`` is the spec's blended score
    (``nan`` when the spec carries no weightable measurement, or no
    spec at all); ``measured`` is the ``{axis: value}`` map the
    ``pay`` leg priced; ``missing`` names the axes the spec asked
    for that were never measured; ``skipped`` records a
    :class:`SkipReferee` refusal when the referee declined — a
    declined claim is not a failed claim.
    """

    claim: str
    referee: str
    passed: bool
    value: float = math.nan
    measured: dict[str, float] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    stages: tuple[ComponentVerdict, ...] = ()
    skipped: str = ""

    @property
    def reason(self) -> str:
        """The first failing stage, the skip, or the cleared verdict."""
        if self.skipped:
            return f"skipped: {self.skipped}"
        for s in self.stages:
            if not s.passed:
                return f"{s.name}: {s.detail}"
        return "cleared" if self.stages else "no components ran"


#: Module key → the folder's module path.  The folder itself declares
#: its ``SPEC``; this table only names where to look, so :func:`names`
#: and error messages never pay an import beyond the cheap folder
#: bodies.
_MODULES: dict[str, str] = {
    "naive": "bench.referees.naive",
    "oracle": "bench.referees.oracle",
    "pay": "bench.referees.pay",
    "sweep": "bench.referees.sweep",
    "full": "bench.referees.full",
}

#: spec name → (module path, RefereeSpec) — built by :func:`specs`.
_INDEX: dict[str, tuple[str, RefereeSpec]] = {}


def _load_specs() -> None:
    """Import every referee folder once; index its spec by name."""
    for mod_name in _MODULES.values():
        mod = importlib.import_module(mod_name)
        s = mod.SPEC
        _INDEX[s.name] = (mod_name, s)


def specs() -> dict[str, RefereeSpec]:
    """Return ``{name: RefereeSpec}`` over the registered roster."""
    if not _INDEX:
        _load_specs()
    return {name: s for name, (_m, s) in _INDEX.items()}


def names() -> tuple[str, ...]:
    """Return the registered referee names, sorted."""
    specs()
    return tuple(sorted(_INDEX))


def _known(name: str) -> ValueError:
    """Build the unknown-referee error, with the roster in it."""
    return ValueError(
        f"unknown referee {name!r} — registered: {', '.join(names())}"
    )


def spec(name: str) -> RefereeSpec:
    """Return the named referee's spec; raise ``ValueError`` if absent."""
    specs()
    if name not in _INDEX:
        raise _known(name)
    return _INDEX[name][1]


def availability(name: str, *, device: str = "cpu") -> str | None:
    """``None`` when the referee can run, else the reason it cannot."""
    s = spec(name)
    for mod in s.requires:
        if importlib.util.find_spec(mod) is None:
            return f"{mod} not installed (scratch dep)"
    if device.startswith("cuda") is False and getattr(
        s, "needs_cuda", False
    ):
        return "requires CUDA — pass --device cuda"
    return None


def judge(
    name: str,
    claim: Any,
    ctx: RefereeCtx | None = None,
    *,
    evidence: Any = None,
) -> RefereeVerdict:
    """Judge *claim* (a ``Rewrite``) under referee *name*.

    Runs the spec's components in canonical order, stops at the first
    refusal, then scores the ``pay`` leg's measurement against the
    spec's :class:`ScoringSpec` when it carries one.  *evidence* is
    the shared :class:`~bench.referees.components.ClaimEvidence` — a
    matrix run passes the same per-claim object to every referee so a
    measured sweep is never re-paid (the memoized-verdict seam).
    Raises :class:`SkipReferee` when the referee cannot run here —
    callers map that to a declined verdict, never a crash.
    """
    from bench.referees import components

    s = spec(name)
    reason = availability(name, device=(ctx or RefereeCtx()).device)
    if reason is not None:
        raise SkipReferee(reason)
    ctx = RefereeCtx() if ctx is None else ctx
    evd = (
        evidence
        if evidence is not None
        else components.ClaimEvidence(claim, ctx)
    )
    stages: list[ComponentVerdict] = []
    for comp in COMPONENT_ORDER:
        if comp not in s.components:
            continue
        v = components.run_component(comp, claim, ctx, evd)
        stages.append(v)
        if not v.passed:
            break
    measured = dict(evd.measured)
    verdict = RefereeVerdict(
        claim=claim.name,
        referee=name,
        passed=(bool(stages) and all(v.passed for v in stages))
        or (not stages and s.components == ()),
        measured=measured,
        stages=tuple(stages),
    )
    if s.spec is not None:
        sv = score(s.spec, measured)
        verdict.missing = sv.missing
        verdict.value = sv.value
        verdict.passed = verdict.passed and sv.passed
    return verdict

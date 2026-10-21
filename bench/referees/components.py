"""The referee components — the gauntlet's legs, as shared machinery.

Each component is one function ``(claim, ctx, evd) -> ComponentVerdict``
over a claimed equality (a :class:`~mlopt_core.egraph.Rewrite` — a
pattern claim with optional guard hooks, or a concrete term pair).
They are *consumers* of the shipped machinery — the numeric oracle
(:mod:`mlopt_discovery.oracle` tri-state eval over the view-oracle's
binding enumeration), the typed-pay probe
(:mod:`mlopt_discovery.impact`), the adversarial sweep
(:mod:`mlopt_discovery.evidence`'s guarded-region evals over a
widened binding bank) and the certificate leg
(:func:`mlopt_discovery.verifier.verify_law`) — never rewrites of it.

:class:`ClaimEvidence` is the memoization seam: the per-claim
artifacts (the real-match region, the synthesized sweeps, the
accepted instances, the measured axes) are computed once per claim
per context and shared across every component — and across every
referee a matrix cell runs — so the expensive legs are never repaid.

The sweep's leaf bank is deliberately *wider* than the oracle
window's: the shipped ``_leaf_bindings`` bank (the ``360``-env
window auto_cond measured in) provably never produces the
unequal-rank mid-insert corner — the domain in which the grid-eq
``mul_unsq_nat`` guard's 212 bad accepts live.  The sweep feeds the
same :func:`mlopt_discovery.oracle._binding_envs` enumeration an
adversarial leaf bank (the retro's domain: ranks ≤ 3, extents
{1, 2, 3}) — same machinery, deeper domain.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any

from mlopt_core.cost import (
    count_cost,
    dag_cost,
    executor_cost_for,
    flops_cost,
    launch_aware_cost,
    param_bytes_cost_for,
)
from mlopt_core.egraph.terms import _term_instantiate, _term_match
from mlopt_core.ir import Const, TensorType, Var
from mlopt_core.meta import pattern_metavars

from bench.referees.registry import ComponentVerdict, RefereeCtx

#: Saturation budget for a firing probe — ``impact``'s own knobs.
_FIRING_ITERS = 4
_FIRING_NODES = 40_000
_RTOL = 1e-4

#: The default ``{axis: dag_cost fn}`` table the ``pay`` leg prices —
#: the symbolic axes, rendered as ``after / before`` cost ratios
#: (lower is better).  ``speed`` / ``memory`` (the measured lanes)
#: are absent on purpose: a spec gating them reports ``missing``
#: honestly rather than being priced off a symbolic proxy.
DEFAULT_AXIS_FNS: dict[str, Any] = {
    "launch": launch_aware_cost,
    "flops": flops_cost,
    "count": count_cost,
    "exec": executor_cost_for(lowering="generic"),
    "params": param_bytes_cost_for(),
}

#: The adversarial sweep's leaf bank — the retro's domain (ranks ≤ 3,
#: extents {1, 2, 3}), led by the unequal-rank corners the shipped
#: bank never enumerates.  The scalar corner rides the ``derived``
#: channel like the stock bank.
_ADVERSARIAL_CORNERS: tuple[tuple[int, ...], ...] = (
    (2,),
    (1, 2, 2),
    (1, 2),
    (2, 1),
    (1, 2, 1),
    (2, 1, 2),
    (1, 1, 2, 2),
    (3,),
    (1, 3),
    (3, 2),
)
_ADVERSARIAL_GRID: tuple[tuple[int, ...], ...] = tuple(
    s for r in range(4) for s in itertools.product((1, 2, 3), repeat=r)
)
ADVERSARIAL_SHAPES: tuple[tuple[int, ...], ...] = (
    _ADVERSARIAL_CORNERS
    + tuple(
        s for s in _ADVERSARIAL_GRID if s not in _ADVERSARIAL_CORNERS
    )
)


def _is_pattern(claim: Any) -> bool:
    """Whether *claim* is a pattern (bindable metavars) or concrete."""
    return bool(
        pattern_metavars(claim.lhs) | pattern_metavars(claim.rhs)
    )


def _leaves_of(term: Any) -> list[Any]:
    """Every ``Var`` leaf of *term*, deduped, deterministic order."""
    from mlopt_discovery.proposal import _leaves

    return sorted(_leaves(term), key=lambda v: v.name)


def _instance_case(name: str, lhs_i: Any) -> Any:
    """A ``TermCase`` over one instantiated site — fp64 feed +1.0.

    Mirrors ``impact._synthetic_case``: the claim's own site is the
    term it *should* fire on; the ``+1.0`` offset keeps zeros out of
    the feed so a law whose truth is multiplicative still evaluates.
    """
    import torch
    from mlopt_discovery.impact import TermCase

    inps = tuple(_leaves_of(lhs_i))
    return TermCase(
        source="referee",
        name=name,
        term=lhs_i,
        inputs=inps,
        feed=tuple(
            torch.randn(tuple(v.typ.shape), dtype=torch.float64) + 1.0
            for v in inps
        ),
        param_vals={},
    )


def _adversarial_bank(
    mv: str,
    parents: set[str],
    derived: Iterable,
    extra_viewed: tuple = (),
) -> list[Any]:
    """The sweep's leaf bank — the retro's domain through the same API.

    A drop-in ``source`` for :func:`mlopt_discovery.oracle._binding_envs`:
    the same ``(mv, parents, derived, extra_viewed)`` contract
    :func:`oracle._leaf_bindings` implements, over a wider shape grid
    so the enumeration reaches the unequal-rank mid-insert corner the
    shipped window provably misses.
    """
    from mlopt_discovery import oracle as lvo

    out: list[Any] = []
    if "getitem" in parents:
        out.extend(lvo._tuple_sources(mv))
    for s in itertools.chain(derived, ADVERSARIAL_SHAPES, extra_viewed):
        out.append(Var(mv, TensorType(tuple(s))))
    if not parents or all(p not in lvo._VIEWISH for p in parents):
        out.insert(1, Var(mv, TensorType(())))
        out.insert(1, Const(0.5))
        lvo._insert_op_consts(out, parents)
    return out


def _sites(
    lhs_pat: Any, rhs_pat: Any, *, limit: int, source: Any = None
) -> Iterable:
    """``evidence._synth_sites`` over an optional binding ``source``.

    The shipped helper hardwires the leaf bank; this is the same
    dedup loop with the source forwarded, so the sweep can enumerate
    the adversarial domain through the identical machinery.
    """
    from mlopt_discovery import oracle as lvo

    seen: set = set()
    for full in lvo._binding_envs(lhs_pat, rhs_pat, source):
        try:
            lhs_i = _term_instantiate(lhs_pat, full)
            rhs_i = _term_instantiate(rhs_pat, full)
        except Exception:
            continue
        if (lhs_i, rhs_i) in seen:
            continue
        seen.add((lhs_i, rhs_i))
        yield full, lhs_i
        if len(seen) >= limit:
            return


def _real_sites(claim: Any, real_terms: Iterable) -> list:
    """``(subst, matched-term)`` pairs over the real-match domain."""
    return [
        (s, m)
        for m in real_terms
        if (s := _term_match(claim.lhs, m)) is not None
    ]


def _rhs_inst(claim: Any, subst: dict) -> Any:
    """Instantiate the claim's RHS under *subst* (+ ``derive``), or None."""
    inst = dict(subst)
    if claim.derive is not None:
        try:
            extra = claim.derive(subst)
        except Exception:
            return None
        if extra is None:
            return None
        inst.update(extra)
    try:
        return _term_instantiate(claim.rhs, inst)
    except Exception:
        return None


class ClaimEvidence:
    """Per-claim measurement artifacts — computed once, shared.

    The memoized-verdict seam a matrix run exploits: the real-match
    region, the synthesized sweeps (per ``(limit, adversarial)``
    window), the accepted sites' instantiated pairs and the
    ``pay`` leg's measured axes are computed lazily and shared
    across components *and* across referees judging the same claim.
    """

    def __init__(self, claim: Any, ctx: RefereeCtx) -> None:
        """Bind *claim* to the measurement context."""
        self.claim = claim
        self.ctx = ctx
        self.pattern = _is_pattern(claim)
        self._real_sites: list | None = None
        self._real_region: Any = None
        self._synth_regions: dict[tuple, Any] = {}
        self._instances: list | None = None
        self.measured: dict[str, float] = {}

    def real_region(self) -> Any:
        """The guarded-region evals over ``ctx.real_terms``."""
        from mlopt_discovery import evidence as ev

        if self._real_region is None:
            if self._real_sites is None:
                self._real_sites = _real_sites(
                    self.claim, self.ctx.real_terms
                )
            self._real_region = ev._guarded_evals(
                self.claim, iter(self._real_sites)
            )
        return self._real_region

    def synth_region(self, limit: int, *, adversarial: bool) -> Any:
        """The guarded-region evals over the synthesized domain.

        ``adversarial`` swaps the leaf bank for the widened grid —
        the same enumeration, a deeper window.
        """
        from mlopt_discovery import evidence as ev

        key = (limit, adversarial)
        if key not in self._synth_regions:
            sites = _sites(
                self.claim.lhs,
                self.claim.rhs,
                limit=limit,
                source=_adversarial_bank if adversarial else None,
            )
            self._synth_regions[key] = ev._guarded_evals(
                self.claim, sites
            )
        return self._synth_regions[key]

    def instances(self, cap: int) -> list[tuple[Any, Any]]:
        """Concrete ``(lhs_i, rhs_i)`` pairs at accepted sites.

        The accepted region of the stock-window synth sweep plus the
        real matches — the sites the claim's own guard admits, each
        instantiated both ways.  Capped; the pay/cert legs probe a
        prefix.
        """

        if self._instances is not None:
            return self._instances[:cap]
        out: list[tuple[Any, Any]] = []
        if not self.pattern:
            out.append((self.claim.lhs, self.claim.rhs))
        else:
            for subst, lhs_i in [
                *_sites(
                    self.claim.lhs,
                    self.claim.rhs,
                    limit=self.ctx.oracle_limit,
                ),
                *self.real_sites(),
            ]:
                try:
                    ok = self.claim.check is None or bool(
                        self.claim.check(subst)
                    )
                except Exception:
                    continue
                if not ok:
                    continue
                rhs_i = _rhs_inst(self.claim, subst)
                if rhs_i is not None:
                    out.append((lhs_i, rhs_i))
                if len(out) >= cap:
                    break
            # Fall back to the generic-instance path for claims the
            # enumeration cannot bind (attr-free leaf spellings).
            if not out:
                from mlopt_discovery.verifier import generic_instance

                inst = generic_instance(self.claim)
                if inst is not None:
                    out.append(inst)
        self._instances = out
        return out[:cap]

    def real_sites(self) -> list:
        """The ``(subst, lhs_i)`` real-match sites."""
        if self._real_sites is None:
            self._real_sites = _real_sites(
                self.claim, self.ctx.real_terms
            )
        return self._real_sites


def _cost_fn(ctx: RefereeCtx) -> Any:
    """The extraction model — the launch axis by default."""
    return ctx.cost_fn if ctx.cost_fn is not None else launch_aware_cost


def _axis_fns(ctx: RefereeCtx) -> dict[str, Any]:
    return dict(ctx.axis_fns or DEFAULT_AXIS_FNS)


def _region_detail(tag: str, r: Any) -> str:
    """One region's counts, rendered for a stage detail."""
    return (
        f"{tag}: {r.equal}eq/{r.unequal}ne/{r.rhs_err}rerr "
        f"({r.accepted} acc, {r.declined} dec, {r.envs} envs)"
    )


# ---------------------------------------------------------------------------
#  oracle — the numeric oracle over the accepted region
# ---------------------------------------------------------------------------


def _oracle(claim: Any, ctx: RefereeCtx, evd: ClaimEvidence) -> Any:
    """Numeric truth on the claim's accepted region, stock window.

    Concrete claim: one fp64 eval decides.  Pattern claim: the
    guarded-region evals over real matches + the shipped synthesized
    window — the auto_cond-era measurement domain.  Passes iff no
    measured counterexample and at least one equal site (a vacuous
    acceptance is unproven, not verified).
    """
    from mlopt_discovery import evidence as ev
    from mlopt_discovery import oracle as lvo

    if not evd.pattern:
        outcome, note = lvo.eval_instance(claim.lhs, claim.rhs)
        ok = outcome == "equal"
        return ComponentVerdict(
            "oracle",
            ok,
            f"{outcome}" + (f" — {note}" if not ok and note else ""),
        )
    real = evd.real_region()
    synth = evd.synth_region(ctx.oracle_limit, adversarial=False)
    clean = ev._region_clean(real) and ev._region_clean(synth)
    equal = real.equal + synth.equal
    ok = clean and equal >= 1
    detail = (
        _region_detail("real", real)
        + " | "
        + _region_detail("synth", synth)
    )
    if not equal:
        detail += " — no equal site measured"
    return ComponentVerdict("oracle", ok, detail)


# ---------------------------------------------------------------------------
#  pay — fires, well-typed, verifies, and strictly lowers cost
# ---------------------------------------------------------------------------


def _probe_case(
    case: Any, claim: Any, ctx: RefereeCtx
) -> tuple[int, Any, bool | None, str]:
    """Fire *claim* alone on *case*; return ``(fires, best, verified, note)``.

    ``impact._probe``'s shape, kept close — the one difference is the
    extracted member rides home so the caller can price it on every
    scoring axis, not just the extraction model's.
    """
    from mlopt_core.egraph import EGraph
    from mlopt_discovery.impact import _ir_of
    from mlopt_orchestrator.optimize import _lower_extracted

    sink = ctx.sink
    eg = EGraph()
    root = eg.add_term(case.term)
    eg.run(
        [claim],
        root,
        max_iterations=_FIRING_ITERS,
        max_nodes=_FIRING_NODES,
    )
    fires = eg.rule_fires.get(claim.name, 0)
    if not fires:
        return 0, None, None, ""
    best = eg.extract_best(root, _cost_fn(ctx))
    if best is None:
        return fires, None, None, "extraction returned no member"
    if best == case.term or sink is None:
        return fires, best, None, ""
    try:
        before = _lower_extracted(
            case.term,
            _ir_of(case.term, case.inputs),
            case.param_vals,
            sink,
        )
        after = _lower_extracted(
            best, _ir_of(best, case.inputs), case.param_vals, sink
        )
        vr = sink.verify(before, after, case.feed, rtol=_RTOL)
    except Exception as e:
        return fires, best, False, f"{type(e).__name__}: {e}"
    return (
        fires,
        best,
        bool(vr.passed),
        (
            ""
            if vr.passed
            else f"lowered terms differ (max_rel={vr.max_rel:.2e})"
        ),
    )


def _pay(claim: Any, ctx: RefereeCtx, evd: ClaimEvidence) -> Any:
    """The typed-pay leg: fire, verify, price the extraction.

    Probes ``ctx.probe`` plus TermCases minted from the claim's own
    accepted instances.  A changed extraction must verify through the
    sink; the measured axes are the mean ``after / before`` ratios
    over changed cases — the ``{axis: value}`` map the spec scores.
    """
    cost_fn = _cost_fn(ctx)
    axis_fns = _axis_fns(ctx)
    cases = list(ctx.probe)
    cases += [
        _instance_case(f"{claim.name}#{i}", lhs_i)
        for i, (lhs_i, _r) in enumerate(
            evd.instances(ctx.probe_instances)
        )
    ]
    if not cases:
        return ComponentVerdict(
            "pay", False, "no probe cases — nothing to fire on"
        )
    fires = changed = paid = vfail = 0
    ratios: dict[str, list[float]] = {a: [] for a in axis_fns}
    for case in cases:
        f, best, verified, _note = _probe_case(case, claim, ctx)
        fires += f
        if best is None or best == case.term:
            continue
        changed += 1
        if verified is False:
            vfail += 1
        if dag_cost(best, cost_fn) < dag_cost(case.term, cost_fn):
            paid += 1
        for axis, fn in axis_fns.items():
            before = dag_cost(case.term, fn)
            if before > 0:
                ratios[axis].append(dag_cost(best, fn) / before)
    evd.measured.update(
        {axis: sum(rs) / len(rs) for axis, rs in ratios.items() if rs}
    )
    ok = fires > 0 and paid > 0 and vfail == 0
    detail = (
        f"fires={fires} changed={changed} paid={paid} "
        f"verify_fail={vfail} cases={len(cases)}"
    )
    if evd.measured:
        detail += " | " + " ".join(
            f"{a}={v:.3f}x" for a, v in sorted(evd.measured.items())
        )
    return ComponentVerdict("pay", ok, detail)


# ---------------------------------------------------------------------------
#  sweep — the adversarial guarded-region enumeration
# ---------------------------------------------------------------------------


def _sweep(claim: Any, ctx: RefereeCtx, evd: ClaimEvidence) -> Any:
    """The adversarial leg: the guarded region over the widened domain.

    Concrete claim: ``ctx.sweep_envs`` env draws (a concrete pair has
    no binding domain — the sweep is over the *env*, the only
    adversary left).  Pattern claim: the guarded-region evals over
    the widened leaf bank at ``ctx.sweep_limit``, with the shipped
    selective-cap escalation for a starved guarded window — plus the
    real-match region.  A measured counterexample (``unequal`` /
    ``rhs_err`` / ``guard_err``) refuses; at least one equal site
    must exist.
    """
    import torch
    from mlopt_discovery import evidence as ev
    from mlopt_discovery import oracle as lvo

    if not evd.pattern:
        outcomes = []
        for i in range(ctx.sweep_envs):
            torch.manual_seed(10_019 + i)
            outcomes.append(lvo.eval_instance(claim.lhs, claim.rhs)[0])
        bad = [o for o in outcomes if o != "equal"]
        return ComponentVerdict(
            "sweep",
            not bad and bool(outcomes),
            f"{len(outcomes) - len(bad)}/{len(outcomes)} equal envs"
            + (f" — {bad[0]}" if bad else ""),
        )
    synth = evd.synth_region(ctx.sweep_limit, adversarial=True)
    eff = lvo.escalate_limit(
        ctx.sweep_limit,
        guarded=claim.check is not None or claim.cond is not None,
        accepted=synth.accepted,
    )
    if eff > ctx.sweep_limit:
        from dataclasses import replace

        wider = evd.synth_region(eff, adversarial=True)
        synth = replace(wider, envs=synth.envs + wider.envs)
    real = evd.real_region()
    clean = ev._region_clean(synth) and ev._region_clean(real)
    equal = synth.equal + real.equal
    ok = clean and equal >= 1
    detail = (
        _region_detail("synth", synth)
        + " | "
        + _region_detail("real", real)
    )
    if synth.counterexample:
        detail += f" — cex: {synth.counterexample[:60]}"
    return ComponentVerdict("sweep", ok, detail)


# ---------------------------------------------------------------------------
#  cert — verify_law derivability + standalone certificate replay
# ---------------------------------------------------------------------------


def _cert(claim: Any, ctx: RefereeCtx, evd: ClaimEvidence) -> Any:
    """The certificate leg: a standalone derivation must exist.

    Concrete claims replay the pair directly; pattern claims replay
    the first accepted instance (the same witness ``verify_law``
    would mint).  Passes only on ``derivable`` *and* ``replayable``
    — an e-graph-dependent witness is not a rule derivation.
    """
    from mlopt_discovery.verifier import verify_law

    inst = evd.instances(1)
    if not inst:
        return ComponentVerdict(
            "cert", False, "no accepted instance to replay"
        )
    rules = ctx.base_rules
    if rules is None:
        from mlopt_core.laws import ALL_RULES

        rules = tuple(ALL_RULES)
    lhs_i, rhs_i = inst[0]
    res = verify_law(lhs_i, rhs_i, list(rules))
    ok = res.derivable and res.replayable
    if ok:
        detail = (
            f"{res.direction} {res.witness_steps} steps "
            f"({', '.join(res.witness_rules[:4])})"
        )
    elif res.derivable:
        detail = (
            f"{res.direction}: {res.note or 'witness not replayable'}"
        )
    else:
        detail = f"not derivable ({res.stop})"
    return ComponentVerdict("cert", ok, detail)


#: Component name → implementation.  The registry's canonical order
#: decides sequencing; this table decides *what runs*.
_COMPONENTS: dict[str, Any] = {
    "oracle": _oracle,
    "pay": _pay,
    "sweep": _sweep,
    "cert": _cert,
}


def run_component(
    name: str, claim: Any, ctx: RefereeCtx, evd: ClaimEvidence
) -> Any:
    """Run one component; unknown names raise ``ValueError``."""
    if name not in _COMPONENTS:
        raise ValueError(
            f"unknown component {name!r} — known: "
            f"{', '.join(_COMPONENTS)}"
        )
    return _COMPONENTS[name](claim, ctx, evd)

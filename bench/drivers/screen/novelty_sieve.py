"""The novelty sieve — three gates between a minted law and review.

A minted ``Rewrite`` is cheap; human attention is not.  This driver is
the automatic filter that decides which candidates are worth a
reviewer: a candidate is a **novelty candidate** iff it passes three
gates —

1. **cert** — the machine certifies the law itself: the candidate's
   own rule derives its instance sides with a *standalone-replayable*
   certificate (``mlopt_discovery.verifier.verify_law`` +
   ``verify_certificate`` — reused, not rebuilt).  A rule whose own
   rewrite cannot carry a replayable witness fails here.
2. **measured** — the candidate *pays*: on at least one
   ``bench/discovery`` case the certified extraction under
   ``case rules + {candidate}`` lands **strictly below** the best
   registered baseline (the ``beat_best`` floor — a tie with the
   hand-known impl is a rediscovery, not a find), AND the derivation
   certificate actually invokes the candidate (a win the rule never
   touched is not the rule's win).
3. **derivable** — the law is NOT a theorem of the known bank: the
   candidate's instance is saturated under the bank with the candidate
   — and every alpha-sibling (the same ``(lhs, rhs)`` law under
   metavariable renaming, keyed by ``laws.serialize.alpha_key``) —
   *removed*.  ``derivable`` means the bank already proves it: the law
   adds no theorem, and is demoted silently.  An inverse spelling is a
   *different* law and stays in the premises — a bank holding both
   directions honestly knows the equation.

The bank defaults to the **widest shipped surface** —
``default_rules()`` + ``ALL_RULES_WITH_LAYOUT`` + ``ATTENTION`` + the
scan monoids — so derivability is tested against everything the
library knows, not just the DEFAULT subset: the strictest demotion
criterion, the fewest false novelty claims.

Verdicts: ``review`` (all three gates — enqueued for a human),
``known`` (bank-derivable), ``refuted`` (the fp64 oracle measured the
instance unequal), ``uncertified``, ``unpaid`` (underivable but no
case win), ``unmeasured`` (underivable but no case exercises it),
``no-instance`` (no binding produced a site — untestable, reported
honestly).

Instance sources, in order: the ``law_bench`` registry /
``generic_instance`` tables (:func:`verifier.instance_of`), a strict
LHS match inside a spelled discovery-corpus module, the gauntlet's
own synthesized binding domain (``evidence._synth_sites`` +
``apply_rewrite_at`` — the enumeration that admitted the unsq-pad
family), and finally a small uniform-shape metavariable binding for
patterns outside every registered domain.

Run::

    .venv/bin/python bench/drivers/screen/novelty_sieve.py
    .venv/bin/python bench/drivers/screen/novelty_sieve.py --names softsign_fold,cand_exp_factor
    .venv/bin/python bench/drivers/screen/novelty_sieve.py --no-measure --json /tmp/sieve.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlopt_core.ir import Const, Op, TensorType, Var, op_repr
from mlopt_core.laws import (
    ALL_RULES_WITH_LAYOUT,
    ATTENTION_RULES,
    PRIOR_ART,
    SCAN_DIAG_LAWS,
    SCAN_LAWS,
    tags,
)
from mlopt_core.laws.provenance import MACHINE_SHIPPED, law_provenance
from mlopt_core.laws.serialize import alpha_key
from mlopt_core.meta import (
    _positions,
    _subterm,
    apply_rewrite_at,
    instantiate_pattern,
    match_pattern,
    pattern_metavars,
)
from mlopt_discovery import REPO_ROOT, lawdata, verifier
from mlopt_discovery.verifier import verify_law
from mlopt_orchestrator.optimize import _resolve_rules, default_rules

__all__ = [
    "BeatRow",
    "SieveRow",
    "case_contexts",
    "instance_of",
    "known_bank",
    "main",
    "measure_gate",
    "render_markdown",
    "shipped_candidates",
    "sieve",
    "sieve_all",
]

#: Where the review-queue document lands.
DOC_PATH = REPO_ROOT / "docs" / "results_novelty_sieve.md"

#: Cap on the synthesized-binding enumeration per law
#: (``evidence._synth_sites`` window — same order as the gauntlet's).
_SYNTH_LIMIT = 400

#: Per-candidate cap on measured cases — a cheap law whose LHS is a
#: common pattern (``add``/``mul``) matches most of the corpus; the
#: signal-named cases are tried first, then strict-match cases in name
#: order.
_MAX_CASES = 8

#: Uniform-shape fallbacks for patterns no registered domain binds.
_LOOSE_SHAPES = ((4, 4), (2, 4), (4,), (1, 4), (4, 1), (2, 2, 4))
_LOOSE_ATTR_DOM = (-1, -2, 0, 1, 2)


# ---------------------------------------------------------------------------
#  The known bank + the candidate pool
# ---------------------------------------------------------------------------


def known_bank() -> list:
    """Return the widest shipped law surface, deduplicated by name.

    ``PRIOR_ART`` contributes only its ``FUSION``/``CATEGORICAL``
    rules — the structural transforms (the TASO merges, rope/STE/
    quant migrations) that are the actual novelty-floor content.
    ``verify_law``'s bounded e-graph applies every premise at every
    step: the prior-art ``SIMPLIFICATION``/``EXPANSIVE`` families —
    the pointwise-identity and decomp-direction rules that cycle on
    neg/add/sub/mul — explode its node budget and turn the
    derivable-positive gate into a hang (measured: the composite
    demotion test exceeds a 200 s box vs ~2 s without them).  Those
    rules remain in the ``prior_art`` preset itself; the sieve's
    known floor is the slice it can saturate against.
    """
    seen: dict[str, Any] = {}
    for group in (
        default_rules(),
        ALL_RULES_WITH_LAYOUT,
        ATTENTION_RULES,
        SCAN_LAWS,
        SCAN_DIAG_LAWS,
    ):
        for r in group:
            seen.setdefault(r.name, r)
    for r in PRIOR_ART:
        if tags.EXPANSIVE in r.tags:
            continue
        if tags.FUSION in r.tags or tags.CATEGORICAL in r.tags:
            seen.setdefault(r.name, r)
    return list(seen.values())


def shipped_candidates() -> list:
    """The sweep population: shipped machine laws + lawdata candidates.

    ``MACHINE_SHIPPED`` is resolved against the bank by name (a name
    missing from the bank is reported, not dropped); the ``cand_*``
    rows of :data:`lawdata.CANDIDATE_LAWS` arrive as
    :meth:`impact.new_laws` ``Rewrite``s — the minted-but-unshipped
    pool.  ``lawdata.MACROS`` rows are meta-game move sequences, not
    rewrite laws — there is nothing to sieve there.
    """
    by_name = {r.name: r for r in known_bank()}
    laws = [by_name[n] for n in sorted(MACHINE_SHIPPED) if n in by_name]
    from mlopt_discovery import impact

    laws += impact.new_laws()
    return laws


# ---------------------------------------------------------------------------
#  Instances — where the law's two sides come from
# ---------------------------------------------------------------------------


def _corpus_instance(
    rule: Any, ctxs: list[_CaseCtx]
) -> tuple[str, tuple[Any, Any]] | None:
    """First strict LHS match inside the spelled discovery corpus."""
    for ctx in ctxs:
        if ctx.term is None:
            continue
        for path, sub in _positions(ctx.term):
            if match_pattern(rule.lhs, sub) is None:
                continue
            out = apply_rewrite_at(rule, ctx.term, path)
            if out is not None:
                return ctx.case.name, (sub, _subterm(out, path))
    return None


def _synth_instance(rule: Any, limit: int) -> tuple[Any, Any] | None:
    """A guard-accepted site from the gauntlet's binding enumeration."""
    from mlopt_discovery.evidence import _synth_sites

    for bound, lhs_t in _synth_sites(rule.lhs, rule.rhs, limit=limit):
        try:
            if rule.check is not None and not rule.check(bound):
                continue
        except Exception:
            continue
        out = apply_rewrite_at(rule, lhs_t, ())
        if out is not None:
            return lhs_t, out
    return None


def _loose_instance(rule: Any) -> tuple[Any, Any] | None:
    """Uniform-shape metavariable binding — the last-resort instance.

    Every non-attr metavariable gets a ``Var`` of the same trial shape
    (scalar-metavariable names mint ``Const``s per the lawdata table);
    unknown ``$attr:`` names sweep a small dim domain.  The rule's own
    ``check``/``derive`` vetoes a binding that cannot denote, so a
    returned pair is a real firing site — and ``None`` is the honest
    "no instance" outcome, never a guess.
    """
    mvs = sorted(pattern_metavars(rule.lhs))
    attr_mvs = [m for m in mvs if m.startswith("$attr:")]
    leaf_mvs = [m for m in mvs if not m.startswith("$attr:")]
    attr_doms = [
        [lawdata.INSTANCE_ATTR_DEFAULTS[m[len("$attr:") :]]]
        if m[len("$attr:") :] in lawdata.INSTANCE_ATTR_DEFAULTS
        else list(_LOOSE_ATTR_DOM)
        for m in attr_mvs
    ]
    combos = itertools.product(*attr_doms) if attr_doms else [()]
    for combo in combos:
        for shape in _LOOSE_SHAPES:
            subst: dict[str, Any] = dict(
                zip(attr_mvs, combo, strict=True)
            )
            for mv in leaf_mvs:
                if mv in lawdata.INSTANCE_SCALAR_MVARS:
                    subst[mv] = Const(lawdata.INSTANCE_SCALAR_MVARS[mv])
                else:
                    subst[mv] = Var(mv, TensorType(shape))
            try:
                if rule.check is not None and not rule.check(subst):
                    continue
                lhs = instantiate_pattern(rule.lhs, subst)
                inst = dict(subst)
                if rule.derive is not None:
                    extra = rule.derive(subst)
                    if extra is None:
                        continue
                    inst.update(extra)
                return lhs, instantiate_pattern(rule.rhs, inst)
            except Exception:
                continue
    return None


def instance_of(
    rule: Any,
    ctxs: list[_CaseCtx],
    *,
    synth_limit: int = _SYNTH_LIMIT,
) -> tuple[str, tuple[Any, Any]] | None:
    """Return ``(source, (lhs, rhs))`` — a real firing site of *rule*.

    Sources in order: ``registry`` (``verifier.instance_of`` — the
    law_bench synthetic registry plus the pattern-driven generic
    tables), ``corpus:<case>`` (a strict match inside a spelled
    discovery module), ``synth`` (the gauntlet's binding enumeration),
    ``loose`` (uniform-shape binding).  ``None`` when no source
    produces a site.
    """
    inst = verifier.instance_of(rule)
    if inst is not None:
        return "registry", inst
    got = _corpus_instance(rule, ctxs)
    if got is not None:
        cname, inst = got
        return f"corpus:{cname}", inst
    inst = _synth_instance(rule, synth_limit)
    if inst is not None:
        return "synth", inst
    inst = _loose_instance(rule)
    if inst is not None:
        return "loose", inst
    return None


# ---------------------------------------------------------------------------
#  The derivability gate — is the law already a theorem of the bank?
# ---------------------------------------------------------------------------


def _premises(
    candidate: Any, bank: list
) -> tuple[list, tuple[str, ...]]:
    """The premise bank: *bank* minus the candidate's alpha-siblings.

    A sibling is a rule whose ``alpha_key`` — metavariables abstracted
    to positional indices — equals the candidate's: the same law under
    renaming (and the candidate itself when it sits in the bank).
    Leaving a sibling in would self-derive the candidate trivially.
    The inverse spelling is *not* a sibling: a bank holding the
    reversed equation honestly derives the candidate.
    """
    key = alpha_key(candidate.lhs, candidate.rhs)
    sibs = tuple(r.name for r in bank if alpha_key(r.lhs, r.rhs) == key)
    premises = [r for r in bank if alpha_key(r.lhs, r.rhs) != key]
    return premises, sibs


# ---------------------------------------------------------------------------
#  The measured gate — beat the best registered baseline, with attribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CaseCtx:
    """A discovery case with its spelled fixture and IR prebuilt."""

    case: Any
    model: Any
    feed: Any
    term: Any  # spelled IR root (None when fixture/export failed)
    ops: frozenset = frozenset()


@dataclass(frozen=True)
class BeatRow:
    """One case's measured-gate outcome for a candidate."""

    case: str
    certified: bool = False
    fired: int = 0  # rule_fires during saturation
    used: bool = False  # candidate appears in the cert's steps
    cost_before: float | None = None
    cost_after: float | None = None
    floor: float | None = None
    beat_best: bool | None = (
        None  # certified ∧ below floor (unattributed)
    )
    beat: bool = False  # beat_best ∧ the derivation used the candidate
    error: str = ""


def _ops_of(term: Any) -> frozenset:
    """Op names reachable in *term* (pattern-level, no torch needed)."""
    out: set[str] = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Op):
            out.add(t.op)
            stack.extend(t.args)
    return frozenset(out)


def case_contexts(cases: list | None = None) -> list[_CaseCtx]:
    """Build ``(model, feed, spelled-IR)`` for every case once.

    Fixture-build or export failures are recorded as ``None``-term
    contexts — the case is skipped for relevance, never silently
    dropped: it simply can never match.
    """
    from mlopt_torch.adapters import TorchSource

    from bench.discovery import runner as dr
    from bench.discovery.cases import load_cases

    source = TorchSource()
    ctxs: list[_CaseCtx] = []
    for case in load_cases() if cases is None else cases:
        try:
            model, feed = dr._build_fixture(case)
            ir, _leaves = source.to_ir(model, feed)
            ctxs.append(
                _CaseCtx(case, model, feed, ir.root, _ops_of(ir.root))
            )
        except Exception:
            ctxs.append(_CaseCtx(case, None, None, None, frozenset()))
    return ctxs


def _relevant(candidate: Any, ctx: _CaseCtx) -> bool:
    """Whether *ctx* can exercise *candidate*.

    A ``law:<name>`` signal names the case's own law; otherwise the
    candidate's LHS must strictly match some subterm of the spelled
    IR — the same criterion the e-graph applies before firing.
    """
    if ctx.term is None:
        return False
    if f"law:{candidate.name}" in ctx.case.expect.signals:
        return True
    for _path, sub in _positions(ctx.term):
        if match_pattern(candidate.lhs, sub) is not None:
            return True
    return False


def _measure_case(candidate: Any, ctx: _CaseCtx) -> BeatRow:
    """Run the symbolic discovery lane with the candidate in the bank.

    Mirrors ``runner.run_case``'s cost/baseline arm with one change —
    ``rules = resolved(case.rules) + {candidate}`` — so a candidate
    outside the shipped bank is measured as if admitted, and a shipped
    candidate reproduces the registered run exactly (union by name is
    a no-op).  ``beat`` requires the derivation certificate to name the
    candidate: a floor the rule never touched is not its win.
    """
    from mlopt_core.egraph import verify_certificate
    from mlopt_orchestrator.optimize import search
    from mlopt_torch.adapters import TorchSink, TorchSource

    from bench.discovery import runner as dr
    from bench.discovery.refs import REFS

    case = ctx.case
    sink, source = TorchSink(), TorchSource()
    cost_fn = dr.COST_FNS[case.cost]
    try:
        base = _resolve_rules(dr._case_rules(case.rules))
        rules = base.union([candidate], override=True)
        sr = search(
            ctx.model,
            ctx.feed,
            source=source,
            capabilities=sink,
            cost_fn=cost_fn,
            rules=rules,
            max_iterations=case.max_iterations,
            max_enodes=case.max_enodes,
        )
    except Exception as e:
        return BeatRow(case=case.name, error=f"search: {e}")
    try:
        cert = sr.eg.certificate(sr.ir.root, sr.term)
        verify_certificate(sr.ir.root, cert)
        certified = True
        used = any(s.rule == candidate.name for s in cert.steps)
    except Exception:
        certified, used = False, False
    fires = sr.stats.get("rule_fires") or getattr(
        sr.eg, "rule_fires", {}
    )
    cost_before = dr._term_cost(sr.ir.root, cost_fn)
    cost_after = dr._term_cost(sr.term, cost_fn)
    floors: list[float] = []
    if cost_before is not None and any(
        b.kind == "self" for b in case.baselines
    ):
        floors.append(cost_before)
    for b in case.baselines:
        if b.kind != "ref":
            continue
        try:
            mod = REFS[b.ref](ctx.model, ctx.feed)
            c = dr._module_cost(mod, ctx.feed, cost_fn, source)
            if c is not None:
                floors.append(c)
        except Exception:
            pass
    floor = min(floors) if floors else None
    beat_best = certified and floor is not None and cost_after < floor
    return BeatRow(
        case=case.name,
        certified=certified,
        fired=int(fires.get(candidate.name, 0)),
        used=used,
        cost_before=cost_before,
        cost_after=cost_after,
        floor=floor,
        beat_best=beat_best if certified else False,
        beat=bool(beat_best and used),
    )


def measure_gate(
    candidate: Any,
    ctxs: list[_CaseCtx],
    *,
    max_cases: int = _MAX_CASES,
) -> tuple[bool | None, list[BeatRow]]:
    """The measured gate: ``(beat, per-case rows)``.

    Runs every relevant case — signal-named first — up to
    *max_cases*.  ``beat`` is ``True`` when at least one certified,
    candidate-attributed extraction lands strictly below the best
    registered baseline floor; ``None`` when no case exercises the
    candidate at all (the gate is unevaluable, not failed).
    """
    named = [c for c in ctxs if c.term is not None]
    signal = [
        c
        for c in named
        if f"law:{candidate.name}" in c.case.expect.signals
    ]
    sig_ids = {id(c) for c in signal}
    rest = [
        c
        for c in named
        if id(c) not in sig_ids and _relevant(candidate, c)
    ]
    todo = (signal + rest)[:max_cases]
    rows = [_measure_case(candidate, c) for c in todo]
    if not rows or all(r.error for r in rows):
        return None, rows
    return bool(any(r.beat for r in rows)), rows


# ---------------------------------------------------------------------------
#  The sieve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SieveRow:
    """One candidate's three-gate record and verdict."""

    name: str
    provenance: str = ""
    verdict: str = ""  # review|known|refuted|uncertified|unpaid|unmeasured|no-instance
    instance_src: str = ""
    lhs_repr: str = ""
    rhs_repr: str = ""
    truth: str = ""  # fp64 oracle outcome on the instance
    cert: bool = False
    cert_note: str = ""
    derivable: bool | None = None
    witness_rules: tuple[str, ...] = ()
    deriv_note: str = ""
    siblings: tuple[str, ...] = ()
    beat: bool | None = None
    beats: tuple[BeatRow, ...] = field(default_factory=tuple)
    measure_note: str = ""

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "provenance": self.provenance,
            "verdict": self.verdict,
            "instance_src": self.instance_src,
            "truth": self.truth,
            "cert": self.cert,
            "derivable": self.derivable,
            "witness_rules": list(self.witness_rules),
            "deriv_note": self.deriv_note,
            "siblings": list(self.siblings),
            "beat": self.beat,
            "measure_note": self.measure_note,
            "lhs": self.lhs_repr,
            "rhs": self.rhs_repr,
            "cases": [
                {
                    "case": b.case,
                    "certified": b.certified,
                    "fired": b.fired,
                    "used": b.used,
                    "cost_before": b.cost_before,
                    "cost_after": b.cost_after,
                    "floor": b.floor,
                    "beat_best": b.beat_best,
                    "beat": b.beat,
                    "error": b.error,
                }
                for b in self.beats
            ],
        }
        return d


def sieve(
    candidate: Any,
    bank: list | None = None,
    *,
    ctxs: list[_CaseCtx] | None = None,
    measure: bool = True,
    max_cases: int = _MAX_CASES,
    measurer: Any = None,
) -> SieveRow:
    """Run the three gates on *candidate* against *bank*.

    ``bank`` defaults to :func:`known_bank` — the widest shipped
    surface.  ``ctxs`` defaults to :func:`case_contexts` when the
    measured gate is on; pass ``[]`` (and/or ``measure=False``) for a
    corpus-free check.  ``measurer`` overrides the measured gate's
    callable ``(candidate, ctxs) -> (beat, rows)`` — the seam the test
    suite stubs; production callers leave it ``None``.
    """
    bank = known_bank() if bank is None else list(bank)
    if ctxs is None:
        ctxs = case_contexts() if measure else []
    measure_fn = measurer or measure_gate
    row_kw: dict[str, Any] = {
        "name": candidate.name,
        "provenance": law_provenance(candidate),
    }
    got = instance_of(candidate, ctxs)
    if got is None:
        return SieveRow(
            verdict="no-instance",
            measure_note="no instance: gates unevaluable",
            **row_kw,
        )
    src, (lhs, rhs) = got
    row_kw.update(
        instance_src=src, lhs_repr=op_repr(lhs), rhs_repr=op_repr(rhs)
    )

    # -- truth: fp64 numeric eval of the instance (informational; an
    #    "unequal" is a measured counterexample and refutes outright)
    from mlopt_discovery import oracle

    truth, _tnote = oracle.eval_instance(lhs, rhs)
    row_kw["truth"] = truth
    if truth == "unequal":
        return SieveRow(verdict="refuted", **row_kw)

    # -- gate 1: cert — the candidate's own rule must carry a replayable
    #    certificate on its instance
    cres = verify_law(lhs, rhs, [candidate])
    row_kw["cert"] = bool(cres.derivable and cres.replayable)
    row_kw["cert_note"] = cres.note
    if not row_kw["cert"]:
        return SieveRow(verdict="uncertified", **row_kw)

    # -- gate 3 (before 2 — cheaper): derivability under bank minus
    #    alpha-siblings
    premises, sibs = _premises(candidate, bank)
    row_kw["siblings"] = sibs
    dres = verify_law(lhs, rhs, premises)
    row_kw["derivable"] = dres.derivable
    row_kw["witness_rules"] = dres.witness_rules
    row_kw["deriv_note"] = dres.note or dres.stop
    if dres.derivable:
        return SieveRow(verdict="known", **row_kw)

    # -- gate 2: measured pay on the discovery corpus
    if measure:
        beat, beats = measure_fn(candidate, ctxs, max_cases=max_cases)
        row_kw["beat"] = beat
        row_kw["beats"] = tuple(beats)
        row_kw["measure_note"] = (
            f"{len(beats)} case(s) run"
            if beats
            else "no case exercises this law"
        )
    else:
        row_kw["beat"] = None
        row_kw["measure_note"] = "measurement skipped"

    if row_kw["beat"] is True:
        verdict = "review"
    elif row_kw["beat"] is False:
        verdict = "unpaid"
    else:
        verdict = "unmeasured"
    return SieveRow(verdict=verdict, **row_kw)


def sieve_all(
    candidates: list | None = None,
    bank: list | None = None,
    *,
    measure: bool = True,
    max_cases: int = _MAX_CASES,
    measurer: Any = None,
    progress: Any = None,
) -> list[SieveRow]:
    """Batch mode: sieve every candidate against *bank*.

    Case contexts (fixture + spelled IR per discovery case) are built
    once and shared across candidates.
    """
    bank = known_bank() if bank is None else list(bank)
    laws = shipped_candidates() if candidates is None else candidates
    ctxs = case_contexts() if measure else []
    rows = []
    for c in laws:
        row = sieve(
            c,
            bank,
            ctxs=ctxs,
            measure=measure,
            max_cases=max_cases,
            measurer=measurer,
        )
        rows.append(row)
        if progress is not None:
            progress(row)
    return rows


# ---------------------------------------------------------------------------
#  The review-queue document
# ---------------------------------------------------------------------------


def _bool(b: bool | None) -> str:
    return {True: "yes", False: "no", None: "—"}[b]


def _num(v: float | None) -> str:
    return "—" if v is None else f"{v:g}"


def render_markdown(
    rows: list[SieveRow], *, env: dict | None = None
) -> str:
    """The human document — one row per candidate, then the queue."""
    env = env or {}
    queue = [r for r in rows if r.verdict == "review"]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    lines = [
        "# Novelty sieve",
        "",
        "Automatic three-gate filter between a minted law and the "
        "human-review queue.  **cert** = the law's own rewrite carries "
        "a standalone-replayable certificate on its instance "
        "(`verify_law` + `verify_certificate`); **derivable** = the "
        "widest shipped bank (`default_rules` + `ALL_RULES_WITH_LAYOUT` "
        "+ `ATTENTION` + scan monoids + the prior-art FUSION/"
        "CATEGORICAL slice — `known_bank`), minus the candidate and its "
        "alpha-siblings, already proves the instance — derivable laws "
        "are theorems of known rules, demoted silently; **beat** = on "
        "some `bench/discovery` case the certified extraction (run "
        "with the candidate in the rules) lands strictly below the "
        "best registered baseline AND the derivation certificate "
        "invokes the candidate (the `beat_best` floor plus "
        "attribution).  A candidate is a novelty iff cert ∧ beat ∧ "
        "¬derivable.  `truth` is the fp64 oracle's numeric verdict on "
        "the instance — an `unequal` refutes outright.",
        "",
        f"- date: {env.get('timestamp_utc', '—')} · "
        f"git: `{env.get('git_sha', 'unknown')}`"
        f"{' (dirty)' if env.get('git_dirty') else ''} · "
        f"torch {env.get('torch', '?')} on {env.get('device', '?')} · "
        f"bank {env.get('bank_size', '?')} rules",
        "- verdicts: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "",
        "| law | prov | instance | truth | cert | derivable | beat | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        wit = (
            f" ({', '.join(r.witness_rules[:3])})"
            if r.derivable and r.witness_rules
            else ""
        )
        lines.append(
            f"| `{r.name}` | {r.provenance or '—'} | {r.instance_src or '—'} "
            f"| {r.truth or '—'} | {_bool(r.cert)} "
            f"| {_bool(r.derivable)}{wit} | {_bool(r.beat)} "
            f"| **{r.verdict}** |"
        )
    lines += ["", "## Review queue", ""]
    if queue:
        lines.append(
            "Certified, measured-paying, and underivable from the "
            "known bank — these are for a human to look at:"
        )
        lines.append("")
        for r in queue:
            wins = ", ".join(
                f"{b.case} ({_num(b.cost_before)} → "
                f"{_num(b.cost_after)} vs floor {_num(b.floor)})"
                for b in r.beats
                if b.beat
            )
            lines.append(f"- `{r.name}` — {wins}")
    else:
        lines.append(
            "**Empty.** Every candidate either fails a gate or is a "
            "theorem of the bank — nothing currently needs a "
            "reviewer's eyes."
        )
    lines += ["", "## Per-law detail", ""]
    for r in rows:
        lines += [
            f"### `{r.name}` — {r.verdict}",
            "",
            f"- instance ({r.instance_src or 'none'}): "
            f"`{r.lhs_repr or '—'}` → `{r.rhs_repr or '—'}`",
            f"- truth: {r.truth or '—'} · cert: {_bool(r.cert)}"
            + (f" ({r.cert_note})" if r.cert_note else ""),
            f"- derivable: {_bool(r.derivable)}"
            + (
                f" via {', '.join(r.witness_rules)}"
                if r.witness_rules
                else ""
            )
            + (f" — {r.deriv_note}" if r.deriv_note else ""),
            f"- siblings excluded: {', '.join(r.siblings) or 'none'}",
            f"- beat: {_bool(r.beat)} — {r.measure_note or '—'}",
        ]
        for b in r.beats:
            if b.error:
                lines.append(f"  - `{b.case}`: error — {b.error}")
                continue
            lines.append(
                f"  - `{b.case}`: cert {_bool(b.certified)}, "
                f"fired {b.fired}, in-deriv {_bool(b.used)}, "
                f"cost {_num(b.cost_before)} → {_num(b.cost_after)} "
                f"vs floor {_num(b.floor)}"
                f" → beat {_bool(b.beat)}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the sieve over the shipped/minted bank; emit the queue doc."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--names",
        help="comma-separated rule names to sieve (default: shipped "
        "machine laws + lawdata candidates)",
    )
    parser.add_argument(
        "--no-measure",
        action="store_true",
        help="skip the measured gate (cert + derivability only)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=_MAX_CASES,
        help="per-candidate cap on measured cases",
    )
    parser.add_argument("--json", help="write machine-readable results")
    parser.add_argument(
        "--out",
        default=str(DOC_PATH),
        help=f"markdown output (default {DOC_PATH})",
    )
    args = parser.parse_args(argv)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    bank = known_bank()
    if args.names:
        pool = {r.name: r for r in bank}
        pool.update({r.name: r for r in shipped_candidates()})
        laws = []
        for n in args.names.split(","):
            n = n.strip()
            if n in pool:
                laws.append(pool[n])
            else:
                print(f"unknown law {n!r}", file=sys.stderr)
                return 2
    else:
        laws = shipped_candidates()

    print(  # stdout-compat
        f"sieving {len(laws)} candidates against a "
        f"{len(bank)}-rule bank"
    )
    rows = sieve_all(
        laws,
        bank,
        measure=not args.no_measure,
        max_cases=args.max_cases,
        progress=lambda r: print(  # stdout-compat
            f"  {r.name:<34} {r.verdict}", flush=True
        ),
    )

    from bench.benchkit.env import collect_env
    from bench.benchkit.provenance import with_provenance

    env = collect_env("cpu")
    env["bank_size"] = len(bank)
    md = render_markdown(rows, env=env)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"wrote {out}")  # stdout-compat

    if args.json:
        payload = with_provenance(
            {
                "schema": 1,
                "candidates": [r.to_dict() for r in rows],
            },
            seed=None,
            device="cpu",
            env=env,
        )
        Path(args.json).write_text(
            json.dumps(payload, indent=2, default=str) + "\n"
        )
        print(f"wrote {args.json}")  # stdout-compat

    queue = [r.name for r in rows if r.verdict == "review"]
    print(  # stdout-compat
        f"review queue: {queue or 'empty'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

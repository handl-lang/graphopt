"""Law-discovery pipeline — census -> propose -> verify -> measure -> rank.

The loop that produced ``select_mul`` (retro
``project/retros/law-shape-aware.md``) was a sequence of one-off
tools with a human deciding which experiment to run at each step:
the census (``mlopt_discovery.census``), the shape-aware proposer
(``mlopt_discovery.shape_proposal``), the two oracles
(``mlopt_discovery.verifier`` derivability, ``mlopt_discovery.proposal``
numeric truth) and the impact measurement
(``mlopt_discovery.impact``).  Each did its job; nothing ran them end
to end and decided.

This tool is that end-to-end runnable: **one entry point** that

1. **census** — mines the shape frequencies of the real corpus
   (``mlopt_discovery.census.run_census``).
2. **propose** — collects candidate equalities from the proposers:
   the census generators (same-view and mixed-view naturality over the
   frequent op-tuples, plus composed-then-reduced pattern
   recognition), ``mlopt_discovery.shape_proposal.schemas`` shape-aware schemas
   and ``mlopt_discovery.proposal.schema_candidates`` algebraic grammar — unified
   to pattern rules and de-duplicated by alpha-normal key.
3. **verify** — runs *both* oracles on every candidate's concrete
   instance: the numeric-truth oracle (``mlopt_discovery.proposal._numeric_true``)
   and the derivability oracle (``mlopt_discovery.verifier.verify_law``), the
   latter against the *search rule set* (the library, minus any
   ``--holdout`` names).
4. **measure** — fires the rule alone over every real model
   (``mlopt_discovery.impact._probe``: fires, cost delta, lowered-module
   ``sink.verify``) and audits every merged fire for *well-typedness*
   (:func:`_typed_probe` — a fire whose instantiated RHS does not
   shape-resolve, or does not evaluate, minted a member that does
   not denote; it counts in ``fires_ill_typed``, and a cost drop
   whose extraction picked an ill-typed member is suppressed, not
   paid).  Then, for any candidate that fires, it saturates each
   model under the search rule set with and without the rule
   (``mlopt_discovery.impact._saturate`` / ``_cert_ok``): end-to-end cost delta,
   certificate replay, and a **closure-safety** check — the enode
   ratio ``out/in`` (a rule that explodes the closure is a search
   hazard, not a win).
5. **rank** — a single deterministic ordering over the evidence and a
   **ship / no-ship** verdict per candidate, with the reason.
6. **emit** (``--emit-admission``) — for a SHIP candidate, writes the
   *admission artifact* (``mlopt_discovery.emit``): the ``R(...)`` source
   with named ``check`` / ``derive`` hooks, a generated
   ``tests/test_admitted_<law>.py``, and a review patch
   (``admission_<law>.patch``) showing the exact ``tensor.py``
   insertion — a report, never a mutation.

Held-out rediscovery
--------------------

``select_mul`` is now shipped (``mlopt_core.laws.tensor.SELECT_MUL``,
in ``SIMPLIFICATION_RULES``).  To validate the pipeline, remove it —
and only it — from the search rule set with ``--holdout select_mul``,
re-run, and locate the proposal whose equality matches the held-out
law in the ranking.  If the pipeline cannot rank a known winner at the
top, it is not trustworthy; the report says so plainly.

Reuse, not reinvention: every stage calls the shipped tool that
already implements it.  The only new logic here is the composition,
the closure-safety ratio, and the ranking.

Run::

    .venv/bin/python -m mlopt_discovery.pipeline
    .venv/bin/python -m mlopt_discovery.pipeline --holdout select_mul
    .venv/bin/python -m mlopt_discovery.pipeline --json /tmp/pipeline.json
    .venv/bin/python -m mlopt_discovery.pipeline --holdout softmax_fold \
        --emit-admission recognize:softmax --out /tmp/admission

CPU-only, bounded to a few minutes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mlopt_core.cost import dag_cost
from mlopt_core.egraph import EGraph, Rewrite
from mlopt_core.egraph.terms import _term_instantiate, _term_match
from mlopt_core.ir import Const, Op, op_repr
from mlopt_core.laws import ALL_RULES
from mlopt_core.laws.priorart import (
    GROUP_NORM_EXPAND,
    INSTANCE_NORM_EXPAND,
)
from mlopt_core.opmeta import (
    GENERATOR_VIEW_OPS,
    POINTWISE_BINARY_OPS,
)
from mlopt_core.typing import _shape_of

# Sibling tools own every stage; reuse them, never duplicate.
from mlopt_discovery import evidence as ev_store
from mlopt_discovery import oracle as lvo
from mlopt_discovery import proposal as lp
from mlopt_discovery.census import (
    _op_of,
    run_census,
    shape_key,
    shape_repr,
)
from mlopt_discovery.impact import (
    _FIRING_ITERS,
    _FIRING_NODES,
    TermCase,
    _bench_cases,
    _cert_ok,
    _cost_fn,
    _iter_subterms,
    _probe,
    _saturate,
    model_cases,
)
from mlopt_discovery.shape_proposal import (
    Schema,
    _sink,
    real_matches,
    relaxed_matches,
    schemas,
)
from mlopt_discovery.verifier import verify_law

__all__ = [
    "Evidence",
    "Proposal",
    "main",
    "propose",
    "run_pipeline",
]

#: Default number of ranked rows printed.
_DEFAULT_TOP = 20

#: Enode ``out/in`` ratio above which a candidate's *unexplained*
#: growth is closure-unsafe: a rule that grows the e-graph more than
#: this — beyond what its own firings provably minted — is a search
#: hazard.
_CLOSURE_LIMIT = 2.0

#: How much *secondary* growth a firing footprint may explain.  A
#: composite's spelled-out RHS legitimately mints many enodes in one
#: application, and each minted member can wake the base bank (comm /
#: assoc re-spellings of the folded term).  The gate forgives that
#: cascade while it stays within this multiple of the enodes the
#: candidate's own recorded applications created; growth beyond it is
#: unexplained — the blow-up signal the ratio alone cannot separate
#: from a large one-shot footprint on a small graph.
_CLOSURE_CASCADE_FACTOR = 2.0

#: Census rows to keep — large enough to hold every distinct op-tuple
#: (the corpus has ~124), so the frequency lookup is exact.
_CENSUS_TOP = 400


# ---------------------------------------------------------------------------
#  Proposal — one candidate equality, unified across the proposers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Proposal:
    """One candidate law: a pattern equality, plus an optional instance.

    ``lhs`` / ``rhs`` are *patterns* (bare ``str`` leaves are
    metavariables), so a single rule fires on any instantiation and
    the census matchers apply uniformly.  ``instance`` is a concrete
    ``(lhs, rhs)`` pair for the numeric / derivability oracles when the
    proposer already supplies one (the algebraic grammar); the
    shape-aware schemas supply patterns only and are instantiated from
    a real match instead.

    ``check`` / ``derive`` are the ``Rewrite`` side-condition hooks,
    for candidates whose RHS needs an attribute the LHS does not carry
    verbatim — e.g. the softmax fold, whose ``dim`` is a single int
    derived from ``sum``'s ``dim`` tuple.  They pass through to
    :meth:`as_rule`, so firing, certificate replay and the oracles all
    see the same side condition.

    ``cond`` / ``dspec`` are the *declarative* twins (pure data per
    ``mlopt_core.laws.cond``): :meth:`as_rule` hands them to
    ``Rewrite``, which folds them into the same ``check``/``derive``
    hooks — so a proposal minted from a shipped fold keeps the law's
    side condition *and* serializes as data (the admission ledger's
    ``missing_hooks`` stays empty where the DSL covers the guard).

    ``premises`` is the proposal's declared *premise closure* — extra
    axioms the cert leg unions into its ``verify_law`` axiom set when
    the bare bank cannot derive the claim (the
    ``_derivable_result``-anchored channel; the fusion-schema mint
    attaches the fused op's definitional expansion).  It rides the
    minted ``Rewrite`` as a ``premises`` attribute — like the
    composite ``parents``/``guard_pats`` record, it is *not* part of
    the serialized law: the ledger records the premise names for
    audit and a deterministic re-mint re-attaches them.
    """

    name: str
    lhs: Any
    rhs: Any
    family: str
    sources: tuple[str, ...] = ()
    instance: tuple[Any, Any] | None = None
    check: Any = None
    derive: Any = None
    cond: Any = None
    dspec: Any = None
    premises: tuple = ()

    def as_rule(self) -> Rewrite:
        """Return the proposal as a fireable pattern ``Rewrite``."""
        rule = Rewrite(
            self.name,
            self.lhs,
            self.rhs,
            check=self.check,
            derive=self.derive,
            cond=self.cond,
            dspec=self.dspec,
        )
        # ``Rewrite`` is frozen — the closure attaches by the
        # ``parents``/``guard_pats`` convention (``()`` for none).
        object.__setattr__(rule, "premises", tuple(self.premises))
        return rule


def _to_pattern(term: Any, mv: dict[str, str]) -> Any:
    """Abstract the concrete leaves of *term* to shared metavariables.

    A leaf (``Var`` / ``Param``) keyed by ``repr`` becomes ``M0``,
    ``M1``, … under the shared *mv* map, so two occurrences of the
    same leaf stay the same metavariable and a repeated-slot equality
    survives abstraction.  ``Const`` leaves stay literal.
    """
    if isinstance(term, Op):
        return Op.make(
            term.op,
            *(_to_pattern(a, mv) for a in term.args),
            **dict(term.attrs),
        )
    if isinstance(term, Const):
        return term
    key = repr(term)
    if key not in mv:
        mv[key] = f"M{len(mv)}"
    return mv[key]


#: Pointwise ops a view-naturality law may push through a view op.
#: A projection of :mod:`mlopt_core.opmeta` (``pointwise-binary``).
_POINTWISE = tuple(sorted(POINTWISE_BINARY_OPS))

#: View / re-layout ops (the census's frequent inner ops).  A
#: projection of :mod:`mlopt_core.opmeta` (the ``generator-view``
#: tag) — a documented *subset* of the ``relayout`` set
#: ``mlopt_orchestrator.morphisms.signature._VIEW_OPS`` reads, so the
#: two no longer drift (the subset relation is machine-checked in
#: ``tests/test_opmeta.py``).
_VIEW_OPS = tuple(sorted(GENERATOR_VIEW_OPS))


def _shape_aware() -> list[Proposal]:
    """Return the shape-aware grammar's schemas as proposals."""
    return [
        Proposal(
            name=s.name,
            lhs=s.lhs,
            rhs=s.rhs,
            family=s.family,
            sources=("shape-aware",),
        )
        for s in schemas()
    ]


def _grammar() -> list[Proposal]:
    """Return the algebraic grammar's candidates as pattern proposals."""
    out: list[Proposal] = []
    for c in lp.schema_candidates():
        mv: dict[str, str] = {}
        out.append(
            Proposal(
                name=f"grammar:{c.label}",
                lhs=_to_pattern(c.lhs, mv),
                rhs=_to_pattern(c.rhs, mv),
                family="algebraic-grammar",
                sources=("algebraic-grammar",),
                instance=(c.lhs, c.rhs),
            )
        )
    return out


def _view_attrs(terms: list[Any], view_op: str) -> list[str]:
    """Return the sorted attr keys of every *view_op* node in *terms*."""
    keys: set[str] = set()
    for term in terms:
        for sub in _iter_subterms(term):
            if isinstance(sub, Op) and sub.op == view_op:
                keys |= set(sub.attrs)
    return sorted(keys)


def _view_node(
    view_op: str, leaf: Any, keys: list[str], tag: str = "V"
) -> Any:
    """Return a view-op pattern node with *keys* as attr metavariables.

    The attribute values are shared metavariable names prefixed by
    *tag*, so the two view nodes of a naturality law must carry the
    same attrs — the matcher enforces the precondition structurally.
    Distinct tags let a mixed-view pattern give each view node its own
    attr metavariables (no cross-op attr equality is implied).
    """
    attrs: dict[str, Any] = {k: f"{tag}_{k}" for k in keys}
    return Op.make(view_op, leaf, **attrs)


def _vocab_sets(vocab: str) -> tuple[tuple, tuple]:
    """Return ``(pointwise, views)`` for the requested vocabulary.

    ``"hand"`` is the two hand-written tuples below; ``"derived"``
    classifies every corpus op by property (``mlopt_discovery.vocab``),
    so the generator's op alphabet is machine-derived too.
    """
    if vocab == "derived":
        from mlopt_discovery import vocab as vocab_mod

        v = vocab_mod.derive_vocabulary()
        return v.pointwise, v.views
    return _POINTWISE, _VIEW_OPS


def _census_naturality(
    census_op: dict,
    terms: list[Any],
    pointwise: tuple = _POINTWISE,
    views: tuple = _VIEW_OPS,
) -> list[Proposal]:
    """Mechanically derive view-naturality candidates from the census.

    For every frequent op-tuple ``f(g, g)`` — a pointwise *f* whose two
    operands are the *same* view op *g* — propose the naturality

        f(g(u, A), g(v, A)) -> g(f(u, v), A)

    where *A* are *g*'s own attribute keys, shared across both nodes.
    This is the census -> propose step done by machine: the shape
    comes from ``law_shape_census``'s op-tuple counts, not a hand-
    written schema list.  The truth oracle (not this generator)
    decides whether each candidate is sound.  The op alphabet is the
    ``pointwise`` / ``views`` pair — property-derived by default
    (``mlopt_discovery.vocab``), or hand-written (``--vocab hand``).
    """
    out: list[Proposal] = []
    for op, kids in census_op:
        if op not in pointwise or len(kids) != 2:
            continue
        if kids[0] != kids[1] or kids[0] not in views:
            continue
        g = kids[0]
        keys = _view_attrs(terms, g)
        lhs = Op.make(
            op,
            _view_node(g, "U", keys),
            _view_node(g, "V", keys),
        )
        rhs = _view_node(g, Op.make(op, "U", "V"), keys)
        out.append(
            Proposal(
                name=f"census:{op}_{g}",
                lhs=lhs,
                rhs=rhs,
                family="census-naturality",
                sources=("census-naturality",),
            )
        )
    return out


def _two_view_family(
    op: str, g: str, h: str, terms: list[Any]
) -> list[Proposal]:
    """Emit the RHS family for a two-view tuple ``f(g(u,A), h(v,B))``.

    The plausible wrap targets are the left view, the right view, and
    no view at all (the ``id``/strip variant) — three shallow guesses,
    each letting the oracles be the referee rather than an algebraic
    theory of view inversion.  Each view keeps its own attr
    metavariables (``A_*`` / ``B_*``), so the RHS re-wraps with the
    bound attrs of whichever side it picks.
    """
    kg, kh = _view_attrs(terms, g), _view_attrs(terms, h)
    lhs = Op.make(
        op,
        _view_node(g, "U", kg, tag="A"),
        _view_node(h, "V", kh, tag="B"),
    )
    inner = Op.make(op, "U", "V")
    variants = {
        "wl": _view_node(g, inner, kg, tag="A"),
        "wr": _view_node(h, inner, kh, tag="B"),
        "id": inner,
    }
    return [
        Proposal(
            name=f"mixed:{op}_{g}_{h}_{tag}",
            lhs=lhs,
            rhs=rhs,
            family="census-mixed-view",
            sources=("census-mixed-view",),
        )
        for tag, rhs in variants.items()
    ]


def _one_view_family(
    op: str, view: str, view_left: bool, terms: list[Any]
) -> list[Proposal]:
    """Emit the RHS family for ``f(view(u,A), v)`` with ``v`` opaque.

    The non-view operand stays a bare metavariable — the census tuple
    (``mul(slice, ·)``, ``mul(select, add)``, ``mul(unsqueeze,
    stack)``) only fixes that a view feeds one side of a pointwise
    op, so every asymmetric tuple collapses onto this one pattern and
    de-dup merges the provenance.  The two plausible equalities are
    the push-through ``w(f(u, v))`` and the strip ``f(u, v)``; truth
    is per-instance and the oracles decide.
    """
    keys = _view_attrs(terms, view)
    node = _view_node(view, "U", keys, tag="A")
    lhs = (
        Op.make(op, node, "V") if view_left else Op.make(op, "V", node)
    )
    side = "l" if view_left else "r"
    inner = Op.make(op, "U", "V")
    variants = {
        "w": _view_node(view, inner, keys, tag="A"),
        "id": inner,
    }
    return [
        Proposal(
            name=f"mixed:{op}_{view}_{side}_{tag}",
            lhs=lhs,
            rhs=rhs,
            family="census-mixed-view",
            sources=("census-mixed-view",),
        )
        for tag, rhs in variants.items()
    ]


def _census_mixed_naturality(
    census_op: dict,
    terms: list[Any],
    pointwise: tuple = _POINTWISE,
    views: tuple = _VIEW_OPS,
) -> list[Proposal]:
    """Generalize census naturality to *mixed* operand views.

    :func:`_census_naturality` only reaches ``f(g, g)`` — a pointwise
    op over two copies of the SAME view.  The expanded corpus's more
    frequent shape is asymmetric: ``mul(slice, ·)``, ``mul(select,
    add)``, ``add(matmul, select)``, ``mul(unsqueeze, stack)`` — the
    retro's MoE-dispatch lead.  For every census tuple ``f(g, h)``
    where at least one child is a view this emits a small family of
    ``f(g(u, A), h(v, B)) -> w(f(u, v))`` candidates (``w`` over the
    views present, plus the identity).

    A *same-view* tuple ``f(g, g)`` gets the two-view treatment too:
    :func:`_census_naturality` shares one attr-metavariable namespace
    across both children, so its candidate only binds sites whose two
    ``g`` nodes spell identical attrs — the ``mul(unsq, unsq)`` case
    the zoo-real corpus measured (``docs/results_overnight_real.md``),
    where the real site carries *different* dims and the shared-attr
    precondition never fires.  Emitting the family with independent
    namespaces (``A_*`` / ``B_*``) proposes over the spelling the
    module actually wrote; whether ``wl``/``wr``/``id`` is a true
    equality on that spelling is again the oracles' call.  The numeric
    oracle and the measurement gates — not this generator — decide
    truth and worth; a false or worthless variant is an honest
    rejection, not a generator bug.
    """
    out: list[Proposal] = []
    for op, kids in census_op:
        if op not in pointwise or len(kids) != 2:
            continue
        g, h = kids
        if g == h:
            if g in views:
                out.extend(_two_view_family(op, g, g, terms))
            continue
        gl, hl = g in views, h in views
        if gl and hl:
            out.extend(_two_view_family(op, g, h, terms))
        elif gl or hl:
            out.extend(_one_view_family(op, g if gl else h, gl, terms))
    return out


def _reduce_chains(census_op: dict) -> list[tuple[str, str, str]]:
    """Return ``(f, u, r)`` triples: ``f`` combines ``u(·)`` with ``r(u(·))``.

    The census signature of a composed-then-reduced chain — a manual
    normalization fold: a unary ``u`` feeding a reduction ``r`` (the
    ``(r, (u,))`` tuple) AND a binary ``f`` combining ``u(·)`` with an
    ``r`` node (``(f, (u, r))`` or ``(f, (r, u))``) both occur.  The
    corpus's instance is ``div(exp(·), sum(exp(·)))`` — a softmax
    spelled by hand.
    """
    out: list[tuple[str, str, str]] = []
    for op, kids in census_op:
        if len(kids) != 2:
            continue
        for i in (0, 1):
            u, r = kids[i], kids[1 - i]
            if u in ("·", "const") or r in ("·", "const"):
                continue
            if (r, (u,)) in census_op:
                out.append((op, u, r))
    return out


def _check_sum_keepdim(bound: dict) -> bool:
    """Guard the softmax fold: keepdim and a single reduce axis.

    ``x / sum(x, dim)`` only broadcasts to a softmax when the sum
    keeps its dim (a dropped dim broadcasts wrongly — or not at all —
    against the numerator), and a multi-axis sum has no single-dim
    ``softmax`` image.
    """
    dims = bound.get("$attr:RD")
    if bound.get("$attr:RK") is not True:
        return False
    return isinstance(dims, int) or (
        isinstance(dims, tuple) and len(dims) == 1
    )


def _derive_softmax_dim(bound: dict) -> dict:
    """Unwrap ``sum``'s ``dim`` tuple into ``softmax``'s scalar dim."""
    dims = bound.get("$attr:RD")
    return {"$attr:SD": dims[0] if isinstance(dims, tuple) else dims}


def _softmax_fold() -> Proposal:
    """Return the ``exp/sum`` spelling of softmax — a true shape fold.

    ``div(exp(u), sum(exp(u), dim, keepdim)) -> softmax(u, dim)`` is
    the kernel-recognition candidate ``ManualSoftmaxAttention`` puts
    in the corpus.  ``check``/``derive`` carry the side condition
    (keepdim, single axis) and the attr translation (the sum's
    ``dim`` tuple to softmax's int) — the oracles see them through
    :meth:`Proposal.as_rule` exactly as firing does.
    """
    e = Op.make("exp", "U")
    return Proposal(
        name="recognize:softmax",
        lhs=Op.make(
            "div", e, Op.make("sum", e, dim="RD", keepdim="RK")
        ),
        rhs=Op.make("softmax", "U", dim="SD"),
        family="pattern-recognition",
        sources=("pattern-recognition",),
        check=_check_sum_keepdim,
        derive=_derive_softmax_dim,
    )


#: Composed-then-reduced chains with a known kernel image, keyed by
#: ``(f, u, r)``.  Only the softmax fold is recognized today; a chain
#: with no recognizer stays a census fact, never a guessed equality.
_RECOGNIZERS: dict[tuple[str, str, str], Any] = {
    ("div", "exp", "sum"): _softmax_fold,
}


def _pattern_recognition(census_op: dict) -> list[Proposal]:
    """Emit candidates for recognized composed-then-reduced chains.

    This is the shape-proposal source beyond view naturality: the
    census is scanned for ``f(u(·), r(u(·)))`` chains and each chain
    with a registered recognizer yields one candidate.  Chains with no
    recognizer are evidence of a missing fold, silently skipped — the
    candidate pool stays honest guesses only.
    """
    out: list[Proposal] = []
    seen: set[tuple[str, str, str]] = set()
    for chain in _reduce_chains(census_op):
        if chain in seen:
            continue
        seen.add(chain)
        build = _RECOGNIZERS.get(chain)
        if build is not None:
            out.append(build())
    return out


# ---------------------------------------------------------------------------
#  Fusion schemas — fold-class recognizers beyond the view grammar
# ---------------------------------------------------------------------------
#
# The ``zoo-real`` mint (``docs/reports/results_overnight_real_v2.md``)
# is dominated by ``mixed:``/view naturality candidates while the real
# exports carry recurring *fusion-class* spellings the view grammar
# cannot see: the manual silu/glu/sdpa/rms chains the shipped fold
# laws exist for, plus kernel-fold classes the bank lacks entirely
# (the quant modules' matmul-bias epilogues, the manual norm chains).
# This arm mints on those: every shipped fold/fuse law is a schema —
# its LHS pattern is the recognizer, its ``cond``/``dspec`` the
# transported side condition — and the ``_FUSION_SCHEMAS`` table adds
# the measured-but-unshipped classes.  A schema mints only when a real
# subterm *satisfies* it (lhs match + side condition + denoteable
# instantiated RHS): the candidate's claim is module-bound — the
# ``instance`` pair carries the seed site's own attrs/terms to the
# oracles.


def _fusion_laws() -> list[Rewrite]:
    """Return the shipped fold/fuse laws — the fusion-class schemas.

    The schema bank is the law bank: every shipped ``*_fold*`` (silu,
    glu, softmax, softsign, rms-norm, conv-bn, the sdpa family) and
    ``*_fuse`` (swiglu, parallel-mul) rule, plus the two fusion laws
    whose names carry neither suffix (``qkv_fuse_asym``,
    ``gqa_absorb_repeat``).  Reading ``ALL_RULES`` — not a parallel
    name list — keeps the schema table honest: a shipped fold is a
    schema by construction.
    """
    return [
        r
        for r in ALL_RULES
        if r.name.endswith(("_fold", "_fuse"))
        or "_fold_" in r.name
        or r.name in ("qkv_fuse_asym", "gqa_absorb_repeat")
    ]


def _inorm_spelling() -> tuple[Any, Any]:
    """``(x-μ)·rsqrt(σ²+ε)·w+b`` — the manual instance-norm chain.

    Per-(n, c) statistics broadcast by ``unsqueeze`` rather than
    ``keepdim`` (``dim=MD`` shared between the mean/var branches, the
    broadcast dims ``K1``/``K2`` likewise), ``σ² = E[(x-μ)²]``, the
    channel affine spelled ``reshape``-broadcast — the
    ``InstanceNormBroadcast`` export verbatim.  ``W``/``B`` bind the
    raw affine leaves, so the RHS is the fused op directly.
    """
    d = Op.make(
        "sub",
        "U",
        Op.make(
            "unsqueeze",
            Op.make(
                "unsqueeze", Op.make("mean", "U", dim="MD"), dim="K1"
            ),
            dim="K2",
        ),
    )
    v = Op.make(
        "unsqueeze",
        Op.make(
            "unsqueeze",
            Op.make("mean", Op.make("mul", d, d), dim="MD"),
            dim="K1",
        ),
        dim="K2",
    )
    lhs = Op.make(
        "add",
        Op.make(
            "mul",
            Op.make(
                "mul",
                d,
                Op.make("rsqrt", Op.make("add", v, "EPS")),
            ),
            Op.make("reshape", "W", shape="WS"),
        ),
        Op.make("reshape", "B", shape="BS"),
    )
    rhs = Op.make("instance_norm", "U", "W", "B", eps="EP")
    return lhs, rhs


def _gnorm_spelling() -> tuple[Any, Any]:
    """``reshape((x'-μ)·rsqrt(E[(x'-μ)²]+ε))·w+b`` — manual group-norm.

    The group fold: ``x' = reshape(x, (1, G, ·))`` so the per-(n, g)
    reduce is a plain trailing ``mean`` — the ``ManualGroupNorm``
    export's spelling.  ``num_groups`` is the reshape target's second
    dim (``("nth", ("attr", "US"), 1)``); the affine is again the
    broadcast-``reshape`` form with ``W``/``B`` the raw leaves.
    """
    ru = Op.make("reshape", "U", shape="US")
    d = Op.make("sub", ru, Op.make("mean", ru, dim="MD", keepdim="MK"))
    v = Op.make("mean", Op.make("mul", d, d), dim="MD", keepdim="MK")
    lhs = Op.make(
        "add",
        Op.make(
            "mul",
            Op.make(
                "reshape",
                Op.make(
                    "mul", d, Op.make("rsqrt", Op.make("add", v, "EPS"))
                ),
                shape="RS",
            ),
            Op.make("reshape", "W", shape="WS"),
        ),
        Op.make("reshape", "B", shape="BS"),
    )
    rhs = Op.make("group_norm", "U", "W", "B", num_groups="G", eps="EP")
    return lhs, rhs


_INORM_LHS, _INORM_RHS = _inorm_spelling()
_GNORM_LHS, _GNORM_RHS = _gnorm_spelling()


#: The *definitional* bridges the kernel-fold schemas prove against —
#: now shipped substrate-side in the prior-art bank
#: (``mlopt_core.laws.priorart.INSTANCE_NORM_EXPAND`` /
#: ``GROUP_NORM_EXPAND``), the same axiomatic role
#: ``layer_norm_expand_plain`` / ``layer_norm_expand_affine`` play for
#: ``layer_norm``.  They closed the derivability hole
#: ``fold:instance_norm`` / ``fold:group_norm`` measured (a paying,
#: verifying fold whose cert saturated to a 30-50-enode fixed point
#: without ever touching the fused op) — and, being ``FUSION``-tagged
#: prior art, they now sit in the sieve's ``known_bank`` floor, so the
#: folds classify ``known`` outright rather than machine-original
#: under the bare bank.  The premise channel still matters for folds
#: whose denotation the bank lacks: the cert leg unions a claim's
#: premise closure into its axiom set and site-verifies every
#: non-axiom step — an anchored derivation credits only premises
#: measured true at the applied site.  The private aliases keep
#: ``pipeline._INORM_EXPAND`` naming the declared premise the referee
#: tests reach for.
_INORM_EXPAND = INSTANCE_NORM_EXPAND
_GNORM_EXPAND = GROUP_NORM_EXPAND

#: Schema name → its declared premise closure.  ``_schema_proposal``
#: carries it onto the minted claim as ``rule.premises`` — the cert
#: leg's anchored axiom set.  Both members are shipped prior-art laws
#: now; the mechanism stays for the next fold whose denotation the
#: bank lacks.
_FUSION_PREMISES: dict[str, tuple[Rewrite, ...]] = {
    "fold:instance_norm": (_INORM_EXPAND,),
    "fold:group_norm": (_GNORM_EXPAND,),
}

#: Kernel-fold classes the shipped bank lacks — measured true on the
#: zoo-real corpus before entering the table.  Each entry is a
#: ``Rewrite`` built at import: ``(lhs, rhs, cond, dspec)`` follows the
#: same declarative posture as ``tensor.py``, so the minted candidate
#: serializes side-condition and all.
_FUSION_SCHEMAS: tuple[Rewrite, ...] = (
    # add(mm(A, Bᵀ), C) IS linear(A, B, C) — the matmul-bias epilogue
    # the quant modules spell by hand.  axes-last2 guards the transpose
    # to the last-two swap (linear's weight convention); a transpose on
    # any other axis pair declines.
    Rewrite(
        "fold:addmm_linear",
        Op.make(
            "add",
            Op.make(
                "matmul",
                "A",
                Op.make("transpose", "B", dim0="T0", dim1="T1"),
            ),
            "C",
        ),
        Op.make("linear", "A", "B", "C"),
        cond=("axes-last2", "B", "T0", "T1"),
    ),
    # (x-μ)·rsqrt(E[(x-μ)²]+ε)·w+b IS instance_norm(x, w, b): the
    # unsqueeze-broadcast per-channel norm chain.  eps must be a
    # numeric Const leaf (the kernel's float attr); the affine
    # reshapes pin the (1, C, 1, 1) broadcast spelling.
    Rewrite(
        "fold:instance_norm",
        _INORM_LHS,
        _INORM_RHS,
        cond=(
            "and",
            ("const-num", "EPS"),
            ("attr-len", "WS", "==", 4),
            ("attr-len", "BS", "==", 4),
        ),
        dspec={"EP": ("float", ("const", "EPS"))},
    ),
    # reshape((x'-μ)·rsqrt(E[(x'-μ)²]+ε))·w+b IS group_norm(x, w, b, G):
    # the reshape-to-(1, G, ·) group fold.  keepdim must hold (a
    # dropped reduce axis cannot broadcast back), the reduce must be
    # the reshape target's last axis, and ``num_groups`` derives from
    # the target's second dim.
    Rewrite(
        "fold:group_norm",
        _GNORM_LHS,
        _GNORM_RHS,
        cond=(
            "and",
            ("const-num", "EPS"),
            ("attr-is", "MK", True),
            ("attr-eq", "MD", (-1,)),
            ("attr-len", "US", "==", 3),
            ("attr-len", "WS", "==", 4),
            ("attr-len", "BS", "==", 4),
        ),
        dspec={
            "G": ("nth", ("attr", "US"), 1),
            "EP": ("float", ("const", "EPS")),
        },
    ),
    # mul(sigmoid(u), u) IS silu(u) — the commuted spelling ``silu_fold``
    # pins only in its operand order.  No side condition: the op tree
    # is the whole claim.
    Rewrite(
        "fold:silu_comm",
        Op.make("mul", Op.make("sigmoid", "u"), "u"),
        Op.make("silu", "u"),
    ),
    # u/sqrt(mean(u^2)+eps) IS rms_norm(u): the div/sqrt spelling of the
    # shipped pow-mean-rsqrt fold — same guard shape (keepdim,
    # numeric eps, literal square, trailing reduce block).
    Rewrite(
        "fold:rms_div",
        Op.make(
            "div",
            "u",
            Op.make(
                "sqrt",
                Op.make(
                    "add",
                    Op.make(
                        "mean",
                        Op.make("pow", "u", "P"),
                        dim="MD",
                        keepdim="MK",
                    ),
                    "EPS",
                ),
            ),
        ),
        Op.make("rms_norm", "u", dim="ND", eps="EP"),
        cond=(
            "and",
            ("attr-is", "MK", True),
            ("const-num", "EPS"),
            ("const-cmp", "P", "==", 2),
            ("shaped", ("tail-block", "u", "MD")),
        ),
        dspec={
            "ND": ("shape", ("tail-block", "u", "MD")),
            "EP": ("float", ("const", "EPS")),
        },
    ),
)


def _schema_proposal(rule: Rewrite) -> Proposal:
    """Mint one fusion-schema candidate from a binding rule.

    ``rule`` is the schema — a shipped fold/fuse law or a
    ``_FUSION_SCHEMAS`` entry.  The declarative ``cond``/``dspec`` pass
    through verbatim (the minted rule serializes like the shipped
    one); a *procedural* check/derive (``conv_bn_fold``'s transported
    hooks) passes through too — the ledger's ``missing_hooks`` then
    reports them honestly.
    """
    from mlopt_core.laws.serialize import _proc_check, _proc_derive

    return Proposal(
        name=f"fold:{rule.name.removeprefix('fold:')}",
        lhs=rule.lhs,
        rhs=rule.rhs,
        family="fusion-schema",
        sources=("fusion-schema", rule.name),
        cond=rule.cond,
        dspec=rule.dspec,
        check=rule.check if _proc_check(rule) else None,
        derive=rule.derive if _proc_derive(rule) else None,
        premises=_FUSION_PREMISES.get(rule.name, ()),
    )


def _fusion_schemas(terms: list[Any]) -> list[Proposal]:
    """Mint fusion-class candidates bound to real subterms.

    Every shipped fold/fuse law plus the ``_FUSION_SCHEMAS`` classes
    is tried as a recognizer over *terms*: a schema mints only when a
    subterm satisfies its full side condition (the rule's own
    ``check``/``derive``, exercised through the same
    ``_instance_from_match`` path the oracles use) — so the pool
    carries no fold-class claim whose real region is dead by its own
    guard, and the recorded ``instance`` is the module's own spelling.
    """
    subs = [
        s for t in terms for s in _iter_subterms(t) if isinstance(s, Op)
    ]
    out: list[Proposal] = []
    for rule in [*_fusion_laws(), *_FUSION_SCHEMAS]:
        p = _schema_proposal(rule)
        inst = _instance_from_match(p, subs)
        if inst is None:
            continue
        out.append(replace(p, instance=inst))
    return out


def _binds_any(lhs: Any, subterms: list[Any]) -> bool:
    """Return whether *lhs* binds at least one of the *subterms*.

    The binding is ``_term_match`` — the same match the referee
    context's ``real_terms`` domain measures (one accepted-or-declined
    site is already ``envs >= 1``).  A proposal that binds nothing
    among the census's own terms is a candidate whose whole real
    region is empty: it can still be judged on the synthesized domain,
    but it is not a claim about *this* corpus.
    """
    return any(_term_match(lhs, s) is not None for s in subterms)


def _binding_filter(
    proposals: list[Proposal], terms: list[Any]
) -> list[Proposal]:
    """Keep only *proposals* whose LHS binds a subterm of *terms*."""
    subterms = [
        s for t in terms for s in _iter_subterms(t) if isinstance(s, Op)
    ]
    return [p for p in proposals if _binds_any(p.lhs, subterms)]


def propose(
    census_op: dict,
    terms: list[Any],
    vocab: str = "derived",
    *,
    require_bind: bool = False,
) -> list[Proposal]:
    """Collect, unify and de-duplicate every proposer's candidates.

    Six sources feed the pool: the **census** generators (same-view
    naturality and the mixed-view family — including the
    attr-independent same-view variants — both over the frequent
    op-tuples and the requested op *vocab*), **pattern recognition**
    (composed-then-reduced chains like the ``exp/sum`` softmax fold),
    the **fusion schemas** (shipped fold/fuse laws plus the
    ``_FUSION_SCHEMAS`` kernel-fold classes, minted only where a real
    subterm satisfies the schema's side condition — the
    module-bound-instance mint), the shape-aware schemas
    (``mlopt_discovery.shape_proposal.schemas``), and the
    algebraic grammar (``mlopt_discovery.proposal.schema_candidates``, abstracted
    to patterns with its concrete instance retained for the oracles).
    De-dup is by ``mlopt_discovery.proposal._key`` — the alpha-normal equality —
    and a duplicate's provenance is merged into ``sources``, so a
    candidate reachable from the census generator is recorded as such.

    ``require_bind`` is the binding-first gate: keep only proposals
    whose LHS pattern matches at least one ``Op`` subterm of *terms*
    — the corpus the census was read from.  Callers whose downstream
    domain *is* the corpus (the ``zoo-real`` overnight arm mints from
    a module's own term and judges against real subterm matches) opt
    in so minting effort is spent only on claims that can fire there;
    the gauntlet itself is untouched — a bound-but-false claim is
    still refused.  The default ``False`` keeps the historical pool
    for the grammar-driven callers, whose oracle domain is
    synthesized rather than the census corpus.
    """
    pointwise, views = _vocab_sets(vocab)
    by_key: dict = {}
    pool = [
        *_census_naturality(census_op, terms, pointwise, views),
        *_census_mixed_naturality(census_op, terms, pointwise, views),
        *_pattern_recognition(census_op),
        *_fusion_schemas(terms),
        *_shape_aware(),
        *_grammar(),
    ]
    for p in pool:
        key = lp._key(p.lhs, p.rhs)
        cur = by_key.get(key)
        if cur is None:
            by_key[key] = p
        else:
            by_key[key] = replace(
                cur,
                sources=tuple(dict.fromkeys(cur.sources + p.sources)),
            )
    out = list(by_key.values())
    if require_bind:
        out = _binding_filter(out, terms)
    return out


# ---------------------------------------------------------------------------
#  Evidence — the measured record the ranking reads
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """One proposal plus every measured verdict.

    The properties ``truth`` and ``shippable`` are the verdict; every
    field above them is raw evidence, so the report can show *why*.

    ``fires`` is the e-graph's merged-application count
    (``rule_fires``), split by the typedness audit into
    ``fires_typed`` + ``fires_ill_typed``: a fire whose instantiated
    RHS does not shape-resolve or does not evaluate minted a member
    that cannot denote.  ``changed`` / ``paid`` count only cases
    whose extracted winner is well-typed — a cost drop produced by
    an ill-typed member is suppressed into ``paid_ill_typed`` (and
    the case named in ``ill_typed_cases``), since cost-dropping an
    invalid program is not evidence the law pays.  ``reach_ill``
    counts saturation rows where the with-rule cost drop was
    likewise suppressed.
    """

    proposal: Proposal
    census_sites: int = 0
    relaxed: int = 0
    matches: int = 0
    example: str = ""
    match_term: Any = None
    num_true: bool | None = None
    derivable: bool = False
    witness: tuple[str, ...] = ()
    #: The certificate ``verify_law`` reported for ``derivable`` —
    #: kept so the admission gauntlet can audit which rules the
    #: derivation actually used (premise-anchored derivability
    #: site-verifies every non-axiom step before crediting it).
    #: ``None`` when no derivation was found, and never restored from
    #: a cached verdict row.
    derivation_cert: Any = None
    #: ``True`` when ``derivable`` was credited under the stored
    #: object's premise closure rather than ``base_rules`` alone —
    #: the proof it was minted under, site-verified by the gauntlet.
    premise_anchored: bool = False
    relation: str = "new"
    fires: int = 0
    fires_typed: int = 0
    fires_ill_typed: int = 0
    fire_cases: tuple[str, ...] = ()
    ill_typed_cases: tuple[str, ...] = ()
    changed: int = 0
    paid: int = 0
    paid_ill_typed: int = 0
    verify_fail: int = 0
    reach: tuple[dict, ...] = ()
    cost_drop: float = 0.0
    reach_ill: int = 0
    cert_fail: int = 0
    closure_ratio: float = 1.0
    # The view/index oracle's verdict for view-family candidates
    # (``mlopt_discovery.oracle``): ``conditional`` candidates carry
    # the separating guard in ``view_guard`` — evidence for a
    # guarded law, reported for review, never auto-admitted.
    view_verdict: str = ""
    view_guard: str = ""
    view_note: str = ""

    @property
    def truth(self) -> bool:
        """True iff derivable, or numerically true on a real instance.

        A measured ``num_true is False`` is a counterexample — it
        outranks a derivation, matching the gauntlet's truth gate and
        the store-boundary verdict.  ``derivable`` may only waive an
        *absent* measurement (``num_true is None``).
        """
        if self.num_true is False:
            return False
        return self.derivable or self.num_true is True

    @property
    def closure_safe(self) -> bool:
        """True iff adding the rule did not blow up the closure.

        The raw ``closure_ratio`` (``out/in``) conflates two things:
        runaway growth — the search hazard the gate exists to refuse —
        and a large but *bounded* firing footprint, which a composite
        legitimately carries (its spelled-out premise structure mints
        many enodes in a single application; on a small base graph the
        ratio is inflated by the denominator, not by the growth).

        The honest bound is therefore on *unexplained* growth, per
        reach row: the marginal ``add - base`` must either fit the
        legacy slack ``(limit - 1) * base_enodes`` — the old bound,
        unchanged — or exceed it only by enodes the candidate's own
        recorded applications minted (``cand_enodes``) plus a bounded
        cascade factor for the secondary firings those minted members
        wake in the base bank.  A candidate whose minting exhausted
        its rule budget (``cand_capped``) has a truncated footprint —
        it cannot prove bounded and gets no forgiveness.  Rows
        measured before the footprint audit (no ``cand_enodes`` key)
        and cached verdicts (no reach rows at all) keep the plain
        ratio bound.
        """
        rows = [r for r in self.reach if r.get("base_enodes")]
        if not rows:
            return self.closure_ratio <= _CLOSURE_LIMIT
        for r in rows:
            marginal = r["add_enodes"] - r["base_enodes"]
            if marginal <= (_CLOSURE_LIMIT - 1.0) * r["base_enodes"]:
                continue
            minted = r.get("cand_enodes", 0)
            excess = marginal - minted
            if r.get("cand_capped", True) or excess > (
                _CLOSURE_CASCADE_FACTOR * minted
            ):
                return False
        return True

    @property
    def shippable(self) -> bool:
        """True iff every gate a library entry must clear holds."""
        return (
            self.truth
            and self.relation == "new"
            and self.fires > 0
            and self.fires_ill_typed == 0
            and self.paid > 0
            and self.verify_fail == 0
            and self.cert_fail == 0
            and self.closure_safe
        )

    @property
    def no_ship_reason(self) -> str:
        """Return the first gate a no-ship candidate fails."""
        if self.shippable:
            return ""
        if not self.truth:
            if self.view_verdict == "conditional":
                return (
                    "conditional truth (view-oracle): "
                    f"{self.view_guard or 'guard not isolated'}"
                )
            if self.view_verdict == "ill-formed":
                return "ill-formed RHS (view-oracle)"
            if self.num_true is False:
                return "false (numeric oracle rejects)"
            if self.matches == 0:
                return "inapplicable (no real match)"
            return "unproven (no oracle)"
        if self.relation != "new":
            return f"not new ({self.relation})"
        if self.fires == 0:
            return "no firing on a real model"
        if self.paid == 0:
            if self.paid_ill_typed:
                return (
                    "pays only on ill-typed sites "
                    f"({self.paid_ill_typed} suppressed, "
                    f"{self.fires_ill_typed}/{self.fires} fires "
                    "ill-typed)"
                )
            if self.fires_typed == 0:
                return (
                    "every fire mints an ill-typed member "
                    f"({self.fires_ill_typed}/{self.fires})"
                )
            return "fires but never lowers cost" + (
                f" ({self.fires_ill_typed}/{self.fires} fires "
                "ill-typed)"
                if self.fires_ill_typed
                else ""
            )
        if self.fires_ill_typed:
            return (
                "mints ill-typed members on real sites "
                f"({self.fires_ill_typed}/{self.fires} fires)"
            )
        if self.verify_fail:
            return "lowered modules differ"
        if self.cert_fail:
            return "certificate fails to replay"
        return "closure blow-up"


# ---------------------------------------------------------------------------
#  Measurement — firing, cost delta, certificate, closure safety
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  Typedness audit — a fire is evidence only if the minted RHS denotes
# ---------------------------------------------------------------------------
#
# ``eg.rule_fires`` counts every application that merged — it says
# nothing about whether the instantiated RHS was well-typed.  The
# view/index candidates mint members like ``mul(u, v)`` on bindings
# where ``u``/``v`` do not broadcast, or ``add(u, v)`` where the
# bound ``u`` evaluates to a *tuple* (``getitem`` over
# ``topk``/``var_mean``): the member enters the e-class anyway, the
# cost model prices it (an unknown shape falls back to ~free — see
# ``cost/basic.py``), extraction can pick it, and the probe would
# record a cost drop on a program that does not denote.  The audit
# below re-runs the lone rule on a proof-tracking e-graph (the
# default truncation level already records every application and
# merge) and classifies each merged fire by shape-resolving — and,
# when an input env is buildable, evaluating — the instantiated RHS.


def _bad_subterms(term: Any, memo: dict | None = None) -> frozenset:
    """Return the subterms of *term* that do not shape-resolve.

    A subterm is bad when ``mlopt_core.typing._shape_of`` raises,
    returns ``_INVALID`` or ``None``, or returns a shape carrying a
    non-int/negative dim.  The walk is per-subterm, not root-only:
    ``None`` (unknown) is a wildcard inside ``_broadcast``, so a bad
    inner node does not always propagate to the root.  ``memo`` is a
    shared ``_shape_of`` cache — extracted terms are DAGs.
    """
    if not isinstance(term, Op):
        return frozenset()
    memo = {} if memo is None else memo
    bad: set = set()
    for sub in _iter_subterms(term):
        try:
            s = _shape_of(sub, memo)
        except Exception:
            bad.add(sub)
            continue
        if not isinstance(s, tuple) or any(
            not isinstance(d, int) or d < 0 for d in s
        ):
            bad.add(sub)
    return frozenset(bad)


def _evals(term: Any) -> bool | None:
    """Return whether *term* fp64-evaluates; ``None`` = cannot tell.

    The eval leg exists because shape inference cannot see every
    invalid member: ``transpose``/``select``/``unsqueeze`` attrs are
    ``%``-normalised inside the shape rules, so an out-of-range axis
    shape-checks fine and only fails at evaluation — and a
    tuple-valued operand (``topk``/``var_mean`` under ``getitem``)
    reports a tensor-looking shape but fails at eval.  ``None``
    means a leaf carried a non-int dim, so no env exists and the
    shape verdict stands alone.
    """
    try:
        env = lvo._env_for(term)
    except Exception:
        return None
    if env is None:
        return None
    ok, _ = lvo._eval(term, env)
    return ok


def _term_typed(term: Any) -> bool:
    """Return whether every subterm shape-resolves and term evaluates."""
    if _bad_subterms(term):
        return False
    return _evals(term) is not False


def _pick_ill_typed(best: Any, *refs: Any) -> bool:
    """Return whether the pick *best* carries a minted bad member.

    A bad subterm already present in a *refs* term (the input, or the
    base-ruleset extraction) is not this candidate's mint and does
    not count.  When no shape-bad subterm is new, eval decides:
    *best* failing to evaluate where a reference evaluates cleanly
    means the picked member does not denote (an attr the shape rules
    ``%``-normalised into range, a tuple operand read as a tensor).
    If every reference also fails to evaluate, the invalidity is
    ambient — not attributable to this rule — and the pick stands.
    """
    new_bad = _bad_subterms(best)
    for ref in refs:
        new_bad -= _bad_subterms(ref)
    if new_bad:
        return True
    return _evals(best) is False and any(
        _evals(r) is True for r in refs
    )


def _subst_key(subst: Any) -> tuple:
    """Return a hashable normal form of a fired binding.

    Application records carry the substitution as a dict while
    merge-log edges freeze it as a sorted ``(key, value)`` tuple;
    values are e-class ids and attribute values, repr-normalised so
    either form keys the same multiset when an unhashable attr value
    shows up.
    """
    items = subst.items() if isinstance(subst, dict) else subst
    return tuple(sorted((k, repr(v)) for k, v in items))


def _app_typed(eg: Any, proposal: Proposal, app: dict) -> bool:
    """Return whether the application's instantiated RHS is well-typed.

    The recorded substitution maps metavariables to e-class ids;
    ``eg._any_term_cached`` resolves each to the same minimum-size
    member the rule's own ``check``/``derive`` hooks saw, and the RHS
    instantiates at term level exactly as ``EGraph._instantiate``
    minted it at enode level — including the ``derive``-produced
    ``$attr:`` bindings, which the application record carries.  A
    binding that no longer resolves, or an instantiation that
    raises, cannot be certified typed and counts as ill-typed,
    never silently as typed.
    """
    bound: dict = {}
    for k, v in app["subst"].items():
        if k.startswith("$attr:"):
            bound[k] = v
            continue
        term = eg._any_term_cached(v)
        if term is None:
            return False
        bound[k] = term
    try:
        rhs = _term_instantiate(proposal.rhs, bound)
    except Exception:
        return False
    return _term_typed(rhs)


@dataclass(frozen=True)
class _TypedAudit:
    """The typedness split of a lone rule's firing on one case."""

    fires: int = 0
    typed: int = 0
    ill: int = 0
    pick_ill: bool = False


def _typed_probe(
    case: TermCase, proposal: Proposal, cost_fn: Any
) -> _TypedAudit:
    """Re-fire the candidate on a tracked e-graph; audit the fires.

    ``_probe`` reports the firing *count* and the cost delta; this
    pass classifies each merged application.  The audit rebuilds the
    identical run (one rule, same bounds — ``EGraph`` at the default
    truncation level records every application and every merge), so
    ``audit.fires`` equals the probe's count.  Aligning the recorded
    applications to the rule's ``merge_log`` edges by frozen binding
    recovers which applications merged: an application is recorded
    when it merged *or minted enodes*, but only merges count as
    fires, and within one binding the merged applications form a
    prefix — once a merge lands, a later same-binding application
    finds the classes already joined and can never merge again.

    The pick check asks whether the extracted winner denotes: a cost
    drop produced by an ill-typed member is not evidence the law
    pays.  This second saturation runs only on cases the probe
    reported firing, so the audit adds one bounded lone-rule run per
    firing (case, proposal) pair — nothing when the rule did not
    fire.
    """
    rule = proposal.as_rule()
    eg = EGraph()
    root = eg.add_term(case.term)
    eg.run(
        [rule],
        root,
        max_iterations=_FIRING_ITERS,
        max_nodes=_FIRING_NODES,
    )
    fires = eg.rule_fires.get(rule.name, 0)
    typed = ill = 0
    if fires:
        merges = Counter(
            _subst_key(e.subst)
            for e in eg.merge_log
            if e.rule == rule.name
        )
        seen: Counter = Counter()
        for app in eg.applications:
            if app["rule"] != rule.name:
                continue
            key = _subst_key(app["subst"])
            seen[key] += 1
            if seen[key] > merges.get(key, 0):
                # Recorded for enode creation without a merge —
                # ``rule_fires`` does not count it either.
                continue
            if _app_typed(eg, proposal, app):
                typed += 1
            else:
                ill += 1
    pick_ill = False
    if fires:
        best = eg.extract_best(root, cost_fn)
        if best is not None and best != case.term:
            pick_ill = _pick_ill_typed(best, case.term)
    return _TypedAudit(
        fires=fires, typed=typed, ill=ill, pick_ill=pick_ill
    )


def _reach_row(
    case: TermCase,
    base_rules: list[Rewrite],
    laws: list[Rewrite],
    cost_fn: Any,
) -> dict:
    """Compare saturation of *case* with vs without *laws*.

    This is ``mlopt_discovery.impact.reach_row`` with the base rule set
    parameterized (the impact tool hard-codes ``ALL_RULES``; the
    held-out run needs a different base).  It reuses the impact tool's
    ``_saturate`` / ``_cert_ok`` helpers, and adds nothing but the
    base argument.
    """
    base_eg, _br, base_best, base_stats = _saturate(
        case.term, list(base_rules), cost_fn
    )
    add_eg, _ar, add_best, add_stats = _saturate(
        case.term, [*base_rules, *laws], cost_fn
    )
    fires = {
        r.name: add_eg.rule_fires.get(r.name, 0)
        for r in laws
        if add_eg.rule_fires.get(r.name, 0)
    }
    base_cost = (
        dag_cost(base_best, cost_fn) if base_best else float("inf")
    )
    add_cost = dag_cost(add_best, cost_fn) if add_best else float("inf")
    # A drop only counts when the cheaper extraction denotes — an
    # ill-typed member minted by the candidate can look arbitrarily
    # cheap (an unknown shape falls back to ~free), so a with-rule
    # extraction carrying a bad subterm neither the input nor the
    # base extraction had is suppressed rather than accrued.
    add_typed = True
    if (
        add_best is not None
        and base_cost not in (0.0, float("inf"))
        and add_cost < base_cost
    ):
        add_typed = not _pick_ill_typed(add_best, case.term, base_best)
    # The candidate's own firing footprint: the enodes its recorded
    # applications created (``_enode_app`` tracks provenance under
    # level-2 truncation, which ``_saturate``'s default ``EGraph``
    # runs).  Enodes the term or witness merges created carry no
    # application and count as unexplained — conservative.
    cand_names = {r.name for r in laws}
    apps = add_eg.applications
    minted = sum(
        1
        for ai in getattr(add_eg, "_enode_app", {}).values()
        if apps[ai]["rule"] in cand_names
    )
    return {
        "model": case.name,
        "base_enodes": base_stats["n_enodes"],
        "add_enodes": add_stats["n_enodes"],
        "cand_enodes": minted,
        "cand_capped": bool(
            cand_names & set(add_stats.get("budget_suspended", ()))
        ),
        "base_cost": base_cost,
        "add_cost": add_cost,
        "changed": add_best != base_best,
        "add_typed": add_typed,
        "new_fires": fires,
        "base_cert": _cert_ok(base_eg, case.term, base_best, cost_fn),
        "add_cert": _cert_ok(add_eg, case.term, add_best, cost_fn),
    }


def _instance_from_match(
    proposal: Proposal, matches: list[Any]
) -> tuple[Any, Any] | None:
    """Instantiate *proposal*'s RHS on its first viable real match.

    Applies the proposal's ``check`` / ``derive`` hooks (when present)
    to each match's substitution — a check veto or a ``None`` derive
    skips that match, exactly as an e-graph firing would — so the
    oracles see the same instance the rule would produce.  The hooks
    are read off the materialized ``as_rule()`` so a declarative
    ``cond``/``dspec`` gates the same way the procedural hooks do.
    """
    rule = proposal.as_rule()
    for sub in matches:
        subst = _term_match(proposal.lhs, sub)
        if subst is None:
            continue
        if rule.check is not None:
            try:
                if not rule.check(subst):
                    continue
            except Exception:
                continue
        inst = dict(subst)
        if rule.derive is not None:
            try:
                extra = rule.derive(subst)
            except Exception:
                continue
            if extra is None:
                continue
            inst.update(extra)
        return sub, _term_instantiate(proposal.rhs, inst)
    return None


def _fire(
    proposal: Proposal,
    models: list[TermCase],
    sink: Any,
    cost_fn: Any,
    ev: Evidence,
) -> None:
    """Fire *proposal* alone over every model; fill the firing fields.

    ``_probe`` measures fires, the cost delta and the lowered-module
    verify; ``_typed_probe`` audits whether each merged fire minted
    a well-typed member and whether the extracted winner denotes.
    ``ev.fires`` still counts every merged fire — the split into
    ``fires_typed`` / ``fires_ill_typed`` makes it honest — but a
    case whose cheaper extraction is ill-typed contributes neither
    ``changed`` nor ``paid`` (a cost drop on a program that does not
    denote is not evidence), nor a ``verify_fail`` (the lowering was
    doomed, not disagreeing); the suppressed drop is recorded in
    ``paid_ill_typed`` so the inflation stays visible.
    """
    rule = proposal.as_rule()
    cases: list[str] = []
    ill_cases: list[str] = []
    for case in models:
        f = _probe(case, rule, sink, cost_fn)
        if not f.fires:
            continue
        ev.fires += f.fires
        cases.append(case.name)
        audit = _typed_probe(case, proposal, cost_fn)
        ev.fires_typed += audit.typed
        ev.fires_ill_typed += audit.ill
        if audit.pick_ill:
            ill_cases.append(case.name)
            if f.paid:
                ev.paid_ill_typed += 1
            continue
        if f.verified in ("FAIL", "error"):
            ev.verify_fail += 1
        if f.changed:
            ev.changed += 1
        if f.paid:
            ev.paid += 1
    ev.fire_cases = tuple(cases)
    ev.ill_typed_cases = tuple(ill_cases)


def _reach(
    proposal: Proposal,
    models: list[TermCase],
    base_rules: list[Rewrite],
    cost_fn: Any,
    ev: Evidence,
) -> None:
    """Measure end-to-end reach, certificate and closure ratio."""
    rows = tuple(
        _reach_row(c, base_rules, [proposal.as_rule()], cost_fn)
        for c in models
    )
    ev.reach = rows
    drops: list[float] = []
    ratios: list[float] = []
    for r in rows:
        base, add = r["base_cost"], r["add_cost"]
        if base not in (0.0, float("inf")) and add < base:
            if r["add_typed"]:
                drops.append((base - add) / base)
            else:
                ev.reach_ill += 1
        if r["base_enodes"]:
            ratios.append(r["add_enodes"] / r["base_enodes"])
        if r["add_cert"] != "pass":
            ev.cert_fail += 1
    ev.cost_drop = max(drops) if drops else 0.0
    ev.closure_ratio = max(ratios) if ratios else 1.0


def _lhs_tuple(term: Any) -> tuple | None:
    """Return the census op-tuple key ``(op, child-op tuple)`` of *term*.

    The same key ``mlopt_discovery.census.op_tuple_census`` counts, so a
    candidate's LHS can be located in the census directly — the
    census -> propose link made explicit.
    """
    if not isinstance(term, Op):
        return None
    return (term.op, tuple(_op_of(a) for a in term.args))


def _has_view_op(proposal: Proposal) -> bool:
    """Return whether the proposal's patterns contain a view/index op.

    The view oracle's scope: ``select``/``slice``/``getitem``/
    ``unsqueeze``/``transpose``/``reshape``/``chunk``/… — the ops whose
    naturality laws need shape/index instantiation the generic numeric
    oracle cannot do honestly.
    """
    return any(
        isinstance(t, Op) and t.op in lvo._VIEWISH
        for t in [
            *_iter_subterms(proposal.lhs),
            *_iter_subterms(proposal.rhs),
        ]
    )


def _derivable_result(
    inst: tuple, base_rules: list[Rewrite], premise_rules: Iterable
) -> Any:
    """Return the derivability oracle's result, premise-anchored on a miss.

    ``verify_law`` under ``base_rules`` first — an axiomatic
    derivation is the stronger claim.  When that fails and
    *premise_rules* is non-empty, re-run under ``base_rules +
    premise_rules`` (the stored object's minting premises); the
    anchored result is returned only when it actually derives — the
    base verdict is the honest fallback the gauntlet revokes *to*.
    """
    res = verify_law(inst[0], inst[1], base_rules)
    anchored_rules = [*base_rules, *premise_rules]
    if res.derivable or len(anchored_rules) == len(base_rules):
        return res
    anchored = verify_law(inst[0], inst[1], anchored_rules)
    return anchored if anchored.derivable else res


def measure(
    proposal: Proposal,
    real_terms: list[Any],
    models: list[TermCase],
    base_rules: list[Rewrite],
    lib: list,
    census_op: dict,
    sink: Any,
    cost_fn: Any,
    view_oracle: bool = True,
    premise_rules: Iterable[Rewrite] = (),
) -> Evidence:
    """Run both oracles and every measurement for one *proposal*.

    *premise_rules* is the admission gauntlet's premise-anchored
    derivability channel: when the candidate is not derivable under
    ``base_rules`` alone, the prover re-runs under ``base_rules +
    premise_rules`` — the stored object's minting premises.  Only the
    ``derivable`` leg sees them; the reach baseline, the census and
    the novelty relation all stay scoped to ``base_rules``.  The
    reported ``res.cert`` rides home on ``ev.derivation_cert`` so the
    caller can audit which rules the derivation used — an anchored
    ``derivable`` is credit *pending* that audit
    (:func:`mlopt_discovery.evidence._premise_steps_verify`).
    """
    ev = Evidence(proposal=proposal)
    ev.census_sites = census_op.get(_lhs_tuple(proposal.lhs), 0)
    schema = Schema(proposal.name, proposal.lhs, proposal.rhs)
    ev.relaxed = relaxed_matches(real_terms, schema)
    matches = real_matches(real_terms, schema)
    ev.matches = len(matches)
    ev.relation = lp._relation(proposal.lhs, proposal.rhs, lib)
    inst = proposal.instance or _instance_from_match(proposal, matches)
    if inst is not None:
        ev.num_true = lp._numeric_true(inst[0], inst[1])
        res = _derivable_result(inst, base_rules, premise_rules)
        ev.derivable = res.derivable
        ev.witness = res.witness_rules
        ev.derivation_cert = res.cert
    # The view/index oracle resolves every non-derivable view-family
    # candidate — it sweeps *every* real match (the stock oracle reads
    # only the first) and synthesizes satisfiable instantiations (view
    # attrs + leaf shapes, tuple-sources for getitem).  It also
    # *downgrades*: a single-instance ``True`` that is only
    # conditionally true is reported as ``conditional``, which keeps
    # the candidate out of the ship set until its guard is written.
    if view_oracle and not ev.derivable and _has_view_op(proposal):
        vo = lvo.verify_view_candidate(
            proposal.name,
            proposal.lhs,
            proposal.rhs,
            matches,
            check=proposal.check,
            derive=proposal.derive,
        )
        ev.view_verdict = vo.verdict
        ev.view_guard = vo.guard
        ev.view_note = vo.note
        if vo.verdict == "true":
            ev.num_true = True
        elif vo.verdict in ("false", "ill-formed"):
            ev.num_true = False
        elif vo.verdict == "conditional":
            ev.num_true = None
    if matches:
        ev.example = op_repr(matches[0])
        ev.match_term = matches[0]
    _fire(proposal, models, sink, cost_fn, ev)
    if ev.fires:
        _reach(proposal, models, base_rules, cost_fn, ev)
    return ev


def _evidence_from_row(proposal: Proposal, row: dict) -> Evidence:
    """Reconstruct the measured record a cached verdict row carries.

    Some fields are not restored: ``match_term`` is not
    serializable, ``reach``'s per-model rows are only read as
    aggregates (``cost_drop`` / ``cert_fail`` / ``closure_ratio``,
    all stored), and the typedness audit's ``fires_typed`` /
    ``fires_ill_typed`` / ``paid_ill_typed`` / ``ill_typed_cases`` /
    ``reach_ill`` have no columns in the (frozen) verdict schema —
    a cached row restores them at their defaults, so a cache-served
    candidate under-reports its ill-typed evidence while the gates
    it affects (``fires``, gated ``paid``) still round-trip.
    Everything else the ranking, the report, the JSON dump and the
    admission emitter consume round-trips verbatim — the emitter
    binds its test substitutions from ``fire_cases``' live model
    terms, not ``match_term``.
    """
    ev = Evidence(proposal=proposal)
    ev.census_sites = row["census_sites"]
    ev.relaxed = row["relaxed"]
    ev.matches = row["matches"]
    ev.example = row["example"]
    nt = row["numeric_true"]
    ev.num_true = None if nt is None else bool(nt)
    ev.derivable = bool(row["derivable"])
    ev.witness = tuple(json.loads(row["witness_json"]))
    ev.relation = row["relation"]
    ev.fires = row["fires"]
    ev.fire_cases = tuple(json.loads(row["fire_cases_json"]))
    ev.changed = row["changed"]
    ev.paid = row["paid"]
    ev.verify_fail = row["verify_fail"]
    ev.cost_drop = row["drop_pct"] / 100.0
    ev.cert_fail = row["cert"]
    ev.closure_ratio = row["enode_ratio"]
    return ev


# ---------------------------------------------------------------------------
#  Ranking — the deterministic order and the ship verdict
# ---------------------------------------------------------------------------


def _rank_key(ev: Evidence) -> tuple:
    """Return the documented best-first ordering key for *ev*.

    Shippable candidates first; then the largest end-to-end cost drop,
    the number of models it pays on, and the firing count; then truth,
    applicability and (as a tie-break) the smaller closure ratio.
    """
    return (
        1 if ev.shippable else 0,
        round(ev.cost_drop, 9),
        ev.paid,
        ev.fires,
        1 if ev.truth else 0,
        ev.matches,
        ev.relaxed,
        -ev.closure_ratio,
    )


def rank(evs: list[Evidence]) -> list[Evidence]:
    """Return *evs* sorted best-first by :func:`_rank_key`."""
    return sorted(evs, key=_rank_key, reverse=True)


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def _search_rules(holdout: str | None) -> list[Rewrite]:
    """Return the search rule set, minus *holdout* if given."""
    if holdout is None:
        return list(ALL_RULES)
    names = {n for n in holdout.split(",") if n}
    return [r for r in ALL_RULES if r.name not in names]


def run_pipeline(
    holdout: str | None = None,
    vocab: str = "derived",
    evidence_db: str | None = None,
    use_cache: bool = False,
    view_oracle: bool = True,
) -> dict:
    """Run census -> propose -> verify -> measure -> rank.

    ``holdout`` is a comma-separated list of library rule names to
    remove from the search rule set (the library used for duplicate
    detection, derivability, and the reach baseline) — the held-out
    rediscovery test.  ``vocab`` selects the generator's op alphabet:
    ``"hand"`` (the ``_POINTWISE`` / ``_VIEW_OPS`` tuples) or
    ``"derived"`` (property-classified over the corpus by
    ``mlopt_discovery.vocab``).

    ``view_oracle`` (default on) runs the view/index oracle
    (``mlopt_discovery.oracle``) on every non-derivable candidate
    whose patterns contain a view/index op — resolving the
    ``unproven``/``false`` verdicts the single-instance numeric oracle
    cannot reach honestly (ill-typed instantiations, conditional
    truths, tuple-valued ``getitem`` operands).

    ``evidence_db`` opts into the evidence store
    (``mlopt_discovery.evidence``): after the run every candidate and
    verdict is upserted under a ``(corpus, rules, code)`` content
    key.  ``use_cache`` additionally serves verdicts already recorded
    for *this* key — a hit skips ``measure`` entirely, which is sound
    because a verdict is deterministic given its scope.
    """
    base_rules = _search_rules(holdout)
    lib = [lp._key(r.lhs, r.rhs) for r in base_rules]
    census = run_census(_CENSUS_TOP)
    census_op = {
        (e["op"], tuple(e["children"])): e["count"]
        for e in census["op_tuples"]
    }

    bench, _be = _bench_cases()
    models, _me = model_cases()
    # The side-file census (mlopt_discovery.intake): exported workloads
    # union into the matcher terms; the probe-eligible subset joins
    # the firing/reach stage.  No file -> both lists are empty.
    from mlopt_discovery import intake as li

    intake = li.load_cases()
    probe = [*models, *li.probe_cases()]
    real_terms = [c.term for c in [*bench, *models, *intake]]
    proposals = propose(census_op, real_terms, vocab)
    sink = _sink()
    cost_fn = _cost_fn(sink)

    conn = ev_store.connect(evidence_db) if evidence_db else None
    corpus_h = rules_h = rev = ""
    if conn is not None:
        corpus_h = ev_store.corpus_hash(
            f"{c.source}:{c.name}:{shape_repr(shape_key(c.term, {}))}"
            for c in [*bench, *models, *intake]
        )
        rules_h = ev_store.rules_hash(
            repr(lp._key(r.lhs, r.rhs)) for r in base_rules
        )
        rev = ev_store.code_rev()
    cached = (
        ev_store.latest_verdicts(conn, corpus_h, rules_h, rev)
        if conn is not None and use_cache
        else {}
    )
    hits = 0
    evs = []
    for p in proposals:
        row = cached.get(repr(lp._key(p.lhs, p.rhs)))
        if row is not None:
            evs.append(_evidence_from_row(p, row))
            hits += 1
        else:
            evs.append(
                measure(
                    p,
                    real_terms,
                    probe,
                    base_rules,
                    lib,
                    census_op,
                    sink,
                    cost_fn,
                    view_oracle=view_oracle,
                )
            )
    if conn is not None:
        meta = {
            "corpus_hash": corpus_h,
            "rules_hash": rules_h,
            "code_rev": rev,
            "run_id": ev_store.new_run_id(),
            "holdout": holdout or "",
            "ts": ev_store.now(),
        }
        ev_store.record_run(
            conn,
            meta,
            (
                ev_store.verdict_row(
                    repr(lp._key(ev.proposal.lhs, ev.proposal.rhs)),
                    ev,
                    op_repr(ev.proposal.lhs),
                    op_repr(ev.proposal.rhs),
                )
                for ev in evs
            ),
        )
        conn.close()
    ranked = rank(evs)
    return {
        "holdout": holdout,
        "vocab": vocab,
        "n_search_rules": len(base_rules),
        "n_bench": len(bench),
        "n_models": len(models),
        "n_intake": len(intake),
        "census": {
            "n_terms": census["n_terms"],
            "n_op_nodes": census["n_op_nodes"],
            "n_op_tuples": len(census["op_tuples"]),
            "n_shapes": census["n_shapes"],
        },
        "proposals": len(proposals),
        "ranked": ranked,
        "held_out": _held_out(ranked, holdout),
        # Kept out of the JSON dump (terms are not serializable); the
        # admission emitter reads the firing case for its e2e test.
        "models": probe,
        # The evidence-store cache record; None when --evidence-db is
        # not given.
        "cache": (
            {
                "db": evidence_db,
                "hits": hits,
                "total": len(proposals),
                "code_rev": rev,
                "corpus_hash": corpus_h[:12],
                "rules_hash": rules_h[:12],
                "use_cache": use_cache,
            }
            if evidence_db
            else None
        ),
    }


def _held_out(ranked: list[Evidence], holdout: str | None) -> dict:
    """Locate the held-out law's candidate in the ranking, if any.

    The held-out rule is taken from the *library* (``ALL_RULES``); its
    alpha-normal equality is matched against every ranked proposal so
    the location is structural, not a name lookup.
    """
    if holdout is None:
        return {"holdout": None, "found": False}
    name = next((n for n in holdout.split(",") if n), "")
    rule = next((r for r in ALL_RULES if r.name == name), None)
    if rule is None:
        return {"holdout": name, "found": False, "note": "unknown rule"}
    key = lp._key(rule.lhs, rule.rhs)
    swap = (key[1], key[0])
    for i, ev in enumerate(ranked, start=1):
        k = lp._key(ev.proposal.lhs, ev.proposal.rhs)
        if k in (key, swap):
            return {
                "holdout": name,
                "found": True,
                "rank": i,
                "of": len(ranked),
                "candidate": ev.proposal.name,
                "shippable": ev.shippable,
                "top": i == 1,
                "sources": list(ev.proposal.sources),
                "census_generated": (
                    "census-naturality" in ev.proposal.sources
                ),
                "census_sites": ev.census_sites,
                "matches": ev.matches,
                "fires": ev.fires,
                "paid": ev.paid,
                "cost_drop": ev.cost_drop,
            }
    return {"holdout": name, "found": False}


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _row(i: int, ev: Evidence) -> str:
    """Render one ranked evidence row."""
    true = {True: "yes", False: "no", None: "-"}[ev.num_true]
    verdict = "SHIP" if ev.shippable else "no"
    return (
        f"{i:>4} {ev.proposal.name:<26} {ev.proposal.family:<18} "
        f"{ev.census_sites:>6} {true:>4} {ev.relation:<9} "
        f"{ev.matches:>5} {ev.fires:>5} {ev.fires_ill_typed:>4} "
        f"{ev.paid:>4} {ev.cost_drop * 100:>6.1f} "
        f"{'pass' if ev.cert_fail == 0 else 'FAIL':>4} "
        f"{ev.closure_ratio:>6.2f}x {verdict:>4}"
    )


def _table(ranked: list[Evidence], top: int) -> str:
    """Render the ranked evidence table."""
    head = (
        f"{'rank':>4} {'candidate':<26} {'family':<18} "
        f"{'census':>6} {'true':>4} {'rel':<9} {'match':>5} "
        f"{'fires':>5} {'ill':>4} {'paid':>4} {'drop%':>6} "
        f"{'cert':>4} {'enode':>7} {'ship':>4}"
    )
    lines = [head, "-" * len(head)]
    for i, ev in enumerate(ranked[:top], start=1):
        lines.append(_row(i, ev))
    return "\n".join(lines)


def _print_report(result: dict, top: int) -> None:
    """Print the full human-readable pipeline report."""
    ranked: list[Evidence] = result["ranked"]
    c = result["census"]
    print(  # stdout-compat
        "== law_pipeline — census -> propose -> verify -> measure =="
    )
    ho = result["holdout"]
    print(  # stdout-compat
        f"   search rule set: {result['n_search_rules']} rules"
        + (f" (held out: {ho})" if ho else " (ALL_RULES)")
    )
    print(  # stdout-compat
        f"   op vocabulary: {result.get('vocab', 'hand')}"
    )
    print(  # stdout-compat
        f"   corpus: {result['n_bench']} bench + "
        f"{result['n_models']} models + "
        f"{result.get('n_intake', 0)} intake"
    )
    print(  # stdout-compat
        f"   census: {c['n_op_nodes']} op nodes, "
        f"{c['n_op_tuples']} op-tuples, {c['n_shapes']} shapes"
    )
    print(f"   proposals: {result['proposals']}")  # stdout-compat
    cache = result.get("cache")
    if cache is not None:
        line = (
            f"   evidence db: {cache['db']} "
            f"(rev {cache['code_rev']}, corpus {cache['corpus_hash']}, "
            f"rules {cache['rules_hash']})"
        )
        if cache["use_cache"]:
            line += (
                f" — {cache['hits']}/{cache['total']} "
                "verdicts served from cache"
            )
        print(line)  # stdout-compat
    ship = [e for e in ranked if e.shippable]
    ill = [e for e in ranked if e.fires_ill_typed]
    print(  # stdout-compat
        f"   firing on a real model: "
        f"{sum(1 for e in ranked if e.fires)} "
        f"({len(ill)} mint ill-typed members, "
        f"{sum(e.fires_ill_typed for e in ranked)} ill-typed fires, "
        f"{sum(e.paid_ill_typed for e in ranked)} suppressed pays); "
        f"shippable: {len(ship)}"
    )
    vres = [e for e in ranked if e.view_verdict]
    if vres:
        from collections import Counter

        tally = Counter(e.view_verdict for e in vres)
        cond = [e for e in vres if e.view_verdict == "conditional"]
        print(  # stdout-compat
            f"   view-oracle: {len(vres)} view candidates — "
            + ", ".join(f"{k}={n}" for k, n in sorted(tally.items()))
        )
        for e in cond:
            print(  # stdout-compat
                f"      {e.proposal.name}: {e.view_guard}"
            )
    print()  # stdout-compat
    print(f"-- ranked candidates (top {top}) --")  # stdout-compat
    print(_table(ranked, top))  # stdout-compat
    print()  # stdout-compat
    print(  # stdout-compat
        "-- ship recommendations (evidence per candidate) --"
    )
    if not ship:
        print(  # stdout-compat
            "  none — no candidate clears the ship bar"
        )
    for i, ev in enumerate(ranked, start=1):
        if not ev.shippable:
            continue
        print(  # stdout-compat
            f"  #{i} {ev.proposal.name} [{ev.proposal.family}] — "
            f"true={ev.num_true} new={ev.relation} "
            f"census={ev.census_sites} match={ev.matches} "
            f"fires={ev.fires} ill={ev.fires_ill_typed} "
            f"paid={ev.paid} "
            f"drop={ev.cost_drop * 100:.1f}% cert=pass "
            f"enode={ev.closure_ratio:.2f}x"
        )
        print(  # stdout-compat
            f"      rule: {op_repr(ev.proposal.lhs)}"
        )
        print(  # stdout-compat
            f"         -> {op_repr(ev.proposal.rhs)}"
        )
        print(  # stdout-compat
            f"      cases: {', '.join(ev.fire_cases)}"
        )
    print()  # stdout-compat
    print("-- why the rest did not ship --")  # stdout-compat
    for i, ev in enumerate(ranked, start=1):
        if ev.shippable:
            continue
        print(  # stdout-compat
            f"  #{i:>2} {ev.proposal.name:<26} {ev.no_ship_reason}"
        )
    print()  # stdout-compat
    _print_held_out(result)


def _print_held_out(result: dict) -> None:
    """Print the held-out rediscovery verdict."""
    held = result["held_out"]
    print("-- held-out rediscovery --")  # stdout-compat
    if not held.get("holdout"):
        print(  # stdout-compat
            "  (no holdout — run with --holdout <rule> to validate)"
        )
        return
    if not held.get("found"):
        print(  # stdout-compat
            f"  held out: {held['holdout']} — NOT rediscovered "
            f"(no structurally-matching candidate was proposed)"
        )
        return
    print(  # stdout-compat
        f"  held out: {held['holdout']} — rediscovered as "
        f"{held['candidate']}: rank {held['rank']} of {held['of']}, "
        f"shippable={held['shippable']}"
    )
    print(  # stdout-compat
        f"  evidence: census LHS sites={held['census_sites']}, "
        f"match={held['matches']}, fires={held['fires']}, "
        f"paid={held['paid']}, drop={held['cost_drop'] * 100:.1f}%"
    )
    print(  # stdout-compat
        f"  proposed by: {', '.join(held['sources'])}"
        f" — census-generated={held['census_generated']}"
    )
    print(  # stdout-compat
        "  verdict: "
        + (
            "PASS — the pipeline ranks the known winner top"
            if held["top"] and held["shippable"]
            else "FAIL — the known winner is not ranked top/shippable"
        )
    )


def _dump_json(path: str, result: dict) -> None:
    """Write the machine-readable pipeline result."""
    ranked: list[Evidence] = result["ranked"]
    payload = {
        "holdout": result["holdout"],
        "vocab": result.get("vocab", "hand"),
        "n_search_rules": result["n_search_rules"],
        "n_bench": result["n_bench"],
        "n_models": result["n_models"],
        "n_intake": result.get("n_intake", 0),
        "census": result["census"],
        "proposals": result["proposals"],
        "held_out": result["held_out"],
        "cache": result.get("cache"),
        "ranked": [
            {
                "rank": i,
                "name": ev.proposal.name,
                "family": ev.proposal.family,
                "sources": list(ev.proposal.sources),
                "lhs": op_repr(ev.proposal.lhs),
                "rhs": op_repr(ev.proposal.rhs),
                "census_sites": ev.census_sites,
                "relaxed": ev.relaxed,
                "matches": ev.matches,
                "example": ev.example,
                "num_true": ev.num_true,
                "derivable": ev.derivable,
                "witness": list(ev.witness),
                "relation": ev.relation,
                "fires": ev.fires,
                "fires_typed": ev.fires_typed,
                "fires_ill_typed": ev.fires_ill_typed,
                "fire_cases": list(ev.fire_cases),
                "ill_typed_cases": list(ev.ill_typed_cases),
                "changed": ev.changed,
                "paid": ev.paid,
                "paid_ill_typed": ev.paid_ill_typed,
                "verify_fail": ev.verify_fail,
                "cost_drop": ev.cost_drop,
                "reach_ill": ev.reach_ill,
                "cert_fail": ev.cert_fail,
                "closure_ratio": ev.closure_ratio,
                "view_verdict": ev.view_verdict,
                "view_guard": ev.view_guard,
                "view_note": ev.view_note,
                "shippable": ev.shippable,
                "no_ship_reason": ev.no_ship_reason,
            }
            for i, ev in enumerate(ranked, start=1)
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and print (or dump) the ranked report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout",
        help="comma-separated library rule names to exclude "
        "(held-out rediscovery)",
    )
    parser.add_argument("--json", help="write machine-readable results")
    parser.add_argument("--top", type=int, default=_DEFAULT_TOP)
    parser.add_argument(
        "--vocab",
        choices=("hand", "derived"),
        default="derived",
        help="the generator's op alphabet: property-classified over "
        "the corpus (default, mlopt_discovery.vocab) or the "
        "hand-written tuples",
    )
    parser.add_argument(
        "--emit-admission",
        metavar="CANDIDATE",
        help="emit the admission artifact for a SHIP candidate "
        "(law + hooks + generated tests + review patch) — see "
        "mlopt_discovery.emit",
    )
    parser.add_argument(
        "--out",
        metavar="DIR",
        default="admission_out",
        help="output directory for --emit-admission "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--evidence-db",
        metavar="PATH",
        help="sqlite evidence store (mlopt_discovery.evidence): upsert "
        "candidates + verdicts after the run, keyed by corpus/rule/"
        "code content hashes.  Keep PATH outside bench/ and scripts/ — "
        "untracked "
        "code-path files are hashed into the cache key",
    )
    parser.add_argument(
        "--use-evidence-cache",
        action="store_true",
        help="serve verdicts already recorded in --evidence-db for "
        "this corpus/rule-set/code-revision key instead of "
        "re-measuring them",
    )
    parser.add_argument(
        "--no-view-oracle",
        action="store_true",
        help="skip the view/index oracle (mlopt_discovery.oracle): "
        "view-family candidates keep the single-instance numeric "
        "verdict — the pre-oracle behaviour",
    )
    args = parser.parse_args(argv)

    if args.use_evidence_cache and not args.evidence_db:
        print(  # stdout-compat
            "note: --use-evidence-cache without --evidence-db — "
            "nothing to read, measuring fresh"
        )
    result = run_pipeline(
        args.holdout,
        args.vocab,
        evidence_db=args.evidence_db,
        use_cache=bool(args.use_evidence_cache and args.evidence_db),
        view_oracle=not args.no_view_oracle,
    )
    _print_report(result, args.top)
    if args.json:
        _dump_json(args.json, result)
        print(f"\nwrote {args.json}")  # stdout-compat
    if args.emit_admission:
        from mlopt_discovery import emit

        em = emit.emit_admission(result, args.emit_admission, args.out)
        print("\n-- admission emission --")  # stdout-compat
        if not em.emitted:
            print(f"  REFUSED: {em.reason}")  # stdout-compat
            return 1
        for f in em.files:
            print(f"  wrote {f}")  # stdout-compat
        for n in em.notes:
            print(f"  note: {n}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

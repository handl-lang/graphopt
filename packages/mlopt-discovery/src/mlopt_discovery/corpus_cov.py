"""Coverage-driven corpus generation — the meta-arena's gap census.

Stage-1 self-driving lever 1 (the corpus half): the meta-arena's
``gen_cases`` stream was *hand*-extended twice (the T-009 coverage
round added ``square``/``dual``/``tri`` kinds by hand).  This module
makes the extension *mechanical*: it enumerates the op / attr /
handler / rule space the arena machinery can express, diffs it
against what a case stream actually contains, and emits the case
kinds that close the measured gaps — re-runnable as the vocabulary
grows (a new ``ELEMENTWISE`` op, ``HANDLERS`` tag, ``REDUCE_DIM``
head or shipped law is picked up by :func:`vocabulary` and reported
by :func:`coverage_gaps` on the next run).

The expressible space (:func:`vocabulary`):

* **ops** — the union of ``genkernel.ELEMENTWISE`` (the fusion
  tier's alphabet, clamps included), ``genkernel.reduce_heads()``
  (``REDUCE_DIM_OPS`` + ``softmax``), every op named by a shipped
  law side (``DEFAULT`` plus the scan tier's ``SCAN_LAWS`` /
  ``SCAN_DIAG_LAWS`` — the carriers ``apply``/``aff``/…), the
  ``HANDLERS`` patterns and kernels, and the ``opmeta`` view/pointwise
  classes (``VIEWISH_OPS`` / ``POINTWISE_OPS`` / ``ACTIVATION_OPS``)
  the IR declares;
* **attrs** — the ``(op, attr)`` pairs ``opmeta.ATTR_SCHEMA``
  declares for those ops;
* **handlers** — the ``lawdata.HANDLERS`` tags;
* **rules** — the shipped law names (``DEFAULT`` + scan tier).

:func:`coverage_gaps` is the diff: ops/attrs/handlers/rules the
corpus never touches.  The first measurement (2026-11, 400-case
stream) found the hand corpus exercises **7 ops, 0 attrs, 3/3
handlers**; only **1/153 shipped rules armed** under the boards'
own rulesets (the paying folds are deliberately stripped so the
handler line is the only path — ``silu_fold``/``softsign_fold``/
``mul_square`` LHSes do occur, 4/131 ``DEFAULT`` rules total, but
no base board carries them).  After the coverage arm:
**57 ops, 46 attrs, 38/153 rules armed**.  The entire
``fire``/``saturate`` apply-law payoff dimension and the
reduction/view/attr vocabulary were uncorpus'd.
:func:`coverage_builders` emits the gap-targeting
kinds ``meta_player.gen_cases`` appends to its builder table;
:func:`gen_cases_coverage` is the standalone seeded stream.

Every emitted case is a real arena case — ``(name, term, rules,
decls, supported)`` probe arity — verified through
:class:`mlopt_discovery.meta_arena.MetaArena`'s own observe/legal/
referee path (:func:`probe_case`), never padding.

Run::

    .venv/bin/python -m mlopt_discovery.corpus_cov
    .venv/bin/python -m mlopt_discovery.corpus_cov --cases 400
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable, Iterable
from typing import Any

from mlopt_core.egraph.terms import _term_match
from mlopt_core.ir import Const, Op, TensorType, Var
from mlopt_core.laws import DEFAULT
from mlopt_core.laws.scan import SCAN_DIAG_LAWS, SCAN_LAWS
from mlopt_core.opmeta import (
    ACTIVATION_OPS,
    ATTR_SCHEMA,
    POINTWISE_OPS,
    VIEWISH_OPS,
)

from . import lawdata
from . import meta_arena as ma
from .genkernel import ELEMENTWISE, reduce_heads

__all__ = [
    "coverage_builders",
    "coverage_gaps",
    "gen_cases_coverage",
    "probe_case",
    "vocabulary",
]


# ---------------------------------------------------------------------------
#  The expressible vocabulary
# ---------------------------------------------------------------------------


_SCAN_RULES = None


def scan_rules() -> Any:
    """Lazily build the DEFAULT+scan ruleset (153 laws)."""
    global _SCAN_RULES
    if _SCAN_RULES is None:
        _SCAN_RULES = DEFAULT + SCAN_LAWS + SCAN_DIAG_LAWS
    return _SCAN_RULES


def _rule_ops(rule: Any) -> set:
    """Collect the op names on both sides of a law."""
    out: set = set()

    def walk(t: Any) -> None:
        if isinstance(t, Op):
            out.add(t.op)
            for a in t.args:
                walk(a)

    walk(rule.lhs)
    walk(rule.rhs)
    return out


def _spec_ops(spec: Any) -> set:
    """Op names inside a handler spec pattern (``str`` leaves are metas)."""
    out: set = set()
    if isinstance(spec, (tuple, list)) and spec:
        out.add(str(spec[0]))
        for e in spec[1:]:
            if not isinstance(e, dict):
                out |= _spec_ops(e)
    return out


def vocabulary(rules: Any = None) -> dict:
    """Return the expressible space the arena machinery names.

    *rules* defaults to ``DEFAULT``; the scan tier's laws and ops are
    always included (they are expressible — a case opts in by passing
    the extended ruleset, exactly as ``arena.py``'s scan probe does).
    The result is a plain dict of frozensets/tuples — stable to
    diff against any case stream.
    """
    base = DEFAULT if rules is None else rules
    all_rules = [*base, *SCAN_LAWS, *SCAN_DIAG_LAWS]
    law_ops: set = set()
    rule_names: list = []
    for r in all_rules:
        law_ops |= _rule_ops(r)
        rule_names.append(r.name)
    handler_ops: set = set()
    for h in lawdata.HANDLERS.values():
        handler_ops |= _spec_ops(h["pattern"])
        handler_ops.add(str(h["kernel"]))
    ops = (
        set(ELEMENTWISE)
        | set(reduce_heads())
        | set(VIEWISH_OPS)
        | set(POINTWISE_OPS)
        | set(ACTIVATION_OPS)
        | law_ops
        | handler_ops
    )
    attrs = {
        (op, attr)
        for op, amap in ATTR_SCHEMA.items()
        if op in ops
        for attr in amap.values()
    }
    return {
        "ops": frozenset(ops),
        "attrs": frozenset(attrs),
        "handlers": tuple(sorted(lawdata.HANDLERS)),
        "rules": tuple(rule_names),
    }


# ---------------------------------------------------------------------------
#  The diff — what a case stream never touches
# ---------------------------------------------------------------------------


def _subterms(term: Any) -> Iterable[Any]:
    """Yield *term* and every ``Op`` descendant (identity-deduped)."""
    seen: set[int] = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        yield t
        if isinstance(t, Op):
            stack.extend(t.args)


def _pattern_occurs(pattern: Any, term: Any) -> bool:
    """Check a handler spec pattern matches some program subterm.

    The same concrete-occurrence test ``meta_arena._claim_tags``
    applies (metavar leaves bind consistently, op heads and attrs
    literally).
    """

    def walk(t: Any) -> bool:
        if ma._concrete_match(pattern, t, {}):
            return True
        if isinstance(t, Op):
            return any(walk(a) for a in t.args)
        return False

    return walk(term)


def _ruleset_names(rules: Any) -> set:
    """Return the live rule names a case's ruleset carries."""
    if rules is None:
        return {r.name for r in DEFAULT}
    return {r.name for r in rules}


def coverage_gaps(
    cases: Iterable,
    *,
    vocab: dict | None = None,
) -> dict:
    """Diff *cases* against :func:`vocabulary` — the gap report.

    *cases* are meta-arena probe tuples ``(name, term, rules, …)``.
    Ops and attrs are counted over every subterm (the census's own
    convention — a board "contains" an op at any depth).  A rule is
    **armed** when its LHS matches a subterm of a case whose ruleset
    carries it — the ``fire``/``saturate`` payoff precondition; a
    handler tag is **covered** when its concrete pattern occurs
    (the ``claim`` precondition).  The report is data — sort the
    ``missing`` lists to see what the next corpus round owes.
    """
    vocab = vocabulary() if vocab is None else vocab
    covered_ops: set = set()
    covered_attrs: set = set()
    handler_hit = {tag: False for tag in vocab["handlers"]}
    armed: set = set()
    lhs_by_name = {
        r.name: r.lhs for r in (*DEFAULT, *SCAN_LAWS, *SCAN_DIAG_LAWS)
    }
    n = 0
    for case in cases:
        n += 1
        term = case[1]
        names = _ruleset_names(case[2] if len(case) > 2 else None)
        for tag, h in lawdata.HANDLERS.items():
            if not handler_hit[tag] and _pattern_occurs(
                h["pattern"], term
            ):
                handler_hit[tag] = True
        for s in _subterms(term):
            if not isinstance(s, Op):
                continue
            covered_ops.add(s.op)
            covered_attrs.update((s.op, k) for k in s.attrs)
        # rule arming: LHS match per live rule, short-circuit per name
        for rname in names:
            lhs = lhs_by_name.get(rname)
            if lhs is None or rname in armed:
                continue
            for s in _subterms(term):
                if (
                    isinstance(s, Op)
                    and _term_match(lhs, s) is not None
                ):
                    armed.add(rname)
                    break
    missing_ops = sorted(vocab["ops"] - covered_ops)
    missing_attrs = sorted(vocab["attrs"] - covered_attrs)
    missing_handlers = sorted(
        tag for tag, hit in handler_hit.items() if not hit
    )
    unarmed = sorted(set(vocab["rules"]) - armed)
    return {
        "n_cases": n,
        "ops": {
            "covered": sorted(covered_ops & vocab["ops"]),
            "missing": missing_ops,
            "n_missing": len(missing_ops),
        },
        "attrs": {
            "covered": sorted(covered_attrs & vocab["attrs"]),
            "missing": missing_attrs,
            "n_missing": len(missing_attrs),
        },
        "handlers": {
            "covered": sorted(t for t, h in handler_hit.items() if h),
            "missing": missing_handlers,
        },
        "rules": {
            "armed": sorted(armed),
            "unarmed": unarmed,
            "n_vocab": len(vocab["rules"]),
        },
    }


# ---------------------------------------------------------------------------
#  Verification — every emitted case through the real arena path
# ---------------------------------------------------------------------------


def probe_case(
    case: list | tuple,
    *,
    budget: int = 24,
    max_nodes: int = 20_000,
) -> dict:
    """Run one case tuple through ``MetaArena``'s real path.

    Builds the board exactly as ``meta_player._board`` does
    (``count_cost`` pricing, the case's ruleset and ``supported``
    bound), observes the enumerated move set, and plays the
    ``saturate → extract`` playbook — returning the coverage
    counters (legal moves, mined specs, occurring sites, claimable
    tags) and the scripted reward.  A case that raises here is not
    a real case.
    """
    from mlopt_core.cost.basic import count_cost

    rules = case[2] if len(case) > 2 else None
    sup = case[4] if len(case) > 4 else None
    arena = ma.MetaArena(
        case[1],
        rules,
        cost_fn=count_cost,
        supported=sup,
        max_nodes=max_nodes,
    )
    st = arena.observe()
    traj = ma.run_episode(
        arena,
        ma.ScriptedPlayer([ma.Action.saturate(), ma.Action.extract()]),
        budget,
    )
    return {
        "name": case[0],
        "legal": len(ma.legal_actions(st)),
        "specs": len(st.specs),
        "sites": len(st.sites),
        "claimable": st.claimable,
        "sat_reward": round(traj.total, 4),
        "cert": (
            traj.terminal.certificate_ok if traj.terminal else None
        ),
    }


# ---------------------------------------------------------------------------
#  The gap-targeting builders — real cases for the measured holes
# ---------------------------------------------------------------------------
#
#  Boards are ``(term, rules, supported)`` triples in ``gen_cases``
#  arity; each builder takes the stream's ``random.Random`` and
#  returns one.  Kinds fall into four coverage classes:
#
#  * ``law_*`` — spelled forms that arm specific shipped laws, so
#    ``fire``/``saturate`` moves have real targets (the corpus's
#    biggest hole: 4/131 rules armed, zero paying apply-law lines);
#  * ``scan`` — the carrier ops (``apply``/``aff``/…) under the
#    scan-extended ruleset;
#  * ``ew_gap`` / ``reduce_gap`` / ``view_gap`` — the fusion and IR
#    vocabularies the base stream never spells;
#  * ``kernel_ops`` / ``mixed_gap`` — kernel names as board ops, and
#    paying silu sites carrying gap ops along.

_X = Var("x", TensorType((4, 4)))
_Y = Var("y", TensorType((4, 4)))
_Z = Var("z", TensorType((4, 4)))
_W = Var("w", TensorType((4, 4)))
_V = Var("v", TensorType((4, 4)))
_P = Var("p", TensorType((4, 4)))
_Q = Var("q", TensorType((4, 4)))
_A = Var("A", TensorType((4, 4)))
_B = Var("B", TensorType((4, 4)))
_H = Var("h", TensorType((4, 4)))
_F = Var("f", TensorType((4, 4)))
_G = Var("g", TensorType((4, 4)))

_ZERO = Const(0)
_ONE = Const(1)
_TWO = Const(2)
_EPS = Const(1e-5)

_SANS = DEFAULT - DEFAULT.named("silu_fold", "softsign_fold")
_NOSQ = _SANS - DEFAULT.named("mul_square")
_SILU_SITE = Op.make("mul", _X, Op.make("sigmoid", _X))


def _sup(term: Any, rules: Any) -> frozenset:
    """Feasibility bound for a gap board: ambient + the swiglu kernel.

    ``supported=None`` would already admit every op the term spells
    (``_supported_or``'s ambient vocabulary); spelled out here so the
    ``swiglu`` kernel claim stays priceable on boards whose swiglu
    site is a pattern, not an op.  Passing the narrow
    ``_SWIGLU_SUP`` bound instead would price every gap op
    infeasible — the board could never pay.
    """
    return ma._supported_or(None, term, rules) | {"swiglu"}


_SCAN = scan_rules()


def _add(*args: Any) -> Any:
    """Return left-nested ``add`` over *args*."""
    out = args[0]
    for a in args[1:]:
        out = Op.make("add", out, a)
    return out


def _pick(rng: random.Random, *vals: Any) -> Any:
    return vals[rng.randrange(len(vals))]


# -- law_* : spelled forms arming the shipped apply-laws ------------


def _law_silu(rng: random.Random) -> tuple:
    """Spelled ``mul(x,sigmoid(x))`` under DEFAULT — ``silu_fold`` live.

    The fold law and the ``claim(silu)`` line are *both* offered:
    the oracle prices the claim macro above the fire, so the board
    teaches law-vs-claim selectivity the stripped-``sans`` corpus
    never produced.
    """
    inner = _pick(rng, _X, _Y, Op.make("tanh", _Z))
    site = Op.make("mul", inner, Op.make("sigmoid", inner))
    return (_add(site, _pick(rng, _Y, _Z, _W)), DEFAULT, None)


def _law_softsign(rng: random.Random) -> tuple:
    """Spelled ``div(x,|x|+1)`` under DEFAULT — ``softsign_fold`` pays."""
    t = _pick(rng, _X, _add(_Y, _Z))
    site = Op.make("div", t, Op.make("add", Op.make("abs", t), _ONE))
    return (_add(site, _pick(rng, _Y, _Z)), DEFAULT, None)


def _law_square(rng: random.Random) -> tuple:
    """Arm ``mul_square``/``pow_to_square`` — op-count neutral folds."""
    t = _pick(rng, _add(_X, _Y), Op.make("tanh", _Z), _W)
    site = (
        Op.make("mul", t, t)
        if rng.random() < 0.5
        else Op.make("pow", t, _TWO)
    )
    return (_add(site, _pick(rng, _Z, _V)), DEFAULT, None)


def _law_softmax(rng: random.Random) -> tuple:
    """Spelled ``div(exp, sum(exp))`` under DEFAULT — ``softmax_fold``."""
    t = _pick(rng, _add(_X, _Y), Op.make("tanh", _Z), _W)
    dim = _pick(rng, -1, 0, 1)
    keep = rng.random() < 0.7
    site = Op.make(
        "div",
        Op.make("exp", t),
        Op.make("sum", Op.make("exp", t), dim=dim, keepdim=keep),
    )
    return (_add(site, _pick(rng, _Z, _V)), DEFAULT, None)


def _law_glu(rng: random.Random) -> tuple:
    """Spelled ``mul(chunk0, sigmoid(chunk1))`` — ``glu_fold``."""
    t = _pick(rng, _add(_X, _Y), Op.make("tanh", _Z), _W)
    dim = _pick(rng, 0, 1)
    site = Op.make(
        "mul",
        Op.make("chunk", t, chunks=2, dim=dim, index=0),
        Op.make(
            "sigmoid",
            Op.make("chunk", t, chunks=2, dim=dim, index=1),
        ),
    )
    return (_add(site, _pick(rng, _Z, _V)), DEFAULT, None)


def _law_rsqrt(rng: random.Random) -> tuple:
    """``div(1,sqrt(t))`` / ``reciprocal(sqrt(t))`` — the rsqrt folds."""
    t = _pick(rng, _X, _add(_Y, _Z), Op.make("abs", _W))
    site = (
        Op.make("div", _ONE, Op.make("sqrt", t))
        if rng.random() < 0.6
        else Op.make("reciprocal", Op.make("sqrt", t))
    )
    return (_add(site, _pick(rng, _Y, _Z)), DEFAULT, None)


def _law_mm(rng: random.Random) -> tuple:
    """Matmul/linear nests — the assoc/distribute/naturality laws.

    Both factor and distribute directions occur across the stream
    (one pays, the other is the honest inverse), plus the scalar
    naturality and weight-sharing families.
    """
    forms = (
        lambda: Op.make("linear", Op.make("linear", _X, _A), _B),
        lambda: Op.make(
            "linear",
            Op.make("linear", _X, _A, _Y),
            _B,
            _Z,
        ),
        lambda: Op.make(
            "add",
            Op.make("matmul", _W, _X),
            Op.make("matmul", _W, _Y),
        ),
        lambda: Op.make("matmul", _W, Op.make("add", _X, _Y)),
        lambda: Op.make(
            "matmul",
            Op.make("matmul", _A, _B),
            _G,
        ),
        lambda: Op.make("matmul", _W, Op.make("mul", _X, _TWO)),
        lambda: Op.make(
            "add",
            Op.make("linear", _X, _A),
            Op.make("linear", _X, _B),
        ),
        lambda: Op.make("linear", _X, Op.make("add", _A, _B)),
    )
    return (
        _add(forms[rng.randrange(len(forms))](), _pick(rng, _Z, _V)),
        DEFAULT,
        None,
    )


def _law_split(rng: random.Random) -> tuple:
    """Emit attr-bearing index/view ops — ``split_scalar_*`` et al."""
    t = _pick(rng, _X, _add(_Y, _Z), _W)
    c = _pick(rng, _ONE, _TWO, Const(3))
    forms = (
        lambda: Op.make(
            "split",
            Op.make("add", t, c),
            sizes=[2, 2],
            dim=_pick(rng, 0, 1),
            index=_pick(rng, 0, 1),
        ),
        lambda: Op.make(
            "mul",
            Op.make("select", t, dim=0, index=1),
            Op.make("select", t, dim=0, index=1),
        ),
        lambda: Op.make(
            "add",
            Op.make("select", t, dim=_pick(rng, 0, 1), index=1),
            Op.make("slice", t, dim=0, start=0, end=2, step=1),
        ),
        lambda: Op.make(
            "chunk",
            Op.make("add", t, c),
            chunks=2,
            dim=_pick(rng, 0, 1),
            index=0,
        ),
    )
    return (
        _add(forms[rng.randrange(len(forms))](), _pick(rng, _Z, _V)),
        DEFAULT,
        None,
    )


def _law_unsq(rng: random.Random) -> tuple:
    """Emit ``unsqueeze`` pad/naturality forms (the intake-law family)."""
    t = _pick(rng, _X, _add(_Y, _Z), Op.make("tanh", _W))
    o = _pick(rng, _Y, _Z, _V)
    d = _pick(rng, 0, 1, 2)
    forms = (
        lambda: Op.make("mul", Op.make("unsqueeze", t, dim=d), o),
        lambda: Op.make("add", t, Op.make("unsqueeze", o, dim=d)),
        lambda: Op.make(
            "mul",
            Op.make("unsqueeze", t, dim=d),
            Op.make("unsqueeze", o, dim=d),
        ),
        lambda: Op.make("sub", Op.make("unsqueeze", t, dim=d), o),
    )
    return (
        _add(forms[rng.randrange(len(forms))](), _pick(rng, _P, _Q)),
        DEFAULT,
        None,
    )


def _law_idem(rng: random.Random) -> tuple:
    """Emit identity/annihilator forms — the biggest paying folds."""
    t = _pick(rng, _X, _add(_Y, _Z), Op.make("tanh", _W))
    forms = (
        lambda: Op.make("add", t, _ZERO),
        lambda: Op.make("add", _ZERO, t),
        lambda: Op.make("mul", t, _ONE),
        lambda: Op.make("neg", Op.make("neg", t)),
        lambda: Op.make("sub", _X, _Y),
    )
    return (
        _add(forms[rng.randrange(len(forms))](), _pick(rng, _Z, _V)),
        DEFAULT,
        None,
    )


# -- scan : the carrier vocabulary under the scan-extended ruleset --


def _scan(rng: random.Random) -> tuple:
    """``apply``/``aff`` carriers + spelled step forms.

    ``aff_lift`` (``add(matmul(A,h),x) → apply(aff(A,x),h)``) arms
    only when the scan laws sit in the ruleset — the case passes
    ``scan_rules()`` exactly as ``arena.py``'s scan probe does.
    """
    forms = (
        lambda: Op.make("add", Op.make("matmul", _A, _H), _X),
        lambda: Op.make("apply", Op.make("aff", _A, _X), _H),
        lambda: Op.make("add", Op.make("apply", _F, _H), _X),
        lambda: Op.make("applyd", _A, _H),
    )
    return (
        _add(forms[rng.randrange(len(forms))](), _pick(rng, _Y, _Z)),
        _SCAN,
        None,
    )


# -- ew_gap : the ELEMENTWISE fusion alphabet -----------------------


def _ew_apply(op: str, t: Any, o: Any) -> Any:
    """Apply one ELEMENTWISE op to *t* (clamp/pow attr forms)."""
    if op == "pow":
        return Op.make("pow", t, _TWO)
    if op == "clamp":
        return Op.make("clamp", t, min=-1.0, max=1.0)
    if op == "clamp_min":
        return Op.make("clamp_min", t, min=0.0)
    if op == "clamp_max":
        return Op.make("clamp_max", t, max=1.0)
    if op == "sub":
        return Op.make("sub", t, o)
    return Op.make(op, t)


#: ELEMENTWISE ops the base corpus already spells — everything else
#: in the fusion alphabet is a coverage target.  Derived, not listed:
#: a new ELEMENTWISE member joins the gap set automatically.
_EW_BASE_OPS = frozenset(
    {"add", "mul", "div", "sigmoid", "tanh", "abs"}
)
_EW_GAP_OPS = tuple(sorted(ELEMENTWISE - _EW_BASE_OPS - {"matmul"}))


def _ew_gap(rng: random.Random) -> tuple:
    """One uncovered ELEMENTWISE op on a real subterm.

    Half the draws ride beside a paying spelled-silu site under
    ``nosq`` (the claim line stays live — coverage *with* signal);
    half pair two gap ops as an honest decline board.
    """
    op = _EW_GAP_OPS[rng.randrange(len(_EW_GAP_OPS))]
    t = _pick(rng, _X, _add(_Y, _Z), Op.make("tanh", _W))
    o = _pick(rng, _Y, _Z, _V)
    piece = _ew_apply(op, t, o)
    if rng.random() < 0.5:
        term = _add(_SILU_SITE, piece)
        return (term, _NOSQ, _sup(term, _NOSQ))
    op2 = _EW_GAP_OPS[rng.randrange(len(_EW_GAP_OPS))]
    term = _add(piece, _ew_apply(op2, o, _pick(rng, _X, _Z)))
    return (term, _SANS, None)


# -- reduce_gap : the reduction-headed vocabulary --------------------


def _reduce_gap(rng: random.Random) -> tuple:
    """Emit a ``reduce_heads()`` op over an elementwise body.

    Reduction-bodied specs are exactly what the genkernel reduction
    tier claims; the arena's ``declare`` inventory and ``fire``
    moves see them for the first time here.  ``softmax`` takes
    ``dim`` only; the rest take ``dim``/``keepdim``.
    """
    op = tuple(sorted(reduce_heads()))[
        rng.randrange(len(reduce_heads()))
    ]
    body = _pick(
        rng,
        Op.make("mul", _X, Op.make("relu", _X)),
        _add(_X, _Y),
        Op.make("mul", _X, Op.make("sigmoid", _X)),
    )
    dim = _pick(rng, -1, 0, 1)
    if op == "softmax":
        red = Op.make("softmax", body, dim=dim)
    else:
        red = Op.make(
            "sum" if op == "sum" else op,
            body,
            dim=dim,
            keepdim=rng.random() < 0.6,
        )
    term = _add(red, _pick(rng, _Y, _Z))
    return (term, _SANS, None)


# -- view_gap : the attr-bearing view/index vocabulary ---------------


def _view_gap(rng: random.Random) -> tuple:
    """One attr-bearing view/index op — the ``ATTR_SCHEMA`` surface.

    The base corpus carries **zero** attrs; every ``dim``/``index``/
    ``shape``/``sizes`` site the IR declares is a gap.  Templates
    cover the VIEWISH class the laws and handlers pattern-match on.
    """
    t = _pick(rng, _X, _add(_Y, _Z), _W)
    o = _pick(rng, _Y, _Z, _V)
    forms = (
        lambda: Op.make("transpose", t, dim0=0, dim1=1),
        lambda: Op.make("permute", t, dims=(1, 0)),
        lambda: Op.make(
            "reshape", t, shape=_pick(rng, (16,), (2, 8), (4, 4))
        ),
        lambda: Op.make(
            "select", t, dim=_pick(rng, 0, 1), index=_pick(rng, 0, 2)
        ),
        lambda: Op.make(
            "slice",
            t,
            dim=0,
            start=0,
            end=_pick(rng, 2, 4),
            step=1,
        ),
        lambda: Op.make(
            "split",
            Op.make("add", t, _ONE),
            sizes=[2, 2],
            dim=_pick(rng, 0, 1),
            index=_pick(rng, 0, 1),
        ),
        lambda: Op.make(
            "chunk",
            Op.make("add", t, _ONE),
            chunks=2,
            dim=_pick(rng, 0, 1),
            index=_pick(rng, 0, 1),
        ),
        lambda: Op.make("unsqueeze", t, dim=_pick(rng, 0, 1, 2)),
        lambda: Op.make("squeeze", t, dim=_pick(rng, 0, 1)),
        lambda: Op.make("expand", t, shape=_pick(rng, (8, 4), (4, 8))),
        lambda: Op.make(
            "broadcast_to", t, shape=_pick(rng, (8, 4), (4, 8))
        ),
        lambda: Op.make(
            "flatten",
            t,
            start_dim=_pick(rng, 0, 1),
            end_dim=_pick(
                rng,
                1,
            ),
        ),
        lambda: Op.make("getitem", t, index=_pick(rng, 0, 1)),
        lambda: Op.make("unbind", t, dim=_pick(rng, 0, 1)),
        lambda: Op.make(
            "movedim",
            t,
            source=_pick(rng, 0, 1),
            destination=_pick(rng, 0, 1),
        ),
        lambda: Op.make(
            "narrow",
            t,
            dim=0,
            start=0,
            length=_pick(rng, 2, 4),
        ),
        lambda: Op.make("concat", t, o, dim=_pick(rng, 0, 1)),
        lambda: Op.make("stack", t, o, dim=_pick(rng, 0, 1)),
        lambda: Op.make("view", t, shape=(16,)),
    )
    piece = forms[rng.randrange(len(forms))]()
    if rng.random() < 0.5:
        term = _add(_SILU_SITE, piece)
        return (term, _NOSQ, _sup(term, _NOSQ))
    return (_add(piece, _pick(rng, _Z, _V)), _SANS, None)


# -- kernel_ops : kernel names as board ops ---------------------------


def _kernel_ops(rng: random.Random) -> tuple:
    """Emit kernel vocabulary occurring *as ops* — ``silu`` et al.

    The handlers' kernels and the laws' RHS heads: under DEFAULT the
    ``*_expand``/``*_fold`` pairs arm in *both* directions, which no
    base board ever offered.
    """
    t = _pick(rng, _X, _add(_Y, _Z), Op.make("tanh", _W))
    o = _pick(rng, _Y, _Z, _V)
    forms = (
        lambda: Op.make("silu", t),
        lambda: Op.make("square", t),
        lambda: Op.make("softsign", t),
        lambda: Op.make("softsign_gate", t),
        lambda: Op.make("glu", t, dim=_pick(rng, 0, 1)),
        lambda: Op.make("rms_norm", t, _G, dim=0, eps=1e-5),
        lambda: Op.make("rms_norm", t, dim=0, eps=1e-5),
        lambda: Op.make("softmax", t, dim=_pick(rng, -1, 0)),
        lambda: Op.make("log_softmax", t, dim=_pick(rng, -1, 0)),
        lambda: Op.make("swiglu", t, o),
    )
    return (
        _add(forms[rng.randrange(len(forms))](), _pick(rng, _Z, _V)),
        DEFAULT,
        None,
    )


# -- mixed_gap : paying sites carrying gap ops ------------------------


def _mixed_gap(rng: random.Random) -> tuple:
    """Emit a paying silu/swiglu site *plus* a random gap piece.

    The claim line stays the oracle's pick while the gap op/attr
    rides along in the term — coverage that never costs the board
    its signal (the ``square_mix`` pattern generalized).
    """
    t = _pick(rng, _Y, _add(_Z, _W), Op.make("tanh", _V))
    piece = (
        _ew_apply(_EW_GAP_OPS[rng.randrange(len(_EW_GAP_OPS))], t, _X)
        if rng.random() < 0.6
        else Op.make(
            "softmax",
            Op.make("mul", t, Op.make("relu", t)),
            dim=_pick(rng, -1, 0),
        )
    )
    site = (
        _SILU_SITE
        if rng.random() < 0.6
        else Op.make("mul", _SILU_SITE, _Z)
    )
    term = _add(site, piece)
    return (term, _NOSQ, _sup(term, _NOSQ))


_BUILDER_TABLE: tuple[
    tuple[str, Callable[[random.Random], tuple]], ...
] = (
    ("law_silu", _law_silu),
    ("law_softsign", _law_softsign),
    ("law_square", _law_square),
    ("law_softmax", _law_softmax),
    ("law_glu", _law_glu),
    ("law_rsqrt", _law_rsqrt),
    ("law_mm", _law_mm),
    ("law_split", _law_split),
    ("law_unsq", _law_unsq),
    ("law_idem", _law_idem),
    ("scan", _scan),
    ("ew_gap", _ew_gap),
    ("reduce_gap", _reduce_gap),
    ("view_gap", _view_gap),
    ("kernel_ops", _kernel_ops),
    ("mixed_gap", _mixed_gap),
)


def coverage_builders() -> list:
    """Return ``(kind, build)`` pairs — ``build(rng) -> (term, rules, sup)``.

    The gap-targeting half of ``meta_player.gen_cases``'s builder
    table.  Kinds are data: a new ``ELEMENTWISE`` op, ``HANDLERS``
    tag or shipped law widens :func:`vocabulary` and is reported by
    :func:`coverage_gaps`; the ``ew_gap``/``reduce_gap`` cycles pick
    new members up automatically, while genuinely new *classes*
    (a new attr schema shape, a new carrier tier) need a new
    template here — the honest extension seam.
    """
    return list(_BUILDER_TABLE)


def gen_cases_coverage(seed: int, n: int) -> list:
    """Seeded coverage-case stream — ``gen_cases`` arity.

    Same probe tuple shape ``meta_player.gen_cases`` emits:
    ``("cov{i}:{kind}", term, rules, (), supported)``.  Deterministic
    per *seed*; disjoint-name ``cov`` prefix keeps the streams
    distinguishable in ledgers.
    """
    rng = random.Random(seed)
    builders = coverage_builders()
    out = []
    for i in range(n):
        kind, build = builders[rng.randrange(len(builders))]
        term, rules, sup = build(rng)
        out.append((f"cov{i}:{kind}", term, rules, (), sup))
    return out


# ---------------------------------------------------------------------------
#  CLI — the gap report over a generated stream
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Print the coverage-gap report for a generated case stream."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cases",
        type=int,
        default=400,
        help="stream size to census (default 400)",
    )
    ap.add_argument("--seed", type=int, default=0, help="stream seed")
    ap.add_argument(
        "--stream",
        choices=("base", "coverage", "all"),
        default="all",
        help="which generator to census",
    )
    args = ap.parse_args(argv)

    cases: list = []
    if args.stream in ("base", "all"):
        from . import meta_player as mp

        cases += mp.gen_cases(args.seed, args.cases, coverage=False)
    if args.stream in ("coverage", "all"):
        cases += gen_cases_coverage(args.seed, args.cases)

    report = coverage_gaps(cases)
    print(  # stdout-compat
        f"== corpus coverage — {report['n_cases']} cases =="
    )
    print(  # stdout-compat
        f"ops:      {len(report['ops']['covered'])} covered, "
        f"{report['ops']['n_missing']} missing"
    )
    print(  # stdout-compat
        "  missing ops:",
        ", ".join(report["ops"]["missing"]) or "(none)",
    )
    print(  # stdout-compat
        f"attrs:    {len(report['attrs']['covered'])} covered, "
        f"{report['attrs']['n_missing']} missing"
    )
    print(  # stdout-compat
        "  missing attrs:",
        ", ".join(
            f"{o}.{a}" for o, a in report["attrs"]["missing"][:40]
        )
        or "(none)",
    )
    print(  # stdout-compat
        f"handlers: covered={report['handlers']['covered']} "
        f"missing={report['handlers']['missing']}"
    )
    print(  # stdout-compat
        f"rules:    {len(report['rules']['armed'])}/"
        f"{report['rules']['n_vocab']} armed"
    )
    print(  # stdout-compat
        "  unarmed:",
        ", ".join(report["rules"]["unarmed"][:40]) or "(none)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

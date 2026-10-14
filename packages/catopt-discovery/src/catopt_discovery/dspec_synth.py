"""Synthesize declarative ``dspec`` derives from measured bindings.

The derive-as-data migration (ADR 0004) moved every hook the grammar
could express onto ``dspec`` data; the residual is procedural because
its expression needs an op the bank lacks.  This module closes that
loop the same way ``auto_cond`` closes the guard loop: measure the
hook's outputs over a binding domain, enumerate candidate ``dspec``
expressions over the bank, keep the ones that reproduce *every*
measured value exactly, and mint the smallest.

A refusal is a finding, not a failure: when no expression matches,
the missing bank op is reported (``_near_misses`` names the closest
candidates and their first divergence), which is the honest signal
that the DSL needs extending — the same shape as ``auto_cond``'s
"no declarable conjunction of ≤ N covers …" decline.

The bank the synthesizer searches is ``_DOPS``' arithmetic plus the
``("len", ("shape", T))`` / ``("len", ("vshape", T))`` rank readers
(the cost-convention and carrier-aware shape sources) and the
``max``/``min`` pair the residual survey measured as missing (the
``om``/``xcarrier`` derives are rank arithmetic over bound
metavariables).  Enumeration propagates
each candidate's *value vector* over the samples and collapses a level
to one representative per vector (``_dedupe``) — the blind product is
quartic in the leaf count, so without the collapse a rank bank's
level-2 product explodes.
"""

from __future__ import annotations

from itertools import product
from typing import Any

from catopt_core.laws.cond import _DBINOPS, _DOPS, _DUNOPS

__all__ = [
    "samples_from_hook",
    "synth_dspec",
]

#: Arity-2 derive ops the synthesizer composes (the arithmetic bank
#: plus the two int-tuple readers — everything else in ``_DOPS`` is
#: either a leaf form or shape-specialised).  ``max``/``min`` are the
#: rank-selection pair a carrier-aware derive spells as
#: ``max(rank a, rank b) - 1``.
BINOPS: tuple[str, ...] = (
    "add",
    "sub",
    "mul",
    "floordiv",
    "max",
    "min",
)
UNOPS: tuple[str, ...] = ("neg", "int", "float")


def _eval(expr: Any, bound: dict) -> Any:
    """Evaluate one expr — ``None`` when it declines."""
    try:
        return (
            _DOPS[expr[0]](expr[1:], bound)
            if isinstance(expr, tuple)
            else None
        )
    except Exception:
        return None


def _leaves(bound: dict, values: set) -> list[Any]:
    """Leaf exprs: attr refs, rank readers, observed constants.

    Attr refs read ``$attr:`` bindings; each bound *term* contributes a
    rank reader ``("len", ("shape", T))`` — the carrier-aware derives
    (``om``/``trace``/``xcarrier``) are rank arithmetic over bound
    metavariables, so the reader is seeded as a leaf (reached at depth
    one rather than composing ``len`` over ``shape`` at depth two, past
    the bank's ``max_depth``).  A term whose shape is unknown declines
    the reader, exactly as the procedural hook vetoed.  Observed int
    values seed literals, so the ``- 1`` / ``- 2`` offsets a rank
    derive spells are expressible.
    """
    out: list[Any] = []
    for key in sorted(bound):
        if key.startswith("$attr:"):
            out.append(("attr", key[len("$attr:") :]))
    for key in sorted(bound):
        if key.startswith("$attr:"):
            continue
        # Two rank readers: the cost-convention ``shape`` and the
        # carrier-aware ``vshape`` (the derive DSL's value-shape source,
        # resolved through the registered resolver — ``_shape_of`` when
        # none is registered, so the two dedupe to one for a plain
        # tensor term).  A carrier member whose true value rank differs
        # from its convention rank is reachable only through ``vshape``.
        out.append(("len", ("shape", key)))
        out.append(("len", ("vshape", key)))
    for v in sorted(values, key=repr):
        if isinstance(v, int) and not isinstance(v, bool):
            out.append(("lit", v))
    return out


def _signature(expr: Any, samples: list[tuple]) -> tuple:
    """Return *expr*'s value over every sample — its fit fingerprint."""
    return tuple(_eval(expr, b) for b, _e in samples)


def _vbin(op: str, a: Any, b: Any) -> Any:
    """Apply an arity-2 bank op to two *values* — ``None`` on decline.

    The value-level twin of ``_DOPS[op]``: enumeration folds operands'
    value vectors instead of re-walking the tree (measured: the tree
    walk made a rank bank's level-2 product cost minutes per rule).
    ``None`` is the shared decline/value sentinel — an operand that
    declines vetoes the op exactly as ``_dexpr``'s ``_Decline`` did.
    """
    if op in ("max", "min"):
        if not (isinstance(a, int) and isinstance(b, int)):
            return None
        return (max if op == "max" else min)(a, b)
    if a is None or b is None:
        return None
    try:
        return _DBINOPS[op](a, b)
    except (ArithmeticError, TypeError, ValueError):
        return None


def _vun(op: str, a: Any) -> Any:
    """Apply an arity-1 bank op to a *value* — ``None`` on decline."""
    if a is None:
        return None
    try:
        return _DUNOPS[op](a)
    except (ArithmeticError, TypeError, ValueError):
        return None


def _vlen(a: Any) -> Any:
    """``len`` of a value — ``None`` when it is not sized."""
    return len(a) if isinstance(a, (tuple, list, str)) else None


def _dedupe(pairs: list[tuple]) -> list[tuple]:
    """Keep one ``(expr, sig)`` per fingerprint — smallest, then repr.

    Two expressions that agree on *every* sample are interchangeable
    for the fit (``add(a, b)`` vs ``add(b, a)``), so collapsing each
    level to one representative per fingerprint prunes the quartic
    blow-up of the blind product without losing a distinguishable
    candidate.
    """
    best: dict = {}
    for expr, sig in pairs:
        prev = best.get(sig)
        if prev is None or (_size(expr), repr(expr)) < (
            _size(prev[0]),
            repr(prev[0]),
        ):
            best[sig] = (expr, sig)
    return list(best.values())


def _candidates(
    bound: dict, values: set, *, max_depth: int, samples: list[tuple]
) -> list[tuple]:
    """Enumerate ``(expr, sig)`` trees up to *max_depth* over the bank.

    Each level is deduped by fingerprint (:func:`_dedupe`) so the
    quartic product stays tractable; the smallest representative is
    kept, so the final ``min`` by size still mints the smallest fit.
    """
    level = _dedupe(
        [(e, _signature(e, samples)) for e in _leaves(bound, values)]
    )
    out = list(level)
    for _ in range(max_depth):
        nxt: list[tuple] = []
        for op in BINOPS:
            for (ea, sa), (eb, sb) in product(level, level):
                nxt.append(
                    (
                        (op, ea, eb),
                        tuple(
                            _vbin(op, a, b)
                            for a, b in zip(sa, sb, strict=True)
                        ),
                    )
                )
        for op in UNOPS:
            for ea, sa in level:
                nxt.append(((op, ea), tuple(_vun(op, a) for a in sa)))
        for ea, sa in level:
            nxt.append((("len", ea), tuple(_vlen(a) for a in sa)))
        nxt = _dedupe(nxt)
        out.extend(nxt)
        level = nxt
    return out


def _size(expr: Any) -> int:
    if not isinstance(expr, tuple):
        return 1
    return 1 + sum(_size(a) for a in expr[1:])


def samples_from_hook(
    derive: Any, bounds: list[dict]
) -> list[tuple[dict, Any]]:
    """Measure a procedural *derive* over *bounds* — the sample set.

    Rows whose hook declines are dropped (they carry no value to
    reproduce); a hook that declines everywhere yields an empty set
    and the caller refuses honestly.
    """
    out: list[tuple[dict, Any]] = []
    for bound in bounds:
        got = derive(bound)
        if got is None:
            continue
        for value in got.values():
            out.append((bound, value))
    return out


def _reads_bound(expr: Any) -> bool:
    """Return True iff *expr* reads a bound value.

    A constant reproduces any sample set whose values agree; that is
    a degenerate fit, not a derivation (the hook would be replaced by
    a literal that ignores its inputs).  A synth must depend on at
    least one bound attribute.
    """
    if not isinstance(expr, tuple):
        return False
    if expr[0] in (
        "attr",
        "attr0",
        "shape",
        "vshape",
        "dim",
        "leaf-dim",
    ):
        return True
    return any(_reads_bound(a) for a in expr[1:])


#: Minimum measured samples before synthesis is attempted.  With
#: two points an expression can fit by coincidence (measured: the
#: ``max`` hook admitted ``float(a) + c // a`` on two samples), so
#: the fit is validated on a held-out half and refused below four.
MIN_SAMPLES: int = 4


def synth_dspec(
    samples: list[tuple],
    *,
    name: str | None = None,
    max_depth: int = 2,
) -> dict | None:
    """Return the smallest ``dspec`` reproducing every sample.

    ``samples`` is ``[(bound, expected_value)]`` — the measured
    outputs of the procedural hook.  Returns ``{name: expr}`` (the
    canonical dspec form) or ``None`` when no expression in the bank
    matches — a refusal is the honest signal that the bank needs an
    op, not that the hook must stay.

    Refusals are deliberate: a sample set whose values all agree
    carries no signal (any constant "matches"), and a fit is
    validated on the even/odd split — a candidate must reproduce
    *both* halves, so a coincidence on a thin set never ships
    (``MIN_SAMPLES`` is the floor; below it the synth refuses).
    """
    if len(samples) < MIN_SAMPLES:
        return None
    values = {v for _b, v in samples}
    if len(values) < 2:
        # every measured value agrees — no signal to fit; any
        # expr producing that constant "matches" and the synth
        # would be a coincidence, not a derivation.
        return None
    fit = range(0, len(samples), 2)
    holdout = range(1, len(samples), 2)
    bound = samples[0][0]
    cands = _candidates(
        bound, values, max_depth=max_depth, samples=samples
    )
    hits = [
        expr
        for expr, sig in cands
        if _reproduces(expr, sig, samples, fit, holdout)
    ]
    if not hits:
        return None
    best = min(hits, key=lambda c: (_size(c), repr(c)))
    return {(name or "OUT"): best}


def _reproduces(
    expr: Any,
    sig: tuple,
    samples: list[tuple],
    fit: range,
    holdout: range,
) -> bool:
    """Return True iff *expr*'s value vector fits the even/odd split.

    The read-guard plus the held-out validation, split out of
    :func:`synth_dspec` so its cyclomatic complexity stays under the
    ratchet ceiling.
    """
    if not _reads_bound(expr):
        return False
    return all(sig[i] == samples[i][1] for i in fit) and all(
        sig[i] == samples[i][1] for i in holdout
    )


def _near_misses(
    samples: list[tuple], *, max_depth: int = 2, k: int = 5
) -> list[tuple[Any, int]]:
    """Closest candidates by sample-agreement — the bank-gap probe."""
    if not samples:
        return []
    values = {v for _b, v in samples}
    cands = _candidates(
        samples[0][0], values, max_depth=max_depth, samples=samples
    )
    scored = [
        (
            expr,
            sum(
                1
                for v, (_b, e) in zip(sig, samples, strict=True)
                if v == e
            ),
        )
        for expr, sig in cands
    ]
    scored.sort(key=lambda p: (-p[1], _size(p[0]), repr(p[0])))
    return scored[:k]

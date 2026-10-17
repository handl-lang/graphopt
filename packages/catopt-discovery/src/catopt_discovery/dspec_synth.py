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
(the cost-convention and carrier-aware shape sources), the
``("dim", S, i)`` dim readers, the ``max``/``min`` pair the residual
survey measured as missing (the ``om``/``xcarrier`` derives are rank
arithmetic over bound metavariables), the ``mod`` (``%``) axis
normalisation a carrier-aware derive spells as ``dim % rank``, and
the real-root pair ``sqrt``/``rsqrt`` plus ``fdiv`` — the division a
measured refusal spelled literally (``scale = gamma / sqrt(var +
eps)``, the weight-space fold's missing piece) — and, when the
registered tensor algebra carries a ``source`` (ADR 0012), a
``("tval", T)`` leaf per bound term: the tensor-value read a
weight-space derive spells.  Fingerprints box every value through
:func:`_vkey`, so a tensor-valued sample set hashes through the
algebra's ``key`` (or refuses honestly by identity) instead of
raising ``TypeError: unhashable`` — the gate-4 crash the
weight-space probe measured.
Enumeration propagates
each candidate's *value vector* over the samples and collapses a level
to one representative per vector (``_dedupe``) — the blind product is
quartic in the leaf count, so without the collapse a rank bank's
level-2 product explodes.

A hook that returns *several* attrs is fitted per-attr
(:func:`synth_dspec_multi`): the ``dspec`` map is ``{NAME: expr}``, so
flattening the outputs into one expr (what :func:`samples_from_hook`
does) is not a spelling of the hook at all — two attrs with different
values on one bound can never be a single expr.
"""

from __future__ import annotations

from itertools import product
from typing import Any

from catopt_core.laws.cond import (
    _DBINOPS,
    _DOPS,
    _DUNOPS,
    tensor_algebra,
)
from catopt_core.typing import _shape_of

__all__ = [
    "samples_by_attr",
    "samples_from_hook",
    "synth_dspec",
    "synth_dspec_multi",
]

#: Arity-2 derive ops the synthesizer composes (the arithmetic bank
#: plus the two int-tuple readers — everything else in ``_DOPS`` is
#: either a leaf form or shape-specialised).  ``max``/``min`` are the
#: rank-selection pair a carrier-aware derive spells as
#: ``max(rank a, rank b) - 1``; ``mod`` is the ``%`` axis
#: normalisation a carrier-aware derive spells as ``dim % rank``;
#: ``fdiv`` is the true division the weight-space fold's measured
#: refusal spelled (``gamma / sqrt(var + eps)``) — the probe's raw
#: tier could not fit without it
#: (``project/retros/weightspace-fold.md``).
BINOPS: tuple[str, ...] = (
    "add",
    "sub",
    "mul",
    "fdiv",
    "floordiv",
    "mod",
    "max",
    "min",
)
#: Arity-1 ops.  ``sqrt``/``rsqrt`` are the real roots the same
#: refusal named — ``rsqrt(var + eps)`` is the normalisation
#: statistic's
#: other spelling (their ``_DUNOPS`` handlers decline on a negative /
#: non-positive operand, so a fit never mints a complex or infinite
#: value).
UNOPS: tuple[str, ...] = ("neg", "int", "float", "sqrt", "rsqrt")


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


def _vkey(v: Any) -> Any:
    """Hashable fingerprint of a sample value (ADR 0012).

    Signatures and sample-equality keys must hash: a scalar boxes as
    ``("s", v)`` (its own value — the numeric tower's ``8 == 8.0``
    collapse the raw-value fingerprints already had), a sequence boxes
    elementwise, and anything else — a tensor payload — boxes through
    the registered algebra's ``key`` (a content fingerprint) or, when
    no ``key`` is installed, ``id(v)``: two distinct objects then
    fingerprint unequal, so a tensor fit without a ``key`` fn refuses
    honestly rather than crashing on the unhashable — the
    ``TypeError: unhashable type: 'numpy.ndarray'`` the weight-space
    probe's gate 4 measured.
    """
    if v is None or isinstance(v, (bool, int, float, str)):
        return ("s", v)
    if isinstance(v, (tuple, list)):
        return ("t", tuple(_vkey(x) for x in v))
    key = tensor_algebra()["key"]
    return ("a", key(v) if key is not None else id(v))


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

    When the tensor algebra carries a ``source`` (ADR 0012) each bound
    term also contributes ``("tval", T)`` — the tensor-value read a
    weight-space derive spells; without a source the leaf would only
    decline on every sample, so it is not seeded.
    """
    out: list[Any] = []
    for key in sorted(bound):
        if key.startswith("$attr:"):
            out.append(("attr", key[len("$attr:") :]))
    has_tsrc = tensor_algebra()["source"] is not None
    for key in sorted(bound):
        if key.startswith("$attr:"):
            continue
        if has_tsrc:
            out.append(("tval", key))
        # Two rank readers: the cost-convention ``shape`` and the
        # carrier-aware ``vshape`` (the derive DSL's value-shape source,
        # resolved through the registered resolver — ``_shape_of`` when
        # none is registered, so the two dedupe to one for a plain
        # tensor term).  A carrier member whose true value rank differs
        # from its convention rank is reachable only through ``vshape``.
        out.append(("len", ("shape", key)))
        out.append(("len", ("vshape", key)))
        out.extend(_dim_leaves(key, bound[key]))
    for v in sorted(values, key=repr):
        if isinstance(v, int) and not isinstance(v, bool):
            out.append(("lit", v))
    return out


#: The axes the dim-reader bank seeds — the last axis (the
#: ``q[-1]`` / ``k1[-1]`` feature idiom a carrier-aware derive names
#: literally) and the first.  Deliberately *not* the whole
#: ``[-r, r)`` window: a *data-dependent* axis (the ``dim % rank``
#: cat axis ``_cat_axis_plan`` computes) is not expressible by a
#: literal-index reader anyway, so seeding every index only inflates
#: the level-2 product — measured on ``om_merge``: 32 dim leaves
#: turned it into 1.4 M candidates / 67 s against 13.6 k / 1.9 s for
#: the same fit.
_DIM_AXES: tuple[int, ...] = (-1, 0)


def _dim_leaves(key: str, term: Any) -> list[Any]:
    """``("dim", S, i)`` readers of a bound term's shape.

    The cost-convention dim read (``("dim", ("shape", T), i)``) and
    the carrier-aware twin (``("dim", ("vshape", T), i)``) seeded at
    the literal axes of :data:`_DIM_AXES` that the term's rank admits.
    An index past a *different* sample's rank declines at eval, so
    seeding the window is honest.
    """
    s = _shape_of(term)
    r = len(s) if isinstance(s, tuple) else 0
    out: list[Any] = []
    for i in _DIM_AXES:
        if not (-r <= i < r):
            continue
        out.append(("dim", ("shape", key), i))
        out.append(("dim", ("vshape", key), i))
    return out


def _signature(expr: Any, samples: list[tuple]) -> tuple:
    """Return *expr*'s value over every sample — its fit fingerprint."""
    return tuple(_eval(expr, b) for b, _e in samples)


def _box(sig: tuple) -> tuple:
    """Fingerprint a raw value vector — every element hashable.

    The signature itself stays raw (the level products evaluate
    children from it); boxing happens only where the vector is a dict
    key or an equality side — ``_dedupe``, ``_reproduces``,
    ``_near_misses``.
    """
    return tuple(_vkey(v) for v in sig)


def _vbin(op: str, a: Any, b: Any) -> Any:
    """Apply an arity-2 bank op to two *values* — ``None`` on decline.

    The value-level twin of ``_DOPS[op]``: enumeration folds operands'
    value vectors instead of re-walking the tree (measured: the tree
    walk made a rank bank's level-2 product cost minutes per rule).
    ``None`` is the shared decline/value sentinel — an operand that
    declines vetoes the op exactly as ``_dexpr``'s ``_Decline`` did.
    The int-only ops (``max``/``min``/``mod``) delegate to their
    strictness twins below.
    """
    if op in ("max", "min"):
        return _vsel(op, a, b)
    if a is None or b is None:
        return None
    if op == "mod":
        return _vmod(a, b)
    if isinstance(a, (bool, int, float)) and isinstance(
        b, (bool, int, float)
    ):
        try:
            return _DBINOPS[op](a, b)
        except (ArithmeticError, TypeError, ValueError):
            return None
    # Mirror ``cond._binop``'s tensor tier (ADR 0012): the pushed op
    # wins, the Python operator is the fallback; a pushed fn is
    # foreign code — any exception it raises is a decline.
    try:
        fn = tensor_algebra()["ops"].get(op) or _DBINOPS[op]
        return fn(a, b)
    except Exception:
        return None


def _vsel(op: str, a: Any, b: Any) -> Any:
    """``max``/``min`` over int operands — ``None`` on a non-int side.

    The value-level twin of ``_DOPS[op]``'s int-only selection: both
    operands must be ``int`` (``bool`` admitted, the handler's
    ``isinstance`` contract).
    """
    if not (isinstance(a, int) and isinstance(b, int)):
        return None
    return (max if op == "max" else min)(a, b)


def _vmod(a: Any, b: Any) -> Any:
    """Apply ``a % b`` over int operands — ``None`` on a decline.

    The value-level twin of ``_DOPS["mod"]``: both operands must be
    ``int`` (``bool`` admitted, matching the handler) and the divisor
    nonzero — exactly the handler's strictness.
    """
    if not (isinstance(a, int) and isinstance(b, int)) or b == 0:
        return None
    return a % b


def _vun(op: str, a: Any) -> Any:
    """Apply an arity-1 bank op to a *value* — ``None`` on decline."""
    if a is None:
        return None
    if isinstance(a, (bool, int, float)):
        try:
            return _DUNOPS[op](a)
        except (ArithmeticError, TypeError, ValueError):
            return None
    # Mirror ``cond._unop``'s tensor tier (ADR 0012): the pushed op
    # wins, the Python op is the fallback — a pushed fn's exceptions
    # are declines (foreign code).
    try:
        fn = tensor_algebra()["ops"].get(op) or _DUNOPS[op]
        return fn(a)
    except Exception:
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
        fp = _box(sig)
        prev = best.get(fp)
        if prev is None or (_size(expr), repr(expr)) < (
            _size(prev[0]),
            repr(prev[0]),
        ):
            best[fp] = (expr, sig)
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
    and the caller refuses honestly.  A hook that returns *several*
    attrs is flattened here — the single-expr fit's view; the per-attr
    fit reads :func:`samples_by_attr`.
    """
    out: list[tuple[dict, Any]] = []
    for bound in bounds:
        got = derive(bound)
        if got is None:
            continue
        for value in got.values():
            out.append((bound, value))
    return out


def samples_by_attr(
    derive: Any, bounds: list[dict]
) -> dict[str, list[tuple[dict, Any]]]:
    """Measure a hook, grouping its samples by the attr it returns.

    The multi-output twin of :func:`samples_from_hook`: a hook
    returning ``{"$attr:A": …, "$attr:B": …}`` yields one sample list
    per attr, so each is fitted by its own expr.  The ``$attr:``
    prefix is stripped (the ``dspec`` key convention); a declining
    bound contributes to no group — the same honest drop the
    single-output path takes.
    """
    out: dict[str, list[tuple[dict, Any]]] = {}
    for bound in bounds:
        got = derive(bound)
        if got is None:
            continue
        for key, value in got.items():
            name = (
                key[len("$attr:") :]
                if key.startswith("$attr:")
                else key
            )
            out.setdefault(name, []).append((bound, value))
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
        "tval",
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
    # Two views of the measured outputs: the *fingerprint* set for the
    # all-agree guard (boxed — a tensor payload hashes through the
    # registered ``key`` or by identity, never raising ``TypeError:
    # unhashable``), and the literal seeds ``_leaves`` reads — only
    # ints seed literals anyway, so the raw set is filtered rather
    # than materialised over unhashable values.
    if len({_vkey(v) for _b, v in samples}) < 2:
        # every measured value agrees — no signal to fit; any
        # expr producing that constant "matches" and the synth
        # would be a coincidence, not a derivation.
        return None
    values = {
        v
        for _b, v in samples
        if isinstance(v, int) and not isinstance(v, bool)
    }
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


def synth_dspec_multi(
    derive: Any, bounds: list[dict], *, max_depth: int = 2
) -> dict | None:
    """Fit a multi-output hook — one expr per returned attr.

    The hook is measured over *bounds* with :func:`samples_by_attr`,
    and each ``$attr:NAME`` it returns is fitted *separately* over the
    shared domain.  This is the honest spelling of the hook: a
    ``dspec`` is a ``{NAME: expr}`` map, so the single flattened expr
    the single-output path would build (:func:`samples_from_hook` then
    :func:`synth_dspec`) cannot denote a hook that returns two attrs —
    two different values on one bound are never one expr.  A *partial*
    map is refused: if any attr lacks a fit, ``None`` is returned, so a
    half-expressed hook never ships as a derivation.

    The per-attr guards are :func:`synth_dspec`'s own (thin-set and
    constant-only refusals, the even/odd held-out split), applied to
    each attr's sample list.
    """
    out: dict = {}
    by_attr = samples_by_attr(derive, bounds)
    if not by_attr:
        return None
    for name, samples in by_attr.items():
        spec = synth_dspec(samples, name=name, max_depth=max_depth)
        if spec is None:
            return None
        out.update(spec)
    return out


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
    fp = _box(sig)
    exp = tuple(_vkey(samples[i][1]) for i in range(len(samples)))
    return all(fp[i] == exp[i] for i in fit) and all(
        fp[i] == exp[i] for i in holdout
    )


def _near_misses(
    samples: list[tuple], *, max_depth: int = 2, k: int = 5
) -> list[tuple[Any, int]]:
    """Closest candidates by sample-agreement — the bank-gap probe."""
    if not samples:
        return []
    values = {
        v
        for _b, v in samples
        if isinstance(v, int) and not isinstance(v, bool)
    }
    cands = _candidates(
        samples[0][0], values, max_depth=max_depth, samples=samples
    )
    scored = [
        (
            expr,
            sum(
                1
                for v, (_b, e) in zip(sig, samples, strict=True)
                if _vkey(v) == _vkey(e)
            ),
        )
        for expr, sig in cands
    ]
    scored.sort(key=lambda p: (-p[1], _size(p[0]), repr(p[0])))
    return scored[:k]

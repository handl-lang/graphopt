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
"""

from __future__ import annotations

from itertools import product
from typing import Any

from catopt_core.laws.cond import _DOPS

__all__ = [
    "samples_from_hook",
    "synth_dspec",
]

#: Arity-2 derive ops the synthesizer composes (the arithmetic bank
#: plus the two int-tuple readers — everything else in ``_DOPS`` is
#: either a leaf form or shape-specialised).
BINOPS: tuple[str, ...] = ("add", "sub", "mul", "floordiv")
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
    """Leaf exprs: attr refs, metavar refs, and observed constants."""
    out: list[Any] = []
    for key in sorted(bound):
        if key.startswith("$attr:"):
            out.append(("attr", key[len("$attr:") :]))
    for v in sorted(values, key=repr):
        if isinstance(v, int) and not isinstance(v, bool):
            out.append(("lit", v))
    return out


def _candidates(
    bound: dict, values: set, *, max_depth: int
) -> list[Any]:
    """Enumerate expr trees up to *max_depth* over the leaf set."""
    leaves = _leaves(bound, values)
    level = list(leaves)
    out = list(leaves)
    for _ in range(max_depth):
        nxt = []
        for op in BINOPS:
            for a, b in product(level, level):
                nxt.append((op, a, b))
        for op in UNOPS:
            for a in level:
                nxt.append((op, a))
        # int readers over shape-ish leaves
        for a in level:
            nxt.append(("len", a))
        out.extend(nxt)
        level = nxt
    return out


def _matches(expr: Any, samples: list[tuple]) -> bool:
    """Return True iff *expr* reproduces every measured value."""
    for bound, expected in samples:
        got = _eval(expr, bound)
        if got != expected:
            return False
    return True


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
    if expr[0] in ("attr", "attr0", "shape", "dim", "leaf-dim"):
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
    fit = samples[0::2]
    holdout = samples[1::2]
    bound = samples[0][0]
    cands = _candidates(bound, values, max_depth=max_depth)
    hits = [
        c
        for c in cands
        if _reads_bound(c) and _matches(c, fit) and _matches(c, holdout)
    ]
    if not hits:
        return None
    best = min(hits, key=lambda c: (_size(c), repr(c)))
    return {(name or "OUT"): best}


def _near_misses(
    samples: list[tuple], *, max_depth: int = 2, k: int = 5
) -> list[tuple[Any, int]]:
    """Closest candidates by sample-agreement — the bank-gap probe."""
    if not samples:
        return []
    values = {v for _b, v in samples}
    cands = _candidates(samples[0][0], values, max_depth=max_depth)
    scored = []
    for c in cands:
        agree = sum(1 for b, e in samples if _eval(c, b) == e)
        scored.append((c, agree))
    scored.sort(key=lambda p: (-p[1], _size(p[0]), repr(p[0])))
    return scored[:k]

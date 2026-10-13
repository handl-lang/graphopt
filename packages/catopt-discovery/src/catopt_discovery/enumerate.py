"""Law-shape enumeration — generate candidates over the op grammar.

``catopt_discovery.proposal.schema_candidates`` replays 23 hand-written
algebraic identities; ``catopt_discovery.grammar`` *searches* the schema
space by mutating those seeds.  This tool is the third point of the
design: a **systematic enumerator** that writes down every small
rewriting form over the op vocabulary, so proposal is not limited to
schemas a human listed.  The question it answers is coverage — which
algebraic-identity families does a bounded grammar already contain,
and how does the oracle split them into true / false / duplicate /
new-and-useful?

Enumeration grammar
-------------------

Shapes are the ``object_synthesis.term_from_spec`` trees: a leaf is a
metavariable (the *metas* names) or a literal constant (the
*literals* values), an interior node is ``(op, *children)`` over the
declared arities (``catopt_core.opmeta.arity``, with ``pow`` and
``matmul`` overridden to binary — the registry leaves their arity
per-overload).  :func:`enumerate_shapes` yields every tree of depth
``<= max_depth``, stratified: depth-0 leaves first, then op nodes
whose deepest child is exactly one level up.  The full tree space at
depth 2 over ~9 ops x 3 metas + 3 literals is already ~10^5, so it
streams and honours ``max_shapes``.

Candidates are **not** the shape-squared product — almost every pair
of depth-2 trees is a non-law.  Pairs are enumerated by *rewriting
form*: a finite family of equation skeletons, each one a place a
single algebraic identity could live.  The forms (``b``/``b1``/``b2``
range over the binary ops, ``u``/``u1``/``u2`` over the unary ops,
``k``/``k2`` over the literals, ``X``/``Y``/``Z`` are the first three
metavariables)::

    comm            b(X,Y)                = b(Y,X)
    assoc           b(b(X,Y),Z)           = b(X,b(Y,Z))
    rotate          b1(b2(X,Y),Z)         = b2(X,b1(Y,Z))
    rekey           b1(b1(X,Y),Z)         = b1(X,b2(Y,Z))
    factor-left     b1(b2(X,Y),b2(X,Z))   = b2(X,b1(Y,Z))
    factor-right    b1(b2(Y,X),b2(Z,X))   = b2(b1(Y,Z),X)
    unary-hom       u(b(X,Y))             = b(u(X),u(Y))
    pull-left       u(b(X,Y))             = b(u(X),Y)
    pull-right      u(b(X,Y))             = b(X,u(Y))
    unary-swap      u1(u2(X))             = u2(u1(X))
    involution      u(u(X))               = X
    unary-inverse   u1(u2(X))             = X
    unary-absorb    u1(u2(X))             = u1(X)
    self-op         b(X,X)                = u(X)
    idempotent      b(X,X)                = X
    lit-self        b1(X,X)               = b2(X,k)
    const-op        b(X,k)                = u(X)
    id-right        b(X,k)                = X
    id-left         b(k,X)                = X
    annihilate-r    b(X,k)                = k2
    annihilate-l    b(k,X)                = k2
    cancel-self     b(X,X)                = k
    cancel-unary-r  b(X,u(X))             = k
    cancel-unary-l  b(u(X),X)             = k
    hom-cross       b1(u(X),u(Y))         = u(b2(X,Y))
    bin-unary-r     b1(X,Y)               = b2(X,u(Y))
    bin-unary-l     b1(X,Y)               = b2(u(X),Y)
    unary-of-lit    u(X)                  = b(k,X)

Each form emits **both** orientations — ``x*(y+z)`` expanding and the
factorisation contracting are different candidate laws, and only the
contracting one lowers cost.  Symmetric reverses (``b(X,Y)=b(Y,X)``
spelled backwards is the same equality) collapse in the alpha-key
dedup, so orientation costs nothing there.

Forms stream **round-robin** over the families: a ``max_candidates``
cap cuts a cross-section of every family, not just the earliest.

Pre-evaluation filters (honest pruning — sound vs heuristic):

* *tautology* — alpha-equal sides.  Sound: ``X = X`` is not a law
  proposal, it is the absence of one.
* *unsynthesizable* — an RHS metavariable absent from the LHS (the
  ``meta._synthesizable`` condition).  Sound: a rewrite may not mint
  a fresh metavariable.  The forms never produce one; the check is
  the guard.
* *missing-meta* — a metavariable slot *metas* did not supply reads
  as a ``None`` leaf (a two-metavar enumeration cannot spell the
  three-slot factor shape).  Sound: the pair is unspellable.
* *domain* — the op's operand domain: ``pow``'s exponent child must
  be a literal (the scalar-exponent spelling the library uses), and
  ``matmul``'s children may not be bare literals (an ill-typed
  operand).  Sound-by-grammar: these are shape-type constraints, not
  truth guesses.
* *lhs-not-op* — the LHS must contain at least one op node.
  Heuristic, not sound: ``0 = mul(X,0)``-style reversed annihilators
  are true equalities, but a leaf-rooted LHS matches *every* term
  and saturates pathologically, so the enumerator declines them.
* *dedup* — the ``proposal._relation`` key (alpha-normal pair);
  repeated emissions collapse, and the surviving candidate carries
  the library relation (``duplicate`` / ``inverse`` / ``new``) in
  its ``note``.

Metavariables instantiate on ``(4,4)`` fp64-vars — uniform leaf
shapes, the same convention ``proposal``/``grammar`` use — so every
emitted :class:`catopt_discovery.proposal.Candidate` flows through
``proposal.evaluate`` (verify_law + the numeric oracle + cost delta)
unchanged.  The CLI evaluates under the bounded-saturation variant
``grammar._evaluate`` established (the unbounded verify can
comm-assoc-explode on mul/add chains).

Run::

    .venv/bin/python -m catopt_discovery.enumerate \
        --ops add,mul,neg,sub --max-candidates 200 --evaluate 40

CPU-only; the evaluated prefix is the bounded part.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import deque
from pathlib import Path
from typing import Any

from catopt_core.ir import Const, Op, TensorType, Var, op_repr
from catopt_core.opmeta import arity as _op_arity

from catopt_discovery import lawdata
from catopt_discovery import proposal as lp
from catopt_discovery.grammar import _evaluate as _bounded_evaluate
from catopt_discovery.object_synthesis import term_from_spec

__all__ = [
    "DEFAULT_OPS",
    "enumerate_candidates",
    "enumerate_shapes",
    "main",
]

#: Default op vocabulary — the pointwise core (4 binary + 4 unary).
#: ``pow``/``matmul``/the rest of ``GRAMMAR_*`` are opt-in via
#: ``--ops``; ``matmul`` instantiates on the same rank-2 leaves.
DEFAULT_OPS: tuple = (
    "add",
    "mul",
    "sub",
    "div",
    "neg",
    "exp",
    "square",
    "sqrt",
)

#: Arity overrides for ops whose declared arity is per-overload
#: (``opmeta.arity`` returns ``None``): ``pow``'s tensor-exponent
#: spelling and ``matmul`` are both two-operand here.
_ARITY_OVERRIDES: dict[str, int] = {"pow": 2, "matmul": 2}

#: Child positions that must be a literal (``pow``'s exponent — the
#: scalar-exponent spelling every shipped pow law uses).
_LITERAL_CHILDREN: dict[str, tuple] = {"pow": (1,)}

#: Child positions that must be a tensor — never a bare literal
#: (``matmul(x, 0)`` is ill-typed, not a false law).
_TENSOR_CHILDREN: dict[str, tuple] = {"matmul": (0, 1)}

#: Uniform leaf shape the metavars instantiate on (``proposal._D``).
_D: int = lp._D


# ---------------------------------------------------------------------------
#  Op vocabulary
# ---------------------------------------------------------------------------


def _split(ops: tuple) -> tuple[tuple, tuple]:
    """Split *ops* into ``(binary, unary)`` by declared arity."""
    bin_: list[str] = []
    un_: list[str] = []
    for op in ops:
        a = _ARITY_OVERRIDES.get(op, _op_arity(op))
        if a == 2:
            bin_.append(op)
        elif a == 1:
            un_.append(op)
        else:
            raise ValueError(
                f"op {op!r} has no usable arity "
                f"(opmeta.arity={_op_arity(op)})"
            )
    return tuple(bin_), tuple(un_)


def _depth(spec: Any) -> int:
    """Return the depth of a spec tree (a leaf is 0)."""
    if not isinstance(spec, tuple):
        return 0
    return 1 + max(_depth(c) for c in spec[1:])


def _metas_used(spec: Any, out: set | None = None) -> set:
    """Return every metavariable (``str`` leaf) used in *spec*."""
    out = set() if out is None else out
    if isinstance(spec, tuple):
        for c in spec[1:]:
            _metas_used(c, out)
    elif isinstance(spec, str):
        out.add(spec)
    return out


# ---------------------------------------------------------------------------
#  The raw shape stream
# ---------------------------------------------------------------------------


def _level(bin_: tuple, un_: tuple, below: list, prev: list) -> list:
    """Return the trees whose depth is exactly one above *prev*.

    Unary children come from the previous level only; binary children
    range over everything below with at least one side at the
    previous level — the stratification that emits each tree once.
    """
    lo = len(below) - len(prev)
    level: list = [(u, t) for u in un_ for t in prev]
    for b in bin_:
        for i, t1 in enumerate(below):
            for j, t2 in enumerate(below):
                if i < lo and j < lo:
                    continue
                level.append((b, t1, t2))
    return level


def _tree_specs(
    ops: tuple,
    max_depth: int,
    metas: tuple,
    literals: tuple,
):
    """Yield every spec tree of depth ``<= max_depth``, stratified."""
    bin_, un_ = _split(ops)
    prev: list = [*metas, *literals]
    below = list(prev)
    yield from below
    for _ in range(max_depth):
        level = _level(bin_, un_, below, prev)
        yield from level
        below.extend(level)
        prev = level


def enumerate_shapes(
    ops: tuple | None = None,
    *,
    max_depth: int = 2,
    metas: tuple = ("A", "B", "C"),
    literals: tuple | None = None,
    max_shapes: int = 100_000,
):
    """Yield every expression tree of depth ``<= max_depth``.

    Returns pattern terms (``Op`` nodes over ``str`` metavariable
    leaves and ``Const`` literals — the same shape
    ``shape_proposal`` schemas use), in a deterministic stratified
    order: depth-0 leaves (metavariables in *metas* order, then
    literals), then each deeper level whose deepest child sits one
    level up.  This is the *raw* grammar — operand-domain
    constraints (``pow``'s literal exponent) apply at candidate
    emission, not here.  The stream is a generator bounded by
    *max_shapes*; depth 3 is impractical (the level below is already
    ~10^5 trees, and binary pairing squares it).
    """
    ops = DEFAULT_OPS if ops is None else tuple(ops)
    lits = lawdata.GRAMMAR_LITERALS if literals is None else literals
    for n, spec in enumerate(
        _tree_specs(ops, max_depth, metas, lits), start=1
    ):
        yield term_from_spec(spec)
        if n >= max_shapes:
            return


# ---------------------------------------------------------------------------
#  Rewriting forms — one generator per equation family
# ---------------------------------------------------------------------------
#
#  Each generator yields unoriented ``(lhs_spec, rhs_spec)`` pairs;
#  the emission pipeline tries both directions.  Missing
#  metavariables (``metas`` shorter than a form needs) read as
#  ``None`` and skip the form's yield.


def _f_comm(B, U, X, Y, Z, K):
    """Yield ``b(X,Y) = b(Y,X)`` — commutativity probes."""
    for b in B:
        yield "comm", (b, X, Y), (b, Y, X)


def _f_assoc(B, U, X, Y, Z, K):
    """Yield ``b(b(X,Y),Z) = b(X,b(Y,Z))`` — associativity."""
    if Z is None:
        return
    for b in B:
        yield "assoc", (b, (b, X, Y), Z), (b, X, (b, Y, Z))


def _f_rotate(B, U, X, Y, Z, K):
    """Yield ``b1(b2(X,Y),Z) = b2(X,b1(Y,Z))`` — nest rotation."""
    if Z is None:
        return
    for b1 in B:
        for b2 in B:
            yield (
                "rotate",
                (b1, (b2, X, Y), Z),
                (b2, X, (b1, Y, Z)),
            )


def _f_rekey(B, U, X, Y, Z, K):
    """Yield ``b1(b1(X,Y),Z) = b1(X,b2(Y,Z))`` — mixed assoc."""
    if Z is None:
        return
    for b1 in B:
        for b2 in B:
            yield (
                "rekey",
                (b1, (b1, X, Y), Z),
                (b1, X, (b2, Y, Z)),
            )


def _f_factor_left(B, U, X, Y, Z, K):
    """Yield ``b1(b2(X,Y),b2(X,Z)) = b2(X,b1(Y,Z))`` — factor."""
    if Z is None:
        return
    for b1 in B:
        for b2 in B:
            yield (
                "factor-left",
                (b1, (b2, X, Y), (b2, X, Z)),
                (b2, X, (b1, Y, Z)),
            )


def _f_factor_right(B, U, X, Y, Z, K):
    """Yield ``b1(b2(Y,X),b2(Z,X)) = b2(b1(Y,Z),X)`` — factor."""
    if Z is None:
        return
    for b1 in B:
        for b2 in B:
            yield (
                "factor-right",
                (b1, (b2, Y, X), (b2, Z, X)),
                (b2, (b1, Y, Z), X),
            )


def _f_unary_hom(B, U, X, Y, Z, K):
    """Yield ``u(b(X,Y)) = b(u(X),u(Y))`` — unary homomorphism."""
    for u in U:
        for b in B:
            yield (
                "unary-hom",
                (u, (b, X, Y)),
                (b, (u, X), (u, Y)),
            )


def _f_pull_left(B, U, X, Y, Z, K):
    """Yield ``u(b(X,Y)) = b(u(X),Y)`` — pull through arg 0."""
    for u in U:
        for b in B:
            yield "pull-left", (u, (b, X, Y)), (b, (u, X), Y)


def _f_pull_right(B, U, X, Y, Z, K):
    """Yield ``u(b(X,Y)) = b(X,u(Y))`` — pull through arg 1."""
    for u in U:
        for b in B:
            yield "pull-right", (u, (b, X, Y)), (b, X, (u, Y))


def _f_unary_swap(B, U, X, Y, Z, K):
    """Yield ``u1(u2(X)) = u2(u1(X))`` — unary commutation."""
    for u1 in U:
        for u2 in U:
            yield "unary-swap", (u1, (u2, X)), (u2, (u1, X))


def _f_involution(B, U, X, Y, Z, K):
    """Yield ``u(u(X)) = X`` — involutions."""
    for u in U:
        yield "involution", (u, (u, X)), X


def _f_unary_inverse(B, U, X, Y, Z, K):
    """Yield ``u1(u2(X)) = X`` — inverse unary pairs."""
    for u1 in U:
        for u2 in U:
            yield "unary-inverse", (u1, (u2, X)), X


def _f_unary_absorb(B, U, X, Y, Z, K):
    """Yield ``u1(u2(X)) = u1(X)`` — inner-op absorption."""
    for u1 in U:
        for u2 in U:
            yield "unary-absorb", (u1, (u2, X)), (u1, X)


def _f_self_op(B, U, X, Y, Z, K):
    """Yield ``b(X,X) = u(X)`` — self-operation spellings."""
    for b in B:
        for u in U:
            yield "self-op", (b, X, X), (u, X)


def _f_idempotent(B, U, X, Y, Z, K):
    """Yield ``b(X,X) = X`` — idempotence."""
    for b in B:
        yield "idempotent", (b, X, X), X


def _f_lit_self(B, U, X, Y, Z, K):
    """Yield ``b1(X,X) = b2(X,k)`` — ``x+x = x*2`` family."""
    for b1 in B:
        for b2 in B:
            for k in K:
                yield "lit-self", (b1, X, X), (b2, X, k)


def _f_const_op(B, U, X, Y, Z, K):
    """Yield ``b(X,k) = u(X)`` — ``pow(x,2) = square(x)`` family."""
    for b in B:
        for k in K:
            for u in U:
                yield "const-op", (b, X, k), (u, X)


def _f_id_right(B, U, X, Y, Z, K):
    """Yield ``b(X,k) = X`` — right identity elements."""
    for b in B:
        for k in K:
            yield "id-right", (b, X, k), X


def _f_id_left(B, U, X, Y, Z, K):
    """Yield ``b(k,X) = X`` — left identity elements."""
    for b in B:
        for k in K:
            yield "id-left", (b, k, X), X


def _f_annihilate_right(B, U, X, Y, Z, K):
    """Yield ``b(X,k) = k2`` — right annihilators."""
    for b in B:
        for k in K:
            for k2 in K:
                yield "annihilate-right", (b, X, k), k2


def _f_annihilate_left(B, U, X, Y, Z, K):
    """Yield ``b(k,X) = k2`` — left annihilators."""
    for b in B:
        for k in K:
            for k2 in K:
                yield "annihilate-left", (b, k, X), k2


def _f_cancel_self(B, U, X, Y, Z, K):
    """Yield ``b(X,X) = k`` — self-cancellation."""
    for b in B:
        for k in K:
            yield "cancel-self", (b, X, X), k


def _f_cancel_unary_right(B, U, X, Y, Z, K):
    """Yield ``b(X,u(X)) = k`` — ``x + (-x) = 0`` family."""
    for b in B:
        for u in U:
            for k in K:
                yield "cancel-unary-right", (b, X, (u, X)), k


def _f_cancel_unary_left(B, U, X, Y, Z, K):
    """Yield ``b(u(X),X) = k`` — left-inverse cancellation."""
    for b in B:
        for u in U:
            for k in K:
                yield "cancel-unary-left", (b, (u, X), X), k


def _f_hom_cross(B, U, X, Y, Z, K):
    """Yield ``b1(u(X),u(Y)) = u(b2(X,Y))`` — cross homomorphism."""
    for b1 in B:
        for u in U:
            for b2 in B:
                yield (
                    "hom-cross",
                    (b1, (u, X), (u, Y)),
                    (u, (b2, X, Y)),
                )


def _f_bin_unary_right(B, U, X, Y, Z, K):
    """Yield ``b1(X,Y) = b2(X,u(Y))`` — ``x-y = x+(-y)`` family."""
    for b1 in B:
        for b2 in B:
            for u in U:
                yield "bin-unary-right", (b1, X, Y), (b2, X, (u, Y))


def _f_bin_unary_left(B, U, X, Y, Z, K):
    """Yield ``b1(X,Y) = b2(u(X),Y)`` — left-slot variant."""
    for b1 in B:
        for b2 in B:
            for u in U:
                yield "bin-unary-left", (b1, X, Y), (b2, (u, X), Y)


def _f_unary_of_lit(B, U, X, Y, Z, K):
    """Yield ``u(X) = b(k,X)`` — ``-x = 0-x`` family."""
    for u in U:
        for b in B:
            for k in K:
                yield "unary-of-lit", (u, X), (b, k, X)


#: The rewriting-form generators, in canonical-first emission order.
_FORMS: tuple = (
    _f_comm,
    _f_assoc,
    _f_rotate,
    _f_rekey,
    _f_factor_left,
    _f_factor_right,
    _f_unary_hom,
    _f_pull_left,
    _f_pull_right,
    _f_unary_swap,
    _f_involution,
    _f_unary_inverse,
    _f_unary_absorb,
    _f_self_op,
    _f_idempotent,
    _f_lit_self,
    _f_const_op,
    _f_id_right,
    _f_id_left,
    _f_annihilate_right,
    _f_annihilate_left,
    _f_cancel_self,
    _f_cancel_unary_right,
    _f_cancel_unary_left,
    _f_hom_cross,
    _f_bin_unary_right,
    _f_bin_unary_left,
    _f_unary_of_lit,
)


def _form_pairs(bin_: tuple, un_: tuple, metas: tuple, literals: tuple):
    """Yield ``(form, lhs_spec, rhs_spec)`` round-robin over forms."""
    x, y, z = (*metas, None, None, None)[:3]
    gens = deque(iter(f(bin_, un_, x, y, z, literals)) for f in _FORMS)
    while gens:
        gen = gens.popleft()
        try:
            yield next(gen)
        except StopIteration:
            continue
        gens.append(gen)


# ---------------------------------------------------------------------------
#  Pre-evaluation filters
# ---------------------------------------------------------------------------


def _domain_ok(spec: Any) -> bool:
    """Return whether *spec*'s operand domains hold, recursively.

    ``pow``'s exponent child must be a literal; ``matmul``'s children
    must not be bare literals.  Both are shape-type constraints —
    a ``pow`` tree with a tensor exponent or a ``matmul`` tree with
    a literal operand is ill-typed, not a false law.
    """
    if not isinstance(spec, tuple):
        return True
    op, children = spec[0], spec[1:]
    for pos in _LITERAL_CHILDREN.get(op, ()):
        if pos < len(children) and isinstance(
            children[pos], (tuple, str)
        ):
            return False
    for pos in _TENSOR_CHILDREN.get(op, ()):
        if pos < len(children) and not isinstance(
            children[pos], (tuple, str)
        ):
            return False
    return all(_domain_ok(c) for c in children)


def _has_hole(spec: Any) -> bool:
    """Return whether *spec* carries a ``None`` leaf.

    A ``None`` leaf is a metavariable slot the caller's *metas* did
    not supply (a two-metavar enumeration cannot spell a three-slot
    factor shape) — the pair is unspellable, so it is dropped.
    """
    if not isinstance(spec, tuple):
        return spec is None
    return any(_has_hole(c) for c in spec[1:])


def _filter_reason(lhs: Any, rhs: Any, max_depth: int) -> str | None:
    """Return why a pair is skipped pre-evaluation, or ``None``.

    Tautology and the alpha-key dedup are handled by the caller (the
    key is computed there anyway); this covers the remaining
    sound-by-construction filters (synthesizability, operand domains,
    missing metavariables) plus the documented heuristic ones (depth
    cap, op-rooted LHS).
    """
    if not isinstance(lhs, tuple):
        return "lhs-not-op"
    if _has_hole(lhs) or _has_hole(rhs):
        return "missing-meta"
    if not _metas_used(rhs) <= _metas_used(lhs):
        return "unsynthesizable"
    if not _domain_ok(lhs) or not _domain_ok(rhs):
        return "domain"
    if _depth(lhs) > max_depth or _depth(rhs) > max_depth:
        return "deep"
    return None


def _instantiate(spec: Any, env: dict) -> Any:
    """Instantiate a spec: metas -> env vars, ints -> ``Const``."""
    if isinstance(spec, tuple):
        return Op.make(
            spec[0], *(_instantiate(c, env) for c in spec[1:])
        )
    if isinstance(spec, str):
        return env[spec]
    return Const(spec)


# ---------------------------------------------------------------------------
#  The candidate stream
# ---------------------------------------------------------------------------


def enumerate_candidates(
    ops: tuple | None = None,
    *,
    metas: tuple = ("A", "B", "C"),
    literals: tuple | None = None,
    max_depth: int = 2,
    max_candidates: int | None = None,
):
    """Yield ``Candidate``s paired by rewriting form, deduped.

    Streams the form generators round-robin (a cap cuts a
    cross-section of every family), tries each pair in both
    orientations, drops the pre-evaluation-filtered pairs
    (``_filter_reason`` plus tautology), dedupes on the
    alpha-normal ``proposal._key``, and annotates each survivor
    with its library relation in ``note``.  Metavariables
    instantiate on ``(4,4)`` vars, so the emitted candidates run
    through ``proposal.evaluate`` unchanged.
    """
    ops = DEFAULT_OPS if ops is None else tuple(ops)
    lits = lawdata.GRAMMAR_LITERALS if literals is None else literals
    bin_, un_ = _split(ops)
    lib = lp._library_keys()
    env = {m: Var(m, TensorType((_D, _D))) for m in metas}
    seen: set = set()
    n = 0
    for form, lhs, rhs in _form_pairs(bin_, un_, metas, lits):
        for ls, rs in ((lhs, rhs), (rhs, lhs)):
            if _filter_reason(ls, rs, max_depth) is not None:
                continue
            lt, rt = term_from_spec(ls), term_from_spec(rs)
            key = lp._key(lt, rt)
            if key[0] == key[1] or key in seen:
                continue
            seen.add(key)
            rel = lp._relation(lt, rt, lib)
            yield lp.Candidate(
                "enumerate",
                f"{form}:{n}",
                _instantiate(ls, env),
                _instantiate(rs, env),
                note=(f"{op_repr(lt)} = {op_repr(rt)}; relation={rel}"),
            )
            n += 1
            if max_candidates is not None and n >= max_candidates:
                return


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _relation_counts(cands: list) -> dict[str, int]:
    """Histogram the ``relation=…`` annotation over emitted candidates."""
    out: dict[str, int] = {}
    for c in cands:
        rel = c.note.rsplit("relation=", 1)[-1]
        out[rel] = out.get(rel, 0) + 1
    return out


def _top_ranked(outcomes: list, k: int) -> str:
    """Render the top-*k* candidates by the cost-delta heuristic."""
    scored = sorted(
        outcomes,
        key=lambda o: -lp._heuristic_score(o.candidate),
    )
    lines = []
    for o in scored[:k]:
        c = o.candidate
        verdict = (
            f"rel={o.relation} ntrue={o.num_true} "
            f"deriv={o.derivable} useful={o.useful}"
        )
        lines.append(
            f"  {op_repr(c.lhs)} = {op_repr(c.rhs)}\n      {verdict}"
        )
    return "\n".join(lines)


def _parse_csv(text: str, cast: Any = str) -> tuple:
    """Parse a comma-separated CLI list into a tuple."""
    return tuple(cast(x) for x in text.split(",") if x)


def _dump_json(path: str, result: dict) -> None:
    """Write the machine-readable enumeration report."""
    Path(path).write_text(json.dumps(result, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Enumerate law shapes, evaluate a prefix, print the yield."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ops",
        default=",".join(DEFAULT_OPS),
        help="comma-separated op vocabulary",
    )
    parser.add_argument(
        "--metas", default="A,B,C", help="comma-separated metavar names"
    )
    parser.add_argument(
        "--literals", default="0,1,2", help="comma-separated literals"
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=100_000,
        help="cap on the raw shape census",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=400,
        help="cap on emitted candidates",
    )
    parser.add_argument(
        "--evaluate",
        type=int,
        default=60,
        help="bounded-evaluate the first N emitted candidates",
    )
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    ops = _parse_csv(args.ops)
    metas = _parse_csv(args.metas)
    literals = _parse_csv(
        args.literals,
        lambda s: int(s) if s.lstrip("-").isdigit() else float(s),
    )

    n_shapes = sum(
        1
        for _ in enumerate_shapes(
            ops,
            max_depth=args.max_depth,
            metas=metas,
            literals=literals,
            max_shapes=args.max_shapes,
        )
    )
    stream = enumerate_candidates(
        ops,
        metas=metas,
        literals=literals,
        max_depth=args.max_depth,
        max_candidates=args.max_candidates,
    )
    cands = list(itertools.islice(stream, args.max_candidates))
    outcomes = _bounded_evaluate(cands[: args.evaluate])
    table = lp.yield_table(outcomes)

    print(  # stdout-compat
        "== law-shape enumeration — candidates from the grammar =="
    )
    print(  # stdout-compat
        f"   ops={list(ops)} metas={list(metas)} "
        f"literals={list(literals)} depth<={args.max_depth}"
    )
    print(  # stdout-compat
        f"   shapes enumerated: {n_shapes}"
        + (" (capped)" if n_shapes >= args.max_shapes else "")
    )
    print(  # stdout-compat
        f"   candidates emitted: {len(cands)}"
        + (
            f" (capped at {args.max_candidates})"
            if len(cands) >= args.max_candidates
            else ""
        )
        + f"   relation: {_relation_counts(cands)}"
    )
    print(  # stdout-compat
        f"   evaluated (bounded): {len(outcomes)}"
    )
    print()  # stdout-compat
    print(lp._fmt_yield(table))  # stdout-compat
    print()  # stdout-compat
    print("== genuinely-new, useful laws found ==")  # stdout-compat
    print(lp._fmt_useful(outcomes))  # stdout-compat
    print()  # stdout-compat
    print("== top-ranked (cost-delta heuristic) ==")  # stdout-compat
    print(_top_ranked(outcomes, 8))  # stdout-compat

    if args.json:
        _dump_json(
            args.json,
            {
                "ops": list(ops),
                "n_shapes": n_shapes,
                "emitted": len(cands),
                "relation": _relation_counts(cands),
                "yield": table,
                "outcomes": [
                    {
                        "form": o.candidate.label.split(":", 1)[0],
                        "lhs": op_repr(o.candidate.lhs),
                        "rhs": op_repr(o.candidate.rhs),
                        "derivable": o.derivable,
                        "num_true": o.num_true,
                        "relation": o.relation,
                        "useful": o.useful,
                    }
                    for o in outcomes
                ],
            },
        )
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

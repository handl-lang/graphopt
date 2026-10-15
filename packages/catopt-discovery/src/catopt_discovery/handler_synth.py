"""Handler-interpreted atoms — ADR 0007 layer 2.

``genkernel`` mints handlers whose kernel evaluates the spec's own
spelled body — the *elementwise* special case: the proposed
semantics is the pattern itself.  This module generalises the
proposal site: a minted atom's semantics is a **handler body** — a
small descriptor drawn from a structured grammar of loop-shaped and
contraction-shaped templates — proposed, then verified numerically
against the pre-mint subterm on a measured domain, the same bar a
law clears.  Verify-or-refuse is the whole contract: a body that
does not reproduce the spelled term on every measured binding is a
refusal, recorded with its measured error — never a silent
fallback.

THE GRAMMAR (descriptors are plain data — tuples, strings, numbers):

* ``("ew", sub)`` — elementwise: evaluate *sub* pointwise.  The
  ``genkernel`` tier re-spelled as a descriptor; proposed only for
  genuinely elementwise specs — the replay handler is the
  degenerate layer-2 case, already shipped.
* ``("contract", eq, lhs, rhs)`` — two-factor contraction:
  ``einsum(eq, EW(lhs), EW(rhs))`` where *lhs* / *rhs* are
  elementwise trees over metavar subsets that together cover the
  spec's metas.  The ``om_decayed`` element ``(s ⊙ d) @ v`` is
  ``("contract", "...ik,...kj->...ij", ("mul","S","D"), "V")`` — a
  handler the hand-written carriers needed an executor table for.
* ``("fold", a, b, state, axis)`` — first-order affine recurrence:
  ``acc_t = ew(a)_t ⊙ acc_{t-1} + ew(b)_t`` walked along *axis*;
  the operands broadcast into a shared step space exactly as the
  scan tier's ``_seq_args`` normalises them.  *state* names the
  metavariable bound to the initial accumulator.
* ``("fold_emit", a, b, state, emit, axis)`` — the recurrence plus
  a per-step projection ``out_t = ew(emit)`` evaluated against the
  step-sliced operands and the running accumulator bound at the
  reserved leaf ``"$acc"``; outputs stack back on *axis*.
* ``("fold_zip", steps, state)`` — the flat chronological fold:
  *steps* is a tuple of ``(a_i, b_i)`` metavar pairs applied in
  order — ``acc = a_i ⊙ acc + b_i`` — the semantics an unrolled
  ``applyd`` nest already spells.
* ``("reduce", op, sub, axis)`` — the reduction sibling of the
  contraction: ``op`` in ``_REDUCE_OPS`` applied along *axis* to the
  elementwise *sub* over every meta.  ``sum(X, dim=-1)`` is
  ``("reduce", "sum", "X", -1)`` — the shape the loop/contraction
  families cannot spell.
* ``("norm", op, lhs, red_op, rhs, axis, keepdim)`` — the
  normalization: an elementwise ``op`` combining *lhs* with a
  reduction ``red_op`` of *rhs* along *axis*.  ``softmax`` is
  ``("norm", "div", ("exp","X"), "sum", ("exp","X"), -1, True)`` —
  the composite the ``reduce`` family's whole-spec operand cannot
  spell (a reduction is one of the operands of an elementwise op,
  not the whole spec).  *keepdim* is read off the spec
  (:func:`_spec_keepdim`), not searched: a normalization's
  denominator must keep the reduced axis to broadcast back against
  the operand, so the spec's own ``keepdim`` is the only sound
  value.

A fold side or carry may be a *constant*: the a-side ``1.0`` (the
multiplicative identity) and the b-side ``0.0`` (the additive
identity) are proposed for an empty partition, and the carry may be
the neutral ``0.0``/``1.0`` instead of a metavar — so a spec with no
explicit carry meta (``cumsum(X)``, ``cumprod(X)``, the corpus's
selective scan) spells its recurrence without inventing a state.

Every descriptor is JSON-able provenance data: grammar path,
verified domain, measured error.  The
:func:`handler_record` / :func:`handler_from_record` pair
round-trips a minted atom without the live callable — the atom IS
the record.

The minted entry is a handler-table row ``{pattern, kernel, args,
handler_body, provenance}`` — the same shape ``genkernel``'s tiers
mint, so the arena's ``claim`` / ``handle`` binds it with no new
machinery, and :func:`minted_bindings` / :func:`minted_sink`
deliver it exactly as ``gen_bindings`` / ``gen_sink`` deliver the
elementwise claims.
"""

from __future__ import annotations

from itertools import combinations, permutations, product
from typing import Any, NamedTuple

from catopt_discovery.genkernel import (
    _eval_spec,
    _spec_metas,
    elementwise_spec,
)

__all__ = [
    "CONTRACT_EQS",
    "FOLD_AXES",
    "FOLD_CARRIES",
    "NORM_OPS",
    "REDUCE_AXES",
    "REDUCE_OPS",
    "VerifyResult",
    "body_from_data",
    "body_to_data",
    "eval_body",
    "handler_from_record",
    "handler_record",
    "mint_handlers",
    "minted_bindings",
    "minted_sink",
    "propose_bodies",
    "verify_body",
]

# ---------------------------------------------------------------------------
#  The grammar — vocabularies as data
# ---------------------------------------------------------------------------

#: Elementwise unary ops a handler body may apply to a meta leaf.
_UNARY = ("neg", "sigmoid", "exp", "tanh", "square", "relu", "silu")

#: Elementwise binary ops — the combination vocabulary.  ``mul``
#: first: the carrier elements are products, so the cheapest
#: candidate in simplest-first order is the plausible one.
_BINARY = ("mul", "add", "sub", "div")

#: Maximum op nodes in one elementwise factor — the grammar's
#: bounded size, chosen so ``matmul(mul(mul(S,D),M),V)`` (two ops)
#: stays reachable while the candidate space stays small.
_MAX_EW_OPS = 2

#: Maximum op nodes in a ``fold_emit`` projection's non-accumulator
#: factor.  Kept at one: the emitted-scan corpus shape projects the
#: accumulator itself (``out = $acc``) and the next-deepest useful
#: form is one op (``out = q ⊙ acc``); a two-op projection
#: (``out = q ⊙ acc + r``) is outside the cap — confirmed by
#: measurement, not assumed (see :func:`_emit_exprs`).
_MAX_EMIT_OPS = 1

#: The contraction equations the grammar may propose — batched
#: matmul in its four orientations, the output-transposed form, the
#: matvec and the full inner product.  A closed set: an equation
#: outside it is a grammar extension, not a candidate.
CONTRACT_EQS: tuple[str, ...] = (
    "...ik,...kj->...ij",
    "...ik,...jk->...ij",
    "...ki,...kj->...ij",
    "...ki,...jk->...ij",
    "...ik,...kj->...ji",
    "...ij,...j->...i",
    "...i,...i->...",
)

#: Step axes a fold body may declare.
FOLD_AXES: tuple[int, ...] = (-2, -1)

#: The neutral carries a fold may start from — the additive and
#: multiplicative identities.  A constant carry lets a spec with no
#: explicit state metavar spell its recurrence (``cumsum(X)``,
#: ``cumprod(X)``, the corpus's zero-init selective scan); it is the
#: scan's own neutral element, not a guessed parameter.
FOLD_CARRIES: tuple[float, ...] = (0.0, 1.0)

#: Reductions the grammar may propose — exactly the dim-reductions
#: the torch bridge lowers (``sum``/``mean``/``prod``/``amax``/
#: ``amin``).  A reduction outside this set is a grammar extension,
#: not a candidate.
REDUCE_OPS: tuple[str, ...] = ("sum", "mean", "prod", "amax", "amin")

#: Reduction axes the grammar may declare.  The spec's own ``dim``
#: attrs join this per-spec (read off the signature), so the common
#: last-axis reduction is always reachable while a middle-axis one is
#: reached through the spec's own declared axis.
REDUCE_AXES: tuple[int, ...] = (-1, -2)

#: Outer elementwise ops a normalization may use — the center
#: (``x - red(x)``) and scale (``x / red(x)``) shapes.  A
#: normalization combines a value with a reduction of itself, which
#: the ``reduce`` family (reduction of the whole spec) cannot spell.
NORM_OPS: tuple[str, ...] = ("div", "sub")

#: The template families, in proposal order.
_FAMILIES: tuple[str, ...] = (
    "ew",
    "contract",
    "fold",
    "fold_emit",
    "fold_zip",
    "reduce",
    "norm",
)

#: fp64 verification gate — a handler reproduces its pre-mint term
#: to float64 round-off or it refuses.
_VERIFY_RTOL = 1e-9
_VERIFY_ATOL = 1e-12

#: Candidate-verification cap per spec — the bounded search the
#: refusal records quote against ("no candidate verified within
#: budget" is a measured refusal, not a guess).
_BUDGET = 2048

#: Grammar version stamped into provenance — a minted record names
#: the vocabulary it was verified under.
_GRAMMAR_VERSION = "handler-grammar-v1"


# ---------------------------------------------------------------------------
#  The descriptor interpreter — a minted atom's semantics
# ---------------------------------------------------------------------------


def eval_body(body: Any, env: dict, bindings: dict) -> Any:
    """Evaluate a handler-body descriptor on bound tensor values.

    *env* maps metavar names to tensors — the minted op's declared
    semantics.  *bindings* is the op-name → callable table the
    elementwise leaves dispatch through (the ambient lowering
    table).  An unknown descriptor tag raises ``ValueError``: a
    malformed body fails loudly, never silently degrades.
    """
    import torch

    kind = body[0] if isinstance(body, (tuple, list)) else None
    if kind == "ew":
        return _eval_spec(body[1], env, bindings)
    if kind == "contract":
        _, eq, lhs, rhs = body
        return torch.einsum(
            eq,
            _eval_spec(lhs, env, bindings),
            _eval_spec(rhs, env, bindings),
        )
    if kind in ("fold", "fold_emit"):
        return _fold_eval(body, env, bindings)
    if kind == "fold_zip":
        _, steps, state = body
        acc = _as_tensor(_carry_value(env, state))
        for a_s, b_s in steps:
            acc = _as_tensor(_eval_spec(a_s, env, bindings)) * acc + (
                _as_tensor(_eval_spec(b_s, env, bindings))
            )
        return acc
    if kind == "reduce":
        _, op, sub, axis = body
        return bindings[op](_eval_spec(sub, env, bindings), dim=axis)
    if kind == "norm":
        _, op, lhs, red_op, rhs, axis, keepdim = body
        return bindings[op](
            _eval_spec(lhs, env, bindings),
            bindings[red_op](
                _eval_spec(rhs, env, bindings),
                dim=axis,
                keepdim=keepdim,
            ),
        )
    raise ValueError(f"unknown handler-body tag: {kind!r}")


def _carry_value(env: dict, state: Any) -> Any:
    """Resolve a fold carry: a metavar's value or a constant neutral.

    *state* is a metavar name (``env[state]``) or a bare constant —
    the neutral ``0.0``/``1.0`` a spec with no explicit carry meta
    needs.  A constant passes through untouched; ``_as_tensor`` lifts
    it downstream.
    """
    return env[state] if isinstance(state, str) else state


def _as_tensor(v: Any) -> Any:
    """Lift a scalar operand to a 0-dim tensor so it can broadcast.

    A law or term may spell a fold operand — the initial state above
    all — as a bare constant: ``Const(0.0)`` is the zero-carry scan,
    the selective-scan site's init.  ``_eval_spec`` returns that
    constant as a python number, and ``broadcast_tensors`` /
    ``broadcast_to`` reject a bare number, so the operand is lifted
    here; a tensor passes through untouched.  Without this the
    recognition law that rewrites the spelled emitted sequence into
    the minted atom is structurally unevaluable.
    """
    import torch

    return v if torch.is_tensor(v) else torch.as_tensor(v)


def _fold_eval(body: Any, env: dict, bindings: dict) -> Any:
    """Run the sequential recurrence of a ``fold``/``fold_emit`` body.

    The ``a``/``b`` operands broadcast together into one step space;
    *axis* selects the step axis of that shared shape and the state
    metavar broadcasts into the state space (the step axis removed).
    A rank-0 broadcast — operands carrying no step axis — raises: a
    fold over no steps is a malformed call, not a degenerate one.
    """
    import torch

    if body[0] == "fold":
        _, a_s, b_s, state, axis = body
        emit = None
    else:
        _, a_s, b_s, state, emit, axis = body
    av, bv = torch.broadcast_tensors(
        _as_tensor(_eval_spec(a_s, env, bindings)),
        _as_tensor(_eval_spec(b_s, env, bindings)),
    )
    full = tuple(av.shape)
    if not full:
        raise ValueError("fold operands carry no step axis")
    ax = axis % len(full)
    acc = torch.broadcast_to(
        _as_tensor(_carry_value(env, state)), full[:ax] + full[ax + 1 :]
    )
    a_seq = av.movedim(ax, 0).unbind(0)
    b_seq = bv.movedim(ax, 0).unbind(0)
    slices = (
        _step_slices(emit, env, full, ax) if emit is not None else {}
    )
    outs = []
    for t in range(full[ax]):
        acc = a_seq[t] * acc + b_seq[t]
        if emit is not None:
            env_t = {m: v[t] for m, v in slices.items()}
            env_t["$acc"] = acc
            outs.append(_eval_spec(emit, env_t, bindings))
    if emit is None:
        return acc
    return torch.stack(outs, dim=ax)


def _step_slices(emit: Any, env: dict, full: tuple, ax: int) -> dict:
    """Per-step operand slices for an emit expression.

    Every metavar the emit names (besides the reserved ``"$acc"``)
    broadcasts into the step space and unbinds along the step axis.
    An operand outside that space raises through ``broadcast_to`` —
    a loud refusal of a site the fold cannot read.
    """
    import torch

    out: dict = {}
    for m in _spec_metas(emit):
        if m == "$acc":
            continue
        out[m] = (
            torch.broadcast_to(_as_tensor(env[m]), full)
            .movedim(ax, 0)
            .unbind(0)
        )
    return out


# ---------------------------------------------------------------------------
#  Elementwise expression enumeration — the factors of every template
# ---------------------------------------------------------------------------


def _atom_metas(a: Any) -> set:
    """Metavariable set of one atom (spec constants carry none)."""
    return {a} if isinstance(a, str) else set()


def _expr_metas(e: Any) -> set:
    """Metavariable set of a built expression tree."""
    return set(_spec_metas(e))


def _ew1_any(atoms: tuple) -> list:
    """One-op EW exprs over *atoms*, any coverage."""
    out: list[Any] = [
        (u, x) for u in _UNARY for x in atoms if isinstance(x, str)
    ]
    for b in _BINARY:
        out.extend((b, x, y) for x in atoms for y in atoms)
    return out


def _ew1(required: frozenset, atoms: tuple) -> list:
    """One-op EW exprs over *atoms* covering exactly *required*."""
    out: list[Any] = [
        (u, x)
        for u in _UNARY
        for x in atoms
        if _atom_metas(x) == required
    ]
    for b in _BINARY:
        out.extend(
            (b, x, y)
            for x in atoms
            for y in atoms
            if _atom_metas(x) | _atom_metas(y) == required
        )
    return out


def _ew2(required: frozenset, atoms: tuple) -> list:
    """Two-op EW exprs over *atoms* covering exactly *required*."""
    out: list[Any] = [
        (u, e) for u in _UNARY for e in _ew1(required, atoms)
    ]
    anys = _ew1_any(atoms)
    for b in _BINARY:
        for x in atoms:
            mx = _atom_metas(x)
            for e in anys:
                if mx | _expr_metas(e) == required:
                    out.append((b, x, e))
                    out.append((b, e, x))
    return out


def _ew_exprs(required: frozenset, atoms: tuple, n: int) -> list:
    """EW exprs with exactly *n* op nodes covering *required*.

    *atoms* is the leaf pool — the covered metas plus the spec's
    constants, which join any expr for free.  ``n`` tops out at
    ``_MAX_EW_OPS``; deeper trees are the grammar's declared
    vocabulary gap, not a hidden search.
    """
    if n == 0:
        return [a for a in atoms if _atom_metas(a) == required]
    if n == 1:
        return _ew1(required, atoms)
    if n == 2:
        return _ew2(required, atoms)
    return []


def _spec_consts(spec: Any) -> list:
    """Ordered distinct constant leaves inside a spec."""
    out: list = []

    def walk(s: Any) -> None:
        if isinstance(s, str):
            return
        if isinstance(s, (tuple, list)):
            for e in s[1:]:
                if not isinstance(e, dict):
                    walk(e)
            return
        if isinstance(s, bool):
            return
        if s not in out:
            out.append(s)

    walk(spec)
    return out


def _spec_dims(spec: Any) -> list:
    """Ordered distinct ``dim`` axes declared as attrs inside a spec.

    The reduction family reads the spec's own declared axis off the
    signature: a corpus site spells ``dim=-1`` or ``dim=1``, and the
    reduction descriptor must carry that literal.  A ``dim`` given as
    a sequence contributes each entry.
    """
    out: list = []

    def add(d: Any) -> None:
        if isinstance(d, bool):
            return
        if isinstance(d, int):
            if d not in out:
                out.append(d)
        elif isinstance(d, (tuple, list)):
            for e in d:
                add(e)

    def walk(s: Any) -> None:
        if isinstance(s, (tuple, list)):
            for e in s[1:]:
                if isinstance(e, dict):
                    add(e.get("dim"))
                else:
                    walk(e)

    walk(spec)
    return out


def _spec_keepdim(spec: Any) -> bool:
    """Whether the spec's reductions declare ``keepdim``.

    A normalization's denominator must broadcast back against the
    operand, which requires the reduced axis to survive — so the
    normalization's ``keepdim`` is *read off the spec*, not
    enumerated.  Enumerating both values would propose the
    non-``keepdim`` form against a ``keepdim`` spec (and vice versa):
    dead space, since a non-``keepdim`` reduction only broadcasts
    back on a coincidental shape (measured: 141 of the first 600
    ``cumsum`` candidates raised, all non-``keepdim`` norm bodies).
    """
    found = False

    def walk(s: Any) -> None:
        nonlocal found
        if isinstance(s, (tuple, list)):
            for e in s[1:]:
                if isinstance(e, dict) and e.get("keepdim"):
                    found = True
                elif not isinstance(e, dict):
                    walk(e)

    walk(spec)
    return found


# ---------------------------------------------------------------------------
#  Proposal enumeration — candidate bodies, simplest first
# ---------------------------------------------------------------------------


def _round_robin(gens: Any) -> Any:
    """Yield from *gens* in turn until all are exhausted.

    Fair interleave across columns: no partition's deep region can
    starve another partition's cheap candidates behind it.
    """
    live = [iter(g) for g in gens]
    while live:
        nxt = []
        for it in live:
            try:
                yield next(it)
            except StopIteration:
                continue
            nxt.append(it)
        live = nxt


def _fold_states(metas: list) -> list:
    """Return the carries a fold may start from: metas plus neutrals.

    A spec's explicit carry (``H0``) is a metavar; a spec that spells
    its recurrence without one (``cumsum(X)``, the corpus's
    zero-init selective scan) starts from the scan's neutral element
    — ``FOLD_CARRIES``.  Both are proposed; the verifier decides.
    """
    return [*metas, *FOLD_CARRIES]


def _canonical_emit_seeds(metas: list) -> list:
    """Propose the canonical emitted-scan bodies first.

    ``acc_t = g(m_a) ⊙ acc_{t-1} + m_b`` with the running
    accumulator as the per-step output (``out_t = $acc``) — the
    structural instance the ``fold_emit`` grammar exists to express,
    and the selective-scan total the atom-synth loop targets.  The
    round-robin column walk buried this form behind every column's
    non-accumulator projections (the canonical ``$acc`` emit sits
    past 200k proposals for a three-meta spec); seeding it ahead of
    the walk puts it within the first few hundred.

    The seed is structural, not a hard-coded answer: the gate ``g``
    ranges over the grammar's unary vocabulary and the plain
    identity, ``m_a``/``m_b`` over the non-state metas, and the state
    over the metas plus the neutral carries (``FOLD_CARRIES``) — so a
    spec with no explicit carry meta (``cumsum(X)``, ``cumprod(X)``,
    the corpus's zero-init selective scan) still reaches its
    canonical form, and the verifier decides which instance (if any)
    is the atom.  It is bounded to the token axis ``FOLD_AXES[0]`` —
    the corpus's ``(B, T, C)`` middle axis, where the emitted scan
    lives — so the seed stays a few hundred candidates rather than
    scaling with the meta count and crowding out the other families'
    budget.

    The neutral carries and identity sides are the *small-spec*
    reach: for a carry-less spec (one or two metas) they are the only
    way to spell the recurrence, and the ``fold_emit`` *family* is
    gated at ``>= 3`` metas so nothing else reaches it.  For three or
    more metas the ``fold_emit`` family itself carries the neutral
    carries (:func:`_fold_states`) and the identity sides
    (:func:`_side_exprs`), so the seed keeps the narrower meta-only
    vocabulary — otherwise the extra states/sides multiply a
    five-meta seed from ~640 to ~1260 candidates and consume a
    default budget before ``fold_zip`` is reached (measured: the
    ``applyd``-nest mint regressed to a refusal at ``budget=1500``).
    """
    small = len(metas) < 3
    states = _fold_states(metas) if small else list(metas)
    a_extra = (1.0,) if small else ()
    b_extra = (0.0,) if small else ()
    out = []
    for state in states:
        rest = tuple(m for m in metas if m != state)
        for a_side, b_side in _seed_pairs(rest, a_extra, b_extra):
            out.append(
                (
                    "fold_emit",
                    a_side,
                    b_side,
                    state,
                    "$acc",
                    FOLD_AXES[0],
                )
            )
    return out


def _seed_pairs(rest: tuple, a_extra: tuple, b_extra: tuple) -> list:
    """Return the seed's (a-side, b-side) pairs, minus dead ones.

    ``a_extra``/``b_extra`` are the small-spec identities (``1.0`` /
    ``0.0``).  The pair where *both* sides are the identity is a
    constant fold — no step axis, so ``_fold_eval`` raises on it: a
    structurally dead candidate, never proposed.
    """
    out = []
    for a_m in rest:
        for a_side in (a_m, *a_extra, *((u, a_m) for u in _UNARY)):
            for b_side in (*rest, *b_extra):
                if a_side == 1.0 and b_side == 0.0:
                    continue
                out.append((a_side, b_side))
    return out


def _contract_bodies(metas: list, consts: tuple, max_ops: int) -> Any:
    """Round-robin the contraction columns over ordered bipartitions."""
    parts = []
    for size in range(1, len(metas)):
        for lhs in combinations(metas, size):
            rhs = tuple(m for m in metas if m not in lhs)
            parts.append((frozenset(lhs), frozenset(rhs)))
    return _round_robin(
        _contract_column(lhs, rhs, consts, max_ops)
        for lhs, rhs in parts
    )


def _contract_column(
    lhs: frozenset, rhs: frozenset, consts: tuple, max_ops: int
) -> Any:
    """Contraction candidates for one ordered bipartition.

    Simplest-first by total op count: the descriptor
    ``einsum(eq, ew(lhs), ew(rhs))`` enumerates factor pairs shallow
    to deep, equations innermost.
    """
    la = tuple(sorted(lhs)) + consts
    ra = tuple(sorted(rhs)) + consts
    for total in range(2 * max_ops + 1):
        lo = max(0, total - max_ops)
        hi = min(max_ops, total)
        for i in range(lo, hi + 1):
            for e1 in _ew_exprs(lhs, la, i):
                for e2 in _ew_exprs(rhs, ra, total - i):
                    for eq in CONTRACT_EQS:
                        yield ("contract", eq, e1, e2)


def _side_exprs(
    required: frozenset, consts: tuple, n: int, identity: float
) -> list:
    """EW exprs for one fold side, covering exactly *required*.

    An empty side is the side's *identity* — ``1.0`` for the
    multiplier ``a``, ``0.0`` for the addend ``b`` — so a spec whose
    recurrence has no operand on one side (``cumsum(X)``: ``a ≡ 1``;
    ``cumprod(X)``: ``b ≡ 0``) is spellable.  The former ``1.0``/
    ``0.0`` fillers were structurally dead while ``_fold_eval``
    broadcast a bare float; ``_as_tensor`` (the scalar-operand lift)
    made them live again, so the identity is re-emitted — only at
    ``n == 0`` (the identity is a zero-op expr; a deeper expr over an
    empty set is the grammar's declared gap, not a hidden search).
    """
    if not required:
        return [identity] if n == 0 else []
    return _ew_exprs(required, tuple(sorted(required)) + consts, n)


def _fold_column(
    state: str,
    a_set: frozenset,
    b_set: frozenset,
    consts: tuple,
    axis: int,
    max_ops: int,
) -> Any:
    """``fold`` candidates for one (state, a-partition, axis) column."""
    for total in range(2 * max_ops + 1):
        lo = max(0, total - max_ops)
        for i in range(lo, min(max_ops, total) + 1):
            j = total - i
            for ea in _side_exprs(a_set, consts, i, 1.0):
                for eb in _side_exprs(b_set, consts, j, 0.0):
                    yield ("fold", ea, eb, state, axis)


def _fold_bodies(metas: list, consts: tuple, max_ops: int) -> Any:
    """Round-robin the fold columns: (state, a-partition, axis).

    The state ranges over the metas *and* the neutral carries
    (:func:`_fold_states`), so a spec with no explicit carry meta
    still reaches a foldable column.
    """
    gens = []
    for state in _fold_states(metas):
        rest = tuple(m for m in metas if m != state)
        for k in range(len(rest) + 1):
            for a_set in combinations(rest, k):
                b_set = tuple(m for m in rest if m not in a_set)
                for axis in FOLD_AXES:
                    gens.append(
                        _fold_column(
                            state,
                            frozenset(a_set),
                            frozenset(b_set),
                            consts,
                            axis,
                            max_ops,
                        )
                    )
    return _round_robin(gens)


def _emit_exprs(rest: tuple, consts: tuple) -> list:
    """Emit exprs for ``fold_emit`` — bounded per-step projections.

    The running accumulator rides the reserved ``"$acc"`` leaf, so
    every emit names it — the ``out_i = g(q_i, state_i)`` shape.
    Ordering is cheapest-first: a binary over one meta and ``$acc``
    (``q_i ⊙ acc_i`` is the canonical form), then the trivial emits,
    then non-accumulator factors up to ``_MAX_EMIT_OPS`` deep.  The
    depth is a real cap: an emit needing two non-acc ops
    (``add(mul(Q,$acc),R)``) is outside it (measured — no corpus
    site needs one; the emitted-scan corpus emit is ``$acc``).
    """
    out: list = []
    for b in _BINARY:
        for m in sorted(rest):
            out.append((b, m, "$acc"))
            out.append((b, "$acc", m))
    out.append("$acc")
    out.extend((u, "$acc") for u in _UNARY)
    for size in range(1, min(2, len(rest)) + 1):
        for s_e in combinations(sorted(rest), size):
            req = frozenset(s_e)
            atoms = tuple(sorted(req)) + consts
            for e in _ew_exprs(req, atoms, _MAX_EMIT_OPS):
                for b in _BINARY:
                    out.append((b, e, "$acc"))
                    out.append((b, "$acc", e))
    return out


def _fold_emit_column(
    state: str,
    a_set: frozenset,
    b_set: frozenset,
    emits: list,
    consts: tuple,
    axis: int,
) -> Any:
    """``fold_emit`` candidates for one (state, a-part, axis) column.

    *emits* is the column's pre-filtered projection list — emits
    covering the metas the a/b sides left uncovered.  A column whose
    a/b sides are both empty is skipped: the only non-empty sides are
    the identities (``1.0``/``0.0``), a constant fold with no step
    axis — a structurally dead candidate.
    """
    if not a_set and not b_set:
        return
    for emit in emits:
        for total in range(2 * _MAX_EW_OPS + 1):
            lo = max(0, total - _MAX_EW_OPS)
            for i in range(lo, min(_MAX_EW_OPS, total) + 1):
                for ea in _side_exprs(a_set, consts, i, 1.0):
                    for eb in _side_exprs(
                        b_set, consts, total - i, 0.0
                    ):
                        yield ("fold_emit", ea, eb, state, emit, axis)


def _fold_emit_cover_gens(
    state: str,
    rest: tuple,
    covered: tuple,
    emits: list,
    consts: tuple,
) -> list:
    """Columns for one a/b-covered subset: (a-part, axis) x emits.

    The column's emit list is pre-filtered to projections covering
    the metas the a/b sides left uncovered.
    """
    leftover = frozenset(rest) - frozenset(covered)
    col_emits = [e for e in emits if leftover <= _expr_metas(e)]
    gens = []
    for a_size in range(len(covered) + 1):
        for a_set in combinations(covered, a_size):
            b_set = tuple(m for m in covered if m not in a_set)
            for axis in FOLD_AXES:
                gens.append(
                    _fold_emit_column(
                        state,
                        frozenset(a_set),
                        frozenset(b_set),
                        col_emits,
                        consts,
                        axis,
                    )
                )
    return gens


def _fold_emit_state_gens(
    state: str, rest: tuple, consts: tuple
) -> list:
    """Columns for one state choice, shrinking a/b coverage first."""
    emits = _emit_exprs(rest, consts)
    gens = []
    for k in range(len(rest) + 1):
        for covered in combinations(rest, len(rest) - k):
            gens.extend(
                _fold_emit_cover_gens(
                    state, rest, covered, emits, consts
                )
            )
    return gens


def _fold_emit_bodies(metas: list, consts: tuple) -> Any:
    """Round-robin fold-emit columns: (state, cover, part, axis).

    The recurrence sides cover a subset of the non-state metas; the
    projection must name the rest — an emit-only operand (``q`` in
    ``q_i ⊙ acc_i``) is covered *by the emit*, never forced into
    ``a``/``b``.  A meta may also serve both roles here (the emit
    reads it while a recurrence side carries it) — the column's
    emit filter only requires covering the leftover.  Columns order
    by shrinking a/b coverage so full-recurrence candidates lead.
    """
    gens = []
    for state in _fold_states(metas):
        rest = tuple(m for m in metas if m != state)
        gens.extend(_fold_emit_state_gens(state, rest, consts))
    return _round_robin(gens)


def _pairings(items: tuple) -> Any:
    """All partitions of *items* into unordered pairs."""
    if not items:
        yield ()
        return
    first, rest = items[0], items[1:]
    for i in range(len(rest)):
        second = rest[i]
        others = rest[:i] + rest[i + 1 :]
        for tail in _pairings(others):
            yield ((first, second), *tail)


def _fold_zip_column(state: str, pairing: tuple) -> Any:
    """``fold_zip`` candidates for one (state, pairing) column.

    The chronological order of the pairs permutes and each pair
    flips — ``(a, b)`` reads ``acc = a ⊙ acc + b``, the
    ``aff_diag`` convention — so a two-pair column yields 2!·2²
    candidates.
    """
    for order in permutations(pairing):
        for flips in product((0, 1), repeat=len(pairing)):
            steps = tuple(
                p if not f else (p[1], p[0])
                for p, f in zip(order, flips, strict=True)
            )
            yield ("fold_zip", steps, state)


def _fold_zip_bodies(metas: list) -> Any:
    """Round-robin the flat-fold columns: (state, pairing)."""
    gens = []
    for state in metas:
        rest = tuple(m for m in metas if m != state)
        if len(rest) % 2 or not rest:
            continue
        for pairing in _pairings(rest):
            gens.append(_fold_zip_column(state, pairing))
    return _round_robin(gens)


def _reduce_column(op: str, subs: list, axis: int) -> Any:
    """``reduce`` candidates for one (op, axis) column."""
    for sub in subs:
        yield ("reduce", op, sub, axis)


def _reduce_bodies(metas: list, consts: tuple, axes: tuple) -> Any:
    """Round-robin reduce columns: (op, axis) x full-coverage subs.

    The reduction's operand must name every meta (the reduction is
    the spec's whole value, so nothing is left outside); the subs are
    the grammar's elementwise exprs over the full meta set, bounded
    to ``_MAX_EW_OPS``.  Ops come from :data:`REDUCE_OPS` — the
    dim-reductions the backend lowers — and axes from the spec's own
    ``dim`` attrs plus :data:`REDUCE_AXES`.
    """
    req = frozenset(metas)
    atoms = tuple(sorted(req)) + consts
    subs: list = []
    for n in range(_MAX_EW_OPS + 1):
        subs.extend(_ew_exprs(req, atoms, n))
    gens = [
        _reduce_column(op, subs, axis)
        for op in REDUCE_OPS
        for axis in axes
    ]
    return _round_robin(gens)


def _norm_column(
    op: str, red_op: str, subs: list, axis: int, keepdim: bool
) -> Any:
    """``norm`` candidates for one (op, red_op, axis, keepdim) column."""
    for lhs in subs:
        for rhs in subs:
            yield ("norm", op, lhs, red_op, rhs, axis, keepdim)


def _norm_bodies(
    metas: list, consts: tuple, axes: tuple, keepdim: bool
) -> Any:
    """Round-robin norm columns: (op, red_op, axis) x operand pairs.

    A normalization combines a value with a reduction of (another)
    value; both operands are the grammar's elementwise exprs over the
    full meta set, bounded to ``_MAX_EW_OPS``.  Ops come from
    :data:`NORM_OPS`, reductions from :data:`REDUCE_OPS`, axes from
    the spec's own ``dim`` attrs plus :data:`REDUCE_AXES`.  *keepdim*
    is read off the spec (:func:`_spec_keepdim`) — a property of the
    reduction, not a search dimension.
    """
    req = frozenset(metas)
    atoms = tuple(sorted(req)) + consts
    subs: list = []
    for n in range(_MAX_EW_OPS + 1):
        subs.extend(_ew_exprs(req, atoms, n))
    gens = [
        _norm_column(op, red_op, subs, axis, keepdim)
        for op in NORM_OPS
        for red_op in REDUCE_OPS
        for axis in axes
    ]
    return _round_robin(gens)


def _top_op(spec: Any) -> Any:
    """Return the spec's outermost op name, or ``None`` for a leaf."""
    if isinstance(spec, (tuple, list)) and spec:
        return spec[0]
    return None


def _canonical_norm_seeds(
    metas: list, consts: tuple, axes: tuple, keepdim: bool
) -> list:
    """Propose the canonical normalization bodies first.

    ``op(ew, red_op(ew, axis, keepdim))`` — a value normalized by a
    reduction of *itself*, the softmax / ``x - mean(x)`` shape.  Like
    the ``fold_emit`` seed it is structural, not a hard-coded answer:
    the operand ``ew`` ranges over the grammar's one-op vocabulary and
    the plain meta, ``op``/``red_op``/``axis`` over their
    vocabularies, so the verifier decides which instance (if any) is
    the atom.  The equal-operand seed keeps the plausible
    normalization within the first few hundred proposals — the full
    column walk interleaves the unequal-operand pairs ahead of it.

    Called only when the spec's top op is a :data:`NORM_OPS` op (the
    normalization shape): the seed is a few hundred candidates, so
    proposing it for a plain reduction would push that spec's mint
    from ~37 to ~277 (measured).
    """
    req = frozenset(metas)
    atoms = tuple(sorted(req)) + consts
    ews: list = []
    for n in range(min(1, _MAX_EW_OPS) + 1):
        ews.extend(_ew_exprs(req, atoms, n))
    out = []
    for op in NORM_OPS:
        for red_op in REDUCE_OPS:
            for axis in axes:
                for ew in ews:
                    out.append(
                        ("norm", op, ew, red_op, ew, axis, keepdim)
                    )
    return out


def _admitted_families(n: int) -> set:
    """Return the families whose meta gate admits an *n*-meta spec.

    The fold families need ``>= 3`` metas (the ``state + a + b``
    count the full walk's columns need); a smaller spec's canonical
    recurrence is reached through the ``fold_emit`` seed
    (:func:`_canonical_emit_seeds`) instead, so the floor is not a
    reach limit.  The contraction needs two factors; ``fold_zip`` an
    even non-state count; the ``reduce``/``norm`` families any
    non-empty meta set.
    """
    out = {"reduce", "norm"} if n else set()
    if n >= 2:
        out.add("contract")
    if n >= 3:
        out.update(("fold", "fold_emit"))
    if not (n - 1) % 2:
        out.add("fold_zip")
    return out


def _family_gens(
    fams: tuple,
    metas: list,
    consts: tuple,
    max_ops: int,
    axes: tuple,
    keepdim: bool,
) -> list:
    """One candidate generator per admitted template family.

    The fold families keep their ``>= 3`` meta floor (the
    ``state + a + b`` count the walk's full columns need): a
    one- or two-meta spec's canonical recurrence is reached through
    the ``fold_emit`` seed (:func:`_canonical_emit_seeds`), not the
    exhaustive walk, so relaxing the floor here only explodes the
    candidate space without enabling a mint — measured: with the
    floor relaxed to any non-empty meta set, ``cumsum(X)``'s
    ``fold_emit`` column grows from 36 (the seed) to 147700
    candidates and a two-meta spec's from 144 to >200000, while every
    spec in the battery mints the *same* body at the *same* budget
    index (the seed leads).  The floor stays; the seed carries the
    small-spec reach.  The ``reduce``/``norm`` families admit any
    non-empty meta set — a one-meta reduction (``sum(X, dim)``) is
    their whole point.  The gates live in :func:`_admitted_families`.
    """
    adm = set(fams) & _admitted_families(len(metas))
    gens = []
    if "contract" in adm:
        gens.append(_contract_bodies(metas, consts, max_ops))
    if "fold" in adm:
        gens.append(_fold_bodies(metas, consts, max_ops))
    if "fold_emit" in adm:
        gens.append(_fold_emit_bodies(metas, consts))
    if "fold_zip" in adm:
        gens.append(_fold_zip_bodies(metas))
    if "reduce" in adm:
        gens.append(_reduce_bodies(metas, consts, axes))
    if "norm" in adm:
        gens.append(_norm_bodies(metas, consts, axes, keepdim))
    return gens


def propose_bodies(
    spec: Any,
    *,
    families: tuple | None = None,
    max_ops: int = _MAX_EW_OPS,
) -> Any:
    """Enumerate candidate handler bodies for *spec*, lazily.

    Deterministic and simplest-first: the ``ew`` replay (elementwise
    specs only — the ``genkernel`` baseline) leads, then the
    ``fold_emit`` canonical emitted-scan forms (:func:`_canonical_emit_seeds`
    — the ordering gap the round-robin walk left) and the ``norm``
    canonical normalization forms (:func:`_canonical_norm_seeds`),
    then every admitted family's columns interleave round-robin.
    *families* restricts the template vocabulary; *max_ops* bounds
    each elementwise factor's depth (``_MAX_EW_OPS`` tops the grammar).
    """
    fams = families if families is not None else _FAMILIES
    metas = _spec_metas(spec)
    consts = tuple(_spec_consts(spec))
    axes = tuple(dict.fromkeys([*_spec_dims(spec), *REDUCE_AXES]))
    keepdim = _spec_keepdim(spec)
    # A normalization is only proposed for a normalization-shaped spec
    # (top op in NORM_OPS): its candidates on any other spec are dead
    # — a non-``keepdim`` reduction of a scan operand cannot broadcast
    # back (measured: 141 of a scan spec's first 600 candidates
    # raised, all norm bodies).
    if _top_op(spec) not in NORM_OPS:
        fams = tuple(f for f in fams if f != "norm")
    if "ew" in fams and elementwise_spec(spec):
        yield ("ew", spec)
    if "fold_emit" in fams and metas:
        yield from _canonical_emit_seeds(metas)
    if "norm" in fams and metas:
        yield from _canonical_norm_seeds(metas, consts, axes, keepdim)
    gens = _family_gens(fams, metas, consts, max_ops, axes, keepdim)
    yield from _round_robin(gens)


# ---------------------------------------------------------------------------
#  Verification — the same bar as a law, on the measured domain
# ---------------------------------------------------------------------------


class VerifyResult(NamedTuple):
    """The verify-or-refuse verdict for one candidate body.

    ``max_err`` is the measured worst absolute deviation over every
    sample taken — inf when the body raised or the shapes could not
    agree; ``n_samples`` how many draws ran before the verdict;
    ``detail`` names the refusal when ``ok`` is false.
    """

    ok: bool
    max_err: float = float("inf")
    n_samples: int = 0
    detail: str = ""


def _sample_env(shapes: dict, seed: int) -> dict:
    """Draw fp64 operands for one domain entry under *seed*."""
    import torch

    g = torch.Generator().manual_seed(seed)
    return {
        m: torch.randn(tuple(sh), dtype=torch.float64, generator=g)
        for m, sh in shapes.items()
    }


def _domain_list(domain: Any) -> list:
    """Normalise a domain: one ``{meta: shape}`` dict or a list."""
    if isinstance(domain, dict):
        return [domain]
    return list(domain)


def _domain_draws(
    spec: Any,
    domain: Any,
    seed: int,
    bindings: dict,
    adversarial: int,
) -> tuple:
    """Sample the domain once; return ``(envs, spelled refs)``.

    One draw per domain entry plus *adversarial* re-draws under an
    offset seed — the paired comparison every candidate sees.
    """
    envs: list = []
    refs: list = []
    for i, shapes in enumerate(_domain_list(domain)):
        for sweep in range(1 + adversarial):
            env = _sample_env(shapes, seed + i + sweep * (1 << 20))
            envs.append(env)
            refs.append(_eval_spec(spec, env, bindings))
    return envs, refs


def _verify_envs(
    body: Any,
    envs: list,
    refs: list,
    bindings: dict,
    rtol: float,
    atol: float,
) -> VerifyResult:
    """Run the numeric gate over pre-drawn ``(env, ref)`` pairs.

    A shape mismatch, an eval error, or a numeric miss is a refusal
    carrying the measured ``max_err``; a body that raises fails
    loudly — the ``inf`` verdict is a *louder* refusal than a
    numeric one, never a quieter pass.
    """
    import torch

    max_err = 0.0
    n = 0
    for env, ref in zip(envs, refs, strict=True):
        try:
            got = eval_body(body, env, bindings)
        except Exception as exc:
            return VerifyResult(
                False,
                float("inf"),
                n,
                f"body raised: {type(exc).__name__}: {exc}",
            )
        if not isinstance(got, torch.Tensor) or got.shape != ref.shape:
            return VerifyResult(
                False, float("inf"), n, "shape mismatch"
            )
        err = float((got - ref).abs().max()) if ref.numel() else 0.0
        max_err = max(max_err, err)
        n += 1
        if not torch.allclose(got, ref, rtol=rtol, atol=atol):
            return VerifyResult(False, max_err, n, "numeric miss")
    return VerifyResult(True, max_err, n)


def verify_body(
    body: Any,
    spec: Any,
    domain: Any,
    *,
    seed: int = 0,
    bindings: dict | None = None,
    rtol: float = _VERIFY_RTOL,
    atol: float = _VERIFY_ATOL,
    adversarial: int = 1,
) -> VerifyResult:
    """Check *body* reproduces *spec*'s spelled eval on *domain*.

    *domain* is the measured bindings — a ``{meta: shape}`` dict or
    a list of them.  Every entry draws fp64 operands from a seeded
    generator — the same draw for every candidate, a paired
    comparison — evaluates the spelled spec through the ambient
    bindings and the descriptor through :func:`eval_body`, and
    requires ``allclose`` at the fp64 gate on all of them, plus
    *adversarial* re-draws per entry under an offset seed (the
    gauntlet's sweep: a candidate that fits its own draw but not a
    fresh one refuses here).
    """
    if bindings is None:
        import catopt_torch.torch_bridge as tb

        bindings = tb._IR_TO_TORCH
    envs, refs = _domain_draws(
        spec, domain, seed, bindings, adversarial
    )
    return _verify_envs(body, envs, refs, bindings, rtol, atol)


# ---------------------------------------------------------------------------
#  Minting — verified bodies become handler-table atoms
# ---------------------------------------------------------------------------


def _item_spec_domain(item: Any, domain: Any) -> tuple:
    """Split a mint item into ``(spec, domain)``."""
    if isinstance(item, dict):
        return item["spec"], item.get("domain") or domain
    return item, domain


def _refusal_note(tried: int, budget: int) -> str:
    """Name the refusal: vocabulary gap, budget cap, or exhausted."""
    if tried == 0:
        return "spec outside the handler grammar (no candidates)"
    if tried >= budget:
        return f"no candidate verified within budget={budget}"
    return f"grammar exhausted: none of {tried} candidates verified"


def _provenance(
    body: Any,
    dom: list,
    res: VerifyResult,
    *,
    seed: int,
    budget: int,
    tried: int,
    rtol: float,
    atol: float,
) -> dict:
    """Assemble the minted atom's provenance — path + domain."""
    return {
        "origin": "minted",
        "grammar": _GRAMMAR_VERSION,
        "family": body[0],
        "seed": seed,
        "budget": budget,
        "tried": tried,
        "domain": [
            {m: list(s) for m, s in entry.items()} for entry in dom
        ],
        "n_samples": res.n_samples,
        "max_err": res.max_err,
        "rtol": rtol,
        "atol": atol,
    }


def mint_handlers(
    items: Any,
    *,
    domain: Any = None,
    families: tuple | None = None,
    covered: set | None = None,
    seed: int = 0,
    budget: int = _BUDGET,
    rtol: float = _VERIFY_RTOL,
    atol: float = _VERIFY_ATOL,
    adversarial: int = 1,
) -> tuple:
    """Mint verified handler entries for measured subterm specs.

    Each *item* is a spec or a ``{"spec": ..., "domain": ...}``
    record (*domain* fills records that don't carry one).  Candidates
    from :func:`propose_bodies` verify in enumeration order against
    the spelled spec on the measured domain — the domain draws and
    the spelled refs are computed once per spec, so every candidate
    faces the same paired comparison.  The first survivor mints
    ``minted_<i>`` carrying its ``handler_body`` descriptor and a
    provenance record.  Exhausting *budget* candidates — or the
    grammar — without a verify is a measured refusal, recorded
    with the best (lowest) error seen.

    Returns ``(handlers, report)``: the handler table plus one
    verdict row per item — the falsification data.
    """
    import catopt_torch.torch_bridge as tb

    from catopt_discovery.genkernel import _canon_taken

    bindings = tb._IR_TO_TORCH
    skip = covered if covered is not None else set()
    handlers: dict[str, dict] = {}
    report: list[dict] = []
    for item in items:
        spec, dom = _item_spec_domain(item, domain)
        dom = _domain_list(dom)
        row: dict = {"spec": spec}
        if _canon_taken(spec, handlers, skip):
            row["verdict"] = "covered"
            report.append(row)
            continue
        envs, refs = _domain_draws(
            spec, dom, seed, bindings, adversarial
        )
        verdict, tried, best = None, 0, float("inf")
        for body in propose_bodies(spec, families=families):
            if tried >= budget:
                break
            res = _verify_envs(body, envs, refs, bindings, rtol, atol)
            tried += 1
            best = min(best, res.max_err)
            if res.ok:
                verdict = (body, res)
                break
        if verdict is None:
            row.update(
                {
                    "verdict": "refused",
                    "tried": tried,
                    "best_err": best,
                    "note": _refusal_note(tried, budget),
                }
            )
            report.append(row)
            continue
        body, res = verdict
        tag = f"minted_{len(handlers)}"
        handlers[tag] = {
            "pattern": spec,
            "kernel": f"{tag}_k",
            "args": tuple(_spec_metas(spec)),
            "handler_body": body,
            "provenance": _provenance(
                body,
                dom,
                res,
                seed=seed,
                budget=budget,
                tried=tried,
                rtol=rtol,
                atol=atol,
            ),
        }
        row.update(
            {
                "verdict": "minted",
                "tag": tag,
                "family": body[0],
                "tried": tried,
                "max_err": res.max_err,
            }
        )
        report.append(row)
    return handlers, report


def minted_bindings(handlers: dict) -> dict:
    """Compile each minted handler's body into a torch binding.

    The delivered callable is the descriptor interpreter — the
    verified semantics, not a replay of the spelled site (the ``ew``
    family excepted, whose body IS the spelled site by definition).
    Exactly :func:`genkernel.gen_bindings`' contract: kernel name →
    callable over the handler's positional metavar args.
    """
    import catopt_torch.torch_bridge as tb

    bindings = tb._IR_TO_TORCH
    out: dict = {}
    for h in handlers.values():
        body = h["handler_body"]

        def fn(*args: Any, b=body, hh=h, **_attrs: Any) -> Any:
            env = dict(zip(hh["args"], args, strict=True))
            return eval_body(b, env, bindings)

        out[h["kernel"]] = fn
    return out


def minted_sink(handlers: dict) -> Any:
    """Build a ``TorchSink`` binding minted atom names to bodies.

    ``supported_ops`` reports the ambient table plus every minted
    kernel — the feasibility bound the board prices the claim
    against — exactly as :func:`genkernel.gen_sink` reports its
    elementwise claims.
    """
    import catopt_torch.torch_bridge as tb
    from catopt_core.ops import OpTable
    from catopt_torch.adapters import TorchSink

    bindings = dict(tb._IR_TO_TORCH)
    bindings.update(minted_bindings(handlers))
    table = OpTable.core().register(torch_bindings=bindings)
    return TorchSink(ops=table)


# ---------------------------------------------------------------------------
#  Serialisation — the minted atom is provenance data
# ---------------------------------------------------------------------------


def body_to_data(body: Any) -> Any:
    """Encode a body descriptor as JSON-safe data (tuples → lists)."""
    if isinstance(body, (tuple, list)):
        return [body_to_data(e) for e in body]
    if isinstance(body, dict):
        return {k: body_to_data(v) for k, v in body.items()}
    return body


def body_from_data(data: Any) -> Any:
    """Decode :func:`body_to_data` — lists fold back to tuples."""
    if isinstance(data, list):
        return tuple(body_from_data(e) for e in data)
    if isinstance(data, dict):
        return {k: body_from_data(v) for k, v in data.items()}
    return data


def handler_record(tag: str, h: dict) -> dict:
    """Serialise a minted handler as a provenance record.

    The record is the atom: grammar path (``handler_body``), verified
    domain and measured error ride along — :func:`handler_from_record`
    reconstructs a working handler entry with no live code.
    """
    return {
        "tag": tag,
        "kind": "handler",
        "pattern": body_to_data(h["pattern"]),
        "kernel": h["kernel"],
        "args": list(h["args"]),
        "handler_body": body_to_data(h["handler_body"]),
        "provenance": dict(h.get("provenance") or {}),
    }


def handler_from_record(rec: dict) -> dict:
    """Rebuild a handler-table entry from its record."""
    return {
        "pattern": body_from_data(rec["pattern"]),
        "kernel": rec["kernel"],
        "args": tuple(rec["args"]),
        "handler_body": body_from_data(rec["handler_body"]),
        "provenance": dict(rec.get("provenance") or {}),
    }


# ---------------------------------------------------------------------------
#  CLI — reproduce the measured battery
# ---------------------------------------------------------------------------


def _battery() -> list:
    """Return the canonical measurement cases — mints and refusals."""
    return [
        {
            "spec": ("matmul", ("mul", "S", "D"), "V"),
            "domain": [{"S": (6, 5), "D": (6, 5), "V": (5, 7)}],
        },
        {
            "spec": (
                "add",
                ("mul", "H", ("prod", "A", {"dim": 0})),
                (
                    "sum",
                    (
                        "mul",
                        "B",
                        (
                            "div",
                            ("prod", "A", {"dim": 0}),
                            ("cumprod", "A", {"dim": 0}),
                        ),
                    ),
                    {"dim": 0},
                ),
            ),
            "domain": [{"A": (5,), "B": (5,), "H": ()}],
        },
        {
            "spec": (
                "applyd",
                ("aff_diag", "A", "B"),
                ("applyd", ("aff_diag", "C", "D"), "H"),
            ),
            "domain": [
                {
                    "A": (4,),
                    "B": (4,),
                    "C": (4,),
                    "D": (4,),
                    "H": (4,),
                }
            ],
        },
        {"spec": ("sum", "X", {"dim": -1}), "domain": [{"X": (4, 4)}]},
        {
            "spec": (
                "div",
                ("exp", "X"),
                ("sum", ("exp", "X"), {"dim": -1, "keepdim": True}),
            ),
            "domain": [{"X": (4, 5)}],
        },
        # the residual refusal: a meta-less non-elementwise graph the
        # templates cannot shape (every meta-bearing spec now proposes)
        {
            "spec": ("applyd", ("aff_diag", 1.0, 2.0), 3.0),
            "domain": [{}],
        },
    ]


def main(argv: list | None = None) -> int:
    """Run the canonical mint battery; print one verdict row per spec."""
    import argparse
    import json

    ap = argparse.ArgumentParser(
        prog="catopt_discovery.handler_synth",
        description=(
            "ADR-0007 layer 2: propose handler bodies from the "
            "grammar, verify them fp64 against the pre-mint spec."
        ),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=_BUDGET)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    handlers, report = mint_handlers(
        _battery(), seed=args.seed, budget=args.budget
    )
    rows = [{k: v for k, v in r.items() if k != "spec"} for r in report]
    out = {
        "verdicts": rows,
        "minted": {
            tag: handler_record(tag, h) for tag, h in handlers.items()
        },
    }
    text = json.dumps(out, indent=2, default=str)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text)
    for r, row in zip(report, rows, strict=True):
        print(f"{r['spec']!r:.80} -> {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

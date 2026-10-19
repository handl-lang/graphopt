"""ADR-0008 prototype — a minimal combinator presentation of the IR.

This is the *bridge only* — enough substrate to run the ADR's
falsification gates 1 (round-trip on the corpus, Δ-expansion size) and
2 (can the shared-state carrier be spelled as a ``loop`` term).  It is
deliberately small: no laws, no e-graph coupling, no backend.  Terms
are morphisms of a symmetric monoidal category with diagonals; the
corpus DAGs compile to flat sequences of "context morphisms".

THE BASIS (as data — :data:`BASIS`)

    leaf(t)    I → T          a Var/Param/Const as a nullary generator
    arr(o,n,a) T^n → T        an IR ``Op`` as an n-ary arrow (attrs a)
    seq(f…)    composition    f ; g ; …  (left to right on the context)
    par(f…)    ⊗              n-ary monoidal product (assoc built in)
    id(n)      A^n → A^n      identity on a wire bundle
    dup(k)     A → A^⊗k       Δ — explicit fan-out (sharing made a
                              morphism, not a pointer convention)
    perm(π)    A^n → A^n      reindexing — *derived* structure, stands
                              for the composite of swap+assoc it
                              denotes; kept primitive here so terms
                              stay small enough to read
    drop       A → I          terminal map (dead wires; the ``loop``
                              output convention needs it)
    loop(s,n)  S⊗X^n → S⊗Y^n  bounded trace: ``s : S⊗X → S⊗Y`` folded
                              over ``n`` per-step inputs — the one
                              combinator the DAG bridge never mints
                              (the DAG has already erased recurrence;
                              ``loop`` must be *proposed*, not
                              recovered).  Gate 2's subject.

Presentation decision: every ``Op`` translates as ``arr`` of arity
``len(args)`` — including the IR's structural ops (``pack``,
``getitem``, ``split``, ``select``, …).  The single exception is a
``pack`` ROOT: ``pack(t0,…,tn)`` as the program output translates as
the bare output bundle (the tupling itself, ``⊗``), and expansion
re-mints the ``pack`` — the ADR's "pack becomes ⊗" read literally.
This keeps every corpus term translatable; what is *lost* (and the
probe reports) is that e.g. ``getitem`` presents as an opaque arrow
rather than a projection — a named gap, not a refusal.

Wire order convention: a morphism consumes a PREFIX of the ambient
wire bundle (in order) and prepends its outputs.  ``par(f0,…,fk)``
splits the consumed prefix contiguously — f0 takes the first
n_in(f0) wires, and so on.  ``perm(π)`` emits ``out[i] = in[π[i]]``.

Bridge construction (``dag_to_comb``): postorder the DAG; the context
bundle B starts as all leaf generators, then for each op node emit
``seq(mux, par(arr, id))`` where ``mux = seq(par(dup-or-id per wire),
perm)`` gathers that node's argument wires (duplicating a wire that is
consumed now but still live, duplicating multiply-used arguments) and
keeps the rest.  Sharing thus becomes exactly the set of ``dup``
nodes; the extracted term's fan-out profile is in :func:`morph_stats`.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mlopt_core.ir import Const, Op, Param, Var

__all__ = [
    "BASIS",
    "Arr",
    "CombStats",
    "Drop",
    "Dup",
    "Id",
    "Leaf",
    "Loop",
    "Par",
    "Perm",
    "Seq",
    "arity",
    "comb_to_dag",
    "dag_size",
    "dag_to_comb",
    "morph_stats",
    "retnet_fold_program",
    "retnet_step",
]


@dataclass(frozen=True)
class BasisEntry:
    """One combinator of the basis — the authored surface, as data."""

    name: str
    sig: str
    note: str


#: The minimal basis.  ``perm`` is documented as derived (a composite
#: of swap + n-ary par); it stays primitive here so gate-1 terms are
#: readable and sized honestly.
BASIS: tuple[BasisEntry, ...] = (
    BasisEntry("leaf", "I → T", "Var/Param/Const — nullary generator"),
    BasisEntry("arr", "T^n → T", "an IR Op as an arrow, attrs ride"),
    BasisEntry("seq", "—", "n-ary composition, left to right"),
    BasisEntry("par", "—", "n-ary monoidal product (assoc built in)"),
    BasisEntry("id", "A^n → A^n", "identity on a wire bundle"),
    BasisEntry("dup", "A → A^⊗k", "Δ — explicit fan-out"),
    BasisEntry(
        "perm",
        "A^n → A^n",
        "rewiring — derived: a composite of swap + par, primitive here",
    ),
    BasisEntry("drop", "A → I", "terminal map — dead wires"),
    BasisEntry(
        "loop",
        "(S⊗X→S⊗Y) → S⊗X^n → S⊗Y^n",
        "bounded trace/scan — the extension gate 2 needs; fixed state "
        "shape, statically known iteration count",
    ),
)


# ---------------------------------------------------------------------------
#  Morphism terms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Leaf:
    """A nullary generator ``I → T`` wrapping an IR leaf term."""

    term: Any  # Var | Param | Const


@dataclass(frozen=True)
class Arr:
    """An IR op as an arrow ``T^n → T``; ``attrs`` are ``(k, v)`` pairs."""

    op: str
    n: int
    attrs: tuple = ()


@dataclass(frozen=True)
class Id:
    """Identity on ``n`` wires; ``Id(0)`` is the unit object map."""

    n: int = 1


@dataclass(frozen=True)
class Dup:
    """Diagonal ``Δ_k : A → A^⊗k`` — the sharing morphism."""

    k: int


@dataclass(frozen=True)
class Perm:
    """Reindex ``n`` wires: ``out[i] = in[order[i]]``."""

    order: tuple


@dataclass(frozen=True)
class Drop:
    """Terminal map ``A → I`` — discard a wire."""


@dataclass(frozen=True)
class Seq:
    """Composition ``f0 ; f1 ; …`` — each part maps the whole context."""

    parts: tuple


@dataclass(frozen=True)
class Par:
    """Monoidal product ``f0 ⊗ … ⊗ fk`` over contiguous wire segments."""

    parts: tuple


@dataclass(frozen=True)
class Loop:
    """Bounded trace: fold ``step : S⊗X^nx → S⊗Y^ny`` over ``n`` steps.

    ``n_in = 1 + n·nx``, ``n_out = 1 + n·ny`` — the first wire is the
    initial state; outputs are the final state followed by the
    per-step outputs in order.  State shape must be constant per term
    (the ADR's checkable contract); nothing here checks it — the
    prototype trusts the speller.
    """

    step: Any
    n: int
    nx: int = 1
    ny: int = 1


def arity(m: Any) -> tuple[int, int]:
    """Return ``(n_in, n_out)`` of a morphism term."""
    if isinstance(m, Leaf):
        return (0, 1)
    if isinstance(m, Arr):
        return (m.n, 1)
    if isinstance(m, Id):
        return (m.n, m.n)
    if isinstance(m, Dup):
        return (1, m.k)
    if isinstance(m, Perm):
        return (len(m.order), len(m.order))
    if isinstance(m, Drop):
        return (1, 0)
    if isinstance(m, Seq):
        if not m.parts:
            raise ValueError("empty seq")
        # parts act on the WHOLE current bundle: part i consumes a
        # prefix and prepends its outputs.  The footprint is the
        # minimum initial bundle that never starves a part.
        demand = 0
        delta = 0
        for p in m.parts:
            i, o = arity(p)
            demand = max(demand, i - delta)
            delta += o - i
        return (demand, demand + delta)
    if isinstance(m, Par):
        ins = outs = 0
        for p in m.parts:
            i, o = arity(p)
            ins += i
            outs += o
        return (ins, outs)
    if isinstance(m, Loop):
        return (1 + m.n * m.nx, 1 + m.n * m.ny)
    raise TypeError(f"not a morphism: {m!r}")


# ---------------------------------------------------------------------------
#  Smart constructors — keep compiled terms small enough to read
# ---------------------------------------------------------------------------


def _seq(*parts: Any) -> Any:
    flat: list = []
    for p in parts:
        if isinstance(p, Seq):
            flat.extend(p.parts)
        else:
            flat.append(p)
    if len(flat) == 1:
        return flat[0]
    return Seq(tuple(flat))


def _par(*parts: Any) -> Any:
    flat: list = []
    for p in parts:
        if isinstance(p, Par):
            flat.extend(p.parts)
        elif isinstance(p, Id) and p.n == 0:
            continue  # the unit object contributes no wires
        else:
            flat.append(p)
    if not flat:
        return Id(0)
    if all(isinstance(p, Id) for p in flat):
        return Id(sum(p.n for p in flat))
    if len(flat) == 1:
        return flat[0]
    return Par(tuple(flat))


def _perm(order: Sequence[int]) -> Any:
    o = tuple(order)
    if o == tuple(range(len(o))):
        return Id(len(o))
    return Perm(o)


def _mux(maps: list, order: list[int]) -> Any:
    """Combine per-wire maps and a permutation into a feed."""
    par = _par(*maps)
    prm = _perm(order)
    parts = [p for p in (par, prm) if not isinstance(p, Id)]
    if not parts:
        return Id(len(maps))
    return _seq(*parts)


def _arr(node: Op) -> Arr:
    """Wrap an op node as an arrow — attrs copied verbatim."""
    return Arr(node.op, len(node.args), tuple(node.attrs.items()))


# ---------------------------------------------------------------------------
#  dag → combinator
# ---------------------------------------------------------------------------


def _postorder(root: Any) -> list[Op]:
    """Deduped postorder of the op nodes in ``root``'s DAG."""
    order: list[Op] = []
    expanded: set = set()
    work: list[tuple[Any, bool]] = [(root, False)]
    while work:
        t, done = work.pop()
        if not isinstance(t, Op):
            continue
        if done:
            order.append(t)
            continue
        if t in expanded:
            continue
        expanded.add(t)
        work.append((t, True))
        work.extend((a, False) for a in t.args)
    return order


def dag_to_comb(root: Any) -> Any:
    """Compile an IR term DAG to a morphism term.

    A ``pack`` root translates as the output bundle itself (``⊗``, not
    an ``arr``) — see the module docstring; expansion re-mints it.
    Returns a morphism ``I → outputs``.  Raises ``TypeError`` on an
    argument class the basis cannot represent (a named gap).
    """
    order = _postorder(root)
    is_pack = isinstance(root, Op) and root.op == "pack"
    nodes = order[:-1] if is_pack else order
    outputs = list(root.args) if is_pack else [root]
    for n in nodes:
        for a in n.args:
            if not isinstance(a, (Op, Var, Param, Const)):
                raise TypeError(
                    f"{n.op}: arg of type {type(a).__name__} has no "
                    "leaf/generator presentation"
                )
    leaves: list[Any] = []
    leaf_seen: set = set()
    for n in nodes:
        for a in n.args:
            if not isinstance(a, Op) and a not in leaf_seen:
                leaf_seen.add(a)
                leaves.append(a)
    for a in outputs:
        if not isinstance(a, Op) and a not in leaf_seen:
            leaf_seen.add(a)
            leaves.append(a)

    total: Counter = Counter()
    for n in nodes:
        total.update(n.args)
    total.update(outputs)
    remaining = Counter(total)

    ctx = list(leaves)
    morphs: list = [_par(*(Leaf(lf) for lf in ctx))]
    for node in nodes:
        args = list(node.args)
        use = Counter(args)
        maps: list = []
        slots: list[tuple[Any, str]] = []
        for w in ctx:
            u = use.get(w, 0)
            rem_after = remaining[w] - u
            if u and rem_after:
                maps.append(Dup(u + 1))
            elif u > 1:
                maps.append(Dup(u))
            else:
                maps.append(Id(1))
            slots.extend((w, "arg") for _ in range(u))
            if rem_after or u == 0:
                slots.append((w, "live"))
        for w in args:
            remaining[w] -= 1
        # post-perm order: arg copies in arg order, then live wires
        arg_pos: dict[Any, deque] = {}
        for i, (w, role) in enumerate(slots):
            if role == "arg":
                arg_pos.setdefault(w, deque()).append(i)
        order_pi = [arg_pos[w].popleft() for w in args]
        order_pi += [i for i, (_, r) in enumerate(slots) if r == "live"]
        live = [w for w, r in slots if r == "live"]
        feed = _mux(maps, order_pi)
        step_parts = ([] if isinstance(feed, Id) else [feed]) + [
            _par(_arr(node), Id(len(live)))
        ]
        morphs.append(_seq(*step_parts))
        ctx = [node, *live]
    # closing: arrange the surviving context into the output bundle
    out_use = Counter(outputs)
    maps = []
    slots = []
    for w in ctx:
        u = out_use.get(w, 0)
        if u == 0:
            maps.append(Drop())
            continue
        maps.append(Dup(u) if u > 1 else Id(1))
        slots.extend((w, "out") for _ in range(u))
    pos: dict[Any, deque] = {}
    for i, (w, _r) in enumerate(slots):
        pos.setdefault(w, deque()).append(i)
    order_pi = [pos[w].popleft() for w in outputs]
    close = _mux(maps, order_pi)
    if not (isinstance(close, Id) and close.n == len(ctx)):
        morphs.append(close)
    return _seq(*morphs)


# ---------------------------------------------------------------------------
#  combinator → dag  (expansion — Δ becomes shared children)
# ---------------------------------------------------------------------------


def _apply(m: Any, wires: list) -> list:
    """Apply ``m`` to a prefix of ``wires``; return out-wires + tail."""
    if isinstance(m, Leaf):
        return [m.term, *wires]
    if isinstance(m, Arr):
        if len(wires) < m.n:
            raise ValueError(f"{m.op}: needs {m.n} wires")
        node = Op.make(m.op, *wires[: m.n], **dict(m.attrs))
        return [node, *wires[m.n :]]
    if isinstance(m, Id):
        return list(wires)
    if isinstance(m, Dup):
        return [wires[0]] * m.k + list(wires[1:])
    if isinstance(m, Perm):
        n = len(m.order)
        return [wires[i] for i in m.order] + list(wires[n:])
    if isinstance(m, Drop):
        return list(wires[1:])
    if isinstance(m, Seq):
        for p in m.parts:
            wires = _apply(p, wires)
        return list(wires)
    if isinstance(m, Par):
        cur = 0
        out: list = []
        for p in m.parts:
            i, _o = arity(p)
            out.extend(_apply(p, wires[cur : cur + i]))
            cur += i
        return out + list(wires[cur:])
    if isinstance(m, Loop):
        state = wires[0]
        pos = 1
        outs: list = []
        for _ in range(m.n):
            seg = [state, *wires[pos : pos + m.nx]]
            pos += m.nx
            res = _apply(m.step, seg)
            state = res[0]
            outs.extend(res[1:])
        return [state, *outs, *wires[pos:]]
    raise TypeError(f"not a morphism: {m!r}")


def comb_to_dag(m: Any) -> Any:
    """Expand a closed morphism ``I → T^out`` back to an IR term.

    A single output wire returns its term; multiple outputs re-mint
    the ``pack`` the translation dissolved.  ``dup`` produces shared
    children — the same hash-consed object at every use site, i.e.
    the DAG's sharing restored from a *morphism*.
    """
    wires = _apply(m, [])
    if not wires:
        raise ValueError("program produced no output wires")
    if len(wires) == 1:
        return wires[0]
    return Op.make("pack", *wires)


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------


@dataclass
class CombStats:
    """Size census of a morphism term — the Δ-blowup measurement."""

    nodes: int = 0  # total morphism nodes
    arr: int = 0
    leaf: int = 0
    dup: int = 0
    dup_copies: int = 0  # sum of k over Dup nodes
    perm: int = 0
    perm_wires: int = 0  # sum of perm widths
    drop: int = 0
    loop: int = 0
    structural: int = 0  # seq/par/id nodes


def morph_stats(m: Any, st: CombStats | None = None) -> CombStats:
    """Count morphism-node census of *m* (recursive over children)."""
    st = CombStats() if st is None else st
    st.nodes += 1
    if isinstance(m, Arr):
        st.arr += 1
    elif isinstance(m, Leaf):
        st.leaf += 1
    elif isinstance(m, Dup):
        st.dup += 1
        st.dup_copies += m.k
    elif isinstance(m, Perm):
        st.perm += 1
        st.perm_wires += len(m.order)
    elif isinstance(m, Drop):
        st.drop += 1
    elif isinstance(m, Loop):
        st.loop += 1
        morph_stats(m.step, st)
    elif isinstance(m, (Seq, Par, Id)):
        st.structural += 1
        if not isinstance(m, Id):
            for p in m.parts:
                morph_stats(p, st)
    return st


def dag_size(root: Any) -> int:
    """Count distinct subterms of a DAG (ops + leaves)."""
    seen: set = set()
    stack = [root]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        if isinstance(t, Op):
            stack.extend(t.args)
    return len(seen)


# ---------------------------------------------------------------------------
#  Gate 2 — the RetNet shared-state fold spelled as ``loop (arr step)``
#
#  step : S ⊗ (q,k,v) → S' ⊗ o
#      S' = add(mul(gammaᶜ, S), B)      B = ((k ⊙ ηᵤ)ᵀ) v   η = decay_vec(gamma,C,-1)
#      o  = add(intra, cross)      intra = om_decayed(q̃ kᵀ, D, v)
#                                  cross = ξᵤ ⊙ (q̃ S)   q̃ = q/√T
#                                  ξ = decay_vec(gamma,C,offset=C),
#                                  η = decay_vec(gamma,C,sign=-1),
#                                  D = decay(gamma, C)
#  — exactly the ``add(mul(g,S),B)`` spelling extraction prefers (the
#  ``applyd(aff_diag(a,B),S)`` carrier is a cost-neutral respelling).
# ---------------------------------------------------------------------------


def retnet_step(g: Any, a: Any, c: float, chunk: int) -> Any:
    """Spell the shared-state step ``S ⊗ (q,k,v) → S' ⊗ o``.

    ``g``/``a`` are leaf terms for the decay rate gamma and the per-chunk
    decay gamma^C (Var/Param/Const); ``c`` is the score denominator √T and
    ``chunk`` the block size C.  Per-step constants (η, ξ, D, a) are
    nullary generators inside the step — the categorical "closed
    term" reading: a generator ``I → T`` is placeable anywhere.
    """
    c_c, C = Const(c), chunk
    eta_u = _seq(
        Leaf(g),
        Arr("decay_vec", 1, (("t", C), ("sign", -1))),
        Arr("unsqueeze", 1, (("dim", -1),)),
    )
    xi_u = _seq(
        Leaf(g),
        Arr("decay_vec", 1, (("t", C), ("offset", C))),
        Arr("unsqueeze", 1, (("dim", -1),)),
    )
    d_gen = _seq(Leaf(g), Arr("decay", 1, (("t", C),)))
    a_gen = Leaf(a)
    c_gen = Leaf(c_c)

    # update half: [S, k, v] → [S']
    k_eta_t = _seq(
        _par(Id(1), eta_u),
        Arr("mul", 2),
        Arr("transpose", 1, (("dim0", -2), ("dim1", -1))),
    )
    b_mk = _seq(_par(k_eta_t, Id(1)), Arr("matmul", 2))
    gs_mk = _seq(_par(a_gen, Id(1)), Arr("mul", 2))
    upd = _seq(
        Perm((1, 2, 0)),  # [S,k,v] → [k,v,S]
        _par(b_mk, gs_mk),  # [B, gammaS]
        Perm((1, 0)),  # → [gammaS, B]
        Arr("add", 2),
    )

    # output half: [S, q, k, v] → [o].  The score denominator √T
    # multiplies BOTH the intra block and every cross contribution —
    # fold it into q once per step (q̃ = q/√T).
    qtilde = _seq(_par(Id(1), c_gen), Arr("div", 2))
    kt = Arr("transpose", 1, (("dim0", -2), ("dim1", -1)))
    intra = _seq(
        _par(Id(1), kt),  # [q̃,k] → [q̃,kᵀ]
        Arr("matmul", 2),  # → [score]
        _par(Id(1), d_gen, Id(1)),  # [score,v] → [score,D,v]
        Arr("om_decayed", 3),
    )
    cross = _seq(
        Arr("matmul", 2),  # [q̃,S] → [q̃@S]
        _par(xi_u, Id(1)),  # → [ξu, qS]
        Arr("mul", 2),
    )
    out = _seq(
        _par(Id(1), qtilde, Id(2)),  # [S,q,k,v] → [S,q̃,k,v]
        _par(Id(1), Dup(2), Id(2)),  # → [S,q̃,q̃,k,v]
        Perm((1, 3, 4, 2, 0)),  # → [q̃,k,v | q̃,S]
        _par(intra, cross),  # [intra, cross]
        Arr("add", 2),
    )
    step = _seq(
        _par(Dup(2), Id(1), Dup(2), Dup(2)),  # [S,S,q,k,k,v,v]
        Perm((0, 3, 5, 1, 2, 4, 6)),  # → [S,k,v | S,q,k,v]
        _par(upd, out),
    )
    return step


def retnet_fold_program(
    x: Var,
    s0: Var,
    params: dict[str, Param],
    g: Any,
    a: Any,
    t: int,
    k: int,
    d: int,
) -> Any:
    """Assemble the whole chunked-retention fold as ONE morphism term.

    ``seq(pre, loop(step, K), post)`` — ``pre`` computes the q/k/v
    projections and per-block slices, ``loop`` runs the shared-state
    recurrence, ``post`` drops the final state and concats the block
    outputs.  ``I → (1 + 3K) → (1 + K) → 1``.
    """
    C = t // k
    sz = (C,) * k
    wq, bq = params["q_weight"], params["q_bias"]
    wk, bk = params["k_weight"], params["k_bias"]
    wv, bv = params["v_weight"], params["v_bias"]
    qf = Op.make("linear", x, wq, bq)
    kf = Op.make("linear", x, wk, bk)
    vf = Op.make("linear", x, wv, bv)

    def blk(m: Any, i: int) -> Op:
        return Op.make("split", m, sizes=sz, dim=-2, index=i)

    wires = [s0]
    for i in range(k):
        wires += [blk(qf, i), blk(kf, i), blk(vf, i)]
    pre = dag_to_comb(Op.make("pack", *wires))
    step = retnet_step(g, a, float(t) ** 0.5, C)
    post = _seq(_par(Drop(), Id(k)), Arr("concat", k, (("dim", -2),)))
    return _seq(pre, Loop(step, k, nx=3, ny=1), post)

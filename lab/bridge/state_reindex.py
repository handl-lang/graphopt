"""The state-reindex law family — the gate-3 residual's named close.

ADR-0008 gate 3, the last residual.  ``form_propose`` reaches two
``Loop(step,3)`` on the RetNet winner; a single ``Loop(step,6)`` is
blocked by ``state-chain-broken:block3->block4`` — a *value* mismatch,
not a reach gap: the 4-wide chain's terminal state is ``gamma^12·M_4``
(the fold carries ``S_i = gamma^{phi_i}·A_i`` normalised to the
previous own-partition chunk's start) while the 16-wide family's
initial state ``M_4`` is the same accumulation at ``phi = 0``.

The compensated arm (:func:`form_propose.compensated_loop_check`)
showed the uniform step is value-correct across the boundary *if* two
classes of terms exist: the steering pair ``decay_vec(t=1,
offset=-8)`` / ``decay_vec(t=4, sign=-1, offset=-12)`` at the
producing step, and the merges that land the steered state in ``M_4``'s
e-class.  Saturation never minted either — negative-offset decay
vectors do not exist in the graph, and no staged rule bisects a bare
``matmul`` along its contraction axis.  This module is the law family
that closes both, as data:

* :func:`matmul_contract_split` — the contraction-axis bisection:
  ``matmul(A,B) = Σ_i A[:,P_i] @ B[P_i,:]`` for a given partition
  ``P``.  On ``M_4``'s singleton class (a 16-wide
  ``matmul(transpose(mul(k,eta_16)), v)``) the ``(4,4,4,4)`` split
  mints the four 4-chunk contributions — and, through the
  split-naturality chain below, the boundary's negative-offset decay
  vectors.
* :data:`SPLIT_MUL_ALIGNED` / :data:`SPLIT_UNSQ` / the
  :data:`DV_SPLIT` set — the naturality chain that pushes the minted
  splits through ``transpose`` (shipped ``split_transpose``),
  ``mul`` and ``unsqueeze`` down to ``decay_vec``, whose block-split
  mints *signed* offsets: ``split(dv(g,T,-1), (4,4,4,4), i) =
  dv(g, 4, sign=-1, offset=-4i)`` — the ``dv(t=4,-1,-12)`` steering
  weight the arm names.
* :data:`DECAY_VEC_USQ_FACTOR` — the scalar factor
  ``dv(g,T,-1,O) = g^O ⊙ dv(g,T,-1)`` spelled in the wires'
  ``unsqueeze`` packaging; mints ``usq(dv(g,1,-8))``, the steering
  ``w'`` the arm names.
* :data:`DECAY_GATHER` / :data:`DECAY_ID` / :data:`STATE_ETA_SLIDE`
  — the scalar slide/merge laws: ``gamma^a·(gamma^b·X) =
  gamma^{a+b}·X``, ``gamma^0·X = X``, and ``gamma^a ⊙
  ((k⊙eta_o)ᵀv) = (k⊙eta_{o+a})ᵀv`` — which is what lets
  ``gamma^{-8}·(gamma^4·S + B)`` re-index into the absolute
  accumulation ``A_0:12`` the bisection spells as
  ``add(add(m_0,m_1),m_2)``.

DRIVER
    :func:`reindex_close` measures the boundary (the missing steering
    pair gives the fine width ``w``, the target state's contraction
    gives ``T``), runs the family once to mint the wires and the
    bisected state, then re-walks the compensated arm with a
    ``post_add`` saturation hook so the steered terms it mints get the
    merges immediately.  When every covered block verifies it emits
    one ``unroll(step,6)`` — folded to ``Loop(step,6)`` by
    ``PROPOSE_LOOP`` — and the assembled program's e-class check
    (``comb_to_dag`` landing in the winner's root class) is the
    gate-3 certificate.
"""

from __future__ import annotations

from typing import Any

from mlopt_carriers.om import SPLIT_TRANSPOSE
from mlopt_carriers.xcarrier import _xshape
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Op
from mlopt_core.laws import R as _R
from mlopt_core.laws.tensor import DISTRIBUTE_MUL_ADD
from mlopt_discovery import combinator as cb
from mlopt_discovery.combinator import comb_to_dag

from lab.bridge import form_propose as fp
from lab.bridge.index_abstract import block_outputs
from lab.bridge.loop_propose import (
    EXPAND_LOOP,
    PROPOSE_LOOP,
    loop_headed,
    saturate_comb,
    unroll,
)

__all__ = [
    "DECAY_GATHER",
    "DECAY_ID",
    "DECAY_VEC_USQ_FACTOR",
    "DV_CANON",
    "DV_SPLIT",
    "SPLIT_MUL_ALIGNED",
    "SPLIT_UNSQ",
    "STATE_ETA_SLIDE",
    "matmul_contract_split",
    "reindex_close",
    "state_reindex_laws",
]


def R(name: str, lhs, rhs, **kw) -> Rewrite:
    """Module-local constructor — reindex laws carry no extra tags."""
    return _R(name, lhs, rhs, **kw)


# ---------------------------------------------------------------------------
#  1.  The contraction-axis bisection
# ---------------------------------------------------------------------------


def _check_contract_split(sizes: tuple[int, ...], bound: dict) -> bool:
    """Require the contraction extents of ``a``/``b`` to equal ``sum(sizes)``.

    ``matmul(a, b)`` contracts ``a``'s last axis with ``b``'s ``-2``
    axis; the partition is exact only when it tiles that shared extent
    and the batch dims broadcast.
    """
    a, b = _xshape(bound.get("a")), _xshape(bound.get("b"))
    if not (
        isinstance(a, tuple)
        and isinstance(b, tuple)
        and len(a) >= 2
        and len(b) >= 2
    ):
        return False
    tot = sum(sizes)
    if a[-1] != tot or b[-2] != tot:
        return False
    from mlopt_core.typing import INVALID, broadcast

    return broadcast(a[:-2], b[:-2]) is not INVALID


def matmul_contract_split(sizes: tuple[int, ...]) -> Rewrite:
    """``matmul(A,B)`` → ``Σ_i A[:,P_i] @ B[P_i,:]`` — block-column sum.

    The contraction-axis bisection the boundary needs: no staged rule
    splits a bare ``matmul`` (only ``om_decayed`` has
    ``chunkify_keys``), so the 16-wide state class stays a singleton.
    The partition is a law *parameter* — like
    ``om_decayed_kway_blocks(k)`` — supplied by the boundary
    measurement (the fine family's chunk width tiling the coarse
    state's contraction), so the minted pieces land in exactly the
    other family's spellings.
    """
    sizes = tuple(int(s) for s in sizes)
    if len(sizes) < 2 or any(s <= 0 for s in sizes):
        raise ValueError(
            f"sizes must be >=2 positive ints, got {sizes}"
        )

    def check(bound: dict) -> bool:
        return _check_contract_split(sizes, bound)

    parts = [
        Op.make(
            "matmul",
            Op.make("split", "a", sizes=sizes, dim=-1, index=i),
            Op.make("split", "b", sizes=sizes, dim=-2, index=i),
        )
        for i in range(len(sizes))
    ]
    rhs = parts[0]
    for p in parts[1:]:
        rhs = Op.make("add", rhs, p)
    return R(
        f"matmul_contract_split_{'_'.join(map(str, sizes))}",
        Op.make("matmul", "a", "b"),
        rhs,
        check=check,
        law="a matmul decomposes over any partition of its "
        "contraction axis: A@B = sum_i A[:,P_i]@B[P_i,:] — exact "
        "bilinearity, the block-sum spelling the boundary's "
        "reindexed state needs.",
    )


# ---------------------------------------------------------------------------
#  2.  The split-naturality chain: mul, unsqueeze, decay_vec
# ---------------------------------------------------------------------------


def _check_split_mul_aligned(bound: dict) -> bool:
    """Both operands must cover the split axis at ``sum(SZ)``.

    Right-aligned to the mul's output grid, the split axis ``D`` is a
    real axis in both operands only when each extent there is the
    partition total — then ``split(mul(a,b)) = mul(split(a),split(b))``
    is exact elementwise.  A broadcast-1 or absent axis declines (the
    scalar/broadcast variant is ``split_scalar_*``'s domain).
    """
    sz, d = bound.get("$attr:SZ"), bound.get("$attr:D")
    i = bound.get("$attr:I")
    if not (
        isinstance(sz, (tuple, list))
        and all(isinstance(x, int) and x > 0 for x in sz)
        and isinstance(d, int)
        and isinstance(i, int)
        and 0 <= i < len(sz)
    ):
        return False
    a, b = _xshape(bound.get("a")), _xshape(bound.get("b"))
    if not (isinstance(a, tuple) and isinstance(b, tuple) and a and b):
        return False
    r = max(len(a), len(b))
    na = d % r
    for s in (a, b):
        nb = na - (r - len(s))
        if nb < 0 or s[nb] != sum(sz):
            return False
    return True


SPLIT_MUL_ALIGNED = R(
    "split_mul_aligned",
    Op.make(
        "split",
        Op.make("mul", "a", "b"),
        sizes="SZ",
        dim="D",
        index="I",
    ),
    Op.make(
        "mul",
        Op.make("split", "a", sizes="SZ", dim="D", index="I"),
        Op.make("split", "b", sizes="SZ", dim="D", index="I"),
    ),
    check=_check_split_mul_aligned,
    law="a partition shared by both operands distributes over "
    "elementwise mul: split_D(a⊙b) = split_D(a)⊙split_D(b) when both "
    "cover axis D at the partition total — the bisection's splits "
    "enter the decay-weighted k·eta factor.",
)


def _derive_split_unsq(bound: dict) -> dict | None:
    """``DP`` := the pre-image axis of the split, ``u``'s axis skipped."""
    sz, d, u = (
        bound.get("$attr:SZ"),
        bound.get("$attr:D"),
        bound.get("$attr:UD"),
    )
    i = bound.get("$attr:I")
    if not (
        isinstance(sz, (tuple, list))
        and all(isinstance(x, int) and x > 0 for x in sz)
        and isinstance(d, int)
        and isinstance(u, int)
        and isinstance(i, int)
        and 0 <= i < len(sz)
    ):
        return None
    a = _xshape(bound.get("a"))
    if not isinstance(a, tuple):
        return None
    ra = len(a)
    u_out, d_out = u % (ra + 1), d % (ra + 1)
    if d_out == u_out:
        return None  # the inserted size-1 axis cannot carry a partition
    p = d_out if d_out < u_out else d_out - 1
    if a[p] != sum(sz):
        return None
    return {"$attr:DP": p - ra}


SPLIT_UNSQ = R(
    "split_unsqueeze",
    Op.make(
        "split",
        Op.make("unsqueeze", "a", dim="UD"),
        sizes="SZ",
        dim="D",
        index="I",
    ),
    Op.make(
        "unsqueeze",
        Op.make("split", "a", sizes="SZ", dim="DP", index="I"),
        dim="UD",
    ),
    derive=_derive_split_unsq,
    law="a split commutes through a size-1 axis insertion on any "
    "other axis: split_D(unsq(a,u)) = unsq(split_{D'}(a), u) — the "
    "view is a broadcast pad, the partition passes under it.",
)


def _dv_split_attrs(bound: dict) -> tuple | None:
    """``(SZ, I)`` of a decay_vec block-split, or ``None``."""
    sz, t, i, d = (
        bound.get("$attr:SZ"),
        bound.get("$attr:T"),
        bound.get("$attr:I"),
        bound.get("$attr:D"),
    )
    if not (
        isinstance(sz, (tuple, list))
        and all(isinstance(x, int) and x > 0 for x in sz)
        and isinstance(t, int)
        and t == sum(sz)
        and isinstance(i, int)
        and 0 <= i < len(sz)
        and isinstance(d, int)
    ):
        return None
    gs = _xshape(bound.get("g"))
    if isinstance(gs, tuple) and d % (len(gs) + 1) != len(gs):
        return None  # the vector axis is the last one
    return sz, i


def _derive_dv_split(sign: int):
    """Derive ``TI``/``OD`` for a ``decay_vec`` block-split.

    ``dv(g,T,o,s)[k] = g^{s·k+o}``; block ``i`` covers positions
    ``start_i..start_i+SZ_i``, so its vector is
    ``dv(g, SZ_i, offset = o + s·start_i, sign = s)`` — **signed**
    offsets fall out of ``sign=-1`` vectors (the minting hole the
    retro named).
    """

    def derive(bound: dict) -> dict | None:
        attrs = _dv_split_attrs(bound)
        if attrs is None:
            return None
        sz, i = attrs
        o = bound.get("$attr:O")
        start = sum(sz[:i])
        return {
            "$attr:TI": sz[i],
            "$attr:OD": (o if isinstance(o, int) else 0) + sign * start,
        }

    return derive


def _check_dv_split(bound: dict) -> bool:
    return _dv_split_attrs(bound) is not None


DV_SPLIT = [
    R(
        "decay_vec_split_neg_off",
        Op.make(
            "split",
            Op.make("decay_vec", "g", t="T", sign=-1, offset="O"),
            sizes="SZ",
            dim="D",
            index="I",
        ),
        Op.make("decay_vec", "g", t="TI", sign=-1, offset="OD"),
        check=_check_dv_split,
        derive=_derive_dv_split(-1),
        law="a block of a sign=-1 decay vector is a decay vector "
        "with a *negative* offset: dv(g,T,-1,o)[start_i+k] = "
        "g^{-k+o-start_i} = dv(g,SZ_i,-1,o-start_i) — the signed "
        "windows the boundary's steering weights need.",
    ),
    R(
        "decay_vec_split_neg",
        Op.make(
            "split",
            Op.make("decay_vec", "g", t="T", sign=-1),
            sizes="SZ",
            dim="D",
            index="I",
        ),
        Op.make("decay_vec", "g", t="TI", sign=-1, offset="OD"),
        check=_check_dv_split,
        derive=_derive_dv_split(-1),
        law="the offset-free variant: dv(g,T,-1)[start_i+k] = "
        "g^{-k-start_i} = dv(g,SZ_i,-1,-start_i) — mints the "
        "negative-offset members nothing else produces.",
    ),
    R(
        "decay_vec_split_off",
        Op.make(
            "split",
            Op.make("decay_vec", "g", t="T", offset="O"),
            sizes="SZ",
            dim="D",
            index="I",
        ),
        Op.make("decay_vec", "g", t="TI", offset="OD"),
        check=_check_dv_split,
        derive=_derive_dv_split(1),
        law="the sign=+1 twin: dv(g,T,o)[start_i+k] = g^{k+o+start_i}"
        " = dv(g,SZ_i,o+start_i).",
    ),
    R(
        "decay_vec_split_plain",
        Op.make(
            "split",
            Op.make("decay_vec", "g", t="T"),
            sizes="SZ",
            dim="D",
            index="I",
        ),
        Op.make("decay_vec", "g", t="TI", offset="OD"),
        check=_check_dv_split,
        derive=_derive_dv_split(1),
        law="dv(g,T)[start_i+k] = g^{k+start_i} = "
        "dv(g,SZ_i,start_i) — the plain base vector's block split.",
    ),
]


def _check_offset_zero(bound: dict) -> bool:
    return bound.get("$attr:O") == 0


DV_CANON = [
    R(
        "decay_vec_offset_zero",
        Op.make("decay_vec", "g", t="T", offset="O"),
        Op.make("decay_vec", "g", t="T"),
        check=_check_offset_zero,
        law="dv(g,t,offset=0) IS dv(g,t) — the canonical (attr-free) "
        "spelling; merges the block-split's index-0 piece into the "
        "offset-free member the bindings expect.",
    ),
    R(
        "decay_vec_offset_zero_neg",
        Op.make("decay_vec", "g", t="T", offset="O", sign=-1),
        Op.make("decay_vec", "g", t="T", sign=-1),
        check=_check_offset_zero,
        law="dv(g,t,-1,offset=0) IS dv(g,t,-1) — same "
        "canonicalisation for the signed spelling.",
    ),
]


# ---------------------------------------------------------------------------
#  3.  Scalar slide / merge laws — gamma^a·X reindexing
# ---------------------------------------------------------------------------


def _usq_dv_scalar(offset: str):
    """``usq(dv(g, t=1, offset), D)`` — a decay scalar as a broadcast."""
    return Op.make(
        "unsqueeze",
        Op.make("decay_vec", "g", t=1, offset=offset),
        dim="D",
    )


def _derive_od_sum(bound: dict) -> dict | None:
    o1, o2 = bound.get("$attr:O1"), bound.get("$attr:O2")
    if not (isinstance(o1, int) and isinstance(o2, int)):
        return None
    return {"$attr:OD": o1 + o2}


def _check_od_sum(bound: dict) -> bool:
    return _derive_od_sum(bound) is not None


DECAY_GATHER = R(
    "decay_scalar_gather",
    Op.make(
        "mul",
        _usq_dv_scalar("O1"),
        Op.make("mul", _usq_dv_scalar("O2"), "x"),
    ),
    Op.make("mul", _usq_dv_scalar("OD"), "x"),
    check=_check_od_sum,
    derive=_derive_od_sum,
    law="gamma^a·(gamma^b·X) = gamma^{a+b}·X — two decay scalars on "
    "one base merge their offsets; the reindexing step that gathers "
    "the carried chain's per-chunk weights into one exponent.",
)


def _is_decay_scalar(bound: dict) -> bool:
    """Return True — the pattern already pins the t=1 spelling."""
    return True


DECAY_ID = [
    R(
        "decay_scalar_id",
        Op.make(
            "mul",
            Op.make(
                "unsqueeze",
                Op.make("decay_vec", "g", t=1),
                dim="D",
            ),
            "x",
        ),
        "x",
        check=_is_decay_scalar,
        law="gamma^0·X = X: a bare t=1 decay vector is the scalar 1.",
    ),
    R(
        "decay_scalar_id_off",
        Op.make("mul", _usq_dv_scalar("O"), "x"),
        "x",
        check=_check_offset_zero,
        law="gamma^0·X = X under the explicit offset=0 spelling the "
        "gather mints.",
    ),
]


def _check_eta_slide(bound: dict) -> bool:
    """``OD := O + OE`` — the scalar's offset slides into eta's."""
    return _derive_od_sum(bound) is not None


def _state_mm_lhs(o_eta):
    return Op.make(
        "mul",
        _usq_dv_scalar("O1"),
        Op.make(
            "matmul",
            Op.make(
                "transpose",
                Op.make(
                    "mul",
                    "k",
                    Op.make(
                        "unsqueeze",
                        o_eta,
                        dim="DE",
                    ),
                ),
                dim0="D0",
                dim1="D1",
            ),
            "v",
        ),
    )


def _state_mm_rhs(o_eta):
    return Op.make(
        "matmul",
        Op.make(
            "transpose",
            Op.make(
                "mul",
                "k",
                Op.make(
                    "unsqueeze",
                    o_eta,
                    dim="DE",
                ),
            ),
            dim0="D0",
            dim1="D1",
        ),
        "v",
    )


STATE_ETA_SLIDE = [
    R(
        "state_eta_slide",
        _state_mm_lhs(
            Op.make("decay_vec", "g", t="T", sign=-1, offset="O2")
        ),
        _state_mm_rhs(
            Op.make("decay_vec", "g", t="T", sign=-1, offset="OD")
        ),
        check=_check_eta_slide,
        derive=_derive_od_sum,
        law="a decay scalar on a decayed state contribution re-"
        "indexes the eta offset: gamma^a ⊙ ((k⊙eta_o)ᵀv) = "
        "(k⊙eta_{o+a})ᵀv — the scalar is constant on every axis, so "
        "it exits the matmul into the transpose, reorders inside "
        "the elementwise mul, and merges offsets (a 5-step "
        "naturality chain compressed, the "
        "decay_state_weight_sink precedent).",
    ),
    R(
        "state_eta_slide_base",
        _state_mm_lhs(Op.make("decay_vec", "g", t="T", sign=-1)),
        _state_mm_rhs(
            Op.make("decay_vec", "g", t="T", sign=-1, offset="OD")
        ),
        check=_check_od_sum,
        derive=lambda bound: (
            {"$attr:OD": bound["$attr:O1"]}
            if isinstance(bound.get("$attr:O1"), int)
            else None
        ),
        law="the offset-free eta variant: gamma^a ⊙ ((k⊙eta)ᵀv) = "
        "(k⊙eta_{o+a})ᵀv — the own-chunk eta (offset absent) "
        "inherits the scalar's whole offset.",
    ),
]


def _check_usq_factor(bound: dict) -> bool:
    o = bound.get("$attr:O")
    return isinstance(o, int) and o != 0


DECAY_VEC_USQ_FACTOR = R(
    "decay_vec_usq_factor",
    Op.make(
        "unsqueeze",
        Op.make("decay_vec", "g", t="T", sign=-1, offset="O"),
        dim="D",
    ),
    Op.make(
        "mul",
        Op.make(
            "unsqueeze",
            Op.make("decay_vec", "g", t=1, offset="O"),
            dim="D",
        ),
        Op.make(
            "unsqueeze",
            Op.make("decay_vec", "g", t="T", sign=-1),
            dim="D",
        ),
    ),
    check=_check_usq_factor,
    law="the signed-offset decay vector splits into the scalar "
    "offset and the offset-free base in the wires' unsqueeze "
    "packaging: usq(dv(g,T,-1,o)) = usq(dv(g,1,o)) ⊙ usq(dv(g,T,-1))"
    " — mints the scalar steering weight (decay_vec_offset's "
    "signed twin, packaged).",
)


#: The full family: the bisection (parameterized by the boundary's
#: fine partition), the split-naturality chain, the scalar
#: slide/merge laws, the canonicalisation pair, the scalar factor,
#: plus the shipped pieces it composes with — ``split_transpose``
#: (the chain's transpose hop) and ``distribute_mul_add`` (the
#: pointwise distributivity the carried chain's ``mul(w, add)`` needs).
def state_reindex_laws(sizes: tuple[int, ...]) -> list[Rewrite]:
    """Return the state-reindex law family for a boundary partition.

    ``sizes`` is the partition the contraction-axis bisection mints —
    the fine family's chunk width tiling the coarse state's
    contraction extent (``(4,4,4,4)`` on the RetNet winner's 16-wide
    boundary state).
    """
    return [
        matmul_contract_split(sizes),
        SPLIT_MUL_ALIGNED,
        SPLIT_UNSQ,
        SPLIT_TRANSPOSE,
        *DV_SPLIT,
        *DV_CANON,
        DECAY_GATHER,
        *DECAY_ID,
        *STATE_ETA_SLIDE,
        DECAY_VEC_USQ_FACTOR,
        DISTRIBUTE_MUL_ADD,
    ]


#: Bounded-saturation budgets for the close pass — counted in NEW
#: ENODES and persisted across ``eg.run`` calls, so they must cover
#: both the mint pass and every ``post_add`` merge round.
#: ``distribute_mul_add`` is the workhorse of the scalar slide (each
#: ``mul(c, add)`` mints an ``add(mul,mul)``); the bisection is the
#: graph's loudest minter.
_REINDEX_BUDGETS: dict[str, int] = {
    "split_mul_aligned": 3000,
    "split_unsqueeze": 3000,
    "decay_vec_split_neg_off": 2000,
    "decay_vec_split_neg": 2000,
    "decay_vec_split_off": 2000,
    "decay_vec_split_plain": 2000,
    "decay_vec_offset_zero": 2000,
    "decay_vec_offset_zero_neg": 2000,
    "decay_scalar_gather": 6000,
    "decay_scalar_id": 600,
    "decay_scalar_id_off": 600,
    "state_eta_slide": 8000,
    "state_eta_slide_base": 8000,
    "decay_vec_usq_factor": 2000,
    "distribute_mul_add": 60000,
    "split_transpose": 3000,
}


def reindex_budgets(sizes: tuple[int, ...]) -> dict[str, int]:
    """Bounded-saturation budgets, including the generated bisection."""
    name = "matmul_contract_split_" + "_".join(map(str, sizes))
    return {**_REINDEX_BUDGETS, name: 8000}


# ---------------------------------------------------------------------------
#  The driver: measure the boundary, mint, close, verify, emit
# ---------------------------------------------------------------------------


def _boundary_sizes(rep: Any) -> tuple[int, ...] | None:
    """Derive the contraction partition the boundary needs, measured.

    ``w`` — the fine family's chunk width: the producing step's
    missing steering eta carries ``t = w``.  ``T`` — the coarse
    state's contraction extent: the target's value split carries it.
    The partition ``(w,)*q (+ rem)`` tiles ``T`` into fine-width
    pieces — exactly the alignment that makes the bisected pieces land
    in the fine family's own spellings.
    """
    ms = rep.boundary.get("missing_steer") or {}
    if ms:
        ce = next(iter(ms.values()))["ce"]
        a = fp._dv_attrs(ce) or {}
        w = a.get("t")
    else:
        w = None
    tt = rep.boundary.get("target_term")
    extent = None
    if isinstance(tt, Op) and tt.op == "matmul" and len(tt.args) == 2:
        b = tt.args[1]
        if isinstance(b, Op) and b.op == "split":
            sza = dict(b.attrs)
            idx = sza.get("index")
            sz = sza.get("sizes")
            if (
                isinstance(sz, (tuple, list))
                and isinstance(idx, int)
                and 0 <= idx < len(sz)
            ):
                extent = sz[idx]
    if not (isinstance(w, int) and w > 0):
        # fall back to the producing block's own span
        at = rep.boundary.get("at")
        if at is not None:
            span = fp._split_span(rep.bindings.get(at, {}))
            if span is not None:
                w = span[1]
    if not (isinstance(w, int) and w > 0) or not (
        isinstance(extent, int) and extent >= w
    ):
        return None
    q, r = divmod(extent, w)
    return (w,) * q + ((r,) if r else ())


def reindex_close(
    eg: Any,
    rep: Any,
    root_term: Any,
    cost_fn: Any,
    *,
    root_eid: int | None = None,
    sizes: tuple[int, ...] | None = None,
    max_iterations: int = 8,
    patience: int = 3,
) -> Any:
    """Close the family boundary with the state-reindex law family.

    After :func:`form_propose.form_propose` has run (the two-chain
    result), this derives the boundary's fine partition, runs the
    family once to mint the steering wires and bisect the coarse
    state, then re-walks the compensated arm with a ``post_add``
    saturation hook — so the steered terms the arm mints are merged by
    the slide/merge laws immediately.  On success it emits ONE
    ``unroll(step, n)`` over the covered blocks (folded to a
    ``Loop``), the assembled program, and the e-class certificate.
    """
    rec: dict = {"closed": False}
    rep.reindex = rec
    covered = sorted(i for r in rep.chains for i in r.blocks if r.ok)
    if len(covered) < 2:
        rec["obstruction"] = "no covered blocks"
        return rep
    if rep.step_morph is None:
        rec["obstruction"] = "no step morphism"
        return rep
    if sizes is None:
        sizes = _boundary_sizes(rep)
    if sizes is None:
        rec["obstruction"] = "boundary sizes underivable"
        return rep
    rec["sizes"] = sizes
    rules = state_reindex_laws(sizes)
    rid = (
        eg.find(root_eid)
        if root_eid is not None
        else fp._eclass(eg, root_term)
    )
    fires0 = dict(eg.rule_fires)

    def run() -> dict:
        return eg.run(
            rules,
            rid,
            max_iterations=max_iterations,
            max_nodes=eg.n_enodes + 80_000,
            patience=patience,
            rule_budgets=reindex_budgets(sizes),
        )

    mint_stats = run()
    rec["mint"] = {
        "n_enodes": eg.n_enodes,
        "iterations": mint_stats.get("iterations"),
    }

    def post_add() -> None:
        run()

    verdicts = fp.compensated_loop_check(
        eg, rep, root_term, cost_fn, post_add=post_add
    )
    rep.single_loop_compensated = verdicts
    rec["verdicts"] = dict(verdicts)
    rec["n_enodes"] = eg.n_enodes
    rec["fired"] = {
        k: eg.rule_fires.get(k, 0) - fires0.get(k, 0)
        for k in set(eg.rule_fires) | set(fires0)
        if eg.rule_fires.get(k, 0) - fires0.get(k, 0) != 0
    }
    if not (verdicts and all(verdicts.values())):
        return rep

    rec["closed"] = True
    _emit_single_loop(eg, rep, root_term, covered)
    return rep


def _emit_single_loop(
    eg: Any, rep: Any, root_term: Any, covered: list[int]
) -> None:
    """Assemble ``seq(pre, unroll(step,n), post)`` and certify it.

    The compensated arm recorded each iteration's effective wire
    bindings in ``rep.single_envs`` and the carried seed in
    ``rep.single_seed``; this builds the program morphism exactly as
    ``form_propose`` does per chain — one loop instead of two — and
    checks the lowered DAG lands in the winner's root e-class.
    """
    blocks = block_outputs(root_term)
    nx = len(fp.STEP_INPUTS) - 1
    morph = unroll(rep.step_morph, len(covered))
    loop = cb.Loop(rep.step_morph, len(covered), nx=nx, ny=1)

    wire_terms = [rep.single_seed]
    for i in covered:
        env = rep.single_envs[i]
        wire_terms.extend(env[v] for v in fp.STEP_INPUTS[1:])

    skel = fp._concat_skeleton(root_term, blocks)
    skel_leaves = fp._skel_leaves(skel)
    n_out = len(skel_leaves)
    raw_leaves = [x for x in skel_leaves if isinstance(x, tuple)]
    peeled = [i for i in range(rep.n_blocks) if i not in covered]
    extras = [blocks[i] for i in peeled] + [r[1] for r in raw_leaves]

    pre = cb._par(*(cb.Leaf(t) for t in wire_terms + extras))
    mid = cb._par(morph, cb.Id(len(extras)))
    # mid emits [S_end, o_1..o_n, *extras]; the state wire is dropped.
    pos_of: dict[Any, int] = {}
    pos = 1
    for i in covered:
        pos_of[i] = pos
        pos += 1
    for k in range(len(extras)):
        pos_of[("x", k)] = pos + k
    keep = sorted(pos_of.values())
    drop_seg = cb._par(
        cb.Drop(), cb.Id(len(covered)), cb.Id(len(extras))
    )
    shift = {old: new for new, old in enumerate(keep)}
    peeled_set = set(peeled)
    scan: list[int] = []
    for leaf_ in skel_leaves:
        if isinstance(leaf_, tuple):
            scan.append(
                pos_of[("x", len(peeled) + raw_leaves.index(leaf_))]
            )
        elif leaf_ in peeled_set:
            scan.append(pos_of[("x", peeled.index(leaf_))])
        else:
            scan.append(pos_of[leaf_])
    scan_perm = cb._perm([shift[s] for s in scan])
    post = cb._seq(
        drop_seg,
        scan_perm,
        fp._concat_tree_morph(fp._positional(skel, [0]), n_out),
    )
    rep.single_prog = cb._seq(pre, mid, post)
    rep.single_prog_stats = cb.morph_stats(rep.single_prog)

    r = saturate_comb(morph, [PROPOSE_LOOP, EXPAND_LOOP])
    rep.single_closure = {
        "n_terms": r["n_terms"],
        "fired": r["fired"],
        "cost": r["cost"],
    }
    rep.single_loop = r["best"] if loop_headed(r["best"]) else loop

    dag = comb_to_dag(rep.single_prog)
    root_cls = fp._eclass(eg, root_term)
    rep.single_whole_equal = eg.find(eg.add_term(dag)) == root_cls

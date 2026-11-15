"""Shape-guarded law differential — the b9fe227 phantom-scalar coverage.

engine-eqsat commit ``b9fe227`` fixed ``__mlopt_leaf__`` bridge leaves
reading as scalar ``()`` — before the fix every broadcast/shape guard
(``bcast-eq``/``bcast-into``/``unsq-out``/``reshape-out``/``mm-out``/
``dim-*``/``rank``/…) saw phantom scalars that broadcast-compatibly
absorb every grid, so unsound-direction laws could fire where the real
shapes decline, and decline-direction laws under-fired on rank/shape
floors a scalar can never reach.

This driver exercises BOTH directions differentially.  Each case is a
*concrete* term — ``Var``/``Param`` leaves carry ``TensorType`` shapes
so the same declared shape reaches the Python ``_shape_of`` and the
Rust ``shape::mlopt_leaf_info`` (via the ``__mlopt_leaf__`` wire
payload ``eqsat_differential._kterm`` now emits).  For every case the
law runs alone on both engines and we record:

* ``expect="fire"`` — the guard SHOULD hold: both engines must fire
  the law (fire-parity), the instantiated RHS must land in the root
  e-class, and input/Python-best/Rust-best must be semantically equal
  under the shared numpy evaluator.
* ``expect="decline"`` — a near-miss shape where the guard SHOULD
  veto: neither engine may fire (decline-parity), and the would-be RHS
  must stay out of the root e-class.

A ``divergence`` verdict — one engine fires where the other declines,
or fired extractions disagree semantically — is a residual soundness
signal and is reported, never smoothed over.  ``no-fire`` on an
``expect="fire"`` case is an honest coverage gap (both engines agree
the guard declines), distinct from divergence.

Coverage: the 12 unsound-direction laws get fire AND decline cases;
the decline-direction families (``sdpa_fold_*``, ``glu_fold``,
``distribute/factor_matmul*``, ``assoc_linear_bias*``,
``linear_channel_scale*``) get representative now-fire cases.

Usage::

    .venv/bin/python bench/drivers/audit/eqsat_shape_guards.py
    .venv/bin/python bench/drivers/audit/eqsat_shape_guards.py --law mul_unsq_pad_l
    .venv/bin/python bench/drivers/audit/eqsat_shape_guards.py --json /tmp/sg.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (str(_HERE), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eqsat_differential as d  # noqa: E402
from mlopt_core.egraph import EGraph  # noqa: E402
from mlopt_core.ir import (  # noqa: E402
    Const,
    Op,
    Param,
    TensorType,
    Var,
)

# ---------------------------------------------------------------------------
#  Term builders — typed leaves are the whole point of this driver
# ---------------------------------------------------------------------------


def V(name: str, shape: tuple) -> Var:
    return Var(name, TensorType(tuple(shape)))


def P(name: str, shape: tuple) -> Param:
    return Param(name, TensorType(tuple(shape)))


def C(v: Any) -> Const:
    return Const(v)


def unsq(t: Any, dim: int) -> Op:
    return Op.make("unsqueeze", t, dim=dim)


# ---------------------------------------------------------------------------
#  The case table
# ---------------------------------------------------------------------------


def _cases() -> list[dict]:
    """(id, law, term, expected-rhs, expect, note) — built lazily so the
    table reads as data."""
    cases: list[dict] = []

    def add(id_, law, term, rhs, expect, note=""):
        cases.append(
            {
                "id": id_,
                "law": law,
                "term": term,
                "rhs": rhs,
                "expect": expect,
                "note": note,
            }
        )

    # -- mul_unsq_pad_l: mul(unsq(u,d), v) -> mul(u,v) --------------------
    u3, v23 = V("u", (3,)), V("v", (2, 3))
    add(
        "mul_unsq_pad_l/lead-pad",
        "mul_unsq_pad_l",
        Op.make("mul", unsq(u3, 0), v23),
        Op.make("mul", u3, v23),
        "fire",
        "u=(3,) v=(2,3) d=0: leading pad, grids coincide",
    )
    u23, v123 = V("u2", (2, 3)), V("v2", (1, 2, 3))
    add(
        "mul_unsq_pad_l/rank3-pad",
        "mul_unsq_pad_l",
        Op.make("mul", unsq(u23, 0), v123),
        Op.make("mul", u23, v123),
        "fire",
        "u=(2,3) v=(1,2,3) d=0: pad into a broadcastable rank-3",
    )
    u46, w46 = V("u3", (4, 6)), V("w3", (4, 6))
    add(
        "mul_unsq_pad_l/wp4-verbatim",
        "mul_unsq_pad_l",
        Op.make("mul", unsq(u46, -1), unsq(w46, -1)),
        Op.make("mul", u46, unsq(w46, -1)),
        "decline",
        "the WP4 parity failure: bcast(u,v) is ill-typed",
    )
    u23b, v234 = V("u4", (2, 3)), V("v4", (2, 3, 4))
    add(
        "mul_unsq_pad_l/grid-mismatch",
        "mul_unsq_pad_l",
        Op.make("mul", unsq(u23b, -1), v234),
        Op.make("mul", u23b, v234),
        "decline",
        "unsq grid (2,3,4) != bcast(u,v) (ill-typed)",
    )
    u23c, v23c = V("u5", (2, 3)), V("v5", (2, 3))
    add(
        "mul_unsq_pad_l/mid-insert",
        "mul_unsq_pad_l",
        Op.make("mul", unsq(u23c, 1), v23c),
        Op.make("mul", u23c, v23c),
        "decline",
        "mid-axis insert reshuffles the read — grids differ",
    )

    # -- mul_unsq_pad_r: mul(u, unsq(v,d)) -> mul(u,v) --------------------
    u23d, v3b = V("u6", (2, 3)), V("v6", (3,))
    add(
        "mul_unsq_pad_r/lead-pad",
        "mul_unsq_pad_r",
        Op.make("mul", u23d, unsq(v3b, 0)),
        Op.make("mul", u23d, v3b),
        "fire",
        "mirror of the l-case",
    )
    u234, v23d = V("u7", (2, 3, 4)), V("v7", (2, 3))
    add(
        "mul_unsq_pad_r/grid-mismatch",
        "mul_unsq_pad_r",
        Op.make("mul", u234, unsq(v23d, -1)),
        Op.make("mul", u234, v23d),
        "decline",
        "bcast(v,u) ill-typed — the WP4 shape on the right operand",
    )
    u3c, v23e = V("u8", (3,)), V("v8", (2, 3))
    add(
        "mul_unsq_pad_r/mid-insert",
        "mul_unsq_pad_r",
        Op.make("mul", u3c, unsq(v23e, 1)),
        Op.make("mul", u3c, v23e),
        "decline",
        "unsq(v,1)=(2,1,3) grid (2,1,3) != bcast(v,u) (2,3)",
    )

    # -- sub_unsq_pad_l ----------------------------------------------------
    u3d, v23f = V("u9", (3,)), V("v9", (2, 3))
    add(
        "sub_unsq_pad_l/lead-pad",
        "sub_unsq_pad_l",
        Op.make("sub", unsq(u3d, 0), v23f),
        Op.make("sub", u3d, v23f),
        "fire",
        "same guard family as mul_unsq_pad_l, asymmetric op",
    )
    u23e, v234b = V("u10", (2, 3)), V("v10", (2, 3, 4))
    add(
        "sub_unsq_pad_l/grid-mismatch",
        "sub_unsq_pad_l",
        Op.make("sub", unsq(u23e, -1), v234b),
        Op.make("sub", u23e, v234b),
        "decline",
    )

    # -- add_unsq_pad_r ----------------------------------------------------
    u23f, v3e = V("u11", (2, 3)), V("v11", (3,))
    add(
        "add_unsq_pad_r/lead-pad",
        "add_unsq_pad_r",
        Op.make("add", u23f, unsq(v3e, 0)),
        Op.make("add", u23f, v3e),
        "fire",
        "the RelativePositionBias paying-site spelling",
    )
    u234c, v23g = V("u12", (2, 3, 4)), V("v12", (2, 3))
    add(
        "add_unsq_pad_r/grid-mismatch",
        "add_unsq_pad_r",
        Op.make("add", u234c, unsq(v23g, -1)),
        Op.make("add", u234c, v23g),
        "decline",
    )

    # -- mul_reshape_inert_l ----------------------------------------------
    u13, v23h = V("u13", (1, 3)), V("v13", (2, 3))
    add(
        "mul_reshape_inert_l/squeeze-1",
        "mul_reshape_inert_l",
        Op.make("mul", Op.make("reshape", u13, shape=(3,)), v23h),
        Op.make("mul", u13, v23h),
        "fire",
        "reshape (1,3)->(3,) drops a broadcast-1 axis",
    )
    u23i, v123b = V("u14", (2, 3)), V("v14", (1, 2, 3))
    add(
        "mul_reshape_inert_l/lead-1",
        "mul_reshape_inert_l",
        Op.make(
            "mul", Op.make("reshape", u23i, shape=(1, 2, 3)), v123b
        ),
        Op.make("mul", u23i, v123b),
        "fire",
        "reshape inserts a leading 1 — grids coincide",
    )
    u23j, v23i = V("u15", (2, 3)), V("v15", (2, 3))
    add(
        "mul_reshape_inert_l/permute",
        "mul_reshape_inert_l",
        Op.make("mul", Op.make("reshape", u23j, shape=(3, 2)), v23i),
        Op.make("mul", u23j, v23i),
        "decline",
        "(3,2) reshape permutes data AND breaks the grid",
    )
    u6, v23j = V("u16", (6,)), V("v16", (2, 3))
    add(
        "mul_reshape_inert_l/rhs-illtyped",
        "mul_reshape_inert_l",
        Op.make("mul", Op.make("reshape", u6, shape=(2, 3)), v23j),
        Op.make("mul", u6, v23j),
        "decline",
        "bcast(u,v) itself is ill-typed — must not fire",
    )

    # -- linear_row_scale / _rev ------------------------------------------
    x35, w45, r31 = V("x17", (3, 5)), V("w17", (4, 5)), V("r17", (3, 1))
    add(
        "linear_row_scale/fire",
        "linear_row_scale",
        Op.make("linear", Op.make("mul", x35, r31), w45),
        Op.make("mul", Op.make("linear", x35, w45), r31),
        "fire",
        "r=(3,1) broadcasts into out=(3,4), last dim 1",
    )
    x35b, w45b = V("x18", (3, 5)), V("w18", (4, 5))
    add(
        "linear_row_scale/scalar-r",
        "linear_row_scale",
        Op.make("linear", Op.make("mul", x35b, C(2)), w45b),
        Op.make("mul", Op.make("linear", x35b, w45b), C(2)),
        "fire",
        "scalar scale — the scalar() disjunct",
    )
    x235, w45c, r231 = (
        V("x19", (2, 3, 5)),
        V("w19", (4, 5)),
        V("r19", (2, 3, 1)),
    )
    add(
        "linear_row_scale/batched",
        "linear_row_scale",
        Op.make("linear", Op.make("mul", x235, r231), w45c),
        Op.make("mul", Op.make("linear", x235, w45c), r231),
        "fire",
        "batched x, row scale (2,3,1) into out (2,3,4)",
    )
    x35c, w45d, r4 = V("x20", (3, 5)), V("w20", (4, 5)), V("r20", (4,))
    add(
        "linear_row_scale/channel-shaped-r",
        "linear_row_scale",
        Op.make("linear", Op.make("mul", x35c, r4), w45d),
        Op.make("mul", Op.make("linear", x35c, w45d), r4),
        "decline",
        "r=(4,) last dim != 1 — phantom scalar fired pre-fix",
    )
    r34 = V("r21", (3, 4))
    add(
        "linear_row_scale/out-shaped-r",
        "linear_row_scale",
        Op.make("linear", Op.make("mul", x35c, r34), w45d),
        Op.make("mul", Op.make("linear", x35c, w45d), r34),
        "decline",
        "r matches the OUTPUT shape — not a row scale",
    )
    w245 = V("w22", (2, 4, 5))
    r221 = V("r22", (2, 3, 1))
    add(
        "linear_row_scale/rank3-w",
        "linear_row_scale",
        Op.make("linear", Op.make("mul", x35c, r221), w245),
        Op.make("mul", Op.make("linear", x35c, w245), r221),
        "decline",
        "rank(W)=3 violates F.linear's weight arity",
    )
    w46 = V("w23", (4, 6))
    add(
        "linear_row_scale/bad-contraction",
        "linear_row_scale",
        Op.make("linear", Op.make("mul", x35c, r31), w46),
        Op.make("mul", Op.make("linear", x35c, w46), r31),
        "decline",
        "x[-1]=5 vs W.T[-2]=6 — mm-out is None",
    )
    # rev twin
    add(
        "linear_row_scale_rev/fire",
        "linear_row_scale_rev",
        Op.make("mul", Op.make("linear", x35, w45), r31),
        Op.make("linear", Op.make("mul", x35, r31), w45),
        "fire",
    )
    add(
        "linear_row_scale_rev/decline",
        "linear_row_scale_rev",
        Op.make("mul", Op.make("linear", x35c, w45d), r4),
        Op.make("linear", Op.make("mul", x35c, r4), w45d),
        "decline",
        "r=(4,) — same near-miss on the hoist direction",
    )

    # -- linear_channel_to_row_scale --------------------------------------
    x35d, w45e = V("x23", (3, 5)), V("w24", (4, 5))
    add(
        "linear_channel_to_row_scale/scalar-c",
        "linear_channel_to_row_scale",
        Op.make("linear", x35d, Op.make("mul", w45e, C(2))),
        Op.make("mul", Op.make("linear", x35d, w45e), C(2)),
        "fire",
        "scalar c satisfies both premise guards",
    )
    x31, w41, c1 = V("x25", (3, 1)), V("w25", (4, 1)), V("c25", (1,))
    add(
        "linear_channel_to_row_scale/unit-c",
        "linear_channel_to_row_scale",
        Op.make("linear", x31, Op.make("mul", w41, c1)),
        Op.make("mul", Op.make("linear", x31, w41), c1),
        "fire",
        "c=(1,) is channel (in=1) AND row (last dim 1)",
    )
    c5 = V("c26", (5,))
    add(
        "linear_channel_to_row_scale/channel-only-c",
        "linear_channel_to_row_scale",
        Op.make("linear", x35d, Op.make("mul", w45e, c5)),
        Op.make("mul", Op.make("linear", x35d, w45e), c5),
        "decline",
        "c=(5,) is a channel scale but NOT a row scale — the "
        "intersection guard must veto",
    )
    c15 = V("c27", (1, 5))
    add(
        "linear_channel_to_row_scale/padded-channel-c",
        "linear_channel_to_row_scale",
        Op.make("linear", x35d, Op.make("mul", w45e, c15)),
        Op.make("mul", Op.make("linear", x35d, w45e), c15),
        "decline",
        "c=(1,5) ones-but-last channel scale, last dim 5 != 1",
    )

    # -- select_mul ---------------------------------------------------------
    u23k, v23k = V("u28", (2, 3)), V("v28", (2, 3))
    add(
        "select_mul/fire",
        "select_mul",
        Op.make(
            "mul",
            Op.make("select", u23k, dim=0, index=1),
            Op.make("select", v23k, dim=0, index=1),
        ),
        Op.make("select", Op.make("mul", u23k, v23k), dim=0, index=1),
        "fire",
        "u,v agree on the selected axis D=0",
    )
    u23l, v21 = V("u29", (2, 3)), V("v29", (2, 1))
    add(
        "select_mul/broadcast-fire",
        "select_mul",
        Op.make(
            "mul",
            Op.make("select", u23l, dim=0, index=0),
            Op.make("select", v21, dim=0, index=0),
        ),
        Op.make("select", Op.make("mul", u23l, v21), dim=0, index=0),
        "fire",
        "off-axis broadcast is safe — grids still shaped",
    )
    u23m, v24 = V("u30", (2, 3)), V("v30", (2, 4))
    add(
        "select_mul/dim-mismatch",
        "select_mul",
        Op.make(
            "mul",
            Op.make("select", u23m, dim=1, index=0),
            Op.make("select", v24, dim=1, index=0),
        ),
        Op.make("select", Op.make("mul", u23m, v24), dim=1, index=0),
        "decline",
        "u[1]=3 != v[1]=4 on the selected axis",
    )
    u23n, v231 = V("u31", (2, 3)), V("v31", (2, 3, 1))
    add(
        "select_mul/bcast-illtyped",
        "select_mul",
        Op.make(
            "mul",
            Op.make("select", u23n, dim=0, index=0),
            Op.make("select", v231, dim=0, index=0),
        ),
        Op.make("select", Op.make("mul", u23n, v231), dim=0, index=0),
        "decline",
        "the derivable-gate audit corner: dim-eq holds, bcast fails",
    )

    # -- swiglu_fuse / parallel_mul_fuse ------------------------------------
    x23o, a43, b43 = (
        V("x32", (2, 3)),
        V("A32", (4, 3)),
        V("B32", (4, 3)),
    )
    fused = Op.make(
        "mul",
        Op.make("silu", Op.make("linear", x23o, a43)),
        Op.make("linear", x23o, b43),
    )
    cat = Op.make("concat", a43, b43, dim=0)
    lin = Op.make("linear", x23o, cat)
    rhs_swiglu = Op.make(
        "mul",
        Op.make(
            "silu",
            Op.make("chunk", lin, chunks=2, dim=-1, index=0),
        ),
        Op.make("chunk", lin, chunks=2, dim=-1, index=1),
    )
    add(
        "swiglu_fuse/fire",
        "swiglu_fuse",
        fused,
        rhs_swiglu,
        "fire",
        "A,B same (out,in) — concat+chunk halves align",
    )
    a45, b53 = V("A33", (4, 3)), V("B33", (5, 3))
    add(
        "swiglu_fuse/uneven-out",
        "swiglu_fuse",
        Op.make(
            "mul",
            Op.make("silu", Op.make("linear", x23o, a45)),
            Op.make("linear", x23o, b53),
        ),
        rhs_swiglu,
        "decline",
        "A,B out-dims differ — concat gives 9, chunks split 5/4",
    )
    b3 = V("B34", (3,))
    add(
        "swiglu_fuse/rank1-b",
        "swiglu_fuse",
        Op.make(
            "mul",
            Op.make("silu", Op.make("linear", x23o, a43)),
            Op.make("linear", x23o, b3),
        ),
        rhs_swiglu,
        "decline",
        "rank(B)=1 violates the rank>=2 floor",
    )
    par = Op.make(
        "mul",
        Op.make("linear", x23o, a43),
        Op.make("linear", x23o, b43),
    )
    rhs_par = Op.make(
        "mul",
        Op.make("chunk", lin, chunks=2, dim=-1, index=0),
        Op.make("chunk", lin, chunks=2, dim=-1, index=1),
    )
    add(
        "parallel_mul_fuse/fire",
        "parallel_mul_fuse",
        par,
        rhs_par,
        "fire",
    )
    add(
        "parallel_mul_fuse/uneven-out",
        "parallel_mul_fuse",
        Op.make(
            "mul",
            Op.make("linear", x23o, a45),
            Op.make("linear", x23o, b53),
        ),
        rhs_par,
        "decline",
    )

    # -- decline-direction: now-fire representatives ----------------------
    w34, a42, b42 = V("w35", (3, 4)), V("a35", (4, 2)), V("b35", (4, 2))
    add(
        "distribute_matmul_over_add/fire",
        "distribute_matmul_over_add",
        Op.make("matmul", w34, Op.make("add", a42, b42)),
        Op.make(
            "add",
            Op.make("matmul", w34, a42),
            Op.make("matmul", w34, b42),
        ),
        "fire",
        "a,b same shape — rank-eq + shape-eq disjunct",
    )
    a41, b41 = V("a36", (4,)), V("b36", (1,))
    add(
        "distribute_matmul_over_add/broadcast-addends",
        "distribute_matmul_over_add",
        Op.make("matmul", w34, Op.make("add", a41, b41)),
        Op.make(
            "add",
            Op.make("matmul", w34, a41),
            Op.make("matmul", w34, b41),
        ),
        "decline",
        "rank-1 addends (4,)vs(1,): shape-eq fails — the audit's "
        "rhs-err corner",
    )
    add(
        "factor_matmul/fire",
        "factor_matmul",
        Op.make(
            "add",
            Op.make("matmul", w34, a42),
            Op.make("matmul", w34, b42),
        ),
        Op.make("matmul", w34, Op.make("add", a42, b42)),
        "fire",
        "reverse distributivity, same addends",
    )

    # assoc_linear_bias: (B,b2)∘(A,b1) = linear(x, BA, B·b1) + b2
    x23p, A43p, b14, B54, b25 = (
        V("x36", (2, 3)),
        V("A36", (4, 3)),
        V("b1_36", (4,)),
        V("B36", (5, 4)),
        V("b2_36", (5,)),
    )
    add(
        "assoc_linear_bias/fire",
        "assoc_linear_bias",
        Op.make(
            "linear",
            Op.make("linear", x23p, A43p, b14),
            B54,
            b25,
        ),
        Op.make(
            "add",
            Op.make(
                "linear",
                x23p,
                Op.make("matmul", B54, A43p),
                Op.make("matmul", B54, b14),
            ),
            b25,
        ),
        "fire",
        "A=(4,3) B=(5,4) b1=(4,) b2=(5,) — every dim-eq holds",
    )
    b15 = V("b1_37", (3,))
    add(
        "assoc_linear_bias/bad-inner-bias",
        "assoc_linear_bias",
        Op.make(
            "linear",
            Op.make("linear", x23p, A43p, b15),
            B54,
            b25,
        ),
        Op.make(
            "add",
            Op.make(
                "linear",
                x23p,
                Op.make("matmul", B54, A43p),
                Op.make("matmul", B54, b15),
            ),
            b25,
        ),
        "decline",
        "b1=(3,) != A[0]=4 — dim-eq veto",
    )

    # linear_channel_scale: linear(mul(x,c),W) -> linear(x, mul(W,c))
    c53 = V("c37", (5,))
    add(
        "linear_channel_scale/fire",
        "linear_channel_scale",
        Op.make("linear", Op.make("mul", x35d, c53), w45e),
        Op.make("linear", x35d, Op.make("mul", w45e, c53)),
        "fire",
        "c=(5,) is the in-feature channel scale",
    )
    c54 = V("c38", (5, 1))
    add(
        "linear_channel_scale/non-channel",
        "linear_channel_scale",
        Op.make("linear", Op.make("mul", x35d, c54), w45e),
        Op.make("linear", x35d, Op.make("mul", w45e, c54)),
        "decline",
        "c=(5,1) last dim 1 != W[-1]=5 — not a channel scale",
    )

    # sdpa_fold_nomask: matmul(softmax(matmul(Q,K^T),-1),V) -> sdpa
    q238, k238, v238 = (
        V("Q38", (2, 3, 8)),
        V("K38", (2, 3, 8)),
        V("V38", (2, 3, 8)),
    )
    scores = Op.make(
        "matmul", q238, Op.make("transpose", k238, dim0=2, dim1=1)
    )
    add(
        "sdpa_fold_nomask/fire",
        "sdpa_fold_nomask",
        Op.make("matmul", Op.make("softmax", scores, dim=-1), v238),
        Op.make("sdpa", q238, k238, v238, scale=1.0),
        "fire",
        "mask-free spelled attention, K transposed on last two axes",
    )
    q232, k232, v232 = (
        V("Q39", (2, 3, 2)),
        V("K39", (2, 3, 2)),
        V("V39", (2, 3, 2)),
    )
    scores2 = Op.make(
        "matmul", q232, Op.make("transpose", k232, dim0=2, dim1=1)
    )
    add(
        "sdpa_fold_nomask/small-extents",
        "sdpa_fold_nomask",
        Op.make("matmul", Op.make("softmax", scores2, dim=-1), v232),
        Op.make("sdpa", q232, k232, v232, scale=1.0),
        "fire",
        "same shape family, smaller extents",
    )
    # a decline case: the transpose must swap K's LAST TWO axes —
    # (0,1) swaps batch/head instead, so axes-last2 vetoes.
    scores3 = Op.make(
        "matmul", q238, Op.make("transpose", k238, dim0=0, dim1=1)
    )
    add(
        "sdpa_fold_nomask/wrong-transpose-axes",
        "sdpa_fold_nomask",
        Op.make("matmul", Op.make("softmax", scores3, dim=-1), v238),
        Op.make("sdpa", q238, k238, v238, scale=1.0),
        "decline",
        "transpose(0,1) is not the last-two-axes swap — "
        "axes-last2 vetoes (and the LHS cannot even contract)",
    )

    # glu_fold: mul(chunk(u,2,D,0), sigmoid(chunk(u,2,D,1))) -> glu(u,D)
    u28g = V("u40", (2, 8))
    add(
        "glu_fold/fire",
        "glu_fold",
        Op.make(
            "mul",
            Op.make("chunk", u28g, chunks=2, dim=1, index=0),
            Op.make(
                "sigmoid",
                Op.make("chunk", u28g, chunks=2, dim=1, index=1),
            ),
        ),
        Op.make("glu", u28g, dim=1),
        "fire",
        "split axis extent 8 is even",
    )
    u23g = V("u41", (2, 3))
    add(
        "glu_fold/odd-axis",
        "glu_fold",
        Op.make(
            "mul",
            Op.make("chunk", u23g, chunks=2, dim=1, index=0),
            Op.make(
                "sigmoid",
                Op.make("chunk", u23g, chunks=2, dim=1, index=1),
            ),
        ),
        Op.make("glu", u23g, dim=1),
        "decline",
        "odd split axis: chunk goes 2+1, glu halves exactly — "
        "the rhs-err corner",
    )

    # -- second batch: the rest of the shape-reading surface ------------
    # sdpa_fold_mul_nomask — the scaled spelling, dspec mints scale=s
    scores_m = Op.make("mul", scores, C(0.5))
    add(
        "sdpa_fold_mul_nomask/fire",
        "sdpa_fold_mul_nomask",
        Op.make("matmul", Op.make("softmax", scores_m, dim=-1), v238),
        Op.make("sdpa", q238, k238, v238, scale=0.5),
        "fire",
        "mul-form scale: S=0.5 lands in the sdpa scale attr",
    )

    # weight_distribute/factor_matmul — _COND_MM_WEIGHTS on W,W2
    x23q, w34q, w2_34 = (
        V("x42", (2, 3)),
        V("W42", (3, 4)),
        V("W2_42", (3, 4)),
    )
    add(
        "weight_distribute_matmul/fire",
        "weight_distribute_matmul",
        Op.make("matmul", x23q, Op.make("add", w34q, w2_34)),
        Op.make(
            "add",
            Op.make("matmul", x23q, w34q),
            Op.make("matmul", x23q, w2_34),
        ),
        "fire",
        "W,W2 same shape — the shape-eq disjunct",
    )
    w2_35 = V("W2_43", (5, 4))
    add(
        "weight_distribute_matmul/weight-mismatch",
        "weight_distribute_matmul",
        Op.make("matmul", x23q, Op.make("add", w34q, w2_35)),
        Op.make(
            "add",
            Op.make("matmul", x23q, w34q),
            Op.make("matmul", x23q, w2_35),
        ),
        "decline",
        "W[-2]=3 vs W2[-2]=5 — neither disjunct holds",
    )
    add(
        "weight_factor_matmul/fire",
        "weight_factor_matmul",
        Op.make(
            "add",
            Op.make("matmul", x23q, w34q),
            Op.make("matmul", x23q, w2_34),
        ),
        Op.make("matmul", x23q, Op.make("add", w34q, w2_34)),
        "fire",
    )

    # weight_factor_linear — the linear spelling (dim-eq on -1)
    w43l, w2_43l = V("W44", (4, 3)), V("W2_44", (4, 3))
    add(
        "weight_factor_linear/fire",
        "weight_factor_linear",
        Op.make(
            "add",
            Op.make("linear", x23q, w43l),
            Op.make("linear", x23q, w2_43l),
        ),
        Op.make("linear", x23q, Op.make("add", w43l, w2_43l)),
        "fire",
        "W,W2 (out,in) agree on the contraction axis W[-1]",
    )
    w2_44l = V("W2_45", (4, 4))
    add(
        "weight_factor_linear/contraction-mismatch",
        "weight_factor_linear",
        Op.make(
            "add",
            Op.make("linear", x23q, w43l),
            Op.make("linear", x23q, w2_44l),
        ),
        Op.make("linear", x23q, Op.make("add", w43l, w2_44l)),
        "decline",
        "W[-1]=3 vs W2[-1]=4 — the linear contraction axis "
        "disagrees (the _COND_MM_WEIGHTS_LINEAR veto)",
    )

    # weight_distribute_linear — the merged-weight spelling; the
    # same _COND_MM_WEIGHTS guard reads W,W2 inside the add.
    add(
        "weight_distribute_linear/fire",
        "weight_distribute_linear",
        Op.make("linear", x23q, Op.make("add", w43l, w2_43l)),
        Op.make(
            "add",
            Op.make("linear", x23q, w43l),
            Op.make("linear", x23q, w2_43l),
        ),
        "fire",
        "W,W2 (out,in) identical — the shape-eq disjunct",
    )
    w2_53l = V("W2_46", (5, 3))
    add(
        "weight_distribute_linear/out-dim-differs",
        "weight_distribute_linear",
        Op.make("linear", x23q, Op.make("add", w43l, w2_53l)),
        Op.make(
            "add",
            Op.make("linear", x23q, w43l),
            Op.make("linear", x23q, w2_53l),
        ),
        "fire",
        "out dims differ but the contraction axis W[-1]=3 agrees — "
        "the rank>=2 + dim-eq(-1) disjunct (the add itself is "
        "ill-broadcast; both engines must still agree on the guard)",
    )
    add(
        "weight_distribute_linear/contraction-mismatch",
        "weight_distribute_linear",
        Op.make("linear", x23q, Op.make("add", w43l, w2_44l)),
        Op.make(
            "add",
            Op.make("linear", x23q, w43l),
            Op.make("linear", x23q, w2_44l),
        ),
        "decline",
        "W[-1]=3 vs W2[-1]=4 — neither shape-eq nor dim-eq(-1) holds",
    )

    # naturality_scalar(+rev) — scalar(c) guard
    w34n, x42n = V("W46", (3, 4)), V("x46", (4, 2))
    add(
        "naturality_scalar/fire",
        "naturality_scalar",
        Op.make("matmul", w34n, Op.make("mul", x42n, C(2))),
        Op.make("mul", Op.make("matmul", w34n, x42n), C(2)),
        "fire",
        "scalar factor slides through the matmul",
    )
    c11 = V("c46", (1, 1))
    add(
        "naturality_scalar/non-scalar-c",
        "naturality_scalar",
        Op.make("matmul", w34n, Op.make("mul", x42n, c11)),
        Op.make("mul", Op.make("matmul", w34n, x42n), c11),
        "decline",
        "c=(1,1) broadcasts but is not scalar — the guard is "
        "scalar-only (conservative, not wrong)",
    )
    add(
        "naturality_scalar_rev/fire",
        "naturality_scalar_rev",
        Op.make("mul", Op.make("matmul", w34n, x42n), C(2)),
        Op.make("matmul", w34n, Op.make("mul", x42n, C(2))),
        "fire",
    )

    # matmul_div_distribute / div_add_factor — scalar(c) guards
    a34, v42 = V("a47", (3, 4)), V("v47", (4, 2))
    add(
        "matmul_div_distribute/fire",
        "matmul_div_distribute",
        Op.make("matmul", Op.make("div", a34, C(2)), v42),
        Op.make("div", Op.make("matmul", a34, v42), C(2)),
        "fire",
    )
    c4 = V("c47", (4,))
    add(
        "matmul_div_distribute/non-scalar-c",
        "matmul_div_distribute",
        Op.make("matmul", Op.make("div", a34, c4), v42),
        Op.make("div", Op.make("matmul", a34, v42), c4),
        "decline",
        "c=(4,) divides columns, not a scalar — veto",
    )
    x23r, y23r = V("x48", (2, 3)), V("y48", (2, 3))
    add(
        "div_add_factor/fire",
        "div_add_factor",
        Op.make(
            "add",
            Op.make("div", x23r, C(2)),
            Op.make("div", y23r, C(2)),
        ),
        Op.make("div", Op.make("add", x23r, y23r), C(2)),
        "fire",
    )
    c3 = V("c48", (3,))
    add(
        "div_add_factor/non-scalar-c",
        "div_add_factor",
        Op.make(
            "add", Op.make("div", x23r, c3), Op.make("div", y23r, c3)
        ),
        Op.make("div", Op.make("add", x23r, y23r), c3),
        "decline",
    )

    # linear_channel_scale_rev + assoc_linear_bias_rev — the _rev twins
    c53b = V("c49", (5,))
    add(
        "linear_channel_scale_rev/fire",
        "linear_channel_scale_rev",
        Op.make("linear", x35d, Op.make("mul", w45e, c53b)),
        Op.make("linear", Op.make("mul", x35d, c53b), w45e),
        "fire",
    )
    add(
        "assoc_linear_bias_rev/fire",
        "assoc_linear_bias_rev",
        Op.make(
            "add",
            Op.make(
                "linear",
                x23p,
                Op.make("matmul", B54, A43p),
                Op.make("matmul", B54, b14),
            ),
            b25,
        ),
        Op.make(
            "linear",
            Op.make("linear", x23p, A43p, b14),
            B54,
            b25,
        ),
        "fire",
        "the expand direction under the same premise guard",
    )
    return cases


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------

_OPTS = {"max_iterations": 60, "max_nodes": 50_000}
_COST = "uniform"


def _run_py(rule: Any, term: Any, rhs: Any) -> dict:
    g = EGraph()
    root = g.add_term(term)
    g.run([rule], root, **_OPTS)
    cost_fn = d._cost_fn(d.COST_PROFILES[_COST])
    best = g.extract_best(root, cost_fn)
    fired = sorted({e.rule for e in g.merge_log if e.rule is not None})
    eid = g.add_term(rhs)
    return {
        "fired": fired,
        "in_root": g.find(eid) == g.find(root),
        "best": best,
        "best_cost": cost_fn(best) if best is not None else None,
    }


def run_case(case: dict, rule: Any, rec: dict, bin_path: Path) -> dict:
    cost = d.COST_PROFILES[_COST]
    py = _run_py(rule, case["term"], case["rhs"])
    rs = d._run_rust(
        bin_path,
        rec,
        case["term"],
        case["rhs"],
        cost,
        _OPTS,
        case["id"],
    )

    law = case["law"]
    py_fired = law in py["fired"]
    rs_fired = law in rs.get("fired", [])
    expect = case["expect"]

    row = {
        "id": case["id"],
        "law": law,
        "expect": expect,
        "note": case.get("note", ""),
        "py_fired": py_fired,
        "rs_fired": rs_fired,
        "py_in_root": py["in_root"],
        "rs_in_root": rs["in_root"],
        "py_cost": py["best_cost"],
        "rs_cost": rs.get("cost"),
        "py_fired_all": py["fired"],
        "rs_fired_all": rs.get("fired", []),
    }

    if py_fired != rs_fired:
        row["verdict"] = (
            "fire-divergence" if py_fired else "decline-divergence"
        )
        row["detail"] = (
            f"python {'fired' if py_fired else 'declined'}, "
            f"rust {'fired' if rs_fired else 'declined'}"
        )
        return row
    if expect == "decline":
        row["verdict"] = (
            "decline-parity" if not py_fired else "over-fire"
        )
        if py_fired:
            row["detail"] = "guard failed to veto on BOTH engines"
        return row
    # expect == "fire"
    if not py_fired:
        row["verdict"] = "no-fire"
        row["detail"] = "guard declined on both engines (coverage gap)"
        return row
    sem = d._semverdict([case["term"], py["best"], rs.get("best")])
    if sem == "equal":
        row["verdict"] = "fire-parity"
    elif sem.startswith("diverge"):
        row["verdict"] = "semantic-diverge"
        row["detail"] = sem
    else:
        # Both engines fired and extracted — the numpy evaluator just
        # cannot denotate the terms (ill-typed broadcast, uncovered op).
        # Parity of *firing* still holds; report, don't flag.
        row["verdict"] = "fire-parity-unevaluated"
        row["detail"] = sem
    row["py_best"] = repr(py["best"])
    row["rs_best"] = repr(rs.get("best"))
    return row


def _shape_reading_laws(pybanks: dict) -> set[str]:
    """Portable laws whose cond tree reads shapes — the b9fe227 surface."""
    shape_ops = {
        # predicates
        "shaped",
        "concrete",
        "scalar",
        "uniform",
        "ones-but-last",
        "ones-before",
        "rank",
        "rank-eq",
        "shape-eq",
        "shape-compat",
        "dim-eq",
        "dim-compat",
        "dim-eq-const",
        "dim-eq-attr",
        "dim-mod",
        "bcast-into",
        "bcast-eq",
        "mm-shape-ok",
        "axes-last2",
        "axes-distinct",
        "axes-eq",
        "axis",
        "attr-cmp-dim",
        "flat-pair-unsq",
        "flat-map-unsq",
        "repeat-chain",
        "repeat-heads",
        # shape specs — a spec node anywhere in the cond means the
        # guard reads inferred shapes
        "unsq-out",
        "reshape-out",
        "mm-out",
        "bcast",
        "transpose-out",
        "select-out",
        "slice-out",
        "chunk-out",
        "getitem-out",
        "tail-block",
        "isel-out",
        "vshape",
        "shape",
        "dim",
    }

    def reads(x: Any) -> bool:
        if isinstance(x, (list, tuple)) and x:
            if x[0] in shape_ops:
                return True
            return any(reads(i) for i in x[1:])
        if isinstance(x, dict):
            return any(reads(v) for v in x.values())
        return False

    out = set()
    for name, rule in pybanks.items():
        cond = d.cond_to_data(getattr(rule, "cond", None))
        if cond and reads(cond):
            out.add(name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--law", help="run cases for a single law only")
    ap.add_argument("--id", help="run a single case id (substring)")
    ap.add_argument("--json", type=Path, help="write rows as JSONL")
    ap.add_argument(
        "--coverage",
        action="store_true",
        help="print the shape-reading-law coverage table",
    )
    args = ap.parse_args()

    pybanks = d._python_banks()
    erecs = d._engine_records()
    bin_path = d._bin()
    if bin_path is None:
        print(
            "eqsat-run not found — build it: "
            "cargo build --bin eqsat-run (engine-eqsat)",
            file=sys.stderr,
        )
        return 2
    print(f"== eqsat-run: {bin_path} ==")

    all_cases = _cases()
    if args.law:
        all_cases = [c for c in all_cases if c["law"] == args.law]
    if args.id:
        all_cases = [c for c in all_cases if args.id in c["id"]]

    if args.coverage:
        shape_laws = _shape_reading_laws(pybanks)
        exercised = {c["law"] for c in _cases()}
        portable = {
            r["law"]
            for r in d.census(pybanks, erecs)
            if r["status"] == "match"
        }
        covered = shape_laws & portable & exercised
        print(
            f"shape-reading portable laws: {len(shape_laws & portable)}; "
            f"exercised here: {len(covered)}"
        )
        for n in sorted(shape_laws & portable - exercised):
            print(f"  uncovered: {n}")

    rows = []
    for case in all_cases:
        rule = pybanks.get(case["law"])
        rec = erecs.get(case["law"])
        if rule is None or rec is None:
            row = {
                "id": case["id"],
                "law": case["law"],
                "verdict": "error",
                "detail": f"missing counterpart "
                f"(py={rule is not None} rs={rec is not None})",
            }
        else:
            try:
                row = run_case(case, rule, rec, bin_path)
            except Exception as e:
                row = {
                    "id": case["id"],
                    "law": case["law"],
                    "verdict": "error",
                    "detail": f"{type(e).__name__}: {e}",
                }
        rows.append(row)
        flag = "" if "parity" in row.get("verdict", "") else "  <<<"
        print(
            f"{row['id']:44s} {row.get('expect', '?'):7s} "
            f"py={row.get('py_fired')} rs={row.get('rs_fired')} "
            f"root=({row.get('py_in_root')},{row.get('rs_in_root')}) "
            f"{row.get('verdict')}{flag}"
        )
        if row.get("detail"):
            print(f"    {row['detail']}")

    if args.json:
        with args.json.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")

    bad = [
        r
        for r in rows
        if r.get("verdict")
        not in (
            "fire-parity",
            "fire-parity-unevaluated",
            "decline-parity",
            "no-fire",
        )
    ]
    gaps = [r for r in rows if r.get("verdict") == "no-fire"]
    unev = [
        r for r in rows if r.get("verdict") == "fire-parity-unevaluated"
    ]
    print(
        f"\n{len(rows)} cases: "
        f"{sum(1 for r in rows if r.get('verdict', '').endswith('parity'))} parity, "
        f"{len(unev)} unevaluated, {len(gaps)} no-fire, {len(bad)} flagged"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

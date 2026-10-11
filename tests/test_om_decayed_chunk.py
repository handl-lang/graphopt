# ruff: noqa: RUF002, RUF003
"""decay + the decayed-attention chunk laws — the structure made value.

``decay(γ, t=T)`` names the causal decay triangle ``tril(γ^{i−j})`` as
a generator, so its block structure is equational rather than a
spelled ``arange/pow/tril`` subgraph.  On top of it the
``om_decayed_chunk_*`` splits decompose ``(S ⊙ D) @ V`` over
concatenated operands along BOTH matmul axes:

* ROWS — ``om_decayed(cat_T(s1,s2), d, v)`` splits into per-block
  calls, the decay sliced on its image of the row axis ("slice") or
  reused when it broadcasts along it ("reuse": extent 1, a shared
  per-chunk decay under equal blocks, or no such axis).  A ``_cat``
  variant pairs operands that both arrive concat'd.
* KEYS — ``om_decayed(s, d, cat_K(v1,v2))`` splits the contraction
  into an ``add`` of key-block calls (s split at −1 or arriving
  already concat'd), ``d`` sliced/reused on the key axis.

Composed on ``om_decayed(cat_T(s1,s2), decay(γ,T), cat_K(v1,v2))``
they yield the four-block chunked form: the diagonal blocks fold
back to ``decay(γ,T_i)`` by ``decay_block_diag`` (the intra terms)
and the (1,0) block is the cross-decay the recurrence carries — the
(0,1) zero block stays materialised (a typed ``zeros`` cannot be
minted: leaves carry no dtype).

Covered:

* binding — ``decay`` evaluates ``tril(γ^{i−j})``, scalar and
  per-head-batched, fp64-bitwise against the spelled formula;
* decay_lift — the export-spelled ``tril(pow(γ, clamp(i−j,0)))``
  folds into ``decay(γ,T)``; every guard (same arange, step 1,
  unsqueeze orientation, clamp bound 0) declines honestly;
* chunk laws — each fires on well-typed inputs and evaluates
  fp64-equal to the unchunked term (reassociated matmuls are NOT
  bitwise — kernel scheduling moves the last ulp, so allclose);
* decay_block_diag — the (i,i) split regenerates ``decay(γ,T_i)``;
* e2e — the chunked spelled term saturates under OM_LAWS, the root
  class carries both the fused element and the chunked composite,
  extraction honestly prefers the fused form (identical FLOPs,
  fewer launches), and the chunked member lowers+verifies;
* declines — wrong-axis concats, non-concrete extents, ill-typed
  blocks and mismatched pairs all veto, in the check hooks and in
  the e-graph;
* probe — fp64 timing of full vs chunked delivery; a report, not a
  gate (chunking this element is launch-neutral at best — the win
  needs the cross block's rank structure, see om.py's note).
"""

import time

import catopt_carriers.om as OM
import pytest
import torch
from catopt_core import meta
from catopt_core.cost import flops_cost, launch_aware_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import IR, Const, Op, TensorType, Var
from catopt_core.typing import _INVALID, _shape_of
from catopt_torch.adapters import TorchSink
from catopt_torch.torch_bridge import _IR_TO_TORCH, ir_to_torch_module

torch.manual_seed(0)


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _v(name: str, *shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _rand(*shape, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(9000 + seed)
    return torch.randn(*shape, dtype=torch.float64, generator=g)


def _eval(term, env):
    return meta._eval_term(term, env)


def _decay_ref(g: torch.Tensor, t: int) -> torch.Tensor:
    pos = torch.arange(t, dtype=g.dtype, device=g.device)
    return torch.tril(
        g ** (pos.unsqueeze(-1) - pos.unsqueeze(-2)).clamp_min(0)
    )


def _spelled_decay(g, t: int) -> Op:
    """The export spelling: tril(γ^{clamp_min(i−j, 0)})."""
    a = Op.make("arange", arg0=t)
    diff = Op.make(
        "clamp_min",
        Op.make(
            "sub",
            Op.make("unsqueeze", a, dim=1),
            Op.make("unsqueeze", a, dim=0),
        ),
        min=0,
    )
    return Op.make("tril", Op.make("pow", g, diff))


def _ir(term, inputs):
    return IR(
        root=term,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params={},
    )


def _close(a, b) -> bool:
    return torch.allclose(a, b, atol=1e-12, rtol=1e-12)


#: A leaf that is not a term at all — ``_shape_of`` reports ``None``.
RAW = "raw_leaf"

_ROWS_SLICE, _ROWS_REUSE = (
    OM.OM_DECAYED_CHUNK_LAWS[0],
    OM.OM_DECAYED_CHUNK_LAWS[1],
)
_ROWS_CAT = OM.OM_DECAYED_CHUNK_LAWS[2]
_KEYS_SLICE, _KEYS_REUSE = (
    OM.OM_DECAYED_CHUNK_LAWS[3],
    OM.OM_DECAYED_CHUNK_LAWS[4],
)
_KEYS_CAT_SLICE, _KEYS_CAT_REUSE = (
    OM.OM_DECAYED_CHUNK_LAWS[5],
    OM.OM_DECAYED_CHUNK_LAWS[6],
)


# ---------------------------------------------------------------------------
#  binding — the executor IS the spelled tril(γ^{i−j})
# ---------------------------------------------------------------------------


def test_binding_is_the_decay_triangle():
    """``decay(γ, t=T)`` == ``tril(γ^{i−j})`` — fp64-identical."""
    g = torch.tensor(0.9, dtype=torch.float64)
    out = _IR_TO_TORCH["decay"](g, t=8)
    assert torch.equal(out, _decay_ref(g, 8))


def test_binding_broadcasts_per_head_rates():
    """γ's dims are leading batch dims: (H,) → (H,T,T)."""
    g = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float64)
    out = _IR_TO_TORCH["decay"](g, t=5)
    assert out.shape == (3, 5, 5)
    assert torch.equal(
        out, _decay_ref(g.unsqueeze(-1).unsqueeze(-1), 5)
    )


def test_binding_accepts_python_scalars():
    """A Const leaf arrives as a float — ``torch.as_tensor`` carries it."""
    out = _IR_TO_TORCH["decay"](0.5, t=4)
    assert out.shape == (4, 4)
    ref = _decay_ref(torch.tensor(0.5), 4)
    assert torch.allclose(out, ref)


def test_shape_rule_reports_the_triangle():
    """Scalar γ → (T,T); a per-head (H,) rate → (H,T,T)."""
    assert _shape_of(Op.make("decay", _v("g"), t=8)) == (8, 8)
    assert _shape_of(Op.make("decay", _v("g", 3), t=8)) == (3, 8, 8)


def test_shape_rule_handles_bad_attrs():
    """A missing/non-int ``t`` is unknown; a negative one is _INVALID."""
    g = _v("g")
    assert _shape_of(Op.make("decay", g)) is None
    assert _shape_of(Op.make("decay", g, t="x")) is None
    assert _shape_of(Op.make("decay", g, t=-2)) is _INVALID


def test_flops_price_the_spelled_subgraph():
    """decay costs the spelled sub+clamp+pow+tril: 5·T²."""
    assert flops_cost(Op.make("decay", _v("g"), t=8)) == 5 * 64


# ---------------------------------------------------------------------------
#  decay_lift — the spelled subgraph folds into the generator
# ---------------------------------------------------------------------------


def test_decay_lift_fires_and_evals():
    """tril(γ^{clamp(i−j,0)}) → decay(γ, t=T), numerically equal."""
    g = _v("g")
    term = _spelled_decay(g, 8)
    out = meta.apply_rewrite_at(OM.DECAY_LIFT, term, ())
    assert out is not None
    assert out.op == "decay" and out.attrs == {"t": 8}
    env = {g: torch.tensor(0.9, dtype=torch.float64)}
    assert torch.equal(_eval(out, env), _eval(term, env))


def test_decay_lift_fires_on_export_spelling():
    """The zoo's ChunkedRetention decay folds on its real export."""
    from catopt_discovery.zoo import _ChunkedRetention
    from catopt_torch.torch_bridge import export_to_ir

    torch.manual_seed(0)
    model = _ChunkedRetention(16).double()
    x = torch.randn(2, 8, 16, dtype=torch.float64)
    ir, _leaves = export_to_ir(model, (x,))

    def find(t, op):
        if isinstance(t, Op):
            if t.op == op:
                return t
            for a in t.args:
                r = find(a, op)
                if r is not None:
                    return r
        return None

    out = meta.apply_rewrite_at(
        OM.DECAY_LIFT, find(ir.root, "tril"), ()
    )
    assert (
        out is not None and out.op == "decay" and out.attrs == {"t": 8}
    )


def test_decay_lift_diagonal_zero_spelling():
    """``tril(·, diagonal=0)`` folds too — the attributed spelling."""
    g = _v("g")
    a = Op.make("arange", arg0=4)
    diff = Op.make(
        "clamp_min",
        Op.make(
            "sub",
            Op.make("unsqueeze", a, dim=1),
            Op.make("unsqueeze", a, dim=0),
        ),
        min=0,
    )
    term = Op.make("tril", Op.make("pow", g, diff), diagonal=0)
    out = meta.apply_rewrite_at(OM.DECAY_LIFT_DIAG, term, ())
    assert out is not None and out.attrs == {"t": 4}


def test_decay_lift_const_arange_spelling():
    """Const-operand arange bounds lift just like attr bounds."""
    g = _v("g")
    a = Op.make("arange", Const(0), Const(6))
    diff = Op.make(
        "clamp_min",
        Op.make(
            "sub",
            Op.make("unsqueeze", a, dim=1),
            Op.make("unsqueeze", a, dim=0),
        ),
        min=0,
    )
    term = Op.make("tril", Op.make("pow", g, diff))
    out = meta.apply_rewrite_at(OM.DECAY_LIFT, term, ())
    assert out is not None and out.attrs == {"t": 6}


def test_decay_lift_declines_non_unit_steps_and_mismatched_ranges():
    """Step ≠ 1 or different bounds: ``p_i − p_j ≠ i − j`` — decline."""
    g = _v("g")
    a1 = Op.make("arange", arg0=0, arg1=8, arg2=2)
    a2 = Op.make("arange", arg0=8)
    diff = Op.make(
        "clamp_min",
        Op.make(
            "sub",
            Op.make("unsqueeze", a1, dim=1),
            Op.make("unsqueeze", a1, dim=0),
        ),
        min=0,
    )
    assert (
        meta.apply_rewrite_at(
            OM.DECAY_LIFT, Op.make("tril", Op.make("pow", g, diff)), ()
        )
        is None
    )
    diff2 = Op.make(
        "clamp_min",
        Op.make(
            "sub",
            Op.make("unsqueeze", a1, dim=1),
            Op.make("unsqueeze", a2, dim=0),
        ),
        min=0,
    )
    assert (
        meta.apply_rewrite_at(
            OM.DECAY_LIFT, Op.make("tril", Op.make("pow", g, diff2)), ()
        )
        is None
    )


def test_decay_lift_declines_swapped_unsqueezes():
    """sub(row, col) is j−i, not i−j — a different triangle."""
    g = _v("g")
    a = Op.make("arange", arg0=4)
    diff = Op.make(
        "clamp_min",
        Op.make(
            "sub",
            Op.make("unsqueeze", a, dim=0),
            Op.make("unsqueeze", a, dim=1),
        ),
        min=0,
    )
    assert (
        meta.apply_rewrite_at(
            OM.DECAY_LIFT, Op.make("tril", Op.make("pow", g, diff)), ()
        )
        is None
    )


def test_decay_lift_check_covers_the_residual_guards():
    """The check's own branches: attrs, bounds, non-arange leaves."""
    g = _v("g")
    a = Op.make("arange", arg0=8)
    ok = {
        "a1": a,
        "a2": a,
        "g": g,
        "$attr:U1": 1,
        "$attr:U2": 0,
        "$attr:CM": 0,
    }
    assert OM.DECAY_LIFT.check(ok)
    for delta in (
        {"$attr:U1": "x"},
        {"$attr:U2": None},
        {"$attr:CM": "x"},
        {"$attr:CM": 1},
    ):
        assert not OM.DECAY_LIFT.check({**ok, **delta})
    # a non-arange a1: _arange_params declines
    assert not OM.DECAY_LIFT.check({**ok, "a1": _v("p", 8)})
    # a bare arange() mint: no bounds at all → derive declines
    assert not OM.DECAY_LIFT.check(
        {**ok, "a1": Op.make("arange"), "a2": Op.make("arange")}
    )
    # non-numeric arange bound → _arange_params declines
    assert not OM.DECAY_LIFT.check(
        {**ok, "a1": Op.make("arange", arg0="x")}
    )
    # derive: no arange → None; a declining extent → None
    assert OM.DECAY_LIFT.derive({"a1": _v("p", 8)}) is None
    assert (
        OM.DECAY_LIFT.derive({"a1": Op.make("arange", arg0=0)}) is None
    )
    assert OM.DECAY_LIFT.derive({"a1": a}) == {"$attr:T": 8}


def test_arange_params_variants():
    """The bound reader: Const args, attrs, and the declines."""
    assert OM._arange_params(_v("p", 4)) is None
    assert OM._arange_params(Op.make("arange")) is None
    assert OM._arange_params(Op.make("arange", arg0="x")) is None
    assert OM._arange_params(Op.make("arange", Const(5))) == (0, 5, 1)
    assert OM._arange_params(Op.make("arange", arg0=5)) == (0, 5, 1)
    assert OM._arange_params(Op.make("arange", Const(2), Const(9))) == (
        2,
        9,
        1,
    )
    assert OM._arange_params(
        Op.make("arange", arg0=1, arg1=9, arg2=2)
    ) == (1, 9, 2)
    assert (
        OM._arange_params(
            Op.make("arange", arg0=1, arg1=9, arg2=2, arg3=3)
        )
        is None
    )


# ---------------------------------------------------------------------------
#  row split — om_decayed over concat'd queries
# ---------------------------------------------------------------------------


def _rows_term(s1, s2, d, v, dim=-2):
    return Op.make(
        "om_decayed", Op.make("concat", s1, s2, dim=dim), d, v
    )


def test_rows_slice_fires_and_evals():
    """(cat_T(s1,s2) ⊙ d) @ v = cat_T((s1⊙d1)v, (s2⊙d2)v)."""
    s1, s2 = _v("s1", 2, 8, 16), _v("s2", 2, 8, 16)
    g, v = _v("g"), _v("v", 2, 16, 4)
    d = Op.make("decay", g, t=16)
    term = _rows_term(s1, s2, d, v)
    out = meta.apply_rewrite_at(_ROWS_SLICE, term, ())
    assert out is not None and out.op == "concat"
    assert all(a.op == "om_decayed" for a in out.args)
    assert out.args[0].args[1].op == "split"
    env = {
        s1: _rand(2, 8, 16, seed=1),
        s2: _rand(2, 8, 16, seed=2),
        g: torch.tensor(0.9, dtype=torch.float64),
        v: _rand(2, 16, 4, seed=3),
    }
    assert _close(_eval(out, env), _eval(term, env))


def test_rows_reuse_fires_on_broadcast_decay():
    """A decay that broadcasts on the row axis is reused, not sliced."""
    s1, s2 = _v("s1", 2, 8, 16), _v("s2", 2, 8, 16)
    d, v = _v("d", 16), _v("v", 2, 16, 4)
    term = _rows_term(s1, s2, d, v)
    out = meta.apply_rewrite_at(_ROWS_REUSE, term, ())
    assert out is not None
    assert out.args[0].args[1] is d and out.args[1].args[1] is d
    env = {
        s1: _rand(2, 8, 16, seed=4),
        s2: _rand(2, 8, 16, seed=5),
        d: _rand(16, seed=6),
        v: _rand(2, 16, 4, seed=7),
    }
    assert _close(_eval(out, env), _eval(term, env))


def test_rows_shared_local_decay_is_reuse():
    """Equal chunks sharing one (C,C) decay reuse it verbatim."""
    s1, s2 = _v("s1", 2, 8, 8), _v("s2", 2, 8, 8)
    g, v = _v("g"), _v("v", 2, 8, 4)
    d = Op.make("decay", g, t=8)
    term = _rows_term(s1, s2, d, v)
    out = meta.apply_rewrite_at(_ROWS_REUSE, term, ())
    assert out is not None and out.args[0].args[1] is d


def test_rows_mode_guards_are_mutual():
    """Slice declines a broadcasting decay; reuse declines a sliced one."""
    s1, s2 = _v("s1", 2, 8, 16), _v("s2", 2, 8, 16)
    g, v = _v("g"), _v("v", 2, 16, 4)
    full = _rows_term(s1, s2, Op.make("decay", g, t=16), v)
    assert meta.apply_rewrite_at(_ROWS_REUSE, full, ()) is None
    vec = _rows_term(s1, s2, _v("d", 16), v)
    assert meta.apply_rewrite_at(_ROWS_SLICE, vec, ()) is None


def test_rows_slice_uneven_split():
    """T=14 split (8,6): uneven blocks keep the decomposition exact."""
    s1, s2 = _v("s1", 2, 8, 14), _v("s2", 2, 6, 14)
    g, v = _v("g"), _v("v", 2, 14, 4)
    d = Op.make("decay", g, t=14)
    term = _rows_term(s1, s2, d, v)
    out = meta.apply_rewrite_at(_ROWS_SLICE, term, ())
    assert out is not None
    env = {
        s1: _rand(2, 8, 14, seed=40),
        s2: _rand(2, 6, 14, seed=41),
        g: torch.tensor(0.85, dtype=torch.float64),
        v: _rand(2, 14, 4, seed=42),
    }
    assert _close(_eval(out, env), _eval(term, env))


def test_rows_slice_iterates_to_three_blocks():
    """Nested concats split recursively — T=18 as (8,6)+4."""
    s1, s2, s3 = (
        _v("s1", 2, 8, 18),
        _v("s2", 2, 6, 18),
        _v("s3", 2, 4, 18),
    )
    g, v = _v("g"), _v("v", 2, 18, 4)
    d = Op.make("decay", g, t=18)
    inner = Op.make("concat", s1, s2, dim=-2)
    term = Op.make(
        "om_decayed", Op.make("concat", inner, s3, dim=-2), d, v
    )
    first = meta.apply_rewrite_at(_ROWS_SLICE, term, ())
    assert first is not None and first.op == "concat"
    second = meta.apply_rewrite_at(_ROWS_SLICE, first, (0,))
    assert second is not None
    blocks = (*second.args[0].args, second.args[1])
    assert len(blocks) == 3
    assert all(b.op == "om_decayed" for b in blocks)
    env = {
        s1: _rand(2, 8, 18, seed=43),
        s2: _rand(2, 6, 18, seed=44),
        s3: _rand(2, 4, 18, seed=45),
        g: torch.tensor(0.9, dtype=torch.float64),
        v: _rand(2, 18, 4, seed=46),
    }
    assert _close(_eval(second, env), _eval(term, env))


def test_rows_declines_wrong_axis_and_ill_typed():
    """Cat on the key axis, mismatched batches, dead contraction."""
    s1, s2 = _v("s1", 2, 8, 16), _v("s2", 2, 8, 16)
    g, v = _v("g"), _v("v", 2, 16, 4)
    d = Op.make("decay", g, t=16)
    # cat on -1 (the key axis) — the row split must decline.
    assert (
        meta.apply_rewrite_at(
            _ROWS_SLICE, _rows_term(s1, s2, d, v, dim=-1), ()
        )
        is None
    )
    # s2 batch mismatch — an ill-typed cat, vetoed by the plan.
    bad = _rows_term(s1, _v("s2b", 3, 8, 16), d, v)
    assert meta.apply_rewrite_at(_ROWS_SLICE, bad, ()) is None
    # d's key extent broadcasts nowhere — per-block mul is ill-typed.
    bad_d = _rows_term(s1, s2, _v("d", 16, 4), v)
    assert meta.apply_rewrite_at(_ROWS_SLICE, bad_d, ()) is None
    # contraction dead: v's key axis ≠ the score's last axis.
    bad_v = _rows_term(s1, s2, d, _v("v2", 2, 3, 4))
    assert meta.apply_rewrite_at(_ROWS_SLICE, bad_v, ()) is None


def test_rows_check_covers_residual_guards():
    """Direct check/derive calls for the uncovered guard arms."""
    s1, s2 = _v("s1", 2, 8, 16), _v("s2", 2, 8, 16)
    d, v = _v("d", 16, 16), _v("v", 2, 16, 4)
    ok = {"s1": s1, "s2": s2, "d": d, "v": v, "$attr:D": -2}
    assert _ROWS_SLICE.check(ok)
    assert _ROWS_REUSE.check({**ok, "d": _v("d", 16)})
    for delta in (
        {"s1": RAW},
        {"v": RAW},
        {"d": RAW},
        {"$attr:D": "x"},
        {"s2": _v("s2", 2, 8, 16, 9)},
    ):
        assert not _ROWS_SLICE.check({**ok, **delta})
    # rank-1 scores have no row axis.
    assert not _ROWS_SLICE.check(
        {**ok, "s1": _v("s1", 16), "s2": _v("s2", 16)}
    )
    # derive: no concat plan → None; a reuse shape → None; slice → map.
    assert _ROWS_SLICE.derive({**ok, "d": RAW}) is None
    assert _ROWS_SLICE.derive({**ok, "d": _v("d", 16)}) is None
    assert _ROWS_SLICE.derive(ok) == {
        "$attr:SZ": (8, 8),
        "$attr:MD": 0,
    }


def test_rows_cat_variant_fires():
    """Score and decay both concat'd on the row axis pair up."""
    s1, s2 = _v("s1", 2, 8, 16), _v("s2", 2, 8, 16)
    d1, d2 = _v("d1", 8, 16), _v("d2", 8, 16)
    v = _v("v", 2, 16, 4)
    term = Op.make(
        "om_decayed",
        Op.make("concat", s1, s2, dim=-2),
        Op.make("concat", d1, d2, dim=-2),
        v,
    )
    out = meta.apply_rewrite_at(_ROWS_CAT, term, ())
    assert out is not None
    assert out.args[0].args[1] is d1 and out.args[1].args[1] is d2
    env = {
        s1: _rand(2, 8, 16, seed=8),
        s2: _rand(2, 8, 16, seed=9),
        d1: _rand(8, 16, seed=10),
        d2: _rand(8, 16, seed=11),
        v: _rand(2, 16, 4, seed=12),
    }
    assert _close(_eval(out, env), _eval(term, env))


def test_rows_cat_declines_off_row_cats():
    """Both cats on the key axis agree — but it isn't the row axis."""
    s1, s2 = _v("s1", 2, 8, 8), _v("s2", 2, 8, 8)
    d1, d2 = _v("d1", 2, 8, 8), _v("d2", 2, 8, 8)
    v = _v("v", 2, 16, 4)
    term = Op.make(
        "om_decayed",
        Op.make("concat", s1, s2, dim=-1),
        Op.make("concat", d1, d2, dim=-1),
        v,
    )
    assert meta.apply_rewrite_at(_ROWS_CAT, term, ()) is None
    # cats on different axes entirely — _check_cat_pair declines.
    term2 = Op.make(
        "om_decayed",
        Op.make("concat", s1, s2, dim=-2),
        Op.make("concat", d1, d2, dim=-1),
        v,
    )
    assert meta.apply_rewrite_at(_ROWS_CAT, term2, ()) is None
    # an ill-typed block: v's key axis dead.
    term3 = Op.make(
        "om_decayed",
        Op.make("concat", s1, s2, dim=-2),
        Op.make("concat", d1, d2, dim=-2),
        _v("v2", 2, 3, 4),
    )
    assert meta.apply_rewrite_at(_ROWS_CAT, term3, ()) is None
    # helper arms: rank/off-axis mismatches in the cat-ok gate.
    assert not OM._outs_cat_ok((2, 8, 4), (2, 8, 4, 1), 1)
    assert not OM._outs_cat_ok((2, 8, 4), (3, 8, 4), 1)
    assert OM._outs_cat_ok((2, 8, 4), (2, 8, 4), 1)
    assert OM._block_out_ok((4, 8), (4, 8), (8, 3)) == (4, 3)
    assert OM._block_out_ok((2, 3), (4, 5), (3, 4)) is None
    assert OM._block_out_ok((4, 8), (4, 8), (3, 5)) is None


# ---------------------------------------------------------------------------
#  key split — om_decayed over concat'd values
# ---------------------------------------------------------------------------


def _keys_term(s, d, v1, v2, dim=-2):
    return Op.make(
        "om_decayed", s, d, Op.make("concat", v1, v2, dim=dim)
    )


def test_keys_slice_fires_and_evals():
    """(s ⊙ d) @ cat(v1,v2) = (s_L⊙d_L)v1 + (s_R⊙d_R)v2."""
    s, g = _v("s", 2, 16, 16), _v("g")
    d = Op.make("decay", g, t=16)
    v1, v2 = _v("v1", 2, 8, 4), _v("v2", 2, 8, 4)
    term = _keys_term(s, d, v1, v2)
    out = meta.apply_rewrite_at(_KEYS_SLICE, term, ())
    assert out is not None and out.op == "add"
    for a in out.args:
        assert a.op == "om_decayed"
        assert a.args[0].op == "split" and a.args[1].op == "split"
    env = {
        s: _rand(2, 16, 16, seed=13),
        g: torch.tensor(0.9, dtype=torch.float64),
        v1: _rand(2, 8, 4, seed=14),
        v2: _rand(2, 8, 4, seed=15),
    }
    assert _close(_eval(out, env), _eval(term, env))


def test_keys_reuse_fires_on_broadcast_decay():
    """A decay broadcasting on the key axis is reused in both blocks."""
    s, d = _v("s", 2, 8, 16), _v("d", 8, 1)
    v1, v2 = _v("v1", 2, 8, 4), _v("v2", 2, 8, 4)
    term = _keys_term(s, d, v1, v2)
    out = meta.apply_rewrite_at(_KEYS_REUSE, term, ())
    assert out is not None
    assert out.args[0].args[1] is d and out.args[1].args[1] is d
    env = {
        s: _rand(2, 8, 16, seed=16),
        d: _rand(8, 1, seed=17),
        v1: _rand(2, 8, 4, seed=18),
        v2: _rand(2, 8, 4, seed=19),
    }
    assert _close(_eval(out, env), _eval(term, env))


def test_keys_reuse_fires_on_scalar_decay():
    """A () decay broadcasts everywhere — the degenerate reuse."""
    s = _v("s", 2, 8, 16)
    v1, v2 = _v("v1", 2, 8, 4), _v("v2", 2, 8, 4)
    term = _keys_term(s, _v("d"), v1, v2)
    assert meta.apply_rewrite_at(_KEYS_REUSE, term, ()) is not None


def test_keys_mode_guards_are_mutual():
    """Slice declines a broadcasting decay; reuse declines a full one."""
    s, g = _v("s", 2, 8, 16), _v("g")
    v1, v2 = _v("v1", 2, 8, 4), _v("v2", 2, 8, 4)
    full = _keys_term(s, Op.make("decay", g, t=16), v1, v2)
    assert meta.apply_rewrite_at(_KEYS_REUSE, full, ()) is None
    brd = _keys_term(s, _v("d", 8, 1), v1, v2)
    assert meta.apply_rewrite_at(_KEYS_SLICE, brd, ()) is None


def test_keys_cat_variants_fire():
    """Scores already key-blocked: (cat(s1,s2) ⊙ d) @ cat(v1,v2)."""
    s1, s2 = _v("s1", 2, 16, 8), _v("s2", 2, 16, 8)
    g = _v("g")
    d = Op.make("decay", g, t=16)
    v1, v2 = _v("v1", 2, 8, 4), _v("v2", 2, 8, 4)
    term = Op.make(
        "om_decayed",
        Op.make("concat", s1, s2, dim=-1),
        d,
        Op.make("concat", v1, v2, dim=-2),
    )
    out = meta.apply_rewrite_at(_KEYS_CAT_SLICE, term, ())
    assert out is not None
    assert out.args[0].args[0] is s1 and out.args[1].args[0] is s2
    env = {
        s1: _rand(2, 16, 8, seed=20),
        s2: _rand(2, 16, 8, seed=21),
        g: torch.tensor(0.9, dtype=torch.float64),
        v1: _rand(2, 8, 4, seed=22),
        v2: _rand(2, 8, 4, seed=23),
    }
    assert _close(_eval(out, env), _eval(term, env))
    # the reuse companion — a broadcast decay pairs without slicing.
    term_b = Op.make(
        "om_decayed",
        Op.make("concat", s1, s2, dim=-1),
        _v("d", 16, 1),
        Op.make("concat", v1, v2, dim=-2),
    )
    out_b = meta.apply_rewrite_at(_KEYS_CAT_REUSE, term_b, ())
    assert out_b is not None and out_b.args[0].args[1].name == "d"


def test_keys_declines_wrong_axes_and_extents():
    """Cat'd on the wrong axis, unknown or mismatched extents veto."""
    s, g = _v("s", 2, 8, 16), _v("g")
    d = Op.make("decay", g, t=16)
    v1, v2 = _v("v1", 2, 8, 4), _v("v2", 2, 8, 4)
    # v cat on the feature axis, not the key axis.
    assert (
        meta.apply_rewrite_at(
            _KEYS_SLICE, _keys_term(s, d, v1, v2, dim=-1), ()
        )
        is None
    )
    # score's last axis ≠ K1+K2.
    bad_s = _keys_term(_v("s2", 2, 8, 12), d, v1, v2)
    assert meta.apply_rewrite_at(_KEYS_SLICE, bad_s, ()) is None
    # decay's key extent is neither 1 nor K1+K2.
    bad_d = _keys_term(s, _v("d", 8, 8), v1, v2)
    assert meta.apply_rewrite_at(_KEYS_SLICE, bad_d, ()) is None
    # unknown key extents — split sizes cannot be derived.
    bad_v = _keys_term(s, d, _v("v3", 2, None, 4), v2)
    assert meta.apply_rewrite_at(_KEYS_SLICE, bad_v, ()) is None
    # ill-typed batch: d's leading dims don't broadcast s's.
    bad_db = _keys_term(s, _v("d", 7, 8, 16), v1, v2)
    assert meta.apply_rewrite_at(_KEYS_SLICE, bad_db, ()) is None


def test_keys_cat_declines_pair_mismatch():
    """The cat variant: key-block sizes must pair s_i with v_i."""
    s1, s2 = _v("s1", 2, 16, 8), _v("s2", 2, 16, 8)
    d, v1, v2 = _v("d", 16, 16), _v("v1", 2, 8, 4), _v("v2", 2, 8, 4)

    def term(sd=-1, vd=-2, s2v=None, v2v=None):
        return Op.make(
            "om_decayed",
            Op.make("concat", s1, s2v or s2, dim=sd),
            d,
            Op.make("concat", v1, v2v or v2, dim=vd),
        )

    # d's key extent is K1+K2 → the slice law fires.
    assert (
        meta.apply_rewrite_at(_KEYS_CAT_SLICE, term(), ()) is not None
    )
    # s cat on the wrong axis.
    assert (
        meta.apply_rewrite_at(_KEYS_CAT_SLICE, term(sd=-2), ()) is None
    )
    # s2's key extent ≠ v2's.
    assert (
        meta.apply_rewrite_at(
            _KEYS_CAT_SLICE, term(s2v=_v("s2b", 2, 16, 5)), ()
        )
        is None
    )
    # off-axis score mismatch — an ill-typed cat.
    assert (
        meta.apply_rewrite_at(
            _KEYS_CAT_SLICE, term(s2v=_v("s2c", 3, 16, 8)), ()
        )
        is None
    )
    # v cat on the feature axis, not the key axis.
    assert (
        meta.apply_rewrite_at(_KEYS_CAT_SLICE, term(vd=-1), ()) is None
    )


def test_keys_check_covers_residual_guards():
    """Direct calls for the plan/check/derive arms the laws can't hit."""
    s1, s2 = _v("s1", 2, 8, 8), _v("s2", 2, 8, 8)
    d = _v("d", 8, 16)
    v1, v2 = _v("v1", 2, 8, 4), _v("v2", 2, 8, 4)
    ok = {
        "s1": s1,
        "s2": s2,
        "d": d,
        "v1": v1,
        "v2": v2,
        "$attr:SD": -1,
        "$attr:VD": -2,
    }
    assert _KEYS_CAT_REUSE.check({**ok, "d": _v("d", 8, 1)})
    for delta in (
        {"v1": RAW},
        {"$attr:VD": "x"},
        {"v2": _v("v2", 2, 8)},
        {"v2": _v("v2", 3, 8, 4)},
        {"$attr:SD": "x"},
        {"s2": RAW},
        {"d": RAW},
    ):
        assert not _KEYS_CAT_SLICE.check({**ok, **delta})
    # s-whole variant guard arms.
    okw = {
        "s": _v("s", 2, 8, 16),
        **{k: ok[k] for k in ok if k != "s1" and k != "s2"},
    }
    del okw["$attr:SD"]
    assert _KEYS_SLICE.check({**okw})
    assert not _KEYS_SLICE.check({**okw, "s": RAW})
    # a scalar () score has no key axis to split.
    assert not _KEYS_SLICE.check({**okw, "s": Const(1.0)})
    # derive: a bad bound → None; a good one → the (K1,K2) map.
    assert _KEYS_SLICE.derive({**okw, "v1": RAW}) is None
    assert _KEYS_SLICE.derive(okw) == {"$attr:SZ": (8, 8)}
    assert _KEYS_CAT_SLICE.derive(ok) == {"$attr:SZ": (8, 8)}


# ---------------------------------------------------------------------------
#  decay_block_diag — the (i,i) block is the chunk's local decay
# ---------------------------------------------------------------------------


def _block_of_decays(g, t, i):
    return Op.make(
        "split",
        Op.make(
            "split",
            Op.make("decay", g, t=t),
            sizes=(t // 2, t - t // 2),
            dim=-2,
            index=i,
        ),
        sizes=(t // 2, t - t // 2),
        dim=-1,
        index=i,
    )


def test_block_diag_fires_and_evals():
    """split_i(split_i(decay(γ,T))) regenerates decay(γ,T_i)."""
    g = _v("g")
    for i, ti in ((0, 8), (1, 8)):
        term = _block_of_decays(g, 16, i)
        out = meta.apply_rewrite_at(OM.DECAY_BLOCK_DIAG, term, ())
        assert out is not None
        assert out.op == "decay" and out.attrs == {"t": ti}
        env = {g: torch.tensor(0.9, dtype=torch.float64)}
        assert torch.equal(_eval(out, env), _eval(term, env))


def test_block_diag_uneven_split():
    """An uneven (10,6) split: block extents derive individually."""
    g = _v("g")
    term = Op.make(
        "split",
        Op.make(
            "split",
            Op.make("decay", g, t=16),
            sizes=(10, 6),
            dim=-2,
            index=1,
        ),
        sizes=(10, 6),
        dim=-1,
        index=1,
    )
    out = meta.apply_rewrite_at(OM.DECAY_BLOCK_DIAG, term, ())
    assert out is not None and out.attrs == {"t": 6}
    env = {g: torch.tensor(0.9, dtype=torch.float64)}
    assert torch.equal(_eval(out, env), _eval(term, env))


def test_block_diag_declines_non_diagonal():
    """Off-diagonal blocks and wrong-axis splits decline."""
    g = _v("g")
    # different block indices — pattern-level no-match (one metavar I).
    cross = Op.make(
        "split",
        Op.make(
            "split",
            Op.make("decay", g, t=16),
            sizes=(8, 8),
            dim=-2,
            index=1,
        ),
        sizes=(8, 8),
        dim=-1,
        index=0,
    )
    assert meta.apply_rewrite_at(OM.DECAY_BLOCK_DIAG, cross, ()) is None
    # check-level declines: bad sizes, bad index, wrong t, wrong axes.
    ok = {
        "g": g,
        "$attr:SZ": (8, 8),
        "$attr:I": 0,
        "$attr:T": 16,
        "$attr:DR": -2,
        "$attr:DC": -1,
    }
    assert OM.DECAY_BLOCK_DIAG.check(ok)
    for delta in (
        {"$attr:SZ": 8},
        {"$attr:SZ": (8, "x")},
        {"$attr:SZ": (8, 8, 8)},
        {"$attr:I": 2},
        {"$attr:I": "x"},
        {"$attr:T": "x"},
        {"$attr:T": 12},
        {"$attr:DR": "x"},
        {"$attr:DR": -1},
        {"$attr:DC": -2},
    ):
        assert not OM.DECAY_BLOCK_DIAG.check({**ok, **delta})
    # unknown γ rank: only the canonical −2/−1 axes are provable.
    assert OM.DECAY_BLOCK_DIAG.check({**ok, "g": RAW})
    assert not OM.DECAY_BLOCK_DIAG.check(
        {**ok, "g": RAW, "$attr:DR": 0}
    )
    # batched γ: the decay's rank grows — 0/1 spellings normalize too.
    g3 = _v("g", 3)
    assert OM.DECAY_BLOCK_DIAG.check(
        {**ok, "g": g3, "$attr:DR": 1, "$attr:DC": 2}
    )
    assert not OM.DECAY_BLOCK_DIAG.check({**ok, "g": g3, "$attr:DR": 0})
    # derive arms.
    assert OM.DECAY_BLOCK_DIAG.derive(ok) == {"$attr:TC": 8}
    assert OM.DECAY_BLOCK_DIAG.derive({**ok, "$attr:SZ": 8}) is None
    assert OM.DECAY_BLOCK_DIAG.derive({**ok, "$attr:I": 5}) is None


# ---------------------------------------------------------------------------
#  e2e — the chunked composite saturates, extracts, and delivers
# ---------------------------------------------------------------------------


def _chunked_inputs():
    s1, s2 = _v("s1", 2, 8, 16), _v("s2", 2, 8, 16)
    g = _v("g")
    v1, v2 = _v("v1", 2, 8, 4), _v("v2", 2, 8, 4)
    d = Op.make("decay", g, t=16)
    term = Op.make(
        "matmul",
        Op.make(
            "mul",
            Op.make("concat", s1, s2, dim=-2),
            d,
        ),
        Op.make("concat", v1, v2, dim=-2),
    )
    env = {
        s1: _rand(2, 8, 16, seed=30),
        s2: _rand(2, 8, 16, seed=31),
        g: torch.tensor(0.9, dtype=torch.float64),
        v1: _rand(2, 8, 4, seed=32),
        v2: _rand(2, 8, 4, seed=33),
    }
    return term, [s1, s2, g, v1, v2], env


def test_chunked_term_saturates_to_both_forms():
    """Under OM_LAWS the root class carries the fused AND chunked forms."""
    term, _inputs, _env = _chunked_inputs()
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(OM.OM_LAWS, root, max_iterations=15, max_nodes=200_000)
    root_ops = {n.op for n in eg.get_class(root).nodes}
    assert "matmul" in root_ops
    assert "om_decayed" in root_ops
    assert "concat" in root_ops  # the chunked composite
    assert "add" in root_ops


def test_extraction_prefers_fused_but_chunked_delivers():
    """Honest cost: identical FLOPs + more launches → fused wins.

    The chunked member still lowers through the torch sink and runs
    fp64-equal — the laws are exact decompositions, whatever the
    extraction policy prefers.
    """
    term, inputs, env = _chunked_inputs()
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(OM.OM_LAWS, root, max_iterations=15, max_nodes=200_000)
    best = eg.extract_best(root, launch_aware_cost)
    assert best.op == "om_decayed"

    mod = ir_to_torch_module(_ir(best, inputs))
    ref = _eval(term, env)
    with torch.no_grad():
        out = mod(*(env[v] for v in inputs))
    assert torch.equal(out, ref)

    # The hand-built chunked composite delivers identically:
    #   cat_T( add(intra1, zero-block), add(cross, intra2) )
    s1, s2, g, v1, v2 = inputs
    d = Op.make("decay", g, t=16)

    def row_block(si, i):
        dri = Op.make("split", d, sizes=(8, 8), dim=-2, index=i)
        left = Op.make(
            "om_decayed",
            Op.make("split", si, sizes=(8, 8), dim=-1, index=0),
            Op.make("split", dri, sizes=(8, 8), dim=-1, index=0),
            v1,
        )
        right = Op.make(
            "om_decayed",
            Op.make("split", si, sizes=(8, 8), dim=-1, index=1),
            Op.make("split", dri, sizes=(8, 8), dim=-1, index=1),
            v2,
        )
        return Op.make("add", left, right)

    chunked = Op.make(
        "concat", row_block(s1, 0), row_block(s2, 1), dim=-2
    )
    sink = TorchSink()
    assert "decay" in sink.supported_ops
    cmod = ir_to_torch_module(_ir(chunked, inputs))
    with torch.no_grad():
        cout = cmod(*(env[v] for v in inputs))
    assert _close(cout, ref)


def test_lift_declines_in_the_egraph_on_bad_chunks():
    """An ill-typed chunked term gains no chunked members."""
    s1, s2 = _v("s1", 2, 8, 16), _v("s2", 3, 8, 16)  # batch mismatch
    g = _v("g")
    v1, v2 = _v("v1", 2, 8, 4), _v("v2", 2, 8, 4)
    d = Op.make("decay", g, t=16)
    term = Op.make(
        "om_decayed",
        Op.make("concat", s1, s2, dim=-2),
        d,
        Op.make("concat", v1, v2, dim=-2),
    )
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(OM.OM_LAWS, root, max_iterations=10, max_nodes=200_000)
    # the fused lift still fires (the check there is about s⊙d@v —
    # an ill-typed cat fails it too), and NO concat-of-decayed member
    # may appear at the root.
    root_nodes = eg.get_class(root).nodes
    assert not any(n.op == "concat" for n in root_nodes)


# ---------------------------------------------------------------------------
#  probe — fp64 timing: full vs chunked delivery (a report, not a gate)
# ---------------------------------------------------------------------------


def _bench(fn, args, warmup=10, iters=50) -> float:
    """Median-free wall time per call — warmup then a timed loop."""
    for _ in range(warmup):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def _chunked_eval(s, D, v, C):
    """The chunked composite in plain torch — blocks over both axes."""
    outs = []
    for i, sc in enumerate(
        torch.chunk(s, -(-s.shape[-2] // C), dim=-2)
    ):
        drow = D[i * C : i * C + sc.shape[-2]]
        acc = None
        off = 0
        for vc in torch.chunk(v, -(-v.shape[-2] // C), dim=-2):
            w = vc.shape[-2]
            blk = (sc[..., off : off + w] * drow[:, off : off + w]) @ vc
            acc = blk if acc is None else acc + blk
            off += w
        outs.append(acc)
    return torch.cat(outs, dim=-2)


@pytest.mark.requires_cuda
def test_probe_full_vs_chunked():
    """Full decayed eval vs the chunked composite — honest numbers.

    The chunked form pays extra splits/adds for identical FLOPs: it
    should NOT beat the fused element.  Report, and assert only the
    fp64 agreement.
    """
    B, T, dv, C = 2, 512, 64, 64
    ts = torch.randn(B, T, T, dtype=torch.float64, device="cuda")
    tg = torch.tensor(0.95, dtype=torch.float64, device="cuda")
    tv = torch.randn(B, T, dv, dtype=torch.float64, device="cuda")
    D = _decay_ref(tg, T)
    ref = (ts * D) @ tv
    assert _close(_chunked_eval(ts, D, tv, C), ref)
    full_t = _bench(lambda: (ts * D) @ tv, ())
    chunk_t = _bench(lambda: _chunked_eval(ts, D, tv, C), ())
    comp_t = _bench(torch.compile(lambda: (ts * D) @ tv), ())
    print(
        f"chunked-decay probe B{B} T{T} dv{dv} C{C}: "
        f"full={full_t * 1e6:.0f}us compile={comp_t * 1e6:.0f}us "
        f"chunked={chunk_t * 1e6:.0f}us"
    )

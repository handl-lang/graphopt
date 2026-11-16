"""``mlopt.emit`` — term → ``.hnl`` bridge tests (no GPU/models needed)."""

from __future__ import annotations

import pytest
from mlopt_core.ir import Const, Op, Param, TensorType, Var

from mlopt.emit import (
    KitRunner,
    UnsupportedOpError,
    emit,
    emit_function,
    emit_program,
    find_hnl_run,
    leaf_kinds,
    leaf_names,
)

T = TensorType


def _x(shape=(2, 3)) -> Var:
    return Var("x", T(shape))


def _w(shape=(4, 3, 1, 1)) -> Param:
    return Param("conv.weight", T(shape))


# --- emit: structure -------------------------------------------------------


def test_emit_binary_ops() -> None:
    x, w = _x(), _w()
    assert emit(Op.make("matmul", x, w)) == "tensor.matmul(x, conv_weight)"
    assert emit(Op.make("sigmoid", x)) == "tensor.sigmoid(x)"


def test_emit_scalar_operand_wraps_rank0_for_tensor_slots() -> None:
    # add/mul/sub/div require Tensor rhs on the host — a Const operand
    # wraps as a rank-0 tensor, the same 0-d wrap the JSON bindings do.
    out = emit(Op.make("add", _x(), Const(2.0)))
    assert out == ("tensor.add(x, tensor.reshape(tensor.from_list([2.0]), []))")


def test_emit_scalar_rhs_stays_literal_where_host_accepts() -> None:
    # pow/comparisons/clamp take bare Scalar rhs (verified vs hnl-run).
    assert emit(Op.make("pow", _x(), Const(2))) == "tensor.pow(x, 2)"
    assert emit(Op.make("eq", _x(), Const(0))) == "tensor.eq(x, 0)"
    assert emit(Op.make("clamp", _x(), Const(0), Const(1))) == ("tensor.clamp(x, 0, 1)")


def test_emit_decimal_safe_float() -> None:
    # The handl lexer cannot read 1e-05 — Decimal expansion keeps it exact.
    assert emit(Op.make("pow", _x(), Const(1e-5))) == "tensor.pow(x, 0.00001)"


def test_emit_nonfinite_declines() -> None:
    with pytest.raises(UnsupportedOpError, match="non-finite"):
        emit(Op.make("add", _x(), Const(float("nan"))))
    with pytest.raises(UnsupportedOpError, match="non-finite"):
        emit(Op.make("add", _x(), Const(float("inf"))))


def test_emit_unknown_op_declines_by_name() -> None:
    with pytest.raises(UnsupportedOpError, match="dropout"):
        emit(Op.make("dropout", _x(), p=0.5, train=False))
    # sort IS covered now (pair + getitem); cummax has no host port.
    with pytest.raises(UnsupportedOpError, match="cummax"):
        emit(Op.make("cummax", _x(), dim=0))
    # needs-work names from the phase-0 split decline the same way.
    with pytest.raises(UnsupportedOpError, match="batch_norm"):
        emit(Op.make("batch_norm", _x(), training=False))


def test_emit_reshape_neg1_resolves_via_shape_of() -> None:
    x = _x((2, 3))
    term = Op.make("reshape", x, shape=[-1, 2])
    assert emit(term) == "tensor.reshape(x, [3, 2])"


def test_emit_reshape_neg1_unresolvable_declines() -> None:
    # shape_of can't see through unknown dims — a -1 wildcard over an
    # unknown-shape input declines honestly rather than guessing.
    x = Var("x", T((None, None)))
    term = Op.make("reshape", Op.make("relu", x), shape=[-1, 2])
    with pytest.raises(UnsupportedOpError, match="-1"):
        emit(term)


def test_emit_linear_synthesizes_zero_bias() -> None:
    out = emit(Op.make("linear", _x((1, 3)), Param("w", T((4, 3)))))
    assert "tensor.zeros([tensor.shape(w)[0]])" in out
    # With a bias operand it binds directly.
    b = Param("b", T((4,)))
    out2 = emit(Op.make("linear", _x((1, 3)), Param("w", T((4, 3))), b))
    assert out2.endswith(", b)")


def test_emit_chunk_index_wraps() -> None:
    out = emit(Op.make("chunk", _x(), chunks=2, dim=0, index=1))
    assert out == "(tensor.chunk(x, 2, 0))[1]"


def test_emit_conv2d_dilation_declines() -> None:
    with pytest.raises(UnsupportedOpError, match="dilation"):
        emit(
            Op.make(
                "conv2d",
                _x(),
                _w(),
                stride=1,
                padding=0,
                dilation=2,
                groups=1,
            )
        )


def test_emit_conv1d_maps_through_conv2d() -> None:
    # NCL input/weight unsqueeze to NCHW×1; stride/padding live on the
    # W axis; the (N,O,1,L') output squeezes back to NCL.
    out = emit(
        Op.make(
            "conv1d",
            _x((1, 4, 8)),
            Param("conv.weight", T((8, 4, 3))),
            Param("conv.bias", T((8,))),
            stride=(2,),
            padding=(1,),
            dilation=(1,),
            groups=1,
        )
    )
    assert out == (
        "tensor.squeeze("
        "tensor.conv2d("
        "tensor.unsqueeze(x, 2), "
        "tensor.unsqueeze(conv_weight, 2), "
        "conv_bias, [1, 2], [0, 1], 1"
        "), 2)"
    )


def test_emit_conv1d_dilation_declines() -> None:
    with pytest.raises(UnsupportedOpError, match="dilation"):
        emit(Op.make("conv1d", _x((1, 4, 8)), _w((8, 4, 3)), dilation=2))


def test_emit_conv1d_scalar_attrs() -> None:
    # torch's _single convention also spells the geometry as bare ints.
    out = emit(Op.make("conv1d", _x((1, 4, 8)), _w((8, 4, 3)), stride=2, padding=1))
    assert "[1, 2], [0, 1]" in out


def test_emit_conv1d_biasless() -> None:
    out = emit(Op.make("conv1d", _x((1, 4, 8)), _w((8, 4, 3))))
    assert "null, [1, 1], [0, 0], 1" in out


def test_emit_conv1d_nonintegral_geometry_declines() -> None:
    # Non-integral floats decline by name — never int()-truncated
    # (stride=1.5 must not silently become 1).
    with pytest.raises(UnsupportedOpError, match="not an integer"):
        emit(Op.make("conv1d", _x((1, 4, 8)), _w((8, 4, 3)), stride=1.5))
    with pytest.raises(UnsupportedOpError, match="not an integer"):
        emit(Op.make("conv1d", _x((1, 4, 8)), _w((8, 4, 3)), padding=(0.5,)))


def test_emit_matmul_rank4_flattens_batch_dims() -> None:
    # The host port is rank-2/3 only — a rank-4 [B,H,M,K]×[B,H,K,N]
    # spelling (whisper attention scores) flattens the leading dims
    # into the one batch slot and reshapes the product back.
    q = Var("q", T((1, 6, 32, 64)))
    k = Var("k", T((1, 6, 64, 32)))
    assert emit(Op.make("matmul", q, k)) == (
        "tensor.reshape(tensor.matmul("
        "tensor.reshape(q, [6, 32, 64]), tensor.reshape(k, [6, 64, 32])"
        "), [1, 6, 32, 32])"
    )


def test_emit_matmul_rank4_with_rank2_operand() -> None:
    # A rank-2 operand keeps its spelling and broadcasts over the
    # flattened batch; the product reshapes back to torch's [2,3,M,N].
    a = Var("a", T((2, 3, 4, 5)))
    w = Var("w", T((5, 7)))
    assert emit(Op.make("matmul", a, w)) == (
        "tensor.reshape(tensor.matmul(tensor.reshape(a, [6, 4, 5]), w), [2, 3, 4, 7])"
    )


def test_emit_matmul_rank3_rank4_matching_batch() -> None:
    # rank-3 batch 6 == the rank-4 side's flattened batch 6 — the
    # rank-3 operand needs no reshape (host batch broadcast covers it).
    a = Var("a", T((6, 4, 5)))
    b = Var("b", T((1, 6, 5, 7)))
    out = emit(Op.make("matmul", a, b))
    assert "tensor.matmul(a, tensor.reshape(b, [6, 5, 7]))" in out
    assert out.endswith("), [1, 6, 4, 7])")


def test_emit_matmul_rank4_needs_expand_declines() -> None:
    # [2,1]×[1,3] batch dims broadcast to [2,3] — flattening would
    # need an expand the host surface lacks; decline by name.
    a = Var("a", T((2, 1, 4, 5)))
    b = Var("b", T((1, 3, 5, 7)))
    with pytest.raises(UnsupportedOpError, match="expand"):
        emit(Op.make("matmul", a, b))


def test_emit_matmul_rank4_unknown_dims_declines() -> None:
    # A rank>3 operand whose dims shape_of cannot prove declines —
    # the flatten needs concrete extents, never a guess.
    u = Var("u", T((None, 6, 32, 64)))
    k = Var("k", T((1, 6, 64, 32)))
    with pytest.raises(UnsupportedOpError, match="unknown dims"):
        emit(Op.make("matmul", u, k))


def test_emit_matmul_rank4_eval_matches_when_hnl_run_available() -> None:
    """The flattened spelling evaluates to torch's rank-4 matmul."""
    binary = find_hnl_run()
    if binary is None:
        pytest.skip("hnl-run not built")
    np = pytest.importorskip("numpy")
    runner = KitRunner(binary)
    q = Var("q", T((1, 2, 3, 4)))
    k = Var("k", T((1, 2, 4, 5)))
    term = Op.make("matmul", q, k)
    rng = np.random.default_rng(0)
    qv, kv = rng.standard_normal((1, 2, 3, 4)), rng.standard_normal((1, 2, 4, 5))
    feed = {
        "q": {"shape": [1, 2, 3, 4], "data": qv.reshape(-1).tolist()},
        "k": {"shape": [1, 2, 4, 5], "data": kv.reshape(-1).tolist()},
    }
    out = runner.tensor(term, feed)
    assert out["shape"] == [1, 2, 3, 5]
    assert np.allclose(np.asarray(out["data"]).reshape(1, 2, 3, 5), qv @ kv, atol=1e-12)


def test_emit_permute_decomposes_to_swapaxes() -> None:
    # (0,2,3,1) on rank 4 decomposes into transpositions — verified
    # against hnl-run (whisper-tiny-encoder export).
    out = emit(Op.make("permute", _x((1, 4, 8, 2)), dim=(0, 2, 3, 1)))
    # Two transpositions suffice: (1,2) then (2,3) → [a,c,d,b].
    assert out == "tensor.swapaxes(tensor.swapaxes(x, 1, 2), 2, 3)"


def test_emit_permute_identity_is_passthrough() -> None:
    assert emit(Op.make("permute", _x((2, 3)), dim=(0, 1))) == "x"


def test_emit_permute_negative_dims() -> None:
    # -1 == last axis of the rank; (-1, 0, 1) on rank 3 → [2, 0, 1].
    out = emit(Op.make("permute", _x((2, 3, 4)), dim=(-1, 0, 1)))
    assert out.startswith("tensor.swapaxes(")


def test_emit_permute_non_permutation_declines() -> None:
    # (1, -1) on rank 2 normalizes to [1, 1] — not bijective.
    with pytest.raises(UnsupportedOpError, match="permutation"):
        emit(Op.make("permute", _x((2, 3)), dim=(1, -1)))


def test_emit_permute_declines_non_literal_dims() -> None:
    with pytest.raises(UnsupportedOpError, match="permute"):
        emit(Op.make("permute", _x((2, 3)), dim=Op.make("relu", _x())))
    with pytest.raises(UnsupportedOpError, match="permutation"):
        emit(Op.make("permute", _x((2, 3)), dim=(0, 0)))


def test_emit_permute_positional_dims() -> None:
    # The torch binding spells permute(x, *dims) — each trailing
    # positional arg is one dim, not just args[1].
    out = emit(Op.make("permute", _x((1, 4, 8, 2)), 0, 2, 3, 1))
    assert out == "tensor.swapaxes(tensor.swapaxes(x, 1, 2), 2, 3)"


def test_emit_permute_positional_single_list_arg() -> None:
    # A lone list-valued positional arg stays the list spelling.
    out = emit(Op.make("permute", _x((2, 3)), (1, 0)))
    assert out == "tensor.swapaxes(x, 0, 1)"


def test_emit_permute_nonintegral_dim_declines() -> None:
    # Non-integral floats decline by name — never int()-truncated.
    with pytest.raises(UnsupportedOpError, match="not an integer"):
        emit(Op.make("permute", _x((2, 3)), dim=(0, 1.5)))
    with pytest.raises(UnsupportedOpError, match="not an integer"):
        emit(Op.make("permute", _x((2, 3)), 0, 1.5))
    # Integral floats still spell the same permutation.
    assert emit(Op.make("permute", _x((2, 3)), dim=(1.0, 0.0))) == ("tensor.swapaxes(x, 0, 1)")


def test_emit_split_int_sizes_declines() -> None:
    with pytest.raises(UnsupportedOpError, match="split-size"):
        emit(Op.make("split", _x(), sizes=2, dim=0, index=0))


def test_emit_var_correction_declines() -> None:
    with pytest.raises(UnsupportedOpError, match="correction"):
        emit(Op.make("var", _x(), dim=1, correction=2, keepdim=False))


def test_emit_expand_implicit_declines() -> None:
    with pytest.raises(UnsupportedOpError, match="implicit"):
        emit(Op.make("expand", _x(), shape=[2, 3], implicit=True))


def test_emit_dtype_attr_declines_nonfloat() -> None:
    with pytest.raises(UnsupportedOpError, match="dtype"):
        emit(Op.make("add", _x(), _x(), dtype="int64"))


def test_emit_dtype_attr_full_exact_fill_roundtrips() -> None:
    # 1.0 is exactly representable in f32 — the f64 artifact carries the
    # identical value, so dropping the dtype attr is honest.
    out = emit(Op.make("full", fill_value=1.0, shape=[2, 2], dtype="float32"))
    assert "tensor.full" in out
    # 1.5 is exactly representable in f16 too.
    out = emit(Op.make("full", fill_value=1.5, shape=[4], dtype="f16"))
    assert "tensor.full" in out


def test_emit_dtype_attr_full_inexact_fill_declines() -> None:
    # 0.1 is not f32-representable — emitting it as f64 would silently
    # change the value the dtype promised, so the decline stays.
    with pytest.raises(UnsupportedOpError, match="dtype"):
        emit(Op.make("full", fill_value=0.1, shape=[2, 2], dtype="float32"))
    # Non-fill ops keep declining even for an exactly-representable
    # value — the attr there claims compute precision, not a constant.
    with pytest.raises(UnsupportedOpError, match="dtype"):
        emit(Op.make("add", _x(), _x(), dtype="float32"))
    # Integer dtypes have no float round-trip path — still decline.
    with pytest.raises(UnsupportedOpError, match="dtype"):
        emit(Op.make("full", fill_value=1, shape=[2], dtype="int64"))


def test_emit_to_cast_identity_spellings() -> None:
    # tensor.to's host port honours the f64 string / tensor exemplar —
    # the identity cast — plus the real narrowing casts k_to landed
    # (host-tensor ecc57d9).  The no-attr spelling, the f64 dtype
    # attr, and the to.other exemplar form all emit it.
    assert emit(Op.make("to", _x())) == 'tensor.to(x, "float64")'
    assert emit(Op.make("to", _x(), dtype="float64")) == 'tensor.to(x, "float64")'
    assert emit(Op.make("to", _x(), dtype="double")) == 'tensor.to(x, "float64")'
    y = Var("y", T((2, 3)))
    assert emit(Op.make("to", _x(), y)) == "tensor.to(x, y)"
    # A positional cast spec lands as arg1 — the f64 name passes.
    assert emit(Op.make("to", _x(), "float64")) == 'tensor.to(x, "float64")'


def test_emit_to_cast_narrowing_dtypes_emit() -> None:
    # Every spelling host-tensor's k_to accepts emits the canonical
    # dtype string — the cast is real now, not dropped.  Keep this
    # table in sync with ``_TO_DTYPES`` in emit.py (itself mirroring
    # k_to's match arms + ``int_cast_range``).
    cases = {
        "float32": "float32",
        "f32": "float32",
        "torch.float32": "float32",
        "int64": "int64",
        "i64": "int64",
        "long": "int64",
        "torch.int64": "int64",
        "int32": "int32",
        "int16": "int16",
        "int8": "int8",
        "uint8": "uint8",
        "u8": "uint8",
        "bool": "bool",
    }
    for spelling, canonical in cases.items():
        assert emit(Op.make("to", _x(), dtype=spelling)) == (f'tensor.to(x, "{canonical}")'), (
            spelling
        )
    # Positional cast specs (arg1 attr / string Const exemplar) take
    # the same table — the gate never sees them.
    assert emit(Op.make("to", _x(), "int64")) == 'tensor.to(x, "int64")'
    assert emit(Op.make("to", _x(), Const("float32"))) == 'tensor.to(x, "float32")'  # ty: ignore[invalid-argument-type] -- Const is spelling-strict; the string exemplar is deliberate


def test_emit_to_cast_unsupported_dtype_declines() -> None:
    # f16/bf16/float8 need a rounding mode the host surface does not
    # define — they decline at the dtype gate before the builder runs.
    for bad in ("float16", "f16", "bfloat16", "bf16", "float8_e4m3fn", "complex64"):
        with pytest.raises(UnsupportedOpError, match="dtype"):
            emit(Op.make("to", _x(), dtype=bad))
        # A positional cast spec bypasses the dtype attr gate — the
        # builder re-checks it against the same k_to table.
        with pytest.raises(UnsupportedOpError, match="dtype"):
            emit(Op.make("to", _x(), bad))
        with pytest.raises(UnsupportedOpError, match="dtype"):
            emit(Op.make("to", _x(), Const(bad)))  # ty: ignore[invalid-argument-type] -- deliberate non-float exemplar
    # Other ops keep the compute-precision contract: a non-f64 dtype
    # attr still declines even for names k_to would accept.
    with pytest.raises(UnsupportedOpError, match="dtype"):
        emit(Op.make("add", _x(), _x(), dtype="int64"))
    with pytest.raises(UnsupportedOpError, match="dtype"):
        emit(Op.make("add", _x(), _x(), dtype="float32"))
    # Conflicting cast specs and exemplar+spec mixes have no aten
    # spelling.
    with pytest.raises(UnsupportedOpError, match="conflicting"):
        emit(Op.make("to", _x(), "int64", dtype="float32"))
    y = Var("y", T((2, 3)))
    with pytest.raises(UnsupportedOpError, match="exemplar"):
        emit(Op.make("to", _x(), y, dtype="int64"))
    # Surplus operands have no aten spelling.
    with pytest.raises(UnsupportedOpError, match="operands"):
        emit(Op.make("to", _x(), _x(), _x()))


def test_emit_to_cast_evals_identity_when_hnl_run_available() -> None:
    """The emitted ``tensor.to`` call evaluates as the identity cast."""
    binary = find_hnl_run()
    if binary is None:
        pytest.skip("hnl-run not built")
    runner = KitRunner(binary)
    x = Var("x", T((2,)))
    term = Op.make("to", Op.make("add", x, Const(1.0)), dtype="float64")
    feed = {"x": {"shape": [2], "data": [1.0, -2.5]}}
    out = runner.tensor(term, feed)
    assert out["shape"] == [2]
    assert out["data"] == [2.0, -1.5]


# --- leaves -----------------------------------------------------------------


def test_leaf_names_and_kinds() -> None:
    term = Op.make("add", _x(), Op.make("mul", _w(), _x()))
    assert leaf_names(term) == ["x", "conv.weight"]
    assert leaf_kinds(term) == [("x", "var"), ("conv.weight", "param")]


# --- emit_program ------------------------------------------------------------


FEED: dict = {
    "x": {"shape": [2, 3], "data": [1.0] * 6},
    "conv.weight": {"shape": [4, 3, 1, 1], "data": [0.5] * 12},
}


def test_emit_program_binds_leaves() -> None:
    term = Op.make("matmul", _x(), _w())
    src = emit_program(term, FEED)
    assert src.startswith('from "host-tensor" import tensor\ndo {')
    assert "x :: tensor.reshape(tensor.from_list(" in src
    assert "tensor.matmul(x, conv_weight)" in src


def test_emit_program_missing_leaf_raises() -> None:
    with pytest.raises(ValueError, match="missing leaves"):
        emit_program(Op.make("matmul", _x(), _w()), {"x": FEED["x"]})


def test_emit_program_hnl_channel() -> None:
    term = Op.make("matmul", _x(), _w())
    feed = dict(FEED)
    feed["conv.weight"] = {"hnl": "tensor.ones([4, 3, 1, 1])"}
    src = emit_program(term, feed)
    assert "conv_weight :: tensor.ones([4, 3, 1, 1])" in src


def test_emit_program_array_feed() -> None:
    import numpy as np

    # 'x' binds once (dedup); a second Var with a different name:
    y = Var("y", T((2,)))
    term2 = Op.make("add", _x((2,)), y)
    src = emit_program(term2, {"x": np.array([1.0, 2.0]), "y": [3.0, 4.0]})
    assert "tensor.from_list([1.0, 2.0])" in src
    assert "tensor.from_list([3.0, 4.0])" in src


# --- hash-consed (shared-subterm) emission ---------------------------------
#
# Terms are hash-consed DAGs; emit_program/emit_function bind every
# Op node with >=2 parent references once as ``__sN ::`` lines so the
# artifact carries the DAG, not the tree expansion.


def test_emit_program_binds_shared_subterm() -> None:
    shared = Op.make("matmul", _x(), _w())
    term = Op.make("add", shared, shared)
    src = emit_program(term, FEED)
    assert "  __s0 :: tensor.matmul(x, conv_weight)" in src
    assert "tensor.add(__s0, __s0)" in src
    # The shared spelling renders once, not twice.
    assert src.count("tensor.matmul") == 1


def test_emit_program_shared_bindings_topological() -> None:
    # add(b,b) shared under mul(c,c): children bind before parents.
    a = Op.make("relu", _x())
    b = Op.make("sigmoid", a)
    c = Op.make("add", b, b)
    term = Op.make("mul", c, c)
    src = emit_program(term, {"x": FEED["x"]})
    lines = [ln.strip() for ln in src.splitlines() if " :: " in ln]
    assert "__s0 :: tensor.sigmoid(tensor.relu(x))" in lines
    assert "__s1 :: tensor.add(__s0, __s0)" in lines
    assert src.rstrip().endswith("tensor.mul(__s1, __s1)\n}")
    # __s0's line precedes __s1's — a binding only names earlier names.
    assert lines.index("__s0 :: tensor.sigmoid(tensor.relu(x))") < lines.index(
        "__s1 :: tensor.add(__s0, __s0)"
    )


def test_emit_function_binds_shared_subterm() -> None:
    shared = Op.make("matmul", _x(), _w())
    term = Op.make("add", shared, shared)
    src = emit_function(term, {"conv.weight": FEED["conv.weight"]})
    assert "__s0 :: tensor.matmul(x, conv_weight)" in src
    assert "tensor.add(__s0, __s0)" in src


def test_emit_unshared_term_gets_no_shared_bindings() -> None:
    # A tree-shaped term (every node referenced once) emits exactly
    # the pre-sharing spelling — no __s lines at all.
    term = Op.make("add", Op.make("mul", _x(), _x()), _w())
    src = emit_program(term, FEED)
    assert "__s" not in src
    assert "tensor.add(tensor.mul(x, x), conv_weight)" in src


# --- emit_function ------------------------------------------------------------


def test_emit_function_vars_become_params() -> None:
    term = Op.make("matmul", _x(), _w())
    src = emit_function(term, {"conv.weight": FEED["conv.weight"]})
    assert "export optimized := (x) ~> do {" in src
    assert "conv_weight :: tensor.reshape(" in src
    assert "tensor.matmul(x, conv_weight)" in src


def test_emit_function_var_in_feed_declines() -> None:
    term = Op.make("matmul", _x(), _w())
    feed = dict(FEED)
    with pytest.raises(ValueError, match="function parameters"):
        emit_function(term, feed)


def test_emit_function_missing_param_raises() -> None:
    with pytest.raises(ValueError, match="missing leaves"):
        emit_function(Op.make("matmul", _x(), _w()), {})


# --- KitRunner / find_hnl_run -------------------------------------------------


def test_find_hnl_run_returns_path_or_none() -> None:
    found = find_hnl_run()
    assert found is None or found.exists()


def test_kitrunner_explicit_missing_binary() -> None:
    with pytest.raises(FileNotFoundError):
        KitRunner(binary="/nonexistent/hnl-run")


def test_emit_eval_roundtrip_when_hnl_run_available() -> None:
    """Real emit → hnl-run eval on a tiny term (skips without binary)."""
    binary = find_hnl_run()
    if binary is None:
        pytest.skip("hnl-run not built")
    runner = KitRunner(binary)
    x = Var("x", T((2,)))
    w = Param("w", T((2,)))
    term = Op.make("add", Op.make("mul", x, w), Const(1.0))
    feed = {
        "x": {"shape": [2], "data": [1.0, 2.0]},
        "w": {"shape": [2], "data": [10.0, 20.0]},
    }
    out = runner.tensor(term, feed)
    assert out["shape"] == [2]
    assert out["data"] == [11.0, 41.0]


def test_emit_shared_subterm_eval_matches_when_hnl_run_available() -> None:
    """Hash-consed emit evaluates to the same value as the tree spelling."""
    binary = find_hnl_run()
    if binary is None:
        pytest.skip("hnl-run not built")
    runner = KitRunner(binary)
    x = Var("x", T((2,)))
    w = Param("w", T((2,)))
    half = Op.make("mul", x, w)
    term = Op.make("add", half, half)  # shared subtree → __s0 binding
    feed = {
        "x": {"shape": [2], "data": [1.0, 2.0]},
        "w": {"shape": [2], "data": [10.0, 20.0]},
    }
    src = emit_program(term, feed)
    assert "__s0 ::" in src
    out = runner.tensor(term, feed)
    assert out["shape"] == [2]
    # x*w + x*w = 2*x*w = [20, 80]
    assert out["data"] == [20.0, 80.0]


def test_emit_conv1d_permute_roundtrip_when_hnl_run_available() -> None:
    """Numeric check of the two arch-adapter decompositions vs torch."""
    binary = find_hnl_run()
    if binary is None:
        pytest.skip("hnl-run not built")
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    runner = KitRunner(binary)
    x = Var("x", T((1, 2, 5)))
    w = Param("w", T((3, 2, 3)))
    b = Param("b", T((3,)))
    term = Op.make("conv1d", x, w, b, stride=(2,), padding=(1,), dilation=(1,), groups=1)
    feed = {
        "x": {
            "shape": [1, 2, 5],
            "data": [0.1, -0.2, 0.3, 0.4, -0.5, 1.0, 0.5, -0.5, 0.25, 0.75],
        },
        "w": {"shape": [3, 2, 3], "data": [0.1] * 18},
        "b": {"shape": [3], "data": [0.01, 0.02, 0.03]},
    }
    out = runner.tensor(term, feed)
    ref = F.conv1d(
        torch.tensor(feed["x"]["data"]).reshape(1, 2, 5),
        torch.tensor(feed["w"]["data"]).reshape(3, 2, 3),
        torch.tensor(feed["b"]["data"]),
        stride=2,
        padding=1,
    )
    assert out["shape"] == list(ref.shape)
    assert torch.allclose(torch.tensor(out["data"]).reshape(ref.shape), ref, atol=1e-6)

    xp = Var("xp", T((1, 4, 8, 2)))
    pterm = Op.make("permute", xp, dim=(0, 2, 3, 1))
    feed2 = {"xp": {"shape": [1, 4, 8, 2], "data": [float(i) for i in range(64)]}}
    out2 = runner.tensor(pterm, feed2)
    ref2 = torch.arange(64, dtype=torch.float32).reshape(1, 4, 8, 2).permute(0, 2, 3, 1)
    assert out2["shape"] == list(ref2.shape)
    assert torch.equal(torch.tensor(out2["data"]).reshape(ref2.shape), ref2)


# --- safetensors sidecar (tensor.load) --------------------------------------
#
# Opt-in ``sidecar=`` on emit_program/emit_function: Param leaves at or
# above ``sidecar_min_elems`` bind as ``tensor.load("<basename>",
# "<leaf>")`` and their values land in one .safetensors file.  Var
# leaves and {"hnl"} feeds always stay inline; the default path never
# writes files.


def _st_header(path) -> tuple[dict, bytes]:  # noqa: ANN001, ANN202
    """Parse a safetensors file → (header dict, payload bytes)."""
    import json as _json
    import struct as _struct

    raw = path.read_bytes()
    hlen = _struct.unpack("<Q", raw[:8])[0]
    return _json.loads(raw[8 : 8 + hlen]), raw[8 + hlen :]


def test_write_safetensors_layout(tmp_path) -> None:  # noqa: ANN001
    import numpy as np

    from mlopt.emit import write_safetensors

    p = write_safetensors(
        tmp_path / "w.safetensors",
        {
            "a.w": ((2, 2), [1.0, 2.0, 3.0, 4.0]),
            "b.b": ((3,), [0.5, -0.5, 0.25]),
        },
    )
    header, payload = _st_header(p)
    assert set(header) == {"a.w", "b.b"}
    assert header["a.w"]["dtype"] == "F64"
    assert header["a.w"]["shape"] == [2, 2]
    b0, b1 = header["a.w"]["data_offsets"]
    assert (b1 - b0) == 4 * 8  # F64 payload
    c0, c1 = header["b.b"]["data_offsets"]
    assert c0 == b1  # offsets are payload-relative and contiguous
    vals = np.frombuffer(payload[b0:b1], dtype="<f8")
    assert vals.tolist() == [1.0, 2.0, 3.0, 4.0]
    vals2 = np.frombuffer(payload[c0:c1], dtype="<f8")
    assert vals2.tolist() == [0.5, -0.5, 0.25]


def test_write_safetensors_f32_rounds(tmp_path) -> None:  # noqa: ANN001
    import numpy as np

    from mlopt.emit import write_safetensors

    p = write_safetensors(tmp_path / "w.safetensors", {"w": ((2,), [1.5, 2.5])}, dtype="F32")
    header, payload = _st_header(p)
    assert header["w"]["dtype"] == "F32"
    b0, b1 = header["w"]["data_offsets"]
    assert b1 - b0 == 2 * 4
    assert np.frombuffer(payload[b0:b1], dtype="<f4").tolist() == [1.5, 2.5]
    with pytest.raises(ValueError, match="dtype"):
        write_safetensors(tmp_path / "x.safetensors", {}, dtype="F16")


def test_emit_function_sidecar_binds_tensor_load(tmp_path) -> None:  # noqa: ANN001
    big = Param("big.w", T((10,)))
    small = Param("small.b", T((2,)))
    term = Op.make("add", Op.make("mul", _x((10,)), big), small)
    feed = {
        "big.w": {"shape": [10], "data": [float(i) for i in range(10)]},
        "small.b": {"shape": [2], "data": [0.5, 0.25]},
    }
    sidecar = tmp_path / "sub" / "m.safetensors"
    src = emit_function(term, feed, sidecar=sidecar, sidecar_min_elems=5)
    # The rendered path is the basename — $TENSOR_LOAD_DIR / cwd
    # resolves it, keeping the .hnl + .safetensors pair relocatable.
    assert 'big_w :: tensor.load("m.safetensors", "big.w")' in src
    # Below-threshold params keep the inline literal.
    assert "small_b :: tensor.reshape(tensor.from_list([0.5, 0.25])" in src
    assert sidecar.exists()
    header, _payload = _st_header(sidecar)
    assert list(header) == ["big.w"]
    assert header["big.w"]["shape"] == [10]


def test_emit_program_sidecar_never_sidecars_vars(tmp_path) -> None:  # noqa: ANN001
    # Var leaves are runtime inputs — even a huge data feed stays
    # inline in emit_program's prelude; only the Param sidecars.
    y = Var("y", T((10,)))
    w = Param("w", T((10,)))
    term = Op.make("add", Op.make("mul", _x((10,)), w), y)
    feed = {
        "x": {"shape": [10], "data": [1.0] * 10},
        "w": {"shape": [10], "data": [3.0] * 10},
        "y": {"shape": [10], "data": [2.0] * 10},
    }
    src = emit_program(term, feed, sidecar=tmp_path / "m.safetensors", sidecar_min_elems=5)
    assert 'w :: tensor.load("m.safetensors", "w")' in src
    assert "y :: tensor.reshape(tensor.from_list([2.0, 2.0" in src
    header, _ = _st_header(tmp_path / "m.safetensors")
    assert list(header) == ["w"]


def test_emit_sidecar_hnl_and_default_unchanged(tmp_path) -> None:  # noqa: ANN001
    # {"hnl"} computed feeds stay inline even for Param leaves, and a
    # sidecar request with nothing eligible writes no file.
    feed = dict(FEED)
    feed["conv.weight"] = {"hnl": "tensor.ones([4, 3, 1, 1])"}
    src = emit_program(Op.make("matmul", _x(), _w()), feed, sidecar=tmp_path / "m.st")
    assert "conv_weight :: tensor.ones([4, 3, 1, 1])" in src
    assert not (tmp_path / "m.st").exists()
    # Default path: identical spelling, no file.
    assert emit_program(Op.make("matmul", _x(), _w()), FEED) == emit_program(
        Op.make("matmul", _x(), _w()), FEED, sidecar=None
    )


def test_emit_sidecar_array_feed(tmp_path) -> None:  # noqa: ANN001
    import numpy as np

    w = Param("w", T((3, 2)))
    term = Op.make("matmul", _x((1, 3)), w)
    src = emit_function(
        term,
        {"w": np.arange(6, dtype=np.float64).reshape(3, 2)},
        sidecar=tmp_path / "m.safetensors",
        sidecar_min_elems=5,
    )
    assert 'w :: tensor.load("m.safetensors", "w")' in src
    header, payload = _st_header(tmp_path / "m.safetensors")
    assert header["w"]["shape"] == [3, 2]
    b0, b1 = header["w"]["data_offsets"]
    assert np.frombuffer(payload[b0:b1], dtype="<f8").tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_sidecar_eval_matches_inline_when_hnl_run_available(tmp_path) -> None:  # noqa: ANN001
    """tensor.load sidecar evaluates to the same tensor as from_list."""
    binary = find_hnl_run()
    if binary is None:
        pytest.skip("hnl-run not built")
    runner = KitRunner(binary, tensor_load_dir=tmp_path)
    w = Param("w", T((2, 5)))
    feed = {"w": {"shape": [2, 5], "data": [float(i) for i in range(10)]}}
    src_side = emit_program(w, feed, sidecar=tmp_path / "m.safetensors", sidecar_min_elems=5)
    try:
        out_side = runner.run(src_side)
    except Exception as e:  # noqa: BLE001 — stale hnl-run lacks tensor.load
        if "tensor.load" in str(e):
            pytest.skip(f"hnl-run predates tensor.load: {e}")
        raise
    out_inline = KitRunner(binary).run(emit_program(w, feed))
    assert out_side == out_inline


# --- CLI -----------------------------------------------------------------------


def test_export_parser_args() -> None:
    from mlopt.cli import build_parser

    args = build_parser().parse_args(
        ["export", "whisper:openai/whisper-tiny", "--out", "m.hnl", "--check"]
    )
    assert args.cmd == "export"
    assert args.model == "whisper:openai/whisper-tiny"
    assert args.out == "m.hnl"
    assert args.check is True
    assert args.iters == 4


def test_export_unknown_ref(capsys) -> None:  # noqa: ANN001
    from mlopt.cli import main

    assert main(["export", "bert-base-uncased"]) == 2
    assert "no adapter" in capsys.readouterr().err


def test_export_declines_when_load_fails(capsys) -> None:  # noqa: ANN001
    # Hermetic load failure: point kokoro_src at a nonexistent tree so
    # load() fails honestly regardless of which extras/sources this
    # venv happens to have (the kokoro deps may well be installed).
    from mlopt.cli import main
    from mlopt.models import catalog  # noqa: F401 — populates registry

    assert (
        main(
            [
                "export",
                "kokoro:hexgrad/Kokoro-82M",
                "--adapter-opt",
                "kokoro_src=/nonexistent/kokoro-src",
            ]
        )
        == 1
    )
    assert "load failed" in capsys.readouterr().err


# --- carrier expansion ------------------------------------------------------
#
# Carrier-winning extractions spell attention as om_apply(om_elem(s,v))
# and decayed attention as om_decayed(s,d,v) — ops with no tensor.*
# surface.  emit() re-spells the two definitional unfolds
# (OM_UNLIFT / OM_DECAYED_UNLIFT) before rendering; every other carrier
# shape still declines by name.


def test_expand_carriers_om_apply_elem() -> None:
    from mlopt.carrier_expand import expand_carriers

    s = Op.make("matmul", _x((1, 4, 8)), _w((8, 8)))
    v = Op.make("transpose", _w((8, 4)))
    term = Op.make("om_apply", Op.make("om_elem", s, v))
    out = expand_carriers(term)
    assert out == Op.make("matmul", Op.make("softmax", s, dim=-1), v)
    assert "om_" not in repr(out)


def test_expand_carriers_om_decayed() -> None:
    from mlopt.carrier_expand import expand_carriers

    s = _x((2, 4, 8))
    d = _w((8,))
    v = _w((8, 4))
    term = Op.make("om_decayed", s, d, v)
    out = expand_carriers(term)
    assert out == Op.make("matmul", Op.make("mul", s, d), v)


def test_expand_carriers_leaves_other_carriers() -> None:
    from mlopt.carrier_expand import expand_carriers

    # om_compose has no definitional tensor spelling — it must pass
    # through untouched so emit declines by name, never guesses.
    e1 = Op.make("om_elem", _x((1, 4, 8)), _w((8, 4)))
    e2 = Op.make("om_elem", _x((1, 4, 8)), _w((8, 4)))
    term = Op.make("om_apply", Op.make("om_compose", e1, e2))
    assert expand_carriers(term) is term
    with pytest.raises(UnsupportedOpError, match="om_apply"):
        emit(term)


def test_expand_carriers_nested_and_noop() -> None:
    from mlopt.carrier_expand import expand_carriers

    plain = Op.make("matmul", _x((2, 3)), _w((4, 3)))
    assert expand_carriers(plain) is plain
    # carrier ops nested inside tensor-land args still expand.
    inner = Op.make("om_apply", Op.make("om_elem", _x((1, 4, 8)), _w((8, 4))))
    outer = Op.make("transpose", Op.make("add", inner, _x((1, 4, 4))))
    out = expand_carriers(outer)
    assert out.op == "transpose"
    assert "om_apply" not in repr(out)


def test_emit_renders_expanded_carrier_term() -> None:
    # The end-to-end spelling: om_apply(om_elem) emits as
    # tensor.matmul(tensor.softmax(s, -1), v).
    s = _x((1, 4, 8))
    v = _w((8, 4))
    src = emit(Op.make("om_apply", Op.make("om_elem", s, v)))
    assert src == "tensor.matmul(tensor.softmax(x, -1), conv_weight)"


def test_expand_carriers_noncanonical_om_apply_passes_through() -> None:
    from mlopt.carrier_expand import expand_carriers

    s = _x((1, 4, 8))
    v = _w((8, 4))
    el = Op.make("om_elem", s, v)
    # Extra arg / attrs on om_apply would be silently dropped by the
    # unfold — the term passes through and emit declines by name.
    for bad in (
        Op.make("om_apply", el, _x()),
        Op.make("om_apply", el, scale=2),
        Op.make("om_apply", Op.make("om_elem", s, v, scale=2)),
    ):
        assert expand_carriers(bad) is bad
        with pytest.raises(UnsupportedOpError, match="om_apply"):
            emit(bad)


def test_expand_carriers_attr_carrying_om_decayed_passes_through() -> None:
    from mlopt.carrier_expand import expand_carriers

    s, d, v = _x((2, 4, 8)), _w((8,)), _w((8, 4))
    term = Op.make("om_decayed", s, d, v, decay=0.9)
    assert expand_carriers(term) is term
    with pytest.raises(UnsupportedOpError, match="om_decayed"):
        emit(term)


# --- KitRunner timeout knob ---------------------------------------------------


def test_hnl_run_timeout_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``MLOPT_HNL_RUN_TIMEOUT`` overrides the 600 s default; bad values
    are named errors, never silent defaults."""
    from mlopt.emit import _hnl_run_timeout

    monkeypatch.delenv("MLOPT_HNL_RUN_TIMEOUT", raising=False)
    assert _hnl_run_timeout() == 600.0
    monkeypatch.setenv("MLOPT_HNL_RUN_TIMEOUT", "1200.5")
    assert _hnl_run_timeout() == 1200.5
    monkeypatch.setenv("MLOPT_HNL_RUN_TIMEOUT", "junk")
    with pytest.raises(ValueError, match="MLOPT_HNL_RUN_TIMEOUT"):
        _hnl_run_timeout()
    monkeypatch.setenv("MLOPT_HNL_RUN_TIMEOUT", "0")
    with pytest.raises(ValueError, match="positive"):
        _hnl_run_timeout()


def test_kitrunner_timeout_passes_env_and_names_it_on_expiry(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The env value reaches ``subprocess.run(timeout=...)`` and a
    ``TimeoutExpired`` surfaces as ``KitEvalError`` naming the knob —
    the message ``export --verify-run`` prints verbatim."""
    import subprocess as sp

    import mlopt.emit as emit_mod
    from mlopt.emit import KitEvalError

    fake = tmp_path / "hnl-run"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    runner = KitRunner(fake)

    seen: dict = {}

    def _timeout(cmd, **kw):  # noqa: ANN001, ANN202
        seen.update(kw)
        raise sp.TimeoutExpired(cmd=cmd, timeout=kw.get("timeout"))  # ty: ignore[invalid-argument-type] -- fake stand-in; None is fine here

    monkeypatch.setattr(emit_mod.subprocess, "run", _timeout)
    monkeypatch.setenv("MLOPT_HNL_RUN_TIMEOUT", "42")
    with pytest.raises(KitEvalError, match="MLOPT_HNL_RUN_TIMEOUT"):
        runner.run("1 + 1")
    assert seen["timeout"] == 42.0

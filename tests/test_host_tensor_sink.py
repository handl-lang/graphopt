"""HostTensorSink — the ``mlopt_core.ports.Sink`` over host-tensor.

Phase 1 of plan 0036 (``docs/reports/host_tensor_sink_phase0.md``).
The sink lowers the 137 liftable-now IR ops onto the sibling
``handl-lang/host-tensor`` repo's ``tensor.*`` registry through its
PyO3 binding (``handl_host_tensor``).  The extension resolves a
maturin-installed wheel first, then the sibling-checkout cargo
dev artifact (``../host-tensor/target/<profile>/``); every test that
touches it skips when neither resolves.
"""

from __future__ import annotations

import numpy as np
import pytest
from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var
from mlopt_core.ports import Executor, Sink, VerifyResult
from mlopt_host_tensor import HostTensorSink, _binding
from mlopt_host_tensor.errors import (
    HostTensorError,
    UnsupportedHostOpError,
)


def _extension() -> None:
    """Skip the test when the PyO3 extension isn't built."""
    try:
        _binding.module()
    except HostTensorError as exc:
        pytest.skip(f"handl_host_tensor extension not built: {exc}")


@pytest.fixture(scope="module")
def sink() -> HostTensorSink:
    """A sink, skipped entirely without the extension."""
    _extension()
    return HostTensorSink()


def _ir(term: Op, *vars_: Var, params: dict | None = None) -> IR:
    return IR(root=term, inputs=list(vars_))


def _lower(sink: HostTensorSink, term: Op, *vars_: Var, params=None):
    return sink.lower(_ir(term, *vars_), params)


# ---------------------------------------------------------------------------
#  Binding surface
# ---------------------------------------------------------------------------


def test_supported_ops_is_the_liftable_137(
    sink: HostTensorSink,
) -> None:
    """The declared surface is exactly the phase-0 liftable set."""
    assert len(sink.supported_ops) == 137
    assert "add" in sink.supported_ops
    assert "matmul" in sink.supported_ops
    # needs-work / honest-decline names must NOT be claimed.
    for op in (
        "dropout",
        "sort",
        "to",
        "einsum",
        "getitem",
        "pack",
        "new_zeros",
        "detach_",
        "tan",
    ):
        assert op not in sink.supported_ops


def test_sink_and_executor_protocols(sink: HostTensorSink) -> None:
    """Structural conformance to the ports."""
    assert isinstance(sink, Sink)
    x = Var("x", TensorType((2,)))
    ex = sink.lower(_ir(Op.make("neg", x), x))
    assert isinstance(ex, Executor)


def test_binding_lists_and_calls_ops() -> None:
    """The raw extension: enumerate + dispatch ``tensor.add``."""
    _extension()
    ops = _binding.host_ops()
    assert len(ops) == 238
    assert "tensor.add" in ops
    out = _binding.call(
        "add",
        [
            {
                "__tensor__": {
                    "shape": [2],
                    "data": [1.0, 2.0],
                    "dtype": "f64",
                }
            },
            {
                "__tensor__": {
                    "shape": [2],
                    "data": [3.0, 4.0],
                    "dtype": "f64",
                }
            },
        ],
    )
    assert "__tensor__" in out
    assert '"data":[4.0,6.0]' in out.replace(" ", "")


def test_binding_unknown_op_raises() -> None:
    """A bogus op name is a loud ValueError, not a silent miss."""
    _extension()
    with pytest.raises(Exception, match="no op registered"):
        _binding.call("no_such_op", [])


# ---------------------------------------------------------------------------
#  Pointwise — NumPy references
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "np_fn"),
    [
        ("abs", np.abs),
        ("neg", np.negative),
        ("exp", np.exp),
        ("sqrt", np.sqrt),
        ("sin", np.sin),
        ("cos", np.cos),
        ("tanh", np.tanh),
        ("sigmoid", lambda a: 1 / (1 + np.exp(-a))),
        ("relu", np.maximum),
    ],
)
def test_pointwise_unary(sink: HostTensorSink, op: str, np_fn) -> None:
    """Unary pointwise ops match the NumPy reference."""
    x = Var("x", TensorType((2, 3)))
    a = np.linspace(-1.0, 1.0, 6).reshape(2, 3)
    if op == "sqrt":
        # sqrt of a negative is NaN — the wire decodes it as nan;
        # keep the comparison on the non-negative half.
        a = np.abs(a) + 0.25
        expected = np.sqrt(a)
    elif op == "relu":
        expected = np.maximum(a, 0.0)
    else:
        expected = np_fn(a)
    got = _lower(sink, Op.make(op, x), x).forward(a)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("op", "np_fn"),
    [
        ("add", np.add),
        ("mul", np.multiply),
        ("sub", np.subtract),
        ("maximum", np.maximum),
    ],
)
def test_pointwise_binary(sink: HostTensorSink, op: str, np_fn) -> None:
    """Binary pointwise ops match the NumPy reference."""
    x = Var("x", TensorType((2, 3)))
    y = Var("y", TensorType((2, 3)))
    a = np.arange(6.0).reshape(2, 3) + 1
    b = np.linspace(0.5, 2.0, 6).reshape(2, 3)
    got = _lower(sink, Op.make(op, x, y), x, y).forward(a, b)
    np.testing.assert_allclose(got, np_fn(a, b), rtol=1e-12, atol=1e-12)


def test_scalar_operand_wraps(sink: HostTensorSink) -> None:
    """A Const scalar operand broadcasts like torch's 0-dim tensor."""
    x = Var("x", TensorType((3,)))
    term = Op.make("add", x, Const(2.5))
    got = _lower(sink, term, x).forward(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(got, [3.5, 4.5, 5.5])


def test_comparison_and_where(sink: HostTensorSink) -> None:
    """``gt`` + ``where`` compose into a select."""
    x = Var("x", TensorType((4,)))
    y = Var("y", TensorType((4,)))
    term = Op.make("where", Op.make("gt", x, y), x, y)
    a = np.array([1.0, 5.0, 3.0, 8.0])
    b = np.array([2.0, 4.0, 6.0, 7.0])
    got = _lower(sink, term, x, y).forward(a, b)
    np.testing.assert_allclose(got, np.maximum(a, b))


# ---------------------------------------------------------------------------
#  Reductions
# ---------------------------------------------------------------------------


def test_sum_dim_keepdim(sink: HostTensorSink) -> None:
    """``dim``/``keepdim`` attrs map onto the port's positional args."""
    x = Var("x", TensorType((2, 3)))
    a = np.arange(1.0, 7.0).reshape(2, 3)
    got = _lower(
        sink, Op.make("sum", x, dim=0, keepdim=True), x
    ).forward(a)
    np.testing.assert_allclose(got, a.sum(axis=0, keepdims=True))


def test_sum_tuple_dim(sink: HostTensorSink) -> None:
    """A tuple ``dim`` reduces over both axes."""
    x = Var("x", TensorType((2, 3)))
    a = np.arange(1.0, 7.0).reshape(2, 3)
    got = _lower(sink, Op.make("sum", x, dim=(0, 1)), x).forward(a)
    np.testing.assert_allclose(got, a.sum(axis=(0, 1)))


def test_mean_all_axes(sink: HostTensorSink) -> None:
    """No ``dim`` attr reduces everything to a scalar."""
    x = Var("x", TensorType((2, 3)))
    a = np.arange(1.0, 7.0).reshape(2, 3)
    got = _lower(sink, Op.make("mean", x), x).forward(a)
    np.testing.assert_allclose(got, a.mean())


def test_softmax_dim(sink: HostTensorSink) -> None:
    """``softmax`` matches the NumPy softmax along ``dim``."""
    x = Var("x", TensorType((2, 3)))
    a = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    got = _lower(sink, Op.make("softmax", x, dim=1), x).forward(a)
    e = np.exp(a - a.max(axis=1, keepdims=True))
    np.testing.assert_allclose(got, e / e.sum(axis=1, keepdims=True))


def test_var_unbiased(sink: HostTensorSink) -> None:
    """``var`` with ``correction=1`` is torch's unbiased variance."""
    x = Var("x", TensorType((4,)))
    a = np.array([1.0, 2.0, 3.0, 4.0])
    got = _lower(
        sink, Op.make("var", x, dim=0, correction=1), x
    ).forward(a)
    np.testing.assert_allclose(got, a.var(ddof=1))


def test_cumsum(sink: HostTensorSink) -> None:
    """``cumsum`` along a dim."""
    x = Var("x", TensorType((2, 3)))
    a = np.arange(1.0, 7.0).reshape(2, 3)
    got = _lower(sink, Op.make("cumsum", x, dim=1), x).forward(a)
    np.testing.assert_allclose(got, np.cumsum(a, axis=1))


# ---------------------------------------------------------------------------
#  Linalg / shape / constructors
# ---------------------------------------------------------------------------


def test_matmul(sink: HostTensorSink) -> None:
    """``matmul`` matches NumPy matmul."""
    x = Var("x", TensorType((2, 3)))
    y = Var("y", TensorType((3, 4)))
    a = np.arange(1.0, 7.0).reshape(2, 3)
    b = np.arange(1.0, 13.0).reshape(3, 4)
    got = _lower(sink, Op.make("matmul", x, y), x, y).forward(a, b)
    np.testing.assert_allclose(got, a @ b, rtol=1e-12)


def test_linear_with_bias(sink: HostTensorSink) -> None:
    """``linear(x, w, b)`` = x @ w.T + b."""
    x = Var("x", TensorType((2, 3)))
    w = Param("w", TensorType((4, 3)))
    b = Param("b", TensorType((4,)))
    term = Op.make("linear", x, w, b)
    a = np.ones((2, 3))
    wv = np.arange(12.0).reshape(4, 3)
    bv = np.arange(4.0)
    got = sink.lower(_ir(term, x), {"w": wv, "b": bv}).forward(a)
    np.testing.assert_allclose(got, a @ wv.T + bv, rtol=1e-12)


def test_reshape_neg1(sink: HostTensorSink) -> None:
    """``reshape`` resolves the ``-1`` wildcard."""
    x = Var("x", TensorType((2, 3)))
    a = np.arange(6.0).reshape(2, 3)
    got = _lower(sink, Op.make("reshape", x, shape=[3, -1]), x).forward(
        a
    )
    assert got.shape == (3, 2)
    np.testing.assert_allclose(got, a.reshape(3, 2))


def test_transpose(sink: HostTensorSink) -> None:
    """``transpose`` swaps the two dims."""
    x = Var("x", TensorType((2, 3)))
    a = np.arange(6.0).reshape(2, 3)
    got = _lower(
        sink, Op.make("transpose", x, dim0=0, dim1=1), x
    ).forward(a)
    np.testing.assert_allclose(got, a.T)


def test_stack_concat_unbind(sink: HostTensorSink) -> None:
    """``stack``/``concat``/``unbind[index]`` compose."""
    x = Var("x", TensorType((2,)))
    y = Var("y", TensorType((2,)))
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    st = _lower(sink, Op.make("stack", x, y, dim=0), x, y).forward(a, b)
    np.testing.assert_allclose(st, [[1.0, 2.0], [3.0, 4.0]])
    cat = _lower(sink, Op.make("concat", x, y, dim=0), x, y).forward(
        a, b
    )
    np.testing.assert_allclose(cat, [1.0, 2.0, 3.0, 4.0])
    un = _lower(
        sink,
        Op.make(
            "unbind",
            st if False else Op.make("stack", x, y, dim=0),
            dim=0,
            index=1,
        ),
        x,
        y,
    ).forward(a, b)
    np.testing.assert_allclose(un, b)


def test_zeros_full_arange_eye(sink: HostTensorSink) -> None:
    """Constructor ops produce the declared shapes/values."""
    x = Var("x", TensorType((1,)))  # keeps IR.inputs non-empty
    z = _lower(sink, Op.make("zeros", shape=[2, 2]), x).forward(
        np.zeros(1)
    )
    assert z.shape == (2, 2) and not z.any()
    f = _lower(
        sink, Op.make("full", shape=[3], fill_value=2.5), x
    ).forward(np.zeros(1))
    np.testing.assert_allclose(f, [2.5, 2.5, 2.5])
    r = _lower(sink, Op.make("arange", Const(0), Const(5)), x).forward(
        np.zeros(1)
    )
    np.testing.assert_allclose(r, [0.0, 1.0, 2.0, 3.0, 4.0])
    e = _lower(sink, Op.make("eye", dim=3), x).forward(np.zeros(1))
    np.testing.assert_allclose(e, np.eye(3))


def test_numel_item(sink: HostTensorSink) -> None:
    """Metadata ops: ``numel`` counts elements, ``item`` extracts."""
    x = Var("x", TensorType((2, 3)))
    n = _lower(sink, Op.make("numel", x), x).forward(np.zeros((2, 3)))
    assert float(np.asarray(n)) == 6.0
    one = Var("one", TensorType(()))
    it = _lower(sink, Op.make("item", one), one).forward(np.array(4.0))
    assert float(np.asarray(it)) == 4.0


def test_param_leaf(sink: HostTensorSink) -> None:
    """``Param`` leaves resolve through ``lower(params=...)``."""
    x = Var("x", TensorType((3,)))
    w = Param("w", TensorType((3,)))
    term = Op.make("mul", x, w)
    got = sink.lower(
        _ir(term, x), {"w": np.array([1.0, 2.0, 3.0])}
    ).forward(np.array([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(got, [4.0, 10.0, 18.0])


def test_missing_param_declines_at_lower(sink: HostTensorSink) -> None:
    """A ``Param`` with no leaf value fails at ``lower``, loudly."""
    x = Var("x", TensorType((3,)))
    w = Param("w", TensorType((3,)))
    with pytest.raises(HostTensorError, match="no leaf value"):
        sink.lower(_ir(Op.make("mul", x, w), x), {})


# ---------------------------------------------------------------------------
#  Honest declines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op", ["dropout", "sort", "to", "einsum", "nonzero"]
)
def test_unsupported_op_declines_at_lower(
    sink: HostTensorSink, op: str
) -> None:
    """Ops outside ``supported_ops`` fail loudly at lower time."""
    x = Var("x", TensorType((2, 2)))
    with pytest.raises(UnsupportedHostOpError, match=op):
        sink.lower(_ir(Op.make(op, x), x))


def test_pack_root_multi_output(sink: HostTensorSink) -> None:
    """A ``pack`` root yields one output leaf per element."""
    x = Var("x", TensorType((2, 2)))
    term = Op.make("pack", Op.make("sum", x), Op.make("max", x))
    a = np.arange(4.0).reshape(2, 2) + 1
    out = _lower(sink, term, x).forward(a)
    assert isinstance(out, tuple) and len(out) == 2
    np.testing.assert_allclose(out[0], a.sum())
    np.testing.assert_allclose(out[1], a.max())


def test_nonfinite_input_declines(sink: HostTensorSink) -> None:
    """NaN/Inf cannot cross the JSON wire — refused at encode."""
    x = Var("x", TensorType((2,)))
    ex = _lower(sink, Op.make("neg", x), x)
    with pytest.raises(HostTensorError, match="non-finite"):
        ex.forward(np.array([1.0, np.inf]))


# ---------------------------------------------------------------------------
#  verify()
# ---------------------------------------------------------------------------


def test_verify_identical_passes(sink: HostTensorSink) -> None:
    """``verify`` returns the structural VerifyResult on a match."""
    x = Var("x", TensorType((3,)))
    ex = _lower(sink, Op.make("add", x, Const(1.0)), x)
    rep = sink.verify(ex, ex, (np.array([1.0, 2.0, 3.0]),))
    assert isinstance(rep, VerifyResult)
    assert rep.passed and rep.max_abs == 0.0 and rep.max_rel == 0.0


def test_verify_detects_difference(sink: HostTensorSink) -> None:
    """``verify`` fails the gate when outputs diverge."""
    x = Var("x", TensorType((3,)))
    a = _lower(sink, Op.make("add", x, Const(1.0)), x)
    b = _lower(sink, Op.make("add", x, Const(2.0)), x)
    rep = sink.verify(a, b, (np.array([1.0, 2.0, 3.0]),))
    assert not rep.passed
    assert rep.max_abs == pytest.approx(1.0)


def test_verify_atol_gate(sink: HostTensorSink) -> None:
    """``atol`` adds the absolute bound on top of ``max_rel``."""
    x = Var("x", TensorType((3,)))
    a = _lower(sink, Op.make("add", x, Const(0.0)), x)
    b = _lower(sink, Op.make("add", x, Const(1e-9)), x)
    tight = sink.verify(
        a, b, (np.ones(3) * 1e6,), rtol=1e-12, atol=1e-12
    )
    loose = sink.verify(
        a, b, (np.ones(3) * 1e6,), rtol=1e-12, atol=1e-6
    )
    assert not tight.passed
    assert loose.passed

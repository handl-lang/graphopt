"""``.hnl`` → mlopt term — the reverse bridge's contract.

``mlopt_discovery.hnl_import.hnl_to_term`` parses the expression
grammar ``hnl_surface.emit`` produces (plus the ``do { … }`` program
form and handl-native ``tensor.*`` calls on the covered surface) back
into ``mlopt_core.ir`` terms.  These tests pin:

* **round-trip identity** — ``emit → hnl_to_term → emit`` is the
  identity on the surface projection for every ``_SPECS`` op;
* **handl-native coverage** — the broader ``tensor.*`` ops the
  importer accepts mint the canonical mlopt spellings;
* **honest declines** — ops outside the covered surface, non-tensor
  results, malformed syntax and ambiguous spellings all refuse by
  name, never silently;
* **drift** — every covered op name is a real mlopt IR op
  (``mlopt_host_tensor.bindings.HOST_BINDINGS`` plus the three
  surface ops the executor table does not carry).
"""

from __future__ import annotations

import pytest
from mlopt_core.ir import Const, Op, Param, TensorType, Var
from mlopt_discovery.hnl_import import (
    HnlImportError,
    covered_ops,
    hnl_to_term,
)
from mlopt_discovery.hnl_surface import UnsupportedOpError, emit


def _x() -> Var:
    return Var("x", TensorType((2, 3)))


def _w() -> Param:
    return Param("w", TensorType((3, 4)))


# ---------------------------------------------------------------------------
#  Round-trip: emit → import → emit is the identity on _SPECS ops
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Every op ``hnl_surface._SPECS`` emits must re-emit identically."""

    @pytest.mark.parametrize(
        "term",
        [
            Op.make("add", _x(), _w()),
            Op.make("mul", _x(), _w()),
            Op.make("sub", _x(), _x()),
            Op.make("div", _w(), _x()),
            Op.make("matmul", _x(), _w()),
            Op.make(
                "conv2d",
                _x(),
                _w(),
                _w(),
                stride=[1, 1],
                padding=[1, 1],
                dilation=[1, 1],
                groups=1,
            ),
            Op.make(
                "conv2d",
                _x(),
                _w(),
                stride=(2, 2),
                padding=0,
                groups=2,
            ),
            Op.make(
                "batch_norm",
                _x(),
                _w(),
                _w(),
                _w(),
                _w(),
                training=False,
                momentum=0.1,
                eps=1e-5,
                cudnn_enabled=True,
            ),
            Op.make(
                "max_pool2d",
                _x(),
                kernel_size=[2, 2],
                stride=[2, 2],
                padding=0,
                dilation=1,
                ceil_mode=False,
            ),
            Op.make("reshape", _x(), shape=[6, 1]),
            Op.make("transpose", _x(), dim0=0, dim1=1),
            Op.make("concat", _x(), _w(), dim=0),
            Op.make("pack", _x(), _w()),
            # nested composition
            Op.make(
                "batch_norm",
                Op.make(
                    "conv2d",
                    _x(),
                    _w(),
                    _w(),
                    stride=[1, 1],
                    padding=[1, 1],
                ),
                _w(),
                _w(),
                _w(),
                _w(),
                training=False,
                eps=0.00001,
            ),
        ],
    )
    def test_emit_import_emit_identity(self, term: Op) -> None:
        src = emit(term)
        back = hnl_to_term(src)
        assert emit(back) == src

    def test_imported_term_shape(self) -> None:
        """The imported term carries the surface attrs canonically."""
        t = hnl_to_term("tensor.conv2d(x, w, b, [1, 1], [1, 1], 2)")
        assert isinstance(t, Op)
        assert t.op == "conv2d"
        assert t.attrs["stride"] == [1, 1]
        assert t.attrs["padding"] == [1, 1]
        assert t.attrs["groups"] == 2
        assert "dilation" not in t.attrs  # no surface slot

    def test_batch_norm_eval_mode(self) -> None:
        t = hnl_to_term("tensor.batch_norm(x, w, b, m, v, 0.00001)")
        assert t.attrs["training"] is False
        assert t.attrs["eps"] == pytest.approx(1e-5)


# ---------------------------------------------------------------------------
#  Handl-native coverage — the broader tensor.* surface
# ---------------------------------------------------------------------------


class TestNativeOps:
    """Ops beyond ``_SPECS`` that the importer covers."""

    @pytest.mark.parametrize(
        "src,op",
        [
            ("tensor.rsqrt(x)", "rsqrt"),
            ("tensor.sigmoid(x)", "sigmoid"),
            ("tensor.neg(x)", "neg"),
            ("tensor.maximum(x, y)", "maximum"),
            ("tensor.add(x, 2)", "add"),
            ("tensor.pow(x, 2)", "pow"),
            ("tensor.eq(x, y)", "eq"),
            ("tensor.tan(x)", "tan"),
        ],
    )
    def test_unary_binary(self, src: str, op: str) -> None:
        t = hnl_to_term(src)
        assert isinstance(t, Op)
        assert t.op == op

    def test_scalar_rhs_is_const(self) -> None:
        t = hnl_to_term("tensor.mul(x, 2.5)")
        assert t.args[1] == Const(2.5)

    def test_free_names_are_vars(self) -> None:
        t = hnl_to_term("tensor.add(x, y)")
        assert isinstance(t.args[0], Var)
        assert t.args[0].name == "x"
        assert t.args[0].typ.shape is None  # wildcard

    def test_leaves_override_shape(self) -> None:
        t = hnl_to_term(
            "tensor.add(x, y)", leaves={"x": (4, 4), "y": None}
        )
        assert t.args[0].typ.shape == (4, 4)
        assert t.args[1].typ.shape is None

    def test_leaves_tensor_type(self) -> None:
        t = hnl_to_term(
            "tensor.add(x, y)", leaves={"x": TensorType((2,))}
        )
        assert t.args[0].typ.shape == (2,)

    @pytest.mark.parametrize(
        "src,check",
        [
            ("tensor.softmax(x, 1)", lambda t: t.attrs == {"dim": 1}),
            (
                "tensor.sum(x, [0, 1], true)",
                lambda t: t.attrs == {"dim": [0, 1], "keepdim": True},
            ),
            (
                "tensor.sum(x)",
                lambda t: t.attrs == {},
            ),
            (
                "tensor.var(x, [0], false, true)",
                lambda t: (
                    t.attrs
                    == {"dim": [0], "correction": 0, "keepdim": True}
                ),
            ),
            (
                "tensor.count_nonzero(x, 0)",
                lambda t: t.attrs == {"dim": 0},
            ),
            ("tensor.unsqueeze(x, 0)", lambda t: t.attrs == {"dim": 0}),
            (
                "tensor.select(x, 1, 0)",
                lambda t: t.attrs == {"dim": 1, "index": 0},
            ),
            ("tensor.squeeze(x)", lambda t: t.attrs == {}),
            (
                "tensor.squeeze(x, 1)",
                lambda t: t.attrs == {"dim": 1},
            ),
            (
                "tensor.flatten(x, 0, -1)",
                lambda t: t.attrs == {"start_dim": 0, "end_dim": -1},
            ),
            (
                "tensor.narrow(x, 0, 1, 2)",
                lambda t: (
                    t.attrs == {"dim": 0, "start": 1, "length": 2}
                ),
            ),
            (
                "tensor.cumsum(x, 0)",
                lambda t: t.attrs == {"dim": 0},
            ),
            (
                "tensor.cumprod(x, -1)",
                lambda t: t.attrs == {"dim": -1},
            ),
            (
                "tensor.log_softmax(x, 0)",
                lambda t: t.attrs == {"dim": 0},
            ),
            (
                "tensor.layer_norm(x, [4], w, b, 0.00001)",
                lambda t: (
                    t.attrs == {"dim": [4], "eps": pytest.approx(1e-5)}
                ),
            ),
            (
                "tensor.rms_norm(x, [4], w, 0.000001)",
                lambda t: (
                    t.attrs == {"dim": [4], "eps": pytest.approx(1e-6)}
                ),
            ),
            (
                "tensor.expand(x, 4, 4)",
                lambda t: t.attrs == {"shape": [4, 4]},
            ),
            (
                "tensor.expand(x, [4, 4])",
                lambda t: t.attrs == {"shape": [4, 4]},
            ),
            (
                "tensor.repeat(x, 2, 3)",
                lambda t: t.attrs == {"shape": [2, 3]},
            ),
            (
                "tensor.flip(x, [0, 1])",
                lambda t: t.attrs == {"dim": [0, 1]},
            ),
            (
                "tensor.tril(x, 1)",
                lambda t: t.attrs == {"diagonal": 1},
            ),
            (
                "tensor.triu(x)",
                lambda t: t.attrs == {},
            ),
            (
                "tensor.pixel_shuffle(x, 2)",
                lambda t: t.attrs == {"upscale_factor": 2},
            ),
            (
                "tensor.pixel_unshuffle(x, 4)",
                lambda t: t.attrs == {"downscale_factor": 4},
            ),
            (
                "tensor.clamp(x, 0, 1)",
                lambda t: t.attrs == {"min": 0, "max": 1},
            ),
            (
                "tensor.clamp_min(x, 0)",
                lambda t: t.attrs == {"min": 0},
            ),
            (
                "tensor.clamp_max(x, 1)",
                lambda t: t.attrs == {"max": 1},
            ),
            (
                "tensor.nan_to_num(x, 0, null, null)",
                lambda t: t.attrs == {"nan": 0},
            ),
            (
                "tensor.elu(x, 1.5, 2.0, 3.0)",
                lambda t: (
                    t.attrs
                    == {"alpha": 1.5, "scale": 2.0, "input_scale": 3.0}
                ),
            ),
            (
                "tensor.celu(x, 2.0)",
                lambda t: t.attrs == {"alpha": 2.0},
            ),
            (
                "tensor.leaky_relu(x, 0.1)",
                lambda t: t.attrs == {"negative_slope": 0.1},
            ),
            (
                "tensor.softplus(x, 2, 20)",
                lambda t: t.attrs == {"beta": 2, "threshold": 20},
            ),
            (
                "tensor.hardtanh(x, -1, 1)",
                lambda t: t.attrs == {"min": -1, "max": 1},
            ),
            (
                'tensor.gelu(x, "tanh")',
                lambda t: t.attrs == {"approximate": "tanh"},
            ),
            (
                "tensor.lerp(x, y, 0.5)",
                lambda t: len(t.args) == 3,
            ),
            (
                "tensor.where(c, x, y)",
                lambda t: t.op == "where" and len(t.args) == 3,
            ),
            (
                "tensor.stack([x, y], 0)",
                lambda t: t.op == "stack" and t.attrs == {"dim": 0},
            ),
            (
                "tensor.stack([x, y])",
                lambda t: t.op == "stack" and t.attrs == {},
            ),
            (
                "tensor.concat([x, y], 1)",
                lambda t: t.op == "concat" and t.attrs == {"dim": 1},
            ),
            (
                "tensor.linear(x, w, b)",
                lambda t: t.op == "linear" and len(t.args) == 3,
            ),
            (
                "tensor.max_pool(x, [2, 2], [2, 2])",
                lambda t: (
                    t.op == "max_pool2d"
                    and t.attrs
                    == {"kernel_size": [2, 2], "stride": [2, 2]}
                ),
            ),
            (
                "tensor.zeros([2, 3])",
                lambda t: (
                    t.op == "zeros" and t.attrs == {"shape": [2, 3]}
                ),
            ),
            (
                "tensor.ones([2])",
                lambda t: t.op == "ones" and t.attrs == {"shape": [2]},
            ),
            (
                "tensor.full([2], 0.5)",
                lambda t: t.attrs == {"shape": [2], "fill_value": 0.5},
            ),
            (
                "tensor.full_like(x, 1.0)",
                lambda t: (
                    t.op == "full_like"
                    and t.attrs == {"fill_value": 1.0}
                ),
            ),
            (
                "tensor.arange(0, 5, 2)",
                lambda t: t.op == "arange" and len(t.args) == 3,
            ),
            (
                "tensor.eye(3, 4)",
                lambda t: (
                    t.op == "eye" and t.attrs == {"dim": 3, "m": 4}
                ),
            ),
            (
                "tensor.t(x)",
                lambda t: (
                    t.op == "transpose"
                    and t.attrs == {"dim0": -2, "dim1": -1}
                ),
            ),
            (
                "tensor.slice(x, 0, 1, 5, 2)",
                lambda t: (
                    t.attrs
                    == {"dim": 0, "start": 1, "end": 5, "step": 2}
                ),
            ),
            (
                "tensor.slice(x, null, 1, 5)",
                lambda t: (
                    t.attrs.get("dim") == 0
                    and t.attrs.get("start") == 1
                ),
            ),
        ],
    )
    def test_native_attr_ops(self, src: str, check) -> None:
        t = hnl_to_term(src)
        assert isinstance(t, Op)
        assert check(t)


# ---------------------------------------------------------------------------
#  do-blocks — bindings, marshaling Params, inline substitution
# ---------------------------------------------------------------------------


class TestDoBlocks:
    def test_marshaled_bindings_become_params(self) -> None:
        t = hnl_to_term(
            "do {\n"
            "  w :: tensor.reshape(tensor.from_list([1.0, 2.0]), [1, 2])\n"
            "  tensor.matmul(x, w)\n"
            "}"
        )
        assert isinstance(t.args[1], Param)
        assert t.args[1].name == "w"
        assert t.args[1].typ.shape == (1, 2)

    def test_from_list_binding_shape(self) -> None:
        t = hnl_to_term(
            "do {\n"
            "  b :: tensor.from_list([[1.0, 2.0], [3.0, 4.0]])\n"
            "  tensor.add(x, b)\n"
            "}"
        )
        assert isinstance(t.args[1], Param)
        assert t.args[1].typ.shape == (2, 2)

    def test_zeros_ones_bindings(self) -> None:
        t = hnl_to_term(
            "do {\n"
            "  z :: tensor.zeros([2, 2])\n"
            "  o :: tensor.ones([3])\n"
            "  tensor.add(z, o)\n"
            "}"
        )
        assert isinstance(t.args[0], Param)
        assert t.args[0].typ.shape == (2, 2)
        assert isinstance(t.args[1], Param)
        assert t.args[1].typ.shape == (3,)

    def test_expr_binding_inlines(self) -> None:
        t = hnl_to_term(
            "do {\n"
            "  s :: tensor.mul(g, tensor.rsqrt(v))\n"
            "  tensor.add(x, s)\n"
            "}"
        )
        # s substitutes its subterm — no env leaf survives
        rhs = t.args[1]
        assert isinstance(rhs, Op) and rhs.op == "mul"
        assert rhs.args[1].op == "rsqrt"

    def test_later_binding_shadows(self) -> None:
        t = hnl_to_term(
            "do {\n"
            "  a :: tensor.mul(x, y)\n"
            "  a :: tensor.add(x, y)\n"
            "  tensor.mul(a, a)\n"
            "}"
        )
        assert t.args[0].op == "add"
        assert t.args[1].op == "add"

    def test_empty_do_declines(self) -> None:
        with pytest.raises(HnlImportError):
            hnl_to_term("do {\n}")

    def test_module_directives_stripped(self) -> None:
        src = (
            "--- a model\n"
            'import "host-tensor"\n'
            "from other import thing\n"
            "# comment\n"
            "tensor.add(x, y)\n"
        )
        t = hnl_to_term(src)
        assert isinstance(t, Op) and t.op == "add"

    def test_pack_program(self) -> None:
        t = hnl_to_term("do {\n  tensor.add(x, y)\n}")
        assert t.op == "add"


# ---------------------------------------------------------------------------
#  Honest declines
# ---------------------------------------------------------------------------


class TestDeclines:
    @pytest.mark.parametrize(
        "src",
        [
            # ops with no covered surface spelling
            "tensor.chunk(x, 2, 0)",
            "tensor.split(x, [1, 1], 0)",
            "tensor.unbind(x, 0)",
            "tensor.topk(x, 2)",
            "tensor.sort(x)",
            "tensor.meshgrid([x])",
            "tensor.nonzero(x)",
            "tensor.gather(x, 0, y)",
            "tensor.einsum(x, y)",
            "tensor.norm(x)",
            "tensor.logsumexp(x, 0)",
            "tensor.glu(x)",
            "tensor.prelu(x, y)",
            "tensor.threshold(x, 0, 1)",
            "tensor.signbit(x)",
            "tensor.hypot(x, y)",
            "tensor.floor_divide(x, y)",
            "tensor.copysign(x, y)",
            "tensor.float_power(x, y)",
            "tensor.nextafter(x, y)",
            "tensor.logaddexp(x, y)",
            "tensor.mT(x)",
            "tensor.swapaxes(x, 0, 1)",
            "tensor.moveaxis(x, [0], [1])",
            "tensor.rot90(x)",
            "tensor.diag(x)",
            "tensor.diagonal(x)",
            "tensor.trace(x)",
            "tensor.constant_pad_nd(x, [1], 0)",
            "tensor.kron(x, y)",
            "tensor.cross(x, y)",
            "tensor.dist(x, y)",
            "tensor.cdist(x, y)",
            "tensor.pdist(x)",
            "tensor.bincount(x)",
            "tensor.quantile(x, 0.5)",
            "tensor.searchsorted(x, y)",
            "tensor.nonzero_dim(x, 0)",
            "tensor.to(x)",
            "tensor.mm(x, y)",
            "tensor.bmm(x, y)",
            "tensor.addmm(x, y, z)",
            "tensor.index_select(x, 0, y)",
            "tensor.index_add(x, 0, y, z)",
            "tensor.index_fill(x, 0, y, 1)",
            "tensor.take(x, y)",
            "tensor.embedding(w, idx)",
            "tensor.one_hot(idx, 4)",
            "tensor.sdpa(q, k, v)",
            "tensor.adaptive_avg_pool2d(x, [1, 1])",
            "tensor.logcumsumexp(x, 0)",
            "tensor.cummax(x, 0)",
            "tensor.cummin(x, 0)",
            "tensor.unflatten(x, 0, [2, 2])",
            "tensor.permute(x, [1, 0])",
            "tensor.roll(x, 1)",
            "tensor.movedim(x, 0, 1)",
            "tensor.broadcast_to(x, [2, 2])",
            "tensor.getitem(x)",
            "tensor.dropout(x)",
            "tensor.size(x)",
            "tensor.stride(x)",
            "tensor.dim(x)",
            "tensor.is_contiguous(x)",
            "tensor.allclose(x, y)",
            "tensor.is_nonzero(x)",
            "tensor.avg_pool2d(x, [2, 2])",
            "tensor.unique(x)",
            "tensor.median(x)",
            "tensor.mode(x)",
            "tensor.kthvalue(x, 2)",
            "tensor.argsort(x)",
            "tensor.linspace(0, 1, 5)",
            "tensor.logspace(0, 1, 5)",
            "tensor.empty([2])",
            "tensor.rand([2])",
            "tensor.normal([2])",
            # max/min are reductions with an ambiguous elementwise
            # spelling on the host surface
            "tensor.max(x, y)",
            "tensor.min(x)",
            # marshaling / non-term ops in term position
            "tensor.from_list([1, 2])",
            "tensor.to_list(x)",
            "tensor.shape(x)",
            "tensor.derive(x)",
            "tensor.add_backward(x, y, z)",
            "tensor.conv2d_backward(x, y, z)",
            "tensor.batch_norm_backward(x, y, z)",
            # unknown ops and non-tensor callees
            "tensor.foo(x)",
            "tensor.add_backward",
            "foo.bar(x)",
            "add(x, y)",
            "tensor",
            # structural declines
            "tensor.add(x)",
            "tensor.add(x, y, z)",
            "tensor.concat(x, 0)",
            "tensor.concat([], 0)",
            "tensor.batch_norm(x, w, b, m, v, 1e-5, extra)",
            "tensor.batch_norm(x, null, b, m, v)",
            "tensor.clamp(x, y, 1)",
            "tensor.clamp(x, [0], 1)",
            "tensor.reshape(x, 4)",
            "tensor.reshape(x, shape)",
            "tensor.gelu(x, 1)",
            "tensor.layer_norm(x, 4, w)",
            "tensor.layer_norm(x, [4], null, b)",
            "tensor.slice(x, 0, y)",
            "tensor.flip(x, 0)",
            "tensor.full(0.5)",
            "tensor.zeros(2)",
            "tensor.eye()",
            "tensor.t(x, y)",
            "tensor.transpose(x, 0)",
            "tensor.var(x, 0, 2)",
            "tensor.softmax(x, 0, 1)",
            'tensor.add(x, "s")',
            "tensor.add(x, null)",
            "tensor.add(tensor.foo(x), y)",
        ],
    )
    def test_unsupported_declines_by_name(self, src: str) -> None:
        with pytest.raises((UnsupportedOpError, HnlImportError)):
            hnl_to_term(src)

    @pytest.mark.parametrize(
        "src",
        [
            "",
            "tensor.add(x,",
            "do {",
            "do {\n}",
            "tensor.add(x y)",
            "tensor.add(x,)",
            "[x, y",
            "tensor.add(x, y))",
            "tensor.add((x), y",  # unbalanced paren
            'tensor.gelu(x, "unterminated)',
            "tensor.add(x @ y)",
        ],
    )
    def test_syntax_errors(self, src: str) -> None:
        with pytest.raises(HnlImportError):
            hnl_to_term(src)

    def test_dotted_leaf_declines(self) -> None:
        with pytest.raises(UnsupportedOpError, match="dotted name"):
            hnl_to_term("tensor.add(tensor.add, y)")


# ---------------------------------------------------------------------------
#  Edge branches — parser corners, marshaling fallthroughs, strict shapes
# ---------------------------------------------------------------------------


class TestEdgeBranches:
    def test_field_name_required_after_dot(self) -> None:
        with pytest.raises(HnlImportError, match="field name"):
            hnl_to_term("tensor.add.(x)")

    def test_bad_string_escape_is_syntax_error(self) -> None:
        with pytest.raises(HnlImportError, match="string literal"):
            hnl_to_term('tensor.add(x, "\\q")')

    def test_non_tensor_call_target_declines(self) -> None:
        with pytest.raises(UnsupportedOpError):
            hnl_to_term("(x)(y)")

    def test_scalar_rhs_bool_is_const_one(self) -> None:
        t = hnl_to_term("tensor.add(x, true)")
        assert t.args[1] == Const(1)

    def test_list_literal_operand_declines(self) -> None:
        with pytest.raises(UnsupportedOpError, match="list literal"):
            hnl_to_term("tensor.add([1, 2], x)")

    def test_string_operand_declines(self) -> None:
        with pytest.raises(UnsupportedOpError, match="string literal"):
            hnl_to_term('tensor.add(x, "s")')

    @pytest.mark.parametrize(
        "src",
        [
            "tensor.softmax(x, true)",
            'tensor.softmax(x, "d")',
            'tensor.select(x, "d", 0)',
            'tensor.cumsum(x, "d")',
            "tensor.var(x, 0, 1)",
            "tensor.sum(x, [0.5])",  # float dim entries reject
            'tensor.zeros([2, "s"])',  # non-int shape entries reject
            "tensor.zeros([2.5])",
            'tensor.full(["a"], 1)',
        ],
    )
    def test_strict_scalar_and_shape_validation(self, src: str) -> None:
        with pytest.raises(UnsupportedOpError):
            hnl_to_term(src)

    # -- do-block binding fallthroughs ------------------------------------

    def test_non_call_binding_inlines(self) -> None:
        t = hnl_to_term("do {\n  w :: x\n  tensor.add(w, y)\n}")
        assert t.args[0].name == "x"

    def test_unknown_tensor_binding_declines_at_use(self) -> None:
        with pytest.raises(UnsupportedOpError, match=r"tensor\.foo"):
            hnl_to_term(
                "do {\n  w :: tensor.foo(x)\n  tensor.add(w, y)\n}"
            )

    def test_from_list_non_list_arg_declines(self) -> None:
        # tensor.from_list(5) is not a nested array — no Param mint;
        # the binding falls through to the (declined) term spelling.
        with pytest.raises(UnsupportedOpError, match="from_list"):
            hnl_to_term(
                "do {\n  w :: tensor.from_list(5)\n"
                "  tensor.add(w, y)\n}"
            )

    def test_from_list_scalar_data_declines(self) -> None:
        with pytest.raises(UnsupportedOpError, match="from_list"):
            hnl_to_term(
                "do {\n  w :: tensor.from_list([true])\n"
                "  tensor.add(w, y)\n}"
            )

    def test_reshape_binding_without_literal_shape_is_term(
        self,
    ) -> None:
        # args[0] is not tensor.from_list → ordinary term binding
        t = hnl_to_term(
            "do {\n  w :: tensor.reshape(x, [2])\n  tensor.add(w, y)\n}"
        )
        assert isinstance(t.args[0], Op) and t.args[0].op == "reshape"

    def test_reshape_binding_bad_shape_declines(self) -> None:
        with pytest.raises(UnsupportedOpError, match="shape"):
            hnl_to_term(
                "do {\n"
                "  w :: tensor.reshape(tensor.from_list([1.0]), x)\n"
                "  tensor.add(w, y)\n}"
            )

    def test_from_list_empty_binds_scalarish_param(self) -> None:
        t = hnl_to_term(
            "do {\n  w :: tensor.from_list([])\n  tensor.add(w, y)\n}"
        )
        assert isinstance(t.args[0], Param)
        assert t.args[0].typ.shape == (0,)

    # -- handler edge branches --------------------------------------------

    @pytest.mark.parametrize(
        "src,check",
        [
            (
                "tensor.batch_norm(x, w, b, m, v, 0.001)",
                lambda t: t.attrs["eps"] == pytest.approx(1e-3),
            ),
            (
                "tensor.max_pool(x, [2, 2])",
                lambda t: t.attrs == {"kernel_size": [2, 2]},
            ),
            (
                "tensor.flatten(x)",
                lambda t: t.attrs == {},
            ),
            (
                "tensor.flatten(x, 1, 2)",
                lambda t: t.attrs == {"start_dim": 1, "end_dim": 2},
            ),
            (
                "tensor.layer_norm(x, [4])",
                lambda t: t.attrs == {"dim": [4]} and len(t.args) == 1,
            ),
            (
                "tensor.layer_norm(x, [4], w)",
                lambda t: t.attrs == {"dim": [4]} and len(t.args) == 2,
            ),
            (
                "tensor.rms_norm(x, [4], w, 0.1)",
                lambda t: (
                    t.attrs == {"dim": [4], "eps": pytest.approx(0.1)}
                ),
            ),
            (
                "tensor.clamp(x, 0)",
                lambda t: t.attrs == {"min": 0},
            ),
            (
                "tensor.clamp(x, null, 1)",
                lambda t: t.attrs == {"max": 1},
            ),
            (
                "tensor.clamp(x, y, z)",
                lambda t: len(t.args) == 3,
            ),
            (
                "tensor.nan_to_num(x, 1, 2, 3)",
                lambda t: (
                    t.attrs == {"nan": 1, "posinf": 2, "neginf": 3}
                ),
            ),
            ("tensor.elu(x)", lambda t: t.attrs == {}),
            (
                "tensor.repeat(x, [2, 3])",
                lambda t: t.attrs == {"shape": [2, 3]},
            ),
            (
                "tensor.full([2])",
                lambda t: t.attrs == {"shape": [2]},
            ),
            ("tensor.full_like(x)", lambda t: t.attrs == {}),
            (
                "tensor.full_like(x, 2)",
                lambda t: t.attrs == {"fill_value": 2},
            ),
            (
                "tensor.eye(3)",
                lambda t: t.attrs == {"dim": 3},
            ),
            ("tensor.var(x)", lambda t: t.attrs == {}),
            (
                "tensor.std(x, [0], true, true)",
                lambda t: (
                    t.attrs
                    == {
                        "dim": [0],
                        "correction": 1,
                        "keepdim": True,
                    }
                ),
            ),
            (
                "tensor.sum(x, null, true)",
                lambda t: t.attrs == {"keepdim": True},
            ),
            (
                "tensor.sum(x, 0)",
                lambda t: t.attrs == {"dim": 0},
            ),
            (
                "tensor.gelu(x)",
                lambda t: t.attrs == {},
            ),
        ],
    )
    def test_handler_edge_branches(self, src: str, check) -> None:
        t = hnl_to_term(src)
        assert isinstance(t, Op)
        assert check(t)

    def test_stack_empty_operand_list_declines(self) -> None:
        with pytest.raises(UnsupportedOpError, match="empty"):
            hnl_to_term("tensor.stack([], 0)")

    def test_bias_without_weight_declines(self) -> None:
        with pytest.raises(UnsupportedOpError, match="bias"):
            hnl_to_term("tensor.layer_norm(x, [4], null, b)")

    def test_norm_non_list_shape_declines(self) -> None:
        with pytest.raises(UnsupportedOpError, match="list"):
            hnl_to_term("tensor.layer_norm(x, 4, w)")

    def test_nested_list_dim_is_not_a_number(self) -> None:
        """``_num`` rejects non-scalar literals by name."""
        with pytest.raises(UnsupportedOpError, match="number"):
            hnl_to_term("tensor.softmax(x, [[0]])")

    def test_unclosed_do_block_is_syntax_error(self) -> None:
        with pytest.raises(HnlImportError, match="expected"):
            hnl_to_term("do { tensor.add(x, y)")

    def test_non_name_call_target_is_syntax_error(self) -> None:
        with pytest.raises(HnlImportError, match="call target"):
            hnl_to_term("(tensor.add(x, y))(z)")

    def test_dim_rejects_bool_and_string(self) -> None:
        for src in ("tensor.sum(x, true)", 'tensor.sum(x, "d")'):
            with pytest.raises(UnsupportedOpError, match="dim"):
                hnl_to_term(src)

    def test_non_tensor_binding_declines_at_use(self) -> None:
        with pytest.raises(UnsupportedOpError, match=r"foo\.bar"):
            hnl_to_term(
                "do {\n  w :: foo.bar(x)\n  tensor.add(w, y)\n}"
            )

    def test_zeros_binding_bad_shape_declines(self) -> None:
        with pytest.raises(UnsupportedOpError, match="shape"):
            hnl_to_term(
                "do {\n  z :: tensor.zeros(x)\n  tensor.add(z, y)\n}"
            )

    def test_from_list_with_term_data_declines(self) -> None:
        with pytest.raises(UnsupportedOpError, match="from_list"):
            hnl_to_term(
                "do {\n  w :: tensor.from_list([x])\n"
                "  tensor.add(w, y)\n}"
            )

    def test_batch_norm_without_eps(self) -> None:
        t = hnl_to_term("tensor.batch_norm(x, w, b, m, v)")
        assert t.attrs == {"training": False}

    def test_clamp_side_term_bound(self) -> None:
        t = hnl_to_term("tensor.clamp_min(x, y)")
        assert t.args[1].name == "y"
        assert t.attrs == {}


# ---------------------------------------------------------------------------
#  Surface-projection honesty
# ---------------------------------------------------------------------------


class TestSurfaceHonesty:
    def test_synthesized_operands_decline(self) -> None:
        """batch_norm arity<5 emits tensor.zeros(shape(...)) — no term."""
        from mlopt_core.ir import Op as _Op

        bn = _Op.make(
            "batch_norm",
            _x(),
            _w(),
            _w(),
            _w(),
            training=False,
        )
        src = emit(bn)  # synthesizes tensor.zeros(tensor.shape(...))
        assert "tensor.zeros" in src
        with pytest.raises(UnsupportedOpError):
            hnl_to_term(src)

    def test_marshaled_from_list_in_term_position_declines(
        self,
    ) -> None:
        with pytest.raises(UnsupportedOpError):
            hnl_to_term("tensor.add(tensor.from_list([1]), x)")

    def test_ragged_from_list_binding_falls_through(self) -> None:
        """A ragged 'shape' is not a Param — the binding imports as a
        term and declines on tensor.from_list instead."""
        with pytest.raises(UnsupportedOpError):
            hnl_to_term(
                "do {\n"
                "  w :: tensor.from_list([[1.0], [2.0, 3.0]])\n"
                "  tensor.add(x, w)\n"
                "}"
            )


# ---------------------------------------------------------------------------
#  Drift guard — covered ops are real mlopt IR ops
# ---------------------------------------------------------------------------


class TestCoveredSetDrift:
    def test_covered_ops_are_ir_ops(self) -> None:
        """Every handler name is a HOST_BINDINGS key or the three
        surface ops the executor table does not carry."""
        pytest.importorskip("mlopt_host_tensor")
        from mlopt_host_tensor.bindings import HOST_BINDINGS

        # Tensor-name space: ``batch_norm``/``max_pool`` are the host
        # spellings of IR ``batch_norm``/``max_pool2d`` (neither is a
        # HOST_BINDINGS key), and ``t`` is the derived transpose
        # spelling — every other covered name IS the IR op name.
        extra = {"batch_norm", "max_pool", "t"}
        known = set(HOST_BINDINGS) | extra
        unknown = covered_ops() - known
        assert not unknown, (
            f"covered ops with no IR spelling: {unknown}"
        )

    def test_covered_ops_nonempty_and_shaped(self) -> None:
        ops = covered_ops()
        assert len(ops) > 100
        assert "add" in ops and "conv2d" in ops and "rsqrt" in ops

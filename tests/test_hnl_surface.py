"""``mlopt_discovery.hnl_surface`` — the term→``.hnl`` emitter.

Unit-level pins for the emit machinery moved out of
``bench/drivers/kit/term_hnl.py`` (the e2e kit eval lives in
``tests/test_kit_bridge.py``): the ``_SPECS`` table's slot
rendering, the literal rules, the program prelude's feed forms, and
the compat shim's identity with the shipped module.
"""

from __future__ import annotations

import pytest
from mlopt_core.ir import Const, Op, Param, TensorType, Var
from mlopt_discovery.hnl_surface import (
    KitEvalError,
    KitRunner,
    UnsupportedOpError,
    emit,
    emit_program,
    leaf_names,
)


def _x() -> Var:
    return Var("x", TensorType((2, 3)))


def _w() -> Param:
    return Param("w", TensorType((3, 4)))


# ---------------------------------------------------------------------------
#  The op table
# ---------------------------------------------------------------------------


class TestEmitTable:
    def test_elementwise_binary(self) -> None:
        assert emit(Op.make("add", _x(), _w())) == "tensor.add(x, w)"
        assert emit(Op.make("mul", _x(), _w())) == "tensor.mul(x, w)"

    def test_matmul(self) -> None:
        assert (
            emit(Op.make("matmul", _x(), _w())) == "tensor.matmul(x, w)"
        )

    def test_conv2d_full_signature(self) -> None:
        t = Op.make(
            "conv2d",
            _x(),
            _w(),
            _w(),
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
        )
        assert emit(t) == ("tensor.conv2d(x, w, w, [1, 1], [1, 1], 1)")

    def test_conv2d_positional_scalars(self) -> None:
        """Aten-style positional scalar tail folds to named attrs."""
        t = Op.make(
            "conv2d",
            _x(),
            _w(),
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        # no bias arg → the (_OPT, 2) slot spells `null`
        assert emit(t) == "tensor.conv2d(x, w, null, [1, 1], [1, 1], 2)"

    def test_conv2d_positional_attr_conflict_declines(self) -> None:
        t = Op.make(
            "conv2d",
            _x(),
            _w(),
            (2, 2),
            stride=[1, 1],
        )
        with pytest.raises(UnsupportedOpError, match="both"):
            emit(t)

    def test_conv2d_dilation_nondefault_declines(self) -> None:
        t = Op.make("conv2d", _x(), _w(), dilation=[2, 2])
        with pytest.raises(UnsupportedOpError, match="dilation"):
            emit(t)

    def test_batch_norm_full_arity(self) -> None:
        t = Op.make(
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
        )
        # momentum/cudnn drop (inert in eval); eps spells without sci-notation
        assert emit(t) == ("tensor.batch_norm(x, w, w, w, w, 0.00001)")

    def test_batch_norm_training_declines(self) -> None:
        t = Op.make(
            "batch_norm",
            _x(),
            _w(),
            _w(),
            _w(),
            _w(),
            training=True,
            eps=1e-5,
        )
        with pytest.raises(UnsupportedOpError, match="training"):
            emit(t)

    @pytest.mark.parametrize(
        "args,expect",
        [
            # 5 args: full affine — x plus four rank-1 tensors
            (
                (_w(), _w(), _w(), _w()),
                "tensor.batch_norm(x, w, w, w, w, 0.00001)",
            ),
            # 4 args: beta synthesized as zeros(mean.shape)
            (
                (_w(), _w(), _w()),
                "tensor.batch_norm(x, w, "
                "tensor.zeros(tensor.shape(w)), w, w, 0.00001)",
            ),
            # 3 args: gamma=ones/beta=zeros on the mean's shape
            (
                (_w(), _w()),
                "tensor.batch_norm(x, tensor.ones(tensor.shape(w)), "
                "tensor.zeros(tensor.shape(w)), w, w, 0.00001)",
            ),
        ],
    )
    def test_batch_norm_arity_synthesis(
        self, args: tuple, expect: str
    ) -> None:
        t = Op.make("batch_norm", _x(), *args, training=False, eps=1e-5)
        assert emit(t) == expect

    def test_batch_norm_bad_arity_declines(self) -> None:
        # arities 3/4/5 synthesize; 2 and 6 have no surface signature
        for args in ((_w(),), (_w(),) * 5):
            t = Op.make("batch_norm", _x(), *args)
            with pytest.raises(UnsupportedOpError, match="arity"):
                emit(t)

    def test_max_pool2d(self) -> None:
        t = Op.make(
            "max_pool2d",
            _x(),
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=0,
            dilation=1,
            ceil_mode=False,
        )
        assert emit(t) == "tensor.max_pool(x, [2, 2], [2, 2])"

    def test_max_pool2d_nondefault_padding_declines(self) -> None:
        t = Op.make("max_pool2d", _x(), kernel_size=[2, 2], padding=1)
        with pytest.raises(UnsupportedOpError, match="padding"):
            emit(t)

    def test_reshape_positional_and_attr(self) -> None:
        a = Op.make("reshape", _x(), (6, 1))
        b = Op.make("reshape", _x(), shape=(6, 1))
        assert emit(a) == emit(b) == "tensor.reshape(x, [6, 1])"

    def test_transpose(self) -> None:
        t = Op.make("transpose", _x(), dim0=0, dim1=1)
        assert emit(t) == "tensor.transpose(x, 0, 1)"

    def test_concat_and_pack(self) -> None:
        t = Op.make("concat", _x(), _w(), dim=0)
        assert emit(t) == "tensor.concat([x, w], 0)"
        p = Op.make("pack", _x(), _w())
        assert emit(p) == "[x, w]"

    def test_unknown_op_names_itself(self) -> None:
        t = Op.make("sigmoid", _x(), validate=False)
        with pytest.raises(UnsupportedOpError, match="sigmoid"):
            emit(t)

    def test_scalar_tail_without_name_declines(self) -> None:
        t = Op.make("add", _x(), Const(2.0), validate=False)
        with pytest.raises(UnsupportedOpError, match="positional"):
            emit(t)

    def test_unemittable_attr_declines(self) -> None:
        # a fresh object() attr keeps the term out of the intern pool
        t = Op.make("transpose", _x(), dim0=0, dim1=1, weird=object())
        with pytest.raises(UnsupportedOpError, match="no slot"):
            emit(t)


# ---------------------------------------------------------------------------
#  Literals — the handl lexer has no scientific notation
# ---------------------------------------------------------------------------


class TestLiterals:
    def test_epsilon_expands(self) -> None:
        t = Op.make(
            "batch_norm",
            _x(),
            _w(),
            _w(),
            _w(),
            _w(),
            training=False,
            eps=1e-5,
        )
        assert "0.00001" in emit(t)
        assert "1e-05" not in emit(t)

    def test_tiny_and_large(self) -> None:
        t = Op.make(
            "batch_norm",
            _x(),
            _w(),
            _w(),
            _w(),
            _w(),
            training=False,
            eps=1e-20,
        )
        assert "0.00000000000000000001" in emit(t)

    def test_nonfinite_declines(self) -> None:
        with pytest.raises(UnsupportedOpError, match="non-finite"):
            emit(Op.make("concat", _x(), _w(), dim=float("nan")))

    def test_bool_literal(self) -> None:
        t = Op.make(
            "max_pool2d",
            _x(),
            kernel_size=[2, 2],
            ceil_mode=True,
        )
        with pytest.raises(UnsupportedOpError, match="ceil_mode"):
            emit(t)  # accepts pins False only


# ---------------------------------------------------------------------------
#  Leaf naming and the program prelude
# ---------------------------------------------------------------------------


class TestLeavesAndProgram:
    def test_leaf_names_first_use_order(self) -> None:
        t = Op.make(
            "add",
            Op.make("mul", _x(), _w()),
            Op.make("sub", _x(), _w()),
        )
        assert leaf_names(t) == ["x", "w"]

    def test_sanitize_dotted_and_reserved(self) -> None:
        t = Op.make(
            "add",
            Param("conv.weight", TensorType((2,))),
            Var("tensor", TensorType(None)),
        )
        src = emit(t)
        assert "conv_weight" in src
        assert "v_tensor" in src  # 'tensor' is shadowed-safe

    def test_emit_program_data_feed(self) -> None:
        import torch  # noqa: F401  (shape/data only need lists)

        t = Op.make("matmul", _x(), _w())
        src = emit_program(
            t,
            {
                "x": {"shape": [2, 3], "data": list(range(6))},
                "w": {"shape": [3, 4], "data": list(range(12))},
            },
        )
        assert src.startswith("do {")
        assert "tensor.reshape(tensor.from_list" in src
        assert src.rstrip().endswith("tensor.matmul(x, w)\n}")

    def test_emit_program_hnl_channel(self) -> None:
        """``{"hnl": expr}`` binds verbatim — the marshaling channel."""
        t = Op.make("mul", _w(), Param("s", TensorType((3, 4))))
        src = emit_program(
            t,
            {
                "w": {"shape": [3, 4], "data": [0.0] * 12},
                "s": {"hnl": "tensor.from_list([[1.0]])"},
            },
        )
        assert "s :: tensor.from_list([[1.0]])" in src

    def test_emit_program_nested_list_feed(self) -> None:
        t = Op.make("add", _x(), Param("b", TensorType((2, 3))))
        src = emit_program(t, {"x": [1, 2, 3, 4, 5, 6], "b": [0.0] * 6})
        assert "tensor.from_list([1, 2, 3, 4, 5, 6])" in src

    def test_emit_program_missing_leaf_names_it(self) -> None:
        t = Op.make("add", _x(), _w())
        with pytest.raises(ValueError, match="w"):
            emit_program(t, {"x": {"shape": [2, 3], "data": [0.0] * 6}})


# ---------------------------------------------------------------------------
#  The compat shim — bench.drivers.kit.term_hnl IS hnl_surface
# ---------------------------------------------------------------------------


class TestCompatShim:
    def test_shim_is_the_same_objects(self) -> None:
        from mlopt_discovery import hnl_surface as canon

        from bench.drivers.kit import term_hnl as shim

        for name in (
            "KitEvalError",
            "KitRunner",
            "UnsupportedOpError",
            "emit",
            "emit_program",
            "leaf_names",
        ):
            assert getattr(shim, name) is getattr(canon, name)

    def test_shim_emit_matches(self) -> None:
        from bench.drivers.kit import term_hnl as shim

        t = Op.make("matmul", _x(), _w())
        assert shim.emit(t) == "tensor.matmul(x, w)"

    def test_kit_runner_missing_binary_is_named(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="hnl-run"):
            KitRunner(tmp_path / "nope")


# ---------------------------------------------------------------------------
#  KitRunner — a stub binary stands in for hnl-run
# ---------------------------------------------------------------------------


class TestKitRunner:
    """The subprocess door, exercised against a shell stub — the real
    binary's e2e coverage lives in ``tests/test_kit_bridge.py``."""

    @pytest.fixture()
    def stub(self, tmp_path):
        """A fake ``hnl-run``: echoes a fixed JSON boundary value."""
        import stat

        path = tmp_path / "hnl-run"
        path.write_text(
            "#!/bin/sh\n"
            "cat > /dev/null\n"
            'echo \'{"__tensor__": {"shape": [1], "data": [1.0]}}\'\n'
        )
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def test_run_returns_decoded_json(self, stub) -> None:
        r = KitRunner(stub)
        assert r.run("tensor.add(x, y)") == {
            "__tensor__": {"shape": [1], "data": [1.0]}
        }

    def test_run_passes_project_and_repeat(self, stub) -> None:
        """--project/--repeat land on the command line."""
        wrap = stub.parent / "wrap"
        wrap.write_text(
            "#!/bin/sh\n"
            f'echo "$@" > {stub.parent}/argv.txt\n'
            "cat > /dev/null\n"
            "echo '{}'\n"
        )
        wrap.chmod(0o755)
        KitRunner(wrap, project="/proj").run("src", repeat=3)
        argv = (stub.parent / "argv.txt").read_text()
        assert "--project /proj" in argv
        assert "--repeat 3" in argv

    def test_eval_and_tensor(self, stub) -> None:
        r = KitRunner(stub)
        out = r.eval(
            Op.make("add", _x(), _w()),
            {
                "x": {"shape": [2, 3], "data": [0.0] * 6},
                "w": {"shape": [3, 4], "data": [0.0] * 12},
            },
        )
        assert out["__tensor__"]["data"] == [1.0]
        t = r.tensor(
            Op.make("add", _x(), _w()),
            {
                "x": {"shape": [2, 3], "data": [0.0] * 6},
                "w": {"shape": [3, 4], "data": [0.0] * 12},
            },
        )
        assert t["shape"] == [1]

    def test_tensor_non_tensor_result_raises(
        self, stub, tmp_path
    ) -> None:
        bad = tmp_path / "bad"
        bad.write_text("#!/bin/sh\ncat > /dev/null\necho '{}\n'")
        bad.chmod(0o755)
        with pytest.raises(KitEvalError, match="tensor result"):
            KitRunner(bad).tensor(
                Op.make("add", _x(), _w()), {"x": [], "w": []}
            )

    def test_nonzero_exit_raises_kit_eval_error(self, tmp_path) -> None:
        bad = tmp_path / "bad"
        bad.write_text(
            "#!/bin/sh\ncat > /dev/null\necho boom >&2\nexit 3\n"
        )
        bad.chmod(0o755)
        with pytest.raises(KitEvalError, match="boom"):
            KitRunner(bad).run("tensor.foo(x)")


# ---------------------------------------------------------------------------
#  emit() leaf/fallthrough branches
# ---------------------------------------------------------------------------


class TestEmitBranches:
    def test_bool_attr_literal(self) -> None:
        t = Op.make("reshape", _x(), shape=True)
        assert "true" in emit(t)

    def test_tuple_shape_attr(self) -> None:
        t = Op.make("reshape", _x(), shape=(6, 1))
        assert emit(t) == "tensor.reshape(x, [6, 1])"

    def test_unemittable_slot_value(self) -> None:
        t = Op.make("reshape", _x(), shape=object())
        with pytest.raises(UnsupportedOpError, match="unemittable"):
            emit(t)

    def test_leaf_name_sanitize_collision(self) -> None:
        t = Op.make(
            "add",
            Param("a.b", TensorType((2,))),
            Param("a_b", TensorType((2,))),
        )
        src = emit(t)
        assert "a_b" in src and "a_b_" in src

    def test_missing_operand_slot(self) -> None:
        t = Op.make("conv2d", _x(), validate=False)
        with pytest.raises(UnsupportedOpError, match="missing operand"):
            emit(t)

    def test_emit_const_and_scalars(self) -> None:
        assert emit(Const(2.5)) == "2.5"
        assert emit(3) == "3"
        assert emit(None) == "null"
        assert emit([1, 2]) == "[1, 2]"
        assert emit((1, 2)) == "[1, 2]"

    def test_emit_raw_string_and_object_decline(self) -> None:
        with pytest.raises(UnsupportedOpError, match="metavar"):
            emit("x")
        with pytest.raises(UnsupportedOpError, match="unemittable"):
            emit(object())

    def test_leaf_names_non_op_leaf(self) -> None:
        assert leaf_names(Const(1.0)) == []
        assert leaf_names(_x()) == ["x"]

    def test_str_literal_inside_list_attr(self) -> None:
        """``_lit``'s string branch — a str inside a list attr."""
        t = Op.make("reshape", _x(), shape=["a", 2])
        assert '"a"' in emit(t)

    def test_slot_lit_and_bad_slot_kinds(self) -> None:
        """``_slot``'s literal and unknown-kind branches, directly."""
        from mlopt_discovery.hnl_surface import _slot

        assert _slot(("lit", 5), [], {}, lambda t: "?") == "5"
        with pytest.raises(UnsupportedOpError, match="bad emit slot"):
            _slot(("bogus", 1), [], {}, lambda t: "?")  # type: ignore[arg-type]

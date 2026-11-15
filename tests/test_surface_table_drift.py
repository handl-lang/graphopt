"""Codec-table drift guards — ADR-0174 invariant 4 (audit F5).

The interim consolidation rule (``host/project/adrs/0174-…`` §4):
*every* term↔surface table is either generated from
``host_surface.json`` or carries a drift guard against it.
``mlopt.emit._OPS`` got the behavioral guard
(``mlopt/tests/test_emit_surface.py``); ``HOST_BINDINGS`` got the
key-existence guard (``test_host_surface.py``).  This file closes
the remaining three:

* ``mlopt_discovery.hnl_surface._SPECS`` — the 12-op measurement
  table.  Forward guard: every spec's ``target`` is a real
  ``tensor.*`` op in the artifact with an emittable status, and a
  canonical probe term per op emits a primary call whose arg count
  sits inside the port's signature arity (concrete signatures only —
  ``Value...`` ports are variadic by declaration).
* ``mlopt_discovery.modelproj`` — the spec-side emitter's three
  tables (``_BINARY_OPS`` / ``_ATTR_OPS`` / ``_FIXED_OPS`` plus the
  ``concat`` special case).  Same forward guard, driven through the
  real ``spec_to_hnl`` renderer.
* ``mlopt_torch.torch_bindings`` — the IR→torch lowering table, the
  torch sink's sibling of ``HOST_BINDINGS``.  Two guards: every op in
  ``_LOCAL_ATTR_SCHEMA_OVERRIDES`` (declared "the host-tensor port
  surface's scalar tails") either has a ``tensor.*`` port in the
  artifact or sits in the pinned torch-only exception list; and every
  ``HOST_BINDINGS`` key is torch-bound — the two sink lowerings may
  not silently diverge (``view`` is the one pinned alias: torch
  canonicalises it to ``reshape`` at export).

The probes are *behavioral* — they run the real emitters and read the
emitted ``.hnl`` text — so a table entry that silently retargets a
renamed/removed surface op fails here; there is no parallel name
table to drift.
"""

from __future__ import annotations

import re
from typing import Any

from mlopt_core.ir import Op, TensorType, Var
from mlopt_host_tensor.surface import host_ops

_T = TensorType((2, 3))


def _v(name: str) -> Var:
    return Var(name, _T)


_X = _v("x")
_W = _v("w")

_EMITTABLE = {"Ported", "Handwritten"}

_CALL = re.compile(r"tensor\.(\w+)\(")


def _primary_call(expr: str) -> tuple[str, int] | None:
    """``(tensor-op-name, top-level arg count)`` of the first call.

    Returns ``None`` when the expression spells no ``tensor.*`` call
    (``hnl_surface``'s ``pack`` lands as a bare ``.hnl`` list literal).
    """
    m = _CALL.search(expr)
    if m is None:
        return None
    depth = 1
    instr = esc = False
    i = m.end()
    while i < len(expr) and depth:
        c = expr[i]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
        elif c == '"':
            instr = True
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        i += 1
    inner = expr[m.end() : i - 1]
    argc = 0
    depth = 0
    instr = esc = False
    for c in inner:
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
            continue
        if c == '"':
            instr = True
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            argc += 1
    return m.group(1), argc + (1 if inner.strip() else 0)


def _sig_arity(signature: str) -> int | None:
    """Parameter count of a surface signature, ``None`` if variadic.

    ``"(Tensor * (List Int)) -> Tensor"`` → 2; ``"Value... -> Tensor"``
    → ``None`` (the handwritten variadic ports pin existence + status,
    not arity — same posture as ``mlopt/tests/test_emit_surface.py``).
    """
    params = signature.split("->", 1)[0].strip()
    if params.startswith("(") and params.endswith(")"):
        params = params[1:-1]
    if "..." in params:
        return None
    parts: list[str] = []
    cur = ""
    depth = 0
    for ch in params:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "*" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts) if parts else None


def _check_emitted(
    op_name: str, expr: str, ops: dict[str, dict[str, Any]]
) -> None:
    """Assert *expr*'s primary ``tensor.*`` call is a real surface op."""
    call = _primary_call(expr)
    assert call is not None, f"{op_name}: no tensor.* call in {expr!r}"
    tensor_op, argc = call
    key = f"tensor.{tensor_op}"
    entry = ops.get(key)
    assert entry is not None, (
        f"{op_name}: emits {key} — not in host_surface.json ({expr})"
    )
    assert entry["status"] in _EMITTABLE, (
        f"{op_name}: {key} status={entry['status']!r} — not emittable"
    )
    want = _sig_arity(entry["signature"])
    if want is not None:
        assert argc == want, (
            f"{op_name}: {key} emitted {argc} args, signature "
            f"{entry['signature']!r} wants {want} ({expr})"
        )
    lo, hi = entry["min_args"], entry["max_args"]
    if lo is not None:
        assert argc >= lo, (
            f"{op_name}: {key} emitted {argc} args < min_args={lo}"
        )
    if hi is not None:
        assert argc <= hi, (
            f"{op_name}: {key} emitted {argc} args > max_args={hi}"
        )


# ---------------------------------------------------------------------------
#  hnl_surface._SPECS — the measurement table
# ---------------------------------------------------------------------------


def _hnl_probe(op: str) -> Op:
    """A canonical term that reaches *op*'s surface call."""
    if op in {"add", "mul", "sub", "div", "matmul"}:
        return Op.make(op, _X, _W)
    if op == "conv2d":
        return Op.make(
            "conv2d",
            _X,
            _W,
            stride=[1, 1],
            padding=[1, 1],
            groups=1,
        )
    if op == "batch_norm":
        return Op.make(
            "batch_norm",
            _X,
            _W,
            _W,
            _W,
            _W,
            training=False,
            eps=1e-5,
        )
    if op == "max_pool2d":
        return Op.make(
            "max_pool2d", _X, kernel_size=[2, 2], stride=[2, 2]
        )
    if op == "reshape":
        return Op.make("reshape", _X, shape=(6, 1))
    if op == "transpose":
        return Op.make("transpose", _X, dim0=0, dim1=1)
    if op == "concat":
        return Op.make("concat", _X, _W, dim=0)
    if op == "pack":
        return Op.make("pack", _X, _W)
    raise AssertionError(f"no probe for _SPECS op {op!r}")


class TestHnlSurfaceSpecs:
    def test_every_spec_target_is_a_host_op(self) -> None:
        """Forward guard — each ``_SPECS`` target exists + emittable."""
        from mlopt_discovery.hnl_surface import _SPECS, emit

        ops = host_ops()
        assert ops, "host_surface.json loaded empty — dep broke"
        for name, spec in sorted(_SPECS.items()):
            if not spec.target:
                # ``pack`` special-cases to a bare ``.hnl`` list
                # literal — the multi-output root has no tensor.* op.
                expr = emit(_hnl_probe(name))
                assert _primary_call(expr) is None, (
                    f"{name}: expected a list literal, got {expr!r}"
                )
                continue
            expr = emit(_hnl_probe(name))
            _check_emitted(name, expr, ops)
            call = _primary_call(expr)
            assert call is not None and call[0] == spec.target, (
                f"{name}: spec targets tensor.{spec.target} but "
                f"emitted {call} ({expr})"
            )

    def test_spec_targets_match_declared_table(self) -> None:
        """The target column itself is pinned — no silent retarget."""
        from mlopt_discovery.hnl_surface import _SPECS

        targets = {n: s.target for n, s in _SPECS.items()}
        assert targets == {
            "add": "add",
            "mul": "mul",
            "sub": "sub",
            "div": "div",
            "matmul": "matmul",
            "conv2d": "conv2d",
            "batch_norm": "batch_norm",
            "max_pool2d": "max_pool",
            "reshape": "reshape",
            "transpose": "transpose",
            "concat": "concat",
            "pack": "",
        }

    def test_claimed_surface_coverage_ops_exist(self) -> None:
        """The ops the emitter/prelude itself spells are real ports.

        ``emit_program`` spells ``zeros``/``ones``/``from_list``/
        ``reshape``/``shape`` and the module docstring names the
        marshaling four plus ``to_list``/``derive``/
        ``batch_norm_backward`` — every one must exist in the
        artifact, so the claimed coverage cannot silently rot.
        """
        ops = host_ops()
        for name in (
            "zeros",
            "ones",
            "from_list",
            "to_list",
            "reshape",
            "shape",
            "derive",
            "batch_norm_backward",
        ):
            assert f"tensor.{name}" in ops, name


# ---------------------------------------------------------------------------
#  modelproj — the spec-side emitter's op tables
# ---------------------------------------------------------------------------


def _model_probe(op: str) -> tuple[Any, list[str]]:
    """A spec node + metavar list that reaches *op*'s surface call."""
    if op in {"add", "sub", "mul", "div", "matmul"}:
        return (op, "a", "b"), ["a", "b"]
    if op == "concat":
        return (op, "a", "b", {"dim": 0}), ["a", "b"]
    if op == "linear":
        return (op, "a", "b", "c"), ["a", "b", "c"]
    if op == "reshape":
        return (op, "a", {"shape": [2, 3]}), ["a"]
    if op == "transpose":
        return (op, "a", {"dim0": 0, "dim1": 1}), ["a"]
    if op == "unsqueeze":
        return (op, "a", {"dim": 0}), ["a"]
    if op == "select":
        return (op, "a", {"dim": 0, "index": 1}), ["a"]
    raise AssertionError(f"no probe for modelproj op {op!r}")


class TestModelprojTables:
    def test_every_table_op_is_a_host_op(self) -> None:
        """Forward guard — each modelproj op lands on a real port."""
        from mlopt_discovery import modelproj as mp

        ops = host_ops()
        table_ops = (
            set(mp._BINARY_OPS)
            | set(mp._ATTR_OPS)
            | set(mp._FIXED_OPS)
            | {"concat"}
        )
        assert table_ops, "modelproj tables are empty — import broke"
        for op in sorted(table_ops):
            spec, mvars = _model_probe(op)
            expr = mp.spec_to_hnl(spec, mvars=mvars)
            _check_emitted(op, expr, ops)
            call = _primary_call(expr)
            assert call is not None and call[0] == op, (
                f"{op}: emitted {call} — spec retargeted ({expr})"
            )

    def test_table_membership_is_pinned(self) -> None:
        """The covered-op set itself is pinned — additions are loud."""
        from mlopt_discovery import modelproj as mp

        covered = (
            set(mp._BINARY_OPS)
            | set(mp._ATTR_OPS)
            | set(mp._FIXED_OPS)
            | {"concat"}
        )
        assert covered == {
            "add",
            "sub",
            "mul",
            "div",
            "matmul",
            "concat",
            "linear",
            "reshape",
            "transpose",
            "unsqueeze",
            "select",
        }


# ---------------------------------------------------------------------------
#  torch_bindings — the torch sink's lowering table
# ---------------------------------------------------------------------------

#: ``_LOCAL_ATTR_SCHEMA_OVERRIDES`` ops with NO ``tensor.*`` port —
#: torch-adapter vocabulary, not host-tensor ports.  ``weight_norm``
#: is explicitly adapter-declared (``torch_bindings.py`` registers it
#: through ``opmeta.register_op_meta``); ``embedding``'s ATen port is
#: simply absent from the 238-op surface.  Pinned so a *new* override
#: op must either land a host port or join this list deliberately.
_TORCH_ONLY_ATTR_OPS = frozenset({"embedding", "weight_norm"})

#: ``HOST_BINDINGS`` keys torch does not bind under the same name.
#: ``view`` is the aten alias — export canonicalises ``view.default``
#: to ``reshape`` (``_IR_TO_TORCH_EXTRA``) and ``HOST_BINDINGS``
#: binds both spellings to ``_reshape_view``.
_TORCH_ALIAS_OPS = frozenset({"view"})


class TestTorchBindingsSurface:
    def test_attr_schema_overrides_have_host_ports_or_declared_exceptions(
        self,
    ) -> None:
        """``_LOCAL_ATTR_SCHEMA_OVERRIDES`` claims host-port parity.

        The table's own comment scopes it to "the host-tensor port
        surface's scalar tails" — every key must therefore name a
        ``tensor.*`` op in the artifact, bar the pinned torch-only
        exceptions.
        """
        from mlopt_torch.torch_bindings import (
            _LOCAL_ATTR_SCHEMA_OVERRIDES,
        )

        ops = host_ops()
        missing = sorted(
            op
            for op in _LOCAL_ATTR_SCHEMA_OVERRIDES
            if f"tensor.{op}" not in ops
        )
        unexpected = set(missing) - _TORCH_ONLY_ATTR_OPS
        assert not unexpected, (
            "_LOCAL_ATTR_SCHEMA_OVERRIDES names ops with no "
            "tensor.* port and no declared exception: "
            f"{sorted(unexpected)}"
        )

    def test_host_bindings_are_torch_bound(self) -> None:
        """Sink parity — every host-lowerable IR op lowers on torch.

        ``HOST_BINDINGS`` and ``_IR_TO_TORCH`` are the same-role
        lowering tables for the two sinks; an op the host sink claims
        must not silently lack a torch binding (delivery would diverge
        by backend).  Membership uses ``in`` so lazy carrier bindings
        resolve — ``eye`` binds through a carrier module, not
        ``_CORE_TORCH_BINDINGS``.
        """
        from mlopt_host_tensor.bindings import HOST_BINDINGS
        from mlopt_torch.torch_bindings import _IR_TO_TORCH

        missing = sorted(
            op for op in HOST_BINDINGS if op not in _IR_TO_TORCH
        )
        unexpected = set(missing) - _TORCH_ALIAS_OPS
        assert not unexpected, (
            "HOST_BINDINGS lowers ops torch cannot bind: "
            f"{sorted(unexpected)}"
        )

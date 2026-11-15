"""The lowered runnable — a term interpreter over the host registry.

``HostTensorSink.lower`` materialises an IR as a
:class:`HostTensorExecutor`: the root term, the leaf values (``Param``
values passed to ``lower`` plus every ``Const`` literal), and the
``HOST_BINDINGS`` table that translates each ``Op`` node's attrs into
the host op's positional JSON args.  ``forward(*xs)`` evaluates the
term bottom-up through ``handl_host_tensor.call`` — one extension
call per op node, wire records on both sides.

Term coverage: ``Var`` (bound positionally to ``forward`` args),
``Const``/``Param`` leaves, ``Op`` nodes in :data:`HOST_BINDINGS`,
and the IR-internal ``pack`` pairing (multi-output roots — it is *not*
a host op and not in ``supported_ops``; the evaluator expands it
directly).  Anything else — an op outside the table, a ``getitem``
consumer of a tuple-valued op — raises
:class:`~mlopt_host_tensor.errors.UnsupportedHostOpError` loudly.
"""

from __future__ import annotations

from typing import Any

from mlopt_core.ir import Const, Op, Param, Var

from mlopt_host_tensor import _binding
from mlopt_host_tensor.bindings import HOST_BINDINGS, _Indexed
from mlopt_host_tensor.errors import (
    HostTensorError,
    UnsupportedHostOpError,
)
from mlopt_host_tensor.wire import decode_result

__all__ = ["HostTensorExecutor"]


class HostTensorExecutor:
    """A lowered IR program, executable through host-tensor.

    Parameters
    ----------
    root : Any
        The IR root term (``Var``/``Const``/``Param``/``Op``).
    inputs : tuple[Var, ...]
        The program's input variables, in ``forward`` argument order.
    params : dict[str, Any]
        Leaf values for ``Param`` nodes, keyed by param name.

    """

    def __init__(
        self,
        root: Any,
        inputs: tuple[Var, ...],
        params: dict[str, Any],
    ) -> None:
        """Store the root term, inputs and leaf values."""
        self._root = root
        self._inputs = inputs
        self._params = params

    @property
    def inputs(self) -> tuple[Var, ...]:
        """Return the program's input variables, in call order."""
        return self._inputs

    def forward(self, *xs: Any) -> Any:
        """Run the program on positional inputs ``xs``.

        Each input is array-like; it is wire-encoded per use (a
        shared input re-encodes per consumer — cheap at this scale,
        and keeps leaf handling uniform).
        """
        if len(xs) != len(self._inputs):
            raise HostTensorError(
                f"host-tensor executor: expected {len(self._inputs)} "
                f"input(s), got {len(xs)}"
            )
        env: dict[Any, Any] = dict(zip(self._inputs, xs, strict=True))
        return self._eval(self._root, env, {})

    # ------------------------------------------------------------------

    def _eval(self, term: Any, env: dict[Any, Any], memo: dict) -> Any:
        """Evaluate ``term``; ``memo`` shares results across siblings."""
        if term in memo:
            return memo[term]
        if isinstance(term, Var):
            if term not in env:
                raise HostTensorError(
                    f"host-tensor executor: unbound input {term.name!r}"
                )
            return env[term]
        if isinstance(term, Const):
            return term.value
        if isinstance(term, Param):
            try:
                return self._params[term.name]
            except KeyError:
                raise HostTensorError(
                    f"host-tensor executor: no value for param "
                    f"{term.name!r} (pass params= to lower())"
                ) from None
        if not isinstance(term, Op):
            raise UnsupportedHostOpError(
                f"host-tensor executor: cannot evaluate {type(term).__name__}"
            )
        if term.op == "pack":
            # IR-internal multi-output pairing — expand directly.
            return tuple(self._eval(a, env, memo) for a in term.args)
        try:
            builder = HOST_BINDINGS[term.op]
        except KeyError:
            raise UnsupportedHostOpError(
                f"host-tensor executor: op {term.op!r} is outside "
                f"supported_ops ({len(HOST_BINDINGS)} ops)"
            ) from None
        vals = [self._eval(a, env, memo) for a in term.args]
        try:
            built = builder(*vals, **dict(term.attrs))
        except UnsupportedHostOpError:
            raise
        except Exception as exc:
            raise HostTensorError(
                f"host-tensor executor: {term.op} binding failed: {exc}"
            ) from exc
        out = self._call(term.op, built)
        memo[term] = out
        return out

    @staticmethod
    def _call(op: str, built: Any) -> Any:
        """JSON-call ``tensor.<op>`` and decode; resolve _Indexed."""
        if isinstance(built, _Indexed):
            raw = _binding.call(built.op, built.args)
            outs = decode_result(raw)
            seq = outs if isinstance(outs, (tuple, list)) else (outs,)
            try:
                return seq[built.index]
            except IndexError:
                raise HostTensorError(
                    f"host-tensor {built.op}: index {built.index} "
                    f"out of range ({len(seq)} outputs)"
                ) from None
        raw = _binding.call(op, built)
        return decode_result(raw)

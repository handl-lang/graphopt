"""Shared numpy ``Sink`` double for cross-stack tests.

``NumpySink`` implements ``mlopt_core.ports.Sink`` over numpy: it
lowers core-IR ops to numpy closures, reports a numpy op set as
``supported_ops``, and verifies by running both executables.  Carved
out of ``tests/test_pluggable_sink.py`` when that file moved to
handl-lang/kit-torch in P3 Step D-1e — ``test_search_lower_split``
(stays here for its ``kit_cuda`` edge) still needs the double.
"""

import numpy as np
import torch
import torch.nn as nn
from mlopt_core.ir import Const, Op, Param, Var
from mlopt_core.ops import OpTable
from mlopt_core.ports import ExecutorSpec, Sink

from kit_torch.report import VerifyReport


def _to_np(v):
    if hasattr(v, "detach"):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def _dim(kw, default=-1):
    d = kw.get("dim", kw.get("axis", default))
    if isinstance(d, (list, tuple)):
        return tuple(int(i) for i in d)
    return int(d)


def _transpose(x, **kw):
    d0 = int(kw.get("dim0", -2))
    d1 = int(kw.get("dim1", -1))
    return np.swapaxes(x, d0, d1)


def _linear(x, w, b=None, **kw):
    out = x @ np.swapaxes(w, -1, -2)
    return out if b is None else out + b


_NP_BINDINGS = {
    "add": lambda a, b, **kw: a + b,
    "mul": lambda a, b, **kw: a * b,
    "sub": lambda a, b, **kw: a - b,
    "div": lambda a, b, **kw: a / b,
    "neg": lambda x, **kw: -x,
    "square": lambda x, **kw: x * x,
    "sqrt": lambda x, **kw: np.sqrt(x),
    "exp": lambda x, **kw: np.exp(x),
    "relu": lambda x, **kw: np.maximum(x, 0.0),
    "silu": lambda x, **kw: x / (1.0 + np.exp(-x)),
    "sigmoid": lambda x, **kw: 1.0 / (1.0 + np.exp(-x)),
    "tanh": lambda x, **kw: np.tanh(x),
    "matmul": lambda a, b, **kw: a @ b,
    "linear": _linear,
    "transpose": _transpose,
    "reshape": lambda x, **kw: x.reshape(tuple(kw["shape"])),
    "sum": lambda x, **kw: np.sum(
        x, axis=_dim(kw), keepdims=bool(kw.get("keepdim", False))
    ),
    "mean": lambda x, **kw: np.mean(
        x, axis=_dim(kw), keepdims=bool(kw.get("keepdim", False))
    ),
    "concat": lambda *ts, **kw: np.concatenate(
        list(ts), axis=_dim(kw, 0)
    ),
}


class _NumpyModule:
    """A lowered IR term evaluated with numpy arrays."""

    def __init__(self, ir, params):
        self._ir = ir
        self._params = params

    def forward(self, *xs):
        env = {
            v.name: x for v, x in zip(self._ir.inputs, xs, strict=False)
        }
        return self._eval(self._ir.root, env)

    __call__ = forward

    def _eval(self, t, env):
        if isinstance(t, Var):
            return env[t.name]
        if isinstance(t, Param):
            return self._params[t.name]
        if isinstance(t, Const):
            return np.asarray(t.value)
        args = [self._eval(a, env) for a in t.args]
        return _NP_BINDINGS[t.op](*args, **dict(t.attrs))


class NumpySink:
    """A ``Sink`` whose runtime is numpy."""

    supported_ops = frozenset(_NP_BINDINGS)

    def __init__(self):
        self._ops = OpTable.full()

    @property
    def ops(self):
        return self._ops

    @property
    def executors(self):
        """The generic lowering is this sink's only executor path."""
        return {
            "generic": ExecutorSpec(
                name="generic",
                accepts=lambda ir, term: True,
                lower=self.lower,
                engaged=lambda mod: True,
            )
        }

    def lower(self, ir, params=None):
        return _NumpyModule(
            ir, {k: _to_np(v) for k, v in (params or {}).items()}
        )

    def verify(self, ref, opt, inputs, *, rtol=1e-4, atol=None):
        args = inputs if isinstance(inputs, tuple) else (inputs,)
        args = [_to_np(a) for a in args]
        ref_out = np.asarray(ref(*args))
        out = np.asarray(opt(*args))
        max_abs = float(np.max(np.abs(ref_out - out)))
        max_rel = max_abs / (float(np.max(np.abs(ref_out))) + 1e-8)
        passed = bool(max_rel < rtol) and (
            atol is None or bool(max_abs <= atol)
        )
        return VerifyReport(
            max_abs=max_abs, max_rel=max_rel, passed=passed
        )


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))

# ruff: noqa: RUF002, RUF003 — math notation (≡, σ, −, ·) in docstrings.
"""``KernelSink`` — the kernel codegen backend behind the Sink port.

A peer of ``mlopt_onnx.OnnxSink``, smaller: the backend is a *torch
composite*.  The binding table is a deliberately small op set (the
elementwise + reduction surface the codegen law bank can reach);
each ``kf_*`` fused op lowers to ``torch.compile`` of its composite
body — one fused kernel (Inductor-generated, Triton on CUDA) where
the spelled program launched an op at a time.

Verify-first, per the ONNX precedent: :meth:`KernelSink.verify`
runs the spelled reference and the scheduled candidate in *this*
sink's runtime and returns the equivalence report — a fused kernel
ships only when it is numerically equal, never on the law's
say-so.  ``torch`` is imported lazily: the package's hard
dependency is ``mlopt-core`` alone, matching the sink packages'
adapter layering.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mlopt_core.ir import Const, Op, Param, Var
from mlopt_core.ops import OpTable
from mlopt_core.ports import ExecutorSpec, OpRegistry

from mlopt_kernel.laws import FUSED_KERNEL_OPS

if TYPE_CHECKING:
    from mlopt_core.ir import IR

__all__ = [
    "KernelExecutor",
    "KernelSink",
    "KernelVerifyReport",
    "kernel_ops",
]


def _torch() -> Any:
    """Return torch lazily — the sink's only third-party reach."""
    import torch

    return torch


def _bindings() -> dict[str, Any]:
    """Build the kernel op table — a small torch composite vocabulary.

    Covers the elementwise surface plus the reduction/view ops the
    spelled fixtures reach, and the ``kf_*`` fused-kernel ops the law
    bank mints — each bound to its *composite body* (what the fused
    kernel computes; the executor compiles it into one launch).
    Args arrive evaluated (tensor or scalar); attrs come through as
    canonical kwargs.
    """
    torch = _torch()

    return {
        "abs": torch.abs,
        "add": torch.add,
        "atan2": torch.atan2,
        "ceil": torch.ceil,
        "clamp": torch.clamp,
        "cos": torch.cos,
        "cosh": torch.cosh,
        "div": torch.div,
        "elu": torch.nn.functional.elu,
        "erf": torch.erf,
        "exp": torch.exp,
        "exp2": torch.exp2,
        "expm1": torch.expm1,
        "flatten": lambda x, start_dim=0, end_dim=-1: torch.flatten(
            x, start_dim, end_dim
        ),
        "floor": torch.floor,
        "fmax": torch.fmax,
        "fmin": torch.fmin,
        "fmod": torch.fmod,
        "gelu": torch.nn.functional.gelu,
        "hardsigmoid": torch.nn.functional.hardsigmoid,
        "hardswish": torch.nn.functional.hardswish,
        "leaky_relu": torch.nn.functional.leaky_relu,
        "log": torch.log,
        "log10": torch.log10,
        "log1p": torch.log1p,
        "log2": torch.log2,
        "matmul": torch.matmul,
        "maximum": torch.maximum,
        "mean": lambda x, dim=None, keepdim=False: torch.mean(
            x, dim=dim, keepdim=keepdim
        ),
        "minimum": torch.minimum,
        "mish": torch.nn.functional.mish,
        "mul": torch.mul,
        "neg": torch.neg,
        "permute": lambda x, dims: x.permute(*dims),
        "pow": torch.pow,
        "reciprocal": torch.reciprocal,
        "relu": torch.relu,
        "remainder": torch.remainder,
        "reshape": lambda x, shape: (
            x.reshape(*shape)
            if isinstance(shape, (tuple, list))
            else x.reshape(shape)
        ),
        "round": torch.round,
        "rsqrt": torch.rsqrt,
        "select": lambda x, dim, index: x.select(dim, index),
        "sigmoid": torch.sigmoid,
        "sign": torch.sign,
        "silu": torch.nn.functional.silu,
        "sin": torch.sin,
        "sinh": torch.sinh,
        "softmax": lambda x, dim=-1: torch.softmax(x, dim=dim),
        "softplus": torch.nn.functional.softplus,
        "sqrt": torch.sqrt,
        "square": torch.square,
        "squeeze": lambda x, dim=None: (
            x.squeeze() if dim is None else x.squeeze(dim)
        ),
        "sub": torch.sub,
        "sum": lambda x, dim=None, keepdim=False: torch.sum(
            x, dim=dim, keepdim=keepdim
        ),
        "tanh": torch.tanh,
        "transpose": lambda x, dim0=0, dim1=1: x.transpose(dim0, dim1),
        "unsqueeze": lambda x, dim: x.unsqueeze(dim),
        # --- the kernel vocabulary: composite bodies, one launch each
        "kf_silu": lambda a, b: a * torch.sigmoid(b),
        "kf_softsign": lambda a, b, c: a / (b.abs() + c),
        "kf_addmul": lambda a, b, c: a * b + c,
    }


#: The binding table, built once — the lambdas are stateless.
_BINDINGS: dict[str, Any] | None = None


def kernel_ops() -> frozenset[str]:
    """Return the op names the sink can lower — ``supported_ops``."""
    global _BINDINGS
    if _BINDINGS is None:
        _BINDINGS = _bindings()
    return frozenset(_BINDINGS)


def _table() -> dict[str, Any]:
    """Return the shared binding dict — one build, every consumer."""
    kernel_ops()
    assert _BINDINGS is not None
    return _BINDINGS


@dataclass(frozen=True)
class KernelVerifyReport:
    """Hold the sink's equivalence report — a ``VerifyResult``."""

    max_abs: float
    max_rel: float
    passed: bool


class KernelExecutor:
    """The lowered artifact — a callable executor over an IR term.

    Evaluates the term against the sink's binding table; each
    ``kf_*`` fused op dispatches through a ``torch.compile``-compiled
    callable (cached per op) — Inductor fuses the composite body into
    a single kernel launch where the spelled ops launched one at a
    time.  With ``compile_fused=False`` the fused ops run their
    spelled bodies eagerly — the fallback verify-first compares
    against.
    """

    def __init__(
        self,
        ir: IR,
        params: Mapping[str, Any] | None = None,
        *,
        compile_fused: bool = True,
        bindings: Mapping[str, Any] | None = None,
    ) -> None:
        """Bind the program, its leaf values, and the codegen flag."""
        self._ir = ir
        self._params = dict(params or {})
        self._bindings = (
            dict(bindings) if bindings is not None else dict(_table())
        )
        self.compile_fused = compile_fused
        self._compiled: dict[str, Any] = {}

    def forward(self, *xs: Any) -> Any:
        """Run the program on positional inputs in ``ir.inputs`` order."""
        env = dict(self._params)
        env.update(
            {
                v.name: x
                for v, x in zip(self._ir.inputs, xs, strict=True)
            }
        )
        return self._eval(self._ir.root, env)

    def __call__(self, *xs: Any) -> Any:
        """Alias of :meth:`forward` — the callable executor surface."""
        return self.forward(*xs)

    def _leaf(self, t: Any, env: dict) -> Any:
        """Resolve a leaf: Var/Param by name, Const by value."""
        if isinstance(t, Const):
            return t.value
        if isinstance(t, (Var, Param)):
            return env[t.name]
        raise TypeError(f"unsupported leaf term {t!r}")

    def _fused(self, op: str) -> Any:
        """Return the compiled callable for one ``kf_*`` op (cached)."""
        fn = self._compiled.get(op)
        if fn is None:
            raw = self._bindings[op]
            fn = (
                _torch().compile(raw, fullgraph=False)
                if self.compile_fused
                else raw
            )
            self._compiled[op] = fn
        return fn

    def _eval(self, t: Any, env: dict) -> Any:
        """Evaluate *t* — ``kf_*`` ops through the compiled path."""
        if not isinstance(t, Op):
            return self._leaf(t, env)
        fn = (
            self._fused(t.op)
            if t.op in FUSED_KERNEL_OPS
            else self._bindings.get(t.op)
        )
        if fn is None:
            raise KeyError(
                f"KernelSink cannot lower op {t.op!r} — not in "
                "supported_ops"
            )
        return fn(
            *[self._eval(a, env) for a in t.args], **dict(t.attrs)
        )


class KernelSink:
    """``mlopt_core.ports.Sink`` over the kernel composite backend.

    ``supported_ops`` bounds the search the same way ``OnnxSink``'s
    does — any member using an op outside the table prices at ``+inf``
    under ``backend_cost``, so the search never commits to a form this
    sink cannot execute.
    """

    def __init__(self, *, compile_fused: bool = True) -> None:
        """Configure the sink; ``compile_fused`` toggles the codegen."""
        self.compile_fused = compile_fused
        self._ops: OpTable | None = None

    @property
    def supported_ops(self) -> frozenset[str]:
        """Every op name this sink lowers — the extraction bound."""
        return kernel_ops()

    @property
    def ops(self) -> OpRegistry:
        """The lowering registry — an ``OpTable`` of torch bindings."""
        if self._ops is None:
            table = OpTable.core()
            table.torch_bindings = dict(_table())
            self._ops = table
        return self._ops

    @property
    def executors(self) -> Mapping[str, ExecutorSpec]:
        """Empty — one lowering, no carrier routing (OnnxSink's shape)."""
        return {}

    def lower(
        self, ir: IR, params: dict[str, Any] | None = None
    ) -> KernelExecutor:
        """Materialise ``ir`` as a runnable kernel executor."""
        return KernelExecutor(
            ir, params, compile_fused=self.compile_fused
        )

    def verify(
        self,
        ref: Any,
        opt: Any,
        inputs: Any,
        *,
        rtol: float = 1e-4,
        atol: float | None = None,
    ) -> KernelVerifyReport:
        """Run ``ref`` and ``opt`` on ``inputs``; compare in this runtime.

        ``ref`` may be any callable (the spelled torch module, or a
        second executor); ``opt`` is the lowered candidate.  The
        metric is the repo's rel-diff gate: ``max|ref−out| /
        (max|ref| + 1e-8) ≤ rtol`` — or ``max_abs ≤ atol`` when the
        caller supplies an absolute floor.
        """
        torch = _torch()
        del torch  # comparison is tensor arithmetic, not torch calls
        args = (
            inputs if isinstance(inputs, (tuple, list)) else (inputs,)
        )
        ref_out = ref(*args)
        out = opt(*args)
        diff = (ref_out - out).abs()
        max_abs = float(diff.max()) if diff.numel() else 0.0
        denom = (
            float(ref_out.abs().max()) + 1e-8
            if ref_out.numel()
            else 1e-8
        )
        max_rel = max_abs / denom
        passed = max_rel <= rtol or (
            atol is not None and max_abs <= atol
        )
        return KernelVerifyReport(
            max_abs=max_abs, max_rel=max_rel, passed=bool(passed)
        )

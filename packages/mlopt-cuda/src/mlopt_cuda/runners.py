"""CUDA-graph delivery runner.

Moved verbatim from ``mlopt_orchestrator.runners`` (plan 0007): the
capture mechanism is torch-coupled *and* CUDA-device-coupled, so it
sits in its own package — the orchestrator imports no torch and the
torch package imports no CUDA-specific machinery.

:class:`CudaGraphRunner` conforms to
:class:`mlopt_core.ports.Runner`: ``apply(module, example_input,
stats) -> module``.  Two capture paths, in precedence order:

* modules exposing ``capture_cuda_graph`` (the batched carrier
  executors) capture *themselves* — the historical path;
* any other ``torch.nn.Module`` whose example inputs are all CUDA
  tensors is captured *generically*: the forward's dispatch sequence
  is recorded once into a :class:`torch.cuda.CUDAGraph` over static
  input buffers, and the returned :class:`_GraphReplayModule` replays
  it per call.  This is the generic ``IRModule`` plan-tape case —
  ``docs/results_wholemodel_gap.md`` measured the tape capturable
  and dispatch-dominated at fp32 (the carrier recovered
  ~2.9 → ~1.1 ms under a manual capture).  Non-CUDA inputs, non-
  tensor inputs, and non-``nn.Module`` deliverables refuse quietly
  (``stats["cuda_graph"]`` stays False).

A capture that raises still degrades gracefully — the module drops its
partial graph (executor path) or is returned untouched (generic path)
and delivery continues — but the reason is no longer swallowed
(plan 0016 stage 3): the exception is classified through
:func:`mlopt_core.failures.classify` and recorded under
``stats["cuda_graph_failure"]``, so an OOM capture is distinguishable
from a kernel fault or a missing device.
"""

from __future__ import annotations

from typing import Any, Self, cast

import torch
from mlopt_core.failures import FailureClass, classify

__all__ = ["CudaGraphRunner"]


def _capture_failure_class(exc: BaseException) -> FailureClass:
    """Classify a failed CUDA-graph capture into a ``FailureClass``."""
    return classify(exc)


class _GraphReplayModule(torch.nn.Module):
    """A generic module whose forward was captured into a CUDA graph.

    Built by :meth:`CudaGraphRunner.apply`'s generic path: the wrapped
    module's op sequence is recorded once over static input buffers,
    and each matching call copies its arguments in and replays — one
    launch total, same shape-baked contract the batched executors
    expose: calls whose arity/shape/dtype/device differ from the
    capture-time signature fall back to the wrapped module's eager
    forward, and the returned value is the *static* output, overwritten
    by the next replay (clone it to keep it).

    ``inner`` is a real submodule, so ``parameters()`` /
    ``state_dict()`` / ``.to()`` still reach the wrapped module, and
    attribute lookups the wrapper doesn't own (``_root``,
    ``eval_mod``, ``is_batched``, …) delegate to it.
    """

    def __init__(
        self,
        inner: torch.nn.Module,
        graph: torch.cuda.CUDAGraph,
        static_ins: list[torch.Tensor],
        static_out: Any,
    ) -> None:
        super().__init__()
        self.inner = inner
        self._graph = graph
        self._graph_ins = static_ins
        self._graph_out = static_out

    @classmethod
    def capture(  # pragma: no cover — CUDA-only body
        cls,
        module: torch.nn.Module,
        xs: tuple[torch.Tensor, ...],
        warmup: int = 3,
    ) -> Self:
        """Record ``module(*xs)`` into a CUDA graph; return the wrapper.

        Same recipe as the batched executors'
        ``capture_cuda_graph``: clone inputs into static buffers, warm
        up on a side stream (lazy inits, cudnn/cublas handle setup —
        nothing host-side may remain at capture), then record one
        forward under ``no_grad``.
        """
        static_ins = [t.detach().clone() for t in xs]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(warmup):
                module(*static_ins)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            static_out = module(*static_ins)
        return cls(module, g, static_ins, static_out)

    def __getattr__(self, name: str) -> Any:
        """Delegate introspection attrs (``_root`` &c.) to ``inner``."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self.__dict__.get("_modules", {}).get("inner")
            if inner is None:
                raise
            return getattr(inner, name)

    def _matches(self, xs: tuple[Any, ...]) -> bool:
        """Return True when ``xs`` fits the capture-time signature."""
        return len(xs) == len(self._graph_ins) and all(
            isinstance(t, torch.Tensor)
            and t.shape == b.shape
            and t.dtype == b.dtype
            and t.device == b.device
            for t, b in zip(xs, self._graph_ins, strict=True)
        )

    def forward(self, *xs: Any) -> Any:
        """Replay the captured graph, or fall back to eager on mismatch."""
        if self._matches(xs):
            for buf, t in zip(xs, self._graph_ins, strict=True):
                buf.copy_(t, non_blocking=True)
            self._graph.replay()
            return self._graph_out
        return self.inner(*xs)


def _capture_executor(
    module: torch.nn.Module,
    xs: tuple[Any, ...],
    stats: dict[str, Any],
) -> torch.nn.Module | None:
    """Capture through the executor path — *module* handled, or ``None``.

    ``None`` means fall through to the generic path: the module has no
    ``capture_cuda_graph`` of its own, or it has one but declined —
    a batched executor whose plan didn't match (``_plan is None``,
    ``is_batched`` False) no-ops in capture, so ``is_graph_captured``
    stays False and claiming ``cuda_graph=True`` would be a lie.
    """
    capture = getattr(module, "capture_cuda_graph", None)
    if capture is None:
        return None
    try:  # pragma: no cover — CUDA-only body
        capture(*xs)
    except Exception as exc:  # pragma: no cover — CUDA-only
        stats["cuda_graph_failure"] = _capture_failure_class(exc)
        cast(Any, module).drop_cuda_graph()
        return module
    if getattr(module, "is_graph_captured", True):
        stats["cuda_graph"] = True
        return module
    return None


def _capture_generic(
    module: torch.nn.Module,
    xs: tuple[Any, ...],
    stats: dict[str, Any],
) -> torch.nn.Module:
    """Capture a generic module's forward into a CUDA-graph replay.

    The ``IRModule`` plan-tape case — modules without their own
    ``capture_cuda_graph``.  Refuses (returns *module* unchanged,
    ``stats["cuda_graph"]`` left False) unless the deliverable is an
    ``nn.Module`` and *every* positional input is a CUDA tensor —
    non-tensor arguments can't be refreshed per call, so baking them
    would be silently wrong.  A capture that raises classifies the
    reason into ``stats["cuda_graph_failure"]`` and returns *module*
    untouched — the generic path mutates nothing.
    """
    if not isinstance(module, torch.nn.Module) or not all(
        isinstance(t, torch.Tensor) and t.is_cuda for t in xs
    ):
        return module
    try:  # pragma: no cover — CUDA-only body
        wrapped = _GraphReplayModule.capture(module, xs)
    except Exception as exc:  # pragma: no cover — CUDA-only
        stats["cuda_graph_failure"] = _capture_failure_class(exc)
        return module
    stats["cuda_graph"] = True
    return wrapped


class CudaGraphRunner:
    """Capture the delivered module into a CUDA graph.

    Collapses the carrier's per-level launches — or a generic
    ``IRModule``'s per-leaf tape dispatch — into one replayable
    graph (measured ~2.4x on the batched scan; ~2.6x on the fp32
    generic carrier per ``docs/results_wholemodel_gap.md``).
    Batched executors capture themselves via ``capture_cuda_graph``;
    any other ``nn.Module`` is wrapped in a :class:`_GraphReplayModule`
    recording its forward.  Shape is baked at capture — mismatched
    calls fall back to eager internally.  Non-CUDA, non-tensor, or
    non-module deliverables degrade quietly.

    Compiled delivery wins: when ``stats["compiled"]`` is set the
    module is already a dynamo wrapper (which also fuses launches),
    so capture is skipped entirely — ``stats["cuda_graph"]`` stays
    unset, matching the historical flag behaviour.
    """

    name = "cuda_graph"

    def apply(
        self,
        module: torch.nn.Module,
        example_input: Any,
        stats: dict[str, Any],
    ) -> torch.nn.Module:
        """Capture *module* into a CUDA graph when the input is CUDA."""
        if stats.get("compiled"):
            return module
        stats["cuda_graph"] = False
        xs = (
            tuple(example_input)
            if isinstance(example_input, (tuple, list))
            else (example_input,)
        )
        if xs and isinstance(xs[0], torch.Tensor) and xs[0].is_cuda:
            out = _capture_executor(module, xs, stats)
            if out is not None:
                return out
            return _capture_generic(module, xs, stats)
        return module

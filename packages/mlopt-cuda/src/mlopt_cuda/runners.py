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

Phase B (``docs/reports/production_readiness.md``) adds two honest
slices here:

* ``CudaGraphRunner(multi_device=True)`` wraps generic deliverables
  in :class:`_DeviceGraphModule` — a per-device capture cache keyed
  by ``(device_index, shape/dtype signature)``.  One graph per device,
  never shared; a signature or parameter-identity mismatch falls back
  to the wrapped module's eager forward.  Capture is lazy (first
  matching call), so ``stats["cuda_graph"]`` means *armed*, with
  ``stats["cuda_graph_per_device"]`` marking the mode.  Batched
  executors keep their own single-device ``capture_cuda_graph`` —
  the flag does not reach them.
* ``apply`` runs :class:`mlopt_torch.device_placement.ShardedAwareDevicePolicy`'s
  :meth:`~mlopt_torch.device.DevicePolicy.check_module` first: DDP /
  FSDP / TP-wrapped modules, DTensor/``FlatParameter`` shard leaves,
  and modules binding collective ops raise
  :class:`~mlopt_torch.device.UnsupportedDistributedError` (shard
  leaves raise the ``ShardedParamError`` subclass) — an explicit
  refusal, never a silently wrong single-rank replay.
"""

from __future__ import annotations

from typing import Any, Self, cast

import torch
from mlopt_core.failures import FailureClass, classify
from mlopt_torch.device import (
    device_graph_key,
    storage_fingerprint,
)
from mlopt_torch.device_placement import (
    ShardedAwareDevicePolicy as DevicePolicy,
)

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
            for buf, t in zip(self._graph_ins, xs, strict=True):
                buf.copy_(t, non_blocking=True)
            self._graph.replay()
            return self._graph_out
        return self.inner(*xs)


class _DeviceGraphModule(torch.nn.Module):
    """Per-device CUDA-graph cache over one wrapped module (Phase B).

    ``_GraphReplayModule`` bakes ONE device: its static buffers and the
    captured kernels live on the capture device, and its ``_matches``
    already refuses foreign-device calls — but it has no way to serve
    a *second* device.  This wrapper keys captures by
    ``(device_index, shape/dtype signature)`` from
    :func:`mlopt_torch.device.device_graph_key`: each device+signature
    pair gets its own :class:`_GraphReplayModule`, captured lazily on
    first use inside ``torch.cuda.device(idx)``.  A graph is NEVER
    shared across devices.

    Safety: a captured graph holds raw pointers to the storages live
    at capture time — it does not keep them alive, and a module that
    moves or rebinds its parameters (``Module.to`` swaps ``.data``;
    ``IRModule._align_to`` materialises fresh tensors on a device
    switch) would leave the graph replaying freed storage.
    :func:`mlopt_torch.device.storage_fingerprint` snapshots the
    untyped storages — holding them keeps the captured memory alive
    AND detects every real move, since neither ``.to()`` nor
    ``_align_to`` preserves storage identity.  Before every replay the
    fingerprint is recomputed; any mismatch drops the entry and falls
    back to eager (which realigns the parameters — the next call
    recaptures against the new storage).

    Honest limits: capture is per (device, signature) and lazy, so
    ``stats`` cannot know at ``apply`` time whether it will succeed —
    a capture failure is recorded per key in ``capture_failures`` and
    that key serves eager forever after.  Modules whose parameters
    migrate between devices pay eager realignment plus a fresh
    capture per switch; per-rank replay and sharded parameters are
    out of scope (see ``docs/reports/device_policy.md``).
    """

    def __init__(self, inner: torch.nn.Module, warmup: int = 3) -> None:
        super().__init__()
        self.inner = inner
        self._warmup = warmup
        self._graphs: dict[tuple, _GraphReplayModule] = {}
        self._snapshots: dict[tuple, tuple] = {}
        self._eager_keys: set[tuple] = set()
        self.capture_failures: dict[tuple, FailureClass] = {}

    def _fingerprint(self) -> tuple:
        """Storage-identity snapshot of the inner module's tensors."""
        return storage_fingerprint(self.inner)

    def _params_match(self, snap: tuple) -> bool:
        """Return True when the storages ARE the captured ones."""
        cur = self._fingerprint()
        return len(cur) == len(snap) and all(
            a is b for a, b in zip(cur, snap, strict=True)
        )

    def _capture(
        self, key: tuple, xs: tuple
    ) -> _GraphReplayModule | None:
        """Record ``inner(*xs)`` on ``key``'s device; ``None`` on failure."""
        idx = key[0]
        try:
            with torch.cuda.device(idx):  # pragma: no cover — CUDA-only
                cap = _GraphReplayModule.capture(
                    self.inner, xs, warmup=self._warmup
                )
        except Exception as exc:  # pragma: no cover — CUDA-only
            self.capture_failures[key] = _capture_failure_class(exc)
            self._eager_keys.add(key)
            return None
        self._graphs[key] = cap
        self._snapshots[key] = self._fingerprint()
        return cap

    def forward(self, *xs: Any) -> Any:
        """Replay the per-device graph, or run eager honestly."""
        key = device_graph_key(xs)
        if key is None:
            # Non-CUDA / mixed-device / non-tensor inputs: the same
            # refusal the single-capture path applies — eager only.
            return self.inner(*xs)
        entry = self._graphs.get(key)
        if entry is not None and not self._params_match(
            self._snapshots[key]
        ):
            # Parameters were rebound since capture (a device/dtype
            # realignment) — the graph's pointers are stale.  Drop it
            # and recapture below against the current storage.
            del self._graphs[key]
            del self._snapshots[key]
            entry = None
        if key in self._eager_keys:
            return self.inner(*xs)
        if entry is None:
            entry = self._capture(key, xs)
            if entry is None:
                return self.inner(*xs)
        return entry(*xs)


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

    ``multi_device=True`` (Phase B) routes the generic path through
    :class:`_DeviceGraphModule` — a lazy per-(device, signature)
    capture cache — instead of the single eager capture.  Executor
    modules keep their own ``capture_cuda_graph`` either way.
    Distributed wrappers are refused in both modes.
    """

    name = "cuda_graph"

    def __init__(self, *, multi_device: bool = False) -> None:
        """``multi_device``: per-device capture cache (Phase B)."""
        self.multi_device = multi_device
        self._policy = DevicePolicy()

    def apply(
        self,
        module: torch.nn.Module,
        example_input: Any,
        stats: dict[str, Any],
    ) -> torch.nn.Module:
        """Capture *module* into a CUDA graph when the input is CUDA."""
        # Explicit refusal (Phase B): DDP/FSDP/TP wrappers and modules
        # binding collective ops raise rather than deliver a
        # silently wrong single-rank replay — checked before every
        # early-out so no delivery path slips past it.
        self._policy.check_module(module)
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
            if self.multi_device:
                # Lazy per-device cache: capture happens on first
                # matching call, so ``cuda_graph`` here means *armed*;
                # ``cuda_graph_per_device`` marks the mode.
                stats["cuda_graph"] = True
                stats["cuda_graph_per_device"] = True
                return _DeviceGraphModule(module)
            return _capture_generic(module, xs, stats)
        return module

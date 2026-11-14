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

Alias-aware capture (``docs/reports/alias_aware_capture.md``):
``CudaGraphRunner(alias_inputs=...)`` opts the generic path into
capturing *selected* positional inputs by reference — the graph
reads the caller's live storage at replay instead of copying into a
static buffer first.  This is the decode-loop case measured in
``docs/reports/decode_step_fusion.md`` (-0.9 ms/tok: 24 persistent
K/V-cache inputs skip ~36.8 MB of D2D copies per step).  It is
opt-in because object-reuse across calls is a caller property the
runner cannot observe: the spec names WHICH positions may alias
(``True`` = all, an iterable of positions, or a
``(position, tensor) -> bool`` predicate), capture binds exactly
those storages, and every replay re-verifies the binding — a
mismatched tensor falls back to the wrapped module's eager forward
(never a stale read, never a write into a captured tensor).  Inputs
the recorded graph mutates in place are detected via the tensor
version counter during capture and demoted to the copy path
automatically.  ``multi_device=True`` composes: the lazy per-device
capture binds whichever tensors the FIRST matching call passes —
for a decode loop that is exactly the persistent buffers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
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


#: Per-position alias predicate: ``(position, capture-time tensor) ->
#: bool``.  ``True`` means the graph binds that input's live storage
#: and replay requires the caller to pass the same binding.
AliasPredicate = Callable[[int, torch.Tensor], bool]


def _normalize_alias_inputs(
    spec: bool | Iterable[int] | AliasPredicate | None,
) -> AliasPredicate | None:
    """Normalize the ``alias_inputs`` runner spec to a predicate.

    ``None``/``False`` → no aliasing (the historical copy-everything
    path); ``True`` → every position eligible; an iterable of ints →
    those positions; a callable → used verbatim.
    """
    if spec is None or spec is False:
        return None
    if spec is True:
        return lambda _i, _t: True
    if callable(spec):
        return cast(AliasPredicate, spec)
    positions = frozenset(cast(Iterable[int], spec))
    if not all(isinstance(p, int) for p in positions):
        raise TypeError(
            "alias_inputs positions must be ints, "
            f"got {sorted(positions, key=repr)!r}"
        )
    return lambda i, _t: i in positions


def _alias_bound(t: torch.Tensor, cap: torch.Tensor) -> bool:
    """Return True when *t* binds the same memory *cap* was captured on.

    The graph recorded raw pointers into *cap*'s storage, so replay is
    only honest when the caller's tensor addresses the SAME bytes with
    the SAME layout.  ``t is cap`` is the fast path (the common
    decode-loop case — the caller reuses the buffer objects); the
    full check accepts re-wrapped views of the same region (``detach``,
    ``view``, ``as_strided`` aliases) while rejecting clones, fresh
    allocations, and same-shape impostors.  Deliberately NOT part of
    the check: ``_version`` — in-place mutation of the bound storage
    by the *caller* between calls is exactly what alias capture is
    for (the decode loop writes new K/V into its buffers every step);
    only mutation *recorded into the graph* is guarded (capture-side
    version-counter demotion in ``_GraphReplayModule.capture``).
    """
    if t is cap:
        return True
    return bool(
        t.layout == cap.layout
        and t.shape == cap.shape
        and t.dtype == cap.dtype
        and t.device == cap.device
        and t.stride() == cap.stride()
        and t.storage_offset() == cap.storage_offset()
        and t.untyped_storage().data_ptr()
        == cap.untyped_storage().data_ptr()
    )


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

    Alias-aware mode (``alias_inputs`` on the runner): ``_graph_ins``
    is a full-length parallel list whose entries are EITHER the
    static input buffer (copy positions — refreshed per call) OR the
    capture-time caller tensor itself (alias positions — the graph
    reads its live storage).  ``_alias`` marks which is which.
    Consumers iterating ``_graph_ins`` must NOT ``copy_`` into alias
    entries: that would write into the caller's tensor.
    """

    def __init__(
        self,
        inner: torch.nn.Module,
        graph: torch.cuda.CUDAGraph,
        static_ins: list[torch.Tensor],
        static_out: Any,
        alias: tuple[bool, ...] | None = None,
    ) -> None:
        super().__init__()
        self.inner = inner
        self._graph = graph
        self._graph_ins = static_ins
        self._graph_out = static_out
        self._alias = (
            alias if alias is not None else (False,) * len(static_ins)
        )

    @classmethod
    def capture(  # pragma: no cover — CUDA-only body
        cls,
        module: torch.nn.Module,
        xs: tuple[torch.Tensor, ...],
        warmup: int = 3,
        alias: AliasPredicate | None = None,
    ) -> Self:
        """Record ``module(*xs)`` into a CUDA graph; return the wrapper.

        Same recipe as the batched executors'
        ``capture_cuda_graph``: clone inputs into static buffers, warm
        up on a side stream (lazy inits, cudnn/cublas handle setup —
        nothing host-side may remain at capture), then record one
        forward under ``no_grad``.

        ``alias`` (opt-in): a per-position predicate naming inputs the
        graph may bind BY REFERENCE — replay then reads the caller's
        live storage with no copy (the decode-loop K/V-cache case).
        Three guards keep it honest:

        * warmup still runs on all-clone inputs, so the extra
          capture-time forwards never touch caller tensors;
        * the record call runs on the REAL aliased tensors, and any
          position whose tensor ``_version`` bumps during recording —
          i.e. the graph writes that input in place — is demoted to
          the copy path and the graph re-recorded (bounded: each
          pass demotes at least one position);
        * replay re-verifies the storage binding per aliased
          position (:func:`_alias_bound`) and falls back to the
          wrapped module's eager forward on any mismatch.
        """
        n = len(xs)
        flags = (
            [bool(alias(i, t)) for i, t in enumerate(xs)]
            if alias is not None
            else [False] * n
        )
        warm_ins = [t.detach().clone() for t in xs]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(warmup):
                module(*warm_ins)
        torch.cuda.current_stream().wait_stream(s)
        # Static buffers exist only for copy positions; alias
        # positions bind the caller's tensor itself.
        bufs: list[torch.Tensor | None] = [
            None if flags[i] else xs[i].detach().clone()
            for i in range(n)
        ]
        while True:
            call_ins = tuple(
                xs[i] if flags[i] else cast(torch.Tensor, bufs[i])
                for i in range(n)
            )
            vers = {i: xs[i]._version for i in range(n) if flags[i]}
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g):
                static_out = module(*call_ins)
            # A version bump means the RECORDED graph mutates that
            # input's storage in place — replay would write into the
            # caller's live buffer.  Demote to the copy path and
            # re-record; each pass shrinks the alias set, so this
            # terminates after at most ``n`` re-records.
            bad = [i for i, v in vers.items() if xs[i]._version != v]
            if not bad:
                break
            for i in bad:
                flags[i] = False
                bufs[i] = xs[i].detach().clone()
        graph_ins = [xs[i] if flags[i] else bufs[i] for i in range(n)]
        # ty: the None holes are exactly the alias positions, filled
        # by the caller tensors in ``graph_ins``.
        return cls(
            module,
            g,
            cast(list[torch.Tensor], graph_ins),
            static_out,
            tuple(flags),
        )

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
        """Return True when ``xs`` fits the capture-time signature.

        Every position must match shape/dtype/device (as before);
        alias positions additionally must bind the SAME storage the
        graph recorded — :func:`_alias_bound`.  A mismatch on an
        aliased position cannot be served by copying (there is no
        static buffer for it), so the whole call falls back to eager.
        """
        if len(xs) != len(self._graph_ins):
            return False
        for i, (t, b) in enumerate(
            zip(xs, self._graph_ins, strict=True)
        ):
            if not (
                isinstance(t, torch.Tensor)
                and t.shape == b.shape
                and t.dtype == b.dtype
                and t.device == b.device
            ):
                return False
            if self._alias[i] and not _alias_bound(t, b):
                return False
        return True

    def forward(self, *xs: Any) -> Any:
        """Replay the captured graph, or fall back to eager on mismatch."""
        if not self._matches(xs):
            return self.inner(*xs)
        for i, (buf, t) in enumerate(
            zip(self._graph_ins, xs, strict=True)
        ):
            if not self._alias[i]:
                # Copy positions only — an alias entry IS the
                # caller's tensor; writing into it would corrupt the
                # caller's buffer (and the graph reads it live).
                buf.copy_(t, non_blocking=True)
        self._graph.replay()
        return self._graph_out


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

    ``alias`` forwards the runner's ``alias_inputs`` predicate to each
    lazy capture: the FIRST call matching a key binds the flagged
    positions' storages, and later calls must pass the same binding
    (``_alias_bound``) or that entry serves eager — see
    ``docs/reports/alias_aware_capture.md``.
    """

    def __init__(
        self,
        inner: torch.nn.Module,
        warmup: int = 3,
        alias: AliasPredicate | None = None,
    ) -> None:
        super().__init__()
        self.inner = inner
        self._warmup = warmup
        self._alias = alias
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
                    self.inner,
                    xs,
                    warmup=self._warmup,
                    alias=self._alias,
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
    alias: AliasPredicate | None = None,
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
        wrapped = _GraphReplayModule.capture(module, xs, alias=alias)
    except Exception as exc:  # pragma: no cover — CUDA-only
        stats["cuda_graph_failure"] = _capture_failure_class(exc)
        return module
    stats["cuda_graph"] = True
    if any(wrapped._alias):
        # Alias-aware capture armed: replay binds the capture-time
        # storages for the flagged positions (see the module docstring
        # for the identity-guard contract).
        stats["cuda_graph_alias"] = True
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

    ``alias_inputs`` (opt-in, ``docs/reports/alias_aware_capture.md``)
    selects which positional inputs the generic capture may bind BY
    REFERENCE instead of copying per replay: ``True`` aliases every
    position, an iterable of ints aliases those positions, and a
    ``(position, tensor) -> bool`` callable decides per input.
    Aliased positions replay only when the caller passes a tensor
    binding the same storage (identity fast-path, else
    storage/offset/shape/stride/dtype/device equality); any mismatch
    falls back to the wrapped module's eager forward — correct but
    uncaptured, so alias only positions the caller actually reuses
    (persistent KV-cache-style buffers).  Inputs the recorded graph
    mutates in place are auto-demoted to the copy path.  ``None``
    (default) is the historical copy-everything behaviour.  The flag
    reaches only the generic path — batched executors keep their own
    ``capture_cuda_graph``.
    """

    name = "cuda_graph"

    def __init__(
        self,
        *,
        multi_device: bool = False,
        alias_inputs: bool
        | Iterable[int]
        | AliasPredicate
        | None = None,
    ) -> None:
        """``multi_device``: per-device capture cache (Phase B).

        ``alias_inputs``: opt-in alias-aware capture spec — see the
        class docstring and ``docs/reports/alias_aware_capture.md``.
        """
        self.multi_device = multi_device
        self._alias_pred = _normalize_alias_inputs(alias_inputs)
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
                # ``cuda_graph_per_device`` marks the mode.  With
                # ``alias_inputs`` the armed entries bind the FIRST
                # call's tensors at the flagged positions — the
                # decode-loop case, where that first call already
                # carries the persistent buffers.
                stats["cuda_graph"] = True
                stats["cuda_graph_per_device"] = True
                if self._alias_pred is not None:
                    stats["cuda_graph_alias"] = True
                return _DeviceGraphModule(
                    module, alias=self._alias_pred
                )
            return _capture_generic(
                module, xs, stats, alias=self._alias_pred
            )
        return module

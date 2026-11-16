"""Delivery — pick the fastest honest runner and measure it.

``mlopt optimize --deliver`` extends the verified pipeline with the
fast-artifact half: after ``search`` + ``lower(verify=True)`` ships
the *eager* lowering, this module selects a delivery runner from the
sibling domain packages and re-lowers under it — so the stock
``lower(verify=True)`` gate still compares the DELIVERED module
against a fresh lowering of the certified IR.

Runner selection (``select_runner``):

* **region_fuse** — :class:`kit_torch.runners.RegionFusionRunner`
  when the lowered module is an ``IRModule`` AND the extracted term
  has census-eligible elementwise regions
  (``kit_torch.region_fuse.collect_regions``).  Non-``IRModule``
  deliveries (carrier executors) have no term surface to fuse —
  skipped here; ``fuse_module`` would decline them anyway.
* **cuda_graph** — :class:`kit_cuda.CudaGraphRunner` when any
  positional input (or, for parameter-only modules, any parameter)
  is a CUDA tensor.  ``alias_inputs`` / ``mutable_inputs`` are taken
  verbatim from ``spec.meta`` — the adapter declares them (the
  decode-loop KV-cache case of
  ``graphopt:bench/drivers/probes/whisper_kv_static_probe.py``:
  persistent buffers alias by reference, in-graph ``index_copy_``
  write-back positions need ``mutable_inputs`` or the version-counter
  demotion silently moves them to the copy path).  No meta flag →
  copy-path capture, which is always correct but copies inputs per
  replay.

Both compose through ``ChainedRunner`` (fusion first — the graph
records the fused dispatch; compiled region bodies are host-side at
capture and irrelevant at replay).

Honesty contract:

* CPU runs get no CUDA graph — the note says so; nothing fakes a
  speedup.
* A delivered module that fails ``verify`` does NOT ship: the
  verified eager lowering stands and the report says why.
* ``--no-verify`` ships the runner unverified — noted, never silent.
* Timing is min/median over ``bench_iters`` forwards after warmup,
  with ``torch.cuda.synchronize`` around each call on CUDA — a
  shared-GPU caveat is recorded in the notes.  The delivery game
  times through :func:`_bench_paired` instead: interleaved
  eager/config rounds so drift on a contended GPU can't
  systematically favor whichever side benches second.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Delivery",
    "deliver",
    "select_runner",
]


@dataclass
class Delivery:
    """The outcome of one ``--deliver`` run.

    ``lowered`` is the artifact to ship (the runner-delivered
    ``LowerResult`` when it verified, else the eager one).
    ``runner_name`` is ``None`` when no fast runner shipped — either
    none applied OR the selected runner was declined (verify gate,
    lower failure); the notes still name the declined runner.
    Timing fields are ``None`` when no comparison was measured;
    ``eager_*`` is always populated on a successful delivery attempt
    (it is the shipped artifact when no runner applies).
    """

    lowered: Any
    runner_name: str | None = None
    eager_min_ms: float | None = None
    eager_med_ms: float | None = None
    delivered_min_ms: float | None = None
    delivered_med_ms: float | None = None
    notes: list[str] = field(default_factory=list)
    #: Which mechanism picked the runner — ``"heuristic"`` for
    #: :func:`select_runner`, ``"game"`` for the measured delivery
    #: game (:mod:`mlopt.deliver_game`).  The report names it; a
    #: silent swap would violate the honesty contract.
    mechanism: str = "heuristic"


# ---------------------------------------------------------------------------
#  Capability probes — all lazy/duck-typed so import stays torch-free
# ---------------------------------------------------------------------------


def _torch() -> Any:
    try:
        import torch

        return torch
    except Exception:  # noqa: BLE001 — torch absent is a valid decline
        return None


def _inputs_cuda(inputs: Any) -> bool:
    """True when any positional input is a CUDA tensor."""
    torch = _torch()
    if torch is None or not isinstance(inputs, (tuple, list)):
        return False
    return any(isinstance(t, torch.Tensor) and bool(t.is_cuda) for t in inputs)


def _module_cuda(module: Any) -> bool:
    """True when an ``nn.Module`` binds CUDA parameters."""
    torch = _torch()
    if torch is None or not isinstance(module, torch.nn.Module):
        return False
    try:
        return any(bool(p.is_cuda) for p in module.parameters())
    except Exception:  # noqa: BLE001 — exotic module: don't guess
        return False


def _has_cuda(module: Any, inputs: Any) -> bool:
    return _inputs_cuda(inputs) or _module_cuda(module)


def _alias_spec(meta: dict[str, Any], key: str) -> Any:
    """Read an adapter-declared alias/mutable spec, validated.

    The value is passed verbatim to ``CudaGraphRunner`` (``True`` /
    iterable-of-positions / ``(pos, tensor) -> bool`` predicate).  A
    malformed spec degrades to ``None`` — copy-path capture — with
    the reason returned for the notes; never a crash, never a guess.
    """
    spec = meta.get(key)
    if spec is None or spec is True or callable(spec):
        return spec, None
    if spec is False:
        return None, None
    try:
        positions = frozenset(spec)
    except TypeError:
        return None, f"meta[{key!r}]={spec!r} is not a valid alias spec — ignored"
    if not all(isinstance(p, int) for p in positions):
        return None, f"meta[{key!r}] positions must be ints — ignored"
    return positions, None


# ---------------------------------------------------------------------------
#  Runner selection
# ---------------------------------------------------------------------------


def _fusible(term: Any, module: Any) -> tuple[bool, str | None]:
    """``(eligible, detail)`` — elementwise regions worth fusing.

    Needs BOTH the extracted term (the census input) and an
    ``IRModule`` delivery (``fuse_module``'s only accepted shape —
    the rewrite runs on ``module._root``).  The census itself is
    honest about skips; ``detail`` summarizes for the notes.
    """
    if term is None:
        return False, "no extracted term on the search result"
    try:
        from kit_torch.region_fuse import collect_regions
        from kit_torch.torch_eval import IRModule
    except Exception as e:  # noqa: BLE001 — kit_torch absent/broken
        return False, f"kit_torch region_fuse unavailable: {e}"
    if not isinstance(module, IRModule):
        return False, (
            f"delivered module is {type(module).__name__}, not IRModule "
            "— carrier executors expose no term to fuse"
        )
    plan = collect_regions(term)
    fused = int(plan.stats.get("fused", 0))
    ops = int(plan.stats.get("ops_fused", 0))
    if fused <= 0:
        skips = {k: int(v) for k, v in plan.stats.items() if k not in ("fused", "ops_fused") and v}
        return False, f"no eligible regions (skips: {skips or 'none'})"
    return True, f"{fused} regions / {ops} ops eligible"


def select_runner(
    *,
    module: Any,
    term: Any,
    inputs: Any,
    meta: dict[str, Any] | None = None,
) -> tuple[Any | None, list[str]]:
    """Pick the delivery runner for an eager-lowered *module*.

    Returns ``(runner, notes)`` — ``runner`` is a single runner or a
    ``ChainedRunner`` (region_fuse ahead of cuda_graph), ``None``
    when nothing honest applies.  *term* is ``SearchResult.term``;
    *meta* is ``ModelSpec.meta`` (adapter-declared delivery hints).
    """
    meta = meta or {}
    notes: list[str] = []
    members: list[Any] = []

    eligible, detail = _fusible(term, module)
    notes.append(f"region_fuse: {detail}")
    if eligible:
        from kit_torch.runners import RegionFusionRunner

        members.append(RegionFusionRunner())

    if _has_cuda(module, inputs):
        try:
            from kit_cuda import CudaGraphRunner
        except Exception as e:  # noqa: BLE001
            notes.append(f"cuda_graph: kit_cuda unavailable: {e}")
        else:
            alias, a_note = _alias_spec(meta, "alias_inputs")
            mutable, m_note = _alias_spec(meta, "mutable_inputs")
            for n in (a_note, m_note):
                if n:
                    notes.append(f"cuda_graph: {n}")
            if alias is None and mutable is None:
                notes.append(
                    "cuda_graph: copy-path capture (no alias_inputs/"
                    "mutable_inputs declared in adapter meta)"
                )
            elif alias is not None:
                notes.append(
                    "cuda_graph: alias-aware capture "
                    f"(alias_inputs={alias!r}"
                    + (f", mutable_inputs={mutable!r}" if mutable is not None else "")
                    + ")"
                )
            members.append(CudaGraphRunner(alias_inputs=alias, mutable_inputs=mutable))
    else:
        notes.append(
            "cuda_graph: no CUDA device — eager dispatch only (no graph capture attempted)"
        )

    if not members:
        return None, notes
    if len(members) == 1:
        return members[0], notes
    from mlopt_orchestrator.runners import ChainedRunner

    return ChainedRunner(members), notes


# ---------------------------------------------------------------------------
#  Measurement
# ---------------------------------------------------------------------------


def _bench(
    module: Any,
    inputs: Any,
    iters: int,
    *,
    warmup: int = 3,
    call_kwargs: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """``(min_ms, median_ms)`` per call over *iters* timed forwards.

    ``torch.no_grad`` + eval-mode mirror ``verify_module``'s
    convention; CUDA calls are bracketed by ``torch.cuda.synchronize``
    so async launches can't hide inside the timer.  Shared-GPU noise
    is why BOTH min and median are reported — the caller notes it.
    ``call_kwargs`` (``ModelSpec.kwargs``) are forwarded to every
    ``module(*args, **kwargs)`` invocation.
    """
    torch = _torch()
    cuda = _inputs_cuda(inputs) or _module_cuda(module)
    args = tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)
    kwargs = dict(call_kwargs or {})

    def call() -> Any:
        return module(*args, **kwargs) if args else module(**kwargs)

    def sync() -> None:
        if cuda and torch is not None:
            torch.cuda.synchronize()

    eval_fn = getattr(module, "eval", None)
    if callable(eval_fn):
        eval_fn()
    for _ in range(max(0, warmup)):
        with torch.no_grad() if torch is not None else _nullctx():
            call()
    sync()
    times: list[float] = []
    n = max(1, int(iters))
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad() if torch is not None else _nullctx():
            call()
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
    return min(times), statistics.median(times)


def _bench_paired(
    eager_module: Any,
    config_module: Any,
    inputs: Any,
    iters: int,
    *,
    rounds: int = 5,
    warmup: int = 3,
    call_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Interleaved paired rounds — ``{"eager": stats, "config": stats}``.

    Each of *rounds* rounds times ``iters`` forwards of BOTH modules
    back-to-back, the within-round order randomized by *seed* — the
    delivery-prior autopsy's contention defense: a slow drift across
    an episode can't systematically favor whichever side happens to
    bench second, and eager is re-measured inside every evaluation
    instead of once up front.  Per side the score is the min over
    per-round minima (min is the robust statistic under contention
    spikes — the contiguous-block median doesn't defend against
    drift); the median of per-round medians is reported alongside,
    and ``rounds`` keeps every per-round ``(min_ms, med_ms)`` pair
    for audit and persisted drain records.  Warmup runs once per
    side, at the top of round 0.
    """
    rng = random.Random(seed)
    sides = {"eager": eager_module, "config": config_module}
    order = ["eager", "config"]
    per_round: dict[str, list[tuple[float, float]]] = {"eager": [], "config": []}
    for r in range(max(1, int(rounds))):
        rng.shuffle(order)
        for side in order:
            mn, md = _bench(
                sides[side],
                inputs,
                iters,
                warmup=warmup if r == 0 else 0,
                call_kwargs=call_kwargs,
            )
            per_round[side].append((mn, md))
    out: dict[str, dict[str, Any]] = {}
    for side, samples in per_round.items():
        out[side] = {
            "min_ms": min(mn for mn, _md in samples),
            "med_ms": statistics.median(md for _mn, md in samples),
            "rounds": samples,
        }
    return out


class _nullctx:
    """Stand-in for ``contextlib.nullcontext`` (torch absent)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


# ---------------------------------------------------------------------------
#  The deliver pass
# ---------------------------------------------------------------------------


def deliver(
    *,
    optz: Any,
    res: Any,
    lowered_eager: Any,
    inputs: Any,
    meta: dict[str, Any] | None = None,
    bench_iters: int = 50,
    verify: bool = True,
    runner: Any = None,
    call_kwargs: dict[str, Any] | None = None,
) -> Delivery:
    """Deliver *lowered_eager* under the best honest runner.

    Re-lowers ``res`` under the selected runner so ``verify`` gates
    the DELIVERED module exactly as ``lower`` always has; a runner
    delivery that fails verify — or produces NO verify report under
    ``verify=True`` — is declined and the verified eager lowering
    ships instead (the same mandatory-gate rule
    :mod:`mlopt.deliver_game` applies).  ``runner`` (caller-supplied)
    skips selection — ``optimize_model``'s existing ``runner=`` knob.
    ``call_kwargs`` (``ModelSpec.kwargs``) ride every benchmark call.
    """
    notes: list[str] = []
    eager = lowered_eager.module

    if runner is None:
        runner, sel_notes = select_runner(
            module=eager,
            term=getattr(res, "term", None),
            inputs=inputs,
            meta=meta,
        )
        notes.extend(sel_notes)
    else:
        runner_name = getattr(runner, "name", type(runner).__name__)
        notes.append(f"runner: caller-supplied {runner_name}")

    if runner is None:
        notes.append("deliver: no fast runner applies — eager lowering only")
        emin, emed = _bench(eager, inputs, bench_iters, call_kwargs=call_kwargs)
        return Delivery(
            lowered=lowered_eager,
            eager_min_ms=emin,
            eager_med_ms=emed,
            notes=notes,
        )

    runner_name = getattr(runner, "name", type(runner).__name__)
    try:
        lowered_d = optz.lower(res, inputs, verify=verify, runner=runner)
    except Exception as e:  # noqa: BLE001 — report honestly, keep eager
        notes.append(
            f"deliver: runner {runner_name} lower failed "
            f"({type(e).__name__}: {e}) — shipped the verified eager lowering"
        )
        emin, emed = _bench(eager, inputs, bench_iters, call_kwargs=call_kwargs)
        return Delivery(
            lowered=lowered_eager,
            # runner_name stays None — the runner was DECLINED; the
            # report must not imply a runner shipped.
            runner_name=None,
            eager_min_ms=emin,
            eager_med_ms=emed,
            notes=notes,
        )

    stats = getattr(lowered_d, "stats", {}) or {}
    _ledger_notes(stats, notes)

    verified = getattr(lowered_d, "verified", None)
    passed = None if verified is None else bool(getattr(verified, "passed", True))
    if verify and passed is not True:
        # verify=True makes the gate MANDATORY — an explicit failure
        # AND a missing gate are both declines (the deliver_game rule):
        # a runner delivery that cannot prove equivalence never ships.
        reason = (
            "no verify report"
            if verified is None
            else f"FAILED verify (max_abs={getattr(verified, 'max_abs', '?')})"
        )
        notes.append(
            f"deliver: {runner_name} delivery {reason} — shipped the verified eager lowering"
        )
        emin, emed = _bench(eager, inputs, bench_iters, call_kwargs=call_kwargs)
        return Delivery(
            lowered=lowered_eager,
            runner_name=None,  # declined — eager ships, not the runner
            eager_min_ms=emin,
            eager_med_ms=emed,
            notes=notes,
        )
    if not verify:
        notes.append(f"deliver: --no-verify — {runner_name} shipped WITHOUT the equivalence gate")

    emin, emed = _bench(eager, inputs, bench_iters, call_kwargs=call_kwargs)
    dmin, dmed = _bench(lowered_d.module, inputs, bench_iters, call_kwargs=call_kwargs)
    notes.append(
        "deliver: timings are min/median over "
        f"{max(1, int(bench_iters))} iters on this (possibly shared) GPU"
    )
    return Delivery(
        lowered=lowered_d,
        runner_name=runner_name,
        eager_min_ms=emin,
        eager_med_ms=emed,
        delivered_min_ms=dmin,
        delivered_med_ms=dmed,
        notes=notes,
    )


def _ledger_notes(stats: dict[str, Any], notes: list[str]) -> None:
    """Surface the runner-written stats keys as honest notes."""
    if stats.get("cuda_graph"):
        alias = " (alias-aware)" if stats.get("cuda_graph_alias") else ""
        notes.append(f"cuda_graph: capture armed{alias}")
    elif stats.get("cuda_graph_per_device"):
        notes.append("cuda_graph: per-device capture armed (lazy)")
    if stats.get("cuda_graph_failure") is not None:
        notes.append(f"cuda_graph: capture failed — {stats['cuda_graph_failure']}")
    if stats.get("cuda_graph_skipped"):
        # The runner declined to capture (e.g. an inner compile won
        # precedence) — say so rather than letting the config name
        # imply an armed graph.
        notes.append(f"cuda_graph: skipped — {stats['cuda_graph_skipped']}")
    rf = stats.get("region_fuse")
    if isinstance(rf, dict):
        if rf.get("applied"):
            notes.append(
                f"region_fuse: applied — {rf.get('regions_fused')} regions "
                f"({rf.get('ops_fused')} ops, "
                f"{rf.get('distinct_kernels')} kernels), "
                f"fused-vs-unfused max_rel="
                f"{(rf.get('fused_vs_unfused') or {}).get('max_rel')}"
            )
        elif rf.get("reason"):
            notes.append(f"region_fuse: declined — {rf['reason']}")
    if stats.get("compiled"):
        notes.append("torch_compile: module compiled")

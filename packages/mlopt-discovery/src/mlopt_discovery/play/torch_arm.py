"""The ``torch`` domain arm — real ``nn.Module``s under the sink bound.

Lifts modules through ``mlopt_torch.adapters.TorchSource`` (imported
lazily — this module stays torch-free at import), builds the case
stream and the ``deliver`` hook that lowers a winning extraction to
a verified runnable module.
"""

from __future__ import annotations

import random
from typing import Any

from mlopt_core.cost.basic import count_cost

from .. import meta_arena as ma
from .gen import _time_module

# ---------------------------------------------------------------------------
#  Domain: torch — real modules under the sink's bound
# ---------------------------------------------------------------------------


def _torch_modules() -> dict:
    """Return the torch domain's module zoo (``name -> d -> nn.Module``).

    Data, not code-paths: domain specs select builders by name
    (``corpus.modules``) instead of naming Python.  Torch imports
    lazily — the registry itself stays torch-free.
    """
    import torch
    import torch.nn as nn

    class _SpelledSiLU(nn.Module):
        """x·sigmoid(x) spelled out — the claim move's live site."""

        def forward(self, x):
            """Spell ``x·sigmoid(x)`` — no fused op named."""
            return x * torch.sigmoid(x)

    return {
        "spelled_silu": lambda d: _SpelledSiLU(),
        "mlp": lambda d: nn.Sequential(
            nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)
        ),
        "relu_lin": lambda d: nn.Sequential(nn.Linear(d, d), nn.ReLU()),
        "lin": lambda d: nn.Linear(d, d),
        "softsign_mlp": lambda d: nn.Sequential(
            nn.Linear(d, d), nn.Softsign()
        ),
    }


def _torch_dtype(name: Any) -> Any:
    """Resolve a spec ``hardware.dtype`` spelling to a torch dtype."""
    import torch

    table = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name is None:
        return None
    if name not in table:
        raise ValueError(
            f"hardware.dtype: expected one of {sorted(table)!r}, "
            f"got {name!r}"
        )
    return table[name]


def _torch_cases(
    seed: int,
    n: int,
    *,
    modules: Any = None,
    widths: Any = None,
    batch: int = 4,
    device: Any = None,
    dtype: Any = None,
) -> list:
    """Small real ``nn.Module``s lifted to IR — the product board.

    Each case is a torch module exported through
    ``mlopt_torch.adapters.TorchSource`` and played under the real
    sink's ``supported_ops`` bound — kernel claims price at sink
    costs, not ambient vocabulary.  Modules that fail to export are
    skipped honestly.  Torch is imported lazily — the registry
    itself stays torch-free.

    *modules*/*widths*/*batch*/*device*/*dtype* are the domain-spec
    seams: a spec names zoo builders and shape/hardware bounds as
    data; the built-in ``torch`` domain passes none and gets the
    historical stream.
    """
    import torch
    from mlopt_torch.adapters import TorchSink, TorchSource

    zoo = _torch_modules()
    names = list(zoo) if modules is None else list(modules)
    for name in names:
        if name not in zoo:
            raise ValueError(
                f"corpus.modules: unknown torch builder {name!r} "
                f"(known: {sorted(zoo)!r})"
            )
    widths = [4, 8, 16] if widths is None else list(widths)
    if not widths or any(
        not isinstance(w, int) or w <= 0 for w in widths
    ):
        raise ValueError(
            f"shapes.widths: expected positive ints, got {widths!r}"
        )
    dt = _torch_dtype(dtype)
    supported = TorchSink().supported_ops
    src = TorchSource()
    rng = random.Random(seed)
    out = []
    for i in range(n):
        # draw order matches the historical stream: width, then builder
        d = widths[rng.randrange(len(widths))]
        name = names[rng.randrange(len(names))]
        torch.manual_seed(seed * 10_007 + i)
        model = zoo[name](d).eval()
        if device is not None and dt is not None:
            model = model.to(device=device, dtype=dt)
        elif device is not None:
            model = model.to(device)
        elif dt is not None:
            model = model.to(dtype=dt)
        example = torch.randn(
            batch,
            d,
            device=device,
            dtype=dt,
        )
        ir, leaves = src.to_ir(model, example)
        out.append(
            (
                f"torch{i}:{name}",
                ir.root,
                None,
                (),
                supported,
                {
                    "ir": ir,
                    "leaves": leaves,
                    "input": example,
                    "model": model,
                },
            )
        )
    return out


def _torch_board(case: Any) -> ma.MetaArena:
    """Board a torch-lifted program under the real sink bound."""
    rules = case[2] if len(case) > 2 else None
    return ma.MetaArena(
        case[1],
        rules,
        cost_fn=count_cost,
        supported=case[4] if len(case) > 4 else None,
    )


def _is_plan_tape(module: Any) -> bool:
    """Return True when *module* evaluates through a plan tape.

    The two shapes ``CudaGraphRunner`` knows how to capture: an
    ``IRModule`` whose root flattened to an ``_EvalPlan`` (the
    ``_root_plan`` slot tape — the ~350 µs host-dispatch loss of
    ``docs/reports/t034_module_costing.md``), or a batched executor
    exposing ``capture_cuda_graph``.
    """
    if hasattr(module, "capture_cuda_graph"):
        return True
    try:
        from mlopt_torch.torch_bridge import IRModule
    except ImportError:  # pragma: no cover — torch-free env
        return False
    return isinstance(module, IRModule) and (
        getattr(module, "_root_plan", None) is not None
    )


def _max_abs_delta(
    a: Any, b: Any
) -> float:  # pragma: no cover — CUDA-only caller
    """Max |a - b| over tensors or (nested) tuples of them; NaN else."""
    import torch

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return float("nan")
        return float((a - b).abs().max().item())
    if (
        isinstance(a, (tuple, list))
        and isinstance(b, (tuple, list))
        and len(a) == len(b)
    ):
        vals = [
            v
            for v in (
                _max_abs_delta(x, y) for x, y in zip(a, b, strict=True)
            )
            if v == v
        ]
        return max(vals) if vals else float("nan")
    return float("nan")


def _deliver_runner_arm(
    module: Any, ex: dict, *, reps: int = 50
) -> dict:
    """Attempt the ``CudaGraphRunner`` arm on the delivered module.

    ``docs/reports/t034_module_costing.md`` measured the assembled
    module's loss as ~350 µs of tape-interpreter host dispatch per
    call; the shipped ``CudaGraphRunner`` replays the *same*
    verified forward as one CUDA graph.  This arm runs it on the
    deliver path: capture is attempted only when every example input
    is already a CUDA tensor (the case's device is the deliver
    contract — a CPU case is never silently promoted), only after
    ``verify`` passed, and every failure degrades to the eager
    module plus a ``runner_note`` — capture never fails a deliver.

    Additive keys: ``runner``/``cuda_graph`` record the attempt,
    ``plan_tape`` whether the module is the dispatch-bound shape,
    ``deliver_us``/``graph_us`` the honest eager-vs-replay medians
    (the replay's per-call static-input copies are *inside* the
    timed ``wrapped(*xs)`` — ``_GraphReplayModule.forward`` does
    ``buf.copy_(t)`` then ``graph.replay()``, so the copy cost is
    measured, not hidden), ``graph_max_abs`` the replay-vs-eager
    output delta, and ``module`` is overridden with the replay
    wrapper only when capture *and* the numeric check succeeded.
    """
    import torch

    out: dict[str, Any] = {"runner": "cuda_graph"}
    xin = ex.get("input")
    xs = (
        tuple(xin)
        if isinstance(xin, (tuple, list))
        else (() if xin is None else (xin,))
    )
    out["plan_tape"] = _is_plan_tape(module)
    if not (
        xs
        and all(isinstance(t, torch.Tensor) and t.is_cuda for t in xs)
    ):
        out["cuda_graph"] = False
        out["runner_note"] = "inputs not CUDA tensors"
        return out
    try:  # pragma: no cover — CUDA-only body
        from kit_cuda import CudaGraphRunner
    except ImportError:  # pragma: no cover — optional dep absent
        out["cuda_graph"] = False
        out["runner_note"] = "kit-cuda not installed"
        return out
    stats: dict[str, Any] = {}
    try:  # pragma: no cover — CUDA-only body
        wrapped = CudaGraphRunner().apply(module, xin, stats)
    except Exception as exc:  # pragma: no cover — CUDA-only
        out["cuda_graph"] = False
        out["runner_note"] = f"apply raised {type(exc).__name__}: {exc}"
        return out
    if not stats.get("cuda_graph"):  # pragma: no cover — CUDA-only
        failure = stats.get("cuda_graph_failure")
        out["cuda_graph"] = False
        out["runner_note"] = (
            f"capture declined ({failure})"
            if failure is not None
            else "capture declined"
        )
        return out
    try:  # pragma: no cover — CUDA-only body
        t_eager = _time_module(module, xin, reps)
        t_graph = _time_module(wrapped, xin, reps)
        with torch.no_grad():
            out_eager = module(*xs)
            out_graph = wrapped(*xs)
    except Exception as exc:  # pragma: no cover — CUDA-only
        out["cuda_graph"] = False
        out["runner_note"] = (
            f"post-capture check failed: {type(exc).__name__}: {exc}"
        )
        return out
    out.update(  # pragma: no cover — CUDA-only body
        {
            "cuda_graph": True,
            "deliver_us": round(t_eager, 1),
            "graph_us": round(t_graph, 1),
            "graph_speedup": (
                round(t_eager / t_graph, 3) if t_graph else None
            ),
            "graph_max_abs": _max_abs_delta(out_eager, out_graph),
            "module": wrapped,
        }
    )
    return out  # pragma: no cover — CUDA-only success path


def _torch_deliver(board: ma.MetaArena, case: Any, traj: Any) -> dict:
    """Lower the winning extraction and verify it against the model.

    ``deliverable`` resolves handled claims to their kernels and
    unhandled abbreviations to spelled form; ``TorchSink.lower``
    rebuilds the module from it; ``verify`` runs both on the case's
    example input.  The game score becomes a delivered artifact.
    On a verified CUDA deliverable the ``CudaGraphRunner`` arm
    (:func:`_deliver_runner_arm`) additionally attempts graph
    capture — additive keys only, eager module on any decline.

    Honesty contract: this is a measurement arm, not the enforced
    ``Optimizer.lower(verify=True)`` shipping gate — it reports the
    verdict rather than raising on it.  ``delivered`` is therefore
    ``rep.passed``: a lowered-but-unverified module is marked
    ``delivered_unverified`` (the module stays on the record for
    timing/inspection) and is NEVER reported as ``delivered``.
    """
    from mlopt_core.ir import IR
    from mlopt_torch.adapters import TorchSink

    ex = case[5]
    best = board.eg.extract_best(board.root, board.feasible_cost)
    if best is None:
        return {"delivered": False, "reward": traj.total}
    term = board.deliverable(best)
    ir = IR(
        root=term,
        inputs=list(ex["ir"].inputs),
        input_names=set(ex["ir"].input_names),
        params=dict(ex["ir"].params),
    )
    sink = ex.get("sink") or TorchSink()
    module = sink.lower(ir, params=ex["leaves"])
    rep = sink.verify(ex["model"], module, (ex["input"],))
    t = traj.terminal
    out = {
        "delivered": bool(rep.passed),
        "module": module,
        "verified": rep.passed,
        "max_abs": rep.max_abs,
        "cost": getattr(t, "cost", None),
        "cert": getattr(t, "certificate_ok", None),
        "reward": traj.total,
    }
    if rep.passed:
        out.update(_deliver_runner_arm(module, ex))
    else:
        out["delivered_unverified"] = True
    return out

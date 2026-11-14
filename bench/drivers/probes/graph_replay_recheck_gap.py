"""Post-8589162 recheck: generic-path graph claims in
``docs/results_wholemodel_gap.md`` (the retnet ``_ChunkedRetention``
case and the chainfuse ``_mlp_case``).

Those reports' ``*+graph(run)`` rows went through the same buggy
``_GraphReplayModule`` generic path as the zoo sweep arms — timing
survives the fix by construction, but no fresh-input correctness
check was ever run.  This driver reuses the probe's own fixtures
(``_saturate`` / ``Optimizer`` flow, imported by path —
``bench/drivers`` is not a package) and checks replay vs eager vs
the source module on FRESH inputs only; it skips every timing arm.

Run::

    .venv/bin/python -u -m bench.drivers.probes.graph_replay_recheck_gap
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

_PROBE = Path(__file__).resolve().parent / "wholemodel_gap_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "wholemodel_gap_probe", _PROBE
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _fresh(x: torch.Tensor, variant: int) -> torch.Tensor:
    if variant % 2 == 0:
        return x * (0.5 + torch.rand((), device=x.device).item()) + (
            0.05 * torch.randn_like(x)
        )
    return torch.randn_like(x) * (x.std() + 1e-3)


def _delta(a: torch.Tensor, b: torch.Tensor) -> float:
    both_nan = torch.isnan(a) & torch.isnan(b)
    diff = torch.where(both_nan, torch.zeros_like(a - b), (a - b).abs())
    return float(diff.max().item())


def _check(
    label: str,
    wrapped: torch.nn.Module,
    base: torch.nn.Module,
    x: torch.Tensor,
    k: int,
) -> dict:
    """Replay vs its own eager inner vs the source model.

    ``wrapped.inner`` is the ``_GraphReplayModule``'s eager module —
    the correct reference regardless of which lowering route produced
    it (a routed delivery legitimately differs from the unrouted
    ``generic`` module).
    """
    eager = getattr(wrapped, "inner", wrapped)
    out = {"arm": label, "checks": []}
    with torch.no_grad():
        for i in range(k):
            xf = _fresh(x, i)
            o_e = eager(xf)
            o_g = wrapped(xf).clone()
            o_b = base(xf)
            out["checks"].append(
                {
                    "variant": i,
                    "graph_vs_eager": _delta(o_g, o_e),
                    "eager_vs_base": _delta(o_e, o_b),
                }
            )
    out["max_abs_graph_vs_eager"] = max(
        c["graph_vs_eager"] for c in out["checks"]
    )
    return out


def retnet_case(probe, *, t: int, k: int, chunk: int) -> dict:
    """``_retnet_case``'s generic-capture arms, correctness only."""
    from typing import cast

    import torch.nn as nn
    from mlopt_core.ir import IR
    from mlopt_cuda import CudaGraphRunner
    from mlopt_torch.adapters import TorchSink

    row: dict = {"case": f"retnet_T{t}_K{k}", "arms": []}
    sat = probe._saturate(t, k, iters=14, budget=200, chunk=chunk)
    sink = TorchSink()
    ir = sat["ir"]
    ex = IR(
        root=sat["best"],
        inputs=list(ir.inputs),
        input_names=set(ir.input_names),
        params=dict(ir.params),
    )
    base = sat["model"].to("cuda").double()
    carrier = (
        cast(nn.Module, sink.lower(ex, params=sat["leaves"]))
        .to("cuda")
        .double()
    )
    x64 = sat["x"].to("cuda")

    # generic carrier (fp64) — the 'carrier64+graph(run)' arm
    stats: dict = {}
    wrapped = CudaGraphRunner().apply(carrier, (x64,), stats)
    row["carrier64"] = {
        "cuda_graph": bool(stats.get("cuda_graph")),
        "failure": stats.get("cuda_graph_failure"),
    }
    if stats.get("cuda_graph"):
        row["arms"].append(
            _check("carrier64+graph(run)", wrapped, base, x64, 4)
        )

    # fp32 arm — 'carrier32+graph(run)'
    leaves32 = {n: v.detach().float() for n, v in sat["leaves"].items()}
    carrier32 = cast(nn.Module, sink.lower(ex, params=leaves32)).to(
        "cuda"
    )
    base32 = base.float()
    x32 = x64.float()
    stats32: dict = {}
    wrapped32 = CudaGraphRunner().apply(carrier32, (x32,), stats32)
    row["carrier32"] = {
        "cuda_graph": bool(stats32.get("cuda_graph")),
        "failure": stats32.get("cuda_graph_failure"),
    }
    if stats32.get("cuda_graph"):
        row["arms"].append(
            _check("carrier32+graph(run)", wrapped32, base32, x32, 4)
        )
    row["sat_secs"] = sat["sat_secs"]
    row["root_op"] = sat["root_op"]
    return row


def chainfuse_case(probe, *, reps: int) -> dict:
    """``_mlp_case``'s generic-IRModule graph arms, correctness only."""
    from typing import cast

    import torch.nn as nn
    from mlopt_core.ir import IR
    from mlopt_cuda import CudaGraphRunner
    from mlopt_orchestrator.optimize import Optimizer
    from mlopt_torch.adapters import TorchSink
    from mlopt_torch.backend import TorchBackend

    row: dict = {"case": "chainfuse_mlp", "arms": []}
    model = probe._ChainFuse().to("cuda").double().eval()
    x64 = torch.randn(256, 512, device="cuda", dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(model, x64, max_iterations=8)
    ex_ir = IR(
        root=res.term,
        inputs=list(res.ir.inputs),
        input_names=set(res.ir.input_names),
        params=dict(res.ir.params),
    )
    generic = (
        cast(
            nn.Module,
            TorchSink().lower(ex_ir, params=dict(res.param_values)),
        )
        .to("cuda")
        .double()
    )
    stats_g: dict = {}
    generic_run = CudaGraphRunner().apply(generic, (x64,), stats_g)
    row["generic64"] = {
        "cuda_graph": bool(stats_g.get("cuda_graph")),
        "failure": stats_g.get("cuda_graph_failure"),
    }
    if stats_g.get("cuda_graph"):
        row["arms"].append(
            _check("generic64+graph(run)", generic_run, model, x64, 4)
        )
    # the delivery-routed module through opt.lower(runner=...) —
    # 'delivered64+graph(run)'
    del_r = opt.lower(res, x64, runner=CudaGraphRunner(), verify=False)
    row["delivered64"] = {
        "cuda_graph": bool(del_r.stats.get("cuda_graph")),
        "failure": del_r.stats.get("cuda_graph_failure"),
    }
    if del_r.stats.get("cuda_graph"):
        mod_r = cast(nn.Module, del_r.module)
        row["arms"].append(
            _check("delivered64+graph(run)", mod_r, model, x64, 4)
        )
    return row


def main() -> int:
    probe = _load_probe()
    rows: list[dict] = []
    try:
        rows.append(retnet_case(probe, t=1024, k=8, chunk=60))
    except Exception as exc:
        rows.append(
            {
                "case": "retnet",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"retnet FAILED: {exc}", flush=True)
    try:
        rows.append(chainfuse_case(probe, reps=30))
    except Exception as exc:
        rows.append(
            {
                "case": "chainfuse_mlp",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"chainfuse FAILED: {exc}", flush=True)
    out = Path("/tmp/graph_recheck_gap.json")
    out.write_text(json.dumps(rows, indent=1, default=str))
    for r in rows:
        print(json.dumps(r, default=str)[:2000], flush=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

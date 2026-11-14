"""Phase C serving/autograd probe — the measured boundary, as data.

``docs/reports/phase_c_serving.md`` is the write-up; this driver is
the reproducible measurement behind it.  It exercises, on small
modules through the real ``Optimizer`` pipeline:

* **artifact paths** — ``.pt2`` (``export_optimized``),
  ``torch.jit.trace`` (the ``torchscript`` fmt), and the ONNX route
  (``Optimizer(source=TorchSource(), sink=OnnxSink())`` →
  ``OnnxExecutor.save`` → fresh ``onnxruntime`` session on the file),
  plus the ``torch.onnx.export`` attempt on a loaded ``.pt2``.
* **dynamic shapes** — static vs ``dynamic_shapes`` ``.pt2``, eager
  ``IRModule`` forwards at off-example batch sizes, and the ONNX
  graph's fixed input dims.
* **autograd** — forward+backward on the delivered module (input grad
  and per-parameter grads vs the eager original), the ``.pt2``
  loaded module's grad surface, and the ``_align_to`` rebind boundary
  (CPU-built module served CUDA inputs without ``.to()`` — needs
  ``--cuda``).

Honesty contract: every cell reports ``ok`` / ``fail:<ExcType>`` plus
the measured numbers; nothing is asserted away.  Run::

    .venv/bin/python bench/drivers/probes/phase_c_serving_probe.py
    .venv/bin/python bench/drivers/probes/phase_c_serving_probe.py --cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn


def _mlp() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)
    ).eval()


def _optimize(mod: nn.Module, x: torch.Tensor, **kw):
    from mlopt_orchestrator import Optimizer
    from mlopt_torch.backend import TorchBackend

    module, _stats = Optimizer(backend=TorchBackend()).optimize(
        mod, x, verify=False, verbose=False, **kw
    )
    return module


def _cell(fn):
    """Run ``fn``; report ok/fail honestly."""
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


def artifact_paths(workdir: Path) -> dict:
    """The serving-artifact matrix: pt2 / torchscript / onnx."""
    import numpy as np
    from mlopt_torch.export import export_optimized, load_optimized

    x = torch.randn(2, 8)
    opt = _optimize(_mlp(), x)
    ref = opt(x).detach()
    out: dict = {}

    # -- .pt2 ---------------------------------------------------------
    pt2 = workdir / "opt.pt2"
    export_optimized(None, opt, pt2, example_input=x)
    loaded = load_optimized(pt2)
    out["pt2"] = {
        "runs": _cell(lambda: float((loaded(x) - ref).abs().max())),
        "manifest_kind": "runnable",
    }

    # -- torchscript (trace) ------------------------------------------
    def _ts():
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = torch.jit.trace(opt, x)
        ok = bool(torch.allclose(ts(x), ref, atol=1e-6))
        # autograd surface of the traced module
        xi = x.clone().requires_grad_(True)
        ts(xi).square().sum().backward()
        return {
            "forward_close": ok,
            "input_grad": xi.grad is not None,
            "param_grads": [
                p.grad is not None for p in ts.parameters()
            ],
        }

    out["torchscript_trace"] = _cell(_ts)

    # -- ONNX via OnnxSink (the alternative-pipeline route) ------------
    def _onnx():
        import onnxruntime as ort
        from mlopt_onnx import OnnxSink
        from mlopt_onnx.executor import OnnxExecutor
        from mlopt_orchestrator import Optimizer
        from mlopt_torch.backend import TorchSource

        torch.manual_seed(0)
        mod2 = _mlp()
        res = Optimizer(source=TorchSource(), sink=OnnxSink()).optimize(
            mod2, x, max_iterations=4
        )
        verified = (
            res.verified.passed if res.verified is not None else None
        )
        executor = res.module
        if not isinstance(executor, OnnxExecutor):
            raise TypeError(
                f"OnnxSink delivered {type(executor).__name__}, "
                "expected OnnxExecutor"
            )
        path = workdir / "opt.onnx"
        executor.save(path)
        sess = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        feed = {executor.input_names[0]: x.numpy()}
        y = sess.run(None, feed)
        y = y[0] if len(y) == 1 else y
        with torch.no_grad():
            ref2 = mod2(x).numpy()
        close = bool(np.allclose(np.asarray(y), ref2, atol=1e-5))
        # static-shape check: a different batch must be refused
        try:
            sess.run(
                None,
                {
                    executor.input_names[0]: np.zeros(
                        (5, 8), dtype=np.float32
                    )
                },
            )
            batch5 = "accepted"
        except Exception:
            batch5 = "refused"
        return {
            "verified": verified,
            "file_roundtrip_close": close,
            "batch5": batch5,
            "providers_available": ort.get_available_providers(),
        }

    out["onnx_sink"] = _cell(_onnx)

    # -- torch.onnx.export on the loaded .pt2 --------------------------
    def _torch_onnx():
        onnx_path = workdir / "pt2.onnx"
        torch.onnx.export(
            loaded,
            (x,),
            str(onnx_path),
            input_names=["x"],
            output_names=["y"],
            opset_version=18,
        )
        return "exported"

    out["torch_onnx_from_pt2"] = _cell(_torch_onnx)
    return out


def dynamic_shapes(workdir: Path) -> dict:
    """Which delivered forms accept off-example shapes."""
    from mlopt_torch.export import export_optimized, load_optimized
    from torch.export import Dim

    x = torch.randn(2, 8)
    opt = _optimize(_mlp(), x)
    out: dict = {}

    # static .pt2 — guard pins the example shape
    sp = workdir / "static.pt2"
    export_optimized(None, opt, sp, example_input=x)
    ls = load_optimized(sp)

    def _static(b):
        return tuple(ls(torch.randn(b, 8)).shape)

    out["pt2_static"] = {
        "batch2": _cell(lambda: _static(2)),
        "batch5": _cell(lambda: _static(5)),
    }

    # dynamic .pt2 — the *xs varargs pytree needs one extra nesting
    dp = workdir / "dyn.pt2"
    batch = Dim("batch")
    dyn_export = _cell(
        lambda: export_optimized(
            None,
            opt,
            dp,
            example_input=x,
            dynamic_shapes=(({0: batch},),),
        )
    )
    out["pt2_dynamic_export"] = dyn_export
    if dyn_export["ok"]:
        ld = load_optimized(dp)
        out["pt2_dynamic"] = {
            f"batch{b}": _cell(
                lambda b=b: tuple(ld(torch.randn(b, 8)).shape)
            )
            for b in (2, 5, 7)
        }

    # eager IRModule off-example batch (the delivered module itself)
    out["eager_irmodule"] = {
        f"batch{b}": _cell(
            lambda b=b: tuple(opt(torch.randn(b, 8)).shape)
        )
        for b in (2, 5)
    }
    return out


def dynamic_shapes_compositional() -> dict:
    """Off-example batch on a Compositional delivery (MiniGPT)."""
    try:
        sys.path.insert(0, str(Path(__file__).parents[3] / "tests"))
        import importlib

        minigpt_mod = importlib.import_module("test_compositional")
        MiniGPT = minigpt_mod.MiniGPT
    except Exception as e:
        return {"skip": f"{type(e).__name__}: {e}"[:200]}
    from mlopt_orchestrator import Compositional, Optimizer
    from mlopt_torch.backend import TorchBackend

    torch.manual_seed(0)
    model = MiniGPT(dim=32, n_heads=2, depth=2, hidden_mult=2).eval()
    x = torch.randn(1, 8, 32)
    opt, _ = Optimizer(backend=TorchBackend()).optimize(
        model, x, strategy=Compositional(), verbose=False
    )
    out: dict = {}
    for b in (1, 3):
        out[f"batch{b}"] = _cell(
            lambda b=b: tuple(opt(torch.randn(b, 8, 32)).shape)
        )
    return out


def autograd(workdir: Path, use_cuda: bool) -> dict:
    """The measured autograd surface of delivered modules."""
    from mlopt_torch.export import export_optimized, load_optimized

    dev = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 8, device=dev)
    mod = _mlp().to(dev)
    opt = _optimize(mod, x)
    ref_mod = _mlp().to(dev)
    ref_mod.load_state_dict(mod.state_dict())

    out: dict = {"device": dev}

    # delivered module: input + param grads vs eager
    def _delivered():
        xin = x.clone().requires_grad_(True)
        opt(xin).square().sum().backward()
        xe = x.clone().requires_grad_(True)
        ref_mod(xe).square().sum().backward()
        assert xin.grad is not None and xe.grad is not None
        pg = [p.grad is not None for p in opt.parameters()]
        return {
            "input_grad_close": bool(
                torch.allclose(xin.grad, xe.grad, atol=1e-5)
            ),
            "input_grad_max_abs": float(
                (xin.grad - xe.grad).abs().max()
            ),
            "param_grads": f"{sum(pg)}/{len(pg)}",
        }

    out["delivered_module"] = _cell(_delivered)

    # .pt2 loaded module: what does backward reach?
    pt2 = workdir / "autograd.pt2"
    export_optimized(None, opt, pt2, example_input=x)
    loaded = load_optimized(pt2)

    def _pt2():
        xi = x.clone().requires_grad_(True)
        loaded(xi).square().sum().backward()
        xe = x.clone().requires_grad_(True)
        ref_mod(xe).square().sum().backward()
        assert xi.grad is not None and xe.grad is not None
        return {
            "input_grad_close": bool(
                torch.allclose(xi.grad, xe.grad, atol=1e-5)
            ),
            "param_grads": [
                p.grad is not None for p in loaded.parameters()
            ],
        }

    out["pt2_loaded"] = _cell(_pt2)

    # _align_to rebind boundary: CPU-built module, CUDA inputs, no .to()
    if use_cuda and torch.cuda.is_available():
        xc = torch.randn(2, 8)
        opt_cpu = _optimize(_mlp(), xc)

        def _rebind():
            xin = xc.cuda().clone().requires_grad_(True)
            opt_cpu(xin).square().sum().backward()
            registered = [
                p.grad is not None for p in opt_cpu.parameters()
            ]
            live = [
                v.grad is not None for v in opt_cpu._param_map.values()
            ]
            same = all(
                id(opt_cpu._param_map[k]) == id(opt_cpu._parameters[k])
                for k in opt_cpu._param_map
                if k in opt_cpu._parameters
            )
            return {
                "input_grad": xin.grad is not None,
                "registered_param_grads": f"{sum(registered)}/{len(registered)}",
                "live_table_param_grads": f"{sum(live)}/{len(live)}",
                "param_map_is_parameters": same,
            }

        out["align_to_rebind_cuda_input"] = _cell(_rebind)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cuda",
        action="store_true",
        help="also run the CUDA autograd cells (needs a GPU)",
    )
    args = ap.parse_args()
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        report = {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "artifact_paths": artifact_paths(workdir),
            "dynamic_shapes": dynamic_shapes(workdir),
            "dynamic_shapes_compositional": dynamic_shapes_compositional(),
            "autograd": autograd(workdir, args.cuda),
        }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

"""kernelopt×triton wiring probe — hand-Triton kernels in the delivery path.

Follow-up to ``kernelopt_bridge_probe`` (which found the ``kf_*``
``torch.compile`` composites tie-or-lose on whisper subterms) and the
kernelopt_integration_design note's closing recommendation: *"revisit
if hand-Triton epilogue kernels get wired into ``KernelSink``
lowering (today they are a separate bench arm, not the sink's codegen
path)."*  This probe wires exactly that — the
``kernelopt:bench/limits/triton`` hand-written kernels — onto the
``kf_mm_*`` epilogue ops the ``KERNEL_RULES`` bank mints, and measures
honestly per site.

The wiring seam needs **no kernelopt source change**:
``KernelExecutor`` already accepts a ``bindings`` override, and only
``FUSED_KERNEL_OPS`` route through ``torch.compile`` — so a term
rewrite ``kf_mm_bias → triton_mm_bias`` (etc.) plus a bindings table
carrying the Triton callables *is* the delivery path.  The Triton
builders are loaded from the kernelopt checkout by file path (the
``bench`` package name would collide with graphopt's own); their
``bench.limits.registry`` import resolves to this repo's registry —
the same ``benchkit.baselines`` contract types.

Family wired: the ``_triton_mm`` epilogue family —
``kernel_mm_bias`` / ``kernel_mm_scale`` (baked scale 0.125, the
whisper ``1/√64`` attention scale) / ``kernel_mm_gelu`` /
``kernel_mm_scale_bias`` (baked 0.5).  ``kf_sdpa``/``kf_softmax_mm``
are atomic fused ops the 2-D family does not cover — they stay on the
compiled-composite path and are counted, not silently dropped.

Honest-cost notes baked into the measurement:

- ``w`` arrives as the evaluated ``transpose(param)`` — a non-
  contiguous ``(K,N)`` view, but a *static* subtree (Param leaf), so
  the executor caches it once.  The Triton wrapper packs it
  contiguous keyed by ``data_ptr`` — the load-time weight-packing a
  shipped kernel always does (cuBLAS does the same).  Activations
  are reshaped ``(...,K) → (∏,K)`` and the output reshaped back —
  views, no copies.
- ``kf_mm_scale``/``kf_mm_sb`` rewrite only when the scale leaf is a
  ``Const`` matching a baked builder (0.125 / 0.5); a live or
  unmatched scale declines to the compiled composite and is counted.

Run::

    .venv/bin/python bench/drivers/probes/kernelopt_triton_probe.py [name ...]
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Const, IR, Op

_PROBES = Path(__file__).resolve().parent
_ROOT = _PROBES.parents[2]  # graphopt repo root — makes ``bench`` importable
for _p in (str(_PROBES), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import kernelopt_bridge_probe as bridge  # noqa: E402

from mlopt_kernel import KERNEL_RULES, KernelSink, kernel_cost  # noqa: E402
from mlopt_kernel.sink import KernelExecutor  # noqa: E402

_TRITON_INIT = (
    "/home/luis/GitHub/handl-lang/kernelopt/bench/limits/triton/"
    "__init__.py"
)


def _load_triton_baseline():
    """Import kernelopt's ``bench.limits.triton`` by file path."""
    spec = importlib.util.spec_from_file_location(
        "kernelopt_triton_baseline", _TRITON_INIT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------
# term rewrite: kf_mm_* → triton_mm_* (the measured arm)
# ------------------------------------------------------------------

#: fused op → (triton op name, kernelopt bench case builder name,
#:             scale gate).  ``scale`` is the Const value the baked
#: builder implements; ``None`` means scale-free.
_MM_MAP = {
    "kf_mm_bias": ("triton_mm_bias", "kernel_mm_bias", None),
    "kf_mm_gelu": ("triton_mm_gelu", "kernel_mm_gelu", None),
    "kf_mm_scale": ("triton_mm_scale", "kernel_mm_scale", 0.125),
    "kf_mm_sb": ("triton_mm_sb", "kernel_mm_scale_bias", 0.5),
}


def _scale_of(arg) -> float | None:
    """The scale leaf's baked value, or None when not a Const."""
    if isinstance(arg, Const):
        try:
            return float(arg.value)
        except (TypeError, ValueError):
            return None
    return None


def rewire_mm(t, memo=None, stats=None):
    """Rewrite ``kf_mm_*`` ops whose scale a baked Triton kernel has."""
    if memo is None:
        memo = {}
    if stats is None:
        stats = {"rewired": [], "declined": []}
    k = id(t)
    if k in memo:
        return memo[k]
    if not isinstance(t, Op):
        return t
    args = tuple(rewire_mm(a, memo, stats) for a in t.args)
    op = t.op
    if op in _MM_MAP:
        triton_op, builder, want_scale = _MM_MAP[op]
        # scale-bearing ops: the scale is the LAST arg; only rewire
        # when it is a Const the baked builder implements.
        if want_scale is not None:
            got = _scale_of(args[-1])
            if got != want_scale:
                stats["declined"].append((op, got))
                out = Op(op, args, dict(t.attrs))
                memo[k] = out
                return out
        stats["rewired"].append(op)
        out = Op(triton_op, args, dict(t.attrs))
        memo[k] = out
        return out
    out = Op(op, args, dict(t.attrs))
    memo[k] = out
    return out


# ------------------------------------------------------------------
# Triton bindings — the _triton_mm epilogue family, batch- and
# contiguity-adapted (the "packed weight" a shipped kernel assumes)
# ------------------------------------------------------------------

_PACK_CACHE: dict = {}


def _packed_w(w):
    """Contiguous ``(K,N)`` weight, cached by identity (param pack)."""
    key = (w.data_ptr(), tuple(w.shape), tuple(w.stride()))
    hit = _PACK_CACHE.get(key)
    if hit is None:
        hit = w.contiguous()
        _PACK_CACHE[key] = hit
    return hit


def _mm_wrap(run):
    """Adapt a 2-D ``run(a, w, *rest)`` builder to nd activations."""

    def fn(a, w, *rest):
        lead = a.shape[:-1]
        k = a.shape[-1]
        a2 = a.reshape(-1, k) if a.dim() > 2 else a
        out = run(a2, _packed_w(w), *rest)
        return out.reshape(*lead, out.shape[-1])

    return fn


def triton_bindings(kt) -> dict:
    """The sink binding table + the Triton mm-epilogue family."""
    sink = KernelSink()
    table = dict(sink.ops.torch_bindings)
    for triton_op, builder_name, _ in _MM_MAP.values():
        mk = kt.kernel_builder(builder_name)
        if mk is None:
            raise RuntimeError(f"no builder {builder_name!r}")
        run = mk()  # builders return the ``fn(*feeds)`` callable
        table[triton_op] = _mm_wrap(run)
    return table


# ------------------------------------------------------------------
# per-site measurement
# ------------------------------------------------------------------


def extract_best(ir, decompose=True):
    """kernelopt path: decompose → KERNEL_RULES → extract_best."""
    ir_d = bridge._rebuild_ir(ir, bridge._decompose(ir.root))
    eg = EGraph()
    root = eg.add_term(ir_d.root)
    for r in KERNEL_RULES:
        try:
            eg.apply_rule(r, root)
        except Exception:
            pass
    best = eg.extract_best(root, kernel_cost) or ir_d.root
    return ir_d, best


def count_ops(t, counts):
    if isinstance(t, Op):
        counts[t.op] = counts.get(t.op, 0) + 1
        for a in t.args:
            count_ops(a, counts)


def measure(name, mod, args, ir, param_values, kt):
    """One subterm: eager / kernelopt-incumbent / triton-wired arm."""
    ir_d, best = extract_best(ir)
    counts: dict = {}
    count_ops(best, counts)
    fused = {k: v for k, v in counts.items()
             if k.startswith(("kf_", "triton_")) or k == "om_decayed"}

    ir2 = IR(
        root=best,
        inputs=list(ir_d.inputs),
        input_names=set(ir_d.input_names),
        params=dict(ir_d.params),
    )
    sink = KernelSink()
    try:
        spelled = sink.lower(ir_d, params=param_values)
        sched = sink.lower(ir2, params=param_values)
    except Exception as e:
        return {"name": name, "note": f"lower failed: {e}"}

    stats = {"rewired": [], "declined": []}
    best_t = rewire_mm(best, stats=stats)
    changed = repr(best_t) != repr(best)
    triton_exec = None
    if changed or stats["rewired"]:
        ir3 = IR(
            root=best_t,
            inputs=list(ir_d.inputs),
            input_names=set(ir_d.input_names),
            params=dict(ir_d.params),
        )
        try:
            triton_exec = KernelExecutor(
                ir3,
                param_values,
                bindings=triton_bindings(kt),
            )
        except Exception as e:
            return {"name": name,
                    "note": f"triton executor failed: {e}"}

    feeds = tuple(
        torch.randn(tuple(v.typ.shape), dtype=torch.float32,
                    device="cuda")
        for v in ir.inputs
    )
    row = {"name": name, "fused_ops": fused, "rewired": stats["rewired"],
           "declined": stats["declined"], "changed": changed}

    with torch.no_grad():
        ref_out = mod(*feeds)
    try:
        row["sched_verify"] = bool(
            sink.verify(spelled, sched, feeds).passed
        )
    except Exception as e:
        row["sched_verify"] = f"err:{e}"
    if triton_exec is not None:
        try:
            rep = sink.verify(spelled, triton_exec, feeds)
            row["triton_verify"] = bool(rep.passed)
            row["triton_max_rel"] = rep.max_rel
            with torch.no_grad():
                got = triton_exec(*feeds)
            d = (ref_out - got).abs().max().item()
            row["triton_vs_module_rel"] = d / (
                ref_out.abs().max().item() + 1e-8
            )
        except Exception as e:
            row["triton_verify"] = f"err:{type(e).__name__}: {e}"

    def _t(label, fn):
        try:
            row[label] = bridge.timeit(fn)
        except Exception as e:
            row[label] = f"err:{type(e).__name__}"

    _t("eager_us", lambda: mod(*feeds))
    _t("spelled_us", lambda: spelled(*feeds))
    _t("sched_us", lambda: sched(*feeds))
    if triton_exec is not None:
        _t("triton_us", lambda: triton_exec(*feeds))
    return row


def main() -> None:
    from mlopt_orchestrator.optimize import Optimizer
    from mlopt_torch.backend import TorchBackend

    kt = _load_triton_baseline()
    only = set(sys.argv[1:])
    cases = bridge._cases()
    print(f"[setup] cuda={torch.cuda.is_available()} "
          f"dev={torch.cuda.get_device_name(0)}")
    print(f"[setup] triton builders: "
          f"{sorted(kt._KERNEL_BUILDERS)}")

    for name, (mod, args_cpu) in cases.items():
        if only and name not in only:
            continue
        mod = mod.cuda().eval()
        args = tuple(a.cuda() for a in args_cpu)
        try:
            res = Optimizer(backend=TorchBackend()).search(
                mod, args, max_iterations=20
            )
        except Exception as e:
            print(f"\n=== {name} ===\n  search FAILED: {e}")
            continue
        row = measure(name, mod, args, res.ir, res.param_values, kt)
        print(f"\n=== {name} ===")
        for k, v in row.items():
            if k == "name":
                continue
            print(f"  {k}: {v}")
        if (
            isinstance(row.get("triton_us"), float)
            and isinstance(row.get("sched_us"), float)
            and isinstance(row.get("eager_us"), float)
        ):
            print(f"  [cmp] sched/triton: "
                  f"{row['sched_us'] / row['triton_us']:.2f}x  "
                  f"eager/triton: "
                  f"{row['eager_us'] / row['triton_us']:.2f}x")


if __name__ == "__main__":
    main()

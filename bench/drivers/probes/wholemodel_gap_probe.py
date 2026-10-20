"""Whole-model ~1x parity gap — attribution probe (scope, not fix).

The honest-limits negative: delivered modules run at ~parity with the
incumbent spelling while FLA-style fused kernels sit ~7-11x ahead
(``project/retros/retnet-external-baselines.md``, ``docs/results.md``
§5).  The suspected cause is lowering, not semantics: the shipped
delivery lowers extracted terms to per-leaf generic torch ops (the
:class:`~mlopt_torch.torch_bridge.IRModule` plan tape), fp64 GEMMs on a
1:64-rate consumer GPU, and materialized intermediates — while the
carrier executor batching (``decayed_lower``) and the CUDA-graph
runner exist but only cover what claims them.

This probe decomposes the measured gap on the two strongest cases:

- **retnet/chunked** (``zoo._ChunkedRetention``, T in {1024, 2048},
  K=8 — the same shape as the external-baseline retro): a ladder of
  arms from the spelled model through the generic lowering, the
  batched carrier executor (the real delivery path), and CUDA-graph
  replay, each in fp64 and fp32, each split by the profiler into
  wall-clock vs CUDA-kernel-busy time (dispatch/Python overhead is the
  difference), with kernel time classified GEMM / pointwise /
  copy-mem.
- **chainfuse MLP** (x@A@B@C + 6-op pointwise tail, (256,512) d=512 —
  the gen-domain case): laws-only delivery lowers to one folded GEMM +
  the tail spelled per-leaf; arms isolate per-leaf dispatch (manual
  graph capture of the generic module), tail fusion (Inductor), and
  the fp64 GEMM penalty.

The CUDA-graph lane is measured two ways:

1. ``CudaGraphRunner().apply(module, x, stats)`` — the shipped cheap
   lane.  It only reaches modules exposing ``capture_cuda_graph``
   (batched carrier executors); generic ``IRModule``s degrade quietly
   (``stats["cuda_graph"]`` stays False).  Both outcomes are printed.
2. A manual ``torch.cuda.CUDAGraph`` capture of the generic module —
   what a graph-capable generic runner *would* recover, if the tape
   is capturable at all.

The FLA/Triton rows are NOT re-run here (scratch deps — ``fla-core``,
``torchscale`` — are not installed); the retro's numbers are carried
into the document as the external reference, clearly marked.

Usage::

    .venv/bin/python bench/drivers/probes/wholemodel_gap_probe.py
    .venv/bin/python bench/drivers/probes/wholemodel_gap_probe.py --case retnet --t 2048
    .venv/bin/python bench/drivers/probes/wholemodel_gap_probe.py --case mlp
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from typing import cast

import torch
import torch.nn as nn
from mlopt_carriers.om import OM_CHUNKIFY, OM_LAWS, om_kway_laws
from mlopt_core.cost import dag_cost, flops_cost
from mlopt_core.egraph import EGraph
from mlopt_core.ir import IR, Op
from mlopt_core.laws.tensor import (
    ASSOC_MATMUL_REV,
    DIV_ADD_FACTOR,
    FACTOR_MUL,
    FACTOR_MUL_ADD,
    MATMUL_DIV_DISTRIBUTE,
    MUL_DIV_SWAP,
    SPLIT_SCALAR_RULES,
)
from mlopt_orchestrator.optimize import Optimizer, _lower_extracted
from mlopt_torch.adapters import TorchSink, TorchSource
from mlopt_torch.backend import TorchBackend

GAMMA = 0.9

#: The naturality bridges the fold's last hops need (plan 0023) —
#: same set as ``om_decayed_kway_deliver_probe``.
_BRIDGES = [
    MUL_DIV_SWAP,
    MATMUL_DIV_DISTRIBUTE,
    ASSOC_MATMUL_REV,
    FACTOR_MUL,
    FACTOR_MUL_ADD,
    DIV_ADD_FACTOR,
]


# ---------------------------------------------------------------------------
#  Vendored fixtures — mlopt_discovery is mid-refactor this session
#  (lawdata/game -> gamekit/gamedata, staged); the two pieces this probe
#  needs are copied verbatim so it stays runnable while the tree moves.
# ---------------------------------------------------------------------------


class _ChunkedRetention(nn.Module):
    """Verbatim copy of ``mlopt_discovery.zoo._ChunkedRetention``."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.gamma = 0.9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[-2]
        pos = torch.arange(t, dtype=x.dtype, device=x.device)
        decay = torch.tril(
            torch.pow(
                torch.tensor(
                    self.gamma, dtype=x.dtype, device=x.device
                ),
                (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp_min(0),
            )
        )
        s = self.q(x) @ self.k(x).transpose(-1, -2) / (t**0.5)
        return (s * decay) @ self.v(x)


def _time_module(model, x, reps: int) -> float:
    """Median per-call latency (us) — ``mlopt_discovery.play``'s harness."""
    args = x if isinstance(x, (tuple, list)) else (x,)
    dev = args[0].device
    for _ in range(10):
        model(*args)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(*args)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def _saturate(
    t: int, k: int, iters: int, budget: int, chunk: int
) -> dict:
    """``om_decayed_kway_deliver_probe._saturate`` minus the bisect arm."""
    torch.manual_seed(0)
    rules = (
        list(OM_LAWS)
        + list(OM_CHUNKIFY)
        + list(SPLIT_SCALAR_RULES)
        + _BRIDGES
        + om_kway_laws(k)
    )
    budgets = {
        "om_decayed_chunkify_rows": chunk,
        "om_decayed_chunkify_keys": chunk,
        f"om_decayed_kway_blocks_{k}": budget,
    }
    model = _ChunkedRetention(16).double()
    x = torch.randn(2, t, 16, dtype=torch.float64)
    ir, leaves = TorchSource().to_ir(model, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    t0 = time.perf_counter()
    eg.run(
        rules,
        root,
        max_iterations=iters,
        max_nodes=200_000,
        patience=8,
        rule_budgets=budgets,
    )
    secs = time.perf_counter() - t0
    best = eg.extract_best(eg.find(root), flops_cost)
    return {
        "model": model,
        "x": x,
        "ir": ir,
        "leaves": leaves,
        "best": best,
        "cost": dag_cost(best, flops_cost),
        "root_op": best.op,
        "n_enodes": eg.n_enodes,
        "sat_secs": secs,
    }


# ---------------------------------------------------------------------------
#  Measurement helpers
# ---------------------------------------------------------------------------


def _peak(fn) -> float:
    """Peak CUDA allocation (MiB) of one forward under no_grad."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        fn()
    return torch.cuda.max_memory_allocated() / 2**20


def _classify(kernel_name: str) -> str:
    """Bucket a CUDA kernel name into the attribution categories."""
    k = kernel_name.lower()
    if (
        "gemm" in k
        or "cutlass" in k
        or "matmul" in k
        or "dot_kernel" in k
        or "mma" in k
        or "bmm" in k
    ):
        return "gemm"
    if "memcpy" in k or "memset" in k or "copy" in k or "catarray" in k:
        return "copy/mem"
    if "elementwise" in k or "vectorized" in k or "reduce" in k:
        return "pointwise/reduce"
    return "other"


def _kernel_us(fn, reps: int = 30) -> dict:
    """Per-call CUDA-kernel-busy us, kernel count, and category split."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad():
        fn()
        fn()
        torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
    busy = 0.0
    count = 0
    cats: dict[str, float] = {}
    for e in prof.key_averages():
        dt = getattr(e, "self_device_time_total", 0.0) or 0.0
        if dt <= 0 or e.device_type != DeviceType.CUDA:
            continue
        busy += dt
        count += e.count
        cat = _classify(e.key)
        cats[cat] = cats.get(cat, 0.0) + dt
    return {
        "busy_us": busy / reps,
        "n_kernels": count / reps,
        "cats": {k: v / reps for k, v in cats.items()},
    }


def _launch_overhead_us() -> float:
    """CPU-side dispatch cost of one trivial CUDA op on this host.

    A tight loop of tiny ``mul`` launches, no per-call sync: wall per
    launch is the enqueue cost the plan tape pays per leaf.
    """
    import time

    t = torch.zeros(16, device="cuda", dtype=torch.float64)
    torch.cuda.synchronize()
    n = 2000
    for _ in range(200):
        t.mul(1.0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        t.mul(1.0)
    cpu_us = (time.perf_counter() - t0) / n * 1e6
    torch.cuda.synchronize()
    return cpu_us


def _capture_manual(fn, x: torch.Tensor):
    """Capture ``fn(x)`` into a CUDA graph; return a replay callable.

    Distinct from the executor's own ``capture_cuda_graph`` — this
    asks whether *any* fixed-shape torch program (incl. the generic
    IRModule plan tape) can be graph-replayed, i.e. how much of the
    wall clock is pure CPU dispatch.
    """
    static_x = x.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(s):
        for _ in range(3):
            fn(static_x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        static_out = fn(static_x)

    def call(inp: torch.Tensor) -> torch.Tensor:
        static_x.copy_(inp, non_blocking=True)
        g.replay()
        return static_out

    return call


def _count_ops(term) -> int:
    """Count distinct ``Op`` nodes in a term DAG (memoized walk)."""
    seen: set[int] = set()
    stack = [term]
    n = 0
    while stack:
        t = stack.pop()
        if isinstance(t, Op) and id(t) not in seen:
            seen.add(id(t))
            n += 1
            stack.extend(t.args)
    return n


def _measure_arms(arms: dict, x: torch.Tensor, reps: int) -> dict:
    """Time + profile every arm; returns ``name -> row dict``."""
    rows = {}
    for name, fn in arms.items():
        try:
            with torch.no_grad():
                fn()  # warm (compile/JIT/graph lazy-capture)
            mem = _peak(fn)
            ms = _time_module(lambda *_, f=fn: f(), (x,), reps) / 1e3
            prof = _kernel_us(fn)
            rows[name] = {
                "ms": ms,
                "mem": mem,
                "busy_us": prof["busy_us"],
                "n_kernels": prof["n_kernels"],
                "cats": prof["cats"],
            }
        except Exception as e:  # record, don't hide
            rows[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {name:>22}  FAILED: {type(e).__name__}: {e}")
    return rows


def _print_table(rows: dict, ref: str = "spelled-eager") -> None:
    """Print the arm table with the wall-vs-kernel-busy split."""
    hdr = (
        f"{'arm':>22} {'peak MiB':>9} {'med ms':>8} {'vs ref':>7} "
        f"{'kern us':>8} {'n_k':>5} {'dispatch us':>11} "
        f"{'gemm us':>8} {'pw us':>8} {'cpy us':>7} {'oth us':>7}"
    )
    print(hdr)
    ref_ms = rows.get(ref, {}).get("ms", float("nan"))
    for name, r in rows.items():
        if "error" in r:
            continue
        c = r["cats"]
        print(
            f"{name:>22} {r['mem']:>9.2f} {r['ms']:>8.3f} "
            f"{ref_ms / r['ms']:>6.2f}x "
            f"{r['busy_us']:>8.1f} {r['n_kernels']:>5.1f} "
            f"{r['ms'] * 1e3 - r['busy_us']:>11.1f} "
            f"{c.get('gemm', 0.0):>8.1f} "
            f"{c.get('pointwise/reduce', 0.0):>8.1f} "
            f"{c.get('copy/mem', 0.0):>7.1f} {c.get('other', 0.0):>7.1f}"
        )


# ---------------------------------------------------------------------------
#  Case A — the retnet chunked carrier
# ---------------------------------------------------------------------------


def _paper_eqn(x: torch.Tensor, model: _ChunkedRetention, c: int):
    """Paper chunkwise recurrence, exact (no stability rescale), fp64."""
    b, t, d = x.shape
    nc = t // c
    q = model.q(x) / math.sqrt(t)
    k = model.k(x)
    v = model.v(x)
    pos = torch.arange(c, dtype=x.dtype, device=x.device)
    decay = torch.tril(
        torch.tensor(GAMMA, dtype=x.dtype, device=x.device)
        ** (pos[:, None] - pos[None, :]).clamp_min(0)
    )
    cross_decay = torch.tensor(GAMMA**c, dtype=x.dtype, device=x.device)
    q_dec = torch.tensor(
        [GAMMA ** (i + 1) for i in range(c)],
        dtype=x.dtype,
        device=x.device,
    )
    v_dec = torch.tensor(
        [GAMMA ** (c - 1 - i) for i in range(c)],
        dtype=x.dtype,
        device=x.device,
    )
    qc = q.view(b, nc, c, d)
    kc = k.view(b, nc, c, d)
    vc = v.view(b, nc, c, d)
    inner = ((qc @ kc.transpose(-1, -2)) * decay) @ vc
    kv_chunk = kc.transpose(-1, -2) @ (vc * v_dec[None, None, :, None])
    state = torch.zeros(b, d, d, dtype=x.dtype, device=x.device)
    cross = torch.zeros_like(inner)
    for i in range(nc):
        cross[:, i] = (qc[:, i] * q_dec[None, :, None]) @ state
        state = state * cross_decay + kv_chunk[:, i]
    return (inner + cross).reshape(b, t, d)


def _retnet_case(
    t: int, k: int, reps: int, iters: int, budget: int, chunk: int
) -> None:
    """Saturate, lower both ways, and run the attribution ladder."""
    sat = _saturate(t, k, iters, budget, chunk)
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
    fused = _lower_extracted(ex.root, ex, sat["leaves"], sink)
    fused = fused.to("cuda").double()
    x64 = sat["x"].to("cuda")
    c = t // k
    n_ops = _count_ops(sat["best"])

    leaves32 = {n: v.detach().float() for n, v in sat["leaves"].items()}
    carrier32 = cast(nn.Module, sink.lower(ex, params=leaves32)).to(
        "cuda"
    )
    fused32 = _lower_extracted(ex.root, ex, leaves32, sink).to("cuda")
    x32 = x64.float()

    arms: dict = {
        "spelled-eager": lambda: base(x64),
        "carrier(gen)64": lambda: carrier(x64),
        "carrier(gen)32": lambda: carrier32(x32),
        "fused(exec)64": lambda: fused(x64),
        "fused(exec)32": lambda: fused32(x32),
        "paper-eqn64": lambda: _paper_eqn(x64, base, c),
    }

    # Lane 2a — the shipped cheap lane on the generic lowering:
    # CudaGraphRunner only claims batched executors; record the
    # degrade honestly, then measure a manual capture anyway.
    from mlopt_cuda import CudaGraphRunner

    stats_generic: dict = {}
    CudaGraphRunner().apply(carrier, (x64,), stats_generic)
    print(
        f"  CudaGraphRunner on generic carrier: "
        f"cuda_graph={stats_generic.get('cuda_graph')} "
        f"failure={stats_generic.get('cuda_graph_failure')}"
    )
    try:
        carrier_g = _capture_manual(lambda t_: carrier(t_), x64)
        arms["carrier64+graph(man)"] = lambda: carrier_g(x64)
    except Exception as e:
        print(f"  manual capture of generic carrier FAILED: {e}")
    try:
        carrier_g32 = _capture_manual(lambda t_: carrier32(t_), x32)
        arms["carrier32+graph(man)"] = lambda: carrier_g32(x32)
    except Exception as e:
        print(f"  manual capture of carrier32 FAILED: {e}")

    # Lane 2b — the same lane on the executor-routed module (its own
    # capture path, exactly what CudaGraphRunner calls).
    if hasattr(fused, "capture_cuda_graph"):
        fusedg = _lower_extracted(ex.root, ex, sat["leaves"], sink)
        fusedg = fusedg.to("cuda").double()

        def _fg():
            if not fusedg.is_graph_captured:
                fusedg.capture_cuda_graph(x64)
            return fusedg(x64)

        arms["fused64+graph"] = _fg
        fusedg32 = _lower_extracted(ex.root, ex, leaves32, sink)
        fusedg32 = fusedg32.to("cuda")

        def _fg32():
            if not fusedg32.is_graph_captured:
                fusedg32.capture_cuda_graph(x32)
            return fusedg32(x32)

        arms["fused32+graph"] = _fg32
    else:
        print("  executor-routed module has no capture_cuda_graph")

    # The shipped runner on the executor module (records the stats).
    stats_exec: dict = {}
    fused_r = _lower_extracted(ex.root, ex, sat["leaves"], sink)
    fused_r = fused_r.to("cuda").double()
    CudaGraphRunner().apply(fused_r, (x64,), stats_exec)
    print(
        f"  CudaGraphRunner on executor module: "
        f"cuda_graph={stats_exec.get('cuda_graph')}"
    )
    if stats_exec.get("cuda_graph"):
        arms["fused64+graph(runner)"] = lambda: fused_r(x64)

    print(
        f"\n=== retnet T={t} K={k} C={c} === "
        f"(extracted cost {sat['cost']:,.0f}, root={sat['root_op']}, "
        f"distinct ops {n_ops}, sat {sat['sat_secs']:.1f}s)"
    )
    rep = sink.verify(base, carrier, (x64,))
    rep_f = sink.verify(base, fused, (x64,))
    print(
        f"  verify: carrier max_abs {rep.max_abs:.2e} "
        f"| fused max_abs {rep_f.max_abs:.2e} "
        f"| executor batched={getattr(fused, 'is_batched', False)}"
    )
    rows = _measure_arms(arms, x64, reps)
    _print_table(rows)

    # Manual capture of paper-eqn — isolates the Python chunk-loop
    # cost from the batched-vs-loop schedule difference.
    del sat
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
#  Case B — the chainfuse MLP (fold + per-leaf pointwise tail)
# ---------------------------------------------------------------------------


class _ChainFuse(torch.nn.Module):
    """The gen-domain case: mm chain + 6-op pointwise tail."""

    def __init__(self, d: int = 512) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.a = torch.nn.Parameter(
            torch.randn(d, d, generator=g) / d**0.5
        )
        self.b = torch.nn.Parameter(
            torch.randn(d, d, generator=g) / d**0.5
        )
        self.c = torch.nn.Parameter(
            torch.randn(d, d, generator=g) / d**0.5
        )

    def forward(self, x):
        y = x @ self.a @ self.b @ self.c
        return (
            y * torch.relu(y)
            - torch.nn.functional.softplus(y)
            + torch.tanh(y * y)
            * torch.nn.functional.softplus(y * torch.sigmoid(y))
        )


class _HandFolded(torch.nn.Module):
    """The same program a fused codegen would emit: 1 GEMM + tail."""

    def __init__(self, ref: _ChainFuse) -> None:
        super().__init__()
        with torch.no_grad():
            self.w = torch.nn.Parameter(
                (ref.a @ ref.b @ ref.c).detach().clone()
            )

    def forward(self, x):
        y = x @ self.w
        return (
            y * torch.relu(y)
            - torch.nn.functional.softplus(y)
            + torch.tanh(y * y)
            * torch.nn.functional.softplus(y * torch.sigmoid(y))
        )


def _mlp_case(reps: int) -> None:
    """Deliver chainfuse laws-only and attribute the residual gap."""
    model = _ChainFuse().to("cuda").double().eval()
    x64 = torch.randn(256, 512, device="cuda", dtype=torch.float64)
    x32 = x64.float()

    opt = Optimizer(backend=TorchBackend())
    res = opt.search(model, x64, max_iterations=8)
    low = opt.lower(res, x64, verify=True)
    delivered = cast(nn.Module, low.module)
    n_ops = _count_ops(res.term)
    ex_ir = IR(
        root=res.term,
        inputs=list(res.ir.inputs),
        input_names=set(res.ir.input_names),
        params=dict(res.ir.params),
    )
    # The genuinely generic lowering (no executor routing): a plain
    # IRModule plan tape — the "per-leaf generic torch ops" artifact.
    generic = (
        cast(
            nn.Module,
            TorchSink().lower(ex_ir, params=dict(res.param_values)),
        )
        .to("cuda")
        .double()
    )

    model32 = copy.deepcopy(model).float()
    res32 = opt.search(model32, x32, max_iterations=8)
    del32 = cast(nn.Module, opt.lower(res32, x32, verify=True).module)

    hand = _HandFolded(model).to("cuda").double()
    hand32 = _HandFolded(model32).to("cuda")
    comp = torch.compile(model)
    comp_del = torch.compile(delivered)
    comp_hand = torch.compile(hand)
    comp_hand32 = torch.compile(hand32)

    arms: dict = {
        "spelled-eager": lambda: model(x64),
        "spelled-eager32": lambda: model32(x32),
        "spelled-compiled": lambda: comp(x64),
        "generic64(IRModule)": lambda: generic(x64),
        "delivered64(routed)": lambda: delivered(x64),
        "delivered(gen)32": lambda: del32(x32),
        "delivered+compile": lambda: comp_del(x64),
        "handfold-eager": lambda: hand(x64),
        "handfold+compile": lambda: comp_hand(x64),
        "handfold32+compile": lambda: comp_hand32(x32),
    }

    # Lane 2a — the shipped cheap lane on a generic IRModule:
    # degrade-quietly or capture?  Recorded honestly.
    from mlopt_cuda import CudaGraphRunner

    stats_g: dict = {}
    CudaGraphRunner().apply(generic, (x64,), stats_g)
    print(
        f"  CudaGraphRunner on generic IRModule: "
        f"cuda_graph={stats_g.get('cuda_graph')} "
        f"failure={stats_g.get('cuda_graph_failure')}"
    )
    # ...and on the delivery-routed module (a fresh lowering — capture
    # mutates the module into replay mode).
    del_r = opt.lower(res, x64, runner=CudaGraphRunner(), verify=False)
    print(
        f"  lower(runner=CudaGraphRunner): "
        f"stats cuda_graph={del_r.stats.get('cuda_graph')} "
        f"runner={del_r.stats.get('runner')}"
    )
    if del_r.stats.get("cuda_graph"):
        mod_r = cast(nn.Module, del_r.module)
        arms["delivered64+graph(run)"] = lambda: mod_r(x64)
    # Lane 2b — manual capture of the plain IRModule: what a
    # graph-capable generic runner would recover.
    try:
        gen_g = _capture_manual(lambda t_: generic(t_), x64)
        arms["generic64+graph(man)"] = lambda: gen_g(x64)
    except Exception as e:
        print(f"  manual capture of generic IRModule FAILED: {e}")

    print(
        f"\n=== chainfuse (256,512) d=512 === "
        f"delivered root={getattr(res.term, 'op', res.term)} "
        f"distinct ops={n_ops} "
        f"lowering={low.stats.get('lowering')} "
        f"verified={getattr(low.verified, 'max_abs', None)}"
    )
    rows = _measure_arms(arms, x64, reps)
    _print_table(rows)


def main() -> None:
    """Run the attribution probes."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case", choices=["retnet", "mlp", "all"], default="all"
    )
    ap.add_argument("--t", type=int, nargs="*", default=[1024, 2048])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--iters", type=int, default=14)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=60)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    print(f"launch overhead: {_launch_overhead_us():.2f} us/op")
    if a.case in ("retnet", "all"):
        for t in a.t:
            _retnet_case(t, a.k, a.reps, a.iters, a.budget, a.chunk)
    if a.case in ("mlp", "all"):
        _mlp_case(a.reps)


if __name__ == "__main__":
    main()

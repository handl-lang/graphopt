"""BiRefNet parity probe — does ``torch.compile(delivered)`` recover parity.

Context: the BiRefNet trial (commit ``34607e0``) delivered a module
that timed at ~0.90x vs eager and ~0.80x vs ``torch.compile`` on a
reconstructed Swin stage.  The extraction was FLOP-neutral (only a
couple of laws fired), so the delivered artifact was the *same* graph
re-lowered through the plan-tape interpreter — the loss looked like
lowering overhead, not semantics.  This probe tests the hypothesis:

    compile(delivered) ~= compile(source) ?

i.e. whether inductor fuses the re-lowered tape back to the same
kernel schedule it produces for the spelled module.

Block: two timm ``SwinTransformerBlock``s shaped like BiRefNet's
encoder stage 1 (Swin-v1-L: dim 192, heads 6, window 7, alternating
shift 0/3 at 56x56 tokens — a 224^2 image through patch size 4).
~890K params, matching the earlier trial's ~870K reconstruction.
This uses the *real* timm block (``timm==1.0.30``), not a hand-built
stand-in.

Protocol (``lab/probes/om_decayed_kway_deliver_probe.py`` harness):

1. Export via ``TorchSource().to_ir``; saturate with the composed
   pipeline default (``mlopt_orchestrator.default_rules()`` — core
   ``laws.DEFAULT`` + the carrier preset), pipeline-style bounded
   saturation (``symmetry_budget`` on EXPANSIVE-tagged rules).
2. Extract the cheapest term under ``backend_cost(flops_cost,
   sink.supported_ops)``; lower via ``TorchSink().lower``; verify
   against the source (``sink.verify``, ``max_abs`` reported).
3. Time the 2x2: source eager / source compiled / delivered eager /
   delivered compiled — ``mlopt_discovery.play._time_module``
   (10 warm-ups, median of *reps*), plus steady-state peak CUDA
   memory (peak counters reset *after* compilation, so compile-time
   allocations do not pollute the steady-state number).

Usage::

    .venv/bin/python lab/probes/birefnet_parity_probe.py
    .venv/bin/python lab/probes/birefnet_parity_probe.py --batch 2 --reps 200
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
from mlopt_core.cost import backend_cost, dag_cost, flops_cost
from mlopt_core.egraph import EGraph
from mlopt_core.ir import IR, Op
from mlopt_core.laws import tags as _law_tags
from mlopt_discovery.play import _time_module
from mlopt_orchestrator import default_rules
from mlopt_torch.adapters import TorchSink, TorchSource

try:
    from timm.models.swin_transformer import SwinTransformerBlock
except ImportError:  # pragma: no cover - environment guard
    SwinTransformerBlock = None  # type: ignore[assignment]


class SwinStage(nn.Module):
    """A BiRefNet-stage-1-shaped pair of timm Swin blocks.

    Two blocks, alternating the window shift (0, window//2) exactly
    as a Swin stage does; input/output ``(B, H, W, C)`` — timm's
    block convention.
    """

    def __init__(
        self,
        dim: int,
        res: int,
        heads: int,
        window: int,
        depth: int,
    ) -> None:
        """Build *depth* timm blocks with alternating window shift."""
        super().__init__()
        self.blocks = nn.ModuleList(
            SwinTransformerBlock(
                dim=dim,
                input_resolution=(res, res),
                num_heads=heads,
                window_size=window,
                shift_size=0 if i % 2 == 0 else window // 2,
            )
            for i in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the alternating-shift block pair."""
        for blk in self.blocks:
            x = blk(x)
        return x


def _ops_of(term: object) -> set:
    """Every op name reachable in *term* (args-first)."""
    acc: set = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Op):
            acc.add(t.op)
            stack.extend(t.args)
    return acc


def _op_counts(term: object) -> dict:
    """Op-name histogram of *term* (args-first walk)."""
    acc: dict[str, int] = {}
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Op):
            acc[t.op] = acc.get(t.op, 0) + 1
            stack.extend(t.args)
    return acc


def _saturate(
    model: nn.Module,
    x: torch.Tensor,
    iters: int,
    max_nodes: int,
    symmetry_budget: int,
    rules_mode: str,
) -> dict:
    """Export + saturate + extract; return the e-graph facts.

    ``rules_mode='none'`` skips saturation entirely and extracts the
    source term as-is — reproducing the FLOP-*neutral* delivered
    artifact of the earlier trial (the same graph re-lowered through
    the plan-tape interpreter), which is the regime the parity
    hypothesis is about.
    """
    ir, leaves = TorchSource().to_ir(model, x)
    rules = default_rules() if rules_mode == "default" else []
    eg = EGraph()
    root = eg.add_term(ir.root)
    rule_budgets = (
        {
            r.name: symmetry_budget
            for r in rules.tagged(_law_tags.EXPANSIVE)
        }
        if rules_mode == "default"
        else None
    )
    t0 = time.perf_counter()
    stats = eg.run(
        rules,
        root,
        max_iterations=iters,
        max_nodes=max_nodes,
        patience=3,
        rule_budgets=rule_budgets,
    )
    secs = time.perf_counter() - t0
    sink = TorchSink()
    cost = backend_cost(flops_cost, sink.supported_ops)
    best = eg.extract_best(eg.find(root), cost)
    fires = {k: v for k, v in eg.rule_fires.items() if v}
    return {
        "ir": ir,
        "leaves": leaves,
        "best": best,
        "src_cost": dag_cost(ir.root, flops_cost),
        "best_cost": dag_cost(best, flops_cost),
        "n_enodes": eg.n_enodes,
        "sat_secs": secs,
        "stop": stats.get("stop"),
        "fires": fires,
        "sink": sink,
    }


def _bench(model: nn.Module, x: torch.Tensor, reps: int) -> dict:
    """Median latency + steady-state peak CUDA memory for *model*.

    Compile/warm-up happens before the peak counters are reset, so the
    reported peak is the *steady-state* footprint, not inductor's
    compile-time workspace.
    """
    for _ in range(3):
        model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    us = _time_module(model, (x,), reps)
    peak = torch.cuda.max_memory_allocated() if x.is_cuda else 0
    return {"us": us, "peak_mb": peak / 2**20}


def main(argv: list[str] | None = None) -> int:
    """Run the 2x2 eager/compiled timing matrix and print the verdict."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--res", type=int, default=56)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--max-nodes", type=int, default=200_000)
    ap.add_argument("--symmetry-budget", type=int, default=2048)
    ap.add_argument(
        "--rules",
        choices=("default", "none"),
        default="default",
        help=(
            "'default' = the composed pipeline rule set; 'none' = no "
            "rewrites, the FLOP-neutral re-lowering of the earlier "
            "trial (the arm the parity hypothesis is about)"
        ),
    )
    ap.add_argument(
        "--cpu", action="store_true", help="force CPU timing"
    )
    args = ap.parse_args(argv)

    if SwinTransformerBlock is None:
        print("timm is not installed — pip install timm")
        return 1
    device = (
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    torch.manual_seed(0)
    model = (
        SwinStage(
            args.dim, args.res, args.heads, args.window, args.depth
        )
        .eval()
        .to(device)
    )
    x = torch.randn(
        args.batch, args.res, args.res, args.dim, device=device
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"block: timm SwinTransformerBlock x{args.depth} "
        f"(dim={args.dim} res={args.res} heads={args.heads} "
        f"win={args.window}) params={n_params / 1e3:.0f}K "
        f"batch={args.batch} device={device} rules={args.rules}",
        flush=True,
    )

    sat = _saturate(
        model,
        x,
        args.iters,
        args.max_nodes,
        args.symmetry_budget,
        args.rules,
    )
    print(
        f"saturation: {sat['n_enodes']} enodes in {sat['sat_secs']:.1f}s "
        f"(stop={sat['stop']})",
        flush=True,
    )
    print(f"rule fires: {sat['fires'] or '{}'}", flush=True)
    ratio = (
        sat["best_cost"] / sat["src_cost"] if sat["src_cost"] else 1.0
    )
    print(
        f"cost: source={sat['src_cost']:,.0f} extracted="
        f"{sat['best_cost']:,.0f} (x{ratio:.3f}) root="
        f"{sat['best'].op}",
        flush=True,
    )
    src_ops = _op_counts(sat["ir"].root)
    del_ops = _op_counts(sat["best"])
    diff = {
        k: del_ops.get(k, 0) - src_ops.get(k, 0)
        for k in set(src_ops) | set(del_ops)
    }
    diff = {k: v for k, v in diff.items() if v}
    print(f"op-count delta (delivered - source): {diff}", flush=True)

    sink: TorchSink = sat["sink"]
    ir = sat["ir"]
    missing = _ops_of(sat["best"]) - sink.supported_ops
    if missing:
        print(f"LOWER FAILED — unsupported ops: {sorted(missing)}")
        return 1
    ex_ir = IR(
        root=sat["best"],
        inputs=list(ir.inputs),
        input_names=set(ir.input_names),
        params=dict(ir.params),
    )
    delivered = sink.lower(ex_ir, params=sat["leaves"]).to(device)
    rep = sink.verify(model, delivered, (x,))
    print(
        f"verify: passed={rep.passed} max_abs={rep.max_abs:.3e}",
        flush=True,
    )

    cfg = torch._dynamo.config
    cfg.recompile_limit = max(int(cfg.recompile_limit), 256)

    r_src_eager = _bench(model, x, args.reps)
    t0 = time.perf_counter()
    compiled_src = torch.compile(model)
    r_src_comp = _bench(compiled_src, x, args.reps)
    print(f"  source compile+time: {time.perf_counter() - t0:.1f}s")
    r_del_eager = _bench(delivered, x, args.reps)
    t0 = time.perf_counter()
    compiled_del = torch.compile(delivered)
    r_del_comp = _bench(compiled_del, x, args.reps)
    print(f"  delivered compile+time: {time.perf_counter() - t0:.1f}s")

    print(
        f"\n{'arm':>22} {'us':>10} {'peak_MB':>10} {'vs_src_eager':>13}",
        flush=True,
    )
    base = r_src_eager["us"]
    rows = [
        ("source eager", r_src_eager),
        ("source compile", r_src_comp),
        ("delivered eager", r_del_eager),
        ("delivered compile", r_del_comp),
    ]
    for name, r in rows:
        print(
            f"{name:>22} {r['us']:>10.1f} {r['peak_mb']:>10.1f} "
            f"{base / r['us']:>12.3f}x",
            flush=True,
        )
    parity = r_del_comp["us"] / r_src_comp["us"]
    print(
        f"\ncompiled(delivered) / compiled(source) = {parity:.3f}x "
        f"({'PARITY' if abs(parity - 1) < 0.05 else 'NOT parity'})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Delivered wall-time of the shipped ``prim:synop*`` folds on bound modules.

The pay leg for the shipped primitive folds
(``mlopt_discovery.shipped_primitives`` — the ``zoo-prim`` /
``prim-bootstrap`` mints, core-shipped default-on) was measured in the
pipeline's *own* cost model: a ``synop_*`` call prices as one dispatch
against the manual subgraph's N, and extraction reports 0.5-0.8x
modeled costs.  Nothing so far has timed the *delivered* modules.

What the fold actually delivers at runtime is narrower than the
"fused kernel" framing suggests: ``shipped_primitives.register()``
seats one ambient binding per op whose body is a Python spec-walk —
``_eval_spec`` over the definitional premise's RHS, calling the same
ambient torch functions the spelled subgraph would.  Inside an
``IRModule`` the difference between the two spellings is dispatch
shape only: N plan-tape steps versus 1 tape step + a recursive
spec-eval of N nodes.  There is no fused ATen kernel — the win, if
any, is Python-dispatch elimination, and the spec-walk's per-call
overhead (zip/dict env build, ``_spec_bind`` attr threading, ambient
table lookups) can exceed the tape dispatches it saves.

Honesty contract
----------------

Four eager arms decompose the comparison; the verdict never rests on
the raw-vs-lowered delta alone:

* ``spelled`` — the manual ``nn.Module``, fp64 CPU, eager.  The task
  baseline: what a practitioner ships unoptimized.
* ``tape`` — ``export_to_ir`` + ``ir_to_torch_module`` with no search
  at all: the identical graph through the same ``IRModule`` plan-tape
  interpreter.  ``tape/spelled`` isolates the interpreter overhead
  every lowered arm pays.
* ``noprim`` — shipped ``default_rules()`` minus every
  ``prim:synop*`` fold, then search + lower.  Same machinery, same
  non-prim rewrites; ``synop/noprim`` is the fold family's marginal
  inside the executor — the number that answers "does the fold pay".
* ``synop`` — shipped ``default_rules()`` + ``sp.register()``, then
  search + lower(verify=True): the delivered module.

All four then get ``torch.compile`` twins (``*_compiled``) — testing
whether Inductor already recovers whatever the folds claim to win: if
``synop_compiled`` converges to ``spelled_compiled`` the dispatch
story is compile's, not the fold's.

Verdict rule (per module x size): the **marginal** is
``synop_ms / noprim_ms`` — <=0.95 "delivers", >=1.05 "loses", else
"tie".  The **end-to-end** column ``synop_ms / spelled_ms`` is what a
caller of the shipped pipeline actually receives.

Run::

    .venv/bin/python bench/drivers/probes/prim_delivery_probe.py
    .venv/bin/python bench/drivers/probes/prim_delivery_probe.py \
        --sizes small --no-compile --json /tmp/prim.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# Keep the inductor compile fan-out small on shared boxes (the suite's
# parallel warning is measured; a probe needn't spawn 16 workers).
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

import mlopt_carriers  # noqa: F401 — carrier bindings in the ambient table
import torch
from mlopt_core.ir import Op
from mlopt_core.laws import minted
from mlopt_core.laws.ruleset import RuleSet
from mlopt_orchestrator.optimize import (
    default_rules,
    lower,
    search,
)
from mlopt_torch.adapters import TorchSink, TorchSource
from mlopt_torch.torch_bridge import (
    export_to_ir,
    ir_to_torch_module,
)

# The tree can sit mid-pin-bump: domain-side ``shipped_primitives``
# already reads ``minted.SYNOP3_SHIPPED`` while the installed core pin
# still predates the depth-3 promotion.  Seat an empty tier-3 ledger
# against the old pin so ``register()`` covers exactly the shipped
# tiers this core carries; post-bump the attribute exists and the
# shim is a no-op.
if not hasattr(minted, "SYNOP3_SHIPPED"):
    minted.SYNOP3_SHIPPED = frozenset()

from mlopt_discovery import shipped_primitives as sp
from mlopt_discovery import zoo as zoo_mod

#: The bound modules — the records' ``mods`` ledger (every SSM-cluster
#: module binds the mint's recurring-scan folds; ``synop_bb8c54c201`` —
#: ``add(mul(a,b), mul(c,d))`` — binds all fourteen mint modules
#: including ``HighwayGate``/``RotaryPair``/``GRUFusedGate``).
BOUND_MODULES: tuple[str, ...] = (
    "GatedDeltaRule",
    "GatedLinearAttention",
    "MambaSelectiveScan",
    "RealGatedLRU",
    "HighwayGate",
    "RotaryPair",
    "GRUFusedGate",
)

#: The size ladder.  ``small`` is the mint's own probe site
#: (``zoo()`` defaults — tiny, dispatch-bound); ``large`` checks the
#: crossover where pointwise work starts paying for dispatches.
SIZES: dict[str, dict[str, int]] = {
    "small": {"d": 16, "t": 8, "b": 2},
    "large": {"d": 64, "t": 32, "b": 8},
}

#: Search budget — mirrors the shipped-pipeline pins
#: (``test_shipped_primitives``).
_MAX_ITERS = 20

#: Timing budget — enough interleaved samples to matter at sub-ms
#: scales (``iters`` is per-arm samples; arm order rotates each
#: iteration).
_WARMUP = 60
_ITERS = 400

#: Verdict band on the synop/noprim marginal.
_TIE_BAND = 0.05


# ---------------------------------------------------------------------------
#  Terms / rules helpers
# ---------------------------------------------------------------------------


def _ops_of(term: Any) -> dict[str, int]:
    """Op-name histogram of an extracted term."""
    out: dict[str, int] = {}

    def go(t: Any) -> None:
        if isinstance(t, Op):
            out[t.op] = out.get(t.op, 0) + 1
            for a in t.args:
                go(a)

    go(term)
    return out


def _noprim_rules() -> RuleSet:
    """Shipped ``default_rules()`` minus the whole ``prim:synop*`` family."""
    full = default_rules()
    return replace(
        full,
        name=f"{full.name}-noprim",
        rules=tuple(
            r for r in full.rules if not r.name.startswith("prim:synop")
        ),
        priorities={
            k: v
            for k, v in full.priorities.items()
            if not k.startswith("prim:synop")
        },
    )


# ---------------------------------------------------------------------------
#  Timing
# ---------------------------------------------------------------------------


def _bench(
    arms: dict[str, Any],
    *,
    warmup: int = _WARMUP,
    iters: int = _ITERS,
) -> dict[str, dict[str, Any]]:
    """Warm every arm, then interleave per-iteration samples in ms.

    Arms are timed **interleaved** — each iteration calls every arm
    once, order rotating — so an external load spike (this box runs
    shared workloads) lands on all arms inside the same window and the
    *relative* comparison stays fair even when the absolute level is
    contended.  ``median`` is the sample median; ``min`` is the
    least-contended iteration — the dispatch-floor estimate.
    """
    for fn in arms.values():
        for _ in range(warmup):
            fn()
    names = list(arms)
    samples: dict[str, list[float]] = {n: [] for n in names}
    for it in range(iters):
        shift = it % len(names)
        order = names[shift:] + names[:shift]
        for name in order:
            t0 = time.perf_counter()
            arms[name]()
            samples[name].append((time.perf_counter() - t0) * 1e3)
    return {
        n: {
            "median": statistics.median(v),
            "min": min(v),
            "p25": statistics.quantiles(v, n=4)[0],
            "p75": statistics.quantiles(v, n=4)[2],
            "samples": len(v),
        }
        for n, v in samples.items()
    }


# ---------------------------------------------------------------------------
#  Case machinery
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    """Everything measured for one ``(module, size)``."""

    module: str
    size: str
    feed_shapes: list[tuple[int, ...]] = field(default_factory=list)
    search_s: float = 0.0
    synops: list[str] = field(default_factory=list)
    n_ops_synop: int = 0
    n_ops_noprim: int = 0
    n_ops_tape: int = 0
    prim_fires: dict[str, int] = field(default_factory=dict)
    verified: dict[str, str] = field(default_factory=dict)
    allclose: dict[str, bool] = field(default_factory=dict)
    executors: dict[str, str] = field(default_factory=dict)
    times: dict[str, dict[str, Any]] = field(default_factory=dict)
    compile_errors: dict[str, str] = field(default_factory=dict)
    warmup: int = _WARMUP
    iters: int = _ITERS
    note: str = ""


def _workload(name: str, zkws: dict[str, int]) -> Any:
    """Resolve *name* in the zoo registry built at size *zkws*."""
    return {w.name: w for w in zoo_mod.zoo(**zkws)}[name]


def _feed_shapes(feed: Any) -> list[tuple[int, ...]]:
    """Shape summary of a feed (tensor or tensor tuple)."""
    feeds = feed if isinstance(feed, tuple) else (feed,)
    return [tuple(f.shape) for f in feeds]


def measure_module(
    name: str,
    size: str = "small",
    *,
    compile_arms: bool = True,
    warmup: int = _WARMUP,
    iters: int = _ITERS,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> CaseResult:
    """Build the four arms for *name* at *size* and time them.

    *device* / *dtype* default to the historical fp64-on-CPU contract (the
    verify dtype).  They move the feeds and the exported parameters, but
    **not yet the plan executor**: ``torch_bridge.Plan.run_xs`` allocates
    its intermediates from a CPU ``like`` tensor, so a CUDA run raises
    "Expected all tensors to be on the same device".  That is the open gap
    (`docs/reports/prim_kernel.md`) — the delivery path is CPU-pinned, so a
    generated kernel cannot be measured end to end yet.
    """
    zkws = SIZES[size]
    sp.register()
    w = _workload(name, zkws)
    torch.manual_seed(0)
    model, feed = w.build()
    feeds = feed if isinstance(feed, tuple) else (feed,)
    feeds = tuple(f.to(device=device, dtype=dtype) for f in feeds)
    spelled = model.eval().to(device=device, dtype=dtype)

    row = CaseResult(
        module=name,
        size=size,
        feed_shapes=_feed_shapes(feed),
        warmup=warmup,
        iters=iters,
    )

    # -- arm 1: identity tape (export + lower, no search) ----------------
    ir, params = export_to_ir(spelled, feeds)
    # parameters leave the exporter on CPU — move them with the feeds so
    # the tape arm (and every lowered arm) runs on one device/dtype
    params = {
        k: (v.to(device=device, dtype=dtype) if hasattr(v, "to") else v)
        for k, v in params.items()
    }
    tape = ir_to_torch_module(ir, param_values=params)
    row.n_ops_tape = sum(_ops_of(ir.root).values())

    # -- arm 2: shipped DEFAULT minus the prim folds ---------------------
    res_np = search(
        spelled,
        feeds,
        source=TorchSource(),
        rules=_noprim_rules(),
        max_iterations=_MAX_ITERS,
    )
    lr_np = lower(res_np, feeds, sink=TorchSink(), verify=True)
    row.n_ops_noprim = sum(_ops_of(res_np.term).values())
    row.verified["noprim"] = (
        "pass" if (lr_np.verified and lr_np.verified.passed) else "FAIL"
    )

    # -- arm 3: shipped DEFAULT — the delivered module -------------------
    t0 = time.perf_counter()
    res_syn = search(
        spelled,
        feeds,
        source=TorchSource(),
        max_iterations=_MAX_ITERS,
    )
    row.search_s = time.perf_counter() - t0
    lr_syn = lower(res_syn, feeds, sink=TorchSink(), verify=True)
    ops = _ops_of(res_syn.term)
    row.n_ops_synop = sum(ops.values())
    row.synops = sorted(o for o in ops if "synop" in o)
    row.prim_fires = {
        k: v
        for k, v in res_syn.stats.get("rule_fires", {}).items()
        if k.startswith("prim:synop")
    }
    row.verified["synop"] = (
        "pass"
        if (lr_syn.verified and lr_syn.verified.passed)
        else "FAIL"
    )
    if not row.synops:
        row.note += " extracted term carries no synop — folds fired but lost the tie. "

    mods: dict[str, Any] = {
        "spelled": spelled,
        "tape": tape,
        "noprim": lr_np.module,
        "synop": lr_syn.module,
    }
    row.executors = {k: type(m).__name__ for k, m in mods.items()}
    if row.executors["synop"] != row.executors["noprim"]:
        row.note += (
            f" executor confound: synop routes to "
            f"{row.executors['synop']}, noprim to "
            f"{row.executors['noprim']} — the marginal compares "
            "different executors, not just the fold. "
        )
    arms: dict[str, Any] = {
        name: (lambda m=m: m(*feeds)) for name, m in mods.items()
    }

    # -- reference output + per-arm equality ----------------------------
    with torch.inference_mode():
        ref = spelled(*feeds)
        for arm, fn in arms.items():
            got = fn()
            row.allclose[arm] = bool(
                torch.allclose(ref, got, rtol=1e-4, atol=1e-6)
            )

    if compile_arms:
        # Compile the modules, not shared lambda code objects: every
        # distinct module instance recompiles under its own guards, so
        # the per-code-object recompile limit must cover the whole run
        # (7 modules x 2 sizes x 4 arms).  Raise it up front — hitting
        # the default (8) silently demotes later arms to eager.
        torch._dynamo.config.recompile_limit = max(
            256, torch._dynamo.config.recompile_limit
        )
        for arm, m in mods.items():
            try:
                compiled = torch.compile(m)
                with torch.inference_mode():
                    compiled(*feeds)  # trigger the compile off-timer
                arms[f"{arm}_compiled"] = lambda mc=compiled: mc(*feeds)
            except Exception as e:  # honest per-arm failure
                row.compile_errors[arm] = f"{type(e).__name__}: {e}"

    with torch.inference_mode():
        row.times = _bench(arms, warmup=warmup, iters=iters)
    return row


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------


def _ratio(num: float, den: float) -> float | None:
    """*num/den*, or ``None`` when either side failed to time."""
    return num / den if num and den else None


def _verdict(row: CaseResult) -> str:
    """Classify the fold family's marginal: delivers / tie / loses."""
    t = row.times
    marg = _ratio(
        t.get("synop", {}).get("median", 0),
        t.get("noprim", {}).get("median", 0),
    )
    end = _ratio(
        t.get("synop", {}).get("median", 0),
        t.get("spelled", {}).get("median", 0),
    )
    if marg is None:
        return "untimed"
    if marg <= 1 - _TIE_BAND:
        head = "delivers"
    elif marg >= 1 + _TIE_BAND:
        head = "loses"
    else:
        head = "tie"
    if end is not None and end >= 1 + _TIE_BAND and head != "loses":
        head += " (still slower than spelled e2e)"
    elif end is not None and end <= 1 - _TIE_BAND and head == "tie":
        head += " (faster than spelled e2e anyway)"
    return head


def _ms(entry: dict[str, Any]) -> str:
    """Render one timing entry as ``median [p25-p75]`` ms."""
    if "error" in entry:
        return f"ERR {entry['error'][:24]}"
    return (
        f"{entry['median']:.4f} [{entry['p25']:.4f}-{entry['p75']:.4f}]"
    )


def _report(rows: list[CaseResult]) -> str:
    """Render the per-module table."""
    budget = (
        f"warmup {rows[0].warmup}  iters {rows[0].iters}"
        if rows
        else "no cases"
    )
    lines = [
        "== prim_delivery — do the shipped synop_* folds pay on real wall? ==",
        f"   device: CPU  dtype: fp64  {budget}  "
        "interleaved samples, median [p25-p75] ms",
        "",
    ]
    for r in rows:
        t = r.times
        marg = _ratio(
            t.get("synop", {}).get("median", 0),
            t.get("noprim", {}).get("median", 0),
        )
        end = _ratio(
            t.get("synop", {}).get("median", 0),
            t.get("spelled", {}).get("median", 0),
        )
        tape_over = _ratio(
            t.get("tape", {}).get("median", 0),
            t.get("spelled", {}).get("median", 0),
        )
        lines.append(
            f"-- {r.module} @ {r.size}  feeds={r.feed_shapes} --"
        )
        lines.append(
            f"   synops in term: {r.synops or '-'}   "
            f"ops tape/noprim/synop: "
            f"{r.n_ops_tape}/{r.n_ops_noprim}/{r.n_ops_synop}   "
            f"verify noprim/synop: "
            f"{r.verified.get('noprim')}/{r.verified.get('synop')}   "
            f"exec: {r.executors.get('synop', '-')}"
            + (
                f" (noprim: {r.executors['noprim']})"
                if r.executors.get("noprim") != r.executors.get("synop")
                else ""
            )
        )
        for arm, entry in t.items():
            lines.append(f"   {arm:<18} {_ms(entry):>24}")
        lines.append(
            f"   ratios: tape/spelled {tape_over and f'{tape_over:.3f}x'}   "
            f"synop/noprim {marg and f'{marg:.3f}x'} (fold marginal)   "
            f"synop/spelled {end and f'{end:.3f}x'} (e2e)   "
            f"verdict: {_verdict(r)}"
        )
        for arm, ok in r.allclose.items():
            if not ok:
                lines.append(f"   !! {arm} output differs from spelled")
        for arm, err in r.compile_errors.items():
            lines.append(f"   !! {arm} compile failed: {err}")
        if r.note:
            lines.append(f"   note:{r.note}")
        lines.append("")
    return "\n".join(lines)


def _dump_json(path: str, rows: list[CaseResult]) -> None:
    """Write the machine-readable result."""
    payload = [
        {
            "module": r.module,
            "size": r.size,
            "feed_shapes": r.feed_shapes,
            "search_s": r.search_s,
            "synops": r.synops,
            "n_ops": {
                "tape": r.n_ops_tape,
                "noprim": r.n_ops_noprim,
                "synop": r.n_ops_synop,
            },
            "prim_fires": r.prim_fires,
            "verified": r.verified,
            "allclose": r.allclose,
            "executors": r.executors,
            "times": r.times,
            "compile_errors": r.compile_errors,
            "verdict": _verdict(r),
            "note": r.note,
        }
        for r in rows
    ]
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def run(
    *,
    modules: tuple[str, ...] = BOUND_MODULES,
    sizes: tuple[str, ...] = tuple(SIZES),
    compile_arms: bool = True,
    warmup: int = _WARMUP,
    iters: int = _ITERS,
) -> list[CaseResult]:
    """Measure every (module, size) case; return the rows."""
    rows: list[CaseResult] = []
    for size in sizes:
        for name in modules:
            rows.append(
                measure_module(
                    name,
                    size,
                    compile_arms=compile_arms,
                    warmup=warmup,
                    iters=iters,
                )
            )
            print(f"  done {name} @ {size}", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Run the delivery measurement and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modules",
        nargs="*",
        default=list(BOUND_MODULES),
        choices=list(BOUND_MODULES),
    )
    parser.add_argument(
        "--sizes",
        nargs="*",
        default=list(SIZES),
        choices=list(SIZES),
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip the torch.compile'd twin arms",
    )
    parser.add_argument("--iters", type=int, default=_ITERS)
    parser.add_argument("--warmup", type=int, default=_WARMUP)
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    rows = run(
        modules=tuple(args.modules),
        sizes=tuple(args.sizes),
        compile_arms=not args.no_compile,
        warmup=args.warmup,
        iters=args.iters,
    )
    print()
    print(_report(rows))
    if args.json:
        _dump_json(args.json, rows)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

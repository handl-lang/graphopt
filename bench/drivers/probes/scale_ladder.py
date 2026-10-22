"""The scale ladder — attention-block terms, T = 512 -> 4096.

Pure measurement, no new machinery: each rung pushes one zoo
attention-block term through the *shipped* pipeline and the
*shipped* referee gauntlet, timing every stage, until the box says
stop.  The question is where each stage's wall actually is — the
report claims only what the rows measured.

Stages per (workload, T) rung:

* ``export``  — ``TorchSource.to_ir`` (``torch_bridge.export_to_ir``)
  on the fp64 module.  For the unrolled blocks
  (``GatedLinearAttention``, ``MambaSelectiveScan`` — ``steps=t``)
  the traced term grows linearly in T; for the symbolic blocks
  (``ChunkedRetention``, ``ALiBiSlopeAttention``, ``RWKVMixBlock``)
  the term is T-blind and only the *example tensors* grow — the two
  regimes are the ladder's first finding.
* ``search``  — ``Optimizer.search`` under the shipped defaults:
  ``DEFAULT_RULES`` (273 rules — core DEFAULT + the machine-shipped
  mint + carriers), ``symmetry_budget=2048`` on the
  ``EXPANSIVE``-tagged closure generators, ``max_enodes`` /
  ``max_memory_mb`` bounds.  This call bundles saturation + the
  non-local passes + first extraction; ``stats`` carries the
  e-graph's own record (iterations, enodes, classes, proof edges,
  ``stop`` reason, ``budget_suspended``, ``rule_fires``).
* ``extract`` — a *second* ``eg.extract_best`` on the saturated
  graph, isolating the extraction walk's cost on the grown e-graph
  (the shared cost memo makes it a lower bound on the in-search
  pass).
* ``lower``   — ``Optimizer.lower(..., verify=False)``: executor
  routing + lowering, no referee.
* ``verify``  — the referee leg of delivery: a fresh reference
  lowering of ``res.ir`` vs the delivered module through
  ``sink.verify`` — this leg *runs the program*, so it is the
  first stage that feels T in real FLOPs/bytes.
* ``gauntlet`` — the ``bench/referees`` admission legs
  (``oracle`` -> ``pay`` -> ``sweep`` -> ``cert``) on one fixed
  claim per workload: the shipped law that fired most on the
  module at the ladder's first rung.  Legs run in gauntlet order
  through ``run_component`` on a shared ``ClaimEvidence`` — the
  same memoized-verdict seam ``judge`` uses — and stop at the
  first refusal, as ``judge`` does.  The claim is shipped art, so
  the verdict itself is not the finding; the *cost* of re-proving
  it against a T-scaled real term is.

Honesty contract — where the box says stop:

* every stage is wrapped: ``OptimizationResourceError`` (the
  machinery's own bounded refusal) records ``"resource"``,
  allocator failures record ``"oom"``, a wall-clock SIGALRM
  records ``"timeout"``, anything else ``"error"`` — with the
  exception text in ``note``, never a silent zero;
* a failed stage blocks the rung's remaining stages *and ends the
  workload's ladder* — a rung that OOMed at T is not retried at 2T;
* before every rung the sweep checks ``MemAvailable`` against
  ``--min-avail-gb`` (this is an 11 GB box shared with other
  agents) and refuses the rung — ``"skipped"`` — rather than
  racing the box into swap;
* ``VmHWM`` is sampled after every stage — the *cumulative* peak,
  reported as ``hwm_mb`` per stage boundary; per-stage peaks
  cannot be isolated (HWM is monotone) and the doc says so.

Run::

    .venv/bin/python bench/drivers/probes/scale_ladder.py
    .venv/bin/python bench/drivers/probes/scale_ladder.py \
        --workloads GatedLinearAttention --ts 512,1024,2048,4096
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlopt_carriers  # noqa: F401 — registers the carrier machinery
import torch
from mlopt_core.cost import dag_cost
from mlopt_discovery.impact import TermCase, _iter_subterms
from mlopt_orchestrator.optimize import (
    OptimizationResourceError,
    Optimizer,
    default_rules,
)
from mlopt_torch.adapters import TorchSink
from mlopt_torch.backend import TorchBackend

# ``bench`` is a repo-root package, not an installed distribution —
# ``mlopt_discovery.impact`` (imported above) puts the repo root on
# ``sys.path`` at module import, which is what makes this resolve
# under ``python bench/drivers/probes/scale_ladder.py``.
from bench.referees import components
from bench.referees.registry import COMPONENT_ORDER, RefereeCtx

#: The ladder: unrolled blocks (term size grows with T) plus the
#: symbolic attention spellings (term fixed, real tensors grow).
DEFAULT_WORKLOADS = (
    "GatedLinearAttention",
    "MambaSelectiveScan",
    "ChunkedRetention",
    "ALiBiSlopeAttention",
    "RWKVMixBlock",
)

#: The shipped default budgets — the bounds ``search`` applies when
#: the caller says nothing.  Exposed here so the report can name them.
DEFAULT_MAX_ENODES = 100_000
DEFAULT_MAX_MEMORY_MB = 6_144.0


class _Truncated(Exception):
    """Raised by the SIGALRM handler when a rung overruns."""


@dataclass
class _Clock:
    """A stage timing + the post-stage high-water mark."""

    seconds: float = 0.0
    hwm_mb: float = 0.0
    status: str = "ok"
    note: str = ""


@dataclass
class _Rung:
    """One (workload, T) measurement row."""

    workload: str
    t: int
    status: str = "ok"
    note: str = ""
    term_nodes: int = 0
    stages: dict[str, _Clock] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    legs: dict[str, dict[str, Any]] = field(default_factory=dict)
    claim: str = ""


def _avail_gb() -> float:
    """``MemAvailable`` in GiB — the pre-rung safety floor."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return float("inf")


def _vm(field: str) -> float:
    """Read one ``/proc/self/status`` Vm field in MiB."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field):
            return int(line.split()[1]) / 1024.0
    return 0.0


def _hwm() -> float:
    """Peak RSS so far (``VmHWM``) in MiB — monotone, cumulative."""
    return _vm("VmHWM:")


def _looks_oom(exc: BaseException) -> bool:
    """The allocator-failure vocabulary, host-side."""
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "can't allocate memory" in msg
        or "cannot allocate memory" in msg
        or "bad alloc" in msg
    )


def _run_stage(rung: _Rung, name: str, fn: Any) -> Any:
    """Time *fn* into ``rung.stages[name]``; return its value or None.

    ``_Truncated`` propagates — the alarm means the rung is over.
    Every other failure lands as a status on the stage clock.
    """
    clk = _Clock()
    rung.stages[name] = clk
    t0 = time.perf_counter()
    try:
        out = fn()
    except _Truncated:
        clk.seconds = time.perf_counter() - t0
        clk.hwm_mb = _hwm()
        clk.status = "timeout"
        clk.note = "wall clock exceeded the per-rung alarm"
        raise
    except OptimizationResourceError as e:
        clk.status, clk.note = "resource", f"{type(e).__name__}: {e}"
    except Exception as e:
        if _looks_oom(e):
            clk.status = "oom"
        else:
            clk.status = "error"
        clk.note = f"{type(e).__name__}: {e}"
    clk.seconds = time.perf_counter() - t0
    clk.hwm_mb = _hwm()
    if clk.status != "ok":
        rung.status = clk.status
        rung.note = f"{name}: {clk.note}"
        return None
    return out


def _term_nodes(term: Any) -> int:
    """Distinct subterm count — the DAG's node size."""
    return len(_iter_subterms(term))


def _pick_claim(fires: dict[str, int], bank: dict[str, Any]) -> Any:
    """The shipped law that fired most — the per-workload claim.

    Re-judging shipped art measures the gauntlet's cost at scale;
    the busiest law is the one whose legs actually exercise the
    rung's real term.
    """
    name = max(
        (n for n in fires if n in bank),
        key=lambda n: fires[n],
        default=None,
    )
    return bank.get(name) if name else None


def _gauntlet(rung: _Rung, claim: Any, ctx: RefereeCtx) -> None:
    """Run the admission legs in gauntlet order, timing each.

    Mirrors ``judge``: one shared ``ClaimEvidence``, legs stop at
    the first refusal.  A refused leg is a verdict, not a failure —
    the rung keeps its ``ok`` status either way; a leg that *raises*
    is recorded as ``error`` and ends the leg sequence.
    """
    evd = components.ClaimEvidence(claim, ctx)
    for leg in COMPONENT_ORDER:
        clk = _Clock()
        t0 = time.perf_counter()
        try:
            v = components.run_component(leg, claim, ctx, evd)
            clk.status = "ok" if v.passed else "refused"
            rung.legs[leg] = {
                "passed": bool(v.passed),
                "detail": v.detail,
            }
        except _Truncated:
            clk.seconds = time.perf_counter() - t0
            clk.hwm_mb = _hwm()
            clk.status = "timeout"
            rung.stages[f"gauntlet:{leg}"] = clk
            raise
        except Exception as e:
            clk.status = "oom" if _looks_oom(e) else "error"
            clk.note = f"{type(e).__name__}: {e}"
            rung.legs[leg] = {"passed": None, "detail": clk.note}
            clk.seconds = time.perf_counter() - t0
            clk.hwm_mb = _hwm()
            rung.stages[f"gauntlet:{leg}"] = clk
            return
        clk.seconds = time.perf_counter() - t0
        clk.hwm_mb = _hwm()
        rung.stages[f"gauntlet:{leg}"] = clk
        if not v.passed:
            return


def _rung(
    rung: _Rung,
    workload: Any,
    args: argparse.Namespace,
    opt: Optimizer,
    sink: TorchSink,
    bank: dict[str, Any],
    claims: dict[str, Any],
) -> _Rung:
    """Measure one (workload, T) rung through every stage."""
    built = _run_stage(rung, "build", workload.build)
    if built is None:
        return rung
    model, feed = built
    model = model.eval().double()
    x = feed

    exported = _run_stage(
        rung, "export", lambda: opt.source.to_ir(model, x)
    )
    if exported is None:
        return rung
    ir, tensors = exported
    rung.term_nodes = _term_nodes(ir.root)

    res = _run_stage(
        rung,
        "search",
        lambda: opt.search(
            model,
            x,
            max_iterations=args.max_iterations,
            max_enodes=args.max_enodes,
            max_memory_mb=args.max_memory_mb,
        ),
    )
    if res is None:
        return rung
    st = res.stats
    rung.stats = {
        "iterations": st.get("iterations"),
        "n_enodes": st.get("n_enodes"),
        "n_classes": st.get("n_classes"),
        "n_proof_edges": st.get("n_proof_edges"),
        "stop": st.get("stop"),
        "engine": st.get("engine"),
        "budget_suspended": st.get("budget_suspended", []),
        "rule_fires": st.get("rule_fires", {}),
        "cost_before": dag_cost(res.ir.root, res.cost_fn),
        "cost_after": dag_cost(res.term, res.cost_fn),
    }

    _run_stage(
        rung,
        "extract",
        lambda: res.eg.extract_best(res.root_eid, res.cost_fn),
    )

    low = _run_stage(
        rung, "lower", lambda: opt.lower(res, x, verify=False)
    )
    if low is None:
        return rung

    def _verify() -> Any:
        ref = sink.lower(res.ir, params=dict(res.param_values))
        return sink.verify(ref, low.module, x, rtol=1e-4)

    rep = _run_stage(rung, "verify", _verify)
    if rep is not None:
        rung.stats["verify_passed"] = bool(rep.passed)
        rung.stats["verify_max_rel"] = rep.max_rel

    # -- the referee gauntlet on the rung's own real term ---------------
    claim = claims.get(rung.workload)
    if claim is None:
        claim = _pick_claim(rung.stats["rule_fires"], bank)
        claims[rung.workload] = claim
    if claim is not None:
        rung.claim = claim.name
        ctx = RefereeCtx(
            real_terms=(ir.root,),
            probe=(
                TermCase(
                    source="zoo",
                    name=rung.workload,
                    term=ir.root,
                    inputs=tuple(ir.inputs),
                    feed=tuple(x) if isinstance(x, tuple) else (x,),
                    param_vals=dict(tensors),
                ),
            ),
            sink=sink,
        )
        _gauntlet(rung, claim, ctx)
    return rung


def _serialize(rung: _Rung) -> dict[str, Any]:
    """The JSON-safe row for one rung."""
    return {
        "workload": rung.workload,
        "t": rung.t,
        "status": rung.status,
        "note": rung.note,
        "term_nodes": rung.term_nodes,
        "claim": rung.claim,
        "stages": {
            k: {
                "s": round(c.seconds, 3),
                "hwm_mb": round(c.hwm_mb, 1),
                "status": c.status,
                **({"note": c.note} if c.note else {}),
            }
            for k, c in rung.stages.items()
        },
        "stats": rung.stats,
        "legs": rung.legs,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the ladder; stream JSON rows to ``--json`` after each rung."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--workloads",
        default=",".join(DEFAULT_WORKLOADS),
        help="comma-separated zoo workload names",
    )
    ap.add_argument(
        "--ts",
        default="512,1024,2048,4096",
        help="comma-separated sequence lengths",
    )
    ap.add_argument("--d", type=int, default=16, help="model width")
    ap.add_argument("--b", type=int, default=2, help="batch size")
    ap.add_argument(
        "--per-case",
        type=float,
        default=600.0,
        help="per-rung wall-clock alarm, seconds",
    )
    ap.add_argument(
        "--min-avail-gb",
        type=float,
        default=2.0,
        help="refuse a rung below this MemAvailable floor",
    )
    ap.add_argument(
        "--max-enodes", type=int, default=DEFAULT_MAX_ENODES
    )
    ap.add_argument(
        "--max-memory-mb", type=float, default=DEFAULT_MAX_MEMORY_MB
    )
    ap.add_argument("--max-iterations", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--json",
        default="docs/reports/results_scale_ladder.json",
        help="where the rung rows stream",
    )
    args = ap.parse_args(argv)

    from mlopt_discovery import zoo as zoo_mod

    wanted = [w.strip() for w in args.workloads.split(",") if w.strip()]
    ts = [int(s) for s in args.ts.split(",") if s.strip()]

    opt = Optimizer(backend=TorchBackend())
    sink = TorchSink()
    bank = {r.name: r for r in default_rules()}
    claims: dict[str, Any] = {}

    def _alarm(_sig: int, _frm: Any) -> None:
        raise _Truncated()

    signal.signal(signal.SIGALRM, _alarm)

    rows: list[dict[str, Any]] = []
    out_path = Path(args.json)

    def _flush() -> None:
        doc = {
            "schema": 1,
            "protocol": {
                "d": args.d,
                "b": args.b,
                "ts": ts,
                "seed": args.seed,
                "per_case_s": args.per_case,
                "min_avail_gb": args.min_avail_gb,
                "max_enodes": args.max_enodes,
                "max_memory_mb": args.max_memory_mb,
                "max_iterations": args.max_iterations,
                "rules": "default_rules()",
                "n_rules": len(bank),
            },
            "rows": rows,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, indent=2, default=str))

    for name in wanted:
        for t in ts:
            avail = _avail_gb()
            if avail < args.min_avail_gb:
                rows.append(
                    {
                        "workload": name,
                        "t": t,
                        "status": "skipped",
                        "note": (
                            f"MemAvailable {avail:.2f} GiB < "
                            f"floor {args.min_avail_gb} GiB"
                        ),
                    }
                )
                _flush()
                break
            # Fresh thunks per rung — zoo(d, t, b) binds T into the
            # workload's example input (and the unroll count, for
            # the recurrent spellings).
            reg = {
                w.name: w for w in zoo_mod.zoo(d=args.d, t=t, b=args.b)
            }
            if name not in reg:
                print(f"unknown zoo workload {name!r}", file=sys.stderr)
                break
            torch.manual_seed(args.seed * 100_003 + t)
            rung = _Rung(workload=name, t=t)
            # Pre-flush a ``running`` row: a rung the kernel OOM-kills
            # (SIGKILL — no unwinding, no flush) otherwise leaves no
            # record at all.  The completed rung overwrites it below.
            rows.append({"workload": name, "t": t, "status": "running"})
            _flush()
            signal.setitimer(signal.ITIMER_REAL, args.per_case)
            try:
                _rung(rung, reg[name], args, opt, sink, bank, claims)
            except _Truncated:
                # The alarm fired mid-stage: ``_run_stage`` already
                # recorded the stage clock before re-raising, so the
                # rung keeps its partial timings.
                rung.status = "timeout"
                if not rung.note:
                    rung.note = "wall clock exceeded the per-rung alarm"
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            row = _serialize(rung)
            rows[-1] = row
            _flush()
            st = row["status"]
            top = sorted(
                row["stats"].get("rule_fires", {}).items(),
                key=lambda kv: -kv[1],
            )[:5]
            print(
                f"{name} T={t}: {st} "
                f"nodes={row['term_nodes']} "
                f"enodes={row['stats'].get('n_enodes')} "
                f"stop={row['stats'].get('stop')} "
                f"top_fires={top}"
            )
            if st != "ok":
                # The box said stop: a failed rung ends this
                # workload's ladder — bigger T only gets worse.
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())

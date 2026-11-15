"""ADR-0178 WP4 gate — measured parity sweep: Python ``EGraph`` vs
``engine="eqsat"`` on real liftable models.

WP3 landed the ``engine="eqsat"`` opt-in
(``mlopt_orchestrator.eqsat.EqsatEngine`` — the Rust ``engine-eqsat``
saturation engine behind the ``Engine`` port, driven through the
``eqsat-run`` JSON CLI).  WP4 proposes flipping the *default* engine;
this driver is the measurement that gates that flip.  For each
selected ``bench.discovery`` case it runs the pipeline's own
``search -> lower(verify=True)`` path twice — once with the default
Python ``EGraph`` (``engine=None``), once with ``engine="eqsat"`` —
and records, per arm:

* extracted term size/cost on the case's symbolic axis
  (``dag_cost`` with the case's ``COST_FNS`` entry), plus the
  engine's own stats (``n_enodes`` / ``iterations`` / ``stop``);
* the delivered-module verify verdict (``lr.verified.passed`` /
  ``max_abs``) — the same ``TorchSink.verify`` gate every search
  result faces;
* for the eqsat arm, the honest coverage record
  (``stats["eqsat"]["laws"]`` — requested / applied / uncovered) and
  the fired-law table, since the portable law set (55
  ``LAW_FORMAT = 2`` records) is a strict subset of the Python bank.

Honesty contract: a search error, a lower/verify decline, or an
eqsat-run refusal is a *row*, never a dropped case — the parity
verdict counts declines as failures-to-parity, not as absences.
Non-local passes (pairing / lifts) are Python-``EGraph`` machinery
and are skipped for the eqsat engine by ``search`` itself; that
asymmetry is the thing being measured, so both arms run the *same*
call shape otherwise (same rules, cost axis, iteration/node caps).

**Environment.**  The eqsat arm needs ``mlopt_orchestrator`` new
enough to carry ``eqsat.py`` (eqsat main, post-WP3) and an
``eqsat-run`` binary.  graphopt pins eqsat at tag ``v0.1.2`` (pre-WP3),
so run with the eqsat-main packages overlaid::

    EQSAT=/home/luis/GitHub/handl-lang/eqsat
    PYTHONPATH=$EQSAT/packages/mlopt-orchestrator/src:\
$EQSAT/packages/mlopt-core/src:\
$EQSAT/packages/mlopt-dtypes/src \
    MLOPT_EQSAT_RUN=$HOME/GitHub/handl-lang/engine-eqsat/target/debug/eqsat-run \
    .venv/bin/python bench/drivers/audit/eqsat_parity_sweep.py \
        --out /tmp/wp4_parity.json

``MLOPT_EQSAT_RUN`` is optional when the binary sits in the
engine-eqsat checkout's ``target/{release,debug}`` (the resolver's
built-in probe order).  ``--cases`` takes a comma-separated subset of
the default sweep list.

Usage::

    .venv/bin/python bench/drivers/audit/eqsat_parity_sweep.py --dry
    .venv/bin/python bench/drivers/audit/eqsat_parity_sweep.py \
        --out docs/reports/wp4_parity_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: The WP4-gate sweep — a handful of small, honest, real lifts across
#: the corpus's fold/lift families.  Names are ``bench.discovery``
#: case ids; ``--cases`` overrides.
DEFAULT_CASES = [
    "conv_bn_fold",
    "silu_fold",
    "softmax_fold",
    "glu_fold",
    "swiglu_mlp",
    "select_mul",
    "rms_norm_fold",
    "mul_unsq_nat",
    "weight_merge",
]


def _term_ops(term: Any) -> list[str]:
    """Reachable op names in *term*, sorted — the cheap shape record."""
    from mlopt_core.ir import Op

    seen: set[str] = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Op):
            seen.add(t.op)
            stack.extend(t.args)
    return sorted(seen)


def _run_arm(
    case: Any,
    model: Any,
    feed: Any,
    engine: Any,
    cost_fn: Any,
    rules: Any,
) -> dict[str, Any]:
    """One ``search -> lower(verify=True)`` pass; every failure is data."""
    import torch  # noqa: F401 — fixture builders need it seeded
    from mlopt_orchestrator.optimize import (
        VerifyDeclinedError,
        lower,
        search,
    )
    from mlopt_torch.adapters import TorchSink, TorchSource

    row: dict[str, Any] = {
        "engine": getattr(engine, "engine_name", None)
        or ("eqsat" if engine == "eqsat" else "python")
    }
    t0 = time.time()
    try:
        sr = search(
            model,
            feed,
            source=TorchSource(),
            capabilities=TorchSink(),
            cost_fn=cost_fn,
            rules=rules,
            max_iterations=case.max_iterations,
            max_enodes=case.max_enodes,
            engine=engine,
        )
    except Exception as e:  # search itself refused/failed — a row
        row["error"] = f"search: {type(e).__name__}: {e}"
        row["search_s"] = round(time.time() - t0, 3)
        return row
    row["search_s"] = round(time.time() - t0, 3)
    row["stats_engine"] = sr.stats.get("engine")
    row["n_enodes"] = sr.stats.get("n_enodes")
    row["n_classes"] = sr.stats.get("n_classes")
    row["iterations"] = sr.stats.get("iterations")
    row["stop"] = sr.stats.get("stop")
    row["nonlocal_passes"] = sr.stats.get("nonlocal_passes")
    eqsat_stats = sr.stats.get("eqsat")
    if isinstance(eqsat_stats, dict):
        laws = eqsat_stats.get("laws") or {}
        row["eqsat"] = {
            "binary": eqsat_stats.get("binary"),
            "cost": eqsat_stats.get("cost"),
            "laws_requested": laws.get("requested"),
            "laws_applied": len(laws.get("applied") or []),
            "laws_uncovered_n": len(laws.get("uncovered") or []),
            "fired": list(eqsat_stats.get("fired") or []),
            "rule_fires": dict(sr.stats.get("rule_fires") or {}),
        }
    row["term_ops"] = _term_ops(sr.term)
    try:
        from mlopt_core.cost import dag_cost

        row["cost_before"] = dag_cost(sr.ir.root, cost_fn)
        row["cost_after"] = dag_cost(sr.term, cost_fn)
    except Exception as e:
        row["cost_error"] = f"{type(e).__name__}: {e}"

    # -- lower + verify: the delivery gate both arms must pass -------
    try:
        lr = lower(sr, feed, sink=TorchSink(), verify=True)
    except VerifyDeclinedError as e:
        lr = e.result
        row["verify_declined"] = str(e)[:300]
    except Exception as e:
        row["error"] = f"lower: {type(e).__name__}: {e}"
        return row
    rep = getattr(lr, "verified", None)
    row["verified"] = bool(rep.passed) if rep is not None else None
    row["verify_max_abs"] = (
        float(rep.max_abs) if rep is not None else None
    )
    return row


def _case_rules(name: str):
    """Mirror ``bench.discovery.runner._case_rules``."""
    from bench.discovery.runner import _case_rules as impl

    return impl(name)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eqsat_parity_sweep")
    p.add_argument("--cases", default=",".join(DEFAULT_CASES))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--dry",
        action="store_true",
        help="list the sweep and exit",
    )
    args = p.parse_args(argv)

    names = [n.strip() for n in args.cases.split(",") if n.strip()]
    if args.dry:
        for n in names:
            print(n)
        return 0

    # The eqsat arm must be importable BEFORE any search runs — fail
    # loudly with the overlay recipe rather than mid-sweep.
    try:
        from mlopt_orchestrator.eqsat import EqsatEngine  # noqa: F401
    except ImportError as e:
        print(
            "eqsat_parity_sweep needs mlopt_orchestrator with "
            "eqsat.py (eqsat main, post-WP3) — graphopt pins "
            "v0.1.2 which predates it.  Overlay the eqsat-main "
            "packages on PYTHONPATH (see this file's docstring); "
            f"import failed: {e}",
            file=sys.stderr,
        )
        return 2

    import torch

    from bench.discovery.cases import load_cases
    from bench.discovery.fixtures import SOURCES
    from bench.discovery.runner import COST_FNS

    torch.manual_seed(0)
    by_name = {c.name: c for c in load_cases()}
    missing = [n for n in names if n not in by_name]
    if missing:
        print(f"unknown cases: {missing}", file=sys.stderr)
        return 2

    out_rows = []
    for name in names:
        case = by_name[name]
        if case.source not in SOURCES:
            print(
                f"{name}: unknown source {case.source!r}",
                file=sys.stderr,
            )
            continue
        torch.manual_seed(case.seed)
        try:
            model, feed = SOURCES[case.source](case.size)
        except Exception as e:
            out_rows.append({"case": name, "error": f"fixture: {e}"})
            continue
        cost_fn = COST_FNS[case.cost]
        rules = _case_rules(case.rules)
        row = {"case": name, "source": case.source, "size": case.size}
        for engine in (None, "eqsat"):
            arm = _run_arm(case, model, feed, engine, cost_fn, rules)
            row[engine or "python"] = arm
            tag = engine or "python"
            err = arm.get("error")
            print(
                f"  {name:<16} {tag:<7} "
                + (
                    f"ERROR {err}"
                    if err
                    else f"cost {arm.get('cost_before')} -> "
                    f"{arm.get('cost_after')}  "
                    f"verified={arm.get('verified')} "
                    f"max_abs={arm.get('verify_max_abs')}"
                ),
                flush=True,
            )
        out_rows.append(row)

    doc = {
        "sweep": "wp4-parity",
        "adr": "ADR-0178 WP4 gate",
        "cases": out_rows,
    }
    text = json.dumps(doc, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

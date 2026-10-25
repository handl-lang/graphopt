"""``pay_wall`` — the measured pay leg for the minted primitive folds.

The admission gauntlet's ``pay`` leg prices a claim in the extraction
*cost model*.  For the shipped ``prim:synop*`` family that has been the
only pay evidence, and it is wrong about the executor: a ``synop`` call
is one dispatch in the model and a Python spec-walk at runtime
(``docs/reports/prim_delivery.md``).  This leg is the missing
measurement — the claim has to be faster on a delivered module, or it
does not get to say "paying".

What it measures, per module and size, reusing the delivery probe's own
four-arm protocol (``bench/drivers/probes/prim_delivery_probe.py``):

* ``marginal = synop_ms / noprim_ms`` — the fold family's own effect
  inside the same executor (the pay number);
* ``e2e = synop_ms / spelled_ms`` — what a caller of the pipeline
  receives.

Verdict band, per module: ``marginal <= 0.95`` delivers, ``>= 1.05``
loses, else ties.  A fold's verdict aggregates the modules whose
extraction carries that op by the *worst* module — a family that loses
anywhere has not paid.

Claims are recorded beside their records when ``--store`` is given:
``measurements(name=<synop>, metric="pay_wall", arm="<module>@<size>")``.
Nothing is demoted here: this leg produces the number a demotion
decision cites.

Run::

    .venv/bin/python -m bench.referees.pay_wall --modules RotaryPair --sizes small
    .venv/bin/python -m bench.referees.pay_wall --json /tmp/pay.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bench.drivers.probes import prim_delivery_probe as probe

#: delivers / loses bands on the marginal ratio.
DELIVERS = 0.95
LOSES = 1.05

DEFAULT_MODULES = (
    "GatedDeltaRule",
    "GatedLinearAttention",
    "MambaSelectiveScan",
    "RealGatedLRU",
    "HighwayGate",
    "RotaryPair",
    "GRUFusedGate",
)
DEFAULT_SIZES = ("small", "large")


def verdict(marginal: float | None) -> str:
    """Return the band verdict for a marginal ratio."""
    if marginal is None:
        return "unmeasured"
    if marginal <= DELIVERS:
        return "delivers"
    if marginal >= LOSES:
        return "loses"
    return "ties"


def measure(
    module: str,
    size: str,
    *,
    compile_arms: bool = False,
    iters: int | None = None,
    warmup: int | None = None,
) -> dict[str, Any]:
    """Time one module at one size; return its arms and verdicts."""
    kwargs: dict[str, Any] = {"compile_arms": compile_arms}
    if iters is not None:
        kwargs["iters"] = iters
    if warmup is not None:
        kwargs["warmup"] = warmup
    row = probe.measure_module(module, size, **kwargs)
    times = row.times

    def med(arm: str) -> float | None:
        entry = times.get(arm)
        if not entry or "median" not in entry:
            return None
        return float(entry["median"])

    synop = med("synop")
    noprim = med("noprim")
    spelled = med("spelled")
    marginal = (synop / noprim) if synop and noprim else None
    e2e = (synop / spelled) if synop and spelled else None
    executors = dict(row.executors)
    # A marginal is only the fold's own effect when both arms run on the
    # same executor; the probe flags routing differences, and a cell that
    # routes differently measures the executor, not the fold.
    confounded = executors.get("synop") != executors.get("noprim")
    cell_verdict = "confounded" if confounded else verdict(marginal)
    return {
        "module": module,
        "size": size,
        "ms": {
            "spelled": spelled,
            "tape": med("tape"),
            "noprim": noprim,
            "synop": synop,
        },
        "marginal": marginal,
        "e2e": e2e,
        "verdict": cell_verdict,
        "synops": list(row.synops),
        "executors": executors,
        "verified": bool(row.verified.get("synop") == "pass"),
        "note": row.note.strip(),
    }


def run(
    *,
    modules: tuple[str, ...] = DEFAULT_MODULES,
    sizes: tuple[str, ...] = DEFAULT_SIZES,
    compile_arms: bool = False,
    iters: int | None = None,
    warmup: int | None = None,
) -> dict[str, Any]:
    """Measure every requested module/size pair; aggregate per fold."""
    rows = [
        measure(
            m, s, compile_arms=compile_arms, iters=iters, warmup=warmup
        )
        for m in modules
        for s in sizes
    ]
    per_op: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for op in row["synops"]:
            per_op.setdefault(op, []).append(
                {
                    "module": row["module"],
                    "size": row["size"],
                    "marginal": row["marginal"],
                    "verdict": row["verdict"],
                }
            )
    folds = {}
    for op, hits in sorted(per_op.items()):
        clean = [h for h in hits if h["verdict"] != "confounded"]
        measured = [
            h["marginal"] for h in clean if h["marginal"] is not None
        ]
        worst = max(measured) if measured else None
        folds[op] = {
            "modules": hits,
            "worst": worst,
            "best": min(measured) if measured else None,
            "mean": (sum(measured) / len(measured))
            if measured
            else None,
            "verdict": verdict(worst) if measured else "confounded",
        }
    return {
        "rows": rows,
        "folds": folds,
        "modules": list(modules),
        "sizes": list(sizes),
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def summary(result: dict[str, Any]) -> str:
    """Render the run as a short table."""
    counts: dict[str, int] = {}
    for row in result["rows"]:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    lines = [
        f"modules {result['modules']} sizes {result['sizes']}",
        f"cells: {counts}",
        "",
        "| module | size | spelled | noprim | synop | marginal | e2e | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        ms = row["ms"]
        lines.append(
            f"| {row['module']} | {row['size']} | {_fmt(ms['spelled'])} | "
            f"{_fmt(ms['noprim'])} | {_fmt(ms['synop'])} | "
            f"{_fmt(row['marginal'])} | {_fmt(row['e2e'])} | {row['verdict']} |"
        )
    if result["folds"]:
        lines += [
            "",
            "| fold op | worst | mean | verdict |",
            "|---|---|---|---|",
        ]
        for op, agg in result["folds"].items():
            lines.append(
                f"| `{op}` | {_fmt(agg['worst'])} | {_fmt(agg['mean'])} | "
                f"{agg['verdict']} |"
            )
    return "\n".join(lines)


def _store_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """One measurement row per (fold, module, size) — the store's shape."""
    rows: list[dict[str, Any]] = []
    for op, agg in result["folds"].items():
        for hit in agg["modules"]:
            if hit["marginal"] is None:
                continue
            rows.append(
                {
                    "name": op,
                    "metric": "pay_wall",
                    "arm": f"{hit['module']}@{hit['size']}",
                    "value": hit["marginal"],
                    "unit": "ratio",
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Run the leg, print the table, and record the claims."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--modules", nargs="*", default=list(DEFAULT_MODULES)
    )
    ap.add_argument("--sizes", nargs="*", default=list(DEFAULT_SIZES))
    ap.add_argument(
        "--compile", action="store_true", help="time compiled twins too"
    )
    ap.add_argument("--iters", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--store", help="record into this minted store")
    ap.add_argument(
        "--run", default=None, help="run tag stored with each number"
    )
    args = ap.parse_args(argv)

    result = run(
        modules=tuple(args.modules),
        sizes=tuple(args.sizes),
        compile_arms=args.compile,
        iters=args.iters,
        warmup=args.warmup,
    )
    print(summary(result))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\nwrote {path}")
    if args.store:
        from mlopt_core.laws import store as law_store

        written = law_store.record_measurements(
            _store_rows(result), args.store, run=args.run
        )
        print(f"recorded {written} measurements → {args.store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

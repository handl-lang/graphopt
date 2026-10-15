"""Scale the exposed-math discovery sweep across the corpus top.

``lab/screen/exposure_screen.py`` ranked 321 workloads by e-graph
reachability; ``lab/screen/exposed_discovery.py`` then ran the admission
pipeline on eight of them and minted one paying unpublished law
(``census:mul_unsqueeze`` on DeltaRuleBlock — see
``project/retros/exposed-math.md``).  This tool is the follow-through:
run the same three-arm discovery machinery across the **rest** of the
high-exposure class, budgeted, and answer the corpus-level question —
how much unpublished paying structure actually exists in the exposed
class?

Selection: rows from the screen JSON with ``status == "ok"`` and
``lhs_cover >= --floor`` (the ~0% fused-op class is skipped entirely —
the e-graph cannot see them), minus the eight models the first round
already ran, ranked by ``lhs_cover * strict_sites`` (exposure x real
match count), top ``--top``.

Budgets, all explicit:

* ``--per-model SECONDS`` — a real wall-clock alarm per model.  A
  model that overruns is recorded ``"status": "truncated"`` and the
  sweep moves on.
* ``--total-budget SECONDS`` — the whole sweep stops allocating new
  models past this deadline (the running model finishes).
* ``--min-avail-gb`` — the sweep aborts before starting a model when
  MemAvailable drops below the floor (this is an 11 GB box shared
  with other agents).
* ``--max-candidates`` / ``--evaluate`` — forwarded to the
  enumerator's bounded-oracle prefix, unchanged.

The paying bar is the one the DeltaRuleBlock run used: a pipeline
candidate pays when it is ``relation=new``, numerically true, fires,
drops extracted cost, verifies, and its certificate replays; an
enumerate candidate pays when it fires, pays, and the lowered module
verifies.  Every distinct paying candidate then gets the **novelty
sweep**: is its LHS strictly matchable anywhere else in the corpus
(the DeltaRuleBlock check — ``mul(unsqueeze, unsqueeze)`` existed on
one architecture only)?  Intake terms come from the committed
``tools/data/intake_corpus.json`` ledger; zoo and corpus-model terms are
re-exported live.

Results stream to ``--json`` after every model, so a dead box still
leaves the partial harvest on disk.

Run::

    .venv/bin/python lab/screen/discovery_scale.py --top 40
    .venv/bin/python lab/screen/discovery_scale.py --top 12 --per-model 120
"""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

import catopt_carriers  # noqa: F401 — registers the carrier machinery
import torch
from catopt_core.egraph.terms import _term_match
from catopt_core.ir import Op, term_from_data
from catopt_discovery import intake_dir
from catopt_discovery.impact import _iter_subterms, new_laws

from lab.screen.exposed_discovery import discover
from lab.screen.exposure_screen import _export, _workloads

#: The eight models ``project/retros/exposed-math.md`` already ran —
#: excluded by default so this sweep measures only *new* ground.
DONE = frozenset(
    {
        "LoRAAdapter",
        "GatedAttentionUnit",
        "DeepEquilibrium",
        "CosineAttention",
        "AdaLNBlock",
        "CapsuleRouting",
        "ChunkedRetention",
        "DeltaRuleBlock",
    }
)

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SCREEN = (
    _ROOT / "project" / "retros" / "exposed-math-screen.json"
)
_DEFAULT_JSON = _ROOT / "project" / "retros" / "discovery-scale.json"
_INTAKE_JSON = intake_dir() / "intake_corpus.json"


class _BudgetExceeded(Exception):
    """Raised by the SIGALRM handler when a model overruns."""


def _avail_gb() -> float:
    """Return MemAvailable in GiB (Linux ``/proc/meminfo``)."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20
    return float("inf")


@contextlib.contextmanager
def _budget(seconds: float):
    """Raise ``_BudgetExceeded`` *seconds* of wall-clock from now."""

    def _raise(signum: int, frame: Any) -> None:
        raise _BudgetExceeded(f"budget {seconds}s exceeded")

    old = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _eligible(
    screen_path: Path, floor: float, done: frozenset[str]
) -> list[tuple[str, str, float]]:
    """Rank screen rows by exposure x strict matches; drop done/low."""
    rows = json.loads(screen_path.read_text())["rows"]
    pick = []
    for r in rows:
        if r["status"] != "ok" or r["lhs_cover"] < floor:
            continue
        if r["name"] in done:
            continue
        pick.append(
            (
                r["source"],
                r["name"],
                r["lhs_cover"] * r["strict_sites"],
            )
        )
    pick.sort(key=lambda t: (-t[2], t[1]))
    return pick


def _workload_map() -> dict[tuple[str, str], Any]:
    """Index every registry workload by ``(source, name)``."""
    return {(s, w.name): w for s, w in _workloads("all")}


def _run_model(
    key: str, wl: Any, args: argparse.Namespace
) -> dict[str, Any]:
    """Export + run all three discovery arms on one workload."""
    _row, case = _export(wl, key.split(":", 1)[0])
    if case is None:
        return {
            "model": key,
            "status": "rejected",
            "note": _row.note,
        }
    res = discover(case, args)
    res["model"] = key
    res["status"] = "ok"
    return res


def _sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run the budgeted per-model sweep; return per-model results."""
    picked = _eligible(Path(args.screen), args.floor, DONE)
    picked = picked[: args.top]
    wls = _workload_map()
    print(  # stdout-compat
        f"== discovery_scale — {len(picked)} model(s) queued, "
        f"per-model {args.per_model}s, total {args.total_budget}s =="
    )
    results: list[dict[str, Any]] = []
    t_start = time.time()
    for source, name, _score in picked:
        key = f"{source}:{name}"
        wl = wls.get((source, name))
        if wl is None:
            results.append(
                {"model": key, "status": "rejected", "note": "no wl"}
            )
            continue
        if _avail_gb() < args.min_avail_gb:
            print(f"!! RAM below {args.min_avail_gb}G — aborting sweep")
            break
        if time.time() - t_start > args.total_budget:
            print("!! total budget spent — stopping after queued run")
            break
        t0 = time.time()
        try:
            with _budget(args.per_model):
                res = _run_model(key, wl, args)
        except _BudgetExceeded:
            res = {
                "model": key,
                "status": "truncated",
                "elapsed_s": round(time.time() - t0, 1),
            }
        except Exception as e:  # honest per-model failure
            res = {
                "model": key,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            }
        results.append(res)
        _write_json(args.json, results, score_done=len(results))
        _print_row(res)
    return results


def _write_json(
    path: str | None, results: list, score_done: int = 0
) -> None:
    """Stream the partial harvest to disk (crash-safe)."""
    if path:
        Path(path).write_text(
            json.dumps(
                {"results": results, "done": score_done},
                indent=2,
                default=str,
            )
            + "\n"
        )


def _print_row(res: dict[str, Any]) -> None:
    """Print one sweep row as it lands."""
    if res["status"] != "ok":
        print(f"  {res['model']:<30} {res['status']}")
        return
    evs = res["pipeline"]
    en = res["enumerate"]
    paying = _paying_of(res)
    print(  # stdout-compat
        f"  {res['model']:<30} {res['elapsed_s']:>6.1f}s "
        f"cand={len(evs) + en['emitted']:>4} "
        f"pipe={len(evs)}(fire {sum(1 for e in evs if e['fires'])}) "
        f"enum={en['emitted']}(nt {en.get('novel_true', 0)}) "
        f"paying={len(paying)}"
    )
    for p in paying:
        print(f"      -> {p['name']}  {p['lhs']} = {p['rhs']}")


def _paying_of(res: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect the paying candidates of one model's harvest.

    The bar is the DeltaRuleBlock run's: pipeline candidates must be
    ``new`` + numerically true + firing + cost-dropping + verifying +
    certificate-replaying; enumerate candidates must fire + pay +
    verify.  Each row carries ``lhs_data``/``rhs_data`` so the novelty
    sweep can rebuild the term.
    """
    out: list[dict[str, Any]] = []
    if res.get("status") != "ok":
        return out
    for e in res["pipeline"]:
        if (
            e["relation"] == "new"
            and e["num_true"]
            and e["fires"]
            and e["paid"]
            and not e["verify_fail"]
            and not e["cert_fail"]
        ):
            out.append(
                {
                    "arm": "pipeline",
                    "model": res["model"],
                    "name": e["name"],
                    "lhs": e["lhs"],
                    "rhs": e["rhs"],
                    "lhs_data": e["lhs_data"],
                    "rhs_data": e["rhs_data"],
                    "derivable": e["derivable"],
                    "fires": e["fires"],
                    "fires_typed": e["fires_typed"],
                    "cost_drop": e["cost_drop"],
                    "shippable": e["shippable"],
                    "no_ship_reason": e["no_ship_reason"],
                }
            )
    for p in res["enumerate"]["probes"]:
        if p["fires"] and p["paid"] and p["verified"] == "pass":
            out.append(
                {
                    "arm": "enumerate",
                    "model": res["model"],
                    "name": p["label"],
                    "lhs": p["lhs"],
                    "rhs": p["rhs"],
                    "lhs_data": p["lhs_data"],
                    "rhs_data": p["rhs_data"],
                    "derivable": p["derivable"],
                    "fires": p["fires"],
                    "fires_typed": p["fires"],
                    "cost_drop": 0.0,
                    "shippable": False,
                    "no_ship_reason": "enumerate arm",
                }
            )
    return out


def _corpus_terms() -> dict[str, Any]:
    """Return ``{model_key: term}`` for every corpus term.

    Intake terms ride the committed ledger (``intake_corpus.json`` —
    the same ``term_to_data`` records the screen exported); zoo and
    corpus-model terms are re-exported live so the novelty sweep
    covers the same 321 terms the screen saw.
    """
    terms: dict[str, Any] = {}
    ledger = json.loads(_INTAKE_JSON.read_text())
    for rec in ledger.get("workloads", []):
        if rec.get("status") != "ingested" or "term" not in rec:
            continue
        name = rec["name"]
        if not name.startswith("intake:"):
            name = f"intake:{name}"
        terms[name] = term_from_data(rec["term"])
    for source, wl in _workloads("zoo") + _workloads("models"):
        if f"{source}:{wl.name}" in terms:
            continue
        try:
            _row, case = _export(wl, source)
        except Exception:
            continue
        if case is not None:
            terms[f"{source}:{wl.name}"] = case.term
    return terms


def _strict_sites(lhs: Any, term: Any) -> int:
    """Count subterms of *term* strictly matching *lhs*."""
    return sum(
        1
        for s in _iter_subterms(term)
        if isinstance(s, Op) and _term_match(lhs, s) is not None
    )


def _known_law_of(lhs: Any, rhs: Any) -> str | None:
    """Return the ``new_laws()`` name a candidate duplicates, if any.

    A paying candidate the enumerator/pipeline mints may be a
    re-derivation of one of the 11 already-found-but-unshipped machine
    laws — honest reporting must not double-count those as fresh
    finds.  Structural identity is checked bidirectionally so metavar
    naming differences do not matter.
    """
    for law in new_laws():
        if (
            _term_match(lhs, law.lhs) is not None
            and _term_match(law.lhs, lhs) is not None
            and _term_match(rhs, law.rhs) is not None
            and _term_match(law.rhs, rhs) is not None
        ):
            return law.name
    return None


def _novelty(
    results: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Corpus-sweep every distinct paying candidate's LHS.

    Rebuilds each paying LHS once (deduped by its ``term_to_data``
    key) and strict-matches it against every corpus term — the same
    "exists only on that architecture" check the DeltaRuleBlock run
    applied to ``mul_unsqueeze``.
    """
    paying: dict[str, dict[str, Any]] = {}
    for res in results:
        for p in _paying_of(res):
            k = json.dumps(
                [p["lhs_data"], p["rhs_data"]], sort_keys=True
            )
            paying.setdefault(k, p)
    if not paying:
        return []
    print(f"-- novelty sweep: {len(paying)} distinct paying LHS --")
    terms = _corpus_terms()
    rows = []
    for p in paying.values():
        lhs = term_from_data(p["lhs_data"])
        rhs = term_from_data(p["rhs_data"])
        where = {}
        for key, t in terms.items():
            n = _strict_sites(lhs, t)
            if n:
                where[key] = n
        rows.append(
            {
                "name": p["name"],
                "arm": p["arm"],
                "minted_on": p["model"],
                "lhs": p["lhs"],
                "rhs": p["rhs"],
                "derivable": p["derivable"],
                "fires": p["fires"],
                "cost_drop": p["cost_drop"],
                "known_law": _known_law_of(lhs, rhs),
                "corpus_models": len(where),
                "corpus_sites": sum(where.values()),
                "where": where,
            }
        )
        print(  # stdout-compat
            f"  {p['name']:<32} on={p['model']:<28} "
            f"corpus={len(where)} models / "
            f"{sum(where.values())} sites"
        )
    return rows


def _table(results: list[dict[str, Any]]) -> str:
    """Render the honest per-model harvest table."""
    head = (
        f"{'model':<30} {'status':<10} {'cand':>5} {'fire':>5} "
        f"{'pay':>4} {'ver':>4} {'ship':>4} {'enum-nt':>7} "
        f"{'known':>5}"
    )
    lines = [head, "-" * len(head)]
    tot = {"cand": 0, "fire": 0, "pay": 0, "ship": 0}
    for r in results:
        if r["status"] != "ok":
            lines.append(
                f"{r['model']:<30} {r['status']:<10} {'--':>5} "
                f"{'--':>5} {'--':>4} {'--':>4} {'--':>4} {'--':>7} "
                f"{'--':>5}"
            )
            continue
        evs = r["pipeline"]
        en = r["enumerate"]
        n_cand = len(evs) + en["emitted"]
        n_fire = sum(1 for e in evs if e["fires"]) + sum(
            1 for p in en["probes"] if p["fires"]
        )
        n_pay = len(_paying_of(r))
        n_ver = sum(
            1 for e in evs if e["fires"] and not e["verify_fail"]
        ) + sum(1 for p in en["probes"] if p["verified"] == "pass")
        n_ship = sum(1 for e in evs if e["shippable"])
        tot["cand"] += n_cand
        tot["fire"] += n_fire
        tot["pay"] += n_pay
        tot["ship"] += n_ship
        lines.append(
            f"{r['model']:<30} {'ok':<10} {n_cand:>5} {n_fire:>5} "
            f"{n_pay:>4} {n_ver:>4} {n_ship:>4} "
            f"{en.get('novel_true', 0):>7} {len(r['known_laws']):>5}"
        )
    lines.append("-" * len(head))
    lines.append(
        f"{'TOTAL':<30} {'':<10} {tot['cand']:>5} {tot['fire']:>5} "
        f"{tot['pay']:>4} {'':>4} {tot['ship']:>4}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the scaled discovery sweep; print the honest table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen", default=str(_DEFAULT_SCREEN))
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument(
        "--floor",
        type=float,
        default=0.95,
        help="minimum lhs_cover (skips the ~0%% fused-op class)",
    )
    ap.add_argument(
        "--per-model",
        type=float,
        default=240.0,
        help="wall-clock seconds per model before 'truncated'",
    )
    ap.add_argument(
        "--total-budget",
        type=float,
        default=3600.0,
        help="seconds for the whole sweep phase",
    )
    ap.add_argument(
        "--min-avail-gb",
        type=float,
        default=1.5,
        help="abort below this MemAvailable",
    )
    ap.add_argument("--max-candidates", type=int, default=400)
    ap.add_argument("--evaluate", type=int, default=200)
    ap.add_argument("--json", default=str(_DEFAULT_JSON))
    ap.add_argument(
        "--skip-novelty",
        action="store_true",
        help="skip the corpus strict-match novelty sweep",
    )
    ap.add_argument(
        "--novelty-only",
        metavar="RESULTS_JSON",
        help="skip the sweep; load per-model results from this JSON "
        "and run only the corpus novelty phase",
    )
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    if args.novelty_only:
        payload = json.loads(Path(args.novelty_only).read_text())
        results = (
            payload["results"] if isinstance(payload, dict) else payload
        )
        novelty = _novelty(results, args)
        print()
        print(_table(results))
        if args.json:
            payload["novelty"] = novelty
            Path(args.json).write_text(
                json.dumps(payload, indent=2, default=str) + "\n"
            )
            print(f"\nwrote {args.json}")  # stdout-compat
        return 0
    results = _sweep(args)
    novelty = [] if args.skip_novelty else _novelty(results, args)
    print()
    print(_table(results))
    if novelty:
        print()
        print("-- paying candidates (novelty) --")
        for n in novelty:
            print(
                f"  {n['name']} on {n['minted_on']}: "
                f"{n['corpus_models']} corpus models, "
                f"derivable={n['derivable']}"
            )
    if args.json:
        payload = {
            "results": results,
            "novelty": novelty,
            "done_excluded": sorted(DONE),
        }
        Path(args.json).write_text(
            json.dumps(payload, indent=2, default=str) + "\n"
        )
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    sys.exit(main())

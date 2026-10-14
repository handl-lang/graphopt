"""Per-model law discovery — run the admission pipeline on one graph.

``tools/exposure_screen.py`` ranks the corpus by *e-graph-reachable*
structure.  This tool is the follow-through: given workload names, it
runs the discovery machinery **scoped to each model's own term** and
reports what the machine finds there — the honest harvest, including
the nulls.  Three arms, all shipped machinery:

* **pipeline** — the shipped admission loop scoped to the model:
  ``census.op_tuple_census`` on the term feeds ``pipeline.propose``
  (census-naturality + mixed-view + pattern-recognition + shape-aware
  + grammar), and every candidate goes through ``pipeline.measure``
  unchanged — the real gates: numeric oracle → derivability →
  view oracle → lone-rule fire → typedness audit → reach/cost →
  certificate → closure safety → ``shippable``.
* **enumerate** — ``enumerate_candidates`` over the model's own op
  vocabulary (the ops the enumerator can spell — declared arity 1/2
  plus the matmul/pow overrides), ``grammar._evaluate``'s bounded
  oracle, then ``impact._probe`` on every *new + true* candidate:
  does the machine-found law actually fire on this graph, does it
  pay, does the lowered module verify?
* **known-laws** — ``impact.new_laws`` (the 11 machine-found laws the
  proposal round minted and never shipped) probed on the model —
  the "already found, not shipped" baseline the new harvest is
  compared against.

Budgets are explicit: ``--max-candidates`` caps the enumerator pool,
``--evaluate`` the bounded-oracle prefix, and every e-graph run uses
the impact tool's firing/reach budgets.  On an 11 GB box that keeps
the whole sweep under a few minutes and far from the quartic
cond/if blow-up the unbounded enumerator hit before.

Run::

    .venv/bin/python tools/exposed_discovery.py LoRAAdapter \
        GatedAttentionUnit DeepEquilibrium
    .venv/bin/python tools/exposed_discovery.py --all-zoo --json /tmp/d.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import catopt_carriers  # noqa: F401 — registers the carrier machinery
import torch
from catopt_core.ir import Op, op_repr
from catopt_core.laws import ALL_RULES
from catopt_core.opmeta import arity as _op_arity
from catopt_discovery import pipeline as pipe
from catopt_discovery import proposal as lp
from catopt_discovery.census import CorpusTerm, op_tuple_census
from catopt_discovery.enumerate import (
    _ARITY_OVERRIDES,
    enumerate_candidates,
)
from catopt_discovery.grammar import _evaluate as _bounded_evaluate
from catopt_discovery.impact import (
    TermCase,
    _cost_fn,
    _iter_subterms,
    _probe,
    new_laws,
    reach_row,
)
from catopt_discovery.shape_proposal import _sink
from exposure_screen import _export, _workloads


def _vocab_of(term: Any) -> tuple:
    """Return the enumerator-usable op vocabulary inside *term*.

    An op is usable when the enumerator can spell it — a declared
    arity of 1 or 2, or one of its own overrides (``matmul``/``pow``)
    — AND its bare (attr-free) spelling mints a valid ``Op``.  Attr
    schemas that reject a bare call are reported to stderr and
    skipped, not crashed on.
    """
    import sys as _sys

    ops = sorted(
        {s.op for s in _iter_subterms(term) if isinstance(s, Op)}
    )
    out: list[str] = []
    for op in ops:
        a = _ARITY_OVERRIDES.get(op, _op_arity(op))
        if a not in (1, 2):
            continue
        try:
            Op.make(op, *("X",) * a)
        except Exception as e:
            print(  # stdout-compat
                f"   note: {op} declines a bare spelling "
                f"({type(e).__name__}) — skipped",
                file=_sys.stderr,
            )
            continue
        out.append(op)
    return tuple(out)


def _cases_by_name(names: list[str] | None) -> dict[str, TermCase]:
    """Export the requested workloads (or the whole zoo) to cases."""
    want = set(names) if names else None
    out: dict[str, TermCase] = {}
    for source, wl in _workloads("zoo" if want is None else "all"):
        if want is not None and wl.name not in want:
            continue
        _row, case = _export(wl, source)
        if case is not None:
            out[wl.name] = case
        else:
            print(  # stdout-compat
                f"   {source}:{wl.name} — rejected: {_row.note}",
                file=sys.stderr,
            )
    if want is not None:
        for m in sorted(want - set(out)):
            print(  # stdout-compat
                f"   {m}: not found or failed export",
                file=sys.stderr,
            )
    return out


def _pipeline_arm(
    case: TermCase, sink: Any, cost_fn: Any, lib: list
) -> list:
    """Run propose + measure + rank scoped to *case*'s term."""
    ct = CorpusTerm(case.source, case.name, case.term)
    counts, _ = op_tuple_census([ct])
    census_op = dict(counts)
    props = pipe.propose(census_op, [case.term], "derived")
    base_rules = list(ALL_RULES)
    evs = [
        pipe.measure(
            p,
            [case.term],
            [case],
            base_rules,
            lib,
            census_op,
            sink,
            cost_fn,
        )
        for p in props
    ]
    return pipe.rank(evs)


def _enumerate_arm(
    case: TermCase,
    args: argparse.Namespace,
    sink: Any,
    cost_fn: Any,
) -> dict:
    """Enumerate candidates on the model's vocab; probe the new+true."""
    vocab = _vocab_of(case.term)
    if not vocab:
        return {"vocab": (), "emitted": 0, "outcomes": []}
    cands = []
    for c in enumerate_candidates(
        ops=vocab, max_candidates=args.max_candidates
    ):
        cands.append(c)
        if len(cands) >= args.max_candidates:
            break
    outcomes = _bounded_evaluate(cands[: args.evaluate])
    novel = [o for o in outcomes if o.new and o.truth]
    probes = []
    for o in novel:
        rule = o.candidate.as_rule()
        f = _probe(case, rule, sink, cost_fn)
        probes.append(
            {
                "label": o.candidate.label,
                "lhs": op_repr(o.candidate.lhs),
                "rhs": op_repr(o.candidate.rhs),
                "relation": o.relation,
                "num_true": o.num_true,
                "derivable": o.derivable,
                "useful_on_lhs": o.useful,
                "fires": f.fires,
                "changed": f.changed,
                "paid": f.paid,
                "verified": f.verified,
                "base_cost": f.base_cost,
                "out_cost": f.out_cost,
                "note": f.note,
            }
        )
    paying = [
        o.candidate.as_rule()
        for o, p in zip(novel, probes, strict=True)
        if p["fires"] and p["paid"] and p["verified"] == "pass"
    ]
    reach = reach_row(case, paying, sink, cost_fn) if paying else None
    return {
        "vocab": vocab,
        "emitted": len(cands),
        "evaluated": len(outcomes),
        "novel_true": len(novel),
        "probes": probes,
        "paying_rules": [r.name for r in paying],
        "reach": reach,
    }


def _ev_row(ev: Any) -> dict:
    """Compact an ``Evidence`` for the report."""
    return {
        "name": ev.proposal.name,
        "family": ev.proposal.family,
        "sources": list(ev.proposal.sources),
        "relation": ev.relation,
        "num_true": ev.num_true,
        "derivable": ev.derivable,
        "view_verdict": ev.view_verdict,
        "view_guard": ev.view_guard,
        "matches": ev.matches,
        "fires": ev.fires,
        "fires_typed": ev.fires_typed,
        "fires_ill_typed": ev.fires_ill_typed,
        "paid": ev.paid,
        "paid_ill_typed": ev.paid_ill_typed,
        "verify_fail": ev.verify_fail,
        "cost_drop": ev.cost_drop,
        "cert_fail": ev.cert_fail,
        "closure_ratio": ev.closure_ratio,
        "shippable": ev.shippable,
        "no_ship_reason": ev.no_ship_reason,
        "fire_cases": list(ev.fire_cases),
        "lhs": op_repr(ev.proposal.lhs),
        "rhs": op_repr(ev.proposal.rhs),
    }


def _known_arm(case: TermCase, sink: Any, cost_fn: Any) -> list[dict]:
    """Probe the 11 already-found (unshipped) laws on *case*."""
    rows = []
    for law in new_laws():
        f = _probe(case, law, sink, cost_fn)
        if f.fires:
            rows.append(
                {
                    "law": law.name,
                    "fires": f.fires,
                    "changed": f.changed,
                    "paid": f.paid,
                    "verified": f.verified,
                    "base_cost": f.base_cost,
                    "out_cost": f.out_cost,
                    "note": f.note,
                }
            )
    return rows


def discover(
    case: TermCase, args: argparse.Namespace
) -> dict[str, Any]:
    """Run all three arms on one case; return the harvest."""
    sink = _sink()
    cost_fn = _cost_fn(sink)
    lib = [lp._key(r.lhs, r.rhs) for r in ALL_RULES]
    t0 = time.time()
    ranked = _pipeline_arm(case, sink, cost_fn, lib)
    enum = _enumerate_arm(case, args, sink, cost_fn)
    known = _known_arm(case, sink, cost_fn)
    return {
        "model": case.name,
        "source": case.source,
        "n_nodes": sum(
            1 for s in _iter_subterms(case.term) if isinstance(s, Op)
        ),
        "elapsed_s": round(time.time() - t0, 1),
        "pipeline": [_ev_row(e) for e in ranked],
        "enumerate": enum,
        "known_laws": known,
    }


def _print_model(res: dict, top: int) -> None:
    """Print one model's discovery report."""
    print(  # stdout-compat
        f"=== {res['model']} ({res['source']}, "
        f"{res['n_nodes']} nodes, {res['elapsed_s']}s) ==="
    )
    evs = res["pipeline"]
    ship = [e for e in evs if e["shippable"]]
    fired = [e for e in evs if e["fires"]]
    print(  # stdout-compat
        f"  pipeline: {len(evs)} candidates, {len(fired)} fire, "
        f"{sum(1 for e in evs if e['paid'])} paid, "
        f"{len(ship)} shippable"
    )
    interesting = [
        e
        for e in evs
        if e["fires"] and e["relation"] == "new" and e["num_true"]
    ]
    for e in (ship or interesting or fired)[:top]:
        print(  # stdout-compat
            f"    {e['name']:<30} rel={e['relation']:<9} "
            f"true={e['num_true']} fires={e['fires']} "
            f"paid={e['paid']} drop={e['cost_drop'] * 100:.1f}% "
            f"{'SHIP' if e['shippable'] else e['no_ship_reason'][:50]}"
        )
        if e["lhs"] != e["rhs"]:
            print(  # stdout-compat
                f"        {e['lhs']}  =  {e['rhs']}"
            )
    en = res["enumerate"]
    print(  # stdout-compat
        f"  enumerate: vocab={list(en['vocab'])}, "
        f"{en['emitted']} emitted, {en.get('evaluated', 0)} evaluated, "
        f"{en.get('novel_true', 0)} new+true"
    )
    for p in en["probes"]:
        if p["fires"]:
            print(  # stdout-compat
                f"    {p['label']:<22} {p['lhs']} = {p['rhs']}  "
                f"fires={p['fires']} paid={p['paid']} "
                f"verify={p['verified']} {p['note'][:40]}"
            )
    if en.get("reach"):
        r = en["reach"]
        print(  # stdout-compat
            f"    reach(+{len(en['paying_rules'])} new): "
            f"cost {r['base_cost']:.4g}→{r['add_cost']:.4g} "
            f"cert={r['add_cert']}"
        )
    if res["known_laws"]:
        print("  known (unshipped) laws firing:")  # stdout-compat
        for k in res["known_laws"]:
            print(  # stdout-compat
                f"    {k['law']:<22} fires={k['fires']} "
                f"paid={k['paid']} verify={k['verified']}"
            )


def main(argv: list[str] | None = None) -> int:
    """Run discovery on the named workloads; report the harvest."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="*", help="workload names")
    ap.add_argument(
        "--all-zoo", action="store_true", help="run the whole zoo"
    )
    ap.add_argument("--max-candidates", type=int, default=400)
    ap.add_argument(
        "--evaluate",
        type=int,
        default=200,
        help="bounded-oracle prefix of the emitted candidates",
    )
    ap.add_argument(
        "--top", type=int, default=12, help="pipeline rows to print"
    )
    ap.add_argument("--json", help="write machine-readable results")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    names = None if args.all_zoo else args.models
    cases = _cases_by_name(names)
    print(  # stdout-compat
        f"== exposed_discovery — {len(cases)} model(s) =="
    )
    results = []
    for name, case in cases.items():
        t0 = time.time()
        try:
            res = discover(case, args)
        except Exception as e:  # honest per-model failure
            res = {
                "model": name,
                "error": f"{type(e).__name__}: {e}",
            }
        results.append(res)
        if "error" in res:
            print(f"=== {name}: ERROR {res['error']} ===")
        else:
            _print_model(res, args.top)
        if time.time() - t0 > 600:
            print("(wall-clock budget: stopping after this model)")
            break

    if args.json:
        Path(args.json).write_text(
            json.dumps(results, indent=2, default=str) + "\n"
        )
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    sys.exit(main())

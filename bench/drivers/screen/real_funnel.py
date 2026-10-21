"""Where do discovery candidates die on strictly-real modules?

``project/retros/real-corpus-yield.md`` measured ``shippable: 0`` on
the real half of the intake corpus and named the residual signal —
nine proposals fire *and pay* on real modules but are refused at
``truth`` as conditional equalities — without producing the exact
per-stage funnel.  This driver makes the funnel exact and then runs
the named lever: the ``auto_cond`` → ``run_gauntlet`` admission path,
re-measured on a gauntlet corpus that *includes* the real intake
terms (the auto-cond retro's corpus had none of the real sites in
scope, so its ``fires=0`` typed-pay refusals could not see them).

Stages:

1. ``corpus`` — load the intake side-file (``--intake-json`` /
   ``--tensors``, produced by ``python -m mlopt_discovery.intake
   --skip-pipeline --out …``), split the cases by provenance
   (``_PURPOSE_BUILT`` vs real), and ingest the held-out ``zoo``.
2. ``funnel`` — run ``workload_gen._run_pipeline`` (the same
   census → propose → measure → rank mirror the intake driver uses)
   on four arms: baseline (bench+models), +real intake,
   +purpose-built, +zoo.  ``pipeline._probe`` is wrapped to keep the
   per-(rule, case) firing record, so every ``paid`` count is
   attributable to a named site.  Each candidate's *first failed
   gate* in ``Evidence.shippable`` order is the death stage.
3. ``auto-cond`` — every candidate that dies at ``truth`` with a real
   match is retried through ``object_synthesis.auto_cond_object``
   (measured domain = synth windows + real matches), and each minted
   guard's accepted region is swept over the real (and zoo) sites via
   ``evidence._guarded_evals`` — "is the guarded region inhabited on
   real code?".
4. ``gauntlet`` — candidates whose minted guard accepts an equal real
   site are stored and re-run through ``run_gauntlet`` on a
   :class:`GauntletCorpus` whose ``probe`` includes the real intake
   cases — the full admission path, unchanged.

Run::

    .venv/bin/python -m mlopt_discovery.intake --skip-pipeline \
        --out /tmp/intake.json --tensors /tmp/intake_tensors.pt
    .venv/bin/python bench/drivers/screen/real_funnel.py \
        --intake-json /tmp/intake.json --tensors /tmp/intake_tensors.pt \
        --json /tmp/real_funnel.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
#  Corpus assembly
# ---------------------------------------------------------------------------


def _bucket(name: str, purpose: set, zoo: set) -> str:
    """Provenance bucket of a probe-case name."""
    short = name.removeprefix("intake:")
    if short in purpose:
        return "purpose"
    if short in zoo:
        return "zoo"
    if name.startswith("intake:"):
        return "real-intake"
    return "baseline"


def _eligible(cases: list, records: list[dict]) -> list:
    """Probe-eligible subset: ingested + under the node cap."""
    from mlopt_discovery.intake.ledger import _PROBE_MAX_NODES

    meta = {r["name"]: (r["status"], r["n_op_nodes"]) for r in records}
    return [
        c
        for c in cases
        if meta.get(c.name, ("", 0))[0] == "ingested"
        and meta[c.name][1] <= _PROBE_MAX_NODES
    ]


def _load_corpus(args: argparse.Namespace) -> dict:
    """Assemble the baseline + per-arm case lists."""
    from mlopt_discovery import intake as li
    from mlopt_discovery import zoo as zoo_mod
    from mlopt_discovery.impact import _bench_cases, model_cases

    bench, _be = _bench_cases()
    models, _me = model_cases()
    intake_cases = li.load_cases(
        Path(args.intake_json), Path(args.tensors)
    )
    records = li.load_records(Path(args.intake_json))
    purpose = set(li._PURPOSE_BUILT)
    zoo_names = {w.name for w in zoo_mod.zoo()}

    real_names = {
        c.name
        for c in intake_cases
        if c.name.removeprefix("intake:") not in purpose
    }
    real = [c for c in intake_cases if c.name in real_names]
    built = [c for c in intake_cases if c.name not in real_names]
    eligible = _eligible(intake_cases, records)
    probe_real = [c for c in eligible if c.name in real_names]
    probe_built = [c for c in eligible if c.name not in real_names]

    # The held-out zoo runs the same export boundary — it is "real"
    # code by construction, just never registered in the census.
    zcases, zrecords, zrej = li.ingest(cands=list(zoo_mod.zoo()))
    zprobe = _eligible(zcases, zrecords)

    return {
        "bench": bench,
        "models": models,
        "real": real,
        "built": built,
        "probe_real": probe_real,
        "probe_built": probe_built,
        "zoo": zcases,
        "probe_zoo": zprobe,
        "zoo_rejected": [f"{r.name}:{r.stage}" for r in zrej],
        "purpose": purpose,
        "zoo_names": zoo_names,
        "records": records,
        "status": Counter(r["status"] for r in records),
        "zoo_status": Counter(r["status"] for r in zrecords),
    }


# ---------------------------------------------------------------------------
#  The funnel — first-failed gate per candidate, shippable order
# ---------------------------------------------------------------------------


def _death_stage(ev: Any) -> str:
    """First ``Evidence.shippable`` gate *ev* fails (``shipped`` if none)."""
    if not ev.truth:
        if ev.view_verdict == "conditional":
            return "truth:conditional"
        if ev.view_verdict == "ill-formed":
            return "truth:ill-formed"
        if ev.num_true is False:
            return "truth:false"
        if ev.matches == 0:
            return "truth:no-match"
        return "truth:unproven"
    if ev.relation != "new":
        return "novelty"
    if ev.fires == 0:
        return "fires"
    if ev.fires_ill_typed:
        return "ill-typed"
    if ev.paid == 0:
        return "pay"
    if ev.verify_fail:
        return "verify"
    if ev.cert_fail:
        return "cert"
    if not ev.closure_safe:
        return "closure"
    return "shipped"


_FUNNEL_ORDER = [
    "truth:conditional",
    "truth:ill-formed",
    "truth:false",
    "truth:unproven",
    "truth:no-match",
    "novelty",
    "fires",
    "ill-typed",
    "pay",
    "verify",
    "cert",
    "closure",
    "shipped",
]


def _funnel(ranked: list) -> dict:
    """Per-candidate first-failed-gate table for one arm."""
    deaths: dict[str, list[str]] = {}
    for ev in ranked:
        deaths.setdefault(_death_stage(ev), []).append(ev.proposal.name)
    return {
        "proposals": len(ranked),
        "deaths": {k: sorted(v) for k, v in deaths.items()},
        "counts": {k: len(deaths.get(k, [])) for k in _FUNNEL_ORDER},
    }


def _run_arm(
    corpus_cases: list,
    probe_models: list,
    vocab: str,
    holdout: str | None,
    bucket_of: Any,
) -> dict:
    """Run the pipeline mirror; return evidence + per-case probe rows."""
    from mlopt_discovery import pipeline as lpipe
    from mlopt_discovery import workload_gen as lwg
    from mlopt_discovery.impact import _probe as orig_probe

    seen: dict[tuple[str, str], Any] = {}

    def _watch(case: Any, rule: Any, sink: Any, cost_fn: Any) -> Any:
        f = orig_probe(case, rule, sink, cost_fn)
        seen[(rule.name, case.name)] = f
        return f

    saved = lpipe._probe
    lpipe._probe = _watch
    try:
        res = lwg._run_pipeline(
            corpus_cases, probe_models, vocab, holdout
        )
    finally:
        lpipe._probe = saved

    rows = []
    for ev in res["ranked"]:
        p = ev.proposal
        fire_cases = list(ev.fire_cases)
        pay_cases = sorted(
            n for (r, n), f in seen.items() if r == p.name and f.paid
        )
        rows.append(
            {
                "name": p.name,
                "family": p.family,
                "rule": p.as_rule(),
                "stage": _death_stage(ev),
                "matches": ev.matches,
                "relation": ev.relation,
                "num_true": ev.num_true,
                "derivable": ev.derivable,
                "view_verdict": ev.view_verdict,
                "view_guard": ev.view_guard,
                "fires": ev.fires,
                "fires_typed": ev.fires_typed,
                "fires_ill_typed": ev.fires_ill_typed,
                "paid": ev.paid,
                "paid_ill_typed": ev.paid_ill_typed,
                "verify_fail": ev.verify_fail,
                "cert_fail": ev.cert_fail,
                "closure_ratio": ev.closure_ratio,
                "cost_drop": ev.cost_drop,
                "fire_cases": fire_cases,
                "pay_cases": pay_cases,
                "fire_buckets": dict(
                    Counter(bucket_of(n) for n in fire_cases)
                ),
                "pay_buckets": dict(
                    Counter(bucket_of(n) for n in pay_cases)
                ),
                "shippable": ev.shippable,
                "reason": ev.no_ship_reason,
            }
        )
    return {
        "n_terms": res["n_terms"],
        "n_tuples": res["n_tuples"],
        "n_probe": len(probe_models),
        "funnel": _funnel(res["ranked"]),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
#  The auto-cond retry — does the conditional blocker admit on real code?
# ---------------------------------------------------------------------------


def _real_sites(rule: Any, terms: list) -> list:
    """Return ``(env, lhs)`` pairs for every real match of *rule*."""
    from mlopt_core.egraph.terms import _term_match
    from mlopt_discovery.shape_proposal import Schema, real_matches

    schema = Schema(rule.name, rule.lhs, rule.rhs)
    sites = []
    for m in real_matches(list(terms), schema):
        subst = _term_match(rule.lhs, m)
        if subst is not None:
            sites.append((subst, m))
    return sites


def _autocond_sweep(
    cands: list, corpus_terms: list, real_terms: list, zoo_terms: list
) -> list[dict]:
    """Mint guards for truth-refused candidates; sweep real regions."""
    from mlopt_discovery import evidence as evd
    from mlopt_discovery import object_synthesis as obs

    out = []
    for row in cands:
        name = row["name"]
        t0 = time.time()
        torch.manual_seed(0)
        res = obs.auto_cond_object(
            row["rule"], corpus_terms=corpus_terms, synth_limit=360
        )
        rec: dict[str, Any] = {
            "name": name,
            "pipeline_stage": row["stage"],
            "paid": row["paid"],
            "fires": row["fires"],
            "found": res.object is not None,
            "detail": res.detail,
            "measured": res.measured,
            "equal": res.equal,
            "bad": res.bad,
            "unstable": res.unstable,
            "accepted": res.accepted,
            "elapsed_s": round(time.time() - t0, 1),
        }
        if res.object is not None:
            rule = res.object.rule
            rec["cond"] = repr(res.cond)
            for tag, terms in (
                ("real", real_terms),
                ("zoo", zoo_terms),
            ):
                sites = _real_sites(rule, terms)
                reg = evd._guarded_evals(rule, iter(sites))
                rec[f"{tag}_sites"] = len(sites)
                rec[f"{tag}_region"] = {
                    "accepted": reg.accepted,
                    "declined": reg.declined,
                    "equal": reg.equal,
                    "unequal": reg.unequal,
                    "rhs_err": reg.rhs_err,
                    "other_err": reg.other_err,
                }
            rec["minted_rule"] = rule
        out.append(rec)
        print(  # stdout-compat
            f"    auto-cond {name:<30} found={rec['found']} "
            f"eq={res.equal} bad={res.bad} "
            f"real_reg={rec.get('real_region', '-')} "
            f"({rec['elapsed_s']}s)",
            flush=True,
        )
    return out


def _gauntlet_retry(recs: list[dict], corpus: Any) -> list[dict]:
    """Run the armed admission gauntlet on minted candidates."""
    from mlopt_discovery import evidence as evd

    out = []
    for rec in recs:
        conn = evd.connect(":memory:")
        try:
            key = evd.store_object(
                conn, rec["bare_rule"], kind="abstraction"
            )
            t0 = time.time()
            rep = evd.run_gauntlet(
                conn, key, corpus=corpus, auto_cond=True
            )
            out.append(
                {
                    "name": rec["name"],
                    "usable": rep.usable,
                    "stages": [
                        {
                            "name": s.name,
                            "passed": s.passed,
                            "detail": s.detail,
                        }
                        for s in rep.stages
                    ],
                    "auto_cond": rep.auto_cond,
                    "elapsed_s": round(time.time() - t0, 1),
                }
            )
            print(  # stdout-compat
                f"    gauntlet {rec['name']:<30} usable={rep.usable} "
                f"({out[-1]['elapsed_s']}s)",
                flush=True,
            )
        finally:
            conn.close()
    return out


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def _clean(o: Any) -> Any:
    """Strip rule objects / non-JSON values from the payload."""
    if isinstance(o, dict):
        return {
            k: _clean(v)
            for k, v in o.items()
            if k not in ("rule", "minted_rule", "bare_rule", "_rule")
        }
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return repr(o)


def _checkpoint(path: str | None, payload: dict) -> None:
    """Persist the payload-so-far — the funnel arms are expensive."""
    if path:
        Path(path).write_text(
            json.dumps(_clean(payload), indent=2) + "\n"
        )


def main(argv: list[str] | None = None) -> int:
    """Measure the real-corpus funnel and retry the conditional blocker."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--intake-json", default="/tmp/full_intake_corpus.json")
    ap.add_argument("--tensors", default="/tmp/full_intake_tensors.pt")
    ap.add_argument("--vocab", default="derived")
    ap.add_argument("--holdout", default=None)
    ap.add_argument(
        "--stages",
        default="funnel,autocond,gauntlet",
        help="comma list: funnel,autocond,gauntlet",
    )
    ap.add_argument("--json", help="write machine-readable results")
    args = ap.parse_args(argv)
    stages = set(args.stages.split(","))

    torch.manual_seed(0)
    print("== real_funnel — where discovery dies on real modules ==")
    c = _load_corpus(args)
    print(
        f"   intake: {len(c['real'])} real + {len(c['built'])} "
        f"purpose-built; probe-eligible {len(c['probe_real'])} + "
        f"{len(c['probe_built'])}; statuses {dict(c['status'])}"
    )
    print(
        f"   zoo: {len(c['zoo'])} exported "
        f"({dict(c['zoo_status'])}), probe-eligible "
        f"{len(c['probe_zoo'])}; rejected {c['zoo_rejected']}"
    )

    purpose, zoo_names = c["purpose"], c["zoo_names"]
    bucket_of = lambda n: _bucket(n, purpose, zoo_names)  # noqa: E731

    base = [*c["bench"], *c["models"]]
    payload: dict[str, Any] = {
        "corpus": {
            "baseline_terms": len(base),
            "real_intake": len(c["real"]),
            "purpose_built": len(c["built"]),
            "zoo": len(c["zoo"]),
            "probe_real": len(c["probe_real"]),
            "probe_built": len(c["probe_built"]),
            "probe_zoo": len(c["probe_zoo"]),
            "intake_status": dict(c["status"]),
            "zoo_status": dict(c["zoo_status"]),
        }
    }

    real_ranked_rows: list[dict] = []
    if "funnel" in stages:
        arms = {
            "baseline": (base, c["models"]),
            "real": ([*base, *c["real"]], [*c["models"], *c["probe_real"]]),
            "purpose": (
                [*base, *c["built"]],
                [*c["models"], *c["probe_built"]],
            ),
            "zoo": ([*base, *c["zoo"]], [*c["models"], *c["probe_zoo"]]),
        }
        payload["arms"] = {}
        for tag, (cc, pm) in arms.items():
            t0 = time.time()
            print(f"-- arm {tag}: {len(cc)} terms, {len(pm)} probe --")
            payload["arms"][tag] = _run_arm(
                cc, pm, args.vocab, args.holdout, bucket_of
            )
            _checkpoint(args.json, payload)
            fc = payload["arms"][tag]["funnel"]
            print(
                f"   proposals={fc['proposals']} "
                f"shipped={fc['counts']['shipped']} "
                f"({time.time() - t0:.0f}s)"
            )
        real_ranked_rows = payload["arms"]["real"]["rows"]

        # the funnel table
        print()
        stages_order = [s for s in _FUNNEL_ORDER]
        head = f"{'first-failed gate':<20}" + "".join(
            f"{a:>10}" for a in arms
        )
        print(head)
        print("-" * len(head))
        for s in stages_order:
            print(
                f"{s:<20}"
                + "".join(
                    f"{payload['arms'][a]['funnel']['counts'][s]:>10}"
                    for a in arms
                )
            )
        print(
            f"{'proposals':<20}"
            + "".join(
                f"{payload['arms'][a]['funnel']['proposals']:>10}"
                for a in arms
            )
        )
        print()
        print("-- candidates past `fires` on the real arm --")
        for r in real_ranked_rows:
            if r["stage"] in ("fires", "truth:no-match"):
                continue
            print(
                f"  {r['name']:<32} {r['stage']:<18} fires={r['fires']} "
                f"paid={r['paid']} pay_cases={r['pay_cases']} "
                f"guard={r['view_guard'] or '-'}"
            )

    if ("autocond" in stages or "gauntlet" in stages) and (
        "funnel" not in stages
    ):
        print(
            "\n   (--stages without `funnel`: no real-arm rows — "
            "auto-cond/gauntlet skipped)"
        )
    elif "autocond" in stages or "gauntlet" in stages:
        from mlopt_core.laws import ALL_RULES
        from mlopt_discovery import evidence as evd
        from mlopt_discovery import pipeline as pl
        from mlopt_discovery.census import run_census
        from mlopt_discovery.impact import _cost_fn
        from mlopt_discovery.shape_proposal import _sink

        # candidates that die at truth on the real arm but have a real
        # match to measure — the blocker the retro named.
        truth_died = [
            r
            for r in real_ranked_rows
            if r["stage"].startswith("truth:") and r["matches"] > 0
        ]
        print()
        print(
            f"-- auto-cond retry: {len(truth_died)} truth-refused "
            "candidates with real matches --"
        )
        real_terms = [t.term for t in c["real"]]
        zoo_terms = [t.term for t in c["zoo"]]
        corpus_terms = [
            t.term for t in [*c["bench"], *c["models"], *c["real"]]
        ]
        ac = _autocond_sweep(
            truth_died, corpus_terms, real_terms, zoo_terms
        )
        payload["autocond"] = [
            {
                k: v
                for k, v in r.items()
                if k not in ("rule", "minted_rule", "bare_rule")
            }
            for r in ac
        ]
        _checkpoint(args.json, payload)
        n_found = sum(1 for r in ac if r["found"])
        inhabited = [
            r
            for r in ac
            if r["found"] and r.get("real_region", {}).get("equal", 0) > 0
        ]
        print(
            f"   minted: {n_found}/{len(ac)}; "
            f"guarded region inhabited on real terms: {len(inhabited)}"
        )

        if "gauntlet" in stages and inhabited:
            census = run_census(pl._CENSUS_TOP)
            census_op = {
                (e["op"], tuple(e["children"])): e["count"]
                for e in census["op_tuples"]
            }
            sink = _sink()
            corpus = evd.GauntletCorpus(
                real_terms=tuple(corpus_terms),
                probe=tuple([*c["models"], *c["probe_real"]]),
                base_rules=tuple(ALL_RULES),
                census_op=census_op,
                sink=sink,
                cost_fn=_cost_fn(sink),
            )
            # store the bare pattern so the armed gauntlet mints it
            for r in inhabited:
                src = next(
                    x for x in truth_died if x["name"] == r["name"]
                )
                r["bare_rule"] = src["rule"]
            payload["gauntlet"] = _gauntlet_retry(inhabited, corpus)
            usable = [g["name"] for g in payload["gauntlet"] if g["usable"]]
            print(f"   admitted: {usable or 'none'}")

    if args.json:
        _checkpoint(args.json, payload)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The agenda miner — Stage-3 self-driving, first cut.

The system measures its own headroom and files research proposals.
This driver scans the *existing* measurement surfaces — the overnight
ledgers, the frontier/sweep/pay-measured report JSONs, and the
meta-arena coverage census — and emits **ranked proposals**: each
with the measured evidence, a suggested lever, an estimated
headroom, and a confidence.  Proposals only — nothing here
auto-executes.  Humans approve (``myco todo add``), the system does
the driving; closing that loop is the trust boundary and is
deliberately out of scope for this cut.

Signals mined (each is a pure function over report JSONs — cheap,
re-runnable, no torch):

* ``ledger_pay_decline`` — overnight-ledger candidates that passed
  oracle but were declined at ``pay`` *after changing the graph*
  (``changed>0, paid=0, verify_fail=0`` — correctness declines are
  split out), ranked by how close the measured ``exec``
  ratio sits to the keep bar.  These are laws the gauntlet almost
  admitted — the headroom is pricing/codegen, not search.
* ``ledger_verify_decline`` — candidates that changed the graph but
  failed verification (``verify_fail>0``).  Correctness declines —
  the lever is guard/denotation repair, not pricing.
* ``ledger_oracle_gap`` — candidate *families* whose stream dies
  almost entirely at ``oracle`` with no equal site measured.  The
  mint is producing claims the corpus cannot evaluate — corpus/env
  headroom, not law headroom.
* ``ledger_vacuous`` — candidates classified ``vacuous`` (no binding
  produced a site): the prior-art gates are unevaluable, so novelty
  is unmeasured headroom.
* ``frontier_delivery_gap`` — ``frontier*.json`` rows with material
  algebraic headroom (``headroom_us``) the delivered body fails to
  capture (``captured`` null/low, or ``beats_compile`` false).  The
  ceiling is real; the delivery is the gap.
* ``sweep_straddle`` — ``zoo_chain_sweep*.json`` rows whose per-run
  speedups straddle 1.0: the keep/drop outcome is decided by
  referee noise, not by the workload.  Measurement-integrity
  headroom.
* ``sweep_delivered_loser`` — delivered+verified rows whose median
  speedup is below parity: claims shipped that lose to eager.  The
  keep threshold has headroom.
* ``paymeasured_confound`` — ``pay_measured*.json`` rows verdict
  ``confounded``: the marginal compares different executors, so the
  measurement itself is the gap.
* ``coverage_*`` — :func:`mlopt_discovery.corpus_cov.coverage_gaps`
  over a fresh base+coverage case stream: ops/attrs/handlers/rules
  the arena vocabulary expresses but no case exercises.  Corpus
  headroom — the Stage-1 lever-1 residual.

Outputs (``--out-prefix``, default ``docs/reports/agenda``):

* ``<prefix>.jsonl`` — one proposal per line, the machine-readable
  agenda (schema below).
* ``<prefix>.md`` — the human-readable agenda: ranked table, per-
  signal evidence, and the ``myco todo add`` ingest commands.

Proposal record::

    {
      "id": "agenda-<signal>-<slug>",
      "kind": "agenda-proposal",
      "title": "...",                # myco-safe: no '·', no newlines
      "target": {"domain": ..., "tier": ..., "op": ...},
      "signal": "<signal-name>",
      "evidence": {"sources": [...], "metrics": {...}},
      "lever": "suggested next step",
      "headroom": 0.0-1.0,           # estimated share of the gap
      "confidence": "high|medium|low",
      "score": headroom * conf_w     # the ranking key
    }

Usage::

    .venv/bin/python bench/drivers/agenda/agenda_miner.py
    .venv/bin/python bench/drivers/agenda/agenda_miner.py --cases 800
    .venv/bin/python bench/drivers/agenda/agenda_miner.py --no-coverage
    .venv/bin/python bench/drivers/agenda/agenda_miner.py --myco   # ingest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: The repo root — ``bench.*`` imports are function-level (the
#: overnight.py convention): standalone runs put the driver dir on
#: ``sys.path`` and :func:`main` inserts the root first.
_REPO = Path(__file__).resolve().parents[3]

_DEFAULT_REPORTS = _REPO / "docs" / "reports"

#: Pay-stage detail grammar — ``fires=N changed=N paid=N
#: verify_fail=N cases=M`` plus optional ``| count=Xx exec=Xx …``
#: ratios.  ``exec`` is kernel-time/spelled-time: <1 pays, ≥1 declines.
_PAY_RE = re.compile(
    r"fires=(\d+) changed=(\d+) paid=(\d+) verify_fail=(\d+)"
    r" cases=(\d+)(?:\s*\|\s*(.*))?"
)
_RATIO_RE = re.compile(r"(\w+)=([\d.]+)x")

#: Oracle-stage detail grammar — ``real: Eeq/Ene/Ererr (A acc, D
#: dec, N envs) | synth: …``.  ``eq`` is the equal-site count the
#: admission gate needs; ``dec`` is declines (expected-negative
#: mass, not a gap).
_ORACLE_RE = re.compile(
    r"(\d+)eq/(\d+)ne/(\d+)rerr\s*\((\d+) acc,\s*(\d+) dec,\s*(\d+) envs\)"
)

_CONFIDENCE_W = {"high": 1.0, "medium": 0.6, "low": 0.3}


# ---------------------------------------------------------------------------
#  Proposal plumbing
# ---------------------------------------------------------------------------


def _slug(text: str, n: int = 48) -> str:
    """Filesystem/myco-safe slug — lowercase, ``-``-joined.

    Truncated slugs get a short content hash appended: two titles
    differing only past char ``n`` (e.g. the ``..._w_6@``/``_w_10@``
    law names) must not mint the same proposal id — the ingest dedup
    keys on it.
    """
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    out = s[:n].strip("-") or "x"
    if len(s) > n:
        out += "-" + hashlib.sha1(text.encode()).hexdigest()[:6]
    return out


def _proposal(
    signal: str,
    title: str,
    *,
    target: dict | None = None,
    evidence: dict | None = None,
    lever: str,
    headroom: float,
    confidence: str,
) -> dict:
    """Build one ranked proposal record (headroom clipped to [0,1])."""
    headroom = max(0.0, min(1.0, float(headroom)))
    return {
        "id": f"agenda-{signal}-{_slug(title)}",
        "kind": "agenda-proposal",
        "title": title,
        "target": target or {},
        "signal": signal,
        "evidence": evidence or {},
        "lever": lever,
        "headroom": round(headroom, 4),
        "confidence": confidence,
        "score": round(
            headroom * _CONFIDENCE_W.get(confidence, 0.5), 4
        ),
    }


def _load_json(path: Path) -> Any:
    """Load one report JSON; ``None`` on parse failure (honest skip)."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _reports(reports_dir: Path, pattern: str) -> list[Path]:
    """Sorted report paths matching *pattern* under *reports_dir*."""
    return sorted(reports_dir.glob(pattern))


# ---------------------------------------------------------------------------
#  Signal: overnight ledgers — pay declines, oracle gaps, vacuous
# ---------------------------------------------------------------------------


def _stage(candidate: dict, name: str) -> dict | None:
    for s in candidate.get("stages", []):
        if s.get("name") == name:
            return s
    return None


def _sig_ledgers(reports_dir: Path) -> list[dict]:
    """Mine ``results_overnight*.json`` ledgers for near-miss claims.

    Dedupes candidates by name across ledger versions (the v2..v6
    files re-run the same protocols): a candidate's *best* pay
    record wins — declined-with-change beats declined-unchanged,
    admitted beats all.  Oracle-gap accounting is per-family across
    every ledger that ran the family.
    """
    # name -> best record seen
    best: dict[str, dict] = {}
    fam_oracle: dict[str, dict] = {}
    vacuous: list[dict] = []
    sources: set[str] = set()

    for path in _reports(reports_dir, "results_overnight*.json"):
        doc = _load_json(path)
        if not isinstance(doc, dict):
            continue
        rel = str(path.relative_to(_REPO))
        for cand in doc.get("candidates") or []:
            name = cand.get("name") or "?"
            fam = cand.get("family") or "ungrouped"
            sources.add(rel)
            rec = {
                "name": name,
                "family": fam,
                "source": rel,
                "corpus": doc.get("corpus"),
                "admitted": bool(cand.get("admitted")),
                "novelty": cand.get("novelty") or "",
                "arena": cand.get("arena") or {},
                "pay": None,
                "oracle": None,
                "failed_leg": None,
            }
            for s in cand.get("stages", []):
                if not s.get("passed"):
                    rec["failed_leg"] = s.get("name")
                if s.get("name") == "pay":
                    m = _PAY_RE.search(s.get("detail") or "")
                    if m:
                        fires, changed, paid, vfail, cases, rest = (
                            m.groups()
                        )
                        ratios = {
                            k: float(v)
                            for k, v in _RATIO_RE.findall(rest or "")
                        }
                        rec["pay"] = {
                            "fires": int(fires),
                            "changed": int(changed),
                            "paid": int(paid),
                            "verify_fail": int(vfail),
                            "cases": int(cases),
                            "ratios": ratios,
                            "passed": bool(s.get("passed")),
                        }
                if s.get("name") == "oracle":
                    om = _ORACLE_RE.search(s.get("detail") or "")
                    if om:
                        eq, ne, rerr, acc, dec, envs = (
                            int(g) for g in om.groups()
                        )
                        rec["oracle"] = {
                            "eq": eq,
                            "ne": ne,
                            "rerr": rerr,
                            "acc": acc,
                            "dec": dec,
                            "envs": envs,
                            "passed": bool(s.get("passed")),
                        }
            # oracle-gap family accounting (every non-admitted cand)
            if not rec["admitted"]:
                f = fam_oracle.setdefault(
                    fam,
                    {
                        "family": fam,
                        "n": 0,
                        "oracle_fail": 0,
                        "eq_total": 0,
                        "acc_total": 0,
                        "examples": [],
                    },
                )
                f["n"] += 1
                if rec["failed_leg"] == "oracle" or (
                    rec["oracle"] and not rec["oracle"]["passed"]
                ):
                    f["oracle_fail"] += 1
                if rec["oracle"]:
                    f["eq_total"] += rec["oracle"]["eq"]
                    f["acc_total"] += rec["oracle"]["acc"]
                if len(f["examples"]) < 4:
                    f["examples"].append(name)
            if rec["novelty"] == "vacuous" and not rec["admitted"]:
                vacuous.append(rec)
            # best-record dedupe: prefer admitted, then pay-with-
            # change, then any record
            cur = best.get(name)
            score = (
                2 if rec["admitted"] else 0,
                (rec["pay"] or {}).get("changed", 0),
                (rec["pay"] or {}).get("paid", 0),
            )
            prev = (
                (
                    2 if cur["admitted"] else 0,
                    (cur.get("pay") or {}).get("changed", 0),
                    (cur.get("pay") or {}).get("paid", 0),
                )
                if cur
                else (-1, -1, -1)
            )
            if cur is None or score > prev:
                best[name] = rec

    out: list[dict] = []

    # -- pay declines that changed the graph -------------------------
    declines = [
        r
        for r in best.values()
        if r["pay"]
        and not r["pay"]["passed"]
        and r["pay"]["changed"] > 0
        and not r["admitted"]
    ]
    # verify_fail>0 rows are correctness declines, not pricing
    # near-misses — they get their own signal below.
    price_declines = [
        r for r in declines if r["pay"]["verify_fail"] == 0
    ]
    verify_declines = [
        r for r in declines if r["pay"]["verify_fail"] > 0
    ]
    price_declines.sort(
        key=lambda r: (
            -r["pay"]["changed"],
            abs((r["pay"]["ratios"].get("exec") or 9.9) - 1.0),
        )
    )
    for r in price_declines[:8]:
        pay = r["pay"]
        exec_r = pay["ratios"].get("exec")
        # closeness: exec in [0.7, 1.3] is a near miss; further out
        # the gap is structural (codegen, not pricing)
        close = exec_r is not None and 0.7 <= exec_r <= 1.3
        headroom = min(1.0, 0.35 + 0.1 * pay["changed"])
        if close:
            headroom = min(1.0, headroom + 0.25)
        ev = {
            "sources": sorted({r["source"]}),
            "metrics": {
                "fires": pay["fires"],
                "changed": pay["changed"],
                "paid": pay["paid"],
                "exec_ratio": exec_r,
                "count_ratio": pay["ratios"].get("count"),
                "flops_ratio": pay["ratios"].get("flops"),
                "launch_ratio": pay["ratios"].get("launch"),
                "cases": pay["cases"],
                "corpus": r["corpus"],
                "arena": r["arena"],
            },
        }
        out.append(
            _proposal(
                "ledger_pay_decline",
                f"Pay-declined law near the bar: {r['name']} "
                f"(changed={pay['changed']}, exec={exec_r}x)",
                target={"domain": "discovery", "op": r["name"]},
                evidence=ev,
                lever=(
                    "Re-price or re-codegen this law's payoff: it "
                    "fires and rewrites the graph but the measured "
                    "kernel doesn't clear the keep bar — check the "
                    "cost model's exec term against a hand-tuned "
                    "kernel for the same site."
                ),
                headroom=headroom,
                confidence="medium" if close else "low",
            )
        )

    # -- verify declines: changed the graph but failed verification ---
    verify_declines.sort(key=lambda r: -r["pay"]["verify_fail"])
    for r in verify_declines[:4]:
        pay = r["pay"]
        exec_r = pay["ratios"].get("exec")
        ev = {
            "sources": sorted({r["source"]}),
            "metrics": {
                "fires": pay["fires"],
                "changed": pay["changed"],
                "paid": pay["paid"],
                "verify_fail": pay["verify_fail"],
                "exec_ratio": exec_r,
                "cases": pay["cases"],
                "corpus": r["corpus"],
                "arena": r["arena"],
            },
        }
        out.append(
            _proposal(
                "ledger_verify_decline",
                f"Verify-declined law: {r['name']} "
                f"(changed={pay['changed']}, vfail={pay['verify_fail']})",
                target={"domain": "discovery", "op": r["name"]},
                evidence=ev,
                lever=(
                    "Correctness decline, not pricing: the rewrite "
                    "fires and changes the graph but fails "
                    "verification on real cases — strengthen the "
                    "law's guard/denotation before re-pricing."
                ),
                headroom=0.4,
                confidence="high",
            )
        )

    # -- oracle-gap families ------------------------------------------
    gaps = [
        f
        for f in fam_oracle.values()
        if f["n"] >= 4
        and f["oracle_fail"] / f["n"] >= 0.6
        and f["eq_total"] == 0
    ]
    gaps.sort(key=lambda f: (-f["oracle_fail"], f["family"]))
    for f in gaps[:4]:
        frac = f["oracle_fail"] / f["n"]
        out.append(
            _proposal(
                "ledger_oracle_gap",
                f"Oracle never sees an equal site in family "
                f"{f['family']} ({f['oracle_fail']}/{f['n']} died at "
                "oracle, 0 eq)",
                target={"domain": "discovery", "tier": f["family"]},
                evidence={
                    "sources": sorted(sources),
                    "metrics": {
                        "candidates": f["n"],
                        "oracle_failures": f["oracle_fail"],
                        "eq_total": f["eq_total"],
                        "acc_total": f["acc_total"],
                        "examples": f["examples"],
                    },
                },
                lever=(
                    "The mint produces claims this family's corpus "
                    "cannot evaluate — extend the corpus/env coverage "
                    "for the family's op patterns (corpus_cov gap "
                    "report names the missing ops/attrs) before "
                    "touching the laws."
                ),
                headroom=min(1.0, 0.3 + 0.5 * frac),
                confidence="medium",
            )
        )

    # -- vacuous novelty ----------------------------------------------
    for r in vacuous[:4]:
        out.append(
            _proposal(
                "ledger_vacuous",
                f"Vacuous claim: {r['name']} binds no real site "
                "(novelty unmeasured)",
                target={"domain": "discovery", "op": r["name"]},
                evidence={
                    "sources": [r["source"]],
                    "metrics": {
                        "corpus": r["corpus"],
                        "arena": r["arena"],
                        "family": r["family"],
                    },
                },
                lever=(
                    "No binding produced a site, so the novelty "
                    "gates are unevaluable — either the claim's LHS "
                    "never occurs in the corpus (corpus gap) or the "
                    "pattern is over-specific (mint gap).  Measure "
                    "which before re-running the gauntlet."
                ),
                headroom=0.4,
                confidence="low",
            )
        )
    return out


# ---------------------------------------------------------------------------
#  Signal: frontier rows — algebraic headroom the delivery misses
# ---------------------------------------------------------------------------


def _sig_frontier(reports_dir: Path) -> list[dict]:
    """Mine ``frontier*.json`` for uncaptured algebraic headroom."""
    out: list[dict] = []
    rows: list[tuple[str, dict]] = []
    for path in _reports(reports_dir, "frontier*.json"):
        doc = _load_json(path)
        if isinstance(doc, dict):
            for r in doc.get("rows") or []:
                if isinstance(r, dict):
                    rows.append((str(path.relative_to(_REPO)), r))
    # material headroom = headroom_us above the dispatch-noise floor
    # (~5us); uncaptured = captured missing or below half the ceiling
    cands = []
    for src, r in rows:
        hu = r.get("headroom_us")
        if hu is None or hu < 5.0 or not r.get("verified"):
            continue
        cap = r.get("captured")
        uncaptured = cap is None or cap < 0.5
        loses_compile = r.get("beats_compile") is False
        if not (uncaptured or loses_compile):
            continue
        deficit = 1.0 - cap if isinstance(cap, (int, float)) else 1.0
        cands.append((hu * max(deficit, 0.2), src, r, cap))
    cands.sort(key=lambda t: -t[0])
    for score, src, r, cap in cands[:8]:
        regime = r.get("regime") or "?"
        out.append(
            _proposal(
                "frontier_delivery_gap",
                f"Uncaptured headroom on {r.get('case')}: "
                f"{r.get('headroom_us')}us ceiling, captured="
                f"{cap}, vs_compile={r.get('vs_compile')}",
                target={
                    "domain": "gen",
                    "tier": regime,
                    "op": r.get("case"),
                },
                evidence={
                    "sources": [src],
                    "metrics": {
                        "case": r.get("case"),
                        "regime": regime,
                        "headroom_us": r.get("headroom_us"),
                        "ceiling": r.get("ceiling"),
                        "speedup": r.get("speedup"),
                        "captured": cap,
                        "vs_compile": r.get("vs_compile"),
                        "beats_compile": r.get("beats_compile"),
                        "kernels": r.get("kernels"),
                    },
                },
                lever=(
                    "The algebraic ceiling is real but the delivered "
                    "body doesn't reach it — profile the delivered "
                    "module against the ceiling decomposition "
                    "(dispatch vs device share) and attack the "
                    "largest residual term, not the rewrite bank."
                ),
                headroom=min(1.0, 0.25 + score / 200.0),
                confidence="medium",
            )
        )
    return out


# ---------------------------------------------------------------------------
#  Signal: chain sweeps — referee straddles and delivered losers
# ---------------------------------------------------------------------------


def _sig_sweeps(reports_dir: Path) -> list[dict]:
    """Mine ``zoo_chain_sweep*.json`` run spreads.

    A row *straddles* when its per-run speedups cross 1.0 — the
    keep/drop call is noise-decided.  A *delivered loser* shipped a
    verified module slower than eager.
    """
    straddles: list[tuple[str, dict, list[float]]] = []
    losers: list[tuple[str, dict, float]] = []
    n_rows = 0
    n_claim_rows = 0
    for path in _reports(reports_dir, "zoo_chain_sweep*.json"):
        doc = _load_json(path)
        src = str(path.relative_to(_REPO))
        # list docs are row tables; dict docs (e.g. the T-016
        # postgate re-sweep) nest row tables under named keys —
        # flatten every list-of-dicts value.
        if isinstance(doc, list):
            row_groups = [(src, doc)]
        elif isinstance(doc, dict):
            row_groups = [
                (src, v) for v in doc.values() if isinstance(v, list)
            ]
        else:
            continue
        for gsrc, rows in row_groups:
            for r in rows:
                if not isinstance(r, dict):
                    continue
                # speedup samples: the per-run table when present,
                # else the row's own single-shot `speedup`.
                sps = [
                    float(run["speedup"])
                    for run in r.get("runs", [])
                    if isinstance(run, dict) and "speedup" in run
                ]
                if not sps and "speedup" in r:
                    try:
                        sps = [float(r["speedup"])]
                    except (TypeError, ValueError):
                        sps = []
                if not sps:
                    continue
                n_rows += 1
                has_claims = bool(r.get("claims"))
                if has_claims:
                    n_claim_rows += 1
                # keep/drop is only a decision when a claim shipped —
                # claim-less rows measure timing noise, not the gate
                if (
                    has_claims
                    and len(sps) >= 2
                    and (min(sps) < 1.0 <= max(sps))
                ):
                    straddles.append((gsrc, r, sps))
                med = sorted(sps)[len(sps) // 2]
                if has_claims and r.get("delivered") and med < 1.0:
                    losers.append((gsrc, r, med))

    out: list[dict] = []
    if straddles:
        straddles.sort(key=lambda t: -(max(t[2]) - min(t[2])))
        worst = straddles[0]
        frac = len(straddles) / max(n_claim_rows, 1)
        out.append(
            _proposal(
                "sweep_straddle",
                f"{len(straddles)}/{n_claim_rows} claim-bearing "
                "sweep rows straddle parity — referee noise "
                "decides keep/drop",
                target={"domain": "gen", "tier": "referee"},
                evidence={
                    "sources": sorted({s for s, _, _ in straddles}),
                    "metrics": {
                        "straddling_rows": len(straddles),
                        "claim_rows": n_claim_rows,
                        "total_rows": n_rows,
                        "fraction": round(frac, 4),
                        "worst_spread": [
                            min(worst[2]),
                            max(worst[2]),
                        ],
                        "worst_case": worst[1].get("case"),
                        "examples": [
                            r.get("case") for _, r, _ in straddles[:6]
                        ],
                    },
                },
                lever=(
                    "Interleaved-window referee noise flips marginal "
                    "claims between runs — the two-sample borderline "
                    "guard exists but the residual band is still "
                    "wide.  Widen confirmation windows or add a "
                    "third arm (eager-vs-compiled delta) before the "
                    "keep vote."
                ),
                headroom=min(1.0, 0.3 + frac),
                confidence="high",
            )
        )
    if losers:
        losers.sort(key=lambda t: t[2])
        worst = losers[0]
        out.append(
            _proposal(
                "sweep_delivered_loser",
                f"{len(losers)}/{n_claim_rows} claim-bearing "
                "delivered+verified rows run below parity "
                f"(worst {worst[1].get('case')} at "
                f"{worst[2]:.3f}x)",
                target={"domain": "gen", "tier": "referee"},
                evidence={
                    "sources": sorted({s for s, _, _ in losers}),
                    "metrics": {
                        "delivered_losers": len(losers),
                        "claim_rows": n_claim_rows,
                        "total_rows": n_rows,
                        "worst_case": worst[1].get("case"),
                        "worst_speedup": worst[2],
                        "examples": [
                            (r.get("case"), round(m, 3))
                            for _, r, m in losers[:6]
                        ],
                    },
                },
                lever=(
                    "Verified-but-slower deliveries mean the keep "
                    "threshold admits losers on some regimes — "
                    "re-derive the margin per regime (launch-bound "
                    "vs compute-bound) instead of one global bar."
                ),
                headroom=min(
                    1.0,
                    0.2 + 0.4 * len(losers) / max(n_claim_rows, 1),
                ),
                confidence="high",
            )
        )
    return out


# ---------------------------------------------------------------------------
#  Signal: pay_measured — confounded executor comparisons
# ---------------------------------------------------------------------------


def _sig_paymeasured(reports_dir: Path) -> list[dict]:
    """Mine ``pay_measured*.json`` for confounded marginal rows."""
    confounded: list[tuple[str, dict]] = []
    total = 0
    for path in _reports(reports_dir, "pay_measured*.json"):
        doc = _load_json(path)
        if not isinstance(doc, dict):
            continue
        src = str(path.relative_to(_REPO))
        for r in doc.get("rows") or []:
            if not isinstance(r, dict):
                continue
            total += 1
            if r.get("verdict") == "confounded":
                confounded.append((src, r))
    if not confounded:
        return []
    confounded.sort(key=lambda t: -(t[1].get("marginal") or 0.0))
    worst = confounded[0][1]
    frac = len(confounded) / max(total, 1)
    return [
        _proposal(
            "paymeasured_confound",
            f"{len(confounded)}/{total} pay_measured rows are "
            "executor-confounded — the marginal measures routing, "
            "not the fold",
            target={"domain": "gen", "tier": "measurement"},
            evidence={
                "sources": sorted({s for s, _ in confounded}),
                "metrics": {
                    "confounded": len(confounded),
                    "total": total,
                    "fraction": round(frac, 4),
                    "worst_module": worst.get("module"),
                    "worst_size": worst.get("size"),
                    "worst_marginal": worst.get("marginal"),
                    "worst_note": worst.get("note"),
                    "examples": [
                        f"{r.get('module')}/{r.get('size')}"
                        for _, r in confounded[:6]
                    ],
                },
            },
            lever=(
                "synop routes to IRModule while noprim routes to "
                "BatchedScanModule — the A/B compares executors, "
                "not the fold.  Re-run marginals with a pinned "
                "executor or add a same-executor arm before "
                "trusting the pay table."
            ),
            headroom=min(1.0, 0.4 + frac),
            confidence="high",
        )
    ]


# ---------------------------------------------------------------------------
#  Signal: corpus coverage — the Stage-1 residual gaps
# ---------------------------------------------------------------------------


def _sig_coverage(seed: int, n: int) -> list[dict]:
    """Run ``corpus_cov.coverage_gaps`` over a fresh stream.

    Base kinds (``coverage=False``) plus the coverage arm — the
    residual missing ops/attrs/handlers/rules are the next corpus
    round's agenda.  Torch-free: terms only, no arena runs.
    """
    try:
        from mlopt_discovery import corpus_cov, meta_player
    except Exception as exc:  # honest skip — report, don't crash
        return [
            _proposal(
                "coverage_unavailable",
                f"Coverage census unavailable: {exc}",
                lever="Fix the mlopt_discovery import and re-run.",
                headroom=0.0,
                confidence="low",
            )
        ]
    cases = meta_player.gen_cases(seed, n, coverage=False)
    cases += corpus_cov.gen_cases_coverage(seed, n)
    gaps = corpus_cov.coverage_gaps(cases)
    vocab = corpus_cov.vocabulary()
    out: list[dict] = []

    def _cov_prop(
        signal,
        title,
        missing_n,
        total,
        missing_sample,
        covered_n,
        lever,
    ):
        frac = missing_n / max(total, 1)
        return _proposal(
            signal,
            title,
            target={"domain": "meta-arena", "tier": "corpus"},
            evidence={
                "sources": ["corpus_cov.coverage_gaps (live census)"],
                "metrics": {
                    "cases": gaps["n_cases"],
                    "missing": missing_n,
                    "total": total,
                    "covered": covered_n,
                    "missing_sample": missing_sample,
                },
            },
            lever=lever,
            headroom=min(1.0, 0.2 + 0.6 * frac),
            confidence="medium",
        )

    ops = gaps["ops"]
    out.append(
        _cov_prop(
            "coverage_ops",
            f"{ops['n_missing']} expressible ops never occur in "
            f"the {gaps['n_cases']}-case stream",
            ops["n_missing"],
            len(vocab["ops"]),
            ops["missing"][:20],
            len(ops["covered"]),
            "Extend coverage_builders' templates to the missing "
            "op classes (synop*/new classes need new templates — "
            "the honest extension seam named in corpus_cov).",
        )
    )
    attrs = gaps["attrs"]
    out.append(
        _cov_prop(
            "coverage_attrs",
            f"{attrs['n_missing']} declared (op, attr) pairs never "
            "occur in the stream",
            attrs["n_missing"],
            len(vocab["attrs"]),
            [f"{o}.{a}" for o, a in attrs["missing"][:20]],
            len(attrs["covered"]),
            "Attr-carrying case kinds: the zoo census found zero "
            "attr elementwise ops — decide whether attrs need "
            "synthetic boards or real-workload intake first.",
        )
    )
    rules = gaps["rules"]
    out.append(
        _cov_prop(
            "coverage_rules",
            f"{len(rules['unarmed'])}/{rules['n_vocab']} shipped "
            "rules never arm under the stream's rulesets",
            len(rules["unarmed"]),
            rules["n_vocab"],
            rules["unarmed"][:20],
            len(rules["armed"]),
            "Unarmed rules need boards whose rulesets carry them "
            "AND terms matching their LHS — the fire/saturate "
            "payoff dimension stays unmeasured until then.",
        )
    )
    handlers = gaps["handlers"]
    if handlers["missing"]:
        out.append(
            _cov_prop(
                "coverage_handlers",
                f"{len(handlers['missing'])} handler tags' concrete "
                "patterns never occur",
                len(handlers["missing"]),
                len(vocab["handlers"]),
                handlers["missing"],
                len(handlers["covered"]),
                "Handler patterns with no occurrence can't be "
                "claimed — extend the census or accept the tag as "
                "dead vocabulary.",
            )
        )
    return out


# ---------------------------------------------------------------------------
#  Render — JSONL + markdown agenda + myco ingest
# ---------------------------------------------------------------------------


def _myco_cmd(p: dict) -> str:
    """The ``myco todo add`` invocation ingesting one proposal."""
    pri = "high" if p["score"] >= 0.5 else "med"
    labels = "agenda"
    if p["confidence"] == "low":
        labels += ",low-confidence"
    title = p["title"].replace('"', "'")
    return (
        f'myco todo add "{title}" --pri {pri} '
        f"--label {labels} --no-commit"
    )


def _render_md(proposals: list[dict], meta: dict) -> str:
    """Human-readable agenda document."""
    lines = [
        "# Research agenda — self-mined proposals",
        "",
        f"Mined {meta['date']} from {meta['n_ledgers']} overnight "
        f"ledgers, {meta['n_frontier_rows']} frontier rows, "
        f"{meta['n_sweep_rows']} sweep rows, "
        f"{meta['n_pay_rows']} pay_measured rows, and a "
        f"{meta['coverage_cases']}-case coverage census.",
        "",
        "**Proposals only — nothing here auto-executes.**  Each "
        "item carries measured evidence and a suggested lever; "
        "humans approve by ingesting into `project/TODO.md` "
        "(commands below) before any work starts.",
        "",
        "## Ranked proposals",
        "",
        "| # | score | headroom | conf | signal | proposal |",
        "|---|-------|----------|------|--------|----------|",
    ]
    for i, p in enumerate(proposals, 1):
        title = p["title"].replace("|", "\\|")
        lines.append(
            f"| {i} | {p['score']:.3f} | {p['headroom']:.2f} | "
            f"{p['confidence']} | `{p['signal']}` | {title} |"
        )
    lines += ["", "## Evidence per proposal", ""]
    for i, p in enumerate(proposals, 1):
        lines += [
            f"### {i}. {p['title']}",
            "",
            f"- **signal:** `{p['signal']}` · **target:** "
            f"`{json.dumps(p['target'])}`",
            f"- **lever:** {p['lever']}",
            "- **evidence:**",
        ]
        for src in p["evidence"].get("sources", []):
            lines.append(f"  - source: `{src}`")
        metrics = p["evidence"].get("metrics", {})
        for k, v in metrics.items():
            vs = json.dumps(v)
            if len(vs) > 300:
                vs = vs[:300] + "…"
            lines.append(f"  - {k}: {vs}")
        lines.append("")
    lines += [
        "## Ledger ingest (human approval required)",
        "",
        "Each proposal maps to one `myco todo add` — run the ones "
        "you accept (`--no-commit` leaves the commit to `myco "
        "check --fix` / your own commit):",
        "",
        "```sh",
    ]
    for p in proposals:
        lines.append(_myco_cmd(p))
    lines += ["```", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reports",
        type=Path,
        default=_DEFAULT_REPORTS,
        help="report JSON directory (default docs/reports)",
    )
    ap.add_argument(
        "--out-prefix",
        type=Path,
        default=_REPO / "docs" / "reports" / "agenda",
        help="output prefix for .jsonl/.md (default docs/reports/agenda)",
    )
    ap.add_argument(
        "--cases",
        type=int,
        default=400,
        help="coverage-stream size per generator (default 400)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-coverage",
        action="store_true",
        help="skip the live corpus_cov census (artifact signals only)",
    )
    ap.add_argument(
        "--no-sweeps",
        action="store_true",
        help="skip zoo_chain_sweep*.json mining",
    )
    ap.add_argument(
        "--myco",
        action="store_true",
        help="ingest every proposal via `myco todo add --no-commit`",
    )
    args = ap.parse_args(argv)

    sys.path.insert(0, str(_REPO))

    proposals: list[dict] = []
    proposals += _sig_ledgers(args.reports)
    proposals += _sig_frontier(args.reports)
    if not args.no_sweeps:
        proposals += _sig_sweeps(args.reports)
    proposals += _sig_paymeasured(args.reports)
    coverage_cases = 0
    if not args.no_coverage:
        cov = _sig_coverage(args.seed, args.cases)
        proposals += cov
        coverage_cases = args.cases * 2

    # deterministic order: score desc, then id
    proposals.sort(key=lambda p: (-p["score"], p["id"]))
    # stable ids after sort — prefix rank so ids are unique per run
    for i, p in enumerate(proposals, 1):
        p["rank"] = i

    # meta for the report header
    def _rows_of(pattern: str, key: str | None = None) -> int:
        total = 0
        for p in _reports(args.reports, pattern):
            doc = _load_json(p)
            if isinstance(doc, dict) and key:
                total += len(doc.get(key) or [])
            elif isinstance(doc, dict):
                total += len(doc.get("rows") or [])
            elif isinstance(doc, list):
                total += len(doc)
        return total

    meta = {
        "date": datetime.now(UTC).date().isoformat(),
        "n_ledgers": len(
            _reports(args.reports, "results_overnight*.json")
        ),
        "n_frontier_rows": _rows_of("frontier*.json"),
        "n_sweep_rows": _rows_of("zoo_chain_sweep*.json"),
        "n_pay_rows": _rows_of("pay_measured*.json"),
        "coverage_cases": coverage_cases,
    }

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_prefix.with_suffix(".jsonl")
    md_path = args.out_prefix.with_suffix(".md")
    with jsonl_path.open("w") as fh:
        for p in proposals:
            fh.write(json.dumps(p, sort_keys=True) + "\n")
    md_path.write_text(_render_md(proposals, meta))

    # summary to stdout
    from collections import Counter

    counts = Counter(p["signal"] for p in proposals)
    print(f"agenda: {len(proposals)} proposals -> {jsonl_path}")
    print(f"        {md_path}")
    for sig, n in counts.most_common():
        print(f"  {sig}: {n}")
    if args.myco:
        import subprocess

        for p in proposals:
            # shell-free argv form for subprocess
            argv = [
                "myco",
                "todo",
                "add",
                p["title"],
                "--pri",
                "high" if p["score"] >= 0.5 else "med",
                "--label",
                "agenda"
                + (
                    ",low-confidence"
                    if p["confidence"] == "low"
                    else ""
                ),
                "--no-commit",
            ]
            r = subprocess.run(argv, capture_output=True, text=True)
            print(
                f"  myco[{p['id']}]: rc={r.returncode} {r.stdout.strip()[:80]}"
            )
            if r.returncode != 0:
                print(
                    f"    stderr: {r.stderr.strip()[:160]}",
                    file=sys.stderr,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

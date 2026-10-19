"""The pipeline delta and the report tables.

The baseline-vs-enlarged pipeline delta (mirroring
``mlopt_discovery.workload_gen``'s harness), the per-workload /
rejection / binding-gap report tables and the probe-eligibility
filter.  The ``_report_pipeline`` orchestrator and the ``main`` CLI
driver live in the package ``__init__`` (they drive ``ingest`` /
``_bench_cases`` / ``_run_delta`` through the package namespace the
test suite patches).
"""

from __future__ import annotations

from typing import Any

from mlopt_discovery.impact import TermCase

from .ledger import (
    _PROBE_MAX_NODES,
    Rejection,
)

# ---------------------------------------------------------------------------
#  Pipeline delta — mirror law_workload_gen's harness verbatim
# ---------------------------------------------------------------------------


def _run_delta(
    base_cases: list[TermCase],
    probe_base: list[TermCase],
    intake_cases: list[TermCase],
    probe_intake: list[TermCase],
    vocab: str,
    holdout: str | None,
) -> dict:
    """Run the pipeline on the corpus, then corpus+intake.

    Reuses ``mlopt_discovery.workload_gen._run_pipeline`` — the same census ->
    propose -> measure -> rank mirror that tool validated — with the
    intake cases as the corpus extension and the probe-eligible
    subset as the firing/reach additions.
    """
    from mlopt_discovery import workload_gen as lwg

    base = lwg._run_pipeline(base_cases, probe_base, vocab, holdout)
    big = lwg._run_pipeline(
        [*base_cases, *intake_cases],
        [*probe_base, *probe_intake],
        vocab,
        holdout,
    )
    return {"baseline": base, "enlarged": big}


def _intake_fires(ev: Any) -> int:
    """Count an evidence row's firings on intake cases."""
    return sum(1 for c in ev.fire_cases if c.startswith("intake:"))


def _delta_table(res: dict) -> str:
    """Render baseline-vs-enlarged pipeline comparison."""
    b, e = res["baseline"], res["enlarged"]
    b_names = {ev.proposal.name for ev in b["ranked"]}
    e_names = {ev.proposal.name for ev in e["ranked"]}
    b_rank = {ev.proposal.name: ev for ev in b["ranked"]}
    e_rank = {ev.proposal.name: ev for ev in e["ranked"]}
    lines = [
        f"corpus: {b['n_terms']} -> {e['n_terms']} terms, "
        f"{b['n_tuples']} -> {e['n_tuples']} op-tuples",
        f"proposals: {len(b_names)} -> {len(e_names)}",
    ]
    new_props = sorted(e_names - b_names)
    lines.append(f"new proposals ({len(new_props)}):")
    for name in new_props:
        ev = e_rank[name]
        lines.append(
            f"  {name:<28} [{ev.proposal.family}] "
            f"match={ev.matches} fires={ev.fires} "
            f"(intake={_intake_fires(ev)}) paid={ev.paid} "
            f"ship={'SHIP' if ev.shippable else ev.no_ship_reason}"
        )
    if not new_props:
        lines.append("  (none)")
    lines.append("-- firing/paid deltas on shared proposals --")
    shown = 0
    for name in sorted(b_names & e_names):
        be, ee = b_rank[name], e_rank[name]
        extra = _intake_fires(ee)
        if not (
            extra or ee.fires != be.fires or ee.matches != be.matches
        ):
            continue
        lines.append(
            f"  {name:<28} match {be.matches}->{ee.matches} "
            f"fires {be.fires}->{ee.fires} (intake={extra}) "
            f"paid {be.paid}->{ee.paid} "
            f"ship: {'Y' if ee.shippable else ee.no_ship_reason}"
        )
        shown += 1
    if not shown:
        lines.append("  (no shared proposal changed)")
    b_ship = [ev.proposal.name for ev in b["ranked"] if ev.shippable]
    e_ship = [ev.proposal.name for ev in e["ranked"] if ev.shippable]
    lines.append(
        f"shippable: {len(b_ship)} -> {len(e_ship)}"
        + (
            f"  new: {[s for s in e_ship if s not in b_ship]}"
            if e_ship
            else ""
        )
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Reporting + driver
# ---------------------------------------------------------------------------


def _status_table(records: list[dict], delta: dict) -> str:
    """Render the per-workload intake table."""
    head = (
        f"{'workload':<36} {'kind':<12} {'nodes':>6} {'status':<13} "
        f"{'+tup':>5} {'+shp':>5}  missing ops"
    )
    lines = [head, "-" * len(head)]
    for rec in records:
        nt, ns = delta["per_case"].get(rec["name"], ([], 0))
        lines.append(
            f"{rec['name']:<36} {rec['builder']:<12} "
            f"{rec['n_op_nodes']:>6} {rec['status']:<13} "
            f"{len(nt):>5} {ns:>5}  "
            f"{', '.join(rec['unsupported_ops']) or '-'}"
        )
    return "\n".join(lines)


def _rejection_table(rejections: list[Rejection]) -> str:
    """Render the export-boundary rejection backlog."""
    lines = [f"{'workload':<36} {'stage':<8} error"]
    lines.append("-" * 80)
    for r in rejections:
        lines.append(f"{r.name:<36} {r.stage:<8} {r.error}")
    return "\n".join(lines)


def _gap_table(records: list[dict]) -> str:
    """Aggregate unbound ops -> the workloads that need them."""
    gaps: dict[str, list[str]] = {}
    for rec in records:
        for op in rec["unsupported_ops"]:
            gaps.setdefault(op, []).append(rec["name"])
    if not gaps:
        return "  (no binding gaps)"
    lines = [f"{'missing op':<26} {'#':>3}  needed by"]
    lines.append("-" * 80)
    for op, names in sorted(gaps.items()):
        lines.append(f"{op:<26} {len(names):>3}  {', '.join(names)}")
    return "\n".join(lines)


def _eligible(cases: list[TermCase], records: list[dict]) -> list:
    """Return the in-memory probe-eligible cases (ingested + cap)."""
    meta = {r["name"]: (r["status"], r["n_op_nodes"]) for r in records}
    return [
        c
        for c in cases
        if meta.get(c.name, ("", 0))[0] == "ingested"
        and meta[c.name][1] <= _PROBE_MAX_NODES
    ]

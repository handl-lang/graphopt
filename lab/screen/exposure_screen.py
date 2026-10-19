"""Screen the model corpus for e-graph-reachable structure.

Two measured lessons motivate this probe (see
``project/retros/birefnet-compile-parity.md`` and
``project/retros/retnet-external-baselines.md``): a fused black-box
op (``scaled_dot_product_attention``) is invisible to the e-graph —
no law's LHS names it — while math *spelled out* elementwise is
exactly what the search can rewrite.  Nobody has yet screened the
model zoo for that property.  This tool computes an **exposure**
metric per workload:

* ``lhs_cover`` — the FLOPs-weighted fraction of op nodes whose op
  appears in at least one shipped law's LHS pattern (the fireable
  surface: ``ALL_RULES_WITH_LAYOUT`` + the composed
  ``DEFAULT_RULES`` incl. carrier presets + ``ATTENTION`` +
  ``SCAN_LAWS`` / ``SCAN_DIAG_LAWS``).  A BiRefNet-class model —
  one fused ``sdpa`` — scores ~0 here.
* ``node_cover`` — the same fraction unweighted (per op node).
* ``strict`` / ``relaxed`` — the count of subterms matching at
  least one law's LHS pattern, strict (metavars must repeat-equal)
  and relaxed (repeated metavars as wildcards — "the op shape is
  present, the equality precondition may not be").
* ``invisible`` — the op census the laws cannot see at all, with
  its FLOPs share; these are the black-box ops (sdpa / conv* /
  fused norms / fft / einsum …).
* ``--fire N`` — a bounded real saturation on the N most exposed
  terms, reporting how many distinct rules actually fire and the
  extracted cost delta — the ground-truth check that the static
  exposure score predicts e-graph reachability.

Static analysis on exported IR only — random-init modules, no
weights, no lowering, CPU.  Unpriceable nodes (``_INVALID_COST``
or a shape rule that raises) are excluded from the FLOPs weights
and counted separately — an honest gap, not a silent zero.

Run::

    .venv/bin/python lab/screen/exposure_screen.py
    .venv/bin/python lab/screen/exposure_screen.py --set zoo --fire 8
    .venv/bin/python lab/screen/exposure_screen.py --json /tmp/exposure.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlopt_carriers  # noqa: F401 — registers the carrier machinery
import torch
from mlopt_core.cost.basic import _flops_of
from mlopt_core.egraph import EGraph
from mlopt_core.egraph.terms import _term_match
from mlopt_core.ir import Op
from mlopt_core.laws import (
    ALL_RULES_WITH_LAYOUT,
    ATTENTION_RULES,
    SCAN_DIAG_LAWS,
    SCAN_LAWS,
)
from mlopt_discovery import intake as li
from mlopt_discovery import zoo as zoo_mod
from mlopt_discovery.impact import (
    TermCase,
    _iter_subterms,
    _relax,
)
from mlopt_orchestrator import DEFAULT_RULES
from mlopt_torch.adapters import TorchSink
from mlopt_torch.torch_bridge import export_to_ir

#: Per-node FLOPs at/above this are the cost model's "cannot price"
#: mark — excluded from the weights, counted honestly as unpriced.
_INVALID = 1e15


# ---------------------------------------------------------------------------
#  The fireable law surface
# ---------------------------------------------------------------------------


def _ops_of(t: Any, out: set) -> set:
    """Collect op names inside a (pattern) term."""
    if isinstance(t, Op):
        out.add(t.op)
        for a in t.args:
            _ops_of(a, out)
    return out


def law_surface() -> tuple[list, set, set]:
    """Return ``(rules, lhs_ops, rhs_ops)`` over the fireable surface.

    ``lhs_ops`` is what the search can *see into* — an op appearing
    in no LHS is a black box no rewrite can decompose.  ``rhs_ops``
    is what laws may *introduce* (reported for context only).
    """
    seen: dict[str, Any] = {}
    for r in [
        *ALL_RULES_WITH_LAYOUT,
        *DEFAULT_RULES.rules,
        *ATTENTION_RULES,
        *SCAN_LAWS,
        *SCAN_DIAG_LAWS,
    ]:
        seen.setdefault(r.name, r)
    rules = list(seen.values())
    lhs_ops: set = set()
    rhs_ops: set = set()
    for r in rules:
        _ops_of(r.lhs, lhs_ops)
        _ops_of(r.rhs, rhs_ops)
    return rules, lhs_ops, rhs_ops


# ---------------------------------------------------------------------------
#  Per-term exposure record
# ---------------------------------------------------------------------------


@dataclass
class Row:
    """One workload's exposure measurement."""

    source: str
    name: str
    status: str = "ok"
    note: str = ""
    n_nodes: int = 0
    n_ops: int = 0
    flops_total: float = 0.0
    flops_exposed: float = 0.0
    unpriced_nodes: int = 0
    exposed_nodes: int = 0
    strict_sites: int = 0
    strict_rules: int = 0
    relaxed_sites: int = 0
    invisible_ops: dict = field(default_factory=dict)
    unsupported_ops: tuple = ()
    # --fire: bounded saturation ground truth
    fired_rules: int = 0
    fire_names: tuple = ()
    base_cost: float | None = None
    best_cost: float | None = None

    @property
    def lhs_cover(self) -> float:
        """FLOPs-weighted exposure: law-visible share of priced work."""
        if self.flops_total <= 0:
            return 0.0
        return self.flops_exposed / self.flops_total

    @property
    def node_cover(self) -> float:
        """Node-count exposure: law-visible share of op nodes."""
        if self.n_nodes <= 0:
            return 0.0
        return self.exposed_nodes / self.n_nodes


def _match_counts(subterms: list, pats: list) -> tuple[int, int, int]:
    """Count strict/relaxed LHS matches of *subterms* against *pats*.

    Returns ``(strict_sites, strict_rule_count, relaxed_sites)`` — a
    site counts once per (rule, subterm) pair that matches.
    """
    strict_sites = 0
    relaxed_sites = 0
    rules_hit: set = set()
    for name, lhs, relaxed in pats:
        for s in subterms:
            if not isinstance(s, Op):
                continue
            if _term_match(lhs, s) is not None:
                strict_sites += 1
                rules_hit.add(name)
            if _term_match(relaxed, s) is not None:
                relaxed_sites += 1
    return strict_sites, len(rules_hit), relaxed_sites


def screen_term(
    source: str,
    name: str,
    term: Any,
    pats: list,
    lhs_ops: set,
) -> Row:
    """Compute the exposure record of one exported term."""
    row = Row(source=source, name=name)
    memo: dict = {}
    subs = [s for s in _iter_subterms(term) if isinstance(s, Op)]
    row.n_nodes = len(subs)
    inv: dict[str, float] = {}
    for s in subs:
        try:
            f = _flops_of(s, memo)
        except Exception:
            f = _INVALID
        if f >= _INVALID:
            row.unpriced_nodes += 1
            f = 0.0
        row.flops_total += f
        if s.op in lhs_ops:
            row.exposed_nodes += 1
            row.flops_exposed += f
        else:
            inv[s.op] = inv.get(s.op, 0.0) + f
    row.n_ops = len({s.op for s in subs})
    row.invisible_ops = dict(sorted(inv.items(), key=lambda kv: -kv[1]))
    row.strict_sites, row.strict_rules, row.relaxed_sites = (
        _match_counts(subs, pats)
    )
    return row


def _export(workload: Any, source: str) -> tuple[Row, TermCase | None]:
    """Build + export one workload; return its row (or rejection)."""
    try:
        model, feed = workload.build()
    except Exception as e:
        return (
            Row(
                source=source,
                name=workload.name,
                status="rejected",
                note=f"build: {type(e).__name__}: {e}",
            ),
            None,
        )
    feeds = feed if isinstance(feed, tuple) else (feed,)
    try:
        ir, tensors = export_to_ir(model.eval().double(), feeds)
    except Exception as e:
        return (
            Row(
                source=source,
                name=workload.name,
                status="rejected",
                note=f"export: {type(e).__name__}: {e}",
            ),
            None,
        )
    case = TermCase(
        source=source,
        name=workload.name,
        term=ir.root,
        inputs=tuple(ir.inputs),
        feed=tuple(feeds),
        param_vals=dict(tensors),
    )
    return Row(source=source, name=workload.name), case


def _fire_probe(case: TermCase, rules: list, cost_fn: Any) -> dict:
    """Saturate *case* under the default set; report the firing."""
    eg = EGraph()
    root = eg.add_term(case.term)
    eg.run(rules, root, max_iterations=6, max_nodes=60_000)
    fired = {n: c for n, c in eg.rule_fires.items() if c}
    best = eg.extract_best(root, cost_fn)
    return {
        "fired_rules": len(fired),
        "fire_names": tuple(sorted(fired)),
        "best": best,
    }


def _workloads(which: str) -> list[tuple[str, Any]]:
    """Return ``(source, Workload-like)`` pairs for the requested set."""
    out: list[tuple[str, Any]] = []
    if which in ("zoo", "all"):
        out += [("zoo", w) for w in zoo_mod.zoo()]
    if which in ("intake", "all"):
        out += [("intake", w) for w in li.candidates()]
    if which in ("models", "all"):
        from mlopt_discovery.impact import _model_cases

        class _WL:
            def __init__(self, name: str, m: Any, x: Any) -> None:
                self.name = name
                self._m = m
                self._x = x

            def build(self) -> tuple:
                return self._m, self._x

        out += [("model", _WL(n, m, x)) for n, m, x in _model_cases()]
    return out


def _table(rows: list[Row]) -> str:
    """Render the exposure ranking."""
    head = (
        f"{'model':<26} {'src':<7} {'nodes':>5} {'ops':>4} "
        f"{'lhs%':>6} {'node%':>6} {'flops':>11} {'strict':>6} "
        f"{'rlx':>6} {'unp':>4}  invisible ops"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        if r.status != "ok":
            lines.append(
                f"{r.name:<26} {r.source:<7} {'--':>5} {'--':>4} "
                f"{'--':>6} {'--':>6} {'--':>11} {'--':>6} "
                f"{'--':>6} {'--':>4}  {r.status}: {r.note[:60]}"
            )
            continue
        inv = ", ".join(
            f"{o}({f / max(r.flops_total, 1) * 100:.0f}%)"
            for o, f in list(r.invisible_ops.items())[:5]
        )
        lines.append(
            f"{r.name:<26} {r.source:<7} {r.n_nodes:>5} {r.n_ops:>4} "
            f"{r.lhs_cover * 100:>5.1f}% {r.node_cover * 100:>5.1f}% "
            f"{r.flops_total:>11.4g} {r.strict_sites:>6} "
            f"{r.relaxed_sites:>6} {r.unpriced_nodes:>4}  {inv}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Screen the workloads; rank by exposure; print the table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--set",
        choices=("zoo", "intake", "models", "all"),
        default="all",
        help="which registry to screen (default all)",
    )
    ap.add_argument(
        "--fire",
        type=int,
        default=0,
        metavar="N",
        help="bounded-saturation ground truth on the N most exposed",
    )
    ap.add_argument("--json", help="write machine-readable results")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    rules, lhs_ops, rhs_ops = law_surface()
    pats = [(r.name, r.lhs, _relax(r.lhs)) for r in rules]
    print(  # stdout-compat
        "== exposure_screen — e-graph-reachable structure =="
    )
    print(  # stdout-compat
        f"   law surface: {len(rules)} rules, "
        f"{len(lhs_ops)} lhs-visible ops "
        f"(rhs-only introduces: {sorted(rhs_ops - lhs_ops)})"
    )
    sink = TorchSink()
    supported = sink.supported_ops

    rows: list[Row] = []
    cases: dict[str, TermCase] = {}
    for source, wl in _workloads(args.set):
        row, case = _export(wl, source)
        if case is None:
            rows.append(row)
            continue
        row = screen_term(source, wl.name, case.term, pats, lhs_ops)
        row.unsupported_ops = tuple(
            sorted(_ops_of(case.term, set()) - supported)
        )
        rows.append(row)
        cases[wl.name] = case

    ok = [r for r in rows if r.status == "ok"]
    ok.sort(key=lambda r: (-r.lhs_cover, -r.strict_sites, r.name))
    rej = [r for r in rows if r.status != "ok"]
    print(  # stdout-compat
        f"   screened: {len(ok)} exported, {len(rej)} rejected"
    )
    print()
    print(_table(ok))
    if rej:
        print()
        print("-- rejected --")
        for r in rej:
            print(f"  {r.source}:{r.name} — {r.note}")

    if args.fire:
        print()
        print(f"-- fire probe (top {args.fire} by exposure) --")
        from mlopt_core.cost import backend_cost, dag_cost
        from mlopt_core.cost.executor import executor_cost_for

        cost_fn = backend_cost(
            executor_cost_for(lowering="generic"), supported
        )
        default_rules = list(DEFAULT_RULES.rules)
        for r in ok[: args.fire]:
            case = cases[r.name]
            res = _fire_probe(case, default_rules, cost_fn)
            r.fired_rules = res["fired_rules"]
            r.fire_names = res["fire_names"]
            r.base_cost = dag_cost(case.term, cost_fn)
            r.best_cost = (
                dag_cost(res["best"], cost_fn)
                if res["best"] is not None
                else None
            )
            drop = (
                ""
                if r.best_cost is None or r.base_cost <= 0
                else f" {100 * (r.base_cost - r.best_cost) / r.base_cost:+.2f}%"
            )
            print(  # stdout-compat
                f"  {r.name:<26} exposure={r.lhs_cover * 100:5.1f}% "
                f"fired={r.fired_rules} cost={r.base_cost:.4g}"
                f"→{r.best_cost if r.best_cost is not None else float('nan'):.4g}{drop}"
            )
            if r.fire_names:
                print(f"      {', '.join(r.fire_names)}")

    if args.json:
        payload = {
            "n_rules": len(rules),
            "lhs_ops": sorted(lhs_ops),
            "rhs_only_ops": sorted(rhs_ops - lhs_ops),
            "rows": [
                {
                    "source": r.source,
                    "name": r.name,
                    "status": r.status,
                    "note": r.note,
                    "n_nodes": r.n_nodes,
                    "n_ops": r.n_ops,
                    "flops_total": r.flops_total,
                    "flops_exposed": r.flops_exposed,
                    "lhs_cover": r.lhs_cover,
                    "node_cover": r.node_cover,
                    "exposed_nodes": r.exposed_nodes,
                    "unpriced_nodes": r.unpriced_nodes,
                    "strict_sites": r.strict_sites,
                    "strict_rules": r.strict_rules,
                    "relaxed_sites": r.relaxed_sites,
                    "invisible_ops": r.invisible_ops,
                    "unsupported_ops": list(r.unsupported_ops),
                    "fired_rules": r.fired_rules,
                    "fire_names": list(r.fire_names),
                }
                for r in ok + rej
            ],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    sys.exit(main())

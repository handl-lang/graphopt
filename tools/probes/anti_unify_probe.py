"""Measure ADR-0007 layer-1 falsification #1 — does naming expand reach.

Layer-1's claim (falsification #1 of the ADR): *naming structure
expands reach* — definitional atoms minted by anti-unification should
let a bounded saturation reach deeper or cheaper structure than the
same budget without names.  This probe measures it directly.

Protocol (per workload, one seeded export, one hash seed):

* ``base`` — saturate the fresh e-graph with the rule set under the
  full budget (``--iters`` iterations, ``--max-nodes``), extract under
  ``flops_cost``.
* ``atoms`` (the strict arm) — saturate ``--iters-a`` iterations on
  the same graph, mine + mint + admit atoms
  (:func:`catopt_discovery.antiunify`), then continue with the
  *compressed* rule set: intro rules plus ``fold_ruleset(rules,
  atoms, sides, replace=True)`` under the remaining iterations and
  the same node budget.  ``replace=True`` drops rules that gained an
  rhs-folded variant — folded firings mint one ``atom`` enode where
  the parent spelled the body, so the enode budget stretches.
* ``mix`` — the same protocol but ``replace=False``: parents stay
  beside their folded variants (more coverage, less saving).
  Extraction prices through :func:`unfolding_cost` and the winner is
  unfolded back to pure library ops before reporting — the atom
  never counts as cheaper than its expansion.

Per arm: the stop reason (``fixed_point`` vs ``max_nodes`` vs
``max_iterations`` — whether the budget bound), enodes/classes at
stop, the extracted cost on the *unfolded* program, that winner's
unfolded op count and depth, and wall seconds.  If a compressed arm
reaches fixed point where ``base`` truncated, or a lower cost at the
same budget — the pay.  If not, the honest null.

Workloads::

    --workload delta   _DeltaRuleBlock(dk=8, dv=8, steps=S) — the
                       unrolled rank-1 recurrence, spelled elementwise;
                       rules = catopt_core.laws.DEFAULT.
    --workload deq     _DeepEquilibrium(d=8, iters=N) — the shared-
                       weight fixed-point tower; rules = DEFAULT.
    --workload retnet  _ChunkedRetention(16), T/K chunks — the
                       measured shared-state frontier; rules = the
                       reach stage of ``om_shared_state_stages(K)``
                       with its budgets.

Usage::

    .venv/bin/python tools/probes/anti_unify_probe.py --workload delta \
        --steps 6 --iters 8 --max-nodes 20000
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import Op
from catopt_discovery.antiunify import (
    MintedAtom,
    admit_atoms,
    fold_ruleset,
    mint_atoms,
    unfold_term,
    unfolding_cost,
)


def _export(workload: str, args: argparse.Namespace) -> tuple:
    """Build the seeded workload module's ``(IR, leaves)`` export."""
    import torch
    from catopt_discovery.zoo import (
        _ChunkedRetention,
        _DeepEquilibrium,
        _DeltaRuleBlock,
    )
    from catopt_torch.adapters import TorchSource

    torch.manual_seed(0)
    if workload == "delta":
        model = _DeltaRuleBlock(8, 8, args.steps).double()
        x = torch.randn(2, args.steps, 8, dtype=torch.float64)
    elif workload == "deq":
        model = _DeepEquilibrium(8, args.steps).double()
        x = torch.randn(2, 8, dtype=torch.float64)
    else:  # retnet
        model = _ChunkedRetention(16).double()
        x = torch.randn(2, args.t, 16, dtype=torch.float64)
    return TorchSource().to_ir(model, x)


def _rules(workload: str, args: argparse.Namespace) -> list:
    """Return the saturation set for the workload's stage(s)."""
    if workload == "retnet":
        from catopt_carriers.om import om_shared_state_stages

        reach, _reassoc = om_shared_state_stages(args.k)
        return reach
    from catopt_core.laws import DEFAULT

    return list(DEFAULT)


def _budgets(workload: str, args: argparse.Namespace) -> dict | None:
    """Rule budgets for the workload (None when the set is unbudgeted)."""
    if workload == "retnet":
        from catopt_carriers.om import om_shared_state_budgets

        return om_shared_state_budgets(args.k, kway=args.budget)
    return None  # DEFAULT sets run unbudgeted for delta/deq


def _unfolded_stats(term: Any) -> tuple[int, int]:
    """``(op count, depth)`` of a term's DAG — semantic, not spelling."""
    memo: dict[int, tuple[int, int]] = {}

    def go(t: Any, stack: frozenset) -> tuple[int, int]:
        if not isinstance(t, Op):
            return (0, 0)
        if id(t) in stack:
            return (0, 0)
        if id(t) in memo:
            return memo[id(t)]
        kids = [go(a, stack | {id(t)}) for a in t.args]
        n = 1 + sum(k[0] for k in kids)
        d = 1 + max((k[1] for k in kids), default=0)
        memo[id(t)] = (n, d)
        return memo[id(t)]

    return go(term, frozenset())


def _mint_report(atoms: list[MintedAtom]) -> None:
    """Print each minted atom and its provenance."""
    for a in atoms:
        print(
            f"    {a.name}: arity={a.arity} uses={a.n_instances} "
            f"gain={a.gain} :: {a.expansion!r}",
            flush=True,
        )


def _run_arm(
    workload: str,
    args: argparse.Namespace,
    mode: str,
) -> dict:
    """Run one arm and return the measured row.

    ``mode`` is ``base`` (rules only), ``atoms`` (stage A, mint,
    compressed stage B with folded rules replacing parents), or
    ``mix`` (the same but parents kept beside their folded variants).
    """
    term = _export(workload, args)[0].root
    rules = _rules(workload, args)
    budgets = _budgets(workload, args)
    eg = EGraph()
    root = eg.add_term(term)
    atoms: list[MintedAtom] = []
    t0 = time.perf_counter()
    if mode == "base":
        stats = eg.run(
            rules,
            root,
            max_iterations=args.iters,
            max_nodes=args.max_nodes,
            rule_budgets=budgets,
            patience=8,
        )
    else:
        it1 = max(1, args.iters_a)
        stats = eg.run(
            rules,
            root,
            max_iterations=it1,
            max_nodes=args.max_nodes,
            rule_budgets=budgets,
            patience=8,
        )
        atoms = mint_atoms(
            eg,
            count=args.atoms,
            prefix="defn",
            min_uses=args.min_uses,
            max_ops=args.max_ops,
        )
        fires = admit_atoms(eg, root, atoms)
        sides = ("rhs", "lhs") if args.lhs_folds else ("rhs",)
        compressed = [a.intro for a in atoms] + fold_ruleset(
            rules,
            atoms,
            sides=sides,
            replace=mode == "atoms",
        )
        stats = eg.run(
            compressed,
            root,
            max_iterations=max(1, args.iters - it1),
            max_nodes=args.max_nodes,
            rule_budgets=budgets,
            patience=8,
        )
        stats["intro_fires"] = fires
    secs = time.perf_counter() - t0
    cfn = unfolding_cost(flops_cost, atoms)
    best = eg.extract_best(eg.find(root), cfn)
    un = unfold_term(best, atoms) if atoms else best
    ops, depth = _unfolded_stats(un)
    # Semantic-reach measure: the cheapest extractable member once
    # every defn_ enode is banned — pure reach, no compressed
    # spellings.  (``bans`` skips banned enodes at their classes.)
    cost = None
    if atoms:
        bans: dict[int, set] = {}
        for cid, ec in eg._classes.items():
            ds = {n for n in ec.nodes if n.op.startswith("defn_")}
            if ds:
                bans[cid] = ds
        spelled = eg.extract_best(eg.find(root), cfn, bans=bans)
        if spelled is not None:
            cost = dag_cost(spelled, flops_cost)
    elif un is not None:
        cost = dag_cost(un, flops_cost)
    compressed = dag_cost(un, flops_cost) if un is not None else None
    cops, cdepth = _unfolded_stats(best) if best is not None else (0, 0)
    return {
        "mode": mode,
        "cost": cost,
        "compressed_cost": compressed,
        "n_enodes": eg.n_enodes,
        "n_classes": eg.n_classes,
        "stop": stats["stop"],
        "iters": stats["iterations"],
        "ops": ops,
        "cops": cops,
        "cdepth": cdepth,
        "depth": depth,
        "secs": secs,
        "atoms": atoms,
        "stats": stats,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the arms and print the measured comparison."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--workload",
        default="delta",
        choices=("delta", "deq", "retnet"),
    )
    ap.add_argument("--steps", type=int, default=6, help="unroll/iters")
    ap.add_argument("--t", type=int, default=64, help="retnet T")
    ap.add_argument("--k", type=int, default=4, help="retnet chunks K")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--iters-a", type=int, default=4)
    ap.add_argument("--max-nodes", type=int, default=40000)
    ap.add_argument(
        "--budget", type=int, default=200, help="kway budget"
    )
    ap.add_argument("--atoms", type=int, default=8)
    ap.add_argument("--min-uses", type=int, default=2)
    ap.add_argument("--max-ops", type=int, default=8)
    ap.add_argument(
        "--lhs-folds",
        action="store_true",
        help="also fold rule LHSs (compressed-input variants)",
    )
    ap.add_argument(
        "--arms",
        default="base,atoms",
        help="comma-separated: base, atoms, mix",
    )
    args = ap.parse_args(argv)
    print(
        f"workload={args.workload} steps={args.steps} "
        f"iters={args.iters}+{args.iters_a} max_nodes={args.max_nodes}",
        flush=True,
    )
    for arm in args.arms.split(","):
        r = _run_arm(args.workload, args, arm)
        print(
            f"{arm:>8} stop={r['stop']:<14} iters={r['iters']:>3} "
            f"enodes={r['n_enodes']:>7} classes={r['n_classes']:>7} "
            f"reach={r['cost']!s:>14} compr={r['compressed_cost']!s:>14} "
            f"cops={r['cops']:>4}/{r['ops']:>4} "
            f"depth={r['depth']:>4} secs={r['secs']:>6.1f}",
            flush=True,
        )
        if r["atoms"]:
            _mint_report(r["atoms"])
        if r["stats"].get("intro_fires"):
            print(f"    intro fires: {r['stats']['intro_fires']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

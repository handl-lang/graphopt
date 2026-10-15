"""ADR-0008 falsification gates 1-2 over the combinator bridge.

Gate 1 — ``dag → combinator → dag`` on the zoo/intake corpus (the same
export path as ``lab/screen/exposure_screen.py``).  Per workload: does the
round-trip return a structurally equal term (``==`` — hash-consed, so
equal terms are literally the same interned object), and how large is
the morphism term against the DAG node count and the 60k enode budget
the saturation probes use?

Gate 2 — hand-spell the RetNet shared-state carrier as
``loop (arr step)`` (:func:`combinator.retnet_fold_program`), expand,
and verify end-to-end against ``zoo._ChunkedRetention`` through
``TorchSink``.  ``--extracted`` additionally runs one bounded staged
saturation (``om_shared_state_stages``) and projects the extracted
winner through the bridge — the "does the found term live in the
presentation" check.

Run::

    .venv/bin/python lab/probes/combinator_probe.py            # gate 1, all
    .venv/bin/python lab/probes/combinator_probe.py --verify 8 # + end-to-end
    .venv/bin/python lab/probes/combinator_probe.py --gate2    # loop spelling
    .venv/bin/python lab/probes/combinator_probe.py --gate2 --extracted
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from catopt_core.ir import Op
from catopt_discovery.impact import _iter_subterms

from lab.bridge.combinator import combinator as cb
from lab.probes.om_shared_state_probe import (
    om_shared_state_probe as osp,
)
from lab.screen.exposure_screen import _export, _workloads

#: Ops that ARE structural morphisms (product/projection/wire plumbing)
#: presented as opaque ``arr`` — they translate, but the presentation
#: doesn't yet *know* they are structure rather than generators.
STRUCTURAL_AS_ARR = {
    "pack",
    "getitem",
    "unbind",
    "stack",
    "broadcast_tensors",
    "split",
    "chunk",
    "tensor_split",
    "select",
    "slice",
    "index",
    "select_scatter",
    "slice_scatter",
    "scatter_add",
    "expand",
    "expand_as",
    "broadcast_to",
    "broadcast",
    "squeeze",
    "unsqueeze",
    "flatten",
    "unflatten",
    "reshape",
    "permute",
    "transpose",
    "movedim",
    "contiguous",
    "alias",
    "copy",
    "to",
    "detach_",
    "real",
}

#: Ops whose semantics IS a ``loop`` — fused scans/recurrences the DAG
#: exports as opaque atoms.  They translate as ``arr`` but the bridge
#: cannot recover the recurrence (the DAG erased it); ``loop`` forms
#: must be proposed by search, not read off.
HIDDEN_LOOP = {"cumsum", "cumprod", "lstm.input"}


@dataclass
class Row:
    """One workload's gate-1 record."""

    source: str
    name: str
    status: str = "ok"
    note: str = ""
    dag_nodes: int = 0
    comb_nodes: int = 0
    dup: int = 0
    dup_copies: int = 0
    perm: int = 0
    perm_wires: int = 0
    drop: int = 0
    structural_as_arr: int = 0
    hidden_loop_ops: tuple = ()
    roundtrip: bool = False
    verified: float | None = None

    @property
    def blowup(self) -> float:
        """Combinator-term nodes per DAG node (the Δ-expansion cost)."""
        return self.comb_nodes / max(self.dag_nodes, 1)


def _op_census(term: Any) -> Counter:
    """Op-name counts over the distinct nodes of a term DAG."""
    return Counter(
        s.op for s in _iter_subterms(term) if isinstance(s, Op)
    )


def screen_case(source: str, name: str, term: Any) -> Row:
    """Round-trip one exported term through the bridge; measure."""
    row = Row(source=source, name=name)
    row.dag_nodes = cb.dag_size(term)
    try:
        morph = cb.dag_to_comb(term)
    except Exception as e:
        row.status = "refused"
        row.note = f"{type(e).__name__}: {e}"
        return row
    st = cb.morph_stats(morph)
    row.comb_nodes = st.nodes
    row.dup, row.dup_copies = st.dup, st.dup_copies
    row.perm, row.perm_wires = st.perm, st.perm_wires
    row.drop = st.drop
    try:
        rt = cb.comb_to_dag(morph)
    except Exception as e:
        row.status = "expand-fail"
        row.note = f"{type(e).__name__}: {e}"
        return row
    row.roundtrip = rt == term
    census = _op_census(term)
    row.structural_as_arr = sum(
        census[o] for o in census if o in STRUCTURAL_AS_ARR
    )
    row.hidden_loop_ops = tuple(
        sorted(o for o in census if o in HIDDEN_LOOP)
    )
    return row


def _verify_pair(case: Any, rt_term: Any) -> float:
    """End-to-end: lower both terms, run on feeds, return max|Δ|."""
    from catopt_core.ir import IR
    from catopt_torch.adapters import TorchSink

    sink = TorchSink()
    ir_a = IR(
        root=case.term,
        inputs=list(case.inputs),
        params={},
    )
    ir_b = IR(root=rt_term, inputs=list(case.inputs), params={})
    mod_a = sink.lower(ir_a, params=dict(case.param_vals))
    mod_b = sink.lower(ir_b, params=dict(case.param_vals))
    feeds = case.feed if isinstance(case.feed, tuple) else (case.feed,)
    with torch.no_grad():
        out_a = mod_a(*feeds)
        out_b = mod_b(*feeds)
    if not isinstance(out_a, tuple):
        out_a, out_b = (out_a,), (out_b,)
    return max(
        float((a - b).abs().max())
        for a, b in zip(out_a, out_b, strict=True)
    )


def gate1(which: str, verify: int) -> list[Row]:
    """Run the bridge over the corpus; return per-workload rows."""
    torch.manual_seed(0)
    wls = _workloads(which)
    rows: list[Row] = []
    rebuilt: dict[str, Any] = {}
    cases: dict[str, Any] = {}
    t0 = time.time()
    for source, wl in wls:
        row, case = _export(wl, source)
        if case is None:
            rows.append(
                Row(
                    source=source,
                    name=wl.name,
                    status="rejected",
                    note=row.note,
                )
            )
            continue
        r = screen_case(source, wl.name, case.term)
        rows.append(r)
        if r.roundtrip:
            rebuilt[wl.name] = cb.comb_to_dag(cb.dag_to_comb(case.term))
            cases[wl.name] = case
    ok = [r for r in rows if r.status == "ok"]
    print(
        f"== gate 1: corpus round-trip ({which}) — "
        f"{len(ok)} exported of {len(wls)} in "
        f"{time.time() - t0:.0f}s =="
    )
    n_rt = sum(1 for r in ok if r.roundtrip)
    print(
        f"   round-trip equal: {n_rt}/{len(ok)}"
        + ("" if n_rt == len(ok) else "   <-- FAILURES BELOW")
    )
    for r in ok:
        if not r.roundtrip:
            print(f"      {r.source}:{r.name} round-trip FAILED")
    for r in rows:
        if r.status not in ("ok", "rejected"):
            print(f"      {r.source}:{r.name} {r.status}: {r.note}")
    rej = [r for r in rows if r.status == "rejected"]
    for r in rej:
        print(
            f"      (export-rejected) {r.source}:{r.name} {r.note[:70]}"
        )

    head = (
        f"{'model':<26} {'src':<7} {'dag':>5} {'comb':>5} "
        f"{'x':>5} {'dup':>4} {'copies':>6} {'perm':>4} "
        f"{'wires':>6} {'struct':>6} {'loops':<18}"
    )
    print()
    print(head)
    print("-" * len(head))
    ok.sort(key=lambda r: (-r.blowup, r.name))
    for r in ok:
        print(
            f"{r.name:<26} {r.source:<7} {r.dag_nodes:>5} "
            f"{r.comb_nodes:>5} {r.blowup:>5.2f} {r.dup:>4} "
            f"{r.dup_copies:>6} {r.perm:>4} {r.perm_wires:>6} "
            f"{r.structural_as_arr:>6} "
            f"{','.join(r.hidden_loop_ops) or '':<18}"
        )
    n = len(ok)
    if n:
        tot = cb.CombStats()
        for r in ok:
            tot.dup += r.dup
            tot.dup_copies += r.dup_copies
            tot.perm_wires += r.perm_wires
        print(
            f"\n   totals: {n} terms, "
            f"Σdag={sum(r.dag_nodes for r in ok)} "
            f"Σcomb={sum(r.comb_nodes for r in ok)} "
            f"mean-blowup="
            f"{sum(r.blowup for r in ok) / n:.2f} "
            f"max-blowup={max(r.blowup for r in ok):.2f} "
            f"Σdup={tot.dup} Σperm-wires={tot.perm_wires}"
        )
        big = max(ok, key=lambda r: r.comb_nodes)
        print(
            f"   largest morph: {big.name} {big.comb_nodes} nodes "
            f"(enode budget 60000 → {big.comb_nodes / 60000:.3f})"
        )
    if verify:
        print(
            f"\n-- end-to-end verify (first {verify} round-tripped) --"
        )
        for r in ok[:verify]:
            case = cases[r.name]
            d = _verify_pair(case, rebuilt[r.name])
            r.verified = d
            print(f"   {r.name:<26} max|Δ| = {d:.3e}")
    return rows


def gate2(extracted: bool) -> None:
    """Spell the RetNet fold as ``loop (arr step)``; verify + project."""
    import catopt_carriers  # noqa: F401 — carrier op bindings
    from catopt_core.cost import dag_cost, flops_cost
    from catopt_core.ir import IR, Const, Param, TensorType, Var
    from catopt_discovery.zoo import _ChunkedRetention
    from catopt_torch.adapters import TorchSink

    torch.manual_seed(0)
    print("== gate 2: the shared-state carrier as a `loop` term ==")
    for t, k in ((8, 2), (64, 4), (128, 8)):
        d, b = 16, 2
        C = t // k
        model = _ChunkedRetention(d).double()
        x = Var("x", TensorType((b, t, d)))
        s0 = Var("s0", TensorType((d, d)))
        params = {
            "q_weight": Param("wq", TensorType((d, d))),
            "q_bias": Param("bq", TensorType((d,))),
            "k_weight": Param("wk", TensorType((d, d))),
            "k_bias": Param("bk", TensorType((d,))),
            "v_weight": Param("wv", TensorType((d, d))),
            "v_bias": Param("bv", TensorType((d,))),
        }
        g, a = Const(0.9), Const(0.9**C)
        prog = cb.retnet_fold_program(x, s0, params, g, a, t, k, d)
        term = cb.comb_to_dag(prog)
        st = cb.morph_stats(prog)
        ir = IR(
            root=term,
            inputs=[x, s0],
            params={p.name: p for p in params.values()},
        )
        mod = TorchSink().lower(
            ir,
            params={
                "wq": model.q.weight.detach(),
                "bq": model.q.bias.detach(),
                "wk": model.k.weight.detach(),
                "bk": model.k.bias.detach(),
                "wv": model.v.weight.detach(),
                "bv": model.v.bias.detach(),
            },
        )
        xt = torch.randn(b, t, d, dtype=torch.float64)
        s0t = torch.zeros(d, d, dtype=torch.float64)
        with torch.no_grad():
            ref = model(xt)
            out = mod(xt, s0t)
        diff = float((ref - out).abs().max())
        print(
            f"   T={t} K={k} C={C}: verify max|Δ|={diff:.3e}  "
            f"morph={st.nodes} (loop=1 dup={st.dup} perm={st.perm})  "
            f"dag={cb.dag_size(term)}  flops={dag_cost(term, flops_cost):.4g}"
        )
    print(
        "   basis expresses it: YES — `loop(step, n)` is the one "
        "extension used; step itself needs only dup/perm/par/arr.\n"
        "   the DAG bridge never mints `loop` — it is a *proposed* "
        "form, not a recovered one."
    )
    if extracted:
        _gate2_projection()


def _gate2_projection(t: int = 64, k: int = 4, iters: int = 10) -> None:
    """Saturate the RetNet export (staged arms) and project the winner.

    Reports whether the extracted term round-trips through the bridge,
    and whether its shared-state chain carries the same fold signature
    the spelled ``loop`` expands to.
    """
    from catopt_carriers.om import (
        om_shared_state_budgets,
        om_shared_state_stages,
    )
    from catopt_core.cost import flops_cost
    from catopt_core.egraph import EGraph

    reach, reassoc = om_shared_state_stages(k)
    budgets = om_shared_state_budgets(k)
    eg = EGraph()
    root = eg.add_term(osp._export(t))
    t0 = time.time()
    osp._saturate(eg, root, reach, iters, budgets)
    osp._saturate(eg, root, reassoc, iters, budgets)
    best = eg.extract_best(eg.find(root), flops_cost)
    secs = time.time() - t0
    reads, chained = osp._fold_signature(best)

    morph = cb.dag_to_comb(best)
    rt = cb.comb_to_dag(morph)
    st = cb.morph_stats(morph)
    print(
        f"-- projection (staged arm, T={t} K={k}, {secs:.0f}s, "
        f"{eg.n_enodes} enodes) --"
    )
    print(
        f"   extracted term: dag={cb.dag_size(best)}  "
        f"reads={reads} chained={chained}"
    )
    print(
        f"   projects: round-trip {'EQUAL' if rt == best else 'FAIL'}"
        f"  morph={st.nodes} dup={st.dup} copies={st.dup_copies}"
    )
    # Same fold signature on the spelled loop's expansion?  The
    # probe's ``_fold_signature`` is leaf-name-keyed (it expects the
    # export's ``p_*`` leaves), so compare a name-agnostic structural
    # depth instead: the longest ``add(mul(w, S), B)`` nesting chain —
    # the ``S_i = a·S_{i-1} + B_i`` recurrence shape, whoever spelled
    # the leaves.
    spelled = _expanded_fold_term(t, k)
    print(
        f"   fold-chain depth: spelled={_fold_depth(spelled)}  "
        f"extract={_fold_depth(best)}"
    )


def _fold_depth(term: Any) -> int:
    """Longest nesting chain of ``add(mul(w, S), B)`` state updates.

    Name-agnostic: a fold node is an ``add`` with a ``mul`` child whose
    one factor is a leaf (the scalar rate) and whose other factor is a
    leaf or another fold node (the state).  Depth 0 = no recurrence.
    """
    memo: dict[int, int] = {}

    def is_leaf(x: Any) -> bool:
        return not isinstance(x, Op)

    def fold_arg(n: Op) -> tuple[Any, bool]:
        """Return the state child of an ``add(mul(w,S),B)`` node."""
        if n.op != "add":
            return None, False
        for a in n.args:
            if isinstance(a, Op) and a.op == "mul" and len(a.args) == 2:
                lf, rt = a.args
                if is_leaf(lf) or is_leaf(rt):
                    # spelled form: mul(rate-leaf, state)
                    cand = rt if is_leaf(lf) else lf
                    return cand, True
                # scaled form: mul(rate, add(state, B)) — the state
                # nests under the decay mul
                for x in (lf, rt):
                    if isinstance(x, Op) and x.op == "add":
                        return x, True
        return None, False

    def depth(n: Any) -> int:
        if not isinstance(n, Op):
            return 0
        if id(n) in memo:
            return memo[id(n)]
        cand, ok = fold_arg(n)
        memo[id(n)] = 1 + depth(cand) if ok else 0
        return memo[id(n)]

    return max(
        (depth(s) for s in _iter_subterms(term) if isinstance(s, Op)),
        default=0,
    )


def _expanded_fold_term(t: int, k: int) -> Any:
    """Expand the spelled ``loop`` program to a DAG at (t, k)."""
    import catopt_carriers  # noqa: F401
    from catopt_core.ir import Const, Param, TensorType, Var

    d, b = 16, 2
    C = t // k
    x = Var("x", TensorType((b, t, d)))
    s0 = Var("s0", TensorType((d, d)))
    params = {
        "q_weight": Param("wq", TensorType((d, d))),
        "q_bias": Param("bq", TensorType((d,))),
        "k_weight": Param("wk", TensorType((d, d))),
        "k_bias": Param("bk", TensorType((d,))),
        "v_weight": Param("wv", TensorType((d, d))),
        "v_bias": Param("bv", TensorType((d,))),
    }
    prog = cb.retnet_fold_program(
        x, s0, params, Const(0.9), Const(0.9**C), t, k, d
    )
    return cb.comb_to_dag(prog)


def main(argv: list[str] | None = None) -> int:
    """Drive the falsification gates."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--set",
        choices=("zoo", "intake", "models", "all"),
        default="all",
    )
    ap.add_argument(
        "--verify",
        type=int,
        default=0,
        help="end-to-end check on N round-tripped workloads",
    )
    ap.add_argument("--gate2", action="store_true")
    ap.add_argument("--extracted", action="store_true")
    ap.add_argument("--json", help="write machine-readable gate-1 rows")
    args = ap.parse_args(argv)

    if args.gate2:
        gate2(args.extracted)
        return 0
    rows = gate1(args.set, args.verify)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                [
                    {
                        "source": r.source,
                        "name": r.name,
                        "status": r.status,
                        "note": r.note,
                        "dag_nodes": r.dag_nodes,
                        "comb_nodes": r.comb_nodes,
                        "blowup": r.blowup,
                        "dup": r.dup,
                        "dup_copies": r.dup_copies,
                        "perm": r.perm,
                        "perm_wires": r.perm_wires,
                        "structural_as_arr": r.structural_as_arr,
                        "hidden_loop_ops": list(r.hidden_loop_ops),
                        "roundtrip": r.roundtrip,
                        "verified_maxdiff": r.verified,
                    }
                    for r in rows
                ],
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Diagnostic — *why* extraction mixes two state spellings (ADR-0008 gate 3).

The gate-3 parametric-abstraction retro
(``project/retros/gate3-parametric-abstraction.md``) measured that the
search's winner does not collapse to one shared ``step``, and named
cause 2: **extraction mixed two state spellings** — the shared-state
read ``matmul(q, S)`` and the intra re-expansion
``om_decayed(matmul(q, k_prevᵀ), D, v_prev)``.

This probe measures the *mechanism* of that mix, not its consequence.
For the winner's extracted term it reports, per block, the state-read
subterm, its e-class, and every member of that e-class with its
``flops_cost`` — so the reading is "is the spelling a cost tie broken
arbitrarily, or is one genuinely cheaper?"  It also enumerates the
root e-class's top-k distinct members (``extract_alternatives``) and
prices the *uniform* spellings directly.

Run::

    .venv/bin/python bench/drivers/probes/gate3_state_spelling_probe.py
    .venv/bin/python bench/drivers/probes/gate3_state_spelling_probe.py --t 64 --k 4
"""
# ruff: noqa: RUF002

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Any

from mlopt_core.cost import dag_cost, flops_cost
from mlopt_core.egraph import EGraph
from mlopt_core.egraph.types import _member_key
from mlopt_core.ir import Op
from mlopt_discovery.impact import _iter_subterms

from bench.drivers.probes import om_shared_state_probe as osp


def _blocks(root: Any) -> list[Any]:
    """Flatten the root ``concat`` chain into per-block outputs."""
    out: list[Any] = []
    stack = [root]
    while stack:
        x = stack.pop()
        if isinstance(x, Op) and x.op == "concat":
            stack.extend(reversed(x.args))
        else:
            out.append(x)
    return out


def _census(term: Any) -> Counter:
    """Op-name census over a term's distinct nodes."""
    return Counter(
        s.op for s in _iter_subterms(term) if isinstance(s, Op)
    )


def _shape(x: Any) -> Any:
    from mlopt_core.typing import _shape_of

    return _shape_of(x, None)


def _print_tree(
    term: Any,
    eg: EGraph | None,
    cost_memo: dict,
    *,
    indent: int = 0,
    max_depth: int = 7,
    budget: list[int] | None = None,
) -> None:
    """Indented tree with local cost and (optionally) e-class member count."""
    if budget is None:
        budget = [80]
    if budget[0] <= 0:
        return
    budget[0] -= 1
    pad = "  " * indent
    if not isinstance(term, Op):
        print(f"{pad}{term!r}")
        return
    local = flops_cost(term, cost_memo) - sum(
        flops_cost(a, cost_memo) for a in term.args
    )
    cls = ""
    nmem = ""
    if eg is not None:
        eid, _en = eg._locate(term)
        if eid is not None:
            cls = f" cls={eg.find(eid)}"
            nmem = f" mem={len(eg.get_class(eg.find(eid)).nodes)}"
    attrs = f" {dict(term.attrs)}" if term.attrs else ""
    print(
        f"{pad}{term.op}{attrs}  local={local:.4g} "
        f"tot={flops_cost(term, cost_memo):.4g}{cls}{nmem}"
    )
    if indent >= max_depth:
        if term.args:
            print(f"{pad}  …")
        return
    for a in term.args:
        _print_tree(
            a,
            eg,
            cost_memo,
            indent=indent + 1,
            max_depth=max_depth,
            budget=budget,
        )


def _class_members(
    eg: EGraph, term: Any
) -> tuple[int | None, list[tuple[float, int, Any]]]:
    """``(eid, members)`` — every member of *term*'s e-class, priced.

    ``members`` is ``(flops, nops, member)`` sorted cheapest-first.
    """
    eid, _en = eg._locate(term)
    if eid is None:
        return None, []
    eid = eg.find(eid)
    ec = eg.get_class(eid)
    rows: list[tuple[float, int, Any]] = []
    # Canonical member order, not the ``set``'s hash order: ``rows.sort``
    # is stable, so equal-cost members would otherwise rank by hash
    # order — a per-process tie table.
    for node in sorted(ec.nodes, key=_member_key):
        if node.op == "leaf":
            continue
        m = eg.extract_best(eid, flops_cost, overrides={eid: node})
        if m is None:
            continue
        rows.append(
            (dag_cost(m, flops_cost), len(list(_iter_subterms(m))), m)
        )
    rows.sort(key=lambda r: (r[0], r[1]))
    return eid, rows


def _state_read(block: Any) -> Any:
    """Return the state-read subterm of a block (the ``add``'s 2nd addend).

    A block is ``add(intra, mul(ξ, read))`` (or the degenerate block 0
    ``intra``).  The read is the ``mul``'s second factor — the subterm
    that carries the recurrence state.
    """
    if (
        isinstance(block, Op)
        and block.op == "add"
        and len(block.args) == 2
    ):
        tail = block.args[1]
        if (
            isinstance(tail, Op)
            and tail.op == "mul"
            and len(tail.args) == 2
        ):
            return tail.args[1]
        return tail
    return block


def main(argv: list[str] | None = None) -> int:
    """Run the diagnostic over the gate-3 winner."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--alt", type=int, default=12)
    ap.add_argument(
        "--full",
        action="store_true",
        help="print untruncated member reprs",
    )
    args = ap.parse_args(argv)

    from mlopt_carriers.om import (
        om_shared_state_budgets,
        om_shared_state_stages,
    )

    t, k = args.t, args.k
    reach, reassoc = om_shared_state_stages(k)
    budgets = om_shared_state_budgets(k, kway=args.budget)
    eg = EGraph()
    root = eg.add_term(osp._export(t))
    t0 = time.time()
    osp._saturate(eg, root, reach, args.iters, budgets)
    osp._saturate(eg, root, reassoc, args.iters, budgets)
    root_eid = eg.find(root)
    best = eg.extract_best(root_eid, flops_cost)
    secs = time.time() - t0
    print(
        f"T={t} K={k} C={t // k}  saturation {secs:.0f}s  "
        f"{eg.n_enodes} enodes  winner dag={len(list(_iter_subterms(best)))} "
        f"cost={dag_cost(best, flops_cost):,.0f}"
    )
    reads, chained = osp._fold_signature(best)
    print(f"  fold signature: reads={reads} chained={chained}")

    blocks = _blocks(best)
    print(f"  blocks={len(blocks)}")
    cost_memo: dict = {}
    for i, b in enumerate(blocks):
        cen = _census(b)
        sr = _state_read(b)
        print(
            f"\n-- block {i}: cost={dag_cost(b, flops_cost):,.0f} "
            f"nodes={len(list(_iter_subterms(b)))} census={dict(cen)}"
        )
        print("   state-read tree:")
        _print_tree(
            sr,
            eg,
            cost_memo,
            indent=3,
            max_depth=30 if args.full else 5,
        )

    # --- the crux: is the state-read spelling a cost tie? ---------------
    print("\n== state-read e-classes (members priced) ==")
    for i, b in enumerate(blocks):
        sr = _state_read(b)
        eid, rows = _class_members(eg, sr)
        ops = Counter()
        for _c, _n, m in rows:
            ops[_top_op(m)] += 1
        print(
            f"  block {i}: cls={eid} members={len(rows)} "
            f"top-ops={dict(ops.most_common(6))}"
        )
        if rows:
            best_c = rows[0][0]
            tied = [r for r in rows if r[0] == best_c]
            chosen_c = dag_cost(sr, flops_cost)
            print(
                f"      cheapest={best_c:,.0f} tied_at_min={len(tied)} "
                f"chosen={chosen_c:,.0f} "
                f"chosen_is_min={chosen_c == best_c}"
            )
        for c, n, m in rows[: args.alt]:
            width = 10**9 if args.full else 96
            print(f"      {c:>12,.0f}  n={n:<4} {_sketch(m, width)}")

    # --- the root class: uniform spellings? ----------------------------
    print("\n== root e-class alternatives (top-k) ==")
    alts = eg.extract_alternatives(root_eid, flops_cost, top_k=args.alt)
    for c, m in alts:
        nblocks = len(_blocks(m))
        uniform = _uniform_spelling(m)
        print(
            f"   cost={c:>12,.0f}  blocks={nblocks}  "
            f"uniform_read={uniform}  {_sketch(m)}"
        )

    # --- cost-model sweep: does any principled model unify the read? ---
    print("\n== cost-model sweep (same saturated e-graph) ==")
    from mlopt_core.cost import (
        count_cost,
        launch_aware_cost,
        roofline_cost,
    )

    for name, fn in (
        ("flops", flops_cost),
        ("launch", launch_aware_cost),
        ("roofline", roofline_cost),
        ("count", count_cost),
    ):
        t1 = time.time()
        w = eg.extract_best(root_eid, fn)
        seq = _read_sequence(w)
        print(
            f"   {name:>9}: blocks={len(_blocks(w))} "
            f"reads={seq}  ({time.time() - t1:.1f}s)"
        )

    # --- override experiment: force the *shared* read per block --------
    print(
        "\n== override experiment: force the factored `matmul(q,S)` read =="
    )
    base_total = dag_cost(best, flops_cost)
    for i, b in enumerate(blocks):
        sr = _state_read(b)
        eid, rows = _class_members(eg, sr)
        if eid is None:
            continue
        shared = [m for _c, _n, m in rows if _is_shared_read(m)]
        if not shared:
            print(
                f"   block {i}: cls={eid} — no factored read reachable"
            )
            continue
        _e2, en = eg._locate(shared[0], eid)
        if en is None:
            continue
        forced = eg.extract_best(
            root_eid, flops_cost, overrides={eid: en}
        )
        if forced is None:
            continue
        ft = dag_cost(forced, flops_cost)
        print(
            f"   block {i}: override cls={eid} to factored read -> "
            f"root cost {ft:,.0f} (base {base_total:,.0f}, "
            f"Δ={ft - base_total:+,.0f})  reads={_read_sequence(forced)}"
        )

    # --- combined: force every block's read to the factored spelling ---
    print(
        "\n== combined override: every read forced to `matmul(q,S)` =="
    )
    forced_ov: dict[int, Any] = {}
    forced_bans: dict[int, set] = {}
    for b in blocks:
        sr = _state_read(b)
        eid, rows = _class_members(eg, sr)
        if eid is None:
            continue
        for _c, _n, m in rows:
            if _is_shared_read(m):
                _e2, en = eg._locate(m, eid)
                if en is not None:
                    forced_ov[eid] = en
                break
    combo = eg.extract_best(root_eid, flops_cost, overrides=forced_ov)
    if combo is not None:
        ct = dag_cost(combo, flops_cost)
        print(
            f"   all-shared: root cost {ct:,.0f} "
            f"(base {base_total:,.0f}, Δ={ct - base_total:+,.0f})  "
            f"reads={_read_sequence(combo)}"
        )
        _abstraction(combo, t, k)

    # --- banned forcing: remove every non-factored read member ----------
    print("\n== banned forcing: delete every unfactored read member ==")
    for b in blocks:
        sr = _state_read(b)
        eid, rows = _class_members(eg, sr)
        if eid is None:
            continue
        bad: set = set()
        for _c, _n, m in rows:
            if _is_shared_read(m):
                continue
            _e2, en = eg._locate(m, eid)
            if en is not None:
                bad.add(en)
        if bad:
            forced_bans[eid] = bad
    combo2 = eg.extract_best(root_eid, flops_cost, bans=forced_bans)
    if combo2 is not None:
        ct2 = dag_cost(combo2, flops_cost)
        print(
            f"   banned: root cost {ct2:,.0f} "
            f"(base {base_total:,.0f}, Δ={ct2 - base_total:+,.0f})  "
            f"reads={_read_sequence(combo2)}"
        )
        _abstraction(combo2, t, k)
    else:
        print("   banned: no extractable term (all reads banned)")

    # --- per-family abstraction (localise the remaining cause) ---------
    print("\n== per-family abstraction ==")
    _family_report(best)

    # --- the fix: sharing-aware coordinated extraction ----------------
    print("\n== sharing-aware extraction (the principled fix) ==")
    from bench.drivers.bridge.sharing_extract import (
        extract_sharing_aware,
    )

    t2 = time.time()
    res = extract_sharing_aware(eg, root_eid, flops_cost)
    print(
        f"   cost {res.cost:,.0f} (greedy {res.base_cost:,.0f}, "
        f"Δ={res.delta:+,.0f})  trials={res.n_trials} "
        f"({time.time() - t2:.1f}s)  reads={_read_sequence(res.term)}"
    )
    print(f"   overrides kept: {res.overrides}")
    _abstraction(res.term, t, k)
    print("   per-family on the sharing-aware term:")
    _family_report(res.term)

    # --- numeric check: is the cheaper term really the same program? ---
    _verify_pair_terms(t, best, res.term)
    return 0


def _verify_pair_terms(t: int, a: Any, b: Any) -> None:
    """Lower two terms with ``TorchSink`` and report ``max|Δ|``."""
    import mlopt_carriers  # noqa: F401 — carrier op bindings
    import torch
    from mlopt_core.ir import IR
    from mlopt_discovery.zoo import _ChunkedRetention
    from mlopt_torch.adapters import TorchSink
    from mlopt_torch.torch_bridge import export_to_ir

    torch.manual_seed(0)
    model = _ChunkedRetention(16).eval().double()
    x = torch.randn(2, t, 16, dtype=torch.float64)
    ir0, tensors = export_to_ir(model, (x,))
    sink = TorchSink()
    mod_a = sink.lower(
        IR(root=a, inputs=list(ir0.inputs), params=dict(ir0.params)),
        params=dict(tensors),
    )
    mod_b = sink.lower(
        IR(root=b, inputs=list(ir0.inputs), params=dict(ir0.params)),
        params=dict(tensors),
    )
    with torch.no_grad():
        out_a = mod_a(x)
        out_b = mod_b(x)
    if not isinstance(out_a, tuple):
        out_a, out_b = (out_a,), (out_b,)
    d = max(
        float((u - v).abs().max())
        for u, v in zip(out_a, out_b, strict=True)
    )
    print(
        f"\n== numeric check: greedy vs sharing-aware max|Δ| = {d:.3e} =="
    )


def _abstraction(term: Any, t: int, k: int) -> None:
    """Report the gate-3 abstraction verdict on a term."""
    from mlopt_discovery import index_abstract as ia

    rep = ia.abstract_index(term)
    print(
        f"      abstraction: blocks={rep.n_blocks} agree={rep.agree} "
        f"verdict={rep.verdict} diff={rep.diff!r}"
    )


def _block_family(block: Any) -> tuple:
    """Return the block's ``split`` family signature — the ``(sizes, dim)``.

    The most common ``dim == -2`` split signature in the block: a
    C-width chunk block slices its q/k/v with the C-partition, so the
    signature is the chunking family the block belongs to.
    """
    sigs: Counter = Counter()
    for s in _iter_subterms(block):
        if isinstance(s, Op) and s.op == "split":
            a = dict(s.attrs)
            if a.get("dim") == -2:
                sigs[(repr(a.get("sizes")), a.get("dim"))] += 1
    if not sigs:
        return ("?",)
    return sigs.most_common(1)[0][0]


def _family_report(root: Any) -> None:
    """Run the abstraction on each chunking family separately."""
    import mlopt_core.ir as ir
    from mlopt_discovery import index_abstract as ia

    blocks = _blocks(root)
    fams: dict[tuple, list[Any]] = {}
    for b in blocks:
        fams.setdefault(_block_family(b), []).append(b)
    for sig, bs in fams.items():
        if len(bs) < 2:
            print(f"   family {sig}: {len(bs)} block — skipped")
            continue
        out = bs[-1]
        for b in reversed(bs[:-1]):
            out = ir.Op.make("concat", b, out, dim=-2)
        print(f"   family {sig}: {len(bs)} blocks")
        _abstraction(out, 0, 0)
        # Which subterm does the abstraction bind as the state?
        fb = _blocks(out)
        own = ia._block_splits(fb)
        si = ia._state_inputs(fb, own)
        for i, (b, o, s) in enumerate(zip(fb, own, si, strict=True)):
            op = _top_op(s) if s is not None else None
            print(
                f"      block {i}: read={_classify_read(_state_read(b))} "
                f"own-splits={len(o)} state_in={op}"
            )


def _top_op(term: Any) -> str:
    return term.op if isinstance(term, Op) else "leaf"


def _leaf_names(t: Any, acc: set | None = None) -> set:
    """Leaf-name set of a term."""
    acc = set() if acc is None else acc
    if not isinstance(t, Op):
        acc.add(repr(t))
        return acc
    for a in t.args:
        _leaf_names(a, acc)
    return acc


def _is_shared_read(t: Any) -> bool:
    """Return True for the *factored* shared-state read ``matmul(q, S)``.

    Same test :func:`bench.drivers.probes.om_shared_state_probe._fold_signature`
    uses: a ``matmul`` whose left operand is free of the k/v leaves and
    whose right operand carries BOTH the k and v paths and no q — i.e.
    the q-free accumulated state.
    """
    if not isinstance(t, Op) or t.op != "matmul" or len(t.args) != 2:
        return False
    left, right = t.args
    if not (isinstance(left, Op) and isinstance(right, Op)):
        return False
    ll, rl = _leaf_names(left), _leaf_names(right)
    return (
        not (ll & osp._KV_LEAVES)
        and {"p_k_weight", "p_v_weight"} <= rl
        and not (rl & osp._Q_LEAVES)
    )


def _classify_read(read: Any) -> str:
    """Classify a block's state-read spelling.

    ``"read:shared"`` — the *factored* shared-state read ``matmul(q, S)``.
    ``"read:reexp"`` — the *unfactored* re-expansion through the carrier
    (``om_decayed`` / a ``div`` wrapping it, or the spelled
    ``matmul(mul(...))`` twin).
    ``"read:fold"`` — the folded state ``add(mul(γ, S), B)``.
    """
    op = _top_op(read)
    if _is_shared_read(read):
        return "read:shared"
    if op == "add":
        return "read:fold"
    if op in ("om_decayed", "div", "matmul"):
        return "read:reexp"
    return f"read:{op}"


def _read_sequence(root: Any) -> str:
    """Return the per-block state-read spelling sequence (compact)."""
    return "|".join(
        _classify_read(_state_read(b)) for b in _blocks(root)
    )


def _sketch(term: Any, width: int = 96) -> str:
    """Return a one-line structural sketch of a term (op names, DFS)."""
    from mlopt_core.ir import op_repr

    s = op_repr(term)
    return s if len(s) <= width else s[: width - 3] + "..."


def _uniform_spelling(root: Any) -> str:
    """Classify the winner's state spelling across its blocks.

    ``"uniform"`` when every block's state-read is the same op at the
    same position; otherwise ``"mixed:<ops>"`` naming the distinct
    state-read root ops.
    """
    blocks = _blocks(root)
    if len(blocks) < 2:
        return "n/a"
    tops = {_top_op(_state_read(b)) for b in blocks[1:]}
    if len(tops) == 1:
        return "uniform:" + next(iter(tops))
    return "mixed:" + ",".join(sorted(tops))


if __name__ == "__main__":
    raise SystemExit(main())

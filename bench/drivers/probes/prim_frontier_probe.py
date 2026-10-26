"""Diagnostic probe — where do synop-carrying shapes die in the
antiunify frontier on the REAL zoo corpus?

Replicates ``overnight._prim_pool``'s bootstrap seating, then runs an
instrumented copy of ``antiunify.mine_candidates``'s loop that counts,
for candidates whose pattern carries a ``synop*`` op:

* ``seeded`` — all-holes seed candidates carrying a synop op;
* ``pushed`` — carrying children pushed onto the worklist;
* ``truncated`` — carrying candidates dropped by the beam cut;
* ``popped`` / ``emitted`` — carrying candidates popped / emitted;
* the best-scoring carrying candidate's rank vs the beam cutoff;
* where the emitted synop-carriers land in the final gain order.

Run::

    .venv/bin/python bench/drivers/probes/prim_frontier_probe.py \
        [beam] [max_expand] [max_ops]
"""

from __future__ import annotations

import heapq
import itertools
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from bench.drivers import overnight as on


def _carries(pat, synops):
    from mlopt_core.ir import Op

    return isinstance(pat, Op) and (
        pat.op in synops or any(_carries(a, synops) for a in pat.args)
    )


def _carried_set(pat, synops):
    from mlopt_core.ir import Op

    if not isinstance(pat, Op):
        return set()
    out = {pat.op} & synops
    for a in pat.args:
        out |= _carried_set(a, synops)
    return out


def _seated_graph(roots):
    """The bootstrap-seated corpus e-graph (``_prim_pool``'s leg)."""
    from mlopt_core.egraph import EGraph
    from mlopt_core.laws import minted as _minted
    from mlopt_discovery import shipped_primitives as _sp

    eg = EGraph()
    for r in roots.values():
        eg.add_term(r)
    _sp.register()
    eg.rebuild()
    depth = on._PRIM_DEPTH
    seatable = [
        f
        for f in _minted.prim_folds()
        if on._synop_depth(_minted.synop_name(f.name)) < depth
    ]
    fires = {}
    passes = 0
    while passes < depth:
        passes += 1
        fired = 0
        for f in seatable:
            before = eg.rule_fires.get(f.name, 0)
            eg.apply_rule(f, 0)
            n = eg.rule_fires.get(f.name, 0) - before
            if n:
                fires[f.name] = fires.get(f.name, 0) + n
                fired += n
        eg.rebuild()
        if not fired:
            break
    return eg, fires, passes


def _mine_instrumented(
    eg,
    *,
    synops,
    synop2,
    min_ops,
    min_uses,
    max_ops,
    beam,
    max_expand,
    vocab=None,
):
    import heapq as hq

    from mlopt_discovery.antiunify import (
        _alpha_norm,
        _emit,
        _grow_all,
        _member_index,
        _metavar_count,
        _n_ops,
        _refine_options,
        _score,
        _seed,
        _seed_groups,
    )

    midx = _member_index(eg)
    vctr = itertools.count()
    actr = itertools.count()
    seeds = _seed_groups(midx, True, min_uses, set())
    work = []
    seq = itertools.count()
    pushed = set()
    stats = {
        "seeds": 0,
        "seed_carry": 0,
        "push_carry": 0,
        "trunc_events": 0,
        "trunc_carry": 0,
        "pops": 0,
        "pop_carry": 0,
        "emit_carry": 0,
        "carry_min_rank": None,
    }
    carry_heap_entries = []
    for occ in seeds:
        c = _seed(occ, vctr, actr)
        pushed.add(_alpha_norm(c.pat))
        carry = _carries(c.pat, synops)
        stats["seeds"] += 1
        stats["seed_carry"] += carry
        heapq.heappush(work, (-_score(c, vocab), next(seq), c))
        if carry:
            carry_heap_entries.append(c)
    out = {}
    pops = 0
    while work and pops < max_expand:
        _ns, _q, cand = heapq.heappop(work)
        pops += 1
        carry = _carries(cand.pat, synops)
        carry2 = _carries(cand.pat, synop2)
        stats["pops"] += 1
        stats["pop_carry"] += carry
        stats["pop_carry2"] = stats.get("pop_carry2", 0) + carry2
        key = _alpha_norm(cand.pat)
        if (
            _n_ops(cand.pat) >= min_ops
            and len(cand.rows) >= min_uses
            and key not in out
        ):
            out[key] = _emit(eg, cand, 512, vocab)
            stats["emit_carry"] += carry
            stats["emit_carry2"] = stats.get("emit_carry2", 0) + carry2
        # _grow_children inlined so carrying children are counted at
        # push time.
        ops = _n_ops(cand.pat)
        if ops < max_ops:
            for mvar in list(cand.cols):
                if _metavar_count(cand.pat, mvar) + ops > max_ops:
                    continue
                for sig in _refine_options(
                    cand, mvar, midx, True, min_uses
                ):
                    if sig[0] in synops:
                        stats["sig_synop_offered"] = (
                            stats.get("sig_synop_offered", 0) + 1
                        )
                    if sig[0] in synop2:
                        stats["sig2_offered"] = (
                            stats.get("sig2_offered", 0) + 1
                        )
                    child = _grow_all(
                        cand, mvar, sig, midx, vctr, actr, True
                    )
                    ckey = _alpha_norm(child.pat)
                    if ckey in pushed:
                        continue
                    pushed.add(ckey)
                    stats["push_carry"] += _carries(child.pat, synops)
                    stats["push_carry2"] = stats.get(
                        "push_carry2", 0
                    ) + _carries(child.pat, synop2)
                    heapq.heappush(
                        work, (-_score(child, vocab), next(seq), child)
                    )
        if len(work) > beam:
            keep = hq.nsmallest(beam, work)
            keep_ids = {id(t) for t in keep}
            dropped = [t for t in work if id(t) not in keep_ids]
            stats["trunc_events"] += 1
            stats["trunc_carry"] += sum(
                _carries(t[2].pat, synops) for t in dropped
            )
            stats["trunc_carry2"] = stats.get("trunc_carry2", 0) + sum(
                _carries(t[2].pat, synop2) for t in dropped
            )
            work = keep
            heapq.heapify(work)
        # frontier snapshot: are synop2-carriers sitting in the beam?
        stats["beam_carry"] = sum(
            _carries(t[2].pat, synops) for t in work
        )
        stats["beam_carry2"] = sum(
            _carries(t[2].pat, synop2) for t in work
        )
        stats["sig2_offered"] = stats.get("sig2_offered", 0)
        stats["pop_carry2"] = stats.get("pop_carry2", 0)
        stats["emit_carry2"] = stats.get("emit_carry2", 0)
    stats["mined"] = len(out)
    mined = sorted(
        out.values(),
        key=lambda c: (-c.gain, -c.n_instances, _alpha_norm(c.pattern)),
    )
    carry_idx = [
        i for i, c in enumerate(mined) if _carries(c.pattern, synops)
    ]
    stats["emit_rank_first"] = carry_idx[0] if carry_idx else None
    stats["emit_rank_last"] = carry_idx[-1] if carry_idx else None
    stats["n_emit_carry"] = len(carry_idx)
    carry2_idx = [
        i for i, c in enumerate(mined) if _carries(c.pattern, synop2)
    ]
    stats["n_emit_carry2"] = len(carry2_idx)
    stats["emit2_rank_first"] = carry2_idx[0] if carry2_idx else None
    for i in carry2_idx[:8]:
        c = mined[i]
        print(
            f"    carry2 rank {i}: ops={c.op_count} "
            f"inst={c.n_instances} gain={c.gain} "
            f"pat={_alpha_norm(c.pattern)[:110]}"
        )
    for i in carry_idx[:5]:
        c = mined[i]
        print(
            f"    rank {i}: ops={c.op_count} inst={c.n_instances} "
            f"gain={c.gain} pat={_alpha_norm(c.pattern)[:110]}"
        )
    return stats


def main() -> int:
    beam = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    max_expand = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    max_ops = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    k = 2

    from mlopt_core.laws import minted

    t0 = time.time()
    cache = Path("/tmp/_prim_probe_roots.pkl")
    if cache.exists():
        import pickle

        roots = pickle.loads(cache.read_bytes())
    else:
        env = on._xmodule_env(0)
        roots = {n: rec[0].root for n, rec in env.items() if rec}
        import pickle

        cache.write_bytes(pickle.dumps(roots))
    print(f"env: {len(roots)} roots in {time.time() - t0:.1f}s")

    t0 = time.time()
    eg, fires, passes = _seated_graph(roots)
    print(
        f"seated: {len(eg._classes)} classes, {sum(fires.values())} "
        f"fold fires over {passes} passes ({time.time() - t0:.1f}s)"
    )
    synops = set(minted.SYNOP_SHIPPED)
    synop2 = set(minted.SYNOP2_SHIPPED)
    # how many classes hold a synop2 member?
    n2 = sum(
        1
        for ec in eg._classes.values()
        if any(n.op in synop2 for n in ec.nodes)
    )
    n1 = sum(
        1
        for ec in eg._classes.values()
        if any(n.op in (synops - synop2) for n in ec.nodes)
    )
    print(f"classes holding synop_*: {n1} · synop2_*: {n2}")

    use_vocab = "--no-vocab" not in sys.argv
    t0 = time.time()
    stats = _mine_instrumented(
        eg,
        synops=synops,
        synop2=synop2,
        min_ops=2,
        min_uses=k,
        max_ops=max_ops,
        beam=beam,
        max_expand=max_expand,
        vocab=on._synop_vocab() if use_vocab else None,
    )
    print(f"vocab: {use_vocab}")
    print(
        f"mine beam={beam} max_expand={max_expand} max_ops={max_ops} "
        f"in {time.time() - t0:.1f}s"
    )
    for kk, v in stats.items():
        print(f"  {kk}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

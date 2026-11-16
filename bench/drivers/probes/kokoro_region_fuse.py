"""Kokoro region-fuser — delivery-path measurement driver.

Follow-up to ``kokoro_fusion_sites.md``: the census found 48 repeated
9-op transcendental regions on the ``decoder`` arm (9.5-14x per-region
fusion ceiling, ~35-45 ms/call prize) but ``genkernel.gen_bindings``
cannot compile them (Const *tensor* leaves break inductor
FakeTensorMode).  This driver lands the working delivery path the
report spec'd: a **lowering-time region fuser** that rewrites the
*delivered term* — each eligible elementwise region's root becomes a
single ``fused_region`` op whose args are the region's boundary
leaves — and lowers it through the REAL ``IRModule`` plan tape with a
custom :class:`OpTable` binding ``fused_region`` to a
``torch.compile``\\ d unrolled body (the probe's positional-args
approach: every boundary leaf, Consts included, is a runtime tensor
arg, so no Const is ever inlined into the compiled spec).

Distinct regions dedup by *internal structure* (op/attrs/arg-wiring)
— leaf identity and Const values are runtime inputs, so the 48
SineGen regions compile to ~2 kernels total.

Eligibility (conservative, single-exit):

* region has a UNIQUE root (one member with no elementwise parent);
* every non-root member's parents are all inside the member set —
  no intermediate value escapes the region (the root is the only
  exit, exactly the ``_fusion_sites`` collector-legal shape);
* ``len(members) >= --min-ops`` (default 2 — the fusion-site floor;
  singleton regions have nothing to fuse).

Everything else is counted and reported honestly (``multi_root`` /
``multi_exit`` / ``small`` skips).

Usage::

    .venv/bin/python bench/drivers/probes/kokoro_region_fuse.py \
        [decoder|decoder-nc|f0n|textenc-p] [iters] [min_ops]
"""

from __future__ import annotations

import importlib.util
import sys
import time
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


kokoro = _load("tts_kokoro_probe", "tts_kokoro_probe.py")


# ---------------------------------------------------------------------------
#  Region census + eligibility
# ---------------------------------------------------------------------------
#
# The census is ``kokoro_fusion_probe._region_census`` inlined (the
# DAG-corner-fixed copy of ``fusion_floor_probe._region_census``):
# importing the probe chain would pull ``whisper_decode_loop`` ->
# ``transformers``, which is not installed in this venv.  Only the
# census + helpers are needed here — the spelled/fused *timing*
# helpers are superseded by the delivered-module path below.


def _elem_ops():
    from kit_discovery import genkernel as gk

    return gk.ELEMENTWISE


def _subtree_ops(root, elem):
    """Count elementwise-Op descendants of *root* (DAG-deduped)."""
    from mlopt_core.ir import Op

    seen = set()
    stack = [root]
    count = 0
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        if isinstance(t, Op):
            if t.op in elem:
                count += 1
            stack.extend(t.args)
    return count


def _region_census(term):
    """Maximal elementwise-connex regions of *term* (DAG view).

    Same semantics as ``fusion_floor_probe._region_census`` with the
    multi-root ``len(int)`` fix ``kokoro_fusion_probe`` carries.
    Returns ``(regions, n_elementwise)`` sorted by size desc; each
    region is ``(root, members, boundary_leaf_nodes)``.
    """
    from collections import defaultdict

    from mlopt_core.ir import Const, Op
    from mlopt_core.traversal import ops as _ops

    elem = _elem_ops()
    nodes = [t for t in _ops(term) if t.op in elem]
    parents: dict[int, list] = defaultdict(list)
    for t in _ops(term):
        if isinstance(t, Op):
            for a in t.args:
                if isinstance(a, Op):
                    parents[id(a)].append(t)
    uf = {id(t): id(t) for t in nodes}

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            uf[rb] = ra

    for t in nodes:
        for p in parents.get(id(t), ()):
            if p.op in elem:
                union(id(t), id(p))

    comps: dict[int, list] = defaultdict(list)
    for t in nodes:
        comps[find(id(t))].append(t)

    regions = []
    for _cid, members in comps.items():
        member_ids = {id(m) for m in members}
        roots = [
            m
            for m in members
            if not any(p.op in elem for p in parents.get(id(m), ()))
        ]
        if len(roots) != 1:
            roots = sorted(roots, key=lambda r: -_subtree_ops(r, elem))
        root = roots[0]
        leaves = {}
        for m in members:
            for a in m.args:
                if id(a) in member_ids or isinstance(a, Const):
                    continue
                leaves.setdefault(id(a), a)
        regions.append((root, members, list(leaves.values())))
    regions.sort(key=lambda r: -len(r[1]))
    return regions, len(nodes)


def _parents_map(term):
    """``id(child Op) -> [parent Ops]`` over the whole term DAG."""
    from mlopt_core.ir import Op
    from mlopt_core.traversal import ops as _ops

    parents: dict[int, list] = {}
    for t in _ops(term):
        if isinstance(t, Op):
            for a in t.args:
                if isinstance(a, Op):
                    parents.setdefault(id(a), []).append(t)
    return parents


def _canon_region(root, members):
    """Topo-order *members* from *root*; return ``(order, leaf_pos)``.

    ``order`` is the post-order member list (root last);
    ``leaf_pos`` maps ``id(boundary node) -> positional index`` in
    first-appearance order.  Boundary nodes are EVERY non-member arg —
    ``Op``/``Var``/``Param``/``Const`` alike: all become runtime tensor
    inputs of the fused op, which is what keeps Const *tensor* leaves
    out of the compiled spec (the genkernel inductor blocker).
    Returns ``None`` when a member is unreachable from the root.
    """
    from mlopt_core.ir import Op

    member_ids = {id(m) for m in members}
    order: list = []
    seen: set[int] = set()

    def dfs(t):
        if id(t) in seen or id(t) not in member_ids:
            return
        seen.add(id(t))
        for a in t.args:
            if isinstance(a, Op):
                dfs(a)
        order.append(t)

    dfs(root)
    if len(order) != len(members):
        return None
    pos = {id(t): i for i, t in enumerate(order)}
    leaf_pos: dict[int, int] = {}
    leaves: list = []
    specs: list = []
    for m in order:
        ms = []
        for a in m.args:
            if isinstance(a, Op) and id(a) in pos:
                ms.append(("m", pos[id(a)]))
            else:
                if id(a) not in leaf_pos:
                    leaf_pos[id(a)] = len(leaves)
                    leaves.append(a)
                ms.append(("x", leaf_pos[id(a)]))
        specs.append(ms)
    return order, specs, leaves


def _freeze_attrs(attrs) -> tuple:
    return tuple(sorted((k, repr(v)) for k, v in attrs.items()))


def _region_key(order, specs) -> tuple:
    """Dedup key: internal structure only — leaves are positional."""
    return tuple(
        (m.op, _freeze_attrs(m.attrs), tuple(ms))
        for m, ms in zip(order, specs, strict=True)
    )


def _collect_regions(term, *, min_ops: int = 2):
    r"""Eligible single-exit regions of *term*.

    Returns ``(fuse_map, stats)`` where ``fuse_map`` maps each region
    root term to ``(key_id, leaf_nodes)`` and ``stats`` counts the
    honest skips.  ``key_id`` indexes the shared compiled-fn
    registry — regions with identical internal structure share one
    ``torch.compile``\ d kernel.
    """

    elem = _elem_ops()
    regions, _n = _region_census(term)
    parents = _parents_map(term)
    fuse_map: dict = {}
    keys: dict = {}
    compiled: dict = {}
    stats = Counter()
    for _root, members, _lv in regions:
        if len(members) < min_ops:
            stats["small"] += 1
            continue
        member_ids = {id(m) for m in members}
        roots = [
            m
            for m in members
            if not any(p.op in elem for p in parents.get(id(m), ()))
        ]
        if len(roots) != 1:
            stats["multi_root"] += 1
            continue
        rt = roots[0]
        # single-exit: every non-root member's consumers stay inside
        bad = False
        for m in members:
            if m is rt:
                continue
            for p in parents.get(id(m), ()):
                if id(p) not in member_ids:
                    bad = True
                    break
            if bad:
                break
        if bad:
            stats["multi_exit"] += 1
            continue
        canon = _canon_region(rt, members)
        if canon is None:
            stats["unreachable"] += 1
            continue
        order, specs, leaves = canon
        key = _region_key(order, specs)
        if key not in keys:
            keys[key] = len(keys)
        fuse_map[rt] = (keys[key], leaves, order, specs)
        stats["fused"] += 1
        stats["ops_fused"] += len(members)
    return fuse_map, keys, compiled, stats


# ---------------------------------------------------------------------------
#  Term rewrite + compiled-region registry
# ---------------------------------------------------------------------------


_REGISTRY: dict[int, object] = {}

#: ``compiled`` (default) or ``spelled`` — diagnostic arm selector.
_MODE = "compiled"


def _fused_region_binding(*xs, fused_k: int):
    """``fused_region`` op binding — dispatch to the compiled body."""
    return _REGISTRY[fused_k](*xs)


def _compile_region(order, specs, bindings):
    r"""``torch.compile``\ d unrolled region body — positional args.

    Every boundary leaf is a positional tensor input (Consts included
    — minted by the plan tape, never inlined into the spec), so the
    compiled graph is pure tensor ops and inductor's FakeTensorMode
    never sees a Const.  Body shape mirrors the probe's
    ``_compiled_region`` minus its Const special-casing.
    """

    member_pos = {id(t): i for i, t in enumerate(order)}
    arg_specs = []
    for m, ms in zip(order, specs, strict=True):
        row = []
        for kind, v in ms:
            row.append((kind, v))
        arg_specs.append((m, tuple(row)))
    root_i = member_pos[id(order[-1])]
    binds = bindings

    def spelled(*vals):
        memo = [None] * len(order)
        for i, (op, row) in enumerate(arg_specs):
            args = [
                memo[v] if kind == "m" else vals[v] for kind, v in row
            ]
            memo[i] = binds[op.op](*args, **dict(op.attrs))
        return memo[root_i]

    if _MODE == "spelled":
        # Diagnostic arm: the same unrolled body WITHOUT torch.compile
        # — isolates term-rewrite correctness from inductor numerics.
        return spelled
    return torch.compile(spelled, dynamic=False)


def build_fused(term, bindings, *, min_ops: int = 2):
    """Rewrite *term*: eligible region roots -> ``fused_region`` ops.

    Returns ``(new_term, n_fused_regions, n_distinct_kernels,
    skip_stats)``.  Compiled bodies are registered in ``_REGISTRY``
    keyed by structural dedup id.
    """
    from mlopt_core.ir import Op

    fuse_map, _keys, _compiled, stats = _collect_regions(
        term, min_ops=min_ops
    )
    if not fuse_map:
        return term, 0, 0, stats

    # Compile one body per distinct key (exemplar = first region).
    exemplars: dict[int, tuple] = {}
    for _rt, (kid, _leaves, order, specs) in fuse_map.items():
        if kid not in exemplars:
            exemplars[kid] = (order, specs)
    for kid, (order, specs) in exemplars.items():
        _REGISTRY[kid] = _compile_region(order, specs, bindings)

    memo: dict = {}

    def rewrite(t):
        if not isinstance(t, Op):
            return t
        hit = memo.get(t)
        if hit is not None:
            return hit
        fr = fuse_map.get(t)
        if fr is not None:
            kid, leaves, _o, _s = fr
            new_leaves = tuple(
                rewrite(lv) if isinstance(lv, Op) else lv
                for lv in leaves
            )
            try:
                out = Op.make(
                    "fused_region",
                    *new_leaves,
                    fused_k=kid,
                    validate=False,
                )
            except TypeError:  # pragma: no cover — attrs kwarg clash
                out = Op("fused_region", new_leaves, {"fused_k": kid})
        else:
            new_args = tuple(
                rewrite(a) if isinstance(a, Op) else a for a in t.args
            )
            if all(
                n is a for n, a in zip(new_args, t.args, strict=True)
            ):
                out = t
            else:
                try:
                    out = Op.make(
                        t.op, *new_args, validate=False, **t.attrs
                    )
                except TypeError:  # pragma: no cover
                    out = Op(t.op, new_args, dict(t.attrs))
        memo[t] = out
        return out

    new_term = rewrite(term)
    return (
        new_term,
        stats["fused"],
        len(exemplars),
        stats,
    )


def fused_op_table(ops):
    """Copy of *ops*' binding dict plus the ``fused_region`` dispatch."""
    from mlopt_core.ops import OpTable

    table = OpTable.full() if ops is None else ops
    tb = dict(table.torch_bindings)
    tb["fused_region"] = _fused_region_binding
    table.torch_bindings = tb
    return table


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def main() -> None:
    global _MODE
    which = sys.argv[1] if len(sys.argv) > 1 else "decoder"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    min_ops = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    if len(sys.argv) > 4:
        _MODE = sys.argv[4]
    packed = which.endswith("-p")
    if packed:
        which = which[: -len("-p")]
    dtype = "fp32"

    # Each exemplar region body is a distinct closure over the SAME
    # ``spelled`` code object, so dynamo counts every exemplar's first
    # compile against ONE recompile entry — the default limit of 8
    # silently degrades later regions to eager.  Region shapes are
    # bounded by the census; raise it for the measurement.
    torch._dynamo.config.recompile_limit = 256  # ty: ignore[unresolved-attribute]
    print(
        f"[arm] {which} packed={packed} dtype={dtype} "
        f"iters={iters} min_ops={min_ops} mode={_MODE}",
        flush=True,
    )
    m, wrap, inp, ref, ref_u1, ref_u2, _cap = kokoro._harvest(
        which, dtype, packed=packed
    )
    del m, _cap, ref_u1, ref_u2
    import gc as _gc0

    _gc0.collect()
    torch.cuda.empty_cache()

    from mlopt_orchestrator.optimize import Optimizer
    from kit_torch.backend import TorchBackend
    from kit_torch.torch_bridge import _IR_TO_TORCH

    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    res = optz.search(
        wrap,
        inp,
        max_iterations=iters,
        max_enodes=100_000,
    )
    print(
        f"[search] {time.time() - t0:.1f}s "
        f"enodes={getattr(res, 'n_enodes', '?')}",
        flush=True,
    )
    term = res.term

    # Restore the probe's creator monkeypatches BEFORE any in-process
    # torch.compile (inductor FakeTensorMode rejects the wrappers).
    for _name, _orig in kokoro._ORIG_CREATORS.items():
        setattr(torch, _name, _orig)
    torch._tts_probe_devcreators_patched = False  # ty: ignore[unresolved-attribute]

    # ---- region fuser ---------------------------------------------------
    t0 = time.time()
    fused_term, n_regions, n_kernels, rstats = build_fused(
        term, _IR_TO_TORCH, min_ops=min_ops
    )
    print(
        f"[fuse] {time.time() - t0:.1f}s regions_fused={n_regions} "
        f"distinct_kernels={n_kernels} skips={dict(rstats)}",
        flush=True,
    )
    if n_regions == 0:
        print("[fuse] nothing eligible — done", flush=True)
        return

    # ---- lower both sides ----------------------------------------------
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None  # ty: ignore[invalid-assignment]
    res.source = None  # ty: ignore[invalid-assignment]
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    low = optz.lower(res, inp, verify=False, runner=None)
    print(f"[lower] {time.time() - t0:.1f}s", flush=True)
    del_mod = low.module

    from mlopt_core.ir import IR
    from kit_torch.report import verify_module
    from kit_torch.torch_eval import ir_to_torch_module

    fused_ir = IR(
        root=fused_term,
        inputs=res.ir.inputs,
        input_names=res.ir.input_names,
        params=res.ir.params,
    )
    table = fused_op_table(None)
    t0 = time.time()
    fus_mod = ir_to_torch_module(
        fused_ir, param_values=res.param_values, ops=table
    )
    print(f"[lower-fused] {time.time() - t0:.1f}s", flush=True)

    # ---- verify: fused vs delivered (the real contract) -----------------
    print("[verify] fused vs delivered ...", flush=True)
    t0 = time.time()
    rep = verify_module(del_mod, fus_mod, inp, rtol=1e-4)
    print(
        f"[verify] {time.time() - t0:.1f}s passed={rep.passed} "
        f"max_abs={rep.max_abs:.3e} max_rel={rep.max_rel:.3e} "
        f"rtol={rep.rtol}",
        flush=True,
    )
    for extra_rtol in (1e-3, 2e-3, 1e-2):
        rep2 = verify_module(del_mod, fus_mod, inp, rtol=extra_rtol)
        print(
            f"[verify] rtol={extra_rtol:g} passed={rep2.passed} "
            f"max_rel={rep2.max_rel:.3e}",
            flush=True,
        )
    # delivered-vs-eager baseline (how much the SEARCH itself moved)
    with torch.no_grad():
        del_out = (
            del_mod(*inp) if isinstance(inp, tuple) else del_mod(inp)
        )
    if isinstance(ref, tuple):
        dd = max(
            (a.float() - b.float()).abs().max().item()
            for a, b in zip(ref, del_out, strict=True)
        )
    else:
        dd = (ref.float() - del_out.float()).abs().max().item()
    print(
        f"[verify] delivered vs eager-wrap max_abs={dd:.3e}", flush=True
    )
    if rep.per_output is not None:
        print(f"[verify] per_output={rep.per_output}", flush=True)
    # and vs the eager wrap reference
    with torch.no_grad():
        fus_out = (
            fus_mod(*inp) if isinstance(inp, tuple) else fus_mod(inp)
        )
    if isinstance(ref, tuple):
        d = max(
            (a.float() - b.float()).abs().max().item()
            for a, b in zip(ref, fus_out, strict=True)
        )
    else:
        d = (ref.float() - fus_out.float()).abs().max().item()
    print(f"[verify] fused vs eager-wrap max_abs={d:.3e}", flush=True)

    # ---- timing ---------------------------------------------------------
    call = lambda mm: (  # noqa: E731
        mm(*inp) if isinstance(inp, tuple) else mm(inp)
    )
    e_ms = kokoro.timeit(lambda: call(wrap))
    d_ms = kokoro.timeit(lambda: call(del_mod))
    f_ms = kokoro.timeit(lambda: call(fus_mod))
    print(
        f"[step] eager-wrap {e_ms:.2f} ms  delivered {d_ms:.2f} ms  "
        f"region-fused {f_ms:.2f} ms  (delta {d_ms - f_ms:+.2f} ms, "
        f"{d_ms / f_ms:.2f}x vs delivered)",
        flush=True,
    )

    del res, fused_term
    import gc as _gc

    _gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""Fusion-floor feasibility probe — measurement driver.

Follow-up to ``decode_step_fusion.md`` / ``tts_stt_coverage_map.md``
item #1: the conv/resblock/pointwise+norm chains launch one kernel
per op where ``torch.compile``/Inductor fuses them.  This probe
quantifies that wall on the whisper decode step *before* any
implementation:

1. **Term + tape census** — op histogram of the extracted term
   (``res.term``) and the delivered plan tape; maximal
   elementwise-connex regions (the ``genkernel.ELEMENTWISE``
   vocabulary — the same set the ``gen`` tier's ``_fusion_sites``
   collector uses) counted as fusion regions: launches now vs
   launches after a pointwise-fusion pass.

2. **Routing constraint** — carrier executors only engage
   carrier-apply ROOTED terms (``_route_spec`` /
   ``_carrier_upgrade`` in ``mlopt_orchestrator.pricing``); verified
   by reading, cited in the report.

3. **Measured ceiling** — for the largest fusion regions: spelled
   eval (one binding call per op — the plan-tape dispatch shape)
   vs the ``genkernel.gen_bindings`` ``torch.compile``d kernel —
   the actual shipped fusion-delivery mechanism — timed on the
   real harvested decode-step values.  Whole-step arm times the
   delivered ``IRModule`` and ``torch.compile(wrap)`` for the
   end-to-end reference.

4. **Kernel-time attribution** — torch.profiler CUDA-time
   categorisation of the delivered module (GEMM / cat / layernorm /
   sdpa-family / pointwise residue) so the fusion prize is bounded
   by measured device time, not launch counts.

All term traversal uses the DAG view (``mlopt_core.traversal.ops``)
— ``expand`` is the exponential tree spelling and is never called
on this sharing-heavy term.

Usage::

    PYTHONPATH=/tmp/xfm_pkgs .venv/bin/python \
        bench/drivers/probes/fusion_floor_probe.py [iters] [ntok] [L]
"""

from __future__ import annotations

import importlib.util
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "whisper_decode_loop", HERE / "whisper_decode_loop.py"
)
probe = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(probe)


def timeit(fn, warm=3, reps=20):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def _elem_ops():
    from mlopt_discovery import genkernel as gk

    return gk.ELEMENTWISE


def _region_census(term):
    """Maximal elementwise-connex regions of *term* (DAG view).

    Regions are weakly-connected components of the elementwise-op
    graph (edges = elementwise parent↔child).  Each component has a
    unique root — the member with no elementwise parent — whose
    value the fused kernel would compute from the region's boundary
    leaves (non-elementwise args of member ops).  Returns
    ``(regions, n_elementwise)`` with regions sorted by size desc;
    each region is ``(root, members, boundary_leaf_nodes)``.
    """
    from mlopt_core.ir import Const, Op
    from mlopt_core.traversal import ops as _ops

    elem = _elem_ops()
    nodes = [t for t in _ops(term) if t.op in elem]

    # parent lists over ALL ops (a node may have several parents)
    parents: dict[int, list] = defaultdict(list)
    all_ops = [t for t in _ops(term) if isinstance(t, Op)]
    for t in all_ops:
        for a in t.args:
            if isinstance(a, Op):
                parents[id(a)].append(t)

    # union-find over elementwise nodes
    uf: dict[int, int] = {id(t): id(t) for t in nodes}

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
            # DAG corner: report honestly rather than silently pick
            roots = sorted(
                roots, key=lambda r: -len(_subtree_ops(r, elem))
            )
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


def _subtree_ops(root, elem):
    """Count elementwise-Op descendants of *root* (DAG-deduped)."""
    from mlopt_core.ir import Op

    # collect the subtree by walking args (DAG-safe via id memo)
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


def _eval_region_spelled(root, leaf_tensors: dict, bindings):
    """Eval one fusion region op-by-op — the plan-tape dispatch shape.

    ``leaf_tensors`` maps ``id(boundary leaf node)`` → tensor; region
    interior is only elementwise ``Op``s and ``Const``s.
    """
    from mlopt_core.ir import Const, Op

    def go(t):
        hit = leaf_tensors.get(id(t))
        if hit is not None:
            return hit
        if isinstance(t, Op):
            return bindings[t.op](
                *[go(a) for a in t.args], **dict(t.attrs)
            )
        if isinstance(t, Const):
            return t.value
        raise TypeError(f"unexpected leaf in region: {type(t)}")

    return go(root)


def _region_leaves(root, members):
    """Boundary leaf nodes of a region (deduped, Consts excluded)."""
    from mlopt_core.ir import Const

    member_ids = {id(m) for m in members}
    leaves = {}
    for m in members:
        for a in m.args:
            if id(a) in member_ids or isinstance(a, Const):
                continue
            leaves.setdefault(id(a), a)
    return list(leaves.values())


def _eval_leaves(leaf_nodes, var_env, param_env, bindings, memo):
    """Evaluate boundary leaves once; shared memo dedups shared DAG."""
    from mlopt_torch.torch_eval import eval_term

    out = {}
    for leaf in leaf_nodes:
        out[id(leaf)] = eval_term(
            leaf,
            var_env=var_env,
            param_env=param_env,
            bindings=bindings,
            strict=True,
            memo_env=memo,
        )
    return out


def _fused_kernel_for(root):
    """Mint the gen-tier compiled kernel for this region.

    Returns ``(fn, metas, env)`` — ``fn(*leaf_tensors_in_meta_order)``
    is the ``torch.compile``d spelled body; ``env`` maps each metavar
    name to its bound leaf *node* (from ``_concrete_match`` against
    the region root itself).
    """
    from discoverykit.meta_arena import _concrete_match
    from mlopt_discovery import genkernel as gk
    from mlopt_discovery.play.sites import _fusion_pattern_sz

    pat, _size = _fusion_pattern_sz(root, {}, {}, None)
    metas = gk._spec_metas(pat)
    h = {"pattern": pat, "kernel": "k0", "args": tuple(metas)}
    bound = gk.gen_bindings({"k0": h}, compile_kernels=True)
    fn = bound["k0"]
    env: dict = {}
    assert _concrete_match(pat, root, env), "region pattern self-match"
    return fn, metas, env


def _cat_kernel_time(prof_rows):
    """Bucket profiler CUDA rows into fusion-relevant categories."""
    buckets = Counter()
    pats = [
        # inductor fused kernels first — their names embed the fused
        # op vocabulary (``triton_red_fused__softmax_…`` would
        # otherwise land in sdpa)
        ("triton_fused", ("triton_",)),
        (
            "gemm",
            ("gemm", "gemv", "sgemm", "cutlass", "nvjet", "matmul"),
        ),
        ("cat", ("cat", "CatArray", "BatchedCopy")),
        ("layernorm", ("layer_norm", "LayerNorm", "layernorm")),
        ("sdpa", ("fmha", "attention", "sdpa", "softmax")),
        ("memcpy", ("memcpy", "copy_", "CopyKernel")),
        (
            "pointwise",
            (
                "elementwise",
                "mul",
                "add",
                "gelu",
                "GELU",
                "relu",
                "sigmoid",
                "fill",
                "vectorized",
            ),
        ),
    ]
    for t, _c, k in prof_rows:
        kl = k.lower()
        for name, keys in pats:
            if any(s.lower() in kl for s in keys):
                buckets[name] += t
                break
        else:
            buckets["other"] += t
    return buckets


def main() -> None:
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    L = int(sys.argv[3]) if len(sys.argv) > 3 else 40

    import kit_carriers  # noqa: F401
    import mlopt_torch  # noqa: F401
    from mlopt_orchestrator.optimize import Optimizer
    from mlopt_torch.backend import TorchBackend
    from mlopt_torch.torch_bridge import _IR_TO_TORCH

    HARVEST = min(6, ntok - 2)
    m = probe.load()
    ehs = probe.encoder_out(m)
    _step_ms, ref = probe.greedy_loop_hf(
        m, ehs, ntok, harvest_at=HARVEST
    )
    print(
        f"[eager] harvest ok (P={ref['past']['sk'][0].shape[-2]})",
        flush=True,
    )

    inputs, _meta = probe.build_inputs(ref, L)
    wrap = probe._DecodeStepWrap(m, ref["ehs"], m.config).eval()
    with torch.no_grad():
        o = wrap(*inputs)
        print(
            f"[wrap] logits max|diff|="
            f"{(o[0][:, -1].float() - ref['logits']).abs().max().item():.2e}",
            flush=True,
        )

    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    res = optz.search(wrap, inputs, max_iterations=iters)
    print(f"[search] {time.time() - t0:.1f}s", flush=True)
    term = res.term

    # ---- 1. term census -------------------------------------------------
    from mlopt_core.ir import Op
    from mlopt_core.traversal import ops as _term_ops

    all_nodes = list(_term_ops(term))
    hist = Counter(
        t.op if isinstance(t, Op) else type(t).__name__
        for t in all_nodes
    )
    print(
        f"[term] nodes={len(all_nodes)} "
        f"ops={sum(1 for t in all_nodes if isinstance(t, Op))}"
    )
    for name, c in hist.most_common(30):
        print(f"    {name}: {c}")

    regions, n_elem = _region_census(term)
    region_ops = sum(len(r[1]) for r in regions)
    print(
        f"[regions] elementwise ops={n_elem} in {len(regions)} regions "
        f"(member ops={region_ops}); launches now≈{region_ops} "
        f"vs fused≈{len(regions)}  (-{region_ops - len(regions)})"
    )
    for i, (root, members, leaves) in enumerate(regions[:15]):
        ops_in = Counter(m.op for m in members)
        print(
            f"    region{i}: {len(members)} ops {dict(ops_in)} "
            f"boundary={len(leaves)} root={root.op}"
        )

    # the gen-tier collector's own verdict — sites need >=2 DISTINCT
    # ops in one elementwise-connex pattern
    from mlopt_discovery.play.sites import _fusion_sites

    fs = _fusion_sites(term)
    print(f"[fusion_sites] {_fusion_sites.__name__} sites={len(fs)}")
    for j, pat in enumerate(fs[:10]):
        print(f"    site{j}: {pat}")

    # free the e-graph before lowering (4 GB card)
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None  # ty: ignore[invalid-assignment]
    res.source = None  # ty: ignore[invalid-assignment]
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # ---- 2. plan tape ---------------------------------------------------
    t0 = time.time()
    low = optz.lower(res, inputs, verify=False, runner=None)
    print(f"[lower] {time.time() - t0:.1f}s", flush=True)
    mod = low.module
    plan = getattr(mod, "_root_plan", None)
    if plan is not None:
        steps = plan.steps
        phist = Counter(plan.op_names[fi] for _o, fi, _a, _k in steps)
        print(f"[plan] slots={plan.n_slots} steps={len(steps)}")
        for name, c in phist.most_common(25):
            print(f"    {name}: {c}")
    e_ms = timeit(lambda: wrap(*inputs))
    d_ms = timeit(lambda: mod(*inputs))
    print(
        f"[step] eager-wrap {e_ms:.2f} ms  delivered {d_ms:.2f} ms",
        flush=True,
    )

    # ---- 3. measured ceiling: per-region spelled vs fused --------------
    var_env = {
        v.name: t for v, t in zip(res.ir.inputs, inputs, strict=True)
    }
    param_env = res.param_values
    bindings = _IR_TO_TORCH

    results = []
    # only regions with >=2 member ops can fuse into fewer kernels;
    # singletons were measured in earlier runs (compiled call
    # overhead ~3-4x one eager op — fusion loses).
    candidates = [r for r in regions if len(r[1]) >= 2][:4]
    for i, (root, members, _leaves) in enumerate(candidates):
        try:
            leaf_nodes = _region_leaves(root, members)
            memo: dict = {}
            leaf_tensors = _eval_leaves(
                leaf_nodes, var_env, param_env, bindings, memo
            )
            with torch.no_grad():
                ref_out = _eval_region_spelled(
                    root, leaf_tensors, bindings
                )
            reps = max(50, min(400, int(2e5 / max(1, len(members)))))
            t_sp = timeit(
                lambda r=root, lt=leaf_tensors, b=bindings: (
                    _eval_region_spelled(r, lt, b)
                ),
                warm=5,
                reps=reps,
            )
            fn, metas, menv = _fused_kernel_for(root)
            # metavar name -> bound leaf node -> tensor (shared memo
            # dedups any prefix the leaf evals already materialised —
            # but memo keys are terms, tensors already computed are
            # NOT re-fetched; eval fresh is fine, leaves are few)
            mvals = [
                leaf_tensors.get(id(menv[mn]))
                if id(menv[mn]) in leaf_tensors
                else None
                for mn in metas
            ]
            missing = [
                mn
                for mn, v in zip(metas, mvals, strict=True)
                if v is None
            ]
            if missing:
                # metavar bound a leaf _region_leaves didn't list
                # (e.g. a repeated occurrence) — eval it directly
                for mn in missing:
                    leaf = menv[mn]
                    mvals[metas.index(mn)] = _eval_leaves(
                        [leaf], var_env, param_env, bindings, memo
                    )[id(leaf)]
            with torch.no_grad():
                fout = fn(*mvals)
            if isinstance(ref_out, tuple):
                diff = max(
                    (a.float() - b.float()).abs().max().item()
                    for a, b in zip(ref_out, fout, strict=True)
                )
            else:
                diff = (
                    (ref_out.float() - fout.float()).abs().max().item()
                )
            t_fu = timeit(
                lambda f=fn, v=mvals: f(*v), warm=5, reps=reps
            )
            results.append((i, len(members), t_sp, t_fu, diff))
            print(
                f"[region{i}] ops={len(members)} leaves={len(leaf_nodes)} "
                f"spelled={t_sp * 1e3:.1f}us fused={t_fu * 1e3:.1f}us "
                f"speedup={t_sp / t_fu:.2f}x diff={diff:.1e} reps={reps}",
                flush=True,
            )
        except Exception as ex:
            print(
                f"[region{i}] ops={len(members)} FAILED "
                f"{type(ex).__name__}: {ex}",
                flush=True,
            )

    if results:
        tot_sp = sum(r[2] for r in results)
        tot_fu = sum(r[3] for r in results)
        tot_ops = sum(r[1] for r in results)
        print(
            f"[regions-top] measured {len(results)} regions "
            f"({tot_ops} ops): spelled={tot_sp:.3f}ms "
            f"fused={tot_fu:.3f}ms per call",
            flush=True,
        )

    # ---- 4. profiler attribution of the delivered module --------------
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as tprof

    with torch.no_grad():
        for _ in range(3):
            mod(*inputs)
        torch.cuda.synchronize()
        with tprof(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(10):
                mod(*inputs)
            torch.cuda.synchronize()
    rows = []
    nk = 0
    for e in prof.key_averages():
        dev = getattr(e, "self_device_time_total", 0) or 0
        # CUDA-kernel rows only — CPU op rows (``aten::addmm`` etc.)
        # carry self_device_time_total for the kernels they launched
        # and would double-count every bucket.
        if e.device_type == DeviceType.CUDA:
            nk += e.count
            rows.append((dev, e.count, e.key))
    buckets = _cat_kernel_time(rows)
    tot_cuda = sum(buckets.values()) / 1e3 / 10
    print(
        f"[prof] delivered: kernels/call={nk / 10:.0f} "
        f"cuda={tot_cuda:.2f}ms/call",
        flush=True,
    )
    for name, t in buckets.most_common():
        print(f"    {name}: {t / 1e3 / 10:.3f} ms/call", flush=True)
    print("    [top kernels]")
    for t, c, k in sorted(rows, reverse=True)[:12]:
        print(
            f"      {t / 1e3:8.1f}us x{c:4d}  {k[:95]}",
            flush=True,
        )

    # ---- 5. whole-step reference ---------------------------------------
    cmod = torch.compile(wrap, dynamic=False)
    with torch.no_grad():
        cmod(*inputs)
        torch.cuda.synchronize()
    c_ms = timeit(lambda: cmod(*inputs))
    print(f"[step] torch.compile(wrap) {c_ms:.2f} ms", flush=True)

    with torch.no_grad():
        for _ in range(3):
            cmod(*inputs)
        torch.cuda.synchronize()
        with tprof(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as prof2:
            for _ in range(10):
                cmod(*inputs)
            torch.cuda.synchronize()
    rows2 = []
    nk2 = 0
    for e in prof2.key_averages():
        dev = getattr(e, "self_device_time_total", 0) or 0
        if e.device_type == DeviceType.CUDA:
            nk2 += e.count
            rows2.append((dev, e.count, e.key))
    buckets2 = _cat_kernel_time(rows2)
    tot2 = sum(buckets2.values()) / 1e3 / 10
    print(
        f"[prof] compile: kernels/call={nk2 / 10:.0f} cuda={tot2:.2f}ms/call",
        flush=True,
    )
    for name, t in buckets2.most_common():
        print(f"    {name}: {t / 1e3 / 10:.3f} ms/call", flush=True)
    print("    [top kernels]")
    for t, c, k in sorted(rows2, reverse=True)[:12]:
        print(
            f"      {t / 1e3:8.1f}us x{c:4d}  {k[:95]}",
            flush=True,
        )

    del m, res
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""Kokoro fusion-site census — measurement driver.

Follow-up to ``fusion_floor_probe.md``: the whisper decode step had
``_fusion_sites(term) == 0`` (pointwise-fusion carrier thesis dead
for STT), but Kokoro's conv/resblock decoder was explicitly flagged
as unmeasured — its SiLU-mul spellings, conv1d+pointwise chains and
norm epilogues plausibly produce real fusion sites.

Per arm (``decoder`` complex TorchSTFT / ``decoder-nc`` CustomSTFT /
``f0n`` / ``textenc-p``):

1. ``tts_kokoro_probe._harvest`` → ``Optimizer.search`` → the
   delivered term (``res.term``).
2. Op histogram + ``_region_census`` (elementwise-connex regions,
   the ``genkernel.ELEMENTWISE`` vocabulary) + the gen-tier
   collector's own verdict ``_fusion_sites(term)``.
3. Measured ceiling: for the largest regions (>=2 member ops),
   spelled eval (one binding call per op — the plan-tape dispatch
   shape) vs the ``genkernel.gen_bindings`` ``torch.compile``d
   kernel — the shipped fusion-delivery mechanism — timed on the
   real harvested tensors.  Same machinery as
   ``fusion_floor_probe.py`` (imported, not duplicated).
4. Delivered-module wall times (eager wrap vs ``IRModule``) and a
   torch.profiler CUDA-kernel attribution (conv vs pointwise vs
   gemm buckets) for the decoder arms — is the wall conv-bound or
   fusion-bound?

Usage::

    PYTHONPATH=/tmp/xfm_pkgs .venv/bin/python \
        bench/drivers/probes/kokoro_fusion_probe.py <which> [iters] [compile]

``which``: decoder | decoder-nc | f0n | textenc | textenc-p
           (``-p`` = the packed-LSTM custom_op wrap from
           ``tts_kokoro_packed``; ``textenc`` without ``-p`` walls
           at PackedSequence export — use ``textenc-p``)
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
ffp = _load("fusion_floor_probe", "fusion_floor_probe.py")


def _region_census(term):
    """``fusion_floor_probe._region_census`` with the DAG-corner fix.

    Upstream sorts multi-root regions by ``len(_subtree_ops(...))`` —
    ``_subtree_ops`` already returns an int count, so ``len()`` raises
    ``TypeError``.  Kokoro's conv/resblock DAG hits that path (whisper's
    did not).  Identical logic otherwise; upstream file left untouched.
    """
    from collections import defaultdict

    from mlopt_core.ir import Const, Op
    from mlopt_core.traversal import ops as _ops

    elem = ffp._elem_ops()
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
            roots = sorted(
                roots, key=lambda r: -ffp._subtree_ops(r, elem)
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


def _compiled_region(root, members):
    """torch.compile'd fused region — positional-args fallback.

    ``genkernel.gen_bindings`` inlines Const *tensor* leaves into
    the compiled spec and inductor's FakeTensorMode rejects them
    (``aten.expand.default(tensor(...))``); lifting tensor Consts
    to positional inputs sidesteps that.  The body is unrolled to
    a flat op list with constant indices so dynamo traces it
    cleanly — one inductor kernel for the whole elementwise DAG.
    """
    from mlopt_core.ir import Const, Op
    from mlopt_torch.torch_bridge import _IR_TO_TORCH as bindings

    member_ids = {id(m) for m in members}
    # topo order over members (post-order DFS from root)
    order = []
    seen = set()

    def dfs(t):
        if id(t) in seen or id(t) not in member_ids:
            return
        seen.add(id(t))
        for a in t.args:
            if isinstance(a, Op):
                dfs(a)
        order.append(t)

    dfs(root)
    pos = {id(t): i for i, t in enumerate(order)}
    # inputs: boundary leaves + tensor-valued Consts, in order
    inputs = []
    input_idx: dict[int, int] = {}

    def need(node):
        if id(node) not in input_idx:
            input_idx[id(node)] = len(inputs)
            inputs.append(node)

    arg_specs = []
    for op in order:
        specs = []
        for a in op.args:
            if isinstance(a, Op) and id(a) in pos:
                specs.append(("op", pos[id(a)]))
            elif isinstance(a, Const):
                v = a.value
                if torch.is_tensor(v):
                    need(a)
                    specs.append(("in", input_idx[id(a)]))
                else:
                    specs.append(("c", v))
            else:
                need(a)
                specs.append(("in", input_idx[id(a)]))
        arg_specs.append((op, specs))
    root_i = pos[id(root)]
    binds = bindings

    def spelled(*vals):
        memo = [None] * len(order)
        for i, (op, specs) in enumerate(arg_specs):
            args = []
            for kind, v in specs:
                if kind == "op":
                    args.append(memo[v])
                elif kind == "in":
                    args.append(vals[v])
                else:
                    args.append(v)
            memo[i] = binds[op.op](
                *args, **dict(op.attrs)
            )
        return memo[root_i]

    # materialise input tensors: leaves via eval_term, Consts
    # via .value — caller merges both into one vals tuple.
    return torch.compile(spelled, dynamic=False), inputs, order


def _cat_kernel_time(rows):
    """``fusion_floor_probe._cat_kernel_time`` + a conv bucket.

    conv kernels (``implicit_convolve_*``, ``volta_scudnn_*conv*``,
    ``sm80_xmma_*conv*``) contain "gemm"/"cutlass" substrings, so the
    conv check MUST run before gemm — checked first here.
    """
    buckets = Counter()
    pats = [
        ("triton_fused", ("triton_",)),
        (
            "conv",
            (
                "conv",
                "implicit_convolve",
                "scudnn",
                "cudnn",
                "volta",
                "sm80",
                "winograd",
                "fft",
            ),
        ),
        (
            "gemm",
            ("gemm", "gemv", "sgemm", "nvjet", "matmul"),
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
                "silu",
                "fill",
                "vectorized",
                "unrolled_elementwise",
            ),
        ),
    ]
    for t, _c, k in rows:
        kl = k.lower()
        for name, keys in pats:
            if any(s.lower() in kl for s in keys):
                buckets[name] += t
                break
        else:
            buckets["other"] += t
    return buckets


def _profile(mod, inp, reps=10, warm=3):
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as tprof

    def call():
        return mod(*inp) if isinstance(inp, tuple) else mod(inp)

    with torch.no_grad():
        for _ in range(warm):
            call()
        torch.cuda.synchronize()
        with tprof(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(reps):
                call()
            torch.cuda.synchronize()
    rows = []
    nk = 0
    for e in prof.key_averages():
        dev = getattr(e, "self_device_time_total", 0) or 0
        if e.device_type == DeviceType.CUDA:
            nk += e.count
            rows.append((dev, e.count, e.key))
    return nk, rows, reps


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "decoder"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    do_compile = len(sys.argv) > 3 and sys.argv[3] == "compile"
    packed = which.endswith("-p")
    if packed:
        which = which[: -len("-p")]
    dtype = "fp32"

    print(f"[arm] {which} packed={packed} dtype={dtype}", flush=True)
    m, wrap, inp, _ref, _ref_u1, _ref_u2, _cap = kokoro._harvest(
        which, dtype, packed=packed
    )
    # The full KModel holds text_encoder/predictor/bert — none of it
    # is needed past harvest (``wrap`` keeps only the arm submodule,
    # ``res.param_values`` keeps the arm's params).  Free it before
    # the region measurements on the 4 GB card.
    del m, _cap, _ref, _ref_u1, _ref_u2
    import gc as _gc0

    _gc0.collect()
    torch.cuda.empty_cache()
    print(
        "[arm] inputs: "
        + ", ".join(
            f"{tuple(x.shape) if torch.is_tensor(x) else type(x).__name__}"
            for x in (inp if isinstance(inp, tuple) else (inp,))
        ),
        flush=True,
    )

    from mlopt_orchestrator.optimize import Optimizer
    from mlopt_torch.backend import TorchBackend
    from mlopt_torch.torch_bridge import _IR_TO_TORCH

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

    # ``tts_kokoro_probe.load`` monkeypatches ``torch.rand/zeros/…``
    # (device-kwarg export workaround).  The wrappers call
    # ``ref.expand`` on a real CUDA tensor, which inductor's
    # FakeTensorMode rejects — EVERY torch.compile in this process
    # dies until the originals are restored.  Search/lower are done;
    # restore now so region kernels and the compile reference run.
    for _name, _orig in kokoro._ORIG_CREATORS.items():
        setattr(torch, _name, _orig)
    torch._tts_probe_devcreators_patched = False  # ty: ignore[unresolved-attribute]

    # ---- 1. term census -------------------------------------------------
    from mlopt_core.ir import Const, Op
    from mlopt_core.traversal import ops as _term_ops

    all_nodes = list(_term_ops(term))
    hist = Counter(
        t.op if isinstance(t, Op) else type(t).__name__
        for t in all_nodes
    )
    print(
        f"[term] nodes={len(all_nodes)} "
        f"ops={sum(1 for t in all_nodes if isinstance(t, Op))}",
        flush=True,
    )
    for name, c in hist.most_common(40):
        print(f"    {name}: {c}", flush=True)

    regions, n_elem = _region_census(term)
    region_ops = sum(len(r[1]) for r in regions)
    print(
        f"[regions] elementwise ops={n_elem} in {len(regions)} regions "
        f"(member ops={region_ops}); launches now≈{region_ops} "
        f"vs fused≈{len(regions)}  (-{region_ops - len(regions)})",
        flush=True,
    )
    for i, (root, members, leaves) in enumerate(regions[:20]):
        ops_in = Counter(mm.op for mm in members)
        print(
            f"    region{i}: {len(members)} ops {dict(ops_in)} "
            f"boundary={len(leaves)} root={root.op}",
            flush=True,
        )

    from kit_discovery.play.sites import _fusion_sites

    fs = _fusion_sites(term)
    print(f"[fusion_sites] sites={len(fs)}", flush=True)
    for j, pat in enumerate(fs[:10]):
        print(f"    site{j}: {pat}", flush=True)

    # ---- 2. lower + delivered timing -----------------------------------
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None  # ty: ignore[invalid-assignment]
    res.source = None  # ty: ignore[invalid-assignment]
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    low = optz.lower(res, inp, verify=False, runner=None)
    print(f"[lower] {time.time() - t0:.1f}s", flush=True)
    mod = low.module

    def call(mm, i):
        return mm(*i) if isinstance(i, tuple) else mm(i)

    e_ms = kokoro.timeit(lambda: call(wrap, inp))
    d_ms = kokoro.timeit(lambda: call(mod, inp))
    print(
        f"[step] eager-wrap {e_ms:.2f} ms  delivered {d_ms:.2f} ms",
        flush=True,
    )

    # ---- 3. per-region spelled vs fused --------------------------------
    var_env = {
        v.name: t for v, t in zip(res.ir.inputs, inp, strict=True)
    }
    param_env = res.param_values
    bindings = _IR_TO_TORCH

    results = []
    # Dedup near-identical regions by op histogram — Kokoro repeats
    # the same resblock/SineGen shape ~10x; measuring each copy
    # wastes the 4 GB card.  Keep the largest of each shape.
    seen_sig = set()
    candidates = []
    for r in regions:
        if len(r[1]) < 2:
            continue
        sig = tuple(
            sorted(Counter(mm.op for mm in r[1]).items())
        )
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        candidates.append(r)
        if len(candidates) >= 5:
            break
    print(
        f"[regions] measuring {len(candidates)} distinct regions "
        f"(of {sum(1 for r in regions if len(r[1]) >= 2)} with >=2 ops; "
        f"{len(regions)} total)",
        flush=True,
    )


    results = []
    for i, (root, members, _leaves) in enumerate(candidates):
        leaf_nodes = None
        try:
            leaf_nodes = ffp._region_leaves(root, members)
            # Per-leaf fresh memo: a shared memo_env retains EVERY
            # intermediate of the whole decoder prefix (~GBs on the
            # 4 GB card) — the whisper probe never hit this because
            # its leaves sat near the term root.
            leaf_tensors = {}
            for leaf in leaf_nodes:
                leaf_tensors.update(
                    ffp._eval_leaves(
                        [leaf], var_env, param_env, bindings, {}
                    )
                )
            with torch.no_grad():
                ref_out = ffp._eval_region_spelled(
                    root, leaf_tensors, bindings
                )
            reps = max(50, min(400, int(2e5 / max(1, len(members)))))
            t_sp = kokoro.timeit(
                lambda r=root, lt=leaf_tensors, b=bindings: (
                    ffp._eval_region_spelled(r, lt, b)
                ),
                warm=5,
                reps=reps,
            )
            # fused path: positional-args torch.compile'd region
            # (genkernel's spec-inlined Const tensors break inductor —
            # see _compiled_region docstring).
            cfn, cinputs, _order = _compiled_region(root, members)
            dev = next(
                iter(leaf_tensors.values())
            ).device
            mvals = []
            for n in cinputs:
                if isinstance(n, Const):
                    v = n.value
                    if torch.is_tensor(v):
                        v = v.to(dev)
                    mvals.append(v)
                else:
                    mvals.append(leaf_tensors[id(n)])
            with torch.no_grad():
                fout = cfn(*mvals)
            if isinstance(ref_out, tuple):
                diff = max(
                    (a.float() - b.float()).abs().max().item()
                    for a, b in zip(ref_out, fout, strict=True)
                )
            else:
                diff = (
                    (ref_out.float() - fout.float()).abs().max().item()
                )
            t_fu = kokoro.timeit(
                lambda f=cfn, v=mvals: f(*v), warm=5, reps=reps
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
                f"{type(ex).__name__}: {str(ex)[:200]}",
                flush=True,
            )
        finally:
            leaf_nodes = None
            leaf_tensors = None  # ty: ignore[possibly-undefined]
            memo = None  # ty: ignore[possibly-undefined]
            ref_out = None  # ty: ignore[possibly-undefined]
            fout = None  # ty: ignore[possibly-undefined]
            mvals = None  # ty: ignore[possibly-undefined]
            cfn = None  # ty: ignore[possibly-undefined]
            import gc as _gc

            _gc.collect()
            torch.cuda.empty_cache()

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
    try:
        nk, rows, reps = _profile(mod, inp)
    except Exception as ex:
        print(
            f"[prof] FAILED {type(ex).__name__}: {str(ex)[:160]}",
            flush=True,
        )
        nk, rows, reps = 0, [], 1
    buckets = _cat_kernel_time(rows)
    tot_cuda = sum(buckets.values()) / 1e3 / reps
    print(
        f"[prof] delivered: kernels/call={nk / reps:.0f} "
        f"cuda={tot_cuda:.2f}ms/call",
        flush=True,
    )
    for name, t in buckets.most_common():
        print(f"    {name}: {t / 1e3 / reps:.3f} ms/call", flush=True)
    print("    [top kernels]", flush=True)
    for t, c, k in sorted(rows, reverse=True)[:15]:
        print(
            f"      {t / 1e3:8.1f}us x{c:4d}  {k[:95]}",
            flush=True,
        )

    # ---- 5. optional whole-module compile reference --------------------
    if do_compile:
        try:
            cmod = torch.compile(wrap, dynamic=False)
            call(cmod, inp)
            torch.cuda.synchronize()
            c_ms = kokoro.timeit(lambda: call(cmod, inp))
            print(
                f"[step] torch.compile(wrap) {c_ms:.2f} ms", flush=True
            )
        except Exception as ex:
            print(
                f"[compile] FAILED {type(ex).__name__}: {ex}",
                flush=True,
            )

    del res
    import gc as _gc

    _gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

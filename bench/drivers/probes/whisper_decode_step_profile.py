"""Decode-step dispatch profile — measurement driver.

Follow-up to ``whisper_decode_loop.md`` / ``decode_step_fusion.md``.
Reuses ``whisper_decode_loop``'s wrap/harvest machinery and asks WHERE
the delivered step's ~5.5 ms goes:

* plan-tape step count + op histogram for the delivered ``IRModule``;
* torch.profiler kernel count / CUDA time / CPU time for delivered,
  eager-wrap and ``torch.compile`` arms;
* CUDA-graph arm decomposed: input-copy time vs ``graph.replay()`` time,
  plus a *zero-copy* variant that captures directly on the loop's
  persistent K/V buffers (24 of 26 inputs never change identity — only
  ``hs``/``mask`` do), and a *bundled* variant packing all 24 past
  tensors into one flat buffer (1 copy instead of 24).

Usage::

    .venv/bin/python bench/drivers/probes/whisper_decode_step_profile.py [iters] [ntok] [L]
"""

from __future__ import annotations

import importlib.util
import sys
import time
from collections import Counter
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


def profile(fn, warm=3, reps=10):
    """Return (kernels, cuda_ms, cpu_ms, op histogram) via torch.profiler."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
    n_kernels = 0
    cuda_us = 0.0
    cpu_us = 0.0
    op_count: Counter = Counter()
    cuda_rows = []
    for e in prof.key_averages():
        dev_t = getattr(e, "self_device_time_total", 0) or 0
        cpu_t = getattr(e, "cpu_time_total", 0) or 0
        if e.device_type == DeviceType.CUDA or dev_t:
            n_kernels += e.count
            cuda_us += dev_t
            cuda_rows.append((dev_t, e.count, e.key))
        else:
            cpu_us += cpu_t
        op_count[e.key] += e.count
    top_cuda = sorted(cuda_rows, reverse=True)[:15]
    return (
        n_kernels,
        cuda_us / 1e3 / reps,
        cpu_us / 1e3 / reps,
        op_count,
        top_cuda,
    )


def main() -> None:
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    L = int(sys.argv[3]) if len(sys.argv) > 3 else max(40, ntok)

    import kit_carriers  # noqa: F401
    import kit_torch  # noqa: F401
    from kit_cuda import CudaGraphRunner
    from mlopt_orchestrator.optimize import (
        Optimizer,
        VerifyDeclinedError,
    )
    from kit_torch.backend import TorchBackend

    HARVEST = min(8, ntok - 2)
    m = probe.load()
    ehs = probe.encoder_out(m)
    step_ms, ref = probe.greedy_loop_hf(
        m, ehs, ntok, harvest_at=HARVEST
    )
    print(
        f"[eager] warm {sum(step_ms[1:]) / (ntok - 1):.2f} ms/tok",
        flush=True,
    )

    inputs, _meta = probe.build_inputs(ref, L)
    wrap = probe._DecodeStepWrap(m, ref["ehs"], m.config).eval()
    with torch.no_grad():
        o = wrap(*inputs)
        print(
            f"[wrap] logits max|diff|={(o[0][:, -1].float() - ref['logits']).abs().max().item():.2e}",
            flush=True,
        )

    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    res = optz.search(wrap, inputs, max_iterations=iters)
    print(f"[search] {time.time() - t0:.1f}s", flush=True)
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None
    res.source = None
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    try:
        low = optz.lower(res, inputs, verify=True, runner=None)
    except VerifyDeclinedError as e:
        # declined verify still carries the LowerResult — record
        # verified=False and keep measuring, as pre-gate lower did
        low = e.result
    print(
        f"[lower] verified={low.verified.passed if low.verified else None}",
        flush=True,
    )
    mod = low.module

    # ---- plan tape introspection ----
    plan = getattr(mod, "_root_plan", None)
    if plan is not None:
        steps = plan.steps
        hist = Counter(
            mod._ops and plan.op_names[fi] for _o, fi, _a, _k in steps
        )
        print(
            f"[plan] slots={plan.n_slots} steps={len(steps)} ops={len(plan.op_names)}"
        )
        for name, c in hist.most_common(25):
            print(f"    {name}: {c}")
        # how many steps have kwargs / >3 args
        kw = sum(1 for _o, _f, _a, k in steps if k is not None)
        print(f"[plan] steps-with-attrs={kw}")
    else:
        print("[plan] root unplannable")

    # ---- timing + profiler per arm ----
    e_ms = timeit(lambda: wrap(*inputs))
    d_ms = timeit(lambda: mod(*inputs))
    print(
        f"[step] eager-wrap {e_ms:.2f} ms  delivered {d_ms:.2f} ms",
        flush=True,
    )

    cmod = torch.compile(wrap, dynamic=False)
    with torch.no_grad():
        cmod(*inputs)
        torch.cuda.synchronize()
    c_ms = timeit(lambda: cmod(*inputs))
    print(f"[step] torch.compile {c_ms:.2f} ms", flush=True)

    for tag, fn in [
        ("eager-wrap", lambda: wrap(*inputs)),
        ("delivered", lambda: mod(*inputs)),
        ("compile", lambda: cmod(*inputs)),
    ]:
        nk, cuda_ms, cpu_ms, _ops, top_cuda = profile(fn)
        print(
            f"[prof] {tag}: kernels/call={nk:.0f} cuda={cuda_ms:.2f}ms "
            f"cpu={cpu_ms:.2f}ms",
            flush=True,
        )
        for t, c, k in top_cuda:
            print(
                f"      {t / 1e3:9.1f}us tot x{c:5d}  {k[:90]}",
                flush=True,
            )

    # ---- cuda graph decomposition ----
    low2 = optz.lower(
        res, inputs, verify=False, runner=CudaGraphRunner()
    )
    gmod = low2.module
    print(f"[graph] armed={low2.stats.get('cuda_graph')}", flush=True)
    if low2.stats.get("cuda_graph"):
        g_ms = timeit(lambda: gmod(*inputs))
        print(f"[step] delivered+graph {g_ms:.2f} ms", flush=True)

        # gmod is _GraphReplayModule: _graph_ins list, _graph CUDAGraph
        ins = gmod._graph_ins
        graph = gmod._graph
        xs = inputs

        def copies_only():
            for buf, t in zip(ins, xs, strict=True):
                buf.copy_(t, non_blocking=True)

        def copies_plus_replay():
            copies_only()
            graph.replay()

        def replay_only():
            graph.replay()

        for tag, fn in [
            ("copies", copies_only),
            ("copies+replay", copies_plus_replay),
            ("replay", replay_only),
        ]:
            print(
                f"[graph-prof] {tag}: {timeit(fn):.3f} ms", flush=True
            )

        # profile the replay alone — what does the GPU actually run?
        nk, cuda_ms, cpu_ms, _ops, top_cuda = profile(
            replay_only, warm=2, reps=10
        )
        print(
            f"[prof] graph-replay: kernels/call={nk / 10:.0f} cuda={cuda_ms:.2f}ms cpu={cpu_ms:.2f}ms",
            flush=True,
        )
        for t, c, k in top_cuda:
            print(
                f"      {t / 1e3:9.1f}us tot x{c:5d}  {k[:90]}",
                flush=True,
            )

        # correctness of replay-only vs delivered (outputs are static buffers)
        with torch.no_grad():
            ref_out = mod(*inputs)
            copies_plus_replay()
            torch.cuda.synchronize()
            go = gmod._graph_out
            if isinstance(go, tuple):
                d = max(
                    (a.float() - b.float()).abs().max().item()
                    for a, b in zip(go, ref_out, strict=False)
                )
            else:
                d = (go.float() - ref_out.float()).abs().max().item()
            print(
                f"[graph] replay vs delivered max|diff|={d:.2e}",
                flush=True,
            )

    # ---- shipped alias-aware capture (kit_cuda alias_inputs) ----
    # Same contract as _ZeroCopyGraphCapture but through the runner:
    # positions >= 2 (the 24 past tensors) bind by reference; hs/mask
    # (0,1) stay on the copy path.  ``inputs`` objects are stable
    # across these calls, so every replay takes the alias path.
    try:
        low3 = optz.lower(
            res,
            inputs,
            verify=False,
            runner=CudaGraphRunner(
                alias_inputs=lambda i, _t: i >= 2
            ),
        )
        amod = low3.module
        print(
            f"[graph] alias armed={low3.stats.get('cuda_graph')} "
            f"alias={low3.stats.get('cuda_graph_alias')} "
            f"flags={getattr(amod, '_alias', None)}",
            flush=True,
        )
        if low3.stats.get("cuda_graph"):
            a_ms = timeit(lambda: amod(*inputs))
            print(
                f"[step] delivered+alias-graph {a_ms:.2f} ms",
                flush=True,
            )
            with torch.no_grad():
                ref_out = mod(*inputs)
                amod(*inputs)
                torch.cuda.synchronize()
                ao = amod._graph_out
                if isinstance(ao, tuple):
                    d = max(
                        (a.float() - b.float()).abs().max().item()
                        for a, b in zip(ao, ref_out, strict=False)
                    )
                else:
                    d = (ao.float() - ref_out.float()).abs().max().item()
                print(
                    f"[graph] alias replay vs delivered max|diff|={d:.2e}",
                    flush=True,
                )
    except Exception as e:
        print(
            f"[graph] alias FAILED {type(e).__name__}: {e}",
            flush=True,
        )

    # ---- zero-copy graph: capture directly on persistent buffers ----
    # In run_delivered_loop the 24 past tensors are the SAME buffer
    # objects every step (skb/svb/ck/cv); only hs and mask change.
    # Capture the INNER module on clones once, then... no — capture on
    # the actual loop buffers so replay reads current contents with NO
    # copies for them.  hs/mask still need 2 copies into static bufs.
    try:
        zmod = _ZeroCopyGraphCapture.capture(mod, inputs)
        z_ms = timeit(lambda: zmod(*inputs))
        print(f"[graph] zero-copy-24 {z_ms:.3f} ms", flush=True)
        with torch.no_grad():
            ref_out = mod(*inputs)
            zmod(*inputs)
            torch.cuda.synchronize()
            zo = zmod._graph_out
            if isinstance(zo, tuple):
                d = max(
                    (a.float() - b.float()).abs().max().item()
                    for a, b in zip(zo, ref_out, strict=False)
                )
            else:
                d = (zo.float() - ref_out.float()).abs().max().item()
            print(f"[graph] zero-copy max|diff|={d:.2e}", flush=True)
    except Exception as e:
        print(
            f"[graph] zero-copy FAILED {type(e).__name__}: {e}",
            flush=True,
        )

    # ---- zero-copy graph inside the REAL greedy loop ----
    # Capture on the loop's own persistent K/V buffers: replay reads
    # their live contents, so only hs/mask are copied per step.
    try:
        ck = list(ref["past"]["ck"])
        cv = list(ref["past"]["cv"])
        dec = m.model.decoder
        skb = [
            torch.zeros(1, probe.NH, L, probe.HDIM, device="cuda")
            for _ in range(probe.NLAYERS)
        ]
        svb = [
            torch.zeros(1, probe.NH, L, probe.HDIM, device="cuda")
            for _ in range(probe.NLAYERS)
        ]
        ids = torch.tensor(
            [[m.generation_config.decoder_start_token_id or 50257]],
            device="cuda",
        )
        # capture on step-0 inputs (skb/svb still zero — correct for t=0)
        hs0 = probe.embed_step(dec, ids, 0)
        m0 = probe.step_mask(0, L)
        zcap = _ZeroCopyGraphCapture.capture(
            mod, (hs0, m0, *skb, *svb, *ck, *cv)
        )
        print("[graph] zero-copy capture ok", flush=True)
        step_ms_z = []
        gids_z = []
        with torch.no_grad():
            for t in range(ntok):
                hs = probe.embed_step(dec, ids, t)
                past = (*skb, *svb, *ck, *cv)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                outs = zcap(hs, probe.step_mask(t, L), *past)
                torch.cuda.synchronize()
                step_ms_z.append((time.perf_counter() - t0) * 1e3)
                logits = outs[0][:, -1]
                for i in range(probe.NLAYERS):
                    knew = outs[1 + 2 * i]
                    vnew = outs[2 + 2 * i]
                    if t < L:
                        skb[i][:, :, t : t + 1, :] = knew
                        svb[i][:, :, t : t + 1, :] = vnew
                ids = logits.argmax(-1, keepdim=True)
                gids_z.append(ids[:, -1].item())
        agree = sum(
            1
            for a, b in zip(gids_z, ref["gen_ids"], strict=False)
            if a == b
        )
        warm = sum(step_ms_z[1:])
        print(
            f"[loop] zero-copy-graph {ntok} tok: warm {warm:.1f} ms "
            f"({warm / (ntok - 1):.2f} ms/tok) last8 {sum(step_ms_z[-8:]) / 8:.2f} "
            f"ids-match {agree}/{ntok}",
            flush=True,
        )
        print(
            "[loop] zero-copy per-step:",
            " ".join(f"{v:.1f}" for v in step_ms_z),
            flush=True,
        )
    except Exception as e:
        print(
            f"[graph] zero-copy LOOP FAILED {type(e).__name__}: {e}",
            flush=True,
        )

    # ---- shipped alias-aware graph inside the REAL greedy loop ----
    # multi_device capture is lazy: the FIRST loop call carries the
    # loop's own persistent buffers (skb/svb/ck/cv), so positions
    # >= 2 bind exactly those storages — the shipped equivalent of
    # _ZeroCopyGraphCapture above.
    try:
        dmod = CudaGraphRunner(
            multi_device=True,
            alias_inputs=lambda i, _t: i >= 2,
        ).apply(mod, inputs, {})
        step_ms_a, gids_a = probe.run_delivered_loop(
            dmod, m, ehs, ntok, L,
            list(ref["past"]["ck"]), list(ref["past"]["cv"]),
        )
        agree = sum(
            1
            for a, b in zip(gids_a, ref["gen_ids"], strict=False)
            if a == b
        )
        warm = sum(step_ms_a[1:])
        print(
            f"[loop] shipped-alias-graph {ntok} tok: warm {warm:.1f} ms "
            f"({warm / (ntok - 1):.2f} ms/tok) last8 {sum(step_ms_a[-8:]) / 8:.2f} "
            f"ids-match {agree}/{ntok}",
            flush=True,
        )
        print(
            "[loop] shipped-alias per-step:",
            " ".join(f"{v:.1f}" for v in step_ms_a),
            flush=True,
        )
    except Exception as e:
        print(
            f"[graph] shipped-alias LOOP FAILED {type(e).__name__}: {e}",
            flush=True,
        )

    # ---- copy-everything graph in the same loop (honest delta) ----
    if low2.stats.get("cuda_graph"):
        try:
            step_ms_c, gids_c = probe.run_delivered_loop(
                gmod, m, ehs, ntok, L,
                list(ref["past"]["ck"]), list(ref["past"]["cv"]),
            )
            agree = sum(
                1
                for a, b in zip(gids_c, ref["gen_ids"], strict=False)
                if a == b
            )
            warm = sum(step_ms_c[1:])
            print(
                f"[loop] shipped-copy-graph {ntok} tok: warm {warm:.1f} ms "
                f"({warm / (ntok - 1):.2f} ms/tok) last8 {sum(step_ms_c[-8:]) / 8:.2f} "
                f"ids-match {agree}/{ntok}",
                flush=True,
            )
        except Exception as e:
            print(
                f"[graph] shipped-copy LOOP FAILED {type(e).__name__}: {e}",
                flush=True,
            )

    del m
    import gc as _gc

    _gc.collect()
    torch.cuda.empty_cache()


class _ZeroCopyGraphCapture(torch.nn.Module):
    """CUDA graph over ``module`` whose 24 past-tensor inputs alias the
    caller's persistent buffers — replay sees their CURRENT contents
    with zero copies.  Only ``hs``/``mask`` (first two inputs) get
    static buffers + per-call copies.

    Deliberately a probe-local device: the shipped
    ``_GraphReplayModule`` clones every input for safety; this variant
    is honest only when the caller guarantees the aliased tensors are
    the same objects every call (the decode loop's K/V buffers are).
    """

    def __init__(self, inner, graph, static_head, static_out, tail):
        super().__init__()
        self.inner = inner
        self._graph = graph
        self._static_head = static_head  # [hs_buf, mask_buf]
        self._graph_out = static_out
        self._tail = tail  # the exact tail tensor objects captured

    @classmethod
    def capture(cls, module, xs, warmup=3):
        # xs: (hs, mask, *24 past).  Tail tensors are used AS the graph
        # inputs — NOT cloned — so replay reads live buffer contents.
        hs_buf = xs[0].detach().clone()
        mask_buf = xs[1].detach().clone()
        call_ins = (hs_buf, mask_buf, *xs[2:])
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(warmup):
                module(*call_ins)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            out = module(*call_ins)
        return cls(module, g, [hs_buf, mask_buf], out, xs[2:])

    def _matches(self, xs):
        return len(xs) == self._n_total() and all(
            a is b for a, b in zip(xs[2:], self._tail, strict=True)
        )

    def _n_total(self):
        return 2 + len(self._tail)

    def forward(self, *xs):
        # The tail tensors must be the SAME objects captured — replay
        # reads their live storage, so a different object would silently
        # compute on stale data.  Identity check is ~24 pointer compares.
        if not self._matches(xs):
            return self.inner(*xs)
        self._static_head[0].copy_(xs[0], non_blocking=True)
        self._static_head[1].copy_(xs[1], non_blocking=True)
        self._graph.replay()
        return self._graph_out


if __name__ == "__main__":
    main()

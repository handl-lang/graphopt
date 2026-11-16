"""Kokoro per-W artifact build-cost / caching probe (scratch, uncommitted).

Sibling of ``kokoro_split_probe.py``.  Measures:

* ``reuse``  — search ONCE at a canonical W, then ``lower`` /
  ``RegionFusionRunner``-lower at *other* Ws: is the extracted term
  shape-polymorphic (re-lower cheap) or W-baked (per-W search
  required)?  Reports per-stage wall times and delivered parity.
* ``persist``— can a built module be serialized?  Tries
  ``torch.export.export`` + ``torch.export.save`` round-trip on the
  plain-lowered and regionfused modules, plus a pickle attempt.

Usage::

    .venv/bin/python bench/drivers/probes/kokoro_artifact_probe.py reuse [iters]
    .venv/bin/python bench/drivers/probes/kokoro_artifact_probe.py persist [W]
"""

from __future__ import annotations

import importlib.util
import os
import pickle
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


ksp = _load("kokoro_split_probe_mod", "kokoro_split_probe.py")
kokoro = ksp.kokoro
SEED = ksp.SEED


def _maxdiff(a, b) -> float:
    return (a.float() - b.float()).abs().max().item()


def _prep(m, utt="bench"):
    """M1 once -> (d, t_en, pd, ref_s, S)."""
    ids = ksp.utterance_ids(m, utt)
    ref_s = kokoro._voice(ids.shape[1] - 1)
    m1e = ksp._M1Wrap(m).eval()
    with torch.no_grad():
        torch.manual_seed(SEED)
        d, t_en, pd = ksp._call(m1e, (ids, ref_s))
    S = int(pd.sum())
    return d, t_en, pd, ref_s, S


def _dec_inputs(m, d, t_en, pd, ref_s, W: int):
    """glue + eager F0Ntrain -> decoder inputs at width W."""
    en, asr, s_, s128_, _ = ksp.glue(d, t_en, pd, ref_s, W)
    with torch.no_grad():
        F0, N = m.predictor.F0Ntrain(en, s_)
    return (asr, F0, N, s128_)


def _eager_ref(m, inp):
    wrap = ksp._DecWrap(m).eval()
    torch.manual_seed(SEED)
    with torch.no_grad():
        return ksp._call(wrap, inp)


def _timeit(fn, warm=3, reps=10):
    return kokoro.timeit(fn, warm=warm, reps=reps)


def reuse(iters: int = 3, Ws=(128, 144, 160, 176)) -> None:
    from mlopt_orchestrator.optimize import Optimizer
    from kit_torch.backend import TorchBackend
    from kit_torch.runners import RegionFusionRunner

    m = kokoro.load("fp32", disable_complex=False)
    m.eval()
    packed = __import__("tts_kokoro_packed")
    packed.install(m)
    d, t_en, pd, ref_s, S = _prep(m)
    print(f"[reuse] S={S} search-Ws={Ws}", flush=True)

    W0 = Ws[0]
    inp0 = _dec_inputs(m, d, t_en, pd, ref_s, W0)
    wrap = ksp._DecWrap(m).eval()

    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    res = optz.search(
        wrap, inp0, max_iterations=iters, max_enodes=100_000
    )
    t_search = time.time() - t0
    print(
        f"[reuse] SEARCH W={W0} {t_search:.1f}s "
        f"enodes={getattr(res, 'n_enodes', '?')}",
        flush=True,
    )
    # free the e-graph; keep term/ir/param_values for lower
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None  # ty: ignore[invalid-assignment]
    res.source = None  # ty: ignore[invalid-assignment]
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    for W in Ws:
        inp = _dec_inputs(m, d, t_en, pd, ref_s, W)
        ref = _eager_ref(m, inp)

        # ---- plain lower (verify=False) ----
        t0 = time.time()
        try:
            low = optz.lower(res, inp, verify=False, runner=None)
            t_low = time.time() - t0
            mod = low.module
        except Exception as e:
            print(
                f"[reuse] W={W} LOWER FAILED {time.time() - t0:.1f}s "
                f"{type(e).__name__}: {str(e)[:160]}",
                flush=True,
            )
            continue
        try:
            torch.manual_seed(SEED)
            with torch.no_grad():
                out = ksp._call(mod, inp)
            if isinstance(out, tuple):
                dd = max(_maxdiff(a, b) for a, b in zip(out, ref, strict=True))
                shape_ok = all(
                    tuple(a.shape) == tuple(b.shape)
                    for a, b in zip(out, ref, strict=True)
                )
            else:
                dd = _maxdiff(out, ref)
                shape_ok = tuple(out.shape) == tuple(ref.shape)
            ms = _timeit(lambda: ksp._call(mod, inp))
            print(
                f"[reuse] W={W} lower {t_low:.2f}s shape_ok={shape_ok} "
                f"max|diff|={dd:.3e} delivered={ms:.1f}ms",
                flush=True,
            )
        except Exception as e:
            print(
                f"[reuse] W={W} lower {t_low:.2f}s CALL FAILED "
                f"{type(e).__name__}: {str(e)[:160]}",
                flush=True,
            )
        del low, mod
        gc.collect()
        torch.cuda.empty_cache()

        # ---- regionfuse lower ----
        ksp._restore_creators()
        t0 = time.time()
        try:
            low_f = optz.lower(
                res,
                inp,
                verify=False,
                runner=RegionFusionRunner(output_rtol=2e-3),
            )
            t_rf = time.time() - t0
            rf = low_f.stats.get("region_fuse", {})
            mod_f = low_f.module
            applied = bool(rf.get("applied"))
            try:
                torch.manual_seed(SEED)
                with torch.no_grad():
                    outf = ksp._call(mod_f, inp)
                if isinstance(outf, tuple):
                    ddf = max(
                        _maxdiff(a, b)
                        for a, b in zip(outf, ref, strict=True)
                    )
                    shape_okf = all(
                        tuple(a.shape) == tuple(b.shape)
                        for a, b in zip(outf, ref, strict=True)
                    )
                else:
                    ddf = _maxdiff(outf, ref)
                    shape_okf = tuple(outf.shape) == tuple(ref.shape)
                msf = _timeit(lambda: ksp._call(mod_f, inp))
                print(
                    f"[reuse] W={W} regionfuse {t_rf:.1f}s "
                    f"applied={applied} regions={rf.get('regions_fused')} "
                    f"ops={rf.get('ops_fused')} kernels={rf.get('distinct_kernels')} "
                    f"shape_ok={shape_okf} max|diff|={ddf:.3e} "
                    f"delivered+fuse={msf:.1f}ms",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[reuse] W={W} regionfuse {t_rf:.1f}s "
                    f"applied={applied} CALL FAILED "
                    f"{type(e).__name__}: {str(e)[:160]}",
                    flush=True,
                )
            del low_f, mod_f
        except Exception as e:
            print(
                f"[reuse] W={W} regionfuse FAILED {time.time() - t0:.1f}s "
                f"{type(e).__name__}: {str(e)[:160]}",
                flush=True,
            )
        gc.collect()
        torch.cuda.empty_cache()

    del m, wrap, res
    gc.collect()
    torch.cuda.empty_cache()


def persist(W: int = 128, iters: int = 3) -> None:
    from mlopt_orchestrator.optimize import Optimizer
    from kit_torch.backend import TorchBackend
    from kit_torch.runners import RegionFusionRunner

    m = kokoro.load("fp32", disable_complex=False)
    m.eval()
    packed = __import__("tts_kokoro_packed")
    packed.install(m)
    d, t_en, pd, ref_s, S = _prep(m)
    assert W >= S, f"W={W} < S={S}"
    inp = _dec_inputs(m, d, t_en, pd, ref_s, W)
    wrap = ksp._DecWrap(m).eval()

    optz = Optimizer(backend=TorchBackend())
    res = optz.search(wrap, inp, max_iterations=iters, max_enodes=100_000)
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None  # ty: ignore[invalid-assignment]
    res.source = None  # ty: ignore[invalid-assignment]
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # ---- plain lowered module ----
    low = optz.lower(res, inp, verify=False, runner=None)
    mod = low.module
    print(f"[persist] plain module type={type(mod).__name__}", flush=True)

    # pickle attempt
    t0 = time.time()
    try:
        blob = pickle.dumps(mod)
        print(f"[persist] pickle OK {len(blob)/1e6:.1f}MB {time.time()-t0:.1f}s", flush=True)
        mod2 = pickle.loads(blob)
        torch.manual_seed(SEED)
        with torch.no_grad():
            o1 = ksp._call(mod, inp)
            o2 = ksp._call(mod2, inp)
        dd = _maxdiff(o1, o2) if not isinstance(o1, tuple) else max(
            _maxdiff(a, b) for a, b in zip(o1, o2, strict=True)
        )
        print(f"[persist] pickle roundtrip max|diff|={dd:.3e}", flush=True)
    except Exception as e:
        print(f"[persist] pickle FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)

    # torch.export attempt
    try:
        t0 = time.time()
        ep = torch.export.export(mod, inp)
        print(f"[persist] torch.export OK {time.time()-t0:.1f}s", flush=True)
        out_ep = ep.module()(*inp)
        ref = _eager_ref(m, inp)
        dd = _maxdiff(out_ep, ref) if not isinstance(out_ep, tuple) else max(
            _maxdiff(a, b) for a, b in zip(out_ep, ref, strict=True)
        )
        print(f"[persist] exported-vs-eager max|diff|={dd:.3e}", flush=True)
        t0 = time.time()
        torch.export.save(ep, "/tmp/kokoro_dec_W128.pt2")
        print(f"[persist] export.save {time.time()-t0:.1f}s", flush=True)
        ep2 = torch.export.load("/tmp/kokoro_dec_W128.pt2")
        out2 = ep2.module()(*inp)
        dd2 = _maxdiff(out_ep, out2) if not isinstance(out_ep, tuple) else max(
            _maxdiff(a, b) for a, b in zip(out_ep, out2, strict=True)
        )
        print(f"[persist] pt2 roundtrip max|diff|={dd2:.3e}", flush=True)
        # does the .pt2 run at a different W?
        inp2 = _dec_inputs(m, d, t_en, pd, ref_s, W + 16)
        try:
            o = ep2.module()(*inp2)
            print(
                f"[persist] pt2 at W={W+16} ran, out shape "
                f"{tuple(o.shape) if hasattr(o,'shape') else [tuple(x.shape) for x in o]}",
                flush=True,
            )
        except Exception as e:
            print(f"[persist] pt2 at W={W+16} FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
    except Exception as e:
        print(f"[persist] torch.export FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)

    del low, mod
    gc.collect()
    torch.cuda.empty_cache()

    # ---- regionfused module ----
    ksp._restore_creators()
    low_f = optz.lower(
        res, inp, verify=False, runner=RegionFusionRunner(output_rtol=2e-3)
    )
    mod_f = low_f.module
    rf = low_f.stats.get("region_fuse", {})
    print(
        f"[persist] regionfuse applied={rf.get('applied')} "
        f"regions={rf.get('regions_fused')} kernels={rf.get('distinct_kernels')}",
        flush=True,
    )
    try:
        blob = pickle.dumps(mod_f)
        print(f"[persist] fused pickle OK {len(blob)/1e6:.1f}MB", flush=True)
    except Exception as e:
        print(f"[persist] fused pickle FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
    try:
        ep_f = torch.export.export(mod_f, inp)
        print("[persist] fused torch.export OK", flush=True)
        o = ep_f.module()(*inp)
        print(f"[persist] fused exported out shape {tuple(o.shape) if hasattr(o,'shape') else 'tuple'}", flush=True)
    except Exception as e:
        print(f"[persist] fused torch.export FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)

    del m, wrap, res, low_f, mod_f
    gc.collect()
    torch.cuda.empty_cache()


def dyn(iters: int = 3, Ws=(128, 144, 160)) -> None:
    """Force ``torch.export`` dynamic shapes -> is the searched term
    shape-polymorphic?  Patches ``torch.export.export`` (probe-local)
    to mark every non-batch input dim dynamic, searches once at W0,
    then lowers + regionfuses at other Ws.
    """
    from mlopt_orchestrator.optimize import Optimizer
    from kit_torch.backend import TorchBackend
    from kit_torch.runners import RegionFusionRunner

    m = kokoro.load("fp32", disable_complex=False)
    m.eval()
    packed = __import__("tts_kokoro_packed")
    packed.install(m)
    d, t_en, pd, ref_s, S = _prep(m)
    print(f"[dyn] S={S} Ws={Ws}", flush=True)

    W0 = Ws[0]
    inp0 = _dec_inputs(m, d, t_en, pd, ref_s, W0)
    inp1 = _dec_inputs(m, d, t_en, pd, ref_s, Ws[1])
    print(
        "[dyn] input shapes W0: "
        + ", ".join(f"{tuple(t.shape)}" for t in inp0),
        flush=True,
    )
    print(
        "[dyn] input shapes W1: "
        + ", ".join(f"{tuple(t.shape)}" for t in inp1),
        flush=True,
    )
    # dims that differ between W0/W1 are the polymorphic candidates;
    # KAP_DYN_AUTO=1 uses Dim.AUTO (dynamo decides per-use) instead of
    # forcing Dim — static-specialized uses then stay static honestly.
    auto = os.environ.get("KAP_DYN_AUTO") == "1"
    specs = []
    for i, (a, b) in enumerate(zip(inp0, inp1, strict=True)):
        specs.append(
            {
                dim: (
                    torch.export.Dim.AUTO
                    if auto
                    else torch.export.Dim(f"in{i}_d{dim}", min=0)
                )
                for dim in range(a.dim())
                if a.shape[dim] != b.shape[dim]
            }
        )
    print(f"[dyn] dynamic spec: {specs}", flush=True)
    wrap = ksp._DecWrap(m).eval()

    # pre-flight: bare export with the spec (cheap) before paying for search
    t0 = time.time()
    try:
        ep = torch.export.export(wrap, inp0, dynamic_shapes=tuple(specs))
        print(
            f"[dyn] preflight export OK {time.time() - t0:.1f}s",
            flush=True,
        )
        # does the exported program run at W1?
        try:
            o = ep.module()(*inp1)
            print(
                f"[dyn] preflight export ran at W={Ws[1]}, "
                f"out={tuple(o.shape) if hasattr(o, 'shape') else o}",
                flush=True,
            )
        except Exception as e:
            print(
                f"[dyn] preflight export at W={Ws[1]} FAILED "
                f"{type(e).__name__}: {str(e)[:200]}",
                flush=True,
            )
        del ep
        import gc as _gc

        _gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(
            f"[dyn] preflight export FAILED {time.time() - t0:.1f}s "
            f"{type(e).__name__}: {str(e)[:400]}",
            flush=True,
        )

    orig_export = torch.export.export

    def _patched_export(model, args, **kw):
        kw.setdefault("dynamic_shapes", tuple(specs))
        return orig_export(model, args, **kw)

    torch.export.export = _patched_export  # ty: ignore[invalid-assignment]
    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    try:
        res = optz.search(
            wrap, inp0, max_iterations=iters, max_enodes=100_000
        )
        t_search = time.time() - t0
        print(f"[dyn] SEARCH(dynamic) W={W0} {t_search:.1f}s", flush=True)
    except Exception as e:
        print(
            f"[dyn] SEARCH(dynamic) FAILED {time.time() - t0:.1f}s "
            f"{type(e).__name__}: {str(e)[:300]}",
            flush=True,
        )
        torch.export.export = orig_export  # ty: ignore[invalid-assignment]
        return
    finally:
        torch.export.export = orig_export  # ty: ignore[invalid-assignment]
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None  # ty: ignore[invalid-assignment]
    res.source = None  # ty: ignore[invalid-assignment]
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    for W in Ws:
        inp = _dec_inputs(m, d, t_en, pd, ref_s, W)
        ref = _eager_ref(m, inp)

        t0 = time.time()
        try:
            low = optz.lower(res, inp, verify=False, runner=None)
            t_low = time.time() - t0
            mod = low.module
        except Exception as e:
            print(
                f"[dyn] W={W} LOWER FAILED {time.time() - t0:.1f}s "
                f"{type(e).__name__}: {str(e)[:200]}",
                flush=True,
            )
            continue
        try:
            torch.manual_seed(SEED)
            with torch.no_grad():
                out = ksp._call(mod, inp)
            dd = _maxdiff(out, ref) if not isinstance(out, tuple) else max(
                _maxdiff(a, b) for a, b in zip(out, ref, strict=True)
            )
            shape_ok = (
                tuple(out.shape) == tuple(ref.shape)
                if not isinstance(out, tuple)
                else all(
                    tuple(a.shape) == tuple(b.shape)
                    for a, b in zip(out, ref, strict=True)
                )
            )
            ms = _timeit(lambda: ksp._call(mod, inp))
            print(
                f"[dyn] W={W} lower {t_low:.2f}s shape_ok={shape_ok} "
                f"max|diff|={dd:.3e} delivered={ms:.1f}ms",
                flush=True,
            )
        except Exception as e:
            print(
                f"[dyn] W={W} lower {t_low:.2f}s CALL FAILED "
                f"{type(e).__name__}: {str(e)[:200]}",
                flush=True,
            )
        del low, mod
        gc.collect()
        torch.cuda.empty_cache()

        # regionfuse with dynamic=True — one fused module should then
        # serve every W if the term really is polymorphic
        ksp._restore_creators()
        t0 = time.time()
        try:
            low_f = optz.lower(
                res,
                inp,
                verify=False,
                runner=RegionFusionRunner(
                    output_rtol=2e-3, dynamic=True
                ),
            )
            t_rf = time.time() - t0
            rf = low_f.stats.get("region_fuse", {})
            mod_f = low_f.module
            applied = bool(rf.get("applied"))
            try:
                torch.manual_seed(SEED)
                with torch.no_grad():
                    outf = ksp._call(mod_f, inp)
                ddf = (
                    _maxdiff(outf, ref)
                    if not isinstance(outf, tuple)
                    else max(
                        _maxdiff(a, b)
                        for a, b in zip(outf, ref, strict=True)
                    )
                )
                shape_okf = (
                    tuple(outf.shape) == tuple(ref.shape)
                    if not isinstance(outf, tuple)
                    else all(
                        tuple(a.shape) == tuple(b.shape)
                        for a, b in zip(outf, ref, strict=True)
                    )
                )
                msf = _timeit(lambda: ksp._call(mod_f, inp))
                print(
                    f"[dyn] W={W} regionfuse(dyn) {t_rf:.1f}s "
                    f"applied={applied} regions={rf.get('regions_fused')} "
                    f"ops={rf.get('ops_fused')} kernels={rf.get('distinct_kernels')} "
                    f"shape_ok={shape_okf} max|diff|={ddf:.3e} "
                    f"delivered+fuse={msf:.1f}ms",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[dyn] W={W} regionfuse(dyn) {t_rf:.1f}s "
                    f"applied={applied} CALL FAILED "
                    f"{type(e).__name__}: {str(e)[:200]}",
                    flush=True,
                )
            del low_f, mod_f
        except Exception as e:
            print(
                f"[dyn] W={W} regionfuse(dyn) FAILED "
                f"{time.time() - t0:.1f}s {type(e).__name__}: {str(e)[:200]}",
                flush=True,
            )
        gc.collect()
        torch.cuda.empty_cache()

    del m, wrap, res
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == "reuse":
        iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        Ws = (
            tuple(int(x) for x in sys.argv[3].split(","))
            if len(sys.argv) > 3
            else (128, 144, 160, 176)
        )
        reuse(iters, Ws)
    elif cmd == "persist":
        W = int(sys.argv[2]) if len(sys.argv) > 2 else 128
        iters = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        persist(W, iters)
    elif cmd == "dyn":
        iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        dyn(iters)
    else:
        raise SystemExit(f"unknown mode {cmd}")


if __name__ == "__main__":
    main()

"""Kokoro production-path benchmark — delivered+optimized vs direct HF.

Sibling of ``whisper_prod_bench.py``: the startup-facing "our full
delivered path vs the direct model" table for Kokoro-82M, completing
the STT picture (``whisper_production_bench.md``).

Per arm (the ``tts_kokoro_probe._harvest`` census):

* **eager** — the raw HF/kokoro module wrap, warm timed.
* **compile** — ``torch.compile(wrap, dynamic=False)`` reference.
* **delivered** — ``Optimizer.search`` → ``lower(verify=True)``
  (falls back to ``verify=False`` when the stochastic decoder's
  wall-A verify refuses — seeded parity is reported instead).
* **delivered+graph** — a second ``lower`` under
  ``CudaGraphRunner()`` (capture attempted honestly).
* **delivered+regionfuse** — a third ``lower`` under
  ``RegionFusionRunner(output_rtol=2e-3)`` — the shipped fusion
  runner; the stock ``output_rtol=1e-4`` gate is documented to
  decline the decoder arms (fused-vs-unfused max_rel ≈ 1.4e-3), so
  the bench uses the documented override and reports the ledger.

Parity: seeded ``torch.manual_seed(1234)`` pairs (the SineGen source
module draws ``rand``/``randn_like`` in-graph — unseeded outputs
differ run-to-run by ~6e-2, the measured stochastic floor), plus
the unseeded floor itself.  fp32 only — bf16 is walled by cuDNN
LSTM (``tts_kokoro_r1.md`` wall B).

Usage::

    .venv/bin/python bench/drivers/probes/kokoro_prod_bench.py \
        arm {f0n|decoder|decoder-nc|textenc-p|full-p} [iters]
    .venv/bin/python bench/drivers/probes/kokoro_prod_bench.py e2e [dtype]
"""

from __future__ import annotations

import importlib.util
import os
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


kokoro = _load("tts_kokoro_probe", "tts_kokoro_probe.py")

SEED = kokoro.SEED


def _call(mm, inp):
    return mm(*inp) if isinstance(inp, tuple) else mm(inp)


def _maxdiff(a, b) -> float:
    if isinstance(a, tuple):
        return max(
            _maxdiff(x, y) for x, y in zip(a, b, strict=True)
        )
    return (a.float() - b.float()).abs().max().item()


def _seeded_out(mm, inp):
    torch.manual_seed(SEED)
    with torch.no_grad():
        return _call(mm, inp)


def _restore_creators() -> None:
    """Undo the probe's device-creator monkeypatch before torch.compile.

    Same landmine ``kokoro_fusion_probe`` documents: the wrappers call
    ``ref.expand`` on a real CUDA tensor, which inductor's
    FakeTensorMode rejects — every in-process ``torch.compile`` dies
    until the originals are restored.  Search/lower are done by the
    time we get here.
    """
    for name, orig in kokoro._ORIG_CREATORS.items():
        setattr(torch, name, orig)
    torch._tts_probe_devcreators_patched = False  # ty: ignore[unresolved-attribute]


def arm(which: str, iters: int = 3) -> None:
    packed = which.endswith("-p")
    if packed:
        which = which[: -len("-p")]
    dtype = "fp32"
    tag = f"[{which}{'-p' if packed else ''}/{dtype} iters={iters}]"

    print(f"{tag} harvest ...", flush=True)
    t0 = time.time()
    m, wrap, inp, ref, ref_u1, ref_u2, cap = kokoro._harvest(
        which, dtype, packed=packed
    )
    print(f"{tag} harvest {time.time() - t0:.1f}s", flush=True)
    floor = _maxdiff(ref_u1, ref_u2)
    print(f"{tag} eager stochastic floor max|diff| {floor:.4e}", flush=True)

    e_ms = kokoro.timeit(lambda: _call(wrap, inp))
    print(f"{tag} EAGER wrap {e_ms:.2f} ms", flush=True)

    from mlopt_orchestrator.optimize import Optimizer
    from kit_torch.backend import TorchBackend

    optz = Optimizer(backend=TorchBackend())
    res = None
    try:
        t0 = time.time()
        res = optz.search(wrap, inp, max_iterations=iters, max_enodes=100_000)
        print(
            f"{tag} search {time.time() - t0:.1f}s "
            f"enodes={getattr(res, 'n_enodes', '?')}",
            flush=True,
        )
    except Exception as e:
        print(
            f"{tag} search FAILED {time.time() - t0:.1f}s "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return
    del m, cap
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None  # ty: ignore[invalid-assignment]
    res.source = None  # ty: ignore[invalid-assignment]
    import gc

    if os.environ.get("KPB_NOCOMPILE") == "1":
        # Lean arm: the eager wrap is dead weight (~decoder params are
        # shared with res.param_values, but the module + any private
        # buffers are not) — free it before the lowers on a shared card.
        wrap = None
    gc.collect()
    torch.cuda.empty_cache()

    # ---- delivered -----------------------------------------------------
    # ``verify=True`` lowers a SECOND reference module — on the shared
    # 4 GB card that OOMs alongside a co-tenant, and on the stochastic
    # decoder arms it can only refuse anyway (wall A: unseeded RNG
    # floor ~6e-2).  Deterministic arms (floor == 0) verify honestly;
    # stochastic arms skip straight to verify=False and rely on the
    # seeded-parity pair below.
    deterministic = floor == 0.0
    del_mod = None
    low = None
    try:
        if deterministic:
            t0 = time.time()
            low = optz.lower(res, inp, verify=True, runner=None)
            ok = low.verified.passed if low.verified else None
            print(
                f"{tag} lower+verify {time.time() - t0:.1f}s verified={ok}",
                flush=True,
            )
            del_mod = low.module
        else:
            print(
                f"{tag} lower(verify skipped — stochastic floor "
                f"{floor:.2e})", flush=True
            )
    except Exception as e:
        print(
            f"{tag} lower+verify FAILED {type(e).__name__}: {e}",
            flush=True,
        )
    if del_mod is None:
        try:
            t0 = time.time()
            low = optz.lower(res, inp, verify=False, runner=None)
            print(f"{tag} lower(verify=False) {time.time() - t0:.1f}s", flush=True)
            del_mod = low.module
        except Exception as e:
            print(
                f"{tag} lower FAILED {type(e).__name__}: {e}",
                flush=True,
            )
            return
    low = None  # LowerResult holds .module — release the extra ref

    d_ms = kokoro.timeit(lambda: _call(del_mod, inp))
    d_seed = _seeded_out(del_mod, inp)
    dd = _maxdiff(d_seed, ref)
    print(
        f"{tag} DELIVERED {d_ms:.2f} ms  vs-eager-seeded max|diff| {dd:.3e}",
        flush=True,
    )
    try:
        with torch.no_grad():
            du1 = _call(del_mod, inp)
            du2 = _call(del_mod, inp)
        print(
            f"{tag} delivered unseeded run-to-run {_maxdiff(du1, du2):.4e}",
            flush=True,
        )
    except Exception as e:
        print(f"{tag} delivered unseeded FAILED {e}", flush=True)

    # Free the delivered module before the graph/fuse lowers — the
    # seeded output ``d_seed`` is all the fused-vs-delivered compare
  # needs, and each lower duplicates the plan tape (~GBs on decoder).
    del del_mod
    del_mod = None
    import gc as _gc1

    _gc1.collect()
    torch.cuda.empty_cache()

    # ---- delivered + CUDA graph ---------------------------------------
    try:
        from kit_cuda import CudaGraphRunner

        t0 = time.time()
        low_g = optz.lower(res, inp, verify=False, runner=CudaGraphRunner())
        g_ms = None
        if low_g.stats.get("cuda_graph"):
            g_ms = kokoro.timeit(lambda: _call(low_g.module, inp))
        print(
            f"{tag} delivered+graph cuda_graph={low_g.stats.get('cuda_graph')} "
            f"fail={low_g.stats.get('cuda_graph_failure')} "
            f"{f'{g_ms:.2f} ms' if g_ms else 'n/a'} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )
        del low_g
    except Exception as e:
        print(f"{tag} graph lower FAILED {type(e).__name__}: {e}", flush=True)
    gc.collect()
    torch.cuda.empty_cache()

    # ---- torch.compile reference (needs unpatched creators) ------------
    # Skipped under KPB_NOCOMPILE=1 — the compiled wrap's inductor
    # workspace can be the straw that OOMs a shared 4 GB card; the
    # ``compile`` mode measures it in a lean process instead.
    if os.environ.get("KPB_NOCOMPILE") != "1":
        _restore_creators()
        try:
            cwrap = torch.compile(wrap, dynamic=False)
            _call(cwrap, inp)
            torch.cuda.synchronize()
            c_ms = kokoro.timeit(lambda: _call(cwrap, inp))
            print(f"{tag} COMPILE wrap {c_ms:.2f} ms", flush=True)
            del cwrap
        except Exception as e:
            print(
                f"{tag} COMPILE FAILED {type(e).__name__}: {str(e)[:200]}",
                flush=True,
            )
    wrap = None  # eager wrap no longer needed — free before fuse lower
    gc.collect()
    torch.cuda.empty_cache()

    # ---- delivered + RegionFusionRunner --------------------------------
    # Stock output_rtol=1e-4 declines the decoder arms (measured
    # fused-vs-unfused max_rel ~1.4e-3 — kokoro_region_fuse.md §4);
    # the bench uses the documented 2e-3 override and reports the
    # ledger either way.
    try:
        from kit_torch.runners import RegionFusionRunner

        t0 = time.time()
        low_f = optz.lower(
            res,
            inp,
            verify=False,
            runner=RegionFusionRunner(output_rtol=2e-3),
        )
        rf = low_f.stats.get("region_fuse", {})
        census = rf.get("census", {})
        print(
            f"{tag} regionfuse lower {time.time() - t0:.1f}s "
            f"applied={rf.get('applied')} reason={rf.get('reason')} "
            f"regions={rf.get('regions_fused')} ops={rf.get('ops_fused')} "
            f"kernels={rf.get('distinct_kernels')} census={census}",
            flush=True,
        )
        fv = rf.get("fused_vs_unfused") or {}
        if fv:
            print(
                f"{tag}   fused-vs-unfused max_abs={fv.get('max_abs'):.3e} "
                f"max_rel={fv.get('max_rel'):.3e} passed={fv.get('passed')}",
                flush=True,
            )
        if rf.get("applied"):
            f_ms = kokoro.timeit(lambda: _call(low_f.module, inp))
            f_seed = _seeded_out(low_f.module, inp)
            fd = _maxdiff(f_seed, ref)
            dd2 = _maxdiff(f_seed, d_seed)
            print(
                f"{tag} DELIVERED+FUSE {f_ms:.2f} ms  "
                f"vs-eager-seeded {fd:.3e}  vs-delivered-seeded {dd2:.3e}",
                flush=True,
            )
        del low_f
    except Exception as e:
        print(
            f"{tag} regionfuse FAILED {type(e).__name__}: {str(e)[:300]}",
            flush=True,
        )

    del res, wrap, del_mod, low
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{tag} done", flush=True)


def e2e(dtype: str = "fp32") -> None:
    """Full ``forward_with_tokens`` — eager stage breakdown + walls."""
    import torch.nn as nn

    m = kokoro.load(dtype, disable_complex=False)
    m.eval()
    ids = kokoro._input_ids()
    ref_s = kokoro._voice(ids.shape[1] - 1)

    # ---- per-stage timing: replicate forward_with_tokens with syncs ---
    names = [
        "bert",
        "bert_encoder",
        "predictor.text_encoder",
        "predictor.lstm",
        "duration_proj+align",
        "F0Ntrain",
        "text_encoder",
        "decoder",
    ]
    acc = {n: 0.0 for n in names}

    def run_once():
        nonlocal acc
        acc = {n: 0.0 for n in names}

        def tick(name, fn):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            acc[name] += (time.perf_counter() - t) * 1e3
            return out

        with torch.no_grad():
            T = ids.shape[1]
            input_lengths = torch.full(
                (ids.shape[0],), T, device=ids.device, dtype=torch.long
            )
            text_mask = (
                torch.arange(T, device=ids.device)
                .unsqueeze(0)
                .expand(ids.shape[0], -1)
                .type_as(input_lengths)
            )
            text_mask = torch.gt(
                text_mask + 1, input_lengths.unsqueeze(1)
            ).to(ids.device)
            bert_dur = tick(
                "bert",
                lambda: m.bert(
                    ids, attention_mask=(~text_mask).int()
                ),
            )
            d_en = tick(
                "bert_encoder",
                lambda: m.bert_encoder(bert_dur).transpose(-1, -2),
            )
            s = ref_s[:, 128:]
            d = tick(
                "predictor.text_encoder",
                lambda: m.predictor.text_encoder(
                    d_en, s, input_lengths, text_mask
                ),
            )
            x, _ = tick("predictor.lstm", lambda: m.predictor.lstm(d))
            duration = m.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1) / 1.0
            t = time.perf_counter()
            pred_dur = (
                torch.round(duration).clamp(min=1).long().squeeze()
            )
            indices = torch.repeat_interleave(
                torch.arange(ids.shape[1], device=m.device), pred_dur
            )
            pred_aln_trg = torch.zeros(
                (ids.shape[1], indices.shape[0]), device=m.device
            )
            pred_aln_trg[
                indices, torch.arange(indices.shape[0])
            ] = 1
            pred_aln_trg = pred_aln_trg.unsqueeze(0).to(m.device)
            en = d.transpose(-1, -2) @ pred_aln_trg
            torch.cuda.synchronize()
            acc["duration_proj+align"] += (
                time.perf_counter() - t
            ) * 1e3
            F0_pred, N_pred = tick(
                "F0Ntrain", lambda: m.predictor.F0Ntrain(en, s)
            )
            t_en = tick(
                "text_encoder",
                lambda: m.text_encoder(
                    ids, input_lengths, text_mask
                ),
            )
            asr = t_en @ pred_aln_trg
            audio = tick(
                "decoder",
                lambda: m.decoder(
                    asr, F0_pred, N_pred, ref_s[:, :128]
                ).squeeze(),
            )
        return audio, pred_dur

    # warm + timed
    audio, pred_dur = run_once()
    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        run_once()
    torch.cuda.synchronize()
    total = (time.perf_counter() - t0) / reps * 1e3

    print(
        f"[e2e/{dtype}] forward_with_tokens total {total:.1f} ms "
        f"audio {tuple(audio.shape)} pred_dur_sum={int(pred_dur.sum())}",
        flush=True,
    )
    for n in names:
        # ``acc`` was reset inside the last ``run_once`` — it already
        # holds ONE call's per-stage wall times; do NOT divide by reps.
        print(f"    {n:22s} {acc[n]:8.2f} ms", flush=True)

    # stochastic floor of the full model
    with torch.no_grad():
        a1, _ = m.forward_with_tokens(ids, ref_s, speed=1.0)
        a2, _ = m.forward_with_tokens(ids, ref_s, speed=1.0)
    print(
        f"[e2e/{dtype}] full-model unseeded run-to-run "
        f"{_maxdiff(a1, a2):.4e}",
        flush=True,
    )

    # whole-model torch.compile attempt (bounded — expect wall).
    # ``KModel.forward`` takes a phoneme STRING; the tensor entry
    # point is ``forward_with_tokens``.  Restore the probe's creator
    # monkeypatch first — dynamo cannot trace the wrappers (same
    # landmine kokoro_fusion_probe documents).
    _restore_creators()
    try:
        cfwt = torch.compile(m.forward_with_tokens, dynamic=False)
        torch.manual_seed(SEED)
        with torch.no_grad():
            cfwt(ids, ref_s, speed=1.0)
        torch.cuda.synchronize()
        c_ms = kokoro.timeit(
            lambda: cfwt(ids, ref_s, speed=1.0), warm=2, reps=5
        )
        print(
            f"[e2e/{dtype}] torch.compile(forward_with_tokens) "
            f"{c_ms:.1f} ms",
            flush=True,
        )
        # seeded parity vs eager (the model is stochastic — SineGen
        # draws rand/randn_like in-graph)
        torch.manual_seed(SEED)
        with torch.no_grad():
            eo, _ = m.forward_with_tokens(ids, ref_s, speed=1.0)
        torch.manual_seed(SEED)
        with torch.no_grad():
            co = cfwt(ids, ref_s, speed=1.0)
            co = co[0] if isinstance(co, tuple) else co
        print(
            f"[e2e/{dtype}] compile-vs-eager seeded max|diff| "
            f"{_maxdiff(co, eo):.3e}",
            flush=True,
        )
    except Exception as e:
        print(
            f"[e2e/{dtype}] torch.compile(forward_with_tokens) FAILED "
            f"{type(e).__name__}: {str(e)[:300]}",
            flush=True,
        )
    del m
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def compile_arm(which: str) -> None:
    """Lean torch.compile reference for one arm (separate process).

    Harvest + eager timing + ``torch.compile(wrap)`` only — no
    search/lower, so the inductor workspace gets the whole card.
    """
    packed = which.endswith("-p")
    if packed:
        which = which[: -len("-p")]
    tag = f"[compile {which}{'-p' if packed else ''}/fp32]"
    m, wrap, inp, ref, ref_u1, ref_u2, _cap = kokoro._harvest(
        which, "fp32", packed=packed
    )
    del m, _cap, ref_u1, ref_u2
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    e_ms = kokoro.timeit(lambda: _call(wrap, inp))
    print(f"{tag} EAGER wrap {e_ms:.2f} ms", flush=True)
    _restore_creators()
    try:
        cwrap = torch.compile(wrap, dynamic=False)
        _call(cwrap, inp)
        torch.cuda.synchronize()
        c_ms = kokoro.timeit(lambda: _call(cwrap, inp))
        print(f"{tag} COMPILE wrap {c_ms:.2f} ms", flush=True)
    except Exception as e:
        print(
            f"{tag} COMPILE FAILED {type(e).__name__}: {str(e)[:300]}",
            flush=True,
        )
    del wrap
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "arm":
        which = sys.argv[2] if len(sys.argv) > 2 else "f0n"
        iters = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        arm(which, iters)
    elif sys.argv[1] == "compile":
        compile_arm(sys.argv[2] if len(sys.argv) > 2 else "decoder")
    elif sys.argv[1] == "e2e":
        e2e(sys.argv[2] if len(sys.argv) > 2 else "fp32")
    else:
        raise SystemExit(f"unknown mode {sys.argv[1]}")


if __name__ == "__main__":
    main()

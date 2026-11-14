"""Decode-step sdpa-backend probe — measurement driver.

Follow-up to ``whisper_decode_loop.md`` / ``decode_step_fusion.md``.
The step profile showed the delivered ``sdpa`` binding dispatching to
the mem-efficient (fmha cutlass) backend, which microbenchmarks
~2.4-4.7x SLOWER than the math backend at the cross-attention shape
(q=(1,8,1,64), kv=(1,8,1500,64), fp32).  This probe measures the fix:

* ``delivered``      — stock lowering (sdpa -> efficient backend)
* ``sdpa-math``      — same module, ``_IR_TO_TORCH["sdpa"]`` rebound to
                       force ``SDPBackend.MATH`` (plan tape resolves
                       bindings per call, so the patch reaches the
                       already-built module — a probe-level stand-in
                       for a torch_bridge binding fix)
* ``sdpa-heur``      — math when ``k.shape[-2] >= 256`` else default
                       (self-attn kv<=41 keeps efficient, cross kv=1500
                       goes math)
* ``sdpa-cudnn``     — cuDNN attention backend for comparison
* each best arm also gets the zero-copy CUDA-graph treatment from
  ``whisper_decode_step_profile`` and a real greedy-loop run with an
  ids-match check.

Usage::

    .venv/bin/python bench/drivers/probes/whisper_decode_sdpa_backend.py [iters] [ntok] [L]
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "whisper_decode_loop", HERE / "whisper_decode_loop.py"
)
probe = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(probe)
_spec2 = importlib.util.spec_from_file_location(
    "whisper_decode_step_profile",
    HERE / "whisper_decode_step_profile.py",
)
profmod = importlib.util.module_from_spec(_spec2)
assert _spec2.loader is not None
_spec2.loader.exec_module(profmod)


def _sdpa_kwargs(q, k, v, a, kw):
    out = {"attn_mask": a[0]} if a else {}
    out.update(
        {
            (
                "attn_mask"
                if kk == "arg3"
                else "dropout_p"
                if kk == "arg4"
                else "is_causal"
                if kk == "arg5"
                else "scale"
                if kk == "arg6"
                else "enable_gqa"
                if kk == "arg7"
                else kk
            ): vv
            for kk, vv in kw.items()
            if vv is not None
        }
    )
    return out


def sdpa_with_backend(backend):
    def fn(q, k, v, *a, **kw):
        with sdpa_kernel([backend]):
            return F.scaled_dot_product_attention(
                q, k, v, **_sdpa_kwargs(q, k, v, a, kw)
            )

    return fn


def sdpa_heuristic(q, k, v, *a, **kw):
    # kv_len >= 256 -> math (mem-efficient fmha is slower there at fp32);
    # short kv keeps the default backend pick.
    be = SDPBackend.MATH if k.shape[-2] >= 256 else None
    if be is None:
        return F.scaled_dot_product_attention(
            q, k, v, **_sdpa_kwargs(q, k, v, a, kw)
        )
    with sdpa_kernel([be]):
        return F.scaled_dot_product_attention(
            q, k, v, **_sdpa_kwargs(q, k, v, a, kw)
        )


def timeit(fn, warm=3, reps=20):
    return profmod.timeit(fn, warm, reps)


def run_loop(step_fn, m, ehs, ntok, L, ck, cv, ref_ids):
    "Same bookkeeping as probe.run_delivered_loop; returns (warm ms/tok, ids)."
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
    steps = []
    gids = []
    with torch.no_grad():
        for t in range(ntok):
            hs = probe.embed_step(dec, ids, t)
            past = (*skb, *svb, *ck, *cv)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = step_fn(hs, probe.step_mask(t, L), *past)
            torch.cuda.synchronize()
            steps.append((time.perf_counter() - t0) * 1e3)
            logits = outs[0][:, -1]
            for i in range(probe.NLAYERS):
                if t < L:
                    skb[i][:, :, t : t + 1, :] = outs[1 + 2 * i]
                    svb[i][:, :, t : t + 1, :] = outs[2 + 2 * i]
            ids = logits.argmax(-1, keepdim=True)
            gids.append(ids[:, -1].item())
    agree = sum(
        1 for a, b in zip(gids, ref_ids, strict=False) if a == b
    )
    warm = sum(steps[1:])
    return warm / (ntok - 1), steps[-8:], agree, gids


def main() -> None:
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    L = int(sys.argv[3]) if len(sys.argv) > 3 else max(40, ntok)

    import mlopt_carriers  # noqa: F401
    import mlopt_torch  # noqa: F401
    from mlopt_orchestrator.optimize import Optimizer
    from mlopt_torch.backend import TorchBackend
    from mlopt_torch.torch_bridge import _IR_TO_TORCH

    HARVEST = min(8, ntok - 2)
    m = probe.load()
    ehs = probe.encoder_out(m)
    step_ms, ref = probe.greedy_loop_hf(
        m, ehs, ntok, harvest_at=HARVEST
    )
    warm_e = sum(step_ms[1:]) / (ntok - 1)
    print(f"[eager] warm {warm_e:.2f} ms/tok", flush=True)

    inputs, _meta = probe.build_inputs(ref, L)
    wrap = probe._DecodeStepWrap(m, ref["ehs"], m.config).eval()
    optz = Optimizer(backend=TorchBackend())
    res = optz.search(wrap, inputs, max_iterations=iters)
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None
    res.source = None
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    low = optz.lower(res, inputs, verify=True, runner=None)
    print(
        f"[lower] verified={low.verified.passed if low.verified else None}",
        flush=True,
    )
    mod = low.module

    # reference logits under stock binding for correctness checks
    with torch.no_grad():
        ref_logits = mod(*inputs)[0][:, -1].float().clone()

    orig_sdpa = _IR_TO_TORCH["sdpa"]
    arms = [
        ("stock", None),
        ("math", SDPBackend.MATH),
        ("cudnn", SDPBackend.CUDNN_ATTENTION),
        ("heur", "heur"),
    ]
    best = None
    for tag, be in arms:
        try:
            if be is None:
                _IR_TO_TORCH["sdpa"] = orig_sdpa
            elif be == "heur":
                _IR_TO_TORCH["sdpa"] = sdpa_heuristic
            else:
                _IR_TO_TORCH["sdpa"] = sdpa_with_backend(be)
            with torch.no_grad():
                out = mod(*inputs)
                d = (
                    (out[0][:, -1].float() - ref_logits)
                    .abs()
                    .max()
                    .item()
                )
            ms = timeit(lambda: mod(*inputs))
            print(
                f"[arm] {tag}: step {ms:.2f} ms  max|diff|={d:.2e}",
                flush=True,
            )
            if d < 1e-3 and (best is None or ms < best[1]):
                best = (tag, ms)
        except Exception as e:
            print(
                f"[arm] {tag}: FAILED {type(e).__name__}: {e}",
                flush=True,
            )
    _IR_TO_TORCH["sdpa"] = orig_sdpa
    print(f"[best] {best}", flush=True)

    # ---- loop-level for the best backend arm ----
    ck = list(ref["past"]["ck"])
    cv = list(ref["past"]["cv"])
    eager_ids = ref["gen_ids"]
    for tag, be in arms:
        if tag != (best[0] if best else None):
            continue
        if be is None:
            _IR_TO_TORCH["sdpa"] = orig_sdpa
        elif be == "heur":
            _IR_TO_TORCH["sdpa"] = sdpa_heuristic
        else:
            _IR_TO_TORCH["sdpa"] = sdpa_with_backend(be)
        try:
            w, last8, agree, _g = run_loop(
                lambda *a: mod(*a), m, ehs, ntok, L, ck, cv, eager_ids
            )
            print(
                f"[loop] {tag}: warm {w:.2f} ms/tok last8 {sum(last8) / len(last8):.2f} "
                f"ids-match {agree}/{ntok}",
                flush=True,
            )
        finally:
            _IR_TO_TORCH["sdpa"] = orig_sdpa

    # ---- zero-copy CUDA graph on top of the best backend ----
    if best is not None:
        tag, _ms = best
        be = dict(arms)[tag]
        if be is None:
            _IR_TO_TORCH["sdpa"] = orig_sdpa
        elif be == "heur":
            _IR_TO_TORCH["sdpa"] = sdpa_heuristic
        else:
            _IR_TO_TORCH["sdpa"] = sdpa_with_backend(be)
        try:
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
            hs0 = probe.embed_step(dec, ids, 0)
            m0 = probe.step_mask(0, L)
            zcap = profmod._ZeroCopyGraphCapture.capture(
                mod, (hs0, m0, *skb, *svb, *ck, *cv)
            )
            steps = []
            gids = []
            with torch.no_grad():
                for t in range(ntok):
                    hs = probe.embed_step(dec, ids, t)
                    past = (*skb, *svb, *ck, *cv)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    outs = zcap(hs, probe.step_mask(t, L), *past)
                    torch.cuda.synchronize()
                    steps.append((time.perf_counter() - t0) * 1e3)
                    logits = outs[0][:, -1]
                    for i in range(probe.NLAYERS):
                        if t < L:
                            skb[i][:, :, t : t + 1, :] = outs[1 + 2 * i]
                            svb[i][:, :, t : t + 1, :] = outs[2 + 2 * i]
                    ids = logits.argmax(-1, keepdim=True)
                    gids.append(ids[:, -1].item())
            agree = sum(
                1
                for a, b in zip(gids, eager_ids, strict=False)
                if a == b
            )
            warm = sum(steps[1:])
            print(
                f"[loop] {tag}+zerograph: warm {warm:.1f} ms "
                f"({warm / (ntok - 1):.2f} ms/tok) last8 {sum(steps[-8:]) / 8:.2f} "
                f"ids-match {agree}/{ntok}",
                flush=True,
            )
        except Exception as e:
            print(
                f"[loop] {tag}+zerograph FAILED {type(e).__name__}: {e}",
                flush=True,
            )
        finally:
            _IR_TO_TORCH["sdpa"] = orig_sdpa

    del m
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

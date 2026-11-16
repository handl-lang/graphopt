"""Whisper production-representative benchmark — measurement driver.

Compares, on real log-mel features from synthetic audio, per model
size (``base`` fp32 / ``small`` fp32 / ``turbo`` bf16):

* ``hf-generate``  — ``WhisperForConditionalGeneration.generate``
  end-to-end (encoder + HF greedy decode loop, real logits
  processors: suppress_tokens + begin_suppress_tokens).
* ``hf-compile``   — same call under ``torch.compile(model)``
  (separate mode; Inductor may fail — reported honestly).
* ``hf-loop``      — manual greedy decode loop through the real HF
  decoder (same prompt prefix + suppression as ``generate``), giving
  honest per-step ms/tok and ids for the decode component alone.
* ``wrap``         — the probe-local static-K/V ``_DecodeStepWrapStatic``
  (caller-owned buffers, in-place ``index_copy_`` slot writes, zero
  cats) run eagerly inside the same greedy loop.
* ``delivered``    — ``Optimizer.search`` → ``lower(verify=True)``
  module inside the same loop.
* ``delivered+graph`` — the delivered module under
  ``CudaGraphRunner(alias_inputs=i>=3, mutable_inputs=3<=i<3+2L)`` —
  THE production candidate.
* ``compile-wrap`` — ``torch.compile`` on the same static wrap.

Correctness: generated token ids must match the HF eager loop
exactly (same prompt prefix, same suppression, same argmax).  The
prompt prefix is extracted empirically from a 1-token ``generate``
call so our manual loop is seeded identically to HF's.

Usage (transformers lives in /tmp/whisper_deps — uv sync prunes
.venv installs)::

    PYTHONPATH=/tmp/whisper_deps .venv/bin/python \
        bench/drivers/probes/whisper_prod_bench.py eager base [ntok]
    PYTHONPATH=/tmp/whisper_deps .venv/bin/python \
        bench/drivers/probes/whisper_prod_bench.py opt small [iters] [ntok] [L]
    PYTHONPATH=/tmp/whisper_deps .venv/bin/python \
        bench/drivers/probes/whisper_prod_bench.py opt turbo bf16 [iters] [ntok] [L]
    PYTHONPATH=/tmp/whisper_deps .venv/bin/python \
        bench/drivers/probes/whisper_prod_bench.py compilegen base [ntok]
    PYTHONPATH=/tmp/whisper_deps .venv/bin/python \
        bench/drivers/probes/whisper_prod_bench.py all base
"""

from __future__ import annotations

import gc
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


probe = _load("whisper_decode_loop", "whisper_decode_loop.py")
kv = _load("whisper_kv_static_probe", "whisper_kv_static_probe.py")

MODELS = {
    "base": ("openai/whisper-base", torch.float32, 80),
    "small": ("openai/whisper-small", torch.float32, 80),
    "turbo": ("openai/whisper-large-v3-turbo", torch.bfloat16, 128),
}
SR = 16000
AUDIO_S = 8.0
REPS = 3


def load(model: str):
    from transformers import WhisperForConditionalGeneration

    mid, dt, n_mels = MODELS[model]
    m = WhisperForConditionalGeneration.from_pretrained(
        mid, torch_dtype=dt, attn_implementation="eager"
    )
    return m.to("cuda").eval()  # ty: ignore[invalid-argument-type]


def make_features(model: str):
    """Real log-mel features from synthetic 8 s audio (sines+AM noise).

    ``WhisperFeatureExtractor`` zero-pads to the 30 s window exactly
    like the production pipeline; the encoder still sees 1500
    positions.  Constructed directly (defaults = whisper's) so no
    preprocessor_config download is needed.
    """
    import numpy as np
    from transformers import WhisperFeatureExtractor

    _mid, _dt, n_mels = MODELS[model]
    fe = WhisperFeatureExtractor(
        feature_size=n_mels,
        sampling_rate=SR,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    )
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * AUDIO_S)) / SR
    audio = (
        0.30 * np.sin(2 * np.pi * 220 * t)
        + 0.20 * np.sin(2 * np.pi * 440 * t + 0.5)
        + 0.15 * np.sin(2 * np.pi * 880 * t)
        * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
        + 0.05 * rng.standard_normal(t.shape)
    ).astype(np.float32)
    out = fe(audio, sampling_rate=SR, return_tensors="pt")
    return out["input_features"].to("cuda")


def encoder_out(m, feats):
    dt = next(m.parameters()).dtype
    feats = feats.to(dt)
    with torch.no_grad():
        return m.model.encoder(feats).last_hidden_state


def configure(model: str, m):
    """Point the reused probe modules at this model's decoder shape.

    ``kv`` loaded its own private copy of ``whisper_decode_loop`` as
    ``kv.probe`` — all three module objects must agree on the shape
    globals or ``build_inputs_static``/``pad_self`` mis-size tensors.
    """
    dec = m.model.decoder
    layer0 = dec.layers[0]
    nh = layer0.self_attn.num_heads
    hdim = layer0.self_attn.head_dim
    n_layers = len(dec.layers)
    for mod in (probe, kv, kv.probe):
        mod.NLAYERS = n_layers  # type: ignore[attr-defined]
        mod.NH = nh  # type: ignore[attr-defined]
        mod.HDIM = hdim  # type: ignore[attr-defined]
    # transformers 4.57 CacheLayerMixin adds abstract
    # ``get_max_cache_shape`` (absent in the 5.x the kv probe was
    # written against) — subclass-patch rather than edit the shared
    # probe file.
    class _KVWriteLayer457(kv._KVWriteLayer):  # type: ignore[attr-defined]
        def get_max_cache_shape(self):
            k = getattr(self, "keys", None)
            return int(k.shape[-2]) if k is not None else 0

    class _KVRefLayer457(kv._KVRefLayer):  # type: ignore[attr-defined]
        def get_max_cache_shape(self):
            k = getattr(self, "keys", None)
            return int(k.shape[-2]) if k is not None else 0

    kv._KVWriteLayer = _KVWriteLayer457  # type: ignore[attr-defined]
    kv._KVRefLayer = _KVRefLayer457  # type: ignore[attr-defined]
    return n_layers, nh, hdim


class _DecodeStepWrap457(kv._DecodeStepWrapStatic):  # type: ignore[attr-defined]
    """``kv._DecodeStepWrapStatic`` adapted to transformers 4.57:

    * ``WhisperDecoderLayer.forward`` returns ``outputs = (hidden,)``
      tuple in 4.57 (bare tensor in 5.x) — unpack defensively.
    * layer count taken from ``self.layers`` (no module-global
      ``NLAYERS``).
    """

    def forward(self, hs, mask, pos, *past):
        from transformers.cache_utils import (
            DynamicCache,
            EncoderDecoderCache,
        )

        n = len(self.layers)
        sk = past[0:n]
        sv = past[n : 2 * n]
        ck = past[2 * n : 3 * n]
        cv = past[3 * n : 4 * n]
        sc = DynamicCache()
        cc = DynamicCache()
        sc.layers = [kv._KVWriteLayer(pos) for _ in range(n)]  # type: ignore[attr-defined]
        cc.layers = [kv._KVRefLayer() for _ in range(n)]  # type: ignore[attr-defined]
        cache = EncoderDecoderCache(sc, cc)
        for i in range(n):
            sc.update(sk[i], sv[i], i)
            cc.update(ck[i], cv[i], i)
            cache.is_updated[i] = True
        h = hs
        for layer in self.layers:
            out = layer(
                h,
                mask,
                self.ehs,
                None,
                past_key_values=cache,
                use_cache=True,
            )
            h = out[0] if isinstance(out, tuple) else out
        h = self.norm(h)
        logits = self.head(h)
        outs = [logits]
        for i in range(n):
            outs.append(sc.layers[i].last_k)
            outs.append(sc.layers[i].last_v)
        return tuple(outs)


def suppress_sets(m):
    gc_ = m.generation_config
    supp = list(getattr(gc_, "suppress_tokens", None) or [])
    begin = list(getattr(gc_, "begin_suppress_tokens", None) or [])
    eos = getattr(gc_, "eos_token_id", None)
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    return supp, begin, eos


def apply_suppress(logits, supp, begin, first, eos=None):
    """HF greedy policy: suppress_tokens every step, begin_suppress at
    the first generated token only, plus ``eos`` banned for the whole
    fixed horizon — equivalent to HF's MinLengthLogitsProcessor under
    ``min_new_tokens=ntok`` (the benchmark runs a fixed decode
    horizon, so EOS never ends the loop early).  Non-mutating."""
    logits = logits.clone()
    idx = torch.tensor(
        supp + (begin if first else []) + (eos or []),
        device=logits.device,
    )
    if idx.numel():
        logits[:, idx] = float("-inf")
    return logits.argmax(-1, keepdim=True)


def embed_step(dec, ids, t):
    return probe.embed_step(dec, ids, t)


def hf_greedy_loop(m, ehs, prefix, ntok, harvest_call=None):
    """HF decoder greedy loop with prompt prefix + HF suppression.

    Feeds ``prefix[:-1]`` as untimed prefill, then ``ntok`` timed
    decode steps (first feeds ``prefix[-1]``).  Returns
    ``(gen_ids, step_ms, harvest)`` where harvest (if requested) is
    the dict from a mid-decode call: ``{args cache snapshot}`` built
    like ``probe.greedy_loop_hf``.
    """
    dec = m.model.decoder
    supp, begin, eos = suppress_sets(m)
    K = len(prefix)
    cache = None
    harvest = {}
    call_idx = 0
    gen_ids = []
    step_ms = []

    def snap_harvest(c, P):
        n_layers = len(c.self_attention_cache.layers)
        sc = c.self_attention_cache
        cc = c.cross_attention_cache
        harvest["past"] = {
            "sk": [
                sc.layers[j].keys[..., :P, :].detach().clone()
                for j in range(n_layers)
            ],
            "sv": [
                sc.layers[j].values[..., :P, :].detach().clone()
                for j in range(n_layers)
            ],
            "ck": [
                cc.layers[j].keys.detach().clone() for j in range(n_layers)
            ],
            "cv": [
                cc.layers[j].values.detach().clone()
                for j in range(n_layers)
            ],
        }
        harvest["P"] = P
        harvest["ehs"] = ehs
        harvest["hs"] = embed_step(dec, cur, P)
        harvest["logits"] = logits.float().clone()
        harvest["next_id"] = nxt.clone()

    with torch.no_grad():
        # untimed prefill: prefix[:-1]
        for j in range(K - 1):
            ids = torch.tensor(prefix[j : j + 1], device="cuda").view(1, 1)
            out = dec(
                input_ids=ids,
                encoder_hidden_states=ehs,
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            call_idx += 1
        cur = torch.tensor(prefix[K - 1 : K], device="cuda").view(1, 1)
        for i in range(ntok):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = dec(
                input_ids=cur,
                encoder_hidden_states=ehs,
                past_key_values=cache,
                use_cache=True,
            )
            logits = m.proj_out(out.last_hidden_state)[:, -1]
            nxt = apply_suppress(logits, supp, begin, first=(i == 0), eos=eos)
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - t0) * 1e3)
            if harvest_call is not None and call_idx == harvest_call:
                # P = self-attn cache length BEFORE this call = K-1+i
                snap_harvest(out.past_key_values, K - 1 + i)
            cache = out.past_key_values
            gen_ids.append(int(nxt[0, 0].item()))
            cur = nxt.view(1, 1)
            call_idx += 1
    return gen_ids, step_ms, harvest


def get_prefix(m, feats):
    """Decoder prompt prefix = the ``input_ids`` HF feeds the decoder
    on the first generate step (init tokens: startoftranscript, lang,
    task, no_timestamps, ...).  ``generate`` returns generated tokens
    only, so capture via a one-shot forward pre-hook."""
    box = {}
    feats = feats.to(next(m.parameters()).dtype)

    def hook(mod, args, kwargs):
        if "prefix" not in box:
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is not None:
                box["prefix"] = ids[0].detach().cpu().tolist()

    h = m.model.decoder.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            m.generate(
                feats,
                max_new_tokens=1,
                min_new_tokens=1,
                do_sample=False,
                language="en",
                task="transcribe",
                return_timestamps=False,
            )
    finally:
        h.remove()
    return box["prefix"]


def gen_e2e(m, feats, ntok, reps=REPS, force_full=False):
    """HF ``generate`` end-to-end latency + ids (encoder included).

    ``force_full`` sets ``min_new_tokens=ntok`` so the decode loop is
    exercised for the whole horizon (natural runs may stop at EOS —
    realistic, but decode-light on degenerate input).
    """
    feats = feats.to(next(m.parameters()).dtype)
    times = []
    ids = None
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        kw = {"min_new_tokens": ntok} if force_full else {}
        with torch.no_grad():
            seq = m.generate(
                feats,
                max_new_tokens=ntok,
                do_sample=False,
                language="en",
                task="transcribe",
                return_timestamps=False,
                **kw,
            )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        if isinstance(seq, dict):
            seq = seq["sequences"]
        ids = seq[0].tolist() if ids is None else ids
    return times, ids


def static_loop(step_fn, m, ehs, prefix, ntok, L, dtype, ckcv):
    """Greedy loop for the static wrap — caller-owned buffers, module
    writes slot ``pos`` itself.  ``step_fn(hs, mask, pos, *past)``.
    Prefill feeds ``prefix[:-1]`` untimed, then ``ntok`` timed steps.
    ``ckcv`` = (ck_list, cv_list) constant cross-attn K/V.
    """
    dec = m.model.decoder
    n_layers = len(dec.layers)
    nh = dec.layers[0].self_attn.num_heads
    hdim = dec.layers[0].self_attn.head_dim
    supp, begin, eos = suppress_sets(m)
    K = len(prefix)
    dtype = dtype
    skb = [
        torch.zeros(1, nh, L, hdim, device="cuda", dtype=dtype)
        for _ in range(n_layers)
    ]
    svb = [
        torch.zeros(1, nh, L, hdim, device="cuda", dtype=dtype)
        for _ in range(n_layers)
    ]
    ck, cv = ckcv
    pos = torch.empty(1, dtype=torch.long, device="cuda")
    gen_ids = []
    step_ms = []
    past = (*skb, *svb, *ck, *cv)
    with torch.no_grad():
        for j in range(K - 1):
            ids = torch.tensor(prefix[j : j + 1], device="cuda").view(1, 1)
            hs = embed_step(dec, ids, j)
            mask = torch.zeros(1, 1, 1, L, device="cuda", dtype=dtype)
            mask[..., j + 1 : L] = torch.finfo(dtype).min
            pos[0] = j
            step_fn(hs, mask, pos, *past)
        cur_dev = torch.tensor(prefix[K - 1 : K], device="cuda").view(1, 1)
        for i in range(ntok):
            p = K - 1 + i
            hs = embed_step(dec, cur_dev, p)
            mask = torch.zeros(1, 1, 1, L, device="cuda", dtype=dtype)
            mask[..., p + 1 : L] = torch.finfo(dtype).min
            pos[0] = p
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = step_fn(hs, mask, pos, *past)
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - t0) * 1e3)
            logits = outs[0][:, -1].float()
            nxt = apply_suppress(logits, supp, begin, first=(i == 0), eos=eos)
            gen_ids.append(int(nxt[0, 0].item()))
            cur_dev = nxt.view(1, 1)
    return gen_ids, step_ms


def peak_mem():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30


def reset_peak():
    torch.cuda.reset_peak_memory_stats()


def time_loop(fn, m, ehs, prefix, ntok, L, dtype, ckcv, reps=REPS):
    """Run static_loop ``reps`` times; return (ms/tok list-of-means,
    per-step lists, ids, peak GiB)."""
    all_steps = []
    ids = None
    for _ in range(reps):
        reset_peak()
        g, s = static_loop(fn, m, ehs, prefix, ntok, L, dtype, ckcv)
        all_steps.append(s)
        ids = g if ids is None else ids
    per_tok = [sum(s[1:]) / max(1, len(s) - 1) for s in all_steps]
    # warm ms/tok excludes first (capture/alloc) step
    return per_tok, all_steps, ids, peak_mem()


def arm_eager(model: str, ntok: int) -> None:
    m = load(model)
    configure(model, m)
    feats = make_features(model)
    with torch.no_grad():
        ehs = encoder_out(m, feats)
    enc_ms = None
    # encoder timing (one forward, warm)
    for _ in range(3):
        encoder_out(m, feats)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    encoder_out(m, feats)
    torch.cuda.synchronize()
    enc_ms = (time.perf_counter() - t0) * 1e3

    prefix = get_prefix(m, feats)
    print(f"[{model}] prefix={prefix} (K={len(prefix)})", flush=True)

    # HF manual loop — decode component, 3 reps (fresh cache each)
    rep_ms = []
    ids_ref = None
    harvest = {}
    for r in range(REPS):
        g, s, h = hf_greedy_loop(m, ehs, prefix, ntok)
        rep_ms.append(sum(s[1:]) / max(1, len(s) - 1))
        ids_ref = g if ids_ref is None else ids_ref
        if r == 0:
            harvest = h
    print(
        f"[{model}] hf-loop decode {ntok} tok: "
        f"{['%.2f' % v for v in rep_ms]} ms/tok(ids-match-self={ids_ref == g})",
        flush=True,
    )

    # HF generate e2e — natural stop AND forced-full-horizon
    gt, gids = gen_e2e(m, feats, ntok)
    print(
        f"[{model}] generate e2e {ntok} tok: "
        f"{['%.1f' % v for v in gt]} ms  enc~{enc_ms:.1f} ms "
        f"(natural stop, {len(gids)} ids)",
        flush=True,
    )
    gt2, gids2 = gen_e2e(m, feats, ntok, force_full=True)
    print(
        f"[{model}] generate e2e {ntok} tok min_new={ntok}: "
        f"{['%.1f' % v for v in gt2]} ms ({len(gids2)} ids)",
        flush=True,
    )
    gids = gids2 if len(gids2) >= ntok else gids
    print(f"[{model}] gen-ids={gids}", flush=True)
    print(f"[{model}] loop-ids={ids_ref}", flush=True)
    # generate() returns generated tokens only — compare directly
    # against the manual loop's ids (same greedy+suppression policy).
    gen_part = gids[:ntok]
    n_match = sum(
        1 for a, b in zip(gen_part, ids_ref) if a == b
    )
    first_div = next(
        (i for i, (a, b) in enumerate(zip(gen_part, ids_ref)) if a != b),
        None,
    )
    print(
        f"[{model}] generate-vs-loop id match "
        f"{n_match}/{min(len(gen_part), len(ids_ref))} "
        f"first-div={first_div}",
        flush=True,
    )

    # stash harvest for opt mode via file?  opt mode re-runs; keep
    # arms separate — eager mode doesn't need harvest.
    del m
    gc.collect()
    torch.cuda.empty_cache()


def arm_opt(model: str, iters: int, ntok: int, L: int) -> None:
    import kit_carriers  # noqa: F401
    import mlopt_torch  # noqa: F401
    from mlopt_cuda import CudaGraphRunner
    from mlopt_orchestrator.optimize import Optimizer
    from mlopt_torch.backend import TorchBackend

    m = load(model)
    n_layers, nh, hdim = configure(model, m)
    dtype = next(m.parameters()).dtype
    feats = make_features(model)
    with torch.no_grad():
        ehs = encoder_out(m, feats)
    prefix = get_prefix(m, feats)
    K = len(prefix)
    print(f"[{model}] prefix={prefix} K={K} dtype={dtype}", flush=True)

    # The encoder is dead weight for the rest of this arm (decode
    # benchmark): ehs is computed, prefix captured.  Dropping it
    # frees ~1.4 GB on turbo — the difference between search fitting
    # and OOMing on a 4 GB card.  ``m.model.decoder`` + ``proj_out``
    # stay for the HF reference loop and embed bookkeeping.
    del m.model.encoder
    gc.collect()
    torch.cuda.empty_cache()

    # harvest a mid-decode step: call index = (K-1 prefill) + g
    g_harvest = 8
    harvest_call = (K - 1) + g_harvest
    gen_ids, step_ms, harvest = hf_greedy_loop(
        m, ehs, prefix, ntok, harvest_call=harvest_call
    )
    P = harvest["P"]
    print(
        f"[{model}] hf-loop warm {sum(step_ms[1:])/(ntok-1):.2f} ms/tok "
        f"P={P}",
        flush=True,
    )

    # build static inputs; mask dtype must match model dtype
    inputs_s, meta = kv.build_inputs_static(harvest, L)
    inputs_s = list(inputs_s)
    inputs_s[1] = inputs_s[1].to(dtype)
    inputs_s = tuple(inputs_s)
    wrap_s = _DecodeStepWrap457(m, harvest["ehs"], m.config).eval()
    supp, begin, eos = suppress_sets(m)
    with torch.no_grad():
        o = wrap_s(*inputs_s)
        d = (o[0][:, -1].float() - harvest["logits"]).abs().max().item()
        # harvest step is decode index g_harvest (>0) → first=False
        nxt_wrap = apply_suppress(
            o[0][:, -1].float(), supp, begin, first=False, eos=eos
        )
        same = bool(
            (nxt_wrap.reshape(-1) == harvest["next_id"].reshape(-1)).all().item()
        )
    print(
        f"[{model}] wrap-static sanity P={P} L={L}: "
        f"logits max|diff|={d:.2e} argmax_equal={same}",
        flush=True,
    )
    if not same:
        print(f"[{model}] SANITY FAIL — refusing to optimize", flush=True)
        return

    # cross-attn constants for the loop (constant after step 0)
    ck = list(harvest["past"]["ck"])
    cv = list(harvest["past"]["cv"])

    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    res = optz.search(wrap_s, inputs_s, max_iterations=iters)
    print(
        f"[{model}] search {time.time()-t0:.1f}s "
        f"enodes={getattr(res, 'n_enodes', '?')}",
        flush=True,
    )
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None
    res.source = None
    gc.collect()
    torch.cuda.empty_cache()

    low = None
    try:
        low = optz.lower(res, inputs_s, verify=True, runner=None)
        print(
            f"[{model}] lower+verify "
            f"verified={low.verified.passed if low.verified else None}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"[{model}] lower+verify FAILED {type(e).__name__}: {e}",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
    if low is None:
        res.param_values = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in res.param_values.items()
        }
        gc.collect()
        torch.cuda.empty_cache()
        try:
            low = optz.lower(res, inputs_s, verify=False, runner=None)
            print(f"[{model}] lower(verify=False) ok", flush=True)
        except Exception as e:  # noqa: BLE001
            print(
                f"[{model}] lower(verify=False) FAILED "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            return
    mod = low.module
    ckcv = (ck, cv)

    # ---- loop arms ----
    arms = [("static-wrap", wrap_s), ("delivered", mod)]

    # delivered+graph (alias+mutable) — production candidate
    dmod = None
    try:
        dmod = CudaGraphRunner(
            multi_device=True,
            alias_inputs=lambda i, _t: i >= 3,
            mutable_inputs=lambda i, _t: 3 <= i < 3 + 2 * n_layers,
        ).apply(mod, inputs_s, {})
        arms.append(("delivered+graph", dmod))
    except Exception as e:  # noqa: BLE001
        print(f"[{model}] graph arm FAILED {type(e).__name__}: {e}", flush=True)

    # torch.compile on the same wrap
    cmod = None
    try:
        cmod = torch.compile(wrap_s, dynamic=False)
        with torch.no_grad():
            cmod(*inputs_s)
            torch.cuda.synchronize()
        arms.append(("compile-wrap", cmod))
    except Exception as e:  # noqa: BLE001
        print(f"[{model}] compile-wrap FAILED {type(e).__name__}: {e}", flush=True)

    ref_ids = gen_ids
    for tag, fn in arms:
        try:
            per_tok, all_steps, gids, pk = time_loop(
                fn, m, ehs, prefix, ntok, L, dtype, ckcv
            )
            agree = sum(
                1 for a, b in zip(gids, ref_ids) if a == b
            )
            print(
                f"[{model}] loop {tag}: ms/tok "
                f"{['%.2f' % v for v in per_tok]} "
                f"ids-match {agree}/{len(ref_ids)} peak={pk:.2f}GiB",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[{model}] loop {tag} FAILED {type(e).__name__}: {e}",
                flush=True,
            )

    # HF eager loop reference, timed fresh (3 reps already above —
    # re-time here for same-process A/B)
    rep = []
    for _ in range(REPS):
        g, s, _h = hf_greedy_loop(m, ehs, prefix, ntok)
        rep.append(sum(s[1:]) / max(1, len(s) - 1))
    print(
        f"[{model}] loop hf-eager: ms/tok "
        f"{['%.2f' % v for v in rep]}",
        flush=True,
    )

    del m
    gc.collect()
    torch.cuda.empty_cache()


def arm_compilegen(model: str, ntok: int) -> None:
    m = load(model)
    feats = make_features(model).to(next(m.parameters()).dtype)
    cm = torch.compile(m)
    # warm/compile on one generate call (untimed compile cost noted)
    t0 = time.time()
    with torch.no_grad():
        seq = cm.generate(
            feats,
            max_new_tokens=ntok,
            min_new_tokens=ntok,
            do_sample=False,
            language="en",
            task="transcribe",
            return_timestamps=False,
        )
    torch.cuda.synchronize()
    print(
        f"[{model}] compile+first-generate {time.time()-t0:.1f}s",
        flush=True,
    )
    times = []
    for _ in range(REPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            cm.generate(
                feats,
                max_new_tokens=ntok,
                min_new_tokens=ntok,
                do_sample=False,
                language="en",
                task="transcribe",
                return_timestamps=False,
            )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    print(
        f"[{model}] compiled-generate e2e {ntok} tok: "
        f"{['%.1f' % v for v in times]} ms",
        flush=True,
    )
    del m, cm
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "eager"
    model = sys.argv[2] if len(sys.argv) > 2 else "base"
    if model not in MODELS:
        raise SystemExit(f"unknown model {model}; choose {list(MODELS)}")
    if mode == "eager":
        arm_eager(model, int(sys.argv[3]) if len(sys.argv) > 3 else 64)
    elif mode == "opt":
        iters = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        ntok = int(sys.argv[4]) if len(sys.argv) > 4 else 64
        L = int(sys.argv[5]) if len(sys.argv) > 5 else max(80, ntok + 16)
        arm_opt(model, iters, ntok, L)
    elif mode == "compilegen":
        arm_compilegen(model, int(sys.argv[3]) if len(sys.argv) > 3 else 64)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()

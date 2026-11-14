"""Whisper decode/generation-loop probe — measurement driver.

Round-3 follow-up to ``whisper_probe.py`` / ``decoder_law_coverage.md``:
the per-layer spelled-attention→sdpa folds are verified, but the
autoregressive decode loop (token-by-token with KV cache) had never
been measured end-to-end.  This probe measures it honestly.

What is and isn't possible through the pipeline:

* ``model.generate`` / the HF decode loop is *control flow + mutable
  cache state* — not a static tensor function, so it cannot be handed
  to ``Optimizer.search`` directly (same refusal class as the r2
  decoder e-graph wall, one level up).
* What CAN be optimized is one decode *step* as a pure function.  The
  KV-cache append is ``cat(past, new)``; for whisper (learned absolute
  positions, no RoPE) attention is a permutation-invariant weighted
  sum over the valid key set, so a **fixed-capacity padded buffer**
  (``L`` slots, mask-gated) is mathematically exact: every step calls
  the same static-shape module, the caller writes each step's new K/V
  into the buffer eagerly, and the mask (a module *input*) blocks the
  pad slots.  Verified numerically against the real HF step before any
  timing is reported.

Arms (all on ``whisper-base`` fp32 CUDA by default):

* ``eager``  — encoder once + N-step greedy loop through the real HF
  decoder with a persistent ``EncoderDecoderCache``; per-step and
  total decode-loop timings.
* ``loop``   — harvest a mid-decode step's inputs, wrap the 6 decoder
  layers + final norm + ``proj_out`` as one static-shape module
  (``_DecodeStepWrap``), run ``Optimizer.search`` → ``lower``
  (verify=True, then ``CudaGraphRunner``), then run the *same* N-step
  greedy loop with the delivered module replacing the HF decoder.
  Also times ``torch.compile`` on the same wrap.

Usage::

    .venv/bin/python bench/drivers/probes/whisper_decode_loop.py eager [ntok]
    .venv/bin/python bench/drivers/probes/whisper_decode_loop.py loop [iters] [ntok] [L] [cpuparams]
"""

from __future__ import annotations

import copy
import gc
import sys
import time

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers import WhisperForConditionalGeneration

MODEL = "openai/whisper-base"
N_MELS, N_FRAMES = 80, 3000
NLAYERS = 6
NH, HDIM = 8, 64  # whisper-base decoder: 8 heads x 64


def load():
    m = WhisperForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    )
    return m.to("cuda").eval()  # ty: ignore[invalid-argument-type]


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


def encoder_out(m):
    x = torch.randn(1, N_MELS, N_FRAMES, device="cuda")
    with torch.no_grad():
        return m.model.encoder(x).last_hidden_state


def embed_step(dec, ids, t):
    """Replicate WhisperDecoder's embed+positions for one decode step."""
    e = dec.embed_tokens(ids)
    pos = dec.embed_positions(
        ids,
        past_key_values_length=t,
        position_ids=torch.arange(t, t + 1, device=ids.device)
        .unsqueeze(0)
        .expand(ids.shape[0], -1),
    )
    return e + pos


def greedy_loop_hf(m, ehs, ntok, harvest_at=None):
    """Real HF decode loop.  Returns (per-step ms list, harvest dict).

    ``harvest_at`` = call index whose layer-0 inputs + pre-call cache
    state are captured (past length == harvest_at).
    """
    dec = m.model.decoder
    cap = {}

    def hook(mod, args, kwargs):
        if cap.get("n", 0) == harvest_at:
            cap["args"] = args
            cap["kwargs"] = kwargs
        cap["n"] = cap.get("n", 0) + 1

    h = None
    if harvest_at is not None:
        h = dec.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    ids = torch.tensor(
        [[m.generation_config.decoder_start_token_id or 50257]],
        device="cuda",
    )
    cache = None
    step_ms = []
    gen_ids = []
    ref = {}
    with torch.no_grad():
        for t in range(ntok):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = dec(
                input_ids=ids,
                encoder_hidden_states=ehs,
                past_key_values=cache,
                use_cache=True,
            )
            logits = m.proj_out(out.last_hidden_state)
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - t0) * 1e3)
            if t == harvest_at:
                ref["logits"] = logits[:, -1].float().clone()
                ref["next_id"] = logits[:, -1].argmax(-1, keepdim=True).clone()
            cache = out.past_key_values
            ids = logits[:, -1].argmax(-1, keepdim=True)
            gen_ids.append(ids[:, -1].item())
    ref["gen_ids"] = gen_ids
    if h is not None:
        h.remove()
    if harvest_at is not None:
        # cache now holds ntok entries; the harvest step saw the first
        # ``harvest_at`` of them (call t has past length t).
        sc, cc = cache.self_attention_cache, cache.cross_attention_cache
        P = harvest_at
        ref["past"] = {
            "sk": [
                sc.layers[i].keys[..., :P, :].detach().clone()
                for i in range(NLAYERS)
            ],
            "sv": [
                sc.layers[i].values[..., :P, :].detach().clone()
                for i in range(NLAYERS)
            ],
            "ck": [cc.layers[i].keys.detach().clone() for i in range(NLAYERS)],
            "cv": [cc.layers[i].values.detach().clone() for i in range(NLAYERS)],
        }
        ref["hs"] = cap["args"][0].detach().clone()
        ref["ehs"] = (
            cap["args"][2].detach().clone()
            if len(cap["args"]) > 2
            else cap["kwargs"]["encoder_hidden_states"].detach().clone()
        )
    return step_ms, ref


class _DecodeStepWrap(nn.Module):
    """One decode step as a pure static-shape function.

    Inputs: ``hs`` (1,1,512 post-embed+pos), ``mask`` (1,1,1,L+1 float
    additive), then 24 past tensors — per layer ``sk, sv`` self-attn
    (1,8,L,64, chronological, pad slots zeroed) and ``ck, cv``
    cross-attn (1,8,1500,64, constant after step 0).  Ordering:
    ``sk[0..5], sv[0..5], ck[0..5], cv[0..5]``.

    Outputs: ``(logits, knew_0, vnew_0, ..., knew_5, vnew_5)`` where the
    K/V slices are (1,8,1,64) — the caller writes them into the buffer.

    Internally builds a fresh ``EncoderDecoderCache`` per call (exactly
    what ``torch.export`` can trace — the r2 decoder wall was the
    *whole-decoder* IR exploding, not this shape), prefills self+cross
    caches from the inputs, and forces ``is_updated`` so cross-attn
    takes the steady-state cache-read path (no K/V recompute — that is
    what real decode steps >= 1 do).
    """

    def __init__(self, m, ehs_in, cfg):
        super().__init__()
        dec = m.model.decoder
        self.layers = dec.layers
        self.norm = dec.layer_norm
        self.head = m.proj_out
        self.cfg = cfg
        self.register_buffer("ehs", ehs_in)

    def forward(self, hs, mask, *past):
        sk = past[0:NLAYERS]
        sv = past[NLAYERS : 2 * NLAYERS]
        ck = past[2 * NLAYERS : 3 * NLAYERS]
        cv = past[3 * NLAYERS : 4 * NLAYERS]
        c = copy.deepcopy(self.cfg)
        c.num_hidden_layers = self.cfg.decoder_layers
        sc = DynamicCache(config=c)
        cc = DynamicCache(config=c)
        cache = EncoderDecoderCache(sc, cc)
        for i in range(NLAYERS):
            sc.update(sk[i], sv[i], i)
            cc.update(ck[i], cv[i], i)
            cache.is_updated[i] = True
        h = hs
        for i, layer in enumerate(self.layers):
            h = layer(
                h,
                mask,
                self.ehs,
                None,
                past_key_values=cache,
                use_cache=True,
            )
        h = self.norm(h)
        logits = self.head(h)
        outs = [logits]
        for i in range(NLAYERS):
            outs.append(sc.layers[i].keys[..., -1:, :])
            outs.append(sc.layers[i].values[..., -1:, :])
        return tuple(outs)


def pad_self(t, L, P):
    """(1,8,P,64) -> (1,8,L,64) zero-padded on the right."""
    out = torch.zeros(1, NH, L, HDIM, device=t.device, dtype=t.dtype)
    out[:, :, :P] = t
    return out


def build_inputs(ref, L):
    """Harvest dict -> (inputs tuple, meta).  P = harvest past length."""
    P = ref["past"]["sk"][0].shape[-2]
    sk = [pad_self(t, L, P) for t in ref["past"]["sk"]]
    sv = [pad_self(t, L, P) for t in ref["past"]["sv"]]
    ck = list(ref["past"]["ck"])
    cv = list(ref["past"]["cv"])
    mask = torch.zeros(1, 1, 1, L + 1, device="cuda")
    # k_all = cat(buf_L, knew): allow buf slots 0..P-1 and last index L;
    # block pad slots P..L-1.
    mask[..., P:L] = torch.finfo(torch.float32).min
    inputs = (ref["hs"], mask, *(sk + sv + ck + cv))
    return inputs, {"P": P, "L": L}


def step_mask(t, L, device="cuda"):
    """Mask for decode step ``t`` (0-based): t valid past slots + the
    appended current-token slot at index L."""
    mask = torch.zeros(1, 1, 1, L + 1, device=device)
    mask[..., t:L] = torch.finfo(torch.float32).min
    return mask


def run_delivered_loop(step_fn, m, ehs, ntok, L, ck, cv):
    """Greedy loop driven by a delivered/static step module.

    ``step_fn(hs, mask, *past)`` -> (logits, knew_0, vnew_0, ...).
    Bookkeeping (embed+pos, buffer writes, argmax) stays eager — it is
    control flow / mutation, exactly what the pipeline cannot own.
    """
    dec = m.model.decoder
    # persistent fixed-capacity buffers, chronological order
    skb = [
        torch.zeros(1, NH, L, HDIM, device="cuda") for _ in range(NLAYERS)
    ]
    svb = [
        torch.zeros(1, NH, L, HDIM, device="cuda") for _ in range(NLAYERS)
    ]
    ids = torch.tensor(
        [[m.generation_config.decoder_start_token_id or 50257]],
        device="cuda",
    )
    step_ms = []
    gen_ids = []
    with torch.no_grad():
        for t in range(ntok):
            hs = embed_step(dec, ids, t)
            past = (
                *skb,
                *svb,
                *ck,
                *cv,
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = step_fn(hs, step_mask(t, L), *past)
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - t0) * 1e3)
            logits = outs[0][:, -1]
            # write this step's new self-attn K/V into slot t
            for i in range(NLAYERS):
                knew = outs[1 + 2 * i]
                vnew = outs[2 + 2 * i]
                if t < L:
                    skb[i][:, :, t : t + 1, :] = knew
                    svb[i][:, :, t : t + 1, :] = vnew
            ids = logits.argmax(-1, keepdim=True)
            gen_ids.append(ids[:, -1].item())
    return step_ms, gen_ids


def arm_eager(ntok: int) -> None:
    m = load()
    ehs = encoder_out(m)
    enc_ms = timeit(lambda: m.model.encoder(torch.randn(1, N_MELS, N_FRAMES, device="cuda")))
    print(f"[eager] encoder {enc_ms:.2f} ms", flush=True)
    step_ms, _ = greedy_loop_hf(m, ehs, ntok)
    tot = sum(step_ms)
    tail = sum(step_ms[-8:]) / 8
    print(
        f"[eager] decode loop {ntok} tok: total {tot:.1f} ms "
        f"({tot / ntok:.2f} ms/tok)  mean-step {sum(step_ms)/len(step_ms):.2f} "
        f"last8-mean {tail:.2f}",
        flush=True,
    )
    print("[eager] per-step ms:", " ".join(f"{v:.1f}" for v in step_ms), flush=True)


def arm_loop(iters: int, ntok: int, L: int, cpuparams: bool) -> None:
    import mlopt_carriers  # noqa: F401 — DECODE_RULES / ATN_SDPA_LAWS
    import mlopt_torch  # noqa: F401 — torch bindings
    from mlopt_cuda import CudaGraphRunner
    from mlopt_orchestrator.optimize import Optimizer
    from mlopt_torch.backend import TorchBackend

    HARVEST = min(8, ntok - 2)
    m = load()
    ehs = encoder_out(m)
    step_ms, ref = greedy_loop_hf(m, ehs, ntok, harvest_at=HARVEST)
    warm = sum(step_ms[1:])
    print(
        f"[eager] decode loop {ntok} tok: total {sum(step_ms):.1f} ms "
        f"warm {warm:.1f} ms ({warm/(ntok-1):.2f} ms/tok)  "
        f"last8 {sum(step_ms[-8:])/8:.2f}",
        flush=True,
    )
    print("[eager] per-step ms:", " ".join(f"{v:.1f}" for v in step_ms), flush=True)

    inputs, meta = build_inputs(ref, L)
    P = meta["P"]
    wrap = _DecodeStepWrap(m, ref["ehs"], m.config).eval()

    # --- correctness gate: wrap must reproduce the harvested step ---
    with torch.no_grad():
        o = wrap(*inputs)
        d = (o[0][:, -1].float() - ref["logits"]).abs().max().item()
        same = bool((o[0][:, -1].argmax(-1) == ref["next_id"]).all().item())
    print(
        f"[wrap] harvest step P={P} L={L}: logits max|diff|={d:.2e} "
        f"argmax_equal={same}",
        flush=True,
    )
    if not same:
        print("[wrap] SANITY FAIL — refusing to optimize", flush=True)
        return

    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    try:
        res = optz.search(wrap, inputs, max_iterations=iters)
        print(
            f"[search] {time.time() - t0:.1f}s "
            f"enodes={getattr(res, 'n_enodes', '?')}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[search] FAILED {time.time() - t0:.1f}s: {type(e).__name__}: {e}", flush=True)
        return

    # free the e-graph/provenance handles — they pin tensors (r2 lesson)
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None
    res.source = None
    if cpuparams:
        res.param_values = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in res.param_values.items()
        }
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    low = None
    try:
        low = optz.lower(res, inputs, verify=True, runner=None)
        ok = low.verified.passed if low.verified else None
        print(f"[lower+verify] {time.time() - t0:.1f}s verified={ok}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[lower+verify] FAILED {time.time() - t0:.1f}s: {type(e).__name__}: {e}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    if low is None:
        t0 = time.time()
        try:
            low = optz.lower(res, inputs, verify=False, runner=None)
            print(f"[lower(verify=False)] {time.time() - t0:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[lower(verify=False)] FAILED: {type(e).__name__}: {e}", flush=True)
            return

    gc.collect()
    torch.cuda.empty_cache()

    # --- per-step timings ---
    e_ms = timeit(lambda: wrap(*inputs), warm=3, reps=20)
    d_ms = timeit(lambda: low.module(*inputs), warm=3, reps=20)
    print(f"[step] eager-wrap {e_ms:.2f} ms   delivered {d_ms:.2f} ms", flush=True)

    # --- cuda-graph delivery ---
    low2 = None
    try:
        low2 = optz.lower(res, inputs, verify=False, runner=CudaGraphRunner())
        print(
            f"[graph] cuda_graph={low2.stats.get('cuda_graph')} "
            f"fail={low2.stats.get('cuda_graph_failure')}",
            flush=True,
        )
        if low2.stats.get("cuda_graph"):
            g_ms = timeit(lambda: low2.module(*inputs), warm=3, reps=20)
            print(f"[step] delivered+graph {g_ms:.2f} ms", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[graph] lower FAILED: {type(e).__name__}: {e}", flush=True)

    # --- torch.compile on the same wrap ---
    # ``wrap`` shares the decoder parameters with ``m`` (still alive),
    # so compiling it is the honest "compile the same decode step" arm.
    cmod = None
    try:
        cmod = torch.compile(wrap, dynamic=False)
        with torch.no_grad():
            cmod(*inputs)
            torch.cuda.synchronize()
        c_ms = timeit(lambda: cmod(*inputs), warm=3, reps=20)
        print(f"[step] torch.compile(wrap) {c_ms:.2f} ms", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[compile] FAILED: {type(e).__name__}: {e}", flush=True)
        cmod = None

    # --- end-to-end: delivered module inside the real greedy loop ---
    # ``m`` stays alive for embed_tokens/embed_positions bookkeeping;
    # ``ck``/``cv`` are the constant cross-attn cache (identical every
    # step after step 0 — reuse the harvested tensors).
    ck = list(ref["past"]["ck"])
    cv = list(ref["past"]["cv"])
    eager_ids = ref["gen_ids"]

    def report_loop(tag, fn):
        steps, gids = run_delivered_loop(fn, m, ehs, ntok, L, ck, cv)
        warm = sum(steps[1:])
        agree = sum(1 for a, b in zip(gids, eager_ids) if a == b)
        first_div = next(
            (i for i, (a, b) in enumerate(zip(gids, eager_ids)) if a != b),
            None,
        )
        print(
            f"[loop] {tag} {ntok} tok: total {sum(steps):.1f} ms "
            f"warm {warm:.1f} ms ({warm/(ntok-1):.2f} ms/tok)  "
            f"last8 {sum(steps[-8:])/8:.2f}  ids-match {agree}/{ntok} "
            f"first-div={first_div}",
            flush=True,
        )
        print(f"[loop] {tag} per-step:", " ".join(f"{v:.1f}" for v in steps), flush=True)

    report_loop("delivered-module", lambda *a: low.module(*a))
    if low2 is not None and low2.stats.get("cuda_graph"):
        report_loop("delivered+graph", lambda *a: low2.module(*a))
    if cmod is not None:
        report_loop("torch.compile", lambda *a: cmod(*a))
    del m
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "loop"
    if mode == "eager":
        arm_eager(int(sys.argv[2]) if len(sys.argv) > 2 else 32)
    elif mode == "loop":
        iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 32
        L = int(sys.argv[4]) if len(sys.argv) > 4 else max(40, ntok)
        cpuparams = len(sys.argv) > 5 and sys.argv[5] == "cpuparams"
        arm_loop(iters, ntok, L, cpuparams)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()

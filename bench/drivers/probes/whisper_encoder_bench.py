"""Whisper ENCODER lift + optimize benchmark — measurement driver.

The encoder is the untouched half of whisper e2e (62% of turbo's
810 ms in ``whisper_production_bench.md``).  This probe lifts the HF
``WhisperEncoder`` (conv1d frontend + sinusoidal/learned pos-embed +
N pre-norm transformer encoder layers) through the normal
export → IRModule path and measures honestly vs HF eager / compile.

Modes::

    census <model>            export_to_ir census: op histogram +
                              opaque-leaf aten quals (no search).
    opt <model> [iters]       search + lower(verify=True); timing arms:
                              hf-eager / delivered / delivered+graph /
                              torch.compile; max|diff| vs HF output.
    e2e turbo [iters] [ntok]  delivered encoder + the prod-bench
                              delivered+graph decode loop = the real
                              production e2e number.

Usage (transformers in /tmp/whisper_deps)::

    PYTHONPATH=/tmp/whisper_deps .venv/bin/python \
        bench/drivers/probes/whisper_encoder_bench.py census base
    PYTHONPATH=/tmp/whisper_deps .venv/bin/python \
        bench/drivers/probes/whisper_encoder_bench.py opt turbo 3
    PYTHONPATH=/tmp/whisper_deps .venv/bin/python \
        bench/drivers/probes/whisper_encoder_bench.py e2e turbo 3 48
"""

from __future__ import annotations

import gc
import importlib.util
import sys
import time
from collections import Counter
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


prod = _load("whisper_prod_bench", "whisper_prod_bench.py")

MODELS = prod.MODELS  # base fp32 / small fp32 / turbo bf16
SR = prod.SR
AUDIO_S = prod.AUDIO_S
REPS = prod.REPS


class EncWrap(nn.Module):
    """WhisperEncoder → last_hidden_state tensor (sink needs a tensor)."""

    def __init__(self, enc: nn.Module):
        super().__init__()
        self.enc = enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x).last_hidden_state


def load_encoder(model: str):
    """Load the HF model, keep ONLY the encoder on GPU.

    turbo's encoder alone is ~1.4 GB bf16 — dropping the decoder up
    front leaves headroom for search/verify on the 4 GB card.
    """
    m = prod.load(model)
    enc = m.model.encoder
    for attr in ("decoder", "proj_out", "lm_head"):
        parent = m.model if attr == "decoder" else m
        if hasattr(parent, attr):
            delattr(parent, attr)
    gc.collect()
    torch.cuda.empty_cache()
    return enc.eval()


def hf_enc_out(enc: nn.Module, feats: torch.Tensor) -> torch.Tensor:
    dt = next(enc.parameters()).dtype
    with torch.no_grad():
        return enc(feats.to(dt)).last_hidden_state


def time_forward(fn, *args, reps: int = REPS, warm: int = 2):
    """Warm ``warm`` calls then ``reps`` synchronized timings (ms)."""
    for _ in range(warm):
        fn(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def term_walk(root):
    """Yield every Op node in an IR term DAG (dedup by identity)."""
    from mlopt_core.ir import Op, _dag_postorder

    return [n for n in _dag_postorder(root) if isinstance(n, Op)]


def census(model: str) -> None:
    from mlopt_torch.torch_bridge import export_to_ir

    enc = load_encoder(model)
    feats = prod.make_features(model).to(next(enc.parameters()).dtype)
    wrap = EncWrap(enc)
    try:
        ir, _src = export_to_ir(wrap, feats)
    except Exception as e:  # noqa: BLE001
        print(f"[{model}] export_to_ir FAILED {type(e).__name__}: {e}", flush=True)
        return
    ops = term_walk(ir.root)
    hist = Counter(o.op for o in ops)
    opaque = Counter(
        str(o.attrs.get("aten", "?")) for o in ops if o.op == "opaque"
    )
    print(f"[{model}] IR ops={len(ops)} distinct={len(hist)}", flush=True)
    for name, n in hist.most_common():
        print(f"[{model}]   {name}: {n}", flush=True)
    if opaque:
        print(f"[{model}] OPAQUE leaves={sum(opaque.values())}:", flush=True)
        for qual, n in opaque.most_common():
            print(f"[{model}]   {qual}: {n}", flush=True)
    else:
        print(f"[{model}] OPAQUE leaves: none", flush=True)
    del enc, wrap, ir
    gc.collect()
    torch.cuda.empty_cache()


def _verify_vs_hf(ref, feats, delivered, tag: str) -> None:
    """max|diff| delivered-vs-HF — lower(verify=True) only gates
    delivered-vs-relowered-IR, so the HF-faithfulness check is ours.
    ``ref`` is a precomputed HF encoder output tensor."""
    with torch.no_grad():
        out = delivered(feats)
        if isinstance(out, (tuple, list)):
            out = out[0]
    d = (out.float() - ref.float()).abs().max().item()
    rel = d / ref.float().abs().max().item()
    print(
        f"[{tag}] vs-HF-ref max|diff|={d:.3e} rel={rel:.3e}",
        flush=True,
    )


def opt(model: str, iters: int, exclude: str = "") -> None:
    import kit_carriers  # noqa: F401
    import mlopt_torch  # noqa: F401
    from kit_cuda import CudaGraphRunner
    from mlopt_orchestrator.optimize import Optimizer
    from mlopt_torch.backend import TorchBackend

    enc = load_encoder(model)
    wrap = EncWrap(enc)
    feats = prod.make_features(model).to(next(enc.parameters()).dtype)
    print(f"[{model}] encoder params={sum(p.numel() for p in enc.parameters())/1e6:.1f}M "
          f"dtype={next(enc.parameters()).dtype} feats={tuple(feats.shape)}",
          flush=True)

    # ---- HF eager reference timing + snapshot ----
    t_hf = time_forward(lambda: hf_enc_out(enc, feats))
    print(f"[{model}] hf-eager enc ms: {['%.1f' % v for v in t_hf]}", flush=True)
    with torch.no_grad():
        ref_hf = hf_enc_out(enc, feats).detach().clone()

    # ---- search + lower ----
    optz = Optimizer(backend=TorchBackend())
    search_kw = {}
    if exclude:
        from mlopt_orchestrator.optimize import DEFAULT_RULES
        rules = [r for r in DEFAULT_RULES if exclude not in r.name]
        print(f"[{model}] rules={len(rules)} (excluded '*{exclude}*')",
              flush=True)
        search_kw["rules"] = rules
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        res = optz.search(wrap, feats, max_iterations=iters, **search_kw)
    except Exception as e:  # noqa: BLE001
        print(
            f"[{model}] search FAILED {type(e).__name__}: {e}",
            flush=True,
        )
        return
    print(
        f"[{model}] search {time.time()-t0:.1f}s "
        f"enodes={getattr(res, 'n_enodes', '?')}",
        flush=True,
    )
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None
    res.source = None
    # The HF encoder (~1.4 GB on turbo) is dead weight now — verify
    # re-lowers result.ir rather than re-running the source module,
    # so dropping it is what makes lower(verify=True) fit in 4 GB.
    del enc, wrap
    gc.collect()
    torch.cuda.empty_cache()

    low = None
    try:
        low = optz.lower(res, feats, verify=True, runner=None)
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
            low = optz.lower(res, feats, verify=False, runner=None)
            print(f"[{model}] lower(verify=False) ok", flush=True)
        except Exception as e:  # noqa: BLE001
            print(
                f"[{model}] lower(verify=False) FAILED "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            return
    mod = low.module
    _verify_vs_hf(ref_hf, feats, mod, model)

    # ---- arms ----
    arms = [("delivered", mod)]

    gmod = None
    try:
        gmod = CudaGraphRunner(multi_device=True).apply(
            mod, feats, {}
        )
        arms.append(("delivered+graph", gmod))
    except Exception as e:  # noqa: BLE001
        print(f"[{model}] graph arm FAILED {type(e).__name__}: {e}", flush=True)

    for tag, fn in arms:
        try:
            ts = time_forward(lambda: fn(feats))
            _verify_vs_hf(ref_hf, feats, fn, f"{model}/{tag}")
            print(
                f"[{model}] enc {tag} ms: "
                f"{['%.1f' % v for v in ts]}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[{model}] arm {tag} FAILED {type(e).__name__}: {e}",
                flush=True,
            )

    # compile on the bare encoder (HF-side comparison) — re-load a
    # throwaway encoder copy only if VRAM allows; skip silently on OOM.
    try:
        enc2 = load_encoder(model)
        cenc = torch.compile(enc2, dynamic=False)
        with torch.no_grad():
            cenc(feats)
            torch.cuda.synchronize()
        ts = time_forward(
            lambda: cenc(feats).last_hidden_state
        )
        print(
            f"[{model}] enc compile-hf ms: "
            f"{['%.1f' % v for v in ts]}",
            flush=True,
        )
        del cenc, enc2
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        print(
            f"[{model}] compile-hf FAILED/skipped {type(e).__name__}: {e}",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[{model}] peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",
          flush=True)
    del mod, gmod, low, res, ref_hf
    gc.collect()
    torch.cuda.empty_cache()




def _mem(tag: str) -> None:
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    print(f"    [mem {tag}] alloc={a:.2f}GiB reserved={r:.2f}GiB", flush=True)


def diag(model: str, iters: int, exclude: str = "") -> None:
    """Three-way correctness + memory attribution for the encoder.

    delivered(IRModule of extracted term) vs fresh lowering of the
    UNOPTIMIZED exported IR vs HF eager — isolates whether a
    delivered-vs-HF gap is export fidelity or extraction/lowering.
    """
    import kit_carriers  # noqa: F401
    import mlopt_torch  # noqa: F401
    from mlopt_orchestrator.optimize import Optimizer
    from mlopt_torch.backend import TorchBackend
    from mlopt_torch.adapters import TorchSink

    enc = load_encoder(model)
    wrap = EncWrap(enc)
    feats = prod.make_features(model).to(next(enc.parameters()).dtype)
    with torch.no_grad():
        ref_hf = hf_enc_out(enc, feats).detach().clone()
    _mem("after hf ref")

    optz = Optimizer(backend=TorchBackend())
    if exclude:
        from mlopt_orchestrator.optimize import DEFAULT_RULES
        rules = [r for r in DEFAULT_RULES if exclude not in r.name]
        print(f"[diag] rules={len(rules)} (excluded '*{exclude}*')",
              flush=True)
        res = optz.search(wrap, feats, max_iterations=iters, rules=rules)
    else:
        res = optz.search(wrap, feats, max_iterations=iters)
    print(f"[diag] search done", flush=True)
    _mem("after search")
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None
    res.source = None
    # Dump the extracted term + IR + weights so follow-up evals skip
    # the ~150 s research (pickle is small; weights via torch.save).
    try:
        import pickle
        with open(f"/tmp/enc_term_{model}.pkl", "wb") as f:
            pickle.dump({"term": res.term, "ir": res.ir}, f)
        torch.save(dict(res.param_values), f"/tmp/enc_params_{model}.pt")
        print(f"[diag] dumped term+params to /tmp/enc_*_{model}.*",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[diag] dump FAILED {type(e).__name__}: {e}", flush=True)
    del enc, wrap
    gc.collect()
    torch.cuda.empty_cache()
    _mem("after del enc/wrap")

    sink = TorchSink()
    low = optz.lower(res, feats, verify=False, runner=None)
    print(f"[diag] lower stats={dict(low.stats)}", flush=True)
    _mem("after lower(verify=False)")
    mod = low.module

    # fresh lowering of the UNOPTIMIZED ir = the verify reference
    ref_mod = sink.lower(res.ir, dict(res.param_values))
    _mem("after ref lower")

    with torch.no_grad():
        out_del = mod(feats)
        if isinstance(out_del, (tuple, list)):
            out_del = out_del[0]
    _mem("after delivered fwd")
    torch.cuda.empty_cache()
    with torch.no_grad():
        out_ref = ref_mod(feats)
        if isinstance(out_ref, (tuple, list)):
            out_ref = out_ref[0]
    _mem("after ref-ir fwd")
    torch.cuda.empty_cache()

    for tag, a, b in (
        ("delivered-vs-IRref", out_del, out_ref),
        ("IRref-vs-HF", out_ref, ref_hf),
        ("delivered-vs-HF", out_del, ref_hf),
    ):
        d = (a.float() - b.float()).abs().max().item()
        rel = d / b.float().abs().max().item()
        print(f"[diag] {tag}: max|diff|={d:.3e} rel={rel:.3e}", flush=True)

    del mod, ref_mod, low, res
    gc.collect()
    torch.cuda.empty_cache()



def termtest(model: str, mode: str, dtype: str = "") -> None:
    """Re-eval a dumped extracted term vs the dumped IR reference.

    ``mode``: ``default`` | ``mathsdpa`` (force SDPBackend.MATH) |
    ``nosdpa`` is meaningless here (term already contains sdpa).
    """
    import pickle
    import kit_carriers  # noqa: F401  (seats carrier bindings)
    import mlopt_torch  # noqa: F401
    from mlopt_torch.torch_eval import ir_to_torch_module
    from mlopt_torch.adapters import TorchSink
    from mlopt_core.ir import IR

    _OPS = TorchSink()._ops

    with open(f"/tmp/enc_term_{model}.pkl", "rb") as f:
        blob = pickle.load(f)
    term, ir = blob["term"], blob["ir"]
    params = torch.load(f"/tmp/enc_params_{model}.pt", weights_only=False)

    if mode == "mathsdpa":
        import mlopt_torch.torch_bindings as tb
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import torch.nn.functional as F

        orig = tb._IR_TO_TORCH.get("sdpa")

        def _math_sdpa(q, k, v, *a, **kw):
            kwargs = {"attn_mask": a[0]} if a else {}
            kwargs.update(kw)
            with sdpa_kernel([SDPBackend.MATH]):
                return F.scaled_dot_product_attention(q, k, v, **kwargs)

        tb._IR_TO_TORCH["sdpa"] = _math_sdpa
        print("[termtest] patched sdpa -> MATH backend", flush=True)

    feats = prod.make_features(model)
    # feats dtype must match param dtype
    dt = next(iter(params.values())).dtype
    if dtype == "fp32":
        # cast in place — a dict comprehension would hold the bf16
        # source AND the fp32 copy (~3.8 GiB) and OOM the 4 GB card
        for k in list(params):
            params[k] = params.pop(k).float()
        dt = torch.float32
    elif dtype == "fp64":
        for k in list(params):
            params[k] = params.pop(k).double()
        dt = torch.float64
    if dtype == "cpu":
        params = {k: v.cpu() for k, v in params.items()}
        dt = next(iter(params.values())).dtype
        feats = feats.to(dt).cpu()
    else:
        feats = feats.to(dt)

    # delivered module from extracted term
    opt_ir = IR(root=term, inputs=ir.inputs,
                input_names=ir.input_names, params=ir.params)
    mod = ir_to_torch_module(opt_ir, param_values=dict(params), ops=_OPS)
    with torch.no_grad():
        out_del = mod(feats)
        if isinstance(out_del, (tuple, list)):
            out_del = out_del[0]
        out_del = out_del.detach().clone()
    del mod
    gc.collect()
    torch.cuda.empty_cache()
    # reference: fresh lowering of unoptimized ir
    ref_mod = ir_to_torch_module(ir, param_values=dict(params), ops=_OPS)
    with torch.no_grad():
        out_ref = ref_mod(feats)
        if isinstance(out_ref, (tuple, list)):
            out_ref = out_ref[0]
        out_ref = out_ref.detach().clone()
    del ref_mod
    gc.collect()
    torch.cuda.empty_cache()
    # timing: warm 2 + 3 reps on the delivered module (rebuild cheaply
    # is avoided — mod was deleted; rebuild for timing only if asked)
    d = (out_del.float() - out_ref.float()).abs().max().item()
    rel = d / out_ref.float().abs().max().item()
    print(f"[termtest:{model}:{mode}] delivered-vs-IRref "
          f"max|diff|={d:.3e} rel={rel:.3e} "
          f"del_dtype={out_del.dtype} ref_dtype={out_ref.dtype} "
          f"param_dtype={next(iter(params.values())).dtype}",
          flush=True)

    # does the term contain sdpa? count + scale attrs
    from mlopt_core.ir import Op, _dag_postorder
    ops = [n for n in _dag_postorder(term) if isinstance(n, Op)]
    sds = [n for n in ops if n.op == "sdpa"]
    print(f"[termtest] ops={len(ops)} sdpa={len(sds)} "
          f"scales={sorted({str(n.attrs.get('scale')) for n in sds})}",
          flush=True)

def e2e(model: str, iters: int, ntok: int) -> None:
    """Delivered encoder + prod-bench delivered+graph decode loop.

    Reuses ``whisper_prod_bench.arm_opt``'s decode machinery by
    re-implementing the composition: ehs comes from the DELIVERED
    encoder module (not HF eager), everything else identical.
    """
    import kit_carriers  # noqa: F401
    import mlopt_torch  # noqa: F401
    from kit_cuda import CudaGraphRunner
    from mlopt_orchestrator.optimize import (
        Optimizer,
        VerifyDeclinedError,
    )
    from mlopt_torch.backend import TorchBackend

    if model != "turbo":
        print("e2e mode is turbo-focused; run opt for others", flush=True)
    m = prod.load(model)
    n_layers, nh, hdim = prod.configure(model, m)
    dtype = next(m.parameters()).dtype
    enc = m.model.encoder
    dec = m.model.decoder
    feats = prod.make_features(model).to(dtype)

    # get_prefix runs a full HF generate (needs the encoder) — do it
    # before the encoder is dropped for VRAM.
    prefix = prod.get_prefix(m, feats)
    K = len(prefix)
    print(f"[e2e] prefix={prefix} K={K}", flush=True)

    # ---- optimize the encoder (decoder kept for the decode loop) ----
    wrap = EncWrap(enc)
    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    res = optz.search(wrap, feats, max_iterations=iters)
    print(f"[e2e] enc search {time.time()-t0:.1f}s", flush=True)
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None
    res.source = None
    gc.collect()
    torch.cuda.empty_cache()
    # Snapshot the HF reference, then drop the HF encoder (~1.4 GB):
    # lower(verify=True) re-lowers result.ir rather than re-running
    # the source module, so this is what makes verify fit in 4 GB.
    with torch.no_grad():
        ref_hf = hf_enc_out(enc, feats).detach().clone()
    del enc, wrap, m.model.encoder
    gc.collect()
    torch.cuda.empty_cache()
    try:
        low = optz.lower(res, feats, verify=True, runner=None)
    except VerifyDeclinedError as e:
        # declined verify still carries the LowerResult — record
        # verified=False and keep measuring, as pre-gate lower did
        low = e.result
    print(
        f"[e2e] enc lower+verify "
        f"verified={low.verified.passed if low.verified else None}",
        flush=True,
    )
    enc_mod = low.module
    del res, low
    gc.collect()
    torch.cuda.empty_cache()

    # ---- delivered encoder timing ----
    t_enc = time_forward(lambda: enc_mod(feats))
    print(f"[e2e] delivered-enc ms: {['%.1f' % v for v in t_enc]}", flush=True)
    with torch.no_grad():
        ehs = enc_mod(feats)
        if isinstance(ehs, (tuple, list)):
            ehs = ehs[0]
    # HF-faithfulness of the delivered encoder
    d = (ehs.float() - ref_hf.float()).abs().max().item()
    rel = d / ref_hf.float().abs().max().item()
    print(f"[e2e] enc delivered-vs-HF max|diff|={d:.3e} rel={rel:.3e}",
          flush=True)
    del ref_hf
    gc.collect()
    torch.cuda.empty_cache()

    # ---- decode side: identical to prod bench arm_opt ----
    L = max(80, ntok + 16)
    g_harvest = 8
    harvest_call = (K - 1) + g_harvest
    gen_ids, step_ms, harvest = prod.hf_greedy_loop(
        m, ehs, prefix, ntok, harvest_call=harvest_call
    )
    P = harvest["P"]
    print(f"[e2e] hf-loop warm {sum(step_ms[1:])/(ntok-1):.2f} ms/tok P={P}",
          flush=True)
    inputs_s, meta = prod.kv.build_inputs_static(harvest, L)
    inputs_s = list(inputs_s)
    inputs_s[1] = inputs_s[1].to(dtype)
    inputs_s = tuple(inputs_s)
    wrap_s = prod._DecodeStepWrap457(m, harvest["ehs"], m.config).eval()
    supp, begin, eos = prod.suppress_sets(m)
    with torch.no_grad():
        o = wrap_s(*inputs_s)
        nxt_wrap = prod.apply_suppress(
            o[0][:, -1].float(), supp, begin, first=False, eos=eos
        )
        same = bool(
            (nxt_wrap.reshape(-1) == harvest["next_id"].reshape(-1)).all().item()
        )
    if not same:
        print("[e2e] wrap sanity FAIL — aborting decode opt", flush=True)
        return

    ck = list(harvest["past"]["ck"])
    cv = list(harvest["past"]["cv"])
    res2 = optz.search(wrap_s, inputs_s, max_iterations=iters)
    print(f"[e2e] dec search {time.time()-t0:.1f}s "
          f"enodes={getattr(res2, 'n_enodes', '?')}", flush=True)
    res2.eg = None  # ty: ignore[invalid-assignment]
    res2.model = None
    res2.source = None
    gc.collect()
    torch.cuda.empty_cache()
    try:
        low2 = optz.lower(res2, inputs_s, verify=True, runner=None)
    except VerifyDeclinedError as e:
        low2 = e.result
    print(
        f"[e2e] dec lower+verify "
        f"verified={low2.verified.passed if low2.verified else None}",
        flush=True,
    )
    dec_mod = low2.module
    dmod = CudaGraphRunner(
        multi_device=True,
        alias_inputs=lambda i, _t: i >= 3,
        mutable_inputs=lambda i, _t: 3 <= i < 3 + 2 * n_layers,
    ).apply(dec_mod, inputs_s, {})

    # ---- the composed loop: delivered encoder → delivered+graph decode ----
    ckcv = (ck, cv)
    per_tok, all_steps, gids, pk = prod.time_loop(
        dmod, m, ehs, prefix, ntok, L, dtype, ckcv
    )
    agree = sum(1 for a, b in zip(gids, gen_ids) if a == b)
    enc_mean = sum(t_enc) / len(t_enc)
    dec_ms = sum(per_tok) / len(per_tok) * ntok
    e2e_ms = enc_mean + dec_ms
    print(
        f"[e2e] composed: enc {enc_mean:.1f} ms + dec "
        f"{['%.2f' % v for v in per_tok]} ms/tok ×{ntok} "
        f"= ~{e2e_ms:.0f} ms  ids-match {agree}/{len(gen_ids)} "
        f"peak={pk:.2f}GiB",
        flush=True,
    )
    print(f"[e2e] dec ids={gids}", flush=True)
    print(f"[e2e] hf  ids={gen_ids}", flush=True)

    del m, enc_mod, dec_mod, dmod, low2, res2, wrap_s
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "census"
    model = sys.argv[2] if len(sys.argv) > 2 else "base"
    if model not in MODELS:
        raise SystemExit(f"unknown model {model}; choose {list(MODELS)}")
    if mode == "census":
        census(model)
    elif mode == "opt":
        iters = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        opt(model, iters)
    elif mode == "optx":
        iters = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        excl = sys.argv[4] if len(sys.argv) > 4 else ""
        opt(model, iters, excl)
    elif mode == "diag":
        iters = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        excl = sys.argv[4] if len(sys.argv) > 4 else ""
        diag(model, iters, excl)
    elif mode == "termtest":
        tm = sys.argv[3] if len(sys.argv) > 3 else "default"
        td = sys.argv[4] if len(sys.argv) > 4 else ""
        termtest(model, tm, td)
    elif mode == "e2e":
        iters = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        ntok = int(sys.argv[4]) if len(sys.argv) > 4 else 48
        e2e(model, iters, ntok)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()

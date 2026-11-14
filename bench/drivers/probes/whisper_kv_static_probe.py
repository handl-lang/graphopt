"""K/V-cache cat-elimination probe — measurement driver.

Follow-up to ``docs/reports/fusion_floor_probe.md`` §6.1(a): the decode
step's ~0.95 ms/call ``CatArrayBatchedCopy`` prize.  The delivered
``_DecodeStepWrap`` pays ``torch.cat`` inside ``DynamicCache.update``
36x/call: 12 cross-attn prefills copying the ~3 MB encoder K/V,
12 self-attn prefills, 12 self-attn appends.

This probe replaces the per-call ``DynamicCache`` with **caller-owned
fixed-capacity buffers + in-place slot writes**:

* self-attn K/V inputs are the loop's persistent ``(1,8,L,64)``
  buffers; a custom cache layer adopts them BY REFERENCE (no prefill
  copy) and its decode ``update`` does
  ``keys.index_copy_(2, pos, knew)`` — an in-place write at the
  dynamic position ``pos`` (a module input).  ``aten.index_copy_``
  survives ``torch.export`` as a mutation node OUTSIDE the
  ``copy_/fill_/zero_/masked_fill_`` threading prefix, so it lifts as
  an ``opaque`` leaf that replays the real aten op at eval — a TRUE
  in-place write, not a functionalised full-buffer copy.  The sdpa
  consumer reads the same (mutated) buffer: zero cats, zero extra
  kernels of size.
* cross-attn K/V inputs (constant after step 0) are adopted by
  reference — attention's ``is_updated`` path reads
  ``layers[i].keys`` directly, so the ~36 MB/call of cross-prefill
  ``cat`` copies disappear entirely.

Attention is permutation-invariant over the mask-valid key set, so
writing the current token at chronological slot ``t`` (mask allows
``0..t``) is exactly equivalent to the old ``cat(past, knew@L)`` +
``{0..t-1, L}`` mask.

Arms: static-wrap eager vs harvested HF logits (gate),
``Optimizer.search`` → ``lower(verify=True)``, delivered eager /
``CudaGraphRunner`` (copy and ``alias_inputs`` — the alias path binds
the persistent buffers so the in-graph ``index_copy_`` maintains the
cache with zero copies), and the full greedy loop with ids-match.

Usage::

    .venv/bin/python bench/drivers/probes/whisper_kv_static_probe.py [iters] [ntok] [L]
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers.cache_utils import (
    CacheLayerMixin,
    DynamicCache,
    EncoderDecoderCache,
)

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "whisper_decode_loop", HERE / "whisper_decode_loop.py"
)
probe = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(probe)

NLAYERS = probe.NLAYERS


class _KVWriteLayer(CacheLayerMixin):
    """Self-attn cache layer over a caller-owned fixed buffer.

    ``update`` is called twice per step: the wrap's prefill adopts the
    ``(1,8,L,64)`` input buffer BY REFERENCE (``_primed`` flips), then
    the attention module's append does ``index_copy_`` at position
    ``pos`` — an in-place slot write, not a ``cat``.  ``last_k`` /
    ``last_v`` retain the 1-slot ``knew``/``vnew`` for the wrap's
    outputs (same values the old ``keys[..., -1:, :]`` returned).
    """

    def __init__(self, pos):
        super().__init__()
        self.pos = pos
        self._primed = False
        self.last_k = None
        self.last_v = None

    def lazy_initialization(self, key_states, value_states):
        # Buffers are adopted by reference at prefill — nothing to
        # allocate.  Present only to satisfy the ABC.
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self._primed:
            self._primed = True
            self.keys = key_states
            self.values = value_states
            self.is_initialized = True
            return key_states, value_states
        # Decode append: in-place write of the 1-slot K/V at slot ``pos``.
        # ``aten.index_copy_`` exports as an opaque mutation node — the
        # sdpa below reads the SAME buffer object, post-write.
        self.keys.index_copy_(2, self.pos, key_states)
        self.values.index_copy_(2, self.pos, value_states)
        self.last_k = key_states
        self.last_v = value_states
        return self.keys, self.values

    def get_mask_sizes(self, query_length):
        kv_length = self.keys.shape[-2] if self.keys is not None else 0
        return kv_length, 0

    def get_seq_length(self):
        if self.keys is None:
            return 0
        return int(self.keys.shape[-2])

    def get_max_length(self):
        return -1

    def reset(self):
        self.keys = self.values = None
        self.last_k = self.last_v = None
        self._primed = False
        self.is_initialized = False


class _KVRefLayer(CacheLayerMixin):
    """Cross-attn cache layer: adopt the constant encoder K/V by
    reference.  Attention's ``is_updated`` path reads
    ``layers[i].keys`` directly — zero copy."""

    def lazy_initialization(self, key_states, value_states):
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        self.keys = key_states
        self.values = value_states
        self.is_initialized = True
        return key_states, value_states

    def get_mask_sizes(self, query_length):
        kv_length = self.keys.shape[-2] if self.keys is not None else 0
        return kv_length, 0

    def get_seq_length(self):
        if self.keys is None:
            return 0
        return int(self.keys.shape[-2])

    def get_max_length(self):
        return -1

    def reset(self):
        self.keys = self.values = None
        self.is_initialized = False


class _DecodeStepWrapStatic(nn.Module):
    """``_DecodeStepWrap`` with caller-owned K/V buffers.

    Inputs: ``hs`` (1,1,512), ``mask`` (1,1,1,L float additive,
    allows positions ``0..t``), ``pos`` (1,) int64 write position,
    then the same 24 past tensors — self K/V are now the *whole*
    ``L``-slot buffers the module mutates in place (slot ``pos`` gets
    this step's ``knew``/``vnew`` before sdpa reads them), cross K/V
    are adopted by reference.

    Outputs: ``(logits, knew_0, vnew_0, ..., knew_5, vnew_5)`` —
    identical contract to ``_DecodeStepWrap``.
    """

    def __init__(self, m, ehs_in, cfg):
        super().__init__()
        dec = m.model.decoder
        self.layers = dec.layers
        self.norm = dec.layer_norm
        self.head = m.proj_out
        self.cfg = cfg
        self.register_buffer("ehs", ehs_in)

    def forward(self, hs, mask, pos, *past):
        sk = past[0:NLAYERS]
        sv = past[NLAYERS : 2 * NLAYERS]
        ck = past[2 * NLAYERS : 3 * NLAYERS]
        cv = past[3 * NLAYERS : 4 * NLAYERS]
        sc = DynamicCache()
        cc = DynamicCache()
        sc.layers = [_KVWriteLayer(pos) for _ in range(NLAYERS)]
        cc.layers = [_KVRefLayer() for _ in range(NLAYERS)]
        cache = EncoderDecoderCache(sc, cc)
        for i in range(NLAYERS):
            sc.update(sk[i], sv[i], i)
            cc.update(ck[i], cv[i], i)
            cache.is_updated[i] = True
        h = hs
        for layer in self.layers:
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
            outs.append(sc.layers[i].last_k)
            outs.append(sc.layers[i].last_v)
        return tuple(outs)


def build_inputs_static(ref, L):
    """Harvest dict -> static-wrap inputs.  ``pos`` = harvest past
    length P; mask allows positions ``0..P`` (slot P receives the
    current token's K/V in-place)."""
    inputs, _meta = probe.build_inputs(ref, L)
    P = _meta["P"]
    # Rebuild the mask at length L (not L+1): the buffer IS the kv.
    mask = torch.zeros(1, 1, 1, L, device="cuda")
    mask[..., P + 1 : L] = torch.finfo(torch.float32).min
    pos = torch.tensor([P], device="cuda")
    return (inputs[0], mask, pos, *inputs[2:]), {"P": P, "L": L}


def step_mask_static(t, L, device="cuda"):
    """Mask for decode step ``t``: allow positions ``0..t``."""
    mask = torch.zeros(1, 1, 1, L, device=device)
    mask[..., t + 1 : L] = torch.finfo(torch.float32).min
    return mask


def run_static_loop(step_fn, m, ehs, ntok, L, ck, cv):
    """Greedy loop for the static wrap — the module owns the K/V
    cache: it writes slot ``t`` itself, so the caller performs NO
    buffer writes (unlike ``run_delivered_loop``)."""
    dec = m.model.decoder
    skb = [
        torch.zeros(1, probe.NH, L, probe.HDIM, device="cuda")
        for _ in range(NLAYERS)
    ]
    svb = [
        torch.zeros(1, probe.NH, L, probe.HDIM, device="cuda")
        for _ in range(NLAYERS)
    ]
    ids = torch.tensor(
        [[m.generation_config.decoder_start_token_id or 50257]],
        device="cuda",
    )
    pos = torch.empty(1, dtype=torch.long, device="cuda")
    step_ms = []
    gen_ids = []
    with torch.no_grad():
        for t in range(ntok):
            hs = probe.embed_step(dec, ids, t)
            pos[0] = t
            past = (*skb, *svb, *ck, *cv)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = step_fn(hs, step_mask_static(t, L), pos, *past)
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - t0) * 1e3)
            logits = outs[0][:, -1]
            ids = logits.argmax(-1, keepdim=True)
            gen_ids.append(ids[:, -1].item())
    return step_ms, gen_ids


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


def main() -> None:
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    L = int(sys.argv[3]) if len(sys.argv) > 3 else max(40, ntok)

    import mlopt_carriers  # noqa: F401
    import mlopt_torch  # noqa: F401
    from mlopt_cuda import CudaGraphRunner
    from mlopt_orchestrator.optimize import Optimizer
    from mlopt_torch.backend import TorchBackend

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

    inputs_s, meta = build_inputs_static(ref, L)
    P = meta["P"]
    wrap_s = _DecodeStepWrapStatic(m, ref["ehs"], m.config).eval()
    with torch.no_grad():
        o = wrap_s(*inputs_s)
        d = (o[0][:, -1].float() - ref["logits"]).abs().max().item()
        same = bool(
            (o[0][:, -1].argmax(-1) == ref["next_id"]).all().item()
        )
    print(
        f"[wrap-static] harvest step P={P} L={L}: logits "
        f"max|diff|={d:.2e} argmax_equal={same}",
        flush=True,
    )
    if not same:
        print(
            "[wrap-static] SANITY FAIL — refusing to optimize",
            flush=True,
        )
        return

    # also verify the static wrap's K/V outputs match the old wrap's
    inputs_o, _ = probe.build_inputs(ref, L)
    wrap_o = probe._DecodeStepWrap(m, ref["ehs"], m.config).eval()
    with torch.no_grad():
        oo = wrap_o(*inputs_o)
        dkv = max(
            (a.float() - b.float()).abs().max().item()
            for a, b in zip(o[1:], oo[1:], strict=True)
        )
    print(
        f"[wrap-static] knew/vnew max|diff| vs old wrap={dkv:.2e}",
        flush=True,
    )

    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    res = optz.search(wrap_s, inputs_s, max_iterations=iters)
    print(f"[search] {time.time() - t0:.1f}s", flush=True)
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None
    res.source = None
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    low = optz.lower(res, inputs_s, verify=True, runner=None)
    print(
        f"[lower] verified={low.verified.passed if low.verified else None}",
        flush=True,
    )
    mod = low.module

    # ---- term census: concats should be gone, opaque index_copy_ in ----
    from collections import Counter

    def census(term, acc):
        from mlopt_core.ir import Op

        if isinstance(term, Op):
            acc[term.op] += 1
            for a in term.args:
                census(a, acc)

    acc: Counter = Counter()
    best = getattr(res, "term", None) or getattr(res, "best_term", None)
    if best is not None:
        census(best, acc)
        print(f"[term] ops={dict(acc.most_common(20))}", flush=True)
    else:
        print(
            "[term] best_term not exposed — skipping census", flush=True
        )

    # ---- timing: eager-wrap-static vs delivered ----
    e_ms = timeit(lambda: wrap_s(*inputs_s))
    d_ms = timeit(lambda: mod(*inputs_s))
    print(
        f"[step] static-wrap {e_ms:.2f} ms  delivered {d_ms:.2f} ms",
        flush=True,
    )

    # ---- delivered inside the real greedy loop (module owns cache) --
    ck = list(ref["past"]["ck"])
    cv = list(ref["past"]["cv"])
    eager_ids = ref["gen_ids"]

    def report_loop(tag, fn, _m=m):
        steps, gids = run_static_loop(fn, _m, ehs, ntok, L, ck, cv)
        warm = sum(steps[1:])
        agree = sum(
            1 for a, b in zip(gids, eager_ids, strict=True) if a == b
        )
        first_div = next(
            (
                i
                for i, (a, b) in enumerate(
                    zip(gids, eager_ids, strict=True)
                )
                if a != b
            ),
            None,
        )
        print(
            f"[loop] {tag} {ntok} tok: warm {warm:.1f} ms "
            f"({warm / (ntok - 1):.2f} ms/tok) "
            f"last8 {sum(steps[-8:]) / 8:.2f}  "
            f"ids-match {agree}/{ntok} first-div={first_div}",
            flush=True,
        )
        print(
            f"[loop] {tag} per-step:",
            " ".join(f"{v:.1f}" for v in steps),
            flush=True,
        )

    report_loop("static-wrap", lambda *a: wrap_s(*a))
    report_loop("delivered", lambda *a: mod(*a))

    # ---- CUDA graph: copy-path vs alias path ----
    # inputs order: (hs, mask, pos, *24 past) — past positions are
    # 3..26, so alias_inputs binds i >= 3 by reference.
    try:
        low_g = optz.lower(
            res,
            inputs_s,
            verify=False,
            runner=CudaGraphRunner(),
        )
        gmod = low_g.module
        print(
            f"[graph] copy armed={low_g.stats.get('cuda_graph')} "
            f"fail={low_g.stats.get('cuda_graph_failure')}",
            flush=True,
        )
        if low_g.stats.get("cuda_graph"):
            # copy-path graph clones inputs per call — the in-graph
            # index_copy_ mutates the CLONE, so the caller buffers
            # never advance: the loop arm is meaningless here.  Time
            # the single-step call only.
            g_ms = timeit(lambda: gmod(*inputs_s))
            print(
                f"[step] delivered+graph(copy) {g_ms:.2f} ms",
                flush=True,
            )
    except Exception as e:
        print(
            f"[graph] copy lower FAILED {type(e).__name__}: {e}",
            flush=True,
        )

    try:
        # inputs order: (hs, mask, pos, sk0-5, sv0-5, ck0-5, cv0-5).
        # Positions 3..14 (self K/V) are caller-owned WRITE-BACK
        # buffers — the in-graph index_copy_ maintains the cache, so
        # they need mutable_inputs or the version-counter demotion
        # silently moves them to the copy path.  ck/cv (15..26) are
        # read-only aliases.
        dmod = CudaGraphRunner(
            multi_device=True,
            alias_inputs=lambda i, _t: i >= 3,
            mutable_inputs=lambda i, _t: 3 <= i < 15,
        ).apply(mod, inputs_s, {})
        report_loop("delivered+alias-graph", lambda *a: dmod(*a))
    except Exception as e:
        print(
            f"[graph] alias FAILED {type(e).__name__}: {e}",
            flush=True,
        )

    del m
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

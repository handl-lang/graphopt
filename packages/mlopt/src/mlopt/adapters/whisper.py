"""HF Whisper adapter — lifts one decode step or the encoder forward.

Ports the verified ``opt`` arm of
``graphopt:bench/drivers/probes/whisper_prod_bench.py`` (itself built on
``whisper_decode_loop.py`` + ``whisper_kv_static_probe.py``) for the
default ``arm="decoder"`` unit, and the encoder half of
``graphopt:bench/drivers/probes/whisper_encoder_bench.py`` for
``arm="encoder"``.  Whisper's autoregressive ``generate`` is control
flow + mutable cache state and cannot go through ``Optimizer.search``
directly; what *can* be optimized is one decode step as a pure
function — or the whole encoder forward, which is already a pure
static-shape function of ``input_features``.

``load`` therefore returns, per arm:

* ``arm="decoder"`` (default) — the probe's ``_DecodeStepWrapStatic``:
  caller-owned fixed-capacity K/V buffers, in-place ``index_copy_``
  slot writes, zero cats — plus the exact input tuple the bench fed
  ``Optimizer.search``/``lower``.
* ``arm="encoder"`` — a thin ``EncWrap`` (``WhisperEncoder`` →
  ``last_hidden_state`` tensor) over real log-mel features; the
  decoder/proj_out are dropped after the wrap is built to free VRAM
  (the encoder bench's 4 GB trick, inverted — here the encoder is
  the payload).  The encoder bench measured this lift clean (zero
  opaque leaves) with ``verify=True`` passing on fp32 models; on bf16
  large/turbo the default ruleset's ``sdpa`` folds diverge with depth
  (rel 0.418 at 32 layers —
  ``graphopt:docs/reports/turbo_bf16_delivery.md``), so bf16 encoder
  runs should pass ``--rules bf16_safe``; the verify gate enforces
  this honestly either way.

Ported faithfully:

* ``WhisperForConditionalGeneration.from_pretrained(...,
  attn_implementation="eager")`` — the encoder stays HF-eager; only the
  decoder step is lifted (per the probe design).
* ``make_features`` — real log-mel features from synthetic 8 s audio via
  ``WhisperFeatureExtractor`` constructed directly (no
  preprocessor_config download).
* ``encoder_out`` + ``get_prefix`` — encoder hidden states and the
  decoder prompt prefix captured by a one-shot ``generate`` pre-hook.
* ``hf_greedy_loop`` harvest variant — real HF suppression policy
  (``suppress_tokens`` every step, ``begin_suppress_tokens`` at the
  first generated token, EOS banned over the fixed horizon) and the
  mid-decode cache snapshot at ``(K-1) + g_harvest``.
* ``build_inputs_static`` — self K/V padded to ``(1, nh, L, hdim)``,
  mask ``(1,1,1,L)`` in model dtype allowing positions ``0..P``,
  ``pos = [P]``, then the 4N-layer ordering
  ``sk[0..n), sv[0..n), ck[0..n), cv[0..n)``.
* ``_DecodeStepWrapStatic`` + ``_KVWriteLayer``/``_KVRefLayer`` with the
  transformers-4.57 adaptations baked in (``get_max_cache_shape``,
  tuple-unpack of layer outputs, layer count from ``self.layers``).
* The bench's argmax sanity gate — the wrap must reproduce the
  harvested step's suppressed argmax before the spec is returned.
* VRAM care — after the harvest the encoder is dropped
  (``del m.model.encoder`` + ``gc`` + ``empty_cache``), the ~1.4 GB
  turbo saving that made search fit on a 4 GB card in the bench.  The
  decoder stays alive in ``meta["model"]`` for embed/bookkeeping.

Intentionally left out (bench-only machinery):

* All timing loops (``timeit``/``time_loop``/``static_loop``/reps) and
  the ms/tok reporting.
* The delivered/static/compile/CUDA-graph *loop arms* — driving the
  greedy loop is the consumer's job; the helpers it needs
  (``embed_step``, ``suppress_sets``, ``apply_suppress``,
  ``decode_step_mask``) are exported at module level.
* ``torch.compile`` wrap and ``CudaGraphRunner`` delivery — those are
  ``Optimizer.lower`` runner concerns, not lift concerns.
* The non-static ``_DecodeStepWrap`` (cat-based) — the static-K/V
  variant strictly dominates it and is what the bench verified.

``transformers`` is an optional runtime dep: ``probe`` reports its
availability without importing, and ``load`` raises ``RuntimeError``
naming the missing package rather than touching ``pyproject``.
"""

from __future__ import annotations

import gc
import importlib.util
from typing import Any

from mlopt.models import ModelSpec

_PREFIX = "whisper:"
_SR = 16000
_AUDIO_S = 8.0
_NTOK = 64
_G_HARVEST = 8

# Canonical size labels, matched against the model-id leaf.  Order
# matters: "large-v3-turbo" and "distil-large-v3" must be tested before
# bare "large".
_SIZE_KEYS = (
    "distil-large-v3",
    "large-v3-turbo",
    "large-v3",
    "large-v2",
    "turbo",
    "large",
    "medium",
    "small",
    "base",
    "tiny",
)
_SIZE_LABELS = {
    "distil-large-v3": "distil-large-v3",
    "large-v3-turbo": "turbo",
    "large-v3": "large-v3",
    "large-v2": "large-v2",
    "turbo": "turbo",
    "large": "large",
    "medium": "medium",
    "small": "small",
    "base": "base",
    "tiny": "tiny",
}
_MEL_BINS = {
    "tiny": 80,
    "base": 80,
    "small": 80,
    "medium": 80,
    "large": 128,
    "large-v2": 128,
    "large-v3": 128,
    "turbo": 128,
    "distil-large-v3": 128,
}
_BF16_SIZES = {"large", "large-v2", "large-v3", "turbo", "distil-large-v3"}


def _hf_id(ref: str) -> str:
    """Strip the ``whisper:`` prefix; bare HF ids pass through."""
    ref = ref.strip()
    if ref.startswith(_PREFIX):
        ref = ref[len(_PREFIX) :]
    if not ref:
        raise ValueError("whisper adapter: empty HF model id")
    return ref


def _size_hint(hf_id: str) -> str | None:
    leaf = hf_id.rstrip("/").rsplit("/", 1)[-1].lower()
    for key in _SIZE_KEYS:
        if key in leaf:
            return _SIZE_LABELS[key]
    return None


def _dtype_hint(hf_id: str) -> str:
    size = _size_hint(hf_id)
    return "bfloat16" if size in _BF16_SIZES else "float32"


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


# ---------------------------------------------------------------------------
# Public loop-bookkeeping helpers (ported from whisper_decode_loop.py /
# whisper_prod_bench.py) — the consumer's greedy loop needs these; they are
# eager bookkeeping, not part of the optimized step.
# ---------------------------------------------------------------------------


def embed_step(decoder: Any, ids: Any, t: int) -> Any:
    """Replicate ``WhisperDecoder`` embed+positions for one decode step."""
    import torch

    e = decoder.embed_tokens(ids)
    pos = decoder.embed_positions(
        ids,
        past_key_values_length=t,
        position_ids=torch.arange(t, t + 1, device=ids.device)
        .unsqueeze(0)
        .expand(ids.shape[0], -1),
    )
    return e + pos


def suppress_sets(model: Any) -> tuple[list[int], list[int], list[int]]:
    """HF greedy suppression sets from ``generation_config``."""
    gc_ = model.generation_config
    supp = list(getattr(gc_, "suppress_tokens", None) or [])
    begin = list(getattr(gc_, "begin_suppress_tokens", None) or [])
    eos = getattr(gc_, "eos_token_id", None)
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    return supp, begin, eos


def apply_suppress(
    logits: Any,
    supp: list[int],
    begin: list[int],
    first: bool,
    eos: list[int] | None = None,
) -> Any:
    """HF greedy policy, non-mutating: ``suppress_tokens`` every step,
    ``begin_suppress_tokens`` at the first generated token only, plus
    ``eos`` banned over a fixed horizon (equivalent to HF's
    MinLengthLogitsProcessor under ``min_new_tokens``)."""
    import torch

    logits = logits.clone()
    idx = torch.tensor(supp + (begin if first else []) + (eos or []), device=logits.device)
    if idx.numel():
        logits[:, idx] = float("-inf")
    return logits.argmax(-1, keepdim=True)


def decode_step_mask(t: int, L: int, dtype: Any, device: Any) -> Any:
    """Mask for static-wrap decode step ``t``: allow buffer positions
    ``0..t`` (slot ``t`` receives this step's K/V in place)."""
    import torch

    mask = torch.zeros(1, 1, 1, L, device=device, dtype=dtype)
    mask[..., t + 1 : L] = torch.finfo(dtype).min
    return mask


# ---------------------------------------------------------------------------
# Lazy module construction — keeps ``import mlopt.adapters.whisper`` free of
# torch/transformers so ``probe``/``claims`` stay cheap.
# ---------------------------------------------------------------------------

_CLASSES: dict[str, type] | None = None


def _classes() -> dict[str, type]:
    """Build (once) the cache-layer and wrap classes; needs torch +
    transformers at call time only."""
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES
    import torch.nn as nn
    from transformers.cache_utils import (
        CacheLayerMixin,
        DynamicCache,
        EncoderDecoderCache,
    )

    class _KVWriteLayer(CacheLayerMixin):
        """Self-attn cache layer over a caller-owned fixed buffer.

        First ``update`` adopts the ``(1,nh,L,hdim)`` input buffer by
        reference; the decode append does ``index_copy_`` at position
        ``pos`` — an in-place slot write that exports as an opaque
        mutation node (the probe-verified path)."""

        def __init__(self, pos: Any) -> None:
            super().__init__()
            self.pos = pos
            self._primed = False
            self.last_k = None
            self.last_v = None

        def lazy_initialization(self, key_states: Any, value_states: Any) -> None:
            self.is_initialized = True

        def update(self, key_states: Any, value_states: Any, *args: Any, **kwargs: Any):
            if not self._primed:
                self._primed = True
                self.keys = key_states
                self.values = value_states
                self.is_initialized = True
                return key_states, value_states
            # _primed ⇒ buffers adopted on the first update.
            assert self.keys is not None and self.values is not None  # nosec B101
            self.keys.index_copy_(2, self.pos, key_states)
            self.values.index_copy_(2, self.pos, value_states)
            self.last_k = key_states
            self.last_v = value_states
            return self.keys, self.values

        def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
            kv_length = self.keys.shape[-2] if self.keys is not None else 0
            return kv_length, 0

        def get_seq_length(self) -> int:
            if self.keys is None:
                return 0
            return int(self.keys.shape[-2])

        def get_max_length(self) -> int:
            return -1

        def get_max_cache_shape(self) -> int:
            # transformers >=4.57 CacheLayerMixin ABC (absent in 5.x).
            k = getattr(self, "keys", None)
            return int(k.shape[-2]) if k is not None else 0

        def reset(self) -> None:
            self.keys = self.values = None
            self.last_k = self.last_v = None
            self._primed = False
            self.is_initialized = False

    class _KVRefLayer(CacheLayerMixin):
        """Cross-attn cache layer: adopt the constant encoder K/V by
        reference — attention's ``is_updated`` path reads
        ``layers[i].keys`` directly, zero copy."""

        def lazy_initialization(self, key_states: Any, value_states: Any) -> None:
            self.is_initialized = True

        def update(self, key_states: Any, value_states: Any, *args: Any, **kwargs: Any):
            self.keys = key_states
            self.values = value_states
            self.is_initialized = True
            return key_states, value_states

        def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
            kv_length = self.keys.shape[-2] if self.keys is not None else 0
            return kv_length, 0

        def get_seq_length(self) -> int:
            if self.keys is None:
                return 0
            return int(self.keys.shape[-2])

        def get_max_length(self) -> int:
            return -1

        def get_max_cache_shape(self) -> int:
            k = getattr(self, "keys", None)
            return int(k.shape[-2]) if k is not None else 0

        def reset(self) -> None:
            self.keys = self.values = None
            self.is_initialized = False

    class _DecodeStepWrapStatic(nn.Module):
        """One Whisper decode step as a pure static-shape function.

        Inputs: ``hs`` (1,1,dim post-embed+pos), ``mask``
        (1,1,1,L additive, allows positions ``0..t``), ``pos`` (1,)
        int64 write position, then ``4*n_layers`` past tensors ordered
        ``sk[0..n), sv[0..n), ck[0..n), cv[0..n)`` — self K/V are the
        whole ``L``-slot caller buffers the module mutates in place,
        cross K/V are adopted by reference.

        Outputs: ``(logits, knew_0, vnew_0, ..., knew_{n-1},
        vnew_{n-1})`` — identical contract to the probe's
        ``_DecodeStepWrapStatic``.
        """

        def __init__(self, model: Any, ehs_in: Any, cfg: Any) -> None:
            super().__init__()
            dec = model.model.decoder
            self.layers = dec.layers
            self.norm = dec.layer_norm
            self.head = model.proj_out
            self.cfg = cfg
            self.register_buffer("ehs", ehs_in)

        def forward(self, hs: Any, mask: Any, pos: Any, *past: Any):
            n = len(self.layers)
            sk = past[0:n]
            sv = past[n : 2 * n]
            ck = past[2 * n : 3 * n]
            cv = past[3 * n : 4 * n]
            sc = DynamicCache()
            cc = DynamicCache()
            sc.layers = [_KVWriteLayer(pos) for _ in range(n)]
            cc.layers = [_KVRefLayer() for _ in range(n)]
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
                # transformers 4.57 returns a tuple; 5.x a bare tensor.
                h = out[0] if isinstance(out, tuple) else out
            h = self.norm(h)
            logits = self.head(h)
            outs = [logits]
            for i in range(n):
                # _KVWriteLayer attributes — not on the HF cache union,
                # so getattr is deliberate (ty can't see them either).
                outs.append(getattr(sc.layers[i], "last_k"))  # noqa: B009
                outs.append(getattr(sc.layers[i], "last_v"))  # noqa: B009
            return tuple(outs)

    class EncWrap(nn.Module):
        """``WhisperEncoder`` → ``last_hidden_state`` tensor.

        Port of ``whisper_encoder_bench.EncWrap`` — the sink needs a
        tensor, not a ``BaseModelOutput``.  The encoder forward is a
        pure static-shape function of ``input_features``; batch and
        feature dims are whatever the traced input carries."""

        def __init__(self, enc: Any) -> None:
            super().__init__()
            self.enc = enc

        def forward(self, x: Any) -> Any:
            return self.enc(x).last_hidden_state

    _CLASSES = {
        "kv_write": _KVWriteLayer,
        "kv_ref": _KVRefLayer,
        "wrap": _DecodeStepWrapStatic,
        "enc_wrap": EncWrap,
    }
    return _CLASSES


# ---------------------------------------------------------------------------
# Load-path helpers — ports of the bench's model/feature/harvest pieces.
# ---------------------------------------------------------------------------


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as e:  # pragma: no cover - torch is a declared dep
        raise RuntimeError(
            "mlopt whisper adapter requires torch (a declared mlopt dependency)"
        ) from e
    return torch


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as e:
        raise RuntimeError(
            "mlopt whisper adapter requires `transformers` — install it "
            "alongside mlopt (e.g. `uv pip install transformers`) or pick "
            "another adapter; it is intentionally not a pyproject dep"
        ) from e
    return transformers


def _default_dtype(hf_id: str, device: Any) -> Any:
    """bf16 for large/turbo ids on CUDA (the bench's policy — turbo was
    loaded bf16, base/small fp32), fp32 elsewhere."""
    torch = _import_torch()
    if device.type == "cuda" and _size_hint(hf_id) in _BF16_SIZES:
        return torch.bfloat16
    return torch.float32


def _make_features(n_mels: int, audio_s: float, device: Any) -> Any:
    """Real log-mel features from synthetic audio — port of the bench's
    ``make_features`` (``WhisperFeatureExtractor`` built directly with
    whisper defaults so no preprocessor_config download is needed)."""
    import numpy as np

    transformers = _import_transformers()
    fe = transformers.WhisperFeatureExtractor(
        feature_size=n_mels,
        sampling_rate=_SR,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    )
    rng = np.random.default_rng(0)
    t = np.arange(int(_SR * audio_s)) / _SR
    audio = (
        0.30 * np.sin(2 * np.pi * 220 * t)
        + 0.20 * np.sin(2 * np.pi * 440 * t + 0.5)
        + 0.15 * np.sin(2 * np.pi * 880 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
        + 0.05 * rng.standard_normal(t.shape)
    ).astype(np.float32)
    out = fe(audio, sampling_rate=_SR, return_tensors="pt")
    return out["input_features"].to(device)


def _encoder_out(model: Any, feats: Any) -> Any:
    torch = _import_torch()
    feats = feats.to(next(model.parameters()).dtype)
    with torch.no_grad():
        return model.model.encoder(feats).last_hidden_state


def _get_prefix(model: Any, feats: Any) -> list[int]:
    """Decoder prompt prefix = the ``input_ids`` HF feeds the decoder on
    the first ``generate`` step, captured via a one-shot pre-hook."""
    torch = _import_torch()
    box: dict[str, Any] = {}
    feats = feats.to(next(model.parameters()).dtype)

    def hook(mod: Any, args: Any, kwargs: Any) -> None:
        if "prefix" not in box:
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is not None:
                box["prefix"] = ids[0].detach().cpu().tolist()

    h = model.model.decoder.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model.generate(
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


def _greedy_loop_harvest(
    model: Any,
    ehs: Any,
    prefix: list[int],
    ntok: int,
    device: Any,
    harvest_call: int,
) -> tuple[list[int], dict[str, Any]]:
    """Port of the bench's ``hf_greedy_loop``: HF decoder greedy loop
    with prompt prefix + HF suppression; snapshots the cache at
    ``harvest_call`` (a decoder-call index) into ``harvest``."""
    torch = _import_torch()
    dec = model.model.decoder
    supp, begin, eos = suppress_sets(model)
    K = len(prefix)
    cache = None
    harvest: dict[str, Any] = {}
    call_idx = 0
    gen_ids: list[int] = []

    def snap_harvest(c: Any, P: int) -> None:
        n_layers = len(c.self_attention_cache.layers)
        sc = c.self_attention_cache
        cc = c.cross_attention_cache
        harvest["past"] = {
            "sk": [sc.layers[j].keys[..., :P, :].detach().clone() for j in range(n_layers)],
            "sv": [sc.layers[j].values[..., :P, :].detach().clone() for j in range(n_layers)],
            "ck": [cc.layers[j].keys.detach().clone() for j in range(n_layers)],
            "cv": [cc.layers[j].values.detach().clone() for j in range(n_layers)],
        }
        harvest["P"] = P
        harvest["ehs"] = ehs
        harvest["hs"] = embed_step(dec, cur, P)
        harvest["logits"] = logits.float().clone()
        harvest["next_id"] = nxt.clone()

    with torch.no_grad():
        # untimed prefill: prefix[:-1]
        for j in range(K - 1):
            ids = torch.tensor(prefix[j : j + 1], device=device).view(1, 1)
            out = dec(
                input_ids=ids,
                encoder_hidden_states=ehs,
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            call_idx += 1
        cur = torch.tensor(prefix[K - 1 : K], device=device).view(1, 1)
        for i in range(ntok):
            out = dec(
                input_ids=cur,
                encoder_hidden_states=ehs,
                past_key_values=cache,
                use_cache=True,
            )
            logits = model.proj_out(out.last_hidden_state)[:, -1]
            nxt = apply_suppress(logits, supp, begin, first=(i == 0), eos=eos)
            if call_idx == harvest_call:
                # P = self-attn cache length BEFORE this call = K-1+i
                snap_harvest(out.past_key_values, K - 1 + i)
            cache = out.past_key_values
            gen_ids.append(int(nxt[0, 0].item()))
            cur = nxt.view(1, 1)
            call_idx += 1
    return gen_ids, harvest


def _pad_self(t: Any, L: int, P: int) -> Any:
    """(1,nh,P,hdim) -> (1,nh,L,hdim) zero-padded on the right."""
    torch = _import_torch()
    nh, hdim = t.shape[1], t.shape[-1]
    out = torch.zeros(1, nh, L, hdim, device=t.device, dtype=t.dtype)
    out[:, :, :P] = t
    return out


def _build_static_inputs(harvest: dict[str, Any], L: int) -> tuple[Any, ...]:
    """Port of ``build_inputs_static``: ``(hs, mask, pos, *24-ish past)``
    with mask length ``L`` allowing positions ``0..P`` and ``pos=[P]``."""
    torch = _import_torch()
    P = harvest["P"]
    dtype = harvest["hs"].dtype
    sk = [_pad_self(t, L, P) for t in harvest["past"]["sk"]]
    sv = [_pad_self(t, L, P) for t in harvest["past"]["sv"]]
    ck = list(harvest["past"]["ck"])
    cv = list(harvest["past"]["cv"])
    mask = torch.zeros(1, 1, 1, L, device=harvest["hs"].device, dtype=dtype)
    mask[..., P + 1 : L] = torch.finfo(dtype).min
    pos = torch.tensor([P], device=harvest["hs"].device)
    return (harvest["hs"], mask, pos, *(sk + sv + ck + cv))


def _decoder_alias_positions(n_layers: int) -> tuple[list[int], list[int]]:
    """``(alias_inputs, mutable_inputs)`` positional indices for the
    decoder arm's static-input tuple — the ``CudaGraphRunner`` alias
    spec (``mlopt.deliver`` ``_alias_spec`` / ``deliver_game``).

    ``inputs`` is ``(hs, mask, pos, sk[0..n), sv[0..n), ck[0..n),
    cv[0..n))`` — positions ``0..2`` are rebuilt every decode step
    (fresh embed / mask / write-position tensor), so they stay on the
    copy path.  Positions ``3..3+4n-1`` are the caller-owned persistent
    cache buffers — they alias by reference across replays (the
    measured decode-loop win: no per-step D2D copies of the K/V
    caches).  Within them, ``sk``/``sv`` (positions ``3..3+2n-1``) are
    write-back: ``_KVWriteLayer.update`` does an in-place
    ``index_copy_`` at ``pos`` on the adopted buffer, so the recorded
    graph mutates the input's storage — they must be named
    ``mutable_inputs`` or the capture-side version-counter demotion
    silently moves them to the copy path and the caller's cache never
    sees the write.  ``ck``/``cv`` (positions ``3+2n..3+4n-1``) are
    adopted read-only by ``_KVRefLayer`` — constant after step 0, no
    in-graph mutation, so alias alone is correct.
    """
    first_past = 3
    alias = list(range(first_past, first_past + 4 * n_layers))
    mutable = list(range(first_past, first_past + 2 * n_layers))
    return alias, mutable


def _load_encoder(
    ref: str,
    hf_id: str,
    device: Any,
    dtype: Any,
    audio_s: float,
) -> ModelSpec:
    """``arm="encoder"`` — lift the whole ``WhisperEncoder`` forward.

    Port of ``whisper_encoder_bench.load_encoder`` + ``EncWrap``: keep
    ONLY the encoder on the device (decoder/proj_out/lm_head dropped),
    wrap it so the sink sees a tensor, feed it real log-mel features.
    No harvest is needed — the encoder has no cache state — so this
    path skips the decode-loop machinery entirely.
    """
    torch = _import_torch()
    transformers = _import_transformers()

    model = transformers.WhisperForConditionalGeneration.from_pretrained(  # nosec B615
        hf_id, torch_dtype=dtype, attn_implementation="eager"
    )
    model = model.to(device).eval()

    n_mels = int(getattr(model.config, "num_mel_bins", 0) or 0) or (
        _MEL_BINS.get(_size_hint(hf_id) or "", 80)
    )
    feats = _make_features(n_mels, audio_s, device).to(dtype)

    enc = model.model.encoder
    wrap = _classes()["enc_wrap"](enc).eval()

    # Sanity gate (cheap): the wrap must produce a finite
    # last_hidden_state of the expected (1, T//2, d_model) shape.
    # Equivalence itself is the optimizer's verify gate — the wrap IS
    # the HF eager encoder, so a pre-search "HF reference" would be
    # tautological; callers wanting a faithfulness number re-run
    # ``spec.module`` (still the eager encoder) post-lower.
    with torch.no_grad():
        out = wrap(feats)
    if not bool(torch.isfinite(out).all()):
        raise RuntimeError(
            f"whisper encoder arm: non-finite encoder output for {hf_id} "
            "— refusing to hand an unverified module to the optimizer"
        )
    d_model = int(getattr(model.config, "d_model", 0) or out.shape[-1])
    T = int(out.shape[-2])
    sanity = {
        "out_shape": list(out.shape),
        "out_absmax": float(out.float().abs().max().item()),
        "d_model": d_model,
        "encoder_positions": T,
    }

    # VRAM: the decoder/proj_out are dead weight once the encoder is
    # the payload — dropping them is the encoder bench's headroom
    # trick (turbo's decoder is ~0.4B params).
    dropped = []
    for attr, parent in (
        ("decoder", model.model),
        ("proj_out", model),
        ("lm_head", model),
    ):
        if hasattr(parent, attr):
            delattr(parent, attr)
            dropped.append(attr)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    meta = {
        "ref": ref,
        "hf_id": hf_id,
        "family": "whisper",
        "arm": "encoder",
        "unit": "encoder",
        "size": _size_hint(hf_id),
        "device": str(device),
        "dtype": str(dtype),
        "n_mels": n_mels,
        "audio_s": audio_s,
        "input_layout": "(input_features (B,n_mels,3000))",
        "output_layout": f"(B,{T},{d_model}) last_hidden_state",
        "encoder_params": sum(p.numel() for p in enc.parameters()),
        "sanity": sanity,
        "dropped_for_vram": dropped,
        "model": model,
        "notes": (
            "module is the whole WhisperEncoder forward over real "
            "log-mel features; spec.module stays callable as the HF "
            "eager reference for post-lower faithfulness checks.  "
            "bf16 large/turbo: pass --rules bf16_safe — the default "
            "ruleset's sdpa folds diverge with depth under bf16 "
            "(verify gate will (correctly) fail otherwise; measured "
            "rel 0.418 at 32 layers, graphopt turbo_bf16_delivery.md)"
        ),
    }
    return ModelSpec(module=wrap, inputs=(feats,), kwargs={}, meta=meta)


class _WhisperAdapter:
    name = "whisper"
    description = (
        "HF Whisper (all sizes: tiny/base/small/medium/large/v2/v3/"
        "large-v3-turbo/distil) — lifts one decoder step as a "
        "static-shape module over caller-owned K/V buffers, or the "
        "whole encoder forward with arm=encoder; encoder stays "
        "HF-eager on the decoder arm"
    )
    available = True
    unavailable_reason = ""

    def claims(self, ref: str) -> bool:
        return ref.strip().startswith(_PREFIX)

    def probe(self, ref: str) -> dict[str, Any]:
        """Cheap metadata for ``mlopt doctor`` — parses the HF id and
        reports size/dtype hints + optional-dep availability; never
        downloads weights or imports torch/transformers."""
        hf_id = _hf_id(ref)
        size = _size_hint(hf_id)
        return {
            "adapter": self.name,
            "available": True,
            "hf_id": hf_id,
            "family": "whisper",
            "size": size,
            "n_mels": _MEL_BINS.get(size) if size else None,
            "dtype_default": _dtype_hint(hf_id),
            "arms": ["decoder", "encoder"],
            "transformers": _have("transformers"),
            "torch": _have("torch"),
        }

    def load(self, ref: str, **opts: Any) -> ModelSpec:
        """Lift ``whisper:<hf-id>`` into a ``ModelSpec``.

        Returns the static decode-step wrap plus its harvested input
        tuple — the exact ``(module, inputs)`` pair the bench fed
        ``Optimizer.search``/``lower``.  Options: ``arm``
        (``"decoder"`` default — one decode step; ``"encoder"`` — the
        whole ``WhisperEncoder`` forward over real log-mel features,
        decoder dropped for VRAM), ``device`` (default
        cuda-if-available else cpu), ``dtype`` (default bf16 for
        large/turbo on cuda, else fp32), ``ntok`` (harvest decode
        horizon, default 64), ``L`` (static K/V capacity, default
        ``max(80, ntok+16)``), ``g_harvest`` (harvested decode index,
        default 8), ``audio_s`` (synthetic audio length, default 8.0 —
        the feature extractor zero-pads to the 30 s window either
        way), ``keep_encoder`` (decoder arm only; default False — the
        encoder is dropped after the harvest to save VRAM).
        """
        torch = _import_torch()
        transformers = _import_transformers()

        hf_id = _hf_id(ref)
        device_name = opts.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_name)
        dtype = opts.get("dtype") or _default_dtype(hf_id, device)
        if isinstance(dtype, str):
            # ``--adapter-opt dtype=bfloat16`` arrives as a string;
            # from_pretrained tolerates it but tensor .to() does not.
            resolved = getattr(torch, dtype, None)
            if not isinstance(resolved, torch.dtype):
                raise ValueError(
                    f"whisper adapter: unknown dtype {dtype!r} — "
                    "expected a torch.dtype name like 'float32'/'bfloat16'"
                )
            dtype = resolved
        audio_s = float(opts.get("audio_s", _AUDIO_S))
        arm = str(opts.get("arm", "decoder")).lower()
        if arm not in ("decoder", "encoder"):
            raise ValueError(
                f"whisper adapter: unknown arm {arm!r} — expected "
                "'decoder' (one decode step) or 'encoder' (the whole "
                "WhisperEncoder forward)"
            )
        if arm == "encoder":
            return _load_encoder(ref, hf_id, device, dtype, audio_s)

        ntok = int(opts.get("ntok", _NTOK))
        L = int(opts.get("L") or max(80, ntok + 16))
        g_harvest = int(opts.get("g_harvest", _G_HARVEST))
        keep_encoder = bool(opts.get("keep_encoder", False))

        model = transformers.WhisperForConditionalGeneration.from_pretrained(  # nosec B615
            hf_id, torch_dtype=dtype, attn_implementation="eager"
        )
        model = model.to(device).eval()

        n_mels = int(getattr(model.config, "num_mel_bins", 0) or 0) or (
            _MEL_BINS.get(_size_hint(hf_id) or "", 80)
        )
        feats = _make_features(n_mels, audio_s, device)
        with torch.no_grad():
            ehs = _encoder_out(model, feats)
        prefix = _get_prefix(model, feats)
        K = len(prefix)

        # Harvest a mid-decode step: call index = (K-1 prefill) + g.
        harvest_call = (K - 1) + g_harvest
        _gen_ids, harvest = _greedy_loop_harvest(model, ehs, prefix, ntok, device, harvest_call)
        if "logits" not in harvest:
            raise RuntimeError(
                f"whisper adapter: harvest at decode index {g_harvest} "
                f"never fired (prefix K={K}, ntok={ntok}) — lower "
                "g_harvest or raise ntok"
            )
        inputs = _build_static_inputs(harvest, L)

        wrap = _classes()["wrap"](model, harvest["ehs"], model.config).eval()

        # Bench sanity gate: the wrap must reproduce the harvested
        # step's suppressed argmax before the spec is returned.
        supp, begin, eos = suppress_sets(model)
        with torch.no_grad():
            out = wrap(*inputs)
            logits_diff = (out[0][:, -1].float() - harvest["logits"]).abs().max().item()
            nxt_wrap = apply_suppress(out[0][:, -1].float(), supp, begin, first=False, eos=eos)
            same = bool((nxt_wrap.reshape(-1) == harvest["next_id"].reshape(-1)).all().item())
        if not same:
            raise RuntimeError(
                f"whisper adapter: static-wrap sanity check failed for "
                f"{hf_id} (logits max|diff|={logits_diff:.2e}) — refusing "
                "to hand an unverified module to the optimizer"
            )

        # VRAM: the encoder is dead weight after ehs+prefix are fixed —
        # dropping it frees ~1.4 GB on turbo (the bench's 4 GB trick).
        encoder_dropped = False
        if not keep_encoder and hasattr(model.model, "encoder"):
            del model.model.encoder
            encoder_dropped = True
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        n_layers = len(model.model.decoder.layers)
        alias_inputs, mutable_inputs = _decoder_alias_positions(n_layers)
        # inputs layout: (hs, mask, pos, sk[n], sv[n], ck[n], cv[n])
        ck = list(inputs[3 + 2 * n_layers : 3 + 3 * n_layers])
        cv = list(inputs[3 + 3 * n_layers : 3 + 4 * n_layers])
        meta = {
            "ref": ref,
            "hf_id": hf_id,
            "family": "whisper",
            "arm": "decoder",
            "size": _size_hint(hf_id),
            "device": str(device),
            "dtype": str(dtype),
            "n_mels": n_mels,
            "n_layers": n_layers,
            "prefix": prefix,
            "prefix_len": K,
            "harvest_decode_index": g_harvest,
            "past_len": harvest["P"],
            "L": L,
            "ntok": ntok,
            "audio_s": audio_s,
            "input_layout": ("(hs, mask, pos, sk[0..n), sv[0..n), ck[0..n), cv[0..n))"),
            # CUDA-graph alias spec for the delivery runners: the 4*n
            # persistent K/V cache buffers alias by reference (no
            # per-step copies); sk/sv are write-back (in-graph
            # index_copy_ mutates them) and must be named mutable or
            # the version-counter demotion breaks the caller's cache.
            "alias_inputs": alias_inputs,
            "mutable_inputs": mutable_inputs,
            "sanity": {
                "logits_max_abs_diff": logits_diff,
                "argmax_equal": same,
            },
            "encoder_dropped": encoder_dropped,
            "model": model,
            "cross_attn_kv": {"ck": ck, "cv": cv},
            "notes": (
                "module is one decoder step; drive the greedy loop with "
                "mlopt.adapters.whisper.embed_step / decode_step_mask / "
                "apply_suppress over zero-init (1,nh,L,hdim) self-K/V "
                "buffers plus meta['cross_attn_kv'] (constant after "
                "step 0); meta['model'] keeps the decoder for "
                "embed_tokens/embed_positions bookkeeping"
            ),
        }
        return ModelSpec(module=wrap, inputs=inputs, kwargs={}, meta=meta)


ADAPTER = _WhisperAdapter()

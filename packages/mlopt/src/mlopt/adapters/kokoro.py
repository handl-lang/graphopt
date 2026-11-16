"""Kokoro TTS adapter — lifts one acoustic submodule as a static-shape module.

Ports the ``_harvest`` arms of
``graphopt:bench/drivers/probes/tts_kokoro_probe.py`` (consumed by
``kokoro_prod_bench.py`` / ``kokoro_split_probe.py`` /
``kokoro_region_fuse.py``).  ``hexgrad/Kokoro-82M`` is a
StyleTTS2-derived acoustic model: ALBERT text encoder + prosody
predictor + conv text encoder + ISTFTNet decoder with a neural
source-filter excitation.  The autoregressive/e2e entry points
(``forward_with_tokens``) are control flow + dynamic
``repeat_interleave`` lengths and cannot go through
``Optimizer.search`` directly; what *can* be optimized is one
submodule as a pure function.  ``load`` therefore returns the
probe's region wrap plus the exact input tuple the bench fed
``Optimizer.search``/``lower``.

Ported faithfully:

* The stub-package import — ``kokoro/__init__.py`` pulls
  ``kokoro.pipeline`` -> ``misaki`` (G2P, espeak, spacy — none
  needed for the acoustic forward).  A stub module with
  ``__path__`` lets the submodule files import normally while
  ``__init__.py`` is skipped entirely, exactly as the probe does.
  The source tree defaults to ``$KOKORO_SRC`` or ``/tmp/kokoro-src``
  (``git clone --depth 1 https://github.com/hexgrad/kokoro``);
  override with the ``kokoro_src`` load option.
* ``KModel(repo_id, config, model, disable_complex)`` built from
  ``config.json`` + the repo's ``.pth`` via ``huggingface_hub`` —
  the same files the probe downloads; weight filename resolved
  through ``KModel.MODEL_NAMES`` (unknown repos raise an honest
  error naming the supported ids).
* ``_input_ids`` — random-but-valid phoneme ids with the 0
  bookends, ``_voice`` — the ``voices/<voice>.pt`` style pack row.
* ``_synth_once`` — one ``forward_with_tokens`` with pre-hooks
  capturing the decoder call args and the ``F0Ntrain`` inputs.
* The region wraps — ``_DecWrap`` (decoder), ``_F0NWrap``
  (predictor.F0Ntrain, cat'd output), ``_TextEncWrap``
  (text_encoder), ``_FullWrap`` (forward_with_tokens — expected
  export wall, attempted honestly).
* The device-creator monkeypatch — ``SineGen._f02sine`` draws
  ``torch.rand/randn_like(..., device=...)`` inside the forward and
  the torch_bridge kwargs->attrs loop drops ``torch.device``, so
  unpatched the decoder arms wall at export (kokoro_tts probe
  doc).  ``load`` installs the ``*_like`` respelling by default
  (``patch_device_creators=False`` opts out); it stays installed
  for the consumer's ``Optimizer.search``/``lower`` — call
  :func:`restore_device_creators` to undo.

Intentionally left out (bench-only / consumer-side machinery):

* All timing loops (``timeit``/reps) and ms reporting.
* The e2e split orchestration (``kokoro_split_probe.py``): the
  ``forward_with_tokens`` -> ``(M1: ids,ref_s -> d,t_en,pred_dur)``
  + host-side alignment glue + ``(M2: en,asr,s,s128 -> audio)``
  decomposition, the ``W == S`` exact-mode parity finding, the
  per-W artifact cache, and the ``bucket`` contamination caveat.
  Those are consumer-side composition — the spec's ``meta`` notes
  document the seam.
* ``torch.compile`` wrap, ``CudaGraphRunner`` /
  ``RegionFusionRunner`` delivery — ``Optimizer.lower`` runner
  concerns.  Note the bench's region-fuse finding: stock
  ``output_rtol=1e-4`` declines the decoder arms (fused-vs-unfused
  max_rel ~1.4e-3); the verified override is
  ``RegionFusionRunner(output_rtol=2e-3)``.
* The ``packed`` variant — ported as the ``textenc-p`` arm
  (``mlopt.adapters._kokoro_packed``, a faithful port of the
  probe's ``tts_kokoro_packed.py``): the three pack→lstm→pad sites
  in ``kokoro/modules.py`` are wrapped as
  ``torch.library.custom_op`` leaves so ``PackedSequence`` never
  crosses the export boundary.  Measured in the bench:
  ``textenc-p`` delivered **verified** (diff 0.0) at 3.13 ms vs
  3.51 ms eager (``kokoro_production_bench.md`` /
  ``kokoro_e2e_shapes.md``).  Only ``textenc`` gets the ``-p``
  suffix — ``full`` would still wall on the dynamic
  ``repeat_interleave`` alignment lengths downstream of the LSTM
  sites, and the decoder/``f0n`` arms contain no pack→lstm→pad
  site.

Honest coverage notes:

* fp32 only in practice — the bench documents bf16 as walled by
  the cuDNN LSTM (``tts_kokoro_r1.md`` wall B); ``dtype="bf16"``
  is accepted but recorded as unverified in ``meta``.
* The decoder arms are *stochastic*: ``SineGen`` draws
  ``torch.rand``/``randn_like`` in-graph (unseeded run-to-run floor
  ~6e-2), so unseeded output-parity ``verify=True`` cannot pass —
  the bench reports seeded ``manual_seed`` parity instead.  ``meta``
  flags ``stochastic`` per arm so consumers can choose
  ``verify=False`` + seeded parity honestly.  ``f0n``/``textenc``/
  ``textenc-p`` are deterministic.
* ``misaki``/espeak/G2P are NOT required — the stub import skips
  the pipeline entirely; only the acoustic forward is lifted.  The
  full ``KPipeline`` (phoneme->audio) is out of scope.

``torch``, ``huggingface_hub``, ``transformers``, ``loguru``,
``attr`` (attrs) and ``numpy`` are needed at ``load`` time only —
all declared by the ``kokoro`` optional extra
(``uv sync --extra kokoro``); ``probe`` reports their availability
via ``importlib.util.find_spec`` without importing anything.  The
kokoro source tree is NOT a pip dep: clone
``https://github.com/hexgrad/kokoro`` to ``$KOKORO_SRC`` (default
``/tmp/kokoro-src``) or pass ``kokoro_src=`` to ``load``.
"""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import json
import os
import sys
import types
from typing import Any

from mlopt.models import ModelSpec

_PREFIX = "kokoro:"
_DEFAULT_REPO = "hexgrad/Kokoro-82M"
_DEFAULT_VOICE = "af_heart"
_N_TOKEN = 178
_TEXT_LEN = 42  # input_ids incl. the two bookend zeros
_SEED = 1234
_ARMS = ("decoder", "decoder-nc", "f0n", "textenc", "textenc-p", "full")
# Arms whose forward draws torch.rand/randn_like in-graph (SineGen).
_STOCHASTIC_ARMS = frozenset({"decoder", "decoder-nc", "full"})
_DTYPE_NAMES = {"fp32": "float32", "bf16": "bfloat16"}


def _hf_id(ref: str) -> str:
    """Strip the ``kokoro:`` prefix; bare HF ids pass through."""
    ref = ref.strip()
    if ref.startswith(_PREFIX):
        ref = ref[len(_PREFIX) :]
    if not ref:
        raise ValueError("kokoro adapter: empty HF model id")
    return ref


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _kokoro_src(opts: dict[str, Any]) -> str:
    """Source-tree resolution: ``kokoro_src`` opt > ``$KOKORO_SRC`` >
    ``/tmp/kokoro-src`` (the probe's default)."""
    return str(opts.get("kokoro_src") or os.environ.get("KOKORO_SRC") or "/tmp/kokoro-src")  # nosec B108


# ---------------------------------------------------------------------------
# Device-creator patch — ported verbatim from tts_kokoro_probe.py.  See the
# module docstring: without it the decoder arms wall at export because the
# torch_bridge kwargs->attrs loop drops ``torch.device``.  Probe-local, not
# a substrate fix — ``restore_device_creators`` undoes it.
# ---------------------------------------------------------------------------

_DEVREF: dict[str, Any] = {}
_ORIG_CREATORS: dict[str, Any] = {}


def _devref(dev: Any) -> Any:
    """A tiny fp32 tensor living on ``dev`` — the ``*_like`` reference.

    Pre-warmed in ``load`` so the ``torch.empty`` miss-branch is never
    traced: during export the dict hit returns a plain global tensor,
    which dynamo lifts as a constant (already on-device).  Uses the
    UNPATCHED ``torch.empty`` — the patched wrapper would recurse.
    """
    import torch

    key = str(dev)
    t = _DEVREF.get(key)
    if t is None:
        empty = _ORIG_CREATORS.get("empty", torch.empty)
        # 1-D size-1: ``expand`` right-aligns, so it broadcasts to any
        # target rank (a (1,1) ref cannot shrink to a 1-D target).
        t = empty(1, device=dev)
        _DEVREF[key] = t
    return t


#: creator -> the ``*_like`` spelling that carries the device on its
#: reference operand instead of a (dropped) ``device`` kwarg.
_DEVICE_CREATOR_LIKE = {
    "rand": "rand_like",
    "randn": "randn_like",
    "zeros": "zeros_like",
    "ones": "ones_like",
    "empty": "empty_like",
    "full": "full_like",
}


def _shape_dims(args: tuple) -> list[int] | None:
    """Normalise creator shape args (varargs ints or one tuple)."""
    dims: list[int] = []
    for a in args:
        if isinstance(a, (list, tuple)):
            dims.extend(int(x) for x in a)
        elif isinstance(a, int):
            dims.append(a)
        else:
            return None
    return dims


def _patch_device_creators() -> None:
    """Re-spell ``f(shape..., device=d)`` as ``f_like(devref(d).expand(...))``.

    Only plain shape+dtype kwargs are rewritten; anything else
    (``generator=``, ``layout=``, …) falls back to the original call —
    eager-correct, may still wall at export.  Idempotent; the originals
    are kept in ``_ORIG_CREATORS`` for :func:`restore_device_creators`.
    """
    import torch

    if getattr(torch, "_kokoro_devcreators_patched", False):
        return
    for name, like_name in _DEVICE_CREATOR_LIKE.items():
        orig = getattr(torch, name)
        _ORIG_CREATORS[name] = orig
        like_fn = getattr(torch, like_name)

        def make(fn: Any, lfn: Any, tag: str = name) -> Any:
            def wrapper(*a: Any, _fn: Any = fn, _lfn: Any = lfn, _tag: str = tag, **k: Any) -> Any:
                dev = k.pop("device", None)
                extra = set(k) - {"dtype", "fill_value"}
                dims: list[int] | None = None
                fill = k.get("fill_value")
                pos_fill = None
                if _tag == "full" and fill is None and a:
                    # torch.full(shape..., fill_value): the last
                    # positional scalar is the fill, the rest is shape.
                    if isinstance(a[-1], (int, float)) and not isinstance(a[-1], bool):
                        pos_fill = a[-1]
                        dims = _shape_dims(a[:-1])
                    else:
                        dims = _shape_dims(a)
                else:
                    dims = _shape_dims(a)
                if (
                    dev is not None
                    and not extra
                    and dims is not None
                    and (pos_fill is not None or fill is not None or _tag != "full")
                ):
                    dt = dev.type if hasattr(dev, "type") else str(dev)
                    if dt != "cpu":
                        ref = _devref(dev)
                        like_in = ref.expand(*dims) if dims else ref.reshape(())
                        kw = {}
                        if "dtype" in k:
                            kw["dtype"] = k["dtype"]
                        if _tag == "full":
                            return _lfn(like_in, pos_fill if pos_fill is not None else fill, **kw)
                        return _lfn(like_in, **kw)
                if dev is not None:
                    k["device"] = dev
                return _fn(*a, **k)

            return wrapper

        setattr(torch, name, make(orig, like_fn))
    torch._kokoro_devcreators_patched = True  # ty: ignore[missing-attribute]


def restore_device_creators() -> None:
    """Undo :func:`_patch_device_creators` — restores the original
    ``torch.rand``/``randn``/``zeros``/``ones``/``empty``/``full``.

    Needed before ``torch.compile`` (inductor's FakeTensorMode rejects
    the wrapper's ``ref.expand`` on a real CUDA tensor — the landmine
    ``kokoro_fusion_probe`` documents).  Search/lower must run with the
    patch installed; restore afterwards.
    """
    import torch

    for name, orig in _ORIG_CREATORS.items():
        setattr(torch, name, orig)
    torch._kokoro_devcreators_patched = False  # ty: ignore[missing-attribute]


# ---------------------------------------------------------------------------
# Lazy model construction — keeps ``import mlopt.adapters.kokoro`` free of
# torch/huggingface_hub so ``probe``/``claims`` stay cheap.
# ---------------------------------------------------------------------------


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as e:  # pragma: no cover - torch is a declared dep
        raise RuntimeError(
            "mlopt kokoro adapter requires torch (a declared mlopt dependency)"
        ) from e
    return torch


def _import_huggingface_hub() -> Any:
    try:
        import huggingface_hub
    except ImportError as e:
        raise RuntimeError(
            "mlopt kokoro adapter requires `huggingface_hub` — install the "
            "`kokoro` extra (`uv sync --extra kokoro`) or pick another "
            "adapter"
        ) from e
    return huggingface_hub


def _kokoro_model_cls(kokoro_src: str) -> Any:
    """Import ``kokoro.model.KModel`` without running ``kokoro/__init__``.

    The package ``__init__`` pulls ``kokoro.pipeline`` -> ``misaki``
    (G2P, espeak, spacy — none needed for the acoustic forward).  A
    stub module with ``__path__`` lets the submodule files import
    normally (relative imports resolve) while ``__init__.py`` is
    skipped entirely — verbatim port of the probe's trick.  If the
    consumer already imported ``kokoro`` (e.g. a real install with
    misaki present), that module is used as-is.
    """
    if "kokoro" in sys.modules:
        from kokoro.model import KModel  # ty: ignore[import-not-found]

        return KModel
    src = os.path.join(kokoro_src, "kokoro")
    if not os.path.isdir(src):
        raise RuntimeError(
            f"mlopt kokoro adapter: kokoro source tree not found at "
            f"{kokoro_src!r} (expected {src!r}).  The `kokoro` pip "
            "package's __init__ pulls the misaki G2P chain, so the "
            "adapter imports the model code from a source checkout "
            "instead — clone it with `git clone --depth 1 "
            "https://github.com/hexgrad/kokoro /tmp/kokoro-src` or "
            "pass `kokoro_src=` / set $KOKORO_SRC"
        )
    pkg = types.ModuleType("kokoro")
    pkg.__path__ = [src]
    sys.modules["kokoro"] = pkg
    from kokoro.model import KModel  # ty: ignore[import-not-found]

    return KModel


def _hf_file(repo_id: str, filename: str) -> str:
    hf = _import_huggingface_hub()
    return hf.hf_hub_download(repo_id=repo_id, filename=filename)


def _input_ids(n: int, device: Any, seed: int, n_token: int = _N_TOKEN) -> Any:
    """Random-but-valid phoneme ids with the 0 bookends the model adds."""
    torch = _import_torch()
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.randint(1, n_token, (1, n - 2), device=device, generator=g)
    return torch.cat(
        [
            torch.zeros(1, 1, dtype=torch.long, device=device),
            ids,
            torch.zeros(1, 1, dtype=torch.long, device=device),
        ],
        dim=1,
    )


def _voice(repo_id: str, voice: str, idx: int, device: Any) -> Any:
    """Voice style vector.

    ``voices/*.pt`` packs are ``(510, 1, 256)`` — the pipeline picks
    ``pack[len(ps)-1]`` (a per-length style row).  For the probe any
    row is a valid ``(1, 256)`` ``ref_s``; ``idx`` approximates the
    pipeline's length indexing (default ``ids.shape[1] - 1``, as the
    probe does).
    """
    torch = _import_torch()
    pack = torch.load(_hf_file(repo_id, f"voices/{voice}.pt"), weights_only=True)
    return pack[min(idx, pack.shape[0] - 1)].to(device)


def _synth_once(m: Any, ids: Any, ref_s: Any) -> dict[str, Any]:
    """One ``forward_with_tokens``; returns the capture dict.

    ``cap["decoder"]`` = the decoder's positional args
    ``(asr, f0, n, s)``; ``cap["f0n_x"]``/``cap["f0n_s"]`` = the
    ``F0Ntrain`` inputs; ``cap["audio"]``/``cap["pred_dur"]`` = the
    outputs.
    """
    torch = _import_torch()
    cap: dict[str, Any] = {}

    def dec_hook(mod: Any, args: Any) -> None:
        cap["decoder"] = tuple(a.detach().clone() if torch.is_tensor(a) else a for a in args)

    def f0n_cap(x: Any, s: Any) -> Any:
        cap["f0n_x"] = x.detach().clone()
        cap["f0n_s"] = s.detach().clone()
        return orig_f0n(x, s)

    orig_f0n = m.predictor.F0Ntrain
    m.predictor.F0Ntrain = f0n_cap
    h = m.decoder.register_forward_pre_hook(dec_hook)
    try:
        with torch.no_grad():
            audio, pred_dur = m.forward_with_tokens(ids, ref_s, speed=1.0)
    finally:
        h.remove()
        m.predictor.F0Ntrain = orig_f0n
    cap["audio"] = audio.detach().clone()
    cap["pred_dur"] = pred_dur.detach().clone() if pred_dur is not None else None
    return cap


# ---------------------------------------------------------------------------
# Region wraps — ported verbatim from tts_kokoro_probe.py, rebuilt lazily
# (torch stays out of module import so probe/claims stay cheap).
# ---------------------------------------------------------------------------

_WRAP_CLS: dict[str, type] | None = None


def _wrap_cls(which: str) -> type:
    """The nn.Module class for ``which`` (built once, needs torch)."""
    global _WRAP_CLS
    if _WRAP_CLS is None:
        import torch
        import torch.nn as nn

        class _DecWrap(nn.Module):
            """``decoder(asr, f0, n, s)`` -> waveform tensor."""

            def __init__(self, dec: Any) -> None:
                super().__init__()
                self.dec = dec

            def forward(self, asr: Any, f0: Any, n: Any, s: Any) -> Any:
                return self.dec(asr, f0, n, s)

        class _F0NWrap(nn.Module):
            """``F0Ntrain(x, s)`` -> (F0, N) cat'd on dim 1 (single tensor out)."""

            def __init__(self, pred: Any) -> None:
                super().__init__()
                self.pred = pred

            def forward(self, x: Any, s: Any) -> Any:
                f0, n = self.pred.F0Ntrain(x, s)
                return torch.cat([f0, n], dim=1)

        class _TextEncWrap(nn.Module):
            """``text_encoder(ids, lengths, mask)`` — packed-padded LSTM inside."""

            def __init__(self, enc: Any) -> None:
                super().__init__()
                self.enc = enc

            def forward(self, ids: Any, lengths: Any, mask: Any) -> Any:
                return self.enc(ids, lengths, mask)

        class _FullWrap(nn.Module):
            """``forward_with_tokens`` — dynamic ``repeat_interleave``
            lengths (expected export wall; attempted honestly)."""

            def __init__(self, model: Any) -> None:
                super().__init__()
                self.model = model

            def forward(self, ids: Any, ref_s: Any) -> Any:
                audio, _ = self.model.forward_with_tokens(ids, ref_s, speed=1.0)
                return audio

        _WRAP_CLS = {
            "decoder": _DecWrap,
            "decoder-nc": _DecWrap,
            "f0n": _F0NWrap,
            "textenc": _TextEncWrap,
            "full": _FullWrap,
        }
    return _WRAP_CLS[which]


def _make_wrap(which: str, m: Any) -> Any:
    if which in ("decoder", "decoder-nc"):
        return _wrap_cls(which)(m.decoder).eval()
    if which == "f0n":
        return _wrap_cls(which)(m.predictor).eval()
    if which == "textenc":
        return _wrap_cls(which)(m.text_encoder).eval()
    return _wrap_cls(which)(m).eval()


def _harvest(
    which: str,
    *,
    hf_id: str,
    device: Any,
    dtype_name: str,
    voice: str,
    voice_idx: int | None,
    text_len: int,
    seed: int,
    kokoro_src: str,
    packed: bool = False,
) -> tuple[Any, Any, tuple[Any, ...], Any, dict[str, Any]]:
    """Build (model, wrapped module, inputs, seeded-ref, capture).

    Port of the probe's ``_harvest`` minus the timing/unseeded-floor
    extras the bench measured separately — ``load`` keeps the model
    alive decision to the caller (see ``keep_model``).
    """
    torch = _import_torch()
    KModel = _kokoro_model_cls(kokoro_src)

    # r2 fix carried over: ``decoder`` is the complex TorchSTFT variant
    # (disable_complex=False), ``decoder-nc`` the CustomSTFT conv-iSTFT
    # one — the arms were swapped in r1.
    disable_complex = which == "decoder-nc"
    with open(_hf_file(hf_id, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    model_path = _hf_file(hf_id, "kokoro-v1_0.pth")
    m = KModel(
        repo_id=hf_id,
        config=config,
        model=model_path,
        disable_complex=disable_complex,
    )
    m = m.to(device).to(getattr(torch, _DTYPE_NAMES[dtype_name])).eval()
    packed_ops: dict[str, int] = {}
    if packed:
        # tts_walls_fixed.md option 1 (ported): wrap the three
        # pack→lstm→pad sites as torch.library.custom_op leaves so
        # PackedSequence never crosses the export boundary.  Install
        # AFTER the dtype/device move and BEFORE the harvest capture
        # run so every site sees the patched forward — the ops are
        # semantics-identical (pad tail re-masked downstream).
        from mlopt.adapters import _kokoro_packed

        packed_ops = _kokoro_packed.install(m)

    ids = _input_ids(text_len, device, seed)
    # The probe indexes the voice pack by ``ids.shape[1] - 1`` (the
    # pipeline's per-length style row); ``voice_idx`` overrides it.
    ref_s = _voice(hf_id, voice, voice_idx if voice_idx is not None else ids.shape[1] - 1, device)
    cap = _synth_once(m, ids, ref_s)
    cap["packed_ops"] = packed_ops

    wrap = _make_wrap(which, m)
    if which in ("decoder", "decoder-nc"):
        inp: tuple[Any, ...] = tuple(cap["decoder"])
    elif which == "f0n":
        inp = (cap["f0n_x"], cap["f0n_s"])
    elif which == "textenc":
        # Rebuild the exact call the model makes inside
        # forward_with_tokens (model.py:93-100).
        T = ids.shape[1]
        input_lengths = torch.full((ids.shape[0],), T, device=ids.device, dtype=torch.long)
        text_mask = (
            torch.arange(T, device=ids.device)
            .unsqueeze(0)
            .expand(ids.shape[0], -1)
            .type_as(input_lengths)
        )
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)
        inp = (ids, input_lengths, text_mask)
    else:  # full
        inp = (ids, ref_s)

    # Seeded eager reference of the wrapped module — the stochastic
    # decoder arms are only comparable under aligned manual_seed.
    torch.manual_seed(seed)
    with torch.no_grad():
        ref = wrap(*inp)
    return m, wrap, inp, ref, cap


class _KokoroAdapter:
    name = "kokoro"
    description = (
        "Kokoro TTS (hexgrad/Kokoro-82M family) — lifts one acoustic "
        "submodule (decoder / decoder-nc / f0n / textenc / textenc-p / "
        "full) as a "
        "static-shape module over harvested forward_with_tokens inputs; "
        "G2P/pipeline (misaki) not required"
    )
    available = True
    unavailable_reason = ""

    def claims(self, ref: str) -> bool:
        return ref.strip().startswith(_PREFIX)

    def probe(self, ref: str) -> dict[str, Any]:
        """Cheap metadata for ``mlopt doctor`` — parses the HF id and
        reports arm/dtype hints + optional-dep availability; never
        downloads weights, imports torch, or touches the network."""
        hf_id = _hf_id(ref)
        src = _kokoro_src({})
        return {
            "adapter": self.name,
            "available": True,
            "hf_id": hf_id,
            "family": "kokoro",
            "voice": _DEFAULT_VOICE,
            "arms": list(_ARMS),
            "dtype_default": "float32",
            "dtype_note": "fp32 verified; bf16 walled by cuDNN LSTM in the bench",
            "stochastic_arms": sorted(_STOCHASTIC_ARMS),
            "torch": _have("torch"),
            "huggingface_hub": _have("huggingface_hub"),
            "transformers": _have("transformers"),
            "loguru": _have("loguru"),
            "attrs": _have("attr"),
            "numpy": _have("numpy"),
            "misaki": _have("misaki"),
            "kokoro_src": src,
            "kokoro_src_present": os.path.isdir(os.path.join(src, "kokoro")),
        }

    def load(self, ref: str, **opts: Any) -> ModelSpec:
        """Lift ``kokoro:<hf-id>`` into a ``ModelSpec``.

        Returns the region wrap plus its harvested input tuple — the
        exact ``(module, inputs)`` pair the bench fed
        ``Optimizer.search``/``lower``.  Options: ``arm`` (default
        ``"decoder"``; one of ``decoder``, ``decoder-nc``, ``f0n``,
        ``textenc``, ``textenc-p``, ``full`` — ``textenc-p`` wraps
        the pack→lstm→pad sites as custom_op leaves so the arm
        exports; plain ``textenc`` walls at
        ``pack_padded_sequence``), ``device`` (default
        cuda-if-available else cpu), ``dtype`` (``"fp32"`` default;
        ``"bf16"`` accepted but unverified — cuDNN LSTM wall),
        ``voice`` (default ``"af_heart"``), ``voice_idx`` (default
        ``ids.shape[1] - 1``), ``text_len`` (default 42), ``seed``
        (default 1234), ``kokoro_src`` (source tree; default
        ``$KOKORO_SRC`` or ``/tmp/kokoro-src``),
        ``patch_device_creators`` (default True — required for the
        decoder arms to export; see module docstring),
        ``keep_model`` (default False — the whole ``KModel`` is
        dropped after harvest; the wrap keeps only its submodule, so
        the rest of the 82M params are freed).
        """
        torch = _import_torch()

        hf_id = _hf_id(ref)
        arm = str(opts.get("arm", "decoder"))
        if arm not in _ARMS:
            raise ValueError(f"kokoro adapter: unknown arm {arm!r} — expected one of {list(_ARMS)}")
        # ``textenc-p`` = the textenc arm with the pack→lstm→pad sites
        # wrapped as custom_op leaves (see _kokoro_packed / the module
        # docstring) — the only arm where the ``-p`` suffix changes
        # exportability.
        packed = arm.endswith("-p")
        which = arm[: -len("-p")] if packed else arm
        device_name = opts.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_name)
        dtype_name = str(opts.get("dtype", "fp32"))
        if dtype_name not in _DTYPE_NAMES:
            raise ValueError(
                f"kokoro adapter: unknown dtype {dtype_name!r} — "
                f"expected one of {sorted(_DTYPE_NAMES)}"
            )
        voice = str(opts.get("voice", _DEFAULT_VOICE))
        text_len = int(opts.get("text_len", _TEXT_LEN))
        seed = int(opts.get("seed", _SEED))
        src = _kokoro_src(opts)

        # Preflight the kokoro.model import chain (loguru/attr/
        # transformers/numpy/huggingface_hub) so a missing dep names
        # itself instead of surfacing as a raw ImportError from inside
        # the source tree.
        missing = [
            m
            for m in ("loguru", "attr", "transformers", "numpy", "huggingface_hub")
            if not _have(m)
        ]
        if missing:
            raise RuntimeError(
                f"mlopt kokoro adapter: missing required packages {missing} "
                "— install the `kokoro` extra (`uv sync --extra kokoro` "
                "or `uv pip install 'mlopt[kokoro]'`) or pick another "
                "adapter; misaki/espeak are NOT needed (the G2P pipeline "
                "is stubbed out — only the acoustic model is lifted)"
            )

        # Weight filename is repo-specific (KModel.MODEL_NAMES) — check
        # BEFORE downloading config.json so unknown repos fail cheap.
        KModel_probe = _kokoro_model_cls(src)
        if hf_id not in getattr(KModel_probe, "MODEL_NAMES", {}):
            known = sorted(getattr(KModel_probe, "MODEL_NAMES", {}) or {})
            raise RuntimeError(
                f"kokoro adapter: no known weight file for repo {hf_id!r} "
                f"(KModel.MODEL_NAMES: {known}) — the acoustic config/"
                "weight schema is Kokoro-release-specific; pass a "
                "supported hexgrad repo id"
            )

        if bool(opts.get("patch_device_creators", True)):
            _patch_device_creators()
            # Pre-warm the device refs so the torch.empty miss-branch
            # inside _devref is never traced (see its docstring) —
            # "cuda" and "cuda:0" are distinct _devref keys and the
            # model code may spell either.
            warm = {str(device)}
            if device.type == "cuda":
                warm.update({"cuda", "cuda:0"})
            for d in warm:
                with contextlib.suppress(Exception):  # unallocatable devs skip
                    _devref(torch.device(d))

        voice_idx = opts.get("voice_idx")
        m, wrap, inp, _ref, cap = _harvest(
            which,
            hf_id=hf_id,
            device=device,
            dtype_name=dtype_name,
            voice=voice,
            voice_idx=int(voice_idx) if voice_idx is not None else None,
            text_len=text_len,
            seed=seed,
            kokoro_src=src,
            packed=packed,
        )

        # Drop the KModel: the wrap holds only its submodule
        # (m.decoder / m.predictor / m.text_encoder), so deleting m
        # frees bert/bert_encoder/text_encoder/decoder weights the arm
        # doesn't need — the shared-GPU VRAM win the bench relied on.
        model_kept = bool(opts.get("keep_model", False))
        model_obj = m if model_kept else None
        if not model_kept:
            del m
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        input_shapes = [tuple(x.shape) if torch.is_tensor(x) else type(x).__name__ for x in inp]
        meta = {
            "ref": ref,
            "hf_id": hf_id,
            "family": "kokoro",
            "arm": arm,
            "device": str(device),
            "dtype": dtype_name,
            "dtype_verified": dtype_name == "fp32",
            "voice": voice,
            "voice_idx": int(voice_idx) if voice_idx is not None else "ids.shape[1]-1",
            "text_len": text_len,
            "seed": seed,
            "kokoro_src": src,
            "stochastic": arm in _STOCHASTIC_ARMS,
            "packed": packed,
            "input_shapes": input_shapes,
            "input_layout": _input_layout(which),
            "patch_device_creators": bool(opts.get("patch_device_creators", True)),
            "keep_model": model_kept,
            "model": model_obj,
            "decoder_capture": _capture_meta(cap),
            "notes": _notes(arm, dtype_name, packed=packed),
        }
        return ModelSpec(module=wrap, inputs=inp, kwargs={}, meta=meta)


def _input_layout(arm: str) -> str:
    return {
        "decoder": "decoder(asr, f0, N, s) -> waveform",
        "decoder-nc": "decoder(asr, f0, N, s) -> waveform (CustomSTFT, disable_complex=True)",
        "f0n": "predictor.F0Ntrain(x, s) -> cat(F0, N) on dim 1",
        "textenc": "text_encoder(ids, input_lengths, text_mask)",
        "full": "forward_with_tokens(ids, ref_s) -> audio (expected export wall)",
    }[arm]


def _capture_meta(cap: dict[str, Any]) -> dict[str, Any]:
    import torch

    def _shape(t: Any) -> Any:
        return tuple(t.shape) if torch.is_tensor(t) else None

    out: dict[str, Any] = {}
    if "decoder" in cap:
        out["decoder_args"] = [_shape(a) for a in cap["decoder"]]
    if "f0n_x" in cap:
        out["f0n_x"] = _shape(cap["f0n_x"])
        out["f0n_s"] = _shape(cap["f0n_s"])
    if cap.get("audio") is not None:
        out["audio"] = _shape(cap["audio"])
    if cap.get("pred_dur") is not None:
        out["pred_dur_sum"] = int(cap["pred_dur"].sum().item())
    if cap.get("packed_ops"):
        out["packed_ops"] = dict(cap["packed_ops"])
    return out


def _notes(arm: str, dtype_name: str, *, packed: bool = False) -> str:
    base = (
        "module is one lifted submodule; drive the consumer loop "
        "yourself — the e2e split orchestration (M1 ids/ref_s -> "
        "d,t_en,pred_dur + host glue + M2 en/asr/s/s128 -> audio, "
        "per-W artifact caching, W==S exact-mode parity) lives "
        "consumer-side per kokoro_split_probe.py / "
        "docs/reports/kokoro_e2e_split.md"
    )
    if packed:
        base += (
            "; pack→lstm→pad sites wrapped as mlopt_kokoro_packed "
            "custom_op leaves (PackedSequence never crosses export; "
            "pad tail total_length=T, re-masked downstream — "
            "bench: verified, diff 0.0)"
        )
    if arm in _STOCHASTIC_ARMS:
        base += (
            "; arm is STOCHASTIC (SineGen draws rand/randn_like "
            "in-graph) — unseeded verify=True cannot pass; lower with "
            "verify=False and compare under aligned torch.manual_seed"
        )
    if arm in ("decoder", "decoder-nc"):
        base += (
            "; region-fuse delivery needs "
            "RegionFusionRunner(output_rtol=2e-3) — stock 1e-4 "
            "declines (fused-vs-unfused max_rel ~1.4e-3)"
        )
    if dtype_name != "fp32":
        base += "; bf16 is unverified — bench documents a cuDNN LSTM wall"
    return base


ADAPTER = _KokoroAdapter()

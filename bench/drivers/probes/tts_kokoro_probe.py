"""Kokoro-82M TTS probe — "lift a TTS model" measurement driver.

Sibling of ``t5_probe.py`` / ``whisper_probe.py``: same pipeline
(eager / torch.compile / mlopt search+lower / ``CudaGraphRunner``)
on a *third* real model family — a text-to-speech acoustic model,
``hexgrad/Kokoro-82M`` (StyleTTS2-derived: ALBERT text encoder +
prosody predictor + conv text encoder + ISTFTNet decoder with a
neural source-filter excitation).  4 GB GPU (RTX 2050), bf16.

The ``kokoro`` pip package is NOT installed (its ``misaki`` G2P dep
chain is heavy and irrelevant — we only need the torch acoustic
model).  The probe loads the model code straight from a shallow
clone of ``github.com/hexgrad/kokoro`` via a stub package module
(so ``kokoro/__init__.py`` — which imports the G2P pipeline — never
runs)::

    git clone --depth 1 https://github.com/hexgrad/kokoro /tmp/kokoro-src

Weights/config/voice come from HF ``hexgrad/Kokoro-82M``
(``kokoro-v1_0.pth``, ``config.json``, ``voices/af_heart.pt``) via
``huggingface_hub`` — all cached under ``~/.cache/huggingface``.
Override the source tree with ``KOKORO_SRC``.

Usage::

    .venv/bin/python bench/drivers/probes/tts_kokoro_probe.py bench [dtype]
    .venv/bin/python bench/drivers/probes/tts_kokoro_probe.py synth [dtype]
    .venv/bin/python bench/drivers/probes/tts_kokoro_probe.py opt <which> [dtype] [max_enodes] [cpuparams]

``which``: decoder | decoder-nc | f0n | textenc | full
  - ``decoder``    — the ISTFTNet vocoder, ``disable_complex=False``
                     (TorchSTFT: complex ``torch.stft``/``istft``).
  - ``decoder-nc`` — same weights, ``disable_complex=True``
                     (CustomSTFT: conv1d/conv_transpose1d iSTFT).
  - ``f0n``        — ``predictor.F0Ntrain`` (shared LSTM + AdaIN
                     resblocks -> F0/N curves).
  - ``textenc``    — ``model.text_encoder`` (emb + conv stack +
                     packed-padded LSTM).
  - ``full``       — ``forward_with_tokens`` end-to-end (control
                     flow + dynamic ``repeat_interleave`` lengths —
                     expected export wall; attempted honestly).

Honest-wall notes (see docs/reports/tts_kokoro_r1.md):

* The decoder's ``SineGen`` source module draws ``torch.rand`` /
  ``randn_like`` *inside* the forward — synthesis is stochastic, so
  unseeded output-parity verify cannot pass; the probe measures
  parity under aligned ``torch.manual_seed`` instead.
* ``torch.stft``/``torch.istft`` have no IR binding — they lift as
  ``opaque`` passthrough leaves at best; complex dtypes may wall
  earlier in export.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import sys
import time
import types
from typing import Any

import torch
import torch.nn as nn

KOKORO_SRC = os.environ.get("KOKORO_SRC", "/tmp/kokoro-src")
HF_REPO = "hexgrad/Kokoro-82M"
VOICE = "af_heart"
N_TOKEN = 178
TEXT_LEN = 42  # input_ids incl. the two bookend zeros
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
SEED = 1234


def _hf_file(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_REPO, filename=filename)


def _kokoro_model_cls():
    """Import ``kokoro.model.KModel`` without running ``kokoro/__init__``.

    The package ``__init__`` pulls ``kokoro.pipeline`` -> ``misaki``
    (G2P, espeak, spacy — none needed for the acoustic forward).  A
    stub module with ``__path__`` lets the submodule files import
    normally (relative imports resolve) while ``__init__.py`` is
    skipped entirely.
    """
    if "kokoro" not in sys.modules:
        pkg = types.ModuleType("kokoro")
        pkg.__path__ = [os.path.join(KOKORO_SRC, "kokoro")]  # ty: ignore[unresolved-attribute]
        sys.modules["kokoro"] = pkg
    from kokoro.model import KModel  # ty: ignore[unresolved-import]

    return KModel


_DEVREF: dict[str, torch.Tensor] = {}
_ORIG_CREATORS: dict[str, Any] = {}


def _devref(dev) -> torch.Tensor:
    """A tiny fp32 tensor living on ``dev`` — the ``*_like`` reference.

    Pre-warmed in ``load()`` so the ``torch.empty`` miss-branch is
    never traced: during export the dict hit returns a plain global
    tensor, which dynamo lifts as a constant (already on-device).
    Uses the UNPATCHED ``torch.empty`` — the patched wrapper would
    recurse back into here.
    """
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
    """Work around the device-kwarg export wall (see tts_kokoro_r1.md).

    ``torch.rand/zeros/ones/full/empty/randn(..., device=d)`` loses
    ``device`` at export: the kwargs->attrs loop in torch_bridge only
    records int/float/bool/list/str/dtype, so ``torch.device`` is
    silently dropped, and the creator bindings mint on CPU — the
    delivered module then dies ``Expected all tensors to be on the
    same device`` at the first add (reproduced minimally: any
    ``torch.rand(n, device=x.device)`` inside a lifted forward).  The
    ``to``/``_to_copy`` binding drops device the same way, so a
    ``.to(dev)` re-spelling does NOT work either (measured).

    Kokoro's ``SineGen._f02sine`` does exactly this
    (``torch.rand(..., device=f0_values.device)``), so BOTH decoder
    variants wall here.  The patch re-spells ``f(shape..., device=d)``
    as ``f_like(devref(d).expand(shape...))``: the reference carries
    the device as a real operand.  ``zeros_like``/``ones_like``/
    ``full_like`` are bound (device follows the reference);
    ``rand_like``/``randn_like``/``empty_like`` are unbound and lift
    as ``opaque`` passthroughs — which replay the real torch op
    verbatim, device included.  Only plain shape+dtype kwargs are
    rewritten; anything else (``generator=``, ``layout=``, …) falls
    back to the original call — eager-correct, may still wall at
    export.  Probe-local, not a fix.
    """
    if getattr(torch, "_tts_probe_devcreators_patched", False):
        return
    for name, like_name in _DEVICE_CREATOR_LIKE.items():
        orig = getattr(torch, name)
        _ORIG_CREATORS[name] = orig
        like_fn = getattr(torch, like_name)

        def make(fn, lfn, tag=name):
            def wrapper(*a, _fn=fn, _lfn=lfn, _tag=tag, **k):
                dev = k.pop("device", None)
                extra = set(k) - {"dtype", "fill_value"}
                dims: list[int] | None = None
                fill = k.get("fill_value")
                pos_fill = None
                if _tag == "full" and fill is None and a:
                    # torch.full(shape..., fill_value): the last
                    # positional scalar is the fill, the rest is shape.
                    if isinstance(
                        a[-1], (int, float)
                    ) and not isinstance(a[-1], bool):
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
                    and (
                        pos_fill is not None
                        or fill is not None
                        or _tag != "full"
                    )
                ):
                    dt = dev.type if hasattr(dev, "type") else str(dev)
                    if dt != "cpu":
                        ref = _devref(dev)
                        like_in = (
                            ref.expand(*dims)
                            if dims
                            else ref.reshape(())
                        )
                        kw = {}
                        if "dtype" in k:
                            kw["dtype"] = k["dtype"]
                        if _tag == "full":
                            return _lfn(
                                like_in,
                                pos_fill
                                if pos_fill is not None
                                else fill,
                                **kw,
                            )
                        return _lfn(like_in, **kw)
                if dev is not None:
                    k["device"] = dev
                return _fn(*a, **k)

            return wrapper

        setattr(torch, name, make(orig, like_fn))
    torch._tts_probe_devcreators_patched = True  # ty: ignore[unresolved-attribute]


def load(dtype: str = "fp32", disable_complex: bool = False):
    _patch_device_creators()
    # Pre-warm the device references so the ``torch.empty`` miss-branch
    # inside ``_devref`` is never traced (see its docstring).
    for d in ("cuda", "cuda:0"):
        with contextlib.suppress(Exception):  # CPU-only envs just skip
            _devref(d)
    KModel = _kokoro_model_cls()
    with open(_hf_file("config.json"), encoding="utf-8") as f:
        config = json.load(f)
    m = KModel(
        repo_id=HF_REPO,
        config=config,
        model=_hf_file("kokoro-v1_0.pth"),
        disable_complex=disable_complex,
    )
    return m.to("cuda").to(DTYPES[dtype]).eval()


def _voice(idx: int = 0) -> torch.Tensor:
    """Voice style vector.

    ``voices/*.pt`` packs are ``(510, 1, 256)`` — the pipeline picks
    ``pack[len(ps)-1]`` (a per-length style row).  For the probe any
    row is a valid ``(1, 256)`` ``ref_s``; ``idx`` approximates the
    pipeline's length indexing.
    """
    pack = torch.load(_hf_file(f"voices/{VOICE}.pt"), weights_only=True)
    return pack[min(idx, pack.shape[0] - 1)].to("cuda")


def _input_ids(n: int = TEXT_LEN, seed: int = 0) -> torch.Tensor:
    """Random-but-valid phoneme ids with the 0 bookends the model adds."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(
        1, N_TOKEN, (1, n - 2), device="cuda", generator=g
    )
    return torch.cat(
        [
            torch.zeros(1, 1, dtype=torch.long, device="cuda"),
            ids,
            torch.zeros(1, 1, dtype=torch.long, device="cuda"),
        ],
        dim=1,
    )


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


def _synth_once(m, ids, ref_s):
    """One ``forward_with_tokens``; returns (audio, harvest dict)."""
    cap: dict = {}

    def dec_hook(mod, args):
        cap["decoder"] = tuple(
            a.detach().clone() if torch.is_tensor(a) else a
            for a in args
        )

    def f0n_cap(x, s):
        cap["f0n_x"] = x.detach().clone()
        cap["f0n_s"] = s.detach().clone()
        return orig_f0n(x, s)

    orig_f0n = m.predictor.F0Ntrain
    m.predictor.F0Ntrain = f0n_cap
    h = m.decoder.register_forward_pre_hook(dec_hook)
    try:
        with torch.no_grad():
            audio, pred_dur = m.forward_with_tokens(
                ids, ref_s, speed=1.0
            )
    finally:
        h.remove()
        m.predictor.F0Ntrain = orig_f0n
    cap["audio"] = audio.detach().clone()
    cap["pred_dur"] = (
        pred_dur.detach().clone() if pred_dur is not None else None
    )
    return cap


class _DecWrap(nn.Module):
    """``decoder(asr, f0, n, s)`` -> waveform tensor (already a Tensor)."""

    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, asr, f0, n, s):
        return self.dec(asr, f0, n, s)


class _F0NWrap(nn.Module):
    """``F0Ntrain(x, s)`` -> (F0, N) cat'd on dim 1 (single tensor out)."""

    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self, x, s):
        f0, n = self.pred.F0Ntrain(x, s)
        return torch.cat([f0, n], dim=1)


class _TextEncWrap(nn.Module):
    """``text_encoder(ids, lengths, mask)`` — packed-padded LSTM inside."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, ids, lengths, mask):
        return self.enc(ids, lengths, mask)


class _FullWrap(nn.Module):
    """``forward_with_tokens`` — dynamic ``repeat_interleave`` lengths."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, ref_s):
        audio, _ = self.model.forward_with_tokens(ids, ref_s, speed=1.0)
        return audio


def _harvest(which: str, dtype: str, packed: bool = False):
    """Build (wrapped module, inputs, eager-output) for ``which``."""
    # r2 fix: the arms were swapped in r1 — ``decoder`` is the complex
    # TorchSTFT variant (disable_complex=False), ``decoder-nc`` the
    # CustomSTFT conv-iSTFT one.  r1's delivered-0.89x "decoder" numbers
    # were actually CustomSTFT; its "decoder-nc" unary-mul wall was the
    # complex graph (``phase * 1j`` lives in TorchSTFT.inverse).
    disable_complex = which == "decoder-nc"
    m = load(dtype, disable_complex=disable_complex)
    if packed:
        # tts_walls_fixed.md option 1: wrap the three pack→lstm→pad
        # sites as torch.library.custom_op leaves so PackedSequence
        # never crosses the export boundary (see tts_kokoro_packed.py).
        import tts_kokoro_packed

        ops = tts_kokoro_packed.install(m)
        print(
            f"[{which}/{dtype}] packed-lstm ops installed: "
            f"{sorted(ops)}",
            flush=True,
        )
    ids = _input_ids()
    ref_s = _voice(ids.shape[1] - 1)
    cap = _synth_once(m, ids, ref_s)
    if which in ("decoder", "decoder-nc"):
        asr, f0, n, s = cap["decoder"]
        wrap = _DecWrap(m.decoder).eval()
        inp = (asr, f0, n, s)
    elif which == "f0n":
        wrap = _F0NWrap(m.predictor).eval()
        inp = (cap["f0n_x"], cap["f0n_s"])
    elif which == "textenc":
        # Rebuild the exact call the model makes inside forward_with_tokens.
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
        wrap = _TextEncWrap(m.text_encoder).eval()
        inp = (ids, input_lengths, text_mask)
    elif which == "full":
        wrap = _FullWrap(m).eval()
        inp = (ids, ref_s)
    else:
        raise SystemExit(f"unknown which {which}")
    # eager reference output of the *wrapped* module, computed once
    # under a fixed seed so the stochastic source module is comparable.
    torch.manual_seed(SEED)
    with torch.no_grad():
        ref = wrap(*inp)
    # also an UNSEEDED eager pair — the stochastic floor of the arm.
    with torch.no_grad():
        ref_u1 = wrap(*inp)
        ref_u2 = wrap(*inp)
    return m, wrap, inp, ref, ref_u1, ref_u2, cap


def _maxdiff(a, b) -> float:
    if isinstance(a, tuple):
        return max(_maxdiff(x, y) for x, y in zip(a, b, strict=True))
    return (a.float() - b.float()).abs().max().item()


def bench() -> None:
    dtype = sys.argv[2] if len(sys.argv) > 2 else "fp32"
    for dc in (False, True):
        tag = "complex" if not dc else "no-complex"
        m = load(dtype, disable_complex=not dc)
        dec = m.decoder
        # harvest decoder inputs once (shapes fixed for timing)
        ids = _input_ids()
        ref_s = _voice(ids.shape[1] - 1)
        cap = _synth_once(m, ids, ref_s)
        asr, f0, n, s = cap["decoder"]

        def dec_call(dec=dec, asr=asr, f0=f0, n=n, s=s):
            return dec(asr, f0, n, s)

        print(
            f"[{tag}/{dtype}] decoder-eager   {timeit(dec_call):8.2f} ms",
            flush=True,
        )
        # stochasticity: two unseeded eager calls
        with torch.no_grad():
            o1 = dec_call()
            o2 = dec_call()
        print(
            f"[{tag}/{dtype}] decoder eager run-to-run max|diff| "
            f"{_maxdiff(o1, o2):.4e}",
            flush=True,
        )
        try:
            cdec = torch.compile(dec, dynamic=False)
            with torch.no_grad():
                cdec(asr, f0, n, s)
                torch.cuda.synchronize()

            def comp_call(cdec=cdec, asr=asr, f0=f0, n=n, s=s):
                return cdec(asr, f0, n, s)

            print(
                f"[{tag}/{dtype}] decoder-compile "
                f"{timeit(comp_call):8.2f} ms",
                flush=True,
            )
        except Exception as e:
            print(
                f"[{tag}/{dtype}] decoder-compile FAILED: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
        del m, dec
        gc.collect()
        torch.cuda.empty_cache()


def synth() -> None:
    """Full-model context: one synthesis + waveform stats."""
    dtype = sys.argv[2] if len(sys.argv) > 2 else "fp32"
    m = load(dtype, disable_complex=True)
    ids = _input_ids()
    ref_s = _voice(ids.shape[1] - 1)
    t0 = time.perf_counter()
    with torch.no_grad():
        audio, pred_dur = m.forward_with_tokens(ids, ref_s, speed=1.0)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    print(
        f"[synth/{dtype}] forward_with_tokens {ms:.1f} ms  "
        f"audio {tuple(audio.shape)} dtype={audio.dtype} "
        f"min={audio.min().item():.3f} max={audio.max().item():.3f} "
        f"pred_dur sum={int(pred_dur.sum()) if pred_dur is not None else '?'}",
        flush=True,
    )
    # run-to-run stochasticity of the FULL acoustic model
    with torch.no_grad():
        a1, _ = m.forward_with_tokens(ids, ref_s, speed=1.0)
        a2, _ = m.forward_with_tokens(ids, ref_s, speed=1.0)
    print(
        f"[synth/{dtype}] full-model run-to-run max|diff| "
        f"{_maxdiff(a1, a2):.4e}",
        flush=True,
    )


def opt(which: str, iters: int = 3) -> None:
    from kit_cuda import CudaGraphRunner
    from mlopt_orchestrator.optimize import Optimizer
    from kit_torch.backend import TorchBackend

    packed = which.endswith("-p")
    if packed:
        which = which[: -len("-p")]
    dtype = sys.argv[3] if len(sys.argv) > 3 else "fp32"
    max_enodes = int(sys.argv[4]) if len(sys.argv) > 4 else 100_000
    m, wrap, inp, ref, ref_u1, ref_u2, cap = _harvest(
        which, dtype, packed=packed
    )
    print(
        f"[{which}/{dtype}] inputs: "
        + ", ".join(
            f"{tuple(x.shape) if torch.is_tensor(x) else type(x).__name__}"
            for x in (inp if isinstance(inp, tuple) else (inp,))
        ),
        flush=True,
    )
    if torch.is_tensor(ref):
        print(
            f"[{which}/{dtype}] ref out {tuple(ref.shape)} {ref.dtype}",
            flush=True,
        )
    print(
        f"[{which}/{dtype}] decoder-in shapes: "
        f"asr={tuple(cap['decoder'][0].shape)} f0={tuple(cap['decoder'][1].shape)} "
        f"N={tuple(cap['decoder'][2].shape)} s={tuple(cap['decoder'][3].shape)}"
        if "decoder" in cap
        else f"[{which}/{dtype}] (no decoder capture)",
        flush=True,
    )

    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    try:
        res = optz.search(
            wrap,
            inp,
            max_iterations=iters,
            max_enodes=max_enodes,
        )
        print(
            f"[{which}/{dtype}] search {time.time() - t0:.1f}s "
            f"enodes={getattr(res, 'n_enodes', '?')}",
            flush=True,
        )
    except Exception as e:
        print(
            f"[{which}/{dtype}] search FAILED after {time.time() - t0:.1f}s: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return
    print(
        f"[{which}/{dtype}] eager stochastic floor "
        f"(unseeded run-to-run) max|diff| {_maxdiff(ref_u1, ref_u2):.4e}",
        flush=True,
    )
    del m
    # ``wrap`` stays alive: it shares parameters with the delivered
    # module's ``param_values`` and is needed for the seeded
    # eager-vs-delivered parity pair below.
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None
    res.source = None
    if len(sys.argv) > 5 and sys.argv[5] == "cpuparams":
        res.param_values = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in res.param_values.items()
        }
    gc.collect()
    torch.cuda.empty_cache()

    def call(mm, i):
        return mm(*i) if isinstance(i, tuple) else mm(i)

    t0 = time.time()
    low = None
    try:
        low = optz.lower(res, inp, verify=True, runner=None)
        ok = low.verified.passed if low.verified else None
        print(
            f"[{which}/{dtype}] lower+verify {time.time() - t0:.1f}s "
            f"verified={ok}",
            flush=True,
        )
    except Exception as e:
        print(
            f"[{which}/{dtype}] lower+verify FAILED after "
            f"{time.time() - t0:.1f}s: {type(e).__name__}: {e}",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
    if low is None:
        t0 = time.time()
        try:
            low = optz.lower(res, inp, verify=False, runner=None)
            print(
                f"[{which}/{dtype}] lower(verify=False) {time.time() - t0:.1f}s",
                flush=True,
            )
        except Exception as e:
            print(
                f"[{which}/{dtype}] lower(verify=False) FAILED: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            return

    # --- honest parity: seeded eager vs seeded delivered ---
    # The decoder's SineGen draws torch.rand/randn_like inside the
    # forward, so UNSEEDED outputs differ run-to-run even for eager.
    # Aligning manual_seed before each call isolates semantic drift:
    # identical RNG op sequences -> near-bitwise agreement.
    try:
        torch.manual_seed(SEED)
        with torch.no_grad():
            o_eager = wrap(*inp)
        torch.manual_seed(SEED)
        with torch.no_grad():
            o_del = call(low.module, inp)
        d = _maxdiff(o_del, o_eager)
        print(
            f"[{which}/{dtype}] delivered-vs-eager max|diff| {d:.4e} "
            f"(both under manual_seed({SEED}))",
            flush=True,
        )
        # unseeded delivered run-to-run = stochastic floor of delivery
        with torch.no_grad():
            d_u1 = call(low.module, inp)
            d_u2 = call(low.module, inp)
        print(
            f"[{which}/{dtype}] delivered unseeded run-to-run "
            f"max|diff| {_maxdiff(d_u1, d_u2):.4e}",
            flush=True,
        )
    except Exception as e:
        print(
            f"[{which}/{dtype}] delivered eval FAILED: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

    try:
        print(
            f"[{which}/{dtype}] delivered "
            f"{timeit(lambda: call(low.module, inp)):.2f} ms",
            flush=True,
        )
    except Exception as e:
        print(
            f"[{which}/{dtype}] delivered timing FAILED: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

    del wrap
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    try:
        low2 = optz.lower(
            res, inp, verify=False, runner=CudaGraphRunner()
        )
        print(
            f"[{which}/{dtype}] graph "
            f"cuda_graph={low2.stats.get('cuda_graph')} "
            f"fail={low2.stats.get('cuda_graph_failure')}",
            flush=True,
        )
        if low2.stats.get("cuda_graph"):
            print(
                f"[{which}/{dtype}] delivered+graph "
                f"{timeit(lambda: call(low2.module, inp)):.2f} ms",
                flush=True,
            )
    except Exception as e:
        print(
            f"[{which}/{dtype}] graph lower FAILED: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "bench":
        bench()
    elif sys.argv[1] == "synth":
        synth()
    elif sys.argv[1] == "opt":
        opt(sys.argv[2] if len(sys.argv) > 2 else "decoder-nc")
    else:
        raise SystemExit(f"unknown mode {sys.argv[1]}")


if __name__ == "__main__":
    main()

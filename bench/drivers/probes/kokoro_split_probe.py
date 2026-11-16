"""Kokoro e2e split-delivery probe — the two-wall workaround, measured.

Sibling of ``kokoro_prod_bench.py`` / ``kokoro_region_fuse.py``.
Implements ``docs/reports/kokoro_e2e_shapes.md`` option A (and B as
the parity-clean degenerate): ``forward_with_tokens`` is split at the
``pred_dur`` boundary into

* **M1** (T-static): ``input_ids, ref_s -> (d, t_en, pred_dur)`` —
  bert -> bert_encoder -> DurationEncoder -> predictor.lstm ->
  duration_proj, plus ``model.text_encoder``.  Wall A is respelled:
  ``input_lengths`` is provably ``full((B,), T)`` (model.py:93-98
  hardcodes it), so ``text_mask`` is built from ``arange(T)`` with T
  an input dim — identical values, static shape.
* **glue** (host-side, eager — the compositional-fallback precedent):
  ``S = pred_dur.sum()``, ``W`` = exact ``S`` (mode ``exact``, option
  B) or ``ceil16(S)`` (mode ``bucket``, option A); the ``(T, W)``
  alignment is built as ``one_hot(searchsorted(cumsum(dur), w))`` —
  the broadcast-mask reformulation that needs no ``repeat_interleave``
  binding — and ``en = dᵀ @ aln`` / ``asr = t_en @ aln`` are two small
  matmuls.
* **M2** (W-static per artifact): ``(en, asr, s, s128) -> audio`` —
  ``F0Ntrain(en, s)`` then ``decoder(asr, F0, N, s128)``.  Keyed by
  ``W`` only (T never enters), so one artifact serves every
  utterance landing in the bucket.

Parity findings baked into the measurement (see
``docs/reports/kokoro_e2e_split.md``):

* ``W == S`` is **bit-exact** vs ``forward_with_tokens`` (seeded
  max|diff| = 0.0) — the split itself is semantics-preserving.
* ``W > S`` is NOT: the decoder/F0Ntrain conv stack has a
  utterance-spanning receptive field, so even ONE pad column
  perturbs F0/N/audio globally (measured: W=S+1 -> waveform
  max|diff| ~0.36, log1p-STFT mean|d| ~0.028 vs the model's own
  unseeded floor ~0.007).  ``bucket`` mode therefore reports the
  contamination honestly; ``exact`` mode is the parity-clean
  delivery (per-S artifact, LRU-cacheable — option B of the spec).
* The decoder's SineGen draws ``rand``/``randn_like`` with
  width-dependent shapes, so seeded waveform parity across
  different W is impossible by construction; parity is reported
  both raw and under a deterministic-noise patch (probe-local
  ``rand_like``/``randn_like`` -> ``zeros_like``) that isolates
  semantic+contamination error from RNG-stream desync.

Usage::

    .venv/bin/python bench/drivers/probes/kokoro_split_probe.py check
    .venv/bin/python bench/drivers/probes/kokoro_split_probe.py arm m2 64 [iters]
    .venv/bin/python bench/drivers/probes/kokoro_split_probe.py arm m1 [iters]
    .venv/bin/python bench/drivers/probes/kokoro_split_probe.py e2e [mode] [iters]
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


kokoro = _load("tts_kokoro_probe", "tts_kokoro_probe.py")
SEED = kokoro.SEED


# ---------------------------------------------------------------------------
#  The split
# ---------------------------------------------------------------------------


class _M1Wrap(nn.Module):
    """``forward_with_tokens`` through ``pred_dur`` (+ ``t_en``).

    Returns ``(d, t_en, pred_dur)`` — a tensor tuple; the IR packs
    multi-outputs via ``pack`` (torch_export._graph_root).
    """

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids, ref_s):
        m = self.m
        T = ids.shape[-1]
        input_lengths = torch.full(
            (ids.shape[0],), T, device=ids.device, dtype=torch.long
        )
        # Wall-A respell: arange over the STATIC input dim, not
        # input_lengths.max() (model.py:100).  Values identical —
        # input_lengths is hardcoded full((B,), T) at model.py:93-98.
        text_mask = (
            torch.arange(T, device=ids.device)
            .unsqueeze(0)
            .expand(ids.shape[0], -1)
            .type_as(input_lengths)
        )
        text_mask = torch.gt(
            text_mask + 1, input_lengths.unsqueeze(1)
        ).to(m.device)
        bert_dur = m.bert(ids, attention_mask=(~text_mask).int())
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = m.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = m.predictor.lstm(d)
        duration = m.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / 1.0
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        t_en = m.text_encoder(ids, input_lengths, text_mask)
        return d, t_en, pred_dur


class _M2Wrap(nn.Module):
    """``(en, asr, s, s128) -> decoder audio`` — the W-static tail.

    ``en``/``asr`` are the glue's ``dᵀ@aln`` / ``t_en@aln`` products at
    width ``W``; everything inside is shape-static per artifact.
    NOTE: measured to defeat the RegionFusionRunner output gate
    (fused-vs-unfused max_rel ~7e-2 > 2e-3) — the e2e path uses the
    three-module decomposition instead (F0Ntrain stays eager glue:
    its delivered arm measured NO faster than eager, 7.33 vs 7.24 ms).
    """

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, en, asr, s, s128):
        F0, N = self.m.predictor.F0Ntrain(en, s)
        return self.m.decoder(asr, F0, N, s128)


class _DecWrap(nn.Module):
    """``decoder(asr, f0, n, s128) -> audio`` — the proven decoder arm.

    Identical signature to ``tts_kokoro_probe._DecWrap``; F0/N are
    computed host-side by ``F0Ntrain`` (eager — delivery measured no
    faster).  This is the composition ``kokoro_prod_bench`` already
    proved fuses (81.5 ms delivered+regionfuse at S≈127).
    """

    def __init__(self, m):
        super().__init__()
        self.dec = m.decoder

    def forward(self, asr, f0, n, s128):
        return self.dec(asr, f0, n, s128)


# ---------------------------------------------------------------------------
#  Glue
# ---------------------------------------------------------------------------


def bucket_W(S: int, mode: str) -> int:
    if mode == "exact":
        return S
    if mode == "bucket":
        return ((S + 15) // 16) * 16
    raise ValueError(f"unknown mode {mode}")


def build_alignment(pred_dur: torch.Tensor, W: int, pad: str = "edge"):
    """``(1, T, W)`` 0/1 alignment from ``pred_dur`` — host-side glue.

    Column ``j`` belongs to token ``i`` iff ``cum[i-1] <= j < cum[i]``
    (``cum = cumsum(pred_dur)``).  ``pad`` selects the pad-column
    policy for ``j >= S``: ``edge`` replicates the last token's
    column (clamped ``searchsorted``), ``zero`` emits an all-zero
    column.  ``W == S`` makes both identical — bit-exact vs the
    model's ``repeat_interleave`` construction (verified).
    """
    T = pred_dur.shape[0]
    dev = pred_dur.device
    cum = torch.cumsum(pred_dur.long(), 0)
    S = int(cum[-1])
    w = torch.arange(W, device=dev)
    idx = torch.searchsorted(cum, w, right=True)
    if pad == "zero":
        valid = idx < T
        idx = idx.clamp(max=T - 1)
        aln = F.one_hot(idx, T).to(torch.float32)
        aln = aln * valid.unsqueeze(-1).to(torch.float32)
        aln = aln.T.unsqueeze(0)
    elif pad == "edge":
        idx = idx.clamp(max=T - 1)
        aln = F.one_hot(idx, T).to(torch.float32).T.unsqueeze(0)
    else:
        raise ValueError(f"unknown pad {pad}")
    return aln.contiguous(), S


def glue(d, t_en, pred_dur, ref_s, W: int, pad: str = "edge"):
    """M1 outputs -> M2 inputs.  Returns ``(en, asr, s, s128, S)``."""
    aln, S = build_alignment(pred_dur, W, pad=pad)
    en = d.transpose(-1, -2) @ aln
    asr = t_en @ aln
    return en, asr, ref_s[:, 128:], ref_s[:, :128], S


# ---------------------------------------------------------------------------
#  Parity metrics
# ---------------------------------------------------------------------------


def _maxdiff(a, b) -> float:
    if isinstance(a, tuple):
        return max(
            _maxdiff(x, y) for x, y in zip(a, b, strict=True)
        )
    return (a.float() - b.float()).abs().max().item()


def spec_dist(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """STFT-domain distance — the honest metric for a stochastic TTS.

    Returns ``(mean|log1p|STFT| diff, mean relative STFT-magnitude
    diff|)`` on the overlapping prefix, fp32 CPU.
    """
    a = a.reshape(-1).float().cpu()
    b = b.reshape(-1).float().cpu()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    A = torch.stft(a, 512, 256, return_complex=True).abs()
    B = torch.stft(b, 512, 256, return_complex=True).abs()
    ld = (torch.log1p(A) - torch.log1p(B)).abs().mean().item()
    rd = ((A - B).abs() / (A + B + 1e-6)).mean().item()
    return ld, rd


class _DetNoise:
    """Probe-local deterministic-noise patch for parity isolation.

    ``SineGen`` draws ``torch.rand``/``torch.randn_like`` with
    width-dependent shapes, so two calls at different ``W`` consume
    different RNG streams even under ``manual_seed`` — seeded
    waveform parity across buckets is impossible *by construction*.
    Zeroing the draws (both sides patched identically) removes the
    stochastic channel and exposes the semantic + pad-contamination
    error alone.  Context manager; restores on exit.
    """

    def __enter__(self):
        self._rl = torch.rand_like
        self._rnl = torch.randn_like
        torch.rand_like = lambda t, *a, **k: torch.zeros_like(t)  # ty: ignore[invalid-assignment]
        torch.randn_like = lambda t, *a, **k: torch.zeros_like(t)  # ty: ignore[invalid-assignment]
        return self

    def __exit__(self, *exc):
        torch.rand_like = self._rl  # ty: ignore[invalid-assignment]
        torch.randn_like = self._rnl  # ty: ignore[invalid-assignment]
        return False


# ---------------------------------------------------------------------------
#  Utterances — real phoneme strings through ``KModel.vocab``
# ---------------------------------------------------------------------------

PHONEME_STRINGS = {
    "hello": "hˈɛloʊ wɜːld",
    "second": "ðə kwɪk bɹˈaʊn fˈɑks dʒˈʌmps ˈoʊvɚ ðə lˈeɪzi dˈɔg",
}


def utterance_ids(m, name: str = "hello") -> torch.Tensor:
    """Vocab-mapped phoneme ids with the 0 bookends (model.forward).

    ``bench`` is the prod_bench e2e input verbatim — random valid ids
    at ``TEXT_LEN=42`` — for direct comparison with the 150.6 ms
    ``forward_with_tokens`` baseline in ``kokoro_production_bench.md``.
    """
    if name == "bench":
        return kokoro._input_ids()
    s = PHONEME_STRINGS[name]
    ids = [m.vocab[p] for p in s if p in m.vocab]
    return torch.LongTensor([[0, *ids, 0]]).to(m.device)


def _call(mm, inp):
    return mm(*inp) if isinstance(inp, tuple) else mm(inp)


def _seeded_out(mm, inp):
    torch.manual_seed(SEED)
    with torch.no_grad():
        return _call(mm, inp)


def _restore_creators() -> None:
    for name, orig in kokoro._ORIG_CREATORS.items():
        setattr(torch, name, orig)
    torch._tts_probe_devcreators_patched = False  # ty: ignore[unresolved-attribute]


# ---------------------------------------------------------------------------
#  check — eager split correctness + stage timing
# ---------------------------------------------------------------------------


def check() -> None:
    m = kokoro.load("fp32", disable_complex=False)
    m.eval()
    ids = utterance_ids(m, "hello")
    ref_s = kokoro._voice(ids.shape[1] - 1)
    T = ids.shape[1]
    print(f"[check] T={T}", flush=True)

    def t(name, fn, reps=20):
        return kokoro.timeit(fn, warm=3, reps=reps)

    with torch.no_grad():
        # reference
        torch.manual_seed(SEED)
        a_ref, pd_ref = m.forward_with_tokens(ids, ref_s, speed=1.0)
        S = int(pd_ref.sum())
        print(f"[check] eager S={S} audio={tuple(a_ref.shape)}", flush=True)

        m1 = _M1Wrap(m).eval()
        inp1 = (ids, ref_s)
        torch.manual_seed(SEED)
        d, t_en, pd = m1(*inp1)
        print(
            f"[check] M1 out d={tuple(d.shape)} t_en={tuple(t_en.shape)} "
            f"pred_dur={tuple(pd.shape)} {pd.dtype} "
            f"equal={torch.equal(pd.cpu(), pd_ref.cpu())}",
            flush=True,
        )
        s, s128 = ref_s[:, 128:], ref_s[:, :128]
        W = bucket_W(S, "bucket")
        for mode in ("exact", "bucket"):
            WW = bucket_W(S, mode)
            en, asr, s_, s128_, S_ = glue(d, t_en, pd, ref_s, WW)
            m2 = _M2Wrap(m).eval()
            torch.manual_seed(SEED)
            au = m2(en, asr, s_, s128_)
            a = au.reshape(-1)[: S * 600]
            ar = a_ref.reshape(-1)[: S * 600]
            ld, rd = spec_dist(a, ar)
            print(
                f"[check] mode={mode} W={WW} seeded max|diff| "
                f"{(a - ar).abs().max().item():.4e} spec=({ld:.5f},{rd:.5f})",
                flush=True,
            )
        # deterministic-noise: exposes semantic+contamination only
        with _DetNoise():
            torch.manual_seed(SEED)
            a_det, _ = m.forward_with_tokens(ids, ref_s, speed=1.0)
            ar = a_det.reshape(-1)[: S * 600]
            for mode in ("exact", "bucket"):
                WW = bucket_W(S, mode)
                en, asr, s_, s128_, _ = glue(d, t_en, pd, ref_s, WW)
                torch.manual_seed(SEED)
                au = m2(en, asr, s_, s128_)
                a = au.reshape(-1)[: S * 600]
                ld, rd = spec_dist(a, ar)
                print(
                    f"[check] det-noise mode={mode} W={WW} max|diff| "
                    f"{(a - ar).abs().max().item():.4e} spec=({ld:.5f},{rd:.5f})",
                    flush=True,
                )
            # contamination vs pad amount
            for extra in (1, 2, 4, 8, 15, 16, 32):
                WW = S + extra
                en, asr, s_, s128_, _ = glue(d, t_en, pd, ref_s, WW)
                torch.manual_seed(SEED)
                au = m2(en, asr, s_, s128_)
                a = au.reshape(-1)[: S * 600]
                ld, rd = spec_dist(a, ar)
                print(
                    f"[check] det-noise W=S+{extra:2d} max|diff| "
                    f"{(a - ar).abs().max().item():.4e} spec=({ld:.5f},{rd:.5f})",
                    flush=True,
                )
        # unseeded stochastic floor (spec metric)
        with torch.no_grad():
            u1, _ = m.forward_with_tokens(ids, ref_s, speed=1.0)
            u2, _ = m.forward_with_tokens(ids, ref_s, speed=1.0)
        ld, rd = spec_dist(u1, u2)
        print(
            f"[check] eager unseeded floor max|diff| "
            f"{(u1 - u2).abs().max().item():.4e} spec=({ld:.5f},{rd:.5f})",
            flush=True,
        )

    # ---- stage timing (eager halves + glue) ----
    m1e = _M1Wrap(m).eval()
    m1_ms = t("M1", lambda: m1e(ids, ref_s))
    d, t_en, pd = m1e(ids, ref_s)
    W = bucket_W(int(pd.sum()), "bucket")

    def g():
        return glue(d, t_en, pd, ref_s, W)

    en, asr, s_, s128_, S_ = g()
    g_ms = t("glue", g)
    m2e = _M2Wrap(m).eval()
    inp2 = (en, asr, s_, s128_)
    m2_ms = t("M2", lambda: m2e(*inp2))
    full_ms = t("eager-full", lambda: m.forward_with_tokens(ids, ref_s, 1.0))
    print(
        f"[check] TIMING M1={m1_ms:.2f} glue={g_ms:.2f} M2={m2_ms:.2f} "
        f"(W={W})  eager-full={full_ms:.2f}  "
        f"split-sum={m1_ms + g_ms + m2_ms:.2f}",
        flush=True,
    )
    del m
    import gc

    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
#  arm — search + lower one half
# ---------------------------------------------------------------------------


def _deliver(which: str, inp, wrap, iters: int, fuse: bool):
    """search -> lower -> (optional RegionFusionRunner lower)."""
    from mlopt_orchestrator.optimize import Optimizer
    from kit_torch.backend import TorchBackend

    optz = Optimizer(backend=TorchBackend())
    import os

    search_kw = {}
    if os.environ.get("KPS_NOCAUSAL") == "1":
        search_kw["specialize_causal"] = False
    if os.environ.get("KPS_RULES"):
        search_kw["rules"] = os.environ["KPS_RULES"]
    if os.environ.get("KPS_EXCLUDE_RULES"):
        # Comma-separated law names to drop from DEFAULT — law-level
        # bisect for the M2 miscompile (kokoro_e2e_split.md).
        from mlopt_core.laws import DEFAULT as _DEF

        drop = set(os.environ["KPS_EXCLUDE_RULES"].split(","))
        search_kw["rules"] = [r for r in _DEF if r.name not in drop]
    if os.environ.get("KPS_NOCARRIERS") == "1":
        # Probe-local opt-out: the carrier lifts (applyd/aff_diag
        # affine-diagonal offers) run unconditionally inside
        # ``_pairing_and_lifts`` and were measured to miscompile the
        # M2 term (see kokoro_e2e_split.md).  Nulling the registry
        # makes ``get_carriers()`` return None for this process.
        import mlopt_orchestrator.carriers as _car

        _car._MACHINERY = None
    t0 = time.time()
    res = optz.search(
        wrap, inp, max_iterations=iters, max_enodes=100_000, **search_kw
    )
    print(
        f"[{which}] search {time.time() - t0:.1f}s "
        f"enodes={getattr(res, 'n_enodes', '?')}",
        flush=True,
    )
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None  # ty: ignore[invalid-assignment]
    res.source = None  # ty: ignore[invalid-assignment]
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    # M1/M2 are deterministic (no in-graph RNG) — try verify=True
    # honestly; on failure (the decoder arm's opaque STFT leaves have
    # never passed the verify gate — same wall prod_bench documents)
    # fall back to verify=False and rely on the seeded parity pair.
    # KPS_NOVERIFY=1 skips the doomed first build entirely — matters
    # on the shared 4 GB card where two module builds OOM.
    mod = None
    if os.environ.get("KPS_NOVERIFY") == "1":
        t0 = time.time()
        low = optz.lower(res, inp, verify=False, runner=None)
        print(
            f"[{which}] lower(verify=False, KPS_NOVERIFY) "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )
        mod = low.module
    else:
        t0 = time.time()
        try:
            low = optz.lower(res, inp, verify=True, runner=None)
            ok = low.verified.passed if low.verified else None
            print(
                f"[{which}] lower+verify {time.time() - t0:.1f}s verified={ok}",
                flush=True,
            )
            mod = low.module
        except Exception as e:
            print(
                f"[{which}] lower+verify FAILED {time.time() - t0:.1f}s "
                f"{type(e).__name__}: {str(e)[:160]} — retry verify=False",
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
            t0 = time.time()
            low = optz.lower(res, inp, verify=False, runner=None)
            print(
                f"[{which}] lower(verify=False) {time.time() - t0:.1f}s",
                flush=True,
            )
            mod = low.module
    if fuse:
        try:
            from kit_torch.runners import RegionFusionRunner

            _restore_creators()
            t0 = time.time()
            low_f = optz.lower(
                res, inp, verify=False,
                runner=RegionFusionRunner(output_rtol=2e-3),
            )
            rf = low_f.stats.get("region_fuse", {})
            print(
                f"[{which}] regionfuse {time.time() - t0:.1f}s "
                f"applied={rf.get('applied')} reason={rf.get('reason')} "
                f"regions={rf.get('regions_fused')} ops={rf.get('ops_fused')} "
                f"kernels={rf.get('distinct_kernels')}",
                flush=True,
            )
            fv = rf.get("fused_vs_unfused") or {}
            if fv:
                print(
                    f"[{which}]   fused-vs-unfused max_abs={fv.get('max_abs'):.3e} "
                    f"max_rel={fv.get('max_rel'):.3e} passed={fv.get('passed')}",
                    flush=True,
                )
            if rf.get("applied"):
                mod = low_f.module
                del low_f
        except Exception as e:
            print(
                f"[{which}] regionfuse FAILED {type(e).__name__}: "
                f"{str(e)[:200]}",
                flush=True,
            )
    return mod, res


def arm(which: str, W: int | None, iters: int = 3, fuse: bool = True, utt: str = "hello") -> None:
    m = kokoro.load("fp32", disable_complex=False)
    m.eval()
    packed = __import__("tts_kokoro_packed")
    ops = packed.install(m)
    print(f"[{which}] packed ops: {sorted(ops)}", flush=True)
    ids = utterance_ids(m, utt)
    ref_s = kokoro._voice(ids.shape[1] - 1)

    with torch.no_grad():
        m1e = _M1Wrap(m).eval()
        d, t_en, pd = m1e(ids, ref_s)
        S = int(pd.sum())
        print(f"[{which}] T={ids.shape[1]} S={S}", flush=True)
        if which == "m1":
            wrap = m1e
            inp = (ids, ref_s)
            torch.manual_seed(SEED)
            ref = _call(wrap, inp)
        elif which == "m2":
            WW = W if W is not None else bucket_W(S, "bucket")
            assert WW >= S, f"W={WW} < S={S}"
            en, asr, s_, s128_, _ = glue(d, t_en, pd, ref_s, WW)
            wrap = _M2Wrap(m).eval()
            inp = (en, asr, s_, s128_)
            torch.manual_seed(SEED)
            ref = _call(wrap, inp)
            print(f"[{which}] M2 W={WW}", flush=True)
        else:
            raise SystemExit(f"unknown arm {which}")

    e_ms = kokoro.timeit(lambda: _call(wrap, inp))
    print(f"[{which}] EAGER wrap {e_ms:.2f} ms", flush=True)

    mod, res = _deliver(which, inp, wrap, iters, fuse)

    # parity vs eager wrap (deterministic halves — unseeded fine, but
    # seed anyway for reproducibility)
    torch.manual_seed(SEED)
    with torch.no_grad():
        o_del = _call(mod, inp)
    dd = _maxdiff(o_del, ref)
    print(
        f"[{which}] DELIVERED-vs-eager max|diff| {dd:.3e}", flush=True
    )
    d_ms = kokoro.timeit(lambda: _call(mod, inp))
    print(f"[{which}] DELIVERED {d_ms:.2f} ms", flush=True)

    # torch.compile reference on the wrap
    _restore_creators()
    try:
        cwrap = torch.compile(wrap, dynamic=False)
        _call(cwrap, inp)
        torch.cuda.synchronize()
        c_ms = kokoro.timeit(lambda: _call(cwrap, inp))
        print(f"[{which}] COMPILE wrap {c_ms:.2f} ms", flush=True)
    except Exception as e:
        print(
            f"[{which}] COMPILE FAILED {type(e).__name__}: {str(e)[:200]}",
            flush=True,
        )

    del m, wrap, mod, res
    import gc

    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
#  e2e — the full split path, bucket cache, honest timing
# ---------------------------------------------------------------------------


def e2e(mode: str = "exact", iters: int = 3, fuse: bool = True, utt: str = "bench") -> None:
    m = kokoro.load("fp32", disable_complex=False)
    m.eval()
    packed = __import__("tts_kokoro_packed")
    packed.install(m)
    ids = utterance_ids(m, utt)
    ref_s = kokoro._voice(ids.shape[1] - 1)

    # ---- eager baseline ----
    eager_ms = kokoro.timeit(
        lambda: m.forward_with_tokens(ids, ref_s, speed=1.0),
        warm=3, reps=10,
    )
    with torch.no_grad():
        torch.manual_seed(SEED)
        a_eager, pd_eager = m.forward_with_tokens(ids, ref_s, speed=1.0)
    S = int(pd_eager.sum())
    W = bucket_W(S, mode)
    print(
        f"[e2e/{mode}] T={ids.shape[1]} S={S} W={W} "
        f"eager-full={eager_ms:.1f} ms", flush=True,
    )

    # ---- M1: eager by default ----
    # M1 delivery measured NO win (19.68 vs 21.37 ms at T=42; 13.01 vs
    # 12.82 at T=14 — launch-bound like the f0n arm) AND is fragile:
    # the T=42 term crashes carrier shape-inference under default
    # rules (``TensorType dim must be >= 0`` inside xcarrier._xshape)
    # and needed ``transpose_noop`` excluded at T=14.  KPS_M1DELIVER=1
    # re-enables the attempt for completeness.
    m1e = _M1Wrap(m).eval()
    inp1 = (ids, ref_s)
    m1_mod = None
    m1_ms_eager = kokoro.timeit(lambda: m1e(*inp1), warm=3, reps=10)
    print(f"[e2e] M1 eager {m1_ms_eager:.2f} ms", flush=True)
    if os.environ.get("KPS_M1DELIVER") == "1":
        try:
            m1_mod, _res1 = _deliver("m1", inp1, m1e, iters, fuse=False)
            m1_ms = kokoro.timeit(lambda: _deliver_call(m1_mod, inp1))
            print(f"[e2e] M1 delivered {m1_ms:.2f} ms", flush=True)
        except Exception as e:
            print(
                f"[e2e] M1 delivery FAILED {type(e).__name__}: {str(e)[:200]}",
                flush=True,
            )
            import gc as _gc_m1

            _gc_m1.collect()
            torch.cuda.empty_cache()
            m1_mod = None

    # ---- M1 once -> glue inputs ----
    m1call = m1_mod if m1_mod is not None else m1e
    with torch.no_grad():
        torch.manual_seed(SEED)
        d, t_en, pd = _call(m1call, inp1)
    assert torch.equal(pd.cpu(), pd_eager.cpu()), "M1 pred_dur mismatch"

    # ---- decoder artifact per W; F0Ntrain stays eager glue ----
    # Three-module decomposition: the bundled M2 (F0Ntrain+decoder in
    # one search) measured NO fusion win — RegionFusionRunner's output
    # gate declined it (fused-vs-unfused max_rel ~7e-2) and the plain
    # delivered module ran ~eager.  The decoder ALONE is the proven
    # arm (81.5 ms delivered+regionfuse); F0Ntrain's delivered arm
    # measured no faster than eager (7.33 vs 7.24 ms — launch-bound),
    # so it runs host-side.
    cache: dict[int, object] = {}
    build_s = 0.0

    def dec_for(w):
        nonlocal build_s
        if w in cache:
            return cache[w]
        en, asr, s_, s128_, _ = glue(d, t_en, pd, ref_s, w)
        F0, N = m.predictor.F0Ntrain(en, s_)
        wrap2 = _DecWrap(m).eval()
        inp2 = (asr, F0, N, s128_)
        t0 = time.time()
        mod2, _res2 = _deliver(f"dec-W{w}", inp2, wrap2, iters, fuse=fuse)
        build_s += time.time() - t0
        cache[w] = mod2
        return mod2

    # cold call (includes the build)
    t0 = time.perf_counter()
    mod2 = dec_for(W)
    cold_total = (time.perf_counter() - t0) * 1e3
    print(
        f"[e2e] dec-W{W} cold (build+call) {cold_total:.0f} ms "
        f"(build {build_s:.1f}s)", flush=True,
    )

    # ---- timed e2e, warm cache ----
    def split_call():
        with torch.no_grad():
            dd, tt, pp = _call(m1call, inp1)
            en, asr, s_, s128_, SS = glue(dd, tt, pp, ref_s, W)
            F0, N = m.predictor.F0Ntrain(en, s_)
            # The delivered decoder is shape-static in W only; asr/F0/N
            # values vary per call but shapes are fixed.
            au = mod2(asr, F0, N, s128_)
        return au

    # warm + parity
    with torch.no_grad():
        torch.manual_seed(SEED)
        au = split_call()
    a = au.reshape(-1)[: S * 600]
    ar = a_eager.reshape(-1)[: S * 600]
    ld, rd = spec_dist(a, ar)
    print(
        f"[e2e] split-vs-eager seeded max|diff| {(a - ar).abs().max().item():.4e} "
        f"spec=({ld:.5f},{rd:.5f})",
        flush=True,
    )
    # stochastic floor of the split path (unseeded, same W)
    with torch.no_grad():
        s1 = split_call()
        s2 = split_call()
    ld2, rd2 = spec_dist(s1.reshape(-1)[: S * 600], s2.reshape(-1)[: S * 600])
    print(
        f"[e2e] split unseeded run-to-run max|diff| "
        f"{(s1.reshape(-1)[:S*600] - s2.reshape(-1)[:S*600]).abs().max().item():.4e} "
        f"spec=({ld2:.5f},{rd2:.5f})",
        flush=True,
    )

    split_ms = kokoro.timeit(split_call, warm=3, reps=10)
    print(
        f"[e2e/{mode}] SPLIT total {split_ms:.1f} ms  "
        f"vs eager {eager_ms:.1f} ms  speedup {eager_ms / split_ms:.2f}x",
        flush=True,
    )
    # torch.compile(forward_with_tokens) reference, honest caveat:
    # inductor graphs the WHOLE thing incl. the data-dependent wall
    # via a graph break / fallback — measured 95.4 ms previously with
    # unverified parity (max|diff| 1.15e-1).
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
        print(f"[e2e] torch.compile(fwt) {c_ms:.1f} ms", flush=True)
    except Exception as e:
        print(
            f"[e2e] torch.compile(fwt) FAILED {type(e).__name__}: "
            f"{str(e)[:200]}",
            flush=True,
        )

    del m, m1e, m1_mod, cache
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def _deliver_call(mod, inp):
    return _call(mod, inp)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == "check":
        check()
    elif cmd == "arm":
        which = sys.argv[2] if len(sys.argv) > 2 else "m2"
        W = int(sys.argv[3]) if len(sys.argv) > 3 else None
        iters = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        utt = sys.argv[5] if len(sys.argv) > 5 else "hello"
        arm(which, W, iters, utt=utt)
    elif cmd == "e2e":
        mode = sys.argv[2] if len(sys.argv) > 2 else "exact"
        iters = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        utt = sys.argv[4] if len(sys.argv) > 4 else "bench"
        e2e(mode, iters, utt=utt)
    else:
        raise SystemExit(f"unknown mode {cmd}")


if __name__ == "__main__":
    main()

# Kokoro-82M TTS probe — round 1 (third model family: text-to-speech)

Follow-up to `t5_r1.md` / `t5_walls_fixed.md` / `whisper_experiment_r2.md`.
Same pipeline (eager / torch.compile / mlopt search+lower /
`CudaGraphRunner`), a *third* unrelated real model: **Kokoro-82M**
(`hexgrad/Kokoro-82M`, ~82 M params, StyleTTS2-derived acoustic
model — ALBERT text encoder + prosody predictor + conv text encoder +
ISTFTNet neural-vocoder decoder with a harmonic+noise source
module).  Question: does a TTS model go through lift → search →
lower → verify → deliver, and where do TTS's nonstandard ops
(`istft`, conv1d stacks, in-graph RNG, packed LSTMs) wall?

Hardware/software: RTX 2050 4 GB (shared), torch 2.14.0+cu130,
transformers 5.23.0.  **fp32 only** — see wall B (bf16 unusable:
cuDNN LSTM rejects bf16 weights).

## Setup (deps, honestly)

- `kokoro` pip package NOT installed — its `misaki` G2P chain
  (espeak/spacy) is irrelevant: we only need the torch acoustic
  forward (`phoneme_ids → waveform`), not G2P.
- Model code: shallow clone `git clone --depth 1
  https://github.com/hexgrad/kokoro /tmp/kokoro-src`; the probe
  imports `kokoro.model.KModel` through a stub package module so
  `kokoro/__init__.py` (which pulls the G2P pipeline) never runs.
  `KOKORO_SRC` env overrides the path.
- Deps installed into `.venv` with `--no-deps`: **`loguru`, `attr`**
  (tiny, pure-python; `attr` was already present via attrs).
- Weights/config/voice via `huggingface_hub` → HF cache:
  `kokoro-v1_0.pth` (327 MB), `config.json`, `voices/af_heart.pt`
  (`(510,1,256)` — the pipeline indexes `pack[len(ps)-1]`; the probe
  approximates with the input-length row).
- Input: 42 random valid phoneme ids + bookend zeros, `af_heart`
  voice, `forward_with_tokens` → 76 200-sample waveform (~3.2 s
  @ 24 kHz), first-call 1381 ms.

Probe: `bench/drivers/probes/tts_kokoro_probe.py` —
`bench|synth|opt <which> [dtype] [max_enodes] [cpuparams]`,
`which ∈ decoder | decoder-nc | f0n | textenc | full`.
`decoder` = `disable_complex=False` (TorchSTFT: complex
`torch.stft`/`istft`); `decoder-nc` = `disable_complex=True`
(CustomSTFT: conv1d/conv_transpose1d iSTFT — the "no complex ops"
variant).  Decoder inputs are *harvested* from a real
`forward_with_tokens` run via forward hooks (fixed shapes):
`asr (1,512,127)`, `F0/N (1,254)`, `style s (1,128)`.

## Baselines (fp32, warmup 3 / reps 20)

| arm | eager | torch.compile |
|---|---|---|
| decoder (complex STFT) | 123.6 ms | 70.7 ms |
| decoder (CustomSTFT) | 124.2 ms | 71.5 ms |
| `predictor.F0Ntrain` (wrap) | ~10–13 ms | ~14.7 ms (no patch) |

Both STFT spellings cost the same — the conv/resblock stack
dominates; the iSTFT itself is small (n_fft=20, hop=5 ×60 upsample).

**Stochasticity floor (measured, important):** the decoder's
`SineGen` source module draws `torch.rand` (initial phase) and
`torch.randn_like` (noise) *inside* the forward.  Unseeded
eager-vs-eager waveform max|diff| = **5.5–7.1e-02** (waveform range
±0.34) — i.e. ~17 % of full scale.  Full-model run-to-run:
5.65e-02.  Any output-parity verify that does not align the RNG
**cannot pass** — this is wall A below.

## Pipeline results

### `f0n` (prosody head: shared LSTM + AdaIN resblocks → F0/N) — FULL PASS

- `opt.search`: 1.7–3.5 s.
- `lower(verify=True)`: **verified=True** (deterministic module —
  no in-graph RNG).
- delivered-vs-eager max|diff| = **0.0** (bitwise, seeded and
  unseeded).
- delivered **10.05 ms** vs eager-wrap ~10–13 ms (**~1.0–1.3×**);
  `CudaGraphRunner` **cuda_graph=True**, delivered+graph
  **6.66 ms** — beats eager and the ~14.7 ms `torch.compile` arm.
- The only nonstandard op inside is `aten.lstm.input` (bound
  passthrough) + AdaIN convs — all liftable.

### `decoder` (complex TorchSTFT) — LIFTS AND DELIVERS, slower than eager

- `opt.search`: **76.0 s** (the complex `stft`/`istft` region
  exports as `opaque` passthrough leaves — see wall D).
- `lower(verify=True)`: runs (0.9 s) → **verified=False** — wall A
  (unseeded stochastic compare), NOT a semantic failure.
- **Seeded parity: delivered-vs-eager max|diff| = 1.49e-08** —
  effectively bitwise.  The pipeline preserves semantics exactly;
  only the RNG stream differs run-to-run (delivered unseeded
  run-to-run 5.93e-02 ≈ eager floor).
- delivered **138.8 ms** vs eager 123.6 (**0.89× — slower**) and vs
  compile 70.7 (**0.51×**).  Tape-eval overhead + mostly-opaque
  graph: the conv resblocks did not pay enough to cover it.
- `CudaGraphRunner`: `cuda_graph=False, fail=kernel` — same refusal
  class whisper/t5 encoders showed on this card.

### `decoder-nc` (CustomSTFT conv-iSTFT) — export OK, delivery walled

- `opt.search`: 8.7–9.5 s (much smaller graph than complex).
- `lower(verify=True)` / unverified delivery / timing: all die at
  first eval — **`TypeError: mul() missing 1 required positional
  arguments: "other"`** (`torch.mul` bound raw, called with ONE
  operand).  Present at `max_iterations=0` → an **export/lower
  defect, not a search rewrite**.
- Localization: the plan's step 964 is `mul` with a single arg
  produced by `sin` (region: `slice→sin→mul→exp` around the
  STFT magnitude/phase computation).  Isolated arms:
  `stft.inverse(spec, phase)` alone → **clean** (delivered diff
  2.98e-08); `stft.transform(waveform)` alone → no unary mul but
  hits wall E (pad binding).  So the malformed `mul` is minted in
  the wider Generator/SineGen region of the nc graph — most
  consistent with a scalar-mul spelling (`sines * self.sine_amp`,
  `mul.Scalar`-class) losing its scalar Const operand — the same
  *defect class* as t5's `min`/`max` overload collision but unary.
  **Not fixed — upstream torch_bridge territory; documented, not
  patched** (unlike the device wall below, there is no honest
  probe-local re-spelling for a dropped operand).
- The complex variant does NOT contain the defect (its STFT region
  is opaque leaves) — which is why `decoder` delivered and
  `decoder-nc` did not.

### `textenc` / `full` — export wall (packed LSTM)

- `search` fails in 0.3–1.3 s:
  `ValueError: batch_sizes should always be on CPU. Instances of
  PackedSequence should never be created manually…`
- `TextEncoder.forward` and `DurationEncoder.forward` both call
  `nn.utils.rnn.pack_padded_sequence` → `aten.lstm.data` is
  *deliberately unbound* (torch_bridge.py:2393 comment) and the
  `PackedSequence` pytree does not survive `torch.export` here.
  `full` (`forward_with_tokens`) hits the same wall plus dynamic
  `repeat_interleave` lengths — it never gets that far.
- Consequence: the text-side half of Kokoro (ALBERT + DurationEncoder
  + TextEncoder + alignment construction) is currently **unliftable
  end-to-end**; only the post-alignment heads (`F0Ntrain`, decoder
  inputs harvested as tensors) go through.

## Walls — in order

### Wall A — stochastic source module defeats output-parity verify

`SineGen._f02sine` draws `torch.rand` (phase init) and
`torch.randn_like` (noise) inside the forward; unseeded outputs
differ by ~6e-02 max-abs.  `lower(verify=True)` therefore reports
`verified=False` even when semantics are bit-exact (seeded diff
1.5e-8).  Probe measures parity under aligned `torch.manual_seed`
instead.  Upstream options: a verify mode that reseeds per call, or
treating `rand`/`randn_like` leaves as verify-exempt — either way it
is a *verifier* gap, not a correctness gap.  (Also note the RNG
consumption must match op-for-op for seeded parity to hold — it
did, because the `rand`/`randn_like` leaves survived as
bound/opaque ops in the same order.)

### Wall B — bf16 unusable: cuDNN LSTM rejects bf16 params

`.to(bfloat16)` on the full KModel → first LSTM raises
`Input and parameter tensors are not the same dtype … Float vs
BFloat16` (cuDNN fp32-only for this path).  TTS is LSTM-dense
(predictor + text encoder + shared head), so **bf16 is off the
table for Kokoro end-to-end**; fp32 (330 MB weights) is the honest
dtype.  The decoder alone has no LSTM and could run bf16 — not
measured (time-bounded; the device wall below had to be solved
first anyway).

### Wall C — `device=` kwargs are silently dropped at export (two layers)

**Root cause, two cooperating defects in `torch_bridge.py`:**

1. The kwargs→attrs loop (lines ~811–834) records only
   int/float/bool/list/str/dtype — a `torch.device` object is
   **silently dropped**, so `aten.zeros/ rand / …(device=d)` exports
   with no device information at all.
2. The creator bindings (`zeros`/`ones`/`rand`/`randn`/`empty`/
   `full`/`new_*`, lines ~2455–2500) mint on **CPU** — and the
   `"to"`/`_to_copy` binding (line ~1909) applies *dtype only*, so
   even a `.to(device)` re-spelling cannot carry the device.

Minimal repro (committed in the report, rerunnable inline): any
`torch.rand(n, device=x.device)` inside a lifted forward →
delivered `torch.add(cuda, cpu)` →
`RuntimeError: Expected all tensors to be on the same device`.

Kokoro hits it immediately: `SineGen._f02sine` does
`torch.rand(f0.shape[0], f0.shape[2], device=f0_values.device)`
(the (1,9) `rand_ini` tensor — confirmed by instrumented
`torch.add`: `x=(1,9) cuda` vs `y=(1,9) cpu` random values).

**Probe workaround** (`_patch_device_creators`, probe-local like
t5's `_patch_min_max`): re-spell `f(shape…, device=d)` as
`f_like(devref(d).expand(shape…))` — the device rides a real
operand.  `zeros_like`/`ones_like`/`full_like` are bound (device
follows the reference); `rand_like`/`randn_like`/`empty_like` are
unbound → `opaque` passthroughs that replay the real op verbatim.
A pre-warmed `(1,)` CUDA `_devref` per device is expanded to the
requested shape.  Caveat: it changes *where* RNG is drawn is
preserved (still CUDA RNG via `rand_like`), but it is a spelling
change — semantics identical, documented as workaround not fix.

With the patch, `decoder`/`decoder-nc` get past the device wall.
**Upstream fix wanted:** record `device` in attrs (or route device
through `_device_like`-style inference) for creator ops AND for
`to`/`_to_copy`; alternatively mint creator outputs on the
`_sibling_like` device like `_mint_const` already does for tensor
consts.

### Wall D — complex `stft`/`istft` do NOT wall: they lift as `opaque`

`torch.stft`/`torch.istft`, `F.interpolate(mode="linear")`
(`upsample_linear1d`), `torch.randn_like`, `atan2`, index_put
mutations — all unbound → `opaque` passthrough leaves that replay
the real torch callable.  **No complex-dtype refusal occurred**:
the complex intermediates stay inside opaque leaves; the IR never
sees a complex tensor op.  Cost: those regions are unoptimised
(and un-optimisable by laws) — part of why `decoder` is *slower*
than eager.  `fft_*` ops ARE bound (rfft/irfft family exists) but
`aten.stft`/`aten.istft` are different ops and stay opaque.

### Wall E — `F.pad` 1-D reflect unsupported by the pad binding

`CustomSTFT.transform` does `F.pad(waveform, (n//2, n//2),
mode=pad_mode)` on a 1-D tensor → delivered binding raises
`NotImplementedError: Padding size 2 is not supported for 1D input
tensor` (isolated `stft.transform` arm).  Small, precise binding
gap.

### Wall F — `pack_padded_sequence` / `PackedSequence` unexportable

(textenc/full arms — see above.)  `aten.lstm.data` deliberately
unbound + PackedSequence pytree → `torch.export` refuses.

## Scoreboard vs t5/whisper

| stage | whisper-base fp32 | t5-small fp32 | kokoro fp32 |
|---|---|---|---|
| export/lift | pass | pass (workarounds) | pass (after device patch) |
| search | 4–29 s | 4.8–6.5 s | f0n 2–3.5 s; decoder 76–9 s |
| lower+verify | verified | verified | f0n **verified**; decoder verified=False (wall A) |
| delivered vs eager | 1.1× | 1.14× | f0n ~1.0–1.3× (+graph 6.7 ms); decoder **0.89×** |
| delivered vs compile | ~parity | 0.88–0.99× | f0n **beats** (~14.7 ms); decoder 0.51× |
| cuda_graph | refused | refused | f0n **captured**; decoder refused |
| exotic ops | — | int feeds, min/max | **in-graph RNG, device creators, PackedSequence, complex STFT (opaque), unary-mul defect** |

## Verdict for "any model" (kokoro r1)

- A real TTS model **partially** goes through the pipeline: the
  post-alignment neural heads (`F0Ntrain`) deliver verified with
  bitwise parity and a captured CUDA graph that beats both eager
  and `torch.compile`; the full ISTFTNet decoder **lifts, searches,
  lowers and delivers** (complex variant) with seeded parity
  1.5e-8 — but runs *slower* than eager because its STFT/iSTFT and
  RNG regions survive only as opaque leaves.
- New wall classes found (all characterised, none fixed upstream):
  **in-graph stochasticity vs output-parity verify** (wall A — a
  verifier gap, semantics proven by seeded diff), **`device=`
  kwargs dropped at export + CPU-minting creator/`to` bindings**
  (wall C — worked around probe-locally via `*_like` respelling),
  **a unary `mul` export defect** in the CustomSTFT decoder graph
  (TypeError at eval, iters=0 — the nc variant cannot deliver at
  all), **PackedSequence unexportable** (text side unliftable
  end-to-end), **bf16 blocked by cuDNN LSTM** (fp32 only),
  **1-D reflect `F.pad` binding gap**, and confirmation that
  **complex `stft`/`istft` degrade gracefully to opaque leaves
  rather than refusing**.
- The "heavy conv+ISTFT = our strength" hypothesis is **not yet
  confirmed on TTS**: the vocoder delivered but slower than eager
  (0.89×); the win so far is the small deterministic head +
  CUDA-graph capture.  The decoder's opaque-STFT region and the
  unary-mul defect are the two things standing between this arm
  and a real measurement.

## Reproduce

```sh
git clone --depth 1 https://github.com/hexgrad/kokoro /tmp/kokoro-src
uv sync
.venv/bin/pip install --no-deps loguru attr   # probe-only deps

.venv/bin/python bench/drivers/probes/tts_kokoro_probe.py bench fp32
.venv/bin/python bench/drivers/probes/tts_kokoro_probe.py synth fp32
.venv/bin/python bench/drivers/probes/tts_kokoro_probe.py opt f0n fp32
.venv/bin/python bench/drivers/probes/tts_kokoro_probe.py opt decoder fp32      # complex STFT
.venv/bin/python bench/drivers/probes/tts_kokoro_probe.py opt decoder-nc fp32    # CustomSTFT -> unary-mul wall
.venv/bin/python bench/drivers/probes/tts_kokoro_probe.py opt textenc fp32       # PackedSequence wall
.venv/bin/python bench/drivers/probes/tts_kokoro_probe.py opt full fp32          # same wall upstream
```

Stage-isolation diagnostics were run from `/tmp/tts_diag*.py`
(ephemeral, not committed): device-mismatch tensor scan, instrumented
`torch.add` mismatch print, plan-step unary-op census, isolated
`stft.transform`/`stft.inverse` arms, `max_iterations=0` ablation.

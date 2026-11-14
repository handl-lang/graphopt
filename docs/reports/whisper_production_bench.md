# Whisper production-representative benchmark — delivered+optimized vs direct HF

Date: 2026-11-14.  Hardware: RTX 2050 4 GB (**shared GPU** — absolute ms
swing run to run; replicate spreads reported honestly).  torch
2.14.0+cu130, **transformers 4.57.1** (scratch-installed to
`/tmp/whisper_deps` + `PYTHONPATH` — `uv sync` prunes non-lockfile
installs from `.venv`; earlier probes ran against transformers 5.x).
Probe: `bench/drivers/probes/whisper_prod_bench.py` (new, uncommitted
scratch).  Reuses the static-K/V machinery of
`whisper_kv_static_probe.py` via import, generalized to arbitrary
`decoder_layers`/`num_heads`/`head_dim` and patched for 4.57 API
drift (`WhisperDecoderLayer` returns a tuple; `CacheLayerMixin`
gained abstract `get_max_cache_shape`).

## What was measured

Real log-mel features from **synthetic 8 s audio** (3 sines +
amplitude-modulated noise, 16 kHz) through `WhisperFeatureExtractor`
— the same frontend HF uses, including the 30 s zero-pad window.
Honest caveat: the audio is not speech, so the models emit degenerate
repetitive tokens (`291`, `370`, …).  That does not affect the
benchmark — all arms see identical features and the correctness gate
is token-id equality, not transcription quality.

Decode is a **fixed 48-token greedy horizon** (no early EOS stop —
equivalent to HF `min_new_tokens=48`, whose MinLengthLogitsProcessor
EOS ban we replicate exactly).  The decoder prompt prefix
(`[sot, <|en|>, <|transcribe|>, <|notimestamps|>]`, K=4) is captured
from a real `generate` call via forward hook, so our manual loop is
seeded identically to HF.  HF's `suppress_tokens` +
`begin_suppress_tokens` are applied in every manual arm.

**Correctness — the number that matters:**

| check | result |
|---|---|
| `generate(min_new_tokens=48)` ids vs manual HF-loop ids | **48/48 exact, all 3 models** |
| wrap-static harvest-step logits vs real HF step | max\|diff\| 0–1.1e-5, argmax equal |
| `Optimizer.lower(verify=True)` | **verified=True, all 3 models** |
| every loop arm (static-wrap / delivered / delivered+graph / compile-wrap) vs HF-loop ids | **48/48 exact, all 3 models** |

## Decode-step latency (ms/token, warm mean of 3 reps)

| arm | whisper-base fp32 (6L, d512) | whisper-small fp32 (12L, d768) | whisper-large-v3-**turbo** bf16 (4L, d1280) |
|---|---|---|---|
| HF eager decoder loop | ~5.1 (4.6–5.5 across runs) | ~9.3–9.6 | ~5.1 |
| static-wrap eager (probe wrap, no pipeline) | ~4.1 | ~8.75 | ~4.4 |
| **delivered `Optimizer.lower` module** | **4.39** | **~10.2** | **4.4** |
| **delivered + `CudaGraphRunner(alias+mutable)`** | **~4.2** (3.7–5.0, noisy) | **~10.4** | **~4.4** |
| `torch.compile` on same wrap | ~3.3 | ~8.8 | ~4.34 |
| decode-loop peak VRAM (weights resident) | 0.64–0.66 GiB | 1.85–1.88 GiB | 1.01–1.03 GiB |

Search cost: base 8.7 s, small 43 s, turbo 26 s (`max_iterations=3`).

## End-to-end `generate()` (encoder + decode, 48 tokens)

| arm | base | small | turbo |
|---|---|---|---|
| HF eager `generate` (natural stop — EOS after ~1 tok on this input) | ~86 ms | ~297 ms | ~526 ms |
| HF eager `generate(min_new_tokens=48)` | ~362 ms (one 678 ms outlier — shared GPU) | ~860 ms (one 1535 ms outlier) | ~810 ms |
| HF `torch.compile(model)` + `generate(min_new=48)` | ~340 ms | ~990 ms | ~790 ms |
| encoder forward alone (HF eager, once) | ~68–79 ms | ~243 ms | ~505 ms |
| **our path estimate** = HF encoder + delivered+graph decode | **~255–275 ms** | **~640–660 ms** | **~710–720 ms** |

The "our path" row is a derived estimate: encoder is **not**
optimized (still HF eager — see exclusions), decode uses the
delivered module under alias+mutable CUDA graph.  It assumes the
per-step Python bookkeeping (embed+pos, mask build, argmax, D2H sync)
delivered-loop timings exclude is comparable to what HF pays inside
`generate` — roughly 0.2–0.4 ms/step, inside the estimate's slack.

## Honest read

* **turbo-large-v3 fits and verifies on 4 GB — but only decoder-side.**
  bf16 weights 1.62 GB; the encoder (32L, d1280) runs HF-eager fine
  (~505 ms) but was never lifted — and for the decode benchmark it was
  *dropped from VRAM* after `encoder_hidden_states` was computed
  (~1.4 GB freed) so `Optimizer.search`/`lower(verify=True)` fit.
  Without that, export OOMs (measured).  `lower(verify=True)` passed
  for turbo — the earlier r2 verify-OOM was the *encoder*; the
  4-layer decoder step is small.
* **The win is real but modest on this GPU.**  Decode step: ~15–20 %
  vs HF eager on base/turbo, ~parity on small (delivered 10.2 vs
  eager 9.3 — *slower*; only the wrap-eager 8.75 and compile-wrap 8.8
  beat it).  `delivered+graph` is not clearly better than `delivered`
  here — the RTX 2050 is GEMM/dispatch-bound differently than the
  numbers the static-KV machinery was tuned on; per-rep spread
  (3.7–5.0 ms base) exceeds the arm deltas.  `torch.compile` on the
  same wrap is the strongest single arm on base/small.
* **E2E is encoder-dominated for turbo** (505/810 ≈ 62 %).  Even a
  free decoder cannot move turbo e2e much; the decode win shows up
  mainly on base/small where the encoder is cheap.
* **`torch.compile(model)`+`generate` ≈ parity** (base slightly
  better, small *worse* — Inductor graph breaks through the cache/
  generation boundary cost more than they save on this part).
* Our decode loop is **greedy, batch-1, fixed horizon, single
  sequence** — no batching, no concurrency, no streaming.  HF
  `generate`'s own overhead (logits processors, stopping criteria,
  segment bookkeeping) is *included* in its e2e numbers and excluded
  from ours.

## What's included / excluded

* Included: WhisperFeatureExtractor frontend (30 s window), encoder
  forward (HF eager, bf16 for turbo / fp32 else), full greedy decode
  with real KV cache, HF suppression policy, CUDA-graph replay with
  alias+mutable inputs, `verify=True` equivalence gate.
* Excluded: encoder optimization (our pipeline only owns the decode
  step — encoder stays HF eager in "our path"), batching, beam
  search, streaming/long-form (>30 s) segmentation, audio
  frontend beyond feature extraction, CUDA graphs around the encoder.
* The delivered decode module is probe-local machinery
  (`_DecodeStepWrap457` + cache layers) — the shipped surface is the
  `CudaGraphRunner(alias_inputs=, mutable_inputs=)` opt-in plus the
  verified `IRModule` tape; the wrap itself is benchmark scaffolding,
  same status as `whisper_kv_static_probe.py`.

## Reproduce

```sh
uv pip install --target /tmp/whisper_deps "transformers==4.57.1"
export PYTHONPATH=/tmp/whisper_deps
.venv/bin/python bench/drivers/probes/whisper_prod_bench.py eager base 48
.venv/bin/python bench/drivers/probes/whisper_prod_bench.py opt base 3 48 64
.venv/bin/python bench/drivers/probes/whisper_prod_bench.py opt small 3 48 64
.venv/bin/python bench/drivers/probes/whisper_prod_bench.py opt turbo 3 48 64
.venv/bin/python bench/drivers/probes/whisper_prod_bench.py compilegen {base,small,turbo} 48
```

Raw logs: `/tmp/{eager_{base,small,turbo},opt_{base2,small2,turbo},compilegen_{base,small,turbo}}.log` (ephemeral).

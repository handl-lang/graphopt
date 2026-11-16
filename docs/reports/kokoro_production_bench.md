# Kokoro-82M production-path benchmark — delivered+optimized vs direct HF

Date: 2026-11-14.  Hardware: RTX 2050 4 GB (**shared GPU** — an
encoder-agent co-tenant held ~2.7–3.8 GiB for part of the session;
absolute ms swing run to run, ranges reported honestly).  torch
2.14.0+cu130, **fp32 only** (bf16 is walled end-to-end: cuDNN LSTM
rejects bf16 — `tts_kokoro_r1.md` wall B).  transformers 4.57.1
already in `.venv` (no scratch install needed this time).
Probe: `bench/drivers/probes/kokoro_prod_bench.py` (new, uncommitted
scratch) — modes `arm <which> [iters]` / `compile <which>` / `e2e`.
Reuses `tts_kokoro_probe._harvest` for the arm census and inputs.

This completes the startup perf picture: STT shipped in
`whisper_production_bench.md`; TTS was the pending half.

## What was measured

Per arm: **eager** (raw kokoro module wrap, warm timed) vs
**delivered** (`Optimizer.search` → `lower`) vs
**delivered+`CudaGraphRunner`** vs **delivered+`RegionFusionRunner`**
vs **`torch.compile(wrap)`**.  Parity is seeded
(`torch.manual_seed(1234)`) because the decoder's SineGen source
module draws `rand`/`randn_like` in-graph — the unseeded run-to-run
floor is ~6–8e-2 max|diff| on decoder arms (measured per arm below).
`lower(verify=True)` is only attempted on deterministic arms
(floor == 0): on stochastic arms it can only refuse (wall A) and its
second reference lowering is what OOM'd the shared card.

`RegionFusionRunner` is the shipped runner
(`kit_torch.runners`, commit 2f7c6b9).  The bench passes
`output_rtol=2e-3` — the documented override: the stock `1e-4`
whole-module gate declines the decoder arms (fused-vs-unfused
max_rel ≈ 7e-4–1.4e-3, `kokoro_region_fuse.md` §4/§5).  The per-kernel
screening gate stays at its 1e-4 default.

## Per-arm results (fp32, warm mean)

| arm | eager ms | delivered ms | +CUDA-graph | +RegionFuse ms | compile ms | parity (seeded max\|diff\|) |
|---|---|---|---|---|---|---|
| `f0n` (F0/N head) | 7.24 | **7.33** (verify=True, diff 0.0) | **5.92** (captured) | 7.43 (applied, no gain) | 7.52 | delivered 0.0; fuse-vs-unfused rel 2.5e-5 |
| `textenc-p` (TextEncoder, packed-LSTM wrap) | 3.51 | **3.13** (verify=True, diff 0.0) | refused (`fail=kernel`) | n/a (0 eligible regions) | FAILED — custom_op autograd | delivered 0.0 |
| `decoder` (complex TorchSTFT) | ~123 | **116.5** | refused (`fail=kernel`) | **81.5** | **70.1** | delivered 0.0; fuse-vs-delivered rel 7.1e-4 |
| `decoder-nc` (CustomSTFT) | ~123 | **116.8** | refused (`fail=kernel`) | **87.4** | **70.7** | delivered 3.0e-8; fuse-vs-delivered rel 4.5e-4 |
| `full-p` (whole `forward_with_tokens`) | 149.2 | — | — | — | — | **export wall, 0.5 s** |

Search cost: f0n 1.8 s, textenc-p 0.4 s, decoder-nc 36.5 s,
decoder 4.4 s this run (76 s in r1 — e-graph/search variance).
Decoder unseeded stochastic floor: 5.8e-2 (`decoder`),
6.6e-2 (`decoder-nc`); delivered unseeded run-to-run matches the
floor (5.9e-2 / 6.7e-2) — delivery preserves the RNG stream.

Notes per arm:

* **f0n** — the only arm where delivered+CUDA-graph *beats*
  `torch.compile` (5.92 vs 7.52 ms).  Region fusion applies (4
  regions/10 ops/2 kernels) but the regions are tiny — no wall-time
  gain (7.43 ms).  `verified=True`, bitwise parity.
* **textenc-p** — delivers verified at 1.12× eager, but requires the
  probe-local `tts_kokoro_packed` custom_op wrap (PackedSequence never
  crosses export).  `torch.compile` on the same wrap **fails**:
  inductor hits `no autograd formula was registered` on the custom
  op — a real compile wall for the packed-LSTM spelling.
* **decoder / decoder-nc** — search alone is ~parity with eager
  (116.5/116.8 vs ~123 ms, i.e. ~1.05×); the r1 "slower than eager"
  result is gone (weight_norm fold + torch_port_gen schema bindings
  landed since).  The **RegionFusionRunner is the whole win**:
  −35 ms (`decoder`, 116.5→81.5) and −29 ms (`decoder-nc`,
  116.8→87.4), matching `kokoro_region_fuse.md`'s prediction.
  Census this run: 57 regions fused, 408/345 ops, 7/8 distinct
  compiled kernels; skips honest (2–3 multi_root, 17–30 small).
  Fused-vs-unfused gates passed at rtol=2e-3 (max_rel 7.1e-4 /
  4.5e-4); fused-vs-eager-seeded ≈ 1.7e-4–2.9e-4 — inductor
  reassociation noise, not a rewrite bug (the spelled control in the
  region-fuse report is bitwise).
* **full-p** — still walled, and earlier than expected: export dies
  at `model.py:101` `torch.arange(input_lengths.max())` →
  `GuardOnDataDependentSymNode` (data-dependent shape) in 0.5 s,
  *before* the `repeat_interleave`/alignment construction.  The
  packed-LSTM ops get the LSTMs through; the data-dependent shapes
  (coverage-map item #5) are the remaining wall for any single-call
  delivery.

## End-to-end `forward_with_tokens` (the production-shaped number)

Eager stage breakdown (fp32, per-call wall, sums to the total):

| stage | ms | deliverable today? |
|---|---|---|
| `bert` (ALBERT text encoder) | ~7.7 | not lifted (per-module unmeasured) |
| `bert_encoder` (Linear) | ~0.08 | trivial, unmeasured |
| `predictor.text_encoder` (DurationEncoder, packed LSTM) | ~7.1–8.1 | packed-wrap machinery exists; arm unmeasured |
| `predictor.lstm` + duration head | ~1.4 | raw padded LSTM, unmeasured |
| duration_proj + `repeat_interleave`/alignment | ~0.4 | **data-dependent wall** — stays eager |
| `F0Ntrain` | ~6.9–7.1 | **delivered**: 7.33 / +graph **5.92** / +fuse 7.43 |
| `text_encoder` (packed) | ~3.3–3.6 | **delivered**: 3.13 (needs packed wrap) |
| `decoder` (complex STFT) | ~123 | **delivered+fuse: 81.5** (+nc variant 87.4) |
| **total eager** | **~150–152** | measured `forward_with_tokens` 148–152 ms |

Estimated our-path e2e (sum of measured delivered arms + eager
remainder — **an estimate, not a single delivered call**):

| composition | est. ms | vs eager ~151 |
|---|---|---|
| delivered only (no fusion, no graph) | ~142 | ~1.06× |
| delivered + graph on f0n + fused decoder | **~108** | **~1.40×** |
| same with decoder-nc fused | ~113 | ~1.33× |
| `torch.compile(forward_with_tokens)` — measured | **95.4** | ~1.58× |

The torch.compile e2e number is real but comes with an honest
caveat: **seeded parity vs eager was NOT established** —
compile-vs-eager seeded max|diff| = **1.15e-1**, above the
stochastic floor (~7e-2).  Consistent with RNG-stream/reassociation
divergence under inductor rather than a semantic break, but it is
unverified output — report it as "compiles and runs ~37% faster,
correctness not gated", not as a clean win.  It also specializes on
this input's data-dependent shapes; other utterances recompile.

## Honest read

* **Per-module, the delivered path is real**: f0n verifies bitwise
  and its CUDA-graph arm beats everything including compile;
  textenc verifies bitwise at 1.12×; both decoder arms deliver at
  eager parity and RegionFusionRunner recovers ~60% of the
  eager→compile gap (81.5/87.4 vs compile's ~70 ms — compile still
  wins the decoder arms outright).
* **Nothing composes end-to-end today.**  `full-p` walls at export
  on `input_lengths.max()`→`arange` (data-dependent shape), upstream
  of `repeat_interleave`.  The ~108 ms "our path" row is a
  sum-of-parts estimate across module boundaries with the
  alignment/duration glue left eager — per-module wins that do not
  yet compose into a single callable.
* **Shape specialization is the quiet killer for production**:
  `pred_dur` (and hence `asr`/F0/alignment lengths) is
  data-dependent per utterance; the fused-region kernels are
  `dynamic=False` — every new output length recompiles ~7 kernels.
  A production TTS path needs either dynamic-shape fusion or
  bucketed lengths.
* **Verify policy still blocks shipping the decoder win through
  `lower(verify=True)`**: stochastic arms refuse unseeded, and the
  fused module needs `output_rtol=2e-3` (stock 1e-4 declines).
  Seeded parity is proven (2.98e-8 delivered; ~1e-4 fused-vs-eager
  through ~50 regions).
* bf16 remains off the table (cuDNN LSTM); all numbers fp32.

## What's included / excluded

* Included: all five `_harvest` arms end-to-end through
  search→lower→deliver (+graph, +RegionFusionRunner, compile
  reference), seeded parity, e2e stage breakdown, whole-model
  `forward_with_tokens` eager + `torch.compile` attempt.
* Excluded: G2P/phonemization (misaki — irrelevant to the acoustic
  path), `KPipeline` batching, voice-pack indexing beyond the probe's
  approximation, ONNX.  `bert`/`bert_encoder`/DurationEncoder/
  `predictor.lstm` were timed eagerly inside e2e but **not** lifted
  per-module — listed as unmeasured, not claimed.
* The packed-LSTM custom_op wrap (`tts_kokoro_packed`) and the
  device-creator monkeypatch are probe-local machinery, same status
  as in r1 — the textenc delivery number depends on them.

## Reproduce

```sh
# arms (each its own process; sequential on the shared 4 GB card)
KPB_NOCOMPILE=1 .venv/bin/python bench/drivers/probes/kokoro_prod_bench.py arm f0n 3
KPB_NOCOMPILE=1 .venv/bin/python bench/drivers/probes/kokoro_prod_bench.py arm textenc-p 3
KPB_NOCOMPILE=1 .venv/bin/python bench/drivers/probes/kokoro_prod_bench.py arm decoder-nc 3
KPB_NOCOMPILE=1 .venv/bin/python bench/drivers/probes/kokoro_prod_bench.py arm decoder 3
KPB_NOCOMPILE=1 .venv/bin/python bench/drivers/probes/kokoro_prod_bench.py arm full-p 1
# compile references (lean processes — inductor needs the room)
.venv/bin/python bench/drivers/probes/kokoro_prod_bench.py compile decoder
.venv/bin/python bench/drivers/probes/kokoro_prod_bench.py compile decoder-nc
# e2e
.venv/bin/python bench/drivers/probes/kokoro_prod_bench.py e2e fp32
```

Raw logs: `/tmp/kpb_{f0n,textenc,decnc,dec,full,e2e,c_dec,c_decnc}.log`
(ephemeral).  Requires `/tmp/kokoro-src` (`KOKORO_SRC` override) and
the HF cache for `hexgrad/Kokoro-82M`.

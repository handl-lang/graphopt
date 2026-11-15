# Kokoro e2e split delivery — measured

Date: 2026-11-14.  Implements option A (+B) of
`kokoro_e2e_shapes.md` as a probe-local driver:
`bench/drivers/probes/kokoro_split_probe.py`.  fp32, RTX 2050 4 GB
(shared card — runs interleaved with a co-tenant; timings are
same-run comparisons).  transformers 4.57.1 model code from
`/tmp/kokoro-src`, weights `hexgrad/Kokoro-82M`, voice `af_heart`.

## The split (what the driver does)

`forward_with_tokens` is cut at the `pred_dur` boundary:

* **M1** `_M1Wrap(ids, ref_s) -> (d, t_en, pred_dur)` — model.py:100-116
  verbatim through `pred_dur`, **plus** `model.text_encoder` (the
  `asr` path's T-static half).  Wall A respelled: `input_lengths` is
  provably `full((B,), T)` (model.py:93-98), so `text_mask` is built
  from `arange(input_ids.shape[-1])` — identical values, static
  shape.  The three pack→LSTM→pad sites use the shipped
  `tts_kokoro_packed.install(m)` custom-op wrap.
* **glue** (host-side, eager): `S = pred_dur.sum()`, `W` = `S`
  (`exact`) or `ceil16(S)` (`bucket`); the `(1,T,W)` alignment is
  `one_hot(searchsorted(cumsum(pred_dur), w))` — no
  `repeat_interleave` needed; `en = dᵀ @ aln`, `asr = t_en @ aln`
  (two small matmuls), then `F0, N = predictor.F0Ntrain(en, s)`
  **eager** (~7 ms — its delivered arm measured no faster, see §4).
* **decoder artifact** `_DecWrap(asr, f0, n, s128) -> audio`, keyed
  by `W` only (T never enters the tail).  Trim `audio[..., :S*600]`.

A bundled `_M2Wrap(en, asr, s, s128) = F0Ntrain + decoder` also
exists for reference — it is the composition that exposed two
substrate bugs (§3).

## Measured e2e (GPU, fp32, `KPS_NOVERIFY=1`)

| run | T | S | W | eager `forward_with_tokens` | SPLIT | speedup | split-vs-eager seeded max\|diff\| | spec dist (log1p-STFT mean\|d\|, rel-magnitude mean) |
|---|---|---|---|---|---|---|---|---|
| bench (prod_bench input, random ids) | 42 | 127 | 127 exact | 152.7 ms | **110.6 ms** | **1.38×** | 2.90e-4 | (7e-5, 6e-4) |
| bench, contended run | 42 | 127 | 127 exact | 160.9 ms | 119.1 ms | 1.35× | 2.90e-4 | (7e-5, 6e-4) |
| bench, bucket mode | 42 | 127 | 128 (+1) | 153.1 ms | **110.8 ms** | **1.38×** | **3.93e-1** | **(4.2e-2, 0.25)** |
| second (real phoneme string) | 50 | 131 | 131 exact | 161.0 ms | **117.9 ms** | **1.37×** | 4.12e-4 | (8e-5, 6e-4) |

References, same runs: `torch.compile(forward_with_tokens)` =
98.2–103.1 ms (faster than the split, **parity unverified** — the
known 1.15e-1 seeded diff from `kokoro_production_bench.md`).
Against the prod_bench eager baseline (150.6 ms) the clean exact
run is **1.36×**.

Split breakdown at T=42/S=127 (eager halves): M1 20.4–22.4 ms,
glue+F0Ntrain ~8 ms, delivered+regionfused decoder ~82 ms.  The
decoder arm's regionfuse applied cleanly at every W tested
(57 regions / 408 ops / 7 kernels; fused-vs-unfused max_rel
1.1–1.2e-3 < the 2e-3 gate).

## Parity — honest read

* **`W == S` is bit-exact through the eager split** (seeded
  max|diff| = 0.0, `check` mode) — the glue/alignment construction
  is semantics-preserving.
* **Delivered path parity = the fused-decoder parity**: seeded
  max|diff| 2.9–4.1e-4, log1p-STFT spec distance ~7e-5 — 100× below
  the model's own unseeded stochastic floor (unseeded run-to-run
  max|diff| ~7e-2–1e-1, spec ~0.009–0.011 log1p / ~0.06 rel —
  SineGen draws width-dependent `rand`/`randn_like`, so even eager
  never reproduces itself bit-wise).
* Seeded waveform parity across *different* W is impossible by
  construction (the RNG stream desyncs); the driver therefore also
  reports a deterministic-noise mode (`_DetNoise` — probe-local
  `rand_like`/`randn_like -> zeros_like` on both sides) that
  isolates semantic + contamination error.

## The S-boundary caveat — measured, worse than spec'd

The spec warned "convs are non-causal so outputs within one
receptive field of the S-boundary are contaminated".  Measured
(`check` mode, deterministic noise, T=14/S=49): **the
receptive field spans the whole utterance** — even `W = S+1`
contaminates globally:

| pad | waveform max\|diff\| | spec (log1p, rel) |
|---|---|---|
| W=S | 0.0 | (0, 0) |
| W=S+1 | 0.36 | (0.028, 0.21) |
| W=S+2 | 0.40 | (0.034, 0.24) |
| W=S+4 | 0.55 | (0.045, 0.33) |
| W=S+8 | 0.50 | (0.055, 0.39) |
| W=S+15 | 0.56 | (0.074, 0.42) |
| W=S+16 | 0.59 | (0.076, 0.41) |
| W=S+32 | 0.55 | (0.095, 0.45) |

Mechanism: `en`/`asr` pad columns are edge-replicated last-token
values → `F0Ntrain`'s non-causal conv stack perturbs F0/N **at
every position** (measured F0 error ~1 Hz globally, spiking ~180
near the boundary) → SineGen phase (`cumsum`) turns that into a
global waveform diff.  A trim margin does **not** help — the
contamination is not local.  Bucket mode (W=128 vs S=127) shows it
end-to-end: spec distance 0.042 ≈ **5× the stochastic floor**,
waveform max|diff| 0.39 — probably inaudible (sub-Hz F0 error) but
waveform-parity-failing.  **Recommendation: ship `exact`-W
artifacts (per-S, LRU-cached) unless a pad-content strategy that
survives the global RF is found** — e.g. predicting pad F0/N/asr
consistently, or accepting the spec-distance hit deliberately.
Per-S artifacts cost the same ~10–14 s build each; S takes ~600
distinct values over the observed range, so hit rate depends on
traffic repetition (spec option B).

## Bucket/artifact cache behavior

* Artifact key = `W` only (the decoder tail never sees T).  One
  build serves every utterance landing in the bucket.
* Per-artifact build (search iters=3 + lower + RegionFusionRunner
  2e-3): **~10–14 s** wall (search ~6 s, lower <0.1 s, regionfuse
  4–8 s, plus ~3 s harvest/inputs).  Amortizes after ~1 call at
  ~110 ms vs ~150 ms eager (payback < 1 s of served audio).
* `torch.compile` reference on the decoder wrap: 45 ms (W=64) —
  inductor still beats the delivered decoder on this arm; the
  delivered path's edge is gated parity + pre-built artifacts, not
  raw speed vs inductor.

## Substrate bugs found (report upstream — eqsat/orchestrator)

1. **`transpose_noop` law miscompiles the bundled M2 term.**
   `F0Ntrain + decoder` in one module: search (even
   `max_iterations=0` — the rewrite comes from the non-local
   passes/extraction, not saturation) drops a *needed* `transpose`
   between `upsample_nearest1d` and the SineGen harmonic
   `multiply`, producing `aten::multiply.Tensor` on
   `(1,1,38400) × (1,1,9)` → RuntimeError at module call, and
   `verify=True` fails the same way.  `KPS_EXCLUDE_RULES=transpose_noop`
   → delivered module **bit-exact (0.0)**.  The decoder arm alone
   never triggered it — the bug needs the F0N→decoder composition.
   **ROOT-CAUSED + FIXED in eqsat** (see below §4): the law and its
   `axes-noop` guard were correct — the guard's *evidence* was
   fabricated.  `_infer_op_shape`'s partial-unknown fallback
   reported a known broadcast operand's shape for `add`/`mul`/…
   even when the sibling operand's shape was `None` (the
   `nn.LSTM` inside `F0Ntrain.shared` is unshapeable).  The
   fabricated `(1,256,1)` cascaded through `conv1d` → `(1,1,1)`,
   and `transpose(1,2)` on that fake shape satisfied
   `axes-noop`'s "both swapped extents are 1" clause.
   Verified end-to-end: `arm m2 64 0` with the patched core is
   **bit-exact (0.0)** with no rule exclusions.
2. **Carrier shape-inference crash on M1 (T=42).**  `optz.search`
   dies inside `eg.run → apply_rule → mlopt_carriers.xcarrier._xshape`
   building `TensorType(dim=-1)` — reproduced 2026-11-14:
   `KPS_M1DELIVER=1 KPS_NOVERIFY=1 .venv/bin/python
   bench/drivers/probes/kokoro_split_probe.py arm m1 3`.
   **ROOT-CAUSED (report only — carriers/typing lane):** the
   `_M1Wrap` respell contains `x.expand(ids.shape[0], -1)`
   (probe line ~108; torch `expand` accepts `-1` = keep-dim).
   `_infer_op_shape`'s `case "expand" | "broadcast_to"` returns
   the `shape` attr **verbatim** — `(1, -1)` — and
   `xcarrier._xshape_rec` (xcarrier.py:305) wraps it in
   `Var("_xs", TensorType(tuple(s)))`, which since eqsat e18b7ef
   rejects negative dims → `ValueError: TensorType dim must be
   >= 0, got -1`.  Candidate fixes (other lane): resolve `-1`
   dims against the operand shape in the `expand` arm (like
   `reshape` does), or sanitize negatives → `None` in `_xshape`'s
   surrogate construction.  M1 delivery stays **off by default**
   (it measured no win anyway: 19.7 vs 21.4 ms — launch-bound
   like f0n).
3. **Bundled M2 defeats RegionFusionRunner's output gate**:
   fused-vs-unfused max_rel ~7e-2 > 2e-3 → `applied=False`.  The
   three-module decomposition sidesteps it (decoder alone fuses
   at 1.1e-3).  Still reproduces post-fix (max_rel 4.8e-2 at
   W=64) — unrelated to `transpose_noop`.

## 4. The `transpose_noop` root cause (eqsat fix, pending retag)

* **Symptom:** `transpose(upsample_nearest1d(…), dim0=1, dim1=2)`
  before `multiply(f0, harmonics(1,1,9))` deleted at
  `max_iterations=0` → `(1,1,38400) × (1,1,9)` broadcast crash.
* **Mechanism:** `F0Ntrain` contains `nn.LSTM`
  (`predictor.shared`); `lstm.input`/`getitem` infer `None`.
  Inside `AdainResBlk1d` the AdaIN tail computes
  `(1+gamma)*x + beta` where `gamma`/`beta` are style chunks of
  concrete shape `(1,C,1)`.  `_infer_op_shape`'s early return
  answered the KNOWN operand's shape for the broadcast `mul`/`add`
  (the `_ARG0_SHAPE_OPS` + `_COMMUTATIVE_BROADCAST` fallback added
  in eqsat e18b7ef), fabricating `(1,256,1)` for a tensor that is
  really `(1,256,64)`.  `F0_proj`'s `conv1d` then computed output
  length `conv_window(1,…) = 1` → `(1,1,1)`; `squeeze`/`unsqueeze`/
  opaque-`upsample` propagated it; `axes-noop` saw both swapped
  extents `== 1` and fired.
* **Why the law was innocent:** `transpose(u,d0,d1) == u` when the
  swapped extents are both 1 is a true semantic no-op — the guard
  simply cannot prove it from a fabricated shape.
* **Fix (eqsat `packages/mlopt-core/src/mlopt_core/typing.py`):**
  new `_BROADCAST_BINOPS` frozenset (the 27 ops whose match arm is
  `_broadcast`); the partial-unknown early return now declines to
  `None` for them on ANY unknown operand — the output can exceed
  every known operand, so borrowing one fabricates a too-small
  concrete shape.  Removed from `_ARG0_SHAPE_OPS`; the
  `_COMMUTATIVE_BROADCAST` alias is gone.  Shape-preserving unary
  ops (`relu`, `contiguous`, `masked_fill`, scatter/index family,
  `sdpa`, …) keep the operand-0 answer — it IS their output shape.
* **Regression pins:** `tests/test_typing.py::
  test_broadcast_binop_unknown_operand_declines` (the kokoro chain:
  `mul(param(1,256,1), unshapeable) → conv1d → transpose(1,2)` now
  infers `None` and `axes-noop` declines) and updated
  `test_partial_unknown_shape_not_operand0` /
  `test_typing_edges.py::mixed/first-shaped` which had pinned the
  old fabrication.
* **Verification:** eqsat `pytest -m "not slow"` 1699 passed;
  ruff/format/ty/lint-imports clean; graphopt law/export slice
  215 passed (one consumer-identity test updated for the retired
  alias); `arm m2 64 0` bit-exact.

## Honest bottom line

* **The split works and is fast**: ~110–118 ms vs ~153–161 ms eager
  = **1.35–1.38× e2e**, replicated on two utterances, with
  delivered-path parity at the fused-decoder level (2.9–4.1e-4
  waveform, ~1e-4 spec) — far inside the model's own stochastic
  floor.  Slightly *below* the spec's 1.44× optimistic estimate
  because M1 stays eager (~21 ms) and glue+F0Ntrain costs ~8 ms.
* **It does not beat `torch.compile(forward_with_tokens)`**
  (~99–103 ms) on wall time; the delivered path's advantages are
  gated parity, per-W pre-built artifacts (no per-utterance
  compile), and CUDA-graph potential — none of which showed up as
  extra speed here (decoder graph capture still refuses
  `fail=kernel`, unchanged from prod_bench).
* **Bucket (W-polymorphic) mode is parity-lossy by construction** —
  the decoder's receptive field is global, so any pad column
  perturbs the whole utterance (~5× stochastic floor at +1 col).
  Recommend exact-W artifacts; bucketing only if the spec-distance
  hit is consciously accepted.
* Two substrate bugs (§above) need eqsat/orchestrator attention
  before the bundled-M2 or M1-delivered variants are viable.

## Reproduce

```sh
# eager split correctness + contamination sweep (~2 min)
.venv/bin/python bench/drivers/probes/kokoro_split_probe.py check
# per-half arm measurement
KPS_NOVERIFY=1 .venv/bin/python bench/drivers/probes/kokoro_split_probe.py arm m2 128 3 bench
# full e2e (builds the W artifact in-process, ~10-14 s)
KPS_NOVERIFY=1 .venv/bin/python bench/drivers/probes/kokoro_split_probe.py e2e exact 3 bench
KPS_NOVERIFY=1 .venv/bin/python bench/drivers/probes/kokoro_split_probe.py e2e bucket 3 bench
# law bisect / opt-outs
KPS_EXCLUDE_RULES=transpose_noop KPS_NOVERIFY=1 ... arm m2 64 3
KPS_M1DELIVER=1 ... e2e exact 3 bench   # M1 delivery (fragile, no win)
```

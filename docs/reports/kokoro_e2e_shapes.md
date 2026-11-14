# Kokoro e2e data-dependent shapes — blocker anatomy + workaround feasibility

Date: 2026-11-14.  Read-only analysis + measured probes (scratch:
`/tmp/kokoro_shape_probe.py`, `/tmp/kokoro_bucket_analysis.py`, two
inline `torch.export` spikes).  No package source touched.

Context: `docs/reports/kokoro_production_bench.md` — `full-p` walls at
export in 0.5 s at `model.py:101` `torch.arange(input_lengths.max())`
→ `GuardOnDataDependentSymNode`, *before* `repeat_interleave`.  This
doc maps the whole chain, measures the prize and the padding cost,
and ranks the workaround options.

## 1. Blocker anatomy — `KModel.forward_with_tokens`
(`/tmp/kokoro-src/kokoro/model.py:87-119`, transformers 4.57.1 for the
ALBERT `bert`, torch 2.14.0+cu130, fp32 — bf16 is walled by cuDNN LSTM)

```
input_ids (1,T) long,  ref_s (1,256) fp32,  speed float
input_lengths = full((B,), T)                      # ALWAYS T — hardcoded
text_mask  = arange(input_lengths.max()) … gt     # ← WALL A (reported)
bert_dur   = bert(input_ids, attention_mask)      # (1,T,C)  ~7.7 ms
d_en       = bert_encoder(bert_dur).T             # (1,C,T)
d          = predictor.text_encoder(d_en, s, lengths, mask)  # packed LSTM
x          = predictor.lstm(d)                    # raw padded BiLSTM (1,T,C)
duration   = sigmoid(duration_proj(x)).sum(-1)/speed
pred_dur   = round(duration).clamp(min=1).long().squeeze()   # (T,)
indices    = repeat_interleave(arange(T), pred_dur)          # len S=Σpred_dur
pred_aln_trg = zeros((T, S)); [indices, arange(S)] = 1       # (1,T,S)
en   = d.transpose @ pred_aln_trg                  # (1,C,S)
F0,N = predictor.F0Ntrain(en, s)                   # width-S conv/LSTM head
asr  = text_encoder(input_ids,…) @ pred_aln_trg    # (1,C,S)
audio = decoder(asr, F0, N, ref_s[:, :128])        # width ∝ S → S·600 samples
```

Two *distinct* data-dependent points:

* **Wall A (the reported one, model.py:100-101).**  `input_lengths` is
  *provably* constant — lines 93-98 hardcode `torch.full((B,),
  input_ids.shape[-1])` — but the bound reaches `arange` as a *tensor
  value* (`lengths.max()`), which dynamo exports as a data-dependent
  expression (`u0`) → `GuardOnDataDependentSymNode`.  Semantically
  static; a respell to `arange(input_ids.shape[-1])` (or precomputed
  `text_mask` passed in) is identical *given the hardcoded `full`*.
  Cheap fix, necessary but not sufficient.
* **Wall B (the real one, model.py:110-118).**  `S = Σ pred_dur` is a
  *model output* — genuinely data-dependent.  `repeat_interleave` →
  `zeros((T, indices.shape[0]))` → `en`/`asr`/F0/N/decoder-input widths
  → audio length `S·600` (measured: S=127 → 76 200 samples).  No
  respelling removes this; `torch.export` cannot express "shape =
  f(tensor values)" at all, with or without `dynamic_shapes`.

Shape flow summary: everything up to `pred_dur` is **T-static**
(B=1; `KPipeline.infer` calls `model()` per utterance — production is
always B=1, `pipeline.py:393`).  Everything after is **S-dynamic**,
and decoder cost is ~linear in S (it is ~80% of the 150 ms wall).

## 2. Precedent / existing machinery

* **PackedSequence**: three pack→LSTM→pad sites (`modules.py:60,113,164`)
  already wrapped probe-locally as `torch.library.custom_op` leaves —
  `bench/drivers/probes/tts_kokoro_packed.py`, used by the `*-p` arms
  (`textenc-p` delivered verified, 3.13 ms).  Same wrap pattern covers
  `DurationEncoder` inside M1.
* **Device-creator patch** (`tts_kokoro_probe._patch_device_creators`)
  — probe-local monkeypatch for `torch.rand/zeros/…(device=)`; needed
  by `SineGen` inside the decoder regardless of the shape fix.
* **Dynamic shapes in the IR pipeline: NOT supported today.**
  `TorchSource`/`export_to_ir` calls `torch.export.export(model, args)`
  with **no `dynamic_shapes`** (`torch_export.py:337`);
  `_infer_shape` concretizes `int(d)` (`torch_export.py:91-95`);
  `TensorType(shape)` and `mlopt_core.typing._shape_of` are concrete
  int tuples throughout.  `dynamic_shapes` passthrough exists only in
  the *pt2 artifact* path (`export.py:557`), downstream of optimization.
  RegionFusionRunner compiles `dynamic=False` (shape-specialized).
* **Compositional fallback precedent** (`tests/test_e2e_pipeline.py:221`):
  a block that can't lift keeps its original module — the pipeline
  already tolerates eager glue between delivered blocks.
* **`repeat_interleave` has no binding** (coverage item #8, "0
  measured") — the recommended reformulation avoids needing it.

**Spike result (measured):** a reformulated tail — alignment built as
`((w >= starts[:,None]) & (w < ends[:,None]))` with `w =
arange(W)` where `W` is an *input tensor dim* — exports under
`torch.export` **with `dynamic_shapes` on W** and reproduces eager
exactly at W ∈ {48, 64, 96} from ONE `ExportedProgram`.  So
shape-polymorphic-W is expressible at the torch.export layer today;
carrying it through mlopt's IR is the missing machinery (§4, option D).

## 3. Measured prize and padding cost

Probe: 32 eager `forward_with_tokens` calls, T ∈ {16,24,42,60,84,120,
168,240} × 4 seeds, fp32, `disable_complex=False`.

* `S = Σpred_dur ≈ 2.6·T + 30`; S ∈ [68, 661] over the sweep.
* **Per-T variance is tiny**: at fixed T, S varies only ~±4% across
  seeds — the data-dependence is real but *narrow*.
* Eager wall scales ~linearly with S (73 ms @ T=8 → 865 ms @ T=336).

Padding-waste analysis (decoder-side compute ∝ padded width W; useful
work ∝ S; efficiency = ΣS/ΣW — long utterances dominate):

| bucket scheme | widths | compute efficiency | mean overhead | max overhead |
|---|---|---|---|---|
| fixed W = max(S) | {661} | ~34% | — | — |
| linear step 8 | 75 buckets | **98.7%** | 2.0% | 6.7% |
| linear step 16 | 38 buckets | **97.1%** | 5.2% | 17.6% |
| linear step 32 | 19 buckets | **94.2%** | 11.1% | 41.2% |
| linear step 64 | 11 buckets | 87.7% | 26.1% | 88.2% |
| geometric ~1.3× | 9-12 buckets | 86.6% | 17.7% | 45.5% |
| oracle per-T max | — | 97.2% | — | — |

Per-bucket **build cost is trivial**: one arm's search+lower+regionfuse
measured 5-8 s on this box (`/tmp/kpb_dec.log`: search 4.4 s, lower
0.0 s, regionfuse 2.4 s; f0n 1.8 s + 0.9 s).  19-38 buckets × ~7 s ≈
2-4 min one-time per arm — amortizes immediately.

Estimated e2e if the wall falls (sum-of-parts, same honesty level as
`kokoro_production_bench.md` §e2e):

| composition | est. ms | vs eager ~151 |
|---|---|---|
| M1 eager (~20 ms) + glue (~0.4) + M2 delivered+fused (~87 × 1.03 pad) | **~110** | ~1.37× |
| same, M1 also delivered (bert unmeasured, maybe −5 ms) | ~105 | ~1.44× |
| `torch.compile(forward_with_tokens)` measured | 95.4 | ~1.58× (parity NOT gated, 1.15e-1 diff) |

Padding waste does **not** eat the win at step ≤32 granularity (~6%
compute overhead on the S-linear tail ≈ ~4-5 ms on a ~110 ms call).
It only kills the naive "one fixed width" reformulation (34%
efficiency — dead on arrival).

## 4. Options ranked

**A. Two-call split + W-bucketing — RECOMMENDED (v0, today's machinery).**
Respell `forward_with_tokens` model-side into:

* **M1** (T-static): `input_ids, ref_s → {d, t_en, s, ref_s[:, :128],
  pred_dur, starts, ends}` — bert → bert_encoder →
  predictor.text_encoder (packed wrap) → predictor.lstm →
  duration_proj → pred_dur, plus text_encoder (packed wrap) for `t_en`.
  Fix Wall A by passing `text_mask`/`input_lengths` precomputed or
  respelling `arange(T)` (identical — `full()` is hardcoded).
* **Glue** (~0.4 ms, stays eager — the compositional-fallback
  precedent): `starts = cumsum(dur)-dur`, `ends = cumsum(dur)`,
  `S = ends[-1]`, `W = bucket_ceil(S)`, build `(T,W)` alignment or
  pass `starts/ends` + a `(1,W)` pad carrier into M2.
* **M2** (W-static per bucket): `en/asr` matmuls → F0Ntrain →
  decoder → `audio[:, :, :S·600]` trim.  Contains the two heaviest
  delivered arms (f0n 5.92 ms +graph, decoder 81.5 ms fused) plus two
  cheap matmuls — the ~87 ms tail.

Deliver M1 once per T (or leave eager initially — it is only ~20 ms
and bert/DurationEncoder delivery is unmeasured); deliver M2 per W
bucket (step 16-32 → 19-38 artifacts, ~2-4 min build).  Parity gate:
seeded max|diff| per bucket; **edge caveat** — padded alignment columns
are zeros → `en`/`asr` pad columns are zeros → F0N/decoder produce
garbage in the pad region; convs are non-causal so outputs within one
receptive field of the S-boundary are contaminated.  Trim `S·600`
plus a safety margin, or replicate the last valid alignment column
into the pad (edge replication) — must be measured, not assumed.
Expected: ~1.33-1.44× e2e, verified parity, no per-utterance
recompile (buckets pre-built), CUDA-graph capture possible per bucket
(f0n already captures; decoder refused on `fail=kernel` — unchanged).

**B. Per-utterance exact-S lower, LRU-cached.**  Same split, W = S
exactly (zero padding waste), re-run search+lower per new S (~5-8 s
first call).  S takes ~600 distinct values over the observed range →
cache hit rate poor on varied traffic; fine for repeated fixed
inputs.  Dominated by A unless traffic is highly repetitive.  Keep as
the degenerate case of A with step=1.

**C. Dynamic-shape IR extension — real fix, wrong time.**  Extend
`TensorType`/`typing._shape_of`/`torch_export._infer_shape` to
symbolic dims, pass `dynamic_shapes` through `TorchSource`, teach
cost/runners about symint shapes.  torch.export already proves the
reformulated tail exports W-polymorphic (§2 spike).  Payoff over A is
only the ~3-6% padding waste + fewer artifacts; cost is a multi-package
eqsat+torch project, and CUDA-graph capture still needs static shapes
(per-bucket capture regardless).  Phase-2 candidate, not the wall-breaker.

**D. Single-call padded reformulation — DEAD (measured).**  Pad the
whole forward to one W: 187% decoder-side waste (§3).  Custom-op wrap
of the glue doesn't help either: a custom_op returning a
data-dependent-shaped tensor still walls every downstream shape —
the wall is the *shape*, not the op.  Only viable wrapped inside option
A's split.

**E. repeat_interleave binding (coverage #8) — not needed.**  The
broadcast-mask construction `(w >= starts) & (w < ends)` replaces it
and is exportable; skip the binding.

## 5. Honest bottom line

* The wall is **two** walls: a fake-static one (`arange(lengths.max())`,
  trivially respelled) and a real one (`S = Σpred_dur` → all downstream
  widths).  Fixing only the reported line moves the wall ~15 lines down.
* The prize is real but modest: **~1.35-1.45× e2e** estimated
  (~105-114 ms vs eager ~151 ms), *below* the measured
  `torch.compile` e2e (95.4 ms) whose parity is unverified.  The
  delivered path's edge would be gated parity + pre-built buckets +
  CUDA-graph/fusion wins that per-module arms already showed
  (f0n 5.92 ms beats compile's 7.52) — unproven until M1/M2 are built.
* Padding waste is **not** the blocker at sane granularity (≤6% at
  step-32); the blockers are (a) model-side respell/split work,
  (b) M1 delivery being unmeasured (bert + two packed encoders), and
  (c) the pad-region parity question at the S boundary.
* Recommended path: **A now** (probe-local wrap first, like
  `tts_kokoro_packed` — measure M1/M2 parity + timing before any
  package work), **C later** if W-polymorphism proves valuable.

## Reproduce

```sh
# shape census + waste tables (GPU, ~6 min, fp32)
.venv/bin/python /tmp/kokoro_shape_probe.py          # ephemeral scratch
.venv/bin/python /tmp/kokoro_bucket_analysis.py
# export spikes (CPU, seconds)
#   static-W tail export; dynamic-W export via Dim("W") on a (1,W) carrier
# per-bucket build cost: /tmp/kpb_{dec,f0n}.log (search+lower+regionfuse ~5-8 s/arm)
```

Scratch probes are in `/tmp` (ephemeral by design); promote to
`bench/drivers/probes/` only if the split wrap graduates past the
measurement stage.

# TTS/STT coverage map — ranked worklist (2026-11 session)

Source: read-only investigation across kokoro/whisper probes + reports.
Full detail in session transcript; key tables preserved here.

## Kokoro opaque census (complex decoder): 87 opaque / 1090 ops = 8%

| leaf | count | time share (~123ms eager) |
|---|---|---|
| aten::_weight_norm | 70 | 6.36ms ≈ 5.0% |
| aten::stft+istft | 2 | ~1.9ms ≈ 1.5% |
| upsample_*1d/conv_transpose1d/add_/angle/rand_like | ~16 | ~1-2% combined |

Verdict: opaque coverage is NOT the wall — conv/resblock stack ~93%,
delivered at eager parity (123.09 vs 123.54ms, verified). compile ~70ms
= fusion floor, same class as whisper decode.

## Ranked worklist

| # | wall | share | fix class | payoff |
|---|---|---|---|---|
| 1 | conv/resblock fusion floor (no non-root carrier executor / pointwise-fusion carrier / Triton epilogue at scale) | ~53ms TTS / ~1.2ms/tok STT | carrier+orchestrator, LARGE | largest prize |
| 2 | _weight_norm x70 | 5% decoder | binding OR const-fold at lower (pure fn of params) | ~5% + general unblocking |
| 3 | stft/istft | 1.5% | new IR op + binding (spec in tts_depth_r2.md) | 1.5% + law reach |
| 4 | PackedSequence custom_op upstream | enables textenc+full | vendor probe pattern as torch_bridge helper | coverage prerequisite |
| 5 | data-dependent shapes (arange/repeat_interleave/alignment) | blocks 100% end-to-end kokoro | padded reformulation, model-side wrap | unlocks single-call TTS |
| 6 | upsample1d/conv_transpose1d/add_/angle/rand_like | ~1-2% | small bindings | hygiene |
| 7 | F.pad 1-D reflect | isolated arm | manual reflect impl | CustomSTFT arm |
| 8 | repeat_interleave | 0 measured | binding for static repeats | needed only if #5 |
| 9 | eqsat typing._shape_of concat numel-0 poison | decode step-1 law veto | eqsat one-rule fix | unblocks SDPA_CAT_LAWS class |
| 10 | alias-aware CUDA-graph capture | -0.9ms/tok measured | mlopt_cuda.runners feature | ~15% of step time |
| 11 | aten::nonzero_numpy | unmeasured | binding/respell | hygiene |
| 12 | verify weight-dup OOM (turbo 4GB) | blocks verified turbo | streaming verify | verified turbo delivery |

## Whisper post-sdpa status

- Encoder: ~1.16x eager / compile parity. Decode step best arm 4.34ms/tok
  vs eager 3.6 / compile 3.2 (ids 32/32). Remaining walls structural:
  fusion floor, shape_of numel-0 poison (#9), alias-aware capture (#10).
- bf16 hard wall for kokoro end-to-end (cuDNN LSTM rejects bf16) — fp32 honest.

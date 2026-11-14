# Kokoro TTS probe — round 2 (PackedSequence wrap + decoder coverage audit)

Follow-up to `tts_kokoro_r1.md` / `tts_walls_fixed.md`.  Two bounded
pieces: (1) the recommended `pack → lstm → pad` `custom_op` wrap
(walls-fixed option 1) applied to the three Kokoro sites, and
(2) a coverage audit of the decoder's complex-`stft`/`istft` opaque
leaves.  Same hardware (RTX 2050 4 GB shared, fp32 — bf16 still
cuDNN-blocked), same probe pipeline.

**Headline correction first:** `_harvest` had the two decoder arms
**swapped** since r1 — `which == "decoder"` loaded
`disable_complex=True` (CustomSTFT) and `"decoder-nc"` loaded the
complex TorchSTFT variant.  Verified by inspecting
`m.decoder.generator.stft`: `decoder → CustomSTFT`,
`decoder-nc → TorchSTFT`.  Consequences for r1's record:

- r1's "decoder delivered 0.89× with opaque `stft`/`istft` leaves"
  was actually the **CustomSTFT** arm — its opaque leaves were
  `_weight_norm`/`conv_transpose1d`/`upsample`, never `stft`.
- r1's "decoder-nc unary-`mul` wall" was actually the **complex**
  graph — consistent, since the `phase * 1j` defect lives in
  `TorchSTFT.inverse` (`magnitude * torch.exp(phase * 1j)`).
- The probe is fixed (`disable_complex = which == "decoder-nc"`,
  matching the docstring and `bench()`); all r2 numbers below are
  under the corrected mapping.  r1's timing rows remain valid *per
  variant*, only the labels were inverted.

## Piece 1 — PackedSequence `custom_op` wrap: textenc LIFTS, verifies, pays

New helper `bench/drivers/probes/tts_kokoro_packed.py`
(probe-local, no package edits).  Each of the three
`pack_padded_sequence → lstm → pad_packed_sequence` sites
(`kokoro/modules.py:60-63` TextEncoder, `:113-116`
ProsodyPredictor, `:164-169` DurationEncoder — all `batch_first`,
all discarding hidden state) is wrapped as a
`torch.library.custom_op("tts_packed::lstm_N", mutates_args=())`
taking `(x, lengths)` dense tensors and returning the padded output.
The op closes over its `nn.LSTM` (weights stay in `state_dict` /
`param_values`; the delivered module replays the same module through
the opaque leaf).  Patched forwards are verbatim copies of the
kokoro originals with only the triple replaced; `install(model)` is
called from `_harvest` via a `-p` suffix on `which`
(`opt textenc-p`, `opt full-p`).

Two documented semantics notes:

- `pad_packed_sequence` is called with `total_length=T` instead of
  the default `max(lengths)` — the default is *data-dependent*, an
  output shape `torch.export` cannot determine statically (half of
  why the raw call walls).  The extra tail is zeros and every
  downstream consumer re-masks the tail (`masked_fill_` / zero
  `x_pad` in all three sites), so delivered values are identical —
  verified bitwise below.
- Custom ops mint `call_function` nodes that `_opaque_qual`
  resolves as `tts_packed::lstm_N` through `torch.ops` — the
  predicted opaque-leaf path works exactly as walls-fixed §option-1
  described; no `torch_bridge` change was needed.

Unit check before the real run: op vs raw triple on random tensors —
**bitwise identical** on the overlap, zeros in the tail, and
`torch.export` emits a single `call_function tts_packed.lstm_test1`.

### `opt textenc-p fp32` — FULL PASS

```
search 0.3s          (was: export wall in 0.3-1.3s)
lower+verify 0.1s    verified=True
delivered-vs-eager max|diff| 0.0000e+00   (BITWISE, seeded AND unseeded)
delivered 3.04 ms    vs eager-wrap 4.44 ms  → ~1.46×
graph cuda_graph=False fail=kernel         (same refusal class as before)
```

The conv+LSTM text encoder — previously 100% unliftable — now goes
through the whole pipeline deterministically.  The packed-LSTM
regions are opaque leaves (laws can't reach inside), but the
surrounding conv stack/LayerNorm/masking is fully optimizable.

### `opt full-p fp32` — PackedSequence wall GONE, deeper wall behind it

`forward_with_tokens` now gets *past* every LSTM site but dies in
`torch.export` at `model.py:101`:

```
GuardOnDataDependentSymNode: Could not extract specialized integer
from data-dependent expression u0
  text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1))
  # arange(input_lengths.max()) → SymInt u0
```

This is a **different wall class**: data-dependent shapes, not the
pytree.  `input_lengths.max()` feeds `torch.arange` (mask width),
and downstream `repeat_interleave(pred_dur)` +
`pred_aln_trg[indices, ...]` make the alignment/acoustic lengths
genuine runtime data — the IR has static shapes only.  Honest
verdict: `full` stays unliftable end-to-end; the packed wrap removed
its first wall and exposed the structural one.  The text-side heads
that *can* lift (`textenc`, `f0n`, harvested decoder inputs) are
unaffected.

## Piece 2 — decoder coverage audit (corrected arms)

### What is actually opaque (complex `decoder`, `export_to_ir` census)

1090 dedup'd IR ops; **87 opaque = 8.0%**.  By qualname:

| opaque leaf | count | share of opaque |
|---|---|---|
| `aten::_weight_norm` | **70** | **80%** |
| `aten::add_.Tensor` | 4 | |
| `aten::upsample_{nearest,linear}1d.vec` | 4 | |
| `aten::conv_transpose1d` | 3 | |
| `aten::istft` / `aten::stft` | 1 + 1 | |
| `aten::{multiply.Tensor, angle, rand_like, randn_like}` | 4 | |

The complex `stft`/`istft` are **two leaves of 87** — coverage there
is nearly irrelevant to this graph.  The opaque mass is
`_weight_norm` ×70: every `weight_norm`-parametrized conv
recomputes its normalized weight *every forward* (eager does the
same — it is a parametrization, not a cached transform).

### Where the time actually goes (eager, clean GPU)

| region | ms/call | % of ~123-126 ms eager |
|---|---|---|
| `torch.stft` | 0.50 | 0.4% |
| `torch.istft` | 1.42 | 1.1% |
| `torch._weight_norm` ×70 | 6.36 | **5.0%** |
| everything else (bound conv/linear/resblock stack) | ~115 | ~93% |

(stft/istft measured by monkeypatching `torch.stft`/`torch.istft`;
`_weight_norm` likewise.  An earlier run under host contention read
41 ms — discarded; clean re-measure is 6.4 ms.)

**Verdict: opaque coverage is NOT the decoder's wall.**  All opaque
leaves together account for roughly **6-8% of eager time**; the
bound conv/resblock stack is ~93% and simply has no profitable
rewrite on this card — delivery lands at eager parity, not above it.
Real bindings for `stft`/`istft` would buy ~1.5%; a `_weight_norm`
binding (or better, a param-fold precompute — the normalized weight
is a pure function of `param_values`) is worth ~5% and is the
largest single opaque cluster by op count.

### Post-fix pipeline results (both decoder variants)

| arm | search | verify | seeded parity | delivered | eager | ratio |
|---|---|---|---|---|---|---|
| `decoder` (complex TorchSTFT) | 4.3 s | **True** | 0.0 bitwise | 123.09 ms | 123.54 ms | **1.00×** |
| `decoder-nc` (CustomSTFT) | 38.3 s | **True** | 2.98e-08 | 123.43 ms | ~123.4 ms | **~1.00×** |

Both now deliver at **eager parity** — r1's 0.89× was measured
pre-`torch_bridge`-fix (and on the swapped arm).  `verified=True`
on both this round (r1's `verified=False` was wall-A stochastic
noise on the mislabeled arm; unseeded run-to-run floors are
unchanged ~6e-2).  `CudaGraphRunner` still refuses (`fail=kernel`)
on both — same class as whisper/t5 encoders on this card.  Neither
approaches `torch.compile` (~70 ms, 0.57× of delivered).

## Complex-op coverage assessment

**(a) Does verify handle complex dtype?**  Yes at the comparison
level — nothing complex ever reaches it today, but if it did:
`report._verify_pair` computes `(ref - out).abs().max()` (complex
abs → magnitude, fine); `_dtype_rtol_floor` catches the
`torch.finfo(complex)` `TypeError` and contributes 0;
`_flatten_outputs` is dtype-agnostic; `export._diff` already
special-cases `is_complex()`.  The gap is entirely on the
binding/export side, not the verifier.

**(b) Real `stft`/`istft` bindings — feasible WITHOUT editing
`torch_bridge.py`, verified probe-side.**  `_IR_TO_TORCH` is a
*deliberately mutable* ambient dict (torch_bridge.py:2847 —
"`_IR_TO_TORCH[op] = fn` overrides keep reaching already-built
tables"), and `export_to_ir` mints a real `Op` for any name present
in it (line ~676 `ir_op not in _IR_TO_TORCH → opaque`).  Patching
`tb._IR_TO_TORCH["stft"] = fn` from probe code made `export_to_ir`
mint a real `stft` op on a toy module (verified — no `opaque`
qualname).  The proper upstream fix is still a `torch_bridge`
entry, but the binding spec is:

- `stft`: FX emits `aten.stft(x, n_fft, hop_length, win_length,
  window, …)` — tensor operands `x` and `window`; scalar positionals
  land in `attrs["arg1".."argN"]` (n_fft/hop/win_length ints,
  `return_complex`/`center`/`boundary`/`normalized` bools) and
  string kwargs (`pad_mode`) in attrs.  Binding shape:
  `lambda x, win, **kw: torch.stft(x, int(kw["arg1"]), …,
  window=win, return_complex=…, pad_mode=kw.get("pad_mode"))` —
  output complex64/128 flows to bound `abs` (line 2391,
  `torch.abs` → magnitude) and `angle` (currently `aten::angle`
  opaque; would want `"angle": torch.angle` too).  `istft` is the
  mirror: complex operand in, real waveform out; its input complex
  is built by bound `mul`/`exp` on `Const(1j)` (the post-fix
  complex-scalar channel).  No IR/schema changes needed — complex
  rides operands like any tensor.
- Honest caveat: verified at *export* level only (op mints as
  `stft`, not `opaque`); full lower+deliver of a bound-stft decoder
  was not run — worth ~1.5% of decoder time, so low priority vs the
  `_weight_norm` cluster.

**(c) Cheap wins ranked by measured decoder time:**

1. `_weight_norm` ×70 ≈ **5%** of eager — either a real binding
   (`torch._weight_norm(v, g, dim)` replay) or, better, a
   non-local const-fold: normalized weights are pure functions of
   `param_values`, so they could be precomputed once at lower time
   instead of recomputed 70×/forward.  Largest single opaque
   cluster (80% of opaque ops).
2. `stft`+`istft` ≈ **1.5%** — bindings spec'd above; low value.
3. `upsample_*1d` ×4, `conv_transpose1d` ×3, `add_` ×4, `angle`,
   `rand_like`/`randn_like` — remainder, ~1-2% combined.

## Files

- `bench/drivers/probes/tts_kokoro_packed.py` — NEW: the
  `custom_op` wrapper + patched forwards (`install(model)`).
- `bench/drivers/probes/tts_kokoro_probe.py` — `-p` suffix plumbing
  in `opt()`/`_harvest()`; **decoder-arm swap fix**
  (`disable_complex = which == "decoder-nc"`).
- No package source touched.  `pytest` smoke (81 tests) passes.

## Reproduce

```sh
.venv/bin/python bench/drivers/probes/tts_kokoro_probe.py opt textenc-p fp32
.venv/bin/python bench/drivers/probes/tts_kokoro_probe.py opt full-p fp32
.venv/bin/python bench/drivers/probes/tts_kokoro_probe.py opt decoder fp32     # complex
.venv/bin/python bench/drivers/probes/tts_kokoro_probe.py opt decoder-nc fp32   # CustomSTFT
```

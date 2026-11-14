# Whisper "lift any model" experiment

Real-world test of the claim that `TorchSource.to_ir` +
`Optimizer.optimize` + `CudaGraphRunner` can take an arbitrary HF
model and deliver a faster verified module.  Target: OpenAI Whisper
on this machine's **4 GB RTX 2050**, torch 2.14.0+cu130,
transformers 5.23.0 (installed into `.venv` for this probe —
`transformers`, `soundfile`, `huggingface-hub`; no faster-whisper/
ctranslate2 baseline was installed and none was added).

Probe script: `bench/drivers/probes/whisper_probe.py`
(scratch-quality, uncommitted-safe).  Inputs are synthetic
log-mel features `(1, n_mels, 3000)` — 30 s of audio, batch 1 —
not a real waveform; timings are forward-pass only, no
token-generation loop.

## What was tried

| checkpoint | dtype | device | role |
|---|---|---|---|
| `openai/whisper-large-v3-turbo` (809 M) | bf16 | CUDA | largest that loads (~1.75 GB) |
| `openai/whisper-small` (242 M) | fp32 | CUDA / CPU | middle rung |
| `openai/whisper-base` (74 M) | fp32 | CUDA | largest that completes the pipeline |

## Baselines (isolated forward, warmup 5 / reps 30, bf16 turbo / fp32 others)

| arm | whisper-turbo bf16 | whisper-small fp32 | whisper-base fp32 |
|---|---|---|---|
| encoder eager | 493.7 ms | 237.6 ms | 67.6 ms |
| encoder `torch.compile` | 424.0 ms | 236.1 ms | 57.1 ms |
| decoder step eager (seq=1, no cache) | 8.5 ms | 29.7 ms | 7.4 ms |
| decoder step `torch.compile` | 7.4 ms | 27.3 ms | 6.3 ms |

This GPU is compute-bound (weak SM count, no bf16-tensor-core
advantage visible); `torch.compile` buys ≤15 % on the encoder and
~12 % on the decode step.  Peak memory: turbo bf16 ~1.75 GB,
small fp32 ~0.92 GB weights.

## Pipeline results

### whisper-base fp32 — the only full pass

- `opt.search(model.model.encoder, x, max_iterations=3)`:
  **3.0 s**, e-graph saturated at 620-enodes-class scale.
- `opt.lower(..., verify=True)`: **verified=True** (0.1 s).
- Delivered module: **59.8 ms** vs eager 67.6 ms (**1.13×**) and
  vs compile 57.1 ms (**0.95× — slower than Inductor**).
- `CudaGraphRunner` delivery: `cuda_graph=True`, replay
  **59.9 ms — no change**.  Expected: graph capture collapses
  *dispatch* overhead; the encoder is GEMM-bound, and per-op
  dispatch is already amortised over 1500-token matmuls.

### whisper-small fp32 — search OK, delivery OOMs

- Search on GPU: 26.0 s.  `lower(verify=True)` OOMs: the verify
  gate materialises a fresh reference lowering *and* the delivered
  module, and `res` itself retains the exported leaf tensors —
  ~3× the 0.97 GB fp32 weight set plus activations exceeds 4 GB.
- `lower(verify=False)` fits (post-lower 1.01 GB allocated) but the
  **first forward OOMs**: the generic `IRModule` eval tape keeps
  every intermediate slot tensor alive until the root returns
  (`_EvalPlan.v` is never refcounted/freed), so peak memory scales
  with *graph size*, not just weights.  ~3.7 GB in use at the first
  softmax.  CUDA-graph lane reports `cuda_graph=False,
  fail=oom`.
- CPU-side verify is not a workaround: one fp32 CPU forward of the
  small encoder is tens of seconds and `lower`'s verify path did
  not finish inside 40 min.

### whisper-turbo bf16 — export OK, search blocked

- `torch.export` of `model.model.encoder` on bf16 CUDA:
  **succeeds in 1.6 s** (487 params).  The decoder needs a wrapper
  returning `.last_hidden_state` (see below); the wrapped decoder
  exports in 0.6 s.
- `opt.search` dies inside `share_duplicate_param_slices`
  (`mlopt_core/laws/pairing.py:873`): `.numpy()` on a **BFloat16**
  tensor — `TypeError: Got unsupported ScalarType BFloat16`.  So
  bf16 — the only dtype turbo fits in 4 GB — cannot even reach
  saturation through the stock path.
- Whole-model export (`WhisperForConditionalGeneration` wrapped to
  return `.logits`): **OOM during `torch.export`** itself on 4 GB
  (needs ~3.7 GB already resident for weights+tracing).

### Decoder (whisper-base) — search does not terminate

`opt.search(_DecWrap(whisper-base.decoder), (start_ids, ehs),
max_iterations=3)` ran **>20 min single-threaded CPU and was
killed** (e-graph saturation over the decoder's cross-attention +
self-attention IR).  The encoder saturated in 3 s; the decoder's
IR is structurally much larger per weight.  This is the sharpest
"any model" boundary found: **lifting works, search does not
scale to the seq2seq decode graph.**

## Export-boundary findings (the honest "where it breaks")

1. **Decoder return type.** `model.model.decoder(...)` returns
   `BaseModelOutputWithPast` carrying an `EncoderDecoderCache` —
   `torch.export` refuses it ("not a known type … register a
   pytree").  A 3-line wrapper returning `.last_hidden_state`
   exports cleanly.  Calling the decoder positionally also
   mis-binds `encoder_hidden_states` to `attention_mask` — kwargs
   required.
2. **bf16 unsupported in a non-local pass.** `.numpy()` in
   `share_duplicate_param_slices` — bf16 blocks search entirely.
3. **Non-float tensor leaves stay on CPU.** `IRModule._align_to`
   (`torch_bridge.py` ~line 3360) moves only `is_floating_point()`
   baked tensors to the input device/dtype; lifted integer consts
   (the positional-embedding `arange` index feeding
   `nn.Embedding`) remain CPU → `Expected all tensors to be on the
   same device` at first eval on CUDA.  The probe monkeypatches
   `_align_to` to move *all* tensor leaves; upstream should fix
   `_align_to`/`_mint_const` to cover non-float tensor operands.
4. **`IRModule` tape retains all intermediates** (see small-fp32
   OOM) — fine for zoo-scale terms, a wall for 100 k-node models.
5. **KV-cache / `generate()` is out of scope for export**, as
   expected: `past_key_values` mutation and data-dependent loop
   control are not exportable; only the stateless single-step
   forward (no cache) was lifted.  CudaGraphRunner therefore could
   only be measured on the encoder (compute-bound → no gain) — the
   dispatch-bound decode-step win it was designed for could not be
   reached because the decoder IR never saturates.

## model_corpus on Whisper

`python -m mlopt_discovery.model_corpus --module <whisper-base
encoder callable> --mutants 0 --no-probe --corpus bench` produced
**no report in 30 min** (killed).  Even with mutant generation and
probing disabled, `model_board → play._case_tail` runs
`_mint_tiers` + `genkernel.profitable` — the measured-cost
referee evals every minted claim `gen_reps` times — over an IR
with thousands of nodes.  The corpus machinery is zoo-scale; a
real HF encoder does not fit its time budget.  (The callable had
to expose `model.model.encoder` + a features tensor because
`_as_dtype` casts *all* feed tensors to the intake dtype, which
would corrupt the decoder's integer `input_ids`.)

## Verdict for "any model"

- **Lift**: mostly true.  `torch.export` handles the encoder and a
  wrapped decoder step of Whisper without opaque-leaf help; the
  failures are HF-API shape issues (cache return type, kwarg
  binding), not control flow — *for the stateless forward*.  The
  full generative model (cache mutation, `generate` loop) is
  categorically outside what export sees.
- **Optimize**: delivered a *verified* module on whisper-base
  encoder (1.13× vs eager, 0.95× vs `torch.compile`) — honest
  parity-to-slight-win, not a speedup story on this GPU.
- **The walls are downstream of lifting**: bf16 in the law bank,
  non-float leaf device placement, `IRModule` memory retention,
  e-graph search cost on the decoder, and 4 GB VRAM for
  verify-time duplication.  "Any model" currently means "any
  zoo-scale model, fp32, single stateless submodule, CUDA or
  smaller".

## Reproduce

```sh
.venv/bin/pip install transformers soundfile
.venv/bin/python bench/drivers/probes/whisper_probe.py bench     # baselines
.venv/bin/python bench/drivers/probes/whisper_probe.py opt encoder
.venv/bin/python bench/drivers/probes/whisper_probe.py opt decoder  # >20 min, killed
```

Raw logs from this run: `/tmp/whisper_exp/{enc3,base,base_dec,
small_nv,corpus0}.log` (ephemeral).

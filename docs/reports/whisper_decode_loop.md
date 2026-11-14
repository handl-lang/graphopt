# Whisper decode/generation loop — first end-to-end measurement

Follow-up to `whisper_experiment_r2.md` / `verify_lowmem.md` /
`decoder_law_coverage.md`.  Prior rounds lifted the encoder and proved
the per-layer spelled-attention→`sdpa` folds on ONE decoder layer
(CPU, ~20–28 %/layer).  The autoregressive decode loop — token-by-token
with KV cache — had never been measured through the pipeline.  This
report measures it honestly, end-to-end, on the same box as r2
(RTX 2050 4 GB, torch 2.14.0+cu130, transformers 5.23.0,
`openai/whisper-base` fp32, synthetic `(1,80,3000)` log-mel, greedy
decode, 32 tokens).

Probe: `bench/drivers/probes/whisper_decode_loop.py`
(`eager|loop [iters] [ntok] [L] [cpuparams]`).

## What the pipeline can and cannot see

`model.generate()` / the HF decode loop is control flow plus mutable
cache state — not a static tensor function — so it cannot be handed to
`Optimizer.search` (same refusal class as the r2 whole-decoder wall,
one level up).  What *can* be optimized is one decode **step** as a
pure function, and the KV-cache append (`cat(past, new)`) can be made
static-shape:

- `_DecodeStepWrap` wraps all 6 `WhisperDecoderLayer`s + final
  `layer_norm` + `proj_out` (lm head) as one module.  Inputs: post-embed
  `hs` (1,1,512), a float additive `mask` (1,1,1,L+1), and 24 past
  tensors — per layer self-attn K/V `(1,8,L,64)` and cross-attn K/V
  `(1,8,1500,64)`.  Cross-attn K/V are constant after step 0 (the
  steady-state cache-read path, forced via `is_updated`), exactly what
  real decode steps ≥ 1 do.
- **Fixed-capacity padded buffer.**  Self-attn past lives in an
  `L`-slot buffer, zero-padded; the step's new K/V is appended at index
  `L` by the layer's own cache `cat`, and the mask (a module *input*,
  rebuilt per step by the caller) blocks pad slots.  Attention is a
  masked weighted sum over the valid key set — a permutation-invariant
  softmax — and whisper has learned absolute positions (no RoPE), so
  padding is **mathematically exact**, not an approximation.  The
  caller writes each step's new K/V into slot `t` eagerly (12 small
  copies) — cache bookkeeping stays outside the optimized module,
  which is honest: it is mutation/control flow, not matmul.
- Correctness gate before any timing: the wrap reproduces the harvested
  real decode step — **logits max|diff| ≈ 6e-6, argmax identical** —
  and every delivered arm below generates the **identical 32-token
  sequence** as eager HF (`ids-match 32/32`).

## Measured (whisper-base fp32, 32-token greedy loop, warm = excl. first-step warmup)

Pipeline: `search` 7.5 s; `lower(verify=True)` 0.4 s, **verified=True**
(fp32 passes the pointwise gate, consistent with r2/verify_lowmem).

| arm | step ms (fixed L=40) | loop warm ms/tok | loop last-8 ms/tok | ids vs eager |
|---|---|---|---|---|
| eager HF decoder+head (persistent cache) | ~3.56 (kv 1→32) | **3.59** | **3.56** | — |
| eager `_DecodeStepWrap` (per-call cache build — overestimate) | 4.01 | — | — | — |
| delivered `IRModule` | 5.39 | 5.99 | 5.55 | 32/32 |
| delivered + `CudaGraphRunner` | 6.10 | 6.16 | 6.17 | 32/32 (post-fix) |
| `torch.compile(wrap)` | 2.98 | **3.24** | **3.18** | 32/32 |

First decode call costs ~0.45–0.9 s one-time warmup in *every* arm
(cudnn/cublas init, `IRModule._align_to`, compile) — excluded from
"warm" numbers above; raw totals are in the probe output.

## Verdict

**The decode step lifts, verifies, and runs inside the real loop —
but on this shape it is slower than eager.**  The delivered module is
~1.6× slower per step than the HF decoder (5.5–6.0 vs ~3.6 ms); graph
replay is slower still; `torch.compile` on the same wrap is ~10 %
*faster* than eager (3.2 vs 3.6 ms/tok).  The per-layer CPU win
(20–28 %) does not transfer: at seq-len-1 decode the step is
launch/dispatch-bound, not attention-bound, and `IRModule`'s plan-tape
eval over 26 inputs / 13 outputs costs more than the `sdpa` folds
save.  The fixed-L attention also does slightly *more* work than eager
at early steps (41 keys vs t+1), so the gap is not flattered.

Scope notes: embed+pos and the 12 K/V buffer writes are timed *outside*
the delivered step (~0.05 ms combined — negligible, but noted); eager
includes them inside `decoder.forward`.  Only whisper-base fp32 was
run; the picture at longer decode (kv→150+) may differ since eager
attention grows with prefix while the delivered step is L-fixed — but
at these sizes GEMMs dominate either way.

## Found and fixed: `_GraphReplayModule` copied inputs backwards

The cuda-graph arm initially produced **input-insensitive stale
outputs** — every call replayed the capture-time inputs, and the
caller's tensors were silently overwritten with them.  Root cause,
`packages/mlopt-cuda/src/mlopt_cuda/runners.py` `_GraphReplayModule.
forward`: the refresh loop zipped `(xs, self._graph_ins)` and ran
`buf.copy_(t)` — i.e. `new_input.copy_(static)`, backwards.  The
carrier executors (`om_lower`/`scan_lower`/`omd_lower`/`decayed_lower`)
zip `(self._graph_inputs, xs)` correctly; only the generic
`IRModule` path was wrong.  Effects: frozen replay + in-place
corruption of caller inputs, and it *masked itself in tests* —
`out(x2)` corrupted `x2` to capture values before `mod(x2)` compared,
so `test_cuda_graph_runner_captures_generic_irmodule` passed vacuously.

Fix: swap the zip order (`self._graph_ins, xs`) — one line.  Strengthened
the test to assert inputs are unmutated, fresh inputs give fresh
outputs, and the replayed result differs from the capture-input result.
Post-fix minimal repro: replay tracks inputs (diffs exact), no caller
mutation; `tests/test_runners.py` 38/38 pass.  **Impact on prior
claims:** every generic-path `CudaGraphRunner` delivery (any
`IRModule` — the whisper encoder arm included) was silently computing
on capture inputs.  Timing numbers stand (replay speed is
input-independent at fixed shapes); any correctness-dependent use was
broken.  The "graph capture beat compile" frontier rows measured
speed only, so they survive — but they should be re-verified for
correctness before being cited.

Post-fix graph arm on the decode step: correct (`ids 32/32`) but
*slower* than plain delivery (6.16 vs 5.99 ms/tok) — 26 input
`copy_`s + replay overhead exceeds the launch savings at this size,
consistent with the encoder finding (GEMM-bound, dispatch already
amortised).

## Honest limits

- The "end-to-end" claim is: delivered decode *step* inside the real
  greedy loop with eager bookkeeping — not `generate()` itself, which
  remains outside the pipeline's reach (control flow, mutation,
  growing shapes).  The fixed-buffer reformulation is exact for
  whisper specifically; models with in-attention position encoding
  (RoPE) or sliding-window/ALiBi masks would need per-step mask
  semantics re-derived.
- `lower(verify=True)` compares delivered vs a fresh lowering of the
  *same IR* on the harvest inputs — it does not see the t=0 degenerate
  mask or later steps.  The loop-level `ids-match 32/32` is the real
  correctness evidence here.
- `_DecodeStepWrap` constructs a fresh `EncoderDecoderCache` per call
  (what `torch.export` can trace); the eager-wrap timing therefore
  overstates a "real" eager step — the HF loop number is the baseline.
- GPU was shared during measurement; run-to-run eager step varied
  ~3.4–5.3 ms across runs.  The ordering delivered > compile > eager
  was consistent in the clean final run; absolute numbers ±30 %.

## Reproduce

```sh
uv sync
.venv/bin/pip install transformers   # scratch dep, pruned by uv sync
.venv/bin/python bench/drivers/probes/whisper_decode_loop.py eager 32
.venv/bin/python bench/drivers/probes/whisper_decode_loop.py loop 3 32 40
```

Files touched: `bench/drivers/probes/whisper_decode_loop.py` (new
probe), `packages/mlopt-cuda/src/mlopt_cuda/runners.py` (one-line
copy-direction fix), `tests/test_runners.py` (strengthened generic
replay test: non-mutation + fresh-input correctness).

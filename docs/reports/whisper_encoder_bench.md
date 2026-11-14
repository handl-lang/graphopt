# Whisper ENCODER lift + optimize benchmark

Date: 2026-11-14.  Hardware: RTX 2050 4 GB (**shared GPU** — absolute ms
swing run to run; replicate spreads reported honestly).  torch
2.14.0+cu130, **transformers 4.57.1** via `/tmp/whisper_deps` +
`PYTHONPATH` (same scratch install as `whisper_production_bench.md` —
`uv sync` prunes non-lockfile installs).  Probe:
`bench/drivers/probes/whisper_encoder_bench.py` (new, uncommitted
scratch; reuses `whisper_prod_bench.py`'s feature extractor and decode
machinery by import).

Motivation: the encoder is the untouched half of whisper e2e — **62 %
of turbo's 810 ms** in `whisper_production_bench.md` (505 ms encoder +
~305 ms delivered decode estimate).  This probe lifts the HF
`WhisperEncoder` (conv1d frontend + pos-embed + N pre-norm transformer
encoder layers) through the normal export → `Optimizer.search` →
`lower(verify=True)` path and measures honestly vs HF eager /
`torch.compile`.

## Lift census — zero opaque leaves

`export_to_ir` on a thin `EncWrap` (encoder → `last_hidden_state`
tensor) succeeds for all three sizes; **no op mints an `opaque` leaf**.

| model | IR ops | distinct ops | opaque leaves |
|---|---|---|---|
| base (fp32, 6L d512 h8) | 208 | 15 | **0** |
| turbo (bf16, 32L d1280 h20) | 1066 | 15 | **0** |

Op histogram (turbo; base is the same shape per layer):
`linear` 192, `transpose` 160, `dropout` 129, `reshape` 128,
`contiguous` 128, `add` 65, `layer_norm` 65, `mul` 64, `matmul` 64,
`gelu` 34, `softmax` 32, `conv1d` 2, `arange` 1, `embedding` 1,
`permute` 1.  conv1d / layer_norm / gelu / softmax / sdpa-shaped
attention all bind — the encoder is fully inside the lowering
vocabulary.  (First turbo export attempt failed on a dtype mismatch —
probe-side fix: cast `input_features` to the model dtype, exactly like
HF's own `encoder_out` helper.)

## Correctness architecture (important for reading the numbers)

`lower(verify=True)` gates **delivered vs a fresh lowering of the
un-optimized exported IR** — *not* vs HF eager.  So this probe adds a
second, explicit HF-faithfulness check: `max|diff|` of each arm's
output vs a snapshotted HF encoder output.  Two findings make that
check load-bearing here (below).

## Results — encoder forward, RTX 2050 4 GB

| arm | base fp32 (6L) | small fp32 (12L) | turbo bf16 (32L) |
|---|---|---|---|
| HF eager encoder | ~67–68 ms (one 86 ms cold) | ~248 ms | ~495–508 ms |
| search time (`max_iterations=3`) | 3.7 s | 27.6 s | **153.9 s** |
| e-nodes / classes | 326 / 314 | 632 / 608 | 1652 / 1588 |
| `lower(verify=True)` | **verified=True** | **verified=True** | **OOM — see below** |
| delivered `IRModule` | **59.1 ms** | **~218 ms** | ~300–308 ms ⚠ wrong |
| delivered + `CudaGraphRunner` | ~59–63 ms | ~221–226 ms | not run (OOM in crowded process) |
| `torch.compile(HF encoder)` | ~58.5 ms | ~248–251 ms | OOM |
| delivered-vs-HF `max|diff|` (rel) | 9.3e-4 (**4.0e-5**) | 5.7e-4 (**1.9e-5**) | **4.13 (0.42)** ⚠ |
| IRref-vs-HF `max|diff|` | **0.000e+00 — bit-exact** | bit-exact | bit-exact |

Reps: warm 2 + 3 timed, `cuda.synchronize` around each.  `verify=True`
passed for base and small; delivered beats HF eager by **~13 % (base)
/ ~12 % (small)** — real but modest, and roughly at `torch.compile`
parity on base (58.5 vs 59.1 ms).  `delivered+graph` shows **no gain**
on either fp32 model (the encoder is GEMM-bound; launch overhead is not
the bottleneck).  Peak VRAM during small's full search+verify:
3.46 GiB.

## Honest negatives — turbo does NOT deliver on this GPU

Two independent blockers, both measured:

1. **`lower(verify=True)` OOMs on 4 GB even with the HF encoder
   dropped.**  After `Optimizer.search` (which itself fits — peak
   ~3.35 GiB process-wide), the verify phase lowers *two* `IRModule`s
   over the shared 1.27 GB bf16 `param_values` and runs both forwards;
   the process sat at 3.58–3.72 GiB allocated and died allocating
   ~14–86 MiB (`OptimizationResourceError: OutOfMemoryError`).
   `expandable_segments:True` did not save it.  `lower(verify=False)`
   succeeds, and a lone delivered forward fits comfortably
   (**peak 1.23 GiB**, ~300 ms).

2. **The delivered term is WRONG on turbo — and on small-bf16.**
   `delivered-vs-IRref max|diff|` = **4.129 (rel 0.42)** for turbo,
   **3.398 (rel 0.11)** for whisper-small cast to bf16 — vs **9.3e-4
   (rel 4e-5)** for base fp32 and **bit-exact** whenever the
   `attn_sdpa_*` carrier laws are excluded from the rule set
   (ablation: `diag turbo 3 attn_sdpa` → `delivered-vs-IRref =
   0.000e+00`).  Excluding `naturality_scalar_rev` instead leaves the
   bug intact → **`attn_sdpa_mul_drop` alone is sufficient to produce
   the divergence** (it is also the only sdpa-family row that fires:
   whisper encoder self-attention is mask-free with an eval-dropout
   wrapper).

   What was ruled out:
   * **Export/IR fidelity is perfect** — the un-optimized IR lowered
     fresh is *bit-exact* vs HF eager on all three models.
   * **The sdpa attrs are right** — all 32 turbo `sdpa` nodes carry
     `scale=1.0`, which is *correct*: transformers 4.57's eager
     attention pre-scales Q by `head_dim**-0.5 = 0.125` before the
     `q@kᵀ` matmul and applies a no-op `*1.0` after; the fold binds S
     to that post-matmul Const.
   * **Not the sdpa backend** — forcing `SDPBackend.MATH` changes the
     diff by <2 % (4.13 → 4.25).
   * **Not `naturality_scalar_rev`** — it fires 32× and the result is
     bit-exact without the sdpa laws.
   * Single-layer standalone check (bf16, whisper shapes): sdpa-fold
     vs spelled softmax attention differs only ~1.3e-2 rel — normal
     bf16 kernel noise.

   Best current interpretation: **bf16 sdpa-vs-spelled-softmax
   divergence compounds over depth** — the trend is monotone in
   depth×dtype (6L fp32 → 4e-5, 12L bf16 → 0.11, 32L bf16 → 0.42).
   Whether that is acceptable "numerics" or a subtle eval/law bug at
   these shapes needs law-level investigation — **flagged for
   parent**: the `attn_sdpa_*` fold family (`mlopt-carriers
   decode_laws.py`) delivers wrong encoder output under bf16, and
   `verify=True` is the only thing standing between that and a bad
   delivery.  Note the verify gate *would* have caught it — it simply
   never got to run on 4 GB.

## e2e composition — not measured (blocked honestly)

`e2e` mode (delivered encoder → existing delivered+graph decode loop)
was **not run to completion**: composing a known-wrong encoder would
produce meaningless `encoder_hidden_states`.  For the record, *if* the
sdpa-fold divergence were fixed, the delivered turbo encoder measures
**~300–308 ms vs ~500 ms HF eager (−40 %)** — the composed estimate
would be roughly **300 + ~211 decode + overhead ≈ 520–560 ms vs HF
eager ~810 ms e2e** (vs the current 710–720 ms estimate with an eager
encoder).  That is the prize worth chasing once the fold is fixed.

## Reproduce

```sh
uv pip install --target /tmp/whisper_deps "transformers==4.57.1"
export PYTHONPATH=/tmp/whisper_deps \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python bench/drivers/probes/whisper_encoder_bench.py census base
.venv/bin/python bench/drivers/probes/whisper_encoder_bench.py opt base 3
.venv/bin/python bench/drivers/probes/whisper_encoder_bench.py opt small 3
.venv/bin/python bench/drivers/probes/whisper_encoder_bench.py opt turbo 3   # verify OOMs; verify=False path shows the divergence
.venv/bin/python bench/drivers/probes/whisper_encoder_bench.py diag turbo 3 zzz          # dumps /tmp/enc_{term,params}_turbo.*
.venv/bin/python bench/drivers/probes/whisper_encoder_bench.py diag turbo 3 attn_sdpa    # ablation: correct
.venv/bin/python bench/drivers/probes/whisper_encoder_bench.py diag turbo 3 naturality_scalar_rev
.venv/bin/python bench/drivers/probes/whisper_encoder_bench.py termtest turbo default    # re-eval dumped term, no research
.venv/bin/python bench/drivers/probes/whisper_encoder_bench.py termtest turbo mathsdpa   # force SDPBackend.MATH
```

Artifacts (ephemeral, /tmp): `enc_term_{base,small,turbo}.pkl` +
`enc_params_{base,small,turbo}.pt` — the extracted terms and weights,
re-evaluable via `termtest` without re-running search.

## Included / excluded

* Included: real log-mel features (same `WhisperFeatureExtractor`
  synthetic-audio fixture as the prod bench), encoder-only forward
  timing, `verify=True` gate, explicit HF-faithfulness `max|diff|`,
  rule-ablation diags, dumped-term re-evaluation.
* Excluded: turbo `verify=True` (OOM — reported, not worked around),
  e2e composition (blocked by the wrong delivery), small-turbo
  fp32 casts (OOM), CUDA graphs around the delivered turbo encoder
  (moot — module incorrect; also OOM'd inside the crowded opt process).

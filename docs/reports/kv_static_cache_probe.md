# K/V-cache cat elimination — static-buffer decode step (landed)

Date: 2026-11-14.  Hardware: RTX 2050 4 GB (shared), torch
2.14.0+cu130, fp32, CUDA.  Probe:
`bench/drivers/probes/whisper_kv_static_probe.py` (new).  Implements
`docs/reports/fusion_floor_probe.md` §6.1(a) — the ~0.95 ms/call
`CatArrayBatchedCopy` prize, the single biggest item in the
whisper-base decode-step gap.

**Verdict: landed.**  All 36 `concat` steps per decode call are gone;
the delivered+alias-graph loop runs **3.65 ms/tok warm** vs the
previous best shipped arm, with **ids-match 32/32** against the real
HF greedy loop.

## Mechanism — no new IR op was needed

The spec's route (a) asked for "caller-maintained K/V buffers +
in-place `cache_write` op".  The honest version turned out cheaper:
the write survives the pipeline as an **`opaque` leaf**, not a new
binding.

* `_DecodeStepWrapStatic` (probe-local) replaces the per-call
  `DynamicCache` internals with two `CacheLayerMixin` layers:
  * self-attn `_KVWriteLayer` — prefill **adopts the caller's
    `(1,8,L,64)` buffer by reference** (no copy); the attention
    module's append does
    `keys.index_copy_(2, pos, knew)` — an in-place slot write at the
    dynamic position `pos` (a new module input, `(1,)` int64).
  * cross-attn `_KVRefLayer` — adopts the constant encoder K/V by
    reference; attention's `is_updated` path reads
    `layers[i].keys` directly, so the ~36 MB/call of cross-prefill
    `cat` copies vanish.
* `aten.index_copy_` is **outside** the export lifter's
  `copy_/fill_/zero_/masked_fill_` mutation-threading prefix, so it
  lifts as `opaque(aten::index_copy_.default)` — replayed verbatim
  at eval, a TRUE in-place write (a functionalised
  `index_put`/`slice_scatter` would have been a full-buffer copy —
  the same cost class as `cat`).  The sdpa consumer's K/V operand IS
  the opaque node's output, so the write→read order is a real DAG
  edge, not an ordering assumption.
* Attention is permutation-invariant over the mask-valid key set,
  so writing the current token at chronological slot `t` (mask
  allows `0..t`) is exactly equivalent to the old
  `cat(past, knew@L)` + `{0..t-1, L}` mask.  The module now OWNS the
  cache: the greedy loop performs **zero** K/V writes.

Term census after `search`: `concat` count **0** (was 36);
`opaque`×12 (the K/V slot writes); everything else unchanged.

## The runner gap — `mutable_inputs` (mlopt-cuda)

The alias-aware `CudaGraphRunner` demotes any aliased input whose
`_version` bumps during capture — correct for read-only buffers, but
it silently broke this mechanism: all 12 self K/V buffers demoted to
the copy path, the in-graph `index_copy_` wrote the per-call clone,
and the caller's cache never advanced (ids diverged at t=1).

`CudaGraphRunner` gained **`mutable_inputs`** (same spec shapes as
`alias_inputs`: `True` / iterable of positions / `(pos, tensor) ->
bool`): positions named there are exempt from the version-counter
demotion — the caller-owned write-back opt-in.  Threaded through
`_capture_generic`, `_GraphReplayModule.capture`, and
`_DeviceGraphModule`.  Default `None` = unchanged behaviour;
`alias_inputs` alone still demotes mutated positions (guard
verified by the pre-existing demotion test, still passing).

## Measured (whisper-base fp32, iters=3, ntok=32, L=40)

Same-process A/B, interleaved timing rounds (GPU shared — absolute
ms swing run to run; deltas replicated):

| arm | ms/call or ms/tok |
|---|---|
| old delivered (`_DecodeStepWrap` + `DynamicCache`) | 4.55–4.59 |
| **static delivered (`_DecodeStepWrapStatic`)** | **3.59–3.63** |
| static-wrap eager | 3.07 |

Full probe run (`whisper_kv_static_probe.py 3 32 40`):

| arm | warm ms/tok | ids-match |
|---|---|---|
| static-wrap (eager) | 4.08 | 32/32 |
| delivered (plan tape) | 4.42 | 32/32 |
| delivered + copy-graph | 9.44 single-step (copy path clones all 27 inputs — meaningless for the loop arm, timed for reference only) | — |
| **delivered + alias+mutable graph** | **3.65** (per-step flat 3.6–3.7 after capture) | **32/32** |

vs references: HF eager decode ~4.2 ms/tok, `torch.compile(wrap)`
~3.0 ms (prior runs).  The static+graph arm lands essentially at the
compile floor for this step — the remaining gap is GEMM + dispatch,
not cache copies.

Verify: `lower(verify=True)` passed; harvest-step logits
max|diff| = 7.6e-6 vs real HF (argmax equal); `knew`/`vnew` outputs
max|diff| = 4.8e-6 vs the old wrap (fp32 reassociation through the
different-but-equivalent attention operand order — expected, not
bitwise).

## Honest ceiling / caveats

* **Probe-local wrap.**  `_DecodeStepWrapStatic` + the two cache
  layers live in the bench driver, not `kit_torch` — they are
  whisper/HF-`Cache`-protocol-specific.  The *shipped* surface is
  just `mutable_inputs` on `CudaGraphRunner` (additive kwarg,
  signature pin updated, `docs/public_api.md` row updated).
* **`opaque` leaves are opaque.**  Laws/rewrites can't see inside the
  `index_copy_` — by design (it's a mutation, not a pure op).  Cost
  model prices it as unknown; extraction keeps it.
* **Purity contract.**  The delivered module now mutates its self
  K/V inputs — honest only because the caller owns those buffers
  exclusively (the decode loop).  `mutable_inputs` makes the same
  demand explicit at capture; the non-graph path has always allowed
  it (plan tape binds `xs[pos]` directly).
* **Position capacity.**  `pos` must stay `< L`; same capacity bound
  as the old padded-buffer scheme.
* **`index_copy_` cost.**  12 tiny scatter kernels/call
  (~0.03 ms/call measured) replace 36 full-tensor `cat`s (~0.95
  ms/call) — and under graph replay the launches are ~free.
* Cross-attn `ck`/`cv` are read-only aliases — correctly NOT marked
  mutable (they never bump `_version`).

## Files

* `bench/drivers/probes/whisper_kv_static_probe.py` — new driver
  (wrap, cache layers, masks, loop, all arms).
* `packages/mlopt-cuda/src/mlopt_cuda/runners.py` — `mutable_inputs`
  opt-in threaded through capture paths (+ docstrings).
* `tests/test_runners.py` — `test_cuda_graph_alias_mutable_input_
  keeps_write_back`.
* `tests/test_public_api.py` — signature pin extended for the
  additive kwarg.
* `docs/public_api.md` — runner row updated.
* `docs/reports/kv_static_cache_probe.md` — this report.

No `kit-torch`, `mlopt-core`/eqsat, discovery, or gamekit changes.
`ruff check`/`ruff format`/`ty check` clean on touched package files;
`pytest tests/test_runners.py -m "not slow"` 47/47.

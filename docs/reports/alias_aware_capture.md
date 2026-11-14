# Alias-aware CUDA-graph capture — `mlopt_cuda`

Spec: `tts_stt_coverage_map.md` item #10; mechanism proven by
`decode_step_fusion.md` §3 and `sdpa_backend_fix.md` ("Follow-up
assessed").  Implementation: `packages/mlopt-cuda/src/mlopt_cuda/runners.py`.

## What shipped

`CudaGraphRunner(alias_inputs=...)` — an opt-in spec selecting which
positional inputs the generic capture may bind **by reference**
instead of copying into static buffers per replay:

* `None` / `False` (default) — historical copy-everything behaviour,
  byte-for-byte unchanged;
* `True` — every position eligible;
* an iterable of ints — those positions;
* a `(position, tensor) -> bool` callable — per-input decision.

Composes with `multi_device=True`: the lazy per-(device, signature)
capture binds whichever tensors the **first** matching call passes —
for a decode loop that first call already carries the persistent
buffers, so no extra wiring is needed.

```python
runner = CudaGraphRunner(
    multi_device=True,
    alias_inputs=lambda i, _t: i >= 2,   # whisper: 24 past tensors
)
```

Executor modules (`capture_cuda_graph`) are untouched — the flag
reaches only the generic `_GraphReplayModule` path.

## Mechanism

`_GraphReplayModule.capture(module, xs, alias=pred)`:

1. **Warmup on all-clone inputs** (unchanged): the side-stream warmup
   forwards never touch caller tensors, so capture-time side effects
   (in-place writes, RNG) hit clones exactly as before.
2. **Static buffers for copy positions; live tensors for alias
   positions.**  The record call runs `module(*call_ins)` where
   `call_ins[i]` is the static clone (copy pos) or `xs[i]` itself
   (alias pos).  The graph therefore records raw pointers into the
   caller's storage for aliased inputs — replay reads their *current*
   contents with zero copies.
3. **Mutation guard (version counter).**  Around the record call the
   capture snapshots `xs[i]._version` for every aliased position.  A
   bump means the *recorded graph itself* writes that input in place —
   replay would then corrupt the caller's live buffer every call.
   Such positions are demoted to the copy path and the graph
   re-recorded; each pass shrinks the alias set, so it terminates
   after at most `n` re-records.  Verified empirically: `x.add_(w)`
   bumps `_version` during `torch.cuda.graph` capture even though the
   kernel does not execute until replay.
4. `_graph_ins` stays a full-length parallel list: static buffer
   (copy pos) **or the bound caller tensor** (alias pos); `_alias`
   marks which.  `stats["cuda_graph_alias"] = True` records that the
   armed wrapper has ≥1 bound position.

## Replay contract (the paranoid part)

`forward` → `_matches(xs)` requires, per position: tensor with the
capture-time shape/dtype/device **and**, for aliased positions,
`_alias_bound(t, cap)`:

* fast path `t is cap` (the decode loop reuses buffer objects);
* else full binding equality: `layout`, `shape`, `dtype`, `device`,
  `stride`, `storage_offset`, and `untyped_storage().data_ptr()`.

Deliberately **excluded**: `_version` — in-place mutation of the
bound storage *by the caller between calls* is the entire point
(the loop writes new K/V into its buffers each step; the graph must
see them).  Only mutation *recorded into the graph* is guarded,
capture-side.

Any mismatch — fresh tensor, clone, different storage, same-shape
impostor, different stride, different offset — fails `_matches` and
the **whole call** falls back to `inner(*xs)` eager.  There is no
per-position partial replay and no copy-into-captured-storage
(would silently mutate a tensor the caller still owns).  Eager is
correct but uncaptured, so the honest advice is: alias only
positions the caller verifiably reuses (persistent KV-cache-style
buffers); `alias_inputs=True` on fresh-per-call inputs degenerates
to eager-every-call.

Residual risks, documented: a module that writes an input's storage
through an alias that shares no version counter (raw-pointer
tricks) escapes the demotion guard — same caveat class as any
graph capture; callers mutating bound storage from a *different*
CUDA stream than the replay stream race exactly as they would
racing any aliased buffer.

## Measured (whisper-base fp32 decode step, RTX 2050, L=40, ntok=32)

`bench/drivers/probes/whisper_decode_step_profile.py 3 32 40`
(new arms: shipped alias graph step + both shipped graph arms in
the real greedy loop):

| arm | step ms | notes |
|---|---|---|
| delivered `IRModule` | 4.53 / 5.26 | two runs (GPU shared) |
| shipped graph (26 clones+copies) | 5.22 / 5.40 | `copies:` alone = **0.881 / 0.890 ms** |
| **shipped alias graph** | **4.36 / 4.67** | **−0.86 / −0.73 ms vs copy graph** |
| probe `_ZeroCopyGraphCapture` | 4.358 / 4.363 | reference device |
| graph replay alone | 4.361 / 4.368 | floor |

Alias replay vs delivered: `max|diff| = 0.00e+00` (bitwise, both
runs).  Loop level (`run_delivered_loop`, ids vs eager greedy):

* run 1 (quiet GPU): zero-copy 4.40 ms/tok, **shipped-alias 4.60
  ms/tok**, ids-match 32/32;
* run 2 (contended GPU — all arms inflated ~2x): shipped-alias
  9.68, copy-graph 6.54, zero-copy 10.51 ms/tok, ids-match 32/32
  everywhere.  Loop-level deltas are noise-dominated on this shared
  host; the step-level copy elimination (−0.73..−0.86 ms, matching
  the predicted ~0.88 ms of D2D copies for ~36.8 MB of constant
  cross-attn K/V) is the honest number.  The residual ~0.2–0.3 ms
  vs the probe-local device is the `_DeviceGraphModule` per-call
  `storage_fingerprint` param check plus 2 remaining small copies —
  both pre-existing costs of the multi_device path, not alias
  overhead (`_alias_bound` is ~24 pointer compares).

## Tests (`tests/test_runners.py`, all `requires_cuda`)

* `test_normalize_alias_inputs_forms` — spec normalization +
  `TypeError` on non-int positions;
* `test_cuda_graph_alias_binds_live_storage` — in-place mutation of
  the bound buffer between calls is observed by replay (live read,
  not capture-time snapshot);
* `test_cuda_graph_alias_identity_guard_falls_back_eager` — clone →
  eager + correct + bound tensor uncorrupted; `detach()` re-wrap of
  same storage → still replays;
* `test_cuda_graph_alias_rejects_stride_mismatch` — same
  storage/offset/shape, different stride → eager;
* `test_cuda_graph_alias_mutated_input_auto_demoted` —
  `x.add_(w)` module: mutated position demoted to copy path,
  read-only position stays aliased, caller's `x1` uncorrupted;
* `test_cuda_graph_alias_mismatch_whole_call_eager` — one bad
  position sinks the whole call;
* `test_cuda_graph_alias_multi_device_binds_first_call` — lazy
  capture binds first-call tensors; fresh tensor → eager correct;
* `test_cuda_graph_no_alias_unchanged_default` — default path
  byte-identical behaviour, `cuda_graph_alias` stat absent.

`tests/test_public_api.py::test_runner_constructor_shapes` pin
updated for the new keyword-only `alias_inputs=None`;
`docs/public_api.md` runner row updated to match.

## Gates

* `pytest tests/test_runners.py tests/test_public_api.py -q` —
  **286 passed, 1 skipped** (CUDA present; ran under a
  `PYTHONPATH` shim for `mlopt_dtypes` — see issues).
* `ruff check` / `ruff format --check packages/mlopt-cuda` — clean.
* `ty check` — **0 diagnostics in `mlopt_cuda`** (repo-wide ty is
  currently red from the in-flight mlopt-discovery→discoverykit
  split; unrelated).
* Broader `-k "cuda or runner or graph"` slice: `test_runners.py` +
  `test_public_api.py` green; ~170 failures in
  `test_egraph_differential`/`test_handl_egraph` parity tests are
  the other agents' mid-split breakage (deleted
  `packages/mlopt-dtypes`, moved discovery modules), not this
  change — verified unrelated by file ownership.

## Issues / handoff notes

* **Shared-tree turbulence**: the Phase-1 split agent has
  `packages/mlopt-dtypes` deleted in the working tree and
  `mlopt-discovery` mid-move; `.venv` is stale until `uv sync`.
  I ran tests with `PYTHONPATH=/tmp/dtypes_shim` (the HEAD content
  of the deleted `__init__.py`) — no repo files touched for this.
* `transformers` is not in `.venv`; the whisper probes ran with
  `PYTHONPATH=/tmp/xfm_pkgs` (pre-existing scratch install,
  transformers 5.23).
* The coverage-map row (#10) can be flipped to landed; left
  untouched since the map may be curated by another lane.
* Future work, not done: dual-graph (alias + copy variants chosen
  per call by identity) would serve mixed callers without eager
  fallback; per-position recapture on binding change under
  `_DeviceGraphModule` (currently eager-forever per key).

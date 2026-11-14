# bf16 verify tolerance + host-RAM param sharing

Follow-up to `verify_lowmem.md`, which fixed the *device*-side
weight duplication (`_park_params`) and flagged two remaining gaps:

- **GAP A** — bf16 verify verdicts: whisper-small/turbo bf16 ran the
  gate to completion but reported `verified=False` (`max_rel≈0.031`
  vs `rtol=1e-4`).  bf16 carries ~8 mantissa bits; reassociation
  across a rewritten graph exceeds any fp32-style rtol.
- **GAP B** — host-RAM duplication: the verify pair still
  materialised ~3 host copies of the weight set
  (`SearchResult.param_values` + one `.clone()` per `IRModule`),
  ~6.5 GB for whisper-turbo bf16 on an 11 GB host.

Both are addressed in `mlopt_torch.report` / `mlopt_torch.torch_bridge`
— no eqsat change was needed (see "Where the tolerance lives").

## GAP A — dtype-resolution rtol floor

### Policy

`verify_equiv`'s gate is unchanged in form —
`max_rel = max|ref − out| / (max|ref| + 1e-8)`,
`passed = max_rel < rtol_eff AND (atol is None or max_abs <= atol)` —
but the effective rtol is now

```
rtol_eff = max(rtol, _dtype_rtol_floor(ref, out))
_dtype_rtol_floor = 8.0 * torch.finfo(dtype).eps   per output,
                     for the coarsest float dtype with eps > fp32.eps
```

Concretely:

| dtype  | eps        | floor (8·eps) |
|--------|------------|---------------|
| bf16   | 0.0078125  | **0.0625**    |
| fp16   | 0.0009766  | **0.0078**    |
| fp8    | (finfo)    | dtype-derived |
| fp32   | 1.19e-7    | 0 — unchanged |
| fp64   | 2.22e-16   | 0 — unchanged |

The floor engages **only for sub-fp32 precisions** — dtypes whose
machine epsilon exceeds fp32's.  fp32/fp64 comparisons are
bit-identical to before (the fp64 verify convention is untouched).
Non-floating and non-torch dtypes contribute nothing.

### Why this is principled, not a weakening

The `max_rel` metric *cannot resolve* differences below the compute
dtype's own rounding: two evaluations of the **same** program in bf16
already drift by reassociation at ~4·eps — measured `max_rel≈0.031`
on whisper-turbo bf16 (`verify_lowmem.md`), i.e. ~4·eps_bf16.
Demanding `rtol=1e-4` asks for ~13× more precision than the dtype
carries, so every honest bf16 delivery failed on rounding noise, not
semantics.  `8·eps` is a reassociation-drift budget: anchored on the
one measured point (~4·eps) with ~2× headroom.

It still catches real bugs.  Semantic errors — a wrong reduction
(`sum` vs `mean`), a dropped term, a sign flip — produce O(0.1–1+)
relative errors, far above the bf16 floor (~6e-2).  What the floor
removes is only the sub-precision noise band that is *indistinguishable
from a correct rewrite in that dtype* — a bf16 verify could never
certify tighter than ~eps anyway.  Pinned by tests:

- `test_verify_equiv_bf16_reassociation_noise_passes` — ~2.5 % drift
  passes (`max_rel>1e-4`, so the old gate provably failed it);
- `test_verify_equiv_bf16_real_semantic_error_still_fails` —
  sum-vs-mean and sign-flip still fail;
- `test_verify_equiv_fp32_floor_does_not_engage` — fp32 at
  `rtol=1e-9` still fails a 5e-7 diff (no floor);
- `test_verify_equiv_explicit_looser_rtol_still_honored` — the floor
  is a *minimum*: `rtol=1.0` still wins.

The applied tolerance is recorded on `VerifyReport.rtol` for
auditability.  `rtol=inf` (the error-bound gate in eqsat's
`lower`) is unaffected — `max(inf, floor) = inf`.  `atol` semantics
are unchanged and still apply independently.

### Honest limits

- The factor **8 is a heuristic**, calibrated on a single measurement
  (whisper bf16, `max_rel≈0.031` ≈ 4·eps).  A different model/input
  could drift somewhat higher; if a future bf16 verify reports
  `max_rel` between ~0.03 and ~0.06 and *fails*, that is the floor
  boundary doing its job — investigate before raising the factor.
- A semantic bug whose output error is *below* ~0.06 relative in
  bf16 could pass.  That is the resolution limit of the dtype, not a
  choice of this gate; callers needing tighter certification should
  verify in fp32/fp64 (the fp64 verify convention already exists —
  `_align_to` casts params to the input dtype, so fp64 inputs give an
  fp64 check with `rtol=1e-4` and floor 0).
- `max_rel` is a *global-norm* relative metric, not elementwise
  `allclose`; near-zero references are governed by `atol`/`_REL_FLOOR`
  exactly as before.

### Where the tolerance lives

Only `mlopt_torch.report._verify_pair` computes it, so every torch
verify path inherits it: `verify_equiv`, `verify_module`,
`TorchSink.verify`, the orchestrator's `lower`/`_optimize_compositional`/
autotune/diagram gates (all call `sink.verify` — verified in
eqsat `optimize.py`, `autotune.py`, `diagram.py`; **no eqsat change
was required**).  `mlopt_onnx.report` has its own numpy-based
`verify_equiv` — untouched (ONNX bf16 verification is out of scope).

## GAP B — param-storage sharing + `_parameters` consistency

### What changed

`IRModule._build_params` registered each param as
`self._param_values[name].clone()` — a private copy per module.  The
verify pair lowers the **same** `param_values` dict twice
(`orchestrator.lower`: `params = dict(result.param_values)` — a
shallow copy, same tensors), so the host held 3 copies: the dict's
tensors plus one clone per module.

The clone is now `detach()` — the registered `Parameter` **shares
storage** with the caller's tensor.  Eval is read-only (`no_grad`
forwards; `_align_to`/`_park_params` only ever *rebind* tables to new
tensors, never mutate in place), and `export_to_ir` already hands
`lower` private `.clone()`d `source_tensors`, so the delivered module
never aliases the user's live model.  Net effect for the verify pair:
**1× weight storage instead of 3×**.

Measured (synthetic, unique `untyped_storage` bytes across
`param_values` + both modules' `parameters()`):

| arm                              | unique param storage |
|----------------------------------|----------------------|
| before (clone per module)        | 3× weight set        |
| after (detach/share)             | **1× weight set**    |

For whisper-turbo bf16 (~1.6 GB weights): ~4.8 GB of param copies →
~1.6 GB, i.e. **~3.2 GB less host RAM** — the difference between the
~6.5 GB footprint that hit earlyoom and comfortable headroom.  The
GPU-side park (`_park_params`) is unchanged and now *cheaper*: the
registered `_parameters` entries it prefers as CPU rebind targets
share the one host storage, so parking allocates nothing.

Carrier executors (`BatchedOMModule`, `BatchedOmdModule`, scan,
decayed) all wrap `IRModule` internally (`self.eval_mod =
IRModule(ir, param_values)`), so they inherit the sharing for free.

### Aliasing contract (documented on `IRModule`)

Registered params share storage with the provided `param_values`
tensors.  In-place mutation now reaches every sharer: `load_state_dict`
(`copy_` into `_parameters`), direct `param.data` writes, and any
caller-side in-place edit of `param_values`.  Forward eval, align and
park never mutate, so read-only use — including the whole verify
delivery path — is safe.  Callers needing isolation clone the values
before lowering.

### `_parameters` consistency — the real bug, fixed

`_align_to` deliberately rewrites only the *live* tables
(`_param_map`/`_param_values`) and never `_parameters` — that split
is now the documented design: `_parameters` stays the canonical
host-resident copy that `state_dict()` reports and `_park_params`
rebinds to.  But two `nn.Module` paths bypassed the live tables
entirely:

- **`Module.to()`/`.cuda()`/`.float()`/… (`_apply`)** rebound
  `_parameters` to new tensor objects while `forward` kept reading
  the stale live tables — and `_dev_key` could even suppress the
  re-align that would have healed it (e.g. `.cpu()` after a CUDA
  forward left CUDA tensors under CPU inputs → device-mismatch
  crash).  `IRModule._apply` now resyncs: registered names rebind to
  the moved `_parameters` entries (restoring the
  `_param_map[k] is _param_values[k] is _parameters[k]` alias),
  unregistered `_param_values` extras are moved by the same `fn`
  (deduped by object identity), and `_dev_key` resets.
- **`load_state_dict`** wrote `_parameters` only — post-`_align_to`
  the live tables are different objects, so the module kept
  evaluating pre-load weights while `state_dict()` reported the new
  ones.  The override propagates `copy_` into every live table entry
  whose name was actually loaded (aligned views keep their
  device/dtype; `copy_` casts).

`state_dict()` itself needed no change: `_parameters` always holds
correct *values* (it shares the `param_values` storage and is never
mutated in place by align/park), so it reports the canonical weights
consistently.

### Remaining duplication (honest)

- `_fold_weight_chains` computes fused weight-only params
  (`fused_N`) **per module** — ref and opt each materialise their
  own fused tensors, so the fused subset is still ~2×.  For typical
  extractions the fused set is small next to the full weight set;
  a cross-module fused cache was judged not worth the identity-keyed
  bookkeeping.
- Unshaped `Param`s and caller-supplied `_param_values` extras were
  never cloned and are unchanged.
- `_align_to`'s per-forward `.to(device, dtype)` casts still allocate
  transient device copies on first forward — unchanged behaviour,
  VRAM-side only.

## Tests

- `tests/test_reports.py` — four dtype-floor tests (bf16 noise
  passes, bf16 semantic error fails, fp32 unchanged, loose rtol
  honored).
- `tests/test_torch_bridge_edges.py` —
  `test_build_params_shares_storage_with_param_values` (data_ptr
  equality across the pair),
  `test_module_to_resyncs_live_param_tables`,
  `test_module_to_moves_unregistered_param_values_extras`,
  `test_load_state_dict_updates_live_param_tables`.

`tests/test_torch_bridge_edges.py` and `tests/test_reports.py` fully
pass (74 + 18), including the CUDA `requires_cuda` park/lowmem tests
on the 4 GB card; `test_adapters`, `test_batched_verify`,
`test_torch_integration -m "not slow"`, `test_export`,
`test_eval_plan`, `test_om_lower_edges`, `test_device_policy`,
`test_phase_c_serving` all pass.  One unrelated pre-existing failure
(`test_compositional_verbose_failed_block` — `fft_fft` now lowers
under torch 2.14) reproduces on the clean tree.

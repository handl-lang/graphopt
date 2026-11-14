# Whisper bf16 bugfixes — findings

Follow-up to `whisper_experiment.md` findings #2 (bf16 in the law
bank) and #3 (non-float tensor leaves stay on CPU).  Both are fixed;
the whisper-base encoder now runs end-to-end in **bf16** through
`Optimizer.search → lower(verify=True)` (previously died inside
`share_duplicate_param_slices` before saturation).

## BUG 1 — bf16 in the law bank (fixed in eqsat)

`share_duplicate_param_slices` (`mlopt_core/laws/pairing.py`) grouped
head-blocks by `.numpy().tobytes()` — `TypeError: Got unsupported
ScalarType BFloat16` on any bf16 param, killing `opt.search` for
the only dtype whisper-turbo fits in 4 GB.

**Fix.** New shared helper `_raw_bytes(t)` in
`packages/mlopt-core/src/mlopt_core/laws/pairing.py` (~line 53):
detach → cpu → contiguous → `numpy().tobytes()`, and on `TypeError`
widens through the tensor's own `.float()` before retrying.  The
widening is *exact* — every bf16 value is representable in fp32 — so
distinct bit patterns still give distinct signatures and the dedup
stays bitwise-faithful (the convention is deliberately stricter than
`torch.equal`: `-0.0`/`+0.0` and NaN payloads stay distinct).  A
tensor-like object with no `.float()` escape re-raises the original
`TypeError` — honest failure, no silent wrong signature.  Core stays
torch/numpy-free: everything is duck-typed method calls, same as
before.

Call sites unified on the helper:

- `pairing.share_duplicate_param_slices` — the whisper blocker.
- `specials._sig` (`laws/specials.py`) — same raw-bytes convention;
  also fixes `headshare.share_duplicate_attention_heads`, which
  imports `_sig`.
- `mlopt_orchestrator.optimize._dedup_blocks` — the replay-side
  mirror used by `_derive_dedup`; it already imported private helpers
  from `laws.pairing`, so it now shares `_raw_bytes` too.

**Other dtype-fragile spots audited — clean:**

- `share_duplicate_params` groups by `(shape, str(dtype))` then
  `_exact_equal` (`(a == b).all()`) — dtype-agnostic, bf16-safe.
- `laws/factored.py` (low-rank offers) uses pure tensor ops
  (matmul/cumsum/`float(sum)`), no numpy — bf16-safe on CUDA.
- `composer.py`'s `torch.equal` calls — torch.equal supports bf16.
- safetensors export (`mlopt_torch/export.py::_tensor_bytes`) views
  the tensor as `uint8` — dtype-agnostic, and `_TORCH_TO_ST` already
  maps `bfloat16`/`float16`.
- Verify path (`report.verify_module` / `verify_equiv`) computes
  `max|ref−out|` in the tensors' own dtype — works on bf16.  Caveat:
  `rtol=1e-4` is tighter than bf16 epsilon (~7.8e-3), so a rewrite
  that changes bf16 rounding *can* legitimately fail verify — that is
  honest semantics, not a bug; the whisper-base bf16 run verified
  `True` because the delivered term was numerically identical.

## BUG 2 — `_align_to` non-float leaves (fixed in graphopt)

`IRModule._align_to` (`packages/mlopt-torch/src/mlopt_torch/torch_bridge.py`)
moved only `is_floating_point()` baked tensors to the inputs'
device/dtype.  Lifted integer tensors — the positional-embedding
`arange` feeding `nn.Embedding`, token-id buffers — stayed CPU, and
`F.embedding`/`index_select` require index and weight on the same
device (verified: `Expected all tensors to be on the same device`).

**Fix.** `_align_to` now walks the same two tables
(`_param_map`, `_param_values`) but splits by dtype kind:

- floating/complex → unchanged: `.to(device=like.device,
  dtype=like.dtype)` (the delivery-dtype contract is preserved).
- everything else (int/bool) → `.to(device=like.device)` **only** —
  dtype is semantic; casting an index to the float dtype would
  corrupt the gather it feeds.
- Re-wrapped `Parameter`s now keep `requires_grad=t.requires_grad`
  (previously a moved `requires_grad=False` param silently regained
  grad; int params would have raised).

`_mint_const` got the matching one-line-class fix for tensor-valued
`Const`s: non-float tensor consts now move to `like.device` with
dtype preserved (previously returned untouched → same CPU/device
wall).  Scalar `Const` minting is unchanged — 0-dim CPU scalars stay
CPU-safe for weak-scalar broadcast and CUDA-graph capture, which is
the distinction the task called out: table entries are real operands
(must be device-matched), `Const` scalars are minted per-eval (stay
CPU).

`bench/drivers/probes/whisper_probe.py`'s monkeypatch of `_align_to`
is now redundant (it moved *all* tensors device-only, which also
skipped the float dtype-cast — upstream's split is the better
semantics); left in place since the probe is scratch.

## Tests added

- `eqsat/tests/test_value_passes.py` — `_BF16Tensor` fake (numpy
  raises `TypeError`, `.float()` widens): `_raw_bytes` fallback +
  injectivity, re-raise without a `.float()` escape, and
  `share_duplicate_param_slices` end-to-end on the hostile dtype.
- `graphopt/tests/test_substrate_domain_edges.py` — real-torch
  bf16 + fp16 `share_duplicate_param_slices` tests (dedup offer,
  dtype preserved, lowered member evaluates `torch.equal` to W).
- `graphopt/tests/test_torch_bridge_edges.py` —
  `test_align_to_moves_integer_index_leaves` (`requires_cuda`):
  int64 CPU index param + CUDA embedding evaluates correctly, moves
  device-only, keeps `requires_grad=False`; plus `_mint_const`
  tensor-valued const coverage and the CPU-scalar contract.

## Verification

- eqsat: `pytest tests/ -q -m "not slow"` — **1656 passed, 15
  skipped**; `ruff check packages`, `ruff format --check packages`,
  `ty check` all clean.
- graphopt (run with
  `PYTHONPATH=eqsat/packages/mlopt-core/src:eqsat/packages/mlopt-orchestrator/src`
  so the working-tree eqsat fix is live — the pinned git dep is
  still `v0.1.0`): bridge/adapters/export/optimize slice —
  **174 passed, 1 pre-existing failure**
  (`test_compositional_verbose_failed_block` fails identically
  *without* the PYTHONPATH shadow and without these changes —
  orthogonal fft/opaque-leaf drift); lowering/runners slice —
  **121 passed**.  `ruff check`/`format`/`ty` clean on
  `torch_bridge.py`.
- Whisper probe (`bench/drivers/probes/whisper_probe.py opt encoder`,
  fp32, with the eqsat shadow): search 3.5 s, lower+verify
  **verified=True**, delivered 60.9 ms, `cuda_graph=True` — matches
  the pre-fix record (59.8 ms / verified), no regression.
- **bf16 end-to-end**: `whisper-base` loaded `torch_dtype=
  bfloat16` on CUDA → `Optimizer.search(max_iterations=2)` 3.1 s
  (previously `TypeError` inside `share_duplicate_param_slices`),
  `lower(verify=True)` **verified=True** in 0.3 s.  whisper-turbo
  bf16 itself was not re-run (809 M params; the encoder search is
  the same code path that used to crash).

## Notes for the parent agent

- **eqsat changes are uncommitted** in
  `/home/luis/GitHub/handl-lang/eqsat` (pairing.py, specials.py,
  optimize.py, test_value_passes.py).  graphopt pins mlopt-core at
  `tag = "v0.1.0"` — the bf16 fix only reaches graphopt's venv after
  eqsat commits + a pin bump (`[tool.uv.sources]` in graphopt
  `pyproject.toml`).  Until then the new
  `test_substrate_domain_edges` bf16/fp16 tests fail against the
  pinned dep.
- **Pre-existing uncommitted work in the tree** (not mine, left
  alone): `eqsat/README.md`, and the `_EvalPlan.releases`
  liveness/peak-memory work already present in
  `torch_bridge.py` (the whisper-small OOM wall #4 — evidently a
  parallel fix in progress).
- Remaining whisper walls from the experiment doc are untouched:
  decoder-search non-termination, `IRModule` tape retention (the
  `releases` WIP may address it), 4 GB verify duplication, and
  `torch.compile`-parity delivery economics.

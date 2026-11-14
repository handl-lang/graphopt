# T5 walls #1–#3 — fixes (torch_bridge)

Follow-up to `t5_r1.md`.  All three pipeline walls lived in
`packages/mlopt-torch/src/mlopt_torch/torch_bridge.py`; two of them
were *silent wrong answers*, the worst kind.  Wall 4
(`_seq_scan_sites` host-RAM OOM in `mlopt_discovery.play`) is a
different package and is **not** addressed here.

Hardware/software identical to the t5 rounds: RTX 2050 4 GB,
torch 2.14.0+cu130.  Verification is synthetic-module CUDA tests in
`tests/test_torch_bridge_edges.py` (the T5 probe itself was not
re-run — see "Not verified" below).

## Wall 2 — `torch.min(a, b)` / `torch.max(a, b)` minted reduction ops (silent wrong answer)

**Root cause.**  `torch.min(a, b)` (elementwise, two operands)
exports as `OpOverload(op='aten.min', overload='other')` whose
`__name__` is `"min.other"`.  `_aten_name` canonicalised it to `min`
— the IR op whose binding is the *whole-tensor/dim reduction*
(`torch.min(x)`).  The second operand was dropped and the scalar
reduction broadcast: T5's `relative_position_bucket` collapsed to
INT64_MIN and the corrupted index OOB'd the `embedding` gather (on
CUDA: a sticky device-side assert that poisoned the context).

**Fix — two layers, both in `torch_bridge.py`:**

1. **Export rename** (`_ATEN_TO_IR`): `"min.other" → "minimum"`,
   `"max.other" → "maximum"` — the substrate's pointwise binary ops
   (`torch.minimum`/`torch.maximum` bindings already existed).  The
   `default`/`dim` overloads still canonicalise to `min`/`max`.
   `_canon_aten_name` now strips an `aten::`/`aten.` namespace prefix
   and `_aten_name`'s `aten::` string path keeps the overload tail
   (`aten::min.other` → `min.other` → `minimum`) instead of truncating
   at the first dot.
2. **Binding-layer guard** (`_extreme_torch`): the `min`/`max`
   bindings now route a second positional operand to
   `torch.minimum`/`torch.maximum` — a hand-minted `min(x, y)` term
   binds elementwise or raises `ValueError`; it can never again
   silently drop an operand.  Reduction tails (`dim`/`keepdim`)
   remain attrs, unchanged.

**Same-class bug found and fixed during the audit:**
`torch.clamp(a, max=b)` exported `aten.clamp.Tensor(a, None, b)`;
the `None` bound drops out of the flat arg list, so `clamp(a, max=b)`
minted the *identical* term as `clamp(a, min=b)` — another silent
wrong answer (`clamp(a, max=b)` evaluated as `clamp(a, min=b)`,
verified `torch.equal(...) == False` before the fix).  The exporter
now records tensor-bound aten positions in an `argpos` attr
(positions 1=min, 2=max; `self` excluded) and the new `_clamp_torch`
binding maps positional args back through it.  Scalar-bound clamps
(`min`/`max` attrs) are unchanged.  Mixed tensor/scalar bounds are
rejected by torch itself — no mixed case exists.

**Residual known instance (documented, not fixed):**
`F.batch_norm`/`F.instance_norm` have the same middle-optional
pattern — `weight`/`bias` sit before `running_mean`/`running_var`,
and a *bias-without-weight* call would read the bias as the weight.
`nn.BatchNorm`/`nn.InstanceNorm` cannot produce that shape (affine is
both-or-neither), so it needs a hand-written `F.*` call to trigger;
flagged here rather than silently "fixed".

## Wall 1 — int-only feeds skipped `_align_to`

**Root cause.**  `_align_to` picked its device/dtype reference via
`_sibling_like` — the first *floating-point* input tensor.  T5's
encoder feed is all int64 (`input_ids`, `attention_mask`), so
`like is None` → silent no-op → CPU-baked `fused_*` tensors collided
with CUDA params (`Expected all tensors to be on the same device`).

**Fix.**  New `_device_like` picks the first **non-CPU tensor input
of any dtype** (int/bool operands pin the delivery device exactly
like float activations — indices must live on the weight's device);
with no tensor input at all it falls back to the first tensor in the
live param tables so CPU-baked `fused_*` tensors still reach CUDA
params.  The CPU early-return is preserved, so the fp64 verify
contract is untouched.  Non-float table tensors still move
device-only with dtype preserved.

## Wall 3 — bf16 silently upcast to the input's dtype

**Root cause.**  `_align_to` cast every floating param to
`(like.device, like.dtype)` where `like` was the first float *input*.
An fp32 mask against bf16 weights cast the whole weight set to fp32
— a "bf16" delivery at fp32 speed (`verified=False`).

**Fix.**  `_align_to` now returns the **module's** delivery dtype —
`_delivery_dtype`, the majority floating dtype across the registered
params (first-seen wins ties, so a stray fp32 norm weight can't claim
a bf16 module) — and casts weights to *that*, never to an input
dtype.  `forward` casts real-floating **inputs** to the delivery
dtype instead (`_cast_to_delivery`): integer/bool operands keep their
dtype (indices!), non-tensors pass through, and a complex input is
never silently stripped to a real dtype.  The cast only runs on the
non-CPU align path, so CPU/fp64-verify semantics are unchanged; the
`_RTOL_DTYPE_FACTOR` floor in `report.py` (commit `87496ee`) already
covers bf16 verify tolerance — no `report.py` change was needed.

Behaviour note: a CUDA fp64 input against an fp32 module now casts
to fp32 rather than upcasting the weights to fp64 — that is the
prescribed "inputs cast to the module" semantics; the verify pair
still compares within `rtol` (fp32-vs-fp64 drift ≈1e-7 ≪ 1e-4).

## Verification

New tests in `tests/test_torch_bridge_edges.py` (all passing on the
RTX 2050):

- `test_min_max_other_overload_exports_elementwise` — export mints
  `minimum`/`maximum`; lowered module `torch.equal` vs eager.
- `test_min_max_reduction_spellings_still_bind` — `min(x)`,
  `min(x, dim)`, `max(x)`, `max(x, dim, keepdim)` still reduce.
- `test_canon_aten_name_min_max_other` — `min.other`/`max.other` and
  `aten::`/`aten.` spellings canonicalise; `*.default`/`*.dim` stay
  reductions.
- `test_handminted_min_max_two_operands_bind_elementwise` —
  `Op.make("min", x, y)` binds elementwise; 3-operand and
  operand+dim spellings raise `ValueError`.
- `test_align_to_int_only_feed_moves_baked_tensors` (CUDA) — all-int64
  feed aligns CPU int buffer + CUDA weight; output `torch.equal`.
- `test_align_to_keeps_module_dtype_not_input_dtype` (CUDA) — bf16
  module + fp32 mask input stays bf16 end-to-end.
- `test_delivery_dtype_and_cast_helpers` — majority-dtype vote,
  input-dtype fallback, int-only `None`, cast rules (int preserved,
  complex→real refused).

Manual spot checks (not committed): `clamp` min-only / max-only /
both-bounds export→lower round-trips `torch.equal` vs eager; scalar
clamps unchanged.

Test runs:

- `tests/test_torch_bridge_edges.py` — 81 passed.
- Focused batch (`test_eval_plan`, `test_executor_base`,
  `test_memo_safety`, `test_semantic_ops`, `test_attrs`,
  `test_contracts`, `test_om{,d}_lower_edges`, `test_scan_lower_edges`,
  `test_batch_norm_expand`, `test_compositional`,
  `test_factored_params`, `test_headshare`) — 900 passed; **4
  failures in `test_semantic_ops.py` are PRE-EXISTING on HEAD**
  (verified via `git stash`): torch-version drift in export naming /
  unbound-op fixtures, unrelated to this change.
- `test_runners.py`, `test_device_policy.py`,
  `test_phase_c_serving.py`, `test_batched_verify.py`,
  `test_om_mha.py` — 119 passed.
- Smoke trio (`test_law_order`, `test_law_provenance`,
  `test_decode_geom`) — 81 passed.
- `ruff check` + `ruff format --check` + `ty check` on
  `torch_bridge.py` — clean.

## Not verified / caveats

- The T5 probe itself was **not** re-run (model download + hours of
  pipeline); fixes are verified on synthetic modules reproducing each
  wall's exact mechanism.  A `bench/drivers/probes/t5_probe.py opt
  encoder fp32` re-run is the natural confirmation — expected: no
  float-mask workaround needed, `verified=True`, bf16 arm stays bf16.
- `_seq_scan_sites` OOM (wall 4) remains open — different package
  (`mlopt_discovery.play`), out of this session's scope.
- `batch_norm`/`instance_norm` bias-without-weight residual noted
  above.
- Other middle-optional-tensor aten ops may exist beyond the audit
  done here (clamp was the only realistic one found); the `argpos`
  mechanism generalises if another surfaces.

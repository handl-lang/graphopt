# `where`/`argN` eval-binding fix — torch_bridge

Follow-up to `decoder_search_scaling.md` "second blocker": after the
`_shape_of` memo fix, whisper-base decoder `search` completes in ~8 s
but `lower(verify=True)` died with
`TypeError: <lambda>() missing 1 required positional argument: 'y'` —
the extracted term carried `where(cond, zeros, arg2=-3.4028e38)` (the
decoder self-attention mask: `le(arange, arange)` → `where` →
`expand`), the third operand stranded in the positional-attr spelling
while the binding demanded three positionals.

**Status: FIXED, and the whole decoder path now delivers.**
`bench/drivers/probes/decoder_search_fixed.py` on `openai/whisper-base`
fp32 CPU: `search` 7.8 s (`fixed_point`, 616 enodes),
**`lower+verify` 0.2 s `verified=True`**, delivered module runs at
parity (eager 48.4 ms vs delivered 49.4 ms per decoder step).

## Root cause

`export_to_ir` mints a non-node positional arg at an
`ATTR_SCHEMA`-undeclared position as `attrs["arg{i}"]`.  `where` had
no schema entry and was not in `_SCALAR_OPERAND_OPS`, so
`aten.where.ScalarOther(x, y, -inf)` exported as
`where(c, x, arg2=-inf)` — two children + one attr.  The binding
(`lambda c, x, y, *a, **kw: torch.where(c, x, y)`) requires three
positionals → `TypeError`.

## The fix — `packages/mlopt-torch/src/mlopt_torch/torch_bridge.py`

Two-sided, matching the file's existing conventions:

1. **Export side — `_SCALAR_OPERAND_OPS` += `where`,
   `bitwise_and`, `bitwise_or`.**  A scalar in an operand position is
   a real operand (the `masked_fill`/`eq` precedent: "the scalar is a
   Const operand so rewrites can bind and check it").  Export now
   mints `where(c, x, Const(-inf))` — three children, binding
   unchanged.  The 0-dim `Const` mint preserves weak-scalar dtype
   semantics (verified: `int32 & Const(3)` stays `int32`, matching
   `aten.__and__.Scalar`).

2. **Binding side — rebuild `argN`-carried spellings** for terms
   minted before the entry or by hand:
   - `where` → new `_where_torch` helper: `arg1`/`arg2` claim their
     operand slots; remaining children fill unclaimed slots in order.
     Rebuilds all three aten spellings exactly — `where(c, x, arg2=v)`
     (ScalarOther), `where(c, arg1=v, y)` (ScalarSelf), and the
     Const-operand form.  A term with no self/other anywhere raises a
     loud `TypeError` instead of torch's overload-mismatch error.
   - `bitwise_and`/`bitwise_or` → `a[0] if a else attr_of(kw, "arg1")`.

## Sibling audit — same class, fixed where cheap

Method: listed every `_IR_TO_TORCH` binding, cross-checked each op's
aten signature for scalar/positional tails, and **verified against
real `torch.export` output** (not just reading code).  Confirmed by
minting terms through `export_to_ir` and evaluating:

| op | export spelling (before) | failure mode | fix |
|---|---|---|---|
| `where` | `where(c, x, arg2=-inf)` | **TypeError** (the blocker) | `_SCALAR_OPERAND_OPS` + `_where_torch` |
| `bitwise_and`/`bitwise_or` (`x & 3`, `x \| 3`) | `bitwise_and(a, arg1=3)` | **TypeError** missing `y` — same crash class | `_SCALAR_OPERAND_OPS` + `arg1` fallback |
| `nan_to_num(x, 0.0, 1.0, -1.0)` | `nan_to_num(a, arg1=0.0, arg2=1.0, arg3=-1.0)` | **silent wrong values** — binding read only kwarg names, dropped posinf/neginf | `_LOCAL_ATTR_SCHEMA {1:nan,2:posinf,3:neginf}` + `attr_of` reads |
| `F.celu(x, 0.5)` | `celu(a, arg1=0.5)` | **silent wrong** — alpha dropped, evaluated α=1 | `_LOCAL_ATTR_SCHEMA {1:alpha}` + binding |
| `F.embedding(…, padding_idx=2)` | `embedding(w, idx, arg2=2)` | attr dropped (forward-equal — padding_idx is gradient-masking metadata — but unfaithful) | `_LOCAL_ATTR_SCHEMA {2:padding_idx}` + binding replays it |
| `torch.add/sub/rsub(…, alpha=2)` | `add(a, a, alpha=2.0)` (kwargs channel) | **silent wrong** — alpha attr ignored, evaluated α=1 | bindings pass `alpha` — **only when recorded**: aten rejects a float alpha on integral inputs even at the default 1.0 (caught by the decoder run itself: `RuntimeError: For integral input tensors, argument alpha must not be a floating point number`) |
| `F.gelu(x, approximate='tanh')` | `gelu(a, approximate=tanh)` | **silent wrong** — bare `F.gelu` evaluated the exact kernel | binding reads `approximate` (kw-only at the Python surface; no schema needed) |
| `torch.round(x, decimals=2)` | `round(a, decimals=2)` | already worked (kwarg channel) | `arg1` fallback added for hand-minted terms |
| `F.leaky_relu(x, 0.1)` | Const operand already | worked | `arg1` fallback added for hand-minted terms |
| `torch.isclose(…, rtol, atol, equal_nan)` | canonical attrs via substrate `ATTR_SCHEMA` | worked | `argN` fallbacks added for hand-minted terms |
| `torch.nonzero(x, as_tuple=True)` | exports as `aten.nonzero_numpy` + getitems — a **different op** | n/a (pre-existing; `nonzero_numpy` is unbound → opaque) | `as_tuple` attr honoured for hand-minted terms only |

Checked and **already correct** (no change): all `ATTR_SCHEMA`-covered
ops (dims/keepdim/correction/eps/padding tails), `_SCALAR_OPERAND_OPS`
members (`pow`/`masked_fill`/comparisons/`scatter`/`lerp`/…),
`searchsorted`/`one_hot`/`pad`/`roll`/`movedim`/`repeat`/`expand`,
`index`/`index_put`/`scatter` helpers, `_dim_args` consumers,
`sdpa`'s argN→canonical kwarg mapping, norm/pool/fft helpers.
Checked and **correctly untouched**: `torch.atan2`/`logical_and`/
`logical_or`/`logical_xor` — torch itself rejects scalar `other`
(`TypeError: argument 'other' must be Tensor`), so export can never
produce the attr-carried form; `bitwise_xor`/`histc`/`bincount`/
`repeat_interleave`/`unique`/`bucketize`/`quantile`/`grid_sample` —
no `_IR_TO_TORCH` binding (opaque leaves).

## Semantics preserved

- `where(cond, x, -inf)` ≡ `where(cond, x, Const(-inf))`: the Const
  mints a 0-dim tensor at the computation dtype — torch's weak-scalar
  broadcast gives identical values (verified `torch.equal` against
  eager for both ScalarOther and ScalarSelf spellings, fp32 and int
  dtypes).
- `argN`-carried rebuilds pass the raw Python scalar to torch — the
  same `Scalar` overload aten dispatched to.
- `alpha`/`approximate`/`padding_idx`/`nan`-family attrs are passed
  through verbatim; absent attrs keep the previous defaults exactly
  (`alpha`/`approximate`-style kwargs are omitted entirely when not
  recorded where aten is dtype-strict).

## Tests — `tests/test_torch_bridge_edges.py`

New section "argN / scalar-operand attr-carriage" (8 tests):
Const-operand minting for both `where` overloads, `arg1`/`arg2`
hand-minted eval (incl. the loud-failure malformed term), `x & 3`/`x |
3` mint+lower with dtype pin, `arg1` bitwise eval, and a round-trip
sweep over every sibling fix (`nan_to_num`, `celu`, `add`/`sub`/`rsub`
alpha, `gelu` approximate, `round`, `leaky_relu`, `embedding`
padding_idx) asserting `torch.allclose`/`equal` vs the eager module.

## Verification

- `tests/test_torch_bridge_edges.py`: **67 passed** (8 new).
- Targeted slice under `scripts/pytest-capped.sh` (18 files:
  `test_torch_bridge_edges`, `test_eval_plan`, `test_om_lower_edges`,
  `test_omd_lower_edges`, `test_decayed_lower`, `test_scan_lower_edges`,
  `test_model_blocks_lowering`, `test_compositional`,
  `test_batched_verify`, `test_trace_pipeline`, `test_hybrid`,
  `test_meta_edges`, `test_weight_specials`, `test_memo_safety`,
  `test_partial_install`, `test_regime`, `test_regime_edges`,
  `test_calibrate_edges`, `-m "not e2e and not slow"`): **374 passed,
  6 failed — ALL pre-existing/environmental**, verified by re-running
  each on unmodified HEAD:
  - `test_aff_diag::test_batched_scan_module_handles_applyd` (from an
    earlier `-x` run), `test_hybrid` ×3,
    `test_regime::test_end_to_end_and_outputs`,
    `test_regime_edges::test_scan_executor_module_properties` — the
    `BatchedScanModule.is_batched`/scan-plan family; fails identically
    on HEAD (torch-version drift in plan matching, unrelated).
  - `test_calibrate_cuda_real_device` — CUDA OOM; another process
    holds the 4 GB GPU.
- Separately, `test_semantic_ops.py` shows 4 failures
  (`unschemed_*_argN`, `unbound_op*`) — also identical on unmodified
  HEAD (unbound-op→opaque drift under the installed torch), unrelated
  to this change.
- New-code coverage: every added/changed line in `torch_bridge.py` is
  exercised by the new tests (verified via `coverage run` on the
  bridge-edge file — the package's 100% floor is a whole-suite
  property).  The full `-m "not e2e and not slow"` lane was attempted
  under `scripts/pytest-capped.sh` (8 G cap) but the host OOM-killed
  it at ~30% — the box is shared and memory-contended; the targeted
  slice above plus the HEAD A/B checks are the verification of
  record.
- `ruff check`/`ruff format --check` on `packages/mlopt-torch`: clean
  (the two ruff findings in the test file are pre-existing at HEAD;
  `tests/` is outside the lint gate's `packages scripts examples`
  scope).
- `ty check packages/mlopt-torch`: clean.

## Remaining notes for the owner / follow-ups

- `aten.nonzero_numpy` (the `as_tuple=True` spelling) has no binding —
  exports containing it become `opaque` leaves.  Pre-existing gap,
  orthogonal to this fix.
- `embedding`'s remaining tail (`scale_grad_by_freq`, `sparse`,
  `per_sample_weights`, `include_last_offset`) stays `argN` — all
  backward/storage hints with no forward-eval effect; documented in
  the schema comment.
- The decoder extraction's near-zero rewrite yield
  (`decoder_search_scaling.md` caveat 1) is unchanged — this fix is
  delivery plumbing, not coverage.  The `sdpa`-canonicalising lift is
  still the highest-leverage next step for actual decode wins.

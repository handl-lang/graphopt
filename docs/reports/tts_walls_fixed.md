# Kokoro TTS probe — walls C (device kwargs) + decoder-nc (unary mul) fixed

Follow-up to `tts_kokoro_r1.md`.  Both defects lived in
`packages/mlopt-torch/src/mlopt_torch/torch_bridge.py` (the export
boundary + the lowering bindings).  All changes are in that file plus
tests; no probe-local workarounds were needed to verify the fixes.

## Bug 1 — wall C: `device=` kwargs silently dropped at export — FIXED

### Root cause (as reported, confirmed)

Two cooperating defects:

1. The kwargs→attrs loop only kept int/float/bool/list/str/dtype — a
   `torch.device` object was silently dropped, so
   `aten.zeros/rand/…(device=d)` exported with no device information.
2. The creator bindings (`zeros`/`ones`/`rand`/`randn`/`empty`/`full`/
   `new_*`) minted on CPU, and the `to` binding applied dtype only —
   so even a `.to(device)` re-spelling could not carry the device.

A third, related gap found while fixing: **positional** `torch.device`
args (`x.to(y)` exports `aten.to.device(x, device, dtype)` — the
device arrives as a positional object, not a kwarg) were dropped by
the same else-branch, and `_opaque_encode` *refused* device/complex
literals outright — so unbound ops carrying `device=` kwargs
(`randint`, `rand_like`, `tensor`, …) failed the whole export loudly
instead of lifting as opaque leaves.

### Changes (`torch_bridge.py`)

- **kwargs loop** (~line 862): `torch.device` values record as attrs
  under their kwarg name, stringified to the canonical form
  (`"cuda:0"`) — the same JSON-safe pattern `dtype` already uses
  (`str(v).split(".")[-1]`).  `attr_to_data`/`term_to_data` stay
  JSON-safe.
- **arg loop** (~line 741): positional `torch.device` records
  `attrs["device"] = str(arg_node)` — covers `aten.to.device`.
- **arg loop** (~line 715): `complex` joined `(int, float)` in the
  scalar-operand channel — see bug 2 (same edit site).
- **`_creator_device(kw)`** (~line 2954): resolves the `device` attr
  back to `torch.device` (string or object both accepted), `None` =
  torch default.
- **Bindings**: `zeros`, `ones`, `empty`, `randn`, `rand`, `full`,
  `zeros_like`, `ones_like`, `full_like`, `new_zeros`, `new_ones`,
  `new_empty`, `new_full`, `arange`, `linspace` now pass
  `device=_creator_device(kw)`; `to` is `_to_torch` (~line 1024),
  which applies recorded `dtype` AND `device` (either may be absent;
  neither → the same identity cast as before).  `*_like`/`new_*`/`to`
  inherit the reference operand's device through torch itself when
  `device=` isn't explicit — matching torch semantics.
- **`_fill_src`** (~line 378): `zero_`/`fill_`-minted `zeros`/`full`
  now carry `device` from `meta["val"].device` — a CPU fill scattered
  into a CUDA view was the same latent mismatch.
- **`copy` binding** (~line 2600): now `.to(device=t.device,
  dtype=t.dtype)` — `copy_` semantics keep the value on the
  destination's device at the destination's dtype.
- **`_opaque_encode`/`_opaque_torch`** (~lines 1580–1654): the `sig`
  grammar gained two JSON-safe spec kinds — `("dev", "cuda:0")` for
  `torch.device` and `("cx", re, im)` for `complex` literals — so
  unbound ops with `device=`/`complex` args lift as opaque leaves and
  replay verbatim instead of refusing export.

### Honest residual gaps (documented, not fixed)

- `memory_format`/`layout` kwargs (`torch.layout.strided`,
  `contiguous_format`, `channels_last`) are still dropped — they are
  layout/perf hints, not values; `to(memory_format=…)` delivers
  value-correct output.
- `pin_memory` is recorded as an attr (pre-existing) but no binding
  honours it — page-locking is a host-side allocation hint with no
  effect on computed values on the paths we verify.
- Complex-valued *kwargs* (e.g. a hypothetical `fill_value=1j`
  spelling) are still dropped by the kwargs loop — positional complex
  operands (the real Kokoro case) are fixed.  No bound op in the
  current vocabulary takes a complex kwarg.
- Stochastic siblings (`randint` + `randn_like` in one module)
  evaluate in plan-tape order, which can differ from FX order — an
  RNG-*stream* divergence, same class as wall A, not a value defect
  (each op verified bitwise individually).

## Bug 2 — decoder-nc unary `mul` (`TypeError: mul() missing 1 required positional argument: "other"`) — FIXED

### Root cause (isolated, minimal repro found)

`CustomSTFT`/`Generator` contains the polar spelling
`magnitude * torch.exp(phase * 1j)` — `phase * 1j` exports as
`aten.mul.Tensor(sin_out, 1j)` where `1j` is a **Python `complex`
literal positional**.  The arg loop's scalar check was
`isinstance(arg_node, (int, float))` — `complex` matched nothing, fell
to the env-lookup else, `str(1j)` wasn't a node name, and the operand
was **silently dropped** → `Op("mul", sin_term)` with ONE arg.  At
eval `torch.mul(x)` raised the reported TypeError.  Present at
`max_iterations=0` — an export defect, exactly as the report suspected
("scalar-mul spelling losing its operand").

Reproduced on the real graph: plan step 964 was `mul` with a single
arg produced by `sin` (step 963), consumed by `exp` (965) — the
`exp(sin(phase) * 1j)` region.  (The complex `decoder` arm never hit
it because its STFT region is opaque leaves.)

### Fix

`complex` joined the scalar-operand channel (same edit as above):
scalar-operand ops mint `Const(1j)` (kept complex — `float(1j)` would
raise, and the Const must retain the imaginary part);
`_mint_const` already handles it via `torch.tensor(1j)` → complex64
0-dim weak scalar, which promotes `float32 * 1j` to complex64
correctly.  `Const`'s `int | float` annotation is a typing contract,
not enforced — the value passes through a `cast` with a comment.

Residual honesty note: `Const(1j)` serialises as `{"const": 1j}` in
`term_to_data`, which is not JSON-`dumps`-safe — the same exposure
class as any non-JSON attr; the lemma-store/rulecache paths that
consume `term_to_data` are law-spec machinery, not the delivery path.
Flagged rather than silently widened.

### End-to-end verification (the real arm)

```
.venv/bin/python  # /tmp/nc_verify.py — probe _harvest("decoder-nc")
search 4.1s (max_iterations=0)
lower 0.02s
delivered out (1, 1, 76200) torch.float32 device cuda:0
maxdiff vs seeded eager: 0.0   # BITWISE
```

Before the fix this died at first eval with the reported TypeError.

## Tests

Added to `tests/test_torch_bridge_edges.py`:

- `test_export_complex_scalar_operand_is_a_const` — `exp(sin(x)*1j)`
  exports a 2-arg `mul` with a complex `Const`, round-trips
  `term_to_data`, lowers, and matches eager bitwise (complex128).
- `test_export_device_kwarg_recorded_cpu` — `zeros/rand(device=…)`,
  `zeros_like`, `new_zeros`, `to(dtype)` record `device='cpu'` attrs;
  CPU delivery is seed-aligned bitwise.
- `test_to_device_cast_positional_and_kwarg` — `x.to(y)` (positional
  `to.device`) and `x.to(device=…, dtype=…)` record both attrs; the
  binding applies them.
- `test_device_kwargs_mint_on_input_device` (`requires_cuda`) — the
  wall-C repro: a CUDA module with `torch.zeros/rand(device=x.device)`,
  `zeros_like`, `new_zeros`, `to(dtype)` exports `device='cuda:*'`
  attrs, lowers, runs on GPU, and matches eager.

Updated `tests/test_semantic_ops.py::test_export_arange_stays_int64`
to scope its all-ints assertion to the scalar bounds (the `device`
attr is legitimately a string now).

## Verification run

- `pytest tests/test_torch_bridge_edges.py tests/test_export.py
  tests/test_contracts.py tests/test_attrs.py
  tests/test_intake_defects.py -q` → **505 passed, 102 skipped**.
- `pytest tests/test_semantic_ops.py tests/test_torch_integration.py
  tests/test_trace.py tests/test_model_blocks_lowering.py
  tests/test_omd_lower.py -q -m "not slow"` → 557 passed; **4
  pre-existing failures** in `test_semantic_ops.py`
  (`unschemed_*_argN`, `unbound_op_is_a_boundary`,
  `compositional_falls_back`) — confirmed failing on unmodified HEAD
  (torch 2.14 drift, unrelated).
- `ruff check` / `ruff format --check` / `ty check` on
  `torch_bridge.py` → clean.  (The test file has 2 pre-existing ruff
  findings at lines 966/981, untouched; `tests/` is outside the
  shipped-source lint gate.)
- Real-model check: `decoder-nc` delivers bitwise at
  `max_iterations=0` (above).  The probe's `_patch_device_creators`
  workaround is now unnecessary but harmless — it respells to
  `*_like`, which still works.

## PackedSequence export — feasibility assessment (not implemented)

The wall is **upstream of the lifter**: `torch.export` itself dies
inside `pack_padded_sequence`'s fake-tensor propagation
(`ValueError: batch_sizes should always be on CPU`) before any FX
graph exists — so no `torch_bridge` change can see the region.
`aten.lstm.data` staying unbound is correct; the blocker is the
`PackedSequence` pytree, which cannot be an IR operand (operands are
tensor terms).

Bounded options, ranked:

1. **Opaque-region custom op (recommended, ~½–1 day).**  Wrap each
   `pack → lstm → pad_packed_sequence` triple in a
   `torch.library.custom_op` (dense tensors in, dense tensors out;
   the PackedSequence never crosses the IR boundary).  torch.export
   lifts registered custom ops atomically and our `_opaque_qual`
   already resolves `ns::op` qualnames through `torch.ops` — the
   region replays verbatim, laws simply can't reach inside (same
   contract as `stft`/`istft` today).  Kokoro has exactly three such
   sites (`modules.py:60-63`, `:113-116`, `:164-168`), all
   `batch_first`, all discarding the hidden state — a clean fit.
   Needs a `register_fake` meta for export + the wrapper applied
   model-side (a `torch_bridge` helper could vend the registration;
   it cannot retrofit arbitrary modules automatically).
2. **Dense-LSTM respelling — NOT honest in general.**  Replacing
   pack/pad with a dense `lstm.input` call is exact only when every
   sequence length equals the padded length (the probe's batch-1
   fixed-shape case).  For true variable lengths the *backward* pass
   of the bidirectional LSTM reads the padded tail and corrupts
   valid positions — `pad_packed_sequence` exists precisely to
   prevent that.  A length-gathered respelling is possible but is a
   model-level transform with real correctness risk, not a binding.
3. **Status quo.**  Text-side modules stay unliftable end-to-end;
   post-alignment heads (the arms that already deliver) are
   unaffected.

Effort estimate: option 1 is small and bounded; option 2 is medium
with a correctness caveat; the wall itself is structural
(PackedSequence pytree vs tensor-term IR), not a missing binding.

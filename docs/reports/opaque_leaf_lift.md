# Opaque-leaf lifting — total lift over the torch op surface

**Status:** implemented (adapter-side; `mlopt_core` untouched).
**Scope:** `packages/mlopt-torch` (`torch_bridge.py`), tests in
`tests/test_torch_bridge_edges.py`, `tests/test_torch_integration.py`,
`tests/test_e2e_pipeline.py`.

## Goal

Make model lifting *total*: any `torch.nn.Module` whose
`torch.export` graph is a DAG of `call_function` nodes lifts to IR,
even when individual ops sit outside the 31+-op lowering vocabulary.
Unknown ops become **opaque leaves** — atomic IR terms that carry the
torch op identity plus a passthrough denotation (semantics = the torch
op itself, verified by calling it).  Laws and collectors cannot
rewrite inside them; the model still optimizes *around* them and
delivery still works.

## The path (as traced)

- `TorchSource.to_ir` (`mlopt_torch/adapters.py`) delegates to
  `export_to_ir` (`mlopt_torch/torch_bridge.py`).
- `export_to_ir` runs `torch.export.export(model, args)` and walks the
  exported FX `call_function` graph.  Each node's target is canonicalised
  by `_aten_name`, mapped through `_ATEN_TO_IR`, and minted as an `Op`
  term (`Var`/`Param`/`Const`/`Op`/`IR` from `mlopt_core.ir`).
- Lowering is `ir_to_torch_module` → `IRModule`, whose eval runs an
  `_EvalPlan` slot tape (or delegates to `eval_term`); op dispatch goes
  through a bindings table (`_IR_TO_TORCH` ambient dict, or an explicit
  `mlopt_core.ops.OpTable`).
- Optimization is `mlopt_core.egraph.EGraph` + `Optimizer`; the torch
  backend's `TorchSink.supported_ops` / `backend_cost` decide which op
  names extraction may keep (unknown names are `+inf` cost *unless* the
  name is in the supported set).

## Where unsupported ops used to die

1. **Lift:** an unbound op still minted an `Op` (with its raw aten
   name), so `export_to_ir` itself succeeded.
2. **Delivery:** `IRModule._eval` / `eval_term` raised
   `ValueError("No torch binding for op ...")` — and, worse,
   `backend_cost` priced any op absent from `supported_ops` at `+inf`,
   so extraction could never select a term containing one.  The
   composed/fallback pipeline therefore kept the original block.
3. **Argument encoding:** `_opaque_encode` refused FX-node arguments
   that had no IR term (e.g. produced by `call_module` /
   `call_method` / unresolved outputs) — correctly, since dropping an
   operand would silently change semantics.

`mlopt_core` (pinned git dep, `packages/mlopt-core` subdir of
`handl-lang/eqsat`) was inspected for an existing extern/opaque
mechanism: **none exists** — `Op` is the only generic constructor, the
substrate has no passthrough/custom-op term, and the package must not
be edited.  The mechanism is therefore implemented entirely on the
adapter side, which is also the right layer: the passthrough denotation
*is* a torch callable, and only the torch adapter can resolve it.

## Design: the `opaque` term

An unbound `call_function` node now mints (`_opaque_term`,
`torch_bridge.py` ~line 1343):

```python
Op.make(
    "opaque",
    *operands,               # tensor-producing FX args, encounter order
    aten=<qualname>,         # resolved+verified torch identity
    sig=(arg_specs, kw_specs),  # JSON-safe replay spec
)
```

- **Identity** — `aten` is a qualname string resolved *at export* by
  `_opaque_qual` (candidate chain, first resolvable wins):
  `OpOverload.name()` → `"aten::upsample_bicubic2d.vec"` /
  `"myns::scale_shift"`; `::`-qualified `str(target)`; a dotted `str`
  re-read as a `torch.ops` path; `__module__:__qualname__`; bare dotted
  `torch.*` attribute path.  Resolution is *verified now* — a target
  that resolves to nothing refuses the lift loudly rather than minting
  an undeliverable term.
- **Operands** — every FX-node argument (in encounter order, shared
  across args and kwargs) becomes a positional operand term.  Scalars,
  `None`, `bool/int/float/str`, `torch.dtype`, and nested
  `list`/`tuple`/`mapping` literals ride inside `sig` as spec tuples
  (`("t", i)` operand slot, `("l", v)` literal, `("dt", name)`,
  `("lst"|"tpl"|"map", ...)`).  Anything that cannot cross the IR
  boundary — raw tensors, `SymInt`s, devices, memory formats, or an FX
  node with no IR term — refuses the lift with a `ValueError`.  `sig`
  is JSON-safe: `term_to_data`/`term_from_data` round-trips an opaque
  term losslessly (tested).
- **Denotation** — `"opaque"` is registered in
  `_CORE_TORCH_BINDINGS` → `_opaque_torch`, which rebuilds the arg
  list from `sig` and calls `_resolve_opaque_callable(aten)(*args,
  **kwargs)`.  `ns::op[.overload]` walks `torch.ops` — this covers
  **both** stock aten ops and `torch.library` custom ops (they live in
  the same `torch.ops` registry); `mod:qualname` imports a module and
  walks attributes; a bare dotted path walks `torch` attributes.
  Semantics *is* the torch op itself — there is no re-implementation
  to drift.
- **Atomicity** — laws, collectors, e-matching and shape rules see only
  the generator name `"opaque"` plus its attrs.  No shipped law matches
  `"opaque"` (it is not in any rule's LHS vocabulary), so nothing
  rewrites inside the leaf; `mlopt_core.typing._shape_of` already passes
  an unknown op's first operand shape through conservatively, and the
  cost model prices it like any unknown op.
- **Supported/bound** — because `"opaque"` sits in
  `_CORE_TORCH_BINDINGS`, it is inside `OpTable.core()`/`OpTable.full()`
  and therefore inside every `supported_ops` set: extraction never
  `+inf`s a term containing an opaque leaf, and both the `_EvalPlan`
  tape and `eval_term` resolve the binding at eval.

Net effect: lifting is total over the `call_function` surface —
`export_to_ir` either produces a complete IR (every node either a known
op or an `opaque` leaf) or refuses loudly for a reason that is *not*
"op outside the vocabulary".

## What still refuses (honest limits)

- **Data-dependent control flow.**  `torch.export` itself raises
  `GuardOnDataDependentSymNode` before the lifter sees a graph — there
  is no DAG to lift.  The compositional pipeline keeps its original-
  block fallback for this case (see
  `test_compositional_fallback_keeps_original_block`, which now uses a
  data-dependent `if` rather than an unbound op).
- **Unencodable op arguments.**  `SymInt`s, raw tensors, devices,
  memory formats, `call_module`/`call_method`-produced values, and
  unresolved outputs cannot cross the IR boundary — `_opaque_encode`
  raises `ValueError` rather than silently dropping an operand.
- **Unresolvable targets.**  A `call_function` target that yields no
  working qualname refuses at export (`_opaque_qual` collects every
  candidate's error into the message).
- **View/mutation hazards.**  The functionalised `copy_`/`fill_`/
  `zero_`/`masked_fill_` family is threaded through `env` as before;
  other mutation patterns and view-aliasing that violate the DAG
  assumption remain out of scope — `torch.export`'s own functionalisation
  is the first line of defence, and anything it cannot functionalise
  never reaches the lifter.
- **Portability of the denotation.**  The `aten` qualname is resolved
  against *this* process's `torch.ops`/module namespace at eval time.
  The IR term serialises losslessly, but delivering it elsewhere
  requires the same op to be resolvable there (a `torch.library` op
  must be re-registered in the consumer process).  This matches the
  eager-passthrough contract: the leaf executes the real kernel, not a
  portable artefact.

## Test evidence

New coverage in `tests/test_torch_bridge_edges.py::TestOpaqueLeaves`:

- `test_unbound_aten_op_lifts_as_opaque` — bicubic
  `upsample_bicubic2d.vec` (no core binding) lifts as one `opaque`
  term with the right `aten`/`sig`, lowers, and matches eager.
- `test_custom_library_op_lifts_as_opaque` — a `torch.library.custom_op`
  (`opaque_test::scale_shift`) exports as `ns::op`, lifts, replays
  verbatim, and composes with covered ops (`softmax`).
- `test_opaque_term_json_roundtrip` — `term_to_data`/`term_from_data`
  round-trip.
- `test_opaque_is_a_supported_op` — `"opaque" ∈ OpTable.core()/full()`.
- `test_opaque_binding_replays_a_hand_minted_term` — table-driven eval
  of a hand-built `opaque` term.
- `test_opaque_qual_*`, `test_resolve_opaque_callable_*`,
  `test_opaque_encode_*`, `test_opaque_torch_*` — candidate ordering,
  error collection, all three resolution schemes, every `sig` spec
  kind, and the loud-failure paths (keeps the 100% branch floor).

New e2e in `tests/test_torch_integration.py`:
`test_opaque_ops_optimize_around_and_verify` — conv2d → bicubic
upsample (opaque) → custom `wiggle` op (opaque) → matmul → softmax →
sum; `Optimizer` runs, the delivered module verifies against eager
(`diff < 1e-4`).

`tests/test_e2e_pipeline.py::test_compositional_fallback_keeps_original_block`
was updated: its old fixture (an op with no binding) now *succeeds* via
opaque leaves, so the fallback case moved to genuinely unliftable
data-dependent control flow.

Commands run (memory-capped via `scripts/pytest-capped.sh`):

- `pytest tests/test_torch_bridge_edges.py -k "Opaque or opaque"` —
  **13 passed**.
- `pytest tests/test_torch_bridge_edges.py tests/test_e2e_pipeline.py
  -m "not slow"` — **58 passed**.
- `pytest tests/test_torch_integration.py -k "opaque or fallback"` —
  **16 passed** (with the bridge file).
- Broader regression slice (optimize/compositional/delivery/adapters/
  contracts/model-blocks files, `-m "not slow"`) — **430 passed,
  102 skipped** in ~7.4 min.
- `coverage run` over the two boundary test files: **zero missed
  lines/branches inside the new opaque code** (remaining misses are
  pre-existing regions covered by other suite files).
- `.venv/bin/ruff check` + `ruff format --check` on
  `packages/mlopt-torch/.../torch_bridge.py` — clean (test files carry
  pre-existing lint/format drift outside the gate's `packages/` scope;
  new test code matches the surrounding style).
- `.venv/bin/ty check` — clean (0 errors).

## Substrate note (for `handl-lang/eqsat`, not done here)

If a first-class extern/custom-op term is ever wanted in
`mlopt_core` (e.g. for non-torch adapters or certified replay), the
adapter-side encoding here is the reference: a generator name outside
the law vocabulary, JSON-safe attrs carrying (a) a resolvable op
identity and (b) a positional operand/literal spec, plus a per-backend
binding that replays the native callable.  No core change was required
for total torch lifting.

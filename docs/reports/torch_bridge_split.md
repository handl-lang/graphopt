# torch_bridge.py split — implementation report

Arch-audit item (`docs/reports/arch_audit_graphopt_packages.md` §2/§6.5):
the ~4.5k-line `kit_torch.torch_bridge` god-module is now three
cohesive modules plus a thin re-export shim.  **No commit** — working
tree only.

## Modules

| Module | Contents | Lines |
|---|---|---|
| `torch_export.py` | FX lift: `_strip_backend_suffix`/`_canon_aten_name`/`_aten_name`, `_fill_src`/`_view_chain`/`_scatter_view`/`_read_view`/`_handle_copy_`, `export_to_ir`, `_graph_root`/`_graph_outputs`/`_resolve_attr`, `_FFT_SHAPE_OPS`, the opaque-leaf mint cluster (`_is_dotted_name`, `_resolve_opaque_callable`, `_opaque_qual`, `_opaque_encode`, `_opaque_term`) | ~1,049 |
| `torch_bindings.py` | Override literals (`_ATEN_TO_IR_OVERRIDES`, `_IR_TO_TORCH_EXTRA`, `_LOCAL_ATTR_SCHEMA_OVERRIDES`, `_SCALAR_OPERAND_OPS_OVERRIDES`), all `_xxx_torch` binding impls, `_fft_kw`, `_opaque_torch`/`_extreme_torch`/`_clamp_torch`, sdpa backend policy, `_CORE_TORCH_BINDINGS` (~265 entries), shape rules + `register_shape_rule`, `_AmbientTorchBindings`/`_IR_TO_TORCH` + `register_core_bindings`/`register_ambient_bindings`, the `aten_classify` merge (`_ATEN_TO_IR`/`_SCALAR_OPERAND_OPS`/`_LOCAL_ATTR_SCHEMA`/`_BOUND_OPS`), carrier helpers (`_om_elem`/`_om_compose`/`_split_sizes`/`_creator_*`/`_dim_args`) | ~2,307 |
| `torch_eval.py` | `_MISS`/`_UNPLANNED`, `_randn_param`, `_EvalPlan`, `_const_like`/`_mint_const`/`_sibling_like`/`_device_like`/`_COMPLEX_DTYPES`/`_cast_to_delivery`, `eval_term`, `IRModule`, `ir_to_torch_module` | ~1,244 |
| `torch_bridge.py` | Shim: docstring, third-party re-exports, explicit re-export of every split name, `__all__` (121 names), `__getattr__` fallback over the three submodule namespaces | ~316 |

Before: `torch_bridge.py` = **4,498** lines.  After: 4,498 → 316 +
1,049 + 2,307 + 1,244 = 4,916 (headers/imports/`__all__` overhead).

Dependency direction is acyclic: `torch_export → torch_bindings`;
`torch_eval` imports neither; `torch_bridge` imports all three.

## Import compatibility

- Every one of the **117** module-level names the old file bound
  (verified by AST diff) resolves on `kit_torch.torch_bridge` —
  explicit re-exports plus a `__getattr__` fallback that scans the
  three submodule `__dict__`s (covers third-party names like `attr_of`,
  `torch`, `FOLD_OPS` that callers also import from the bridge, e.g.
  `mlopt_discovery.genkernel`).
- `_IR_TO_TORCH`, `_CORE_TORCH_BINDINGS`, `_ATEN_TO_IR`, etc. are the
  **same objects** — `OpTable.full()` still mutates the one ambient
  dict; `register_core_bindings`/`register_ambient_bindings` still run
  at `torch_bindings` import (pulled in by the shim).
- Function sources are byte-identical to HEAD except two intended
  one-line edits (below) — verified by `ast.get_source_segment`
  comparison of all 89 moved defs/classes.

## Monkeypatch seam (the hard part)

Two patch sites exist in the suite; both keep working:

1. `tests/test_om_stream.py` —
   `monkeypatch.setattr(torch_bridge, "_om_elem", spy)` must reach the
   leaf evaluator.  `_om_elem`/`_om_compose` canonically live in
   `torch_bindings`, but the `_CORE_TORCH_BINDINGS` entries now call
   `_om_elem_shim`/`_om_compose_shim`, which resolve
   `torch_bridge._om_elem`/`._om_compose` **at call time** — a patch on
   the shim is seen; unpatched it is the same function object.
2. `tests/test_torch_bridge_edges.py` —
   `monkeypatch.setattr("kit_torch.torch_bridge._resolve_opaque_callable", …)`
   then calls `_opaque_torch`.  `_opaque_torch` (bindings) and
   `_opaque_qual` (export) both call `_resolve_opaque(qual)`, which
   late-binds `torch_bridge._resolve_opaque_callable` — patches reach
   both the replay path and the export-time qual probe.

Verified live: patched `torch_bridge._om_elem` fires through
`_IR_TO_TORCH["om_elem"]`; patched `_resolve_opaque_callable` fires
through both `_opaque_torch` and `_opaque_qual`.

## `_om_elem`/`_om_compose` → carriers?  NO (deliberate)

The audit suggested `mlopt_carriers.om`.  Rejected: `mlopt-carriers`
already imports `IRModule`/`_om_elem`/`_om_compose` **from**
`kit_torch.torch_bridge` — moving them to carriers would create a
package-level import cycle (torch → carriers → torch).  They now live
in `torch_bindings` beside the other carrier-binding helpers; the shim
keeps them patchable as above.

## Other edits inside moved code (complete list)

- `_opaque_torch`: `_resolve_opaque_callable(qual)` →
  `_resolve_opaque(qual)` (shim dispatch).
- `_opaque_qual`: same substitution (export side).
- `_CORE_TORCH_BINDINGS`: `om_elem`/`om_compose` lambdas →
  `_om_elem_shim`/`_om_compose_shim` (+ explanatory comment).
- New dispatcher block in `torch_bindings` (`_resolve_opaque`,
  `_om_elem_shim`, `_om_compose_shim`) documenting the seam.

## `scripts/complexity_baseline.json`

The radon ratchet keys on `path::qualname`, so the split orphaned all
63 `torch_bridge.py::*` baseline entries.  Migrated surgically: every
baseline function in the four `torch_*.py` files was re-scanned and
re-keyed to its new module path (values = measured complexity — the
baseline was already drifted at HEAD, e.g. `export_to_ir` pinned 62,
measures 78; the split records reality rather than the stale pin).
No non-torch entries touched.  Remaining ratchet noise is all
pre-existing: `build/lib/` stale artifacts, `backend.py`/`export.py`/
`meter.py`/`report.py` rises, and dirty `mlopt-discovery`
`workload_gen.py` — none mine.

## Gates

- `ruff check` + `ruff format --check` on the four files: clean.
- `ty check`: 0 errors.
- `lint-imports`: 1 contract kept.
- `vulture`, `bandit`, `semgrep` on the new files: clean.
- `pytest tests/ -q -k "torch_bridge or semantic or intake or sdpa or
  torch_port"`: **662 passed, 11 failed — byte-identical failure set
  with the pre-split file** (verified by swapping HEAD's
  `torch_bridge.py` back in: same 11 — `test_discovery_intake` ×6,
  `test_semantic_ops` ×4, `test_synthesis_seeds` ×1, all from the
  dirty discovery/carriers tree).
- Extra sweep (16 bridge-heavy files: om_*, hybrid, scan_fused, trace,
  attrs, ports, layout, decode_geom, memo_safety…): 348 passed, 5
  failed — again identical with the original file (`test_hybrid` ×3
  `synop2_*` binding misses, `test_scan_fused` ×2 — dirty
  carriers/discovery, pre-existing).
- `test_om_stream.py::test_streaming_block_working_set_is_bounded`
  (the `_om_elem` monkeypatch test): passes.

## Notes for the parent agent

- `packages/kit-torch/build/lib/kit_torch/torch_bridge.py` is a
  **stale build artifact** (old monolith) still scanned by
  `radon_ratchet.py` (`TARGETS=["packages"]` rglobs `build/`).  It
  predates this change and fails the ratchet independent of the split;
  a `uv build`/`pip install -e` refresh or deleting `build/` silences
  it.  Left untouched.
- `pyproject.toml` and the `mlopt-discovery`/`mlopt-carriers` dirty
  files belong to other agents — untouched.
- `torch_bridge.py`'s old module docstring ("Fixed torch_bridge.py…")
  was stale per audit D7; the shim docstring now documents the real
  layout.

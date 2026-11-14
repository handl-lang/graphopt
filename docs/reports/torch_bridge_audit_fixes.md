# torch_bridge audit fixes — B1–B7

Fixes for the silent-wrong findings in
`docs/reports/arch_audit_graphopt_packages.md` §1, all in
`packages/mlopt-torch/src/mlopt_torch/torch_bridge.py`.  Regression
tests: `tests/test_torch_bridge_audit_fixes.py` (22 tests);
pre-existing decline-pinning tests updated in
`tests/test_torch_bridge_edges.py`, `tests/test_intake_defects.py`
and `tests/test_semantic_ops.py::test_handle_copy_early_returns`.

## Reachability findings (torch 2.14.0+cu130, CPU)

Probed `torch.export.export` directly before fixing:

- **`node.kwargs` are almost always normalised to positional `args`.**
  `F.pad` pad list, `F.conv2d`/`conv_transpose2d` stride+padding,
  `F.interpolate` size, sdpa `attn_mask`/`is_causal`, `F.grid_sample`
  `align_corners`, `F.embedding` `padding_idx` all arrive positionally.
  The kwargs that DO survive are factory/dispatch literals:
  `device`, `pin_memory`, `dtype`, `layout` (`torch.strided`, seen on
  `one_hot`), `memory_format` (`torch.channels_last`, seen on
  `contiguous(memory_format=)`), and bool flags (`upper` on
  `solve_triangular`).  **No probed op emitted an FX-node or tuple
  kwarg** — the B1 latent path is real but unreached by stock aten on
  this version; it is now handled AND loudly refuses what it cannot
  encode (tests drive it through a stubbed `torch.export.export`,
  since real export never produces the shapes).
- **`node.op` kinds:** every probed model (submodules, ModuleList,
  method-call syntax, dict/list locals, tuple returns, non-Module
  helper objects, `.item()`, `reshape`) exported only
  `placeholder`/`call_function`/`output` — torch.export flattens
  `call_module`/`call_method` away.  The `else: raise` is therefore
  a guard, not a behaviour change, on common models.
- **B4 was fully reachable:** `nn.BatchNorm2d(track_running_stats=
  False)` in eval exports
  `aten.batch_norm(x, w, b, None, None, True, 0.1, 1e-5, False)` —
  the Nones drop and `(w, b)` was read as `(rm, rv)`.

## Per-bug status

### B1 — kwargs drop — FIXED (latent path) + loud refusal for the unencodable rest

`export_to_ir`'s `node.kwargs` loop (now ~870-952) rewritten:

- FX-node kwarg values (`getattr(v, "name")` is a str) resolve via
  `env[name]` and append to `args` in order — mirroring
  `_opaque_encode`.  Unminted producer → `ValueError` (the
  `masked_fill_` contract).
- `list`/`tuple` values → `attrs[k] = tuple(v)` for ANY key (the old
  `k == "dim"` special case is subsumed — B7).  A container carrying
  FX nodes → `ValueError` (tensor operands cannot ride the attr
  channel).
- `v is None` → explicit skip (absent-operand spelling; recording it
  would poison e-matching's exact attr-set comparison).
- `torch.layout` / `torch.memory_format` → short-name string attr
  (`"strided"`, `"channels_last"`) — real exports DO emit these
  (`one_hot`, `contiguous(memory_format=)`); previously silently
  dropped.
- Everything else unencodable → `ValueError` naming the kwarg, value
  and op (previously silently dropped).

Evidence: `test_kwarg_fx_node_operand_is_not_dropped` (sdpa with
`attn_mask` as a kwarg lowers and matches `F.sdpa` exactly),
`test_tuple_kwarg_lands_as_attr` (`pad=(1,1,1,1)` + `value=None` +
`layout`), `test_str_and_memory_format_kwargs_recorded`,
`test_kwarg_fx_node_unminted_raises`,
`test_kwarg_container_with_fx_nodes_raises`,
`test_unencodable_kwarg_raises`.

### B2 — unhandled `node.op` — FIXED (loud refusal)

`elif node.op == "output": continue` (resolved post-loop by
`_graph_root`) + `else: raise ValueError` naming the kind/node/
target.  Verified `call_module`/`call_method` do not appear on
common torch 2.14 exports (flattened to `call_function`); if a future
op or exotic module produces one, the export now refuses instead of
silently dropping the value and every downstream operand.
Evidence: `test_unhandled_node_op_raises`,
`test_output_node_kind_is_allowed`.

### B3 — `_handle_copy_` silent early-returns — FIXED (loud rejection)

All five silent returns converted to `ValueError` with node context
(`_aten_name(node.target)`, `node.name` via `getattr` for hand-built
nodes): empty `node.args`, unminted dst, unminted copy_ source,
unminted view-chain root, unminted intermediate view base.  Matches
`masked_fill_`'s deliberate `KeyError` contract — a write that cannot
be threaded is an export rejection, not a no-op.
Evidence: 5 new `_handle_copy_` raise tests in the audit file; the
two pre-existing decline-pinning tests updated
(`test_handle_copy_rejects_on_an_unminted_intermediate_view` in
`test_torch_bridge_edges.py`,
`test_copy_family_{slice_write_unminted_base,unminted_src}_rejects`
in `test_intake_defects.py`).

### B4 — `_batch_norm_torch` operand misread — FIXED (flag + guard)

Two-sided fix because len-2 is genuinely ambiguous (`(w, b)` vs
`(rm, rv)` — no eval-side arity check can decide):

- **Export** (`export_to_ir`, ~966-981): for `ir_op == "batch_norm"`,
  when aten slots 3,4 (`running_mean`/`running_var`) are not both
  minted FX nodes, record `attrs["running_stats"] = False`.  Only
  the absent case is tagged — stats-present terms are byte-identical
  to before (no new attr, so e-matching's exact attr-set comparison
  is untouched for them).
- **Binding** (`_batch_norm_torch`, ~1272+): `running_stats=False`
  → all operands are the affine pair (`w = a[0]`, `b = a[1]`),
  `rm = rv = None`; `len(a) > 2` → `TypeError` (the stats slots
  cannot reappear).  Untagged (hand-minted) terms keep the legacy
  last-two heuristic — documented in the docstring.

Verified: `nn.BatchNorm2d(track_running_stats=False)` eval export →
term carries `running_stats=False` with exactly `(x, w, b)` operands
→ lowered module matches the original (previously computed with the
weight matrix as `running_mean`).  `affine=False` +
`track_running_stats=False` (zero post-x operands — pre-fix
`IndexError`) also round-trips.  `track_running_stats=True`
unchanged: no attr, `(x, w, b, rm, rv)`, round-trips.
Evidence: `test_batch_norm_no_track_export_roundtrip`,
`test_batch_norm_no_track_affine_false`,
`test_batch_norm_track_still_works`,
`test_batch_norm_partial_stats_slot_still_tags_absent`,
`test_batch_norm_truncated_args_tag_absent` (the two hand-built
graphs cover the compound condition's short-circuit arcs the real
export can't produce),
`test_batch_norm_binding_running_stats_false_surplus`.

### B5 — `_where_torch` surplus children — FIXED

After the two operand slots fill, a leftover child raises
`TypeError("where takes exactly two value operands … surplus
children")` — previously silently ignored (computed `where/2`).
Evidence: `test_where_torch_surplus_children_raise`,
`test_where_torch_two_children_still_work`.

### B6 — `_dim_args` 0-dim unwrap — FIXED

`dim` attr arriving as a 0-dim tensor (hand-minted Const-evaluated)
is unwrapped via `.item()`; a non-scalar tensor still passes through
to fail loudly downstream, as before.
Evidence: `test_dim_args_unwraps_0dim_tensor`.

### B7 — `k == "dim"` only tuple-kwarg key — SUBSUMED by B1

The special case is gone; the generic `list`/`tuple` branch records
`attrs[k] = tuple(v)` for every key.

## Verification

- `pytest tests/test_torch_bridge_audit_fixes.py` — 22 passed.
- `pytest tests/test_torch_bridge_edges.py tests/test_intake_defects.py
  tests/test_batch_norm_expand.py tests/test_sdpa_expand.py
  tests/test_torch_integration.py tests/test_corpus_models.py
  -m "not slow"` — all pass.
- `ruff check` + `ruff format --check` on `torch_bridge.py` — clean;
  `ty check` — 0 errors.
- Coverage: every new line/branch in the diff is exercised (the
  audit-file tests plus real exports); the only partials in the
  touched regions are pre-existing (`clamp` `argpos`, covered by
  other suite files).
- Targeted regression slice — every test file that imports
  `torch_bridge`/`export_to_ir` (~90 files, `-m "not e2e and not
  slow"`): **2253+362+200 passed**; the only failure attributable to
  this change was `test_semantic_ops.py::test_handle_copy_early_returns`
  (pinned the old silent-decline contract — updated to assert the
  rejections).  An A/B run with `torch_bridge.py` temporarily restored
  to HEAD showed the remaining 18 failures in the slice
  (`test_batched_verify`, `test_decode_laws` ×2, `test_headshare`,
  `test_hybrid` ×3, `test_modelproj` ×3, `test_regime`,
  `test_regime_edges`, `test_scan_fused` ×2, `test_semantic_ops` ×4,
  `test_aff_diag`) **all reproduce on unmodified HEAD** — they are
  pre-existing/environmental (concurrent WIP in
  `mlopt_discovery`/`model_corpus`/eqsat owned by other agents), not
  regressions from this change.
- `ruff check` + `ruff format --check` on `torch_bridge.py` — clean.
  `ty check` — the only diagnostics are in
  `mlopt_discovery/workload_gen.py` (another agent's WIP file);
  `torch_bridge.py` itself is clean.

## Notes / non-goals

- The stale one-line module docstring (audit D7) was left alone —
  out of scope for the B-fixes.
- Hand-minted `batch_norm` terms without the `running_stats` attr
  keep the legacy heuristic; the ambiguous `(w, b)`/`(rm, rv)`
  spelling is only resolvable via the export tag (documented in
  `_batch_norm_torch`'s docstring).
- `git stash` was used mid-session to A/B the two pre-existing test
  failures; it captured a concurrent agent's in-progress
  `model_corpus.py` edit.  My files were restored via
  `git checkout stash@{0} -- <paths>`; **`stash@{0}` still exists and
  contains that agent's model_corpus.py snapshot — do not drop it
  without checking with them.**

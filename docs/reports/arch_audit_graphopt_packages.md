# Architecture audit — graphopt/packages

Read-only audit (subagent). Scope: torch_bridge.py read IN FULL
(4,227 lines), adapters, report, backend, device, device_placement,
export, folds, runners, executors/base, onnx opmap/liftmap (partial),
cuda runners, carrier entry points, discovery play/vocab/vocabdir/
engine headers, eqsat attrs.py.

## 1. Bugs (silent-wrong first)

- **B1 HIGH — FX-node and tuple kwargs silently dropped at export**
  (torch_bridge.py:837-870). node.kwargs loop handles dim-lists,
  int/float/bool, list, str, dtype, device — nothing else:
  - FX-node kwargs (tensor operand as keyword, e.g. attn_mask=m):
    falls through all isinstance → operand never appended → binding
    sees fewer operands → mask silently absent → unmasked attention.
  - tuple kwargs (stride=(2,2), size=(8,8), pad=(1,1) on
    F.pad/interpolate/conv2d): `isinstance(v, list)` misses tuple →
    attr dropped → binding falls to aten defaults → silent wrong.
  Fix: add tuple branch + FX-node branch (hasattr name → env[v.name]),
  mirroring _opaque_encode which already handles both correctly.
- **B2 MEDIUM — unhandled FX node.op kinds silently skipped**
  (603-908): no else clause — call_module/call_method produce no env
  entry → downstream operand drops (831-832) → binding defaults
  (linear bias→None). Fix: else: raise for ops not in
  {placeholder,get_attr,call_function,output}.
- **B3 MEDIUM — _handle_copy_ silent early-returns drop writes**
  (491-527): dst/src/root None or missing base binding → return
  without minting node.name → mutation silently dropped → downstream
  reads hit B2's operand-drop. masked_fill_ arm deliberately raises —
  copy_/fill_/zero_ should match: unresolvable write target = export
  rejection, not no-op.
- **B4 MEDIUM — _batch_norm_torch misreads operands when running
  stats absent** (1144-1166): rm,rv = a[-2],a[-1] unconditionally;
  track_running_stats=False exports running_mean/var=None (dropped)
  → a=(w,b) → weight/bias read as running stats → F.batch_norm runs
  with WEIGHT MATRIX as running_mean — silent wrong. Needs arity/flag
  guard or loud refusal. Test: nn.BatchNorm2d(track_running_stats=
  False).
- **B5 LOW — _where_torch ignores surplus children** (1132-1141):
  extra children beyond two slots silently ignored.
- **B6 LOW — _dim_args returns raw evaluated args** (2995): Const dim
  arrives 0-dim tensor → Tensor.sum rejects (loud). Hand-minted only.
- **B7 LOW — k=="dim" is the ONLY tuple-kwarg key recorded** (843) —
  subsumed by B1.

## 2. torch_bridge.py structure map (4,227 lines)

32-292 literal tables (_ATEN_TO_IR ~100, _IR_TO_TORCH_EXTRA ~25,
_LOCAL_ATTR_SCHEMA ~15, _SCALAR_OPERAND_OPS ~50); 295-535 name
canonicalisation + copy_/view helpers; 538-921 export_to_ir (~380-line
FX lifter); 924-985 graph root/outputs/attr resolve; 988-1236 named
binding helpers; 1404-1669 opaque-leaf machinery; 1672-1755
_extreme/_clamp; 1764-2654 _CORE_TORCH_BINDINGS ~190-entry dict;
2657-2835 shape rules + register_shape_rule; 2838-2905 ambient
bindings; 2908-2940 _om_elem/_om_compose (carrier monoid kernels —
LAYERING SMELL, belong in mlopt_carriers.om); 2943-3003 split/creator/
dim helpers; 3006-3285 _EvalPlan; 3288-3407 const/mint/device/cast
helpers; 3410-3540 eval_term; 3543-4211 IRModule; 4214-4227
ir_to_torch_module.

Natural splits: torch_export.py (32-985), torch_bindings.py
(988-3003), torch_eval.py (3006-4227).

Binding dispatch is ALREADY table-driven (_CORE_TORCH_BINDINGS dict →
OpTable), not an if-chain. ~190 hand lambdas; ~60% follow 2-3
patterns, ~25 genuinely irregular named functions. A declarative spec
could collapse the repetitive third but irregular tail means a
generator emits data+thin wrappers, not zero code.

## 3. Hardcoded vs generated — inventory

| Table | Hand-maintained? | Generator? |
|---|---|---|
| _ATEN_TO_IR + _IR_TO_TORCH_EXTRA | Yes — hand-curated alias/overload | host-tensor has aten_port_list.txt (207 pins) + generators driven by aten_schemas.json (3,757). NO torch-side equivalent — torch bridge re-derives same aten→op knowledge by hand incl. measured walls (min.other, tile, linalg_inv). A torch_port_list + schema-driven emitter for the NAME-CANONICALISATION half is feasible; binding bodies not fully generatable (torch callables ≠ host kernels). |
| _LOCAL_ATTR_SCHEMA | Yes | Derivable from aten_schemas.json (data exists in host-tensor/tools/) |
| _SCALAR_OPERAND_OPS | Yes — semantic class | Derivable: "does overload take Scalar at this position" is schema data |
| _FFT_SHAPE_OPS | Yes (4) | Schema-derivable |
| _CORE_TORCH_BINDINGS | Yes ~190 lambdas | Partially: ~40 pure passthroughs (torch.cos etc.) emittable from port list; attr-carrying ops cannot |
| ATTR_SCHEMA/ATTR_REQUIRED (eqsat) | Yes substrate-owned | Canonical contract; _LOCAL_ATTR_SCHEMA exists because it can't be mutated downstream |
| ONNX ONNX_BINDINGS/ONNX_LIFTS | Yes but WELL-FACTORED | Declarative _UNARY/_BINARY/_COMPARISON/_CASTS/_REDUCERS + named builders — THE MODEL the torch table could imitate |
| Carrier law tables | Yes — domain content | N/A — laws are the product |
| vocab fragment files | ALREADY GENERATED (vocabdir emit_all/sync_projections) | Done (plan 0033) — repo's answer to generated tables |
| lawdata weight tables | Generated by scripts/train_* | Done |
| COLLECTIVE_OP_NAMES etc | Yes small curated | Low value |
| supported_ops | Derived frozenset | Good — not hardcoded |
| dtype tables (_TORCH_TO_ST/_ST_TO_TORCH/_CASTS/_torch_dtype ×4) | YES ×4 parallel | Should be ONE shared dtype registry (name↔torch.dtype↔numpy↔safetensors) — drift risk |
| _SHAPE_MEMO+_shape_of | DUPLICATED in 6 carrier files | One shared helper — each re-declares same dict+wrapper |
| Verify paths | THREE different gates: report.verify_equiv (max_rel+rtol floor+RNG+park), export._verify_forward (assert_close), onnx executor | Inconsistent metrics/tolerances — a delivery verified by one path isn't certified by another. Unify or document. |

## 4. Duplication / drift

- D1 _flatten_outputs (report.py:138) vs _diff pytree walk (export.py:363)
  — two hand-rolled pytree impls (+likely a third elsewhere).
- D2 _input_env logic duplicated: executors/base.py:228-242 vs
  IRModule.forward fallback (torch_bridge.py:4131-4136).
- D3 _park_params (3935) vs _align_to (3814) ~70%-identical table-
  rewrite loops, already disagree subtly — shared _rewrite_live_tables.
- D4 _apply (4007) third variant of same pattern.
- D5 _torch_dtype ×4+ dtype tables — see §3.
- D6 _GRAPH capture recipe duplicated: executors/base.py:180-220 vs
  cuda/runners.py:107-132 — share capture(module,xs,warmup) helper.
- D7 doc drift: torch_bridge.py:2 docstring is stale one-liner;
  AGENTS.md doesn't document the 4k-line hub.
- D8 folds._eval_const permissive eval swallows binding exceptions →
  None → fold skipped (correct-but-slower; documented).

## 5. Other notes

- play.py 4,938 lines — DOMAINS registry+CLI+per-domain generators in
  one file; natural play/ package + domains/ split (kernelopt's
  spec_domain proves seam works).
- engine/scoring/advantage/players/lawdata.game = documented compat
  shims to gamekit — correct per plan 0032.
- onnx/opmap.py best-factored table in repo. Verify: _arg_reducer/
  _concat/_select/_unsqueeze pass dim/axis UNNORMALIZED (negative
  axes) vs opset-18 pin; _flip→Slice step=-1 negative-step opset
  support — e2e suite likely covers but worth checking.
- cuda/runners.py clean; storage-fingerprint invalidation well-
  reasoned; device_graph_key all-CUDA honest.
- device.py/device_placement.py consistent.

## 6. Prioritized recommendations

1. Fix B1 (kwargs FX-node + tuple drop) — highest-value silent-wrong;
   test kwarg-tensor-operand + tuple kwarg (F.pad pad=(1,1),
   F.conv2d stride=(2,2)).
2. Fix B2 — else: raise on unhandled node.op.
3. Fix B3 — _handle_copy_ silent returns → loud rejections.
4. Verify B4 — track_running_stats=False BatchNorm; test either way.
5. Split torch_bridge.py → export/bindings/eval; move
   _om_elem/_om_compose to mlopt_carriers.om.
6. Unify dtype tables (one registry) + shape-memo helper (one in
   carriers).
7. Torch-side port-list generator: aten_port_list-equivalent feeding
   _ATEN_TO_IR/_LOCAL_ATTR_SCHEMA/_SCALAR_OPERAND_OPS/_FFT_SHAPE_OPS
   from aten_schemas.json; binding bodies stay hand-written (ONNX
   grouping pattern shrinks repetitive third).
8. Split play.py along DOMAINS.
9. Document/unify three verify gates.

## Could not do (read-only)

Run tests, confirm B1/B4 reachability in practice (whether
torch.export actually emits kwarg-carried FX nodes/tuples for the
suspect ops — code path unambiguously wrong IF reached; 10-line
repro settles severity), check ONNX negative-axis/step opset
behavior vs onnxruntime.

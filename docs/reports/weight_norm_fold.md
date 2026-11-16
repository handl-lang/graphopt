# `aten::_weight_norm` — binding + construction-time fold

Worklist item #2 of `tts_stt_coverage_map.md`: 70 opaque
`aten::_weight_norm` leaves = 80% of the Kokoro decoder's opaque
census, ~6.36 ms ≈ 5% of the ~123 ms eager forward.

## Approach chosen: (B) bind + const-fold at `IRModule` build

Both halves landed; they are complementary, not alternatives:

1. **Binding** — `aten::_weight_norm` now lifts to a real IR op
   `weight_norm` instead of an `opaque` passthrough, so laws,
   collectors, the cost model and shape inference can see it.
2. **Fold** — every *parameter-only* `weight_norm(v, g, dim)` subtree
   evaluates **once** at `IRModule` construction and materialises as a
   fused `Param` (the existing `_fold_weight_chains` machinery), so
   the 70 per-forward recomputations disappear from the delivered
   graph entirely.  Data-dependent instances keep the runtime
   binding.

Approach (A) alone (binding without the fold) was rejected: it makes
the leaf *visible* but leaves the 5% on the table — the aten kernel
recomputes `g·v/‖v‖` on every forward even though `v`/`g` are frozen
parameters.  Approach (3) — a carrier/value-pass law — was rejected:
`mlopt_core.opmeta.FOLD_OPS` is a frozenset snapshot inside eqsat
(read-only for this change) and the fold alphabet is deliberately a
positive list; the right seam turned out to be the lowerer itself,
which already owns exactly this fold for `FOLD_OPS` members.

## Where each piece lives

| piece | file |
|---|---|
| op declaration (`register_op_meta`) | `packages/kit-torch/src/kit_torch/torch_bindings.py:57` |
| `_weight_norm → weight_norm` alias | `torch_bindings.py:169` (`_ATEN_TO_IR_OVERRIDES`) |
| `dim` positional-attr schema `{2: "dim"}` | `torch_bindings.py:259` (`_LOCAL_ATTR_SCHEMA_OVERRIDES`) |
| `_weight_norm_torch` binding | `torch_bindings.py:1010` |
| `"weight_norm"` in `_CORE_TORCH_BINDINGS` | `torch_bindings.py:1943` |
| shape rule (output = `v`'s shape) | `torch_bindings.py:2128` |
| `_EXTRA_FOLD_OPS = {"weight_norm"}` | `packages/kit-torch/src/kit_torch/torch_eval.py:39` |
| fold dispatch (`FOLD_OPS or _EXTRA_FOLD_OPS`) | `torch_eval.py:723` |

## Semantics — exactness argument

`torch.nn.utils.parametrizations._WeightNorm.forward` calls
`torch._weight_norm(weight_v, weight_g, self.dim)`; torch.export
emits `aten::_weight_norm.default(v, g[, dim])` with schema
`aten::_weight_norm(Tensor v, Tensor g, int dim=0) -> Tensor` —
`dim` is **elided at its default 0**, so the common spelling has two
operands and no attr.

- The binding calls `torch._weight_norm(v, g, dim)` — the *same*
  function the parametrization calls (verified:
  `torch._weight_norm(v,g,d)` ≡ `torch.ops.aten._weight_norm(v,g,d)`,
  `torch.equal` → `True`, max diff `0.0`).  `dim` is read from the
  canonical `dim` attr, the legacy `arg2` spelling, or a trailing
  positional; absent → `0` (the aten default).  `g=None` is rejected
  loudly by torch itself — honest failure for a malformed term.
- The fold evaluates the *same binding* once via
  `eval_term(param_env=…, tensor_only=True)` inside
  `_fold_weight_chains` — bit-identical inputs ⇒ bit-identical fused
  weight (tests assert `torch.equal`, not `allclose`).
- Fold gating is unchanged from `FOLD_OPS`: `not _uses_input(term)`
  (no `Var` leaf anywhere in the subtree) **and** all args already
  `Param`/`Const` leaves **and** the eval returns a tensor.  Any miss
  leaves the term as a runtime `weight_norm` op — never a wrong
  answer.

## Measured payoff

Synthetic bench (CPU, fp32): 70× `weight_norm(Conv1d(32,32,3))` +
SiLU stack, export → `ir_to_torch_module` → forward parity
(`allclose`), then 50 timed forwards:

| arm | ms/forward |
|---|---|
| eager source module | 3.383 |
| lowered `IRModule` (folded) | 2.522 |

**−25.5%** on this microbench — the weight_norm recompute is a larger
share here than in Kokoro's real conv stack (tiny 1-D convs).  The
coverage map's ~5% (~6.36 ms of ~123 ms) is the expected decoder
payoff; the GPU probe was not re-run (shared GPU, minutes-scale).

Secondary win: each folded site also *shrinks the delivered weight
set* — the `weight_v`/`weight_g` pair collapses to one fused
`Param`, and unregistered originals drop out of `_param_map`.

## Tests

`tests/test_torch_bridge_edges.py` (+4):

- `test_weight_norm_param_only_folds_to_fused_param` — fused param is
  `torch.equal` to `torch._weight_norm(v, g, 0)`; v/g unregistered;
  forward exact.
- `test_weight_norm_dim_attr_folds` — `dim=1` rides the canonical
  attr into the fold.
- `test_weight_norm_data_dependent_stays_runtime` — `weight_norm(x,
  g)` over an input `Var` does **not** fold (survives in `_root`),
  evaluates via the binding, output `torch.equal` to spelled torch.
- `test_export_weight_norm_mints_bound_op_not_opaque` —
  `parametrizations.weight_norm` exports as `weight_norm` (never
  `opaque`), `dim` attrs `{0, 1}` canonical, lowered module
  bit-identical.

Gate run: `pytest tests/ -q -k "torch or weight or semantic or
contract or export" -m "not slow"` → 2503 passed; the 7 failures are
**pre-existing/environmental** (verified by stashing this change:
missing `torchvision` ×2, `mlopt_onnx` tan mapping, 4
`test_semantic_ops` failures from the in-flight discoverykit
refactor — none touch weight_norm).  `ruff check`/`ruff format
--check` clean on both edited modules; `ty check` shows 406
pre-existing diagnostics, all in `mlopt_discovery` (mid-refactor),
none in the touched files.

## Notes / follow-ups

- `weight_norm` is declared via `register_op_meta` with empty tags —
  it cannot join `FOLD_OPS` (eqsat snapshot) so the cost model's
  `_folds_to_param`/extraction discount do **not** price it at ~0
  marginal.  Extraction may still bill the op per-call; a future
  eqsat change could ship it in `_CLASSIFIED` with the `FOLD` tag and
  drop `_EXTRA_FOLD_OPS`.
- `norm_except_dim` (the `right_inverse` path) is init-time only and
  was not needed.
- Kokoro's remaining opaque leaves after this: `stft`/`istft` (#3),
  `upsample_*`/`conv_transpose1d`/`add_`/`angle`/`rand_like` (#6).

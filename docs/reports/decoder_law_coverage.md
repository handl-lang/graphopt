# Decoder law coverage — spelled-attention → sdpa

Follow-up to `decoder_search_scaling.md`.  Post-mortem question: after
the `_shape_of` memo fix made decoder search terminate in ~9 s, why
did it yield ~zero rewrites (~25 enodes, 1 pairing group)?  This
documents the census, the three distinct coverage gaps found, the fix
that landed, and the honest measurement.

**Verdict: the diagnosis was right, and it was THREE gaps, not one.**
The spelled eager-attention decomposition *is* mostly covered by law
LHS patterns — the blockers were (1) an eval-mode `dropout` wrapper
with no mask-free fold variant, (2) the KV-cache `concat` whose empty
lazy-init operand poisons `_shape_of` to `(0,)`, vetoing every
shape-gated check, and (3) — measured but not blocking — identity
relayout ops (`detach_`/`contiguous`) that only prior-art rules strip.
Fix: `ATTN_SDPA_LAWS`, 8 spelled-attention→`sdpa` fold rows added to
`mlopt_carriers.decode_laws` (the decode-slice law module).  On a
single decoder layer both attention blocks now fold: extraction is
`sdpa=2, matmul=0, softmax=0`, `lower(verify=True)` passes, and the
delivered module is **faster than eager** on CPU (7.1–8.4 ms vs
9.0–11.6 ms).  Full decoder: 12 folds fire (6 layers × self+cross),
search still ~11 s.

## Setup

- `openai/whisper-base` fp32 CPU, `attn_implementation="eager"`.
- One decoder *layer* (`decoder.layers[0]`) exported standalone via
  hook-captured decode-step inputs (seq len 1): `hidden_states`
  (1,1,512), causal mask (1,1,1,1), `encoder_hidden_states`
  (1,1500,512), fresh `EncoderDecoderCache` per call — the KV-cache
  `cat` plumbing is inside the traced graph.
- Probe: `bench/drivers/probes/decoder_layer_census.py` (op census +
  per-rule match/check survival table) and
  `bench/drivers/probes/decoder_layer_pay.py` (search → extract →
  lower+verify → eager-vs-delivered timing).

## IR op census vs law-LHS coverage

Exported layer: 66 distinct ops, 13 op kinds — `linear`×10,
`transpose`×10, `reshape`×8, `dropout`×6, `contiguous`×6, `add`×4,
`matmul`×4, `mul`×4, `concat`×4, `detach_`×4, `layer_norm`×3,
`softmax`×2, `gelu`×1.  99 e-nodes / 99 classes at saturation input.

Law-LHS op census over `default_rules()` (331 rules = core DEFAULT +
carrier CARRIERS): every IR op kind except `layer_norm`, `detach_`,
`contiguous` appears on some LHS.  Match-level table (substitutions →
surviving check/derive), before the fix:

| rule | lhs-substs | survive |
|---|---|---|
| transpose_noop | 10 | 0 |
| naturality_scalar_rev | 2 | 2 |
| matmul_t_concat | 2 | 0 |
| sdpa_fold_addmul_drop | 1 | 0 |
| decay_state_lift | 1 | 0 |

So coverage was *nearly* there: `sdpa_fold_addmul_drop` (core
DEFAULT) matched the self-attention block but its check vetoed; the
cross-attention block matched **nothing**.

## The three gaps (ranked by what actually blocked the fold)

### Gap 1 — eval `dropout` wrapper, no mask-free `_drop` variant

`eager_attention_forward` spells
`matmul(dropout(softmax(·), p=0.0, train=False), v)`.  Core's
`sdpa_fold_*_drop` variants exist **only inside the add/masked_fill
family** — the mask-free rows (`sdpa_fold_nomask`,
`sdpa_fold_mul_nomask`, `sdpa_fold_div_nomask`) have no dropout
wrapper, so cross-attention (`encoder_attention_mask=None` → no mask
add) could never fold.  This is a pure family-completeness gap in the
core bank.

### Gap 2 — `(0,)` empty-cache concat poisons `_shape_of`

The KV-cache append is
`concat(dim=-2){detach_(contiguous(c_*_lifted_tensor)),
transpose(reshape(linear(…)))}` where `c_*_lifted_tensor` is the
`DynamicLayer.lazy_initialization` placeholder `torch.tensor([])` —
shape `(0,)`.  `torch.cat` skips numel-0 operands at runtime (verified:
`cat([empty(0), x]) == x` for any rank), but `typing._shape_of`'s
concat rule sums *every* operand's cat axis → `(0,)` → rank 1 →
`axes-last2` on the bound `K` transpose fails → `sdpa_fold_addmul_drop`
check vetoes (measured: `axes-last2 K TD1 TD2 → False` with
`_shape_of(K) == (0,)`).  Same poison hits the cross-attention K/V
concats.  `detach_`/`contiguous` themselves pass shape through fine —
the *operand* is the problem, not the wrappers.

Note this is **first-step-only**: with a non-empty past, the cache
leaf carries a real `(B,H,T,D)` shape and the concat infers correctly.
Decode-step-1 IR is exactly where the `(0,)` appears.

### Gap 3 — identity ops have no DEFAULT coverage (measured, not blocking)

`contiguous_id`, `detach_id`, `detach_inplace_id`, `dropout_eval`,
`dropout_zero`, `clone_id`, `alias_id` all exist — in
`mlopt_core.laws.priorart.PRIOR_ART_RULES`, deliberately held out of
`DEFAULT`/`ALL_RULES` (opt-in `prior_art` preset).  Adding the five
relevant ones to the layer's ruleset strips the wrappers (+20
enodes) but **does not unblock the fold** — gap 2 vetoes regardless.
Recorded for completeness: they are not required by the landed fix.

## The fix that landed

`packages/mlopt-carriers/src/mlopt_carriers/decode_laws.py` —
`ATTN_SDPA_LAWS`: 8 rows, {masked, mask-free} × {mul-scale,
scale-free} × {bare, eval-`dropout`}, tagged `CARRIER`+`DECODE`
(package convention), shipped inside `DECODE_LAWS` → `DECODE_RULES` →
the `CARRIERS` preset → `default_rules()` (331 → 339 rules).

- LHS mirrors the core `sdpa_fold` family
  (`matmul([dropout](softmax([add](scores))), v)` →
  `sdpa(q,k,v[,m], scale=s)`), with two deliberate differences:
  - the dropout wrapper spells `train=False` **literally** (eval
    dropout is the identity for any `p` — exact; the core `_drop`
    rows bind `train` as a metavar and would also fold `train=True`,
    which is not the identity);
  - checks re-implement the core preconditions (K-transpose on the
    last two axes, softmax on the last axis, Const numeric scale)
    against `_eff_shape`, a cat-empty-aware resolver.
- `_eff_term`/`_eff_shape`: bottom-up, memoized rebuild that drops
  *provably*-empty concat operands (any literal `0` dim) before
  delegating to `_xshape`.  `None`/non-tuple shapes stay live, so
  unprovable cases decline exactly where the plain resolver would;
  the rebuilt term is used for shape reasoning only, never minted.
- Scale lands in the `sdpa` `scale` attr via the same
  `_DSPEC_SCALE_MUL`/`_DSPEC_SCALE_ONE` specs the core folds use.

Hand-written carrier law (human-authored, like its `DECODE_LAWS`
siblings) — no `admit_prim_record`/gauntlet path applies; verified
through the repo's carrier-law convention: `tests/test_decode_laws.py`
section 5 (fires per row, declines per guard, fp64-exact extracted
`sdpa` vs spelled reference, empty-concat firing, `train=True`
decline), 100% statement+branch coverage on the module.

## Prototype measurement (single layer)

`decoder_layer_pay.py`, `Optimizer.search` stock path:

| metric | before | after |
|---|---|---|
| search | ~1.1 s, ~0 folds | 1.2–1.6 s, `attn_sdpa_mask_mul_drop`×1 + `attn_sdpa_mul_drop`×1 |
| extracted term | spelled matmul/softmax | **`sdpa`=2, `matmul`=0, `softmax`=0** |
| `causal_specialized` | — | True (mask → `is_causal`) |
| `lower(verify=True)` | n/a | **passed** |
| CPU time/repl | eager 9.0–11.6 ms | delivered **7.1–8.4 ms** (~20–28% faster) |

Full decoder (`decoder_search_fixed.py`): `attn_sdpa_mask_mul_drop`×6
+ `attn_sdpa_mul_drop`×6 = all 12 layer-attention blocks fold;
search ~11 s (baseline ~9–10 s — the +8 rules cost ~1–2 s, within
run-to-run noise).  One caveat surfaced: printing `res.stats` now hits
a `MemoryError` inside `mlopt_core.ir.op_repr` — it recurses the
extracted term's DAG exponentially.  Pre-existing eqsat-side issue
(triggered whenever a stats value holds a large term; the earlier
post-mortem run printed smaller stats), not caused by the folds —
but worth an upstream `op_repr` memo/`repr` budget.

## Honest boundaries

- **Paying but modest at this shape.**  Seq-len-1 decode: attention
  scores are `(B,H,1,T)` — the GEMMs dominate; sdpa fusion saves the
  softmax + two small matmuls + kernel launches.  ~20–28% on CPU is
  real but not dramatic; the win grows with prefix length and on GPU
  (one fused kernel vs three ops + mask add).  The delivered module
  uses the same `F.scaled_dot_product_attention` binding an
  `attn_implementation="sdpa"` export would produce — so this is
  parity-with-sdpa-export *plus* it works on the eager spelling
  (and any other decomposed export: ONNX lifts, handwritten IR).
- **Downstream carrier laws still decline on the same `(0,)`.**  The
  folded `sdpa(q, concat(…), concat(…), …)` keeps the concat operands
  — `SDPA_CAT_LAWS`/`om` chunking checks hit the identical
  `_shape_of` poison, so enode growth stayed small (+6).  The
  highest-leverage *upstream* fix is teaching `typing._shape_of`'s
  concat rule the numel-0-operand semantics (eqsat is read-only for
  this task) — one change there would unblock every shape-gated law
  at once, including the core `sdpa_fold` family on step-1 decode IR.
- **Not covered spellings:** `div`-scale (`sdpa_fold_div_nomask`
  exists in core but no `_drop` variant and same `(0,)` veto),
  `masked_fill`-style masks (whisper uses additive float mask →
  covered), strided/`permute` transposes, rotary (whisper has none —
  learned absolute positions).
- **Remaining thin coverage:** `layer_norm` (3 nodes), `gelu`,
  `contiguous`/`detach_` have zero DEFAULT LHS coverage —
  `prior_art`'s identity laws cover the latter two if ever wanted in
  the default set (deliberate preset decision, not taken here).
- **Full-decoder delivery still blocked** by the `where`/`arg2`
  torch_bridge binding gap (where-fix agent's area) — unchanged.
- **Unrelated test failure observed:** `tests/test_aff_diag.py::
  test_batched_scan_module_handles_applyd` fails
  (`is_batched == False`) — uses `SCAN_DIAG_LAWS` only, does not
  touch `DECODE_LAWS`; pre-existing/environmental (torch meta-trace).

## Alternatives evaluated

- **(a) Laws for the decomposition** — what landed.  Core already had
  `sdpa_fold_*`; the gap was family completeness + the `(0,)`
  shape poison, both worked around decode-side without touching
  eqsat.
- **(b) Collector/route** — `_seq_scan_sites` & `TIER_SITES` mint
  *handler specs* for the genkernel codegen tiers, a different
  mechanism than e-graph laws; an attention-site collector could mint
  sdpa specs for Triton codegen later, but the law path already
  delivers the fold into the e-graph where extraction/cost decide.
- **(c) sdpa spelling normalization** — implemented *as* the laws
  (canonicalize spelled→sdpa inside saturation).  The zero-law
  alternative remains `attn_implementation="sdpa"` at export — worth
  documenting as the cheapest path when the model config allows it.

## Files

- `packages/mlopt-carriers/src/mlopt_carriers/decode_laws.py` —
  `ATTN_SDPA_LAWS` (8 rows), `_eff_term`/`_eff_shape`,
  `_numel_zero`, `_norm_axes`, `_check_spelled_sdpa{,_scaled}`,
  `_spelled_sm`, `_pattern_vars`, `_attn_sdpa_rules`; added to
  `DECODE_LAWS`/`__all__`.
- `tests/test_decode_laws.py` — section 5: 12 new tests (family
  shape/names, per-row firing, scale attr, fp64 exactness, empty-concat
  firing, train-dim/non-last-two/wrong-dim/non-const-scale/div
  declines, helper unit coverage).  Module coverage 100%.
- `bench/drivers/probes/decoder_layer_census.py`,
  `bench/drivers/probes/decoder_layer_pay.py` — new probes.

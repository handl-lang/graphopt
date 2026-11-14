# Law transfer — do minted/admitted laws fire off their minting corpus?

Measurement driver: `bench/drivers/probes/law_transfer_probe.py`
(data: `docs/reports/law_transfer.json`, this run 2026-11-14 — CPU fp32,
`default_rules()` = core `DEFAULT` + the carrier preset, the shipped
`rules=None` composition; saturation bounds = `symmetry_budget=2048`
per `EXPANSIVE` rule, `max_nodes=100_000`, 120 s/module wall budget —
every module reached `fixed_point`, none hit a bound).

## Question

The bank ships 71 machine-authored laws beside the hand kernel —
21 depth-1 `prim:synop_*` folds, 2 `synop2_*`, 34 `synop3_*`,
`conv_bn_fold`, and 1 `AUTO_TIER` record (`prim:synop_6ed0a07c7d`,
the first auto-admission).  Every prim record carries `mods` — the
zoo modules whose op-patterns the mint saw.  Does a law minted on
`GatedLinearAttention`/`MambaSelectiveScan` shapes fire on a whisper
decoder graph, an off-corpus zoo module, or a `model_corpus` mutant?

## Module set (27 rows, all `fixed_point`)

| group | modules | fires (distinct rules) |
|---|---|---|
| zoo mint-corpus | GatedLinearAttention, MambaSelectiveScan, RealGatedLRU, GatedDeltaRule, BlockSparseAttention, SlidingWindowAttention, RWKVMixBlock, DecoupledMLA | 2–60 |
| zoo off-corpus | ChunkedRetention, RMSNormPerHead, ConvNeXtStage, GEGLUBlock, RotaryPair, DiffAttention, AdditiveAttention, TalkingHeadsAttention, SigmoidAttention, SqueezeExcite | 0–5 |
| whisper-base (CPU export) | decoder step (628 enodes), encoder (326 enodes) | 4 / 2 |
| whisper-shaped fragments | attn_core, mlp_block, conv_block, o_proj, ln_res | 0–3 |
| model_corpus mutants | 2× ChunkedRetention, 2× DiffAttention (`generate_mutants`, donor stats over the probe's own exports) | 4–8 |

"Fires" = `EGraph.rule_fires` merge counts; `app_fires` (witnessed
firings incl. non-merging) is recorded per row in the JSON.

## Finding 1 — the prim folds are corpus-locked

**57 of 58 `prim:synop*` folds fired exclusively on their minting
modules.**  `fired_on ⊆ minted_on` held for every fold; the fired
set is typically a strict subset (a fold minted on 4 modules fires
on exactly those 4; the `synop3_*` family minted on
{GLA, MambaSelectiveScan, RealGatedLRU} fires on exactly those 3).

The single exception is `prim:synop_bb8c54c201` — the widest-`mods`
record (14 modules spanning the attention/gated-scan families).  It
fired on 9 probe modules, all on-corpus **plus one genuine transfer**:
`mutant:ChunkedRetention:1`, a `model_corpus` mutant of an
off-corpus module — the only off-corpus prim firing in the whole
run (3 merges).

Concretely: **zero prim folds fired on whisper encoder/decoder, on
any of the 10 off-corpus zoo modules, on the 5 whisper fragments, or
on 3 of 4 mutants.**  The gated-delta/linear-attention manual
spellings the mints saw are simply absent from atomic-op model
exports (`softmax`, `layer_norm`, `linear`, `conv1d` as single ops)
and from the other zoo families' spellings.

What *did* fire off-corpus is the hand/carrier layer:
`attn_sdpa_mul_drop`/`attn_sdpa_mask_mul_drop` (the ATTN_SDPA_LAWS
carrier laws) on both whisper subgraphs, `naturality_scalar_rev`,
`id_add`, `sub_to_add`, `mul_div_swap`, `matmul_div_distribute`,
`square_expand`, `pow_to_square`, `rms_norm_fold_nogain`,
`right_distribute_matmul`, `sdpa_fold_{mul,div}_nomask`, `om_lift`,
`om_decayed_lift`, `decay_lift`, `decay_state_lift_post`,
`assoc_linear_bias`.  These are generic equational laws — they
transfer because their LHS op-patterns (`add`/`mul`/`matmul`/
`transpose`/`unsqueeze`) are spelling-agnostic.

## Finding 2 — dead weight on this module set

9 machine-shipped laws never fired on any of the 27 modules:

| law | LHS ops | likely reason |
|---|---|---|
| `conv_bn_fold` | conv2d, batch_norm | no module spells conv2d+BN (whisper uses conv1d, no BN) |
| `sdpa_fold_nomask` | matmul×2, softmax, transpose | needs the *decomposed* attention spelling; exports carry atomic `softmax` |
| `linear_channel_to_row_scale` | linear, mul | narrow diagonal-scale pattern, no site here |
| `softsign_fold` | div, add, abs | no softsign spelling in the set |
| `transpose_noop` | transpose | no bare transpose-op sites |
| `chunk_single` | chunk | no `chunk` op in any export |
| `mul_unsq_pad_l`, `add_unsq_pad_r`, `mul_reshape_inert_l` | mul/add + unsqueeze/reshape | pad/reshape identity strips — no matching sites |

The AUTO_TIER record `prim:synop_6ed0a07c7d` fired — but only on its
two minting modules (`GatedLinearAttention`, `MambaSelectiveScan`),
6 merges each.  Corpus-locked like the rest.

**Honest caveat:** "never fired" is measured on *this* module set —
small zoo shapes, two whisper subgraphs, 4 mutants.  `conv_bn_fold`
et al. plausibly fire on conv-BN intake workloads the set lacks;
this is a pruning-review candidate list, not a deletion list.
(245 human-classed bank rules also never fired — mostly laws whose
LHS families these modules don't contain; expected, not suspicious.)

## Finding 3 — coverage gaps = future minting targets

Ops present in module IRs that appear on **no** shipped rule LHS
(`coverage_gaps` in the JSON):

| module | uncovered ops |
|---|---|
| whisper:decoder | `layer_norm`×19, `embedding`, `to`, `index`, `repeat`, `arange`, `detach_`×24, `contiguous`×36, `le`, `zeros` |
| whisper:encoder | `layer_norm`×13, `permute`, `conv1d`×2, `embedding`, `arange`, `contiguous`×24 |
| zoo mint-corpus | `einsum`×8, `softplus`, `zeros`, `rsub`×8, `squeeze`×4 |
| zoo off-corpus | `permute`, `layer_norm`, `flatten`/`unflatten`, `cos`/`sin`/`outer`, `arange`, `detach_`, `contiguous` |
| fragments | `layer_norm`, `conv1d` |

The signal that matters: **`layer_norm`, `conv1d`, `embedding`,
`einsum`, `softplus`, `permute`, `flatten`/`unflatten`, `cos`/`sin`**
are real compute ops with zero law-LHS coverage — exactly the
op-patterns a next minting corpus (whisper/conv/embedding-family
models) would need.  `detach_`/`contiguous`/`to`/`arange`/`zeros`/
`le` are export plumbing — worth *excluding* from minting corpora,
not covering.

## Interpretation

The minted folds are **vocabulary, not general laws**: each
`prim:synop*` LHS is a skeleton spelled through the gated-scan
 family's manual ops, so it can only fire where that spelling
 recurs.  That is by design (the mint admits what the gauntlet can
 verify on corpus shapes) but it means the 58-fold family's value
 is concentrated in the ~4-module gated-scan family it was minted
 from.  Cross-model transfer today runs entirely through the
 hand-written generic layer — which is also where the whisper
 fires landed (`attn_sdpa_*`, `naturality_scalar_rev`).

Two honest gaps this exposes, both already-known-shape problems:

1. **Atomic-op exports vs decomposed law LHSs** — `softmax`,
   `layer_norm`, `linear` arrive as single ops; the decomposed
   spellings the folds match live one `_decompose`-shim away
   (`kernelopt_bridge_probe`).  Minting on *exported* model graphs
   (this probe's corpus) rather than hand-decomposed spellings is
   the fix direction.
2. **Minting-corpus breadth** — `mods` lists are zoo-family names;
   nothing in the prim records came from a transformer export.  The
   `model_corpus` machinery (census → mutants → `_case_tail`) is the
   existing path to mint on real model terms; this probe shows its
   mutants do reach new fire sites (`synop_bb8c54c201` on
   `mutant:ChunkedRetention:1`).

## Reproduce

```sh
.venv/bin/python bench/drivers/probes/law_transfer_probe.py
# subsets:
.venv/bin/python bench/drivers/probes/law_transfer_probe.py --only zoo
.venv/bin/python bench/drivers/probes/law_transfer_probe.py --only whisper --no-mutants
.venv/bin/python bench/drivers/probes/law_transfer_probe.py --only mutant --no-whisper --no-fragments
```

Full run: ~3 min wall on CPU (whisper-base download cached; the
saturation itself is seconds per module — the exports dominate).

# Retro — the exposure screen: which models have e-graph-reachable math?

Date: 2026-10-15
Context: two measured lessons set this up.  (1)
`project/retros/birefnet-compile-parity.md` — BiRefNet delivered
0.80× *vs compiled* because its attention is a fused
`scaled_dot_product_attention` call: a black box no law's LHS can
see inside.  (2) `project/retros/retnet-external-baselines.md` —
RetNet delivered a real structural win because its retention was
spelled elementwise.  Discovery only pays where the model's math is
visible to the IR, and nobody had screened the model zoo for that
property.  This retro builds the screen, ranks the corpus by it,
and runs the discovery machinery on the most exposed models.

**Headline.**  Exposure is a clean, bimodal property: of 321
exported workloads, 81 score exactly 0% FLOPs-weighted law
coverage (the `nn.*` fused-op class — BiRefNet's class) while 140
score ≥95%.  Exposure predicts *reachability* but not *payoff*: of
the zoo's eight most exposed models, five fire shipped rules and
four pay (ChunkedRetention −48%, LoRAAdapter −20%, GAU −9.5%,
GCNLayer −7.7%); the other three score 100% and fire nothing.
Running the full admission pipeline
scoped to each model's own term found **one genuinely unpublished
machine-found rewrite that pays and verifies** —
`census:mul_unsqueeze`, minted by the census-naturality generator
from `DeltaRuleBlock`'s own `mul(unsqueeze, unsqueeze)` op-tuple:
`mul(unsq U d, unsq V d) = unsq(mul U V, d)`.  It is `relation=new`,
non-derivable, fires 4× (all typed), and drops the extracted cost
**−12.27%** against the ALL_RULES baseline with a replaying
certificate and a verified lowering.  A strict-match sweep of the
whole corpus shows the pattern exists *only* on DeltaRuleBlock —
the new architecture generated a proposal no corpus term could.

## The screen — `lab/screen/exposure_screen.py`

For each workload (23 zoo, ~250 intake candidates, 43 corpus
models — 321 exported, 2 export-rejected), export to IR and measure:

* `lhs%` — FLOPs-weighted share of op nodes whose op appears in at
  least one LHS pattern of the fireable surface (417 rules:
  `ALL_RULES_WITH_LAYOUT` + `DEFAULT_RULES` incl. carrier presets +
  `ATTENTION` + `SCAN_LAWS`/`SCAN_DIAG_LAWS`; 71 lhs-visible ops).
* `node%` — the same, unweighted.
* `strict` / `rlx` — subterms matching ≥1 LHS strictly / relaxed
  (repeated metavars as wildcards — "the shape is present, the
  equality precondition may not be").
* `invisible` — the op census no LHS names (the black boxes), with
  FLOPs share.  Unpriceable nodes (`_INVALID_COST`, e.g. the
  gather/scatter region of `ExpertChoiceMoE`) are excluded from the
  weights and counted in `unp`.
* `--fire N` — a bounded saturation ground truth on the N most
  exposed terms (DEFAULT_RULES, 6 iters / 60k nodes).

### The zoo table (`--set zoo`, sorted by `lhs%`)

| model | nodes | ops | lhs% | node% | strict | rlx | invisible ops (share) |
|---|---|---|---|---|---|---|---|
| LoRAAdapter | 5 | 3 | 100.0 | 100.0 | 12 | 13 | — |
| GatedAttentionUnit | 11 | 7 | 100.0 | 100.0 | 10 | 11 | — |
| DeepEquilibrium | 12 | 3 | 100.0 | 100.0 | 9 | 9 | — |
| GCNLayer | 13 | 8 | 100.0 | 100.0 | 9 | 11 | — |
| AdditiveAttention | 15 | 8 | 100.0 | 100.0 | 7 | 8 | — |
| CovarianceHead | 5 | 5 | 100.0 | 100.0 | 3 | 3 | — |
| ChunkedRetention | 17 | 13 | 100.0 | 82.4 | 11 | 12 | arange, detach_, contiguous (0%) |
| SoftSlotMoE | 9 | 6 | 100.0 | 77.8 | 2 | 2 | einsum (0%) |
| MoSHead | 8 | 7 | 99.8 | 87.5 | 6 | 7 | einsum (0%) |
| AdaLNBlock | 14 | 7 | 99.3 | 92.9 | 21 | 23 | layer_norm (1%) |
| AffineCoupling | 12 | 8 | 99.0 | 91.7 | 12 | 13 | clamp (1%) |
| MLPMixerBlock | 10 | 5 | 98.7 | 80.0 | 8 | 8 | layer_norm (1%) |
| HighwayGate | 8 | 6 | 98.6 | 87.5 | 11 | 13 | rsub (1%) |
| CosineAttention | 27 | 12 | 98.6 | 81.5 | 17 | 18 | expand_as, linalg_vector_norm, clamp |
| **DeltaRuleBlock** | 57 | 10 | **97.2** | **98.2** | **58** | **66** | zeros (3%) |
| TalkingHeadsAttention | 24 | 8 | 96.2 | 83.3 | 13 | 13 | permute (4%) |
| CapsuleRouting | 30 | 11 | 91.3 | 93.3 | 25 | 32 | zeros (9%), einsum |
| FiLMHead | 7 | 6 | 86.2 | 85.7 | 9 | 10 | layer_norm (14%) |
| ExpertChoiceMoE | 19 | 13 | 85.3 | 73.7 | 4 | 5 | gather, zeros_like, topk, einsum, scatter_add |
| SplineKAN | 5 | 5 | 80.2 | 60.0 | 5 | 5 | sin (19%), einsum |
| WaveNetGate | 9 | 6 | 54.5 | 44.4 | 4 | 5 | conv1d (27%), alias (18%) |
| FNOBlock | 11 | 10 | 11.6 | 54.5 | 8 | 9 | conv2d (73%), fft_rfft2/irfft2, real |
| TimeCondConv | 22 | 17 | 3.5 | 68.2 | 12 | 14 | conv2d (96%), group_norm, sin/cos/arange |

### The corpus picture (`--set all`)

The ranking is bimodal at the op level, not the architecture level.
The 0%-FLOPs tail is exactly the fused-op class — every `nn.*`
native that exports to a single opaque node (`nn.LayerNorm`,
`nn.Conv*`, `nn.*Pool*`, `nn.*Loss`, `nn.Embedding`, `nn.GRU*`,
`nn.LSTM*`, `nn.Upsample*`, `nn.Bilinear`, `nn.CTCLoss`-adjacent
natives), plus hand workloads that are one fused call
(`ResNetBlock`/`TinyCNN`/`MBConvBlock`/`Bottleneck` at 1–8%,
`DepthwiseConv*` at 5.3%).  These are the BiRefNet class: **the
e-graph literally cannot see the computation** — there is nothing
to rewrite.  `sdpa`-spelling workloads (`GQA-sdpa`, `SdpaBiasParam`)
are instructive edge cases: 100% coverage (the `sdpa_fold` family
names `sdpa` in an LHS) but **0 strict matches** — the op is
law-visible, the fold's full shape is absent.  Visible ≠ reachable.

Mid-range models are the interesting honest middle: `LinearAttention`
(intake) at 51% — `cumsum` carries 44% of its FLOPs invisible;
`SelectiveScan`/`RunningNorm`/`CumsumScan`/`CumulativeGateRNN` sit at
33–97% on `cumsum`/`cumprod` shares the laws can't touch;
`SinusoidalEncoding` 69% (sin/cos invisible); `SincKernel` 79%
(sinc/i0 invisible); `DigammaLn`/`TrigHead`/`CoshAcosh` ~33–60% on
special-function tails.

Top of the whole corpus by exposure × size: `TwoLayerHybrid` (120
strict sites), `SelectiveSSM` (84), `HybridBlock` (68),
`DiagDenseSSM` (60), `TransformerBlock` (52), `DiagonalSSM` (50) —
all 100% — and `DeltaRuleBlock` (58 strict, 97.2%).  The SSM/hybrid
family is the most law-dense real architecture in the corpus, which
matches the carriers' measured wins there.

## Does exposure predict firing? (`--fire 8` on the zoo top)

Bounded saturation under `DEFAULT_RULES`, executor-priced cost:

| model | exposure | rules fired | cost in → out | Δ |
|---|---|---|---|---|
| LoRAAdapter | 100% | 1 | 87045 → 69650 | **−19.98%** (`assoc_linear`) |
| GatedAttentionUnit | 100% | 5 | 182973 → 165600 | **−9.52%** (`matmul_div_distribute`, `mul_div_swap`, `mul_square`, `om_decayed_lift`, `square_to_pow`) |
| DeepEquilibrium | 100% | 0 | 208939 → same | 0 |
| GCNLayer | 100% | 1 | 226357 → 208950 | **−7.69%** (`om_decayed_lift`!) |
| AdditiveAttention | 100% | 0 | 261510 → same | 0 |
| CovarianceHead | 100% | 1 | 78365 → same | 0 (`sub_to_add`) |
| ChunkedRetention | 100% | 5 | 252512 → 130740 | **−48.25%** (`decay_lift`, `matmul_div_distribute`, `mul_div_swap`, `om_decayed_lift`, `sub_to_add`) |
| SoftSlotMoE | 100% | 0 | 148145 → same | 0 |

Reading: 100% exposure is *necessary* for the search to see a model
but nowhere near sufficient for payoff — DeepEquilibrium,
AdditiveAttention, SoftSlotMoE and CovarianceHead are fully visible
and pay nothing (no structure the laws improve).  Two findings the
screen alone would not have predicted: **`om_decayed_lift` — the
RetNet decayed-matmul carrier — fires on `GCNLayer`** (the
normalized-adjacency `d_inv·A·d_inv @ X` is a decayed contraction in
disguise, −7.7%), and on `GatedAttentionUnit` (the `relu(uuᵀ)²·V`
score-masking site, part of its −9.5%).  Carrier laws reach real
held-out architectures where the spelled math exists.

## Discovery on the exposed models — `lab/screen/exposed_discovery.py`

Three arms per model, all shipped machinery: (a) `pipeline.propose`
scoped to the model's own op-tuple census → `pipeline.measure` →
`rank` (the full admission gauntlet: oracle → typed gate →
certificate → measured); (b) `enumerate_candidates` over the
model's own op vocabulary, bounded oracle, `impact._probe` on every
new+true candidate; (c) the 11 known unshipped machine laws probed.
Ran on the top-3 by the screen (LoRAAdapter, GatedAttentionUnit,
DeepEquilibrium) plus the four highest-site runners-up (AdaLNBlock,
CosineAttention, CapsuleRouting, ChunkedRetention) and the new
`DeltaRuleBlock`.

### The find: `census:mul_unsqueeze` on DeltaRuleBlock

    (mul (unsqueeze U, dim=D), (unsqueeze V, dim=D))
        = (unsqueeze (mul U, V), dim=D)

Minted by `_census_naturality` from DeltaRuleBlock's `mul(unsqueeze,
unsqueeze)` op-tuple — the `beta_t.unsqueeze(-1) * k_t.unsqueeze(-1)`
sites of the rank-1 write `S += β·k ⊗ (v − kᵀS)`.  Measured verdict:
`relation=new` (no shipped law carries this two-sided naturality —
`mul_unsq_pad_{l,r}` are *strips*, `op(unsq u, v) → op(u, v)`), not
derivable from ALL_RULES, 4 strict matches / 4 typed fires / 0
ill-typed, `paid=1`, verify pass, certificate replays,
closure 1.03×.  Reach: extracted cost `992496 → 870723` =
**−12.27% over the ALL_RULES baseline** (the full shipped
`DEFAULT_RULES` alone gets only −3.5% on this term, via
`decay_state_lift_post`/`sub_to_add`/matmul distribution).  The
*only* gate it fails is the view-oracle's guard isolation
("conditional truth: no single mechanical feature separates") — a
candidate that is true on every real site it found but whose
blanket form the guard machinery cannot certify.  This is exactly
the `mul_unsq_pad` admission story one family over: the law is real
where the dims share, and needs a guard (or a same-rank operand
precondition) the machinery cannot yet write.

A strict-match sweep of all 320 corpus terms: the pattern appears
**nowhere else** — DeltaRuleBlock's 4 sites are the only
`mul(unsqueeze, unsqueeze)` the corpus has ever seen.  The new
architecture contributed an op-tuple the proposal grammar could not
have minted from the corpus.

### Per-model, what fired / paid (pipeline arm)

* **DeltaRuleBlock** (57 nodes, 97.2% exposed, 40 candidates):
  `census:mul_unsqueeze` above; `mixed:mul_unsqueeze_r_w` (16 fires,
  9 typed — the `β·outer` site, conditional `w:v_commutes_view`);
  `mixed:mul_unsqueeze_r_id` (12 fires, conditional `unsq:d_in_pad`);
  `mixed:sub_select_l_w` — `sub(select U,V) = select(sub U,V)` —
  fires 4× but every fire mints an ill-typed member (select over
  broadcast shapes — a real guarded-law lead, correctly refused);
  `mixed:sub_select_l_id` fired 4× and the numeric oracle caught it
  **false** (the honest rejection working); `sub_to_add_dup`
  duplicate.
* **AdaLNBlock** (41 candidates, 7 fire): `grammar:mul_distribute`
  is `new`, `num_true`, fires — and never lowers cost; six
  `mixed:{mul,add}_chunk_*` candidates fire on the triple-chunk
  conditioning site, all conditional (`ck:single` /
  `w:v_commutes_view`), three minting ill-typed members.
* **CosineAttention**: `reshape_transpose` fires 3×, paid=1 —
  **verify_fail=1**: the known fake drop from
  `model-zoo-yield.md`, correctly refused.
* **CapsuleRouting**: four `mixed:mul_unsqueeze_*` candidates fire
  (5 fires total), all conditional, 5 ill-typed mints.
* **ChunkedRetention**: `grammar:sub_to_add_dup` fires once —
  library duplicate.
* **LoRAAdapter / GatedAttentionUnit / DeepEquilibrium**: 35
  candidates each, **0 fires** — their shipped wins
  (`assoc_linear`, the om lifts) are not shapes the census/view/
  grammar proposers mint; the proposal families can't see matmul
  associativity or carrier lifts.

### Enumerate arm — an honest null, with a machinery finding

Across 8 models, the op-vocabulary enumerator emitted 74–400
candidates each (61 `new + numerically-true` on ChunkedRetention
alone — `contiguous`/`pull`/`hom` forms over real ops like
`select`, `sigmoid`, `div`).  **Zero fired on any model.**  The
reason is structural and worth recording: the enumerator mints
*attr-free* patterns (`select X`), but `_term_match` requires exact
attr-key equality — `select(x, dim=1, index=0)` can never match a
bare `select` pattern.  The grammar's unary-op arm is blind to the
attr-bearing ops real exports are made of.  Candidates like
`select(add A B) = add(select A, select B)` are *the same
naturality* the `mixed:*_select_*_w` proposals found on
DeltaRuleBlock — discovered independently, unable to fire.

### Known-laws arm — also a null

The 11 unshipped machine laws (`impact.new_laws()` — mul-over-add
factoring, neg distribution, the annihilators, `x−x`, `x/x`, …)
**fire on none of the eight models**.  They remain synthetic-
witness laws: true, cheap, and absent from real spelled math.

## The new architecture

`zoo._DeltaRuleBlock` (registered as `DeltaRuleBlock`): a DeltaNet-
style unrolled delta-rule recurrence — `S ← S + β·k ⊗ (v − kᵀS)` —
spelled elementwise over `select`/`matmul`/`sub`/`add`/`mul`/
`unsqueeze`/`sigmoid`.  fp64, random-init, ingests clean through
the real export boundary (57 nodes).  Its efficient form — the
chunked delta-rule scan — is **not** in the law library: the
aff/affd scan monoids lift `h′ = a·h + b`, not a data-dependent
`(I − βkkᵀ)` contraction on a matrix state (the shipped
`decay_state_lift_post` does fire on it — a carrier law partially
seeing the decayed structure — but only for −3.5%).  At 97.2%
FLOPs exposure and 58 strict LHS sites it is the most
law-matchable term in the corpus per node, and it is the only
model that minted a paying unpublished candidate this round.

## Verdict

* The exposure screen works as a filter: the 0%-tail (fused-op
  natives, conv stacks) is provably undiscoverable, the 100%-class
  is where discovery can see anything — and where payoff is still
  not guaranteed (DeepEquilibrium, AdditiveAttention, SoftSlotMoE
  score 100% and pay 0).
* **Unpublished machine-found rewrites this round: one that pays
  and verifies** (`census:mul_unsqueeze`, −12.27%, DeltaRuleBlock,
  blocked at ship only by the guard-isolation gap), plus three
  conditional-truth leads (`sub_select_l_w`, `mul_unsqueeze_r_w`,
  `mul_unsqueeze_r_id`) in the same family, and `grammar:
  mul_distribute` (true, fires, pays nothing).
* The honest nulls matter: the generic depth-2 enumerator is
  *structurally unable to fire* on real exported terms (attr-free
  patterns vs attr-bearing ops — fix = mint attr metavariables in
  the grammar's unary forms), and the 11 stored machine laws still
  fire nowhere real.
* The discovery loop closed: a new spelled-out architecture →
  novel census op-tuple → machine-minted candidate → paid,
  verified, certified — and the admission gates correctly stopped
  it at "can't isolate the guard yet" rather than shipping a
  too-broad law.

Probe: `lab/screen/exposure_screen.py` (the screen),
`lab/screen/exposed_discovery.py` (the per-model discovery driver).
Machine-readable: `project/retros/exposed-math-screen.json`,
`project/retros/exposed-math-discovery.json`.

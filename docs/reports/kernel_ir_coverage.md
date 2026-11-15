# kernel-ir coverage census — which mlopt IR ops actually appear, and how many bridge today?

Date: 2026-11-15.  CPU-only (fp32).  Plan-0039 input: an honest
coverage census before kernel-ir vocabulary work.  Question: of the
mlopt IR ops that real lifted modules actually contain, how many does
`mlopt_kernel.ir_bridge.translate` (kernelopt `src/mlopt_kernel/
ir_bridge.py`, lib-kernel-ir v0.1.1's 21 ops) cover — and what is the
difficulty class of the rest?

## Method

Lifted **9 real modules** through `mlopt_torch.adapters.TorchSource
().to_ir` on CPU fp32 (mlopt repo venv — the only one with
`transformers`; `mlopt-torch`/`mlopt-core` there are graphopt
`v0.1.1`, identical to this repo's HEAD on those packages):

| module | recipe | nodes (unique Op) |
|---|---|---|
| `whisper-base:encoder` | r4 `_whisper_modules` `_EncWrap`, feed `(1,80,3000)` | 208 |
| `whisper-base:decoder` | r4 `_DecWrap`, feed `(start, ehs)` | 413 |
| `t5-small:encoder` | `t5_probe.corpus_encoder` recipe, SEQ=128 | 314 |
| `t5-small:decoder` | `t5_probe.corpus_decoder` recipe | 571 |
| `whisper-tiny:decstep` | `mlopt.adapters.whisper` `arm="decoder"` static-K/V wrap | 195 |
| `kokoro:decoder` | `mlopt.adapters.kokoro` arm, text_len=16 | 1,089 |
| `kokoro:decoder-nc` | same, CustomSTFT variant | 1,099 |
| `kokoro:f0n` | same | 194 |
| `minilm` | `all-MiniLM-L6-v2` encoder wrap, (1,32) | 206 |

Two further lifts **declined honestly at `torch.export`** and are
absent from the census: `kokoro:textenc` and `kokoro:full`
(`GuardOnDataDependentSymNode` — `pack_padded_sequence`/length-mask
data dependence; the same wall the bench reports).  That is itself a
finding: the text-encoder arm cannot even be lifted statically.

Each lifted `ir.root` was serialized structure-only (op names, attrs,
DAG child links, leaf names/shapes — tensor payloads dropped; Const
values are irrelevant because the bridge declines Const anyway) and
rebuilt in the kernelopt venv.  Every **DAG-distinct** `Op` node was
run through `ir_bridge.translate` (memoized so shared subtrees
translate once).  Outcomes:

- **bridged** — `translate` returned a `lib_kernel_ir` node;
- **self_blocked** — the failure originates at this node (missing
  counterpart, or a violated constraint: batched ranks, Const
  operand, attrs);
- **child_blocked** — a descendant is the origin.

Scripts lived in `/tmp` (`kir_census_lift*.py`, `kir_ser.py`,
`kir_census_analyze.py`); dumps in `/tmp/kir_terms/`.  Nothing was
committed; numbers below are regenerable.

**Caveats.** Counts are DAG-distinct nodes — a residual-stream
subterm shared by N consumers counts once, so "frequency" is
structural, not dynamic.  `dropout` appears because eval-mode export
keeps it (it is an identity at inference — see Tier-0 below).
`TensorType` shapes survive on leaves; shapes *through* ops are not
propagated, which is exactly what the bridge sees too (its
`_leaf_typ` reads `Var`/`Param` only — `matmul`/`linear` nodes whose
args are computed tensors report `ranks (None, …)` and decline even
when the shapes would be GEMV).

## Headline numbers

- **4,289 unique Op nodes, 71 distinct ops, 9 modules.**
- **Bridged today: 5 nodes — 0.12 %.**  All five are `transpose`
  nodes sitting directly on a leaf.  **0 of 9 modules** contains a
  meaningfully bridgeable subgraph; the only successes are stray
  weight-layout transposes whose parents all fail.
- **Op-name coverage: 12 of 71 distinct census ops have a bridge
  mapping** (`add`, `mul`, `sub`, `neg`, `relu`, `exp`, `tanh`,
  `gelu`, `linear`, `matmul`, `transpose`, `permute` — the mapped
  `sigmoid`/`silu` do not even appear in this census) ≈
  **17 % by name, ≈ 0.1 % by node weight.**
- The bridge's covered subset is real but aimed at the wrong shapes:
  `dot` is a **GEMV** (`[k,n]·[k]`/`[k]·[k,n]`), `linear` maps only
  **rank-1 inputs**.  Real lifted modules are batched GEMMs.

## Ranked census — op → modules → nodes → bridge verdict → difficulty

`self` = self_blocked (origin here), `child` = blocked by descendant.
Difficulty: **E** elementwise/view (cheap kernel-ir additions),
**M** medium (layout/composition or small kernels), **H** hard
(norms/attention/GEMM — needs pattern lowering or new op kinds),
**B** blocker (data-dependent/control/opaque — cannot lower
op-by-op).

| op | mods | nodes | bridged | self | child | difficulty |
|---|---:|---:|---:|---:|---:|---|
| add | 9 | 610 | 0 | 171 (Const operand) | 439 | E — but see scalar-Const note |
| mul | 9 | 595 | 0 | 0 | 595 | E (blocked only by descendants) |
| linear | 9 | 391 | 0 | 385 (batched ranks) | 6 | **H** — GEMM, not GEMV |
| reshape | 9 | 322 | 0 | 322 | 0 | E — pure view |
| transpose | 9 | 256 | **5** | 0 | 251 | E — already bridged |
| dropout | 9 | 201 | 0 | 201 | 0 | E — identity in eval; strip |
| contiguous | 8 | 126 | 0 | 126 | 0 | E — no-op view |
| matmul | 6 | 100 | 0 | 100 (non-GEMV/batched) | 0 | **H** — attention/QKᵀ GEMMs |
| to | 6 | 54 | 0 | 23 | 31 | E — dtype cast (needs dtype in IR) |
| softmax | 6 | 50 | 0 | 50 | 0 | **H** — attention norm |
| unsqueeze | 6 | 37 | 0 | 19 | 18 | E — view |
| detach_ | 5 | 64 | 0 | 64 | 0 | E — no-op |
| concat | 5 | 61 | 0 | 61 | 0 | M — layout join |
| rsqrt | 5 | 48 | 0 | 48 | 0 | E — unary (norm component) |
| opaque | 5 | 44 | 0 | 37 | 7 | **B** — un-lifted aten (e.g. `aten::istft`) |
| arange | 5 | 10 | 0 | 10 | 0 | E — index materialization |
| embedding | 5 | 9 | 0 | 9 | 0 | M — static gather mostly |
| zeros | 5 | 6 | 0 | 6 | 0 | E — constant materialization |
| conv1d | 4 | 158 | 0 | 155 | 3 | **H** — conv (im2col+GEMM or direct) |
| pow | 4 | 130 | 0 | 130 | 0 | E — elementwise |
| layer_norm | 4 | 58 | 0 | 58 | 0 | **H** — norm (mean/var/rsqrt) |
| gelu | 4 | 24 | 0 | 0 | 24 | E — bridged name-wise |
| div | 4 | 12 | 0 | 12 | 0 | E — elementwise |
| bitwise_and | 4 | 9 | 0 | 5 | 4 | E — mask ops |
| expand | 4 | 7 | 0 | 7 | 0 | E — view |
| where | 4 | 7 | 0 | 7 | 0 | M — select (mask materialization) |
| index | 4 | 4 | 0 | 4 | 0 | M — dynamic-ish indexing |
| chunk | 3 | 256 | 0 | 256 | 0 | M — kokoro STFT splits; view-ish |
| weight_norm | 3 | 156 | 0 | 156 | 0 | M — param transform; fold at export? |
| instance_norm | 3 | 128 | 0 | 128 | 0 | **H** — norm variant |
| leaky_relu | 3 | 38 | 0 | 38 | 0 | E — unary |
| slice | 3 | 6 | 0 | 6 | 0 | E — view |
| squeeze | 3 | 4 | 0 | 4 | 0 | E — view |
| sub | 3 | 3 | 0 | 0 | 3 | E — bridged name-wise |
| lt / gt / ge / le / eq / minimum | ~3 ea | ~15 | 0 | all | — | E — predicates/fused-minmax |
| new_ones / full_like / full / zeros_like | ~3 ea | ~9 | 0 | all | — | E — constants |
| permute | 3 | 3 | 0 | 0 | 3 | E — bridged name-wise |
| sin / cos / atan2 / recip / log / sqrt / abs / rsub / remainder | kokoro+t5 | ~210 | 0 | all | — | E — elementwise unary/binary |
| relu / exp / tanh / neg | 1–2 | ~18 | 0 | 0 | all | E — bridged name-wise |
| mean | 2 | 32 | 0 | 32 | 0 | **H** — reduction (norm/attention parts) |
| pad / select_scatter / masked_fill / repeat / copy | 1–2 | ~12 | 0 | all | — | M–B — padding & scatter |
| cumsum / rand / select / getitem / lstm.input / pack / gather | 1–2 | ~13 | 0 | all | — | **B** — sequential/scatter/control |

## Must-have vocabulary, ranked by frequency × impact

Weighting: node count across 9 modules, boosted by module
breadth (an op in all 9 modules shapes every lowered program).

**Tier 0 — free wins (no new kernel-ir op needed):**
`dropout`+`detach_` (265 nodes, 5–9 mods) are eval-mode identities —
strip them at export or in a lowering pre-pass; `contiguous` (126) is
mostly a no-op view.  ~9 % of the census disappears for free.

**Tier 1 — view/layout ops (~500 nodes, all 9 modules touched):**
`reshape` (322), `unsqueeze`/`squeeze`/`expand` (48), `to`/dtype
(54), `concat` (61), `slice`/`pad`/`select`/`repeat` (~12),
`chunk` (256 — concentrated in kokoro's STFT but a view family).
These are stride/bookkeeping ops: kernel-ir needs either real view
ops with shape tracking or a layout-normalization pass that erases
them before search.  Highest count-per-effort on the list.

**Tier 2 — the missing elementwise tail (~400 nodes):**
`pow` (130), `reciprocal` (96), `rsqrt` (48), `sin`/`cos`/`atan2`
(~104, kokoro DSP), `div`/`minimum`/`maximum`/`abs`/`log`/`sqrt`/
`rsub`/`remainder` (~30), predicate+`where` mask materialization
(~20), constant materializers `arange`/`full`/`zeros`/`ones`/`full_like`
(~30).  All are broadcast elementwise — same grade class as the
existing 9 pointwise ops.  Cheap additions to the kernel-ir op table.

**Tier 3 — scalar-Const relaxation (unblocks ~200+ nodes indirectly):**
`add` self-blocks 171× on a *direct Const operand* (`x + eps`,
`x * scale`, mask biases), and Const leaves poison every mapped op
that touches them (`Term.cost` needs `numel`).  Minting scalar Consts
as broadcast tiles — or letting pointwise ops take fp args — would
unblock a large fraction of the Tier-2 surface *and* many `add`/`mul`
sites currently counted as blocked.

**Tier 4 — the compute core (the real wall):**
`linear` (391) + `matmul` (100): every instance is batched
(`[B,T,in]·[out,in]`, attention `QKᵀ`/`AV`, ranks `(2,2)`/`(3,2)`/
`(None,…)`) — kernel-ir `dot` is GEMV-only.  Either generalize `dot`
to GEMM/tiled-batched forms or lower these as *patterns*
(linear→dot+transpose+ bias already exists for rank-1; the batched
case needs a tile loop).  `conv1d` (158), `layer_norm` (58) +
`instance_norm` (128) + `weight_norm` (156 — likely foldable at
export), `softmax` (50), `mean` (32), `rsqrt` (norm component).
Norms/attention should be **pattern-lowered as blocks**, not added as
primitive ops — see below.

**Tier 5 — honest blockers:**
`opaque` (44 — un-lifted `aten::` ops incl. `aten::istft` in kokoro's
decoder), `pack`/`lstm.input`/`cumsum`/`rand`/`select_scatter`/
`masked_fill`/`getitem`/`gather`/`index` (~30) — sequential, scatter,
or data-dependent; `kokoro:textenc`/`full` don't even export.  These
need either upstream IR support or admission that the arm is
out of scope.

## Not bridgeable / needs pattern lowering

- **Attention as a block.** The census's `softmax`+`matmul`+`add`
  triples *are* attention (QKᵀ·scale → mask-add → softmax → AV).
  Op-by-op lowering would need GEMM + row-softmax + masked-select +
  GEMM — four hard pieces.  A single `attention`/`sdpa`-class
  kernel-ir op (or a registered pattern rewrite to a fused kernel)
  is the honest path; the bridge probe
  (`docs/reports/kernelopt_bridge_probe.md`) already showed the
  kernelopt bank *wants* decomposed spellings it cannot see in
  exported terms.
- **GEMM.** `dot` denotes `einsum("kn,k->n")`.  Every real `linear`/
  `matmul` is batched.  This is the single highest-impact gap:
  491 nodes (11.5 %) sit behind it, and they sit *under* most of the
  pointwise/norm traffic too.
- **Norms.** `layer_norm`/`instance_norm`/`rsqrt`/`mean` are one
  mathematical pattern (center→scale→normalize); a `norm` kernel-ir
  op or a certified decomposition law beats four primitives.
- **Data-dependent shapes/structure.** `pack_padded_sequence`
  (`pack`), `lstm.input`, `cumsum`, `rand`, `select_scatter`,
  `masked_fill`, dynamic `index`/`gather`, and the two kokoro arms
  that fail `torch.export` outright.  Static-shape lowering cannot
  express these; they are scope boundaries, not vocabulary gaps.
- **STFT/opaque leaves.** kokoro's decoder contains `aten::istft`
  as `opaque` — the IR itself doesn't cover it, so no kernel-ir
  vocabulary can.

## Bottom line for plan 0039

Today the bridge covers **0.12 % of real-module op-nodes** (5/4,289,
all stray transposes) and **0/9 modules** end-to-end.  The gap is not
primarily vocabulary breadth — it is (1) **shape class**: the covered
ops are GEMV/rank-1 while real modules are batched GEMM, and (2)
**missing mundane ops**: views, casts, the elementwise tail, and
scalar-Const handling account for ~1,500 nodes (35 %) that are cheap
to add.  Recommended order: strip eval no-ops → view/layout family →
elementwise tail + scalar Consts → batched-GEMM `dot` (or pattern
lowering) → norm/softmax/conv1d as pattern blocks → declare the
data-dependent/opaque tail out of scope.

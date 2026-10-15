<!-- Hand-maintained curated results.  Appended verbatim to
     `docs/results.md` by `bench.benchkit.render.results`
     (`python -m bench results`) — edit it HERE, never in the
     generated file, or the doc-sync test will fail. -->

## Machine-derived structure — results by kind (2026-10-15)

*What the derivation machinery achieved, separated by **what was
measured on what**.*  Scope legend:

- **[model]** — a real `nn.Module`, delivered + fp64-verified
- **[form]** — a synthetic comparison of two computational forms
  (random tensors; measures the asymptote, not a model)
- **[symbolic]** — a cost-model count over the extracted program
  (`flops_cost` / `param_bytes_cost`), not wall time

### 1. Optimized models, end-to-end [model]

| model | T | vs eager | vs `torch.compile` | max_abs |
|---|---|---|---|---|
| `ChunkedRetention` (RetNet-style), T=512 — fused + minted kernel | 512 | — | **1.82–2.33×** | 1.3e-8 |
| `ChunkedRetention`, T=512 — **the derived carrier itself** | 512 | 0.51× (loss) | 0.48× | 1.6e-8 |
| `ChunkedRetention` | 2048 | **5.0–5.9×** | **6.2–7.4×** | 9.8e-9 |
| `chainfuse` (x@A@B@C + pointwise tail) | 256×512 | 3.71× | 2.90× | 1.7e-5 |
| `swiglu_mlp` (llama-style) | — | 1.17× | 2.00–2.04× | 6e-8 |
| `chain_mlp` | — | 1.06× | 2.52× | 4e-6 |
| `mulrelu` | — | 1.22× | 1.39× | **0.0** |
| `mulsoftplus` | — | 1.31× | 1.46× | 1e-6 |
| `mlp` (base) | — | 1.13× | 2.82× | 0.0 |

`gen` probe re-run 2026-10-15: **10/10 delivered and verified**;
9/10 faster than eager; all faster than `torch.compile`.

**The crossover is the honest shape of the carrier claim:** the
derived carrier *loses* at T=512 and wins asymptotically.

### 2. Peak activation memory [model] vs [form]

**Real module** — `_ChunkedRetention(16)` forward vs its extracted
carrier lowered through the ambient sink, peak
`torch.cuda.max_memory_allocated`, fp64, B=2:

| T | model | carrier | saved | max_abs |
|---|---|---|---|---|
| 512 | 18.52 MiB | 10.91 MiB | 1.70× | 1.6e-08 |
| 1024 | 48.90 MiB | 14.39 MiB | 3.40× | 1.7e-16 |
| 2048 | 169.66 MiB | 23.26 MiB | **7.30×** | 9.8e-09 |

**Synthetic proxy** (same shapes, random tensors — extrapolates the
asymptote past the model's feasible range):

| T | `(q@kᵀ)@v` | C=16 chunked | saved |
|---|---|---|---|
| 512 | 12.6 MiB | 8.8 MiB | 1.4× |
| 1024 | 25.1 MiB | 9.4 MiB | 2.7× |
| 2048 | 74.1 MiB | 10.6 MiB | 7.0× |
| 4096 | **268.1 MiB** | **13.1 MiB** | **20.4×** |

The carrier's peak is near-flat (8.8 → 13.1 MiB across an 8× T range)
while the fused form grows 21×.  O(T·C) against O(T²), as memory.

### 3. Structural savings [symbolic]

**Compute — FLOPs** (instrument `flops_cost`; ratio = before ÷ after):

| optimization | before (FLOPs) | after (FLOPs) | **compute saved** |
|---|---|---|---|
| weight fold, k=8 (d=512) | 8,388,608 | 1,048,576 | **8.0×** |
| weight fold, k=16 (d=512) | 16,777,216 | 1,048,576 | **16.0×** |
| K-way chunk vs bisection-chunked (T=512) | 29,386,752 | 6,700,800 | **4.4×** |
| K-way chunk vs fused quadratic (T=512) | 39,059,456 | 6,700,800 | **5.83×** |
| shared-state reassociation (T=512, K=8) | 8,572,928 | 7,643,912 | **10.8%** (24.5% at T=128) |

**Storage — parameter bytes** (instrument `param_bytes_cost`; d=512,
fp32, `x@W₀@…@W_{k-1}` vs the folded `x@W_cat`):

| k | before (stored) | after (stored) | **storage saved** |
|---|---|---|---|
| 2 | 524,288 B | 262,144 B | **262,144 B** (2.0×) |
| 4 | 1,048,576 B | 262,144 B | **786,432 B** (4.0×) |
| 8 | 2,097,152 B | 262,144 B | **1,835,008 B** (8.0×) |
| 16 | 4,194,304 B | 262,144 B | **3,932,160 B** (16.0×) |

Plus the shared-input MoE fold: **196,608 B** of stored parameters
eliminated (pinned by test).

### 4. Oracle-evaluation savings [symbolic]

| quantity | before | after | **saved** |
|---|---|---|---|
| programs encoded per e-node (k=11) | — | 16,796 programs in 231 e-nodes | **72.7 programs/e-node** |
| ranking quality (AP, held-out pages) | heuristic 0.045 | trained prior **0.133** | **3.0×** |
| useful laws found per episode | random 0.083 | trained prior **0.167** | **2.0×** |
| wall time, second run (evidence store) | 128 s | 6.6 s | **19.4×** |

### 5. Honest losses

| finding | measured |
|---|---|
| the carrier below the crossover | **0.51×** vs eager at T=512 |
| the *folded* carrier vs the spelled form | a fold is a **cost-neutral relabelling** — the gap is search, not pricing |
| pure pointwise fusion | **0.82× loss** — Inductor's home turf |
| shared-state on the real export | ~~unreached~~ — **reached**: the staged extract carries the chained shared recurrence (7 fold reads + 7 single-computed state blocks at K=8; 6,943,236 vs the like-for-like floor 7,082,496 — ~2% under). The earlier "0/13 fold" probe counted `applyd`-spelled members, which extraction never prefers; the like-for-like reference also omitted projections + in-graph score construction. fp64-verified via `TorchSink` (6.7e-16) |
| the composed atom-synthesis loop's first atom | **0 certified laws fire on real terms** — `scan_fold` computes the fold's *total*, the corpus *emits* the sequence; the paying `fold_emit` sibling is enumeration-ordered >200k deep (it verifies at 6.7e-16 when reached directly) |
| anti-unification compression (ADR 0007 layer 1) | **NULL on reach-expansion** — identical cheapest extractable member with/without definitional atoms on all three workloads; compression real (12 vs 26 ops) |
| discovery yield in the exposed corpus | **~1.5%** — a 60-model sweep found no paying new law beyond `mul_unsq_nat`; exposure itself is bimodal (81/321 workloads expose 0%) |
| the precision (bounded) lane on real BiRefNet (220M) | **certified negative** — 20 offers at 1e-2, 56 at 3e-2, all correctly refused (bounded member prices 1.32× worse than dense); fp32 gate bitwise-clean |
| the ONNX backend's memory/speed | **not measured as a win** — the value is portability + verified round-trip (1.11e-16 fp64), not throughput |

### 6. The discovery loop — minted and shipped (2026-10-14) [model] + [tooling]

| result | measured |
|---|---|
| **first corpus-minted law shipped in `DEFAULT`** | `mul_unsq_nat` — minted from DeltaRuleBlock's own `β·k` op-tuple; naive guard **falsified by the adversarial sweep** (212 bad accepts), exact `flat-map-unsq` guard synthesized (0 bad / 0 miss); closure **+18 enodes**; pays **−12.27%**; provenance `MACHINE_SHIPPED` |
| its apparent sibling | **correctly rejected** at admission — `mul_unsqueeze_r_id` (−14.3%, 12 models / 22 sites) has semantically unequal sites; the guard declines all of them |
| handler synthesis (ADR 0007 layer 2) | machine-minted verified handler for the `om_decayed` element at **8.9e-16** — previously a hand-written executor; every named non-elementwise shape verifies in bounded search |
| law synthesis per atom (layer 3) | **25 certified / 4 rejected / 0 refused** rows around shipped atoms; new primitives (`affd.compose` slot-solved, `dv.merge`, `omd.absorb_score`); found a real executor bug (`decay_vec` `sign=0`) by refusing to certify |
| fused carrier lowering vs the paper's own chunked impl | **speed parity+** — 2.08 vs 2.12 ms (T=2048), **faster under graph replay** (0.670 vs 0.822 ms at T=1024); fp64 1.4e-16; FLA still ahead (kernel-level) |
| ONNX backend, both directions | **zero edits under `catopt-core`**; fp64 round-trip **1.11e-16** via `onnxruntime`; 107 in-op types (opsets 13–21), 169 out-ops; package coverage 100% |
| combinator bridge (ADR 0008 prototype) | **321/321** corpus workloads round-trip structurally exact, zero ops refuse translation, mean blowup 2.61×; the RetNet carrier spells as a `loop` term verified at 1e-7–2.6e-7 |

## Learned models (2026-10-15)

*What the project trains, and how each performs.*  Separate from the
architectures above: these **steer** the search; the models in §1 are
what gets optimized.

### Neural — contraction ordering

| model | training | metric | value (vs teacher = 1.0) |
|---|---|---|---|
| `PairPolicyNet` | **RL** (curriculum REINFORCE), 25 KB | equal-rollouts cost ratio | n=20 **0.96** / n=30 **0.94** ✓; n=40 1.26 ✗ |
| `PairPolicyNet` **distilled** (`oe-all`, *shipped default*) | supervised (teacher's trial distribution) | equal-rollouts cost ratio | n=40 **0.91** ✓ — first arm to beat the teacher |
| `oe-best` (distilled, selected order) | supervised | equal-rollouts cost ratio | n=40 1.17 ✗ |
| `dp` (imitation of the RL curriculum) | supervised | equal-rollouts cost ratio | n=40 **127×** — the curriculum is a bad teacher |

Player is `best_order` (best-of-N sampled rollouts); the single argmax
is brittle at n=40 for every arm (RL 4.15×).  Not wired into
`Optimizer.optimize` — a standalone research artifact.

### Neural — rule value (supervised)

| model | metric | value | baseline | verdict |
|---|---|---|---|---|
| `RuleValueNet` (MLP 32) | ranking AP (`rank_pool`) | 0.023 | heuristic 0.045 | **loses** |

### Linear — the discovery game (RL, baked presets)

| model | metric | random | cold `{}` | trained | baseline |
|---|---|---|---|---|---|
| **open-game prior** (`OPEN_GAME_PLAYER_WEIGHTS`) | reward / episode | 0.134 | 0.143 | **0.304** | — |
| | useful laws / episode | 0.083 | 0.083 | **0.167** | — |
| | ranking AP | 0.012 | 0.012 | **0.133** | heuristic 0.045 |
| **meta-arena prior** (`META_ARENA_PLAYER_WEIGHTS`) | reward / episode | −0.57 | −0.49 | **−0.06** | scripted −0.10 |

Regenerate with `tools/train_open_prior.py` / `tools/train_meta_prior.py`.

**Honest note:** in *both* families the best result is not the neural
one — distillation beats neural RL at n=40, and the linear prior beats
the neural rule-value net.

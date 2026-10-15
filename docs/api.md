# API surface

The orchestration surface lives in `catopt_orchestrator`
(backend-neutral — it imports no torch); the torch ports, runners and
autotune candidate builders live in `catopt_torch` / `catopt_cuda`;
the engine itself is `catopt_core`.  There is no `catopt` façade — import
the domain packages directly.

## The entry point

```python
from catopt_orchestrator import Optimizer
from catopt_torch import TorchBackend

opt, stats = Optimizer(backend=TorchBackend()).optimize(model, x)
```

`Optimizer.optimize(model, x, **kw) -> (module, stats)`.  `stats`
carries `rule_fires`, `lowering`, `runner`, and the strategy's own keys.

## Choose how the result is delivered

```python
from catopt_cuda import CudaGraphRunner
from catopt_orchestrator import (
    Autotuned, ChainedRunner, Compositional, Optimizer,
)
from catopt_torch import TorchBackend, TorchCompileRunner
from catopt_torch.autotune import TORCH_BUILDERS

opt = Optimizer(backend=TorchBackend())

opt_mod, stats = opt.optimize(model, x, runner=TorchCompileRunner())
# torch.compile wraps the delivered module (stats["compiled"])

opt_mod, stats = opt.optimize(model, x, runner=CudaGraphRunner())
# captures the executor into a CUDA graph (stats["cuda_graph"]) —
# compile-free; works without Inductor

opt_mod, stats = opt.optimize(
    model, x,
    runner=ChainedRunner([TorchCompileRunner(), CudaGraphRunner()]),
)
# runners compose left-to-right; the Runner protocol is duck-typed,
# so your own runner drops in

opt_mod, stats = opt.optimize(
    model, x, strategy=Autotuned(builders=TORCH_BUILDERS),
)
# re-lowers the same extracted term through each candidate executor
# ("generic", "batched", "torch_compile", "cuda_graph"), verifies
# each, times them on the real input, returns the measured winner

opt_mod, stats = opt.optimize(model, x, strategy=Compositional())
# multi-block models: optimizes each block against its captured real
# input, recomposes with per-block + end-to-end verification and
# automatic fallback (stats["blocks"], stats["end_to_end"])
```

## Steer what "cheapest" means

```python
from catopt_orchestrator import LatencyCriterion, MemoryCriterion

opt_mod, stats = Optimizer(backend=TorchBackend()).optimize(
    model, x,
    criteria=LatencyCriterion() * 0.7 + MemoryCriterion("peak") * 0.3,
)
# or the shorthand: criteria={"latency": 1.0, "memory": 0.5}
```

## Reference

| Name | Signature / role |
|---|---|
| `Optimizer` | `(backend=..., source=..., sink=..., composer=..., meter=..., criteria=..., runner=...)` — the configured entry point; `.optimize(model, x, **kw)` → `(module, stats)`, plus `.search` / `.lower` / `.discover` phase verbs |
| `Monolithic` / `Compositional` / `Autotuned` / `MorphismSearch` | the `strategy=` argument of `Optimizer.optimize`: whole-model search (default), per-block + recompose (`block_pred=`, `verify_tol=`, `cache=` replays structurally-identical blocks — N identical blocks cost ~1 search + N−1 verified replays, `max_cross_pairs=` re-judges adjacent pairs jointly), measured autotune (`candidates=`, `budget_s=`, `profile=` persists measured corrections, `builders=TORCH_BUILDERS`), or block-signature algebra (`optimize_morphisms` — lifts each block to a `BlockSig`, rewrites the tiny morphism graph with `WindowCompose`/`ResidualReassoc`/`NormCascade`/`WeightTie`/`KVLatentShare`/opt-in `CrossBlockCSE`, reifies into certified term rewrites) |
| `search` / `lower` | the phase verbs: `model -> SearchResult`, `SearchResult -> LowerResult` — re-lower one search under different runners |
| `discover_alternatives` | `(model, x, *, source, ...)` → `SearchResult` — enumerate the equivalence frontier (`.alternatives(top_k)`, `.certificate()`) |
| `SearchResult.frontier` | `(cost_fns: {axis: CostFn}, candidates=32, top_k=8)` → `[(CostVector, term)]` — the **non-dominated** members over named axes (`catopt_core.pareto`); feed them to `pareto.best` for the scalar view |
| `export_optimized` / `load_optimized` | `catopt_torch.export` — `(model, opt, path, fmt="module"|"safetensors"|"state_dict"|"torchscript", …)`; `.pt2` roundtrips run standalone, no catopt at inference |
| Rule sets | `search(..., rules=DEFAULT)` — composable `RuleSet` algebra (`FULL - SYMMETRY`, `WITH_LAYOUT`, presets in `catopt_core.laws.ruleset`). `DEFAULT` ships **11 machine-admitted laws** (`catopt_core.laws.provenance.MACHINE_SHIPPED` — promoted verbatim after clearing the admission gauntlet; `project/retros/promoted-laws.md`) alongside the hand-written kernel; the newest is `mul_unsq_nat`, the first minted from a *corpus op-tuple* (`mul(unsqueeze U d, unsqueeze V d) = unsqueeze(mul U V, d)`, guard `flat-map-unsq`, −12.27% on its source workload). Four further laws were machine-*assisted* but hand-integrated before the admission machinery existed and classify as human-authored (`select_mul`, `softmax_fold`, `glu_fold`, `rms_norm_fold`) |
| Law structure | `catopt_core.egraph.Rewrite` — `name` / `lhs` / `rhs` pattern trees, `law=` prose, `check` / `derive` side-condition hooks, `tags`. `derivation: tuple[str, ...]` records the shipped premise(s) proving the law; `rule.kind` reads `"axiom"` (empty derivation — kernel member), `"lemma"` (recorded derivation) or `"redundant"` (derivation + the `tags.REDUNDANT` mark, an alpha-duplicate spelling). Measured on `ALL_RULES` (82, re-run 2026-10-15): **58 axioms / 22 lemmas / 2 redundant / 19 no-instance** — generated by `catopt_discovery.coherence --emit-basis`, not hand-picked (`project/retros/axiom-lemma-split.md`) |
| Cell store | `catopt_core.cells` — `Cell{role, name, body, kind, provenance}` where role ∈ `op`/`law`/`object`/`handle`, the one record every vocabulary item becomes (the Sanada operations-and-handlers mapping made literal). `merge_fragments(*Fragment)` assembles a `CellStore` — duplicate `(role,name)` raises by default, `keep_first`/`keep_last` policies record conflicts, alpha-equal 2-cells report in `MergeReport` (subsumption is a discovery fact, not an error). Views: `laws()`/`ruleset()` → `EGraph.run`, `objects()`, `opdefs()`, `handlers()` (tag-keyed), `to_data`/`from_data` — one schema over the law/object/opdef/spec codecs. `Provenance.origin` ∈ `shipped`/`constructed`/`admitted`/`minted`/`declared`/`imported`. `catopt_discovery.cellstore` adapts: `ambient_store` (shipped rules + handlers), `handlers_fragment`, `machine_fragment` (evidence rows), `cell_of_constructed`; `MetaArena(store=)` + `board.cellstore()` make episodes read/write stores directly |
| Handl bridge | `catopt_discovery.handl` — the Phase-3a schema bridge: `store_to_handl(store)` emits a handl fragment shape — op cells → `AstShapeDef` rows, law/object cells → `TransformDef` rows (lhs root → `node_kind`, inner op children → `when:!KindAt`, metavars → `!KeepPath` child positions, rhs → `!Declarative` `!Create`/`!KeepPath`), handle cells → `!Native` records (`catopt:<kernel>`). Every row carries a `catopt:` extension with the cell's serialized body — lossless by construction; `cell_from_handl` round-trips (provenance `imported`) |
| Engines | `search(..., engine=NativeEngine())` — the pure-Python engine is the default/reference; `catopt-native` (PyO3/Rust) is an explicit opt-in (~17× on match-bound closures) |
| Detection passes | `search(..., detect_factors=True)` — certified low-rank weight factoring; `detect_specials=True` — exact dead/diag/dup/block-diag weight elision; `detect_headshare=True` — shared-head weight elision; `error_budget=` — certified bounded approximations (bound ledger + output-propagated verify) |
| Criteria | `LatencyCriterion`, `FlopsCriterion`, `DepthCriterion`, `MemoryCriterion("weights"|"peak"|"combined")`, `CompiledCriterion` — compose with `*` / `+`, or pass `{"axis": weight}` dicts |
| `PredictedCriterion` | `(model: PerformanceModel, profiler=None, hardware=None)` — a `Criterion` that prices a term by the model's **prediction** (`catopt_core.perf_model.AnalyticalPerformanceModel` is the shipped model). Non-additive inside extraction; opt in when you want a model to steer selection |
| `policy=` | on `search` / `Optimizer.optimize`: a `Policy` (`catopt_core.policies`: `RandomPolicy` / `ExistingPolicy` / `GreedyPolicy` / `BeamPolicy`; `catopt_torch.learned_policy.LearnedPolicy`; `catopt_torch.rl.RLPolicy`). Consulted once per saturation iteration by `EGraph.run(..., policy=)`; it may only **reorder** — every rule still runs, so the fixed point and the certificate are unchanged. Recorded in `stats["policy"]`; a policy-less engine declines with a clear error |
| Profiling | `catopt_core.features.StaticProfiler().profile(term)` → `ProgramFeatures` (flops, bytes, depth, reuse, …) — torch-free, computed without running the program |
| Measurement | `catopt_core.timing` — the one contract (`TorchMeter`, `calibrate`, `benchkit.Runner` all reduce through it); `catopt_core.failures.FailureClass` / `classify` — OOM / timeout / kernel / NaN / device-unavailable, and `TimingResult` carries `device` / `warmup` / `failure` provenance |
| Runners | `IdentityRunner` (default), `TorchCompileRunner()`, `CudaGraphRunner()`, `ChainedRunner([...])` — duck-typed `Runner` protocol |
| Ports | `Source` / `Sink` (`catopt_core.ports`; torch impls `catopt_torch.adapters.TorchSource`/`TorchSink`, bundled as `TorchBackend`) — a new backend implements `Sink`; the engine never imports it. The evaluation dimension adds `Profiler` / `Policy` / `PerformanceModel` (`catopt_core.ports`), each with a shipped implementation and a consumer in the pipeline (ADR 0003) |
| Verification | `catopt_core.egraph.verify_certificate` — replays the derivation shipped with every extracted program |
| Contraction player | `catopt_torch.load_contraction_policy(path=None, *, device="cpu")` → `ContractionPolicy` — the trained contraction-ordering player, a bundled ~25 KiB distilled artifact (`catopt_torch/artifacts/contraction_policy_distilled.pt`, the `oe-all` arm of `lab/probes/contraction/contraction_distill.py`) loaded lazily with `weights_only=True` (`ValueError` on format/feature drift; `path=` loads a checkpoint from `save_contraction_policy`, e.g. the bundled RL-lineage `contraction_policy_curriculum.pt`). Methods: `.order(tensors, sizes)` — one deterministic pass (a fallback, brittle at n=40); `.cost(tensors, sizes)`; `.sample_orders(...)` / `.best_order(tensors, sizes, samples=64, seed=0)` — seeded sampled rollouts, the real player. A research artifact behind a documented API — not wired into `Optimizer.optimize` (measured: **beats `opt_einsum`'s randomised greedy at equal wall-clock at n=40, 0.91–0.98× across three seeds at fed budgets**, and 0.51–0.90× the deterministic greedy; `project/retros/contraction-synthesis.md`) |
| Measured pricing | `delivered_cost_for(profile=None, *, x=None, bucket=None, compiled=False)` → `CostFn` (`catopt_orchestrator`) — extraction bills each term under the lowering it would actually be delivered by (batched carrier vs generic), corrected by the profile's measured `(route, shape_bucket)` factors; `x=` derives the bucket (explicit `bucket=` wins; `bucket=None` is the delivered-aware but uncalibrated model). Siblings in `catopt_core.cost`: `executor_cost_for(profile)` (measured constants only), `fused_cost_for`, `lowering_aware_cost_for`, `backend_cost`. The `profile` marker rides `backend_cost` into `_select_best_term`, so selection and the `_carrier_upgrade` comparison consume one calibration |
| Profiles | `catopt_core.profile.TargetProfile` — the *measured* target. `scripts/calibrate_profile.py` writes the JSON (measured constants + executor-family corrections); `TargetProfile.load(name_or_path)` reads it back, `profile.save()` / `--save` persist under `profiles_dir()` (`$CATOPT_PROFILE_DIR`, else `$XDG_CACHE_HOME/catopt/profiles`, else `~/.cache/catopt/profiles`) |
| Law discovery | `catopt_discovery.pipeline` — the closed loop: census → propose → verify → measure → rank → emit (`--vocab hand|derived`, `--holdout <rules>`, `--emit-admission <candidate> --out <dir>` — emits the `R(...)` + tests as a reviewable `git apply` patch; `--evidence-db PATH [--use-evidence-cache]` — sqlite verdict cache keyed by corpus × rules × code revision). `catopt_discovery.coherence --emit-basis` — the measured axiom/lemma table; `catopt_discovery.evidence --report PATH [--ships|--flips|--history <substr>]` — the verdict store's query CLI |
| Semantic-atom minting (ADR 0007) | `catopt_discovery.antiunify` — layer 1: `mine_candidates` (occurrence-matrix anti-unification over the corpus), `mint_atoms` (registers `defn_N` + `DEFINITIONAL` intro/expand rewrites + `ConstructedObject` provenance), `fold_ruleset`, `unfold_term`/`unfolding_cost`; measured NULL on reach-expansion (compression only — see the ADR). `catopt_discovery.handler_synth` — layer 2: structured handler grammar (`ew`/`contract`/`fold`/`fold_emit`/`fold_zip` descriptors, JSON-able data, no eval), `propose_bodies` (simplest-first, round-robin), `verify_body` (fp64 gate, louder refusal on raise), `mint_handlers` (genkernel-shaped handler rows + provenance), `handler_record`/`handler_from_record` round-trip; measured: verified handlers minted for the `om_decayed` element (8.9e-16) and every named non-elementwise shape |
| Combinator bridge (ADR 0008) | `lab/bridge/combinator.py` — `BASIS` (as data): `leaf`/`arr`/`seq`/`par`/`id`/`dup` (Δ)/`perm`/`drop`/`loop` (bounded trace); `dag_to_comb` (fan-out → explicit `dup`, `pack` root → output bundle) and `comb_to_dag` (hash-consed expansion). Measured: 321/321 corpus workloads round-trip structurally exact, zero ops refuse translation, mean blowup 2.61×; the RetNet shared-state fold spells as `seq(pre, loop(step, K), post)` and verifies at 1e-7–2.6e-7; the staged extract projects round-trip-equal (92 Δ sites). `lab/bridge/combinator_probe.py` drives the gates |
| Discovery drivers (tools) | `lab/screen/exposure_screen.py` — law-LHS exposure census over the corpus (bimodal: 81/321 expose 0%, 140 ≥95%); `lab/screen/exposed_discovery.py` — per-model discovery driver (census/mixed/grammar arms, `source:name` selection, `term_to_data` evidence rows); `lab/screen/discovery_scale.py` — budgeted corpus sweep + corpus-wide novelty pass (60-model run: 0 new paying laws beyond `mul_unsq_nat`; ~1.5% yield); `lab/mint/atom_law_loop.py` — layer-3 law synthesis per atom (measured envs → oracle → `auto_cond`/dspec → `verify_law`; 25 certified rows on the shipped carrier atoms); `lab/mint/atom_synth_loop.py` — the composed ADR-0007 loop (need census → signature → handler → laws → admission → pay probe) |
| ONNX backend (ADR 0006) | `catopt_onnx` — `OnnxSink` (IR → checked ONNX artifact; 169 ops, opset 18, refuse-never-drop) and `OnnxSource` (ONNX → IR; 107 op types, opsets 13–21, shape inference once, symbolic dims as `None`, attribute-operand folding). Both verified through `onnxruntime`; **zero changes under `catopt-core`** for either direction (the hexagonal boundary held twice) |
| Game layer | `python -m catopt_discovery.play --domain meta|joint|search|torch|gen` — boards as RL environments (`--train-cases`/`--eval-cases`/`--budget`/`--deliver`/`--probe`/`--zoo`/`--zoo-d`/`--zoo-t`/`--zoo-b`/`--scan-probe`/`--decay-probe`/`--json`). `catopt_discovery.engine` — the `Board` protocol + `run_episode`/`evaluate`/`train_policy` drivers; `catopt_discovery.players` — `LinearPolicy`, the learned arm; `catopt_discovery.meta_arena` — the mid-search construction board (`fire`/`saturate`/`declare`/`claim`/`extract`); `catopt_discovery.genkernel` — minted handlers bound to generated `triton.jit` source across three codegen tiers (pointwise fusion, diagonal-scan `applyd`, fused `om_decayed` contraction — `triton_sink`/`hybrid_sink`), with `profitable` as the measured referee and `kernel_for` timing each tier on its real delivery path |

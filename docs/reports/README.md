# Research reports — index

Dated measurement and design reports from the discovery program.
Each `.md` has a JSON sidecar where the run emitted one; the
`results_overnight*` files are the resumable ledger exports of
`bench/drivers/overnight.py`. Reports are evidence, not docs — read
the newest in each series for the current verdict.

## Self-driving program (dated reports)

| report | what it established |
|---|---|
| `2026-10-27-overview-m1.md` + `architecture-m1.md` + `build-list-m1.md` + `session-plan.md` + `TODO-m1.md` | the kernel-work M1 milestone plan: repos, drivers, own harnesses (pre-split naming) |
| `2026-10-27-substrate-integration.md`, `plan-roofline-to-search.md`, `automatic-tuning-margin.md`, `proposal-int4-kernel-vocabulary.md`, `proposal-kernel-ir-lib.md` | M1 design inputs: substrate integration, roofline→search plan, tuning margin, int4 vocabulary + kernel-IR proposals |
| `2026-11-12-meta-arena-corpus-coverage-gen.md` | Stage-1 lever 1: `corpus_cov` — coverage-driven corpus generation, mechanical extension of `gen_cases` |
| `2026-11-12-law-auto-admission.md` | Stage-1 lever 2: `admit_prim_record` / store `AUTO_TIER` — verified+paying prim folds promote into `DEFAULT` without a human; `AUTO_ADMITTED ⊂ MACHINE_SHIPPED` |
| `2026-11-12-domain-spec-domains-as-data.md` | Stage-2 first cut: `domain_spec` — a JSON spec emits a whole playable `play` domain (meta-arena board); `--domain-spec` CLI |
| `2026-11-12-domain-spec-second-cut.md` | second cut: torch board spec-able, per-spec player priors (`bake_prior` oracle-imitation) |
| `2026-11-13-domain-spec-code-boards.md` | third cut: all seven code-shaped boards (`gen`/`joint`/`search`/`fragment`/`open`/`player`/`bilinear`) spec-able with per-board whitelists; `gen-fusion`/`search-folds` demos deliver verified |

## Sequence-scan series (p34–p37, 2026-11-12/13)

| report | what it established |
|---|---|
| `p34_seq_scan_denotation.md` | revived the seq-scan claim path with a correct spelled denotation (`seq_scan_denotation`/`handler_denotation` in genkernel) |
| `p35_seq_scan_collector.md` | `_seq_scan_sites` collector: raw recurrence chains mint seq specs; first real site pays |
| `p36_seq_view_lift.md` | `unsqueeze`/`squeeze` step exprs mint honest seq scans — GatedLinearAttention + MambaSelectiveScan pay; `deliver` spec field |
| `p37_seqm_matrix_scan.md` | `seqm`: matrix-state scans with state feedback — the delta-rule family (DeltaRuleBlock, GatedDeltaRule) mints, verifies, pays |

## Scan-tier delivery and corpus rounds (2026-11)

| report | what it established |
|---|---|
| `applyd_route.md` | `chain→applyd` e-class route: seq/seqm claims bind real Triton kernels end-to-end |
| `seq_deindex_params.md` / `seq_deindex_boundaries2.md` | `_seq_deindex` const-meta + projection-lift fixes; generalized commute/projection lift across ~15 op families |
| `unrolled_route.md` / `resweep_scan_route.md` | unrolled-scan routing shipped; frontier re-swept — graph arm beats `torch.compile` 7/8 rows |
| `module_loss_gla_mamba.md` | GLA/Mamba module losses are structural (per-step outputs need a trajectory kernel), not routing |
| `t010_increment2.md` | T-010 census: 0 paying sites; stride-1 reduction path measured neutral-to-slower, reverted |
| `t010_codegen_increment.md` | T-010 increment: the clamp attr grammar reaches the Triton renderer — the 28 corpus clamp sites become generatable kernels |
| `egraph_divergence.md` | Rust e-graph `FxHashMap` hash-order vs Python id-order divergence — fixed upstream (`egraph` `c48f75e`), parity now 0 divergences |
| `rust_saturation_gap.md` | Rust-vs-Python e-graph saturation re-measured (release build): Rust ~4× faster aggregate, RealGatedLRU 12.6 s vs a 120 s debug timeout; the two stats divergences it flagged are the ones `egraph_divergence.md` resolved |
| `t029_executor_confound.md` … `t035_036_pay_declined.md` | agenda round: executor-confounded pay measurements fixed (`noprim_ir` arm); ChunkedSSD headroom = dispatch; pay declines vs verify declines split in the miner |
| `cumsum_carrier.md` | `kind="seqs"` emitted-scan tier for `cumsum`/`cumprod` — verifies bit-exact, honestly declines to pay vs `torch.cumsum` |
| `seq_dedup_fix.md` | site-record dedup fix: same-spec opposite-direction chains (`_BidirectionalScan`) now mint both sites |
| `trajectory_scan.md` / `trajectory_scan_impl.md` | trajectory tier: recurrence chains whose per-step states are read out (the GLA/Mamba shape `module_loss_gla_mamba.md` blamed on routing) mint `kind="seqs"` trajectory kernels + spelled `select(traj, t)` read-outs |
| `seqt_readout.md` | `seqt` fused read-out tier — the per-step contraction folds into the scan kernel; GLA 2.06×/2.39×, Mamba 6.29×/2.25×, HeadedSelectiveScan 5.38×/2.11× (fp64/fp32); BidirectionalScan honestly stays `seqs` |
| `seqs_pay_sweep.md` | the `seqs` cumsum carrier's measured boundary curve — no stable paying regime vs `torch.cumsum`, `T > 256` tile-capped: covered-but-not-paying |
| `deliver_runner_arm.md` | the `--deliver` hooks (`torch`/`gen`) run a `CudaGraphRunner` arm after a passed verify — CUDA-tensor inputs only, opt-in-by-availability with fallback |
| `t028_spec_attrs.md`, `t032_rescope.md`, `t034_module_costing.md`, `t032_034_frontier_delivery.md`, `t037_resweep_frontier.md` | agenda continuation: `vocab.attr_values` landed; DecoupledMLA headroom re-scoped to the attention half; GatedDeltaRule's module loss is payload compute, not routing; frontier "uncaptured" rows stale → re-measured, graph arm first-class, `BatchedScanModule` executor confound fixed via the payload-dedup plan gate |

## Fragment tower (plan 0033 / ADR-0157)

`fragment_tower_phase0.md` … `phase5.md` (+ `phase2_verdicts.md`):
`vocab/` as the canonical fragment vocabulary (phase 0), then the
tower phases through phase 5 — **native handl-side e-graph
saturation** over the fragment rewrite rows via `handl-egraph`'s
`EGraph::run` reached through `mlopt-kit`'s `egraph-run`.

## Kernel / executor campaign (p5–p33)

The quantized-GEMM and genkernel-codegen measurement series —
vLLM fusion ceilings, AWQ/Marlin baselines, split-K (retracted in
`p13_retraction_splitk.md` — every "beat awq_gemm" claim in p8–p12
withdrawn), routing/deploy regimes, the M1 verdict (`p23`), roofline
arm (`p25`, 82–86 % of roofline), then the synop-fold delivery
arc (`p26`–`p30`: fused bodies verify but lose wall-clock → strided
codegen fix → post-gate verdict) and the reduction-Triton tier
(`p31`–`p33`, re-attacked in `t010_fp32_attr_levers.md`).
`prim_kernel*.json` + `red_*probe*.json` + `attr_elementwise_probe*`
are the raw sweep sidecars.

## Overnight discovery ledgers

`results_overnight_{deep,deep2,real_preview,real_fusion,real_v2…v6,real_full,
xmodule,xmodule_v2,prim,prim_v2…v4,prim_deep,prim_depth3,prim_vocab,
prim_bootstrap,prim_auto_admit}.{md,json}` — the resumable
JSONL-backed runs of `bench/drivers/overnight.py` over the `grammar`,
`zoo`, `zoo-real` and `zoo-xmodule` corpora (plus `zoo-prim`, the
primitive-minting arm). Each row: episodes → mint → gauntlet →
novelty classify (`machine-original` / `known-art` / `vacuous`).
`results_overnight_prim_auto_admit.md` is the AUTO_TIER demonstration.

## Zoo sweeps and probes

`zoo_chain*.{md,json}` (chain-level pay across the then-68-workload
zoo — 73 after `zoo_corpus_expansion.md` — fp32/fp64 rounds), `zoo_kcount*`, `zoo_dtype_rmsnorm`,
`red_triton_probe*`/`red_tier_probe`, `scale_ladder` (attention-block
terms T=512→4096), `bank_growth_*` — raw measurement sweeps behind
the delivery verdicts.  `zoo_corpus_expansion.md` — the workload-shape
gap census and the fourth expansion (registry now 73): conv+scan
hybrid, headed scan, cumsum carrier, bidirectional scan, latent-KV
decode; measured collector boundaries (cumsum invisibility,
spec-identity dedup) and the one non-claim payer (LatentKVDecode).

## Synthesis and analysis

| report | what it established |
|---|---|
| `integration.md` | index of the four-suite measured programme (re-reads committed measurements) |
| `frontier{,_fp32}.{md,json}` | theoretical headroom vs delivered capture per rewrite — the "where can discovery pay" bound |
| `delivery.{md,json}` | minted folds measured twice: in-situ vs isolated kernel — a rule pays as a kernel, not as a step |
| `pay_measured.md` / `pay_coverage.md` | the measured pay leg + which shipped folds have a wall-clock number at all |
| `prim_delivery.md` / `prim_kernel.md` / `prim_kernel_{fp64,gla,mamba}.json` | do shipped `synop*` folds pay on real wall time — cost-model win, measured loss; a real kernel flips it (~2.6×) |
| `primitives_finds.md` / `prim_handlers.md` | what the machine minted (21 admitted folds, machine-original) + the fused-body handler hunt |
| `family_synth.md` / `fusion_schema_mint.md` | minting family *descriptors* (not just bodies) + widening the zoo-real candidate families |
| `referee{,_proof}.{md,json}` | do the tensor and kernel rule banks share one vocabulary — largely disjoint (Jaccard 0.17); crossing works via explicit denotation rows, not a shared vocabulary |
| `vocabulary.md` | originality is a vocabulary event: fixed-bank 0 % machine-original vs vocabulary arm 99 % |
| `play_mint_admit.json` | per-candidate admit ledger from a play-mint run |
| `agenda.{md,jsonl}` | Stage-3: the agenda miner — 9 signal classes → 23 ranked research proposals (`--myco` ingest) |

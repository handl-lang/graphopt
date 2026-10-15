## Tooling-level measurements (hand-maintained)

This file is appended verbatim to `docs/results.md` by
`benches results` / `render_results_doc` — edit it here, never in the
generated file.  These measurements live in `tools/` and are
recorded in the retros, not in `benches/baselines/`.

All measured on the dev box (RTX 2050 / i5-12500H, fp64 — same
hardware class as the baselines); magnitudes do not extrapolate to
datacenter hardware:

| measurement | result | tool | retro |
|---|---|---|---|
| `softmax_fold` on `ManualSoftmaxAttention` | **+13–27% eager, +29–48% CUDA-graph** vs raw | `tools/law_wallclock.py` | [law-wallclock-verification](../project/retros/law-wallclock-verification.md) |
| `select_mul` marginal (optimized vs law-ablated) | **faster in 19/20 measurements** | `tools/law_wallclock.py` | [law-wallclock-verification](../project/retros/law-wallclock-verification.md) |
| executor routing after measured pricing | **12/12 cases ship the measured-fastest member** (uncorrected picks ran 2–3× slower) | `tools/executor_cost_probe.py` | [executor-pricing-fix](../project/retros/executor-pricing-fix.md) |
| `silu_fold` on the bench case | **−53–55% modeled cost, 1.43–2.62× wall-clock** | `benches run law_bench` | [three-cell-mediator](../project/retros/three-cell-mediator.md) |
| contraction player, n=40 vs `opt_einsum` | **0.89–1.01× of randomised greedy at equal wall-clock** (0.91–0.98 at T≤1, three seeds, fed budgets); **0.51–0.90× deterministic greedy** | `tools/contraction_guided_restart.py` | [contraction-synthesis](../project/retros/contraction-synthesis.md), [contraction-player-shipped](../project/retros/contraction-player-shipped.md) |
| pipeline held-out rediscovery | **winner re-ranks #1, SHIP, every run** | `catopt_discovery.pipeline --holdout` | [law-proposer-extensions](../project/retros/law-proposer-extensions.md), [groupnorm-convnext-corpus](../project/retros/groupnorm-convnext-corpus.md) |
| coherence catalogue over `ALL_RULES` (82, re-run 2026-10-15) | **58 axioms / 22 lemmas / 2 redundant; 19 no-instance; divergence 1** (`rms_norm_fold` × `_nogain` — measured, unresolved) | `catopt_discovery.coherence --emit-basis` | [axiom-lemma-split](../project/retros/axiom-lemma-split.md) |
| evidence store, second run | **50/50 verdicts cached; 128 s → 6.6 s (~19×)** | `catopt_discovery.pipeline --evidence-db` | [evidence-store](../project/retros/evidence-store.md) |
| `chainfuse` — laws + `claim(gen_0)`, Triton-bound | synthetic composite (`x@A@B@C` + 6-op pointwise tail, 256×512): **2.94× vs `torch.compile`** (86µs vs 253µs), verified 1.7e-05; laws alone reach 181µs — the minted kernel is required for the full win | `python -m catopt_discovery.play --domain gen --probe` | [gen-kernels](../project/retros/gen-kernels.md) |
| delivered module vs `torch.compile` (`gen` probe) | spelled matmul chain **3.96×** (22µs vs 87µs; Inductor keeps all three GEMMs even under `freezing`); swiglu **2.11×**; pointwise fusion **0.82–0.88× loss** | `python -m catopt_discovery.play --domain gen --probe` | [gen-kernels](../project/retros/gen-kernels.md) |
| held-out zoo probe | **11/18 delivered verified modules beat `torch.compile`** — LoRAAdapter 2.18×, WaveNetGate 1.51×, DeepEquilibrium 1.35×; every winner ran `claims=[]` (the `gk.profitable` measured referee filtered unprofitable minted claims) | `python -m catopt_discovery.play --zoo` | [gen-kernels](../project/retros/gen-kernels.md) |
| joint fwd+bwd step vs `torch.autograd` | **+13%** (68µs vs 78µs, verified 9.5e-07) — one shared memo across forward + derived VJP; unshared costs 197µs | `catopt_discovery.training.joint_step` | [joint-training-graph](../project/retros/joint-training-graph.md) |
| machine law pack | **16/16 store objects deployable** as an opt-in `RuleSet`; machine-only arm wins 3/22 zoo models the human library misses (`linear_chain_t` owns LoRAAdapter's −20%) | `catopt_discovery.machine_pack` | [human-bar](../project/retros/human-bar.md) |
| `mul_unsq_nat` — first corpus-minted law, shipped in `DEFAULT` | **−12.27% on its source workload** (DeltaRuleBlock, 992496→870722), closure **+18 enodes**; exact `flat-map-unsq` guard after the naive broadcast guard was falsified (212 bad accepts); apparent sibling correctly rejected at admission | `tools/exposed_discovery.py` + admission gate | [mul-unsq-nat](../project/retros/mul-unsq-nat.md) |
| exposure screen (321 workloads) | **bimodal: 81 score 0% law-LHS coverage, 140 score ≥95%**; exposure predicts reachability, not payoff (100%-exposed models fire nothing) | `tools/exposure_screen.py` | [exposed-math](../project/retros/exposed-math.md) |
| corpus-wide discovery sweep (60 models) | **0 paying new laws beyond `mul_unsq_nat`**; fresh-paying yield ~1.5% of the exposed class; enum arm 0→198 fires after the attr-gap fix but 0 paying | `tools/discovery_scale.py` | [discovery-scale](../project/retros/discovery-scale.md) |
| handler synthesis (layer 2) | **machine-minted verified handler for the `om_decayed` element** (8.9e-16 fp64) — previously a hand-written executor; contraction/fold/fold_emit/fold_zip all verify in bounded search; reductions/softmax refuse (named grammar gaps) | `catopt_discovery.handler_synth` | [handler-proposals](../project/retros/handler-proposals.md) |
| law synthesis per atom (layer 3) | **25 certified rows / 4 rejected / 0 refused** on the shipped carrier atoms; new primitives incl. `affd.compose` (slot-solved), `dv.merge`, `omd.absorb_score`; false twins all caught; found a real executor bug (`decay_vec` `sign=0` coercion) | `tools/atom_law_loop.py` | [layer3-loop](../project/retros/layer3-loop.md) |
| composed atom-synthesis loop | one atom minted through all four layers (`scan_fold`, 1.78e-15) + certified law sets for `cumsum`/`cumprod`/`logcumsumexp`; **0 certified laws fire on real terms** (the minted total vs the corpus's emitted sequence; the paying `fold_emit` sibling is enumeration-ordered >200k deep) | `tools/atom_synth_loop.py` | [atom-synth-loop](../project/retros/atom-synth-loop.md) |
| anti-unification compression (layer 1) | **NULL on reach-expansion** — identical cheapest extractable member with/without atoms on all 3 workloads; compression real (12 vs 26 ops), strict arm worse under tight `max_nodes` | `tools/anti_unify_probe.py` | [ADR 0007](../project/adrs/0007-semantic-atom-minting.md) |
| combinator bridge (ADR 0008) | **321/321 corpus workloads round-trip structurally exact**, zero ops refuse translation, mean blowup **2.61×**; RetNet shared-state fold spells as `loop` term and verifies at 1e-7–2.6e-7; extract projects round-trip-equal (92 Δ sites) | `tools/combinator_probe.py` | [ADR 0008](../project/adrs/0008-combinator-presentation.md) |
| fused carrier lowering (`decayed_lower`) | **parity with the paper's own chunked impl** (2.08 vs 2.12 ms T=2048; 0.838 vs 0.822 ms T=1024 eager), **beats it under graph replay** (0.670 ms); fp64-exact 1.4e-16; FLA still ~7–11× ahead (kernel-level: fp64 GEMMs + materialized score tensor) | `tools/retnet_baseline_probe.py` | [retnet-external-baselines](../project/retros/retnet-external-baselines.md) |
| bounded lane on real BiRefNet (220M, HF weights) | **certified negative** — 20 offers at 1e-2, 56 at 3e-2, all correctly refused (thin dedup: bounded member prices 1.32× worse than dense); fp32 gate bitwise-clean; **BN census: 43 `BatchNorm2d`** (conv+BN targets exist) | `tools/birefnet_bounded_probe.py` | [birefnet-bounded-lane](../project/retros/birefnet-bounded-lane.md) |
| ONNX backend, both directions (ADR 0006) | **zero changes under `catopt-core`** for `OnnxSink` and `OnnxSource`; fp64 round-trip **1.11e-16** (machine eps) through `onnxruntime`; 107 op types / opsets 13–21 in, 169 ops out; package coverage 100% | `catopt_onnx` + `tests/test_onnx_*.py` | [onnx-package](../project/retros/onnx-package.md) |

Two reconciliation notes:

- **The contraction "~1.8×" claim is stale.**  Early numbers had
  the learned policy at 1.54–1.78× of `opt_einsum`'s randomised
  greedy at n=40; the RL curriculum regime narrowed the gap to
  ~1.04×, and the oe-all **distilled** player that now ships as
  `catopt_torch.load_contraction_policy()`'s default *beats* the
  teacher at equal wall-clock (the figure in the table above).  The
  curriculum-RL weights remain bundled as an alternate artifact.
- **The `bounded_e2e` baseline's headline is vs eager** (best
  1.322×, above).  Its per-cell `speedup_vs_inductor` spans
  0.94–1.25 (1.16–1.25 over the budgets that delivered a bounded
  member) — so the README's "1.15–1.32× vs Inductor" is not the
  pinned baseline's literal column: the 1.32 is vs eager and the
  vs-Inductor delivered range tops out at 1.25 on this baseline.


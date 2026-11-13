# Research agenda — self-mined proposals

Mined 2026-11-13 from 21 overnight ledgers, 70 frontier rows, 271 sweep rows, 14 pay_measured rows, and a 800-case coverage census.

**Proposals only — nothing here auto-executes.**  Each item carries measured evidence and a suggested lever; humans approve by ingesting into `project/TODO.md` (commands below) before any work starts.

## Ranked proposals

| # | score | headroom | conf | signal | proposal |
|---|-------|----------|------|--------|----------|
| 1 | 0.829 | 0.83 | high | `paymeasured_confound` | 6/14 pay_measured rows are executor-confounded — the marginal measures routing, not the fold |
| 2 | 0.600 | 1.00 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:ChunkedSSD: 1543.317us ceiling, captured=0.0584, vs_compile=None |
| 3 | 0.600 | 1.00 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:ChunkedSSD: 1751.576us ceiling, captured=-0.0615, vs_compile=0.286 |
| 4 | 0.600 | 1.00 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:DecoupledMLA: 140.084us ceiling, captured=-0.2396, vs_compile=0.6241 |
| 5 | 0.600 | 1.00 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:DeltaRuleBlock: 163.416us ceiling, captured=0.0055, vs_compile=0.7611 |
| 6 | 0.600 | 1.00 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:GatedDeltaRule: 260.755us ceiling, captured=-0.032, vs_compile=0.9556 |
| 7 | 0.600 | 1.00 | medium | `ledger_pay_decline` | Pay-declined law near the bar: grammar:sub_add_factor (changed=4, exec=1.0x) |
| 8 | 0.600 | 1.00 | medium | `ledger_pay_decline` | Pay-declined law near the bar: syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_10@RMSNormPerHead (changed=5, exec=0.914x) |
| 9 | 0.600 | 1.00 | medium | `ledger_pay_decline` | Pay-declined law near the bar: syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_6@RMSNormPerHead (changed=5, exec=0.914x) |
| 10 | 0.600 | 1.00 | medium | `ledger_pay_decline` | Pay-declined law near the bar: syn_naturality_scalar_rev__slice_mul_1@DilatedBlockAttention (changed=5, exec=0.979x) |
| 11 | 0.600 | 1.00 | medium | `ledger_pay_decline` | Pay-declined law near the bar: syn_naturality_scalar_rev__slice_mul_0@DilatedBlockAttention (changed=4, exec=0.931x) |
| 12 | 0.581 | 0.58 | high | `sweep_straddle` | 18/64 claim-bearing sweep rows straddle parity — referee noise decides keep/drop |
| 13 | 0.543 | 0.91 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:CapsuleRouting: 140.666us ceiling, captured=0.0687, vs_compile=0.4235 |
| 14 | 0.525 | 0.53 | high | `sweep_delivered_loser` | 52/64 claim-bearing delivered+verified rows run below parity (worst zoo:MambaSelectiveScan at 0.481x) |
| 15 | 0.488 | 0.81 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:RotaryPair: 224.173us ceiling, captured=0.4979, vs_compile=1.1796 |
| 16 | 0.457 | 0.76 | medium | `ledger_oracle_gap` | Oracle never sees an equal site in family layout (12/13 died at oracle, 0 eq) |
| 17 | 0.449 | 0.75 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:FNOBlock: 97.51us ceiling, captured=-0.023, vs_compile=1.8239 |
| 18 | 0.382 | 0.64 | medium | `coverage_rules` | 112/154 shipped rules never arm under the stream's rulesets |
| 19 | 0.343 | 0.57 | medium | `coverage_ops` | 102 expressible ops never occur in the 800-case stream |
| 20 | 0.285 | 0.95 | low | `ledger_pay_decline` | Pay-declined law near the bar: fold:sdpa_fold_addmul (changed=6, exec=0.21x) |
| 21 | 0.270 | 0.45 | medium | `coverage_attrs` | 38 declared (op, attr) pairs never occur in the stream |
| 22 | 0.225 | 0.75 | low | `ledger_pay_decline` | Pay-declined law near the bar: linear_factor (changed=4, exec=0.667x) |
| 23 | 0.225 | 0.75 | low | `ledger_pay_decline` | Pay-declined law near the bar: matmul_factor (changed=4, exec=0.667x) |

## Evidence per proposal

### 1. 6/14 pay_measured rows are executor-confounded — the marginal measures routing, not the fold

- **signal:** `paymeasured_confound` · **target:** `{"domain": "gen", "tier": "measurement"}`
- **lever:** synop routes to IRModule while noprim routes to BatchedScanModule — the A/B compares executors, not the fold.  Re-run marginals with a pinned executor or add a same-executor arm before trusting the pay table.
- **evidence:**
  - source: `docs/reports/pay_measured.json`
  - confounded: 6
  - total: 14
  - fraction: 0.4286
  - worst_module: "HighwayGate"
  - worst_size: "small"
  - worst_marginal: 1.161099203857289
  - worst_note: "executor confound: synop routes to IRModule, noprim to BatchedScanModule \u2014 the marginal compares different executors, not just the fold."
  - examples: ["HighwayGate/small", "HighwayGate/large", "GRUFusedGate/large", "GRUFusedGate/small", "GatedDeltaRule/small", "GatedDeltaRule/large"]

### 2. Uncaptured headroom on zoo:ChunkedSSD: 1543.317us ceiling, captured=0.0584, vs_compile=None

- **signal:** `frontier_delivery_gap` · **target:** `{"domain": "gen", "tier": "launch-bound", "op": "zoo:ChunkedSSD"}`
- **lever:** The algebraic ceiling is real but the delivered body doesn't reach it — profile the delivered module against the ceiling decomposition (dispatch vs device share) and attack the largest residual term, not the rewrite bank.
- **evidence:**
  - source: `docs/reports/frontier_fp32.json`
  - case: "zoo:ChunkedSSD"
  - regime: "launch-bound"
  - headroom_us: 1543.317
  - ceiling: 2.7248
  - speedup: 1.1007
  - captured: 0.0584
  - vs_compile: null
  - beats_compile: null
  - kernels: 16

### 3. Uncaptured headroom on zoo:ChunkedSSD: 1751.576us ceiling, captured=-0.0615, vs_compile=0.286

- **signal:** `frontier_delivery_gap` · **target:** `{"domain": "gen", "tier": "launch-bound", "op": "zoo:ChunkedSSD"}`
- **lever:** The algebraic ceiling is real but the delivered body doesn't reach it — profile the delivered module against the ceiling decomposition (dispatch vs device share) and attack the largest residual term, not the rewrite bank.
- **evidence:**
  - source: `docs/reports/frontier.json`
  - case: "zoo:ChunkedSSD"
  - regime: "launch-bound"
  - headroom_us: 1751.576
  - ceiling: 3.0211
  - speedup: 0.8757
  - captured: -0.0615
  - vs_compile: 0.286
  - beats_compile: false
  - kernels: 16

### 4. Uncaptured headroom on zoo:DecoupledMLA: 140.084us ceiling, captured=-0.2396, vs_compile=0.6241

- **signal:** `frontier_delivery_gap` · **target:** `{"domain": "gen", "tier": "launch-leaning", "op": "zoo:DecoupledMLA"}`
- **lever:** The algebraic ceiling is real but the delivered body doesn't reach it — profile the delivered module against the ceiling decomposition (dispatch vs device share) and attack the largest residual term, not the rewrite bank.
- **evidence:**
  - source: `docs/reports/frontier.json`
  - case: "zoo:DecoupledMLA"
  - regime: "launch-leaning"
  - headroom_us: 140.084
  - ceiling: 1.3889
  - speedup: 0.9068
  - captured: -0.2396
  - vs_compile: 0.6241
  - beats_compile: false
  - kernels: 17

### 5. Uncaptured headroom on zoo:DeltaRuleBlock: 163.416us ceiling, captured=0.0055, vs_compile=0.7611

- **signal:** `frontier_delivery_gap` · **target:** `{"domain": "gen", "tier": "launch-bound", "op": "zoo:DeltaRuleBlock"}`
- **lever:** The algebraic ceiling is real but the delivered body doesn't reach it — profile the delivered module against the ceiling decomposition (dispatch vs device share) and attack the largest residual term, not the rewrite bank.
- **evidence:**
  - source: `docs/reports/frontier.json`
  - case: "zoo:DeltaRuleBlock"
  - regime: "launch-bound"
  - headroom_us: 163.416
  - ceiling: 2.1186
  - speedup: 1.0062
  - captured: 0.0055
  - vs_compile: 0.7611
  - beats_compile: false
  - kernels: 9

### 6. Uncaptured headroom on zoo:GatedDeltaRule: 260.755us ceiling, captured=-0.032, vs_compile=0.9556

- **signal:** `frontier_delivery_gap` · **target:** `{"domain": "gen", "tier": "launch-bound", "op": "zoo:GatedDeltaRule"}`
- **lever:** The algebraic ceiling is real but the delivered body doesn't reach it — profile the delivered module against the ceiling decomposition (dispatch vs device share) and attack the largest residual term, not the rewrite bank.
- **evidence:**
  - source: `docs/reports/frontier.json`
  - case: "zoo:GatedDeltaRule"
  - regime: "launch-bound"
  - headroom_us: 260.755
  - ceiling: 2.5316
  - speedup: 0.951
  - captured: -0.032
  - vs_compile: 0.9556
  - beats_compile: false
  - kernels: 12

### 7. Pay-declined law near the bar: grammar:sub_add_factor (changed=4, exec=1.0x)

- **signal:** `ledger_pay_decline` · **target:** `{"domain": "discovery", "op": "grammar:sub_add_factor"}`
- **lever:** Re-price or re-codegen this law's payoff: it fires and rewrites the graph but the measured kernel doesn't clear the keep bar — check the cost model's exec term against a hand-tuned kernel for the same site.
- **evidence:**
  - source: `docs/reports/results_overnight_real_preview.json`
  - fires: 4
  - changed: 4
  - paid: 0
  - exec_ratio: 1.0
  - count_ratio: 1.0
  - flops_ratio: 1.0
  - launch_ratio: 1.0
  - cases: 40
  - corpus: "zoo-real"
  - arena: {"verdict": "false", "relation": "new", "num_true": null, "base_cost": 2.0, "cand_cost": 2.0}

### 8. Pay-declined law near the bar: syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_10@RMSNormPerHead (changed=5, exec=0.914x)

- **signal:** `ledger_pay_decline` · **target:** `{"domain": "discovery", "op": "syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_10@RMSNormPerHead"}`
- **lever:** Re-price or re-codegen this law's payoff: it fires and rewrites the graph but the measured kernel doesn't clear the keep bar — check the cost model's exec term against a hand-tuned kernel for the same site.
- **evidence:**
  - source: `docs/reports/results_overnight_deep.json`
  - fires: 5
  - changed: 5
  - paid: 4
  - exec_ratio: 0.914
  - count_ratio: 1.0
  - flops_ratio: 0.9
  - launch_ratio: 0.9
  - cases: 57
  - corpus: "zoo-real"
  - arena: {"verdict": "false", "relation": "new", "num_true": null, "base_cost": 9.0, "cand_cost": 9.0}

### 9. Pay-declined law near the bar: syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_6@RMSNormPerHead (changed=5, exec=0.914x)

- **signal:** `ledger_pay_decline` · **target:** `{"domain": "discovery", "op": "syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_6@RMSNormPerHead"}`
- **lever:** Re-price or re-codegen this law's payoff: it fires and rewrites the graph but the measured kernel doesn't clear the keep bar — check the cost model's exec term against a hand-tuned kernel for the same site.
- **evidence:**
  - source: `docs/reports/results_overnight_real_full.json`
  - fires: 5
  - changed: 5
  - paid: 4
  - exec_ratio: 0.914
  - count_ratio: 1.0
  - flops_ratio: 0.9
  - launch_ratio: 0.9
  - cases: 57
  - corpus: "zoo-real"
  - arena: {"verdict": "false", "relation": "new", "num_true": null, "base_cost": 9.0, "cand_cost": 9.0}

### 10. Pay-declined law near the bar: syn_naturality_scalar_rev__slice_mul_1@DilatedBlockAttention (changed=5, exec=0.979x)

- **signal:** `ledger_pay_decline` · **target:** `{"domain": "discovery", "op": "syn_naturality_scalar_rev__slice_mul_1@DilatedBlockAttention"}`
- **lever:** Re-price or re-codegen this law's payoff: it fires and rewrites the graph but the measured kernel doesn't clear the keep bar — check the cost model's exec term against a hand-tuned kernel for the same site.
- **evidence:**
  - source: `docs/reports/results_overnight_deep.json`
  - fires: 5
  - changed: 5
  - paid: 4
  - exec_ratio: 0.979
  - count_ratio: 0.978
  - flops_ratio: 0.981
  - launch_ratio: 0.973
  - cases: 57
  - corpus: "zoo-real"
  - arena: {"verdict": "false", "relation": "new", "num_true": null, "base_cost": Infinity, "cand_cost": Infinity}

### 11. Pay-declined law near the bar: syn_naturality_scalar_rev__slice_mul_0@DilatedBlockAttention (changed=4, exec=0.931x)

- **signal:** `ledger_pay_decline` · **target:** `{"domain": "discovery", "op": "syn_naturality_scalar_rev__slice_mul_0@DilatedBlockAttention"}`
- **lever:** Re-price or re-codegen this law's payoff: it fires and rewrites the graph but the measured kernel doesn't clear the keep bar — check the cost model's exec term against a hand-tuned kernel for the same site.
- **evidence:**
  - source: `docs/reports/results_overnight_deep.json`
  - fires: 5
  - changed: 4
  - paid: 4
  - exec_ratio: 0.931
  - count_ratio: 0.929
  - flops_ratio: 0.952
  - launch_ratio: 0.943
  - cases: 57
  - corpus: "zoo-real"
  - arena: {"verdict": "false", "relation": "new", "num_true": null, "base_cost": Infinity, "cand_cost": Infinity}

### 12. 18/64 claim-bearing sweep rows straddle parity — referee noise decides keep/drop

- **signal:** `sweep_straddle` · **target:** `{"domain": "gen", "tier": "referee"}`
- **lever:** Interleaved-window referee noise flips marginal claims between runs — the two-sample borderline guard exists but the residual band is still wide.  Widen confirmation windows or add a third arm (eager-vs-compiled delta) before the keep vote.
- **evidence:**
  - source: `docs/reports/zoo_chain_sweep.json`
  - source: `docs/reports/zoo_chain_sweep_fp32.json`
  - source: `docs/reports/zoo_chain_sweep_prim.json`
  - source: `docs/reports/zoo_chain_sweep_t016_strided_all.json`
  - source: `docs/reports/zoo_chain_sweep_t019_gate.json`
  - straddling_rows: 18
  - claim_rows: 64
  - total_rows: 277
  - fraction: 0.2812
  - worst_spread: [0.986, 1.282]
  - worst_case: "zoo:AttnLogitSoftcap"
  - examples: ["zoo:AttnLogitSoftcap", "zoo:GatedAttentionUnit", "zoo:GEGLUBlock", "zoo:PartialRotary", "zoo:DiffAttention", "zoo:DeltaRuleBlock"]

### 13. Uncaptured headroom on zoo:CapsuleRouting: 140.666us ceiling, captured=0.0687, vs_compile=0.4235

- **signal:** `frontier_delivery_gap` · **target:** `{"domain": "gen", "tier": "launch-bound", "op": "zoo:CapsuleRouting"}`
- **lever:** The algebraic ceiling is real but the delivered body doesn't reach it — profile the delivered module against the ceiling decomposition (dispatch vs device share) and attack the largest residual term, not the rewrite bank.
- **evidence:**
  - source: `docs/reports/frontier.json`
  - case: "zoo:CapsuleRouting"
  - regime: "launch-bound"
  - headroom_us: 140.666
  - ceiling: 2.0161
  - speedup: 1.0698
  - captured: 0.0687
  - vs_compile: 0.4235
  - beats_compile: false
  - kernels: 13

### 14. 52/64 claim-bearing delivered+verified rows run below parity (worst zoo:MambaSelectiveScan at 0.481x)

- **signal:** `sweep_delivered_loser` · **target:** `{"domain": "gen", "tier": "referee"}`
- **lever:** Verified-but-slower deliveries mean the keep threshold admits losers on some regimes — re-derive the margin per regime (launch-bound vs compute-bound) instead of one global bar.
- **evidence:**
  - source: `docs/reports/zoo_chain_sweep.json`
  - source: `docs/reports/zoo_chain_sweep_fp32.json`
  - source: `docs/reports/zoo_chain_sweep_prim.json`
  - source: `docs/reports/zoo_chain_sweep_t016_drain.json`
  - source: `docs/reports/zoo_chain_sweep_t016_postgate.json`
  - source: `docs/reports/zoo_chain_sweep_t016_strided_all.json`
  - source: `docs/reports/zoo_chain_sweep_t019_gate.json`
  - delivered_losers: 52
  - claim_rows: 64
  - total_rows: 277
  - worst_case: "zoo:MambaSelectiveScan"
  - worst_speedup: 0.481
  - examples: [["zoo:MambaSelectiveScan", 0.481], ["zoo:MambaSelectiveScan", 0.481], ["zoo:GatedDeltaRule", 0.486], ["zoo:MambaSelectiveScan", 0.497], ["zoo:GatedDeltaRule", 0.5], ["zoo:MambaSelectiveScan", 0.504]]

### 15. Uncaptured headroom on zoo:RotaryPair: 224.173us ceiling, captured=0.4979, vs_compile=1.1796

- **signal:** `frontier_delivery_gap` · **target:** `{"domain": "gen", "tier": "launch-bound", "op": "zoo:RotaryPair"}`
- **lever:** The algebraic ceiling is real but the delivered body doesn't reach it — profile the delivered module against the ceiling decomposition (dispatch vs device share) and attack the largest residual term, not the rewrite bank.
- **evidence:**
  - source: `docs/reports/frontier.json`
  - case: "zoo:RotaryPair"
  - regime: "launch-bound"
  - headroom_us: 224.173
  - ceiling: 2.1459
  - speedup: 1.5705
  - captured: 0.4979
  - vs_compile: 1.1796
  - beats_compile: true
  - kernels: 12

### 16. Oracle never sees an equal site in family layout (12/13 died at oracle, 0 eq)

- **signal:** `ledger_oracle_gap` · **target:** `{"domain": "discovery", "tier": "layout"}`
- **lever:** The mint produces claims this family's corpus cannot evaluate — extend the corpus/env coverage for the family's op patterns (corpus_cov gap report names the missing ops/attrs) before touching the laws.
- **evidence:**
  - source: `docs/reports/results_overnight_deep.json`
  - source: `docs/reports/results_overnight_deep2.json`
  - source: `docs/reports/results_overnight_prim.json`
  - source: `docs/reports/results_overnight_prim_auto_admit.json`
  - source: `docs/reports/results_overnight_prim_bootstrap.json`
  - source: `docs/reports/results_overnight_prim_deep.json`
  - source: `docs/reports/results_overnight_prim_depth3.json`
  - source: `docs/reports/results_overnight_prim_v2.json`
  - source: `docs/reports/results_overnight_prim_v3.json`
  - source: `docs/reports/results_overnight_prim_v4.json`
  - source: `docs/reports/results_overnight_prim_vocab.json`
  - source: `docs/reports/results_overnight_real_full.json`
  - source: `docs/reports/results_overnight_real_fusion.json`
  - source: `docs/reports/results_overnight_real_preview.json`
  - source: `docs/reports/results_overnight_real_v2.json`
  - source: `docs/reports/results_overnight_real_v3.json`
  - source: `docs/reports/results_overnight_real_v4.json`
  - source: `docs/reports/results_overnight_real_v5.json`
  - source: `docs/reports/results_overnight_real_v6.json`
  - source: `docs/reports/results_overnight_xmodule.json`
  - source: `docs/reports/results_overnight_xmodule_v2.json`
  - candidates: 13
  - oracle_failures: 12
  - eq_total: 0
  - acc_total: 47
  - examples: ["reshape_transpose", "reshape_transpose", "reshape_transpose", "reshape_transpose"]

### 17. Uncaptured headroom on zoo:FNOBlock: 97.51us ceiling, captured=-0.023, vs_compile=1.8239

- **signal:** `frontier_delivery_gap` · **target:** `{"domain": "gen", "tier": "launch-bound", "op": "zoo:FNOBlock"}`
- **lever:** The algebraic ceiling is real but the delivered body doesn't reach it — profile the delivered module against the ceiling decomposition (dispatch vs device share) and attack the largest residual term, not the rewrite bank.
- **evidence:**
  - source: `docs/reports/frontier.json`
  - case: "zoo:FNOBlock"
  - regime: "launch-bound"
  - headroom_us: 97.51
  - ceiling: 5.9524
  - speedup: 0.8859
  - captured: -0.023
  - vs_compile: 1.8239
  - beats_compile: true
  - kernels: 11

### 18. 112/154 shipped rules never arm under the stream's rulesets

- **signal:** `coverage_rules` · **target:** `{"domain": "meta-arena", "tier": "corpus"}`
- **lever:** Unarmed rules need boards whose rulesets carry them AND terms matching their LHS — the fire/saturate payoff dimension stays unmeasured until then.
- **evidence:**
  - source: `corpus_cov.coverage_gaps (live census)`
  - cases: 800
  - missing: 112
  - total: 154
  - covered: 42
  - missing_sample: ["aff_assoc", "aff_assoc_rev", "aff_compose_unfold", "aff_lift_step", "affd_assoc", "affd_assoc_rev", "affd_compose_unfold", "affd_lift", "affd_lift_post", "affd_lift_post_swap", "affd_lift_step", "affd_lift_step_post", "affd_lift_step_post_swap", "affd_lift_step_swap", "affd_lift_swap", "affd_lift_…

### 19. 102 expressible ops never occur in the 800-case stream

- **signal:** `coverage_ops` · **target:** `{"domain": "meta-arena", "tier": "corpus"}`
- **lever:** Extend coverage_builders' templates to the missing op classes (synop*/new classes need new templates — the honest extension seam named in corpus_cov).
- **evidence:**
  - source: `corpus_cov.coverage_gaps (live census)`
  - cases: 800
  - missing: 102
  - total: 165
  - covered: 63
  - missing_sample: ["addcdiv", "addcmul", "aff_compose", "aff_diag", "affd_compose", "argmin", "atan2", "batch_norm", "bitwise_and", "bitwise_or", "conv2d", "count_nonzero", "dropout", "eq", "fmax", "fmin", "fmod", "ge", "glu", "gt"]

### 20. Pay-declined law near the bar: fold:sdpa_fold_addmul (changed=6, exec=0.21x)

- **signal:** `ledger_pay_decline` · **target:** `{"domain": "discovery", "op": "fold:sdpa_fold_addmul"}`
- **lever:** Re-price or re-codegen this law's payoff: it fires and rewrites the graph but the measured kernel doesn't clear the keep bar — check the cost model's exec term against a hand-tuned kernel for the same site.
- **evidence:**
  - source: `docs/reports/results_overnight_deep.json`
  - fires: 6
  - changed: 6
  - paid: 6
  - exec_ratio: 0.21
  - count_ratio: 0.311
  - flops_ratio: 0.326
  - launch_ratio: 0.326
  - cases: 57
  - corpus: "zoo-real"
  - arena: {"verdict": "known", "relation": "duplicate", "num_true": null, "base_cost": 7.0, "cand_cost": 7.0}

### 21. 38 declared (op, attr) pairs never occur in the stream

- **signal:** `coverage_attrs` · **target:** `{"domain": "meta-arena", "tier": "corpus"}`
- **lever:** Attr-carrying case kinds: the zoo census found zero attr elementwise ops — decide whether attrs need synthetic boards or real-workload intake first.
- **evidence:**
  - source: `corpus_cov.coverage_gaps (live census)`
  - cases: 800
  - missing: 38
  - total: 91
  - covered: 53
  - missing_sample: ["argmin.dim", "argmin.keepdim", "batch_norm.cudnn_enabled", "batch_norm.eps", "batch_norm.momentum", "batch_norm.training", "conv2d.dilation", "conv2d.groups", "conv2d.padding", "conv2d.stride", "count_nonzero.dim", "dropout.p", "dropout.train", "glu.dim", "isclose.atol", "isclose.equal_nan", "iscl…

### 22. Pay-declined law near the bar: linear_factor (changed=4, exec=0.667x)

- **signal:** `ledger_pay_decline` · **target:** `{"domain": "discovery", "op": "linear_factor"}`
- **lever:** Re-price or re-codegen this law's payoff: it fires and rewrites the graph but the measured kernel doesn't clear the keep bar — check the cost model's exec term against a hand-tuned kernel for the same site.
- **evidence:**
  - source: `docs/reports/results_overnight_real_preview.json`
  - fires: 4
  - changed: 4
  - paid: 4
  - exec_ratio: 0.667
  - count_ratio: 0.667
  - flops_ratio: 0.6
  - launch_ratio: 0.625
  - cases: 40
  - corpus: "zoo-real"
  - arena: {"verdict": "known", "relation": "duplicate", "num_true": null, "base_cost": 5.0, "cand_cost": 3.0}

### 23. Pay-declined law near the bar: matmul_factor (changed=4, exec=0.667x)

- **signal:** `ledger_pay_decline` · **target:** `{"domain": "discovery", "op": "matmul_factor"}`
- **lever:** Re-price or re-codegen this law's payoff: it fires and rewrites the graph but the measured kernel doesn't clear the keep bar — check the cost model's exec term against a hand-tuned kernel for the same site.
- **evidence:**
  - source: `docs/reports/results_overnight_real_preview.json`
  - fires: 4
  - changed: 4
  - paid: 4
  - exec_ratio: 0.667
  - count_ratio: 0.667
  - flops_ratio: 0.6
  - launch_ratio: 0.625
  - cases: 40
  - corpus: "zoo-real"
  - arena: {"verdict": "known", "relation": "inverse", "num_true": null, "base_cost": 5.0, "cand_cost": 3.0}

## Ledger ingest (human approval required)

Each proposal maps to one `myco todo add` — run the ones you accept (`--no-commit` leaves the commit to `myco check --fix` / your own commit):

```sh
myco todo add "6/14 pay_measured rows are executor-confounded — the marginal measures routing, not the fold" --pri high --label agenda --no-commit
myco todo add "Uncaptured headroom on zoo:ChunkedSSD: 1543.317us ceiling, captured=0.0584, vs_compile=None" --pri high --label agenda --no-commit
myco todo add "Uncaptured headroom on zoo:ChunkedSSD: 1751.576us ceiling, captured=-0.0615, vs_compile=0.286" --pri high --label agenda --no-commit
myco todo add "Uncaptured headroom on zoo:DecoupledMLA: 140.084us ceiling, captured=-0.2396, vs_compile=0.6241" --pri high --label agenda --no-commit
myco todo add "Uncaptured headroom on zoo:DeltaRuleBlock: 163.416us ceiling, captured=0.0055, vs_compile=0.7611" --pri high --label agenda --no-commit
myco todo add "Uncaptured headroom on zoo:GatedDeltaRule: 260.755us ceiling, captured=-0.032, vs_compile=0.9556" --pri high --label agenda --no-commit
myco todo add "Pay-declined law near the bar: grammar:sub_add_factor (changed=4, exec=1.0x)" --pri high --label agenda --no-commit
myco todo add "Pay-declined law near the bar: syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_10@RMSNormPerHead (changed=5, exec=0.914x)" --pri high --label agenda --no-commit
myco todo add "Pay-declined law near the bar: syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_6@RMSNormPerHead (changed=5, exec=0.914x)" --pri high --label agenda --no-commit
myco todo add "Pay-declined law near the bar: syn_naturality_scalar_rev__slice_mul_1@DilatedBlockAttention (changed=5, exec=0.979x)" --pri high --label agenda --no-commit
myco todo add "Pay-declined law near the bar: syn_naturality_scalar_rev__slice_mul_0@DilatedBlockAttention (changed=4, exec=0.931x)" --pri high --label agenda --no-commit
myco todo add "18/64 claim-bearing sweep rows straddle parity — referee noise decides keep/drop" --pri high --label agenda --no-commit
myco todo add "Uncaptured headroom on zoo:CapsuleRouting: 140.666us ceiling, captured=0.0687, vs_compile=0.4235" --pri high --label agenda --no-commit
myco todo add "52/64 claim-bearing delivered+verified rows run below parity (worst zoo:MambaSelectiveScan at 0.481x)" --pri high --label agenda --no-commit
myco todo add "Uncaptured headroom on zoo:RotaryPair: 224.173us ceiling, captured=0.4979, vs_compile=1.1796" --pri med --label agenda --no-commit
myco todo add "Oracle never sees an equal site in family layout (12/13 died at oracle, 0 eq)" --pri med --label agenda --no-commit
myco todo add "Uncaptured headroom on zoo:FNOBlock: 97.51us ceiling, captured=-0.023, vs_compile=1.8239" --pri med --label agenda --no-commit
myco todo add "112/154 shipped rules never arm under the stream's rulesets" --pri med --label agenda --no-commit
myco todo add "102 expressible ops never occur in the 800-case stream" --pri med --label agenda --no-commit
myco todo add "Pay-declined law near the bar: fold:sdpa_fold_addmul (changed=6, exec=0.21x)" --pri med --label agenda,low-confidence --no-commit
myco todo add "38 declared (op, attr) pairs never occur in the stream" --pri med --label agenda --no-commit
myco todo add "Pay-declined law near the bar: linear_factor (changed=4, exec=0.667x)" --pri med --label agenda,low-confidence --no-commit
myco todo add "Pay-declined law near the bar: matmul_factor (changed=4, exec=0.667x)" --pri med --label agenda,low-confidence --no-commit
```

# Research agenda — self-mined proposals

Mined 2026-11-18 from 21 overnight ledgers, 72 frontier rows, 271 sweep rows, 14 pay_measured rows, and a 800-case coverage census.

**Proposals only — nothing here auto-executes.**  Each item carries measured evidence and a suggested lever; humans approve by ingesting into `project/TODO.md` (commands below) before any work starts.

## Ranked proposals

| # | score | headroom | conf | signal | proposal |
|---|-------|----------|------|--------|----------|
| 1 | 0.600 | 1.00 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:ChunkedSSD: 1445.441us ceiling, captured=-0.0695, vs_compile=0.2548 |
| 2 | 0.600 | 1.00 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:ChunkedSSD: 1465.21us ceiling, captured=0.0025, vs_compile=0.2981 |
| 3 | 0.600 | 1.00 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:GatedDeltaRule: 242.786us ceiling, captured=0.1461, vs_compile=1.3175 |
| 4 | 0.600 | 1.00 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:GatedLinearAttention: 1196.296us ceiling, captured=-0.5062, vs_compile=0.1756 |
| 5 | 0.600 | 1.00 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:MambaSelectiveScan: 1448.219us ceiling, captured=-0.1543, vs_compile=0.1308 |
| 6 | 0.600 | 1.00 | medium | `ledger_pay_decline` | Pay-declined law near the bar: grammar:sub_add_factor (changed=4, exec=1.0x) |
| 7 | 0.555 | 0.93 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:DecoupledMLA: 132.44us ceiling, captured=-0.0197, vs_compile=0.8042 |
| 8 | 0.550 | 0.55 | high | `sweep_straddle` | 17/68 claim-bearing sweep rows straddle parity — referee noise decides keep/drop |
| 9 | 0.543 | 0.91 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:CapsuleRouting: 140.666us ceiling, captured=0.0687, vs_compile=0.4235 |
| 10 | 0.512 | 0.51 | high | `sweep_delivered_loser` | 53/68 claim-bearing delivered+verified rows run below parity (worst zoo:GatedLinearAttention at 0.287x) |
| 11 | 0.488 | 0.81 | medium | `frontier_delivery_gap` | Uncaptured headroom on zoo:RotaryPair: 224.173us ceiling, captured=0.4979, vs_compile=1.1796 |
| 12 | 0.457 | 0.76 | medium | `ledger_oracle_gap` | Oracle never sees an equal site in family layout (12/13 died at oracle, 0 eq) |
| 13 | 0.400 | 0.40 | high | `ledger_verify_decline` | Verify-declined law: fold:sdpa_fold_addmul (changed=6, vfail=4) |
| 14 | 0.400 | 0.40 | high | `ledger_verify_decline` | Verify-declined law: syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_10@RMSNormPerHead (changed=5, vfail=4) |
| 15 | 0.400 | 0.40 | high | `ledger_verify_decline` | Verify-declined law: syn_naturality_scalar_rev__slice_mul_0@DilatedBlockAttention (changed=4, vfail=4) |
| 16 | 0.400 | 0.40 | high | `ledger_verify_decline` | Verify-declined law: syn_naturality_scalar_rev__slice_mul_1@DilatedBlockAttention (changed=5, vfail=4) |
| 17 | 0.227 | 0.38 | medium | `coverage_attrs` | 27 declared (op, attr) pairs never occur in the stream |
| 18 | 0.225 | 0.75 | low | `ledger_pay_decline` | Pay-declined law near the bar: reshape_reshape (changed=4, exec=0.5x) |
| 19 | 0.195 | 0.32 | medium | `coverage_ops` | 35 expressible ops never occur in the 800-case stream |
| 20 | 0.120 | 0.20 | medium | `coverage_rules` | 0/156 shipped rules never arm under the stream's rulesets |

## Evidence per proposal

### 1. Uncaptured headroom on zoo:ChunkedSSD: 1445.441us ceiling, captured=-0.0695, vs_compile=0.2548

- **signal:** `frontier_delivery_gap` · **target:** `{"domain": "gen", "tier": "launch-bound", "op": "zoo:ChunkedSSD"}`
- **lever:** The algebraic ceiling is real but the delivered body doesn't reach it — profile the delivered module against the ceiling decomposition (dispatch vs device share) and attack the largest residual term, not the rewrite bank.
- **evidence:**
  - source: `docs/reports/frontier_fp32.json`
  - case: "zoo:ChunkedSSD"
  - regime: "launch-bound"
  - headroom_us: 1445.441
  - ceiling: 3.0211
  - speedup: 0.8594
  - captured: -0.0695
  - vs_compile: 0.2548
  - beats_compile: false
  - kernels: 16

### 2. Uncaptured headroom on zoo:ChunkedSSD: 1465.21us ceiling, captured=0.0025, vs_compile=0.2981

- **signal:** `frontier_delivery_gap` · **target:** `{"domain": "gen", "tier": "launch-bound", "op": "zoo:ChunkedSSD"}`
- **lever:** The algebraic ceiling is real but the delivered body doesn't reach it — profile the delivered module against the ceiling decomposition (dispatch vs device share) and attack the largest residual term, not the rewrite bank.
- **evidence:**
  - source: `docs/reports/frontier.json`
  - case: "zoo:ChunkedSSD"
  - regime: "launch-bound"
  - headroom_us: 1465.21
  - ceiling: 3.0211
  - speedup: 1.005
  - captured: 0.0025
  - vs_compile: 0.2981
  - beats_compile: false
  - kernels: 16

### 3. Uncaptured headroom on zoo:GatedDeltaRule: 242.786us ceiling, captured=0.1461, vs_compile=1.3175

- **signal:** `frontier_delivery_gap` · **target:** `{"domain": "gen", "tier": "launch-bound", "op": "zoo:GatedDeltaRule"}`
- **lever:** The algebraic ceiling is real but the delivered body doesn't reach it — profile the delivered module against the ceiling decomposition (dispatch vs device share) and attack the largest residual term, not the rewrite bank.
- **evidence:**
  - source: `docs/reports/frontier.json`
  - case: "zoo:GatedDeltaRule"
  - regime: "launch-bound"
  - headroom_us: 242.786
  - ceiling: 2.5316
  - speedup: 1.2238
  - captured: 0.1461
  - vs_compile: 1.3175
  - beats_compile: true
  - kernels: 12

### 4. Uncaptured headroom on zoo:GatedLinearAttention: 1196.296us ceiling, captured=-0.5062, vs_compile=0.1756

- **signal:** `frontier_delivery_gap` · **target:** `{"domain": "gen", "tier": "launch-bound", "op": "zoo:GatedLinearAttention"}`
- **lever:** The algebraic ceiling is real but the delivered body doesn't reach it — profile the delivered module against the ceiling decomposition (dispatch vs device share) and attack the largest residual term, not the rewrite bank.
- **evidence:**
  - source: `docs/reports/frontier.json`
  - case: "zoo:GatedLinearAttention"
  - regime: "launch-bound"
  - headroom_us: 1196.296
  - ceiling: 2.4096
  - speedup: 0.2864
  - captured: -0.5062
  - vs_compile: 0.1756
  - beats_compile: false
  - kernels: 7

### 5. Uncaptured headroom on zoo:MambaSelectiveScan: 1448.219us ceiling, captured=-0.1543, vs_compile=0.1308

- **signal:** `frontier_delivery_gap` · **target:** `{"domain": "gen", "tier": "launch-bound", "op": "zoo:MambaSelectiveScan"}`
- **lever:** The algebraic ceiling is real but the delivered body doesn't reach it — profile the delivered module against the ceiling decomposition (dispatch vs device share) and attack the largest residual term, not the rewrite bank.
- **evidence:**
  - source: `docs/reports/frontier.json`
  - case: "zoo:MambaSelectiveScan"
  - regime: "launch-bound"
  - headroom_us: 1448.219
  - ceiling: 4.329
  - speedup: 0.4862
  - captured: -0.1543
  - vs_compile: 0.1308
  - beats_compile: false
  - kernels: 10

### 6. Pay-declined law near the bar: grammar:sub_add_factor (changed=4, exec=1.0x)

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

### 7. Uncaptured headroom on zoo:DecoupledMLA: 132.44us ceiling, captured=-0.0197, vs_compile=0.8042

- **signal:** `frontier_delivery_gap` · **target:** `{"domain": "gen", "tier": "launch-leaning", "op": "zoo:DecoupledMLA"}`
- **lever:** The algebraic ceiling is real but the delivered body doesn't reach it — profile the delivered module against the ceiling decomposition (dispatch vs device share) and attack the largest residual term, not the rewrite bank.
- **evidence:**
  - source: `docs/reports/frontier.json`
  - case: "zoo:DecoupledMLA"
  - regime: "launch-leaning"
  - headroom_us: 132.44
  - ceiling: 1.3889
  - speedup: 0.9923
  - captured: -0.0197
  - vs_compile: 0.8042
  - beats_compile: false
  - kernels: 17

### 8. 17/68 claim-bearing sweep rows straddle parity — referee noise decides keep/drop

- **signal:** `sweep_straddle` · **target:** `{"domain": "gen", "tier": "referee"}`
- **lever:** Interleaved-window referee noise flips marginal claims between runs — the two-sample borderline guard exists but the residual band is still wide.  Widen confirmation windows or add a third arm (eager-vs-compiled delta) before the keep vote.
- **evidence:**
  - source: `docs/reports/zoo_chain_sweep.json`
  - source: `docs/reports/zoo_chain_sweep_fp32.json`
  - source: `docs/reports/zoo_chain_sweep_prim.json`
  - source: `docs/reports/zoo_chain_sweep_t016_strided_all.json`
  - source: `docs/reports/zoo_chain_sweep_t019_gate.json`
  - straddling_rows: 17
  - claim_rows: 68
  - total_rows: 281
  - fraction: 0.25
  - worst_spread: [0.986, 1.282]
  - worst_case: "zoo:AttnLogitSoftcap"
  - examples: ["zoo:AttnLogitSoftcap", "zoo:GatedAttentionUnit", "zoo:GEGLUBlock", "zoo:PartialRotary", "zoo:DiffAttention", "zoo:DeltaRuleBlock"]

### 9. Uncaptured headroom on zoo:CapsuleRouting: 140.666us ceiling, captured=0.0687, vs_compile=0.4235

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

### 10. 53/68 claim-bearing delivered+verified rows run below parity (worst zoo:GatedLinearAttention at 0.287x)

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
  - delivered_losers: 53
  - claim_rows: 68
  - total_rows: 281
  - worst_case: "zoo:GatedLinearAttention"
  - worst_speedup: 0.287
  - examples: [["zoo:GatedLinearAttention", 0.287], ["zoo:MambaSelectiveScan", 0.481], ["zoo:MambaSelectiveScan", 0.481], ["zoo:MambaSelectiveScan", 0.482], ["zoo:MambaSelectiveScan", 0.497], ["zoo:GatedDeltaRule", 0.5]]

### 11. Uncaptured headroom on zoo:RotaryPair: 224.173us ceiling, captured=0.4979, vs_compile=1.1796

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

### 12. Oracle never sees an equal site in family layout (12/13 died at oracle, 0 eq)

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

### 13. Verify-declined law: fold:sdpa_fold_addmul (changed=6, vfail=4)

- **signal:** `ledger_verify_decline` · **target:** `{"domain": "discovery", "op": "fold:sdpa_fold_addmul"}`
- **lever:** Correctness decline, not pricing: the rewrite fires and changes the graph but fails verification on real cases — strengthen the law's guard/denotation before re-pricing.
- **evidence:**
  - source: `docs/reports/results_overnight_deep.json`
  - fires: 6
  - changed: 6
  - paid: 6
  - verify_fail: 4
  - exec_ratio: 0.21
  - cases: 57
  - corpus: "zoo-real"
  - arena: {"verdict": "known", "relation": "duplicate", "num_true": null, "base_cost": 7.0, "cand_cost": 7.0}

### 14. Verify-declined law: syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_10@RMSNormPerHead (changed=5, vfail=4)

- **signal:** `ledger_verify_decline` · **target:** `{"domain": "discovery", "op": "syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_10@RMSNormPerHead"}`
- **lever:** Correctness decline, not pricing: the rewrite fires and changes the graph but fails verification on real cases — strengthen the law's guard/denotation before re-pricing.
- **evidence:**
  - source: `docs/reports/results_overnight_deep.json`
  - fires: 5
  - changed: 5
  - paid: 4
  - verify_fail: 4
  - exec_ratio: 0.914
  - cases: 57
  - corpus: "zoo-real"
  - arena: {"verdict": "false", "relation": "new", "num_true": null, "base_cost": 9.0, "cand_cost": 9.0}

### 15. Verify-declined law: syn_naturality_scalar_rev__slice_mul_0@DilatedBlockAttention (changed=4, vfail=4)

- **signal:** `ledger_verify_decline` · **target:** `{"domain": "discovery", "op": "syn_naturality_scalar_rev__slice_mul_0@DilatedBlockAttention"}`
- **lever:** Correctness decline, not pricing: the rewrite fires and changes the graph but fails verification on real cases — strengthen the law's guard/denotation before re-pricing.
- **evidence:**
  - source: `docs/reports/results_overnight_deep.json`
  - fires: 5
  - changed: 4
  - paid: 4
  - verify_fail: 4
  - exec_ratio: 0.931
  - cases: 57
  - corpus: "zoo-real"
  - arena: {"verdict": "false", "relation": "new", "num_true": null, "base_cost": Infinity, "cand_cost": Infinity}

### 16. Verify-declined law: syn_naturality_scalar_rev__slice_mul_1@DilatedBlockAttention (changed=5, vfail=4)

- **signal:** `ledger_verify_decline` · **target:** `{"domain": "discovery", "op": "syn_naturality_scalar_rev__slice_mul_1@DilatedBlockAttention"}`
- **lever:** Correctness decline, not pricing: the rewrite fires and changes the graph but fails verification on real cases — strengthen the law's guard/denotation before re-pricing.
- **evidence:**
  - source: `docs/reports/results_overnight_deep.json`
  - fires: 5
  - changed: 5
  - paid: 4
  - verify_fail: 4
  - exec_ratio: 0.979
  - cases: 57
  - corpus: "zoo-real"
  - arena: {"verdict": "false", "relation": "new", "num_true": null, "base_cost": Infinity, "cand_cost": Infinity}

### 17. 27 declared (op, attr) pairs never occur in the stream

- **signal:** `coverage_attrs` · **target:** `{"domain": "meta-arena", "tier": "corpus"}`
- **lever:** Attr-carrying case kinds: the zoo census found zero attr elementwise ops — decide whether attrs need synthetic boards or real-workload intake first.
- **evidence:**
  - source: `corpus_cov.coverage_gaps (live census)`
  - cases: 800
  - missing: 27
  - total: 91
  - covered: 64
  - missing_sample: ["argmax.dim", "argmax.keepdim", "conv2d.dilation", "conv2d.groups", "isclose.atol", "isclose.equal_nan", "isclose.rtol", "linalg_vector_norm.ord", "max.dim", "max.keepdim", "nanmean.dim", "nanmean.keepdim", "prod.dim", "prod.keepdim", "sdpa.attn_mask", "sdpa.enable_gqa", "sdpa.scale", "softplus.bet…

### 18. Pay-declined law near the bar: reshape_reshape (changed=4, exec=0.5x)

- **signal:** `ledger_pay_decline` · **target:** `{"domain": "discovery", "op": "reshape_reshape"}`
- **lever:** Re-price or re-codegen this law's payoff: it fires and rewrites the graph but the measured kernel doesn't clear the keep bar — check the cost model's exec term against a hand-tuned kernel for the same site.
- **evidence:**
  - source: `docs/reports/results_overnight_real_preview.json`
  - fires: 4
  - changed: 4
  - paid: 0
  - exec_ratio: 0.5
  - count_ratio: null
  - flops_ratio: null
  - launch_ratio: null
  - cases: 40
  - corpus: "zoo-real"
  - arena: {"verdict": "false", "relation": "new", "num_true": null, "base_cost": 0.0, "cand_cost": 0.0}

### 19. 35 expressible ops never occur in the 800-case stream

- **signal:** `coverage_ops` · **target:** `{"domain": "meta-arena", "tier": "corpus"}`
- **lever:** Extend coverage_builders' templates to the missing op classes (synop*/new classes need new templates — the honest extension seam named in corpus_cov).
- **evidence:**
  - source: `corpus_cov.coverage_gaps (live census)`
  - cases: 800
  - missing: 35
  - total: 169
  - covered: 74
  - missing_sample: ["addcdiv", "addcmul", "argmax", "atan2", "bitwise_and", "bitwise_or", "eq", "fmax", "fmin", "fmod", "ge", "gelu", "gt", "heaviside", "isclose", "le", "lerp", "logical_and", "logical_not", "logical_or"]

### 20. 0/156 shipped rules never arm under the stream's rulesets

- **signal:** `coverage_rules` · **target:** `{"domain": "meta-arena", "tier": "corpus"}`
- **lever:** Unarmed rules need boards whose rulesets carry them AND terms matching their LHS — the fire/saturate payoff dimension stays unmeasured until then.
- **evidence:**
  - source: `corpus_cov.coverage_gaps (live census)`
  - cases: 800
  - missing: 0
  - total: 156
  - covered: 120
  - missing_sample: []

## Ledger ingest (human approval required)

Each proposal maps to one `myco todo add` — run the ones you accept (`--no-commit` leaves the commit to `myco check --fix` / your own commit):

```sh
myco todo add "Uncaptured headroom on zoo:ChunkedSSD: 1445.441us ceiling, captured=-0.0695, vs_compile=0.2548" --pri high --label agenda --no-commit
myco todo add "Uncaptured headroom on zoo:ChunkedSSD: 1465.21us ceiling, captured=0.0025, vs_compile=0.2981" --pri high --label agenda --no-commit
myco todo add "Uncaptured headroom on zoo:GatedDeltaRule: 242.786us ceiling, captured=0.1461, vs_compile=1.3175" --pri high --label agenda --no-commit
myco todo add "Uncaptured headroom on zoo:GatedLinearAttention: 1196.296us ceiling, captured=-0.5062, vs_compile=0.1756" --pri high --label agenda --no-commit
myco todo add "Uncaptured headroom on zoo:MambaSelectiveScan: 1448.219us ceiling, captured=-0.1543, vs_compile=0.1308" --pri high --label agenda --no-commit
myco todo add "Pay-declined law near the bar: grammar:sub_add_factor (changed=4, exec=1.0x)" --pri high --label agenda --no-commit
myco todo add "Uncaptured headroom on zoo:DecoupledMLA: 132.44us ceiling, captured=-0.0197, vs_compile=0.8042" --pri high --label agenda --no-commit
myco todo add "17/68 claim-bearing sweep rows straddle parity — referee noise decides keep/drop" --pri high --label agenda --no-commit
myco todo add "Uncaptured headroom on zoo:CapsuleRouting: 140.666us ceiling, captured=0.0687, vs_compile=0.4235" --pri high --label agenda --no-commit
myco todo add "53/68 claim-bearing delivered+verified rows run below parity (worst zoo:GatedLinearAttention at 0.287x)" --pri high --label agenda --no-commit
myco todo add "Uncaptured headroom on zoo:RotaryPair: 224.173us ceiling, captured=0.4979, vs_compile=1.1796" --pri med --label agenda --no-commit
myco todo add "Oracle never sees an equal site in family layout (12/13 died at oracle, 0 eq)" --pri med --label agenda --no-commit
myco todo add "Verify-declined law: fold:sdpa_fold_addmul (changed=6, vfail=4)" --pri med --label agenda --no-commit
myco todo add "Verify-declined law: syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_10@RMSNormPerHead (changed=5, vfail=4)" --pri med --label agenda --no-commit
myco todo add "Verify-declined law: syn_naturality_scalar_rev__slice_mul_0@DilatedBlockAttention (changed=4, vfail=4)" --pri med --label agenda --no-commit
myco todo add "Verify-declined law: syn_naturality_scalar_rev__slice_mul_1@DilatedBlockAttention (changed=5, vfail=4)" --pri med --label agenda --no-commit
myco todo add "27 declared (op, attr) pairs never occur in the stream" --pri med --label agenda --no-commit
myco todo add "Pay-declined law near the bar: reshape_reshape (changed=4, exec=0.5x)" --pri med --label agenda,low-confidence --no-commit
myco todo add "35 expressible ops never occur in the 800-case stream" --pri med --label agenda --no-commit
myco todo add "0/156 shipped rules never arm under the stream's rulesets" --pri med --label agenda --no-commit
```

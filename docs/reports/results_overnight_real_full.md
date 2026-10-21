# Overnight discovery run

Episodes -> mint -> admit -> classify.  Open-game episodes mint candidate rewrites; each minted claim is judged by the ``bench/referees`` gauntlet (oracle -> pay -> sweep -> cert); admitted laws are classified against the prior-art bank: **machine-original** (the widest shipped bank cannot derive it), **known-art** (an alpha-sibling exists or the bank proves it), **vacuous** (no binding produced a site — unevaluable).

- date: 2026-10-21T22:27:46+00:00 · git: `04201f2` (dirty) · device cpu
- protocol: `{"budget": 48, "compose": 2, "corpus": "zoo-real", "dry": false, "episodes": 120, "mint": "corpus-bind-compose", "page": 48, "player": "playbook", "referee": "full", "scan": 2048, "seed": 0}`
- corpus: `zoo-real` — held-out zoo modules (mlopt_discovery.zoo), candidates minted from each module's own op-tuple census (pipeline.propose); real_terms are the corpus's distinct subterms
- episodes: 53 · evaluated: 418 · minted: 318 (244 composites) · admitted: 0 · wall: 114.4s
- arena verdicts: false=397, known=21

## Admission funnel

| first-failed leg | count |
|---|---|
| oracle | 310 |
| pay | 7 |
| cert | 1 |
| **admitted** | 0 |

## Novelty

| verdict | count |
|---|---|
| machine-original | 0 |
| known-art | 0 |
| vacuous | 0 |

## Machine-original finds

**None.** Every admitted law was already a theorem of the shipped bank — or nothing survived the gauntlet.

## Candidates

| candidate | arena | admitted | reason | novelty |
|---|---|---|---|---|
| `syn_mixed:mul_unsqueeze_unsqueeze_id__mixed:add_unsqueeze_unsqueeze_id_8@AdditiveAttention` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:add_unsqueeze_unsqueeze_id_2@AdditiveAttention` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wr__mixed:add_unsqueeze_unsqueeze_id_5@AdditiveAttention` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_id__mixed:add_unsqueeze_unsqueeze_wl_6@AdditiveAttention` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_id__mixed:add_unsqueeze_unsqueeze_wr_7@AdditiveAttention` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:mul_unsqueeze_unsqueeze_id` | false | no | oracle: real: 0eq/13ne/9rerr (22 acc, 0 dec, 22 envs) \| synth: 0eq/142ne/51rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:add_unsqueeze_unsqueeze_id` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/142ne/51rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:add_unsqueeze_unsqueeze_wl_0@AdditiveAttention` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:add_unsqueeze_unsqueeze_wr_1@AdditiveAttention` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wr__mixed:add_unsqueeze_unsqueeze_wl_3@AdditiveAttention` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wr__mixed:add_unsqueeze_unsqueeze_wr_4@AdditiveAttention` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:mul_unsqueeze_unsqueeze_wl` | false | no | oracle: real: 4eq/9ne/9rerr (22 acc, 0 dec, 22 envs) \| synth: 63eq/79ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_unsqueeze_wr` | false | no | oracle: real: 5eq/8ne/9rerr (22 acc, 0 dec, 22 envs) \| synth: 65eq/77ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_unsqueeze_unsqueeze_wl` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 63eq/79ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_unsqueeze_unsqueeze_wr` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 65eq/77ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `reshape_transpose` | false | no | oracle: real: 0eq/4ne/0rerr (4 acc, 0 dec, 4 envs) \| synth: 61eq/100ne/0rerr (161 acc, 0 dec, 161 envs) | — |
| `mixed:add_chunk_r_w` | false | no | oracle: real: 0eq/0ne/5rerr (5 acc, 0 dec, 5 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_r_id` | false | no | oracle: real: 0eq/0ne/5rerr (5 acc, 0 dec, 5 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_r_w` | false | no | oracle: real: 0eq/0ne/3rerr (3 acc, 0 dec, 3 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_r_id` | false | no | oracle: real: 0eq/0ne/3rerr (3 acc, 0 dec, 3 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:add_chunk_r_w__mixed:mul_chunk_r_w_0@FiLMHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_r_w__mixed:mul_chunk_r_id_1@FiLMHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_r_id__mixed:mul_chunk_r_w_2@FiLMHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_r_id__mixed:mul_chunk_r_id_3@FiLMHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:add_chunk_r_w__mixed:mul_chunk_r_w_0__mixed:add_chunk_r_w_4@FiLMHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:add_chunk_r_w__mixed:mul_chunk_r_w_0__mixed:add_chunk_r_id_5@FiLMHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:add_chunk_r_id__mixed:mul_chunk_r_w_2__mixed:add_chunk_r_w_6@FiLMHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:add_chunk_r_id__mixed:mul_chunk_r_w_2__mixed:add_chunk_r_id_7@FiLMHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:mul_chunk_l_w` | false | no | oracle: real: 0eq/0ne/5rerr (5 acc, 0 dec, 5 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_l_id` | false | no | oracle: real: 0eq/0ne/5rerr (5 acc, 0 dec, 5 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_l_w` | false | no | oracle: real: 1eq/0ne/3rerr (4 acc, 0 dec, 4 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_l_id` | false | no | oracle: real: 0eq/1ne/3rerr (4 acc, 0 dec, 4 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:mul_chunk_l_w__mixed:add_chunk_r_w_0@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_w__mixed:add_chunk_r_w_1@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_w__mixed:add_chunk_r_id_2@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_w__mixed:add_chunk_r_id_3@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_w__mixed:add_chunk_l_w_4@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_w__mixed:add_chunk_l_id_5@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_id__mixed:add_chunk_r_w_7@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_id__mixed:add_chunk_r_id_8@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_id__mixed:add_chunk_l_w_9@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_id__mixed:add_chunk_l_id_10@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_r_w__mixed:add_chunk_l_w_12@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_r_w__mixed:add_chunk_l_id_13@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_r_id__mixed:add_chunk_l_w_15@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_r_id__mixed:add_chunk_l_id_16@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_w__syn_mixed:add_chunk_r_w__mixed:add_chunk_l_w_12_7@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_w__syn_mixed:add_chunk_r_w__mixed:add_chunk_l_id_13_8@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_w__syn_mixed:add_chunk_r_id__mixed:add_chunk_l_w_15_10@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_w__syn_mixed:add_chunk_r_id__mixed:add_chunk_l_id_16_11@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_id__syn_mixed:add_chunk_r_w__mixed:add_chunk_l_w_12_19@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_id__syn_mixed:add_chunk_r_w__mixed:add_chunk_l_id_13_20@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_id__syn_mixed:add_chunk_r_id__mixed:add_chunk_l_w_15_22@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_id__syn_mixed:add_chunk_r_id__mixed:add_chunk_l_id_16_23@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_chunk_l_w__mixed:add_chunk_r_w_0__mixed:add_chunk_r_w_35@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_chunk_l_w__mixed:add_chunk_r_w_0__mixed:add_chunk_r_id_36@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_chunk_l_w__mixed:add_chunk_r_w_0__mixed:add_chunk_l_w_37@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_chunk_l_w__mixed:add_chunk_r_w_0__mixed:add_chunk_l_id_38@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_chunk_l_w__mixed:add_chunk_r_w_0__syn_mixed:add_chunk_r_w__mixed:add_chunk_l_w_12_40@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_chunk_l_w__mixed:add_chunk_r_w_0__syn_mixed:add_chunk_r_w__mixed:add_chunk_l_id_13_41@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_chunk_l_w__mixed:add_chunk_r_w_0__syn_mixed:add_chunk_r_id__mixed:add_chunk_l_w_15_43@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_chunk_l_w__mixed:add_chunk_r_w_0__syn_mixed:add_chunk_r_id__mixed:add_chunk_l_id_16_44@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_chunk_l_w__mixed:add_chunk_r_w_1__mixed:add_chunk_r_id_47@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `grammar:mul_distribute` | false | no | pay: fires=49 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `syn_mixed:mul_chunk_l_w__grammar:mul_distribute_6@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_id__grammar:mul_distribute_11@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_r_w__grammar:mul_distribute_14@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_r_id__grammar:mul_distribute_17@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_id__grammar:mul_distribute_18@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_w__syn_mixed:add_chunk_r_w__grammar:mul_distribute_14_9@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_w__syn_mixed:add_chunk_r_id__grammar:mul_distribute_17_12@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_w__syn_mixed:add_chunk_l_id__grammar:mul_distribute_18_13@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_id__syn_mixed:add_chunk_r_w__grammar:mul_distribute_14_21@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_chunk_l_id__syn_mixed:add_chunk_r_id__grammar:mul_distribute_17_24@AdaLNBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:add_transpose_r_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 103eq/33ne/45rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_transpose_r_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 55eq/81ne/45rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_slice_unsqueeze_id` | false | no | oracle: real: 0eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 9eq/119ne/71rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_slice_unsqueeze_wl` | false | no | oracle: real: 0eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 22eq/106ne/71rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_slice_unsqueeze_wr` | false | no | oracle: real: 0eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 22eq/106ne/71rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:mul_unsqueeze_r_id__mixed:mul_unsqueeze_l_id_3@GCNLayer` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 1eq/6ne/1rerr (15 acc, 0 dec, 15 envs) | — |
| `syn_syn_mixed:mul_unsqueeze_r_id__mixed:mul_unsqueeze_l_w_2__mixed:mul_unsqueeze_r_id_9@GCNLayer` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 1eq/6ne/1rerr (15 acc, 0 dec, 15 envs) | — |
| `mixed:mul_unsqueeze_r_id` | false | no | oracle: real: 3eq/4ne/28rerr (36 acc, 0 dec, 36 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_l_id` | known | no | oracle: real: 1eq/8ne/27rerr (36 acc, 0 dec, 36 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:mul_unsqueeze_r_w__mixed:mul_unsqueeze_l_id_1@GCNLayer` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/7ne/1rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_r_id__mixed:mul_unsqueeze_l_w_2@GCNLayer` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/7ne/1rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_l_w__mixed:mul_unsqueeze_l_id_5@GCNLayer` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 1eq/6ne/1rerr (15 acc, 0 dec, 15 envs) | — |
| `syn_syn_mixed:mul_unsqueeze_r_w__mixed:mul_unsqueeze_l_w_0__mixed:mul_unsqueeze_r_id_7@GCNLayer` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/7ne/1rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_unsqueeze_r_id__mixed:mul_unsqueeze_l_w_2__mixed:mul_unsqueeze_r_w_8@GCNLayer` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/7ne/1rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_unsqueeze_l_w__mixed:mul_unsqueeze_l_w_4__mixed:mul_unsqueeze_r_id_11@GCNLayer` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/7ne/1rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `mixed:mul_unsqueeze_r_w` | false | no | oracle: real: 0eq/7ne/28rerr (36 acc, 0 dec, 36 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_l_w` | false | no | oracle: real: 0eq/9ne/27rerr (36 acc, 0 dec, 36 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:mul_unsqueeze_r_w__mixed:mul_unsqueeze_l_w_0@GCNLayer` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/7ne/1rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_l_w__mixed:mul_unsqueeze_l_w_4@GCNLayer` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/7ne/1rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_unsqueeze_r_w__mixed:mul_unsqueeze_l_w_0__mixed:mul_unsqueeze_r_w_6@GCNLayer` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/7ne/1rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_unsqueeze_l_w__mixed:mul_unsqueeze_l_w_4__mixed:mul_unsqueeze_r_w_10@GCNLayer` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/7ne/1rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_r_w__mixed:mul_chunk_l_w_0@AffineCoupling` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_r_w__mixed:mul_chunk_l_id_1@AffineCoupling` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_r_id__mixed:mul_chunk_l_w_2@AffineCoupling` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_r_id__mixed:mul_chunk_l_id_3@AffineCoupling` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:add_chunk_r_w__mixed:mul_chunk_l_w_0__mixed:add_chunk_r_w_4@AffineCoupling` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:add_chunk_r_w__mixed:mul_chunk_l_w_0__mixed:add_chunk_r_id_5@AffineCoupling` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:add_chunk_r_id__mixed:mul_chunk_l_w_2__mixed:add_chunk_r_id_6@AffineCoupling` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:sub_unsqueeze_unsqueeze_id` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/142ne/51rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:sub_unsqueeze_unsqueeze_wl` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 63eq/79ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_unsqueeze_wr` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 65eq/77ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:sub_unsqueeze_unsqueeze_wl__grammar:sub_to_add_dup_3@ChunkedRetention` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 63eq/79ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:sub_unsqueeze_unsqueeze_wr__grammar:sub_to_add_dup_4@ChunkedRetention` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 65eq/77ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `grammar:sub_to_add_dup` | known | no | pay: fires=22 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `census:mul_unsqueeze` | known | no | oracle: real: 4eq/0ne/0rerr (4 acc, 0 dec, 4 envs) \| synth: 69eq/8ne/8rerr (174 acc, 0 dec, 174 envs) | — |
| `mixed:sub_select_l_id` | false | no | oracle: real: 0eq/0ne/4rerr (4 acc, 0 dec, 4 envs) \| synth: 0eq/201ne/69rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:sub_select_l_w` | false | no | oracle: real: 0eq/0ne/4rerr (4 acc, 0 dec, 4 envs) \| synth: 89eq/110ne/71rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_unsqueeze_r_id` | false | no | oracle: real: 3eq/0ne/2rerr (5 acc, 0 dec, 5 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_unsqueeze_r_w` | false | no | oracle: real: 0eq/3ne/2rerr (5 acc, 0 dec, 5 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:mul_select_l_id_1@MambaSelectiveScan` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/6ne/8rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:mul_select_l_id_6@MambaSelectiveScan` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/6ne/7rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:mul_select_l_id_14@MambaSelectiveScan` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/2ne/4rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:mul_select_l_id_25@MambaSelectiveScan` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:mul_select_l_id_39@MambaSelectiveScan` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:mul_select_l_id_56@MambaSelectiveScan` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:mul_select_l_id_76@MambaSelectiveScan` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:mul_select_l_id` | false | no | oracle: real: 0eq/0ne/8rerr (8 acc, 0 dec, 8 envs) \| synth: 0eq/201ne/69rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:mul_select_l_w_0@MambaSelectiveScan` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 2eq/4ne/8rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:mul_select_l_w_5@MambaSelectiveScan` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 2eq/4ne/7rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:mul_select_l_w_13@MambaSelectiveScan` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 2eq/0ne/4rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:mul_select_l_w_24@MambaSelectiveScan` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:mul_select_l_w_38@MambaSelectiveScan` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:mul_select_l_w_55@MambaSelectiveScan` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_unsqueeze_wl__mixed:mul_select_l_w_75@MambaSelectiveScan` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:mul_select_l_w` | false | no | oracle: real: 0eq/0ne/8rerr (8 acc, 0 dec, 8 envs) \| synth: 89eq/110ne/71rerr (360 acc, 0 dec, 360 envs) | — |
| `mul_neg_left` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `mixed:eq_unsqueeze_unsqueeze_id` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/142ne/51rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:eq_unsqueeze_unsqueeze_wl` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 63eq/79ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:eq_unsqueeze_unsqueeze_wr` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 65eq/77ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `grammar:exp_distribute` | false | no | cert: not derivable (fixed_point) | — |
| `grammar:FALSE_exp_add` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `mixed:mul_chunk_transpose_wl` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 40eq/16ne/56rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_transpose_wr` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 25eq/31ne/56rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_transpose_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 27eq/29ne/56rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_l_id` | known | no | oracle: real: 1eq/3ne/0rerr (4 acc, 0 dec, 4 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_l_w` | false | no | oracle: real: 2eq/2ne/0rerr (4 acc, 0 dec, 4 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `slice_mul` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 80eq/38ne/39rerr (294 acc, 0 dec, 294 envs) | — |
| `mixed:mul_reshape_r_w` | false | no | oracle: real: 0eq/3ne/5rerr (8 acc, 0 dec, 8 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_r_id` | false | no | oracle: real: 4eq/3ne/1rerr (8 acc, 0 dec, 8 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `census:add_chunk` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 76eq/2ne/18rerr (204 acc, 0 dec, 204 envs) | — |
| `mixed:add_chunk_chunk_wl` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 66eq/33ne/30rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_chunk_wr` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 65eq/34ne/30rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_chunk_id` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 47eq/52ne/30rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_census:add_chunk__mixed:add_chunk_l_w_0@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_census:add_chunk__mixed:add_chunk_l_id_1@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_census:add_chunk__mixed:mul_chunk_r_w_2@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_census:add_chunk__mixed:mul_chunk_r_id_3@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_chunk_id__mixed:add_chunk_l_w_4@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_chunk_id__mixed:add_chunk_l_id_5@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_w__mixed:add_chunk_l_w_8@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_w__mixed:add_chunk_l_id_9@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_w__mixed:mul_chunk_r_w_10@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_w__mixed:mul_chunk_r_id_11@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_w__mixed:add_chunk_l_id_12@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_w__mixed:mul_chunk_r_w_13@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_w__mixed:mul_chunk_r_id_14@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_id__mixed:add_chunk_l_id_15@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_id__mixed:mul_chunk_r_w_16@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_id__mixed:mul_chunk_r_id_17@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_id__mixed:mul_chunk_r_w_18@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_id__mixed:mul_chunk_r_id_19@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_w__syn_census:add_chunk__mixed:mul_chunk_r_w_2_12@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_w__syn_census:add_chunk__mixed:mul_chunk_r_id_3_13@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_w__syn_mixed:add_chunk_l_w__mixed:mul_chunk_r_w_13_14@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_w__syn_mixed:add_chunk_l_w__mixed:mul_chunk_r_id_14_15@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_w__syn_mixed:add_chunk_l_id__mixed:mul_chunk_r_w_18_16@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_w__syn_mixed:add_chunk_l_id__mixed:mul_chunk_r_id_19_17@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_id__syn_census:add_chunk__mixed:mul_chunk_r_w_2_24@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_id__syn_census:add_chunk__mixed:mul_chunk_r_id_3_25@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_id__syn_mixed:add_chunk_l_w__mixed:mul_chunk_r_w_13_26@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_id__syn_mixed:add_chunk_l_w__mixed:mul_chunk_r_id_14_27@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_id__syn_mixed:add_chunk_l_id__mixed:mul_chunk_r_w_18_28@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_chunk_l_id__syn_mixed:add_chunk_l_id__mixed:mul_chunk_r_id_19_29@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:add_chunk_chunk_id__mixed:add_chunk_l_w_4__mixed:mul_chunk_r_w_32@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:add_chunk_chunk_id__mixed:add_chunk_l_id_5__mixed:mul_chunk_r_w_34@GRUFusedGate` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:add_reshape_r_w` | false | no | oracle: real: 0eq/1ne/2rerr (3 acc, 0 dec, 3 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_reshape_r_id` | false | no | oracle: real: 1eq/2ne/0rerr (3 acc, 0 dec, 3 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_reshape_wl` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 22eq/24ne/86rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_reshape_wr` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 24eq/24ne/84rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_reshape_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 19eq/56ne/57rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_reshape_l_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_reshape_l_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:add_reshape_r_w__mixed:mul_reshape_reshape_wl_0@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:mul_reshape_reshape_wr_1@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:mul_reshape_reshape_id_2@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_id__mixed:mul_reshape_reshape_wl_12@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_id__mixed:mul_reshape_reshape_wr_13@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_id__mixed:mul_reshape_reshape_id_14@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:sub_reshape_l_w_3@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:sub_reshape_l_w_4@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:sub_reshape_l_w_5@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:sub_reshape_l_id_6@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:sub_reshape_l_id_7@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:sub_reshape_l_id_8@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_id__mixed:sub_reshape_l_w_15@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_id__mixed:sub_reshape_l_w_16@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_id__mixed:sub_reshape_l_w_17@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_id__mixed:sub_reshape_l_id_18@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_id__mixed:sub_reshape_l_id_19@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_id__mixed:sub_reshape_l_id_20@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_wl__mixed:sub_reshape_l_w_24@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_wl__mixed:sub_reshape_l_w_25@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_wl__mixed:sub_reshape_l_w_26@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_wl__mixed:sub_reshape_l_id_27@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_wl__mixed:sub_reshape_l_id_28@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_wl__mixed:sub_reshape_l_id_29@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_wr__mixed:sub_reshape_l_w_33@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_wr__mixed:sub_reshape_l_w_34@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_wr__mixed:sub_reshape_l_w_35@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_wr__mixed:sub_reshape_l_id_36@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_wr__mixed:sub_reshape_l_id_37@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_wr__mixed:sub_reshape_l_id_38@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_id__mixed:sub_reshape_l_w_42@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_id__mixed:sub_reshape_l_w_43@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_id__mixed:sub_reshape_l_w_44@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_id__mixed:sub_reshape_l_id_45@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_id__mixed:sub_reshape_l_id_46@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_id__mixed:sub_reshape_l_id_47@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__grammar:sub_to_add_dup_9@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__grammar:sub_to_add_dup_10@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__grammar:sub_to_add_dup_11@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_reshape_wl__grammar:sub_to_add_dup_30@ManualGroupNorm` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:add_unsqueeze_l_id` | false | no | oracle: real: 1eq/1ne/1rerr (3 acc, 0 dec, 3 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_r_id` | false | no | oracle: real: 0eq/3ne/0rerr (3 acc, 0 dec, 3 envs) \| synth: 0eq/178ne/39rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:add_unsqueeze_l_id_3@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__mixed:add_unsqueeze_l_id_27@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__syn_mixed:mul_reshape_r_w__mixed:add_unsqueeze_l_id_27_14@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_id__syn_mixed:mul_reshape_r_w__mixed:add_unsqueeze_l_id_27_41@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:add_unsqueeze_l_w` | false | no | oracle: real: 1eq/1ne/1rerr (3 acc, 0 dec, 3 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_r_w` | false | no | oracle: real: 0eq/3ne/0rerr (3 acc, 0 dec, 3 envs) \| synth: 0eq/178ne/39rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:mul_reshape_r_w_0@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:mul_reshape_r_id_1@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:add_unsqueeze_l_w_2@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_id__mixed:mul_reshape_r_w_13@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__mixed:add_unsqueeze_l_w_26@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_unsqueeze_l_w__mixed:add_unsqueeze_l_w_48@InstanceNormBroadcast` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `syn_mixed:add_reshape_r_w__syn_mixed:mul_reshape_r_w__mixed:add_unsqueeze_l_w_26_13@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__syn_mixed:add_unsqueeze_l_w__mixed:add_unsqueeze_l_w_48_24@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_id__syn_mixed:mul_reshape_r_w__mixed:add_unsqueeze_l_w_26_40@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:sub_unsqueeze_r_id_7@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:sub_unsqueeze_r_id_8@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:sub_unsqueeze_r_id_9@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__mixed:sub_unsqueeze_r_id_31@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__mixed:sub_unsqueeze_r_id_32@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__mixed:sub_unsqueeze_r_id_33@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__syn_mixed:mul_reshape_r_w__mixed:sub_unsqueeze_r_id_31_18@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__syn_mixed:mul_reshape_r_w__mixed:sub_unsqueeze_r_id_32_19@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__syn_mixed:mul_reshape_r_w__mixed:sub_unsqueeze_r_id_33_20@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_id__syn_mixed:mul_reshape_r_w__mixed:sub_unsqueeze_r_id_31_45@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:sub_unsqueeze_r_w_4@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:sub_unsqueeze_r_w_5@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:sub_unsqueeze_r_w_6@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__grammar:sub_to_add_dup_10@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__grammar:sub_to_add_dup_11@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__grammar:sub_to_add_dup_12@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__mixed:sub_unsqueeze_r_w_28@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__mixed:sub_unsqueeze_r_w_29@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__mixed:sub_unsqueeze_r_w_30@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__grammar:sub_to_add_dup_34@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__grammar:sub_to_add_dup_35@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__grammar:sub_to_add_dup_36@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_unsqueeze_l_w__grammar:sub_to_add_dup_54@InstanceNormBroadcast` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `syn_mixed:add_unsqueeze_l_w__grammar:sub_to_add_dup_55@InstanceNormBroadcast` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `syn_mixed:add_reshape_r_w__syn_mixed:mul_reshape_r_w__mixed:sub_unsqueeze_r_w_28_15@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__syn_mixed:mul_reshape_r_w__mixed:sub_unsqueeze_r_w_29_16@InstanceNormBroadcast` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:mul_reshape_l_w` | false | no | oracle: real: 0eq/0ne/3rerr (3 acc, 0 dec, 3 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_l_id` | known | no | oracle: real: 0eq/1ne/2rerr (3 acc, 0 dec, 3 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:pow_reshape_l_w` | false | no | oracle: real: 1eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 155eq/14ne/73rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:pow_reshape_l_id` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 73eq/126ne/43rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:mul_reshape_r_w__mixed:mul_reshape_l_w_0@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__mixed:mul_reshape_l_id_1@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__mixed:pow_reshape_l_w_2@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__mixed:pow_reshape_l_id_3@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_id__mixed:mul_reshape_l_w_4@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_id__mixed:mul_reshape_l_id_5@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_6@RMSNormPerHead` | false | no | pay: fires=5 changed=5 paid=4 verify_fail=4 cases=57 \| count=1.000x exec=0.914x flops=0.900x launch=0.900x params=1.000x | — |
| `syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_id_7@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_l_w__mixed:mul_reshape_l_w_8@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_l_w__mixed:mul_reshape_l_id_9@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_l_w__mixed:pow_reshape_l_w_10@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_l_id__mixed:pow_reshape_l_w_12@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__syn_mixed:mul_reshape_l_w__mixed:pow_reshape_l_w_10_4@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_w__syn_mixed:mul_reshape_l_id__mixed:pow_reshape_l_w_12_5@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_id__syn_mixed:mul_reshape_l_w__mixed:pow_reshape_l_w_10_10@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:mul_reshape_r_id__syn_mixed:mul_reshape_l_id__mixed:pow_reshape_l_w_12_11@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_reshape_r_w__mixed:mul_reshape_l_w_0__mixed:mul_reshape_r_w_18@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_reshape_r_w__mixed:mul_reshape_l_w_0__mixed:mul_reshape_r_id_19@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_reshape_r_w__mixed:mul_reshape_l_w_0__mixed:pow_reshape_l_id_20@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_reshape_r_w__mixed:mul_reshape_l_id_1__mixed:pow_reshape_l_id_21@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_reshape_r_id__mixed:mul_reshape_l_w_4__mixed:mul_reshape_r_w_22@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_reshape_r_id__mixed:mul_reshape_l_w_4__mixed:mul_reshape_r_id_23@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_reshape_r_id__mixed:mul_reshape_l_id_5__mixed:pow_reshape_l_id_25@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_reshape_l_w__mixed:mul_reshape_l_w_8__mixed:mul_reshape_r_w_26@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_reshape_l_w__mixed:mul_reshape_l_w_8__mixed:mul_reshape_r_id_27@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_reshape_l_w__mixed:mul_reshape_l_w_8__mixed:pow_reshape_l_w_28@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_reshape_l_w__mixed:mul_reshape_l_w_8__mixed:pow_reshape_l_id_29@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_reshape_l_w__mixed:mul_reshape_l_id_9__mixed:pow_reshape_l_w_30@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_syn_mixed:mul_reshape_l_w__mixed:mul_reshape_l_id_9__mixed:pow_reshape_l_id_31@RMSNormPerHead` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_reshape_r_w__mixed:mul_reshape_r_w_0@DynamicTanhBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 21eq/1ne/35rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:add_reshape_r_w__mixed:mul_reshape_r_id_1@DynamicTanhBlock` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 25eq/6ne/26rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:add_reshape_r_id__mixed:mul_reshape_r_w_2@DynamicTanhBlock` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 21eq/1ne/35rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:add_reshape_r_id__mixed:mul_reshape_r_id_3@DynamicTanhBlock` | false | no | oracle: real: 1eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/31ne/26rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_syn_mixed:add_reshape_r_w__mixed:mul_reshape_r_w_0__mixed:add_reshape_r_w_4@DynamicTanhBlock` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 25eq/6ne/26rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_syn_mixed:add_reshape_r_w__mixed:mul_reshape_r_w_0__mixed:add_reshape_r_id_5@DynamicTanhBlock` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 25eq/6ne/26rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_syn_mixed:add_reshape_r_id__mixed:mul_reshape_r_w_2__mixed:add_reshape_r_id_6@DynamicTanhBlock` | false | no | oracle: real: 1eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/31ne/26rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_unsqueeze_wl` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 8eq/45ne/125rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_unsqueeze_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 6eq/96ne/76rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_unsqueeze_wr` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 33eq/69ne/76rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:sub_unsqueeze_unsqueeze_wl__grammar:sub_to_add_dup_11@RelPosTableAttention` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 63eq/79ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:sub_unsqueeze_unsqueeze_wr__grammar:sub_to_add_dup_12@RelPosTableAttention` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 65eq/77ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_slice_r_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 79eq/125ne/34rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_slice_r_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 135eq/69ne/34rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:mul_unsqueeze_r_id__reshape_transpose_1@GroupedQuantLinear` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (105 acc, 0 dec, 105 envs) — no equal site measured | — |
| `syn_mixed:mul_unsqueeze_r_w__reshape_transpose_0@GroupedQuantLinear` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/0ne/0rerr (105 acc, 0 dec, 105 envs) — no equal site measured | — |

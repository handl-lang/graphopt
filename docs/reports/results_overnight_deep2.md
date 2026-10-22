# Overnight discovery run

Episodes -> mint -> admit -> classify.  Open-game episodes mint candidate rewrites; each minted claim is judged by the ``bench/referees`` gauntlet (oracle -> pay -> sweep -> cert); admitted laws are classified against the prior-art bank: **machine-original** (the widest shipped bank cannot derive it), **known-art** (an alpha-sibling exists or the bank proves it), **vacuous** (no binding produced a site — unevaluable).

- date: 2026-10-22T05:32:38+00:00 · git: `605a4db` (dirty) · device cpu
- protocol: `{"budget": 96, "compose": 6, "corpus": "zoo-real", "dry": false, "episodes": 53, "fusion": true, "mint": "corpus-bind-compose-strata", "page": 48, "player": "playbook", "referee": "full", "scan": 2048, "seed": 0}`
- corpus: `zoo-real` — held-out zoo modules (mlopt_discovery.zoo), candidates minted from each module's own op-tuple census (pipeline.propose); real_terms are the corpus's distinct subterms
- episodes: 53 · evaluated: 206 · minted: 104 (22 composites, 10 guarded) · admitted: 3 · wall: 84.2s
- arena verdicts: false=179, known=27

## Admission funnel

| first-failed leg | count |
|---|---|
| oracle | 75 |
| pay | 25 |
| cert | 1 |
| **admitted** | 3 |

## Novelty

| verdict | count |
|---|---|
| machine-original | 0 |
| known-art | 3 |
| vacuous | 0 |

## Machine-original finds

**None.** Every admitted law was already a theorem of the shipped bank — or nothing survived the gauntlet.

## Candidates

| candidate | arena | admitted | reason | novelty |
|---|---|---|---|---|
| `mixed:mul_unsqueeze_unsqueeze_id` | false | no | oracle: real: 0eq/13ne/9rerr (22 acc, 0 dec, 22 envs) \| synth: 0eq/142ne/51rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:add_unsqueeze_unsqueeze_id` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/142ne/51rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:mul_unsqueeze_unsqueeze_wl` | false | no | oracle: real: 4eq/9ne/9rerr (22 acc, 0 dec, 22 envs) \| synth: 63eq/79ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_unsqueeze_wr` | false | no | oracle: real: 5eq/8ne/9rerr (22 acc, 0 dec, 22 envs) \| synth: 65eq/77ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_unsqueeze_unsqueeze_wl` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 63eq/79ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_unsqueeze_unsqueeze_wr` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 65eq/77ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `reshape_transpose` | false | no | oracle: real: 0eq/4ne/0rerr (4 acc, 0 dec, 4 envs) \| synth: 61eq/100ne/0rerr (161 acc, 0 dec, 161 envs) | — |
| `mixed:add_chunk_r_w` | false | no | oracle: real: 0eq/0ne/5rerr (5 acc, 0 dec, 5 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_r_id` | false | no | oracle: real: 0eq/0ne/5rerr (5 acc, 0 dec, 5 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_r_w` | false | no | oracle: real: 0eq/0ne/3rerr (3 acc, 0 dec, 3 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_r_id` | false | no | oracle: real: 0eq/0ne/3rerr (3 acc, 0 dec, 3 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_l_w` | false | no | oracle: real: 0eq/0ne/5rerr (5 acc, 0 dec, 5 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_l_id` | false | no | oracle: real: 0eq/0ne/5rerr (5 acc, 0 dec, 5 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_l_w` | false | no | oracle: real: 1eq/0ne/3rerr (4 acc, 0 dec, 4 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_l_id` | false | no | oracle: real: 0eq/1ne/3rerr (4 acc, 0 dec, 4 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `grammar:mul_distribute` | false | no | pay: fires=49 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `syn_linear_row_scale_rev__linear_row_scale_rev_0@LoRAAdapter` | false | no | pay: fires=3 changed=3 paid=0 verify_fail=2 cases=55 \| count=1.000x exec=1.000x flops=1.000x launch=1.000x params=1.000x | — |
| `mixed:add_transpose_r_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 103eq/33ne/45rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_transpose_r_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 55eq/81ne/45rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_slice_unsqueeze_id` | false | no | oracle: real: 0eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 9eq/119ne/71rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_slice_unsqueeze_wl` | false | no | oracle: real: 0eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 22eq/106ne/71rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_slice_unsqueeze_wr` | false | no | oracle: real: 0eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 22eq/106ne/71rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_r_id` | false | no | oracle: real: 3eq/4ne/28rerr (36 acc, 0 dec, 36 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_l_id` | known | no | oracle: real: 1eq/8ne/27rerr (36 acc, 0 dec, 36 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_r_w` | false | no | oracle: real: 0eq/7ne/28rerr (36 acc, 0 dec, 36 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_l_w` | false | no | oracle: real: 0eq/9ne/27rerr (36 acc, 0 dec, 36 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
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
| `mixed:mul_select_l_id` | false | no | oracle: real: 0eq/0ne/8rerr (8 acc, 0 dec, 8 envs) \| synth: 0eq/201ne/69rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
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
| `fold:addmm_linear` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 47eq/0ne/20rerr (218 acc, 142 dec, 360 envs) | — |
| `fold:sdpa_fold_div_nomask` | known | no | pay: fires=5 changed=5 paid=5 verify_fail=1 cases=57 \| count=0.435x exec=0.417x flops=0.493x launch=0.491x params=1.000x | — |
| `mixed:sub_unsqueeze_l_id` | known | no | oracle: real: 1eq/3ne/0rerr (4 acc, 0 dec, 4 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_l_w` | false | no | oracle: real: 2eq/2ne/0rerr (4 acc, 0 dec, 4 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_sdpa_fold_mul_nomask__slice_mul_2@DilatedBlockAttention` | false | no | pay: fires=2 changed=0 paid=0 verify_fail=0 cases=54 | — |
| `syn_sdpa_fold_mul_nomask__slice_mul_3@DilatedBlockAttention` | false | no | pay: fires=2 changed=0 paid=0 verify_fail=0 cases=54 | — |
| `syn_slice_mul__sdpa_fold_mul_nomask_4@DilatedBlockAttention` | false | no | pay: fires=2 changed=0 paid=0 verify_fail=0 cases=54 | — |
| `syn_slice_mul__sdpa_fold_mul_nomask_5@DilatedBlockAttention` | false | no | pay: fires=2 changed=0 paid=0 verify_fail=0 cases=54 | — |
| `fold:sdpa_fold_mul_nomask` | known | yes | cleared | known-art |
| `slice_mul` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 80eq/38ne/39rerr (294 acc, 0 dec, 294 envs) | — |
| `syn_naturality_scalar_rev__slice_mul_0@DilatedBlockAttention` | false | no | pay: fires=5 changed=4 paid=4 verify_fail=4 cases=57 \| count=0.929x exec=0.931x flops=0.952x launch=0.943x | — |
| `syn_naturality_scalar_rev__slice_mul_1@DilatedBlockAttention` | false | no | pay: fires=5 changed=5 paid=4 verify_fail=4 cases=57 \| count=0.978x exec=0.979x flops=0.981x launch=0.973x params=1.000x | — |
| `mixed:mul_reshape_r_w` | false | no | oracle: real: 0eq/3ne/5rerr (8 acc, 0 dec, 8 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_r_id` | false | no | oracle: real: 4eq/3ne/1rerr (8 acc, 0 dec, 8 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `census:add_chunk` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 76eq/2ne/18rerr (204 acc, 0 dec, 204 envs) | — |
| `mixed:add_chunk_chunk_wl` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 66eq/33ne/30rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_chunk_wr` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 65eq/34ne/30rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_chunk_id` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 47eq/52ne/30rerr (360 acc, 0 dec, 360 envs) | — |
| `fold:glu_fold` | known | yes | cleared | known-art |
| `fold:group_norm` | false | no | pay: fires=2 changed=2 paid=2 verify_fail=1 cases=54 \| count=0.111x exec=0.100x flops=0.198x launch=0.198x params=1.000x | — |
| `mixed:add_reshape_r_w` | false | no | oracle: real: 0eq/1ne/2rerr (3 acc, 0 dec, 3 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_reshape_r_id` | false | no | oracle: real: 1eq/2ne/0rerr (3 acc, 0 dec, 3 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_reshape_wl` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 22eq/24ne/86rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_reshape_wr` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 24eq/24ne/84rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_reshape_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 19eq/56ne/57rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_reshape_l_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_reshape_l_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `fold:instance_norm` | false | no | pay: fires=2 changed=2 paid=2 verify_fail=1 cases=54 \| count=0.077x exec=0.077x flops=0.193x launch=0.193x params=1.000x | — |
| `mixed:add_unsqueeze_l_id` | false | no | oracle: real: 1eq/1ne/1rerr (3 acc, 0 dec, 3 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_r_id` | false | no | oracle: real: 0eq/3ne/0rerr (3 acc, 0 dec, 3 envs) \| synth: 0eq/178ne/39rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:add_unsqueeze_l_w` | false | no | oracle: real: 1eq/1ne/1rerr (3 acc, 0 dec, 3 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_r_w` | false | no | oracle: real: 0eq/3ne/0rerr (3 acc, 0 dec, 3 envs) \| synth: 0eq/178ne/39rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_unsqueeze_l_w__mixed:add_unsqueeze_l_w_48@InstanceNormBroadcast` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `syn_syn_mixed:add_unsqueeze_l_w__grammar:sub_to_add_dup_54@InstanceNormBroadcast__grammar:sub_to_add_dup_17@InstanceNormBroadcast` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `syn_syn_syn_mixed:add_unsqueeze_l_w__mixed:add_unsqueeze_l_w_48@InstanceNormBroadcast__grammar:sub_to_add_dup_8@InstanceNormBroadcast__grammar:sub_to_add_dup_6@InstanceNormBroadcast` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `syn_mixed:add_unsqueeze_l_w__grammar:sub_to_add_dup_54@InstanceNormBroadcast` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `syn_mixed:add_unsqueeze_l_w__grammar:sub_to_add_dup_55@InstanceNormBroadcast` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `syn_syn_mixed:add_unsqueeze_l_w__mixed:add_unsqueeze_l_w_48@InstanceNormBroadcast__grammar:sub_to_add_dup_8@InstanceNormBroadcast` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `syn_syn_mixed:add_unsqueeze_l_w__mixed:add_unsqueeze_l_w_48@InstanceNormBroadcast__grammar:sub_to_add_dup_9@InstanceNormBroadcast` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `fold:rms_norm_fold_nogain` | known | yes | cleared | known-art |
| `syn_rms_norm_fold_nogain__mixed:mul_reshape_r_id_1@RMSNormPerHead` | false | no | pay: fires=2 changed=2 paid=2 verify_fail=1 cases=54 \| count=0.333x exec=0.407x flops=0.381x launch=0.381x params=1.000x | — |
| `syn_mixed:mul_reshape_r_id__rms_norm_fold_nogain_7@RMSNormPerHead` | false | no | pay: fires=2 changed=2 paid=2 verify_fail=1 cases=54 \| count=0.333x exec=0.407x flops=0.381x launch=0.381x params=1.000x | — |
| `mixed:mul_reshape_l_w` | false | no | oracle: real: 0eq/0ne/3rerr (3 acc, 0 dec, 3 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_l_id` | known | no | oracle: real: 0eq/1ne/2rerr (3 acc, 0 dec, 3 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:pow_reshape_l_w` | false | no | oracle: real: 1eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 155eq/14ne/73rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:pow_reshape_l_id` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 73eq/126ne/43rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_10@RMSNormPerHead` | false | no | pay: fires=5 changed=5 paid=4 verify_fail=4 cases=57 \| count=1.000x exec=0.914x flops=0.900x launch=0.900x params=1.000x | — |
| `syn_mixed:add_reshape_r_w__mixed:mul_reshape_r_id_1@DynamicTanhBlock` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 25eq/6ne/26rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:add_reshape_r_id__mixed:mul_reshape_r_id_3@DynamicTanhBlock` | false | no | oracle: real: 1eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/31ne/26rerr (360 acc, 0 dec, 360 envs) | — |
| `fold:sdpa_fold_addmul` | known | no | pay: fires=6 changed=6 paid=6 verify_fail=4 cases=57 \| count=0.311x exec=0.210x flops=0.326x launch=0.326x params=1.000x | — |
| `mixed:mul_reshape_unsqueeze_wl` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 8eq/45ne/125rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_unsqueeze_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 6eq/96ne/76rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_unsqueeze_wr` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 33eq/69ne/76rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_slice_r_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 79eq/125ne/34rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_slice_r_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 135eq/69ne/34rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_naturality_scalar_rev__mixed:mul_reshape_r_id_1@PagedKVAttention` | false | no | pay: fires=5 changed=1 paid=0 verify_fail=1 cases=57 \| count=1.000x exec=1.000x flops=1.000x launch=1.000x | — |

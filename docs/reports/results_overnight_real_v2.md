# Overnight discovery run

Episodes -> mint -> admit -> classify.  Open-game episodes mint candidate rewrites; each minted claim is judged by the ``bench/referees`` gauntlet (oracle -> pay -> sweep -> cert); admitted laws are classified against the prior-art bank: **machine-original** (the widest shipped bank cannot derive it), **known-art** (an alpha-sibling exists or the bank proves it), **vacuous** (no binding produced a site — unevaluable).

- date: 2026-10-21T22:56:36+00:00 · git: `1000ffd` (dirty) · device cpu
- protocol: `{"budget": 48, "compose": 2, "corpus": "zoo-real", "dry": false, "episodes": 120, "mint": "corpus-bind-compose-viable", "page": 48, "player": "playbook", "referee": "full", "scan": 2048, "seed": 0}`
- corpus: `zoo-real` — held-out zoo modules (mlopt_discovery.zoo), candidates minted from each module's own op-tuple census (pipeline.propose); real_terms are the corpus's distinct subterms
- episodes: 53 · evaluated: 187 · minted: 87 (13 composites) · admitted: 0 · wall: 95.5s
- arena verdicts: false=166, known=21

## Admission funnel

| first-failed leg | count |
|---|---|
| oracle | 79 |
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
| `mixed:add_unsqueeze_r_id` | false | no | oracle: real: 2eq/1ne/2rerr (5 acc, 0 dec, 5 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
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
| `mixed:sub_unsqueeze_l_id` | known | no | oracle: real: 1eq/3ne/0rerr (4 acc, 0 dec, 4 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_l_w` | false | no | oracle: real: 2eq/2ne/0rerr (4 acc, 0 dec, 4 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `slice_mul` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 80eq/38ne/39rerr (294 acc, 0 dec, 294 envs) | — |
| `mixed:mul_reshape_r_w` | false | no | oracle: real: 0eq/3ne/5rerr (8 acc, 0 dec, 8 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_r_id` | false | no | oracle: real: 4eq/3ne/1rerr (8 acc, 0 dec, 8 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `census:add_chunk` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 76eq/2ne/18rerr (204 acc, 0 dec, 204 envs) | — |
| `mixed:add_chunk_chunk_wl` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 66eq/33ne/30rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_chunk_wr` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 65eq/34ne/30rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_chunk_id` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 47eq/52ne/30rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_reshape_r_w` | false | no | oracle: real: 0eq/1ne/2rerr (3 acc, 0 dec, 3 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_reshape_r_id` | false | no | oracle: real: 1eq/2ne/0rerr (3 acc, 0 dec, 3 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_reshape_wl` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 22eq/24ne/86rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_reshape_wr` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 24eq/24ne/84rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_reshape_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 19eq/56ne/57rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_reshape_l_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_reshape_l_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_unsqueeze_l_id` | false | no | oracle: real: 1eq/1ne/1rerr (3 acc, 0 dec, 3 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_r_id` | false | no | oracle: real: 0eq/3ne/0rerr (3 acc, 0 dec, 3 envs) \| synth: 0eq/178ne/39rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:add_unsqueeze_l_w` | false | no | oracle: real: 1eq/1ne/1rerr (3 acc, 0 dec, 3 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_r_w` | false | no | oracle: real: 0eq/3ne/0rerr (3 acc, 0 dec, 3 envs) \| synth: 0eq/178ne/39rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `syn_mixed:add_unsqueeze_l_w__mixed:add_unsqueeze_l_w_48@InstanceNormBroadcast` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `syn_mixed:add_unsqueeze_l_w__grammar:sub_to_add_dup_54@InstanceNormBroadcast` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `syn_mixed:add_unsqueeze_l_w__grammar:sub_to_add_dup_55@InstanceNormBroadcast` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `mixed:mul_reshape_l_w` | false | no | oracle: real: 0eq/0ne/3rerr (3 acc, 0 dec, 3 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_l_id` | known | no | oracle: real: 0eq/1ne/2rerr (3 acc, 0 dec, 3 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:pow_reshape_l_w` | false | no | oracle: real: 1eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 155eq/14ne/73rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:pow_reshape_l_id` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 73eq/126ne/43rerr (360 acc, 0 dec, 360 envs) | — |
| `syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_6@RMSNormPerHead` | false | no | pay: fires=5 changed=5 paid=4 verify_fail=4 cases=57 \| count=1.000x exec=0.914x flops=0.900x launch=0.900x params=1.000x | — |
| `syn_mixed:add_reshape_r_w__mixed:mul_reshape_r_id_1@DynamicTanhBlock` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 25eq/6ne/26rerr (360 acc, 0 dec, 360 envs) | — |
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

# Overnight discovery run

Episodes -> mint -> admit -> classify.  Open-game episodes mint candidate rewrites; each minted claim is judged by the ``bench/referees`` gauntlet (oracle -> pay -> sweep -> cert); admitted laws are classified against the prior-art bank: **machine-original** (the widest shipped bank cannot derive it), **known-art** (an alpha-sibling exists or the bank proves it), **vacuous** (no binding produced a site — unevaluable).

- date: 2026-10-21T21:38:03+00:00 · git: `911163f` (dirty) · device cpu
- protocol: `{"budget": 48, "corpus": "zoo-real", "dry": false, "episodes": 16, "page": 48, "player": "playbook", "referee": "full", "scan": 2048, "seed": 0}`
- corpus: `zoo-real` — held-out zoo modules (mlopt_discovery.zoo), candidates minted from each module's own op-tuple census (pipeline.propose); real_terms are the corpus's distinct subterms
- episodes: 16 · evaluated: 592 · minted: 55 · admitted: 2 · wall: 47.4s
- arena verdicts: false=477, known=83, true=32

## Admission funnel

| first-failed leg | count |
|---|---|
| oracle | 34 |
| pay | 11 |
| cert | 8 |
| **admitted** | 2 |

## Novelty

| verdict | count |
|---|---|
| machine-original | 0 |
| known-art | 2 |
| vacuous | 0 |

## Machine-original finds

**None.** Every admitted law was already a theorem of the shipped bank — or nothing survived the gauntlet.

## Candidates

| candidate | arena | admitted | reason | novelty |
|---|---|---|---|---|
| `grammar:div_self` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 3eq/14ne/0rerr (17 acc, 0 dec, 17 envs) | — |
| `linear_factor` | known | no | pay: fires=4 changed=4 paid=4 verify_fail=4 cases=20 \| count=0.667x exec=0.667x flops=0.600x launch=0.625x | — |
| `matmul_factor` | known | no | pay: fires=4 changed=4 paid=4 verify_fail=4 cases=20 \| count=0.667x exec=0.667x flops=0.600x launch=0.625x | — |
| `grammar:pow_one` | false | no | cert: not derivable (fixed_point) | — |
| `grammar:add_inv` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 2eq/13ne/0rerr (15 acc, 0 dec, 15 envs) | — |
| `factor_left` | false | no | cert: not derivable (fixed_point) | — |
| `factor_right` | false | no | cert: not derivable (fixed_point) | — |
| `factor_sub_left` | false | no | cert: not derivable (fixed_point) | — |
| `factor_sub_right` | false | no | cert: not derivable (fixed_point) | — |
| `select_mul` | known | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 63eq/20ne/32rerr (198 acc, 0 dec, 198 envs) | — |
| `select_add` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 63eq/20ne/32rerr (198 acc, 0 dec, 198 envs) | — |
| `select_sub` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 63eq/20ne/32rerr (198 acc, 0 dec, 198 envs) | — |
| `slice_mul` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 80eq/38ne/39rerr (294 acc, 0 dec, 294 envs) | — |
| `neg_add` | false | no | cert: not derivable (fixed_point) | — |
| `sub_neg` | true | yes | cleared | known-art |
| `exp_add` | false | no | cert: not derivable (fixed_point) | — |
| `square_neg` | false | no | cert: not derivable (fixed_point) | — |
| `mul_zero` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 2eq/13ne/0rerr (15 acc, 0 dec, 15 envs) | — |
| `id_mul_lit` | known | yes | cleared | known-art |
| `sub_self` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 2eq/13ne/0rerr (15 acc, 0 dec, 15 envs) | — |
| `grammar:mul_zero_left` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 2eq/13ne/0rerr (15 acc, 0 dec, 15 envs) | — |
| `grammar:FALSE_mul_factor` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 11eq/288ne/0rerr (360 acc, 0 dec, 360 envs) | — |
| `reshape_reshape` | false | no | pay: fires=4 changed=4 paid=0 verify_fail=0 cases=20 \| exec=0.500x | — |
| `reshape_transpose` | false | no | oracle: real: 0eq/3ne/0rerr (3 acc, 0 dec, 3 envs) \| synth: 61eq/100ne/0rerr (161 acc, 0 dec, 161 envs) | — |
| `mul_neg_left` | false | no | pay: fires=4 changed=0 paid=0 verify_fail=0 cases=20 | — |
| `grammar:sub_add_factor` | false | no | pay: fires=4 changed=4 paid=0 verify_fail=0 cases=20 \| count=1.000x exec=1.000x flops=1.000x launch=1.000x | — |
| `grammar:sigmoid_neg` | false | no | pay: fires=4 changed=0 paid=0 verify_fail=0 cases=20 | — |
| `grammar:mul_distribute` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=20 | — |
| `grammar:neg_distribute` | false | no | pay: fires=4 changed=0 paid=0 verify_fail=0 cases=20 | — |
| `grammar:exp_distribute` | false | no | pay: fires=4 changed=0 paid=0 verify_fail=0 cases=20 | — |
| `grammar:square_mul` | true | no | pay: fires=4 changed=0 paid=0 verify_fail=0 cases=20 | — |
| `grammar:sub_to_add_dup` | known | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=20 | — |
| `grammar:FALSE_exp_add` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `grammar:FALSE_square_add` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `grammar:div_add` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 333eq/27ne/0rerr (360 acc, 0 dec, 360 envs) | — |
| `census:mul_unsqueeze` | known | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 69eq/8ne/8rerr (174 acc, 0 dec, 174 envs) | — |
| `census:add_unsqueeze` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 69eq/8ne/8rerr (174 acc, 0 dec, 174 envs) | — |
| `mixed:add_chunk_r_w` | false | no | oracle: real: 0eq/0ne/3rerr (3 acc, 0 dec, 3 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_r_id` | false | no | oracle: real: 0eq/0ne/3rerr (3 acc, 0 dec, 3 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_r_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_r_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_l_w` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_l_id` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_l_w` | false | no | oracle: real: 1eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_l_id` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_transpose_r_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 103eq/33ne/45rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_transpose_r_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 55eq/81ne/45rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_slice_unsqueeze_id` | false | no | oracle: real: 0eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 9eq/119ne/71rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_slice_unsqueeze_wl` | false | no | oracle: real: 0eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 22eq/106ne/71rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_slice_unsqueeze_wr` | false | no | oracle: real: 0eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 22eq/106ne/71rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_r_id` | false | no | oracle: real: 0eq/0ne/4rerr (5 acc, 0 dec, 5 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_l_id` | known | no | oracle: real: 0eq/0ne/4rerr (4 acc, 0 dec, 4 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_r_w` | false | no | oracle: real: 0eq/0ne/4rerr (5 acc, 0 dec, 5 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_l_w` | false | no | oracle: real: 0eq/0ne/4rerr (4 acc, 0 dec, 4 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `census:sub_unsqueeze` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 69eq/8ne/8rerr (174 acc, 0 dec, 174 envs) | — |

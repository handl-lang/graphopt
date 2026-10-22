# Overnight discovery run

Episodes -> mint -> admit -> classify.  Open-game episodes mint candidate rewrites; each minted claim is judged by the ``bench/referees`` gauntlet (oracle -> pay -> sweep -> cert); admitted laws are classified against the prior-art bank: **machine-original** (the widest shipped bank cannot derive it), **known-art** (an alpha-sibling exists or the bank proves it), **vacuous** (no binding produced a site — unevaluable).

- date: 2026-10-22T04:14:49+00:00 · git: `40463d3` (dirty) · device cpu
- protocol: `{"budget": 48, "compose": 0, "corpus": "zoo-real", "dry": false, "episodes": 10, "fusion": true, "mint": "corpus-bind", "page": 48, "player": "playbook", "referee": "full", "scan": 0, "seed": 0}`
- corpus: `zoo-real` — held-out zoo modules (mlopt_discovery.zoo), candidates minted from each module's own op-tuple census (pipeline.propose); real_terms are the corpus's distinct subterms
- episodes: 10 · evaluated: 55 · minted: 40 · admitted: 3 · wall: 18.2s
- arena verdicts: false=44, known=11

## Admission funnel

| first-failed leg | count |
|---|---|
| oracle | 32 |
| pay | 5 |
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
| `fold:addmm_linear` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 47eq/0ne/20rerr (218 acc, 142 dec, 360 envs) | — |
| `fold:sdpa_fold_div_nomask` | known | no | pay: fires=5 changed=5 paid=5 verify_fail=1 cases=14 \| count=0.435x exec=0.417x flops=0.493x launch=0.491x params=1.000x | — |
| `fold:sdpa_fold_mul_nomask` | known | yes | cleared | known-art |
| `slice_mul` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 80eq/38ne/39rerr (294 acc, 0 dec, 294 envs) | — |
| `fold:glu_fold` | known | yes | cleared | known-art |
| `mixed:mul_chunk_l_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_l_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `fold:group_norm` | false | no | pay: fires=2 changed=2 paid=2 verify_fail=1 cases=11 \| count=0.111x exec=0.100x flops=0.198x launch=0.198x params=1.000x | — |
| `mixed:add_reshape_r_w` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_reshape_r_id` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_reshape_wl` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 22eq/24ne/86rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_reshape_wr` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 24eq/24ne/84rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_reshape_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 19eq/56ne/57rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_reshape_l_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_reshape_l_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `grammar:sub_to_add_dup` | known | no | pay: fires=8 changed=0 paid=0 verify_fail=0 cases=14 | — |
| `fold:instance_norm` | false | no | pay: fires=2 changed=2 paid=2 verify_fail=1 cases=11 \| count=0.077x exec=0.077x flops=0.193x launch=0.193x params=1.000x | — |
| `mixed:add_unsqueeze_l_id` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_r_id` | false | no | oracle: real: 0eq/3ne/0rerr (3 acc, 0 dec, 3 envs) \| synth: 0eq/178ne/39rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:mul_reshape_r_w` | false | no | oracle: real: 0eq/0ne/3rerr (3 acc, 0 dec, 3 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_r_id` | false | no | oracle: real: 1eq/2ne/0rerr (3 acc, 0 dec, 3 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_unsqueeze_l_w` | false | no | oracle: real: 1eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_r_w` | false | no | oracle: real: 0eq/3ne/0rerr (3 acc, 0 dec, 3 envs) \| synth: 0eq/178ne/39rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `fold:rms_norm_fold_nogain` | known | yes | cleared | known-art |
| `mixed:mul_reshape_l_w` | false | no | oracle: real: 0eq/0ne/3rerr (3 acc, 0 dec, 3 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_l_id` | known | no | oracle: real: 0eq/1ne/2rerr (3 acc, 0 dec, 3 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:pow_reshape_l_w` | false | no | oracle: real: 1eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 155eq/14ne/73rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:pow_reshape_l_id` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 73eq/126ne/43rerr (360 acc, 0 dec, 360 envs) | — |
| `fold:sdpa_fold_addmul` | known | no | pay: fires=6 changed=6 paid=6 verify_fail=4 cases=14 \| count=0.311x exec=0.210x flops=0.326x launch=0.326x params=1.000x | — |
| `mixed:sub_unsqueeze_unsqueeze_id` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/142ne/51rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:add_unsqueeze_r_id` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_unsqueeze_wl` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 8eq/45ne/125rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_unsqueeze_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 6eq/96ne/76rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_unsqueeze_wl` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 63eq/79ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_unsqueeze_wr` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 65eq/77ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_unsqueeze_r_w` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_unsqueeze_wr` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 33eq/69ne/76rerr (360 acc, 0 dec, 360 envs) | — |
| `reshape_transpose` | false | no | oracle: real: 0eq/4ne/0rerr (4 acc, 0 dec, 4 envs) \| synth: 61eq/100ne/0rerr (161 acc, 0 dec, 161 envs) | — |
| `mixed:mul_unsqueeze_r_id` | false | no | oracle: real: 1eq/0ne/1rerr (2 acc, 0 dec, 2 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_r_w` | false | no | oracle: real: 0eq/1ne/1rerr (2 acc, 0 dec, 2 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |

# Overnight discovery run

Episodes -> mint -> admit -> classify.  Open-game episodes mint candidate rewrites; each minted claim is judged by the ``bench/referees`` gauntlet (oracle -> pay -> sweep -> cert); admitted laws are classified against the prior-art bank: **machine-original** (the widest shipped bank cannot derive it), **known-art** (an alpha-sibling exists or the bank proves it), **vacuous** (no binding produced a site — unevaluable).

- date: 2026-10-22T04:05:54+00:00 · git: `81cf176` (dirty) · device cpu
- protocol: `{"budget": 48, "compose": 2, "corpus": "zoo-real", "dry": false, "episodes": 8, "mint": "corpus-bind-compose-guarded", "page": 48, "player": "playbook", "referee": "full", "scan": 2048, "seed": 0}`
- corpus: `zoo-real` — held-out zoo modules (mlopt_discovery.zoo), candidates minted from each module's own op-tuple census (pipeline.propose); real_terms are the corpus's distinct subterms
- episodes: 8 · evaluated: 22 · minted: 19 (1 composites, 1 guarded) · admitted: 0 · wall: 16.3s
- arena verdicts: false=22

## Admission funnel

| first-failed leg | count |
|---|---|
| oracle | 17 |
| pay | 2 |
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
| `mixed:mul_unsqueeze_unsqueeze_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/142ne/51rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:add_unsqueeze_unsqueeze_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/142ne/51rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:mul_unsqueeze_unsqueeze_wl` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 63eq/79ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_unsqueeze_wr` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 65eq/77ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_unsqueeze_unsqueeze_wl` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 63eq/79ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_unsqueeze_unsqueeze_wr` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 65eq/77ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `reshape_transpose` | false | no | oracle: real: 0eq/3ne/0rerr (3 acc, 0 dec, 3 envs) \| synth: 61eq/100ne/0rerr (161 acc, 0 dec, 161 envs) | — |
| `mixed:add_chunk_r_w` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_r_id` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_r_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_r_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_l_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_l_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_l_w` | false | no | oracle: real: 1eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_l_id` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `grammar:mul_distribute` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=12 | — |
| `syn_linear_row_scale_rev__linear_row_scale_rev_0@LoRAAdapter` | false | no | pay: fires=3 changed=3 paid=0 verify_fail=2 cases=10 \| count=1.000x exec=1.000x flops=1.000x launch=1.000x params=1.000x | — |
| `mixed:add_transpose_r_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 103eq/33ne/45rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_transpose_r_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 55eq/81ne/45rerr (360 acc, 0 dec, 360 envs) | — |

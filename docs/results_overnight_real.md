# Overnight discovery run

Episodes -> mint -> admit -> classify.  Open-game episodes mint candidate rewrites; each minted claim is judged by the ``bench/referees`` gauntlet (oracle -> pay -> sweep -> cert); admitted laws are classified against the prior-art bank: **machine-original** (the widest shipped bank cannot derive it), **known-art** (an alpha-sibling exists or the bank proves it), **vacuous** (no binding produced a site — unevaluable).

- date: 2026-10-21T22:06:50+00:00 · git: `863825c` (dirty) · device cpu
- protocol: `{"budget": 48, "corpus": "zoo-real", "dry": false, "episodes": 16, "mint": "corpus-bind", "page": 48, "player": "playbook", "referee": "full", "scan": 2048, "seed": 0}`
- corpus: `zoo-real` — held-out zoo modules (mlopt_discovery.zoo), candidates minted from each module's own op-tuple census (pipeline.propose); real_terms are the corpus's distinct subterms
- episodes: 16 · evaluated: 42 · minted: 29 · admitted: 0 · wall: 25.2s
- arena verdicts: false=39, known=3

## Admission funnel

| first-failed leg | count |
|---|---|
| oracle | 27 |
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
| `mixed:add_chunk_r_w` | false | no | oracle: real: 0eq/0ne/3rerr (3 acc, 0 dec, 3 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_r_id` | false | no | oracle: real: 0eq/0ne/3rerr (3 acc, 0 dec, 3 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_r_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_r_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_l_w` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_chunk_l_id` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_l_w` | false | no | oracle: real: 1eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_l_id` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `grammar:mul_distribute` | false | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=20 | — |
| `mixed:add_transpose_r_w` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 103eq/33ne/45rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_transpose_r_id` | false | no | oracle: real: 0eq/0ne/1rerr (1 acc, 0 dec, 1 envs) \| synth: 55eq/81ne/45rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_slice_unsqueeze_id` | false | no | oracle: real: 0eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 9eq/119ne/71rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_slice_unsqueeze_wl` | false | no | oracle: real: 0eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 22eq/106ne/71rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_slice_unsqueeze_wr` | false | no | oracle: real: 0eq/0ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 22eq/106ne/71rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_r_id` | false | no | oracle: real: 0eq/0ne/4rerr (5 acc, 0 dec, 5 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_l_id` | known | no | oracle: real: 0eq/0ne/4rerr (4 acc, 0 dec, 4 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_r_w` | false | no | oracle: real: 0eq/0ne/4rerr (5 acc, 0 dec, 5 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_l_w` | false | no | oracle: real: 0eq/0ne/4rerr (4 acc, 0 dec, 4 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_unsqueeze_id` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 0eq/142ne/51rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:sub_unsqueeze_unsqueeze_wl` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 63eq/79ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_unsqueeze_wr` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 65eq/77ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `grammar:sub_to_add_dup` | known | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=20 | — |

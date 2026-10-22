# Overnight discovery run

Episodes -> mint -> admit -> classify.  Open-game episodes mint candidate rewrites; each minted claim is judged by the ``bench/referees`` gauntlet (oracle -> pay -> sweep -> cert); admitted laws are classified against the prior-art bank: **machine-original** (the widest shipped bank cannot derive it), **known-art** (an alpha-sibling exists or the bank proves it), **vacuous** (no binding produced a site — unevaluable).

- date: 2026-10-22T22:33:43+00:00 · git: `8f9bf88` (dirty) · device cpu
- protocol: `{"budget": 48, "compose": 0, "corpus": "zoo-xmodule", "dry": false, "episodes": 8, "fusion": true, "mint": "corpus-xbind", "page": 48, "player": "playbook", "referee": "full", "scan": 2048, "seed": 0, "xbind": 2}`
- corpus: `zoo-xmodule` — one corpus-level mint: op-patterns and antiunified deep shapes recurring across >= --xbind distinct modules seed pipeline.propose + the xanti family; every claim must bind >= K real modules to pool
- episodes: 8 · evaluated: 29 · minted: 29 · admitted: 1 · wall: 0.0s
- arena verdicts: false=24, known=5
- cross-module mint: k=2 · cross-tuples=106 · proposals=69 · xanti=0 (+32 emit-gated) · pool=29 · bind histogram (pre-gate): {'0': 30, '1': 10, '2': 6, '3': 3, '4': 6, '5': 5, '6': 2, '8': 2, '9': 2, '12': 1, '14': 2}
- minted candidates' module-bind counts: 2 mods x6, 3 mods x3, 4 mods x6, 5 mods x5, 6 mods x2, 8 mods x2, 9 mods x2, 12 mods x1, 14 mods x2

## Admission funnel

| first-failed leg | count |
|---|---|
| oracle | 25 |
| pay | 3 |
| **admitted** | 1 |

## Novelty

| verdict | count |
|---|---|
| machine-original | 0 |
| known-art | 1 |
| vacuous | 0 |

## Machine-original finds

**None.** Every admitted law was already a theorem of the shipped bank — or nothing survived the gauntlet.

## Candidates

| candidate | arena | admitted | reason | novelty |
|---|---|---|---|---|
| `fold:sdpa_fold_addmul` | known | no | pay: fires=6 changed=6 paid=6 verify_fail=4 cases=57 \| count=0.311x exec=0.210x flops=0.326x launch=0.326x params=1.000x | — |
| `mixed:mul_unsqueeze_unsqueeze_id` | false | no | oracle: real: 0eq/13ne/9rerr (22 acc, 0 dec, 22 envs) \| synth: 0eq/142ne/51rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:add_unsqueeze_unsqueeze_id` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/142ne/51rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `mixed:sub_unsqueeze_unsqueeze_id` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/142ne/51rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `fold:sdpa_fold_mul_nomask` | known | yes | cleared | known-art |
| `mixed:mul_unsqueeze_unsqueeze_wl` | false | no | oracle: real: 4eq/9ne/9rerr (22 acc, 0 dec, 22 envs) \| synth: 63eq/79ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_unsqueeze_wr` | false | no | oracle: real: 5eq/8ne/9rerr (22 acc, 0 dec, 22 envs) \| synth: 65eq/77ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_unsqueeze_unsqueeze_wl` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 63eq/79ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_unsqueeze_unsqueeze_wr` | false | no | oracle: real: 0eq/0ne/2rerr (2 acc, 0 dec, 2 envs) \| synth: 65eq/77ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_r_id` | false | no | oracle: real: 3eq/4ne/28rerr (36 acc, 0 dec, 36 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_l_id` | known | no | oracle: real: 1eq/8ne/27rerr (36 acc, 0 dec, 36 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_unsqueeze_wl` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 63eq/79ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_unsqueeze_wr` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 65eq/77ne/51rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_l_id` | known | no | oracle: real: 1eq/3ne/0rerr (4 acc, 0 dec, 4 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_unsqueeze_r_id` | false | no | oracle: real: 3eq/0ne/2rerr (5 acc, 0 dec, 5 envs) \| synth: 23eq/155ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `fold:addmm_linear` | false | no | oracle: real: 2eq/0ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 47eq/0ne/20rerr (218 acc, 142 dec, 360 envs) | — |
| `mixed:add_chunk_r_w` | false | no | oracle: real: 0eq/0ne/5rerr (5 acc, 0 dec, 5 envs) \| synth: 173eq/27ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_chunk_r_id` | false | no | oracle: real: 0eq/0ne/5rerr (5 acc, 0 dec, 5 envs) \| synth: 98eq/102ne/20rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_r_w` | false | no | oracle: real: 0eq/7ne/28rerr (36 acc, 0 dec, 36 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_unsqueeze_l_w` | false | no | oracle: real: 0eq/9ne/27rerr (36 acc, 0 dec, 36 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:sub_unsqueeze_l_w` | false | no | oracle: real: 2eq/2ne/0rerr (4 acc, 0 dec, 4 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_reshape_r_w` | false | no | oracle: real: 0eq/1ne/2rerr (3 acc, 0 dec, 3 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_reshape_r_id` | false | no | oracle: real: 1eq/2ne/0rerr (3 acc, 0 dec, 3 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_r_w` | false | no | oracle: real: 0eq/3ne/5rerr (8 acc, 0 dec, 8 envs) \| synth: 95eq/15ne/97rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:mul_reshape_r_id` | false | no | oracle: real: 4eq/3ne/1rerr (8 acc, 0 dec, 8 envs) \| synth: 60eq/84ne/63rerr (360 acc, 0 dec, 360 envs) | — |
| `mixed:add_unsqueeze_r_w` | false | no | oracle: real: 0eq/3ne/2rerr (5 acc, 0 dec, 5 envs) \| synth: 100eq/78ne/39rerr (360 acc, 0 dec, 360 envs) | — |
| `reshape_transpose` | false | no | oracle: real: 0eq/4ne/0rerr (4 acc, 0 dec, 4 envs) \| synth: 61eq/100ne/0rerr (161 acc, 0 dec, 161 envs) | — |
| `grammar:mul_distribute` | false | no | pay: fires=49 changed=0 paid=0 verify_fail=0 cases=57 | — |
| `grammar:sub_to_add_dup` | known | no | pay: fires=22 changed=0 paid=0 verify_fail=0 cases=57 | — |

# Overnight discovery run

Episodes -> mint -> admit -> classify.  Open-game episodes mint candidate rewrites; each minted claim is judged by the ``bench/referees`` gauntlet (oracle -> pay -> sweep -> cert); admitted laws are classified against the prior-art bank: **machine-original** (the widest shipped bank cannot derive it), **known-art** (an alpha-sibling exists or the bank proves it), **vacuous** (no binding produced a site — unevaluable).

- date: 2026-10-21T07:12:21+00:00 · git: `3324191` · device cpu
- protocol: `{"budget": 48, "corpus": "zoo", "dry": false, "episodes": 48, "page": 48, "player": "playbook", "referee": "full", "scan": 2048, "seed": 0}`
- corpus: `zoo` — held-out zoo modules (mlopt_discovery.zoo)
- episodes: 36 · evaluated: 1600 · minted: 248 · admitted: 1 · wall: 63.3s
- arena verdicts: false=1497, known=69, true=34

## Admission funnel

| first-failed leg | count |
|---|---|
| oracle | 232 |
| pay | 14 |
| error | 1 |
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
| `enumerate:factor-left:284` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 118eq/242ne/0rerr (360 acc, 0 dec, 360 envs) | — |
| `enumerate:factor-right:286` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 145eq/215ne/0rerr (360 acc, 0 dec, 360 envs) | — |
| `enumerate:rekey:283` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 80eq/276ne/0rerr (360 acc, 0 dec, 360 envs) | — |
| `enumerate:idempotent:61` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:cancel-self:160` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:cancel-self:198` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:cancel-self:229` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:comm:36` | known | no | pay: fires=102 changed=9 paid=0 verify_fail=0 cases=40 \| count=1.000x exec=1.000x flops=1.000x launch=1.000x params=1.000x | — |
| `enumerate:comm:126` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 3eq/196ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:self-op:188` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:self-op:219` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:self-op:259` | known | no | pay: fires=6 changed=1 paid=0 verify_fail=0 cases=40 \| count=1.000x exec=1.000x flops=1.000x launch=1.000x params=1.000x | — |
| `enumerate:rotate:280` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 76eq/276ne/0rerr (360 acc, 0 dec, 360 envs) | — |
| `enumerate:lit-self:486` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:rotate:501` | true | no | pay: fires=6 changed=5 paid=0 verify_fail=0 cases=40 \| count=1.000x exec=1.000x flops=1.000x launch=1.000x params=1.000x | — |
| `enumerate:lit-self:519` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:lit-self:554` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:lit-self:584` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:lit-self:600` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:lit-self:616` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:lit-self:630` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:lit-self:646` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:lit-self:662` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:self-op:299` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:factor-left:505` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 54eq/302ne/0rerr (360 acc, 0 dec, 360 envs) | — |
| `enumerate:factor-right:507` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 54eq/302ne/0rerr (360 acc, 0 dec, 360 envs) | — |
| `enumerate:bin-unary-right:610` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:612` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:624` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:626` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:640` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:642` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:672` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:674` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:686` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:688` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:702` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:704` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:734` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:736` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:748` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:656` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:658` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:718` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:720` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:lit-self:678` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/17ne/0rerr (17 acc, 0 dec, 17 envs) — no equal site measured | — |
| `enumerate:lit-self:692` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 2eq/15ne/0rerr (17 acc, 0 dec, 17 envs) | — |
| `enumerate:lit-self:708` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 2eq/15ne/0rerr (17 acc, 0 dec, 17 envs) | — |
| `enumerate:comm:0` | known | no | pay: fires=61 changed=2 paid=0 verify_fail=0 cases=40 \| count=1.000x exec=1.000x flops=1.000x launch=1.000x params=1.000x | — |
| `enumerate:bin-unary-right:30` | false | no | oracle: real: 0eq/15ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:32` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) | — |
| `enumerate:bin-unary-right:75` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) | — |
| `enumerate:bin-unary-left:77` | false | no | oracle: real: 0eq/15ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:120` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) | — |
| `enumerate:bin-unary-left:122` | false | no | oracle: real: 0eq/15ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:201` | false | no | oracle: real: 0eq/15ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:203` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) | — |
| `enumerate:bin-unary-right:234` | false | no | oracle: real: 0eq/15ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:236` | false | no | oracle: real: 0eq/15ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:274` | false | no | oracle: real: 0eq/15ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:276` | false | no | oracle: real: 0eq/15ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:350` | true | yes | cleared | known-art |
| `enumerate:bin-unary-left:352` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) | — |
| `enumerate:bin-unary-right:388` | false | no | oracle: real: 0eq/15ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:390` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) | — |
| `enumerate:bin-unary-right:421` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) | — |
| `enumerate:bin-unary-left:423` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) | — |
| `enumerate:bin-unary-right:165` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) | — |
| `enumerate:bin-unary-left:167` | false | no | oracle: real: 0eq/15ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:314` | false | no | oracle: real: 0eq/15ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:316` | false | no | oracle: real: 0eq/15ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:461` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) | — |
| `enumerate:bin-unary-left:463` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) | — |
| `enumerate:bin-unary-right:494` | false | no | oracle: real: 0eq/15ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:496` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) | — |
| `enumerate:bin-unary-right:529` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) | — |
| `enumerate:bin-unary-left:531` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) | — |
| `enumerate:bin-unary-right:564` | false | no | oracle: real: 0eq/15ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:566` | false | no | oracle: real: 0eq/15ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:594` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) | — |
| `enumerate:bin-unary-left:596` | false | no | oracle: real: 1eq/14ne/0rerr (16 acc, 0 dec, 16 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) | — |
| `enumerate:bin-unary-right:1074` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/199ne/0rerr (289 acc, 0 dec, 289 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:1076` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 1eq/198ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-right:1080` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/199ne/0rerr (289 acc, 0 dec, 289 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:1082` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/199ne/0rerr (289 acc, 0 dec, 289 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:1086` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/199ne/0rerr (289 acc, 0 dec, 289 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:1088` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/199ne/0rerr (289 acc, 0 dec, 289 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:1096` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 16eq/183ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-left:1098` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 16eq/183ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-right:1102` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 16eq/183ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-left:1104` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/199ne/0rerr (289 acc, 0 dec, 289 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:1108` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 32eq/167ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-left:1110` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 17eq/182ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-right:1118` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 1eq/198ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-left:1120` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/199ne/0rerr (289 acc, 0 dec, 289 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:1124` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/199ne/0rerr (289 acc, 0 dec, 289 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:1126` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 1eq/198ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-right:1130` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/199ne/0rerr (289 acc, 0 dec, 289 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:1132` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/199ne/0rerr (289 acc, 0 dec, 289 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:1090` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/199ne/0rerr (289 acc, 0 dec, 289 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:1092` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/199ne/0rerr (289 acc, 0 dec, 289 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:1112` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 18eq/181ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-left:1114` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 17eq/182ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-right:1134` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/199ne/0rerr (289 acc, 0 dec, 289 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:1136` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/199ne/0rerr (289 acc, 0 dec, 289 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:1140` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 33eq/166ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-left:1142` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 17eq/182ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-right:1146` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 16eq/183ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-left:1148` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 4eq/195ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-right:1152` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 49eq/150ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-left:1154` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 37eq/162ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-right:1156` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 37eq/162ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-left:1158` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 37eq/162ne/0rerr (289 acc, 0 dec, 289 envs) | — |
| `enumerate:bin-unary-left:750` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:764` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 1eq/134ne/0rerr (225 acc, 0 dec, 225 envs) | — |
| `enumerate:bin-unary-left:766` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:780` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:782` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:796` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 15eq/150ne/0rerr (255 acc, 0 dec, 255 envs) | — |
| `enumerate:bin-unary-left:798` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:810` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 15eq/150ne/0rerr (255 acc, 0 dec, 255 envs) | — |
| `enumerate:bin-unary-left:812` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:826` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 15eq/150ne/0rerr (255 acc, 0 dec, 255 envs) | — |
| `enumerate:bin-unary-left:828` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:842` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 1eq/164ne/0rerr (255 acc, 0 dec, 255 envs) | — |
| `enumerate:bin-unary-left:844` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) — no equal site measured | — |
| `enumerate:rotate:39` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:factor-right:178` | false | no | oracle: real: 0eq/2ne/0rerr (2 acc, 0 dec, 2 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:rotate:40` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:rekey:42` | false | no | oracle: real: 0eq/7ne/0rerr (7 acc, 0 dec, 7 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:rotate:172` | false | no | oracle: real: 0eq/7ne/0rerr (7 acc, 0 dec, 7 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:rekey:174` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:factor-left:44` | true | no | pay: fires=49 changed=0 paid=0 verify_fail=0 cases=40 | — |
| `enumerate:factor-left:176` | false | no | oracle: real: 0eq/7ne/0rerr (7 acc, 0 dec, 7 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:assoc:1` | known | no | pay: fires=24 changed=5 paid=0 verify_fail=0 cases=40 \| count=1.000x exec=1.000x flops=1.000x launch=1.000x params=1.000x | — |
| `enumerate:rekey:41` | false | no | oracle: real: 0eq/5ne/0rerr (5 acc, 0 dec, 5 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:rekey:86` | false | no | oracle: real: 0eq/5ne/0rerr (5 acc, 0 dec, 5 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:factor-right:6` | false | no | oracle: real: 1eq/4ne/0rerr (5 acc, 0 dec, 5 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) | — |
| `enumerate:rekey:131` | false | no | oracle: real: 0eq/5ne/0rerr (5 acc, 0 dec, 5 envs) \| synth: 10eq/346ne/0rerr (360 acc, 0 dec, 360 envs) | — |
| `enumerate:assoc:37` | known | no | cleared | — |
| `enumerate:rekey:173` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:rekey:242` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:factor-right:210` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:rekey:282` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 80eq/276ne/0rerr (360 acc, 0 dec, 360 envs) | — |
| `enumerate:bin-unary-right:235` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:self-op:60` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:const-op:65` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:pull-right:216` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:self-op:220` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:const-op:224` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:self-op:374` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:const-op:378` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-of-lit:498` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:const-op:522` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-of-lit:533` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-absorb:187` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-absorb:218` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-absorb:258` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-absorb:298` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:self-op:518` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/17ne/0rerr (17 acc, 0 dec, 17 envs) — no equal site measured | — |
| `enumerate:self-op:150` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:const-op:155` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:self-op:300` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:const-op:304` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:self-op:447` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:const-op:451` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:self-op:105` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:const-op:110` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:rotate:171` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:self-op:260` | known | no | pay: fires=6 changed=4 paid=0 verify_fail=0 cases=40 \| count=1.000x exec=1.000x flops=1.000x launch=1.000x params=1.000x | — |
| `enumerate:const-op:264` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:rotate:281` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 76eq/276ne/0rerr (360 acc, 0 dec, 360 envs) | — |
| `enumerate:self-op:407` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:const-op:411` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:factor-right:46` | true | no | pay: fires=7 changed=0 paid=0 verify_fail=0 cases=40 | — |
| `enumerate:unary-absorb:336` | true | no | pay: fires=24 changed=0 paid=0 verify_fail=0 cases=40 | — |
| `enumerate:unary-absorb:372` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-absorb:405` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-absorb:445` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:factor-left:285` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 118eq/242ne/0rerr (360 acc, 0 dec, 360 envs) | — |
| `enumerate:factor-right:287` | false | no | oracle: real: 0eq/1ne/0rerr (1 acc, 0 dec, 1 envs) \| synth: 145eq/215ne/0rerr (360 acc, 0 dec, 360 envs) | — |
| `enumerate:comm:81` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 1eq/134ne/0rerr (225 acc, 0 dec, 225 envs) | — |
| `enumerate:bin-unary-right:858` | known | no | pay: fires=15 changed=0 paid=0 verify_fail=0 cases=40 | — |
| `enumerate:bin-unary-left:860` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 1eq/134ne/0rerr (225 acc, 0 dec, 225 envs) | — |
| `enumerate:bin-unary-right:874` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:876` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:888` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:890` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:918` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:920` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:932` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:904` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:906` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:self-op:17` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:const-op:22` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-of-lit:34` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-of-lit:79` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-of-lit:124` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:pull-left:140` | true | no | pay: fires=6 changed=0 paid=0 verify_fail=0 cases=40 | — |
| `enumerate:unary-of-lit:169` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:self-op:189` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:const-op:193` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-of-lit:205` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:pull-left:213` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:pull-right:215` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:unary-of-lit:238` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-absorb:15` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-absorb:58` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-absorb:103` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:unary-hom:211` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:hom-cross:233` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:unary-absorb:148` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:934` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:944` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:946` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:972` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:974` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:986` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:988` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:998` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:1000` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:958` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:960` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:1012` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:1014` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:1026` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:1028` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 1eq/164ne/0rerr (255 acc, 0 dec, 255 envs) | — |
| `enumerate:bin-unary-right:1040` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:1042` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:1052` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:1054` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) — no equal site measured | — |
| `enumerate:bin-unary-right:1066` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:1068` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/165ne/0rerr (255 acc, 0 dec, 255 envs) — no equal site measured | — |
| `enumerate:bin-unary-left:204` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:pull-left:50` | true | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=40 | — |
| `enumerate:bin-unary-right:202` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:pull-right:52` | true | no | pay: fires=5 changed=0 paid=0 verify_fail=0 cases=40 | — |
| `enumerate:pull-left:181` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:pull-right:183` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:unary-hom:179` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/135ne/0rerr (225 acc, 0 dec, 225 envs) — no equal site measured | — |
| `enumerate:rotate:356` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:factor-right:247` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:rotate:241` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |
| `enumerate:unary-of-lit:278` | true | no | pay: fires=8 changed=0 paid=0 verify_fail=0 cases=40 | — |
| `enumerate:unary-of-lit:318` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:self-op:338` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:const-op:342` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/15ne/0rerr (15 acc, 0 dec, 15 envs) — no equal site measured | — |
| `enumerate:factor-left:245` | false | no | oracle: real: 0eq/0ne/0rerr (0 acc, 0 dec, 0 envs) \| synth: 0eq/299ne/0rerr (360 acc, 0 dec, 360 envs) — no equal site measured | — |

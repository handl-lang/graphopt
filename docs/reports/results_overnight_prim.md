# Overnight discovery run

Episodes -> mint -> admit -> classify.  Open-game episodes mint candidate rewrites; each minted claim is judged by the ``bench/referees`` gauntlet (oracle -> pay -> sweep -> cert); admitted laws are classified against the prior-art bank: **machine-original** (the widest shipped bank cannot derive it), **known-art** (an alpha-sibling exists or the bank proves it), **vacuous** (no binding produced a site — unevaluable).

- date: 2026-10-24T07:53:05+00:00 · git: `9b69e8e` (dirty) · device cpu
- protocol: `{"budget": 48, "compose": 0, "corpus": "zoo-prim", "dry": false, "episodes": 24, "fusion": false, "mint": "corpus-prim", "page": 48, "player": "playbook", "prim": {"beam": 32, "cap": 24, "max_expand": 2000, "max_ops": 6, "min_ops": 2}, "referee": "full", "scan": 2048, "seed": 0, "xbind": 2}`
- corpus: `zoo-prim` — the primitive mint: manual shapes recurring across >= --xbind distinct modules and denoted by no bank op mint NEW synop_* primitives (denotation = the subgraph; the fold is novel-by-construction)
- episodes: 24 · evaluated: 24 · minted: 24 · admitted: 21 · wall: 12.6s
- arena verdicts: false=24
- primitive mint: k=2 · window=2..6 ops · cap=24 · mined=24/211 total (scan capped at pool) · k-bound=24 · denoted=0 · unevaluable=0 · gate-failed=0 · no-domain=0 · handler minted=24 refused=0 · pool=24
- qualified shapes by op-count: 3 ops x1, 5 ops x9, 6 ops x14
- minted folds' module-bind counts: 2 mods x1, 3 mods x4, 4 mods x18, 14 mods x1
- synop-in-shape (vocabulary bootstrap): `synop_5e58f26875` embeds `synop_bb8c54c201`; `synop_5147826430` embeds `synop_b246dcb838`; `synop_d5b128282a` embeds `synop_889d740672`; `synop_39f76b66f8` embeds `synop_93e57c595e`; `synop_571fb768b0` embeds `synop_2b8cfd1c90`; `synop_62d36aefdb` embeds `synop_f3290a1163`; `synop_629f9ef13b` embeds `synop_6c709f774a`/`synop_bb8c54c201`; `synop_6c709f774a` embeds `synop_bb8c54c201`; `synop_4c657cb2c0` embeds `synop_00cd48d9a2`
- minted primitives:
  - `synop_5f1bba5798` (prim:synop_5f1bba5798) — 6 ops {linear,mul,select,unsqueeze}, 4 mods, 16/16 equal sites, handler=minted (gather, err=5.33e-15)
  - `synop_4c657cb2c0` (prim:synop_4c657cb2c0) — 6 ops {add,linear,mul,select,unsqueeze}, 2 mods, 16/16 equal sites, handler=minted (gather, err=3.55e-15) — embeds synop_00cd48d9a2
  - `synop_5e58f26875` (prim:synop_5e58f26875) — 6 ops {add,mul}, 4 mods, 16/16 equal sites, handler=minted (ew, err=0) — embeds synop_bb8c54c201
  - `synop_4f8fefdd38` (prim:synop_4f8fefdd38) — 6 ops {add,mul}, 4 mods, 16/16 equal sites, handler=minted (ew, err=0)
  - `synop_5147826430` (prim:synop_5147826430) — 6 ops {add,mul}, 4 mods, 16/16 equal sites, handler=minted (ew, err=0) — embeds synop_b246dcb838
  - `synop_d5b128282a` (prim:synop_d5b128282a) — 6 ops {add,mul}, 4 mods, 16/16 equal sites, handler=minted (ew, err=0) — embeds synop_889d740672
  - `synop_39f76b66f8` (prim:synop_39f76b66f8) — 6 ops {add,mul}, 4 mods, 16/16 equal sites, handler=minted (ew, err=0) — embeds synop_93e57c595e
  - `synop_571fb768b0` (prim:synop_571fb768b0) — 6 ops {add,mul}, 4 mods, 16/16 equal sites, handler=minted (ew, err=0) — embeds synop_2b8cfd1c90
  - `synop_62d36aefdb` (prim:synop_62d36aefdb) — 6 ops {add,mul}, 4 mods, 16/16 equal sites, handler=minted (ew, err=0) — embeds synop_f3290a1163
  - `synop_629f9ef13b` (prim:synop_629f9ef13b) — 6 ops {add,mul}, 4 mods, 16/16 equal sites, handler=minted (ew, err=0) — embeds synop_6c709f774a/synop_bb8c54c201
  - `synop_976f9836d7` (prim:synop_976f9836d7) — 6 ops {add,mul,unsqueeze}, 3 mods, 16/16 equal sites, handler=minted (ew, err=0)
  - `synop_f08a53f9a0` (prim:synop_f08a53f9a0) — 6 ops {add,mul,unsqueeze}, 3 mods, 16/16 equal sites, handler=minted (ew, err=0)
  - `synop_4c5296b3ba` (prim:synop_4c5296b3ba) — 6 ops {add,mul,unsqueeze}, 3 mods, 16/16 equal sites, handler=minted (ew, err=0)
  - `synop_98a002d64b` (prim:synop_98a002d64b) — 6 ops {add,mul,unsqueeze}, 3 mods, 16/16 equal sites, handler=minted (ew, err=0)
  - `synop_00cd48d9a2` (prim:synop_00cd48d9a2) — 5 ops {linear,mul,select,unsqueeze}, 4 mods, 16/16 equal sites, handler=minted (gather, err=4.44e-15)
  - `synop_2b8cfd1c90` (prim:synop_2b8cfd1c90) — 5 ops {add,mul}, 4 mods, 16/16 equal sites, handler=minted (ew, err=0)
  - `synop_f3290a1163` (prim:synop_f3290a1163) — 5 ops {add,mul}, 4 mods, 16/16 equal sites, handler=minted (ew, err=0)
  - `synop_6c709f774a` (prim:synop_6c709f774a) — 5 ops {add,mul}, 4 mods, 16/16 equal sites, handler=minted (ew, err=0) — embeds synop_bb8c54c201
  - `synop_03889d8fe1` (prim:synop_03889d8fe1) — 5 ops {add,mul}, 4 mods, 16/16 equal sites, handler=minted (ew, err=0)
  - `synop_b246dcb838` (prim:synop_b246dcb838) — 5 ops {add,mul}, 4 mods, 16/16 equal sites, handler=minted (ew, err=0)
  - `synop_889d740672` (prim:synop_889d740672) — 5 ops {add,mul}, 4 mods, 16/16 equal sites, handler=minted (ew, err=0)
  - `synop_93e57c595e` (prim:synop_93e57c595e) — 5 ops {add,mul}, 4 mods, 16/16 equal sites, handler=minted (ew, err=0)
  - `synop_336e34145b` (prim:synop_336e34145b) — 5 ops {mul,select,unsqueeze}, 4 mods, 16/16 equal sites, handler=minted (gather, err=0)
  - `synop_bb8c54c201` (prim:synop_bb8c54c201) — 3 ops {add,mul}, 14 mods, 16/16 equal sites, handler=minted (ew, err=0)

## Admission funnel

| first-failed leg | count |
|---|---|
| error | 3 |
| **admitted** | 21 |

## Novelty

| verdict | count |
|---|---|
| machine-original | 21 |
| known-art | 0 |
| vacuous | 0 |

## Machine-original finds

- `prim:synop_4c657cb2c0` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_5e58f26875` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_4f8fefdd38` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_5147826430` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_d5b128282a` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_39f76b66f8` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_571fb768b0` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_62d36aefdb` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_629f9ef13b` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_976f9836d7` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_f08a53f9a0` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_4c5296b3ba` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_98a002d64b` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_2b8cfd1c90` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_f3290a1163` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_6c709f774a` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_03889d8fe1` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_b246dcb838` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_889d740672` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_93e57c595e` — arena `false`, instance corpus:MambaSelectiveScan
- `prim:synop_bb8c54c201` — arena `false`, instance corpus:HighwayGate

## Candidates

| candidate | arena | admitted | reason | novelty |
|---|---|---|---|---|
| `prim:synop_5f1bba5798` | false | no | cleared | — |
| `prim:synop_4c657cb2c0` | false | yes | cleared | machine-original |
| `prim:synop_5e58f26875` | false | yes | cleared | machine-original |
| `prim:synop_4f8fefdd38` | false | yes | cleared | machine-original |
| `prim:synop_5147826430` | false | yes | cleared | machine-original |
| `prim:synop_d5b128282a` | false | yes | cleared | machine-original |
| `prim:synop_39f76b66f8` | false | yes | cleared | machine-original |
| `prim:synop_571fb768b0` | false | yes | cleared | machine-original |
| `prim:synop_62d36aefdb` | false | yes | cleared | machine-original |
| `prim:synop_629f9ef13b` | false | yes | cleared | machine-original |
| `prim:synop_976f9836d7` | false | yes | cleared | machine-original |
| `prim:synop_f08a53f9a0` | false | yes | cleared | machine-original |
| `prim:synop_4c5296b3ba` | false | yes | cleared | machine-original |
| `prim:synop_98a002d64b` | false | yes | cleared | machine-original |
| `prim:synop_00cd48d9a2` | false | no | cleared | — |
| `prim:synop_2b8cfd1c90` | false | yes | cleared | machine-original |
| `prim:synop_f3290a1163` | false | yes | cleared | machine-original |
| `prim:synop_6c709f774a` | false | yes | cleared | machine-original |
| `prim:synop_03889d8fe1` | false | yes | cleared | machine-original |
| `prim:synop_b246dcb838` | false | yes | cleared | machine-original |
| `prim:synop_889d740672` | false | yes | cleared | machine-original |
| `prim:synop_93e57c595e` | false | yes | cleared | machine-original |
| `prim:synop_336e34145b` | false | no | cleared | — |
| `prim:synop_bb8c54c201` | false | yes | cleared | machine-original |

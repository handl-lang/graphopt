# T-009 — corpus-side coverage: the missing handler tags

Date: 2026-11-07.  Follows
`2026-11-07-meta-arena-macros-swiglu.md`, whose verdict named the
residual: "richer handler tags / mined specs the `gen_cases`
corpus never produces (e.g. `square`)".  This round diffed the
handler-tag space against the corpus and extended the generator.

## The tag census

`lawdata.HANDLERS` ships exactly three tags — `silu`, `square`,
`swiglu` — and a `MetaArena` board offers `claim(tag)` for every
tag whose *concrete* pattern occurs in the program
(`meta_arena._claim_tags`).  A census of `gen_cases(0, 400)` +
both held-out streams (pre-change) found:

| tag | boards offering it | oracle argmax |
|---|---:|---|
| `silu` | 250/400 train, 20/40 + 28/40 held-out | `claim_silu_extract` macro |
| `swiglu` | 82/400 (always beside `silu` — the inner site) | `claim_swiglu_extract` macro |
| `square` | **0 — never produced** | — |

Every claimable board paid, every paying board was a single
2-step macro, and `softsign`/`control` were the only declines.
Three coverage holes followed:

1. **`square` never occurs** — the third shipped handler had no
   board at all.
2. **No claimable-but-unprofitable board** — every offered claim
   tag paid, so the policy never had to *refuse* a claim.
   (`square`'s fold `mul(X,X) → square(X)` is op-count neutral:
   both sides price as one op over the shared `X`, so the claim
   binds honestly and still loses the step cost.)
3. **No multi-site boards** — every paying board had exactly one
   winning site, so the oracle argmax was always a `claim_*_extract`
   compression.  Nothing exercised the spelled
   `declare→saturate→handle→extract` macro as the *winner*.

## What was added (`meta_player.gen_cases`)

Five new kinds beside the five base ones (10 builders; vars
`z/w/v` join `x/y`):

- **`square`** — `mul(t,t)` under `nosq = sans − {mul_square}`:
  `claim(square)` is offered (pattern occurs, kernel in the
  ambient bound) and the oracle declines at −0.05.  First
  claimable-decline boards.
- **`square_mix`** — `add(silu(x), mul(y,y))` under `nosq`: the
  silu site pays, `claim(square)` stays offered and must lose —
  per-tag selectivity.
- **`tri`** — `add(add(silu(x), swiglu(y,z)), mul(w,w))` under
  `nosq`: three claim tags (`silu`, `swiglu`, `square`), the
  richest board in the corpus.
- **`dual`** — `add(silu(x), swiglu(y,z))`: two disjoint
  silu-family sites.  Oracle argmax moves to the spelled
  `declare_saturate_handle_silu_extract` macro (+0.450 vs
  `claim_silu/claim_swiglu_extract` +0.283) — the unbounded
  `saturate` inside the seq reaches *both* sites where one
  `claim` fire does not.
- **`dual_swiglu`** — `add(swiglu(y,z), swiglu(w,v))`: two
  disjoint swiglu sites; `claim(swiglu)` fires on both, oracle
  stays on `claim_swiglu_extract` at the corpus's highest payout
  (+0.521).

All boards are real evaluable cases — same probe arity, same
refereed moves, kernels priced through the same `supported`
bound; no synthetic padding.  `macro_moves.offered` coverage on
the new boards needed **no new MACROS rows**: every oracle argmax
is already a shipped offer (`declare_saturate_handle_silu_extract`
on `dual`/`square_mix`/`tri`, `claim_swiglu_extract` on
`dual_swiglu`, plain `extract` on `square`).

## Measurement — bake recipe unchanged

```
.venv/bin/python -m mlopt_discovery.meta_player \
    --seed <S> --train-cases 400 --eval-cases 40 --budget 24 \
    --temperature 1.2 --epsilon 0.25 \
    --imitate-epochs 3 --epochs 5 --json run.json
```

Held-out offsets `S=0 → gen_cases(7919, 40)`, `S=23418 →
gen_cases(31337, 40)`.  1200/1200 demonstrated moves cloned at
both seeds; oracle plans on the 400 train boards: ~166 claim-line,
~117 spelled-macro picks (the new winning shape), rest decline.

### Held-out reward — learned-greedy (sampled in parens)

| arm | offset 7919 | offset 31337 |
|---|---:|---:|
| oracle ceiling (extended corpus) | +0.352 | +0.349 |
| old five-kind corpus record | +0.20 (+0.18) | +0.27 (+0.23) |
| **old weights on extended corpus** (control) | +0.292 | +0.301 |
| **new bake on extended corpus** | **+0.32 (+0.30)** | **+0.33 (+0.29)** |

The corpus change lifts *both* weight tables (the new paying
boards are largely free money for the existing policy); the
re-bake itself is worth ~**+0.03** at both offsets.  Oracle gap
on the extended stream: ~0.03 (7919) / ~0.02 (31337).

### Per-kind held-out reads (learned-greedy vs oracle, offset 7919)

| kind | n | policy | oracle | read |
|---|---:|---:|---:|---|
| `square` | 5 | −0.050 | −0.050 | **exact** — refuses `claim(square)` |
| `square_mix` | 6 | +0.398 | +0.450 | right line, non-terminal macro spelling (−1 step) |
| `dual` | 4 | +0.283 | +0.450 | plays `claim_swiglu_extract`; spelled macro wins argmax |
| `dual_swiglu` | 7 | +0.521 | +0.521 | **exact** |
| `tri` | 2 | +0.200 | +0.325 | plays `claim_swiglu_extract`; spelled silu macro wins |
| `silu`/`swiglu`/`silu_nested` | 17 | oracle-equal | — | unchanged |
| `control`/`softsign` | 3 | −0.050 | −0.050 | exact declines |

## Verdict: MOVED — and the residual is now named precisely

Held-out moved at both offsets → `meta_prior.META_ARENA_PLAYER_WEIGHTS`
re-baked from the seed-0 run (shipped convention; seed 23418 is the
replication stream).  The policy generalized to every new board
class without being told the kinds: `square` declines exactly,
`dual_swiglu` matches the oracle, `square_mix` loses only the
terminal-step compression.

**Remaining headroom is macro-choice blindness, not coverage.**
On `dual`/`tri` the oracle's winner is the four-step spelled macro
while the policy picks a `claim_*_extract` compression — both are
viable offers whose feature rows differ only in `x:macro:len` /
`x:macro:score` / handler-bucket legs, and the winner is
*anti*-correlated with the miner's score (spelled macro scores
3.85 vs `claim_silu_extract` 4.13 vs `claim_swiglu_extract`
10.48).  No shipped feature lets the policy price "this board has
two silu sites the unbounded saturate reaches" — a schema limit,
consistent with the V2 second-order arm measuring flat.

**Honest caveat — handler-bucket collision.**  `META_ARENA_HASH_BUCKETS["handler"]
= 4` and `silu`/`square` hash to the *same* bucket (3): literal
`claim(silu)` and `claim(square)` featurize identically, so on
`square_mix`/`tri` boards the policy cannot prefer one tag over
the other by identity — it survives because the oracle line there
is a macro, not a literal claim.  Widening the bucket table is a
lawdata/gamekit-side change, out of this lever's scope; recorded
here as the next corpus-adjacent lever if literal-claim selectivity
ever matters.

## Files

- `packages/mlopt-discovery/src/mlopt_discovery/meta_player.py` —
  `gen_cases` +5 kinds, `nosq` ruleset, `z/w/v` vars, docstring.
- `packages/mlopt-discovery/src/mlopt_discovery/meta_prior.py` —
  re-baked `META_ARENA_PLAYER_WEIGHTS` (43 entries, 4dp) + bake
  notes.
- `project/reports/t009_corpus_s0.json`,
  `t009_corpus_s31337.json` — bake JSONs (uncommitted, `project/`
  is ignored).
- `project/retros/meta-arena-player.md`, `project/TODO.md` —
  lever record.

Tests: `test_discovery_meta_player` (53), `test_discovery_meta_arena`
+ `meta_prior` + `play` + `player_arena` + `meta_edges` +
`law_order` — all green under `-m "not slow"`; ruff/format/ty clean
on the touched files.

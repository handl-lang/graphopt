# T-009 — MACROS move-space extension: `claim_swiglu_extract`

> **Corpus caveat (added later):** the held-out numbers below were
> measured on the *five-kind* `gen_cases` stream.  `gen_cases` later
> grew to 10 kinds (see `2026-11-07-meta-arena-corpus-coverage.md`),
> so the same offsets now deal different boards — the figures are a
> historical record, not comparable to post-extension bakes.

Date: 2026-11-07.  Follows
`2026-11-07-meta-arena-candidate-lines.md`, whose verdict named the
residual: the oracle argmax on swiglu boards was still the *spelled*
`claim(swiglu) → extract` (+0.566) because `lawdata.MACROS` shipped
no compression for it — a synthesized macro is not in the legal set,
so it could be neither demonstrated nor played.

## What was measured before shipping

A probe replayed 18 hand-built candidate compressions (every
claim/saturate/extract ordering and budget arm the oracle
enumerates, plus all-claims compositions) through `macro_moves.offered`
— i.e. only *viable* rows, mirroring the real offer seam — against
the oracle argmax on both held-out streams (`gen_cases(7919, 40)`,
`gen_cases(31337, 40)`, budget 24):

- **`claim(swiglu) → extract` as one macro: +0.617 vs argmax +0.566
  — beats on all 15 swiglu boards** (+0.051 each: one step charge
  plus the enode term a terminal macro never pays — the same +0.05
  mechanism `claim_silu_extract` already earned).
- `saturate → extract` / `saturate(256) → extract` macros: exact
  tie with plain `extract` on every decline board (−0.050) —
  saturation finds no further reach and the terminal step waives
  the enode spend, so they neither win nor lose.  Not shipped:
  zero measured gain, and a viable-everywhere row would spend
  offer-cap slots on every board.
- All-claims compositions (`claim(silu)→claim(swiglu)→extract`,
  both orders, ±saturate, both budgets): never beat `swiglu_ext`
  alone — the swiglu kernel subsumes the inner silu fold.
- `claim → saturate → extract` / `saturate → claim → extract`
  variants (unbounded + 256): never beat the plain fused macro —
  consistent with the oracle-side measurement that the extra step
  buys no reach.

## The row shipped

`scripts/macro_miner.py` re-run over the same train stream
(`gen_cases(0, 24)`, 206 trajectories) surfaces the gram as its
**rank-1 candidate** under current machinery — the shipped macros
are themselves offers now, so trajectories include macro plays:

| ops | count | certified | mean_terminal | score |
|---|---:|---:|---:|---:|
| `claim(swiglu) → extract` | 17 | 1.00 | 0.6167 | **10.4833** |
| `claim(silu) → extract` | 28 | 1.00 | 0.3131 | 8.7667 |

`lawdata.MACROS` gains `claim_swiglu_extract` (inserted at rank 2,
beside `claim_silu_extract` so both fused compressions lead the
table; the silu row's stats are the original bake's — a full
`--emit` re-mine is future work).  `viable` gates it on
`swiglu ∈ state.claim_tags`; on swiglu boards the offer set is 5
viable rows, under `DEFAULT_MACRO_CAP = 8`.

Post-change oracle argmax (`oracle_line`, budget 24): **every**
paying board is now a single macro step — `macro:claim_swiglu_extract`
+0.617 on all swiglu cases at both offsets; `macro:claim_silu_extract`
unchanged on silu/silu_nested; plain `extract` on decline boards.
No spelled `claim` remains argmax anywhere on the corpus.

## Bake recipe (unchanged)

```
.venv/bin/python -m mlopt_discovery.meta_player \
    --seed <S> --train-cases 400 --eval-cases 40 --budget 24 \
    --temperature 1.2 --epsilon 0.25 \
    --imitate-epochs 3 --epochs 5 --json run.json
```

Held-out offsets: `S=0` → `gen_cases(7919, 40)`;
`S=23418` → `gen_cases(31337, 40)`.  Both warm-start from the
shipped `meta_prior` bake (the spec default).

## Results — learned-greedy mean reward (sampled in parens)

| arm | offset 7919 | offset 31337 |
|---|---:|---:|
| candidate-lines bake, pre-extension MACROS | +0.19 (+0.18) | +0.25 (+0.23) |
| **MACROS extension** | **+0.20 (+0.18)** | **+0.27 (+0.23)** |

Consistent positive delta at both offsets and both stochastic
reads — the swiglu boards (~20% of the stream) now pay the
compressed line end-to-end.  Imitation cloned 1200/1200
demonstrated moves at each seed; oracle plan tiers unchanged
(250/236 claim-line, 0 spelled-pick, rest decline).

The new demonstrations are *all* macro actions on paying boards,
so `op:macro` flipped −1.2 → **+2.2** while `op:claim` stays
+2.1 — the policy holds both routes and picks per board via the
handler-bucket/`x:macro:*` features.  `learned-greedy` plays
`declared-use` 20/40 (7919) and 28/40 (31337) with zero spelled
`declare`/`handle` moves — every composed line goes through
`claim` or `macro`.

## Re-bake

Held-out moved at both offsets → `meta_prior.META_ARENA_PLAYER_WEIGHTS`
regenerated from the seed-0 bake (the shipped convention — seed
23418 is the replication stream, not a second bake).  Rounded
4dp / near-zero-dropped table verified against the held-out
stream: learned-greedy **+0.198**, learned +0.192 at budget 24 —
the measurement reproduces.

## Test pin updated

`test_imitate_oracle_clones_demonstrations` asserted the
demonstrated *claim* earns positive `op:claim` weight.  With every
paying oracle line now a macro offer, no spelled `claim` is
demonstrated on `gen_cases(0, 8)` and `op:claim` sits at −0.006 —
the pin moves to `op:macro` (+0.28 after 8/8 cloned demos), which
is where the composed-line credit now lands.

## Files

- `packages/mlopt-discovery/src/mlopt_discovery/lawdata/laws/macros.py`
  — the `claim_swiglu_extract` row + docstring.
- `packages/mlopt-discovery/src/mlopt_discovery/meta_prior.py` —
  re-baked weights + bake-notes paragraph.
- `tests/test_discovery_meta_player.py` — the imitation pin above.
- Bake JSONs (uncommitted): `t009_macros_s0.json`,
  `t009_macros_s31337.json` (this directory).

## Verdict: MOVED — the move-space residual is closed on this corpus

The last spelled argmax (`claim(swiglu) → extract`, +0.566) is now
a legal one-step offer paying +0.617; held-out learned-greedy
+0.19 → +0.20 / +0.25 → +0.27.  Every oracle plan on the generated
corpus is a shipped-macro or plain-extract line — the oracle's own
move space is now fully compressed.  Remaining headroom, if any, is
upstream: richer handler tags / mined specs the `gen_cases`
corpus never produces (e.g. `square`), or a `macro_miner --emit`
re-bake once the shipped table and the miner's top-N agree again.

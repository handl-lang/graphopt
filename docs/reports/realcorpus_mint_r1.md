# Real-corpus mint, round 1 — Whisper + T5

**Question.** The zoo mint produced 58 admitted `prim:synop_*` folds, but
the transfer probe (`docs/reports/law_transfer.md`) found 57/58 fire only
on their mint family — none fired on real Whisper/T5 exports. Does
minting *from* real-model terms produce laws that admit **and** pay on
those same models?

**Answer (r1): no.** Real-model terms yield bindable candidates —
including cross-module anti-unified shapes recurring across Whisper and
T5 — but the referee gauntlet admitted **0 of 48 judged**, and the one
candidate that reached the pay stage fired without paying. The
`zoo-prim` arm (the machinery that minted the shipped `synop_*` family)
could not complete on real-model roots inside a 900 s wall, so no
primitive folds were even produced. No admitted law means no fire or
pay-arm measurements exist; that is the honest result, not a skipped
step.

## Setup

Driver: `bench/drivers/mint/real_model_mint.py` (new, uncommitted —
bench tooling only, no package source touched). It reuses the existing
export path (`WhisperForConditionalGeneration` /
`T5ForConditionalGeneration` wrapped to return plain tensors →
`model_corpus.export_ir`), then runs the existing mint machinery:

- per-module `pipeline.propose(..., require_bind=True)` over each
  module's op-tuple census;
- cross-module `antiunify.mine_candidates` pooled over all roots with
  an `--xbind`-module recurrence gate;
- the `overnight._prim_pool` zoo-prim arm (`synop_*` primitive mint)
  over the real roots;
- admission through `overnight._admit` → `referees.judge` (oracle →
  pay → sweep → cert), with real-term referee contexts drawn from the
  exported subterm domains and `TorchSink` lowering.

Machine: `.venv/bin/python`, CPU-preferred; wall ≈ **1199 s** (~20 min)
for the full run below, inside the 30–60 min budget.

Command:

```sh
.venv/bin/python -u bench/drivers/mint/real_model_mint.py \
    --models whisper,t5 --pay-arm \
    --out docs/reports/realcorpus_mint_r1.json
```

JSON evidence: `docs/reports/realcorpus_mint_r1.json` (same directory).

## Exported corpus

| module | export | op-tuples | bind-domain kept/skipped |
|---|---|---|---|
| whisper:encoder | 0.5 s | 32 | 90 / 118 |
| whisper:decoder | 1.0 s | 46 | 216 / 197 |
| t5:encoder | 0.6 s | 69 | 108 / 206 |
| t5:decoder | 1.0 s | 74 | 228 / 343 |

The bind domain is the bounded distinct-`Op` subterm set per module
(`real_cap=2500`, `site_cap=1500`, `max_real=6000` ceilings; 864
subterms skipped as duplicates/over-cap). Referee context: 614 real
terms + 4 probe cases, `referee=full`.

## Mint funnel

Per-module `pipeline.propose` is cheap on real exports but binds almost
nothing on Whisper and moderately on T5:

| module | proposals | bound | families contributing |
|---|---|---|---|
| whisper:encoder | 35 | 1 | shape-aware 21, grammar 23 → 1 bound |
| whisper:decoder | 35 | 1 | same → 1 bound |
| t5:encoder | 42 | 9 | census-mixed 6, fusion 1, shape-aware 21, grammar 23 |
| t5:decoder | 42 | 9 | same |

Census naturality and pattern-recognition produced **zero** proposals
on all four modules — consistent with the earlier finding that the
generic proposers' patterns don't match these exact exports.

Cross-module arm (`xbind=2`, wall 600 s, used 138 s): 78 cross op-tuple
census entries → 420 anti-unification patterns → 99 proposals →
**64 pooled** after the ≥2-module bind gate
(`bind_hist`: 57 bind exactly 2 modules, 6 bind 3, 1 binds all 4).

Primitive arm (`--prim`, wall 900 s): **wall exceeded, zero
candidates**. `overnight._prim_pool` mines recurring manual shapes no
bank op denotes, then runs handler synthesis + emit-gate verification
per shape; on the four real-model roots that funnel did not finish in
15 min (the same knobs complete on zoo modules). Recorded honestly as
`mint._prim = {"error": "wall exceeded"}` — the shipped `synop_*`
machinery does not obviously transfer to real-term scale within this
budget. `family_mint` was off for r1.

Dedup by `alpha_key(lhs, rhs)` → **64 unique candidates**, 48 sent to
the gauntlet (`--max-candidates 48`).

## Candidates and admission verdicts

Family/orig-name distribution of the 48 judged:

| family | count | dominant shapes |
|---|---|---|
| xanti-view | 39 | `matmul(x, transpose(transpose(w)))`, `matmul(x, transpose(w))`, `matmul(transpose(a), transpose(b))` idioms |
| census-mixed-view | 6 | `add/sub` + `unsqueeze` view laws (`mixed:add_unsqueeze_*`) |
| layout | 1 | `reshape_transpose` (binds all 4 modules) |
| fusion-schema | 1 | `fold:rms_norm_fold_nogain` |
| algebraic-grammar | 1 | `sub_to_add_dup` (`sub → add(neg)`) |

Module coverage of judged candidates: t5:decoder 48, whisper:decoder
37, t5:encoder 15, whisper:encoder 1 — the cross-module pool is
decoder-dominated (Whisper encoder contributes almost nothing, matching
the earlier transfer gap around `conv1d`/`layer_norm`-heavy encoder
shapes).

**All 48 refused.** 47 died at the `oracle` stage — mostly
`0eq/0ne/0rerr` on 2–7 real envs (the minted LHS binds real sites but
the instances are degenerate or the equality never materializes), plus
`reshape_transpose` at `0eq/33ne` (real evidence the law is *false* on
these terms, not merely untested).

One candidate reached the pay stage:

- `rmint:008` `grammar:sub_to_add_dup` — `sub(M0,M1) → add(M0, neg(M1))`.
  Oracle **passed** but on thin evidence: `real: 1eq/0ne` (1 env) vs
  `synth: 135eq/0ne` (225 envs). Pay stage **failed**:
  `fires=6 changed=0 paid=0 verify_fail=0 cases=8` — the law fires on
  the real terms but changes nothing the extractor can deliver. This is
  the closest thing to a paying law in r1 and it does not pay.

Gauntlet walls were 0.0–57.9 s per candidate (median 1.7 s) — refusals
are cheap; nothing hit the 600 s wall.

## Fire / pay arms

Not run: both arms are gated on admission, and admission count is 0.
`report.fire = []`, `report.pay = []`. No timing-vs-eager numbers exist
because no law survived to measure — stated plainly rather than
implied.

## What this means vs the zoo-mint result

- Zoo mint: 58 `synop_*` folds **admitted** (they were *built* to fold
  recurring manual shapes, so admission was near-certain) but **did not
  transfer** — 57/58 fire only on their mint family.
- Real-model mint r1: candidates bind real terms organically
  (cross-module recurrence across two different architectures is real
  signal), but **none survive the oracle/pay gauntlet**, and the
  primitive-mint arm can't finish in budget. Real-model minting
  produced **no paying laws** — an honest negative for r1.

Interpretation caveats (not proof of impossibility):

- The prim arm's 900 s wall may just be too small — the funnel is
  per-shape expensive and real e-graphs are larger than zoo modules.
  A longer wall or tighter `budget`/`beam`/`max_expand` knobs is the
  obvious r2 knob.
- 16 of 64 unique candidates were never judged (`max_candidates=48`);
  they are recorded in `n_minted_total` but carry no verdict.
- T5 binds far more candidates than Whisper (9/42 vs 1/35 per module);
  a Whisper-only or T5-only rerun would sharpen per-model numbers.
- The oracle's `0eq/0ne/0rerr` refusals on 2–7 envs are weak-evidence
  refusals — real-term coverage of the bound sites is thin even with
  614 real terms, because bindable real subterms concentrate in a few
  decoder shapes.

## Reproduce / inspect

```sh
.venv/bin/python -m py_compile bench/drivers/mint/real_model_mint.py
.venv/bin/ruff check bench/drivers/mint/real_model_mint.py
cat docs/reports/realcorpus_mint_r1.json | python3 -m json.tool | less
```

Driver notes for future rounds: `_Wall` is a `BaseException` so broad
`except Exception` handlers inside `prim_mint`/`pipeline` cannot swallow
the wall budget (an `Exception`-based wall previously let the prim arm
run >20 min unbounded); the report JSON is checkpoint-written after the
mint and admit stages so a killed run still leaves evidence.

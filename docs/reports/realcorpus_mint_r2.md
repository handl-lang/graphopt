# Real-corpus mint, round 2 — Whisper + T5, DAG-safe + prim arm

**Question.** Same as r1 (`docs/reports/realcorpus_mint_r1.md`): does
minting laws *from* real-model terms produce candidates that admit
**and** pay on Whisper/T5? r2 asks whether the two r1 blockers —
the `zoo-prim` arm dying on the 900 s wall and the 48-candidate
judging cap — can be pushed through with DAG-safe evaluation and a
resumable judge, and whether the primitive-fold arm transfers at all.

**Answer (r2): still no — but the funnel now completes.** With
runtime DAG-safety patches the `zoo-prim` arm **finished** on real
roots (mined 503 recurring shapes, pooled 24 primitive-fold
candidates — r1 produced zero), and **all 90 unique candidates were
judged** (r1 judged 48 of 64). The referee gauntlet admitted **0 of
90**. 89 died at the oracle stage — mostly `0eq/0ne` on thin real
envs, and all 24 prim folds with `rhs_err` on *every* measured env
because the `synop_*` RHS is an uninterpreted op with no torch
binding. The one candidate that reached pay (`sub → add(neg)`,
`rmint:008`) again fired 6 times and paid 0 — identical to r1. No
admitted law means no novelty classification, fire, or pay-arm
measurements exist; that is the honest result.

## Setup

Driver: `bench/drivers/mint/real_model_mint.py` (uncommitted bench
tooling — `git diff` touches **only** this file; no package source
changed). r2 additions over r1:

- **`_install_dag_patches()`** — runtime-only monkeypatches, active
  only inside this driver process:
  - `proposal._leaves` → iterative, `id()`-memoized walk. Returns the
    identical leaf set (the shipped version already dedups
    structurally via `traversal.leaves`; the patch only removes
    re-walks of shared subtrees on DAG-shaped real terms).
  - `meta_eval._eval_term` → `id()`-memoized eval. Sound because the
    `_IR_TO_TORCH` bindings are pure tensor ops (the only stochastic
    binding, `aten.dropout`, is the eval-mode identity) and env
    leaves are fixed tensors — a shared `Op` object denotes the same
    value, so memoizing by `id()` within one eval is value-identical.
    The memo is per top-level call, so lhs/rhs evals don't share.
  - `prim_mint._term_match` → this driver's bounded `_match` (the same
    `repr(term)`-free divergence the candidate arms already run
    under; inert here since mined prim shapes have metavar leaves
    only).
- **Eval guard** — wraps `oracle.eval_instance` run-wide: counts
  calls, purges the torch host cache + `gc` every `--eval-purge` N
  calls (the CPU allocator retains ~200 MB/eval on real-model
  instances — measured in the r2 prim probe: 576 calls / 288 purges,
  1332 s wall, no OOM), and during the prim arm only, skips instances
  whose leaf env alone would exceed `--prim-eval-env-mb` (recorded as
  `rerr`/`env-err`, never silent).
- **`--resume <prior.json>`** — skips export-domain rebuild and all
  minting, rebuilds judgeable rules from the stored `law` records via
  `law_from_data` (unfaithfully-rebuildable records — non-empty
  `missing_hooks` — are marked `resume_skip`, never re-judged under a
  different rule), keeps existing verdicts, and continues at admit.
  Checkpoints after every stage and every candidate; a SIGTERM
  handler flushes the report before dying (`killed` field records it).

Machine: CPU-preferred, shared 11 GB host (earlyoom active). The
first r2 attempt was **SIGTERM-killed mid-judging** (9 eval calls in,
`wall_s 17.8` on the resumed retry — the host was under external
memory pressure). The completed judging pass ran under
`MALLOC_ARENA_MAX=2` and finished in **457 s** with no kill
(`eval_guard`: 10 398 eval calls, 1 299 purges).

Command (the completing run):

```sh
MALLOC_ARENA_MAX=2 .venv/bin/python -u bench/drivers/mint/real_model_mint.py \
    --models whisper,t5 --pay-arm \
    --resume docs/reports/realcorpus_mint_r2.json \
    --out docs/reports/realcorpus_mint_r2.json
```

(JSON evidence: `docs/reports/realcorpus_mint_r2.json` — mint stats
preserved across the resume; `docs/reports/realcorpus_mint_r2_prim.json`
is the standalone prim-arm probe that scoped the DAG-patch knobs.)

## Exported corpus

Identical to r1 — same four modules, same feeds:

| module | export | feeds |
|---|---|---|
| whisper:encoder | 0.5 s | `[1,80,3000]` |
| whisper:decoder | 1.0 s | `[1,1]`, `[1,1500,512]` |
| t5:encoder | 0.7 s | `[1,128]` ×2 |
| t5:decoder | 1.0 s | `[1,128]` ×2, `[1,128,512]` |

Referee context rebuilt identically: 614 real terms (+864 skipped
over-cap), 4 probe cases, `referee=full`.

## Mint funnel

Per-module `pipeline.propose` — unchanged from r1 (same census, same
bind rates):

| module | proposals | bound | wall |
|---|---|---|---|
| whisper:encoder | 35 | 1 | 1.6 s |
| whisper:decoder | 35 | 1 | 1.4 s |
| t5:encoder | 42 | 9 | 1.6 s |
| t5:decoder | 42 | 9 | 1.4 s |

Cross-module arm (`xbind=2`, wall 600 s, used 163.5 s): 78 cross
op-tuples → 420 anti-unification patterns → 99 proposals → **66
pooled** after the ≥2-module bind gate (`bind_hist`: 33 bind 0
modules and drop out of the pool, 59 bind 2, 6 bind 3, 1 binds all
4).

**Primitive arm — completed this time.** Scoped to the two decoder
roots (`--prim-models t5:decoder,whisper:decoder` — the encoder roots
were dropped to bound memory after the first OOM), wall 1489 s:

- 503 recurring manual shapes mined, 127 K-bound (≥2 modules),
  composition `depth_dist` {2:6, 3:15, 4:28, 5:36, 6:42}, 16 nested
  fold compositions recorded;
- emit gate: `gate_failed` 103, `denoted` 0, `unevaluable` 0,
  `tier_capped` 86; `synop_folds`/`synop_fires`/`synop_passes` all 0;
- **24 candidates pooled** (`refused` 24 is the pool's own
  admission-refusal count inside `_prim_pool`, distinct from the
  referee verdict below);
- eval guard during the arm: 2 316 `eval_instance` calls, 1 158
  cache purges, `env_skipped` 0 — the 1.5 GiB env cap never triggered
  on decoder-scoped roots.

Dedup by `alpha_key` → **90 unique candidates, all 90 judged**
(r1: 64 unique, 48 judged). Family mix: `xanti-view` 57,
`prim-fold` 24, `census-mixed-view` 6, `layout` 1,
`fusion-schema` 1, `algebraic-grammar` 1. Module coverage:
t5:decoder 90, whisper:decoder 76, t5:encoder 24, whisper:encoder 4 —
still decoder-dominated.

## Candidates and admission verdicts

**0 of 90 admitted.** Verdict-stage breakdown:

- **89 refused at `oracle`.** The dominant signature is
  `real: 0eq/0ne/Nrerr` — real sites bind but *no environment
  produces an equal measurement*: either too few accepted envs (2–33
  per candidate) or every eval errored. `reshape_transpose`
  (`rmint:000`) is the one candidate with positive real evidence —
  and it is *false*: `real: 0eq/33ne` vs `synth: 61eq` — a real
  counterexample, not thin coverage.
- **All 24 `prim-fold` candidates refused at oracle with `rerr` on
  every env** (both real and synth regions): e.g. `rmint:086`
  `real: 0eq/0ne/48rerr (48 envs) | synth: 0eq/0ne/15rerr`. The RHS
  is an uninterpreted `synop_*` op — `_eval_term` raises `KeyError`
  because no `_IR_TO_TORCH` binding exists — so the numeric oracle
  *cannot* measure them one way or the other. This is a machinery
  gap, not evidence the laws are false: on the zoo path these folds
  are admitted through expansion/denotation machinery the
  `referee=full` torch oracle here does not replay. `denoted: 0` in
  the prim stats is the same gap seen from the mint side.
- **1 reached `pay`: `rmint:008` `grammar:sub_to_add_dup`** —
  `sub(M0,M1) → add(M0, neg(M1))`. Oracle passed on thin evidence
  (`real: 1eq`, `synth: 135eq/225 envs`); pay stage failed
  `fires=6 changed=0 paid=0 verify_fail=0 cases=8` — fires on real
  terms, changes nothing deliverable. **Identical verdict to r1.**

Gauntlet walls: min 1.0 s / median 3.2 s / max 20.2 s per candidate —
no candidate approached the 600 s alarm; refusals are cheap.

## Fire / pay arms / novelty

Not run: both are gated on admission, and admission count is 0.
`report.fire = []`, `report.pay = []`, `report.admitted = []` — the
novelty sieve (`_classify`) never executed. Stated plainly rather
than implied.

## What r2 adds over r1

- The **prim arm is no longer the blocker**: with decoder-scoped
  roots, the id-memoized `_eval_term`, and purge-every-2 evals it
  completed in ~25 min (vs r1's 900 s wall-exceeded with zero
  candidates) and produced 24 judgeable primitive folds.
- Judging is now **complete and resumable**: all 90 candidates carry
  verdicts; the OOM-killed first attempt lost nothing (per-candidate
  checkpoints + `--resume`).
- The negative is *stronger*, not weaker: r1's 16 unjudged tail and
  missing prim arm are now measured, and the prim folds' refusal mode
  is identified (uninterpreted `synop_*` RHS → universal `rerr`),
  which localizes the next fix: give minted prim ops a torch binding
  or route them through the expand/denotation path before the oracle.

## Honest negatives / caveats

- **0 admissions is not proof no real-model law exists** — the oracle
  refuses on thin evidence (`0eq` with 2–7 envs is weak), and 24
  candidates were *unevaluable* rather than *false*. A wider real-term
  context or per-candidate env floors would sharpen this.
- The prim arm ran on **decoder roots only** — encoder-side recurring
  shapes were never mined in r2.
- `synop_*` candidates' `rerr` refusals are a **harness artifact**;
  treat the prim arm's transfer question as *unanswered by this
  referee*, not answered negatively.
- The mint stats in the JSON are from the pre-OOM original run
  (restored via `--resume`); `wall_s: 457.0` covers only the
  completing judging pass. Total r2 wall across attempts ≈ 1650 s
  mint + 457 s judge + the killed partials.
- `MALLOC_ARENA_MAX=2` and the eval purge were needed to keep the
  11 GB host alive; without them the judge OOMs again (observed).

## Reproduce / inspect

```sh
.venv/bin/python -m py_compile bench/drivers/mint/real_model_mint.py
.venv/bin/ruff check bench/drivers/mint/real_model_mint.py
cat docs/reports/realcorpus_mint_r2.json | python3 -m json.tool | less
```

Resume semantics for r3: point `--resume` at this JSON — judged
candidates keep verdicts, unjudged ones rebuild from `law` records.

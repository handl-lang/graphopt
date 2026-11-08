---
title: Retro — the meta-arena player (Q3, first answer)
type: retro
date: 2026-10-14
---

# Retro — the meta-arena player (Q3, first answer)

Plan 0021's Q3: is the mixed board deep enough that a *learned*
player beats the scripted `saturate → extract` playbook?  The
machinery is landed — `catopt_discovery.meta_player` is a linear
softmax policy over `meta_arena.legal_actions` with an
exact (uncapped) softmax, played-move masking, ε-exploration, and
reward-to-go REINFORCE; the schema is data
(`lawdata.META_ARENA_FEATURES`).  Reproduce:

    .venv/bin/python -m catopt_discovery.meta_player \
        --train-cases 1000 --eval-cases 20 --budget 24 \
        --temperature 1.2 --epsilon 0.25 --json run.json

## The paying line exists — verified by hand

The Q2 probe's baseline table already showed the headroom:
scripted/greedy/random all tie at baseline on every demo case.
What they miss is a *four-move composition*: `declare` a fresh
fold of a mined spec → `saturate` so the minted rule fires →
`handle` to bind a supported kernel → `extract` to collect.
Hand-set weights prove the schema expresses it: silu sites go
2→1, swiglu 3→1, nested 4→3, controls and infeasible-kernel
boards correctly decline.  `test_winning_line_reachable` pins it.

## The measurement — cold RL does not find it (at 1000 episodes)

`gen_cases` mixes paying boards (silu/swiglu sites under
`sans = DEFAULT - {silu_fold, softsign_fold}`, kernel still in
the ambient vocabulary) with non-paying ones (softsign under
`sans` — kernel out of vocab; controls under `DEFAULT`).

| player | reward | cert | decl | steps |
|---|---:|---:|---:|---:|
| scripted | −0.10 | 20 | 0 | 2.0 |
| greedy | −1.20 | 0 | 0 | 24.0 (drain) |
| random | −1.09 | 5 | 0 | 21.7 |
| learned | **−0.07** | 20 | 0 | 1.4 |
| learned-greedy | −0.05 | 20 | 0 | 1.0 |

Train decade means climb steadily (−0.94 → −0.41, plateau
~−0.44) — the learner is doing real work: `op:extract` +3.47,
`op:fire` −2.54.  What it converged to is the *stoppable-time*
policy: extract immediately, skip the wasteful saturate.  That
honestly beats scripted on mean reward — but on declared-use
columns it wins nothing.

## Why the line is hard — and what this measures

The net payout for the composed line is ~+0.25 over
immediate-extract; per-step cost −0.05 makes the margin thin,
and on 3/5 board kinds the line does not exist at all — a
policy that learns "declare usually doesn't pay" is *locally
correct*.  This is the sparse-reward credit-assignment wall:
the memoryless linear policy cannot plan a composition.  Two
features were added on the way and confirmed load-bearing-weak:
`st:unhandled`/`x:unhandled:handle` (declared-but-unhandled is
exactly where `handle` pays) and `st:handleable`/`x:handleable:
declare` (a mined spec whose canon form a handler covers — the
precondition, computable from `MetaState` data alone).

## Verdict and next levers

**Q3 is answered in the negative at this budget**: a cold-start
linear REINFORCE policy beats the scripted baseline on reward but
only via the degenerate line — the composed strategy is in the
policy class (hand weights reach it) yet unreachable by sampling
at ~10³ episodes.  The construction-arena result transfers:
learning beat baselines where paying moves were dense; here they
are composed, and composition is what the policy lacks.

Levers, in order of how well they match the tower's own thesis:

1. ~~**Options/macros**~~ — **landed.** ``claim(tag)`` is the
   composed line as one refereed move (declare a covered spec →
   fire its rule → bind the interpretation; Sanada's
   ``handle (declare spec) with H`` fused).  With it in the legal
   set the line compresses to ``claim → extract``: *sampling
   reachability is fixed* — the 300-episode run found 4 structural
   wins (silu 2→1) where zero existed before.  Final reading —
   800-episode run over occurrence-gated claims: **learned −0.03
   mean reward, 3 certified structural wins on 40 held-out cases
   (swiglu 3→1, silu 2→1, swiglu 3→2) — beats scripted (−0.10,
   zero structural wins) on both axes.**  **Caveat: the wins ride
   sampling variance, not learned preference** — ``op:claim``'s
   weight sits ≈+0.04, ``learned-greedy`` never claims.  The board
   question is answered: a learned player beats scripted on the
   mixed board, *because the option move compressed the
   composition* — action-space structure is what the board was
   missing; consolidation by a memoryless linear policy remains
   the open gap for a stronger policy class.
2. ~~**Curriculum**~~ — **partially landed.**  Training on the
   paying-board subset (``claimable > 0``; 250 of 400 generated)
   shifts the policy measurably: the stochastic arm's eval
   declared-wins tripled (9 of 40 incl. swiglu/silu sites and a
   `control` board with a paying subterm), mean cost 2.73 <
   scripted's 2.92 — but mean reward degrades (−0.35) because the
   policy over-claims where the extraction then can't improve.
   Curriculum concentrates the gradient on the interesting
   boards; it does not teach *when to stop*.
3. **Imitation warm-start** — behavior-clone the known line, then
   RL.  Honest about which part is discovery vs consolidation.
4. More training — 10⁴+ episodes; the decade trend was still
   rising at plateau's edge.

## Housekeeping

- `lawdata.META_ARENA_PLAYER_WEIGHTS` stays empty — no trained
  snapshot ships until one earns it (the eval arms re-train in
  `main`; nothing memorized).
- The ε knob and reward-to-go are *learner* machinery, not board
  changes — the reward table is untouched.

## Lever results — 2026-11-07 (T-009)

All three levers landed in `meta_player` (`oracle_line` /
`plan_oracles` / `imitate_oracle` / `curriculum_order`, CLI flags
`--epochs --imitate-epochs --curriculum`), measured on the same
protocol — `gen_cases(seed, n)` train, `gen_cases(seed+7919, 40)`
eval, budget 24, T=1.2, eps=0.25:

| arm | episodes | reward | cost | cert | decl-use | claims |
|---|---:|---:|---:|---:|---:|---:|
| scripted | — | −0.10 | 2.92 | 40 | 0 | 0 |
| cold RL (this run) | 2000 | **+0.14** | 2.42 | 40 | 20/40 | 0 (macro moves) |
| imitate 3ep + RL 5ep | 400×5 | **+0.17** | 2.25 | 40 | 20/40 | 20–23 literal `claim` |
| curriculum + RL 5ep | 400×5 | +0.14 | 2.42 | 40 | 21/40 | 0 (macro) |

Seed-1 replication of the imitation arm: +0.15, claims=21,
decl-use 21/40 — the effect is not a seed artifact.

**Consolidation: achieved — but read the mechanism honestly.**  The
learned-greedy arm now plays the composed line on ~half the held-out
boards (vs 3/40 structural wins in the retro).  Two routes get there:
cold RL and curriculum converge on **`macro` offers** — the mined
claim→extract compressions in `lawdata.MACROS`, which shipped
*after* this retro (959671b, 2026-10-19) — while the imitation arm
learns the primitive **`claim`** move itself (`op:claim` ≈ +5.9).
Action-space structure closed the gap a second time: what the
memoryless policy could not sequence, the offer set now spells for
it.  The spelled declare→saturate→handle line remains unplayed.

**The imitation warm-start is the kernelopt recipe and it works**:
1950/1950 demonstrated moves cloned on seed 0 (1914/1914 seed 1);
the oracle replays the per-case best of {extract,
saturate→extract, claim→extract, claim→saturate→extract,
all-claims→extract} — 250/400 train boards have a paying claim
line, 150 decline.  It buys +0.03 mean reward over cold and, more
to the point, routes through `claim` rather than the macro shortcut.

**Curriculum: measured negative at this scale.**  Tier-sorting
(paying claim-lines first, decline boards last) reproduces cold
RL's eval table exactly — the gradient concentration the partial
curriculum showed does not compound into better held-out play.

**Bug found and fixed on the way**: `LinearPolicy._observe` only
detects an episode boundary when the previous episode was observed
past step 0 (`prev_steps > 0`).  An episode ending on its first
move — exactly the extract-early line this board rewards — leaves
`extract` masked into the next episode during training *and*
drops ~23% of imitation demonstrations (1950→1500 shown/imitated
in the ablation).  Eval arms are fresh `frozen` players so held-out
numbers were never affected.  The boundary is not inferable inside
`_observe` (a `steps==0` re-observe is identical for "new board"
and "same state queried twice"), so `MetaLearnedPlayer.reset_episode`
forces it and the drivers call it per episode.  **Upstream fix
belongs in gamekit** — every `LearnedPlayer` domain has the same
exposure; this repo carries the local workaround.

## Lever result — the bake ships (2026-11-07, T-009)

**Lever 1 landed: the imitation arm's weights are now the shipped
prior.**  Recipe (`meta_player.main --imitate-epochs 3 --epochs 5`,
400 train cases, budget 24, T=1.2, eps=0.25): 1950/1950 demonstrated
moves cloned, then 5 REINFORCE passes — train decade means flat at
~+0.17 from the first decade (the imitation does the work; RL
polishes).

Head-to-head on the held-out stream, same protocol, frozen arms:

| arm | offset 7919 (n=40, b=24) | offset 31337 (n=40, b=24) | test protocol (n=16, b=12) |
|---|---:|---:|---:|
| shipped macro prior | +0.13 / greedy +0.14 | +0.17 / +0.18 | +0.11 / +0.12 |
| **imitation bake** | **+0.17 / +0.17** | **+0.22 / +0.23** | **+0.14 / +0.14** |
| scripted | −0.10 | −0.10 | −0.10 |

The bake wins every cell and — the point — routes through the
primitive `claim` move (22 literal claims, `op:macro` weight −1.2:
the policy *avoids* the macro shortcut) where the macro prior
played `op:macro` (0 literal claims).  Declared-use ~20/40 both.

**Where it lives:** `mlopt_discovery.meta_prior.META_ARENA_PLAYER_WEIGHTS`
— a new graphopt-owned module beside `meta_player`, bound directly
as `META_SPEC`'s weights.  `lawdata.META_ARENA_PLAYER_WEIGHTS`
still resolves to the gamekit-shipped macro-move prior: the
lawdata/gamekit split invariant (player weight tables are game-side
content; no table in both halves — `test_lawdata.py`) rules out a
lawdata shadow, and the bake's recipe (`imitate_oracle`/
`plan_oracles`) is domain machinery the shared game layer cannot
run anyway.  Two bakes of the same schema, distinct homes;
`player_arena`'s `baked` objective table follows the new preset.
`test_preset_is_baked` and the whole `test_discovery_meta_prior`
held-out suite pass (the preset assertion now names
`meta_prior`).

**Housekeeping on the way:** gamekit's upstream `_observe` boundary
fix (played-mask non-empty + `steps==0` ⇒ fresh board) landed in the
pinned dep, which staled two tests that pinned the *buggy* masking
(`test_declines_move_outside_legal_set`,
`test_reset_episode_clears_stale_played_mask`) — both updated to the
fixed semantics; `MetaLearnedPlayer.reset_episode` stays as the
explicit boundary hook (docstring updated).

**Still open:** levers 2–3 unmeasured — the 2nd-order feature /
neural arm, and a curriculum over *claim availability* rather than
board difficulty.  The spelled declare→saturate→handle line remains
unplayed; consolidation is now real but rides `claim`, not the
four-move spelling.

## Levers 2–3 measured (2026-11-07, T-009 remainder)

Both remaining levers landed as *measured* arms and both come back
honest: the policy is not the bottleneck — the oracle's own
candidate set is.

**Lever 3 — claim-availability curriculum: measured negative.**
`meta_player` grew `case_availability` (live `claimable` count +
`unhandled` count off a fresh observe — a *state* property, not
the oracle's verdict) and `availability_order`, exposed as
`--availability`.  On the bake recipe (400 train / 40 held-out,
budget 24, imitate 3 + RL 5, T=1.2, eps=0.25) the decade means
alternate ~0.36 / ~0.01 — the ordering really did change — and the
held-out table is *identical* to baseline and to every other arm:
learned-greedy +0.17, 20 literal claims, declared-use 20/40.
Availability ordering cannot help a policy that already imitates
the oracle perfectly on every board where a claim exists.

**The spelled line is reachable, priced, and dominated.**
`spelled_candidates` enumerates the primitive
`declare → saturate → handle → extract` lines (one per
occurring-spec × covering-handler pair, pre-filtered through
`meta_arena._handle`'s own legality test), `plan_spelled` scores
them, and `--oracle-spelled` / `--spelled-demo-epochs` wire them
into the oracle argmax and the imitation stream.  Verified
end-to-end: the spelled line *pays* (~+0.30 total on the silu
board) — but `claim → extract` pays ~+0.40 for the same rewrite,
so across all 400 training boards the oracle argmax picks the
spelled line **0 times** (`spelled-pick=0`; 250 claim-line, 150
decline).  The fused move is strictly better wherever it exists;
the spelled line is a demonstration curriculum for the
primitives, not a strategy.  Replaying it through imitation
(`--spelled-demo-epochs`) changes nothing held-out — same table.

**Lever 2 — richer policy: measured negative at the ceiling.**
`META_SPEC_V2` adds the second-order block the task asked for:
`MetaHist.last_op` carries the previous move into the featurizer
and `_prev_move_feats` emits `x:seq:<prev>><cand>` (last-move ×
candidate-kind — the composed line's edges: `declare>handle`,
`claim>extract`) plus `x:seqh:<prev>:<bucket>` (last-move ×
handler-identity on catalog-binding moves).  Broadcast
`x:prev:*`-style rows would cancel in the softmax, so only
candidate-varying products are emitted; `--prev-features` binds
the arm.  Same bake recipe, seed 0: the held-out table is
*identical* to V1 — learned-greedy +0.17, claims 20–22,
declared-use 20/40, imitation 1950/1950.  This was the expected
null: the earlier audit already put learned-greedy at the oracle
ceiling (eval mean +0.172 vs learned-greedy gap ≈ −0.0001, no case
below oracle).  There is no residual gap for a richer function
class to close — the remaining difference lives in lines the
oracle itself never picks.

**Neural arm: not re-run, prior says no.**  `gamekit.players.neural`
exists, but its measured record is the open domain (AP 0.041 vs
heuristic 0.045, trained linear prior 0.133) — it has never beaten
the linear arm anywhere, and the meta arena's bottleneck is shown
above to be upstream of policy capacity.

**Disposition.**  T-009's policy question is answered: imitation
bake + first-order features already sit at the oracle ceiling for
the current candidate set; second-order features, availability
curriculum, and spelled-line demonstrations are all implemented,
measured, and flat.  What remains genuinely open is *candidate
side*, not policy side: oracle lines beyond the current
{extract, saturate→extract, claim→extract, claim→saturate→extract,
multi-claim} set — e.g. spelled multi-declare lines, or claims
over-handler compositions the mint doesn't yet offer.

## Lever 4 — candidate-side oracle lines (2026-11-07, T-009 remainder)

The last open lever: the oracle's candidate set itself.  `_candidate_lines`
(`meta_player.py`) grew in two directions, both restricted to moves the
arena's own `legal_actions` enumerates so every oracle pick stays
imitable (`imitate` matches on the legal-set action key):

* **Offered macros** — `macro_moves.offered(state)`'s viable
  `lawdata.MACROS` compressions replayed as single refereed steps
  (`[macro]` when the seq ends terminal, else `macro → extract`).
  A macro charges one `step` and one budget tick for the whole seq,
  so the spelled `claim → extract` collects 0.05 less than the
  `claim_silu_extract` macro covering the same rewrite — and a
  terminal macro never pays the enode term at all.  This was the one
  candidate class the oracle had never tried despite being *the*
  legal compressed spelling of its own best lines.
* **Ordering/budget variants** — `claim → saturate(b) → extract`
  per bounded arm, `saturate → claim → extract` (closure-first,
  both arms), bounded saturates after all-claims, saturate-first
  multi-claim, and the reversed claim order.  Enumerated for
  completeness; measured below as never argmax.

`_plan_rank` and `evaluate`'s claims/spelled counters now flatten
macro seqs (`_line_ops`), so a `macro[claim,extract]` plan still
reads as the tier-0 fused line and macro plays count their inner
claims.

**Measured — bake recipe** (`--imitate-epochs 3 --epochs 5`, 400
train / 40 held-out, budget 24, T=1.2, eps=0.25; warm-starts from
the shipped `meta_prior` bake, same as any re-run of the recipe):

| arm | offset 7919 | offset 31337 |
|---|---:|---:|
| prior bake (recorded) | +0.17 / +0.17 | +0.22 / +0.23 |
| **control re-bake, old candidates, same prior** | +0.17 / +0.17 | +0.23 / +0.23 |
| **new candidates** | **+0.19 / +0.18** | **+0.25 / +0.23** |

Control runs isolate the candidate effect from the warm start:
re-baking the old candidate set from the shipped prior reproduces
the recorded table (the prior is a fixed point of its own recipe),
so the delta is the candidates'.  ~+0.02 mean at both offsets —
exactly the mechanism's size: `claim_silu_extract` is viable and
argmax on the silu/silu_nested boards (~25–30% of the stream),
+0.05 each.  `op:macro` weight moved −1.2 → **+2.7**; imitation
clones 1446/1446 (seed 0) and 1413/1413 (seed 23418) demonstrated
moves — every macro demo matched a live legal offer.  Oracle plan
tiers unchanged: 250 claim-line / 150 decline / 0 spelled-pick;
the new ordering variants never win argmax (verified per-board:
`claim→saturate→extract` and friends all pay −0.05 vs plain
`claim→extract` — the extra step buys no further reach).

**Verdict: MOVED, modestly — and the residual is now provably the
move space, not the oracle.**  The oracle argmax itself rose only
where a *shipped* macro covers the winning line; on swiglu boards
the argmax is still spelled `claim(swiglu) → extract` (+0.566)
because `lawdata.MACROS` ships no `claim_swiglu_extract` (or
all-claims / saturate compressions) — a synthesized macro would
pay ~+0.05 more there (one step charge plus the claim's enode
term, which a terminal macro never pays) but is not in the legal set, so it can be
neither demonstrated nor played.  Closing that last gap is a
`lawdata.MACROS` table extension (new mined rows → new offers for
every player), i.e. arena move-space work, not oracle-candidate
work.  Spelled multi-declare lines and fire-restricted variants
were checked and are strictly dominated (they tie the spelled
line's payout at 4-step cost; `claim`/macro compress the same
rewrite cheaper).

## Lever 5 — MACROS move-space extension (2026-11-07, landed)

The candidate-lines lever's residual was provably the *move space*:
`lawdata.MACROS` shipped no `claim_swiglu_extract`, so the oracle's
best swiglu line stayed spelled `claim(swiglu) → extract` (+0.566)
and could be neither demonstrated nor played as a macro.  A probe
replayed 18 candidate compressions (all claim/saturate orderings,
budgets, all-claims compositions) through `macro_moves.offered`'s
viability filter against the oracle argmax on both held-out
streams:

- `claim(swiglu)→extract` as one macro: **+0.617 vs +0.566 — wins
  all 15 swiglu boards** (+0.051 each, the same one-step-charge +
  waived-enode mechanism `claim_silu_extract` earns).
- `saturate→extract` / `saturate(256)→extract` macros: exact ties
  with `extract` on every decline board — not shipped (zero gain,
  viable-everywhere offer noise).
- All-claims and claim±saturate compositions: never beat the plain
  fused macro — the swiglu kernel subsumes the inner silu fold.

`macro_miner` re-run over the train stream ranks the gram **#1**
(count 17, certified 1.00, mean_terminal 0.6167, score 10.4833 —
above the silu gram's 8.77 under current machinery, since shipped
macros are now themselves offers in the mined trajectories).  The
row ships as `claim_swiglu_extract` at rank 2 beside
`claim_silu_extract`.

**Measured — same bake recipe** (imitate 3 / RL 5, 400 train, 40
held-out, budget 24, T=1.2, eps=0.25, warm-start from shipped
prior):

| arm | offset 7919 | offset 31337 |
|---|---:|---:|
| pre-extension bake | +0.19 / +0.18 | +0.25 / +0.23 |
| **MACROS extension** | **+0.20 / +0.18** | **+0.27 / +0.23** |

Positive at both offsets; imitation 1200/1200 cloned per seed.
Every paying oracle plan is now a single macro step — the oracle's
move space on this corpus is fully compressed.  `op:macro` flipped
−1.2 → +2.2 while `op:claim` holds +2.1: the policy keeps both
routes and picks per board.  `meta_prior` re-baked from the seed-0
run (verified +0.198 learned-greedy on the held-out stream at
budget 24).  `test_imitate_oracle_clones_demonstrations`'s pin
moved from `op:claim` to `op:macro` — no spelled claim is
demonstrated anymore on that corpus.

Report: `project/reports/2026-11-07-meta-arena-macros-swiglu.md`.

## Lever result — corpus-side coverage (2026-11-07, T-009)

The named residual — "handler tags/specs the `gen_cases` corpus
never produces (e.g. `square`)" — diffed and closed.  Census:
`lawdata.HANDLERS` ships three tags; pre-change `gen_cases`
produced boards offering only `silu`/`swiglu`, every claimable
board paid, every paying board was one 2-step macro.

`gen_cases` gained five kinds (10 builders): `square` (`mul(t,t)`
under `nosq = sans − {mul_square}` — `claim(square)` offered,
op-count-neutral, oracle declines: the corpus's first
claimable-but-unprofitable boards), `square_mix`/`tri` (paying
silu/swiglu site beside a square site — per-tag selectivity), and
`dual`/`dual_swiglu` (two disjoint silu-family sites — the first
boards whose oracle argmax is the *spelled*
`declare_saturate_handle_silu_extract` macro, +0.450 vs the
claim compressions' +0.283, because the unbounded `saturate`
reaches both sites where one `claim` fire does not).
`macro_moves.offered` needed no new rows — every new argmax is
already a shipped offer.

Same bake recipe, extended corpus, oracle ceiling +0.35:

| arm | offset 7919 | offset 31337 |
|---|---:|---:|
| old weights on extended corpus (control) | +0.292 | +0.301 |
| **new bake** | **+0.32 / +0.30** | **+0.33 / +0.29** |

Per-kind: `square` declines exactly (−0.05), `dual_swiglu` matches
oracle (+0.521), `square_mix` loses only the terminal-step
compression.  Residual gap ~0.03 sits on `dual`/`tri` — macro-choice
blindness: the spelled 4-step macro and the `claim_*_extract`
compressions differ only in `x:macro:len`/`score`/handler-bucket
legs and the winner is anti-correlated with the miner's score.
Caveat recorded: `silu`/`square` collide in handler bucket 3, so
literal `claim` rows can't distinguish the tags (harmless today —
the paying lines are macros).

`meta_prior` re-baked from the seed-0 run.  Report:
`project/reports/2026-11-07-meta-arena-corpus-coverage.md`.

## Lever result — macro move-shape legs + handler-bucket fix (2026-11-08, T-009)

Two levers from the corpus-coverage residual, both in
`meta_player` (domain-side; lawdata/gamekit untouched):

- **Lever A — spelled-vs-claim legs.** `_macro_feats` emits
  `x:macro:spelled`/`saturating`/`nclaims`, `reach_sites`/
  `reach_claimable` (spelled∧saturating × board broadcasts), and
  role-qualified handler buckets `x:mc:*`/`x:mh:*`.  New
  `META_FEATURES` schema = lawdata rows + `a:h:4..7` + legs;
  `META_SPEC`/`META_SPEC_V2` build on it.
- **Lever B — `silu`/`square` collision.** `_META_HANDLER_BUCKETS
  = 8` overrides the handler width in `_bucket`: silu→7,
  square→3, swiglu→0 — all shipped tags distinct.  The `a:h:*`
  index remap forced a re-bake regardless of Lever A's payoff.

Same bake recipe, 10-kind corpus, learned-greedy held-out:

| arm | offset 7919 | offset 31337 |
|---|---:|---:|
| previous bake | +0.32 / +0.30 | +0.33 / +0.29 |
| **this bake (A+B), seed 0** | **+0.33 / +0.30** | **+0.33 / +0.31** |
| replication, seed 23418 | +0.34 / +0.29 | +0.33 / +0.31 |

**Lever A: honest negative** — deltas +0.01/+0.00 and +0.00/+0.02,
within noise; the ~0.03 dual/tri residual did not close.  Weights
did learn the legs (spelled −0.85, reach_sites +2.23, x:mc:7 +1.00)
but held-out reward is flat — lever closed; the schema now names
the distinction, conversion likely needs more multi-site mass or a
nonlinear policy.  **Lever B: shipped as correctness** — latent
collision removed; unmeasurable today (square declines everywhere)
but the next paying square board would have been invisible.
`meta_prior` re-baked (53 nonzero rows) — required by the bucket
remap, performance-neutral.

Note: `test_candidate_lines_multi_claim_variants` fails under the
`-k "meta"` cross-file selection on clean HEAD too — pre-existing
ordering pollution, not this change (passes in isolation and in
the focused file run).

Report: `project/reports/2026-11-08-meta-arena-macro-legs.md`.

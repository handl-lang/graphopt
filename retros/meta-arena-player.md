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

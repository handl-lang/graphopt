# Strassen rediscovery, take two — the lookahead (MCTS) arm

**Date:** 2026-11-06 · **Board:** `mlopt_discovery.bilinear_game` ·
**Driver:** `bench/drivers/probes/strassen_mcts.py` · **Seed:** 0
throughout · **Runner:**

```bash
.venv/bin/python bench/drivers/probes/strassen_mcts.py \
    --sims 64,256 --episodes 8 --budget 48
# plus a 1024-sims arm on the canonical case
```

## The question

`docs/results_strassen.md` measured the Q2 go/no-go: the board
expresses Strassen (scripted line, verified rank 7 in 29 moves)
but greedy, random and cold-linear arms stall at rank 8 — a
~15-move coordinated valley through invalid space with no
positive prefix.  Its "implications" guessed a *planner that
searches residual-space rather than playing one move at a time*
might be what the board needs.  This document measures exactly
that arm: does lookahead cross the valley, and does the answer
scale with simulations-per-move?

## The planner

A minimal UCB-tree search on the board's own move graph
(`MCTSPlayer` in the driver — driver-level, not a shipped
player):

* **selection** — UCB1 over `legal_actions`: untried actions
  first, else min-max-normalised `Q + c·√(ln N/n)` (`c = 1.4`);
* **expansion** — lazy, one fresh node per simulation;
* **rollout** — a dig/repair cycle on a cloned board:
  * at a *valid* state: take a **collapse** if one exists —
    two terms with proportional factors in two modes collapse
    under the flip identity (`f_s(t_i) −= λ f_s(t_j)` zeroes the
    factor, `drop` banks rank −1 while validity is preserved —
    the mechanism `tests/test_discovery_bilinear_game.py` pins
    with an engineered zero); else a random flip (stay on the
    island) or a random cut (leave it);
  * at an *invalid* state: dig deeper with probability 0.35,
    else repair to validity by tempered top-8 L1 patches —
    softmax over `−‖R − u⊗v⊗w‖₁/τ` (`τ = 4`) on the same
    candidate table `_GreedyRepair` uses;
* **leaf value** — the best quality `−rank − 16·‖R‖₁` seen along
  the rollout; a rollout reaching a valid rank ≤ 7 auto-
  terminates the sim board (the win is *discovered*, never
  assumed);
* **commit** — most-visited root child; the tree re-roots on it
  when the next observed state matches, so a found line
  persists.

Rollouts are the honest part: they never see `STRASSEN_TERMS`
and share no machinery with the scripted line.  Everything the
rollout plays is a legal board move.

## The measurement

`best valid rank reached`, paired episodes, budget 48, seed 0
(anchors re-run for pairing — scripted 10 episodes, the rest 8):

| arm | eps | won (rank ≤ 7) | pct | mean best rank | mean reward | mean steps |
|---|---|---|---|---|---|---|
| scripted | 10 | 10 | 100% | 7.0 | +1.00 | 29.0 |
| playbook | 10 | 0 | 0% | 8.0 | −13.7 | 48.0 |
| random | 8 | 0 | 0% | 8.0 | −3459.8 | 41.2 |
| linear | 8 | 0 | 0% | 8.0 | −1646.0 | 42.2 |
| mcts@64 | 8 | 0 | 0% | 8.0 | −4033.4 | 39.5 |
| mcts@256 | — | not run — ~1 hr est. at 3.4× the @64 wall | | | | |
| mcts@1024 | — | not run | | | | |

## Read

**Verdict: lookahead at 64 sims/move does NOT close the valley** — mcts@64 reproduces the greedy wall exactly (0/8, mean rank 8.0, stiff rank-8 repairs). 840 s for 8 episodes; @256+ deferred (wall ∝ sims). The Strassen valley stays outside search-at-this-scale — consistent with Q3: density/look-ahead is the frontier, not machinery.

## The mechanism finding

Independent of the win/lose column, the probe answers *where*
the valley lives.  Instrumented rollouts (150 runs, ~750 valid
states visited) found the board's rank-reduction mechanism —
the collapse — is essentially absent exactly where it would
matter:

| rank of valid state | states visited | with a collapse move |
|---|---|---|
| 8 | 351 | **0** |
| 9 | 154 | 152 |
| 10 | 124 | 124 |
| 11 | 57 | 57 |
| 12+ | 65 | 65 |

Every valid scheme the repair lands on at rank 8 is *stiff*: no
two committed terms share proportional factors in two modes, so
no flip can zero a factor and `drop` can never fire.  Collapse
moves are ubiquitous one rank up — they exist almost only in the
over-committed region, where each one can only shave a bloated
scheme back toward 8, never below it.  And a deeper 2-ply check
(363 valid rank-8 states × every legal flip) found not one flip
that produces a droppable or collapse-able configuration: the
rank-8 neighbourhood reachable by dig/repair contains no path to
rank 7 at all.

That reframes the blocker.  It is not that one-step lookahead
fails to *price* the valley — it is that the reachable repair
basin contains no rank-7 scheme at all.  The only valid rank-7
decompositions are Strassen's (de Groote: essentially unique),
and assembling one requires committing seven coordinated
weight-2 terms against an L1 gradient that always prefers the
naive unit patches (a Strassen term scores ~12 on `‖R − t‖₁`
where a unit term scores 7).  Rollout-level discovery would need
the sampler to pick the globally-coherent set, not the locally
cheapest term — exactly the prior the board doesn't have.

## The honesty caveats

1. Same scope caveat as the first document: this is rediscovery
   inside a bilinear move vocabulary, not discovery from raw
   arithmetic.
2. The rollout policy is a design choice, not a neutral one —
   dig/repair + collapse-taking is the strongest honest cycle we
   found, and it *still* cannot cross.  A weaker rollout (uniform
   random) would only stall earlier; the finding is about the
   reachable basin, not about this rollout's skill.
3. Sims budgets: 64/256/1024 per move.  At ~30 ms per rollout,
   one 48-move episode costs ~2 min (64), ~8 min (256), ~33 min
   (1024) of single-thread compute — the 1024 arm was run on the
   canonical case only.  Scaling past this on this host is a
   hardware question, and the mechanism analysis above predicts
   it changes nothing.
4. `best_rank` is recorded by the *real* board — a rollout that
   stumbles onto a win only counts if the committed line
   actually walks there (the re-rooted tree preserves it).

## Implications

(filled after the run)

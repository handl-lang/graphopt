# Strassen rediscovery spike — the Q2 go/no-go measurement

**Date:** 2026-11-06 · **Board:** `mlopt_discovery.bilinear_game` ·
**Driver:** `bench/drivers/probes/strassen_probe.py` · **Seed:** 0
throughout · **Runner:**

```bash
.venv/bin/python bench/drivers/probes/strassen_probe.py \
    --episodes 100 --budget 48
.venv/bin/python bench/drivers/probes/strassen_probe.py \
    --episodes 60 --budget 96
```

## The question

Q2 is the falsifiable "can it find new algorithms" test: put the
2×2×2 matrix-multiplication tensor's decomposition space on the
`play` registry as a move-graph and ask whether search — random,
greedy, or the learned linear arm — walks the naive
8-multiplication scheme to a valid rank-7 one (Strassen, the
proven optimum for this tensor).

## The board

State = a list of rank-1 terms `(uₖ, vₖ, wₖ)` plus the residual
`R = T − Σ uₖ⊗vₖ⊗wₖ` (exact `Fraction`s — `R = 0` ⟺ the scheme
computes `C = A·B`).  Cost = term count.  Moves are the classical
Brent-equation vocabulary:

* `cut(i)` — un-commit a term (the Las Vegas cancel: rank −1 but
  the scheme goes wrong until the residual is repaired);
* `pick(k)` — the add move, staged ×3 (u, v, w from the 32-vector
  `{0,±1}`-weight-≤2 factor dictionary — a flat add would
  enumerate 32³ per step);
* `abort` — abandon a pending pick;
* `flip(i,j,m,s,λ)` — the flip-graph move (auto-normalizing);
* `drop(i)` — remove a zero-factored term (the flip-graph
  "reduction");
* `rescale(i,m,n,λ)` — factor-norm rebalancing;
* `stop`.

Reward is the quality delta `−rank − 16·‖R‖₁`, telescoping to
`initial − final` rank over valid endpoints: the Strassen valley
(cut 8 terms, `‖R‖` climbs to 16+ before repair) is real and
priced.  The board auto-terminates on a verified rank ≤ 7 — the
win.

## Expressibility — the non-vacuity pin

The scripted line (`bilinear_game.strassen_line`) plays: cut
every committed term, then commit Strassen's seven terms through
staged picks.  **10/10 episodes, verified rank 7 in exactly 29
moves** (8 cuts + 21 picks), terminal certificate on — on the
canonical naive board *and* on every scrambled case (the line is
case-agnostic).  The move set expresses Strassen; a zero result
elsewhere is a property of the searchers, not the board.
(`tests/test_discovery_bilinear_game.py` pins this.)

## The measurement

`best valid rank reached`, paired episodes, budget 48
(100 stochastic episodes, 10 deterministic):

| arm | eps | won (rank ≤ 7) | pct | mean best rank | mean reward | mean steps |
|---|---|---|---|---|---|---|
| scripted | 10 | 10 | 100% | 7.0 | +1.00 | 29.0 |
| playbook (greedy repair) | 10 | 0 | 0% | 8.0 | −13.7 | 48.0 |
| random | 100 | 0 | 0% | 8.0 | −3459.8 | 41.2 |
| linear (cold spec) | 100 | 0 | 0% | 8.0 | −1646.0 | 42.2 |

Budget 96 (60 stochastic episodes) — same verdict:

| arm | eps | won | pct | mean best rank | mean steps |
|---|---|---|---|---|---|
| scripted | 10 | 10 | 100% | 7.0 | 29.0 |
| playbook | 10 | 0 | 0% | 8.0 | 96.0 |
| random | 60 | 0 | 0% | 8.0 | 69.8 |
| linear | 60 | 0 | 0% | 8.0 | 69.2 |

## Read

**Verdict: NO-GO.**  Nothing but the recorded line found rank ≤ 7
in 200+ measured episodes across both budgets.  The specifics
matter:

* *Greedy repair* (`playbook`) reliably reaches **valid rank 8**
  — it patches the residual it drills, and the cheapest L1
  repair of any cut is a naive-shape term.  It rediscovers the
  *naive* scheme — the local optimum — every episode.  The
  Strassen valley is real: the cut-add path dips to
  `−16·‖R‖` before the seventh add restores it, and no
  one-step-lookahead crosses that.
* *Random* and the *cold linear arm* wander the same valley; the
  telescoped reward only pays at the far end (+1 if it ever
  lands), so the gradient signal for crossing is absent until
  the crossing is nearly complete.  Doubling the budget changed
  nothing (the extra steps buy more valley, not more depth).
* The reward's *shape* is not the blocker — a cut-then-add path
  does bank +1 net.  The blocker is that the productive
  trajectory is a ~15-move coordinated sequence through invalid
  space with no positive prefix.

## The honesty caveats

1. **This tests rediscovery inside a move vocabulary built for
   bilinear algorithms** — factor dictionaries, rank-1 terms,
   Brent-equation identities — not from-scratch discovery.  A
   "yes" here would already be weaker than "the system invented a
   fast algorithm"; the measured "no" is correspondingly
   stronger.
2. **The naive → Strassen move path is long and indirect.**
   Prototyping showed the flip graph is nearly a collection of
   islands at rank 8 (the split-Strassen scheme offers only ~24
   legal flips at depth 1; forward BFS from naive sees ~120k
   states by depth 3 with no droppable term).  The reachable
   route is the Las Vegas detour — cut terms into the residual
   and re-commit — which is exactly what makes the search hard:
   the intermediate states are *worse* under any rank-count
   metric.
3. The factor dictionary (`{0,±1}` weight ≤ 2, the AlphaTensor
   alphabet) hands the players the Strassen factors by
   construction — and they still could not assemble them.  The
   bottleneck is search credit assignment, not vocabulary.
4. `linear` here is cold — `BILINEAR_SPEC` ships no trained
   prior — so its row is an untrained baseline, not the trained
   prior's best effort.

## Implications

The game machinery itself is sound — the scripted witness
traverses it cleanly and the referee certifies the rank-7 scheme
exactly.  What the spike shows is that *the episode-game
formulation does not create the discovery*: within a per-step
budget, neither uniform search nor greedy residual repair nor a
cold linear policy crosses the invalid-space valley between the
naive scheme and Strassen's.  Any future "yes" on this board
likely needs either a trained prior over move shapes (the
`linear` arm's untrained row is the control), a planner that
searches residual-space rather than playing one move at a time,
or the mint/discovery machinery feeding candidate macro-moves —
not more budget.

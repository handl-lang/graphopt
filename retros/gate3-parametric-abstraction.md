---
title: 'Retro — gate 3: parametric abstraction over the step (a measured FAIL, deeper than the last)'
type: retro
date: 2026-10-16
---

# Retro — gate 3: parametric abstraction over the step (a measured FAIL, deeper than the last)

**Commit**: `e4a193a` (catopt `main`).
**Decides**: ADR 0008 gate 3 → still **FAIL**, now with the *cause*
measured; ADR 0009 stage 4 stays unjustified on this evidence.

## What the gate was, and what was missing

The gate-3 retro (`gate3-loop-proposing-rule.md`) recorded that the
`unroll(step, n) → Loop(step, n)` proposing rule works (control PASS)
but fires **0×** on the search's winner, because the export's unrolled
recurrence is *parametrically* regular (each block's step differs by the
block index, `split(…, index=i)`) but not *syntactically* regular — there
is no shared `step` subterm to bind.  It named the missing piece:
**abstraction over the step** — a form-level / parametric move, not an
e-graph rewrite.

This task built that move and re-ran the protocol.

## What was built

`lab/bridge/index_abstract.py` (+ the `--abstract-index` arm in
`lab/bridge/combinator_probe.py`; 12 tests in
`tests/test_index_abstract.py`).

The move operates on the term's **per-block decomposition** — the leaves
of the `concat` chain a chunked scan's output is — and does four things:

1. **Decompose** — flatten the `concat` chain into the `K` block outputs;
   find the `split(…, index=i)` families.
2. **Abstract the index** — in each block, replace its *own* indexed
   `split` with a fresh **block-input parameter** (a wire).  The index is
   no longer a baked constant.
3. **Abstract the state** — replace the block's **recurrence state** (the
   outermost shared subterm that carries none of the block's own splits)
   with a fresh **state parameter**.  The state is no longer a baked
   subterm.
4. **Test regularity, then fold** — if all `K` abstracted templates are
   the *same* term, compile the shared template to a step morphism
   `S ⊗ X → S' ⊗ Y` and emit `unroll(step, K)` (the spelling
   `PROPOSE_LOOP` folds to `Loop(step, K)`); otherwise report the *first*
   structural difference, so the negative names its own cause.

**Vocabulary**: only `split`/`Arr`/`Leaf`/`Loop` — the abstraction binds
a constant as a **parameter wire**, so **no new IR node kind** is needed.
That answers the brief's "do not invent a node kind unless it is
necessary": a parameter is just a wire, and the presentation already has
wires and the `Loop` combinator.

The verdict requires *both* that the templates collapse **and** that the
fold reproduces the original blocks (`roundtrip`).  Templates agreeing is
necessary but not sufficient — a family can collapse onto the wrong
subterm and then fail to reproduce; that case is reported `irregular`
(see the `old_linear` control below), never as a false pass.

## The commands and the numbers

```sh
# gate 3, unchanged protocol, both gate-3 arms armed
.venv/bin/python lab/bridge/combinator_probe.py \
    --gate2 --extracted --propose-loop --abstract-index
```

```
-- projection (staged arm, T=64 K=4, 15s, 9755 enodes) --
   extracted term: dag=139  reads=2 chained=True
   projects: round-trip EQUAL  morph=1590 dup=92 copies=184
   fold-chain depth: spelled=3  extract=2
   the presentation's spelled scan is in the search's root e-class: False
-- gate 3: the `loop`-proposing rule --
   loop-propose: winner morph=1590 spelled-scan subterms (n>=2) found=0 ns=[]
      saturation: terms=1 fired={'loop_propose': 0, 'loop_expand': 0}  loop-headed reachable=False  best_is_loop=False  best_cost=1590
   control (authored spelled scan): spelled-scan subterms=1 ns=[4] fired={'loop_propose': 1, 'loop_expand': 0}  loop-headed reachable=True  best_is_loop=True  best_cost=69 (spelled 285)
-- gate 3: parametric abstraction over the step --
   abstract-index: blocks=7 split-families=2
      family sizes/dim=('(16, 16, 16, 16)', -2) indices=[0, 1, 2, 3]
      family sizes/dim=('(4, 4, 4, 4, 16, 16, 16)', -2) indices=[0, 1, 2, 3]
      verdict=IRREGULAR — the 7 blocks do NOT collapse to one shared step
      first difference: block1/block2/add[1]/mul[1]: op om_decayed vs add
   control (regular fold): blocks=4 verdict=regular round-trip=True  closure: best_is_loop=True best_cost=109
```

| seed | blocks | abstraction verdict | fold round-trips | closure |
|---|---|---|---|---|
| **winner** (the gate) | 7 | **IRREGULAR** | — (no fold built) | — |
| control — the *presentation's spelled fold*, expanded | 4 | regular | **True** | `best_is_loop=True`, best_cost 109 |
| synthetic `new` / `old_shielded` folds | 4 | regular | True | — |
| synthetic `old_linear` fold | 4 | irregular (agrees, does not reproduce) | False | — |

The control is the presentation's own spelled scan expanded
(`_expanded_fold_term(t, k)`) — the gate-2 program, a genuine regular
fold.  The move collapses it to one shared step, round-trips, and the
closure reaches a `Loop` (`best_is_loop=True`).  Measured at `K = 2, 4, 8`
(`T = 8, 64, 128`): **regular + round-trip in all three**.

## Verdict — FAIL, against the ADR's criterion

> **FAIL** = the winner is the spelled/unrolled form again → search did
> not reach `loop`.

The winner does **not** collapse.  The parametric abstraction — the move
the previous retro named — cannot make its `K` blocks one shared `step`,
and the move reports the first difference itself:

```
block1/block2/add[1]/mul[1]: op om_decayed vs add
```

i.e. at the **cross-term position** of the block, block 1 spells the
state contribution as `om_decayed(matmul(q, k_prevᵀ), D, v_prev)` (a
re-expansion of the state through the intra carrier) while block 2 spells
it as an additive nesting.  The blocks are not the same step modulo any
constant.

## Why — and why it is *deeper* than the last retro's diagnosis

The last retro blamed the split index alone ("each block's step differs by
`split(…, index=i)`").  The abstraction **removes the index** and the
blocks still do not agree, so the index was not the whole story.
Measured on the winner's 139-node DAG:

1. **The chunking is not uniform.**  The output is a `concat` of **7**
   blocks of **two** sizes — four of size 4 and three of size 16
   (4·4 + 3·16 = 64) — drawn from **two** `split` families
   (`sizes=(4,4,4,4,16,16,16)` and `sizes=(16,16,16,16)`).  The step's own
   `t`/`sizes` attrs differ across blocks; a `Loop` has one step shape.
2. **The state spelling is non-uniform.**  Extraction mixed two ways of
   reading the recurrence state: the *shared-state read* `matmul(q, S)`
   and the *intra re-expansion* `om_decayed(matmul(q, k_prevᵀ), D, v_prev)`
   (the state `S = k_prevᵀ v_prev` re-expanded through the carrier).  The
   blocks do not share a state subterm, so there is nothing to bind.
3. **The block outputs are pairwise distinct modulo the varying integers.**
   Independent check: canonicalising every block modulo
   `(index, t, offset, sign)` leaves **7 distinct** blocks (and distinct
   even modulo *all* subterms shared with another block).  So no
   constant-parameter abstraction can collapse them.

The presentation's whole advantage — a *single shared* `step` — is
exactly what the flat DAG erased, and the erasure is **not only the
index**: it is the index, the chunk size, and the cross-term spelling at
once.  Removing the index (this task) is necessary but not sufficient.

## What it does and does not prove

**Does prove:**

* The abstraction the last retro named **exists and is expressible in the
  existing vocabulary** — a parameter is a wire; `split`/`Arr`/`Leaf`/
  `Loop` suffice, no new node kind.  On a regular fold it collapses the
  blocks, round-trips, and the closure reaches a `Loop`.
* On the real gate-3 export the blocks **do not** collapse, and the move
  names *why*: the cross-term spelling (plus the uneven chunking).  So
  gate 3 is still FAIL, for a reason strictly deeper than "the rule was
  too narrow".
* Therefore **the bridge is a presentation, not a search space** — the
  ADR's own FAIL branch, now with the missing structure measured.

**Does not prove:**

* That *no* move could ever work — only that abstraction over the
  **index** cannot, and that the remaining gap is the *state spelling*
  (two spellings extraction mixed) plus the *chunking* (two sizes).  A
  move that also normalises the state spelling is not ruled out; it is
  named here as the next piece.
* Anything about a `catopt-kernel` on other grounds.  It rules out one
  justification — "the presentation is a discovery substrate" — again.

## Honest limitation of the move

The state is identified structurally ("the outermost subterm that carries
none of the block's own splits").  That is right when the block reads the
state *through* an own-dependent node — which a real attention step does
(the state is read via a query, `q̃ @ S`) — and it is right when the state
update lives inside the block.  It is **not** right when the block reads
the state *linearly* (`o_i = c·S_{i-1} + b_i`): there the block-local
transform `c·S_{i-1}` is the largest own-split-free subterm, the templates
still agree, and the fold does not reproduce.  The move detects this
(`roundtrip` False → `verdict=irregular`, with a note) rather than
mis-folding it, and the limitation is a named residual, not a silent one.
A complete move would need the step's *arity* (which inputs are state),
which the DAG does not carry.

## Consequence for ADR 0009 stage 4

Unchanged and strengthened.  Stage 4 was gated on gate 3; gate 3 is still
negative, and this task shows the missing piece is **larger** than the
previous retro implied: not "a form-level proposer" but a move that
normalises the state spelling *and* the chunking, on top of abstracting
the index.  Stage 4 still rests on something else or on nothing.

## Cost

* ~40 s wall for the full `--gate2 --extracted --propose-loop
  --abstract-index` run (gate 2 verify + one 15–16 s staged saturation +
  both gate-3 arms); peak RSS well under 1 GB.
* The abstraction itself is trivial on the winner (7 blocks; the family
  check is sub-second) and on the control (4–8 blocks).
* New surface: one lab module (~660 lines), one test file (13 tests),
  ~35 lines of probe wiring.  No shipped-package change, no law-bank
  change, no `catopt_core.ir` node kind.

## Gates run

`.venv/bin/ruff check` + `ruff format --check` (clean),
`.venv/bin/ty check` (clean), `.venv/bin/vulture` (clean),
`.venv/bin/lint-imports` (4 kept, 0 broken),
`scripts/radon_ratchet.py` (ok), and
`pytest tests/test_index_abstract.py tests/test_loop_propose.py
tests/test_combinator_bridge.py` (35 passed).  The full suite was **not**
run (memory/time budget; no shipped-package change).

# Learned mint proposer — ranking and mutating candidates by measured viability

**Date:** 2026-10-23 · **Shipped:** `mlopt_discovery/learned_propose.py`
(module + `DEFAULT_WEIGHTS` bake), `scripts/train_learned_proposer.py`
(baker), `tests/test_learned_propose.py` · **Seed:** 0 throughout ·
**Runner:**

```bash
.venv/bin/python scripts/train_learned_proposer.py --lolo
```

The enumeration-side proposers generate by construction — every
grammar form, census tuple and fusion schema gets an episode slot and
the oracle sorts it out afterwards.  The overnight ledgers now carry
**406 measured candidate cells** across five arms
(`bench/results/checkpoints/overnight*.jsonl`), each with its
serialized law and its gauntlet stage list — a labeled dataset of
"which proposal shapes survive the oracle".  `learned_propose` is the
learned half of the mint on top of it: a torch-free logistic ranker
(`sigmoid(dot)` over a `str -> float` feature dict, plain SGD — the
gamekit `_dot`/`_grad_row` idiom) plus a mutation generator that
proposes variants of laws that already earned admission.

**Contract:** a learned proposal is a *prior*, never a verdict —
everything emitted still binds against the corpus, is alpha-deduped
against the shipped bank, and faces the full referee gauntlet +
novelty sieve.

## The labels — oracle-pass rate is wildly proposer-dependent

Label `y` = *survived the oracle leg* (the stage list cleared
`oracle`).  59/406 positive — **14.5% base rate** — but per family:

| family | n | oracle-pass |
|---|---|---|
| synth-composite | 21 | 19 (90%) |
| prim-fold | 8 | 8 (100%) |
| fusion-schema | 11 | 9 (82%) |
| algebraic-grammar | 7 | 6 (86%) |
| enumerate (all forms) | 248 | 16 (6.5%) |
| census-mixed-view | 104 | 0 (0%) |
| census-naturality / layout / slice / elementwise | 7 | 1 |

A ranker that merely learns "prefer composite/fusion/grammar-schema
over mixed-view census mints" already reorders the pool usefully;
structural features (depth, op histogram, shared/new metavariables,
`rhs`-is-leaf, root-op) carry the rest to shapes with no family
signal — which is exactly what the mutation generator emits.

## Featurization

Three blocks (see `featurize`): `fam:<family>` +
`enum:<grammar-family>` one-hots (parsed from the
`enumerate:<fam>:<n>` naming); `lhs:*`/`rhs:*`/`pair:*` structural
statistics (node/leaf/metavar/attr-metavar counts, depth, shared- and
new-rhs metavariables, `rhs` shrinks / is-leaf / same-root /
alpha-equality, `log1p`-scaled); `op:<name>` counts +
`top:<lhs-root-op>` — the vocabulary the pattern actually spells.

## Ranking — measured

`scripts/train_learned_proposer.py` (406 rows, 25% seeded holdout):

| split | n | base rate | AP | P@10 | P@25 |
|---|---|---|---|---|---|
| train | 304 | 0.138 | 0.984 | — | — |
| **held-out** | 102 | **0.167** | **0.832** | **1.000** | 0.520 |

Leave-one-ledger-out (train on the other arms — the cross-mint
generalization check, since each file is a different mint arm's
distribution):

| held-out ledger | n | base | AP | P@10 |
|---|---|---|---|---|
| overnight_prim | 8 | 1.000 | 1.000 | 1.000 |
| overnight_real | 102 | 0.284 | 0.909 | 1.000 |
| overnight_real_guarded | 19 | 0.105 | 1.000 | 0.200 |
| overnight_xmodule | 29 | 0.138 | 1.000 | 0.400 |
| overnight (grammar arm) | 248 | 0.065 | 0.142 | 0.200 |

The mint-corpus arms transfer almost perfectly; the pure-grammar arm
is the hard generalization (its 16 positives are thin, and its form
one-hots are the memorized feature) — AP 0.142 is still ~2× its 0.065
base rate.  The baked table is the all-406-row fit (96 nonzero
weights in `DEFAULT_WEIGHTS`).

### Live rerank — the corpus pool, real referee

`pipeline.propose(counts, roots, "derived", require_bind=True)` over
ten exported zoo modules mints a 25-candidate bound pool containing
exactly one oracle-survivor (`grammar:mul_distribute`, the
algebraically-true distributive fold).  Under `bench.referees.judge("oracle")`:

- **pool order:** the survivor sits beyond the top-12 → precision@12 = **0/12**;
- **`rerank(pool)`:** survivor at rank 1 (score 1.000) → precision@1 = **1/1**.

Same story per-module: `DeltaRuleBlock`'s bound pool had
`grammar:sub_to_add_dup` at position 9 → the model put it first.

## Generation — mutations of the known-good bank

`propose_learned(terms)` mutates the machine-shipped bank
(`MACHINE_SHIPPED` ∩ `ALL_RULES`, 13 laws) with five operators —
`op-sub` (same-arity, `OpMeta.arity`-gated), `arg-swap`,
`mvar-rewire`, `wrap`, `attr-drop` — deduplicated against the shipped
bank by alpha-key, bind-filtered to real subterms, ranked by the
model.  On the same zoo modules the mutant stream is large
(~900 raw mutants over one term) and the bind gate is what keeps the
pool honest (8–15 bound mutants per module).

**Measured oracle precision: 0/15 bound mutants passed** across
`GatedAttentionUnit`/`DeltaRuleBlock`/`FiLMHead`.  That is the
honest number and it is expected: near-neighbours of true laws are
almost never true — the same reason `census-mixed-view` measures
0/104 in the ledger.  The arm's value is *coverage*, not precision:
it mints claims the fixed grammar cannot spell (rewired
metavariables, substituted ops, unguarded attr-relaxations), at a
bounded cost (bind-filtered before an episode slot is spent, deduped
before the sieve), and the ranker already orders them below the
high-scoring pool.  Where it earns is as a *frontier* — a mutant
that does hit is a claim no enumeration family could have produced;
whether that yield beats the grammar's 6.5% needs a bigger run than
this box-sample.

## Integration seam (for `bench/drivers/overnight.py`)

Not wired — the driver is owned elsewhere.  Three seams are ready:

```python
from mlopt_discovery import learned_propose, open_game

# --proposer learned: mix mutation-minted candidates into the pool
extra = learned_propose.propose_learned(
    terms, census_op=dict(counts), top_k=page // 4, seeds=None
)  # -> list[Proposal], family="learned-mint" — same element type
   #    as pipeline.propose; wrap in Candidate like _RealArena does

# ordering: rerank any pool by predicted viability (blend>0 keeps
# an exploration tail)
pool = learned_propose.rerank(pool, blend=0.15)

# eval-ordering inside the open game: feed the open:next-val part
open_game.set_value_preds(learned_propose.value_fn())
```

`propose_learned`'s signature mirrors `pipeline.propose`
(`terms`, `census_op=` accepted-and-ignored) so the seam is a
`pool.extend(...)`; `seeds=` overrides the bank (e.g. ledger-replayed
true laws via `load_ledger_rows(paths, with_rules=True)`).

## Regenerating the bake

```bash
.venv/bin/python scripts/train_learned_proposer.py --lolo
# prints DEFAULT_WEIGHTS literal → paste into learned_propose.py
```

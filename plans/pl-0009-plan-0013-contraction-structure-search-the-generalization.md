+++
id = "PL-009"
title = "Plan 0013 — contraction-structure search (the generalization)"
status = "open"
priority = "medium"
+++

---
title: Plan 0013 — contraction-structure search (the generalization)
type: plan
date: 2026-10-14
status_note: |-
  draft.  The one-line version: stop rewriting op-trees;
  search the *contraction structure* of the model's string diagram.
---

# Plan 0013 — contraction-structure search (the generalization)

## The reframe

Every existing law is a special case of "contract the diagram
differently":

- pairing = merge two fan-in branches at a shared leaf
- weight-chain fold = contract the param-only subtree first
- KV latent share = restructure a family to contract through a
  shared factor
- morphism windows = contract a block subsequence to a single
  morphism

The general problem — *find the optimal contraction/composition of
the model's monoidal diagram* — subsumes them and is NP-hard in
general (optimal tensor-network contraction), but the morphism
layer already collapses blocks to signature objects, so the search
runs on a ~N-node graph, not a ~10k-enode graph.

## What this unlocks

Discovery instead of matching: "does this model contain an
MLA-style sharing" becomes a search answer, not a hand-written law.
The law library stops being the coverage bottleneck.

## Stages

1. **Diagram IR** — model → hypergraph/monoidal diagram over the
   morphism layer (blocks = nodes, wires = shared tensor edges,
   multi-arity edges included).
2. **Contraction moves** — a small move set (merge-projection,
   factor-shared-subtree, reorder-composition, split-shared-leaf)
   operating on the diagram; each move reifies to existing certified
   term rewrites (witness machinery reused).
3. **Search** — bounded lookahead + e-graph over diagram states
   (morphism-level, polynomial); cost model prices the reified term.
4. **Verification** — every delivered move is still a certified
   rewrite; bounded mode composes.
5. **RL guidance (later)** — once the space exists, a learned
   policy over move selection; deterministic verify = clean reward.

## Honest risks

- Contraction-order search is NP-hard; morphism collapse is the
  answer but coverage is signature-bounded (multi-input blocks
  needed first — in flight).
- The wins remain structure-conditional (census: stock dense models
  are thin) — the value is turning "does this model have structure"
  into a search result.

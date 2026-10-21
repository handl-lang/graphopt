# How mlopt works — syntax, structure, and the rewrite loop

<!-- provenance: graphopt:docs/mechanism.md@eea02f6 — summarized; the full account lives upstream -->

The model graph is **syntax**; a category-theoretic **structure**
interprets it; the structure's **axioms** generate rewrites; **search**
picks the cheapest; the **certificate** proves it.

## The tower

A program is a **1-morphism** between objects (tensor types); a
rewrite law is a **2-morphism** between programs; a *derivation* — one
law proven from premises — is a **3-morphism** (a coherence; the
machinery measures confluence and divergence between laws). Declared
objects — folds, lifts, compositions — are machine-constructed cells
written as data. Two games are played on this tower: **search**
applies 2-cells inside one equivalence class; **construction** writes
new cells (played as episodes on arena boards — see
[the game](./the-game)).

## The loop, step by step

Running example: the recurrence `h_t = A·h_{t-1} + x_t`.

1. **Syntax** — the graph as written: ops and wires, no meaning.
2. **Structure** — a signature (the *shape*: "a fold over `(A, b)`
   pairs") plus axioms (the equations it obeys: associativity + unit).
   *Example: the monoid `aff(A, b)`.*
3. **Match (the lift)** — recognize that a piece of the graph has the
   signature's shape. Matching runs on the **signature**, never the
   axioms.
4. **Rewrite** — the axioms become rewrite rules. Side conditions
   (`check`/`derive` hooks) are the soundness boundary.
5. **Saturate** — apply the axioms to a fixed point. The e-graph *is*
   this closure — the consequences, not a search over them.
6. **Extract** — pick the cheapest program the backend can run.
   *Example: the Blelloch tree.*
7. **Certify** — the path original → extracted is a replayable
   certificate (a 2-cell).
8. **Evaluate** — describe and price the extracted program. Evaluation
   *ranks*; it never changes what is equivalent.

```text
syntax → structure(signature + axioms) → match(signature)
       → rewrite(axioms) → saturate → extract → certify → evaluate
```

If the structure names the target directly, **construct** it and skip
saturate/extract.

## Two consequences worth stating

- **Laws are entailed by the structure**, in two tiers: *axioms*
  (defining equations) and *consequences* (theorems derived from
  them). The split is measured on the shipped library, not asserted —
  `rule.kind` marks each member `axiom`/`lemma`/`redundant`, and
  `rule.derivation` records the premises proving each non-axiom.
- **Choosing the interpretation is the discovery act.** The same graph
  reads as plain composition, a comonoid, a low-rank approximation, or
  a contraction problem — each exposing different laws.

## The meta-layer — laws about laws

- **Coherence (3-cells).** `mlopt_discovery.coherence`/`coherence2`
  measure the pair layer over the shipped library: critical pairs,
  derivability edges, mediators. The derived layer is *flat* — every
  derivation is a self-supporting cycle plus one emergent law — and
  "derivable" is not "disposable": removing a derivable law costs real
  bounded reach.
- **Discovery (the closed loop).** `mlopt_discovery.pipeline` runs
  census → propose → verify → measure → rank → emit over a corpus of
  real exported models; adversarial witness generation
  (`gap_gen`) produces bindings the corpus cannot, and an sqlite
  evidence store caches verdicts keyed by corpus × rules × revision.
- **Minting (ADR-0007).** Four layers grow the object language itself:
  definitional atoms (anti-unification), handler-interpreted atoms,
  law-theoretic atoms, and fenced induced atoms — all under one
  admission [gauntlet](./the-gauntlet).

## The kernel, and the platform next door

The same calculus is implemented as a *language* by the handl
platform: laws emit as `TransformDef` data rows, handl's rewrite
engine fires them, and a mlopt law produces exactly the term mlopt's
own `EGraph.apply_rule` produces. The seam is **data in, terms out** —
the `mlopt:` extension field on every row keeps the two
representations from drifting silently.

## Where this lives

`mlopt_core` (IR, `EGraph`, laws, `verify_certificate`, cost models) ·
`mlopt_orchestrator` (`Optimizer`, `search`, `lower`) · `mlopt_torch` /
`mlopt_onnx` (the `Source`/`Sink` ports) · `mlopt_discovery` (the
meta-layer) · `bench/drivers/` (measurement probes).

The full account — with the measured asides — is
[`docs/mechanism.md`](https://github.com/handl-lang/graphopt/blob/main/docs/mechanism.md)
upstream. See also [evaluation](./evaluation) and
[the three costs](./three-costs).

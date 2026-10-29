---
title: 'which representation is canonical: the calculus for laws, the op-graph for execution'
type: adr
date: 2026-10-14
status: proposed
---

# ADR 0011 — which representation is canonical: the calculus for laws, the op-graph for execution

**Depends on**: ADR 0002 (categorical re-expression), ADR 0003 (four
dimensions), ADR 0008 (combinator presentation — *and* its two-way-drift
warning), ADR 0009 (the ladder; stage 3 names this decision), **ADR 0010
(one language, many interpretations)**.
**Informed by**: spike Q0 (catopt's vocabulary assembles as handl data —
PASS, `handl b20b978a`, catopt `8d5b47d`) and spike Q1a (a tensor crosses
by reference as a `ForeignResource`).

## Context

Two representations of the same programs coexist:

1. **The op-graph IR** (`catopt_core.ir`) — what the sinks lower, what the
   cost model prices, what the certificate replays against.
2. **The arrow calculus** — `lab/bridge/combinator.py` today, handl's
   `Term` + `Language` in the direction ADR 0010 adopts.

ADR 0008 warned that two *peers* drift: every law must exist in both, or
one becomes second-class.  ADR 0009 deferred the question ("decide the
canonical representation — its own ADR") rather than answering it.  The
spike has now measured the facts the decision needs:

- **Q0 (PASS)** — catopt's vocabulary *is* handl data: a `Language`
  assembles from catopt-emitted fragments; a law lands in
  `lang.transforms` with its fields intact; the `catopt:` extension field
  is ignored by handl's serde and therefore rides along **losslessly**.
  Two representational requirements were found and fixed (fold `when`
  into `And([...])`; spell enum variants as YAML tags).
- **Q1a (yes)** — a tensor crosses **by reference**: `TaggedValue::Resource(Rc<ForeignResource>)`
  is an opaque host pointer with a finalizer.  No copy, no JSON.  So
  tensor semantics can live outside the kernel entirely.

## Decision

**Two representations, but not peers: each is canonical for a different
question, and the bridge carries an authoritative record so neither is a
lossy view of the other.**

1. **The calculus is canonical for *rewriting*.**  A law is a
   `TransformDef` row (data) — the user-facing rule "laws are exclusively
   data" lands here.  Guards, conditions and derivations are the
   calculus's business; handl's rewrite/typecheck engines read this
   representation.  *Rationale*: the engines that matter for discovery
   read it, and it is the representation that makes laws diffable,
   versionable and mintable.
2. **The op-graph IR is canonical for *execution*.**  Lowering, cost,
   verification against the source, and every executor (generic,
   batched, fused, graph, Triton) speak the op-graph.  *Rationale*: the
   sinks are written against it, and Q1a shows tensors are host
   resources — there is nothing to gain by making the calculus carry
   execution, and a great deal to lose (every sink would need a second
   implementation).
3. **The `catopt:` extension field is the authority.**  The bridge
   already states this ("lossless by construction"): the handl-shaped
   fields are a structural view, the `catopt` field is the record, and
   `cell_from_handl` reconstructs the cell exactly.  So the two
   representations cannot drift *silently* — a drift is a failed
   round-trip, which is testable.
4. **Execution stays catopt's; rewriting becomes handl's.**  handl's exec
   engines are an interpretation of *handl's* terms (used by its rewrite
   and typecheck passes); tensor execution stays in catopt's executors.
   This is not the "fallback" of ADR 0010 — it is the decision: the
   kernel is the *language and its rewrite machinery*, not the tensor
   runtime.
5. **Laws must be expressible in the calculus** (the data-only
   direction), with the procedural-hook count as the ratchet.

## Falsification

1. **The bridge's refusal rate is the metric.**  If laws cannot be
   expressed in the calculus vocabulary, the calculus is not canonical
   for laws.  The first measured gap is already known: catopt's leaf
   step (`{"catopt:leaf": …}`) has **no counterpart** in handl's
   `RewriteStep`.  If that gap cannot be mapped (candidate: a `Create`d
   node carrying the value in its `payload`, or `Fresh(hint)` for
   variables), then laws with literal leaves cannot be data, and the
   decision above is wrong for them.
2. **Round-trip is the seam test.**  `store → handl → cell` must be
   identity on the shipped bank; a failure is a drift, not a detail.
3. **If execution ever needs the calculus**, the decision in (2) is
   wrong — but that must be shown by a *measured* need (a form the
   op-graph cannot express), not by symmetry.
4. **Data-only must not raise the procedural-hook count** (ADR 0010's
   ratchet).  A law that cannot be data stays procedural and is counted.

## Honest scars

- **This decision is only as good as Q1b.**  Q1a shows the *channel*
  exists (a resource variant); it does not yet show a tensor op crossing
  it end-to-end through exec.  If Q1b fails, the calculus still cannot
  *run* anything tensor-shaped, and (4) becomes the split by necessity
  rather than by choice.
- **The type of a resource is by convention.**  Nothing in handl
  distinguishes a tensor from a file descriptor; the host must.
- **Two representations still cost something.**  The bridge, its tests
  and the round-trip discipline are permanent overhead.  They are worth
  it only because the alternative — laws in Python and the calculus as a
  notation — is what the project is trying to leave behind.
- **The op-graph's authority for execution is contingent on the sinks.**
  If a future backend speaks terms natively, (2) should be revisited.

## Consequences

- **"Laws are exclusively data" becomes implementable** with a defined
  home: the calculus, loaded as fragments, with the `catopt` record as
  the authority.
- **The e-graph is one strategy** over the calculus's laws, not the
  architecture.
- **The kernel does not need to run tensors** — which is what makes
  ADR 0010's "handl is the kernel" cheap: no tensor runtime to port, no
  second set of executors.
- **The seam has one test**: round-trip identity on the shipped bank.

## Outcome (spike, 2026-10-14)

The decision is now backed by measurements rather than symmetry:

* **The calculus side works for the whole bank.**  83 laws load as
  `TransformDef` rows, and catopt's own seam test proves the bridge is
  lossless in both directions on the shipped bank (no cell silently
  dropped; every row re-emits byte-identically).  So "laws are data in
  the calculus" is not aspiration — it is the current behaviour of the
  bridge, one mapping away from being the *only* representation.
* **The op-graph side is untouched and needs to be**: no measurement
  asked execution to move, and Q1b showed the boundary is JSON-valued,
  so moving it would buy nothing and cost every sink a second
  implementation.
* **The authority rule held**: the `catopt:` extension is ignored by
  handl's serde and round-trips exactly, which is what makes the two
  representations non-drifting *by construction* rather than by
  discipline.
* **The named refusal metric moved**: the leaf gap (the bridge's first
  measured refusal) is closed; the leaf now maps to `!Fresh` or a
  `!Create` of a declared kind.

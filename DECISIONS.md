---
title: Decisions
type: index
---

# Decisions

One line per decision rule, generated from each ADR's `## Decision`
section with `host/scripts/gen-decisions.py`. The ADR is the argument;
this is the recall surface (ADR-0161 §8). Superseded records are excluded.

### The categorical re-expression thesis

- **Name your structure.** Every capability must name the — [ADR-0002](adrs/0002-categorical-re-expression-thesis.md)
- **The certificate is the artifact.** The deliverable is a — [ADR-0002](adrs/0002-categorical-re-expression-thesis.md)
- **Directed search, not learning.** You never enumerate the — [ADR-0002](adrs/0002-categorical-re-expression-thesis.md)
- **Construction primary, saturation fallback.** Where the — [ADR-0002](adrs/0002-categorical-re-expression-thesis.md)
- **Exact and bounded are one object.** Exact rewrites are the — [ADR-0002](adrs/0002-categorical-re-expression-thesis.md)
- **Scope claims to structure.** Parity where Inductor fuses — [ADR-0002](adrs/0002-categorical-re-expression-thesis.md)

### Evaluation is an independent dimension

- **Evaluation splits from execution.** A Profiler observes a — [ADR-0003](adrs/0003-evaluation-is-an-independent-dimension.md)
- **Keep the frontier; do not scalarize early.** The candidate set — [ADR-0003](adrs/0003-evaluation-is-an-independent-dimension.md)
- **One reference hardware backend, not one architecture.** The — [ADR-0003](adrs/0003-evaluation-is-an-independent-dimension.md)
- **Policy is a replaceable port, RL included.** RL is a *policy* — [ADR-0003](adrs/0003-evaluation-is-an-independent-dimension.md)
- **The performance model is a port.** Initially analytical, later — [ADR-0003](adrs/0003-evaluation-is-an-independent-dimension.md)
- **Benchmarks are last.** Rewrite benchmarks only after the — [ADR-0003](adrs/0003-evaluation-is-an-independent-dimension.md)

### Abstraction as a legal move

- **Objects-as-data.** An "introduced abstraction" is a record: — [ADR-0004](adrs/0004-abstraction-as-move.md)
- **derive= moves toward data.** Most shipped derives are — [ADR-0004](adrs/0004-abstraction-as-move.md)
- **The player's action space is *construction operations*, not — [ADR-0004](adrs/0004-abstraction-as-move.md)
- **The verifier for a synthesized abstraction is the adversarial — [ADR-0004](adrs/0004-abstraction-as-move.md)
- **The four dimensions hold.** Semantics decides equivalence; — [ADR-0004](adrs/0004-abstraction-as-move.md)

### The open game: human-free discovery inside anchored anchors

- **Three anchors, nothing else.** (a) *The substrate* — torch, — [ADR-0005](adrs/0005-the-open-game.md)
- **The VM picture is already the architecture.** Ops are — [ADR-0005](adrs/0005-the-open-game.md)
- **The alphabet stays open by construction.** New vocabulary — [ADR-0005](adrs/0005-the-open-game.md)
- **The learned prior guides a verified search — RL does not — [ADR-0005](adrs/0005-the-open-game.md)
- **The four dimensions hold at the meta level too.** A — [ADR-0005](adrs/0005-the-open-game.md)

### ONNX as a second backend, in two directions

- **The op-mapping long tail.** The torch backend needed ~260 — [ADR-0006](adrs/0006-onnx-backend.md)
- **Shape inference.** The IR's typing is explicit; ONNX graphs — [ADR-0006](adrs/0006-onnx-backend.md)
- **Opset drift.** ONNX ops change semantics across opsets; — [ADR-0006](adrs/0006-onnx-backend.md)

### Combinator presentation of the IR (monoidal + trace)

- Terms are morphisms of a symmetric monoidal category with Δ — [ADR-0008](adrs/0008-combinator-presentation.md)
- A bridge, both directions: dag → combinator (fan-out becomes — [ADR-0008](adrs/0008-combinator-presentation.md)
- Laws remain equations — now between combinator expressions — [ADR-0008](adrs/0008-combinator-presentation.md)
- **Carrier atoms as loop instances.** applyd/om_decayed/ — [ADR-0008](adrs/0008-combinator-presentation.md)
- **Weight-space laws as ordinary rewrites.** Params as morphisms — [ADR-0008](adrs/0008-combinator-presentation.md)
- **Sharing as structure.** Δ makes fan-out explicit — the — [ADR-0008](adrs/0008-combinator-presentation.md)
- **A uniform discovery space.** Proposers enumerate morphism — [ADR-0008](adrs/0008-combinator-presentation.md)

### The machinery is domain-general; tensor programs are the first domain

- the substrate is a general arrow calculus (kernel/host) — [ADR-0009](adrs/0009-domain-generality.md)
- the discovery loop (enumerate → measure → truth → novelty → pay → — [ADR-0009](adrs/0009-domain-generality.md)
- the game layer's boards are injected per-domain (DomainSpec = feature — [ADR-0009](adrs/0009-domain-generality.md)
- its operations as host functions (the leaf vocabulary), — [ADR-0009](adrs/0009-domain-generality.md)
- its laws expressible in the calculus (the base alphabet), — [ADR-0009](adrs/0009-domain-generality.md)
- a cost signal (the score) — [ADR-0009](adrs/0009-domain-generality.md)

### the kernel calculus as a package, and the layer ladder

- **The naming rule (the two axes).** exec- is the **role**, the domain name is the **package**; the domain name wins.… — [ADR-0009](adrs/0009-kernel-calculus-and-layers.md)
- **Ports to add in core (see Revision 2 — most already exist).** Each lands **with a conformance test that the current… — [ADR-0009](adrs/0009-kernel-calculus-and-layers.md)
- **Revision (2026-10-14) — what the first draft got wrong.** Writing the first draft made two weaknesses visible, and… — [ADR-0009](adrs/0009-kernel-calculus-and-layers.md)
- **Revision 2 (2026-10-14) — most of the "ports to add" already exist.** Checking the list against the code before… — [ADR-0009](adrs/0009-kernel-calculus-and-layers.md)
- **Staged plan (revised order).** 1. **Ports + conformance tests** — cheap, no behaviour change, no import churn. 2.… — [ADR-0009](adrs/0009-kernel-calculus-and-layers.md)

### one language, many interpretations (the VM *is* the language)

- **One language.** A single Language value whose tables are **data**: — [ADR-0010](adrs/0010-one-language-many-interpretations.md)
- **Interpretations are engines over that data.** eval (tensor — [ADR-0010](adrs/0010-one-language-many-interpretations.md)
- **The kernel is the shared term type.** The arrow-calculus Term — [ADR-0010](adrs/0010-one-language-many-interpretations.md)
- **Use handl *as* the kernel — do not reimplement it.** The first — [ADR-0010](adrs/0010-one-language-many-interpretations.md)
- **Laws become rows.** A law is a transforms entry (data), not a — [ADR-0010](adrs/0010-one-language-many-interpretations.md)

### which representation is canonical: the calculus for laws, the op-graph for execution

- **The calculus is canonical for *rewriting*.** A law is a — [ADR-0011](adrs/0011-canonical-representation.md)
- **The op-graph IR is canonical for *execution*.** Lowering, cost, — [ADR-0011](adrs/0011-canonical-representation.md)
- **The catopt: extension field is the authority.** The bridge — [ADR-0011](adrs/0011-canonical-representation.md)
- **Execution stays catopt's; rewriting becomes handl's.** handl's exec — [ADR-0011](adrs/0011-canonical-representation.md)
- **Laws must be expressible in the calculus** (the data-only — [ADR-0011](adrs/0011-canonical-representation.md)

### the stop decision: an eight-round characterization

- Features that see the decision-relevant quantity — [ADR-0013](adrs/0013-the-stop-decision.md)
- Credit assigned per-step, not per-episode — [ADR-0013](adrs/0013-the-stop-decision.md)
- Calibrated magnitudes, held out — [ADR-0013](adrs/0013-the-stop-decision.md)
- A delivery channel isolated from the ranking logits — [ADR-0013](adrs/0013-the-stop-decision.md)

### public API freeze (plan 0032 phase 1)

- **frozen** — the manifest tables (substrate → eqsat, game → — [ADR-0014](adrs/0014-public-api-freeze.md)
- **public, unfrozen** — other __all__ members (public today, not — [ADR-0014](adrs/0014-public-api-freeze.md)
- **internal** — everything else; _-prefixed compat re-exports stay — [ADR-0014](adrs/0014-public-api-freeze.md)

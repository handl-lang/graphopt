# ADR 0010 — one language, many interpretations (the VM *is* the language)

**Status**: proposed — **revised** (the first draft recommended porting the
pattern into Python; the revision uses `handl` *as* the kernel).
**A prerequisite**: ADR 0009's calculus stage does not proceed until this
model is fixed.
**Date**: 2026-10-14.
**Depends on**: ADR 0002 (categorical re-expression), ADR 0003 (four
dimensions), ADR 0004 (abstraction as move), ADR 0008 (combinator
presentation), ADR 0009 (the ladder and the kernel question).
**Prior art, verified**: the `handl-lang/host` (Peregrin) workspace —
`crates/handl-language` (`Language`), `handl-kit-arrow-lower`,
`plugin-engine-{exec,typecheck,transform-rewrite,fmt,lint,test}`,
ADR-0044 ("language as data"), ADR-0052 (eval schemes), ADR-0054 (type
rules), ADR-0081 (data-driven child roles), ADR-0094 (generic transform
engine).  catopt already bridges to it: `catopt_discovery.handl`.

## Context

ADR 0009 asks whether the calculus needs a runtime VM, and its first
draft implied *two machines* — a rewrite machine (apply 2-cells) and an
evaluation machine (run 1-cells).  That framing is wrong in a way that
matters, because it implies **separate languages** for types, runtime
and rewriting.

The `handl` platform settles the question empirically: it has **one**
language — `Language`, whose tables are all data (`tokens`, `grammar`,
`lexer_rules`, `eval` → arrow-calculus `Term` per node kind, `types` →
synth/check `Term`, `ast` → shapes and named child roles, `transforms` →
`(pass, kind, priority)` rules) — and **many** engines that read it:
lexing, parsing, execution (**three** strategies: tree-walk, bytecode,
cranelift JIT), typechecking, rewriting (`RewriteEngine` with
`BottomUp`/`TopDown`/`Fixpoint` traversals), formatting, linting,
testing.  A language construct is a **YAML fragment**; only the calculus
vocabulary (combinators, primitive forms, value shapes) is fixed in the
engine.

So `eval`, `typecheck` and `rewrite` are not three languages.  They are
**three interpretations of one language**, differing in the scheme table
they read and the globals bound at instantiation.

### The mapping to catopt is nearly 1:1

| handl | catopt today | dimension |
|---|---|---|
| `Language.eval` (kind → arrow `Term`) | carrier/op semantics — lowering to tensor ops | semantics / execution |
| `Language.types` | shape rules (`catopt_core.typing`) | semantics |
| `Language.transforms` (`TransformDef`) | **laws** (`Rewrite`: lhs/rhs/cond) | semantics / search |
| `plugin-engine-exec-{tree-walk,bytecode,cranelift}` | executors (generic, batched, fused, graph, Triton) | execution |
| `plugin-engine-transform-rewrite` (`Traversal`) | the **e-graph** — *one* rewrite **strategy** | search |
| `handl-kit-arrow-lower` | the `lab/bridge` combinator prototype | semantics |
| `merge_fragments` (YAML fragments → `Language`) | the law bank (currently Python) | — |

## Decision

**Adopt the interpretation model: one object language, many engines.**

1. **One language.**  A single `Language` value whose tables are **data**:
   `semantics` (kind → term), `types` (kind → shape/type rule),
   `shapes` (node shapes + named child roles), `transforms` (laws), and
   the cost/executor tables.  No separate language for types, runtime or
   rewriting.
2. **Interpretations are engines over that data.**  `eval` (tensor
   execution), `types` (shape inference), `rewrite` (laws — the e-graph
   is **one strategy**, alongside simpler passes), `lower` (to the
   op-graph the sinks consume), `cost`.  Each takes the `Language` and
   its own globals; none owns the syntax.
3. **The kernel is the shared term type.**  The arrow-calculus `Term`
   (composition, `⊗`, `Δ`/`dup`, `loop`/trace, `arr`, handlers) is what
   schemes compile *into* — one representation the interpretations
   agree on.  This replaces ADR 0009's `Machine` port with a
   **`Calculus`** (the term/rule structure) plus per-interpretation
   entry points.
4. **Use `handl` *as* the kernel — do not reimplement it.**  The first
   draft proposed porting the pattern into Python; that was wrong.
   `handl` **already is** the arrow calculus (`arr`, `flow`/`>>>`,
   `axis_flow`, `loop`, `if`, `op`/`handle`, `graded_arrow`), its
   language is data (ADR-0044), and it has the engines — rewrite (with
   traversals, traces, stats), typecheck, parse, fmt, lint, test, and
   **three** exec strategies.  Critically, it **supports host
   functions** (`crates/handl-runtime/src/vm_primitives.rs` — "host
   functions for the exec model's VM instruction set"; the conformance
   corpus has a case calling a foreign host function), so **catopt's
   tensor primitives are host functions**, not a reason to fork the
   kernel.

   catopt then contributes only what is genuinely catopt's: the
   **language fragments** (laws as `TransformDef` rows, shapes,
   semantics tables), the **host functions** for tensor ops, the
   **domain adapters** (`catopt-torch`/`onnx`/`cuda`) and the
   **evaluation dimension** (cost, profiles, meters), plus discovery and
   RL.  Precedent in this repository: `catopt-native` is already an
   optional Rust engine, excluded from the workspace, built with
   maturin — a Rust kernel is an established shape here, not a new kind
   of dependency.
5. **Laws become rows.**  A law is a `transforms` entry (data), not a
   Python constructor — the *data-only* principle, in a format that
   already has a proven engine behind it.  This is what gives minted
   laws a versioning story (ADR 0009's standing gap).

### Why this is a prerequisite

ADR 0009's calculus stage cannot be scoped without it: "promote the
calculus to a package" presumes we know whether the calculus is *a
language with interpretations* (this ADR) or *a second representation
beside the op-graph* (0009's original framing, which risks the drift
ADR 0008 warned about).  Fix the model first; then the package boundary
follows.

## Falsification

0. **The boundary spike decides the shape.**  Three unknowns, each
   measurable: (a) **exec boundary performance** — one host-function
   dispatch per tensor op versus catopt's batched/fused/graph paths (the
   kernel's job is *search/rewrite*; the hot loop is the *delivered*
   module, so this may not matter); (b) **embedding direction** — catopt
   embeds handl (a PyO3 peer, like `catopt-native`) or handl drives
   catopt as a plugin; (c) **the new verification seam** — term/fragment
   round-trip and semantics agreement, which handl's own suite cannot
   cover.  If the spike shows the boundary cannot carry tensor semantics
   faithfully or efficiently, the fallback is the split: handl owns the
   language + rewrite + typecheck, catopt owns the tensor exec
   interpretation.
1. **Interpretation leakage.**  If the interpretations cannot share one
   term type without special-casing — e.g. `eval` needs a distinction
   `types` cannot express, or a law needs a node kind the shapes table
   cannot declare — the single-language claim fails and separate
   languages are justified.  Measure: count the interpretations'
   kind-specific special cases; the number may not grow.
2. **Data-only must not reduce expressiveness.**  The **procedural-hook
   count** (guards/derives the DSL cannot express — measured today:
   ~28, via `lab/audit/dspec_audit.py`) is a **ratchet**: it may only go
   down.  If moving laws into data raises it, the data format is too
   weak.
3. **Performance must not regress.**  A generic interpretation is
   usually slower than a bespoke path.  The existing delivery paths
   (generic/batched/fused/graph/Triton) must stay at least as fast; a
   generic interpreter is *additional*, not a replacement, until
   measured.
4. **No interop drift.**  If catopt's `Language` fragments and handl's
   diverge — the bridge stops round-tripping — the seam is worse than
   no seam.

## Honest scars

- **`handl` is Rust + YAML with its own crate graph**, so using it as the
  kernel means a compiled dependency and a second toolchain.  It cannot
  live *inside* `catopt-core` (zero-dependency, `import-linter`-enforced);
  it is a **peer** — exactly the shape `catopt-native` already has.
- **The first draft was wrong** and is recorded rather than quietly
  dropped: porting the pattern into Python would have duplicated a
  proven engine and created the drift this ADR's own falsification
  forbids.
- **The current e-graph is op-graph-specific.**  Calling it "one rewrite
  strategy" is the *direction*; today it is the only one, and it is
  bound to the op-graph IR.
- **Two representations still coexist** (op-graph IR + calculus term)
  until the canonical-representation decision lands — this ADR fixes the
  *model*, not that choice.
- **The data format is unproven at bank scale.**  82 rules as rows is
  plausible; the guards/derives are where it will be tested.

## Consequences

- **The "laws are exclusively data" rule becomes implementable** — with a
  ratchet (the procedural-hook count) as the honest form of "exclusively".
- **The e-graph stops being the architecture.**  It becomes one search
  strategy behind a `rewrite` interpretation, so simpler passes and
  future strategies are peers.
- **New semantics = new schemes.**  Extending the vocabulary is a data
  change (plus, where genuinely new, a handler), which is what makes
  "the machine can create new semantics" a *mechanism* rather than a
  hope.
- **catopt and handl share one idea**, so a construct expressed in one
  has a target in the other — via the existing bridge.

## Outcome (spike, 2026-10-14)

The boundary spike (`plan 0029`) measured this ADR's central claim and
**it holds**:

* **The load works.**  `language_from_dir` assembles a `Language` from a
  directory of catopt-emitted fragments, and after closing two gaps the
  **whole bank loads — 83 laws** in `lang.transforms`, with the three
  minted leaf kinds (`catopt:const` / `catopt:var` / `catopt:data`)
  declared (`handl 9372bd32`, catopt `5ca8aef`).
* **The two gaps were representational, not semantic**: handl's
  `when` is ONE condition (fold the clause list into `And([...])`), and
  enum variants are YAML *tags* under `serde_yaml`, not JSON maps.  A
  third — the `pass:node_kind:priority` rule identity — required a
  load-time disambiguator, since catopt's saturation has no order to
  give.
* **The tensor channel exists but is not needed**: `TaggedValue::Resource`
  carries an opaque host pointer by reference, while the *host-function
  seam* is JSON-valued, so a tensor would cross as a debug process-local
  pointer or as copied bytes.  ADR 0011 keeps tensors out of v1
  entirely, which is why this is a footnote rather than a blocker.
* **Direction**: (A) — catopt embeds handl as an optional peer, the
  `catopt-native` pattern (`retros/handl-q3-integration-direction.md`).

What this ADR still does **not** have: a *built* peer, or a rewrite run
through handl's engine on a catopt term.  Those are sequenced in
`plans/0030-handl-peer-integration.md`.

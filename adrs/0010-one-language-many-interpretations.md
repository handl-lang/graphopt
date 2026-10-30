---
title: one language, many interpretations (the VM *is* the language)
type: adr
date: 2026-10-14
status: proposed
status_note: proposed — **revised** (the first draft recommended porting the
---

# ADR 0010 — one language, many interpretations (the VM *is* the language)

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


## Why

ADR 0009 asks whether the calculus needs a runtime VM, and its first draft implied *two machines* — a rewrite machine (apply 2-cells) and an evaluation machine (run 1-cells).  That framing is wrong in a way that matters, because it implies **separate languages** for types, runtime and rewriting.

## Not doing

- none recorded — legacy record; the design doc lists what was deferred.

## Consequences

- **The "laws are exclusively data" rule becomes implementable** — with a
- **The e-graph stops being the architecture.**  It becomes one search
- **New semantics = new schemes.**  Extending the vocabulary is a data
- **catopt and handl share one idea**, so a construct expressed in one

## Enforcement

- Code and tests are named in the design doc.

## Design

The full record (context, tables, phases, trade-offs): [`design/0010-one-language-many-interpretations.md`](../design/0010-one-language-many-interpretations.md).

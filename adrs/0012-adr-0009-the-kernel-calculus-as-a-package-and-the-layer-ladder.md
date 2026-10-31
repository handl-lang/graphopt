+++
id = "ADR-0012"
title = "ADR 0009 — the kernel calculus as a package, and the layer ladder"
status = "open"
priority = "medium"
+++

---
title: the kernel calculus as a package, and the layer ladder
type: adr
date: 2026-10-14
status: proposed
status_note: proposed — **revised** (see *Revision*, which reorders the
---


# ADR 0009 — the kernel calculus as a package, and the layer ladder

## Decision

1. **The naming rule (the two axes).** `exec-` is the **role**, the domain name is the **package**; the domain name wins.  `catopt-native` keeps its name: it is a *search* implementation (`Engine`), not a domain.
2. **Ports to add in core (see Revision 2 — most already exist).** Each lands **with a conformance test that the current implementation already satisfies** — ADR 0003's lesson: an unconsumed port is decoration.
3. **Revision (2026-10-14) — what the first draft got wrong.** Writing the first draft made two weaknesses visible, and one of them reorders the whole plan.
4. **Revision 2 (2026-10-14) — most of the "ports to add" already exist.** Checking the list against the code before writing any of it (ADR 0003's lesson applied to this ADR's own proposal):
5. **Staged plan (revised order).** 1. **Ports + conformance tests** — cheap, no behaviour change, no import churn. 2. **Fix the carrier inversion** — carrier *laws* to the semantic side, carrier *executors* into the domain package. 3. **Adopt the interpretation model** (ADR 0010) — one language, many engines; and **decide the canonical representation** (its own ADR) — op-graph IR vs arrow calculus.  *Then*, and only then: 4. **Promote the calculus** to a dimension-named package — **gated on ADR 0008 gate 3** (does search *reach* `loop`-headed forms?).  If the gate measures negative, the calculus is a presentation win and this stage is not justified.  **MEASURED 2026-10-16: the gate is negative.**  A `loop`-proposing rule over the presentation was built (`lab/bridge/loop_propose.py`, catopt `b7ba9ff`); it works on the presentation's spelled scan, but the search's winner has no regular spelled-scan subterm and the presentation's spelled form is not even in the search's root e-class — so the rule fires 0× and no `loop`-headed winner is reached.  **Stage 4 is therefore not justified by the presentation's discovery value**; it may proceed only on a different argument, and a *form-level* proposer (which abstracts the step, rather than a syntactic e-graph rewrite) is the piece of work that would re-open the gate.  See `project/retros/gate3-loop-proposing-rule.md`.  **Re-measured 2026-10-16 (later): that form-level proposer was built (`lab/bridge/index_abstract.py`, catopt `e4a193a`) — a parametric abstraction binding the index *and* the recurrence state as parameters.  It collapses a *regular* fold to one shared step and the closure reaches a `Loop` (control PASS), but the RetNet winner still does not collapse: its 7 blocks span two chunk sizes and extraction mixed two state spellings, so no constant-parameter abstraction can unite them.  Stage 4 is still not justified, and the missing piece is larger than this bullet implied.  See `project/retros/gate3-parametric-abstraction.md`.  **Re-measured 2026-10-16 (later still): still not justified.**  The block width was abstracted (`514888c`) and the winner is still 7 distinct templates spelling (the cross term is the unfolded geometric sum), not a parameter.  See `project/retros/gate3-sizes-abstraction.md`.** 5. **Extract `catopt-game`** — the RL layer out of the 50k-line package (the clear navigability win). 6. **Fold `genkernel` into a domain package** — no `catopt-kernels`. 7. Each stage updates the import-linter layer contract; **each new package gets its own coverage ratchet** (the discovery-style `--fail-under` climb), never a global widening.

## Why

Measured, this week:

## Not doing

- none recorded — legacy record; the design doc lists what was deferred.

## Consequences

- **The thesis becomes structural.**  "Programs are 1-cells, laws are
- **Discovery gains a uniform enumeration space** — morphism shapes
- **Execution adapters multiply cleanly.**  The domain axis stays
- **Learning becomes optional, not embedded.**  The RL layer can be

## Enforcement

- Code and tests are named in the design doc.

## Design

The full record (mechanism, tables, proofs, trade-offs): [`design/0009-kernel-calculus-and-layers.md`](../design/0009-kernel-calculus-and-layers.md).

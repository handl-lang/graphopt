---
title: 'Retro: domains-as-data — third cut (the code-shaped boards)'
type: retro
date: 2026-11-13
---

# Retro: domains-as-data — third cut

Follow-up to `domain-spec-second-cut.md`.  The last Stage-2 gap —
boards whose cases need code — closed wider than the gap note
expected: **all seven remaining `play` domains are spec-able**,
and the "code-shaped" label turned out to mean "has its own case
generator and player vocabulary", not "unreachable from data".

## What the third cut taught

**The boards were already data; only the names were missing.**
Every remaining domain already *was* a `(cases, board_of, spec,
scripted)` tuple in `play.DOMAINS`.  The work was not inventing
seams but *naming* the existing ones: `corpus.builders` for the
gen fusion builders, named pools (`_search_pool`,
`_fragment_pool`) so `corpus.kinds` can select terms, the
`vocab.rules` profile reinterpreted per board (a ruleset filter
on search/joint, a rewrite-name filter on fragment, an
admission check on gen/torch).  The expensive-looking boards
(`gen` with per-case kernel minting, `player` with nested
train+eval) needed zero new machinery — the spec just points at
the shipped generator.

**Player/incumbent coupling is the real invariant.**  The
subtle part was not cases but *who may sit at the table*: each
board's `DomainSpec` is bound to its state shape (`SearchState`
vs `FragmentState` vs spec-records vs tensors), so
`_BOARD_DEFAULTS`/`_BOARD_INCUMBENTS` make the pairing explicit
and refuse cross-board seating at load.  Five player specs and
four playbooks joined the registries lazily — `import
domain_spec` stays light until a spec actually parses.

**`vocab.rules` is board-relative, honestly.**  On meta-arena it
bounds per-case rulesets; on search/joint it *is* the seated
ruleset (filtered by name); on fragment it filters `vocab/`
rewrite names; on gen/torch it must merely admit DEFAULT.  One
profile field, four truthful semantics — documented per board
rather than forced into one meaning.

**The deliver seam generalized for free.**  `gen` cases carry
the same `ex` record torch cases do (IR, leaves, input, sink) —
`_torch_deliver` works unchanged once `"gen"` joins
`_TORCH_DELIVER_BOARDS`.  The only play.py surgery was threading
the ambient handler subset through `ex["ambient_handlers"]` so
`vocab.handler_tags` can narrow the gen board's store without
forking `_gen_board`.

## What is still code, and why that's correct

A *new* board — new case semantics, new player vocabulary, a
new nested objective — remains code by construction.  The seam
is the four registries (`BOARDS`/`_EXTRA_CASES`/`PLAYERS`/
`INCUMBENTS` plus the schema tables), the same shape kernelopt
uses for `play.DOMAINS`.  JSON names which shipped board to
seat; it does not pretend to describe one that doesn't exist.

## Numbers

* 7 boards seated; ~30 new spec fields validated per board.
* `tests/test_domain_spec.py`: 58 → 74 tests.
* Demo: `play --domain-spec gen-fusion --deliver` → verified
  module, max_abs 5.2e-06; `search-folds` plays clean at smoke
  scale.

Report: `docs/reports/2026-11-13-domain-spec-code-boards.md`.

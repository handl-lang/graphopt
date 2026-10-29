---
title: 'Plan 0022 — the cell store: one record, fragment-merged'
type: plan
date: 2026-10-14
---

# Plan 0022 — the cell store: one record, fragment-merged

## Motivation

The project spells "a cell as data" five ways — ``Rewrite`` laws,
``ConstructedObject`` declarations, ``OpDef`` ops, handler-table
dicts, and evidence-store rows — each with its own admission path
and bookkeeping.  The handl platform (``/home/luis/GitHub/handl-lang``)
proves the simpler shape at scale: language behavior is *data
merged at assembly* — fragments → validate → one store → generic
engines as views.  LWY's arrow calculus and Sanada's
operations-and-handlers extension are the published theory this
matches: an ``op`` cell is a Sanada operation; a ``handle`` cell is
a handler (one description, many interpretations — run it, price
it, verify it, lower it).

## Done — Phase 1: the record and the store

``catopt_core.cells`` (new):

- ``Cell{role, name, body, kind, provenance}`` — role ∈
  ``op | law | object | handle``; body is the role's native record
  (``OpDef`` / ``Rewrite`` / ``Rewrite`` / handler dict) so cells
  normalise keying and audit without re-encoding payloads.
- ``Provenance{origin, op, premises, note}`` — where the cell came
  from (``shipped`` / ``constructed`` / ``admitted`` / ``minted`` /
  ``declared``), flattening what ``ConstructedObject.construction``
  carried.
- ``Fragment`` — a named cell bundle; the merge unit.
- ``merge_fragments(*fragments, on_conflict=) -> CellStore`` —
  assembly with validation: duplicate ``(role, name)`` is a
  reported conflict (default: raise — silent override is how
  vocabularies drift); alpha-equal 2-cells across fragments are
  reported (subsumption is a discovery fact, not a schema error).
- ``CellStore`` views: ``laws()`` / ``ruleset()`` (the 2-cells for
  ``EGraph.run`` — law *and* object roles), ``objects()``,
  ``opdefs()``, ``handlers()``, ``to_data`` / ``from_data`` (one
  schema over the existing law/object/opdef codecs + a spec codec
  for handler patterns).
- Converters: ``law_cell`` / ``object_cell`` / ``op_cell`` /
  ``handle_cell`` / ``rewrite_of`` / ``laws_fragment``.

Verified: 26 tests, 100% coverage of the module, all gates green;
real shipped laws round-trip through the store by alpha-key.

## Next

- **Phase 1b — admission writes cells.** The construction moves
  (``fold`` / ``compose`` / ``claim`` / ``declare`` / mint) emit
  cells; the episode's store is ``merge(base_fragments,
  constructed_fragment)``.  ``ConstructedObject`` becomes a
  provenance-carrying view, not a parallel record.  The evidence
  store reads/writes cell records.
- **Phase 2 — engines over the store.** ``Optimizer``, the arenas,
  ``gen_probe`` / ``zoo_probe`` consume a ``CellStore``; the
  gauntlet admits into a fragment and re-merges.
- **Phase 3 — the handl seam.** 3a: serialize cells in a
  handl-compatible fragment shape (op ≈ shape def, law ≈
  transform def, handle ≈ native binding).  3b (deferred): handl
  as the language front-end, torch/Triton/carriers as ``!Native``
  handlers.

## Landed since

- `32997c6` Phase 1b — admission writes cells: `cellstore.py`
  adapters, provenance in `law_json`, `board.cellstore()`.
- `ef628d6` `om_decayed` — first NEW semantic atom (decayed
  attention element); verified on ChunkedRetention.
- `2eb78fd` Phase 2 (first bite) — `MetaArena(store=)`; handlers
  key by tag; `_gen_board` reads `ambient_store` + minted gen
  fragment.  gen-domain play behavior-identical.
- Compose sweep over the grown alphabet: 1,920 attempts → 27
  novel → 1 verified → 0 survivors.  `om_decayed`'s compose
  neighborhood is semantically atomic — its arg positions are
  meaningful, so depth-2 produces no new territory.  The next
  frontier is decay *factorization* (RetNet chunk recurrence:
  `D = γ^(i−j)` splits into intra-chunk + inter-chunk ratio) —
  a declared decay-matrix abstraction, the same "new atom"
  pattern as om_decayed itself.
- In flight: fused Triton kernel for `om_decayed` (codegen
  tier 3) — the runtime win attempt; reference executor
  currently ties eager.

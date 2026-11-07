# Fragment tower — phase 3 progress note

Plan: `project/plans/pl-0029-plan-0033-the-fragment-tower-mlopt-vocabularies-as-language-data.md`
(phase 3 — the fragment-emitting primitive mint; the `zoo-prim` vocabulary loop)
Host ADR: `host/project/adrs/0157-adr-0157-mlopt-on-the-fragment-tower-domain-vocabularies-as-language-fragments.md`
Ledger: `T-017` (open — phases 4–5 remain)
Sibling repo touched: `handl-lang/host-tensor` (`tensor.unsqueeze`)

## What phase 3 delivered

The mint is now a vocabulary *consumer and producer*: `vocab/` is
no longer a passive projection of Python registries — it is the
mint's own memory.  A `--corpus zoo-prim --prim-vocab-dir DIR` run
loads the directory's fragments before mining (fragment-loaded
primitives become mintable atoms, their `prim:*` folds join the
denoted gate, their minted families seed the handler grammar) and
writes every pooled mint back as fragments, so the next run composes
over this one's output.  Depth-2 remint is proven end-to-end: a
primitive emitted in round 1 seats its fold and feeds round 2's
pool.

### `vocabdir` — fragment emission API (phase-3 surface)

All emission is batched and merge-semantic — owner rows upsert by
key, machine-added fragments survive re-emission, and
`check_vocab`/`emit --check` stay fresh:

- `emit_prim(directory, name, record)` / `emit_prims(directory,
  items)` — one minted primitive writes its full fragment set:
  the `ops.yaml` row (with generator metadata — the op is
  materialized via `prim_fold_of` *before* the row is collected),
  the `laws.yaml` `prim:<name>` fold, the minted `handlers.yaml`
  row in the collector's own shape (`op`/`family`/`max_err`/
  `record`, no `source`), the `prims.yaml` record, the
  denotation + handler `models.yaml` rows, and the `fold:<name>`
  schema row with its `.expand` premise closure in
  `schemas.yaml`.  `emit_prims` syncs the ast/transforms/language
  projections **once** for the whole pool.
- `emit_atom(directory, atom)` — an anti-unification
  `MintedAtom` lands as a `fold:<atom>` schema recognizer (its
  `intro` claim) with the `expand` premise in the closure, the
  `intro` law in `laws.yaml`, and the op row in `ops.yaml`.
- `emit_family_rows(directory, families=…, refusals=…)` — minted
  family descriptors (serialized provenance + coverage) and
  refused-spec ledger rows into `families.yaml`.
- `vocab_extra_families(frag)` — decodes `families.minted_families`
  rows back into live handler-grammar descriptors (`body_from_data`)
  for `extra_families` injection.
- `upsert_model`, `upsert_schema_premise`, `upsert_family`,
  `upsert_refusal`, `upsert_op`, `upsert_law`,
  `upsert_model_verdicts` — the keyed upsert primitives; machine
  verdicts remain provenance, never owner rows, so `emit_all`
  preserves them.
- `collect_ops` drift fix: runtime-registered generators whose
  names match the minted shapes (`^.+_(?:[0-9a-f]{10}|\d+)$`,
  outside the shipped prim set) no longer count as owner rows —
  `_OP_REGISTRY` is process-global, so a live mint in the same
  process used to make every directory drift.  `antiunify.mint_atom`
  now records its caller-chosen names in `antiunify._MINTED_NAMES`
  (`minted_names()`), which the census also excludes — carrier/hand
  ops (`attnbias`, `trace`, …) keep their owner rows.

### `prim_mint` — consume and emit

`prim_pool(…, vocab_dir=…, emit_vocabulary=…)`:

- **Consume** (`_vocab_fragments`): `load_vocab(dir, fresh=True)` →
  `shipped_from_vocab` builds a `ShippedPrims` table from
  `prims.yaml`; `merge_shipped` unions it with the domain's core
  table; `seat_shipped` binds fragment-only prims the domain seater
  cannot know; the records' `alpha` fields and `prim:*` fold LHSes
  join `denoted` so already-vocabularized shapes are not re-minted;
  `vocab_extra_families` seeds the handler grammar.
- **Emit** (`_emit_pool_fragments`): every pooled candidate's
  `minted_prim_record` (fold law + optional depth-N `expand`
  premise) goes through `vocabdir.emit_prims`; family descriptors
  and refusal rows through `emit_family_rows`.  Stats record
  `vocab_emitted` / `vocab_families` / `vocab_refusals`.
- Refused handler specs are vocabulary too: `_refusal_ledger_row`
  writes the frozen spec + measured bound-site domain + family-loop
  accounting into `families.yaml` — replayable offline.
- The handler stage factored into `_pool_handler_row` /
  `_family_retry` / `_refused_handler_row` / `_refusal_ledger_row`
  (radon ratchet: `prim_pool` back under its complexity baseline).

### `overnight.py` — the loop's driver surface

- `--prim-vocab-dir DIR` consumes the directory; emission defaults
  on when a directory is supplied.
- `--prim-vocab-emit` / `--no-prim-vocab-emit`
  (`BooleanOptionalAction`) — a read-only probe consumes without
  writing.
- Both flags are `zoo-prim`-only (loud `ap.error` elsewhere) and
  land in `protocol["prim"]` (`vocab_dir`, `emit_vocabulary`) — the
  pool cache key, so a resume under different mint settings cannot
  reuse a stale pool.

### Anti-unification arm

`bench/drivers/probes/anti_unify_probe.py --vocab-dir DIR` emits each
`mint_atoms` product through `vocabdir.emit_atom` — mined candidates
become schema rows end-to-end, not just a sqlite evidence record.

### `tensor.unsqueeze` — the host-surface gap (item 6)

`host-tensor` grew the op the five phase-2 `python-shipped`
denotations were waiting on:

- `src/ops/shape_ops.rs`: `Unsqueeze` — inserts a size-1 axis,
  preserves payload order, torch-compatible negative-dim wrapping
  (`rem_euclid(rank + 1)`), `require_dim` validation; focused test
  `unsqueeze_inserts_axis_and_wraps_negative_dim` covers leading /
  trailing / negative / scalar / wrapped-out-of-range / invalid dim.
- `src/ops/mod.rs`: exported + registered in `TENSOR_OPS` and
  `tensor_op_registry`.
- `src/index.hnl`: `cap`/`op`/`handle` declarations and the exported
  tensor record entry.
- `modelproj._ATTR_OPS["unsqueeze"] = ("dim",)`.
- No backward pair — `tensor.derive("unsqueeze")` stays `null`
  (`squeeze` is not on the surface).

`hnl-run` was rebuilt (`cargo build -p mlopt-kit --bin hnl-run`;
see "Environment note" below) and `scripts/verify_vocab_models.py`
re-ran: **109 verdicts rewritten** — 104 `handl-shipped`, 5
`python-shipped`.  The five survivors are no longer blocked on
`unsqueeze`:

- 4 rows (`synop_4c5296b3ba`, `synop_976f9836d7`,
  `synop_98a002d64b`, `synop_f08a53f9a0`) now **project `.hnl`**
  (`unsupported: null`) but stay `python-shipped` with
  `no recorded bound-site domain` — all five prims' handlers were
  *refused* at mint time (pre-phase-3 records carry no
  `provenance.domain`), and `verify_model` honestly refuses to guess
  shapes.
- `synop_4c657cb2c0` now reports the *next* gap:
  `op 'select' is not on the tensor.* model surface`.

`vocab/models.yaml` diff: the four denotation rows gained `hnl`
projections; `unsupported` cleared; verdicts refreshed via
`upsert_model_verdicts`.

## Verified

- `tests/test_vocabdir.py` — fragment emission round-trips:
  `emit_prim` full fragment set, machine-added preservation,
  `emit_atom` schema fragments, `emit_family_rows` round-trip,
  upserts, projection drift, handl-peer vocab load.
- `tests/test_prim_mint.py::TestVocabDirLoop::
  test_emit_then_remint_mints_depth2` — **the depth-2 remint
  proof**: round 1 mints `ksynop_*` into a scratch `vocab/` copy;
  round 2 loads it (`fresh=True`), seats the emitted prim, and
  mints `ksynop2_*`; fragments, fold laws, schemas, refusal rows
  and `check_vocab` freshness all asserted.
- `tests/test_overnight.py` — new `test_prim_vocab_dir_flags_
  require_zoo_prim` and `test_prim_vocab_dir_is_protocol_pinned`
  (a real `--dry` zoo-prim run emits new prim fragments into a
  vocab copy; protocol pins `vocab_dir`/`emit_vocabulary`).
- Scoped suite: **155 passed** (`test_vocabdir`, `test_modelproj`,
  `test_vocab_model_parity`, `test_antiunify`, `test_prim_mint`,
  `test_shipped_primitives`, `test_prim_delivery_probe`) +
  **69 passed** `test_overnight.py` (incl. warnings-only torch
  export notices).
- `scripts/emit_vocab_fragments.py --check` → `vocabulary fresh`.
- `scripts/verify_vocab_models.py` → 109 verdicts, 104
  `handl-shipped` / 5 `python-shipped` (residuals above).
- Gates: `ruff check packages scripts examples` clean;
  `ruff format --check` clean; `ty check` clean; `vulture` clean;
  `lint-imports` contracts kept; `radon_ratchet` ok.
- `host-tensor`: `cargo test` — 77 unit + 6 + 6 integration
  passed, 1 doctest ignored, 0 failures (incl. the focused
  unsqueeze test).

## Environment note — Cargo patch stubs (not a repo change)

`/home/luis/GitHub/handl-lang/.cargo/config.toml`'s workspace-root
`[patch]` table references ~25 sibling checkouts that do not exist
locally (`lib-lang-parse`, `kit-lang`, `kit-ffi`, `lib-kv`, …).
Every one is `patch.unused` in graphopt's resolve graph, but cargo
validates patch paths eagerly and refused to build.  Minimal stub
crates (name + version only, never compiled) were created at the
missing paths to satisfy validation; `cargo build -p mlopt-kit
--bin hnl-run` then completed and the binary was refreshed.
**`Cargo.lock` was restored** — the unlocked build's patch-entry
rewrites are not part of this change.  The stubs are local
environment scaffolding; a real checkout of those repos (or a
config cleanup) should replace them.

## Honest residuals / phase-4 needs

- **5 `python-shipped` verdicts remain** — none blocked on
  `unsqueeze` anymore.  Four need a bound-site domain that was never
  recorded (pre-phase-3 refused handlers); one additionally needs
  `select` on the `tensor.*` surface.  Phase 4 options: re-mint
  those shapes under the phase-3 plumbing so refusals record
  domains, extend `verify_vocab_models` with an honest domain
  source, or grow `tensor.select`.
- **`emit_atom` coverage** is the probe path; the overnight
  anti-unification arm does not yet take a vocab dir (the prim arm
  does).  Phase 4/5: thread `--vocab-dir` through whichever driver
  owns the corpus-scale atom mint.
- **`models.yaml` rows for newly emitted prims** land via
  `emit_prim`'s batched model upserts; their `verdicts` are written
  by `verify_vocab_models.py` — a vocab-dir mint run should be
  followed by a verify pass before the directory is treated as
  shipped vocabulary.
- Play/saturation arms (phase 4) and handl-native saturation
  (phase 5) remain untouched.

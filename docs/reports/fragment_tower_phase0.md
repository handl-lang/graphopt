# Fragment tower — phase 0 progress note

Plan: `project/plans/pl-0029-plan-0033-the-fragment-tower-mlopt-vocabularies-as-language-data.md`
Host ADR: `host/project/adrs/0157-adr-0157-mlopt-on-the-fragment-tower-domain-vocabularies-as-language-fragments.md`
Ledger: `T-017` (open — phases 1–5 remain)

## What phase 0 delivered

`vocab/` at the repo root is now the canonical fragment vocabulary —
the single source of truth the plan's exit gate asks for.  Six
canonical JSON files, each pinned `"schema": "mlopt-vocab/1"`:

| file | rows | owner (today's Python registry) |
|---|---|---|
| `ops.yaml` | 288 | `opmeta.REGISTRY` ∪ `mlopt_core.ir._OP_REGISTRY` GenDefs |
| `laws.yaml` | 143 | `law_to_data` over `mlopt_core.laws.ALL_RULES` |
| `schemas.yaml` | 5 schemas + premise closures | `pipeline._FUSION_SCHEMAS` / `_FUSION_PREMISES` |
| `handlers.yaml` | 3 shipped + 57 prim + 52 minted | `lawdata.HANDLERS`, `shipped_primitives.handler_entries()`, minted `handler_record`s |
| `prims.yaml` | 57 | `mlopt_core.laws.minted.prim_records()` verbatim |
| `families.yaml` | family tables | `handler_synth` / `family_synth` / `genkernel.ELEMENTWISE` constants |

plus the handl-facing projections `language_from_dir` consumes:
`ast.yaml` (op shapes), `transforms.yaml` (143 law rules), and the
seven empty shell files (`tokens`/`grammar`/`eval`/`types`/`caps`/
`options`/`lexer_rules`).

### Machinery (`mlopt_discovery.vocabdir`)

- **One emitter per registry** — `collect_ops`/`collect_laws`/
  `collect_schemas`/`collect_handlers`/`collect_prims`/`collect_families`,
  driven by `emit_all`; `scripts/emit_vocab_fragments.py` is the CLI
  (`--check` = drift gate).
- **Emit merges, never wipes** — owner rows upsert by key; hand- or
  machine-added rows survive re-emission.  The directory is a source,
  not a build artifact.
- **Load = projection** — `load_vocab(dir)` rebuilds `Rewrite`s via
  `law_from_data`; records with `missing_hooks` (today only
  `conv_bn_fold`) re-bind procedural hooks *by name* from `ALL_RULES`
  and are reported in `Vocab.hooked`.
- **Mint seam (workstream 0b)** — `upsert_op`/`upsert_law`/
  `upsert_schema`/`upsert_handler`/`upsert_prim`/`emit_prim`; one
  `emit_prim` call lands the op row, the minted handler record and
  the verbatim prim record, then resyncs projections.
- **Drift gate** — `check_vocab` reports a bad `schema:` pin, an
  owner row missing/edited, or a stale projection.

### Consumers

- `pipeline.propose(..., vocab_dir=...)` — `None` auto-resolves the
  repo `vocab/` (an `MLOPT_VOCAB_DIR` override is honoured first);
  `False` forces the historical in-code tables; a path loads that
  directory.  The fusion-schema arm reads `schemas.yaml` + the
  fold/fuse subset of `laws.yaml` through the same
  `_instance_from_match` path — proposal pools are identical either
  way (pinned by `test_propose_vocab_dir_parity`).
- `mlopt-handl-peer` — `transform_ids(vocab/)` /
  `rewrite(vocab/, …)` read the projections unchanged; the
  `transforms.yaml` rows are the same `law_to_handl` shapes the
  `fragments-all` spike fixture proved loadable.
- `scripts/dump_fragments.py` now shares `vocabdir.emit_yaml_rows` —
  one tag-aware YAML emitter for both the spike fixtures and the
  canonical projections (verified byte-identical output).

### Exit gate status

A hand-added law row in `laws.yaml` + `sync_projections` is visible to
both engines: `load_vocab` rebuilds it into the Python law bank and
`transforms.yaml` carries `transform.mlopt.<name>` for the peer —
pinned by `test_hand_added_law_visible_to_both_engines`.  The
handl-side half is additionally pinned by the e2e
`test_handl_peer_loads_the_vocab_dir`, which runs wherever the
maturin extension is built (it skips here — `engine-transform-rewrite`
is not checked out locally, so the peer cannot build on this box).

## Verified

- `pytest tests/test_vocabdir.py` — 15 passed, 1 e2e skip
- `pytest tests/test_discovery_pipeline.py tests/test_handl_peer.py
  tests/test_family_synth.py -m "not slow"` — 134 passed, 5 skipped
- `scripts/emit_vocab_fragments.py --check` → `vocabulary fresh`
- `ruff check` / `ruff format --check` (packages, scripts, examples) — clean
- `ty check` — clean; `vulture` — clean; `lint-imports` — 1 kept
- `radon_ratchet` — ok (refactored `_emit`/`check_vocab`/`collect_ops`
  and the pipeline fusion helpers back under threshold)
- `bandit` — only the pre-existing `genkernel.py` B110 finding;
  `semgrep` cannot run in this env (mcp 2.x breakage, unrelated)

## What remains (phases 1–5)

- **Phase 1** — first-class `engine="handl"` differential backend.
- **Phase 2** — denotations and minted handlers as handl Models over
  `host-tensor`, fp64 parity against `TorchSink`.
- **Phase 3** — mint emits op + denotation + handler-family + schema
  fragments; loop-closure depth-2 remint test.
- **Phase 4** — play and saturation over fragment-application moves,
  typed guards, handl-egraph performance parity.
- (Phase 5 per the plan text — handl-native saturation — folds into
  the same arc; re-read the plan sections before starting.)

T-017 stays open.

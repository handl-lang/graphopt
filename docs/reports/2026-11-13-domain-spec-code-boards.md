# Domain-spec third cut — the code-shaped boards

2026-11-13.  Stage 2's last gap: boards whose cases need *code*
(module builders, custom games, nested objectives) could not be
named in JSON.  This cut seats **all seven remaining `play`
domains** — `gen`, `joint`, `search`, `fragment`, `open`,
`player`, `bilinear` — behind `domain_spec`, each wired to its
shipped generator/board/player-spec/playbook as data.

## What a spec can now name, per board

| board | `corpus.*` | `vocab.*` | player / incumbent (defaults) | deliver |
|---|---|---|---|---|
| `gen` | `builders` (fusion-builder names) | `ops`,`attrs`,`rules`,`handler_tags` (ambient subset) | `meta` / `scripted` | `auto`/`torch` → `_torch_deliver` |
| `joint` | `kinds` (`"0".."3"` archetype digits) | `ops`,`attrs`,`rules` (DEFAULT-name filter),`handler_tags` | `meta` / `scripted` | none |
| `search` | `kinds` (pool names), `horizon`, `patience` | `ops`,`attrs`,`rules` (SearchEnv ruleset filter) | `search` / `search` | none |
| `fragment` | `kinds` (pool names), `max_moves` | `ops`,`attrs`,`rules` (rewrite-name filter),`handler_tags` (vocab table) | `fragment` / `fragment` | none |
| `open` | `ops` (candidate op set), `page`, `start`, `step` | — (cases aren't terms) | `open` / `none` | none |
| `player` | `objective` (`meta`\|`open`), `max_evals` | — | `player` / `player` | none |
| `bilinear` | `scramble` | — | `bilinear` / `bilinear` | none |

Mechanics:

* **Per-board field schemas.**  `_CORPUS_KEYS` / `_VOCAB_KEYS` /
  `_CORPUS_PARAMS` whitelist what each board consumes; anything
  else is a *load error*, not a silently ignored key.  `open`,
  `player` and `bilinear` refuse `vocab.*` outright — their
  cases are dicts/term-lists the op-whitelist can't bind.
* **Board-bound player + incumbent.**  `player` and `incumbent`
  default per board (`_BOARD_DEFAULTS`) and are validated against
  it — a spec cannot seat `META_SPEC` on the search board or
  `ScriptedPlayer` on the bilinear board (`_BOARD_INCUMBENTS`).
  The five non-meta player specs (`SEARCH_SPEC`,
  `FRAGMENT_SPEC`, `OPEN_SPEC`, `PLAYER_SPEC`, `BILINEAR_SPEC`)
  and four playbooks (`_scripted_search`, `_fragment_scripted`,
  `GreedyEditor`, `strassen_line`) join `PLAYERS`/`INCUMBENTS`
  lazily via `_ensure_registries()`.
* **`vocab.rules` is now board-relative.**  On `search`/`joint`
  it filters the DEFAULT ruleset by name (an allowlist surviving
  zero rules refuses at build); on `fragment` it filters the
  `vocab/` directory's rewrite names; on `gen`/`torch` it must
  *admit* every DEFAULT name (those boards always seat DEFAULT).
* **`vocab.handler_tags` on `gen`/`fragment`** subsets real
  tables: the ambient `lawdata.HANDLERS` on gen (rides the case
  `ex` record into `_gen_board`'s `ambient_store`), the fragment
  vocab's `handler_table()` on fragment.  Tag validation is
  per-board (`_handler_tag_names`).
* **`gen` deliver.**  `deliver: "auto"` and `"torch"` both seat
  `_torch_deliver` on `gen` — its cases carry the same `ex`
  record (IR, leaves, example input, gen sink) the torch cases
  do; `_gen_board` reads `ex["ambient_handlers"]` for the store
  merge.
* **`bake_prior`** refuses boards whose states aren't
  `MetaArena`-shaped (`_ORACLE_BOARDS` = meta-arena, torch, gen,
  joint) rather than failing inside the oracle planner.

## play.py seams added (all backward-compatible kwargs)

* `_gen_cases(builders=, ambient_handlers=)` — builder-name
  selection + ambient handler-table substitution; unknown names
  raise before the draw.
* `_search_cases`/`_fragment_cases(kinds=)` — the case pools are
  now *named* (`_search_pool`, `_fragment_pool`, lazily built
  shared-leaf terms); case names carry the pool name
  (`search3:silu`, `frag1:mulsig`).
* `_search_board(rules=, horizon=, patience=)` and
  `_fragment_board(rules=, handlers=, max_moves=)` — the board
  factories take the spec's seated subsets/bounds.
* `_joint_cases` needed no change — `corpus.kinds` filters the
  `joint<i>:<archetype>` name suffix post-draw (validated
  `"0".."3"` at build).

## Demo — `play --domain-spec` end to end

Packaged specs: `domains/gen-fusion.json` (gen, 4 builders,
deliver auto) and `domains/search-folds.json` (search, 4 pool
kinds, 4-rule allowlist, horizon 10).

```
.venv/bin/python -m mlopt_discovery.play \
    --domain-spec gen-fusion --domain gen-fusion \
    --train-cases 4 --eval-cases 2 --budget 8
# scripted 2 cases reward 0.10 cert 2/2; random/learned -0.40

.venv/bin/python -m mlopt_discovery.play \
    --domain-spec gen-fusion --domain gen-fusion \
    --train-cases 2 --eval-cases 1 --budget 8 --deliver
# deliver[spec0:chain_mlp]: delivered=True verified=True
#   max_abs 5.2e-06 cost 3.0 cert True
```

The gen spec's delivered module is a real lowered artifact —
game score → verified runnable module, on a board that was
"code-shaped" yesterday.

## Still code-bound — the plugin seam, precisely

Everything *shipped* is now spec-able.  What still can't be
named in JSON is a **new** board: a new case generator, a new
player vocabulary, a new incumbent playbook, or a new nested
objective.  The extension seam is four registries — seat data,
not code paths:

* `domain_spec.BOARDS[name] = spec -> case -> Board` and
  `domain_spec._EXTRA_CASES[name] = spec -> (seed,n) -> cases`
  (the cases/board pair — mirror `_make_*_cases`/`_make_*_board`),
* `domain_spec.PLAYERS[name] = DomainSpec` (the learned arm's
  feature schema + legal-move enumerator),
* `domain_spec.INCUMBENTS[name] = () -> player`,
* plus rows in `_CORPUS_KEYS`/`_VOCAB_KEYS`/`_CORPUS_PARAMS`/
  `_BOARD_DEFAULTS`/`_BOARD_INCUMBENTS` saying what the board
  consumes and which player/incumbent it defaults to.

That is exactly the shape kernelopt already uses for
`play.DOMAINS` (`mlopt_kernel.play.register()`); a downstream
board repo would call the same four updates before
`register_file`.  What JSON genuinely cannot express — and
shouldn't pretend to — is *novel case semantics*: the `player`
board's `OBJECTIVES` scorers, the gen board's per-case kernel
minting, the open board's oracle evaluator are code by
construction; the spec names which shipped ones to seat.

## Honest edges

* `corpus.kinds` on `search`/`fragment` validates at first draw
  (inside `play._*_cases`), on `joint` at build; on `gen`,
  `corpus.builders` validates eagerly at build via an `n=0`
  probe.
* `vocab.ops`/`attrs` post-filter applies to every term board —
  on the tiny search/fragment pools it can honestly return
  short streams.
* `shapes`/`hardware` remain recorded-only outside `torch` —
  gen builders spell fixed input shapes; nothing materializes a
  tensor the fields could bind.
* `corpus.coverage` is accepted only where a `gen_cases`
  coverage seam exists (meta-arena, torch) — elsewhere it's a
  load error, not a recorded no-op.

## Files

* `packages/mlopt-discovery/src/mlopt_discovery/domain_spec.py`
  — per-board schemas, lazy `_ensure_registries`, seven
  cases/board factory pairs, board-bound player/incumbent
  validation, gen deliver, `bake_prior` guard, census/CLI.
* `packages/mlopt-discovery/src/mlopt_discovery/play.py` —
  `_gen_cases` builder/ambient seams, named search/fragment
  pools + `kinds`, parameterized `_search_board`/`_fragment_board`,
  `_gen_board` ambient-handler read.
* `packages/mlopt-discovery/src/mlopt_discovery/domains/
  gen-fusion.json`, `domains/search-folds.json` — packaged demos.
* `tests/test_domain_spec.py` — +16 tests (74 total).

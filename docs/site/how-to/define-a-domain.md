# Define a domain from a spec

<!-- provenance: graphopt:mlopt_discovery/domain_spec.py + docs/reports/2026-11-12-domain-spec-*.md,2026-11-13-domain-spec-code-boards.md -->

[Add a board](./add-a-board) seats a domain in Python. When the domain
is a *variation* of a shipped board — a vocabulary subset, a corpus
selection, a ruleset profile, an incumbent recipe — a JSON
**domain-spec** seats the same `Domain` with no code at all.
`mlopt_discovery.domain_spec` validates the spec, builds the cases
generator, board factory, player spec and scripted arm, and registers
the result in `play.DOMAINS`.

## 1. Pick a board

Every shipped `play` domain is spec-able — `board` names which one:

| `board` | `corpus.*` selects | `vocab.*` binds | delivers? |
|---|---|---|---|
| `meta-arena` | `kinds`, `coverage` | `ops`, `attrs`, `handler_tags`, `rules` | no (symbolic) |
| `torch` | `modules`, `coverage` (+ top-level `shapes`/`hardware`) | `ops`, `attrs`, `rules`, `handler_tags` | `auto`/`torch` |
| `gen` | `builders` | `ops`, `attrs`, `rules`, `handler_tags` | `auto`/`torch` |
| `joint` | `kinds` | `ops`, `attrs`, `rules`, `handler_tags` | no |
| `search` | `kinds`, `horizon`, `patience` | `ops`, `attrs`, `rules` | no |
| `fragment` | `kinds`, `max_moves` | `ops`, `attrs`, `rules`, `handler_tags` | no |
| `open` | `ops`, `page`, `start`, `step` | — (refused) | no |
| `player` | `objective`, `max_evals` | — (refused) | no |
| `bilinear` | `scramble` | — (refused) | no |

Validation is strict: a `corpus.*`/`vocab.*` key the board does not
consume is a **load error**, not a silently ignored field. Boards
whose cases are not op-terms (`open`/`player`/`bilinear`) refuse
`vocab.*` outright. `player`/`incumbent` default per board and are
checked against it — a spec cannot seat the meta-arena player on the
search board.

## 2. Write the spec

```json
{
  "name": "search-folds",
  "board": "search",
  "description": "the paying-fold half of the search case pool",
  "corpus": {"kinds": ["silu", "addid", "mulid", "negneg"], "horizon": 10},
  "vocab": {"rules": ["silu_fold", "id_add", "id_mul", "double_neg"]},
  "player": "search",
  "incumbent": {"kind": "search"},
  "deliver": "none"
}
```

Top-level fields: `name` (required, `^[a-z][a-z0-9_-]*$` — it becomes
the `--domain` name), `board`, `corpus`, `vocab`, `player`,
`incumbent`, `deliver`, `shapes`,
`hardware`, `description`. Notable seams:

- `vocab.rules` — `"all"` (default), `"default"`, `"scan"`, an
  explicit name list, or `{"base": ..., "minus": [names]}` (the
  subtraction profile). On `search`/`joint`/`fragment` it filters
  the board's ruleset by name.
- `player` — the board's player-spec name (validated: each board
  plays its own spec), optionally as `{"kind": ..., "prior": <ref>}`
  where the ref is `"none"` (cold start), `"default"`, a packaged
  `priors/<name>.json`, a path, or an inline weight object.
  `domain_spec.bake_prior(spec)` mints a spec-matched table by
  oracle-imitation on the spec's own boards.
- `deliver` — `"auto"` (the board's own hook: torch and gen lower +
  verify, symbolic boards do not), `"none"`, or `"torch"` (requires
  a `torch`/`gen` board — the hook verifies against the case's
  example input).
- `shapes`/`hardware` — consumed on `torch` (module width/batch,
  device/dtype); recorded-only on the symbolic boards.

Four specs ship packaged under
`mlopt_discovery/domains/` — `pointwise` (meta-arena vocabulary
demo), `torch-silu`, `gen-fusion`, `search-folds` — and a bare name
resolves to them.

## 3. Preview, register, play

```sh
# resolve the spec to its census (kinds, vocab sizes, bounds) — no episodes
.venv/bin/python -m mlopt_discovery.domain_spec --spec search-folds

# register + play in one command — repeatable flag
.venv/bin/python -m mlopt_discovery.play \
    --domain-spec search-folds --domain search-folds \
    --train-cases 4 --eval-cases 2 --budget 8

# the deliver hook rides the same flag
.venv/bin/python -m mlopt_discovery.play \
    --domain-spec gen-fusion --domain gen-fusion \
    --train-cases 2 --eval-cases 1 --budget 8 --deliver
```

Programmatically, `register_file(source, domains=DOMAINS)` takes a
JSON path, a packaged name, or a plain dict — the same loader the CLI
uses. When running `play` as `__main__` pass your own module's
`DOMAINS` dict (the `__main__`/`mlopt_discovery.play` split makes the
registries distinct objects).

## 4. When a spec is not enough

A spec names *variations of shipped boards*. A genuinely new board —
a new case generator, player vocabulary, or playbook — is the plugin
seam: seat factories into `domain_spec.BOARDS` / `_EXTRA_CASES` /
`PLAYERS` / `INCUMBENTS` (or `play.DOMAINS` directly) before calling
`register_file` — the same shape `kernelopt`'s
`mlopt_kernel.play.register()` uses. See
[add a board](./add-a-board) for the Python path.

## See also

- [Add a board](./add-a-board) — the code path: `Board` protocol, `spec_domain`, `DOMAINS`.
- [Register a player](./register-a-player) — the arm side of the same table.
- [The game](../explanation/the-game) — boards, players, and the referee seam.

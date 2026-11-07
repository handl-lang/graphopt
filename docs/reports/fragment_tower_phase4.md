# Fragment tower — phase 4: fragment-application play

Plan 0033 / host ADR-0157.  Phase 4 seats the game/saturation layer on
*fragment-application moves* instead of only the Python rule bank.

## Delivered

- `mlopt_discovery.fragment_game.FragmentArena` — a `gamekit.engine.Board`
  over the canonical `vocab/` directory (`vocabdir.load_vocab`).
- `FragmentMove(kind, ref, site)` — data-only move: which fragment row,
  which e-class/site.  `fragment_legal(state)` exposes the state's moves.
- Executable move kinds:
  - `law` / `prim` — `Rewrite` rows from `laws.yaml` (incl. `prim:*` folds),
    applied at one matching e-class via `EGraph.apply_rule(..., search={site})`.
  - `schema` — `fold:*` schema rows from `schemas.yaml`, same site-targeted
    rewrite application.
  - `handler` — `handlers.yaml` rows applied through the meta-arena's
    referee `claim` move (declare → fire → bind kernel).
  - `saturate` — bounded closure over all executable rewrite fragments.
  - `extract` — terminal; delegates to `MetaArena.step(Action.extract())`,
    so `verify_certificate` remains the referee.
- `FragmentState.catalog` counts every vocab row by kind (`law`, `prim`,
  `schema`, `handler`, `op`, `model`, `family`) — `op`/`model`/`family`
  rows are data/provenance, not term moves, and are honestly not offered.
- `engine="handl"` swaps the board's `EGraph` for `HandlEngine`
  (phase-1 differential parity path); native handl-egraph saturation is
  **not** claimed — that remains phase 5 behind the perf-parity gate.
- `play` domain `"fragment"` registered (`_fragment_cases`,
  `_fragment_board`, `FRAGMENT_SPEC`, `_fragment_scripted`) so the generic
  train/eval drivers can play the board.

## Verified

- `tests/test_fragment_game.py` — move-space shape, site-targeted law
  application, saturate→extract with `certificate_ok`, handler claim,
  dict-move spelling, terminal refusal, `engine="handl"` parity path, and
  a `gamekit.engine.run_episode` smoke.
- `tests/test_discovery_play.py` — registry updated for `"fragment"`.
- Focused run: `pytest tests/test_fragment_game.py tests/test_discovery_play.py -q`
  → 61 passed.
- `ruff check` / `ruff format --check` clean on the touched shipped files.

## Skipped / deferred

- Native `handl-egraph` saturation and the 2× perf-parity measurement —
  phase 5.  `engine="handl"` is the differential oracle path only.
- `op`, `model`, and `family` rows are catalogued but have no term-
  application semantics yet (no rewrite/interpretation body to apply).
- `check`/`derive`/`cond` hooks still evaluate in Python; migration toward
  typecheck-nbe predicates is the dependent-surface work named in the plan.

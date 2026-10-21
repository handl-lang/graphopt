# Register a player

<!-- provenance: graphopt:gamekit/players/registry.py@bac1a61 -->

A player is a folder in `packages/gamekit/src/gamekit/players/` — a
docstring carrying the measured record, a `SPECS` tuple of `PlayerSpec`
rows, and a `make` arm factory. The registry maps name → spec so
`--player <name>` on `mlopt_discovery.play` and the comparison runner
(`bench/players`) enumerate arms from one table.

## 1. Create the player folder

`gamekit/players/<name>/__init__.py`:

```python
"""``<name>`` — what it plays and the measured record that earned it
a seat."""

from collections.abc import Callable
from typing import Any

from gamekit.players.registry import EPISODE_BOARDS, PlayerSpec

SPECS: tuple[PlayerSpec, ...] = (
    PlayerSpec(
        name="<name>",
        kind="linear",           # chance | scripted | linear | neural | distilled
        boards=EPISODE_BOARDS,   # the honest capability statement —
                                 # list only the boards make() accepts
        schema="the domain's DomainSpec row",
        train="scripts/train_<name>.py",
        artifact="lawdata.<NAME>_WEIGHTS",
        record=(
            "meta +0.0 reward/ep — beat random by X "
            "(docs/results_<name>.md)"
        ),
    ),
)

def make(
    domain: str,
    *,
    spec: PlayerSpec,
    seed: int = 0,
    ctx: dict | None = None,
) -> Callable[[int], Any]:
    """Return the arm factory for *domain* — ``i -> player``."""
    from mlopt_discovery.play import DOMAINS

    spec_for_domain = DOMAINS[domain].spec
    return lambda i: MyPlayer(spec_for_domain, seed=seed + i)
```

The `PlayerSpec` fields are the registration contract: `boards` is the
capability statement, `schema`/`train`/`artifact` record where the
weights come from, `record` is the one-line measured claim the module
docstring argues at length, and `variant` carries decode/weights flags
the factory reads (`{"greedy": True}` ships a second spec row — see
`linear` / `linear-greedy` / `linear-cold`).

## 2. Honor the arm-factory contract

`registry.make(name, domain, seed=seed, ctx=ctx)` calls your `make` and
returns `i -> player`:

- on an **episode board** (`play.DOMAINS`), each `player` is a
  `state -> action | None` callable as `engine.run_episode` consumes;
- on the **`contraction` board**, each `player` is a
  `(tensors, sizes) -> cost` callable — the ordering game scores final
  contraction cost, not episode reward;
- a factory may also write back into `ctx` (`ctx["score_fn"]` — the
  page-ordering scorer the leaderboard prices with average precision).

`registry.make` raises `ValueError` when the player does not play the
board — the caller maps that to a skipped row, **never a silent run**.
Keep `spec.boards` honest.

## 3. Register the module

```python
# gamekit/players/registry.py — the _MODULES table
_MODULES["<name>"] = "gamekit.players.<name>"
```

Loading is lazy: `import gamekit.players` stays stdlib-only, and the
folders import on first `spec()`/`make()` — so import inside `make`
whatever the player needs.

## 4. Measure it and record the record

```sh
.venv/bin/python -m mlopt_discovery.play --domain meta --player <name>
```

`--player` runs eval-only, paired against `random`. Write the result as
a `docs/results_*.md` document upstream, then put the one-line claim in
`spec.record` and argue it in the module docstring — the record is what
earns the seat.

## See also

- [Add a board](./add-a-board) — the board side of the same contract.
- [The game](../explanation/the-game) — how arms, boards, and the referee fit.
- [The player leaderboard](../reference/leaderboard) — the published measured records.

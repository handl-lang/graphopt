# Add a board

<!-- provenance: graphopt:gamekit/engine.py + mlopt_discovery/play.py@bac1a61 -->

A **board** is one optimization game as an RL environment — observe a
state, play a move, get a report, repeat until done. A **domain** is the
registration that binds a family of boards to the drivers: the cases
generator, the `case -> Board` factory, and the learned player's
vocabulary. This recipe adds both.

## 1. Implement the `Board` protocol

```python
from dataclasses import dataclass
from gamekit.engine import Report

@dataclass(frozen=True)
class MyState:
    # ...everything the featurizer reads, plus the terminator:
    done: bool = False

class MyBoard:
    def observe(self) -> MyState:
        """Return the read-only state snapshot (state.done ends)."""
        return self._state

    def step(self, action) -> tuple[MyState, Report]:
        """Apply action; return (state', report)."""
        state = self._apply(action)
        return state, Report(
            action=action,
            reward=self._reward(action, state),   # the learning signal
            terminal=state.done,                   # the game over
        )
```

The contract is two methods: `observe() -> state` where the state
carries `done`, and `step(action) -> (state, report)` where the report
carries `reward` and `terminal`. `gamekit.engine.Report` is the floor —
boards may return richer reports (the shipped arenas do).

## 2. Smoke-test with `run_episode`

```python
from gamekit.engine import run_episode

traj = run_episode(MyBoard(), lambda state: pick_move(state), budget=24)
traj.total      # summed step rewards — the REINFORCE return
traj.terminal   # the terminal report, or None
```

A player is any callable `state -> action | None`; returning `None` ends
the episode early, as does `state.done`.

## 3. Register the domain

```python
# mlopt_discovery/play.py — one row in DOMAINS
DOMAINS["mine"] = _spec_domain(
    my_cases,          # (seed, n) -> [case, ...]
    my_board_of,       # case -> Board
    MY_SPEC,           # a DomainSpec — fills legal/featurizer/features
    scripted=MyScriptedPlayer,   # optional baseline arm
    deliver=my_deliver,          # optional: (board, case, traj) -> artifact
)
```

Or set the fields directly — `Domain` is data:

| field | role |
|---|---|
| `cases` | `(seed, n) -> [case, ...]` — the train/eval split generator |
| `board_of` | `case -> Board` — the factory the drivers call per episode |
| `legal` | `state -> tuple[action, ...]` — the move enumerator |
| `featurizer` / `features` | `(state, action, hist) -> dict` and the learned arm's schema — a `DomainSpec` bound via `_spec_domain` fills all three and records itself on `spec` |
| `scripted` | `() -> player` — the hand-written baseline (optional) |
| `deliver` | `(board, case, traj) -> artifact` — only if the board produces a runnable artifact (the `torch` domain ships one) |

## 4. Drive it

```sh
.venv/bin/python -m mlopt_discovery.play \
    --domain mine --train-cases 12 --eval-cases 4 --budget 24
.venv/bin/python -m mlopt_discovery.play --domain mine --player linear
.venv/bin/python -m mlopt_discovery.play --domain mine --player linear --deliver
```

`--player` evaluates a [registered player](./register-a-player)
eval-only, paired against `random`; `--deliver` lowers and verifies the
winning extraction on domains that ship a `deliver` hook.

## 5. Compare arms and train

```python
from gamekit.engine import evaluate, train_policy

table = evaluate(arms, cases, board_of, budget=24)
# arms: {name: i -> player} — every arm sees the same cases in the same
# order, so means compare like for like

totals = train_policy(player, cases, board_of, budget=24)
# one episode per case; the player's finish_episode gets each return
```

## See also

- [The game](../explanation/the-game) — boards, players, and the referee seam.
- [Register a player](./register-a-player) — the arm side of the same table.
- [The frozen API reference](../reference/frozen-api) — `Board`, `Report`, `run_episode`, `Domain`, `DOMAINS` are frozen names.

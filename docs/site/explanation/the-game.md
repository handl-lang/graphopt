# The game — boards, players, and the learned-policy loop

<!-- provenance: gamekit:packages/gamekit/src/gamekit/engine.py + graphopt:docs/results_player_on_board.md@eea02f6 -->

Because programs, rules, and derivations are all values in one
calculus, *discovery itself* is a game: observe a state, play a move,
get a report, repeat until done. `gamekit` names that shape once and
puts the drivers over it.

## The `Board` protocol

```python
class Board(Protocol):
    def observe(self) -> state: ...          # state.done ends the episode
    def step(self, action) -> (state, report): ...   # report: reward + terminal
```

One contract, every domain. Three drivers ride on it:

- `run_episode(board, player, budget)` — play one episode, get a `Trajectory`.
- `evaluate(arms, cases, board_of, *, budget)` — compare players under identical cases.
- `train_policy(player, cases, board_of, *, budget)` — REINFORCE over episode returns.

A new domain is data + a small adapter: a `case -> Board` factory and
a featurizer. The domain registry (`mlopt_discovery.play.DOMAINS`)
holds the boards — `meta`/`joint`/`search`/`torch`/`gen`/`open`/`player` —
played through one CLI (the registry is open: downstream domain repos
seat their own boards into `DOMAINS` through `play.spec_domain` at
import):

```sh
python -m mlopt_discovery.play --domain meta --player linear   # eval-only
python -m mlopt_discovery.play --domain torch --deliver        # lower + verify the win
```

## The players

The registry (`gamekit.players.registry`) names every playable arm —
`random` (chance), `playbook`/`greedy` (scripted), `linear` (the
trained linear prior, plus `-greedy`/`-cold` variants), `neural`,
`distilled` (the contraction player), `gated` (the isolated stop
channel) — each as a `PlayerSpec` (`name`/`kind`/`boards`/`schema`/
`train`/`artifact`/`record`/`variant`) with a `make(name, domain)`
factory that raises `ValueError` on an unplayed board rather than
silently running. The learned arm is a featurizer-injected
`LinearPolicy`: state features + action features → a score per legal
move.

The two-tier meta-game goes one level up: `PlayerArena` is a board
whose *state is a player spec* — moves toggle feature parts, edit
weight priors, set the child's learning rate, probe the trained child,
and `extract()` pays the delta. The leaderboard measures every
registered player on every board it can play under one paired protocol
(see [the leaderboard](../reference/leaderboard)).

## The learned-policy loop

Inside *search*, a `Policy` only reorders the legal moves — every rule
still fires, so the fixed point and the certificate are unchanged
(evaluation never decides equivalence — see
[evaluation](./evaluation)). A trained linear prior ranks
law candidates at ~4.7× the tuned heuristic's average precision
(held-out); a distilled contraction-ordering player beats
`opt_einsum`'s randomized greedy at equal wall-clock. Neural arms lose
to the linear/distilled players today — reported as measured, not
tuned.

## The referee seam

The split that keeps the game honest: **the value spec is gamekit's;
the measurement is the gauntlet's.** `ScoringSpec`/`ScoreGate` declare
the axes and thresholds (speed, memory, launch, params, win-rate) as
*data*; `score(spec, measured)` consumes a plain `Mapping[str, float]`
emitted by the measurement side. The game layer imports no measurement
machinery and the measurement side never imports the value spec — an
enforced import contract.

## Why a game at all

The claim under test is the *connection*: TASO/Isaria mint + verify
rules offline; CompilerGym-style systems learn over fixed banks.
mlopt's bank grows under play — a minted law that survives
[the gauntlet](./the-gauntlet) becomes a legal move in the next
episode. The open research question is exactly that: can a player
learn in a game that grows under it?

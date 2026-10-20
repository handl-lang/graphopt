"""The player registry — every player is a module under ``players/``.

Plan 0031 workstream 2 ("players as modules"): a player is a folder in
this package — one measured record in its docstring, its
:data:`PlayerSpec` rows, and a ``make`` that binds the domain's
enumerator/featurizer/weights into the arm factory the game drivers
consume.  The registry maps *name → spec* so ``--player <name>`` on
:mod:`mlopt_discovery.play` and the comparison runner
(``bench/players``) enumerate arms from one table.

A :class:`PlayerSpec` is *data*: which boards the player can play, the
feature schema it scores on, the entry point that trained its shipped
weights, the artifact that carries them, and the measured record that
earned it a seat.  Loading is lazy — :func:`spec` imports the folders
on first touch, so ``import mlopt_discovery.players`` stays
stdlib-only.

The arm-factory contract: ``registry.make(name, domain, seed, ctx)``
returns ``i -> player``.  On an *episode* board (the
``play.DOMAINS``) the player is a ``state -> action | None`` callable
as :func:`engine.run_episode` expects; on the ``contraction`` board it
is a ``(tensors, sizes) -> cost`` callable — the ordering game scores
final contraction cost, not episode reward.  A factory may also write
back into *ctx* (``ctx["score_fn"]`` — the page-ordering scorer the
leaderboard prices with average precision).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

__all__ = [
    "BOARDS",
    "EPISODE_BOARDS",
    "PlayerSpec",
    "for_board",
    "make",
    "module",
    "names",
    "spec",
    "specs",
]

#: The episode boards — ``play.DOMAINS`` keys (declared here so the
#: player modules stay importable without pulling the domain layer).
EPISODE_BOARDS: tuple[str, ...] = (
    "meta",
    "joint",
    "torch",
    "search",
    "gen",
    "open",
    "player",
)

#: Every board the registry knows — the episode domains plus the
#: ``order``-family contraction board the comparison runner drives.
BOARDS: tuple[str, ...] = (*EPISODE_BOARDS, "contraction")


@dataclass(frozen=True)
class PlayerSpec:
    """One player's registration — the module's seat on the roster.

    *boards* is the honest capability statement: the boards the
    ``make`` factory accepts.  *schema* names the feature vocabulary
    the player scores over (the ``DomainSpec`` it binds, or the net's
    own input vector); *train* is the entry point that produced the
    shipped weights; *artifact* is where those weights live;
    *record* is the one-line measured record the module docstring
    argues at length.  *variant* carries decode/weights flags the
    factory reads (``{"greedy": True}``, ``{"cold": True}``).
    """

    name: str
    kind: (
        str  # "chance" | "scripted" | "linear" | "neural" | "distilled"
    )
    boards: tuple[str, ...]
    schema: str = ""
    train: str = ""
    artifact: str = ""
    record: str = ""
    variant: Mapping[str, Any] = field(default_factory=dict)


#: Module key → the folder's module path.  The folder itself declares
#: its ``SPECS``; this table only names where to look, so
#: :func:`names` and error messages never pay an import beyond the
#: cheap folder bodies.
_MODULES: dict[str, str] = {
    "random": "mlopt_discovery.players.random",
    "playbook": "mlopt_discovery.players.playbook",
    "linear": "mlopt_discovery.players.linear",
    "neural": "mlopt_discovery.players.neural",
    "distilled": "mlopt_discovery.players.distilled",
}

#: spec name → (module path, PlayerSpec) — built by :func:`specs`.
_INDEX: dict[str, tuple[str, PlayerSpec]] = {}


def _load_specs() -> None:
    """Import every player folder once; index its ``SPECS`` by name."""
    for mod_name in _MODULES.values():
        mod = importlib.import_module(mod_name)
        for s in mod.SPECS:
            _INDEX[s.name] = (mod_name, s)


def specs() -> dict[str, PlayerSpec]:
    """Return ``{name: PlayerSpec}`` over the whole registered roster."""
    if not _INDEX:
        _load_specs()
    return {name: s for name, (_m, s) in _INDEX.items()}


def names() -> tuple[str, ...]:
    """Return the registered player names, sorted."""
    specs()
    return tuple(sorted(_INDEX))


def _known(name: str) -> ValueError:
    """Build the unknown-player error, with the roster in it."""
    return ValueError(
        f"unknown player {name!r} — registered: {', '.join(names())}"
    )


def module(name: str) -> ModuleType:
    """Import the player folder for *name* (a spec or module key)."""
    if name in _MODULES:
        return importlib.import_module(_MODULES[name])
    specs()
    if name not in _INDEX:
        raise _known(name)
    return importlib.import_module(_INDEX[name][0])


def spec(name: str) -> PlayerSpec:
    """Return the named player's spec; raise ``ValueError`` if absent."""
    specs()
    if name not in _INDEX:
        raise _known(name)
    return _INDEX[name][1]


def for_board(board: str) -> list[PlayerSpec]:
    """Return every registered player that can play *board*, sorted."""
    return [s for _n, s in sorted(specs().items()) if board in s.boards]


def make(
    name: str,
    domain: str,
    *,
    seed: int = 0,
    ctx: dict[str, Any] | None = None,
) -> Callable[[int], Any]:
    """Return the arm factory for *name* on *domain* — ``i -> player``.

    Raises ``ValueError`` when the player does not play the board —
    the caller maps that to a skipped row, never a silent run.
    """
    s = spec(name)
    if domain not in s.boards:
        raise ValueError(
            f"player {name!r} does not play board {domain!r} "
            f"(boards: {', '.join(s.boards)})"
        )
    return module(name).make(
        domain, spec=s, seed=seed, ctx={} if ctx is None else ctx
    )

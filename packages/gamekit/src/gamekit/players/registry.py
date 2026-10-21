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
on first touch, so ``import gamekit.players`` stays
stdlib-only.  The old ``mlopt_discovery.players`` path is a
re-exporting shim (plan 0032 lane A).

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
    "register_player",
    "spec",
    "specs",
]

#: The episode boards this repo ships — ``play.DOMAINS`` keys
#: (declared here so the player modules stay importable without
#: pulling the domain layer).  Downstream domain repos seat their
#: own boards through :func:`register_player` (plan 0032 phase 4 —
#: kernelopt's ``kernel`` registers from ``mlopt_kernel.play``); a
#: registered board is not an ``EPISODE_BOARDS`` member, it is an
#: extension seat on the named players.
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
    "random": "gamekit.players.random",
    "playbook": "gamekit.players.playbook",
    "linear": "gamekit.players.linear",
    "neural": "gamekit.players.neural",
    "distilled": "gamekit.players.distilled",
    "gated": "gamekit.players.gated",
}

#: spec name → (module path, PlayerSpec) — built by :func:`specs`.
_INDEX: dict[str, tuple[str, PlayerSpec]] = {}

#: External seats — downstream domain repos (kernelopt's ``kernel``
#: board) register players the folder roster cannot know about:
#: name → (PlayerSpec, make | None).  A *fresh* name must supply its
#: own ``make`` (same signature the folder makes use); a name that
#: matches the shipped roster unions its ``boards`` into the shipped
#: spec, and an explicit ``make`` claims exactly the boards the
#: shipped spec didn't — the shipped folder keeps its own domains.
_EXTERNAL: dict[str, tuple[PlayerSpec, Callable | None]] = {}


def _load_specs() -> None:
    """Import every player folder once; index its ``SPECS`` by name."""
    for mod_name in _MODULES.values():
        mod = importlib.import_module(mod_name)
        for s in mod.SPECS:
            _INDEX[s.name] = (mod_name, s)


def _merged(name: str) -> PlayerSpec | None:
    """Return the shipped spec with any external boards unioned in."""
    shipped = _INDEX.get(name)
    ext = _EXTERNAL.get(name)
    if shipped is None:
        return ext[0] if ext is not None else None
    s = shipped[1]
    if ext is None:
        return s
    boards = tuple(dict.fromkeys((*s.boards, *ext[0].boards)))
    return PlayerSpec(
        name=s.name,
        kind=s.kind,
        boards=boards,
        schema=s.schema,
        train=s.train,
        artifact=s.artifact,
        record=s.record,
        variant=s.variant,
    )


def register_player(
    spec: PlayerSpec, make: Callable | None = None
) -> None:
    """Seat a downstream player — the domain-repo plugin seam.

    Domain packages outside this repo (plan 0032 phase 4 — the
    kernelopt split seats ``kernel`` from ``mlopt_kernel.play``)
    ship players the folder roster cannot know about.  Two shapes:

    * *spec.name* matches a shipped player — the external spec's
      ``boards`` union into the shipped seat.  Spec-driven
      factories (``linear`` reads ``play.DOMAINS[domain].spec``,
      ``random`` is uniform) then bind the new board with no code
      here.  An explicit *make* claims exactly the boards the
      shipped spec did not already play — a scripted line lives
      where the domain lives (the kernel playbook).
    * *spec.name* is fresh — the spec registers outright and
      *make* is required; :func:`make` calls it with the
      folder-make signature
      ``(domain, *, spec=..., seed=..., ctx=...)``.

    Re-registering a name replaces its external seat — plugin
    imports stay idempotent.
    """
    specs()
    if make is None and spec.name not in _INDEX:
        raise ValueError(
            f"external player {spec.name!r} is not on the shipped "
            "roster — it must supply its own make"
        )
    _EXTERNAL[spec.name] = (spec, make)


def specs() -> dict[str, PlayerSpec]:
    """Return ``{name: PlayerSpec}`` over the whole registered roster."""
    if not _INDEX:
        _load_specs()
    out = {name: s for name, (_m, s) in _INDEX.items()}
    for name in _EXTERNAL:
        merged = _merged(name)
        if merged is not None:
            out[name] = merged
    return out


def names() -> tuple[str, ...]:
    """Return the registered player names, sorted."""
    specs()
    return tuple(sorted(set(_INDEX) | set(_EXTERNAL)))


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
    s = _merged(name)
    if s is None:
        raise _known(name)
    return s


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
    ext = _EXTERNAL.get(name)
    if ext is not None and ext[1] is not None:
        shipped = _INDEX.get(name)
        if shipped is None or domain not in shipped[1].boards:
            return ext[1](
                domain,
                spec=s,
                seed=seed,
                ctx={} if ctx is None else ctx,
            )
    return module(name).make(
        domain, spec=s, seed=seed, ctx={} if ctx is None else ctx
    )

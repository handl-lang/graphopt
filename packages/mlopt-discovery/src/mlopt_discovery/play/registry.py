"""The domain registry — the kernelopt-facing seam (frozen public API).

:class:`Domain` is data (cases generator, ``case -> Board`` factory,
player vocabulary); :func:`spec_domain` builds one from a
:class:`~gamekit.players.DomainSpec`; :func:`play` / :func:`deliver`
run episodes.  ``DOMAINS`` starts empty here and is populated by
:mod:`mlopt_discovery.play.domains` at package import — downstream
repos seat their own entries by mutating the dict (kernelopt's
``mlopt_kernel.play.register`` writes ``DOMAINS["kernel"]``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gamekit import engine
from gamekit.players import DomainSpec


@dataclass(frozen=True)
class Domain:
    """One board family's registration — data, not code.

    A domain binds a *cases* generator, a ``case -> Board`` factory
    and the learned player's vocabulary: its legal-move enumerator
    (``legal``), featurizer and feature schema (``features``).  A
    domain that ships a :class:`~mlopt_discovery.players.DomainSpec`
    builds its record with :func:`spec_domain`, which fills those
    three from the spec and records it on ``spec``; a domain may set
    them directly instead.  ``scripted`` and ``deliver`` are extras.
    """

    cases: Any  # (seed, n) -> [case, ...]
    board_of: Any  # case -> Board
    legal: Any = None  # state -> tuple[action, ...]
    featurizer: Any = None  # (state, action, hist) -> dict
    features: Any = ()  # the learned arm's schema
    scripted: Any = None  # () -> player, or None
    deliver: Any = None  # (board, case, traj) -> artifact | None
    spec: DomainSpec | None = None  # the player spec, when one is bound


def spec_domain(
    cases: Any,
    board_of: Any,
    spec: DomainSpec,
    *,
    scripted: Any = None,
    deliver: Any = None,
) -> Domain:
    """Build a :class:`Domain` whose player fields come from *spec*."""
    return Domain(
        cases=cases,
        board_of=board_of,
        legal=spec.legal,
        featurizer=spec.featurize,
        features=spec.features,
        scripted=scripted,
        deliver=deliver,
        spec=spec,
    )


def play(
    domain: str,
    case: Any,
    player: Any,
    *,
    budget: int = 24,
) -> engine.Trajectory:
    """Play one episode of *domain*/*case* under *player*."""
    d = DOMAINS[domain]
    return engine.run_episode(d.board_of(case), player, budget)


def deliver(
    domain: str,
    case: Any,
    player: Any,
    *,
    budget: int = 24,
) -> dict:
    """Play an episode on *case*; return the lowered artifact.

    The product loop: the board's winning extraction resolved by
    ``MetaArena.deliverable`` (handled ops instantiate their
    kernels, unhandled abbreviations spell back), lowered through
    the domain's sink, verified against the source model.  Domains
    without a ``deliver`` hook raise — the game score needs no
    artifact; delivery does.
    """
    d = DOMAINS[domain]
    if d.deliver is None:
        raise ValueError(f"domain {domain!r} has no deliver hook")
    board = d.board_of(case)
    traj = engine.run_episode(board, player, budget)
    return d.deliver(board, case, traj)


#: The live registry — populated by ``play.domains`` on package
#: import; downstream domains mutate it in place.
DOMAINS: dict[str, Domain] = {}

__all__ = [
    "DOMAINS",
    "Domain",
    "deliver",
    "play",
    "spec_domain",
]

"""Players for the generic optimization game — one package, two halves.

Plan 0032 lane A moved this package out of ``mlopt_discovery`` into
``gamekit``; the old ``mlopt_discovery.players`` path stays
importable as a re-exporting shim.

The *base machinery* lives in :mod:`._base` (re-exported here, so the
pre-package ``gamekit.players`` surface is unchanged):
:class:`LinearPolicy` is the reusable learner — a softmax linear
policy over whatever ``legal(state)`` enumerates, scored by an
injected ``featurizer`` — and :class:`LearnedPlayer` is the one policy
built from a domain's :class:`DomainSpec` (feature schema, the
featurizer's composable parts, the legal-move enumerator).

The *players-as-modules* half lives in :mod:`.registry`: every player
is a subpackage here — :mod:`.random`, :mod:`.playbook`,
:mod:`.linear`, :mod:`.neural`, :mod:`.distilled` — carrying its
:class:`~gamekit.players.registry.PlayerSpec` (board support,
feature schema, train entry point, shipped artifact) and a measured
record in its module docstring.  ``players.registry.make(name,
board)`` returns the arm factory the comparison runner and the
``--player`` flag on :mod:`mlopt_discovery.play` consume; adding a
player is adding a folder.
"""

from __future__ import annotations

from . import registry
from ._base import _BASELINE_EMA as _BASELINE_EMA
from ._base import (
    LEGALS,
    PARTS,
    RANKS,
    ActionFeats,
    Bias,
    DomainSpec,
    GatedPlayer,
    LearnedPlayer,
    LinearPolicy,
    OpOneHot,
    PlayHist,
    PlayRates,
    RelFeats,
    StateBroadcast,
    compose_featurizer,
    register_legal,
    register_part,
    register_rank,
    spec_from_record,
)
from ._base import _action_key as _action_key
from ._base import _dot as _dot
from ._base import _grad_row as _grad_row
from ._base import _Hist as _Hist
from .registry import PlayerSpec

__all__ = [
    "LEGALS",
    "PARTS",
    "RANKS",
    "ActionFeats",
    "Bias",
    "DomainSpec",
    "GatedPlayer",
    "LearnedPlayer",
    "LinearPolicy",
    "OpOneHot",
    "PlayHist",
    "PlayRates",
    "PlayerSpec",
    "RelFeats",
    "StateBroadcast",
    "compose_featurizer",
    "register_legal",
    "register_part",
    "register_rank",
    "registry",
    "spec_from_record",
]

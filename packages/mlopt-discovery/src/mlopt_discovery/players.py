"""Compatibility shim — the players package moved to :mod:`gamekit`.

Plan 0032 lane A extracted the player machinery into
``packages/gamekit``: the base machinery (``players/_base.py``) and
the registry/folders now live under :mod:`gamekit.players`.  This
module re-exports the whole pre-move ``mlopt_discovery.players``
surface unchanged — including the ``_dot``/``_grad_row``-family
compat re-exports — and aliases ``mlopt_discovery.players.registry``
/ ``._base`` in ``sys.modules`` to the gamekit modules, so both
``from mlopt_discovery.players import registry`` and
``import mlopt_discovery.players.registry`` resolve to the *same*
module object (the registry's mutable ``_INDEX`` stays single).

The player *folders* (``random``/``playbook``/``linear``/``neural``/
``distilled``) are only importable under the canonical
``gamekit.players.<name>`` path — the registry's ``_MODULES`` table
points there; nothing imported them through the discovery path.
"""

from __future__ import annotations

import sys

from gamekit.players import (
    LEGALS,
    PARTS,
    RANKS,
    ActionFeats,
    Bias,
    DomainSpec,
    LearnedPlayer,
    LinearPolicy,
    OpOneHot,
    PlayerSpec,
    PlayHist,
    PlayRates,
    RelFeats,
    StateBroadcast,
    compose_featurizer,
    register_legal,
    register_part,
    register_rank,
    registry,
    spec_from_record,
)
from gamekit.players import _base as _base
from gamekit.players._base import _BASELINE_EMA as _BASELINE_EMA
from gamekit.players._base import _action_key as _action_key
from gamekit.players._base import _dot as _dot
from gamekit.players._base import _grad_row as _grad_row
from gamekit.players._base import _Hist as _Hist

# Alias the submodule names so ``import mlopt_discovery.players.X``
# resolves to the gamekit module object — one registry, one _base.
sys.modules[__name__ + ".registry"] = registry
sys.modules[__name__ + "._base"] = _base

__all__ = [
    "LEGALS",
    "PARTS",
    "RANKS",
    "ActionFeats",
    "Bias",
    "DomainSpec",
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

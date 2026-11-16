"""Baked player weights — the deliver board's learned prior.

Domain data, not code: ``mlopt.play.DELIVERY_SPEC`` binds this table
as its weight prior, so ``linear`` / ``linear-greedy`` play warm the
moment a non-empty literal lands here — ``{}`` keeps them honestly
cold.  Retrain with::

    .venv/bin/python scripts/train_delivery_prior.py \
        --refs <model-ref> [--refs ...] --adapter-opt device=cuda

which prints the ``DELIVERY_PLAYER_WEIGHTS`` literal to paste below
(the ``mlopt_kernel.weights.KERNEL_PLAYER_WEIGHTS`` convention).
"""

from __future__ import annotations

#: The deliver board's player prior — empty until
#: ``scripts/train_delivery_prior.py`` is run on real workloads and
#: its literal pasted here.
DELIVERY_PLAYER_WEIGHTS: dict[str, float] = {}

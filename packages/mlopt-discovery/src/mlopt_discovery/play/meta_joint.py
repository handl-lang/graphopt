"""The ``meta`` and ``joint`` domain arms.

``meta`` plays the meta-arena over the mined-spec corpus;
``joint`` plays a forward term plus its derived gradients under one
``joint`` root on the same board.
"""

from __future__ import annotations

import random
from typing import Any

from mlopt_core.cost.basic import count_cost
from mlopt_core.ir import Op, TensorType, Var

from .. import meta_arena as ma
from .. import meta_player as mp
from .. import training

# ---------------------------------------------------------------------------
#  Domain: meta — the mixed board
# ---------------------------------------------------------------------------


def _meta_board(case: Any) -> ma.MetaArena:
    """Probe-arity case tuple -> MetaArena (count_cost)."""
    return mp._board(case, count_cost, 20_000)


# ---------------------------------------------------------------------------
#  Domain: joint — forward + gradients on one board
# ---------------------------------------------------------------------------


def _joint_cases(seed: int, n: int) -> list:
    """Joint programs over shape-varied forward archetypes.

    Each case joints a forward (silu / softmax-spine / linear+tanh
    at a random width) with every derived gradient under one
    ``joint`` root — the reverse handler's programs are ordinary
    terms, so the meta-arena's move set applies unchanged.
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        d, b = 4 * rng.choice((1, 2)), 4 * rng.choice((1, 2))
        x = Var("x", TensorType((b, d)))
        w = Var("w", TensorType((d, d)))
        archetype = rng.randrange(4)
        if archetype == 0:
            fwd = Op.make("mul", x, Op.make("sigmoid", x))
        elif archetype == 1:
            fwd = Op.make("softmax", x, dim=-1)
        elif archetype == 2:
            fwd = Op.make(
                "div",
                Op.make("exp", x),
                Op.make("sum", Op.make("exp", x), dim=1, keepdim=True),
            )
        else:
            fwd = Op.make("tanh", Op.make("matmul", x, w))
        cot = training._cotangent_for(fwd)
        grads = training.backward(fwd, cotangent=cot)
        joint = Op.make(
            "joint",
            fwd,
            *[grads[k] for k in sorted(grads)],
            validate=False,
        )
        out.append((f"joint{i}:{archetype}", joint))
    return out


def _joint_board(case: Any) -> ma.MetaArena:
    """Board a joint program under the meta-arena's moves."""
    return ma.MetaArena(case[1], cost_fn=count_cost)

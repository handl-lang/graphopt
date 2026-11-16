"""Ambient computation-dtype context for creator ops.

``_EvalPlan`` / ``eval_term`` already resolve a *like* tensor — the
first floating operand or Param — to mint ``Const`` scalars at the
computation's dtype (``_mint_const``: an ambient-fp64 process default
must not promote an fp32 delivery, and vice versa).  Creator ops with
no dtype attr had no such channel: ``torch.arange``/``zeros``/``full``
fall back to torch's *ambient* default dtype (float32), so an
unPinned float ``arange`` inside an fp64 evaluation silently computed
in f32 — the llama verify-run 4.5e-7 divergence (the whole RoPE table
— positions → inv_freq → freqs → sin/cos — ran f32 on the reference
side while host-tensor's dense-f64 ``tensor.*`` ran f64; position 0
stayed exact because ``freqs = 0`` gives ``sin = 0``/``cos = 1``
regardless).

This module carries that same *like* tensor through a
:class:`contextvars.ContextVar` for the float creators that have no
operand to infer from.  It is set only while an ``IRModule`` /
``eval_term`` evaluation runs; outside eval the var is ``None`` and
``_creator_dtype`` keeps returning ``None`` — torch's default, the
historical behaviour for standalone binding calls.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import torch

#: The reference tensor whose dtype/device creator ops inherit —
#: ``None`` outside an eval (ambient torch default applies).
LIKE_CTX: ContextVar[Any] = ContextVar(
    "mlopt_torch_like_ctx", default=None
)


def get() -> Any:
    """Return the current like tensor, or ``None`` outside eval."""
    return LIKE_CTX.get()


def like_dtype() -> Any:
    """``like.dtype`` when the context tensor is floating/complex.

    Integer contexts contribute nothing — an int-bound ``arange``
    must keep torch's int64 inference, not be dragged to a float
    dtype (T5's position-bucket arithmetic depends on it).
    """
    like = LIKE_CTX.get()
    if isinstance(like, torch.Tensor) and (
        like.is_floating_point() or like.is_complex()
    ):
        return like.dtype
    return None

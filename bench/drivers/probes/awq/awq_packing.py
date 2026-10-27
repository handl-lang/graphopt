"""AWQ nibble order: the convention my unpack was missing.

AWQ packs 8 int4 values into an int32 in the order `[0, 4, 1, 5, 2, 6,
3, 7]` (vLLM's `_REVERSE_AWQ_PACK_ORDER`), not in plain `0..7` order.
My `unpack_ref` and the kernel both assumed plain order — and because the
"reference" used the same wrong convention, the standalone check passed
while the model diverged at the first token (cosine 0.19 against
`awq_gemm`).

:func:`to_standard_order` rewrites a packed tensor into plain order, so a
kernel written for the standard convention reads the right values.
"""

from __future__ import annotations

import torch

__all__ = ["AWQ_PACK_PERM", "to_standard_order"]

#: vLLM's `_REVERSE_AWQ_PACK_ORDER`
AWQ_PACK_PERM = (0, 4, 1, 5, 2, 6, 3, 7)


def to_standard_order(packed: torch.Tensor) -> torch.Tensor:
    """Rewrite int4-packed int32 so nibble *j* is output column *j*.

    ``packed`` is ``[rows, cols//8]``; the result has the same shape and
    dtype, with each word's nibbles permuted from AWQ's order into plain
    order. Values and layout are otherwise untouched — no unpacking to a
    wide tensor, so the memory cost is one int32 per word.
    """
    shifts = torch.arange(
        0, 32, 4, dtype=torch.int32, device=packed.device
    )
    nibbles = (
        packed.unsqueeze(-1) >> shifts
    ) & 0xF  # [rows, cols/8, 8]
    nibbles = nibbles[:, :, list(AWQ_PACK_PERM)]
    return (
        (nibbles.to(torch.int32) << shifts).sum(
            dim=-1, dtype=torch.int32
        )
    ).contiguous()

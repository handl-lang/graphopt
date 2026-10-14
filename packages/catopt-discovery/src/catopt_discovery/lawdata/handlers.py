"""Interpretation tables — handlers and the reverse-mode VJP table.

The ``handle`` move's Sanada-arrow handler table (tag → pattern /
kernel / args) and the reverse-mode VJP table ``training.backward``
reads to *derive* a gradient program.  Both are scoped
interpretations as data, not hand-written optimizer rules.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "HANDLERS",
    "REVERSE",
]

#: The ``handle`` move's interpretation table — the Sanada arrow
#: handler made data: tag → ``{pattern, kernel, args}``.  A handler
#: is a *scoped interpretation* of a declared object: ``pattern`` is
#: the spelled form the kernel covers (spec syntax — ``str`` leaves
#: are metavars), ``kernel`` the op name the interpretation lowers
#: the declared abbreviation to, ``args`` the kernel's argument
#: order over the pattern's metavars.  The meta-arena prices a
#: handled name at the kernel's cost instead of spelled parity —
#: under an op-count/launch-aware model the fused interpretation is
#: measurably cheaper; under ``flops_cost`` it is honestly flat
#: (fused and spelled move the same data — the evaluation gap the
#: arena-probe retro documented).  A handler whose kernel is outside
#: the board's ``supported`` bound applies but prices infeasible.
HANDLERS: dict[str, dict] = {
    "silu": {
        "pattern": ("mul", "X", ("sigmoid", "X")),
        "kernel": "silu",
        "args": ("X",),
    },
    "square": {
        "pattern": ("mul", "X", "X"),
        "kernel": "square",
        "args": ("X",),
    },
    "swiglu": {
        "pattern": (
            "mul",
            ("mul", "A", ("sigmoid", "A")),
            "B",
        ),
        "kernel": "swiglu",
        "args": ("A", "B"),
    },
}

# ---------------------------------------------------------------------------
#  The reverse handler — backprop as a data-driven interpretation
# ---------------------------------------------------------------------------

#: Reverse-mode VJP table — ``op -> (per-arg grad spec, ...)``.  A
#: spec is data: ``"$g"`` is the node's output cotangent, ``"$a0"``,
#: ``"$a1"``… its forward arguments; ``None`` marks a child that
#: receives no gradient (index/shape arguments).  ``backward()``
#: reads this table to *derive* the gradient program — the reverse
#: handler is interpretation, not a hand-written optimizer rule.
#: ``"$attrs"`` inside a spec splices the forward node's attrs onto
#: the built op (transpose re-applies with its dims, etc.).
#: Shape-position splices resolve from the forward node's inputs:
#: ``"$shape:N"`` is arg N's inferred shape, ``"$rcount"`` a Const
#: counting a reduction's elements, ``"$invdims"`` a permute's
#: inverse permutation.  A row may be ``{"args": (…), "requires":
#: ((attr, value), …)}`` — ``backward`` declines (ValueError) when
#: the node's attrs fail the guard, e.g. ``sum`` with
#: ``keepdim=False`` whose cotangent needs an un-reduce first.
#:
#: Honest scope: shape-free ops only.  Reductions (``sum``/``mean``)
#: need a broadcast back to the input's shape — spelled with
#: ``broadcast_to`` when the caller supplies a shaped cotangent —
#: and reshape/expand-class ops are view-shaped.  Those VJPs are
#: expressible but carry shape data the spec grammar does not yet
#: bind; they decline (``backward`` raises) rather than guess.
REVERSE: dict[str, Any] = {
    "add": ("$g", "$g"),
    "sub": ("$g", ("neg", "$g")),
    "neg": (("neg", "$g"),),
    "mul": (("mul", "$g", "$a1"), ("mul", "$g", "$a0")),
    "div": (
        ("div", "$g", "$a1"),
        ("mul", "$g", ("neg", ("div", "$a0", ("mul", "$a1", "$a1")))),
    ),
    "matmul": (
        ("matmul", "$g", ("transpose", "$a1")),
        ("matmul", ("transpose", "$a0"), "$g"),
    ),
    "transpose": (("transpose", "$g", "$attrs"),),
    "sigmoid": (
        (
            "mul",
            "$g",
            ("mul", ("sigmoid", "$a0"), ("sub", 1, ("sigmoid", "$a0"))),
        ),
    ),
    "tanh": (
        (
            "mul",
            "$g",
            ("sub", 1, ("mul", ("tanh", "$a0"), ("tanh", "$a0"))),
        ),
    ),
    "exp": (("mul", "$g", ("exp", "$a0")),),
    "log": (("div", "$g", "$a0"),),
    "sqrt": (("div", "$g", ("mul", 2, ("sqrt", "$a0"))),),
    "square": (("mul", "$g", ("mul", 2, "$a0")),),
    # alternative rows: the first whose ``requires`` all hold wins;
    # ``"$ABSENT"`` means the attr must be missing.  sum/mean with a
    # dim but keepdim=False decline — the cotangent would need an
    # un-reduce the spec cannot spell.
    "sum": [
        {
            "args": (("broadcast_to", "$g", {"shape": "$shape:0"}),),
            "requires": (("keepdim", True),),
        },
        {
            "args": (("broadcast_to", "$g", {"shape": "$shape:0"}),),
            "requires": (("dim", "$ABSENT"),),
        },
    ],
    "mean": [
        {
            "args": (
                (
                    "div",
                    ("broadcast_to", "$g", {"shape": "$shape:0"}),
                    "$rcount",
                ),
            ),
            "requires": (("keepdim", True),),
        },
        {
            "args": (
                (
                    "div",
                    ("broadcast_to", "$g", {"shape": "$shape:0"}),
                    "$rcount",
                ),
            ),
            "requires": (("dim", "$ABSENT"),),
        },
    ],
    "reshape": (("reshape", "$g", {"shape": "$shape:0"}),),
    "permute": (("permute", "$g", {"dims": "$invdims"}),),
    # softmax(x)⊙(g - Σ_dim g⊙softmax(x)) — the spelled body
    # references its own forward output by re-spelling it; the
    # joint e-graph is where that reuse dedups.
    "softmax": (
        (
            "mul",
            ("softmax", "$a0", "$attrs"),
            (
                "sub",
                "$g",
                (
                    "sum",
                    ("mul", "$g", ("softmax", "$a0", "$attrs")),
                    {"dim": "$dim", "keepdim": True},
                ),
            ),
        ),
    ),
    # log_softmax: s = exp(lsm(x)); dx = g - s⊙(Σ_dim g) — the
    # forward output is again spelled inside (exp∘lsm), deduping
    # under the joint e-graph.
    "log_softmax": (
        (
            "sub",
            "$g",
            (
                "mul",
                ("exp", ("log_softmax", "$a0", "$attrs")),
                ("sum", "$g", {"dim": "$dim", "keepdim": True}),
            ),
        ),
    ),
    # relu: grad = g * (x > 0) — the mask is a gt on the input
    "relu": (("mul", "$g", ("gt", "$a0", 0)),),
    # rank ops: backward swaps squeeze/unsqueeze, same dims
    "unsqueeze": (("squeeze", "$g", "$attrs"),),
    "squeeze": (("unsqueeze", "$g", "$attrs"),),
    # linear(x, W, b) = xW^T + b:  dx = gW, dW = g^T x,
    # db = sum over the batch dims ($batchsum — leading dims of g)
    "linear": (
        ("matmul", "$g", "$a1"),
        ("matmul", ("transpose", "$g"), "$a0"),
        "$batchsum",
    ),
    # an expand/broadcast copies the input along $expdims; the
    # cotangent sums those copies back, keepdim keeps the rank,
    # reshape restores the exact input shape (new leading dims
    # drop out through the same sum-then-reshape).
    "expand": (
        (
            "reshape",
            (
                "sum",
                "$g",
                {"dim": "$expdims", "keepdim": True},
            ),
            {"shape": "$shape:0"},
        ),
    ),
    "broadcast_to": (
        (
            "reshape",
            (
                "sum",
                "$g",
                {"dim": "$expdims", "keepdim": True},
            ),
            {"shape": "$shape:0"},
        ),
    ),
}

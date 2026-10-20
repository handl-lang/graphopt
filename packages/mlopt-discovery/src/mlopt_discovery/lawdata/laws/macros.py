"""The shipped macro table — minted move sequences as baked data.

Like the learned-prior tables in
:mod:`mlopt_discovery.lawdata.game.weights`, this is *content*:
the 8-macro set ``scripts/macro_miner.py`` mints off the train
stream (``meta_player.gen_cases(0, 24)`` — 206 trajectories, mined
n-grams ranked by ``count x mean terminal reward``), serialized as
plain records and replayed by :mod:`mlopt_discovery.macro_moves`
through the refereed ``"macro"`` handler.  ``meta_arena.legal_actions``
offers the viable entries to every player — a macro mints no
behaviour a player could not already spell; it compresses the line
into one step.

Each row: ``name`` (ops + the handler tag each carries), ``seq``
(the recorded ``{"op", "params"}`` sub-actions — ``declare``'s
``spelled`` spec and ``saturate``'s budget replay verbatim), and the
miner's evidence — ``count`` occurrences, ``certified`` share in
certificate-passing episodes, ``mean_terminal`` reward, ``score``
the rank key.  Regenerate with::

    .venv/bin/python scripts/macro_miner.py --emit
"""

from __future__ import annotations

__all__ = ["MACROS"]

#: The shipped offer set, rank order — ``viable`` filters per state.
MACROS: tuple[dict, ...] = (
    {  # claim(silu)→extract — the fused winning line
        "name": "claim_silu_extract",
        "seq": (
            {"op": "claim", "params": {"handler": "silu"}},
            {"op": "extract", "params": {}},
        ),
        "count": 14,
        "certified": 1.0,
        "mean_terminal": 0.2952,
        "score": 4.1333,
    },
    {  # declare(silu-spec)→saturate — mints + fires the fold
        "name": "declare_saturate",
        "seq": (
            {
                "op": "declare",
                "params": {
                    "construction": {
                        "op": "fold",
                        "params": {
                            "name": "foldabs_0",
                            "spelled": ("mul", "X1", ("sigmoid", "X1")),
                            "kernel": ("foldabs_0", "X1"),
                        },
                    },
                    "unfold": True,
                },
            },
            {
                "op": "saturate",
                "params": {
                    "rules": None,
                    "budget": None,
                    "iterations": 8,
                    "stop": "fixed_point",
                    "patience": 3,
                },
            },
        ),
        "count": 13,
        "certified": 1.0,
        "mean_terminal": 0.2962,
        "score": 3.85,
    },
    {  # declare→saturate→handle(silu) — the spelled line, bound
        "name": "declare_saturate_handle_silu",
        "seq": (
            {
                "op": "declare",
                "params": {
                    "construction": {
                        "op": "fold",
                        "params": {
                            "name": "foldabs_0",
                            "spelled": ("mul", "X1", ("sigmoid", "X1")),
                            "kernel": ("foldabs_0", "X1"),
                        },
                    },
                    "unfold": True,
                },
            },
            {
                "op": "saturate",
                "params": {
                    "rules": None,
                    "budget": None,
                    "iterations": 8,
                    "stop": "fixed_point",
                    "patience": 3,
                },
            },
            {
                "op": "handle",
                "params": {"object": "foldabs_0", "handler": "silu"},
            },
        ),
        "count": 13,
        "certified": 1.0,
        "mean_terminal": 0.2962,
        "score": 3.85,
    },
    {  # declare→saturate→handle(silu)→extract — the mined win
        "name": "declare_saturate_handle_silu_extract",
        "seq": (
            {
                "op": "declare",
                "params": {
                    "construction": {
                        "op": "fold",
                        "params": {
                            "name": "foldabs_0",
                            "spelled": ("mul", "X1", ("sigmoid", "X1")),
                            "kernel": ("foldabs_0", "X1"),
                        },
                    },
                    "unfold": True,
                },
            },
            {
                "op": "saturate",
                "params": {
                    "rules": None,
                    "budget": None,
                    "iterations": 8,
                    "stop": "fixed_point",
                    "patience": 3,
                },
            },
            {
                "op": "handle",
                "params": {"object": "foldabs_0", "handler": "silu"},
            },
            {"op": "extract", "params": {}},
        ),
        "count": 13,
        "certified": 1.0,
        "mean_terminal": 0.2962,
        "score": 3.85,
    },
    {  # handle(foldabs_0, silu)→extract — completes a played declare
        "name": "handle_silu_extract",
        "seq": (
            {
                "op": "handle",
                "params": {"object": "foldabs_0", "handler": "silu"},
            },
            {"op": "extract", "params": {}},
        ),
        "count": 13,
        "certified": 1.0,
        "mean_terminal": 0.2962,
        "score": 3.85,
    },
    {  # saturate→handle(foldabs_0, silu)
        "name": "saturate_handle_silu",
        "seq": (
            {
                "op": "saturate",
                "params": {
                    "rules": None,
                    "budget": None,
                    "iterations": 8,
                    "stop": "fixed_point",
                    "patience": 3,
                },
            },
            {
                "op": "handle",
                "params": {"object": "foldabs_0", "handler": "silu"},
            },
        ),
        "count": 13,
        "certified": 1.0,
        "mean_terminal": 0.2962,
        "score": 3.85,
    },
    {  # saturate→handle(foldabs_0, silu)→extract
        "name": "saturate_handle_silu_extract",
        "seq": (
            {
                "op": "saturate",
                "params": {
                    "rules": None,
                    "budget": None,
                    "iterations": 8,
                    "stop": "fixed_point",
                    "patience": 3,
                },
            },
            {
                "op": "handle",
                "params": {"object": "foldabs_0", "handler": "silu"},
            },
            {"op": "extract", "params": {}},
        ),
        "count": 13,
        "certified": 1.0,
        "mean_terminal": 0.2962,
        "score": 3.85,
    },
    {  # saturate→handle(foldabs_1, swiglu) — the second site's line
        "name": "saturate_handle_swiglu",
        "seq": (
            {
                "op": "saturate",
                "params": {
                    "rules": None,
                    "budget": None,
                    "iterations": 8,
                    "stop": "fixed_point",
                    "patience": 3,
                },
            },
            {
                "op": "handle",
                "params": {"object": "foldabs_1", "handler": "swiglu"},
            },
        ),
        "count": 7,
        "certified": 0.857,
        "mean_terminal": 0.5286,
        "score": 3.7,
    },
)

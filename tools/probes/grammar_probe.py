"""Grammar-refusal census — candidate counts per spec per family.

The ADR-0007 layer-2 handler grammar (``catopt_discovery.handler_synth``)
is verify-or-refuse: a spec the templates cannot shape refuses with
*zero candidates proposed*, and a spec the templates shape but cannot
match refuses with *candidates exhausted*.  This instrument separates
the two by counting proposals, per family, over the measured
refusal battery — the SelectiveScan / cumsum-family specs the brief
names.

Run::

    .venv/bin/python tools/probes/grammar_probe.py
    .venv/bin/python tools/probes/grammar_probe.py cumsum-verbatim softmax
"""

from __future__ import annotations

import sys
from itertools import islice

from catopt_discovery import handler_synth as hs
from catopt_discovery.genkernel import _spec_metas

#: The measured refusal battery — each spelling is a corpus shape the
#: grammar was asked to mint.  ``selective-*`` are the SelectiveScan
#: site (clamped verbatim / explicit-carry), ``cumsum-*`` the
#: carry-less and initialized prefix sums, ``*-reduce``/``softmax``
#: the reduction family.
_CP = ("cumprod", ("sigmoid", "A"), {"dim": 1})
_CPC = ("clamp_min", _CP, {"min": 1e-6})
_H0U = ("unsqueeze", "H0", {"dim": 1})

SPECS: dict[str, object] = {
    "selective-verbatim": (
        "mul",
        _CP,
        ("cumsum", ("div", "B", _CPC), {"dim": 1}),
    ),
    "cumsum-verbatim": ("cumsum", "X", {"dim": 1}),
    "cumsum-init": (
        "add",
        _H0U,
        ("cumsum", "X", {"dim": 1}),
    ),
    "sum-reduce": ("sum", "X", {"dim": -1}),
    "mean-reduce": ("mean", "X", {"dim": -1}),
    "softmax": (
        "div",
        ("exp", "X"),
        ("sum", ("exp", "X"), {"dim": -1, "keepdim": True}),
    ),
    "softmax-nokeepdim": (
        "div",
        ("exp", "X"),
        ("sum", ("exp", "X"), {"dim": -1}),
    ),
    "selective-init-clamped": (
        "mul",
        _CP,
        ("add", _H0U, ("cumsum", ("div", "B", _CPC), {"dim": 1})),
    ),
}


def count(
    spec: object, families: tuple | None = None, cap: int = 2_000_000
):
    """Count proposals for *spec*, optionally restricted to *families*.

    Streamed with ``islice`` so an unbounded column walk cannot
    materialize — the census is a count, never a list.
    """
    n = 0
    for _ in islice(hs.propose_bodies(spec, families=families), cap):
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    """Print one census row per battery spec."""
    which = argv or list(SPECS)
    for name in which:
        spec = SPECS[name]
        nmetas = len(_spec_metas(spec))
        per = {f: count(spec, families=(f,)) for f in hs._FAMILIES}
        print(f"{name:<26} metas={nmetas} total={count(spec)} {per}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

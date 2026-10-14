"""Vocabulary *proposal* — turn a measured need into a checkable atom.

``catopt_discovery.vocab`` measures the vocabulary the derive bank
*lacks*: :func:`vocab.detect_need` / :func:`vocab.scan_needs` rank the
missing bank ops by how many procedural derives each would unblock, and
:func:`vocab.propose_atom` turns one need into a ``declare``-shaped
proposal.  This module extends the *proposal* side (plan 0024 part B):

* :func:`build_proposal` emits a **complete, checkable** proposal — the
  declared op with its arity/attrs, its observable semantics table, and
  an **observational semantics probe**: the *spelled* form (a term over
  the existing derive DSL) the atom must equal, evaluated on real ints
  so a human or the gauntlet can verify it.  ``mod`` *is* bank-
  derivable (``a - b*(a//b)``), so its probe carries a spelled term and
  the check agrees; ``pow`` is **primitive** — no expression in the
  shipped bank spells it — so its probe is the honest ``None``.

* :func:`mod_example` and :func:`shape_source_example` are the two
  **worked examples** the plan's measured backlog names: ``mod`` (the
  measured leader — six carrier rules) as a *bank-op* proposal, and the
  carrier-aware **shape source** (the measured ``vshape`` gap) as a
  *structural* proposal.  The shape-source case is where the arity-2
  probe honestly finds **nothing** — no bank op unblocks the carrier-
  aware planner declines — so the machinery proposes a *shape source*
  (``vshape``), not an atom.

* :func:`backlog` runs the whole carrier rule set through
  :func:`vocab.scan_needs` and pairs every ranked need with its
  proposal; :func:`main` prints that artifact — the ranked backlog with
  the per-need proposal and the measured evidence (agreement counts).

A proposal is data, never an admission (ADR 0005): ``admitted`` is
always ``False``; a human or the gauntlet acts on it.

Run::

    .venv/bin/python -m catopt_discovery.propose
    .venv/bin/python -m catopt_discovery.propose --rules synthetic
    .venv/bin/python -m catopt_discovery.propose \
        --deadline 30 --max-rules 12 --json /tmp/propose.json

CPU-only, bounded by ``--deadline`` / ``--max-rules``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws.cond import _DOPS, _Decline

from catopt_discovery import vocab as vc

__all__ = [
    "SPELLED",
    "StructuralNeed",
    "backlog",
    "build_proposal",
    "check_spelled",
    "examples",
    "main",
    "mod_example",
    "render_term",
    "shape_source_example",
    "spelled_term",
]

# ---------------------------------------------------------------------------
#  Spelled forms — the atom's semantics in the existing derive DSL
# ---------------------------------------------------------------------------
#
#  A spelled form is a ``_DOPS`` term over the two bound operand values
#  ``("attr", "a")`` / ``("attr", "b")`` that the minted atom must equal
#  observationally.  ``mod`` is bank-derivable (``a - b*(a//b)`` equals
#  ``a % b`` for every int pair, decline included); ``pow`` is a
#  *primitive* — the bank has no expression for a variable exponent, so
#  its spelled form is honestly absent and the atom is declared
#  primitive.  The distinction is the whole point: a minted op that the
#  bank can already spell is sugar; one it cannot is a new primitive.

#: The spelled form of each candidate bank op, or ``None`` when the op
#: is primitive (not expressible with the shipped derive bank).
SPELLED: dict[str, Any] = {
    "mod": (
        "sub",
        ("attr", "a"),
        (
            "mul",
            ("attr", "b"),
            ("floordiv", ("attr", "a"), ("attr", "b")),
        ),
    ),
    "pow": None,
}

#: The two operand metavariable names the spelled forms read.
_OPERANDS: tuple[str, str] = ("a", "b")


def spelled_term(op: str) -> Any:
    """Return the spelled form of *op* — a DSL term, or ``None``.

    ``None`` means the op is **primitive**: no expression over the
    shipped bank reproduces its declared semantics (a variable
    exponent), so the proposal declares the atom rather than unfolding
    it.  An unknown op raises ``ValueError``.
    """
    if op not in SPELLED:
        raise ValueError(f"unknown candidate op: {op!r}")
    return SPELLED[op]


def render_term(term: Any) -> str:
    """Render a spelled DSL term as a readable infix-ish string."""
    if isinstance(term, str):
        return term
    if not isinstance(term, tuple):
        return repr(term)
    if term[0] == "attr":
        return str(term[1])
    if term[0] == "lit":
        return repr(term[1])
    return f"{term[0]}({', '.join(render_term(a) for a in term[1:])})"


def _eval_term(term: Any, a: Any, b: Any) -> Any:
    """Evaluate a spelled DSL term on the operand values, or ``None``."""
    bound = {"$attr:a": a, "$attr:b": b}
    try:
        return _DOPS[term[0]](term[1:], bound)
    except _Decline:
        return None


def _atom_value(op: str, a: Any, b: Any) -> Any:
    """Evaluate the atom's shipped handler on the operand values."""
    return vc._candidate_value(op)(a, b)


def check_spelled(
    op: str, *, inputs: tuple[tuple[int, int], ...] | None = None
) -> dict[str, Any]:
    """Return the observational check of *op* against its spelled form.

    Evaluates the atom's shipped handler and its spelled form on every
    probe input and reports the per-input agreement.  ``agrees`` is
    ``True`` when they match everywhere (declines included — a decline
    on one side must be a decline on the other), ``False`` on a
    disagreement, and ``None`` when the op is primitive (no spelled
    form to check).  This is the *checkable* half of the proposal: the
    gauntlet can replay it on real ints.
    """
    spelled = spelled_term(op)
    pairs = vc._PROBE_INPUTS if inputs is None else tuple(inputs)
    if spelled is None:
        return {
            "spelled": None,
            "rows": [
                {"in": [a, b], "atom": _atom_value(op, a, b)}
                for a, b in pairs
            ],
            "agrees": None,
        }
    rows: list[dict[str, Any]] = []
    agrees = True
    for a, b in pairs:
        atom = _atom_value(op, a, b)
        spelled_v = _eval_term(spelled, a, b)
        ok = atom == spelled_v
        agrees = agrees and ok
        rows.append(
            {
                "in": [a, b],
                "atom": atom,
                "spelled": spelled_v,
                "agree": ok,
            }
        )
    return {
        "spelled": render_term(spelled),
        "rows": rows,
        "agrees": agrees,
    }


# ---------------------------------------------------------------------------
#  The complete proposal
# ---------------------------------------------------------------------------


def build_proposal(need: Any) -> dict[str, Any]:
    """Emit a complete, checkable ``declare`` proposal for *need*.

    Starts from :func:`vocab.propose_atom`'s declare shape and completes
    the atom: its arity and (empty) attrs, the spelled form (a term, or
    ``None`` for a primitive), the observational check
    (:func:`check_spelled`), and a ``primitive`` flag.  ``admitted``
    stays ``False`` — proposing is machine work, admitting is gated.
    """
    prop = vc.propose_atom(need)
    atom = prop["params"]["construction"]["params"]
    spelled = spelled_term(need.op)
    atom["attrs"] = {}
    atom["operands"] = list(_OPERANDS)
    atom["spelled"] = None if spelled is None else render_term(spelled)
    atom["primitive"] = spelled is None
    atom["spelled_check"] = check_spelled(need.op)
    return prop


# ---------------------------------------------------------------------------
#  Worked examples — mod (the leader) and the carrier-aware shape source
# ---------------------------------------------------------------------------


def _rank_var(shape: tuple) -> Var:
    """Return a ``Var`` named ``x`` of *shape*."""
    return Var("x", TensorType(shape))


def _rank_domain() -> list[dict]:
    """Five bindings whose ranks interleave — no single rank fits."""
    return [
        {"x": _rank_var(sh)}
        for sh in [
            (4,),
            (2, 4),
            (2, 3, 4),
            (2, 2, 3, 4),
            (2, 2, 2, 3, 4),
        ]
    ]


def _rank_of(bound: dict) -> int | None:
    """Return the rank of the bound ``x`` term, or ``None``."""
    sh = getattr(getattr(bound.get("x"), "typ", None), "shape", None)
    return len(sh) if sh else None


def _mod_rank_hook(bound: dict) -> dict | None:
    """Return a derive spelling ``rank % 2`` — the ``mod`` need."""
    r = _rank_of(bound)
    return None if r is None else {"$attr:M": r % 2}


def _pow_rank_hook(bound: dict) -> dict | None:
    """Return a derive spelling ``2 ** rank`` — the ``pow`` need."""
    r = _rank_of(bound)
    return None if r is None else {"$attr:M": 2**r}


def _vshape_rank_hook(bound: dict) -> dict | None:
    """Return a carrier-aware derive: ``rank(value-shape) - 1``.

    Reads the value shape (``vshape``) rather than the cost convention,
    so on a *carrier member* whose true value rank differs from its
    convention rank it is not expressible with ``shape`` — the measured
    structural gap.  On the synthesis domain (leaf-bound terms) the two
    agree, which is exactly why the arity-2 probe finds nothing.
    """
    from catopt_core.typing import value_shape_of

    r = getattr(getattr(bound.get("x"), "typ", None), "shape", None)
    if not r:
        return None
    vs = value_shape_of(bound.get("x"), None)
    return {"$attr:M": len(vs) - 1 if vs else 0}


@dataclass(frozen=True)
class StructuralNeed:
    """A measured gap no arity-2 bank op unblocks — a shape source.

    ``op`` is the *source* to declare (``vshape``), ``kind`` its
    category (``"shape-source"``), ``rules_blocked`` the derives the
    source would unblock, and ``note`` the honest reason the arity-2
    probe found nothing.  The arity-2 ``needs`` are the vocabulary the
    bank *lacks*; a :class:`StructuralNeed` is the same question one
    level up — a *structural* gap, not a missing op.
    """

    op: str
    kind: str
    rules_blocked: tuple[str, ...] = ()
    note: str = ""

    def to_data(self) -> dict[str, Any]:
        """Return the finding as JSON-safe data."""
        return {
            "op": self.op,
            "kind": self.kind,
            "rules_blocked": list(self.rules_blocked),
            "note": self.note,
        }


def mod_example() -> dict[str, Any]:
    """Return the ``mod`` worked example: a measured *bank-op* need.

    Measures :func:`_mod_rank_hook` over the interleaved-rank domain,
    detects the ``mod`` need, and builds the complete proposal with its
    spelled form (``a - b*(a//b)``) and the agreeing observational
    check.  ``None`` in the ``need`` slot would mean the derive was
    already expressible — the honest refusal.
    """
    need = vc.detect_need(_mod_rank_hook, _rank_domain())
    return {
        "name": "mod",
        "kind": "bank-op",
        "need": None if need is None else need.to_data(),
        "proposal": None if need is None else build_proposal(need),
    }


def propose_shape_source(need: StructuralNeed) -> dict[str, Any]:
    """Emit a ``declare``-shaped proposal for a shape-source *need*.

    The structural twin of :func:`build_proposal`: the atom is a
    *shape source* (``vshape``) whose spelled form is the default
    resolver (``shape``), so the observational check is the honest
    "on a plain term ``vshape`` equals ``shape``; only a carrier member
    separates them".  ``admitted`` is always ``False``.
    """
    spelled = ("shape", "T")
    return {
        "op": "declare",
        "params": {
            "construction": {
                "op": "atom",
                "params": {
                    "kind": need.kind,
                    "name": need.op,
                    "arity": 1,
                    "attrs": {},
                    "semantics": (
                        "value-shape source — the carrier-aware rank "
                        "of a bound term"
                    ),
                    "spelled": render_term(spelled),
                    "primitive": False,
                    "observable": _shape_source_check(),
                },
            },
            "unfold": False,
        },
        "evidence": need.to_data(),
        "admitted": False,
    }


def _shape_source_check() -> dict[str, Any]:
    """Return the observational check of ``vshape`` against ``shape``.

    On a leaf-bound term (the synthesis domain) the registered
    value-shape resolver is absent, so ``vshape`` falls back to the
    cost convention and the two agree — the measured
    ``vshape_diff_envs == 0`` finding.  The two separate only at
    law-evaluation time, on a carrier member the synth cannot reach.
    """
    x = _rank_var((2, 3))
    bound = {"x": x}
    shape = _shape_of_op("shape", bound)
    vshape = _shape_of_op("vshape", bound)
    return {
        "term": "x: (2, 3)",
        "shape": list(shape) if isinstance(shape, tuple) else None,
        "vshape": list(vshape) if isinstance(vshape, tuple) else None,
        "agrees": shape == vshape,
        "note": (
            "on the synthesis domain vshape == shape (the measured "
            "vshape_diff_envs == 0); they differ only on a carrier "
            "member at law-evaluation time"
        ),
    }


def _shape_of_op(op: str, bound: dict) -> Any:
    """Evaluate a shape-source op on the bound ``x`` term, or ``None``."""
    try:
        return _DOPS[op](("x",), bound)
    except _Decline:
        return None


def shape_source_example() -> dict[str, Any]:
    """Return the carrier-aware shape-source worked example.

    Demonstrates the measured ``vshape`` gap as a *structural* need:
    the arity-2 probe over :func:`_vshape_rank_hook` finds **nothing**
    (``detect_need`` refuses — on the synthesis domain ``vshape`` and
    ``shape`` agree, so the hook is expressible and no bank op helps),
    and the machinery proposes a shape *source* rather than an atom.
    """
    need = vc.detect_need(_vshape_rank_hook, _rank_domain())
    structural = StructuralNeed(
        op="vshape",
        kind="shape-source",
        rules_blocked=(),
        note=(
            "no arity-2 bank op unblocks the carrier-aware planner "
            "declines; the gap is a shape source, not a missing op "
            "(the measured vshape_diff_envs == 0 on the synth domain)"
        ),
    )
    return {
        "name": "vshape",
        "kind": "shape-source",
        "arity2_need": None if need is None else need.to_data(),
        "arity2_refused": need is None,
        "proposal": propose_shape_source(structural),
    }


def examples() -> list[dict[str, Any]]:
    """Return both worked examples: ``mod`` and the shape source."""
    return [mod_example(), shape_source_example()]


# ---------------------------------------------------------------------------
#  The backlog driver
# ---------------------------------------------------------------------------


def backlog(
    rules: Any,
    *,
    measure: Any = None,
    synth_limit: int = 60,
    max_depth: int = 2,
    deadline: float | None = None,
    max_rules: int | None = None,
) -> list[dict[str, Any]]:
    """Rank the missing bank ops and pair each with its proposal.

    Runs :func:`vocab.scan_needs` over *rules* (bounded by *deadline* /
    *max_rules*) and builds the complete proposal for every ranked
    need.  Returns the ranked backlog — the artifact a human acts on,
    each row ``{"need": …, "proposal": …}``.  A partial scan (a
    deadline) is honest: fewer rows, never a wrong one.
    """
    needs = vc.scan_needs(
        rules,
        synth_limit=synth_limit,
        max_depth=max_depth,
        deadline=deadline,
        max_rules=max_rules,
        measure=measure,
    )
    return [
        {"need": n.to_data(), "proposal": build_proposal(n)}
        for n in needs
    ]


def _synthetic_rules() -> list[Any]:
    """Return a small self-contained rule set for the ``synthetic`` mode.

    Two procedural derives — a ``mod`` need and a ``pow`` need — so the
    driver runs in well under a second without the carrier package.
    """
    from catopt_core.egraph import Rewrite

    def _rule(name: str, hook: Any) -> Any:
        return Rewrite(
            name,
            Op.make("add", "x", "y"),
            Op.make("add", "x", "y"),
            derive=hook,
        )

    return [
        _rule("synth_mod", _mod_rank_hook),
        _rule("synth_pow", _pow_rank_hook),
    ]


def _synthetic_measure(
    rule: Any, corpus: Any, limit: int
) -> list[tuple]:
    """Return a measure stub over the interleaved-rank domain."""
    return [(env, "equal") for env in _rank_domain()]


def _carrier_rules() -> list[Any]:
    """Return every carrier law (the residual procedural derives)."""
    from catopt_carriers import (
        decode_geom,
        decode_laws,
        om,
        trace,
        xcarrier,
    )

    return [
        *om.OM_RULES,
        *trace.TRACE_RULES,
        *xcarrier.XC_RULES,
        *decode_laws.DECODE_RULES,
        *decode_geom.DECODE_GEOM_RULES,
    ]


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _fmt_proposal(prop: dict[str, Any]) -> list[str]:
    """Render one proposal's atom, spelled form and check."""
    atom = prop["params"]["construction"]["params"]
    lines = [
        f"     declare {atom['kind']} {atom['name']} "
        f"(arity {atom['arity']})",
        f"       semantics: {atom['semantics']}",
        f"       spelled:   {atom.get('spelled') or '(primitive)'}",
    ]
    check = atom.get("spelled_check") or atom.get("observable") or {}
    rows = check.get("rows")
    if isinstance(rows, list) and rows:
        pairs = ", ".join(
            f"{r['in']}->{r.get('atom')}"
            + ("" if "spelled" not in r else f"/{r['spelled']}")
            for r in rows
        )
        lines.append(f"       probe:     {pairs}")
    if "agrees" in check:
        lines.append(f"       agrees:    {check['agrees']}")
    return lines


def _fmt_backlog(rows: list[dict[str, Any]]) -> str:
    """Render the ranked backlog with each need's proposal."""
    lines = ["== vocabulary proposal — the ranked backlog =="]
    if not rows:
        lines.append("   (no missing bank op detected)")
        return "\n".join(lines)
    for i, row in enumerate(rows, 1):
        need = row["need"]
        blocked = need["rules_blocked"]
        lines.append(
            f"   {i}. {need['op']}: unblocks {len(blocked)} rule(s), "
            f"agreement gain {need['agreement_gain']} "
            f"({need['before']}->{need['after']} of {need['samples']})"
        )
        for name in blocked:
            lines.append(f"       - {name}")
        lines.extend(_fmt_proposal(row["proposal"]))
    return "\n".join(lines)


def _fmt_examples(exs: list[dict[str, Any]]) -> str:
    """Render the worked examples."""
    lines = ["== worked examples =="]
    for ex in exs:
        if ex["kind"] == "shape-source":
            lines.append(
                f"   {ex['name']} (shape source): "
                f"arity-2 probe refused={ex['arity2_refused']}"
            )
        else:
            need = ex["need"]
            if need is None:
                lines.append(f"   {ex['name']}: no need (expressible)")
                continue
            lines.append(
                f"   {ex['name']}: gain {need['agreement_gain']} "
                f"unblocks {len(need['rules_blocked'])}"
            )
        if ex["proposal"] is not None:
            lines.extend(_fmt_proposal(ex["proposal"]))
    return "\n".join(lines)


def _dump_json(path: str, payload: dict) -> None:
    """Write the machine-readable result."""
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Print the ranked backlog, the proposals and the worked examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rules",
        choices=("carrier", "synthetic"),
        default="carrier",
        help="rule source: the carrier laws or a self-contained demo",
    )
    parser.add_argument(
        "--max-rules",
        type=int,
        default=None,
        help="cap on residual derives measured",
    )
    parser.add_argument(
        "--deadline",
        type=float,
        default=60.0,
        help="seconds before the scan returns a partial backlog",
    )
    parser.add_argument("--synth-limit", type=int, default=60)
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="skip the worked examples",
    )
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    if args.rules == "synthetic":
        rules = _synthetic_rules()
        measure: Any = _synthetic_measure
    else:
        rules = _carrier_rules()
        measure = None

    rows = backlog(
        rules,
        measure=measure,
        synth_limit=args.synth_limit,
        deadline=args.deadline,
        max_rules=args.max_rules,
    )
    print(_fmt_backlog(rows))  # stdout-compat
    exs = [] if args.no_examples else examples()
    if exs:
        print()  # stdout-compat
        print(_fmt_examples(exs))  # stdout-compat
    if args.json:
        _dump_json(args.json, {"backlog": rows, "examples": exs})
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

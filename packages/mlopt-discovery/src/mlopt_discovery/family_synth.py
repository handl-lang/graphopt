"""Family synthesis — the meta-grammar that mints handler FAMILIES.

``handler_synth`` mints bodies *inside* a family; the families
themselves (``ew``, ``gather``, ``contract``, …) are hand-authored
Python.  This module mints the *family descriptor* — plain data

    {gate, leaves, broadcast, requires, orders}

that parameterises the generic ``("family", desc, sub)`` body arm of
:func:`handler_synth.eval_body`.  A descriptor names the spec ops it
re-expresses (*gate*), the leaf vocabulary it re-seats them under
(*leaves* — each leaf a rule ``{kind, from, binds}`` resolving
through the ambient bindings table), the broadcast views its trees
may interleave (*broadcast*), the marker leaf a translated tree must
carry (*requires* — the no-tautology gate), and the read/map
composition orders its candidate bodies try (*orders* —
:data:`ORDERS`, the fused ``read_then_map`` first, the
denotation-equal ``map_then_read`` fallback).

THE SEARCH (everything is enumerated, nothing is derived):

* the spec set's *movable* ops — the op names outside
  ``ELEMENTWISE | BROADCAST_OPS`` the specs carry — are the family's
  claim surface;
* *gate* ranges over the non-empty subsets of that inventory,
  smallest first;
* each gated op is assigned a leaf form from :data:`LEAF_FORMS` —
  the candidate leaf vocabulary: ``at`` the index-read (re-seats the
  ambient ``select``, carrying literal ``dim``/``index`` attrs) and
  ``aff`` the per-position affine (re-seats ``linear``, metavar args
  only).  An op no leaf form re-seats closes the gate containing it;
* *broadcast* ranges over the subsets of
  ``handler_synth.BROADCAST_OPS``;
* *requires* ranges over the family's own leaf names, then ``"*"``
  (any leaf) — the marker that keeps a leaf-less translation from
  claiming a spec;
* *orders* is fixed to :data:`ORDERS` — the translation discipline
  every descriptor shares (the fused composition leads; the
  denotation-equal order is the fallback seed).  The orders are the
  candidate *translations* inside one family, exactly as
  ``_gather_seeds`` emits both.

THE CONTRACT (the same bar a body clears):

Each candidate family translates every spec (per order, marker
gated) into a ``("family", desc, sub)`` body and runs the fp64
verify — same draws, same ``allclose`` gate, same adversarial
re-draws as ``mint_handlers``.  A family that verifies ≥1 spec is
minted; one whose translation rejects every spec is ``no-claim``
(the marker or vocabulary refused); one that translates but never
verifies is ``refused`` — mint-a-label is refused, the same honesty
as bodies.  The verified *order* is recorded per spec:
``read_then_map`` is the strong claim (the fused kernel's own
composition — a genuinely different evaluation order the verifier
can and does reject); a spec carried only by ``map_then_read`` is
reported as the weaker, rename-equivalent claim.

Measured on the grammar-v3 refusal class — the four
``select``/``linear``-bearing prim specs embedded in
``docs/reports/results_overnight_prim_v3.json`` — the enumeration
recovers a ``gather``-equivalent descriptor (gate ``{linear,
select}`` re-seated as ``{aff: linear, at: select}``, broadcast
views admitted, marker ``at``) covering all four specs fp64, plus a
narrower ``{select}``-gate sub-family covering the select-by-select
spec alone.  The measurement is reproduced in
``docs/reports/family_synth.md``.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations, product
from typing import Any

from mlopt_discovery import handler_synth as hs
from mlopt_discovery.genkernel import ELEMENTWISE, _spec_ops

__all__ = [
    "LEAF_FORMS",
    "ORDERS",
    "descriptor_name",
    "family_record",
    "mint_families",
]

#: Candidate-family cap — the bounded meta-search a refusal quotes
#: against ("no family verified within budget" is a measured
#: refusal, not a guess).  The descriptor space is small by
#: construction — gate subsets x leaf assignments x broadcast
#: subsets x markers over the specs' own op inventory.
_BUDGET = 256

#: The read/map composition orders a descriptor's seeds try — the
#: fused ``read_then_map`` (``aff(at(x), w, b)``: project only the
#: gathered slice, the fused kernel's own order, equal to
#: read-of-map exactly when the read axis is not the map's
#: contracted axis — the verifier measures that) then
#: ``map_then_read`` (``at(aff(x, w, b))``, the denotation-equal
#: fallback).  Fixed, not enumerated: the orders are the candidate
#: translations *inside* one family — ``_gather_seeds``' own
#: discipline — not a family-level axis.
ORDERS: tuple[str, ...] = ("read_then_map", "map_then_read")

#: The candidate leaf vocabulary — the leaf forms a family may
#: re-seat spec ops under.  ``at`` is the index-read: it re-seats
#: the ambient ``select`` (``from`` the spelled op, ``binds`` the
#: ambient one), carrying the read's literal ``dim``/``index``
#: attrs.  ``aff`` is the per-position affine: it re-seats
#: ``linear``, metavar args only (a map over composite operands is
#: the contract family's claim).  A leaf form outside this inventory
#: is a meta-grammar extension, not a candidate.
LEAF_FORMS: dict[str, dict] = {
    "at": {
        "kind": "read",
        "from": "select",
        "binds": "select",
        "attrs": ("dim", "index"),
    },
    "aff": {"kind": "map", "from": "linear", "binds": "linear"},
}


def _movable_ops(spec: Any) -> set:
    """Return the spec's claim surface — the ops a family must re-express.

    The op names outside ``ELEMENTWISE | BROADCAST_OPS`` — indexing,
    contraction, view ops that move data.  These are what a family
    must re-express to claim the spec; everything else is interior
    vocabulary every family shares.
    """
    return _spec_ops(spec) - ELEMENTWISE - set(hs.BROADCAST_OPS)


def descriptor_name(desc: dict) -> str:
    """Name the descriptor honestly — its own hash, not a coined label.

    Same discipline as the prim mint's ``synop_<hash>`` — the name
    is the canonical descriptor, so two enumerations of the same
    family get one name.
    """
    canon = json.dumps(
        {
            "gate": desc["gate"],
            "leaves": desc["leaves"],
            "broadcast": desc["broadcast"],
            "requires": desc["requires"],
            "orders": desc["orders"],
        },
        sort_keys=True,
        default=list,
    )
    return f"fam_{hashlib.sha256(canon.encode()).hexdigest()[:8]}"


def _gate_options(movable: list) -> Any:
    """Non-empty subsets of *movable*, smallest first."""
    for r in range(1, len(movable) + 1):
        yield from combinations(movable, r)


def _candidates(movable: list) -> Any:
    """Enumerate candidate family descriptors over *movable* ops.

    Gate subsets x leaf assignments x broadcast subsets x markers —
    the descriptor space, smallest/simplest first.  A gate op with no
    leaf form closes the whole gate (the family could never translate
    a spec carrying it, so the gate adds nothing over its subset).
    """
    for gate in _gate_options(movable):
        choices = [
            [(n, f) for n, f in LEAF_FORMS.items() if f["from"] == op]
            for op in gate
        ]
        if any(not c for c in choices):
            continue
        for combo in product(*choices):
            leaves = {n: dict(f) for n, f in combo}
            for r in range(len(hs.BROADCAST_OPS) + 1):
                for bc in combinations(sorted(hs.BROADCAST_OPS), r):
                    for req in (*leaves, "*"):
                        yield {
                            "gate": list(gate),
                            "leaves": leaves,
                            "broadcast": list(bc),
                            "requires": req,
                            "orders": list(ORDERS),
                        }


def _fused_composition(sub: Any, desc: dict) -> bool:
    """Whether *sub* carries a genuine map∘read reorder.

    The ``read_then_map`` translation composes a ``map`` leaf over a
    ``read`` leaf — ``aff(at(x), w, b)``, a different evaluation
    order than the spelled site, which the fp64 gate measures.  A
    body with no such composition (a read-only tree, or the
    ``map_then_read`` fallback) is rename-equivalent to the spelled
    spec — its verify is by construction, and the report flags it
    honestly as ``fused=False``.
    """
    if isinstance(sub, (tuple, list)) and sub:
        rule = desc["leaves"].get(sub[0])
        if rule is not None and rule["kind"] == "map":
            arg0 = sub[1] if len(sub) > 1 else None
            if (
                isinstance(arg0, (tuple, list))
                and arg0
                and desc["leaves"].get(arg0[0], {}).get("kind")
                == "read"
            ):
                return True
        return any(
            _fused_composition(e, desc)
            for e in sub[1:]
            if not isinstance(e, dict)
        )
    return False


def _family_verdicts(
    desc: dict, entries: list, bindings: dict, rtol: float, atol: float
) -> tuple:
    """Translate + verify *desc* against every spec entry.

    Returns ``(covered, spec_rows)``: the verified spec indices and
    one verdict row per spec — ``verified`` (with the order and
    measured error), ``refused`` (translated but failed the fp64
    gate, with the measured error), ``outside`` (the family's
    vocabulary or marker refused the spec), or ``unevaluable`` (the
    spelled spec itself could not be drawn).
    """
    covered: list[int] = []
    rows: list[dict] = []
    for i, e in enumerate(entries):
        if e["envs"] is None:
            rows.append(
                {
                    "index": i,
                    "verdict": "unevaluable",
                    "detail": e["error"],
                }
            )
            continue
        hit = None
        last: dict[str, Any] = {"best_err": float("inf"), "detail": ""}
        seen: list = []
        for order in desc["orders"]:
            sub = hs._family_translate(e["spec"], desc, order)
            if (
                sub is None
                or not hs._family_marked(sub, desc)
                or sub in seen
            ):
                continue
            seen.append(sub)
            body = ("family", desc, sub)
            res = hs._verify_envs(
                body, e["envs"], e["refs"], bindings, rtol, atol
            )
            if res.ok:
                hit = (order, body, res)
                break
            if res.max_err < last["best_err"]:
                last = {"best_err": res.max_err, "detail": res.detail}
        if hit is None:
            if not seen:
                rows.append({"index": i, "verdict": "outside"})
            else:
                rows.append(
                    {
                        "index": i,
                        "verdict": "refused",
                        "best_err": last["best_err"],
                        "detail": last["detail"],
                    }
                )
            continue
        order, body, res = hit
        covered.append(i)
        rows.append(
            {
                "index": i,
                "verdict": "verified",
                "order": order,
                # the fused map∘read composition is the strong claim
                # — a genuinely different evaluation order the
                # verifier can reject; a body without it is
                # rename-equivalent and verifies by construction.
                "fused": _fused_composition(body[2], desc),
                "body": body,
                "max_err": res.max_err,
            }
        )
    return covered, rows


def mint_families(
    items: Any,
    *,
    domain: Any = None,
    bindings: dict | None = None,
    seed: int = 0,
    budget: int = _BUDGET,
    rtol: float = hs._VERIFY_RTOL,
    atol: float = hs._VERIFY_ATOL,
    adversarial: int = 1,
) -> tuple:
    """Mint verified family descriptors for measured subterm specs.

    Each *item* is a spec or a ``{"spec": ..., "domain": ...}``
    record (*domain* fills records that don't carry one) — the same
    mint contract as :func:`handler_synth.mint_handlers`, one level
    up: the candidates are family descriptors, not bodies.  The
    spec set's movable ops bound the gate enumeration; every
    candidate translates each spec and faces the fp64 verify on the
    measured domain, so a minted family is one that *reproduces* a
    spec, not one that merely names it.

    Returns ``(families, report)``: the minted descriptors — each
    carrying a provenance record — plus the falsification data: one
    verdict row per candidate family, per-spec
    verified/refused/outside rows, and the first-coverer coverage
    map.
    """
    if bindings is None:
        import mlopt_torch.torch_bridge as tb

        bindings = tb._IR_TO_TORCH
    entries: list[dict] = []
    for item in items:
        spec, dom = hs._item_spec_domain(item, domain)
        dom = hs._domain_list(dom)
        entry: dict[str, Any] = {"spec": spec, "domain": dom}
        try:
            envs, refs = hs._domain_draws(
                spec, dom, seed, bindings, adversarial
            )
            entry["envs"], entry["refs"] = envs, refs
        except Exception as exc:
            # the spelled spec itself could not be evaluated — no
            # family can verify it; recorded, never silent.
            entry["envs"], entry["refs"] = None, None
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entries.append(entry)
    movable = sorted(
        set().union(*(_movable_ops(e["spec"]) for e in entries))
        if entries
        else set()
    )
    families: list[dict] = []
    rows: list[dict] = []
    coverage: dict[int, str] = {}
    tried = 0
    budget_hit = False
    for cand in _candidates(movable):
        if tried >= budget:
            budget_hit = True
            break
        tried += 1
        desc = {"name": descriptor_name(cand), **cand}
        covered, spec_rows = _family_verdicts(
            desc, entries, bindings, rtol, atol
        )
        if covered:
            verdict = "minted"
            for i in covered:
                coverage.setdefault(i, desc["name"])
            fam = dict(desc)
            fam["provenance"] = {
                "origin": "minted-family",
                "grammar": hs._GRAMMAR_VERSION,
                "covered": covered,
                "orders_verified": sorted(
                    {
                        r["order"]
                        for r in spec_rows
                        if r["verdict"] == "verified"
                    }
                ),
                "fused": [
                    r["index"]
                    for r in spec_rows
                    if r["verdict"] == "verified" and r["fused"]
                ],
                "max_err": max(
                    r["max_err"]
                    for r in spec_rows
                    if r["verdict"] == "verified"
                ),
                "seed": seed,
                "rtol": rtol,
                "atol": atol,
            }
            families.append(fam)
        else:
            verdict = (
                "refused"
                if any(r["verdict"] == "refused" for r in spec_rows)
                else "no-claim"
            )
        rows.append(
            {
                "name": desc["name"],
                "verdict": verdict,
                "gate": desc["gate"],
                "leaves": list(desc["leaves"]),
                "broadcast": desc["broadcast"],
                "requires": desc["requires"],
                "covered": covered,
                "specs": [
                    {k: v for k, v in r.items() if k != "body"}
                    for r in spec_rows
                ],
            }
        )
    report = {
        "searched": tried,
        "budget": budget,
        "budget_hit": budget_hit,
        "n_specs": len(entries),
        "movable": movable,
        "coverage": {str(i): n for i, n in coverage.items()},
        "families": rows,
    }
    return families, report


def family_record(fam: dict) -> dict:
    """Serialise a minted family descriptor as a provenance record.

    The record is the family: descriptor + verified coverage ride
    along as JSON-able data — an ``extra_families``-ready row, no
    live code.
    """
    return hs.body_to_data(
        {k: v for k, v in fam.items() if k != "provenance"}
    ) | {"provenance": hs.body_to_data(fam.get("provenance") or {})}

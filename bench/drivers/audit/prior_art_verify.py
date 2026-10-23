"""Certify the prior-art bank — sweep guarded instances, cert the law.

For each rule in ``mlopt_core.laws.priorart.PRIOR_ART_RULES`` this
enumerates synthesized bindings (``oracle._binding_envs``), applies
the rule's own ``check``/``derive`` with firing semantics, evaluates
the instantiated pair fp64 (``oracle.eval_instance``), and on the
first ``equal`` site runs the cert gate
(``verify_law(lhs, rhs, [rule])`` — the e-graph must merge the two
sides under the rule itself and replay a standalone certificate).

``CASES`` supplies curated bindings where the generic enumeration
can't reach the accepted corner — correlated metavar shapes the
independent-binding bank rarely produces (concat-compatible
operands, normalized-shape-matched layer_norm weights, the rotary
half-split bounds, ``one_hot``'s int-typed index).

Verdicts: ``certified`` (>=1 equal site, 0 unequal/rhs-err sites in
the window, replayable cert), ``refuted`` (an accepted site measured
unequal — the law is false on its guard region), ``rhs-err`` (an
accepted site minted an unlowerable RHS), ``vacuous`` (no accepted
site in the window), ``uncertified`` (equal but no replayable cert),
``err`` (every accepted site failed both sides).

Run::

    .venv/bin/python bench/drivers/audit/prior_art_verify.py
    .venv/bin/python bench/drivers/audit/prior_art_verify.py --names concat_single,ste_fold
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mlopt_core.ir import Const, TensorType, Var
from mlopt_core.laws.priorart import PRIOR_ART_RULES
from mlopt_core.meta import instantiate_pattern

from mlopt_discovery import REPO_ROOT, oracle
from mlopt_discovery.verifier import verify_law

__all__ = ["LawRow", "certify_all", "certify_rule", "main"]

DOC_PATH = REPO_ROOT / "docs" / "results_prior_art_bank.md"

#: Env budget per law — the accepted corner of every law in this
#: bank sits far below this (the enumeration is diagonal over the
#: (viewed binding x attr) product).
_MAX_ENVS = 20_000
#: Accepted sites to evaluate per law before calling it certified.
_MAX_SITES = 24


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


#: Curated binding dicts for the enum-unreachable corners.  Keys are
#: leaf metavars (bound to Var/Const) and ``$attr:NAME`` attr
#: metavars — the same dict shape ``oracle._binding_envs`` yields.
CASES: dict[str, list[dict]] = {
    # concat structure — the enum binds operands independently and
    # almost never produces off-axis-matching pairs.
    "concat_flat_l": [
        {"a": _v("a", 4, 2), "b": _v("b", 4, 2), "c": _v("c", 3, 2), "$attr:D": 0}
    ],
    "concat_flat_r": [
        {"a": _v("a", 4, 2), "b": _v("b", 4, 2), "c": _v("c", 3, 2), "$attr:D": 0}
    ],
    **{
        f"concat_pull_{op}": [
            {"a": _v("a", 4, 2), "b": _v("b", 3, 2), "$attr:D": 0}
        ]
        for op in (
            "neg", "abs", "relu", "sigmoid", "silu",
            "tanh", "gelu", "exp", "square",
        )
    },
    **{
        f"concat_pull_{op}": [
            {
                "a1": _v("a1", 4, 2), "a2": _v("a2", 3, 2),
                "b1": _v("b1", 4, 2), "b2": _v("b2", 3, 2),
                "$attr:D": 0,
            }
        ]
        for op in ("add", "mul", "sub")
    },
    "mm_concat_left": [
        {"a": _v("a", 4, 3), "b": _v("b", 2, 3), "W": _v("W", 3, 5), "$attr:CD": 0}
    ],
    "mm_concat_left_rev": [
        {"a": _v("a", 4, 3), "b": _v("b", 2, 3), "W": _v("W", 3, 5), "$attr:CD": 0}
    ],
    "mm_concat_right": [
        {"x": _v("x", 4, 3), "b": _v("b", 3, 5), "c": _v("c", 3, 2), "$attr:CD": -1}
    ],
    "mm_concat_right_rev": [
        {"x": _v("x", 4, 3), "b": _v("b", 3, 5), "c": _v("c", 3, 2), "$attr:CD": -1}
    ],
    "mm_splitk": [
        {"a": _v("a", 4, 2), "b": _v("b", 4, 3), "W": _v("W", 5, 6), "$attr:CD": -1}
    ],
    "mm_splitk_rev": [
        {"a": _v("a", 4, 2), "b": _v("b", 4, 3), "W": _v("W", 5, 6), "$attr:SZ": (2, 3)}
    ],
    "linear_concat_w": [
        {"x": _v("x", 4, 3), "A": _v("A", 5, 3), "B": _v("B", 2, 3), "$attr:CD": -1}
    ],
    "linear_concat_w_rev": [
        {"x": _v("x", 4, 3), "A": _v("A", 5, 3), "B": _v("B", 2, 3), "$attr:AD": 0}
    ],
    # no-op attr domains the enum doesn't reach.
    "expand_noop": [
        {"u": _v("u", 2, 3), "$attr:S": (2, 3)},
        {"u": _v("u", 2, 3), "$attr:S": (-1, 3)},
    ],
    "broadcast_to_noop": [{"u": _v("u", 2, 3), "$attr:S": (2, 3)}],
    "pad_noop": [{"u": _v("u", 2, 3), "$attr:P": (0, 0, 0, 0)}],
    "permute_noop": [{"u": _v("u", 2, 3, 4), "$attr:P": (0, 1, 2)}],
    "permute_compose": [
        {"u": _v("u", 2, 3, 4), "$attr:P": (1, 2, 0), "$attr:Q": (2, 0, 1)},
        {"u": _v("u", 2, 3, 4), "$attr:P": (2, 0, 1), "$attr:Q": (1, 0, 2)},
    ],
    "index_select_id": [
        {"u": _v("u", 2, 3), "$attr:D": 1, "$attr:I": (0, 1, 2)}
    ],
    "split_single": [
        {"u": _v("u", 2, 3), "$attr:SZ": (3,), "$attr:D": 1, "$attr:I": 0}
    ],
    # const-scalar folds — the scalar-Const bank lacks these values.
    "sigmoid_fold": [{"x": _v("x", 4), "ONE": Const(1)}],
    "sigmoid_fold_div": [{"x": _v("x", 4), "ONE": Const(1), "ONE2": Const(1)}],
    "tanh_fold": [{"x": _v("x", 4), "TWO": Const(2), "TWO2": Const(2), "ONE": Const(1)}],
    "hardsigmoid_fold": [{"x": _v("x", 4), "THREE": Const(3), "SIX": Const(6)}],
    # the rotate-half ↔ cat-pair rotary equivalence — the enum can't
    # reach the aligned half-split bounds.
    "rope_rh_to_cat": [
        {
            "x": _v("x", 2, 4), "ch": _v("ch", 2, 2), "sh": _v("sh", 2, 2),
            "$attr:CD": 1,
            "$attr:I1D": 1, "$attr:I1A": 0, "$attr:I1E": 2,
            "$attr:I2D": 1, "$attr:I2A": 2, "$attr:I2E": 4,
        }
    ],
    # the layer_norm affine weight must match normalized_shape.
    "layer_norm_expand_affine": [
        {
            "u": _v("u", 2, 4), "w": _v("w", 4), "b": _v("b", 4),
            "$attr:ND": (4,), "$attr:E": 1e-5,
        }
    ],
    # the norm denotations — W/B must be (C,) vectors over U's channel
    # axis; the independent-binding enum almost never correlates them.
    "instance_norm_expand": [
        {
            "U": _v("u", 2, 4, 3, 3), "W": _v("w", 4), "B": _v("b", 4),
            "$attr:EP": 1e-5,
        }
    ],
    "group_norm_expand": [
        {
            "U": _v("u", 2, 4, 3, 3), "W": _v("w", 4), "B": _v("b", 4),
            "$attr:G": 2, "$attr:EP": 1e-5,
        },
        # rank-3 site + a non-divisible group count (the derive veto:
        # 48 inner elements split by 5 groups leaves a remainder).
        {
            "U": _v("u", 1, 6, 8), "W": _v("w", 6), "B": _v("b", 6),
            "$attr:G": 3, "$attr:EP": 1e-5,
        },
        {
            "U": _v("u", 1, 6, 8), "W": _v("w", 6), "B": _v("b", 6),
            "$attr:G": 5, "$attr:EP": 1e-5,
        },
    ],
    # one_hot's index is int-typed — bind it to a Const.
    "onehot_matmul_embed": [
        {"i": Const(1), "W": _v("W", 4, 3), "$attr:N": 4}
    ],
    # conv merges — correlated channel dims.
    "conv_out_merge": [
        {
            "x": _v("x", 1, 4, 8, 8), "W1": _v("W1", 2, 4, 3, 3),
            "W2": _v("W2", 3, 4, 3, 3), "b1": _v("b1", 2), "b2": _v("b2", 3),
            "$attr:CD": 1,
        }
    ],
    "conv_out_merge_nobias": [
        {
            "x": _v("x", 1, 4, 8, 8), "W1": _v("W1", 2, 4, 3, 3),
            "W2": _v("W2", 3, 4, 3, 3), "$attr:CD": 1,
        }
    ],
    "conv_out_merge_sp": [
        {
            "x": _v("x", 1, 4, 8, 8), "W1": _v("W1", 2, 4, 3, 3),
            "W2": _v("W2", 3, 4, 3, 3), "b1": _v("b1", 2), "b2": _v("b2", 3),
            "$attr:CD": 1, "$attr:ST": 1, "$attr:PD": 0,
        }
    ],
    "conv_out_merge_sp_nobias": [
        {
            "x": _v("x", 1, 4, 8, 8), "W1": _v("W1", 2, 4, 3, 3),
            "W2": _v("W2", 3, 4, 3, 3),
            "$attr:CD": 1, "$attr:ST": 1, "$attr:PD": 0,
        }
    ],
    "conv_out_merge_spdg": [
        {
            "x": _v("x", 1, 4, 8, 8), "W1": _v("W1", 2, 4, 3, 3),
            "W2": _v("W2", 3, 4, 3, 3), "b1": _v("b1", 2), "b2": _v("b2", 3),
            "$attr:CD": 1, "$attr:ST": 1, "$attr:PD": 0,
            "$attr:DL": 1, "$attr:GR": 1,
        }
    ],
    "conv_out_merge_spdg_nobias": [
        {
            "x": _v("x", 1, 4, 8, 8), "W1": _v("W1", 2, 4, 3, 3),
            "W2": _v("W2", 3, 4, 3, 3),
            "$attr:CD": 1, "$attr:ST": 1, "$attr:PD": 0,
            "$attr:DL": 1, "$attr:GR": 1,
        }
    ],
}


@dataclass
class LawRow:
    name: str
    verdict: str = "vacuous"
    n_accept: int = 0
    n_equal: int = 0
    n_unequal: int = 0
    n_rhs_err: int = 0
    n_err: int = 0
    cert: bool = False
    replayable: bool = False
    instance: str = ""
    note: str = ""


def _accepted(rule: Any, bound: dict) -> dict | None:
    """Firing semantics: check then derive; return the full subst."""
    if rule.check is not None:
        try:
            if not rule.check(bound):
                return None
        except Exception:
            return None
    inst = dict(bound)
    if rule.derive is not None:
        try:
            extra = rule.derive(bound)
        except Exception:
            return None
        if extra is None:
            return None
        inst.update(extra)
    return inst


def _sweep(
    rule: Any, envs: Iterable[dict], row: LawRow, max_sites: int
) -> None:
    certified = row.n_equal > 0
    for bound in envs:
        if row.n_accept >= max_sites:
            break
        inst = _accepted(rule, bound)
        if inst is None:
            continue
        try:
            lhs_i = instantiate_pattern(rule.lhs, inst)
            rhs_i = instantiate_pattern(rule.rhs, inst)
        except Exception:
            continue
        row.n_accept += 1
        outcome, note = oracle.eval_instance(lhs_i, rhs_i)
        if outcome == "equal":
            row.n_equal += 1
            if not certified:
                res = verify_law(lhs_i, rhs_i, [rule])
                row.cert = bool(res.derivable)
                row.replayable = bool(res.replayable)
                row.instance = f"{lhs_i!r} ~ {rhs_i!r}"
                certified = True
        elif outcome == "unequal":
            row.n_unequal += 1
            row.note = row.note or f"counterexample: {lhs_i!r}"
        elif outcome == "rhs-err":
            row.n_rhs_err += 1
            row.note = row.note or note
        else:
            row.n_err += 1
            row.note = row.note or f"{outcome}: {note}"


def _verdict(row: LawRow) -> str:
    if row.n_unequal:
        return "refuted"
    if row.n_rhs_err:
        return "rhs-err"
    if row.n_accept == 0:
        return "vacuous"
    if row.n_equal == 0:
        return "err"
    if row.replayable:
        return "certified"
    return "uncertified"


def certify_rule(
    rule: Any, *, max_envs: int = _MAX_ENVS, max_sites: int = _MAX_SITES
) -> LawRow:
    """Curated cases first, then the bounded generic sweep."""
    row = LawRow(name=rule.name)
    for case in CASES.get(rule.name, ()):
        _sweep(rule, (case,), row, max_sites)
    _sweep(
        rule,
        itertools.islice(
            oracle._binding_envs(rule.lhs, rule.rhs), max_envs
        ),
        row,
        max_sites,
    )
    row.verdict = _verdict(row)
    return row


def certify_all(
    rules: Any = None,
    *,
    max_envs: int = _MAX_ENVS,
    max_sites: int = _MAX_SITES,
    progress: Any = print,
) -> list[LawRow]:
    rules = PRIOR_ART_RULES if rules is None else rules
    rows = [certify_rule(r, max_envs=max_envs, max_sites=max_sites) for r in rules]
    if progress:
        for row in rows:
            progress(
                f"  {row.name:<28} {row.verdict:<12} "
                f"acc={row.n_accept} eq={row.n_equal} "
                f"ne={row.n_unequal} rerr={row.n_rhs_err} "
                f"cert={row.cert}/{row.replayable}"
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="prior_art_verify")
    p.add_argument("--names", default="")
    p.add_argument("--max-envs", type=int, default=_MAX_ENVS)
    p.add_argument("--max-sites", type=int, default=_MAX_SITES)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)
    rules = PRIOR_ART_RULES
    if args.names:
        keep = set(args.names.split(","))
        rules = [r for r in rules if r.name in keep]
        missing = keep - {r.name for r in rules}
        if missing:
            print(f"unknown: {sorted(missing)}")
            return 2
    rows = []
    for r in rules:
        row = certify_rule(r, max_envs=args.max_envs, max_sites=args.max_sites)
        print(
            f"  {row.name:<28} {row.verdict:<12} "
            f"acc={row.n_accept} eq={row.n_equal} "
            f"ne={row.n_unequal} rerr={row.n_rhs_err} "
            f"err={row.n_err} cert={row.cert}/{row.replayable}",
            flush=True,
        )
        rows.append(row)
    ok = sum(v.verdict == "certified" for v in rows)
    print(f"{ok}/{len(rows)} certified")
    if args.json:
        args.json.write_text(
            json.dumps([vars(r) for r in rows], indent=2, default=str)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

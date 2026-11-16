"""Differential parity: ``engine-eqsat`` (Rust) vs ``mlopt_core.egraph``.

ADR-0178 WP2 prerequisite for the ``engine="eqsat"`` opt-in: prove the
Rust e-graph interpretation and the Python e-graph agree on the laws
they share *before* any caller is allowed to select it.

The boundary
------------
``eqsat-run`` (``engine-eqsat/src/bin/eqsat-run.rs``) reads a JSON
spec — kernel ``Term``s (the ``hnl-engine`` serde envelope,
``{"Op": …}``/``{"Const": …}``), ``LAW_FORMAT = 2`` law records in the
``pat`` authoring spelling (the ``laws/*.json`` corpus), saturation
options, a ``CostSpec`` record, and optional ``check_terms`` —
saturates, extracts the min-cost member, and prints a canonical
report.  This driver runs the *same* law + instance through
``mlopt_core.egraph.EGraph`` and compares.

What is compared, per law
-------------------------
* **Firing** — the law's name appears in each engine's merge-log
  provenance (``fired`` multiset intersection, reported not asserted:
  saturation order differs legitimately).
* **RHS membership** — the instantiated RHS term lands in the root
  e-class on both sides (``check_terms`` / the ``_node_to_class``
  probe).  This is the rewrite-fired contract.
* **Extraction** — both sides extract *a* member; the terms need not
  be syntactically equal (cost models and tie-breaks differ), so
  equivalence is **semantic**: input term, Python best, and Rust best
  are all evaluated on shared random inputs through a small numpy
  evaluator (:func:`_eval`).  ``equal`` / ``diverge`` /
  ``unevaluated`` are the honest verdicts.

Law correspondence
------------------
A law is *portable* when its Python ``Rewrite`` and the engine-eqsat
record are the same 2-cell: the driver translates the Python
``lhs``/``rhs`` into the ``pat`` record spelling and alpha-normal
compares (:func:`_correspondence`).  ``cond``/``dspec`` are compared
as data against ``cond_to_data``/``derive_to_data``.  Attrs follow
the bridge rules: an all-literal attrs dict is ``{"attrs": {"lit":
…}}``; a **single-key** dict whose value is a string metavar is
``{"attrs": {"bind": <name>}}`` — whole-``__attrs__``-payload bind,
equivalent on instances whose payload has exactly that key (the
engine-eqsat expressiveness gap is reported, not hidden).

Guard-aware instantiation
-------------------------
Metavariables named by ``const-cmp``/``const-num`` conditions are
bound to numeric ``Const`` leaves satisfying the predicate (``== v``
binds ``v``; ``const-num`` binds a fresh odd constant) — otherwise
the guarded law would correctly refuse to fire on *both* sides and
the case would measure nothing.

Usage::

    .venv/bin/python bench/drivers/audit/eqsat_differential.py
    .venv/bin/python bench/drivers/audit/eqsat_differential.py --law comm_add
    .venv/bin/python bench/drivers/audit/eqsat_differential.py --census-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from mlopt_core.egraph import EGraph
from mlopt_core.laws import ALL_RULES
from mlopt_core.laws.cond import cond_to_data, derive_to_data
from mlopt_core.laws.priorart import PRIOR_ART_RULES
from mlopt_core.laws.tensor import POINTWISE_FACTOR_RULES
from mlopt_core.term import (
    Const,
    Op,
    Param,
    TensorType,
    Var,
    attr_to_data,
)

_REPO = Path(__file__).resolve().parents[3]
_ENGINE_EQSAT = Path.home() / "GitHub" / "handl-lang" / "engine-eqsat"
_LAWS_DIR = _ENGINE_EQSAT / "laws"

#: The `eqsat-run` binary — env override first, then the target dirs a
#: `cargo build` lands in (repo-local vs the user-global shared
#: target-dir).  Debug is acceptable: per-law saturations are small.
_BIN_CANDIDATES = [
    _ENGINE_EQSAT / "target" / "release" / "eqsat-run",
    _ENGINE_EQSAT / "target" / "debug" / "eqsat-run",
    Path.home() / ".cargo" / "shared-target" / "release" / "eqsat-run",
    Path.home() / ".cargo" / "shared-target" / "debug" / "eqsat-run",
]


def _bin() -> Path | None:
    env = os.environ.get("MLOPT_EQSAT_RUN")
    if env:
        return Path(env)
    for cand in _BIN_CANDIDATES:
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------------
#  Law banks + engine-eqsat corpus
# ---------------------------------------------------------------------------


def _python_banks() -> dict[str, Any]:
    """name -> Rewrite across every Python bank that overlaps."""
    banks: dict[str, Any] = {}
    for rules in (ALL_RULES, PRIOR_ART_RULES, POINTWISE_FACTOR_RULES):
        for r in rules:
            banks.setdefault(r.name, r)
    return banks


def _engine_records() -> dict[str, dict]:
    """name -> LAW_FORMAT=2 record across ``engine-eqsat/laws/*.json``."""
    out: dict[str, dict] = {}
    for f in sorted(_LAWS_DIR.glob("*.json")):
        for rec in json.loads(f.read_text()):
            out[rec["name"]] = rec
    return out


#: The one engine-eqsat record that is a *demonstration*, not a law —
#: `reshape → flatten` is a non-equivalence kept for matcher coverage
#: (``builtin.rs`` says so in capitals).  Excluded from the parity
#: run; reported in the census.
_UNSOUND = {"reshape_flatten_demo"}


def _gap_category(name: str, rule: Any) -> str:
    """Primary gap bucket for a Python-bank law with no engine record.

    Precedence: the structural families first (a conv fold is a conv
    fold whether or not it is also guarded), then guarded/procedural,
    then guard-free.  ``procedural`` means the rule carries
    ``check``/``derive`` hooks — channels engine-eqsat refuses by
    name rather than approximating.
    """
    n = name.lower()
    if "conv" in n:
        return "conv-fold"
    if "sdpa" in n:
        return "sdpa"
    if n.startswith("prim:") or "synop" in n:
        return "synop-minted"
    if any(
        k in n for k in ("attn", "attention", "qkv", "layout", "mha")
    ):
        return "attention/layout"
    if rule.cond is not None or rule.check or rule.derive:
        return "guarded/procedural"
    return "guard-free"


# ---------------------------------------------------------------------------
#  Python term -> pat-record translation + alpha-normal correspondence
# ---------------------------------------------------------------------------


def _J(v: Any) -> str | None:
    return (
        None
        if v is None
        else json.dumps(v, sort_keys=True, default=repr)
    )


def _py_attrs_spec(attrs: dict) -> tuple | None:
    """Translate a Python ``Op.attrs`` dict to the pat attrs spec.

    ``None`` → no ``__attrs__`` key; all-literal → ``("lit", …)``;
    single-key str-metavar → ``("bind", name)`` (the whole-payload
    bind — equivalent only when the payload has exactly that key);
    anything else → ``("UNTRANSLATABLE", …)``.
    """
    if not attrs:
        return None
    d = {k: attr_to_data(v) for k, v in attrs.items()}
    if all(not isinstance(v, str) for v in d.values()):
        return ("lit", _J(d))
    if len(d) == 1:
        (v,) = d.values()
        if isinstance(v, str):
            return ("bind", v)
    return ("UNTRANSLATABLE", _J(d))


def _pat_of(t: Any, mv: dict) -> tuple | None:
    """Python term -> canonical pat-view tuple (``None`` = leaf kind
    the pat language lacks — ``Var``/``Param`` pattern leaves)."""
    if isinstance(t, str):
        return ("m", mv.setdefault(t, len(mv)))
    if isinstance(t, Const):
        return ("c", _J(t.value))
    if isinstance(t, (Var, Param)):
        return None
    if isinstance(t, Op):
        args = tuple(_pat_of(a, mv) for a in t.args)
        if any(a is None for a in args):
            return None
        return (
            "op",
            t.op,
            None,  # dim — the pat record's "dim": null
            args,
            "native",
            None,  # lit — Python terms carry no lit channel here
            _py_attrs_spec(t.attrs),
        )
    return ("leaf", repr(t))


def _eng_pat(p: dict, mv: dict) -> tuple:
    """Engine pat record -> the same canonical tuple."""
    if "mvar" in p:
        return ("m", mv.setdefault(p["mvar"], len(mv)))
    if "const" in p:
        return ("c", _J(p["const"]))
    dim = p.get("dim")
    dimn = None if dim is None else _eng_pat(dim, mv)
    attrs = p.get("attrs")
    if attrs is None:
        an = None
    elif "lit" in attrs:
        an = ("lit", _J(attrs["lit"]))
    elif "bind" in attrs:
        an = ("bind", attrs["bind"])
    else:
        an = ("?", _J(attrs))
    return (
        "op",
        p["op"],
        dimn,
        tuple(_eng_pat(a, mv) for a in p["args"]),
        p.get("source"),
        _J(p.get("lit")),
        an,
    )


def _realpha(pair: tuple) -> tuple:
    """Rename metavars to first-occurrence indices over the pair."""
    mv: dict = {}

    def go(t: Any) -> Any:
        if isinstance(t, tuple):
            if t and t[0] == "m":
                return ("m", mv.setdefault(t[1], len(mv)))
            return tuple(go(x) for x in t)
        if isinstance(t, list):
            return [go(x) for x in t]
        return t

    return (go(pair[0]), go(pair[1]))


def _correspondence(rule: Any, rec: dict | None) -> tuple[str, str]:
    """Classify one law pair: ``(verdict, detail)``.

    verdict ∈ ``match`` / ``pattern-diff`` / ``untranslatable`` /
    ``no-engine-record``.
    """
    mv: dict = {}
    pl = _pat_of(rule.lhs, mv)
    pr = _pat_of(rule.rhs, mv)
    if pl is None or pr is None or "UNTRANSLATABLE" in repr((pl, pr)):
        return (
            "untranslatable",
            "leaf kind or attrs the pat language lacks",
        )
    if rec is None:
        return "no-engine-record", "no laws/*.json record"
    mv2: dict = {}
    ne = (_eng_pat(rec["lhs"], mv2), _eng_pat(rec["rhs"], mv2))
    if _realpha((pl, pr)) != _realpha(ne):
        return "pattern-diff", "alpha-normal (lhs, rhs) differ"
    if _J(cond_to_data(rule.cond)) != _J(rec.get("cond")):
        return "pattern-diff", "cond data differs"
    if _J(derive_to_data(rule.dspec)) != _J(rec.get("dspec")):
        return "pattern-diff", "dspec data differs"
    return "match", ""


# ---------------------------------------------------------------------------
#  Kernel Term JSON <-> mlopt term (the eqsat-run wire)
# ---------------------------------------------------------------------------
#
#  Deliberate engine-wire exception (stepe_tail_plan §4/§7): this codec
#  is NOT ``mlopt_core.term_json``.  It must round-trip what the
#  *deployed* ``eqsat-run`` binary actually emits and accepts — the
#  transitional ``const-str`` leaf profile plus the legacy
#  ``__var__:``/``__var__:param:`` untyped-leaf spellings (the
#  harness's ``Var(name, "f64")`` shorthand carries no ``TensorType``,
#  so ``term_json``'s typed-leaf emit cannot represent it), and decode
#  with ``validate=False`` for law-extracted terms.  ``term_json``
#  decodes ``__mlopt_leaf__:`` but treats ``__var__:`` as an opaque
#  ``Const`` string — porting would silently lose the Var/Param
#  leaves.  When the release binary is rebuilt (plan Step 5,
#  ``leaf="arr"`` flip) and fixtures carry real ``TensorType``s, this
#  can be re-evaluated; until then it stays hand-rolled on purpose.

_VAR_PREFIX = "__var__:"
_PARAM_PREFIX = "__var__:param:"

#: The typed-leaf wire convention — identical to
#: ``mlopt_orchestrator.eqsat._LEAF_PREFIX``: a ``Const`` string
#: ``__mlopt_leaf__:{"k":…,"n":…,"s":…}`` carrying the leaf's
#: ``TensorType`` shape (``s: null`` = wholly unknown).  engine-eqsat's
#: ``shape::mlopt_leaf_info`` decodes it, which is what keeps the
#: shape-reading conds honest on differential cases — the b9fe227
#: phantom-scalar fix.  Leaves whose ``typ`` is not a ``TensorType``
#: (the harness's ``Var(name, "f64")`` shorthand) keep the legacy
#: ``__var__:`` spelling and read as scalars on the Rust side, exactly
#: as before.
_LEAF_PREFIX = "__mlopt_leaf__:"


def _leaf_wire(t: Any) -> str | None:
    """``__mlopt_leaf__`` payload for a typed ``Var``/``Param``, else None."""
    shape = getattr(getattr(t, "typ", None), "shape", None)
    if shape is None and not hasattr(getattr(t, "typ", None), "shape"):
        return None
    kind = "var" if isinstance(t, Var) else "param"
    s = None if shape is None else [_wire_attr(d) for d in shape]
    return _LEAF_PREFIX + json.dumps(
        {"k": kind, "n": t.name, "s": s}, separators=(",", ":")
    )


def _kterm(t: Any) -> dict:
    """mlopt term -> ``hnl_engine::Term`` serde JSON.

    ``Var``/``Param`` leaves ride as ``__mlopt_leaf__`` consts when
    they carry a ``TensorType`` (name *and* declared shape reach the
    Rust shape inference); untyped leaves keep the ``__var__:``
    string convention; attrs ride the ADR-0173 ``attrs`` field — the
    channel that lands as the bridge's ``__attrs__`` attr, which is
    what ``pat``'s ``attrs`` spec reads.
    """
    if isinstance(t, Op):
        node: dict[str, Any] = {
            "dim": {"Const": {"value": 1}},
            "name": t.op,
            "args": [_kterm(a) for a in t.args],
            "source": "Native",
        }
        if t.attrs:
            node["attrs"] = {
                k: _wire_attr(v) for k, v in t.attrs.items()
            }
        return {"Op": node}
    if isinstance(t, (Var, Param)):
        payload = _leaf_wire(t)
        if payload is not None:
            return {"Const": {"value": payload}}
        prefix = _VAR_PREFIX if isinstance(t, Var) else _PARAM_PREFIX
        return {"Const": {"value": prefix + t.name}}
    if isinstance(t, Const):
        return {"Const": {"value": _wire_attr(t.value)}}
    raise TypeError(f"not a wireable leaf: {t!r}")


def _wire_attr(v: Any) -> Any:
    """Python attr value -> plain JSON (tuples become lists)."""
    if isinstance(v, (tuple, list)):
        return [_wire_attr(x) for x in v]
    if isinstance(v, dict):
        return {k: _wire_attr(x) for k, x in v.items()}
    return v


def _pyterm(j: Any) -> Any:
    """``hnl_engine::Term`` serde JSON -> mlopt term (best-term decode)."""
    if "Op" in j:
        node = j["Op"]
        attrs = node.get("attrs")
        kw: dict[str, Any] = {}
        if isinstance(attrs, dict):
            kw = dict(attrs)
        lit = node.get("lit")
        if isinstance(lit, dict) and "__mlopt_attrs__" in lit:
            kw = dict(lit["__mlopt_attrs__"])
        return Op.make(
            node["name"],
            *[_pyterm(a) for a in node["args"]],
            validate=False,
            **kw,
        )
    if "Const" in j:
        v = j["Const"]["value"]
        if isinstance(v, str) and v.startswith(_LEAF_PREFIX):
            return _leaf_from_payload(v)
        if isinstance(v, str) and v.startswith(_VAR_PREFIX):
            name = v[len(_VAR_PREFIX) :]
            if name.startswith("param:"):
                from mlopt_core.term import Param, TensorType

                return Param(name[7:], TensorType((2, 2)))
            return Var(name, "f64")
        return Const(v)
    raise ValueError(f"undecodable term record: {j!r}")


def _leaf_from_payload(value: str) -> Any:
    """Decode a ``__mlopt_leaf__`` const back to ``Var``/``Param``.

    Mirrors ``mlopt_orchestrator.eqsat._leaf_from_payload``: ``s`` is
    the declared ``TensorType`` shape (``null`` = wholly unknown —
    *not* a scalar).
    """
    from mlopt_core.term import TensorType

    rec = json.loads(value[len(_LEAF_PREFIX) :])
    s = rec.get("s")
    shape = None if s is None else tuple(s)
    if rec.get("k") == "var":
        return Var(rec["n"], TensorType(shape))
    if rec.get("k") == "param":
        return Param(rec["n"], TensorType(shape))
    raise ValueError(f"bad eqsat leaf record {value!r}")


# ---------------------------------------------------------------------------
#  Guard-aware instantiation
# ---------------------------------------------------------------------------


def _const_reqs(cond: Any) -> dict[str, Any]:
    """mvar name -> required Const value, from ``const-cmp`` atoms.

    Only *positive* occurrences count: a ``const-cmp`` under ``not``
    is a negative constraint (``not (C == 0)`` wants C ≠ 0), which a
    fresh nonzero ``const-num`` binding already satisfies — pinning
    the compared value would make the guard unsatisfiable.
    """
    reqs: dict[str, Any] = {}

    def cmp_val(op: str, v: float) -> float:
        return {
            "==": v,
            "!=": v + 1,
            ">": v + 1,
            ">=": v,
            "<": v - 1,
            "<=": v,
        }.get(op, v)

    def walk(x: Any, neg: bool = False) -> None:
        if isinstance(x, list):
            if x and x[0] == "not" and len(x) == 2:
                walk(x[1], not neg)
                return
            if (
                len(x) == 4
                and x[0] == "const-cmp"
                and isinstance(x[1], str)
                and not neg
            ):
                reqs.setdefault(x[1], cmp_val(str(x[2]), x[3]))
            for i in x[1:] if x and x[0] in ("and", "or") else x:
                walk(i, neg)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v, neg)

    walk(cond)
    return reqs


def _const_num_names(cond: Any) -> set[str]:
    """mvar names a *positive* ``const-num`` atom requires to be
    numeric Consts (under ``not`` the name must stay non-const)."""
    out: set[str] = set()

    def walk(x: Any, neg: bool = False) -> None:
        if isinstance(x, list):
            if x and x[0] == "not" and len(x) == 2:
                walk(x[1], not neg)
                return
            if (
                len(x) == 2
                and x[0] == "const-num"
                and isinstance(x[1], str)
                and not neg
            ):
                out.add(x[1])
            for i in x[1:] if x and x[0] in ("and", "or") else x:
                walk(i, neg)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v, neg)

    walk(cond)
    return out


def _instantiate(
    pat: Any, rule: Any, bind: dict, attr_bind: dict
) -> Any:
    """Bind a pattern's metavars to fresh leaves/attrs -> concrete term.

    ``bind`` is pre-seeded with guard-required ``Const`` bindings;
    remaining term metavars get ``Var`` leaves; attr metavariables
    (string attr values) get a fresh concrete value shared by name.
    """
    if isinstance(pat, str):
        if pat not in bind:
            # A typed leaf, not the ``"f64"`` placeholder: shape-reading
            # conds call ``_shape_of`` -> ``typ.shape``, which a bare
            # string crashes on, and the ``__mlopt_leaf__`` wire payload
            # carries this same (2, 2) to the Rust shape inference —
            # both engines then judge the guard against identical
            # declared shapes.
            bind[pat] = Var(f"v_{pat}", TensorType((2, 2)))
        return bind[pat]
    if isinstance(pat, Op):
        attrs: dict[str, Any] = {}
        for k, v in pat.attrs.items():
            if isinstance(v, str):
                key = "$attr:" + v
                if key not in attr_bind:
                    attr_bind[key] = _fresh_attr(v)
                attrs[k] = attr_bind[key]
            else:
                attrs[k] = v
        return Op.make(
            pat.op,
            *[_instantiate(a, rule, bind, attr_bind) for a in pat.args],
            validate=False,
            **attrs,
        )
    return pat  # Const / other concrete leaves ride through


def _fresh_attr(name: str) -> Any:
    """A concrete stand-in for an attr metavariable.

    Names the corpus uses: ``S``-family shape payloads, ``SC``-family
    scales, ``NC``-family minted consts (those are dspec outputs, never
    instantiated).  Anything else gets a small int — matching only
    needs *a* concrete value; evaluation coverage is reported
    separately.
    """
    low = name.lower()
    if "scale" in low or low.endswith("sc"):
        return 0.5
    if (
        "shape" in low
        or low == "s"
        or (low.endswith("s") and "dim" not in low)
    ):
        return [2, 2]
    return 2


def _dspec_expr(expr: Any, bind: dict) -> Any:
    """Evaluate a ``dspec`` expr over the const bindings.

    The corpus vocabulary is ``const``/``neg``/``add``/``sub``/``mul``/
    ``recip``/``fdiv`` over ``("const", NAME)`` leaves — the minted
    ``$leaf:`` values the RHS patterns reference by bare name.
    """
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return expr
    if not (isinstance(expr, (list, tuple)) and expr):
        raise ValueError(f"bad dspec expr {expr!r}")
    op = expr[0]
    if op == "const":
        # ("const", NAME) — NAME is a bound-metavar reference, not an
        # expr; resolve it before the generic arg recursion.
        leaf = expr[1]
        name = leaf.split(":", 1)[-1] if isinstance(leaf, str) else leaf
        v = bind.get(name)
        if isinstance(v, Const):
            return v.value
        if isinstance(v, (int, float)):
            return v
        raise ValueError(
            f"dspec const {name!r} unbound or non-const: {v!r}"
        )
    args = [_dspec_expr(a, bind) for a in expr[1:]]
    if op == "neg":
        return -args[0]
    if op == "add":
        return args[0] + args[1]
    if op == "sub":
        return args[0] - args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "recip":
        return 1.0 / args[0]
    if op == "fdiv":
        return args[0] / args[1]
    raise ValueError(f"uncovered dspec op {op!r}")


def _dspec_values(rule: Any, bind: dict) -> dict[str, Any]:
    """Minted-leaf name -> value, from the rule's ``dspec`` pairs."""
    vals: dict[str, Any] = {}
    for entry in rule.dspec or ():
        name, expr = entry
        mvar = name.split(":", 1)[-1] if ":" in name else name
        vals[mvar] = _dspec_expr(expr, bind)
    return vals


def _expected_rhs(
    rule: Any,
    bind: dict,
    attr_bind: dict,
    dspec_vals: dict | None = None,
) -> Any:
    """Instantiate the law's RHS under the LHS bindings — the term the
    rewrite must land in the root e-class.  ``dspec_vals`` carries the
    minted ``$leaf:`` constants the derive hook would compute."""
    dspec_vals = dspec_vals or {}

    def go(pat: Any) -> Any:
        if isinstance(pat, str):
            if pat in bind:
                return bind[pat]
            if pat in dspec_vals:
                return Const(dspec_vals[pat])
            return Var(f"v_{pat}", TensorType((2, 2)))
        if isinstance(pat, Op):
            attrs: dict[str, Any] = {}
            for k, v in pat.attrs.items():
                if isinstance(v, str):
                    attrs[k] = attr_bind.get("$attr:" + v, v)
                else:
                    attrs[k] = v
            return Op.make(
                pat.op,
                *[go(a) for a in pat.args],
                validate=False,
                **attrs,
            )
        return pat

    return go(rule.rhs)


# ---------------------------------------------------------------------------
#  Cost profiles — identical data on both engines
# ---------------------------------------------------------------------------


def _cost_fn(spec: dict) -> Any:
    """Python-side additive cost matching the ``op_table`` record."""
    ops = spec.get("ops", {})
    default = spec.get("default", 1.0)
    leaf = spec.get("leaf", 0.0)

    def cost(t: Any) -> float:
        if isinstance(t, Op):
            return ops.get(t.op, default) + sum(cost(a) for a in t.args)
        return leaf

    return cost


_FUSED = (
    "sigmoid",
    "silu",
    "tanh",
    "softsign",
    "hardsigmoid",
    "rsqrt",
    "sqrt",
    "exp",
    "sdpa",
    "linear",
    "matmul",
    "softmax",
)

COST_PROFILES: dict[str, dict] = {
    # Every op costs 1 — extraction minimizes op count (folds win).
    "uniform": {
        "kind": "op_table",
        "default": 1.0,
        "leaf": 0.0,
        "dim_leaf": 0.0,
        "ops": {},
    },
    # Fused/primitive ops cheap — the folded member wins ties.
    "fold": {
        "kind": "op_table",
        "default": 1.0,
        "leaf": 0.0,
        "dim_leaf": 0.0,
        "ops": {op: 0.5 for op in _FUSED},
    },
    # Expansions cheap — the unfolded member wins ties (the honest
    # other direction; both extractions must still be equivalent).
    "expand": {
        "kind": "op_table",
        "default": 1.0,
        "leaf": 0.0,
        "dim_leaf": 0.0,
        "ops": {op: 1.5 for op in _FUSED},
    },
}


# ---------------------------------------------------------------------------
#  Semantic evaluator — numpy over scalars/arrays
# ---------------------------------------------------------------------------


def _eval(t: Any, env: dict[str, np.ndarray]) -> Any:
    """Evaluate an mlopt term on numpy values.

    Raises ``KeyError``/``ValueError`` on ops or leaf kinds outside
    the covered set — the caller reports ``unevaluated`` honestly.
    """
    if isinstance(t, Var):
        return env[t.name]
    if isinstance(t, Param):
        return env.get("param:" + t.name, np.ones((2, 2)))
    if isinstance(t, Const):
        v = t.value
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return np.asarray(float(v))
        if isinstance(v, str):
            return np.asarray(float(v)) if _is_num(v) else v
        return np.asarray(v, dtype=float)
    if not isinstance(t, Op):
        raise ValueError(f"uncovered leaf {t!r}")
    a = [_eval(x, env) for x in t.args]
    op = t.op
    at = t.attrs or {}

    if op == "add":
        return a[0] + a[1]
    if op == "mul":
        return a[0] * a[1]
    if op == "sub":
        return a[0] - a[1]
    if op == "neg":
        return -a[0]
    if op == "abs":
        return np.abs(a[0])
    if op == "div":
        return a[0] / a[1]
    if op == "reciprocal":
        return 1.0 / a[0]
    if op == "pow":
        return np.power(a[0], a[1] if len(a) > 1 else 2.0)
    if op == "sqrt":
        return np.sqrt(a[0])
    if op == "rsqrt":
        return 1.0 / np.sqrt(a[0])
    if op == "square":
        return a[0] * a[0]
    if op == "exp":
        return np.exp(a[0])
    if op == "sigmoid":
        return 1.0 / (1.0 + np.exp(-a[0]))
    if op == "tanh":
        return np.tanh(a[0])
    if op == "silu":
        return a[0] * (1.0 / (1.0 + np.exp(-a[0])))
    if op == "softsign":
        return a[0] / (1.0 + np.abs(a[0]))
    if op == "hardsigmoid":
        # min(max(x/6 + 1/2, 0), 1) — slope 1/6, not 1/5.
        return np.clip(a[0] / 6.0 + 0.5, 0.0, 1.0)
    if op == "relu6":
        return np.clip(a[0], 0.0, 6.0)
    if op == "linear":
        # F.linear(x[..., in], W[out, in]) = x @ Wᵀ — the weight's
        # last two axes swap (kit_torch.torch_bindings); an optional
        # third arg is the bias, broadcast-added like torch.
        out = a[0] @ np.swapaxes(a[1], -1, -2)
        if len(a) > 2:
            out = out + a[2]
        return out
    if op == "matmul":
        return a[0] @ a[1]
    if op == "unsqueeze":
        return np.expand_dims(
            a[0], int(at.get("dim", -1)) % (np.ndim(a[0]) + 1)
        )
    if op == "select":
        # torch.select drops the indexed axis (np.take does the same
        # for a scalar index).
        return np.take(
            a[0],
            int(at.get("index", 0)),
            axis=int(at.get("dim", 0)) % np.ndim(a[0]),
        )
    if op == "glu":
        # F.glu: first half ⊙ sigmoid(second half) along ``dim``.
        d = int(at.get("dim", -1)) % np.ndim(a[0])
        first, second = np.array_split(a[0], 2, axis=d)
        return first * (1.0 / (1.0 + np.exp(-second)))
    if op == "softmax":
        d = int(at.get("dim", -1)) % np.ndim(a[0])
        e = np.exp(a[0] - a[0].max(axis=d, keepdims=True))
        return e / e.sum(axis=d, keepdims=True)
    if op == "transpose":
        return np.swapaxes(a[0], at.get("dim0", 0), at.get("dim1", 1))
    if op == "reshape":
        return np.reshape(a[0], tuple(at["shape"]))
    if op == "chunk":
        chunks = int(at["chunks"])
        dim = int(at.get("dim", 0))
        idx = int(at.get("index", 0))
        parts = np.array_split(a[0], chunks, axis=dim)
        return parts[idx]
    if op == "concat":
        return np.concatenate(a, axis=int(at.get("dim", 0)))
    if op == "ones_like":
        return np.ones_like(a[0])
    if op == "sdpa":
        scale = float(at.get("scale", 1.0))
        q, k, v = a[0], a[1], a[2]
        scores = q @ np.swapaxes(k, -1, -2) * scale
        scores = scores - scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w = w / w.sum(axis=-1, keepdims=True)
        return w @ v
    raise ValueError(f"uncovered op {op!r}")


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _env_for(
    terms: Any, shapes: dict | None = None, seed: int = 0
) -> dict[str, np.ndarray]:
    """Random inputs for every Var/Param leaf across *terms*.

    ``terms`` may be a single term or a list of them (compound
    fixtures share one environment).  ``shapes`` overrides per-name
    extents (``{"x": (1, 2, 8)}``); the default is a (2, 2) matrix —
    consistent for every pointwise and 2-D matmul law in the
    portable set.
    """
    rng = np.random.default_rng(seed)
    env: dict[str, np.ndarray] = {}
    shapes = shapes or {}
    if not isinstance(terms, (list, tuple)):
        terms = [terms]

    def leaf_shape(t: Any, key: str) -> tuple:
        """Env extent for a leaf: explicit ``shapes`` override, then the
        leaf's declared ``TensorType`` shape (all-concrete dims only),
        then the (2, 2) default."""
        if key in shapes:
            return shapes[key]
        s = getattr(getattr(t, "typ", None), "shape", None)
        if isinstance(s, (tuple, list)) and all(
            isinstance(d, int) and not isinstance(d, bool) for d in s
        ):
            return tuple(s)
        return (2, 2)

    def walk(t: Any) -> None:
        if isinstance(t, Var):
            env.setdefault(
                t.name,
                rng.uniform(0.5, 1.5, size=leaf_shape(t, t.name)),
            )
        elif isinstance(t, Param):
            env.setdefault(
                "param:" + t.name,
                rng.uniform(0.5, 1.5, size=leaf_shape(t, t.name)),
            )
        elif isinstance(t, Op):
            for a_ in t.args:
                walk(a_)

    for t in terms:
        walk(t)
    return env


def _semverdict(terms: list[Any], shapes: dict | None = None) -> str:
    """``equal`` / ``diverge`` / ``unevaluated:<why>`` for a term list."""
    env = _env_for(terms, shapes=shapes)
    try:
        vals = [_eval(t, env) for t in terms]
    except Exception as e:
        return f"unevaluated:{type(e).__name__}:{e}"
    base = vals[0]
    for v in vals[1:]:
        try:
            if not np.allclose(
                np.asarray(v, dtype=float),
                np.asarray(base, dtype=float),
                atol=1e-9,
                equal_nan=True,
            ):
                return "diverge"
        except Exception:
            if v is not base and v != base:
                return "diverge"
    return "equal"


# ---------------------------------------------------------------------------
#  The per-law run
# ---------------------------------------------------------------------------


def _run_python(
    rule: Any, term: Any, expected: Any, cost: dict, opts: dict
) -> dict:
    """Saturate + extract on the Python e-graph; return the report row."""
    g = EGraph()
    root = g.add_term(term)
    stats = g.run(
        [rule],
        root,
        max_iterations=opts.get("max_iterations", 100),
        max_nodes=opts.get("max_nodes", 100_000),
        rule_budgets=opts.get("rule_budgets"),
    )
    cost_fn = _cost_fn(cost)
    best = g.extract_best(root, cost_fn)
    fired = sorted({e.rule for e in g.merge_log if e.rule is not None})
    eid = g.add_term(expected)  # idempotent when already present
    return {
        "fired": fired,
        "in_root": g.find(eid) == g.find(root),
        "best": best,
        "best_cost": cost_fn(best) if best is not None else None,
        "stats": {
            "iterations": stats["iterations"],
            "n_enodes": stats["n_enodes"],
            "n_classes": stats["n_classes"],
            "stop": stats.get("stop"),
        },
    }


def _run_rust(
    bin_path: Path,
    rule_rec: dict,
    term: Any,
    expected: Any,
    cost: dict,
    opts: dict,
    name: str,
) -> dict:
    """Drive ``eqsat-run`` on the same law + instance."""
    spec = {
        "name": name,
        "terms": [_kterm(term)],
        "laws": [rule_rec],
        "options": opts,
        "cost": cost,
        "check_terms": [_kterm(expected)],
    }
    out = subprocess.run(
        [str(bin_path)],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if out.returncode != 0:
        raise RuntimeError(f"eqsat-run failed: {out.stderr.strip()}")
    rep = json.loads(out.stdout)
    if not rep.get("ok"):
        raise RuntimeError(f"eqsat-run error: {rep.get('error')}")
    rep["best"] = _pyterm(rep["term"]) if rep.get("term") else None
    rep["best_cost"] = rep.get("cost")
    rep["in_root"] = bool(
        rep["checks"] and rep["checks"][0].get("in_root")
    )
    return rep


#: Per-law fixture overrides — attr-metavar values and leaf shapes a
#: law needs before its instance is evaluable.  ``qkv_fuse`` is the
#: only attr-metavar law in the portable set: ``S`` is the
#: heads x dim reshape payload, ``SC`` the sdpa scale, and the leaves
#: need attention-consistent extents.
_LAW_FIXTURES: dict[str, dict] = {
    "qkv_fuse": {
        "attrs": {"S": (1, 2, 2, 2), "SC": 0.5},
        "shapes": {
            "v_x": (1, 2, 8),
            "v_Q": (4, 8),
            "v_K": (4, 8),
            "v_V": (4, 8),
        },
    },
}


def run_case(
    name: str,
    rule: Any,
    rec: dict,
    bin_path: Path,
    cost_name: str,
    opts: dict,
) -> dict:
    """One law x one cost profile on both engines."""
    cost = COST_PROFILES[cost_name]
    cond_data = cond_to_data(rule.cond)
    reqs = _const_reqs(cond_data)
    cnames = _const_num_names(cond_data)
    fixture = _LAW_FIXTURES.get(name, {})

    bind: dict[str, Any] = {}
    for m, v in reqs.items():
        bind[m] = Const(v)
    # const-num names without a cmp pin get a fresh odd constant.
    for m in sorted(cnames - set(reqs)):
        bind[m] = Const(3 + 2 * len(bind))

    attr_bind: dict[str, Any] = {
        "$attr:" + k: v for k, v in fixture.get("attrs", {}).items()
    }
    term = _instantiate(rule.lhs, rule, bind, attr_bind)
    try:
        dspec_vals = _dspec_values(rule, bind)
    except Exception:
        dspec_vals = {}
    expected = _expected_rhs(rule, bind, attr_bind, dspec_vals)

    py = _run_python(rule, term, expected, cost, opts)
    rs = _run_rust(bin_path, rec, term, expected, cost, opts, name)

    verdict = (
        _semverdict(
            [term, py["best"], rs["best"]],
            shapes=fixture.get("shapes"),
        )
        if py["best"] is not None and rs["best"] is not None
        else "no-extract"
    )

    return {
        "law": name,
        "cost": cost_name,
        "py_fired": py["fired"],
        "rs_fired": rs["fired"],
        "py_in_root": py["in_root"],
        "rs_in_root": rs["in_root"],
        "py_cost": py["best_cost"],
        "rs_cost": rs["best_cost"],
        "py_stats": py["stats"],
        "rs_stats": rs["stats"],
        "verdict": verdict,
        "rs_error": None,
    }


def _run_python_multi(
    rules: list, terms: list, cost: dict, opts: dict
) -> dict:
    """Multi-law saturation — the compound-fixture variant of
    :func:`_run_python` (no single RHS to probe)."""
    g = EGraph()
    roots = [g.add_term(t) for t in terms]
    root = roots[-1]
    # One run per root over the shared graph — later runs are
    # incremental (the dirty frontier), matching engine-eqsat's
    # `saturate` loop.
    stats = None
    for r in roots:
        stats = g.run(
            rules,
            r,
            max_iterations=opts.get("max_iterations", 100),
            max_nodes=opts.get("max_nodes", 100_000),
            rule_budgets=opts.get("rule_budgets"),
        )
    assert stats is not None
    cost_fn = _cost_fn(cost)
    best = g.extract_best(root, cost_fn)
    fired = sorted({e.rule for e in g.merge_log if e.rule is not None})
    return {
        "fired": fired,
        "best": best,
        "best_cost": cost_fn(best) if best is not None else None,
        "stats": {
            "iterations": stats["iterations"],
            "n_enodes": stats["n_enodes"],
            "n_classes": stats["n_classes"],
            "stop": stats.get("stop"),
        },
    }


# ---------------------------------------------------------------------------
#  Compound fixtures — several laws saturating together
# ---------------------------------------------------------------------------


def _compound_fixtures(pybanks: dict, erecs: dict) -> list[dict]:
    """Multi-law saturation cases: interactions single-law runs miss.

    Each fixture names Python rules + engine records + input terms;
    the verdict is semantic equality of the extracted bests (there is
    no single RHS to probe).
    """
    x = Var("cx", "f64")
    y = Var("cy", "f64")
    z = Var("cz", "f64")
    a = Var("ca", "f64")
    b = Var("cb", "f64")
    w = Var("cw", "f64")
    C1, C0 = Const(3), Const(0)

    def rules(*names: str) -> tuple[list, list]:
        rs = [pybanks[n] for n in names]
        return rs, [erecs[n] for n in names]

    out = []

    pr, er = rules(
        "comm_add",
        "comm_mul",
        "assoc_add",
        "assoc_mul",
        "id_add",
        "id_mul",
        "double_neg",
        "sub_to_add",
    )
    out.append(
        {
            "name": "algebra-closure",
            "rules": pr,
            "records": er,
            "terms": [
                Op.make(
                    "add",
                    Op.make("sub", Op.make("mul", x, C1), C0),
                    Op.make("neg", Op.make("neg", y)),
                ),
                Op.make("add", z, Op.make("mul", x, y)),
            ],
            "options": {
                "max_iterations": 40,
                "max_nodes": 50_000,
                "rule_budgets": {
                    "comm_add": 400,
                    "comm_mul": 400,
                    "assoc_add": 400,
                    "assoc_mul": 400,
                },
            },
        }
    )

    pr, er = rules(
        "sigmoid_fold",
        "silu_fold",
        "silu_mul_form",
        "tanh_fold",
        "softsign_fold",
        "hardsigmoid_fold",
        "div_sqrt_to_rsqrt",
        "pow_to_rsqrt",
        "recip_sqrt_to_rsqrt",
        "mul_div_swap",
        "square_expand",
        "square_to_pow",
        "pow_to_square",
        "mul_square",
        "silu_expand",
        "sub_to_add",
        "double_neg",
        "id_add",
        "id_mul",
    )
    sx = Var("sx", "f64")
    out.append(
        {
            "name": "activation-closure",
            "rules": pr,
            "records": er,
            "terms": [
                Op.make(
                    "add",
                    Op.make(
                        "mul",
                        Op.make(
                            "div",
                            Op.make(
                                "reciprocal",
                                Op.make(
                                    "add",
                                    Const(1),
                                    Op.make("exp", Op.make("neg", sx)),
                                ),
                            ),
                            Op.make("sqrt", sx),
                        ),
                        Op.make("square", sx),
                    ),
                    Op.make("silu", sx),
                ),
            ],
            "options": {"max_iterations": 40, "max_nodes": 50_000},
        }
    )

    pr, er = rules(
        "assoc_matmul",
        "assoc_matmul_rev",
        "assoc_linear",
        "assoc_linear_rev",
        "right_distribute_matmul",
        "right_factor_matmul",
        "right_factor_linear",
        "distribute_mul_add",
        "factor_mul_add",
    )
    out.append(
        {
            "name": "linear-chain",
            "rules": pr,
            "records": er,
            "terms": [
                Op.make("matmul", a, Op.make("matmul", b, w)),
                Op.make(
                    "add",
                    Op.make("matmul", x, w),
                    Op.make("matmul", y, w),
                ),
                Op.make("linear", Op.make("linear", x, w), w),
                Op.make("mul", C1, Op.make("add", x, y)),
            ],
            "options": {"max_iterations": 40, "max_nodes": 50_000},
        }
    )
    return out


def run_compound(fixture: dict, bin_path: Path, cost_name: str) -> dict:
    """One compound fixture on both engines."""
    cost = COST_PROFILES[cost_name]
    opts = fixture["options"]
    py = _run_python_multi(
        fixture["rules"], fixture["terms"], cost, opts
    )

    spec = {
        "name": fixture["name"],
        "terms": [_kterm(t) for t in fixture["terms"]],
        "root": len(fixture["terms"]) - 1,
        "laws": fixture["records"],
        "options": opts,
        "cost": cost,
    }
    out = subprocess.run(
        [str(bin_path)],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if out.returncode != 0:
        raise RuntimeError(f"eqsat-run failed: {out.stderr.strip()}")
    rep = json.loads(out.stdout)
    if not rep.get("ok"):
        raise RuntimeError(f"eqsat-run error: {rep.get('error')}")
    rs_best = _pyterm(rep["term"]) if rep.get("term") else None

    # Semantic verdict: the extracted root term plus both engines'
    # best members — the other input roots saturate the shared graph
    # but are not extraction targets.
    verdict = "no-extract"
    if py["best"] is not None and rs_best is not None:
        verdict = _semverdict(
            [fixture["terms"][-1], py["best"], rs_best]
        )
    return {
        "law": fixture["name"],
        "compound": True,
        "n_laws": len(fixture["rules"]),
        "cost": cost_name,
        "py_fired": py["fired"],
        "rs_fired": rep.get("fired", []),
        "py_cost": py["best_cost"],
        "rs_cost": rep.get("cost"),
        "py_stats": py["stats"],
        "rs_stats": rep.get("stats"),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def census(pybanks: dict, erecs: dict) -> list[dict]:
    """The coverage table: every engine-eqsat record classified."""
    rows = []
    for name, rec in sorted(erecs.items()):
        rule = pybanks.get(name)
        if name in _UNSOUND:
            rows.append(
                {
                    "law": name,
                    "status": "unsound-demo",
                    "detail": "non-equivalence; matcher coverage only",
                }
            )
            continue
        if rule is None:
            rows.append(
                {
                    "law": name,
                    "status": "no-python-rule",
                    "detail": "engine-only law",
                }
            )
            continue
        verdict, detail = _correspondence(rule, rec)
        rows.append({"law": name, "status": verdict, "detail": detail})
    # Python-bank laws with no engine record — the gap table.
    engine_names = set(erecs) | _UNSOUND
    for name in sorted(set(pybanks) - engine_names):
        rows.append(
            {
                "law": name,
                "status": "python-only",
                "detail": "no engine-eqsat record",
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--law", help="run a single law by name")
    ap.add_argument(
        "--cost", default="uniform", choices=sorted(COST_PROFILES)
    )
    ap.add_argument(
        "--census-only",
        action="store_true",
        help="print the coverage table and exit",
    )
    ap.add_argument(
        "--compound",
        action="store_true",
        help="also run the multi-law compound fixtures",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", type=Path, help="write rows as JSONL")
    args = ap.parse_args()

    pybanks = _python_banks()
    erecs = _engine_records()
    rows = census(pybanks, erecs)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("== census ==")
    for k in sorted(counts):
        print(f"  {k:20s} {counts[k]}")
    portable = [
        r["law"]
        for r in rows
        if r["status"] == "match" and r["law"] in pybanks
    ]
    print(f"  {'portable':20s} {len(portable)}")

    # The gap table: python-only laws bucketed by what engine-eqsat
    # would need to express them (ADR-0178 WP2 coverage census).
    gap: dict[str, int] = {}
    for r in rows:
        if r["status"] != "python-only":
            continue
        rule = pybanks[r["law"]]
        cat = _gap_category(r["law"], rule)
        gap[cat] = gap.get(cat, 0) + 1
    print("== python-only gap by category ==")
    for k in sorted(gap):
        print(f"  {k:20s} {gap[k]}")
    n_proc = sum(
        1
        for r in rows
        if r["status"] == "python-only"
        and (pybanks[r["law"]].check or pybanks[r["law"]].derive)
    )
    print(f"  {'(of which check/derive-hooked)':20s} {n_proc}")
    if args.census_only:
        for r in rows:
            if r["status"] not in ("match",):
                print(
                    f"  {r['status']:20s} {r['law']:32s} {r['detail']}"
                )
        return 0

    bin_path = _bin()
    if bin_path is None:
        print(
            "eqsat-run not found — build it: "
            "cargo build --bin eqsat-run (engine-eqsat)",
            file=sys.stderr,
        )
        return 2
    print(f"== eqsat-run: {bin_path} ==")

    out_rows = []
    if args.compound:
        for fx in _compound_fixtures(pybanks, erecs):
            try:
                row = run_compound(fx, bin_path, args.cost)
            except Exception as e:
                row = {
                    "law": fx["name"],
                    "verdict": "error",
                    "rs_error": str(e),
                }
            out_rows.append(row)
            flag = "" if row.get("verdict") == "equal" else "  <<<"
            print(
                f"{fx['name']:32s} {args.cost:7s} "
                f"n_laws={row.get('n_laws', '?')} "
                f"py_fired={len(row.get('py_fired') or [])} "
                f"rs_fired={len(row.get('rs_fired') or [])} "
                f"verdict={row.get('verdict')}{flag}"
            )
            if row.get("rs_error"):
                print(f"    error: {row['rs_error']}")

    names = [args.law] if args.law else portable
    if args.limit:
        names = names[: args.limit]
    opts = {"max_iterations": 60, "max_nodes": 50_000}

    for n in names:
        rule = pybanks.get(n)
        rec = erecs.get(n)
        if rule is None or rec is None:
            print(f"{n:32s} SKIP no counterpart")
            continue
        try:
            row = run_case(n, rule, rec, bin_path, args.cost, opts)
        except Exception as e:
            row = {
                "law": n,
                "cost": args.cost,
                "verdict": "error",
                "rs_error": str(e),
            }
        out_rows.append(row)
        flag = "" if row.get("verdict") == "equal" else "  <<<"
        print(
            f"{n:32s} {args.cost:7s} "
            f"py_root={row.get('py_in_root')} rs_root={row.get('rs_in_root')} "
            f"py_fired={len(row.get('py_fired') or [])} "
            f"rs_fired={len(row.get('rs_fired') or [])} "
            f"verdict={row.get('verdict')}{flag}"
        )
        if row.get("rs_error"):
            print(f"    error: {row['rs_error']}")

    if args.json:
        with args.json.open("w") as f:
            for r in out_rows:
                f.write(json.dumps(r, default=str) + "\n")

    bad = [r for r in out_rows if r.get("verdict") not in ("equal",)]
    print(f"\n{len(out_rows)} cases, {len(bad)} non-equal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

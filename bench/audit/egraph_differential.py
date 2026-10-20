"""Differential saturation check: mlopt's ``EGraph`` vs handl-egraph's.

The sibling of ``handl_differential.py``, one level up: that probe
diffs single ``EGraph.apply_rule`` firings through handl's rewrite
engine; this one diffs **whole runs** — ``EGraph.run`` +
``extract_best`` against ``handl-egraph``'s ``EGraph::run`` /
``extract_best`` — over shared fixtures.

The boundary
------------
``egraph-run`` (``packages/mlopt-kit/src/bin/egraph-run.rs``) reads a
JSON spec — concrete terms, declarative rules, run options, a cost
table — in the wire forms ``handl-discovery``'s codec shares with
``mlopt_core.ir.term_to_data`` (``{"mvar"}/{"leaf"}/{"op"}``), runs the
Rust port, and prints a canonical report.  The Python side runs
``mlopt_core.egraph.EGraph`` on the same fixture and renders the same
report; the two reports are compared field by field.

What "the same e-graph" means structurally: internal class ids differ
between engines, so the report never leaks one — classes are identified
by a Weisfeiler-Lehman label (a fixpoint refinement over member enodes
rendered with their children's *labels*), members are rendered as
``[op, attrs, [child labels]]`` with a tagged canonical attr encoding
(spelling-strict: ``2``/``2.0``/``"2"`` distinct, floats keyed on bits —
the Rust ``AttrValue`` identity contract).  Both sides serialize
identically, so the whole ``classes`` section is compared as data.

What is asserted
----------------
``stop``, ``iterations``, ``n_enodes``, ``n_classes``,
``n_proof_edges``, the ``proof_rules``/``app_rules`` provenance
multisets, ``rule_budgets``/``budget_suspended``, the full
``classes``/``signatures`` structure, and the extracted ``best`` term
with its cost.  ``rep`` (the ``any_term`` representative) is *reported*
but not asserted: an equal-shape tie between same-op members resolves
through each engine's (documented, different) member-order key.

``best`` ties get an honest verdict: if the winners differ, the driver
re-runs the Rust side with ``check_terms`` carrying the Python winner —
``in_root`` + equal cost means a member-order tie (both optimal, both
present), which reports ``tie``; anything else reports ``diverge``.

Scope, stated honestly
----------------------
* **Declarative rules only.**  ``check``/``derive`` hooks are Python
  closures — inexpressible on the wire — so fixtures restrict to rules
  with neither.  That still covers 26 shipped laws (identities,
  comm/assoc, the silu/softsign folds, the matmul/linear algebra) plus
  the attr-metavar patterns (``shape=S``-style) those rules carry.
* **No param-only subtrees.**  ``extract_best``'s param-only discount
  is *not* ported on the Rust side (a named divergence — it's domain
  policy, not substrate); fixtures keep every op-subtree
  ``Var``-reachable so the discount can never bite.
* **Literal string attrs are metavar-only.**  Python patterns read any
  ``str`` attr as an attribute metavariable, so a spec pattern cannot
  express a literal string attr — the one asymmetry the wire exposes
  (the port's ``Lit``/``Meta`` split is strictly more honest).

Usage::

    .venv/bin/python bench/audit/egraph_differential.py            # report
    .venv/bin/python bench/audit/egraph_differential.py --limit 5  # a slice
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

from mlopt_core.egraph import EGraph
from mlopt_core.egraph.types import ENode, _member_key
from mlopt_core.ir import Const, Op, Var
from mlopt_core.laws import ALL_RULES

_REPO = Path(__file__).resolve().parents[2]

#: The `egraph-run` binary — env override first, then the two target
#: dirs a `cargo build` lands in on this host (the repo's `.cargo`
#: layout vs the user-global `target-dir` in ~/.cargo/config.toml).
_BIN_CANDIDATES = [
    _REPO / "target" / "debug" / "egraph-run",
    Path.home() / ".cargo" / "shared-target" / "debug" / "egraph-run",
]


def _bin() -> Path | None:
    """Locate the compiled ``egraph-run``; ``None`` when unbuilt."""
    # ``CATOPT_EGRAPH_RUN`` is the pre-rename spelling, kept as a
    # fallback for existing setups.
    env = os.environ.get("MLOPT_EGRAPH_RUN") or os.environ.get(
        "CATOPT_EGRAPH_RUN"
    )
    if env:
        return Path(env)
    for cand in _BIN_CANDIDATES:
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------------
#  Term <-> spec encoding (the wire forms handl-discovery's codec reads)
# ---------------------------------------------------------------------------


def _spec_attr(v: Any, pattern: bool) -> Any:
    """Encode an attr value; in patterns a ``str`` is an attr metavar."""
    if isinstance(v, str) and pattern:
        return {"amvar": v}
    if isinstance(v, (tuple, list)):
        return {"__tuple__": [_spec_attr(x, pattern) for x in v]}
    return v


def _spec_term(t: Any, pattern: bool) -> dict:
    """Encode a term (concrete or pattern) in the wire form."""
    if isinstance(t, Op):
        return {
            "op": t.op,
            "args": [_spec_term(a, pattern) for a in t.args],
            "attrs": {
                k: _spec_attr(v, pattern) for k, v in t.attrs.items()
            },
        }
    if isinstance(t, str):
        return {"mvar": t}
    return {"leaf": repr(t)}


def _rep_attr(v: Any) -> Any:
    """Attr values in a *reported* term — the codec's loose JSON form."""
    if isinstance(v, (tuple, list)):
        return [_rep_attr(x) for x in v]
    if isinstance(v, dict):
        return {k: _rep_attr(x) for k, x in v.items()}
    return v


def _rep_data(t: Any) -> dict | None:
    """Render a Python term exactly as ``term_to_data`` does on Rust."""
    if t is None:
        return None
    if isinstance(t, Op):
        return {
            "op": t.op,
            "args": [_rep_data(a) for a in t.args],
            "attrs": {k: _rep_attr(v) for k, v in t.attrs.items()},
        }
    if isinstance(t, str):
        return {"mvar": t}
    return {"leaf": repr(t)}


def _leaf_for(key: str) -> Any:
    """Return a Python leaf object whose ``repr`` is *key*.

    Used for membership probes of Rust-reported terms.
    """
    try:
        return Const(int(key))
    except ValueError:
        pass
    try:
        return Const(float(key))
    except ValueError:
        pass
    return Var(key, "f64")


def _decode_term(d: Any) -> Any:
    """Spec term-data → a Python term (membership probes only)."""
    if "mvar" in d:
        return d["mvar"]
    if "leaf" in d:
        return _leaf_for(str(d["leaf"]))
    return Op.make(
        d["op"],
        *[_decode_term(a) for a in d["args"]],
        **{k: _decode_attr(v) for k, v in d["attrs"].items()},
        validate=False,
    )


def _decode_attr(v: Any) -> Any:
    if isinstance(v, dict) and "amvar" in v:
        return v["amvar"]
    if isinstance(v, dict) and "__tuple__" in v:
        return tuple(_decode_attr(x) for x in v["__tuple__"])
    if isinstance(v, list):
        return [_decode_attr(x) for x in v]
    return v


# ---------------------------------------------------------------------------
#  The canonical report — mirrored field-for-field in
#  packages/mlopt-kit/src/egraph.rs (run_spec).  Keep them in lockstep.
# ---------------------------------------------------------------------------


def _canon_attr(v: Any) -> list:
    """Attr values in the tagged canonical form — spelling-strict.

    Floats key on IEEE-754 bits (``0.0 != -0.0``, NaN payloads
    distinct) — the Rust ``AttrValue`` identity contract.
    """
    if v is None:
        return ["z"]
    if isinstance(v, bool):  # before int — Python bool is int
        return ["b", v]
    if isinstance(v, int):
        return ["i", v]
    if isinstance(v, float):
        return ["f", struct.unpack(">Q", struct.pack(">d", v))[0]]
    if isinstance(v, str):
        return ["s", v]
    if isinstance(v, (tuple, list)):
        return ["a", [_canon_attr(x) for x in v]]
    if isinstance(v, dict):
        return ["o", [[k, _canon_attr(v[k])] for k in sorted(v)]]
    raise TypeError(f"uncanonical attr value: {v!r}")


def _dump(v: Any) -> str:
    """Serialize *v* in the compact, deterministic form both sides share."""
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False)


def _label_hash(marshal: str) -> str:
    """``sha256`` hex of a marshal — the fixed-size label content.

    Labels embed inside each other at every refinement round; digesting
    keeps each round's payload flat (matches the Rust side's
    ``Sha256`` byte-for-byte).
    """
    return hashlib.sha256(marshal.encode()).hexdigest()


def _member_sig(
    graph: EGraph, node: ENode, labels: dict[int, str]
) -> str:
    """``[op, attrs, [child labels]]`` — the engine-independent member."""
    attrs = [[k, _canon_attr(v)] for k, v in node.attrs]
    kids = [labels[graph.find(c)] for c in node.children]
    return _dump([node.op, attrs, kids])


def _wl_labels(
    graph: EGraph, members: dict[int, list[ENode]]
) -> dict[int, str]:
    """Compute Weisfeiler-Lehman class labels.

    The same fixed-round refinement as the Rust side (``n_classes + 1``
    rounds, each round digesting the sorted member marshal —
    deterministic whether or not the refinement has converged).
    """
    labels = {c: "" for c in members}
    for _ in range(len(members) + 1):
        nxt = {
            c: _label_hash(
                _dump(
                    sorted(
                        _member_sig(graph, n, labels)
                        for n in members[c]
                    )
                )
            )
            for c in members
        }
        if nxt == labels:
            break
        labels = nxt
    return labels


def _cone_ids(graph: EGraph, roots: list[int]) -> list[int]:
    """Enumerate every canonical class id in the cone of *roots*.

    Member-enode children are walked downward — the same traversal the
    Rust side performs (``egraph.rs::cone_ids``): every live class is
    in the cone — an instantiated RHS merges into the matched class,
    nothing mints unreachable classes.
    """
    seen: set[int] = set()
    stack = [graph.find(r) for r in roots]
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        for n in graph._classes[c].nodes:
            stack.extend(graph.find(ch) for ch in n.children)
    return sorted(seen)


def _report(
    name: str,
    graph: EGraph,
    roots: list[int],
    root: int,
    stats: dict,
    cost: Any,
) -> dict:
    """Render the canonical report for a saturated Python EGraph."""
    canon = _cone_ids(graph, roots)
    members = {
        c: sorted(graph._classes[c].nodes, key=_member_key)
        for c in canon
    }
    labels = _wl_labels(graph, members)
    classes = []
    for c in canon:
        ms = sorted(_member_sig(graph, n, labels) for n in members[c])
        classes.append(
            {
                "sig": labels[c],
                "rep": _rep_data(graph.any_term(c)),
                "members": ms,
            }
        )
    classes.sort(key=lambda e: e["sig"])
    signatures = sorted(labels.values())
    root_sig = labels[graph.find(root)]

    best = graph.extract_best(root, cost)
    proof_rules = sorted(
        e.rule for e in graph.merge_log if e.rule is not None
    )
    app_rules = sorted(a["rule"] for a in graph.applications)
    return {
        "name": name,
        "ok": True,
        "stop": stats["stop"],
        "iterations": stats["iterations"],
        "n_enodes": stats["n_enodes"],
        "n_classes": stats["n_classes"],
        "n_proof_edges": stats["n_proof_edges"],
        "truncation_level": stats["truncation_level"],
        "rule_budgets": stats["rule_budgets"],
        "budget_suspended": sorted(stats["budget_suspended"]),
        "improved": stats.get("improved"),
        "proof_rules": proof_rules,
        "app_rules": app_rules,
        "root_sig": root_sig,
        "signatures": signatures,
        "classes": classes,
        "best": _rep_data(best),
        "best_cost": cost(best) if best is not None else None,
        "checks": [],  # populated by the caller when probing terms
    }


def _cost_fn(costs: dict | None) -> Any:
    """Per-op additive cost — same contract as the Rust closure."""
    c = costs or {}
    ops = c.get("ops", {})
    default = c.get("default", 1.0)
    leaf = c.get("leaf", 0.0)

    def cost(t: Any) -> float:
        if isinstance(t, Op):
            return ops.get(t.op, default) + sum(cost(a) for a in t.args)
        return leaf

    return cost


def run_python(fx: dict) -> tuple[dict, EGraph, int]:
    """Run one fixture through mlopt's EGraph; return (report, g, root)."""
    g = EGraph()
    roots = [g.add_term(t) for t in fx["terms"]]
    root = roots[fx.get("root", len(roots) - 1)]
    opts = fx.get("options", {})
    cost = _cost_fn(fx.get("costs"))
    stats = g.run(
        fx["rules"],
        root,
        max_iterations=opts.get("max_iterations", 100),
        max_nodes=opts.get("max_nodes", 100_000),
        rule_budgets=opts.get("rule_budgets"),
        stop=opts.get("stop", "fixed_point"),
        patience=opts.get("patience", 3),
        cost_fn=cost,
    )
    return _report(fx["name"], g, roots, root, stats, cost), g, root


def spec_for(fx: dict, check_terms: list | None = None) -> dict:
    """Encode *fx* as the JSON spec the Rust side consumes."""
    spec = {
        "name": fx["name"],
        "terms": [_spec_term(t, pattern=False) for t in fx["terms"]],
        "root": fx.get("root", len(fx["terms"]) - 1),
        "rules": [
            {
                "name": r.name,
                "lhs": _spec_term(r.lhs, pattern=True),
                "rhs": _spec_term(r.rhs, pattern=True),
            }
            for r in fx["rules"]
        ],
        "options": fx.get("options", {}),
        "costs": fx.get("costs", {}),
    }
    if check_terms:
        spec["check_terms"] = check_terms
    return spec


def run_rust(bin_path: Path, spec: dict) -> dict:
    """Drive ``egraph-run`` on *spec*; return the parsed report."""
    out = subprocess.run(
        [str(bin_path)],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if out.returncode != 0:
        raise RuntimeError(f"egraph-run failed: {out.stderr.strip()}")
    return json.loads(out.stdout)


# ---------------------------------------------------------------------------
#  Fixtures — shared terms + rules both engines run
# ---------------------------------------------------------------------------


def _declarative_rules() -> list:
    """Collect the shipped laws carrying no ``check``/``derive`` hook.

    That subset is what the wire can express (a guard is a Python
    closure).
    """
    return [
        r for r in ALL_RULES if r.check is None and r.derive is None
    ]


def _instantiate(pat: Any) -> Any:
    """Bind a pattern's metavars to fresh leaves/attrs.

    The result is a concrete instance of the lhs the law can fire on.
    Attr metavars get distinct ints by name; term metavars get ``Var``
    leaves.  Structural instantiation only — ``validate=False`` like
    the sibling probe's bare-``f64`` vars.
    """
    attr_vals: dict[str, int] = {}

    def go(t: Any) -> Any:
        if isinstance(t, str):
            return Var(f"v_{t}", "f64")
        if isinstance(t, Op):
            attrs = {}
            for k, v in t.attrs.items():
                if isinstance(v, str):
                    attrs[k] = attr_vals.setdefault(v, len(attr_vals))
                else:
                    attrs[k] = v
            return Op.make(
                t.op,
                *[go(a) for a in t.args],
                validate=False,
                **attrs,
            )
        return t  # Const/int/etc. leaves ride through

    return go(pat)


def _by_name(rules: list) -> dict[str, Any]:
    return {r.name: r for r in rules}


def fixtures() -> list[dict]:
    """Build the shared fixture set.

    * ``law:<name>`` — each declarative shipped law, alone, on a term
      synthesized from its own lhs.  Per-law coverage of the whole
      declarative bank.
    * compound closures — several laws over a term exercising
      interactions (identities folding into a comm/assoc closure;
      the matmul/linear factoring chain).
    """
    dec = _declarative_rules()
    named = _by_name(dec)
    fx: list[dict] = []
    for r in dec:
        fx.append(
            {
                "name": f"law:{r.name}",
                "terms": [_instantiate(r.lhs)],
                "rules": [r],
                "options": {"max_iterations": 30, "max_nodes": 10_000},
                "costs": {"default": 1.0, "leaf": 0.0},
            }
        )

    # The algebra closure — identities + involution + sub + comm/assoc
    # over a compound term; comm/assoc budgeted to also exercise the
    # bounded-saturation contract.
    x, y, z = (Var(s, "f64") for s in "xyz")
    algebra = [
        named[n]
        for n in (
            "comm_add",
            "comm_mul",
            "assoc_add",
            "assoc_mul",
            "id_add",
            "id_mul",
            "double_neg",
            "sub_to_add",
        )
        if n in named
    ]
    fx.append(
        {
            "name": "algebra-closure",
            "terms": [
                Op.make(
                    "add",
                    Op.make(
                        "sub",
                        x,
                        Op.make("mul", y, Const(1)),
                    ),
                    Const(0),
                ),
                Op.make("neg", Op.make("neg", z)),
            ],
            "root": 0,
            "rules": algebra,
            "options": {
                "max_iterations": 40,
                "max_nodes": 20_000,
                "rule_budgets": {
                    "comm_add": 400,
                    "comm_mul": 400,
                    "assoc_add": 400,
                    "assoc_mul": 400,
                },
            },
            "costs": {
                "ops": {"add": 1.0, "mul": 1.0, "neg": 1.0, "sub": 2.0},
                "default": 1.0,
                "leaf": 0.0,
            },
        }
    )

    # The matmul/linear factoring chain — the real optimization surface:
    # associativity both directions plus distribute/factor.
    a, b, w, w2 = (Var(s, "f64") for s in ("a", "b", "w", "w2"))
    mm = [
        named[n]
        for n in (
            "assoc_matmul",
            "assoc_matmul_rev",
            "assoc_linear",
            "assoc_linear_rev",
            "right_distribute_matmul",
            "right_factor_matmul",
            "right_factor_linear",
        )
        if n in named
    ]
    fx.append(
        {
            "name": "linear-chain",
            "terms": [
                Op.make(
                    "add",
                    Op.make("matmul", a, w),
                    Op.make("matmul", b, w),
                ),
                Op.make(
                    "linear",
                    Op.make("linear", a, w),
                    w2,
                ),
            ],
            "root": 0,
            "rules": mm,
            "options": {"max_iterations": 40, "max_nodes": 20_000},
            "costs": {
                "ops": {"matmul": 3.0, "linear": 3.0, "add": 1.0},
                "default": 1.0,
                "leaf": 0.0,
            },
        }
    )

    # The full declarative bank over one compound term — every law the
    # wire can express, saturating together.
    compound = Op.make(
        "add",
        Op.make("silu", x),
        Op.make("mul", Op.make("pow", y, 2), Const(1)),
    )
    fx.append(
        {
            "name": "full-declarative",
            "terms": [compound],
            "root": 0,
            "rules": dec,
            "options": {
                "max_iterations": 40,
                "max_nodes": 20_000,
                "rule_budgets": {
                    "comm_add": 300,
                    "comm_mul": 300,
                    "assoc_add": 300,
                    "assoc_mul": 300,
                },
            },
            "costs": {
                "ops": {
                    "silu": 1.0,
                    "mul": 2.0,
                    "sigmoid": 2.0,
                    "add": 1.0,
                    "pow": 3.0,
                    "square": 1.0,
                    "neg": 1.0,
                    "sub": 2.0,
                    "matmul": 3.0,
                    "linear": 3.0,
                },
                "default": 1.0,
                "leaf": 0.0,
            },
        }
    )
    return fx


# ---------------------------------------------------------------------------
#  The comparison
# ---------------------------------------------------------------------------

#: Report fields compared for exact equality — the structural verdict.
_ASSERTED = (
    "stop",
    "iterations",
    "n_enodes",
    "n_classes",
    "n_proof_edges",
    "truncation_level",
    "rule_budgets",
    "budget_suspended",
    "improved",
    "proof_rules",
    "app_rules",
    "root_sig",
    "signatures",
)


def _strip_reps(classes: list[dict]) -> list[dict]:
    """Return *classes* without the ``rep`` field.

    ``any_term``'s first-resolving member is member-order-dependent
    across engines, so ``rep`` is reported but never asserted.
    """
    return [{"sig": c["sig"], "members": c["members"]} for c in classes]


def _rep_diffs(py: dict, rs: dict) -> list[str]:
    """List where the ``any_term`` representatives differ (informational)."""
    if len(py["classes"]) != len(rs["classes"]):
        return []
    diffs = []
    for pc, rc in zip(py["classes"], rs["classes"], strict=True):
        if pc["rep"] != rc["rep"]:
            diffs.append(pc["sig"][:60])
    return diffs


def _membership(py_term_data: dict, rs: dict) -> bool:
    """Return whether the Rust side interned *py_term_data* in root."""
    for chk in rs.get("checks", []):
        if chk["term"] == py_term_data:
            return bool(chk["in_root"])
    return False


def compare(fx: dict, bin_path: Path) -> tuple[str, dict, dict, str]:
    """Run both engines on *fx*.

    Returns ``(verdict, py_report, rs_report, detail)``; verdicts are
    ``agree`` / ``tie`` / ``diverge`` / ``error``.
    """
    py_rep, g, root = run_python(fx)
    rs_rep = run_rust(bin_path, spec_for(fx))

    diffs: list[str] = []
    for f in _ASSERTED:
        if py_rep[f] != rs_rep[f]:
            diffs.append(
                f"{f}: python={json.dumps(py_rep[f])[:200]} "
                f"rust={json.dumps(rs_rep[f])[:200]}"
            )
    if _strip_reps(py_rep["classes"]) != _strip_reps(rs_rep["classes"]):
        diffs.append("classes: member structures differ")
    rep_diffs = _rep_diffs(py_rep, rs_rep)

    # The extracted winner: equal, or an honest tie (both members of
    # the same class at equal cost — the member-order caveat).
    best_ok = py_rep["best"] == rs_rep["best"]
    tie = False
    if not best_ok:
        py_best_spec = py_rep["best"]
        rs2 = run_rust(
            bin_path,
            spec_for(
                fx, check_terms=[py_best_spec] if py_best_spec else []
            ),
        )
        in_root = (
            _membership(py_best_spec, rs2) if py_best_spec else False
        )
        rs_best_py = (
            _decode_term(rs_rep["best"]) if rs_rep["best"] else None
        )
        rs_in_py = False
        if rs_best_py is not None:
            eid, _en = g._locate(rs_best_py)
            rs_in_py = eid is not None and g.find(eid) == g.find(root)
        same_cost = py_rep["best_cost"] == rs_rep["best_cost"]
        if in_root and rs_in_py and same_cost:
            tie = True
        else:
            diffs.append(
                f"best: python={json.dumps(py_rep['best'])[:160]} "
                f"rust={json.dumps(rs_rep['best'])[:160]} "
                f"(py_in_rust_root={in_root} rs_in_py_root={rs_in_py} "
                f"costs {py_rep['best_cost']} vs {rs_rep['best_cost']})"
            )

    verdict = "diverge" if diffs else ("tie" if tie else "agree")
    detail = "; ".join(diffs)
    if rep_diffs:
        detail += (" | " if detail else "") + (
            f"rep diffs (member-order only): {len(rep_diffs)}"
        )
    return verdict, py_rep, rs_rep, detail


def main(argv: list[str] | None = None) -> int:
    """Run the differential over the fixtures and print a report."""
    args = list(sys.argv[1:] if argv is None else argv)
    limit = None
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])

    bin_path = _bin()
    if bin_path is None or not Path(bin_path).exists():
        print(
            "egraph-run not available — build it:\n"
            "  cargo build -p mlopt-kit --bin egraph-run",
            file=sys.stderr,
        )
        return 2

    fxs = fixtures()
    if limit is not None:
        fxs = fxs[:limit]

    outcomes = {"agree": 0, "tie": 0, "diverge": 0, "error": 0}
    divergent: list[str] = []
    for fx in fxs:
        try:
            verdict, _py, _rs, detail = compare(fx, bin_path)
        except Exception as e:  # one side raised — a finding too
            verdict, detail = "error", f"{type(e).__name__}: {e}"
        outcomes[verdict] += 1
        if verdict in ("diverge", "error"):
            divergent.append(f"{fx['name']}: {detail}")
        print(
            f"  {verdict:8s}  {fx['name']}"
            + (f"   ({detail})" if verdict != "agree" else "")
        )

    print(
        "differential saturation check — mlopt EGraph vs handl-egraph"
    )
    print(f"  fixtures        : {len(fxs)}")
    print(f"  agree           : {outcomes['agree']}")
    print(f"  tie (member-ord): {outcomes['tie']}")
    print(f"  DIVERGE         : {outcomes['diverge']}")
    print(f"  errors          : {outcomes['error']}")
    if divergent:
        for d in divergent[:10]:
            print(f"    - {d}")
    return 1 if (outcomes["diverge"] or outcomes["error"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())

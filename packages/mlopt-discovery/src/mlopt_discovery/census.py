"""Shape census — which term shapes real models actually contain.

The proposal retro (``project/retros/law-proposal.md``) found laws by
enumerating an *algebraic* grammar; the impact retro
(``project/retros/law-impact.md``) then measured them on real graphs
and got **zero firings** — the ops were present, the *shapes* were
not.  The lesson: law proposal optimised algebraic novelty, not
applicability.  Before proposing anything shape-aware, measure the
distribution it must hit.

This tool mines the real corpus — every ``mlopt_torch.models`` block
exported to IR, plus every ``bench`` law case — and reports the
**shape frequencies** of the subterms actually present:

* **op-tuple census** — for each op node, ``(op, child-op tuple)``;
  how often it occurs and in how many distinct terms.
* **shape census** — each distinct subterm abstracted to a shape
  (leaves become numbered placeholders, so a *repeated* leaf stays
  repeated); how often the shape occurs and in how many terms.
* **sharing census** — for the common binary shapes, whether the two
  operands share a subterm (the precondition a factoring/absorption
  law needs), and how many distinct leaves each shape touches.

The census is the deliverable that outlives the experiment: it is
what a shape-aware proposer must read, and it is the honest statement
of what the corpus contains.

Torch seam (plan 0034): this module imports **without** torch — the
census machinery (:func:`shape_key`, the three census passes, the
renderers) is pure ``mlopt_core.ir`` term walking.  The corpus
builders it reuses live in :mod:`mlopt_discovery.impact`, which binds
torch lazily too, so :func:`corpus` / :func:`run_census` / ``main``
raise ``ImportError`` honestly at call time on a torch-free domain;
the census functions themselves accept any ``CorpusTerm`` list.

Run::

    .venv/bin/python -m mlopt_discovery.census
    .venv/bin/python -m mlopt_discovery.census --json /tmp/census.json
    .venv/bin/python -m mlopt_discovery.census --top 40

CPU-only, bounded (seconds).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlopt_core.ir import Const, Op

# ``law_impact`` owns the corpus builders (bench + real models); reuse
# them so the census reads exactly the graphs the impact tool probes.
from mlopt_discovery.impact import _bench_cases, model_cases

__all__ = [
    "CorpusTerm",
    "corpus",
    "main",
    "op_tuple_census",
    "shape_census",
    "sharing_census",
]

#: Default number of rows in each ranked table.
_DEFAULT_TOP = 30


@dataclass(frozen=True)
class CorpusTerm:
    """One exported graph, tagged with where it came from."""

    source: str  # "bench" | "model"
    name: str
    term: Any


# ---------------------------------------------------------------------------
#  Corpus
# ---------------------------------------------------------------------------


def corpus() -> list[CorpusTerm]:
    """Return every real term: bench law cases + exported models.

    Reuses ``law_impact``'s builders so the census and the impact
    probe read byte-identical graphs.  A builder that raised is
    already reported by ``law_impact``; here it is simply absent.
    When the cache census (``mlopt_discovery.intake_dir()``) exists (written by
    ``mlopt_discovery.intake``), the ingested workloads union in under
    the ``"intake"`` source — the side-file census.
    """
    bench, _be = _bench_cases()
    models, _me = model_cases()
    out: list[CorpusTerm] = []
    for c in bench:
        out.append(CorpusTerm("bench", c.name, c.term))
    for c in models:
        out.append(CorpusTerm("model", c.name, c.term))
    # Lazy sibling import: law_intake already imports this module's
    # census functions, so a top-level import would be circular.
    from mlopt_discovery.intake import load_cases

    for c in load_cases():
        out.append(CorpusTerm("intake", c.name, c.term))
    return out


def _iter_subterms(term: Any) -> list[Any]:
    """Return every distinct subterm of *term* (deduped by identity)."""
    seen: set[int] = set()
    out: list[Any] = []
    stack = [term]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        out.append(t)
        if isinstance(t, Op):
            stack.extend(t.args)
    return out


def _postorder(term: Any) -> list[Any]:
    """Every distinct node in *term*'s DAG, children before parents.

    ``id``-deduped — terms are hash-consed, so structural sharing is
    object sharing.  Explicit stack: deep terms would otherwise hit
    the recursion limit.
    """
    expanded: set[int] = set()
    order: list[Any] = []
    stack: list[tuple[Any, bool]] = [(term, False)]
    while stack:
        t, done = stack.pop()
        if done:
            order.append(t)
            continue
        if id(t) in expanded:
            continue
        expanded.add(id(t))
        stack.append((t, True))
        if isinstance(t, Op):
            stack.extend((a, False) for a in t.args)
    return order


def _expansion(term: Any, count_leaves: bool, cap: int | None) -> int:
    """Tree-expansion size of *term*, memoised per distinct node.

    ``E(t) = self + sum(E(a))`` over ``Op`` args — the recurrence a
    tree render (``shape_key``/``_spec_of``/the site collectors)
    implicitly computes, so sharing multiplies exactly like the
    render would, in O(distinct nodes).  ``count_leaves`` decides
    whether non-``Op`` leaves add 1 (``term_expansion``) or 0
    (``op_expansion``).  *cap* early-exits a node's partial sum.
    """
    leaf = 1 if count_leaves else 0
    memo: dict[int, int] = {}
    for t in _postorder(term):
        if not isinstance(t, Op):
            memo[id(t)] = leaf
            continue
        total = 1
        for a in t.args:
            total += memo[id(a)]
            if cap is not None and total > cap:
                break
        memo[id(t)] = total
    return memo[id(term)]


def term_expansion(term: Any, cap: int | None = None) -> int:
    """Return the *tree-expansion* size of *term*, DAG-aware.

    This is the ``ProbeBounds.spec_tree_cap`` / ``key_cap`` /
    ``shape_key_cap`` metric (see ``mlopt_discovery.bounds``): every
    ``Op`` and leaf occurrence counts 1 — ops + metavar/const leaves,
    matching what a spec/pattern render spells (the site collectors
    add one per attrs dict on top).

    ``shape_key``/``_spec_of``/the site collectors all render a term
    as a tree: a subterm referenced by two parents is spelled once
    per use, so a sharing-heavy DAG (a residual stream — every block
    output feeds the next block AND the skip) expands exponentially
    even though its distinct-node count is small.  This counts that
    expansion exactly — every ``Op`` and leaf occurrence counts 1 —
    in O(distinct nodes).  *cap*, when given, early-exits at
    ``cap + 1`` — callers comparing against a budget never need the
    exact astronomic value.
    """
    return _expansion(term, True, cap)


def op_expansion(term: Any, cap: int | None = None) -> int:
    """Return the ``Op``-only tree-expansion size of *term*.

    This is the ``ProbeBounds.spec_op_cap`` metric (see
    ``mlopt_discovery.bounds``): same recurrence as
    :func:`term_expansion` but non-``Op`` leaves count 0 — the count
    equals the number of op nodes a spec rendering
    (``meta_arena._spec_of``/``play._fusion_pattern``) spells, which
    is the size a claim/spec bound should compare against.
    """
    return _expansion(term, False, cap)


def _op_of(term: Any) -> str:
    """Return the op name of *term*, or a leaf/const marker."""
    if isinstance(term, Op):
        return term.op
    if isinstance(term, Const):
        return "const"
    return "·"


# ---------------------------------------------------------------------------
#  Shape abstraction
# ---------------------------------------------------------------------------


def _attr_key(attrs: dict) -> tuple:
    """Return a hashable, order-independent key for a node's attrs."""
    out = []
    for k, v in attrs.items():
        try:
            hash(v)
            out.append((k, v))
        except TypeError:
            out.append((k, repr(v)))
    return tuple(sorted(out))


def shape_key(term: Any, memo: dict | None = None) -> Any:
    """Abstract *term* to a leaf-numbered shape key.

    ``Op`` nodes keep op + attrs + the shape of each child; a ``Const``
    keeps its literal; every other leaf (``Var`` / ``Param``) becomes a
    numbered placeholder.  The numbering is by *object identity*, so a
    leaf that appears twice in one term stays the *same* number — the
    census sees the sharing a factoring law needs.

    Iterative post-order with ``id``-dedup on ``Op`` nodes: a shared
    subterm's key is occurrence-invariant (its leaves were numbered
    at first encounter and never renumber), so computing it once and
    reusing the tuple yields the identical key the tree render
    produced — while a deep model term can no longer exhaust the
    recursion limit (``model_corpus`` admits ~20k-deep mutants).
    """
    memo = {} if memo is None else memo
    key_of: dict[int, Any] = {}
    # (node, expanded) post-order; children pushed reversed so args
    # evaluate left-to-right — the leaf-numbering order the recursive
    # render produced.
    stack: list[tuple[Any, bool]] = [(term, False)]
    while stack:
        t, expanded = stack.pop()
        if expanded:
            key_of[id(t)] = (
                t.op,
                _attr_key(t.attrs),
                tuple(key_of[id(a)] for a in t.args),
            )
            continue
        if isinstance(t, Op):
            if id(t) in key_of:
                continue
            stack.append((t, True))
            stack.extend((a, False) for a in reversed(t.args))
        elif isinstance(t, Const):
            key_of[id(t)] = ("const", t.value)
        else:
            key_of[id(t)] = ("leaf", memo.setdefault(id(t), len(memo)))
    return key_of[id(term)]


def shape_repr(key: Any) -> str:
    """Render a shape key as ``op(child, child)`` notation.

    Iterative post-order: shape keys mirror term depth, so the
    recursive render shared the model term's recursion-depth
    exposure.  ``id``-dedup is safe — the render is a pure function
    of the key, and ``shape_key`` returns the same tuple object for
    a shared subterm.
    """
    if not isinstance(key, tuple):
        return repr(key)
    repr_of: dict[int, str] = {}
    stack: list[tuple[Any, bool]] = [(key, False)]
    while stack:
        k, expanded = stack.pop()
        if not isinstance(k, tuple):
            repr_of[id(k)] = repr(k)
            continue
        head = k[0]
        if head == "leaf":
            repr_of[id(k)] = _letter(k[1])
            continue
        if head == "const":
            repr_of[id(k)] = repr(k[1])
            continue
        if id(k) in repr_of:
            continue
        if expanded:
            _op, attrs, kids = k
            inner = ", ".join(repr_of[id(c)] for c in kids)
            if attrs:
                a = ",".join(f"{kk}={v}" for kk, v in attrs)
                repr_of[id(k)] = f"{_op}[{a}]({inner})"
            else:
                repr_of[id(k)] = f"{_op}({inner})"
            continue
        stack.append((k, True))
        stack.extend((c, False) for c in reversed(k[2]))
    return repr_of[id(key)]


def _letter(i: int) -> str:
    """Return ``a, b, …`` for placeholder index *i*."""
    return chr(ord("a") + i) if i < 26 else f"v{i}"


def _leaf_ids(key: Any, out: list | None = None) -> list:
    """Return the placeholder indices of every leaf in a shape key.

    Iterative — key depth mirrors term depth.  Callers only test
    set-equality of the ids, so visit order is irrelevant.
    """
    out = [] if out is None else out
    stack = [key]
    while stack:
        k = stack.pop()
        if not isinstance(k, tuple):
            continue
        if k[0] == "leaf":
            out.append(k[1])
        elif k[0] not in ("const",) and len(k) == 3:
            stack.extend(k[2])
    return out


def _has_shared_leaf(key: Any) -> bool:
    """Return whether a placeholder repeats — the shape shares a subterm."""
    ids = _leaf_ids(key)
    return len(ids) != len(set(ids))


# ---------------------------------------------------------------------------
#  Census 1 — op-tuple frequencies
# ---------------------------------------------------------------------------


def op_tuple_census(
    terms: list[CorpusTerm],
) -> tuple[Counter, dict]:
    """Count ``(op, child-op tuple)`` nodes and the terms they span."""
    counts: Counter = Counter()
    terms_of: dict = {}
    for ct in terms:
        for s in _iter_subterms(ct.term):
            if not isinstance(s, Op):
                continue
            key = (s.op, tuple(_op_of(a) for a in s.args))
            counts[key] += 1
            terms_of.setdefault(key, set()).add((ct.source, ct.name))
    return counts, terms_of


# ---------------------------------------------------------------------------
#  Census 2 — shape frequencies (leaf-abstracted)
# ---------------------------------------------------------------------------


def shape_census(
    terms: list[CorpusTerm],
) -> tuple[Counter, dict]:
    """Count distinct subterm shapes and the terms they span."""
    counts: Counter = Counter()
    terms_of: dict = {}
    for ct in terms:
        for s in _iter_subterms(ct.term):
            if not isinstance(s, Op):
                continue
            key = shape_key(s, {})
            counts[key] += 1
            terms_of.setdefault(key, set()).add((ct.source, ct.name))
    return counts, terms_of


# ---------------------------------------------------------------------------
#  Census 3 — sharing structure of binary shapes
# ---------------------------------------------------------------------------


def sharing_census(terms: list[CorpusTerm]) -> dict:
    """Report the sharing structure of the corpus's binary op nodes.

    For every node whose op is a two-argument algebraic op, record
    whether its operands are structurally *equal* (the precondition a
    factoring / absorption law needs) and how many distinct leaves the
    node touches.  Aggregated by ``(op, child-op tuple)``.
    """
    binary = {"add", "sub", "mul", "div", "matmul"}
    agg: dict = {}
    for ct in terms:
        for s in _iter_subterms(ct.term):
            if not isinstance(s, Op) or s.op not in binary:
                continue
            if len(s.args) != 2:
                continue
            key = (s.op, tuple(_op_of(a) for a in s.args))
            rec = agg.setdefault(
                key,
                {"sites": 0, "shared_child": 0, "shared_leaf": 0},
            )
            rec["sites"] += 1
            a = shape_key(s.args[0], {})
            b = shape_key(s.args[1], {})
            if a == b:
                rec["shared_child"] += 1
            if _has_shared_leaf(shape_key(s, {})):
                rec["shared_leaf"] += 1
    return agg


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _op_tuple_table(counts: Counter, terms_of: dict, top: int) -> str:
    """Render the op-tuple frequency table."""
    head = f"{'count':>6} {'terms':>6}  op-tuple"
    lines = [head, "-" * len(head)]
    for key, n in counts.most_common(top):
        kids = ", ".join(key[1])
        lines.append(
            f"{n:>6} {len(terms_of[key]):>6}  {key[0]}({kids})"
        )
    return "\n".join(lines)


def _shape_table(counts: Counter, terms_of: dict, top: int) -> str:
    """Render the shape frequency table (shared shapes flagged)."""
    head = f"{'count':>6} {'terms':>6} {'share':>5}  shape"
    lines = [head, "-" * len(head)]
    for key, n in counts.most_common(top):
        shared = "yes" if _has_shared_leaf(key) else "-"
        lines.append(
            f"{n:>6} {len(terms_of[key]):>6} {shared:>5}  "
            f"{shape_repr(key)}"
        )
    return "\n".join(lines)


def _sharing_table(agg: dict) -> str:
    """Render the binary-shape sharing table."""
    head = (
        f"{'op-tuple':<24} {'sites':>6} {'same-kids':>9} "
        f"{'shared-leaf':>11}"
    )
    lines = [head, "-" * len(head)]
    for key, rec in sorted(agg.items(), key=lambda kv: -kv[1]["sites"]):
        kids = ", ".join(key[1])
        name = f"{key[0]}({kids})"
        lines.append(
            f"{name:<24} {rec['sites']:>6} "
            f"{rec['shared_child']:>9} {rec['shared_leaf']:>11}"
        )
    return "\n".join(lines)


def _dump_json(path: str, result: dict) -> None:
    """Write the machine-readable census."""
    Path(path).write_text(json.dumps(result, indent=2) + "\n")


def run_census(top: int = _DEFAULT_TOP) -> dict:
    """Compute every census table; return a machine-readable result."""
    terms = corpus()
    op_counts, op_terms = op_tuple_census(terms)
    sh_counts, sh_terms = shape_census(terms)
    agg = sharing_census(terms)
    return {
        "n_terms": len(terms),
        "n_bench": sum(1 for t in terms if t.source == "bench"),
        "n_models": sum(1 for t in terms if t.source == "model"),
        "n_intake": sum(1 for t in terms if t.source == "intake"),
        "n_op_nodes": sum(op_counts.values()),
        "n_shapes": len(sh_counts),
        "op_tuples": [
            {
                "op": k[0],
                "children": list(k[1]),
                "count": n,
                "terms": len(op_terms[k]),
            }
            for k, n in op_counts.most_common(top)
        ],
        "shapes": [
            {
                "shape": shape_repr(k),
                "count": n,
                "terms": len(sh_terms[k]),
                "shared": _has_shared_leaf(k),
            }
            for k, n in sh_counts.most_common(top)
        ],
        "sharing": [
            {
                "op": k[0],
                "children": list(k[1]),
                **rec,
            }
            for k, rec in sorted(
                agg.items(), key=lambda kv: -kv[1]["sites"]
            )
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the census and print (or dump) the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write machine-readable results")
    parser.add_argument("--top", type=int, default=_DEFAULT_TOP)
    args = parser.parse_args(argv)

    terms = corpus()
    op_counts, op_terms = op_tuple_census(terms)
    sh_counts, sh_terms = shape_census(terms)
    agg = sharing_census(terms)

    print(  # stdout-compat
        "== law_shape_census — what shapes do real graphs contain? =="
    )
    print(  # stdout-compat
        f"   corpus: {len(terms)} terms "
        f"({sum(1 for t in terms if t.source == 'bench')} bench, "
        f"{sum(1 for t in terms if t.source == 'model')} models, "
        f"{sum(1 for t in terms if t.source == 'intake')} intake)"
    )
    print(  # stdout-compat
        f"   {sum(op_counts.values())} op nodes, "
        f"{len(op_counts)} distinct op-tuples, "
        f"{len(sh_counts)} distinct shapes"
    )
    print()  # stdout-compat
    print(f"-- op-tuple census (top {args.top}) --")  # stdout-compat
    print(  # stdout-compat
        _op_tuple_table(op_counts, op_terms, args.top)
    )
    print()  # stdout-compat
    print(  # stdout-compat
        f"-- shape census (top {args.top}; share = repeated leaf) --"
    )
    print(_shape_table(sh_counts, sh_terms, args.top))  # stdout-compat
    print()  # stdout-compat
    print("-- sharing census (binary op nodes) --")  # stdout-compat
    print(_sharing_table(agg))  # stdout-compat
    print()  # stdout-compat

    if args.json:
        _dump_json(args.json, run_census(args.top))
        print(f"wrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

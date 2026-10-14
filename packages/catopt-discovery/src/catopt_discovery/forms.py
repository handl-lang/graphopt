"""Form-level enumeration — generate law *skeletons*, not instances.

``catopt_discovery.enumerate`` closes the *instance* half of candidate
generation: 27 hand-written rewriting forms (``_f_comm`` …) are
instantiated over the op grammar and deduped.  The wall that remains is
the **form list itself** — a shape the list lacks is unreachable, and
the list is human.  Plan 0024 asks the one-level-up question: *propose
the shape, not the instance.*

This module enumerates **skeletons** over a meta-grammar.  A skeleton is
a pair of meta-terms ``(lhs, rhs)``:

* an interior node is an **op-class** — a ``("bin", slot, …)`` or
  ``("un", slot, …)`` node whose *slot* names an abstract operation the
  instantiation step later assigns a concrete op;
* a leaf is a **metavariable** (a ``str`` name) or a **literal-class**
  (the sentinel :data:`_LIT`, a slot a concrete literal fills).

The 27 shipped forms are a hand sample of this space, so the stream is
a **superset** of them: every ``_f_*`` skeleton (as produced by
``enumerate._form_pairs`` on a small op set) reappears here.  The space
is generatable; nothing regresses.

Meta-grammar
------------

Trees are built over the leaf alphabet (metavariables + one
literal-class) with ``("un", c)`` and ``("bin", a, b)`` nodes, in
increasing node count.  A skeleton is a pair ``(lhs, rhs)`` whose
**op-classes are canonical slots**: walking the pair depth-first
(lhs then rhs) each op node either reuses an already-seen slot of its
kind or introduces the next one, and the metavariables are renamed in
first-appearance order to ``X``, ``Y``, ``Z``.  Canonical labelling
counts every skeleton exactly once.

Sound-by-construction filters only (the plan's discipline): the pair is
skipped when the two sides are *identical* (a tautology — no identity),
when the ``rhs`` uses a metavariable the ``lhs`` does not (an
uninstantiable fresh metavariable — the ``meta._synthesizable``
condition), and when a side would use more distinct op-classes than the
vocabulary has ops.  Everything else flows to the existing
``proposal.evaluate`` + gauntlet — the refusal machinery is already
honest, so the form level inherits it.

Instantiation
-------------

:func:`enumerate_form_candidates` assigns concrete ops to the skeleton's
op-class slots (a bounded, deterministic product over the vocabulary)
and concrete literals to its literal-class occurrences, producing the
same declarative spec trees ``enumerate`` emits.  Each survivor flows
through the existing :func:`enumerate._filter_reason`, the alpha-key
dedup (``proposal._key``) and the library-relation annotation, so it is
a :class:`proposal.Candidate` that runs through ``proposal.evaluate``
unchanged.

Falsification
-------------

:func:`falsification_report` is the plan's test: over a fixed evaluation
budget it counts how many *admitted / paying* laws come from forms
**outside** the shipped 27.  A count of ``0`` is the plan's stated
falsification outcome — if every paying law is a re-instantiation of a
shipped form, the level adds nothing and the honest conclusion is that
the form list is the real grammar.

**Depth.**  The first run of that harness admitted 7 paying laws from
new forms, but each was a *composite re-spelling* of two shipped
identities (the cross-op-agreement family ``sub(X,0)=add(X,0)``,
``B0(X,K)=B1(K,X)``, …) — shallow, because ``proposal._relation`` only
checks alpha-equality, so a two-class form that agrees with the library
on one operand is indistinguishable from a real one.  The fix is the
**non-derivability filter** (:func:`non_derivable_true`): an *admitted*
form must produce at least one instance that is ``num_true`` yet
``verify_law`` cannot derive from ``ALL_RULES``.  The ``min_classes``
depth filter (:func:`enumerate_forms`) focuses the stream on forms over
**three** distinct op classes, which have a third identity to spend and
cannot be a single re-spelling.

Run::

    .venv/bin/python -m catopt_discovery.forms \
        --ops add,mul,neg,sub --budget 400 --evaluate 80
    .venv/bin/python -m catopt_discovery.forms --min-classes 3 \
        --budget 600 --evaluate 200

CPU-only; the enumerated prefix is the bounded part.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catopt_core.ir import TensorType, Var, op_repr

from catopt_discovery import lawdata
from catopt_discovery import proposal as lp
from catopt_discovery.enumerate import (
    DEFAULT_OPS,
    _filter_reason,
    _instantiate,
    _relation_counts,
    _split,
    _top_ranked,
)
from catopt_discovery.grammar import _evaluate as _bounded_evaluate
from catopt_discovery.object_synthesis import term_from_spec

__all__ = [
    "Skeleton",
    "deep_form_keys",
    "enumerate_form_candidates",
    "enumerate_forms",
    "falsification_report",
    "main",
    "non_derivable_true",
    "shipped_skeleton_keys",
    "skeleton_key",
]


class LitClass:
    """A meta-grammar literal-class leaf — a slot a literal fills.

    One class is enough: the instantiation step assigns a concrete
    literal to every *occurrence*, so a skeleton covers both the
    ``k = k`` and ``k ≠ k`` spellings of a literal slot.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the canonical literal-class marker ``K``."""
        return "K"


#: The single literal-class sentinel (identity-compared).
_LIT = LitClass()

#: Uniform leaf shape the metavariables instantiate on
#: (``proposal._D`` — the convention ``enumerate``/``grammar`` use).
_D: int = lp._D


@dataclass(frozen=True)
class Skeleton:
    """One law skeleton: a pair of meta-terms over op-class slots.

    ``lhs``/``rhs`` are the slot-labelled meta-terms; ``bin_ops`` /
    ``un_ops`` / ``literals`` are the instantiation vocabulary the
    skeleton was enumerated against, carried so the stream is
    self-contained.
    """

    lhs: Any
    rhs: Any
    bin_ops: tuple = ()
    un_ops: tuple = ()
    literals: tuple = ()


# ---------------------------------------------------------------------------
#  Meta-grammar trees
# ---------------------------------------------------------------------------


def _trees_by_count(max_nodes: int, leaves: tuple) -> dict[int, list]:
    """Return every meta-tree with exactly *n* op nodes, for ``n`` ≤ cap.

    ``by_n[n]`` is the list of trees with *n* op nodes over *leaves* —
    a unary node over a level-``n-1`` tree, or a binary node whose
    children split the ``n-1`` remaining nodes.  The leaves are the
    metavariable names plus the literal-class sentinel.
    """
    by_n: dict[int, list] = {0: list(leaves)}
    for n in range(1, max_nodes + 1):
        level: list = [("un", c) for c in by_n[n - 1]]
        for i in range(n):
            for a in by_n[i]:
                for b in by_n[n - 1 - i]:
                    level.append(("bin", a, b))
        by_n[n] = level
    return by_n


def _children(node: Any) -> tuple:
    """Return the child subtrees of a shape or slot-labelled node.

    A shape node is ``("bin", a, b)`` / ``("un", c)``; a slot-labelled
    node carries the op-class slot after the kind, so its children start
    at index 2.  The slot is always an ``int``, which disambiguates.
    """
    if isinstance(node, str) or node is _LIT:
        return ()
    return node[2:] if isinstance(node[1], int) else node[1:]


def _metas_of(node: Any) -> frozenset:
    """Return every metavariable name used in *node*."""
    if isinstance(node, str):
        return frozenset((node,))
    if node is _LIT:
        return frozenset()
    out: frozenset = frozenset()
    for c in _children(node):
        out |= _metas_of(c)
    return out


def _nodes(node: Any) -> int:
    """Return the number of op nodes in *node*."""
    if isinstance(node, str) or node is _LIT:
        return 0
    return 1 + sum(_nodes(c) for c in _children(node))


def _is_canonical(node: Any, metas: tuple) -> bool:
    """Return whether *node*'s metavariables are in first-appearance order.

    A canonical tree names its metavariables ``X``, ``Y``, ``Z`` … in
    the order they first appear, so the metavariable set is a prefix of
    *metas*.  Renaming a non-canonical tree into this form is the
    alpha-equivalence that lets each skeleton be enumerated once.
    """
    order: list = []

    def walk(x: Any) -> None:
        if isinstance(x, str):
            if x not in order:
                order.append(x)
        elif x is _LIT:
            return
        else:
            for c in _children(x):
                walk(c)

    walk(node)
    return order == list(metas[: len(order)])


def _kinds(pair: tuple) -> list:
    """Return the op-node kinds of *pair* in joint depth-first order.

    The traversal walks the lhs then the rhs, pre-order — the canonical
    order the slot labelling counts a skeleton once in.
    """
    kinds: list = []

    def walk(x: Any) -> None:
        if isinstance(x, str) or x is _LIT:
            return
        kinds.append(x[0][0])  # "b" or "u"
        for c in _children(x):
            walk(c)

    walk(pair[0])
    walk(pair[1])
    return kinds


def _labelings(
    kinds: list, max_bin: int, max_un: int
) -> Iterator[tuple]:
    """Yield every canonical op-class slot labelling of *kinds*.

    Each node either reuses an already-seen slot of its kind or
    introduces the next one, so the labelling is first-appearance
    canonical — the partition of op nodes into op-classes, counted
    once.  ``max_bin`` / ``max_un`` bound the distinct slots by the
    vocabulary size (a skeleton cannot name more op-classes than the
    vocabulary has ops).
    """

    def rec(i: int, b: int, u: int) -> Iterator[tuple]:
        if i == len(kinds):
            yield ()
            return
        if kinds[i] == "b":
            for s in range(min(b + 1, max_bin)):
                for rest in rec(i + 1, max(b, s + 1), u):
                    yield (s, *rest)
        else:
            for s in range(min(u + 1, max_un)):
                for rest in rec(i + 1, b, max(u, s + 1)):
                    yield (s, *rest)

    yield from rec(0, 0, 0)


def _relabel(pair: tuple, slots: tuple) -> tuple:
    """Apply an op-class slot labelling to *pair*."""
    it = iter(slots)

    def build(x: Any) -> Any:
        if isinstance(x, str) or x is _LIT:
            return x
        return (x[0], next(it), *(build(c) for c in x[1:]))

    return build(pair[0]), build(pair[1])


def _render(node: Any) -> str:
    """Render a slot-labelled skeleton (or spec) as a canonical string."""
    if isinstance(node, str):
        return node
    if node is _LIT:
        return "K"
    if isinstance(node, int):
        return str(node)
    if node[0] == "bin":
        return f"B{node[1]}({_render(node[2])}, {_render(node[3])})"
    if node[0] == "un":
        return f"U{node[1]}({_render(node[2])})"
    return f"{node[0]}({', '.join(_render(c) for c in node[1:])})"


def _canonical_slots(lhs: Any, rhs: Any, metas: tuple) -> tuple:
    """Return the slot-labelled skeleton of a concrete spec pair.

    Concrete op names become canonical op-class slots (first appearance
    in joint depth-first order) and concrete literals become the
    literal-class; metavariables are renamed to first-appearance order.
    """
    order: list = []

    def order_walk(x: Any) -> None:
        if isinstance(x, str):
            if x not in order:
                order.append(x)
        elif isinstance(x, int):
            return
        else:
            for c in x[1:]:
                order_walk(c)

    order_walk(lhs)
    order_walk(rhs)
    ren = {m: metas[i] for i, m in enumerate(order)}
    bmap: dict = {}
    umap: dict = {}

    def build(x: Any) -> Any:
        if isinstance(x, str):
            return ren[x]
        if isinstance(x, int):
            return _LIT
        is_bin = len(x) == 3  # (op, a, b) vs (op, a)
        m = bmap if is_bin else umap
        if x[0] not in m:
            m[x[0]] = len(m)
        slot = m[x[0]]
        label = "bin" if is_bin else "un"
        return (label, slot, *(build(c) for c in x[1:]))

    return build(lhs), build(rhs)


def skeleton_key(
    lhs: Any, rhs: Any, metas: tuple = ("X", "Y", "Z")
) -> str:
    """Return the canonical string key of a spec pair's skeleton.

    Concrete ops are abstracted to op-class slots and metavariables
    renamed, so two pairs with the same *shape* (and the same sharing of
    op-classes) share a key — the identity the falsification harness
    attributes a paying law to.
    """
    sl, sr = _canonical_slots(lhs, rhs, metas)
    return f"{_render(sl)} = {_render(sr)}"


# ---------------------------------------------------------------------------
#  The skeleton stream
# ---------------------------------------------------------------------------


def _pair_shapes(max_nodes: int, metas: tuple) -> Iterator[tuple]:
    """Yield ``(lhs, rhs)`` shape pairs in increasing total node count.

    Both sides are canonical shape trees (metavariables in
    first-appearance order); the rhs may use only metavariables the lhs
    does — a fresh rhs metavariable is uninstantiable.
    """
    by_n = _trees_by_count(max_nodes, (*metas, _LIT))
    for total in range(1, 2 * max_nodes + 1):
        for ln in range(1, max_nodes + 1):
            rn = total - ln
            if rn < 0 or rn > max_nodes:
                continue
            for lhs in by_n[ln]:
                if not _is_canonical(lhs, metas):
                    continue
                used = _metas_of(lhs)
                for rhs in by_n[rn]:
                    if _metas_of(rhs) <= used:
                        yield lhs, rhs


def _skeletons_of(
    pair: tuple, max_bin: int, max_un: int
) -> Iterator[tuple]:
    """Yield the non-tautological slot-labelled skeletons of *pair*."""
    for slots in _labelings(_kinds(pair), max_bin, max_un):
        sl, sr = _relabel(pair, slots)
        if sl != sr:  # a tautology is not an identity
            yield sl, sr


def _distinct_classes(lhs: Any, rhs: Any) -> int:
    """Return the number of distinct op-classes the pair names.

    Binary and unary op-classes are counted separately (a ``bin`` slot
    ``0`` and a ``un`` slot ``0`` are different classes).  This is the
    skeleton's *depth* in the sense the falsification harness measures:
    the shipped forms mostly name one or two classes, and the shallow
    cross-op-agreement family is exactly the two-class corner — a
    form that names three classes has a third identity to spend and
    cannot be a single re-spelling of one shipped law.
    """
    bins: set = set()
    uns: set = set()

    def walk(x: Any) -> None:
        if isinstance(x, str) or x is _LIT:
            return
        (bins if x[0] == "bin" else uns).add(x[1])
        for c in x[2:]:
            walk(c)

    walk(lhs)
    walk(rhs)
    return len(bins) + len(uns)


def enumerate_forms(
    *,
    max_nodes: int = 3,
    bin_ops: tuple,
    un_ops: tuple,
    literals: tuple,
    metas: tuple = ("X", "Y", "Z"),
    max_forms: int | None = None,
    min_classes: int = 1,
) -> Iterator[Skeleton]:
    """Yield law skeletons over the meta-grammar, lazily.

    *max_nodes* caps the op nodes on **each** side (the shipped 27 all
    fit at 3).  Trees are enumerated in increasing total node count, so
    a ``max_forms`` cap cuts the smallest forms first.  *bin_ops* /
    *un_ops* bound the distinct op-class slots; *literals* is carried
    for the instantiation step (one literal-class suffices — literal
    occurrences instantiate independently).

    *min_classes* is the **depth** filter: a skeleton is skipped when
    it names fewer than *min_classes* distinct op-classes.  The default
    ``1`` admits the whole space (the superset property is unaffected);
    ``3`` restricts the stream to forms over **three** distinct op
    classes — the measured reach that carries content beyond the
    two-class cross-op-agreement family the shallow falsification found
    (see :func:`falsification_report`).

    The stream is deterministic and a **superset** of the shipped
    ``enumerate._FORMS``: every hand-written form's skeleton reappears
    as a member (at the default ``min_classes=1``).
    """
    bin_ops = tuple(bin_ops)
    un_ops = tuple(un_ops)
    literals = tuple(literals)
    if not bin_ops or not un_ops:
        raise ValueError("need at least one binary and one unary op")
    metas = tuple(metas)
    max_bin = min(len(bin_ops), max_nodes)
    max_un = min(len(un_ops), max_nodes)
    n = 0
    for lhs, rhs in _pair_shapes(max_nodes, metas):
        for sl, sr in _skeletons_of((lhs, rhs), max_bin, max_un):
            if (
                min_classes > 1
                and _distinct_classes(sl, sr) < min_classes
            ):
                continue
            yield Skeleton(sl, sr, bin_ops, un_ops, literals)
            n += 1
            if max_forms is not None and n >= max_forms:
                return


# ---------------------------------------------------------------------------
#  Instantiation — skeletons to candidates
# ---------------------------------------------------------------------------


def _slot_count(sk: Skeleton) -> tuple[int, int, int]:
    """Return ``(bin slots, un slots, literal occurrences)`` of *sk*."""
    bins: set = set()
    uns: set = set()
    lits = 0

    def walk(x: Any) -> None:
        nonlocal lits
        if isinstance(x, str):
            return
        if x is _LIT:
            lits += 1
            return
        if x[0] == "bin":
            bins.add(x[1])
        else:
            uns.add(x[1])
        for c in x[2:]:
            walk(c)

    walk(sk.lhs)
    walk(sk.rhs)
    return len(bins), len(uns), lits


def _build_spec(
    node: Any, bin_assign: tuple, un_assign: tuple, lit_it: Iterator
) -> Any:
    """Instantiate *node*: op-class slots -> ops, literals -> literals."""
    if isinstance(node, str):
        return node
    if node is _LIT:
        return next(lit_it)
    if node[0] == "bin":
        return (
            bin_assign[node[1]],
            _build_spec(node[2], bin_assign, un_assign, lit_it),
            _build_spec(node[3], bin_assign, un_assign, lit_it),
        )
    return (
        un_assign[node[1]],
        _build_spec(node[2], bin_assign, un_assign, lit_it),
    )


def _assignments(sk: Skeleton) -> Iterator[tuple]:
    """Yield ``(bin_assign, un_assign, lit_assign)`` for *sk*, bounded.

    A deterministic product: each op-class slot ranges over the
    vocabulary and each literal occurrence over the literals.  The
    caller caps the number consumed.
    """
    nb, nu, nl = _slot_count(sk)
    for ba in itertools.product(sk.bin_ops, repeat=nb):
        for ua in itertools.product(sk.un_ops, repeat=nu):
            for la in itertools.product(sk.literals, repeat=nl):
                yield ba, ua, la


def _instantiated(
    sk: Skeleton, lib: list, seen: set, depth: int
) -> Iterator[tuple]:
    """Yield ``(lhs, rhs, note)`` for every instantiation of *sk*.

    Assigns concrete ops and literals, tries both orientations, drops
    the pre-filtered and alpha-duplicate pairs, and annotates each
    survivor with its library relation and skeleton key.
    """
    key_str = f"{_render(sk.lhs)} = {_render(sk.rhs)}"
    for ba, ua, la in _assignments(sk):
        ls = _build_spec(sk.lhs, ba, ua, iter(la))
        rs = _build_spec(sk.rhs, ba, ua, iter(la))
        for lhs, rhs in ((ls, rs), (rs, ls)):
            if _filter_reason(lhs, rhs, depth) is not None:
                continue
            lt, rt = term_from_spec(lhs), term_from_spec(rhs)
            key = lp._key(lt, rt)
            if key[0] == key[1] or key in seen:
                continue
            seen.add(key)
            rel = lp._relation(lt, rt, lib)
            yield (
                lhs,
                rhs,
                (
                    f"form={key_str}; {op_repr(lt)} = {op_repr(rt)}; "
                    f"relation={rel}"
                ),
            )


def enumerate_form_candidates(
    *,
    max_nodes: int = 3,
    bin_ops: tuple | None = None,
    un_ops: tuple | None = None,
    literals: tuple | None = None,
    metas: tuple = ("X", "Y", "Z"),
    max_forms: int | None = None,
    max_candidates: int | None = None,
    max_depth: int | None = None,
    min_classes: int = 1,
) -> Iterator[lp.Candidate]:
    """Yield ``Candidate``s instantiated from the skeleton stream.

    Each skeleton's op-class slots are assigned concrete ops and its
    literal-class occurrences concrete literals (a bounded product),
    the concrete pair is filtered by the existing
    :func:`enumerate._filter_reason`, deduped on the alpha-normal
    ``proposal._key``, and annotated with its library relation and its
    skeleton key.  Metavariables instantiate on ``(4,4)`` vars, so the
    emitted candidates run through ``proposal.evaluate`` unchanged.

    *min_classes* forwards the depth filter to
    :func:`enumerate_forms` — ``3`` restricts the stream to forms over
    three distinct op classes (see :func:`falsification_report`).
    """
    if bin_ops is None or un_ops is None:
        bin_ops, un_ops = _split(tuple(DEFAULT_OPS))
    lits = (
        lawdata.GRAMMAR_LITERALS
        if literals is None
        else tuple(literals)
    )
    depth = max_nodes if max_depth is None else max_depth
    lib = lp._library_keys()
    env = {m: Var(m, TensorType((_D, _D))) for m in metas}
    seen: set = set()
    n = 0
    for sk in enumerate_forms(
        max_nodes=max_nodes,
        bin_ops=bin_ops,
        un_ops=un_ops,
        literals=lits,
        metas=metas,
        max_forms=max_forms,
        min_classes=min_classes,
    ):
        for lhs, rhs, note in _instantiated(sk, lib, seen, depth):
            yield lp.Candidate(
                "form",
                f"{n}",
                _instantiate(lhs, env),
                _instantiate(rhs, env),
                note=note,
            )
            n += 1
            if max_candidates is not None and n >= max_candidates:
                return


# ---------------------------------------------------------------------------
#  Falsification — paying laws from forms outside the shipped 27
# ---------------------------------------------------------------------------


def shipped_skeleton_keys(
    bin_ops: tuple,
    un_ops: tuple,
    metas: tuple = ("X", "Y", "Z"),
) -> set[str]:
    """Return the skeleton keys of the 27 hand-written forms.

    ``enumerate._form_pairs`` is instantiated over the vocabulary and
    each concrete pair is abstracted to its skeleton key — the set the
    falsification harness attributes a paying law against.
    """
    from catopt_discovery.enumerate import _form_pairs

    keys: set[str] = set()
    for _form, lhs, rhs in _form_pairs(
        tuple(bin_ops), tuple(un_ops), metas, (0,)
    ):
        sl, sr = _canonical_slots(lhs, rhs, metas)
        if sl == sr:
            continue  # a degenerate instantiation, not a form
        keys.add(f"{_render(sl)} = {_render(sr)}")
    return keys


def _form_key(cand: lp.Candidate) -> str:
    """Extract the skeleton key a form candidate was annotated with."""
    return cand.note.split(";", 1)[0].removeprefix("form=")


def non_derivable_true(outcomes: list) -> list:
    """Return the outcomes that are numerically true but not derivable.

    The plan's **non-derivability filter**: an instance is *deep* when
    ``num_true is True`` (a real equality on random fp64 tensors) and
    ``derivable`` is ``False`` (``verify_law`` could not prove it from
    ``ALL_RULES`` under the bounded saturation).  A form that admits
    such an instance carries content the shipped library does not —
    the signal that separates a genuine new form from the shallow
    cross-op-agreement re-spellings, which are all derivable.
    """
    return [
        o for o in outcomes if o.num_true is True and not o.derivable
    ]


def deep_form_keys(outcomes: list) -> list[str]:
    """Return the sorted form keys admitting ≥1 non-derivable true law."""
    return sorted(
        {_form_key(o.candidate) for o in non_derivable_true(outcomes)}
    )


def _paying_split(paying: list, shipped: set[str]) -> dict:
    """Split paying laws by whether their form is new, then by depth."""
    new_paying = [
        o for o in paying if _form_key(o.candidate) not in shipped
    ]
    new_deep = [o for o in new_paying if not o.derivable]
    return {
        "paying_new_form": len(new_paying),
        "new_forms": sorted(
            {_form_key(o.candidate) for o in new_paying}
        ),
        "new_form_instances": len(new_paying),
        "new_form_derivable": len(new_paying) - len(new_deep),
        "new_form_non_derivable": len(new_deep),
    }


def _deep_summary(deep: list, shipped: set[str]) -> dict:
    """Summarize the non-derivable-true laws by their form."""
    keys = sorted({_form_key(o.candidate) for o in deep})
    return {
        "non_derivable_true": len(deep),
        "deep_forms": keys,
        "deep_new_forms": [k for k in keys if k not in shipped],
        "deep_shipped_forms": [k for k in keys if k in shipped],
    }


def falsification_report(
    *,
    budget: int = 400,
    evaluate_n: int = 80,
    max_nodes: int = 3,
    bin_ops: tuple | None = None,
    un_ops: tuple | None = None,
    literals: tuple | None = None,
    metas: tuple = ("X", "Y", "Z"),
    max_forms: int | None = None,
    min_classes: int = 1,
) -> dict:
    """Count paying laws from forms *outside* the shipped 27.

    Enumerates form candidates to *budget*, bounded-evaluates the first
    *evaluate_n* through the shipped oracle, and splits the *paying*
    (useful) laws into those whose skeleton is a shipped form and those
    whose skeleton is new.  The latter count is the plan's discovery
    number — ``0`` is the honest falsification outcome.

    *min_classes* restricts the stream to forms naming at least that
    many distinct op classes (the **depth** filter); ``3`` focuses on
    forms over three distinct op classes.

    The report also applies the **non-derivability filter**
    (:func:`non_derivable_true`): ``deep_forms`` are the form keys that
    admit at least one ``num_true`` instance ``verify_law`` cannot
    derive from ``ALL_RULES``, and ``deep_new_forms`` are those outside
    the shipped set — the *admitted* forms under the filter.  For the
    paying laws attributed to new forms the report splits them into
    derivable (*shallow* — a re-spelling of shipped identities) and
    non-derivable (*deep*), which is the machine-checkable statement of
    *why* the earlier falsification's new forms were shallow.
    """
    if bin_ops is None or un_ops is None:
        bin_ops, un_ops = _split(tuple(DEFAULT_OPS))
    lits = (
        lawdata.GRAMMAR_LITERALS
        if literals is None
        else tuple(literals)
    )
    shipped = shipped_skeleton_keys(bin_ops, un_ops, metas)
    stream = enumerate_form_candidates(
        max_nodes=max_nodes,
        bin_ops=bin_ops,
        un_ops=un_ops,
        literals=lits,
        metas=metas,
        max_forms=max_forms,
        max_candidates=budget,
        min_classes=min_classes,
    )
    cands = list(itertools.islice(stream, budget))
    outcomes = _bounded_evaluate(cands[:evaluate_n])
    paying = [o for o in outcomes if o.useful]
    return {
        "forms_shipped": len(shipped),
        "candidates": len(cands),
        "evaluated": len(outcomes),
        "paying": len(paying),
        "min_classes": min_classes,
        "shipped": shipped,
        "outcomes": outcomes,
        **_paying_split(paying, shipped),
        **_deep_summary(non_derivable_true(outcomes), shipped),
    }


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _fmt_report(rep: dict) -> str:
    """Render the falsification report."""
    lines = [
        "== form-level enumeration — falsification ==",
        f"   shipped form skeletons: {rep['forms_shipped']}",
        f"   candidates enumerated:  {rep['candidates']}",
        f"   evaluated (bounded):    {rep['evaluated']}",
        f"   paying (useful):        {rep['paying']}",
        f"   paying from NEW forms:  {rep['paying_new_form']}",
    ]
    if rep["paying_new_form"] == 0:
        lines.append(
            "   (falsified — every paying law re-instantiates a "
            "shipped form)"
        )
    else:
        for key in rep["new_forms"]:
            lines.append(f"     new form: {key}")
    if "non_derivable_true" in rep:
        lines.extend(_fmt_deep(rep))
    return "\n".join(lines)


def _fmt_deep(rep: dict) -> list[str]:
    """Render the non-derivability (depth) section of the report."""
    lines = [
        "   -- non-derivability filter --",
        f"   non-derivable true laws: {rep['non_derivable_true']}",
        f"   deep forms (>=1 such):   {len(rep['deep_forms'])}",
        f"   deep NEW forms:          {len(rep['deep_new_forms'])}",
        f"   new-form paying laws:    {rep['new_form_instances']} "
        f"(derivable/shallow={rep['new_form_derivable']}, "
        f"deep={rep['new_form_non_derivable']})",
    ]
    for key in rep["deep_new_forms"]:
        lines.append(f"     deep new form: {key}")
    return lines


def _parse_csv(text: str, cast: Any = str) -> tuple:
    """Parse a comma-separated CLI list into a tuple."""
    return tuple(cast(x) for x in text.split(",") if x)


def _dump_json(path: str, result: dict) -> None:
    """Write the machine-readable report."""
    Path(path).write_text(json.dumps(result, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Enumerate skeletons, instantiate, and print the falsification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ops",
        default=",".join(DEFAULT_OPS),
        help="comma-separated op vocabulary",
    )
    parser.add_argument(
        "--metas", default="X,Y,Z", help="comma-separated metavar names"
    )
    parser.add_argument(
        "--literals", default="0,1,2", help="comma-separated literals"
    )
    parser.add_argument("--max-nodes", type=int, default=3)
    parser.add_argument(
        "--max-forms",
        type=int,
        default=None,
        help="cap on enumerated skeletons",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=400,
        help="cap on instantiated candidates",
    )
    parser.add_argument(
        "--evaluate",
        type=int,
        default=80,
        help="bounded-evaluate the first N candidates",
    )
    parser.add_argument(
        "--min-classes",
        type=int,
        default=1,
        help="depth filter: min distinct op-classes per form",
    )
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    ops = _parse_csv(args.ops)
    metas = _parse_csv(args.metas)
    literals = _parse_csv(
        args.literals,
        lambda s: int(s) if s.lstrip("-").isdigit() else float(s),
    )
    bin_ops, un_ops = _split(ops)

    rep = falsification_report(
        budget=args.budget,
        evaluate_n=args.evaluate,
        max_nodes=args.max_nodes,
        bin_ops=bin_ops,
        un_ops=un_ops,
        literals=literals,
        metas=metas,
        max_forms=args.max_forms,
        min_classes=args.min_classes,
    )
    outcomes = rep.pop("outcomes")
    shipped = rep.pop("shipped")
    print(_fmt_report(rep))  # stdout-compat
    print()  # stdout-compat
    print(lp._fmt_yield(lp.yield_table(outcomes)))  # stdout-compat
    print()  # stdout-compat
    print("== genuinely-new, useful laws found ==")  # stdout-compat
    print(lp._fmt_useful(outcomes))  # stdout-compat
    print()  # stdout-compat
    print("== top-ranked (cost-delta heuristic) ==")  # stdout-compat
    print(_top_ranked(outcomes, 8))  # stdout-compat

    if args.json:
        _dump_json(
            args.json,
            {
                **rep,
                "shipped_forms": sorted(shipped),
                "relation": _relation_counts(
                    [o.candidate for o in outcomes]
                ),
                "outcomes": [
                    {
                        "lhs": op_repr(o.candidate.lhs),
                        "rhs": op_repr(o.candidate.rhs),
                        "derivable": o.derivable,
                        "num_true": o.num_true,
                        "relation": o.relation,
                        "useful": o.useful,
                        "form": _form_key(o.candidate),
                    }
                    for o in outcomes
                ],
            },
        )
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

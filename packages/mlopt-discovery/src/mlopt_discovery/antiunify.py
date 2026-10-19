"""ADR 0007 layer 1 — definitional atoms via anti-unification.

The cheapest layer of the semantic-atom pipeline: find recurring
subgraph patterns shared across the e-classes of a saturated e-graph
(least-general generalisation — Babble-style library learning), mint a
named *definitional atom* whose semantics IS its expansion, and admit
it as an extra member of every matching class.  An atom is a macro —
sound by construction, no new meaning is claimed — and it carries two
rewrites:

* ``intro`` — ``expansion → atom(args)`` — the *search artifact*.
  Fired during saturation it adds the compressed spelling to each
  class the expansion matches (:func:`admit_atoms`); folded into
  shipped rules (:func:`fold_ruleset`) it lets a rewrite mint one
  ``atom`` enode where it used to instantiate the spelled body — the
  mechanism by which an enode/iteration budget stretches further.
* ``expand`` — ``atom(args) → expansion`` — the *elaboration*
  direction, unfoldable on demand (:func:`unfold_term`).  It is kept
  OUT of every saturation set minted here: co-resident intro+expand is
  a pure two-cycle, and per the ADR nothing in this module lands in a
  default rule set — atoms are opt-in search structure only.

Honest accounting: an atom member is a *spelling*, never a cheaper
program.  :func:`unfolding_cost` prices every member through its
expansion so extraction compares semantics, and :func:`unfold_term`
rewrites an extracted term back to pure library ops before
lowering/verification — the sink never sees a ``defn_`` op.

Provenance is first-class: :class:`MintedAtom` records the mined
pattern, the e-class sites it generalises, and the instance count /
compression gain that admitted it.  ``store_atoms`` persists each
atom's ``intro`` as a ``kind="abstraction"`` declared-object record
(the ``expand`` rule embedded in provenance) through the same
:func:`mlopt_discovery.evidence.store_object` path every constructed
object takes — construction is a claim, the gauntlet is the referee.
"""

from __future__ import annotations

import heapq
import itertools
import sqlite3
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from mlopt_core.egraph import EGraph, Rewrite
from mlopt_core.egraph.terms import _replace_subterm
from mlopt_core.egraph.types import ENode
from mlopt_core.ir import Op, op_def, op_repr
from mlopt_core.meta import (
    _positions,
    instantiate_pattern,
    match_pattern,
    pattern_metavars,
)

from mlopt_discovery.object_synthesis import ConstructedObject

__all__ = [
    "DEFINITIONAL",
    "AtomCandidate",
    "MintedAtom",
    "admit_atoms",
    "atom_map",
    "fold_rule",
    "fold_ruleset",
    "mine_candidates",
    "mint_atom",
    "mint_atoms",
    "store_atoms",
    "unfold_term",
    "unfolding_cost",
]


#: Provenance tag on every rule this module mints (atom ``intro`` /
#: ``expand`` and folded variants).  Not a ``laws.tags`` constant —
#: it is a *provenance* mark, not a structural classification, and it
#: exists so callers can exclude minted rules by tag.
DEFINITIONAL = "definitional"

#: Bound on instances enumerated per emitted candidate — provenance
#: completeness, not correctness (the ``intro`` rule re-matches every
#: site itself at admit time).
_SITE_CAP = 512


# ---------------------------------------------------------------------------
#  Candidates — the mined patterns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomCandidate:
    """A recurring subgraph pattern mined from a saturated e-graph.

    ``pattern`` is an :class:`~mlopt_core.ir.Op` tree whose leaves
    are term metavariables (``v0``, ``v1``, ...) and whose attribute
    values may be attribute metavariables (``a0``, ...) — the
    least-general generalisation of every instance.  ``sites`` is the
    e-class id of each observed occurrence's root — the provenance the
    minted atom carries.  ``gain`` is the description-length estimate
    ``uses * (op_count - 1) - op_count``: positive means naming the
    pattern compresses the term bank it was mined from.
    """

    pattern: Any
    leaf_mvars: tuple[str, ...]
    attr_mvars: tuple[str, ...]
    sites: tuple[int, ...]
    n_instances: int
    op_count: int
    gain: int


@dataclass
class _Cand:
    """Mutable mining candidate: a pattern plus one row per instance.

    ``rows`` holds each instance's root e-class; ``cols`` maps each
    leaf-metavar name to the column of e-classes it binds (one entry
    per row), ``acols`` each attr-metavar name to its column of attr
    values.  Two positions share a metavariable iff their columns are
    equal — the anti-unification constraint lifted to e-classes, which
    is what captures shared subterms (``mul(x, x)``).
    """

    pat: Any
    rows: list[int]
    cols: dict[str, tuple[int, ...]] = field(default_factory=dict)
    acols: dict[str, tuple[Any, ...]] = field(default_factory=dict)


def _n_ops(pat: Any) -> int:
    """Count :class:`Op` nodes in a pattern term."""
    if not isinstance(pat, Op):
        return 0
    return 1 + sum(_n_ops(a) for a in pat.args)


def _leaf_mvars(pat: Any) -> tuple[str, ...]:
    """Leaf metavariable names in DFS first-occurrence order."""
    out: list[str] = []

    def go(t: Any) -> None:
        if isinstance(t, str):
            if t not in out:
                out.append(t)
        elif isinstance(t, Op):
            for a in t.args:
                go(a)

    go(pat)
    return tuple(out)


def _attr_mvars_of(pat: Any) -> tuple[str, ...]:
    """Attr metavariable names (string attr values) in DFS order."""
    out: list[str] = []

    def go(t: Any) -> None:
        if isinstance(t, Op):
            for v in t.attrs.values():
                if isinstance(v, str) and v not in out:
                    out.append(v)
            for a in t.args:
                go(a)

    go(pat)
    return tuple(out)


def _alpha_norm(pat: Any) -> str:
    """Alpha-normal key: metavariables renamed by first occurrence."""
    env: dict[str, str] = {}
    aenv: dict[str, str] = {}

    def go(t: Any) -> Any:
        if isinstance(t, str):
            if t not in env:
                env[t] = f"v{len(env)}"
            return env[t]
        if not isinstance(t, Op):
            return t
        attrs: dict[str, Any] = {}
        for k, v in t.attrs.items():
            if isinstance(v, str):
                if v not in aenv:
                    aenv[v] = f"a{len(aenv)}"
                attrs[k] = aenv[v]
            else:
                attrs[k] = v
        return Op.make(t.op, *(go(a) for a in t.args), **attrs)

    return op_repr(go(pat))


def _metavar_count(pat: Any, name: str) -> int:
    """Occurrences of leaf metavariable *name* in *pat*."""
    if isinstance(pat, str):
        return int(pat == name)
    if isinstance(pat, Op):
        return sum(_metavar_count(a, name) for a in pat.args)
    return 0


# ---------------------------------------------------------------------------
#  The member index and refinement algebra
# ---------------------------------------------------------------------------


def _member_index(eg: EGraph) -> dict[int, dict[tuple[str, int], list]]:
    """Non-leaf members by ``(class, (op, arity))``, canonically sorted."""
    idx: dict[int, dict[tuple[str, int], list[ENode]]] = {}
    for cid, ec in eg._classes.items():
        by: dict[tuple[str, int], list[ENode]] = {}
        for n in ec.nodes:
            if n.op != "leaf":
                by.setdefault((n.op, len(n.children)), []).append(n)
        for members in by.values():
            members.sort(key=lambda n: (n.children, repr(n.attrs)))
        idx[cid] = by
    return idx


def _seed(
    occ: list[tuple[int, ENode]],
    vctr: Iterator[int],
    actr: Iterator[int],
) -> _Cand:
    """Build the all-holes seed candidate for one occurrence group.

    *occ* is ``[(root class, enode)]`` — every enode already shares
    ``(op, arity, attr key-set)`` (or full attrs in structure-only
    mode).  Leaf positions become ``v*`` metavariables merged on equal
    columns; an attr whose value differs across rows becomes an ``a*``
    metavar under the same column-equality rule — the grouping
    guarantees key-set agreement, so a difference IS a metavariable
    (in structure-only mode the group key IS the full attr tuple, so
    values can never differ here).
    """
    first = occ[0][1]
    cols: dict[str, tuple[int, ...]] = {}
    names: list[str] = []
    for pos in range(len(first.children)):
        col = tuple(n.children[pos] for _, n in occ)
        names.append(_col_name(col, cols, vctr))
    attrs: dict[str, Any] = {}
    acols: dict[str, tuple[Any, ...]] = {}
    for k, _ in first.attrs:
        vals = tuple(dict(n.attrs)[k] for _, n in occ)
        if all(v == vals[0] for v in vals):
            attrs[k] = vals[0]
        else:
            attrs[k] = _acol_name(vals, acols, actr)
    return _Cand(
        pat=Op.make(first.op, *names, **attrs),
        rows=[cid for cid, _ in occ],
        cols=cols,
        acols=acols,
    )


def _col_name(
    col: tuple[int, ...],
    cols: dict[str, tuple[int, ...]],
    ctr: Iterator,
) -> str:
    """Metavariable name for a bound-class column — shared iff equal."""
    for name, existing in cols.items():
        if existing == col:
            return name
    name = f"v{next(ctr)}"
    cols[name] = col
    return name


def _acol_name(
    vals: tuple[Any, ...],
    acols: dict[str, tuple[Any, ...]],
    ctr: Iterator,
) -> str:
    """Attr-metavar name for a bound-value column — shared iff equal."""
    for name, existing in acols.items():
        if existing == vals:
            return name
    name = f"a{next(ctr)}"
    acols[name] = vals
    return name


def _refine_options(
    cand: _Cand,
    mvar: str,
    midx: dict[int, dict[tuple[str, int], list]],
    attr_mvars: bool,
    min_uses: int,
) -> list[tuple]:
    """Signatures ``(op, arity, keyset)`` a metavariable can grow into.

    A row supports a signature when the class bound to *mvar* in that
    row holds a member of that op/arity/key-set — support counts
    *rows* (occurrences), matching the pattern's instance semantics.
    """
    cnt: Counter = Counter()
    for c in cand.cols[mvar]:
        sigs: set[tuple] = set()
        for (op, arity), members in midx[c].items():
            for n in members:
                ks = (
                    tuple(k for k, _ in n.attrs)
                    if attr_mvars
                    else n.attrs
                )
                sigs.add((op, arity, ks))
        cnt.update(sigs)
    return sorted(
        (sig for sig, n in cnt.items() if n >= min_uses), key=repr
    )


def _pick(
    midx: dict[int, dict[tuple[str, int], list]],
    cid: int,
    sig: tuple,
    attr_mvars: bool,
) -> ENode | None:
    """Canonical (sorted-first) member of *cid* matching *sig*."""
    op, arity, ks = sig
    for n in midx[cid].get((op, arity), ()):
        key = tuple(k for k, _ in n.attrs) if attr_mvars else n.attrs
        if key == ks:
            return n
    return None


def _restrict(
    cand: _Cand, mvar: str, support: list[int]
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[Any, ...]]]:
    """Columns of every metavariable except *mvar*, cut to *support*."""
    cols = {
        name: tuple(col[i] for i in support)
        for name, col in cand.cols.items()
        if name != mvar
    }
    acols = {
        name: tuple(col[i] for i in support)
        for name, col in cand.acols.items()
    }
    return cols, acols


def _grow_attrs(
    members: list[ENode],
    acols: dict[str, tuple[Any, ...]],
    actr: Iterator[int],
) -> dict[str, Any]:
    """Attr map for the grown node — literals or ``a*`` metavars.

    All-equal values stay literal; differing values become attr
    metavars — the signature's key-set groups on attr keys, so in
    structure-only mode ``_pick`` already required identical values
    and the else-branch is unreachable.
    """
    attrs: dict[str, Any] = {}
    for k, _ in members[0].attrs:
        vals = tuple(dict(n.attrs)[k] for n in members)
        if all(v == vals[0] for v in vals):
            attrs[k] = vals[0]
        else:
            attrs[k] = _acol_name(vals, acols, actr)
    return attrs


def _grow(
    cand: _Cand,
    mvar: str,
    sig: tuple,
    support: list[int],
    members: list[ENode],
    vctr: Iterator[int],
    actr: Iterator[int],
) -> _Cand:
    """Refine *mvar* into *sig*'s node over the supporting rows.

    Non-supporting rows are dropped — refinement strictly specialises
    the pattern.  Child columns merge onto existing metavariables when
    equal (the ``mul(x, x)`` sharing rule); differing attr values
    become attr metavariables under the same column rule.
    """
    op, arity, _ = sig
    cols, acols = _restrict(cand, mvar, support)
    names: list[str] = []
    for pos in range(arity):
        col = tuple(m.children[pos] for m in members)
        names.append(_col_name(col, cols, vctr))
    node = Op.make(op, *names, **_grow_attrs(members, acols, actr))
    return _Cand(
        pat=_swap_hole(cand.pat, mvar, node),
        rows=[cand.rows[i] for i in support],
        cols=cols,
        acols=acols,
    )


def _swap_hole(pat: Any, mvar: str, node: Any) -> Any:
    """Replace every leaf *mvar* in *pat* by *node* (shared => all)."""
    if isinstance(pat, str):
        return node if pat == mvar else pat
    if isinstance(pat, Op):
        return Op.make(
            pat.op,
            *(_swap_hole(a, mvar, node) for a in pat.args),
            **pat.attrs,
        )
    return pat


def _grow_all(
    cand: _Cand,
    mvar: str,
    sig: tuple,
    midx: dict[int, dict[tuple[str, int], list]],
    vctr: Iterator[int],
    actr: Iterator[int],
    attr_mvars: bool,
) -> _Cand:
    """Collect the supporting rows of (*mvar*, *sig*) and refine."""
    support: list[int] = []
    members: list[ENode] = []
    for i, c in enumerate(cand.cols[mvar]):
        n = _pick(midx, c, sig, attr_mvars)
        if n is not None:
            support.append(i)
            members.append(n)
    return _grow(cand, mvar, sig, support, members, vctr, actr)


def _score(cand: _Cand) -> int:
    """Heap ordering key — larger is more promising."""
    return len(cand.rows) * _n_ops(cand.pat)


def _emit(eg: EGraph, cand: _Cand, site_cap: int) -> AtomCandidate:
    """Freeze a candidate, re-deriving its true instance set.

    The mining row set is a lower bound — the pattern's real coverage
    is every e-class where the real matcher binds it.  Re-deriving at
    emit keeps provenance exact and removes row-subset artifacts of
    the greedy refinement order.  Sites and counts are capped by
    *site_cap* (provenance completeness, not correctness — ``intro``
    re-matches exhaustively at admit time).
    """
    sites: list[int] = []
    n = 0
    for cid in sorted(eg._classes):
        hits = list(eg.matches(cand.pat, cid, max_results=site_cap))
        n += len(hits)
        if hits:
            sites.append(cid)
    ops = _n_ops(cand.pat)
    return AtomCandidate(
        pattern=cand.pat,
        leaf_mvars=_leaf_mvars(cand.pat),
        attr_mvars=_attr_mvars_of(cand.pat),
        sites=tuple(sites),
        n_instances=n,
        op_count=ops,
        gain=n * (ops - 1) - ops,
    )


def mine_candidates(
    eg: EGraph,
    *,
    min_ops: int = 2,
    min_uses: int = 2,
    max_ops: int = 8,
    attr_mvars: bool = True,
    beam: int = 32,
    max_expand: int = 2000,
    skip_ops: Iterable[str] = (),
    site_cap: int = _SITE_CAP,
) -> list[AtomCandidate]:
    """Anti-unify recurring subgraph patterns over *eg*'s classes.

    Seeds one all-holes candidate per ``(op, arity, attr-key-set)``
    occurrence group with at least *min_uses* enodes, then grows each
    by refining a metavariable into an op node its bound classes
    share — best-first under a *beam* cap, alpha-deduped, terminating
    at *max_ops* per pattern or *max_expand* pops.  Every candidate
    with ``op_count >= min_ops`` and ``instances >= min_uses`` is a
    result; *skip_ops* excludes op names from seeding AND refinement
    (already-minted atoms stay minable for iterated rounds — pass
    ``skip_ops`` to exclude them explicitly).

    ``attr_mvars=False`` is structure-only mode: enodes differing in
    any attr value never share a pattern.
    """
    midx = _member_index(eg)
    skip = set(skip_ops)
    vctr = itertools.count()
    actr = itertools.count()
    seeds = _seed_groups(midx, attr_mvars, min_uses, skip)
    work: list[tuple[int, int, _Cand]] = []
    seq = itertools.count()
    pushed: set[str] = set()
    for occ in seeds:
        c = _seed(occ, vctr, actr)
        # Distinct seed groups differ in (op, arity, key-set), so their
        # alpha-normal seeds can never collide — push unconditionally.
        pushed.add(_alpha_norm(c.pat))
        heapq.heappush(work, (-_score(c), next(seq), c))
    out: dict[str, AtomCandidate] = {}
    pops = 0
    while work and pops < max_expand:
        _ns, _q, cand = heapq.heappop(work)
        pops += 1
        _maybe_emit(eg, out, cand, min_ops, min_uses, site_cap)
        _grow_children(
            cand,
            _n_ops(cand.pat),
            midx,
            work,
            pushed,
            seq,
            vctr,
            actr,
            attr_mvars,
            min_uses,
            max_ops,
        )
        if len(work) > beam:
            work = heapq.nsmallest(beam, work)
            heapq.heapify(work)
    return sorted(
        out.values(),
        key=lambda c: (-c.gain, -c.n_instances, _alpha_norm(c.pattern)),
    )


def _maybe_emit(
    eg: EGraph,
    out: dict[str, AtomCandidate],
    cand: _Cand,
    min_ops: int,
    min_uses: int,
    site_cap: int,
) -> None:
    """Freeze *cand* into *out* when it clears the mint bar."""
    key = _alpha_norm(cand.pat)
    if (
        _n_ops(cand.pat) >= min_ops
        and len(cand.rows) >= min_uses
        and key not in out
    ):
        out[key] = _emit(eg, cand, site_cap)


def _grow_children(
    cand: _Cand,
    ops: int,
    midx: dict[int, dict[tuple[str, int], list]],
    work: list,
    pushed: set,
    seq: Iterator[int],
    vctr: Iterator[int],
    actr: Iterator[int],
    attr_mvars: bool,
    min_uses: int,
    max_ops: int,
) -> None:
    """Push every deduped refinement of *cand* onto the worklist."""
    if ops >= max_ops:
        return
    for mvar in list(cand.cols):
        if _metavar_count(cand.pat, mvar) + ops > max_ops:
            continue
        for sig in _refine_options(
            cand, mvar, midx, attr_mvars, min_uses
        ):
            child = _grow_all(
                cand,
                mvar,
                sig,
                midx,
                vctr,
                actr,
                attr_mvars,
            )
            ckey = _alpha_norm(child.pat)
            if ckey in pushed:
                continue
            pushed.add(ckey)
            heapq.heappush(work, (-_score(child), next(seq), child))


def _seed_groups(
    midx: dict[int, dict[tuple[str, int], list]],
    attr_mvars: bool,
    min_uses: int,
    skip: set,
) -> list[list[tuple[int, ENode]]]:
    """Occurrence groups ``[(class, enode)]`` by signature, min_uses+."""
    groups: dict[tuple, list[tuple[int, ENode]]] = {}
    for cid, by in midx.items():
        for (op, arity), members in by.items():
            if op in skip:
                continue
            for n in members:
                ks = (
                    tuple(k for k, _ in n.attrs)
                    if attr_mvars
                    else n.attrs
                )
                groups.setdefault((op, arity, ks), []).append((cid, n))
    return [g for g in groups.values() if len(g) >= min_uses]


# ---------------------------------------------------------------------------
#  Minted atoms — the definitional object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MintedAtom:
    """A definitional atom: a macro with provenance.

    ``name`` is the registered op name; ``expansion`` the mined
    pattern the atom unfolds to; ``call`` the pattern spelling
    ``atom(leaf_mvars, attr_mvars-as-attrs)``.  ``intro`` /
    ``expand`` are the definitional pair — *intro* is the search
    artifact, *expand* the elaboration direction (never saturated
    here).  ``object`` is the ADR-0004 :class:`ConstructedObject`
    wrapping ``intro``.  ``sites`` / ``n_instances`` / ``gain`` are
    the measurement that admitted the atom.
    """

    name: str
    arity: int
    expansion: Any
    leaf_mvars: tuple[str, ...]
    attr_mvars: tuple[str, ...]
    call: Op
    intro: Rewrite
    expand: Rewrite
    object: ConstructedObject
    sites: tuple[int, ...]
    n_instances: int
    gain: int


def mint_atom(cand: AtomCandidate, *, name: str) -> MintedAtom:
    """Register *cand* as a definitional atom under op *name*.

    Mints the generator (:func:`~mlopt_core.ir.op_def`), the
    ``intro``/``expand`` definitional pair, and the
    :class:`ConstructedObject` record — the atom's semantics IS the
    expansion, so no guard is needed anywhere: a macro is sound by
    construction at every site it matches.
    """
    op_def(
        name,
        len(cand.leaf_mvars),
        1,
        law=f"definitional atom for {op_repr(cand.pattern)} "
        f"(ADR 0007 layer 1 — macro, unfolds through `expand`)",
    )
    call_attrs: dict[str, Any] = {a: a for a in cand.attr_mvars}
    call = Op.make(name, *cand.leaf_mvars, **call_attrs)
    intro = Rewrite(
        name=f"{name}.intro",
        lhs=cand.pattern,
        rhs=call,
        law=f"{name} is a macro for {op_repr(cand.pattern)} — "
        "definitional equality, sound by construction",
        tags=frozenset({DEFINITIONAL}),
    )
    expand = Rewrite(
        name=f"{name}.expand",
        lhs=call,
        rhs=cand.pattern,
        law=f"{name} unfolds to {op_repr(cand.pattern)} — "
        "elaboration direction only; keep out of saturation",
        tags=frozenset({DEFINITIONAL}),
    )
    obj = ConstructedObject(
        rule=intro,
        kind="abstraction",
        construction=("antiunify", name),
        note=(
            f"anti-unified over {cand.n_instances} instances "
            f"({len(cand.sites)} classes); op_count={cand.op_count} "
            f"gain={cand.gain}"
        ),
    )
    return MintedAtom(
        name=name,
        arity=len(cand.leaf_mvars),
        expansion=cand.pattern,
        leaf_mvars=cand.leaf_mvars,
        attr_mvars=cand.attr_mvars,
        call=call,
        intro=intro,
        expand=expand,
        object=obj,
        sites=cand.sites,
        n_instances=cand.n_instances,
        gain=cand.gain,
    )


def mint_atoms(
    eg: EGraph,
    *,
    count: int = 6,
    prefix: str = "defn",
    min_gain: int = 1,
    **mine_kw: Any,
) -> list[MintedAtom]:
    """Mine *eg* and mint the top *count* paying candidates as atoms.

    Only candidates with ``gain >= min_gain`` — a strictly positive
    description-length saving — are minted; the rest stay honest
    near-misses.  Names are ``{prefix}_{i}`` with *i* advanced past
    any already-registered generator, so iterated minting rounds
    never collide.
    """
    from mlopt_core.ir import generator

    atoms: list[MintedAtom] = []
    i = 0
    for cand in mine_candidates(eg, **mine_kw):
        if len(atoms) >= count:
            break
        if cand.gain < min_gain:
            continue
        while generator(f"{prefix}_{i}") is not None:
            i += 1
        atoms.append(mint_atom(cand, name=f"{prefix}_{i}"))
        i += 1
    return atoms


def atom_map(atoms: Iterable[MintedAtom] | dict) -> dict:
    """Name-keyed table of minted atoms (idempotent)."""
    if isinstance(atoms, dict):
        return atoms
    return {a.name: a for a in atoms}


# ---------------------------------------------------------------------------
#  Admission — the atom joins the graph
# ---------------------------------------------------------------------------


def admit_atoms(
    eg: EGraph, root_eid: int, atoms: Iterable[MintedAtom]
) -> dict[str, int]:
    """Fire each atom's ``intro`` once over the whole graph.

    Every class matching the expansion gains the ``atom`` member as
    an ordinary rule application (witnessed, certifiable) — *refuse
    never drop*: nothing is removed, the spelled members stay.
    Returns ``{atom name: merge count}``.
    """
    out: dict[str, int] = {}
    for a in atoms:
        eg.apply_rule(a.intro, root_eid)
        out[a.name] = eg.rule_fires.get(a.intro.name, 0)
    eg.rebuild()
    return out


# ---------------------------------------------------------------------------
#  Folded rules — the budget-stretching mechanism
# ---------------------------------------------------------------------------


def fold_rule(
    rule: Rewrite,
    atom: MintedAtom,
    *,
    side: str = "rhs",
    max_positions: int = 1,
) -> list[Rewrite]:
    """Mint *rule*'s folded variants at positions matching the expansion.

    ``side="rhs"`` — each position of ``rule.rhs`` where
    ``atom.expansion`` matches is rewritten to the ``atom`` call: the
    variant produces one enode where the parent instantiated the
    spelled body (sound — it is the composite ``rule ; intro``).  A
    position is folded only when every metavariable the atom call
    references is bound by ``rule.lhs`` — a folded rule that could
    never instantiate is not minted.

    ``side="lhs"`` — the variant's LHS carries the atom spelling, so
    the rule also fires on classes that hold ONLY the compressed
    member — partial compensation for the coverage a strictly-folded
    RHS hides inside the atom's body (a narrowing: always sound).

    Names are ``{rule}~{atom}.{side[0]}{i}``; hooks, cond/dspec, tags
    (plus ``definitional``) and error bounds carry through.
    """
    if side not in ("lhs", "rhs"):
        raise ValueError(f"side must be 'lhs' or 'rhs', got {side!r}")
    src = rule.lhs if side == "lhs" else rule.rhs
    bound_ok = pattern_metavars(rule.lhs)
    out: list[Rewrite] = []
    for path, sub in _positions(src):
        if len(out) >= max_positions:
            break
        s = match_pattern(atom.expansion, sub, {})
        if s is None:
            continue
        call = instantiate_pattern(atom.call, s)
        if side == "rhs" and not (pattern_metavars(call) <= bound_ok):
            continue
        new = _replace_subterm(src, path, call)
        lhs, rhs = (new, rule.rhs) if side == "lhs" else (rule.lhs, new)
        out.append(
            Rewrite(
                name=f"{rule.name}~{atom.name}.{side[0]}{len(out)}",
                lhs=lhs,
                rhs=rhs,
                law=f"{rule.law} [{side}-folded by {atom.name}]",
                check=rule.check,
                derive=rule.derive,
                tags=rule.tags | {DEFINITIONAL},
                error_bound=rule.error_bound,
                bound_norm=rule.bound_norm,
                derivation=(rule.name, atom.intro.name),
                cond=rule.cond,
                dspec=rule.dspec,
            )
        )
    return out


def _unfold_hook(fn: Any, amap: dict) -> Any:
    """Wrap a rule ``check``/``derive`` hook to unfold defn terms.

    ``apply_rule`` builds ``bound`` metavar → class-term values via
    ``any_term`` — under an atom-admitted graph that can hand a hook a
    ``defn``-headed term it cannot shape-infer (carrier ``_vshape``
    walks the op tree).  The wrapper elaborates every bound term back
    to pure library ops before the hook sees it — the hook's contract
    is unchanged.
    """
    if fn is None:
        return None

    def wrapped(bound: dict) -> Any:
        return fn(
            {
                k: (
                    v
                    if k.startswith("$attr:")
                    else unfold_term(v, amap)
                )
                for k, v in bound.items()
            }
        )

    return wrapped


def _wrap_hooks(rule: Rewrite, amap: dict) -> Rewrite:
    """*rule* with ``check``/``derive`` hooks unfolded through *amap*."""
    if rule.check is None and rule.derive is None:
        return rule
    return Rewrite(
        name=rule.name,
        lhs=rule.lhs,
        rhs=rule.rhs,
        law=rule.law,
        check=_unfold_hook(rule.check, amap),
        derive=_unfold_hook(rule.derive, amap),
        tags=rule.tags,
        error_bound=rule.error_bound,
        bound_norm=rule.bound_norm,
        derivation=rule.derivation,
        cond=rule.cond,
        dspec=rule.dspec,
    )


def fold_ruleset(
    rules: Iterable[Rewrite],
    atoms: Iterable[MintedAtom],
    *,
    sides: tuple[str, ...] = ("rhs",),
    max_positions: int = 1,
    replace: bool = True,
) -> list[Rewrite]:
    """Fold a rule set over the atoms — the compressed saturation set.

    For every ``rule`` x ``atom`` pair the foldable positions mint
    variants via :func:`fold_rule`.  ``replace=True`` (the strict
    arm) drops a rule that gained an ``rhs``-folded variant — the
    folded spelling covers its sites with one enode instead of the
    body, which is the entire budget mechanism.  ``replace=False``
    keeps parents alongside (more coverage, less saving).  Folded
    rules inherit their parent's ``rule_budgets`` treatment by NAME
    — remap budgets on the folded names when scheduling.

    Every returned rule's ``check``/``derive`` hooks are wrapped to
    unfold bound terms first (``_unfold_hook``): an atom-admitted
    graph can hand a hook a ``defn``-headed term through
    ``any_term``, and carrier hooks shape-infer on the tree — the
    wrapper keeps their contract honest without core knowing defn
    names exist.
    """
    amap_list = list(atoms)
    amap = atom_map(amap_list)
    folded: list[Rewrite] = []
    folded_names: set[str] = set()
    for r in rules:
        gained_rhs = False
        for a in amap_list:
            if "rhs" in sides:
                vs = fold_rule(
                    r, a, side="rhs", max_positions=max_positions
                )
                folded.extend(vs)
                gained_rhs = gained_rhs or bool(vs)
            if "lhs" in sides:
                folded.extend(
                    fold_rule(
                        r, a, side="lhs", max_positions=max_positions
                    )
                )
        if gained_rhs:
            folded_names.add(r.name)
    out = [
        _wrap_hooks(r, amap)
        for r in rules
        if not (replace and r.name in folded_names)
    ]
    return out + [_wrap_hooks(r, amap) for r in folded]


# ---------------------------------------------------------------------------
#  Elaboration — unfold at the extraction boundary
# ---------------------------------------------------------------------------


def unfold_term(term: Any, atoms: Iterable[MintedAtom] | dict) -> Any:
    """Expand every atom op in *term* — the macro is unfolded away.

    The extracted term's ``atom(args)`` is replaced by the atom's
    expansion instantiated on the (already-unfolded) args and attrs —
    recursively, so atoms minted over earlier atoms elaborate fully.
    The result carries only the op vocabulary the term bank started
    with; the sink never sees a ``defn_`` name.
    """
    amap = atom_map(atoms)
    if not isinstance(term, Op):
        return term
    args = tuple(unfold_term(a, amap) for a in term.args)
    a = amap.get(term.op)
    if a is None:
        return Op.make(term.op, *args, **term.attrs)
    subst = dict(zip(a.leaf_mvars, args, strict=True))
    for k, v in term.attrs.items():
        subst["$attr:" + k] = v
    return unfold_term(instantiate_pattern(a.expansion, subst), amap)


def _expansion_body_cost(a: MintedAtom, term: Any, raw: Any) -> float:
    """Return the atom member's honest local bill — its body ops.

    Walks the expansion instantiated on the call's *bound* children —
    internal ops contribute ``cost_fn(node) - sum cost_fn(child)``,
    the same local identity ``extract_best`` uses; bound leaf
    positions contribute 0 (they are priced at the call site, per
    position).  The subtraction is exact for additive models even
    when a bound child is itself atom-headed: ``cost_fn``'s own
    recursion subtracts the same value on both sides.
    """
    subst = dict(zip(a.leaf_mvars, term.args, strict=True))
    for k, v in term.attrs.items():
        subst["$attr:" + k] = v

    def resolve(p: Any) -> Any:
        if isinstance(p, str):
            return subst[p]
        attrs = {
            k: subst.get("$attr:" + str(v), v)
            if isinstance(v, str)
            else v
            for k, v in p.attrs.items()
        }
        return Op.make(p.op, *(resolve(x) for x in p.args), **attrs)

    def go(p: Any) -> float:
        if not isinstance(p, Op):
            return 0.0  # bound leaf position — priced at the call site
        node = resolve(p)
        local = raw(node) - sum(raw(c) for c in node.args)
        return max(local, 0.0) + sum(go(x) for x in p.args)

    return go(a.expansion)


def unfolding_cost(
    cost_fn: Any, atoms: Iterable[MintedAtom] | dict
) -> Any:
    """Wrap *cost_fn* to price members through their expansions.

    The honest-pricing contract: a ``defn`` member must cost exactly
    its body's ops plus the per-position cost of its bound children —
    ``priced(atom(cs)) = body_ops + Σ priced(child)`` — so extraction's
    ``local = ct - child_tc`` lands on the body's own ops (a naive
    ``cost_fn ∘ unfold`` dedups shared bound children inside the
    unfolded tree while ``child_tc`` bills each position — the
    difference clamps ``local`` to 0 and the extractor prefers
    *free* atom spellings; this wrapper removes that bias).  Non-atom
    terms price as ``cost_fn(unfold(term))``.  The wrapped model's
    ``charges_param_only`` / ``dag_exact`` / ``profile`` markers
    propagate, matching :func:`mlopt_core.cost.backend_cost`'s
    contract.
    """
    import inspect

    amap = atom_map(atoms)
    try:
        accepts_memo = "memo" in inspect.signature(cost_fn).parameters
    except (TypeError, ValueError):  # uninspectable callable
        accepts_memo = False

    def raw(t: Any, memo: dict | None = None) -> float:
        return cost_fn(t, memo) if accepts_memo else cost_fn(t)

    def priced(term: Any, memo: dict | None = None) -> float:
        if not isinstance(term, Op):
            return raw(term, memo)
        a = amap.get(term.op)
        if a is None:
            return raw(unfold_term(term, amap), memo)
        body = _expansion_body_cost(a, term, lambda t: raw(t, memo))
        return body + sum(priced(c, memo) for c in term.args)

    priced.__name__ = getattr(cost_fn, "__name__", "unfolding_cost")
    for marker in ("charges_param_only", "dag_exact", "profile"):
        if hasattr(cost_fn, marker):
            setattr(priced, marker, getattr(cost_fn, marker))
    return priced


# ---------------------------------------------------------------------------
#  Storage — the declared-object record
# ---------------------------------------------------------------------------


def store_atoms(
    conn: sqlite3.Connection,
    atoms: Iterable[MintedAtom],
    corpus_hash: str = "",
) -> list[str]:
    """Persist each atom's ``intro`` as a declared-object record.

    The record is the ``kind="abstraction"`` law record (the
    ADR-0004 admission format) with provenance carrying the atom's
    own data — name, arity, the ``expand`` rule serialised as data,
    the mined sites and the gain.  Returns the alpha keys in atom
    order.  Storage is not admission: the record still faces
    ``evidence.run_gauntlet``.
    """
    from mlopt_core.laws.serialize import law_to_data

    from mlopt_discovery import evidence as ev

    keys: list[str] = []
    for a in atoms:
        prov = {
            "origin": "constructed",
            "op": "antiunify",
            "premises": [],
            "note": a.object.note,
            "atom": a.name,
            "arity": a.arity,
            "expansion": law_to_data(a.expand),
            "sites": list(a.sites),
            "instances": a.n_instances,
            "gain": a.gain,
        }
        keys.append(
            ev.store_object(
                conn,
                a.intro,
                corpus_hash,
                kind="abstraction",
                cert=None,
                provenance=prov,
            )
        )
    return keys

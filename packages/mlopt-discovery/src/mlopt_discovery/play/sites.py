"""Spec-site collectors — the term-traversal knowledge behind minting.

Fusion/reduction/decayed/scan site collectors (saturated-graph
carriers *and* the raw recurrence chains that mint the sequence
forms), the per-tier ``TIER_SITES`` table pairing with
``genkernel.TIERS``, and :func:`_mint_tiers` — the intake tail every
case builder (``gen``, ``zoo``, ``model_corpus``) runs.  Extracted
from the former monolithic ``play.py``; names are re-exported through
``mlopt_discovery.play`` unchanged.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from mlopt_core.ir import Const, Op, Param, Var
from mlopt_core.traversal import ops as _term_ops
from mlopt_core.typing import (
    register_shape_rule as _register_shape_rule,
)

from ..bounds import ProbeBounds

# ---------------------------------------------------------------------------
#  Domain: gen — generated kernels over real torch modules
# ---------------------------------------------------------------------------
#
#  The "discovery builds lowerings" story end to end: spell fusion
#  patterns the ambient handler table has no name for (``x·relu(x)``,
#  ``x·gelu(x)``, …) plus stock-model pieces; the board mints a gen
#  handler + kernel per occurring elementwise subterm; ``claim``
#  binds it; ``deliver`` lowers it through a ``TorchSink`` whose op
#  table includes the generated binding — a runnable module calling
#  a kernel nobody wrote.


def _leaf_name(t: Any, names: dict) -> str:
    """Name a pattern leaf by structural identity.

    Key leaves by the hash-consed node itself, not ``repr``: a
    non-elementwise ``Op`` child's repr tree-expands the whole
    shared DAG — exponential on residual-stream models (the
    whisper-encoder hang).  Interned terms make object identity
    the structural-equality test ``repr`` approximated.
    """
    key = t if isinstance(t, Op) else repr(t)
    if key not in names:
        names[key] = f"X{len(names) + 1}"
    return names[key]


def _fusion_pattern_sz(
    t: Any, names: dict, memo: dict, cap: int | None
) -> tuple:
    """Spell *t*'s fusion pattern; return ``(pattern, tree_size)``.

    ``tree_size`` counts the pattern AS A TREE — a shared
    elementwise subterm is spelled once per occurrence, so the
    count follows the same recurrence ``census.term_expansion``
    measures (clamped to ``cap + 1`` when *cap* is given).  That is
    the honest bound: the spec tuples the tiers consume ARE trees —
    there is no sharing to exploit downstream, so a site whose
    spelled pattern exceeds the cap cannot be claimed usefully.
    ``(None, cap + 1)`` reports overflow; callers skip the site.

    ``memo`` dedups construction per distinct node (the pattern
    OBJECT is shared — only its tree-size is counted per
    occurrence); ``names`` keeps metavar naming consistent across
    the whole site, exactly as before.

    Iterative post-order: the spelled-tree *size* accounting is
    unchanged (still per-occurrence, still capped), but each
    distinct node is visited once — ``id``-dedup — so a deep or
    shared-DAG term cannot exhaust the recursion limit.  Children
    are pushed reversed so leaves are still named left-to-right in
    the recursive render's order, and the per-arg size accumulation
    at expansion runs in arg order, preserving the exact cap
    early-exit points.
    """
    from mlopt_core.ir import Const

    from mlopt_discovery import genkernel as gk

    def _leaf(n: Any) -> tuple:
        if isinstance(n, Const):
            return n.value, 1
        return _leaf_name(n, names), 1

    if not (isinstance(t, Op) and t.op in gk.ELEMENTWISE):
        return _leaf(t)

    result: dict[int, tuple] = {}
    done: set[int] = set()
    stack: list[tuple[Any, bool]] = [(t, False)]
    while stack:
        n, expanded = stack.pop()
        if expanded:
            hit = memo.get(id(n))
            if hit is not None:
                result[id(n)] = hit
                continue
            kids: list = []
            size = 1
            failed = False
            ks = 0
            for a in n.args:
                k, ks = result[id(a)]
                if k is None:
                    failed = True
                    break
                kids.append(k)
                size += ks
                if cap is not None and size > cap:
                    failed = True
                    break
            if failed:
                result[id(n)] = (
                    None,
                    cap + 1 if cap is not None else ks,
                )
                continue
            if n.attrs:
                kids.append(dict(n.attrs))
                size += 1
            pat = (n.op, *kids)
            out = (pat, size if cap is None else min(size, cap + 1))
            memo[id(n)] = out
            result[id(n)] = out
            continue
        if id(n) in done:
            continue
        done.add(id(n))
        if isinstance(n, Op) and n.op in gk.ELEMENTWISE:
            stack.append((n, True))
            stack.extend((a, False) for a in reversed(n.args))
        else:
            result[id(n)] = _leaf(n)
    return result[id(t)]


def _fusion_pattern(t: Any, names: dict) -> Any:
    """Return the elementwise-closure pattern at *t*, leaves abstracted.

    A subterm fuses iff its *head* is elementwise — children below
    the elementwise boundary (matmuls, reductions, …) become bound
    metavariables, consistently named by subtree identity.  So
    ``mul(y, relu(y))`` over ``y = mm(…)`` yields
    ``("mul", "X1", ("relu", "X1"))`` — ``X1`` binds the whole
    folded matmul, and the generated kernel computes it once.
    Unbounded spelling — callers on model-scale terms go through
    :func:`_fusion_pattern_sz` with a cap instead.
    """
    pat, _size = _fusion_pattern_sz(t, names, {}, None)
    return pat


def _fusion_sites(
    term: Any,
    *,
    spec_cap: int | None = None,
    stats: dict | None = None,
) -> list:
    """Elementwise-connex subterms of size ≥2, leaves abstracted.

    Each entry is a *pattern tuple* (metavar leaves for the
    non-elementwise boundary), the shape ``gen_handlers`` consumes.
    *spec_cap* is the ``ProbeBounds.spec_tree_cap`` — the pattern's
    SPELLED TREE-SIZE (op nodes + metavar/const leaves + one per
    attrs dict, the ``_fusion_pattern_sz`` accounting): sites over
    it are skipped and counted in *stats* under
    ``"fusion_sites_over_cap"`` — a claim pattern that large binds
    nothing a fusion kernel could fuse, and cannot even be spelled
    in feasible time on a shared DAG.
    """
    from mlopt_discovery import genkernel as gk

    out: list = []
    # ``traversal.ops`` is the same pre-order, DAG-deduped visit —
    # site order feeds the minted handler table.
    for t in _term_ops(term):
        if t.op not in gk.ELEMENTWISE:
            continue
        pat, size = _fusion_pattern_sz(t, {}, {}, spec_cap)
        if pat is None and spec_cap is not None:
            # cap overflow — the spelled pattern exceeded the
            # tree-size bound; an honest counted decline, never
            # a silent drop (arch audit §1.4).
            if stats is not None:
                stats["fusion_sites_over_cap"] = (
                    stats.get("fusion_sites_over_cap", 0) + 1
                )
        elif pat is not None and (spec_cap is None or size <= spec_cap):
            ops = {e for e in _flat_ops(pat)}
            if len(ops) >= 2:
                out.append(pat)
    return out


def _reduction_pattern(
    t: Any,
    names: dict,
    memo: dict | None = None,
    cap: int | None = None,
) -> tuple:
    """Spell a reduction-headed subterm as a fusion spec.

    The head and its attr dict stay concrete (``dim``/``keepdim``
    are part of the site's meaning — a claim binds that spelling,
    not every reduction of the same op); each arg is spelled by the
    pointwise tier's own boundary rule, so non-elementwise children
    abstract to metavars and the elementwise body stays spelled —
    ``mean(mul(x, relu(x)))`` → ``("mean", ("mul", "X1",
    ("relu", "X1")), {…})``.  :func:`genkernel.reduction_spec` is
    the accept gate.  Returns ``(spec, tree_size)``; ``(None,
    cap + 1)`` on cap overflow — same spelled-tree accounting as
    :func:`_fusion_pattern_sz`.
    """
    memo = {} if memo is None else memo
    kids: list = []
    size = 1
    for a in t.args:
        k, ks = _fusion_pattern_sz(a, names, memo, cap)
        if k is None:
            return None, (cap + 1 if cap is not None else ks)
        kids.append(k)
        size += ks
        if cap is not None and size > cap:
            return None, cap + 1
    if t.attrs:
        kids.append(dict(t.attrs))
        size += 1
    spec = (t.op, *kids)
    return spec, (size if cap is None else min(size, cap + 1))


def _reduction_sites(
    term: Any,
    *,
    spec_cap: int | None = None,
    stats: dict | None = None,
) -> list:
    """Reduction-bodied subterms of *term*, spec'd for the tier.

    The reduction tier's site collector — the mirror of
    ``_fusion_sites`` one level up: every ``Op`` whose head is a
    reduce-dim op (``genkernel.reduce_heads``) mints a candidate
    spec; :func:`genkernel.reduction_spec` keeps only the
    elementwise-fusable bodies (>=2 ops, non-head ops all
    elementwise).  Sites are raw-IR subterms, so minted entries
    carry no ``leaf_env`` — ``_handler_env`` rebinds them by
    concrete match like every other tier.  *spec_cap* is the
    ``ProbeBounds.spec_tree_cap`` — the spelled pattern tree-size,
    exactly as in ``_fusion_sites``; over-cap sites are counted in
    *stats* under ``"reduction_sites_over_cap"``.
    """
    from mlopt_discovery import genkernel as gk

    heads = gk.reduce_heads()
    out: list = []
    # ``traversal.ops`` — identical site order to the recursive DFS,
    # DAG-deduped (see _fusion_sites).
    for t in _term_ops(term):
        if t.op in heads:
            spec, size = _reduction_pattern(t, {}, cap=spec_cap)
            if spec is None and spec_cap is not None:
                if stats is not None:
                    stats["reduction_sites_over_cap"] = (
                        stats.get("reduction_sites_over_cap", 0) + 1
                    )
            elif (
                spec is not None
                and (spec_cap is None or size <= spec_cap)
                and gk.reduction_spec(spec)
            ):
                out.append(spec)
    return out


def _decayed_sites(term: Any) -> list:
    """``matmul(mul(s,d),v)``/``om_decayed`` subterms, spec'd.

    The decayed-tier site collector — the elementwise walk cannot
    see a contraction site, so each ``mul`` feeding a ``matmul``
    (and each already-lifted ``om_decayed``) mints a pattern whose
    operands are metavars binding whole subterms at match time —
    the kernel then receives the evaluated operand tensors.
    """
    out: list = []
    # ``traversal.ops`` — identical site order to the recursive DFS,
    # DAG-deduped (see _fusion_sites).
    for t in _term_ops(term):
        if t.op == "om_decayed":
            out.append(("om_decayed", "S", "D", "V"))
        elif (
            t.op == "matmul"
            and isinstance(t.args[0], Op)
            and t.args[0].op == "mul"
        ):
            out.append(("matmul", ("mul", "S", "D"), "V"))
    return out


def _scan_sites(term: Any) -> list:
    """``applyd``-carrier members of the saturated graph, spec'd.

    The scan tier's site source is not raw IR — carriers only exist
    once the lifts run.  This collector runs a bounded carrier-
    producing saturation (the om laws that introduce ``applyd``)
    and harvests every ``applyd`` member's decoded spec, pinned
    ``kind=unrolled`` — the machine-folded carriers' ambient
    reading, not the fold-over-steps variant the seq form declares.

    Each site is a ``{"spec": ..., "leaf_env": ...}`` record: the
    leaf env binds every spec metavar — the abstracted leaves AND
    the abstracted state — to the decoded subterm it stands for.
    The member the spec came from exists only inside this
    throwaway graph, so no raw-IR match can ever rebind it; the
    record carries the binding the referee needs.

    Sites also carry ``"chain"`` when the applyd member's e-class
    contains a raw-IR subterm of *term* — the outermost one, found
    by mapping every subterm through ``_class_of_term``.  That
    spelled subterm is the deliver-time route's LHS, the same role
    ``"chain"`` plays for the seq/seqm collectors: saturation
    proved it equal to the carrier member, so delivery may rewrite
    the extracted class member to the minted kernel call.  A class
    with no raw member (a carrier synthesized wholesale) records
    no route and delivers spelled, as before.
    """
    from kit_carriers.om import OM_CHUNKIFY, OM_LAWS
    from mlopt_core.egraph import EGraph
    from mlopt_core.laws.tensor import SPLIT_SCALAR_RULES

    from mlopt_discovery.meta_arena import _spec_of

    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    eg = EGraph()
    root = eg.add_term(term)
    eg.run(
        list(OM_LAWS) + list(OM_CHUNKIFY) + list(SPLIT_SCALAR_RULES),
        root,
        max_iterations=10,
        max_nodes=60_000,
        rule_budgets={
            "om_decayed_chunkify_rows": 20,
            "om_decayed_chunkify_keys": 20,
        },
    )
    # e-class -> outermost raw-IR subterm realising it.  The walk is
    # pre-order, so the first subterm seen per class is the largest
    # spelled member — the route's maximal LHS.
    raw_by_class: dict = {}
    cls_memo: dict = {}
    seen_raw: set = set()

    # Iterative preorder — the same first-seen-per-class order the
    # recursive DFS produced (``raw_by_class`` keeps the FIRST
    # subterm per class, so visit order is semantic), without
    # per-level frames on deep terms.
    stack = [term]
    while stack:
        t = stack.pop()
        if not isinstance(t, Op) or id(t) in seen_raw:
            continue
        seen_raw.add(id(t))
        cid = eg._class_of_term(t, cls_memo)
        if cid is not None and cid not in raw_by_class:
            raw_by_class[cid] = t
        stack.extend(reversed(t.args))
    out: list = []
    carrier_ops = ("aff_diag", "affd_compose")
    for cls in eg._classes.values():
        for n in cls.nodes:
            if n.op != "applyd":
                continue
            member = _carrier_member(eg, n, carrier_ops)
            if member is None:
                continue
            f_term = member.args[0]
            if not isinstance(f_term, Op):
                continue
            names: dict = {}
            f_spec = _spec_of(f_term, names)
            leaf_env = _spec_leaf_env(f_term, names)
            # the state abstracts to a metavar — the handler's
            # pattern is ``applyd(<map>, X_h)`` over ANY state;
            # whatever the fold produced there is recorded too
            state = f"X{len(names) + 1}"
            leaf_env[state] = member.args[1]
            spec = ("applyd", f_spec, state, {"kind": "unrolled"})
            rec: dict = {"spec": spec, "leaf_env": leaf_env}
            chain = raw_by_class.get(eg.find(cls.id))
            if chain is not None:
                rec["chain"] = chain
            # dedup on the WHOLE record, not the spec tuple alone:
            # two carrier members can share one spec skeleton while
            # ``leaf_env`` binds different operands — distinct folds
            # the referee must measure and distinct chains the
            # delivery route must rewrite.
            if rec in out:
                continue
            out.append(rec)
    return out


def _spec_leaf_env(term: Any, names: dict) -> dict:
    """Bind each metavar ``_spec_of`` minted to its leaf subterm.

    The inverse of leaf abstraction: every non-``Op``/non-``Const``
    leaf maps to ``names[repr(leaf)]`` — the env a spec harvested
    post-saturation carries so its metavars evaluate without a
    match (``eval_term`` on the term, the literal itself on a raw
    scalar leaf like a ``pow`` exponent).
    """
    env: dict = {}
    # Iterative id-deduped walk: the env maps each metavar name to
    # its leaf — revisiting a shared leaf rewrites the same key with
    # the same value, so dedup cannot change the result and keeps
    # the walk DAG-linear on deep/shared terms.
    seen: set[int] = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        if isinstance(t, Op):
            stack.extend(t.args)
        elif not isinstance(t, Const):
            env[names[repr(t)]] = t
    return env


def _carrier_member(eg: Any, node: Any, carrier_ops: tuple) -> Any:
    """Decode an ``applyd`` enode preferring carrier-shaped children.

    ``any_term`` prefers leaves — useless for the map slot, which
    must surface as ``aff_diag``/``affd_compose``.  Children decode
    to the carrier op member when their class carries one, else the
    class rep.
    """
    args = []
    for c in node.children:
        cls = eg._classes[eg.find(c)]
        pick = next((m for m in cls.nodes if m.op in carrier_ops), None)
        if pick is not None:
            args.append(_carrier_member(eg, pick, carrier_ops))
        else:
            args.append(eg.any_term(eg.find(c)))
    if any(a is None for a in args):
        return None
    return Op.make(node.op, *args, **dict(node.attrs))


# ---------------------------------------------------------------------------
#  Seq-form scan sites — raw recurrence chains, not saturated carriers
# ---------------------------------------------------------------------------

#: Cap on tied spine readings kept per node and seq sites emitted per
#: term — ambiguity is rare; the caps bound the collector's work on
#: pathological terms without dropping the common case.
_SEQ_SPINE_CAP = 4
_SEQ_SITE_CAP = 16


def _seq_step_cands(t: Any) -> tuple:
    """``(map, state, inp)`` readings of one ``add(mul(·,·),·)`` step.

    The diagonal recurrence ``h' = a ⊙ h + x`` arrives unrolled as
    ``add(mul(a, h), x)`` — either add operand may hold the product
    and either mul factor may be the state, so a node yields up to
    four candidate decompositions.  The spine walk keeps only the
    longest readings; :func:`_seq_site` arbitrates the rest.  The
    ``a * h`` spelling (map first, state second) is tried first so
    the common torch order wins ties deterministically.
    """
    if not (
        isinstance(t, Op)
        and t.op == "add"
        and len(t.args) == 2
        and not t.attrs
    ):
        return ()
    c0, c1 = t.args
    out = []
    for prod, inn in ((c0, c1), (c1, c0)):
        if (
            isinstance(prod, Op)
            and prod.op == "mul"
            and len(prod.args) == 2
            and not prod.attrs
        ):
            f0, f1 = prod.args
            out.append((f0, f1, inn))
            out.append((f1, f0, inn))
    # Matrix-state form: ``h' = h + f(h)`` — the delta-rule spelling
    # ``s = s + beta·k⊗(v - kᵀs)`` puts the state as a DIRECT add
    # operand, never a ``mul`` factor, so the product readings above
    # cannot see it.  ``None`` marks the implicit unit map; the
    # seq-matrix collector (:func:`_seqm_site`) is the only consumer
    # that accepts it — :func:`_seq_site` declines ``None`` maps.
    out.append((None, c0, c1))
    out.append((None, c1, c0))
    return tuple(out)


def _seq_spines(t: Any, memo: dict) -> list:
    """Decomposition alternatives of *t*, longest first.

    Each entry is ``(steps, base)`` — *steps* a tuple of
    ``(map, inp, state)`` triples in CHRONOLOGICAL order (the
    outermost ``add`` is the last step, so the walk prepends inward)
    and *base* the initial-state term the fold starts from.  A node
    with no step reading is its own base; tied longest readings are
    all kept (up to :data:`_SEQ_SPINE_CAP`) so the seq-ification
    pass can arbitrate readings the spine length cannot.
    """
    key = id(t)
    if key in memo:
        return memo[key]
    cands = _seq_step_cands(t)
    results = []
    for m, s, inn in cands:
        for sub_steps, sub_base in _seq_spines(s, memo):
            results.append(((*sub_steps, (m, inn, s)), sub_base))
    if results:
        top = max(len(r[0]) for r in results)
        memo[key] = [r for r in results if len(r[0]) == top][
            :_SEQ_SPINE_CAP
        ]
    else:
        memo[key] = [((), t)]
    return memo[key]


class _SeqSpecOverflow(Exception):
    """A seq-site verification exceeded its spelled-size bound.

    Internal control flow only — the collectors catch it per
    candidate and count the site as an honest over-cap skip; it is
    never part of a public contract.
    """


def _seq_expansion(t: Any, cap: int) -> int:
    """Tree-expansion of *t* with early exit at ``cap + 1``.

    Same recurrence as ``census.term_expansion`` — every occurrence
    counts, so a shared-DAG child's true expansion is astronomic —
    but the walk aborts the moment any node's partial sum exceeds
    *cap*: ancestors can only add more, so the root's expansion is
    already known to exceed the bound and the exact value is never
    needed (callers only compare against *cap*).  ``shape_of``
    itself walks this same tree expansion, so this is also the cheap
    oracle for "would that shape check hang".
    """
    memo: dict[int, int] = {}
    stack: list = [(t, False)]
    while stack:
        n, expanded = stack.pop()
        if not isinstance(n, Op):
            memo[id(n)] = 1
            continue
        if id(n) in memo:
            continue
        if not expanded:
            stack.append((n, True))
            for a in n.args:
                if id(a) not in memo:
                    stack.append((a, False))
            continue
        total = 1
        for a in n.args:
            total += memo[id(a)]
            if total > cap:
                return cap + 1
        memo[id(n)] = total
    return memo[id(t)]


def _seq_shape(t: Any, cap: int | None):
    """``shape_of(t)`` guarded by the seq work bound.

    ``shape_of`` walks the term's *tree* expansion — unbounded on a
    shared DAG — so under *cap* a child whose :func:`_seq_expansion`
    exceeds it reports ``INVALID`` exactly like a failed shape
    inference: the lift or site simply does not fire.  ``None`` cap
    keeps the historical unguarded call.
    """
    from mlopt_core.typing import INVALID, shape_of

    if cap is not None and _seq_expansion(t, cap) > cap:
        return INVALID
    return shape_of(t)


def _seq_view_lift(t: Any, cap: int | None = None):
    """Lift a view-op chain over a ``select`` onto the select's child.

    ``unsqueeze``/``squeeze`` directly wrapping a per-step ``select``
    (possibly nested) commute onto the sequence tensor: the view is
    re-expressed on the WHOLE child with its ``dim`` re-based into
    child coordinates, and the select's dim shifted to match.  The
    result is ``(viewed_child, select_dim, index)`` — feed it through
    the ordinary operand path so the metavar binds the viewed
    sequence (the step axis's position in *viewed-child* coords).

    Why lift instead of rendering the view in the step expr: the
    seq fold's state space is the operand broadcast minus the step
    axis.  A view INSIDE the step expr (``unsqueeze`` of a ``(b,d)``
    slice against a ``(b,d,d)`` state) has no flat-space meaning —
    but the same view lifted onto the sequence child grows the
    operand's own shape so the broadcast *includes* the state dims,
    and every downstream consumer (``shape_of`` checks,
    ``seq_scan_denotation``, the Triton kernel's materialised rows)
    is unchanged.  ``unsqueeze(gg,-1)`` on a ``(b,T,d)`` operand
    becomes ``unsqueeze(gg,3)`` selected at dim 1 — the fold then
    reads ``(b,d,d)`` step slices, exactly the recurrence's state.

    Returns ``None`` — the caller falls back to the generic spec
    path, which keeps the view node in the skeleton and lets
    ``scan_spec`` decline honestly — when the chain does not end at
    a ``select``, a ``dim`` attr is missing/non-int, ``shape_of``
    cannot type a child, or a normalised dim falls out of range.
    Other view ops (``reshape``/``view``/``expand``/``permute``)
    and dim-less ``squeeze`` stay outside the grammar.  *cap* is the
    seq work bound — ``shape_of`` checks run through
    :func:`_seq_shape`, which declines lifts whose operand expansion
    exceeds it.
    """
    from mlopt_core.typing import INVALID

    views: list = []
    node = t
    while (
        isinstance(node, Op)
        and node.op in ("unsqueeze", "squeeze")
        and len(node.args) == 1
        and set(node.attrs) <= {"dim"}
        and type(node.attrs.get("dim")) is int
    ):
        views.append(node)
        node = node.args[0]
    if (
        not views
        or not isinstance(node, Op)
        or node.op != "select"
        or len(node.args) != 1
        or type(node.attrs.get("dim")) is not int
        or type(node.attrs.get("index")) is not int
    ):
        return None
    child = node.args[0]
    idx = node.attrs["index"]
    sh = _seq_shape(child, cap)
    if sh is INVALID or not isinstance(sh, tuple):
        return None
    rank = len(sh)
    d = node.attrs["dim"]
    d0 = d if d >= 0 else d + rank
    if not 0 <= d0 < rank:
        return None
    # Fold the views inside-out (``views`` is outermost-first).  At
    # each step the invariant is ``select(child, d0, idx)`` == the
    # tensor the remaining inner views produce; *d0* is the step
    # axis in current-child coords, *rank-1* the sliced rank the
    # view's ``dim`` normalises against.
    for v in reversed(views):
        vd = v.attrs["dim"]
        sh = _seq_shape(child, cap)
        if sh is INVALID or not isinstance(sh, tuple):
            return None
        r = len(sh) - 1  # rank of the post-select tensor
        if v.op == "unsqueeze":
            # insertion position in sliced coords, 0..r
            p = vd if vd >= 0 else vd + r + 1
            if not 0 <= p <= r:
                return None
            # child-coord insertion: before the step axis when the
            # slice position precedes it, else after — either side
            # of the boundary yields the same result axis order, and
            # this choice keeps the step axis index stable (``dims``
            # stays a singleton across operands).
            q = p if p < d0 else p + 1
            child = Op.make("unsqueeze", child, dim=q)
            if q <= d0:
                d0 += 1
        else:  # squeeze — removes the sliced axis at position k
            k = vd if vd >= 0 else vd + r
            if not 0 <= k < r:
                return None
            j = k if k < d0 else k + 1  # the same axis in child coords
            child = Op.make("squeeze", child, dim=j)
            if j < d0:
                d0 -= 1
    sh = _seq_shape(child, cap)
    if (
        sh is INVALID
        or not isinstance(sh, tuple)
        or not 0 <= d0 < len(sh)
    ):
        return None
    return child, d0, idx


def _seq_meta(
    names: dict,
    operands: dict,
    events: list,
    child: Any,
    dim: int,
    idx: int,
) -> str:
    """Mint/reuse the sequence-operand metavar for ``(child, dim)``."""
    key = (child, dim)
    meta = names.get(key)
    if meta is None:
        meta = f"Q{len(names) + 1}"
        names[key] = meta
        operands[meta] = (child, dim)
    events.append((meta, idx))
    return meta


def _seq_const_meta(consts: dict, t: Any) -> str:
    """Mint/reuse the loop-invariant metavar for leaf *t*.

    *consts* maps the leaf term to its ``"P"``-prefixed meta — equal
    leaves (``Param``/``Var`` are value-equality dataclasses) share
    one meta, so a weight used at every step binds once.
    """
    meta = consts.get(t)
    if meta is None:
        meta = f"P{len(consts) + 1}"
        consts[t] = meta
    return meta


def _seq_select_free(t: Any) -> bool:
    """Whether *t* contains no ``select`` node anywhere.

    Iterative over the interned DAG — the lifted projection operand
    is evaluated ONCE by ``leaf_env``, so a ``select`` surviving
    inside it would freeze one step's slice into every step.  Fixed
    -index selects are technically loop-invariant, but rejecting
    them keeps the lift's contract simple: the operand child must
    be a pure sequence tensor.
    """
    return not any(n.op == "select" for n in _term_ops(t))


def _seq_step_select(t: Any):
    """``(child, dim, index)`` when *t* is a bare per-step ``select``."""
    if (
        isinstance(t, Op)
        and t.op == "select"
        and len(t.args) == 1
        and type(t.attrs.get("dim")) is int
        and type(t.attrs.get("index")) is int
    ):
        return t.args[0], t.attrs["dim"], t.attrs["index"]
    return None


def _seq_projection_lift(t: Any, cap: int | None = None):
    """Pull a per-step ``select`` out through a projection op.

    ``matmul``/``linear`` over a ``select``ed sequence commutes with
    the slice whenever the selected axis is not contracted:
    ``matmul(select(x, d, i), w)`` on a non-``k`` axis *d* equals
    ``select(matmul(x, w), d', i)`` — the projection hoists onto the
    WHOLE sequence tensor and the fold slices the projected rows,
    exactly the spelling a hand-hoisted recurrence already uses.
    Returns ``(lifted_term, out_dim, index)`` — feed through the
    ordinary operand path — or ``None`` to decline.

    Accepted shapes (verified against ``shape_of``, torch matmul /
    ``F.linear`` semantics):

    * ``matmul(a, b)`` with the select a DIRECT left arg: *b* must
      have rank ≤ 2 (no batch dims to misalign) and the select dim
      must not be the contracted last axis of *a*; batch and ``m``
      dims keep their position in the output, so ``out_dim = d0``.
    * ``matmul(a, b)`` with the select a DIRECT right arg: *b* must
      be a rank-2 ``(k, n)`` matrix and the select must hit the
      free ``n`` axis; the output's last axis is ``n``.
    * ``linear(x, w[, b])`` with the select a DIRECT first arg:
      ``F.linear`` contracts only the last axis of *x*, so any
      leading dim lifts with ``out_dim = d0``.

    The lifted term must be ``select``-free — a surviving ``select``
    would freeze one step's slice into the once-evaluated operand
    (see :func:`_seq_select_free`).  Selects nested deeper inside an
    argument, selects in BOTH matmul args, and every other op keep
    the generic decline.  *cap* is the seq work bound (see
    :func:`_seq_shape`).
    """
    from mlopt_core.typing import INVALID

    if not (isinstance(t, Op) and not t.attrs):
        return None
    if t.op == "matmul" and len(t.args) == 2:
        a, b = t.args
        sa = _seq_step_select(a)
        sb = _seq_step_select(b)
        if (sa is None) == (sb is None):
            return None  # zero or two direct selects — decline
        if sa is not None:
            c, d, idx = sa
            other, pos = b, 0
        else:
            c, d, idx = sb
            other, pos = a, 1
        sh = _seq_shape(c, cap)
        so = _seq_shape(other, cap)
        if (
            sh is INVALID
            or so is INVALID
            or not isinstance(sh, tuple)
            or not isinstance(so, tuple)
        ):
            return None
        ra, rb = len(sh), len(so)
        d0 = d if d >= 0 else d + ra
        if not 0 <= d0 < ra:
            return None
        if pos == 0:
            # left operand: last axis is the contracted ``k``; every
            # other axis keeps its position in the (…, m, n) output
            # provided *b* brings no batch dims of its own.
            if rb > 2 or d0 == ra - 1:
                return None
            out_dim = d0
        else:
            # right operand: only a rank-2 ``(k, n)`` right tensor
            # has a free axis — ``n``, its last dim — and it lands
            # last in the output.  ``ra`` is the SELECT CHILD's rank
            # (the lifted right operand), ``rb`` the left arg's.
            if ra != 2 or d0 != ra - 1:
                return None
            out_dim = (rb if rb >= 2 else 1) - 1
        lifted = (
            Op.make("matmul", c, other)
            if pos == 0
            else Op.make("matmul", other, c)
        )
    elif t.op == "linear" and len(t.args) in (2, 3):
        s = _seq_step_select(t.args[0])
        if s is None:
            return None
        c, d, idx = s
        sh = _seq_shape(c, cap)
        if sh is INVALID or not isinstance(sh, tuple):
            return None
        ra = len(sh)
        d0 = d if d >= 0 else d + ra
        if not 0 <= d0 < ra - 1:
            return None  # the contracted in-features axis
        out_dim = d0
        lifted = Op.make("linear", c, *t.args[1:])
    else:
        return None
    if cap is not None and _seq_expansion(lifted, cap) > cap:
        return None
    if not _seq_select_free(lifted):
        return None
    sh2 = _seq_shape(lifted, cap)
    if (
        sh2 is INVALID
        or not isinstance(sh2, tuple)
        or not 0 <= out_dim < len(sh2)
    ):
        return None
    return lifted, out_dim, idx


def _seq_shape_first(_op: Any, shapes: list) -> Any:
    """Shape rule for ops that return their first operand's shape."""
    return shapes[0] if shapes else None


# ``shape_of`` rules the seq commute-lift relies on.  These ops
# provably preserve their input's shape, but the substrate's rule
# table (``mlopt_core.typing``, owned by the eqsat repo) does not
# list them yet — ``register_shape_rule`` is the documented
# extension point and the rule is semantically exact, not a patch:
# ``layer_norm``/``rms_norm``/``clone`` always return ``x.shape``.
for _norm_op in ("layer_norm", "rms_norm", "clone"):
    _register_shape_rule(_norm_op, _seq_shape_first)
del _norm_op


def _seq_axis_idx(d: Any, rank: int):
    """Normalise an int axis into ``[0, rank)``; ``None`` out of range."""
    if type(d) is not int:
        return None
    d0 = d if d >= 0 else d + rank
    return d0 if 0 <= d0 < rank else None


def _seq_commute_lift(t: Any, cap: int | None = None):
    """Pull a per-step ``select`` out through an axis-safe op.

    The projection lift (:func:`_seq_projection_lift`) covers
    ``matmul``/``linear``; this is the same hoist for every other op
    whose denotation commutes with the step-axis slice —
    ``op(select(x, d, i), …) == select(op(x, …), D, i)`` — so the
    operand binds the WHOLE-tensor ``op(x, …)`` evaluated once and
    the fold slices it per step.  Each case below states the law it
    relies on; anything not listed, any select that is not a DIRECT
    argument, and any ``argN`` positional attr spelling (which would
    alias the dims being remapped) keeps the generic decline.

    * ``transpose``/``permute`` — pure axis permutations: the select
      axis is re-based through the permutation and re-inserted last.
    * ``softmax``/``log_softmax`` — normalise along ONE axis; they
      commute iff that axis is not the step axis.
    * ``layer_norm``/``rms_norm`` — normalise over the LAST
      ``len(normalized_shape)`` axes; they commute iff the step axis
      sits strictly before that trailing block (and the block's
      sizes match the child's trailing dims).
    * ``sum``/``mean``/``amax``/``amin`` — reductions commute iff no
      reduced axis is the step axis; ``keepdim`` decides whether
      the output axis shifts.
    * ``slice``/``narrow``/``chunk``/``split`` — range restrictions
      along one axis commute iff that axis is not the step axis.
    * ``unbind`` — removes one axis; commutes iff it is not the step
      axis (the output axis count drops by one).
    * ``concat`` — EVERY argument must be a bare ``select`` on the
      SAME child axis with the SAME index (the multi-arg analogue of
      the both-args boundary); the lifted concat joins the children
      along the remapped concat axis.
    * ``to``/``clone``/``contiguous``/``alias``/``dropout``/
      ``broadcast``/``float``/``type_as`` — axis-preserving
      passthroughs; they commute unconditionally.

    Returns ``(lifted_term, out_dim, index)`` — feed through the
    ordinary operand path — or ``None`` to decline.  The lifted term
    must be ``select``-free (:func:`_seq_select_free`) and type via
    ``shape_of`` exactly like the projection lift's output.  *cap* is
    the seq work bound (see :func:`_seq_shape`).
    """
    from mlopt_core.typing import INVALID

    if not isinstance(t, Op) or not t.args:
        return None
    if t.op == "concat":
        return _seq_concat_lift(t, cap)
    s = _seq_step_select(t.args[0])
    if s is None:
        return None
    c, d, idx = s
    sh = _seq_shape(c, cap)
    if sh is INVALID or not isinstance(sh, tuple) or not sh:
        return None
    r = len(sh)
    d0 = _seq_axis_idx(d, r)
    if d0 is None:
        return None
    rs = r - 1  # rank of the per-step (post-select) tensor

    def up(j: int) -> int:
        """Sliced-tensor axis -> child-tensor axis (skips ``d0``)."""
        return j if j < d0 else j + 1

    ctx = _SeqLiftCtx(t, c, d0, idx, rs, r, sh, up)
    if t.op in _SEQ_PASSTHROUGH_OPS:
        # axis-preserving passthroughs need no dim bookkeeping
        return _seq_lift_finish(
            (Op.make(t.op, c, *t.args[1:], **t.attrs), ctx.d0),
            ctx,
            cap,
        )
    # Every remaining case remaps an axis attr — positional ``argN``
    # spellings would alias those dims in child coordinates, so they
    # decline rather than lift under the wrong axis.
    if _seq_argn_attrs(t.attrs):
        return None
    fn = _SEQ_AXIS_LIFTS.get(t.op)
    if fn is None:
        return None
    return _seq_lift_finish(fn(ctx), ctx, cap)


@dataclass(frozen=True)
class _SeqLiftCtx:
    """Shared context for the commute-lift op handlers.

    ``c`` is the select's child (the whole sequence tensor), ``d0``
    its normalised step axis, ``idx`` the step index, ``rs``/``r``
    the sliced/child ranks, ``sh`` the child's ``shape_of`` and
    ``up`` the sliced-axis -> child-axis map (skips ``d0``).
    """

    t: Any
    c: Any
    d0: int
    idx: int
    rs: int
    r: int
    sh: tuple
    up: Any


def _seq_argn_attrs(attrs: dict) -> bool:
    """Whether *attrs* carries a positional ``argN`` spelling."""
    return any(
        isinstance(k, str) and k.startswith("arg") and k[3:].isdigit()
        for k in attrs
    )


def _seq_lift_finish(
    res: Any, ctx: _SeqLiftCtx, cap: int | None = None
):
    """Shared tail: build ``(lifted, out_dim, idx)`` or decline.

    *res* is the op handler's ``(lifted_term, out_dim)`` pair (or
    ``None``).  The lifted term must be ``select``-free — a surviving
    ``select`` would freeze one step's slice into the once-evaluated
    operand — and ``shape_of`` must type it with ``out_dim`` in
    range, the same contract the projection lift keeps.  *cap* is
    the seq work bound (see :func:`_seq_shape`).
    """
    from mlopt_core.typing import INVALID

    if res is None:
        return None
    lifted, out_dim = res
    if cap is not None and _seq_expansion(lifted, cap) > cap:
        return None
    if not _seq_select_free(lifted):
        return None
    sh2 = _seq_shape(lifted, cap)
    if (
        sh2 is INVALID
        or not isinstance(sh2, tuple)
        or not 0 <= out_dim < len(sh2)
    ):
        return None
    return lifted, out_dim, ctx.idx


def _seq_lift_attrs(
    ctx: _SeqLiftCtx, drop: tuple, **set_attrs: Any
) -> dict:
    """Copy *ctx.t*'s attrs minus *drop*, plus *set_attrs*."""
    out = {k: v for k, v in ctx.t.attrs.items() if k not in drop}
    out.update(set_attrs)
    return out


def _seq_lift_transpose(ctx: _SeqLiftCtx):
    """``transpose`` — swap two axes; the select axis keeps position."""
    a = _seq_axis_idx(ctx.t.attrs.get("dim0", -2), ctx.rs)
    b = _seq_axis_idx(ctx.t.attrs.get("dim1", -1), ctx.rs)
    if a is None or b is None:
        return None
    lifted = Op.make(
        "transpose",
        ctx.c,
        **_seq_lift_attrs(
            ctx, ("dim0", "dim1"), dim0=ctx.up(a), dim1=ctx.up(b)
        ),
    )
    return lifted, ctx.d0


def _seq_lift_permute(ctx: _SeqLiftCtx):
    """``permute`` — rebuild the permutation with the step axis last."""
    dims = ctx.t.attrs.get("dim", ctx.t.attrs.get("dims"))
    if not (
        isinstance(dims, (tuple, list))
        and len(dims) == ctx.rs
        and all(type(v) is int for v in dims)
    ):
        return None
    ps = [_seq_axis_idx(v, ctx.rs) for v in dims]
    if any(v is None for v in ps) or sorted(ps) != list(range(ctx.rs)):
        return None
    lifted = Op.make(
        "permute",
        ctx.c,
        **_seq_lift_attrs(
            ctx,
            ("dim", "dims"),
            dim=(*tuple(ctx.up(v) for v in ps), ctx.d0),
        ),
    )
    return lifted, ctx.r - 1


def _seq_lift_softmax(ctx: _SeqLiftCtx):
    """``softmax``/``log_softmax`` — mixes only along ``dim``."""
    k = _seq_axis_idx(ctx.t.attrs.get("dim", -1), ctx.rs)
    if k is None:
        return None
    kc = ctx.up(k)
    if kc == ctx.d0:
        return None
    lifted = Op.make(
        ctx.t.op, ctx.c, **_seq_lift_attrs(ctx, (), dim=kc)
    )
    return lifted, ctx.d0


def _seq_lift_norm(ctx: _SeqLiftCtx):
    """``layer_norm``/``rms_norm`` — normalise the trailing block."""
    ns = ctx.t.attrs.get("dim")
    if not (
        isinstance(ns, tuple) and ns and all(type(v) is int for v in ns)
    ):
        return None
    ln = len(ns)
    if ln > ctx.rs or ctx.d0 >= ctx.r - ln:
        return None
    if tuple(ns) != tuple(ctx.sh[ctx.r - ln :]):
        return None
    lifted = Op.make(ctx.t.op, ctx.c, *ctx.t.args[1:], **ctx.t.attrs)
    return lifted, ctx.d0


def _seq_reduce_dims(ctx: _SeqLiftCtx) -> list | None:
    """Normalised reduced-axis list in sliced coords, or ``None``."""
    dim = ctx.t.attrs.get("dim")
    dims = dim if isinstance(dim, (tuple, list)) else (dim,)
    if not dims or any(type(v) is not int for v in dims):
        return None
    rd = []
    for v in dims:
        j = _seq_axis_idx(v, ctx.rs)
        if j is None or j in rd:
            return None
        rd.append(j)
    return rd


def _seq_lift_reduce(ctx: _SeqLiftCtx):
    """``sum``/``mean``/``amax``/``amin`` — reduce non-step axes."""
    rd = _seq_reduce_dims(ctx)
    if rd is None:
        return None
    rc = {ctx.up(j) for j in rd}
    if ctx.d0 in rc:
        return None
    lifted = Op.make(
        ctx.t.op,
        ctx.c,
        **_seq_lift_attrs(ctx, (), dim=tuple(ctx.up(j) for j in rd)),
    )
    if ctx.t.attrs.get("keepdim", False):
        return lifted, ctx.d0
    return lifted, ctx.d0 - sum(1 for x in rc if x < ctx.d0)


def _seq_lift_range(ctx: _SeqLiftCtx):
    """``slice``/``narrow``/``chunk``/``split`` — range one axis."""
    kd = ctx.t.attrs.get("dim", 0 if ctx.t.op == "slice" else -1)
    k = _seq_axis_idx(kd, ctx.rs)
    if k is None:
        return None
    kc = ctx.up(k)
    if kc == ctx.d0:
        return None
    lifted = Op.make(
        ctx.t.op, ctx.c, **_seq_lift_attrs(ctx, (), dim=kc)
    )
    return lifted, ctx.d0


def _seq_lift_unbind(ctx: _SeqLiftCtx):
    """``unbind`` — removes one non-step axis; positions shift."""
    k = _seq_axis_idx(ctx.t.attrs.get("dim", 0), ctx.rs)
    if k is None:
        return None
    kc = ctx.up(k)
    if kc == ctx.d0:
        return None
    lifted = Op.make(
        ctx.t.op, ctx.c, **_seq_lift_attrs(ctx, (), dim=kc)
    )
    return lifted, ctx.d0 if ctx.d0 < kc else ctx.d0 - 1


#: Op -> commute-lift handler for the axis-remapping families.
_SEQ_AXIS_LIFTS = {
    "transpose": _seq_lift_transpose,
    "permute": _seq_lift_permute,
    "softmax": _seq_lift_softmax,
    "log_softmax": _seq_lift_softmax,
    "layer_norm": _seq_lift_norm,
    "rms_norm": _seq_lift_norm,
    "sum": _seq_lift_reduce,
    "mean": _seq_lift_reduce,
    "amax": _seq_lift_reduce,
    "amin": _seq_lift_reduce,
    "slice": _seq_lift_range,
    "narrow": _seq_lift_range,
    "chunk": _seq_lift_range,
    "split": _seq_lift_range,
    "unbind": _seq_lift_unbind,
}


#: Ops that preserve every axis position exactly — the commute lift
#: needs no dim bookkeeping for them.
_SEQ_PASSTHROUGH_OPS = frozenset(
    {
        "to",
        "clone",
        "contiguous",
        "alias",
        "dropout",
        "broadcast",
        "float",
        "type_as",
    }
)


def _seq_concat_lift(t: Any, cap: int | None = None):
    """Lift ``concat`` when EVERY arg is a per-step ``select``.

    ``concat([select(x,d,i), …], dim=c)`` with all selects on the
    same child axis ``d`` (never the joined axis) and the same index
    ``i`` equals ``select(concat([x,…], dim=c'), d, i)`` — the
    children are joined along the remapped concat axis and the fold
    slices the joined sequence.  Mixed select axes, mixed indices,
    a select ON the joined axis, or any non-select argument all
    decline: a single outer ``select`` cannot reproduce them.
    *cap* is the seq work bound (see :func:`_seq_shape`).
    """
    from mlopt_core.typing import INVALID

    if len(t.args) < 2 or any(
        isinstance(k, str) and k.startswith("arg") and k[3:].isdigit()
        for k in t.attrs
    ):
        return None
    sels = [_seq_step_select(a) for a in t.args]
    if any(s is None for s in sels):
        return None
    children = []
    d0s = []
    idxs = []
    shapes = []
    r = None
    for cc, dd, ii in sels:
        sh = _seq_shape(cc, cap)
        if sh is INVALID or not isinstance(sh, tuple) or not sh:
            return None
        rr = len(sh)
        if r is None:
            r = rr
        elif rr != r:
            return None
        d0 = _seq_axis_idx(dd, rr)
        if d0 is None:
            return None
        children.append(cc)
        d0s.append(d0)
        idxs.append(ii)
        shapes.append(sh)
    if r is None or len(set(d0s)) != 1 or len(set(idxs)) != 1:
        return None
    d0 = d0s[0]
    idx = idxs[0]
    # concat dim is spelled in the rank-(r-1) argument/output coords
    cd_raw = t.attrs.get("dim", -1)
    cd = _seq_axis_idx(cd_raw, r - 1)
    if cd is None:
        return None
    # joined axis in child coords — identical for every arg since
    # all selects sit on the same child axis ``d0``
    jc = cd if cd < d0 else cd + 1
    # the lifted concat must itself be well-formed: every non-joined
    # child axis (including the shared step axis) must agree
    for k in range(r):
        if k == jc:
            continue
        if len({sh[k] for sh in shapes}) != 1:
            return None
    lifted = Op.make(
        "concat",
        *children,
        dim=jc,
        **{k2: v for k2, v in t.attrs.items() if k2 != "dim"},
    )
    if cap is not None and _seq_expansion(lifted, cap) > cap:
        return None
    if not _seq_select_free(lifted):
        return None
    sh2 = _seq_shape(lifted, cap)
    if (
        sh2 is INVALID
        or not isinstance(sh2, tuple)
        or not 0 <= d0 < len(sh2)
    ):
        return None
    return lifted, d0, idx


def _seq_bcast(sh1: tuple, sh2: tuple):
    """Right-aligned broadcast of two shapes; ``None`` on conflict."""
    if len(sh1) < len(sh2):
        sh1 = (1,) * (len(sh2) - len(sh1)) + sh1
    elif len(sh2) < len(sh1):
        sh2 = (1,) * (len(sh1) - len(sh2)) + sh2
    out = []
    for x, y in zip(sh1, sh2, strict=True):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            return None
    return tuple(out)


def _seq_deindex(
    t: Any,
    names: dict,
    operands: dict,
    events: list,
    state: Any = None,
    consts: dict | None = None,
    pullout: bool = False,
    spec_cap: int | None = None,
    _budget: list | None = None,
):
    """Spec of *t* with per-step ``select``s abstracted to metas.

    Every ``select(child, dim=D, index=i)`` node becomes a sequence
    operand metavar bound to *child* — recorded in *operands* as
    ``meta -> (child, dim)`` and in *events* as ``(meta, index)`` so
    the caller can verify the index walks the step axis in step
    order.  ``rsub(x, s)`` normalises to ``sub(s, x)`` — the
    elementwise spelling the scan renderer knows.  Any other bare
    leaf or non-scalar ``Const`` declines the whole term with
    ``None`` — EXCEPT ``Param``/``Var`` leaves when *consts* is
    given: they are loop-invariant, so they mint ``"P"`` metas
    bound whole (recorded in *consts*, never in *operands*/*events*
    — the fold reads the same value every step).

    ``unsqueeze``/``squeeze`` chains wrapping a ``select`` are
    lifted onto the child first (:func:`_seq_view_lift`) — the
    metavar then binds the *viewed* sequence and no view op reaches
    the step-expr skeleton.  With *pullout*, a ``select`` sitting
    DIRECTLY inside ``matmul``/``linear`` is pulled through the
    projection (:func:`_seq_projection_lift`) and one inside any
    other axis-safe op through the commutation lift
    (:func:`_seq_commute_lift`) — the operand binds the hoisted
    whole-tensor op result, never a per-step slice.

    *state* (matrix-state form only): the exact subterm the fold's
    accumulator denotes at this step — an occurrence of it inside
    the step expression abstracts to the ``"H"`` metavar instead of
    recursing into the previous chain link.  The check is *equality*,
    not identity: ``Op.make`` hash-conses ops, but ``Var``/``Param``
    leaves are plain frozen dataclasses — a second, structurally
    equal ``Var("H0", …)`` minted by a sibling recurrence interns the
    SAME ``matmul`` node while the spine's state factor stays the
    other object, so ``t is state`` silently misses (measured: a
    gated recurrence built first made the plain one's site decline).
    Structural equality is also the *sound* reading — an equal term
    denotes the same value the accumulator holds at this step.

    *spec_cap*/*_budget* are the seq work bound: the walk spells the
    term as a TREE (no memo — a shared subterm is spelled once per
    occurrence, exactly like the spec it builds), so on a shared-DAG
    operand the visit count IS the spelled size and grows
    exponentially in the distinct-node count.  ``_budget`` is the
    caller's remaining-visit counter (``[spec_cap]`` per top-level
    call); exhausting it raises :class:`_SeqSpecOverflow`, which the
    site verifiers turn into an honest over-cap decline.  *spec_cap*
    additionally bounds every ``shape_of`` check the pullout lifts
    run (see :func:`_seq_shape`).  Both default to the historical
    unbounded zoo behaviour.
    """
    if _budget is not None:
        _budget[0] -= 1
        if _budget[0] < 0:
            raise _SeqSpecOverflow
    if state is not None and t == state:
        return "H"
    if isinstance(t, Op):
        if t.op in ("unsqueeze", "squeeze") and len(t.args) == 1:
            lifted = _seq_view_lift(t, spec_cap)
            if lifted is not None:
                child, dim, idx = lifted
                return _seq_meta(
                    names, operands, events, child, dim, idx
                )
        if (
            t.op == "select"
            and len(t.args) == 1
            and type(t.attrs.get("dim")) is int
            and type(t.attrs.get("index")) is int
        ):
            child, dim, idx = (
                t.args[0],
                t.attrs["dim"],
                t.attrs["index"],
            )
            return _seq_meta(names, operands, events, child, dim, idx)
        if t.op == "rsub" and len(t.args) == 2 and not t.attrs:
            b = _seq_deindex(
                t.args[1],
                names,
                operands,
                events,
                state=state,
                consts=consts,
                pullout=pullout,
                spec_cap=spec_cap,
                _budget=_budget,
            )
            a = _seq_deindex(
                t.args[0],
                names,
                operands,
                events,
                state=state,
                consts=consts,
                pullout=pullout,
                spec_cap=spec_cap,
                _budget=_budget,
            )
            if a is None or b is None:
                return None
            return ("sub", b, a)
        if pullout:
            proj = _seq_projection_lift(t, spec_cap)
            if proj is None:
                proj = _seq_commute_lift(t, spec_cap)
            if proj is not None:
                child, dim, idx = proj
                return _seq_meta(
                    names, operands, events, child, dim, idx
                )
        parts = []
        for a_ in t.args:
            p = _seq_deindex(
                a_,
                names,
                operands,
                events,
                state=state,
                consts=consts,
                pullout=pullout,
                spec_cap=spec_cap,
                _budget=_budget,
            )
            if p is None:
                return None
            parts.append(p)
        spec = [t.op, *parts]
        if t.attrs:
            spec.append(dict(t.attrs))
        return tuple(spec)
    if isinstance(t, Const):
        return None if isinstance(t.value, str) else t.value
    if consts is not None and isinstance(t, (Param, Var)):
        # loop-invariant leaf — a weight or input the fold reads
        # whole at every step; recorded in *consts*, not *operands*
        return _seq_const_meta(consts, t)
    return None


def _seq_site(steps: tuple, base: Any, spec_cap: int | None = None):
    """Seq-form spec record for one spine, or ``None``.

    The fold ``acc = a_t * acc + b_t`` reproduces the unrolled chain
    exactly when every step's ``a``/``b`` deindex to the SAME spec
    skeleton — per-step ``select``s becoming sequence operands —
    and each select's index equals its chronological position along
    a single shared step axis whose extent is the step count.  The
    shape check via ``shape_of`` is what makes the axis claim
    sound: equal-rank operands with the step extent at one dim mean
    the fold's broadcast slice reads exactly the term the original
    ``select`` read.  Anything unverifiable declines honestly.

    *spec_cap* bounds the spelled size of each deindexed step
    skeleton (``_seq_deindex``'s visit budget) and every ``shape_of``
    operand check below (via :func:`_seq_expansion`): a step operand
    whose tree-expansion exceeds it raises :class:`_SeqSpecOverflow`
    — the collector counts the site as an honest over-cap skip
    rather than spelling a spec no claim could carry anyway.
    ``None`` keeps the historical unbounded behaviour.
    """
    from mlopt_core.typing import INVALID, shape_of

    names: dict = {}
    operands: dict = {}
    consts: dict = {}
    a0 = b0 = None
    events: list = []
    for k, (at, bt, _st) in enumerate(steps):
        ea: list = []
        eb: list = []
        budget = None if spec_cap is None else [spec_cap]
        sa = _seq_deindex(
            at,
            names,
            operands,
            ea,
            consts=consts,
            pullout=True,
            spec_cap=spec_cap,
            _budget=budget,
        )
        budget = None if spec_cap is None else [spec_cap]
        sb = _seq_deindex(
            bt,
            names,
            operands,
            eb,
            consts=consts,
            pullout=True,
            spec_cap=spec_cap,
            _budget=budget,
        )
        if sa is None or sb is None:
            return None
        if k == 0:
            a0, b0 = sa, sb
        elif sa != a0 or sb != b0:
            return None
        events.append(ea + eb)
    if not operands:
        return None
    dims = {d for _c, d in operands.values()}
    if len(dims) != 1:
        return None
    dim = dims.pop()
    t_len = len(steps)
    rank = None
    eff = dim
    state_sh: tuple | None = None
    for _m, (child, _d) in operands.items():
        if (
            spec_cap is not None
            and _seq_expansion(child, spec_cap) > spec_cap
        ):
            raise _SeqSpecOverflow
        sh = shape_of(child)
        if sh is INVALID or not isinstance(sh, tuple):
            return None
        if rank is None:
            rank = len(sh)
            eff = dim if dim >= 0 else dim + rank
        if (
            len(sh) != rank
            or not -rank <= dim < rank
            or sh[eff] != t_len
        ):
            return None
        parts = sh[:eff] + sh[eff + 1 :]
        state_sh = (
            parts if state_sh is None else _seq_bcast(state_sh, parts)
        )
        if state_sh is None:
            return None
    if state_sh is None:
        return None  # operands is non-empty — unreachable, narrows ty
    for k, evs in enumerate(events):
        for _m, idx in evs:
            if idx != k:
                return None
    # Loop-invariant leaves minted as ``"P"`` metas must broadcast
    # into the fold's state shape — the same contract the per-step
    # ``_eval_spec``/kernel broadcast applies at run time.  A const
    # that cannot is not a per-step operand the fold can read
    # honestly; decline rather than emit a spec that raises on call.
    for t_const, _m in consts.items():
        if (
            spec_cap is not None
            and _seq_expansion(t_const, spec_cap) > spec_cap
        ):
            raise _SeqSpecOverflow
        csh = shape_of(t_const)
        if (
            csh is INVALID
            or not isinstance(csh, tuple)
            or _seq_bcast(csh, state_sh) != state_sh
        ):
            return None
    h_meta = "H"
    attrs: dict = {"kind": "seq", "axis": eff}
    if consts:
        attrs["const"] = tuple(consts.values())
    spec = (
        "applyd",
        ("aff_diag", a0, b0),
        h_meta,
        attrs,
    )
    leaf_env = {m: c for m, (c, _d) in operands.items()}
    leaf_env.update({m: t for t, m in consts.items()})
    leaf_env[h_meta] = base
    return spec, leaf_env, eff


def _op_parents(term: Any) -> dict:
    """Map ``id(child) -> [parent ops]`` over the whole term DAG."""
    parents: dict = {}
    for n in _term_ops(term):
        for a in n.args:
            parents.setdefault(id(a), []).append(n)
    return parents


def _op_descendants(root: Any) -> set:
    """``id`` set of every ``Op`` reachable from *root* (inclusive)."""
    return {id(n) for n in _term_ops(root)}


def _seq_traj_states(
    term: Any, root: Any, steps: tuple
) -> tuple | None:
    """Trajectory state map for a chain whose steps are read out.

    Returns ``((state_node, step_index), ...)`` — every prefix
    state ``S_t`` of the *steps*-long chain in chronological order —
    when at least one NON-FINAL state has a consumer outside the
    chain's own spelled subtree (the per-step read-out shape:
    ``einsum(S_t, q_t)`` feeding a ``stack`` — the GLA/Mamba
    modules).  ``None`` when the chain is only consumed at its
    root: the ``seq`` final-state fold already covers that reading,
    and minting ``seqs`` would pay a ``(T, ·)`` trajectory write
    for a ``(·,)`` answer.

    "External" is decided structurally, not by op name: a parent of
    a state node that is not a descendant of *root* cannot be part
    of the recurrence itself — every interior consumer (the next
    step's ``mul`` factor) sits inside *root*'s subtree.  The check
    is deliberately consumer-agnostic: ``select(traj, t)`` denotes
    ``S_t`` whatever reads it, so the collector needs no read-out
    grammar — only the proof that a trajectory is *required*.  The
    route (:func:`meta_arena._route_scan_chains`) rewrites each
    recorded state e-class to that ``select``.
    """
    t_len = len(steps)
    # ``steps[k][2]`` is the state INPUT of step ``k`` — ``S_{k-1}``
    # in chain coordinates — so ``steps[k][2]`` for ``k >= 1`` is
    # ``S_{k-1}`` and *root* is ``S_{t_len-1}``.
    states = tuple(
        [(steps[k][2], k - 1) for k in range(1, t_len)]
        + [(root, t_len - 1)]
    )
    # every node's parents in the module term — the external-
    # consumer census the trajectory decision reads; and the
    # chain's own subtree: interior consumers (the next step's
    # ``mul``) are descendants of *root*, anything else reading a
    # state node is a read-out the final-state fold cannot serve
    parents = _op_parents(term)
    internal = _op_descendants(root)
    for node, _idx in states[:-1]:
        for p in parents.get(id(node), ()):
            if id(p) not in internal:
                return states
    return None


def _seqs_chain_spec(spec: tuple) -> tuple:
    """Return *spec* re-kinded ``seq`` → ``seqs`` (same carrier).

    The aff_diag leaf and operand skeleton are identical — only the
    minted op's declared semantics changes: every prefix state
    (``out_t = acc_t``) instead of the final state, which is what
    the read-out consumers actually read.
    """
    attrs = (
        spec[3] if len(spec) > 3 and isinstance(spec[3], dict) else {}
    )
    return ("applyd", spec[1], spec[2], {**attrs, "kind": "seqs"})


def _seqt_chain_spec(spec: tuple) -> tuple:
    """Return *spec* re-kinded ``seq`` → ``seqt`` (same carrier).

    Same fold skeleton as ``seqs``; the minted op additionally
    denotes the per-step read-out contraction + stack recorded in
    the site's ``"readout"`` — see :func:`_seqt_readout`.
    """
    attrs = (
        spec[3] if len(spec) > 3 and isinstance(spec[3], dict) else {}
    )
    return ("applyd", spec[1], spec[2], {**attrs, "kind": "seqt"})


def _seqt_readout(term: Any, root: Any, states: tuple):
    """Fused read-out record for a trajectory site, or ``None``.

    The ``seqs`` tier records THAT prefix states are consumed; this
    collects the contraction structure the ``seqt`` kernel needs —
    and only what is structurally present in the IR.  The recognised
    shape (the GLA/Mamba/HeadedSelectiveScan spelling)::

        for t:  o_t = einsum(eq, S_t, select(w, dim, t))
        out    = stack(o_0 … o_{T-1}, dim=D)

    Every recorded state node must have EXACTLY ONE consumer outside
    the chain's own subtree — a two-operand ``einsum`` whose state
    argument IS that node — and every step's einsum must share the
    equation, the state operand position, and a ``select`` read-out
    operand over the SAME child at the SAME dim with ``index ==``
    the state's step index.  Those einsums must in turn feed ONE
    ``stack`` whose args are the einsums in step order.  The
    equation itself is classified by
    ``genkernel._seqt_layout`` (one summed state axis, output =
    kept labels in state order, read-out operand = kept-prefix +
    contract-last); anything outside declines to ``None`` and the
    caller keeps the ``seqs`` trajectory offer.

    Returns ``(readout, w_env)`` — the record merged into the site
    (``equation``/``state_pos``/``ops``/``stack``/``stack_dim``) and
    the ``{"W1": child}`` leaf-env extension binding the read-out
    operand's whole sequence tensor.
    """
    from mlopt_discovery import genkernel as gk

    parents = _op_parents(term)
    internal = _op_descendants(root)
    parsed = []
    for node, idx in states:
        ext = [
            p
            for p in parents.get(id(node), ())
            if id(p) not in internal
        ]
        if len(ext) != 1:
            return None
        e = ext[0]
        if not (
            isinstance(e, Op)
            and e.op == "einsum"
            and len(e.args) == 2
            and set(e.attrs) == {"equation"}
        ):
            return None
        ps = gk._seqt_parse_equation(e.attrs.get("equation"), 2)
        if ps is None:
            return None
        hits = [i for i, a in enumerate(e.args) if a == node]
        if len(hits) != 1:
            return None
        spos = hits[0]
        w = e.args[1 - spos]
        if not (
            isinstance(w, Op)
            and w.op == "select"
            and len(w.args) == 1
            and set(w.attrs) <= {"dim", "index"}
            and type(w.attrs.get("dim")) is int
            and type(w.attrs.get("index")) is int
            and w.attrs["index"] == idx
        ):
            return None
        parsed.append((e, spos, ps, w.args[0], w.attrs["dim"]))
    first = parsed[0]
    subs, out = first[2]
    if gk._seqt_layout(subs, out, first[1]) is None:
        return None
    for e, spos, _ps, child, dim in parsed[1:]:
        if (
            spos != first[1]
            or e.attrs["equation"] != parsed[0][0].attrs["equation"]
            or child != first[3]
            or dim != first[4]
        ):
            return None
    # the einsums must feed exactly one stack, in step order
    stack = None
    for e, _spos, _ps, _child, _dim in parsed:
        eps = parents.get(id(e), ())
        if not eps:
            return None
        for p in eps:
            if not (
                isinstance(p, Op)
                and p.op == "stack"
                and set(p.attrs) <= {"dim"}
                and type(p.attrs.get("dim")) is int
            ):
                return None
            if stack is None:
                stack = p
            elif p != stack:
                return None
    assert stack is not None
    t_len = len(states)
    if len(stack.args) != t_len:
        return None
    for idx, (e, *_rest) in enumerate(parsed):
        if stack.args[idx] != e:
            return None
    _e, spos, ps, child, dim = first
    readout = {
        "equation": _e.attrs["equation"],
        "state_pos": spos,
        "ops": [{"meta": "W1", "axis": dim}],
        "stack": stack,
        "stack_dim": stack.attrs["dim"],
    }
    return readout, {"W1": child}


def _seq_scan_sites(
    term: Any,
    *,
    spec_cap: int | None = None,
    stats: dict | None = None,
) -> list:
    """Seq-form scan specs minted from raw recurrence chains.

    The unrolled collectors above only see what saturation already
    built; the *sequence* form — ``applyd(aff_diag(a, b), h)`` whose
    operands carry a step axis and whose declared op is the fold
    ``acc = a_t * acc + b_t`` — has a different, earlier source: the
    exported recurrence itself.  Models spell ``h_t = a_t ⊙ h_{t-1}
    + b_t`` as an unrolled ``add(mul(·, ·), ·)`` chain over per-step
    ``select(x, dim, t)`` slices; this collector walks those chains,
    abstracts the selects into sequence operands and mints the seq
    spec whose fold IS the chain — sound by construction, since the
    spec is only emitted when the deindexed skeletons agree step for
    step and the select indices walk the axis in order.

    Sites are ``{"spec", "leaf_env", "axis"}`` records: ``leaf_env``
    binds each operand metavar to its full sequence tensor (the
    ``select`` child) and ``H`` to the initial-state term, and
    ``axis`` is the resolved step axis — merged into the handler
    entry so ``handler_denotation`` and the Triton binding agree on
    it.  Chains that are strict prefixes of a longer emitted chain
    are skipped (the maximal chain covers them); ambiguous or
    unverifiable readings decline rather than guess.

    *spec_cap* bounds each candidate's deindexed step skeletons and
    operand shape checks (see :func:`_seq_site`) — on shared-DAG
    model terms the unbounded spelling is exponential in the
    distinct-node count and OOMs host RAM (t5_r1 wall 4).  Candidates
    the bound declines are counted in *stats* under
    ``"seq_sites_over_cap"`` — never silently dropped.
    ``None`` keeps the historical unbounded behaviour.
    """
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    memo: dict = {}
    cands: list = []

    # ``traversal.ops`` — the same visit order the recursive DFS
    # produced (``cands`` order feeds site emission), DAG-deduped,
    # without per-level frames on deep terms.
    for t in _term_ops(term):
        for steps, base in _seq_spines(t, memo):
            if len(steps) >= 2:
                cands.append((t, steps, base))

    out: list = []
    emitted_roots: set = set()
    emitted_states: set = set()
    for root, steps, base in cands:
        if id(root) in emitted_roots or id(root) in emitted_states:
            continue
        try:
            site = _seq_site(steps, base, spec_cap=spec_cap)
        except _SeqSpecOverflow:
            if stats is not None:
                stats["seq_sites_over_cap"] = (
                    stats.get("seq_sites_over_cap", 0) + 1
                )
            continue  # next tied reading of the same root
        if site is None:
            continue  # next tied reading of the same root
        spec, leaf_env, axis = site
        traj = _seq_traj_states(term, root, steps)
        rec = {
            "spec": spec,
            "leaf_env": leaf_env,
            "axis": axis,
        }
        if traj is None:
            # the unrolled chain this fold denotes — the
            # deliver-time route's LHS (the intake IR has no
            # ``applyd`` node for a claim to match, so the
            # rewrite fires on the extracted term instead).
            rec["chain"] = root
        else:
            # per-step read-outs consume intermediate states — the
            # trajectory (``seqs``) fold is the honest carrier, and
            # ``states`` is the route's LHS set: every recorded
            # state e-class rewrites to ``select(traj, t)``.
            rec["spec"] = _seqs_chain_spec(spec)
            rec["states"] = traj
            # the fused ``seqt`` offer rides BESIDE the seqs one:
            # when every state's consumer is the same two-operand
            # einsum feeding one stack, a second site mints the
            # fold+read-out kernel — the referee prices both and
            # delivery prefers whichever verifies cheaper (the seqs
            # route stays the fallback where the seqt binding
            # declines).
            selt = _seqt_readout(term, root, traj)
            if selt is not None:
                ro, wenv = selt
                rec2 = {
                    "spec": _seqt_chain_spec(spec),
                    "leaf_env": {**leaf_env, **wenv},
                    "axis": axis,
                    "readout": ro,
                }
                if rec2 not in out:
                    out.append(rec2)
        # dedup on the whole record — spec AND leaf_env AND chain —
        # not the spec tuple alone: the fold's denotation binds the
        # ``leaf_env`` operands, so two same-shaped chains over
        # different sequences (e.g. a bidirectional scan's forward
        # and flipped backward recurrences) are two distinct folds
        # that must each mint a site.
        if rec not in out:
            out.append(rec)
        for _m, _i, s in steps:
            emitted_states.add(id(s))
        emitted_roots.add(id(root))
        if len(out) >= _SEQ_SITE_CAP:
            break
    return out


def _spec_has_meta(spec: Any, meta: str) -> bool:
    """Whether *meta* occurs as a leaf of spec *spec*.

    Iterative walk — spec trees mirror term depth, so the recursive
    version shared the model term's recursion-depth exposure.
    """
    stack = [spec]
    while stack:
        s = stack.pop()
        if s == meta:
            return True
        if isinstance(s, (tuple, list)):
            stack.extend(e for e in s[1:] if not isinstance(e, dict))
    return False


def _seqm_site(steps: tuple, base: Any, spec_cap: int | None = None):
    """Seq-matrix spec record for one spine, or ``None``.

    The matrix-state analogue of :func:`_seq_site`: the fold is
    ``acc = a_t * acc + b_t(acc)`` where the state term may occur
    inside *b_t* — the delta-rule read ``kᵀ·acc`` — abstracted to
    the ``"H"`` metavar by :func:`_seq_deindex`'s *state* argument.
    Steps spelled ``h' = h + f(h)`` carry ``at is None`` (the
    implicit unit map); ``a_t`` must never depend on the
    accumulator, and *b_t* must — a state-free ``b`` is the plain
    seq form's job (or nothing's).

    The operand checks differ from :func:`_seq_site` in ONE way:
    there is no common broadcast frame.  Matrix-state operands live
    in different spaces (``k ∈ R^dk``, ``v ∈ R^dv``, ``acc ∈
    R^(dk,dv)``) that only meet inside the step expr, so instead of
    equal-rank/single-frame ``shape_of`` checks each operand is
    verified on its own: its select dim is in range and its extent
    there is the step count.  The per-operand axis is recorded in
    ``axes`` — the binding and denotation slice each operand at ITS
    own normalized dim, exactly what the original ``select`` read.
    Everything else — identical skeletons step to step, index ==
    chronological position — is the same soundness argument.
    *spec_cap* is the seq work bound, exactly as in :func:`_seq_site`.
    """
    from mlopt_core.typing import INVALID, shape_of

    names: dict = {}
    operands: dict = {}
    axes: dict = {}
    a0 = b0 = None
    events: list = []
    for k, (at, bt, st) in enumerate(steps):
        ea: list = []
        eb: list = []
        budget = None if spec_cap is None else [spec_cap]
        sa = (
            1
            if at is None
            else _seq_deindex(
                at,
                names,
                operands,
                ea,
                state=st,
                spec_cap=spec_cap,
                _budget=budget,
            )
        )
        budget = None if spec_cap is None else [spec_cap]
        sb = _seq_deindex(
            bt,
            names,
            operands,
            eb,
            state=st,
            spec_cap=spec_cap,
            _budget=budget,
        )
        if sa is None or sb is None:
            return None
        if _spec_has_meta(sa, "H"):
            return None  # a_t must be accumulator-free
        if k == 0:
            a0, b0 = sa, sb
        elif sa != a0 or sb != b0:
            return None
        events.append(ea + eb)
    if not operands or not _spec_has_meta(b0, "H"):
        return None
    t_len = len(steps)
    for m, (child, d) in operands.items():
        if (
            spec_cap is not None
            and _seq_expansion(child, spec_cap) > spec_cap
        ):
            raise _SeqSpecOverflow
        sh = shape_of(child)
        if sh is INVALID or not isinstance(sh, tuple) or not sh:
            return None
        eff = d if d >= 0 else d + len(sh)
        if not 0 <= eff < len(sh) or sh[eff] != t_len:
            return None
        axes[m] = eff
    for k, evs in enumerate(events):
        for _m, idx in evs:
            if idx != k:
                return None
    a_spec = a0 if a0 is not None else 1
    spec = (
        "applyd",
        ("aff_diag", a_spec, b0),
        "H",
        {"kind": "seqm", "axis": next(iter(axes.values()))},
    )
    leaf_env = {m: c for m, (c, _d) in operands.items()}
    leaf_env["H"] = base
    return spec, leaf_env, axes


def _seqm_scan_sites(
    term: Any,
    *,
    spec_cap: int | None = None,
    stats: dict | None = None,
) -> list:
    """Seq-matrix specs minted from state-feedback recurrence chains.

    Same traversal and caps as :func:`_seq_scan_sites`; the site
    record additionally carries ``axes`` — each operand metavar's
    normalised step axis — which :func:`_seqm_site` resolved per
    operand (there is no shared frame to derive it from).  Records
    are ``{"spec", "leaf_env", "axis", "axes"}``.  Over-cap declines
    are counted in *stats* under ``"seqm_sites_over_cap"``.
    """
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    memo: dict = {}
    cands: list = []

    # ``traversal.ops`` — identical visit order to the recursive
    # DFS (see ``_seq_scan_sites``), DAG-deduped.
    for t in _term_ops(term):
        for steps, base in _seq_spines(t, memo):
            if len(steps) >= 2:
                cands.append((t, steps, base))

    out: list = []
    emitted_roots: set = set()
    emitted_states: set = set()
    for root, steps, base in cands:
        if id(root) in emitted_roots or id(root) in emitted_states:
            continue
        try:
            site = _seqm_site(steps, base, spec_cap=spec_cap)
        except _SeqSpecOverflow:
            if stats is not None:
                stats["seqm_sites_over_cap"] = (
                    stats.get("seqm_sites_over_cap", 0) + 1
                )
            continue
        if site is None:
            continue
        spec, leaf_env, axes = site
        rec = {
            "spec": spec,
            "leaf_env": leaf_env,
            "axis": next(iter(axes.values())),
            "axes": axes,
            # see ``_seq_scan_sites``: the chain root is the
            # deliver-time route's LHS.
            "chain": root,
        }
        # whole-record dedup, same contract as ``_seq_scan_sites``:
        # the spec tuple does not see ``leaf_env``, so same-shaped
        # chains over different operands would collapse into one
        # site and the second fold would never be offered.
        if rec not in out:
            out.append(rec)
        for _m, _i, s in steps:
            emitted_states.add(id(s))
        emitted_roots.add(id(root))
        if len(out) >= _SEQ_SITE_CAP:
            break
    return out


def _cumsum_sites(
    term: Any,
    *,
    spec_cap: int | None = None,
    stats: dict | None = None,
) -> list:
    """Emitted-scan specs minted from ``cumsum``/``cumprod`` ops.

    The select-chain collectors read recurrences spelled as unrolled
    ``add(mul(·, ·), ·)`` chains; a ``cumsum`` op is ONE node — no
    spine, no site — even though the prefix sum IS the same
    recurrence with ``a ≡ 1`` emitted at every step.  This collector
    reads the op directly and mints the ``kind="seqs"`` spec whose
    fold denotes every prefix state (``out_t = acc_t``), the form
    ``genkernel.seqs_denotation``/the ``seqs`` Triton piece implement.

    ``cumsum(x, dim)`` spells ``acc_t = 1·acc_{t-1} + x_t`` from a
    zero carry; ``cumprod`` spells ``acc_t = x_t·acc_{t-1} + 0`` from
    a unit carry — the carry is a ``Const`` leaf in ``leaf_env``
    (the scan's neutral element, not a guessed parameter).  The
    ``"H"`` metavar binds it; ``"X1"`` binds the scanned child and
    ``axis`` the normalised step dim (``shape_of``-checked, the same
    honesty contract the seq collectors keep).  ``chain`` is the
    ``cumsum``/``cumprod`` node itself — delivery rewrites it to the
    minted kernel call exactly as the unrolled chains are routed.

    *spec_cap* bounds the ``shape_of`` child check (see
    :func:`_seq_expansion`); children over it are skipped and counted
    in *stats* under ``"cumsum_sites_over_cap"``.
    """
    from mlopt_core.ir import Const, Op
    from mlopt_core.typing import INVALID, shape_of

    out: list = []
    seen: set = set()

    # Iterative post-order — the recursive visit processed children
    # before the node itself, so sites append in post-order; the
    # (node, expanded) stack preserves that order exactly, id-
    # deduped, without per-level frames on deep terms.
    stack: list[tuple[Any, bool]] = [(term, False)]
    while stack:
        t, expanded = stack.pop()
        if not isinstance(t, Op):
            continue
        if expanded:
            if id(t) in seen:
                continue
            seen.add(id(t))
        else:
            if id(t) in seen:
                continue
            stack.append((t, True))
            stack.extend((a, False) for a in reversed(t.args))
            continue
        if (
            t.op not in ("cumsum", "cumprod")
            or len(t.args) != 1
            or set(t.attrs) - {"dim"}
            or type(t.attrs.get("dim")) is not int
        ):
            continue
        child = t.args[0]
        if (
            spec_cap is not None
            and _seq_expansion(child, spec_cap) > spec_cap
        ):
            if stats is not None:
                stats["cumsum_sites_over_cap"] = (
                    stats.get("cumsum_sites_over_cap", 0) + 1
                )
            continue
        sh = shape_of(child)
        if sh is INVALID or not isinstance(sh, tuple) or not sh:
            continue
        eff = t.attrs["dim"] % len(sh)
        if t.op == "cumsum":
            a_spec, b_spec, carry = 1, "X1", 0.0
        else:
            a_spec, b_spec, carry = "X1", 0, 1.0
        spec = (
            "applyd",
            ("aff_diag", a_spec, b_spec),
            "H",
            {"kind": "seqs", "axis": eff},
        )
        out.append(
            {
                "spec": spec,
                "leaf_env": {"X1": child, "H": Const(carry)},
                "axis": eff,
                "chain": t,
            }
        )

    return out[:_SEQ_SITE_CAP]


def _all_scan_sites(
    term: Any,
    *,
    spec_cap: int | None = None,
    stats: dict | None = None,
) -> list:
    """All scan site forms: carriers + raw recurrences, both folds.

    *spec_cap*/*stats* bound and count the seq/seqm/cumsum
    collectors' spelled-size work (the carrier collector's bounded
    saturation needs neither) — see :func:`_seq_scan_sites`.
    """
    return [
        *_scan_sites(term),
        *_seq_scan_sites(term, spec_cap=spec_cap, stats=stats),
        *_seqm_scan_sites(term, spec_cap=spec_cap, stats=stats),
        *_cumsum_sites(term, spec_cap=spec_cap, stats=stats),
    ]


#: Site collectors, keyed by codegen-tier name — the term-traversal
#: knowledge that pairs with ``genkernel.TIERS``.  A tier without a
#: collector mints nothing here; the ``scan`` collector joins the
#: saturated-graph carriers with the raw recurrence chains that mint
#: the sequence form.
TIER_SITES: dict[str, Any] = {
    "gen": _fusion_sites,
    "decayed": _decayed_sites,
    "scan": _all_scan_sites,
    "reduction": _reduction_sites,
}


def _mint_tiers(
    root: Any,
    covered: set,
    *,
    spec_cap: int | None = None,
    stats: dict | None = None,
    bounds: ProbeBounds | None = None,
) -> dict:
    """Mint handler entries across every codegen tier.

    The registry walk replacing per-tier call sites: each tier's
    site collector gathers specs, its ``handlers`` builder mints
    the ``{tag: {pattern, kernel, args}}`` table — a tier without
    a collector contributes nothing.  *spec_cap* is the
    ``ProbeBounds.spec_tree_cap`` — the SPELLED TREE-SIZE bound on
    minted site patterns (``gen``/``reduction`` collectors) and the
    seq-form scan collectors' deindexed skeletons/shape checks;
    ``None`` keeps the zoo's unbounded spelling.  *bounds*, when
    given, supplies *spec_cap*'s default (an explicit *spec_cap*
    always wins).  *stats* is the :class:`SkipStats`-compatible
    tally EVERY bounded collector reports its honest over-cap
    declines into (``*_sites_over_cap`` keys — the scan tiers
    counted these before; ``gen``/``reduction`` join them now, the
    arch-audit §1.4 asymmetry fix).
    """
    from mlopt_discovery import genkernel as gk

    if bounds is not None and spec_cap is None:
        spec_cap = bounds.spec_tree_cap
    out: dict = {}
    for tier in gk.TIERS:
        sites_fn = TIER_SITES.get(tier.name)
        if sites_fn is None:
            continue
        if tier.name == "decayed":
            sites = sites_fn(root)
        else:
            sites = sites_fn(root, spec_cap=spec_cap, stats=stats)
        out.update(tier.handlers(sites, covered=covered))
    return out


def _flat_ops(spec: Any):
    """Yield op names inside a fusion pattern tuple.

    Iterative — spec trees mirror term depth; same yield order as
    the recursive render (head first, then children left-to-right).
    """
    stack = [spec]
    while stack:
        s = stack.pop()
        if isinstance(s, (tuple, list)) and s:
            yield s[0]
            stack.extend(
                e for e in reversed(s[1:]) if not isinstance(e, dict)
            )

"""The primitive-op mint — ``synop_*``-class ops for shared manual shapes.

The corpus-level mint the ``zoo-prim`` overnight arm runs
(``bench/drivers/overnight.py --corpus zoo-prim``): mine recurring
manual subgraphs out of a merged corpus e-graph
(:func:`mlopt_discovery.antiunify.mine_candidates`), keep the shapes
binding >= *k* distinct modules that no bank op already *denotes*
(:func:`denoted_alphas`), and mint a NEW primitive op per shape —
``<prefix>_<sha>`` — whose denotation IS the spelled subgraph.  The
emitted candidate is the fold ``pattern -> <prefix>_*(args)``
carrying the ``<prefix>_*.expand`` definitional premise, so the cert
leg's anchored derivation has the axiom the bare bank lacks.

The machinery is domain-agnostic: the domain driver supplies the
*hooks* —

* ``bindings`` — the op-name -> callable table the denotation
  registers onto and the evaluability gate reads (the ambient
  executor table; ADR 0007: the table IS the op's meaning);
* ``eval_pair`` — the emit gate's fp64 truth oracle
  ``(concrete_lhs, concrete_rhs) -> verdict``: ``equal`` counts a
  verified site, ``unequal``/``rhs-err`` refuse the shape, any other
  verdict is inconclusive;
* ``denoted`` — alpha-normal keys of spellings the domain's bank
  already names (:func:`denoted_alphas` collects them);
* ``prefix`` — the minted ops' name stem (``synop`` on the tensor
  domain — a domain minting its own vocabulary picks its own);
* ``handler_bindings`` — the bindings the handler/family grammar
  mints under (defaults to *bindings*);
* ``register_shipped`` / ``shipped`` — the vocabulary bootstrap's
  denotation seater and the shipped-primitive table (default: the
  core-shipped ``synop*`` set in ``mlopt_core.laws.minted``).

Two consumers today: the graphopt ``zoo-prim`` driver (torch
bindings, ``oracle.eval_instance``, the substrate bank's denoted
set, ``shipped_primitives.register``) and any domain repo that
supplies its own sink-side equivalents — the pool, the gates, the
naming and the fold/premise construction live here exactly once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mlopt_core.egraph import EGraph, Rewrite
from mlopt_core.egraph.terms import _term_match
from mlopt_core.ir import Const, Op, op_def
from mlopt_core.laws import minted
from mlopt_core.meta import instantiate_pattern

from mlopt_discovery.antiunify import (
    _alpha_norm,
    _n_ops,
    mine_candidates,
)
from mlopt_discovery.genkernel import _eval_spec
from mlopt_discovery.proposal import Candidate

__all__ = [
    "PRIM_BEAM",
    "PRIM_BUDGET",
    "PRIM_CAP",
    "PRIM_DEPTH",
    "PRIM_DOMAIN",
    "PRIM_FAMILY",
    "PRIM_MAX_EXPAND",
    "PRIM_MAX_OPS",
    "PRIM_MIN_OPS",
    "PRIM_SITES",
    "ShippedPrims",
    "admit_prim",
    "bare_call",
    "core_shipped",
    "denoted_alphas",
    "expand_synops",
    "freeze_spec",
    "make_denotation",
    "merge_shipped",
    "minted_prim_record",
    "modules_bound",
    "op_subterms",
    "pat_ops",
    "pat_spec",
    "prim_domain",
    "prim_fold_of",
    "prim_pool",
    "register_denotation",
    "seat_shipped",
    "shipped_from_records",
    "shipped_from_vocab",
    "spec_bind",
    "sub_alphas",
    "synop_depth",
    "synop_ops",
    "synop_vocab",
]

#: The ``meta.family`` a primitive-minted fold records — the ledger
#: distinguishes the ``prim:<prefix>_*`` folds from rewrite mints.
PRIM_FAMILY = "prim-fold"

#: Cap on mined shapes taken through the mint — the top-``gain``
#: slice, so the pool stays bounded like a page.
PRIM_CAP = 24

#: ``mine_candidates`` size window — the deep-shape bounds the
#: cross-module mint reads.
PRIM_MIN_OPS = 2
PRIM_MAX_OPS = 6

#: The mining beam / pop bound — ``mine_candidates``' own defaults.
#: Deeper windows need the wider frontier the deep preset carries.
PRIM_BEAM = 32
PRIM_MAX_EXPAND = 2000

#: The deepest *composition depth* a bootstrap mint may reach: a
#: pooled pattern carrying a shipped ``<prefix>*`` op of depth ``d``
#: mints a depth-(``d+1``) primitive.  The seating loop only applies
#: folds whose synop is strictly below the cap — a deeper enode could
#: never be embedded anyway — and runs to fixpoint.
PRIM_DEPTH = 3

#: Bound on bound sites a candidate's verify/domain stage walks.
PRIM_SITES = 16

#: Bound on bound sites sampled into the handler-verify domain.
PRIM_DOMAIN = 4

#: ``mint_handlers`` candidate cap per shape — the bounded grammar
#: search the synthesis measurement spends (refusals quote it).
PRIM_BUDGET = 2048


# ---------------------------------------------------------------------------
#  the shipped-primitive table — the bootstrap vocabulary's seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShippedPrims:
    """A domain's shipped primitive set — what mining composes over.

    ``names`` is the op set ``expand_synops``/``synop_ops``/
    ``synop_vocab`` read; ``folds`` the ``prim:<op>`` fold laws the
    bootstrap seats into the corpus e-graph; ``fold_for`` maps one op
    name to its ``(fold, expand_premise)`` pair.  The default table
    (:func:`core_shipped`) is the core-shipped ``synop*`` set —
    laws, not denotations: the domain's ``register_shipped`` hook is
    what makes the seated enodes evaluable.
    """

    names: frozenset
    folds: tuple
    fold_for: Any


_CORE_SHIPPED: ShippedPrims | None = None


def core_shipped() -> ShippedPrims:
    """Return the core-shipped ``synop*`` table — all three tiers.

    Materializing the fold/premise pairs is memoized in
    ``mlopt_core.laws.minted`` and the table itself interned here —
    ``expand_synops`` consults it per mined candidate.
    """
    global _CORE_SHIPPED
    if _CORE_SHIPPED is None:
        # ``synop_shipped`` is the live set (persisted AUTO_TIER
        # admissions join it); the pinned mlopt-core predating it
        # falls back to the static SYNOP_SHIPPED snapshot.
        _shipped_names = getattr(minted, "synop_shipped", None)
        _CORE_SHIPPED = ShippedPrims(
            names=(
                _shipped_names()
                if _shipped_names is not None
                else minted.SYNOP_SHIPPED
            ),
            folds=tuple(minted.prim_folds()),
            fold_for=minted.prim_fold,
        )
    return _CORE_SHIPPED


def prim_fold_of(name: str, rec: dict) -> tuple[Rewrite, Rewrite]:
    """Build ``(fold, expand)`` for one prim *record* — any source.

    The record-generic twin of ``mlopt_core.laws.minted.prim_fold``:
    ``minted`` materializes the core-embedded records; this rebuilds
    the same pair from a ``prims.yaml``-shaped record dict — the
    shape ``vocabdir.emit_prim`` writes and ``vocab.prims`` loads.
    The fold restores through ``object_from_data`` and gets the
    shipped integration fields (``FUSION`` tag, fold law text,
    premise attached); a record embedding ``"expand"`` uses it
    verbatim (depth-2+: the premise RHS is the *expanded* manual
    denotation, checked against the fold's own call), a depth-1
    record's premise is synthesized as ``call -> fold.lhs``.  The
    op's ``op_def`` declaration registers with the fold — the same
    side effect ``minted.prim_fold`` has.  Results are memoized per
    ``(name, record identity)`` — records are treated as immutable.
    """
    key = (name, id(rec))
    hit = _FOLD_OF_MEMO.get(key)
    if hit is not None:
        return hit
    from mlopt_core.laws import minted, tags
    from mlopt_core.laws.serialize import object_from_data

    fold = object_from_data(rec["fold"])
    if "expand" in rec:
        ex = object_from_data(rec["expand"])
        if ex.lhs != fold.rhs:
            raise ValueError(
                f"prim record {name!r} embeds an expand premise "
                f"whose LHS is not the fold's own call — a "
                f"mismatched axiom would anchor nothing"
            )
        denotation = ex.rhs
    else:
        denotation = fold.lhs
    expand = Rewrite(
        minted.expand_name(name),
        fold.rhs,
        denotation,
        law=minted._PRIM_EXPAND_LAW.format(synop=name),
        tags=frozenset({tags.SIMPLIFICATION}),
    )
    out = Rewrite(
        name=fold.name,
        lhs=fold.lhs,
        rhs=fold.rhs,
        law=minted._PRIM_FOLD_LAW.format(synop=name),
        tags=frozenset({tags.FUSION}),
        error_bound=fold.error_bound,
        bound_norm=fold.bound_norm,
        derivation=fold.derivation,
        cond=fold.cond,
        dspec=fold.dspec,
    )
    object.__setattr__(out, "premises", (expand,))
    op_def(
        name,
        len(fold.rhs.args),
        1,
        law=(
            "synthesized primitive — denotation is the mined manual "
            f"subgraph {rec.get('alpha', '')}"
        ),
    )
    _FOLD_OF_MEMO[key] = (out, expand)
    return _FOLD_OF_MEMO[key]


_FOLD_OF_MEMO: dict = {}


def shipped_from_records(records: dict) -> ShippedPrims:
    """Build a :class:`ShippedPrims` table from ``{name: record}``.

    The record-generic ``core_shipped``: *records* is the
    ``prims.yaml`` shape — ``mlopt_core.laws.minted.prim_records()``
    or a fragment directory's ``vocab.prims`` (which carries the
    shipped records verbatim plus whatever the mint emitted since).
    Folds/premises rebuild through :func:`prim_fold_of`.
    """
    table: dict[str, tuple] = {}
    folds: list = []
    for name in sorted(records):
        fold, expand = prim_fold_of(name, records[name])
        table[name] = (fold, expand)
        folds.append(fold)
    return ShippedPrims(
        names=frozenset(records),
        folds=tuple(folds),
        fold_for=table.__getitem__,
    )


def shipped_from_vocab(vocab: Any) -> ShippedPrims:
    """Build the shipped-primitive table a ``Vocab`` carries.

    ``vocab.prims`` is the directory's prim-record map — the shipped
    ``synop*`` set plus every prim the mint has emitted into the
    directory since (plan 0033 phase 3: emitted vocab re-enters the
    next mint's search space by construction).
    """
    return shipped_from_records(dict(vocab.prims))


def merge_shipped(*tables: ShippedPrims) -> ShippedPrims:
    """Union ``ShippedPrims`` tables — later tables win name clashes.

    The vocab bootstrap's ambient table: the core-shipped set plus
    whatever the fragment directory carries that core does not (a
    same-named record is identical by construction — content-hash
    names — so the merge order only decides which object serves).
    """
    names: set = set()
    folds: dict = {}
    fold_for: dict = {}
    for t in tables:
        names |= set(t.names)
        for f in t.folds:
            folds.setdefault(f.name, f)
        for n in t.names:
            fold_for[n] = t.fold_for(n)
    return ShippedPrims(
        names=frozenset(names),
        folds=tuple(folds.values()),
        fold_for=fold_for.__getitem__,
    )


def seat_shipped(
    shipped: ShippedPrims,
    bindings: dict,
    *,
    names: Any = None,
) -> int:
    """Seat shipped synop denotations onto *bindings*; return count.

    The generic bootstrap seater for tables ``register_shipped``
    does not know — a vocab-loaded table's extra prims.  Only names
    missing from *bindings* register (a domain's own seater keeps
    precedence); *names* optionally restricts the seated subset
    (``None`` seats everything missing).  Each registration is
    :func:`register_denotation` over the fold's call/denotation
    patterns — the same mechanism the mint uses at emit time.
    """
    seated = 0
    for name in shipped.names:
        if names is not None and name not in names:
            continue
        if name in bindings:
            continue
        fold, expand = shipped.fold_for(name)
        register_denotation(name, fold.rhs, expand.rhs, bindings)
        seated += 1
    return seated


def admit_prim(
    prim: dict,
    rule: Rewrite,
    *,
    db: Any = None,
) -> tuple[Rewrite, Rewrite]:
    """Admit a gauntlet-cleared prim mint into the core minted store.

    *prim* is the pool's per-shape record (``meta[i]["prim"]`` —
    ``synop``/``alpha``/``mods``/``sites``/``equal_sites``/
    ``handler`` plus the depth-N fields); *rule* is the candidate's
    live fold ``Rewrite`` (``prim:<synop>`` with its ``.expand``
    premise attached — ``Candidate.as_rule()``).  The record is
    built by :func:`minted_prim_record` and handed to
    :func:`mlopt_core.laws.minted.admit_prim_record`, which writes
    the store's ``AUTO_TIER`` and materializes the shipped
    ``(fold, expand)`` pair.

    The caller owns the gate: this function does NOT re-verify —
    pass only a mint the admission gauntlet cleared (every
    ``bench.referees`` stage green, ``machine-original`` novelty).
    *db* defaults to the shipped ``minted.db``; a scratch path
    exercises the same write without touching the artifact.
    Returns the materialized ``(fold, expand)`` pair.
    """
    from mlopt_core.laws import minted

    name = prim["synop"]
    record = minted_prim_record(name, rule, prim)
    # the pinned mlopt-core predates auto-admission until the eqsat
    # change lands — fail loudly, never silently skip the write.
    admit = getattr(minted, "admit_prim_record", None)
    if admit is None:
        raise RuntimeError(
            "mlopt_core.laws.minted.admit_prim_record is missing — "
            "the pinned mlopt-core predates AUTO_TIER admission"
        )
    return admit(name, record, db=db)


def minted_prim_record(name: str, rule: Rewrite, prim: dict) -> dict:
    """Build the ``prims.yaml`` record for one pooled mint.

    *rule* is the candidate's fold ``Rewrite`` (``prim:<name>`` with
    its ``.expand`` premise attached); *prim* is the pool's per-shape
    record (``meta[i]["prim"]`` — ``synop``/``alpha``/``depth``/
    ``mods``/``sites``/``equal_sites``/``handler`` plus the depth-N
    ``alpha_fold``/``embeds_shipped``/``embeds`` fields).  The result
    is the shipped-record shape ``vocabdir.emit_prim`` /
    ``collect_prims`` carry: ``fold``/``expand`` as ``law_to_data``
    records, the mint's measurements verbatim.
    """
    from mlopt_core.laws.serialize import law_to_data

    rec: dict[str, Any] = {
        "alpha": prim["alpha"],
        "mods": list(prim["mods"]),
        "sites": prim["sites"],
        "equal_sites": prim["equal_sites"],
        "handler": prim["handler"],
        "fold": law_to_data(rule),
    }
    for key in (
        "depth",
        "ops",
        "opset",
        "alpha_fold",
        "embeds_shipped",
        "embeds",
    ):
        if key in prim:
            rec[key] = prim[key]
    premises = getattr(rule, "premises", ()) or ()
    # depth-1 mints omit ``expand`` — the premise is derivable as
    # ``call -> fold.lhs`` (the shipped records' own convention);
    # deeper tiers embed it verbatim because the premise RHS is the
    # *expanded* manual denotation, not the synop-carrying LHS.
    if prim.get("depth", 1) > 1 and premises:
        rec["expand"] = law_to_data(premises[0])
    return rec


def synop_depth(op: str) -> int:
    """Return a ``*synop*`` op's depth — ``synop_``=1, ``synopN_``=N.

    The mint names honestly: ``<prefix>_<hash>`` composes over manual
    ops alone (depth 1), ``<prefix>N_<hash>`` over depth-(``N-1``)
    synops (depth N).  Prefix-agnostic — the stem ends in ``synop``
    whatever the domain's prefix is.
    """
    head = op.split("_", 1)[0]
    if "synop" not in head:
        return 0
    digits = head[head.index("synop") + len("synop") :]
    return int(digits) if digits.isdigit() else 1


def expand_synops(pat: Any, shipped: ShippedPrims | None = None) -> Any:
    """Re-expand shipped ``*synop*`` nodes in *pat* to manual ops.

    A bootstrap-mined pattern carries ``*synop*`` ops as leaves of
    its skeleton — each node's denotation is the fold's LHS
    re-instantiated under the call's own bindings (``synop*(v*)`` →
    the spelled subgraph over ``v*``, attr metavars carried through
    ``$attr:`` — the ``synop*.expand`` premises give the definition).
    The result is the shape's *denotation spelling*: the manual
    subgraph the minted op evaluates, the spelling ``modules_bound``/
    sites match (real corpus terms never contain a ``*synop*`` op)
    and the shape the denoted gate compares against the bank.

    The expansion is *structural recursion*, not a pass — a
    ``synopN_*`` node's fold LHS reintroduces ``synop(N-1)_*``
    leaves, which the recursive re-expansion of the instantiated LHS
    unfolds in turn, so depth-3+ patterns bottom out at pure manual
    ops with no fixpoint driver needed.
    """
    shipped = core_shipped() if shipped is None else shipped
    if not isinstance(pat, Op):
        return pat
    args = tuple(expand_synops(a, shipped) for a in pat.args)
    node = (
        Op.make(pat.op, *args, **dict(pat.attrs))
        if args != pat.args
        else pat
    )
    if node.op in shipped.names:
        fold = shipped.fold_for(node.op)[0]
        subst = _term_match(fold.rhs, node)
        if subst is not None:
            return expand_synops(
                instantiate_pattern(fold.lhs, subst), shipped
            )
    return node


def synop_ops(pat: Any, shipped: ShippedPrims | None = None) -> set:
    """Shipped ``*synop*`` op names a pattern composes over."""
    shipped = core_shipped() if shipped is None else shipped
    return pat_ops(pat) & shipped.names


def synop_vocab(shipped: ShippedPrims | None = None) -> dict[str, int]:
    """``{synop op: denoted op-count}`` — the frontier's vocab weights.

    A seated ``*synop*`` enode abbreviates the manual subgraph its
    ``expand`` premise spells: pricing the op at that denotation's
    op-count (``_n_ops(expand.rhs)`` — the manual spelling, already
    re-expanded through the deeper tiers for ``synopN_*``) restores
    the frontier score the shape would have earned unseated.  Without
    the map a ``OP(synop2_*, v)`` skeleton scores ``rows * 2`` — the
    bottom of ``rows * ops`` — and the beam cut drops every
    synop-carrying candidate before the heap reaches it.
    """
    shipped = core_shipped() if shipped is None else shipped
    return {
        name: _n_ops(shipped.fold_for(name)[1].rhs)
        for name in shipped.names
    }


def sub_alphas(pat: Any) -> set:
    """Alpha-normal keys of *pat*'s proper (non-root) Op subterms.

    The loop-closure probe's vocabulary: a deeper mined shape whose
    proper subterm alpha-matches a minted ``*synop*``'s denotation
    could be spelled *through* that synop — the composition the
    mint-over-minted phase would exploit (measured, not wired).
    """
    out: set[str] = set()

    def go(t: Any) -> None:
        for a in t.args:
            if isinstance(a, Op):
                out.add(_alpha_norm(a))
                go(a)

    if isinstance(pat, Op):
        go(pat)
    return out


# ---------------------------------------------------------------------------
#  the corpus helpers — bind domains and denotation naming
# ---------------------------------------------------------------------------


def op_subterms(roots: dict) -> dict:
    """``name -> [Op subterm]`` — each module's bind domain."""
    from mlopt_discovery.census import _iter_subterms

    return {
        n: [s for s in _iter_subterms(r) if isinstance(s, Op)]
        for n, r in roots.items()
    }


def modules_bound(pat: Any, subs_by_mod: dict) -> set:
    """Return module names holding at least one subterm *pat* binds.

    The mint's binding granularity: ``_term_match`` counted per
    module, so a pattern covering five sites inside one module still
    counts once.
    """
    return {
        n
        for n, subs in subs_by_mod.items()
        if any(_term_match(pat, s) is not None for s in subs)
    }


def bare_call(t: Any) -> bool:
    """Whether *t* is a fused call ``f(v0, …)`` over metavar args."""
    return (
        isinstance(t, Op)
        and bool(t.args)
        and all(isinstance(a, str) for a in t.args)
    )


def denoted_alphas(
    rules: Any,
    *,
    allowed_ops: frozenset | None = None,
    extra_alphas: Any = (),
) -> frozenset:
    """Alpha-normal keys of spellings a bank op already denotes.

    For every rule with a bare fused call on one side — ``manual ->
    f(v*)`` folds and ``f(v*) -> manual`` expansions alike — the
    *other* side is that op's spelled denotation.  *allowed_ops*
    restricts the call ops that count (a domain whose sink cannot
    lower the op has no use for its name — minting a primitive for
    the shape is a NEW name for the domain, not a second one);
    ``None`` accepts every call (the denotation-saturated domain).
    *extra_alphas* carries already-computed keys — e.g. the
    handler-table ops' declared patterns.  A mined shape in the
    result is already *named* — minting ``<prefix>_*`` for it would
    be a second name for an existing fused op, not a new primitive,
    so the mint skips it and the funnel records the exclusion.
    """
    denoted: set[str] = set()
    for r in rules:
        lhs_call = bare_call(r.lhs) and (
            allowed_ops is None or r.lhs.op in allowed_ops
        )
        rhs_call = bare_call(r.rhs) and (
            allowed_ops is None or r.rhs.op in allowed_ops
        )
        if lhs_call:
            denoted.add(_alpha_norm(r.rhs))
        if rhs_call:
            denoted.add(_alpha_norm(r.lhs))
    denoted.update(extra_alphas)
    return frozenset(denoted)


# ---------------------------------------------------------------------------
#  the denotation — spec rendering, freezing, evaluation
# ---------------------------------------------------------------------------


def pat_spec(pat: Any) -> Any:
    """Render an ``Op`` pattern as a genkernel spec tuple.

    ``(op, *child_specs[, attrs])`` with metavar leaves *kept by
    name* (``meta_arena._spec_of`` would re-canonicalize them to
    ``X1``… — the mint needs the pattern's own names so the bound
    sites' subst keys line up) and attr metavariables kept as their
    string names inside the attr dict.
    """
    if isinstance(pat, Op):
        parts = [pat_spec(a) for a in pat.args]
        if pat.attrs:
            parts.append(dict(pat.attrs))
        return (pat.op, *parts)
    if isinstance(pat, Const):
        return pat.value
    return pat  # metavar leaf — the name IS the binding key


def pat_ops(pat: Any) -> set:
    """Op names inside a pattern."""
    if not isinstance(pat, Op):
        return set()
    return {pat.op} | set().union(*(pat_ops(a) for a in pat.args))


def spec_bind(spec: Any, attrs: dict, attr_mvars: tuple) -> Any:
    """Instantiate a spec's attr-metavar dict values under *attrs*.

    The minted op call carries its attr metavariables as real op
    attrs (``<prefix>_*(v0, a0=<bound>)``); at eval time the spelled
    denotation needs those values back inside the spec's attr dicts —
    a dict value equal to an attr-mvar name resolves through the
    call's attrs, everything else passes through literal.
    """
    if isinstance(spec, (tuple, list)):
        out = []
        for e in spec:
            if isinstance(e, dict):
                out.append(
                    {
                        k: (
                            attrs.get(v, v)
                            if isinstance(v, str) and v in attr_mvars
                            else v
                        )
                        for k, v in e.items()
                    }
                )
            else:
                out.append(spec_bind(e, attrs, attr_mvars))
        return tuple(out) if isinstance(spec, tuple) else out
    return spec


def freeze_spec(spec: Any, subst: dict, attr_mvars: tuple) -> Any:
    """Instantiate a spec's attr metavars under one bound *subst*.

    ``mint_handlers`` needs a concrete spec — the representative
    bound site's ``$attr:`` values freeze the metavar attr slots
    (leaf metavars stay names — only the attr dicts concretize).
    The runtime binding keeps substituting per call, so the op's
    denotation never depends on the frozen choice; the frozen body is
    the grammar's provenance measurement only.
    """
    attrs = {
        k[len("$attr:") :]: v
        for k, v in subst.items()
        if k.startswith("$attr:")
    }
    return spec_bind(spec, attrs, attr_mvars)


def make_denotation(
    spec: Any,
    leaf_mvars: tuple,
    attr_mvars: tuple,
    bindings: dict,
) -> Any:
    """Return the minted op's binding — its denotation, spelled out.

    ``fn(*args, **attrs) = eval(spec, {mvar: arg}, bindings)`` — the
    op IS the manual subgraph: equal to the fold's LHS at every bound
    site by construction, so the oracle/sweep/cert legs all measure
    the definitional truth.  Attr metavariables ride the call's own
    attrs back into the spec.  *bindings* is the domain's op table —
    the same table the op registers onto.
    """

    def fn(*args: Any, **attrs: Any) -> Any:
        env = dict(zip(leaf_mvars, args, strict=True))
        return _eval_spec(
            spec_bind(spec, attrs, attr_mvars), env, bindings
        )

    return fn


def register_denotation(
    name: str, call: Any, denotation: Any, bindings: dict
) -> None:
    """Seat *name*'s spelled denotation onto the domain *bindings*.

    The generic ``register_shipped`` body: *call* is the
    ``<prefix>_*(v*)`` pattern (its positional args are the leaf
    metavars, its attr names the attr metavars) and *denotation* the
    pattern the op unfolds to — the shipped ``<prefix>_*.expand``
    premise's RHS.  A domain without its own seater calls this once
    per shipped op; the emitted mints' own registrations are the
    same mechanism at mint time.
    """
    leaf_mvars = tuple(a for a in call.args if isinstance(a, str))
    attr_mvars = tuple(call.attrs)
    bindings[name] = make_denotation(
        pat_spec(denotation), leaf_mvars, attr_mvars, bindings
    )


def prim_domain(sites: list, cap: int) -> list:
    """``{meta: shape}`` dicts read off bound sites' inferred shapes.

    The handler-verify domain is the *bound* domain — each site's
    metavar bindings carry real terms whose shapes ``_shape_of``
    infers (composite bound values included).  A site whose bound
    terms leave any dim undecidable is skipped — never a guessed
    shape.  Attr-mvar keys (``$attr:``) are not domain entries.
    """
    from mlopt_core.typing import _INVALID, _shape_of

    out: list = []
    for m, _s in sites:
        shapes: dict[str, Any] = {}
        ok = True
        for mv, t in m.items():
            if mv.startswith("$attr:"):
                continue
            sh = _shape_of(t)
            if (
                sh is _INVALID
                or not isinstance(sh, tuple)
                or any(not isinstance(d, int) for d in sh)
            ):
                ok = False
                break
            shapes[mv] = sh
        if ok and shapes:
            out.append(shapes)
        if len(out) >= cap:
            break
    return out


# ---------------------------------------------------------------------------
#  the pool — mine -> gate -> mint -> measure
# ---------------------------------------------------------------------------


def _vocab_fragments(
    vocab_dir: Any, shipped: ShippedPrims, denoted: frozenset
) -> tuple:
    """Load a fragment directory into the pool's mint inputs.

    Phase 3 (plan 0033): the directory's ``prims.yaml`` joins the
    shipped table (:func:`merge_shipped` + :func:`shipped_from_vocab`
    — a prim the mint emitted last round is a mintable atom this
    round), its ``prim:*`` folds and record alphas join the *denoted*
    gate (an already-vocab'd shape is named, not re-minted), and its
    ``families.minted_families`` rows decode into ``extra_families``
    for the handler grammar.  Returns ``(shipped, denoted,
    extra_names, extra_families)`` — *extra_names* is the subset the
    domain's own seater cannot know (the bootstrap seats just those).
    """
    from mlopt_discovery import vocabdir as _vd

    frag = _vd.load_vocab(vocab_dir, fresh=True)
    vocab_table = shipped_from_vocab(frag)
    extra_names = frozenset(set(vocab_table.names) - set(shipped.names))
    shipped = merge_shipped(shipped, vocab_table)
    extra_families = _vd.vocab_extra_families(frag)
    prim_folds = [r for r in frag.laws if r.name.startswith("prim:")]
    # The gate compares the *expanded manual* alpha (``xnorm``);
    # a depth-N record's fold LHS is the synop-carrying skeleton,
    # so the record's own ``alpha`` field — the mined spelling —
    # is the honest denoted key, with the fold LHSes alongside.
    prim_alphas = {
        rec["alpha"] for rec in frag.prims.values() if rec.get("alpha")
    }
    if prim_folds or prim_alphas:
        denoted = (
            denoted
            | denoted_alphas(prim_folds)
            | frozenset(prim_alphas)
        )
    return shipped, denoted, extra_names, extra_families


def _family_retry(
    name: str,
    item: dict,
    rep: dict,
    handlers: dict,
    *,
    budget: int,
    base_extra_families: tuple,
    family_rows: list,
    stats: dict,
    hb: dict,
) -> tuple:
    """Run the refused-spec meta-loop: mint a family, retry the body.

    ``family_synth.mint_families`` enumerates the descriptor space
    over the spec's movable ops and fp64-verifies on the same
    bound-site domain; survivors inject as ``extra_families`` for a
    second ``mint_handlers`` pass.  Returns ``(rep, handlers,
    fam_names, fam_retry)`` — *fam_retry* is the retry's own report
    row (or the ``no-family`` wall record).
    """
    from mlopt_discovery import family_synth as _fs
    from mlopt_discovery.handler_synth import mint_handlers

    fams, frep = _fs.mint_families([item], bindings=hb)
    fam_names = [f["name"] for f in fams]
    if not fams:
        stats["families_refused"] += 1
        return (
            rep,
            handlers,
            fam_names,
            {
                "verdict": "no-family",
                "searched": frep["searched"],
                "budget": frep["budget"],
            },
        )
    stats["families_minted"] += 1
    seen_fam = {f["name"] for f in base_extra_families}
    retry_fams = list(base_extra_families) + [
        f for f in fams if f["name"] not in seen_fam
    ]
    for f in fams:
        family_rows.append({**_fs.family_record(f), "covers": [name]})
    _h2, rep2 = mint_handlers(
        [item],
        budget=budget,
        extra_families=tuple(retry_fams),
        bindings=hb,
    )
    fam_retry = rep2[0]
    if fam_retry["verdict"] == "minted":
        return fam_retry, _h2, fam_names, fam_retry
    return rep, handlers, fam_names, fam_retry


def _refused_handler_row(
    name: str,
    item: dict,
    rep: dict,
    *,
    fam_names: list,
    fam_retry: dict | None,
) -> dict:
    """Build the prim record's ``handler`` row for a refused shape.

    Carries the frozen spec + measured bound-site domain the refusal
    ran against — the row reproduces offline — plus the meta-loop's
    own accounting (``family_minted`` / ``retry`` /
    ``family_refused``).
    """
    hrow = {
        "verdict": "refused",
        "tried": rep.get("tried"),
        "note": rep.get("note"),
        # inf means every tried candidate raised or shape-mismatched
        # — serialized as a string so the report JSON stays strict.
        "best_err": (
            rep.get("best_err")
            if rep.get("best_err") != float("inf")
            else "inf"
        ),
        "spec": rep.get("spec"),
        "domain": item["domain"],
    }
    if fam_names:
        # descriptors minted but no injected body verified — the
        # family exists, the body wall stands; the retry's own
        # refusal rides along.
        retry = fam_retry or {}
        hrow["family_minted"] = ",".join(fam_names)
        hrow["retry"] = {
            "verdict": retry.get("verdict"),
            "tried": retry.get("tried"),
            "note": retry.get("note"),
        }
    elif fam_retry is not None:
        hrow["family_refused"] = (
            "no family verified within "
            f"budget={fam_retry['budget']} "
            f"(searched {fam_retry['searched']})"
        )
    return hrow


def _refusal_ledger_row(name: str, item: dict, hrow: dict) -> dict:
    """Build the ``families.yaml`` refused-spec row for one shape.

    Refused-spec analysis (plan 0033 phase 3): the refusal is itself
    vocabulary data — the ledger row carries the frozen spec +
    measured domain so a later mint or audit can replay the wall.
    """
    return {
        "key": name,
        "op": name,
        "spec": item["spec"],
        "domain": item["domain"],
        "verdict": "refused",
        "tried": hrow.get("tried"),
        "best_err": hrow.get("best_err"),
        "note": hrow.get("note"),
        "family_minted": hrow.get("family_minted"),
        "family_refused": hrow.get("family_refused"),
        "retry": hrow.get("retry"),
    }


def _pool_handler_row(
    name: str,
    item: dict,
    *,
    budget: int,
    family_mint: bool,
    base_extra_families: tuple,
    family_rows: list,
    refusal_rows: list,
    stats: dict,
    hb: dict,
) -> dict:
    """Run the handler-grammar stage for one gated shape.

    ``mint_handlers`` on the measured ``(spec, domain)`` pair; a
    refusal under *family_mint* feeds ``family_synth.mint_families``
    and retries with the survivors injected — the meta-loop.  The
    returned row is the prim record's ``handler`` field; minted
    family descriptors append to *family_rows* and a surviving
    refusal appends its replayable ledger row to *refusal_rows*
    (consumed only when ``vocab_dir`` emission is on).
    """
    from mlopt_discovery.handler_synth import (
        handler_record,
        mint_handlers,
    )

    _handlers, report = mint_handlers(
        [item],
        budget=budget,
        extra_families=base_extra_families or None,
        bindings=hb,
    )
    rep = report[0]
    fam_names: list[str] = []
    fam_retry: dict | None = None
    if rep["verdict"] == "refused" and family_mint:
        rep, _handlers, fam_names, fam_retry = _family_retry(
            name,
            item,
            rep,
            _handlers,
            budget=budget,
            base_extra_families=base_extra_families,
            family_rows=family_rows,
            stats=stats,
            hb=hb,
        )
    if rep["verdict"] == "minted":
        stats["minted"] += 1
        tag = rep["tag"]
        hrow = {
            "verdict": "minted",
            "family": rep["family"],
            "tried": rep["tried"],
            "max_err": rep["max_err"],
            "record": handler_record(tag, _handlers[tag]),
        }
        if fam_names:
            # the meta-loop carried this shape: the minted descriptor
            # that won the retry (a shipped body winning instead is
            # recorded, not credited).
            hrow["family_minted"] = (
                rep["family"].split(":", 1)[1]
                if rep["family"].startswith("family:")
                else ",".join(fam_names)
            )
        return hrow
    stats["refused"] += 1
    hrow = _refused_handler_row(
        name, item, rep, fam_names=fam_names, fam_retry=fam_retry
    )
    refusal_rows.append(_refusal_ledger_row(name, item, hrow))
    return hrow


def _emit_pool_fragments(
    vocab_dir: Any,
    cands: list,
    meta: dict,
    *,
    family_rows: list,
    refusal_rows: list,
    stats: dict,
) -> None:
    """Write the pool's fragments back to *vocab_dir* (phase 3).

    Every pooled mint emits op row + fold law + handler + prim
    record + denotation/handler Models + ``fold:<name>`` schema via
    one batched :func:`vocabdir.emit_prims`; the minted-family
    descriptors and refused-spec ledger rows land in
    ``families.yaml`` through :func:`vocabdir.emit_family_rows`.
    """
    from mlopt_discovery import vocabdir as _vd

    emit_items = [
        (
            meta[i]["prim"]["synop"],
            minted_prim_record(
                meta[i]["prim"]["synop"],
                cand.as_rule(),
                meta[i]["prim"],
            ),
        )
        for i, cand in enumerate(cands)
    ]
    _vd.emit_prims(vocab_dir, emit_items)
    if family_rows or refusal_rows:
        _vd.emit_family_rows(
            vocab_dir, families=family_rows, refusals=refusal_rows
        )
    stats["vocab_emitted"] = len(emit_items)
    stats["vocab_families"] = len(family_rows)
    stats["vocab_refusals"] = len(refusal_rows)


def prim_pool(
    roots: dict,
    k: int,
    *,
    bindings: dict,
    eval_pair: Callable[[Any, Any], str],
    denoted: frozenset,
    prefix: str = "synop",
    handler_bindings: dict | None = None,
    register_shipped: Callable[[], None] | None = None,
    shipped: ShippedPrims | None = None,
    family: str = PRIM_FAMILY,
    cap: int = PRIM_CAP,
    budget: int = PRIM_BUDGET,
    site_cap: int = PRIM_SITES,
    domain_cap: int = PRIM_DOMAIN,
    min_ops: int = PRIM_MIN_OPS,
    max_ops: int = PRIM_MAX_OPS,
    beam: int = PRIM_BEAM,
    max_expand: int = PRIM_MAX_EXPAND,
    bootstrap: bool = False,
    family_mint: bool = True,
    depth: int = PRIM_DEPTH,
    vocab: bool = True,
    vocab_dir: Any = None,
    emit_vocabulary: bool = True,
) -> tuple:
    """Return the primitive-mint pool: ``(candidates, meta, stats)``.

    For every mined shape binding >= *k* distinct modules and not
    denoted by a bank op (*denoted*), mint ``<prefix>_<hash>`` — a
    NEW op whose denotation is the manual subgraph itself — plus the
    fold ``pattern -> <prefix>_*(args)``, carrying the
    ``<prefix>_*.expand`` definitional premise so the cert leg's
    anchored derivation has the axiom the bare bank lacks.

    ``min_ops``/``max_ops``/``beam``/``max_expand`` bound the
    ``mine_candidates`` hunt; ``cap`` bounds the pool — under
    *vocab* each composition tier's own top-``gain`` slice (the
    stratified pool).  A deeper window re-finds the same skeleton
    under different bracketings — the alpha-dedup inside
    ``mine_candidates`` plus the denoted gate keep one name per
    distinct shape.

    ``bootstrap`` is the vocabulary bootstrap: the corpus e-graph is
    congruence-closed, then the shipped ``prim:<prefix>*`` folds are
    seated in a fixpoint loop (targeted ``apply_rule`` passes, each
    followed by congruence closure, until a pass fires nothing or
    *depth* passes run).  A deeper tier's fold LHS is a
    ``synop(N-1)_*``-carrying skeleton, so it can only match once
    the tier below seated — pass *n* enables depth *n*.  ``*synop*``
    enodes now sit in the e-classes of the manual shapes they name,
    so ``mine_candidates`` grows skeletons *through* them: a pooled
    pattern carrying a shipped depth-``d`` ``*synop*`` op is a
    **depth-(d+1)** primitive — ``<prefix>{d+1}_<hash>``.  Its fold
    fires only where the shipped folds already did (composition,
    honestly recorded), while its *denotation* — binding, expand
    premise, frozen handler spec, module-binding and denoted gates —
    is the shape re-expanded to manual ops (:func:`expand_synops`,
    structural recursion that bottoms out at pure manual ops at any
    depth), so *eval_pair* verifies the whole composition end-to-end
    and the emit gate measures the real bound sites.  The emitted op
    is again new — the fold is underivable by construction.  *depth*
    caps the deepest mintable tier — the seating loop applies only
    folds minting a synop strictly below it.  *register_shipped*
    seats the shipped ops' denotations onto *bindings* before
    seating — the emit gate's eval and every sink reading the table
    see the composed spelling end-to-end.

    ``vocab`` is the frontier fix: ``mine_candidates`` scores
    ``rows * ops`` on the *spelled* pattern, so a skeleton carrying
    a seated ``*synop*`` leaf — which abbreviates a multi-op
    denotation — ranks at the bottom of the heap and the beam cut
    discards every such child before it is ever popped.  With
    *vocab* the miner prices a carried op at its denotation's
    op-count (:func:`synop_vocab`), so the compressed shape competes
    at the score its spelled denotation would have earned — both in
    the beam and in the emitted gain ordering the *cap* slices.
    Under *vocab* the cap is also **stratified by composition
    tier**: each depth keeps its own top-gain slice of *cap*, so
    the depth-2 surface cannot crowd the deeper tier out of the
    pool.  A shape skipped by its tier's slice records
    ``tier_capped``.

    Per-shape funnel, each refusal recorded:

    * ``kbound`` — shapes binding >= k modules (the mint's surface);
      ``depth_dist`` is the qualified set's op-count histogram;
    * ``over_depth`` — skipped by the *depth* cap;
    * ``tier_capped`` — skipped because its composition tier's own
      top-gain slice of *cap* was already full;
    * ``denoted`` — skipped as already-named (a shipped fold or
      handler op spells the same denotation);
    * ``unevaluable`` — an op in the pattern has no binding, so the
      spelled denotation can never run;
    * ``gate_failed`` — the emit gate: no bound site evaluated
      ``equal`` (or a measured ``unequal``/``rhs-err``) — the verify
      requirement from the arm's contract;
    * ``no_domain`` — every bound site left a dim undecidable, so
      the handler stage had no measured domain (the op still mints
      on the definitional denotation — the fold's soundness never
      depends on the grammar);
    * ``minted`` / ``refused`` — ``handler_synth.mint_handlers``'
      grammar verdict on the bound-site domain (a verified body
      rides the record as provenance; a refusal keeps the
      definitional op and names the measured gap — a shape carrying
      ops no grammar family claims is a recorded family wall, not a
      silent drop).

    ``family_mint`` closes the meta-loop on the refusal: the refused
    ``(spec, domain)`` pair feeds ``family_synth.mint_families`` —
    the spec class mints its own family descriptor, fp64-verified on
    the same bound-site domain — and the body mint retries with the
    survivors injected as ``extra_families``.  Per shape the record
    carries ``family_minted: <fam_name>`` or ``family_refused``;
    ``families_minted`` / ``families_refused`` count the stage in
    the pool funnel.

    ``vocab_dir`` is the phase-3 fragment loop (plan 0033): when set,
    the directory's ``prims.yaml`` joins the shipped table
    (:func:`merge_shipped` + :func:`shipped_from_vocab` — a prim the
    mint emitted last round is a mintable atom this round), its
    ``prim:*`` folds join the *denoted* gate (an already-vocab'd
    shape is named, not re-minted), and its ``families.minted_families``
    rows seed ``extra_families`` for the handler grammar.  With
    *emit_vocabulary* (default) every pooled mint then writes its
    fragments back — op + fold law + handler + prim record +
    denotation/handler Models + ``fold:`` schema — through
    ``vocabdir.emit_prims``, plus the minted-family descriptors and
    refused-spec ledger rows into ``families.yaml``.  ``vocab_dir=None``
    is the pre-phase-3 behaviour: measure only, write nothing.

    Post-mint, ``nested`` records the loop-closure measurement:
    every pooled shape whose own proper subterm alpha-matches
    another minted synop's denotation — the cheap read on whether
    mining over a synop-aware corpus would compose.
    """
    from mlopt_discovery.pipeline import Proposal

    hb = bindings if handler_bindings is None else handler_bindings
    shipped = core_shipped() if shipped is None else shipped
    # Phase 3 (plan 0033): the fragment directory is vocabulary data —
    # its prims join the shipped table, its ``prim:*`` folds join the
    # denoted gate, its minted families seed the handler grammar, and
    # the pool's output emits back as fragments.
    vocab_extra_names: frozenset = frozenset()
    base_extra_families: tuple = ()
    family_rows: list = []
    refusal_rows: list = []
    if vocab_dir is not None:
        (
            shipped,
            denoted,
            vocab_extra_names,
            base_extra_families,
        ) = _vocab_fragments(vocab_dir, shipped, denoted)
    subs_by_mod = op_subterms(roots)
    all_subs = [s for subs in subs_by_mod.values() for s in subs]
    eg = EGraph()
    for r in roots.values():
        eg.add_term(r)
    synop_fires: dict[str, int] = {}
    synop_passes = 0
    if bootstrap:
        if register_shipped is not None:
            # the shipped ops' denotations register too — the emit
            # gate's eval and every sink reading the table see the
            # composed spelling end-to-end.
            register_shipped()
        if vocab_extra_names:
            # the directory's own prims — ``register_shipped`` only
            # knows the domain's shipped table; seat just the extras
            # it cannot (missing names only, domain seater wins).
            seat_shipped(shipped, bindings, names=vocab_extra_names)
        eg.rebuild()
        # A deeper tier's fold LHS is a skeleton over ``synop(N-1)_*``
        # enodes — it can only match once the tier below seated — so
        # the seating loops: one targeted ``apply_rule`` pass per
        # tier, congruence-closed between passes, until a pass fires
        # nothing (fixpoint) or ``depth`` passes ran (each pass can
        # seat at most one deeper tier).  Folds minting a synop of
        # depth >= the cap are skipped outright — a ``synopN_*``
        # enode could only be embedded by a depth-(N+1) mint, which
        # the cap already rules out.
        seatable = [
            f
            for f in shipped.folds
            if synop_depth(minted.synop_name(f.name)) < depth
        ]
        while synop_passes < depth:
            synop_passes += 1
            fired = 0
            for f in seatable:
                before = eg.rule_fires.get(f.name, 0)
                eg.apply_rule(f, 0)
                n = eg.rule_fires.get(f.name, 0) - before
                if n:
                    synop_fires[f.name] = synop_fires.get(f.name, 0) + n
                    fired += n
            eg.rebuild()
            if not fired:
                break
    mined_all = mine_candidates(
        eg,
        min_uses=k,
        min_ops=min_ops,
        max_ops=max_ops,
        beam=beam,
        max_expand=max_expand,
        vocab=synop_vocab(shipped) if vocab else None,
    )
    stats: dict[str, Any] = {
        "k": k,
        "prefix": prefix,
        "min_ops": min_ops,
        "max_ops": max_ops,
        "cap": cap,
        "beam": beam,
        "max_expand": max_expand,
        "mined_total": len(mined_all),
        "mined": 0,
        "kbound": 0,
        "depth_dist": {},
        "denoted": 0,
        "unevaluable": 0,
        "gate_failed": 0,
        "no_domain": 0,
        "minted": 0,
        "refused": 0,
        "pool": 0,
        "scan_capped": False,
        "nested": [],
        "bootstrap": bool(bootstrap),
        # the minted-depth cap and the shapes it excluded.
        "prim_depth": depth,
        "over_depth": 0,
        # the refused-spec meta-loop: refused shapes whose spec
        # class minted a family descriptor vs those the meta-grammar
        # refused too.
        "family_mint": bool(family_mint),
        "families_minted": 0,
        "families_refused": 0,
        # the bootstrap seating: shipped folds that fired on the
        # corpus, the enode merges they contributed, and how many
        # fixpoint passes the seating loop ran.
        "synop_folds": len(synop_fires),
        "synop_fires": sum(synop_fires.values()),
        "synop_passes": synop_passes,
        # the frontier pricing fix: carried ``*synop*`` leaves score
        # at their denotation's op-count.
        "vocab": bool(vocab),
        # pooled primitives by composition depth — the loop-closure
        # count (``depth2`` kept as the depth-2 slice).
        "depth2": 0,
        "depth_counts": {},
        # the stratified pool: shapes skipped because their
        # composition tier's own top-gain slice of ``cap`` was
        # already full.
        "tier_capped": 0,
    }
    op_counts: dict[int, int] = {}
    cands: list = []
    meta: dict[int, dict] = {}
    tier_minted: dict[int, int] = {}
    for cand in mined_all:
        if not vocab and len(cands) >= cap:
            stats["scan_capped"] = True
            break
        stats["mined"] += 1
        pat = cand.pattern
        if not isinstance(pat, Op) or not cand.leaf_mvars:
            continue
        embeds_shipped = sorted(synop_ops(pat, shipped))
        # the minted op's composition depth: 1 + the deepest shipped
        # synop the skeleton carries (``synop_`` = depth 1,
        # ``synopN_`` = depth N).
        d = 1 + max((synop_depth(o) for o in embeds_shipped), default=0)
        if d > depth:
            # the depth cap — unreachable under the gated seating
            # (no deep synop enodes exist to embed) unless the
            # corpus itself spells a deeper op; recorded either way.
            stats["over_depth"] += 1
            continue
        if vocab and tier_minted.get(d, 0) >= cap:
            # the tier's own top-gain slice is full — skipped before
            # the bind/eval gates, honestly recorded.
            stats["tier_capped"] += 1
            continue
        # the denotation spelling: shipped synops re-expanded to
        # manual ops — what the op means, what real subterms bind,
        # what the denoted gate names.
        xpat = expand_synops(pat, shipped)
        mods = modules_bound(xpat, subs_by_mod)
        if len(mods) < k:
            continue
        stats["kbound"] += 1
        op_counts[cand.op_count] = op_counts.get(cand.op_count, 0) + 1
        xnorm = _alpha_norm(xpat)
        if xnorm in denoted:
            stats["denoted"] += 1
            continue
        if any(
            bindings.get(op) is None
            for op in pat_ops(pat) | pat_ops(xpat)
        ):
            stats["unevaluable"] += 1
            continue
        # the honest name: the shape's own hash, not a coined label —
        # ``<prefix>{d}_*`` when the fold composes over a
        # depth-(``d-1``) shipped synop.
        norm = _alpha_norm(pat)
        h = hashlib.sha256(norm.encode()).hexdigest()[:10]
        name = f"{prefix}_{h}" if d == 1 else f"{prefix}{d}_{h}"
        call_attrs: dict[str, Any] = {a: a for a in cand.attr_mvars}
        call = Op.make(name, *cand.leaf_mvars, **call_attrs)
        spec = pat_spec(xpat)
        op_def(
            name,
            len(cand.leaf_mvars),
            1,
            law="synthesized primitive — denotation is the mined "
            f"manual subgraph {xnorm}",
        )
        # denotation = the spelled spec — the binding table is the
        # op's semantics (ADR 0007: the executor table IS the
        # meaning); post-hoc registration reaches every sink seated
        # on the shared ambient dict.
        bindings[name] = make_denotation(
            spec, cand.leaf_mvars, cand.attr_mvars, bindings
        )
        sites = [
            (m, s)
            for s in all_subs
            if (m := _term_match(xpat, s)) is not None
        ][:site_cap]
        # verify — the arm's honesty gate: the op's denotation must
        # equal the spelled pattern fp64 on bound sites.  By
        # construction it does (the binding IS the spelling) — the
        # measurement is still run, never assumed.
        equal = bad = 0
        for m, _s in sites:
            outcome = eval_pair(
                instantiate_pattern(pat, m),
                instantiate_pattern(call, m),
            )
            equal += outcome == "equal"
            bad += outcome in ("unequal", "rhs-err")
        if not equal or bad:
            stats["gate_failed"] += 1
            continue
        domain = prim_domain(sites, domain_cap)
        if not domain:
            stats["no_domain"] += 1
            hrow: dict[str, Any] = {"verdict": "skipped-no-domain"}
        else:
            item = {
                "spec": freeze_spec(spec, sites[0][0], cand.attr_mvars),
                "domain": domain,
            }
            hrow = _pool_handler_row(
                name,
                item,
                budget=budget,
                family_mint=family_mint,
                base_extra_families=base_extra_families,
                family_rows=family_rows,
                refusal_rows=refusal_rows,
                stats=stats,
                hb=hb,
            )
        # the definitional premise: the cert leg's anchored axiom
        # set unions it when the bare bank cannot derive the fold —
        # the fusion-schema mint's own channel for a denotation the
        # bank lacks.  The novelty sieve never sees it (its premise
        # bank is the shipped surface minus alpha-siblings), so the
        # fold stays underivable there BY CONSTRUCTION.  The premise
        # unfolds to the FULL denotation — for a depth-N mint the
        # synop-carrying LHS re-connects through the shipped folds'
        # own expansions (an N-step anchored derivation).
        expand = Rewrite(
            f"{name}.expand",
            call,
            xpat,
            law=f"{name} unfolds to its mined denotation — the "
            "definitional premise the cert leg anchors on",
        )
        prop = Proposal(
            name=f"prim:{name}",
            lhs=pat,
            rhs=call,
            family=family,
            sources=(
                "prim-mint",
                f"mods:{len(mods)}",
                f"inst:{cand.n_instances}",
                f"gain:{cand.gain}",
                f"handler:{hrow['verdict']}",
                *((f"depth:{d}",) if d > 1 else ()),
            ),
            premises=(expand,),
        )
        i = len(cands)
        tier_minted[d] = tier_minted.get(d, 0) + 1
        cands.append(
            Candidate(
                strategy="prim",
                label=prop.name,
                lhs=pat,
                rhs=call,
                note=(
                    f"primitive {name} — a manual shape shared by "
                    f"{len(mods)} modules, undenoted in the bank"
                    + (
                        f"; composes over shipped "
                        f"{'/'.join(embeds_shipped)}"
                        if embeds_shipped
                        else ""
                    )
                ),
                rule=prop.as_rule(),
            )
        )
        prim_rec: dict[str, Any] = {
            "synop": name,
            "alpha": xnorm,
            "depth": d,
            "ops": cand.op_count,
            "opset": sorted(pat_ops(pat)),
            "mods": sorted(mods),
            "sites": len(sites),
            "equal_sites": equal,
            "handler": hrow,
        }
        if norm != xnorm:
            # the fold's own LHS spelling — the synop-carrying
            # skeleton the minted op compresses.
            prim_rec["alpha_fold"] = norm
            prim_rec["embeds_shipped"] = embeds_shipped
        meta[i] = {
            "family": family,
            "sources": list(prop.sources),
            "mods": len(mods),
            "prim": prim_rec,
        }
        stats["depth_counts"][str(d)] = (
            stats["depth_counts"].get(str(d), 0) + 1
        )
        if d == 2:
            stats["depth2"] += 1
    stats["pool"] = len(cands)
    if vocab and stats["tier_capped"]:
        # a tier's slice truncated candidates the scan never
        # reached — the stratified analogue of ``scan_capped``.
        stats["scan_capped"] = True
    stats["depth_dist"] = {
        str(n): c for n, c in sorted(op_counts.items())
    }
    # Loop-closure probe: which pooled shapes' *denotations* carry a
    # same-run minted synop's denotation as a proper subterm —
    # measured on the expanded spelling so a deeper fold's embedded
    # inner mints still register.  (The ``depth_counts`` /
    # ``embeds_shipped`` record the composition over *shipped*
    # synops — the bootstrap's own surface.)  Measured over the
    # pooled set only; the scan may be cap-truncated above.
    alpha_name = {
        meta[i]["prim"]["alpha"]: meta[i]["prim"]["synop"] for i in meta
    }
    for i, rec in meta.items():
        p = rec["prim"]
        embeds = sorted(
            {
                alpha_name[a]
                for a in sub_alphas(
                    expand_synops(cands[i].lhs, shipped)
                )
                if a in alpha_name
            }
            - {p["synop"]}
        )
        if embeds:
            p["embeds"] = embeds
            stats["nested"].append(
                {"outer": p["synop"], "inner": embeds}
            )
    if vocab_dir is not None and emit_vocabulary and cands:
        _emit_pool_fragments(
            vocab_dir,
            cands,
            meta,
            family_rows=family_rows,
            refusal_rows=refusal_rows,
            stats=stats,
        )
    return cands, meta, stats

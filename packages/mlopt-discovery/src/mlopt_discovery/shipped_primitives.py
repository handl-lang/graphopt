"""Shipped machine-minted primitives — the ``synop_*`` denotation registry.

The ``overnight.py --corpus zoo-prim`` arm minted 24 primitive ops
for manual subgraph shapes recurring across >=2 held-out zoo modules
and denoted by no bank op; 21 cleared the full admission gauntlet
(``docs/reports/results_overnight_prim_v2.json`` — every stage green,
all ``machine-original``).  The ``--prim-bootstrap`` arm then minted
one loop deeper: 2 depth-2 ``synop2_*`` primitives whose fold LHS is
a synop-carrying skeleton — a manual shape spelled *through* a
shipped ``synop_*`` op — admitted the same way
(``docs/reports/results_overnight_prim_bootstrap.json``).  The minted
objects were runtime-only — the ``synop_*`` op registration, the
torch denotation and the ``prim:synop_*`` fold law all died with the
driver process.

Where the family lives now
--------------------------

The 23 fold laws and their definitional ``synop_*``/``synop2_*``
``.expand`` premises are **core-shipped, default-on**: the verbatim
records embed in :mod:`mlopt_core.laws.minted` (``_PRIM_RECORDS`` +
``_PRIM2_RECORDS`` — laws, not semantics; the patterns are torch-free
data) and ride ``MINTED_RULES`` into ``ALL_RULES`` / ``DEFAULT`` — so
``search(rules=DEFAULT)`` fires them with no opt-in.  This module is
what remains domain-side, exactly the piece a torch-free core cannot
ship: the **denotation**.  A ``synop_*`` op's meaning is the torch
binding that evaluates the spelled subgraph (ADR 0007: the executor
table is the meaning) — :func:`register` seats one binding per op on
the ambient ``mlopt_torch.torch_bridge._IR_TO_TORCH`` table, so every
``OpTable.full()``-backed ``TorchSink`` (existing instances included
— the table IS the ambient dict) can lower it.  The mint's fp64 site
checks all measured ``equal`` — the fold is sound by construction.

The depth-2 ops' denotation is the skeleton **fully re-expanded to
manual ops** — the embedded ``synop2_*.expand`` premise's RHS is the
mint's recorded ``xpat`` (``_expand_synops`` over the fold LHS), so
the binding spells ``add``/``mul`` alone and stands even where the
embedded depth-1 op is unbound.  (The uniform rule: an op's
denotation spec is its premise's RHS — depth-1 premises end at the
fold LHS, depth-2 premises end at the expansion.)

What ships per primitive:

* the **op declaration** — :func:`mlopt_core.ir.op_def` — travels
  with the laws: materializing the folds registers the
  ``synop_<hash>`` / ``synop2_<hash>`` generators (the mint's name IS
  the honest one — ``sha256(alpha-normal pattern)[:10]``, a content
  hash);
* the **denotation** — this module's :func:`register`: one ambient
  torch binding per op, evaluating the spelled pattern itself;
* the **fold law** — ``prim:synop_<hash>`` / ``prim:synop2_<hash>`` —
  materialized by :func:`mlopt_core.laws.minted.prim_folds`; in
  ``DEFAULT``;
* the **definitional premise** — ``synop_<hash>.expand`` /
  ``synop2_<hash>.expand`` (``synop_call -> denotation``), the axiom
  the cert leg's anchored derivation unions in when the bare bank
  cannot derive the fold — materialized by ``minted.prim_premises``
  and attached to each fold as ``rule.premises`` (the convention
  ``pipeline.Proposal.as_rule`` mints).  For a depth-2 fold the
  premise RHS is the *expanded* manual denotation: the anchored
  ``rhs -> lhs`` derivation is 2 steps — the premise, then the
  shipped depth-1 fold re-folding the embedded shape.

Deployment contract
-------------------

::

    from mlopt_discovery import shipped_primitives as sp
    from mlopt_orchestrator.optimize import search, lower
    from mlopt_torch.adapters import TorchSink, TorchSource

    sp.register()                       # torch denotations
    res = search(m, x, source=TorchSource())   # shipped DEFAULT
    out = lower(res, x, sink=TorchSink(), verify=True)

``register()`` is the deliverability half — without it saturation
still fires the folds (pattern matching needs no op table) but
extraction's ``supported_ops`` bound correctly refuses a form no
sink can lower.  An unbound ``synop_*`` prices like any unknown op
(the elementwise FLOP default plus launch — a fusion-region
boundary, never 0, never ``+inf`` by default), so a sink with the
bindings sees the honest win: one fused primitive vs the manual
subgraph's kernel count.  :func:`prim_default` stays for callers
that want the folds composed onto an *explicit* base — onto
``DEFAULT_RULES`` it is now a no-op union.

Provenance
----------

Core's ``mlopt_core.laws.provenance.MACHINE_SHIPPED`` covers the
fold (``prim:synop_*`` / ``prim:synop2_*``) and premise
(``synop_*.expand`` / ``synop2_*.expand``) names.
:data:`MACHINE_MINTED` here is the domain-side ledger for what
*this* module ships — the ``synop_*`` op names whose torch
denotations it registers.  :func:`law_provenance` unions the two.
Each record (``minted.prim_records()`` — the frozen mint artifacts)
also carries the mint's own measurements — ``mods`` (the bound
modules), ``sites``/``equal_sites`` (the fp64 denotation check) and
``handler`` (the grammar minter's verdict — ``"minted"`` where a
handler body verified, ``"refused"`` where the primitive ships on
its definitional denotation alone); the depth-2 records add
``alpha_fold``/``embeds_shipped``/``embeds`` — the skeleton's
spelling and the shipped ops it composes over.
"""

from __future__ import annotations

from typing import Any, Literal

from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Const, Op
from mlopt_core.laws import minted
from mlopt_core.laws.ruleset import RuleSet

__all__ = [
    "FOLD_LAW",
    "MACHINE_MINTED",
    "SYNOP2_SHIPPED",
    "SYNOP_SHIPPED",
    "declare",
    "expand_name",
    "folds",
    "law_provenance",
    "premises",
    "prim_default",
    "records",
    "register",
    "ruleset",
    "spec",
    "synop_name",
]

#: The minted op names the family ships — ``synop_<sha256[:10]>``
#: (depth-1) plus the depth-2 ``synop2_<sha256[:10]>`` — the
#: core-side table, bound here.
SYNOP_SHIPPED: frozenset[str] = minted.SYNOP_SHIPPED

#: The depth-2 subset — the ``--prim-bootstrap`` mint's ``synop2_*``
#: ops (folds over the shipped ``synop_*`` vocabulary).
SYNOP2_SHIPPED: frozenset[str] = minted.SYNOP2_SHIPPED

#: The domain-side provenance ledger: the ``synop_*`` op names this
#: module ships torch denotations for.  The fold (``prim:synop_*``)
#: and premise (``synop_*.expand``) names classify through core's
#: ``MACHINE_SHIPPED`` — the laws moved; the denotations stayed.
MACHINE_MINTED: frozenset[str] = SYNOP_SHIPPED

#: The fold law text — the same template the core materializer
#: stamps on each fold (kept for callers that matched on it).
FOLD_LAW = (
    "machine-minted primitive fold — the manual subgraph collapses "
    "to the {synop} primitive (zoo-prim mint; admitted by the full "
    "gauntlet, machine-original; denotation definitional — the op's "
    "torch binding IS the spelled subgraph)"
)


def synop_name(fold_name: str) -> str:
    """Return the op a ``prim:synop_*`` fold name mints (identity otherwise)."""
    return minted.synop_name(fold_name)


def expand_name(synop: str) -> str:
    """Return the definitional-premise law name for a ``synop_*`` op."""
    return minted.expand_name(synop)


def records() -> dict[str, Any]:
    """Return the embedded records — the frozen mint artifacts."""
    return minted.prim_records()


def spec(synop: str) -> Any:
    """Return the op's denotation as a genkernel spec tuple.

    Rendered from the definitional premise's RHS — the denotation
    spelling — with metavar names kept, the ``_pat_spec`` convention
    the mint used: leaf metavars stay their pattern names (the call's
    positional args bind them) and attr metavars stay their names
    inside the attr dicts (the call's attrs carry the bound values
    back in at eval time).  Depth-1 premises end at the fold LHS;
    depth-2 premises end at the skeleton's *expanded* manual
    denotation, so the spec is manual-ops-only for every op.
    """
    return _spec_of(minted.prim_fold(synop_name(synop))[1].rhs)


def _spec_of(pat: Any) -> Any:
    """Render an ``Op`` pattern as a genkernel spec — metavar names kept."""
    if isinstance(pat, Op):
        parts = [_spec_of(a) for a in pat.args]
        if pat.attrs:
            parts.append(dict(pat.attrs))
        return (pat.op, *parts)
    if isinstance(pat, Const):
        return pat.value
    return pat  # metavar leaf — the name IS the binding key


def _spec_bind(spec_: Any, attrs: dict, attr_mvars: tuple) -> Any:
    """Instantiate a spec's attr-metavar values under the call's attrs.

    The minted call carries its attr metavars as real op attrs; the
    spelled denotation needs those values back inside the spec's attr
    dicts — a dict value equal to an attr-mvar name resolves through
    the call's attrs, everything else passes through literal.
    """
    if isinstance(spec_, (tuple, list)):
        out = []
        for e in spec_:
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
                out.append(_spec_bind(e, attrs, attr_mvars))
        return tuple(out) if isinstance(spec_, tuple) else out
    return spec_


def _binding(spec_: Any, leaf_mvars: tuple, attr_mvars: tuple) -> Any:
    """Return the op's torch binding — its denotation, spelled out.

    ``fn(*args, **attrs) = eval(spec, {mvar: arg}, ambient)`` — the
    op IS the manual subgraph: equal to the fold's LHS at every bound
    site by construction (the mint measured ``equal`` on every site —
    ``sites``/``equal_sites`` in the record).  The ambient table is
    looked up per call so post-hoc overrides keep reaching it.
    """

    def fn(*args: Any, **attrs: Any) -> Any:
        import mlopt_torch.torch_bridge as tb

        from mlopt_discovery.genkernel import _eval_spec

        env = dict(zip(leaf_mvars, args, strict=True))
        return _eval_spec(
            _spec_bind(spec_, attrs, attr_mvars),
            env,
            tb._IR_TO_TORCH,
        )

    return fn


def declare() -> None:
    """Force materialization — the ``synop_*`` ops are core-declared.

    The op *declarations* travel with the laws: importing
    ``mlopt_core.laws`` materializes the folds (``MINTED_RULES``),
    which registers each ``synop_*``/``synop2_*``
    :class:`~mlopt_core.ir.GenDef` so ``generator(name)`` resolves.
    Idempotent — kept as the explicit spelling for callers that want
    the ops declared without the torch bindings.
    """
    minted.prim_folds()


def register() -> int:
    """Install the ops' torch denotations; return the op count.

    ``declare()`` plus one ambient binding per op:
    ``_IR_TO_TORCH[synop] = fn`` where ``fn`` replays the spelled
    subgraph through the same ambient table — so nested ``synop``
    denotations and post-hoc overrides compose.  The spec is the
    definitional premise's RHS (see :func:`spec`): manual ops only
    at both depths, so a ``synop2_*`` binding stands self-contained
    even where its embedded ``synop_*`` is unbound.  Seating on the
    ambient dict reaches every ``OpTable.full()``-backed sink,
    including already-constructed ``TorchSink`` instances
    (``supported_ops`` is the table's live key set).  Idempotent —
    re-registration rewrites byte-identical bindings.
    """
    import mlopt_torch.torch_bridge as tb

    declare()
    for name in SYNOP_SHIPPED:
        fold, expand = minted.prim_fold(name)
        tb._IR_TO_TORCH[name] = _binding(
            _spec_of(expand.rhs),
            tuple(fold.rhs.args),
            tuple(fold.rhs.attrs),
        )
    return len(SYNOP_SHIPPED)


def folds() -> tuple[Rewrite, ...]:
    """Return the shipped fold laws — ``pattern -> synop_*(args)``."""
    return minted.prim_folds()


def premises() -> tuple[Rewrite, ...]:
    """Return the definitional ``synop_*.expand`` premises — cert axioms.

    NOT a saturation set: expanding ``synop -> pattern`` inside a
    search would re-add the unfolded spelling the fold collapses.
    These are the anchored-derivation axioms a cert/verify leg unions
    into its axiom set.
    """
    return minted.prim_premises()


def ruleset(name: str = "prim_folds") -> RuleSet:
    """Return the fold set as a named :class:`RuleSet` for ``rules=``."""
    return RuleSet(
        name,
        folds(),
        description=(
            "the shipped machine-minted primitive folds — "
            f"{len(SYNOP_SHIPPED)} synop_*/synop2_* ops from the "
            "zoo-prim mints (docs/reports/results_overnight_prim_v2"
            ".json + results_overnight_prim_bootstrap.json); "
            "denotations register via shipped_primitives.register()"
        ),
    )


def prim_default(*, base: Any = None) -> RuleSet:
    """Return ``base + prim folds`` — a no-op union onto ``DEFAULT``.

    The folds live in core ``DEFAULT`` now, so onto the default
    *base* (``None`` resolves the orchestrator's composed
    ``DEFAULT_RULES``, imported lazily) this adds nothing — kept as
    the explicit spelling for callers that compose the fold set onto
    a *custom* base ruleset.
    """
    if base is None:
        from mlopt_orchestrator.optimize import default_rules

        base = default_rules()
    return base + ruleset()


def law_provenance(rule: object) -> Literal["machine", "human"]:
    """Classify a rewrite's authorship across BOTH shipped libraries.

    Core's :func:`mlopt_core.laws.provenance.law_provenance` covers
    the fold (``prim:synop_*``) and premise (``synop_*.expand``)
    names — they ship in ``MACHINE_SHIPPED`` now.  This classifier
    unions the ledgers: this module's :data:`MACHINE_MINTED` op names
    report ``"machine"``; anything else delegates to the core
    registry.
    """
    from mlopt_core.laws.provenance import (
        law_provenance as _core_provenance,
    )

    name = (
        rule if isinstance(rule, str) else getattr(rule, "name", rule)
    )
    if name in MACHINE_MINTED:
        return "machine"
    return _core_provenance(name)

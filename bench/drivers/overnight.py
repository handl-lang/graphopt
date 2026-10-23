"""The overnight discovery loop — episodes -> mint -> admit -> classify -> ledger.

The loop the resumable infra was built for: run open-domain episodes
with a registered player, mint every evaluated candidate into a
``Rewrite``, push each through the admission gauntlet (the
``bench/referees`` legs — oracle -> pay -> sweep -> cert), classify
the admitted survivors against the prior-art bank, and persist every
outcome to an append-only, protocol-gated JSONL cell store.

Stages:

1. ``episodes`` — one ``OpenArena`` page per case
   (:func:`mlopt_discovery.open_game.gen_cases` — disjoint slices of
   the deterministic enumerated stream), played by a registered
   player (``--player``, default ``playbook``) under
   :func:`gamekit.engine.run_episode`.  Every ``evaluate`` report
   lands in the ``("episode", i)`` cell: candidate name, arena
   verdict, measured cost delta.
2. ``mint`` — each evaluated candidate's ``as_rule()`` — a bare
   pattern claim ``lhs -> rhs`` — is serialized into its
   ``("candidate", name)`` cell (``laws.serialize.law_to_data``), so
   a resumed run never re-derives the rule.
3. ``admit`` — the referee arm (``--referee``, default ``full``)
   judges each minted claim via
   :func:`bench.referees.judge`; the per-leg ``ComponentVerdict``s
   are stored verbatim (first-failed-leg accounting comes free).
   ``SkipReferee`` and component exceptions are recorded honestly —
   never a silent drop, never a crash.
4. ``classify`` — admitted laws get a novelty verdict:
   ``known-art`` (an alpha-sibling sits in
   ``mlopt_core.laws.priorart.PRIOR_ART_RULES`` or the widest shipped
   bank already derives the instance — the novelty-sieve gates,
   reused), ``vacuous`` (no binding produced a site — the gates are
   unevaluable), ``machine-original`` (certified, admitted, and the
   bank cannot prove it).
5. ``ledger`` — :class:`bench.benchkit.checkpoint.Checkpoint`: an
   append-only JSONL file whose manifest pins the protocol
   (episodes/page/budget/seed/player/referee) and provenance
   (git sha/dirty, device).  ``--resume`` reopens it: done cells
   replay verbatim, a crash costs one cell, a protocol mismatch
   restarts loudly.
6. ``emit`` — ``docs/results_overnight.md`` (the funnel table + the
   machine-original finds) and a provenance-stamped ``.json``.

The referee context is the honest one for this domain: the open game
has no module corpus, so ``probe``/``real_terms`` are empty and the
gauntlet measures on its own synthesized binding domain — the same
domain that admitted (and refuted) the shipped ``syn_*`` mints.

**Corpora** (``--corpus``).  ``grammar`` (default) is the arm above:
episodes page the enumerated candidate stream.  ``zoo`` targets real
structure: each episode is one held-out ``mlopt_discovery.zoo``
module, exported through ``export_to_ir``, and the episode's pool is
the stream's candidates whose pattern LHS strictly binds a subterm of
that real term — the claims that would actually fire on the module.
The referee context is populated honestly on this arm: ``real_terms``
and ``probe`` carry the covered workloads' exported terms (with a
``TorchSink`` so the pay leg lowers and verifies), and the novelty
classify sees the modules as ``corpus:<name>`` instances.  The arm is
recorded in the ledger manifest's protocol (and the emitted document)
so a resumed or resumed-against run can never silently mix corpora.

``zoo-real`` is the **corpus-driven mint**: the enumerated stream is
dropped as the hypothesis generator and each episode's pool is minted
*from* the module's own term — ``census.op_tuple_census`` over the
exported graph feeding ``pipeline.propose`` (census-naturality +
mixed-view + pattern-recognition + fusion-schema + shape-aware +
grammar — the same
mint path that produced ``mul_unsq_nat``, ``census:mul_unsqueeze``
from DeltaRuleBlock's ``β·k`` op-tuple), gated binding-first —
``require_bind=True`` keeps only proposals whose LHS matches a real
subterm of the module, so a claim that cannot fire on the corpus
never spends mint or gauntlet effort.  The grammar arm's
diagnostic — grammar mints binding no real site (``real: 0 eq/0 envs``
for 232/248 candidates in ``docs/results_overnight.md``) — is the arm
this one answers: candidates minted on real subterms, judged against
a ``real_terms`` domain that *is* the corpus's distinct real subterms
(``shape_proposal.real_matches``' own matching granularity — the zoo
arm's root-only domain could never see a subterm binding), unioned
with the intake side-file census (``intake_dir()``) when it exists.
Same gauntlet, same ledger, same doc shape — the document lands at
``docs/results_overnight_real.md``.

``--compose [ROUNDS]`` adds the **composition arm** on ``zoo-real``:
after the single-step pool binds, ``meta.synthesize_rules`` — the
``syn_*`` mint fixpoint that produced the shipped ``conv_bn_fold``
(ADR-0007) — pair-composes the bound candidates with the module's own
term as the seed, iterating *ROUNDS* strata toward a fixpoint
(default 2).  The synthesis bank also carries the arm's **guarded
parents** (:func:`_guarded_parents`): every shipped law whose
``check``/``derive`` hooks actually fire on the module's term — the
premise-bearing singles the ``_compose_guards`` transport was built
for.  Without them the measured mint was all-unguarded (``mixed:``
singles carry no hooks, so the transport never engaged and
``missing_hooks`` stayed empty); with them an emitted composite can
inherit a live side condition — ``missing_hooks`` honestly non-empty.
Every emitted composite then faces the arm's own gate:
it is kept only when its LHS binds a real subterm *and* its
instantiated RHS denotes under every such binding — an ill-typed
composite (the oracle's ``rhs_err`` signal) is dropped at emit, never
spending an episode slot or the gauntlet.  Survivors enter the pool
with their transported premise closure — ``_compose_guards``
check/derive hooks plus the ``parents`` / ``guard_pats`` record —
serialized into the candidate cell's ``law["guard"]`` like the
``mlopt_core.laws.minted`` rows.  Machine-originals live here: a
composite the prior-art bank cannot derive is the sieve's target
verdict.  Each *gated* stratum feeds the next: the mint loops
``synthesize_rules`` a round at a time and re-seats only
gate-surviving composites as parents — enumerated ahead of the
base bank so the round's fuel reaches the pairs that involve them
— until a stratum emits no new viable parent (the fixpoint) or
*ROUNDS* strata have run.  The arm is protocol-pinned as ``mint:
corpus-bind-compose-strata``, so a ledger written before the
strata-chaining fix (or without the arm at all) restarts loudly.

``zoo-xmodule`` is the **cross-module mint** — the same corpus-driven
arm turned inside out.  Where ``zoo-real`` mints each episode's pool
from one module's own census, ``zoo-xmodule`` anti-unifies over the
WHOLE corpus: every zoo module's exported term is collected, the
op-patterns recurring across ``--xbind`` (default 2) or more
*distinct* modules seed ``pipeline.propose``'s generalized candidates
(the census-naturality / mixed-view / pattern-recognition /
fusion-schema / shape-aware / grammar families — the same mint), and
``antiunify.mine_candidates`` over the merged corpus e-graph mines
the deep shared subgraphs no 1-level op-tuple expresses, each minting
the mechanical view-transform family over its own shape.  Every
candidate then faces the arm's **K-bind gate**: a rule must bind
real subterms in at least *K* distinct modules — a claim that binds
exactly one is the old per-module surface, *K* = 2+ is the new one
(a law true on 10 real modules is likelier a real transform than a
one-module coincidence).  The minted antiunified variants also face
the same emit-sanity gate the composition arm applies — the
instantiated RHS must denote under every bound binding — before a
candidate can spend an episode slot.  One shared pool is minted once
and episodes page it (each case is a ``page``-sized slice of the
corpus-level pool, as ``grammar`` pages the stream).  Same gauntlet,
same ledger, same doc shape — the document lands at
``docs/results_overnight_xmodule.md`` and the mint is
protocol-pinned as ``corpus-xbind``.

``zoo-prim`` is the **primitive-minting arm** — the cross-module mint
carried to its conclusion.  The shared shapes ``zoo-xmodule`` mines
are laws over *existing* ops: every claim it pools is a rewrite the
bank could in principle already name.  This arm names the shape
itself: a mined pattern recurring across >= ``--xbind`` distinct
modules and *not denoted by any bank op* (:func:`_denoted_shapes` —
the spelled side of every shipped fold/expansion and the
``lawdata.HANDLERS`` inventory) mints a NEW primitive op
``synop_<hash>`` whose *denotation is the manual subgraph*: the op is
registered (:func:`mlopt_core.ir.op_def`) and bound into the ambient
torch table (``_IR_TO_TORCH``) evaluating the spelled spec — so the
op verifies fp64 on every bound site *by construction*, and it lowers
through the same ``TorchSink`` the pay leg probes.  The minted claim
is the fold ``manual_spelling -> synop_*(args)`` — **novel by
construction**: no catalog contains laws about an op that did not
exist at ship time, so the prior-art sieve cannot derive it.  The
fold still faces the same emit gate (its denotation must equal the
pattern on every bound site — measured, not assumed), the same
gauntlet (the ``synop_*.expand`` definitional premise rides
``rule.premises`` so the cert leg anchors a derivation — the
mechanism the fusion-schema mint uses for denotations the bank
lacks), and the same sieve.  ``handler_synth.mint_handlers`` runs on
each shape as the synthesis measurement: a grammar body verified on
the bound-site domain rides the record as provenance (the machine
found a compressed semantics); a refusal keeps the definitional op
and records the measured gap.  One shared pool, paged like
``zoo-xmodule``; the document lands at
``docs/reports/results_overnight_prim.md`` and the mint is
protocol-pinned as ``corpus-prim``.

Run::

    .venv/bin/python bench/drivers/overnight.py --dry
    .venv/bin/python bench/drivers/overnight.py --dry --corpus zoo
    .venv/bin/python bench/drivers/overnight.py --dry --corpus zoo-real
    .venv/bin/python bench/drivers/overnight.py --dry --corpus zoo-real --compose
    .venv/bin/python bench/drivers/overnight.py --dry --corpus zoo-xmodule
    .venv/bin/python bench/drivers/overnight.py --dry --corpus zoo-prim
    .venv/bin/python bench/drivers/overnight.py --episodes 48 \
        --player playbook --referee full --resume
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gamekit import engine
from gamekit.players import registry as players
from mlopt_core.laws.provenance import law_provenance
from mlopt_core.laws.serialize import (
    alpha_key,
    law_from_data,
    law_to_data,
)
from mlopt_discovery import open_game as og
from mlopt_discovery.verifier import verify_law

__all__ = ["DEFAULTS", "DRY", "main", "render_markdown", "run"]

#: The repo root — ``bench.*`` imports are function-level (the
#: novelty_sieve convention): standalone runs put ``bench/drivers``
#: on ``sys.path``, and :func:`main` inserts the root first.
_REPO = Path(__file__).resolve().parents[2]

#: The full-run defaults — page/budget follow the open game's own
#: episode horizon; 24 episodes of 48 cover the ~1.1k-candidate
#: enumerated stream once.
DEFAULTS: dict[str, Any] = {
    "episodes": 24,
    "page": 48,
    "budget": 48,
    "seed": 0,
    "player": "playbook",
    "referee": "full",
    "corpus": "grammar",
    "scan": 2048,
    "compose": 0,
    "xbind": 2,
}

#: ``--dry`` — seconds-scale smoke: two episodes, a tiny pool, a
#: shallow stream scan on the zoo arm.
DRY: dict[str, Any] = {
    "episodes": 2,
    "page": 6,
    "budget": 4,
    "scan": 384,
}

#: The corpus arms — ``grammar`` pages the enumerated candidate
#: stream; ``zoo`` pages the held-out ``mlopt_discovery.zoo`` module
#: registry, the same stream filtered to real-matching claims;
#: ``zoo-real`` drops the stream entirely and mints each episode's
#: pool from the module's own term (the ``pipeline.propose`` census
#: mint — the ``mul_unsq_nat`` path); ``zoo-xmodule`` anti-unifies
#: over the WHOLE corpus — one shared pool minted from the
#: op-patterns and deep subgraphs recurring across >= ``--xbind``
#: distinct modules, episodes paging it like the stream arm;
#: ``zoo-prim`` mints NEW primitive ops (``synop_*``) for the
#: recurring manual subgraphs no bank op denotes and folds them —
#: laws novel by construction.
CORPORA: tuple[str, ...] = (
    "grammar",
    "zoo",
    "zoo-real",
    "zoo-xmodule",
    "zoo-prim",
)

#: Where the documents land.
DOC_PATH = _REPO / "docs" / "results_overnight.md"

#: The zoo-real arm's own document — a different measurement than the
#: stream-filtered arms, so it does not overwrite their ledger/doc.
DOC_PATH_REAL = _REPO / "docs" / "results_overnight_real.md"

#: The zoo-xmodule arm's own document — one corpus-level pool, not a
#: per-module mint.
DOC_PATH_XMODULE = _REPO / "docs" / "results_overnight_xmodule.md"

#: The zoo-prim arm's own document — the primitive-mint measurement,
#: filed with the bounded-run reports.
DOC_PATH_PRIM = _REPO / "docs" / "reports" / "results_overnight_prim.md"

#: The default ledger — ``benchkit.checkpoint.DEFAULT_DIR``'s lane
#: file, spelled out so the constant needs no ``bench.*`` import.
CKPT_PATH = Path("bench/results/checkpoints/overnight.jsonl")

#: The zoo-real lane file — a non-``--resume`` run truncates its
#: ledger, so the corpus arms never share a default path.
CKPT_PATH_REAL = Path("bench/results/checkpoints/overnight_real.jsonl")

#: The zoo-xmodule lane file.
CKPT_PATH_XMODULE = Path(
    "bench/results/checkpoints/overnight_xmodule.jsonl"
)

#: The zoo-prim lane file.
CKPT_PATH_PRIM = Path("bench/results/checkpoints/overnight_prim.jsonl")

#: The op vocabulary the corpus mint enumerates laws over —
#: ``"derived"`` classifies every corpus op by property
#: (``vocab.derive_vocabulary``); ``"hand"`` is the fixed projection.
_MINT_VOCAB = "derived"

#: The composition arm's per-stratum fuel — bounds each
#: ``synthesize_rules`` round's rule-application attempts, matching
#: the single-pass semantics the ``syn_*`` mints were measured under.
_COMPOSE_FUEL = 256

#: The ``meta.family`` a synthesized composite records — the ledger
#: distinguishes the composition arm's ``syn_*`` pairs from the
#: single-step proposers' families.
_COMPOSE_FAMILY = "synth-composite"

#: The ``meta.family`` an antiunified cross-module mint records — the
#: ledger distinguishes the ``xanti:*`` generalized-shape rules from
#: the ``pipeline.propose`` census mints they sit beside.
_XANTI_FAMILY = "xanti-view"

#: Cap on antiunified patterns taken through the xmodule mint — the
#: top-``gain`` slice, so the corpus pool stays bounded like a page.
_XANTI_CAP = 48

#: Bound on bound sites an xanti variant's emit-sanity gate walks —
#: the same ``rhs-err`` gate the composition arm applies, over a
#: capped cross-corpus site set.
_XANTI_SITES = 64

#: The ``meta.family`` a primitive-minted fold records — the ledger
#: distinguishes the ``prim:synop_*`` folds from the rewrite mints.
_PRIM_FAMILY = "prim-fold"

#: Cap on mined shapes taken through the prim mint — the top-``gain``
#: slice, so the corpus pool stays bounded like a page.
_PRIM_CAP = 24

#: ``mine_candidates`` size window for the prim arm — the same deep-
#: shape bounds the xmodule mint reads.
_PRIM_MIN_OPS = 2
_PRIM_MAX_OPS = 6

#: Bound on bound sites a candidate's verify/domain stage walks.
_PRIM_SITES = 16

#: Bound on bound sites sampled into the handler-verify domain.
_PRIM_DOMAIN = 4

#: ``mint_handlers`` candidate cap per shape — the bounded grammar
#: search the synthesis measurement spends (refusals quote it).
_PRIM_BUDGET = 256

_NOVELTY_ORDER = ("machine-original", "known-art", "vacuous")


# ---------------------------------------------------------------------------
#  the zoo corpus arm — real modules as the episode boards
# ---------------------------------------------------------------------------

#: Exported-workload cache — ``(name, wseed) -> (ir, tensors, feeds)``.
#: The pool filter, the referee context and the resume rebuild all
#: need the same module's term; each build is seed-pinned so a
#: rebuild is byte-identical.
_EXPORT_CACHE: dict[tuple, tuple] = {}


def _zoo_registry() -> dict[str, Any]:
    """``name -> Workload`` over the held-out zoo registry."""
    from mlopt_discovery import zoo as zoo_mod

    return {w.name: w for w in zoo_mod.zoo()}


def _zoo_cases(
    seed: int, episodes: int, *, page: int, scan: int, compose: int = 0
) -> list:
    """One case per covered zoo workload — registry order, JSON-safe.

    ``wseed`` pins the workload's build draws so the episode board —
    and a resume's rebuild of it — is deterministic regardless of how
    much RNG the rest of the run consumed first.  ``episodes`` beyond
    the registry size simply covers the whole zoo.  ``compose`` is the
    zoo-real composition arm's stratum count (0 = off — see
    ``_RealArena``); it travels in the case record so a rebuilt pool
    reproduces its composites verbatim.
    """
    names = list(_zoo_registry())[:episodes]
    return [
        {
            "zoo": name,
            "wseed": seed * 100_003 + i,
            "page": page,
            "scan": scan,
            "compose": compose,
        }
        for i, name in enumerate(names)
    ]


def _zoo_export(case: dict) -> tuple:
    """Build + export the case's workload; ``(ir, tensors, feeds)``.

    The build is seed-pinned (``wseed``) so pool rebuilds and the
    referee context see the identical module.  Export failures
    propagate — the caller decides whether they are an empty pool
    (arena) or a dropped context entry (referee ctx).
    """
    import torch
    from mlopt_torch.torch_bridge import export_to_ir

    key = (case["zoo"], case.get("wseed", 0))
    if key not in _EXPORT_CACHE:
        torch.manual_seed(key[1])
        model, feed = _zoo_registry()[case["zoo"]].build()
        feeds = feed if isinstance(feed, tuple) else (feed,)
        ir, tensors = export_to_ir(model.eval().double(), feeds)
        _EXPORT_CACHE[key] = (ir, dict(tensors), feeds)
    return _EXPORT_CACHE[key]


class _ZooArena(og.OpenArena):
    """An ``OpenArena`` whose pool is the real-match slice.

    The enumerated grammar remains the hypothesis generator; the zoo
    module's exported term is the corpus.  The pool is the stream's
    first ``page`` candidates whose pattern LHS strictly binds a
    subterm of the module — the claims an e-graph would actually fire
    on this architecture.  ``scan`` bounds how much of the ~1.1k-
    candidate stream one module reads (``<= 0`` scans it all).  A
    module that exports nothing or matches nothing is a done state —
    the honest empty pool — and an export failure is recorded in
    ``self.error``, never silently zeroed.
    """

    def __init__(self, case: dict) -> None:
        from mlopt_core.egraph.terms import _term_match
        from mlopt_core.ir import Op
        from mlopt_core.meta import _positions
        from mlopt_discovery.enumerate import enumerate_candidates

        self.pool: list = []
        self.verdicts: dict[int, str] = {}
        self.steps = 0
        self.done = True
        self._n_true = 0
        self._n_useful = 0
        self.error = ""
        try:
            ir, _tensors, _feeds = _zoo_export(case)
        except Exception as e:
            self.error = f"export: {type(e).__name__}: {e}"
            return
        subs: dict[str, list] = {}
        flat: list = []
        for _p, s in _positions(ir.root):
            flat.append(s)
            if isinstance(s, Op):
                subs.setdefault(s.op, []).append(s)
        scan = case.get("scan") or None
        for cand in itertools.islice(enumerate_candidates(None), scan):
            pat = cand.as_rule().lhs
            tries = (
                subs.get(pat.op, ()) if isinstance(pat, Op) else flat
            )
            if any(_term_match(pat, s) is not None for s in tries):
                self.pool.append(cand)
                if len(self.pool) >= case.get("page", 48):
                    break
        self.done = not self.pool


def _zoo_export_env(cases: list) -> dict:
    """``name -> (ir, tensors, feeds) | None`` for the covered zoo."""
    env: dict[str, Any] = {}
    for case in cases:
        if case["zoo"] in env:
            continue
        try:
            env[case["zoo"]] = _zoo_export(case)
        except Exception:
            env[case["zoo"]] = None
    return env


def _zoo_probe(env: dict) -> list:
    """The covered modules as ``TermCase``s — the pay leg's domain."""
    from mlopt_discovery.impact import TermCase

    probe = []
    for name, rec in env.items():
        if rec is None:
            continue
        ir, tensors, feeds = rec
        probe.append(
            TermCase(
                source="zoo",
                name=name,
                term=ir.root,
                inputs=tuple(ir.inputs),
                feed=tuple(feeds),
                param_vals=tensors,
            )
        )
    return probe


def _zoo_referee_ctx(env: dict, referees: Any) -> Any:
    """The zoo arm's ``RefereeCtx`` — real terms + a lowering sink.

    ``real_terms`` are the covered modules' exported roots (the
    oracle/sweep real-match domain); ``probe`` is the same modules as
    ``TermCase``s (the pay leg's firing domain), verified through a
    real ``TorchSink`` — the context the grammar arm's docstring
    notes it lacks.
    """
    from mlopt_torch.adapters import TorchSink

    return referees.RefereeCtx(
        real_terms=tuple(
            rec[0].root for rec in env.values() if rec is not None
        ),
        probe=tuple(_zoo_probe(env)),
        sink=TorchSink(),
    )


def _zoo_case_ctxs(env: dict) -> list:
    """``novelty_sieve._CaseCtx``s over the covered zoo terms.

    Fed to ``instance_of`` at classify time so an admitted law's
    firing site is named ``corpus:<module>`` — the real-structure
    provenance the grammar arm cannot produce.
    """
    from bench.drivers.screen import novelty_sieve as ns

    return [
        ns._CaseCtx(
            case=SimpleNamespace(name=name),
            model=None,
            feed=None,
            term=rec[0].root,
            ops=ns._ops_of(rec[0].root),
        )
        for name, rec in env.items()
        if rec is not None
    ]


# ---------------------------------------------------------------------------
#  the zoo-real corpus arm — candidates minted FROM the module's term
# ---------------------------------------------------------------------------


def _guarded_parents(root: Any) -> list:
    """Shipped guarded laws that fire on *root* — the premise parents.

    The composition arm's guarded bank: every ``_synthesizable``
    member of ``ALL_RULES`` carrying ``check``/``derive`` hooks that
    actually fires somewhere on the module's term (``apply_rewrite_at``
    — the seed path's own firing semantics, so a parent in the bank
    is a parent the mint can spend fuel on productively).  A guarded
    law that cannot fire here contributes nothing a guarded
    composite could descend from: the seed path needs a real firing
    and the emit gate would drop a composite whose side condition
    declines every real binding.  These are the *guarded singles*
    ``_compose_guards`` was built to transport — ``mul_unsq_nat`` on
    DeltaRuleBlock, ``rms_norm_fold_nogain`` on RMSNormPerHead,
    ``glu_fold`` on GatedConvUnit — whose premise closures make a
    ``syn_*`` pair a guarded composite instead of another bare
    pattern.
    """
    from mlopt_core.laws import ALL_RULES
    from mlopt_core.meta import (
        _positions,
        _synthesizable,
        apply_rewrite_at,
    )

    return [
        r
        for r in ALL_RULES
        if (r.check is not None or r.derive is not None)
        and _synthesizable(r)
        and any(
            apply_rewrite_at(r, root, p) is not None
            for p, _ in _positions(root)
        )
    ]


class _RealArena(og.OpenArena):
    """An ``OpenArena`` whose pool is the corpus-minted candidate set.

    The enumerated stream never enters this arm: each episode runs
    ``census.op_tuple_census`` on the case module's exported term and
    feeds the counts to ``pipeline.propose`` — the census-naturality /
    mixed-view / pattern-recognition / shape-aware / grammar proposers,
    the same machinery that minted ``mul_unsq_nat`` from DeltaRuleBlock's
    ``β·k`` op-tuple (``exposed_discovery``'s pipeline arm) — under
    ``require_bind=True``: the mint keeps only proposals whose LHS
    binds a real subterm of the module, so the pool never spends an
    episode slot on a claim whose real region is empty by
    construction.  Every minted
    :class:`~mlopt_discovery.pipeline.Proposal` becomes a
    :class:`~mlopt_discovery.proposal.Candidate` carrying the proposal's
    own ``Rewrite`` (``check``/``derive`` hooks intact) — the oracle
    verdict in the episode still measures the bare pair, while the
    gauntlet judges the side-conditioned rule.

    ``meta[i]`` records the proposal's ``family``/``sources`` beside
    the pool so the candidate cell can name which proposer minted it —
    rebuilt deterministically on resume with the pool itself.  An
    export failure lands in ``self.error`` like the zoo arm's; an
    empty mint is the honest done state.

    ``case["compose"] > 0`` seats the **composition arm**: the bound
    single-step pool is pair-composed by
    ``mlopt_core.meta.synthesize_rules`` — the ``syn_*`` fixpoint
    ``conv_bn_fold`` minted through — with the module's own term as
    the seed, for up to ``case["compose"]`` strata (the loop ends
    early at the fixpoint: a stratum emitting no new viable parent).
    The bank pairs the
    bound singles with the module's **guarded parents**
    (:func:`_guarded_parents` — shipped ``check``/``derive`` laws that
    fire on this term), so a composite can inherit a transported side
    condition through ``_compose_guards`` instead of minting bare
    patterns only; the guarded bank also runs when the single-step
    mint bound nothing (shipped-guarded pairs are corpus-driven —
    they fire on the module — even with no bound singles).  Emitted
    composites are held to the arm's own gate — the LHS must bind a
    real subterm (the ``require_bind`` contract, applied a second
    time since the symbolic path can mint a pair that never fired on
    the seed) *and* the instantiated RHS must denote under every such
    binding: a composite whose rhs errors on the bound env is
    ill-typed — the oracle leg measures it as ``rhs_err``, a refusal
    it could never survive — so the emit gate drops it before it can
    spend mint or gauntlet effort.  Survivors append after the
    singles, each
    carrying its transported premise closure: the ``_compose_guards``
    check/derive hooks and the ``parents`` / ``guard_pats`` record
    ``_admit`` serializes into the cell's ``law["guard"]``.  The
    composite slice is bounded by ``page`` like the single-step mint
    it sits beside — the pool can reach two pages under
    ``--compose``.
    """

    def __init__(self, case: dict) -> None:
        import mlopt_carriers  # noqa: F401 — carrier op bindings
        from mlopt_discovery import pipeline as pipe
        from mlopt_discovery.census import CorpusTerm, op_tuple_census
        from mlopt_discovery.proposal import Candidate

        self.pool: list = []
        self.meta: dict[int, dict] = {}
        self.verdicts: dict[int, str] = {}
        self.steps = 0
        self.done = True
        self._n_true = 0
        self._n_useful = 0
        self.error = ""
        try:
            ir, _tensors, _feeds = _zoo_export(case)
        except Exception as e:
            self.error = f"export: {type(e).__name__}: {e}"
            return
        counts, _terms_of = op_tuple_census(
            [CorpusTerm("zoo", case["zoo"], ir.root)]
        )
        page = case.get("page", 48)
        for i, p in enumerate(
            pipe.propose(
                dict(counts), [ir.root], _MINT_VOCAB, require_bind=True
            )
        ):
            if i >= page:
                break
            self.pool.append(
                Candidate(
                    strategy="corpus",
                    label=p.name,
                    lhs=p.lhs,
                    rhs=p.rhs,
                    note=", ".join(p.sources),
                    rule=p.as_rule(),
                )
            )
            self.meta[i] = {
                "family": p.family,
                "sources": list(p.sources),
            }
        if case.get("compose"):
            self._mint_composites(ir.root, case)
        self.done = not self.pool

    def _mint_composites(self, root: Any, case: dict) -> None:
        """Pair-compose the guarded bank over the module's own term.

        The synthesis bank is the module's guarded parents
        (:func:`_guarded_parents`) ahead of the bound singles — the
        premise-bearing rules take enumeration priority under
        ``fuel`` since they are what makes a composite *guarded*:
        ``_compose_guards`` returns ``(None, None)`` for an unguarded
        pair, so without a hooked parent every minted composite is a
        bare pattern.  The seed-guided path fires each parent on
        ``root``'s real
        subterms, so an emitted composite is a chain that actually
        rewrote the module; the symbolic path (unguarded pairs) emits
        fully general derived rules and is then held to the same
        ``require_bind`` gate the singles met.

        Strata chain *through the emit gate*: the mint runs
        ``synthesize_rules`` one stratum at a time and feeds back
        only the composites that survived the gate — a mint that
        binds no real subterm or whose RHS cannot denote on the
        bound env never spawns deeper pairs, so deeper strata stay
        viable-only by construction.  The fresh stratum enumerates
        *ahead* of the base bank so the round's fuel reaches pairs
        that involve it: the pair space is scanned in bank order and
        fuel is spent on successful applications, so a stratum
        appended behind the base rules waits behind every base-pair
        re-mint — the reason ``--compose 4`` measured identically to
        ``--compose 2`` under the single-call ``rounds=`` loop (whose
        ``max_emitted`` is also a cross-round total: a bank that
        fills the cap in stratum 2 never runs stratum 3).  The loop
        ends at the fixpoint — a stratum emitting no new viable
        parent — not at a fixed count; ``case["compose"]`` bounds
        the strata attempted, ``max_emitted`` each round's
        validation spend, and alpha-equivalent re-mints across
        strata are deduplicated by pattern key.

        Composites keep the ``syn_*`` name, namespaced by
        ``@<module>`` — the ``synthesize_rules`` counter restarts
        per stratum as well as per module, so a same-named pair
        minted in a different stratum can be a different rule, and
        the ledger keys candidates by name (a cross-stratum name
        collision on a distinct rule is disambiguated with a
        ``~r<stratum>`` tag).  The appended slice is capped at
        ``page`` after the bind gate.
        """
        from mlopt_core import meta as meta_mod
        from mlopt_core.egraph.terms import _term_match
        from mlopt_core.ir import Op
        from mlopt_discovery.census import _iter_subterms
        from mlopt_discovery.evidence import _site_outcome
        from mlopt_discovery.proposal import Candidate

        page = case.get("page", 48)
        bank = [
            *_guarded_parents(root),
            *(c.as_rule() for c in self.pool),
        ]
        subs = [s for s in _iter_subterms(root) if isinstance(s, Op)]
        seen: set = set()
        named: set = set()
        parents: list = []
        kept = 0
        for rnd in range(1, int(case["compose"]) + 1):
            emitted = meta_mod.synthesize_rules(
                [*parents, *bank],
                [root],
                fuel=_COMPOSE_FUEL,
                numeric_check=True,
                max_emitted=page,
            )
            viable: list = []
            for rule in emitted:
                if kept >= page:
                    break
                key = meta_mod._alpha_key(rule.lhs, rule.rhs)
                if key in seen:
                    continue
                sites = [
                    (m, s)
                    for s in subs
                    if (m := _term_match(rule.lhs, s)) is not None
                ]
                if not sites:
                    continue
                # Seed-eval sanity — the minted RHS must denote under
                # the seed's own binding.  A composite whose rhs
                # errors on a bound env is ill-typed: the oracle leg
                # would measure it as ``rhs_err`` (the
                # ``_site_outcome`` tag), a refusal it can never
                # survive, so it is dropped here rather than
                # spending an episode slot and the full gauntlet —
                # or a parent slot in the next stratum — on it.
                if any(
                    _site_outcome(rule, m, s) == "rhs-err"
                    for m, s in sites
                ):
                    continue
                prov = meta_mod.provenance(rule)
                name = f"{rule.name}@{case['zoo']}"
                if name in named:
                    name = f"{rule.name}~r{rnd}@{case['zoo']}"
                object.__setattr__(rule, "name", name)
                meta_mod.SYNTH_PARENTS[name] = prov
                self.pool.append(
                    Candidate(
                        strategy="corpus",
                        label=name,
                        lhs=rule.lhs,
                        rhs=rule.rhs,
                        note=f"composite: {' then '.join(prov)}",
                        rule=rule,
                    )
                )
                self.meta[len(self.pool) - 1] = {
                    "family": _COMPOSE_FAMILY,
                    "sources": list(prov),
                    "stratum": rnd,
                    # a transported premise closure — the composite's
                    # check/derive re-express the guarded parents'
                    # hooks.
                    "guarded": bool(
                        rule.check is not None
                        or rule.derive is not None
                    ),
                }
                seen.add(key)
                named.add(name)
                viable.append(rule)
                kept += 1
            if not viable:
                break  # the fixpoint — nothing viable left to chain on
            if kept >= page:
                break  # the pool slice is full — deeper mints can't append
            parents = viable


def _intake_cases() -> list:
    """The intake side-file census as ``TermCase``s ([] when absent).

    ``intake_dir()`` is a shared, regenerable cache: when a
    ``mlopt_discovery.intake`` run has populated it, its ingested
    terms join the zoo-real arm's real-match domain and firing probe.
    """
    from mlopt_discovery.intake import load_cases

    try:
        return load_cases()
    except Exception:
        return []


def _real_subterms(roots: list) -> list:
    """Every distinct ``Op`` subterm across *roots*, repr-deduped.

    The zoo-real arm's ``real_terms``: the corpus's real subterms at
    ``shape_proposal.real_matches``' own granularity — the zoo arm's
    root-only domain cannot see a subterm binding, which is exactly
    why stream-minted claims measured ``0 envs``.  Dedup is by
    ``op_repr`` so two structurally identical sites (same binding,
    same eval) count once — ``_iter_subterms``'s id-dedup keeps
    shared nodes but not equal spellings.
    """
    from mlopt_core.ir import Op, op_repr
    from mlopt_discovery.census import _iter_subterms

    seen: set[str] = set()
    out: list = []
    for root in roots:
        for s in _iter_subterms(root):
            if not isinstance(s, Op):
                continue
            k = op_repr(s)
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
    return out


def _real_referee_ctx(env: dict, referees: Any) -> Any:
    """The zoo-real arm's ``RefereeCtx`` — subterm real domain.

    ``real_terms`` is every distinct ``Op`` subterm of the covered
    modules' roots (plus the intake census's when the side-file
    exists) — the domain a corpus-minted claim binds on, matching the
    sites the mint itself read.  ``probe`` stays at module granularity
    (the e-graph finds the subterm firings itself), unioned with
    intake cases that carry a feed; ``sink`` is the same
    ``TorchSink`` the zoo arm verifies through.
    """
    import mlopt_carriers  # noqa: F401 — carrier op bindings
    from mlopt_torch.adapters import TorchSink

    intake = _intake_cases()
    roots = [rec[0].root for rec in env.values() if rec is not None]
    roots += [c.term for c in intake if c.term is not None]
    real_terms = _real_subterms(roots)
    probe = _zoo_probe(env)
    probe += [c for c in intake if c.term is not None and c.feed]
    return referees.RefereeCtx(
        real_terms=tuple(real_terms),
        probe=tuple(probe),
        sink=TorchSink(),
    )


def _real_case_ctxs(env: dict) -> list:
    """``novelty_sieve._CaseCtx``s over zoo + intake corpus terms."""
    from mlopt_discovery.intake.ledger import _DEFAULT_JSON

    from bench.drivers.screen import novelty_sieve as ns

    ctxs = _zoo_case_ctxs(env)
    if _DEFAULT_JSON.exists():
        for c in _intake_cases():
            if c.term is None:
                continue
            ctxs.append(
                ns._CaseCtx(
                    case=SimpleNamespace(name=c.name),
                    model=None,
                    feed=None,
                    term=c.term,
                    ops=ns._ops_of(c.term),
                )
            )
    return ctxs


# ---------------------------------------------------------------------------
#  the zoo-xmodule corpus arm — one pool anti-unified over the WHOLE corpus
# ---------------------------------------------------------------------------

#: Corpus-level pool cache — ``(wseed0, xbind) -> (cands, meta,
#: stats)``.  Every episode's arena slices the same mint, so the
#: whole-corpus export + antiunify pass is paid once per run (and a
#: resume's rebuild reproduces the pool verbatim — the exports are
#: seed-pinned, the mining deterministic).
_XPOOL: dict = {}


def _xmodule_cases(
    seed: int, episodes: int, *, page: int, xbind: int
) -> list:
    """One case per pool page — the corpus-level mint, paged.

    Unlike ``zoo``/``zoo-real`` there is no per-module board: the
    episode boards are ``page``-sized slices of the one shared pool,
    exactly as ``grammar`` pages the enumerated stream.
    ``wseed0`` pins the workload-build draws for the whole-registry
    export so a resume's pool rebuild is byte-identical.
    """
    return [
        {
            "xpage": i,
            "page": page,
            "xbind": xbind,
            "wseed0": seed,
        }
        for i in range(episodes)
    ]


def _xmodule_env(seed: int) -> dict:
    """``name -> (ir, tensors, feeds) | None`` over the WHOLE registry.

    The cross-module corpus is indivisible — the mint reads every
    exported term, so every zoo module is built here (seed-pinned
    per module, matching ``_zoo_cases``' ``wseed`` draws).
    """
    env: dict[str, Any] = {}
    for i, name in enumerate(_zoo_registry()):
        try:
            env[name] = _zoo_export(
                {"zoo": name, "wseed": seed * 100_003 + i}
            )
        except Exception:
            env[name] = None
    return env


def _op_subterms(roots: dict) -> dict:
    """``name -> [Op subterm]`` — each module's bind domain."""
    from mlopt_core.ir import Op
    from mlopt_discovery.census import _iter_subterms

    return {
        n: [s for s in _iter_subterms(r) if isinstance(s, Op)]
        for n, r in roots.items()
    }


def _modules_bound(pat: Any, subs_by_mod: dict) -> set:
    """Module names holding at least one subterm *pat* binds.

    The arm's binding granularity: ``_term_match`` — the same match
    ``require_bind`` and the referee's ``real_terms`` domain use —
    counted per module, so a pattern covering five sites inside one
    module still counts once.
    """
    from mlopt_core.egraph.terms import _term_match

    return {
        n
        for n, subs in subs_by_mod.items()
        if any(_term_match(pat, s) is not None for s in subs)
    }


def _xanti_variants(pat: Any, views: tuple) -> list:
    """The mechanical view-transform family over a mined shared shape.

    For a mined pattern ``f(A, B)`` the plausible equalities are the
    same family ``_two_view_family`` / ``_one_view_family`` emit over
    census op-tuples — pushed through to *this* shape's own structure:
    when a child is a unary view ``g(u)`` the candidates are the
    push-through ``g(f(u, ·))`` (re-wrap under the child's own attrs —
    ``wl``/``wr`` pick which side's when both are views, ``w`` when
    one is) and the strip ``f(u, ·)`` (``id``).  A mined pattern is
    deeper than the census's 1-level tuple, so these are claims the
    census families cannot spell; the oracles — not the mint — decide
    truth, exactly as on the shallow family.
    """
    from mlopt_core.ir import Op

    if not isinstance(pat, Op) or len(pat.args) != 2:
        return []
    f = pat.op
    a, b = pat.args
    av = isinstance(a, Op) and a.op in views and len(a.args) == 1
    bv = isinstance(b, Op) and b.op in views and len(b.args) == 1
    inner = Op.make(f, a.args[0] if av else a, b.args[0] if bv else b)
    if av and bv:
        return [
            ("wl", Op.make(a.op, inner, **dict(a.attrs))),
            ("wr", Op.make(b.op, inner, **dict(b.attrs))),
            ("id", inner),
        ]
    if av:
        return [
            ("w", Op.make(a.op, inner, **dict(a.attrs))),
            ("id", inner),
        ]
    if bv:
        return [
            ("w", Op.make(b.op, inner, **dict(b.attrs))),
            ("id", inner),
        ]
    return []


def _xanti_proposals(roots: dict, k: int) -> list:
    """Antiunify cross-module shared subgraphs into candidate rules.

    The whole corpus merges into one e-graph — congruence puts every
    structurally-equal subterm in one e-class regardless of which
    module wrote it — and ``antiunify.mine_candidates`` mines the
    recurring patterns over it (leaves become ``v*`` metavars,
    differing attr values ``a*`` attr metavars: the least-general
    generalisation of the shape's instances).  A pattern is kept when
    its LHS binds real subterms in at least *k* DISTINCT modules —
    site-recurrence inside one module is the old arm's surface.
    Each surviving shape mints the mechanical view-transform family
    (:func:`_xanti_variants`), then faces the composition arm's own
    emit-sanity gate: the instantiated RHS must denote under every
    bound binding — an ``rhs-err`` site drops the variant before it
    can spend an episode slot or the gauntlet.
    """
    import mlopt_carriers  # noqa: F401 — carrier op bindings for eval
    from mlopt_core.egraph import EGraph
    from mlopt_core.egraph.terms import _term_match
    from mlopt_core.ir import Op
    from mlopt_discovery import pipeline as pipe
    from mlopt_discovery.antiunify import mine_candidates
    from mlopt_discovery.evidence import _site_outcome
    from mlopt_discovery.pipeline import Proposal

    subs_by_mod = _op_subterms(roots)
    all_subs = [s for subs in subs_by_mod.values() for s in subs]
    eg = EGraph()
    for r in roots.values():
        eg.add_term(r)
    _pointwise, views = pipe._vocab_sets(_MINT_VOCAB)
    out: list = []
    dropped = 0
    named: set = set()
    for cand in mine_candidates(eg, min_uses=k, min_ops=2, max_ops=6):
        if len(out) >= _XANTI_CAP:
            break
        pat = cand.pattern
        if not isinstance(pat, Op) or len(pat.args) != 2:
            continue  # the variant family is binary-shaped
        mods = _modules_bound(pat, subs_by_mod)
        if len(mods) < k:
            continue
        g = pat.args[0].op if isinstance(pat.args[0], Op) else "x"
        h = pat.args[1].op if isinstance(pat.args[1], Op) else "x"
        sites = [
            (m, s)
            for s in all_subs
            if (m := _term_match(pat, s)) is not None
        ][:_XANTI_SITES]
        for tag, rhs in _xanti_variants(pat, views):
            name = f"xanti:{pat.op}_{g}_{h}_{tag}"
            if name in named:
                name = f"{name}~{len(named)}"
            rule = Proposal(
                name=name,
                lhs=pat,
                rhs=rhs,
                family=_XANTI_FAMILY,
                sources=(
                    "antiunify",
                    f"mods:{len(mods)}",
                    f"inst:{cand.n_instances}",
                ),
            ).as_rule()
            # emit-sanity — the composition arm's own gate: an
            # ill-typed mint (the oracle's ``rhs_err``) is dropped
            # at emit, never spending mint or gauntlet effort.
            if any(
                _site_outcome(rule, m, s) == "rhs-err" for m, s in sites
            ):
                dropped += 1
                continue
            named.add(name)
            out.append(
                Proposal(
                    name=name,
                    lhs=pat,
                    rhs=rhs,
                    family=_XANTI_FAMILY,
                    sources=(
                        "antiunify",
                        f"mods:{len(mods)}",
                        f"inst:{cand.n_instances}",
                    ),
                )
            )
    return out, dropped


def _xmodule_pool(roots: dict, k: int) -> tuple:
    """The cross-module pool: ``(candidates, meta, stats)``.

    Two recurring-pattern sources feed the pool, both read over the
    WHOLE corpus:

    * ``census.op_tuple_census`` — op-patterns recurring across >=
      *k* distinct modules (``terms_of`` counts module names, not
      sites) seed ``pipeline.propose`` — the same census-naturality /
      mixed-view / pattern-recognition / fusion-schema / shape-aware /
      grammar mint ``zoo-real`` runs per module, here sourced only
      from structure the modules share;
    * ``antiunify.mine_candidates`` over the merged corpus e-graph —
      deep shared subgraphs no 1-level op-tuple expresses, each
      minting the view-transform family over its own shape
      (:func:`_xanti_proposals`).

    Both are unified and alpha-deduplicated (a shape both sources
    mint merges its provenance), then every candidate faces the
    **K-bind gate**: the LHS must ``_term_match`` a real subterm in
    at least *k* distinct modules — a law that binds exactly one is
    the per-module arm's surface, not this one's.  ``meta[i]``
    records the proposer family, sources and the measured module
    count; ``stats`` is the pool-level accounting (proposal count,
    the pre-gate bind histogram, cross-tuple and xanti counts).
    """
    from mlopt_discovery import pipeline as pipe
    from mlopt_discovery.census import CorpusTerm, op_tuple_census
    from mlopt_discovery.proposal import Candidate

    terms = [CorpusTerm("zoo", n, r) for n, r in roots.items()]
    counts, terms_of = op_tuple_census(terms)
    cross = {
        key: n
        for key, n in counts.items()
        if len({m for _, m in terms_of[key]}) >= k
    }
    all_roots = list(roots.values())
    subs_by_mod = _op_subterms(roots)
    proposals = list(pipe.propose(dict(cross), all_roots, _MINT_VOCAB))
    xprops, xanti_dropped = _xanti_proposals(roots, k)
    by_key: dict = {}
    for p in [*proposals, *xprops]:
        key = alpha_key(p.lhs, p.rhs)
        cur = by_key.get(key)
        if cur is None:
            by_key[key] = p
        else:
            merged = tuple(dict.fromkeys(cur.sources + p.sources))
            if cur.family == _XANTI_FAMILY and p.family != cur.family:
                # the census mint's name/family is the informative
                # one — the xanti variant reports through it.
                cur = replace(p, sources=merged)
            else:
                cur = replace(cur, sources=merged)
            by_key[key] = cur
    cands: list = []
    meta: dict[int, dict] = {}
    bind_hist: Counter = Counter()
    named: set = set()
    for p in by_key.values():
        mods = _modules_bound(p.lhs, subs_by_mod)
        bind_hist[len(mods)] += 1
        if len(mods) < k:
            continue
        name = p.name
        if name in named:
            name = f"{name}~{len(named)}"
        named.add(name)
        rule = (
            replace(p, name=name).as_rule()
            if name != p.name
            else p.as_rule()
        )
        cands.append(
            Candidate(
                strategy="xmodule",
                label=name,
                lhs=p.lhs,
                rhs=p.rhs,
                note=", ".join(p.sources),
                rule=rule,
            )
        )
        meta[len(cands) - 1] = {
            "family": p.family,
            "sources": list(p.sources),
            "mods": len(mods),
        }
    stats = {
        "k": k,
        "cross_tuples": len(cross),
        "proposals": len(by_key),
        "xanti": len(xprops),
        "xanti_dropped": xanti_dropped,
        "bind_hist": {str(n): c for n, c in sorted(bind_hist.items())},
        "pool": len(cands),
    }
    return cands, meta, stats


def _xmodule_pool_cached(seed: int, k: int) -> tuple:
    """The run's shared pool — minted once, sliced per episode."""
    key = (seed, k)
    if key not in _XPOOL:
        env = _xmodule_env(seed)
        roots = {n: rec[0].root for n, rec in env.items() if rec}
        _XPOOL[key] = _xmodule_pool(roots, k)
    return _XPOOL[key]


class _XModuleArena(og.OpenArena):
    """An ``OpenArena`` whose pool is one page of the corpus-level mint.

    The whole-corpus anti-unified pool (:func:`_xmodule_pool`) is
    built once per ``(wseed0, xbind)`` — every episode slices a
    ``page`` off it, so the export + mining spend is paid once and a
    resume's arena rebuild re-reads the identical pool.  ``meta[i]``
    carries the proposer family, sources and measured module count
    beside each pooled candidate — rebuilt deterministically with
    the pool itself.  A mint failure lands in ``self.error`` like
    the zoo arms'; an empty slice is the honest done state.
    """

    def __init__(self, case: dict) -> None:
        self.pool: list = []
        self.meta: dict[int, dict] = {}
        self.verdicts: dict[int, str] = {}
        self.steps = 0
        self.done = True
        self._n_true = 0
        self._n_useful = 0
        self.error = ""
        self.pool_total = 0
        try:
            cands, meta, _stats = _xmodule_pool_cached(
                case.get("wseed0", 0), case.get("xbind", 2)
            )
        except Exception as e:
            self.error = f"mint: {type(e).__name__}: {e}"
            return
        page = case.get("page", 48)
        lo = case.get("xpage", 0) * page
        self.pool_total = len(cands)
        self.pool = cands[lo : lo + page]
        self.meta = {
            i: meta[lo + i]
            for i in range(len(self.pool))
            if lo + i in meta
        }
        self.done = not self.pool


# ---------------------------------------------------------------------------
#  the zoo-prim corpus arm — primitives minted for repeated manual shapes
# ---------------------------------------------------------------------------

#: Corpus-level pool cache — ``(wseed0, xbind) -> (cands, meta,
#: stats)``, the same resume-determinism contract as ``_XPOOL``.
_PPOOL: dict = {}


def _pat_spec(pat: Any) -> Any:
    """Render an ``Op`` pattern as a genkernel spec tuple.

    ``(op, *child_specs[, attrs])`` with metavar leaves *kept by
    name* (``meta_arena._spec_of`` would re-canonicalize them to
    ``X1``… — the mint needs the pattern's own names so the bound
    sites' subst keys line up) and attr metavariables kept as their
    string names inside the attr dict.
    """
    from mlopt_core.ir import Const, Op

    if isinstance(pat, Op):
        parts = [_pat_spec(a) for a in pat.args]
        if pat.attrs:
            parts.append(dict(pat.attrs))
        return (pat.op, *parts)
    if isinstance(pat, Const):
        return pat.value
    return pat  # metavar leaf — the name IS the binding key


def _pat_ops(pat: Any) -> set:
    """Op names inside a pattern."""
    from mlopt_core.ir import Op

    if not isinstance(pat, Op):
        return set()
    return {pat.op} | set().union(*(_pat_ops(a) for a in pat.args))


def _spec_bind(spec: Any, attrs: dict, attr_mvars: tuple) -> Any:
    """Instantiate a spec's attr-metavar dict values under *attrs*.

    The minted op call carries its attr metavariables as real op
    attrs (``synop_*(v0, a0=<bound>)``); at eval time the spelled
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
                out.append(_spec_bind(e, attrs, attr_mvars))
        return tuple(out) if isinstance(spec, tuple) else out
    return spec


def _prim_binding(
    spec: Any, leaf_mvars: tuple, attr_mvars: tuple
) -> Any:
    """The minted op's torch binding — its denotation, spelled out.

    ``fn(*args, **attrs) = eval(spec, {mvar: arg}, ambient)`` — the
    op IS the manual subgraph: equal to the fold's LHS at every bound
    site by construction, so the oracle/sweep/cert legs all measure
    the definitional truth.  Attr metavariables ride the call's own
    attrs back into the spec.
    """
    import mlopt_torch.torch_bridge as tb
    from mlopt_discovery.genkernel import _eval_spec

    bindings = tb._IR_TO_TORCH

    def fn(*args: Any, **attrs: Any) -> Any:
        env = dict(zip(leaf_mvars, args, strict=True))
        return _eval_spec(
            _spec_bind(spec, attrs, attr_mvars), env, bindings
        )

    return fn


def _bare_call(t: Any) -> bool:
    """Whether *t* is a fused call ``f(v0, …)`` over metavar args."""
    from mlopt_core.ir import Op

    return (
        isinstance(t, Op)
        and bool(t.args)
        and all(isinstance(a, str) for a in t.args)
    )


def _denoted_shapes() -> frozenset:
    """Alpha-normal keys of spellings a bank op already denotes.

    For every shipped rule with a bare fused call on one side —
    ``manual -> f(v*)`` folds and ``f(v*) -> manual`` expansions
    alike — the *other* side is that op's spelled denotation.
    ``lawdata.HANDLERS`` rows contribute their patterns (the
    handler-table ops' own declared bodies), and the pipeline's
    ``_FUSION_SCHEMAS`` their manual sides.  A mined shape in this
    set is already *named* — minting ``synop_*`` for it would be a
    second name for an existing fused op, not a new primitive, so the
    mint skips it and the funnel records the exclusion.
    """
    from mlopt_core.laws.priorart import PRIOR_ART_RULES
    from mlopt_discovery.antiunify import _alpha_norm
    from mlopt_discovery.lawdata import HANDLERS
    from mlopt_discovery.object_synthesis import term_from_spec
    from mlopt_discovery.pipeline import _FUSION_SCHEMAS

    from bench.drivers.screen import novelty_sieve as ns

    denoted: set[str] = set()
    for r in [*ns.known_bank(), *PRIOR_ART_RULES, *_FUSION_SCHEMAS]:
        if _bare_call(r.lhs):
            denoted.add(_alpha_norm(r.rhs))
        if _bare_call(r.rhs):
            denoted.add(_alpha_norm(r.lhs))
    for h in HANDLERS.values():
        denoted.add(_alpha_norm(term_from_spec(h["pattern"])))
    return frozenset(denoted)


def _prim_domain(sites: list, cap: int) -> list:
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


def _freeze_spec(spec: Any, subst: dict, attr_mvars: tuple) -> Any:
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
    return _spec_bind(spec, attrs, attr_mvars)


def _prim_pool(
    roots: dict,
    k: int,
    *,
    cap: int = _PRIM_CAP,
    budget: int = _PRIM_BUDGET,
    site_cap: int = _PRIM_SITES,
) -> tuple:
    """The primitive-mint pool: ``(candidates, meta, stats)``.

    For every mined shape binding >= *k* distinct modules and not
    denoted by a bank op (:func:`_denoted_shapes`), mint
    ``synop_<hash>`` — a NEW op whose denotation is the manual
    subgraph itself — plus the fold ``pattern -> synop_*(args)``,
    carrying the ``synop_*.expand`` definitional premise so the cert
    leg's anchored derivation has the axiom the bare bank lacks.

    Per-shape funnel, each refusal recorded:

    * ``kbound`` — shapes binding >= k modules (the mint's surface);
    * ``denoted`` — skipped as already-named (a shipped fold or
      handler op spells the same denotation);
    * ``unevaluable`` — an op in the pattern has no torch binding,
      so the spelled denotation can never run;
    * ``gate_failed`` — the emit gate: no bound site evaluated
      ``equal`` (or a measured ``unequal``/``rhs-err``) — the verify
      requirement from the arm's contract;
    * ``no_domain`` — every bound site left a dim undecidable, so the
      handler stage had no measured domain (the op still mints on the
      definitional denotation — the fold's soundness never depends
      on the grammar);
    * ``minted`` / ``refused`` — ``handler_synth.mint_handlers``'
      grammar verdict on the bound-site domain (a verified body
      rides the record as provenance; a refusal keeps the
      definitional op and names the measured gap).
    """
    import hashlib

    import mlopt_carriers  # noqa: F401 — carrier op bindings
    import mlopt_torch.torch_bridge as tb
    from mlopt_core.egraph import EGraph, Rewrite
    from mlopt_core.egraph.terms import _term_match
    from mlopt_core.ir import Op, op_def
    from mlopt_core.meta import instantiate_pattern
    from mlopt_discovery.antiunify import _alpha_norm, mine_candidates
    from mlopt_discovery.handler_synth import (
        handler_record,
        mint_handlers,
    )
    from mlopt_discovery.oracle import eval_instance
    from mlopt_discovery.pipeline import Proposal
    from mlopt_discovery.proposal import Candidate

    subs_by_mod = _op_subterms(roots)
    all_subs = [s for subs in subs_by_mod.values() for s in subs]
    eg = EGraph()
    for r in roots.values():
        eg.add_term(r)
    denoted = _denoted_shapes()
    stats: dict[str, Any] = {
        "k": k,
        "mined": 0,
        "kbound": 0,
        "denoted": 0,
        "unevaluable": 0,
        "gate_failed": 0,
        "no_domain": 0,
        "minted": 0,
        "refused": 0,
        "pool": 0,
    }
    cands: list = []
    meta: dict[int, dict] = {}
    for cand in mine_candidates(
        eg,
        min_uses=k,
        min_ops=_PRIM_MIN_OPS,
        max_ops=_PRIM_MAX_OPS,
    ):
        if len(cands) >= cap:
            break
        stats["mined"] += 1
        pat = cand.pattern
        if not isinstance(pat, Op) or not cand.leaf_mvars:
            continue
        mods = _modules_bound(pat, subs_by_mod)
        if len(mods) < k:
            continue
        stats["kbound"] += 1
        norm = _alpha_norm(pat)
        if norm in denoted:
            stats["denoted"] += 1
            continue
        if any(tb._IR_TO_TORCH.get(op) is None for op in _pat_ops(pat)):
            stats["unevaluable"] += 1
            continue
        # the honest name: the shape's own hash, not a coined label
        h = hashlib.sha256(norm.encode()).hexdigest()[:10]
        name = f"synop_{h}"
        call_attrs: dict[str, Any] = {a: a for a in cand.attr_mvars}
        call = Op.make(name, *cand.leaf_mvars, **call_attrs)
        spec = _pat_spec(pat)
        op_def(
            name,
            len(cand.leaf_mvars),
            1,
            law=f"synthesized primitive — denotation is the mined "
            f"manual subgraph {norm}",
        )
        # denotation = the spelled spec — the ambient binding table is
        # the op's semantics (ADR 0007: the executor table IS the
        # meaning); post-hoc registration reaches every sink seated on
        # the shared ambient dict.
        tb._IR_TO_TORCH[name] = _prim_binding(
            spec, cand.leaf_mvars, cand.attr_mvars
        )
        sites = [
            (m, s)
            for s in all_subs
            if (m := _term_match(pat, s)) is not None
        ][:site_cap]
        # verify — the arm's honesty gate: the op's denotation must
        # equal the spelled pattern fp64 on bound sites.  By
        # construction it does (the binding IS the spelling) — the
        # measurement is still run, never assumed.
        equal = bad = 0
        for m, _s in sites:
            outcome, _n = eval_instance(
                instantiate_pattern(pat, m),
                instantiate_pattern(call, m),
            )
            equal += outcome == "equal"
            bad += outcome in ("unequal", "rhs-err")
        if not equal or bad:
            stats["gate_failed"] += 1
            continue
        domain = _prim_domain(sites, _PRIM_DOMAIN)
        hrow: dict[str, Any] = {"verdict": "skipped-no-domain"}
        if not domain:
            stats["no_domain"] += 1
        else:
            _handlers, report = mint_handlers(
                [
                    {
                        "spec": _freeze_spec(
                            spec, sites[0][0], cand.attr_mvars
                        ),
                        "domain": domain,
                    }
                ],
                budget=budget,
            )
            rep = report[0]
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
            else:
                stats["refused"] += 1
                hrow = {
                    "verdict": "refused",
                    "tried": rep.get("tried"),
                    "note": rep.get("note"),
                }
        # the definitional premise: the cert leg's anchored axiom set
        # unions it when the bare bank cannot derive the fold — the
        # fusion-schema mint's own channel for a denotation the bank
        # lacks.  The novelty sieve never sees it (its premise bank
        # is the shipped surface minus alpha-siblings), so the fold
        # stays underivable there BY CONSTRUCTION.
        expand = Rewrite(
            f"{name}.expand",
            call,
            pat,
            law=f"{name} unfolds to its mined denotation — the "
            "definitional premise the cert leg anchors on",
        )
        prop = Proposal(
            name=f"prim:{name}",
            lhs=pat,
            rhs=call,
            family=_PRIM_FAMILY,
            sources=(
                "prim-mint",
                f"mods:{len(mods)}",
                f"inst:{cand.n_instances}",
                f"gain:{cand.gain}",
                f"handler:{hrow['verdict']}",
            ),
            premises=(expand,),
        )
        i = len(cands)
        cands.append(
            Candidate(
                strategy="prim",
                label=prop.name,
                lhs=pat,
                rhs=call,
                note=(
                    f"primitive {name} — a manual shape shared by "
                    f"{len(mods)} modules, undenoted in the bank"
                ),
                rule=prop.as_rule(),
            )
        )
        meta[i] = {
            "family": _PRIM_FAMILY,
            "sources": list(prop.sources),
            "mods": len(mods),
            "prim": {
                "synop": name,
                "alpha": norm,
                "mods": sorted(mods),
                "sites": len(sites),
                "equal_sites": equal,
                "handler": hrow,
            },
        }
    stats["pool"] = len(cands)
    return cands, meta, stats


def _prim_pool_cached(seed: int, k: int) -> tuple:
    """The run's shared prim pool — minted once, sliced per episode.

    Minting registers the ``synop_*`` ops + bindings, so calling this
    before the episode loop is also what makes a resume's rebuilt
    candidate cells evaluable again (names are content-hashed — a
    rebuild re-registers byte-identical bindings).
    """
    key = (seed, k)
    if key not in _PPOOL:
        env = _xmodule_env(seed)
        roots = {n: rec[0].root for n, rec in env.items() if rec}
        _PPOOL[key] = _prim_pool(roots, k)
    return _PPOOL[key]


class _PrimArena(og.OpenArena):
    """An ``OpenArena`` whose pool is one page of the prim mint.

    Same contract as :class:`_XModuleArena`: the corpus-level pool is
    built once per ``(wseed0, xbind)`` and episodes slice ``page``
    candidates off it; ``meta[i]`` rides the ledger with the
    primitive's own provenance (the minted ``synop_*`` name, the
    module set, the site-verify counts and the handler verdict).
    """

    def __init__(self, case: dict) -> None:
        self.pool: list = []
        self.meta: dict[int, dict] = {}
        self.verdicts: dict[int, str] = {}
        self.steps = 0
        self.done = True
        self._n_true = 0
        self._n_useful = 0
        self.error = ""
        self.pool_total = 0
        try:
            cands, meta, _stats = _prim_pool_cached(
                case.get("wseed0", 0), case.get("xbind", 2)
            )
        except Exception as e:
            self.error = f"mint: {type(e).__name__}: {e}"
            return
        page = case.get("page", 48)
        lo = case.get("xpage", 0) * page
        self.pool_total = len(cands)
        self.pool = cands[lo : lo + page]
        self.meta = {
            i: meta[lo + i]
            for i in range(len(self.pool))
            if lo + i in meta
        }
        self.done = not self.pool


def _gen_cases(
    corpus: str, seed: int, episodes: int, **kw: Any
) -> list:
    """The corpus arm's case list — stream pages or zoo workloads."""
    if corpus in ("zoo-xmodule", "zoo-prim"):
        return _xmodule_cases(
            seed,
            episodes,
            page=kw["page"],
            xbind=kw.get("xbind", 2),
        )
    if corpus in ("zoo", "zoo-real"):
        return _zoo_cases(
            seed,
            episodes,
            page=kw["page"],
            scan=kw.get("scan") or 0,
            # the composition arm is zoo-real only — a ``compose``
            # value on any other corpus is a protocol lie, zeroed here
            # and refused loudly at the CLI.
            compose=kw.get("compose", 0) if corpus == "zoo-real" else 0,
        )
    return og.gen_cases(seed, episodes, page=kw["page"])


def _arena_for(case: dict, corpus: str) -> Any:
    """The episode board for *case* on *corpus*."""
    if corpus == "zoo":
        return _ZooArena(case)
    if corpus == "zoo-real":
        return _RealArena(case)
    if corpus == "zoo-xmodule":
        return _XModuleArena(case)
    if corpus == "zoo-prim":
        return _PrimArena(case)
    return og.OpenArena(case)


# ---------------------------------------------------------------------------
#  mint + admit
# ---------------------------------------------------------------------------


def _eval_records(traj: Any) -> list[dict]:
    """One ledger row per ``evaluate`` step of the trajectory."""
    out = []
    for rep in traj.reports:
        if rep.action.op != "evaluate":
            continue
        cand = rep.action.params["cand"]
        out.append(
            {
                "i": rep.action.params["i"],
                "name": cand.as_rule().name,
                "label": rep.detail.get("label", cand.label),
                "verdict": rep.detail.get("verdict", ""),
                "relation": rep.detail.get("relation", ""),
                "num_true": rep.detail.get("num_true"),
                "base_cost": rep.detail.get("base_cost"),
                "cand_cost": rep.detail.get("cand_cost"),
            }
        )
    return out


def _admit(rule: Any, referee: str, ctx: Any) -> dict:
    """Run the referee gauntlet; return the candidate-cell payload.

    A synthesized composite's law record additionally carries the
    ``"guard"`` premise-closure spec — ``parents`` plus the ``pat1`` /
    ``pat2`` re-expression maps (``evidence._guard_transport_spec``,
    the same shape ``mlopt_core.laws.minted`` embeds): the procedural
    ``check``/``derive`` stay honestly listed in ``missing_hooks``,
    and the spec records exactly which parent objects the transported
    hooks close over.
    """
    from mlopt_discovery.evidence import _guard_transport_spec

    from bench import referees

    law = law_to_data(rule)
    guard = _guard_transport_spec(rule)
    if guard is not None:
        law["guard"] = guard
    premises = getattr(rule, "premises", None)
    if premises:
        # the cert leg's premise-anchored axiom names — the same
        # honesty posture as ``law["guard"]``: record what the
        # closure adds so an audit sees the anchor.  The premise
        # objects themselves re-attach on the deterministic re-mint
        # (``Proposal.premises`` → ``as_rule``), like the composite
        # hooks the pool rebuild restores.
        law["premises"] = [r.name for r in premises]

    rec: dict[str, Any] = {
        "name": rule.name,
        "provenance": law_provenance(rule),
        "law": law,
        "alpha": alpha_key(rule.lhs, rule.rhs),
        "referee": referee,
        "skipped": "",
        "admitted": False,
        "stages": [],
        "measured": {},
        "reason": "",
    }
    try:
        verdict = referees.judge(referee, rule, ctx)
    except referees.SkipReferee as e:
        rec["skipped"] = str(e)
        rec["reason"] = f"skipped: {e}"
        return rec
    except Exception as e:  # a component that throws is a refusal
        rec["reason"] = f"{type(e).__name__}: {e}"
        return rec
    rec["admitted"] = bool(verdict.passed)
    rec["stages"] = [
        {"name": s.name, "passed": bool(s.passed), "detail": s.detail}
        for s in verdict.stages
    ]
    rec["measured"] = dict(verdict.measured)
    rec["reason"] = verdict.reason
    return rec


# ---------------------------------------------------------------------------
#  classify — the novelty verdict against the prior-art bank
# ---------------------------------------------------------------------------


def _classify(rule: Any, bank: list, ctxs: list | None = None) -> dict:
    """``machine-original`` / ``known-art`` / ``vacuous`` for *rule*.

    The bank is the novelty sieve's widest shipped surface; the
    alpha-sibling check additionally scans the raw
    ``PRIOR_ART_RULES`` so a verbatim rediscovery of known art is
    named even when the derivability slice dropped it.  *ctxs* is the
    corpus ``instance_of`` searches for a real firing site — empty on
    the grammar arm, the covered zoo modules' ``_CaseCtx``s on the
    zoo arm (an instance then lands ``corpus:<module>`` provenance).
    """
    from mlopt_core.laws.priorart import PRIOR_ART_RULES

    from bench.drivers.screen import novelty_sieve as ns

    ctxs = ctxs or []
    key = alpha_key(rule.lhs, rule.rhs)
    sibs = sorted(
        {
            r.name
            for r in [*bank, *PRIOR_ART_RULES]
            if alpha_key(r.lhs, r.rhs) == key
        }
    )
    rec: dict[str, Any] = {
        "verdict": "",
        "via": "",
        "siblings": sibs,
        "derivable": None,
        "witness_rules": [],
        "instance_src": "",
    }
    if sibs:
        rec.update(verdict="known-art", via="alpha-match")
        return rec
    got = ns.instance_of(rule, ctxs)
    if got is None:
        rec.update(verdict="vacuous", via="no-instance")
        return rec
    src, (lhs_i, rhs_i) = got
    rec["instance_src"] = src
    premises, _s = ns._premises(rule, bank)
    res = verify_law(lhs_i, rhs_i, premises)
    rec["derivable"] = bool(res.derivable)
    rec["witness_rules"] = list(res.witness_rules)
    rec.update(
        verdict="known-art" if res.derivable else "machine-original",
        via="derivable" if res.derivable else "underivable",
    )
    return rec


# ---------------------------------------------------------------------------
#  the loop
# ---------------------------------------------------------------------------


def run(
    *,
    episodes: int,
    page: int,
    budget: int,
    seed: int,
    player: str,
    referee: str,
    corpus: str = "grammar",
    scan: int = 0,
    compose: int = 0,
    xbind: int = 2,
    ckpt: Any,
    ctx: Any = None,
    arm_factory: Any = None,
    cases: list | None = None,
    log: Any = print,
) -> dict:
    """Run the loop against *ckpt* (an open ``Checkpoint``); summarize.

    ``arm_factory`` is the registry's ``i -> player`` seam — tests
    inject players through it; production callers leave it ``None``
    and the named registered player is built.  ``cases`` overrides
    the generated case list — each is an arena case dict (a stream
    page on ``grammar``, ``{"zoo": name, ...}`` on the zoo arms).
    ``corpus`` picks the arm; ``scan`` bounds the zoo arm's
    per-module stream read (``<= 0`` scans the whole stream; unused
    on ``zoo-real`` — the mint reads the module's own term, not the
    stream).  ``compose`` is the zoo-real composition arm's stratum
    count — ``meta.synthesize_rules`` rounds over the guarded bank
    (the module's guarded parents plus the bound pool), seeded on
    each module's term (0 = off; ignored on other corpora).
    ``xbind`` is the zoo-xmodule arm's K — the distinct-module count
    a minted claim's LHS must bind to pool (2+ is the cross-module
    surface; ignored on other corpora).
    """
    import torch

    from bench import referees

    torch.manual_seed(seed)
    if cases is None:
        cases = _gen_cases(
            corpus,
            seed,
            episodes,
            page=page,
            scan=scan,
            compose=compose,
            xbind=xbind,
        )
    zenv: dict | None = None
    xstats: dict | None = None
    pstats: dict | None = None
    if corpus == "zoo-xmodule":
        wseed0 = cases[0].get("wseed0", seed) if cases else seed
        xk = cases[0].get("xbind", xbind) if cases else xbind
        zenv = _xmodule_env(wseed0)
        # the shared pool is minted once here — the arenas slice
        # the same cached build, never re-exporting per episode.
        xstats = _xmodule_pool_cached(wseed0, xk)[2]
        if ctx is None:
            ctx = _real_referee_ctx(zenv, referees)
    elif corpus == "zoo-prim":
        wseed0 = cases[0].get("wseed0", seed) if cases else seed
        xk = cases[0].get("xbind", xbind) if cases else xbind
        zenv = _xmodule_env(wseed0)
        # the shared pool mints once here — registering the synop_*
        # ops + bindings before any episode, admit or classify leg
        # touches them (a resume's pool rebuild reproduces the
        # registrations verbatim — names are content-hashed).
        pstats = _prim_pool_cached(wseed0, xk)[2]
        if ctx is None:
            ctx = _real_referee_ctx(zenv, referees)
    elif corpus in ("zoo", "zoo-real"):
        zenv = _zoo_export_env(cases)
        if ctx is None:
            ctx = (
                _real_referee_ctx(zenv, referees)
                if corpus == "zoo-real"
                else _zoo_referee_ctx(zenv, referees)
            )
    ctx = referees.RefereeCtx() if ctx is None else ctx
    arm = arm_factory or players.make(player, "open", seed=seed, ctx={})
    bank: list | None = None
    zctxs: list | None = None
    t0 = time.time()

    for i, case in enumerate(cases):
        ekey = ("episode", i)
        arena = None
        if ckpt.done(ekey):
            evals = ckpt.get(ekey)["evals"]
        else:
            arena = _arena_for(case, corpus)
            traj = engine.run_episode(arena, arm(i), budget)
            evals = _eval_records(traj)
            payload: dict[str, Any] = {
                "case": case,
                "steps": len(traj.reports),
                "reward": round(traj.total, 6),
                "evals": evals,
            }
            if getattr(arena, "error", ""):
                payload["error"] = arena.error
            ckpt.record(ekey, payload)
            log(
                f"  episode {i:>3}: {len(evals)} evals "
                f"reward={traj.total:+.3f}",
            )

        # -- mint + admit: one cell per distinct minted rule --------
        missing = [
            ev
            for ev in evals
            if not ckpt.done(("candidate", ev["name"]))
        ]
        pool = None
        for ev in missing:
            if pool is None:
                # the arena pool is deterministic — resume recovers
                # the candidate by index, no episode re-run.
                if arena is None:
                    arena = _arena_for(case, corpus)
                pool = arena.pool
            cand = pool[ev["i"]]
            rule = cand.as_rule()
            rec = _admit(rule, referee, ctx)
            meta = getattr(arena, "meta", {}).get(ev["i"])
            if meta is not None:
                rec["family"] = meta["family"]
                rec["sources"] = meta["sources"]
                if "mods" in meta:
                    # zoo-xmodule / zoo-prim — how many distinct
                    # modules the minted LHS binds (the arms' own
                    # measurement).
                    rec["mods"] = meta["mods"]
                if "prim" in meta:
                    # zoo-prim — the minted op's own record: name,
                    # site-verify counts, handler verdict + record.
                    rec["prim"] = meta["prim"]
            rec["label"] = ev["label"]
            rec["episode"] = i
            rec["arena"] = {
                "verdict": ev["verdict"],
                "relation": ev["relation"],
                "num_true": ev["num_true"],
                "base_cost": ev["base_cost"],
                "cand_cost": ev["cand_cost"],
            }
            ckpt.record(("candidate", ev["name"]), rec)
            log(
                f"    candidate {ev['name']:<28} "
                f"admitted={rec['admitted']} ({rec['reason'][:60]})",
            )

        # -- classify: one cell per admitted law --------------------
        for ev in evals:
            ckey = ("classify", ev["name"])
            if ckpt.done(ckey):
                continue
            crec = ckpt.get(("candidate", ev["name"]))
            if not crec["admitted"]:
                continue
            if bank is None:
                from bench.drivers.screen import novelty_sieve as ns

                bank = ns.known_bank()
            if (
                corpus in ("zoo", "zoo-real", "zoo-xmodule", "zoo-prim")
                and zctxs is None
            ):
                zctxs = (
                    _zoo_case_ctxs(zenv or {})
                    if corpus == "zoo"
                    else _real_case_ctxs(zenv or {})
                )
            rule = law_from_data(crec["law"])
            if (crec["law"] or {}).get("guard"):
                # A composite's transported check/derive are code the
                # record cannot carry — ``missing_hooks`` says so.
                # The pool rebuilds deterministically, so the live
                # mint is recovered by index and the sieve judges the
                # guarded rule, not the bare pattern: ``instance_of``
                # and the derivability gate both consult the hooks.
                if pool is None:
                    if arena is None:
                        arena = _arena_for(case, corpus)
                    pool = arena.pool
                if (
                    ev["i"] < len(pool)
                    and pool[ev["i"]].as_rule().name == crec["name"]
                ):
                    rule = pool[ev["i"]].as_rule()
            cres = _classify(rule, bank, ctxs=zctxs)
            ckpt.record(ckey, cres)
            log(
                f"    classify {ev['name']:<26} "
                f"{cres['verdict']} ({cres['via']})",
            )

    # -- the ledger summary -----------------------------------------
    names: list[str] = []
    for i in range(len(cases)):
        for ev in ckpt.get(("episode", i))["evals"]:
            if ev["name"] not in names:
                names.append(ev["name"])
    cands = [ckpt.get(("candidate", n)) for n in names]
    leg_deaths: Counter = Counter()
    for rec in cands:
        if rec["skipped"]:
            leg_deaths["skipped"] += 1
            continue
        if rec["admitted"]:
            leg_deaths["admitted"] += 1
            continue
        first = next(
            (s["name"] for s in rec["stages"] if not s["passed"]),
            "error",
        )
        leg_deaths[first] += 1
    novelty: Counter = Counter()
    finds: list[dict] = []
    for rec in cands:
        if not rec["admitted"]:
            continue
        cres = ckpt.get(("classify", rec["name"]))
        if cres is None:
            continue
        novelty[cres["verdict"]] += 1
        if cres["verdict"] == "machine-original":
            finds.append(
                {
                    "name": rec["name"],
                    "instance_src": cres["instance_src"],
                    "arena_verdict": rec["arena"]["verdict"],
                    "provenance": rec["provenance"],
                }
            )
    arena = Counter(
        ev["verdict"]
        for i in range(len(cases))
        for ev in ckpt.get(("episode", i))["evals"]
    )
    out: dict[str, Any] = {
        "schema": 1,
        "protocol": dict(ckpt.manifest.get("protocol", {})),
        "corpus": corpus,
        "episodes": len(cases),
        "evaluated": sum(arena.values()),
        "arena_verdicts": dict(arena),
        "minted": len(cands),
        "composed": sum(
            1 for r in cands if r.get("family") == _COMPOSE_FAMILY
        ),
        "guarded": sum(
            1
            for r in cands
            if r.get("family") == _COMPOSE_FAMILY
            and (r.get("law") or {}).get("missing_hooks")
        ),
        "admitted": sum(1 for r in cands if r["admitted"]),
        "skipped": sum(1 for r in cands if r["skipped"]),
        "leg_refusals": dict(leg_deaths),
        "novelty": dict(novelty),
        "finds": finds,
        "candidates": [
            {
                "name": r["name"],
                "family": r.get("family", ""),
                "mods": r.get("mods"),
                "arena": r["arena"],
                "admitted": r["admitted"],
                "skipped": r["skipped"],
                "reason": r["reason"],
                "stages": r["stages"],
                "novelty": (
                    ckpt.get(("classify", r["name"])) or {}
                ).get("verdict", ""),
            }
            for r in cands
        ],
        "wall_s": round(time.time() - t0, 1),
    }
    if corpus == "zoo-xmodule" and xstats is not None:
        # the arm's own accounting: pool shape, the pre-gate bind
        # histogram, and the minted candidates' distinct-module counts
        out["xmodule"] = xstats
        out["bind_dist"] = dict(
            sorted(
                Counter(r["mods"] for r in cands if "mods" in r).items()
            )
        )
    if corpus == "zoo-prim" and pstats is not None:
        # the arm's own accounting: the shape funnel — mined, K-bound,
        # denoted/unevaluable/gate/no-domain exclusions, grammar
        # verdicts, pool size — plus the minted ops' records.
        out["prim"] = pstats
        out["prim_ops"] = {
            r["name"]: r["prim"]
            for r in cands
            if r.get("prim") is not None
        }
        out["bind_dist"] = dict(
            sorted(
                Counter(r["mods"] for r in cands if "mods" in r).items()
            )
        )
    return out


# ---------------------------------------------------------------------------
#  emit
# ---------------------------------------------------------------------------


def _json_path(out: Path) -> Path:
    return (
        out.with_suffix(".json")
        if out.suffix
        else out.parent / f"{out.name}.json"
    )


def render_markdown(summary: dict, *, env: dict | None = None) -> str:
    """The human document — the funnel, the novelty split, the finds."""
    from bench import referees

    env = env or {}
    legs = summary["leg_refusals"]
    nov = summary["novelty"]
    lines = [
        "# Overnight discovery run",
        "",
        "Episodes -> mint -> admit -> classify.  Open-game episodes "
        "mint candidate rewrites; each minted claim is judged by the "
        "``bench/referees`` gauntlet (oracle -> pay -> sweep -> "
        "cert); admitted laws are classified against the prior-art "
        "bank: **machine-original** (the widest shipped bank cannot "
        "derive it), **known-art** (an alpha-sibling exists or the "
        "bank proves it), **vacuous** (no binding produced a site — "
        "unevaluable).",
        "",
        f"- date: {env.get('timestamp_utc', '—')} · "
        f"git: `{env.get('git_sha', 'unknown')}`"
        f"{' (dirty)' if env.get('git_dirty') else ''} · "
        f"device {env.get('device', '?')}",
        f"- protocol: `{json.dumps(summary['protocol'], sort_keys=True)}`",
        f"- corpus: `{summary.get('corpus', 'grammar')}`"
        + (
            {
                "zoo": " — held-out zoo modules "
                "(mlopt_discovery.zoo), the stream filtered to "
                "real-matching claims",
                "zoo-real": " — held-out zoo modules "
                "(mlopt_discovery.zoo), candidates minted from each "
                "module's own op-tuple census (pipeline.propose); "
                "real_terms are the corpus's distinct subterms",
                "zoo-xmodule": " — one corpus-level mint: "
                "op-patterns and antiunified deep shapes recurring "
                "across >= --xbind distinct modules seed "
                "pipeline.propose + the xanti family; every claim "
                "must bind >= K real modules to pool",
                "zoo-prim": " — the primitive mint: manual shapes "
                "recurring across >= --xbind distinct modules and "
                "denoted by no bank op mint NEW synop_* primitives "
                "(denotation = the subgraph; the fold is "
                "novel-by-construction)",
            }.get(
                summary.get("corpus", "grammar"),
                " — the enumerated grammar stream",
            )
        ),
        f"- episodes: {summary['episodes']} · "
        f"evaluated: {summary['evaluated']} · "
        f"minted: {summary['minted']}"
        + (
            f" ({summary['composed']} composites"
            + (
                f", {summary['guarded']} guarded"
                if summary.get("guarded")
                else ""
            )
            + ")"
            if summary.get("composed")
            else ""
        )
        + f" · admitted: {summary['admitted']} · "
        f"wall: {summary['wall_s']}s",
        "- arena verdicts: "
        + (
            ", ".join(
                f"{k}={v}"
                for k, v in sorted(summary["arena_verdicts"].items())
            )
            or "—"
        ),
    ]
    if summary.get("xmodule") is not None:
        x = summary["xmodule"]
        lines.append(
            f"- cross-module mint: k={x['k']} · "
            f"cross-tuples={x['cross_tuples']} · "
            f"proposals={x['proposals']} · "
            f"xanti={x['xanti']} (+{x['xanti_dropped']} emit-gated) · "
            f"pool={x['pool']} · "
            f"bind histogram (pre-gate): {x['bind_hist']}"
        )
        if summary.get("bind_dist"):
            lines.append(
                "- minted candidates' module-bind counts: "
                + ", ".join(
                    f"{n} mods x{c}"
                    for n, c in summary["bind_dist"].items()
                )
            )
    if summary.get("prim") is not None:
        p = summary["prim"]
        lines.append(
            f"- primitive mint: k={p['k']} · "
            f"mined={p['mined']} · "
            f"k-bound={p['kbound']} · "
            f"denoted={p['denoted']} · "
            f"unevaluable={p['unevaluable']} · "
            f"gate-failed={p['gate_failed']} · "
            f"no-domain={p['no_domain']} · "
            f"handler minted={p['minted']} refused={p['refused']} · "
            f"pool={p['pool']}"
        )
        if summary.get("bind_dist"):
            lines.append(
                "- minted folds' module-bind counts: "
                + ", ".join(
                    f"{n} mods x{c}"
                    for n, c in summary["bind_dist"].items()
                )
            )
        if summary.get("prim_ops"):
            lines.append("- minted primitives:")
            for cname, rec in summary["prim_ops"].items():
                h = rec["handler"]
                lines.append(
                    f"  - `{rec['synop']}` ({cname}) — "
                    f"{len(rec['mods'])} mods, "
                    f"{rec['equal_sites']}/{rec['sites']} equal sites, "
                    f"handler={h['verdict']}"
                    + (
                        f" ({h['family']}, err={h['max_err']:.3g})"
                        if h["verdict"] == "minted"
                        else ""
                    )
                )
    lines += [
        "",
        "## Admission funnel",
        "",
        "| first-failed leg | count |",
        "|---|---|",
    ]
    for leg in (*referees.COMPONENT_ORDER, "skipped", "error"):
        if legs.get(leg):
            lines.append(f"| {leg} | {legs[leg]} |")
    lines.append(f"| **admitted** | {legs.get('admitted', 0)} |")
    lines += [
        "",
        "## Novelty",
        "",
        "| verdict | count |",
        "|---|---|",
    ]
    for v in _NOVELTY_ORDER:
        lines.append(f"| {v} | {nov.get(v, 0)} |")
    lines += ["", "## Machine-original finds", ""]
    if summary["finds"]:
        for f in summary["finds"]:
            lines.append(
                f"- `{f['name']}` — arena `{f['arena_verdict']}`, "
                f"instance {f['instance_src'] or '—'}"
            )
    else:
        lines.append(
            "**None.** Every admitted law was already a theorem of "
            "the shipped bank — or nothing survived the gauntlet."
        )
    lines += [
        "",
        "## Candidates",
        "",
        "| candidate | arena | admitted | reason | novelty |",
        "|---|---|---|---|---|",
    ]
    for c in summary["candidates"]:
        reason = (c["reason"] or "—").replace("|", "\\|")
        lines.append(
            f"| `{c['name']}` | {c['arena'].get('verdict', '—')} "
            f"| {'yes' if c['admitted'] else 'no'} "
            f"| {reason} | {c['novelty'] or '—'} |"
        )
    return "\n".join(lines) + "\n"


def emit(summary: dict, out: Path, *, env: dict, seed: Any) -> Path:
    """Write ``out`` (.md) and the sibling .json, provenance-stamped."""
    from bench.benchkit.provenance import with_provenance

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(summary, env=env))
    payload = with_provenance(summary, seed=seed, device="cpu", env=env)
    _json_path(out).write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    return out


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the overnight loop; emit the result documents."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--episodes",
        type=int,
        default=DEFAULTS["episodes"],
        help="episode count — 24 pages of 48 covers the stream once",
    )
    ap.add_argument(
        "--page", type=int, default=DEFAULTS["page"], help="page size"
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=DEFAULTS["budget"],
        help="per-episode action budget",
    )
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument(
        "--player",
        default=DEFAULTS["player"],
        help="a registered players/ name (default playbook)",
    )
    ap.add_argument(
        "--referee",
        default=DEFAULTS["referee"],
        help="a registered bench/referees arm (default full)",
    )
    ap.add_argument(
        "--corpus",
        choices=CORPORA,
        default=DEFAULTS["corpus"],
        help="the episode corpus — 'grammar' pages the enumerated "
        "candidate stream; 'zoo' pages the held-out model zoo "
        "(mlopt_discovery.zoo), each module's exported term "
        "filtering the stream to claims that fire on real structure; "
        "'zoo-real' drops the stream: each episode's pool is minted "
        "from the module's own op-tuple census (pipeline.propose — "
        "the mul_unsq_nat path), binding-first (require_bind — a "
        "candidate that matches no real subterm is never minted), "
        "and judged against the corpus's distinct real subterms; "
        "'zoo-xmodule' anti-unifies over the WHOLE corpus — one "
        "shared pool minted from the op-patterns and deep shapes "
        "recurring across >= --xbind distinct modules, episodes "
        "paging it like the stream arm; "
        "'zoo-prim' mints NEW synop_* primitive ops for the "
        "recurring manual shapes no bank op denotes and folds them "
        "(manual -> synop_*), laws novel by construction",
    )
    ap.add_argument(
        "--compose",
        type=int,
        nargs="?",
        const=2,
        default=DEFAULTS["compose"],
        metavar="ROUNDS",
        help="zoo-real only: after the single-step pool binds, "
        "pair-compose it through meta.synthesize_rules — the syn_* "
        "mint fixpoint conv_bn_fold shipped through — seeded on each "
        "module's own term, for ROUNDS strata (default 2), over a "
        "bank that also carries the module's guarded parents "
        "(shipped check/derive laws that fire on the term, so a "
        "composite can inherit a transported side condition).  "
        "Emitted "
        "composites face the same bind gate plus a seed-eval gate "
        "(the instantiated RHS must denote under the bound env — an "
        "ill-typed composite is dropped before the gauntlet) and "
        "carry their premise-closure guard spec into the ledger.  "
        "Gated strata chain: each round's viable composites re-seat "
        "as parents for the next, enumerated ahead of the base bank "
        "so the fuel reaches them, until a round mints no new viable "
        "parent — the fixpoint (mint: corpus-bind-compose-strata — "
        "ledgers restart loudly)",
    )
    ap.add_argument(
        "--xbind",
        type=int,
        default=DEFAULTS["xbind"],
        metavar="K",
        help="zoo-xmodule / zoo-prim only: the distinct-module bind "
        "threshold — a minted claim pools only when its LHS binds "
        "real subterms in >= K different modules (K=1 is the "
        "per-module arm's surface; K=2+ is the cross-module one; "
        "default 2)",
    )
    ap.add_argument(
        "--scan",
        type=int,
        default=DEFAULTS["scan"],
        help="zoo arm: stream candidates scanned per module "
        "(<=0 unbounded; the stream is ~1.1k).  Unused on zoo-real — "
        "the mint reads the module's term, not the stream",
    )
    ap.add_argument(
        "--dry",
        action="store_true",
        help="seconds-scale smoke: 2 episodes, page 6, budget 4",
    )
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help=f"append-only ledger (default {CKPT_PATH}; "
        f"{CKPT_PATH_REAL} on --corpus zoo-real)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="reuse the ledger's done cells (protocol-verified); "
        "without it the ledger restarts fresh",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"markdown output (default {DOC_PATH}; "
        f"{DOC_PATH_REAL} on --corpus zoo-real); the .json "
        "lands beside it",
    )
    args = ap.parse_args(argv)
    if args.compose and args.corpus != "zoo-real":
        ap.error("--compose requires --corpus zoo-real")
    if args.compose < 0:
        ap.error("--compose ROUNDS must be >= 1 (omit to disable)")
    if args.xbind < 1:
        ap.error("--xbind K must be >= 1")
    if args.dry:
        for k, v in DRY.items():
            setattr(args, k, v)
    if args.ckpt is None:
        args.ckpt = {
            "zoo-real": CKPT_PATH_REAL,
            "zoo-xmodule": CKPT_PATH_XMODULE,
            "zoo-prim": CKPT_PATH_PRIM,
        }.get(args.corpus, CKPT_PATH)
    if args.out is None:
        args.out = {
            "zoo-real": DOC_PATH_REAL,
            "zoo-xmodule": DOC_PATH_XMODULE,
            "zoo-prim": DOC_PATH_PRIM,
        }.get(args.corpus, DOC_PATH)

    #  Standalone runs put ``bench/drivers`` on sys.path — the root
    #  must be present before the ``bench.*`` imports below.
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    from bench.benchkit import checkpoint as ck_mod
    from bench.benchkit.env import collect_env

    protocol = {
        "episodes": args.episodes,
        "page": args.page,
        "budget": args.budget,
        "seed": args.seed,
        "player": args.player,
        "referee": args.referee,
        "corpus": args.corpus,
        "scan": args.scan,
        "dry": bool(args.dry),
        # the mint semantics — a ledger written before the zoo-real
        # arm went binding-first (require_bind) must restart loudly
        # rather than replay ungated pools; likewise a ledger minted
        # without the composition arm — or before composites had to
        # denote under the seed binding, or before the synthesis bank
        # carried guarded parents — never replays under it.
        "mint": (
            "corpus-xbind"
            if args.corpus == "zoo-xmodule"
            else "corpus-prim"
            if args.corpus == "zoo-prim"
            else "corpus-bind-compose-strata"
            if args.corpus == "zoo-real" and args.compose
            else "corpus-bind"
            if args.corpus == "zoo-real"
            else "stream"
        ),
        "compose": args.compose if args.corpus == "zoo-real" else 0,
        # the K-bind threshold is part of the mint — a ledger pooled
        # under a different module-count gate restarts loudly rather
        # than replay a pool this arm would gate differently.
        "xbind": (
            args.xbind
            if args.corpus in ("zoo-xmodule", "zoo-prim")
            else 0
        ),
        # the fusion-schema arm joined ``pipeline.propose`` — a ledger
        # minted before the family existed must restart loudly rather
        # than replay a pool the new arm would now grow.
        "fusion": args.corpus in ("zoo-real", "zoo-xmodule"),
    }
    if not args.resume and args.ckpt.exists():
        args.ckpt.unlink()
    manifest = ck_mod.new_manifest("overnight", protocol, device="cpu")
    ckpt = ck_mod.Checkpoint.open(args.ckpt, manifest)
    if ckpt.entries:
        print(  # stdout-compat
            f"resume: {len(ckpt.entries)} cells already done "
            f"({args.ckpt})",
            flush=True,
        )

    print(  # stdout-compat
        f"== overnight — {args.episodes} episodes, page {args.page}, "
        f"player {args.player}, referee {args.referee}, "
        f"corpus {args.corpus} =="
    )
    summary = run(
        episodes=args.episodes,
        page=args.page,
        budget=args.budget,
        seed=args.seed,
        player=args.player,
        referee=args.referee,
        corpus=args.corpus,
        scan=args.scan,
        compose=args.compose if args.corpus == "zoo-real" else 0,
        xbind=(
            args.xbind
            if args.corpus in ("zoo-xmodule", "zoo-prim")
            else 0
        ),
        ckpt=ckpt,
    )

    env = collect_env("cpu")
    out = emit(summary, args.out, env=env, seed=args.seed)
    print(f"wrote {out} (+ {_json_path(out).name})")  # stdout-compat
    print(  # stdout-compat
        f"episodes={summary['episodes']} evaluated={summary['evaluated']} "
        f"minted={summary['minted']} admitted={summary['admitted']} "
        f"novelty={summary['novelty']} wall={summary['wall_s']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

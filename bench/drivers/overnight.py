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
mixed-view + pattern-recognition + shape-aware + grammar — the same
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
(default 2).  Every emitted composite then faces the arm's own gate:
it is kept only when its LHS binds a real subterm, and it enters the
pool with its transported premise closure — ``_compose_guards``
check/derive hooks plus the ``parents`` / ``guard_pats`` record —
serialized into its candidate cell's ``law["guard"]`` like the
``mlopt_core.laws.minted`` rows.  Machine-originals live here: a
composite the prior-art bank cannot derive is the sieve's target
verdict.  The arm is protocol-pinned as ``mint:
corpus-bind-compose``, so a ledger written without it restarts
loudly.

Run::

    .venv/bin/python bench/drivers/overnight.py --dry
    .venv/bin/python bench/drivers/overnight.py --dry --corpus zoo
    .venv/bin/python bench/drivers/overnight.py --dry --corpus zoo-real
    .venv/bin/python bench/drivers/overnight.py --dry --corpus zoo-real --compose
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
#: mint — the ``mul_unsq_nat`` path).
CORPORA: tuple[str, ...] = ("grammar", "zoo", "zoo-real")

#: Where the documents land.
DOC_PATH = _REPO / "docs" / "results_overnight.md"

#: The zoo-real arm's own document — a different measurement than the
#: stream-filtered arms, so it does not overwrite their ledger/doc.
DOC_PATH_REAL = _REPO / "docs" / "results_overnight_real.md"

#: The default ledger — ``benchkit.checkpoint.DEFAULT_DIR``'s lane
#: file, spelled out so the constant needs no ``bench.*`` import.
CKPT_PATH = Path("bench/results/checkpoints/overnight.jsonl")

#: The zoo-real lane file — a non-``--resume`` run truncates its
#: ledger, so the corpus arms never share a default path.
CKPT_PATH_REAL = Path("bench/results/checkpoints/overnight_real.jsonl")

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
    the seed, for ``case["compose"]`` strata.  Emitted composites are
    held to the arm's own gate (the LHS must bind a real subterm —
    the ``require_bind`` contract, applied a second time since the
    symbolic path can mint a pair that never fired on the seed) and
    appended after the singles, each carrying its transported premise
    closure: the ``_compose_guards`` check/derive hooks and the
    ``parents`` / ``guard_pats`` record ``_admit`` serializes into the
    cell's ``law["guard"]``.  The composite slice is bounded by
    ``page`` like the single-step mint it sits beside — the pool can
    reach two pages under ``--compose``.
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
        if case.get("compose") and self.pool:
            self._mint_composites(ir.root, case)
        self.done = not self.pool

    def _mint_composites(self, root: Any, case: dict) -> None:
        """Pair-compose the bound pool over the module's own term.

        The seed-guided path fires each parent on ``root``'s real
        subterms, so an emitted composite is a chain that actually
        rewrote the module; the symbolic path (unguarded pairs) emits
        fully general derived rules and is then held to the same
        ``require_bind`` gate the singles met.  Composites keep the
        ``syn_*`` name, namespaced by ``@<module>`` — the
        ``synthesize_rules`` counter restarts per module, so a
        same-named pair minted on a different seed can be a different
        rule, and the ledger keys candidates by name.
        ``max_emitted`` bounds validation spend; the appended slice is
        further capped at ``page`` after the bind gate.
        """
        from mlopt_core import meta as meta_mod
        from mlopt_core.egraph.terms import _term_match
        from mlopt_core.ir import Op
        from mlopt_discovery.census import _iter_subterms
        from mlopt_discovery.proposal import Candidate

        page = case.get("page", 48)
        emitted = meta_mod.synthesize_rules(
            [c.as_rule() for c in self.pool],
            [root],
            fuel=_COMPOSE_FUEL,
            numeric_check=True,
            rounds=int(case["compose"]),
            max_emitted=page,
        )
        subs = [s for s in _iter_subterms(root) if isinstance(s, Op)]
        kept = 0
        for rule in emitted:
            if kept >= page:
                break
            if not any(
                _term_match(rule.lhs, s) is not None for s in subs
            ):
                continue
            parents = meta_mod.provenance(rule)
            name = f"{rule.name}@{case['zoo']}"
            object.__setattr__(rule, "name", name)
            meta_mod.SYNTH_PARENTS[name] = parents
            self.pool.append(
                Candidate(
                    strategy="corpus",
                    label=name,
                    lhs=rule.lhs,
                    rhs=rule.rhs,
                    note=f"composite: {' then '.join(parents)}",
                    rule=rule,
                )
            )
            self.meta[len(self.pool) - 1] = {
                "family": _COMPOSE_FAMILY,
                "sources": list(parents),
            }
            kept += 1


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


def _gen_cases(
    corpus: str, seed: int, episodes: int, **kw: Any
) -> list:
    """The corpus arm's case list — stream pages or zoo workloads."""
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
    count — ``meta.synthesize_rules`` rounds over the bound pool,
    seeded on each module's term (0 = off; ignored on other corpora).
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
        )
    zenv: dict | None = None
    if corpus in ("zoo", "zoo-real"):
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
            if corpus in ("zoo", "zoo-real") and zctxs is None:
                zctxs = (
                    _real_case_ctxs(zenv or {})
                    if corpus == "zoo-real"
                    else _zoo_case_ctxs(zenv or {})
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
    return {
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
        "admitted": sum(1 for r in cands if r["admitted"]),
        "skipped": sum(1 for r in cands if r["skipped"]),
        "leg_refusals": dict(leg_deaths),
        "novelty": dict(novelty),
        "finds": finds,
        "candidates": [
            {
                "name": r["name"],
                "family": r.get("family", ""),
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
            }.get(
                summary.get("corpus", "grammar"),
                " — the enumerated grammar stream",
            )
        ),
        f"- episodes: {summary['episodes']} · "
        f"evaluated: {summary['evaluated']} · "
        f"minted: {summary['minted']}"
        + (
            f" ({summary['composed']} composites)"
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
        "and judged against the corpus's distinct real subterms",
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
        "module's own term, for ROUNDS strata (default 2).  Emitted "
        "composites face the same bind gate and the same gauntlet, "
        "and carry their premise-closure guard spec into the ledger "
        "(mint: corpus-bind-compose — ledgers restart loudly)",
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
    if args.dry:
        for k, v in DRY.items():
            setattr(args, k, v)
    if args.ckpt is None:
        args.ckpt = (
            CKPT_PATH_REAL if args.corpus == "zoo-real" else CKPT_PATH
        )
    if args.out is None:
        args.out = (
            DOC_PATH_REAL if args.corpus == "zoo-real" else DOC_PATH
        )

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
        # without the composition arm never replays under it.
        "mint": (
            "corpus-bind-compose"
            if args.corpus == "zoo-real" and args.compose
            else "corpus-bind"
            if args.corpus == "zoo-real"
            else "stream"
        ),
        "compose": args.compose if args.corpus == "zoo-real" else 0,
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

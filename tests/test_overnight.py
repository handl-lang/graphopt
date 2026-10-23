"""The overnight discovery driver's contract.

``bench/drivers/overnight.py`` is the loop the resumable infra was
built for: open-game episodes mint candidates, the referee gauntlet
admits them, the prior-art bank classifies the survivors, and every
outcome lands in a protocol-gated JSONL ledger.

These tests pin the operational contracts — torch-eager, seconds:

* **ledger schema** — a manifest line carrying the protocol and the
  provenance triple (``git_sha`` / ``git_dirty`` / ``device``), then
  ``{"key", "record", "ts"}`` cells: one ``("episode", i)`` per page,
  one ``("candidate", name)`` per minted rule (with the serialized
  law so resume never re-derives it), one ``("classify", name)`` per
  admitted law;
* **resume idempotence** — a ``--resume`` run over a complete store
  replays cells verbatim: the episode engine and the referee are
  never re-entered (spy-pinned, ``test_resilience.py`` style);
* **``--dry``** — two episodes, a tiny pool, the full referee: the
  run completes in seconds and emits both artifacts;
* **honest skip** — a player that never evaluates mints nothing: the
  ledger records zero candidates, the documents still emit, nothing
  crashes.
"""

from __future__ import annotations

import json
import time

import pytest

import bench.referees
from bench.benchkit import checkpoint as ck_mod
from bench.benchkit.env import collect_env
from bench.benchkit.jsonl import read_jsonl
from bench.drivers import overnight as on


def _argv(tmp_path, *extra):
    """A complete, seconds-scale argv pointed at *tmp_path*."""
    return [
        "--episodes",
        "2",
        "--page",
        "6",
        "--budget",
        "4",
        "--ckpt",
        str(tmp_path / "ck.jsonl"),
        "--out",
        str(tmp_path / "out.md"),
        *extra,
    ]


def _records(path):
    recs = read_jsonl(path)
    man, cells = recs[0], recs[1:]
    return man, cells


def _cells(cells, kind):
    return [c for c in cells if c["key"][0] == kind]


class TestDryRun:
    """The end-to-end smoke: a real (tiny) loop, seconds."""

    def test_dry_completes_and_emits(self, tmp_path):
        t0 = time.time()
        assert on.main(_argv(tmp_path, "--dry")) == 0
        assert (
            time.time() - t0 < 120
        )  # measured ~10 s; the box is generous
        md = tmp_path / "out.md"
        js = tmp_path / "out.json"
        assert md.exists() and js.exists()
        payload = json.loads(js.read_text())
        # the provenance contract
        assert payload["generated_utc"]
        assert "seed" in payload
        for f in ("git_sha", "git_dirty", "device", "timestamp_utc"):
            assert f in payload["env"]
        assert payload["episodes"] == 2
        assert payload["evaluated"] >= 0
        text = md.read_text()
        assert "Admission funnel" in text and "Novelty" in text

    def test_ledger_schema(self, tmp_path):
        on.main(_argv(tmp_path, "--dry"))
        man, cells = _records(tmp_path / "ck.jsonl")
        assert man["_manifest"] is True
        assert man["lane"] == "overnight"
        assert man["protocol"]["episodes"] == 2
        assert {"git_sha", "git_dirty", "device"} <= set(
            man["provenance"]
        )
        for c in cells:
            assert {"key", "record", "ts"} <= set(c)
        episodes = _cells(cells, "episode")
        assert len(episodes) == 2
        for e in episodes:
            rec = e["record"]
            assert {"evals", "reward", "steps"} <= set(rec)
            for ev in rec["evals"]:
                assert {"i", "name", "label", "verdict"} <= set(ev)
        for c in _cells(cells, "candidate"):
            rec = c["record"]
            assert rec["name"] == c["key"][1]
            assert rec["law"]["kind"] == "law"  # serialized, resumable
            assert isinstance(rec["admitted"], bool)
            assert isinstance(rec["stages"], list)
            for s in rec["stages"]:
                assert {"name", "passed", "detail"} <= set(s)
            assert rec["arena"]["verdict"] in (
                "useful",
                "true",
                "known",
                "false",
            )
        admitted = [
            c["key"][1]
            for c in _cells(cells, "candidate")
            if c["record"]["admitted"]
        ]
        classified = {c["key"][1] for c in _cells(cells, "classify")}
        assert classified == set(admitted)
        for c in _cells(cells, "classify"):
            assert c["record"]["verdict"] in (
                "machine-original",
                "known-art",
                "vacuous",
            )


class TestCorpusZoo:
    """``--corpus zoo`` — real modules as the episode boards."""

    def test_zoo_arena_pool_is_real_matched(self):
        """Every pooled candidate's pattern LHS binds the module."""
        from mlopt_core.egraph.terms import _term_match
        from mlopt_core.meta import _positions

        case = {
            "zoo": "LoRAAdapter",
            "wseed": 0,
            "page": 6,
            "scan": 384,
        }
        arena = on._ZooArena(case)
        assert not arena.error
        assert arena.pool, "no stream candidate matched the module"
        ir, _t, _f = on._zoo_export(case)
        subs = [s for _p, s in _positions(ir.root)]
        for cand in arena.pool:
            pat = cand.as_rule().lhs
            assert any(_term_match(pat, s) is not None for s in subs), (
                f"{cand.label}: not a real match"
            )

    def test_zoo_dry_completes_and_records_corpus(self, tmp_path):
        """The zoo arm end-to-end: episodes are real modules."""
        t0 = time.time()
        assert on.main(_argv(tmp_path, "--dry", "--corpus", "zoo")) == 0
        assert time.time() - t0 < 180  # exports + referee, seconds
        man, cells = _records(tmp_path / "ck.jsonl")
        # the corpus arm is protocol-pinned — resume can't mix arms
        assert man["protocol"]["corpus"] == "zoo"
        episodes = _cells(cells, "episode")
        assert len(episodes) == 2
        for e in episodes:
            assert "zoo" in e["record"]["case"]
        payload = json.loads((tmp_path / "out.json").read_text())
        assert payload["corpus"] == "zoo"
        assert "corpus: `zoo`" in (tmp_path / "out.md").read_text()

    def test_grammar_is_the_default_corpus(self, tmp_path):
        """Back-compat: a bare ``--dry`` runs the grammar arm."""
        assert on.main(_argv(tmp_path, "--dry")) == 0
        man, _cells = _records(tmp_path / "ck.jsonl")
        assert man["protocol"]["corpus"] == "grammar"


class TestCorpusZooReal:
    """``--corpus zoo-real`` — candidates minted from the module's own term."""

    def test_real_arena_pool_is_corpus_minted(self):
        """The pool is ``pipeline.propose``'s mint, not the stream."""
        case = {"zoo": "AdditiveAttention", "wseed": 0, "page": 48}
        arena = on._RealArena(case)
        assert not arena.error
        assert arena.pool, "no proposal minted from the module"
        for i, cand in enumerate(arena.pool):
            # the pool rides the Candidate contract (as_rule/label)
            # so step()/evaluate/the mint loop see the same shapes;
            # meta carries the proposer's family for the ledger.
            assert cand.strategy == "corpus"
            assert cand.as_rule().lhs is not None
            assert arena.meta[i]["family"]
        # the census proposers saw this module's op-tuples —
        # AdditiveAttention spells mul(unsq, unsq) with *different*
        # dims, so the shared-attr census:mul_unsqueeze mint cannot
        # bind it; the attr-relaxed same-view family can and does.
        labels = [c.label for c in arena.pool]
        assert "mixed:mul_unsqueeze_unsqueeze_id" in labels
        assert "census:mul_unsqueeze" not in labels

    def test_real_arena_pool_is_binding_first(self):
        """Every pooled candidate's LHS binds a module subterm.

        ``require_bind`` is the arm's premise: a claim that matches
        no real subterm is never minted — the ``0 envs`` hole the
        ungated mint measured is closed at proposal time, not at the
        gauntlet (which still refuses bound-but-false claims).
        """
        from mlopt_core.egraph.terms import _term_match
        from mlopt_core.ir import Op
        from mlopt_discovery.census import _iter_subterms

        case = {"zoo": "AdditiveAttention", "wseed": 0, "page": 48}
        arena = on._RealArena(case)
        assert not arena.error
        assert arena.pool
        ir, _t, _f = on._zoo_export(case)
        subs = [s for s in _iter_subterms(ir.root) if isinstance(s, Op)]
        for cand in arena.pool:
            pat = cand.as_rule().lhs
            assert any(_term_match(pat, s) is not None for s in subs), (
                f"{cand.label}: binds no real subterm"
            )

    def test_real_ctx_binds_subterm_matches(self):
        """``real_terms`` is the subterm domain — the arm's premise.

        A corpus mint whose LHS spells a real op-tuple binds real
        sites through the referee context — the ``0 envs`` hole the
        grammar/zoo arms measured cannot occur for a binding mint.
        """
        from mlopt_core.egraph.terms import _term_match

        import bench.referees as referees

        env = on._zoo_export_env(
            [{"zoo": "TalkingHeadsAttention", "wseed": 0, "page": 6}]
        )
        ctx = on._real_referee_ctx(env, referees)
        # subterm granularity: strictly more terms than module roots
        assert len(ctx.real_terms) > len(env)
        assert ctx.probe and ctx.sink is not None
        arena = on._RealArena(
            {"zoo": "TalkingHeadsAttention", "wseed": 0, "page": 48}
        )
        bound = [
            c.label
            for c in arena.pool
            if any(
                _term_match(c.as_rule().lhs, m) is not None
                for m in ctx.real_terms
            )
        ]
        assert bound, "no minted candidate binds a real site"

    def test_zoo_real_dry_completes_and_records_corpus(self, tmp_path):
        """The zoo-real arm end-to-end: pool minted from real terms."""
        t0 = time.time()
        assert (
            on.main(_argv(tmp_path, "--dry", "--corpus", "zoo-real"))
            == 0
        )
        assert time.time() - t0 < 240  # exports + mint + referee
        man, cells = _records(tmp_path / "ck.jsonl")
        assert man["protocol"]["corpus"] == "zoo-real"
        # the binding-first mint is protocol-pinned — a resume over a
        # pre-gate ledger restarts loudly rather than mixing pools
        assert man["protocol"]["mint"] == "corpus-bind"
        episodes = _cells(cells, "episode")
        assert len(episodes) == 2
        for e in episodes:
            assert "zoo" in e["record"]["case"]
        for c in _cells(cells, "candidate"):
            assert c["record"]["family"]  # the proposer is recorded
        payload = json.loads((tmp_path / "out.json").read_text())
        assert payload["corpus"] == "zoo-real"
        assert "corpus: `zoo-real`" in (tmp_path / "out.md").read_text()


class TestCorpusZooRealCompose:
    """``--corpus zoo-real --compose`` — the composition arm."""

    def _case(self, **kw):
        return {
            "zoo": "AdditiveAttention",
            "wseed": 0,
            "page": 48,
            "compose": 2,
            **kw,
        }

    def _composites(self, arena):
        return [
            (i, c)
            for i, c in enumerate(arena.pool)
            if arena.meta.get(i, {}).get("family") == on._COMPOSE_FAMILY
        ]

    def test_compose_mints_bound_composites(self):
        """The syn_* fixpoint extends the pool, seed-gated like the rest.

        ChunkedRetention is the positive control: its bound singles
        pair-compose into composites whose minted RHS denotes under
        the seed binding, so they survive the emit gate.
        """
        from mlopt_core.egraph.terms import _term_match
        from mlopt_core.ir import Op
        from mlopt_core.meta import provenance
        from mlopt_discovery.census import _iter_subterms

        case = self._case(zoo="ChunkedRetention")
        arena = on._RealArena(case)
        assert not arena.error
        comps = self._composites(arena)
        singles = len(arena.pool) - len(comps)
        assert singles, "single-step pool unexpectedly empty"
        assert comps, "no composite minted from a bindable pool"
        # singles keep their indices — composites append behind them
        assert min(i for i, _c in comps) == singles
        ir, _t, _f = on._zoo_export(case)
        subs = [s for s in _iter_subterms(ir.root) if isinstance(s, Op)]
        for i, cand in comps:
            rule = cand.as_rule()
            # the premise-closure record rides the mint — the
            # conv_bn_fold shape: two named parents, guard pats.
            parents = provenance(rule)
            assert len(parents) == 2
            assert arena.meta[i]["family"] == on._COMPOSE_FAMILY
            assert arena.meta[i]["sources"] == list(parents)
            assert getattr(rule, "guard_pats", None) is not None
            # the same require_bind gate the singles met
            assert any(
                _term_match(rule.lhs, s) is not None for s in subs
            ), f"{rule.name}: composite binds no real subterm"
            # the syn_* counter restarts per module — the name is
            # namespaced so a same-named pair elsewhere never collides
            assert rule.name.endswith("@ChunkedRetention")

    def test_compose_drops_ill_typed_composites(self):
        """A composite whose RHS errors on the bound env never pools.

        AdditiveAttention is the measured negative case: its
        ``mixed:*`` parents compose into rules like
        ``mul(unsq(U), unsq(V)) -> mul(U, V)`` whose RHS cannot denote
        under the real binding (the oracle's ``rhs_err`` signal —
        ~100% of composites pre-gate).  The emit gate drops them all:
        no composite enters the pool, while the singles mint
        unaffected.
        """
        from mlopt_core.egraph.terms import _term_match
        from mlopt_core.ir import Op
        from mlopt_discovery.census import _iter_subterms
        from mlopt_discovery.evidence import _site_outcome

        case = self._case()  # AdditiveAttention — the rerr module
        arena = on._RealArena(case)
        assert not arena.error
        assert arena.pool, "single-step pool unexpectedly empty"
        # every emitted composite failed the seed-eval gate — nothing
        # ill-typed rides into the pool
        assert self._composites(arena) == []
        # …and it was the rhs_err leg, not the bind gate, that refused
        # them: synthesizing the same pool emits bound composites
        # whose RHS provably errors on the seed env (a substrate that
        # already refuses them at emit time simply leaves nothing to
        # check — the pool contract holds either way).
        from mlopt_core import meta as meta_mod

        ir, _t, _f = on._zoo_export(case)
        subs = [s for s in _iter_subterms(ir.root) if isinstance(s, Op)]
        emitted = meta_mod.synthesize_rules(
            [c.as_rule() for c in arena.pool],
            [ir.root],
            fuel=on._COMPOSE_FUEL,
            numeric_check=True,
            rounds=int(case["compose"]),
            max_emitted=case["page"],
        )
        for r in emitted:
            sites = [
                (m, s)
                for s in subs
                if (m := _term_match(r.lhs, s)) is not None
            ]
            if not sites or any(r is c.as_rule() for c in arena.pool):
                continue
            # dropped by the gate → the RHS must error on some bound
            # site, i.e. the drop is the measured rhs_err refusal
            # arriving at emit time instead of at the oracle
            assert any(
                _site_outcome(r, m, s) == "rhs-err" for m, s in sites
            ), f"{r.name}: dropped but no rhs-err site found"

    def test_compose_viable_composites_denote_on_seed(self):
        """Every pooled composite's RHS denotes on all bound sites.

        The gate's contract is the oracle leg's own measurement: no
        bound site may come back ``rhs-err``.
        """
        from mlopt_core.egraph.terms import _term_match
        from mlopt_core.ir import Op
        from mlopt_discovery.census import _iter_subterms
        from mlopt_discovery.evidence import _site_outcome

        case = self._case(zoo="ChunkedRetention")
        arena = on._RealArena(case)
        comps = self._composites(arena)
        assert comps, "positive control minted nothing"
        ir, _t, _f = on._zoo_export(case)
        subs = [s for s in _iter_subterms(ir.root) if isinstance(s, Op)]
        for _i, cand in comps:
            rule = cand.as_rule()
            sites = [
                (m, s)
                for s in subs
                if (m := _term_match(rule.lhs, s)) is not None
            ]
            assert sites, f"{rule.name}: no bound site"
            assert all(
                _site_outcome(rule, m, s) != "rhs-err" for m, s in sites
            ), f"{rule.name}: RHS errors on a bound env"

    def test_compose_admit_serializes_guard_spec(self):
        """A composite's cell carries the premise-closure spec."""
        arena = on._RealArena(self._case(zoo="ChunkedRetention"))
        _i, comp = self._composites(arena)[0]
        rec = on._admit(
            comp.as_rule(), "naive", bench.referees.RefereeCtx()
        )
        guard = rec["law"].get("guard")
        assert guard is not None, "no guard spec on a syn_* composite"
        assert len(guard["parents"]) == 2
        assert guard["pat1"] and guard["pat2"]
        # the honesty boundary — procedural hooks are flagged, the
        # spec says which parents they transport from
        assert "missing_hooks" in rec["law"]
        assert rec["law"]["kind"] == "law"

    def test_guarded_parents_seed_the_bank(self):
        """Shipped hook-carrying laws that fire join the mint.

        RMSNormPerHead spells an RMSNorm whose gain-free shape
        ``rms_norm_fold_nogain`` fires on — the guarded single the
        ``_compose_guards`` transport composes from.  AdditiveAttention
        is the negative control: no shipped guarded law fires, so the
        bank is the bound singles alone (the all-unguarded mint the
        arm measured before this change).
        """
        from mlopt_core.meta import _positions, apply_rewrite_at

        ir, _t, _f = on._zoo_export(self._case(zoo="RMSNormPerHead"))
        parents = on._guarded_parents(ir.root)
        names = {r.name for r in parents}
        assert "rms_norm_fold_nogain" in names
        for r in parents:
            # every parent is a guarded single that fires on the term
            assert r.check is not None or r.derive is not None
            assert any(
                apply_rewrite_at(r, ir.root, p) is not None
                for p, _ in _positions(ir.root)
            )
        ir2, _t2, _f2 = on._zoo_export(self._case())
        assert on._guarded_parents(ir2.root) == []

    def test_compose_mints_guarded_composites(self):
        """A guarded-parent pair emits a composite with live hooks.

        ``rms_norm_fold_nogain`` x the bound ``mixed:`` single
        transports the fold's premise through ``_compose_guards``:
        the composite carries a ``check`` (and the derive that fills
        its placeholders), names both parents, and serializes with
        ``missing_hooks`` populated — the class the arm exists to
        mint, where an all-unguarded pool could only yield bare
        patterns.
        """
        from mlopt_core.laws.serialize import law_to_data
        from mlopt_core.meta import provenance

        arena = on._RealArena(self._case(zoo="RMSNormPerHead"))
        assert not arena.error
        guarded = [
            (i, c)
            for i, c in self._composites(arena)
            if c.as_rule().check is not None
            or c.as_rule().derive is not None
        ]
        assert guarded, "no guarded composite minted"
        assert any(
            "rms_norm_fold_nogain" in provenance(c.as_rule())
            for _i, c in guarded
        )
        for i, cand in guarded:
            rule = cand.as_rule()
            assert arena.meta[i]["guarded"] is True
            assert getattr(rule, "guard_pats", None) is not None
            data = law_to_data(rule)
            assert data["missing_hooks"], (
                f"{rule.name}: transported hooks not flagged"
            )
            assert "check" in data["missing_hooks"]

    def test_compose_guarded_check_runs_in_gauntlet(self):
        """The transported check vetoes bindings — the gauntlet sees it.

        An unguarded composite's guarded-region row always reports
        ``0 dec`` (nothing declines what no guard evaluates).  The
        guarded composite's oracle-leg detail must show declines —
        the parent's premise re-expressed on the derived binding,
        vetoing the synthesized sites it was minted to exclude.
        """
        import re

        from mlopt_core.ir import Op
        from mlopt_discovery.census import _iter_subterms

        case = self._case(zoo="RMSNormPerHead")
        arena = on._RealArena(case)
        _i, cand = next(
            (i, c)
            for i, c in self._composites(arena)
            if c.as_rule().check is not None
        )
        ir, _t, _f = on._zoo_export(case)
        subs = [s for s in _iter_subterms(ir.root) if isinstance(s, Op)]
        ctx = bench.referees.RefereeCtx(real_terms=tuple(subs))
        verdict = bench.referees.judge("oracle", cand.as_rule(), ctx)
        detail = verdict.stages[0].detail
        declined = sum(int(n) for n in re.findall(r"(\d+) dec", detail))
        assert declined > 0, f"transported check never vetoed: {detail}"

    def test_compose_runs_on_empty_single_pool(self):
        """The guarded bank mints even when no single binds.

        LoRAAdapter binds no ``pipeline.propose`` single, but
        ``linear_row_scale_rev`` fires on its term — the guarded bank
        alone pairs it with itself and emits a guarded composite, so
        the episode still has a candidate to judge.
        """
        arena = on._RealArena(self._case(zoo="LoRAAdapter"))
        assert not arena.error
        comps = self._composites(arena)
        assert len(arena.pool) == len(comps) == 1
        rule = comps[0][1].as_rule()
        assert rule.check is not None or rule.derive is not None
        assert arena.meta[0]["guarded"] is True

    def test_compose_strata_chain_viable_parents(self):
        """A stratum-k composite becomes a stratum-(k+1) parent.

        InstanceNormBroadcast is the positive control: its stratum-1
        mints survive the bind + seed-eval gate and re-seat as
        parents, so stratum 2 mints ``syn_syn_*`` pairs whose
        provenance names a pooled stratum-1 composite (the
        ``@<module>`` label) and stratum 3 chains off stratum 2 the
        same way.  Before the per-stratum feedback the arm minted
        pair-depth only — ``--compose 4`` measured identically to
        ``--compose 2``.
        """
        from mlopt_core.meta import provenance

        arena = on._RealArena(
            self._case(zoo="InstanceNormBroadcast", compose=4)
        )
        assert not arena.error
        comps = [
            (i, c)
            for i, c in enumerate(arena.pool)
            if arena.meta.get(i, {}).get("family") == on._COMPOSE_FAMILY
        ]
        by_stratum: dict[int, list] = {}
        for i, c in comps:
            by_stratum.setdefault(arena.meta[i]["stratum"], []).append(
                (i, c)
            )
        assert 2 in by_stratum and 3 in by_stratum, by_stratum.keys()
        comp_labels = {c.label for _i, c in comps}
        chained = [
            (depth, cand)
            for depth, cs in by_stratum.items()
            if depth >= 2
            for _i, cand in cs
            if set(provenance(cand.as_rule())) & comp_labels
        ]
        assert chained, "no stratum->stratum parent link"
        for depth, cand in chained:
            parents = set(provenance(cand.as_rule()))
            prior = parents & comp_labels
            assert prior, f"{cand.label}: no pooled-composite parent"
            # the named parent is itself a pooled composite minted
            # exactly one stratum up — the feedback contract
            for pname in prior:
                pi = next(
                    i
                    for i, c in enumerate(arena.pool)
                    if c.label == pname
                )
                assert arena.meta[pi]["stratum"] == depth - 1

    def test_compose_ends_at_fixpoint_not_round_cap(self, monkeypatch):
        """The stratum loop stops on an empty-viable round, not ROUNDS.

        InstanceNormBroadcast's strata saturate below 8: a
        ``compose=8`` mint must not keep re-scanning a dead bank —
        the synthesizer is consulted only until the first round that
        emits no new viable parent, and the pool is identical to
        ``compose=4`` (which itself ends at the same fixpoint).
        """
        from mlopt_core import meta as meta_mod

        calls = []
        real = meta_mod.synthesize_rules

        def spy(rules, seed_terms, **kw):
            calls.append(list(rules))
            return real(rules, seed_terms, **kw)

        monkeypatch.setattr(meta_mod, "synthesize_rules", spy)
        a8 = on._RealArena(
            self._case(zoo="InstanceNormBroadcast", compose=8)
        )
        assert not a8.error
        assert 1 < len(calls) < 8, f"ran {len(calls)} strata"
        # round 2's bank leads with round 1's viable survivors —
        # stratum-first ordering so the fuel reaches them
        first = calls[1][0].name
        assert first.endswith("@InstanceNormBroadcast")
        assert any(c.label == first for c in a8.pool)
        a4 = on._RealArena(
            self._case(zoo="InstanceNormBroadcast", compose=4)
        )
        assert [c.label for c in a8.pool] == [c.label for c in a4.pool]

    def test_zoo_real_compose_dry_records_mint_tag(self, tmp_path):
        """The arm is protocol-pinned — ledgers restart loudly."""
        t0 = time.time()
        argv = _argv(
            tmp_path, "--dry", "--corpus", "zoo-real", "--compose"
        )
        assert on.main(argv) == 0
        assert time.time() - t0 < 300  # exports + synth + referee
        man, cells = _records(tmp_path / "ck.jsonl")
        assert man["protocol"]["corpus"] == "zoo-real"
        assert man["protocol"]["mint"] == "corpus-bind-compose-strata"
        assert man["protocol"]["compose"] == 2
        payload = json.loads((tmp_path / "out.json").read_text())
        assert payload["corpus"] == "zoo-real"
        assert "composed" in payload
        # a composite candidate cell, if any evaluated, carries the
        # guard spec the admit path serializes
        for c in _cells(cells, "candidate"):
            if c["record"].get("family") == on._COMPOSE_FAMILY:
                assert c["record"]["law"]["guard"]["parents"]

    def test_compose_off_zoo_real_refused(self, tmp_path):
        """--compose on a non-zoo-real corpus is a loud refusal."""
        with pytest.raises(SystemExit):
            on.main(
                _argv(tmp_path, "--dry", "--corpus", "zoo", "--compose")
            )

    def test_compose_ledger_restarts_on_mint_change(self, tmp_path):
        """A --compose resume over a bind-only ledger cannot merge."""
        argv = _argv(tmp_path, "--dry", "--corpus", "zoo-real")
        assert on.main(argv) == 0
        man1, _cells1 = _records(tmp_path / "ck.jsonl")
        assert man1["protocol"]["mint"] == "corpus-bind"
        assert on.main([*argv, "--compose", "--resume"]) == 0
        man2, _cells2 = _records(tmp_path / "ck.jsonl")
        assert man2["protocol"]["mint"] == "corpus-bind-compose-strata"


class TestCorpusXModule:
    """``--corpus zoo-xmodule`` — one pool anti-unified over the corpus.

    The mint source is structure shared across DISTINCT modules:
    ``op_tuple_census`` counts the modules an op-tuple spans,
    ``antiunify.mine_candidates`` mines the deep shared subgraphs of
    the merged corpus e-graph, and the K-bind gate keeps only claims
    whose LHS binds >= K different modules' terms.
    """

    def _roots(self):
        """A synthetic corpus: m1/m2 share mul(unsq,unsq); m3 unique."""
        from mlopt_core.ir import Op, TensorType, Var

        def v(n):
            return Var(n, TensorType((2, 4)))

        m1 = Op.make(
            "mul",
            Op.make("unsqueeze", v("x"), dim=0),
            Op.make("unsqueeze", v("y"), dim=0),
        )
        m2 = Op.make(
            "mul",
            Op.make("unsqueeze", v("p"), dim=2),
            Op.make("unsqueeze", v("q"), dim=2),
        )
        # the manual-softmax spelling — a fold-class shape no other
        # synthetic module contains (the module-unique control).
        e = Op.make("exp", v("z"))
        m3 = Op.make("div", e, Op.make("sum", e, dim=-1, keepdim=True))
        return {"m1": m1, "m2": m2, "m3": m3}

    def test_shared_shape_mints_one_generalized_law(self):
        """The two modules' shared shape folds into ONE minted claim.

        ``mul(unsq(u), unsq(v))`` spelled by m1 and m2 is one
        recurring cross-module pattern: the generalized naturality
        law pools once (alpha-deduped across both proposer sources),
        carrying its measured cross-module bind count.
        """
        cands, meta, stats = on._xmodule_pool(self._roots(), 2)
        labels = [c.label for c in cands]
        # the generalized law over the shared shape — minted ONCE,
        # not once per module (the arm's whole premise).
        assert labels.count("census:mul_unsqueeze") == 1
        i = labels.index("census:mul_unsqueeze")
        assert meta[i]["mods"] == 2  # binds m1 and m2 — the new surface
        assert meta[i]["family"]
        # every pooled claim crosses the K-bind bar — module-unique
        # bindings are the old arm's surface, not this one's.
        assert all(m["mods"] >= 2 for m in meta.values())
        assert stats["pool"] == len(cands)

    def test_module_unique_shape_does_not_mint(self):
        """m3's softmax fold — bound by one module — never pools.

        The ``fold:softmax``/``recognize:softmax`` mints would fire on
        m3's manual-softmax spelling, but a claim binding exactly one
        module is the per-module arm's surface: the K-bind gate drops
        it, and m3's unique op-tuples never reach the proposers
        either (the census is filtered to >= K-module patterns).
        """
        cands, meta, _stats = on._xmodule_pool(self._roots(), 2)
        labels = [c.label for c in cands]
        assert "fold:softmax" not in labels
        assert "recognize:softmax" not in labels
        # every minted claim binds at least two of {m1, m2, m3}
        assert all(m["mods"] >= 2 for m in meta.values())

    def test_k_threshold_gates_the_pool(self):
        """K=3 over a 2-of-3 shared shape pools nothing for it."""
        cands, meta, stats = on._xmodule_pool(self._roots(), 3)
        labels = [c.label for c in cands]
        assert "census:mul_unsqueeze" not in labels
        assert all(m["mods"] >= 3 for m in meta.values())
        assert stats["k"] == 3

    def test_xmodule_arena_pages_the_shared_pool(self):
        """Each episode slices ``page`` candidates off the one mint."""
        case0 = {"xpage": 0, "page": 4, "xbind": 2, "wseed0": 0}
        case1 = {"xpage": 1, "page": 4, "xbind": 2, "wseed0": 0}
        a0, a1 = on._XModuleArena(case0), on._XModuleArena(case1)
        assert not a0.error and not a1.error
        assert a0.pool_total == a1.pool_total
        assert len(a0.pool) <= 4
        # disjoint slices of the same minted pool
        names0 = {c.label for c in a0.pool}
        names1 = {c.label for c in a1.pool}
        assert not (names0 & names1)
        for i, _c in enumerate(a0.pool):
            assert a0.meta[i]["family"]

    def test_xmodule_dry_completes_and_records_corpus(self, tmp_path):
        """The xmodule arm end-to-end: corpus-level mint, K-bound."""
        t0 = time.time()
        assert (
            on.main(_argv(tmp_path, "--dry", "--corpus", "zoo-xmodule"))
            == 0
        )
        assert (
            time.time() - t0 < 300
        )  # whole-zoo export + mint + referee
        man, cells = _records(tmp_path / "ck.jsonl")
        assert man["protocol"]["corpus"] == "zoo-xmodule"
        assert man["protocol"]["mint"] == "corpus-xbind"
        assert man["protocol"]["xbind"] == 2
        episodes = _cells(cells, "episode")
        assert len(episodes) == 2
        for e in episodes:
            assert "xpage" in e["record"]["case"]
        for c in _cells(cells, "candidate"):
            assert c["record"]["family"]  # the proposer is recorded
            assert c["record"]["mods"] >= 2  # the K-bind surface
        payload = json.loads((tmp_path / "out.json").read_text())
        assert payload["corpus"] == "zoo-xmodule"
        assert payload["xmodule"]["k"] == 2
        assert (
            "corpus: `zoo-xmodule`" in (tmp_path / "out.md").read_text()
        )


class TestCorpusZooPrim:
    """``--corpus zoo-prim`` — primitives minted for shared manual shapes.

    The mint anti-unifies the corpus for recurring manual subgraphs
    that no bank op denotes, registers each as a NEW ``synop_*``
    primitive whose denotation IS the subgraph (spelled-eval binding
    in ``_IR_TO_TORCH``), and folds ``manual -> synop_*`` — a claim
    the prior-art sieve cannot derive by construction.
    """

    def _roots(self):
        """m1/m2 share ``sub(x, mul(x, sigmoid(y)))`` — undenoted."""
        from mlopt_core.ir import Op, TensorType, Var

        def v(n):
            return Var(n, TensorType((2, 4)))

        shape = lambda a, b: Op.make(  # noqa: E731
            "sub", a, Op.make("mul", a, Op.make("sigmoid", b))
        )
        return {
            "m1": shape(v("x"), v("y")),
            "m2": shape(v("p"), v("q")),
        }

    def test_prim_mints_fold_for_undenoted_shape(self):
        """The shared manual shape mints a synop_* fold binding >=K mods."""
        cands, meta, stats = on._prim_pool(self._roots(), 2)
        labels = [c.label for c in cands]
        assert labels and all(
            lbl.startswith("prim:synop_") for lbl in labels
        )
        i = labels.index(cands[0].label)
        assert meta[i]["mods"] >= 2
        rec = meta[i]["prim"]
        assert rec["synop"].startswith("synop_")
        assert rec["equal_sites"] >= 1
        assert stats["pool"] == len(cands)

    def test_prim_denotation_verifies_on_bound_sites(self):
        """eval_instance(manual, synop) is equal at every bound site."""
        from mlopt_core.egraph.terms import (
            _term_instantiate,
            _term_match,
        )
        from mlopt_core.ir import Op
        from mlopt_discovery.census import _iter_subterms
        from mlopt_discovery.oracle import eval_instance

        cands, _meta, _stats = on._prim_pool(self._roots(), 2)
        assert cands
        subs = [
            s
            for r in self._roots().values()
            for s in _iter_subterms(r)
            if isinstance(s, Op)
        ]
        for cand in cands:
            rule = cand.as_rule()
            sites = [m for s in subs if (m := _term_match(rule.lhs, s))]
            assert sites
            for m in sites:
                out, _n = eval_instance(
                    _term_instantiate(rule.lhs, m),
                    _term_instantiate(rule.rhs, m),
                )
                assert out == "equal", f"{rule.name}: {out}"

    def test_prim_fold_survives_full_gauntlet(self):
        """The minted fold is admitted under referee 'full'.

        The cert leg anchors on the declared ``synop_*.expand``
        premise — the definitional axiom the bare bank lacks by
        construction (novel-by-construction cuts both ways: the fold
        is underivable, so its admission must be anchored).
        """
        import bench.referees

        cands, _meta, _stats = on._prim_pool(self._roots(), 2)
        cand = cands[0]
        rule = cand.as_rule()
        assert "expand" in getattr(rule, "premises", ())[0].name
        subs = tuple(
            s
            for r in self._roots().values()
            for s in on._op_subterms({"x": r})["x"]
        )
        rec = on._admit(
            rule, "full", bench.referees.RefereeCtx(real_terms=subs)
        )
        assert rec["admitted"], rec["reason"]
        names = [s["name"] for s in rec["stages"]]
        assert names == ["oracle", "pay", "sweep", "cert"]

    def test_prim_fold_is_machine_original(self):
        """The sieve cannot derive a law about an op that did not exist."""
        from types import SimpleNamespace

        from bench.drivers.screen import novelty_sieve as ns

        roots = self._roots()
        cands, _meta, _stats = on._prim_pool(roots, 2)
        ctxs = [
            ns._CaseCtx(
                case=SimpleNamespace(name=n),
                model=None,
                feed=None,
                term=r,
                ops=ns._ops_of(r),
            )
            for n, r in roots.items()
        ]
        for cand in cands:
            res = on._classify(
                cand.as_rule(), ns.known_bank(), ctxs=ctxs
            )
            assert res["verdict"] == "machine-original", res
            assert res["via"] == "underivable"

    def test_prim_denoted_shape_does_not_mint(self):
        """A shape a bank op already denotes mints no synop_*.

        Two modules spelling ``mul(u, sigmoid(u))`` manually — the
        ``silu`` denotation in ``lawdata.HANDLERS`` — are excluded at
        the denoted gate: a second name for an existing fused op is
        not a new primitive.
        """
        from mlopt_core.ir import Op, TensorType, Var

        def v(n):
            return Var(n, TensorType((2, 4)))

        silu = lambda a: Op.make(  # noqa: E731
            "mul", a, Op.make("sigmoid", a)
        )
        _cands, _meta, stats = on._prim_pool(
            {"m1": silu(v("x")), "m2": silu(v("y"))}, 2
        )
        assert stats["denoted"] >= 1
        assert stats["pool"] == 0

    def test_prim_arena_pages_the_shared_pool(self):
        """Each episode slices ``page`` folds off the one mint."""
        case0 = {"xpage": 0, "page": 4, "xbind": 2, "wseed0": 0}
        case1 = {"xpage": 1, "page": 4, "xbind": 2, "wseed0": 0}
        a0, a1 = on._PrimArena(case0), on._PrimArena(case1)
        assert not a0.error and not a1.error
        assert a0.pool_total == a1.pool_total
        names0 = {c.label for c in a0.pool}
        names1 = {c.label for c in a1.pool}
        assert not (names0 & names1)

    def test_prim_dry_completes_and_records_corpus(self, tmp_path):
        """The zoo-prim arm end-to-end: mint, fold, gauntlet, sieve."""
        t0 = time.time()
        assert (
            on.main(_argv(tmp_path, "--dry", "--corpus", "zoo-prim"))
            == 0
        )
        assert (
            time.time() - t0 < 300
        )  # whole-zoo export + mint + referee
        man, cells = _records(tmp_path / "ck.jsonl")
        assert man["protocol"]["corpus"] == "zoo-prim"
        assert man["protocol"]["mint"] == "corpus-prim"
        assert man["protocol"]["xbind"] == 2
        episodes = _cells(cells, "episode")
        assert len(episodes) == 2
        for e in episodes:
            assert "xpage" in e["record"]["case"]
        for c in _cells(cells, "candidate"):
            rec = c["record"]
            assert rec["family"] == on._PRIM_FAMILY
            assert rec["mods"] >= 2
            # the minted op's own record: name + handler verdict
            assert rec["prim"]["synop"].startswith("synop_")
            assert rec["prim"]["handler"]["verdict"] in (
                "minted",
                "refused",
                "skipped-no-domain",
            )
            # the definitional premise is recorded for audit
            assert rec["law"]["premises"]
        payload = json.loads((tmp_path / "out.json").read_text())
        assert payload["corpus"] == "zoo-prim"
        assert payload["prim"]["k"] == 2
        assert "corpus: `zoo-prim`" in (tmp_path / "out.md").read_text()


class TestResume:
    """A protocol-matched resume replays cells, never re-measures."""

    def test_resume_reuses_store(self, tmp_path, monkeypatch):
        argv = _argv(tmp_path, "--dry")
        assert on.main(argv) == 0
        _man1, cells1 = _records(tmp_path / "ck.jsonl")
        summary1 = json.loads((tmp_path / "out.json").read_text())

        def boom(*_a, **_k):
            raise AssertionError("recompute on resume")

        monkeypatch.setattr(on.engine, "run_episode", boom)
        monkeypatch.setattr(bench.referees, "judge", boom)
        monkeypatch.setattr(on, "verify_law", boom)
        assert on.main([*argv, "--resume"]) == 0
        _man2, cells2 = _records(tmp_path / "ck.jsonl")
        summary2 = json.loads((tmp_path / "out.json").read_text())
        # verbatim replay — no new cells, identical summary
        assert len(cells2) == len(cells1)
        for k in (
            "episodes",
            "evaluated",
            "minted",
            "admitted",
            "novelty",
            "leg_refusals",
        ):
            assert summary2[k] == summary1[k]

    def test_protocol_mismatch_restarts(self, tmp_path):
        assert on.main(_argv(tmp_path, "--dry")) == 0
        # a different protocol --resume must not merge old cells
        argv = _argv(tmp_path, "--budget", "3", "--resume")
        assert on.main(argv) == 0
        man, _cells = _records(tmp_path / "ck.jsonl")
        assert man["protocol"]["budget"] == 3


class TestHonestSkip:
    """No minted candidates — recorded, not a crash."""

    def test_empty_episodes(self, tmp_path):
        man = ck_mod.new_manifest(
            "overnight", {"protocol": "test"}, device="cpu"
        )
        ckpt = ck_mod.Checkpoint.open(tmp_path / "ck.jsonl", man)
        never = lambda _i: lambda _state: None  # noqa: E731
        summary = on.run(
            episodes=2,
            page=4,
            budget=4,
            seed=0,
            player="playbook",
            referee="full",
            ckpt=ckpt,
            arm_factory=never,
            log=lambda *_a: None,
        )
        assert summary["episodes"] == 2
        assert summary["evaluated"] == 0
        assert summary["minted"] == 0
        assert summary["admitted"] == 0
        assert summary["novelty"] == {}
        # the cells exist — the empty run is a recorded fact
        episodes = _cells(
            read_jsonl(tmp_path / "ck.jsonl")[1:], "episode"
        )
        assert len(episodes) == 2
        assert all(e["record"]["evals"] == [] for e in episodes)
        # and the documents still emit
        out = on.emit(
            summary,
            tmp_path / "out.md",
            env=collect_env("cpu"),
            seed=0,
        )
        assert out.exists()
        assert (
            "None." in out.read_text() or "**None.**" in out.read_text()
        )

    def test_page_beyond_stream_records_empty(self, tmp_path):
        man = ck_mod.new_manifest(
            "overnight", {"protocol": "test"}, device="cpu"
        )
        ckpt = ck_mod.Checkpoint.open(tmp_path / "ck.jsonl", man)

        def boom(_state):
            raise AssertionError("player consulted on a dead board")

        summary = on.run(
            episodes=1,
            page=4,
            budget=4,
            seed=0,
            player="playbook",
            referee="full",
            ckpt=ckpt,
            arm_factory=lambda _i: boom,
            # past the ~1.1k-candidate stream — an empty pool is a
            # done state, so the player is never consulted.
            cases=[{"ops": None, "offset": 10**9, "page": 4}],
            log=lambda *_a: None,
        )
        assert summary["episodes"] == 1
        assert summary["evaluated"] == 0
        assert summary["minted"] == 0
        episodes = _cells(
            read_jsonl(tmp_path / "ck.jsonl")[1:], "episode"
        )
        assert episodes[0]["record"]["evals"] == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

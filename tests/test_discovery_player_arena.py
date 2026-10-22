"""The player arena — the player itself as the rewritable term.

Pins: the spec record round-trips through the parts registry
(``DomainSpec.to_record``/``build_from_record``); the board edits
the record within the objective's vocabulary (toggles respect the
``requires`` constraints, weight/lr edits land); ``evaluate`` is a
bounded oracle probe and ``extract`` pays the held-out delta vs the
parent; ``GreedyEditor`` ablates dead parts and emits a better
child; the domain is registered in ``play.DOMAINS``.

The nested train+eval is stubbed by the ``score_fn`` case field —
the board mechanics don't need the real oracle (the real scorers are
exercised by the headline run documented in
``docs/results_player_on_board.md``).
"""

from __future__ import annotations

import json

import pytest
from mlopt_discovery import meta_player as mp
from mlopt_discovery import open_game as og
from mlopt_discovery import play
from mlopt_discovery import player_arena as pa
from gamekit.engine import run_episode
from mlopt_discovery.meta_arena import Action
from gamekit.players import (
    ActionFeats,
    Bias,
    DomainSpec,
    LinearPolicy,
)


def _toy_case(**kw):
    """A player-board case whose oracle is a stub score function."""
    case = {"objective": "open", "seed": 0, "max_evals": 8}
    case.update(kw)
    return case


def _smaller_is_better(record, *, seed, cfg):
    """Toy oracle — fewer parts score higher, deterministically."""
    return {
        "score": 1.0 - 0.1 * len(record["parts"]),
        "greedy": 0.0,
        "evals_ep": 0.0,
    }


class TestSpecRecord:
    def test_meta_roundtrip(self):
        rec = mp.META_SPEC.to_record()
        assert rec["legal"] == "meta"
        assert rec["parts"] == list(mp.META_PART_TAGS)
        spec = DomainSpec.build_from_record(rec)
        assert spec.features == mp.META_SPEC.features
        assert spec.legal is mp.META_SPEC.legal
        assert spec.lr == mp.META_SPEC.lr and spec.rank is None
        # the rebuilt featurizer is the same object graph's behavior
        st = play._meta_board(mp.gen_cases(0, 1)[0]).observe()
        from mlopt_discovery import meta_arena as ma

        for a in ma.legal_actions(st):
            assert spec.featurize(st, a) == mp.META_SPEC.featurize(
                st, a
            )

    def test_open_roundtrip(self):
        rec = og.OPEN_SPEC.to_record()
        assert rec["legal"] == "open" and rec["rank"] == "open"
        spec = DomainSpec.build_from_record(rec)
        assert spec.legal is og.legal_actions
        assert spec.rank is og._cand_feats
        case = og.gen_cases(0, 1, page=4)[0]
        st = og.OpenArena(case).observe()
        for a in og.legal_actions(st):
            assert spec.featurize(st, a) == og.OPEN_SPEC.featurize(
                st, a
            )

    def test_search_roundtrip(self):
        rec = play.SEARCH_SPEC.to_record()
        assert rec["legal"] == "search"
        assert (
            DomainSpec.build_from_record(rec).legal
            is play._search_legal
        )

    def test_unregistered_part_raises(self):
        spec = DomainSpec.build(
            features=("bias",),
            legal=lambda s: (),
            parts=(Bias(), ActionFeats(lambda a, r: {})),
        )
        with pytest.raises(ValueError, match="unregistered"):
            spec.to_record()
        # and the legal enumerator is unregistered too
        spec2 = DomainSpec.build(
            features=("bias",), legal=lambda s: (), parts=(Bias(),)
        )
        with pytest.raises(ValueError, match="legal enumerator"):
            spec2.to_record()

    def test_record_json_serializable(self):
        import json

        rec = og.OPEN_SPEC.to_record()
        assert json.loads(json.dumps(rec))["parts"] == rec["parts"]


class TestBoard:
    def test_initial_state(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        st = board.observe()
        assert not st.done and st.steps == 0
        assert st.parts == tuple(og.OPEN_PART_TAGS)
        assert st.record["weights"] is None  # cold prior
        assert st.droppable == st.parts and st.addable == ()
        assert st.evals == 0 and st.parent_score is None

    def test_legal_enumerates_vocabulary(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        acts = pa.legal_actions(board.observe())
        ops = {a.op for a in acts}
        assert ops == set(pa._PLAYER_OPS)
        toggles = [a for a in acts if a.op == "toggle_feature"]
        assert {a.params["part"] for a in toggles} == set(
            og.OPEN_PART_TAGS
        )
        assert acts[-1].op == "extract"
        # evaluate is offered while the eval budget lasts
        assert any(a.op == "evaluate" for a in acts)

    def test_toggle_drop_and_readd(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        board.step(
            Action(
                "toggle_feature", {"part": "open:hist", "add": False}
            )
        )
        st = board.observe()
        assert "open:hist" not in st.parts
        assert "open:hist" in st.addable
        board.step(
            Action("toggle_feature", {"part": "open:hist", "add": True})
        )
        # re-added at canonical position, not appended
        assert board.observe().parts == tuple(og.OPEN_PART_TAGS)

    def test_meta_requires_blocks_state_drop(self):
        board = pa.PlayerArena(
            {
                "objective": "meta",
                "seed": 0,
                "score_fn": _smaller_is_better,
            }
        )
        st = board.observe()
        # meta:action reads st:* — meta:state is not droppable
        assert "meta:state" not in st.droppable
        _, rep = board.step(
            Action(
                "toggle_feature", {"part": "meta:state", "add": False}
            )
        )
        assert not rep.applied and rep.reward == 0.0
        # meta:macro also reads st:* (x:claimable:claim & co.) — both
        # readers must drop before meta:state unblocks
        board.step(
            Action(
                "toggle_feature", {"part": "meta:action", "add": False}
            )
        )
        assert "meta:state" not in board.observe().droppable
        board.step(
            Action(
                "toggle_feature", {"part": "meta:macro", "add": False}
            )
        )
        st = board.observe()
        assert "meta:state" in st.droppable

    def test_edit_weight_and_set_lr(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        board.step(
            Action("edit_weight", {"name": "a:score", "delta": 0.5})
        )
        st = board.observe()
        assert st.record["weights"] == {"a:score": 0.5}
        board.step(Action("set_lr", {"lr": 0.2}))
        board.step(Action("set_weights", {"table": "baked"}))
        st = board.observe()
        assert st.record["lr"] == 0.2
        assert st.record["weights"] == dict(lawdata_open_weights())

    def test_evaluate_probe_and_dedup(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        st, rep = board.step(Action("evaluate", {"k": 0}))
        assert rep.applied and rep.reward == (
            pa.PLAYER_ARENA_REWARD["step"]
            + pa.PLAYER_ARENA_REWARD["eval"]
        )
        assert st.evals == 1 and len(st.probes) == 1
        assert st.parent_score == pytest.approx(
            0.1
        )  # 9 parts -> 1 - 0.9
        assert st.probes[0]["delta"] == 0.0  # incumbent resubmitted
        # the same record cannot be probed twice
        st2, rep2 = board.step(Action("evaluate", {"k": 1}))
        assert not rep2.applied and st2.evals == 1
        # ... and legal_actions stops offering it
        assert not any(
            a.op == "evaluate" for a in pa.legal_actions(st2)
        )

    def test_eval_budget_bounds_probes(self):
        calls = []

        def score(record, *, seed, cfg):
            calls.append(tuple(record["parts"]))
            return {"score": 0.0, "greedy": 0.0, "evals_ep": 0.0}

        board = pa.PlayerArena(_toy_case(score_fn=score, max_evals=1))
        board.step(Action("evaluate", {"k": 0}))
        board.step(
            Action("toggle_feature", {"part": "bias", "add": False})
        )
        _, rep = board.step(Action("evaluate", {"k": 1}))
        assert not rep.applied and rep.note == "evals-exhausted"

    def test_extract_without_probe_measures_current(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        st, rep = board.step(Action("extract", {}))
        assert rep.terminal and st.done
        # bare extract resubmits the incumbent -> delta 0
        assert rep.detail["delta"] == 0.0
        assert rep.detail["child"]["parts"] == st.parent["parts"]

    def test_extract_pays_best_delta(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        board.step(
            Action("toggle_feature", {"part": "bias", "add": False})
        )
        board.step(Action("evaluate", {"k": 0}))
        _, rep = board.step(Action("extract", {}))
        assert rep.terminal
        assert rep.reward == pytest.approx(
            pa.PLAYER_ARENA_REWARD["step"]
            + pa.PLAYER_ARENA_REWARD["delta"] * 0.1
        )
        assert rep.detail["child"]["parts"] == [
            t for t in og.OPEN_PART_TAGS if t != "bias"
        ]

    def test_unknown_move_declines(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        _, rep = board.step(Action("flibberty", {}))
        assert not rep.applied and rep.reward == 0.0

    def test_step_after_done_raises(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        board.step(Action("extract", {}))
        with pytest.raises(RuntimeError, match="terminal"):
            board.step(Action("extract", {}))


class TestGreedyEditor:
    def test_ablates_dead_parts_and_extracts(self):
        board = pa.PlayerArena(
            _toy_case(score_fn=_smaller_is_better, max_evals=11)
        )
        traj = run_episode(board, pa.GreedyEditor(), 30)
        t = traj.terminal
        assert t is not None and t.detail["delta"] == pytest.approx(0.9)
        assert t.detail["child"]["parts"] == []
        # nine LOO probes, each kept (smaller is always better)
        assert len(t.detail["probes"]) == 9

    def test_reverts_a_losing_drop(self):
        def score(record, *, seed, cfg):
            # incumbent is optimal; any drop costs 0.2
            return {
                "score": (
                    0.5
                    if len(record["parts"]) == len(og.OPEN_PART_TAGS)
                    else 0.3
                ),
                "greedy": 0.0,
                "evals_ep": 0.0,
            }

        board = pa.PlayerArena(_toy_case(score_fn=score, max_evals=3))
        traj = run_episode(board, pa.GreedyEditor(), 30)
        t = traj.terminal
        assert t is not None
        # eval budget runs out after 3 probes; the child is the best
        # probed record — all probes lose, so the incumbent stands
        assert t.detail["delta"] <= 0.0
        # the board restored each losing drop before extracting? No:
        # it probes LOO and reverts — the last probed (unreverted when
        # budget hit) stays dropped; best is still a losing probe, so
        # the emitted child is the best of a bad lot (honest -0.2).
        assert t.detail["child_score"] == 0.3

    def test_episode_budget_ends_without_extract(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        traj = run_episode(board, pa.GreedyEditor(), 2)
        assert traj.terminal is None


class TestPlayerSpec:
    def test_featurizer_in_schema(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        st = board.observe()
        for a in pa.legal_actions(st):
            feats = pa.featurize(st, a)
            assert set(feats) <= set(pa.PLAYER_FEATURES)
            assert feats["bias"] == 1.0
            assert feats[f"op:{a.op}"] == 1.0

    def test_player_spec_reifies(self):
        rec = pa.PLAYER_SPEC.to_record()
        assert rec["legal"] == "player"
        spec = DomainSpec.build_from_record(rec)
        assert spec.legal is pa.legal_actions

    def test_learned_arm_plays(self):
        # LinearPolicy drives the meta-board like any domain
        p = LinearPolicy(
            0,
            legal=pa.legal_actions,
            featurizer=pa.featurize,
            features=pa.PLAYER_FEATURES,
            learn=False,
        )
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        traj = run_episode(board, p, 8)
        assert traj.reports


class TestDomainRegistration:
    def test_player_domain(self):
        d = play.DOMAINS["player"]
        assert d.spec is pa.PLAYER_SPEC
        assert d.scripted is pa.GreedyEditor
        case = play.DOMAINS["player"].cases(0, 1)[0]
        assert case["objective"] == "open"

    def test_gen_cases_shape(self):
        cases = pa.gen_cases(3, 2, objective="meta")
        assert [c["objective"] for c in cases] == ["meta", "meta"]
        assert cases[0]["seed"] == 3 and cases[1]["seed"] == 4


def lawdata_open_weights():
    """The shipped open prior — imported lazily for the test above."""
    from mlopt_discovery import lawdata

    return lawdata.OPEN_GAME_PLAYER_WEIGHTS


class TestNestedScorersReal:
    """The real nested scorers, at probe scale.

    The headline runs (``docs/results_player_on_board.md``) measure
    these at the leaderboard protocol; here they run at epochs=0 over
    single cases — same code path, test budget.  What is pinned: the
    scorer returns the ``{score, greedy, evals_ep}`` triple the board
    consumes, and the open scorer's memoized verdicts land in the
    shared memo.
    """

    def test_score_meta_smoke(self):
        out = pa._score_meta(
            pa._meta_parent(),
            seed=0,
            cfg={
                "epochs": 1,
                "train_cases": 1,
                "eval_cases": 1,
                "budget": 2,
            },
        )
        assert set(out) == {"score", "greedy", "evals_ep"}
        assert out["evals_ep"] > 0

    def test_score_open_smoke(self):
        memo: dict = {}
        out = pa._score_open(
            pa._open_parent(),
            seed=0,
            cfg={"epochs": 1, "pages": 2, "page": 4, "memo": memo},
        )
        assert set(out) == {"score", "greedy", "evals_ep"}
        # the patched, memoized lp.evaluate recorded the verdicts the
        # episode drew — parent and children share them verbatim
        assert memo


class TestOracleHelpers:
    def test_mean_empty_and_rows(self):
        assert pa._mean([], "x") == 0.0
        assert pa._mean([{"x": 2.0}, {"x": 4.0}], "x") == 3.0

    def test_cand_key_stable_identity(self):
        from types import SimpleNamespace

        cand = SimpleNamespace(label="a:1", lhs="L", rhs="R")
        assert pa._cand_key(cand) == pa._cand_key(cand)
        assert pa._cand_key(cand) == ("a:1", "'L'", "'R'")

    def test_memoized_evaluate_measures_once(self, monkeypatch):
        from types import SimpleNamespace

        calls: list[int] = []
        sentinel = object()

        def fake_evaluate(cands, **kw):
            calls.append(len(cands))
            return [sentinel for _ in cands]

        monkeypatch.setattr(pa.lp, "evaluate", fake_evaluate)
        memo: dict = {}
        evaluate = pa._memoized_evaluate(memo)
        cand = SimpleNamespace(label="a", lhs="L", rhs="R")
        assert evaluate([cand]) == [sentinel]
        assert calls == [1] and pa._cand_key(cand) in memo
        # the replay hits the memo — no second oracle draw
        assert evaluate([cand]) == [sentinel]
        assert calls == [1]

    def test_forked_rng_restores_torch_rng(self):
        import torch

        torch.manual_seed(1234)
        with pa._forked_rng():
            torch.manual_seed(99)
            torch.rand(4)
        after = torch.rand(1)
        torch.manual_seed(1234)
        expected = torch.rand(1)
        assert torch.equal(after, expected)


class TestScoreMemoization:
    def test_objective_score_measures_parent_once(self, monkeypatch):
        """The non-override ``_score`` path: the parent is measured
        once alongside the first probe, and repeat signatures replay
        the memo without a second oracle call."""
        calls: list = []

        def fake_score(record, *, seed, cfg):
            calls.append(tuple(record["parts"]))
            return {"score": 1.0, "greedy": 0.0, "evals_ep": 0.0}

        monkeypatch.setitem(
            pa.OBJECTIVES,
            "fake",
            pa.Objective(
                vocab=(),
                requires=(),
                parent=lambda: {"parts": [], "legal": "fake"},
                tables={},
                weight_names=(),
                score=fake_score,
            ),
        )
        board = pa.PlayerArena({"objective": "fake", "seed": 0})
        out = board._score({"parts": ["x"], "legal": "fake"})
        assert out["score"] == 1.0
        # parent + record measured, parent score banked
        assert calls == [(), ("x",)]
        assert board.parent_score == 1.0
        again = board._score({"parts": ["x"], "legal": "fake"})
        assert again is out and calls == [(), ("x",)]


class TestMoveRefusals:
    def test_toggle_add_blocked_by_requires(self):
        """``requires:<tag>`` — a part whose writer prerequisite is
        missing cannot be re-added (meta's ``action`` needs ``state``)."""
        board = pa.PlayerArena(
            _toy_case(objective="meta", score_fn=_smaller_is_better)
        )
        # drop both readers first, then the writer — only then is the
        # re-add of a reader legitimately refused
        board.step(Action("toggle_feature", {"part": "meta:action"}))
        board.step(Action("toggle_feature", {"part": "meta:macro"}))
        board.step(Action("toggle_feature", {"part": "meta:state"}))
        _, rep = board.step(
            Action("toggle_feature", {"part": "meta:action"})
        )
        assert not rep.applied and rep.note == "requires:meta:action"

    def test_toggle_unknown_part_declines(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        _, rep = board.step(
            Action("toggle_feature", {"part": "not-a-part"})
        )
        assert not rep.applied and rep.note == "unknown-part:not-a-part"

    def test_set_weights_unknown_table_declines(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        _, rep = board.step(
            Action("set_weights", {"table": "not-a-table"})
        )
        assert not rep.applied and rep.note == "unknown-table:not-a-table"

    def test_legal_actions_empty_when_done(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        st, _ = board.step(Action("extract", {}))
        assert st.done and pa.legal_actions(st) == ()

    def test_greedy_editor_stands_down_when_done(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        st, _ = board.step(Action("extract", {}))
        assert pa.GreedyEditor()(st) is None

    def test_greedy_editor_reverts_armed_drop_at_budget(self):
        """``_resolve`` with the eval budget spent re-adds the armed
        drop rather than probing it — the decline-and-restore path."""
        board = pa.PlayerArena(
            _toy_case(score_fn=_smaller_is_better, max_evals=0)
        )
        st = board.observe()
        ed = pa.GreedyEditor()
        ed._awaiting = "open:bias"
        move = ed(st)
        assert move == Action(
            "toggle_feature", {"part": "open:bias", "add": True}
        )
        assert ed._awaiting is None


class TestMainDriver:
    def _episode(self, terminal: bool) -> "object":
        from gamekit import engine as ge

        reports = [
            ge.Report(
                action=Action("toggle_feature", {"part": "open:bias"}),
                reward=-0.002,
            ),
            ge.Report(
                action=Action("extract", {}),
                reward=0.4,
                terminal=terminal,
                note="child:0.5000 vs parent:0.1000",
                detail={
                    "child": {"parts": ["open:bias"]},
                    "child_score": 0.5,
                    "parent_score": 0.1,
                    "delta": 0.4,
                    "probes": [],
                },
            ),
        ]
        return ge.Trajectory(reports=reports)

    def test_report_table_renders_moves(self):
        table = pa._report_table(self._episode(True).reports)
        assert "toggle_feature" in table and "part=open:bias" in table
        assert "extract" in table

    def test_main_success_writes_json(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(
            pa.engine, "run_episode", lambda b, p, n: self._episode(True)
        )
        out = tmp_path / "child.json"
        rc = pa.main(
            [
                "--objective",
                "open",
                "--pages",
                "2",
                "--epochs",
                "1",
                "--json",
                str(out),
            ]
        )
        assert rc == 0
        blob = json.loads(out.read_text())
        assert blob["objective"] == "open" and blob["delta"] == 0.4
        assert "child:  score=0.5000" in capsys.readouterr().out

    def test_main_success_without_json(self, monkeypatch, capsys):
        monkeypatch.setattr(
            pa.engine, "run_episode", lambda b, p, n: self._episode(True)
        )
        assert pa.main(["--objective", "open"]) == 0
        assert "player arena" in capsys.readouterr().out

    def test_main_reports_an_unfinished_episode(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            pa.engine,
            "run_episode",
            lambda b, p, n: self._episode(False),
        )
        assert pa.main(["--objective", "meta"]) == 1
        assert "without extract" in capsys.readouterr().out

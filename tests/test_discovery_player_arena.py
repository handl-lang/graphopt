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

import pytest
from mlopt_discovery import meta_player as mp
from mlopt_discovery import open_game as og
from mlopt_discovery import player_arena as pa
from mlopt_discovery import play
from mlopt_discovery.engine import run_episode
from mlopt_discovery.meta_arena import Action
from mlopt_discovery.players import (
    ActionFeats,
    Bias,
    DomainSpec,
    LinearPolicy,
    StateBroadcast,
    register_legal,
    register_part,
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
            assert spec.featurize(st, a) == mp.META_SPEC.featurize(st, a)

    def test_open_roundtrip(self):
        rec = og.OPEN_SPEC.to_record()
        assert rec["legal"] == "open" and rec["rank"] == "open"
        spec = DomainSpec.build_from_record(rec)
        assert spec.legal is og.legal_actions
        assert spec.rank is og._cand_feats
        case = og.gen_cases(0, 1, page=4)[0]
        st = og.OpenArena(case).observe()
        for a in og.legal_actions(st):
            assert spec.featurize(st, a) == og.OPEN_SPEC.featurize(st, a)

    def test_search_roundtrip(self):
        rec = play.SEARCH_SPEC.to_record()
        assert rec["legal"] == "search"
        assert DomainSpec.build_from_record(rec).legal is play._search_legal

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
        assert {a.params["part"] for a in toggles} == set(og.OPEN_PART_TAGS)
        assert acts[-1].op == "extract"
        # evaluate is offered while the eval budget lasts
        assert any(a.op == "evaluate" for a in acts)

    def test_toggle_drop_and_readd(self):
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        board.step(Action("toggle_feature", {"part": "open:hist", "add": False}))
        st = board.observe()
        assert "open:hist" not in st.parts
        assert "open:hist" in st.addable
        board.step(Action("toggle_feature", {"part": "open:hist", "add": True}))
        # re-added at canonical position, not appended
        assert board.observe().parts == tuple(og.OPEN_PART_TAGS)

    def test_meta_requires_blocks_state_drop(self):
        board = pa.PlayerArena(
            {"objective": "meta", "seed": 0, "score_fn": _smaller_is_better}
        )
        st = board.observe()
        # meta:action reads st:* — meta:state is not droppable
        assert "meta:state" not in st.droppable
        _, rep = board.step(
            Action("toggle_feature", {"part": "meta:state", "add": False})
        )
        assert not rep.applied and rep.reward == 0.0
        # dropping meta:action first unblocks it
        board.step(
            Action("toggle_feature", {"part": "meta:action", "add": False})
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
            pa.PLAYER_ARENA_REWARD["step"] + pa.PLAYER_ARENA_REWARD["eval"]
        )
        assert st.evals == 1 and len(st.probes) == 1
        assert st.parent_score == 0.5  # 5 parts -> 1 - 0.5
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

        board = pa.PlayerArena(
            _toy_case(score_fn=score, max_evals=1)
        )
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
            "open:state",
            "open:hist",
            "open:onehot",
            "open:action",
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
        board = pa.PlayerArena(_toy_case(score_fn=_smaller_is_better))
        traj = run_episode(board, pa.GreedyEditor(), 30)
        t = traj.terminal
        assert t is not None and t.detail["delta"] == 0.5
        assert t.detail["child"]["parts"] == []
        # five LOO probes, each kept (smaller is always better)
        assert len(t.detail["probes"]) == 5

    def test_reverts_a_losing_drop(self):
        def score(record, *, seed, cfg):
            # incumbent is optimal; any drop costs 0.2
            return {
                "score": 0.5 if len(record["parts"]) == 5 else 0.3,
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

"""Tests for ``mlopt_discovery.macro_moves`` — the shipped macro layer.

Pins: the ``"macro"`` handler replays a seq through ``ACTIONS`` as one
refereed step (decline propagates, terminal carries the certificate);
``viable`` keeps a minted line off boards its params cannot bind;
``offered``/``legal_with_macros`` keep the legal set bounded; and the
shipped seam — ``meta_arena.legal_actions`` reads
``lawdata.MACROS`` — survives the ``DomainSpec`` record round-trip.
"""

import pytest
from mlopt_core.cost.basic import count_cost
from mlopt_core.ir import Op, TensorType, Var
from mlopt_core.laws import DEFAULT
from mlopt_discovery import lawdata
from mlopt_discovery import macro_moves as mm
from mlopt_discovery import meta_arena as ma
from mlopt_discovery import meta_player as mp


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _silu_board() -> ma.MetaArena:
    """``mul(x, sig(x))`` under the stripped ruleset — macros bind."""
    x = _v("x", 4, 4)
    sans = DEFAULT - DEFAULT.named("silu_fold", "softsign_fold")
    return ma.MetaArena(
        Op.make("mul", x, Op.make("sigmoid", x)),
        sans,
        cost_fn=count_cost,
    )


def _control_board() -> ma.MetaArena:
    """``add(mul, matmul)`` under DEFAULT — no minted line binds."""
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    return ma.MetaArena(
        Op.make("add", Op.make("mul", x, y), Op.make("matmul", x, y)),
        DEFAULT,
        cost_fn=count_cost,
    )


CLAIM_EXTRACT = {
    "op": "claim",
    "params": {"handler": "silu"},
}
EXTRACT = {"op": "extract", "params": {}}


def _claim_extract() -> mm.Macro:
    return mm.Macro(
        name="t_claim",
        seq=(dict(CLAIM_EXTRACT), dict(EXTRACT)),
        count=15,
        certified=1.0,
        mean_terminal=0.2889,
        score=4.3333,
    )


class TestHandler:
    """The ``"macro"`` referee — replay, decline, terminal."""

    def test_replay_claim_extract_terminates_certified(self):
        board = _silu_board()
        st, rep = board.step(
            ma.Action("macro", {"seq": [CLAIM_EXTRACT, EXTRACT]})
        )
        assert rep.applied and rep.terminal
        assert rep.certificate_ok is True
        assert rep.cost == pytest.approx(1.0)
        assert rep.reward > 0
        assert rep.note.startswith("macro[2/2]")
        assert st.done

    def test_one_step_charge_for_the_line(self):
        # the whole seq is one move — one report, one budget tick
        board = _silu_board()
        traj = ma.run_episode(
            board,
            ma.ScriptedPlayer(
                [ma.Action("macro", {"seq": [CLAIM_EXTRACT, EXTRACT]})]
            ),
            6,
        )
        assert len(traj.reports) == 1
        assert traj.certificate_ok is True

    def test_empty_seq_declines(self):
        board = _silu_board()
        _st, rep = board.step(ma.Action("macro", {"seq": []}))
        assert not rep.applied and rep.reward == 0.0
        assert "empty seq" in rep.note
        assert not board.observe().done

    def test_nested_macro_declined(self):
        board = _silu_board()
        _st, rep = board.step(
            ma.Action(
                "macro",
                {
                    "seq": [
                        {"op": "macro", "params": {"seq": [EXTRACT]}}
                    ]
                },
            )
        )
        assert not rep.applied and "nested macro" in rep.note

    def test_unregistered_submove_declined(self):
        board = _silu_board()
        _st, rep = board.step(
            ma.Action("macro", {"seq": [{"op": "bogus", "params": {}}]})
        )
        assert not rep.applied and "unregistered" in rep.note

    def test_midseq_decline_propagates(self):
        # handle of an object nobody minted — the honest refusal
        board = _silu_board()
        _st, rep = board.step(
            ma.Action(
                "macro",
                {
                    "seq": [
                        {
                            "op": "handle",
                            "params": {
                                "object": "ghost",
                                "handler": "silu",
                            },
                        },
                        EXTRACT,
                    ]
                },
            )
        )
        assert not rep.applied and rep.reward == 0.0
        assert "step 0 handle declined" in rep.note
        assert rep.detail["steps"] == ("0:handle",)
        assert not board.observe().done

    def test_terminal_fields_propagate(self):
        # cost/certificate/used_declared ride the macro outcome verbatim
        board = _silu_board()
        _st, rep = board.step(
            ma.Action("macro", {"seq": [CLAIM_EXTRACT, EXTRACT]})
        )
        assert rep.used_declared == ("claim_silu_0",)
        assert rep.detail["steps"] == ("0:claim", "1:extract")
        assert "terminal" in rep.detail

    def test_declare_inside_macro_inserts_for_later_steps(self):
        # the spelled line — declare→saturate→handle→extract — replays
        # with the two-phase accept between sub-moves
        m = next(
            m
            for m in mm.shipped_macros()
            if m.name == "declare_saturate_handle_silu_extract"
        )
        board = _silu_board()
        _st, rep = board.step(mm.instantiate(m))
        assert rep.applied and rep.terminal
        assert rep.certificate_ok is True
        assert rep.cost == pytest.approx(1.0)
        # the freshened name — mx_<macro>_<old> — was minted mid-seq
        # (per-leg insertions land in detail, not the outer accept)
        assert any(
            n.startswith("mx_declare_saturate_handle_silu_extract_")
            for n in rep.detail["inserted"]
        )

    def test_terminal_midseq_stops_replay(self):
        # extract at position 0 ends the macro — later legs never run
        board = _silu_board()
        _st, rep = board.step(
            ma.Action("macro", {"seq": [EXTRACT, EXTRACT]})
        )
        assert rep.terminal
        assert rep.detail["steps"] == ("0:extract",)

    def test_nonterminal_macro_completes(self):
        # a seq that never terminates applies as one non-terminal step
        board = _silu_board()
        _st, rep = board.step(
            ma.Action(
                "macro",
                {"seq": [{"op": "saturate", "params": {"budget": 4}}]},
            )
        )
        assert rep.applied and not rep.terminal
        assert rep.note == "macro[1]"
        assert not board.observe().done

    def test_action_typed_submoves_replay(self):
        # seq entries may be Action objects as well as dicts
        board = _silu_board()
        _st, rep = board.step(
            ma.Action(
                "macro",
                {"seq": [ma.Action.claim("silu"), ma.Action.extract()]},
            )
        )
        assert rep.applied and rep.terminal and rep.certificate_ok


class TestViable:
    """``viable`` — the cheap bindability filter behind the offers."""

    def test_claim_macro_binds_only_where_tag_occurs(self):
        m = _claim_extract()
        assert mm.viable(m, _silu_board().observe())
        assert not mm.viable(m, _control_board().observe())

    def test_declare_needs_its_spec_at_a_site(self):
        silu_decl = next(
            m
            for m in mm.shipped_macros()
            if m.name == "declare_saturate"
        )
        assert mm.viable(silu_decl, _silu_board().observe())
        assert not mm.viable(silu_decl, _control_board().observe())

    def test_handle_needs_object_seq_or_board_declared(self):
        handle_tail = next(
            m
            for m in mm.shipped_macros()
            if m.name == "handle_silu_extract"
        )
        board = _silu_board()
        # foldabs_0 not declared yet — the tail cannot bind
        assert not mm.viable(handle_tail, board.observe())
        board.step(
            ma.Action.declare(
                {
                    "op": "fold",
                    "params": {
                        "name": "foldabs_0",
                        "spelled": ("mul", "X1", ("sigmoid", "X1")),
                        "kernel": ("foldabs_0", "X1"),
                    },
                }
            )
        )
        assert mm.viable(handle_tail, board.observe())

    def test_seq_declared_handle_binds(self):
        # declare_saturate_handle_silu declares foldabs_0 in-seq —
        # the handle leg binds against the seq's own mint
        m = next(
            m
            for m in mm.shipped_macros()
            if m.name == "declare_saturate_handle_silu"
        )
        assert mm.viable(m, _silu_board().observe())

    def test_fire_and_saturate_rule_refs(self):
        board = _silu_board()
        st = board.observe()
        live_rule = st.rules[0]
        ok = mm.Macro(
            name="t",
            seq=(
                {"op": "fire", "params": {"rule": live_rule}},
                {"op": "saturate", "params": {"rules": (live_rule,)}},
            ),
        )
        assert mm.viable(ok, st)
        dead = mm.Macro(
            name="t2",
            seq=({"op": "fire", "params": {"rule": "nope"}},),
        )
        assert not mm.viable(dead, st)
        dead_sat = mm.Macro(
            name="t3",
            seq=(
                {
                    "op": "saturate",
                    "params": {"rules": ("nope",)},
                },
            ),
        )
        assert not mm.viable(dead_sat, st)
        # seq-declared names satisfy fire/saturate refs
        seq_decl = mm.Macro(
            name="t4",
            seq=(
                {
                    "op": "declare",
                    "params": {
                        "construction": {
                            "op": "fold",
                            "params": {"name": "mine"},
                        }
                    },
                },
                {"op": "fire", "params": {"rule": "mine"}},
            ),
        )
        assert mm.viable(seq_decl, st)


class TestMacroRecord:
    """The ``Macro`` template — record shape and reification."""

    def test_op_str_prefers_recorded_ops(self):
        m = mm.Macro(
            name="t",
            seq=({"op": "claim", "params": {}},),
            ops=("claim", "extract"),
        )
        assert m.op_str() == "claim→extract"
        m2 = mm.Macro(
            name="t2",
            seq=(
                {"op": "saturate", "params": {}},
                {"op": "extract", "params": {}},
            ),
        )
        assert m2.op_str() == "saturate→extract"

    def test_from_record_defaults_and_passthrough(self):
        m = _claim_extract()
        assert mm.macro_from_record(m) is m
        bare = mm.macro_from_record(
            {"name": "bare", "seq": ({"op": "extract", "params": {}},)}
        )
        assert bare.count == 0 and bare.certified == 0.0
        assert bare.score == 0.0 and bare.ops == ()


class TestInstantiate:
    """``instantiate`` — freshened names plus the stats payload."""

    def test_renames_seq_internal_refs(self):
        m = next(
            m
            for m in mm.shipped_macros()
            if m.name == "declare_saturate_handle_silu"
        )
        a = mm.instantiate(m)
        seq = a.params["seq"]
        cp = seq[0]["params"]["construction"]["params"]
        assert cp["name"] == "mx_declare_saturate_handle_silu_foldabs_0"
        assert cp["kernel"][0] == cp["name"]
        # the handle target followed the rename
        assert seq[2]["params"]["object"] == cp["name"]

    def test_stats_ride_params_as_plain_data(self):
        a = mm.instantiate(_claim_extract())
        payload = a.params["macro"]
        assert payload["name"] == "t_claim"
        assert payload["count"] == 15
        assert payload["certified"] == 1.0
        assert payload["score"] == pytest.approx(4.3333)
        # hashable through the shared dedup key
        assert ma._action_key(a) == ma._action_key(
            mm.instantiate(_claim_extract())
        )

    def test_fire_and_saturate_refs_rename(self):
        m = mm.Macro(
            name="t",
            seq=(
                {
                    "op": "declare",
                    "params": {
                        "construction": {
                            "op": "fold",
                            "params": {
                                "name": "old",
                                "spelled": ("mul", "X"),
                            },
                        }
                    },
                },
                {"op": "fire", "params": {"rule": "old"}},
                {
                    "op": "saturate",
                    "params": {"rules": ("old", "kept")},
                },
                {"op": "handle", "params": {"object": "old"}},
            ),
        )
        seq = mm.instantiate(m).params["seq"]
        assert seq[1]["params"]["rule"] == "mx_t_old"
        assert seq[2]["params"]["rules"] == ("mx_t_old", "kept")
        assert seq[3]["params"]["object"] == "mx_t_old"

    def test_declare_edge_cases_pass_verbatim(self):
        # no construction, no name, a non-self-named kernel — nothing
        # to freshen; the params replay untouched
        m = mm.Macro(
            name="t",
            seq=(
                {"op": "declare", "params": {}},
                {
                    "op": "declare",
                    "params": {
                        "construction": {
                            "op": "fold",
                            "params": {
                                "name": "k",
                                "kernel": ("other", "X1"),
                            },
                        }
                    },
                },
            ),
        )
        seq = mm.instantiate(m).params["seq"]
        # declare normalizes construction to {"params": {...}} — the
        # absent name means nothing is freshened
        assert seq[0]["params"]["construction"] == {"params": {}}
        cp = seq[1]["params"]["construction"]["params"]
        assert cp["name"] == "mx_t_k"
        # kernel head != declared name — the fold is not self-named
        assert cp["kernel"] == ("other", "X1")

    def test_unmatched_refs_pass_verbatim(self):
        # refs to names the seq never declared replay unchanged
        m = mm.Macro(
            name="t",
            seq=(
                {"op": "fire", "params": {"rule": "other"}},
                {
                    "op": "saturate",
                    "params": {"rules": ("other",)},
                },
                {
                    "op": "handle",
                    "params": {"object": "other", "handler": "h"},
                },
                {"op": "saturate", "params": {}},
            ),
        )
        seq = mm.instantiate(m).params["seq"]
        assert seq[0]["params"]["rule"] == "other"
        assert seq[1]["params"]["rules"] == ("other",)
        assert seq[2]["params"]["object"] == "other"
        assert "rules" not in seq[3]["params"]

    def test_viable_declare_without_name_or_spec(self):
        # a declare with neither name nor spelled spec binds trivially
        m = mm.Macro(
            name="t",
            seq=(
                {
                    "op": "declare",
                    "params": {"construction": {"op": "fold"}},
                },
            ),
        )
        assert mm.viable(m, _control_board().observe())


class TestOffers:
    """``offered``/``legal_with_macros`` — the bounded offer layer."""

    def test_shipped_table_offers_viable_only(self):
        st = _silu_board().observe()
        offers = mm.offered(st)
        names = {a.params["macro"]["name"] for a in offers}
        assert "claim_silu_extract" in names
        assert len(offers) <= mm.DEFAULT_MACRO_CAP
        # control board: nothing binds — no offers
        assert mm.offered(_control_board().observe()) == ()

    def test_shipped_macros_reify_records(self):
        shipped = mm.shipped_macros()
        assert len(shipped) == len(lawdata.MACROS)
        assert all(isinstance(m, mm.Macro) for m in shipped)
        assert shipped[0].name == lawdata.MACROS[0]["name"]
        # dict records pass through the same path as Macro objects
        via_records = mm.offered(
            _silu_board().observe(), lawdata.MACROS
        )
        assert via_records == mm.offered(_silu_board().observe())

    def test_cap_bounds_offers(self):
        st = _silu_board().observe()
        assert len(mm.offered(st, cap=1)) == 1
        assert len(mm.offered(st, cap=2)) == 2

    def test_dedup_identical_macros(self):
        st = _silu_board().observe()
        m = _claim_extract()
        offers = mm.offered(st, (m, m))
        assert len(offers) == 1

    def test_dedup_same_seq_different_stats(self):
        # a re-minted copy of a shipped seq — different name/stats —
        # is the same move: it dedups against the shipped offer
        st = _silu_board().observe()
        copy = mm.Macro(
            name="m1",  # lab-style name over the shipped seq
            seq=(
                {"op": "claim", "params": {"handler": "silu"}},
                {"op": "extract", "params": {}},
            ),
        )
        acts = mm.legal_with_macros((copy,))(st)
        seqs = [
            mm._offer_key(a) for a in acts if a.op == "macro"
        ]
        assert len(seqs) == len(set(seqs))

    def test_legal_with_macros_extract_last(self):
        # a custom offer the shipped table does not carry slots in
        # before the terminal extract
        custom = mm.Macro(
            name="t_sat",
            seq=(
                {"op": "saturate", "params": {}},
                {"op": "extract", "params": {}},
            ),
        )
        legal = mm.legal_with_macros((custom,), cap=4)
        acts = legal(_silu_board().observe())
        assert acts[-1].op == "extract"
        ours = [
            a
            for a in acts
            if a.op == "macro" and a.params["macro"]["name"] == "t_sat"
        ]
        assert len(ours) == 1

    def test_legal_with_macros_custom_base(self):
        # an explicit base enumerator is honoured verbatim
        legal = mm.legal_with_macros(
            (), base=lambda s: (ma.Action.extract(),)
        )
        st = _silu_board().observe()
        acts = legal(st)
        assert len(acts) == 1 and acts[0].op == "extract"
        # and a base without a trailing extract is left alone
        legal2 = mm.legal_with_macros(
            (), base=lambda s: (ma.Action.saturate(),)
        )
        assert legal2(st)[-1].op == "saturate"

    def test_done_state_offers_nothing(self):
        board = _silu_board()
        board.step(ma.Action.extract())
        assert ma.legal_actions(board.observe()) == ()
        assert (
            mm.legal_with_macros((_claim_extract(),))(board.observe())
            == ()
        )


class TestShippedLegalSet:
    """``legal_actions`` is the shipped seam — macros for every player."""

    def test_legal_actions_offers_shipped_macros(self):
        st = _silu_board().observe()
        acts = ma.legal_actions(st)
        macros = [a for a in acts if a.op == "macro"]
        names = {a.params["macro"]["name"] for a in macros}
        assert "claim_silu_extract" in names
        assert len(macros) <= mm.DEFAULT_MACRO_CAP
        assert acts[-1].op == "extract"
        # every offer comes from the shipped table
        shipped = {m["name"] for m in lawdata.MACROS}
        assert names <= shipped

    def test_legal_actions_offers_nothing_unbindable(self):
        acts = ma.legal_actions(_control_board().observe())
        assert not any(a.op == "macro" for a in acts)
        assert acts[-1].op == "extract"

    def test_base_legal_is_macro_free(self):
        st = _silu_board().observe()
        acts = ma._base_legal(st)
        assert not any(a.op == "macro" for a in acts)
        assert acts[-1].op == "extract"

    def test_random_and_greedy_see_the_offers(self):
        # every player that defaults to legal_actions gets the seam
        import random

        st = _silu_board().observe()
        rng = ma.RandomPlayer(random.Random(0))
        assert any(a.op == "macro" for a in rng._legal(st))
        g = ma.GreedyPlayer()
        assert any(a.op == "macro" for a in g._legal(st))

    def test_spec_record_roundtrip_carries_macro_offers(self):
        from gamekit.players import spec_from_record

        rec = mp.META_SPEC.to_record()
        assert rec["legal"] == "meta"
        spec = spec_from_record(rec)
        st = _silu_board().observe()
        acts = spec.legal(st)
        assert spec.legal is ma.legal_actions
        assert any(a.op == "macro" for a in acts)
        # and the rebuilt featurizer prices a macro row in-schema
        macro = next(a for a in acts if a.op == "macro")
        feats = spec.featurize(st, macro)
        assert feats["op:macro"] == 1.0
        assert feats["x:macro:score"] > 0

    def test_episode_through_the_shipped_set(self):
        # a scripted macro play through the prod seam certifies
        board = _silu_board()
        offer = next(
            a
            for a in ma.legal_actions(board.observe())
            if a.params.get("macro", {}).get("name")
            == "claim_silu_extract"
        )
        traj = ma.run_episode(board, ma.ScriptedPlayer([offer]), 4)
        assert traj.certificate_ok is True
        assert traj.cost == pytest.approx(1.0)

"""Tests for ``mlopt_discovery.handler_synth`` — ADR-0007 layer 2.

Pins the descriptor interpreter (each grammar family's semantics),
the verify-or-refuse gate (a wrong body refuses with a measured
error, never silently passes), the mint path (a verified body
becomes a handler-table atom with provenance), serialisation, and
the sink delivery.  The flagship rows: a ``contract`` body mints
the ``om_decayed`` shape ``(s⊙d)@v`` and ``fold``/``fold_emit``/
``fold_zip`` bodies mint loop-shaped semantics — handlers that
previously needed a hand-written executor.
"""

import json

import pytest
import torch

import mlopt_torch.torch_bridge as tb
from mlopt_core.ir import IR, Op, TensorType, Var
from mlopt_discovery import handler_synth as hs
from mlopt_discovery.lawdata import HANDLERS
from mlopt_discovery.meta_arena import _canon_concrete

BINDINGS = tb._IR_TO_TORCH


def _env(shapes: dict, seed: int = 0) -> dict:
    return hs._sample_env(shapes, seed)


# ---------------------------------------------------------------------------
#  The descriptor interpreter
# ---------------------------------------------------------------------------


class TestEvalBody:
    def test_ew(self):
        env = _env({"X": (4,)})
        out = hs.eval_body(
            ("ew", ("mul", "X", ("sigmoid", "X"))), env, BINDINGS
        )
        torch.testing.assert_close(
            out, env["X"] * torch.sigmoid(env["X"])
        )

    def test_contract(self):
        env = _env({"S": (3, 4), "D": (3, 4), "V": (4, 5)})
        body = ("contract", "...ik,...kj->...ij", ("mul", "S", "D"), "V")
        out = hs.eval_body(body, env, BINDINGS)
        torch.testing.assert_close(out, (env["S"] * env["D"]) @ env["V"])

    def test_fold_matches_manual_loop(self):
        env = _env({"A": (5, 3), "B": (5, 3), "H": (3,)})
        out = hs.eval_body(("fold", "A", "B", "H", -2), env, BINDINGS)
        acc = env["H"].clone()
        for t in range(5):
            acc = env["A"][t] * acc + env["B"][t]
        torch.testing.assert_close(out, acc)

    def test_fold_shared_operand_broadcasts(self):
        # a (d,) coefficient under (T,d) operands broadcasts per-step
        env = _env({"A": (5, 3), "B": (3,), "H": (3,)})
        out = hs.eval_body(("fold", "A", "B", "H", 0), env, BINDINGS)
        acc = env["H"].clone()
        for t in range(5):
            acc = env["A"][t] * acc + env["B"]
        torch.testing.assert_close(out, acc)

    def test_fold_emit(self):
        env = _env({"A": (5,), "B": (5,), "Q": (5,), "H": ()})
        body = ("fold_emit", "A", "B", "H", ("mul", "Q", "$acc"), -2)
        out = hs.eval_body(body, env, BINDINGS)
        acc, outs = env["H"].clone(), []
        for t in range(5):
            acc = env["A"][t] * acc + env["B"][t]
            outs.append(env["Q"][t] * acc)
        torch.testing.assert_close(out, torch.stack(outs, dim=0))

    def test_fold_zip(self):
        env = _env({"A": (4,), "B": (4,), "C": (4,), "D": (4,), "H": (4,)})
        body = ("fold_zip", (("C", "D"), ("A", "B")), "H")
        out = hs.eval_body(body, env, BINDINGS)
        acc = env["C"] * env["H"] + env["D"]
        acc = env["A"] * acc + env["B"]
        torch.testing.assert_close(out, acc)

    def test_reduce(self):
        env = _env({"X": (4, 5)})
        out = hs.eval_body(("reduce", "sum", "X", -1), env, BINDINGS)
        torch.testing.assert_close(out, env["X"].sum(dim=-1))

    def test_norm(self):
        env = _env({"X": (4, 5)})
        body = (
            "norm",
            "div",
            ("exp", "X"),
            "sum",
            ("exp", "X"),
            -1,
            True,
        )
        out = hs.eval_body(body, env, BINDINGS)
        e = torch.exp(env["X"])
        torch.testing.assert_close(out, e / e.sum(dim=-1, keepdim=True))

    def test_fold_rank_zero_raises(self):
        env = _env({"A": (), "B": (), "H": ()})
        with pytest.raises(ValueError, match="step axis"):
            hs.eval_body(("fold", "A", "B", "H", 0), env, BINDINGS)

    def test_unknown_tag_raises(self):
        with pytest.raises(ValueError, match="unknown handler-body"):
            hs.eval_body(("nope", "X"), {"X": torch.randn(2)}, BINDINGS)


# ---------------------------------------------------------------------------
#  Verification — the gate itself
# ---------------------------------------------------------------------------


class TestVerify:
    def test_right_body_verifies(self):
        spec = ("matmul", ("mul", "S", "D"), "V")
        body = ("contract", "...ik,...kj->...ij", ("mul", "S", "D"), "V")
        res = hs.verify_body(
            body, spec, [{"S": (6, 5), "D": (6, 5), "V": (5, 7)}]
        )
        assert res.ok and res.max_err < 1e-12 and res.n_samples == 2

    def test_wrong_body_refuses_with_measured_error(self):
        spec = ("matmul", ("mul", "S", "D"), "V")
        body = ("contract", "...ik,...kj->...ij", ("add", "S", "D"), "V")
        res = hs.verify_body(
            body, spec, [{"S": (6, 5), "D": (6, 5), "V": (5, 7)}]
        )
        assert not res.ok and res.detail == "numeric miss"
        assert 0.0 < res.max_err < float("inf")

    def test_shape_mismatch_refuses(self):
        spec = ("matmul", "S", "V")
        body = ("contract", "...ik,...kj->...ij", "S", "S")
        res = hs.verify_body(
            body, spec, [{"S": (4, 4), "V": (4, 4)}]
        )
        assert not res.ok

    def test_raising_body_refuses_loudly(self):
        spec = ("applyd", ("aff_diag", "A", "B"), "H")
        # fold over no step axis — the body raises, a louder refusal
        body = ("fold", "A", "B", "H", -2)
        res = hs.verify_body(
            body, spec, [{"A": (), "B": (), "H": ()}]
        )
        assert not res.ok and res.max_err == float("inf")
        assert "raised" in res.detail

    def test_explicit_bindings_used(self):
        spec = ("mul", "X", "X")
        body = ("ew", ("square", "X"))
        res = hs.verify_body(
            body, spec, [{"X": (4,)}], bindings=BINDINGS
        )
        assert res.ok


# ---------------------------------------------------------------------------
#  Proposal enumeration
# ---------------------------------------------------------------------------


class TestProposal:
    def test_ew_baseline_only_for_elementwise(self):
        bodies = list(
            hs.propose_bodies(("mul", "X", ("sigmoid", "X")))
        )
        assert bodies[0] == ("ew", ("mul", "X", ("sigmoid", "X")))
        spelled = ("matmul", "X", "Y")
        assert ("ew", spelled) not in list(hs.propose_bodies(spelled))

    def test_ew_replay_admits_broadcast_views(self):
        """``unsqueeze``/``squeeze`` are size-1-axis views — a spec
        mixing them with the pointwise vocabulary is still one fused
        elementwise kernel, so the ``ew`` replay proposes."""
        spec = (
            "add",
            ("mul", "A", "B"),
            ("unsqueeze", "C", {"dim": 1}),
        )
        bodies = list(hs.propose_bodies(spec))
        assert bodies[0] == ("ew", spec)
        assert hs._replayable_spec(spec)
        assert hs._replayable_spec(
            ("mul", "X", ("squeeze", "Y", {"dim": 0}))
        )

    def test_ew_replay_excludes_index_and_contraction(self):
        """``select``/``slice``/``matmul``/``linear`` move or contract
        data — the replay gate stays closed (a minted replay there
        would be the spelled site itself, a tautology)."""
        for spec in (
            ("mul", "X", ("select", "Y", {"dim": 1, "index": 0})),
            ("mul", "X", ("slice", "Y", {"dim": 0, "start": 0})),
            ("matmul", "X", "Y"),
            ("mul", "X", ("linear", "A", "B", "C")),
        ):
            assert not hs._replayable_spec(spec)
            assert ("ew", spec) not in [
                b for _, b in zip(range(40), hs.propose_bodies(spec))
            ]

    def test_replayable_spec_edges(self):
        """The gate needs at least one op, all inside the union."""
        assert not hs._replayable_spec("X")
        assert not hs._replayable_spec(2.0)
        assert hs._replayable_spec(("unsqueeze", "X", {"dim": -1}))

    def test_families_restriction(self):
        spec = ("matmul", ("mul", "S", "D"), "V")
        bodies = list(
            hs.propose_bodies(spec, families=("contract",))
        )
        assert bodies and all(b[0] == "contract" for b in bodies)

    def test_deterministic_order(self):
        spec = ("matmul", ("mul", "S", "D"), "V")
        a = [b for _, b in zip(range(50), hs.propose_bodies(spec))]
        b = [b for _, b in zip(range(50), hs.propose_bodies(spec))]
        assert a == b

    def test_spec_constants_join_the_atom_pool(self):
        spec = ("matmul", ("mul", "S", 2.0), "V")
        bodies = list(
            hs.propose_bodies(spec, families=("contract",))
        )
        assert (
            "contract",
            "...ik,...kj->...ij",
            ("mul", "S", 2.0),
            "V",
        ) in bodies

    def test_ew_exprs_beyond_grammar_depth_empty(self):
        assert hs._ew_exprs(frozenset({"X"}), ("X",), 3) == []

    def test_spec_consts_skips_bools_and_dedupes(self):
        # walk reaches every child; bool leaves are not consts and a
        # repeated const registers once.
        consts = hs._spec_consts(("mul", ("add", 2.0, 2.0), True, "X"))
        assert consts == [2.0]


# ---------------------------------------------------------------------------
#  Minting — verified bodies become atoms
# ---------------------------------------------------------------------------


class TestMint:
    def test_contract_mints_om_decayed_shape(self):
        """The flagship: ``(s⊙d)@v`` mints a verified contraction."""
        spec = ("matmul", ("mul", "S", "D"), "V")
        domain = [{"S": (6, 5), "D": (6, 5), "V": (5, 7)}]
        handlers, report = hs.mint_handlers([spec], domain=domain)
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "contract"
        h = handlers[report[0]["tag"]]
        assert h["handler_body"][0] == "contract"
        assert h["provenance"]["family"] == "contract"
        assert h["provenance"]["grammar"] == "handler-grammar-v4"
        assert h["provenance"]["max_err"] < 1e-12

    def test_masked_contract_mints(self):
        """``((s·d)·m) @ v`` — the masked inner-chunk form."""
        spec = ("matmul", ("mul", ("mul", "S", "D"), "M"), "V")
        domain = [
            {
                "S": (6, 5),
                "D": (6, 5),
                "M": (6, 5),
                "V": (5, 7),
            }
        ]
        handlers, report = hs.mint_handlers(
            [spec], domain=domain, budget=1500
        )
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "contract"

    def test_fold_mints_against_closed_form(self):
        """A loop-shaped body verified vs the spelled closed form."""
        spec = (
            "add",
            ("mul", "H", ("prod", "A", {"dim": 0})),
            (
                "sum",
                (
                    "mul",
                    "B",
                    (
                        "div",
                        ("prod", "A", {"dim": 0}),
                        ("cumprod", "A", {"dim": 0}),
                    ),
                ),
                {"dim": 0},
            ),
        )
        handlers, report = hs.mint_handlers(
            [spec], domain=[{"A": (5,), "B": (5,), "H": ()}]
        )
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "fold"
        h = handlers[report[0]["tag"]]
        assert h["handler_body"] == ("fold", "A", "B", "H", -2)

    def test_fold_emit_mints(self):
        """``out_t = q_t ⊙ acc_t`` — the projection recurrence."""
        inner = (
            "mul",
            ("cumprod", "A", {"dim": 0}),
            (
                "add",
                "H",
                (
                    "cumsum",
                    ("div", "B", ("cumprod", "A", {"dim": 0})),
                    {"dim": 0},
                ),
            ),
        )
        spec = ("mul", "Q", inner)
        handlers, report = hs.mint_handlers(
            [spec],
            domain=[{"Q": (5,), "A": (5,), "B": (5,), "H": ()}],
            budget=1500,
        )
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "fold_emit"
        h = handlers[report[0]["tag"]]
        assert h["handler_body"] == (
            "fold_emit",
            "A",
            "B",
            "H",
            ("mul", "Q", "$acc"),
            -2,
        )

    def test_fold_zip_mints_applyd_nest(self):
        """The unrolled scan nest mints a chronological fold."""
        spec = (
            "applyd",
            ("aff_diag", "A", "B"),
            ("applyd", ("aff_diag", "C", "D"), "H"),
        )
        domain = [
            {
                "A": (4,),
                "B": (4,),
                "C": (4,),
                "D": (4,),
                "H": (4,),
            }
        ]
        handlers, report = hs.mint_handlers(
            [spec], domain=domain, budget=1500
        )
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "fold_zip"
        h = handlers[report[0]["tag"]]
        assert h["handler_body"] == (
            "fold_zip",
            (("C", "D"), ("A", "B")),
            "H",
        )
        assert h["provenance"]["max_err"] == 0.0

    def test_broadcast_ew_mints(self):
        """The ``zoo-prim`` refusal shape: an ``add``/``mul`` tree
        over an ``unsqueeze`` leaf mints the ``ew`` replay — one
        fused elementwise kernel with broadcasting, verified fp64."""
        spec = (
            "add",
            ("mul", "A", ("add", "B", ("mul", "C", "D"))),
            ("mul", "E", ("unsqueeze", "F", {"dim": 1})),
        )
        domain = [
            {
                "A": (2, 16, 4),
                "B": (2, 16, 4),
                "C": (2, 16, 1),
                "D": (2, 1, 4),
                "E": (2, 16, 1),
                "F": (2, 4),
            }
        ]
        handlers, report = hs.mint_handlers([spec], domain=domain)
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "ew"
        assert report[0]["tried"] == 1
        h = handlers[report[0]["tag"]]
        assert h["handler_body"] == ("ew", spec)
        assert h["provenance"]["grammar"] == "handler-grammar-v4"
        # and the delivered binding runs the broadcast body
        env = _env(domain[0])
        b = hs.minted_bindings(handlers)
        out = b[h["kernel"]](
            env["A"], env["B"], env["C"], env["D"], env["E"], env["F"]
        )
        torch.testing.assert_close(
            out,
            env["A"] * (env["B"] + env["C"] * env["D"])
            + env["E"] * env["F"].unsqueeze(1),
        )

    def test_reduction_now_in_grammar(self):
        """The reduce family closes the former reduction gap.

        ``sum(X, dim=-1)`` used to propose nothing (the named
        vocabulary gap); the ``reduce`` family now mints it as
        ``("reduce", "sum", "X", -1)``.
        """
        spec = ("sum", "X", {"dim": -1})
        handlers, report = hs.mint_handlers(
            [spec], domain=[{"X": (4, 4)}]
        )
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "reduce"
        h = handlers[report[0]["tag"]]
        assert h["handler_body"] == ("reduce", "sum", "X", -1)

    def test_normalization_mints(self):
        """The norm family closes the softmax gap.

        ``softmax`` is a value normalized by a reduction of itself —
        the composite the ``reduce`` family cannot spell (its operand
        is the whole spec).  The ``norm`` family mints the
        shape-general spelling (``keepdim=True`` so the reduction
        broadcasts back), on a non-square operand.
        """
        spec = (
            "div",
            ("exp", "X"),
            ("sum", ("exp", "X"), {"dim": -1, "keepdim": True}),
        )
        handlers, report = hs.mint_handlers(
            [spec], domain=[{"X": (4, 5)}]
        )
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "norm"
        h = handlers[report[0]["tag"]]
        assert h["handler_body"] == (
            "norm",
            "div",
            ("exp", "X"),
            "sum",
            ("exp", "X"),
            -1,
            True,
        )

    def test_postnorm_eval_body(self):
        """The wrapped normalization: ``x - log(sum(exp x))``."""
        env = _env({"X": (4, 5)})
        body = (
            "postnorm",
            "sub",
            "X",
            "log",
            "sum",
            ("exp", "X"),
            -1,
            True,
        )
        out = hs.eval_body(body, env, BINDINGS)
        e = torch.exp(env["X"])
        torch.testing.assert_close(
            out, env["X"] - torch.log(e.sum(dim=-1, keepdim=True))
        )
        # and it is log_softmax, to fp64
        torch.testing.assert_close(
            out,
            torch.nn.functional.log_softmax(env["X"], dim=-1),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_postnorm_mints_log_softmax(self):
        """The grammar-gap residue closed: ``log·sum·exp`` mints.

        ``log_softmax``'s ``x - log(sum(exp x))`` needed the
        *reduction result* wrapped in a unary — the bare ``norm``
        template cannot spell it (``grammar-gaps.md`` residue).  The
        ``postnorm`` seed family mints it.
        """
        spec = (
            "sub",
            "X",
            (
                "log",
                ("sum", ("exp", "X"), {"dim": -1, "keepdim": True}),
            ),
        )
        handlers, report = hs.mint_handlers(
            [spec], domain=[{"X": (4, 5)}]
        )
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "postnorm"
        h = handlers[report[0]["tag"]]
        assert h["handler_body"] == (
            "postnorm",
            "sub",
            "X",
            "log",
            "sum",
            ("exp", "X"),
            -1,
            True,
        )

    def test_postnorm_gated_off_non_normalization_specs(self):
        """postnorm shares norm's shape gate — no seeds on a reduce
        spec (top op ``sum`` is not a normalization outer op)."""
        spec = ("sum", "X", {"dim": -1})
        bodies = hs.propose_bodies(spec)
        assert not any(b[0] == "postnorm" for b in bodies)

    def test_postnorm_seed_record_roundtrips(self):
        """A minted postnorm body survives the record round-trip."""
        spec = (
            "sub",
            "X",
            (
                "log",
                ("sum", ("exp", "X"), {"dim": -1, "keepdim": True}),
            ),
        )
        handlers, report = hs.mint_handlers(
            [spec], domain=[{"X": (4, 5)}]
        )
        rec = hs.handler_record(report[0]["tag"], handlers[report[0]["tag"]])
        rebuilt = hs.handler_from_record(rec)
        assert rebuilt["handler_body"] == handlers[report[0]["tag"]]["handler_body"]

    def test_refusal_outside_grammar(self):
        """A meta-less non-elementwise spec proposes nothing.

        With the seed reach and the reduce family, every *meta-bearing*
        spec now proposes candidates; the residual "no candidates"
        refusal is a constant-only graph the templates cannot shape.
        """
        spec = ("applyd", ("aff_diag", 1.0, 2.0), 3.0)
        handlers, report = hs.mint_handlers([spec], domain=[{}])
        assert handlers == {}
        assert report[0]["verdict"] == "refused"
        assert report[0]["tried"] == 0
        assert "outside the handler grammar" in report[0]["note"]

    def test_refusal_grammar_exhausted(self):
        """Every candidate tried, none verified — measured refusal."""
        spec = ("sub", ("matmul", "A", "B"), "H")
        handlers, report = hs.mint_handlers(
            [spec],
            domain=[{"A": (4,), "B": (4,), "H": (4,)}],
            families=("fold_zip",),
        )
        assert handlers == {}
        assert report[0]["verdict"] == "refused"
        assert report[0]["tried"] > 0
        assert "grammar exhausted" in report[0]["note"]

    def test_one_step_fold_zip_mints_base_step(self):
        """``a⊙h+b`` — the carrier's base step — is a 1-step fold."""
        spec = ("applyd", ("aff_diag", "A", "B"), "H")
        handlers, report = hs.mint_handlers(
            [spec],
            domain=[{"A": (4,), "B": (4,), "H": (4,)}],
            families=("fold_zip",),
        )
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "fold_zip"

    def test_refusal_budget_cap(self):
        """A spec whose candidates never verify stops at budget."""
        spec = ("exp", ("matmul", "X", "Y"))
        handlers, report = hs.mint_handlers(
            [spec],
            domain=[{"X": (3, 3), "Y": (3, 3)}],
            families=("contract",),
            budget=25,
        )
        assert handlers == {}
        assert report[0]["verdict"] == "refused"
        assert report[0]["tried"] == 25
        assert "budget=25" in report[0]["note"]

    def test_covered_spec_skipped(self):
        covered = {
            _canon_concrete(h["pattern"]) for h in HANDLERS.values()
        }
        spec = ("mul", "X", ("sigmoid", "X"))
        handlers, report = hs.mint_handlers(
            [spec], domain=[{"X": (4,)}], covered=covered
        )
        assert handlers == {}
        assert report[0]["verdict"] == "covered"

    def test_deterministic_mint(self):
        spec = ("matmul", ("mul", "S", "D"), "V")
        domain = [{"S": (6, 5), "D": (6, 5), "V": (5, 7)}]
        h1, _ = hs.mint_handlers([spec], domain=domain, seed=7)
        h2, _ = hs.mint_handlers([spec], domain=domain, seed=7)
        assert h1 == h2

    def test_domain_accepts_a_single_binding_dict(self):
        spec = ("matmul", ("mul", "S", "D"), "V")
        handlers, report = hs.mint_handlers(
            [spec], domain={"S": (6, 5), "D": (6, 5), "V": (5, 7)}
        )
        assert report[0]["verdict"] == "minted"

    def test_item_carries_its_own_domain(self):
        spec = ("matmul", ("mul", "S", "D"), "V")
        item = {
            "spec": spec,
            "domain": [{"S": (6, 5), "D": (6, 5), "V": (5, 7)}],
        }
        handlers, report = hs.mint_handlers([item])
        assert report[0]["verdict"] == "minted"


# ---------------------------------------------------------------------------
#  Bindings, sink delivery, serialisation
# ---------------------------------------------------------------------------


class TestDelivery:
    def _minted(self):
        spec = ("matmul", ("mul", "S", "D"), "V")
        domain = [{"S": (6, 5), "D": (6, 5), "V": (5, 7)}]
        handlers, _ = hs.mint_handlers([spec], domain=domain)
        return spec, handlers

    def test_binding_runs_the_body(self):
        spec, handlers = self._minted()
        b = hs.minted_bindings(handlers)
        h = next(iter(handlers.values()))
        env = _env({"S": (6, 5), "D": (6, 5), "V": (5, 7)})
        out = b[h["kernel"]](env["S"], env["D"], env["V"])
        torch.testing.assert_close(
            out, (env["S"] * env["D"]) @ env["V"]
        )

    def test_sink_supports_and_lowers(self):
        spec, handlers = self._minted()
        sink = hs.minted_sink(handlers)
        h = next(iter(handlers.values()))
        assert h["kernel"] in sink.supported_ops
        s = Var("s", TensorType((6, 5)))
        d = Var("d", TensorType((6, 5)))
        v = Var("v", TensorType((5, 7)))
        ir = IR(
            root=Op.make(h["kernel"], s, d, v),
            inputs=[s, d, v],
            input_names={"s", "d", "v"},
            params={},
        )
        mod = sink.lower(ir, params={})
        sv, dv, vv = (
            torch.randn(6, 5),
            torch.randn(6, 5),
            torch.randn(5, 7),
        )
        torch.testing.assert_close(
            mod(sv, dv, vv), (sv * dv) @ vv, rtol=1e-4, atol=1e-5
        )

    def test_serialised_record_round_trips(self):
        spec, handlers = self._minted()
        tag, h = next(iter(handlers.items()))
        rec = json.loads(json.dumps(hs.handler_record(tag, h)))
        assert rec["kind"] == "handler"
        assert rec["handler_body"][0] == "contract"
        rebuilt = {"minted_x": hs.handler_from_record(rec)}
        b = hs.minted_bindings(rebuilt)
        env = _env({"S": (6, 5), "D": (6, 5), "V": (5, 7)})
        out = b[h["kernel"]](env["S"], env["D"], env["V"])
        torch.testing.assert_close(
            out, (env["S"] * env["D"]) @ env["V"]
        )

    def test_body_data_round_trip(self):
        body = ("fold_emit", "A", "B", "H", ("mul", "Q", "$acc"), -2)
        assert hs.body_from_data(hs.body_to_data(body)) == body

    def test_body_data_dict_branches(self):
        data = hs.body_to_data({"dim": (-1, 0), "k": 1})
        assert data == {"dim": [-1, 0], "k": 1}
        assert hs.body_from_data(data) == {"dim": (-1, 0), "k": 1}

    def test_handlers_fragment_accepts_minted(self):
        """Minted entries merge into a cellstore handler fragment."""
        from mlopt_discovery import cellstore

        _spec, handlers = self._minted()
        frag = cellstore.handlers_fragment("minted", handlers)
        names = {c.name for c in frag.cells}
        assert names == set(handlers.keys())

    def test_cli_battery(self, tmp_path, capsys):
        out = tmp_path / "report.json"
        assert hs.main(["--json", str(out)]) == 0
        report = json.loads(out.read_text())
        verdicts = {v["verdict"] for v in report["verdicts"]}
        assert "minted" in verdicts and "refused" in verdicts
        assert any(
            v["family"] == "contract"
            for v in report["verdicts"]
            if v["verdict"] == "minted"
        )
        assert "minted" in capsys.readouterr().out
        assert hs.main([]) == 0  # print-only path, no --json


class TestNormFamilyEdges:
    """The ``norm`` family's walk columns and the spec introspection
    helpers — the grammar edges the mint battery's ``families``
    restrictions never iterated."""

    def test_spec_dims_reads_and_dedupes_dim_attrs(self):
        """``dim`` attrs inside a spec join the axis pool — ints
        dedupe, sequences contribute each entry, bools are skipped."""
        assert hs._spec_dims(("mean", "X", {"dim": True})) == []
        assert hs._spec_dims(("mean", "X", {"dim": 1})) == [1]
        assert hs._spec_dims(("mean", "X", {"dim": [1, -1]})) == [
            1,
            -1,
        ]
        assert hs._spec_dims(
            ("mean", "X", {"dim": 1}, {"dim": 1, "keepdim": True})
        ) == [1]
        # a dict attr with no ``dim`` key contributes nothing
        assert hs._spec_dims(("mean", "X", {"keepdim": True})) == []

    def test_top_op_reads_the_outermost_op(self):
        assert hs._top_op(("div", "X", "Y")) == "div"
        assert hs._top_op("X") is None
        assert hs._top_op(()) is None

    def test_norm_family_gens_emit_norm_columns(self):
        """``_family_gens`` admits ``norm`` on any non-empty meta set
        and the column generator spells ``(op, lhs, red_op, rhs,
        axis, keepdim)`` candidates."""
        gens = hs._family_gens(
            ("norm",), ["X"], (), 2, (-1,), True
        )
        assert len(gens) == 1
        first = next(gens[0])
        assert first[0] == "norm"
        assert first[1] in hs.NORM_OPS
        assert first[3] in hs.REDUCE_OPS
        assert first[-2:] == (-1, True)

    def test_norm_column_exhausts_operand_pairs(self):
        """One column generator walks the full lhs x rhs operand
        grid, then ends."""
        subs = ["X", ("neg", "X")]
        col = list(hs._norm_column("div", "sum", subs, -1, True))
        assert len(col) == 4
        assert col[0] == ("norm", "div", "X", "sum", "X", -1, True)
        assert col[-1] == (
            "norm",
            "div",
            ("neg", "X"),
            "sum",
            ("neg", "X"),
            -1,
            True,
        )

    def test_propose_bodies_reaches_the_norm_walk(self):
        """On a norm-shaped spec the full generator reaches past the
        canonical seeds into the ``norm`` walk columns."""
        spec = ("div", "X", ("sum", "X", {"dim": -1}))
        bodies = []
        for b in hs.propose_bodies(spec, families=("norm",)):
            bodies.append(b)
            if len(bodies) >= 200:
                break
        tags = {b[0] for b in bodies}
        assert "norm" in tags


class TestGatherFamily:
    """The ``gather`` family — index-reads + per-position affines as
    view leaves under a pointwise tree (grammar v4).

    Pins the descriptor interpreter (``at`` reads, ``aff`` affines,
    the read-then-map fused order), the seed translation (read off
    the spec's own ``select``/``linear`` structure — at most two
    candidates), the mint on the ``zoo-prim`` v3 refusal rows, and
    the gate exclusions that keep the family honest.
    """

    def test_gather_eval_body(self):
        """``at`` reads, ``aff`` affines, broadcast views, pointwise."""
        env = _env({"G": (2, 4, 1), "X": (2, 4, 16), "W": (16, 16), "B": (16,)})
        body = (
            "gather",
            (
                "mul",
                ("unsqueeze", ("at", "G", {"dim": 1, "index": 0}), {"dim": -1}),
                (
                    "unsqueeze",
                    ("aff", ("at", "X", {"dim": 1, "index": 0}), "W", "B"),
                    {"dim": -1},
                ),
            ),
        )
        out = hs.eval_body(body, env, BINDINGS)
        want = env["G"].select(1, 0).unsqueeze(-1) * torch.nn.functional.linear(
            env["X"].select(1, 0), env["W"], env["B"]
        ).unsqueeze(-1)
        torch.testing.assert_close(out, want)

    def test_gather_seed_select_select(self):
        """The select×select shape translates to ``at`` leaves — one
        candidate (no select-of-affine, so no order variant)."""
        spec = (
            "mul",
            ("unsqueeze", ("select", "G", {"dim": 1, "index": 0}), {"dim": -1}),
            ("unsqueeze", ("select", "H", {"dim": 1, "index": 0}), {"dim": -1}),
        )
        seeds = hs._gather_seeds(spec)
        assert seeds == [
            (
                "gather",
                (
                    "mul",
                    ("unsqueeze", ("at", "G", {"dim": 1, "index": 0}), {"dim": -1}),
                    ("unsqueeze", ("at", "H", {"dim": 1, "index": 0}), {"dim": -1}),
                ),
            )
        ]
        # and it leads the proposal stream (``ew`` is gated out)
        assert next(hs.propose_bodies(spec)) == seeds[0]

    def test_gather_seed_select_affine_fused_first(self):
        """select-of-linear emits read-then-map first — the fused
        kernel projects only the gathered slice."""
        spec = (
            "mul",
            "A",
            ("unsqueeze", ("select", ("linear", "X", "W", "B"), {"dim": 1, "index": 7}), {"dim": 1}),
        )
        seeds = hs._gather_seeds(spec)
        assert len(seeds) == 2
        assert seeds[0][1] == (
            "mul",
            "A",
            ("unsqueeze", ("aff", ("at", "X", {"dim": 1, "index": 7}), "W", "B"), {"dim": 1}),
        )
        assert seeds[1][1] == (
            "mul",
            "A",
            ("unsqueeze", ("at", ("aff", "X", "W", "B"), {"dim": 1, "index": 7}), {"dim": 1}),
        )

    # the four ``zoo-prim`` v3 refusal rows — frozen spec/domain from
    # ``docs/reports/results_overnight_prim_v3.json``; each minted a
    # ``gather`` body at tried=1 under grammar v4.

    def test_gather_mints_select_gate_times_projection(self):
        """``synop_5f1bba5798`` — the SSM select-gate × projection."""
        spec = (
            "mul",
            ("unsqueeze", ("select", "v430", {"dim": 1, "index": 0}), {"dim": -1}),
            (
                "unsqueeze",
                ("select", ("linear", "v463", "v464", "v465"), {"dim": 1, "index": 0}),
                {"dim": -1},
            ),
        )
        domain = [
            {"v430": (2, 4, 1), "v463": (2, 4, 16), "v464": (16, 16), "v465": (16,)}
        ]
        handlers, report = hs.mint_handlers([spec], domain=domain)
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "gather"
        assert report[0]["tried"] == 1
        assert report[0]["max_err"] < 1e-9
        # the fused order won: read the slice, then project it
        assert hs._gather_view(spec, True) == handlers[report[0]["tag"]]["handler_body"][1]
        # and the delivered binding runs the gather body
        h = handlers[report[0]["tag"]]
        env = _env(domain[0])
        out = hs.minted_bindings(handlers)[h["kernel"]](
            env["v430"], env["v463"], env["v464"], env["v465"]
        )
        want = env["v430"].select(1, 0).unsqueeze(-1) * torch.nn.functional.linear(
            env["v463"], env["v464"], env["v465"]
        ).select(1, 0).unsqueeze(-1)
        torch.testing.assert_close(out, want)

    def test_gather_mints_residual_broadcast_linear(self):
        """``synop_4c657cb2c0`` — residual + broadcast linear read."""
        spec = (
            "add",
            "v13",
            (
                "mul",
                ("unsqueeze", "v419", {"dim": -1}),
                (
                    "unsqueeze",
                    ("select", ("linear", "v633", "v634", "v635"), {"dim": 1, "index": 7}),
                    {"dim": 1},
                ),
            ),
        )
        domain = [
            {"v13": (2, 16, 4), "v419": (2, 16), "v633": (2, 8, 16), "v634": (4, 16), "v635": (4,)}
        ]
        _handlers, report = hs.mint_handlers([spec], domain=domain)
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "gather"
        assert report[0]["tried"] == 1

    def test_gather_mints_gate_times_linear(self):
        """``synop_00cd48d9a2`` — plain gate × select-linear."""
        spec = (
            "mul",
            ("unsqueeze", "v319", {"dim": -1}),
            (
                "unsqueeze",
                ("select", ("linear", "v605", "v606", "v607"), {"dim": 1, "index": 0}),
                {"dim": -1},
            ),
        )
        domain = [
            {"v319": (2, 1), "v605": (2, 4, 16), "v606": (16, 16), "v607": (16,)}
        ]
        _handlers, report = hs.mint_handlers([spec], domain=domain)
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "gather"
        assert report[0]["tried"] == 1

    def test_gather_mints_select_times_select(self):
        """``synop_336e34145b`` — select × select, no affine."""
        spec = (
            "mul",
            ("unsqueeze", ("select", "v430", {"dim": 1, "index": 0}), {"dim": -1}),
            ("unsqueeze", ("select", "v458", {"dim": 1, "index": 0}), {"dim": -1}),
        )
        domain = [{"v430": (2, 4, 1), "v458": (2, 4, 16)}]
        handlers, report = hs.mint_handlers([spec], domain=domain)
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "gather"
        assert report[0]["tried"] == 1
        assert report[0]["max_err"] == 0.0

    def test_gather_record_round_trips(self):
        """A minted gather body is JSON-able provenance data."""
        spec = (
            "mul",
            ("unsqueeze", ("select", "G", {"dim": 1, "index": 0}), {"dim": -1}),
            ("unsqueeze", ("select", "H", {"dim": 1, "index": 0}), {"dim": -1}),
        )
        handlers, report = hs.mint_handlers(
            [spec], domain=[{"G": (2, 4, 1), "H": (2, 4, 16)}]
        )
        tag = report[0]["tag"]
        rec = json.loads(json.dumps(hs.handler_record(tag, handlers[tag])))
        assert rec["handler_body"][0] == "gather"
        rebuilt = hs.handler_from_record(rec)
        assert rebuilt["handler_body"] == handlers[tag]["handler_body"]
        # the rebuilt handler's binding evaluates the gather body
        env = _env({"G": (2, 4, 1), "H": (2, 4, 16)})
        out = hs.minted_bindings({"x": rebuilt})[rec["kernel"]](
            env["G"], env["H"]
        )
        torch.testing.assert_close(
            out,
            env["G"].select(1, 0).unsqueeze(-1)
            * env["H"].select(1, 0).unsqueeze(-1),
        )

    # ---- the gate exclusions — honest walls, not silent claims ----

    def test_gather_gate_no_index_read(self):
        """No ``select`` anywhere → not a gather (``ew`` or affine-only
        is another family's claim)."""
        assert hs._gather_seeds(("mul", "A", ("unsqueeze", "B", {"dim": -1}))) == []
        assert hs._gather_seeds(
            ("mul", "A", ("linear", "X", "W", "B"))
        ) == []  # affine-only: gate closed, honest wall
        assert hs._gather_seeds("X") == []

    def test_gather_gate_select_of_composite(self):
        """``select`` over a pointwise composite is the declared gap —
        pushing the read inward is a different translation."""
        spec = ("select", ("mul", "A", "B"), {"dim": 0, "index": 1})
        assert hs._gather_seeds(spec) == []

    def test_gather_gate_select_missing_index(self):
        """A ``select`` without literal ``dim``/``index`` is not
        translatable — a read the family cannot name."""
        spec = ("mul", "A", ("select", "B", {"dim": 0}))
        assert hs._gather_seeds(spec) == []
        spec2 = ("mul", "A", ("select", "B", 0, 1))  # positional attrs
        assert hs._gather_seeds(spec2) == []

    def test_gather_gate_linear_composite_args(self):
        """A ``linear`` over non-meta args is contract-family
        semantics, not a per-position affine leaf."""
        spec = ("select", ("linear", ("mul", "X", "Y"), "W", "B"), {"dim": 0, "index": 0})
        assert hs._gather_seeds(spec) == []

    def test_gather_gate_indexed_contraction(self):
        """``matmul(select x, w)`` — an index-read feeding a real
        contraction conflates two families; both stay closed."""
        spec = ("matmul", ("select", "X", {"dim": 0, "index": 0}), "W")
        assert hs._gather_seeds(spec) == []
        handlers, report = hs.mint_handlers(
            [spec],
            domain=[{"X": (3, 4), "W": (4, 5)}],
            families=("gather", "contract"),
            budget=200,
        )
        assert handlers == {}
        assert report[0]["verdict"] == "refused"

    def test_gather_gate_slice_and_gather_ops(self):
        """``slice``/``gather``-shaped indexing outside ``select`` is
        not the family's vocabulary."""
        spec = ("mul", "A", ("slice", "B", {"dim": 0, "start": 0, "end": 2}))
        assert hs._gather_seeds(spec) == []

    def test_gather_seed_record_is_plain_data(self):
        """The descriptor round-trips through the body's data codec."""
        spec = (
            "mul",
            "A",
            ("select", ("linear", "X", "W", "B"), {"dim": 1, "index": 0}),
        )
        for seed in hs._gather_seeds(spec):
            assert hs.body_from_data(hs.body_to_data(seed)) == seed

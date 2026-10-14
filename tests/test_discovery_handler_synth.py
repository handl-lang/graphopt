"""Tests for ``catopt_discovery.handler_synth`` — ADR-0007 layer 2.

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

import catopt_torch.torch_bridge as tb
from catopt_core.ir import IR, Op, TensorType, Var
from catopt_discovery import handler_synth as hs
from catopt_discovery.lawdata import HANDLERS
from catopt_discovery.meta_arena import _canon_concrete

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
        assert h["provenance"]["grammar"] == "handler-grammar-v1"
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

    def test_refusal_outside_grammar(self):
        """A reduction proposes nothing — the named vocabulary gap."""
        spec = ("sum", "X", {"dim": -1})
        handlers, report = hs.mint_handlers(
            [spec], domain=[{"X": (4, 4)}]
        )
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
        from catopt_discovery import cellstore

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

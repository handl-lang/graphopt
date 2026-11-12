"""Mint-path tests for ``mlopt_discovery.handler_synth`` — the
saturation-heavy half of ``test_discovery_handler_synth.py``.

Split out so the light interpreter/verify/proposal tests can run on
small-memory hosts: everything here calls ``hs.mint_handlers`` or
``hs.main`` (e-graph saturation), which is what OOMs 11 GB boxes.
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
        rec = hs.handler_record(
            report[0]["tag"], handlers[report[0]["tag"]]
        )
        rebuilt = hs.handler_from_record(rec)
        assert (
            rebuilt["handler_body"]
            == handlers[report[0]["tag"]]["handler_body"]
        )

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


@pytest.fixture(scope="module")
def minted_contract():
    """One saturation shared by the delivery tests — minting is
    deterministic (pinned by ``test_deterministic_mint``), so the four
    consumers reuse a single e-graph run instead of four."""
    spec = ("matmul", ("mul", "S", "D"), "V")
    domain = [{"S": (6, 5), "D": (6, 5), "V": (5, 7)}]
    handlers, _ = hs.mint_handlers([spec], domain=domain)
    return spec, handlers


class TestDelivery:
    def test_binding_runs_the_body(self, minted_contract):
        spec, handlers = minted_contract
        b = hs.minted_bindings(handlers)
        h = next(iter(handlers.values()))
        env = _env({"S": (6, 5), "D": (6, 5), "V": (5, 7)})
        out = b[h["kernel"]](env["S"], env["D"], env["V"])
        torch.testing.assert_close(
            out, (env["S"] * env["D"]) @ env["V"]
        )

    def test_sink_supports_and_lowers(self, minted_contract):
        spec, handlers = minted_contract
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

    def test_serialised_record_round_trips(self, minted_contract):
        spec, handlers = minted_contract
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

    def test_handlers_fragment_accepts_minted(self, minted_contract):
        """Minted entries merge into a cellstore handler fragment."""
        from mlopt_discovery import cellstore

        _spec, handlers = minted_contract
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


class TestGatherMint:
    """The ``zoo-prim`` v3 gather mints — moved here with the rest
    of the saturation-heavy tests (see module docstring).
    """

    def test_gather_mints_select_gate_times_projection(self):
        """``synop_5f1bba5798`` — the SSM select-gate × projection."""
        spec = (
            "mul",
            (
                "unsqueeze",
                ("select", "v430", {"dim": 1, "index": 0}),
                {"dim": -1},
            ),
            (
                "unsqueeze",
                (
                    "select",
                    ("linear", "v463", "v464", "v465"),
                    {"dim": 1, "index": 0},
                ),
                {"dim": -1},
            ),
        )
        domain = [
            {
                "v430": (2, 4, 1),
                "v463": (2, 4, 16),
                "v464": (16, 16),
                "v465": (16,),
            }
        ]
        handlers, report = hs.mint_handlers([spec], domain=domain)
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "gather"
        assert report[0]["tried"] == 1
        assert report[0]["max_err"] < 1e-9
        # the fused order won: read the slice, then project it
        assert (
            hs._gather_view(spec, True)
            == handlers[report[0]["tag"]]["handler_body"][1]
        )
        # and the delivered binding runs the gather body
        h = handlers[report[0]["tag"]]
        env = _env(domain[0])
        out = hs.minted_bindings(handlers)[h["kernel"]](
            env["v430"], env["v463"], env["v464"], env["v465"]
        )
        want = env["v430"].select(1, 0).unsqueeze(
            -1
        ) * torch.nn.functional.linear(
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
                    (
                        "select",
                        ("linear", "v633", "v634", "v635"),
                        {"dim": 1, "index": 7},
                    ),
                    {"dim": 1},
                ),
            ),
        )
        domain = [
            {
                "v13": (2, 16, 4),
                "v419": (2, 16),
                "v633": (2, 8, 16),
                "v634": (4, 16),
                "v635": (4,),
            }
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
                (
                    "select",
                    ("linear", "v605", "v606", "v607"),
                    {"dim": 1, "index": 0},
                ),
                {"dim": -1},
            ),
        )
        domain = [
            {
                "v319": (2, 1),
                "v605": (2, 4, 16),
                "v606": (16, 16),
                "v607": (16,),
            }
        ]
        _handlers, report = hs.mint_handlers([spec], domain=domain)
        assert report[0]["verdict"] == "minted"
        assert report[0]["family"] == "gather"
        assert report[0]["tried"] == 1

    def test_gather_mints_select_times_select(self):
        """``synop_336e34145b`` — select × select, no affine."""
        spec = (
            "mul",
            (
                "unsqueeze",
                ("select", "v430", {"dim": 1, "index": 0}),
                {"dim": -1},
            ),
            (
                "unsqueeze",
                ("select", "v458", {"dim": 1, "index": 0}),
                {"dim": -1},
            ),
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
            (
                "unsqueeze",
                ("select", "G", {"dim": 1, "index": 0}),
                {"dim": -1},
            ),
            (
                "unsqueeze",
                ("select", "H", {"dim": 1, "index": 0}),
                {"dim": -1},
            ),
        )
        handlers, report = hs.mint_handlers(
            [spec], domain=[{"G": (2, 4, 1), "H": (2, 4, 16)}]
        )
        tag = report[0]["tag"]
        rec = json.loads(
            json.dumps(hs.handler_record(tag, handlers[tag]))
        )
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

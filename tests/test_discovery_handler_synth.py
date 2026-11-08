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

class TestDomainShapeGate:
    """A domain entry whose shapes cannot broadcast yields no evidence.

    The kernel-prim arm hit 64 vs 128 at dim 2: one metavar bound at
    two shapes.  Evaluating the spelled body then raises out of torch,
    which used to take the whole mint down.
    """

    COLLISION = (
        "The size of tensor a (64) must match the size of tensor b "
        "(128) at non-singleton dimension 2"
    )

    def test_skips_the_colliding_draw_and_keeps_the_rest(self, monkeypatch):
        calls = {"n": 0}

        def _eval(spec, env, bindings):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError(self.COLLISION)
            return "ref"

        monkeypatch.setattr(hs, "_eval_spec", _eval)
        envs, refs = hs._domain_draws("spec", [{"x": (2,)}], 0, {}, 1)
        assert len(envs) == 1
        assert refs == ["ref"]

    def test_names_an_entry_with_no_drawable_shape(self, monkeypatch):
        def _eval(spec, env, bindings):
            raise RuntimeError(self.COLLISION)

        monkeypatch.setattr(hs, "_eval_spec", _eval)
        with pytest.raises(hs.DomainShapeError, match="no drawable"):
            hs._domain_draws("spec", [{"x": (2,)}], 0, {}, 0)

    def test_other_runtime_errors_still_propagate(self, monkeypatch):
        def _eval(spec, env, bindings):
            raise RuntimeError("boom")

        monkeypatch.setattr(hs, "_eval_spec", _eval)
        with pytest.raises(RuntimeError, match="boom"):
            hs._domain_draws("spec", [{"x": (2,)}], 0, {}, 0)

    def test_a_split_size_mismatch_is_the_same_class(self, monkeypatch):
        """`split` refuses the same blind metavar a broadcast does."""

        def _eval(spec, env, bindings):
            raise RuntimeError(
                "split_with_sizes expects split_sizes to sum exactly "
                "to 192 (input tensor's size at dimension 1), but got "
                "split_sizes=[64, 64]"
            )

        monkeypatch.setattr(hs, "_eval_spec", _eval)
        with pytest.raises(hs.DomainShapeError, match="no drawable"):
            hs._domain_draws("spec", [{"x": (2,)}], 0, {}, 0)

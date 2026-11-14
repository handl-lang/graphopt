"""Tests for ``mlopt_discovery.bounds`` — the typed probe-budget object.

Pins the arch-audit §1.2/§3 fix: one :class:`ProbeBounds` threaded
through ``_case_tail`` → ``_mint_tiers`` / ``MetaArena`` /
``genkernel.profitable`` instead of the scattered
``spec_node_cap``/``key_cap``/``referee``-dict/``ex[...]`` spellings.

* the two spec metrics stay distinct — ``spec_tree_cap`` (spelled
  tree-size, site collectors) vs ``spec_op_cap`` (op-expansion,
  ``MetaArena`` inventory) — and each is measured in its own unit;
* caps still enforced: over-cap sites decline honestly AND are
  counted (``SkipStats``), never silently dropped;
* ``ex["bounds"]`` carries the record; the legacy
  ``ex["_referee"]``/``ex["_spec_node_cap"]`` spellings still
  reconstruct it for older case producers.
"""

import time
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
import torch
import torch.nn as nn
from mlopt_core.ir import Op, TensorType, Var
from mlopt_discovery import model_corpus as mc
from mlopt_discovery import play
from mlopt_discovery.bounds import ProbeBounds, SkipStats


class _Toy(nn.Module):
    """Small elementwise-heavy module for ``_case_tail`` intake."""

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        y = self.lin(x)
        return torch.relu(y) * torch.sigmoid(y) + y


@pytest.fixture(scope="module")
def toy_export():
    """One ``(model, feed, ir, leaves, feeds)`` export triple."""
    torch.manual_seed(0)
    model = _Toy().eval()
    feed = torch.randn(2, 8, dtype=torch.float64)
    ir, leaves, feeds = mc.export_ir(model, feed)
    return model, feed, ir, leaves, feeds


def _shared_chain(depth: int, op: str = "add") -> Op:
    """``op(t, t)`` self-sharing chain — expansion ``2**(d+1) - 1``."""
    t: Any = Var("x", TensorType((2, 2)))
    for _ in range(depth):
        t = Op.make(op, t, t)
    return t


class TestProbeBounds:
    def test_frozen_and_views(self):
        b = ProbeBounds(
            spec_tree_cap=10,
            spec_op_cap=20,
            key_cap=30,
            shape_key_cap=40,
            max_handlers=4,
            time_budget=1.5,
            reps=7,
        )
        with pytest.raises(FrozenInstanceError):
            b.spec_tree_cap = 99  # type: ignore[misc]
        assert b.as_dict() == {
            "spec_tree_cap": 10,
            "spec_op_cap": 20,
            "key_cap": 30,
            "shape_key_cap": 40,
            "max_handlers": 4,
            "time_budget": 1.5,
            "reps": 7,
        }
        assert b.referee_dict() == {
            "max_handlers": 4,
            "time_budget": 1.5,
            "reps": 7,
        }
        assert b.profitable_kwargs(reps_default=20) == {
            "reps": 7,
            "max_handlers": 4,
            "time_budget": 1.5,
        }
        # reps_default fills only when the bound leaves it open
        assert (
            ProbeBounds().profitable_kwargs(reps_default=20)["reps"]
            == 20
        )

    def test_unbounded_default(self):
        b = ProbeBounds()
        assert b.spec_tree_cap is None and b.spec_op_cap is None
        assert b.key_cap is None and b.max_handlers is None


class TestSkipStats:
    def test_dict_protocol_and_merge(self):
        s = SkipStats()
        s["fusion_sites_over_cap"] = (
            s.get("fusion_sites_over_cap", 0) + 1
        )
        s.skip("seq_sites_over_cap", 3)
        s.skip("fusion_sites_over_cap")
        assert s["fusion_sites_over_cap"] == 2
        assert s["seq_sites_over_cap"] == 3
        assert s.total == 5
        t = SkipStats()
        t.merge(s)
        t.merge({"seq_sites_over_cap": 1})
        assert t["fusion_sites_over_cap"] == 2
        assert t["seq_sites_over_cap"] == 4
        assert t.total == 6


class TestExBounds:
    def test_typed_record_wins(self):
        b = ProbeBounds(spec_op_cap=64, spec_tree_cap=32, key_cap=8)
        ex = {"bounds": b, "_spec_node_cap": 999, "_referee": {}}
        assert play._ex_bounds(ex) is b

    def test_legacy_fallback_sets_both_spec_metrics(self):
        # the legacy single cap fed BOTH spec metrics — the fallback
        # reproduces that (each side measures its own unit)
        ex = {
            "_spec_node_cap": 512,
            "_referee": {
                "max_handlers": 16,
                "time_budget": 2.0,
                "reps": 5,
            },
        }
        b = play._ex_bounds(ex)
        assert b.spec_tree_cap == 512
        assert b.spec_op_cap == 512
        assert b.max_handlers == 16
        assert b.time_budget == 2.0
        assert b.reps == 5

    def test_empty_ex_is_unbounded(self):
        b = play._ex_bounds({})
        assert b == ProbeBounds()


class TestMintTiersBounds:
    """``_mint_tiers(bounds=…)`` enforces the tree-size cap + counts."""

    def _covered(self):
        from mlopt_discovery import lawdata
        from mlopt_discovery.meta_arena import _canon_concrete

        return {
            _canon_concrete(h["pattern"])
            for h in lawdata.HANDLERS.values()
        }

    def test_bounds_supplies_spec_tree_cap(self):
        term = Op.make("relu", _shared_chain(14, "add"))
        stats = SkipStats()
        genh = play._mint_tiers(
            term,
            self._covered(),
            bounds=ProbeBounds(spec_tree_cap=64),
            stats=stats,
        )
        assert isinstance(genh, dict)
        # the over-cap sites declined AND were counted — the
        # fusion/reduction collectors report like the scan tiers
        assert stats.get("fusion_sites_over_cap", 0) > 0
        assert stats.total > 0

    def test_explicit_spec_cap_beats_bounds(self):
        term = Op.make("relu", _shared_chain(14, "add"))
        stats = SkipStats()
        play._mint_tiers(
            term,
            self._covered(),
            spec_cap=64,
            bounds=ProbeBounds(spec_tree_cap=10_000),
            stats=stats,
        )
        # explicit spec_cap=64 binds (over-cap counted); the looser
        # bounds value does not leak through
        assert stats.get("fusion_sites_over_cap", 0) > 0

    def test_fusion_and_reduction_collectors_count_skips(self):
        # reduction-headed site over a shared chain: spec_cap binds
        # the spelled tree-size and the decline is counted
        term = Op.make(
            "mean",
            Op.make("mul", _shared_chain(12, "add"), _shared_chain(12)),
            dim=-1,
        )
        stats: dict = {}
        sites = play._reduction_sites(term, spec_cap=64, stats=stats)
        assert sites == []
        assert stats.get("reduction_sites_over_cap", 0) > 0
        # uncapped: the same term still declines on the shape gate or
        # mints — never counted as over-cap
        stats2: dict = {}
        play._reduction_sites(term, stats=stats2)
        assert "reduction_sites_over_cap" not in stats2

    def test_unbounded_still_mints(self):
        from mlopt_core.ir import TensorType, Var

        x = Var("x", TensorType((4, 4)))
        y = Op.make("matmul", x, x)
        term = Op.make("mul", y, Op.make("relu", y))
        stats = SkipStats()
        genh = play._mint_tiers(
            term, self._covered(), bounds=ProbeBounds(), stats=stats
        )
        assert stats.total == 0
        # the pointwise fusion site minted a gen_* handler
        assert any(k.startswith("gen_") for k in genh)


class TestProbeKnobsBounds:
    def test_bounds_field_and_legacy_spellings(self):
        knobs = mc._probe_knobs(
            _shared_chain(20),
            spec_node_cap=None,
            key_cap=None,
            max_claims=None,
            claim_time_budget=None,
        )
        b = knobs["bounds"]
        assert isinstance(b, ProbeBounds)
        assert knobs["large_term"] is True
        # auto defaults engage on BOTH spec metrics at the shared
        # value, each measured in its own unit
        assert b.spec_tree_cap == mc._AUTO_SPEC_NODE_CAP
        assert b.spec_op_cap == mc._AUTO_SPEC_NODE_CAP
        assert b.key_cap == mc._AUTO_KEY_CAP
        assert b.max_handlers == mc._AUTO_MAX_CLAIMS
        assert b.time_budget == mc._AUTO_CLAIM_BUDGET
        # legacy keys preserved for existing consumers
        assert knobs["spec_node_cap"] == mc._AUTO_SPEC_NODE_CAP
        assert knobs["referee"] == {
            "max_handlers": mc._AUTO_MAX_CLAIMS,
            "time_budget": mc._AUTO_CLAIM_BUDGET,
        }

    def test_explicit_bounds_win_over_kwargs(self):
        b = ProbeBounds(spec_tree_cap=7, spec_op_cap=9, key_cap=11)
        knobs = mc._probe_knobs(
            _shared_chain(20),
            spec_node_cap=64,
            key_cap=100,
            max_claims=2,
            claim_time_budget=0.5,
            bounds=b,
        )
        assert knobs["bounds"] is b
        assert knobs["spec_node_cap"] == 9  # spec_op_cap
        assert knobs["key_cap"] == 11

    def test_small_term_unbounded(self):
        knobs = mc._probe_knobs(
            _shared_chain(3),
            spec_node_cap=None,
            key_cap=None,
            max_claims=None,
            claim_time_budget=None,
        )
        b = knobs["bounds"]
        assert knobs["large_term"] is False
        assert b.spec_tree_cap is None and b.spec_op_cap is None
        assert b.key_cap is None
        assert b.max_handlers is None and b.time_budget is None


class TestCaseTailBounds:
    """``_case_tail`` seats ``ex["bounds"]`` and the gates read it."""

    def test_case_tail_seats_typed_bounds(self, toy_export):
        model, _feed, ir, leaves, feeds = toy_export
        bounds = ProbeBounds(
            spec_tree_cap=4096,
            spec_op_cap=2048,
            max_handlers=2,
            time_budget=5.0,
            reps=2,
        )
        t0 = time.perf_counter()
        case = mc._case_tail(
            ir,
            leaves,
            feeds,
            model,
            "t:bounds",
            gen_reps=2,
            bounds=bounds,
        )
        assert time.perf_counter() - t0 < 120  # sanity, not a gate
        ex = case[5]
        assert ex["bounds"] is bounds
        # legacy spellings stay consistent: _spec_node_cap is the
        # OP-EXPANSION cap (2048), NOT the tree-size cap (4096)
        assert ex["_spec_node_cap"] == 2048
        assert ex["_referee"]["max_handlers"] == 2
        assert ex["_referee"]["time_budget"] == 5.0
        assert ex["_referee"]["reps"] == 2
        assert isinstance(ex["site_stats"], SkipStats)
        # the MetaArena the board builds reads spec_op_cap
        board = play._gen_board(case)
        assert board._spec_op_cap == 2048

    def test_legacy_kwargs_still_resolve(self, toy_export):
        model, _feed, ir, leaves, feeds = toy_export
        case = mc._case_tail(
            ir,
            leaves,
            feeds,
            model,
            "t:legacy",
            gen_reps=2,
            referee={"max_handlers": 3, "time_budget": 4.0},
            spec_node_cap=111,
        )
        ex = case[5]
        b = play._ex_bounds(ex)
        # legacy spec_node_cap fed BOTH spec metrics
        assert b.spec_tree_cap == 111 and b.spec_op_cap == 111
        assert b.max_handlers == 3 and b.time_budget == 4.0


class TestProfitableBounds:
    """``genkernel.profitable(bounds=…)`` supplies referee defaults."""

    def _handlers(self, n: int) -> dict:
        return {
            f"h{i:02d}": {"pattern": ("add", "X1", "X2")}
            for i in range(n)
        }

    def test_bounds_max_handlers_caps_prefix(self, monkeypatch):
        from mlopt_discovery import genkernel as gk

        monkeypatch.setattr(gk, "_handler_env", lambda h, t: {})
        monkeypatch.setattr(
            gk,
            "_referee_kernel",
            lambda kf, name, h, vals, kerneled: (
                (lambda: torch.ones(2)),
                vals,
            ),
        )
        monkeypatch.setattr(
            gk, "_kernel_pays", lambda *a, **k: (1.0, 0.5, a[4])
        )
        x = Var("x", TensorType((2, 2)))
        term = Op.make("add", x, x)
        keep = gk.profitable(
            self._handlers(10),
            term,
            {},
            reps=2,
            kernel_for=lambda n, h, v: None,
            bounds=ProbeBounds(max_handlers=3),
        )
        assert sorted(keep) == ["h00", "h01", "h02"]

    def test_explicit_kwarg_beats_bounds(self, monkeypatch):
        from mlopt_discovery import genkernel as gk

        monkeypatch.setattr(gk, "_handler_env", lambda h, t: {})
        monkeypatch.setattr(
            gk,
            "_referee_kernel",
            lambda kf, name, h, vals, kerneled: (
                (lambda: torch.ones(2)),
                vals,
            ),
        )
        monkeypatch.setattr(
            gk, "_kernel_pays", lambda *a, **k: (1.0, 0.5, a[4])
        )
        x = Var("x", TensorType((2, 2)))
        term = Op.make("add", x, x)
        keep = gk.profitable(
            self._handlers(10),
            term,
            {},
            reps=2,
            kernel_for=lambda n, h, v: None,
            max_handlers=5,
            bounds=ProbeBounds(max_handlers=3),
        )
        assert len(keep) == 5

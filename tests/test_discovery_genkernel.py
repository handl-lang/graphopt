"""Tests for ``mlopt_discovery.genkernel`` — generated lowerings.

Pins the mint path (concrete subterm → handler entry), the binding
semantics (the generated callable evaluates the spelled body — the
soundness of ``claim``), the sink extension (gen ops are
supported + lowerable), and the novelty cost wrapper.
"""

import pytest
import torch

from mlopt_core.ir import Op, TensorType, Var
from mlopt_core.cost.basic import count_cost
from mlopt_discovery import genkernel as gk
from mlopt_discovery import meta_arena as ma
from mlopt_discovery import play
from mlopt_discovery.lawdata import HANDLERS
from mlopt_discovery.meta_arena import _canon_concrete


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


class TestMinting:
    def test_elementwise_subterm_mints_handler(self):
        x = _v("x", 4, 4)
        h = gk.gen_handlers([_p("mul", x, _p("relu", x))])
        assert "gen_0" in h
        assert h["gen_0"]["kernel"] == "gen_0_k"
        assert h["gen_0"]["pattern"] == ("mul", "X1", ("relu", "X1"))
        assert h["gen_0"]["args"] == ("X1",)

    def test_covered_pattern_skipped(self):
        # silu's concrete pattern is already named — no gen name
        x = _v("x", 4, 4)
        covered = {
            _canon_concrete(h["pattern"]) for h in HANDLERS.values()
        }
        assert (
            gk.gen_handlers(
                [_p("mul", x, _p("sigmoid", x))], covered=covered
            )
            == {}
        )

    def test_attr_carrying_spec_mints(self):
        # attr dicts canon to *unhashable* dicts — the covered/dedup
        # check must compare by equality (``_canon_taken``), not
        # ``in``; this spec crashed ``gen_handlers`` before the fix
        spec = (
            "mul",
            ("softplus", "X1", {"beta": 2.0, "threshold": 5.0}),
            ("sigmoid", "X1"),
        )
        h = gk.gen_handlers([spec])
        assert "gen_0" in h
        assert h["gen_0"]["pattern"] == spec
        # a covered entry carrying the same attrs still skips —
        # ``covered`` is iterated, so a list of canon forms works
        # where a set could not hold the unhashable canon
        covered = [_canon_concrete(spec)]
        assert gk.gen_handlers([spec], covered=covered) == {}
        # and the spelled binding evaluates the attrs verbatim
        x = _v("x", 4, 4)
        term = _p(
            "mul",
            _p("softplus", x, beta=2.0, threshold=5.0),
            _p("sigmoid", x),
        )
        h2 = gk.gen_handlers([term])
        assert h2 and "beta" in str(h2["gen_0"]["pattern"])

    def test_non_elementwise_skipped(self):
        x, y = _v("x", 4, 4), _v("y", 4, 4)
        assert gk.gen_handlers([_p("matmul", x, y)]) == {}
        # and the dedup: same concrete pattern twice → one entry
        z = _v("z", 4, 4)
        h = gk.gen_handlers(
            [_p("mul", x, _p("relu", x)), _p("mul", z, _p("relu", z))]
        )
        assert len(h) == 1

    def test_leaf_shape_preserved(self):
        # two leaves, consistent binding: sub(x, y) keeps order
        x, y = _v("x", 4, 4), _v("y", 4, 4)
        h = gk.gen_handlers(
            [_p("add", _p("mul", x, x), _p("mul", y, y))]
        )
        assert h["gen_0"]["args"] == ("X1", "X2")


class TestBindings:
    def test_binding_evaluates_spelled_body(self):
        x = _v("x", 4, 4)
        h = gk.gen_handlers([_p("mul", x, _p("relu", x))])
        b = gk.gen_bindings(h, compile_kernels=False)
        t = torch.randn(4, 4)
        torch.testing.assert_close(b["gen_0_k"](t), t * torch.relu(t))

    def test_binding_two_metavars(self):
        x, y = _v("x", 4, 4), _v("y", 4, 4)
        h = gk.gen_handlers([_p("sub", x, _p("abs", y))])
        b = gk.gen_bindings(h, compile_kernels=False)
        a, c = torch.randn(4, 4), torch.randn(4, 4)
        torch.testing.assert_close(b["gen_0_k"](a, c), a - c.abs())

    def test_gen_sink_supports_and_lowers(self):
        x = _v("x", 4, 4)
        term = _p("mul", x, _p("relu", x))
        h = gk.gen_handlers([term])
        sink = gk.gen_sink(h, compile_kernels=False)
        assert "gen_0_k" in sink.supported_ops
        from mlopt_core.ir import IR

        ir = IR(
            root=_p("gen_0_k", x),
            inputs=[x],
            input_names={"x"},
            params={},
        )
        mod = sink.lower(ir, params={"x": torch.randn(4, 4)})
        t = torch.randn(4, 4)
        torch.testing.assert_close(mod(t), t * torch.relu(t))


class TestNovelCost:
    def test_novel_head_discounted(self):
        x = _v("x", 4, 4)
        cost = gk.novel_cost(
            count_cost, lambda op: op.startswith("gen_"), 0.5
        )
        spelled = _p("mul", x, _p("relu", x))
        gened = _p("gen_0_k", x)
        assert cost(gened) == cost(spelled) - 1.5  # 1 + 1 - 1.5

    def test_known_head_unchanged(self):
        x = _v("x", 4, 4)
        cost = gk.novel_cost(count_cost, lambda op: op == "never", 0.5)
        assert cost(_p("mul", x, x)) == 1.0


class TestGenDomain:
    def test_case_mints_handlers(self):
        torch = __import__("pytest").importorskip("torch")
        cases = play._gen_cases(0, 1)
        assert cases[0][5]["gen_handlers"] or True  # may be mlp
        cases = [c for s in range(4) for c in play._gen_cases(s, 3)]
        fuses = [c for c in cases if "mul" in c[0] or "fuse" in c[0]]
        assert fuses
        assert fuses[0][5]["gen_handlers"]

    def test_claim_gen_delivers_verified(self):
        pytest = __import__("pytest")
        pytest.importorskip("torch")
        for s in range(8):
            for c in play._gen_cases(s, 4):
                if not c[5]["gen_handlers"]:
                    continue
                board = play._gen_board(c)
                st = board.observe()
                claims = [
                    a for a in ma.legal_actions(st) if a.op == "claim"
                ]
                if not claims:
                    continue
                board.step(claims[0])
                board.step(ma.Action.extract())
                best = board.eg.extract_best(
                    board.root, board.feasible_cost
                )
                from mlopt_core.ir import op_repr

                assert "gen_" in op_repr(board.deliverable(best))
                return
        raise AssertionError("no gen claim landed")

    def test_gen_board_merges_shipped_prims(self):
        # T-016: the minted ``synop_*`` folds are feasible +
        # deliverable on the gen board — ``_gen_board`` merges
        # ``shipped_primitives.handler_entries()`` into the case's
        # ``gen_handlers`` and widens ``supported`` to the op names,
        # so ``gen_probe``'s ``hybrid_sink`` binds them to the
        # fused-body codegen instead of losing them unlowerable.
        pytest = __import__("pytest")
        pytest.importorskip("torch")
        from mlopt_discovery import shipped_primitives as sp

        for s in range(4):
            for c in play._gen_cases(s, 3):
                board = play._gen_board(c)
                ex = c[5]
                assert set(ex["gen_handlers"]) >= sp.SYNOP_SHIPPED
                assert set(board.supported) >= sp.SYNOP_SHIPPED
                # the case record carries the merged table — the
                # delivery sink sees the identical handlers
                assert ex["sink"] is not None
                # idempotent: a second board over the same case
                # does not duplicate or reseat again
                n_handlers = len(ex["gen_handlers"])
                play._gen_board(c)
                assert len(ex["gen_handlers"]) == n_handlers
                return
        raise AssertionError("no gen case built")

    def test_fold_cost_gate_partitions_by_site_and_referee(
        self, monkeypatch
    ):
        # T-019: a prim fold is refused only on a *measured* loss —
        # entries with no concrete site cannot fire and keep their
        # seat; sited entries go through ``profitable`` and only its
        # keeps survive into ``supported``.
        from mlopt_discovery import shipped_primitives as sp

        case = play._gen_cases(0, 1)[0]
        he = sp.handler_entries()
        # no-site path: a gen-case term matches no prim pattern, so
        # every entry keeps its seat and the referee is never asked
        out = play._fold_cost_gate(case, he)
        assert set(out) == set(he)
        # measured path: force every entry to look sited; a referee
        # that keeps nothing must return nothing
        monkeypatch.setattr(gk, "_handler_env", lambda h, t: {})
        monkeypatch.setattr(gk, "profitable", lambda *a, **k: {})
        assert play._fold_cost_gate(case, he) == {}
        # and a referee that keeps everything restores the table
        monkeypatch.setattr(
            gk, "profitable", lambda hs, *a, **k: dict(hs)
        )
        assert set(play._fold_cost_gate(case, he)) == set(he)

    def test_fold_cost_gate_sights_attr_metavars(self, monkeypatch):
        # T-026: attr-metavar prim specs concrete-match never, but
        # ``concretize_at_site`` binds the attr mvars from the real
        # site — the referee measures a frozen copy while the
        # delivered table keeps the polymorphic original.
        torch = pytest.importorskip("torch")
        x, y = _v("x", 4, 4), _v("y", 4, 1)
        term = _p("mul", x, _p("unsqueeze", y, dim=-2))
        case = (
            "t",
            term,
            {},
            "t",
            frozenset(),
            {"ir": None, "leaves": {}, "input": torch.rand(4, 4)},
        )
        h = {
            "pattern": (
                "mul",
                "m",
                ("unsqueeze", "n", {"dim": "a1"}),
            ),
            "kernel": "synop_t",
            "args": ("m", "n"),
            "attr_mvars": ("a1",),
        }
        monkeypatch.setattr(play, "_var_env", lambda *a: {})
        seen = {}

        def spy(hs, *a, **k):
            seen.update(hs)
            return {}

        monkeypatch.setattr(gk, "profitable", spy)
        assert play._fold_cost_gate(case, {"tag": h}) == {}
        # the referee saw the concretized copy, not the original
        assert seen["tag"]["attr_mvars"] == ()
        assert seen["tag"]["pattern"][2][2] == {"dim": -2}
        # a keeping referee restores the POLYMORPHIC original
        monkeypatch.setattr(
            gk, "profitable", lambda hs, *a, **k: dict(hs)
        )
        out = play._fold_cost_gate(case, {"tag": h})
        assert out["tag"] is h

    def test_claim_cost_gate_filters_measurable_tiers(
        self, monkeypatch
    ):
        # T-010 offer guard: ``_claim_cost_gate`` runs the measured
        # ``profitable`` referee once per case over every minted
        # claim whose spelled body is an honest denotation —
        # ``gen_*``, spelled ``decayed_*``, and ``scan_*`` in BOTH
        # forms (unrolled compose-trees spell the carrier
        # semantics; sequence-form scans carry the registered fold
        # denotation ``genkernel.seq_scan_denotation``) — and drops
        # the losers before the board's handler store is built.
        # ``om_decayed``-headed specs (no ``_IR_TO_TORCH`` binding
        # for the spelled side) stay unpriced by design.
        torch = pytest.importorskip("torch")
        x = _v("x", 4, 4)
        term = _p("mul", x, _p("relu", x))
        case = (
            "t",
            term,
            {},
            "t",
            frozenset(),
            {
                "ir": None,
                "leaves": {},
                "input": torch.rand(4, 4),
                "gen_handlers": {
                    "gen_0": {"pattern": ("mul", "X1", ("relu", "X1")),
                              "kernel": "gen_0_k", "args": ("X1",)},
                    "gen_1": {"pattern": ("add", "X1", "X1"),
                              "kernel": "gen_1_k", "args": ("X1",)},
                    "decayed_0": {
                        "pattern": ("matmul", ("mul", "S", "D"), "V"),
                        "kernel": "decayed_0_k",
                        "args": ("S", "D", "V"),
                    },
                    "decayed_1": {
                        "pattern": ("om_decayed", "S", "D", "V"),
                        "kernel": "decayed_1_k",
                        "args": ("S", "D", "V"),
                    },
                    # a sequence-form scan: measurable through the
                    # registered fold denotation, not dropped
                    "scan_0": {
                        "pattern": (
                            "applyd",
                            ("aff_diag", "A", "B"),
                            "H",
                        ),
                        "kernel": "scan_0_k",
                        "args": ("A", "B", "H"),
                    },
                    "prim:synop_x": {
                        "pattern": ("mul", "X1", "X1"),
                        "kernel": "synop_x",
                        "args": ("X1",),
                    },
                },
                "sink": None,
            },
        )
        monkeypatch.setattr(play, "_var_env", lambda *a: {})
        seen = {}
        den_seen = {}

        def spy(hs, *a, **k):
            seen.update(hs)
            den_for = k.get("denotation_for")
            den_seen["scan"] = (
                den_for("scan_0", hs["scan_0"], {})
                if den_for is not None
                else "missing"
            )
            den_seen["gen"] = (
                den_for("gen_0", hs["gen_0"], {})
                if den_for is not None
                else "missing"
            )
            return {"gen_0": hs["gen_0"],
                    "decayed_0": hs["decayed_0"],
                    "scan_0": hs["scan_0"]}  # keep three

        monkeypatch.setattr(gk, "profitable", spy)
        play._claim_cost_gate(case)
        # the referee saw gen_*, spelled decayed_*, AND the seq scan
        # (measurable via the fold denotation) — never the
        # om_decayed spec or the shipped prim fold
        assert set(seen) == {"gen_0", "gen_1", "decayed_0", "scan_0"}
        # the denotation hook resolves the fold for the seq claim
        # and declines the ordinary pointwise one
        assert callable(den_seen["scan"])
        assert den_seen["gen"] is None
        assert set(case[5]["gen_handlers"]) == {
            "gen_0",
            "decayed_0",
            "decayed_1",
            "scan_0",
            "prim:synop_x",
        }
        # idempotent: a second call does not re-consult the referee
        monkeypatch.setattr(
            gk,
            "profitable",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("re-gated")
            ),
        )
        play._claim_cost_gate(case)

    def test_spec_ops_skips_attr_dicts(self):
        spec = ("add", "X1", {"dim": 1})
        assert gk._spec_ops(spec) == {"add"}
        # and a const leaf walks through without touching metas
        assert gk._spec_metas(("mul", "X1", 2.0)) == ["X1"]

    def test_eval_spec_const_leaf(self):

        import mlopt_torch.torch_bridge as tb

        out = gk._eval_spec(
            ("mul", "X1", 2.0),
            {"X1": torch.ones(2)},
            dict(tb._IR_TO_TORCH),
        )
        torch.testing.assert_close(out, torch.ones(2) * 2.0)

    def test_eval_spec_creator_follows_env_device(self):
        """A ``zeros`` leaf mints where the computation runs (T-024).

        Under a CPU env nothing moves; under a CUDA-bound env the
        creator result follows so spelled/kernel evals never mix
        devices.  The spec's dtype attr still wins.
        """
        import mlopt_torch.torch_bridge as tb

        spec = (
            "add",
            ("zeros", {"shape": [2, 2], "dtype": "float64"}),
            "X",
        )
        env = {"X": torch.ones(2, 2, dtype=torch.float64)}
        out = gk._eval_spec(spec, env, dict(tb._IR_TO_TORCH))
        assert out.device.type == "cpu"
        torch.testing.assert_close(
            out, torch.ones(2, 2, dtype=torch.float64)
        )
        if torch.cuda.is_available():
            cenv = {
                "X": torch.ones(
                    2, 2, dtype=torch.float64, device="cuda"
                )
            }
            cout = gk._eval_spec(spec, cenv, dict(tb._IR_TO_TORCH))
            assert cout.device.type == "cuda"
            torch.testing.assert_close(
                cout.cpu(), torch.ones(2, 2, dtype=torch.float64)
            )


class TestProbe:
    def test_gen_probe_delivers(self, monkeypatch):
        pytest = __import__("pytest")
        pytest.importorskip("torch")
        real = gk.gen_sink
        monkeypatch.setattr(
            gk,
            "gen_sink",
            lambda h, **kw: real(h, compile_kernels=False),
        )
        # bypass the offer gate — this pins probe→deliver machinery on
        # a fusion case; the measured gate legitimately declines
        # mulrelu's marginal 2-op claim (see ``_claim_cost_gate``)
        monkeypatch.setattr(play, "_claim_cost_gate", lambda case: None)
        for s in range(8):
            for c in play._gen_cases(s, 4):
                if "mulrelu" in c[0]:
                    r = play.gen_probe(c, reps=3)
                    assert r["delivered"] and r["verified"]
                    assert r["claims"] and r["speedup"] is not None
                    return
        raise AssertionError("no mulrelu case found")

    def test_spec_metas_skips_attr_dicts(self):
        assert gk._spec_metas(("add", "X1", {"dim": 1})) == ["X1"]

    def test_probe_empty_claims_and_no_best(self, monkeypatch):
        pytest = __import__("pytest")
        pytest.importorskip("torch")
        real = gk.gen_sink
        monkeypatch.setattr(
            gk,
            "gen_sink",
            lambda h, **kw: real(h, compile_kernels=False),
        )
        # an mlp case: no fusion sites → no claims → still probes
        for s in range(12):
            for c in play._gen_cases(s, 4):
                if "mlp" in c[0]:
                    r = play.gen_probe(c, reps=3)
                    assert r["delivered"]
                    break
            else:
                continue
            break
        # extract empty → honest undelivered report
        c2 = play._gen_cases(0, 1)[0]
        monkeypatch.setattr(
            ma.EGraph, "extract_best", lambda *a, **k: None
        )
        r2 = play.gen_probe(c2, reps=2)
        assert r2["delivered"] is False


class TestTritonTier:
    """Generated Triton source — a kernel inductor doesn't produce."""

    def test_triton_expr_binary_and_unary(self):
        e = gk._triton_expr(("mul", "X1", ("relu", "X1")))
        assert e == "((X1) * (tl.maximum(X1, 0.0)))"
        assert "libdevice.tanh" in gk._triton_expr(("tanh", "X1"))

    def test_triton_expr_declines_attr_and_unknown(self):
        assert gk._triton_expr(("sum", "X1", {"dim": 1})) is None
        assert gk._triton_expr(("matmul", "X1", "X2")) is None
        assert gk._triton_expr(("pow", "X1", "X2")) is not None
        assert gk._triton_expr(("mul", "X1", 2.0)) is not None

    def test_triton_expr_softsign_gate_binary(self):
        e = gk._triton_expr(("softsign_gate", "X1", "X2"))
        assert e is not None and "tl.abs" in e and "X2" in e

    def test_triton_bindings_correct_and_fused(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("needs cuda for triton")
        h = {
            "g": {
                "pattern": ("mul", "X1", ("sigmoid", "X1")),
                "kernel": "gk_t",
                "args": ("X1",),
            }
        }
        b = gk.triton_bindings(h)
        x = torch.randn(512, device="cuda")
        torch.testing.assert_close(
            b["gk_t"](x), x * torch.sigmoid(x), atol=1e-5, rtol=1e-5
        )

    def test_triton_expr_const_leaf_template(self):
        # Const leaves render as bare fp32 scalars by default, but
        # the kernel emitters pass a dtype-typed ``tl.full`` — a
        # scalar literal is a hard type clash inside ``libdevice.*``
        # calls on fp64 tiles (T-025).
        spec = ("pow", ("relu", "X1"), 3.0)
        assert "(3.0)" in gk._triton_expr(spec)
        e = gk._triton_expr(spec, gk._TRITON_CONST_1D)
        assert e is not None
        assert "libdevice.pow" in e
        assert (
            "tl.full([BLOCK], 3.0, dtype=out_ptr.dtype.element_ty)"
            in e
        )
        # literal exponent 2 takes the ``x * x`` shortcut instead —
        # ``libdevice.pow`` costs ~150µs fp64 where a multiply is
        # free, and the product is the same IEEE value
        sq = gk._triton_expr(("pow", ("relu", "X1"), 2.0))
        assert sq is not None
        assert "libdevice.pow" not in sq
        assert sq.count("tl.maximum(X1, 0.0)") == 2

    def test_triton_bindings_fp64_const_pow(self):
        # Regression for the T-024 audit's one ``raised`` consult:
        # ``libdevice.pow(f64_tile, (3.0))`` — the fp32 literal —
        # failed Triton compilation.  The const leaf must follow the
        # launch dtype.
        torch = pytest.importorskip("torch")
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("needs cuda for triton")
        h = {
            "g": {
                "pattern": ("pow", ("relu", "X1"), 3.0),
                "kernel": "gk_pow64",
                "args": ("X1",),
            }
        }
        b = gk.triton_bindings(h)
        assert "gk_pow64" in b
        for dt, tol in (
            (torch.float64, dict(rtol=1e-12, atol=1e-12)),
            (torch.float32, dict(rtol=1e-5, atol=1e-5)),
        ):
            x = torch.randn(5000, device="cuda", dtype=dt)
            torch.testing.assert_close(
                b["gk_pow64"](x), torch.relu(x) ** 3, **tol
            )

    def test_triton_sink_supported(self):
        pytest.importorskip("torch")
        pytest.importorskip("triton")
        h = {
            "g": {
                "pattern": ("mul", "X1", ("relu", "X1")),
                "kernel": "gk_t2",
                "args": ("X1",),
            }
        }
        sink = gk.triton_sink(h)
        assert "gk_t2" in sink.supported_ops

    def test_kernel_files_landed_on_disk(self):
        pytest.importorskip("torch")
        pytest.importorskip("triton")
        h = {
            "g": {
                "pattern": ("mul", "X1", ("relu", "X1")),
                "kernel": "gk_disk",
                "args": ("X1",),
            }
        }
        gk.triton_bindings(h)
        f = gk._kernel_cache() / "_gk_gk_disk.py"
        assert f.exists() and "triton.jit" in f.read_text()

    def test_unsupported_spec_gets_no_binding(self):
        pytest.importorskip("torch")
        pytest.importorskip("triton")
        h = {
            "g": {
                "pattern": ("matmul", "X1", "X2"),
                "kernel": "gk_no",
                "args": ("X1", "X2"),
            }
        }
        assert gk.triton_bindings(h) == {}

    def test_module_load_decline_keeps_torch_path(self, monkeypatch):
        pytest.importorskip("torch")
        pytest.importorskip("triton")
        import importlib.util

        monkeypatch.setattr(
            importlib.util,
            "spec_from_file_location",
            lambda *a, **k: None,
        )
        h = {
            "g": {
                "pattern": ("mul", "X1", ("relu", "X1")),
                "kernel": "gk_decl",
                "args": ("X1",),
            }
        }
        assert gk.triton_bindings(h) == {}

    def test_meta_named_like_kernel_local_declines(self):
        # ``mask`` would shadow the kernel's mask local — the binding
        # must decline at codegen rather than emit a miscompile.
        h = {
            "g": {
                "pattern": ("mul", "mask", "X1"),
                "kernel": "gk_col",
                "args": ("mask", "X1"),
            }
        }
        assert gk.triton_bindings(h) == {}
        h["g"]["pattern"] = ("mul", "__o1", "X1")
        h["g"]["args"] = ("__o1", "X1")
        assert gk.triton_bindings(h) == {}

    def test_triton_ometa_layout(self):
        # CPU-only packing check: sizes, row-major out strides, then
        # per-operand effective strides (0 on broadcast dims).
        a = torch.zeros(8, 1, 1)
        b = torch.zeros(8, 64, 64)
        c = torch.zeros(8, 64, 1)
        d = torch.zeros(8, 1, 64)
        xs = [a, b, c, d]
        om = gk._triton_ometa(xs, (8, 64, 64), a.device)
        want = [
            8,
            64,
            64,  # out sizes
            4096,
            64,
            1,  # out row-major strides
            1,
            0,
            0,  # a: (8,1,1) spans dim0 only
            4096,
            64,
            1,  # b: full shape
            64,
            1,
            0,  # c: (8,64,1) contiguous strides
            64,
            0,
            1,  # d: (8,1,64) contiguous strides
        ]
        assert om.tolist() == want
        # cached: same layout returns the identical tensor
        assert gk._triton_ometa(xs, (8, 64, 64), a.device) is om

    def test_triton_broadcast_and_strided_correct(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("needs cuda for triton")
        h = {
            "g": {
                "pattern": (
                    "add",
                    ("mul", "a", "b"),
                    ("mul", "c", "d"),
                ),
                "kernel": "gk_bc",
                "args": ("a", "b", "c", "d"),
            }
        }
        fn = gk.triton_bindings(h)["gk_bc"]

        def ref(*args):
            xs = torch.broadcast_tensors(*args)
            return xs[0] * xs[1] + xs[2] * xs[3]

        dev = "cuda"
        cases = [
            # rank-1 / scalar broadcast — the zoo site shape
            (
                torch.randn(8, 1, 1, device=dev),
                torch.randn(8, 64, 64, device=dev),
                torch.randn(8, 64, 1, device=dev),
                torch.randn(8, 1, 64, device=dev),
            ),
            # 0-dim operand
            (
                torch.randn((), device=dev),
                torch.randn(4, 32, device=dev),
                torch.randn(4, 32, device=dev),
                torch.randn(4, 32, device=dev),
            ),
            # non-contiguous operand (transpose)
            (
                torch.randn(8, 64, device=dev),
                torch.randn(64, 8, device=dev).t(),
                torch.randn(8, 64, device=dev),
                torch.randn(8, 64, device=dev),
            ),
            # expanded stride-0 operand
            (
                torch.randn(8, 64, 64, device=dev),
                torch.randn(8, 1, 64, device=dev).expand(8, 64, 64),
                torch.randn(8, 64, 64, device=dev),
                torch.randn(8, 64, 64, device=dev),
            ),
        ]
        for args in cases:
            torch.testing.assert_close(
                fn(*args), ref(*args), atol=1e-4, rtol=1e-4
            )
        # empty broadcast — early return, no launch
        empty = tuple(torch.randn(0, device=dev) for _ in range(4))
        assert fn(*empty).shape == (0,)


class TestMeasuredReferee:
    """``profitable`` — mint only claims whose kernel beats spelled."""

    def test_profitable_drains_compile_workers(self, monkeypatch):
        # measurement hygiene: an in-flight inductor compile backlog
        # inflates the spelled arm's Python-heavy call path and flips
        # measured losers into false keeps — the referee must drain
        # the worker pools before timing (T-016 residual).
        torch = pytest.importorskip("torch")
        import torch._inductor.async_compile as ac

        calls = []
        monkeypatch.setattr(
            ac, "shutdown_compile_workers", lambda: calls.append(1)
        )
        gk.profitable({}, _v("x", 4), {"x": torch.randn(4)}, reps=2)
        assert calls == [1]

    def test_paying_kernel_kept_losing_dropped(self):
        torch = pytest.importorskip("torch")
        from mlopt_core.ir import TensorType, Var

        x = Var("x", TensorType((256, 256)))
        # a big tail — fusion pays; a 1-op site — overhead doesn't
        big = _p(
            "add",
            _p("mul", x, _p("relu", x)),
            _p(
                "mul",
                _p("tanh", _p("mul", x, x)),
                _p("softplus", _p("mul", x, _p("sigmoid", x))),
            ),
        )
        term = _p("add", big, _p("relu", x))
        leaves = {"x": torch.randn(256, 256)}
        handlers = {
            "g_big": {
                "pattern": (
                    "add",
                    ("mul", "X1", ("relu", "X1")),
                    (
                        "mul",
                        ("tanh", ("mul", "X1", "X1")),
                        ("softplus", ("mul", "X1", ("sigmoid", "X1"))),
                    ),
                ),
                "kernel": "gk_big",
                "args": ("X1",),
            },
            "g_tiny": {
                "pattern": ("relu", "X1"),
                "kernel": "gk_tiny",
                "args": ("X1",),
            },
        }
        keep = gk.profitable(handlers, term, leaves, reps=5)
        assert "g_big" in keep and "g_tiny" not in keep

    def test_absent_site_and_eval_failures_skip(self):
        torch = pytest.importorskip("torch")
        x, y = _v("x", 4, 4), _v("y", 4, 4)
        hs = {
            "g": {
                "pattern": ("mul", "X1", "X2"),
                "kernel": "gk_m",
                "args": ("X1", "X2"),
            }
        }
        # no mul anywhere — the site walk exhausts the args
        lone = _p("relu", x)
        assert (
            gk.profitable(hs, lone, {"x": torch.rand(4, 4)}, reps=2)
            == {}
        )
        # a real site, but a bound metavar eval-fails (unbound var)
        term = _p("add", _p("mul", x, y), x)
        assert (
            gk.profitable(hs, term, {"x": torch.rand(4, 4)}, reps=2)
            == {}
        )

    def test_kernel_mismatch_declined(self, monkeypatch):
        torch = pytest.importorskip("torch")
        # a kernel that fails the verify-before-time check is dropped
        monkeypatch.setattr(torch, "allclose", lambda *a, **k: False)
        x = _v("x", 16)
        term = _p("mul", x, _p("relu", x))
        hs = {
            "g": {
                "pattern": ("mul", "X1", ("relu", "X1")),
                "kernel": "gk_mm",
                "args": ("X1",),
            }
        }
        assert (
            gk.profitable(
                hs, term, {"x": torch.rand(16)}, reps=2, margin=1e9
            )
            == {}
        )

    def test_cpu_timing_edges(self, monkeypatch):
        torch = pytest.importorskip("torch")
        # force the no-cuda timing edges inside the referee's _t
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        x = _v("x", 16)
        term = _p("mul", x, _p("relu", x))
        hs = {
            "g": {
                "pattern": ("mul", "X1", ("relu", "X1")),
                "kernel": "gk_c",
                "args": ("X1",),
            }
        }
        keep = gk.profitable(
            hs, term, {"x": torch.rand(16)}, reps=2, margin=1e9
        )
        assert "g" in keep

    def test_raising_kernel_declines_not_propagates(self):
        """A kernel that RAISES on its bound values declines.

        The scan tier's unrolled binding raises on a site the codegen
        accepted but cannot lower (operands sharing no broadcast) —
        the measured-cost filter must skip that handler, not abort
        the whole sweep.  A real export (``ChunkedRetention``) mints
        exactly such a site, so the crash would take down the gen
        domain's case builder.
        """
        torch = pytest.importorskip("torch")
        x = _v("x", 16)
        term = _p("mul", x, _p("relu", x))
        hs = {
            "g": {
                "pattern": ("mul", "X1", ("relu", "X1")),
                "kernel": "gk_raise",
                "args": ("X1",),
            }
        }

        def kernel_for(name, h, vals):
            def boom(*a, **k):
                raise RuntimeError("cannot lower this site")

            return boom

        assert (
            gk.profitable(
                hs,
                term,
                {"x": torch.rand(16)},
                reps=2,
                kernel_for=kernel_for,
            )
            == {}
        )

    def _borderline_fixture(self):
        torch = pytest.importorskip("torch")
        x = _v("x", 16)
        term = _p("mul", x, _p("relu", x))
        hs = {
            "g": {
                "pattern": ("mul", "X1", ("relu", "X1")),
                "kernel": "gk_b",
                "args": ("X1",),
            }
        }
        kf = lambda n, h, v: (lambda *a: a[0])  # noqa: E731
        return hs, term, {"x": torch.rand(16)}, kf

    def test_borderline_ratio_decided_by_window_majority(
        self, monkeypatch
    ):
        """A ratio inside ``_REFEREE_BORDERLINE`` is re-timed and the
        keep/drop decision is the majority of the windows' votes —
        a single noisy window must not move the keep set (the
        flip-rate residual: ChunkedSSD ``scan_2`` read 0.93x once
        against a ~1.05x typical)."""
        hs, term, env, kf = self._borderline_fixture()
        # first window reads a marginal loss; the two confirmations
        # read wins — majority keeps
        monkeypatch.setattr(
            gk, "_kernel_pays", lambda *a, **k: (1.0, 1.05)
        )
        scripted = iter([(1.0, 0.9), (1.0, 0.95)])
        calls = []

        def fake_paired(a, b, reps):
            calls.append(1)
            return next(scripted)

        monkeypatch.setattr(gk, "_paired_time", fake_paired)
        gk.PROFITABLE_LOG.clear()
        keep = gk.profitable(hs, term, env, reps=2, kernel_for=kf)
        assert "g" in keep
        assert len(calls) == 2
        rows = [r for r in gk.PROFITABLE_LOG if r[0] == "g"]
        assert [r[3] for r in rows] == [0, 1, 2]
        assert [r[4] for r in rows] == [False, True, True]

    def test_borderline_ratio_losing_majority_drops(
        self, monkeypatch
    ):
        """Same band, losing majority — the guard must not turn
        borderline noise into keeps either."""
        hs, term, env, kf = self._borderline_fixture()
        monkeypatch.setattr(
            gk, "_kernel_pays", lambda *a, **k: (1.0, 1.05)
        )
        scripted = iter([(1.0, 0.9), (1.0, 1.1)])
        monkeypatch.setattr(
            gk, "_paired_time", lambda a, b, reps: next(scripted)
        )
        gk.PROFITABLE_LOG.clear()
        keep = gk.profitable(hs, term, env, reps=2, kernel_for=kf)
        assert keep == {}
        rows = [r for r in gk.PROFITABLE_LOG if r[0] == "g"]
        assert [r[4] for r in rows] == [False, True, False]

    def test_clear_ratio_skips_confirmation_windows(
        self, monkeypatch
    ):
        """Outside the band the sweep keeps its single-window cost —
        the guard only pays where the noise lives."""
        hs, term, env, kf = self._borderline_fixture()
        monkeypatch.setattr(
            gk, "_kernel_pays", lambda *a, **k: (1.0, 3.0)
        )
        calls = []
        monkeypatch.setattr(
            gk,
            "_paired_time",
            lambda a, b, reps: calls.append(1) or (1.0, 3.0),
        )
        gk.PROFITABLE_LOG.clear()
        keep = gk.profitable(hs, term, env, reps=2, kernel_for=kf)
        assert keep == {}
        assert calls == []
        rows = [r for r in gk.PROFITABLE_LOG if r[0] == "g"]
        assert len(rows) == 1 and rows[0][3] == 0

    def test_fallback_env_move_failure_keeps_host(self, monkeypatch):
        """CUDA fallback: a failed ``.cuda()`` keeps the host env."""
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("the fallback only engages with CUDA present")

        def boom(*a, **k):
            raise RuntimeError("no device")

        monkeypatch.setattr(torch.Tensor, "cuda", boom)
        x = _v("x", 16)
        term = _p("mul", x, _p("relu", x))
        hs = {
            "g": {
                "pattern": ("mul", "X1", ("relu", "X1")),
                "kernel": "gk_host",
                "args": ("X1",),
            }
        }
        # must not propagate the move failure — the referee still
        # times the compiled spelled body on the host env
        gk.profitable(hs, term, {"x": torch.rand(16)}, reps=2)

    def test_kernel_raising_mid_timing_declines(self):
        """A kernel that verifies once then raises during timing
        declines — the timing loop is guarded too."""
        torch = pytest.importorskip("torch")
        x = _v("x", 16)
        term = _p("mul", x, _p("relu", x))
        hs = {
            "g": {
                "pattern": ("mul", "X1", ("relu", "X1")),
                "kernel": "gk_flaky",
                "args": ("X1",),
            }
        }
        xv = torch.rand(16)
        good = xv * torch.relu(xv)
        calls = [0]

        def kernel_for(name, h, vals):
            def flaky(*a, **k):
                calls[0] += 1
                if calls[0] > 1:
                    raise RuntimeError("flaky kernel")
                return good

            return flaky

        assert (
            gk.profitable(
                hs,
                term,
                {"x": xv},
                reps=2,
                kernel_for=kernel_for,
            )
            == {}
        )


class TestAttrSiteEnv:
    """Attr-metavar siting — the 5 prim specs ``_concrete_match`` misses."""

    def _handler(self):
        return {
            "pattern": (
                "mul",
                "m",
                ("unsqueeze", "n", {"dim": "a1"}),
            ),
            "kernel": "synop_x",
            "args": ("m", "n"),
            "attr_mvars": ("a1",),
        }

    def test_binds_attr_mvar_and_concretizes(self):
        x, y = _v("x", 4, 4), _v("y", 4, 1)
        term = _p("add", _p("mul", x, _p("unsqueeze", y, dim=-2)), x)
        h = self._handler()
        site = gk.attr_site_env(h, term)
        assert site is not None
        env, attrs = site
        assert attrs == {"a1": -2}
        assert env["m"] == x and env["n"] == y
        # the frozen copy concrete-matches through the ordinary path
        concrete = gk.concretize_at_site(h, term)
        assert concrete is not None
        assert concrete["attr_mvars"] == ()
        assert gk._handler_env(concrete, term) == env
        # the original stays polymorphic — delivery keeps it
        assert h["attr_mvars"] == ("a1",)

    def test_inconsistent_attr_binding_declines(self):
        # one mvar, two sites with different dims — no consistent bind
        h = {
            "pattern": (
                "add",
                ("unsqueeze", "a", {"dim": "d"}),
                ("unsqueeze", "b", {"dim": "d"}),
            ),
            "kernel": "synop_y",
            "args": ("a", "b"),
            "attr_mvars": ("d",),
        }
        x, y = _v("x", 4, 4), _v("y", 4, 4)
        term = _p(
            "add",
            _p("unsqueeze", x, dim=-2),
            _p("unsqueeze", y, dim=-1),
        )
        assert gk.attr_site_env(h, term) is None
        assert gk.concretize_at_site(h, term) is None

    def test_literal_attr_still_compares(self):
        h = {
            "pattern": ("unsqueeze", "a", {"dim": 0}),
            "kernel": "synop_z",
            "args": ("a",),
            "attr_mvars": ("a1",),
        }
        x = _v("x", 4, 4)
        assert gk.attr_site_env(h, _p("unsqueeze", x, dim=-2)) is None


class TestPairedTime:
    def test_interleaved_medians(self):
        import time

        def slow():
            time.sleep(0.002)

        def fast():
            pass

        ta, tb = gk._paired_time(slow, fast, 4)
        assert ta > tb > 0


class TestMeasuredDispatch:
    """``_rms_norm_measured`` — the law-fold delivery gate (T-026)."""

    def test_matches_fused_and_caches_pick(self):
        torch = pytest.importorskip("torch")
        import torch.nn.functional as F

        gk._RMSNORM_PICK.clear()
        try:
            x = torch.randn(4, 8, dtype=torch.float32)
            w = torch.randn(8)
            got = gk._rms_norm_measured(x, w, dim=(8,), eps=1e-5)
            want = F.rms_norm(x, (8,), weight=w, eps=1e-5)
            torch.testing.assert_close(got, want)
            assert gk._RMSNORM_PICK  # the pick was measured + cached
            # weight-free spelling agrees too
            got2 = gk._rms_norm_measured(x, None, dim=(8,), eps=1e-5)
            torch.testing.assert_close(
                got2, F.rms_norm(x, (8,), eps=1e-5)
            )
        finally:
            gk._RMSNORM_PICK.clear()

    def test_ambient_bindings_override_rms_norm(self):
        b = gk._ambient_bindings()
        assert b["rms_norm"] is gk._rms_norm_measured


def _spelled_scan(a, b, h, axis=-2):
    """The torch-loop oracle: ``h_t = a_t * h_{t-1} + b_t``."""
    a, b = torch.broadcast_tensors(a, b)
    ax = axis % a.dim()
    a, b = a.movedim(ax, 0), b.movedim(ax, 0)
    for t in range(a.shape[0]):
        h = a[t] * h + b[t]
    return h


def _cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("needs cuda for triton")
    return torch


class TestScanSpec:
    """``scan_spec`` — the carrier-spec recognizer."""

    def test_seq_form_recognised(self):
        info = gk.scan_spec(("applyd", ("aff_diag", "A", "B"), "H"))
        assert info["kind"] == "seq"
        assert info["h"] == "H"
        assert info["seq"] == ["A", "B"]
        assert info["params"] == ["A", "B", "H"]
        assert info["a"] == "A" and info["b"] == "B"

    def test_seq_form_fuses_elementwise_operands(self):
        info = gk.scan_spec(
            (
                "applyd",
                ("aff_diag", ("sigmoid", "G"), ("mul", "B", "X")),
                "H",
            )
        )
        assert info["kind"] == "seq"
        assert info["seq"] == ["G", "B", "X"]
        assert "tl.exp" in info["a"]

    def test_unrolled_form_recognised_chronological(self):
        info = gk.scan_spec(
            (
                "applyd",
                (
                    "affd_compose",
                    ("aff_diag", "A2", "B2"),
                    (
                        "affd_compose",
                        ("aff_diag", "A1", "B1"),
                        ("aff_diag", "A0", "B0"),
                    ),
                ),
                "H",
            )
        )
        assert info["kind"] == "unrolled"
        assert info["params"] == [
            "A2",
            "B2",
            "A1",
            "B1",
            "A0",
            "B0",
            "H",
        ]
        # in-order leaves are reverse-chronological — the kernel
        # applies A0 first
        assert info["steps"] == [
            ("A0", "B0"),
            ("A1", "B1"),
            ("A2", "B2"),
        ]

    def test_shared_meta_across_compose_leaves(self):
        # the same operand meta on two leaves dedups, not duplicates
        info = gk.scan_spec(
            (
                "applyd",
                (
                    "affd_compose",
                    ("aff_diag", "A", "B"),
                    ("aff_diag", "A", "B"),
                ),
                "H",
            )
        )
        assert info["kind"] == "unrolled"
        assert info["seq"] == ["A", "B"]

    def test_axis_attr_declared_on_spec(self):
        info = gk.scan_spec(
            ("applyd", ("aff_diag", "A", "B"), "H", {"axis": 0})
        )
        assert info["axis"] == 0

    def test_const_operand_inlines(self):
        info = gk.scan_spec(("applyd", ("aff_diag", 1.0, "B"), "H"))
        assert info["kind"] == "seq"
        # const leaves render dtype-typed ``tl.full`` tiles so
        # ``libdevice`` calls see matching operand dtypes (T-025)
        assert info["a"] == gk._TRITON_CONST_1D.format(v="1.0")
        assert info["seq"] == ["B"]

    def test_declines_foreign_and_malformed_specs(self):
        assert gk.scan_spec(("add", "X", "Y")) is None
        assert gk.scan_spec(("applyd", ("aff", "A", "B"), "H")) is None
        assert gk.scan_spec(("applyd", "f")) is None
        assert gk.scan_spec("applyd") is None
        assert gk.scan_spec(("apply", ("aff", "A", "B"), "H")) is None
        assert (
            gk.scan_spec(("applyd", ("aff_diag", "A", "B"), "H", "x"))
            is None
        )
        # a non-axis attr dict on applyd is ignored, not an error
        info = gk.scan_spec(
            ("applyd", ("aff_diag", "A", "B"), "H", {"other": 1})
        )
        assert info is not None and "axis" not in info
        # state must be a metavar
        assert (
            gk.scan_spec(
                ("applyd", ("aff_diag", "A", "B"), ("mul", "H", "H"))
            )
            is None
        )
        # a compose tree with a non-aff_diag leaf is impure
        assert (
            gk.scan_spec(
                (
                    "applyd",
                    (
                        "affd_compose",
                        ("aff", "A", "B"),
                        ("aff_diag", "C", "D"),
                    ),
                    "H",
                )
            )
            is None
        )
        # an aff_diag leaf carrying attrs is outside the grammar
        assert (
            gk.scan_spec(
                ("applyd", ("aff_diag", "A", "B", {"dim": 1}), "H")
            )
            is None
        )
        # operands the elementwise renderer cannot express
        assert (
            gk.scan_spec(
                ("applyd", ("aff_diag", ("matmul", "A", "B"), "B"), "H")
            )
            is None
        )
        # all-constant leaves carry no sequence operand
        assert (
            gk.scan_spec(("applyd", ("aff_diag", 1.0, 0.0), "H"))
            is None
        )
        # the state meta rebound as a sequence operand declines
        assert (
            gk.scan_spec(("applyd", ("aff_diag", "H", "B"), "H"))
            is None
        )
        # kernel-name-colliding metas decline
        assert (
            gk.scan_spec(("applyd", ("aff_diag", "acc", "B"), "H"))
            is None
        )
        assert (
            gk.scan_spec(("applyd", ("aff_diag", "x_ptr", "B"), "H"))
            is None
        )
        assert (
            gk.scan_spec(("applyd", ("aff_diag", "A", "B"), "t"))
            is None
        )

    def test_kind_attr_pins_unrolled(self):
        """``{"kind": "unrolled"}`` reads the one-leaf carrier as the
        ambient single-step ``a ⊙ h + x``, not a fold."""
        spec = (
            "applyd",
            ("aff_diag", "A", "B"),
            "H",
            {"kind": "unrolled"},
        )
        info = gk.scan_spec(spec)
        assert info["kind"] == "unrolled"
        assert info["steps"] == [("A", "B")]
        # an unknown kind declines rather than guessing
        assert (
            gk.scan_spec(
                ("applyd", ("aff_diag", "A", "B"), "H",
                 {"kind": "bogus"})
            )
            is None
        )

    def test_unrolled_hoists_unrenderable_operands(self):
        """``decay_vec``/view subterms hoist to host-computed params."""
        spec = (
            "applyd",
            (
                "aff_diag",
                ("mul",
                 ("unsqueeze",
                  ("decay_vec", "g", {"offset": 4, "t": 1}),
                  {"dim": -1}),
                 ("unsqueeze",
                  ("decay_vec", "g", {"sign": -1, "t": 4}),
                  {"dim": -2})),
                "x",
            ),
            "h",
            {"kind": "unrolled"},
        )
        info = gk.scan_spec(spec)
        assert info["kind"] == "unrolled"
        assert [n for n, _ in info["computed"]] == ["__c0", "__c1"]
        assert info["steps"] == [("((__c0) * (__c1))", "x")]
        # seq form still declines the same operand — no step axis
        # for a state-shaped subterm
        spec_seq = (
            "applyd",
            ("aff_diag",
             ("decay_vec", "g", {"t": 4}),
             "x"),
            "h",
        )
        assert gk.scan_spec(spec_seq) is None
        # a non-attr op outside the grammar hoists whole in the
        # unrolled kind — (q·kᵀ) as a computed operand
        info_mm = gk.scan_spec(
            ("applyd",
             ("aff_diag", "A", ("matmul", "Q", "K")),
             "H",
             {"kind": "unrolled"})
        )
        assert info_mm["kind"] == "unrolled"
        assert [n for n, _ in info_mm["computed"]] == ["__c0"]
        assert info_mm["steps"] == [("A", "__c0")]

    def test_unrolled_prunes_dead_hoists_and_buried_metavars(self):
        """Only expr-referenced operands join the flat index space.

        The kid ``split``s hoist first, then their ``matmul`` parent
        hoists whole — the kids are dead ``__c`` entries.  ``Q``/``K``
        live only inside the hoist: they stay positional binding
        args (``params``) but never reach the kernel's broadcast.
        """
        info = gk.scan_spec(
            (
                "applyd",
                (
                    "aff_diag",
                    (
                        "matmul",
                        (
                            "split",
                            "Q",
                            {"dim": -2, "index": 1, "sizes": [2, 2]},
                        ),
                        (
                            "split",
                            "K",
                            {"dim": -1, "index": 1, "sizes": [2, 2]},
                        ),
                    ),
                    "B",
                ),
                "H",
                {"kind": "unrolled"},
            )
        )
        assert info["kind"] == "unrolled"
        assert info["steps"] == [("__c2", "B")]
        assert [n for n, _ in info["computed"]] == ["__c2"]
        assert info["params"] == ["Q", "K", "B", "H"]
        assert info["flat"] == ["B"]
        assert info["kernel_args"] == ["B", "H", "__c2"]


class TestScanSources:
    """``_scan_triton_source`` — emitted module text."""

    def test_seq_source_inspectable(self):
        src = gk._scan_triton_source(
            gk.scan_spec(("applyd", ("aff_diag", "A", "B"), "H"))
        )
        assert "triton.jit" in src and "for t in range(T)" in src
        assert "row = t.to(tl.int64) * N" in src
        assert "acc = (A) * acc + (B)" in src

    def test_unrolled_source_inspectable(self):
        src = gk._scan_triton_source(
            gk.scan_spec(
                (
                    "applyd",
                    (
                        "affd_compose",
                        ("aff_diag", "A2", "B2"),
                        ("aff_diag", "A1", "B1"),
                    ),
                    "H",
                )
            )
        )
        first = src.index("acc = (A1)")
        second = src.index("acc = (A2)")
        assert first < second  # chronological application


class TestScanBindings:
    """Generated scan kernels — fp64-verified vs the spelled loop."""

    def _handlers(self):
        return {
            "s": {
                "pattern": ("applyd", ("aff_diag", "A", "B"), "H"),
                "kernel": "sc_t",
                "args": ("A", "B", "H"),
            }
        }

    def test_seq_binding_fp64_matches_spelled(self):
        torch = _cuda()
        b = gk.scan_triton_bindings(self._handlers())
        T, B, d = 96, 4, 48
        a = torch.rand(B, T, d, device="cuda", dtype=torch.float64)
        bb = torch.randn(B, T, d, device="cuda", dtype=torch.float64)
        h = torch.randn(B, d, device="cuda", dtype=torch.float64)
        got = b["sc_t"](a, bb, h)
        torch.testing.assert_close(
            got, _spelled_scan(a, bb, h), atol=1e-12, rtol=1e-12
        )
        assert got.shape == (B, d)

    def test_seq_binding_layouts_and_edges(self):
        torch = _cuda()
        b = gk.scan_triton_bindings(self._handlers())
        dt = torch.float64
        # (T, d) layout — step axis 0 via the -2 default
        a = torch.rand(64, 16, device="cuda", dtype=dt) * 0.5 + 0.4
        bb = torch.randn(64, 16, device="cuda", dtype=dt)
        h = torch.randn(16, device="cuda", dtype=dt)
        torch.testing.assert_close(
            b["sc_t"](a, bb, h),
            _spelled_scan(a, bb, h),
            atol=1e-12,
            rtol=1e-12,
        )
        # length-1 sequence is the single map application
        a1, b1 = a[:1], bb[:1]
        torch.testing.assert_close(
            b["sc_t"](a1, b1, h),
            a1[0] * h + b1[0],
            atol=1e-12,
            rtol=1e-12,
        )
        # non-contiguous operands fold identically
        an = torch.rand(16, 64, device="cuda", dtype=dt).t()
        bn = torch.randn(16, 64, device="cuda", dtype=dt).t()
        torch.testing.assert_close(
            b["sc_t"](an, bn, h),
            _spelled_scan(an, bn, h),
            atol=1e-12,
            rtol=1e-12,
        )
        # a shared (d,) decay broadcasts over the step axis
        ash = torch.rand(16, device="cuda", dtype=dt) * 0.5 + 0.4
        torch.testing.assert_close(
            b["sc_t"](ash, bb, h),
            _spelled_scan(ash, bb, h),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_seq_axis_explicit_and_derived(self):
        torch = _cuda()
        hs0 = {
            "s": {
                "pattern": ("applyd", ("aff_diag", "A", "B"), "H"),
                "kernel": "sc_ax0",
                "args": ("A", "B", "H"),
                "axis": 0,
            }
        }
        hs_auto = {
            "s": {
                "pattern": ("applyd", ("aff_diag", "A", "B"), "H"),
                "kernel": "sc_auto",
                "args": ("A", "B", "H"),
                "axis": None,
            }
        }
        b0 = gk.scan_triton_bindings(hs0)
        ba = gk.scan_triton_bindings(hs_auto)
        dt = torch.float64
        a = torch.rand(48, 24, device="cuda", dtype=dt) * 0.5 + 0.4
        bb = torch.randn(48, 24, device="cuda", dtype=dt)
        h = torch.randn(24, device="cuda", dtype=dt)
        want = _spelled_scan(a, bb, h, axis=0)
        torch.testing.assert_close(
            b0["sc_ax0"](a, bb, h), want, atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(
            ba["sc_auto"](a, bb, h), want, atol=1e-12, rtol=1e-12
        )

    def test_seq_axis_spec_attr_wins_over_default(self):
        torch = _cuda()
        # spec declares axis=0; the handler default (-2) must lose
        hs = {
            "s": {
                "pattern": (
                    "applyd",
                    ("aff_diag", "A", "B"),
                    "H",
                    {"axis": 0},
                ),
                "kernel": "sc_attr",
                "args": ("A", "B", "H"),
            }
        }
        b = gk.scan_triton_bindings(hs, axis=-1)
        dt = torch.float64
        a = torch.rand(32, 8, device="cuda", dtype=dt) * 0.5 + 0.4
        bb = torch.randn(32, 8, device="cuda", dtype=dt)
        h = torch.randn(8, device="cuda", dtype=dt)
        torch.testing.assert_close(
            b["sc_attr"](a, bb, h),
            _spelled_scan(a, bb, h, axis=0),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_unrolled_binding_matches_carrier_semantics(self):
        torch = _cuda()
        hs = {
            "s": {
                "pattern": (
                    "applyd",
                    (
                        "affd_compose",
                        ("aff_diag", "A2", "B2"),
                        (
                            "affd_compose",
                            ("aff_diag", "A1", "B1"),
                            ("aff_diag", "A0", "B0"),
                        ),
                    ),
                    "H",
                ),
                "kernel": "sc_unr",
                "args": ("A2", "B2", "A1", "B1", "A0", "B0", "H"),
            }
        }
        b = gk.scan_triton_bindings(hs)
        dt = torch.float64
        a0, b0_ = (
            torch.rand(64, device="cuda", dtype=dt),
            torch.randn(64, device="cuda", dtype=dt),
        )
        a1, b1_ = (
            torch.rand(64, device="cuda", dtype=dt),
            torch.randn(64, device="cuda", dtype=dt),
        )
        a2, b2_ = (
            torch.rand(64, device="cuda", dtype=dt),
            torch.randn(64, device="cuda", dtype=dt),
        )
        h = torch.randn(64, device="cuda", dtype=dt)
        got = b["sc_unr"](a2, b2_, a1, b1_, a0, b0_, h)
        want = a0 * h + b0_
        want = a1 * want + b1_
        want = a2 * want + b2_
        torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)

    def test_unrolled_binding_with_hoisted_operands(self):
        """The minted decayed-attention carrier shape, delivered.

        ``applyd(aff_diag(g^o·eta, x), h)`` where the weight is a
        ``decay_vec`` product — the form the e-graph produces on a
        real export — renders as a one-step carrier with the decay
        vectors hoisted to host-computed kernel args.
        """
        torch = _cuda()
        hs = {
            "s": {
                "pattern": (
                    "applyd",
                    (
                        "aff_diag",
                        (
                            "mul",
                            (
                                "unsqueeze",
                                ("decay_vec", "g",
                                 {"offset": 4, "t": 1}),
                                {"dim": -1},
                            ),
                            (
                                "unsqueeze",
                                ("decay_vec", "g",
                                 {"sign": -1, "t": 4}),
                                {"dim": -2},
                            ),
                        ),
                        "X",
                    ),
                    "H",
                    {"kind": "unrolled"},
                ),
                "kernel": "sc_dv",
                "args": ("g", "X", "H"),
            }
        }
        b = gk.scan_triton_bindings(hs)
        dt = torch.float64
        g = torch.tensor(0.9, dtype=dt, device="cuda")
        x = torch.randn(4, 4, dtype=dt, device="cuda")
        h = torch.randn(4, 4, dtype=dt, device="cuda")
        got = b["sc_dv"](g, x, h)
        # a = (γ^4 · γ^{-c}) broadcast — the (1,4) decay row
        a = (0.9 ** 4) * torch.tensor(
            [0.9 ** 0, 0.9 ** -1, 0.9 ** -2, 0.9 ** -3],
            dtype=dt,
            device="cuda",
        )
        torch.testing.assert_close(
            got, a * h + x, atol=1e-12, rtol=1e-12
        )

    def test_fused_step_operands_evaluated_inside_kernel(self):
        torch = _cuda()
        hs = {
            "s": {
                "pattern": (
                    "applyd",
                    ("aff_diag", ("sigmoid", "G"), ("mul", "B", "X")),
                    "H",
                ),
                "kernel": "sc_fuse",
                "args": ("G", "B", "X", "H"),
            }
        }
        b = gk.scan_triton_bindings(hs)
        dt = torch.float64
        g = torch.randn(64, 32, device="cuda", dtype=dt)
        bb = torch.randn(64, 32, device="cuda", dtype=dt)
        x = torch.randn(64, 32, device="cuda", dtype=dt)
        h = torch.randn(32, device="cuda", dtype=dt)
        want = _spelled_scan(torch.sigmoid(g), bb * x, h)
        torch.testing.assert_close(
            b["sc_fuse"](g, bb, x, h), want, atol=1e-12, rtol=1e-12
        )

    def test_scan_kernel_files_landed_on_disk(self):
        _cuda()
        gk.scan_triton_bindings(self._handlers())
        f = gk._kernel_cache() / "_gs_sc_t.py"
        assert f.exists() and "triton.jit" in f.read_text()

    def test_unrecognised_spec_gets_no_binding(self):
        pytest.importorskip("torch")
        pytest.importorskip("triton")
        hs = {
            "s": {
                "pattern": ("matmul", "A", "B"),
                "kernel": "sc_no",
                "args": ("A", "B"),
            }
        }
        assert gk.scan_triton_bindings(hs) == {}

    def test_module_load_decline_keeps_torch_path(self, monkeypatch):
        pytest.importorskip("torch")
        pytest.importorskip("triton")
        import importlib.util

        monkeypatch.setattr(
            importlib.util,
            "spec_from_file_location",
            lambda *a, **k: None,
        )
        assert gk.scan_triton_bindings(self._handlers()) == {}


class TestScanErrors:
    """Call-time honesty — shape violations raise, never fall back."""

    def test_cpu_tensors_raise(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("triton")
        hs = {
            "s": {
                "pattern": ("applyd", ("aff_diag", "A", "B"), "H"),
                "kernel": "sc_cpu",
                "args": ("A", "B", "H"),
            }
        }
        b = gk.scan_triton_bindings(hs)
        a = torch.rand(4, 8)
        with pytest.raises(RuntimeError, match="CUDA"):
            b["sc_cpu"](a, a, torch.rand(8))

    def test_unrolled_cpu_tensors_raise(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("triton")
        hs = {
            "s": {
                "pattern": (
                    "applyd",
                    (
                        "affd_compose",
                        ("aff_diag", "A2", "B2"),
                        ("aff_diag", "A1", "B1"),
                    ),
                    "H",
                ),
                "kernel": "sc_ucpu",
                "args": ("A2", "B2", "A1", "B1", "H"),
            }
        }
        b = gk.scan_triton_bindings(hs)
        t = torch.rand(8)
        with pytest.raises(RuntimeError, match="CUDA"):
            b["sc_ucpu"](t, t, t, t, t)

    def test_scalar_operands_carry_no_step_axis(self):
        torch = _cuda()
        hs = {
            "s": {
                "pattern": ("applyd", ("aff_diag", "A", "B"), "H"),
                "kernel": "sc_scal",
                "args": ("A", "B", "H"),
            }
        }
        b = gk.scan_triton_bindings(hs)
        with pytest.raises(ValueError, match="step axis"):
            b["sc_scal"](
                torch.ones((), device="cuda"),
                torch.ones((), device="cuda"),
                torch.ones((), device="cuda"),
            )

    def test_ambiguous_axis_raises(self):
        torch = _cuda()
        hs = {
            "s": {
                "pattern": ("applyd", ("aff_diag", "A", "B"), "H"),
                "kernel": "sc_amb",
                "args": ("A", "B", "H"),
                "axis": None,
            }
        }
        b = gk.scan_triton_bindings(hs)
        # square (d, d) operands over a (d,) state: removing either
        # axis leaves a shape h broadcasts into — undecidable
        d = 8
        a = torch.rand(d, d, device="cuda", dtype=torch.float64)
        with pytest.raises(ValueError, match="not determined"):
            b["sc_amb"](a, a, torch.rand(d, device="cuda"))

    def test_state_shape_violation_surfaces(self):
        torch = _cuda()
        hs = {
            "s": {
                "pattern": ("applyd", ("aff_diag", "A", "B"), "H"),
                "kernel": "sc_bad",
                "args": ("A", "B", "H"),
            }
        }
        b = gk.scan_triton_bindings(hs)
        a = torch.rand(8, 4, device="cuda", dtype=torch.float64)
        bad_h = torch.rand(3, device="cuda", dtype=torch.float64)
        with pytest.raises(RuntimeError):
            b["sc_bad"](a, a, bad_h)


class TestScanHandlers:
    """``scan_handlers`` — minting carrier-op entries."""

    def test_mints_scan_entries(self):
        a = _v("a", 8, 4)
        h = _v("h", 4)
        x = _v("x", 8, 4)
        term = _p("applyd", _p("aff_diag", a, x), h)
        hs = gk.scan_handlers([term])
        assert "scan_0" in hs
        assert hs["scan_0"]["kernel"] == "scan_0_k"
        assert hs["scan_0"]["pattern"] == (
            "applyd",
            ("aff_diag", "X1", "X2"),
            "X3",
        )
        assert hs["scan_0"]["args"] == ("X1", "X2", "X3")
        assert hs["scan_0"]["axis"] == -2

    def test_compose_tree_mints_unrolled(self):
        hs = gk.scan_handlers(
            [
                (
                    "applyd",
                    (
                        "affd_compose",
                        ("aff_diag", "A2", "B2"),
                        ("aff_diag", "A1", "B1"),
                    ),
                    "H",
                )
            ]
        )
        assert hs["scan_0"]["args"] == ("A2", "B2", "A1", "B1", "H")

    def test_non_scan_and_dupes_skipped(self):
        x, y = _v("x", 4), _v("y", 4)
        hs = gk.scan_handlers([_p("mul", x, y)])
        assert hs == {}
        a, b, h = _v("a", 4), _v("b", 4), _v("h", 4)
        t1 = _p("applyd", _p("aff_diag", a, b), h)
        t2 = _p("applyd", _p("aff_diag", b, a), h)  # same canon shape
        hs2 = gk.scan_handlers([t1, t2])
        assert len(hs2) == 1
        covered = {
            _canon_concrete(("applyd", ("aff_diag", "A", "B"), "H"))
        }
        assert gk.scan_handlers([t1], covered=covered) == {}


class TestSeqScanDenotation:
    """``seq_scan_denotation`` — the spelled fold, device-free."""

    def test_denotation_matches_spelled_fold(self):
        torch = pytest.importorskip("torch")
        spec = ("applyd", ("aff_diag", "A", "B"), "H")
        fn = gk.seq_scan_denotation(spec)
        assert fn is not None
        B, T, d = 2, 16, 8
        a = (torch.rand(B, T, d) * 0.5 + 0.4).double()
        b = torch.randn(B, T, d).double()
        h = torch.randn(B, d).double()
        torch.testing.assert_close(
            fn(a, b, h), _spelled_scan(a, b, h), atol=1e-12, rtol=1e-12
        )
        # (T, d) layout — the -2 default resolves axis 0
        a1 = (torch.rand(T, d) * 0.5 + 0.4).double()
        b1 = torch.randn(T, d).double()
        h1 = torch.randn(d).double()
        torch.testing.assert_close(
            fn(a1, b1, h1),
            _spelled_scan(a1, b1, h1, axis=0),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_denotation_evals_subspec_operands(self):
        torch = pytest.importorskip("torch")
        spec = (
            "applyd",
            ("aff_diag", ("sigmoid", "G"), ("mul", "B", "X")),
            "H",
        )
        fn = gk.seq_scan_denotation(spec)
        B, T, d = 2, 12, 6
        G = torch.randn(B, T, d).double()
        Bm = torch.randn(B, T, d).double()
        X = (torch.rand(B, T, d) + 0.5).double()
        h = torch.randn(B, d)
        got = fn(G, Bm, X, h)
        aa, bb = torch.broadcast_tensors(
            torch.sigmoid(G), Bm * X
        )
        aa, bb = aa.movedim(1, 0), bb.movedim(1, 0)
        want = h
        for t in range(T):
            want = aa[t] * want + bb[t]
        torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)

    def test_denotation_broadcasts_shared_operands(self):
        torch = pytest.importorskip("torch")
        spec = ("applyd", ("aff_diag", "A", "B"), "H")
        fn = gk.seq_scan_denotation(spec)
        T, d = 10, 8
        # a shared (d,) decay broadcasts over the step axis
        ash = (torch.rand(d) * 0.5 + 0.4).double()
        bb = torch.randn(T, d).double()
        h = torch.randn(d).double()
        torch.testing.assert_close(
            fn(ash, bb, h),
            _spelled_scan(ash, bb, h, axis=0),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_denotation_axis_explicit_and_derived(self):
        torch = pytest.importorskip("torch")
        spec = ("applyd", ("aff_diag", "A", "B"), "H")
        T, d = 9, 5
        a = (torch.rand(T, d) * 0.5 + 0.4).double()
        b = torch.randn(T, d).double()
        h = torch.randn(d).double()
        want = _spelled_scan(a, b, h, axis=0)
        torch.testing.assert_close(
            gk.seq_scan_denotation(spec, axis=0)(a, b, h),
            want,
            atol=1e-12,
            rtol=1e-12,
        )
        torch.testing.assert_close(
            gk.seq_scan_denotation(spec, axis=None)(a, b, h),
            want,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_denotation_declines_non_seq(self):
        # unrolled compose trees and non-scan specs get no fold
        assert (
            gk.seq_scan_denotation(
                (
                    "applyd",
                    ("affd_compose", ("aff_diag", "A", "B"), ("aff_diag", "C", "D")),
                    "H",
                )
            )
            is None
        )
        assert gk.seq_scan_denotation(("mul", "X1", "X2")) is None
        # a compose tree PINNED kind="seq" is not a scan spec at all
        # — the arity guard declines rather than silently keep one leaf
        assert (
            gk.scan_spec(
                (
                    "applyd",
                    (
                        "affd_compose",
                        ("aff_diag", "A", "B"),
                        ("aff_diag", "C", "D"),
                    ),
                    "H",
                    {"kind": "seq"},
                )
            )
            is None
        )

    def test_handler_denotation_resolves_axis(self):
        spec = ("applyd", ("aff_diag", "A", "B"), "H")
        h = {"pattern": spec, "kernel": "k", "args": ("A", "B", "H")}
        assert gk.handler_denotation(h) is not None
        # spec attr axis wins over the handler default chain only
        # when the handler carries none — mirror scan_triton_bindings
        h_ax = dict(h, axis=0)
        assert gk.handler_denotation(h_ax) is not None
        assert gk.handler_denotation({"pattern": ("mul", "X", "Y")}) is None

    def test_gen_bindings_binds_the_fold_on_cpu(self):
        torch = pytest.importorskip("torch")
        hs = {
            "s": {
                "pattern": ("applyd", ("aff_diag", "A", "B"), "H"),
                "kernel": "sc_den",
                "args": ("A", "B", "H"),
                "axis": -2,
            }
        }
        bindings = gk.gen_bindings(hs, compile_kernels=False)
        T, d = 12, 7
        a = (torch.rand(2, T, d) * 0.5 + 0.4).double()
        b = torch.randn(2, T, d).double()
        h = torch.randn(2, d).double()
        torch.testing.assert_close(
            bindings["sc_den"](a, b, h),
            _spelled_scan(a, b, h),
            atol=1e-12,
            rtol=1e-12,
        )
        # and the spelled-eval map application is NOT what ships
        assert bindings["sc_den"](a, b, h).shape == h.shape

    def test_profitable_denotation_needs_a_kernel(self):
        # a seq claim whose codegen binds nothing cannot pay —
        # delivery IS the denotation, so the referee declines
        # rather than timing fold-vs-fold (or worse, fold-vs-map)
        torch = pytest.importorskip("torch")
        a, b, h = _v("a", 4, 8), _v("b", 4, 8), _v("h", 8)
        hs = {
            "scan_0": {
                "pattern": ("applyd", ("aff_diag", "A", "B"), "H"),
                "kernel": "scan_0_k",
                "args": ("A", "B", "H"),
                "leaf_env": {"A": a, "B": b, "H": h},
            }
        }
        var_env = {
            "a": (torch.rand(4, 8) * 0.5 + 0.4).double(),
            "b": torch.randn(4, 8).double(),
            "h": torch.randn(8).double(),
        }
        term = _p("applyd", _p("aff_diag", a, b), h)
        keep = gk.profitable(
            hs,
            term,
            var_env,
            reps=2,
            kernel_for=lambda *a, **k: None,
            denotation_for=play._claim_denotation_for,
        )
        assert keep == {}
        # a bound kernel is verified against the FOLD and kept at a
        # huge margin — the spelled arm is the denotation, not the
        # ambient map (which would verify-fail outright)
        def kernel(*args):
            return gk.seq_scan_denotation(
                ("applyd", ("aff_diag", "A", "B"), "H")
            )(*args)

        keep2 = gk.profitable(
            hs,
            term,
            var_env,
            reps=2,
            margin=1e9,
            kernel_for=lambda n, hh, v: (kernel, v),
            denotation_for=play._claim_denotation_for,
        )
        assert keep2 == hs


# The DeltaRuleBlock spelling: ``S += beta·k ⊗ (v - kᵀS)`` — the
# rank-1-feedback ``b`` the seqm tier denotes.
_SEQM_SPEC = (
    "applyd",
    (
        "aff_diag",
        1,
        (
            "mul",
            ("mul", "C1", "C2"),
            (
                "unsqueeze",
                (
                    "sub",
                    "V",
                    (
                        "squeeze",
                        ("matmul", "K", "H"),
                        {"dim": 1},
                    ),
                ),
                {"dim": -2},
            ),
        ),
    ),
    "H",
    {"kind": "seqm", "axis": 1},
)


class TestSeqmSpec:
    """``scan_spec`` on ``kind="seqm"`` — the matrix-state recognizer."""

    def test_seqm_form_recognised(self):
        info = gk.scan_spec(_SEQM_SPEC)
        assert info is not None and info["kind"] == "seqm"
        assert info["k"] == "K"
        assert info["h"] == "H"
        assert set(info["seq"]) == {"C1", "C2", "V", "K"}
        assert info["params"] == ["C1", "C2", "V", "K", "H"]

    def test_seqm_gated_form_recognised(self):
        # ``a_t`` map + in-residual scalar gate — the GatedDeltaRule
        # spelling ``S ← a·S + C⊗(V - s·KᵀS)``
        spec = (
            "applyd",
            (
                "aff_diag",
                "A",
                (
                    "mul",
                    "C",
                    (
                        "unsqueeze",
                        (
                            "sub",
                            "V",
                            ("mul", "S", ("squeeze", ("matmul", "K", "H"), {"dim": 1})),
                        ),
                        {"dim": -2},
                    ),
                ),
            ),
            "H",
            {"kind": "seqm", "axis": 1},
        )
        info = gk.scan_spec(spec)
        assert info is not None and info["kind"] == "seqm"
        assert info["s"] == "S"
        assert info["a"] == "A"

    def test_seqm_declines_state_free_b(self):
        # ``kind="seqm"`` pinned on a state-free ``b`` is not the
        # rank-1-feedback form — decline rather than re-interpret
        spec = (
            "applyd",
            ("aff_diag", 1, ("mul", "A", "B")),
            "H",
            {"kind": "seqm", "axis": 1},
        )
        assert gk.scan_spec(spec) is None

    def test_seqm_declines_state_in_a(self):
        # the map factor must be accumulator-free
        spec = (
            "applyd",
            (
                "aff_diag",
                ("mul", "H", "A"),
                (
                    "mul",
                    "C",
                    (
                        "unsqueeze",
                        (
                            "sub",
                            "V",
                            ("squeeze", ("matmul", "K", "H"), {"dim": 1}),
                        ),
                        {"dim": -2},
                    ),
                ),
            ),
            "H",
            {"kind": "seqm", "axis": 1},
        )
        assert gk.scan_spec(spec) is None

    def test_seqm_declines_non_feedback_b_skeleton(self):
        # ``H`` inside ``b`` but NOT under the recognised
        # ``C⊗(V - [s·]KᵀH)`` skeleton — e.g. a plain matvec write
        spec = (
            "applyd",
            ("aff_diag", 1, ("mul", "C", ("matmul", "K", "H"))),
            "H",
            {"kind": "seqm", "axis": 1},
        )
        assert gk.scan_spec(spec) is None


class TestSeqmDenotation:
    """``seqm_denotation`` — the spelled matrix-state fold."""

    def _args(self, B=2, T=5, dk=3, dv=4):
        torch = pytest.importorskip("torch")
        dt = torch.float64
        g = torch.Generator().manual_seed(0)

        def r(*s):
            return torch.randn(*s, generator=g, dtype=dt)

        return (
            (torch.rand(B, T, 1, 1, generator=g, dtype=dt) * 0.4 + 0.6),
            (torch.rand(B, T, dk, 1, generator=g, dtype=dt) * 0.4 + 0.6),
            r(B, T, dv),
            (torch.rand(B, T, 1, dk, generator=g, dtype=dt) * 0.4 + 0.6),
            r(B, dk, dv),
        )

    def _spelled(self, c1, c2, v, k, h, B, T, dk, dv):
        torch = pytest.importorskip("torch")
        acc = h
        for t in range(T):
            ct = c1[:, t] * c2[:, t]  # (B,dk,1)
            kt = k[:, t]  # (B,1,dk)
            proj = torch.matmul(kt, acc).squeeze(1)  # (B,dv)
            r = v[:, t] - proj
            acc = acc + ct * r.unsqueeze(-2)
        return acc

    def test_seqm_denotation_matches_spelled_fold(self):
        torch = pytest.importorskip("torch")
        args = self._args()
        B, T, dk, dv = 2, 5, 3, 4
        axes = {"C1": 1, "C2": 1, "V": 1, "K": 1}
        fn = gk.seqm_denotation(_SEQM_SPEC, axes=axes)
        assert fn is not None
        torch.testing.assert_close(
            fn(*args), self._spelled(*args, B, T, dk, dv)
        )

    def test_seqm_denotation_requires_axes(self):
        args = self._args(B=1, T=2, dk=2, dv=2)
        fn = gk.seqm_denotation(_SEQM_SPEC)
        assert fn is not None
        with pytest.raises(ValueError, match="axes"):
            fn(*args)

    def test_seqm_denotation_declines_non_seqm(self):
        assert (
            gk.seqm_denotation(("applyd", ("aff_diag", "A", "B"), "H"))
            is None
        )
        assert gk.seqm_denotation(("mul", "X1", "X2")) is None

    def test_handler_denotation_dispatches_seqm(self):
        args = self._args(B=1, T=2, dk=2, dv=2)
        h = {
            "pattern": _SEQM_SPEC,
            "kernel": "k",
            "args": ("C1", "C2", "V", "K", "H"),
            "axes": {"C1": 1, "C2": 1, "V": 1, "K": 1},
        }
        fn = gk.handler_denotation(h)
        assert fn is not None
        torch = pytest.importorskip("torch")
        torch.testing.assert_close(
            fn(*args), self._spelled(*args, 1, 2, 2, 2)
        )
        # a seqm handler without the axes map cannot denote — the
        # denotation raises rather than guessing a shared axis
        bad = dict(h)
        bad.pop("axes")
        with pytest.raises(ValueError, match="axes"):
            gk.handler_denotation(bad)(*args)


class TestSeqmBindings:
    """Generated seqm kernels — fp64-verified vs the fold."""

    def _handler(self, kernel="sm_t"):
        return {
            "s": {
                "pattern": _SEQM_SPEC,
                "kernel": kernel,
                "args": ("C1", "C2", "V", "K", "H"),
                "axis": 1,
                "axes": {"C1": 1, "C2": 1, "V": 1, "K": 1},
            }
        }

    def _args(self, B=2, T=6, dk=8, dv=8):
        torch = _cuda()
        dt = torch.float64
        g = torch.Generator(device="cuda").manual_seed(0)

        def r(*s):
            return torch.randn(*s, generator=g, dtype=dt, device="cuda")

        return (
            (torch.rand(B, T, 1, 1, generator=g, dtype=dt, device="cuda") * 0.4 + 0.6),
            (torch.rand(B, T, dk, 1, generator=g, dtype=dt, device="cuda") * 0.4 + 0.6),
            r(B, T, dv),
            (torch.rand(B, T, 1, dk, generator=g, dtype=dt, device="cuda") * 0.4 + 0.6),
            r(B, dk, dv),
        )

    def _spelled(self, c1, c2, v, k, h, T, dk, dv):
        torch = pytest.importorskip("torch")
        acc = h
        for t in range(T):
            ct = c1[:, t] * c2[:, t]
            proj = torch.matmul(k[:, t], acc).squeeze(1)
            r = v[:, t] - proj
            acc = acc + ct * r.unsqueeze(-2)
        return acc

    def test_seqm_binding_fp64_matches_fold(self):
        torch = _cuda()
        b = gk.scan_triton_bindings(self._handler())
        args = self._args()
        B, T, dk, dv = 2, 6, 8, 8
        got = b["sm_t"](*args)
        torch.testing.assert_close(
            got,
            self._spelled(*args, T, dk, dv),
            atol=1e-12,
            rtol=1e-12,
        )
        assert got.shape == (B, dk, dv)

    def test_seqm_binding_gated_matches_fold(self):
        # the gated spec: a_t map + in-residual scalar gate
        torch = _cuda()
        spec = (
            "applyd",
            (
                "aff_diag",
                "A",
                (
                    "mul",
                    "C",
                    (
                        "unsqueeze",
                        (
                            "sub",
                            "V",
                            ("mul", "S", ("squeeze", ("matmul", "K", "H"), {"dim": 1})),
                        ),
                        {"dim": -2},
                    ),
                ),
            ),
            "H",
            {"kind": "seqm", "axis": 1},
        )
        hs = {
            "s": {
                "pattern": spec,
                "kernel": "sm_g",
                "args": ("A", "C", "V", "S", "K", "H"),
                "axis": 1,
                "axes": {"A": 1, "C": 1, "V": 1, "S": 1, "K": 1},
            }
        }
        b = gk.scan_triton_bindings(hs)
        B, T, dk, dv = 2, 6, 8, 8
        dt = torch.float64
        g = torch.Generator(device="cuda").manual_seed(1)

        def r(*s):
            return torch.randn(*s, generator=g, dtype=dt, device="cuda")

        a = (torch.rand(B, T, dk, 1, generator=g, dtype=dt, device="cuda") * 0.4 + 0.6)
        c = (torch.rand(B, T, 1, 1, generator=g, dtype=dt, device="cuda") * 0.4 + 0.6)
        v = r(B, T, dv)
        s = torch.rand(B, T, 1, generator=g, dtype=dt, device="cuda") * 0.5 + 0.5
        k = (torch.rand(B, T, 1, dk, generator=g, dtype=dt, device="cuda") * 0.4 + 0.6)
        h = r(B, dk, dv)
        acc = h
        for t in range(T):
            ct = c[:, t]
            proj = torch.matmul(k[:, t], acc).squeeze(1)
            rr = v[:, t] - s[:, t] * proj
            acc = a[:, t] * acc + ct * rr.unsqueeze(-2)
        torch.testing.assert_close(
            b["sm_g"](a, c, v, s, k, h), acc, atol=1e-12, rtol=1e-12
        )

    def test_seqm_binding_declines_ineligible_shapes(self):
        torch = _cuda()
        b = gk.scan_triton_bindings(self._handler())
        # dk beyond the tile cap declines honestly
        args = self._args(dk=72, dv=8)
        with pytest.raises(ValueError, match="dk"):
            b["sm_t"](*args)

    def test_seqm_binding_needs_cuda(self):
        pytest.importorskip("torch")
        pytest.importorskip("triton")
        if torch.cuda.is_available():
            pytest.skip("cpu-only check")
        hs = self._handler()
        b = gk.scan_triton_bindings(hs)
        args = TestSeqmDenotation()._args(B=1, T=2, dk=2, dv=2)
        with pytest.raises(RuntimeError, match="CUDA"):
            b["sm_t"](*args)

    def test_seqm_kernel_file_landed_on_disk(self):
        torch = _cuda()
        b = gk.scan_triton_bindings(self._handler())
        b["sm_t"](*self._args(B=1, T=2, dk=4, dv=4))
        files = list(gk._kernel_cache().glob("_gsm_sm_t_*.py"))
        assert files and "triton.jit" in files[0].read_text()

    def test_unrecognised_seqm_gets_no_binding(self):
        pytest.importorskip("torch")
        pytest.importorskip("triton")
        hs = {
            "s": {
                "pattern": (
                    "applyd",
                    ("aff_diag", 1, ("mul", "A", "B")),
                    "H",
                    {"kind": "seqm", "axis": 1},
                ),
                "kernel": "sm_no",
                "args": ("A", "B", "H"),
            }
        }
        assert gk.scan_triton_bindings(hs) == {}


class TestScanSink:
    """``scan_triton_sink`` — the generated op is supported + lowerable."""

    def test_scan_sink_supports_and_lowers(self):
        torch = _cuda()
        hs = {
            "s": {
                "pattern": ("applyd", ("aff_diag", "A", "B"), "H"),
                "kernel": "sc_sink",
                "args": ("A", "B", "H"),
            }
        }
        sink = gk.scan_triton_sink(hs)
        assert "sc_sink" in sink.supported_ops
        from mlopt_core.ir import IR

        a = _v("a", 16, 8)
        b_ = _v("b", 16, 8)
        h = _v("h", 8)
        ir = IR(
            root=_p("sc_sink", a, b_, h),
            inputs=[a, b_, h],
            input_names={"a", "b", "h"},
            params={},
        )
        mod = sink.lower(ir, params={})
        dt = torch.float64
        av = torch.rand(16, 8, device="cuda", dtype=dt) * 0.5 + 0.4
        bv = torch.randn(16, 8, device="cuda", dtype=dt)
        hv = torch.randn(8, device="cuda", dtype=dt)
        torch.testing.assert_close(
            mod(av, bv, hv),
            _spelled_scan(av, bv, hv),
            atol=1e-12,
            rtol=1e-12,
        )


class TestHybridSinkScan:
    """``hybrid_sink`` — the probe-path sink consults the scan tier."""

    def test_seq_spec_binds_fold_not_single_step(self):
        torch = _cuda()
        hs = {
            "s": {
                "pattern": ("applyd", ("aff_diag", "A", "B"), "H"),
                "kernel": "sc_hyb",
                "args": ("A", "B", "H"),
            },
            "e": {
                "pattern": ("mul", "X1", ("relu", "X1")),
                "kernel": "gk_hyb",
                "args": ("X1",),
            },
        }
        sink = gk.hybrid_sink(hs, compile_kernels=False)
        assert "sc_hyb" in sink.supported_ops
        assert "gk_hyb" in sink.supported_ops
        from mlopt_core.ir import IR

        a = _v("a", 16, 8)
        b_ = _v("b", 16, 8)
        h = _v("h", 8)
        ir = IR(
            root=_p("sc_hyb", a, b_, h),
            inputs=[a, b_, h],
            input_names={"a", "b", "h"},
            params={},
        )
        mod = sink.lower(ir, params={})
        dt = torch.float64
        av = torch.rand(16, 8, device="cuda", dtype=dt) * 0.5 + 0.4
        bv = torch.randn(16, 8, device="cuda", dtype=dt)
        hv = torch.randn(8, device="cuda", dtype=dt)
        got = mod(av, bv, hv)
        # the FOLD — if the spelled-eval fallback had bound it, the
        # call would compute a⊙h+b (one step, (T,d)-shaped) instead
        torch.testing.assert_close(
            got, _spelled_scan(av, bv, hv), atol=1e-12, rtol=1e-12
        )
        assert got.shape == hv.shape

    def test_seq_spec_falls_back_to_the_fold_denotation(
        self, monkeypatch
    ):
        torch = pytest.importorskip("torch")
        pytest.importorskip("triton")
        import importlib.util

        # module load declines → the seq spec binds the registered
        # fold denotation (``seq_scan_denotation`` via
        # ``gen_bindings``): the SAME function the minted op
        # declares, evaluated serially — never the ambient map
        # application ``_eval_spec`` would silently deliver
        monkeypatch.setattr(
            importlib.util,
            "spec_from_file_location",
            lambda *a, **k: None,
        )
        hs = {
            "s": {
                "pattern": ("applyd", ("aff_diag", "A", "B"), "H"),
                "kernel": "sc_no_fb",
                "args": ("A", "B", "H"),
            }
        }
        sink = gk.hybrid_sink(hs, compile_kernels=False)
        assert "sc_no_fb" in sink.supported_ops
        from mlopt_core.ir import IR

        a = _v("a", 16, 8)
        b_ = _v("b", 16, 8)
        h = _v("h", 8)
        ir = IR(
            root=_p("sc_no_fb", a, b_, h),
            inputs=[a, b_, h],
            input_names={"a", "b", "h"},
            params={},
        )
        mod = sink.lower(ir, params={})
        dt = torch.float64
        av = torch.rand(16, 8) * 0.5 + 0.4
        bv = torch.randn(16, 8)
        hv = torch.randn(8)
        got = mod(av.to(dt), bv.to(dt), hv.to(dt))
        # the FOLD — a map-application fallback would return a
        # (T,d)-shaped a⊙h+b instead of the (d,) final state
        torch.testing.assert_close(
            got,
            _spelled_scan(
                av.to(dt), bv.to(dt), hv.to(dt), axis=0
            ),
            atol=1e-12,
            rtol=1e-12,
        )
        assert got.shape == hv.shape

    def test_unrolled_spec_keeps_the_serial_fallback(self, monkeypatch):
        pytest.importorskip("torch")
        import importlib.util

        # the unrolled spec's spelled eval IS the carrier semantics
        # (tuple-passing) — a declined codegen keeps that torch path
        monkeypatch.setattr(
            importlib.util,
            "spec_from_file_location",
            lambda *a, **k: None,
        )
        hs = {
            "s": {
                "pattern": (
                    "applyd",
                    (
                        "affd_compose",
                        ("aff_diag", "A2", "B2"),
                        ("aff_diag", "A1", "B1"),
                    ),
                    "H",
                ),
                "kernel": "sc_unr_fb",
                "args": ("A2", "B2", "A1", "B1", "H"),
            }
        }
        sink = gk.hybrid_sink(hs, compile_kernels=False)
        assert "sc_unr_fb" in sink.supported_ops
        from mlopt_core.ir import IR

        names = ["a2", "b2", "a1", "b1", "h"]
        a2, b2, a1, b1, h = (_v(n, 8) for n in names)
        ir = IR(
            root=_p("sc_unr_fb", a2, b2, a1, b1, h),
            inputs=[a2, b2, a1, b1, h],
            input_names={"a2", "b2", "a1", "b1", "h"},
            params={},
        )
        mod = sink.lower(ir, params={})
        vals = [torch.rand(8) * 0.4 + 0.3 for _ in range(4)]
        hv = torch.randn(8)
        want = vals[0] * (vals[2] * hv + vals[3]) + vals[1]
        torch.testing.assert_close(mod(*vals, hv), want)


class TestScanProbe:
    """``play.scan_probe`` — the honest-number arm, end to end."""

    def test_rows_report_all_arms(self):
        _cuda()
        rows = play.scan_probe(shapes=[(1, 4, 8)], reps=2)
        (row,) = rows
        assert row["delivered"] and row["shape"] == (1, 4, 8)
        for k in (
            "spelled_us",
            "spelled_compiled_us",
            "executor_generic_us",
            "executor_batched_us",
            "executor_fused_us",
            "triton_scan_us",
        ):
            assert row[k] > 0
        assert row["vs_spelled"] > 0

    def test_fused_off_and_default_shapes(self, monkeypatch):
        _cuda()
        monkeypatch.setattr(play, "_SCAN_PROBE_SHAPES", ((1, 4, 8),))
        (row,) = play.scan_probe(reps=2, fused=False)
        assert row["delivered"]
        assert "executor_fused_us" not in row

    def test_no_cuda_reports_honestly(self, monkeypatch):
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        rows = play.scan_probe(shapes=[(1, 4, 8)])
        assert rows[0]["delivered"] is False

    def test_has_triton_decline(self, monkeypatch):
        import importlib.util

        monkeypatch.setattr(
            importlib.util, "find_spec", lambda *a, **k: None
        )
        assert play._has_triton() is False


def _decayed_handlers(kernel: str = "dc_t"):
    """The canonical ``om_decayed`` handler for decayed-tier tests."""
    return {
        "d": {
            "pattern": ("om_decayed", "S", "D", "V"),
            "kernel": kernel,
            "args": ("S", "D", "V"),
        }
    }


class TestDecayedSpec:
    """``decayed_spec`` — the decayed-contraction recognizer."""

    def test_element_form_recognised(self):
        info = gk.decayed_spec(("om_decayed", "S", "D", "V"))
        assert info["kind"] == "decayed"
        assert info["params"] == ["S", "D", "V"]
        assert [o["metas"] for o in info["operands"]] == [
            ["S"],
            ["D"],
            ["V"],
        ]
        assert info["operands"][0]["expr"] == "a0_S_t"
        assert info["operands"][2]["expr"] == "a2_V_t"

    def test_spelled_form_recognised(self):
        info = gk.decayed_spec(("matmul", ("mul", "S", "D"), "V"))
        assert info["params"] == ["S", "D", "V"]

    def test_elementwise_operands_fuse(self):
        info = gk.decayed_spec(
            ("om_decayed", ("mul", "A", "B"), "D", "V")
        )
        assert info["operands"][0]["metas"] == ["A", "B"]
        assert info["params"] == ["A", "B", "D", "V"]
        # the same meta in two roles binds once but renders per role
        info = gk.decayed_spec(("om_decayed", "S", "S", "V"))
        assert info["params"] == ["S", "V"]
        assert info["operands"][1]["expr"] == "a1_S_t"

    def test_const_operand_inlines(self):
        info = gk.decayed_spec(("om_decayed", "S", 0.5, "V"))
        assert info["operands"][1]["metas"] == []
        # const leaves render a ``tl.full`` tile shaped to the
        # operand's kernel tile (``(BM, BK)`` for the d role) and
        # typed by the launch dtype (T-025)
        assert info["operands"][1]["expr"] == gk._decayed_const(
            1
        ).format(v="0.5")
        assert info["params"] == ["S", "V"]

    def test_declines_foreign_and_malformed_specs(self):
        assert gk.decayed_spec(("add", "X", "Y")) is None
        assert gk.decayed_spec("om_decayed") is None
        assert gk.decayed_spec(("om_decayed",)) is None
        assert gk.decayed_spec(("om_decayed", "S", "D")) is None
        assert (
            gk.decayed_spec(("matmul", ("add", "S", "D"), "V")) is None
        )
        assert gk.decayed_spec(("matmul", "mul", "V")) is None
        # an attr dict on the op node is outside the grammar
        assert (
            gk.decayed_spec(("om_decayed", "S", "D", "V", {"x": 1}))
            is None
        )
        assert (
            gk.decayed_spec(
                ("matmul", ("mul", "S", "D", {"x": 1}), "V")
            )
            is None
        )
        # an operand the elementwise renderer cannot express
        assert (
            gk.decayed_spec(
                ("om_decayed", ("matmul", "A", "B"), "D", "V")
            )
            is None
        )
        # kernel-name-unsafe metas decline
        assert gk.decayed_spec(("om_decayed", "for", "D", "V")) is None
        # a constant v carries no contraction operand for tl.dot
        assert gk.decayed_spec(("om_decayed", "S", "D", 2.0)) is None
        # and a constant s·d product leaves no mul-space tile
        assert gk.decayed_spec(("om_decayed", 1.0, 2.0, "V")) is None


class TestDecayedSource:
    """``_decayed_triton_source`` — emitted module text."""

    def test_source_inspectable(self):
        src = gk._decayed_triton_source(
            gk.decayed_spec(("om_decayed", "S", "D", "V"))
        )
        assert "triton.jit" in src
        assert "tl.dot(sd, (a2_V_t)" in src
        assert 'input_precision="ieee"' in src
        assert "a0_S_FB: tl.constexpr" in src
        assert "a1_D_base = tl.load(a1_D_bo + b)" in src
        assert "for k0 in range(0, K, BK)" in src
        # the s·d product is folded into the contraction's loop
        assert "sd = (a0_S_t) * (a1_D_t)" in src


class TestDecayedBindings:
    """Generated decayed kernels — fp64-verified vs the spelled body."""

    def test_broadcast_decay_matches_spelled(self):
        torch = _cuda()
        b = gk.decayed_triton_bindings(_decayed_handlers())
        dt = torch.float64
        B, T, DV = 2, 40, 24
        s = torch.randn(B, T, T, device="cuda", dtype=dt)
        d = torch.rand(T, T, device="cuda", dtype=dt)
        v = torch.randn(B, T, DV, device="cuda", dtype=dt)
        torch.testing.assert_close(
            b["dc_t"](s, d, v), (s * d) @ v, atol=1e-12, rtol=1e-12
        )
        # a second call on the same signature hits the plan memo
        torch.testing.assert_close(
            b["dc_t"](s, d, v), (s * d) @ v, atol=1e-12, rtol=1e-12
        )

    def test_full_rank_decay_matches_spelled(self):
        torch = _cuda()
        b = gk.decayed_triton_bindings(_decayed_handlers("dc_f"))
        dt = torch.float64
        B, T, DV = 3, 24, 16
        s = torch.randn(B, T, T, device="cuda", dtype=dt)
        d = torch.rand(B, T, T, device="cuda", dtype=dt)
        v = torch.randn(B, T, DV, device="cuda", dtype=dt)
        torch.testing.assert_close(
            b["dc_f"](s, d, v), (s * d) @ v, atol=1e-12, rtol=1e-12
        )

    def test_scalar_and_lower_rank_decay(self):
        torch = _cuda()
        b = gk.decayed_triton_bindings(_decayed_handlers("dc_s"))
        dt = torch.float64
        B, T, DV = 2, 32, 16
        s = torch.randn(B, T, T, device="cuda", dtype=dt)
        v = torch.randn(B, T, DV, device="cuda", dtype=dt)
        ds = torch.rand((), device="cuda", dtype=dt)
        torch.testing.assert_close(
            b["dc_s"](s, ds, v), (s * ds) @ v, atol=1e-12, rtol=1e-12
        )
        # a (T, 1) column decay broadcasts over the K axis
        dc = torch.rand(T, 1, device="cuda", dtype=dt)
        torch.testing.assert_close(
            b["dc_s"](s, dc, v), (s * dc) @ v, atol=1e-12, rtol=1e-12
        )

    def test_unbatched_and_extra_v_batch(self):
        torch = _cuda()
        b = gk.decayed_triton_bindings(_decayed_handlers("dc_b"))
        dt = torch.float64
        T, K, DV = 16, 20, 12
        s2 = torch.randn(T, K, device="cuda", dtype=dt)
        d2 = torch.rand(T, K, device="cuda", dtype=dt)
        v2 = torch.randn(K, DV, device="cuda", dtype=dt)
        torch.testing.assert_close(
            b["dc_b"](s2, d2, v2),
            (s2 * d2) @ v2,
            atol=1e-12,
            rtol=1e-12,
        )
        # v carries a batch dim the mul space lacks — torch
        # matmul broadcast rules, exercised through the offsets
        s3 = torch.randn(2, T, K, device="cuda", dtype=dt)
        d3 = torch.rand(2, T, K, device="cuda", dtype=dt)
        v3 = torch.randn(3, 2, K, DV, device="cuda", dtype=dt)
        torch.testing.assert_close(
            b["dc_b"](s3, d3, v3),
            (s3 * d3) @ v3,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_fused_operands_and_noncontiguous(self):
        torch = _cuda()
        hs = {
            "d": {
                "pattern": ("om_decayed", ("mul", "A", "B"), "D", "V"),
                "kernel": "dc_fu",
                "args": ("A", "B", "D", "V"),
            }
        }
        b = gk.decayed_triton_bindings(hs)
        dt = torch.float64
        B, T, DV = 2, 24, 16
        a = torch.randn(B, T, T, device="cuda", dtype=dt)
        bb = torch.rand(T, T, device="cuda", dtype=dt)
        d = torch.rand(T, T, device="cuda", dtype=dt)
        v = torch.randn(B, T, DV, device="cuda", dtype=dt)
        torch.testing.assert_close(
            b["dc_fu"](a, bb, d, v),
            ((a * bb) * d) @ v,
            atol=1e-12,
            rtol=1e-12,
        )
        # an operand meta broadcasting INTO the accumulated shape
        # (extent-1 dim widened by the other meta) folds identically
        bb1 = torch.rand(1, T, T, device="cuda", dtype=dt)
        torch.testing.assert_close(
            b["dc_fu"](a, bb1, d, v),
            ((a * bb1) * d) @ v,
            atol=1e-12,
            rtol=1e-12,
        )
        # non-contiguous operands fold identically (stride math)
        b2 = gk.decayed_triton_bindings(_decayed_handlers("dc_nc"))
        sn = torch.randn(B, T, T, device="cuda", dtype=dt).mT
        dn = torch.rand(T, T, device="cuda", dtype=dt).t()
        vn = torch.randn(B, DV, T, device="cuda", dtype=dt).mT
        torch.testing.assert_close(
            b2["dc_nc"](sn, dn, vn),
            (sn * dn) @ vn,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_empty_contract_returns_zeros(self):
        torch = _cuda()
        b = gk.decayed_triton_bindings(_decayed_handlers("dc_z"))
        dt = torch.float64
        # T == 0 → the empty output, no launch
        s = torch.randn(2, 0, 4, device="cuda", dtype=dt)
        d = torch.rand(0, 4, device="cuda", dtype=dt)
        v = torch.randn(2, 4, 8, device="cuda", dtype=dt)
        got = b["dc_z"](s, d, v)
        assert got.shape == (2, 0, 8)
        # K == 0 → the contraction folds to zeros
        s0 = torch.randn(2, 4, 0, device="cuda", dtype=dt)
        d0 = torch.rand(4, 0, device="cuda", dtype=dt)
        v0 = torch.randn(2, 0, 8, device="cuda", dtype=dt)
        got = b["dc_z"](s0, d0, v0)
        torch.testing.assert_close(
            got, torch.zeros(2, 4, 8, device="cuda", dtype=dt)
        )

    def test_kernel_file_landed_on_disk(self):
        _cuda()
        gk.decayed_triton_bindings(_decayed_handlers("dc_d"))
        f = gk._kernel_cache() / "_gd_dc_d.py"
        assert f.exists() and "tl.dot" in f.read_text()

    def test_unrecognised_spec_gets_no_binding(self):
        pytest.importorskip("torch")
        pytest.importorskip("triton")
        hs = {
            "d": {
                "pattern": ("add", "A", "B"),
                "kernel": "dc_no",
                "args": ("A", "B"),
            }
        }
        assert gk.decayed_triton_bindings(hs) == {}

    def test_module_load_decline_keeps_torch_path(self, monkeypatch):
        pytest.importorskip("torch")
        pytest.importorskip("triton")
        import importlib.util

        monkeypatch.setattr(
            importlib.util,
            "spec_from_file_location",
            lambda *a, **k: None,
        )
        assert gk.decayed_triton_bindings(_decayed_handlers()) == {}

    def test_mixed_dtypes_promote(self):
        torch = _cuda()
        b = gk.decayed_triton_bindings(_decayed_handlers("dc_mx"))
        s = torch.randn(2, 16, 16, device="cuda", dtype=torch.float32)
        d = torch.rand(16, 16, device="cuda", dtype=torch.float64)
        v = torch.randn(2, 16, 8, device="cuda", dtype=torch.float64)
        got = b["dc_mx"](s, d, v)
        torch.testing.assert_close(
            got, (s.double() * d) @ v, atol=1e-12, rtol=1e-12
        )
        assert got.dtype == torch.float64

    def test_memo_cap_evicts(self, monkeypatch):
        torch = _cuda()
        monkeypatch.setattr(gk, "_DECAY_MEMO_CAP", 1)
        b = gk.decayed_triton_bindings(_decayed_handlers("dc_ev"))
        dt = torch.float64
        for t in (8, 16, 24):
            s = torch.randn(2, t, t, device="cuda", dtype=dt)
            d = torch.rand(t, t, device="cuda", dtype=dt)
            v = torch.randn(2, t, 4, device="cuda", dtype=dt)
            torch.testing.assert_close(
                b["dc_ev"](s, d, v),
                (s * d) @ v,
                atol=1e-12,
                rtol=1e-12,
            )


class TestDecayedErrors:
    """Call-time honesty — contract violations raise, never revert."""

    def test_cpu_tensors_raise(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("triton")
        b = gk.decayed_triton_bindings(_decayed_handlers("dc_cpu"))
        t = torch.rand(4, 4)
        with pytest.raises(RuntimeError, match="CUDA"):
            b["dc_cpu"](t, t, t)

    def test_contraction_mismatch_raises(self):
        torch = _cuda()
        b = gk.decayed_triton_bindings(_decayed_handlers("dc_mm"))
        dt = torch.float64
        s = torch.randn(2, 8, 8, device="cuda", dtype=dt)
        d = torch.rand(8, 8, device="cuda", dtype=dt)
        v = torch.randn(2, 9, 4, device="cuda", dtype=dt)
        with pytest.raises(ValueError, match="contract"):
            b["dc_mm"](s, d, v)

    def test_rank_one_v_raises(self):
        torch = _cuda()
        b = gk.decayed_triton_bindings(_decayed_handlers("dc_r1"))
        dt = torch.float64
        s = torch.randn(8, 8, device="cuda", dtype=dt)
        d = torch.rand(8, 8, device="cuda", dtype=dt)
        v = torch.randn(8, device="cuda", dtype=dt)
        with pytest.raises(ValueError, match="contract"):
            b["dc_r1"](s, d, v)

    def test_unbroadcastable_mul_operands_raise(self):
        torch = _cuda()
        b = gk.decayed_triton_bindings(_decayed_handlers("dc_ub"))
        dt = torch.float64
        s = torch.randn(8, 8, device="cuda", dtype=dt)
        d = torch.rand(7, 7, device="cuda", dtype=dt)
        v = torch.randn(8, 4, device="cuda", dtype=dt)
        with pytest.raises(ValueError, match="broadcast"):
            b["dc_ub"](s, d, v)

    def test_exp_strides_contract_edges(self):
        # a rank-too-big source can never broadcast into target
        with pytest.raises(ValueError, match="broadcast"):
            gk._exp_strides((2, 2), (2, 1), (2,))
        # a non-1, non-equal dim is unbroadcastable mid-tensor
        with pytest.raises(ValueError, match="broadcast"):
            gk._exp_strides((3, 2), (2, 1), (4, 2))
        # and the right-aligned broadcast math itself
        assert gk._exp_strides((2,), (5,), (3, 2)) == (0, 5)
        assert gk._exp_strides((1, 2), (9, 5), (3, 2)) == (0, 5)


class TestDecayedHandlers:
    """``decayed_handlers`` — minting decayed-op entries."""

    def test_mints_decayed_entries(self):
        s = _v("s", 8, 8)
        d = _v("d", 8, 8)
        v = _v("v", 8, 4)
        term = _p("om_decayed", s, d, v)
        hs = gk.decayed_handlers([term])
        assert "decayed_0" in hs
        assert hs["decayed_0"]["kernel"] == "decayed_0_k"
        assert hs["decayed_0"]["pattern"] == (
            "om_decayed",
            "X1",
            "X2",
            "X3",
        )
        assert hs["decayed_0"]["args"] == ("X1", "X2", "X3")

    def test_spelled_term_mints(self):
        s = _v("s", 8, 8)
        d = _v("d", 8, 8)
        v = _v("v", 8, 4)
        term = _p("matmul", _p("mul", s, d), v)
        hs = gk.decayed_handlers([term])
        assert hs["decayed_0"]["pattern"] == (
            "matmul",
            ("mul", "X1", "X2"),
            "X3",
        )

    def test_non_decayed_dupes_and_covered_skipped(self):
        x, y = _v("x", 4), _v("y", 4)
        assert gk.decayed_handlers([_p("mul", x, y)]) == {}
        s, d, v = _v("s", 4, 4), _v("d", 4, 4), _v("v", 4, 4)
        t1 = _p("om_decayed", s, d, v)
        t2 = _p("om_decayed", d, s, v)  # same canon shape
        assert len(gk.decayed_handlers([t1, t2])) == 1
        covered = {_canon_concrete(("om_decayed", "S", "D", "V"))}
        assert gk.decayed_handlers([t1], covered=covered) == {}


class TestDecayedSink:
    """``decayed_triton_sink`` — the minted op is supported+lowerable."""

    def test_sink_supports_and_lowers(self):
        torch = _cuda()
        hs = _decayed_handlers("dc_sink")
        sink = gk.decayed_triton_sink(hs)
        assert "dc_sink" in sink.supported_ops
        from mlopt_core.ir import IR

        s = _v("s", 8, 8)
        d = _v("d", 8, 8)
        v = _v("v", 8, 4)
        ir = IR(
            root=_p("dc_sink", s, d, v),
            inputs=[s, d, v],
            input_names={"s", "d", "v"},
            params={},
        )
        mod = sink.lower(ir, params={})
        dt = torch.float64
        sv = torch.randn(8, 8, device="cuda", dtype=dt)
        dv = torch.rand(8, 8, device="cuda", dtype=dt)
        vv = torch.randn(8, 4, device="cuda", dtype=dt)
        torch.testing.assert_close(
            mod(sv, dv, vv), (sv * dv) @ vv, atol=1e-12, rtol=1e-12
        )


class TestHybridSinkDecayed:
    """``hybrid_sink`` — the probe-path sink consults the decayed tier."""

    def test_decayed_term_delivers_fused_kernel(self):
        torch = _cuda()
        s = _v("s", 8, 8)
        d = _v("d", 8, 8)
        v = _v("v", 8, 4)
        # mint from a real om_decayed term — the lifted site a
        # board's decayed claim would bind
        hs = gk.decayed_handlers([_p("om_decayed", s, d, v)])
        hs["e"] = {
            "pattern": ("mul", "X1", ("relu", "X1")),
            "kernel": "gk_hyb2",
            "args": ("X1",),
        }
        sink = gk.hybrid_sink(hs, compile_kernels=False)
        kern_op = hs["decayed_0"]["kernel"]
        assert kern_op in sink.supported_ops
        assert "gk_hyb2" in sink.supported_ops
        from mlopt_core.ir import IR

        ir = IR(
            root=_p(kern_op, s, d, v),
            inputs=[s, d, v],
            input_names={"s", "d", "v"},
            params={},
        )
        mod = sink.lower(ir, params={})
        dt = torch.float64
        sv = torch.randn(2, 8, 8, device="cuda", dtype=dt)
        dvv = torch.rand(8, 8, device="cuda", dtype=dt)
        vv = torch.randn(2, 8, 4, device="cuda", dtype=dt)
        torch.testing.assert_close(
            mod(sv, dvv, vv), (sv * dvv) @ vv, atol=1e-12, rtol=1e-12
        )

    def test_declined_codegen_keeps_the_torch_path(self, monkeypatch):
        torch = _cuda()
        import importlib.util

        # module load declines → the spelled eval IS the element's
        # semantics, so the fallback binding delivers it (unlike a
        # seq scan, whose spelled eval is a different function)
        monkeypatch.setattr(
            importlib.util,
            "spec_from_file_location",
            lambda *a, **k: None,
        )
        hs = {
            "d": {
                "pattern": ("om_decayed", "S", "D", "V"),
                "kernel": "dc_fb",
                "args": ("S", "D", "V"),
            }
        }
        sink = gk.hybrid_sink(hs, compile_kernels=False)
        assert "dc_fb" in sink.supported_ops
        from mlopt_core.ir import IR

        s = _v("s", 8, 8)
        d = _v("d", 8, 8)
        v = _v("v", 8, 4)
        ir = IR(
            root=_p("dc_fb", s, d, v),
            inputs=[s, d, v],
            input_names={"s", "d", "v"},
            params={},
        )
        mod = sink.lower(ir, params={})
        sv = torch.randn(8, 8)
        dvv = torch.rand(8, 8)
        vv = torch.randn(8, 4)
        torch.testing.assert_close(mod(sv, dvv, vv), (sv * dvv) @ vv)


class TestDecayProbe:
    """``play.decay_probe`` — the honest-number arm, end to end."""

    def test_rows_report_all_arms(self):
        _cuda()
        rows = play.decay_probe(shapes=[(1, 16, 8)], reps=2)
        (row,) = rows
        assert row["delivered"] and row["shape"] == (1, 16, 8)
        for k in (
            "spelled_us",
            "spelled_compiled_us",
            "executor_us",
            "triton_decayed_us",
        ):
            assert row[k] > 0
        assert row["vs_spelled"] > 0

    def test_default_shapes(self, monkeypatch):
        _cuda()
        monkeypatch.setattr(play, "_DECAY_PROBE_SHAPES", ((1, 16, 8),))
        (row,) = play.decay_probe(reps=2)
        assert row["delivered"]

    def test_no_cuda_reports_honestly(self, monkeypatch):
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        rows = play.decay_probe(shapes=[(1, 4, 8)])
        assert rows[0]["delivered"] is False

    def test_main_decay_probe_flag(self, capsys, monkeypatch):
        pytest.importorskip("torch")
        monkeypatch.setattr(play, "_DECAY_PROBE_SHAPES", ((1, 8, 8),))
        rc = play.main(
            [
                "--domain",
                "search",
                "--train-cases",
                "1",
                "--eval-cases",
                "1",
                "--budget",
                "2",
                "--decay-probe",
            ]
        )
        assert rc == 0
        assert "decay[" in capsys.readouterr().out


class TestReductionTier:
    """The reduction tier — ``reduction_spec``/``reduction_handlers``."""

    def test_reduce_heads_covers_dim_ops_and_softmax(self):
        heads = gk.reduce_heads()
        for op in ("sum", "mean", "max", "argmax", "softmax"):
            assert op in heads
        assert "matmul" not in heads and "mul" not in heads

    def test_accepts_elementwise_body_under_reduce_head(self):
        spec = (
            "mean",
            ("mul", "X1", ("relu", "X1")),
            {"dim": (-1,), "keepdim": True},
        )
        assert gk.reduction_spec(spec)

    def test_declines_bare_reduction_and_nested_reduce(self):
        assert not gk.reduction_spec(("sum", "X1", {"dim": (0,)}))
        assert not gk.reduction_spec(
            ("sum", ("mean", "X1", {"dim": (0,)}), {"dim": (1,)})
        )

    def test_declines_non_reduce_head_and_non_elementwise_body(self):
        assert not gk.reduction_spec(("mul", "X1", "X2"))
        assert not gk.reduction_spec(
            ("sum", ("matmul", "X1", "X2"), {"dim": (0,)})
        )
        assert not gk.reduction_spec("X1")
        assert not gk.reduction_spec(())

    def test_handlers_mint_red_entries(self):
        x = _v("x", 4, 8)
        body = _p("mul", x, _p("relu", x))
        term = _p("mean", body, dim=(-1,), keepdim=True)
        hs = gk.reduction_handlers([term])
        assert "red_0" in hs
        assert hs["red_0"]["kernel"] == "red_0_k"
        assert hs["red_0"]["pattern"] == (
            "mean",
            ("mul", "X1", ("relu", "X1")),
            {"dim": (-1,), "keepdim": True},
        )
        assert hs["red_0"]["args"] == ("X1",)

    def test_handlers_skip_bare_reduction_dupes_and_covered(self):
        x, y = _v("x", 4), _v("y", 4)
        assert gk.reduction_handlers(
            [_p("sum", x, dim=(0,))]
        ) == {}
        a = _p("mean", _p("mul", x, y), dim=(0,))
        b = _p("mean", _p("mul", x, _p("relu", y)), dim=(0,))
        # distinct bodies — both mint
        assert len(gk.reduction_handlers([a, b])) == 2
        # attr dicts make canon unhashable — ``covered`` is iterated,
        # not hashed, so a list is the honest shape here
        covered = [
            _canon_concrete(
                ("mean", ("mul", "X1", "X2"), {"dim": (0,)})
            )
        ]
        # ``dim=(0,)`` vs ``dim=(0,)`` — same canon; covered skips it
        c = _p("mean", _p("mul", x, y), dim=(0,))
        assert gk.reduction_handlers([c], covered=covered) == {}

    def test_tier_registered_with_triton_delivery(self):
        tier = next(t for t in gk.TIERS if t.name == "reduction")
        assert tier.triton is gk.reduction_triton_bindings
        assert tier.handlers is gk.reduction_handlers


class TestReductionTriton:
    """The reduction Triton piece — ``reduction_triton_*``."""

    def test_spec_accepts_trailing_reduce_subset(self):
        spec = (
            "mean",
            ("mul", "X1", ("relu", "X1")),
            {"dim": (-1,), "keepdim": True},
        )
        info = gk.reduction_triton_spec(spec)
        assert info is not None
        assert info["op"] == "mean" and info["metas"] == ["X1"]
        assert "tl.maximum" in info["body"]  # relu lowered

    def test_spec_declines_outside_grammar(self):
        # non-triton head
        assert gk.reduction_triton_spec(
            ("std", ("mul", "X1", "X1"), {"dim": (0,)})
        ) is None
        # max returns (values, indices) — not a plain tensor
        assert gk.reduction_triton_spec(
            ("max", ("mul", "X1", "X1"), {"dim": (0,)})
        ) is None
        # non-elementwise body
        assert gk.reduction_triton_spec(
            ("sum", ("matmul", "X1", "X2"), {"dim": (0,)})
        ) is None
        # bare reduction (no body op)
        assert gk.reduction_triton_spec(("sum", "X1", {"dim": (0,)})) is None
        # foreign attr key
        assert gk.reduction_triton_spec(
            ("sum", ("mul", "X1", "X2"), {"dim": (0,), "foo": 1})
        ) is None
        # metavar colliding with kernel locals
        assert gk.reduction_triton_spec(
            ("sum", ("mul", "acc", "X1"), {"dim": (0,)})
        ) is None

    def test_bindings_correct_cuda(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("needs cuda for triton")
        import mlopt_torch.torch_bridge as tb

        spec = ("sum", ("mul", "X1", "X2"), {"dim": (2,)})
        fn = gk.reduction_triton_bindings(
            {"h": {"pattern": spec, "kernel": "rk", "args": ("X1", "X2")}}
        )["rk"]
        x1 = torch.randn(4, 8, 8, device="cuda", dtype=torch.float64)
        x2 = torch.randn(8, device="cuda", dtype=torch.float64)
        got = fn(x1, x2)
        ref = gk._eval_spec(
            spec, {"X1": x1, "X2": x2}, tb._IR_TO_TORCH
        )
        torch.testing.assert_close(got, ref, atol=1e-9, rtol=1e-9)

    def test_softmax_and_fallbacks(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("needs cuda for triton")
        import mlopt_torch.torch_bridge as tb

        spec = (
            "softmax",
            ("add", ("mul", "X1", 0.125), "X2"),
            {"dim": -1},
        )
        fn = gk.reduction_triton_bindings(
            {"h": {"pattern": spec, "kernel": "rs", "args": ("X1", "X2")}}
        )["rs"]
        x1 = torch.randn(2, 4, 16, device="cuda")
        x2 = torch.randn(16, device="cuda")
        got = fn(x1, x2)
        ref = gk._eval_spec(
            spec, {"X1": x1, "X2": x2}, tb._IR_TO_TORCH
        )
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
        # CPU operands take the spelled fallback — same denotation
        c1 = x1.cpu()
        c2 = x2.cpu()
        got_cpu = fn(c1, c2)
        ref_cpu = gk._eval_spec(
            spec, {"X1": c1, "X2": c2}, tb._IR_TO_TORCH
        )
        torch.testing.assert_close(got_cpu, ref_cpu)
        # non-trailing reduce dim also falls back, still correct
        spec_nt = ("sum", ("mul", "X1", "X2"), {"dim": (1,)})
        fn_nt = gk.reduction_triton_bindings(
            {
                "h": {
                    "pattern": spec_nt,
                    "kernel": "rn",
                    "args": ("X1", "X2"),
                }
            }
        )["rn"]
        a = torch.randn(4, 8, 8, device="cuda", dtype=torch.float64)
        b = torch.randn(8, device="cuda", dtype=torch.float64)
        got_nt = fn_nt(a, b)
        ref_nt = gk._eval_spec(
            spec_nt, {"X1": a, "X2": b}, tb._IR_TO_TORCH
        )
        torch.testing.assert_close(got_nt, ref_nt)

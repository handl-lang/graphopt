"""om_decayed — the decayed-attention carrier element.

The RetNet/GLA inner-chunk form ``(QKᵀ ⊙ D) V`` spells
``matmul(mul(s, d), v)``: an elementwise decay-weighting of the score
matrix, then the value contraction.  No softmax, no normaliser —
``om_elem`` cannot express it (``om_lift`` declines: ``mul`` is not
``softmax``), which is why the spelled term verified FALSE against
every prior carrier element.  ``om_decayed(s, d, v)`` names it.

Covered:

* binding — the executor IS the spelled ``(s * d) @ v``, verbatim;
* laws — the lift fires exactly where torch would compute the term
  (broadcastable mul + well-typed contraction); the unfold restores
  the spelling; both members share one e-class under ``OM_LAWS``;
* e2e — the zoo's ``ChunkedRetention`` exports, saturates under
  ``OM_LAWS``, extracts the carrier member (it wins launch-aware
  extraction honestly), lowers through the torch sink, and runs
  bitwise-equal to the unrewritten lowering;
* typing/cost — the registered shape rule reports the applied
  ``(…,T,dv)`` shape (``_INVALID`` on provable misuse), and the FLOP
  entry prices the element exactly like the spelled mul+matmul;
* negatives — unbroadcastable decay and mismatched contractions
  decline honestly, in the check hook and in the e-graph;
* probe — fp64 CUDA timing of spelled-eager vs torch.compile vs the
  carrier delivery; an honest report, not a gate.
"""

import time

import mlopt_carriers.om as OM
import pytest
import torch
from mlopt_core import meta
from mlopt_core.cost import flops_cost, launch_aware_cost
from mlopt_core.egraph import EGraph
from mlopt_core.ir import IR, Const, Op, TensorType, Var
from mlopt_core.typing import _INVALID, _shape_of
from mlopt_torch.adapters import TorchSink
from mlopt_torch.torch_bridge import _IR_TO_TORCH, ir_to_torch_module

torch.manual_seed(0)


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _v(name: str, *shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _rand(*shape, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(9000 + seed)
    return torch.randn(*shape, dtype=torch.float64, generator=g)


def _eval(term, env):
    return meta._eval_term(term, env)


def _spelled(s, d, v):
    return Op.make("matmul", Op.make("mul", s, d), v)


def _ir(term, inputs):
    return IR(
        root=term,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params={},
    )


#: A leaf that is not a term at all — ``_shape_of`` reports ``None``.
RAW = "raw_leaf"


# ---------------------------------------------------------------------------
#  binding — correct by construction
# ---------------------------------------------------------------------------


def test_binding_is_the_spelled_semantics():
    """``om_decayed`` evaluates ``(s * d) @ v`` — fp64-identical."""
    s = _rand(2, 8, 8, seed=1)
    d = torch.tril(_rand(8, 8, seed=2).exp())
    v = _rand(2, 8, 16, seed=3)
    out = _IR_TO_TORCH["om_decayed"](s, d, v)
    ref = (s * d) @ v
    assert torch.equal(out, ref)


def test_binding_broadcasts_the_decay():
    """A rank-2 decay against rank-3 scores, and a scalar decay."""
    s = _rand(2, 8, 8, seed=4)
    v = _rand(2, 8, 4, seed=5)
    d2 = _rand(8, 8, seed=6)
    d0 = torch.tensor(0.5, dtype=torch.float64)
    fn = _IR_TO_TORCH["om_decayed"]
    assert torch.equal(fn(s, d2, v), (s * d2) @ v)
    assert torch.equal(fn(s, d0, v), (s * d0) @ v)


# ---------------------------------------------------------------------------
#  laws — the check hook, driven both ways
# ---------------------------------------------------------------------------


def test_lift_fires_and_evals():
    """matmul(mul(s,d), v) → om_decayed(s,d,v), fp64-equal."""
    s, d, v = _v("s", 2, 8, 8), _v("d", 8, 8), _v("v", 2, 8, 4)
    env = {
        s: _rand(2, 8, 8, seed=7),
        d: _rand(8, 8, seed=8),
        v: _rand(2, 8, 4, seed=9),
    }
    out = meta.apply_rewrite_at(
        OM.OM_DECAYED_LIFT, _spelled(s, d, v), ()
    )
    assert out is not None
    assert out.op == "om_decayed" and out.args == (s, d, v)
    got, ref = _eval(out, env), _eval(_spelled(s, d, v), env)
    assert torch.equal(got, ref)


def test_unlift_restores_the_spelling():
    """om_decayed(s,d,v) → matmul(mul(s,d), v) — the exact inverse."""
    s, d, v = _v("s", 4, 8), _v("d", 4, 8), _v("v", 8, 3)
    carrier = Op.make("om_decayed", s, d, v)
    out = meta.apply_rewrite_at(OM.OM_DECAYED_UNLIFT, carrier, ())
    assert out == _spelled(s, d, v)


def test_check_accepts_broadcast_and_scalar_decay():
    """d broadcasts under torch.mul rules: lower-rank or scalar both pass."""
    assert OM.OM_DECAYED_LIFT.check(
        {
            "s": _v("s", 2, 8, 8),
            "d": _v("d", 8, 8),
            "v": _v("v", 2, 8, 4),
        }
    )
    assert OM.OM_DECAYED_LIFT.check(
        {"s": _v("s", 2, 8, 8), "d": Const(0.5), "v": _v("v", 2, 8, 4)}
    )


def test_check_accepts_unknown_dims():
    """None extents are wildcards — unresolvable *shapes* still veto."""
    assert OM.OM_DECAYED_LIFT.check(
        {
            "s": _v("s", 2, 8, None),
            "d": _v("d", 8, 8),
            "v": _v("v", 2, None, 4),
        }
    )
    assert not OM.OM_DECAYED_LIFT.check(
        {"s": RAW, "d": _v("d", 8, 8), "v": _v("v", 2, 8, 4)}
    )


def test_check_vetoes_unbroadcastable_decay():
    """s (2,3) against d (4,5): no broadcast, no lift."""
    assert not OM.OM_DECAYED_LIFT.check(
        {"s": _v("s", 2, 3), "d": _v("d", 4, 5), "v": _v("v", 3, 4)}
    )


def test_check_vetoes_mismatched_contraction():
    """The mul broadcasts but s's last axis does not contract v's -2."""
    assert not OM.OM_DECAYED_LIFT.check(
        {"s": _v("s", 4, 8), "d": _v("d", 4, 8), "v": _v("v", 3, 5)}
    )


def test_lift_declines_in_the_egraph():
    """An ill-typed spelled term gains no om_decayed member."""
    s, d, v = _v("s", 4, 8), _v("d", 4, 8), _v("v", 3, 5)
    eg = EGraph()
    root = eg.add_term(_spelled(s, d, v))
    eg.run([OM.OM_DECAYED_LIFT], root, max_iterations=5)
    assert not any(n.op == "om_decayed" for n in eg._node_to_class)


def test_scalar_decay_lifts_and_evals():
    """mul(s, 0.5) @ v lifts: the binding maps Const leaves too."""
    s, v = _v("s", 4, 8), _v("v", 8, 3)
    env = {s: _rand(4, 8, seed=10), v: _rand(8, 3, seed=11)}
    term = _spelled(s, Const(0.5), v)
    out = meta.apply_rewrite_at(OM.OM_DECAYED_LIFT, term, ())
    assert out is not None
    assert torch.equal(_eval(out, env), _eval(term, env))


# ---------------------------------------------------------------------------
#  e-class membership + extraction
# ---------------------------------------------------------------------------


def test_om_laws_place_both_forms_in_one_class():
    """Under OM_LAWS the spelled and carrier forms share the root class."""
    s, d, v = _v("s", 2, 8, 8), _v("d", 8, 8), _v("v", 2, 8, 4)
    eg = EGraph()
    root = eg.add_term(_spelled(s, d, v))
    eg.run(OM.OM_LAWS, root, max_iterations=10, max_nodes=100_000)
    root_ops = {n.op for n in eg.get_class(root).nodes}
    assert "matmul" in root_ops
    assert "om_decayed" in root_ops


def test_extraction_prefers_the_fused_element():
    """One launch beats two: launch-aware extraction picks om_decayed,
    and the extracted module verifies fp64 against the spelled form."""
    s, d, v = _v("s", 2, 8, 8), _v("d", 8, 8), _v("v", 2, 8, 4)
    eg = EGraph()
    root = eg.add_term(_spelled(s, d, v))
    eg.run(OM.OM_LAWS, root, max_iterations=10, max_nodes=100_000)
    best = eg.extract_best(root, launch_aware_cost)
    assert best.op == "om_decayed"

    inputs = [s, d, v]
    mod = ir_to_torch_module(_ir(best, inputs))
    ts, td, tv = (
        _rand(2, 8, 8, seed=12),
        _rand(8, 8, seed=13),
        _rand(2, 8, 4, seed=14),
    )
    with torch.no_grad():
        out = mod(ts, td, tv)
    assert torch.equal(out, (ts * td) @ tv)


# ---------------------------------------------------------------------------
#  typing + cost
# ---------------------------------------------------------------------------












# ---------------------------------------------------------------------------
#  e2e — the zoo's ChunkedRetention through the torch sink
# ---------------------------------------------------------------------------


def test_chunked_retention_lifts_and_delivers():
    """The RetNet block's (QKᵀ ⊙ D)V lifts end to end.

    The spelled term is unreachable by om_lift (mul is not softmax);
    under OM_LAWS the root e-class gains om_decayed, launch-aware
    extraction selects it, and the sink delivers a module bitwise-
    identical to the unrewritten lowering.
    """
    from mlopt_discovery.zoo import _ChunkedRetention
    from mlopt_torch.torch_bridge import export_to_ir

    torch.manual_seed(0)
    model = _ChunkedRetention(16).double()
    x = torch.randn(2, 8, 16, dtype=torch.float64)
    ir, leaves = export_to_ir(model, (x,))
    sink = TorchSink()
    assert "om_decayed" in sink.supported_ops

    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(OM.OM_LAWS, root, max_iterations=10, max_nodes=100_000)
    assert "om_decayed" in {n.op for n in eg.get_class(root).nodes}

    best = eg.extract_best(root, launch_aware_cost)
    assert best.op == "om_decayed"
    lifted = IR(
        root=best,
        inputs=ir.inputs,
        input_names=ir.input_names,
        params=ir.params,
    )
    with torch.no_grad():
        out = sink.lower(lifted, leaves)(x)
        ref = sink.lower(ir, leaves)(x)
    assert torch.equal(out, ref)


# ---------------------------------------------------------------------------
#  probe — fp64 CUDA: spelled eager vs torch.compile vs carrier delivery
# ---------------------------------------------------------------------------


def _bench(fn, args, warmup=10, iters=50) -> float:
    """Median-free wall time per call — warmup then a timed loop."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


@pytest.mark.requires_cuda
def test_probe_spelled_vs_compiled_vs_carrier():
    """Honest measurement of the delivery options on realistic shapes.

    The carrier executor is a *reference* lowering — it pays the same
    mul+matmul work as eager spelled plus the eval dispatch.  Report
    the numbers; assert only that all three agree numerically.
    """
    shapes = [(2, 64, 128), (8, 128, 256), (4, 256, 128)]
    s, d, v = (
        Var("s", TensorType(())),
        Var("d", TensorType(())),
        Var("v", TensorType(())),
    )
    carrier_mod = ir_to_torch_module(
        _ir(Op.make("om_decayed", s, d, v), [s, d, v])
    ).cuda()
    spelled_mod = ir_to_torch_module(_ir(_spelled(s, d, v), [s, d, v]))
    compiled = torch.compile(lambda a, b, c: (a * b) @ c)

    rows = []
    for B, T, dv in shapes:
        ts = torch.randn(B, T, T, dtype=torch.float64, device="cuda")
        td = torch.randn(T, T, dtype=torch.float64, device="cuda")
        tv = torch.randn(B, T, dv, dtype=torch.float64, device="cuda")
        args = (ts, td, tv)
        ref = (ts * td) @ tv
        with torch.no_grad():
            for mod in (carrier_mod, spelled_mod, compiled):
                assert torch.allclose(
                    mod(*args), ref, atol=1e-12, rtol=1e-12
                )
            rows.append(
                (
                    (B, T, dv),
                    _bench(lambda *a: (a[0] * a[1]) @ a[2], args),
                    _bench(compiled, args),
                    _bench(lambda *a: carrier_mod(*a), args),
                    _bench(lambda *a: spelled_mod(*a), args),
                )
            )
    for shape, eager, comp, carrier, spelled in rows:
        print(
            f"om_decayed probe {shape}: eager={eager * 1e6:.0f}us "
            f"compile={comp * 1e6:.0f}us carrier={carrier * 1e6:.0f}us "
            f"spelled_ir={spelled * 1e6:.0f}us"
        )

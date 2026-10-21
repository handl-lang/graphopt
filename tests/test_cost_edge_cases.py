"""Coverage-gap tests for mlopt_core.cost.

``tests/test_cost.py`` covers the flagship semantics — broadcasting,
weight dedup, the storage cost axis, the e-graph end-to-end.  This file
drives the per-op ``_flops_of`` arms and the structural edges those
never reach:

* ``_flops_of`` fallbacks: matmul with an unshapeable RHS, conv2d with
  ``groups > 1`` and with non-int weight dims, ``aff_compose`` /
  ``apply`` on short-rank map parts, ``sdpa`` on a rank-2 query;
* ``depth_cost``/``depth_cost_for`` on a provably ill-typed term (the
  launch-not-veto policy) and ``_local_roofline``'s ``_INVALID`` arm;
* ``param_bytes_cost`` storage semantics: numpy ``.size`` and missing
  ``element_size`` fallbacks in ``_param_numel``, ``_folds_to_param``
  on non-``Op`` terms and the ``allow_const=False`` matmul/concat
  rule, ``_fold_ewidth``'s unresolvable-leaf 4.0 default and its
  non-tensor ``None`` leaves, param-only folds billed per
  materialisation (``concat`` re-stores copies; nested folds bill only
  the outermost output), ``by_bytes`` dtype widths;
* ``dag_cost`` on a shared-subterm DAG, the param-only compile-time
  discount and its ``charges_param_only`` opt-out, the ``dag_exact``
  early return (direct and through a ``functools.partial``), and a
  cost fn without a ``memo`` parameter;
* ``_is_strided`` on a non-tuple shape, ``count_cost``/``depth_cost``
  on carrier + view ops;
* ``roofline_cost_for`` keyword calibration and ``CostModel``'s
  matmul/linear arms including the scalar-operand fallbacks.

The one branch listed as unreachable by construction is in
``test_defensive_branch_inventory``.
"""

import functools
import types

import numpy as np
import pytest
import torch

from mlopt_core.cost import (
    _INVALID_COST,
    _LAUNCH_S,
    CostModel,
    _flops_of,
    _fold_ewidth,
    _folds_to_param,
    _is_strided,
    _local_roofline,
    _param_index,
    count_cost,
    dag_cost,
    depth_cost,
    depth_cost_for,
    flops_cost,
    launch_aware_cost,
    param_bytes_cost,
    param_bytes_cost_for,
    roofline_cost,
    roofline_cost_for,
)
from mlopt_core.ir import Const, Op, Param, TensorType, Var
from mlopt_core.typing import _INVALID, _shape_of


def _v(name: str, *shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(name: str, *shape) -> Param:
    return Param(name, TensorType(tuple(shape)))


#: An op whose inferred shape is ``None`` — a carrier-internal ``()``
#: member read as a tensor.  Used wherever an unshapeable operand is
#: needed to reach a fallback arm.
UNKNOWN = Op.make("transpose", Const(1.0))


# ---------------------------------------------------------------------------
#  _flops_of — per-op arms and their fallbacks
# ---------------------------------------------------------------------------






def _numel_of(t) -> int:
    from mlopt_core.typing import _numel

    return _numel(_shape_of(t))










# ---------------------------------------------------------------------------
#  view / carrier ops under the structural models
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
#  depth_cost / _local_roofline — invalid terms
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
#  param_bytes_cost — storage semantics
# ---------------------------------------------------------------------------


def test_param_numel_source_tensor_variants():
    x = _v("x", 4, 8)
    W = _p("W", 8, 8)
    t = Op.make("linear", x, W)
    # numpy arrays: .size is an int attribute (no .numel method).
    assert param_bytes_cost(t, {"W": np.zeros((3, 4))}) == 12.0

    # objects exposing numel() but no element_size(): by_bytes uses
    # the fp32 default of 4 bytes.
    class FakeTensor:
        def numel(self):
            return 10

    assert param_bytes_cost(t, {"W": FakeTensor()}) == 10.0
    assert (
        param_bytes_cost(t, {"W": FakeTensor()}, by_bytes=True) == 40.0
    )
    # A source object with neither a callable numel() nor a numeric
    # .size falls back to the declared TensorType.
    odd = types.SimpleNamespace(numel=None, size="not-a-number")
    assert param_bytes_cost(t, {"W": odd}) == 64.0
    # torch dtype widths under by_bytes.
    assert (
        param_bytes_cost(
            t, {"W": torch.zeros(8, dtype=torch.float16)}, by_bytes=True
        )
        == 16.0
    )
    assert (
        param_bytes_cost(
            t, {"W": torch.zeros(8, dtype=torch.int8)}, by_bytes=True
        )
        == 8.0
    )














def test_param_bytes_fold_ewidth():
    """_fold_ewidth: widest resolvable leaf dtype; 4.0 default; Const
    leaves contribute no width."""
    P = _p("P", 4)
    Q = _p("Q", 4)
    # No source_tensors -> fp32 default per leaf.
    assert _fold_ewidth(P, None) == 4.0
    assert _fold_ewidth(Op.make("add", P, Q), None) == 4.0
    # Consts and other non-tensor leaves report None.
    assert _fold_ewidth(Const(1.0), None) is None
    assert _fold_ewidth("raw_leaf", None) is None
    mul = Op.make("mul", P, Const(2.0))
    assert _fold_ewidth(mul, None) == 4.0
    # All-Const fold: no widths at all -> None -> caller defaults.
    cc = Op.make("add", Const(1.0), Const(2.0))
    assert _fold_ewidth(cc, None) is None
    # ...and by_bytes still prices it at fp32 width.
    assert param_bytes_cost(cc, None, by_bytes=True) == pytest.approx(
        _numel_of(cc) * 4.0
    )
    # The widest leaf wins (fp16's 2 bytes over int8's 1).
    src = {
        "P": torch.zeros(4, dtype=torch.float16),
        "Q": torch.zeros(4, dtype=torch.int8),
    }
    cat = Op.make("concat", P, Q, dim=0)
    assert param_bytes_cost(cat, src, by_bytes=True) == 8 * 2.0
    # A leaf whose tensor has no element_size() (numpy's attr is
    # named ``itemsize``) defaults to the fp32 width — and that leaf
    # still resolves, so the subtree still folds.
    mul = Op.make("mul", P, Const(2.0))
    assert (
        param_bytes_cost(mul, {"P": np.zeros(4)}, by_bytes=True)
        == 4.0 * 4.0
    )


def test_param_bytes_for_closure_markers():
    fn = param_bytes_cost_for({"W": torch.zeros(4, 4)})
    assert fn.__name__ == "param_bytes_cost_for"
    assert fn.charges_param_only is True
    assert fn.dag_exact is True
    W = _p("W", 4, 4)
    assert fn(Op.make("mul", W, Const(2.0))) == 16.0


# ---------------------------------------------------------------------------
#  dag_cost — shared DAG, param-only discount, exact models
# ---------------------------------------------------------------------------






def test_dag_cost_dag_exact_partial_and_markers():
    """functools.partial bindings unwrap .func for both markers."""
    W = _p("W", 4, 4)
    cat = Op.make("concat", W, W, dim=0)
    t = Op.make("add", cat, cat)
    src = {"W": torch.zeros(4, 4)}
    bound = functools.partial(param_bytes_cost, source_tensors=src)
    assert dag_cost(t, bound) == pytest.approx(param_bytes_cost(t, src))
    # charges_param_only via .func: a param-only subtree stays billed.
    only = Op.make("matmul", _p("A", 4, 4), _p("B", 4, 4))
    assert dag_cost(only, bound) == pytest.approx(
        param_bytes_cost(only, src)
    )




# ---------------------------------------------------------------------------
#  _is_strided / _bytes_of edges
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
#  CostModel — configurable weighted model
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
#  defensive branches believed unreachable — pragma candidates
# ---------------------------------------------------------------------------



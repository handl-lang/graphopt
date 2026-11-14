"""Torch bridge — FX lift, lowering tables and eval, re-exported.

The implementation was split out of this module (arch audit — the
~4.5k-line god-module):

* :mod:`mlopt_torch.torch_export` — ``torch.export`` → IR lift
  (``export_to_ir``, FX node helpers, ``opaque`` leaf minting).
* :mod:`mlopt_torch.torch_bindings` — the aten classification tables
  (``_ATEN_TO_IR`` / ``_IR_TO_TORCH_EXTRA`` /
  ``_LOCAL_ATTR_SCHEMA_OVERRIDES`` / ``_SCALAR_OPERAND_OPS_OVERRIDES``
  merged over ``aten_classify``'s generated tables), the
  ``_xxx_torch`` binding implementations, ``_CORE_TORCH_BINDINGS``,
  the ambient ``_IR_TO_TORCH`` and the shape rules.
* :mod:`mlopt_torch.torch_eval` — ``eval_term``, ``_EvalPlan``,
  ``IRModule`` / ``ir_to_torch_module``.

This module re-exports the full historical surface so every existing
``mlopt_torch.torch_bridge.X`` import — including the private names
tests and carriers reach for (``_IR_TO_TORCH``, ``_CORE_TORCH_BINDINGS``,
``_om_elem``, ``_opaque_torch`` …) — keeps working unchanged.

It is also the **monkeypatch seam**: names tests patch
(``_om_elem`` / ``_om_compose`` / ``_resolve_opaque_callable``) are
resolved through this module at call time by the ``_*_shim``
dispatchers in :mod:`mlopt_torch.torch_bindings`, so
``monkeypatch.setattr(torch_bridge, ...)`` keeps reaching the real
implementations exactly as before the split.
"""

from __future__ import annotations

# -- third-party names the historical module also exposed ---------------
import contextlib
import functools
import importlib
from collections.abc import Callable, Mapping
from typing import (
    Any,
    cast,
)

import torch
from mlopt_core.attrs import (
    ATTR_SCHEMA,
    attr_of,
    is_positional_attr,
)
from mlopt_core.ir import (
    IR,
    Const,
    Op,
    Param,
    TensorType,
    Var,
)
from mlopt_core.opmeta import FOLD_OPS, REGISTRY
from mlopt_core.ops import (
    OpTable,
    carrier_torch_bindings,
    register_ambient_bindings,
    register_core_bindings,
)
from mlopt_core.typing import register_shape_rule
from mlopt_dtypes import canonical_name
from mlopt_dtypes import get as _dtype_info
from torch.nn.attention import SDPBackend, sdpa_kernel

from mlopt_torch.aten_classify import (
    ATEN_TO_IR as _GEN_ATEN_TO_IR,
)
from mlopt_torch.aten_classify import (
    ATTR_SCHEMA as _GEN_LOCAL_ATTR_SCHEMA,
)
from mlopt_torch.aten_classify import (
    SCALAR_OPERAND_OPS as _GEN_SCALAR_OPERAND_OPS,
)

# -- the split implementation, re-exported verbatim ---------------------
from mlopt_torch.torch_bindings import (
    _ATEN_TO_IR,
    _ATEN_TO_IR_OVERRIDES,
    _BOUND_OPS,
    _CORE_TORCH_BINDINGS,
    _IR_TO_TORCH,
    _IR_TO_TORCH_EXTRA,
    _LOCAL_ATTR_SCHEMA,
    _LOCAL_ATTR_SCHEMA_OVERRIDES,
    _SCALAR_OPERAND_OPS,
    _SCALAR_OPERAND_OPS_OVERRIDES,
    _SDPA_MATH_KV_LEN_MIN,
    _SDPA_MATH_Q_LEN_MAX,
    AmbientRegistrySnapshot,
    _adaptive_avg_pool2d_torch,
    _AmbientTorchBindings,
    _arange_torch,
    _avg_pool2d_shape,
    _avg_pool2d_torch,
    _batch_norm_torch,
    _cdist_shape,
    _clamp_torch,
    _creator_device,
    _creator_dtype,
    _deform_conv2d_torch,
    _diag_shape,
    _diagonal_shape,
    _dim_args,
    _expand_torch,
    _extreme_torch,
    _fft_kw,
    _index_put_torch,
    _index_torch,
    _instance_norm_torch,
    _linspace_shape,
    _logsumexp_shape,
    _lstm_input_torch,
    _om_compose,
    _om_compose_shim,
    _om_elem,
    _om_elem_shim,
    _opaque_torch,
    _pack_torch,
    _resolve_opaque,
    _rot90_torch,
    _scalar_value,
    _scatter_torch,
    _sdpa_math_backend,
    _sdpa_torch,
    _shape_attr_int,
    _shape_fn,
    _shape_op,
    _shape_pair,
    _split_sizes,
    _to_torch,
    _torchvision_deform_conv2d,
    _upsample_bilinear2d_torch,
    _upsample_nearest2d_torch,
    _where_torch,
    ambient_registry_scope,
    ambient_registry_snapshot,
    restore_ambient_registry,
)
from mlopt_torch.torch_eval import (
    _COMPLEX_DTYPES,
    _MISS,
    _UNPLANNED,
    IRModule,
    _cast_to_delivery,
    _const_like,
    _device_like,
    _EvalPlan,
    _mint_const,
    _randn_param,
    _sibling_like,
    eval_term,
    ir_to_torch_module,
)
from mlopt_torch.torch_export import (
    _FFT_SHAPE_OPS,
    _aten_name,
    _canon_aten_name,
    _fill_src,
    _graph_outputs,
    _graph_root,
    _handle_copy_,
    _infer_shape,
    _is_dotted_name,
    _opaque_encode,
    _opaque_qual,
    _opaque_term,
    _read_view,
    _resolve_attr,
    _resolve_opaque_callable,
    _scatter_view,
    _strip_backend_suffix,
    _view_chain,
    export_to_ir,
)

__all__ = [
    "ATTR_SCHEMA",
    "FOLD_OPS",
    "IR",
    "REGISTRY",
    "_ATEN_TO_IR",
    "_ATEN_TO_IR_OVERRIDES",
    "_BOUND_OPS",
    "_COMPLEX_DTYPES",
    "_CORE_TORCH_BINDINGS",
    "_FFT_SHAPE_OPS",
    "_GEN_ATEN_TO_IR",
    "_GEN_LOCAL_ATTR_SCHEMA",
    "_GEN_SCALAR_OPERAND_OPS",
    "_IR_TO_TORCH",
    "_IR_TO_TORCH_EXTRA",
    "_LOCAL_ATTR_SCHEMA",
    "_LOCAL_ATTR_SCHEMA_OVERRIDES",
    "_MISS",
    "_SCALAR_OPERAND_OPS",
    "_SCALAR_OPERAND_OPS_OVERRIDES",
    "_SDPA_MATH_KV_LEN_MIN",
    "_SDPA_MATH_Q_LEN_MAX",
    "_UNPLANNED",
    "AmbientRegistrySnapshot",
    "Any",
    "Callable",
    "Const",
    "IRModule",
    "Mapping",
    "Op",
    "OpTable",
    "Param",
    "SDPBackend",
    "TensorType",
    "Var",
    "_AmbientTorchBindings",
    "_EvalPlan",
    "__all__",
    "_adaptive_avg_pool2d_torch",
    "_arange_torch",
    "_aten_name",
    "_avg_pool2d_shape",
    "_avg_pool2d_torch",
    "_batch_norm_torch",
    "_canon_aten_name",
    "_cast_to_delivery",
    "_cdist_shape",
    "_clamp_torch",
    "_const_like",
    "_creator_device",
    "_creator_dtype",
    "_deform_conv2d_torch",
    "_device_like",
    "_diag_shape",
    "_diagonal_shape",
    "_dim_args",
    "_dtype_info",
    "_expand_torch",
    "_extreme_torch",
    "_fft_kw",
    "_fill_src",
    "_graph_outputs",
    "_graph_root",
    "_handle_copy_",
    "_index_put_torch",
    "_index_torch",
    "_infer_shape",
    "_instance_norm_torch",
    "_is_dotted_name",
    "_linspace_shape",
    "_logsumexp_shape",
    "_lstm_input_torch",
    "_mint_const",
    "_om_compose",
    "_om_compose_shim",
    "_om_elem",
    "_om_elem_shim",
    "_opaque_encode",
    "_opaque_qual",
    "_opaque_term",
    "_opaque_torch",
    "_pack_torch",
    "_randn_param",
    "_read_view",
    "_resolve_attr",
    "_resolve_opaque",
    "_resolve_opaque_callable",
    "_rot90_torch",
    "_scalar_value",
    "_scatter_torch",
    "_scatter_view",
    "_sdpa_math_backend",
    "_sdpa_torch",
    "_shape_attr_int",
    "_shape_fn",
    "_shape_op",
    "_shape_pair",
    "_sibling_like",
    "_split_sizes",
    "_strip_backend_suffix",
    "_to_torch",
    "_torchvision_deform_conv2d",
    "_upsample_bilinear2d_torch",
    "_upsample_nearest2d_torch",
    "_view_chain",
    "_where_torch",
    "ambient_registry_scope",
    "ambient_registry_snapshot",
    "attr_of",
    "canonical_name",
    "carrier_torch_bindings",
    "cast",
    "contextlib",
    "eval_term",
    "export_to_ir",
    "functools",
    "importlib",
    "ir_to_torch_module",
    "is_positional_attr",
    "register_ambient_bindings",
    "register_core_bindings",
    "register_shape_rule",
    "restore_ambient_registry",
    "sdpa_kernel",
    "torch",
]


def __getattr__(name: str) -> Any:
    """Fallback for names not explicitly re-exported above.

    Resolves against the three implementation modules' namespaces so
    any historical ``torch_bridge.<name>`` — including names bound by
    their own imports — keeps resolving.
    """
    import mlopt_torch.torch_bindings as _bindings
    import mlopt_torch.torch_eval as _eval
    import mlopt_torch.torch_export as _export

    for mod in (_export, _bindings, _eval):
        if name in mod.__dict__:
            return mod.__dict__[name]
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )

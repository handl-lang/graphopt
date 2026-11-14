"""FX graph → mlopt IR lift.

Split from ``torch_bridge.py``.  ``export_to_ir`` plus the FX-node helpers that canonicalise aten
spellings (``_canon_aten_name`` / ``_aten_name``), thread ``copy_``
mutations and view chains (``_handle_copy_`` & friends), resolve the
graph root/outputs, and mint ``opaque`` leaf terms for unbound ops.
The classification tables consulted here (``_ATEN_TO_IR`` …) live in
:mod:`mlopt_torch.torch_bindings`; the lowered-module side lives in
:mod:`mlopt_torch.torch_eval`.  ``mlopt_torch.torch_bridge`` re-exports
this module — import from there for the historical surface.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
from mlopt_core.attrs import ATTR_SCHEMA
from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var
from mlopt_dtypes import canonical_name

from mlopt_torch.torch_bindings import (
    _ATEN_TO_IR,
    _IR_TO_TORCH,
    _IR_TO_TORCH_EXTRA,
    _LOCAL_ATTR_SCHEMA,
    _SCALAR_OPERAND_OPS,
    _resolve_opaque,
)


def _strip_backend_suffix(name: str) -> str:
    for suffix in (
        ".default",
        ".deterministic",
        ".backend",
        ".assigned",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _canon_aten_name(name: str) -> str:
    """Map a raw ATen name to a mlopt generator, handling overloads."""
    # Namespace-qualified spellings ('aten.min.other', 'aten::min.other')
    # canonicalise like their bare ``__name__`` twins ('min.other') —
    # the overload suffix is load-bearing (see ``_ATEN_TO_IR``).
    for prefix in ("aten::", "aten."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name in _ATEN_TO_IR:
        return _ATEN_TO_IR[name]
    stripped = _strip_backend_suffix(name)
    if stripped in _ATEN_TO_IR:
        return _ATEN_TO_IR[stripped]
    if name in _IR_TO_TORCH_EXTRA:
        return _IR_TO_TORCH_EXTRA[name]
    # Generic overload stripping: 'flatten.using_ints' → 'flatten',
    # 'to.dtype' → 'to'.  Try the base packet name against both maps.
    base = name.split(".")[0]
    if base in _ATEN_TO_IR:
        return _ATEN_TO_IR[base]
    if (
        base in _IR_TO_TORCH_EXTRA
    ):  # pragma: no cover — every EXTRA key contains a dot
        return _IR_TO_TORCH_EXTRA[base]
    return base if base in _IR_TO_TORCH else stripped


def _aten_name(target: Any) -> str:
    if hasattr(target, "__name__"):
        return _canon_aten_name(target.__name__)
    if isinstance(target, str):
        return _canon_aten_name(target)
    s = str(target)
    if "aten::" in s:
        # Keep the overload-qualified tail ('min.other', 'add.Tensor')
        # — ``_canon_aten_name`` resolves it; truncating at the first
        # dot would collapse 'aten::min.other' onto the reduction 'min'.
        name = s.split("aten::", 1)[1]
        return _canon_aten_name(name)
    return _canon_aten_name(s)


def _infer_shape(node_or_value: Any) -> tuple:
    if hasattr(node_or_value, "shape"):
        return tuple(
            int(d) if d is not None else None
            for d in node_or_value.shape
        )
    return (None,)


def _fill_src(node: Any, dst_fx: Any, env: dict[str, Any]) -> Any:
    """Materialise a ``fill_``/``zero_`` scalar write as a term.

    The destination's ``meta["val"]`` shape is the broadcast target —
    an absent meta means the write cannot be sized honestly, so the
    ``KeyError`` surfaces as an export rejection rather than minting a
    term that drops the mutation.  ``fill_.Tensor``'s source is a
    lifted 0-dim tensor (broadcast); ``fill_.Scalar``'s is an inline
    number (a ``full`` creator); ``zero_`` carries none (``zeros``).
    """
    val = dst_fx.meta["val"]
    shape = tuple(int(d) for d in val.shape)
    dtype = canonical_name(val.dtype)
    # The minted fill value must land where the destination lives —
    # a CPU ``zeros`` scattered into a CUDA view dies on the device
    # mismatch (same wall-C class as the dropped ``device=`` kwarg).
    # ``meta["val"]`` is a FakeTensor in real exports (always carries
    # ``.device``); the getattr keeps hand-built meta honest.
    dev = getattr(val, "device", None)
    device = str(dev) if dev is not None else None
    src_fx = node.args[1] if len(node.args) > 1 else None
    if _aten_name(node.target).startswith("zero_"):
        return Op.make(
            "zeros",
            shape=shape,
            dtype=dtype,
            **({"device": device} if device else {}),
        )
    if hasattr(src_fx, "name"):
        # KeyError on an unminted source is an honest rejection.
        return Op.make("broadcast_to", env[src_fx.name], shape=shape)
    return Op.make(
        "full",
        Const(cast("int | float", src_fx)),
        shape=shape,
        dtype=dtype,
        **({"device": device} if device else {}),
    )


def _view_chain(dst_fx: Any) -> tuple[Any, list]:
    """Walk a functionalized write target up its view chain.

    Returns ``(root_fx, steps)`` where ``steps`` lists
    ``(view_fx, base_fx, kind, index)`` entries ordered INNERMOST
    first (the write target's own view first) and ``root_fx`` is the
    first non-``slice``/``select`` producer — the tensor the write
    ultimately mutates.  ``kind`` is ``"slice"`` (``index =
    (dim, start, end, step)``) or ``"select"`` (``index = (dim,
    index)``).  A non-view target returns ``(dst_fx, [])`` — the
    whole-tensor write.
    """
    steps: list = []
    node = dst_fx
    while True:
        target_attr = getattr(node, "target", None)
        if target_attr is None:
            # A non-FX operand (an unminted base, a literal) ends the
            # chain — the caller declines on the missing binding.
            return node, steps
        target = _aten_name(target_attr)
        args = getattr(node, "args", ())
        if target == "slice" and len(args) >= 4:
            step = args[4] if len(args) > 4 else 1
            steps.append(
                (
                    node,
                    args[0],
                    "slice",
                    (args[1], args[2], args[3], step),
                )
            )
        elif target == "select" and len(args) >= 3:
            steps.append((node, args[0], "select", (args[1], args[2])))
        else:
            return node, steps
        node = args[0]


def _scatter_view(kind: str, base: Any, val: Any, idx: tuple) -> Any:
    """Scatter *val* into *base* at *idx* — the functional write."""
    if kind == "slice":
        dim, start, end, step = idx
        return Op.make(
            "slice_scatter",
            base,
            val,
            dim=dim,
            start=start,
            end=end,
            step=step,
        )
    dim, index = idx
    return Op.make("select_scatter", base, val, dim=dim, index=index)


def _read_view(kind: str, base: Any, idx: tuple) -> Any:
    """Read *base* at *idx* — the ``slice``/``select`` view term."""
    if kind == "slice":
        dim, start, end, step = idx
        return Op.make(
            "slice", base, dim=dim, start=start, end=end, step=step
        )
    dim, index = idx
    return Op.make("select", base, dim=dim, index=index)


def _handle_copy_(node: Any, env: dict[str, Any]) -> None:
    """Thread a functionalized ``copy_``/``fill_``/``zero_`` write.

    ``copy_(dst, src)`` where dst is a ``slice``/``select`` view rewrites
    the VIEWED BASE's env binding to the matching scatter — downstream
    readers of the base then see the post-write value.  ``fill_`` and
    ``zero_`` are the scalar spellings of the same write: the value is
    first broadcast to the destination's shape (``_fill_src``), so a
    ``y[:, :4] = 0`` slice fill lowers through ``slice_scatter`` like
    the ``copy_`` view-write it functionalises from.  A copy on a
    whole tensor rewrites the dst binding itself to a ``copy`` op.

    The write target may sit behind a CHAIN of views — ``y[:, :3, :2]
    = v`` exports as ``fill_(slice(slice(y, 1, 0, 3), 2, 0, 2), v)``.
    The chain is folded from the innermost write outward
    (:func:`_scatter_view`), each step scattering the already-written
    value onto its own base, and the ROOT base's binding is rebound to
    the composed result — so the mutation reaches the tensor readers
    actually observe instead of orphaning on an intermediate view.
    Each view node's binding is then rebuilt from the rebound root so
    later reads of the chain see the write.

    Every unresolvable step is a LOUD export rejection, matching
    ``masked_fill_``'s deliberate ``KeyError``: a write whose target
    (or a view-chain base) minted no IR term cannot be threaded, and
    silently returning would drop the mutation — downstream readers
    would compute the pre-write value with no signal.
    """
    write = _aten_name(getattr(node, "target", ""))
    if not node.args:
        raise ValueError(
            f"export_to_ir: {write!r} node {getattr(node, 'name', '?')!r} has no "
            "operands — a mutation with no destination cannot be "
            "threaded into env"
        )
    dst_fx = node.args[0]
    dst = env.get(getattr(dst_fx, "name", None))
    if dst is None:
        raise ValueError(
            f"export_to_ir: {write!r} node {getattr(node, 'name', '?')!r} writes to "
            f"{getattr(dst_fx, 'name', dst_fx)!r} which minted no IR "
            "term — refusing rather than silently dropping the "
            "mutation"
        )
    target = _aten_name(node.target)
    if target.startswith(("fill_", "zero_")):
        src = _fill_src(node, dst_fx, env)
    elif target.startswith("masked_fill_"):
        # masked_fill_(dst, mask, v) — the functional ``masked_fill``
        # term over dst is the written value; the view dispatch below
        # scatters it onto the base exactly like a copy_ source.  An
        # unminted mask/value raises (KeyError/IndexError) into the
        # caller's rejection record — never a silently-dropped write.
        mask = env[node.args[1].name]
        v_fx = node.args[2]
        v = env[v_fx.name] if hasattr(v_fx, "name") else Const(v_fx)
        src = Op.make("masked_fill", dst, mask, v)
    else:
        src_fx = node.args[1] if len(node.args) > 1 else None
        src = env.get(getattr(src_fx, "name", None))
        if src is None:
            raise ValueError(
                f"export_to_ir: {write!r} node {getattr(node, 'name', '?')!r} copies "
                f"from {getattr(src_fx, 'name', src_fx)!r} which "
                "minted no IR term — refusing rather than silently "
                "dropping the mutation"
            )
    root_fx, steps = _view_chain(dst_fx)
    if not steps:
        # Whole-tensor write (or an unrecognised view): dst's readers
        # see the broadcast-cloned source.
        env[dst_fx.name] = Op.make("copy", dst, src)
        env[node.name] = env[dst_fx.name]
        return
    root = env.get(getattr(root_fx, "name", None))
    if root is None:
        raise ValueError(
            f"export_to_ir: {write!r} node {getattr(node, 'name', '?')!r} writes "
            f"through a view chain rooted at "
            f"{getattr(root_fx, 'name', root_fx)!r} which minted no "
            "IR term — refusing rather than silently dropping the "
            "mutation"
        )
    # Fold the scatters innermost-outward; each reads its base's
    # current binding so successive writes compose.
    val = src
    for _view_fx, base_fx, kind, idx in steps:
        base = env.get(getattr(base_fx, "name", None))
        if base is None:
            raise ValueError(
                f"export_to_ir: {write!r} node {getattr(node, 'name', '?')!r} writes "
                f"through view {getattr(base_fx, 'name', base_fx)!r} "
                "which minted no IR term — refusing rather than "
                "silently dropping the mutation"
            )
        val = _scatter_view(kind, base, val, idx)
    env[root_fx.name] = val
    # Rebuild each view from the rebound root (root -> outward).
    cur = val
    for view_fx, _base_fx, kind, idx in reversed(steps):
        cur = _read_view(kind, cur, idx)
        env[view_fx.name] = cur
    env[node.name] = env[dst_fx.name]


def export_to_ir(
    model: torch.nn.Module,
    example_input: torch.Tensor | tuple,
) -> tuple[IR, dict[str, torch.Tensor]]:
    """Export a PyTorch model to a mlopt IR via torch.export.

    ``example_input`` may be a single tensor or a tuple of positional
    args for multi-input modules.

    Returns ``(ir, source_tensors)`` — the IR plus the concrete
    parameter/buffer values keyed by IR param name (e.g. ``p_w1``),
    used to materialise the lowered module and to compare exact
    weights in the non-local passes.

    Uses exported._graph_signature.inputs_to_parameters to map graph
    placeholder node targets (e.g. 'p_w1') to actual model attribute
    names (e.g. 'W1'), so we can retrieve parameter shapes correctly.
    """
    model.eval()
    args = (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )
    with torch.no_grad():
        exported = torch.export.export(model, args)

    graph = exported.graph
    mod = exported.module()  # the actual Module instance
    state_dict = (
        exported.state_dict
        if isinstance(exported.state_dict, dict)
        else dict(exported.state_dict)
    )

    # Dict to collect original parameter tensors for IRModule
    source_tensors: dict[str, torch.Tensor] = {}

    # Map from graph placeholder target -> model attribute name
    # e.g. {'p_w1': 'W1', 'p_w2': 'W2', 'p_w3': 'W3'}
    inputs_to_params: dict[str, str] = {}
    with contextlib.suppress(Exception):
        inputs_to_params = dict(
            exported._graph_signature.inputs_to_parameters
        )
    # Buffers (e.g. nanoGPT's causal-mask `bias`) are placeholders too —
    # they are constants, not user inputs: lift them like parameters.
    with contextlib.suppress(Exception):
        inputs_to_params.update(
            exported._graph_signature.inputs_to_buffers
        )
    # Tensor constants created inside forward (``torch.tensor(0.5)``)
    # lift to placeholders keyed into ``exported.constants`` — also
    # constants, never user inputs.  Leaving them as Vars would let
    # eval's "self" fallback silently substitute the first input.
    with contextlib.suppress(Exception):
        inputs_to_params.update(
            exported._graph_signature.inputs_to_lifted_tensor_constants
        )
    constants = getattr(exported, "constants", None) or {}

    env: dict[str, Any] = {}
    inputs: list[Var] = []
    params: dict[str, Param] = {}

    for node in graph.nodes:
        if node.op == "placeholder":
            target = node.target  # e.g. 'p_w1' or 'x'
            model_attr_name = inputs_to_params.get(target)

            if model_attr_name is not None:
                # It's a parameter — get its real name, shape, and original tensor
                # via the exported state_dict (handles dotted names like
                # 'gate.weight'; GraphModule attrs may not itself hold them).
                tensor = state_dict.get(model_attr_name)
                if tensor is None:
                    # Lifted tensor constants live in
                    # ``exported.constants``, not the state dict.
                    tensor = constants.get(model_attr_name)
                if tensor is None or not hasattr(
                    tensor, "shape"
                ):  # pragma: no cover — every signature-mapped tensor
                    # lives in the state dict or exported.constants;
                    # this is the get_attr-era safety net.
                    tensor = _resolve_attr(mod, model_attr_name)
                shape = tuple(int(d) for d in tensor.shape)
                param = Param(name=node.name, typ=TensorType(shape))
                env[node.name] = param
                params[node.name] = param
                # ``detach`` SHARES storage with the model's parameter
                # (``exported.state_dict`` entries ARE the module's own
                # ``nn.Parameter`` objects — verified: every state-dict
                # tensor aliases model storage).  ``clone`` doubled the
                # resident weight set for the whole search+lower
                # window: on a 4 GB card the turbo whisper encoder
                # pinned ~2.4 GiB (clone + still-referenced originals)
                # and ``lower(verify=True)`` OOM'd.  Sharing keeps ONE
                # copy — ``param_values`` holds the storage alive after
                # the caller drops the model, which is exactly what the
                # lowering needs.  Trade-off vs the old clone: the
                # values are a live view — an in-place mutation of the
                # source model's params after export is visible here
                # (verify stays self-consistent: the reference lowering
                # reads the same dict).  ``torch_eval`` already detaches
                # (never clones) for the same reason; a plain detached
                # tensor also avoids registering the user's
                # ``nn.Parameter`` objects inside ``IRModule``.
                source_tensors[node.name] = tensor.detach()
            else:
                # It's a model input (user-provided tensor).  Prefer the
                # exported node's own metadata shape — correct for
                # multi-input graphs where placeholders differ.
                meta_shape = getattr(
                    getattr(node, "meta", {}), "get", lambda k: None
                )("val")
                if meta_shape is not None and hasattr(
                    meta_shape, "shape"
                ):
                    shape = tuple(int(d) for d in meta_shape.shape)
                elif len(inputs) < len(args):
                    shape = _infer_shape(args[len(inputs)])
                else:  # pragma: no cover — placeholders consume an
                    # example arg in order; an unmapped extra (e.g. a
                    # lifted custom object) is the only overrun.
                    shape = _infer_shape(args[0])
                var = Var(name=node.name, typ=TensorType(shape))
                env[node.name] = var
                inputs.append(var)

        elif (
            node.op == "get_attr"
        ):  # pragma: no cover — torch 2.14 export lifts everything to placeholders
            tensor = _resolve_attr(mod, node.target)
            shape = tuple(int(d) for d in tensor.shape)
            param = Param(name=node.name, typ=TensorType(shape))
            env[node.name] = param
            params[node.name] = param

        elif node.op == "call_function":
            op_name = _aten_name(node.target)
            ir_op = _ATEN_TO_IR.get(op_name, op_name)
            if op_name.startswith(
                ("copy_", "fill_", "zero_", "masked_fill_")
            ):
                # Functionalized in-place write ``dst = src`` — the
                # FX graph is not SSA here: ``copy_``/``fill_``/
                # ``zero_``/``masked_fill_`` MUTATE dst's tensor and
                # downstream nodes keep referencing the pre-write
                # node.  Thread the
                # mutation through env: a write through a
                # ``slice``/``select`` view becomes the matching
                # scatter on the viewed base; a whole-tensor write is
                # the ``copy`` op — either way consumers of the dst
                # node see the post-write value.
                _handle_copy_(node, env)
                continue
            if ir_op not in _IR_TO_TORCH:
                # Opaque leaf: an aten/custom op outside the lowering
                # vocabulary still lifts.  The term replays the
                # original torch callable verbatim at eval — laws and
                # collectors cannot rewrite inside it, but the model
                # optimizes around it and delivery still works.
                env[node.name] = _opaque_term(node, env)
                continue
            args = []
            attrs: dict[str, Any] = {}
            # aten.clamp.Tensor(self, min?, max?) drops a ``None``
            # bound out of the arg list entirely — the surviving
            # tensor args' aten positions (1=min, 2=max) must be
            # recorded or ``clamp(a, max=b)`` silently mints the same
            # term as ``clamp(a, min=b)`` (same silent-wrong-answer
            # class as the aten.min 'other' wall).
            clamp_argpos: list[int] = []
            positional_attrs = {
                **_LOCAL_ATTR_SCHEMA.get(ir_op, {}),
                **ATTR_SCHEMA.get(ir_op, {}),
            }
            for i, arg_node in enumerate(node.args):
                if (
                    i in positional_attrs
                    and arg_node is not None
                    and not hasattr(arg_node, "name")
                ):
                    # e.g. conv2d(x, w, b, [s,s], [p,p], [d,d], g) —
                    # non-node positionals at schema-declared positions
                    # are named op attributes.  ``None`` positionals
                    # (an absent sdpa attn_mask) are skipped — a
                    # present-but-None attr would poison e-matching's
                    # exact attr-set comparison.
                    v = arg_node
                    attrs[positional_attrs[i]] = (
                        tuple(v) if isinstance(v, (list, tuple)) else v
                    )
                    continue
                if isinstance(
                    arg_node, (int, float, complex)
                ) and not isinstance(arg_node, bool):
                    # Scalar operands of arithmetic ops are real operands
                    # (e.g. pow(x, 2)); for reductions the scalar is a
                    # dim/arg attribute (e.g. x.mean(-1)).  ``complex``
                    # rides the same channel — ``phase * 1j`` (the
                    # Kokoro decoder-nc ``exp(sin(phase) * 1j)`` polar
                    # spelling) exports the literal as a positional;
                    # dropping it minted a UNARY ``mul`` that died
                    # ``missing 1 required positional argument``.
                    if ir_op in _SCALAR_OPERAND_OPS:
                        # Keep ints as ints — coercing to float
                        # makes e.g. ``x % 2`` promote to float at
                        # eval (aten ``*.Scalar`` is weak-typed).
                        # Complex stays complex for the same reason:
                        # ``float(1j)`` would raise, and the Const
                        # must keep the imaginary part.
                        args.append(
                            Const(
                                cast("int | float", arg_node)
                                if isinstance(arg_node, (int, complex))
                                else float(arg_node)
                            )
                        )
                    else:
                        attrs[f"arg{i}"] = arg_node
                elif isinstance(arg_node, torch.device):
                    # ``to.device`` carries the device POSITIONALLY
                    # (``x.to(y)`` exports ``to.device(x, device, dtype)``)
                    # — record the canonical string like the dtype
                    # channel; a raw torch.device is not JSON-safe.
                    attrs["device"] = str(arg_node)
                elif isinstance(arg_node, str):  # pragma: no cover —
                    # schema'd str positionals (pad.mode, conv padding,
                    # einsum.equation) are named upstream; an unschemed
                    # str positional would land here.
                    continue
                elif isinstance(arg_node, (list, tuple)):
                    if any(hasattr(a, "name") for a in arg_node):
                        # A list of FX nodes (stack/cat take tensor lists)
                        # — each element is an operand.
                        for a in arg_node:
                            key = (
                                a.name if hasattr(a, "name") else str(a)
                            )
                            if key in env:
                                args.append(env[key])
                        if ir_op in ("index", "index_put"):
                            # Advanced indexing keeps real semantics in
                            # the key's None slots: x[:, i] is NOT x[i].
                            # ``layout`` marks which positions carry an
                            # index operand (True) vs a full slice
                            # (False) so the binding can rebuild the
                            # exact key tuple.
                            attrs["layout"] = tuple(
                                hasattr(a, "name") for a in arg_node
                            )
                    # e.g. dim=[-1] lists for reductions; for view/reshape
                    # the list is the target SHAPE, not a dim.
                    elif ir_op == "reshape" or ir_op in (
                        "expand",
                        "repeat",
                        "broadcast_to",
                        # Tensor creators take the shape tuple as the
                        # same-named attr — zeros((2, 3)) etc.
                        "zeros",
                        "ones",
                        "empty",
                        "randn",
                        "rand",
                        "full",
                        "new_zeros",
                        "new_ones",
                        "new_empty",
                        "new_full",
                    ):
                        attrs["shape"] = tuple(arg_node)
                    elif ir_op in ("split", "chunk"):
                        attrs["sizes"] = tuple(
                            arg_node
                        )  # pragma: no cover — schema covers all real positions
                    elif ir_op in _FFT_SHAPE_OPS:
                        # torch.fft.irfft2(x, s=(h, w)) — the positional
                        # tuple is the output SHAPE, not a dim list.
                        # Recorded as ``dim`` it would be read as axis
                        # indices and raise for any real rank.
                        attrs["s"] = tuple(arg_node)
                    else:
                        attrs["dim"] = tuple(arg_node)
                elif isinstance(arg_node, bool):
                    if ir_op in (
                        "mean",
                        "sum",
                        "max",
                        "min",
                        "prod",
                        "var",
                        "std",
                        "amax",
                        "amin",
                    ):
                        attrs["keepdim"] = arg_node
                    else:
                        attrs[f"arg{i}"] = arg_node
                elif isinstance(arg_node, torch.dtype):
                    # ``to``/creator overloads carry a positional
                    # ScalarType; record the short name just like the
                    # dtype kwarg channel below — a dropped cast mints
                    # a wrong-dtype term (one_hot's long->float).
                    attrs["dtype"] = canonical_name(arg_node)
                else:
                    key = (
                        arg_node.name
                        if hasattr(arg_node, "name")
                        else str(arg_node)
                    )
                    if key in env:
                        args.append(env[key])
                        if ir_op == "clamp" and i > 0:
                            # Position 0 is ``self`` — only the bound
                            # slots (1=min, 2=max) are ambiguous.
                            clamp_argpos.append(i)
            for k, v in node.kwargs.items():
                kw_name = getattr(v, "name", None)
                if isinstance(kw_name, str):
                    # FX-node kwarg — a tensor operand passed by
                    # keyword (e.g. ``attn_mask=m``).  torch.export
                    # normalises declared-schema kwargs into
                    # positional args, but an op whose schema keeps a
                    # keyword operand (or a custom/extension op)
                    # lands here; dropping it mints a term that
                    # computes WITHOUT the operand — the silent-wrong
                    # class ``_opaque_encode`` already refuses.
                    # Resolve through env and append as an operand;
                    # an unminted producer raises like masked_fill_.
                    if kw_name not in env:
                        raise ValueError(
                            f"export_to_ir: kwarg {k!r} of "
                            f"{op_name!r} references unminted node "
                            f"{kw_name!r} — refusing rather than "
                            "dropping the operand"
                        )
                    args.append(env[kw_name])
                elif v is None:
                    # An explicit-None kwarg is the absent-operand
                    # spelling — same rule as the positional ``None``
                    # drops above: recording it would poison
                    # e-matching's exact attr-set comparison.
                    continue
                elif isinstance(v, (int, float, bool)):
                    attrs[k] = v
                elif isinstance(v, (torch.layout, torch.memory_format)):
                    # torch.layout / torch.memory_format are not
                    # JSON-safe — record the short name like dtype
                    # ("strided" / "channels_last").  They are
                    # dispatch/memory-format details, not tensor
                    # semantics.
                    attrs[k] = str(v).split(".")[-1]
                elif isinstance(v, (list, tuple)):
                    if any(hasattr(e, "name") for e in v):
                        raise ValueError(
                            f"export_to_ir: kwarg {k!r} of "
                            f"{op_name!r} is a container carrying "
                            "FX nodes — the bound-op attr channel "
                            "cannot encode tensor operands nested "
                            "in a tuple/list; refusing rather than "
                            "dropping them"
                        )
                    # Tuple-valued kwargs are semantics too:
                    # ``pad=(1,1)``, ``stride=(2,2)``, ``size=(8,8)``
                    # — attrs are the only channel that reaches the
                    # binding (``isinstance(v, list)`` alone used to
                    # drop every tuple silently).
                    attrs[k] = tuple(v)
                elif isinstance(v, str):
                    # String-valued flags are semantics: gelu's
                    # ``approximate``, pad's ``mode``, scatter_reduce's
                    # ``reduce``.  Attrs are the only channel that
                    # reaches the binding.
                    attrs[k] = v
                elif isinstance(v, torch.dtype):
                    # ``dtype`` is semantics, not a dispatch detail —
                    # ``zeros(shape, dtype=float64)`` must not lower
                    # to an fp32 tensor.  Record the short name
                    # ("float64"): torch.dtype objects do not survive
                    # term serialization.
                    attrs[k] = canonical_name(v)
                elif isinstance(v, torch.device):
                    # ``device=`` is semantics too — creator/``to``
                    # placement (Kokoro wall C: ``torch.rand(...,
                    # device=f0.device)`` minted CPU and died
                    # ``Expected all tensors to be on the same
                    # device``).  torch.device is not JSON-safe, so
                    # record the canonical string ("cuda:0"); the
                    # creator/``to`` bindings parse it back.
                    attrs[k] = str(v)
                else:
                    raise ValueError(
                        f"export_to_ir: unencodable kwarg {k}={v!r} "
                        f"({type(v).__name__}) on {op_name!r} — only "
                        "FX nodes, scalars, str, list/tuple "
                        "literals and torch dtype/device/layout/"
                        "memory_format values can cross the IR "
                        "boundary; refusing rather than dropping it"
                    )
            # Canonical attr spellings are guaranteed below by
            # ``Op.make`` (ATTR_SCHEMA): positional attrs were named at
            # emission above; any ``argN`` still present for a schema'd
            # op fails loudly at mint.
            #
            # aten elides trailing arguments at their defaults, so an
            # implicit ``dim=0`` on split*/chunk never reaches the FX
            # node.  Record it: an absent ``dim`` reads as the last
            # axis (the minted-term/law convention), which is the
            # WRONG axis for aten's positional default — the packed
            # QKV in-proj split is a dim-0 split of a (3E, E) weight.
            if ir_op in ("split", "chunk") and "dim" not in attrs:
                attrs["dim"] = 0
            # aten positions 3,4 of batch_norm are running_mean/
            # running_var.  A ``track_running_stats=False`` module
            # exports them as ``None`` — dropped by the operand loop
            # above — leaving the affine pair ``(w, b)`` in the
            # operand list where ``_batch_norm_torch``'s arity
            # heuristic would misread it as ``(rm, rv)`` (len-2 is
            # ambiguous: ``(w, b)`` vs ``(rm, rv)``).  Record the
            # absence so eval never has to guess; stats-present terms
            # are unchanged (no new attr — e-matching's exact
            # attr-set comparison stays intact for them).
            if ir_op == "batch_norm" and not (
                len(node.args) > 4
                and hasattr(node.args[3], "name")
                and hasattr(node.args[4], "name")
            ):
                attrs["running_stats"] = False
            if ir_op == "clamp" and clamp_argpos:
                # See the declaration above the arg loop: tensor bounds
                # keep their aten slot (1=min, 2=max) as ``argpos``.
                attrs["argpos"] = tuple(clamp_argpos)
            if (
                ir_op == "getitem"
                and args
                and isinstance(args[0], Op)
                and args[0].op
                in ("split", "chunk", "unbind", "tensor_split")
                and isinstance(node.args[1], int)
            ):
                # getitem(split(t), i) — fold the index into the
                # splitter so the op yields element i directly.
                base = args[0]
                ir_node = Op.make(
                    base.op,
                    *base.args,
                    **{**dict(base.attrs), "index": node.args[1]},
                )
            elif args:
                ir_node = Op.make(ir_op, *args, **attrs)
            else:
                ir_node = Op.make(ir_op, **attrs)
            env[node.name] = ir_node

        elif node.op == "output":
            # The graph's return node — resolved post-loop by
            # ``_graph_root``/``_graph_outputs``; nothing to mint here.
            continue
        else:
            # ``call_module``/``call_method`` (or any future FX kind)
            # would produce no env entry — every downstream operand
            # referencing it would silently drop (the same silent-
            # wrong class as the kwargs drops above).  torch.export
            # flattens submodule calls into ``call_function`` on
            # common models, but an unmodelled kind is refused loudly
            # rather than skipped.
            raise ValueError(
                "export_to_ir: unhandled FX node kind "
                f"{node.op!r} (node {getattr(node, 'name', '?')!r}, "
                f"target {getattr(node, 'target', None)!r}) — the "
                "lifter models placeholder/get_attr/call_function/"
                "output only; refusing rather than silently "
                "dropping its value"
            )

    # Find the graph's output(s) and build the search root — a single
    # output IS the root; several are paired by ``pack`` (the product
    # morphism), so every output is optimised and lowered, none
    # dropped.
    root = _graph_root(graph, env)

    return IR(
        root=root,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params=params,
    ), source_tensors


def _graph_root(graph: Any, env: dict[str, Any]) -> Any:
    """Resolve the FX graph's output(s) into one mlopt root term.

    The FX output node's ``args[0]`` is the tuple of the model's return
    leaves — ``(out,)`` for a single-output module, ``(o0, ..., on)``
    for a tuple-returning one (``torch.export`` flattens the return
    pytree).  Each leaf is resolved through ``env`` (honouring env
    rewrites — e.g. the copy_ mutation threader rebinding ``clone`` to
    its scatter), so NO output is silently dropped.

    A single output returns that term; several return their ``pack``
    pairing — the product morphism the single-root search saturates.
    A non-tensor output leaf cannot be modelled, so it is refused
    loudly rather than dropped.
    """
    outputs, unresolved = _graph_outputs(graph, env)
    if unresolved:
        # A model returning a non-tensor (an int, ``None``) cannot be
        # modelled by the IR — refuse loudly rather than dropping it.
        raise ValueError(
            f"export_to_ir: {unresolved} model output(s) do not resolve "
            "to an IR term (a non-tensor return value); multi-output "
            "lowering supports tensor outputs only"
        )
    return (
        outputs[0] if len(outputs) == 1 else Op.make("pack", *outputs)
    )


def _graph_outputs(
    graph: Any, env: dict[str, Any]
) -> tuple[list[Any], int]:
    """Collect every resolvable output term plus the unresolvable count."""
    outputs: list[Any] = []
    unresolved = 0
    for node in reversed(
        graph.nodes
    ):  # pragma: no branch — output node is always last, first when reversed
        if node.op == "output" and node.args:
            arg = node.args[0]
            leaves = (
                list(arg) if isinstance(arg, (tuple, list)) else [arg]
            )
            for leaf in leaves:
                if hasattr(leaf, "name") and leaf.name in env:
                    outputs.append(env[leaf.name])
                else:  # pragma: no cover — a non-tensor return leaf
                    unresolved += 1
            break
    if (
        not outputs and env
    ):  # pragma: no cover — output args resolve via the tuple flatten
        # above; this only fires on a structurally-degenerate graph.
        outputs = [list(env.values())[-1]]
    return outputs, unresolved


def _resolve_attr(module: torch.nn.Module, target: str) -> torch.Tensor:
    obj: Any = module
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


#: fft ops whose positional tuple is an output SHAPE (``s``), not a dim
#: list.  The generic tuple branch would record it as ``dim`` and the
#: binding would then ask torch to transform axis 8 of a 4-D tensor.
_FFT_SHAPE_OPS = frozenset(
    {
        "fft_irfft2",
        "fft_ifft2",
        "fft_irfft",
        "fft_ifft",
    }
)
# ---------------------------------------------------------------------------
#  Opaque leaves — unbound aten/custom ops lift as passthrough terms
# ---------------------------------------------------------------------------
#
# An op with no entry in ``_IR_TO_TORCH`` used to mint an ``Op`` that
# died at eval with "No torch binding".  Instead it now mints an
# ``opaque`` term: the op's args/kwargs are replayed verbatim through a
# positional ``sig`` spec and the ``aten`` qualname, and the ``opaque``
# binding resolves the qualname back to the torch callable at eval —
# semantics IS the torch op itself.  Laws, collectors and shape rules
# see only the ``opaque`` generator (no rewrite reaches inside), the
# cost model prices it like any unknown op, and ``supported_ops``
# carries ``"opaque"`` because it is a core binding — so extraction
# keeps the leaf and delivery replays the real kernel.
#
# ``sig`` is a ``(arg_specs, kwarg_specs)`` pair of JSON-safe spec
# tuples.  Each spec is ``("t", i)`` — the i-th operand term — or a
# literal ``("l", v)`` / ``("dt", name)`` / ``("lst"|"tpl"|"map", ...)
# container.  Operands are the node's tensor-producing FX args in
# encounter order, so the binding's ``next(operand_iter)`` walk and
# the exporter's append order agree exactly.


def _is_dotted_name(s: str) -> bool:
    """Return ``True`` for ``a.b.c`` identifier paths (no ``<...>``)."""
    return (
        "." in s
        and not s.startswith("<")
        and all(p.isidentifier() for p in s.split("."))
    )


@functools.cache
def _resolve_opaque_callable(qual: str) -> Callable[..., Any]:
    """Resolve an opaque ``aten`` qualname to its torch callable.

    ``ns::op[.overload]`` walks ``torch.ops`` (aten ops AND
    ``torch.library`` custom ops live there); ``mod:qualname`` imports
    the module and walks attributes (plain Python callables);
    a bare dotted path walks ``torch`` attributes.
    """
    if "::" in qual:
        ns, _, rest = qual.partition("::")
        obj: Any = torch.ops
        for part in (*ns.split("."), *rest.split(".")):
            obj = getattr(obj, part)
        return obj
    if ":" in qual:
        mod_name, _, qn = qual.partition(":")
        obj: Any = importlib.import_module(mod_name)
        for part in qn.split("."):
            obj = getattr(obj, part)
        return obj
    obj: Any = torch
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def _opaque_qual(target: Any) -> str:
    """First *resolvable* qualname for an opaque FX ``node.target``.

    Candidates, in order: the target's own ``.name()`` (``OpOverload``
    → ``"aten::mul.Tensor"`` / ``"myns::op"``), its ``str`` when it is
    already ``::``-qualified, the dotted ``str`` re-read as a
    ``torch.ops`` path (``"zz.f"`` → ``"zz::f"``), the Python
    ``__module__:__qualname__`` path, and the bare dotted ``str`` as a
    ``torch.*`` attribute walk.  Resolution is verified NOW — an
    unresolvable target refuses the lift loudly instead of minting a
    term that can never deliver.
    """
    candidates: list[str] = []
    name = getattr(target, "name", None)
    if callable(name):
        with contextlib.suppress(Exception):
            s = name()
            if isinstance(s, str) and "::" in s:
                candidates.append(s)
    s = str(target)
    if "::" in s:
        candidates.append(s)
    elif _is_dotted_name(s):
        head, _, tail = s.partition(".")
        candidates.append(f"{head}::{tail}")
    mod = getattr(target, "__module__", None)
    qn = getattr(target, "__qualname__", None)
    if (
        isinstance(mod, str)
        and mod
        and mod != "builtins"
        and isinstance(qn, str)
        and qn
    ):
        candidates.append(f"{mod}:{qn}")
    if _is_dotted_name(s):
        candidates.append(s)
    errors: list[str] = []
    for qual in candidates:
        try:
            _resolve_opaque(qual)
        except Exception as e:  # collect, report below
            errors.append(f"{qual!r}: {e}")
        else:
            return qual
    raise ValueError(
        "export_to_ir: cannot resolve a torch callable for opaque "
        f"target {target!r} — tried {errors or ['<no candidates>']}"
    )


def _opaque_encode(
    a: Any, env: dict[str, Any], operands: list
) -> tuple:
    """Encode one FX arg as a JSON-safe spec; append operands in order.

    FX nodes become ``("t", i)`` operand slots; scalars/dtypes/devices
    stay literals (``("cx", re, im)`` / ``("dev", str)`` keep the
    JSON-safe boundary); lists/tuples/mappings recurse.  Anything else
    (raw tensors, ``SymInt``s, memory formats) refuses the lift — an
    arg that cannot cross the IR boundary would deliver wrong results
    silently, so it fails loudly here instead.
    """
    name = getattr(a, "name", None)
    if isinstance(name, str):
        # An FX node — ``Tensor.name`` exists too (None for unnamed
        # tensors), so the isinstance guard, not hasattr, selects.
        term = env.get(name)
        if term is None:
            raise ValueError(
                "export_to_ir: opaque op argument "
                f"{name!r} has no IR term — it was produced by a "
                "node kind the lifter does not model (call_module / "
                "call_method / an unresolved output); refusing rather "
                "than dropping the operand"
            )
        operands.append(term)
        return ("t", len(operands) - 1)
    if a is None or isinstance(a, (bool, int, float, str)):
        return ("l", a)
    if isinstance(a, complex):
        # ``("cx", re, im)`` — a complex literal is not JSON-safe raw
        # (``1j`` appears in polar spellings like ``phase * 1j`` on
        # unbound ops too); the pair round-trips through attr_to_data.
        return ("cx", a.real, a.imag)
    if isinstance(a, torch.device):
        # ``("dev", "cuda:0")`` — device placement is semantics for
        # the unbound creators (``rand_like``/``randint``/``tensor``
        # family); the canonical string is JSON-safe, torch.device
        # itself is not.
        return ("dev", str(a))
    if isinstance(a, torch.dtype):
        return ("dt", canonical_name(a))
    if isinstance(a, Mapping):
        return (
            "map",
            tuple(
                (k, _opaque_encode(v, env, operands))
                for k, v in a.items()
            ),
        )
    if isinstance(a, (list, tuple)):
        kind = "tpl" if isinstance(a, tuple) else "lst"
        return (
            kind,
            tuple(_opaque_encode(x, env, operands) for x in a),
        )
    raise ValueError(
        "export_to_ir: opaque op has an unencodable argument "
        f"{a!r} ({type(a).__name__}) — only FX nodes, scalars, complex/"
        "device/dtype literals and list/tuple/mapping containers can "
        "cross the IR boundary"
    )


def _opaque_term(node: Any, env: dict[str, Any]) -> Op:
    """Mint the ``opaque`` term for an unbound ``call_function`` node."""
    operands: list = []
    arg_specs = tuple(
        _opaque_encode(a, env, operands) for a in node.args
    )
    kw_specs = tuple(
        (k, _opaque_encode(v, env, operands))
        for k, v in node.kwargs.items()
    )
    qual = _opaque_qual(node.target)
    return Op.make(
        "opaque",
        *operands,
        aten=qual,
        sig=(arg_specs, kw_specs),
    )

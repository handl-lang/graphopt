"""Denotations and minted handlers as handl Models — plan 0033 phase 2.

A *Model* is the semantic content of a ``synop_*`` denotation or a
minted handler body spelled as a ``.hnl`` expression over the
``host-tensor`` op surface — the same programs ``mlopt-kit``'s
``hnl-run`` evaluates (``bench/drivers/kit/term_hnl.py`` is the
term-side bridge; this module is the spec-side one).  The fragment
directory carries the projection: ``vocabdir.collect_models`` emits
``models.yaml`` rows (``{id, op, kind, mvars, spec|body, hnl,
unsupported}``) and :func:`verify_model` runs the fp64 gate — every
bound site must hold ``torch.allclose`` at the shipped verify floor
(``handler_synth._VERIFY_RTOL`` / ``_VERIFY_ATOL``) before a verdict
may say ``handl-shipped``; anything else is honestly
``python-shipped`` (ADR-0157: ``TorchSink`` stays the delivery oracle
until parity is measured).

The spec grammar is genkernel's: ``(op, *args[, attrs])`` nodes,
``str`` metavar leaves, numeric ``Const`` leaves, and handler bodies
as ``["ew", spec]`` (or another family tag) descriptors.  Only the
ops the ``tensor.*`` surface honours convert — today the elementwise
four, ``matmul``/``linear``, and the shape-manipulation ops with
their attr spellings; anything else raises
:class:`ModelUnsupportedError`
naming the op, and the row records ``unsupported`` instead of an
``hnl``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import Any

__all__ = [
    "ModelUnsupportedError",
    "body_mvars",
    "decode_tensor_result",
    "handler_body_to_hnl",
    "hnl_program",
    "max_rel",
    "model_names",
    "spec_attr_mvars",
    "spec_mvars",
    "spec_to_hnl",
    "tensor_feed",
    "verify_model",
]


class ModelUnsupportedError(ValueError):
    """The spec reaches outside the ``tensor.*`` model surface.

    Carries the op (or family tag) name — the projection's honest
    bound, the same posture ``term_hnl.UnsupportedOpError`` takes.
    """


# ---------------------------------------------------------------------------
#  Literals, names and leaves
# ---------------------------------------------------------------------------


def _num(v: float | int) -> str:
    """Render a number as a handl literal — no scientific notation.

    Same rule ``term_hnl._num`` documents: the lexer reads ``1`` ``e``
    ``-05``, not ``1e-05``; ``Decimal`` expansion keeps every f64
    exact.
    """
    if isinstance(v, bool):  # bool precedes int in the numeric tower
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise ModelUnsupportedError(
            f"non-finite literal {v!r} — the tensor wire carries "
            "f64 data only"
        )
    return format(Decimal(repr(f)), "f")


def _sanitize(name: str, taken: set[str]) -> str:
    """Return a safe ``.hnl`` binding name for a spec metavar.

    Same rule as ``term_hnl._sanitize``: dots are field access in
    handl, leading digits are illegal, and ``tensor`` is reserved —
    binding it would shadow the op namespace.
    """
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if not out or out[0].isdigit() or out == "tensor":
        out = "v_" + out
    while out in taken:
        out += "_"
    taken.add(out)
    return out


def _name_map(mvars: Any) -> dict[str, str]:
    """Deterministically map metavar names to ``.hnl`` names.

    Sorted iteration keeps the map stable — the emitter
    (:func:`spec_to_hnl`) and the program binder
    (:func:`hnl_program`) must agree on every name.
    """
    taken: set[str] = {"tensor"}
    return {m: _sanitize(m, taken) for m in sorted(mvars)}


def _is_call(node: Any) -> bool:
    """Return whether *node* is a spec call ``(op, *args[, attrs])``.

    Any tuple/list headed by a string is call position — an op name
    the surface does not honour must fail loudly in
    :func:`_spec_call_to_hnl`, never degrade to a data literal.
    """
    return (
        isinstance(node, (tuple, list))
        and bool(node)
        and isinstance(node[0], str)
    )


def _spec_mvars(spec: Any, acc: set[str]) -> set[str]:
    """Collect the metavar leaves of a spec (op heads excluded)."""
    if _is_call(spec):
        for e in spec[1:]:
            _spec_mvars(e, acc)
        return acc
    if isinstance(spec, (tuple, list)):
        # A list/tuple whose head is not an op name is not spec
        # grammar — walk it anyway so nested data still reports.
        for e in spec:
            _spec_mvars(e, acc)
        return acc
    if isinstance(spec, str):
        acc.add(spec)
    return acc


def spec_mvars(spec: Any) -> set[str]:
    """Return every metavar name a spec binds, as a set."""
    return _spec_mvars(spec, set())


def body_mvars(body: Any) -> set[str]:
    """Return the metavar names a handler-body descriptor binds."""
    if isinstance(body, (tuple, list)) and len(body) >= 2:
        return spec_mvars(body[1])
    return set()


def _spec_attr_mvars(spec: Any, acc: set[str]) -> set[str]:
    """Collect the attr-metavariable names of a spec.

    Attr metavariables are *string* values inside a call's trailing
    attr dict — the ``pat_spec`` convention keeps them by name (the
    minted call's own attrs carry the bound values back in at eval
    time, ``prim_mint.spec_bind``).  Numeric/bool/list attr values
    are literals and do not collect; strings nested inside a list
    attr (a ``shape`` entry) are data, not metavars.
    """
    if _is_call(spec):
        for e in spec[1:]:
            if isinstance(e, dict):
                for v in e.values():
                    if isinstance(v, str):
                        acc.add(v)
            else:
                _spec_attr_mvars(e, acc)
        return acc
    if isinstance(spec, (tuple, list)):
        for e in spec:
            _spec_attr_mvars(e, acc)
    return acc


def spec_attr_mvars(spec: Any) -> set[str]:
    """Return every attr-metavariable name a spec binds, as a set."""
    return _spec_attr_mvars(spec, set())


# ---------------------------------------------------------------------------
#  The op table — spec op → tensor.* call, as data
# ---------------------------------------------------------------------------

#: Elementwise/binary ops whose ``tensor.*`` spelling is the same
#: name and arity — ``tensor.add(a, b)`` for ``("add", a, b)``.
_BINARY_OPS = frozenset({"add", "sub", "mul", "div", "matmul"})

#: Ops whose attrs map onto trailing positional slots, in order.
#: ``reshape`` takes ``shape``; ``transpose`` the mlopt ``dim0``/``dim1``
#: pair; ``concat`` is handled separately (its args pack into a list).
_ATTR_OPS: dict[str, tuple[str, ...]] = {
    "reshape": ("shape",),
    "transpose": ("dim0", "dim1"),
    # ``tensor.unsqueeze(t, dim)`` — host-tensor phase-3 addition
    # (plan 0033): the size-1-axis insert the five ``unsqueeze``
    # denotations needed; negative dims wrap ``d % (rank+1)`` on
    # both sides.
    "unsqueeze": ("dim",),
    # ``tensor.select(t, dim, index)`` — host-tensor phase-5
    # addition (T-017 follow-up): drop axis ``dim`` at ``index``
    # (torch select: rank -1, negative dim/index wrap).  Attr
    # metavariables (``"aNNN"`` strings) render as string literals,
    # the same convention ``unsqueeze``'s ``dim`` attr uses — the
    # bound-site domain, not the projection, is what gates verdicts.
    "select": ("dim", "index"),
}

#: Fixed-arity ops whose ``tensor.*`` spelling is the same name and
#: argument count but not covered by ``_BINARY_OPS`` —
#: ``tensor.linear(x, w, b)`` for ``("linear", x, w, b)`` (the
#: host-tensor ``F.linear`` spelling: ``x @ wᵀ + b``).
_FIXED_OPS: dict[str, int] = {"linear": 3}


def model_names(mvars: Any) -> dict[str, str]:
    """Return the public metavar→``.hnl``-name map for *mvars*."""
    return _name_map(mvars)


def _lit(v: Any) -> str:
    """Render an attr value / scalar as a ``.hnl`` literal."""
    if v is None:
        return "null"
    if isinstance(v, (bool, int, float)):
        return _num(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (tuple, list)):
        return "[" + ", ".join(_lit(x) for x in v) + "]"
    raise ModelUnsupportedError(f"unemittable attr value: {v!r}")


def _split_attrs(rest: tuple) -> tuple[tuple, dict]:
    """Split call args into positional args and the trailing attr dict."""
    if rest and isinstance(rest[-1], dict):
        return tuple(rest[:-1]), dict(rest[-1])
    return tuple(rest), {}


def _attr_slot(
    v: Any, names: Mapping[str, str], amvars: frozenset
) -> str:
    """Render one attr-op slot: attr metavar → name, else literal."""
    if isinstance(v, str) and v in amvars:
        return names[v]
    return _lit(v)


def _spec_call_to_hnl(
    node: Any,
    names: Mapping[str, str],
    amvars: frozenset = frozenset(),
) -> str:
    """Emit one spec call as a ``tensor.*`` expression."""
    op = node[0]
    args, attrs = _split_attrs(tuple(node[1:]))
    if op in _BINARY_OPS:
        if attrs:
            raise ModelUnsupportedError(
                f"{op}: attrs {sorted(attrs)} have no slot on the "
                "tensor surface"
            )
        if len(args) != 2:
            raise ModelUnsupportedError(
                f"{op}: arity {len(args)} has no surface signature"
            )
        return f"tensor.{op}({_emit_spec(args[0], names, amvars)}, {_emit_spec(args[1], names, amvars)})"
    if op == "concat":
        if set(attrs) - {"dim"}:
            raise ModelUnsupportedError(
                f"concat: attrs {sorted(attrs)} have no slot on the "
                "tensor surface"
            )
        inner = ", ".join(_emit_spec(a, names, amvars) for a in args)
        dim = _attr_slot(attrs.get("dim"), names, amvars)
        return f"tensor.concat([{inner}], {dim})"
    if op in _FIXED_OPS:
        want = _FIXED_OPS[op]
        if attrs:
            raise ModelUnsupportedError(
                f"{op}: attrs {sorted(attrs)} have no slot on the "
                "tensor surface"
            )
        if len(args) != want:
            raise ModelUnsupportedError(
                f"{op}: arity {len(args)} has no surface signature"
            )
        inner = ", ".join(_emit_spec(a, names, amvars) for a in args)
        return f"tensor.{op}({inner})"
    if op in _ATTR_OPS:
        return _attr_op_to_hnl(op, args, attrs, names, amvars)
    raise ModelUnsupportedError(
        f"op {op!r} is not on the tensor.* model surface"
    )


def _attr_op_to_hnl(
    op: str,
    args: tuple,
    attrs: dict,
    names: Mapping[str, str],
    amvars: frozenset = frozenset(),
) -> str:
    """Emit a shape-manipulation op whose attrs fill positional slots."""
    want = _ATTR_OPS[op]
    if set(attrs) != set(want):
        raise ModelUnsupportedError(
            f"{op}: attrs {sorted(attrs)} do not match the "
            f"surface signature {list(want)}"
        )
    if len(args) != 1:
        raise ModelUnsupportedError(
            f"{op}: arity {len(args)} has no surface signature"
        )
    slots = ", ".join(_attr_slot(attrs[k], names, amvars) for k in want)
    return f"tensor.{op}({_emit_spec(args[0], names, amvars)}, {slots})"


def _emit_spec(
    spec: Any,
    names: Mapping[str, str],
    amvars: frozenset = frozenset(),
) -> str:
    """Render one spec node as a ``.hnl`` expression."""
    if _is_call(spec):
        return _spec_call_to_hnl(spec, names, amvars)
    if isinstance(spec, str):
        if spec not in names:
            raise ModelUnsupportedError(
                f"metavar {spec!r} missing from the name map"
            )
        return names[spec]
    if isinstance(spec, dict):
        raise ModelUnsupportedError(
            "attr dict outside call position — not spec grammar"
        )
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return _num(spec)
    if spec is None or isinstance(spec, bool):
        return _lit(spec)
    if isinstance(spec, (tuple, list)):
        # A non-call list is data (e.g. a shape list) — emit literally.
        return _lit(list(spec))
    raise ModelUnsupportedError(f"unemittable spec leaf: {spec!r}")


def spec_to_hnl(
    spec: Any, *, mvars: Any = None, attr_mvars: Any = None
) -> str:
    """Render a genkernel spec as a ``.hnl`` expression over ``tensor.*``.

    *mvars* optionally pins the metavar→name map (see
    :func:`model_names`); when omitted it is derived from the spec.
    *attr_mvars* optionally pins the attr-metavariable set — attr
    dict values that are metavariable *names* render as ``.hnl``
    bindings (bound by :func:`hnl_program`'s *attr_bindings*), not
    string literals; when omitted the set is derived from the spec
    (:func:`spec_attr_mvars` — every string attr value is a metavar
    name under the ``pat_spec`` convention).  Raises
    :class:`ModelUnsupportedError` naming the first op or value the
    surface cannot honour — the caller records the row ``unsupported``
    rather than shipping a wrong model.
    """
    amvars = (
        frozenset(spec_attr_mvars(spec))
        if attr_mvars is None
        else frozenset(attr_mvars)
    )
    leaf = spec_mvars(spec) if mvars is None else set(mvars)
    names = _name_map(set(leaf) | set(amvars))
    return _emit_spec(spec, names, amvars)


def handler_body_to_hnl(body: Any, *, mvars: Any = None) -> str:
    """Render a handler-body descriptor as a ``.hnl`` expression.

    Only the ``ew`` (elementwise) family has a ``tensor.*`` spelling
    today — its body is a plain spec.  Every other family tag
    (``gather``/``family``/``contract``/``fold``/``reduce``/``norm``/
    ``fold_zip``/``fold_emit``) raises :class:`ModelUnsupportedError`
    naming the tag: those semantics need ops or control flow the
    surface does not carry.
    """
    if not isinstance(body, (tuple, list)) or not body:
        raise ModelUnsupportedError(f"malformed handler body: {body!r}")
    tag = body[0]
    if tag != "ew":
        raise ModelUnsupportedError(
            f"handler family {tag!r} has no tensor.* model"
        )
    return spec_to_hnl(body[1], mvars=mvars)


# ---------------------------------------------------------------------------
#  Program emission and result decoding — the hnl-run boundary
# ---------------------------------------------------------------------------


def tensor_feed(t: Any) -> dict:
    """Encode a torch tensor as the ``emit_program`` feed shape.

    ``{"shape": [d…], "data": [f…]}`` — flat row-major f64, the same
    marshaling ``term_hnl.emit_program`` writes for ``Var``/``Param``
    leaves.
    """
    import torch

    if not isinstance(t, torch.Tensor):
        raise ModelUnsupportedError(
            f"feed value is {type(t).__name__}, not a tensor"
        )
    return {
        "shape": [int(d) for d in t.shape],
        "data": [float(x) for x in t.detach().reshape(-1).tolist()],
    }


def hnl_program(
    hnl: str,
    feed: Mapping[str, Any],
    *,
    mvars: Any,
    attr_mvars: Any = (),
    attr_bindings: Mapping[str, Any] | None = None,
) -> str:
    """Bind *feed* and evaluate the model expression *hnl*.

    *hnl* is the :func:`spec_to_hnl` rendering — a ``tensor.*``
    expression over the sanitized metavar names.  *feed* maps metavar
    names to ``{"shape", "data"}`` dicts (or torch tensors, via
    :func:`tensor_feed`); every feed key binds in sorted order as
    ``name :: tensor.reshape(tensor.from_list([data]), [shape])``.
    *attr_mvars*/*attr_bindings* bind the spec's attr metavariables —
    each ``attr_bindings[name]`` literal binds first as
    ``name :: <lit>`` so attr positions in *hnl* read a real value,
    the same role the minted call's own attrs play in
    ``prim_mint.spec_bind``.  An attr metavar without a binding
    raises — a model whose attrs are unbound is not a program.
    """
    amvars = tuple(attr_mvars)
    names = _name_map(set(mvars) | set(amvars))
    lines = ["do {"]
    for a in sorted(amvars):
        if attr_bindings is None or a not in attr_bindings:
            raise ModelUnsupportedError(
                f"attr metavar {a!r} has no .hnl binding"
            )
        lines.append(f"  {names[a]} :: {_lit(attr_bindings[a])}")
    for m in sorted(feed):
        f = feed[m]
        if isinstance(f, Mapping) and "hnl" in f:
            expr = f["hnl"]
        elif isinstance(f, Mapping):
            shape = _lit(list(f["shape"]))
            data = _lit([float(x) for x in f["data"]])
            expr = f"tensor.reshape(tensor.from_list({data}), {shape})"
        else:
            expr = f"tensor.from_list({_lit(f)})"
        lines.append(f"  {names[m]} :: {expr}")
    lines.append(f"  {hnl}")
    lines.append("}")
    return "\n".join(lines)


def decode_tensor_result(value: Any) -> tuple[list[int], list[float]]:
    """Decode the ``{"__tensor__": …}`` wire form to ``(shape, data)``.

    Raises :class:`ModelUnsupportedError` on any other JSON shape —
    a model that does not return a tensor is a failed model, loudly.
    """
    t = value.get("__tensor__") if isinstance(value, dict) else None
    if not isinstance(t, dict):
        raise ModelUnsupportedError(
            f"expected a tensor result, got {value!r}"
        )
    shape = [int(d) for d in t["shape"]]
    data = [float(x) for x in t["data"]]
    return shape, data


def max_rel(
    got: list[float], ref: list[float], *, atol: float
) -> float:
    """Return ``max |got - ref| / (|ref| + atol)`` over paired elements.

    The relative-error half of the shipped fp64 floor: ``allclose``
    passes when ``|d| ≤ atol + rtol·|ref|`, so this normalizes by the
    same ``|ref| + atol`` denominator and stays honest at ref = 0.
    """
    if len(got) != len(ref):
        raise ValueError("max_rel over mismatched element counts")
    worst = 0.0
    for g, r in zip(got, ref, strict=True):
        worst = max(worst, abs(g - r) / (abs(r) + atol))
    return worst


# ---------------------------------------------------------------------------
#  The fp64 gate — verify_model
# ---------------------------------------------------------------------------


def _torch_allclose(
    got: Any, ref: Any, rtol: float, atol: float
) -> bool:
    """``torch.allclose`` on the decoded handl result vs the torch ref."""
    import torch

    shape, data = got
    g = torch.tensor(data, dtype=torch.float64).reshape(shape)
    return bool(torch.allclose(g, ref, rtol=rtol, atol=atol))


def verify_model(
    *,
    name: str,
    hnl: str,
    ref_spec: Any,
    domain: Any,
    mvars: Any,
    eval_hnl: Callable[[str], Any],
    bindings: dict | None = None,
    attr_bindings: Mapping[str, Any] | None = None,
    seed: int = 0,
    adversarial: int = 1,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> dict:
    """Run the phase-2 fp64 gate for one Model; return the verdict.

    *ref_spec* is the torch-side reference — the spec the model
    denotes (a prim denotation verifies the spec against itself; a
    minted handler verifies its ``handler_body`` against the handler
    *pattern*).  *domain* is the measured bound-site shapes
    (``{mvar: shape}`` dicts, the ``provenance.domain`` convention);
    :func:`handler_synth._domain_draws` draws the same seeded fp64
    operands plus *adversarial* re-draws every candidate saw.
    *eval_hnl* maps a ``.hnl`` program source to the decoded JSON
    boundary value (``KitRunner.run`` in the driver; a stub in
    tests).

    Specs whose attr dicts hold attr-metavariable *names*
    (:func:`spec_attr_mvars` — e.g. ``{"dim": "a97"}``) need
    *attr_bindings* — the measured ``{name: value}`` pairs the
    representative bound site froze (``prim_mint.freeze_spec``'s
    ``sites[0]`` convention).  The reference spec binds them through
    ``prim_mint.spec_bind`` before drawing, and the ``.hnl`` program
    binds them as literal prelude lines — an attr metavar with no
    recorded binding refuses honestly rather than evaluating a
    string where torch wants an int.

    The verdict is ``handl-shipped`` only when every draw decodes to
    a same-shaped tensor that ``torch.allclose``s the reference at
    the shipped floor; anything else is ``python-shipped`` with the
    measured ``max_err``/``max_rel`` and a ``detail`` naming the
    refusal — the ADR-0157 provenance record.
    """
    import torch

    from mlopt_discovery import handler_synth

    if bindings is None:
        import mlopt_torch.torch_bridge as tb

        bindings = tb._IR_TO_TORCH
    verdict: dict[str, Any] = {
        "op": name,
        "status": "python-shipped",
        "rtol": rtol,
        "atol": atol,
        "max_err": float("inf"),
        "max_rel": float("inf"),
        "sites": 0,
    }
    amvars = spec_attr_mvars(ref_spec)
    if amvars:
        ab = dict(attr_bindings) if attr_bindings is not None else {}
        missing = sorted(a for a in amvars if a not in ab)
        if missing:
            verdict["detail"] = (
                f"unbound attr metavars: {missing} — no recorded "
                "attr binding for this model's domain"
            )
            return verdict
        from mlopt_discovery.prim_mint import spec_bind

        ref_spec = spec_bind(ref_spec, ab, tuple(amvars))
        verdict["attr_bindings"] = {a: ab[a] for a in sorted(amvars)}
    try:
        envs, refs = handler_synth._domain_draws(
            ref_spec, domain, seed, bindings, adversarial
        )
    except handler_synth.DomainShapeError as exc:
        verdict["detail"] = f"no drawable domain entry: {exc}"
        return verdict
    max_err = 0.0
    worst_rel = 0.0
    n = 0
    for env, ref in zip(envs, refs, strict=True):
        feed = {m: tensor_feed(t) for m, t in env.items()}
        src = hnl_program(
            hnl,
            feed,
            mvars=mvars,
            attr_mvars=amvars,
            attr_bindings=attr_bindings,
        )
        try:
            got = decode_tensor_result(eval_hnl(src))
        except Exception as exc:
            verdict["detail"] = (
                f"hnl eval failed: {type(exc).__name__}: {exc}"
            )
            verdict["sites"] = n
            verdict["max_err"] = max_err
            verdict["max_rel"] = worst_rel
            return verdict
        if got[0] != [int(d) for d in ref.shape]:
            verdict["detail"] = (
                f"shape mismatch: handl {got[0]} vs torch "
                f"{list(ref.shape)}"
            )
            verdict["sites"] = n
            verdict["max_err"] = max_err
            verdict["max_rel"] = worst_rel
            return verdict
        g = torch.tensor(got[1], dtype=torch.float64).reshape(got[0])
        err = float((g - ref).abs().max()) if ref.numel() else 0.0
        max_err = max(max_err, err)
        rel = max_rel(got[1], ref.reshape(-1).tolist(), atol=atol)
        worst_rel = max(worst_rel, rel)
        n += 1
        if not _torch_allclose(got, ref, rtol, atol):
            verdict["detail"] = "numeric miss"
            verdict["sites"] = n
            verdict["max_err"] = max_err
            verdict["max_rel"] = worst_rel
            return verdict
    verdict.update(
        status="handl-shipped",
        max_err=max_err,
        max_rel=worst_rel,
        sites=n,
        detail="",
    )
    return verdict

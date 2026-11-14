"""Regression tests for the arch-audit silent-wrong fixes (B1-B7).

``docs/reports/arch_audit_graphopt_packages.md`` §1 catalogued the
``torch_bridge.py`` export/lower boundary bugs; this file pins each
fix:

* **B1** — FX-node and tuple kwargs were silently dropped by the
  ``node.kwargs`` loop.  torch.export on 2.14 normalises declared-
  schema kwargs into positional args (verified: ``F.pad`` pad list,
  ``F.conv2d`` stride/padding, sdpa ``attn_mask`` all arrive
  positionally), so the latent path is exercised through hand-built
  FX graphs — the same machinery ``_opaque_encode`` already uses to
  carry both shapes correctly.
* **B2** — an unhandled ``node.op`` kind (``call_module`` /
  ``call_method`` / future kinds) minted no env entry and every
  downstream operand silently dropped; now a loud ``ValueError``.
* **B3** — ``_handle_copy_``'s silent early-returns dropped
  ``copy_``/``fill_``/``zero_``/``masked_fill_`` writes whose target
  or view-chain base never minted a term; now ``ValueError`` rejections
  matching ``masked_fill_``'s deliberate ``KeyError``.
* **B4** — ``_batch_norm_torch`` read ``(w, b)`` as ``(rm, rv)`` when
  ``track_running_stats=False`` exported the running stats as ``None``.
  Export now tags the spelling ``running_stats=False`` (len-2 is
  ambiguous: ``(w, b)`` vs ``(rm, rv)``) and the binding honours it.
* **B5** — ``_where_torch`` ignored surplus children beyond the two
  operand slots; now ``TypeError``.
* **B6** — ``_dim_args`` returned a raw evaluated ``dim`` — a
  hand-minted 0-dim Const arrives as a 0-dim tensor; now unwrapped.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from mlopt_core.ir import Op
from mlopt_torch.torch_bridge import (
    _batch_norm_torch,
    _dim_args,
    _handle_copy_,
    _where_torch,
    export_to_ir,
    ir_to_torch_module,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
#  hand-built FX graph helpers — torch.export normalises kwargs to
#  positionals on 2.14, so the latent kwarg paths are driven through a
#  stubbed ``torch.export.export``.
# ---------------------------------------------------------------------------


def _fx(op, name, target=None, args=(), kwargs=None, meta=None):
    return SimpleNamespace(
        op=op,
        name=name,
        target=target,
        args=tuple(args),
        kwargs=dict(kwargs or {}),
        meta=meta if meta is not None else {},
    )


def _placeholder(name):
    return _fx("placeholder", name, target=name)


def _output(*leaves):
    return _fx("output", "output", args=(tuple(leaves),))


def _fake_export(nodes):
    return SimpleNamespace(
        graph=SimpleNamespace(nodes=list(nodes)),
        module=lambda: torch.nn.Module(),
        state_dict={},
        constants={},
        _graph_signature=SimpleNamespace(
            inputs_to_parameters={},
            inputs_to_buffers={},
            inputs_to_lifted_tensor_constants={},
        ),
    )


def _export_graph(monkeypatch, nodes, example):
    fake = _fake_export(nodes)
    monkeypatch.setattr(torch.export, "export", lambda *a, **k: fake)
    return export_to_ir(torch.nn.Module(), example)


# ---------------------------------------------------------------------------
#  B1 — kwargs carrying FX nodes / tuples / unencodable values
# ---------------------------------------------------------------------------


def test_kwarg_fx_node_operand_is_not_dropped(monkeypatch):
    """``attn_mask=m`` spelled as a KWARG must reach the term as an
    operand — the pre-fix loop dropped it and the mask silently
    vanished (unmasked attention)."""
    q = _placeholder("q")
    k = _placeholder("k")
    v = _placeholder("v")
    m = _placeholder("m")
    sdpa = _fx(
        "call_function",
        "sdpa",
        target=torch.ops.aten.scaled_dot_product_attention.default,
        args=(q, k, v),
        kwargs={"attn_mask": m},
    )
    ir, _ = _export_graph(
        monkeypatch,
        [q, k, v, m, sdpa, _output(sdpa)],
        (
            torch.randn(2, 4, 8, 16),
            torch.randn(2, 4, 8, 16),
            torch.randn(2, 4, 8, 16),
            torch.randn(2, 4, 8, 8),
        ),
    )
    term = ir.root
    assert isinstance(term, Op) and term.op == "sdpa"
    # The kwarg-carried mask is the 4th operand, in aten order.
    assert len(term.args) == 4
    lowered = ir_to_torch_module(ir)
    qq = torch.randn(2, 4, 8, 16)
    kk = torch.randn(2, 4, 8, 16)
    vv = torch.randn(2, 4, 8, 16)
    mm = torch.randn(2, 4, 8, 8)
    with torch.no_grad():
        got = lowered(qq, kk, vv, mm)
        want = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mm)
    assert torch.allclose(got, want)


def test_kwarg_fx_node_unminted_raises(monkeypatch):
    """A kwarg referencing a node that minted no term refuses loudly —
    the ``_opaque_encode``/``masked_fill_`` contract, not a drop."""
    x = _placeholder("x")
    ghost = SimpleNamespace(name="ghost")
    mul = _fx(
        "call_function",
        "mul",
        target=torch.ops.aten.mul.Tensor,
        args=(x,),
        kwargs={"other": ghost},
    )
    with pytest.raises(ValueError, match="unminted node 'ghost'"):
        _export_graph(
            monkeypatch,
            [x, mul, _output(mul)],
            (torch.randn(4),),
        )


def test_tuple_kwarg_lands_as_attr(monkeypatch):
    """``pad=(1,1,1,1)`` spelled as a tuple KWARG records the attr —
    pre-fix ``isinstance(v, list)`` dropped every tuple silently and
    the binding fell back to aten's empty-pad default."""
    x = _placeholder("x")
    pad = _fx(
        "call_function",
        "pad",
        target=torch.ops.aten.pad.default,
        args=(x,),
        # ``value=None`` exercises the explicit-None skip on the
        # same node; ``layout`` exercises the torch.layout channel.
        kwargs={
            "pad": (1, 1, 1, 1),
            "value": None,
            "layout": torch.strided,
        },
    )
    ir, _ = _export_graph(
        monkeypatch,
        [x, pad, _output(pad)],
        (torch.randn(1, 1, 4, 4),),
    )
    assert dict(ir.root.attrs)["pad"] == (1, 1, 1, 1)
    assert dict(ir.root.attrs)["layout"] == "strided"
    assert "value" not in dict(ir.root.attrs)
    lowered = ir_to_torch_module(ir)
    xv = torch.randn(1, 1, 4, 4)
    with torch.no_grad():
        got = lowered(xv)
    assert torch.equal(got, F.pad(xv, (1, 1, 1, 1)))


def test_str_and_memory_format_kwargs_recorded(monkeypatch):
    """String flags and ``torch.memory_format`` values are semantics/
    dispatch detail — recorded as JSON-safe strings, not dropped."""
    x = _placeholder("x")
    pad = _fx(
        "call_function",
        "pad",
        target=torch.ops.aten.pad.default,
        args=(x,),
        kwargs={
            "pad": (1, 1, 1, 1),
            "mode": "replicate",
            "memory_format": torch.channels_last,
        },
    )
    ir, _ = _export_graph(
        monkeypatch,
        [x, pad, _output(pad)],
        (torch.randn(1, 1, 4, 4),),
    )
    attrs = dict(ir.root.attrs)
    assert attrs["mode"] == "replicate"
    assert attrs["memory_format"] == "channels_last"
    lowered = ir_to_torch_module(ir)
    xv = torch.randn(1, 1, 4, 4)
    with torch.no_grad():
        got = lowered(xv)
    assert torch.equal(got, F.pad(xv, (1, 1, 1, 1), mode="replicate"))


def test_kwarg_container_with_fx_nodes_raises(monkeypatch):
    """A tuple/list kwarg carrying FX nodes cannot cross the attr
    channel — refuse rather than drop nested tensor operands."""
    x = _placeholder("x")
    y = _placeholder("y")
    weird = _fx(
        "call_function",
        "mul",
        target=torch.ops.aten.mul.Tensor,
        args=(x,),
        kwargs={"pads": (y,)},
    )
    with pytest.raises(ValueError, match="container carrying FX nodes"):
        _export_graph(
            monkeypatch,
            [x, y, weird, _output(weird)],
            (torch.randn(4), torch.randn(4)),
        )


def test_unencodable_kwarg_raises(monkeypatch):
    """A kwarg value outside the encodable set (here a bare object)
    refuses loudly instead of silently vanishing."""
    x = _placeholder("x")
    weird = _fx(
        "call_function",
        "mul",
        target=torch.ops.aten.mul.Tensor,
        args=(x,),
        kwargs={"mystery": object()},
    )
    with pytest.raises(ValueError, match="unencodable kwarg"):
        _export_graph(
            monkeypatch,
            [x, weird, _output(weird)],
            (torch.randn(4),),
        )


# ---------------------------------------------------------------------------
#  B2 — unhandled FX node kinds refuse loudly
# ---------------------------------------------------------------------------


def test_unhandled_node_op_raises(monkeypatch):
    """``call_module``/``call_method`` (or any future kind) minted no
    env entry — downstream operands silently dropped.  Now refused."""
    x = _placeholder("x")
    cm = _fx(
        "call_module",
        "sub",
        target="sub",
        args=(x,),
    )
    with pytest.raises(ValueError, match="unhandled FX node kind"):
        _export_graph(
            monkeypatch,
            [x, cm, _output(cm)],
            (torch.randn(4),),
        )


def test_output_node_kind_is_allowed(monkeypatch):
    """The ``output`` kind is the one non-minting node that is legal —
    it resolves post-loop via ``_graph_root``."""
    x = _placeholder("x")
    ir, _ = _export_graph(
        monkeypatch,
        [x, _output(x)],
        (torch.randn(4),),
    )
    assert isinstance(ir.root, Op) or hasattr(ir.root, "name")


# ---------------------------------------------------------------------------
#  B3 — _handle_copy_ silent early-returns are now rejections
# ---------------------------------------------------------------------------


def _copy_node(name="w", dst=None, src=None, target="copy_"):
    args = []
    if dst is not None:
        args.append(dst)
    if src is not None:
        args.append(src)
    return SimpleNamespace(name=name, target=target, args=tuple(args))


def test_handle_copy_raises_on_empty_args():
    node = _copy_node(dst=None, src=None)
    with pytest.raises(ValueError, match="no operands"):
        _handle_copy_(node, {})


def test_handle_copy_raises_on_unminted_dst():
    dst = SimpleNamespace(name="ghost")
    src = SimpleNamespace(name="s")
    node = _copy_node(dst=dst, src=src)
    with pytest.raises(ValueError, match="minted no IR term"):
        _handle_copy_(node, {"s": torch.randn(2)})


def test_handle_copy_raises_on_unminted_src():
    dst = SimpleNamespace(name="d")
    src = SimpleNamespace(name="ghost")
    node = _copy_node(dst=dst, src=src)
    x = torch.randn(2)
    with pytest.raises(ValueError, match="copies"):
        _handle_copy_(node, {"d": x})


def test_handle_copy_raises_on_unminted_view_root():
    """A write through a ``slice`` view whose ROOT minted no term
    refuses — the mutation cannot reach the tensor readers observe."""
    root = SimpleNamespace(name="r")  # unminted
    view = SimpleNamespace(
        name="v", target="slice", args=(root, 1, 0, 3)
    )
    src = SimpleNamespace(name="s")
    node = _copy_node(dst=view, src=src)
    x = torch.randn(4, 4)
    env = {"v": x, "s": x}  # "r" deliberately absent
    with pytest.raises(ValueError, match="view chain rooted"):
        _handle_copy_(node, env)


def test_handle_copy_raises_on_unminted_intermediate_view():
    """The pre-fix silent decline: a view chain whose middle step never
    minted a binding aborted quietly.  Now it rejects loudly."""
    x = torch.randn(4, 4)
    y = SimpleNamespace(name="y", target=None)
    v1 = SimpleNamespace(name="v1", target="slice", args=(y, 1, 0, 3))
    v2 = SimpleNamespace(name="v2", target="slice", args=(v1, 2, 0, 2))
    node = _copy_node(dst=v2, src=SimpleNamespace(name="s"))
    env = {"y": x, "v2": x, "s": x}  # "v1" deliberately unminted
    with pytest.raises(ValueError, match="minted no IR term"):
        _handle_copy_(node, env)


# ---------------------------------------------------------------------------
#  B4 — batch_norm track_running_stats=False
# ---------------------------------------------------------------------------


class _BN(torch.nn.Module):
    def __init__(self, track, affine=True):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(
            4, track_running_stats=track, affine=affine
        )

    def forward(self, x):
        return self.bn(x)


def test_batch_norm_no_track_export_roundtrip():
    """``track_running_stats=False`` exports ``running_mean``/
    ``running_var`` as None — pre-fix the binding read ``(w, b)`` as
    the running stats.  The tagged term now lowers correctly."""
    m = _BN(False).eval()
    x = torch.randn(2, 4, 4, 4)
    ir, source = export_to_ir(m, x)
    bn_terms = []

    def walk(t):
        if isinstance(t, Op):
            if t.op == "batch_norm":
                bn_terms.append(t)
            for a in t.args:
                walk(a)

    walk(ir.root)
    assert len(bn_terms) == 1
    assert dict(bn_terms[0].attrs).get("running_stats") is False
    # Only the affine pair survives as operands.
    assert len(bn_terms[0].args) == 3  # x, w, b
    lowered = ir_to_torch_module(ir, param_values=source)
    with torch.no_grad():
        got = lowered(x)
        want = m(x)
    assert torch.allclose(got, want)


def test_batch_norm_no_track_affine_false():
    """``affine=False`` + ``track_running_stats=False`` leaves ZERO
    operands after x — pre-fix ``a[-2]`` raised IndexError; the
    tagged path evaluates the all-None F.batch_norm spelling."""
    m = _BN(False, affine=False).eval()
    x = torch.randn(2, 4, 4, 4)
    ir, source = export_to_ir(m, x)
    lowered = ir_to_torch_module(ir, param_values=source)
    with torch.no_grad():
        got = lowered(x)
        want = m(x)
    assert torch.allclose(got, want)


def test_batch_norm_track_still_works():
    """The stats-present spelling is unchanged — no ``running_stats``
    attr is minted and the last-two heuristic still applies."""
    m = _BN(True).eval()
    x = torch.randn(2, 4, 4, 4)
    ir, source = export_to_ir(m, x)
    bn_terms = []

    def walk(t):
        if isinstance(t, Op):
            if t.op == "batch_norm":
                bn_terms.append(t)
            for a in t.args:
                walk(a)

    walk(ir.root)
    assert len(bn_terms) == 1
    assert "running_stats" not in dict(bn_terms[0].attrs)
    assert len(bn_terms[0].args) == 5  # x, w, b, rm, rv
    lowered = ir_to_torch_module(ir, param_values=source)
    with torch.no_grad():
        got = lowered(x)
        want = m(x)
    assert torch.allclose(got, want)


def test_batch_norm_partial_stats_slot_still_tags_absent(monkeypatch):
    """The ``running_stats`` probe checks BOTH aten slots 3,4 — a
    graph carrying only one of them (hand-built; real aten emits
    both-or-neither) still tags the spelling absent."""
    x = _placeholder("x")
    w = _placeholder("w")
    b = _placeholder("b")
    rm = _placeholder("rm")
    # args[3]=rm minted, args[4]=None dropped — stats half-present.
    bn = _fx(
        "call_function",
        "bn",
        target=torch.ops.aten.batch_norm.default,
        args=(x, w, b, rm, None, True, 0.1, 1e-5, False),
    )
    ir, _ = _export_graph(
        monkeypatch,
        [x, w, b, rm, bn, _output(bn)],
        (
            torch.randn(2, 4, 4, 4),
            torch.randn(4),
            torch.randn(4),
            torch.randn(4),
        ),
    )
    assert dict(ir.root.attrs).get("running_stats") is False


def test_batch_norm_truncated_args_tag_absent(monkeypatch):
    """A batch_norm node too short to reach aten slots 3,4 (hand-
    built) is tagged ``running_stats=False`` — the stats positions
    cannot be confirmed present."""
    x = _placeholder("x")
    w = _placeholder("w")
    b = _placeholder("b")
    bn = _fx(
        "call_function",
        "bn",
        target=torch.ops.aten.batch_norm.default,
        args=(x, w, b),
    )
    ir, _ = _export_graph(
        monkeypatch,
        [x, w, b, bn, _output(bn)],
        (
            torch.randn(2, 4, 4, 4),
            torch.randn(4),
            torch.randn(4),
        ),
    )
    assert dict(ir.root.attrs).get("running_stats") is False


def test_batch_norm_binding_running_stats_false_surplus():
    """A ``running_stats=False`` term carrying >2 operands is
    malformed — the stats slots cannot reappear; refuse loudly."""
    x = torch.randn(2, 4)
    w = torch.randn(4)
    b = torch.randn(4)
    extra = torch.randn(4)
    with pytest.raises(TypeError, match="running_stats=False"):
        _batch_norm_torch(
            x, w, b, extra, running_stats=False, training=True
        )


# ---------------------------------------------------------------------------
#  B5 — _where_torch surplus children
# ---------------------------------------------------------------------------


def test_where_torch_surplus_children_raise():
    c = torch.tensor([True, False])
    x = torch.randn(2)
    y = torch.randn(2)
    z = torch.randn(2)
    with pytest.raises(TypeError, match="surplus children"):
        _where_torch(c, x, y, z)


def test_where_torch_two_children_still_work():
    c = torch.tensor([True, False])
    x = torch.randn(2)
    y = torch.randn(2)
    out = _where_torch(c, x, y)
    assert torch.equal(out, torch.where(c, x, y))


# ---------------------------------------------------------------------------
#  B6 — _dim_args 0-dim tensor unwrap
# ---------------------------------------------------------------------------


def test_dim_args_unwraps_0dim_tensor():
    """A hand-minted Const-evaluated dim arrives as a 0-dim tensor —
    unwrap to the Python scalar the aten call expects."""
    assert _dim_args((), {"dim": torch.tensor(1)}) == (1, False)
    assert _dim_args(
        (), {"dim": torch.tensor(-1), "keepdim": True}
    ) == (
        -1,
        True,
    )
    # A non-scalar tensor stays raw (fails loudly downstream, as
    # before) — and plain ints are untouched.
    assert _dim_args((), {"dim": 2}) == (2, False)

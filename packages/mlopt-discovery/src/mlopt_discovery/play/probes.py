"""The generated-kernel timing probes — CUDA + triton arms.

``scan_probe`` times the generated Triton scan fold against the
spelled loop and the carrier executor; ``decay_probe`` times the
fused ``(s·d)@v`` kernel against eager, ``torch.compile`` and the
``om_decayed`` executor.  Both decline honestly without CUDA/triton.
"""

from __future__ import annotations

from typing import Any

from mlopt_core.ir import Op, TensorType, Var

from .gen import _has_triton, _time_module

# ---------------------------------------------------------------------------
#  The scan-tier probe — generated fold kernel vs today's delivery
# ---------------------------------------------------------------------------


def _spelled_scan(a: Any, b: Any, h: Any) -> Any:
    """Fold the torch-loop spelling: ``h_t = a_t*h + b_t`` (axis 1)."""
    for t in range(a.shape[1]):
        h = a.select(1, t) * h + b.select(1, t)
    return h


def _scan_carrier_ir(batch: int, steps: int, width: int) -> Any:
    """``applyd(<balanced affd_compose tree>, h)`` over select leaves.

    The form ``affd`` lifts leave behind — per-step ``aff_diag``
    leaves bracketed into a balanced compose tree (in-order leaves
    are reverse-chronological, so later steps sit in the LEFT
    subtree: ``affd_compose(f, g)`` applies ``g`` first).
    """
    from mlopt_core.ir import IR

    a = Var("a", TensorType((batch, steps, width)))
    b = Var("b", TensorType((batch, steps, width)))
    h = Var("h", TensorType((batch, width)))
    leaves = [
        Op.make(
            "aff_diag",
            Op.make("select", a, dim=1, index=t),
            Op.make("select", b, dim=1, index=t),
        )
        for t in range(steps)
    ]

    def tree(ls: list) -> Any:
        if len(ls) == 1:
            return ls[0]
        k = len(ls) // 2
        return Op.make("affd_compose", tree(ls[k:]), tree(ls[:k]))

    return IR(
        root=Op.make("applyd", tree(leaves), h),
        inputs=[a, b, h],
        input_names={"a", "b", "h"},
        params={},
    )


#: Default probe shapes — AdaLNBlock-style ``(B, T, d)`` fp64 scans.
_SCAN_PROBE_SHAPES = ((2, 64, 128), (8, 128, 512))


def scan_probe(
    *,
    shapes: Any = None,
    reps: int = 100,
    fused: bool = True,
) -> list[dict]:
    """Time the generated scan kernel against today's delivery paths.

    The honest-number arm for the scan tier: per ``(B, T, d)`` shape
    — fp64 CUDA, AdaLNBlock-style token axis — the row reports median
    µs for the spelled torch loop (eager and ``torch.compile``-fused),
    the carrier executor path (generic ``TorchSink`` tuple-passing
    eval of the ``affd_compose`` tree, the level-batched
    ``to_batched_scan_module`` slot-gather schedule, and — with
    *fused* — its ``scan_fused`` compiled schedule), and the
    generated Triton fold kernel from
    :func:`~mlopt_discovery.genkernel.scan_triton_bindings`.
    Every arm is verified against the spelled loop before timing.
    On a host without CUDA+triton the row reports ``delivered:
    False`` honestly.
    """
    import torch
    from kit_carriers.scan_lower import to_batched_scan_module
    from mlopt_torch.adapters import TorchSink

    from mlopt_discovery import genkernel as gk

    if shapes is None:
        shapes = _SCAN_PROBE_SHAPES
    if not (torch.cuda.is_available() and _has_triton()):
        return [{"delivered": False, "note": "needs cuda + triton"}]
    _dyno_cfg: Any = torch._dynamo.config
    _dyno_cfg.recompile_limit = max(int(_dyno_cfg.recompile_limit), 256)
    kern = gk.scan_triton_bindings(
        {
            "s": {
                "pattern": ("applyd", ("aff_diag", "A", "B"), "H"),
                "kernel": "sc_probe",
                "args": ("A", "B", "H"),
            }
        }
    )["sc_probe"]
    compiled = torch.compile(_spelled_scan)
    dt, dev = torch.float64, "cuda"
    rows: list[dict] = []
    for batch, steps, width in shapes:
        a = torch.rand(batch, steps, width, device=dev, dtype=dt)
        a = a * 0.5 + 0.3
        b = torch.randn(batch, steps, width, device=dev, dtype=dt)
        h = torch.randn(batch, width, device=dev, dtype=dt)
        want = _spelled_scan(a, b, h)
        ir = _scan_carrier_ir(batch, steps, width)
        generic = TorchSink().lower(ir, params={})
        batched = to_batched_scan_module(ir)
        arms: dict[str, Any] = {
            "spelled": _spelled_scan,
            "spelled_compiled": compiled,
            "executor_generic": generic,
            "executor_batched": batched,
            "triton_scan": kern,
        }
        if fused:
            arms["executor_fused"] = to_batched_scan_module(
                ir, fused="compile"
            )
        row: dict[str, Any] = {
            "shape": (batch, steps, width),
            "delivered": True,
        }
        for name, fn in arms.items():
            torch.testing.assert_close(
                fn(a, b, h), want, atol=1e-9, rtol=1e-9
            )
            row[f"{name}_us"] = _time_module(fn, (a, b, h), reps)
        row["vs_spelled"] = round(
            row["spelled_us"] / row["triton_scan_us"], 3
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
#  The decayed-tier probe — fused (s ⊙ d) @ v vs today's delivery
# ---------------------------------------------------------------------------


def _spelled_decay(s: Any, d: Any, v: Any) -> Any:
    """Evaluate the spelled body: ``(s * d) @ v``, verbatim."""
    return (s * d) @ v


def _decay_carrier_ir(batch: int, steps: int, width: int) -> Any:
    """``om_decayed(s, d, v)`` over leaf vars — the lifted site.

    RetNet inner-chunk layout: scores ``(B, T, T)``, a ``(T, T)``
    decay matrix broadcast over the batch, values ``(B, T, dv)``.
    """
    from mlopt_core.ir import IR

    s = Var("s", TensorType((batch, steps, steps)))
    d = Var("d", TensorType((steps, steps)))
    v = Var("v", TensorType((batch, steps, width)))
    return IR(
        root=Op.make("om_decayed", s, d, v),
        inputs=[s, d, v],
        input_names={"s", "d", "v"},
        params={},
    )


#: Default probe shapes — ``(B, T, dv)`` RetNet inner-chunk fp64
#: sites.  The last is the memory-bound case: the spelled mul
#: materialises a ``(8, 512, 512)`` fp64 intermediate (~134 MB).
_DECAY_PROBE_SHAPES = (
    (2, 64, 128),
    (8, 128, 256),
    (4, 256, 128),
    (8, 512, 512),
)


def decay_probe(
    *,
    shapes: Any = None,
    reps: int = 100,
) -> list[dict]:
    """Time the generated decayed kernel vs today's delivery paths.

    The honest-number arm for the decayed tier: per ``(B, T, dv)``
    shape — fp64 CUDA, ``s`` ``(B,T,T)``, broadcast decay ``d``
    ``(T,T)``, ``v`` ``(B,T,dv)`` — the row reports median µs for
    the spelled eager ``(s*d)@v``, its ``torch.compile`` (inductor
    may already fuse the mul into the matmul — if it wins, the row
    says so), the reference executor path (generic ``TorchSink``
    eval of ``om_decayed`` — the spelled body verbatim), and the
    generated Triton kernel from
    :func:`~mlopt_discovery.genkernel.decayed_triton_bindings`.
    Every arm is verified against the spelled body before timing.
    On a host without CUDA+triton the row reports ``delivered:
    False`` honestly.
    """
    import torch
    from mlopt_torch.adapters import TorchSink

    from mlopt_discovery import genkernel as gk

    if shapes is None:
        shapes = _DECAY_PROBE_SHAPES
    if not (torch.cuda.is_available() and _has_triton()):
        return [{"delivered": False, "note": "needs cuda + triton"}]
    _dyno_cfg: Any = torch._dynamo.config
    _dyno_cfg.recompile_limit = max(int(_dyno_cfg.recompile_limit), 256)
    kern = gk.decayed_triton_bindings(
        {
            "d": {
                "pattern": ("om_decayed", "S", "D", "V"),
                "kernel": "dc_probe",
                "args": ("S", "D", "V"),
            }
        }
    )["dc_probe"]
    compiled = torch.compile(_spelled_decay)
    dt, dev = torch.float64, "cuda"
    rows: list[dict] = []
    for batch, steps, width in shapes:
        s = torch.randn(batch, steps, steps, device=dev, dtype=dt)
        d = torch.rand(steps, steps, device=dev, dtype=dt)
        v = torch.randn(batch, steps, width, device=dev, dtype=dt)
        want = _spelled_decay(s, d, v)
        executor = TorchSink().lower(
            _decay_carrier_ir(batch, steps, width), params={}
        )
        arms: dict[str, Any] = {
            "spelled": _spelled_decay,
            "spelled_compiled": compiled,
            "executor": executor,
            "triton_decayed": kern,
        }
        row: dict[str, Any] = {
            "shape": (batch, steps, width),
            "delivered": True,
        }
        for name, fn in arms.items():
            torch.testing.assert_close(
                fn(s, d, v), want, atol=1e-9, rtol=1e-9
            )
            row[f"{name}_us"] = _time_module(fn, (s, d, v), reps)
        row["vs_spelled"] = round(
            row["spelled_us"] / row["triton_decayed_us"], 3
        )
        row["vs_compiled"] = round(
            row["spelled_compiled_us"] / row["triton_decayed_us"], 3
        )
        rows.append(row)
    return rows

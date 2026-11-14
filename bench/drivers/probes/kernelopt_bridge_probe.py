"""kernelopt-bridge probe — SCRATCH measurement driver, uncommitted.

Question: can the kernelopt path (``mlopt_kernel.KERNEL_RULES`` +
``KernelSink`` fused-op lowering) beat graphopt's current delivery
(``Optimizer.search`` + ``TorchBackend`` lower) on whisper hot
subterms?

Both repos consume the same ``mlopt_core.ir`` term IR (eqsat
``v0.1.0`` pin), so the bridge is: export the subterm once through
graphopt's ``TorchSource``, then run the *same spelled IR* through
both sinks.  ``mlopt_kernel``/``lib_kernel_ir`` are imported from the
sibling checkouts via ``sys.path`` — no install, no repo mutation.

Subterms are pattern-representative spellings at whisper-base shapes
(d_model=512, ffn=2048, 8 heads x 64, fp32, CUDA):

- ``attn_core``   — ``softmax((q@k^T)*scale) @ v`` attention fragment
- ``mlp_block``   — ``x + fc2(gelu(fc1(layer_norm(x))))``
- ``conv_block``  — conv1d -> gelu -> conv1d (encoder frontend)
- ``o_proj``      — ``x + linear(x)`` residual projection
- ``ln_res``      — ``x + layer_norm(x)`` norm+residual fragment

Usage::

    .venv/bin/python bench/drivers/probes/kernelopt_bridge_probe.py [name ...]
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

_KERNELOPT = "/home/luis/GitHub/handl-lang/kernelopt/src"
_LIBKERNELIR = "/home/luis/GitHub/handl-lang/lib-kernel-ir/src"
for _p in (_LIBKERNELIR, _KERNELOPT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mlopt_kernel import KERNEL_RULES, KernelSink, kernel_cost  # noqa: E402
from mlopt_core.egraph import EGraph  # noqa: E402
from mlopt_core.ir import IR, Const, Op, Param  # noqa: E402


# ------------------------------------------------------------------
# decomposition shim — the integration-cost measurement
# ------------------------------------------------------------------
# torch.export emits ATOMIC ``softmax`` / ``layer_norm`` / ``linear``
# ops; the kernelopt law bank consumes the *decomposed* spellings
# (``div(exp,Σexp)``, ``c·rsqrt(mean(c²)+eps)``, ``matmul``+``add``).
# ``_decompose`` rewrites those three ops into the shapes the laws
# match — the honest price of wiring kernelopt onto real exports.


def _decompose(t, memo=None):
    """Rewrite atomic norm/softmax/linear ops into law-visible form."""
    if memo is None:
        memo = {}
    k = id(t)
    if k in memo:
        return memo[k]
    if not isinstance(t, Op):
        return t
    args = [_decompose(a, memo) for a in t.args]
    memo[k] = t  # guard cycles (there are none, but cheap)
    op = t.op
    attrs = dict(t.attrs)
    if op == "softmax":
        u = args[0]
        dim = attrs.get("dim", -1)
        e = Op.make("exp", u)
        out = Op.make(
            "div",
            e,
            Op.make("sum", e, dim=dim, keepdim=True),
        )
    elif op == "layer_norm" and len(args) >= 3:
        x, w, b = args[0], args[1], args[2]
        # ``dim`` attr is the normalized SHAPE (512,), not an axis —
        # layer_norm reduces the last len(dim) axes.
        shape = attrs.get("dim", (-1,))
        n_axes = len(shape) if isinstance(shape, (tuple, list)) else 1
        dim = tuple(range(-n_axes, 0))
        eps = float(attrs.get("eps", 1e-5))
        c = Op.make(
            "sub", x, Op.make("mean", x, dim=dim, keepdim=True)
        )
        var = Op.make(
            "add",
            Op.make(
                "mean",
                Op.make("pow", c, Const(2)),
                dim=dim,
                keepdim=True,
            ),
            Const(eps),
        )
        n = Op.make("mul", c, Op.make("rsqrt", var))
        out = Op.make("add", Op.make("mul", n, w), b)
    elif op == "linear":
        # F.linear(x, W, b) = x @ W.T + b — the weight leaf is
        # (out, in); expose the transpose so mm-epilogue laws see
        # ``matmul(a, w)`` in matmul orientation.
        x, w = args[0], args[1]
        wt = Op.make("transpose", w, dim0=-1, dim1=-2)
        if len(args) >= 3:
            out = Op.make("add", Op.make("matmul", x, wt), args[2])
        else:
            out = Op.make("matmul", x, wt)
    else:
        out = Op(t.op, tuple(args), dict(attrs))
    memo[k] = out
    return out


def _rebuild_ir(ir, root):
    return IR(
        root=root,
        inputs=list(ir.inputs),
        input_names=set(ir.input_names),
        params=dict(ir.params),
    )


# ------------------------------------------------------------------
# subterm modules (whisper-base shapes)
# ------------------------------------------------------------------


class AttnCore(nn.Module):
    """``softmax((q @ k^T) * scale) @ v`` — the attention fragment."""

    def __init__(self, heads=8, dim=64, scale=None):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.scale = scale if scale is not None else dim**-0.5

    def forward(self, q, k, v):
        # q,k,v: (B, T, H*D) — fold heads like the HF eager path does
        b, t, _ = q.shape
        h, d = self.heads, self.dim
        q = q.view(b, t, h, d).transpose(1, 2)
        k = k.view(b, t, h, d).transpose(1, 2)
        v = v.view(b, t, h, d).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        p = torch.softmax(s, dim=-1)
        o = torch.matmul(p, v)
        return o.transpose(1, 2).reshape(b, t, h * d)


class MlpBlock(nn.Module):
    """``x + fc2(gelu(fc1(layer_norm(x))))`` — whisper MLP + residual."""

    def __init__(self, d=512, ffn=2048):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)

    def forward(self, x):
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln(x))))


class ConvBlock(nn.Module):
    """conv1d -> gelu -> conv1d — the whisper encoder frontend pair."""

    def __init__(self, cin=80, mid=512, cout=512):
        super().__init__()
        self.c1 = nn.Conv1d(cin, mid, 3, stride=1, padding=1)
        self.c2 = nn.Conv1d(mid, cout, 3, stride=2, padding=1)

    def forward(self, x):
        return self.c2(
            torch.nn.functional.gelu(self.c1(x))
        )


class OProj(nn.Module):
    """``x + linear(x)`` — residual output projection."""

    def __init__(self, d=512):
        super().__init__()
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        return x + self.proj(x)


class LnRes(nn.Module):
    """``x + layer_norm(x)`` — bare norm + residual fragment."""

    def __init__(self, d=512):
        super().__init__()
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return x + self.ln(x)


def _cases():
    d, ffn, h, hd = 512, 2048, 8, 64
    t = 512  # encoder-ish sequence length
    return {
        "attn_core": (
            AttnCore(h, hd),
            (
                torch.randn(1, t, h * hd),
                torch.randn(1, t, h * hd),
                torch.randn(1, t, h * hd),
            ),
        ),
        "mlp_block": (
            MlpBlock(d, ffn),
            (torch.randn(1, t, d),),
        ),
        "conv_block": (
            ConvBlock(),
            (torch.randn(1, 80, 1000),),
        ),
        "o_proj": (
            OProj(d),
            (torch.randn(1, t, d),),
        ),
        "ln_res": (
            LnRes(d),
            (torch.randn(1, t, d),),
        ),
    }


# ------------------------------------------------------------------
# timing
# ------------------------------------------------------------------


def timeit(fn, warm=5, reps=30):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e6  # µs


# ------------------------------------------------------------------
# kernelopt arm: same spelled IR, KERNEL_RULES + KernelSink
# ------------------------------------------------------------------


def kernelopt_arm(ir, param_values, *, decompose=False, ref=None):
    """Run the kernelopt path over ``ir``; return a report dict.

    ``decompose=True`` first rewrites atomic softmax/layer_norm/linear
    into the decomposed spellings the KERNEL_RULES bank matches —
    the shim a real integration would need.
    """
    out = {"fires": [], "verified": None, "spelled_us": None,
           "sched_us": None, "note": "", "decompose": decompose}
    if decompose:
        ir = _rebuild_ir(ir, _decompose(ir.root))
    sink = KernelSink()
    # spelled reference through the same sink (verify-first)
    try:
        spelled = sink.lower(ir, params=param_values)
    except Exception as e:
        out["note"] = f"spelled lower failed: {type(e).__name__}: {e}"
        return out

    # fire every kernel law once at the root, extract best
    eg = EGraph()
    root = eg.add_term(ir.root)
    fired = 0
    for r in KERNEL_RULES:
        try:
            if eg.apply_rule(r, root):
                fired += 1
                out["fires"].append(r.name)
        except Exception:
            pass
    best = eg.extract_best(root, kernel_cost) or ir.root
    out["changed"] = best is not ir.root and repr(best) != repr(ir.root)
    out["best"] = repr(best)[:400]

    ir2 = IR(
        root=best,
        inputs=list(ir.inputs),
        input_names=set(ir.input_names),
        params=dict(ir.params),
    )
    try:
        sched = sink.lower(ir2, params=param_values)
    except Exception as e:
        out["note"] = f"sched lower failed: {type(e).__name__}: {e}"
        return out

    feeds = tuple(
        torch.randn(tuple(v.typ.shape), dtype=torch.float32,
                    device="cuda")
        for v in ir.inputs
    )
    try:
        rep = sink.verify(spelled, sched, feeds)
        out["verified"] = bool(rep.passed)
        out["max_rel"] = rep.max_rel
    except Exception as e:
        out["note"] = f"verify failed: {type(e).__name__}: {e}"
        return out

    # external check: scheduled output vs the ORIGINAL module —
    # catches decomp-shim bugs the self-consistent spelled/sched
    # compare cannot (both sides share the rewrite).
    if ref is not None:
        try:
            with torch.no_grad():
                ref_out = ref(*feeds)
                got = sched(*feeds)
            d = (ref_out - got).abs().max().item()
            den = ref_out.abs().max().item() + 1e-8
            out["vs_module_rel"] = d / den
            out["vs_module_ok"] = bool(d / den <= 1e-4)
        except Exception as e:
            out["vs_module_rel"] = None
            out["vs_module_ok"] = False
            out["note"] = f"module check failed: {e}"

    out["spelled_us"] = timeit(lambda: spelled(*feeds))
    out["sched_us"] = timeit(lambda: sched(*feeds))
    return out


# ------------------------------------------------------------------


def main() -> None:
    from mlopt_orchestrator.optimize import Optimizer
    from mlopt_torch.backend import TorchBackend

    only = set(sys.argv[1:])
    cases = _cases()
    print(f"[setup] cuda={torch.cuda.is_available()} "
          f"dev={torch.cuda.get_device_name(0)}")

    for name, (mod, args_cpu) in cases.items():
        if only and name not in only:
            continue
        mod = mod.cuda().eval()
        args = tuple(a.cuda() for a in args_cpu)
        call = lambda m, a: m(*a) if isinstance(a, tuple) else m(a)

        eager_us = timeit(lambda: call(mod, args))
        print(f"\n=== {name} ===")
        print(f"  eager            {eager_us:9.1f} us")

        optz = Optimizer(backend=TorchBackend())
        t0 = time.time()
        try:
            res = optz.search(mod, args, max_iterations=20)
        except Exception as e:
            print(f"  graphopt search FAILED: {type(e).__name__}: {e}")
            continue
        s_us = (time.time() - t0)
        ir = res.ir
        n_ops = ir.root
        print(f"  graphopt search  {s_us:6.1f}s  "
              f"root={getattr(n_ops, 'op', type(n_ops).__name__)}")

        # graphopt delivered arm
        g_us = None
        try:
            low = optz.lower(res, args, verify=True, runner=None)
            g_us = timeit(lambda: low.module(*args))
            vok = low.verified.passed if low.verified else None
            print(f"  graphopt deliver {g_us:9.1f} us  verified={vok}")
        except Exception as e:
            print(f"  graphopt lower FAILED: {type(e).__name__}: {e}")

        # kernelopt arms — same exported IR, as-exported and
        # through the decomposition shim
        for decomp in (False, True):
            tag = "kernelopt+decomp" if decomp else "kernelopt      "
            k = kernelopt_arm(ir, res.param_values, decompose=decomp,
                              ref=mod)
            if k["note"]:
                print(f"  {tag} INFEASIBLE: {k['note']}")
                continue
            print(
                f"  {tag} spelled {k['spelled_us']:9.1f} us  "
                f"sched {k['sched_us']:9.1f} us  "
                f"verified={k['verified']} "
                f"max_rel={k.get('max_rel', 0):.2e} "
                f"vs_module={k.get('vs_module_rel')}"
            )
            print(f"    laws fired: {k['fires']}")
            print(f"    best: {k.get('best', '')[:200]}")
            if g_us:
                print(f"    [vs graphopt {g_us:.1f} us -> "
                      f"{k['sched_us'] / g_us:.2f}x]")
        if g_us:
            print(f"  [cmp] eager/graphopt: {eager_us:.1f} / {g_us:.1f} us")


if __name__ == "__main__":
    main()

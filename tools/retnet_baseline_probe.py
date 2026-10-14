"""RetNet carrier vs *external* baselines — real-baseline benchmark.

The headline "7.2x less memory" was measured against our own spelled
quadratic reference.  This probe re-measures the delivered carrier
against baselines that exist outside this repository:

arms (B=2, d=16, H=1, T in {512, 1024, 2048}):

- ``spelled-eager`` / ``spelled-compiled`` — the optimize-input model
  (``zoo._ChunkedRetention``), eager and under ``torch.compile``.
- ``carrier`` — the extracted K-way carrier, lowered via the ambient
  ``TorchSink`` (the same artifact as ``om_decayed_kway_deliver_probe``).
- ``paper-eqn`` — a faithful in-script reimplementation of the paper's
  chunkwise-recurrent algorithm (Sun et al. 2023): intra-chunk masked
  QK^T V + cross-chunk decayed state recurrence, no stability rescale.
  fp64 — the *numerical agreement* oracle vs our carrier.
- ``paper-code`` — the OFFICIAL torchscale code path
  (``RetNetRelPos`` inner_mask + ``MultiScaleRetention.
  chunk_recurrent_forward`` driven directly, rotary/gating/norm
  bypassed, decay buffer patched to our gamma=0.9).  fp64.  Note: the
  official code carries the fp16 ``inner_scale``/``all_scale``
  stability rescaling, so its outputs are close to — not equal to —
  raw retention; the diff is reported, not hidden.
- ``fla-chunk`` / ``fla-rec`` — fla-org Triton kernels
  (``chunk_retention``, ``fused_recurrent_retention``), fp32 (Triton
  has no fp64).  FLA bakes per-head decay ``1-2^-5`` and ``1/sqrt(d)``
  scaling, so numerics intentionally differ — these are speed/memory
  baselines only.  The carrier is therefore also timed in fp32
  (``carrier32``) for an apples-to-apples cell.

Usage::

    .venv/bin/python tools/retnet_baseline_probe.py
    .venv/bin/python tools/retnet_baseline_probe.py --t 2048
"""

from __future__ import annotations

import argparse
import copy
import math
import types

import om_decayed_kway_deliver_probe as P
import torch
from catopt_core.ir import IR
from catopt_discovery.play import _time_module
from catopt_discovery.zoo import _ChunkedRetention
from catopt_torch.adapters import TorchSink

GAMMA = 0.9


def _peak(fn) -> float:
    """Peak CUDA allocation (MiB) of one forward under no_grad."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        fn()
    return torch.cuda.max_memory_allocated() / 2**20


def _paper_eqn(x: torch.Tensor, model: _ChunkedRetention, c: int):
    """Paper chunkwise recurrence, exact (no stability rescale), fp64."""
    b, t, d = x.shape
    nc = t // c
    q = model.q(x) / math.sqrt(t)
    k = model.k(x)
    v = model.v(x)
    pos = torch.arange(c, dtype=x.dtype, device=x.device)
    decay = torch.tril(
        torch.tensor(GAMMA, dtype=x.dtype, device=x.device)
        ** (pos[:, None] - pos[None, :]).clamp_min(0)
    )
    cross_decay = torch.tensor(GAMMA**c, dtype=x.dtype, device=x.device)
    q_dec = torch.tensor(
        [GAMMA ** (i + 1) for i in range(c)],
        dtype=x.dtype,
        device=x.device,
    )
    v_dec = torch.tensor(
        [GAMMA ** (c - 1 - i) for i in range(c)],
        dtype=x.dtype,
        device=x.device,
    )
    qc = q.view(b, nc, c, d)
    kc = k.view(b, nc, c, d)
    vc = v.view(b, nc, c, d)
    inner = ((qc @ kc.transpose(-1, -2)) * decay) @ vc
    kv_chunk = kc.transpose(-1, -2) @ (vc * v_dec[None, None, :, None])
    state = torch.zeros(b, d, d, dtype=x.dtype, device=x.device)
    cross = torch.zeros_like(inner)
    for i in range(nc):
        cross[:, i] = (qc[:, i] * q_dec[None, :, None]) @ state
        state = state * cross_decay + kv_chunk[:, i]
    return (inner + cross).reshape(b, t, d)


def _torchscale_arm(x: torch.Tensor, model: _ChunkedRetention, c: int):
    """Official torchscale chunk_recurrent_forward, decay patched.

    ``inner_mask`` is built verbatim from ``RetNetRelPos.forward``'s
    chunkwise branch (``torchscale/architecture/retnet.py``) — inlined
    because importing that module pulls ``timm``.  The mask math is
    the official code's; decay patched to our gamma.
    """
    from torchscale.component.multiscale_retention import (
        MultiScaleRetention,
    )

    b, t, d = x.shape
    dev, dt = x.device, x.dtype
    decay_vec = torch.tensor([math.log(GAMMA)], dtype=dt, device=dev)

    block_index = torch.arange(c, device=dev).to(dt)
    mask = torch.tril(torch.ones(c, c, dtype=dt, device=dev))
    mask = torch.masked_fill(
        block_index[:, None] - block_index[None, :],
        ~mask.bool(),
        float("inf"),
    )
    mask = torch.exp(mask * decay_vec[:, None, None])
    mask = torch.nan_to_num(mask)
    value_inner_decay = mask[:, -1] / mask[:, -1].sum(
        dim=-1, keepdim=True
    )
    value_inner_decay = value_inner_decay.unsqueeze(-1)
    scale = mask.sum(dim=-1, keepdim=True).sqrt()
    inner_mask_ = mask / scale
    cross_decay = torch.exp(decay_vec * c)[:, None, None]
    query_inner_decay = torch.exp(
        decay_vec[:, None] * (block_index + 1)
    )
    query_inner_decay = query_inner_decay[:, :, None] / (
        scale / mask[:, -1].sum(dim=-1)[:, None, None]
    )
    inner_mask = (
        inner_mask_,
        cross_decay,
        query_inner_decay,
        value_inner_decay,
    )

    fake = types.SimpleNamespace(num_heads=1, key_dim=d, head_dim=d)
    q = model.q(x).view(b, t, 1, d) / math.sqrt(t)
    k = model.k(x).view(b, t, 1, d)
    v = model.v(x)
    return MultiScaleRetention.chunk_recurrent_forward(
        fake, q, k, v, inner_mask
    ).reshape(b, t, d)


def _fla_inputs(x: torch.Tensor, model: _ChunkedRetention):
    """Project q/k/v and reshape to FLA's [B, T, H, K] fp32."""
    q = model.q(x).unsqueeze(2).float().contiguous()
    k = model.k(x).unsqueeze(2).float().contiguous()
    v = model.v(x).unsqueeze(2).float().contiguous()
    return q, k, v


def _run_t(
    t: int, k: int, iters: int, budget: int, chunk: int, reps: int
):
    sat = P._saturate(t, k, iters, budget, chunk)
    sink = TorchSink()
    ir = sat["ir"]
    ex = IR(
        root=sat["best"],
        inputs=list(ir.inputs),
        input_names=set(ir.input_names),
        params=dict(ir.params),
    )
    base = sat["model"].to("cuda").double()
    carrier = sink.lower(ex, params=sat["leaves"]).to("cuda").double()
    comp = torch.compile(sat["model"].to("cuda").double())
    x64 = sat["x"].to("cuda")
    c = t // k

    arms = {}
    arms["spelled-eager"] = lambda: base(x64)
    arms["spelled-compiled"] = lambda: comp(x64)
    arms["carrier"] = lambda: carrier(x64)
    arms["paper-eqn"] = lambda: _paper_eqn(x64, base, c)
    arms["paper-code"] = lambda: _torchscale_arm(x64, base, c)

    # fp32 arms vs FLA Triton kernels — fresh fp32 leaf copies:
    # carrier params are non-leaf tensors (deepcopy-refused), and
    # nn.Module.to() is in-place so cloning must precede conversion.
    leaves32 = {n: t.detach().float() for n, t in sat["leaves"].items()}
    carrier32 = sink.lower(ex, params=leaves32).to("cuda")
    x32 = x64.float()
    base32 = copy.deepcopy(base).float()
    arms["carrier32"] = lambda: carrier32(x32)
    from fla.ops.retention import (
        chunk_retention,
        fused_recurrent_retention,
    )

    arms["fla-chunk"] = lambda: chunk_retention(
        *_fla_inputs(x32, base32)
    )[0]
    arms["fla-rec"] = lambda: fused_recurrent_retention(
        *_fla_inputs(x32, base32)
    )[0]

    print(
        f"\n=== T={t} K={k} C={c} === "
        f"(extracted cost {sat['cost']:,.0f}, root={sat['root_op']})"
    )
    print(f"{'arm':>17} {'peak MiB':>9} {'med ms':>8} {'vs model':>9}")
    ref = sink.verify(base, carrier, (x64,))
    rows = {}
    for name, fn in arms.items():
        try:
            with torch.no_grad():
                fn()  # JIT warm (torch.compile / triton) before peak
            mem = _peak(fn)
            ms = (
                _time_module(lambda *_, f=fn: f(), (x64,), reps=reps)
                / 1e3
            )
        except Exception as e:  # record, don't hide
            rows[name] = (float("nan"), float("nan"))
            print(f"{name:>17}  FAILED: {type(e).__name__}: {e}")
            continue
        rows[name] = (mem, ms)
        print(
            f"{name:>17} {mem:>9.2f} {ms:>8.3f} {rows['spelled-eager'][0] / mem:>8.2f}x"
        )

    # numeric agreement, fp64
    with torch.no_grad():
        ours = carrier(x64)
        pe = _paper_eqn(x64, base, c)
        ts = _torchscale_arm(x64, base, c)
    print(f"\ncarrier vs spelled-eager : {ref.max_abs:.2e}")
    print(
        f"carrier vs paper-eqn     : {(ours - pe).abs().max().item():.2e}"
    )
    print(
        f"carrier vs paper-code    : {(ours - ts).abs().max().item():.2e}"
    )
    del sat
    torch.cuda.empty_cache()


def main() -> None:
    """Run the baseline sweep; one table per requested T."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--t", type=int, nargs="*", default=[512, 1024, 2048]
    )
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--iters", type=int, default=14)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    for t in a.t:
        _run_t(t, a.k, a.iters, a.budget, a.chunk, a.reps)


if __name__ == "__main__":
    main()

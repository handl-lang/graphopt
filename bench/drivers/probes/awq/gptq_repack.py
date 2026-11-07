"""M3 layout arm: does the AWQ int4 GEMV body cover GPTQ checkpoints?

GPTQ (Qwen2.5-0.5B-Instruct-GPTQ-Int4, downloaded to ``--ckpt``)
stores ``qweight`` transposed: ``[k/8, n]`` int32 where nibble *p*
of word ``[i, j]`` is input row ``8i + p`` of output column *j* —
AWQ stores ``[k, n/8]`` with nibble *m* packing output columns.
``qzeros``/``scales`` keep the AWQ ``[k/128, n/8]`` / ``[k/128, n]``
shapes, and this checkpoint is symmetric (every zero nibble is 7),
so the zero-point packing order is unobservable here.

The question this probe answers: **same dequant body, different host
packing?** If a load-time repack produces the standard-order operands
the shipped kernels already read, no new ``Term`` op is needed — the
repack is O(weights) once per model load, like ``to_standard_order``.

    PYTHONPATH=$PWD:<kernelopt>/src:<lib-kernel-ir>/src \
      python -m bench.drivers.probes.awq.gptq_repack
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from lib_kernel_ir import qgemv
from safetensors import safe_open

from .irtriton import cached_module
from .kernel_v2 import timeit_graph

GPTQ_CKPT = "/tmp/gptq-ckpt/model.safetensors"
REPORT = Path(
    "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
    "p27_gptq_repack.json"
)

#: fallback launch configs per k if the manifest can't be imported;
#: the shipped rows are the authority when present
SHIPPED = {4864: (8, 64, 64, 2), 896: (8, 64, 64, 2)}


def _route_for(lib: Path, operator: str):
    """The generated manifest row for *operator*, or None."""
    src = lib / "src" / "lib_triton_qgemv"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from lib_triton_qgemv.manifest import ROUTES
    except Exception:
        return None
    return ROUTES.get(operator)


def gptq_to_standard(qw: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Repack GPTQ ``qweight [k/8, n]`` int32 -> standard ``[k, n/8]``.

    Nibble ``p`` of ``qw[i, j]`` is input row ``8i + p``; the result
    packs 8 *output* columns per int32 in plain order — exactly what
    ``unpack_i4`` + the P25 body read for AWQ after
    ``to_standard_order``.  Pure load-time transform.
    """
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qw.device)
    nib = (qw.unsqueeze(-1) >> shifts) & 0xF  # [k/8, n, 8]
    wq = nib.permute(0, 2, 1).reshape(k, n)  # [k, n] values 0..15
    packed = wq.view(k, n // 8, 8).to(torch.int32)
    return (
        (packed << shifts).sum(dim=-1, dtype=torch.int32).contiguous()
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=GPTQ_CKPT)
    ap.add_argument(
        "--ops",
        default="gate_proj,down_proj",
        help="comma-separated operator names (layer 0)",
    )
    ap.add_argument(
        "--lib",
        type=Path,
        default=Path("/home/luis/GitHub/handl-lang/lib-triton-qgemv"),
    )
    args = ap.parse_args()

    import vllm._custom_ops as cops

    rows = []
    with safe_open(args.ckpt, "pt") as f:
        # safe_open is not iterable; snapshot the key list once —
        # every operator lookup below scans it
        all_keys = list(f.keys())
        for name in args.ops.split(","):
            # find the layer-0 prefix for this operator
            prefix = None
            for key in all_keys:
                if key.startswith("model.layers.0.") and key.endswith(
                    f"{name}.qweight"
                ):
                    prefix = key[: -len(".qweight")]
                    break
            if prefix is None:
                print(f"{name}: not found")
                continue
            qw = f.get_tensor(f"{prefix}.qweight").cuda()
            qz = f.get_tensor(f"{prefix}.qzeros").cuda()
            sc = f.get_tensor(f"{prefix}.scales").cuda()
            # GPTQ v2 layout: qweight [k/8, n]; n from scales, k = rows*8
            k, n = qw.shape[0] * 8, sc.shape[1]
            x = torch.randn(1, k, device="cuda", dtype=torch.float16)

            t0 = time.perf_counter()
            qw_std = gptq_to_standard(qw, k, n)
            torch.cuda.synchronize()
            repack_ms = (time.perf_counter() - t0) * 1e3

            incumbent = lambda x=x, qw=qw, qz=qz, sc=sc: (  # noqa: E731
                cops.gptq_gemm(x, qw, qz, sc, False, True, 4)
            )
            expected = incumbent()

            route = _route_for(args.lib, name)
            if route is not None:
                split, bn, bk, warps = (
                    route.split,
                    route.block_n,
                    route.block_k,
                    route.warps,
                )
            else:
                split, bn, bk, warps = SHIPPED[k]
            mod = cached_module(qgemv(k=k, split=split))
            xf = x.reshape(-1)

            def ours(
                xf=xf,
                qw_std=qw_std,
                qz=qz,
                sc=sc,
                n=n,
                k=k,
                bn=bn,
                bk=bk,
                warps=warps,
                mod=mod,
            ):
                return mod.run(xf, qw_std, qz, sc, n, k, bn, bk, warps)

            out = ours()
            cos = torch.nn.functional.cosine_similarity(
                out.detach().flatten().float(),
                expected.detach().flatten().float(),
                dim=0,
            ).item()
            us_ours = timeit_graph(ours, calls=10, replays=3)
            us_inc = timeit_graph(incumbent, calls=10, replays=3)
            row = {
                "operator": name,
                "k": k,
                "n": n,
                "repack_ms": round(repack_ms, 2),
                "cosine_vs_gptq_gemm": round(cos, 7),
                "ours_us": round(us_ours, 1) if us_ours else None,
                "gptq_gemm_us": round(us_inc, 1) if us_inc else None,
                "speedup": round(us_inc / us_ours, 2)
                if us_ours and us_inc
                else None,
            }
            rows.append(row)
            print(json.dumps(row))

    REPORT.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {REPORT}")
    ok = all(r["cosine_vs_gptq_gemm"] >= 0.9999 for r in rows)
    print(f"all gated: {ok}")
    return 0 if ok and rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

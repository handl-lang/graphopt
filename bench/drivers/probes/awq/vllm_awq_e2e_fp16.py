"""Phase 2 — end-to-end with the fp16 kernel (no dtype copies).

Phase 1's v2 kernel takes fp16 and returns fp16, so the per-call
`x.half -> float` / `out.float -> half` conversions that made the first
end-to-end test flat are gone. What is left is pure dilution: how much of
a token is the linears we route to ourselves.

Routing (measured, `p8c_splitk_routing_e2e.md`):

    down_proj   4864->896   ours
    q/o_proj    896->896    ours
    gate/up     896->4864   vLLM awq_gemm (ours 0.94x)
    k/v_proj    896->128    plain matmul (fastest at this size)
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time

import torch

from .kernel_split2 import gemv_split2

USE_OURS = {"on": False}
COUNT = {"calls": 0, "routed": 0, "no_cfg": 0, "not_one_token": 0}
#: (in, out) -> (bn, bk, split, warps); from the cold search (p14).
#: gate_proj is deliberately absent: it loses to awq_gemm 0.95x.
ROUTE: dict[tuple[int, int], tuple[int, int, int, int]] = {
    (4864, 896): (64, 64, 8, 2),
    (896, 896): (32, 128, 4, 2),
    (896, 128): (32, 64, 4, 2),
}


def _make_patched(orig):
    def apply(self, layer, x, bias=None):  # noqa: ANN001
        COUNT["calls"] += 1
        qw = getattr(layer, "qweight", None)
        if COUNT["calls"] == 1:
            print(
                "  layer attrs:",
                [
                    a
                    for a in dir(layer)
                    if "weight" in a or "scale" in a or "zero" in a
                ],
                "| shapes:",
                tuple(qw.shape) if qw is not None else None,
                tuple(layer.scales.shape),
                flush=True,
            )
        cfg = ROUTE.get(
            (qw.shape[0], layer.scales.shape[1])
            if qw is not None
            else None
        )
        if cfg is None:
            COUNT["no_cfg"] += 1
        elif USE_OURS["on"] and x.shape[0] != 1:
            COUNT["not_one_token"] += 1
        if not USE_OURS["on"] or cfg is None or x.shape[0] != 1:
            return orig(self, layer, x, bias)
        COUNT["routed"] += 1
        sc = layer.scales
        n, k = sc.shape[1], qw.shape[0]
        bn, bk, split, wp = cfg
        out = gemv_split2(
            x.reshape(-1).to(torch.float16).contiguous(),
            qw,
            layer.qzeros,
            sc,
            n,
            k,
            bn,
            bk,
            split,
            wp,
        )
        if bias is not None:
            out = out + bias.to(torch.float16)
        return out

    return apply


def install_patch() -> None:
    """Patch every AWQ linear method class, not just one.

    vLLM selects the implementation at load time (plain AWQ vs Marlin),
    so patching only `AutoAWQLinearMethod` silently does nothing — the
    first six end-to-end runs measured vLLM against itself. Every class
    that can serve an AWQ linear is wrapped, and `COUNT` proves which one
    ran.
    """
    from vllm.model_executor.layers.quantization import auto_awq

    targets = [
        name
        for name in (
            "BaseAWQLinearMethod",
            "AutoAWQLinearMethod",
            "AutoAWQMarlinLinearMethod",
        )
        if hasattr(auto_awq, name)
    ]
    for name in targets:
        cls = getattr(auto_awq, name)
        cls.apply = _make_patched(cls.apply)
    COUNT["classes"] = targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct-AWQ")
    ap.add_argument(
        "--quant",
        default="awq",
        help="force the AWQ implementation: 'awq' (plain, matches the "
        "checkpoint layout, our kernel's target) or 'auto' (vLLM picks "
        "Marlin, which repacks the weights)",
    )
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    install_patch()
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=args.frac,
        max_model_len=args.ctx,
        seed=0,
        disable_log_stats=True,
        quantization=None if args.quant == "auto" else args.quant,
    )
    torch.manual_seed(0)
    ids = torch.randint(1, 151936, (1, args.ctx // 2)).tolist()
    sp = SamplingParams(
        temperature=0.0, max_tokens=args.gen, ignore_eos=True
    )

    def call(on: bool):
        USE_OURS["on"] = on
        t0 = time.perf_counter()
        out = llm.generate(ids, sampling_params=sp, use_tqdm=False)
        return time.perf_counter() - t0, [
            o.outputs[0].token_ids for o in out
        ]

    for _ in range(args.warm):
        call(False)
        call(True)

    base, ours, toks = [], [], {}
    for i in range(args.rounds):
        for on in (False, True) if i % 2 == 0 else (True, False):
            dt, tk = call(on)
            (ours if on else base).append(dt)
            toks.setdefault("ours" if on else "base", tk)
    USE_OURS["on"] = False

    mb, mo = stats.median(base), stats.median(ours)
    wins = sum(1 for a, b in zip(ours, base, strict=True) if a < b)
    same = toks["ours"] == toks["base"]
    print(f"\n{args.rounds} paired rounds")
    print(
        f"  stock   : {mb * 1e3:7.1f} ms  sd {stats.stdev(base) * 1e3:5.1f}"
        f"  {args.gen / mb:6.1f} tok/s"
    )
    print(
        f"  fp16 ours: {mo * 1e3:7.1f} ms  sd {stats.stdev(ours) * 1e3:5.1f}"
        f"  {args.gen / mo:6.1f} tok/s"
    )
    print(
        f"  -> {100 * (mo - mb) / mb:+.1f}% · wins {wins}/{args.rounds}"
        f" · identical tokens: {same}"
    )
    print(f"  patch fired: {COUNT}")

    if args.out:
        open(args.out, "w").write(
            json.dumps(
                {
                    "rounds": args.rounds,
                    "stock_ms": round(mb * 1e3, 2),
                    "ours_ms": round(mo * 1e3, 2),
                    "delta_pct": round(100 * (mo - mb) / mb, 2),
                    "ours_wins": wins,
                    "identical_tokens": same,
                },
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

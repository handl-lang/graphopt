"""Phase 2, done right — stash the pre-repack weights, then route.

Marlin replaces a layer's ``qweight/qzeros/scales`` with repacked tensors,
which is why every earlier injection found nothing to hook. The fix:
capture the checkpoint-layout tensors *during*
``process_weights_after_loading`` (before Marlin overwrites them), then
route ``apply`` through our kernel when the shapes match the routing
table.

The call counter proves the injection fired before any number is read.
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
STASH: dict[int, tuple] = {}

#: (in, out) -> (bn, bk, split, warps) from the cold search (p14)
ROUTE: dict[tuple[int, int], tuple[int, int, int, int]] = {
    (4864, 896): (64, 64, 8, 2),
    (896, 896): (32, 128, 4, 2),
    (896, 128): (32, 64, 4, 2),
}


def _wrap_process(cls):
    orig = cls.process_weights_after_loading

    def process(self, layer):  # noqa: ANN001
        qw = getattr(layer, "qweight", None)
        qz = getattr(layer, "qzeros", None)
        sc = getattr(layer, "scales", None)
        if qw is not None and qz is not None and sc is not None:
            if qw.dim() == 2 and sc.dim() == 2:
                STASH[id(layer)] = (
                    qw.detach().clone(),
                    qz.detach().clone(),
                    sc.detach().clone(),
                )
        return orig(self, layer)

    cls.process_weights_after_loading = process


def _wrap_apply(cls):
    orig = cls.apply

    def apply(self, layer, x, bias=None):  # noqa: ANN001
        COUNT["calls"] += 1
        stashed = STASH.get(id(layer))
        if stashed is None:
            COUNT["no_cfg"] += 1
            return orig(self, layer, x, bias)
        qw, qz, sc = stashed
        k = qw.shape[0]
        n = sc.shape[1]
        cfg = ROUTE.get((k, n))
        if cfg is None:
            COUNT["no_cfg"] += 1
            return orig(self, layer, x, bias)
        if x.shape[0] != 1:
            COUNT["not_one_token"] += 1
            return orig(self, layer, x, bias)
        COUNT["routed"] += 1
        bn, bk, split, wp = cfg
        out = gemv_split2(
            x.reshape(-1).to(torch.float16).contiguous(),
            qw,
            qz,
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

    cls.apply = apply


def install_patch() -> None:
    from vllm.model_executor.layers.quantization import auto_awq

    names = [
        n
        for n in ("AutoAWQMarlinLinearMethod", "AutoAWQLinearMethod")
        if hasattr(auto_awq, n)
    ]
    for name in names:
        cls = getattr(auto_awq, name)
        _wrap_process(cls)
        _wrap_apply(cls)
    COUNT["classes"] = names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct-AWQ")
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
    )
    print(
        f"  stashed layers: {len(STASH)} · classes {COUNT['classes']}"
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
        f"  ours    : {mo * 1e3:7.1f} ms  sd {stats.stdev(ours) * 1e3:5.1f}"
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
                    "patch_fired": COUNT,
                    "stashed": len(STASH),
                },
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

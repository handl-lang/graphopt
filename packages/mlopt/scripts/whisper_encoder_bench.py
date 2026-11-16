"""Whisper ENCODER delivery bench — eager vs torch.compile vs mlopt delivery.

Measurement driver (not shipped code).  Companion to
``whisper_batch_bench.py`` (which benches the decode-step unit the
adapter lifted first): this one benches the ``arm="encoder"`` unit —
the whole ``WhisperEncoder`` forward over real log-mel features, the
unit the encoder-side wins in ``graphopt``'s
``bench/drivers/probes/whisper_encoder_bench.py`` /
``docs/reports/turbo_bf16_delivery.md`` came from.

For each batch size B in ``--batches`` it reports:

* **HF eager** — ``spec.module`` itself (the ``EncWrap`` over the HF
  eager encoder; the adapter keeps it callable post-load).
* **torch.compile** — ``torch.compile(spec.module, dynamic=False)`` —
  the incumbent.
* **delivered** — ``mlopt.optimize_model(..., deliver=True)``: the
  search + ``lower(verify=True)`` pipeline plus the delivery game
  (``--deliver-budget`` moves) or heuristic (``--deliver-heuristic``).
  Eager-vs-delivered runner timings come from the ``OptimizeReport``;
  this script adds the HF-eager and compile arms plus an explicit
  delivered-vs-HF ``max|diff|`` faithfulness check (the verify gate
  only compares delivered vs re-lowered IR — same caveat as the
  graphopt encoder bench).

Honest declines are printed, not patched: a compile that OOMs, a
search/lower failure, a failed verify, or a delivery config that
fails its gate is recorded as such.

Usage::

    .venv/bin/python scripts/whisper_encoder_bench.py \
        --model openai/whisper-small --batches 1 4 --iters 3
    # turbo bf16 needs the sdpa-safe ruleset (see adapter notes):
    .venv/bin/python scripts/whisper_encoder_bench.py \
        --model openai/whisper-large-v3-turbo --batches 1 \
        --rules bf16_safe --iters 3 --deliver-budget 6
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import Any

import torch

from mlopt.adapters.whisper import ADAPTER
from mlopt.models import ModelSpec
from mlopt.optimize import optimize_model


def _time(fn: Any, *args: Any, reps: int = 10, warm: int = 2) -> list[float]:
    """Warm ``warm`` calls then ``reps`` synchronized timings (ms)."""
    for _ in range(warm):
        fn(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def _ms(ts: list[float]) -> str:
    return " ".join(f"{v:.1f}" for v in ts)


def _summ(ts: list[float]) -> str:
    s = sorted(ts)
    return f"min={s[0]:.1f} med={s[len(s) // 2]:.1f}"


def _batch_feats(spec: Any, B: int) -> Any:
    """Repeat the adapter's (1,n_mels,3000) features to batch B."""
    feats = spec.inputs[0]
    if B == 1:
        return feats
    return feats.repeat(B, 1, 1)


def _max_abs_diff(a: Any, b: Any) -> float:
    with torch.no_grad():
        return (a.float() - b.float()).abs().max().item()


def run(
    model: str,
    batches: list[int],
    iters: int,
    bench_iters: int,
    deliver_budget: int,
    deliver_heuristic: bool,
    rules: str | None,
    adapter_opts: dict[str, Any],
) -> None:
    print(f"[enc] loading {model} arm=encoder opts={adapter_opts}", flush=True)
    spec = ADAPTER.load(model, arm="encoder", **adapter_opts)
    meta = spec.meta
    print(
        f"[enc] size={meta.get('size')} dtype={meta.get('dtype')} "
        f"params={meta.get('encoder_params', 0) / 1e6:.1f}M "
        f"sanity={meta.get('sanity')}",
        flush=True,
    )

    eager_mod = spec.module  # the HF eager encoder wrap — stays callable
    for B in batches:
        feats = _batch_feats(spec, B)
        spec_b = ModelSpec(module=eager_mod, inputs=(feats,), kwargs={}, meta=meta)
        print(f"[enc] --- batch={B} feats={tuple(feats.shape)} ---", flush=True)

        # HF-eager reference + timing
        with torch.no_grad():
            ref = eager_mod(feats)
        t_hf = _time(eager_mod, feats)
        print(f"[enc:B{B}] hf-eager  ms: {_ms(t_hf)}  ({_summ(t_hf)})", flush=True)

        # torch.compile incumbent
        try:
            cmod = torch.compile(eager_mod, dynamic=False)
            with torch.no_grad():
                cmod(feats)
            t_c = _time(cmod, feats)
            d = _max_abs_diff(cmod(feats), ref)
            print(
                f"[enc:B{B}] compile   ms: {_ms(t_c)}  ({_summ(t_c)})  vs-HF max|diff|={d:.3e}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 — honest decline
            print(f"[enc:B{B}] compile DECLINED {type(e).__name__}: {e}", flush=True)
        cmod = None
        gc.collect()
        torch.cuda.empty_cache()

        # mlopt pipeline + delivery
        lowered, report = optimize_model(
            spec_b,
            max_iterations=iters,
            verify=True,
            deliver=deliver_budget > 0 or deliver_heuristic,
            bench_iters=bench_iters,
            deliver_budget=deliver_budget or None,
            deliver_heuristic=deliver_heuristic,
            rules=rules,
        )
        print(report.render(), flush=True)
        if lowered is None:
            print(f"[enc:B{B}] optimize DECLINED — see report.error", flush=True)
            continue
        # Faithfulness: delivered module vs the HF eager snapshot.
        try:
            dmod = lowered.module
            with torch.no_grad():
                out = dmod(feats)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            d = _max_abs_diff(out, ref)
            rel = d / ref.float().abs().max().item()
            print(
                f"[enc:B{B}] delivered-vs-HF max|diff|={d:.3e} rel={rel:.3e}",
                flush=True,
            )
            # Delivered-module timing (the verify-gated artifact itself).
            t_d = _time(dmod, feats)
            print(f"[enc:B{B}] delivered ms: {_ms(t_d)}  ({_summ(t_d)})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[enc:B{B}] delivered check FAILED {type(e).__name__}: {e}", flush=True)
        del lowered, ref
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"[enc:B{B}] peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--batches", type=int, nargs="+", default=[1])
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--bench-iters", type=int, default=20)
    p.add_argument("--deliver-budget", type=int, default=8)
    p.add_argument("--deliver-heuristic", action="store_true")
    p.add_argument("--rules", default=None, help="rule preset (e.g. bf16_safe)")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--audio-s", type=float, default=None)
    args = p.parse_args()

    adapter_opts: dict[str, Any] = {}
    for k, v in (("device", args.device), ("dtype", args.dtype), ("audio_s", args.audio_s)):
        if v is not None:
            adapter_opts[k] = v

    if not torch.cuda.is_available():
        print("[enc] CUDA unavailable — this bench is a GPU measurement", flush=True)
    run(
        args.model,
        args.batches,
        args.iters,
        args.bench_iters,
        args.deliver_budget,
        args.deliver_heuristic,
        args.rules,
        adapter_opts,
    )


if __name__ == "__main__":
    main()

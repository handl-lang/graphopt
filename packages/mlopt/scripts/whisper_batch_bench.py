"""Batched-serving bench — Whisper encoder-decoder forward vs mlopt delivery.

Measurement driver (not shipped code).  For each batch size B in
``--batches`` it reports two tables:

* **full forward** — ``WhisperForConditionalGeneration(input_features,
  decoder_input_ids)`` eager vs ``torch.compile`` (the incumbent).
* **decode step** — the unit ``mlopt.adapters.whisper`` actually lifts:
  one static-shape decoder step over caller-owned K/V buffers.  The
  wrap class is batch-agnostic (``(B,nh,L,hdim)`` buffers broadcast the
  same way ``(1,nh,L,hdim)`` did), so the same harvested-step machinery
  runs at batch B: eager wrap vs ``torch.compile(wrap)`` vs the
  ``mlopt optimize --deliver`` runner selected by the delivery game.

The bench loads the model with ``attn_implementation="sdpa"`` by
default (``--attn eager`` for the adapter's exact config): batched
eager attention materialises ``B x heads x 1500 x 1500`` score tensors
which OOM a 4 GB card at batch >= 4, while sdpa is the realistic
serving configuration anyway.  The decode-step wrap, K/V harvest and
mlopt pipeline are attention-implementation agnostic.

Honest declines are printed, not patched: a compile that OOMs or a
delivery config that fails verify is recorded as such.

Usage::

    .venv/bin/python scripts/whisper_batch_bench.py \
        --model openai/whisper-large-v3-turbo --batches 1 4 8
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import torch
import torch.nn as nn

from mlopt.adapters.whisper import (
    _classes,
    _encoder_out,
    _get_prefix,
    _make_features,
    apply_suppress,
    embed_step,
    suppress_sets,
)
from mlopt.deliver import _bench
from mlopt.models import ModelSpec
from mlopt.optimize import optimize_model


def _pad_self_b(t: Any, L: int, P: int) -> Any:
    """(B,nh,P,hdim) -> (B,nh,L,hdim) zero-padded on the right."""
    B, nh, _, hdim = t.shape
    out = torch.zeros(B, nh, L, hdim, device=t.device, dtype=t.dtype)
    out[:, :, :P] = t
    return out


def _build_inputs_b(harvest: dict[str, Any], L: int) -> tuple[Any, ...]:
    """Batched ``_build_static_inputs``: (hs, mask, pos, sk[n], sv[n],
    ck[n], cv[n]) — mask/pos broadcast over the batch dim."""
    P = harvest["P"]
    dtype = harvest["hs"].dtype
    sk = [_pad_self_b(t, L, P) for t in harvest["past"]["sk"]]
    sv = [_pad_self_b(t, L, P) for t in harvest["past"]["sv"]]
    ck = list(harvest["past"]["ck"])
    cv = list(harvest["past"]["cv"])
    mask = torch.zeros(1, 1, 1, L, device=harvest["hs"].device, dtype=dtype)
    mask[..., P + 1 : L] = torch.finfo(dtype).min
    pos = torch.tensor([P], device=harvest["hs"].device)
    return (harvest["hs"], mask, pos, *(sk + sv + ck + cv))


def _harvest_batched(
    model: Any,
    ehs: Any,
    prefix: list[int],
    ntok: int,
    device: Any,
    B: int,
    harvest_call: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Batched port of the adapter's ``_greedy_loop_harvest``.

    Same HF suppression policy; ``input_ids`` are the shared prefix
    expanded to batch B (identical prompts — the bench measures shape,
    not transcript diversity).
    """
    dec = model.model.decoder
    supp, begin, eos = suppress_sets(model)
    K = len(prefix)
    cache = None
    harvest: dict[str, Any] = {}
    call_idx = 0
    gen_ids: list[list[int]] = []

    def snap(c: Any, P: int, cur: Any, logits: Any, nxt: Any) -> None:
        n_layers = len(c.self_attention_cache.layers)
        sc = c.self_attention_cache
        cc = c.cross_attention_cache
        harvest["past"] = {
            "sk": [sc.layers[j].keys[..., :P, :].detach().clone() for j in range(n_layers)],
            "sv": [sc.layers[j].values[..., :P, :].detach().clone() for j in range(n_layers)],
            "ck": [cc.layers[j].keys.detach().clone() for j in range(n_layers)],
            "cv": [cc.layers[j].values.detach().clone() for j in range(n_layers)],
        }
        harvest["P"] = P
        harvest["ehs"] = ehs
        harvest["hs"] = embed_step(dec, cur, P)
        harvest["logits"] = logits.float().clone()
        harvest["next_id"] = nxt.clone()

    with torch.no_grad():
        # untimed prefill: prefix[:-1], replicated across the batch
        for j in range(K - 1):
            ids = torch.tensor(prefix[j : j + 1], device=device).view(1, 1).expand(B, 1)
            out = dec(
                input_ids=ids,
                encoder_hidden_states=ehs,
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            call_idx += 1
        cur = torch.tensor(prefix[K - 1 : K], device=device).view(1, 1).expand(B, 1).contiguous()
        for i in range(ntok):
            out = dec(
                input_ids=cur,
                encoder_hidden_states=ehs,
                past_key_values=cache,
                use_cache=True,
            )
            logits = model.proj_out(out.last_hidden_state)[:, -1]
            nxt = apply_suppress(logits, supp, begin, first=(i == 0), eos=eos)
            if call_idx == harvest_call:
                snap(out.past_key_values, K - 1 + i, cur, logits, nxt)
            cache = out.past_key_values
            gen_ids.append(nxt[:, 0].detach().cpu().tolist())
            cur = nxt
            call_idx += 1
    return gen_ids, harvest


class _FullFwd(nn.Module):
    """Whole encoder+decoder forward as a pure ``(feats, ids) -> logits``
    module so ``_bench``/``torch.compile`` see a uniform call signature."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, feats: Any, ids: Any) -> Any:
        return self.model(input_features=feats, decoder_input_ids=ids)


def _time(module: Any, inputs: tuple[Any, ...], iters: int) -> tuple[float, float] | str:
    """``(min_ms, med_ms)`` via the deliver bench, or an honest decline."""
    try:
        return _bench(module, inputs, iters)
    except Exception as e:  # noqa: BLE001 — record, don't hide
        return f"declined: {type(e).__name__}: {e}"


def _fmt(t: Any) -> str:
    if isinstance(t, str):
        return t
    if isinstance(t, tuple) and len(t) == 2:
        mn, md = t
        return f"med={md:.3f}ms min={mn:.3f}ms"
    return repr(t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--search-iters", type=int, default=4)
    ap.add_argument("--ntok", type=int, default=16, help="harvest decode horizon")
    ap.add_argument("--deliver-budget", type=int, default=6)
    ap.add_argument("--deliver-player", default="playbook")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument(
        "--skip-full-compile",
        action="store_true",
        help="skip torch.compile on the full encoder+decoder forward "
        "(the heavy compile — isolate it if the host OOMs)",
    )
    ap.add_argument("--skip-full", action="store_true", help="skip the encoder+decoder arm")
    ap.add_argument("--attn", default="sdpa", help="attn_implementation (sdpa|eager)")
    args = ap.parse_args()

    import transformers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    t0 = time.time()
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn
    )
    model = model.to(device).eval()
    n_mels = int(getattr(model.config, "num_mel_bins", 0) or 80)
    print(
        f"loaded {args.model} dtype={dtype} device={device} "
        f"({time.time() - t0:.1f}s) enc={model.config.encoder_layers} "
        f"dec={model.config.decoder_layers} d={model.config.d_model}"
    )

    # One real-feature probe (B=1) for prefix capture; per-batch feats are
    # randn — values don't move the timing, only the shapes do.
    feats1 = _make_features(n_mels, 8.0, device)
    prefix = _get_prefix(model, feats1)
    K = len(prefix)
    print(f"prefix K={K}: {prefix}")
    del feats1
    torch.cuda.empty_cache() if device.type == "cuda" else None

    full = _FullFwd(model)
    compiled_full = None
    if not args.skip_compile and not args.skip_full_compile:
        try:
            compiled_full = torch.compile(full, dynamic=True)
        except Exception as e:  # noqa: BLE001
            print(f"torch.compile(full): compile() call declined: {e}")

    results: dict[int, dict[str, Any]] = {}
    ehs_by_b: dict[int, Any] = {}

    for B in args.batches:
        print(f"\n=== batch {B} ===")
        row: dict[str, Any] = {}
        feats = torch.randn(B, n_mels, 3000, device=device, dtype=dtype)
        ids = torch.tensor(prefix, device=device).unsqueeze(0).expand(B, -1)

        # -- encoder hidden states for the decode-step module (kept;
        #    encoder deleted only after ALL batches harvested) --
        ehs = _encoder_out(model, feats)
        ehs_by_b[B] = ehs

        if not args.skip_full:
            f_inputs = (feats, ids)
            row["full_eager"] = _time(full, f_inputs, args.iters)
            row["full_compile"] = (
                _time(compiled_full, f_inputs, args.iters)
                if compiled_full is not None
                else "declined: torch.compile unavailable"
            )
            print(f"  full eager : {_fmt(row['full_eager'])}")
            print(f"  full compile: {_fmt(row['full_compile'])}")

        # -- batched decode-step harvest --
        harvest_call = (K - 1) + 8
        try:
            _gen, harvest = _harvest_batched(model, ehs, prefix, args.ntok, device, B, harvest_call)
        except Exception as e:  # noqa: BLE001
            row["step"] = f"declined: harvest failed: {type(e).__name__}: {e}"
            results[B] = row
            print(f"  decode-step harvest DECLINED: {e}")
            continue
        L = max(80, args.ntok + 16)
        inputs = _build_inputs_b(harvest, L)

        wrap = _classes()["wrap"](model, harvest["ehs"], model.config).eval()
        # cheap sanity: wrap reproduces harvested logits argmax
        supp, begin, eos = suppress_sets(model)
        with torch.no_grad():
            out = wrap(*inputs)
            nxt_wrap = apply_suppress(out[0][:, -1].float(), supp, begin, first=False, eos=eos)
            same = bool((nxt_wrap.reshape(-1) == harvest["next_id"].reshape(-1)).all().item())
        row["wrap_sanity_argmax"] = same
        print(f"  wrap sanity argmax={same}")

        row["step_eager"] = _time(wrap, inputs, args.iters)
        print(f"  step eager : {_fmt(row['step_eager'])}")

        if not args.skip_compile:
            try:
                cw = torch.compile(wrap, dynamic=True)
                row["step_compile"] = _time(cw, inputs, args.iters)
            except Exception as e:  # noqa: BLE001
                row["step_compile"] = f"declined: {type(e).__name__}: {e}"
            print(f"  step compile: {_fmt(row['step_compile'])}")

        # -- mlopt delivered runner (delivery game) --
        ck = list(
            inputs[
                3 + 2 * len(model.model.decoder.layers) : 3 + 3 * len(model.model.decoder.layers)
            ]
        )
        cv = list(
            inputs[
                3 + 3 * len(model.model.decoder.layers) : 3 + 4 * len(model.model.decoder.layers)
            ]
        )
        spec = ModelSpec(
            module=wrap,
            inputs=inputs,
            kwargs={},
            meta={
                "ref": f"{args.model}#decode-step-b{B}",
                "family": "whisper",
                "device": str(device),
                "dtype": str(dtype),
                "batch": B,
                "cross_attn_kv": {"ck": ck, "cv": cv},
            },
        )
        try:
            _lowered, report = optimize_model(
                spec,
                max_iterations=args.search_iters,
                deliver=True,
                bench_iters=args.iters,
                deliver_budget=args.deliver_budget,
                deliver_player=args.deliver_player,
            )
            row["mlopt"] = {
                "mechanism": report.deliver_mechanism,
                "runner": report.runner_name,
                "verified": report.verified,
                "eager_ms": report.eager_ms,
                "delivered_ms": report.delivered_ms,
                "speedup": report.speedup,
                "error": report.error,
                "notes": list(report.notes),
            }
            m: dict[str, Any] = row["mlopt"]
            print(
                f"  mlopt       : mechanism={m['mechanism']} runner={m['runner']} "
                f"verified={m['verified']} eager={m['eager_ms']} "
                f"delivered={m['delivered_ms']} speedup={m['speedup']}"
            )
            for n in m["notes"]:
                print(f"    note: {n}")
        except Exception as e:  # noqa: BLE001
            row["mlopt"] = f"declined: {type(e).__name__}: {e}"
            print(f"  mlopt       : DECLINED {type(e).__name__}: {e}")

        results[B] = row
        del wrap, inputs, harvest, ehs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # encoder no longer needed
    if hasattr(model.model, "encoder"):
        del model.model.encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n===== SUMMARY =====")
    for B, row in results.items():
        print(f"batch {B}:")
        for k, v in row.items():
            if k == "mlopt":
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {_fmt(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

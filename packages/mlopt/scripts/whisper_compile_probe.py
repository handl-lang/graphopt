"""Isolated torch.compile probe for the full Whisper encoder+decoder
forward — run in its own process so an inductor OOM kills only this
arm.  Prints eager vs compiled med/min ms per batch; flushes after
every line so a SIGKILL still leaves the completed rows on disk.

    .venv/bin/python -u scripts/whisper_compile_probe.py \
        --model openai/whisper-large-v3-turbo --batches 1 4 8
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn

from mlopt.deliver import _bench


class _FullFwd(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feats, ids):
        return self.model(input_features=feats, decoder_input_ids=ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--prefix", type=int, nargs="+", default=None)
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    import transformers

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model = (
        transformers.WhisperForConditionalGeneration.from_pretrained(  # nosec B615
            args.model, torch_dtype=dtype, attn_implementation=args.attn
        )
        .to(device)
        .eval()
    )
    n_mels = int(getattr(model.config, "num_mel_bins", 0) or 80)

    # whisper prompt prefix: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    prefix = args.prefix or [50257, 50259, 50359, 50363]
    full = _FullFwd(model)

    for B in args.batches:
        feats = torch.randn(B, n_mels, 3000, device=device, dtype=dtype)
        ids = torch.tensor(prefix, device=device).unsqueeze(0).expand(B, -1)
        try:
            mn, md = _bench(full, (feats, ids), args.iters)
            print(f"batch {B} full eager   : med={md:.3f}ms min={mn:.3f}ms", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"batch {B} full eager   : declined {type(e).__name__}: {e}", flush=True)
            continue
        try:
            cf = torch.compile(full, dynamic=True)
            mn, md = _bench(cf, (feats, ids), args.iters)
            print(f"batch {B} full compile : med={md:.3f}ms min={mn:.3f}ms", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"batch {B} full compile : declined {type(e).__name__}: {e}", flush=True)
        del feats, ids
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())

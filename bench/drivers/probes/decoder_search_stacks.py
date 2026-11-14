"""Where does decoder search spend >20min? Stack-dump probe.

Runs ``Optimizer.search`` on the whisper-base decoder step (CPU,
fp32 — the e-graph and IR are host-side) and dumps the Python stack
every ``DUMP_EVERY`` seconds so the hot phase is identifiable.

Usage::

    .venv/bin/python bench/drivers/probes/decoder_search_stacks.py [seconds]
"""

from __future__ import annotations

import faulthandler
import sys
import time

import torch
import torch.nn as nn

MODEL = "openai/whisper-base"


class _DecWrap(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, ids, ehs):
        return self.dec(
            input_ids=ids, encoder_hidden_states=ehs
        ).last_hidden_state


def main() -> None:
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 420
    faulthandler.dump_traceback_later(60, repeat=True)
    from transformers import WhisperForConditionalGeneration
    from mlopt_orchestrator.optimize import Optimizer
    from mlopt_torch.backend import TorchBackend

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    x = torch.randn(1, 80, 3000)
    with torch.no_grad():
        ehs = model.model.encoder(x).last_hidden_state
        start = torch.tensor(
            [[model.generation_config.decoder_start_token_id or 50257]]
        )
    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    try:
        res = optz.search(_DecWrap(model.model.decoder), (start, ehs),
                          max_iterations=3, verbose=True)
        print(f"search done {time.time() - t0:.1f}s", flush=True)
        print(res.stats, flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"search raised {exc!r} after {time.time() - t0:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()

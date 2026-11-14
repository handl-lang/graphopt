"""Decoder search WITH the shape-memo prototype patch.

Monkeypatches ``mlopt_core.typing._shape_of`` so ``memo=None``
callers share one content-keyed memo (the upstream-shaped fix —
every unmemoised caller, incl. eqsat's ``laws.pairing``, gets it).
Then runs ``Optimizer.search`` on the whisper-base decoder step.

Usage::

    .venv/bin/python bench/drivers/probes/decoder_search_fixed.py [iters]
"""

from __future__ import annotations

import resource
import sys
import time

resource.setrlimit(resource.RLIMIT_AS, (9 << 30, 9 << 30))

import torch
import torch.nn as nn

# --- prototype patch -------------------------------------------------
import os

if os.environ.get("MLOPT_SHAPE_MEMO_PATCH", "1") == "1":
    import mlopt_core.typing as _ty

    _orig_shape_of = _ty._shape_of
    _SHAPE_MEMO: dict = {}

    def _shape_of_memo(term, memo=None):
        return _orig_shape_of(
            term, _SHAPE_MEMO if memo is None else memo
        )

    _ty._shape_of = _shape_of_memo
# ---------------------------------------------------------------------

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
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3
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
        res = optz.search(
            _DecWrap(model.model.decoder), (start, ehs),
            max_iterations=iters,
        )
        print(f"search done {time.time() - t0:.1f}s", flush=True)
        print(
            {k: v for k, v in res.stats.items() if k != "param_sharing"},
            flush=True,
        )
        inp = (start, ehs)

        def call(m, i):
            return m(*i) if isinstance(i, tuple) else m(i)

        t0 = time.time()
        low = optz.lower(res, inp, verify=True, runner=None)
        ok = low.verified.passed if low.verified else None
        print(
            f"lower+verify {time.time() - t0:.1f}s verified={ok}",
            flush=True,
        )

        def timeit(fn, warm=2, reps=5):
            with torch.no_grad():
                for _ in range(warm):
                    fn()
                t = time.perf_counter()
                for _ in range(reps):
                    fn()
            return (time.perf_counter() - t) / reps * 1e3

        dec = _DecWrap(model.model.decoder)
        print(
            f"eager {timeit(lambda: call(dec, inp)):.1f} ms  "
            f"delivered {timeit(lambda: call(low.module, inp)):.1f} ms",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(
            f"search raised {exc!r} after {time.time() - t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()

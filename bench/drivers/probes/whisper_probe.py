"""Whisper "lift any model" probe — measurement driver, uncommitted scratch.

Real-world test of the mlopt pipeline on OpenAI Whisper
(``transformers`` ``WhisperForConditionalGeneration``) on a 4 GB
GPU (RTX 2050).  Arms: eager / torch.compile / mlopt search+lower /
``CudaGraphRunner`` delivery, on the encoder and (wrapper) decoder
submodules of ``openai/whisper-base`` fp32 — the largest checkpoint
whose lower+verify fit in 4 GB (see docs/reports/whisper_experiment.md
for the turbo-bf16 and whisper-small OOM boundaries).

Probe-only workaround applied at import: ``IRModule._align_to`` moves
only *floating-point* baked tensors to the input device; lifted
integer index tensors (the positional-embedding arange feeding
``nn.Embedding``) stay on CPU and the delivery eval dies with a
device mismatch.  The wrapper below extends the same move to non-float
tensor leaves.  This is a real upstream gap, not probe semantics.

Usage::

    .venv/bin/python bench/drivers/probes/whisper_probe.py bench
    .venv/bin/python bench/drivers/probes/whisper_probe.py opt encoder
    .venv/bin/python bench/drivers/probes/whisper_probe.py opt decoder
"""

from __future__ import annotations

import gc
import sys
import time

import torch
import torch.nn as nn
from mlopt_orchestrator.optimize import Optimizer
from mlopt_torch.backend import TorchBackend
from mlopt_torch.torch_bridge import IRModule
from transformers import WhisperForConditionalGeneration

# --- probe workaround: move non-float tensor leaves too -------------
_orig_align = IRModule._align_to


def _align_all(self, xs):
    _orig_align(self, xs)
    like = next(
        (x for x in xs if isinstance(x, torch.Tensor) and x.dim() > 0),
        None,
    )
    if like is None or like.device.type == "cpu":
        return
    for table in (
        self._param_map,
        getattr(self, "_const_map", {}) or {},
    ):
        for k, t in list(table.items()):
            if isinstance(t, torch.Tensor) and t.device != like.device:
                table[k] = t.detach().to(device=like.device)


IRModule._align_to = _align_all
# ---------------------------------------------------------------------

MODEL = "openai/whisper-base"
N_MELS = 80
N_FRAMES = 3000  # 30 s of log-mel


def load():
    return (
        WhisperForConditionalGeneration.from_pretrained(
            MODEL,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        .cuda()
        .eval()
    )


def timeit(fn, warm=5, reps=30):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def bench() -> None:
    model = load()
    x = torch.randn(1, N_MELS, N_FRAMES, device="cuda")
    with torch.no_grad():
        ehs = model.model.encoder(x).last_hidden_state
    start = torch.tensor(
        [[model.generation_config.decoder_start_token_id or 50257]],
        device="cuda",
    )

    def enc():
        return model.model.encoder(x)

    def dec():
        return model.model.decoder(
            input_ids=start, encoder_hidden_states=ehs
        )

    print(f"encoder-eager   {timeit(enc):8.2f} ms")
    print(f"decoder-eager   {timeit(dec):8.2f} ms")
    ec = torch.compile(model.model.encoder)
    dc = torch.compile(model.model.decoder)
    with torch.no_grad():
        ec(x)
        dc(input_ids=start, encoder_hidden_states=ehs)
        torch.cuda.synchronize()
    print(f"encoder-compile {timeit(lambda: ec(x)):8.2f} ms")
    print(
        f"decoder-compile {timeit(lambda: dc(input_ids=start, encoder_hidden_states=ehs)):8.2f} ms"
    )


class _DecWrap(nn.Module):
    """Decoder step -> logits only.

    ``torch.export`` refuses the decoder's ``EncoderDecoderCache``
    return (unregistered pytree) — the wrapper returns
    ``last_hidden_state`` (already projected to vocab logits).
    """

    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, ids, ehs):
        return self.dec(
            input_ids=ids, encoder_hidden_states=ehs
        ).last_hidden_state


def opt(which: str, iters: int = 3) -> None:
    from mlopt_cuda import CudaGraphRunner

    model = load()
    x = torch.randn(1, N_MELS, N_FRAMES, device="cuda")
    with torch.no_grad():
        ehs = model.model.encoder(x).last_hidden_state
    start = torch.tensor(
        [[model.generation_config.decoder_start_token_id or 50257]],
        device="cuda",
    )
    if which == "encoder":
        mod, inp = model.model.encoder, x
    else:
        mod, inp = _DecWrap(model.model.decoder), (start, ehs)

    def call(m, i):
        return m(*i) if isinstance(i, tuple) else m(i)

    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    res = optz.search(mod, inp, max_iterations=iters)
    print(f"[{which}] search {time.time() - t0:.1f}s", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    low = optz.lower(res, inp, verify=True, runner=None)
    ok = low.verified.passed if low.verified else None
    print(
        f"[{which}] lower+verify {time.time() - t0:.1f}s verified={ok}",
        flush=True,
    )
    print(
        f"[{which}] delivered {timeit(lambda: call(low.module, inp)):.2f} ms",
        flush=True,
    )

    t0 = time.time()
    low2 = optz.lower(res, inp, verify=False, runner=CudaGraphRunner())
    print(
        f"[{which}] graph cuda_graph={low2.stats.get('cuda_graph')} "
        f"fail={low2.stats.get('cuda_graph_failure')}",
        flush=True,
    )
    if low2.stats.get("cuda_graph"):
        print(
            f"[{which}] delivered+graph "
            f"{timeit(lambda: call(low2.module, inp)):.2f} ms",
            flush=True,
        )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "bench":
        bench()
    elif sys.argv[1] == "opt":
        opt(sys.argv[2] if len(sys.argv) > 2 else "encoder")


if __name__ == "__main__":
    main()

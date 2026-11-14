"""Whisper "lift any model" probe — measurement driver, uncommitted scratch.

Real-world test of the mlopt pipeline on OpenAI Whisper
(``transformers`` ``WhisperForConditionalGeneration``) on a 4 GB
GPU (RTX 2050).  Arms: eager / torch.compile / mlopt search+lower /
``CudaGraphRunner`` delivery, on the encoder and (wrapper) decoder
submodules.

Round 2 (docs/reports/whisper_experiment_r2.md): the r1 walls are
fixed upstream — bf16 law-bank dedup (eqsat d7e9a57), non-float leaf
placement in ``IRModule._align_to``, and ``_EvalPlan`` liveness
freeing.  The probe monkeypatch from r1 is gone; if a wall regresses
this script fails honestly.

Usage::

    .venv/bin/python bench/drivers/probes/whisper_probe.py bench [model] [dtype]
    .venv/bin/python bench/drivers/probes/whisper_probe.py opt <which> [model] [dtype] [max_enodes]

``model``: turbo | small | base   (default base)
``dtype``: bf16 | fp32             (default: bf16 for turbo, fp32 else)
``which``: encoder | decoder
"""

from __future__ import annotations

import gc
import sys
import time

import torch
import torch.nn as nn
from mlopt_orchestrator.optimize import Optimizer
from mlopt_torch.backend import TorchBackend
from transformers import WhisperForConditionalGeneration

MODELS = {
    "turbo": "openai/whisper-large-v3-turbo",
    "small": "openai/whisper-small",
    "base": "openai/whisper-base",
}
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
# large-v3 family takes 128 mel bins; tiny/base/small take 80.
N_MELS = {"turbo": 128, "small": 80, "base": 80}
N_FRAMES = 3000  # 30 s of log-mel


def _parse(offset: int = 2) -> tuple[str, str]:
    """``bench``: argv[2]=model argv[3]=dtype; ``opt``: offset=3."""
    model = sys.argv[offset] if len(sys.argv) > offset else "base"
    dtype = (
        sys.argv[offset + 1]
        if len(sys.argv) > offset + 1
        else ("bf16" if model == "turbo" else "fp32")
    )
    return model, dtype


def load(model: str, dtype: str):
    m: nn.Module = WhisperForConditionalGeneration.from_pretrained(
        MODELS[model],
        torch_dtype=DTYPES[dtype],
        attn_implementation="eager",
    )
    # transformers' ``from_pretrained`` return type trips ty's
    # ``_Wrapped.__call__`` resolution on ``.to`` — runtime is fine.
    return m.to("cuda").eval()  # ty: ignore[invalid-argument-type]


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


def _feeds(model, dtype):
    """Encoder input + a precomputed encoder_hidden_states + start ids."""
    m = load(model, dtype)
    x = torch.randn(
        1,
        N_MELS[model],
        N_FRAMES,
        device="cuda",
        dtype=DTYPES[dtype],
    )
    with torch.no_grad():
        ehs = m.model.encoder(x).last_hidden_state
    start = torch.tensor(
        [[m.generation_config.decoder_start_token_id or 50257]],
        device="cuda",
    )
    return m, x, ehs, start


def bench() -> None:
    model, dtype = _parse(2)
    m, x, ehs, start = _feeds(model, dtype)

    def enc():
        return m.model.encoder(x)

    def dec():
        return m.model.decoder(
            input_ids=start, encoder_hidden_states=ehs
        )

    print(f"[{model}/{dtype}] encoder-eager   {timeit(enc):8.2f} ms")
    print(f"[{model}/{dtype}] decoder-eager   {timeit(dec):8.2f} ms")
    ec = torch.compile(m.model.encoder)
    dc = torch.compile(m.model.decoder)
    with torch.no_grad():
        ec(x)
        dc(input_ids=start, encoder_hidden_states=ehs)
        torch.cuda.synchronize()
    print(
        f"[{model}/{dtype}] encoder-compile {timeit(lambda: ec(x)):8.2f} ms"
    )
    print(
        f"[{model}/{dtype}] decoder-compile "
        f"{timeit(lambda: dc(input_ids=start, encoder_hidden_states=ehs)):8.2f} ms"
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

    model, dtype = _parse(3)
    max_enodes = int(sys.argv[5]) if len(sys.argv) > 5 else 100_000
    m, x, ehs, start = _feeds(model, dtype)
    if which == "encoder":
        mod, inp = m.model.encoder, x
    else:
        mod, inp = _DecWrap(m.model.decoder), (start, ehs)

    def call(mm, i):
        return mm(*i) if isinstance(i, tuple) else mm(i)

    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    try:
        res = optz.search(
            mod,
            inp,
            max_iterations=iters,
            max_enodes=max_enodes,
        )
        print(
            f"[{which} {model}/{dtype}] search {time.time() - t0:.1f}s "
            f"enodes={getattr(res, 'n_enodes', '?')}",
            flush=True,
        )
    except Exception as e:
        print(
            f"[{which} {model}/{dtype}] search FAILED after "
            f"{time.time() - t0:.1f}s: {type(e).__name__}: {e}",
            flush=True,
        )
        return
    del m
    # ``lower()`` reads only ir/term/param_values/stats — the
    # saturated e-graph and the provenance handles are dead weight
    # that can pin gigabytes of leaf tensors on a 4 GB card.
    res.eg = None  # ty: ignore[invalid-assignment]
    res.model = None
    res.source = None
    if len(sys.argv) > 6 and sys.argv[6] == "cpuparams":
        # param_values pins a full GPU copy of every weight while
        # the delivered module materialises another — park the
        # source values on CPU; ``_align_to`` moves them back to
        # the input device at first eval.
        res.param_values = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in res.param_values.items()
        }
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    low = None
    try:
        low = optz.lower(res, inp, verify=True, runner=None)
        ok = low.verified.passed if low.verified else None
        print(
            f"[{which} {model}/{dtype}] lower+verify "
            f"{time.time() - t0:.1f}s verified={ok}",
            flush=True,
        )
    except Exception as e:
        print(
            f"[{which} {model}/{dtype}] lower+verify FAILED after "
            f"{time.time() - t0:.1f}s: {type(e).__name__}: {e}",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
    if low is None:
        # Verify-time duplication can OOM where the unverified
        # delivery still fits — measure what we can.
        t0 = time.time()
        try:
            low = optz.lower(res, inp, verify=False, runner=None)
            print(
                f"[{which} {model}/{dtype}] lower(verify=False) "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
        except Exception as e:
            print(
                f"[{which} {model}/{dtype}] lower(verify=False) "
                f"FAILED: {type(e).__name__}: {e}",
                flush=True,
            )
            return
    try:
        print(
            f"[{which} {model}/{dtype}] delivered "
            f"{timeit(lambda: call(low.module, inp)):.2f} ms",
            flush=True,
        )
    except Exception as e:
        print(
            f"[{which} {model}/{dtype}] delivered forward FAILED: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

    t0 = time.time()
    try:
        low2 = optz.lower(
            res, inp, verify=False, runner=CudaGraphRunner()
        )
        print(
            f"[{which} {model}/{dtype}] graph "
            f"cuda_graph={low2.stats.get('cuda_graph')} "
            f"fail={low2.stats.get('cuda_graph_failure')}",
            flush=True,
        )
        if low2.stats.get("cuda_graph"):
            print(
                f"[{which} {model}/{dtype}] delivered+graph "
                f"{timeit(lambda: call(low2.module, inp)):.2f} ms",
                flush=True,
            )
    except Exception as e:
        print(
            f"[{which} {model}/{dtype}] graph lower FAILED: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "bench":
        bench()
    elif sys.argv[1] == "opt":
        opt(sys.argv[2] if len(sys.argv) > 2 else "encoder")


if __name__ == "__main__":
    main()

"""Decoder-layer pay probe — does the spelled-attention fold pay?

Exports one whisper-base decoder layer (``decoder_layers_census``'s
``_LayerWrap`` — hook-captured decode-step inputs, fresh KV cache per
call), runs ``Optimizer.search`` end-to-end, then reports:

* whether the extracted term contains ``sdpa`` enodes (rewrite yield);
* ``lower(verify=True)`` outcome (the ``where``/``arg2`` torch_bridge
  gap does not arise — this layer's IR has no ``where``);
* eager vs delivered wall time on CPU.

Usage::

    .venv/bin/python bench/drivers/probes/decoder_layer_pay.py [iters]
"""

from __future__ import annotations

import resource
import sys
import time

resource.setrlimit(resource.RLIMIT_AS, (9 << 30, 9 << 30))

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import decoder_layer_census as C  # noqa: E402

MODEL = C.MODEL


def count_ops(term, op):
    from mlopt_core.ir import Op

    seen = set()
    n = 0

    def walk(t):
        nonlocal n
        if isinstance(t, Op) and id(t) not in seen:
            seen.add(id(t))
            if t.op == op:
                n += 1
            for a in t.args:
                walk(a)

    walk(term)
    return n


def main() -> None:
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    import mlopt_carriers  # noqa: F401 — registers carriers + new laws
    import mlopt_torch  # noqa: F401
    from mlopt_orchestrator.optimize import Optimizer
    from mlopt_torch.backend import TorchBackend
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    hs, mask, ehs = C.capture_inputs(model)
    wrap = C._LayerWrap(model.model.decoder.layers[0], model.config, mask, ehs)

    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    res = optz.search(wrap, (hs,), max_iterations=iters)
    dt = time.time() - t0
    print(f"[search] {dt:.1f}s", flush=True)
    print(
        {k: v for k, v in res.stats.items() if k != "param_sharing"},
        flush=True,
    )
    term = res.best_term if hasattr(res, "best_term") else None
    if term is None:
        # find the extracted term attribute
        for attr in ("term", "extracted", "best"):
            if hasattr(res, attr):
                term = getattr(res, attr)
                break
    if term is not None:
        print(
            f"[extract] sdpa={count_ops(term, 'sdpa')} "
            f"matmul={count_ops(term, 'matmul')} "
            f"softmax={count_ops(term, 'softmax')}",
            flush=True,
        )

    t0 = time.time()
    low = optz.lower(res, (hs,), verify=True, runner=None)
    ok = low.verified.passed if low.verified else None
    print(f"[lower+verify] {time.time() - t0:.1f}s verified={ok}", flush=True)

    def call(m):
        return m(hs)

    def timeit(fn, warm=3, reps=10):
        with torch.no_grad():
            for _ in range(warm):
                fn()
            t = time.perf_counter()
            for _ in range(reps):
                fn()
        return (time.perf_counter() - t) / reps * 1e3

    print(
        f"[time] eager {timeit(lambda: call(wrap)):.2f} ms  "
        f"delivered {timeit(lambda: call(low.module)):.2f} ms",
        flush=True,
    )


if __name__ == "__main__":
    main()

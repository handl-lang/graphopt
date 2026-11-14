"""T5 "lift any model" probe — second-model plateau test, uncommitted scratch.

Sibling of ``whisper_probe.py``: same pipeline (eager / torch.compile /
mlopt search+lower / ``CudaGraphRunner``) on a *different* real HF
model — ``google-t5/t5-small`` (60.5 M params, encoder-decoder,
relative-position-bias attention, gated-GELU MLP — none of which
whisper-base/small/turbo exercise).  4 GB GPU (RTX 2050), honest
walls: every stage that fails is reported, never forced.

Usage::

    .venv/bin/python bench/drivers/probes/t5_probe.py bench [dtype]
    .venv/bin/python bench/drivers/probes/t5_probe.py opt <which> [dtype] [max_enodes] [cpuparams]
    .venv/bin/python -m mlopt_discovery.model_corpus \
        --module bench.drivers.probes.t5_probe:corpus_encoder \
        --mutants 4 --corpus bench --reps 3 --gen-reps 3 --dtype fp32 \
        --json /tmp/t5_mc.json

``which``: encoder | decoder
``dtype``: bf16 | fp32             (default fp32)
"""

from __future__ import annotations

import gc
import sys
import time

import torch
import torch.nn as nn
from mlopt_orchestrator.optimize import Optimizer
from mlopt_torch.backend import TorchBackend
from transformers import T5ForConditionalGeneration

MODEL_ID = "google-t5/t5-small"
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
SEQ = 128
VOCAB = 32100


def _patch_min_max() -> None:
    """Work around exporter wall #2 (see docs/reports/t5_r1.md).

    ``torch.min(a, b)`` / ``torch.max(a, b)`` — the *elementwise*
    two-operand spellings (``aten.minimum``/``aten.maximum``) — are
    named by their Python builtin (``__name__ == "min"``/``"max"``)
    and mint IR ops ``min``/``max``, whose torch bindings are the
    whole-tensor *reductions* ``torch.min(x)``/``torch.max(x)``.  The
    second operand is silently dropped and the scalar reduction
    broadcasts — wrong values, not a refusal.  T5's
    ``relative_position_bucket`` hits exactly this
    (``torch.min(rel_if_large, full_like(...))``): the bucket index
    collapses to INT64_MIN and the relative-bias ``embedding`` gather
    runs out of bounds (CPU ``IndexError`` / CUDA device-side assert).

    The patch re-dispatches the two-operand spellings to
    ``torch.minimum``/``torch.maximum`` — semantically identical
    (``torch.min(a, b)`` IS elementwise minimum) — so the exporter
    sees ``aten.minimum`` → IR op ``minimum`` → the correct binding.
    Single-operand/reduction spellings pass through untouched.
    Applied inside ``load()`` so every arm (bench/opt/corpus) sees
    the same semantics; it is a probe-local workaround, not a fix.
    """
    if getattr(torch, "_t5_probe_minmax_patched", False):
        return
    _orig_min = torch.min
    _orig_max = torch.max

    def _min(*a, **k):
        if len(a) > 1 or "other" in k:
            return torch.minimum(*a, **k)
        return _orig_min(*a, **k)

    def _max(*a, **k):
        if len(a) > 1 or "other" in k:
            return torch.maximum(*a, **k)
        return _orig_max(*a, **k)

    torch.min = _min  # ty: ignore[invalid-assignment]
    torch.max = _max  # ty: ignore[invalid-assignment]
    torch._t5_probe_minmax_patched = True  # ty: ignore[attr-defined]


def load(dtype: str = "fp32"):
    _patch_min_max()
    m: nn.Module = T5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPES[dtype],
        attn_implementation="eager",
    )
    # transformers' ``from_pretrained`` return type trips ty's
    # ``_Wrapped.__call__`` resolution on ``.to`` — runtime is fine.
    return m.to("cuda").eval()  # ty: ignore[invalid-argument-type]


def _feeds(dtype: str):
    """input_ids + attention_mask + a precomputed encoder output."""
    m = load(dtype)
    g = torch.Generator(device="cuda").manual_seed(0)
    ids = torch.randint(0, VOCAB, (1, SEQ), device="cuda", generator=g)
    # attention_mask is FLOAT, not long: ``IRModule._align_to`` keys
    # device/dtype alignment off the first floating *input* tensor —
    # an all-integer feed (int64 ids + int64 mask) leaves it a no-op,
    # so CPU-baked ``fused_*`` fold tensors collide with CUDA params
    # at first eval.  A float mask is numerically identical for an
    # all-ones mask (verified: max diff 0.0 vs long) and engages the
    # align path.  This is probe wall #1 — see docs/reports/t5_r1.md.
    # Mask dtype follows the model dtype so ``_align_to``'s
    # input-dtype keying keeps bf16 weights bf16 (an fp32 mask would
    # silently upcast every delivered weight — see t5_r1.md).
    mask = torch.ones(1, SEQ, device="cuda", dtype=DTYPES[dtype])
    with torch.no_grad():
        ehs = m.encoder(
            input_ids=ids, attention_mask=mask
        ).last_hidden_state
    return m, ids, mask, ehs


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
    dtype = sys.argv[2] if len(sys.argv) > 2 else "fp32"
    m, ids, mask, ehs = _feeds(dtype)

    def enc():
        return m.encoder(input_ids=ids, attention_mask=mask)

    def dec():
        return m.decoder(
            input_ids=ids,
            attention_mask=mask,
            encoder_hidden_states=ehs,
            encoder_attention_mask=mask,
        )

    print(f"[t5-small/{dtype}] encoder-eager   {timeit(enc):8.2f} ms")
    print(f"[t5-small/{dtype}] decoder-eager   {timeit(dec):8.2f} ms")
    ec = torch.compile(m.encoder)
    dc = torch.compile(m.decoder)
    with torch.no_grad():
        ec(input_ids=ids, attention_mask=mask)
        dc(
            input_ids=ids,
            attention_mask=mask,
            encoder_hidden_states=ehs,
            encoder_attention_mask=mask,
        )
        torch.cuda.synchronize()
    print(
        f"[t5-small/{dtype}] encoder-compile "
        f"{timeit(lambda: ec(input_ids=ids, attention_mask=mask)):8.2f} ms"
    )
    dec_c = timeit(
        lambda: dc(
            input_ids=ids,
            attention_mask=mask,
            encoder_hidden_states=ehs,
            encoder_attention_mask=mask,
        )
    )
    print(f"[t5-small/{dtype}] decoder-compile {dec_c:8.2f} ms")


class _EncWrap(nn.Module):
    """Encoder -> last_hidden_state tensor (export/corpus-safe)."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, ids, mask):
        return self.enc(
            input_ids=ids, attention_mask=mask
        ).last_hidden_state


class _DecWrap(nn.Module):
    """Decoder step -> last_hidden_state only.

    ``torch.export`` refuses the decoder's ``BaseModelOutputWithPast``
    return (unregistered pytree) — the wrapper returns the tensor.
    """

    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, ids, mask, ehs):
        return self.dec(
            input_ids=ids,
            attention_mask=mask,
            encoder_hidden_states=ehs,
            encoder_attention_mask=mask,
        ).last_hidden_state


def corpus_encoder():
    """``model_corpus --module`` callable: (wrapped encoder, feed).

    The wrapper returns a plain Tensor — the corpus verify/delivery
    subtracts outputs and honestly declines non-Tensor returns
    (whisper's ``BaseModelOutput`` row declined the same way).
    """
    m = load("fp32")
    g = torch.Generator(device="cuda").manual_seed(0)
    ids = torch.randint(0, VOCAB, (1, SEQ), device="cuda", generator=g)
    mask = torch.ones(1, SEQ, device="cuda", dtype=torch.float32)
    return _EncWrap(m.encoder), (ids, mask)


def corpus_decoder():
    """Same for the decoder — included for completeness; expect the
    whisper wall #3 (decoder e-graph explosion) to repeat."""
    m = load("fp32")
    g = torch.Generator(device="cuda").manual_seed(0)
    ids = torch.randint(0, VOCAB, (1, SEQ), device="cuda", generator=g)
    mask = torch.ones(1, SEQ, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        ehs = m.encoder(
            input_ids=ids, attention_mask=mask
        ).last_hidden_state
    return _DecWrap(m.decoder), (ids, mask, ehs)


def opt(which: str, iters: int = 3) -> None:
    from mlopt_cuda import CudaGraphRunner

    dtype = sys.argv[3] if len(sys.argv) > 3 else "fp32"
    max_enodes = int(sys.argv[4]) if len(sys.argv) > 4 else 100_000
    m, ids, mask, ehs = _feeds(dtype)
    if which == "encoder":
        mod, inp = _EncWrap(m.encoder), (ids, mask)
    else:
        mod, inp = _DecWrap(m.decoder), (ids, mask, ehs)

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
            f"[{which} t5-small/{dtype}] search {time.time() - t0:.1f}s "
            f"enodes={getattr(res, 'n_enodes', '?')}",
            flush=True,
        )
    except Exception as e:
        print(
            f"[{which} t5-small/{dtype}] search FAILED after "
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
    if len(sys.argv) > 5 and sys.argv[5] == "cpuparams":
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
            f"[{which} t5-small/{dtype}] lower+verify "
            f"{time.time() - t0:.1f}s verified={ok}",
            flush=True,
        )
    except Exception as e:
        print(
            f"[{which} t5-small/{dtype}] lower+verify FAILED after "
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
                f"[{which} t5-small/{dtype}] lower(verify=False) "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
        except Exception as e:
            print(
                f"[{which} t5-small/{dtype}] lower(verify=False) "
                f"FAILED: {type(e).__name__}: {e}",
                flush=True,
            )
            return
    try:
        print(
            f"[{which} t5-small/{dtype}] delivered "
            f"{timeit(lambda: call(low.module, inp)):.2f} ms",
            flush=True,
        )
    except Exception as e:
        print(
            f"[{which} t5-small/{dtype}] delivered forward FAILED: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

    t0 = time.time()
    try:
        low2 = optz.lower(
            res, inp, verify=False, runner=CudaGraphRunner()
        )
        print(
            f"[{which} t5-small/{dtype}] graph "
            f"cuda_graph={low2.stats.get('cuda_graph')} "
            f"fail={low2.stats.get('cuda_graph_failure')}",
            flush=True,
        )
        if low2.stats.get("cuda_graph"):
            print(
                f"[{which} t5-small/{dtype}] delivered+graph "
                f"{timeit(lambda: call(low2.module, inp)):.2f} ms",
                flush=True,
            )
    except Exception as e:
        print(
            f"[{which} t5-small/{dtype}] graph lower FAILED: "
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

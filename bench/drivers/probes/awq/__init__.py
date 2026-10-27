"""AWQ / int4 probes — **DEMO CODE, not a certified component**.

These are the experiments behind `docs/reports/p6*`–`p9*`: the fused
dequant op, the int4 GEMV, split-k, the sweeps against `awq_gemm`, and
the vLLM injections. They measure; they do not ship.

**They are deliberately not substrate-native.** Nothing here goes
through `KERNEL_RULES`, `KernelSink`, the cost model or the sieve, so
none of it is minted, priced or certified — it is a hand-written kernel
plus monkeypatches, kept to answer "is the physics there?".

Phase 3 of
`project/reports/2026-10-27-plan-roofline-to-search.md` is where this
becomes a component. Until then: demo, and labelled as one.

Run them as modules from the repo root (they import each other):

    .venv/bin/python -m bench.drivers.probes.awq.kernel_v2
    VLLM_ENABLE_V1_MULTIPROCESSING=0 /path/to/vllm-python \\
        -m bench.drivers.probes.awq.vllm_awq_e2e --model <awq ckpt>
"""

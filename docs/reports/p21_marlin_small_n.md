# P21 — The `atomic_add` lever is empty, and the small-`n` win is 2.5 µs

Test of the one opportunity P19 named: vLLM's *"better performance for small
size_n with experimental use_atomic_add"*. Probe:
`bench/drivers/probes/awq/marlin_small_n.py`, JSON in
`p21_marlin_small_n.json`.

## First: my reading of the hint was wrong

The gate is not about small `n` in absolute terms — `marlin_utils.py`:

```python
if n >= 2048 or k < 2048 or device.type != "cuda":
    return False
```

So on this model **only `down_proj` (`n=896, k=4864`) is ever eligible**.
`k_proj`/`v_proj` (`n=128, k=896`) fail on `k < 2048`, `qkv` likewise, and
`gate_up` fails on `n >= 2048`. "Small size_n" means small *relative to K*.

Measured anyway, by flipping `VLLM_MARLIN_USE_ATOMIC_ADD` at runtime:

| `down_proj` | synced | b2b | graph | rel err |
|---|---|---|---|---|
| default | 59.4 µs | 31.3 µs | **28.6 µs** | 0.0013 |
| `atomic_add` | 60.2 µs | 31.6 µs | **28.7 µs** | 0.0024 |

**It buys nothing** (−0.3 %). Correctness is fine; the optimisation just does
not pay on this GPU. The lever is closed.

## Second: three clocks, and only one is the truth

Same kernel, same shape, three timers:

| shape | synced/call | back-to-back | CUDA graph |
|---|---|---|---|
| `k_proj` 128×896 | 35.0 | 29.2 | **7.5** |
| `o_proj` 896×896 | 34.5 | 27.7 | **8.3** |
| `down_proj` 896×4864 | 59.4 | 31.3 | **28.6** |
| `gate_up` 9728×896 | 85.7 | 59.0 | **55.0** |

`kernel_v2.timeit` syncs after every call, so at these sizes it times launch
overhead, not the kernel — up to **4.7× off on `k_proj`**. The graph timer is
the one that matters: vLLM captures decode into CUDA graphs by default, and
it reproduces P19's in-engine profiler almost exactly — 8.4 + 8.3 + 55.0 +
28.6 = **100.3 µs/layer** against the profiled **104.8**.

## Third: the small-`n` win exists, and is worth 2.5 µs

All numbers graph-timed (pure GPU), % of the 89.1 GB/s byte roofline:

| shape | roofline | Marlin | `awq_gemm` | ours | fp16 | ours/Marlin |
|---|---|---|---|---|---|---|
| `k_proj` 128×896 | 0.67 | 7.5 (8.9 %) | 23.5 | **5.0 (13.4 %)** | 3.1 | **1.50×** |
| `qkv` 1152×896 | 5.84 | 8.4 (69.6 %) | 31.6 | 16.5 (35.5 %) | 25.2 | 0.51× |
| `o_proj` 896×896 | 4.55 | 8.3 (54.8 %) | 31.5 | 13.7 (33.2 %) | 20.7 | 0.61× |
| `gate_up` 9728×896 | 49.15 | 55.0 (89.4 %) | 55.7 | 84.7 (58.0 %) | 191.6 | 0.65× |
| `down_proj` 896×4864 | 24.59 | 28.6 (85.9 %) | 152.3 | 40.6 (60.6 %) | 98.1 | 0.71× |

So `k_proj` is the **only shape where we beat Marlin** — and it saves 2.5 µs
on a shape whose entire roofline is 0.67 µs. It is a latency win at 13 % of
roofline, not a bandwidth win.

**It also does not survive fusion.** vLLM fuses q/k/v into one `qkv` call,
and on that shape we are **0.51×** — a 8.1 µs loss per layer, three times the
`k_proj` gain, before counting `v_proj`. Routing the model to our kernel
would cost ~0.71 ms/token on `gate_up` alone to save ~0.12 ms/token on the
small shapes.

## Fourth: Marlin is at 86–89 % of the roofline, as advertised

`gate_up` 89.4 %, `down_proj` 85.9 %. Against fp16 on the same shape Marlin
is 3.48× faster (`gate_up` 55.0 vs 191.6 µs) — **87 % of the theoretical 4×**.
The remaining headroom over Marlin on its tuned shapes is ~12 %, and we are
not the ones holding it.

## Defects this exposed in our own harness

1. **The correctness gate was vacuous.** `kernel_v2.main` feeds AWQ's native
   nibble order (`[0,4,1,5,2,6,3,7]`) to *both* the kernel and the "reference",
   so they agree with each other while neither matches AWQ: cosine **0.21**
   against `awq_gemm`. `awq_packing.py` documented exactly this bug; the
   probe bypassed the fix. With `to_standard_order` on both packed tensors,
   the reference matches `awq_gemm` at cos 1.000000 and max|Δ| ≈ 1e-3.
2. **Configs were ranked on the wrong clock.** P9/P10 chose configs with
   sync-per-call timing. Re-rank with graph time before quoting any config.
3. **`max_abs_err` in P9 (4.7 / 8.1 / 11.9) was the reference's fault**, not
   the kernel's. P14's "exact" claim is only exact against that reference.
4. One earlier claim survives: ours vs `awq_gemm` on `down_proj` is **3.75×**
   (40.6 vs 152.3 µs, graph). `awq_gemm` is genuinely bad there.

## Verdict on the opportunity

Closed, on all three counts: the lever does nothing, the small-`n` win is
2.5 µs and disappears under fusion, and Marlin already holds 87 % of the 4×
budget. **The path forward is not "beat Marlin on Qwen2.5-0.5B AWQ"** — it is
the unattended regime (P20), where nobody's heuristic is tuned.

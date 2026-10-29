# P22 — P9/P10 re-derived: the gate was vacuous, the timer was wrong

Probe: `bench/drivers/probes/awq/paired_gate.py`, JSON `p22_paired.json`.
Supersedes the numbers in `p9_kernel_v2_gate.md`, `p10_kernel_search.json`
and the exactness claim in `p14_correct_splitk.md`.

## What was wrong

1. **The reference shared the kernel's mistake.** `unpack_ref` was fed the
   checkpoint's native AWQ nibble order, and so was the kernel. They agreed
   with each other and disagreed with `awq_gemm` at **cosine 0.21**. Every
   `max_abs_err` in P9 (4.7 / 8.1 / 11.9) was the reference's error, not
   the kernel's. `awq_packing.py` had already documented the bug; the
   probe bypassed the fix.
2. **The timer measured launch overhead.** `timeit` syncs after every
   call; at 5–60 µs kernels that is a different quantity from the kernel.
3. **Configs were ranked on (2)**, so P9/P10's chosen configs were chosen
   by the wrong clock.

Fixed here: `to_standard_order` on both packed tensors, `cosine_gate`
against `awq_gemm` (`bench/benchkit/equality.py`), and two timers that
agree — `blocked_paired` (deploy) and a captured CUDA graph (serving).

## Re-derived, every output at cosine 1.000000 vs `awq_gemm`

| shape | roofline | deploy ours | deploy `awq_gemm` | ratio | graph ours | graph `awq_gemm` | ratio | % roof (graph) |
|---|---|---|---|---|---|---|---|---|
| `down_proj` 896×4864 | 24.6 | 49.4 | 159.2 | **3.22×** | 42.3 | 157.8 | **3.73×** | 58.1 % |
| `gate_up` 9728×896 | 49.2 | 103.3 | 62.1 | **0.60×** | 95.9 | 57.0 | **0.59×** | 51.3 % |
| `qkv` 1152×896 | 5.8 | 35.2 | 37.0 | 1.05× | 14.9 | 31.7 | 2.14× | 39.3 % |
| `o_proj` 896×896 | 4.6 | 33.3 | 36.4 | 1.09× | 12.7 | 31.6 | 2.49× | 35.8 % |
| `k_proj` 128×896 | 0.7 | 35.0 | 28.4 | 0.81× | 5.2 | 24.2 | 4.70× | 12.9 % |

## What survives, what changes

- **`down_proj` 3.2–3.7× over `awq_gemm` survives** — now gated against the
  incumbent instead of against ourselves. P9's 3.38× was, by luck, right.
- **`gate_up` is 0.6×, not P9's 0.94×.** On the fused shape the model
  actually calls, we are 1.7× *slower* than `awq_gemm`.
- **Roofline: 58.1 % on `down_proj`**, not P9's 42.3 % — same kernel, an
  honest clock and honest weights.
- **Where the two timers disagree is information, not noise.** On
  `down_proj` and `gate_up` (computation ≫ launch) they agree to 5 %. On
  `qkv`, `o_proj`, `k_proj` (computation ≈ launch) deploy says parity and
  the graph says 2.1–4.7×: our two-stage split kernel pays a **second
  launch**, which vanishes under CUDA-graph serving (vLLM's default) and
  dominates in an eager loop. Any claim about narrow shapes must say which
  regime it is in.

## Two defects in the referee itself

1. **`DEFAULT_EVICT_MB = 64` is 32× this card's L2.** On a 4 GB card it
   turned 5–50 µs kernels into **315–463 µs** measurements — both
   candidates, ~10× inflation — because the flush leaves the allocator
   under pressure. With `evict_mb=8` the referee is sane again. Every
   `blocked_paired` number ever taken on this card with the default is
   suspect.
2. **Cross-run comparison is invalid without locked clocks.** Identical
   code measured `awq_gemm` at 38 µs and 285 µs minutes apart; the second
   run had SM 1755 MHz, 75 °C, 23 W. Ratios are only meaningful when the
   incumbent is timed *inside* the same run, which is why every row above
   carries its own incumbent column.

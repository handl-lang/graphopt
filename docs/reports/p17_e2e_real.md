# P17 — the end-to-end test, finally real: ~2 %, and not resolvable here

Every earlier end-to-end run was invalid — the vLLM patch never fired
(`routed: 0`). P16 identified why: vLLM repacks AWQ checkpoints for
Marlin, so the checkpoint-layout tensors the patch looked for no longer
exist by the time `apply` runs.

**The fix that made the test real:** capture `qweight/qzeros/scales`
during `process_weights_after_loading`, before Marlin overwrites them.
96 layers stashed, and the counter proves the injection runs:

```
patch fired: {'calls': 202848, 'routed': 98208, 'no_cfg': 101424,
              'not_one_token': 3216}
```

98,208 real calls routed through our kernel — `q`, `k`, `v`, `o` and
`down` projections; `gate/up` skipped because they lose to Marlin.

## What it measured

| rounds | stock | ours | delta | ours wins | tokens |
|---|---|---|---|---|---|
| 6 | 462.9 ms | 448.2 | −3.2 % | 4/6 | identical |
| 30 | 462.3 | 443.2 | −4.1 % | 16/30 | identical |
| **100** | **498.7** | **488.5** | **−2.0 %** | **46/100** | identical |

**46 of 100 is a coin flip.** The median difference is ~2 %; the sign
test says we cannot tell it from noise. Within-arm spread was 13 % and
the card ran at 1755 MHz of 2100 at 79 °C.

## The conclusion, and it is a real one this time

Replacing **every** routable quantized linear — 98,208 calls per run —
with a kernel that is 1.16–1.20× faster than the production kernel
produces **≈2 %, unresolvable on this hardware**.

Two causes, both measured:

1. **The linears are not the bottleneck** for a 0.5B model at 32 generated
   tokens on this GPU: attention, the KV-cache path, sampling and the
   scheduler dominate.
2. **This box cannot resolve 2 %**: ±13 % run-to-run on a thermally
   throttled 4 GB laptop card.

So: the kernel-level margin is real and verified; the model-level
consequence is, at this scale, below both the physical share of the work
and the measurement floor.

## What would change that

- **A model where the linear layers dominate** — larger models (7B+),
  batched prefill, or graphs not already fused.
- **A stable measurement environment** — locked clocks, no throttling.
- **Coverage**: `gate/up` is 96 of 168 linears and currently routes to
  Marlin because we lose there; closing it is the largest single
  coverage win available.

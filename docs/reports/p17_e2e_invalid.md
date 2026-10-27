# P17 — every end-to-end number is invalid, and now we know exactly why

Two failures, either of which alone would invalidate the model-level
result.

## 1. The patch mostly never fired

vLLM does not use `AutoAWQLinearMethod` for this checkpoint. It logs
`Using MarlinLinearKernel for AutoAWQMarlinLinearMethod` — an automatic
upgrade that **repacks the AWQ weights into Marlin's internal layout**.
Patching `AutoAWQLinearMethod.apply` therefore counted **0 calls**: the
"ours" arm was the stock arm, which is exactly what a 26/50 coin flip
looks like.

Forcing the non-Marlin path is possible (`VLLM_BATCH_INVARIANT=1`), and
then the patch fires — 768 linear calls per 8-token generate, 43.8 %
routed to us.

## 2. The weights are not the checkpoint weights

With the patch firing, the run measured **−28.5 %, 30/30 rounds** — and
the token sequences differed. Divergence analysis: **first difference at
token index 1**, 1 of 48 tokens identical. That is not rounding, it is a
different function.

The cause is visible in one print:

| | checkpoint | in-model |
|---|---|---|
| `qweight` | 896 × 112 | 896 × **144** |
| `qzeros` | 7 × 112 | 7 × **144** |
| `scales` | 7 × **896** | 7 × **1152** |

vLLM **pads the output dimension** (896 → 1152) for tile alignment when
it prepares weights after loading. Our kernel assumed the checkpoint
layout, so it read padded tensors and produced garbage — faster, because
the work is the same but the answers were meaningless.

## What this invalidates

**All six end-to-end measurements** in this block (P6b, P8c, P9, P15,
P17): every one either compared stock with stock or read a layout that
does not match the tensor it was reading. The "flat" conclusion and the
"−28.5 %" conclusion are both void.

**What survives:** the isolated kernel measurements — cold regime,
exactness-gated, run directly against the checkpoint tensors, not
vLLM's processed ones. Those remain valid: 2.96× / 1.52× / 1.35× on
three of four shapes.

**The profile also survives** (P16): the quantized gemms are **51.6 %** of
device time, so a real 3× on them would be worth ~34 % end-to-end. That
is why the model-level question is still worth answering — it is not
dilution that keeps biting us, it is integration.

## What a valid end-to-end test requires

1. **Capture the pre-processed tensors**: stash each layer's checkpoint
   `qweight`/`qzeros`/`scales` at load time, before vLLM pads or
   repacks them, and drive our kernel from those.
2. **Or implement vLLM's layout**: pad-aware kernels that read the
   896→1152 tiled form exactly as `awq_gemm` does.
3. **Gate on output agreement, not just timing.** Every invalid number
   in this block would have been caught by a single `tokens identical`
   assertion at the top of the harness. That assertion existed and fired
   — it is the only reason the −28.5 % was not published.

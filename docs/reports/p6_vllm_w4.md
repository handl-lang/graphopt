# P6b — our quantized-linear op inside vLLM: integrated, not measurable

**Integration: done. Speed: inconclusive on this box.**

## What ran

Every `Linear` in vLLM's Llama was quantized per-group (W4-ish: group 128,
`scale = max|w|/7`), and the dequant path was toggled per round **inside
one engine**:

* **baseline** — `x @ (w_q * scale).T + bias` (materializes the float
  weight every forward),
* **ours** — `mlopt::grouped_quant_linear_ieee` (one Triton kernel,
  dequant in registers).

**24 linears** were driven by our op (QKV, O, gate/up, down × 6 layers).

## Result

| | torch dequant | our fused op |
|---|---|---|
| median | 154.0 ms | 160.0 ms |
| sd | **±83.2** | **±71.2** |
| ours wins | — | 7 / 20 rounds |
| tokens | — | **identical** |

**Token-identical**: our kernel reproduces the torch dequant path exactly
at the output. That is the integration result, and it is solid.

**±3.9% is not a result.** The within-arm sd is 46–54% of the mean —
bigger than the effect by an order of magnitude. Clocks read 1327 MHz
against a 2100 MHz ceiling (the 1500 lock did not hold) and the card sat
at 73 °C, i.e. thermally throttling.

## Why the end-to-end effect is small even if real

The isolated op wins **1.5–1.7× vs `torch.compile`** at d=64–256
(`p6_grouped_quant_op.md`). In a whole model that op is a fraction of the
24 quantized linears plus attention, embeddings, sampling and the
scheduler — so the end-to-end gain is a few percent at most, which is
below this laptop GPU's noise floor.

## What to do to get the number

1. **Re-lock clocks** (`sudo nvidia-smi -lgc 1500,1500`) and keep the card
   under 70 °C — every run above drifted after the lock was lost.
2. **Measure at the layer, not the model**: time the 24 linears alone
   inside the engine (per-layer pairing), where the 1.5–1.7× lives.
3. **Bigger card** so both arms can be resident and runs don't throttle.

The claim that survives today: **the op is callable from vLLM's linear
path, is exact, and is 1.5–1.7× faster than `torch.compile` in
isolation.** The end-to-end number is pending hardware that can measure
it.

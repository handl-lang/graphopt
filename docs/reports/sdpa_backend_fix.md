# sdpa backend fix — decode-shape math-backend pin in `torch_bridge`

Follow-up to `decode_step_fusion.md` §4 / follow-up #1: the delivered
`sdpa` binding let `F.scaled_dot_product_attention` pick its backend;
fp32 + `attn_mask` rules out flash/cuDNN, so every attention landed on
the mem-efficient fmha cutlass kernel — ~2.4–4.7× slower than the
math backend at decode shapes (kv=1500, q_len=1).  This change lands
the fix in the binding itself.

## Policy chosen

`packages/kit-torch/src/kit_torch/torch_bridge.py`:

* `_sdpa_math_backend(q, k)` returns `SDPBackend.MATH` **only** for the
  measured pathological regime:
  * dtype is fp32 or fp64 (fp16/bf16 keep the default pick — flash /
    mem-efficient are the right choice there);
  * `q.size(-2) <= _SDPA_MATH_Q_LEN_MAX` (= 8) — decode-like short
    query axis (covers q_len=1 decode and small speculative widths);
  * `k.size(-2) >= _SDPA_MATH_KV_LEN_MIN` (= 256) — long key axis.
* `_sdpa_torch` is the `sdpa` binding (`_CORE_TORCH_BINDINGS["sdpa"]`).
  The argN→kwarg remap is unchanged from the previous inline lambda;
  when the pick is `MATH` the call runs inside
  `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])`, otherwise it is
  a plain `F.scaled_dot_product_attention` call — zero overhead for
  non-decode shapes beyond one shape/dtype check per call.

Why shape-aware rather than pin-everything: mem-efficient genuinely
wins at prefill/long-context shapes (math materialises the L×S score
matrix — O(L·S) memory and slower GEMM-shaped work), and wins at
short-kv decode (self-attn kv≤~64: 0.029 vs 0.065 ms in the §4
microbenchmark).  The pin is deliberately narrow so encoder/prefill
and short-cache decode are untouched.

`F.scaled_dot_product_attention` itself takes no `backend=` kwarg on
torch 2.14 (verified signature); backend selection is via the
`sdpa_kernel` context manager (equivalently
`torch.backends.cuda.sdp_kernel`), which is what the probe used.

## Measured

GPU microbenchmark, fp32, RTX 2050 (same box as the profile):

| shape | old binding (default pick) | new binding | pick |
|---|---|---|---|
| decode cross-attn q=(1,8,1,64) kv=(1,8,1500,64) +mask | 0.415 ms | **0.175 ms** | MATH |
| decode self-attn q=(1,8,1,64) kv=(1,8,41,64) +mask | 0.055 ms | 0.056 ms | default (efficient) |
| prefill q=(1,8,512,64) kv=(1,8,1500,64) +mask | 1.928 ms | 1.924 ms | default |

Max|diff| vs the un-pinned call: 1.6e-7 (fp32 reassociation between
fused and unfused attention — same class as the verify gate).

End-to-end probe rerun (`whisper_decode_sdpa_backend.py 3 32 40`,
shared-GPU contention was heavy this run — eager measured 11.28
ms/tok vs the ~3.6–4.2 of the profile runs, so absolute loop numbers
are inflated; orderings and ids-match are what matter):

* `lower verified=True`; stock arm (now the pinned binding)
  max|diff| vs itself 0.0 — the pin is active in the delivered module;
* `math` arm step 5.08 ms, loop warm **5.88 ms/tok**,
  math+zero-copy-graph **4.51 ms/tok**, ids-match **32/32**;
* `heur` arm 8.06 ms step — same policy as the new binding, pure
  timing noise under contention;
* `cudnn` arm still fails on fp32 as before (unsupported dtype).

The ~0.8 ms/tok decode win measured in `decode_step_fusion.md` §4
transfers: the binding now applies the `heur` policy by construction
(q_len ≤ 8 & kv ≥ 256 & fp32/fp64 → MATH), so no `_IR_TO_TORCH`
rebinding is needed.

## Prefill / encoder regression check

Encoder self-attention and decoder prefill run at q_len ≫ 8 → the
pick returns `None` and the binding is a plain `F.sdpa` call
(bitwise-identical dispatch to before; measured 1.928 → 1.924 ms,
noise).  fp16/bf16 decode keeps flash/efficient.  The only behaviour
change is fp32/fp64 decode-like calls with kv ≥ 256.

## Correctness

`tests/test_sdpa_backend.py` (12 tests): boundary coverage of
`_sdpa_math_backend` (q_len 8/9, kv 255/256, fp16/bf16/fp64,
prefill), `_sdpa_torch` output vs `F.scaled_dot_product_attention`
reference on CPU for masked/unmasked decode and prefill shapes, the
argN→kwarg remap incl. the `None` filter, and the `_CORE_TORCH_BINDINGS`
table entry.

Also fixed two pre-existing failures in `tests/test_decode_laws.py`
unrelated to this change: `_v("s", ())` built `TensorType(((),))`,
which the newer `mlopt_core` `TensorType` validation (eqsat
`e18b7ef`) rejects; the tests only need a non-Const scale leaf, so
they now use `_v("s")` (0-dim Var).

## Follow-up assessed (not implemented): alias-aware CUDA-graph capture

`decode_step_fusion.md` §3 measured a probe-local zero-copy capture at
**−0.9 ms/tok** vs the shipped `CudaGraphRunner` (which clones all 26
inputs — 0.88 ms/step of D2D copies, ~36.8 MB of which is the constant
cross-attn cache).  The honest upstream version is an opt-in
"capture on caller-owned buffers / skip provably-unchanged inputs"
mode in `mlopt_cuda.runners`: an identity check on input object
identity (`xs[i] is captured_xs[i]`) per call, falling back to eager
on mismatch.  Effort: moderate — the runner's capture/replay wrapper
needs a per-input aliasing policy + fallback path, plus tests for the
identity-guard contract.  `runners.py` was left untouched this session
(other work in flight); the probe
(`whisper_decode_step_profile.py::_ZeroCopyGraphCapture`) is the
reference implementation.

## Reproduce

```sh
.venv/bin/python -m pytest tests/test_sdpa_backend.py -q
.venv/bin/python bench/drivers/probes/whisper_decode_sdpa_backend.py 3 32 40
```

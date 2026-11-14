# Kokoro fusion-site census — is there a pointwise-fusion prize for TTS?

Date: 2026-11-14.  Hardware: RTX 2050 4 GB (shared), torch
2.14.0+cu130, fp32, CUDA.  Probe:
`bench/drivers/probes/kokoro_fusion_probe.py` (extended from the
dead-agent scratch; census machinery imported from
`fusion_floor_probe.py`, fused-kernel path re-implemented — see
§5).  Follow-up to `fusion_floor_probe.md`, which found
`_fusion_sites == 0` on the whisper decode step and flagged Kokoro's
conv/resblock decoder as the unmeasured case.

**Verdict up front: unlike whisper, the Kokoro decoder arms DO have
a real fusion prize — and it is large.**  The complex (`decoder`)
and CustomSTFT (`decoder-nc`) arms carry **348 / 289 fusion sites**,
~450–500 elementwise ops in ~72–98 regions, and ~48 repeated 9-op
transcendental regions (`sin`/`pow`/`reciprocal`/`mul`/`add` — the
SineGen/phase chains).  Measured on real harvested tensors, fusing
those regions is **9.4–14.0× faster** than the spelled per-op
evaluation; extrapolated over the arm it bounds a prize of roughly
**35–45 ms/call on a ~117–126 ms delivered step** — essentially the
whole eager-vs-`torch.compile` gap (`torch.compile(wrap)` lands at
~71–72 ms on both arms).  The decoder wall is **not** conv-bound:
profiler attribution splits it ~58 ms conv / ~55 ms pointwise.  A
pointwise-fusion carrier is worth building **for the decoder arms
only** — `f0n` is negligible and `textenc` has zero sites.

## 1. Method

Same pipeline as `tts_kokoro_probe.py::_harvest` →
`Optimizer.search(max_iterations=1)` → delivered `res.term`
(DAG census via `mlopt_core.traversal.ops`, not `expand`).  Region
census = `fusion_floor_probe._region_census` semantics (weakly
connected components over `genkernel.ELEMENTWISE`, boundary-leaf
analysis); `_fusion_sites` = the gen-tier collector's own verdict
(`mlopt_discovery.play.sites`).  Search iters=1 — the term is
dominated by the model structure, not search depth; op histograms
match the iters=3 shape.

Per-region ceiling: spelled eval (one binding call per op — the
same dispatch shape the delivered plan tape pays) vs a
`torch.compile`d unrolled region body on the real harvested leaf
tensors, timed with `kokoro.timeit`.  Two probe bugs had to be
fixed first (both in scratch/probe code, no package changes):

* `fusion_floor_probe._region_census` crashes on multi-root regions
  (`len()` of an int count, line ~139) — Kokoro's DAG hits the
  corner whisper's did not.  `kokoro_fusion_probe.py` now carries a
  fixed local copy; upstream file untouched.
* `tts_kokoro_probe.load()` monkeypatches `torch.rand/zeros/…` for
  the device-kwarg export workaround; the wrappers call
  `ref.expand` on a real CUDA tensor, which inductor's
  FakeTensorMode rejects — **every `torch.compile` in the process
  died** until the originals were restored post-search.
* Leaf eval via `eval_term` retains every intermediate (~2.2 GB) —
  OOM on the 4 GB card for deep leaves.  Per-leaf fresh memo helped;
  for the deepest region, shapes were taken under
  `FakeTensorMode` and real `randn` tensors materialised (values
  don't affect kernel timing; diff still verified spelled-vs-fused).

## 2. Per-arm census

| arm | term nodes | elemwise ops | regions | `_fusion_sites` | notes |
|---|---|---|---|---|---|
| `decoder` (complex TorchSTFT) | 1026 | 500 | 72 | **348** | delivered 117.0 ms, eager-wrap 149.1 ms |
| `decoder-nc` (CustomSTFT) | 1037 | 456 | 98 | **289** | delivered 125.9 ms, eager-wrap 124.0 ms |
| `f0n` | 179 | 29 | 13 | 12 | negligible — mostly singletons |
| `textenc-p` | 41 | 0 | 0 | 0 | packed-LSTM `custom_op` leaves swallow the graph |
| `textenc` (unpaked) | — | — | — | — | walls at `pack_padded_sequence` export (known) |

Decoder region-signature histogram (identical structure on
`decoder-nc`, counts shifted):

```
decoder:    x33  9 ops {add3,mul3,pow1,reciprocal1,sin1}
           x15  9 ops {add3,mul4,reciprocal1,sin1}
           x11  1 op  {add}          x3  {div}   x2  {add13}
            x1  11 ops {add5,mul5,rsqrt1}   x1  {exp2,mul2,sin1}
            + small singletons/pairs — 500 ops in 72 regions total
decoder-nc: x25 + x11 of the same two 9-op signatures, x12 {mul,sin,
           square}, x12 {reciprocal} singletons, … 456 ops / 98 regions
```

The two 9-op signatures — **48 regions / ~432 ops on `decoder`,
36 regions / ~324 ops on `decoder-nc`** — are the prize.  They are
the repeated SineGen/phase computation (`sin`, `pow(.,2)`,
`reciprocal`, `mul`, `add` chains; the complex-STFT variant spells
the `exp(1j·φ)` as `exp/sin/mul(·,1j)` — fusion site pattern0:
`('mul', ('exp','X1'), ('exp', ('mul', ('sin','X2'), 1j)))`).

## 3. Measured ceiling (decoder, real harvested tensors)

Distinct region shapes deduped by op histogram; spelled vs
`torch.compile`d fused body, fp32 CUDA:

| region | ops | leaves | spelled | fused | speedup | max diff |
|---|---|---|---|---|---|---|
| 9-op {add3,mul3,pow,recip,sin} | 9 | 4 | 1671.9 µs | 176.8 µs | **9.46×** | 5.7e-6 |
| 9-op {add3,mul4,recip,sin} | 9 | 4 | 2465.3 µs | 176.9 µs | **13.94×** | 2.9e-6 |
| 11-op {add5,mul5,rsqrt} (IN-statistics) | 11 | 11 | 25.1 µs | 44.2 µs | 0.57× — **fusion loses** (small tensors, launch-bound) | 3.8e-6 |
| 5-op {exp2,mul2,sin1} complex phase | 5 | 2 | 118.0 µs | 117.4 µs | 1.01× — wash (already memory-bound) | 0.0 |
| 13-op {add13} residual chain | 13 | 14 | — | — | OOM'd in leaf eval (deep leaves); unmeasured | — |

`decoder-nc` re-measured the same three shapes: **9.43× / 13.95× /
0.60×** — statistically identical (spelled 1672.7/2462.4/25.1 µs,
fused 177.4/176.6/42.1 µs).

Reading: the prize is concentrated exactly where predicted — the
transcendental chains.  `sin`/`pow`/`reciprocal` eager kernels cost
~40–170 µs each on these tensor sizes (profiler: `sin_kernel_cuda`
~5.7 ms/call aggregate, `pow` ~3.7 ms/call); one inductor kernel
replaces nine passes **and** evaluates the transcendentals ~an
order of magnitude cheaper.  Small-tensor regions lose to fusion
dispatch overhead — a carrier needs a size threshold, same lesson
as the whisper singleton result.

## 4. Ceiling estimate

Per-call budget (`decoder`, delivered module, 10-rep profiler):

```
kernels/call ≈ 1100   cuda ≈ 116.5 ms
  conv       ≈ 58.3 ms   (implicit_gemm + cudnn nchw↔nhwc + bn)
  pointwise  ≈ 55.5 ms   (elementwise_kernel, sin, pow, add, …)
  gemm/memcpy/cat/other ≈ 3.7 ms
torch.compile(wrap) = 72.1 ms  (vs delivered 117.0, eager 149.1)
```

Extrapolation, stated as a bound rather than a promise: the 48 big
regions measure ~1.5–2.3 ms spelled-vs-fused each, which sums far
beyond the 55.5 ms pointwise bucket — the spelled figure includes
per-op Python dispatch the delivered tape also pays, so the honest
ceiling is the bucket itself minus what must remain: fused cost
≈ 48 × 0.177 ms ≈ 8.5 ms, leaving ~47 ms of unavoidable
pointwise/conv-adjacent work → **prize ≈ 35–45 ms/call**, i.e.
delivered ~117 → ~75–80 ms, consistent with `torch.compile`'s
72 ms.  `decoder-nc` is the same story (55.4 ms pointwise, compile
70.7 ms).

Caveats: (a) region-level fusion assumes member-internal values
aren't consumed outside the region — `_fusion_sites`' 348/289
sites are the collector-legal subset, but delivery must still
materialise shared intermediates (same caveat as the whisper
add-chain); (b) measurements are fp32 on a shared 4 GB card —
absolute µs will shift, the spelled:fused ratio is the durable
number; (c) `f0n`/`textenc` need no carrier.

## 5. Delivery-machinery notes (blockers found, none in packages/)

* `genkernel.gen_bindings` **cannot currently compile these
  regions**: `_fusion_pattern_sz` inlines Const *tensor* leaves
  into the spec and inductor's FakeTensorMode rejects them
  (`aten.expand.default(tensor(...))`).  The probe's
  `_compiled_region` works around it by lifting tensor Consts to
  positional args and unrolling the region to a flat op list —
  a carrier would need the same treatment (or a genkernel fix).
* `tts_kokoro_probe`'s `torch.*` creator monkeypatch must be
  reverted before any in-process `torch.compile` — it is a
  probe-local landmine, now handled in the probe.
* Region-census `len(int)` bug in `fusion_floor_probe.py` —
  fixed locally in `kokoro_fusion_probe.py`; upstream untouched
  per constraints.

## 6. Verdict

**Build the carrier — scoped to the decoder arms.**  TTS is *not*
purely conv-kernel-bound: the pointwise bucket is half the decoder
wall, and the repeated transcendental regions offer a measured
9–14× per-region speedup worth an estimated ~35–45 ms/call
(~30–38% of the delivered step).  `decoder` and `decoder-nc` are
equivalent targets; `f0n` (12 sites, 29 ops) and `textenc`
(0 sites under the packed-LSTM wrap) are not worth it.  Next step
would be a delivery path that mints these region patterns as
compiled kernels (fixing the Const-tensor inlining) and a
size threshold so small regions stay spelled.

# Kokoro per-W artifact build cost + caching options — measured

Date: 2026-11-14.  Follow-up to `kokoro_e2e_split.md` / `kokoro_production_bench.md`:
the exact-W split delivers ~110 ms vs ~152 ms eager (1.36×) but each new `W`
pays a ~10–14 s artifact build.  This report measures *where* the build time
goes, whether one searched term transfers across `W`, what persistence options
exist, and what that means for interactive TTS.  Probe:
`bench/drivers/probes/kokoro_artifact_probe.py` (new, uncommitted scratch —
modes `reuse` / `persist` / `dyn`).  fp32, RTX 2050 4 GB, bench utterance
(T=42, S=127, search-W=128), `KPS_NOVERIFY=1`, `max_iterations=3` unless noted.

## 1. Build-cost breakdown (decoder arm, per W)

| stage | wall | notes |
|---|---|---|
| inputs (glue + eager F0Ntrain) | <0.5 s | M1 ~20 ms + glue ~1 ms + F0Ntrain ~7 ms per W; the "~3 s harvest" in the split report was first-call warmup, not steady build cost |
| `optz.search` | **5.1–6.9 s** | `iters=1` → 5.5 s, `iters=3` → 5.1–6.9 s — iteration count is *not* the driver; export/e-graph/lift dominates |
| `optz.lower(verify=False)` | **0.03–0.05 s** | essentially free |
| `RegionFusionRunner(output_rtol=2e-3)` | **3.3–5.0 s** | 3.3 s with warm inductor cache, 4.0–5.0 s repeat/cold, **8.5 s** with `dynamic=True` |
| **total per-W build** | **~9–12 s** | consistent with the 10–14 s in the split report (which included warmup) |

The build is dominated by **search (~55%) and regionfuse (~35%)**; lower is a
rounding error.  Two implications:

* Cutting `max_iterations` does not help — the cost is front-loaded in
  export + saturation setup, not iteration count.
* RegionFuse has a floor of ~3 s even with a warm kernel cache (census +
  per-region screening forwards + whole-module gate are Python/forwards
  bound, not inductor bound — see §3).

## 2. Transferability verdict: the searched term is W-BAKED

Hypothesis tested: search once at canonical W, re-lower per actual W.

* `optz.lower(res, inp_W)` reuses `result.term` verbatim — the extracted
  term carries **literal shape constants** from the W=128 trace (e.g. a
  reshape attr `[1, 1, 76800]` = 128·600 output samples).
* Re-lowering the W=128 `SearchResult` at W=144/160/176 and calling the
  module: **`RuntimeError: shape '[1, 1, 76800]' is invalid for input of
  size 86400/96000/105600`** — every non-search-W fails.  Plain lower and
  regionfuse lower both fail identically (regionfuse declines fast,
  `applied=False`, then the unfused module hits the same reshape).
* At the search W itself everything is fine: lower 0.04 s,
  `max|diff| = 0.0` vs eager wrap (bitwise, deterministic arm), delivered
  118 ms; regionfuse `applied=True` (57 regions / 408 ops / 7 kernels),
  `max|diff| = 3.3e-4`, delivered+fuse **83.5 ms**.

**Verdict: no transfer.**  Each exact W needs its own `search` (~5–7 s);
the per-W marginal cost after one search is *not* "lower only" — it is a
full new search.  (Term surgery — cloning the hash-consed `Op` DAG and
rewriting every baked shape attr — is theoretically possible since only
shape ints differ, but the decoder bakes dozens of derived shapes
(conv/STFT/frame lengths); re-deriving them is re-implementing shape
inference.  Not viable as a cheap path; noted for completeness.)

### The dynamic-shape escape route also fails

Forced `torch.export.export(..., dynamic_shapes=...)` (probe-local patch of
`torch.export.export`, the seam `TorchSource.to_ir` calls):

* Marking the W-varying dims (`asr[1,512,W]`, `F0/N[1,2W]`) with `Dim`
  → `UserError: Constraints violated … your code specialized it to be a
  constant`.  The decoder's Python uses the shapes in constant-required
  contexts, so dynamo refuses dynamic marking.
* `Dim.AUTO` exports "successfully" but re-guards the dims static
  (`Guard failed: 1 + (-1 + f0.size()[1]) // 2 == 128` at W=144) — the
  specialization is semantic, not incidental.  The searched term stays
  W-baked (`76800` reshape again).

So **shape-polymorphic delivery is blocked at the export boundary**, not
the search/lower boundary — a decoder-code or lift-layer change would be
needed (out of scope; packages are read-only here).

## 3. Artifact persistence options

| option | result |
|---|---|
| `pickle` the `IRModule` | **FAILS** — `PicklingError: Can't pickle <lambda> … mlopt_torch.torch_bindings` (lambda bindings in the lowered module; same for the regionfused module) |
| `torch.export.export(mod, inp)` | **WORKS** for both plain and regionfused `IRModule` — exported program is **bitwise** vs eager (`max|diff| = 0.0`, seeded) — but the dynamo trace costs **~133 s** |
| `torch.export.save` → `.pt2` | writes a **428 MB** artifact per W; `torch.export.load` round-trips and runs (roundtrip diff 6.8e-2 = the model's own unseeded RNG floor — bitwise not separately gated, but export was 0.0 vs eager) |
| `.pt2` across W | **FAILS** — static guard `xs_0.size()[2] == 128`; still per-W |
| inductor FX cache (`/tmp/torchinductor_luis`, 834 MB) | persists across processes; regionfuse warm-repeat measured **3.3 s vs 4.0–5.0 s** cold — a real but modest ~1–1.5 s saving; the ~3 s regionfuse floor is census/screen/gate, not compile |

**Persistence verdict:** disk-caching *works* technically (`.pt2`,
428 MB/W) but is per-W, enormous, and ~133 s to produce — strictly worse
than keeping the in-process module or just rebuilding.  The only cheap
reuse is the inductor kernel cache already on by default.  A production
cache would be an in-memory `dict[W → module]` (LRU) — exactly the
`cache` dict `kokoro_split_probe.e2e` already uses.

## 4. Interactive-latency verdict

Steady-state numbers (same runs): split e2e **~110 ms** vs eager
`forward_with_tokens` **~152 ms** → the delivered decoder saves
**~42 ms/call** (decoder arm alone: 83.5 ms fused vs ~123 ms eager).

Per-W economics:

* Build ≈ **9–12 s**; saving ≈ **42 ms/call** → payback ≈ **215–285 calls
  at the same exact W**.
* `S` takes ~600 distinct values over the observed range and tracks text
  length — for interactive TTS with varied utterances, per-W hit rate is
  low; **the build never amortizes** on one-off traffic.
* `torch.compile(forward_with_tokens)` (~100 ms) is *not* an escape
  either: `dynamic=False` re-specializes per shape (T and W both vary),
  its first-call compile is tens of seconds, and its parity was never
  gated (seeded diff 1.15e-1 > stochastic floor).

**Honest recommendation:**

* **Interactive/varied TTS: stay eager (152 ms).**  Exact-W artifacts are
  a batch/precompute play — audiobook-style rendering, batch APIs, or
  services where the same (voice, length) recurs hundreds of times.
* **Repetitive-traffic middle ground:** serve eager on a cache miss and
  build the artifact *asynchronously* keyed by `(voice, S)` — S is known
  ~20 ms into the call (after M1).  Fixed prompts, repeated
  syntheses, and template-driven traffic then converge to ~110 ms;
  one-off traffic never pays.  This is the only interactive design that
  doesn't lose.
* **Offline precompute:** if the S distribution is concentrated (short
  sentences cluster), pre-build the top-K W artifacts at deploy time —
  each costs ~10 s of GPU, so K=50 is ~8 min one-time.  A histogram of
  `S` over representative traffic would size K (not measured here —
  follow-up if this path is pursued).
* **Bucket mode stays off the table** for quality-sensitive serving:
  global receptive field → even W=S+1 is ~5× the stochastic floor
  (`kokoro_e2e_split.md` §S-boundary).  Only acceptable if the
  spec-distance hit is consciously tolerated.

## Reproduce

```sh
# build breakdown + transferability (search once @W=128, reuse @144/160/176)
KPS_NOVERIFY=1 .venv/bin/python bench/drivers/probes/kokoro_artifact_probe.py reuse 3 128,144,160,176
# persistence: pickle / torch.export / .pt2 round-trip
KPS_NOVERIFY=1 .venv/bin/python bench/drivers/probes/kokoro_artifact_probe.py persist 128 3
# dynamic-shape export attempt (KAP_DYN_AUTO=1 for Dim.AUTO)
KAP_DYN_AUTO=1 KPS_NOVERIFY=1 .venv/bin/python bench/drivers/probes/kokoro_artifact_probe.py dyn 3
```

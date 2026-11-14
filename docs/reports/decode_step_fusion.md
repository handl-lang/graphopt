# Decode-step dispatch attack — where the 5.5 ms goes and what moved it

Follow-up to `whisper_decode_loop.md` (delivered `IRModule` 5.5–6.0
ms/tok vs eager ~3.6, `torch.compile` ~3.2).  Same box, same probe
machinery (`bench/drivers/probes/whisper_decode_loop.py`), new probes:

* `bench/drivers/probes/whisper_decode_step_profile.py` — plan-tape
  introspection, torch.profiler kernel breakdown, CUDA-graph
  decomposition (copies vs replay), and a **zero-copy graph** variant.
* `bench/drivers/probes/whisper_decode_sdpa_backend.py` — per-backend
  `sdpa` arms on the delivered module + greedy-loop ids-match.

GPU is shared; absolute numbers swing ±30 % run to run (eager warm
measured 3.62–4.24 ms/tok across runs).  Orderings below were
consistent within each run.

## 1. Where the time goes

**Plan tape.**  The delivered `IRModule` evaluates a 261-step tape
(418 slots, 12 distinct ops): `linear`×49, `concat`×36, `reshape`×36,
`transpose`×36, `dropout`×24 (identity binding — no kernel),
`layer_norm`×19, `add`×18, `mul`×12, `sdpa`×12, `slice`×12, `gelu`×6,
`pack`×1.  Reshape/transpose/slice are views; the kernel-bearing ops
are the GEMMs, 36 `cat`s (the per-layer K/V cache appends), 19 layer
norms, 12 sdpa, and the pointwise residue — roughly **~380 kernel
launches per call** in eager eval, **~194 inside graph replay**.

**Graph decomposition** (the shipped `CudaGraphRunner` arm):

| piece | ms/call |
|---|---|
| 26 input `copy_`s into static buffers | **0.880** |
| `graph.replay()` alone | **5.23–5.26** |
| copies + replay | 6.12–6.14 |

The copies are ~38 MB of D2D traffic per step — 12 cross-attn K/V
tensors of (1,8,1500,64) fp32 = 36.8 MB that are **constant across the
whole decode** plus 12 small self-attn buffers.  Replay itself costs
the same as the plain module (~5.3 ms): **the decode step is
GPU-kernel-bound, not CPU-dispatch-bound.**  CUDA graphs remove launch
overhead (~0.1–0.2 ms here) but cannot remove kernel execution time.

**Kernel mix inside replay** (torch.profiler, totals/10 reps):

* `fmha_cutlassF_f32_aligned_64x64` (mem-efficient attention) —
  12 launches/call, ~1.9 ms/call of device time — the single largest
  block;
* GEMM family (`gemv2T`/`gemvx`/`ampere_sgemm`) ~0.9 ms;
* `CatArrayBatchedCopy` + `aten::cat` ~0.5 ms (the 36 cache cats);
* layer_norm ~0.06 ms; pointwise residue small.

## 2. Attempt A — compositional / larger-grain carrier delivery: NOT AVAILABLE

Checked the carrier path (`mlopt_carriers` machinery,
`om_lower`/`scan_lower`/`omd_lower` batched executors, `om_fused`
level-batched kernels):

* The delivered root is `pack` (13 outputs: logits + 12 K/V slices).
  Batched carrier executors only engage when a **carrier-apply term is
  the extracted root** (`_carrier_upgrade` / sink `accepts` routing).
  Carrier members *inside* a mixed term evaluate through the plan
  tape's `om_apply` binding — which is the elementwise `f[2] / f[1]`
  over the `om_compose` tree, i.e. **more** kernels, not fewer.
* No whole-chain pointwise-fusion carrier or law exists.  The law
  bank's fusion surface is the attention/scan families (om, scan,
  trace, decay, decode_geom); the only elementwise folds are
  `silu_*`/`rms_norm_*`.  `concat`/`layer_norm`/`add`/`mul` chains have
  no fused carrier.
* A carrier `TORCH_BINDINGS` export cannot override `sdpa` anyway:
  `OpTable.full()` seats carrier bindings into the ambient
  `_IR_TO_TORCH` via `setdefault`, and the core `sdpa` binding is
  already present — carrier exports only fill *missing* keys.

Verdict: a compositional delivery needs orchestrator-level routing
(carrier executors for non-root subterms) — documented as future work,
not attempted here.

## 3. Attempt B — CUDA graph done right: copies were the whole overhead

The shipped `_GraphReplayModule` clones **all 26** inputs into static
buffers and copies every call — 0.88 ms/step, of which ~0.85 ms is the
36.8 MB of *constant* cross-attn cache that never changes.

Probe-local `_ZeroCopyGraphCapture` (in
`whisper_decode_step_profile.py`): captures the graph **on the loop's
own persistent K/V buffers** — the 24 past tensors are the same
objects every step and replay reads their live contents, so only
`hs`/`mask` are copied (2 tiny D2D copies).  An identity check
(`xs[2:] is captured_tail`, ~24 pointer compares) guards the aliasing
contract; mismatch falls back to the inner module.

| arm | step ms | loop warm ms/tok | last-8 | ids vs eager |
|---|---|---|---|---|
| delivered `IRModule` | 5.37 | 5.85 | 5.46 | 32/32 |
| shipped graph (26 clones+copies) | 6.08–6.52 | 6.16 | 6.15 | 32/32 |
| **zero-copy graph** | **5.23** | **5.27** | **5.27** | **32/32** |

Replay-vs-module max|diff| = 0.0 (bitwise — same kernels, same
order).  **−0.9 ms/tok vs the shipped graph arm**, but still above
eager: launch overhead was never the wall.

This is a probe-level device.  The honest upstream version is an
alias-aware capture option in `mlopt_cuda.runners` (capture on
caller-owned buffers / skip copies for inputs proven unchanged) —
`runners.py` was not edited here per task constraints; the pattern and
the measured win are documented for follow-up.

## 4. Attempt C — the real bug: `sdpa` backend selection

Microbenchmark at the actual decode shapes (fp32, RTX 2050):

| shape | efficient | math | manual math |
|---|---|---|---|
| self-attn q=(1,8,1,64) kv=(1,8,41,64) +mask | **0.029** | 0.065 | 0.037 |
| cross-attn q=(1,8,1,64) kv=(1,8,1500,64) | **0.408** | 0.168 | **0.086** |

The stock `sdpa` binding lets `F.scaled_dot_product_attention` pick —
fp32 + attn_mask rules out flash/cuDNN, so **every** attention lands on
the mem-efficient fmha cutlass kernel, which is ~2.4–4.7× *slower*
than math at the cross-attention shape (1500-key cache reads).  ×6
layers ≈ 1.4–2.4 ms/step — the dominant single cost in the delivered
module.  Eager HF (`attn_implementation="eager"`) uses unfused math
attention and is *faster* here; `torch.compile` fuses the same math
into Triton online-softmax kernels.

Arms on the delivered module (`_IR_TO_TORCH["sdpa"]` rebound — the
plan tape resolves bindings per call, so the patch reaches the
built module; probe-level stand-in for a binding fix):

| arm | step ms | loop warm ms/tok | ids |
|---|---|---|---|
| stock (efficient) | 5.32 | ~5.8 | 32/32 |
| **math everywhere** | **4.51** | **4.64** | 32/32 |
| heuristic (math iff kv_len ≥ 256) | 4.52 | — | — |
| cuDNN | n/a | — | fp32 unsupported |
| **math + zero-copy graph** | — | **4.34** | **32/32** |

Logits max|diff| vs stock delivery: 6–8e-6 (fp32 reassociation
differences between fused and unfused attention — same class as the
existing verify gate).  The heuristic arm ≈ math-everywhere: at kv=41
the efficient-vs-math delta (~0.036 ms × 6) is inside noise, so the
simple rule "math when `k.shape[-2] >= 256`" captures the whole win
without a per-site table.

**Best honest number: 4.34 ms/tok** (math sdpa + zero-copy CUDA
graph), ids-match 32/32 — vs eager ~3.6 and compile ~3.2 on the same
runs.  Still short of the bar; the residual is kernel *count*, not
launches (see §5).

## 5. Why the floor is what it is

* **Launches are ~free under replay.**  Zero-copy replay = 5.23 ms ≈
  plain module 5.37 — the plan tape's Python dispatch was never the
  bottleneck at this size; GPU kernel execution is.
* **The remaining ~380→~194 kernels are mostly irreducible without
  fusion.**  GEMMs (49 linears incl. the 106 MB lm_head read) and the
  36 cache `cat`s are already single kernels each; layer_norm is one
  fused kernel.  What torch.compile has and we don't: Triton-fused
  pointwise+norm chains and fused online-softmax attention — it runs
  ~316 launches but each does several ops' work, which is where its
  ~1.2–1.4 ms edge over our best arm comes from.
* **Weight traffic floor.**  The decoder+head weight set is ~120 MB
  fp32; at this GPU's effective bandwidth the pure weight-read floor
  is ~1 ms.  Eager 3.6 / compile 3.2 / ours 4.3 all sit above it —
  the spread is overhead structure, not arithmetic.

### Concrete follow-ups (not implemented — out of scope/blast radius)

1. **`torch_bridge` sdpa binding heuristic** (the single biggest win,
   ~0.8 ms/step measured): pick `SDPBackend.MATH` when
   `k.shape[-2] >= ~256` and dtype is fp32 (efficient fmha is
   pathologically slow at long-kv/tiny-q fp32; keep efficient for
   short-kv where it wins).  Alternatively expose a backend attr on
   the `sdpa` op so laws/callers choose.  `torch_bridge.py` was
   off-limits this round — the probe demonstrates the win by rebinding
   `_IR_TO_TORCH["sdpa"]`.
2. **Alias-aware CUDA-graph capture in `mlopt_cuda.runners`**: an
   opt-in "capture on caller buffers / skip provably-unchanged inputs"
   mode.  Probe shows −0.9 ms/tok vs the clone-everything capture for
   decode-loop-shaped call sites (constant cross-attn cache).
3. **Carrier-executor routing for non-root subterms** (or a
   pointwise-fusion carrier): the only path to compile-class numbers —
   needs orchestrator work, then re-measure.
4. Optional: fold `embed_step`+mask construction into the captured
   region (~0.05 ms, marginal).

## Reproduce

```sh
uv sync
.venv/bin/pip install transformers   # scratch dep
.venv/bin/python bench/drivers/probes/whisper_decode_loop.py loop 3 32 40
.venv/bin/python bench/drivers/probes/whisper_decode_step_profile.py 3 32 40
.venv/bin/python bench/drivers/probes/whisper_decode_sdpa_backend.py 3 32 40
```

Files added: the two probes above + this report.  No package files
changed; `ruff check`/`ruff format` clean on both probes (`ty` scopes
to `packages/` only).

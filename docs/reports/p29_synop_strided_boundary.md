# P29 — T-016 residual: why the verified fused prim loses, and the strided-codegen fix

Follow-up to `p26_synop_board_delivery.md` / `p28_fold_gate_residuals.md`.
The question: a `synop_*` member that verifies perfectly still loses
wall-clock to the spelled body — why, concretely, and is there an
honest lever left?  Machine-readable rows:
`zoo_chain_sweep_t016_strided.json` (the 8 T-026 rows re-swept
post-fix) and `zoo_chain_sweep_t016_strided_all.json` (all 46 fp64
losing rows).

## 1. Diagnosis — per-site arms, fp64, zoo `d64 t32 b8`, CUDA

For every shipped prim that sites in a losing row's term, the three
candidate bodies were timed paired-interleaved on the case's real
CUDA values (spelled = eager `_eval_spec` spec-walk — the referee's
spelled arm; triton = the `triton_bindings` fused kernel, i.e. what
`hybrid_sink` ships; compiled = `torch.compile`d spelled body — the
declined-tier fallback).  µs, medians.

**GatedDeltaRule** — `synop_bb8c54c201` = `add(mul,mul)`, 3 ops,
output `(8,64,64)` = 32,768 elems, broadcast operands
`(8,1,1)/(8,64,1)/(8,1,64)`:

| arm | µs | vs spelled |
|---|---|---|
| spelled eager | 23–27 | 1.00 |
| triton (pre-fix) | ~45 | **0.48–0.52×** |
| compiled spelled | 60–78 | 0.39–0.45× |

Same row, deeper folds: `synop_5e58f26875` (6 ops) spelled ~38–44 vs
triton ~72–80 (**~0.5×**); `synop2_35bbd03301` (12 ops, 13 metas)
spelled ~68–73 vs triton ~143–150 (**~0.48×**).  The attr-metavar
prims (`synop_4c5296b3ba` &c. — `unsqueeze`/`select` bodies outside
the elementwise grammar) get **no** Triton binding: their delivered
body is the compiled spec-walk at ~0.55–0.65× spelled.

**GRUFusedGate** — `synop_bb8c54c201`, *uniform* `(4,64)` = 256
elems: spelled 21–27 vs triton 25–37 → **0.60–0.85×**.

**ChunkedSSD** — three prims, output `(8,4,4,16)` = 2,048 elems,
broadcast operands: spelled 32–44 vs triton 44–59 → **0.56–0.74×**.

Every corpus prim site is **256–32k output elements**.

## 2. The three loss mechanisms, ranked

**(a) Host-side broadcast materialization — dominant, now fixed.**
The old `triton_bindings` wrapper ran `broadcast_tensors` +
`.contiguous()` per operand: every broadcast operand became a
full-size temporary — O(M·n) extra traffic *plus* a copy kernel
launch each, inside the "fused" path whose entire point is avoiding
exactly that.  Measured on `add(mul,mul)` broadcast sites, pre-fix:
**0.53× @32k, 0.76× @0.52M, 0.77× @2.1M fp64 — it never paid at any
size.**  This is why "verified fused" lost: the fold removed N−1
kernel launches and re-added M materialization launches.

**(b) Launch/host-bound regime — the corpus boundary.**  Even with
zero materialization, one Triton call costs ~25–45µs of host time
(broadcast_shapes, ometa lookup, empty, JIT dispatch, launch) while
the spelled side pays ~7–9µs *per op*.  Crossover measured on the
3-op fold, paired:

| shape regime | fp64 spelled→triton | fp32 |
|---|---|---|
| uniform 32k elems | 0.93× | 0.85× |
| uniform 0.52M | **1.72×** | 1.67× |
| uniform 8.4M | **1.83×** | 1.82× |
| broadcast 32k (pre-fix) | 0.53× | — |
| broadcast 0.52M (post-fix) | **1.10×** | 0.77× |
| broadcast 2.1M (post-fix) | **1.46×** | 0.83× |

So elementwise fusion pays iff output ≳ 0.5M elements (fp64;
fp32 broadcast still loses through ~2M — the in-kernel offset math
is ALU-costly against cheap fp32 ops).  **Every zoo prim site sits
2–4 orders of magnitude below that floor: on this corpus no
elementwise fold can pay, fused or not.**  The T-019/T-026 measured
gate (`profitable`, delivered-binding vs spelled, paired timing) is
exactly the right enforcement — post-fix re-sweep confirms it still
refuses every prim fold (`prim_ops: []` on all rows) and the
~0.95 median is the documented binding-dispatch floor, not fold
loss.

**(c) The compiled-spelled fallback strictly loses.**  `torch.compile`
over the recursive `_eval_spec` spec-walk graph-breaks and pays
dynamo guard overhead the eager spelled side never pays: **0.39–0.74×
at every site measured** (p28 §2 measured the same on the
attr-metavar site: 55.8µs compiled vs 34.9µs eager vs 39.7µs
uncompiled spec-walk).  The 5 attr-metavar prim specs — whose bodies
carry `unsqueeze`/`select`/views outside the Triton grammar — route
to this fallback, so they *cannot* pay through `hybrid_sink` at any
shape short of where inductor finally wins.  Honest boundary: those
folds need unrolled torch-ops codegen (compile a straight-line body,
not a spec walk) or ambient-op delivery; per-op fused bodies are the
wrong instrument for them.

## 3. Fix landed — strided-load Triton codegen

`genkernel.triton_bindings` now emits **two kernels per handler** in
the same generated module:

* `_gk` — the original flat kernel, kept for the common case (every
  operand already broadcast-shape + contiguous); zero regression on
  the uniform path;
* `_gk_s` — a strided kernel for broadcast / non-contiguous /
  expanded operands.  The wrapper packs per-operand *effective*
  strides into a cached `ometa` int64 tensor
  (`_triton_ometa`, bounded 256-entry cache shared across kernels —
  layout is handler-independent) and the kernel reconstructs each
  element's source offset in-register: `coord_d = (off // ostride_d)
  % osize_d`, `offset_i = Σ coord_d · eff_stride_i[d]`, broadcast dims
  contribute stride 0.  **No host materialization at all** — the
  broadcast+contiguous copies are gone.

Also added `_TRITON_LOCALS`: a spec metavar colliding with a kernel
local (`off`, `mask`, `out`, `n`, `BLOCK`, the new `ometa_ptr`/
`off64`/`__o<i>`/`__sz`/`__stb`/`__cd`) now declines at codegen —
previously such a spec generated a kernel whose value load would
silently overwrite the mask/index locals (verify would catch it,
but only after the fact).

Verified correct on CUDA: rank-1/scalar broadcast, 0-dim operands,
non-contiguous (transpose), stride-0 `expand`ed tensors, uniform,
and empty tensors — all `allclose` vs the spelled body.  Tests added
in `tests/test_discovery_genkernel.py` (`TestTritonTier`).

## 4. Re-sweep

`zoo_chain_sweep_t016_strided.json` — the 8 T-026 rows, fp64, 3
isolated rounds, post-fix:

| case | t026 | strided | prim ops |
|---|---|---|---|
| GatedDeltaRule | 0.956 | 0.963 | [] |
| DecoupledMLA | 0.955 | 0.986 | [] |
| DilatedBlockAttention | 0.852 | 0.849 | [] |
| GRUFusedGate | 0.932 | 0.873→0.917 rerun | [] |
| ChunkedSSD | 1.020 | 1.010 | [] |
| RWKVMixBlock | 0.988 | 0.986 | [] |
| HighwayGate | 0.953 | 0.958 | [] |
| RMSNormPerHead | 0.876 | 0.881 | [] |

`claims`, `prim_ops` and extracted `cost` are **identical** to the
T-026 rows on every case — the delivered terms did not change (the
gate still refuses every prim fold, correctly: the sites are
launch-bound).  GRUFusedGate's 0.932→0.873 was timing noise on an
identical delivered module (re-run 0.917, spread overlapping).
`zoo_chain_sweep_t016_strided_all.json` re-sweeps all 46 fp64 losing
rows for claim-level flips (gen claims share `triton_bindings`).

## 4b. Measurement-hygiene fix — compile-worker drain

A post-sweep anomaly forced one more fix: a full 46-row re-run let
`_fold_cost_gate` keep **35** prim folds on GatedDeltaRule and
extraction delivered `synop_2b8cfd1c90` at **0.668×** — a measured
"winner" that is a known loser in isolation (~63µs kernel vs ~42µs
spelled).  Reproduced deterministically: the *same* handler set
measured `kept=35` when run right after `_zoo_case`'s own
`profitable` claim sweep, `kept=0` in a quiet process, `kept=0`
again after `torch._inductor.async_compile.shutdown_compile_workers()`.

Mechanism: the case build compiles dozens of `torch.compile`'d
fallbacks and Triton kernels; in-flight inductor compile workers are
still draining when the fold gate times.  Wall-clock inflates
*disproportionately* on the spelled arm — it issues N torch calls +
a recursive spec walk per invocation where the fused kernel issues
one launch — so losers read as winners.  The fix is one drain at the
top of `genkernel.profitable` (covers both the claim sweep and the
fold gate, since both measure through it):

```python
from torch._inductor.async_compile import shutdown_compile_workers
shutdown_compile_workers()   # waits for pending work, then pools respawn on demand
```

After the drain, each handler's own verify-compile runs
synchronously inside its untimed warmup and subsequent timing
windows are quiet.  Pinned by
`test_profitable_drains_compile_workers`.

Post-drain re-sweep: `zoo_chain_sweep_t016_drain.json` — the same 8
prim rows, fp64, 3 isolated rounds:

| case | strided (pre-drain) | drain | prim ops |
|---|---|---|---|
| GatedDeltaRule | 0.963 | 0.966 | [] |
| DecoupledMLA | 0.986 | 0.994 | [] |
| DilatedBlockAttention | 0.849 | 0.863 | [] |
| GRUFusedGate | 0.873→0.917 | 0.927 | [] |
| ChunkedSSD | 1.010 | 0.837 | [] |
| RWKVMixBlock | 0.986 | 0.988 | [] |
| HighwayGate | 0.958 | 0.958 | [] |
| RMSNormPerHead | 0.881 | 0.814 | [] |

`prim_ops` is empty on every row — the contaminated 35-prim
GatedDeltaRule board is gone and the delivered term is the plain
spelled module again (0.966, matching the T-026 0.956 within noise).
A first post-drain attempt *did* still admit a borderline prim
(`synop_889d740672`, row 0.676) — per-site referee noise at the
margin remains a known T-026 residual (`synop2_35bbd03301` has been
measured 0.64× and 2.2× across processes); the drain removes the
*systematic* inflation, not the irreducible run-to-run variance.
What the drain guarantees is that a borderline admission cannot be
*manufactured* by compile backlog — and even when one slips through,
the row speedup is measured end-to-end, so it can only cost the row
a few percent, never fabricate a win.  ChunkedSSD's gen-claim set
also flipped between runs (`[]` → `gen_10,gen_14` → `gen_12,gen_13,
gen_17` across sweeps) with row speedups 0.84–1.01 — same margin
noise on the claim side, orthogonal to the prim question.

Note: `--only` takes comma-separated names; a typo'd name makes the
child emit `[]` and `zoo_chain._measure_isolated` crashes on
`json.loads(payload)[0]` — driver wart, not fixed here.

## 5. Verdict — the honest boundary

**The fused path cannot pay on this corpus, and now we know exactly
why:** every shipped-prim site is ≤32k output elements in a
launch/host-bound regime where one fused call (~25–45µs host) costs
more than the ~7–9µs-per-op launches it replaces.  The cost model's
"1 op vs N" billing is irrelevant at this scale — wall-clock is set
by launches, not op count.  What changed:

* the *avoidable* loss — O(M·n) broadcast materialization inside the
  fused path — is fixed; broadcast sites now pay at the same ~0.5M-
  element threshold as uniform ones (fp64), instead of never;
* the *unavoidable* boundary is documented: elementwise `synop*`
  folds pay iff output ≳0.5M elems (fp64) — no zoo row qualifies;
* non-elementwise prims route to a compiled spec-walk that loses
  ~2× everywhere — they need unrolled-op codegen or ambient
  delivery, a different mechanism entirely;
* enforcement is already correct: the T-019/T-026 measured gate
  refuses every losing fold, and extraction never sees them.

A measured win exists but not here: post-fix, a broadcast elementwise
fold at ≥0.5M fp64 elements is 1.1–1.5× faster than spelled *and*
~1.5–1.9× faster than the pre-fix fused path.  If the corpus ever
grows wide-tensor sites (the fp32 loser set's larger modules, or
`--d/--t/--b` up), this codegen is what pays.

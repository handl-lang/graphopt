# Scoping the whole-model ~1× parity gap — where the ms go

Probe: `bench/drivers/probes/wholemodel_gap_probe.py` (this doc's
instrument — a measurement driver, not a fix).  Dev box: RTX 2050 /
i5-12500H, fp64 delivery (the verify dtype) unless marked `32`.
Run 2026-10-20; the box was shared, so treat the third significant
figure as noise — the *ordering and decompositions* are what hold.

**Dependency note.**  `mlopt_discovery` was mid-refactor when this ran
(the game layer's move to `gamekit` was staged but not committed), so
the probe vendors its two fixtures verbatim (`zoo._ChunkedRetention`,
`play._time_module`) instead of importing them.  The FLA/Triton and
torchscale rows are **not re-measured** — `fla-core`/`torchscale` are
scratch deps not installed in this venv; the external reference numbers
are carried from `project/retros/retnet-external-baselines.md`
(2026-10-14, same box) and marked *retro*.

## The negative being scoped

Delivered modules run at ~parity with the incumbent spelling while
FLA-style fused kernels sit ~7–11× ahead
([retnet-external-baselines](../project/retros/retnet-external-baselines.md)
row; `docs/results.md` §5).  The claim being tested: the gap is a
**codegen/lowering** gap — per-leaf generic torch ops, fp64 GEMMs on a
1:64-fp64-rate consumer GPU, materialized intermediates — not a
semantics gap.

## Case A — retnet chunked carrier (T=2048, K=8, C=256, B=2, d=16)

The extracted carrier (`concat` root, 218 distinct ops, cost
85.7 M FLOPs) is lowered two ways: `carrier` = the generic
`TorchSink` plan-tape `IRModule` (per-leaf dispatch); `fused` = the
executor-routed `ChunkedDecayModule` — the real delivery path
(`optimize._lower_extracted` → `sink.executors["om_decayed_chunked"]`).
`+graph` = CUDA-graph capture (the executor's own
`capture_cuda_graph`, or a manual `torch.cuda.CUDAGraph` for the
generic module).  `kern us` = per-call CUDA-kernel-busy time
(profiler); `dispatch us` = wall − busy (CPU-side: plan-tape dispatch,
Python loop code, launch gaps).

| arm | peak MiB | med ms | vs eager | kern µs | n_k | dispatch µs | gemm µs | pw µs |
|---|---|---|---|---|---|---|---|---|
| spelled-eager | 230.1 | 21.739 | 1.00× | 21719 | 13 | 20 | 8884 | 12271 |
| carrier(gen)64 | 90.1 | 3.209 | 6.77× | 2980 | 205 | 229 | 2200 | 728 |
| carrier(gen)32 | 79.6 | 2.917 | 7.45× | 1130 | 205 | **1787** | 756 | 345 |
| fused(exec)64 | 87.5 | 2.679 | 8.11× | 2604 | 32 | 75 | 2059 | 460 |
| fused(exec)32 | 78.6 | 0.951 | 22.9× | 445 | 32 | **506** | 194 | 209 |
| paper-eqn64 (hand loop) | 87.3 | 2.171 | 10.0× | 1952 | 61 | 219 | 1441 | 491 |
| carrier64+graph(man) | 69.3 | 2.964 | 7.33× | 2948 | 206 | 16 | 2185 | 702 |
| carrier32+graph(man) | 69.3 | **1.142** | 19.0× | 1101 | 206 | 41 | 739 | 328 |
| fused64+graph | 78.9 | 2.078 | 10.5× | 2081 | 33 | ~0 | 1557 | 436 |
| fused32+graph | 87.8 | **0.454** | 47.9× | 449 | 33 | ~5 | 197 | 205 |
| fla-chunk (fp32, Triton) | ~10 | 0.28 | ~78× | — | — | — | — | — |

*fla-chunk row is the retro's number (fp32 Triton, different numerics
by construction — speed baseline only).  T=1024 shows the same
ordering; there the fp64 carrier is already **50% dispatch-bound**
(2912 µs wall, 1466 µs busy), and `fused64+graph` lands 0.679 ms —
matching the retro's 0.670 ms.*

### Attribution ladder — carrier64 → fla-chunk (~11.5×)

Each step is a measured arm delta at T=2048:

| step | ms | Δ ms | share of gap | what it is |
|---|---|---|---|---|
| carrier64 (generic tape) | 3.209 | — | — | 205 kernels, per-leaf eval |
| → carrier64+graph | 2.964 | −0.245 | ~8% | **per-leaf dispatch** removed (229→16 µs CPU) |
| → fused64+graph | 2.078 | −0.886 | ~30% | **schedule batching**: 206→33 kernels, dedup'd shared operand evals, const-folded decay generators, batched pair products |
| → fused32+graph | 0.454 | −1.624 | ~55% | **fp64→fp32** (RTX 2050 fp64 is ~1:64; gemm 1557→197 µs) |
| → fla-chunk | 0.28 | −0.17 | ~6% | **kernel-level fusion** — the materialized `s⊙d` score tensor, index-select gathers, per-pair segment sums a fused Triton kernel never writes |

Reads:

- **Dispatch/Python overhead is real but second-order at fp64** —
  ~0.25–0.6 ms of a 3.2 ms carrier wall (~8–19%).  It *flips to
  dominant in fp32*: the fp32 generic carrier spends **61%** of its
  2.92 ms wall dispatching (1787 µs CPU vs 1130 µs busy), and a manual
  graph capture lands it at **1.142 ms** — the single biggest cheap
  recovery measured.
- **Python-loop overhead is small**: `paper-eqn`'s explicit `for` over
  8 chunks costs ~219 µs of dispatch on 61 kernels — within noise of
  the executor's flat schedule.  The loop is not where the money is.
- **The fp64 GEMM penalty is the largest single component** (~1.6 ms
  at T=2048, ~55% of the carrier→FLA gap): a deployment-dtype issue,
  not a kernel-quality issue.  (fp32 arms are speed baselines — no
  fp32-tolerance verify was run.)
- **Materialized intermediates** show up as the pointwise+copy kernel
  column (~0.5 ms at fp64, ~0.25 ms at fp32) and peak-memory rows
  (~70–90 MiB vs FLA's ~10 MiB): decay masks, `s⊙d`, gathers, stacked
  pairs — everything a fused scan kernel keeps in registers/smem.
- **Structural nuance**: the executor's pairwise-batched cross terms
  do *more* GEMM work than the paper's recurrent-state loop
  (gemm 1557 vs 1441 µs — all `K(K−1)/2` pair products vs `K` state
  updates) and still wins on wall under replay, because launch count
  dominates on this box.  Closing the last ~1.6× likely needs *both*
  kernel fusion and a different extracted schedule — a search-level
  choice the FLOP cost model can't currently express.

## Case B — chainfuse MLP ((256,512), d=512, fp64)

`x@A@B@C` + a 6-op pointwise tail, searched under `DEFAULT_RULES`
(`max_iterations=8`).  The extracted term (root `applyd`, 13 distinct
ops) delivers through executor routing — the `scan` spec accepts the
`applyd` root but its batched plan declines (`is_batched=False`), so
the module runs the embedded generic tape; identical to a plain
`IRModule` (2.131 vs 2.129 ms — confirmed directly).

| arm | peak MiB | med ms | vs eager | kern µs | n_k | dispatch µs | gemm µs | pw µs |
|---|---|---|---|---|---|---|---|---|
| spelled-eager | 139.9 | 4.926 | 1.00× | 4905 | 14 | 22 | 4168 | 737 |
| spelled-eager32 | 137.4 | 0.282 | 17.5× | 261 | 17 | 21 | 154 | 105 |
| spelled-compiled | 136.9 | 4.733 | 1.04× | 4664 | 4 | 70 | 4167 | 0* |
| generic64 (IRModule) | 145.9 | 2.129 | 2.31× | 2104 | 11 | 26 | 1390 | 714 |
| delivered64 (routed) | 145.9 | 2.131 | 2.31× | 2103 | 11 | 28 | 1390 | 714 |
| delivered32 | 140.4 | 0.172 | 28.6× | 148 | 12 | 25 | 50 | 97 |
| delivered+compile | 135.9 | 1.960 | 2.51× | 1883 | 2 | 77 | 1390 | 0* |
| handfold-eager | 139.9 | 2.149 | 2.29× | 2128 | 12 | 22 | 1390 | 738 |
| handfold+compile | 135.9 | 1.944 | 2.53× | 1886 | 2 | 58 | 1389 | 0* |
| handfold32+compile | 135.4 | 0.112 | 44.0× | 55 | 3 | 57 | 48 | 0* |
| delivered64+graph (runner) | 145.9 | 2.129 | 2.31× | 2103 | 11 | 26 | 1389 | 714 |
| generic64+graph (manual) | 134.9 | 2.134 | 2.31× | 2132 | 12 | 3 | 1392 | 714 |

*fused into Inductor's generated kernel — profiled as "other".

Reads:

- **The structural win is real and already delivered**: the
  weights-first fold cuts GEMM time 4168→1390 µs (3 fp64 GEMMs → 1),
  wall 4.93→2.13 ms.  The delivered module *equals* the hand-folded
  reference op-for-op (2.129 vs 2.149 ms).
- **The residual lowering gap is the pointwise tail**: ~10 per-leaf
  pointwise kernels materializing intermediates (714 µs) vs Inductor's
  single fused kernel (~493 µs incl. epilogue) → 2.129 vs
  1.944 ms (−0.19 ms, ~9%).  `delivered+compile` recovers it: the
  IRModule tape dynamo-traces cleanly, landing at the hand-folded
  floor.
- **Dispatch is invisible at fp64** (26 µs, hidden behind ms-scale
  GEMMs) — the runner-captured delivery is *identical* to eager
  delivery (2.129 ≈ 2.129).  At fp32 the tail residual shrinks to
  ~60 µs absolute (0.172 vs 0.112 ms delivered-vs-compiled-folded).
- fp64→fp32 is again the dominant term: 4.93→0.28 ms eager,
  2.13→0.17 ms delivered.

## Does the shipped CUDA-graph lane recover dispatch? — measured

`CudaGraphRunner` (`mlopt_cuda`) only claims modules exposing
`capture_cuda_graph` — the batched carrier executors.  Measured:

| module | runner result | effect |
|---|---|---|
| generic `IRModule` (carrier, retnet; delivered, chainfuse) | `stats["cuda_graph"]=False` — **degrades quietly, no recovery** | — |
| `ChunkedDecayModule` (executor-routed, retnet) | captures, `cuda_graph=True` | fused64: 2.679→**2.078 ms**; fused32: 0.951→**0.454 ms** |
| generic `IRModule`, *manual* `torch.cuda.CUDAGraph` | capturable — the tape is a fixed op sequence (replay is bit-identical by construction; the underlying carrier verified at 9.8e-9) | carrier64: 3.209→2.964; **carrier32: 2.917→1.142 ms** |
| delivered chainfuse via runner | captures (executor wrapper) | **no effect** — 2.129 ≈ 2.129; dispatch already hidden |

So: **the cheap lane exists and works where a carrier executor claims
the term** (worth ~0.5–0.6 ms at fp64, ~0.5 ms at fp32 on the
carrier).  It does **not** reach generic lowerings — a one-method gap:
the tape *is* capturable (manual capture works, verified-identical
output), and a graph-capable generic runner would recover the full
~1.8 ms dispatch share the fp32 carrier pays.

## What closing this takes (honest answer)

Ordered by measured payoff per unit of machinery:

1. **fp32 delivery where the verify contract allows** — removes ~55%
   of the gap on this GPU class.  Not kernel work: it's a
   dtype/tolerance-contract decision (the fp64 gate is the verify
   default, not a semantic requirement).
2. **Graph-capture generic `IRModule`s** — trivial runner extension
   (the probe demonstrates capturability); buys ~8% at fp64, **~60%
   at fp32** on per-leaf lowerings.
3. **Pointwise/tail fusion at lowering** — `TorchCompileRunner`
   already recovers it where dynamo can trace (chainfuse delivered:
   −9%); the minted-kernel gen tier does it without dynamo
   (chainfuse claim: 86 µs vs compile 253 µs at fp32 in the
   gen-kernels retro).
4. **The last ~1.6× to FLA** (0.454 vs 0.28 ms fp32) is genuinely
   kernel-level: fused scan kernels, no materialized `s⊙d`, plus
   possibly a recurrent-state schedule choice at extraction time.
   That's the kernel-engineer board —
   `project/notes/kernel-engineer-board.md` — i.e. a *new game on the
   same machinery*, not a fix inside this one.

Bottom line: ~7× of the measured ~11.5× carrier→FLA gap is reachable
with **existing machinery + a dtype contract** (fused32+graph =
0.454 ms vs carrier64 = 3.209 ms); the residual ~1.6× is the kernel
board's problem.  On the MLP chain the delivered module is already at
the hand-folded floor — the structural win landed; only the tail
fusion is leaving ~9% (fp64) / ~35% (fp32, absolute ~60 µs) on the
table vs Inductor.

## Caveats

- fp32 arms are speed baselines; no fp32-tolerance verify was run —
  the shipped verify gate is fp64.
- FLA rows are the 2026-10-14 retro's (same box); not re-measured
  (scratch deps absent).
- `bench/drivers/probes/retnet_baseline_probe.py` currently has a
  stale self-import (`from ... import om_decayed_kway_deliver_probe
  as P`) from the `lab/`→`bench/` dissolution — it does not run as
  committed.  Pre-existing; flagged, not fixed.
- `mlopt_discovery` was mid-refactor during measurement; the probe
  vendors its fixtures.  Re-run via upstream `P._saturate` once the
  gamekit move lands — numbers should reproduce within noise.

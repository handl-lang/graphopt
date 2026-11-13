# T-034 — where the assembled GatedDeltaRule module loses (module-level costing)

**Date:** 2026-11-14 · **Scope:** the residual left by
`t032_034_frontier_delivery.md` (assembled-module delivery loses to
torch.compile, 0.33×) after `t037_resweep_frontier.md` showed the
`CudaGraphRunner` arm wins (graph_vs_compile 2.08 fp64 / 1.43 fp32).
Question: where exactly does the *non-graph* assembled module lose —
launch overhead, kernel count, a slow op, or cost-model mispricing?

**Short answer: none of the four hypotheses as stated.**  The module
loses on **host-side dispatch**: the delivered `IRModule` plan tape
re-issues ~65 kernel launches through a Python slot interpreter whose
per-call host cost (~600–850 µs enqueue) dwarfs the ~226 µs of GPU
work — while `torch.compile` pays ~31 launches and ~zero Python, and
`CudaGraphRunner` pays neither.  GPU device time of the delivered
module is *equal* to eager's (226 vs 228 µs).  The graph arm already
is the delivery answer for this gap — and it was never measuring the
seqm kernel anyway (see §3: the priced seqm claim is structurally
undeliverable through the claim path).

## 1. Live re-measurement (this session, RTX 2050 4 GB, idle box)

`zoo:GatedDeltaRule`, `zoo(64, 32, 8)`, `play.gen_probe` reps=50,
serial, one process per dtype:

| dtype | eager µs | compile µs | gen µs | speedup | vs_compile | graph µs | graph speedup | graph vs compile |
|---|---|---|---|---|---|---|---|---|
| fp32 | ~391–439 | ~175–225 | ~562–759 | 0.64–0.77× | 0.29–0.33× | 114.8 | 3.40× | 1.53× |
| fp64 | ~382–406 | ~425–429 | ~399–579 | 0.64–0.96× | ~1.0–3.6×* | ~182–207 | ~2.0× | ~2.1–2.3× |

\* fp64 rows vary with the per-process mint (claim sets are
nondeterministic — documented T-037 caveat); the quiet-box fp64 run
above had `gen_handlers` **empty** (every minted claim declined by
`profitable`) and still read speedup 0.956 / graph 2.10×.

These agree with the committed T-037 sweep rows (fp32 0.642 / graph
3.29× / 1.43×; fp64 0.702 / graph 1.95× / 2.08×).

## 2. Decomposition — the loss is host dispatch, not the GPU

`torch.profiler` (CPU+CUDA) on the three arms, fp32, same verified
module, same input (50 iters):

| arm | kernels/iter | device µs/iter | async wall µs | synced wall µs |
|---|---|---|---|---|
| base eager (`nn.Module`) | 86 | 228.4 | ~592 | ~439 |
| `torch.compile` | 31 | 213.1 | ~236 | ~225 |
| delivered `IRModule` | 65 | 226.1 | ~854 | ~759 |

- **Device time is a wash**: delivered 226 µs ≈ eager 228 µs.  The
  module launches *fewer* kernels than eager (65 vs 86) — the rewrite
  bank + synop folds did reduce dispatch count — but pays it all back
  (and more) on the host.
- **The gap is pure enqueue cost.**  `cudaLaunchKernel` CPU self-time:
  eager 137 µs, delivered 104 µs (fewer launches, as expected) — yet
  delivered wall is ~300–400 µs *higher*.  The difference is the
  `IRModule` plan-tape interpreter (`torch_bridge._EvalPlan.run_xs`,
  `packages/mlopt-torch/src/mlopt_torch/torch_bridge.py:2287-2358`):
  per call it allocates the slot vector, re-reads every param through
  `param_map.get`, **re-mints every `Const` leaf to a fresh CPU
  tensor** (`const_fills`/`_mint_const`, `:2390`), resolves each op
  name through `bindings.get`, then runs ~69 tape steps of
  `f(v[a0], v[a1], **kw)` dispatch — most bindings are `<lambda>`
  wrappers adding a Python frame per aten call.  Eager `nn.Module`
  calls the same aten ops with none of that indirection.
- **Launch count is not the binding constraint** — 65 vs 86 launches
  and the module still loses; what compile buys is fusion (31
  kernels), not merely fewer launches.
- **No slow op**: top device kernels are the same GEMMs/elementwise
  kernels on all three arms; `_gk_s` (a gen-tier Triton kernel) is
  ~49 µs/3 launches and present only when the mint happens to deliver
  a `gen_*` claim that run.

So the frontier "uncaptured headroom" on this row is exactly the
host-dispatch share — the same share `device_share` counts and the
same share `CudaGraphRunner` already removes (replay = device time +
~fixed overhead: 114.8 µs ≈ the ~226 µs device budget amortised over
the captured graph's better overlap, plus no per-call Python).

## 3. The bigger finding: the priced seqm claim never ships

T-032 credited `scan_8` as "kept by `profitable`, Triton bound,
delivered+verified".  The last clause is wrong at module granularity:

- The intake IR has **zero `applyd` nodes** (`TorchSource.to_ir`
  unrolls the recurrence into `select`/`matmul`/`mul`/`add` chains —
  op census: `applyd: 0`).
- `_seqm_scan_sites` (`play.py:1176`) mines those chains and mints a
  handler whose `"pattern"` is the spec tuple
  `("applyd", ("aff_diag", a, b), "H", {"kind": "seqm", …})`
  (`play.py:1165-1170`, via `genkernel.scan_handlers:2907`).
- `profitable` prices it honestly *as a claim*: kernel vs the
  registered fold denotation — fp32 `scan_8` read 352.9→171.1 µs
  (keep vote), `TRITON_REFEREE_LOG` outcome `bound` (tier `scan`).
- **But the claim can never fire.**  `MetaArena._claim`
  (`meta_arena.py:966`) hard-declines unless
  `arena._concrete_occurs(h["pattern"])` — `_concrete_match`
  (`meta_arena.py:408`) requires `term.op == pattern[0]`, i.e. an
  actual `applyd` op in the raw term.  `_claim_tags`
  (`meta_arena.py:1478-1494`) applies the same test, so `scan_*`
  never appears in `claim_tags` (observed: tags were `gen_*`/`synop_*`
  only, in every run).  `handlers_fragment` contributes *handle*
  cells, not rewrite laws, and `DEFAULT`'s 132 rules contain no scan
  law — nothing else can introduce the op.
- Delivered-term census (3 separate processes): heads are ambient ops
  + `synop*`/`synop2*` folds only — **no `applyd`, no `scan_*_k`, no
  `gen_*` heads** in the runs inspected (mint-dependent; `gen_*`
  elementwise claims ARE claimable and do ship when kept).

The same deadness applies to *all* scan-tier claims on raw-IR cases:
`_scan_sites` (`play.py:567`) mints `("applyd", …, {"kind":
"unrolled"})` patterns from a throwaway saturated e-graph whose
`applyd` members exist nowhere in the program term — the `"leaf_env"`
docstring says it outright: "no raw-IR match can ever rebind it",
and `_concrete_occurs` is exactly that raw-IR match.  The seq/seqm
collectors mint from the raw chain but still emit the `applyd`-headed
*denotation* spec — a pattern describing the folded form, not the
spelled site.

**Consequence:** `profitable`'s kept-set, `TRITON_REFEREE_LOG`,
`gen_handlers`, `supported_ops`, and `hybrid_sink`'s binding table
all carry claims that are unreachable by construction.  The referee
time is honest claim-level data (kernel vs fold), but nothing downstream
can route it.  T-037's graph-arm numbers remain valid — they measure
`CudaGraphRunner` on whatever module extraction actually produced —
but the narrative "the same verified module containing the seqm
kernel" is incorrect for this path: the delivered module is the
generic tape (+ occasional synop/gen claims), and the graph win is
pure host-dispatch elimination.

## 4. Does the deliver path offer/route the graph arm?

- `optimize()`/`lower()` **already accept `runner=`**
  (`mlopt_orchestrator.optimize.lower`, `.venv/.../optimize.py:2262,
  2373`); `CudaGraphRunner` is a shipped runner
  (`mlopt_cuda`), composable via `ChainedRunner`, and
  `runner.delivers_compiled` already feeds search-time pricing
  (`Monolithic.run`, `optimize.py:2466-2471`).
- The play deliver hooks do **not** use it: `_torch_deliver`
  (`play.py:390-424`) lowers + verifies only — no runner, no timing.
  `gen_probe` measures the graph arm additively (`_probe_graph_arm`,
  `play.py:1855-1904`) and the frontier carries `graph_*` columns
  (T-037) — measurement is first-class, routing is not.
- `gen_probe`'s headline `speedup`/`vs_compiled` still quote the
  non-runner module — correct as "the lowering number" per its
  docstring, but it means every frontier `speedup` row on a
  launch-bound workload reads a loss the shipped runner would erase.

## 5. Costing-bug assessment and proposed minimal fixes

The hypothesised "profitable prices claims without module context"
is real but secondary.  The defects, in order of size:

1. **Unroutable claims are minted, priced, bound, and reported as
   kept.**  Minimal honest fix: an offer-gate claimability check —
   drop (or mark) handlers whose `h["pattern"]` cannot
   `_concrete_occurs` in the case term before `profitable` spends
   referee time.  `_fold_cost_gate` (`play.py:1496-1500`) already
   articulates the principle for prim folds ("no concrete site →
   cannot fire → keeps its seat, costs nothing") — for *claims* the
   seat is not free: it buys a `profitable` window, a
   `TRITON_REFEREE_LOG` row, and a `gen_handlers`/`supported`
   entry.  **Not implemented**: `test_claim_cost_gate_filters_
   measurable_tiers` (`tests/test_discovery_genkernel.py:281-388`)
   pins `scan_0` (an applyd pattern absent from the term) reaching
   `profitable` and staying in `gen_handlers` — the pinned contract
   is "scan claims are measurable", and changing it also shrinks the
   `PROFITABLE_LOG`/`TRITON_REFEREE_LOG` audit surfaces the
   flip-rate/binding-audit probes read.  This wants a deliberate
   re-scope (either the pin updates, or dead claims get a distinct
   `unclaimable` marker rather than silent dropping).
2. **The real missing piece is a rewrite route, not a price.**  The
   seqm collector already *proves* the fold (identical deindexed
   skeletons step-to-step, index == chronological position); what is
   absent is an e-graph rule `unrolled chain → applyd(aff_diag…, H)`
   (or a claim variant that binds the chain site directly) so
   extraction can select the carrier op and `deliverable`/the sink
   can ship `scan_N_k`.  Until that exists, no amount of module-level
   costing changes what ships.
3. **Module-level costing exists and is honest** — `gen_probe`
   already reports `speedup`/`vs_compiled`/`gen_graph_us` per module.
   The gap is that the *deliver* hooks don't consume the runner arm:
   proposal — `_torch_deliver` (or a `--deliver-runner` flag) should
   attempt `CudaGraphRunner` on CUDA inputs and report/ship the
   wrapped module when `stats["cuda_graph"]` captures, mirroring
   `_probe_graph_arm`'s decline-honest posture.  This is a product
   decision (capture constrains input addressing), not a bug fix —
   flagged for the agenda, not implemented.

## 6. Bottom line for T-034

- The assembled module loses **on host dispatch** (plan-tape
  interpreter + per-call const/param/binding resolution + ~65
  launches), not on kernel time, launch count, a slow op, or a
  mispriced delivered claim.  GPU time ≈ eager; wall ≈ 2–3× eager.
- `CudaGraphRunner` is the correct delivery answer and is already
  shipped in `optimize(runner=…)`; the play/`--deliver` path simply
  never routes it.  The frontier's `graph_*` columns (T-037) are the
  right surface.
- Correcting T-032/T-037's implied claim: the seqm `_gsm` kernel is
  **not** in any delivered module on this path — its claim is
  structurally unclaimable (`applyd` pattern vs unrolled term).  The
  honest residual is twofold: (a) route the runner arm in deliver;
  (b) give seq/seqm claims a real rewrite route or mark them
  unclaimable at mint.

## Caveats

- Single-process `gen_probe` reps=50 rows; mint nondeterminism means
  claim sets differ run to run (documented).  Directionally consistent
  with the committed 2-round sweep medians.
- One fp64 first-run read `baseline_us=3825` under post-profiling GPU
  load; re-run on the idle box gave 382 µs, matching the committed
  sweep — treat sub-500 µs fp64 eager numbers on this box as
  load-sensitive.
- Profiler device times are per-iter medians over 50 captured
  iterations; CPU self-times are profiler-attributed and omit some
  interpreter frames, so the ~300–400 µs host gap is a lower bound on
  interpreter cost, not an exact per-line accounting.

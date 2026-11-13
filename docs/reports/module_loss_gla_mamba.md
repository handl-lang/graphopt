# Module-loss decomposition — GatedLinearAttention & MambaSelectiveScan

**Date:** 2026-11-15 · **Builds on:** `resweep_scan_route.md` (the
chain→`scan_<i>_k` route delivers+verifies on all five seq/seqm zoo
workloads but GLA/Mamba still lose at module level), method identical
to `t034_module_costing.md` (torch.profiler device-vs-host split,
per-op kernel table, `CudaGraphRunner` arm).

**Short answer: the loss is (c) — real device time — plus a host
tape-interpreter term that scales with the *size of the extracted
term*, and the scan route is only part of the story.**  On both
losers the delivered module's GPU time is ~2× eager's, the graph arm
captures cleanly and removes the host share entirely (1.7× eager),
and the remaining device loss decomposes into: multiplied minted
elementwise folds (`_gk_s` Triton launches, one per folded site),
the seq wrapper's broadcast materialisation, and — structurally —
the fact that these two modules return **per-step read-outs**, so the
route can only fold the *final* state while every intermediate state
stays spelled.  The paying workloads (DeltaRuleBlock, GatedDeltaRule)
return only the final state, so their seqm kernel replaces the whole
recurrence.

## 1. Method

Same delivery path as `play.gen_probe`, instrumented:

```python
case = play._zoo_case(wl, dtype="fp32")          # zoo(64,32,8)
board = play._gen_board(case); best = play._probe_play(board)
term = board.deliverable(best)                    # incl. _route_scan_chains
sink = gk.hybrid_sink(ex["gen_handlers"], compile_kernels=True)
module = sink.lower(IR(root=term, ...), params=ex["leaves"]).cuda()
```

then `torch.profiler(activities=[CPU, CUDA])` over 30 iters per arm,
`_time_module`-style synced wall timing (reps=50), and
`CudaGraphRunner().apply(module, xin, stats)` for the graph arm.
Mint is nondeterministic — absolute µs vary run to run (the resweep
quoted GLA gen_us 6219/7140 across dtypes/rounds; this session's mint
delivered a smaller claim set, gen_us 3208).  The *structure* of the
decomposition is stable across mints; per-kernel attribution below is
from one consistent run per workload.

## 2. Headline numbers (fp32, this session's mints)

| workload | eager wall µs | delivered wall µs | speedup | delivered device µs/iter | kernels/iter | graph wall µs | graph vs eager |
|---|---|---|---|---|---|---|---|
| DeltaRuleBlock | 266.7 | 213.7 | 1.25× | 85.3 | 29 | 77.6 | 3.44× |
| GatedDeltaRule | 385.2 | 309.9 | 1.24× | 122.9 | 47 | 53.8 | 7.16× |
| **GatedLinearAttention** | 1861.5 | **3208.5** | **0.58×** | **2010.3** | 313 | 1089.0 | 1.71× |
| **MambaSelectiveScan** | 1814.9 | **3502.1** | **0.52×** | **1321.2** | 204 | 1062.7 | 1.71× |

(Resweep medians over 2 rounds were GLA 6219.6 / Mamba 4148.1 fp32 —
same direction, different mint.)  Wall − device ≈ host share: paying
workloads ~90–190 µs host; losers ~1.2 ms (GLA) / ~2.2 ms (Mamba).

## 3. Per-workload decomposition

### GatedLinearAttention — delivered term census

`scan_64_k` ×1, `gen_1_k` ×**240 nodes**, `einsum` ×32, `mul` ×32,
`zeros` ×32, `add` ×16, `stack` ×1, plus view ops (`select` ×1520,
`unsqueeze` ×1491, `linear` ×1523, `sigmoid` ×497 — mostly duplicated
subtrees the tape memoises).  Binding calls per forward (dedup'd):
`gen_1_k` **30**, `einsum` 32, `select` 125, `unsqueeze` 96,
`scan_64_k` 1, `linear` 4.

Device top (µs/iter):

| kernel | n/iter | device µs |
|---|---|---|
| `_gk_s` (minted elementwise Triton, strided path) | 29 | **451.2** |
| `aten::contiguous/clone/copy_` | 3 | 146.8 **each (~440 total)** |
| `_gs` (seq scan kernel, `scan_64_k`) | 1 | 144.3 |
| `aten::add` | 31 | 120.8 |
| `aten::bmm` (einsum read-outs) | 32 | 93.6 |
| `aten::mul` | 33 | 91.0 |
| `aten::linear` (the 4 real GEMMs) | 4 | 35.4 |

Term structure (verified by walking `board.deliverable(best)`): the
32 `einsum`s are the per-step read-outs `o_t = q_tᵀ S_t`; `S_31` —
the *final* state — is produced directly by `scan_64_k`
(`einsum(scan_64_k(...), select(q, index=31))`), while `S_0…S_30`
are spelled step-by-step, with `gen_1_k` folded over the
`diag(g_t)·S + k_t v_tᵀ` polynomial at ~30 of the 32 sites.  The scan
recomputes the whole 32-step recurrence inside `_gs` to produce a
value needed exactly once.

The three ~147 µs `contiguous/copy` kernels are `_seq_args`
(`genkernel.py:2372-2410`): `torch.broadcast_tensors` over the three
seq operands `(8,32,64,1) × (8,32,64,1) × (8,32,1,64)` materialises
**3 × (8,32,64,64) = 3 × 4 MB** contiguous `(T, N)` row tensors —
the docstring's "price of generality over strides".  ~440 µs of pure
copy per call, larger than the `_gs` kernel itself.

### MambaSelectiveScan — delivered term census

`scan_64_k` ×1, `gen_11_k` ×47, `gen_13_k` ×23, `gen_9_k` ×5,
`gen_7_k` ×4, `gen_6_k` ×1, one `synop_*` ×4, `einsum` ×32,
`zeros` ×32, `stack` ×1; view-op counts similar to GLA
(`select` ×1520, `unsqueeze` ×1491, `linear` ×1026, `exp`/`neg`/
`softplus` ×~497).  Binding calls/forward: `gen_11_k` **21**,
`gen_13_k` 5, `einsum` 32, `select` 125, `scan_64_k` 1.

Device top (µs/iter):

| kernel | n/iter | device µs |
|---|---|---|
| `_gk_s` | 28 | **895.4** (~32 µs/launch) |
| `aten::bmm` (read-outs) | 32 | 70.1 |
| `aten::linear` | 3 | 24.3 |
| `aten::mul` | 7 | 20.0 |
| copies (`contiguous`) | 3 | 13.8 |
| `_gs` (`scan_64_k`) | 1 | **13.5** |

Here the scan kernel itself is cheap (state is only `(8,64,4)` =
2048 elements → 2 blocks); the broadcast copies are small too.  The
loss is almost entirely the 28 `_gk_s` launches — the minted
`gen_*_k` elementwise polynomial folds (patterns are 7–22-metavar
`add(mul(…))` trees over `(8,64,4)/(8,64,1)/(8,1,4)` broadcast
operands) applied once per folded recurrence site — plus ~2.2 ms of
tape-interpreter host time.

### The paying pair, for contrast

| | DeltaRuleBlock | GatedDeltaRule |
|---|---|---|
| delivered ops | 5 (`linear, scan_4_k, sigmoid, unsqueeze, zeros`) | 8 (+`exp, neg, softplus`) |
| scan kernel | `_gsm` seqm, 4.3 µs | `_gsm` seqm, 4.4 µs |
| device µs/iter | 85.3 (eager 158.5) | 122.9 (eager 227.6) |
| host share | ~128 µs | ~187 µs |
| graph replay device | 30.2 µs | 42.0 µs |

Their `forward` returns **only the final state `s`** — the seqm
`_gsm` kernel *is* the whole recurrence, the delivered term is a
handful of ops, and there is no spelled residual to decorate.

## 4. Where the loss lives — answering the four hypotheses

- **(a) Host tape-interpreter overhead — yes, but not the whole
  story, and the graph arm does fix it.**  `CudaGraphRunner`
  captured successfully on both losers (`stats["cuda_graph"]` true;
  no decline) and replay wall ≈ replay device (GLA 1089 vs 1062 µs;
  Mamba 1063 vs 1046 µs) — the ~1.2–2.2 ms of Python dispatch
  (tens of thousands of tape steps over the extracted term's
  duplicated view/gen nodes; same mechanism as T-034's
  `torch_bridge._EvalPlan.run_xs`, now at ~10–50× the node count)
  is gone.  What remains under the graph is still a loss to
  `torch.compile` (resweep: 0.57×/0.11×) because **device** time is
  also worse.
- **(b) A slow non-scan op — partially.**  The `_gk_s` minted
  elementwise kernels are the biggest single device line on both
  losers (451 / 895 µs), but they are not "a" op — they are ~28–29
  launches of the *same generated-kernel family* applied at every
  folded site.
- **(c) The scan kernel itself — mostly no.**  `_gs` costs 144 µs
  (GLA) / 13.5 µs (Mamba).  For GLA the *wrapper* around it —
  `_seq_args`'s 3×4 MB broadcast materialisation — costs ~440 µs,
  3× the kernel.  Neither is the dominant term on Mamba.
- **(d) Capture failing — no.**  Capture succeeds on all four
  measured workloads; replay is honest (graph_max_abs ≈ eager
  deltas).

The residual device loss is therefore: **multiplied minted folds +
scan-wrapper materialisation + the spelled recurrence that could not
be folded** — and the host loss is **tape-interpreter cost proportional
to extracted-term size**, which only the graph arm removes.

## 5. The structural reason the route pays on two and not two

`_route_scan_chains` (`meta_arena.py:1083`) rewrites the matched
chain to `scan_<i>_k`, whose `_gs`/`_gsm` kernels return **only the
final carrier state** (`genkernel.py:2245-2256` stores `acc` once;
`_seq_binding` returns `out.view(state)`).  That is a complete fold
iff the program needs only the final state:

- `DeltaRuleBlock`/`GatedDeltaRule` return `s` → whole recurrence
  folds → tiny term → pays.
- `GatedLinearAttention`/`MambaSelectiveScan` return
  `torch.stack(outs)` — every step's read-out `o_t = q_tᵀ S_t` /
  `einsum("bdn,bn->bd", h_t, c_t)` needs **every intermediate
  `S_t`**.  The route folds the chain feeding the *last* read-out
  (final state), leaves `S_0…S_{T-1}` spelled, and extraction then
  decorates those spelled steps with every minted elementwise claim
  whose pattern recurs there (`gen_1_k` ×240 nodes / `gen_11_k` ×47
  etc.).  The module pays the scan wrapper *and* the spelled chain —
  strictly more work than either alone.

Secondary contributor, measurable but smaller: the `profitable`
referee (`genkernel.py:768`) times each claim **once**, at one
recorded occurrence (`_handler_env`), synced per call — an honest
per-site win.  Extraction then applies the claim at every e-class it
saturates (~30–240 sites here).  Per-site the `_gk_s` strided kernel
on broadcast `(8,64,4)`-class operands (~15–32 µs device + Triton
launcher/`_triton_ometa` host) is at best break-even against the
aten ops it replaces; multiplied, it is the largest device line.

## 6. Fixes — what is concrete vs structural

Concrete but modest:

1. **`_seq_args` broadcast materialisation** (`genkernel.py:2372`):
   GLA pays ~440 µs/call copying 12 MB to feed a 144 µs kernel.
   A stride-aware `_gs` variant (ometa-style in-kernel offsets, the
   `_gk_s` approach) or choosing the non-broadcast operand layout
   would recover most of it.  Real work, bounded scope.
2. **`_gk_s` small-shape weakness**: BLOCK=1024 with an NDIM-loop of
   int64 div/mod + 3 `ometa` loads per element is heavy for
   n≈2–33 K broadcast outputs.  A broadcast-specialised kernel (or
   routing tiny-broadcast sites back to the compiled spelled body)
   would cut the 451/895 µs lines.  Also bounded.

Structural (document, don't patch):

3. **Per-step-output scans need a trajectory kernel.**  The honest
   fix for GLA/Mamba is a `_gs` variant that stores the state
   trajectory `(T, B, d…)` — or better, fuses the read-out
   `q_tᵀS_t`/`C_tᵀh_t` into the same kernel — so one launch replaces
   recurrence *and* read-outs.  Until then the route can only ever
   fold the last step on these modules.  This is a feature, not a
   bug fix; it is the difference between "scan covers the carrier"
   and "scan covers the carrier's *uses*".
4. **Host dispatch scales with extracted-term size.**  Even with
   perfect kernels, a ~5 k-node extracted term pays ~1–2 ms of tape
   interpretation per call; `CudaGraphRunner` is the shipped answer
   and it works — the residual is that Inductor also fuses, so the
   graph arm wins vs eager (1.7×) but not vs compile on these rows.
5. **Claim multiplicity is unpriced.**  `profitable` credits one
   occurrence; extraction spends the claim at every site.  A
   site-count-weighted keep decision (or a per-site loss floor in
   `novel_cost`) would stop borderline elementwise folds from
   multiplying into a loss — but the per-site margins here are
   genuinely positive in the referee's synced measurement, so this
   is a second-order effect behind (3).

## Caveats

- Single mint per workload this session (mint nondeterminism —
  resweep's GLA gen_us was ~2× this run's).  Direction and
  attribution are consistent with the committed sweep rows; exact
  µs are not comparable across mints.
- Profiler `device_time_total` sums kernel durations; overlapping
  kernels can make it exceed wall-clock device occupancy — the
  wall−device "host share" is therefore approximate (same caveat as
  T-034).
- `_gk_s` per-launch device times (~15–32 µs) are profiler-
  attributed and include any launch-gap the tracer assigns; the
  aggregate (451/895 µs) is the reliable figure.
- Binding-call counts came from a wrapper-instrumented rebuild of
  the same sink (`hybrid_sink` bindings + counters); term-node
  counts are raw `Op` census including duplicated subtrees.

## Files / commands

- Probe scripts (scratch, not committed): `/tmp/module_loss_probe.py`
  (timing + profiler + graph arm), `/tmp/module_loss_probe2.py`
  (term census + per-binding call counts),
  `/tmp/module_loss_probe3.py` (operand shapes per binding),
  `/tmp/referee_env_probe.py` (referee env shapes),
  `/tmp/term_dump.py` (term structure around `scan_*_k`).
- Code read: `play.py:2492-2705` (`_gen_board`/`_probe_play`/
  `gen_probe`/`_probe_graph_arm`), `play.py:2913` (`_zoo_case`),
  `meta_arena.py:1083,1471` (`_route_scan_chains`/`deliverable`),
  `genkernel.py:1371-1509` (`triton_bindings`/`_gk_s`),
  `genkernel.py:2218-2257` (`_seq_triton_source`/`_gs`),
  `genkernel.py:2362-2496` (`_seq_args`/`_seq_binding`),
  `genkernel.py:768-1046` (`profitable`/`_paired_time`),
  `zoo.py:516,726,1709` (the four module definitions).

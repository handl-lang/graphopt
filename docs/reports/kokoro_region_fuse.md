# Kokoro region-fuser — delivered-term fusion path, measured

Date: 2026-11-14.  Hardware: RTX 2050 4 GB (shared), torch
2.14.0+cu130, fp32, CUDA.  Driver:
`bench/drivers/probes/kokoro_region_fuse.py` (new; imports only
`tts_kokoro_probe` — the `kokoro_fusion_probe`/`fusion_floor_probe`
chain needs `transformers`, which is not a repo dep; the region
census is inlined with the multi-root fix).  Follow-up to
`kokoro_fusion_sites.md` (the census: 348/289 sites, ~35–45 ms/call
prize) — this report lands and measures the delivery path.

**Verdict: the prize is real and reachable through the delivered
module — `decoder` 116.8 → 81.4 ms/call (−35.4 ms, 1.43×),
`decoder-nc` 116.9 → 87.6 ms/call (−29.3 ms, 1.33×)** — essentially
the whole predicted band, on the REAL `IRModule` plan tape, with a
term rewrite proven **bitwise identical** to the delivered term.
The one honest caveat is the verify gate: inductor's fused kernels
reassociate (FMA contraction etc.), so fused-vs-delivered lands at
max_rel ≈ 1.4e-3 (`decoder`) / 5.9e-4 (`decoder-nc`) — inside
rtol=2e-3 / 1e-3, outside the stock rtol=1e-4.  There is no
bitwise-exact inductor mode; the integration spec below prices the
verify-policy options.

## 1. Design — which option

The spec offered three shapes:

* **(a) carrier law / lowering pass over delivered terms** — match
  elementwise-region patterns and fuse at IRModule build/eval;
* **(b) rewrite law minting `fused_region` leaves pre-search**;
* **(c) orchestrator subterm routing** (eqsat — read-only here, and
  no subterm-routing seam exists: `_lower_extracted` routes the
  WHOLE term to the first carrier `ExecutorSpec` whose `accepts`
  probe holds — carrier executors are whole-spine batched
  lowerers, not region fusers).

**Chosen: (a), implemented as a lowering-time term rewrite +
binding, measured end-to-end through `ir_to_torch_module`.**
Reasons:

* The regions are a property of the *extracted* term, not the
  search — minting `fused_region` leaves pre-search (b) would
  require the e-graph to carry compiled-callable leaves through
  extraction, cost modelling and verify for zero search benefit;
  the census already computes them post-search for free.
* Carrier `ExecutorSpec`s can't express "some subterms fused, rest
  spelled" — `accepts` is a whole-term probe and `spec.lower`
  replaces the entire module.  A region fuser is orthogonal to
  carrier routing: it composes UNDER any routed executor (it is a
  term→term rewrite + one binding).
* Keeping it in `kit_torch` (the eval/lowering package) respects
  the hexagonal boundary: torch.compile is a torch concern;
  `mlopt_carriers` imports `kit_torch`, never the reverse, and
  `mlopt_discovery`/`genkernel` stay out of the shipped path (the
  census vocabulary `ELEMENTWISE` is re-spelled locally — 28 op
  names, see §5).

## 2. The mechanism (`kokoro_region_fuse.py`)

1. **Census** — weakly-connected components over the elementwise op
   set (`genkernel.ELEMENTWISE`, inlined), DAG-safe, with the
   multi-root `len(int)` fix from `kokoro_fusion_probe`.
2. **Eligibility (conservative single-exit)** — a region fuses iff
   it has a UNIQUE root (one member with no elementwise parent) and
   every non-root member's parents are all inside the member set:
   the root is the only exit, exactly the `_fusion_sites`
   collector-legal shape, so no intermediate value is computed
   twice and no external consumer sees a stale spelling.  Skips are
   counted honestly (`multi_root`, `multi_exit`, `small`,
   `unreachable`).
3. **Canonicalisation + dedup** — members topo-ordered from the
   root; every non-member arg (Op/Var/Param/**Const**) becomes a
   positional leaf in first-appearance order.  The dedup key is the
   internal structure only — `(op, frozen attrs, arg wiring)` — so
   the 48 SineGen regions collapse to **6 distinct kernels**
   (`decoder`) / 7 (`decoder-nc`).  Const *values* are runtime
   inputs, not baked constants: two regions differing only in a
   scalar share one compiled body.
4. **Rewrite** — each region root becomes
   `Op.make("fused_region", *leaves, fused_k=kid, validate=False)`;
   the rest of the DAG is rebuilt memoised (shared subterms stay
   shared).  Boundary leaves that contain inner regions are
   themselves rewritten, so nesting composes.
5. **Binding** — a copied `OpTable` adds
   `"fused_region": _fused_region_binding`, which dispatches
   `_REGISTRY[fused_k](*xs)`.  Each registry entry is
   `torch.compile(spelled, dynamic=False)` over the unrolled member
   list — the probe's positional-args approach: **every boundary
   leaf, Consts included, arrives as a runtime tensor arg** (the
   plan tape mints Consts exactly as spelled eval does), so no Const
   tensor is ever inlined into the compiled spec — the
   `genkernel.gen_bindings` FakeTensorMode blocker does not apply.
6. **Evaluation** — the fused term lowers through the stock
   `IRModule`/`_EvalPlan` tape: `fused_region` slots participate in
   liveness/releases like any op; one tape step replaces ~9 binding
   dispatches.

`min_ops=2` (the fusion-site floor).  `mode=spelled` runs the same
unrolled body WITHOUT `torch.compile` — the rewrite-correctness
control (§4).

## 3. Measured results (`iters=1`, `max_enodes=100_000`, fp32)

| arm | regions fused | ops fused | distinct kernels | skips | delivered | region-fused | Δ | verify fused-vs-delivered |
|---|---|---|---|---|---|---|---|---|
| `decoder` | 52 | 446 | 6 | multi_root 3, small 17 | 116.77 ms | **81.42 ms** | **−35.35 ms (1.43×)** | max_abs 3.76e-4, max_rel 1.42e-3 |
| `decoder-nc` | 52 | 371 | 7 | multi_root 4, small 42 | 116.89 ms | **87.60 ms** | **−29.29 ms (1.33×)** | max_abs 1.97e-4, max_rel 5.90e-4 |

Reference points: eager-wrap 123.2/123.3 ms; probe's
`torch.compile(wrap)` ceiling 72.1/70.7 ms.  The fuser recovers
~60% (`decoder`) / ~55% (`decoder-nc`) of the eager→compile gap
with 52 fused sites — consistent with the census estimate that the
prize is concentrated in the transcendental chains (the remaining
pointwise ops — conv-adjacent adds, the 13-op residual chain that
OOM'd leaf-eval, small singletons — stay spelled).

`f0n`/`textenc(-p)` were not re-run: the census already priced them
negligible (12 sites/29 ops; 0 sites).

## 4. Verify — bug vs numerics, settled

* **`mode=spelled` (same rewrite, eager unrolled bodies):
  fused-vs-delivered `max_abs = 0.000e+00` — BITWISE IDENTICAL.**
  The term rewrite, leaf marshaling, Const minting and plan-tape
  integration are exactly correct; every fused region computes the
  same DAG the tape would have.
* **`mode=compiled`: max_rel 1.42e-3 (`decoder`), 5.90e-4
  (`decoder-nc`)** vs delivered — fails stock `rtol=1e-4`, passes
  `rtol=2e-3` / `rtol=1e-3` respectively.  This is inductor
  reassociation noise (FMA contraction, libdevice `pow`/`sin`
  paths), NOT a rewrite bug — the spelled control isolates it
  completely.  There is no exact-mode `torch.compile`; the same
  noise floor applies to any `TorchCompileRunner` delivery.
* Context: delivered-vs-eager-wrap is already `max_abs ≈ 8.7e-2` /
  `5.5e-2` on these arms (the iters=1 search itself moves the
  output), so fused-vs-eager (≈1.1e-1/8.9e-2) is dominated by the
  search delta, not fusion.  The verify contract that matters —
  fused vs DELIVERED — is the numbers above.
* Region-level corroboration (`kokoro_fusion_sites.md` §3): the two
  9-op signatures measured 5.7e-6 / 2.9e-6 max diff spelled-vs-
  fused on real tensors; end-to-end amplification through ~50
  sequential regions lands at the observed ~1e-3 rel.

## 5. Integration spec (not landed — see honest ceiling)

The measured path is a bench driver, not a package change.
Landing it in shipped code has three real costs:

* **Coverage wall** — `fail_under=100` on `kit_torch`/
  `mlopt_carriers` (`pyproject.toml` `source`).  A `region_fuse`
  module + tests is a real work item, not a drive-by.
* **Verify policy** — the stock `lower(verify=True)` gate at
  rtol=1e-4 declines this delivery.  Options, in preference order:
  (i) per-region build-time screening — evaluate each candidate
  region's compiled body vs its spelled body ONCE on the real
  boundary tensors (the probe's `_eval_leaves` machinery) and
  register only regions under a tight bound; keeps rtol=1e-4
  end-to-end for screened regions, costs one prefix eval per
  distinct kernel; (ii) task-level verify (`TaskMetric`) for arms
  where 1e-3 rel is acceptable; (iii) accept a per-delivery rtol
  floor for fused modules, recorded in `stats` — never silent.
* **Dynamo recompile limit** — every exemplar `spelled` closure
  shares ONE code object, so dynamo counts each exemplar's first
  compile against a single recompile entry; the default limit of 8
  silently degrades later regions to eager (observed: decoder-nc
  lost ~7.7 ms/call until `recompile_limit` was raised to 256).
  Production fix: generate per-exemplar code objects (or set the
  limit from the census count at build time).

Suggested landing shape (when scheduled):

* `kit_torch/region_fuse.py` — the census/eligibility/rewrite/
  registry machinery, torch-only, `ELEMENTWISE` re-spelled locally
  (28 names; do NOT import `mlopt_discovery` — wrong direction of
  the hexagonal boundary).
* Entry point: an opt-in flag on `ir_to_torch_module`/`IRModule`
  (e.g. `region_fuse=True`) or a `RegionFusionRunner` in
  `kit_torch.runners` wrapping the lowered module's term before
  `IRModule` construction — the rewrite needs the TERM, so a
  post-module runner cannot reach it; the clean seam is inside
  `IRModule.__init__` between `_fold_weight_chains` and
  `_build_params`, gated by a constructor kwarg threaded from
  `TorchSink.lower`.
* `fused_region` binding lives in `torch_bindings`-adjacent
  registry; `OpTable` unchanged (the driver copies the ambient
  dict — shipped version registers the op name properly).
* Size threshold: the census's one losing shape (11-op rsqrt,
  0.57× on small tensors) costs ~19 µs — below noise; a
  leaf-count/shape-aware gate is future work, not a blocker.
* `dynamic=False` specializes on first-seen shapes — fine for
  fixed-shape decode steps; variable-shape deliveries need
  `dynamic=True` or per-bucket recompile (document).

## 6. Files

* `bench/drivers/probes/kokoro_region_fuse.py` — NEW, the whole
  measurement path (census → eligibility → rewrite → OpTable →
  verify → timing).  `mode=compiled|spelled`, `min_ops`,
  `recompile_limit=256`.
* No package files touched.  Env note: the dev venv needed
  `transformers==4.57.1` + `loguru` (scratch `uv pip install`,
  outside the lockfile — `uv sync` reverts; transformers 5.x is
  broken for this import path).

## 7. Honest ceiling

* Measured, not promised: −35.4 ms (`decoder`), −29.3 ms
  (`decoder-nc`) per call on the real delivered module, verify
  margin documented above.  The census's ~35–45 ms estimate lands
  at the low end — the skipped regions (3–4 multi-root, 17–42
  singletons, the OOM'd 13-op chain) and the non-transcendental
  pointwise remainder account for the difference.
* Single input/seed, iters=1, shared 4 GB card — absolute ms will
  shift; the spelled:compiled mechanism and the verify margin are
  the durable numbers.
* The verify gate is the real integration cost: this delivery
  fails stock rtol=1e-4 and needs either per-region screening or
  an explicit tolerance policy before it can ship through
  `lower(verify=True)`.

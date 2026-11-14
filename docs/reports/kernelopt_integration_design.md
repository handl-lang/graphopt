# kernelopt ↔ graphopt integration — design exploration

Read-only exploration across graphopt/kernelopt/eqsat/lib-kernel-ir/
kit-kernel-ir/lib-tensor/host-tensor. Companion to
`kernelopt_bridge_probe.md` (the measured probe: **no pays on whisper
subterms** — this doc is the architecture map for if/when integration
is revisited).

## What exists today

- **Two different "Term"s.** `mlopt_core.ir.Op/Var/Const/Param/IR`
  (shared torch-model IR) is *not* `lib_kernel_ir.Term` (closed 10-op
  kernel IR: `tile_load/unpack_i4/dequant/dot/split/stage/reduce/
  store/bias/pipeline`). Nothing translates between them.
- **kernelopt never lifts torch models.** Its board cases are
  hand-spelled `mlopt_core` terms (`mlopt_kernel/play.py::
  _kernel_cases`). The game seam is generic:
  `mlopt_kernel.play.register()` seats `kernel` into
  `mlopt_discovery.play.DOMAINS` via `spec_domain` +
  `gamekit.players.registry.register_player`.
- **A bridge exists as a scratch probe** —
  `bench/drivers/probes/kernelopt_bridge_probe.py` exports whisper
  subterms via `TorchSource`, runs `Optimizer.search`+`TorchBackend`,
  then the same spelled IR through `KERNEL_RULES`+`KernelSink`.
- **`mlopt_kernel.search` is real but barely wired** — gate→bound→
  measure over injected Candidates; the only real consumer is
  `bench/drivers/probes/awq/gen_manifest.py`, which sweeps
  split_k × hand grid for AWQ qgemv and vendors the winning Triton
  source into `lib-triton-qgemv` + `manifest.py::ROUTES`.
- **`KernelSink` is the delivery seam** — implements the
  `mlopt_core.ports.Sink` shape; fused ops lower via `torch.compile`
  composites; `om_chunk_scan` routes to `ScanRunner`.

## 1. Bidirectional flow — two tracks

**Track A (cheap, recommended first): graphopt IR → KERNEL_RULES →
KernelSink as an alternate Sink.** Productionize the `_decompose`
shim (atomic `softmax`/`layer_norm`/`linear` → decomposed spellings)
as a real module — belongs in **kernelopt**
(`mlopt_kernel.decompose`; needs only `mlopt_core.ir`). Then
`Optimizer(sink=KernelSink(), rules=KERNEL_RULES, cost_fn=
kernel_cost)` runs end-to-end on a `TorchSource` export. Captures
measured kernelopt wins (silu 2.2×, layernorm ~5×, sdpa ~1.8× per
kernelopt notes) without touching lib-kernel-ir. **Effort: M.**

**Track B (the real "pattern → derived kernel"): extend one family
through lib-kernel-ir.** Only qgemv/AWQ has all pieces today
(`gen_manifest.py` is the proof and the ceiling). Generalizing needs:
(a) mlopt Op-subtree → `lib_kernel_ir.Term` recognizer for a chosen
family; (b) term enumerator (schedule params as term literals);
(c) `Candidate.build` JIT via `compile-triton` (a Rust **binary** via
`COMPILE_TRITON_BIN` — process-boundary wart); (d) delivery artifact
(vendored source + route row, the `lib-triton-qgemv` pattern).
**Effort: L**, requires ADR-level IR extension for anything beyond
GEMV-class (dequant/dot/split/bias are in the IR; pointwise is NOT
expressible today).

**Ownership:** the IR↔Term mapper + candidate enumerator belong in
**kernelopt** (owns `search`, deps on `mlopt-core`+`lib-kernel-ir`).
graphopt sees it only through the plugin seam (`spec_domain`/
`register_player`/`sink=` override). A graphopt→kernelopt *dependency*
is impossible anyway — kernelopt is private.

**Risks:** Track A's measured pay is real but bounded (mm epilogues
sometimes lose; `kernel_flat` control exists for this). Track B's
hard part is the 10-op IR only covers memory-bound GEMV-class
kernels — compute-bound needs vocabulary extension first.

## 2. Schedule/layout as searchable data — partially there

- **lib-kernel-ir:** schedule IS data inside terms — `split.parts`,
  `pipeline.depth` literals (ADR-0166: "a schedule-free IR can only
  express naive schedules"). But enumeration is manual (`qgemv()` is
  a Python constructor; `gen_manifest.py` sweeps grids by hand), and
  **launch config (block_n/block_k/warps) sits outside the IR** —
  `compile-triton` takes it as CLI flags.
- **kernelopt `scan.py`:** `build_scan_plan` *detects* a fixed
  schedule — inference, not search; one plan per term.
  `om_chunk_scan` is a marker law-minted per arity — schedule
  variants enter the e-graph as law-minted members (the existing
  mechanism for searchable layout).
- **mlopt_core IR:** `split`/`chunk`/`concat` carry `sizes`/`dim`/
  `index` in attrs (bindable); `opmeta.py` tags them
  COST_VIEW/RELAYOUT/VIEWISH. Alternative chunkings can be minted as
  e-class members today.
- **Explicitly absent** (`priorart.py:37-42`): vectorization/
  unrolling/pipelining have "no term-level expression: the IR has
  no lane/vector or stage axis."

**Minimal designs:** (i) *cheap S–M*: extend the
`kern_fuse_chunk_scan` pattern — laws whose `derive`/`dspec` compute
chunk attrs from metavariables, e-graph holds multiple schedule
spellings, cost picks. (ii) *medium M*: lib-kernel-ir — generalize
`qgemv()` into a family constructor + pull launch config into the
term/schedule record (touches Rust `term.rs`/`codegen.rs` + JSON wire
format). (iii) *hard L*: tile/block-shape search over mlopt_core IR —
needs denotation-transparent schedule ops (the `pipeline(depth,
body)` identity-denotation trick is the template) and will collide
with e-graph explosion unless budgeted (`rule_budgets`, board-level
player choice rather than saturation).

## 3. Prior art (don't reinvent)

- **TVM/Ansor:** sketch+knobs autoscheduler — lesson: fixed loop
  structure + schedule axes beats free-form search.
- **Triton autotune:** per-kernel config grid — what
  `gen_manifest.py`'s grids hand-roll; lib-triton-qgemv ROUTES
  already imitates its per-shape caching.
- **torch.compile/Inductor:** the incumbent — `KernelSink` IS
  Inductor-fused composites; measured residual gap (real
  flash/online-softmax, chunked-scan batching) is what Inductor
  doesn't do either — kernelopt's whitespace.
- **Halide:** `compute_at`/`vectorize`/`unroll`/`prefetch` are
  schedule attributes with no term-level expression here;
  `priorart.py`'s "honest miss list" is the boundary to respect.

## Bottom line

Cheapest real integration = **Track A** (M effort, machinery exists).
Ambitious Track B = L, currently only GEMV-class expressible.
Schedule-as-data partially exists; launch config outside the term
(M to fix, touches Rust wire format). Measured pay concentrates in
pointwise fusions (already covered by torch.compile); IR-expressible
families are memory-bound GEMVs — real but narrow wins (AWQ manifest
is the proof and the ceiling).

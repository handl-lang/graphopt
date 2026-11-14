# Architecture audit — kernelopt + lib-kernel-ir + kit-kernel-ir

Read-only audit (subagent). Scope: `src/mlopt_kernel/` (laws, cost,
sink, board, weights, priorart, scan, search, play), `lib_kernel_ir`,
`kit_kernel_ir`, 46-case bench corpus, pyproject/CI/docs, plus the
kernelopt integration design doc. One end-to-end trace:
`bench/discovery/kernel.py::run_case` → `EGraph.apply_rule(KERNEL_RULES)`
→ `extract_best(kernel_cost)` → `KernelSink.lower` → `KernelExecutor.forward`
→ `sink.verify` — pipeline coherent.

## 1. Bugs / correctness

- **B1 — `mlopt_kernel.search` crashes on shape-mismatched candidates**
  (`search.py:170`): `_cosine(out, expected)` runs outside the `try`
  guarding `cand.build()`; wrong-shaped/non-tensor output kills the
  whole `search()` instead of recording `Rejected`. Fix: move cosine
  inside try or pre-check shapes.
- **B2 — "bound" stage of `search()` is dead code** (`search.py:153–158`):
  `cand.term.cost({})` raises KeyError for real Terms (leaves not in
  empty env) → `term_cost` always `Cost()`; the prune `if` is a literal
  no-op. Docstring advertises cost-bound pruning that never happens.
  Either plumb shapes/env through `Candidate` or delete + fix docstring.
- **B3 — scan plans parse but crash at run time** (`scan.py:590–623`):
  `_scalar`/`_scales` — `is_chunk_scan`/`build_scan_plan` never validate
  scale leaves are scalar; `float(v)`/`KeyError` raise inside
  `ScanRunner.run` AFTER `_scheduled` committed (no spelled-eval
  fallback). Fix: reject non-scalar/missing scale leaves at plan time.
- **B4 — `kernel_cost` memo requires hashable terms** (`cost.py:88`):
  `ck = ("kc", term)` breaks on unhashable Const values. Low likelihood;
  add try/id-fallback or document precondition.
- **B5 — `KernelSink.ops` mutates `OpTable.core()`** (`sink.py:539–541`):
  assumes fresh table per call; if eqsat returns shared instance,
  bindings leak globally. Verify `OpTable.core()` semantics.
- **B6 — play-domain case generator can't reach 9 of 28 laws**
  (`play.py:84–234`): archetypes cover pointwise/mm/norm only — no
  decayed site, no concat-chunks, no sdpa cap/mask/gqa variants.
  `kern_fuse_om_decayed`, all `kern_fuse_chunk_scan_{2..6}`,
  `kern_fuse_sdpa_{cap,mask,gqa}` can never appear in `state.matches` —
  dead moves; their `a:r:*` weight buckets are untrained noise.
  Fix: add decayed/scan/sdpa-variant archetypes (bench builders
  directly reusable).

## 2. Duplication / two-Term confusion

- Two `Term` types cleanly separated (lib_kernel_ir.Term vs
  mlopt_core.ir) — no leakage, no bridge (Track B gap, architectural).
- Vendored carrier semantics (deliberate, drift-prone): `_decay`/
  `_om_decayed`/`scan.py` are carrier-free ports of
  `mlopt_carriers.decayed_lower` — no shared source of truth.
- `kf_*` composites spelled twice: Python lambda in `sink.py` +
  term-level `pa_kf_*_expand` in priorart.py. Adding one fused op
  touches 4 files (laws/sink/cost/priorart) — consider generating
  priorart expands from the binding table.
- `kernel_cost` vs eqsat cost fns: properly injected, no duplication.

## 3. Inconsistencies / docs drift

- **`lib-kernel-ir` is not a declared dependency** — `search.py:36`
  imports it top-level but pyproject omits it (path/pytest hacks only).
  On public install `import mlopt_kernel.search` → ImportError.
  lib-kernel-ir is public now — add git+tag dep. **Top blocker for
  shipping `search`.**
- `mlopt-orchestrator` is a uv source but not a dependency — Track A's
  `Optimizer` impossible without dep edit.
- Stale "PRIVATE" banners: `paper/README.md:3`,
  `notes/kernel-engineer-board.md:3` — repo is public now.
- lib-kernel-ir "nine ops" docstring/error (OPS has ten since
  ADR-0166); kit-kernel-ir README "Private per ADR-0150";
  lib-kernel-ir/kit-kernel-ir AGENTS.md are myco boilerplate only.
- `board.py:218` calls `self.env._extract()` — private eqsat API.

## 4. Hardcoded vs generated

- `KERNEL_RULES` (28) hand-curated — appropriate; overnight/mint is
  the growth path. Not a defect.
- `cost.py::_OP_WEIGHT` hand-tuned, uncalibrated (`exp`=2.0 but
  `exp2`/`expm1`/`log2`/`silu` fall through to 1.0).
- 46-case bench corpus hand-spelled — fine as fixture.
- lib-kernel-ir `OPS` closed table — intentional (ADR-0165/0166).
- Launch config outside IR — `gen_manifest.py` sweeps by hand;
  pulling into term/schedule record = M effort touching Rust
  `compile-triton` wire format. Real gap.

## 5. Track A/B assessment

- **Track A** architecturally unblocked (`KernelSink` implements Sink
  port, `kernel_cost` the cost_fn convention). Gaps: (a) `_decompose`
  shim lives only in graphopt probe — promote to `mlopt_kernel.
  decompose`; (b) binding-table coverage unverified vs real
  TorchSource exports (~80 ops, lacks chunk/getitem/fma/atomic
  layer_norm/dropout/embedding/conv* — `scan.py::_slice_index` already
  anticipates `chunk`/`getitem` the sink can't execute); (c)
  `Optimizer` dep missing; (d) `KernelSink.verify` vs ports.Sink
  conformance unconfirmed.
- **Track B** entirely unbuilt, genuinely architectural — no
  recognizer/enumerator/lowering path; 10-op IR covers only GEMV-class;
  compute-bound needs ADR-level vocabulary extension. `search.py`
  needs B1/B2 fixed first.

## Priority order

1. Declare `lib-kernel-ir` (git+tag) in pyproject — `search` uninstallable-broken.
2. B1 shape-mismatch crash in `search()`.
3. B2 dead bound stage / misleading docstring.
4. B6 play-domain cases can't exercise 9/28 laws — prior partially untrained.
5. B3 scan-plan validation gap.
6. Docs: PRIVATE banners, "nine ops", kit README, readiness items.
7. Track A prep: promote `_decompose`, audit `kernel_ops()` vs
   TorchSource vocabulary, add `mlopt-orchestrator` dep, verify
   Sink conformance.

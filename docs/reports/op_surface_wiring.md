# Op-surface wiring — host-tensor ATen ports × mlopt-torch bindings

Date: 2026-11-13.  Scope: `packages/mlopt-torch/src/mlopt_torch/torch_bridge.py`
(+ `tests/test_torch_integration.py`).  Read-only references:
`handl-lang/host-tensor` @ `d0de077`, `handl-lang/lib-tensor` @ `82f6f63`.

## TL;DR

- host-tensor grew **31 → 238 `tensor.*` ops** (`src/ops/mod.rs::TENSOR_OPS`):
  the ATen port generator (`tools/aten_port_list.txt`, 207 declared entries —
  202 implemented + 5 honest seams) plus 31 handwritten ops.
- Before this change, every host-tensor op mlopt didn't already bind lifted as
  an `opaque` passthrough leaf (commit `f7619ea`) — correct but unoptimizable.
- **Corpus measurement** (`mlopt_discovery.intake.candidates`, 241 exportable
  workloads; `zoo()`/`zoo_variants()`, 57 each — the zoo corpus lifts with
  **zero** opaque ops): the only host-implemented ops appearing as opaque were
  `aten::selu`, `aten::softshrink`, `aten::avg_pool2d`, `aten::cdist`.
- This change binds those plus the high-value real-model slice: **12 new
  bindings + 4 ATen→IR aliases, 1 scalar-operand op (`floor_divide`),
  6 shape rules**.
- Remaining opaque ops in the corpus are all ops **host-tensor does not
  implement** (losses, RNN cells, conv_transpose, pooling variants,
  upsample variants, `im2col`/`col2im`, `grid_sampler`, `repeat_interleave`,
  …) — they stay opaque by design until host-tensor ports them.

## What changed in `torch_bridge.py`

1. **`_ATEN_TO_IR` aliases** — ATen spellings that ARE bound generators:
   `mT`/`swapaxes` → `transpose` (dim0/dim1 positions line up with
   `ATTR_SCHEMA["transpose"]`), `moveaxis` → `movedim` (source/destination),
   `ger` → `outer` (aten.outer is literally the alias).
2. **`_LOCAL_ATTR_SCHEMA`** — domain-local positional-attr canonicalisation
   for ops the eqsat-owned `ATTR_SCHEMA` doesn't cover (the substrate table
   can't be edited from this repo).  Merged *under* `ATTR_SCHEMA` at the
   export boundary; `Op.make` validation is unaffected.  Covers the scalar
   tails of `softshrink`/`hardshrink` (`lambd`), `logit` (`eps`),
   `logsumexp` (`dim`,`keepdim`), `avg_pool2d` (`kernel_size`…`divisor_override`),
   `rot90` (`k`,`dims`), `diag` (`diagonal`), `diagonal` (`offset`,`dim1`,`dim2`),
   `linspace` (`start`,`end`,`steps`), `cdist` (`p`,`compute_mode`).
3. **New `_CORE_TORCH_BINDINGS`** — `selu`, `softshrink`, `hardshrink`,
   `logit`, `floor_divide`, `logsumexp`, `avg_pool2d` (`_avg_pool2d_torch`),
   `rot90` (`_rot90_torch`), `diag`, `diagonal`, `linspace`, `cdist`.
   Denotation = the ATen kernel host-tensor ports claim provenance from, so
   the torch lowering IS the reference semantics.  `floor_divide` also joined
   `_SCALAR_OPERAND_OPS` (the `// 2` scalar stays a `Const` operand).
4. **Shape rules** via `mlopt_core.typing.register_shape_rule` (public sink,
   same direction `play.py`/carriers use): `logsumexp`, `avg_pool2d`, `diag`,
   `diagonal`, `linspace`, `cdist`.  Elementwise ports need nothing (typing's
   default returns the operand shape); anything not cheaply computable reports
   `None` rather than a wrong shape.  Notable correctness detail found while
   wiring: `torch.diagonal`'s defaults are `dim1=0, dim2=1` (NOT -2,-1) and
   its output appends the diagonal length LAST — the rule and binding both
   encode that.

## Coverage table — host-tensor `TENSOR_OPS` (238) × mlopt binding status

Status is computed by running the host op's ATen name through
`_canon_aten_name` + `_IR_TO_TORCH` membership (carrier bindings resolve
lazily, so `eye` counts as bound — verified `aten.eye` round-trips exactly).

### A. Handwritten host ops (31) — codec/autograd/wire surface

| host op | mlopt status |
|---|---|
| zeros, ones, add, mul, sub, div, matmul, linear, conv2d, max_pool, batch_norm, reshape, transpose, unsqueeze, select, concat | **bound** (via `cat`→`concat`, `mm`-family→`matmul`, etc.) |
| from_list, to_list, shape | n/a — codec/wire ops; mlopt terms are already tensors |
| derive + `*_backward` (10 ops) | n/a — autograd adjoints; mlopt's `derive` machinery is the carrier-side analogue, different surface |

### B. Port-list implemented ops (202)

**Bound: 161** (includes the 12 newly wired + 4 aliases; `eye` resolved via the lazy carrier-table binding).  Highlights already
bound before this change: the whole unary/binary math block, reductions,
sort/select, linalg `mm`/`bmm`/`addmm`-family→`matmul` aliases, pooling
`adaptive_avg_pool2d`/`max_pool2d`, `einsum`, `median`/`mode`/`kthvalue`/
`searchsorted`/`nonzero`/`sort`/`topk`, `normalize`→`linalg_vector_norm`,
`to`→`to`, `eye` (`_eye_torch`), creators, indexing.

**Bound by this change (12 bindings + 4 aliases):**

| host op | ATen | IR op | notes |
|---|---|---|---|
| tensor.selu | aten::selu | `selu` | corpus: `nn.SELU` |
| tensor.softshrink | aten::softshrink | `softshrink` | corpus: `nn.Softshrink`; `lambd` attr |
| tensor.hardshrink | aten::hardshrink | `hardshrink` | same family (not corpus-observed) |
| tensor.avg_pool2d | aten::avg_pool2d | `avg_pool2d` | corpus: `nn.AvgPool2d`; full attr tail |
| tensor.cdist | aten::cdist | `cdist` | corpus: `Cdist`; `p` attr |
| tensor.logsumexp | aten::logsumexp | `logsumexp` | high-value (attention/losses) |
| tensor.floor_divide | aten::floor_divide | `floor_divide` | `//`; scalar→Const operand |
| tensor.logit | aten::logit | `logit` | `eps` attr |
| tensor.rot90 | aten::rot90 | `rot90` | `k`,`dims` attrs |
| tensor.diag | aten::diag | `diag` | 1-D↔2-D dual behaviour = torch.diag |
| tensor.diagonal | aten::diagonal | `diagonal` | offset/dim1/dim2 attrs |
| tensor.linspace | aten::linspace | `linspace` | zero-operand creator; dtype attr |
| tensor.mT | aten::mT | → `transpose` | alias |
| tensor.swapaxes | aten::swapaxes | → `transpose` | alias, dim0/dim1 |
| tensor.moveaxis | aten::moveaxis | → `movedim` | alias, source/destination |
| tensor.ger | aten::ger | → `outer` | alias |

**Implemented in host-tensor, still unbound in mlopt (41)** — would lift as
`opaque` if a model exports them; **none appear in the intake/zoo corpus**.
Grouped for the next wiring pass:

- *Cheap elementwise/binary* (denotation obvious): `tan`, `threshold`,
  `deg2rad`, `rad2deg`, `signbit`, `hypot`, `logaddexp`, `logaddexp2`,
  `nextafter`, `copysign`, `float_power` (pow alias for f64 — needs a
  semantics check vs `pow`'s integral-exponent path), `vdot` (conjugate
  dot — fine for the f64 surface).
- *Shape/select* (need schema + shape rules): `argwhere`, `index_fill`,
  `take`, `constant_pad_nd` (F.pad's primitive spelling — `aten.pad`
  already bound), `logspace`.
- *GEMM variants*: `addmm`, `addbmm`, `baddbmm`, `addr`, `addmv`
  (beta/alpha tails; `bmm`/`mv`/`dot` already fold into `matmul`).
- *Metadata/non-tensor returns*: `size`, `stride`, `dim`, `is_contiguous`
  — likely stay unbound: they return ints/bools/tuples, not tensors.
- *Newly closed host seams* (d0de077 — real kernels now exist, zero corpus
  exposure): `tensordot`, `kron`, `cross`, `norm` (p-norm scalar),
  `renorm`, `quantile`, `bincount`, `meshgrid`, `corrcoef`, `inner`,
  `dist`, `allclose`, `is_nonzero`, `msort`, `pdist`.

### C. Port-list seams (5) — host stubs that fail loudly

`empty`, `rand`, `normal`, `unique`, `nonzero_dim`.  mlopt already binds
`empty`/`rand` for its own torch lowering — a term minted from those binds
evaluates in torch but **cannot** deliver to `.hnl` (host-tensor has no
kernel).  Worth a note wherever the kit bridge assumes op parity.

## Corpus evidence

Probe: every `intake.candidates()` / `zoo()` / `zoo_variants()` workload →
`export_to_ir` → walk the term DAG counting `opaque` nodes and their `aten`
quals (fp64-input dtype mismatches in ~31 workloads are a **pre-existing**
zoo/intake convention issue — f64 feeds vs f32 params — unrelated to this
change; they fail inside `torch.export` itself).

- **Before:** intake showed 46 distinct opaque `aten::*` quals; the
  host-implemented subset was `selu`, `softshrink`, `avg_pool2d`, `cdist`.
- **After:** those four lift first-class; the remaining ~42 opaque quals are
  exclusively ops host-tensor does not implement (losses
  `cross_entropy`/`mse`/`l1`/`smooth_l1`/`huber`/`bce*`/`kl_div`/…, RNN
  cells, `conv_transpose*`, `im2col`/`col2im`, `feature_dropout`,
  `cosine_similarity`, `pairwise_distance`, `embedding_bag`, pooling
  1d/3d/adaptive-max/fractional, upsample bicubic/linear/exact,
  `grid_sampler`/`affine_grid`, `repeat_interleave`, `broadcast_tensors`,
  `log_sigmoid`, `rrelu`, `nll_loss`).
- **zoo:** 0 opaque ops before and after — the held-out registry only
  exercises already-bound ops.

## Reverse direction — does host-tensor's growth break mlopt?

No.  Everything consuming host-tensor goes through `TENSOR_OPS` /
`tensor_op_registry()` dynamically:

- `packages/mlopt-kit` (`CatoptKit`) and `packages/mlopt-experiments` pin
  `handl-host-tensor` git `branch = "main"` and register whatever the crate
  exports — the 238-op surface is picked up automatically on next fetch.
  `term_hnl.py`'s emit table is a deliberately small fixed subset (conv+BN
  proof surface) — unaffected.
- `tests/test_kit_bridge.py`, `tests/test_contracts.py` (234 passed),
  `tests/test_torch_integration.py` (49 passed), `tests/test_e2e_pipeline.py`
  + `test_adapters.py` (11 passed) — all green under
  `scripts/pytest-capped.sh`.
- **Stale docs only:** several comments still say "18 ops" (the count was
  already wrong at 31, now 238): `mlopt-kit/src/lib.rs:17,74`,
  `mlopt-kit/Cargo.toml:41`, `mlopt-kit/README.md:19`,
  `mlopt-experiments/src/lib.rs:16`, `mlopt-experiments/Cargo.toml:51`,
  `bench/drivers/kit/term_hnl.py:6,22,127`,
  `tests/test_kit_bridge.py:25`.  Left untouched (outside this task's file
  ownership; other work landed in adjacent files) — flag for a docs sweep.

## lib-tensor parity (documented, NOT ported)

`handl-lang/lib-tensor` @ `82f6f63` is still at **31 ops** — the pre-generator
handwritten surface (`src/{codec,elementwise,matmul,conv,norm,shape_ops}.hnl`,
mirroring host-tensor's file map 1:1, "full 31-op parity" per its last commit).
Gap vs host-tensor: **207 ops** (202 implemented ports + 5 seams).

Could it consume the same generator output?  Partially:

- `tools/aten_port_list.txt` + `tools/aten_schemas.json` (3,757 schemas) are
  language-neutral — they are a ready-made work queue and provenance ledger.
- The generator itself (`tools/aten_extract.py` → `src/ops/aten_generated.rs`
  + `aten_kernels.rs`) emits **Rust**; lib-tensor's ops are **`.hnl` source**
  (the transparent spec — the whole point is that rewrites see inside the
  ops).  Consuming the generator would need a second emitter targeting `.hnl`,
  or the closures hand-written per op.
- The real constraint is **expressibility**, not coverage: lib-tensor's
  README documents what `.hnl` can/can't say (the wire-form decode,
  reserved names).  Many generated kernels (fused elementwise chains,
  tuple-returning reductions like `median`/`mode`, `searchsorted`) may not
  have honest `.hnl` spellings — the same reason host-tensor kept 5 seams.
  A lib-tensor port should reuse the *list*, not assume the *kernels* port.

## End-to-end verification

`tests/test_torch_integration.py` (4 new tests, all passing):

- `test_canon_host_tensor_aliases` — mT/swapaxes/moveaxis/ger canonicalise.
- `test_host_tensor_port_ops_lift_first_class` — a module exercising 12 newly
  bound ops lifts with **zero `opaque` nodes**, lowers via
  `ir_to_torch_module`, and matches the original **bit-exact** (diff 0.0).
- `test_host_tensor_port_scalar_attrs_canonical` — `lambd`/`dim`/`keepdim`/
  `kernel_size` land under canonical attr names, not `argN`.
- `test_host_tensor_port_search_deliver_verify` — lift → `EGraph.run
  (CATEGORICAL_RULES)` → `extract_best(flops_cost)` → deliver → verify on a
  `linear→silu→selu+logsumexp→//2→linear` module (diff < 1e-9).

Manual spot-checks (all diff 0.0): per-op export→lower round-trips for every
new binding incl. `cdist` with non-default `p=1.0` (attr recorded),
`avg_pool2d` stride/padding variants, `torch.diag` on a `(3,4)` param,
`x.diagonal()` default dims, `mT`/`swapaxes`/`moveaxis` aliases,
`linspace` dtype propagation.

## Gates run

- `scripts/pytest-capped.sh .venv/bin/python -m pytest tests/test_torch_integration.py -q` — **49 passed** (205 s).
- `tests/test_contracts.py` — **234 passed**, 102 skipped.
- `tests/test_e2e_pipeline.py tests/test_adapters.py -m "not slow"` — **11 passed**.
- `ruff check` + `ruff format --check` on `torch_bridge.py` — clean.
- `ty check` on `torch_bridge.py` — clean.

Not run: the full `scripts/gates.sh` suite (hours-scale on this host); the
touched surface is confined to `torch_bridge.py` + one test file, and the
relevant slices above all pass.

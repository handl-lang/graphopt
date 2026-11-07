# P29 — T-024: scan-carrier claims evaluate on CUDA; two ship

The T-018 audit (`triton_binding_audit_after.json`, post-T-026)
recorded **33 bind-then-raise consults**: every `applyd`-carrier
scan claim the scan tier accepted bound a generated Triton kernel,
then raised inside the referee — 32 of them the scan tier
(mixed-device `bmm`/`matmul` on a host `zeros`, broadcast shape
mismatches), plus one unrelated fp64 `pow` codegen
`CompilationError` (T-025). No scan claim had ever been priced on
this corpus.

Machine-readable rows: `triton_binding_audit_t024.json` (this run,
same driver/flags as the audit).

## Root cause — two defects, both in `genkernel`, not the harvest

The `_scan_sites` harvest is *faithful*: the `zeros` op is a real
subterm carried into the `applyd` member by the OM lifts (e.g.
GatedDeltaRule's recurrence init), and `_spec_of` correctly keeps
creator ops concrete in the spec. The failures were downstream:

1. **Device-unaware creator eval** (`_eval_spec`). Spec creator
   nodes — `("zeros", {"shape": …})` — evaluate through the ambient
   torch bindings, which mint on the process default (CPU). Under
   the referee's CUDA-bound metavar env the spelled body *and* the
   generated binding's host-hoisted subterms mixed a CPU `zeros`
   into CUDA `matmul`/`bmm` → "mat2 is on cpu". This is the same
   latent bug class `torch_bridge._randn_param` /
   `IRModule._align_to` already solve for baked tensors: a creator
   mints where the computation runs.

2. **Dead operands poisoned the flat broadcast**
   (`_unrolled_binding` / `_unrolled_triton_source`). The unrolled
   kernel loads every spec metavar *and* every registered hoist
   flat over one index space — but `_leaf_exprs` collects metavars
   from whole leaf specs, including subtrees that hoist wholesale,
   and `_hoist` registers kids of nodes that themselves hoist
   (dead `__cN`s). Those operands carry shapes foreign to the
   state index space (e.g. the full `(8,32,64)` sequence where the
   map consumes `(8,16,64)` chunks), so `broadcast_tensors`
   raised on specs the kernel denotes fine. The kernel only ever
   *loads* names surviving into the rendered step expressions —
   the broadcast set now matches that.

## Fix (`packages/mlopt-discovery/src/mlopt_discovery/genkernel.py`)

* `_eval_spec` now computes a device context (`_spec_like` — first
  non-scalar env tensor, mirroring `_const_like`) and
  `_eval_spec_dev` moves results of `_DEVICE_FOLLOWERS`
  (`zeros/ones/empty/randn/rand/full/arange`) onto it. CPU envs are
  bit-identical (the move is a no-op when devices agree); the
  spec's dtype attr still wins — only placement follows. This also
  repairs the same latent mixed-device path in `gen_bindings`
  delivery and the `handler_synth`/`prim_mint`/`shipped_primitives`
  `_eval_spec` callers.
* `_unrolled_info` computes the *referenced* operand set
  (`_name_used` — bare-token match against the rendered step
  expressions, `.`-guarded so `tl.exp`/`libdevice.tanh` members
  don't count): `info["flat"]` = referenced metavars,
  `info["computed"]` = live hoists only, `info["kernel_args"]` =
  flat + state + live hoists — the kernel signature, the flat
  loads, and the binding's broadcast all use it. `info["params"]`
  is unchanged: hoist-internal metavars stay positional binding
  args (the host subterm eval needs them) but never reach the
  kernel.

## Audit before / after (same driver, `zoo(64,32,8)`, both dtypes)

| | consults | bound | referee raised | timed | kept | scan kept |
|---|---|---|---|---|---|---|
| before (`_after.json`) | 126 | 126 | **33** (32 scan + 1 gen pow) | 93 | 10 | **0** |
| after (`_t024.json`) | 126 | 126 | **1** (the same gen_12 pow — T-025) | 125 | 14 | **2** |

Every scan consult now *evaluates*: the generated kernel verifies
against the spelled body on the real values (`verify_failed: 0`
across all 126) and is timed. Specs whose hoisted operands still
can't share one flat index space raise inside `_kernel_pays` and
decline exactly as designed ("call-time shape violations raise,
never fall back") — but on this corpus **none did**: all 36 scan
consults verified and timed.

## Do scan claims ship? Yes — the first two.

`kept_by_referee` (ratio ≥ 1.0 vs the compiled spelled body):

| workload | dtype | tag | spelled µs | kernel µs | ratio |
|---|---|---|---|---|---|
| GatedDeltaRule | fp64 | `scan_0` | 551.2 | 254.2 | **2.169** |
| RWKVMixBlock | fp64 | `scan_1` | 7454.3 | 6617.6 | **1.126** |

Both are `applyd`-carrier unrolled specs whose step expressions
are entirely host-hoisted subterms — the generated kernel fuses
the carrier step the spelled eval runs as a chain of torch ops.
The other 34 scan consults timed 0.75–0.98× and were honestly
declined by the margin gate (the unrolled kernel pays only when
the hoisted subterms are expensive enough to amortise the flat
reload). The tier is no longer decorative: it prices, verifies,
and — twice — wins.

## Verification

* `pytest tests/test_discovery_genkernel.py` — 109 passed
  (incl. the CUDA scan-binding suite).
* `pytest tests/test_discovery_play.py` — 54 passed;
  `test_shipped_primitives.py::TestSpecHelpers` + genkernel —
  112 passed.
* `ruff check` / `ruff format --check` / `ty check` on the shipped
  source — clean.
* Direct check: `_eval_spec` under a CPU env is unchanged; under a
  CUDA env a `zeros` leaf mints `cuda:0`.
* Not run to completion: `test_discovery_handler_synth.py` /
  `test_family_synth.py` / full `test_shipped_primitives.py` are
  minutes-scale synthesis files that exceeded this 11 GB box's
  memory/time under current load (SIGKILL at ~16 tests / 600 s);
  they don't exercise the changed paths beyond `_eval_spec`,
  whose CPU behaviour is identical. Flagging honestly rather than
  claiming a green full slice.

## Files

* `packages/mlopt-discovery/src/mlopt_discovery/genkernel.py` —
  the fix (uncommitted).
* `docs/reports/triton_binding_audit_t024.json` — the after-audit.
* `docs/reports/p29_scan_carrier_cuda.md` — this note.

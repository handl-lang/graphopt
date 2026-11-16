# RNG-aware verification — stochastic modules under `verify_module`

Follow-up to `tts_kokoro_r1.md` **wall A** and the rtol-floor
precedent in `verify_bf16_mem.md`.  Kokoro's `SineGen` draws
`torch.rand` (initial phase) and `torch.randn_like` (noise) *inside*
the forward; unseeded run-to-run diffs are ~6e-2, so
`lower(verify=True)` reported `verified=False` even when the
delivery was semantically exact (seeded diff 1.5e-8).  The verifier
could not distinguish "stochastic op" from "wrong kernel".

## Policy

**Seeded-eval, auto-detected — never a statistical tolerance.**
`kit_torch.report.verify_module` now:

1. Snapshots the global RNG state — CPU (`torch.random.get_rng_state`)
   plus every visible CUDA device (`torch.cuda.get_rng_state_all`).
2. Runs the verify pair unseeded exactly as before (same park
   choreography, same input clones, same `no_grad` forwards).
3. Compares the post-pair snapshot to the pre-pair one.  **Any** RNG
   consumption on either side marks the pair stochastic.
4. If stochastic and no `seed=` was given, re-runs the pair **once**
   with `torch.manual_seed(_AUTO_VERIFY_SEED)` (= 0) immediately
   before *each* side's forward — both sides draw from the identical
   stream position, so the comparison is again an **exact**
   `max_rel < rtol` equality check, bit-for-bit when the rewrite
   preserves op order.
5. `seed=<int>` forces the aligned run unconditionally (one pass, no
   detection needed) and is recorded on the report either way.
6. The caller's RNG state is **restored** afterwards (`finally`) —
   verify is RNG-neutral and never perturbs the ambient stream.

The applied seed is recorded on `VerifyReport.seeded`
(`None` = unseeded comparison, `verify_equiv`, or a deterministic
pair; an int = the aligned run's seed) for auditability — the same
role `VerifyReport.rtol` plays for the dtype floor.

### Why detection is at the RNG-state level, not the op-name level

A census of stochastic ops would have to be maintained in two
places and would still miss the cases that matter here: Kokoro's
`randn_like` survives export as an **`opaque` passthrough leaf**
(wall D of the Kokoro report) and stochasticity can hide inside
arbitrary user `nn.Module`s on the reference side.  The RNG-state
diff catches *any* consumption of the global generators — bound
creator ops (`rand`, `randn`, `randn_like` bindings), opaque leaves
replaying real torch calls, `F.dropout` in train mode, `torch.rand`
inside a custom autograd Function — with zero op-table coupling.

## Why this is principled, not a weakening

Same philosophy as the bf16 rtol floor: the gate's *form* is
unchanged; only the *alignment* is adapted to what the comparison
can resolve.

- **Deterministic modules are bit-identical to before.**  No RNG
  consumption → no re-run, `seeded=None`, same single unseeded
  comparison, same rtol/atol semantics.  The only added cost is two
  small state snapshots per verify.
- **Stochastic modules get a *stronger* check, not a looser one.**
  Unseeded comparison of two stochastic modules is noise — it fails
  ~always and proves nothing.  Aligned-seed comparison asks the
  precise question "does the rewrite preserve the computation *given
  the same random draws*?" — and answers it exactly (`max_abs ==
  0.0` on the Kokoro decoder's seeded arm: 1.49e-8 fp32 noise).
- **Semantic errors still fail.**  `x + rand_like(x)` vs
  `x * rand_like(x)` under the same seed fails at `max_rel ≈ 0.47`.
  A rewrite that *reorders or drops* a stochastic draw also fails —
  correctly, since that IS an observable behavioural difference
  (pinned by `test_verify_module_stochastic_mismatched_consumption_fails`).
- No new tolerance parameter exists; `rtol`/`atol` and the
  dtype-resolution floor compose unchanged.

## Where it lives

Only `kit_torch.report.verify_module` — so every torch verify path
inherits it with **no eqsat change**:

- `TorchSink.verify` (`adapters.py`) delegates and additionally
  accepts `seed=` for callers that want to force/choose the seed;
- the orchestrator's `lower(verify=True)` /
  `_optimize_compositional` / autotune / diagram gates call
  `sink.verify(...)` without `seed` → the **auto-detect** path —
  Kokoro's `lower(verify=True)` now verifies stochastic deliveries
  honestly;
- `composer.py`'s joint-pair verify calls `verify_module`
  directly → same auto-detect.

`mlopt_onnx.report.verify_module` is a separate numpy/onnxruntime
implementation — untouched (ORT-side RNG alignment is out of scope).

## Honest limits

- **Global generators only.**  `torch.manual_seed` aligns the CPU
  and all CUDA device generators.  Randomness drawn through an
  explicit `torch.Generator` object (e.g. `torch.rand(...,
  generator=g)`), numpy RNG, `random`/`secrets`, and non-CUDA
  accelerator RNG (MPS/XPU have their own generators `manual_seed`
  does not reach) are **out of scope** — such modules stay
  unverifiable by parity, exactly as before.
- **Stream-order sensitivity.**  The seeded check certifies "same
  RNG stream → same output".  It requires the two sides to consume
  the stream *identically* — same draws, same order, same dtypes.
  This held for Kokoro (the `rand`/`randn_like` leaves survived as
  opaque ops in order; seeded diff 1.5e-8).  A law that legitimately
  reorders stochastic draws would fail — the honest answer, since
  draw order is observable.
- **`verify_equiv` cannot reseed.**  It compares already-computed
  outputs; callers holding tensors must align RNG themselves before
  evaluating.
- **Detection is consumption, not dependence.**  A module that draws
  RNG its output ignores (discarded `torch.rand`) triggers one
  extra seeded re-run — harmless (still passes, `seeded` recorded).
  The converse — stochasticity with zero global-RNG consumption —
  is the explicit-Generator limit above.
- **`seed=0` is arbitrary but fixed.**  Any constant works; the
  check is stream *alignment*, not stream quality.  The constant
  keeps the auto path reproducible.

## Tests (`tests/test_reports.py`)

- `test_verify_module_stochastic_auto_seeded_passes` — identical
  stochastic pair verifies, `max_abs == 0.0`, `seeded == 0`;
- `test_verify_module_stochastic_semantic_error_still_fails` —
  `+rand_like` vs `*rand_like` fails under seeding (`max_rel > 0.1`);
- `test_verify_module_explicit_seed` — `seed=` forces the aligned
  run and is recorded, deterministic modules included;
- `test_verify_module_deterministic_unaffected` — no consumption →
  `seeded is None`, ambient RNG untouched;
- `test_verify_module_rng_neutral_for_stochastic` — RNG state
  restored after a stochastic verify;
- `test_verify_module_stochastic_mismatched_consumption_fails` —
  a rewrite that drops a draw fails honestly.

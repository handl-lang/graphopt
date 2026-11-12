# p34 — seq-scan claim path revived with a correct spelled denotation

Date: 2026-11-12.  Scope: `mlopt_discovery.genkernel` (new
`seq_scan_denotation` / `handler_denotation`, `gen_bindings`,
`hybrid_sink`, `profitable.denotation_for`), `mlopt_discovery.play`
(`_claim_measurable`, `_claim_cost_gate`, `_zoo_case`,
`_claim_denotation_for`).  Motivation: the T-010 audit disabled
sequence-form (`kind="seq"`) scan claims at both cost gates because
no correct spelled fold denotation existed — the ambient
`applyd(aff_diag(a,b),h)` binding is ONE map application
(`a⊙h+b`), `_IR_TO_TORCH` has no fold op, and `gen_sink` would have
silently delivered map semantics for a claim whose minted op
declares the fold.

## What was actually disabled

- `_claim_measurable` (`play.py`) returned False for any
  `scan_*` handler whose pattern was a seq spec
  (`gk._seq_scan_spec`), at BOTH `_zoo_case` intake and
  `_claim_cost_gate`.  Seq specs were dropped outright — unpriced
  AND undeliverable, because the seated `gen_sink`'s
  `gen_bindings` evaluated the spec through `_IR_TO_TORCH` = the
  map.
- `hybrid_sink` already refused the wrong fallback: seq specs were
  excluded from `gen_bindings` and stayed unbound when the Triton
  codegen declined.
- The seq form itself is minted by `scan_handlers` on raw
  `applyd(aff_diag(a,b),h)` terms (single aff_diag leaf, no
  `kind` attr → `_scan_kind` resolves `"seq"`).  `_scan_sites`
  pins `{"kind": "unrolled"}` on every harvested spec, so the zoo
  census stays seq=0 / unrolled=87 — no collector produces the seq
  form today; the revival is machinery, not a new claim source.

## The denotation

The minted `scan_*_k` op's declared semantics is the fold
`h_{t+1} = a_t ⊙ h_t + b_t` walked along the step axis.  The spec
tuple `("applyd", ("aff_diag", a, b), h)` denotes the MAP — the
fold is a property of the minted op, so the denotation cannot live
in the spec or in `_IR_TO_TORCH`.  It lives in genkernel as a
plain-torch callable:

- `seq_scan_denotation(spec, *, axis=None)` → `fn(*metavar_vals)`:
  broadcasts the sequence operands (`torch.broadcast_tensors`,
  same regime as `_seq_args`), resolves the step axis through
  `_seq_axis` (explicit axis or derived from the state shape per
  call), then walks the axis evaluating the `aff_diag` operand
  sub-specs per step through the ordinary bindings and folding
  `acc = a_t * acc + b_t` — the same recurrence, in the same
  order, the Triton kernel runs.  Returns `None` for non-seq
  specs; call-time shape violations raise (same contract as the
  Triton binding).
- `handler_denotation(h)` → the per-handler resolution: seq spec →
  the fold callable at the handler's axis (`h["axis"]` → spec attr
  → `-2`, the same chain `scan_triton_bindings` applies, so the
  referee's spelled arm and the delivered kernel agree on the step
  axis); everything else → `None` (its spelled eval already IS the
  denotation).

Delivery-path independence (the project's core rule — verify
against denotation, not incumbent):

- `gen_bindings` binds seq-scan kernel names to the fold denotation
  (uncompiled even under `compile_kernels` — a T-step python loop
  gains nothing from dynamo and the Triton tier is the fast path).
  `gen_sink` therefore delivers the declared semantics on ANY
  device — the silent map delivery is gone.
- `hybrid_sink` now feeds ALL handlers to `gen_bindings` (was
  `plain` minus seq); `scan_triton_bindings` still overrides
  wherever the generated kernel binds, so CUDA delivery is the
  Triton fold and declined codegen falls back to the same
  denotation serially — never the map.
- `profitable` gained `denotation_for(name, h, vals)`: when it
  returns a callable, that callable is BOTH the spelled arm AND the
  only honest fallback — if no codegen tier binds, delivery IS the
  denotation, so the claim declines outright rather than timing
  fold-vs-map or fold-vs-fold.  play passes
  `_claim_denotation_for` → `gk.handler_denotation`.

Also fixed a latent soundness bug found while mapping the path:
`_scan_kind` accepted `{"kind": "seq"}` pinned on a MULTI-leaf
`affd_compose` tree, where `_scan_info` would silently keep only
`steps[0]` — a different recurrence than spelled.  The arity guard
now declines (`scan_spec` → `None`).

## Measured

Denotation verified against the `_spelled_scan` oracle and against
per-step sub-spec evaluation (sigmoid/mul operand bodies), shared
broadcast operands, explicit/derived axis — fp64, atol/rtol 1e-12,
CPU (no Triton needed).

Referee measurement, synthetic seq claim `applyd(aff_diag(A,B),H)`,
fp64 CUDA (RTX 2050), `profitable(reps=30,
kernel_for=_triton_kernel_for, denotation_for=_claim_denotation_for)`:

| shape | spelled fold | Triton kernel | ratio | verdict |
|---|---|---|---|---|
| (2, 32, 16) | 1942 µs | 570 µs | 0.294 | KEPT |
| (8, 128, 64) | 7122 µs | 646 µs | 0.091 | KEPT |

Same claim with `kernel_for=None` → declined (delivery IS the
denotation — correctly unprofitable).  So a seq scan claim now
prices AND pays through `_kernel_pays`/the gate wherever the
Triton fold binds; on CPU it is measurable-but-unkeepable by
construction (nothing faster than the denotation exists there).

**Zoo payoff: none — honestly.**  `_scan_sites` pins
`kind="unrolled"` on every minted spec (census unchanged: seq=0,
unrolled=87 across the 68-workload zoo), so no real corpus produces
a seq claim for the revived path to price.  The tier is now
correctly priced rather than silently wrong: claims stop being
dropped-by-fiat and start being measured-against-denotation the day
a collector mints one.

## Tests

- New `TestSeqScanDenotation` (8 tests): oracle parity across
  layouts/sub-specs/broadcast/axis, non-seq declines incl. the
  compose-pinned-seq arity guard, `gen_bindings` fold on CPU,
  `profitable` denotation needs-a-kernel + kept-at-margin.
- `test_claim_cost_gate_filters_measurable_tiers` reworked: the seq
  scan is now OFFERED to the referee (was dropped outright) and the
  `denotation_for` hook is asserted to resolve the fold for it.
- `test_seq_spec_never_falls_back_to_spelled` →
  `test_seq_spec_falls_back_to_the_fold_denotation`: declined Triton
  now binds the fold (verified vs `_spelled_scan`), not unbound.
- Gates: `tests/test_discovery_genkernel.py` 141 passed,
  `tests/test_discovery_play.py` 58 passed,
  `tests/test_shipped_primitives.py` 32 passed — all `-m "not slow"`
  under `scripts/pytest-capped.sh`.  `ruff check`/`format`/`ty`
  clean on the touched package files.

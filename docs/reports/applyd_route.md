# applyd route — chain → scan kernel at delivery

**Date:** 2026-11-14 · **Builds on:** `t034_module_costing.md` §2/§5.2 —
the seq/seqm collectors mint and `profitable` prices scan claims
honestly, but the intake IR has zero `applyd` nodes so
`MetaArena._claim`/`_claim_tags` hard-decline them
(`meta_arena.py` `_concrete_occurs`): the priced kernel had no route
to the delivered term.

## What was missing, precisely

The claim path needs an `applyd(aff_diag(a,b), H, {kind: seq|seqm})`
*member* in the e-graph for the minted fold rule
(`applyd-spec → claim_scan_i_0`) to fire.  Raw-IR zoo workloads spell
the recurrence as unrolled `add(mul/select…)` chains — the `applyd`
carrier exists only inside the collectors' throwaway reasoning.  Two
further constraints ruled out the obvious fixes:

- **A bank law cannot mint the carrier.**  The seq/seqm fold is a
  *variable-length* collapse (`T` unrolled steps → one op over
  step-axis operands); no fixed-arity `Rewrite` LHS spells it, and
  `DEFAULT`'s rules contain no scan law.
- **Injecting a bare `applyd` member is unsafe.**  The ambient
  `applyd` binding (`torch_bridge._IR_TO_TORCH`) computes
  `f[0] * h + f[1]` — ONE map application, ignoring the
  `kind="seq"` attr — and `is_scan_apply_term`
  (`mlopt_carriers/scan_lower.py:200`) does not read attrs either,
  so a seq-form `applyd` reaching delivery would mis-lower through
  `BatchedScanModule`.  Only the minted `scan_<i>_k` op carries the
  fold denotation (`genkernel.handler_denotation` /
  `seq_scan_denotation` / `seqm_denotation`) and the Triton binding.

## The route that landed: deliver-time chain → `scan_<i>_k`

The collectors already prove the fold (identical deindexed
skeletons step-to-step, select index == chronological position,
`shape_of`-checked step axis).  The missing piece was only the
*rewrite route*, so the site record now carries the chain root and
delivery applies the verified fold directly:

1. **`play._seq_scan_sites` / `play._seqm_scan_sites`** — emitted
   site records gain `"chain": root`, the unrolled chain term the
   fold denotes (`play.py`, both collectors).
2. **`genkernel._site_parts`** — merges `"chain"` verbatim into the
   minted handler entry, beside `leaf_env`/`axis`/`axes`
   (`genkernel.py:~2921`).  Handle cells store `dict(handler)`
   shallowly, so the term survives the cellstore round-trip
   in-memory (same as `leaf_env`'s terms already do).
3. **`MetaArena.deliverable`** — routes the *extracted* term through
   **`_route_scan_chains(term, handlers, eg)`**
   (`meta_arena.py:~1083`) *before* `_deliverable` resolves
   interpretations/definitions.  For every handler carrying
   `"chain"` + `"leaf_env"` it pre-builds
   `scan_<i>_k(*[leaf_env[m] for m in args])` and rewrites matching
   subterms to it.

### Why e-class matching, not term identity

Saturation rewrites inside the chain freely: on GatedDeltaRule the
extracted representative of the chain's e-class was
`claim_synop2_d46808a970_0` — a *declared fold op* holding the whole
class — not the collector's `add(...)` object.  `_route_scan_chains`
therefore resolves each chain's e-class once via
`eg._class_of_term(chain)` and matches every extracted subterm by
class id (`eg._class_of_term(t)`), with `is`/`==` behind an op-name
pre-filter as the cheap path and a structural `==` fallback for
foreign terms.  Every class member denotes the chain's value (the
e-graph only merges verified rewrites), and the collector proved the
chain ≡ fold — so the replacement is sound for *any* member
extraction picked, folded or spelled.

### Honesty properties

- **Verified, not syntactic.**  The rewrite fires only for sites the
  collectors proved; the delivered module is still checked by
  `sink.verify` on real values, and `profitable` already measured
  the kernel against the registered fold denotation — the route
  ships exactly the callable the referee timed (Triton where the
  `scan` tier binds, the spelled fold denotation otherwise).
- **Referee-gated.**  Routes exist only on handlers in the board's
  handler table — i.e. claims that survived `profitable`.  A dropped
  claim carries no route; its chain delivers spelled.
- **Fallback, never wrong.**  A chain folded past recognition, a
  class that never reached extraction, or an incomplete `leaf_env`
  all leave the spelled chain — the route declines rather than
  guesses (`KeyError`/`TypeError` on arg instantiation → skip).
- **Cost model untouched.**  Extraction still prices the spelled
  chain; the route is a delivery transform, so certificates,
  `claim_tags`, and `TRITON_REFEREE_LOG` semantics are unchanged.
  `claim` actions for `scan_*` remain honestly unavailable
  (`_concrete_occurs` still declines — the claim path genuinely
  cannot fire; the deliver path is the route).

## End-to-end proof (zoo(16, 8, 2), fp32, CUDA host)

`_zoo_case` → `_gen_board` → `_probe_play` → `board.deliverable` →
`hybrid_sink.lower` → `sink.verify`, serial:

| workload | site kind | delivered op | binding arm | verify |
|---|---|---|---|---|
| GatedLinearAttention | `seq` (`scan_16`) | `scan_16_k` | scan tier bound | pass, max_abs 2.4e-07 |
| GatedDeltaRule | `seqm` (`scan_8`) | `scan_8_k` | scan tier bound | pass, max_abs 1.4e-07 |
| RealGatedLRU | `seq` (`scan_16`) | `scan_16_k` | scan tier bound | pass, max_abs 1.2e-07 |

`_binding_attempts` confirms the `scan` tier's generated Triton
kernel is the bound arm for both seq and seqm specs on this box
(CPU delivery falls back to the fold denotation — slow but correct).

## The pinned-test conflict — resolved without touching it

`tests/test_discovery_genkernel.py::test_claim_cost_gate_filters_
measurable_tiers` asserts an absent-`applyd` `scan_0` reaches
`profitable` and stays in `gen_handlers`.  That contract is
**unchanged**: this route does not drop unroutable claims at the
offer gate (the report's §5.1 alternative) — it makes them
deliverable downstream.  `profitable`, `PROFITABLE_LOG`, and
`TRITON_REFEREE_LOG` keep their full audit surface; the test passes
as written (verified: 1 passed).

## Known residuals

- **Spec-dedup vs chain-dedup.**  The collectors emit one site per
  distinct *spec*; two different chains folding to the same spec
  skeleton share one handler/route (first chain wins).  A second
  chain with an identical spec but different operands delivers
  spelled — a missed optimisation, never incorrectness.
- **Operand spelling.**  Replacement args are the `leaf_env` terms
  (raw intake subterms), not the extraction's possibly-folded
  representatives of those operand classes — correct values, possibly
  a less-folded spelling inside the kernel's inputs.
- **Serialization.**  `"chain"` is a term object inside the handler
  dict — same non-serializable posture `"leaf_env"` already has;
  stores are only serialized via `to_data`, which this path does not
  call.
- **`unrolled`-kind `_scan_sites` records carry no `"chain"`** —
  their carrier member exists only in the collector's throwaway
  e-graph, so there is no raw-LHS to route; they remain claim-dead
  as before (the seq/seqm collectors are the paying sites).

## Files changed

- `packages/mlopt-discovery/src/mlopt_discovery/play.py` —
  `"chain": root` on seq/seqm site records; two drive-by fixes of
  pre-existing uncommitted-WIP lint/type errors the gates caught
  (`zip(..., strict=True)` in `_seq_bcast`, an unreachable-but-
  un-narrowed `state_sh is None` guard in `_seq_site`).
- `packages/mlopt-discovery/src/mlopt_discovery/genkernel.py` —
  `_site_parts` merges `"chain"` into handler entries.
- `packages/mlopt-discovery/src/mlopt_discovery/meta_arena.py` —
  `deliverable` split into wrapper + `_deliverable` worker;
  new `_route_scan_chains`.
- `tests/test_discovery_play.py` — 4 new tests in
  `TestSeqScanSites` (route data on the handler, term-level
  delivery, e-class-member matching, incomplete-binding decline).

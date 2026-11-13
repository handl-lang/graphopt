# unrolled route — chain → `scan_<i>_k` for `kind="unrolled"` sites

**Date:** 2026-11-13 · **Builds on:** `applyd_route.md` — the
chain→scan_k deliver-time route fired only on `seq`/`seqm` site
records carrying `"chain"`; the T-034 residual noted `unrolled`-kind
`_scan_sites` records carried no route ("no raw LHS exists").

## What the unrolled sites actually are

`play._scan_sites` (`play.py:~740`) is a different collector from
`_seq_scan_sites`/`_seqm_scan_sites`:

- It runs a **throwaway saturation** of the raw term under
  `OM_LAWS + OM_CHUNKIFY + SPLIT_SCALAR_RULES`
  (`max_iterations=10`, `max_nodes=60_000`, chunkify budgets 20/20),
  then harvests every `applyd` enode in every e-class.
- `_carrier_member` decodes each `applyd` node preferring
  `aff_diag`/`affd_compose` children; `_spec_of` +
  `_spec_leaf_env` turn the decoded map into a spec whose metavars
  bind to the decoded leaf subterms, and the state arg abstracts to
  a fresh `"X<n>"` metavar recorded in `leaf_env`.
- The emitted spec is pinned `{"kind": "unrolled"}` — the
  *ambient* reading: `applyd(aff_diag/affd_compose tree, h)` is a
  **fixed composition** of map applications (`affd_compose(f, g)`
  applies `g` first), NOT a fold over a step axis.  `scan_spec`
  (`genkernel.py:1739`) accepts it and `scan_triton_bindings` lowers
  it through `_unrolled_binding`/`_unrolled_triton_source`.

The seq/seqm collectors, by contrast, walk raw
`add(mul/select…)`/`add(s, f(s))` spines and mint `kind="seq"` /
`kind="seqm"` fold specs whose operands carry a step axis — with
`"chain": root`, the raw chain term itself.

## Why they lacked `"chain"` — a collector gap, not a missing LHS

The old docstring claimed "the member the spec came from exists only
inside this throwaway graph, so no raw-IR match can ever rebind it".
That is true of the **`applyd` enode**, but not of its **e-class**:
`decay_state_lift` (`om.py:3853`) introduces `applyd` by rewriting a
spelled `add(mul(a,h),x)`/`add(x,mul(a,h))` subterm, and the compose
laws merge carriers upward — so the class an `applyd` node sits in
almost always also contains a **raw-IR subterm of the intake term**
(the spelled chain the OM laws folded).  The collector simply never
recorded it.  Measured on the `TestScanSites._folded_term` fixture
(`add(mul(pow(g,8),b0), mul(pow(g,4),b1))`): both `applyd` members
decode into e-class 9, and the whole raw term maps to class 9 via
`eg._class_of_term`.

So option (a) applied: the collector can record a route LHS, and the
existing machinery (`_site_parts` → handler `"chain"` →
`_route_scan_chains`) consumes it unchanged.

## The change

`play._scan_sites` now builds `raw_by_class` once per call: a
pre-order walk of `term` mapping every subterm through
`eg._class_of_term(t, memo)` (shared memo; `sys.setrecursionlimit`
guard added, same as the seq collectors) to the **outermost** raw
subterm realising each e-class.  Each emitted site record gains
`"chain": <subterm>` when the applyd node's class (`eg.find(cls.id)`)
has a raw member; classes synthesized wholesale record no key and
deliver spelled exactly as before.  `_site_parts`
(`genkernel.py:~2963`) already merges `"chain"` verbatim into the
handler entry — no downstream change was needed.

### Soundness

Identical argument to the seq/seqm route: the applyd member and the
raw subterm share an e-class in a graph closed under *verified* OM
laws, so they denote the same value; the minted `scan_<i>_k` denotes
the applyd member (the ambient `applyd` binding
`f[0]*h + f[1]` over the composed `(a,b)` pair, the generated
unrolled Triton kernel, and the spelled `_eval_spec` fallback all
agree).  `_route_scan_chains` still matches by **board-e-graph
class**, so whatever member extraction picked — folded, reassociated,
or spelled — is replaced by the kernel call.  A chain that never
reached the board graph, or a handler dropped by `profitable`,
leaves the spelled form: decline, never wrong.

## End-to-end proof

RealGatedLRU, `zoo(16, 8, 2)`, fp64, CUDA host —
`TorchSource.to_ir` → `_all_scan_sites` → `scan_handlers` →
`MetaArena(DEFAULT + ambient + minted)` → claims → `saturate` →
`extract` → `board.deliverable` → `hybrid_sink.lower` →
`sink.verify`, serial:

- `_scan_sites` mints **16 unrolled sites, all 16 with `"chain"`**
  (plus one `seq` site from `_seq_scan_sites`, as before).
- With the full minted table seated (the `profitable` gate bypassed
  for measurement), the delivered term contains **8 unrolled
  `scan_0_k`…`scan_14_k` calls** — every one Triton-bound
  (`scan_triton_bindings` accepts all 17 specs on this box).
- `sink.verify`: **passed, max_abs 4.4e-16**.

A second check on a synthetic `_gen_cases` chainfuse term (offer
gate bypassed): the route rewrote the whole tail to `scan_0_k` and
the delivered module verified at max_abs 3.8e-6 (fp32, CUDA) — the
route is sound even where it *shadows* a `gen_*` claim, because both
denote the same e-class value.

## Interaction with the offer gate — the honest caveat

`profitable` is still the only thing deciding whether an unrolled
route exists at all.  On every zoo workload probed
(ChunkedRetention, DeltaRuleBlock, MambaSelectiveScan,
GatedDeltaRule, ChunkedSSD, RealGatedLRU — fp32) **every unrolled
`scan_*` claim lost the measured-cost gate**; only `seq`/`seqm`
handlers survived.  That is expected economics: an unrolled spec's
spelled eval *is* already the carrier semantics (`a⊙h+x` per compose
leaf — a handful of elementwise ops), so a fused kernel has little to
win unless the compose tree is deep and the tensors large.  The
route therefore fires in production only when an unrolled claim
measurably pays — rare today, but the machinery is now correct for
when it does, and for the synthetic/carrier-IR cases
(`_scan_carrier_ir`-style terms that already spell `applyd`) where
sites are minted deliberately.

## Test-pin conflict — resolved by scoping, not weakening

Three tests bypass `_claim_cost_gate` on purpose to pin
claim→deliver *machinery* (offer-policy-independent):

- `test_discovery_play.py::TestFusionSites::test_chainfuse_claim_binds_gen_over_mm`
- `test_discovery_play.py::TestFusionSites::test_twosite_multi_claim_composes`
- `test_discovery_genkernel.py::TestGenDomain::test_claim_gen_delivers_verified`

With the gate bypassed, minted-but-unprofitable `scan_*` handlers sit
in the board table, and their new `"chain"` routes legitimately
rewrite the extracted term to `scan_*_k` at delivery — masking the
`gen_*_k` composition the tests assert (delivered strings showed
`scan_0_k …`, `scan_1_k(x, x)`).  Numerically the reroute is correct
(verified above), and the exposure is identical in kind to what seq
routes already had — these terms simply never minted seq sites.  The
three tests now filter `scan_*` entries out of `ex["gen_handlers"]`
before `_gen_board`, restoring their original scope (gen/prim claim
composition); scan routing has its own pins in `TestScanSites` /
`TestSeqScanSites`, plus the new
`test_scan_sites_record_chain_route` asserting `"chain" is term` on
the folded fixture.

## Residuals

- **Referee-gated, same as seq.**  Routes exist only for handlers in
  the board's table; an unrolled claim `profitable` drops carries no
  route.  Routes also fire for *unclaimed* table handlers — the
  designed posture (`applyd_route.md`), unchanged here.
- **CPU delivery.**  `scan_*_k` bindings are Triton launchers;
  `_unrolled_binding` raises on non-CUDA tensors.  This is
  pre-existing for seq routes too — delivery of any scan op assumes
  the CUDA host the reports measure on.
- **Spec-dedup vs chain-dedup** — unchanged: one site per distinct
  spec; multiple `applyd` members in one class share the class's
  single recorded chain (first spec wins).
- **`leaf_env` terms are throwaway-graph decodings** — they may
  contain carrier ops (`applyd`/`aff_diag`/`affd_compose`) or OM-law
  products (`split`, `om_*`).  They already had to evaluate under
  `profitable`'s bindings for the claim to survive, and the ambient
  table binds the carrier family, so delivered args lower; a spec
  whose env cannot evaluate never survives the referee anyway.
- **Serialization** — `"chain"` is a term object in the handler dict,
  same non-serializable posture `"leaf_env"` already has.

## Files changed

- `packages/mlopt-discovery/src/mlopt_discovery/play.py` —
  `_scan_sites` records `"chain"` (outermost raw-IR subterm per
  e-class); recursion-limit guard added.
- `tests/test_discovery_play.py` — new
  `TestScanSites.test_scan_sites_record_chain_route`; two
  fusion-composition tests exclude the incidental `scan_*` tier.
- `tests/test_discovery_genkernel.py` —
  `test_claim_gen_delivers_verified` excludes `scan_*` likewise.

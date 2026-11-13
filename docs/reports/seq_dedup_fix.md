# Seq-scan site dedup fix — same-spec chains over different operands

Follow-up to `docs/reports/zoo_corpus_expansion.md` (the
`BidirectionalScan` row: "1 (of 2 chains)").  Confirmed real
coverage bug, fixed.

## The bug

`_BidirectionalScan` unrolls TWO same-shaped opposite-direction
recurrences — `h = a_t ⊙ h + x_t` forward over `x`, and the same
skeleton over `flip(x)` with a different decay map — summed at the
output.  `_seq_scan_sites` minted only ONE seq site: the second
chain's fold was never offered to the referee and never routed at
delivery.

Root cause: **two dedup layers compared spec tuples only, and the
spec tuple does not see `leaf_env`.**

1. `play._seq_scan_sites` (and `_seqm_scan_sites`, `_scan_sites`)
   deduplicated candidates with `spec == s["spec"]`.  `_seq_site`
   mints metavar names fresh per chain (`Q1`, `Q2`, `H`, …), so two
   chains with identical skeletons produce *identical* spec tuples —
   `('applyd', ('aff_diag', 'Q1', 'Q2'), 'H', {'kind': 'seq',
   'axis': 1})` — while `leaf_env` binds `Q2` to `x` in one and to
   `flip(x)` in the other.  The fold's denotation is spec **plus**
   the `leaf_env` operand binding, so these are two distinct folds;
   the spec-only compare collapsed them.

2. `genkernel.scan_handlers` deduplicated again via
   `_canon_taken(pattern, out, skip)` — canon-concrete pattern
   equality, again blind to `leaf_env`/`chain`.  Even with the
   collector fixed, the second site would have been swallowed here
   (`scan_<i>` naming itself was never the problem —
   `tag = f"scan_{len(out)}"` extends cleanly).

## Reproduction (before fix)

`_BidirectionalScan(d=16, t=8)` lifted through `TorchSource`:

- `_seq_scan_sites(root)` → **1** site
  (`leaf_env`: `Q1→sigmoid(dec_f(x))`, `Q2→x`, `H→zeros`)
- instrumenting the candidate walk shows **56** spine candidates;
  exactly **two** full-length chains mint valid sites — candidate
  24 (forward, `Q2→x`) and candidate 52 (backward, `Q2→flip(x)`)
  with byte-identical spec tuples.  The second was dropped at the
  spec-equality dedup.

## Fix

- `packages/mlopt-discovery/src/mlopt_discovery/play.py`
  - `_scan_sites` (~line 799): dedup moved after record assembly —
    `if rec in out` compares the whole `{"spec", "leaf_env",
    "chain"?}` record (terms are hash-consed with structural
    `__eq__`, so dict equality is the identity test).
  - `_seq_scan_sites` (~line 1935) and `_seqm_scan_sites`
    (~line 2087): same — the record `{spec, leaf_env, axis,
    (axes,) chain}` must match in full to dedup.
- `packages/mlopt-discovery/src/mlopt_discovery/genkernel.py`
  `scan_handlers` (~line 3007): canon dedup replaced by
  `(canon, site-record)` pairs.  `covered`/`skip` stays
  canon-keyed (a shipped handler covering the spec still suppresses
  minting, as before); minted entries dedup only when BOTH the canon
  AND the `_site_parts` extra record (`leaf_env`, `chain`, `axis`,
  `axes`) are equal.  Plain spec/term items carry `extra == {}` and
  dedup on canon exactly as before — `test_non_scan_and_dupes_skipped`
  unchanged.

`scan_<i>` indexing needed no change: `scan_0`/`scan_1` mint
distinct kernel names (`scan_0_k`, `scan_1_k`) with identical Triton
bodies — mildly wasteful, honest, and the simplest correct shape.

## Verification

- `_seq_scan_sites` on `_BidirectionalScan` IR → **2** sites, same
  spec, `leaf_env` differing (`Q2→x` vs `Q2→flip(x)`).
- `scan_handlers` → `scan_0`, `scan_1`, each carrying its own
  `chain`/`leaf_env`.
- Each handler's `handler_denotation` (the registered seq fold)
  reproduces **its own** chain's value (`assert_close`, fp64) —
  verified for both sites.
- `MetaArena(...).deliverable(root)` rewrites BOTH chains:
  `scan_0_k` ×1, `scan_1_k` ×1 in the delivered term — the
  `_route_scan_chains` e-class matching distinguishes the two
  `add`-headed chains by class, no collision.
- `_seqm_scan_sites` / `_scan_sites` (unrolled carriers) got the
  same whole-record dedup; no seqm/carrier instance of the bug was
  found in the zoo sweep, but the collapse path was identical.

## Tests added

- `tests/test_discovery_play.py::TestSeqScanSites`
  - `_two_chains` helper — two same-spec recurrences over disjoint
    vars, summed (the bidirectional shape at toy scale).
  - `test_seq_sites_mints_each_same_spec_chain` — 2 sites, equal
    specs, different `leaf_env`, distinct `chain`s.
  - `test_two_chain_folds_each_equal_their_chain` — each fold's
    denotation equals its own chain.
  - `test_two_chain_handlers_route_both` — `scan_0`/`scan_1` mint
    and both `scan_*_k` calls appear in `deliverable`.
- `tests/test_discovery_genkernel.py::TestScanHandlers`
  - `test_same_spec_different_site_records_mint_both` — site-level
    dedup contract, incl. identical records still collapsing.

## Gates

- `scripts/pytest-capped.sh .venv/bin/python -m pytest
  tests/test_discovery_play.py -q -k "seq or scan"` → 43 passed
- `tests/test_discovery_play.py -m "not slow"` → 96 passed
- `tests/test_discovery_genkernel.py -m "not slow"` → 158 passed,
  1 skipped
- `ruff check` + `ruff format --check` on both edited package files
  → clean; `ty check packages/mlopt-discovery` → clean.
  (`tests/` is outside the repo's ruff gate; the files carry
  pre-existing lint noise regardless.)

## Notes / residual

- Cost: each distinct site now pays one referee measurement and one
  generated kernel.  `_SEQ_SITE_CAP = 16` still bounds the seq/seqm
  tiers; `_scan_sites` (unrolled carriers) has no cap but was
  already bounded by distinct saturated members — the relaxation
  only admits records that were previously *wrongly* merged.
- `BidirectionalScan`'s backward chain reads `flip(x)` as its
  sequence operand — the fold is honest because the selects walk
  the flipped tensor in chronological order; the operand handed to
  the kernel is the flipped tensor itself, so no flip-aware kernel
  semantics are needed.

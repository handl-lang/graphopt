# model_corpus t5 bound — `_seq_scan_sites` capped (t5_r1 wall 4)

Date: 2026-11-15.  Fixes the host-RAM OOM `docs/reports/t5_r1.md`
wall 4 isolated: `model_corpus` intake on `google-t5/t5-small` died in
`play._seq_scan_sites` — the `spec_cap` bound added by
`model_corpus_scaling.md` reached the `gen`/`reduction` collectors
but not the scan-tier collectors.

## Root cause (measured, not assumed)

Stage isolation on the real `corpus_encoder()` term
(`term_expansion=292172` → `large_term`, auto `spec_node_cap=512`):

- `_seq_spines` enumeration: **fast** — 314 distinct nodes, 24
  candidates, ~0.01 s (memoised + `_SEQ_SPINE_CAP` already bound it).
- `_seq_site` per candidate: 22 of 24 decline in <2 s; **two
  candidates take ~57 s each** and transiently allocate **~2.5 GB**
  (RSS oscillates 0.95 → 3.5 GB → 1.1 GB per candidate — the "never
  returns" was the stage watchdog, not a literal infinite loop; the
  unbounded run would have finished in ~4–5 min at >3.5 GB peak, and
  the corpus run died earlier under `earlyoom` at 5–7.7 GB).
- cProfile on the worst candidate: **24.5 M `_seq_deindex` calls,
  102 s** — `_seq_deindex` spells step operands as *trees* (no
  memo — a shared subterm is spelled once per occurrence, exactly
  like the spec it builds), so a shared-DAG operand expands
  exponentially in its distinct-node count.  The same unguarded
  `shape_of` calls inside the pullout lifts (`_seq_view_lift` /
  `_seq_projection_lift` / `_seq_commute_lift` /
  `_seq_concat_lift`) walk the same tree expansion per call.

## What changed (`play.py`, `model_corpus.py`)

All bounds default to `None` = historical unbounded zoo behaviour;
only terms whose expansion exceeds `_LARGE_TERM_EXPANSION` get the
`_AUTO_SPEC_NODE_CAP=512` auto bound — the *same* knob
`model_corpus._probe_knobs` already resolves, now threaded through
`_mint_tiers` → `_all_scan_sites` → `_seq_scan_sites` /
`_seqm_scan_sites` / `_cumsum_sites`.

- **`_seq_deindex(..., spec_cap=, _budget=)`** — a per-top-level-call
  visit budget (`[spec_cap]`); every spelled node decrements it and
  exhaustion raises `_SeqSpecOverflow`.  Each deindexed step skeleton
  (`a_t`, `b_t`) is bounded to `spec_cap` spelled nodes — the direct
  analogue of `spec_node_cap` on `gen`/`reduction` specs.
- **`_seq_shape(t, cap)`** — `shape_of` guarded by the new
  **`_seq_expansion(t, cap)`** oracle: exact tree-expansion recurrence
  with early exit at `cap+1` (ancestors can only add more, so the
  root's bound is decided early — `shape_of` itself walks this same
  expansion).  Lifts treat over-cap exactly like `INVALID` (decline
  the lift, not the site); `_seq_site`/`_seqm_site`/`_cumsum_sites`
  operand checks raise/skip+count instead.
- **Honest skip counts** — `_SeqSpecOverflow` is caught per candidate
  in the collectors and counted in a `stats` dict under
  `seq_sites_over_cap` / `seqm_sites_over_cap` /
  `cumsum_sites_over_cap`; `_mint_tiers(..., stats=)` aggregates,
  `_case_tail` seats it on `ex["site_stats"]`, and `model_corpus`
  reports `rep["site_stats"]` (per-board, parallel to `boards`) plus
  a CLI `site skips over spec_cap:` line.  Nothing is silently
  dropped; nothing hangs.
- Lift helpers (`_seq_view_lift`, `_seq_projection_lift`,
  `_seq_commute_lift`, `_seq_concat_lift`, `_seq_lift_finish`) take an
  optional `cap` — all default `None`, all existing call sites pass
  it through only when set.

## Measured before / after (t5-small encoder, fp32, host RAM)

| stage | before | after |
|---|---|---|
| `_seq_scan_sites(term)` unbounded | >3.5 GB RSS, ~57 s × 2 worst candidates, corpus run OOM-killed | — |
| `_all_scan_sites(spec_cap=512)` | — | **0.0 s**, 0 sites, `seq_sites_over_cap=10`, RSS 0.94 GB |
| `_mint_tiers(spec_cap=512)` | — | **0.0 s**, 13 handlers, same skip count |
| `model_corpus(probe=False, n_mutants=0)` | OOM (earlyoom SIGTERM) | **2.6 s** post-export, peak RSS **1.28 GB**, `site_stats=[{"seq_sites_over_cap": 10}]` |

The 10 counted skips are the 24-candidate seq scan's oversized
readings — T5's recurrence-adjacent chains carry step operands far
over 512 spelled nodes, so the seq fold honestly declines them
rather than spelling megabyte specs no claim could carry.

## Tests

`tests/test_discovery_model_corpus.py::TestSeqScanBounded` (new):
`_seq_expansion` exact/capped parity with `census.term_expansion`;
a recurrence whose `b_t` embeds a `_shared_chain(20,"mul")`
(~2²⁰-expansion, 21 distinct nodes) completes `spec_cap=64` in
milliseconds with `seq_sites_over_cap > 0` through
`_seq_scan_sites`, `_all_scan_sites` and `_mint_tiers`; a zoo-scale
recurrence mints identically capped vs uncapped (parity).

Verified: `test_discovery_model_corpus.py` +
`test_discovery_play.py` (148 tests, `-m "not slow"`) and
`test_batched_scan_lift.py` (41) all pass; `ruff check`,
`ruff format --check` and `ty check` clean on touched files.

## Honest limits

- The cap bounds *spelled* work, not claim value: seq/seqm sites on
  model-scale terms now decline when a step skeleton or operand
  child exceeds `spec_cap` — counted, and liftable via
  `--spec-node-cap`.  This is the same trade `gen`/`reduction`
  already made.
- `_seq_spines` itself stays as before (memoised, `_SEQ_SPINE_CAP`)
  — measured trivial on the T5 term; a pathological deep-chain term
  could still cost, but its readings are capped per node.
- `_seq_traj_states`/`_seqt_readout` remain O(distinct nodes) per
  emitted site (≤ `_SEQ_SITE_CAP`+seqt offers) — bounded by the site
  cap, unchanged.
- `spec_cap=None` (every non-`model_corpus` caller, incl. all zoo
  boards) is byte-identical historical behaviour — the bound only
  engages through `model_corpus`'s auto knobs or an explicit
  `spec_node_cap=`.

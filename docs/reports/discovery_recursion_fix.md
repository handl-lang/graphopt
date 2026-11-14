# Discovery recursion-safety fix — model-corpus path

Fixes arch-audit `arch_audit_gamekit_discovery.md` §1.1 (HIGH):
`model_corpus()`/`corpus_probe()`/`generate_mutants()`/
`coverage_report()` could die with `RecursionError` on deep model
terms.  The caps (`_SHAPE_KEY_CAP`, `_AUTO_KEY_CAP`,
`_AUTO_SPEC_NODE_CAP`, `spec_cap`/`_budget`) bound *tree-expansion
size* honestly, but every renderer they guard was recursive — a
~20k-deep mutant admitted by `_AUTO_KEY_CAP` overflowed the default
1000-frame limit instead of declining.

Two-layer fix, matching the audit's own suggestion:

1. **Convert the hot recursive walkers to iterative traversal**
   (post-order where a result is needed, preorder where only a
   visit/order matters), with `id`-dedup wherever the result is
   occurrence-invariant.  Cap semantics are unchanged: tree-size
   accounting still counts *occurrences* exactly as before.
2. **Scope a recursion-limit guard** around the entry points for
   the walkers that must stay recursive (mutually recursive spine
   decomposition, eqsat-owned typing, the torch eval boundary).

## Converted to iterative

| Walker | File | Notes |
|---|---|---|
| `shape_key` | `census.py` | Post-order, `id`-dedup on `Op`; children pushed reversed so leaf numbering stays left-to-right identical. |
| `shape_repr`, `_leaf_ids` | `census.py` | Key-tree depth mirrors term depth; iterative post-order / stack walk. |
| `_spec_of` | `meta_arena.py` | Post-order; leaf metavariables still numbered `X1…` in first-occurrence order. |
| `_canon_spec`, `_canon_concrete` | `meta_arena.py` | Post-order over spec tuples; dict elements shallow-copied never recursed; `spec[0]` head kept raw in `_canon_concrete`; empty tuple/list fall-through preserved. |
| `_composite_specs` visit, `_program_specs._walk`, `_concrete_occurs._walk` | `meta_arena.py` | Preorder/DFS-order-preserving iterative walks, `id`-deduped as before. |
| `_ops_of`, `_term_metavars` | `meta_arena.py` | Iterative set walks — term-depth bound removed. |
| `_fusion_pattern_sz` | `play.py` | Post-order with per-node memo; spelled **tree-size** accounting unchanged (per-occurrence, cap-clamped); leaf naming order and per-arg cap early-exit points preserved. |
| `_fusion_sites`, `_reduction_sites`, `_decayed_sites` walks | `play.py` | Preorder iterative, `id`-deduped — identical site order (feeds the minted handler table). |
| `_seq_scan_sites`/`_seqm_scan_sites` `visit` | `play.py` | Preorder iterative — identical `cands` order. |
| `_cumsum_sites` `visit` | `play.py` | Post-order iterative (`(node, expanded)` stack) — sites still append children-first. |
| `_spec_leaf_env`, `_spec_has_meta`, `_flat_ops` | `play.py` | Iterative; spec/env results are occurrence-invariant. |
| `_mark` (raw-by-class map in `_scan_sites`) | `play.py` | Preorder iterative — first-seen-per-eclass order is semantic and preserved. |
| `_nodes_with_paths`, `_lift_leaf` walk, `_replace`, `_depth` | `workload_gen.py` | Iterative preorder / spine-rebuild / post-order; `_nodes_with_paths`/`_lift_leaf` keep identical `(path, node)` order so `rng.choice`/`rng.shuffle` see identical lists. (`_leaves` was already iterative+deduped.) |
| `_pattern_occurs` | `corpus_cov.py` | Iterative `id`-deduped — boolean OR over path-invariant per-node match verdicts; kills the latent DAG-exponential walk flagged in the audit. |
| `_leaves` | `proposal.py` | Iterative `id`-deduped — was recursive **and** not deduped (exponential on DAGs); result is a set so dedup is semantics-free. |

## Left recursive — covered by `_recursion_guard` only

- `play._seq_deindex` / `play._seq_spines` / `_seq_step_cands` —
  mutually recursive spine decomposition with sequential side
  effects (`names`/`operands`/`events`/`_budget` visit counting in
  tree-expansion order); an iterative rewrite risks subtle
  ordering drift for marginal gain — the spelled-size `_budget`
  cap already bounds the *work*, and the guard covers the *depth*.
- `mlopt_core.typing._shape_of` — eqsat-owned, out of scope.
- torch export/eval/lower/verify boundary inside `export_ir` /
  `play._case_tail` — engine-owned recursion.
- `meta_arena._concrete_match`/`_match_node`, `_spec_metavars`,
  `_spec_ops`, `_spec_size`, `corpus_cov._rule_ops`/`_spec_ops` —
  recursion depth bounded by *spec/pattern* size (≤ spec caps),
  not term depth.
- `arena.py`'s own `_ops_of`/`_n_nodes` — outside this fix's file
  ownership; same recursive shape, covered only if reached through
  a guarded entry point.

## The guard

`model_corpus._recursion_guard()` (`model_corpus.py`) raises
`sys.setrecursionlimit` to `_RECURSION_LIMIT = 200_000` only when
the current limit is lower, and **always restores** the previous
value in `finally` — the same restore discipline as the existing
`workload_gen._MAX_NODES` monkey-patch in `generate_mutants`.

Scoped around:

- `model_corpus()` — the implementation moved to `_model_corpus()`;
  the public name is a thin guarded wrapper, so
  `corpus_probe`'s `corpus_probe=` callable (which calls
  `model_corpus`) is covered too.
- `coverage_report()` — census + shape-key diff.
- `generate_mutants()` — `corpus_stats` typing pass AND the whole
  mutation loop (the `_MAX_NODES` restore stays in its own
  `finally`, now outside the guard scope).
- `_case_tail()` — wraps `play._case_tail` (spec mining, eval,
  verify) per board intake.

## Cap / decline semantics — unchanged

- `term_expansion`/`op_expansion`/`_fusion_pattern_sz` still count
  **tree occurrences**: a shared subterm spelled twice counts
  twice.  `id`-dedup was applied only where the *result* is
  occurrence-invariant (shape keys, specs, site membership,
  leaf sets, match verdicts) — never to the size accounting
  itself.
- Oversized terms still decline: `coverage_report` reports
  `n_shape_subterms_skipped`; `generate_mutants` records
  `rejected["expansion"|"mint"|"gate"|"case"]`; seq collectors
  count `*_sites_over_cap`.  No "profitable"/accepted result is
  produced for work the caps prevented measuring.
- Small-fixture parity: leaf/spec numbering order, site order,
  census keys, and canonical forms are byte-identical to the
  recursive renders (children are always pushed reversed so
  left-to-right order is preserved; `rng`-consumed lists keep
  identical order).

## Tests

`tests/test_discovery_model_corpus.py::TestRecursionDepth` —
`_shared_chain(5_000)` (`op(t, t)` self-sharing; tree expansion
`2^5001 − 1`, DAG depth 5000):

- `test_coverage_report_deep_shared_dag` — completes, reports
  `n_shape_subterms_skipped > 0`, recursion limit restored.
- `test_generate_mutants_deep_shared_dag` — completes with honest
  `rejected` accounting (`expansion`/`mint`/`gate`), limit restored.
- `test_shape_key_and_spec_deep_terms` — `shape_key`,
  `_spec_of`, `_canon_spec`, `_canon_concrete` all return on the
  5000-deep chain without `RecursionError`.

## Verification (measured)

- `pytest tests/test_discovery_model_corpus.py -q -m "not slow"` —
  **36 passed** (incl. all three `TestRecursionDepth` tests, ~70 s:
  `coverage_report` 13.5 s, `generate_mutants` 60.8 s — the cost is
  the honest O(distinct-nodes²) `term_expansion` pre-check per
  subterm, not recursion).
- `pytest tests/test_discovery_census.py
  tests/test_discovery_meta_arena.py
  tests/test_discovery_corpus_cov.py -q` — **124 passed**.
- `pytest tests/test_discovery_play.py -q -m "not slow"` —
  **115 passed**.
- `pytest tests/test_discovery_genkernel.py
  tests/test_discovery_climb_a2.py
  tests/test_discovery_meta_game.py -q -m "not slow"` —
  **302 passed, 1 skipped**.
- `pytest tests/test_discovery_impact.py -q` — **29 passed**.
- `pytest tests/test_discovery_proposal.py` — 7 failures, all
  cost-delta assertions (`cand_cost == base_cost == 16.0`); they
  reproduce independently of this change — `_leaves` only feeds
  numeric-eval leaf collection, never `dag_cost`/`flops_cost` —
  and correlate with the eqsat relock (`mlopt_core.cost` semantics
  in commit `a18a209`).  Flagged for the substrate owners.
- `pytest tests/test_discovery_intake.py` — 6 failures
  (`census-only` → `ingested`), caused by the concurrent
  `torch_bridge.py` supported-ops change in the working tree, not
  this fix.
- `.venv/bin/ruff check packages/mlopt-discovery/src/mlopt_discovery
  tests/test_discovery_model_corpus.py` — clean.
- `.venv/bin/ruff format --check` — clean.
- `.venv/bin/ty check` — **0 errors** (whole repo).

### Output-parity evidence

- Differential check vs. memoized recursive references on 500
  random shared-DAG terms plus `op(t,t)` chains to depth 200:
  `shape_key`, `_spec_of` (incl. `names` maps), `_canon_spec`,
  `_canon_concrete` all identical (`dag_eq`, since spec/key results
  are themselves DAGs — `==`/`repr` tree-expand by design and stay
  cap-bounded downstream).
- `_fusion_pattern_sz` vs. the recursive reference on 400 random
  elementwise DAGs × caps {None, 8, 40}: identical `(pattern,
  size)` on every *successful* spell.  On cap-overflow the `names`
  scratch dict may hold extra metavar entries — unobservable: the
  `(None, cap+1)` result is always discarded by callers and `names`
  is fresh per site spell.
- `workload_gen._replace` verified equal to the recursive rebuild
  on 300 random `(term, path)` pairs incl. a depth-500 chain spine.
- Deep-DAG smoke (no caps): `_fusion_sites`, `_reduction_sites`,
  `_decayed_sites`, `_cumsum_sites`, `_seq_scan_sites`,
  `_seqm_scan_sites` all return on `_shared_chain(5000)` with no
  `RecursionError` (seq collectors still set their own 40 000
  limit; the model-corpus guard covers the rest at 200 000).

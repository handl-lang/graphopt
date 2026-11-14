# ProbeBounds — the typed probe-budget object

Implements the arch audit's highest-leverage recommendation
(`docs/reports/arch_audit_gamekit_discovery.md` §1.2 + §3): one
frozen dataclass replacing the scattered `spec_node_cap` /
`spec_cap` / `key_cap` / referee-dict / `ex[...]` stringly-typed
budget plumbing, and an explicit fix for the dual-metric hazard.

## API

`mlopt_discovery.bounds` (new module):

- **`ProbeBounds`** — frozen dataclass, all fields `None` =
  unbounded (the historical zoo behaviour):

  | field | metric | consumer |
  |---|---|---|
  | `spec_tree_cap` | spelled **tree-size** — every node a spec/pattern render spells (op nodes + metavar/const leaves + one per attrs dict; the `play._fusion_pattern_sz`/`_reduction_pattern` accounting, `_seq_deindex` visit count for seq tiers) | `play._mint_tiers` → site collectors (`gen`/`reduction`/`scan`) |
  | `spec_op_cap` | **op-expansion** — `census.op_expansion`, Op nodes only | `MetaArena(spec_node_cap=…)` → `_composite_specs`/`_program_specs` |
  | `key_cap` | `census.term_expansion` tree-expansion | `workload_gen.corpus_stats`/`valid_term`, `generate_mutants` novelty keys |
  | `shape_key_cap` | `census.term_expansion` per subterm | `model_corpus.coverage_report` |
  | `max_handlers` | count | `genkernel.profitable` claim cap |
  | `time_budget` | seconds per claim paired window | `genkernel.profitable` |
  | `reps` | count | `genkernel.profitable` reps override |

  Methods: `referee_dict()` (legacy dict shape),
  `profitable_kwargs(reps_default=…)`, `as_dict()` (report echo),
  `ProbeBounds.from_legacy_ex(ex)` (rebuild from `ex["_referee"]` +
  `ex["_spec_node_cap"]`).

- **`SkipStats`** — `dict` subclass tallying honest over-cap
  declines (`skip(kind)`, `merge(other)`, `total`), protocol-
  compatible with the existing `stats.get`/`stats[k]=` collector
  contract.

## Metric unification decision — keep both, name distinctly

The two spec caps measure **different artifacts**, so they stay
separate fields rather than being unified:

- `spec_tree_cap` bounds the *minted claim pattern* — the spelled
  tuple the tiers consume.  `_fusion_pattern_sz` abstracts
  non-elementwise children to single metavar leaves, so its tree
  size is NOT the term's op count.
- `spec_op_cap` bounds the *spec-inventory render* —
  `meta_arena._spec_of` spells every op concrete and abstracts
  leaves to metavar names; `census.op_expansion` is its honest
  cost proxy.

The audit's bug was one number (`spec_node_cap`) silently meaning
both.  Now: the legacy `spec_node_cap` **argument** still sets BOTH
caps (each measured in its own unit — byte-identical effective
behaviour to before), while `bounds=ProbeBounds(...)` allows
asymmetric caps.  `MetaArena`'s constructor kwarg keeps the name
`spec_node_cap` (frozen surface — many callers) but is documented
as the op-expansion metric; the attribute is now `_spec_op_cap`.

## Threading

- `model_corpus._probe_knobs` resolves the `_AUTO_*` defaults once
  and returns `{"bounds": ProbeBounds, …legacy keys…}` — auto
  engages on tree-expansion > `_LARGE_TERM_EXPANSION` exactly as
  before.
- `play._case_tail(bounds=…)` seats `ex["bounds"]` (typed) plus the
  legacy `ex["_referee"]`/`ex["_spec_node_cap"]` spellings;
  `play._ex_bounds(ex)` is the single reader — typed record wins,
  `from_legacy_ex` reconstructs for older case producers
  (`_gen_cases`).
- `_fold_cost_gate` / `_claim_cost_gate` / `_gen_board` read
  `_ex_bounds(ex)`; `_gen_board` feeds `spec_op_cap` to
  `MetaArena(spec_node_cap=…)`.
- `genkernel.profitable(bounds=…)` supplies `max_handlers` /
  `time_budget` / `reps` defaults — explicit kwargs always win;
  `reps` default changed `40 → None → 40` (same behaviour).
- `model_corpus.{_case_tail, model_board, mutant_board,
  model_corpus, corpus_probe}` accept `bounds=`; the legacy
  `referee`/`spec_node_cap` kwargs still resolve via
  `_coerce_bounds` (spec cap → both metrics, as before).

## Skip statistics

Every bounded collector now reports into the shared tally:
`fusion_sites_over_cap` and `reduction_sites_over_cap` join the
existing `seq_sites_over_cap`/`seqm_sites_over_cap`/
`cumsum_sites_over_cap` (the audit §1.4 asymmetry fix —
`_mint_tiers` passes `stats=` to all cap-aware tiers; `decayed`
takes none).  `_case_tail` seats a `SkipStats` on
`ex["site_stats"]`; the `model_corpus` report aggregates it as
`rep["skip_stats"]` (and keeps per-board `site_stats`), and the CLI
prints the totals.

## Files

- `packages/mlopt-discovery/src/mlopt_discovery/bounds.py` — NEW
- `play.py` — `_ex_bounds`, `_case_tail(bounds=)`, `_mint_tiers`
  `bounds=` + unified stats dispatch, collector over-cap counters,
  gates/`_gen_board` read typed bounds
- `model_corpus.py` — `_probe_knobs`→`ProbeBounds`, `bounds=`
  threading, `skip_stats`/`bounds` report fields, CLI help
- `meta_arena.py` — `_spec_node_cap`→`_spec_op_cap` + metric docs
- `genkernel.py` — `profitable(bounds=…)`, `reps=None` default
- `census.py`, `workload_gen.py` — metric cross-references in
  docstrings only
- `tests/test_discovery_bounds.py` — NEW (17 tests)

## Verification

- `tests/test_discovery_bounds.py`: 17 passed
- `tests/test_discovery_model_corpus.py -m "not slow"`: 36 passed
  (incl. the recursion-guard regression tests — deep shared-DAG
  terms still decline honestly)
- `tests/test_discovery_play.py -m "not slow"`: 115 passed
- `tests/test_discovery_genkernel.py -k "profitable|cost_gate|fold|
  referee|mint"`: 37 passed
- `tests/test_discovery_{meta_arena,census,meta_game,
  corpus_cov}.py`: pass per-file/per-pair; the 4-file combo shows 4
  `corpus_cov` failures that reproduce identically on HEAD~
  (pre-existing cross-file state pollution, not this change)
- `tests/test_discovery_proposal.py`: 7 cost-delta failures —
  verified identical on HEAD~ (eqsat relock, pre-existing)
- `ruff check`/`ruff format --check`/`ty check` clean on touched
  files

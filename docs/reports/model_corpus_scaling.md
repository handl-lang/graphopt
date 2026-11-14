# model_corpus scaling — arbitrary models, bounded probe

Date: 2025-11-14.  Scope: `mlopt_discovery.model_corpus` and the
shared probe path it drives (`play._case_tail` → `genkernel.
profitable` → `MetaArena` board → `gen_probe`).

## The zoo-scale assumptions that hung Whisper

`python -m mlopt_discovery.model_corpus --module <whisper-base
encoder> --mutants 0 --no-probe --corpus bench` produced **no report
after 30 minutes** (killed).  The measured-cost referee path was
effectively zoo-scale only.  The bound assumptions, in order of
severity found:

1. **Tree-rendered `repr()`/spec walks over shared IR DAGs.**  The
   exported IR is hash-consed: a residual stream references its input
   twice (`y + f(y)`), so *distinct-node* counts stay small (~300 for
   whisper-base encoder) while every *tree expansion* is exponential.
   `play._fusion_pattern` keyed non-elementwise leaves by `repr(t)` —
   a recursive S-expression render — and the site collectors
   (`_fusion_sites`, `_reduction_sites`, `_decayed_sites`),
   `genkernel._site_env`, `meta_arena._composite_specs` /
   `_program_specs` / `_concrete_occurs`, `workload_gen`'s
   `_nodes_with_paths` / `_lift_leaf` / `_leaves` all walked per
   *occurrence*, not per distinct node.
2. **`MetaArena`'s feasible-cost `expand` had no memo.**  Every
   `observe()` → `extract_best`/`dag_cost` re-expanded declared-op
   definitions by naive recursion — cProfile on a 16-block residual
   MLP showed 2.9 M `Op.make` calls inside one `observe()` (19 s);
   `_probe_play`'s ~15 steps × observe ≈ **503 s** on that model.
3. **Fixed repetition counts multiplied by claim count.**  `gen_probe
   (reps=200)`, `_case_tail(gen_reps=20)`, `profitable(reps=40)` —
   reps are shape-blind: one expensive call on a large model
   dominates, and `profitable` priced *every* minted handler
   (borderline claims up to 3 timing windows each).  `gen_probe`'s
   `budget` parameter was never enforced as a wall-clock bound.
4. **Census shape keys tree-render.**  `shape_census`/`shape_key`
   expand subterms as trees — OOM/kill territory on residual DAGs.
5. **`workload_gen.corpus_stats` re-ran typing per subterm.**  Each
   `_concrete_shape` → `_shape_of` call got a fresh memo, so the
   census re-inferred every sub-DAG: **217 s** for the 318-subterm
   whisper encoder term alone.  `mutant_term`'s `_nodes_with_paths`
   also recursed into already-seen nodes (dedup guarded the append,
   not the descent) — **~9 s per mutation attempt**, >4 min for the
   120-attempt generation loop.

## What changed

All bounds are *probe-size* bounds: they cap how much work the probe
spends, never relax a measurement.  A claim the bounds skip is
declined — never reported profitable unmeasured.  Every cap defaults
to `None` = historical unbounded zoo behaviour; only terms whose
`census.term_expansion` exceeds `_LARGE_TERM_EXPANSION` (100 000)
auto-engage the `_AUTO_*` defaults, and every knob is individually
overridable (`model_corpus(..., spec_node_cap=, key_cap=,
max_claims=, claim_time_budget=)`; CLI `--spec-node-cap`,
`--key-cap`, `--max-claims`, `--claim-budget`).

### DAG-safe traversal (correctness, not just speed)

- `census.term_expansion` / `op_expansion` — O(distinct nodes)
  tree-expansion counters with optional cap early-exit, the shared
  "how big would a render be" oracle.
- `workload_gen._nodes_with_paths`, `_lift_leaf`, `_leaves` —
  `id`-deduped walks (the `_nodes_with_paths` descent bug above).
- `workload_gen.corpus_stats` / `valid_term` — `key_cap` bounds
  `shape_key` novelty sets and the validity gate; `_concrete_shape`
  now threads one shared `_shape_of` memo (`CorpusStats.shape_memo`)
  so typing is DAG-linear.
- `play._fusion_pattern_sz` — spells fusion patterns with per-node
  memoized *construction* but tree-accurate *size* accounting
  (a shared elementwise subterm is spelled once per occurrence —
  the spec tuples downstream really are trees).  Overflow returns
  `(None, cap+1)`; `_fusion_sites` / `_reduction_sites` /
  `_mint_tiers` take `spec_cap` and skip oversized sites.
  `_leaf_name` keys non-elementwise `Op` leaves by object identity,
  not `repr`.
- `genkernel._site_env` — pre-order concrete match, `id`-deduped.
- `genkernel._bound_val` — `eval_term(..., memo_env={})` so a bound
  metavar that is a whole shared-DAG chunk evaluates once per node.
- `meta_arena` — `_composite_specs`/`_program_specs` honour
  `spec_node_cap` (skip subterms whose op-expansion exceeds it —
  they can never equal a mined spec anyway); `_concrete_occurs`,
  `unfold`, `deliverable` are `id`-deduped/memoized;
  **`_feasible_cost.priced`'s `expand` is memoized per call** — the
  single biggest win (observe 19 s → 0.1 s on the residual MLP).

### Bounded referee (`genkernel.profitable`)

- `max_handlers` — only the first N entries of the *sorted* handler
  table are priced (deterministic prefix); the rest are dropped.
- `time_budget` — per claim, `_kernel_pays` estimates one call pair
  after verify-warmup and clamps `reps` into `[3, reps]`; the
  effective count is returned and reused for borderline confirmation
  windows, so a claim's worst case is ~3× `time_budget` of *timed*
  reps (compile/verify warmup is untimed, bounded by `max_handlers`).
- `_case_tail` / `_claim_cost_gate` / `_fold_cost_gate` thread the
  same `referee` dict everywhere so gates never re-pay an uncapped
  sweep; `ex["_referee"]`/`ex["_spec_node_cap"]` carry the bounds to
  `_gen_board`.

### Bounded coverage & mutants (`model_corpus`)

- `coverage_report` uses `_bounded_shape_counts`: subterms whose
  `term_expansion` exceeds `shape_key_cap` (10 000) are skipped on
  BOTH sides — a shape nobody could key cannot be reported novel —
  and `n_shape_subterms_skipped` is reported honestly.
- `generate_mutants`/`corpus_stats`/`valid_term` take `key_cap`
  (auto 20 000 on large terms): candidates that cannot be keyed or
  evaluated in feasible time decline as `"expansion"` rejects.
- `_probe_knobs` resolves the auto bounds from the model term's own
  expansion — no zoo dimensions anywhere.

## Measured results

| run | before | after |
|---|---|---|
| whisper-base encoder, `--mutants 0 --no-probe --corpus bench` | >30 min, killed, no report | export+coverage+intake+coverage report in seconds of probe work (full CLI ~4 min incl. compile arms) |
| whisper-base encoder, `--mutants 4 --corpus bench --reps 3 --gen-reps 3` | — | **4 m 02 s** wall (RTX 2050 4 GB, fp32, CUDA arms) |
| synthetic 79 K-param residual MLP (18 shared blocks, expansion 196 607) | `_probe_play` alone >500 s | full `model_corpus` (2 mutants, probe) **95 s** |

Whisper report highlights (`/tmp/mc_whisper_rep2.json` shape):
`term_expansion=119360 → large_term`, `spec_cap=512`,
`key_cap=20000`, `max_claims=16`, `claim_budget=2.0`; coverage
32 op-tuples / 104 shapes with **101 subterms honestly skipped** by
the shape-key cap; 2/4 mutants accepted (116 expansion rejects —
mutants inherit the model's sharing); model row declined honestly
(`deliver failed: TypeError … 'BaseModelOutput' and 'Tensor'` —
non-tensor module output, see limits); two mutant rows delivered,
verified, speedup ≈ 1.01–1.02 → `pay` verdicts *measured*, not
assumed.

## Remaining honest limits

- **Non-tensor module outputs.**  `gen_probe`'s verify/delivery
  subtracts outputs; a `BaseModelOutput` (or any non-Tensor return)
  declines the row with an honest note — whisper encoder's row is
  exactly this.  A tuple/dict-output verify is future work, not a
  silent skip.
- **Integer / non-float feeds.**  `_as_dtype` now casts only
  *floating-point* tensors — `int64` token ids, attention masks and
  position indices pass through untouched (previously every feed
  tensor was cast to the intake dtype, silently corrupting decoder
  `input_ids`).  Whether `TorchSource.to_ir` *lowers* integer-typed
  inputs is a separate intake question this change does not claim.
- **Compile warmup is untimed.**  `time_budget` bounds the paired
  rep windows, not the per-claim verify/`torch.compile` warmup —
  `max_handlers` is what bounds that spend.  On exotic specs a single
  inductor compile can still cost tens of seconds.
- **`budget` on `gen_probe` is still advisory** — the wall-clock
  bound lives in the referee (`time_budget`) and the spec/expansion
  caps, not in a global episode timer.  A hard end-to-end deadline
  remains open.
- **Skipped claims are absent, not measured.**  Claims dropped by
  `max_handlers`/`spec_cap`/`key_cap` never reach `profitable`'s
  keep set; the report's `claims_priced`/`claims_kept` counts in
  `ex` and the mutant `rejected` table are the honest record.
- **`_shape_of` memo is id-keyed.**  `CorpusStats.shape_memo` assumes
  nodes outlive the stats build; a recycled `id` could in principle
  return a stale shape — it can only mis-pick a graft *donor*, and
  `valid_term`'s typing/eval gates still judge every candidate, so
  correctness is preserved (worst case: wasted attempts).
- **Recursion depth.**  `_fusion_pattern_sz` and friends recurse;
  terms deeper than ~1000 nested ops can hit the recursion limit —
  a loud `RecursionError`, not a hang.

## Tests

`tests/test_discovery_model_corpus.py` gained
`TestExpansionBounds`, `TestSiteCollectorsBounded`,
`TestRefereeBounds`, `TestBoundedCoverageAndKnobs` — exact/capped
expansion values on a self-sharing chain, spec-cap behaviour and
parity on zoo-scale sites, `_site_env` dedup counts,
`max_handlers` deterministic-prefix and `time_budget` rep clamping
(including the borderline-window reuse), bounded coverage skip
counts, `_probe_knobs` auto/explicit resolution, and `key_cap`
mutant rejection.  `tests/test_discovery_genkernel.py`'s three
`_kernel_pays` stubs were updated for the new `(t_s, t_k, reps_eff)`
return.

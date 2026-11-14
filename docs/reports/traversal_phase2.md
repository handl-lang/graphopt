# Traversal phase 2 — dedup the discovery walkers onto `mlopt_core.traversal`

Follow-up to the eqsat `mlopt_core.traversal` landing (`7fe2ed6`,
retagged `v0.1.0` = eqsat `8d12593`).  Phase 1 (graphopt `906b37d`)
made the corpus-path walkers iterative to kill `RecursionError`;
this phase replaces the *mechanics* where the semantics are provably
identical, and documents every site that stays local.

The API contract that drove every decision
(`.venv/.../mlopt_core/traversal.py`):

- `subterms` / `ops` / `leaves` — **DAG-deduped** (content
  `__eq__`/`__hash__`; interned `Op`s make that object identity in
  practice), pre-order, root first, children left-to-right.
- `postorder` — children-before-parents, **right-to-left** children
  (the `_dag_postorder` convention), deduped.
- `expand` — **per-occurrence tree spelling**, `(path, node)`
  pre-order, no dedup; `depth`/`budget` bounds available.

A recursive `for a in t.args` walk is an *occurrence* walk →
`expand`.  A walk with a `seen`/`visited` set is a *distinct-node*
walk → `subterms`/`ops`/`leaves`/`postorder` **only if** the result
is order-insensitive or the orders provably match.  `id()`-keyed
dedup maps to content-dedup safely here because `Op.make` interns —
equal content *is* the same object — and every migrated site either
feeds a set/dict (dedup-insensitive) or a boolean OR (verdict is
path-invariant per node).

## Migrated — `mlopt_discovery`

| Module | Sites | Helper used | Semantics preserved |
|---|---|---|---|
| `antiunify.py` | `_n_ops`, `_eff_ops`, `_metavar_count` | `expand` | per-occurrence op/metavar counts (tree spelling) |
| | `_leaf_mvars`, `_attr_mvars_of` | `subterms`, `expand` | first-occurrence preorder; attr scan per occurrence |
| `arena.py` | `_ops_of`, `_n_nodes` | `ops`, `expand` | op-name set; occurrence op count |
| `combinator.py` | `_postorder`, `dag_size` | `postorder`, `subterms` | children-before-parents R-to-L, deduped; distinct-node count |
| `corpus_cov.py` | `_rule_ops`, `_pattern_occurs` | `ops`, `subterms` | law op-name set; boolean OR over per-node match (path-invariant) |
| `domain_spec.py` | `_term_ops_attrs`, `_attr_values_ok` | `ops` | `(op, attr)` set; conjunctive attr check (order-free) |
| `emit.py` | `_mv_counts`, `_mv_positions` | `expand` | metavar/attr-mvar counts; path list with the recursion's post-subtree attr-flush order reproduced via a pending stack |
| `gap_gen.py` | `_metavars`, `_leaf_paths` | `subterms`, `expand` | first-seen leaf-mvar order; `(path, Var/Param)` occurrence list |
| `genkernel.py` | `_site_env`, `attr_site_env` | `expand` (+ explicit `id` seen-set in `_site_env`) | first-match preorder; `id`-dedup kept verbatim where the search contract was identity-keyed |
| `learned_propose.py` | `term_stats`, `_op_hist`, `_mvar_set`, `_op_sites`, `_leaf_paths` | `expand`, `leaves` | per-occurrence stats incl. tree `depth` via `len(path)+1`; op histogram; mvar set; `(path, node)` yields |
| `meta_arena.py` | `_composite_specs`, `_ops_of`, `_term_metavars`, `_concrete_occurs` | `ops`, `leaves`, `subterms` | spec census (sorted+cap output makes visit order irrelevant); name/mvar sets; boolean match OR |
| `meta_game.py` | `_attr_mv_names`, `_share_scan._descs` | `ops`, `expand` | attr-mvar name set; per-edge `(path, child)` multiset (duplicate detection is order-free) |
| `open_game.py` | `_term_depth`, `_walk` | `expand` | tree depth = max `len(path)+1` over Op occurrences; op count + Var-name set |
| `open_verdict.py` | `_ops_preorder` | `expand` | op-name occurrence list, preorder |
| `oracle.py` | `_leaf_metavars`, `_parents` | `subterms`, `expand` | first-occurrence mvar order; metavar→parent-op map (set-valued) |
| `pay_gate.py` | `_subterms` (feeds `term_features`) | `subterms` | distinct-node op census — the "shared subterms counted once" contract |
| `play.py` | `_fusion_sites`, `_reduction_sites`, `_decayed_sites`, `_seq_select_free`, `_op_parents`, `_op_descendants`, `_seq_scan_sites`, `_seqm_scan_sites`, `_ops_walk` | `ops`, `expand` | preorder site order preserved (`ops` yields the same L-to-R preorder the reversed-stack DFS produced); `_ops_walk` deliberately uses `expand` — it is an *occurrence* yield, not a dedup walk |
| `prim_mint.py` | `sub_alphas`, `pat_ops` | `expand`, `ops` | proper-Op-subterm alpha-norm set; pattern op-name set |
| `training.py` | `leaves_of` | `expand` | first-occurrence `name → leaf` map (Var/Param only) |

All cap/budget behavior (`spec_cap`, `node_cap`, `ProbeBounds`,
site caps, memo dicts) is untouched — the walkers only changed
*how* nodes are visited, never *which* or *how many*.

`proposal._size`/`_leaves` were on the list but are now owned by the
concurrent torch-seam split (`proposal_core.py`, untracked, another
agent's in-flight work) — that agent already migrated both to
`traversal.expand`/`leaves` with identical semantics.  No action
taken here; `proposal.py`/`vocab.py`/`oracle.py`'s lazy-`torch`
hunks/`tests/test_discovery_proposal.py` belong to that agent and
were not touched by this work.

## Already iterative — deliberately left alone

These were converted to explicit-stack walkers by `906b37d` (or
were always iterative) and their semantics differ from every
`traversal` helper — migrating would change behavior:

- `census._iter_subterms`, `census._postorder`, `census._expansion`,
  `impact._iter_subterms` — the shared corpus helpers; `id()`-keyed
  dedup and caller-dependent order feed `pipeline`,
  `workload_gen`, `model_corpus`, `vocab`, `shape_proposal`,
  `prim_mint.op_subterms`, `meta_game`.  Content-dedup vs `id()`-dedup
  and preorder direction are load-bearing for the census counts.
- `play._fusion_pattern_sz` / memoized pattern builders, `play`'s
  `leaf_env` env-propagation walk, the `cum*` scan filter — custom
  memoized/env-carrying traversals, not plain node visits.
- `play._seq_deindex` / `_seq_spines` — mutually recursive with
  side-effect/order behavior (explicit skip per task brief).
- `play._seq_step_cands`-adjacent spec walkers, `meta_arena`'s
  `_spec_metavars`/`_spec_ops`/`_spec_size`/`_concrete_match`/
  `_shape_of`-class helpers, `corpus_cov._spec_ops`/`_rule_ops`'s
  spec-side twins, `domain_spec`'s spec walkers — these recurse
  through **spec tuples/lists/dicts and attr values**, not `Op.args`
  DAGs; `expand`/`subterms` do not apply.
- `training._topo` — postorder over the *program* with `seen`
  semantics feeding gradient emission order; `traversal.postorder`
  is R-to-L children while `_topo` is L-to-R — order is
  load-bearing.
- `impact._relax.walk`, `shape_proposal.walk`, `antiunify`'s
  `_alpha_norm`/`priced`/`_swap_hole` `go`s, `learned_propose._rebuild`,
  `prim_mint.expand_synops`, `meta_arena.expand`/`go`,
  `index_abstract` matchers, `handl._metavar_paths` — term
  **reconstruction/renaming/matching** transforms, not enumerations.
- `oracle.d` chain-depth memo, `evidence.walk` (DB record DAG),
  `forms`/`grammar`/`handler_synth`/`object_synthesis`/`enumerate`/
  `dspec_synth`/`family_synth`/`shipped_primitives`/`vocab` spec &
  shape-IR walkers, `combinator` morphism-IR walkers (`arity`,
  `_apply`), `arena._data_*` dict-tree walks — non-`Op`-DAG domains.
- `mlopt_core.typing._shape_of` — eqsat-owned (now itself iterative
  over `traversal.postorder` upstream).

## `bench/drivers` — audited, not migrated

Out of the editable scope (only `mlopt-discovery` +
`mlopt-carriers` were in bounds); candidates recorded for a later
pass: `screen/exposure_screen.py`, `probes/decoder_layer_census.py`,
`decoder_layer_pay.py`, `law_transfer_probe.py`,
`conv_fold_premise_pay_probe.py`, `coordination_probe.py`,
`om_shared_state_probe.py`, `conv_bn_fold_synth_probe.py`,
`attr_elementwise_probe.py`, `prim_frontier_probe.py`,
`gate3_state_spelling_probe.py`, `anti_unify_probe.py`,
`prim_delivery_probe.py`, `contraction_policy.py`,
`contraction_precondition.py`, `om_decayed_kway_probe.py`,
`bridge/combinator_probe.py`, `kit/term_hnl.py`.  Each needs the
same occurrence-vs-distinct audit before touching;
`mint/real_model_mint.py` is another agent's dirty file — excluded.
Torch-boundary walkers everywhere stay local by policy.

## `mlopt-carriers` — shared shape memo

Seven modules carried an identical `_SHAPE_MEMO: dict = {}` feeding
`mlopt_core.typing._shape_of(term, memo)` — same content-keyed,
process-global memo semantics (sound because terms are hash-consed).
Extracted to `mlopt_carriers/_shape.py::shape_of(term)` (one shared
`_MEMO`) and migrated:

- `decayed_lower.py`, `omd_lower.py`, `om_lower.py` —
  `_shape_of(x, _SHAPE_MEMO)` → `shape_of as _shape_of` + `x`.
- `scan_lower.py` — `shape_of(x, _SHAPE_MEMO)` (public alias) →
  `mlopt_carriers._shape.shape_of`.
- `trace_lift.py` — `_shape_of(t, _SHAPE_MEMO)` inside
  `_shape_of_class` → shared helper (the per-class `memo` dict is
  unchanged — it dedupes across plans, a different layer).
- `trace.py`, `xcarrier.py` — local `_shape_of(t)` wrappers now
  delegate to the shared memo; call sites untouched.

`decoder_search_fixed.py`'s `MLOPT_SHAPE_MEMO_PATCH` env patch
monkeypatches `typing._shape_of` itself — still honored (the shared
helper calls the same core function).

## Gates

- `pytest tests/test_discovery_model_corpus.py
  tests/test_discovery_play.py -q -m "not slow"` — **151 passed**
  (131.7 s; re-run after the concurrent seam commits `22219c5` /
  `5eca640` landed mid-session — still 151 passed, 139.3 s).
- Broader touched-module slice (18 files: antiunify, arena,
  combinator_bridge, corpus_cov, domain_spec, emit, genkernel,
  meta_arena, meta_game, open_game, oracle, pay_gate, training,
  learned_propose, prim_mint, propose, vocab) — **931 passed,
  1 skipped, 4 failed**; all four failures verified **pre-existing /
  unrelated to this change**:
  - `test_operand_shape_kinds`, `test_backward_term_is_a_board_citizen`,
    `test_default_seed_bank_is_machine_shipped` — eqsat `v0.1.0`
    retag fallout (new iterative `_shape_of` no longer raises
    `TypeError` on unresolved attrs; `MACHINE_SHIPPED` `.expand`
    premise rows; AUTO_ADMITTED `synop_*` laws reaching extraction).
    None of the migrated functions are on those paths.
  - `test_emit_then_remint_mints_depth2` — passed in isolation and
    on re-run of `test_prim_mint.py` alone (order-dependent state,
    not a regression).
- Carriers slice (xcarrier ×3, decayed_lower, omd_lower ×2,
  scan_lower_edges, trace ×4, scan/om fused+batched+stream ~16
  files) — **384 passed, 2 failed**:
  `test_scan_fused.py::test_fused_dag_shared_leaf_{diag,dense}` —
  reproduced identically with `scan_lower.py` restored to `HEAD`
  → pre-existing (eqsat retag), not the memo consolidation.
- `ruff check` / `ruff format --check` on both touched packages —
  clean.  Repo-wide `ruff check packages scripts examples` reports
  4 pre-existing D205/D209 errors in `packages/mlopt-dtypes`
  (untouched by this work).
- `ty check` — **all checks passed**.
- No commits made.

## Files changed

`packages/mlopt-discovery/src/mlopt_discovery/`: antiunify, arena,
combinator, corpus_cov, domain_spec, emit, gap_gen, genkernel,
learned_propose, meta_arena, meta_game, open_game, open_verdict,
oracle (traversal hunks only — its lazy-`torch` hunks are the other
agent's), pay_gate, play, prim_mint, training.
`packages/mlopt-carriers/src/mlopt_carriers/`: new `_shape.py`;
decayed_lower, om_lower, omd_lower, scan_lower, trace, trace_lift,
xcarrier.

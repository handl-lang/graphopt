# Architecture audit — eqsat (mlopt-core + mlopt-orchestrator)

Read-only audit (subagent). Scope actually read: ir.py, egraph core/
types/terms/match/extract/certs, pipeline/ports/ops/opmeta/attrs/
opdata/lawdata/typing(partial), cost modules (most), laws base/
ruleset/pairing(partial)/tensor(partial)/cond(partial), pareto,
game(partial); orchestrator __init__/criteria(partial)/optimize
(~2500 of 3990 lines incl. all strategy classes)/runners/
morphisms/signature(partial). NOT fully read: laws/{scan,attention,
layout,factored,specials,headshare,priorart,minted,serialize,store,
tags,provenance}, cost/{executor,backend}, egraph/{proof}, meta,
autotune, carriers, regime, diagram*, crossblock_cse,
morphisms/{graph,laws,reify,boundary}, morphisms_kv, core
features/profile/perf_model/rulecache/trajectories/failures/cells/
metrics/timing/policies/search_env.

Flows traced: Monolithic (search→_select_best_term→extract_best/
paired→carrier_upgrade→specialize_causal→bound_gate→lower→
runner.apply→sink.verify), Compositional (per-block + structural-
signature cache + pairwise cross-block + graft), certificate path
(egraph_dependent honestly flagged).

## Bugs (correctness)

1. **`egraph/extract.py:197` — `extract_alternatives` dedup keys on
   `op_repr(term)`** — documented non-injective past
   `_OP_REPR_BUDGET` (the fix we just shipped made this real): two
   distinct members above budget silently collapse to one frontier
   entry. Fix: key on structural key (`_term_eq`-style or
   `op_repr_dag`).
2. **`egraph/types.py:366-373` — `_LeafRegistry._key_to_term` keyed
   by bare `repr(term)`** — `Var("x")`/`Param("x")`/`Const("8")` all
   repr `"x"`/`"8"`; same-name Var+Param land in same leaf e-class →
   congruence merges across unequal leaves → silent wrong answers.
   decode() also returns raw key on unknown. Fix: typed key
   `(type,name,shape)` + loud decode failure.
3. **`typing.py:153-163` — partial-None early return in
   `_infer_op_shape` returns `shapes[0]` for ANY op when operand 0's
   shape is known and a later one is None** — fabricates wrong shape
   for non-commutative ops (matmul(x:(m,k), w:None) → (m,k)) →
   wrong numel → wrong cost → wrong extraction. Rare but genuine
   silent-wrong path. Fix: only return shapes[0] for ops whose shape
   provably equals operand 0's.
4. **Congruence closure deferred to final unrestricted rebuild()** —
   during saturation, structurally-identical enodes in different
   classes never merge; rules needing congruence can't fire →
   fixed point not congruence-closed until after loop ends, no
   re-fire. Completeness gap (not unsoundness) — under-explores on
   congruence-enabled rewrite chains. Verify intended semantics.
5. **`cost/params.py:246,333` — memo keys ignore `source_tensors`/
   `by_bytes`** — sharing one memo across tensor sets returns stale
   results. Latent (callers bind fresh memos). Fix: include dict
   identity in key.
6. `meta.py:385`/`terms.py::_term_match` mutate caller's `subst` on
   failure paths — contract wart, current callers pass fresh dicts.
7. **`ir.py:530-569` + `laws/minted.py` — `_OP_REGISTRY`/`GenDef`/
   `op_def` is a dead write-only registry** — `generator(name)` has
   zero call sites. Delete or wire.
8. **`cost/params.py:373` `_FUSION_POINTWISE_OPS` missing ops opmeta
   tags POINTWISE** (clamp, prelu, hardtanh, softplus, threshold...)
   → compiled-lowering cost overprices graphs containing them.
   Fix: derive as projection of opmeta POINTWISE tags.

## Duplication (drift risk)

9. **~10 parallel term traversals** with subtly different semantics
   (recursive vs iterative, DAG-dedup vs tree, memo conventions) —
   ir.py _dag_postorder/_dag_shared/_tree_sizes; terms.py
   _iter_ops/_term_paths/_term_match/_term_instantiate/_subterm/
   _replace_subterm; meta.py match_pattern/instantiate_pattern/
   _positions/_subterm/_replace/_flatten_chain; optimize.py
   _subterms/_term_params/_param_leaves/etc; signature.py _iter_ops
   (returns list!)/_subtree_nodes/_param_only/_has_param/_shape_tuple;
   cost modules each re-implement walk+memo. Recursive ones hit
   recursion-limit walls that iterative `_generic_overhead` avoids —
   `search()` bumps recursionlimit to 40k as workaround. Fix: one
   shared traversal util.
10. **Four overlapping op-vocabulary authorities**: `_OP_REGISTRY`
    (dead), `opmeta` (live), `attrs.ATTR_SCHEMA`, `typing.
    _infer_op_shape` — `op_def("add",2,1,commutative=True...)`
    restates what opmeta COMMUTATIVE_OPS + lawdata.AC_IDENTITY say.
11. Leaf-equality ×3 implementations (`_leaf_equal` defensive,
    `_exact_equal` assumes .all(), `_raw_bytes` bitwise) — inconsistent
    robustness on duck-typed tensors.
12. `has_var_leaf` ×4 names across modules.
13. Shape readers ×4 (`typing._shape_of`, `laws.base._shape_of`
    wrapper, `signature._shape_tuple` bypasses _SHAPE_RULES,
    `laws.cond._shape`).
14. Additive-cost skeleton duplicated 3× (flops/launch_aware/count).
15. **`optimize.py` is a 3,990-line god-module** (~95 defs); the
    `_bound_*` cluster (~530 lines) wants `bounds.py`; structural-
    signature cache block too.
16. Three parallel "find better member" front-ends (pipeline,
    diagram*, morphisms*) — research surface predating law pipeline,
    will drift.

## Inconsistencies

17. `opmeta *_OPS` projections are import-time snapshots — declared
    ops invisible to consumers (honest boundary but silent
    mis-classification risk).
18. `OpTable.full()` setdefault — ambient dicts silently win over
    canonical; aliasing mutates caller dicts.
19. Naming drift: `_shape_of` private+public+wrapper+_shape_tuple;
    `_INVALID` vs `_INVALID_COST` (latter even used as ns latency in
    roofline via _depth_cost special-case — fragile unit mixing).
20. `IR.inputs` (list) vs `IR.input_names` (set) — no consistency
    validation.
21. `term_from_data` accepts malformed dicts silently; bare-string
    leaf decodes as metavar.
22. `TensorType` accepts negative/zero dims — no validation.

## Hardcoded data — disposition

| Location | ~Size | Verdict |
|---|---|---|
| `attrs.ATTR_SCHEMA`/`ATTR_REQUIRED` | ~75/45 | KEEP — curated, validated |
| `opmeta` classified/arity/fold/ir_only | ~135/80/160/6 | KEEP — designated data home; `_FOLD_AT_LOWERING` mirrors torch lowerer across repo boundary w/ only comment — add boundary validation |
| `typing._infer_op_shape` match | ~150 spellings | MIGRATE gradually — biggest remaining "op table as code"; `register_shape_rule` seam exists |
| `basic._OP_FLOPS` | ~55+overrides | KEEP but fix silent default — unlisted ops fall to `1*n_out` (softmax/clamp/glu unpriced!) |
| `roofline` measured-op frozensets | ~8 | CROSS-REPO literal dup of kit_torch.calibrate classes — move to opmeta tags |
| `params` foldable/fusion sets | ~120 | DERIVE from opmeta — drift visible (#8) |
| `fusion` transparent/solver/scan sets | ~40 | Audit vs RELAYOUT/VIEWISH/COST_VIEW tags |
| `lawdata` tables | small | KEEP — curated law vocabulary |
| law banks (tensor.py 3,024 lines) | large | KEEP — laws are data by design |
| `ir._OP_REGISTRY` | ~30 | DELETE — dead |
| `optimize` delivered candidates, criteria vocab | small | KEEP |
| `opmeta` ATen spelling table | ~80 | KEEP — validated boundary map |

## Gaps

23. No shared traversal utility — highest-leverage refactor.
24. No registry-driven shape inference — `_SHAPE_RULES` exists but
    ~150-op match bypasses it.
25. `RuleSet` no composition combinator (`scoped()` placeholder).
26. **Unbounded global mutable caches**: `laws/base._SHAPE_MEMO`
    (pins terms forever, thread-unsafe), `EGraph._cost_memos`
    (accumulate per egraph lifetime) — long discovery loops → memory
    growth. Fix: weak-key memos or invalidation.
27. extract_alternatives/frontier correctness depends on op_repr
    injectivity (#1).

## What looks good

Const spelling-strict equality, ENode repr attr signatures,
_member_key deterministic ordering, frozen _Epoch matcher journals,
RuleSet collision detection, LowerResult fresh stats, OpTable
explicit composition, certificate egraph_dependent honesty,
ports/Backend contract layer genuinely backend-neutral,
_PeakBytesCost liveness sweep subtle but correct on traced cases,
pareto.py small+correct+honest.

## Priority order

1. #1 extract key, #2 leaf registry collision, #3 shape early-return,
   #8 fusion pointwise drift — silent wrong-answer/mispricing.
2. #4 congruence timing, #7 dead _OP_REGISTRY, #22 shape validation.
3. #9/#14 shared traversal + additive-cost factory, #15 split
   optimize.py, #17/#18 registry snapshot hazards.
4. Hardcode: _OP_FLOPS silent default, roofline cross-repo mirror,
   typing match → shape-rule migration.

## Coverage gap (not verified)

egraph/proof.py, laws/cond.py eval, autotune.py, regime.py,
carriers.py, diagram*.py, morphisms/* bodies, cost/{executor,
backend}.py, features/profile/perf_model/rulecache/trajectories/
failures/cells/metrics/timing/policies/search_env, laws bank files
(scan/attention/layout/factored/specials/headshare/priorart/minted/
serialize/store/tags). Follow-up pass on autotune+regime+
morphisms/reify recommended.

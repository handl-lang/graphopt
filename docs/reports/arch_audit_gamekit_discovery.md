# Cross-repo architecture audit — gamekit / mlopt-discovery / bench

Read-only audit (subagent). Scope: mlopt_discovery ~90 modules ~83k
lines, cap-threading path model_corpus→_probe_knobs→_case_tail→
_mint_tiers/TIER_SITES→genkernel.profitable→MetaArena, site-
collector family, shim layer, gamekit engine/players/registry/
gamedata, bench/drivers layout.

## 1. Correctness bugs / hazards in bounded paths

### 1.1 HIGH — Recursion-limit hazard on model_corpus/corpus_probe path
Bounded collectors cap *tree-expansion size* but every renderer they
guard is RECURSIVE, and nothing on this path raises setrecursionlimit:
census.shape_key (census.py:224-242), meta_arena._spec_of/_canon_spec/
_canon_concrete (307/327/354), play._fusion_pattern_sz/_reduction_
pattern (601-740), play._seq_deindex (1836-1983), workload_gen._leaves,
proposal._leaves (262-270 — recursive AND not id-deduped → exponential
on DAGs). setrecursionlimit(40_000) raised only in play.main,
_scan_sites, _seq/_seqm_scan_sites, meta_arena.meta_probe — NOT in
model_corpus()/corpus_probe()/generate_mutants()/coverage_report().
_AUTO_KEY_CAP=20_000 admits depth ~20k mutants → RecursionError, not
honest decline. Same via _SHAPE_KEY_CAP=10_000 shape_key and user-
passed large --spec-node-cap into _spec_of.
Fix: raise recursion limit inside model_corpus()/corpus_probe()
(or export_ir), or convert shape_key/_spec_of to iterative walks.

### 1.2 MEDIUM — spec_node_cap/spec_cap bounds two different metrics
_case_tail threads one value into: _mint_tiers spec_cap → compared
against _fusion_pattern_sz's SPELLED TREE SIZE (ops+Const/leaves+
attr entries) vs MetaArena spec_node_cap → compared against
census.op_expansion (OP NODES ONLY). _AUTO_SPEC_NODE_CAP=512
documented as op-expansion bound but for site collectors it's a
tree-node bound ~2-4x tighter. Silent semantic split.
Fix: count ops in _fusion_pattern_sz or split spec_tree_cap vs
spec_node_cap.

### 1.3 MEDIUM — time_budget is soft bound sold as hard
profitable docstring claims ~3x worst case — false when per-call est
large: reps_eff floor of 3 runs 3 reps >> budget; borderline re-
windows pay same reps_eff → unbounded in est.
Fix: document as target or enforce elapsed checks in _paired_time.

### 1.4 LOW-MEDIUM — over-cap skip accounting asymmetric
_mint_tiers dispatches caps by tier-name string; seq/seqm/cumsum
count *_sites_over_cap into stats but _fusion_sites/_reduction_sites
drop over-cap sites SILENTLY — no counter. Fragile elif chain too.
Fix: stats sink for every collector + signature/flags on TIER_SITES.

### 1.5 LOW-MEDIUM — max_handlers counts table entries not priced
claims — unmeasurable entries consume cap slots before filters.
### 1.6 LOW-MEDIUM — corpus_cov._pattern_occurs lacks DAG dedup
(corpus_cov.py:198-205) — tree-exponential on shared-DAG; sibling
meta_arena._concrete_occurs id-dedups. Latent hang on model-scale.
### 1.7 LOW — false-novelty edge in workload_gen.corpus_stats:
subterms over key_cap excluded → mutant equal to unkeyable corpus
subterm passes novelty gate as "novel".
### 1.8 LOW — global-state mutations never restored:
torch._dynamo.config.recompile_limit raised globally (genkernel:877,
play:3342); setrecursionlimit never restored (4 sites). workload_gen
._MAX_NODES monkey-patch DOES restore — the pattern others should copy.
### 1.9 LOW — quadratic canon recompute in MetaArena._sites
(meta_arena.py:1767) — O(n²), bounded by caps.

## 2. Duplicate abstractions

### 2.1 MEDIUM — three parallel episode drivers + Trajectory types
survived gamekit extraction: meta_arena.run_episode (missing
forced_action hook — silent delivery-path loss), arena.run_episode,
meta_player.train_player reimplements engine.train_policy. Domain
copies are the LIVE path; gamekit.engine.run_episode canonical.
Fix: route through engine or port forced_action.

### 2.2 MEDIUM — ~10 DAG/subterm walkers, subtly different contracts
(census._iter_subterms, impact._iter_subterms BYTE-IDENTICAL copy,
pay_gate._subterms byte-identical, corpus_cov/loop_propose/
index_abstract/form_propose variants, workload_gen._leaves,
proposal._leaves recursive+no-dedup EXPONENTIAL, emit._walk...).
Fix: one shared traversal module (mlopt_discovery._dagwalk or
promoted mlopt_core): subterms/ops/leaves with id-dedup+order.

### 2.3 MEDIUM-LOW — tree-expansion counted 4 ways
(census.term_expansion canonical; play._seq_expansion admits
"same recurrence" — could delegate; _fusion_pattern_sz third
variant; _seq_deindex _budget fourth mechanism).
### 2.4 LOW — _rule_budgets duplicated verbatim (proposal/impact —
impact docstring says "mirrors").
### 2.5 LOW-MEDIUM — three term→spec renderers (_spec_of variants
with different metavar naming; _spec_leaf_env inverts by repr —
fragile).
### 2.6 LOW-MEDIUM — _seq_scan_sites/_seqm_scan_sites ~90% identical
(~150 lines copy-paste; _seq_sites(site_fn=, record_fn=) driver).
### 2.7 Info — site-collector architecture otherwise well-factored:
TIER_SITES + genkernel.TIERS + _mint_tiers is a genuine registry —
the good pattern.

## 3. Missing shared abstraction — the "cap everything" gap

Caps added per-callsite, NO shared budget type: spec_cap kwargs,
spec_node_cap, key_cap, _MAX_NODES module-global monkey-patched,
max_handlers/time_budget/reps via referee dict, max_claims/
claim_time_budget TWO NAMES for same knobs, _AUTO_*/_SEQ_* literals.
model_corpus._probe_knobs is the ONLY place resolving explicit-vs-
auto-vs-unbounded; ex[...] stringly-typed board state
(ex["_referee"], ex["_spec_node_cap"], ex["_gen_claims_gated"]).
RECOMMENDATION (highest-leverage): frozen ProbeBounds/Budget
dataclass {spec_node_cap, key_cap, max_handlers, time_budget, reps,
shape_key_cap} + resolve(term_expansion) implementing _probe_knobs
auto logic once; thread as one object through _case_tail/_mint_tiers/
MetaArena/profitable instead of referee dict + spec_node_cap +
key_cap + ex keys. Fixes dual-metric (1.2) by making unit explicit.

## 4. Hardcoded vs data-driven

- lawdata genuinely the data home (MACROS/HANDLERS/CANDIDATE_LAWS +
  scripts/ bakers) — correct pattern.
- gamekit.gamedata.weights baked tables regenerated by graphopt
  scripts — cross-repo provenance asymmetry (baker in graphopt,
  table in gamekit).
- PARTS/LEGALS/RANKS/SCORERS registries — curated extension points,
  appropriate.
- vocabdir.py (1749 lines, plan-0033 single-source YAML/JSON +
  generated projections) vs vocab.py (1105 lines, op-classification)
  — related but distinct; confusing pair, rename/cross-ref needed.
- Cap constants (_AUTO_*/_SEQ_*/_MAX_*) = main hardcoding debt.

## 5. gamekit/benchkit

- Substrate-free contract RESPECTED (torch once, lazy, documented).
- benchkit Checkpoint used by overnight.py; impact.py:70-74 inserts
  REPO_ROOT into sys.path for bench.suites — package→repo-root
  back-edge breaking non-editable installs; TermCase/_bench_cases/
  model_cases are de-facto corpus API consumed by census/
  workload_gen/model_corpus/vocab — promotion candidate.
- play.py is a 4,938-line god module (DOMAINS+CLI+4 board factories+
  ~2,200 lines gen-domain collector machinery) — collector family
  wants gen_sites.py/scan_sites.py extraction.

## 6. Pipeline composition

gen tail well-composed (model_corpus→export_ir→_case_tail→_mint_tiers
→profitable→gen_sink→gen_probe→MetaArena→deliver). Special-casing
at edges (_claim_cost_gate, _fold_cost_gate, _gen_claims_gated,
pay_gate) — documented honest gates but communicate via ex[...] string
keys not typed state. ProbeBounds collapses referee dict +
spec_node_cap + key_cap + four ex keys into one value.

## Priority-ordered fix list

1. HIGH — recursion limit on model_corpus path (§1.1).
2. MEDIUM — spec_node_cap dual-metric + ProbeBounds together (§1.2+§3).
3. MEDIUM — meta_arena.run_episode forced_action; driver copies (§2.1).
4. MEDIUM — shared DAG-walk util; proposal._leaves dedup (§2.2).
5. MEDIUM — time_budget honesty; over-cap stats symmetry (§1.3/1.4).
6. LOW — _seq_expansion→term_expansion, _rule_budgets dedup,
   seq/seqm merge, canon precompute, max_handlers doc precision,
   corpus_cov dedup, impact sys.path hack.

## Unverified

Arena/MetaArena full Board Protocol conformance (duck-typing likely
works); loop_propose/index_abstract _subterms contracts; benchkit
Checkpoint bypass in other drivers; test status of proposals.

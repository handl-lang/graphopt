::: warning Vendored — do not edit here
Synced from `graphopt:docs/public_api.md` @ `eea02f6` — the
phase-1 frozen manifest (`API_VERSION = "0.1.0"`). Edit upstream and
re-vendor; this copy drifts on purpose only at the repo split. The
`research-mlopt` → `graphopt` rename has landed; the `eqsat`/`gamekit`
carve-outs land at phase 4 — each row tags its destination repo.
:::

# The frozen public API — plan 0032 phase 1

> **Version:** `API_VERSION = "0.1.0"` — `from mlopt_core import
> API_VERSION`.  A downstream domain (a board, a player, a backend
> sink, a referee) declares the surface it targets against this
> constant.  Bump the **MINOR** digit for additive surface changes
> (a new frozen name, a new optional kwarg), the **MAJOR** digit for
> any change to a frozen name's signature or meaning.  The version
> stays `0.x` until the phase-4 repo extraction, when each new repo
> versions its own slice.

This file is the manifest of the contracts a later repo split must
not break.  Everything below is **frozen**: names, canonical import
paths, and the signature pins the contract tests carry
(`tests/test_public_api.py`).  Each row names its destination repo
per the plan's cut map (`eqsat` = substrate, `gamekit` = the game
layer, `graphopt` = the tensor-domain contents).

## What "frozen" means

* **Frozen** — a name in a table below.  Renaming, retyping, or
  re-homing it without an `API_VERSION` major bump is a contract
  break.
* **Public, unfrozen** — any other name in a module's `__all__`.
  Public today, not promised across the split (e.g. the morphism
  algebra re-exported from `mlopt_orchestrator`).
* **Internal** — everything else.  Underscore names stay internal
  even where a module re-exports them for compat (`egraph`'s
  `_term_match`/`_term_instantiate`/`_LeafRegistry`; `cost`'s
  `_shape_of` family; `players`'s `_dot`/`_grad_row` compat
  re-exports).  "Reachable" is not "public".

## Substrate contracts (→ `eqsat`)

| Contract | Canonical location | Frozen surface |
|---|---|---|
| Term IR | `mlopt_core.ir` | `IR`, `Op` (incl. `Op.make`), `Var`, `Param`, `Const`, `TensorType`, `op_repr`, `op_repr_dag`, `term_to_data`, `term_from_data` |
| E-graph engine | `mlopt_core.egraph` | `EGraph`, `ENode`, `EClass`, `UnionFind`, `Rewrite` |
| `EGraph` protocol | `mlopt_core.egraph.EGraph` | `add_term(term) -> eid`, `find(eid)`, `rebuild(classes=None)`, `extract_best(eid, cost_fn, **kw)`, `certificate(src, dst, root_eid=...)`, **`run(rules, root_eid, *, max_iterations=100, max_nodes=100_000, rule_budgets=None, stop="fixed_point", patience=3, cost_fn=None, policy=None) -> stats`** — `stats` carries `stop`, `n_enodes`, `n_classes`, `rule_fires`, `rule_budgets` |
| Certificates | `mlopt_core.egraph` | `Certificate`, `CertStep`, `ProofEdge`, `CertificateVerificationError`, `cert_to_data`, `cert_from_data`, **`verify_certificate(term, cert, *, strict=False)`** — the standalone derivation replay |
| DAG pricing | `mlopt_core.cost` | **`dag_cost(term, cost_fn, memo=None)`** — the shared-subtree-honest price; plus the shipped models `count_cost`, `flops_cost`, `param_bytes_cost`, `param_bytes_cost_for`, `backend_cost`, `executor_cost_for`, `fused_cost_for`, `lowering_aware_cost_for`, `depth_cost`, `depth_cost_for`, `roofline_cost`, `roofline_cost_for`, `launch_aware_cost`, `fusion_regions`, `fusion_member_key`, `CostModel`, `LOWERINGS` |
| Ports | `mlopt_core.ports` | **`Sink`** (`lower`/`verify`/`executors` on top of **`Capabilities`**'s `supported_ops`/`ops`), **`Source`**, **`Engine`** (the `engine=` contract below), `Composer`, `Executor`, `ExecutorSpec`, `Runner`, `Meter`, `Verifier`, `VerifyResult`, `CostFn`, `Criterion`, `Policy`, `Strategy`, `TaskMetric`, `PerformanceModel`, `Profiler`, `OpRegistry`, `RuleSetLike`, `RuleLike`, `RuleSetProvider`, `Binding`, `TimingResult`, `signature_conforms` |
| Law machinery | `mlopt_core.laws` | `RuleSet`, `preset`, `DEFAULT`, `FULL`, `SYMMETRY`, `ALL_RULES`, `tags`, `provenance` — the *names* are frozen; the **bank contents** (`ALL_RULES` membership, `MACHINE_SHIPPED` rows) are graphopt's, not the substrate's |
| Provenance registry | `mlopt_core.laws.provenance` | `MACHINE_SHIPPED`, `MACHINE_STORED`, `MACHINE_LAWS`, `law_provenance(rule) -> "machine" \| "human"` |
| The pipeline | `mlopt_orchestrator` (→ `pipeline`) | `Optimizer`, **`search(model, x, *, source, capabilities=None, criteria=None, cost_fn=None, rules=None, engine=None, policy=None, …) -> SearchResult`**, `lower`, `SearchResult`, `LowerResult`, `Backend`, `discover_alternatives`, `DEFAULT_RULES`, `OptimizationResourceError`, strategies `Monolithic`/`Compositional`/`Autotuned`, runners `IdentityRunner`/`ChainedRunner`/`runner_candidate`, criteria family (`Criterion`, `Criteria`, `Blend`, `criteria_cost`, `peak_bytes_cost`, `LatencyCriterion`, `MemoryCriterion`, `FlopsCriterion`, `DepthCriterion`, `CompiledCriterion`, `PredictedCriterion`), `delivered_cost_for`, `structural_key` |
| `engine=` | `search`/`Optimizer` kwarg | An `Engine`-port value — instance, class, or zero-arg factory.  Engines are **never auto-detected**; `stats["engine"]` records which ran (`"python"` reference default, `"native"` for `mlopt_native.NativeEngine`).  The port covers the *search* only — certificates and the non-local passes are Python-engine capabilities |

## Game contracts (→ `gamekit`)

Plan 0032 lane A landed the in-repo extraction: these modules now
live in `packages/gamekit` as `gamekit.*`, and the old
`mlopt_discovery.{engine,scoring,advantage,players,lawdata.game}`
paths are re-exporting shims (see `TestGamekitCompatShims` in
`tests/test_public_api.py`).  Canonical paths are `gamekit.*`.

| Contract | Canonical location | Frozen surface |
|---|---|---|
| `Board` protocol | `gamekit.engine` | `Board` (`observe() -> state` / `step(action) -> (state, report)`; `state.done` terminates), `Report` (floor: `reward`, `terminal`), `Trajectory` (`.reports`, `.total`, `.terminal`), `run_episode(board, player, budget)`, `evaluate(arms, cases, board_of, *, budget)`, `train_policy(player, cases, board_of, *, budget)` |
| Player registry | `gamekit.players.registry` | `PlayerSpec` (`name`/`kind`/`boards`/`schema`/`train`/`artifact`/`record`/`variant`), `BOARDS`, `EPISODE_BOARDS`, `names()`, `spec(name)`, `specs()`, `for_board(board)`, `module(name)`, **`make(name, domain, *, seed=0, ctx=None) -> (i -> player)`** — `ValueError` on an unplayed board, never a silent run |
| Player machinery | `gamekit.players` | `LinearPolicy`, `DomainSpec` (incl. `to_record`/`build_from_record`), `LearnedPlayer`, `PlayHist`, the feature parts `Bias`/`StateBroadcast`/`PlayRates`/`OpOneHot`/`ActionFeats`/`RelFeats`, `compose_featurizer`, `register_part`/`register_legal`/`register_rank`, `spec_from_record`, `PARTS`/`LEGALS`/`RANKS` |
| Scoring spec | `gamekit.scoring` | `ScoreGate{axis, threshold, direction}`, `ScoringSpec{gates, weights, references}`, `GateVerdict`, `ScoreVerdict`, `score(spec, measured) -> ScoreVerdict`, `AXES` (`"speed"`, `"memory"`, `"launch"`, `"params"`), `DIRECTIONS` — the referee's value function **as data**; see below |
| Substrate-free kernel | import-linter contract | `gamekit.engine`, `gamekit.scoring`, `gamekit.calibrate`, `gamekit.advantage`, `gamekit.gamedata`, `gamekit.players._base`, `gamekit.players.registry` import nothing outside the stdlib — the `gamekit ⊥ eqsat` edge, enforced (see `pyproject.toml`) |

## Domain-side contracts (→ `graphopt`)

| Contract | Canonical location | Frozen surface |
|---|---|---|
| Law verification | `mlopt_discovery.verifier` | **`verify_law(lhs, rhs, rules, *, max_iterations=30, max_nodes=200_000, rule_budgets=None) -> LawResult`**, `LawResult` (`derivable`, `direction`, `witness_rules`, `replayable`, `cert`) — the admission side the phase-2 referees consume |
| Domain registry | `mlopt_discovery.play` | `DOMAINS`, `Domain` (`cases`/`board_of`/`legal`/`featurizer`/`features`/`scripted`/`deliver`/`spec`), `play(domain, case, player, *, budget)` — the `Domain` *shape* is the contract; the `DOMAINS` *contents* are domain data |

## The scoring spec — the referee seam

The plan's wording is the design: **the value spec is gamekit's; the
measurement is the gauntlet's.**

```python
from gamekit.scoring import ScoreGate, ScoringSpec, score

spec = ScoringSpec(
    gates=(
        ScoreGate("speed", 0.95),     # delivered latency ≤ 0.95 × ref
        ScoreGate("params", 1.0),     # parameter bytes at most the ref
        ScoreGate("win_rate", 0.5, direction="at_least"),
    ),
    weights={"speed": 1.0, "memory": 0.25},
    references={"memory": 4.0e8},     # ungated axes need a normaliser
)
verdict = score(spec, measured)       # measured: {axis: value} — the
                                      # gauntlet's emission shape
verdict.passed                        # every gate cleared?
verdict.value                         # Σ w·(measurement/bar) — nan
                                      # when a weighted axis is absent
verdict.gates                         # per-gate GateVerdicts
verdict.missing                       # axes the measurement lacked
```

The two sides meet only through a plain `Mapping[str, float]`:
measurement code never imports this module (the gauntlet lives on the
substrate side of the split), and this module imports no measurement
machinery (the "game kernel is substrate-free" import contract).
Phase-2 lane B lands the referee registry
(`naive`/`oracle`/`pay`/`sweep`/`full`) and the matrix runner as
*consumers* of this contract.

## Boundary notes — the honest seams

* **`imitate_policy`** (`gamekit.engine`) is a documented
  driver that was never added to `__all__` — reachable but unlisted.
  It is **not** frozen; promote it deliberately or treat it as
  internal (phase-2 decision, recorded in the ADR entry).
* **Compat re-exports are internal** even at module level — anything
  carrying a `_` prefix or a `# noqa: F401 — compat re-export` marker
  (`egraph._term_match`, `players._Hist`, `cost._memo_dispatch`, …)
  stays out of the contract regardless of how reachable it is.
* **Registry contents vs registry shape**: `DOMAINS`, `ALL_RULES`,
  `MACHINE_SHIPPED` contents, and the `lawdata` weight tables are
  *data* — the names and access functions are frozen, the rows are
  each destination repo's business.
* **`mlopt_native`** is an opt-in engine behind `engine=`, not a
  frozen import path: it is excluded from the uv workspace, so the
  contract binds the `Engine` *port*, not the package.

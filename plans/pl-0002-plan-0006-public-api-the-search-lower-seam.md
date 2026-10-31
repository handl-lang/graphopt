+++
id = "PL-002"
title = "Plan 0006 — Public API: the `search` / `lower` seam"
status = "open"
priority = "medium"
+++

---
title: 'Plan 0006 — Public API: the `search` / `lower` seam'
type: plan
date: 2026-10-14
status_note: accepted
---

# Plan 0006 — Public API: the `search` / `lower` seam

Part of a three-plan program — each lands independently and green:

* **0006 (this)** — the API seam: two phase verbs, result objects, a
  `Strategy` seam, renamed runners. **Additive**: package layout
  untouched, old names survive as one-line wrappers.
* **0007** — backend abstraction: `Composer` / `Meter` /
  `Sink.executors` / the `Backend` value; make the orchestrator
  backend-neutral (in place).
* **0008** — package split: `catopt_orchestrator` →
  `catopt_orchestrator`, add `catopt_cuda`, reduce the `catopt`
  façade to a thin torch-free re-export, retire `optimize_*`,
  migrate call sites.

## Goal

The public API grew as free functions around one monolithic
pipeline; the `Source`/`Sink` seam is half-wired and there is no
*strategy* seam.

| function | takes `source`/`sink`? | kind |
|---|---|---|
| `optimize_model` | yes | search + lower |
| `discover_alternatives` | yes | search (frontier) |
| `optimize_model_autotuned` | yes | search + N lowers |
| `optimize_compositional` | **no** | per-block search + lower |
| `regime_dispatch` / `regime_frontier` | **no** | search + per-regime lower |

Defects: two entry points bypass the ports; defaults are rebuilt per
call; `optimize_*` hides two axes (search vs delivery); policies are
hardcoded branches, not composable objects.

## Decisions

1. The entry object is **`Optimizer`**.
2. **Generic machinery lives in `catopt_core`** — all abstractions
   (ports/protocols) and the result types.
3. `discover_alternatives` returns a **`SearchResult`** (no dict).
4. **Split `Sink`** into `Capabilities` (op set + registry) and
   `Sink` (`Capabilities` + lower/verify).
5. **New API only** — the `optimize_*` names retire; until 0008 they
   survive as one-line wrappers so this plan stays additive.
6. **Runner renames** — `CompiledRunner` → `TorchCompileRunner`,
   `GenericRunner` → `IdentityRunner`; name values
   `"torch_compile"` / `"identity"`; autotune candidates
   `"torch_compile"` / `"torch_compile_generic"`.
7. **Strategy layer** — `Monolithic` / `Compositional` / `Autotuned`
   behind a core `Strategy` protocol; `Optimizer.optimize(...,
   strategy=...)` is the combined entry.
8. **Return objects** — `search → SearchResult`, `lower →
   LowerResult` (unpacks as `(module, stats)`; `.stats` = search ∪
   lower).
9. **`specialize_causal`** — the causal-mask const fold is an
   opt-out search pass; `Capabilities.ops` is its declared need.
10. **Explicit ports — no implementation is assumed.** `Optimizer`
    requires `source` and `sink`; `search`/`lower` require the ports
    they use. There is **no torch default** and nothing is
    auto-selected. Torch is chosen by importing `catopt_torch`.
11. **Lifting is its own verb.** `lift(model, x, source=) -> Lifted`
    decouples the frontend from the search; `search` takes a `Lifted`
    (IR + leaves + reference), never a model. The e-graph's
    non-local "lifts" are renamed to non-local **passes**.
12. **Verification is its own verb.** `lower` does **not** verify —
    it only materialises the runnable. `verify(result, x, sink=,
    reference=None)` compares it to the reference. `Optimizer.optimize`
    orchestrates lift → search → lower → verify. (Today verification
    is oddly `verbose`-gated; it becomes explicit and always
    available.)

## Design

Four phases, one job each — lifting, searching, lowering and
verifying are all separate:

```
lift   : (Source)        -> model -> IR + leaves
search : (Capabilities)  -> IR -> EGraph -> saturated -> extracted term
lower  : (Sink + Runner) -> term -> runnable
verify : (Sink)          -> reference vs runnable -> VerifyResult
```

### The seam artifact (`catopt_core/pipeline.py`)

```python
@dataclass(eq=False)
class Lifted:
    ir: IR
    leaves: dict[str, Any]
    reference: Any = None          # the original model/runnable (opaque)

@dataclass(eq=False)
class SearchResult:
    ir: IR
    eg: EGraph
    root_eid: int
    term: Any
    param_values: Mapping[str, Any]
    stats: dict[str, Any]          # the SEARCH record
    cost_fn: CostFn | None = None
    source: Any = None             # provenance (TYPE_CHECKING)
    reference: Any = None          # carried from Lifted, for verify
    def alternatives(self, top_k=8, cost_fn=None): ...
    def certificate(self, a, b): ...

@dataclass(eq=False)
class LowerResult:
    module: Executor
    stats: dict[str, Any]          # SEARCH ∪ LOWER record
    reference: Any = None          # carried from SearchResult
    verified: VerifyResult | None = None   # set by verify() / optimize()
    def __iter__(self):            # mod, stats = lower(...)
        yield self.module
        yield self.stats
```

No `.frontier` method — regime lives in the orchestrator package and
core may not import it.

### The verbs (`catopt_orchestrator/optimize.py`)

Four verbs, one job each:

```python
def lift(model, example_input, *, source) -> Lifted: ...

def search(lifted, *, capabilities=None, criteria=None,
           cost_fn=None, ruleset="all", max_iterations=100,
           max_enodes=100_000, symmetry_budget=2048,
           max_memory_mb=None, specialize_causal=True,
           verbose=False) -> SearchResult: ...

def lower(result, example_input, *, sink, runner=None,
          verbose=False) -> LowerResult: ...

def verify(result, example_input, *, sink, reference=None,
           rtol=1e-4, atol=None) -> VerifyResult: ...
```

`lift` delegates to `source.to_ir`; **`search` takes IR, not a
model** — pure and testable with no frontend. **`lower` does not
verify** — it only materialises the runnable. `verify` is its own
step: it compares the optimized runnable against `reference`
(defaulting to the one carried in `result`). `Optimizer.optimize`
orchestrates all four.

`capabilities` on `search` is optional — it supplies backend-relative
pricing and the const-fold registry; omitted, pricing is
backend-agnostic. The param is named `capabilities`, not `sink`: it
takes a `Capabilities`, not a `Sink`.

`ruleset` / `symmetry_budget` here are superseded by **0009**
(`rules: RuleSet = DEFAULT`; the budget keys off the `EXPANSIVE`
tag).

### Naming: `lift` means model → IR

The e-graph's non-local transformations (today `lift_scan_to_trace`,
`lift_scan_to_applyd`, `omd_tree_lift`, …) are renamed to **non-local
passes** — `pass_scan_to_trace`, … — so `lift` unambiguously means
"lift a model into the IR". They already are described as passes in
the docs; only the identifiers change.

### `Capabilities` vs `Sink` (core)

```python
@runtime_checkable
class Capabilities(Protocol):
    supported_ops: frozenset[str]
    ops: OpRegistry

@runtime_checkable
class Sink(Capabilities, Protocol):
    def lower(self, ir, params=None) -> Executor: ...
    def verify(self, ref, opt, inputs, *, rtol=1e-4, atol=None): ...
```

`isinstance(TorchSink(), Sink)` still holds.

### The strategy seam (core protocol)

```python
@runtime_checkable
class Strategy(Protocol):
    name: str
    def run(self, model, x, *, optimizer, **kw) -> LowerResult: ...

class Monolithic: ...   # lift + search + lower + verify — the default
```

`Optimizer.optimize(model, x, strategy=None)` → `Monolithic()`.
`Compositional` / `Autotuned` are introduced here as objects too
(they still touch torch until 0007 abstracts `Composer`/`Meter`).

### `Optimizer`

```python
@dataclass
class Optimizer:
    source: Source                     # required — no default
    sink: Sink                         # required — no default
    criteria: Criterion | Criteria | dict | None = None
    runner: Runner = field(default_factory=IdentityRunner)

    def lift(self, model, x, **kw) -> Lifted: ...
    def search(self, lifted, **kw) -> SearchResult: ...
    def lower(self, result, x, **kw) -> LowerResult: ...
    def verify(self, result, x, **kw) -> VerifyResult: ...
    def optimize(self, model, x, *, strategy=None, verify=True,
                 **kw) -> LowerResult: ...
    def discover(self, model, x, **kw) -> SearchResult: ...
```

`source` / `sink` are **required** — there is no assumed backend.
`runner` defaults to `IdentityRunner` (the no-op delivery, a
backend-neutral implementation); pass one explicitly to change it.
`optimize` is `lift + search + lower + verify` (set `verify=False` to
skip); `discover` is `lift + search`. (In 0007 `source`/`sink` become
a single required `backend`.)

### Stats ownership

* `SearchResult.stats` — search record.
* `LowerResult.stats` — fresh dict: search record + lowering keys;
  `lower` never mutates `SearchResult.stats`.

### Runner renames

| class | `name` | was |
|---|---|---|
| `IdentityRunner` | `"identity"` | `GenericRunner` / `"generic"` |
| `TorchCompileRunner` | `"torch_compile"` | `CompiledRunner` / `"compiled"` |
| `CudaGraphRunner` | `"cuda_graph"` | unchanged |
| `ChainedRunner` | joined | unchanged |

Moves the four `tests/test_runners.py` assertions. `stats["compiled"]`,
`delivers_compiled`, core `LOWERINGS` and `CompiledCriterion` are
unchanged.

## Public surface (after 0006)

```python
from catopt_orchestrator import (
    Optimizer, lift, search, lower, verify,
    Lifted, SearchResult, LowerResult,
    Monolithic, Compositional, Autotuned, discover_alternatives,
)
from catopt_torch import (
    TorchSource, TorchSink, TorchCompileRunner, IdentityRunner,
)
```

Explicit construction:

```python
opt = Optimizer(source=TorchSource(), sink=TorchSink())
mod, stats = opt.optimize(model, x)
```

`optimize_model` / `optimize_compositional` /
`optimize_model_autotuned` remain as deprecated one-line wrappers
(removed in 0008).

## Execution stages

1. **Runner renames** (independent) — classes, `name` values,
   autotune candidate names, the four test assertions.
2. **Core abstractions** — `Capabilities` + `Strategy` in
   `catopt_core.ports`; `catopt_core/pipeline.py` with `Lifted`,
   `SearchResult` + `LowerResult`. Additive.
3. **Verbs** — split `optimize_model` into `lift` / `search` /
   `lower` / `verify` (required ports; `search` takes IR, not a
   model; `lower` no longer verifies); `optimize_model` becomes
   `verify(lower(search(lift(...))))`; `specialize_causal`;
   `discover_alternatives` → `SearchResult`; rename the e-graph
   non-local "lifts" to passes; thread `source`/`sink` into
   compositional + regime.
4. **Strategy seam** — `Strategy`/`Monolithic`/`Compositional`/
   `Autotuned` objects; `Optimizer` (required ports).

## Tests & gates

`tests/test_search_lower_split.py`:

* `lift` → `Lifted`; a hand-built IR (no frontend) searches end to
  end;
* `search` → `SearchResult`; `lower` → `LowerResult`; unpacking;
  `LowerResult.stats` ⊇ `SearchResult.stats`;
* `lower` alone does **not** verify; `verify` returns a
  `VerifyResult`; `optimize(verify=False)` skips it;
* `NumpySink` flows through `search` + `lower`;
* `specialize_causal=False` skips the fold;
* `Optimizer.optimize(strategy=Compositional()/Autotuned(...))`;
* a custom `Strategy` runs through `Optimizer.optimize`;
* `Optimizer()` with no ports raises;
* wrappers still reproduce the old output.

Coverage stays 100%; `ty` clean (no excludes); ruff `D`; radon (new
functions ≤ rank C); vulture; import-linter (core imports no
orchestrator).

## Capabilities & usage

```python
# 0006 paths — renamed to catopt_orchestrator in 0008
from catopt_orchestrator import (Optimizer, search, lower,
                             Compositional, Autotuned,
                             LatencyCriterion, MemoryCriterion)
from catopt_torch import TorchSource, TorchSink, TorchCompileRunner

# explicit, configured once
opt = Optimizer(
    source=TorchSource(), sink=TorchSink(),
    criteria=LatencyCriterion() * 0.7 + MemoryCriterion("peak") * 0.3,
    runner=TorchCompileRunner(),
)
mod, stats = opt.optimize(model, x)

# phases — lift, search, lower, verify are separate
lifted = lift(model, x, source=TorchSource())    # Lifted (IR + leaves)
res    = search(lifted)                          # SearchResult
out    = lower(res, x, sink=TorchSink())         # LowerResult
mod, stats = out
report = verify(out, x, sink=TorchSink())        # VerifyResult

# search once, deliver many
eager = lower(res, x, sink=TorchSink())
fast  = lower(res, x, sink=TorchSink(), runner=TorchCompileRunner())

# strategies are data
opt.optimize(m, x, strategy=Compositional())
opt.optimize(m, x, strategy=Autotuned(candidates=("generic",)))
```

### Before → after

| capability | before | after |
|---|---|---|
| optimize a model | `optimize_model(m, x)` | `Optimizer(source=…, sink=…).optimize(m, x)` |
| reuse config | re-instantiate per call | `opt = Optimizer(...)` |
| lift a model to IR | fused into `optimize_model` | `lift(m, x, source=…)` → `Lifted` |
| search an IR | ✗ (needed a model) | `search(lifted)` — no frontend |
| lower a term | fused with verify | `lower(res, x, sink=…)` |
| verify | fused + `verbose`-gated | `verify(out, x, sink=…)` |
| phases | ✗ (coupled) | `lift` → `search` → `lower` → `verify` |
| one search, N deliveries | ✗ (re-searches) | `lower(res, runner=…)` |
| inspect mid-pipeline | discover's dict | `SearchResult.eg/term/stats` |
| add a policy | edit orchestrator | implement `Strategy` |
| choose a backend | assumed torch | pass `source=` / `sink=` |

## Deferred

* **0007** — `Composer` / `Meter` / `Sink.executors` / the `Backend`
  value; backend-neutral `Compositional` / `Autotuned` / `regime`.
* **0008** — package layout, `optimize_*` removal, call-site
  migration, façade reduced to a thin torch-free re-export.

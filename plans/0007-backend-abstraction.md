# Plan 0007 — Backend abstraction

Status: accepted

Second of three — depends on **0006** (the API seam). Each lands
independently and green.

## Goal

Make the orchestrator **backend-neutral**: it orchestrates any
backend through ports alone and imports **no** torch. This is what
turns "any source or sink" into "any backend", and it makes
`Compositional` / `Autotuned` / `regime` reusable outside torch.

Done **in place** (the package is still `catopt_orchestrator`), so that
when 0008 renames it to `catopt_orchestrator`, the result is already
torch-free.

The obstacle: the strategies are torch-coupled by nature —
`Compositional` walks `torch.nn.Module` trees with forward hooks,
`Autotuned` builds `torch.compile` / CUDA-graph candidates and times
them, `regime` lowers through torch carrier executors. Abstract
those three couplings and evict the remaining torch-coupled modules
(`export`, `calibrate`'s measurement half, the torch runners).

## New core ports

```python
# catopt_core/ports.py

@runtime_checkable
class Composer(Protocol):
    """Backend-native model structure — the per-block pass."""
    def blocks(self, model, *, predicate=None) -> list[Any]: ...
    def capture_inputs(self, model, blocks, example_input) -> Mapping: ...
    def graft(self, model, replacements: Mapping) -> Any: ...
    def clone_sharing(self, model) -> Any: ...

@dataclass(frozen=True)
class TimingResult:
    median_s: float
    iqr_s: float
    n_calls: int

@runtime_checkable
class Meter(Protocol):
    """Measure a runnable on the backend's runtime."""
    def time(self, runnable, inputs, *, warmup=5,
             n_calls=30) -> TimingResult: ...

@dataclass(frozen=True)
class ExecutorSpec:
    """A named lowering variant (promoted from regime)."""
    name: str
    lower: Callable[[IR, dict | None], Executor]
    accepts: Callable[[Any], bool]
    engaged: Callable[[Executor], bool]
    carrier: tuple[frozenset[str], frozenset[str], frozenset[str]] | None

@runtime_checkable
class Sink(Capabilities, Protocol):
    def lower(self, ir, params=None) -> Executor: ...
    def verify(self, ref, opt, inputs, *, rtol=1e-4, atol=None): ...
    @property
    def executors(self) -> Mapping[str, ExecutorSpec]: ...
```

`ExecutorSpec` is core-legal as re-typed: `lower` returns the core
`Executor`, `engaged` takes one, and `carrier` is neutral op-name
sets.

### Cross-pairs are optional

`Composer` covers only the **per-block** pass. The **cross-pair**
pass (adjacent-block joint analysis) builds backend-native joint
micro-models — in torch, `nn.Module` subclasses (`_JointPair`,
`_FusedPair`, `_Zero`) plus residual / io / boundary probes. That is
deeply backend-specific and is deliberately **not** part of the
`Composer` protocol.

Instead `Compositional` uses an **optional** hook: if the composer
exposes `cross_pairs(model, blocks, io)`, the pass runs; otherwise it
is skipped (`max_cross_pairs=0` semantics). The torch composer ships
one; a backend without it simply skips cross-pairs. This keeps
`Composer` minimal and stops 0007 ballooning into a research
project.

## The `Backend` value — no registry

```python
# catopt_core/pipeline.py
@dataclass(frozen=True)
class Backend:
    source: Source
    sink: Sink
    composer: Composer
    meter: Meter
```

`Backend` is a **plain value**: constructed and passed explicitly.
There is **no registry, no `register_backend`, no `default_backend`**
— nothing auto-registers on import, nothing is assumed. Name-based
selection is user code (`BACKENDS = {"torch": TorchBackend()}`).

`catopt_torch` exposes `TorchBackend` explicitly:

```python
from catopt_torch import TorchBackend
TorchBackend()   # Backend(TorchSource(), TorchSink(),
                 #         TorchComposer(), TorchMeter())
```

## `Optimizer` takes a backend

```python
@dataclass
class Optimizer:
    backend: Backend | None = None
    source: Source | None = None
    sink: Sink | None = None
    composer: Composer | None = None
    meter: Meter | None = None
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

`optimize` is `lift + search + lower + verify`; `search` takes a
`Lifted`, not a model.

Two explicit forms:

```python
Optimizer(backend=TorchBackend())                     # the bundle
Optimizer(source=TorchSource(), sink=TorchSink(),
          composer=TorchComposer(), meter=TorchMeter())   # the ports
```

`__post_init__` resolves the ports from `backend=` (when given) and
raises if the required ones are missing. **No default backend.**

## Neutralise the strategies

| strategy | was torch-coupled by | now uses |
|---|---|---|
| `Monolithic` | — | `source` / `sink` (unchanged) |
| `Compositional` | `nn.Module` walk, hooks, graft | `composer` (per-block) + optional `cross_pairs` |
| `Autotuned` | `torch.compile` candidates, `cuda.synchronize` timing | `meter`, `sink.executors`, `Runner`s |
| `regime` | `EXECUTORS` torch lowerers | `sink.executors` |

All four live in `catopt_orchestrator` and name no torch symbol.

## Evict the torch-coupled modules

For `catopt_orchestrator` to import no torch, the remaining torch-coupled
modules leave it now (they would otherwise be pulled in by the
package `__init__`):

| from (`catopt_orchestrator`) | to |
|---|---|
| `export.py` | `catopt_torch` |
| `calibrate.py`: `TargetProfile` + correction math | `catopt_core/profile.py` |
| `calibrate.py`: `calibrate()` + microbenchmarks | `catopt_torch` |
| `runners.py`: `TorchCompileRunner` | `catopt_torch` |
| `runners.py`: `CudaGraphRunner` | `catopt_cuda` (new) |
| `runners.py`: `IdentityRunner`, `ChainedRunner` | stay (generic) |
| `runners.py`: `Runner` protocol | `catopt_core.ports` |
| `criteria.py`: `Criterion` protocol | `catopt_core.ports` |

## Execution stages

1. **Core ports** — `Composer`, `Meter`, `TimingResult`,
   `ExecutorSpec`; add `Sink.executors`; move `Runner` / `Criterion`
   protocols. Additive.
2. **`Backend` + `Optimizer`** — the value, the `backend=`/ports
   resolution; no registry.
3. **Neutralise `Compositional`** — route blocks / capture / graft /
   clone through `composer`; cross-pairs via the optional
   `cross_pairs` hook (torch ships it; others skip).
4. **Neutralise `Autotuned`** — candidates from `sink.executors` +
   `Runner`s; timing through `meter`.
5. **Neutralise `regime`** — read `sink.executors`.
6. **Evict torch modules** — `export`, `calibrate` split,
   `TorchCompileRunner` → `catopt_torch`; `CudaGraphRunner` →
   `catopt_cuda`.
7. **Boundary test** — assert `catopt_orchestrator` imports no torch/cuda.

## Tests & gates

* A **fake backend** (`catopt_test_backend` fixture) — a minimal
  `Source`/`Sink`/`Composer`/`Meter` with no torch — drives
  `Monolithic`, `Compositional`, `Autotuned` end to end. This is the
  conformance evidence that the abstraction is real (the analogue of
  `tests/test_pluggable_sink.py`).
* Import test: `torch` / `catopt_torch` / `catopt_cuda` absent from
  `catopt_orchestrator`'s module graph.
* `Optimizer()` with no backend or ports raises the documented error.
* Coverage 100%; `ty` clean; ruff `D`; radon; vulture; import-linter
  extended with the orchestrator∌torch edge.

## Risks

* **`Composer` scope** — kept minimal on purpose: per-block
  structure only. The cross-pair pass is an optional hook, so a
  backend can reach parity without porting torch's joint
  micro-models. Land `Compositional` neutralisation last, with the
  torch composer as the reference implementation.
* **Behaviour parity** — the neutralised strategies must reproduce
  the torch path exactly; keep `tests/test_compositional*` and
  `tests/test_autotuned*` as the oracle.
* **Eviction churn** — moving `export` / `calibrate` changes import
  paths used by tests; update them in the same stage.

## Outcome

After 0007 `catopt_orchestrator` is backend-neutral and torch-free; a new
backend is a package that builds a `Backend` and passes it. 0008 then
renames the package and thins the façade to a torch-free re-export.

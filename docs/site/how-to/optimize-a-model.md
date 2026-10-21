# Optimize a model

<!-- provenance: graphopt:docs/api.md@bac1a61 + mlopt_torch.export + eqsat:packages/mlopt-orchestrator/src/mlopt_orchestrator/optimize.py -->

Take a `torch.nn.Module` and an example input to a verified, delivered
optimized module. Assumes [the dev environment is set up](../tutorials/setup);
for the guided walkthrough see [your first `search()`](../tutorials/first-search).

## 1. Pick a backend and run the pipeline

```python
from mlopt_orchestrator import Optimizer
from mlopt_torch import TorchBackend

opt = Optimizer(backend=TorchBackend())
opt_mod, stats = opt.optimize(model, x)
```

`optimize` runs **search** (saturate + extract under a cost model) then
**lower** (emit a module through the backend's sink, numerically verified
against the source). `stats` records `rule_fires`, `lowering`, `runner`,
`engine` — which engine saturated — and the strategy's own keys.

## 2. Choose how the result is delivered

```python
from mlopt_cuda import CudaGraphRunner
from mlopt_orchestrator import Autotuned, ChainedRunner, Compositional
from mlopt_torch import TorchCompileRunner
from mlopt_torch.autotune import TORCH_BUILDERS

opt_mod, stats = opt.optimize(model, x, runner=TorchCompileRunner())
# torch.compile wraps the delivered module (stats["compiled"])

opt_mod, stats = opt.optimize(model, x, runner=CudaGraphRunner())
# captures the executor into a CUDA graph — compile-free

opt_mod, stats = opt.optimize(
    model, x, runner=ChainedRunner([TorchCompileRunner(), CudaGraphRunner()]))
# runners compose left-to-right; the Runner protocol is duck-typed
```

Two strategies cover the common cases beyond the default `Monolithic`:

- `strategy=Autotuned(builders=TORCH_BUILDERS)` — re-lowers the same
  extracted term through each candidate executor, verifies each, times
  them on the real input, returns the measured winner.
- `strategy=Compositional()` — multi-block models: optimizes each block
  against its captured real input, recomposes with per-block +
  end-to-end verification and automatic fallback.

## 3. Steer what "cheapest" means

```python
from mlopt_orchestrator import LatencyCriterion, MemoryCriterion

opt_mod, stats = opt.optimize(
    model, x,
    criteria=LatencyCriterion() * 0.7 + MemoryCriterion("peak") * 0.3)
# or the shorthand: criteria={"latency": 1.0, "memory": 0.5}
```

## 4. Reuse one search across runners

```python
res = opt.search(model, x)            # SearchResult — the extracted term
mod = opt.lower(res, x, verify=True)  # re-lower without re-saturating
```

## 5. Ship the artifact — no mlopt at inference

```python
from mlopt_torch.export import export_optimized

export_optimized(model, opt_mod, "out/model", fmt="module")
# .pt2 ExportedProgram — roundtrips standalone; also "safetensors",
# "state_dict", "torchscript"
```

## 6. Reach for the optional knobs only when you need them

| knob | what it does |
|---|---|
| `rules=` | a `RuleSet`, a preset name (`"default"`, `laws.FULL`, `laws.DEFAULT + laws.SYMMETRY`), or a plain iterable — `None` is `DEFAULT_RULES` |
| `engine=` | an `Engine`-port value — the pure-Python engine is the reference default; the Rust engine is an explicit opt-in, never auto-detected |
| `policy=` | reorder the saturation schedule — every rule still runs, so the fixed point and the certificate are unchanged |
| `detect_factors=` / `detect_specials=` / `detect_headshare=` | certified weight elision passes (low-rank factoring, dead/diag/dup elision, shared-head elision) |
| `error_budget=` | certified bounded approximations — a bound ledger plus output-propagated verify |

Every delivered program ships a **certificate** — replay it standalone
with `mlopt_core.egraph.verify_certificate`. The certificate, not the
timing, is the claim.

## See also

- [The frozen API reference](../reference/frozen-api) — the contract surface these calls sit on.
- [`docs/api.md`](https://github.com/handl-lang/graphopt/blob/main/docs/api.md) — the upstream working walkthrough.
- [The three costs](../explanation/three-costs) — how the engine prices a form.

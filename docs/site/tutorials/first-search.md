# Your first `search()`

<!-- provenance: graphopt:docs/api.md@eea02f6 -->

A `torch.nn.Module` and a sample input go in; a verified-lowered module
and run stats come out. The e-graph in between explores every
equivalent spelling of your model's graph.

## The shortest path

```python
from mlopt_orchestrator import Optimizer
from mlopt_torch import TorchBackend

opt, stats = Optimizer(backend=TorchBackend()).optimize(model, x)
```

`optimize(model, x)` runs the full pipeline — **search** (saturate +
extract under a cost model) then **lower** (emit a module through the
backend's sink, numerically verified against the source). `stats`
carries `rule_fires`, `lowering`, `runner`, and the strategy's own
keys.

## The two phase verbs

```python
res = opt.search(model, x)                 # SearchResult — the extracted term
mod = opt.lower(res, x, verify=True)       # LowerResult — verified module
```

Keeping them separate means you can re-lower one search result under
different runners without re-saturating.

## Steer what "cheapest" means

```python
from mlopt_orchestrator import LatencyCriterion, MemoryCriterion

opt_mod, stats = Optimizer(backend=TorchBackend()).optimize(
    model, x,
    criteria=LatencyCriterion() * 0.7 + MemoryCriterion("peak") * 0.3,
)
# or the shorthand: criteria={"latency": 1.0, "memory": 0.5}
```

## Choose how the result is delivered

```python
from mlopt_cuda import CudaGraphRunner
from mlopt_torch import TorchCompileRunner

opt_mod, stats = opt.optimize(model, x, runner=TorchCompileRunner())
opt_mod, stats = opt.optimize(model, x, runner=CudaGraphRunner())
```

Every delivered program ships a **certificate** — a replayable
derivation (`mlopt_core.egraph.verify_certificate` replays it
standalone). The certificate, not the timing, is the claim.

## Next

→ [Watch a law get minted](./mint-a-law) — or read the full
walkthrough in [`docs/api.md`](https://github.com/handl-lang/graphopt/blob/main/docs/api.md)
and the contract in [the frozen API reference](../reference/frozen-api).

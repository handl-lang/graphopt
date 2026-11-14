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
from mlopt_torch.export import export_optimized, load_optimized

manifest = export_optimized(model, opt_mod, "out/model", fmt="module")
mod = load_optimized("out/model.pt2")   # runnable nn.Module, torch only
```

**`.pt2` is *the* runnable artifact.** `fmt="module"` (the default)
traces the optimized computation with `torch.export` and writes an
`ExportedProgram` — fused kernels and all — that `torch.export.load`
rebuilds with nothing but torch installed. Pass `example_input` (and
`dynamic_shapes=` / `example_kwargs=` when the shapes or signature
demand it); `verify=True` round-trips the artifact before returning.

Every export also writes `<artifact>.manifest.json` carrying the
contract fields: `artifact_schema` (`"mlopt-artifact/1"`),
`mlopt_version`, `artifact_kind` (`"runnable"` vs `"weights"`),
`format`, `torch_version`, the grafted-`mlopt_modules` map, and the
GGUF/weights caveats in `notes`.

The other formats are **weights-only, by design** — parameters with
the *optimized* (fused/derived) `state_dict` names, never the
computation:

| `fmt` | artifact | `load_optimized` returns |
|---|---|---|
| `"module"`/`"pt2"` | `.pt2` ExportedProgram | runnable `nn.Module` |
| `"torchscript"` | `.pt`/`.ts` | runnable `nn.Module` (narrower than .pt2) |
| `"safetensors"` | `.safetensors` + manifest | `dict[str, Tensor]` — needs a matching optimized-layout module to run |
| `"state_dict"` | `.pth` | `dict[str, Tensor]` — same caveat |

GGUF/Ollama/LM Studio are **not** export targets: llama.cpp
reimplements standard architectures from HF-layout weights, and the
optimized fused names don't map — convert the *original* model with
`convert_hf_to_gguf.py` instead (the manifest's `notes` say the same).

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

- [What you can optimize](../explanation/what-you-can-optimize) — the
  pipeline's measured scope: which model classes pay and where the
  honest boundaries are.
- [The frozen API reference](../reference/frozen-api) — the contract surface these calls sit on.
- [`docs/api.md`](https://github.com/handl-lang/graphopt/blob/main/docs/api.md) — the upstream working walkthrough.
- [The three costs](../explanation/three-costs) — how the engine prices a form.

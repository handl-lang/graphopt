# mlopt — landing page

**Catopt is a sound mathematical discovery and optimization engine,
where search is a game that can be learned.**

It does not just apply the transformations someone already wrote down.
It **finds new ones, certifies them, and ships them** — then uses learned
policies to search the space of equivalent programs faster than a
hand-tuned heuristic.

---

## The one-sentence version

> Catopt discovers better solutions by combining mathematical
> reasoning, verified transformations, and learned search.  It treats
> optimization as a game over a structured space of equivalent
> solutions, using reinforcement learning to navigate that space and
> formal foundations to preserve correctness.  Applications and
> backends are replaceable interfaces to this general engine.

---

## Who this is for

- **Compiler and systems researchers** who want a platform where new
  transformations are *discovered and certified*, not hand-coded.
- **Performance engineers** with models whose redundancy is spelled
  out in the graph (chains of linear maps, attention written
  elementwise, scans) — that is where the structural wins live.
- **Anyone building on the ideas**: the engine is torch-free, the
  laws are data, and a new backend is an adapter, not a rewrite.

## Who this is *not* for

If your model is a stack of fused library kernels (`nn.Conv2d`,
pooling, `scaled_dot_product_attention`), there is nothing visible for
the engine to rewrite — expect parity, and use `torch.compile` for
codegen.  We measure this *before* you spend time on it.

---

## Three pillars

### 1 · Mathematical discovery

The engine grows its own library.  A census of real exported models
finds op-and-shape patterns; candidates are proposed, verified,
measured, ranked and emitted.  It has minted:

- **laws** — including `mul_unsq_nat`, the first law minted from a
  model's own op-tuple, now shipping in the default rule set
  (−12.27% on its source workload);
- **handler bodies** for non-elementwise patterns (it re-minted a
  carrier element that had been a hand-written executor, verified to
  8.9e-16);
- **kernels** — mid-search, a new operator can be bound to generated
  `triton.jit` source and delivered (`chainfuse`: **2.94× vs
  `torch.compile`**).

### 2 · Categorical soundness

Programs are morphisms of a category; laws are its equations; a
rewrite is a 2-cell with a replayable certificate.  The formal layer
fixes **what is admissible** — which is precisely what lets the
learner explore aggressively.

Soundness is enforced **per artifact**, not once:

| artifact | its verification boundary |
|---|---|
| an individual transformation | the certificate replays on real terms |
| a discovered law | numeric oracle → typed pay gate → certificate replay → **adversarial sweep** (it falsified a naive guard with 212 bad accepts) |
| the search procedure | a policy may only **reorder** — the fixed point and certificate are unchanged |
| a delivered program | the lowered module is verified numerically against the source |
| an approximation | a **certified error budget**, never an unstated one |

### 3 · Learned search

Optimization is a game over the space of *valid* moves: the state is
the current program's features, the action is a legal rewrite, the
reward is the cost improvement.  The learned prior ships trained —
ranking AP **0.133 vs 0.045** for the heuristic on held-out pages, and
it finds ~2× more useful laws per episode.

---

## Proof points

| what | number |
|---|---|
| held-out zoo (18 architectures) | **11 delivered modules beat `torch.compile`**, best 2.18× |
| deep weight chains, k=16 | **14.92× vs `torch.compile`**, exact |
| spelled RetNet export, T=2048 | **6.2–7.4× vs `torch.compile`**, **7.2× less peak activation memory**, verified 9.8e-9 |
| a law nobody wrote, now shipping | `mul_unsq_nat`, **−12.27%** on its source workload |
| discovery yield on the real corpus | **~1.5%** of the exposed class — measured, not claimed |
| a second backend | ONNX import **and** export, fp64 round-trip **1.11e-16**, **zero core changes** |

**And the losses**, because they are the methodology:

- BiRefNet's Swin block: **0.80× vs compiled** — its attention is
  already a fused kernel; there is nothing to fold.
- The bounded/precision lane on real BiRefNet: **certified negative**
  — 56 members offered, 0 delivered, because the member priced 1.32×
  *worse* than dense.  The machinery refused rather than ship a loss.
- The derived carrier **loses below T≈1024** and wins above.  The
  crossover is in the table, not hidden.

---

## Try it

```bash
git clone https://github.com/luisfuturist/mlopt && cd mlopt
uv sync
python examples/quickstart.py                    # ~15s, no GPU needed
cd docs/tutorials/course && vp install && vp -C apps/learn dev  # the 28-level course
```

```python
from mlopt_orchestrator import Optimizer
from mlopt_torch import TorchBackend

opt, stats = Optimizer(backend=TorchBackend()).optimize(model, x)
out = opt(x)      # verified-equal to model(x)
```

## Where to go next

| | |
|---|---|
| **[`README.md`](../README.md)** | the professional overview — the three questions, quickstart, docs index |
| **[`docs/tutorials/course/`](tutorials/course/README.md)** | the interactive course: 28 gated levels, live widgets, mastery |
| **[`docs/paper_draft.md`](paper_draft.md)** | the paper draft — thesis, mechanism, novelty/soundness/value with evidence |
| **[`project/research_lab_notebook.md`](../project/research_lab_notebook.md)** | the lab notebook — questions, dead ends, process lessons |
| **[`docs/results.md`](results.md)** | every measurement, generated from pinned baselines |
| **[`project/adrs/`](../project/adrs/)** | the decision records, each with its falsification path |

## The honest pitch

Most optimizers are libraries of known tricks applied in better
orders.  Catopt is an attempt to build the other thing: a machine that
**proposes mathematics, certifies it, and then learns how to search with
it** — with every claim, including the failures, measured and kept.

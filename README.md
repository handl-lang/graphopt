# catopt

[![ci](https://github.com/luisfuturist/catopt/actions/workflows/ci.yml/badge.svg)](https://github.com/luisfuturist/catopt/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)

**A sound mathematical discovery and optimization engine, where search
is a game that can be learned.**

Catopt discovers better solutions by combining mathematical reasoning,
verified transformations, and learned search.  It treats optimization
as a game over a structured space of equivalent solutions — using
reinforcement learning to navigate that space and formal foundations
to preserve correctness.  Applications and backends are replaceable
interfaces to this general engine.

```python
from catopt_orchestrator import Optimizer
from catopt_torch import TorchBackend      # one adapter; not the core

opt, stats = Optimizer(backend=TorchBackend()).optimize(model, x)
out = opt(x)          # verified-equal to model(x); ships a replayable certificate
```

## 1. What problem does it solve better?

Existing compilers choose a better implementation **from the
transformations a human already wrote** — Inductor fuses pointwise
ops, TVM autotunes schedules, e-graph superoptimizers (TENSAT) rewrite
with a fixed rule set.  Three things they do not do:

| | existing tools | catopt |
|---|---|---|
| **Discovery** | fixed rule set | **mints new laws, atoms and kernels from measured need** — the first corpus-minted law now ships in the default rule set |
| **Soundness** | trust the pass | **every move carries a replayable certificate**; candidates clear an oracle, a typed gate, an adversarial sweep and a measured-pay gate before shipping |
| **Search** | heuristic ordering | **learned policies** — the search is an RL environment; a policy may only *reorder* moves, never change what is legal |

The wins are therefore **structural, not codegen**: folding a chain of
GEMMs into one, deriving a quadratic→linear recurrence, lifting a
scan.  Those are transformations op-level compilers cannot express.
Where the structure is already fused (a `scaled_dot_product_attention`
call, a dense conv), catopt is honest about the ceiling: parity, and
the delivered graph can be handed to `torch.compile` for codegen.

**The scope is honest:** this is not a universal speedup.  Wins are
regime-bounded, and the project keeps its measured losses in the same
table as its wins.

## 2. What can I run on my own model today?

```bash
uv sync
python demo.py                                  # ~60s end-to-end, no GPU needed
python -m bench run <suite> --device cpu        # the benchmark harness
cd toy && vp install && vp -C apps/learn dev    # the 28-level interactive course
```

**On your model** — export, optimize, verify, and optionally deliver:

```python
from catopt_orchestrator import Optimizer, Compositional
from catopt_torch import TorchBackend

opt, stats = Optimizer(backend=TorchBackend()).optimize(model, x)
assert stats["lowering"]        # which executor the winner was lowered by

# multi-block models, per-block + end-to-end verification with rollback
opt, stats = Optimizer(backend=TorchBackend()).optimize(
    model, x, strategy=Compositional())

# export a standalone artifact (.pt2 runs with no catopt at inference)
from catopt_torch import export_optimized
export_optimized(model, opt, "out.pt2", fmt="torchscript")
```

**Discovery** (the engine pointed at its own library):

```bash
python -m catopt_discovery.pipeline            # census → propose → verify → measure → rank → emit
python -m catopt_discovery.play --domain torch --deliver   # optimize a real module
python -m catopt_discovery.play --domain gen --probe       # mint a kernel mid-search, deliver it
python -m catopt_discovery.play --zoo                      # held-out models
python -m catopt_discovery.evidence --report <db>          # what was measured, and did it change?
```

**When it will not help you:** if your model's math is already inside
fused kernels (`nn.Conv*`, pooling, `sdpa`), the e-graph cannot see
inside it and you should expect parity, not a win.  `tools/exposure_screen.py`
measures this *before* you spend search time — the distribution is
bimodal (81 of 321 exported workloads expose literally nothing).

**Backends are adapters.**  The core (`catopt-core`) is torch-free and
dependency-light; `import-linter` enforces that.  Two backends exist:
PyTorch (`catopt-torch`) and ONNX (`catopt-onnx`, import **and**
export, verified through `onnxruntime`).  The ONNX backend landed with
**zero changes under `catopt-core`** — the boundary is measured, not
asserted.

## 3. What evidence shows it works beyond curated examples?

| claim | measurement | artifact |
|---|---|---|
| **Novelty** — mints laws nobody wrote | `mul_unsq_nat`, minted from a model's own op-tuple, in the default rule set (−12.27% on its source workload); 11 machine-admitted laws ship | [`project/retros/mul-unsq-nat.md`](project/retros/mul-unsq-nat.md) |
| **Novelty, at scale** | 321 exported workloads screened; **~1.5%** of the exposed class carries fresh paying unpublished structure — the honest yield | [`project/retros/discovery-scale.md`](project/retros/discovery-scale.md) |
| **Soundness** — catches its own false candidates | the naive guard was falsified by the adversarial sweep (**212 bad accepts**); the exact guard was synthesized instead; the apparent sibling was **rejected** at admission (12 matching models, semantically unequal sites) | [`project/retros/mul-unsq-nat.md`](project/retros/mul-unsq-nat.md) |
| **Soundness, negative** | deliberately-false twins all rejected; the verifier has repeatedly retracted would-be claims | [`docs/mechanism.md`](docs/mechanism.md) |
| **Value, held out** | **11 of 18** delivered modules beat `torch.compile` on the held-out zoo (best 2.18×); deep weight chains **8.22× / 14.92×** | [`docs/results.md`](docs/results.md) |
| **Value, asymptotic** | spelled RetNet export T=2048: **5.0–5.9× vs eager**, **6.2–7.4× vs `torch.compile`**, 7.2× less peak activation memory — verified 9.8e-9 | [`docs/results.md`](docs/results.md) |
| **Honest losses** | BiRefNet Swin block **0.80× vs compiled** (fused ops expose nothing); the precision lane on real BiRefNet was a **certified negative** (56 offers, 0 delivered); the carrier *loses* below T≈1024 | [`docs/results.md`](docs/results.md) |

Every benchmark report lives with the harness and is regenerated from
pinned baselines — **not** in this README:

- [`bench/`](bench/README.md) — the harness (`run` / `compare` / `gate`), pinned baselines, the ledger
- [`docs/results.md`](docs/results.md) — the measured results, generated from those baselines
- [`docs/results_tooling.md`](docs/results_tooling.md) — tooling-level measurements, each with its retro

## The three pillars

1. **Mathematical discovery.**  `catopt_discovery` grows the library:
   census → propose → verify → measure → rank → emit.  It mints laws
   (guard synthesized, adversarially swept), handler bodies for
   non-elementwise patterns, and — mid-search — Triton kernels bound to
   new operators (`claim`).  ADR 0007's four-layer minting stack is the
   design; the corpus-minted law is the proof.
2. **Categorical soundness.**  Programs are morphisms of a category;
   laws are its equations; a rewrite is a 2-cell.  The formal layer
   fixes what is *admissible* — which is what lets the learner explore
   aggressively.  Verification boundaries are **per artifact**: a law
   (oracle + typed gate + certificate replay + adversarial sweep), the
   search procedure (a policy may only reorder — the fixed point and
   certificate are unchanged), a delivery (the lowered module is
   verified against the source), an approximation (a certified error
   budget).  See [ADR 0002](project/adrs/0002-categorical-re-expression-thesis.md),
   [ADR 0003](project/adrs/0003-evaluation-is-an-independent-dimension.md).
3. **Learned search.**  Optimization is a game over the space of
   *valid* moves.  Boards, players and priors live in
   `catopt_discovery`; the learned prior ships trained (ranking AP
   0.133 vs 0.045 heuristic on held-out pages) and finds ~2× more
   useful laws per episode.  See [ADR 0005](project/adrs/0005-the-open-game.md).

## How it works

```text
model → export → e-graph saturation → extract (cost) → certify → lower
                      ▲                                    ↓
          laws as data (composable rule sets,   verified, runnable module
          every rule serializable)              (+ optional compile / CUDA-graph runner)
```

Four independent dimensions — **semantics / search / evaluation /
execution** — never mixed: a cost model ranks but never decides
equivalence; a policy orders but never changes what is legal.

- [`docs/mechanism.md`](docs/mechanism.md) — the conceptual pipeline
- [`docs/api.md`](docs/api.md) — the API surface
- [`docs/evaluation.md`](docs/evaluation.md) — features, calibration, criteria, policies

## Architecture — adapters, not core

| package | role |
|---|---|
| `catopt-core` | the engine: IR, e-graph, laws as data, cost, ports — **torch-free** |
| `catopt-orchestrator` | the backend-neutral pipeline (takes ports as a `Backend`) |
| `catopt-discovery` | discovery + the game layer (`python -m catopt_discovery.<mod>`) |
| `catopt-carriers` | carrier laws and executors (scan, attention) |
| `catopt-torch` | **adapter** — PyTorch export/lower/verify, `TorchBackend` |
| `catopt-onnx` | **adapter** — `OnnxSource` (import) + `OnnxSink` (export), opset 18 |
| `catopt-cuda` | the CUDA-graph runner |
| `catopt-native` | optional PyO3/Rust search engine |

A new backend implements `Sink`; nothing in `catopt-core` changes.
PyTorch is one application of the engine, not its identity.

## Learn the whole thing

[`toy/`](toy/README.md) is an interactive course — **28 gated levels in
6 modules**, each with one idea, a live widget and a checkpoint.
No term is used before it is defined (a test pins that), every level
ends with *"in the real project"* pointing at the implementing files,
and the last level maps every idea to the project's open work
(idea → blocker → next measurement).  Points, per-concept mastery,
module reviews and badges track comprehension.

## Limits

- **Regime-dependent.**  Wins live where algebraic structure is
  exposed; fused-op models see parity.
- **Inference only** — weight folding destroys per-layer gradients.
- **Compile-time search** — seconds per block; large models need the
  `Compositional` strategy.
- **Dev-box magnitudes** — `tools/calibrate_profile.py` re-targets the
  cost model to your hardware.
- **Not a general speedup.**  The honest losses are kept in
  [`docs/results.md`](docs/results.md) alongside the wins.

## Docs

| doc | contents |
|---|---|
| [`docs/paper_draft.md`](docs/paper_draft.md) | the paper draft — thesis, mechanism, the three questions answered |
| [`project/research_lab_notebook.md`](project/research_lab_notebook.md) | the lab notebook — questions, dead ends, process lessons |
| [`docs/product_landing_page.md`](docs/product_landing_page.md) | the landing page |
| [`toy/`](toy/README.md) | the interactive course (28 levels) |
| [`docs/mechanism.md`](docs/mechanism.md) | syntax → structure → search → certificate |
| [`docs/evaluation.md`](docs/evaluation.md) | features, calibration, criteria, policies |
| [`docs/api.md`](docs/api.md) | the API surface |
| [`docs/results.md`](docs/results.md) | measured results (generated from pinned baselines) |
| [`bench/README.md`](bench/README.md) | the benchmark harness |
| [`project/adrs/`](project/adrs/) | the decision records, with their falsification paths |
| [`AGENTS.md`](AGENTS.md) | repo layout + the full verification gate list |

## Verification

```sh
uv run pytest                              # full suite (serial — no -n auto)
.venv/bin/ty check && .venv/bin/ruff check && .venv/bin/ruff format --check
.venv/bin/vulture && .venv/bin/lint-imports
.venv/bin/bandit -c .bandit.yaml -r packages
.venv/bin/semgrep --config .semgrep.yml packages
```

Coverage is pinned at 100% for the original packages (99% ratchet for
`catopt_discovery`); no new suppressions are accepted.  See
[`AGENTS.md`](AGENTS.md).

## License

MIT.

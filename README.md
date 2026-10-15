# catopt

[![ci](https://github.com/luisfuturist/catopt/actions/workflows/ci.yml/badge.svg)](https://github.com/luisfuturist/catopt/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)

**catopt plays a verified rewrite game over neural-network graphs.** A
model is exported to an e-graph; players take only legal,
certificate-replayable moves — apply a law, saturate, mint a new
operator mid-search — and the winning program is delivered as a
runnable, verified `nn.Module`.

```bash
uv sync
python demo.py   # ~60s: optimize a gated block, replay the certificate,
                 # race vs eager and torch.compile
```

```python
from catopt_orchestrator import Optimizer
from catopt_torch import TorchBackend

opt, stats = Optimizer(backend=TorchBackend()).optimize(model, x)
out = opt(x)   # verified-equal to model(x)
```

## Results

Measured on the dev box (RTX 2050, fp64 — see [limits](#limits)); full
provenance in [`docs/results.md`](docs/results.md).  Each block states
**what was measured on what**.

### Optimized models, end-to-end

| model | speed vs eager | speed vs `torch.compile` | fp64 max_abs |
|---|---|---|---|
| deep weight chains (reassoc to 1 GEMM, k=8 / k=16) | — | **8.22× / 14.92×** | exact |
| spelled `x@A@B@C` block | — | **3.96×** | 1.7e-5 |
| `chainfuse` — laws + a kernel minted mid-search | 3.71× | **2.94×** | 1.7e-5 |
| spelled RetNet export, T=2048 — **derived carrier** | **5.0–5.9×** | **6.2–7.4×** | 9.8e-9 |
| spelled RetNet export, T=512 — fused + minted kernel (not the carrier) | — | **1.82–2.33×** | 1.3e-8 |
| BiRefNet Swin block (SDPA + conv decoder) | 0.90× | **0.80× loss** — fused ops expose nothing to fold | 9.7e-17 |
| llama-style SwiGLU MLP | 1.17× | **2.11×** | 6e-8 |
| held-out model zoo (22 architectures) | — | **11/18 verified wins**, best 2.18× | — |
| joint fwd+bwd step vs `torch.autograd` | **+13%** | — | 9.5e-7 |
| pure pointwise fusion | — | **0.82× loss** — Inductor's home turf | — |

### Savings — memory, compute, storage, evaluations

| quantity | before | after | **saved** | instrument |
|---|---|---|---|---|
| **memory** — RetNet peak activation, T=2048, vs its own spelled input | 169.9 MiB | 30.0 MiB | **5.66×** | `max_memory_allocated` |
| **memory** — same carrier vs paper's own chunkwise impl | 26.9 MiB | 30.0 MiB | **0.90× loses** | torchscale path |
| **memory** — vs FLA tuned kernel (fp32) | 9.96 MiB | 19.5 MiB | **0.51× loses** | `fla.ops.retention` |
| **speed** — fused carrier lowering vs paper impl (T=2048) | 2.12 ms | 2.08 ms | **1.02× — parity+** | `decayed_lower` + graph replay |
| **compute** — K-way chunk, T=512 | 29,386,752 FLOPs | 6,700,800 FLOPs | **4.4×** | `flops_cost` |
| **compute** — shared-state, T=512 K=8 | 8,572,928 FLOPs | 7,643,912 FLOPs | **10.8%** | `flops_cost` |
| **storage** — weight chain, k=16 | 4,194,304 B | 262,144 B | **16.0×** | `param_bytes_cost` |
| **storage** — shared-input MoE | — | — | **196,608 B** | `param_bytes_cost` |
| **evaluations** — prior ranking AP | heuristic 0.045 | **0.133** | **3.0×** | held-out pages |
| **evaluations** — evidence store, 2nd run | 128 s | 6.6 s | **19.4×** | wall time |
| **backend** — ONNX round-trip, both directions | — | **1.11e-16 fp64** | **0 core edits** | `onnxruntime`, opsets 13–21 |

### Learned models

| model | training | metric | value | baseline |
|---|---|---|---|---|
| open-game prior | RL (linear), baked | ranking AP | **0.133** | heuristic 0.045 |
| meta-arena prior | RL (linear), baked | reward / episode | **−0.06** | scripted −0.10, random −0.57 |
| `PairPolicyNet` (contraction) | RL (neural) | equal-rollouts ratio | **0.94–0.96** (n≤30) | teacher 1.0 |
| `PairPolicyNet` distilled | supervised | equal-rollouts ratio | **0.91** (n=40) | teacher 1.0 |
| `RuleValueNet` | supervised | ranking AP | 0.023 | heuristic 0.045 (loses) |

Wins are structural — weight folding, shared-input pairing, scan lifts —
the transforms op-level compilers cannot express. GEMM-bound code with
no structure sees parity; launch-bound decode can lose; the derived
carrier *loses* below T≈1024 and wins above. This is not a universal
speedup.

## What makes it different

- **Laws from algebra, not op lists.** Rewrite rules derive from
  categorical structure (monoid carriers, traces, products), so the
  reachable set is the closure of a law library — larger than any fixed
  pass list.
- **Proof, not spot checks.** Every delivered program ships a replayable
  derivation; `verify_certificate` re-runs each step on real terms.
- **The board invents operators.** Construction moves write new rules
  mid-search; `claim` can bind a minted elementwise pattern to a
  *generated Triton kernel* and deliver it.
- **The machine derives the algorithm.** On a spelled RetNet export the
  e-graph derives the chunked recurrence itself — the cross-block
  carrier is a *member* of the saturated graph, and extraction
  *prefers* it at T=512 (29.4M vs 39.1M FLOPs) — not a hand-written
  template. Honest scope: the derived form coincides with RetNet's
  *published* chunkwise duality (verified vs a faithful reimplementation
  at 9.8e-9) — mechanical reachability + certification of a known
  algorithm, not a new one; hand-tuned baselines still beat the
  delivered artifact on memory (~3× vs FLA Triton).  On **speed** the
  delivered lowering has since caught up to the paper's own chunked
  implementation — 2.08 vs 2.12 ms at T=2048, and *faster* under graph
  replay (0.670 vs 0.822 ms) — so the remaining FLA gap is a
  kernel-level one (fused fp32 Triton vs fp64 torch GEMMs), not a
  structural one.  Constructs the search needs are laws too: the K-way
  chunk split (4.4×) and the shared-state reassociation (10.8–24.5%)
  are both admitted, guarded, fp64-verified data.
- **The prior learns.** The open game's policy ships *trained*
  (regenerable via `tools/train_open_prior.py`); on held-out pages its
  ordering beats both the cold start and the cost-delta heuristic
  (average precision 0.133 vs 0.045), and finds ~2× more useful laws
  per episode.
- **It finds its own laws — and ships one.** The enumerator
  instantiates law shapes over the op grammar; the game synthesizes
  the *guard* a candidate needs (`xy + xz = x(y+z)` iff `x≡y` — a
  conditional law no library had); the synthesizer writes a shipped
  rule's `derive` from measured bindings. The first law minted from a
  *corpus op-tuple* now ships in the default rule set: **`mul_unsq_nat`**
  (`mul(unsqueeze U d, unsqueeze V d) = unsqueeze(mul U V, d)`, found
  in a DeltaNet-style block's own `β·k` write) — the naive guard was
  falsified by the adversarial sweep (212 bad accepts), the exact
  guard was synthesized instead, closure cost +18 enodes, and it pays
  **−12.27%** on its source workload.  Honest scope: a 60-model sweep
  of the exposed corpus found no other paying law — the fresh-paying
  yield is ~1.5%, and an apparent sibling with 12 matching models was
  *correctly rejected* at admission (its sites are semantically
  unequal). Where the machinery is blocked, the refusal names the
  missing vocabulary — measured, not guessed.
- **The vocabulary grows itself.** Beyond laws, the machine mints
  *semantics*: a structured handler grammar proposes and fp64-verifies
  handler bodies (it re-minted the `om_decayed` carrier element at
  8.9e-16 — previously a hand-written executor); a law-synthesis loop
  derives and certifies an atom's law set from measured bindings
  (25 certified rows, deliberate false twins rejected); and the
  composed loop runs need → handler → laws → admission end to end.
  Its honest null: the first composed atom computes a *total* where
  the corpus *emits* a sequence — truth and pay are measured
  separately, and the ADR records the negative.
- **The substrate is being made uniform.** A prototype bridge
  re-expresses the IR as morphisms of a monoidal category with
  explicit sharing (`Δ`) and a bounded trace (`loop`): 321/321 corpus
  workloads round-trip structurally exact with zero ops refusing
  translation, and the RetNet carrier spells as a `loop` term verified
  at 1e-7 — the "carrier atoms become definable, not primitive"
  claim measured (ADR 0008).
- **Four independent dimensions** — semantics / search / evaluation /
  execution. Cost models never change semantics; policies never decide
  equivalence.

## The methodology

Falsifiable directions, measured claims.

1. **State the thesis as an ADR** — a decision record with a
   falsification path, not a hope (`project/adrs/`).
2. **Certify before measuring** — equivalence is a replayable
   certificate, not a benchmark.
3. **Measure to admit** — a transformation only counts if it lands on
   real tensors faster.
4. **Kill what fails** — negative results are kept, not hidden
   (ADR 0001 is closed negative).
5. **The game must find it** — a transform isn't "discovered" until the
   machine reaches it, not when a human writes it.

The claim is *reachability + verification*: categorical semantics expose
moves a tensor-IR can't efficiently state. The methodology is the
result, not the speedup.

## How it works

```text
module → export → e-graph saturation → extract (cost) → certify → lower
                      ▲                                    ↓
          laws as data (82 rules,      verified, runnable nn.Module
          all serializable)            (+ optional compile/CUDA-graph runner)
```

Discovery runs the same machinery on the law library itself:
`catopt_discovery.pipeline` (census → propose → verify → measure →
rank → emit), the **enumerator** (`enumerate` — law shapes
instantiated over the op grammar; `forms` — the shapes themselves),
the **guard/derive synthesizers** (`auto_cond`, `dspec_synth`, and
`vocab`'s need detection, which names the missing vocabulary from
measured refusals), and the **open game** (`open_game` — the
candidate stream as a playable board, scored by verdict).
`catopt_discovery.play` exposes the whole loop as an RL environment
(`--domain meta|joint|search|torch|gen|open`).

```bash
python -m catopt_discovery.play --domain torch --deliver  # optimize a real module
python -m catopt_discovery.play --domain gen --probe      # mint → claim → deliver → time
python -m catopt_discovery.play --domain open             # the enumerator as a board
python -m catopt_discovery.play --zoo                     # the probe over held-out models
```

## Limits

- Inference only — weight folding destroys per-layer gradients.
- Compile-time search — seconds per block; very large models need the
  `Compositional` strategy.
- Regime-dependent — wins live where algebraic structure exists.
- Dev-box magnitudes — `tools/calibrate_profile.py` re-targets the cost
  model to your hardware.

## Packages

| package | role |
|---|---|
| `catopt-core` | torch-free engine: IR, e-graph, laws, cost, ports |
| `catopt-torch` | PyTorch export/lower/verify, `TorchBackend` |
| `catopt-carriers` | carrier laws and executors (scan, attention) |
| `catopt-cuda` | CUDA-graph runner |
| `catopt-orchestrator` | backend-neutral pipeline |
| `catopt-discovery` | law discovery + the game layer (`python -m catopt_discovery.<mod>`) |
| `catopt-onnx` | ONNX backend — `OnnxSource` (import) + `OnnxSink` (export), opset 18 |
| `catopt-native` | optional PyO3/Rust search engine |

The hexagonal boundary is enforced by import-linter: core imports no
`torch`, `numpy`, or GPU library — a new backend implements `Sink` and
nothing in core changes. ONNX exercised exactly that: `OnnxSource` +
`OnnxSink` (import **and** export, opset 18, verified through
onnxruntime to 1.1e-16 fp64) landed with **zero** edits under
`catopt-core`.

## Learn the whole thing

`toy/` is an interactive course on this architecture — **28 levels in
6 modules**, each with one idea, a live widget, and a checkpoint you
must pass to advance. It is not a summary: the ladder builds
progressively (no term is used before it is defined — a test pins
that), every level ends with *"in the real project"* pointing at the
files that implement it, and it finishes with **where to improve the
project** (idea → blocker → next measurement). Points, per-concept
mastery, module reviews (spaced retrieval), and badges track
comprehension; the completion panel reports what you can now attack.

```bash
cd toy && vp install && vp -C apps/learn dev   # http://localhost:5173
```

Modules: *Foundations* → *Correctness & choice* → *Evaluation &
execution* → *Discovery & minting* → *Substrate & backends* →
*Engineering & payoff* (the gates, the honest negatives, and the
payoff). See [`toy/README.md`](toy/README.md).

## Docs

| doc | contents |
|---|---|
| [`toy/`](toy/README.md) | **the interactive course** — 28 gated levels, live widgets, mastery |
| [`docs/mechanism.md`](docs/mechanism.md) | the conceptual pipeline — syntax → structure → search → certificate |
| [`docs/evaluation.md`](docs/evaluation.md) | features, calibration, criteria, policies |
| [`docs/api.md`](docs/api.md) | the API surface — `Optimizer`, strategies, runners, criteria |
| [`docs/results.md`](docs/results.md) | measured results, generated from pinned baselines |
| [`bench/README.md`](bench/README.md) | the benchmark harness |
| [`AGENTS.md`](AGENTS.md) | repo layout and the full verification gate list |

## Verification

```sh
uv run pytest                              # full suite (serial — no -n auto)
.venv/bin/ty check && .venv/bin/ruff check && .venv/bin/ruff format --check
.venv/bin/vulture && .venv/bin/lint-imports
.venv/bin/bandit -c .bandit.yaml -r packages
.venv/bin/semgrep --config .semgrep.yml packages
```

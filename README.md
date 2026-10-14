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

| model | vs `torch.compile` |
|---|---|
| deep weight chains (reassoc to 1 GEMM, k=8/k=16) | **8.93× / 16.12×** |
| spelled `x@A@B@C` block | **3.96×** |
| `chainfuse` — laws + a kernel minted mid-search | **2.94×** |
| spelled RetNet export, T=2048 (derived carrier) | **6.2–7.4×** |
| spelled RetNet export, T=512 (derived carrier) | **1.82–2.33×** |
| llama-style SwiGLU MLP | **2.11×** |
| held-out model zoo (22 architectures) | **11/18 verified wins**, best 2.18× |
| joint fwd+bwd step vs `torch.autograd` | **+13%** |
| pure pointwise fusion | **0.82× loss** — Inductor's home turf |

### Savings — structural and memory

| | saved |
|---|---|
| RetNet export, T=2048 — peak activation memory | **7.30×** (169.66 → 23.26 MiB, verified) |
| RetNet export — K-way chunk construct (FLOPs) | **4.4×** (29.4M → 6.70M) |
| RetNet export — shared-state reassociation (FLOPs) | **10.8–24.5%** |
| deep weight chain, k=16 — stored parameters | **16×** (256 KB vs 4.0 MB) |
| shared-input MoE — stored parameters | **192 KB** eliminated |

### Learned models

| model | training | result |
|---|---|---|
| open-game prior | RL (linear), baked | **AP 0.133** vs cost-delta heuristic 0.045 |
| meta-arena prior | RL (linear), baked | **−0.06** vs scripted −0.10, random −0.57 |
| `PairPolicyNet` (contraction) | RL (neural) | 0.94–0.96 equal-rollouts at n≤30 |
| `PairPolicyNet` distilled | supervised | n=40 **0.91** — beats the teacher (the shipped default) |
| `RuleValueNet` | supervised | AP 0.023 — loses to the heuristic |

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
  template. Constructs the search needs are laws too: the K-way
  chunk split (4.4×) and the shared-state reassociation (10.8–24.5%)
  are both admitted, guarded, fp64-verified data.
- **The prior learns.** The open game's policy ships *trained*
  (regenerable via `tools/train_open_prior.py`); on held-out pages its
  ordering beats both the cold start and the cost-delta heuristic
  (average precision 0.133 vs 0.045), and finds ~2× more useful laws
  per episode.
- **It finds its own laws.** The enumerator instantiates law shapes
  over the op grammar (1,160 candidates at depth ≤2 → five paying
  laws); the game synthesizes the *guard* a candidate needs
  (`xy + xz = x(y+z)` iff `x≡y` — a conditional law no library had);
  and the synthesizer writes a shipped rule's `derive` from measured
  bindings. Where it is blocked, the refusal names the missing
  vocabulary — measured, not guessed.
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
| `catopt-native` | optional PyO3/Rust search engine |

The hexagonal boundary is enforced by import-linter: core imports no
`torch`, `numpy`, or GPU library — a new backend implements `Sink` and
nothing in core changes.

## Docs

| doc | contents |
|---|---|
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

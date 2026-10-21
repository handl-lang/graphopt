# Set up graphopt

<!-- provenance: graphopt:README.md@eea02f6 -->

This step gets a working dev environment for the research stack.
Requires Python 3.13, [`uv`](https://docs.astral.sh/uv/), and (only for
the Rust crates) a Rust toolchain — the Python path needs no Rust.

## 1. Clone and sync

```sh
git clone https://github.com/handl-lang/graphopt
cd graphopt
uv sync                    # creates .venv/ — the dev environment
```

There is no `mlopt` facade package: import the domain packages
directly (`mlopt_core`, `mlopt_torch`, `mlopt_orchestrator`,
`mlopt_discovery`, `gamekit`, …). `.venv/bin/python` is the
interpreter every command in this track uses. `uv sync` pulls the
substrate and game-kernel packages in as pinned git deps on the
`eqsat`/`gamekit` repos — the import names are unchanged.

## 2. Smoke-test it

```sh
.venv/bin/python examples/quickstart.py        # end-to-end pipeline on a toy module (~15 s, CPU)
.venv/bin/python -m bench list                 # catalog the benchmark suites
.venv/bin/python -m pytest -m "not slow" -q    # the dev lane (~6,400 tests)
```

`quickstart.py` exercises the whole pipeline — lift a model to the IR,
saturate, extract, lower, verify — on CPU with no GPU required.

## What you installed

| piece | import | repo | role |
|---|---|---|---|
| the engine | `mlopt_core` | [`eqsat`](https://github.com/handl-lang/eqsat) | the torch-free IR, e-graph, law bank, cost models |
| the pipeline | `mlopt_orchestrator` | `eqsat` | `Optimizer`, `search`, `lower` — backend-neutral |
| the game layer | `gamekit` | [`gamekit`](https://github.com/handl-lang/gamekit) | `Board` protocol, players, scoring, calibration |
| torch adapters | `mlopt_torch` | graphopt | `TorchSource`/`TorchSink`/`TorchBackend`, runners |
| discovery | `mlopt_discovery` | graphopt | census → propose → verify → measure → rank → emit |
| benchmarks | `bench` | graphopt | harnessed suites + the discovery scorecard |

## Next

→ [Your first `search()`](./first-search)

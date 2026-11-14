# Set up graphopt

<!-- provenance: graphopt:README.md@eea02f6 -->

This step gets a working dev environment for the research stack.
Requires Python 3.13, [`uv`](https://docs.astral.sh/uv/), and (only for
the Rust crates) a Rust toolchain — the Python path needs no Rust.

## 1. Clone and sync

```sh
git clone https://github.com/handl-lang/graphopt
cd graphopt
uv sync --group bench      # creates .venv/ — dev env + the bench CLI deps
```

There is no `mlopt` facade package: import the domain packages
directly (`mlopt_core`, `mlopt_torch`, `mlopt_orchestrator`,
`mlopt_discovery`, `gamekit`, …). `.venv/bin/python` is the
interpreter every command in this track uses. `uv sync` pulls the
substrate and game-kernel packages in as pinned git deps on the
`eqsat`/`gamekit` repos — the import names are unchanged.

::: warning git+https pins
The pins fetch over `https://github.com/…` — no credentials needed for
the public repos. For `git+ssh` remotes or private mirrors, rewrite
the scheme once:

```sh
git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"
```

The `--group bench` flag adds the benchmark CLI's deps (`tyro`,
`rich`, …); the `dev` group is always on by default.
:::

::: warning No published wheels — yet
Every distribution is `0.1.0dev` and **nothing is on PyPI or any
package index**. Distribution is **git-URL at a tag**: each repo
ships annotated semver tags (`v0.1.0` on `eqsat`, `gamekit`,
`graphopt`, …), and consumers pin `tag = "vX.Y.Z"` in
`[tool.uv.sources]` or install directly with e.g.
`mlopt-core @ git+https://github.com/handl-lang/eqsat@v0.1.0`
(subdirectory packages need a `[tool.uv.sources]` entry with
`subdirectory = "packages/<pkg>"` — see
[`docs/reports/git_url_distribution.md`](../../reports/git_url_distribution.md)).
For this repo the supported install is exactly what step 1 does:
clone and `uv sync`, which resolves the `eqsat`/`gamekit` substrate
pins as git deps on their `packages/` subdirectories.
`uv build` *does* produce a per-package wheel locally (verified for
`mlopt-torch` — setuptools backend, pure-Python, bundles the
`artifacts/*.pt` policy weights), but a wheel install still needs
`mlopt-core`/`mlopt-orchestrator`/`gamekit` from their own repos —
there is no self-contained install yet. Treat "pip install mlopt"
as unsupported; pin a git tag instead.
:::

## 2. Smoke-test it

```sh
.venv/bin/python examples/quickstart.py        # end-to-end pipeline on a toy module (~15 s, CPU)
.venv/bin/python -m bench list                 # catalog the benchmark suites
.venv/bin/python -m pytest -m "not slow" -q    # the dev lane (~6,800 tests)
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
| benchmarks | `bench` | graphopt | harnessed suites + the discovery scorecard — the harness itself is `benchkit` in [`gamekit`](https://github.com/handl-lang/gamekit) (`bench/benchkit` is the compat alias) |

## Next

→ [Your first `search()`](./first-search)

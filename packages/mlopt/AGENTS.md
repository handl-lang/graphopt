# Agent guide — mlopt

The **product entrypoint** for the mlopt stack: pipe models through
the verified optimizer and get back a faster, verified artifact — or
an honest decline.  This repo is *composition only* — zero
optimization machinery lives here; it lives in the sibling repos:

- `handl-lang/eqsat` — substrate (`mlopt-core`, `mlopt-orchestrator`)
- `handl-lang/kit-torch` — the torch-export bridge (`kit_torch`, split
  from this repo's `packages/mlopt-torch` in P3 Step D-1e; a git dep,
  NOT a workspace member)
- `handl-lang/kit-host-tensor` — the host-tensor backend
  (`mlopt_host_tensor`/`kit_host_tensor`)
- this repo also ships the domain packages as workspace members
  (`packages/mlopt-torch` et al.) — the historical graphopt splits
  below remain accurate provenance:
- `handl-lang/kit-discovery` — the discovery engine (`kit_discovery`,
  split from graphopt in P3 Step D-1d; carries the `.hnl` codec this
  repo's `hnl`/`arch_rust`/`onnx_rust` adapters consume)
- `handl-lang/kit-carriers` — the semantic carriers (`kit_carriers`,
  split from graphopt in P3 Step D-1b)
- `handl-lang/kit-cuda` — the CUDA-graph runner (`kit_cuda`, split
  from graphopt in P3 Step D-1c)
- `handl-lang/kit-onnx` — the ONNX backend (`kit_onnx`, split from
  graphopt in P3 Step D-1; not a direct dep of this repo)
- `handl-lang/kernelopt` — codegen domain
- `handl-lang/gamekit` / `discoverykit` — game + mining layers

## Layout

- `packages/mlopt/src/mlopt/` — the Python product package
  (this member of the merged mlopt repo; paths below are repo-relative):
  - `cli.py` — `mlopt models|optimize|export|doctor|stage` (also
    `python -m mlopt`); `stage.py` is the kit-mlopt JSON protocol
    endpoint (`python -m mlopt stage lift|search|verify|deliver` —
    one JSON value on stdin, one record on stdout; real
    lift/search/verify/deliver stages over the existing pipeline,
    leaf tensors re-materialised per stage since they can't cross
    JSON; `deliver` re-runs the `lower(verify=True)` gate and emits
    the verified term as a self-contained `.hnl` artifact at the
    record's `out` path)
  - `optimize.py` — lift → `Optimizer.search` → `lower(verify=True)`
    → `OptimizeReport` (verify gate reads `VerifyReport.passed` — a
    failed gate reports `verified=False` and exits nonzero)
  - `deliver.py` — the heuristic delivery pass (`select_runner`:
    region_fuse/cuda_graph picks + `_bench` timing)
  - `deliver_game.py` — delivery-as-game: `DeliveryBoard` on the
    gamekit `Board` protocol; moves are runner configs
    (`fuse`/`compile`/`graph`/`graph-alias` + chains), the referee is
    measured min/median ms under a mandatory verify gate, `stop`
    commits the fastest verified config (eager always in the pool).
    `gamekit.scoring`'s absolute `ScoreGate` bars don't fit a
    relative argmin — documented in the module docstring
  - `play.py` — the plugin seam (kernelopt convention): `register()`
    seats `random`/`playbook`/`linear*` on the `deliver` board via
    `gamekit.players.registry.register_player`; import mutates
    nothing, `deliver_game` calls it at run time.  `linear*` are
    cold — no delivery prior is trained yet (`DELIVERY_SPEC`,
    `weights=None`)
  - `deliver_train.py` — the delivery-prior trainer: `DeliveryCase`
    replays + `train`/`cold_train` imitation+REINFORCE over the
    `deliver` board's measured argmin oracle
  - `weights.py` — the baked delivery-player weight table
    (`DELIVERY_PLAYER_WEIGHTS`); regenerate via
    `scripts/train_delivery_prior.py`
  - `emit.py` — term → `.hnl` over `tensor.*` (199 ops, each call
    shape verified against `hnl-run`; honest `UnsupportedOpError`
    declines; lazy `hnl-run` KitRunner; artifacts carry
    `from "host-tensor" import tensor`)
  - `doctor.py` — lift-gap diagnostics (`probe()` = no downloads)
  - `adapters/` — auto-discovered catalog: each module exposes
    `ADAPTER`/`ADAPTERS`; `catalog.py` triggers `load_all()`.
    `whisper.py` live (all sizes; `arm=decoder` default one decode
    step, `arm=encoder` the whole `WhisperEncoder` forward — pair
    bf16 large/turbo with `--rules bf16_safe`), `kokoro.py` live (one acoustic
    submodule per load — `arm=decoder|decoder-nc|f0n|textenc|textenc-p|full`;
    needs the `kokoro` optional extra + a `hexgrad/kokoro` source
    checkout at `$KOKORO_SRC`/`/tmp/kokoro-src` or `kokoro_src=`;
    misaki/espeak deliberately not deps — the model code is
    stub-imported).  `hnl.py` live (ADR-0172 — `.hnl` files and
    `hnl:`-prefixed refs lift through `kit_discovery.hnl_import`
    into a term; the spec carries a `HnlSource` (`ModelSpec.source`)
    so `Optimizer.search` runs the same pipeline with no torch
    frontend.  `.hnl` artifacts carry shapes not data: leaf values
    come from `--adapter-opt feed=<json>` or seeded synthetic
    (`synthetic=false` declines); shapeless `Var`s need
    `--adapter-opt leaves='{"x": [dims]}'`).  `onnx_rust.py` live
    (ADR-0176 — `.onnx` files and `onnx-rust:` refs lift through the
    `source-onnx` Rust lifter's `lift` example binary, a subprocess
    emitting hnl_codec Term-JSON + a safetensors payload the adapter
    decodes into the same `Source`-port seam; weights are real,
    inputs are seeded synthetic probes overridable via
    `--adapter-opt inputs='{"x": [dims]}'`/`feed=<json>`; the lifter's
    named refusals surface verbatim; binary resolved via
    `lift_bin=`/`SOURCE_ONNX_LIFT`/sibling checkout + `cargo build
    --example lift` fallback).  `arch_rust.py` live
    (ADR-0176 — `arch:` refs and `mlopt.arch/v1` spec paths
    (`*.arch.json`, or any `.json` sniffed for the format tag)
    assemble through the `source-onnx` `arch` example binary: HF
    safetensors → spec → Term with **no ONNX file**; spec names
    resolve under the checkout's `data/architectures/`; weights are
    real — `weights=<path>` or the spec's `checkpoint.{repo,file}`
    via the local HF cache (never a download); inputs are seeded
    synthetic probes like onnx-rust; binary via
    `arch_bin=`/`SOURCE_ONNX_ARCH` + `cargo build --example arch`).
    `optimize`/`export` take `--adapter-opt
    KEY=VALUE` (repeatable) to reach `load()` options like `arm`
- `crates/hnl-ext-mlopt/` — **moved to `kit-prelude/crates/hnl-ext-mlopt`**
  (ADR-0175 Step B): the `hnl` extension (ADR-0149) — a ~60-line cdylib
  shim forwarding the namespaced `mlopt` command to `run_mlopt` in
  `kit-prelude/crates/kit-mlopt`, which spawns `python -m mlopt`
  (`hnl mlopt optimize|export|doctor|models …`).  Discovered by
  `hnl` via `HANDL_EXT_DIR` / `~/.handl/ext/` / kit-prelude's
  `target/debug/`.  Build it there: `cargo build -p hnl-ext-mlopt`
  in the kit-prelude checkout.

## The product contract

- **Verified or declined** — `lower(verify=True)` is the default;
  `--no-verify` exists but the report must say so.
- **Honest verdicts** — the report carries measured numbers and
  decline reasons; never a silent fallback to eager.
- **Adapters are explicit** — `import mlopt.models.catalog`
  populates the registry; imports mutate nothing else.

## Running the CLI

```sh
uv sync --extra hf --extra kokoro   # at the REPO ROOT (workspace)
export MLOPT_PYTHON=$PWD/.venv/bin/python   # or activate .venv
# hnl ext (lives in kit-prelude now):
(cd ../kit-prelude && cargo build -p hnl-ext-mlopt)
HANDL_EXT_DIR=$PWD/../kit-prelude/target/debug MLOPT_PYTHON=$PWD/.venv/bin/python \
    hnl mlopt optimize whisper:openai/whisper-turbo
# or without hnl:
.venv/bin/mlopt models
.venv/bin/python -m mlopt optimize <ref>

# CI gate (bupha): bash ci.sh  (doctor|py|ci)

# Timeouts (float seconds): MLOPT_HNL_RUN_TIMEOUT bounds each hnl-run
# eval (default 600 — `export --verify-run` names it on expiry);
# MLOPT_EQSAT_TIMEOUT bounds the eqsat-run search (default 300 —
# raise it for large models whose search legitimately runs longer).
```

## Verification

```sh
uv sync   # at the repo root — packages/mlopt is a workspace member
.venv/bin/python -m pytest packages/mlopt/tests -q
.venv/bin/ruff check packages/mlopt
.venv/bin/ruff format --check packages/mlopt
.venv/bin/ty check   # root config covers packages/
```

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->

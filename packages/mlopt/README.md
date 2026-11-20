# mlopt

> **P3 Step D-2**: this package now lives at `packages/mlopt` inside
> the merged `handl-lang/research-mlopt` repo (formerly `handl-lang/graphopt`).
> Run all commands from the **repo root** — `mlopt` is a uv workspace
> member and the root `mlopt-meta` dist depends on it, so
> `.venv/bin/mlopt` exists after a plain `uv sync`.

The product entrypoint: **pipe a model through the verified
optimizer, get a faster verified artifact — or an honest decline.**

```
your model ──► lift ──► search ──► lower(verify=True) ──► deliver
                (IR)    (laws)      (independent check)   (runner)
```

`mlopt` is composition only — the engine lives in `eqsat`
(substrate), this repo's domain packages + `kit-torch`/
`kit-host-tensor` backends, and `kernelopt` (codegen).  This
member holds the adapter catalog, the optimize orchestration, the
honest report contract, and the `hnl` CLI extension.

## Quick start

```sh
uv sync --extra hf        # hf extra = transformers for HF adapters
uv sync --extra kokoro    # + loguru/attrs for the Kokoro acoustic model
# kokoro also needs its source tree (the adapter stub-imports
# kokoro.model, skipping the misaki/espeak G2P pipeline):
git clone --depth 1 https://github.com/hexgrad/kokoro /tmp/kokoro-src
# (or set KOKORO_SRC / pass --adapter-opt kokoro_src=<path>)
.venv/bin/mlopt models
.venv/bin/mlopt doctor whisper:openai/whisper-turbo   # probe, no downloads
.venv/bin/mlopt optimize whisper:openai/whisper-tiny  # verified pipeline
.venv/bin/mlopt optimize kokoro:hexgrad/Kokoro-82M \
    --adapter-opt arm=f0n --adapter-opt device=cpu    # deterministic arm
```

Or through the handl CLI (the `hnl-ext-mlopt` extension lives in
kit-prelude — `cargo build -p hnl-ext-mlopt` there):

```sh
(cd ../kit-prelude && cargo build -p hnl-ext-mlopt)
HANDL_EXT_DIR=$PWD/../kit-prelude/target/debug MLOPT_PYTHON=$PWD/.venv/bin/python \
    hnl mlopt optimize whisper:openai/whisper-turbo
# or without hnl:
.venv/bin/mlopt models
.venv/bin/mlopt doctor <ref>     # probe metadata, no downloads
.venv/bin/mlopt optimize <ref>   # verified pipeline + report
.venv/bin/mlopt export <ref>     # verified term → runnable .hnl
.venv/bin/mlopt optimize foo.hnl   # .hnl source → same verified pipeline
```

`.hnl` is a first-class input (ADR-0172): `mlopt optimize foo.hnl`
or `mlopt optimize hnl:foo.hnl` parses the Handl surface syntax via
`kit_discovery.hnl_import.hnl_to_term` and runs the identical
search → verify → deliver → report flow — no torch frontend needed.
Leaf values bind at load: `--adapter-opt feed=<json>` supplies real
tensor data (`{"name": {"shape": [...], "data": [...]}}`), otherwise
every leaf gets a deterministic seeded synthetic value so the verify
gate still compares reference-vs-delivered honestly (`synthetic=false`
declines).  Shapeless `Var` leaves need
`--adapter-opt leaves='{"x": [2, 3]}'`; only the codec's `tensor.*`
import surface is covered — anything else declines loudly.

## Status

Working product skeleton: adapter auto-discovery (`adapters/*.py`
drop-in), `doctor` gap reports, `optimize` verified pipeline,
`export` → `.hnl` artifacts (201 ops over the ~207-op host-tensor
surface, each signature verified against `hnl-run`), and the
`hnl mlopt` extension.  The **whisper** adapter is live (all HF
Whisper sizes) with two lift units — `--adapter-opt arm=decoder`
(default: one static-shape decode step over caller-owned K/V
buffers) or `arm=encoder` (the whole `WhisperEncoder` forward over
real log-mel features; bf16 large/turbo should pair it with
`--rules bf16_safe` — the default ruleset's sdpa folds diverge with
depth under bf16 and the verify gate will decline them); the
**kokoro** adapter is live — it lifts one
acoustic submodule (`--adapter-opt arm=decoder|decoder-nc|f0n|
textenc|textenc-p|full`) of `hexgrad/Kokoro-82M` over harvested
`forward_with_tokens` inputs; weights come from the HF hub, model
code from a `hexgrad/kokoro` source checkout (`$KOKORO_SRC`,
default `/tmp/kokoro-src`) so the misaki/espeak G2P chain is never
imported.  `f0n` is deterministic and exports clean
(`verify=True` passes end-to-end on CPU); plain `textenc` is
deterministic but walls at `torch.export`
(`pack_padded_sequence` data-dependent lengths — honest decline),
while `textenc-p` wraps the pack→lstm→pad sites as
`custom_op` leaves and exports clean (bench: verified, diff 0.0);
the decoder/`full` arms are stochastic
(SineGen draws `rand`/`randn_like` in-graph) so unseeded verify
declines honestly — use `--no-verify` + aligned `manual_seed`
parity, per the `graphopt` bench docs the adapter ports.  The
**`hnl`** adapter (ADR-0172) lifts `.hnl`/`hnl:` refs straight into
the term pipeline — see above.

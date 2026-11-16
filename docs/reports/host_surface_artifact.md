# Host `tensor.*` surface artifact — one table, three readers

Date: 2026-11.  Scope: `packages/kit-host-tensor`,
`packages/mlopt-discovery`, `scripts/dump_host_surface.py`.  Read-only
references: `handl-lang/host-tensor` (`src/index.hnl`,
`tools/aten_port_list.txt`, `tools/aten_schemas.json`,
`src/bin/aten_port_gen.rs`), the sibling `mlopt` repo's `emit.py`
(**not modified** — adoption path below).

## The drift this closes

Four hand-maintained "op → signature" tables had drifted apart:

| table | home | size |
|---|---|---|
| `_SPECS` (term → `.hnl` emit) | `mlopt_discovery.hnl_surface` (was `bench/drivers/kit/term_hnl.py`) | 12 ops |
| `HOST_BINDINGS` (IR op → host positional args) | `kit_host_tensor.bindings` | 137 ops |
| `_OPS` (IR op → emit signature) | sibling `mlopt` repo `emit.py` | 137 ops |
| `spec_to_hnl` (spec → `.hnl` model) | `mlopt_discovery.modelproj` | spec-side |

The authoritative data lives in host-tensor: the handwritten
`op "tensor.X" : <sig>` decls in `src/index.hnl` plus the
`aten_port_gen --emit-spec` JSON (207 generated ops resolved from
`tools/aten_port_list.txt` + `tools/aten_schemas.json`).

## What landed

* **`kit_host_tensor/host_surface.json`** — the checked-in,
  generated artifact: 237 ops (`handwritten` + `generated` kinds),
  each `{op, signature, schema?, min_args, max_args, status, kind,
  reason}`.  Regenerate:
  `.venv/bin/python scripts/dump_host_surface.py`
  (`HOST_TENSOR_DIR` overrides the sibling-checkout default;
  `--check` is the CI-shaped drift gate).
* **`kit_host_tensor/surface.py`** — `host_ops()` (artifact
  reader), `collect_host_surface()` (fresh collect from a
  host-tensor checkout, read-only), `render_surface()` (canonical
  bytes).  `[tool.setuptools.package-data]` ships the JSON.
* **`tests/test_host_surface.py`** — the drift guards:
  * every `HOST_BINDINGS` key is a real `tensor.*` op (137/137);
  * every `hnl_import.covered_ops()` name is a real `tensor.*` op
    (minus `t`, the one derived transpose spelling);
  * when the host-tensor checkout + `aten_port_gen` binary are
    present, the artifact regenerates **byte-identical** — the same
    discipline `aten_port_gen --check` enforces inside host-tensor.

## The reverse bridge (piece 2)

`mlopt_discovery.hnl_import.hnl_to_term` parses the `.hnl`
expression grammar `hnl_surface.emit` produces — plus `do { … }`
programs and handl-native `tensor.*` calls on the covered surface —
back into `mlopt_core.ir` terms.  135 tensor ops covered; anything
else (`chunk`/`split`/`unbind` Array results, `tensor.from_list` in
term position, ops the IR does not know, malformed syntax) declines
by name via `UnsupportedOpError`/`HnlImportError`.  This is what lets
handl-native `.hnl` models enter mlopt search: source → term → e-graph.

## Deferred: `emit.py` adoption (sibling mlopt repo)

`mlopt`'s `emit.py::_OPS` cannot be edited from this repo.  The
adoption path, when that repo picks it up:

1. `emit.py` loads `kit_host_tensor.surface.host_ops()` (or vendors
   `host_surface.json` if the dep direction forbids it) instead of
   hand-writing `_OPS` entries.
2. Its per-op emit logic keys off `signature`/`min_args`/`max_args`
   where the grammar is regular (unary/binary/reduce families),
   keeping hand-written renderers only for the irregular ops
   (`conv2d`'s dropped `dilation` slot, `batch_norm`'s synthesized
   affine arities, `pack`'s list literal).
3. The drift guard to copy: `test_artifact_is_fresh` in
   `tests/test_host_surface.py` — regenerate-and-compare against the
   host-tensor checkout.

Until then the honest state is: **the artifact is the single
op→signature source of truth in graphopt; `HOST_BINDINGS` and
`hnl_import` are drift-guarded against it; `emit.py` remains a
separate 137-op table pending mlopt-side adoption.**

# catopt-kit — the modular handl composition, proven end to end

**Date:** 2026-10-19 · **Surface:** `packages/catopt-kit/` (new Rust crate,
root cargo workspace) · **Gate:** `cargo test -p catopt-kit` — **12/12 green**.

## What landed

`catopt-kit` is catopt's seam to the modular handl stack — a plain Rust
library crate (the `catopt-native` layout precedent: no PyO3, no Python
changes beyond a uv-workspace exclude line). It assembles the smallest
honest runtime for catopt's use:

- **Language** — `handl_kit_data_canonical::language_from_yaml()` (the
  canonical `Language` as data), installed process-wide via
  `plugin-lib-handl::install_default_language` so lazily-built `.hnl`
  module bridge runtimes bind it (ADR-0128 §"the 0.8 seam").
- **Engines** — the three the canonical kit defaults to: `data` lex
  (`DataLexerEngine`), `pratt` parse (`PrattEngine`), `tree-walk` eval
  (`TreeWalkEngine`). No sdkgen/compile/lint/lsp. Engine selection is
  by name through `EngineRegistry`/`RuntimeConfig`;
  `CatoptKit::new_with_registry` is the injection seam for more later.
- **Modules** — `host-tensor`'s `TENSOR_HNL` interface as a
  `HandlModule` over `tensor_op_registry()` (all 18 `tensor.*` ops), in
  a `ModuleRegistry` layered under `handl_runtime::fp`'s
  `field_access`/`fanout` host fns. `field_access` makes
  `tensor.matmul(a,b)` dispatch with no injected globals: unbound
  `tensor` → `"tensor"` → `"tensor.matmul"` → `ModuleLookup` → `OpFunction`.
- **Imports** — a `ModuleLoader` is wired into the eval engine (and into
  the loader's own engine for nested imports), so `import "./f.hnl"`
  works relative to `module_root`, and `CatoptKit::new_with_project`
  reads `.handl.toml` `[dependencies]` via
  `handl_platform::resolve_dependencies`.
- **`run_tagged`** — a second wired engine kept for the
  `TaggedValue`-returning eval path (`eval_tagged`), so a tensor op
  result can be observed as the real `TaggedValue::Tensor` variant with
  no `serde_json::Value` round-trip.

Deps float on `main` as git deps (kernel/lang/platform/host-tensor);
the local-override story is `~/.cache/devin/handl_deps.env` (git
insteadOf → `~/GitHub/handl-lang/*` checkouts) with the handl-lang
`[patch]` convention documented in the root `Cargo.toml`.

## Verified end to end (`tests/compose.rs`)

- `tensor.to_list(tensor.matmul(from_list([[1,2],[3,4]]), from_list([[5,6],[7,8]])))`
  → `[[19,22],[43,50]]` through the assembled runtime — real `.hnl` →
  real tensor work.
- `run` returns the `{"__tensor__":…}` wire form;
  `TaggedValue::from` decodes it to `TaggedValue::Tensor`
  (`shape [2,2]`, `data [19,22,43,50]`); `run_tagged` returns the
  **Tensor variant directly** — never an `Obj` fallback.
- Chained ops inside one program (`add(matmul(a,b), ones)`) — the
  tensor round-trips through the `Function` boundary each hop.
- conv2d (1×1 NCHW) and batch_norm (eval mode, `eps=0`) smokes — both
  match `host-tensor`'s own spec values.
- `tensor.derive("batch_norm")` → `"tensor.batch_norm_backward"`;
  unpaired → `null` — the optimization-loop surface.
- **`.handl.toml` name resolution works**: a temp project with
  `host-tensor = { path = "host-tensor" }` (the embedded `TENSOR_HNL`
  vendored as `index.hnl`) supports both
  `from "host-tensor" import tensor` and `import * as t`.
- A consumer-authored `.hnl` module (`linalg.hnl` with a
  `from "host-tensor" import tensor` header and an
  `export double_matmul` arrow) imports and runs — the shape a
  catopt-side `.hnl` module would take.

## Named gap (documented, not patched around)

A *checked-in* `.handl.toml` in catopt-kit can't name the `host-tensor`
dep without a machine-local `path =`: upstream `resolve_dependencies`
maps names to **filesystem dirs** only — no cargo/git-package → dir
bridge (the ADR-0010 package-manager future). The temp-dir test proves
the seam itself works; the missing piece is a dep *specifier*, not
resolution machinery.

## One environment note

The tree-walk engine recurses deeply enough on ordinary expressions to
overflow libtest's 2 MiB default test-thread stack (`conv2d`'s 4-deep
literals hit it). Both upstream suites carry the same knob — repo-root
`.cargo/config.toml` sets `RUST_MIN_STACK = 32 MiB`, matching
`kit-canonical/.cargo/config.toml`. Not a composition blocker — an
upstream-documented constraint.

## What this is not (yet)

No Python surface (that is `catopt-native`'s maturin shape when needed);
no `bytecode`/typecheck/compile engines registered (the registry seam
admits them by name); no `.hnl` module shipped by catopt itself — the
import machinery is proven on a fixture, `packages/catopt-kit` carries
no `.hnl` yet.

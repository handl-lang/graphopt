# `catopt-kit` — catopt's handl composition seam

The smallest honest assembly of the modular handl-lang stack for
catopt's use — a plain Rust library crate (no PyO3; the `catopt-native`
precedent for *when* a Python surface is wanted comes later). It proves,
end to end, that catopt can compose handl's data-driven language +
engines + `host-tensor` op surface as **dependencies** — no upstream
forks, no vendoring.

## What it assembles

| layer | crate | piece |
|---|---|---|
| language | `handl-kit-data-canonical` | `language_from_yaml()` — the canonical `Language` as data |
| value | `handl-value` | `TaggedValue` incl. `Tensor` (the real tensor variant) |
| runtime | `handl-runtime` | `Runtime` + `EngineRegistry` + `RuntimeConfig` + `ModuleLoader` |
| engines | `plugin-engine-{lex-pattern,parse-pratt,exec-tree-walk}` | `"data"`/`"pratt"`/`"tree-walk"` — the smallest honest set (no sdkgen/compile/lint/lsp) |
| module bridge | `plugin-lib-handl` + `plugin-module` | `HandlModule` + `OpRegistry` + `ModuleRegistry` |
| tensor ops | `handl-host-tensor` | `tensor_op_registry()` — the 18 `tensor.*` ops |

`CatoptKit::new()` installs the canonical language as the process-wide
default (the seam `HandlModule` bridge runtimes bind), builds the
registry, and wires the module chain: the `tensor` `HandlModule` under
the canonical fp host functions. `field_access` is what makes
`tensor.matmul(a, b)` dispatch — `tensor` is unbound → `"tensor"`,
`field_access` concatenates to `"tensor.matmul"`, and `call` routes the
qualified name through `ModuleLookup` → the op's Rust impl.

## The `.handl.toml` seam

`CatoptKit::new_with_project(dir)` reads `dir/.handl.toml`'s
`[dependencies]` table into the `ModuleLoader`, so
`from "host-tensor" import tensor` resolves the name to a package
directory (`index.hnl` inside). **It works** — `tests/compose.rs` proves
both `import tensor` (the namespace record of string callees) and
`import * as t` on a temp-dir project.

**What's missing for a checked-in `.handl.toml` here**: upstream
`resolve_dependencies` maps names to *filesystem* dirs only — there is
no cargo/git package → directory bridge, so a committed manifest could
not name the dep without a machine-local `path =`. The ADR-0010
package-manager story is the upstream future for that; for now the
embedded `TENSOR_HNL` interface is what a fixture vendors.

## Build & test

```sh
source ~/.cache/devin/handl_deps.env   # maps the git deps to local checkouts
cargo test -p catopt-kit
```

**Stack note**: the tree-walk engine is a recursive interpreter that
overflows libtest's default 2 MiB test-thread stack on ordinary
expressions — the repo-root `.cargo/config.toml` sets
`RUST_MIN_STACK = 32 MiB`, matching upstream's own
`kit-canonical/.cargo/config.toml`.

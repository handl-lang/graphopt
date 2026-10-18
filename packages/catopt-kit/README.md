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
| catopt program | `hnl/conv_bn.hnl` | the conv→batch_norm composite + its folded form, `.hnl` over the dep surface |
| subprocess door | `src/bin/hnl-run.rs` | `.hnl` in on stdin/file → `serde_json` value out; `--project` wires `.handl.toml` deps, `--repeat N` evaluates N times in one kit (the per-call cost hook) |

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

**The bridge now lands** (platform `90853ec2`): this crate ships a
committed `.handl.toml` that names the dep *portably* —

```toml
[dependencies]
host-tensor = { git = "https://github.com/handl-lang/host-tensor", branch = "main" }

[patch]
host-tensor = "../../../../GitHub/handl-lang/host-tensor/src"
```

`[dependencies]` accepts `{ git = …, rev|branch|tag }` and resolves an
unpatched git dep to its package-cache slot (`$HANDL_PKG_DIR`, else
`~/.handl/pkg/host-tensor`) — the directory the ADR-0010 fetch hook will
materialize (populate it with a checkout or symlink today).
`[patch]` is the consumer-side dev-link (the Cargo `[patch]` story): a
patch whose named directory exists wins over the declared source, a
missing dir is inert. The package dir is host-tensor's `src/` — where
its `index.hnl` lives. `tests/conv_bn.rs` asserts both arms.

## The shipped consumer — `hnl/conv_bn.hnl`

The crate's own `.hnl` module is the program catopt's e-graph
optimizes — the eval-mode conv+BN weight-space fold
`batch_norm(conv2d(x, W, b), γ, β, μ, σ²) → conv2d(x, W′, b′)`
(`catopt_core.laws.tensor`'s `BATCH_NORM_RULES` + `CONV_FOLD_RULES`):
`conv_bn` is the unfolded composite, `conv_bn_fold` computes
`s = γ·rsqrt(σ²+ε)` per channel (scalar `**` over the materialized stat
lists — the op surface has no elementwise pow) and folds it into
`W′ = W∘ŝ′`, `b′ = b∘s + (β − s∘μ)`. Both evaluate through the runtime
to `TaggedValue::Tensor`, value-equal on dyadic fixtures — the minted
rewrite proven end to end through the consumer. `bn_adjoint`/`bn_grad`
expose `tensor.derive`/`batch_norm_backward`.

## The subprocess door — `hnl-run`

`src/bin/hnl-run.rs` is the Python↔kit seam: a `.hnl` source on stdin
(or a file path) evaluates through `CatoptKit` and the
`serde_json::Value` boundary form prints on stdout
(`{"__tensor__": …}` for a tensor). It is deliberately program-generic
— the emitted-terms path (`lab/kit/term_hnl.py`, which renders catopt
e-graph terms as `.hnl` and shells this binary) and the hand-written
`hnl/` modules go through the same door. `--repeat N` evaluates the
source N times inside one kit so a per-call cost is measurable —
how an optimization *decision* is priced by the actual runtime rather
than the term-level `dag_cost`.

```sh
cargo build -p catopt-kit --bin hnl-run
echo 'tensor.matmul(tensor.from_list([[1,2]]), tensor.from_list([[3],[4]]))' \
  | target/debug/hnl-run -
```

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

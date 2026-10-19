# `mlopt-kit` — mlopt's handl composition seam

The smallest honest assembly of the modular handl-lang stack for
mlopt's use — a plain Rust library crate (no PyO3; the `mlopt-native`
precedent for *when* a Python surface is wanted comes later). It proves,
end to end, that mlopt can compose handl's data-driven language +
engines + `host-tensor` op surface as **dependencies** — no upstream
forks, no vendoring.

## What it assembles

| layer | crate | piece |
|---|---|---|
| language | `handl-kit-data-prelude` | `language_from_yaml()` — the canonical `Language` as data |
| value | `handl-value` | `TaggedValue` incl. `Tensor` (the real tensor variant) |
| runtime | `handl-runtime` | `Runtime` + `EngineRegistry` + `RuntimeConfig` + `ModuleLoader` |
| engines | `hnl-engine-{lex-pattern,parse-pratt,exec-tree-walk}` (+`hnl-engine-exec`) | `"data"`/`"pratt"`/`"tree-walk"` — the smallest honest set (no sdkgen/compile/lint/lsp); one crate per `engine-*` repo post-ADR-0128 |
| module bridge | `plugin-lib-handl` + `plugin-module` | `HandlModule` + `OpRegistry` + `ModuleRegistry` |
| tensor ops | `handl-host-tensor` | `tensor_op_registry()` — the 18 `tensor.*` ops |
| mlopt program | `hnl/conv_bn.hnl` → `CONV_BN_HNL` | the conv→batch_norm composite + its folded form, `.hnl` over the dep surface; exported over the crate API (`mlopt_kit::CONV_BN_HNL`, `hnl_module("conv_bn")`) |
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
host-tensor = { git = "https://github.com/handl-lang/host-tensor", branch = "main", subdir = "src" }

[patch]
host-tensor = "../../../host-tensor/src"
```

`[dependencies]` accepts `{ git = …, rev|branch|tag, subdir }` and
resolves an unpatched git dep to its package-cache slot
(`$HANDL_PKG_DIR`, else `~/.handl/pkg/host-tensor`) — the directory
the ADR-0010 fetch hook will materialize, with `subdir` naming the
package root inside the clone (host-tensor's `.hnl` sources live in
`src/`). Populate the slot with a checkout or symlink today.
`[patch]` is the consumer-side dev-link (the Cargo `[patch]` story): a
patch whose named directory exists wins over the declared source, a
missing dir is inert. The package dir is host-tensor's `src/` — where
its `index.hnl` lives. `tests/conv_bn.rs` asserts both arms.

## The shipped consumer — `hnl/conv_bn.hnl`

The crate's own `.hnl` module is the program mlopt's e-graph
optimizes — the eval-mode conv+BN weight-space fold
`batch_norm(conv2d(x, W, b), γ, β, μ, σ²) → conv2d(x, W′, b′)`
(`mlopt_core.laws.tensor`'s `BATCH_NORM_RULES` + `CONV_FOLD_RULES`):
`conv_bn` is the unfolded composite, `conv_bn_fold` computes
`s = γ·rsqrt(σ²+ε)` per channel (scalar `**` over the materialized stat
lists — the op surface has no elementwise pow) and folds it into
`W′ = W∘ŝ′`, `b′ = b∘s + (β − s∘μ)`. Both evaluate through the runtime
to `TaggedValue::Tensor`, value-equal on dyadic fixtures — the minted
rewrite proven end to end through the consumer. `bn_adjoint`/`bn_grad`
expose `tensor.derive`/`batch_norm_backward`.

The module ships **over the crate API**: `mlopt_kit::CONV_BN_HNL`
embeds the source via `include_str!` (`hnl_module("conv_bn")` resolves
it by name), so a consumer reads the artifact without reaching into
the repo tree — and it is part of `cargo package`. A consumer that
wants the `from "<path>" import …` form materializes the string to a
file; `tests/conv_bn.rs` proves the embedded copy imports and
evaluates identically.

## The e-graph leg — `egraph-run`

`src/egraph.rs` (`mlopt_kit::egraph`) is the kit's second leg: the
`handl-egraph` saturation substrate (the named Rust port of
`mlopt_core`'s `EGraph`/`Rewrite` spec) driven over the
`handl-discovery` codec's JSON wire forms
(`{"mvar"}`/`{"leaf"}`/`{"op"}` — the scheme `ir.term_to_data`
canonized). `src/bin/egraph-run.rs` is its subprocess door — the same
shape as `hnl-run`: a JSON run *spec* (terms, declarative rules, run
options, a per-op cost table, optional `check_terms`) on stdin, a
canonical run *report* on stdout. The report is engine-independent —
classes compare by Weisfeiler-Lehman signature, never by internal id —
because the consumer is `lab/audit/egraph_differential.py` /
`tests/test_egraph_differential.py`: a Python↔Rust *differential*
that runs the same fixtures through `mlopt_core.egraph.EGraph` and
`handl-egraph` and asserts identical structure, provenance, and
extracted winners.

```sh
source ~/.cache/devin/handl_deps.env
cargo build -p mlopt-kit --bin egraph-run
echo '{"terms":[{"leaf":"x"}]}' | target/debug/egraph-run -
```

`tests/gauntlet_stub.rs` exercises `handl-discovery`'s other surface —
`GauntletLegs` — with stub domain legs driven through `run_gauntlet`
(the real gate pipeline, stop-on-failure included).

## The subprocess door — `hnl-run`

`src/bin/hnl-run.rs` is the Python↔kit seam: a `.hnl` source on stdin
(or a file path) evaluates through `CatoptKit` and the
`serde_json::Value` boundary form prints on stdout
(`{"__tensor__": …}` for a tensor). It is deliberately program-generic
— the emitted-terms path (`lab/kit/term_hnl.py`, which renders mlopt
e-graph terms as `.hnl` and shells this binary) and the hand-written
`hnl/` modules go through the same door. `--repeat N` evaluates the
source N times inside one kit so a per-call cost is measurable —
how an optimization *decision* is priced by the actual runtime rather
than the term-level `dag_cost`.

```sh
cargo build -p mlopt-kit --bin hnl-run
echo 'tensor.matmul(tensor.from_list([[1,2]]), tensor.from_list([[3],[4]]))' \
  | target/debug/hnl-run -
```

## Build & test

```sh
source ~/.cache/devin/handl_deps.env   # maps the git deps to local checkouts
cargo test -p mlopt-kit
```

**Stack note**: the tree-walk engine is a recursive interpreter that
overflows libtest's default 2 MiB test-thread stack on ordinary
expressions — the repo-root `.cargo/config.toml` sets
`RUST_MIN_STACK = 32 MiB`, matching upstream's own
`kit-prelude/.cargo/config.toml`.

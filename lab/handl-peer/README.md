# `lab/handl-peer` — spike: catopt calls handl

Plan 0030 step 2, in its cheapest form. The *shipping* shape is a PyO3
extension; the *proof* is here.

## What it is

Two Rust functions over handl's crates, and two ways to reach them:

| surface | what it does |
|---|---|
| `transform_ids(dir)` | load a catopt fragment directory as a handl `Language`; return the transform ids |
| `rewrite(dir, ast_json, at, rule_id)` | apply one named rule at one node; return the rewritten AST as JSON |

* `src/lib.rs` — the library **and** the PyO3 bindings (`language_transform_ids`, `apply_rule`).
* `src/bin/rewrite.rs` — the same library as a CLI, because the PyO3
  packaging is not done (see *Gaps*).

## Why it exists

`plan 0029` (the boundary spike) proved catopt's vocabulary can *sit* in
handl — all 83 laws load, and handl's `RewriteEngine` fires one of them
(handl `82194954`). This crate answers the next question: can **catopt**
make that happen? It can: `tests/test_handl_peer.py` drives the CLI from
Python and asserts both that the rewrite happens and that the result
agrees with catopt's own `EGraph.apply_rule`.

## Build and run

```sh
cd lab/handl-peer
CARGO_TARGET_DIR=~/GitHub/handl-lang/platform/target cargo build --bin rewrite
# then, from the repo root:
uv run pytest tests/test_handl_peer.py -q
```

`CARGO_TARGET_DIR` is deliberate: pointing it at handl's own target
directory reuses the artifacts its workspace already built, which turns a
multi-minute cold build into ~15 s.

## Gaps — read these before treating it as a package

1. **The path dependencies are absolute and local.**  `Cargo.toml` points
   at `~/GitHub/handl-lang/platform/crates/...`.  A real package needs
   handl published, or a git dependency — and handl's crates use
   workspace inheritance, which a *git* dependency handles and a bare
   vendored copy does not.
2. **The PyO3 extension does not import.**  `cargo build --features
   pyo3/extension-module` produces a `.so`, but loading it fails with
   `undefined symbol: Py_TYPE`.  The fix is the toolchain `catopt-native`
   uses — `maturin`, which is not installed in this venv — not a code
   change.  Until then the CLI is the working surface.
3. **No tensors cross.**  By design (ADR 0011): v1 is language-only, so
   the unsafe/single-threaded caveats of `TaggedValue::Resource` do not
   apply.
4. **It is a spike, not a package.**  Promotion to `packages/` follows
   the repo rule: tests clearing the coverage floor, and a dependency
   story that does not assume a sibling checkout.

## What it is *not*

It is not a VM, not a host, and not an integration with handl's exec
engines: it loads a language and runs a rewrite. That is exactly what
plan 0030 gates the rest on.

# `lab/handl-peer` — spike: catopt calls handl

Plan 0030 step 2, in its cheapest form. The *shipping* shape is a PyO3
extension, and that shape now works: Python `import catopt_handl_peer`
and calls into handl directly — no shelling out.

## What it is

Two Rust functions over handl's crates, and two ways to reach them:

| surface | what it does |
|---|---|
| `transform_ids(dir)` | load a catopt fragment directory as a handl `Language`; return the transform ids |
| `rewrite(dir, ast_json, at, rule_id)` | apply one named rule at one node; return the rewritten AST as JSON |

* `src/lib.rs` — the library **and** the PyO3 bindings
  (`language_transform_ids`, `apply_rule`; module `catopt_handl_peer`).
* `src/bin/rewrite.rs` — the same library as a CLI. The extension is the
  working surface; the CLI is the fallback for a checkout with no maturin
  build (and is what the tests fall back to).

## Why it exists

`plan 0029` (the boundary spike) proved catopt's vocabulary can *sit* in
handl — all 83 laws load, and handl's `RewriteEngine` fires one of them
(handl `82194954`). This crate answers the next question: can **catopt**
make that happen? It can: `tests/test_handl_peer.py` drives the peer from
Python and asserts both that the rewrite happens and that the result
agrees with catopt's own `EGraph.apply_rule`.

## Build and run

### The extension (the working surface)

```sh
# one-time, if the uv tool is not present: uv tool install maturin
cd lab/handl-peer
VIRTUAL_ENV=../../.venv maturin develop \
    --target-dir ~/GitHub/handl-lang/platform/target

# then, from the repo root:
.venv/bin/python -c \
  "import catopt_handl_peer; print(catopt_handl_peer.language_transform_ids('lab/handl-spike/fragments')[:2])"
uv run pytest tests/test_handl_peer.py -q
```

maturin is the toolchain `catopt-native` already uses; it is installed on
this host as a uv tool (`~/.local/bin/maturin`), not in `.venv`.

### The CLI (the fallback)

```sh
cd lab/handl-peer
PYO3_PYTHON=$(git rev-parse --show-toplevel)/.venv/bin/python \
    CARGO_TARGET_DIR=~/GitHub/handl-lang/platform/target \
    cargo build --bin rewrite
```

`CARGO_TARGET_DIR` is deliberate: pointing it at handl's own target
directory reuses the artifacts its workspace already built, which turns a
multi-minute cold build into ~15 s.

### The `Py_TYPE` trap (why `PYO3_PYTHON` is load-bearing)

A raw `cargo build --features pyo3/extension-module` used to produce a
`.so` that failed to import with `undefined symbol: Py_TYPE`. The cause is
a **build-time interpreter mismatch, not abi3 and not libpython linkage**:

* pyo3's build config picks the interpreter from `PYO3_PYTHON`, then the
  active virtualenv, then `PATH`. With none set it finds `python3` —
  which on this host is **3.14** (`/usr/sbin/python3`) while the runtime
  is **3.13.9**.
* pyo3-ffi makes `Py_TYPE` an **extern** symbol only under `Py_3_14`
  (`pyo3-ffi/src/object.rs`: `#[cfg(Py_3_14)] extern "C" { pub fn
  Py_TYPE(...) }`; otherwise it is an inline `ob_type` accessor). A 3.14
  build therefore emits a `Py_TYPE` *reference*, and CPython 3.13 does
  not export that symbol — hence the import error.
* Fix: build against the venv interpreter. `maturin develop` does this
  automatically; a raw `cargo build` needs `PYO3_PYTHON=<venv>/bin/python`.

## Gaps — read these before treating it as a package

1. **The path dependencies are absolute and local.**  `Cargo.toml` points
   at `~/GitHub/handl-lang/platform/crates/...`.  A real package needs
   handl published, or a git dependency — and handl's crates use
   workspace inheritance, which a *git* dependency handles and a bare
   vendored copy does not.
2. ~~**The PyO3 extension does not import.**~~  **Closed.**  The
   extension builds and imports (maturin, or `PYO3_PYTHON` for the cargo
   route — see *The `Py_TYPE` trap* above), and
   `tests/test_handl_peer.py` drives it directly.  The CLI remains a
   supported fallback, not the only surface.
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

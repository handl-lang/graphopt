---
title: 'Retro — spike Q3: which way does the integration point?'
type: retro
date: 2026-10-14
---

# Retro — spike Q3: which way does the integration point?

**Plan**: 0029, **Q3**.
**Verdict**: **(A) — catopt embeds handl as an optional peer**, on the
measured facts below.  (B) is not excluded forever; it is recorded with
the condition that would make it viable.

## The question

Two shapes were on the table:

- **(A) catopt embeds handl** — a Rust peer (PyO3 / `maturin`), catopt
  calls handl for the language and its rewrite/typecheck engines.
- **(B) handl drives catopt** — catopt's vocabulary is a handl language,
  tensor ops are host functions registered by a catopt plugin, and
  catopt's discovery/RL become handl tools.

## The measured facts

| fact | source |
|---|---|
| The **load path is a Rust API**, not a CLI path: `language_from_dir` reads the ten-file layout; `hnl` selects kits through *registered emitters* | Q0; `crates/host-cli/src/main.rs` |
| `FragmentSource::DataFile` is modeled but **returns `LangError::NotImplemented`** ("the emitters are the active path") | `crates/handl-kit-data/src/lib.rs` |
| A tensor crosses as **`ForeignResource`** — `{ptr: *mut c_void, finalizer, dropped}`, `Rc`-held, **not `Send`/`Sync`**, never dereferenced by handl | Q1a; `crates/handl-value/src/lib.rs` |
| A Rust peer is **already an established shape in catopt**: `catopt-native` is optional, excluded from the uv workspace, `maturin`-built, opt-in via `engine=` | `AGENTS.md`; `packages/catopt-native` |
| catopt's CI installs a CPU torch wheel and expects no Rust toolchain | `.github/workflows/ci.yml` |

## Why (A)

1. **The working path is a Rust-caller API.**  Q0 proved a `Language`
   assembles from catopt's data — *when a Rust caller invokes
   `language_from_dir`*.  (B) would require handl-side work first
   (implementing runtime data-file kits, or a Python host bridge) before
   catopt's vocabulary could even load from the CLI.
2. **The tensor channel is a raw pointer, which a PyO3 peer produces
   natively.**  (A) hands handl a pointer plus a finalizer that decrefs —
   exactly the `ForeignResource` contract.  (B) would require handl to
   embed CPython to reach torch, inverting ownership and dragging a
   Python runtime into the kernel.
3. **The precedent and the CI story already exist.**  `catopt-native`
   is the same shape: an optional Rust peer that the default path does
   not need.  (A) preserves core's zero-dependency contract and keeps
   catopt's CI light; the peer is opt-in.
4. **The useful integration is the language, not the tensor runtime.**
   ADR 0011 decides that rewriting is canonical in the calculus and
   execution stays catopt's.  So the peer's job is small and
   well-bounded: load a `Language` from catopt's fragments, run
   rewrite/typecheck passes, hand back terms — no tensors need cross at
   all in the first version.

## The seam test (the pass criterion)

Named, and half-built already:

1. **Forward** — catopt's store → fragments → handl `Language`:
   *exists* (`catopt_fragments.rs`, handl `b20b978a`).
2. **Reverse** — a handl term/row → a catopt cell, identity on the
   shipped bank: `cell_from_handl` exists in catopt; the **round-trip
   test is not yet written** and is the seam's acceptance test.
3. **No tensors in v1** — the crossing is deliberately value-free, so
   the unsafe/single-threaded caveats of `ForeignResource` do not apply
   yet.

## What (B) would need to become viable

- handl implements **runtime data-file kits** (`FragmentSource::DataFile`
  stops returning `NotImplemented`), so a language is loadable without a
  Rust caller; and
- a **host bridge** so handl can call tensor primitives that live in
  Python/torch (PyO3 embedding, or an FFI to a tensor library).

If both land, (B) is the more ambitious framing — catopt as an
*application of the platform* rather than a project with a peer — and
this decision should be revisited then.  It is not a rejection, it is a
sequencing.

## Honest caveats

- **(A) still costs a second toolchain**, optionally.  The mitigation is
  the `catopt-native` pattern: excluded from the workspace, built with
  `maturin`, opt-in.  If that pattern ever becomes load-bearing for the
  default path, the cost is real.
- **The peer is young.**  handl's own guide notes a ~8.5 GB debug target
  and memory-hungry test binaries; a catopt-side CI job that builds it
  must respect that (`CARGO_PROFILE_DEV_DEBUG=0`, `-j 2`).
- **Nothing here is validated by a tensor run.**  Q1b (a resource
  crossing through exec) is still unmeasured; this decision does not
  depend on it, but the "host functions" claim in ADR 0010 does.

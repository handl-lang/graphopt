//! `mlopt-experiments` — the runnable-evidence crate: each `src/bin/`
//! target is one experiment the paper cites.
//!
//! The kit the experiments run on is `mlopt-kit`'s [`CatoptKit`],
//! consumed as a declared cargo dep (`git` source floating on `main`,
//! dev-linked to `packages/mlopt-kit` by the `[patch]` row in
//! `Cargo.toml`). This crate deliberately composes the stack *only*
//! through declared deps: the evidence that the modular handl-lang
//! stack assembles from declared components is the `Cargo.toml` dep
//! graph itself — the same edges `mlopt-kit`'s manifest declares,
//! mirrored as data in `manifest/kit.toml` — so no builder is
//! re-spelled here.
//!
//! `CatoptKit` is the post-ADR-0128 assembly: the canonical `Language`
//! as data bound to the `data`/`pratt`/`tree-walk` pipeline, the
//! `tensor.*` host op surface (18 ops) composed under the canonical fp
//! host functions, and the file/name import machinery wired through
//! the project's `.handl.toml` ([`CatoptKit::new_with_project`] →
//! `from "host-tensor" import tensor` resolves by dependency name).
//! It is **not** `Send` — build one per thread.
//!
//! Values cross the engine boundary as `serde_json::Value` (the
//! `{"__tensor__": …}` wire form); inside the tree-walk engine they are
//! `TaggedValue`s — a tensor op result is `TaggedValue::Tensor`, which
//! [`CatoptKit::run_tagged`] exposes.

pub use mlopt_kit::CatoptKit;

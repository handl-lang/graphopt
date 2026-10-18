#!/usr/bin/env bash
# bupha-style gate entry for the Rust arm (the Python gates are
# scripts/gates.sh): cargo build + test of the standalone experiments
# crate + the experiment's certification leg (no timing).
set -euo pipefail

# Cross-repo git deps resolve over ssh/cli locally; the insteadOf env
# (~/.cache/devin/handl_deps.env) maps org URLs to sibling checkouts.
export CARGO_NET_GIT_FETCH_WITH_CLI="${CARGO_NET_GIT_FETCH_WITH_CLI:-true}"
if [ -z "${GIT_CONFIG_COUNT:-}" ] && [ -f "$HOME/.cache/devin/handl_deps.env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.cache/devin/handl_deps.env"
fi

# mlopt-experiments is excluded from the root workspace (own lockfile —
# see the `exclude` row in Cargo.toml).
cargo test --manifest-path experiments/Cargo.toml

# The certification leg — value-equality through the real runtime.
cargo run --release --manifest-path experiments/Cargo.toml --bin conv-bn-fold -- --repeat 0

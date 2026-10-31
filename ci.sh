#!/usr/bin/env bash
# bupha-style gate entry for the Rust arm (the Python gates are
# scripts/gates.sh): cargo build + test of the standalone experiments
# crate + the experiment's certification leg (no timing).
set -euo pipefail

# ---------------------------------------------------------------------------
# Layout plumbing - bupha runs this gate in a detached worktree of the
# pushed sha, where the relative sibling paths in
# packages/mlopt-handl-peer/Cargo.toml (`../../../kernel`,
# `../../../kit-prelude-lang`, `../../../engine-transform*`) resolve
# against the worktree's PARENT, not this repo's parent. Recreate the
# sibling layout there before cargo runs: ../graphopt -> this worktree
# (the sha under test), every other ~/GitHub/handl-lang/* -> its real
# checkout. Same model as kernel/scripts/ci.sh and lib-base64/ci.sh.
#
# NOTE: no `cd "$(dirname "$0")"` here — the watcher invokes this script
# by absolute path with the worktree as cwd, so dirname($0) is the dev
# checkout. Worktree detection keys on $PWD instead.
# ---------------------------------------------------------------------------

JOBS="${CI_JOBS:-2}"
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_PROFILE_TEST_DEBUG=0
export CARGO_BUILD_JOBS="${CI_BUILD_JOBS:-2}"
export RUST_TEST_THREADS="$JOBS"

# Workspace root = parent of the *main* checkout (correct from both the
# checkout and any detached worktree of this repo).
GIT_COMMON="$(git rev-parse --path-format=absolute --git-common-dir)"
MAIN_CHECKOUT="$(dirname "$GIT_COMMON")"
WORKSPACE_ROOT="$(dirname "$MAIN_CHECKOUT")"
IN_WORKTREE=0
[ "$PWD" != "$MAIN_CHECKOUT" ] && IN_WORKTREE=1

# Shared target dir - a detached worktree's own target/ would cold-build
# every sha. The watcher serializes runs, so sharing is safe.
export CARGO_TARGET_DIR="${CI_TARGET_DIR:-$WORKSPACE_ROOT/.ci-target/graphopt}"

ensure_siblings() {
  [ "$IN_WORKTREE" = 1 ] || return 0
  local parent; parent="$(cd .. && pwd)"
  if [ "$parent" = "$WORKSPACE_ROOT" ]; then
    echo "ci.sh: CI worktree sits inside $WORKSPACE_ROOT - ../graphopt" >&2
    echo "       would resolve to the dev checkout instead of this sha." >&2
    echo "       Move the worktree base outside the workspace root." >&2
    return 1
  fi
  # Self: ../graphopt must name THIS tree (the sha under test). Only ever
  # (re)point a symlink; refuse to clobber a real directory.
  if [ -e "$parent/graphopt" ] && [ ! -L "$parent/graphopt" ]; then
    echo "ci.sh: $parent/graphopt is a real directory - refusing to clobber" >&2
    return 1
  fi
  ln -sfn "$PWD" "$parent/graphopt"
  local d name
  for d in "$WORKSPACE_ROOT"/*/; do
    name="$(basename "$d")"
    [ "$name" = "graphopt" ] && continue
    # refresh symlinks (a previous run's links may be stale); never clobber
    # a real directory that happens to sit in the parent.
    if [ -L "$parent/$name" ] || [ ! -e "$parent/$name" ]; then
      ln -sfn "$d" "$parent/$name"
    fi
  done
  # Second-spelling hazard: a self-patch `path = "../graphopt"` would
  # spell this tree two ways and the lockfile write collides; rewrite it
  # to this worktree's absolute path - the worktree is throwaway so
  # editing Cargo.toml here is safe.
  [ -f Cargo.toml ] && sed -i "s|\.\./graphopt|$PWD|g" Cargo.toml
}

ensure_siblings

# Cross-repo git deps resolve over ssh/cli locally; the insteadOf env
# (~/.cache/devin/handl_deps.env) maps org URLs to sibling checkouts.
export CARGO_NET_GIT_FETCH_WITH_CLI="${CARGO_NET_GIT_FETCH_WITH_CLI:-true}"
if [ -z "${GIT_CONFIG_COUNT:-}" ] && [ -f "$HOME/.cache/devin/handl_deps.env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.cache/devin/handl_deps.env"
fi

# mlopt-experiments is excluded from the root workspace (own lockfile —
# see the `exclude` row in Cargo.toml).
cargo test --manifest-path packages/mlopt-experiments/Cargo.toml

# The certification leg — value-equality through the real runtime.
cargo run --release --manifest-path packages/mlopt-experiments/Cargo.toml --bin conv-bn-fold -- --repeat 0

#!/usr/bin/env bash
# ci.sh — mlopt CI gate (Python-only since the hnl-ext-mlopt crate moved
# to kit-prelude/crates/hnl-ext-mlopt — ADR-0175 Step B).
#
# Stages: doctor | py | ci (default: everything).
#
#   ci.sh            # uv sync + pytest + ruff + ty
#   ci.sh doctor     # toolchain smoke, no install
#   ci.sh py         # uv sync + pytest + ruff + ty
#
# Env: HANDL_TOKEN / BUPHA_TOKEN (GitHub token for private git deps).
set -euo pipefail

# P3 Step D-2: this member moved into the merged repo at
# packages/mlopt — the uv workspace root is two levels up.
cd "$(dirname "$0")/../.."

# Private git deps: token wins; otherwise rewrite ssh URLs to https so the
# gh credential helper / ssh config can authenticate (kernelopt model).
setup_auth() {
  local tok="${HANDL_TOKEN:-${BUPHA_TOKEN:-}}"
  if [ -n "$tok" ]; then
    export GIT_CONFIG_COUNT=1
    export GIT_CONFIG_KEY_0="url.https://x-access-token:${tok}@github.com/.insteadOf"
    export GIT_CONFIG_VALUE_0="https://github.com/"
  elif [ -z "${GIT_CONFIG_COUNT:-}" ]; then
    export GIT_CONFIG_COUNT=1
    export GIT_CONFIG_KEY_0="url.git@github.com:handl-lang/.insteadOf"
    export GIT_CONFIG_VALUE_0="https://github.com/handl-lang/"
  fi
}

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

stage_doctor() {
  uv --version
  python3 --version
  echo "workdir:   $PWD"
}

stage_py() {
  # Workspace root: `mlopt` is a member dep of the mlopt-meta dist, so
  # the root `uv sync` installs it and .venv/bin/mlopt exists.
  uv sync --extra hf --extra kokoro
  uv run pytest packages/mlopt/tests -q
  uv run ruff check packages/mlopt
  uv run ruff format --check packages/mlopt
  uv run ty check
}

stage_ci() {
  stage_py && echo "ci passed: python"
}

# ---------------------------------------------------------------------------

stage="ci"
if [ $# -gt 0 ]; then stage="$1"; shift; fi

setup_auth
case "$stage" in
  doctor)      stage_doctor ;;
  py)          stage_py ;;
  ci)          stage_ci ;;
  -h|--help|help) sed -n '2,16p' "$0" ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac

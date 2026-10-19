#!/usr/bin/env bash
# scripts/gates.sh — the one-command local verification gate.
#
# Why this exists: the gate list in AGENTS.md ran the *whole test suite
# twice* — once bare (`uv run pytest`) and once under coverage
# (`coverage run -m pytest tests/ -q`).  CI (`.github/workflows/ci.yml`)
# only ever ran it once, under coverage, so the local duplicate bought
# nothing and doubled the wall time.  This script runs the suite
# exactly once, under coverage, and then applies both coverage floors.
#
# Order is cheap-first so a red gate fails fast: the static gates
# (seconds) run before the suite (tens of minutes).
#
# Usage:
#     scripts/gates.sh                    # default parallelism (see NP)
#     MLOPT_GATES_NP=1 scripts/gates.sh  # force the suite serial
#
# Test lanes (the suite is tiered by kind — see AGENTS.md):
#     smoke:  .venv/bin/python -m pytest tests/test_ir.py \
#               tests/test_laws_structure.py tests/test_egraph_internals.py \
#               tests/test_property_ir.py tests/test_property_laws.py \
#               tests/test_cond_laws.py -q          # 230 tests, ~6 s
#     core:   -m "not e2e and not slow"             # unit/integration only
#     dev:    -m "not slow"                         # drops the minutes-scale
#     gate:   this script (everything, once, under coverage)
#
# Env knobs:
#     MLOPT_GATES_LANE
#         `dev` (default) — the day-to-day gate: static gates + the suite
#         with `-m "not slow"` (the minutes-scale soak tests are excluded;
#         coverage is not collected and the floors do not apply — both are
#         full-suite properties).  `full` — everything, once, under
#         coverage, both floors enforced.  CI/release runs should use
#         `full`.
#     MLOPT_GATES_NP
#         xdist worker count for the suite stage.  Default 1 (serial) —
#         see the measured serial-vs-parallel table in AGENTS.md.  Set to
#         2 to opt into the parallel arm (with the inductor cap below).
#     TORCHINDUCTOR_COMPILE_THREADS
#         Inductor compile-worker cap.  Defaults to 2 when NP>1, so the
#         pytest×compile fan-out stays bounded (NP·2 workers, not
#         NP·16); left at torch's default when NP=1.
#
# Every gate is a hard failure (`set -e`): the script stops at the first
# red gate.  Each stage prints its own wall time.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=.venv/bin/python
# Default SERIAL.  Measured 2026-10-15 (6,372 tests, uncontended 11 GB /
# 16-core host, whole-tree RSS sampled from /proc): NEITHER arm
# finished inside a 30-min box — serial reached 36%, `-n 2` (inductor
# cap 2) reached 98% (2.72x further, peak RSS 8,814 MB tree-summed).  A
# *completed* wall-time win is required to flip the default and neither
# arm completed, so it stays serial; opt in with MLOPT_GATES_NP=2.  See
# AGENTS.md for the measured table and the tests that dominate the wall.
NP="${MLOPT_GATES_NP:-1}"
LANE="${MLOPT_GATES_LANE:-dev}"   # dev | full

# Keep Inductor's kernel cache pinned into the repo cache dir so repeat
# runs reuse compiled kernels instead of recompiling from scratch (the
# same default tests/conftest.py installs; setting it here also covers
# the collection phase before conftest runs).
if [[ -z "${TORCHINDUCTOR_CACHE_DIR:-}" ]]; then
    _base="${XDG_CACHE_HOME:-$HOME/.cache}"
    export TORCHINDUCTOR_CACHE_DIR="${MLOPT_CACHE_DIR:-$_base/mlopt}/inductor"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
fi

# Bound the Inductor fan-out only when we are also fanning out at the
# pytest level; a single pytest process can use torch's own default.
if [[ "$NP" -gt 1 && -z "${TORCHINDUCTOR_COMPILE_THREADS:-}" ]]; then
    export TORCHINDUCTOR_COMPILE_THREADS=2
fi

t0=$SECONDS
stage() {  # stage <name> -- <cmd...>
    local name="$1"; shift
    [[ "${1:-}" == "--" ]] && shift
    local s=$SECONDS
    "$@"
    printf '  [ok] %-26s %3ds\n' "$name" "$((SECONDS - s))"
}

echo "== static gates =="
# Shipped source only.  tests/ and bench/ carry pre-existing ruff drift
# under the pinned ruff 0.16.9 (see AGENTS.md "Known drift"); packages,
# lab, scripts and examples are clean and are what the repo ships.
stage "ruff check"      -- .venv/bin/ruff check packages lab scripts examples
stage "ruff format"     -- .venv/bin/ruff format --check packages lab scripts examples
stage "ty check"        -- .venv/bin/ty check
stage "vulture"         -- .venv/bin/vulture
stage "import-linter"   -- .venv/bin/lint-imports
stage "bandit"          -- .venv/bin/bandit -c .bandit.yaml -r packages
stage "semgrep"         -- .venv/bin/semgrep --config .semgrep.yml packages
stage "radon ratchet"   -- $PY scripts/radon_ratchet.py

if [[ "$LANE" == "full" ]]; then
    echo "== test suite (once, under coverage) =="
    rm -f .coverage .coverage.*
    suite_s=$SECONDS
    if [[ "$NP" -gt 1 ]]; then
        # pytest-cov combines xdist workers' coverage (a bare `coverage run`
        # does not measure the worker subprocesses).  Its own fail_under
        # check is disabled here: it would apply the config's 100% floor to
        # the union *including* mlopt_discovery, which sits on a 99%
        # ratchet — the two `coverage report` stages below own both floors.
        $PY -m pytest tests/ -q -n "$NP" --cov --cov-report= --cov-fail-under=0
    else
        .venv/bin/coverage run -m pytest tests/ -q
    fi
    printf '  [ok] %-26s %3ds\n' "suite (NP=$NP)" "$((SECONDS - suite_s))"

    echo "== coverage floors =="
    # The six shipped packages + mlopt_onnx: fail_under=100 (config).
    stage "coverage (six, =100)" -- \
        .venv/bin/coverage report -m --omit="*/mlopt_discovery/*"
    # mlopt_discovery ratchet floor: --fail-under=99 (tighten, never lower).
    stage "coverage (discovery)" -- \
        .venv/bin/coverage report -m --include="*/mlopt_discovery/*" --fail-under=99
else
    # dev lane — the day-to-day gate: suite minus the minutes-scale `slow`
    # tests, no coverage (floors are a full-suite property; see LANE doc).
    echo "== test suite (dev lane: -m 'not slow', no coverage) =="
    suite_s=$SECONDS
    if [[ "$NP" -gt 1 ]]; then
        $PY -m pytest tests/ -q -n "$NP" -m "not slow"
    else
        $PY -m pytest tests/ -q -m "not slow"
    fi
    printf '  [ok] %-26s %3ds\n' "suite dev-lane (NP=$NP)" "$((SECONDS - suite_s))"
fi

echo "== all gates green in $((SECONDS - t0))s =="

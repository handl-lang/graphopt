#!/bin/sh
# Run a command inside a memory-capped systemd user scope.
#
# On small-RAM hosts a runaway pytest/suite can freeze the whole
# machine before earlyoom reacts.  This wraps the command in a cgroup
# cap instead: if it exceeds MLOPT_TEST_MEM, only the scope is killed
# while the rest of the system stays responsive.
#
# Usage:
#   scripts/pytest-capped.sh .venv/bin/python -m pytest tests/ -q
#   MLOPT_TEST_MEM=8G scripts/pytest-capped.sh ...
exec systemd-run --user --scope -q \
    -p "MemoryMax=${MLOPT_TEST_MEM:-6G}" \
    -p MemorySwapMax=1G \
    -- "$@"

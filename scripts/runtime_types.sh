#!/bin/sh
# Runtime-contract gate (typeguard).
#
# Instruments the domain packages (``kit_torch``, ``kit_carriers`` —
# external git deps since the P3 splits; typeguard instruments the
# installed copies).  Runs the typeguard-clean test files that remain
# in this repo — most of the original list travelled to kit-torch with
# the package (P3 Step D-1e) — so
# annotation drift surfaces as a runtime ``TypeCheckError`` rather than
# silent ``Any``.  After the ty burn-down emptied ``[tool.ty.src]
# exclude`` these tests pass under full instrumentation; grow the list
# as more test files become typeguard-clean.  The torch-integration /
# omd / mha suites are excluded: they are slow under instrumentation
# (minutes each) and/or exercise paths blocked by annotations owned
# elsewhere.
#
# Run directly (``sh scripts/runtime_types.sh``) or via the pre-commit
# ``runtime-types`` hook / the CI workflow.
set -eu
exec .venv/bin/python -m pytest -q -p no:cacheprovider \
    --typeguard-packages=kit_torch,kit_carriers \
    tests/test_cost_edge_cases.py \
    tests/test_meta_edges.py \
    tests/test_optimize_routing.py \
    tests/test_exact_corner.py \
    tests/test_rulecache_codec.py \
    tests/test_typecheck_smoke.py \
    tests/test_struct_real.py \
    tests/test_laws_rewrite_edges.py \

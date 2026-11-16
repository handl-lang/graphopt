#!/bin/sh
# Runtime-contract gate (typeguard).
#
# Instruments the shipped domain packages (``mlopt_torch``,
# ``kit_carriers``) — the substrate's own instrumentation moved to
# eqsat with it (plan 0032 phase 4). and runs the typeguard-clean test files, so
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
    --typeguard-packages=mlopt_torch,kit_carriers \
    tests/test_attrs.py \
    tests/test_cost_edge_cases.py \
    tests/test_meta_edges.py \
    tests/test_meta.py \
    tests/test_adapters.py \
    tests/test_contracts.py \
    tests/test_reports.py \
    tests/test_optimize_routing.py \
    tests/test_logging.py \
    tests/test_torch_bridge_edges.py \
    tests/test_optimize_compositional_edges.py \
    tests/test_exact_corner.py \
    tests/test_falsified_pins.py \
    tests/test_batched_verify.py \
    tests/test_calibrate.py \
    tests/test_rulecache_codec.py \
    tests/test_typecheck_smoke.py \
    tests/test_struct_real.py \
    tests/test_compositional.py \
    tests/test_hybrid.py \
    tests/test_laws_rewrite_edges.py \

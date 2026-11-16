"""mlopt terms → `.hnl` programs → CatoptKit — compat shim.

The implementation moved to :mod:`kit_discovery.hnl_surface` (the
shipped package, beside ``modelproj``'s spec-side emitter and
``hnl_import``'s reverse bridge); this module re-exports it so the
historical ``bench.drivers.kit.term_hnl`` import path — pinned by
``tests/test_kit_bridge.py``, ``tests/test_vocab_model_parity.py``,
``scripts/verify_vocab_models.py`` and
``bench/drivers/probes/conv_fold_wall_regimes_probe.py`` — keeps
working.  New code should import ``kit_discovery.hnl_surface``
directly.
"""

from __future__ import annotations

from kit_discovery.hnl_surface import (
    KitEvalError,
    KitRunner,
    UnsupportedOpError,
    emit,
    emit_program,
    leaf_names,
)

__all__ = [
    "KitEvalError",
    "KitRunner",
    "UnsupportedOpError",
    "emit",
    "emit_program",
    "leaf_names",
]

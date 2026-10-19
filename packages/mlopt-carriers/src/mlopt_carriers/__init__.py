"""mlopt-carriers — the semantic carriers and their executors.

Online-softmax monoids (:mod:`~mlopt_carriers.om`,
:mod:`~mlopt_carriers.om_lower`), the deferred omd carrier
(:mod:`~mlopt_carriers.xcarrier`,
:mod:`~mlopt_carriers.omd_lower`), affine scans
(:mod:`~mlopt_carriers.scan_lower`), and the traced-monoidal
extensions (:mod:`~mlopt_carriers.trace`,
:mod:`~mlopt_carriers.trace_lift`).  Each module declares
``TORCH_BINDINGS`` (folded into :meth:`OpTable.full`) and registers
its shape rules on import — OpTable composes them lazily so the
modules themselves stay import-safe.

The carrier law families also own their rule sets:
``mlopt_carriers.om.OM_RULES`` /
``mlopt_carriers.trace.TRACE_RULES`` /
``mlopt_carriers.xcarrier.XC_RULES`` /
``mlopt_carriers.decode_laws.DECODE_RULES`` /
``mlopt_carriers.decode_geom.DECODE_GEOM_RULES`` are
:class:`~mlopt_core.laws.RuleSet` values, and ``CARRIERS`` (below)
is their composition — the preset the orchestrator's
``DEFAULT_RULES`` adds on top of the core default.  Every name
materialises **lazily**: the law modules import their backend
(torch), so touching them at package import would couple every
``import mlopt_carriers`` to torch.
"""

from typing import Any

from mlopt_orchestrator.carriers import register_carriers

from mlopt_carriers.machinery import build_machinery

# Install the carrier machinery with the neutral orchestrator (the
# edge inversion: carriers -> orchestrator, never the reverse).  The
# value holds only thunks, so this loads no tensor library.
register_carriers(build_machinery())

__all__ = [
    "CARRIERS",
    "DECODE_GEOM_RULES",
    "DECODE_RULES",
    "OM_RULES",
    "TRACE_RULES",
    "XC_RULES",
]


#: Family rule set name -> the module that defines it.  Each module
#: imports its backend (torch), so resolution stays lazy per name.
_FAMILY_MODULES = {
    "DECODE_GEOM_RULES": "mlopt_carriers.decode_geom",
    "DECODE_RULES": "mlopt_carriers.decode_laws",
    "OM_RULES": "mlopt_carriers.om",
    "TRACE_RULES": "mlopt_carriers.trace",
    "XC_RULES": "mlopt_carriers.xcarrier",
}


def __getattr__(name: str) -> Any:
    """Compose the carrier rule sets on first access (torch-coupled).

    Resolved per name so ``decode_laws`` stays reachable in a
    torch-free process — only ``CARRIERS`` needs every family.
    """
    if name == "CARRIERS":
        from mlopt_carriers.decode_geom import DECODE_GEOM_RULES
        from mlopt_carriers.decode_laws import DECODE_RULES
        from mlopt_carriers.om import OM_RULES
        from mlopt_carriers.trace import TRACE_RULES
        from mlopt_carriers.xcarrier import XC_RULES

        rs = (
            OM_RULES
            + TRACE_RULES
            + XC_RULES
            + DECODE_RULES
            + DECODE_GEOM_RULES
        )
    elif name in _FAMILY_MODULES:
        import importlib

        rs = getattr(
            importlib.import_module(_FAMILY_MODULES[name]), name
        )
    else:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    globals()[name] = rs
    return rs

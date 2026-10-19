"""``mlopt_native`` — the optional native (Rust) search engine.

An accelerator, never a default: the pure-Python
``mlopt_core.egraph.EGraph`` remains the reference implementation and
the engine the pipeline uses unless ``engine=`` is passed explicitly.
See ``mlopt_native.engine.NativeEngine`` — an
:class:`mlopt_core.ports.Engine` over the compiled search core.
"""

from mlopt_native._native import __version__
from mlopt_native.engine import NativeEngine

__all__ = ["NativeEngine", "__version__"]

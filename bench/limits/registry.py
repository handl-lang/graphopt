"""The baseline registry — every opponent is a folder under ``limits/``.

Plan 0031 workstream 3 ("baselines, same shape"): a baseline is a
folder in this package — one measured record in its docstring, its
:data:`BaselineSpec` row(s), and a ``build`` that binds the case's
fixture into the callable a practitioner would ship.  The registry
maps *name → spec* so the limits runner
(:mod:`bench.limits.runner`) enumerates opponents from one table —
the same shape :mod:`mlopt_discovery.players.registry` gives the
player roster.

A :class:`BaselineSpec` is *data*: which corpus cases the baseline
can price, which imports it needs before it can run, whether it
wants CUDA, whether it carries a symbolic (dag_cost-pricable) form
or is measured-only, and the measured record that earned it a seat.
Loading is lazy — :func:`spec` imports the folders on first touch,
so ``import bench.limits`` stays cheap.

The adapter contract: a folder exports ``SPEC`` (or ``SPECS``) and::

    build(ctx: BaselineCtx) -> Callable

returning the arm the runner times/verifies — a module, or any
callable accepting the case's feed.  A baseline that cannot run on
this case or this environment raises :class:`SkipBaseline` with the
reason — the row reports the skip, never a silent drop.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bench.discovery.cases import DiscoveryCase

__all__ = [
    "BaselineCtx",
    "BaselineSpec",
    "SkipBaseline",
    "availability",
    "build",
    "for_case",
    "names",
    "spec",
    "specs",
]


class SkipBaseline(Exception):
    """A baseline declining a case — the row records *why*."""


@dataclass(frozen=True)
class BaselineSpec:
    """One baseline's registration — its seat on the roster.

    *cases* is the honest capability statement: the discovery-corpus
    case names the ``build`` accepts (``()`` = every case).
    *requires* names modules that must import for the arm to exist
    (``fla`` for the Triton kernels — a scratch dep, absent from the
    lockfile); *needs_cuda* gates CUDA-only kernels; *symbolic* says
    the arm has a dag_cost-pricable form (``False`` = measured-only,
    the fast lane reports it but cannot price it).  *record* is the
    one-line measured record the folder docstring argues at length.
    """

    name: str
    kind: str  # "eager" | "compile" | "order" | "kernel" | "impl" | "carrier"
    title: str = ""
    cases: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    needs_cuda: bool = False
    symbolic: bool = True
    note: str = ""
    record: str = ""


@dataclass
class BaselineCtx:
    """Everything a ``build`` may read — the case, its fixture, the run."""

    case: DiscoveryCase
    model: Any  # the spelled "before" module
    feed: Any  # tensor or positional-args tuple
    sr: Any  # the search() result (extracted term inside)
    sink: Any  # TorchSink — verify + lower ports
    source: Any  # TorchSource — the IR exporter
    device: str = "cpu"
    lane: str = "fast"


#: Module key → the folder's module path.  The folder itself declares
#: its ``SPEC``/``SPECS``; this table only names where to look, so
#: :func:`names` and error messages never pay an import beyond the
#: cheap folder bodies.
_MODULES: dict[str, str] = {
    "eager": "bench.limits.eager",
    "torch_compile": "bench.limits.torch_compile",
    "opt_einsum": "bench.limits.opt_einsum",
    "fla": "bench.limits.fla",
    "paper_impl": "bench.limits.paper_impl",
    "hand_carrier": "bench.limits.hand_carrier",
}

#: spec name → (module path, BaselineSpec) — built by :func:`specs`.
_INDEX: dict[str, tuple[str, BaselineSpec]] = {}


def _load_specs() -> None:
    """Import every baseline folder once; index its specs by name."""
    for mod_name in _MODULES.values():
        mod = importlib.import_module(mod_name)
        mod_specs = getattr(mod, "SPECS", None) or (mod.SPEC,)
        for s in mod_specs:
            _INDEX[s.name] = (mod_name, s)


def specs() -> dict[str, BaselineSpec]:
    """Return ``{name: BaselineSpec}`` over the registered roster."""
    if not _INDEX:
        _load_specs()
    return {name: s for name, (_m, s) in _INDEX.items()}


def names() -> tuple[str, ...]:
    """Return the registered baseline names, sorted."""
    specs()
    return tuple(sorted(_INDEX))


def _known(name: str) -> ValueError:
    """Build the unknown-baseline error, with the roster in it."""
    return ValueError(
        f"unknown baseline {name!r} — registered: {', '.join(names())}"
    )


def _module(name: str) -> Any:
    """Import the baseline folder for *name* (a spec or module key)."""
    if name in _MODULES:
        return importlib.import_module(_MODULES[name])
    specs()
    if name not in _INDEX:
        raise _known(name)
    return importlib.import_module(_INDEX[name][0])


def spec(name: str) -> BaselineSpec:
    """Return the named baseline's spec; raise ``ValueError`` if absent."""
    specs()
    if name not in _INDEX:
        raise _known(name)
    return _INDEX[name][1]


def for_case(case_name: str) -> list[BaselineSpec]:
    """Every registered baseline that can price *case_name*, sorted."""
    return [
        s
        for _n, s in sorted(specs().items())
        if not s.cases or case_name in s.cases
    ]


def availability(name: str, *, device: str = "cpu") -> str | None:
    """``None`` when the baseline can run, else the reason it cannot.

    Checks the spec's ``requires`` imports and ``needs_cuda`` gate —
    the first blocker wins, so the row's caveat names a single cause.
    """
    s = spec(name)
    for mod in s.requires:
        if importlib.util.find_spec(mod) is None:
            return f"{mod} not installed (scratch dep)"
    if s.needs_cuda and not (
        device.startswith("cuda") or _cuda_available()
    ):
        return "requires CUDA — pass --device cuda"
    return None


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def build(name: str, ctx: BaselineCtx) -> Callable:
    """Return the baseline's arm for *ctx* — the callable to time.

    Raises :class:`SkipBaseline` when the baseline cannot run this
    case/environment, ``ValueError`` for unknown names — the runner
    maps the skip to a row, never a silent drop.
    """
    s = spec(name)
    if s.cases and ctx.case.name not in s.cases:
        raise SkipBaseline(
            f"does not price case {ctx.case.name!r} "
            f"(cases: {', '.join(s.cases)})"
        )
    return _module(name).build(ctx, spec=s)

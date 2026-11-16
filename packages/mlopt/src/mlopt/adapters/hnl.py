"""``.hnl`` source adapter — ADR-0172: ``.hnl`` as a first-class input.

``mlopt optimize foo.hnl`` / ``mlopt optimize hnl:foo.hnl`` lifts Handl
surface syntax into the mlopt term pipeline: ``hnl_to_term`` (the
``kit_discovery`` codec — stdlib + ``mlopt_core.term`` only) parses the
source into an ``mlopt_core.term`` term, this adapter binds leaf values,
and the returned spec carries a :class:`HnlSource` — a
``mlopt_core.ports.Source`` that answers ``to_ir`` with the
pre-lifted ``(IR, param_values)``.  ``Optimizer.search`` then runs the
SAME saturate → extract → lower → verify flow a ``torch.export``-lifted
module runs; nothing downstream knows the frontend was text.

Honest bounds, stated up front:

* ``.hnl`` artifacts carry *shapes*, not tensor data —
  ``hnl_to_term`` mints ``Param`` leaves for ``do``-block constant
  bindings but the values stay on the feed side (the codec's
  documented contract).  Real values come from
  ``--adapter-opt feed=<path.json>``; with no feed every leaf gets a
  deterministic synthetic value (seeded ``torch.randn``,
  ``--adapter-opt seed=N``) so the verify gate still compares the
  reference term's lowering against the delivered one — the check is
  real, the weights are not the artifact's.  ``synthetic=false``
  declines instead of inventing values.
* ``Var`` leaves need concrete shapes to synthesize probe inputs —
  ``--adapter-opt leaves='{"x": [2, 3]}'`` (JSON) annotates the
  wildcard-shaped free identifiers; a Var that stays shapeless and
  unfed declines at load, naming the leaf.
* Only the codec's ``tensor.*`` import surface is covered; anything
  else raises ``UnsupportedOpError``/``HnlImportError`` at load —
  the honest bound, not a guess.
* ``mlopt export`` artifacts unwrap: the ``export name := (params) ~>``
  signature line is interface text (the codec strips module
  directives but not ``export``), so :func:`_unwrap_export` reduces it
  to the ``do`` body before parsing — the declared parameters become
  the term's free ``Var`` leaves, exactly as the signature meant.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlopt_core.ports import Source
from mlopt_core.term import IR, Op, Param, TensorType, Var

from mlopt.models import ModelSpec, register_adapter

__all__ = ["ADAPTER", "HnlLift", "HnlSource"]


# ---------------------------------------------------------------------------
#  Source reading — path resolution, export-signature unwrap, leaf census
# ---------------------------------------------------------------------------

_EXPORT_LINE = re.compile(r"^export\s+[A-Za-z_][\w.]*\s*:=\s*.*~>\s*", re.M)


def _ref_path(ref: str) -> Path:
    """Strip the ``hnl:`` scheme (name-routing leaves it on the ref)."""
    ref = ref.strip()
    if ref.startswith("hnl:"):
        ref = ref[len("hnl:") :]
    return Path(ref)


def _unwrap_export(src: str) -> str:
    """Reduce an ``export f := (params) ~> <body>`` signature to *body*.

    ``mlopt export`` artifacts open with the export signature — module
    interface text the codec's directive stripper does not cover (it
    is not an ``import``/``cap``/``op`` line).  The signature declares
    the function's parameters; unwrapping to the body leaves them as
    free identifiers, which ``hnl_to_term`` reads as ``Var`` leaves —
    the same leaves the signature named.  Only the FIRST ``export``
    line is unwrapped; anything else is left for the parser to judge.
    """
    return _EXPORT_LINE.sub("", src, count=1)


def _census(term: Any) -> tuple[list[Var], dict[str, Param]]:
    """First-use-ordered ``Var`` leaves and name-keyed ``Param`` leaves."""
    vars_: list[Var] = []
    params: dict[str, Param] = {}
    seen_v: set[str] = set()

    def go(t: Any) -> None:
        if isinstance(t, Var):
            if t.name not in seen_v:
                seen_v.add(t.name)
                vars_.append(t)
            return
        if isinstance(t, Param):
            params.setdefault(t.name, t)
            return
        if isinstance(t, Op):
            for a in t.args:
                go(a)

    go(term)
    return vars_, params


def _concrete_shape(typ: TensorType) -> tuple[int, ...] | None:
    """The leaf's shape as ints, or ``None`` when it is not concrete."""
    shape = getattr(typ, "shape", None)
    if shape is None:
        return None
    if any(d is None or isinstance(d, bool) or not isinstance(d, int) for d in shape):
        return None
    return tuple(int(d) for d in shape)


# ---------------------------------------------------------------------------
#  Adapter options — leaves / feed / synthetic / seed
# ---------------------------------------------------------------------------


def _parse_leaves(raw: Any) -> dict[str, Any]:
    """``--adapter-opt leaves='{"x": [2, 3]}'`` → the codec's shape map."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"leaves: not valid JSON ({e})") from e
    if not isinstance(raw, dict):
        raise ValueError(f"leaves: expected a JSON object, got {type(raw).__name__}")
    return dict(raw)


def _feed_tensor(spec: Any, name: str) -> Any:
    """One feed entry → a torch tensor (``{"shape","data"}`` or nested list)."""
    import torch

    shape: tuple[int, ...] | None = None
    data: Any = spec
    if isinstance(spec, dict):
        if "hnl" in spec:
            raise ValueError(
                f"feed[{name!r}]: 'hnl' expressions are computed inputs — "
                "the .hnl source path cannot evaluate them; use "
                "{'shape','data'} or a nested list"
            )
        if "shape" not in spec or "data" not in spec:
            raise ValueError(f"feed[{name!r}]: mapping needs 'shape'+'data'")
        shape = tuple(int(d) for d in spec["shape"])
        data = spec["data"]
    try:
        t = torch.as_tensor(data, dtype=torch.float32)
    except Exception as e:  # noqa: BLE001 — report the entry, not a traceback class
        raise ValueError(f"feed[{name!r}]: not array-like ({e})") from e
    if shape is not None:
        if t.numel() != _numel(shape):
            raise ValueError(f"feed[{name!r}]: {t.numel()} values do not fill shape {list(shape)}")
        t = t.reshape(shape)
    return t


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _load_feed(raw: Any) -> dict[str, Any]:
    """``--adapter-opt feed=<path.json>`` → ``{leaf: tensor-spec}``."""
    if raw is None:
        return {}
    if not isinstance(raw, (str, Path)):
        raise ValueError(f"feed: expected a JSON file path, got {type(raw).__name__}")
    path = Path(raw)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ValueError(f"feed: cannot read {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"feed: {path} is not valid JSON ({e})") from e
    if not isinstance(data, dict):
        raise ValueError(f"feed: {path} must contain a JSON object of leaf values")
    return data


def _truthy(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
#  The Source port — the pre-lifted (IR, leaf values) the search consumes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HnlLift:
    """What ``load`` hands ``spec.module``: the lifted term + its IR.

    Carried for introspection (``doctor``/reports); the pipeline reads
    the IR through :class:`HnlSource`, not this payload.
    """

    term: Any
    ir: IR
    path: Path


class HnlSource(Source):
    """``mlopt_core.ports.Source`` over a pre-lifted ``.hnl`` term.

    ``to_ir`` returns the ``(IR, param_values)`` the adapter built at
    load time — the same contract ``TorchSource`` fulfils via
    ``torch.export``, minus the frontend.  ``model``/``example_inputs``
    are accepted for the port signature and ignored: the lift already
    happened.
    """

    def __init__(self, ir: IR, param_values: dict[str, Any]) -> None:
        self._ir = ir
        self._param_values = param_values

    def to_ir(self, model: Any, example_inputs: Any) -> tuple[IR, dict[str, Any]]:
        """Return the lifted IR plus its param leaf values."""
        return self._ir, dict(self._param_values)


# ---------------------------------------------------------------------------
#  Leaf binding — feed values where given, seeded synthetic otherwise
# ---------------------------------------------------------------------------


def _synth(shape: tuple[int, ...], gen: Any) -> Any:
    import torch

    return torch.randn(shape, generator=gen)


def _bind_leaf(
    name: str,
    shape: tuple[int, ...] | None,
    *,
    feed: dict[str, Any],
    feed_tensors: dict[str, Any],
    synthetic: bool,
    gen: Any,
) -> tuple[Any | None, str]:
    """Resolve one leaf's value → ``(tensor, provenance)``.

    Feed wins; a concrete shape gets a seeded synthetic value when
    ``synthetic`` is on; anything else declines naming the leaf.
    """
    if name in feed:
        if name not in feed_tensors:
            feed_tensors[name] = _feed_tensor(feed[name], name)
        return feed_tensors[name], "feed"
    if shape is None:
        return None, "unbound"
    if not synthetic:
        return None, "unbound"
    return _synth(shape, gen), "synthetic"


def _lift(
    path: Path,
    src: str,
    *,
    leaves: Any,
    feed: Any,
    synthetic: bool,
    seed: int,
) -> tuple[Any, IR, tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    """Source text → ``(term, ir, inputs, param_values, census)``."""
    from kit_discovery.hnl_import import hnl_to_term

    leaf_shapes = _parse_leaves(leaves)
    term = hnl_to_term(_unwrap_export(src), leaves=leaf_shapes or None)

    vars_, params = _census(term)
    feed_map = _load_feed(feed)
    feed_tensors: dict[str, Any] = {}

    import torch

    gen = torch.Generator().manual_seed(int(seed))

    inputs: list[Any] = []
    input_prov: dict[str, str] = {}
    unbound: list[str] = []
    for v in vars_:
        shape = _concrete_shape(v.typ)
        val, prov = _bind_leaf(
            v.name,
            shape,
            feed=feed_map,
            feed_tensors=feed_tensors,
            synthetic=synthetic,
            gen=gen,
        )
        if val is None:
            unbound.append(f"Var {v.name!r} (shape={getattr(v.typ, 'shape', None)!r})")
            continue
        inputs.append(val)
        input_prov[v.name] = prov

    param_values: dict[str, Any] = {}
    param_prov: dict[str, str] = {}
    for name, p in params.items():
        shape = _concrete_shape(p.typ)
        val, prov = _bind_leaf(
            name,
            shape,
            feed=feed_map,
            feed_tensors=feed_tensors,
            synthetic=synthetic,
            gen=gen,
        )
        if val is None:
            unbound.append(f"Param {name!r} (shape={getattr(p.typ, 'shape', None)!r})")
            continue
        param_values[name] = val
        param_prov[name] = prov

    if unbound:
        raise ValueError(
            "hnl: unbound leaves — "
            + "; ".join(unbound)
            + ".  Pass --adapter-opt leaves='{\"name\": [dims]}' for "
            "shapeless Vars, feed=<json> for real values, or keep "
            "synthetic=true (the default) for seeded probe values"
        )

    # ``unbound`` raises above, so every Var bound — inputs stay in
    # first-use order, matching ``ir.inputs``.
    ir = IR(
        root=term,
        inputs=list(vars_),
        input_names={v.name for v in vars_},
        params=params,
    )
    census = {
        "vars": {v.name: list(_concrete_shape(v.typ) or ()) for v in vars_},
        "params": {n: list(_concrete_shape(p.typ) or ()) for n, p in params.items()},
        "input_values": input_prov,
        "param_values": param_prov,
    }
    return term, ir, tuple(inputs), param_values, census


# ---------------------------------------------------------------------------
#  The adapter
# ---------------------------------------------------------------------------


class _HnlAdapter:
    """Lifts ``.hnl`` files / ``hnl:`` refs into optimizable specs."""

    name = "hnl"
    description = (
        ".hnl Handl-surface source — hnl_to_term lift, tensor.* op "
        "surface; leaf values from feed=<json> or seeded synthetic"
    )
    available = True
    unavailable_reason = ""

    def claims(self, ref: str) -> bool:
        ref = ref.strip()
        return ref.endswith(".hnl") or ref.startswith("hnl:")

    def _read(self, ref: str) -> tuple[Path, str]:
        path = _ref_path(ref)
        try:
            return path, path.read_text(encoding="utf-8")
        except OSError as e:
            raise ValueError(f"hnl: cannot read {path}: {e}") from e

    def _opts(self, opts: dict[str, Any]) -> dict[str, Any]:
        return {
            "leaves": opts.get("leaves"),
            "feed": opts.get("feed"),
            "synthetic": _truthy(opts.get("synthetic"), True),
            "seed": opts.get("seed", 0),
        }

    def probe(self, ref: str) -> dict[str, Any]:
        """Cheap ``doctor`` report — parse + leaf census, no values."""
        path, src = self._read(ref)
        try:
            from kit_discovery.hnl_import import hnl_to_term

            term = hnl_to_term(_unwrap_export(src))
        except Exception as e:  # noqa: BLE001 — the probe reports, not raises
            return {
                "path": str(path),
                "parse": f"declined — {type(e).__name__}: {e}",
            }
        vars_, params = _census(term)
        return {
            "path": str(path),
            "parse": "ok",
            "vars": [v.name for v in vars_],
            "params": sorted(params),
            "note": (
                "leaf values bind at load: feed=<json> for real data, "
                "else seeded synthetic (synthetic=false declines)"
            ),
        }

    def load(self, ref: str, **opts: Any) -> ModelSpec:
        path, src = self._read(ref)
        term, ir, inputs, param_values, census = _lift(path, src, **self._opts(opts))
        provs = {**census["input_values"], **census["param_values"]}
        kinds = set(provs.values())
        if not provs:
            leaf_source = "none"
        elif kinds == {"feed"}:
            leaf_source = "feed"
        elif kinds == {"synthetic"}:
            leaf_source = "synthetic"
        else:
            leaf_source = "mixed"
        lift = HnlLift(term=term, ir=ir, path=path)
        return ModelSpec(
            module=lift,
            inputs=inputs,
            meta={
                "ref": ref.strip(),
                "adapter": "hnl",
                "path": str(path),
                "leaf_source": leaf_source,
                "leaves": census,
            },
            source=HnlSource(ir, param_values),
        )


ADAPTER = _HnlAdapter()
register_adapter(ADAPTER)

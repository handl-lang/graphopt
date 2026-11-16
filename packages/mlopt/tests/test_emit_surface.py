"""``emit._OPS`` ↔ ``host_surface.json`` drift guard.

Closes the codec-fragmentation debt flagged by
``graphopt:docs/reports/arch_review_0174.md`` §4 (ADR-0174 invariant
4): the ~203-entry hand-written IR-op → ``tensor.*`` builder table had
no guard against the canonical generated surface
(``kit_host_tensor.host_surface.json``, 239 ops).  These tests pin:

* **forward** — every ``_OPS`` builder, executed on a canonical probe
  term, emits a primary ``tensor.<name>(…)`` call that exists in the
  surface artifact with an emittable status (``Ported``/``Handwritten``,
  never ``Seam``) and an arg count inside the port's ``min_args``/
  ``max_args`` bounds (bounds are ``null`` for the handwritten
  variadic ports — existence + status is all they pin);
* **reverse** — every canonical ``mlopt_core.opmeta.REGISTRY`` op the
  emitter does not cover (and ``pack``, which is special-cased to a
  ``.hnl`` list literal rather than a ``tensor.*`` call) declines
  through :class:`UnsupportedOpError` naming the op.

The probe is deliberately *behavioral*: it runs the real builder and
reads the emitted ``.hnl`` text, so a builder that silently retargets
a renamed/removed surface op fails here — no parallel name table to
drift.  ``_PROBE`` carries the minimal ``(args, attrs)`` each builder
needs to reach its call; ops not listed take six tensor operands and
no attrs (every scalar slot then defaults honestly).
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from kit_host_tensor.surface import host_ops
from mlopt_core.ir import Const, Op, TensorType, Var
from mlopt_core.opmeta import REGISTRY

from mlopt.emit import _MECH_SPECS, _MECH_TAGS, _OPS, UnsupportedOpError, _tensor_slots, emit

_T = TensorType((2, 3))


def _v(i: int) -> Var:
    return Var(f"a{i}", _T)


#: Six operand terms — more than any builder consumes; extras are
#: ignored exactly as surplus ``argN``-less operands would be.
_BASE_ARGS: list[Any] = [_v(i) for i in range(6)]

#: Builders that need attrs (or trimmed arity) to reach their call.
#: Values are the *canonical* spellings a real term carries.
_PROBE: dict[str, tuple[list[Any], dict[str, Any]]] = {
    # --- scalar-bound / elementwise pair forms -------------------------
    "clamp": ([_v(0), _v(1), _v(2)], {}),
    "where": ([_v(0), _v(1), _v(2)], {}),
    "max": ([_v(0), _v(1)], {}),
    "min": ([_v(0), _v(1)], {}),
    "norm": ([_v(0)], {}),
    # --- dim-carrying reductions/shape ops ------------------------------
    "cumsum": ([_v(0)], {"dim": 0}),
    "cumprod": ([_v(0)], {"dim": 0}),
    "softmax": ([_v(0)], {"dim": 0}),
    "log_softmax": ([_v(0)], {"dim": 0}),
    "unsqueeze": ([_v(0)], {"dim": 0}),
    "squeeze": ([_v(0)], {"dim": 0}),
    "median": ([_v(0)], {"dim": 0}),
    "logsumexp": ([_v(0)], {"dim": 0}),
    "layer_norm": ([_v(0)], {"dim": [2]}),
    "rms_norm": ([_v(0)], {"dim": [2]}),
    "reshape": ([_v(0)], {"shape": [2, 3]}),
    "view": ([_v(0)], {"shape": [2, 3]}),
    # --- split family ----------------------------------------------------
    "chunk": ([_v(0)], {"chunks": 2}),
    "split": ([_v(0)], {"sizes": [1, 1]}),
    # --- scalar constructors ---------------------------------------------
    "arange": ([Const(0), Const(5), Const(1)], {}),
    # --- string/list-carrying attrs --------------------------------------
    "einsum": ([_v(0), _v(1)], {"equation": "ab,bc->ac"}),
    "tensordot": ([_v(0), _v(1)], {"dims_self": [0], "dims_other": [1]}),
    "movedim": ([_v(0)], {"source": [0], "destination": [1]}),
    # --- pooling ---------------------------------------------------------
    "avg_pool2d": ([_v(0)], {"kernel_size": [2, 2]}),
    "max_pool2d": ([_v(0)], {"kernel_size": [2, 2]}),
    # --- decomposed ops (primary call is the outermost helper) -----------
    "permute": ([_v(0)], {"dim": [1, 0]}),
    "conv1d": ([_v(0), _v(1), _v(2)], {"stride": (1,), "padding": (0,)}),
    # --- casts -------------------------------------------------------------
    # ``to``'s dtype attr is gated before the builder; the probe spells
    # the no-attr identity form (one operand → "float64" spec arg).
    "to": ([_v(0)], {}),
}


def _go(t: Any) -> str:
    """Leaf stand-in — builders only need *a* name back."""
    return getattr(t, "name", "?")


_CALL = re.compile(r"tensor\.(\w+)\(")


def _primary_call(expr: str) -> tuple[str, int]:
    """``(tensor-op-name, top-level arg count)`` of the emitted call.

    The primary call is the *first* ``tensor.<name>(`` in the emitted
    expression — builders spell their target outermost and helper
    calls (``from_list``/``reshape``/``zeros`` rank-0 wraps, the
    synthesized ``linear`` bias) only inside argument position.
    ``(tensor.chunk(…))[i]`` / ``(tensor.sort(…)).values`` wrappers
    keep the call textually first.
    """
    m = _CALL.search(expr)
    if m is None:
        raise AssertionError(f"no tensor.* call emitted: {expr!r}")
    depth = 1
    instr = esc = False
    i = m.end()
    while i < len(expr) and depth:
        c = expr[i]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
        elif c == '"':
            instr = True
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        i += 1
    inner = expr[m.end() : i - 1]
    # Top-level commas = argc - 1 (a lone arg has none).
    argc = 0
    depth = 0
    instr = esc = False
    for c in inner:
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
            continue
        if c == '"':
            instr = True
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            argc += 1
    return m.group(1), argc + (1 if inner.strip() else 0)


def _emitted(op_name: str) -> str:
    args, kw = _PROBE.get(op_name, (list(_BASE_ARGS), {}))
    return _OPS[op_name](list(args), dict(kw), _go)


def test_every_ops_entry_resolves_to_the_canonical_surface() -> None:
    """Forward guard — 203 builders, one artifact, no parallel table."""
    ops = host_ops()
    assert ops, "host_surface.json loaded empty — dep/packaging broke"
    failures: list[str] = []
    for name in sorted(_OPS):
        try:
            expr = _emitted(name)
        except Exception as exc:  # noqa: BLE001 — collect, don't stop at first
            failures.append(f"{name}: probe emit failed — {type(exc).__name__}: {exc}")
            continue
        tensor_op, argc = _primary_call(expr)
        key = f"tensor.{tensor_op}"
        entry = ops.get(key)
        if entry is None:
            failures.append(f"{name}: emits {key} — not in host_surface.json ({expr})")
            continue
        if entry["status"] not in {"Ported", "Handwritten"}:
            failures.append(f"{name}: {key} status={entry['status']!r} — not emittable")
            continue
        lo, hi = entry["min_args"], entry["max_args"]
        if lo is not None and argc < lo:
            failures.append(f"{name}: {key} emitted {argc} args < min_args={lo} ({expr})")
        elif hi is not None and argc > hi:
            failures.append(f"{name}: {key} emitted {argc} args > max_args={hi} ({expr})")
    assert not failures, "emit._OPS drifted from host_surface.json:\n" + "\n".join(failures)


def test_mechanical_specs_resolve_from_the_surface() -> None:
    """Class-(a) builders are generated views over ``host_surface.json``.

    ``_canonicalize_mechanical`` swaps every ``_u1``/``_u2`` candidate
    (recorded in ``_MECH_TAGS``) the artifact fully determines
    (``tensor.<name>`` exists, emittable status, signature = the tagged
    arity of bare ``Tensor`` slots) for a generated builder recorded in
    ``_MECH_SPECS``.  This pins the split in both directions: every
    spec resolves to a real surface op whose signature still decodes to
    the recorded arity, specs are a subset of the tagged candidates,
    and the candidates the artifact *cannot* vouch for keep their
    hand-written factories.
    """
    ops = host_ops()
    assert _MECH_SPECS, "no mechanical specs resolved — surface/dep broke"
    assert set(_MECH_SPECS) <= set(_MECH_TAGS), "specs outside the candidate set"
    assert set(_MECH_TAGS) <= set(_OPS), "tagged candidates missing from _OPS"
    failures: list[str] = []
    for name, (target, arity) in sorted(_MECH_SPECS.items()):
        entry = ops.get(target)
        if entry is None:
            failures.append(f"{name}: spec target {target} not in host_surface.json")
            continue
        if entry["status"] not in {"Ported", "Handwritten"}:
            failures.append(f"{name}: {target} status={entry['status']!r} — not emittable")
        if _tensor_slots(str(entry.get("signature", ""))) != arity:
            failures.append(
                f"{name}: {target} signature={entry.get('signature')!r} "
                f"no longer decodes to {arity} Tensor slots"
            )
    assert not failures, "_MECH_SPECS drifted from host_surface.json:\n" + "\n".join(failures)
    # The undecodable-signature candidates must keep their factories —
    # ``Value...`` signatures cannot prove the slot kinds a
    # name-passthrough builder needs (``Scalar | Tensor`` rhs included).
    # Pin the *whole* kept-factory set: any candidate dropping out of
    # ``_MECH_SPECS`` is a surface/signature change demanding review.
    kept = set(_MECH_TAGS) - set(_MECH_SPECS)
    assert kept == {"zeros_like", "ones_like", "copysign"}, (
        f"unexpected kept factories: {sorted(kept)}"
    )


def test_uncovered_registry_ops_decline_by_name() -> None:
    """Reverse guard — canonical ops outside ``_OPS`` must refuse loudly.

    ``pack`` is the one intentional exception: ``emit`` special-cases
    it to a ``.hnl`` list literal (the multi-output root), not a
    ``tensor.*`` call, so it never reaches ``_OPS``.
    """
    uncovered = sorted(set(REGISTRY) - set(_OPS) - {"pack"})
    assert uncovered, "expected uncovered canonical ops — REGISTRY/_OPS drifted"
    for name in uncovered:
        with pytest.raises(UnsupportedOpError, match=re.escape(name)):
            emit(Op.make(name))

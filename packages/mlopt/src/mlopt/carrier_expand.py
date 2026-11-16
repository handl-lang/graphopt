"""Carrier elimination for the ``.hnl`` emit path — honest re-spelling.

Carrier-winning extractions carry ``om_*`` ops (``om_apply`` /
``om_elem`` / ``om_decayed`` / ``om_compose`` …) that have no
``tensor.*`` spelling — the batched-executor delivery path
(``kit_carriers.om_lower``) is their consumer, not ``hnl-run``.
Two of the carrier laws are *definitional* unfolds, though — the
same equalities ``kit_carriers.om`` ships as ``OM_UNLIFT`` /
``OM_DECAYED_UNLIFT``:

* ``om_apply(om_elem(s, v))`` ≡ ``matmul(softmax(s, dim=-1), v)`` —
  ``om_elem`` IS the online-softmax element; ``om_apply`` its
  ``a / l`` readout.  The unlift is exact, not an approximation:
  both sides are the same e-class member.
* ``om_decayed(s, d, v)`` ≡ ``matmul(mul(s, d), v)`` — the spelled
  ``(s ⊙ d) @ v``, verbatim (``_om_decayed_torch`` lowers it
  identically).

:func:`expand_carriers` applies exactly those two rewrites,
structurally, bottom-up.  Everything else carrier-shaped
(``om_compose`` trees, ``omd_*``, ``aff*``/``apply*``, ``trace``,
``scan``, decode ops, a bare ``om_elem`` outside ``om_apply``)
passes through untouched — :func:`mlopt.emit.emit` then declines by
op name, which stays the honest bound: this module never *guesses* a
carrier's tensor meaning.

The expansion introduces no leaves and changes no values — it is
the same term re-spelled into the surface ``tensor.*`` can express.
The carrier *schedule* win (level-batched executor evaluation) is
not lost silently: ``tensor.*`` never had that spelling, so the
artifact honestly carries the dense form — exactly what the
``verify=True`` gate already compared.
"""

from __future__ import annotations

from typing import Any

from mlopt_core.term import Op

__all__ = ["expand_carriers"]


def _is_op(t: Any, op: str, n: int) -> bool:
    return isinstance(t, Op) and t.op == op and len(t.args) == n


def expand_carriers(term: Any, memo: dict | None = None) -> Any:
    """Rewrite the definitional carrier unfolds out of *term*.

    Bottom-up over the (hash-consed) term tree; non-``Op`` leaves
    (``Var``/``Param``/``Const``/literals) pass through.  Returns
    *term* itself when nothing changed — interned ``Op.make`` makes
    the no-op case free.
    """
    if memo is None:
        memo = {}
    if not isinstance(term, Op):
        return term
    if term in memo:
        return memo[term]

    args = tuple(expand_carriers(a, memo) for a in term.args)

    out: Any = None
    if (
        term.op == "om_apply"
        # Canonical arity/attr contract (the OM_UNLIFT LHS is
        # ``om_apply(om_elem(s, v))`` — one arg, no attrs): a
        # non-canonical ``om_apply`` (extra args, attrs) would have
        # them silently dropped by the rewrite, so it passes through
        # to emit's named decline instead.  The same gate applies to
        # the ``om_elem`` child — its attrs would be dropped too.
        and len(args) == 1
        and not term.attrs
        and _is_op(args[0], "om_elem", 2)
        and not args[0].attrs
    ):
        # OM_UNLIFT: om_apply(om_elem(s, v)) -> matmul(softmax(s,-1), v)
        el = args[0]
        s, v = el.args
        out = Op.make("matmul", Op.make("softmax", s, dim=-1), v)
    elif term.op == "om_decayed" and len(args) == 3 and not term.attrs:
        # OM_DECAYED_UNLIFT: om_decayed(s, d, v) -> matmul(mul(s, d), v)
        # (arity was already checked; attrs would be silently dropped)
        s, d, v = args
        out = Op.make("matmul", Op.make("mul", s, d), v)

    if out is not None:
        memo[term] = out
        return out
    if args == term.args:
        memo[term] = term
        return term
    rebuilt = Op.make(term.op, *args, **term.attrs)
    memo[term] = rebuilt
    return rebuilt

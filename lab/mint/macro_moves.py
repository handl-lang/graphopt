"""Macro actions — minted move sequences replayed as one refereed step.

``claim`` fused one winning line by hand (declare → accept → fire →
handle inside a single move); a *macro* fuses any mined sequence by
data.  ``Action("macro", {"seq": [{"op": ..., "params": {...}}, ...]})``
replays each sub-action through the same :data:`ACTIONS` dispatch
inside one ``MetaArena.step``: every legality check, the two-phase
declare accept, the feasibility bound and the terminal certificate
see exactly what the spelled line would have seen.  A declined
sub-move declines the whole macro; a terminal sub-move ends it.

Soundness is the antiunify argument: a macro mints no behaviour a
player could not already spell — it only compresses the line into
one step (one ``step`` charge, one budget tick).  The handler keeps
no global state; the minted sequences are data mined from
:class:`~mlopt_discovery.meta_arena.Trajectory` reports by
``lab/mint/macro_miner.py``.

The offer seam is :func:`legal_with_macros` — a ``legal(state)``
wrapper the players' ``legal=`` injection point already accepts —
plus :func:`viable`, the cheap filter that keeps a minted line off
cases where its recorded params cannot bind (a ``claim`` whose tag
is absent, a ``declare`` whose spec does not occur, a ``fire`` of a
rule nobody minted).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlopt_discovery import meta_arena as ma
from mlopt_discovery.meta_arena import Action, _Outcome

__all__ = [
    "Macro",
    "instantiate",
    "legal_with_macros",
    "viable",
]

#: Default cap on macro offers per state — the legal set stays small.
DEFAULT_MACRO_CAP = 8


def _sub_parts(sub: Any) -> tuple[str, dict]:
    """Read ``(op, params)`` off a macro step — ``Action`` or dict."""
    if isinstance(sub, dict):
        return str(sub["op"]), dict(sub.get("params") or {})
    return str(sub.op), dict(getattr(sub, "params", {}))


@ma.register_action("macro")
def _macro(arena: ma.MetaArena, action: Action) -> _Outcome:
    """Replay ``params["seq"]`` through ``ACTIONS`` as one refereed step.

    Each sub-outcome that carries rules is accepted two-phase on the
    spot (mirroring ``_claim_one``) so later steps see the inserted
    rules and definitions; a declined sub-move declines the macro at
    that step, a terminal one propagates cost/certificate verbatim.
    ``"macro"`` inside a seq is refused — composition stays flat.
    """
    seq = tuple(action.params.get("seq") or ())
    if not seq:
        return _Outcome(applied=False, note="macro: empty seq")
    notes: list[str] = []
    inserted: list[str] = []
    for i, sub in enumerate(seq):
        op, params = _sub_parts(sub)
        if op == "macro":
            return _Outcome(
                applied=False, note="macro: nested macro declined"
            )
        fn = ma.ACTIONS.get(op)
        if fn is None:
            return _Outcome(
                applied=False,
                note=f"macro: unregistered sub-move {op!r}",
            )
        out = fn(arena, Action(op, params))
        if out.applied and out.rules:
            out, ins = arena._accept(out)
            inserted.extend(ins)
        notes.append(f"{i}:{op}")
        if not out.applied:
            return _Outcome(
                applied=False,
                note=f"macro: step {i} {op} declined ({out.note})",
                detail={
                    "steps": tuple(notes),
                    "inserted": tuple(inserted),
                },
            )
        if out.terminal:
            return _Outcome(
                terminal=True,
                cost=out.cost,
                cost_unfolded=out.cost_unfolded,
                certificate_ok=out.certificate_ok,
                used_declared=out.used_declared,
                note=f"macro[{i + 1}/{len(seq)}] → {out.note}",
                detail={
                    "steps": tuple(notes),
                    "inserted": tuple(inserted),
                    "terminal": dict(out.detail),
                },
            )
    return _Outcome(
        note=f"macro[{len(seq)}]",
        detail={"steps": tuple(notes), "inserted": tuple(inserted)},
    )


@dataclass(frozen=True)
class Macro:
    """A minted move sequence — the template ``legal_with_macros`` offers.

    ``seq`` is recorded action data (``{"op", "params"}`` dicts) from a
    representative winning trace; the stats columns are the miner's
    ranking evidence — ``count`` occurrences, ``certified`` share of
    them in certificate-passing episodes, ``mean_terminal`` the mean
    terminal-step reward of those episodes, ``score`` the rank key.
    """

    name: str
    seq: tuple
    count: int = 0
    certified: float = 0.0
    mean_terminal: float = 0.0
    score: float = 0.0
    ops: tuple = field(default=())

    def op_str(self) -> str:
        """Return the op sequence — ``declare→saturate→handle→extract``."""
        ops = self.ops or tuple(str(s["op"]) for s in self.seq)
        return "→".join(ops)


def _declared_name(sub_params: dict) -> str | None:
    """Return the object name a ``declare`` step mints, when spelled."""
    cons = sub_params.get("construction") or {}
    cp = cons.get("params") or {}
    name = cp.get("name")
    return str(name) if name else None


def instantiate(macro: Macro) -> Action:
    """Bind a minted template into a concrete ``macro`` action.

    Declared object names are freshened to ``mx_<macro>_<name>`` —
    including the self-named ``kernel`` head of a fresh fold — and
    ``handle`` targets declared earlier in the same seq are renamed
    to match.  Everything else replays verbatim: the recorded
    ``spelled`` spec, handler tags, saturate budgets are the data the
    trace earned.

    The miner's evidence rides along on ``params["macro"]`` as plain
    data — ``{"name", "count", "certified", "mean_terminal",
    "score"}``.  The referee only reads ``params["seq"]`` (replay is
    untouched); the stats are what a macro-aware featurizer prices —
    ``score``/``certified`` are the action-level signal the broadcast
    row could never carry.  Plain scalars (not the ``Macro`` record
    itself — its ``seq`` of dicts is unhashable) keep the shared
    ``_action_key``/played-mask hashing and reprs deterministic.
    """
    rename: dict[str, str] = {}
    seq: list[dict] = []
    for sub in macro.seq:
        op, params = _sub_parts(sub)
        if op == "declare":
            cons = dict(params.get("construction") or {})
            cp = dict(cons.get("params") or {})
            old = cp.get("name")
            if old:
                new = f"mx_{macro.name}_{old}"
                rename[str(old)] = new
                cp["name"] = new
                kernel = cp.get("kernel")
                if (
                    isinstance(kernel, (tuple, list))
                    and kernel
                    and kernel[0] == old
                ):
                    cp["kernel"] = (new, *kernel[1:])
            cons["params"] = cp
            params["construction"] = cons
        elif op == "handle":
            obj = str(params.get("object", ""))
            if obj in rename:
                params["object"] = rename[obj]
        elif op == "fire":
            rule = str(params.get("rule", ""))
            if rule in rename:
                params["rule"] = rename[rule]
        elif op == "saturate" and params.get("rules"):
            params["rules"] = tuple(
                rename.get(str(r), r) for r in params["rules"]
            )
        seq.append({"op": op, "params": params})
    return Action(
        "macro",
        {
            "seq": seq,
            "macro": {
                "name": macro.name,
                "count": macro.count,
                "certified": macro.certified,
                "mean_terminal": macro.mean_terminal,
                "score": macro.score,
            },
        },
    )


def viable(macro: Macro, state: ma.MetaState) -> bool:
    """Return whether *macro* is worth an offer slot on *state*.

    A macro whose recorded params cannot bind on this board is not
    offered: ``claim`` needs its tag in ``state.claim_tags`` (the
    concrete-occurrence check the board already publishes);
    ``declare`` needs its spelled spec's coverage skeleton among the
    occurring mined specs (``state.sites`` — a dead declare still
    *applies*, but it burns the step and the paying line needs the
    rule to fire); ``handle`` needs its object minted earlier in the
    seq or already declared; ``fire``/``saturate`` need their named
    rules live or seq-declared.
    """
    sites = {ma._canon_spec(state.specs[i]) for i in state.sites}
    declared = {op for op, _body in state.declared_bodies}
    live = set(state.rules)
    seq_declared: set[str] = set()
    for sub in macro.seq:
        op, params = _sub_parts(sub)
        if op == "declare":
            cons = params.get("construction") or {}
            cp = cons.get("params") or {}
            name = cp.get("name")
            if name:
                seq_declared.add(str(name))
            spelled = cp.get("spelled")
            if (
                spelled is not None
                and ma._canon_spec(spelled) not in sites
            ):
                return False
        elif op == "claim":
            if str(params.get("handler", "")) not in state.claim_tags:
                return False
        elif op == "handle":
            obj = str(params.get("object", ""))
            if obj not in seq_declared and obj not in declared:
                return False
        elif op == "fire":
            rule = str(params.get("rule", ""))
            if rule not in live and rule not in seq_declared:
                return False
        elif op == "saturate":
            for name in params.get("rules") or ():
                name = str(name)
                if name not in live and name not in seq_declared:
                    return False
    return True


def legal_with_macros(
    macros: Any,
    *,
    base: Any = None,
    cap: int = DEFAULT_MACRO_CAP,
) -> Any:
    """Return a ``legal(state)`` enumerator offering minted macros.

    *base* defaults to ``meta_arena.legal_actions``; macros come
    ranked, filtered by :func:`viable`, deduplicated against each
    other and the base set by the shared ``_action_key``, capped at
    *cap* offers, and slotted before the terminal ``extract`` so the
    enumeration's "extract last" convention holds.
    """
    base = ma.legal_actions if base is None else base
    ranked = tuple(macros)

    def legal(state: ma.MetaState) -> tuple:
        acts = list(base(state))
        extract = (
            acts.pop() if acts and acts[-1].op == "extract" else None
        )
        seen = {ma._action_key(a) for a in acts}
        offered = 0
        for m in ranked:
            if offered >= cap:
                break
            if not viable(m, state):
                continue
            a = instantiate(m)
            key = ma._action_key(a)
            if key in seen:
                continue
            seen.add(key)
            acts.append(a)
            offered += 1
        if extract is not None:
            acts.append(extract)
        return tuple(acts)

    return legal

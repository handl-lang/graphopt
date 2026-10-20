"""ADR-0007 layer 3 prototype: an atom's laws from measured bindings.

The runtime-assembly loop, proven on the hand-built carrier atoms
(``om_decayed`` / ``applyd`` / ``decay_vec`` — the ops
:mod:`mlopt_carriers.om` ships with hand-written law sets).  Given an
atom the driver runs the closed loop:

1. **candidate enumeration** — small declarative schema batteries per
   atom (self-composition, pointwise distribution, unfold, cat
   naturality), the RHS metavar slots enumerated over a bounded
   elementwise grammar (``_EW_OPS``, ≤2 op nodes, hard candidate cap);
2. **measured binding environments** — the shipped enumerator
   (``evidence._synth_sites``, the same window ``auto_cond`` measures)
   unioned with each schema's typed shape families and the real corpus
   matches of the LHS pattern;
3. **numeric oracle** — ``oracle.eval_instance`` per env, early-bailed:
   one ``unequal`` kills an unguarded candidate;
4. **guard synthesis** — survivors with a mixed domain go through
   ``object_synthesis.auto_cond_object``; attr gaps go through a
   solver-measured derive fitted by ``dspec_synth.synth_dspec_multi``
   (the ``decay_vec`` merge's ``offset``/``sign`` are *minted*, not
   written);
5. **certification** — ``verify_law`` on a witness instance against
   the composed default ruleset: ``derivable`` = the shipped library
   already implies it (assembly reuses); ``primitive`` = a genuinely
   new candidate law — plus the serializability flag.

Budgets are explicit (the box is 11 GB — the prior hang was a quartic
candidate product): ``--max-candidates`` caps each family's pool,
``--synth-limit`` the env window, ``--seconds`` the per-atom
wall-clock.

Run::

    .venv/bin/python bench/drivers/mint/atom_law_loop.py
    .venv/bin/python bench/drivers/mint/atom_law_loop.py --atom decay_vec --json out.json
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import mlopt_carriers  # noqa: F401 — registers carrier bindings
import torch
from mlopt_core.egraph import Rewrite
from mlopt_core.egraph.terms import _term_instantiate
from mlopt_core.ir import Const, Op, TensorType, Var, op_repr
from mlopt_core.laws.serialize import law_to_data, missing_hooks
from mlopt_core.meta import match_pattern
from mlopt_discovery import evidence as ev
from mlopt_discovery import object_synthesis as obs
from mlopt_discovery import oracle as lvo
from mlopt_discovery import proposal as lp
from mlopt_discovery.dspec_synth import synth_dspec_multi
from mlopt_discovery.impact import _iter_subterms
from mlopt_discovery.verifier import verify_law

from bench.drivers.screen.exposure_screen import _export, _workloads


def _p(op: str, *args: Any, **attrs: Any) -> Op:
    return Op.make(op, *args, **attrs)


#: Marker for a candidate whose RHS is *solved*, not spelled — the
#: affine-in-state slot solve fills ``applyd(aff_diag(E1,E2), h)``
#: from measured slot values (:func:`_affine_candidates`).
_AFFD_SOLVED = "applyd(aff_diag(E1,E2), h) — slots solved"


# ---------------------------------------------------------------------------
#  Candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One candidate equality over the atom — patterns plus hooks."""

    name: str
    lhs: Any
    rhs: Any
    dspec: Any = None
    cond: Any = None
    note: str = ""


def _ew_exprs(
    leaves: tuple[str, ...], ops: tuple[str, ...], max_ops: int
) -> list[Any]:
    """Bounded elementwise IR grammar over *leaves*, simplest first.

    Level 0 = leaves; level n = op over (expr, leaf) or (leaf, expr)
    pairs from the previous level — deduped by ``op_repr``, capped by
    *max_ops* (2 op nodes — the compose/bias forms live there).
    """
    out: list[Any] = list(leaves)
    level: list[Any] = list(leaves)
    seen = {op_repr(t) for t in level}
    for _ in range(max_ops):
        nxt: list[Any] = []
        for op in ops:
            for e in level:
                for leaf in leaves:
                    for t in (_p(op, e, leaf), _p(op, leaf, e)):
                        r = op_repr(t)
                        if r not in seen:
                            seen.add(r)
                            nxt.append(t)
        out.extend(nxt)
        level = nxt
    return out


def _affd_candidates(cap: int) -> list[Candidate]:
    """``applyd`` candidates — the affine-in-state family.

    ``applyd(aff_diag(a,b), h) = a⊙h + b`` is affine in ``h``, so for
    ANY ``lhs`` term affine in its state metavar the folded map is
    recoverable *per env*: ``L = d lhs/d h`` (a two-draw difference) and
    ``B = lhs - L⊙h``.  The battery is one schema — lhs patterns whose
    rhs is the folded ``applyd(aff_diag(E1,E2), h)`` — and the driver
    solves the slots (see :func:`_affine_candidates`): self-compose,
    add-merge, scale-absorb and the matmul readout all land in the
    same grammar, none of them spelled by hand.
    """
    nested = _p(
        "applyd",
        _p("aff_diag", "a1", "b1"),
        _p("applyd", _p("aff_diag", "a2", "b2"), "h"),
    )
    merged = _p(
        "add",
        _p("applyd", _p("aff_diag", "a1", "b1"), "h"),
        _p("applyd", _p("aff_diag", "a2", "b2"), "h"),
    )
    scaled = _p("mul", _p("applyd", _p("aff_diag", "a", "b"), "h"), "r")
    readout = _p(
        "matmul", "e", _p("applyd", _p("aff_diag", "a", "b"), "h")
    )
    cands = [
        # solved families — lhs carries the state metavar "h"
        Candidate("affd.compose", nested, _AFFD_SOLVED, note="solved"),
        Candidate("affd.addmerge", merged, _AFFD_SOLVED, note="solved"),
        Candidate("affd.scale", scaled, _AFFD_SOLVED, note="solved"),
        Candidate("affd.readout", readout, _AFFD_SOLVED, note="solved"),
        # spelled: the zero-bias collapse (rhs is not applyd-shaped)
        Candidate(
            "affd.nobias",
            _p("applyd", _p("aff_diag", "a", Const(0.0)), "h"),
            _p("mul", "a", "h"),
        ),
        # a spelled false twin — kept so the battery's rejections are
        # exercised: addmerge over *different* states cannot fold
        Candidate(
            "affd.addmerge_wrongstate",
            _p(
                "add",
                _p("applyd", _p("aff_diag", "a1", "b1"), "h1"),
                _p("applyd", _p("aff_diag", "a2", "b2"), "h2"),
            ),
            _p(
                "applyd",
                _p(
                    "aff_diag",
                    _p("add", "a1", "a2"),
                    _p("add", "b1", "b2"),
                ),
                "h1",
            ),
            note="deliberately false — states differ",
        ),
    ]
    return cands[: cap or None]


def _om_decayed_candidates(cap: int) -> list[Candidate]:
    """``om_decayed(s,d,v) = (s⊙d)@v`` — unfold, absorb, cat-split."""
    unfold = _p("om_decayed", "s", "d", "v")
    spelled = _p("matmul", _p("mul", "s", "d"), "v")
    return [
        Candidate("omd.unfold", unfold, spelled),
        Candidate("omd.refold", spelled, unfold),
        Candidate(
            "omd.absorb_score",
            _p("om_decayed", _p("mul", "s", "x"), "d", "v"),
            _p("om_decayed", "s", _p("mul", "x", "d"), "v"),
        ),
        Candidate(
            "omd.absorb_value",
            _p("om_decayed", "s", "d", _p("mul", "v", "x")),
            _p("om_decayed", "s", "d", "v"),
        ),
        Candidate(
            "omd.rowcat",
            _p(
                "om_decayed",
                _p("concat", "s1", "s2", dim="D"),
                _p("concat", "d1", "d2", dim="D"),
                "v",
            ),
            _p(
                "concat",
                _p("om_decayed", "s1", "d1", "v"),
                _p("om_decayed", "s2", "d2", "v"),
                dim="D",
            ),
        ),
        Candidate(
            # the false twin — kept to show the loop rejecting it
            "omd.keycat_wrong",
            _p(
                "om_decayed",
                _p("concat", "s1", "s2", dim="D"),
                "d",
                "v",
            ),
            _p(
                "concat",
                _p("om_decayed", "s1", "d", "v"),
                _p("om_decayed", "s2", "d", "v"),
                dim="D",
            ),
        ),
    ]


def _decay_vec_candidates(cap: int) -> list[Candidate]:
    """``decay_vec(g, t, offset, sign) = g^{sign·k + offset}`` — merges."""
    lhs_merge = _p(
        "mul",
        _p("decay_vec", "g", t="T", offset="O1", sign="S1"),
        _p("decay_vec", "g", t="T", offset="O2", sign="S2"),
    )
    rhs_merge = _p("decay_vec", "g", t="T", offset="O", sign="S")
    cands = [
        # the true exponent merge — both attrs derived additively
        Candidate(
            "dv.merge",
            lhs_merge,
            rhs_merge,
            dspec={
                "O": ("add", ("attr", "O1"), ("attr", "O2")),
                "S": ("add", ("attr", "S1"), ("attr", "S2")),
            },
            note="exponents add: sign and offset both sum",
        ),
        # the solver+dspec_synth twin — same pattern, derive withheld:
        # the loop must measure the attr values and synthesize them
        Candidate(
            "dv.merge_synth",
            lhs_merge,
            rhs_merge,
            note="dspec withheld — solver measures (O,S), "
            "synth_dspec_multi mints them",
        ),
        # sign-gated variant — a plausible-but-false attr guess:
        # offset merges but sign must stay S1 (wrong whenever S1 != S2)
        Candidate(
            "dv.merge_sign_s1",
            lhs_merge,
            _p("decay_vec", "g", t="T", offset="O", sign="S1"),
            dspec={"O": ("add", ("attr", "O1"), ("attr", "O2"))},
            note="deliberately wrong sign slot — oracle should kill "
            "or guard it",
        ),
        # unit: decay_vec at sign 0 is a constant vector
        Candidate(
            "dv.unit_mul",
            _p(
                "mul",
                "x",
                _p("decay_vec", "g", t="T", offset=0, sign=0),
            ),
            "x",
            note="g^0 = 1 — the zero-exponent vector is the mul unit",
        ),
    ]
    return cands


_ATOMS: dict[str, Any] = {
    "applyd": _affd_candidates,
    "om_decayed": _om_decayed_candidates,
    "decay_vec": _decay_vec_candidates,
}

#: Typed shape families per schema — the measured-binding corner the
#: generic leaf bank reaches only sparsely (carrier ops have typed
#: slots; a randn env with wrong ranks is a declined site, not
#: evidence).  Each entry maps metavar → shape tuple.
_TYPED_FAMILIES: dict[str, list[dict[str, tuple]]] = {
    "applyd": [
        {"a1": (4,), "b1": (4,), "a2": (4,), "b2": (4,), "h": (4,)},
        {
            "a1": (3, 4),
            "b1": (3, 4),
            "a2": (3, 4),
            "b2": (3, 4),
            "h": (4,),
        },
        {"a": (4,), "b": (4,), "h": (4,), "r": ()},
        {"a": (3, 4), "b": (3, 4), "h": (4,), "r": ()},
        # readout corner: matmul contracts the map's leading (K) axis
        # while h stays broadcast — the elementwise-Jacobian corner.
        {"e": (5, 4), "a": (4, 6), "b": (4, 6), "h": (6,)},
        {"e": (2, 5, 4), "a": (4, 6), "b": (4, 6), "h": (6,)},
    ],
    "om_decayed": [
        {"s": (4, 4), "d": (4, 4), "v": (4, 3), "x": (4, 4)},
        {"s": (2, 4, 4), "d": (4, 4), "v": (2, 4, 3), "x": (1,)},
        {
            "s1": (2, 4),
            "s2": (3, 4),
            "d1": (2, 4),
            "d2": (3, 4),
            "v": (4, 3),
            "d": (5, 4),
            "s": (5, 4),
        },
        # column-split corner — the FALSE region of ``omd.rowcat``
        # made evaluable: concat on dim ±1 is well-typed here, so the
        # equality is measured, not just declined.
        {
            "s1": (2, 3),
            "s2": (2, 5),
            "d1": (2, 3),
            "d2": (2, 5),
            "v": (8, 3),
            "d": (2, 8),
            "s": (2, 8),
        },
    ],
    "decay_vec": [
        {"g": (), "x": (6,)},
        {"g": (3,), "x": (3, 6)},
    ],
}

#: Attr-metavar domains per atom — the measured envs also bind the
#: attr metavars a pattern carries (``decay_vec``'s ``t``/``offset``/
#: ``sign``, the ``concat`` dim).  The product is diagonal-ordered and
#: capped like the site windows.
_ATTR_DOMAINS: dict[str, dict[str, tuple]] = {
    "decay_vec": {
        "T": (5, 6),
        "O1": (-2, -1, 0, 1, 2),
        "O2": (-2, -1, 0, 1, 2),
        "S1": (-1, 0, 1),
        "S2": (-1, 0, 1),
    },
    "om_decayed": {"D": (0, -2, 1, -1)},
    "applyd": {},
}


# ---------------------------------------------------------------------------
#  Affine-in-state solve — the applyd slot recoverer
# ---------------------------------------------------------------------------


def _slot_match(
    bank: list[Any], bound: dict, env: dict, target: Any
) -> list[Any]:
    """Bank exprs whose measured value broadcast-matches *target*."""
    out = []
    for e in bank:
        try:
            t = _term_instantiate(e, bound)
        except Exception:
            continue
        ok, val = lvo._eval(t, env)
        if not ok or not isinstance(val, torch.Tensor):
            continue
        try:
            va, ta = torch.broadcast_tensors(val, target)
        except RuntimeError:
            continue
        if lp._allclose(va, ta, lvo._TOL):
            out.append(e)
    return out


def _affine_candidates(
    cand: Candidate,
    envs: list[tuple[dict, Any]],
    *,
    state: str = "h",
    solve_envs: int = 4,
    cap: int = 12,
) -> list[Candidate]:
    """Solve the folded map's two slots from measured envs.

    ``applyd(aff_diag(a,b), h) = a⊙h + b`` is affine in the state, so
    any lhs affine in *state* decomposes per env as
    ``L = (lhs(h+δ) - lhs(h))/δ`` and ``B = lhs(h) - L⊙h`` (δ drawn
    bounded away from 0).  ``h`` enters applyd-family folds
    elementwise along its own axes, so the per-element division
    recovers the composed map's linear slot exactly — including the
    ``matmul`` readout whose Jacobian is diagonal in the broadcast
    axis.  Each slot's value is matched against the bounded expr bank
    and *intersected across envs*, so a wrong expr dies on one env —
    no pair product, the quartic corner never opens.  Survivors emit
    spelled ``applyd(aff_diag(E1,E2), h)`` candidates that re-enter
    the ordinary measure/guard/certify path; an empty slot set is a
    refusal, not a guess.
    """
    metas = sorted(
        s
        for s in _iter_subterms(cand.lhs)
        if isinstance(s, str) and s != state
    )
    if state not in {
        s for s in _iter_subterms(cand.lhs) if isinstance(s, str)
    }:
        return []
    bank = _ew_exprs(tuple(metas), ("add", "mul", "sub", "matmul"), 2)
    l_set: list[Any] | None = None
    b_set: list[Any] | None = None
    fitted = 0
    for bound, lhs_i in envs:
        if fitted >= solve_envs:
            break
        h_var = bound.get(state)
        if not isinstance(h_var, Var):
            continue
        env = lvo._env_for(lhs_i)
        if env is None or h_var not in env:
            continue
        ok1, v1 = lvo._eval(lhs_i, env)
        if not ok1 or not isinstance(v1, torch.Tensor):
            continue
        delta = torch.randn_like(env[h_var]) + 2.0
        env2 = {**env, h_var: env[h_var] + delta}
        ok2, v2 = lvo._eval(lhs_i, env2)
        if not ok2 or not isinstance(v2, torch.Tensor):
            continue
        try:
            lin = (v2 - v1) / delta
            bias = v1 - lin * env[h_var]
        except (RuntimeError, ZeroDivisionError):
            continue
        l_now = _slot_match(bank, bound, env, lin)
        b_now = _slot_match(bank, bound, env, bias)
        l_set = (
            l_now if l_set is None else [e for e in l_set if e in l_now]
        )
        b_set = (
            b_now if b_set is None else [e for e in b_set if e in b_now]
        )
        fitted += 1
        if not l_set or not b_set:
            break
    if not fitted or not l_set or not b_set:
        return []
    out = []
    for i, (e1, e2) in enumerate(
        itertools.islice(itertools.product(l_set, b_set), cap)
    ):
        out.append(
            Candidate(
                f"{cand.name}[{i}]",
                cand.lhs,
                _p("applyd", _p("aff_diag", e1, e2), state),
                note=f"solved slots L={op_repr(e1)} B={op_repr(e2)}",
            )
        )
    return out


# ---------------------------------------------------------------------------
#  The loop
# ---------------------------------------------------------------------------


@dataclass
class Verdict:
    """One candidate's measured row."""

    name: str
    outcome: str  # unconditional | guarded | rejected-* | refused-*
    envs: int = 0
    equal: int = 0
    unequal: int = 0
    errors: int = 0
    cond: Any = None
    dspec: Any = None
    real_sites: int = 0
    verify: str = ""  # derivable | primitive | ""
    witness: str = ""
    note: str = ""
    rule: Any = None


def _typed_envs(atom: str, lhs: Any) -> list[tuple[dict, Any]]:
    """Instantiate *lhs* under typed shape families x attr domains.

    Leaf metavars bind to fresh ``Var``s of the family's declared
    shapes; attr metavars bind to the atom's attr-domain product
    (diagonal order, capped) — a ``(bound, lhs_term)`` env exactly as
    ``_synth_sites`` yields them.
    """
    out = []
    attr_names = sorted(_attr_metavars(lhs))
    doms = _ATTR_DOMAINS.get(atom, {})
    attr_lists = [doms.get(a, (0, 1, -1)) for a in attr_names]
    attr_combos = itertools.islice(lvo._diag_product(attr_lists), 48)
    for combo in attr_combos:
        base_attrs = {
            f"$attr:{n}": v
            for n, v in zip(attr_names, combo, strict=True)
        }
        for shapes in _TYPED_FAMILIES.get(atom, []):
            mvs = {s for s in _iter_subterms(lhs) if isinstance(s, str)}
            if isinstance(lhs, str):
                mvs.add(lhs)
            if not mvs <= set(shapes):
                continue
            bound = {
                m: Var(f"{m}@{i}", TensorType(tuple(shapes[m])))
                for i, m in enumerate(sorted(mvs))
            }
            bound.update(base_attrs)
            try:
                lhs_i = _term_instantiate(lhs, bound)
            except Exception:
                continue
            out.append((bound, lhs_i))
    return out


def _real_envs(lhs: Any, corpus_terms: Iterable[Any]) -> list:
    """Binding envs of every strict LHS match in the corpus terms."""
    out = []
    for term in corpus_terms:
        for sub in _iter_subterms(term):
            if isinstance(sub, Op):
                s = match_pattern(lhs, sub)
                if s is not None:
                    out.append((s, sub))
    return out


def _outcome(rule: Rewrite, bound: dict, lhs_i: Any) -> str:
    """``_site_outcome`` wrapper — the probe is the rule itself."""
    return ev._site_outcome(rule, bound, lhs_i)


def _attr_metavars(term: Any) -> set[str]:
    """Every attr value that is a metavar string, over all subterms."""
    out: set[str] = set()
    for t in _iter_subterms(term):
        if isinstance(t, Op):
            out |= {a for a in t.attrs.values() if isinstance(a, str)}
    return out


def _measure(
    rule: Rewrite, envs: list[tuple[dict, Any]], *, bail: int = 1
) -> dict:
    """Tally outcomes over envs; early-bail after *bail* bad verdicts."""
    t = {
        "envs": 0,
        "equal": 0,
        "unequal": 0,
        "rhs_err": 0,
        "other": 0,
        "declined": 0,
        "first_lhs": None,
        "first_rhs": None,
    }
    for bound, lhs_i in envs:
        t["envs"] += 1
        o = _outcome(rule, bound, lhs_i)
        if o == "equal":
            t["equal"] += 1
            if t["first_lhs"] is None:
                t["first_lhs"] = lhs_i
                inst = dict(bound)
                if rule.derive is not None:
                    with contextlib.suppress(Exception):
                        inst.update(rule.derive(bound) or {})
                with contextlib.suppress(Exception):
                    t["first_rhs"] = _term_instantiate(rule.rhs, inst)
        elif o == "unequal":
            t["unequal"] += 1
        elif o == "rhs-err":
            t["rhs_err"] += 1
        elif o in ("declined", "guard-err"):
            t["declined"] += 1
        else:
            t["other"] += 1
        if (
            o in ("unequal", "rhs-err")
            and (t["unequal"] + t["rhs_err"]) >= bail
        ):
            break
    return t


def _mint_axis_guard(
    cand: Candidate, envs: list[tuple[dict, Any]]
) -> Rewrite | None:
    """Mint an ``axis``/``attr-eq`` guard covering this env list.

    The ``auto_cond`` fallback for attr-heavy carrier patterns: its
    measured domain is the generic enumerator, which cannot reach a
    ``concat dim`` / ``decay_vec t`` binding the way the typed
    families do.  The bank here is deliberately small — one-clause
    ``("axis", T, A, k)`` normalizations and ``("attr-eq", A, v)``
    constants over the envs' own values — evaluated on THIS measured
    list, so the minted guard is honestly measured, not assumed.  A
    guard is minted only when it accepts every equal env and declines
    every measured-bad one, and the guarded rule is re-measured over
    the whole env list before the caller certifies it.
    """
    bare = Rewrite(
        name=cand.name, lhs=cand.lhs, rhs=cand.rhs, dspec=cand.dspec
    )
    outcomes = [_outcome(bare, b, li) for b, li in envs]
    eq = {i for i, o in enumerate(outcomes) if o == "equal"}
    bad = {
        i
        for i, o in enumerate(outcomes)
        if o in ("unequal", "rhs-err", "guard-err")
    }
    if not eq or not bad:
        return None
    leaf_mvs = [
        s for s in _iter_subterms(cand.lhs) if isinstance(s, str)
    ]
    attr_mvs = sorted(_attr_metavars(cand.lhs))
    attr_vals = {
        a: {
            b.get(f"$attr:{a}")
            for b, _ in envs
            if b.get(f"$attr:{a}") is not None
        }
        for a in attr_mvs
    }
    preds = [
        ("axis", mv, a, k)
        for mv in leaf_mvs
        for a in attr_mvs
        for k in (-3, -2, -1, 0, 1)
    ]
    preds += [("attr-eq", a, v) for a in attr_mvs for v in attr_vals[a]]
    from mlopt_core.laws.cond import eval_cond

    def covers(pred: Any) -> bool:
        try:
            return all(
                eval_cond(pred, envs[i][0]) for i in eq
            ) and not any(eval_cond(pred, envs[i][0]) for i in bad)
        except (ValueError, TypeError, KeyError):
            return False

    good = [p for p in preds if covers(p)]
    if not good:
        return None
    cond = min(good, key=repr)
    guarded = Rewrite(
        name=cand.name,
        lhs=cand.lhs,
        rhs=cand.rhs,
        cond=cond,
        dspec=cand.dspec,
    )
    re = _measure(guarded, envs, bail=len(envs) + 1)
    if re["equal"] < len(eq) or re["unequal"] or re["rhs_err"]:
        return None
    return guarded


def _corpus_with(atom: str) -> list[Any]:
    """Corpus terms containing the atom op (LHS-match source)."""
    terms = []
    for source, wl in _workloads("all"):
        _row, case = _export(wl, source)
        if case is None:
            continue
        if any(
            isinstance(s, Op) and s.op == atom
            for s in _iter_subterms(case.term)
        ) or atom in ("decay_vec", "om_decayed"):
            # the spelled-lhs patterns (mul/concat over plain ops) can
            # match terms that never name the atom, so the corpus
            # corner stays whole for them
            terms.append(case.term)
    return terms


def _default_rules() -> list:
    from mlopt_orchestrator.optimize import default_rules

    return list(default_rules().rules)


def _certify(
    verdict: Verdict, rules: list, lhs_i: Any, rhs_i: Any
) -> None:
    """verify_law on the witness instance; fill the verdict fields."""
    if lhs_i is None or rhs_i is None:
        verdict.verify = "no-instance"
        return
    res = verify_law(lhs_i, rhs_i, rules)
    verdict.verify = "derivable" if res.derivable else "primitive"
    verdict.witness = (
        f"{res.direction} via {list(res.witness_rules)[:4]}"
        if res.derivable
        else res.stop
    )
    # alpha-identity against the shipped library: a measured-true
    # pattern that IS a shipped law's structure is a duplicate, not a
    # new candidate (the store keys on the same identity).
    lib = [lp._key(r.lhs, r.rhs) for r in rules]
    rel = lp._relation(verdict.rule.lhs, verdict.rule.rhs, lib)
    if rel in ("duplicate", "inverse"):
        verdict.verify = f"subsumed-{rel}"
    elif rel == "tautology":
        verdict.verify = "rejected-tautology"


def _solve_dv_merge(lhs_i: Any, bound: dict) -> dict | None:
    """Measured derive for the decay_vec merge: solve (O, S) on values.

    Evaluates the instantiated ``mul(dv.., dv..)`` lhs on the env's
    fp64 draw, then grid-searches ``(o, s)`` in [-4, 4]² for the
    ``decay_vec`` whose value matches — the *measured* output the
    synthesizer fits a declarative ``dspec`` to.
    """
    env = lvo._env_for(lhs_i)
    if env is None:
        return None
    ok, val = lvo._eval(lhs_i, env)
    if not ok or not isinstance(val, torch.Tensor):
        return None
    g_term = bound.get("g")
    if g_term is None:
        return None
    ok2, g_val = lvo._eval(g_term, env)
    if not ok2 or not isinstance(g_val, torch.Tensor):
        return None
    t = int(bound.get("$attr:T", 0) or 0)
    if t <= 0:
        return None
    pos = torch.arange(t, dtype=torch.float64)
    gg = torch.as_tensor(g_val, dtype=torch.float64)[..., None]
    for o in range(-4, 5):
        for s in range(-4, 5):
            cand = gg ** (pos * s + o)
            if torch.allclose(cand, val, rtol=1e-9, atol=1e-12):
                return {"$attr:O": o, "$attr:S": s}
    return None


def _synth_attr_derive(
    cand: Candidate, envs: list[tuple[dict, Any]]
) -> Any:
    """Solve-measure the free RHS attrs, then ``synth_dspec_multi``.

    For a candidate whose RHS carries unbound attr metavars, the
    solver measures the satisfying values per env and the
    synthesizer fits declarative exprs over the bound attrs — the
    derive half of the loop, exercised live on ``dv.merge_synth``.
    """
    rhs_attrs = _attr_metavars(cand.rhs)
    free = rhs_attrs - _attr_metavars(cand.lhs)
    if not free:
        return {}
    if cand.name != "dv.merge_synth":
        return None  # no solver registered — refuse honestly
    bounds: list[tuple[dict, dict]] = []
    for bound, lhs_i in envs:
        got = _solve_dv_merge(lhs_i, bound)
        if got is not None:
            bounds.append((bound, got))
    if len(bounds) < 4:
        return None
    table = {id(b): got for b, got in bounds}

    def measured(bound: dict) -> dict | None:
        return table.get(id(bound))

    return synth_dspec_multi(measured, [b for b, _ in bounds])


def _run_atom(
    atom: str,
    cands: list[Candidate],
    corpus: list[Any],
    rules: list,
    args: argparse.Namespace,
) -> dict:
    """Run the measure→guard→certify loop over one atom's battery."""
    rows: list[Verdict] = []
    deadline = time.time() + args.seconds
    env_cache: dict[str, list] = {}

    def envs_for(cand: Candidate) -> list:
        """Measured envs for one candidate — cached per LHS.

        A battery shares one LHS across many RHS variants (the compose
        enumeration is one pattern over ~160 derive spellings); the
        enumeration depends on the pair only through attr metas, so
        key on ``op_repr(lhs)`` and union the per-candidate attr
        corner inside ``_typed_envs``.
        """
        key = op_repr(cand.lhs)
        if key not in env_cache:
            envs = list(_typed_envs(atom, cand.lhs))
            with contextlib.suppress(Exception):
                envs += list(
                    ev._synth_sites(
                        cand.lhs, cand.rhs, limit=args.synth_limit
                    )
                )
            envs += _real_envs(cand.lhs, corpus)
            env_cache[key] = envs
        return env_cache[key]

    real_cache: dict[str, int] = {}
    for cand in cands:
        if time.time() > deadline:
            rows.append(
                Verdict(cand.name, "skipped-budget", note="wall-clock")
            )
            continue
        envs = envs_for(cand)

        if cand.rhs is _AFFD_SOLVED:
            sub = _affine_candidates(cand, envs)
            if not sub:
                rows.append(
                    Verdict(
                        cand.name,
                        "refused-affine",
                        envs=len(envs),
                        note="no slot expr reproduces the measured "
                        "linear/bias terms",
                    )
                )
                continue
            # expand the solved family into spelled candidates and
            # continue the loop with them on the next pass
            cands.extend(sub)
            continue

        dspec = cand.dspec
        if dspec is None and cand.name.endswith("_synth"):
            dspec = _synth_attr_derive(cand, envs)
            if dspec is None:
                rows.append(
                    Verdict(
                        cand.name,
                        "refused-derive",
                        envs=len(envs),
                        note="solver found no attr fit to mint",
                    )
                )
                continue
        rule = Rewrite(
            name=cand.name,
            lhs=cand.lhs,
            rhs=cand.rhs,
            cond=cand.cond,
            dspec=dspec,
        )
        t = _measure(rule, envs)
        row = Verdict(
            cand.name,
            "unproven",
            envs=t["envs"],
            equal=t["equal"],
            unequal=t["unequal"] + t["rhs_err"],
            errors=t["other"],
            dspec=dspec,
            real_sites=real_cache.setdefault(
                op_repr(cand.lhs), len(_real_envs(cand.lhs, corpus))
            ),
            note=cand.note,
            rule=rule,
        )
        bad = t["unequal"] + t["rhs_err"]
        if t["equal"] == 0:
            row.outcome = (
                "rejected-no-equal" if not bad else "rejected-false"
            )
            row.note = row.note or "no measured equal site"
        elif bad:
            # mixed domain → guard synthesis; auto_cond measures its
            # own domain, which cannot reach attr-heavy carrier
            # corners (decay_vec dims, concat axes) — when it refuses
            # for "no evaluable site", fall back to a small axis/attr
            # guard bank evaluated on THIS env list
            ac = obs.auto_cond_object(
                rule, corpus_terms=corpus, synth_limit=args.synth_limit
            )
            if ac.object is None:
                guarded = _mint_axis_guard(cand, envs)
                if guarded is None:
                    row.outcome = "rejected-undeclarable"
                    row.note = ac.detail
                else:
                    row.rule = guarded
                    row.cond = guarded.cond
                    row.outcome = "guarded"
                    row.note = (
                        f"auto_cond refused ({ac.detail}); "
                        "axis-bank guard minted on typed envs"
                    )
            else:
                row.rule = ac.object.rule
                row.cond = ac.cond
                row.outcome = "guarded"
                row.note = ac.detail
        else:
            row.outcome = "unconditional"
        if row.outcome in ("unconditional", "guarded"):
            _certify(row, rules, t["first_lhs"], t["first_rhs"])
        rows.append(row)
    return {"atom": atom, "rows": rows}


def _report(res: dict) -> None:
    """Print one atom's synthesis table."""
    print(f"== {res['atom']} ==")
    for r in res["rows"]:
        cond_s = ""
        if r.cond not in (None, True):
            cond_s = f" cond={r.cond!r}"[:90]
        ds = f" dspec={r.dspec}" if r.dspec else ""
        print(
            f"  {r.name:<28} {r.outcome:<22} envs={r.envs:<4} "
            f"eq={r.equal:<3} bad={r.unequal:<3} real={r.real_sites:<3} "
            f"cert={r.verify:<10}{cond_s}{ds}"
        )
        if r.note:
            print(f"      {r.note[:100]}")
        if r.witness:
            print(f"      witness: {r.witness[:100]}")


def main(argv: list[str] | None = None) -> int:
    """Run the layer-3 loop over the carrier atoms."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--atom",
        action="append",
        choices=sorted(_ATOMS),
        help="restrict to one atom (repeatable)",
    )
    ap.add_argument("--max-candidates", type=int, default=160)
    ap.add_argument("--synth-limit", type=int, default=160)
    ap.add_argument(
        "--seconds", type=int, default=240, help="per-atom wall-clock"
    )
    ap.add_argument("--json", help="write the report as JSON")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    atoms = args.atom or sorted(_ATOMS)
    rules = _default_rules()
    out = []
    for atom in atoms:
        t0 = time.time()
        corpus = _corpus_with(atom)
        cands = _ATOMS[atom](args.max_candidates)
        res = _run_atom(atom, cands, corpus, rules, args)
        res["elapsed_s"] = round(time.time() - t0, 1)
        out.append(res)
        _report(res)
        print(f"  ({res['elapsed_s']}s)")
    if args.json:
        blob = []
        for res in out:
            blob.append(
                {
                    "atom": res["atom"],
                    "elapsed_s": res["elapsed_s"],
                    "rows": [
                        {
                            "name": r.name,
                            "outcome": r.outcome,
                            "envs": r.envs,
                            "equal": r.equal,
                            "bad": r.unequal,
                            "real_sites": r.real_sites,
                            "verify": r.verify,
                            "witness": r.witness,
                            "cond": repr(r.cond),
                            "dspec": r.dspec,
                            "note": r.note,
                            "law": (
                                law_to_data(r.rule)
                                if r.rule is not None
                                and r.outcome
                                in ("unconditional", "guarded")
                                else None
                            ),
                            "serializable": (
                                not missing_hooks(r.rule)
                                if r.rule is not None
                                else None
                            ),
                        }
                        for r in res["rows"]
                    ],
                }
            )
        from pathlib import Path

        Path(args.json).write_text(
            json.dumps(blob, indent=1, default=str)
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

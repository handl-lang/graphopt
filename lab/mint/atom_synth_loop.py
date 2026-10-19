"""ADR-0007 composed loop: measured need to certified admission.

measured need -> verified handler -> synthesized laws -> admission.

``lab/mint/atom_law_loop.py`` proved layer 3 (an atom's law set from
measured bindings) on the hand-built carrier atoms.  This driver
closes the whole ADR-0007 loop: it takes a *measured need* — the
op census the shipped laws cannot see (``exposure_screen``'s
``invisible_ops``, recomputed live over the intake corpus), a
``vocab`` derive-bank refusal, or an explicit ``--need`` — and runs
it through all four stages:

1. **signature** — the need becomes a candidate atom signature:
   arity and attrs read off the real corpus occurrences
   (``vocab.corpus_ops``), plus the candidate *spec spellings* the
   handler grammar will be measured against.  A census need with no
   signature builder refuses here — the vocabulary-gap list is a
   deliverable, not a failure.
2. **handler** — ``handler_synth.mint_handlers`` proposes grammar
   bodies and verifies them fp64 against the spelled spec (the
   layer-2 verify-or-refuse contract).  An op that already carries
   a torch binding (``cumsum`` & kin — atoms the *runtime* sees but
   the *laws* cannot) reports ``ambient`` — semantics by
   interpretation, the carrier-atom precedent — and the grammar
   probe still runs, so "could the machine have written it" is a
   measured verdict, not an assumption.  A minted body is bound
   into ``_IR_TO_TORCH`` so the oracle can evaluate the new atom.
3. **laws** — the layer-3 loop itself: candidate schemas ->
   measured envs (typed shape families x attr domains, the
   ``_synth_sites`` window, real corpus matches) -> tri-state
   oracle -> ``auto_cond`` / axis-guard fallback -> ``verify_law``.
   Reuses :mod:`atom_law_loop`'s machinery verbatim — the per-atom
   batteries and typed tables are registered into its module maps.
4. **admission** — the atom record: handler (minted record or
   ambient-binding note) plus every law row that cleared the
   gauntlet (``unconditional``/``guarded`` x ``verify_law`` x
   duplicate-vs-library x serializability).  The emitted fragment
   is a ``cellstore`` runtime fragment — the emergent-runtime
   thesis made data.
5. **pay** — the honest read: a bounded ``EGraph.run`` on each
   corpus term naming the need's ops, default vs default+certified
   rulesets — certified-law firings and the *incremental* cost
   delta.  Zero fires is a verdict, not a bug: the corpus's real
   sites may live in a law's guarded-out region.

Test subjects are the measured scan-family needs: ``cumsum``
(largest invisible FLOPs share in the corpus screen — RunningNorm,
LinearAttention, CumsumScan, SelectiveScan sites), ``cumprod``,
``logcumsumexp``, the ``SelectiveScan`` closed-form spec (the
fold/fold_emit mint path — corpus-verbatim, init-generalized and
total-form spellings), and ``pow`` (the vocab-flagged derive-bank
primitive — a different admission lane, kept for completeness).

Budgets are explicit (11 GB box, serial only): ``--budget`` caps
handler candidates, ``--max-candidates`` each law battery,
``--synth-limit`` the env window, ``--seconds`` the per-need
wall-clock.

Run::

    .venv/bin/python lab/mint/atom_synth_loop.py
    .venv/bin/python lab/mint/atom_synth_loop.py --need cumsum --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlopt_carriers  # noqa: F401 — registers carrier bindings
import torch
from mlopt_core.ir import Const, Op, TensorType, Var
from mlopt_core.laws.serialize import law_to_data, missing_hooks
from mlopt_discovery import vocab as vc
from mlopt_discovery.genkernel import _spec_metas
from mlopt_discovery.handler_synth import (
    handler_record,
    mint_handlers,
)
from mlopt_discovery.impact import _iter_subterms

from lab.mint import atom_law_loop as l3
from lab.screen.exposure_screen import _export, _workloads

Candidate = l3.Candidate


def _p(op: str, *args: Any, **attrs: Any) -> Op:
    return Op.make(op, *args, **attrs)


# ---------------------------------------------------------------------------
#  Need registry — the measured needs and their candidate signatures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """One candidate spelling the handler grammar is measured on."""

    label: str
    spec: Any  # the spec tuple (genkernel canonical form)
    domain: tuple  # list of {meta: shape} dicts
    note: str = ""


@dataclass(frozen=True)
class Need:
    """A measured vocabulary need and its candidate atom signature."""

    name: str
    kind: str  # "corpus-op" | "spec" | "derive-bank"
    atom: str  # the op the law stage synthesizes laws over
    specs: tuple = ()  # Spec spellings for the handler stage
    families: tuple = ()  # _TYPED_FAMILIES entries for the atom
    attr_domains: dict = field(default_factory=dict)
    candidates: Any = None  # fn(cap) -> list[Candidate]
    corpus_ops: tuple = ()  # ops marking a corpus term as relevant
    atom_args: tuple = ()  # Op arg order -> spec metavar names
    note: str = ""


# -- the scan-family spec spellings ------------------------------------------

_CP = ("cumprod", ("sigmoid", "A"), {"dim": 1})
_CPc = ("clamp_min", _CP, {"min": 1e-6})
_H0U = ("unsqueeze", "H0", {"dim": 1})

#: The SelectiveScan corpus site, verbatim:
#: ``cumprod(sigmoid(A)) · cumsum(B / clamp(cumprod(sigmoid(A))))`` — the emitted
#: selective-scan sequence, the way a real model spells it.
_SPEC_VERBATIM = (
    "mul",
    _CP,
    ("cumsum", ("div", "B", _CPc), {"dim": 1}),
)
#: init-generalized: an explicit carry ``H0`` makes the fold_emit
#: grammar's state slot spellable (3 metas).
_SPEC_INIT = (
    "mul",
    _CP,
    ("add", _H0U, ("cumsum", ("div", "B", _CPc), {"dim": 1})),
)
#: the same, unclamped — measures the stabilization's exactness cost.
_SPEC_INIT_OPEN = (
    "mul",
    _CP,
    ("add", _H0U, ("cumsum", ("div", "B", _CP), {"dim": 1})),
)
#: total (closed) form — the `fold` family's shape: the final state
#: ``h0·∏a + Σ_t b_t·∏a/∏≤a``, no sigmoid baked in (the general
#: affine scan; the corpus site is its sigmoid-gated emitted instance).
#: ``prod`` drops the scanned axis, so the ratio needs the
#: ``unsqueeze`` to broadcast back along it.
_SPEC_TOTAL = (
    "add",
    ("mul", "H0", ("prod", "A", {"dim": 1})),
    (
        "sum",
        (
            "mul",
            "B",
            (
                "div",
                ("unsqueeze", ("prod", "A", {"dim": 1}), {"dim": 1}),
                ("cumprod", "A", {"dim": 1}),
            ),
        ),
        {"dim": 1},
    ),
)
#: cumsum-with-init, the fold_emit reading of the cumsum need.
_SPEC_CUMSUM_INIT = (
    "add",
    _H0U,
    ("cumsum", "X", {"dim": 1}),
)

_SCAN_DOM = ({"A": (2, 8, 16), "B": (2, 8, 16), "H0": (2, 16)},)
_SCAN_EMIT_DOM = ({"A": (2, 8, 16), "B": (2, 8, 16)},)
_CUMSUM_INIT_DOM = ({"X": (2, 8, 16), "H0": (2, 16)},)


# -- law batteries (the layer-3 hypothesis sets, measured not assumed) --------


def _cs(t: Any, d: str = "D") -> Any:
    return _p("cumsum", t, dim=d)


def _cumsum_candidates(cap: int) -> list[Candidate]:
    """``cumsum`` — the linear prefix map: linearity, absorb, cat."""
    cands = [
        Candidate(
            "cum.add",
            _cs(_p("add", "x", "y")),
            _p("add", _cs("x"), _cs("y")),
            note="cumsum is additive — over same-shape operands "
            "(broadcast-into-scan-axis is the false region)",
        ),
        Candidate(
            "cum.add_same",
            _cs(_p("add", "x", "y")),
            _p("add", _cs("x"), _cs("y")),
            cond=("shape-compat", "x", "y"),
            note="spelled guard: no broadcast into the scanned axis",
        ),
        Candidate(
            "cum.sub",
            _cs(_p("sub", "x", "y")),
            _p("sub", _cs("x"), _cs("y")),
        ),
        Candidate(
            "cum.sub_same",
            _cs(_p("sub", "x", "y")),
            _p("sub", _cs("x"), _cs("y")),
            cond=("shape-compat", "x", "y"),
        ),
        Candidate(
            "cum.neg",
            _cs(_p("neg", "x")),
            _p("neg", _cs("x")),
        ),
        Candidate(
            "cum.scale",
            _cs(_p("mul", "x", "c")),
            _p("mul", _cs("x"), "c"),
            note="scalar c; the vector-c region should guard or die",
        ),
        Candidate(
            "cum.scale_same",
            _cs(_p("mul", "x", "c")),
            _p("mul", _cs("x"), "c"),
            cond=("scalar", "c"),
        ),
        Candidate(
            "cum.cat",
            _cs(_p("concat", "x1", "x2", dim="D")),
            _p(
                "concat",
                _cs("x1"),
                _p(
                    "add",
                    _cs("x2"),
                    _p("sum", "x1", dim="D", keepdim=True),
                ),
                dim="D",
            ),
            note="prefix after the seam = total of x1 + cumsum(x2)",
        ),
        Candidate(
            "cum.add_const",
            _cs(_p("add", "x", "c")),
            _p(
                "add",
                _cs("x"),
                _p("mul", _cs(_p("div", "x", "x")), "c"),
            ),
            cond=("scalar", "c"),
            note="the real corpus shape (cumsum(add(elu k, 1.0))): "
            "x + c accumulates c*(t+1) — true, but div(x,x) "
            "injects nan at x==0 — the ones-spelling trap",
        ),
        Candidate(
            # the false twin — the scan is linear, not multiplicative
            "cum.mul_wrong",
            _cs(_p("mul", "x", "y")),
            _p("mul", _cs("x"), _cs("y")),
            note="deliberately false — cumsum does not distribute "
            "over products",
        ),
    ]
    return cands[: cap or None]


def _cumprod_candidates(cap: int) -> list[Candidate]:
    """``cumprod`` — the prefix homomorphism: mul/div, exp-bridge."""
    cp = lambda t: _p("cumprod", t, dim="D")  # noqa: E731
    cands = [
        Candidate(
            "cp.mul",
            cp(_p("mul", "x", "y")),
            _p("mul", cp("x"), cp("y")),
            note="cumprod IS multiplicative — the cumsum twin's "
            "honest mirror",
        ),
        Candidate(
            "cp.mul_same",
            cp(_p("mul", "x", "y")),
            _p("mul", cp("x"), cp("y")),
            cond=("shape-compat", "x", "y"),
        ),
        Candidate(
            "cp.div",
            cp(_p("div", "x", "y")),
            _p("div", cp("x"), cp("y")),
        ),
        Candidate(
            "cp.div_same",
            cp(_p("div", "x", "y")),
            _p("div", cp("x"), cp("y")),
            cond=("shape-compat", "x", "y"),
        ),
        Candidate(
            "cp.exp",
            cp(_p("exp", "x")),
            _p("exp", _cs("x")),
            note="cross-atom: prod of exps = exp of cumsum",
        ),
        Candidate(
            "cp.add_wrong",
            cp(_p("add", "x", "y")),
            _p("add", cp("x"), cp("y")),
            note="deliberately false — products don't distribute "
            "over sums",
        ),
    ]
    return cands[: cap or None]


def _logcumsumexp_candidates(cap: int) -> list[Candidate]:
    """``logcumsumexp`` — the lse scan: scalar shift only."""
    le = lambda t: _p("logcumsumexp", t, dim="D")  # noqa: E731
    cands = [
        Candidate(
            "lc.shift",
            le(_p("add", "x", "c")),
            _p("add", le("x"), "c"),
            note="logaddexp(a, x+c) = logaddexp(a,x)+c for const c",
        ),
        Candidate(
            "lc.shift_same",
            le(_p("add", "x", "c")),
            _p("add", le("x"), "c"),
            cond=("scalar", "c"),
        ),
        Candidate(
            "lc.mul_wrong",
            le(_p("mul", "x", "c")),
            _p("mul", le("x"), "c"),
            note="deliberately false — lse doesn't scale",
        ),
    ]
    return cands[: cap or None]


def _sf(t1: Any, t2: Any, t3: Any) -> Any:
    return _p("scan_fold", t1, t2, t3)


def _scan_fold_candidates(cap: int) -> list[Candidate]:
    """``scan_fold(a,b,h0) = h0*prod(a) + sum_t b_t*prod(a)/cumprod(a)_t`` — affine fold.

    The minted handler's axis is baked ``-2`` (the second-to-last of
    the broadcast operand space — the corpus's ``dim=1``), so the
    spelled forms carry the literal, not an attr metavar.
    """
    prod = lambda t: _p("prod", t, dim=-2)  # noqa: E731
    cum = lambda t: _p("cumprod", t, dim=-2)  # noqa: E731
    ratio = lambda t: _p(  # noqa: E731
        "div", _p("unsqueeze", prod(t), dim=-2), cum(t)
    )
    closed = _p(
        "add",
        _p("mul", "h", prod("a")),
        _p("sum", _p("mul", "b", ratio("a")), dim=-2),
    )
    cands = [
        Candidate(
            "sf.unfold",
            _sf("a", "b", "h"),
            closed,
            note="the mint-verified semantics spelled out",
        ),
        Candidate("sf.refold", closed, _sf("a", "b", "h")),
        Candidate(
            "sf.corpus_refold",
            _p(
                "mul",
                _p("cumprod", _p("sigmoid", "a"), dim=1),
                _p(
                    "cumsum",
                    _p(
                        "div",
                        "b",
                        _p(
                            "clamp_min",
                            _p("cumprod", _p("sigmoid", "a"), dim=1),
                            min=1e-6,
                        ),
                    ),
                    dim=1,
                ),
            ),
            _sf("a", "b", Const(0.0)),
            cond=(
                "and",
                ("rank", "a", "==", 3),
                ("shape-compat", "a", "b"),
            ),
            note="the SelectiveScan site -> the minted emitted-"
            "sequence atom: scan_fold(a, b, 0) IS the corpus's "
            "per-step output (the atom's body applies the sigmoid "
            "gate itself, so the RHS passes the raw a).  Two exact "
            "preconditions: rank 3 (the corpus spells dim=1, the "
            "atom bakes axis=-2 — equal only at rank 3) and "
            "shape-compat a/b (the atom broadcasts a,b then folds; "
            "the corpus scans then broadcasts — the orders agree "
            "only when nothing broadcasts into the scan axis).  The "
            "law still fails on the real site: the corpus's "
            "clamp_min(.,1e-6) stabilization is active there (the "
            "gate cumprod underflows below 1e-6), and the minted "
            "atom is the ideal unclamped recurrence — a semantics "
            "gap the affine grammar cannot close",
        ),
        Candidate(
            "sf.b0",
            _sf("a", _p("sub", "x", "x"), "h"),
            _p("mul", "h", prod("a")),
            note="zero input stream: pure state carry",
        ),
        Candidate(
            "sf.split",
            _sf("a", _p("add", "b1", "b2"), "h"),
            _p(
                "add",
                _sf("a", "b1", _p("sub", "h", "h")),
                _sf("a", "b2", "h"),
            ),
            note="joint linearity in (b, h0)",
        ),
        Candidate(
            "sf.split_same",
            _sf("a", _p("add", "b1", "b2"), "h"),
            _p(
                "add",
                _sf("a", "b1", _p("sub", "h", "h")),
                _sf("a", "b2", "h"),
            ),
            cond=("shape-compat", "b1", "b2"),
        ),
        Candidate(
            "sf.scale",
            _sf("a", _p("mul", "b", "c"), _p("mul", "h", "c")),
            _p("mul", _sf("a", "b", "h"), "c"),
            cond=("scalar", "c"),
            note="scalar c scales b and h0 together",
        ),
        Candidate(
            "sf.scale_b_wrong",
            _sf("a", _p("mul", "b", "c"), "h"),
            _p("mul", _sf("a", "b", "h"), "c"),
            note="deliberately false — scaling b alone leaves the "
            "h0·prod term unscaled",
        ),
        Candidate(
            "sf.aadd_wrong",
            _sf(_p("add", "a1", "a2"), "b", "h"),
            _p("add", _sf("a1", "b", "h"), _sf("a2", "b", "h")),
            note="deliberately false — the gate is multiplicative, "
            "not additive",
        ),
    ]
    return cands[: cap or None]


#: Typed shape families per atom — rank-3 (B,T,C) sites with the scan
#: on the middle axis (the corpus shape), plus rank-2 and scalar/vec
#: corners for the absorb candidates.
_FAMILIES: dict[str, tuple] = {
    "cumsum": (
        {"x": (4, 8), "y": (4, 8), "c": ()},
        {"x": (2, 6, 4), "y": (2, 6, 4), "c": ()},
        {"x": (6,), "y": (6,)},
        # the vector-c corner — cum.scale's false region
        {"x": (4, 8), "y": (4, 8), "c": (4, 8)},
        # the concat corner — cum.cat's typed site
        {"x1": (2, 4), "x2": (3, 4), "x": (5, 4), "y": (5, 4)},
        {"x1": (5, 4), "x2": (2, 4)},
    ),
    "cumprod": (
        {"x": (4, 8), "y": (4, 8)},
        {"x": (2, 6, 4), "y": (2, 6, 4)},
        {"x": (6,), "y": (6,)},
    ),
    "logcumsumexp": (
        {"x": (4, 8), "c": ()},
        {"x": (2, 6, 4), "c": ()},
        {"x": (4, 8), "c": (4, 8)},
    ),
    "scan_fold": (
        {
            "a": (2, 6, 4),
            "b": (2, 6, 4),
            "h": (2, 4),
            "x": (2, 6, 4),
            "c": (),
            "b1": (2, 6, 4),
            "b2": (2, 6, 4),
            "a1": (2, 6, 4),
            "a2": (2, 6, 4),
        },
        {"a": (4, 8), "b": (4, 8), "h": (8,), "x": (4, 8), "c": ()},
        {
            "a": (2, 6, 4),
            "b": (2, 6, 4),
            "b1": (2, 6, 4),
            "b2": (2, 6, 4),
            "h": (2, 4),
        },
    ),
}

_ATTR_DOMAINS: dict[str, dict] = {
    "cumsum": {"D": (0, 1, -1, -2)},
    "cumprod": {"D": (0, 1, -1, -2)},
    "logcumsumexp": {"D": (0, 1, -1, -2)},
    "scan_fold": {"D": (1, -2)},
}

_NEEDS: dict[str, Need] = {
    "cumsum": Need(
        "cumsum",
        "corpus-op",
        "cumsum",
        specs=(
            Spec(
                "verbatim",
                ("cumsum", "X", {"dim": 1}),
                ({"X": (2, 8, 16)},),
                note="the literal atom — 1 meta",
            ),
            Spec(
                "init-generalized",
                _SPEC_CUMSUM_INIT,
                _CUMSUM_INIT_DOM,
                note="H0 + cumsum(X): the fold_emit reading; "
                "2 metas, still under the >=3 family gate",
            ),
        ),
        families=_FAMILIES["cumsum"],
        attr_domains=_ATTR_DOMAINS["cumsum"],
        candidates=_cumsum_candidates,
        corpus_ops=("cumsum",),
        note="largest invisible-FLOPs scan op in the corpus screen "
        "(RunningNorm x2, LinearAttention x2, CumsumScan, "
        "SelectiveScan)",
    ),
    "cumprod": Need(
        "cumprod",
        "corpus-op",
        "cumprod",
        specs=(
            Spec(
                "verbatim",
                ("cumprod", "X", {"dim": 1}),
                ({"X": (2, 8, 16)},),
            ),
        ),
        families=_FAMILIES["cumprod"],
        attr_domains=_ATTR_DOMAINS["cumprod"],
        candidates=_cumprod_candidates,
        corpus_ops=("cumprod",),
        note="CumProdGate / SelectiveScan / CumulativeGateRNN sites",
    ),
    "logcumsumexp": Need(
        "logcumsumexp",
        "corpus-op",
        "logcumsumexp",
        specs=(
            Spec(
                "verbatim",
                ("logcumsumexp", "X", {"dim": 1}),
                ({"X": (2, 8, 16)},),
            ),
        ),
        families=_FAMILIES["logcumsumexp"],
        attr_domains=_ATTR_DOMAINS["logcumsumexp"],
        candidates=_logcumsumexp_candidates,
        corpus_ops=("logcumsumexp",),
        note="CumsumScan's second scan — the non-affine one",
    ),
    "scan_fold": Need(
        "scan_fold",
        "spec",
        "scan_fold",
        specs=(
            Spec(
                "verbatim",
                _SPEC_VERBATIM,
                _SCAN_EMIT_DOM,
                note="the SelectiveScan corpus site as spelled "
                "(2 metas — the emitted sequence, clamped)",
            ),
            Spec(
                "init-clamped",
                _SPEC_INIT,
                _SCAN_DOM,
                note="explicit carry H0 makes the state slot "
                "spellable; corpus clamp kept",
            ),
            Spec(
                "init-unclamped",
                _SPEC_INIT_OPEN,
                _SCAN_DOM,
                note="same minus clamp_min — the pure semantics",
            ),
            Spec(
                "total",
                _SPEC_TOTAL,
                _SCAN_DOM,
                note="the closed (final-state) form — the fold "
                "family's shape, sigmoid lifted out",
            ),
        ),
        families=_FAMILIES["scan_fold"],
        attr_domains=_ATTR_DOMAINS["scan_fold"],
        candidates=_scan_fold_candidates,
        corpus_ops=("cumprod", "cumsum"),
        atom_args=("A", "B", "H0"),
        note="the SelectiveScan site: mul(cumprod(sigmoid(A)), "
        "cumsum(B/clamp(cumprod(sigmoid(A))))) — the emitted selective-scan "
        "closed form",
    ),
    "pow": Need(
        "pow",
        "derive-bank",
        "pow",
        note="vocab-flagged primitive: dspec_synth's bank cannot "
        "spell int**int derives (the 2**rank refusal)",
    ),
}


# ---------------------------------------------------------------------------
#  Stage 0 — measured needs (the live census)
# ---------------------------------------------------------------------------


def _census_rows(which: str) -> tuple[list, list]:
    """Export the bounded corpus; return (rows, cases) for need ranking.

    The same measurement ``exposure_screen`` ships — an op is a *need*
    when no law's LHS names it (the FLOPs-weighted ``invisible_ops``
    census).  Bounded to the *which* registry (``intake`` by default:
    the scan-family needs all live there and the set is small).
    """
    from mlopt_core.cost.basic import _flops_of
    from exposure_screen import law_surface

    _rules, lhs_ops, _rhs = law_surface()
    cases = []
    for source, wl in _workloads(which):
        _row, case = _export(wl, source)
        if case is not None:
            cases.append(case)
    inv: dict[str, float] = {}
    sites: dict[str, int] = {}
    models: dict[str, set] = {}
    memo: dict = {}
    for case in cases:
        for s in _iter_subterms(case.term):
            if not isinstance(s, Op) or s.op in lhs_ops:
                continue
            try:
                f = _flops_of(s, memo)
            except Exception:
                f = 0.0
            inv[s.op] = inv.get(s.op, 0.0) + f
            sites[s.op] = sites.get(s.op, 0) + 1
            models.setdefault(s.op, set()).add(case.name)
    rows = [
        {
            "op": op,
            "flops": inv[op],
            "sites": sites[op],
            "models": sorted(models[op]),
        }
        for op in sorted(inv, key=lambda o: -inv[o])
    ]
    return rows, cases


# ---------------------------------------------------------------------------
#  Stage 1 — signature (occurrences -> arity/attrs -> spellings)
# ---------------------------------------------------------------------------


def _occurrences(op: str, cases: list) -> list[dict]:
    """Read the op's real signature off its corpus occurrences."""
    out = []
    for case in cases:
        for s in _iter_subterms(case.term):
            if isinstance(s, Op) and s.op == op:
                out.append(
                    {
                        "model": case.name,
                        "attrs": dict(s.attrs),
                        "arity": len(s.args),
                    }
                )
    return out


# ---------------------------------------------------------------------------
#  Stage 2 — handler (mint_handlers / ambient-binding / grammar probe)
# ---------------------------------------------------------------------------


def _handler_stage(need: Need, args: argparse.Namespace) -> dict:
    """Run the layer-2 verify-or-refuse over the need's spellings.

    Returns the handler row: ``status`` is ``minted`` (a verified
    grammar body, bound into the ambient table), ``ambient`` (the op
    already carries a torch binding — the carrier-atom precedent),
    or ``refused-*`` (a measured grammar gap).  ``probe`` records the
    per-spelling verdicts — minted bodies and measured refusals alike.
    """
    import mlopt_torch.torch_bridge as tb

    ambient = need.atom in tb._IR_TO_TORCH
    row: dict[str, Any] = {
        "status": "ambient" if ambient else "pending",
        "ambient_binding": ambient,
        "spellings": [],
        "body": None,
        "record": None,
    }
    for sp in need.specs:
        sub: dict[str, Any] = {"label": sp.label, "note": sp.note}
        items = [{"spec": sp.spec, "domain": list(sp.domain)}]
        handlers, report = mint_handlers(
            items,
            families=args.families,
            seed=args.seed,
            budget=args.budget,
        )
        rep = report[0]
        sub["verdict"] = rep["verdict"]
        if rep["verdict"] == "minted":
            tag = rep["tag"]
            body = handlers[tag]["handler_body"]
            sub.update(
                {
                    "family": rep["family"],
                    "tried": rep["tried"],
                    "max_err": rep["max_err"],
                    "body": body,
                }
            )
            if row["body"] is None:
                row["body"] = body
                row["record"] = handler_record(tag, handlers[tag])
                row["status"] = "minted"
                row["minted_via"] = sp.label
        else:
            sub["tried"] = rep.get("tried")
            sub["best_err"] = rep.get("best_err")
            sub["refusal"] = rep.get("note")
        row["spellings"].append(sub)
    if row["status"] == "pending":
        row["status"] = "refused-grammar"
    return row


def _bind_minted(need: Need, handler_row: dict) -> Any:
    """Register a minted body as ``need.atom``'s torch binding.

    The op's positional arg order is the spec's metavar order — the
    descriptor interpreter reads ``env[meta]``, so the binding zips the
    Op's args against the spec metas.  Registered into ``_IR_TO_TORCH``:
    the oracle, the site-outcome probe and the sink all see the atom —
    "the sink's executor table is the semantics" (ADR 0007).
    """
    import mlopt_torch.torch_bridge as tb

    body = handler_row["body"]
    spec = next(
        s for s in need.specs if s.label == handler_row["minted_via"]
    ).spec
    metas = need.atom_args or tuple(_spec_metas(spec))
    bindings = tb._IR_TO_TORCH

    def fn(*args: Any, **_attrs: Any) -> Any:
        from mlopt_discovery.handler_synth import eval_body

        env = dict(zip(metas, args, strict=True))
        return eval_body(body, env, bindings)

    bindings[need.atom] = fn
    return fn


# ---------------------------------------------------------------------------
#  Stage 3 — the law loop (atom_law_loop's machinery, new tables)
# ---------------------------------------------------------------------------


def _corpus_for(need: Need, cases: list) -> list:
    """Corpus terms relevant to this need's law measurement."""
    if not need.corpus_ops:
        return []
    out = []
    for case in cases:
        ops = {
            s.op for s in _iter_subterms(case.term) if isinstance(s, Op)
        }
        if ops & set(need.corpus_ops):
            out.append(case.term)
    return out


def _law_stage(
    need: Need, corpus: list, rules: list, args: argparse.Namespace
) -> dict:
    """Run the layer-3 loop over the need's candidate battery."""
    if need.candidates is None:
        return {"atom": need.atom, "rows": [], "skipped": "no battery"}
    l3._TYPED_FAMILIES[need.atom] = list(need.families)
    l3._ATTR_DOMAINS[need.atom] = dict(need.attr_domains)
    cands = need.candidates(args.max_candidates)
    return l3._run_atom(need.atom, cands, corpus, rules, args)


# ---------------------------------------------------------------------------
#  Stage 4 — admission (certified laws + handler record + fragment)
# ---------------------------------------------------------------------------


def _admit(
    need: Need,
    handler_row: dict,
    law_res: dict,
    args: argparse.Namespace,
) -> dict:
    """Assemble the atom's admission record from the stage outputs."""
    rows = law_res.get("rows", [])
    certified = [
        r
        for r in rows
        if r.outcome in ("unconditional", "guarded")
        and r.verify in ("primitive", "derivable")
    ]
    serial = {}
    for r in certified:
        serial[r.name] = (
            not missing_hooks(r.rule) if r.rule is not None else False
        )
    admitted = (
        handler_row["status"] in ("ambient", "minted")
        and len(certified) > 0
    )
    laws_data = [
        {
            "name": r.name,
            "outcome": r.outcome,
            "verify": r.verify,
            "serializable": serial[r.name],
            "law": law_to_data(r.rule),
        }
        for r in certified
        if serial[r.name]
    ]
    rec: dict[str, Any] = {
        "atom": need.atom,
        "admitted": admitted,
        "handler": handler_row["status"],
        "handler_record": handler_row["record"],
        "laws_certified": len(certified),
        "laws": laws_data,
    }
    if need.kind == "spec" and handler_row["status"] == "minted":
        rec["mint"] = {
            "via": handler_row["minted_via"],
            "body": handler_row["body"],
        }
    rec["fragment"] = _fragment(need, handler_row, certified)
    return rec


def _fragment(need: Need, handler_row: dict, certified: list) -> dict:
    """Assemble the atom's {op, handler, laws} set as cell data.

    The emergent-runtime thesis as a fragment record: one handle cell
    for the atom's binding (the minted handler record, or the ambient
    torch-binding name) plus the certified law rows — the same shape
    ``mlopt-carriers`` grew by hand.  Emitted for every admitted
    need; the store machinery that merges it is ``cellstore``.
    """
    from mlopt_core.cells import Provenance
    from mlopt_discovery.cellstore import (
        handlers_fragment,
        laws_fragment,
    )

    if handler_row["status"] == "minted":
        handle = dict(handler_row["record"] or {})
        prov = Provenance(origin="minted")
    else:
        handle = {
            "pattern": need.atom,
            "kernel": f"torch:{need.atom}",
            "args": (),
        }
        prov = Provenance(origin="shipped")
    frag_h = handlers_fragment(
        need.atom, {need.atom: handle}, provenance=prov
    )
    frag_l = laws_fragment(
        f"{need.atom}.laws", [r.rule for r in certified]
    )
    return {
        "handles": len(frag_h.cells),
        "laws": len(frag_l.cells),
        "handle_names": [c.name for c in frag_h.cells],
        "law_names": [c.name for c in frag_l.cells],
    }


# ---------------------------------------------------------------------------
#  The derive-bank lane (pow) — a different admission shape
# ---------------------------------------------------------------------------


def _pow_rank_hook(bound: dict) -> dict | None:
    """Return ``{"$attr:M": 2 ** rank(x)}`` — the refused derive."""
    sh = getattr(getattr(bound.get("x"), "typ", None), "shape", None)
    return None if sh is None else {"$attr:M": 2 ** len(sh)}


def _rank_domain() -> list[dict]:
    """Binding envs for the pow-rank hook (per the refusal tests)."""
    return [
        {
            "x": Var(f"x{i}", TensorType(tuple(sh))),
            "$attr:a": i,
        }
        for i, sh in enumerate(
            [(4,), (4, 4), (2, 4, 4), (2, 2, 2, 2), (2, 2, 2, 2, 3)]
        )
    ]


def _derive_bank_lane(rules: list, args: argparse.Namespace) -> dict:
    """Run the vocab lane: refused derives -> missing bank op -> admission.

    ``detect_need`` runs the canonical ``2**rank`` hook (the refusal the
    candidate registry was pinned by), ``propose_atom`` for the minted
    record, and ``scan_needs`` over the shipped rules for the honest
    corpus count — zero real blocked rules is itself the finding.
    """
    need = vc.detect_need(_pow_rank_hook, _rank_domain())
    row: dict[str, Any] = {"lane": "derive-bank", "need": None}
    if need is not None:
        row["need"] = need.to_data()
        row["proposal"] = vc.propose_atom(need)
        row["admitted"] = bool(need.rules_blocked)
    if args.scan_rules:
        row["corpus_scan"] = [
            n.to_data()
            for n in vc.scan_needs(
                rules,
                synth_limit=args.synth_limit,
                deadline=args.seconds,
                max_rules=args.scan_rules,
            )
        ]
        row["scan_rules_seen"] = args.scan_rules
    else:
        row["corpus_scan"] = "skipped (--scan-rules 0)"
    return row


# ---------------------------------------------------------------------------
#  Stage 5 — the honest pay probe (fires on the real corpus, bounded)
# ---------------------------------------------------------------------------


def _fire_stage(
    need: Need,
    cases: list,
    rules: list,
    certified: list,
    args: argparse.Namespace,
) -> dict:
    """Saturate the corpus terms under default + certified laws.

    The admission record says the laws are true; this stage says
    whether they *pay*: a bounded ``EGraph.run`` per corpus term that
    names the need's ops, counting firings of the certified names and
    the extracted cost delta — the same measurement
    ``exposure_screen --fire`` ships.  Zero fires is an honest
    verdict: the corpus's real sites may live entirely in a
    certified law's guarded-out region.
    """
    if not certified:
        return {"fires": 0, "terms": 0, "delta": 0.0, "skipped": True}
    from mlopt_core.cost import backend_cost, dag_cost
    from mlopt_core.cost.executor import executor_cost_for
    from mlopt_core.egraph import EGraph
    from mlopt_torch.adapters import TorchSink

    cost_fn = backend_cost(
        executor_cost_for(lowering="generic"), TorchSink().supported_ops
    )
    names = {r.name for r in certified}
    fires = 0
    delta = 0.0
    terms = 0
    for case in cases:
        ops = {
            s.op for s in _iter_subterms(case.term) if isinstance(s, Op)
        }
        if not (ops & set(need.corpus_ops)):
            continue
        terms += 1
        bests: list = []
        for ruleset in (rules, [*rules, *certified]):
            eg = EGraph()
            root = eg.add_term(case.term)
            eg.run(ruleset, root, max_iterations=6, max_nodes=60_000)
            if ruleset is not rules:
                hits = {
                    n: c
                    for n, c in eg.rule_fires.items()
                    if c and n in names
                }
                fires += sum(hits.values())
            best = eg.extract_best(root, cost_fn)
            bests.append(
                dag_cost(best, cost_fn) if best is not None else None
            )
        if bests[0] is not None and bests[1] is not None:
            delta += bests[0] - bests[1]  # incremental gain only
    return {"fires": fires, "terms": terms, "delta": round(delta, 4)}


# ---------------------------------------------------------------------------
#  Driver + report
# ---------------------------------------------------------------------------


def _need_line(rec: dict) -> str:
    """One funnel line for a need."""
    h = rec["handler"]
    laws = rec["laws"]
    cert = rec["admission"]
    rows = laws.get("rows", [])
    n_unc = sum(1 for r in rows if r.outcome == "unconditional")
    n_gua = sum(1 for r in rows if r.outcome == "guarded")
    n_rej = sum(1 for r in rows if r.outcome.startswith("rejected"))
    n_ref = sum(
        1 for r in rows if r.outcome.startswith(("refused", "skipped"))
    )
    return (
        f"  {rec['need']:<14} sig={rec['signature']:<10} "
        f"handler={h['status']:<16} laws: {n_unc} unc {n_gua} gua "
        f"{n_rej} rej {n_ref} ref  ->  "
        f"{'ADMITTED' if cert['admitted'] else 'not admitted'} "
        f"({cert['laws_certified']} laws) "
        f"fire={rec.get('fire', {}).get('fires', '-')}"
    )


def _run_need(
    need: Need, cases: list, rules: list, args: argparse.Namespace
) -> dict:
    """Run one need through all four stages; return the record."""
    rec: dict[str, Any] = {"need": need.name, "kind": need.kind}
    # stage 1 — signature
    occ = _occurrences(need.atom, cases) if need.corpus_ops else []
    rec["signature"] = (
        "ok"
        if (need.specs or need.candidates or need.kind != "spec")
        else "refused-no-signature"
    )
    rec["occurrences"] = occ
    if rec["signature"].startswith("refused"):
        rec["handler"] = {"status": "n/a"}
        rec["laws"] = {"rows": []}
        rec["admission"] = {"admitted": False, "laws_certified": 0}
        return rec
    # stage 2 — handler
    handler_row = _handler_stage(need, args)
    rec["handler"] = handler_row
    if handler_row["status"] == "minted":
        _bind_minted(need, handler_row)
    # stage 3 — laws (ambient, minted, or refused handler alike: the
    # grammar verdict and the law measurement are independent)
    corpus = _corpus_for(need, cases)
    law_res = _law_stage(need, corpus, rules, args)
    rec["laws"] = law_res
    # stage 4 — admission
    rec["admission"] = _admit(need, handler_row, law_res, args)
    rec["corpus_terms"] = len(corpus)
    # stage 5 — pay probe (bounded saturation on the real terms)
    certified = [
        r.rule
        for r in law_res.get("rows", [])
        if r.outcome in ("unconditional", "guarded")
        and r.verify in ("primitive", "derivable")
    ]
    rec["fire"] = (
        _fire_stage(need, cases, rules, certified, args)
        if args.fire
        else {"skipped": True}
    )
    return rec


def _print_need(rec: dict) -> None:
    """Print one need's stage detail under its funnel line."""
    for sp in rec["handler"].get("spellings", []):
        if sp["verdict"] == "minted":
            print(
                f"      spec[{sp['label']}] MINTED "
                f"family={sp['family']} tried={sp['tried']} "
                f"err={sp['max_err']:.3g} body={sp['body']}"
            )
        else:
            print(
                f"      spec[{sp['label']}] refused tried={sp['tried']} "
                f"{sp.get('refusal', '')}"
            )
    for r in rec["laws"].get("rows", []):
        cond = r.cond
        if cond is None and r.rule is not None:
            cond = r.rule.cond
        cond_s = ""
        if cond not in (None, True):
            cond_s = f" cond={cond!r}"[:80]
        print(
            f"      law {r.name:<22} {r.outcome:<22} "
            f"envs={r.envs:<4} eq={r.equal:<3} bad={r.unequal:<3} "
            f"real={r.real_sites:<3} cert={r.verify}{cond_s}"
        )
        if r.note:
            print(f"          {r.note[:100]}")


def main(argv: list[str] | None = None) -> int:
    """Run the composed loop over the measured needs."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--need",
        action="append",
        choices=sorted(_NEEDS),
        help="restrict to one need (repeatable)",
    )
    ap.add_argument(
        "--set",
        choices=("zoo", "intake", "models", "all"),
        default="intake",
        help="corpus registry for needs + real sites",
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=512,
        help="handler candidate cap per spec",
    )
    ap.add_argument("--max-candidates", type=int, default=160)
    ap.add_argument("--synth-limit", type=int, default=160)
    ap.add_argument("--seconds", type=int, default=240)
    ap.add_argument(
        "--scan-rules",
        type=int,
        default=0,
        help="cap the derive-bank lane's shipped-rule scan (0=off; "
        "the full scan was measured once — zero blocked rules)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--fire",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the bounded pay probe on corpus terms",
    )
    ap.add_argument(
        "--families",
        default=None,
        help="handler grammar families (comma list)",
    )
    ap.add_argument("--json", help="write the report as JSON")
    args = ap.parse_args(argv)
    args.families = (
        tuple(args.families.split(",")) if args.families else None
    )

    torch.manual_seed(args.seed)
    needs = [_NEEDS[n] for n in (args.need or sorted(_NEEDS))]

    t0 = time.time()
    print("== atom_synth_loop — need -> handler -> laws -> admit ==")
    census, cases = _census_rows(args.set)
    scan_needs = [
        r for r in census if r["op"] in _NEEDS or r["op"] == "pow"
    ]
    gap = [r for r in census if r["op"] not in _NEEDS]
    print(
        f"   corpus: {len(cases)} terms exported; "
        f"{len(census)} law-invisible ops, "
        f"{len(gap)} with no signature builder "
        f"({time.time() - t0:.0f}s)"
    )
    for r in scan_needs:
        print(
            f"   need {r['op']:<14} flops={r['flops']:<10.4g} "
            f"sites={r['sites']} models={','.join(r['models'])}"
        )

    rules = l3._default_rules()
    out: list[dict] = []
    for need in needs:
        t1 = time.time()
        if need.kind == "derive-bank":
            rec = {
                "need": need.name,
                "kind": need.kind,
                "derive_lane": _derive_bank_lane(rules, args),
            }
            rec["handler"] = {"status": "n/a"}
            rec["laws"] = {"rows": []}
            rec["admission"] = {
                "admitted": rec["derive_lane"].get("admitted", False),
                "laws_certified": 0,
            }
            rec["signature"] = "ok"
        else:
            rec = _run_need(need, cases, rules, args)
        rec["elapsed_s"] = round(time.time() - t1, 1)
        out.append(rec)
        print(_need_line(rec))
        _print_need(rec)
        print(f"      ({rec['elapsed_s']}s)")

    gap_names = [r["op"] for r in gap[:20]]
    print()
    print(
        f"== vocabulary-gap list ({len(gap)} law-invisible ops with "
        f"no signature): {', '.join(gap_names)}"
        f"{' …' if len(gap) > 20 else ''}"
    )
    if args.json:
        blob = []
        for rec in out:
            laws = rec["laws"]
            blob.append(
                {
                    "need": rec["need"],
                    "kind": rec["kind"],
                    "signature": rec.get("signature"),
                    "occurrences": rec.get("occurrences"),
                    "handler": {
                        k: v
                        for k, v in rec["handler"].items()
                        if k != "record"
                    },
                    "handler_record": rec["handler"].get("record"),
                    "laws": [
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
                            "note": r.note,
                        }
                        for r in laws.get("rows", [])
                    ],
                    "corpus_terms": rec.get("corpus_terms"),
                    "fire": rec.get("fire"),
                    "admission": rec["admission"],
                    "derive_lane": rec.get("derive_lane"),
                    "elapsed_s": rec["elapsed_s"],
                }
            )
        Path(args.json).write_text(
            json.dumps(
                {"needs": blob, "vocab_gap": gap_names},
                indent=1,
                default=str,
            )
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

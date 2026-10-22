"""The bilinear game — tensor-rank reduction as a playable domain.

Q2 of the research plan — the falsifiable "can it find new
algorithms" test.  The board is the ``<2,2,2>`` matrix
multiplication tensor's *decomposition space*: a state is a list
of rank-1 terms ``(u_k, v_k, w_k)`` — a bilinear multiplication
scheme — plus the residual ``R = T - Σ_k u_k ⊗ v_k ⊗ w_k``.
``R = 0`` means the scheme computes ``C = A·B`` correctly; the
cost is the number of terms (the tensor rank of the scheme):
8 for the naive algorithm, 7 for Strassen's — the proven optimum
for this tensor (Hopcroft-Kerr / de Groote).

Moves — the classical bilinear-algorithm identities (the
Brent-equation / flip-graph vocabulary):

* ``cut(i)`` — un-commit term *i*: the residual absorbs its
  tensor back (the Las Vegas cancel move — rank -1, but the
  scheme is wrong until the residual is repaired);
* ``pick(k)`` — the add move, staged: three picks choose
  ``u, v, w`` from :data:`FACTORS`; the committed term is
  subtracted from the residual (rank +1).  Staged because a flat
  ``add(u, v, w)`` action would enumerate ``32³`` candidates per
  step;
* ``abort`` — abandon a pending pick;
* ``flip(i, j, m, s, λ)`` — the flip-graph move: two terms with
  proportional *m*-factors redistribute mass between the other
  two modes (``u⊗x⊗y + u⊗x'⊗y' = u⊗(x-λx')⊗y +
  u⊗x'⊗(y'+λy)`` — rank- and validity-preserving);
* ``drop(i)`` — remove a term carrying a zero factor (the
  flip-graph "reduction": rank -1, validity-preserving);
* ``rescale(i, m, n, λ)`` — rebalance a term's factor norms
  (``u⊗v⊗w`` is invariant under ``u·λ, v/λ``);
* ``stop`` — end the episode.

:data:`FACTORS` is the AlphaTensor coefficient alphabet —
``{0,±1}``-vectors of weight ≤ 2 (32 vectors) — the sparse
factor pool the known fast schemes live in.

The referee is the residual itself: ``valid`` iff ``R = 0``
*exactly* (Fractions — no float noise), and ``verify_terms`` /
``verify_numeric`` re-check a scheme against the true tensor.
Reward is the quality delta ``-rank - K·‖R‖₁`` — telescoping to
``initial_rank - final_rank`` over valid endpoints, with the
honest valley in between: the Strassen line (cut the naive 8,
commit the 7) crosses it at depth ~16‖R‖-mass.  ``best_rank``
records the lowest *valid* rank reached — the episode's
deliverable number, surfaced as ``Report.cost``.

**Honesty note.**  This board tests *rediscovery inside a move
set built for bilinear algorithms* — every legal move is a
Brent-equation identity — not discovery from raw arithmetic.
The scripted line (:func:`strassen_line`) proves the vocabulary
expresses rank 7; whether search finds it without the map is
the measured question (``bench/drivers/probes/strassen_probe.py``,
``docs/results_strassen.md``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from gamekit.players import (
    ActionFeats,
    Bias,
    DomainSpec,
    OpOneHot,
    PlayRates,
    StateBroadcast,
    register_legal,
    register_part,
)
from gamekit.players.registry import PlayerSpec, register_player

from mlopt_discovery.meta_arena import Action

__all__ = [
    "BILINEAR_FEATURES",
    "BILINEAR_RESIDUAL_WEIGHT",
    "BILINEAR_SPEC",
    "FACTORS",
    "FLIP_LAMBDAS",
    "NAIVE_TERMS",
    "RESCALE_LAMBDAS",
    "STRASSEN_TERMS",
    "TENSOR",
    "BilinearBoard",
    "BilinearState",
    "featurize",
    "gen_cases",
    "legal_actions",
    "residual_of",
    "strassen_line",
    "term_tensor",
    "verify_numeric",
    "verify_terms",
]

F = Fraction

# ---------------------------------------------------------------------------
#  The tensor, the schemes, the factor dictionary
# ---------------------------------------------------------------------------
#
#  Entries of A, B, C are flattened in (11, 12, 21, 22) order, so
#  c_ij = Σ_k a_ik b_kj gives T's support:
#    c11 ← a11·b11, a12·b21     c12 ← a11·b12, a12·b22
#    c21 ← a21·b11, a22·b21     c22 ← a21·b12, a22·b22

Vec = tuple  # tuple[Fraction, ...], length 4
Term = tuple  # (u, v, w) — three Vecs


def _e(i: int) -> Vec:
    """Return the *i*-th basis vector of Q⁴."""
    return tuple(F(1) if k == i else F(0) for k in range(4))


#: The <2,2,2> matmul tensor's support — the (a, b, c) index
#: triples whose coefficient is 1.
T_SUPPORT: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 2, 0),
    (0, 1, 1),
    (1, 3, 1),
    (2, 0, 2),
    (3, 2, 2),
    (2, 1, 3),
    (3, 3, 3),
)


def _flat(x: int, y: int, z: int) -> int:
    """Flatten the 4x4x4 tensor index."""
    return x * 16 + y * 4 + z


#: ``T`` as a flat 64-vector of exact coefficients.
TENSOR: tuple = tuple(
    F(1) if (x, y, z) in T_SUPPORT else F(0)
    for x in range(4)
    for y in range(4)
    for z in range(4)
)

#: The naive 8-multiplication scheme — one unit term per tensor
#: entry: ``u = e_a, v = e_b, w = e_c``.
NAIVE_TERMS: tuple = tuple(
    (_e(x), _e(y), _e(z)) for (x, y, z) in T_SUPPORT
)


def _u(*xs: int) -> Vec:
    """Build an integer vector literal — Strassen factor shorthand."""
    return tuple(F(x) for x in xs)


#: Strassen's rank-7 scheme (de Groote — essentially unique for
#: this tensor).  Products p_k = (u_k·a)(v_k·b); the c-mixes ride
#: on w_k.
STRASSEN_TERMS: tuple = (
    (_u(1, 0, 0, 1), _u(1, 0, 0, 1), _u(1, 0, 0, 1)),
    (_u(0, 0, 1, 1), _u(1, 0, 0, 0), _u(0, 0, 1, -1)),
    (_u(1, 0, 0, 0), _u(0, 1, 0, -1), _u(0, 1, 0, 1)),
    (_u(0, 0, 0, 1), _u(-1, 0, 1, 0), _u(1, 0, 1, 0)),
    (_u(1, 1, 0, 0), _u(0, 0, 0, 1), _u(-1, 1, 0, 0)),
    (_u(-1, 0, 1, 0), _u(1, 1, 0, 0), _u(0, 0, 0, 1)),
    (_u(0, 1, 0, -1), _u(0, 0, 1, 1), _u(1, 0, 0, 0)),
)


def _factor_dict() -> list:
    """Return the ``{0,±1}``-vectors of weight ≤ 2 — the factor alphabet.

    Singles ``±e_i`` (8), pairs ``±e_i ± e_j`` for ``i < j``
    (24) — 32 vectors, deterministic order.  Every Strassen
    factor is a member, so the scripted line's ``pick`` actions
    all resolve.
    """
    out = []
    for i in range(4):
        out.append(_e(i))
    for i in range(4):
        out.append(tuple(-x for x in _e(i)))
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (F(1), F(-1)):
                for sj in (F(1), F(-1)):
                    out.append(
                        tuple(
                            si * F(1)
                            if k == i
                            else (sj * F(1) if k == j else F(0))
                            for k in range(4)
                        )
                    )
    return out


#: The add-move's per-stage dictionary (see the module docstring).
FACTORS: tuple = tuple(_factor_dict())

#: Flip shift coefficients offered to the enumerator.
FLIP_LAMBDAS: tuple = (F(-2), F(-1), F(1), F(2))

#: Rescale ratios offered to the enumerator.
RESCALE_LAMBDAS: tuple = (F(-1), F(2), F(1, 2))

#: Residual-mass weight in the reward's quality score — the
#: valley depth a cut-add repair must climb out of.
BILINEAR_RESIDUAL_WEIGHT: float = 16.0


# ---------------------------------------------------------------------------
#  Vector / term / tensor algebra (exact — Fractions)
# ---------------------------------------------------------------------------


def _vadd(u: Vec, v: Vec) -> Vec:
    return tuple(a + b for a, b in zip(u, v, strict=True))


def _vsub(u: Vec, v: Vec) -> Vec:
    return tuple(a - b for a, b in zip(u, v, strict=True))


def _vscale(u: Vec, s: Any) -> Vec:
    return tuple(a * s for a in u)


def _is_zero(u: Vec) -> bool:
    return all(a == 0 for a in u)


def _nnz(u: Vec) -> int:
    return sum(1 for a in u if a != 0)


def _ratio(u: Vec, v: Vec) -> Any:
    """Return rho with ``u = rho·v``, or ``None`` if not proportional.

    A zero *v* has no ratio (the zero factor is ``drop``'s
    business, not ``flip``'s); a zero *u* ratios as 0.
    """
    rho = None
    for a, b in zip(u, v, strict=True):
        if b == 0:
            if a != 0:
                return None
        else:
            r = a / b
            if rho is None:
                rho = r
            elif rho != r:
                return None
    return rho


def term_tensor(term: Term) -> tuple:
    """Return the term's 64-entry tensor — ``u ⊗ v ⊗ w``, exact."""
    u, v, w = term
    return tuple(
        u[x] * v[y] * w[z]
        for x in range(4)
        for y in range(4)
        for z in range(4)
    )


def residual_of(terms: Any) -> tuple:
    """``T - Σ_k u_k ⊗ v_k ⊗ w_k`` — zero iff the scheme is exact."""
    acc = [F(0)] * 64
    for t in terms:
        tt = term_tensor(t)
        acc = [a + e for a, e in zip(acc, tt, strict=True)]
    return tuple(t - a for t, a in zip(TENSOR, acc, strict=True))


def verify_terms(terms: Any) -> bool:
    """Exact check — ``Σ u_k ⊗ v_k ⊗ w_k == T`` over Q."""
    return all(x == 0 for x in residual_of(terms))


def _eval_scheme(terms: Any, a: Any, b: Any) -> list:
    """Evaluate the bilinear form on flat entry vectors a, b (Q)."""
    ms = []
    for u, v, w in terms:
        la = sum(ui * ai for ui, ai in zip(u, a, strict=True))
        lb = sum(vj * bj for vj, bj in zip(v, b, strict=True))
        ms.append((la * lb, w))
    return [sum(w[li] * m for m, w in ms) for li in range(4)]


def verify_numeric(
    terms: Any, *, samples: int = 8, seed: int = 0
) -> bool:
    """Check the scheme as a function vs true 2x2 matmul, sampled.

    Draws random ``A, B``, evaluates the decomposition's linear
    forms, and compares against ``A·B`` entrywise.  The
    oracle-side check mirroring how a numeric referee would judge
    a bilinear form; :func:`verify_terms` is the exact one.
    """
    rng = random.Random(seed)
    for _ in range(samples):
        a = [rng.uniform(-1, 1) for _ in range(4)]
        b = [rng.uniform(-1, 1) for _ in range(4)]
        # c_ij = Σ_k a_ik b_kj — flattened (11,12,21,22):
        want = [
            a[0] * b[0] + a[1] * b[2],
            a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3],
        ]
        got = _eval_scheme(terms, a, b)
        if (
            max(
                abs(float(g) - w)
                for g, w in zip(got, want, strict=True)
            )
            > 1e-9
        ):
            return False
    return True


def _term_nnz(term: Term) -> int:
    """Tensor weight of a term — ``nnz(u)·nnz(v)·nnz(w)``."""
    u, v, w = term
    return _nnz(u) * _nnz(v) * _nnz(w)


def _flip(ti: Term, tj: Term, m: int, s: int, lam: Any) -> Any:
    """One flip on mode *m*, shift mode *s* — or ``None``.

    Requires the *m*-factors proportional; *tj*'s is first
    normalized to *ti*'s (the ratio folds into the third mode's
    factor), then ``f_s(t_i) -= λ·f_s(t_j)`` and
    ``f_t(t_j) += λ·f_t(t_i)`` — the identity
    ``u⊗x⊗y + u⊗x'⊗y' = u⊗(x-λx')⊗y + u⊗x'⊗(y'+λy)``.
    """
    t = 3 - m - s
    fi, fj = ti[m], tj[m]
    if _is_zero(fi) or _is_zero(fj):
        return None
    rho = _ratio(fj, fi)  # f_j = rho·f_i → normalize to f_i
    if rho is None or rho == 0:
        return None
    nj = list(tj)
    nj[m] = fi
    nj[t] = _vscale(tj[t], rho)
    ni = list(ti)
    ni[s] = _vsub(ti[s], _vscale(nj[s], lam))
    nj[t] = _vadd(nj[t], _vscale(ti[t], lam))
    return (tuple(ni), tuple(nj))


def _droppable(terms: Any) -> list:
    """Return indices of terms carrying a zero factor."""
    return [
        i for i, t in enumerate(terms) if any(_is_zero(f) for f in t)
    ]


def _norm(residual: tuple) -> Any:
    """``‖R‖₁`` — the residual's L1 mass (exact Fraction)."""
    return sum(abs(x) for x in residual)


# ---------------------------------------------------------------------------
#  The board
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BilinearState:
    """The read-only snapshot a bilinear-game player decides on.

    ``terms`` are the committed rank-1 summands; ``residual`` is
    the flat 64-entry ``T - Σ`` (zero ⟺ the scheme is a correct
    bilinear form); ``pending`` holds the 0-2 factor picks of an
    add move in flight; ``best_rank`` is the lowest valid rank
    the episode has reached.
    """

    terms: tuple
    residual: tuple
    pending: tuple
    steps: int
    done: bool
    best_rank: int

    @property
    def rank(self) -> int:
        """The scheme's multiplication count."""
        return len(self.terms)

    @property
    def valid(self) -> bool:
        """``R == 0`` — the committed terms compute ``A·B``."""
        return all(x == 0 for x in self.residual)

    @property
    def norm(self) -> Any:
        """``‖R‖₁`` — how wrong the current scheme is."""
        return _norm(self.residual)


class BilinearBoard:
    """One decomposition under the Brent-equation move set.

    *case* is ``(name, terms)`` or ``None`` for the canonical
    naive start.  The episode terminates on ``stop``, on the
    engine's budget, or when a *valid* scheme of rank ≤ 7 is
    reached — the proven optimum, so the win ends the game.
    """

    def __init__(self, case: Any = None) -> None:
        """Board *case*'s decomposition (default: the naive 8)."""
        self.name = case[0] if case is not None else "naive"
        self.terms = (
            list(case[1]) if case is not None else list(NAIVE_TERMS)
        )
        self.residual = residual_of(self.terms)
        self.pending: list = []
        self.steps = 0
        self.done = False
        self.best_rank = (
            len(self.terms)
            if all(x == 0 for x in self.residual)
            else 99
        )

    def observe(self) -> BilinearState:
        """Snapshot terms, residual, pending picks, terminator."""
        return BilinearState(
            terms=tuple(self.terms),
            residual=self.residual,
            pending=tuple(self.pending),
            steps=self.steps,
            done=self.done,
            best_rank=self.best_rank,
        )

    def _quality(self) -> float:
        """``-rank - K·‖R‖₁`` — the score the reward differentiates."""
        return -len(self.terms) - BILINEAR_RESIDUAL_WEIGHT * float(
            _norm(self.residual)
        )

    def _add_residual(self, term: Term, sign: int) -> None:
        """``R += sign · (u⊗v⊗w)`` — cut un-commits, add commits."""
        tt = term_tensor(term)
        self.residual = tuple(
            r + sign * e for r, e in zip(self.residual, tt, strict=True)
        )

    def _finish(self, action: Any, q0: float) -> tuple:
        """Bookkeep and report — rank record, win check, reward."""
        from gamekit.engine import Report

        self.steps += 1
        if self.observe().valid and len(self.terms) < self.best_rank:
            self.best_rank = len(self.terms)
        if self.observe().valid and len(self.terms) <= 7:
            self.done = True  # 7 is optimal for <2,2,2> — the win
        return self.observe(), Report(
            action,
            reward=self._quality() - q0,
            terminal=self.done,
            cost=float(self.best_rank),
            certificate_ok=self.observe().valid,
            detail={
                "rank": len(self.terms),
                "valid": self.observe().valid,
                "norm": float(_norm(self.residual)),
                "best_rank": self.best_rank,
            },
        )

    def step(self, action: Any) -> tuple[BilinearState, Any]:
        """Apply one move; return ``(state', report)``."""
        if self.done:
            raise RuntimeError("episode finished")
        q0 = self._quality()
        op, p = action.op, action.params
        if op == "stop":
            self.done = True
        elif op == "cut":
            t = self.terms.pop(p["i"])
            self._add_residual(t, +1)
        elif op == "pick":
            self.pending.append(FACTORS[p["k"]])
            if len(self.pending) == 3:
                t = tuple(self.pending)
                self.pending = []
                self.terms.append(t)
                self._add_residual(t, -1)
        elif op == "abort":
            self.pending = []
        elif op == "flip":
            r = _flip(
                self.terms[p["i"]],
                self.terms[p["j"]],
                p["m"],
                p["s"],
                p["lam"],
            )
            if r is None:
                raise ValueError(
                    "illegal flip — modes not proportional"
                )
            self.terms[p["i"]], self.terms[p["j"]] = r
        elif op == "drop":
            t = self.terms[p["i"]]
            if not any(_is_zero(f) for f in t):
                raise ValueError("drop needs a zero factor")
            self.terms.pop(p["i"])
        elif op == "rescale":
            i, m, n, lam = p["i"], p["m"], p["n"], p["lam"]
            if m == n or lam == 0:
                raise ValueError("illegal rescale")
            t = list(self.terms[i])
            t[m] = _vscale(t[m], lam)
            t[n] = _vscale(t[n], 1 / lam)
            self.terms[i] = tuple(t)
        else:
            raise ValueError(f"unknown bilinear action {op!r}")
        return self._finish(action, q0)


def legal_actions(state: BilinearState) -> tuple:
    """Enumerate the legal moves — the Brent-equation vocabulary.

    Mid-pick states offer only the 32 factor picks plus ``abort``;
    otherwise: one ``cut`` per term, a ``drop`` per zero-factored
    term, every valid flip (auto-normalized — proportional
    *m*-factors only), rescalings, and ``stop``.
    """
    if state.done:
        return ()
    terms = state.terms
    acts: list = []
    if state.pending:
        stage = len(state.pending)
        acts = [
            Action("pick", {"k": k, "stage": stage, "w": _nnz(f)})
            for k, f in enumerate(FACTORS)
        ]
        acts.append(Action("abort", {}))
        return tuple(acts)
    n = len(terms)
    for i in range(n):
        acts.append(
            Action("cut", {"i": i, "tnnz": _term_nnz(terms[i])})
        )
    for i in _droppable(terms):
        acts.append(Action("drop", {"i": i}))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for m in range(3):
                for s in range(3):
                    if s == m:
                        continue
                    for lam in FLIP_LAMBDAS:
                        r = _flip(terms[i], terms[j], m, s, lam)
                        if r is None:
                            continue
                        acts.append(
                            Action(
                                "flip",
                                {
                                    "i": i,
                                    "j": j,
                                    "m": m,
                                    "s": s,
                                    "lam": lam,
                                    "mz": any(
                                        _is_zero(f)
                                        for t in r
                                        for f in t
                                    ),
                                },
                            )
                        )
    for i in range(n):
        for m in range(3):
            for nn in range(3):
                if nn == m:
                    continue
                for lam in RESCALE_LAMBDAS:
                    acts.append(
                        Action(
                            "rescale",
                            {"i": i, "m": m, "n": nn, "lam": lam},
                        )
                    )
    acts.append(Action("stop", {}))
    return tuple(acts)


# ---------------------------------------------------------------------------
#  The learned arm's spec — schema + parts, data not code
# ---------------------------------------------------------------------------

_OPS = ("cut", "pick", "abort", "flip", "drop", "rescale", "stop")

BILINEAR_FEATURES: tuple[str, ...] = (
    "bias",
    "st:rank",
    "st:norm",
    "st:valid",
    "st:best",
    "st:steps",
    "st:pending",
    *(f"h:{o}" for o in _OPS),
    *(f"op:{o}" for o in _OPS),
    "a:stage",
    "a:wnnz",
    "a:cut_nnz",
    "a:makes_zero",
    "a:shared",
)


def _state_feats(state: BilinearState) -> dict[str, float]:
    """Return the ``st:*`` block — rank, residual mass, record, progress."""
    return {
        "st:rank": len(state.terms) / 8.0,
        "st:norm": float(_norm(state.residual)) / 16.0,
        "st:valid": 1.0 if state.valid else 0.0,
        "st:best": min(state.best_rank, 8) / 8.0,
        "st:steps": state.steps / 24.0,
        "st:pending": len(state.pending) / 3.0,
    }


def _act_feats(
    action: Action, _row: dict[str, float]
) -> dict[str, float]:
    """Return the ``a:*`` block — the move's own shape."""
    p = action.params
    if action.op == "pick":
        return {"a:stage": p["stage"] / 3.0, "a:wnnz": p["w"] / 2.0}
    if action.op == "cut":
        return {"a:cut_nnz": p["tnnz"] / 8.0}
    if action.op == "flip":
        return {
            "a:shared": 1.0,
            "a:makes_zero": 1.0 if p["mz"] else 0.0,
        }
    return {}


register_part("bilinear:state", lambda: StateBroadcast(_state_feats))
register_part("bilinear:hist", lambda: PlayRates(_OPS))
register_part(
    "bilinear:onehot",
    lambda: OpOneHot(frozenset(BILINEAR_FEATURES)),
)
register_part("bilinear:action", lambda: ActionFeats(_act_feats))
register_legal("bilinear", legal_actions)

#: The bilinear player spec — no shipped prior; ``linear`` here is
#: the cold start (the honest uniform baseline, like ``search``).
BILINEAR_SPEC = DomainSpec.build(
    features=BILINEAR_FEATURES,
    legal=legal_actions,
    parts=(
        Bias(),
        StateBroadcast(_state_feats),
        PlayRates(_OPS),
        OpOneHot(frozenset(BILINEAR_FEATURES)),
        ActionFeats(_act_feats),
    ),
)

#: The composed featurizer — public for tests.
featurize = BILINEAR_SPEC.featurize


# ---------------------------------------------------------------------------
#  The scripted line + the greedy playbook
# ---------------------------------------------------------------------------


def _pick_of(f: Vec) -> Action:
    """Return the ``pick`` action choosing factor *f* from the dictionary."""
    return Action("pick", {"k": FACTORS.index(f)})


def strassen_line() -> Any:
    """Play the recorded Strassen line — the expressibility witness.

    Policy: cut every committed term (the residual becomes the
    whole tensor), then commit Strassen's 7 terms through staged
    picks — 8 + 21 = 29 moves, ending at a verified rank-7
    scheme (the board auto-terminates on the win).  Works on any
    starting scheme, so the scrambled cases don't faze it: the
    board *can* express Strassen — the move set is not vacuous.
    """
    plan: list = []

    def go(state: BilinearState) -> Any:
        if plan:
            return plan.pop(0)
        if state.terms:
            return Action(
                "cut", {"i": 0, "tnnz": _term_nnz(state.terms[0])}
            )
        if state.valid:
            return Action("stop", {})
        for u, v, w in STRASSEN_TERMS:
            plan.extend((_pick_of(u), _pick_of(v), _pick_of(w)))
        return plan.pop(0)

    return go


def _patch_matrix() -> Any:
    """Return the 32³x64 int8 table — every dictionary term's tensor.

    Lazy-imported numpy; the greedy repair scores
    ``|R - t|`` over all candidates in one subtract.
    """
    import numpy as np

    fd = np.array([[float(x) for x in f] for f in FACTORS])
    return np.einsum("xi,yj,zl->xyzijl", fd, fd, fd).reshape(-1, 64)


_PATCH_MAT: Any = None


class _GreedyRepair:
    """The ``playbook`` arm — greedy residual patching.

    If a zero-factored term exists, ``drop`` it (the free
    reduction).  If the residual is nonzero, commit the
    dictionary triple that minimizes ``‖R - u⊗v⊗w‖₁`` —
    cheapest-descent repair.  If valid but over rank 7, cut the
    sparsest committed term and repair again — the local-search
    line that (the measurement shows) rediscovers the *naive*
    scheme, not Strassen: patching greedily never crosses the
    valley.  Terms already cut this episode are excluded from
    patching, else the repair re-commits the cut verbatim.
    """

    def __init__(self) -> None:
        self._plan: list = []
        self._cut: set = set()
        self._steps = -1

    def __call__(self, state: BilinearState) -> Any:
        if state.steps < self._steps:
            self._plan = []
            self._cut = set()
        self._steps = state.steps
        if state.pending:
            return (
                self._plan.pop(0) if self._plan else Action("abort", {})
            )
        drops = _droppable(state.terms)
        if drops:
            return Action("drop", {"i": drops[0]})
        if not state.valid:
            if not self._plan:
                self._plan = self._best_patch(state)
            return (
                self._plan.pop(0) if self._plan else Action("stop", {})
            )
        if len(state.terms) <= 7:
            return Action("stop", {})
        i = min(
            range(len(state.terms)),
            key=lambda k: (_term_nnz(state.terms[k]), k),
        )
        self._cut.add(state.terms[i])
        return Action(
            "cut", {"i": i, "tnnz": _term_nnz(state.terms[i])}
        )

    def _best_patch(self, state: BilinearState) -> list:
        """Best dictionary triple under L1 residual — as picks."""
        import numpy as np

        global _PATCH_MAT
        if _PATCH_MAT is None:
            _PATCH_MAT = _patch_matrix()
        r = np.array([float(x) for x in state.residual])
        scores = np.abs(r[None, :] - _PATCH_MAT).sum(axis=1)
        nf = len(FACTORS)
        for idx in np.argsort(scores, kind="stable"):
            iu, rem = divmod(int(idx), nf * nf)
            iv, iw = divmod(rem, nf)
            cand = (FACTORS[iu], FACTORS[iv], FACTORS[iw])
            if cand not in self._cut:
                return [Action("pick", {"k": k}) for k in (iu, iv, iw)]
        return []


def _playbook_make(
    domain: str, *, spec: Any, seed: int = 0, ctx: Any = None
) -> Any:
    """Return the registered ``playbook`` arm factory for the bilinear board."""
    del domain, spec, seed, ctx
    return lambda i: _GreedyRepair()


_RECORD = (
    "bilinear: scripted line reaches verified rank 7 in 29 moves "
    "(docs/results_strassen.md) — the expressibility witness"
)

#: External seats on the shipped roster — ``bilinear`` is not an
#: ``EPISODE_BOARDS`` member, so the domain registers its players
#: here (the plan-0032 plugin seam), exactly like kernelopt's
#: ``kernel`` does from ``mlopt_kernel.play``.
register_player(
    PlayerSpec(
        name="random",
        kind="chance",
        boards=("bilinear",),
        record="uniform over the Brent-equation move set",
    )
)
register_player(
    PlayerSpec(
        name="linear",
        kind="linear",
        boards=("bilinear",),
        schema="BILINEAR_SPEC (no shipped prior — cold start)",
        record="bilinear: cold linear arm (docs/results_strassen.md)",
    )
)
register_player(
    PlayerSpec(
        name="linear-greedy",
        kind="linear",
        boards=("bilinear",),
        schema="BILINEAR_SPEC — argmax decode",
        record="bilinear: cold linear, greedy (docs/results_strassen.md)",
        variant={"greedy": True},
    )
)
register_player(
    PlayerSpec(
        name="linear-cold",
        kind="linear",
        boards=("bilinear",),
        schema="BILINEAR_SPEC — weights={}",
        record="bilinear: the untrained control",
        variant={"cold": True},
    )
)
register_player(
    PlayerSpec(
        name="playbook",
        kind="scripted",
        boards=("bilinear",),
        record="bilinear: greedy repair — cheapest-descent patching",
    ),
    make=_playbook_make,
)


# ---------------------------------------------------------------------------
#  Cases — the naive scheme under a validity-preserving scramble
# ---------------------------------------------------------------------------


def _scramble(terms: list, rng: random.Random, rounds: int) -> list:
    """Random validity-preserving moves — a different start, same T.

    Applies *rounds* random flips (λ ∈ ±1) and rescales, skipping
    any flip that zeroes a factor (no free drops on the deal).
    """
    terms = list(terms)
    applied = 0
    for _ in range(rounds * 40):  # attempts; most pairs share no factor
        if applied >= rounds:
            break
        i, j = rng.randrange(len(terms)), rng.randrange(len(terms))
        if i == j:
            continue
        m, s = rng.randrange(3), rng.randrange(3)
        if m == s:
            continue
        r = _flip(terms[i], terms[j], m, s, rng.choice((F(-1), F(1))))
        if r is None or any(_is_zero(f) for t in r for f in t):
            continue
        terms[i], terms[j] = r
        applied += 1
        if rng.random() < 0.25:
            k = rng.randrange(len(terms))
            m, nn = rng.sample(range(3), 2)
            lam = rng.choice(RESCALE_LAMBDAS)
            t = list(terms[k])
            t[m] = _vscale(t[m], lam)
            t[nn] = _vscale(t[nn], 1 / lam)
            terms[k] = tuple(t)
    return terms


def gen_cases(seed: int, n: int, *, scramble: int = 8) -> list:
    """Generate *n* cases — scrambled naive decompositions.

    Each case is ``(name, terms)``: the same ``<2,2,2>`` tensor,
    a different rank-8 starting scheme (``scramble`` random flips
    / rescales — always valid, never pre-dropped).  Case 0 is the
    canonical naive board, so measurements have one fixed anchor.
    """
    out = [("bilinear0:naive", NAIVE_TERMS)]
    for i in range(1, n):
        crng = random.Random(seed * 10_007 + i)
        terms = _scramble(list(NAIVE_TERMS), crng, scramble)
        out.append((f"bilinear{i}", tuple(terms)))
    return out

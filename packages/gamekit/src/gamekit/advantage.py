"""Per-step advantage — an actor-critic :class:`LearnedPlayer`.

``docs/results_stop_policy.md``'s repeated collapse has a credit-
assignment half: ``LinearPolicy.finish_episode`` prices every sampled
move at ``G_t - b`` where *b* is an EMA of the *episode total* — so
the ``stop`` action's credit is drowned in whatever the episode
happened to pay.  After a good page even a mistimed stop reads
positive; after a bad one a correct stop reads negative; and the
shared-softmax gradient that noise injects randomizes the ``a:*``
ranking columns (the measured AP collapses of rounds 3-5).

:class:`AdvantagePlayer` swaps the global baseline for a learned
*state-value* baseline ``V(s) = u·φ_state``: step *t*'s credit is
``G_t - V(s_t)`` — the realized reward-to-go minus what the model
predicted this state would still pay.  The ``stop`` action's credit
then isolates to ``-V(s_stop)``: quit is pushed up exactly when the
state predicts nothing left to collect, pushed down while a paying
pool remains — the decision quantity the flat baseline could not
express.  The same subtraction also lowers the variance of the
``evaluate`` rows' credit (each step is measured against its own
state's prediction, not the episode's global mean).

V is fit online on the episode's own ``(state_row, G_t)`` pairs —
Monte-Carlo return-to-go targets, ``v_passes`` sweeps per episode of
the *normalized* LMS step ``v += (v_lr·err/‖φ‖²)·φ`` (the residual
clipped at the advantage clip): the raw delta rule diverges on the
``x:stop:self_*`` columns — those features read the policy's own
logits, whose scale grows as the weights sharpen, and an
unnormalized step chases the growing feature norm until the values
overflow (measured: ``ac+self`` overflowed ``fsum`` mid-run).
NLMS bounds each row's step at ``v_lr·|err|`` — stable at any
feature scale.  The state row is injected as
*state_feats* ``(state, hist) -> dict``; for the open game the
``stop`` action's own row is the natural choice — it is the only row
whose columns are a pure state summary (``st:*`` rates, the
``x:stop:*`` pool columns, the ``x:stop:self_*`` posterior block).
"""

from __future__ import annotations

from collections.abc import Iterable
from statistics import fmean
from typing import Any

from gamekit.players import LearnedPlayer
from gamekit.players._base import _CLIP, _dot

__all__ = ["AdvantagePlayer"]


class AdvantagePlayer(LearnedPlayer):
    """A softmax linear policy trained on per-step advantage.

    Identical to :class:`LearnedPlayer` at decode time — same
    featurizer, same sampled/greedy pick — and different only in
    :meth:`finish_episode`: the REINFORCE gradient uses the
    per-step advantage ``clip(G_t - V(s_t))`` instead of
    ``clip(G_t - EMA(total))``, then refits V on the episode's rows.

    *state_feats* is the ``(state, hist) -> dict`` injector for the
    state summary V reads (``None`` → V sees ``{}`` and stays at 0 —
    the arm degrades to no-baseline REINFORCE, the honest cold
    start).  *v_lr* / *v_passes* are the delta-rule rate and the
    per-episode fitting sweeps.
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        state_feats: Any = None,
        v_lr: float = 0.05,
        v_passes: int = 4,
        **kw: Any,
    ) -> None:
        """Bind the spec, the value-function row and the fit knobs."""
        super().__init__(seed, **kw)
        self._state_feats = state_feats
        self._v: dict[str, float] = {}
        self._v_lr = v_lr
        self._v_passes = v_passes
        #: Diagnostics — mean |G_t - V| on the last episode's fit
        #: pass; the driver reports it as the V-fit read.
        self._v_err = 0.0
        self._states: list[dict] = []

    def __call__(self, state: Any) -> Any | None:
        """Pick a move; record the state row V will price it against."""
        if not self._learn:
            return super().__call__(state)
        self._observe(state)
        srow = (
            {}
            if self._state_feats is None
            else self._state_feats(state, self._hist)
        )
        act = super().__call__(state)
        if act is not None:
            self._states.append(srow)
        return act

    def finish_episode(
        self, total: float, rewards: Iterable[float] | None = None
    ) -> None:
        """REINFORCE with the per-step advantage, then refit V.

        ``G_t`` is the reward-to-go (``total`` when *rewards* is not
        given — the flat-return fallback ``LinearPolicy`` also
        honours).  The policy update mirrors
        :meth:`LinearPolicy.finish_episode` exactly except the
        baseline is ``V(s_t)`` per step; the value update is
        ``v_passes`` delta-rule sweeps over the episode's
        ``(state_row, G_t)`` pairs.  No-op when ``learn=False``.
        """
        if not (self._learn and self._ep):
            self._ep = []
            self._states = []
            return
        rs = list(rewards) if rewards is not None else None
        n = len(self._ep)
        g = [
            sum(rs[t:]) if rs is not None and t < len(rs) else total
            for t in range(n)
        ]
        states = self._states[:n]
        states += [{}] * (n - len(states))  # defensive pad
        vhat = [_dot(self._v, s) for s in states]
        grad: dict[str, float] = {}
        for (phi, ephi), g_t, v_t in zip(
            self._ep, g, vhat, strict=True
        ):
            a = max(-_CLIP, min(_CLIP, g_t - v_t))
            for k, v in phi.items():
                grad[k] = grad.get(k, 0.0) + self._lr * a / n * v
            for k, v in ephi.items():
                grad[k] = grad.get(k, 0.0) - self._lr * a / n * v
        for k, gv in grad.items():
            self._w[k] = self._w.get(k, 0.0) + gv
        errs: list[float] = []
        for _ in range(self._v_passes):
            errs = [
                g_t - _dot(self._v, s)
                for s, g_t in zip(states, g, strict=True)
            ]
            for s, err in zip(states, errs, strict=True):
                e = max(-_CLIP, min(_CLIP, err))
                nrm = _dot(s, s) + 1e-6  # ‖φ‖² — the NLMS normalizer
                step = self._v_lr * e / nrm
                for k, fv in s.items():
                    self._v[k] = self._v.get(k, 0.0) + step * fv
        self._v_err = (
            fmean(min(abs(e), _CLIP) for e in errs) if errs else 0.0
        )
        self._ep = []
        self._states = []

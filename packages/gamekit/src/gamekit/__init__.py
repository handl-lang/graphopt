"""The game layer — ``gamekit`` (plan 0032 lane A).

The substrate-free half of the optimization game, extracted out of
``mlopt_discovery`` so the Board protocol, the player machinery and
the referee's value spec can travel to their own repo independently
of the tensor-law substrate (``gamekit ⊥ eqsat``).

* :mod:`gamekit.engine` — the one ``Board`` protocol
  (``observe``/``step``), the episode driver :func:`run_episode`,
  the arm comparison :func:`evaluate`, the training loop
  :func:`train_policy` and the expert-replay driver
  ``imitate_policy``.
* :mod:`gamekit.players` — the player machinery:
  :class:`LinearPolicy`, the featurizer-injected
  :class:`LearnedPlayer` over a :class:`DomainSpec` of composable
  feature parts, and the name→spec :mod:`gamekit.players.registry`
  (each player is a folder — ``random``/``playbook``/``linear``/
  ``neural``/``distilled`` — carrying its ``PlayerSpec`` and
  ``make`` arm factory).
* :mod:`gamekit.advantage` — :class:`AdvantagePlayer`, the
  actor-critic ``LearnedPlayer`` (per-step advantage baseline).
* :mod:`gamekit.scoring` — the referee's value function *as data*:
  ``ScoreGate``/``ScoringSpec``/``score`` over a plain
  ``Mapping[str, float]`` measurement — the value spec is
  gamekit's, the measurement is the gauntlet's.
* :mod:`gamekit.calibrate` — score calibration (Platt scaling,
  isotonic regression) and the metrics that judge it.
* :mod:`gamekit.gamedata` — the game-side content tables: the
  player feature schemas (``features``), the prior/reward/referee/
  learned-player weight tables (``weights``) and the shared move
  vocabulary (``moves``).

The kernel is stdlib-only — an import-linter contract pins
``engine``, ``scoring``, ``calibrate``, ``gamedata``,
``players._base``, ``players.registry`` and ``advantage`` to import
nothing outside it.  The player *folders* are deliberately outside
that contract: their ``make`` factories bind domain machinery
(``mlopt_discovery.play``, ``mlopt_torch.contraction_policy``, …)
lazily, which is the registry contract working as intended.

The ``mlopt_discovery.engine`` / ``.scoring`` / ``.advantage`` /
``.players`` / ``.lawdata.game`` import paths remain as
compatibility shims re-exporting this package.
"""

from __future__ import annotations

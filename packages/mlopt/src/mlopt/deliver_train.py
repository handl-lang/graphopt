"""Delivery-prior training — imitation + REINFORCE over the game.

The ``linear*`` arms seated on the ``deliver`` board
(:mod:`mlopt.play`) are cold today: ``DELIVERY_SPEC.weights`` is
``None`` until a prior is baked into :mod:`mlopt.weights`.  This
module is the trainer that fills that seat — the kernelopt
``scripts/train_kernel_prior.py`` recipe adapted to a board whose
"cases" are delivery problems, not a registry domain:

* a :class:`DeliveryCase` is one workload's already-lifted
  ``(optz, res, lowered_eager, inputs, meta)`` — building it is the
  expensive part (adapter ``load`` + ``Optimizer.search`` + eager
  ``lower``), so a case is *data*, reused across every episode and
  every epoch;
* the *oracle* is measured, not scripted: one scratch-board pass
  drains every eligible config through the real referee
  (re-lower + verify + paired ``_bench_paired`` timing) and names
  the fastest *verified* config that beats ITS OWN paired-round
  eager baseline past the ``WIN_MARGIN`` noise floor — ``None``
  when eager wins outright.  :func:`oracle_drain` returns BOTH the
  pick and every config's measurement record
  (``min_ms``/``med_ms``/``verified``/``reason`` + the paired eager
  baseline) — the persisted answer key the autopsy asked for; the
  old bare-name cache discarded exactly the evidence needed to
  audit a noisy label;
* the expert line is therefore ``[best_config, "stop"]`` (or just
  ``["stop"]`` when eager is the argmin past the ``WIN_MARGIN``
  noise floor) — the argmin over the measured pool, which is
  exactly what the commit computes.  Imitation runs on
  oracle-POSITIVE cases only by default (``imitate_positive_only``)
  — the eager-argmin cases teach a degenerate all-stop policy;
  they remain in the REINFORCE stream as the "confirm eager"
  stratum;
* REINFORCE then runs ``gamekit.engine.train_policy`` over the same
  case stream — reward is the terminal *committed* speedup minus
  ``deliver_game.MOVE_COST`` per spent move
  (``DeliveryState.committed_speedup``: the paired-eager/best ratio
  only when a verified config clears the noise margin, else 1.0),
  so the prior learns *which pending config kinds pay on which
  boards* (cuda/fusible/compile/alias flags are broadcast state
  features), *when* to stop paying for measurements, and exploring
  a losing pool never scores below the eager floor.

:func:`weights_literal` renders the trained table as the
``DELIVERY_PLAYER_WEIGHTS`` assignment to paste into
``src/mlopt/weights.py`` — the same baked-artifact convention as
``mlopt_kernel.weights.KERNEL_PLAYER_WEIGHTS``.  Until that paste
happens the table stays ``{}`` and ``play.DELIVERY_SPEC.weights``
stays ``None`` — the honest cold fallback.

Everything here is CPU-testable: the bench timer and runner builder
are the ``mlopt.deliver_game`` module seams the test-suite already
patches, and a :class:`DeliveryCase` can carry explicit eligibility
``flags`` to skip the real ``_probe``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gamekit import engine
from gamekit.players import LearnedPlayer

import mlopt.deliver_game as _dg
from mlopt.play import DELIVERY_SPEC

__all__ = [
    "DeliveryCase",
    "board_of",
    "baked_table",
    "cold_train",
    "oracle_drain",
    "oracle_pick",
    "train",
    "weights_literal",
]

#: Defaults — a delivery episode's move budget mirrors the CLI's
#: ``--deliver-budget`` convention (pending pool is ≤ 7 configs +
#: stop, so 8 covers a full drain plus the commit).
SEED = 0
EPOCHS = 8
IMITATE_EPOCHS = 4
BUDGET = 8
BENCH_ITERS = 50
#: Round the baked weights to this many decimals (kernelopt parity).
ROUND = 4


# ---------------------------------------------------------------------------
#  Cases — one lifted workload each
# ---------------------------------------------------------------------------


@dataclass
class DeliveryCase:
    """One delivery problem, lifted once and replayed every episode.

    ``optz``/``res``/``lowered_eager`` are the same objects
    ``deliver_game`` consumes — the board re-lowers per config under
    the episode's runner, so sharing them across episodes is exactly
    what the product path does.  ``flags`` optionally pins the
    eligibility probes (``fusible``/``cuda``/``compile_ok``/
    ``alias``/``mutable``) for tests and synthetic streams; ``None``
    runs the real :func:`mlopt.deliver_game._probe`.
    """

    name: str
    optz: Any
    res: Any
    lowered_eager: Any
    inputs: Any
    meta: dict[str, Any] = field(default_factory=dict)
    bench_iters: int = BENCH_ITERS
    verify: bool = True
    budget: int = BUDGET
    flags: dict[str, Any] | None = None
    #: ``ModelSpec.kwargs`` analogue — extra kwargs benchmark forwards
    #: carry (``module(*inputs, **call_kwargs)``); empty for the
    #: shipped cases.
    call_kwargs: dict[str, Any] = field(default_factory=dict)


def board_of(case: DeliveryCase, *, probe: bool = False) -> _dg.DeliveryBoard:
    """Build a fresh board for *case* — new context, new measurements.

    ``probe=True`` arms the board's ``probe:<config>`` screening
    moves (see :mod:`mlopt.deliver_game`) — required for the
    probe-then-commit arms; every other arm plays the standard
    board.
    """
    ctx = _dg._Context(
        optz=case.optz,
        res=case.res,
        lowered_eager=case.lowered_eager,
        inputs=case.inputs,
        meta=case.meta,
        bench_iters=case.bench_iters,
        verify=case.verify,
        call_kwargs=dict(case.call_kwargs),
    )
    if case.flags is None:
        _dg._probe(ctx)
    else:
        for key, value in case.flags.items():
            setattr(ctx, key, value)
    return _dg.DeliveryBoard(
        ctx, player_name="deliver-train", budget=case.budget, probe_armed=probe
    )


# ---------------------------------------------------------------------------
#  The oracle — measured argmin over the config pool
# ---------------------------------------------------------------------------


#: Fields persisted per config in an oracle drain record — the
#: measurement evidence, never the runner ``stats`` blob (which can
#: hold non-JSON values).  ``rounds``/``bench`` are the paired-protocol
#: audit trail (per-round min/med pairs + the round/iter split).
_DRAIN_KEYS = (
    "config",
    "runner_name",
    "min_ms",
    "med_ms",
    "eager_min_ms",
    "eager_med_ms",
    "rounds",
    "bench",
    "verified",
    "unverified",
    "skipped",
    "reason",
)


def _drain_records(st: Any) -> dict[str, dict[str, Any]]:
    """Curate ``st.results`` into JSON-safe per-config drain records."""
    out: dict[str, dict[str, Any]] = {}
    for name, rec in st.results.items():
        out[name] = {k: rec[k] for k in _DRAIN_KEYS if k in rec}
    return out


def _state_pick(st: Any) -> str | None:
    """The oracle argmin over an already-drained board state."""
    best = st.best_config
    best_med = st.best_med_ms
    _emin, eager_med = st._eager_for(best)
    if best is None or best_med is None or eager_med is None:
        return None
    # The same noise-margin gate ``committed()``/reward apply: a
    # sub-margin "winner" is bench noise — the expert line is a bare
    # ``stop`` (eager ships), not a config that only looks faster.
    if best_med * _dg.WIN_MARGIN >= eager_med:
        return None
    return best


def oracle_drain(case: DeliveryCase) -> tuple[str | None, dict[str, dict[str, Any]]]:
    """Drain every eligible config on a scratch board — pick AND records.

    Returns ``(pick, records)``: the argmin name (``None`` = eager
    ships) plus every evaluated config's referee record — the
    persisted answer key.  One extra episode per case: every
    eligible config is evaluated through the real referee (lower +
    verify + paired bench), then the state's own
    ``best_config``/``best_med_ms``/paired-eager fields — the same
    numbers ``committed()`` reads — name the winner.
    """
    board = board_of(case)
    st = board.state
    for name in st.pending:  # snapshot: step() shrinks st.pending
        board.step(name)
    return _state_pick(st), _drain_records(st)


def oracle_pick(case: DeliveryCase) -> str | None:
    """Return the fastest verified config on a scratch drain, or ``None``.

    Thin wrapper over :func:`oracle_drain` for callers that only
    need the name; the records are the persisted evidence — use
    ``oracle_drain`` when the measurements matter (training does).
    """
    return oracle_drain(case)[0]


def _oracle_player(pick: str | None) -> Any:
    """Return the expert callable for one case's cached pick.

    Plays the oracle's config once (when it is still pending), then
    ``stop`` — two moves, the argmin line.  A ``None`` pick stops
    immediately: eager ships, reward 1.0.
    """
    played = False

    def go(state: Any) -> Any | None:
        nonlocal played
        if state.done:
            return None
        if pick is not None and not played and pick in state.pending:
            played = True
            return pick
        return "stop"

    return go


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------


def train(
    cases: Any,
    *,
    seed: int = SEED,
    epochs: int = EPOCHS,
    imitate_epochs: int = IMITATE_EPOCHS,
    budget: int = BUDGET,
    imitate_positive_only: bool = True,
    oracle_picks: dict[int, str | None] | None = None,
    oracle_drains: dict[int, dict[str, dict[str, Any]]] | None = None,
) -> tuple[LearnedPlayer, list[float], dict, dict[int, str | None], dict[int, dict]]:
    """Imitation warm-start, then REINFORCE over the case stream.

    *cases* must be a sized sequence — the oracle cache, the drain
    records and the per-index expert factory key on position.
    Returns ``(player, totals, imitation, picks, drains)``: the
    trained arm, the per-episode returns, the demonstrated/imitated
    counts, the per-case oracle picks, and the per-case persisted
    drain records (every config's ``min_ms``/``med_ms``/``verified``/
    ``reason`` — the answer key keeps its measurements).

    ``imitate_positive_only`` (default) restricts the imitation
    stream to oracle-positive cases — the eager-argmin cases teach
    a degenerate all-stop policy (autopsy §4); they stay in the
    REINFORCE stream as the "confirm eager" stratum.  Excluding
    them from RL too is a corpus-design decision left to the
    caller's case split — a follow-up, not silently done here.

    ``oracle_picks``/``oracle_drains`` optionally supply the answer
    key already measured by an earlier ``oracle_drain`` pass (the
    persisted per-config records a corpus driver like
    ``scripts/census_cases.py --drain`` writes) — both must be given
    together, keyed by case index, and the up-front live drain is
    skipped.  REINFORCE episodes still measure every config through
    the real referee; only the one-shot oracle pass is reused.
    """
    cases = list(cases)
    player = LearnedPlayer(seed, spec=DELIVERY_SPEC, weights={})
    if (oracle_picks is None) != (oracle_drains is None):
        raise ValueError("oracle_picks and oracle_drains must be given together")
    if oracle_picks is not None and oracle_drains is not None:
        picks = dict(oracle_picks)
        drains = dict(oracle_drains)
    else:
        picks = {}
        drains = {}
        # One measured drain per case up front — fills BOTH the pick
        # cache and the persisted records.  This is the same single
        # drain per case the lazy expert cache cost whenever imitation
        # ran; doing it eagerly also prices ``imitate_epochs=0`` runs.
        for i in range(len(cases)):
            pick, records = oracle_drain(cases[i])
            picks[i] = pick
            drains[i] = records

    imitation = {"shown": 0, "imitated": 0}
    if imitate_epochs > 0:
        if imitate_positive_only:
            order = [i for i in range(len(cases)) if picks[i] is not None]
        else:
            order = list(range(len(cases)))
        sub_cases = [cases[i] for i in order]

        def expert(j: int, _order: list[int] = order) -> Any:
            return _oracle_player(picks[_order[j]])

        for _ in range(imitate_epochs if sub_cases else 0):
            r = engine.imitate_policy(player, expert, sub_cases, board_of, budget=budget)
            imitation["shown"] += r["shown"]
            imitation["imitated"] += r["imitated"]
    totals: list[float] = []
    for _ in range(epochs):
        totals.extend(engine.train_policy(player, cases, board_of, budget=budget))
    return player, totals, imitation, picks, drains


def cold_train(
    cases: Any,
    *,
    seed: int = SEED,
    epochs: int = EPOCHS,
    budget: int = BUDGET,
) -> tuple[LearnedPlayer, list[float]]:
    """Cold REINFORCE only — the imitation-dependence control arm."""
    cases = list(cases)
    player = LearnedPlayer(seed, spec=DELIVERY_SPEC, weights={})
    totals: list[float] = []
    for _ in range(epochs):
        totals.extend(engine.train_policy(player, cases, board_of, budget=budget))
    return player, totals


def baked_table(player: LearnedPlayer) -> dict[str, float]:
    """Return the snapshot — rounded, nonzero weights only."""
    return {
        name: round(w, ROUND) for name, w in player.weights_dict().items() if abs(w) > 10.0**-ROUND
    }


def weights_literal(table: dict[str, float]) -> str:
    """Render the snapshot as the ``mlopt.weights`` assignment."""
    head = "DELIVERY_PLAYER_WEIGHTS: dict[str, float] = {"
    rows = "".join(f'\n    "{name}": {value!r},' for name, value in table.items())
    return f"{head}{rows}\n}}" if table else f"{head}}}"


def _block_means(totals: list[float], blocks: int) -> list[float]:
    """Mean train reward per block — the visible training curve."""
    size = max(len(totals) // blocks, 1)
    return [
        round(sum(totals[i : i + size]) / len(totals[i : i + size]), 3)
        for i in range(0, len(totals), size)
    ]


# ---------------------------------------------------------------------------
#  Held-out evaluation — the arms table
# ---------------------------------------------------------------------------


def held_out(
    trained: LearnedPlayer,
    eval_cases: Any,
    *,
    seed: int = SEED,
    budget: int = BUDGET,
    cold_rl: LearnedPlayer | None = None,
) -> tuple[dict[str, float], dict[str, str], dict[str, int]]:
    """Play the arms on held-out cases; mean speedup reward per arm.

    Arms mirror the kernel baker's table: ``random`` and ``playbook``
    from the registry, ``cold`` (weights={}), the *shipped* prior
    (``linear``/``linear-greedy`` — whatever ``DELIVERY_SPEC.weights``
    currently binds, ``None`` → cold today), the trained arm and its
    greedy decode, plus ``cold-rl`` when the control ran.

    Evaluation is per ``(arm, case)`` cell through
    :func:`gamekit.engine.evaluate` on a singleton case list — a
    CUDA-context-poisoning config (device-side assert / illegal
    access) kills every LATER cell in the process, so the monolithic
    all-arms call would lose the whole table to one bad case.  Cells
    are attempted in the same arm-major/case order the monolithic
    call used; once a cell raises a poison signature the remaining
    cells are impossible (the context stays dead) and are recorded
    as skipped, never scored.  Returns ``(means, errors, done)``:
    per-arm mean reward over *completed* cells (an arm with fewer
    completed cells than cases is partial — ``done`` carries the
    count), per-arm first-error text, and per-arm completed-cell
    counts.
    """
    from gamekit.players import registry

    import mlopt.play  # noqa: PLC0415 — seats the deliver players

    mlopt.play.register()
    ev = list(eval_cases)
    shipped = LearnedPlayer(seed, spec=DELIVERY_SPEC)  # spec.weights prior
    cold = LearnedPlayer(seed, spec=DELIVERY_SPEC, weights={})
    # Arm order is survival order, not presentation order: a
    # context-poisoning config kills every LATER cell, so the arms
    # whose numbers the bake decision needs (trained + shipped
    # baselines) evaluate first and the free explorers (`playbook`,
    # `random` — most likely to pick an exotic config) go last.
    arms: dict[str, Any] = {
        "trained": lambda i: trained.frozen(seed + 40_000 + i),
        "trained-greedy": lambda i: trained.frozen(seed + 50_000 + i, greedy=True),
        "shipped": lambda i: shipped.frozen(seed + 30_000 + i),
        "shipped-greedy": lambda i: shipped.frozen(seed + 35_000 + i, greedy=True),
        "cold": lambda i: cold.frozen(seed + 20_000 + i),
        "playbook": registry.make("playbook", "deliver", seed=seed),
        "random": registry.make("random", "deliver", seed=seed),
    }
    if cold_rl is not None:
        arms["cold-rl"] = lambda i: cold_rl.frozen(seed + 60_000 + i, greedy=True)
    rows: dict[str, list[dict]] = {name: [] for name in arms}
    errors: dict[str, str] = {}
    done: dict[str, int] = {name: 0 for name in arms}
    poisoned = False
    for name, mk in arms.items():
        for i, case in enumerate(ev):
            if poisoned:
                break
            try:
                # ``evaluate`` enumerates the singleton from 0 — wrap
                # the factory so the arm still sees the CASE index
                # (paired seeds across arms, as the monolithic call).
                part = engine.evaluate(
                    {name: (lambda _j, mk=mk, i=i: mk(i))},
                    [case],
                    board_of,
                    budget=budget,
                )
            except Exception as e:  # noqa: BLE001 — a dead context must not kill the table
                errors[name] = f"{type(e).__name__}: {e} (case index {i} of {len(ev)})"
                if _dg._is_context_poison(e):
                    poisoned = True
                break
            rows[name].extend(part[name])
            done[name] += 1
    if poisoned:
        for name in arms:
            if done[name] < len(ev) and name not in errors:
                errors[name] = "skipped: CUDA context poisoned — cells not attempted"
    means = {
        name: round(sum(r["reward"] for r in rs) / len(rs), 3) if rs else float("nan")
        for name, rs in rows.items()
    }
    return means, errors, done

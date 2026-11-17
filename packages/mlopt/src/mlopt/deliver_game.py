"""Delivery-as-game — runner selection played, not heuristed.

``mlopt optimize --deliver --deliver-budget N`` replaces
:func:`mlopt.deliver.select_runner`'s deterministic pick with a game
on the gamekit ``Board`` protocol: each *move* proposes one runner
configuration; the *referee* is measured latency — the candidate is
re-lowered through ``Optimizer.lower`` (so ``verify`` gates it
exactly as ``lower`` always has) and timed with
:func:`mlopt.deliver._bench_paired` — interleaved eager/config
rounds with the within-round order randomized, so a slow drift on
a contended GPU can't systematically favor whichever side benches
second and eager is re-measured inside every evaluation, not once
up front (the delivery-prior autopsy §4 protocol).  ``stop`` ends
the episode and commits the fastest *verified* configuration —
``eager`` is always in the candidate set, so the shipped artifact
is never slower than the verified eager lowering by construction.
A config only counts as a win when it beats ITS OWN paired-round
eager median by ``WIN_MARGIN`` — the noise floor shared-GPU decode
benches need; the episode reward is that same committed speedup
minus ``MOVE_COST`` per spent move, never the raw (possibly
sub-1.0) ratio.

Why a custom scorer instead of ``gamekit.scoring``: the value spec's
``ScoreGate``/``ScoringSpec`` prices a measurement against absolute
``{axis, threshold}`` bars — pass/fail ceilings.  Delivery is a
*relative* argmin over a candidate pool measured on the spot; there
is no honest absolute bar to declare up front (the eager baseline is
itself measured, not known).  The referee here is the measurement
itself: ``med_ms`` under a mandatory verify gate — a config that
fails ``verify`` is recorded as declined and can never win.  That is
the honest composition the scoring spec's docstring reserves for
phase-2 referees; this board *is* one.

The option space (``CONFIGS``) is what the installed runners can
actually express — probed lazily, never assumed:

* ``fuse`` — ``kit_torch.runners.RegionFusionRunner`` (needs an
  ``IRModule`` + census-eligible regions, ``deliver._fusible``).
* ``compile`` — ``kit_torch.runners.TorchCompileRunner``.
* ``graph`` — ``kit_cuda.CudaGraphRunner`` copy-path capture (CUDA
  inputs/params only).
* ``graph-alias`` — ``CudaGraphRunner`` with the adapter-declared
  ``alias_inputs``/``mutable_inputs`` meta specs (only offered when a
  spec exists — otherwise it would duplicate ``graph``).
* ``fuse+graph`` / ``fuse+graph-alias`` / ``compile+graph`` —
  ``ChainedRunner`` compositions of the above.

Players come from the gamekit registry (``mlopt.play.register()``
seats ``random`` / ``playbook`` / ``linear`` / ``linear-greedy`` /
``linear-cold`` on the ``deliver`` board).  Importing this module
mutates nothing; :func:`deliver_game` registers at call time.
"""

from __future__ import annotations

import contextlib
import time
import zlib
from dataclasses import dataclass, field
from typing import Any

from mlopt.deliver import (
    Delivery,
    _alias_spec,
    _bench,
    _bench_paired,
    _fusible,
    _has_cuda,
    _ledger_notes,
)

__all__ = [
    "CONFIGS",
    "DeliveryBoard",
    "DeliveryState",
    "GameUnavailable",
    "MOVE_COST",
    "PAIRED_ROUNDS",
    "PROBE_CONFIRM_MARGIN",
    "PROBE_MOVE_COST",
    "WIN_MARGIN",
    "deliver_game",
    "legal_actions",
]

#: Noise floor for the eager-vs-config comparison.  Decode-step
#: workloads bench at ~10 ms where shared-GPU noise is a few percent,
#: so a config must beat the eager median by this factor to count as
#: a delivery win — sub-margin "wins" are measurement noise, not
#: speedup, and ship eager.  ``committed()``, the episode reward and
#: the training oracle all share this gate.
WIN_MARGIN = 1.02

#: Interleaved paired-round count for the referee's timing
#: (``mlopt.deliver._bench_paired``).  Each config evaluation runs
#: this many eager/config round-pairs; ``bench_iters`` is split
#: across them (``bench_iters // PAIRED_ROUNDS`` timed calls per
#: side per round), so the total timed-call budget matches the old
#: contiguous protocol while the measurements interleave.
PAIRED_ROUNDS = 5

#: Per-move cost subtracted from the episode reward:
#: ``reward = committed_speedup − MOVE_COST · steps``.  Without it a
#: player that drains the whole pool scores identically to one that
#: stops immediately — the prior can only learn argmax, never
#: *efficient* stopping (delivery-prior autopsy §4).  0.005 is small
#: next to any real speedup (≥ WIN_MARGIN ≈ 1.02) but compounds over
#: a full drain (~0.03), so wasted moves are priced honestly.
MOVE_COST = 0.005

#: Per-probe-move cost subtracted from the episode reward, priced
#: honestly off the instrument's real work: a ``probe:<config>`` move
#: runs the SAME lower + mandatory verify gate as a full evaluation
#: but benches with ``paired_rounds=1`` and ``bench_iters //
#: PAIRED_ROUNDS`` timed calls per side — 20 timed calls against the
#: full protocol's 100, i.e. ~5x cheaper wall-clock.  ``PROBE_MOVE_COST
#: = MOVE_COST / 5`` prices that difference instead of pretending a
#: screen costs a measurement.  Probes are *screening only*: their
#: records live in ``DeliveryState.probes``, never in ``results``,
#: and can never be committed — the shipped artifact always comes
#: from a full paired-round evaluation (the "confirm" move) or eager.
PROBE_MOVE_COST = MOVE_COST / 5.0

#: Probe-screen win threshold: a probe arm confirms (full-evaluates)
#: the probe-argmin config only when its own paired-eager/config
#: speedup clears this bar.  1.10 sits well above the WIN_MARGIN
#: noise floor (1.02) and the ~±10% noise of a 1-round/10-iter
#: screen, while every real winner in the v7 corpus is ≥1.7x — so
#: predicted-eager boards skip the confirm entirely and ship eager
#: at probe-cost only.  A true sub-1.10 "win" is inside the same
#: noise band WIN_MARGIN already refuses to count.
PROBE_CONFIRM_MARGIN = 1.10

#: Runner configurations in playbook preference order — the scripted
#: line tries the richest honest composition first.  ``eager`` is
#: deliberately absent: it is the measured baseline and the commit
#: fallback, not a move a player can spend budget on.
CONFIGS: tuple[str, ...] = (
    "fuse+graph-alias",
    "fuse+graph",
    "graph-alias",
    "graph",
    "fuse",
    "compile+graph",
    "compile",
)


class GameUnavailable(RuntimeError):
    """The game path cannot run — gamekit absent or player unseated.

    ``optimize_model`` catches this and falls back to the heuristic
    ``deliver`` with an honest note.
    """


# ---------------------------------------------------------------------------
#  Context + state
# ---------------------------------------------------------------------------


@dataclass
class _Context:
    """Everything a config evaluation needs — built once per delivery."""

    optz: Any
    res: Any
    lowered_eager: Any
    inputs: Any
    meta: dict[str, Any]
    bench_iters: int
    verify: bool
    #: ``config -> runner`` builder inputs resolved once.
    alias: Any = None
    mutable: Any = None
    fusible: bool = False
    cuda: bool = False
    compile_ok: bool = False
    lowered_by_config: dict[str, Any] = field(default_factory=dict)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Set when a config evaluation trips a sticky CUDA-context fault
    #: (device-side assert / illegal access): every LATER config would
    #: inherit the poisoned context and record a bogus ``lower
    #: failed`` — corrupting oracle labels silently (the arch:t5-small
    #: ``fuse+graph`` drain).  Once set, remaining configs are recorded
    #: as explicit skips, never attempted.
    poisoned_by: str | None = None
    #: Paired-round count for ``_bench_paired`` evaluations.
    paired_rounds: int = PAIRED_ROUNDS
    #: ``ModelSpec.kwargs`` — extra kwargs every benchmark forward
    #: carries (``module(*inputs, **call_kwargs)``).
    call_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeliveryState:
    """The read-only snapshot players decide on (``state.done`` ends).

    Carries ``steps`` (the ``LinearPolicy`` episode-boundary
    contract), ``pending`` (unevaluated configs), ``results`` (the
    referee's measurements so far) and the eligibility flags the
    featurizer broadcasts.
    """

    done: bool = False
    steps: int = 0
    pending: tuple[str, ...] = ()
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Probe-screen records — ``probe:<config>`` moves only, keyed by
    #: config name.  Screening evidence, NEVER committable: the
    #: commit path reads ``results`` alone.  Empty on boards built
    #: without ``probe_armed``.
    probes: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Probe moves spent — priced at ``PROBE_MOVE_COST`` each, not
    #: ``MOVE_COST`` (the screen is ~5x cheaper than a full eval).
    probe_steps: int = 0
    #: Whether this board offers ``probe:<config>`` actions — set at
    #: construction; ``legal_actions`` reads it.
    probe_armed: bool = False
    eager_min_ms: float | None = None
    eager_med_ms: float | None = None
    cuda: bool = False
    fusible: bool = False
    compile_ok: bool = False
    alias_spec: bool = False
    #: Workload-shape facts the featurizer broadcasts — populated
    #: once at board construction (cheap duck-typed reads, no
    #: measurement).  A delivery prior learns *which config kinds
    #: pay on which shapes*, so the state must carry the shape.
    n_inputs: int = 0
    n_enodes: int = 0
    n_params: int = 0
    lowprec: bool = False

    def _eager_for(self, name: str | None) -> tuple[float | None, float | None]:
        """``(min_ms, med_ms)`` eager baseline for scoring *name*.

        A config measured under the paired protocol carries the
        eager median from ITS OWN rounds (``rec["eager_*_ms"]``) —
        the drift-robust comparison the autopsy asked for.  Records
        without one (or ``name=None``) fall back to the
        board-construction baseline.
        """
        rec = self.results.get(name) if name is not None else None
        if rec is not None and rec.get("eager_med_ms") is not None:
            return rec.get("eager_min_ms"), rec["eager_med_ms"]
        return self.eager_min_ms, self.eager_med_ms

    @property
    def best_med_ms(self) -> float | None:
        """Median ms of the fastest *scored* config, or ``None``."""
        best: float | None = None
        for r in self.results.values():
            med = r.get("med_ms")
            if med is not None and (best is None or med < best):
                best = med
        return best

    @property
    def best_speedup(self) -> float:
        """``eager_med / best_verified_med`` — 1.0 = eager parity."""
        best = self.best_med_ms
        _emin, eager = self._eager_for(self.best_config)
        if best is None or not eager or best <= 0:
            return 1.0
        return eager / best

    @property
    def committed_speedup(self) -> float:
        """The speedup ``committed()`` would actually ship — the reward.

        ``eager_med / best_med`` only when the fastest scored config
        beats the eager baseline by ``WIN_MARGIN``; otherwise eager
        ships and the honest answer is 1.0.  Tying reward to the
        *committed* artifact means exploring a pool of slower configs
        can never score below the eager floor — only a real,
        noise-clearing win pays.
        """
        best = self.best_med_ms
        _emin, eager = self._eager_for(self.best_config)
        if best is None or not eager or best <= 0:
            return 1.0
        if best * WIN_MARGIN >= eager:
            return 1.0
        return eager / best

    @property
    def best_config(self) -> str | None:
        """Name of the fastest scored config, or ``None``."""
        best_name: str | None = None
        best: float | None = None
        for name, r in self.results.items():
            med = r.get("med_ms")
            if med is not None and (best is None or med < best):
                best, best_name = med, name
        return best_name


# ---------------------------------------------------------------------------
#  The board
# ---------------------------------------------------------------------------


def legal_actions(state: DeliveryState) -> tuple[str, ...]:
    """The legal set — pending configs, probes, and ``stop``.

    ``probe:<config>`` actions appear only on probe-armed boards
    (``state.probe_armed``) and only for configs still pending and
    not yet probed — each config probes at most once, and a full
    evaluation (the confirm) removes it from ``pending``.
    """
    acts = list(state.pending)
    if getattr(state, "probe_armed", False):
        acts.extend(f"probe:{name}" for name in state.pending if name not in state.probes)
    acts.append("stop")
    return tuple(acts)


class DeliveryBoard:
    """One delivery decision as an episode — drain configs, commit best.

    ``step(config)`` evaluates that configuration *now* (re-lower +
    verify + bench — the referee's cost is real); ``step("stop")``
    ends the episode.  ``committed()`` is the delivery decision —
    always available, whether the episode stopped early, drained the
    pool, or the budget ran out mid-search.
    """

    def __init__(
        self,
        ctx: _Context,
        *,
        player_name: str = "?",
        budget: int = 0,
        probe_armed: bool = False,
    ) -> None:
        self._ctx = ctx
        self._player_name = player_name
        self._budget = budget
        self.state = DeliveryState(
            pending=_eligible(ctx),
            cuda=ctx.cuda,
            fusible=ctx.fusible,
            compile_ok=ctx.compile_ok,
            alias_spec=ctx.alias is not None or ctx.mutable is not None,
            probe_armed=probe_armed,
        )
        # The eager baseline is measured once at board construction —
        # it is the shipped fallback's number anyway, and every
        # speedup the game reports is relative to it.
        emin, emed = _bench(
            ctx.lowered_eager.module,
            ctx.inputs,
            ctx.bench_iters,
            call_kwargs=ctx.call_kwargs,
        )
        self.state.eager_min_ms = emin
        self.state.eager_med_ms = emed
        _shape_facts(ctx, self.state)

    # -- Board protocol -------------------------------------------------

    def observe(self) -> DeliveryState:
        """Return the read-only state snapshot."""
        return self.state

    def step(self, action: Any) -> tuple[DeliveryState, Any]:
        """Apply a move — evaluate a config, or ``stop`` to commit."""
        from gamekit.engine import Report  # lazy: gamekit is the game layer

        ctx = self._ctx
        st = self.state
        if action == "stop":
            st.done = True
            # The reward is the *committed* speedup — a config only
            # counts when it clears the noise margin over eager;
            # otherwise the artifact that ships is eager (1.0x).
            speedup = st.committed_speedup
            best = st.best_config if speedup > 1.0 else None
            # Move cost: reward = committed speedup − MOVE_COST·steps
            # − PROBE_MOVE_COST·probe_steps — a full drain must not
            # score like an instant stop, or the prior learns argmax
            # but never efficient stopping.  Probe screens are priced
            # at their real ~5x-cheaper instrument cost.
            reward = speedup - MOVE_COST * st.steps - PROBE_MOVE_COST * st.probe_steps
            note = (
                f"deliver-game: committed {best} (speedup {speedup:.3f}x vs eager)"
                if best is not None
                else "deliver-game: no config beat eager — eager ships"
            )
            return st, Report(
                action=action,
                reward=reward,
                terminal=True,
                note=note,
                detail={
                    "best_config": best,
                    "speedup": speedup,
                    "steps": st.steps,
                    "probe_steps": st.probe_steps,
                    "move_cost": MOVE_COST * st.steps + PROBE_MOVE_COST * st.probe_steps,
                },
            )
        # ``probe:<config>`` — the screening move: same lower +
        # mandatory verify gate, ~5x cheaper bench, record lands in
        # ``state.probes`` and can NEVER be committed.  Only legal on
        # probe-armed boards for still-pending, not-yet-probed
        # configs.
        if isinstance(action, str) and action.startswith("probe:"):
            name = action[len("probe:") :]
            if not st.probe_armed or name not in st.pending or name in st.probes:
                st.probe_steps += 1
                return st, Report(
                    action=action,
                    reward=0.0,
                    applied=False,
                    note=f"deliver-game: {action!r} not a legal probe — ignored",
                )
            st.probe_steps += 1
            rec = _evaluate(ctx, name, probe=True)
            st.probes[name] = rec
            # A probe's lowered module is screening scaffolding —
            # never retained for commit (the confirm re-lowers).
            ctx.lowered_by_config.pop(name, None)
            status = (
                f"probe med={rec['med_ms']:.3f}ms"
                if rec.get("med_ms") is not None
                else f"probe declined ({rec.get('reason', '?')})"
            )
            return st, Report(
                action=action,
                reward=0.0,
                note=f"deliver-game: {action} — {status}",
                detail={"config": name, "record": rec, "probe": True},
            )
        if action not in st.pending:
            st.steps += 1
            return st, Report(
                action=action,
                reward=0.0,
                applied=False,
                note=f"deliver-game: {action!r} not pending — ignored",
            )
        st.pending = tuple(c for c in st.pending if c != action)
        st.steps += 1
        rec = _evaluate(ctx, action)
        st.results[action] = rec
        # Bound ``lowered_by_config`` retention: only the current
        # argmin's lowered module can ever ship at ``committed()`` —
        # every LowerResult pins a full module's weights, so keeping
        # the whole pool is unbounded retention for no benefit.
        best = st.best_config
        keep = ctx.lowered_by_config.pop(best, None) if best is not None else None
        ctx.lowered_by_config.clear()
        if best is not None and keep is not None:
            ctx.lowered_by_config[best] = keep
        status = (
            f"med={rec['med_ms']:.3f}ms"
            if rec.get("med_ms") is not None
            else f"declined ({rec.get('reason', '?')})"
        )
        return st, Report(
            action=action,
            reward=0.0,
            note=f"deliver-game: {action} — {status}",
            detail={"config": action, "record": rec},
        )

    # -- The commit ------------------------------------------------------

    def committed(self) -> Delivery:
        """Return the :class:`Delivery` the episode decided.

        The winner is the fastest *scored* config by measured median
        ms — but only if it beats the eager baseline; ties and
        slowdowns ship eager.  A config is scored only when it
        lowered AND passed the verify gate (or verify was off — noted
        honestly).  Failed-verify configs are declines, never winners.
        """
        ctx = self._ctx
        st = self.state
        notes: list[str] = [
            "deliver: mechanism=game "
            f"(player={self._player_name}, "
            f"budget={self._budget} moves; "
            "runner selection is measured, not heuristed)"
        ]
        for name in CONFIGS:
            rec = st.results.get(name)
            if rec is None:
                continue
            if rec.get("med_ms") is not None:
                notes.append(
                    f"deliver-game: {name} — med={rec['med_ms']:.3f}ms "
                    f"min={rec['min_ms']:.3f}ms "
                    f"runner={rec.get('runner_name')}"
                )
            else:
                notes.append(f"deliver-game: {name} — declined: {rec.get('reason')}")
        if st.probes:
            # Screening evidence, reported for audit — probes never
            # commit; only their confirm (a full eval in ``results``)
            # can ship.
            notes.append(
                "deliver-game: probes (screening only, never committed): "
                + ", ".join(
                    f"{name} "
                    + (
                        f"med={rec['med_ms']:.3f}ms"
                        if rec.get("med_ms") is not None
                        else f"declined ({rec.get('reason', '?')})"
                    )
                    for name, rec in st.probes.items()
                )
            )
        if not st.results:
            notes.append("deliver-game: no config evaluated — eager ships")

        best_name = st.best_config
        best_med = st.best_med_ms
        # Eager is always a candidate: a config must beat it by the
        # WIN_MARGIN noise floor — sub-margin "wins" are bench noise
        # on shared GPUs, not delivery.  The baseline is the winning
        # config's OWN paired-round eager median when the record
        # carries one (drift can't favor a config measured in a
        # luckier window); the construction baseline is the fallback.
        eager_min, eager_med = st._eager_for(best_name)
        if (
            best_name is None
            or best_med is None
            or eager_med is None
            or best_med * WIN_MARGIN >= eager_med
        ):
            if best_name is not None:
                notes.append(
                    f"deliver-game: best config {best_name} "
                    f"({best_med:.3f}ms) does not beat eager "
                    f"({eager_med:.3f}ms) by the {WIN_MARGIN}x noise "
                    "margin — eager ships"
                )
            return Delivery(
                lowered=ctx.lowered_eager,
                runner_name=None,
                eager_min_ms=eager_min,
                eager_med_ms=eager_med,
                notes=notes,
                mechanism="game",
            )

        lowered_d = ctx.lowered_by_config[best_name]
        stats = getattr(lowered_d, "stats", {}) or {}
        _ledger_notes(stats, notes)
        if st.results[best_name].get("unverified"):
            notes.append(
                f"deliver-game: {best_name} shipped WITHOUT the equivalence gate (--no-verify)"
            )
        notes.append(
            "deliver-game: timings are min-over-paired-rounds "
            f"({max(1, int(ctx.paired_rounds))} rounds x "
            f"{max(1, int(ctx.bench_iters) // max(1, int(ctx.paired_rounds)))} "
            f"iters/side) on this (possibly shared) GPU; verify gated "
            "every scored config"
        )
        return Delivery(
            lowered=lowered_d,
            runner_name=best_name,
            eager_min_ms=eager_min,
            eager_med_ms=eager_med,
            delivered_min_ms=st.results[best_name]["min_ms"],
            delivered_med_ms=st.results[best_name]["med_ms"],
            notes=notes,
            mechanism="game",
        )


# ---------------------------------------------------------------------------
#  Config space — eligibility probes + the referee
# ---------------------------------------------------------------------------


def _probe(ctx: _Context) -> None:
    """Resolve the cheap eligibility facts once (no measurement)."""
    ctx.cuda = _has_cuda(ctx.lowered_eager.module, ctx.inputs)
    term = getattr(ctx.res, "term", None)
    fusible, _detail = _fusible(term, ctx.lowered_eager.module)
    ctx.fusible = fusible
    try:
        from kit_torch.runners import TorchCompileRunner  # noqa: F401

        ctx.compile_ok = True
    except Exception:  # noqa: BLE001 — compile unavailable is a decline
        ctx.compile_ok = False
    if ctx.cuda:
        alias, _a = _alias_spec(ctx.meta, "alias_inputs")
        mutable, _m = _alias_spec(ctx.meta, "mutable_inputs")
        ctx.alias, ctx.mutable = alias, mutable


def _eligible(ctx: _Context) -> tuple[str, ...]:
    """The configs this board can honestly offer, in preference order."""
    out = []
    alias_spec = ctx.alias is not None or ctx.mutable is not None
    for name in CONFIGS:
        parts = name.split("+")
        if "fuse" in parts and not ctx.fusible:
            continue
        if "compile" in parts and not ctx.compile_ok:
            continue
        if "graph" in parts and not ctx.cuda:
            continue
        if "graph-alias" in parts and not alias_spec:
            continue
        out.append(name)
    return tuple(out)


def _runner_for(ctx: _Context, name: str) -> Any:
    """Build the runner (or ``ChainedRunner``) for config *name*."""
    members: list[Any] = []
    for part in name.split("+"):
        if part == "fuse":
            from kit_torch.runners import RegionFusionRunner

            members.append(RegionFusionRunner())
        elif part == "compile":
            from kit_torch.runners import TorchCompileRunner

            members.append(TorchCompileRunner())
        elif part == "graph":
            from kit_cuda import CudaGraphRunner

            members.append(CudaGraphRunner())
        elif part == "graph-alias":
            from kit_cuda import CudaGraphRunner

            members.append(CudaGraphRunner(alias_inputs=ctx.alias, mutable_inputs=ctx.mutable))
        else:  # pragma: no cover — eligibility guards this
            raise ValueError(f"unknown config part {part!r}")
    if len(members) == 1:
        return members[0]
    from mlopt_orchestrator.runners import ChainedRunner

    return ChainedRunner(members)


def _shape_facts(ctx: _Context, st: DeliveryState) -> None:
    """Fill the state's workload-shape fields — cheap, no measurement.

    Duck-typed reads only: input count/dtype/device off the example
    inputs, ``n_enodes`` off the search result, parameter count off
    the eager module's ``parameters()`` when it has one.  Every read
    is guarded — a shape fact that cannot be read stays 0/False
    rather than sinking the board.
    """
    try:
        inputs = ctx.inputs if isinstance(ctx.inputs, (tuple, list)) else (ctx.inputs,)
        st.n_inputs = len(inputs)
        first = inputs[0] if inputs else None
        dtype = str(getattr(first, "dtype", ""))
        st.lowprec = "bfloat16" in dtype or "float16" in dtype
    except Exception:  # noqa: BLE001 — shape facts never sink a board
        pass  # nosec B110
    with contextlib.suppress(Exception):
        # ``SearchResult`` has no ``n_enodes`` — the count lives on the
        # hash-consed term DAG (``res.term`` is an ``mlopt_core.term.Op``;
        # equal subterms are the same object, so ``id`` memoises the
        # walk).  Count Op nodes only — Var/Const/Param leaves are not
        # enodes.  Without this the ``st:n_enodes`` feature was
        # permanently 0.0.
        from mlopt_core.term import Op as _IRop  # lazy: keeps torch-free import path

        term = getattr(ctx.res, "term", None)
        if term is not None:
            seen: set[int] = set()
            stack = [term]
            n = 0
            while stack and n < 1_000_000:
                node = stack.pop()
                if not isinstance(node, _IRop) or id(node) in seen:
                    continue
                seen.add(id(node))
                n += 1
                stack.extend(a for a in node.args if isinstance(a, _IRop))
            st.n_enodes = n
        else:
            st.n_enodes = int(getattr(ctx.res, "n_enodes", 0) or 0)
    with contextlib.suppress(Exception):
        params = getattr(ctx.lowered_eager.module, "parameters", None)
        if callable(params):
            st.n_params = sum(int(getattr(p, "numel", lambda: 0)()) for p in params())


#: Sticky CUDA-context fault signatures — once one of these fires,
#: EVERY subsequent CUDA call on the context fails the same way, so
#: sibling config measurements are meaningless.  Matched on the
#: exception text (``torch.AcceleratorError`` carries the driver
#: message); deliberately narrow — a plain OOM or launch error is NOT
#: context-poisoning and must not skip siblings.
_POISON_MARKERS = (
    "device-side assert",
    "illegal memory access",
    "sticky error",
)


def _is_context_poison(exc: BaseException) -> bool:
    """Return True when *exc* signals a poisoned CUDA context."""
    text = str(exc).lower()
    return any(m in text for m in _POISON_MARKERS)


def _evaluate(ctx: _Context, name: str, *, probe: bool = False) -> dict[str, Any]:
    """The referee: lower under the config's runner, verify, bench.

    A config that fails to lower, fails verify, or raises is a
    *decline* — recorded with its reason, never scored, never able
    to win.  ``verify=False`` configs are measured but flagged
    unverified (the ``--no-verify`` honesty contract).  A config
    evaluated AFTER a context-poisoning failure is a recorded
    *skip* — the measurement would inherit the dead context and its
    ``lower failed`` record would be a lie about the config itself.

    ``probe=True`` runs the screening protocol: the SAME lower +
    verify gate, but ``paired_rounds=1`` with ``bench_iters //
    PAIRED_ROUNDS`` timed calls per side (~5x cheaper than the full
    ``paired_rounds`` x ``bench_iters // paired_rounds`` protocol).
    The record is marked ``"probe": True`` and is screening evidence
    only — it never enters ``state.results`` and can never be
    committed.
    """
    if ctx.poisoned_by is not None:
        return {
            "config": name,
            "reason": (f"skipped: CUDA context poisoned by {ctx.poisoned_by!r} — not evaluated"),
            "skipped": True,
        }
    try:
        runner = _runner_for(ctx, name)
    except Exception as e:  # noqa: BLE001 — runner build is a decline
        return {"config": name, "reason": f"runner build failed: {e}"}
    runner_name = getattr(runner, "name", type(runner).__name__)
    try:
        lowered = ctx.optz.lower(ctx.res, ctx.inputs, verify=ctx.verify, runner=runner)
    except Exception as e:  # noqa: BLE001 — report honestly
        if _is_context_poison(e):
            ctx.poisoned_by = name
        return {
            "config": name,
            "runner_name": runner_name,
            "reason": f"lower failed: {type(e).__name__}: {e}",
        }
    ctx.lowered_by_config[name] = lowered
    stats = getattr(lowered, "stats", {}) or {}
    verified = getattr(lowered, "verified", None)
    passed = None if verified is None else bool(getattr(verified, "passed", True))
    if ctx.verify and passed is not True:
        # verify=True makes the gate MANDATORY for scoring: an
        # explicit failure AND a missing gate are both declines —
        # a config that cannot prove equivalence never wins.
        reason = (
            "no verify report"
            if verified is None
            else f"FAILED verify (max_abs={getattr(verified, 'max_abs', '?')})"
        )
        return {
            "config": name,
            "runner_name": runner_name,
            "reason": reason,
            "stats": stats,
        }
    try:
        # Paired protocol: ``bench_iters`` timed calls per side are
        # split across ``paired_rounds`` interleaved eager/config
        # round-pairs (order randomized per round, seeded per config
        # name for determinism).  The record carries BOTH sides — the
        # config's min-over-round-mins score AND the eager median
        # measured in the same rounds, so ``committed_speedup`` and
        # the oracle compare against a drift-matched baseline rather
        # than the construction-time one.  A probe runs ONE round at
        # the same per-round iter count — ~5x fewer timed calls, a
        # screening measurement, never a commit candidate.
        if probe:
            rounds = 1
            iters_per_round = max(4, int(ctx.bench_iters) // PAIRED_ROUNDS)
        else:
            rounds = max(1, int(ctx.paired_rounds))
            iters_per_round = max(1, int(ctx.bench_iters) // rounds)
        paired = _bench_paired(
            ctx.lowered_eager.module,
            lowered.module,
            ctx.inputs,
            iters_per_round,
            rounds=rounds,
            call_kwargs=ctx.call_kwargs,
            seed=zlib.crc32(name.encode()),
        )
    except Exception as e:  # noqa: BLE001 — a bench crash is a decline
        if _is_context_poison(e):
            ctx.poisoned_by = name
        return {
            "config": name,
            "runner_name": runner_name,
            "reason": f"bench failed: {type(e).__name__}: {e}",
            "stats": stats,
        }
    eager_side, config_side = paired["eager"], paired["config"]
    rec: dict[str, Any] = {
        "config": name,
        "runner_name": runner_name,
        "min_ms": config_side["min_ms"],
        "med_ms": config_side["med_ms"],
        "eager_min_ms": eager_side["min_ms"],
        "eager_med_ms": eager_side["med_ms"],
        "rounds": [
            {"eager": [float(e[0]), float(e[1])], "config": [float(c[0]), float(c[1])]}
            for e, c in zip(eager_side["rounds"], config_side["rounds"], strict=True)
        ],
        "bench": {"rounds": rounds, "iters_per_round": iters_per_round},
        "verified": passed,
        "stats": stats,
    }
    if probe:
        rec["probe"] = True
    if not ctx.verify:
        rec["unverified"] = True
    return rec


# ---------------------------------------------------------------------------
#  The game driver
# ---------------------------------------------------------------------------


def _gamekit() -> Any:
    """Import gamekit lazily — its absence is an honest decline."""
    try:
        import gamekit.engine  # noqa: F401
        import gamekit.players.registry  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 — gamekit absent is a valid decline
        return False


def deliver_game(
    *,
    optz: Any,
    res: Any,
    lowered_eager: Any,
    inputs: Any,
    meta: dict[str, Any] | None = None,
    bench_iters: int = 50,
    verify: bool = True,
    budget: int = 6,
    player: str = "playbook",
    seed: int = 0,
    call_kwargs: dict[str, Any] | None = None,
    probe: bool = False,
) -> Delivery:
    """Run the delivery game and return the committed :class:`Delivery`.

    One episode on one :class:`DeliveryBoard`: the named registry
    player spends up to *budget* moves evaluating runner configs; the
    commit is the fastest verified measurement — eager included.
    ``probe=True`` arms the board's ``probe:<config>`` screening
    moves (cheap 1-round bench, same lower+verify gate, never
    committable) for the probe-then-commit arms — off by default so
    the shipped game is unchanged.
    Raises :class:`GameUnavailable` when gamekit is absent or the
    player is not seated on the ``deliver`` board; the caller falls
    back to the heuristic path with a note.
    """
    if not _gamekit():
        raise GameUnavailable("gamekit is not installed — game path unavailable")
    import mlopt.play  # noqa: PLC0415 — registers the deliver seats

    mlopt.play.register()
    from gamekit.engine import run_episode
    from gamekit.players import registry

    ctx = _Context(
        optz=optz,
        res=res,
        lowered_eager=lowered_eager,
        inputs=inputs,
        meta=meta or {},
        bench_iters=bench_iters,
        verify=verify,
        call_kwargs=dict(call_kwargs or {}),
    )
    _probe(ctx)
    board = DeliveryBoard(ctx, player_name=player, budget=budget, probe_armed=probe)
    try:
        arm = registry.make(player, "deliver", seed=seed)
    except Exception as e:  # noqa: BLE001 — unknown/unseated player
        raise GameUnavailable(f"player {player!r} cannot play 'deliver': {e}") from e
    t0 = time.time()
    traj = run_episode(board, arm(0), max(1, int(budget)))
    elapsed = time.time() - t0
    notes_extra = [
        f"deliver-game: {player} spent {len(traj.reports)} moves "
        f"in {elapsed:.1f}s (budget {budget})"
    ]
    out = board.committed()
    out.notes[1:1] = notes_extra  # after the mechanism line
    return out

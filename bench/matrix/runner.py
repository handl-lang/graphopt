# ruff: noqa: RUF002 — math notation in prose.

"""The matrix runner — players × boards × referees, designed subset.

One cell ``(player, board, referee)`` asks: *the claims this player
produces on this board — which survive this referee?*  The boards
supply two claim kinds:

* **extraction claims** (the episode boards — ``meta``, ``search``,
  ``torch``, ``joint``, ``gen``): the episode's terminal artifact
  ``(input term, extracted term)`` wrapped as a concrete
  ``Rewrite``.  The referee re-judges the delivered equality —
  ``naive`` trusts it, ``oracle`` evals it, ``pay`` prices it on
  every axis, ``sweep`` redraws its envs, ``cert`` asks
  ``verify_law`` for a standalone derivation.
* **candidate claims** (``open``): the candidates the player chose
  to evaluate, under the memoized-verdict protocol (identical
  per-candidate oracle draws replayed to every arm — the
  ``bench.players.runner`` machinery).  The cell's ``refuted`` count
  is the ablation made a row: picks the board's oracle called
  ``useful`` that the referee refuses.

``contraction`` and ``player`` produce no equality claims — the
coverage audit marks those cells ``unrun`` with the reason, never a
silent drop.  Every unique claim is judged once per referee per run
through a shared :class:`ClaimEvidence` — the memoized-verdict seam
that dedups the expensive legs across cells.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

from mlopt_core.egraph import Rewrite

from bench.referees import components as _comp
from bench.referees import registry as refs

#: ``play``'s held-out seed offset — same eval stream the leaderboard uses.
HELD_OUT_OFFSET = 7919

#: Boards whose terminal artifact is an extraction claim.
_EXTRACTION_BOARDS = ("meta", "joint", "torch", "search", "gen")

#: Boards that produce no equality claims — coverage marks them unrun.
_NO_CLAIM_BOARDS = ("contraction", "player")


@dataclass
class Cell:
    """One ``(player, board, referee)`` outcome."""

    board: str
    player: str
    referee: str
    run: bool = True
    reason: str = ""  # why the cell did not run
    claims: int = 0
    admitted: int = 0
    refuted: int = 0  # picks the board called useful, refused here
    value: float | None = None  # mean scored value over admitted claims
    reward_ep: float | None = None
    evals_ep: float | None = None
    wall_s: float = 0.0
    notes: str = ""

    @property
    def admit_rate(self) -> float | None:
        """Admitted / claims — ``None`` when the cell ran nothing."""
        if not self.claims:
            return None
        return self.admitted / self.claims


def _extracted_pair(board: Any, domain: str) -> tuple[Any, Any] | None:
    """The ``(input, extracted)`` term pair a finished board delivers."""
    if domain == "search":
        best = board.env._extract()
        return (board.env.term, best) if best is not None else None
    # MetaArena-family boards (meta/joint/torch/gen).
    best = board.eg.extract_best(board.root, board.feasible_cost)
    return (board.term, best) if best is not None else None


def _claim_key(claim: Rewrite) -> tuple:
    """Stable claim identity — same equality, same verdict."""
    from mlopt_core.ir import op_repr

    return (op_repr(claim.lhs), op_repr(claim.rhs))


def _judge_claims(
    referee: str,
    claims: dict[tuple, Rewrite],
    ctx: refs.RefereeCtx,
    evds: dict,
) -> tuple[dict[tuple, bool], list[float], list[str]]:
    """Judge every claim under *referee*; reuse the shared evidence.

    Returns ``({key: admitted}, scored values, refusal notes)`` — the
    per-claim map lets a cell count the board-blessed picks the
    referee refused.
    """
    verdicts: dict[tuple, bool] = {}
    values: list[float] = []
    refusals: list[str] = []
    for key, claim in claims.items():
        evd = evds.setdefault(key, _comp.ClaimEvidence(claim, ctx))
        evd.measured.clear()  # measurement is per-referee context
        try:
            v = refs.judge(referee, claim, ctx, evidence=evd)
        except refs.SkipReferee as e:
            refusals.append(str(e))
            continue
        verdicts[key] = v.passed
        if not v.passed:
            refusals.append(v.reason)
        if v.value == v.value:  # not nan
            values.append(v.value)
    return verdicts, values, refusals


def _episode_cells(
    domain: str,
    board: str,
    players: list[str],
    referees: list[str],
    *,
    seed: int,
    cases: int,
    budget: int,
    ctx: refs.RefereeCtx,
) -> list[Cell]:
    """Play each player on held-out cases; referee the extractions."""
    from mlopt_discovery import engine
    from mlopt_discovery.play import DOMAINS
    from mlopt_discovery.players import registry

    d = DOMAINS[board]
    ev = d.cases(seed + HELD_OUT_OFFSET, cases)
    rows: list[Cell] = []
    for pname in players:
        t0 = time.perf_counter()
        try:
            mk = registry.make(pname, board, seed=seed, ctx={})
        except Exception as e:
            for rname in referees:
                rows.append(
                    Cell(
                        board,
                        pname,
                        rname,
                        run=False,
                        reason=f"make: {e}",
                    )
                )
            continue
        claims: dict[tuple, Rewrite] = {}
        rewards: list[float] = []
        evals: list[int] = []
        for i, case in enumerate(ev):
            board_obj = d.board_of(case)
            traj = engine.run_episode(board_obj, mk(i), budget)
            rewards.append(traj.total)
            evals.append(len(traj.reports))
            pair = _extracted_pair(board_obj, board)
            if pair is None or pair[0] == pair[1]:
                continue  # the player changed nothing — no claim
            claim = Rewrite(f"{pname}@{board}:{i}", pair[0], pair[1])
            claims.setdefault(_claim_key(claim), claim)
        wall = time.perf_counter() - t0
        evds: dict = {}
        for rname in referees:
            t1 = time.perf_counter()
            verdicts, values, _r = _judge_claims(
                rname, claims, ctx, evds
            )
            n = len(claims)
            admitted = sum(verdicts.values())
            c = Cell(
                board,
                pname,
                rname,
                claims=n,
                admitted=admitted,
                value=(
                    round(statistics.fmean(values), 4)
                    if values
                    else None
                ),
                reward_ep=round(sum(rewards) / max(len(rewards), 1), 4),
                evals_ep=round(sum(evals) / max(len(evals), 1), 3),
                wall_s=round(wall + time.perf_counter() - t1, 2),
            )
            if not n:
                c.notes = "no claims — the player changed nothing"
            elif _r and admitted == n:
                c.notes = f"skipped: {_r[0]}"
            rows.append(c)
    return rows


def _open_cells(
    players: list[str],
    referees: list[str],
    *,
    seed: int,
    pages: int,
    page: int,
    budget: int,
    ctx: refs.RefereeCtx,
) -> list[Cell]:
    """Play each player on the open board; referee its chosen claims.

    Reuses the leaderboard's memoized-verdict protocol — every pool
    member measured once, the draws replayed to every arm — and adds
    the referee layer on top: the candidates a player *chose* are the
    claims its cell answers for.
    """
    from mlopt_discovery import engine
    from mlopt_discovery import open_game as og
    from mlopt_discovery import proposal as lp
    from mlopt_discovery.players import registry

    from bench.players.runner import (
        _cand_key,
        _measure_pools,
        _patched_evaluate,
    )

    ev = og.gen_cases(seed, pages, page=page, start=1, step=2)
    ev_pools = [list(og.OpenArena(c).pool) for c in ev]
    _outcomes, by_key = _measure_pools(ev_pools, seed + HELD_OUT_OFFSET)
    # The pool's own verdicts — "useful" under the board's oracle.
    useful = {
        k for k, o in by_key.items() if getattr(o, "useful", False)
    }
    orig_evaluate = lp.evaluate
    rows: list[Cell] = []
    try:
        for pname in players:
            t0 = time.perf_counter()
            try:
                mk = registry.make(pname, "open", seed=seed, ctx={})
            except Exception as e:
                for rname in referees:
                    rows.append(
                        Cell(
                            "open",
                            pname,
                            rname,
                            run=False,
                            reason=f"make: {e}",
                        )
                    )
                continue
            chosen: dict[tuple, Rewrite] = {}
            rewards: list[float] = []
            evals: list[int] = []

            def recording_evaluate(
                cands: Any, _chosen=chosen, _v=None, **kw: Any
            ) -> list:
                out = _patched_evaluate(by_key)(cands)
                for c in cands:
                    k = _cand_key(c)
                    _chosen.setdefault(k, c.as_rule())
                return out

            lp.evaluate = recording_evaluate
            for i, case in enumerate(ev):
                traj = engine.run_episode(
                    og.OpenArena(case), mk(i), budget
                )
                rewards.append(traj.total)
                evals.append(len(traj.reports))
            lp.evaluate = _patched_evaluate(by_key)
            wall = time.perf_counter() - t0
            evds: dict = {}
            for rname in referees:
                t1 = time.perf_counter()
                verdicts, values, _r = _judge_claims(
                    rname, chosen, ctx, evds
                )
                n = len(chosen)
                admitted = sum(verdicts.values())
                refuted = sum(
                    1
                    for k, ok in verdicts.items()
                    if k in useful and not ok
                )
                c = Cell(
                    "open",
                    pname,
                    rname,
                    claims=n,
                    admitted=admitted,
                    refuted=refuted,
                    value=(
                        round(statistics.fmean(values), 4)
                        if values
                        else None
                    ),
                    reward_ep=round(
                        sum(rewards) / max(len(rewards), 1), 4
                    ),
                    evals_ep=round(sum(evals) / max(len(evals), 1), 3),
                    wall_s=round(wall + time.perf_counter() - t1, 2),
                )
                if refuted:
                    c.notes = (
                        f"{refuted} pick(s) the oracle called useful "
                        f"refused by {rname}"
                    )
                elif not n:
                    c.notes = "no claims chosen"
                rows.append(c)
    finally:
        lp.evaluate = orig_evaluate
    return rows


def coverage(
    boards: list[str],
    players: list[str],
    referees: list[str],
) -> list[Cell]:
    """The full cell space — every (player, board, referee) marked.

    ``run=False`` cells carry the reason the designed subset does
    not run them: an unsupported board, or a board whose episodes
    produce no equality claims.
    """
    from mlopt_discovery.play import DOMAINS
    from mlopt_discovery.players import registry

    out: list[Cell] = []
    for board in boards:
        for pname in players:
            for rname in referees:
                c = Cell(board, pname, rname)
                if board in _NO_CLAIM_BOARDS:
                    c.run = False
                    c.reason = (
                        "no equality claims — the board scores cost, "
                        "not equalities"
                    )
                elif board != "open" and board not in DOMAINS:
                    c.run = False
                    c.reason = "unknown board"
                elif board not in registry.spec(pname).boards:
                    c.run = False
                    c.reason = "player does not play this board"
                out.append(c)
    return out


def run_matrix(
    boards: list[str],
    players: list[str],
    referees: list[str],
    *,
    seed: int = 0,
    meta_cases: int = 4,
    meta_budget: int = 8,
    open_pages: int = 2,
    open_page: int = 8,
    open_budget: int = 9,
    ctx: refs.RefereeCtx | None = None,
) -> list[Cell]:
    """Run the designed subset; return cells in request order."""
    ctx = ctx or refs.RefereeCtx()
    out: list[Cell] = []
    for board in boards:
        playable = [
            p
            for p in players
            if board
            in __import__(
                "mlopt_discovery.players.registry",
                fromlist=["spec"],
            )
            .spec(p)
            .boards
        ]
        if board in _NO_CLAIM_BOARDS:
            for p in players:
                for r in referees:
                    out.append(
                        Cell(
                            board,
                            p,
                            r,
                            run=False,
                            reason="no equality claims",
                        )
                    )
            continue
        if board == "open":
            out.extend(
                _open_cells(
                    playable,
                    referees,
                    seed=seed,
                    pages=open_pages,
                    page=open_page,
                    budget=open_budget,
                    ctx=ctx,
                )
            )
            continue
        out.extend(
            _episode_cells(
                board,
                board,
                playable,
                referees,
                seed=seed,
                cases=meta_cases,
                budget=meta_budget,
                ctx=ctx,
            )
        )
    return out

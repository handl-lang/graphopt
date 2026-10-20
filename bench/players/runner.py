"""Per-board comparison drivers — one protocol per board family.

Three board families, three honest protocols:

* **episode boards** (``meta``, ``search``, ``joint``, ``torch``,
  ``gen``, ``player``) — every arm plays the same held-out cases
  (``d.cases(seed + 7919, n)``) under ``engine.evaluate``; the row is
  reward/ep, evals/ep (steps), terminal cost, wall time.
* **``open``** — the ``bench/policy/policy_baseline.py`` protocol, reproduced:
  every candidate in every held-out pool is measured *once* under a
  per-candidate ``torch.manual_seed``, the verdicts memoized, and
  every arm's episodes run under a patched ``lp.evaluate`` replaying
  them — literally identical oracle draws for every arm.  ``neural``
  additionally measures the held-in (even) pages to train its
  ``RuleValueNet`` on, exactly as the baseline does.  The row adds
  useful/ep, useful/eval and page-ranking AP (``ctx["score_fn"]``
  deposit, or the base rate for the draw-free arms).
* **``contraction``** — the ordering game, scored by final pairwise
  cost: seeded ``random_bond_network`` instances, each player's
  ``(tensors, sizes) -> cost`` arm plus the ``opt_einsum`` baselines
  (``oe-greedy`` always when installed, ``oe-rand-greedy`` at the
  same rollout count as ``distilled``).  oe orders are scored under
  *our* pair-cost model, so every number on the board is one metric;
  ``ratio`` is cost / oe-greedy-cost per instance.
"""

from __future__ import annotations

import statistics
import string
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from mlopt_discovery import engine
from mlopt_discovery.play import DOMAINS
from mlopt_discovery.players import registry
from mlopt_discovery.players.registry import PlayerSpec

#: ``play``'s held-out seed offset — the eval stream the shipped
#: priors were never trained on.
HELD_OUT_OFFSET = 7919

#: opt_einsum subscript alphabet (letters + digits: 62 distinct labels).
_ALPHA = string.ascii_letters + string.digits


@dataclass
class Row:
    """One leaderboard row — a player on a board under this protocol."""

    board: str
    player: str
    kind: str = ""
    reward_ep: float | None = None
    evals_ep: float | None = None
    useful_per_eval: float | None = None
    ap: float | None = None
    cost: float | None = None
    ratio: float | None = None  # contraction: cost / baseline cost
    wall_s: float = 0.0
    n: int = 0
    notes: str = ""


def _requested(
    board: str, players: list[str] | None
) -> list[PlayerSpec]:
    """Resolve *players* against the registry for *board*.

    ``None`` means every registered player that supports the board;
    an explicit list keeps unsupported names so the row can say
    ``does not play this board`` instead of silently dropping them.
    """
    if players is None:
        return registry.for_board(board)
    return [registry.spec(n) for n in players]


# ---------------------------------------------------------------------------
#  Episode boards — paired held-out cases under engine.evaluate
# ---------------------------------------------------------------------------


def _episode_notes(rws: list[dict], budget: int) -> str:
    """The qualitative footnotes the numbers alone don't carry."""
    notes: list[str] = []
    steps = [r["steps"] for r in rws]
    if steps and all(s >= budget for s in steps):
        notes.append("drains every episode's move budget")
    elif steps and statistics.fmean(steps) <= 2.5:
        notes.append("short lines — extracts early")
    certs = sum(1 for r in rws if r["cert"])
    if certs:
        notes.append(f"certified extraction on {certs}/{len(rws)}")
    return "; ".join(notes)


def run_episode_board(
    board: str,
    players: list[str] | None,
    *,
    seed: int,
    cases: int,
    budget: int,
) -> list[Row]:
    """Play every requested arm on the held-out cases of *board*."""
    d = DOMAINS[board]
    ev = d.cases(seed + HELD_OUT_OFFSET, cases)
    rows: list[Row] = []
    for s in _requested(board, players):
        row = Row(board=board, player=s.name, kind=s.kind)
        if board not in s.boards:
            row.notes = "does not play this board"
            rows.append(row)
            continue
        ctx: dict = {}
        try:
            mk = registry.make(s.name, board, seed=seed, ctx=ctx)
        except Exception as e:  # factory declined — honest row
            row.notes = f"make: {e}"
            rows.append(row)
            continue
        t0 = time.perf_counter()
        table = engine.evaluate(
            {"arm": mk}, ev, d.board_of, budget=budget
        )
        row.wall_s = round(time.perf_counter() - t0, 2)
        rws = table["arm"]
        row.n = len(rws)
        k = max(len(rws), 1)
        costs = [r["cost"] for r in rws if r["cost"] is not None]
        row.reward_ep = round(sum(r["reward"] for r in rws) / k, 4)
        row.evals_ep = round(sum(r["steps"] for r in rws) / k, 3)
        row.cost = round(statistics.fmean(costs), 2) if costs else None
        row.notes = _episode_notes(rws, budget)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
#  The open board — memoized per-candidate verdicts (policy_baseline)
# ---------------------------------------------------------------------------


def _cand_key(cand: Any) -> tuple:
    """Stable identity of a candidate — same proposal, same verdict."""
    return (cand.label, repr(cand.lhs), repr(cand.rhs))


def _measure_pools(
    pools: list[list], seed: int
) -> tuple[list[list], dict]:
    """Evaluate every pool member once under a per-candidate seed."""
    import torch
    from mlopt_discovery import proposal as lp

    by_key: dict = {}
    per_pool: list[list] = []
    idx = 0
    for pool in pools:
        outs: list = []
        for cand in pool:
            torch.manual_seed(seed + idx)
            idx += 1
            outcome = lp.evaluate([cand])[0]
            by_key.setdefault(_cand_key(cand), outcome)
            outs.append(outcome)
        per_pool.append(outs)
    return per_pool, by_key


def _patched_evaluate(by_key: dict) -> Any:
    """An ``lp.evaluate`` replaying the memoized verdicts."""

    def evaluate(cands: Any, **kw: Any) -> list:
        del kw
        return [by_key[_cand_key(c)] for c in cands]

    return evaluate


def _open_episode_stats(mk: Any, ev: list, budget: int) -> dict:
    """Play one arm over the held-out pages; tally yield and reward."""
    from mlopt_discovery import open_game as og

    found = evals = 0
    reward = 0.0
    for i, case in enumerate(ev):
        traj = engine.run_episode(og.OpenArena(case), mk(i), budget)
        reward += traj.total
        for r in traj.reports:
            evals += 1
            if r.note and r.note.split(":", 1)[0] == "useful":
                found += 1
    n = len(ev)
    return {
        "reward": reward / n,
        "useful_per_ep": found / n,
        "useful_per_eval": found / max(evals, 1),
        "evals_per_ep": evals / n,
        "found": found,
    }


def _ap(scores: list[float], outcomes: dict) -> float:
    """Average precision of a best-first ordering under *scores*."""
    from mlopt_discovery import proposal as lp

    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return lp._average_precision(order, outcomes)


def _open_notes(stats: dict, page: int) -> str:
    """Qualitative flags — drain artifacts, premature stops."""
    notes: list[str] = []
    if stats["evals_per_ep"] >= page - 1e-9:
        notes.append("drains every page — no stopping learned")
    elif stats["evals_per_ep"] <= 1.0:
        notes.append("stops after ~one eval — misses the pool")
    if stats["evals_per_ep"] and stats["found"] == 0:
        notes.append("no useful law found on the held-out pages")
    return "; ".join(notes)


def run_open(
    players: list[str] | None,
    *,
    seed: int,
    pages: int,
    page: int,
    budget: int,
    neural_epochs: int,
    neural_hidden: int,
    device: str,
) -> tuple[list[Row], dict]:
    """Play the requested arms on odd pages; rank the held-out pool.

    Returns ``(rows, meta)`` — *meta* carries the pool/useful counts
    and whether the neural train pools were measured (the expensive
    half of the protocol, skipped when ``neural`` is not requested).
    """
    from mlopt_discovery import open_game as og
    from mlopt_discovery import proposal as lp

    specs = _requested("open", players)
    want_neural = any(s.name == "neural" for s in specs)

    train_cases = og.gen_cases(seed, pages, page=page, start=0, step=2)
    ev = og.gen_cases(seed, pages, page=page, start=1, step=2)
    ev_pools = [list(og.OpenArena(c).pool) for c in ev]
    eval_outcomes, by_key = _measure_pools(
        ev_pools, seed + HELD_OUT_OFFSET
    )

    train_pairs: list[tuple] = []
    if want_neural:
        train_pools = [list(og.OpenArena(c).pool) for c in train_cases]
        train_outcomes, tk = _measure_pools(
            train_pools, seed + 1_000_003
        )
        by_key.update(tk)
        train_pairs = [
            (c, o)
            for pool, outs in zip(
                train_pools, train_outcomes, strict=True
            )
            for c, o in zip(pool, outs, strict=True)
        ]

    flat_pool = [c for pool in ev_pools for c in pool]
    flat_outcomes = {}
    pos = 0
    for outs in eval_outcomes:
        for o in outs:
            flat_outcomes[pos] = o
            pos += 1
    n_pos = sum(1 for o in flat_outcomes.values() if o.useful)
    meta = {
        "pool": len(flat_pool),
        "useful": n_pos,
        "random_ap": round(n_pos / max(len(flat_pool), 1), 4),
        "train_pairs": len(train_pairs),
    }

    rows: list[Row] = []
    orig_evaluate = lp.evaluate
    lp.evaluate = _patched_evaluate(by_key)
    try:
        for s in specs:
            row = Row(board="open", player=s.name, kind=s.kind)
            if "open" not in s.boards:
                row.notes = "does not play this board"
                rows.append(row)
                continue
            ctx: dict = {
                "train": train_pairs,
                "epochs": neural_epochs,
                "hidden": neural_hidden,
                "device": device,
            }
            try:
                mk = registry.make(s.name, "open", seed=seed, ctx=ctx)
            except Exception as e:
                row.notes = f"make: {e}"
                rows.append(row)
                continue
            t0 = time.perf_counter()
            stats = _open_episode_stats(mk, ev, budget)
            row.wall_s = round(time.perf_counter() - t0, 2)
            row.n = len(ev)
            row.reward_ep = round(stats["reward"], 4)
            row.evals_ep = round(stats["evals_per_ep"], 3)
            row.useful_per_eval = round(stats["useful_per_eval"], 4)
            scorer = ctx.get("score_fn")
            if scorer is not None:
                row.ap = round(
                    _ap([scorer(c) for c in flat_pool], flat_outcomes),
                    4,
                )
            else:
                row.ap = meta["random_ap"]
            row.notes = _open_notes(stats, page)
            rows.append(row)
    finally:
        lp.evaluate = orig_evaluate
    return rows, meta


# ---------------------------------------------------------------------------
#  The contraction board — final-order cost vs the oe baselines
# ---------------------------------------------------------------------------


def _to_einsum(tensors: Any, sizes: dict[int, int]) -> dict:
    """Map a ``(tensors, sizes)`` bond network onto einsum inputs.

    The lab probe's ``to_einsum``, reduced to what the oe players
    need: single-char subscripts, the open-leg output set and the
    opt_einsum ``(inputs, output, size_dict)`` triple.
    """
    labels = sorted(sizes)
    letters = {lab: _ALPHA[i] for i, lab in enumerate(labels)}
    counts = Counter(i for t in tensors for i in t)
    out = "".join(letters[i] for i in labels if counts[i] == 1)
    subs = ["".join(letters[i] for i in t) for t in tensors]
    return {
        "expr": ",".join(subs) + "->" + out,
        "shapes": [tuple(sizes[i] for i in t) for t in tensors],
        "inputs": [set(s) for s in subs],
        "output": set(out),
        "size_dict": {letters[i]: sizes[i] for i in labels},
    }


def _oe_module() -> Any:
    """Return ``opt_einsum`` or ``None`` (opt-in ``einsum`` group)."""
    try:
        import opt_einsum as oe

        return oe
    except ImportError:
        return None


def _oe_orders(
    e: dict, samples: int
) -> dict[str, list[tuple[int, int]]]:
    """The opt_einsum baseline orders for one instance (if installed)."""
    oe = _oe_module()
    if oe is None:
        return {}
    path, _info = oe.contract_path(
        e["expr"], *e["shapes"], shapes=True, optimize="greedy"
    )
    out = {"oe-greedy": [tuple(p) for p in path]}
    try:
        from opt_einsum.path_random import RandomGreedy

        opt = RandomGreedy(max_repeats=max(samples, 1))
        path = opt(e["inputs"], e["output"], e["size_dict"])
        out["oe-rand-greedy"] = [tuple(p) for p in path]
    except Exception:
        pass
    return out


def _contraction_notes(
    costs: list[float], floors: list[float], baseline: str
) -> str:
    """The ratio summary vs the field baseline, honestly counted."""
    pairs = [
        (c, f) for c, f in zip(costs, floors, strict=True) if f > 0
    ]
    if not pairs:
        return ""
    wins = sum(1 for c, f in pairs if c < f - 1e-9)
    ties = sum(1 for c, f in pairs if abs(c - f) <= 1e-9 * f)
    return (
        f"{wins} beats / {ties} ties / {len(pairs) - wins - ties} "
        f"loses vs {baseline}"
    )


def run_contraction(
    players: list[str] | None,
    *,
    seed: int,
    scales: tuple[int, ...],
    instances: int,
    samples: int,
    device: str,
) -> list[Row]:
    """Score every arm's contraction order vs the oe-greedy floor.

    Each ``(scale, instance)`` board is ``random_bond_network(n,
    seed + i)``; every player produces a cost under the shared
    pairwise model, and ``ratio`` is cost / oe-greedy-cost (or
    / cheapest-pair-greedy when ``opt_einsum`` is absent — the row's
    notes say which floor priced it).
    """
    from mlopt_torch.contraction_policy import (
        cost_of_order,
        greedy,
        random_bond_network,
    )

    specs = _requested("contraction", players)
    # One factory per player, built once — ``distilled`` loads its
    # artifact here, not per instance.
    arms: dict[str, Any] = {}
    per_arm: dict[str, dict] = {}
    for s in specs:
        if "contraction" not in s.boards:
            per_arm[s.name] = {"unsupported": True}
            continue
        ctx = {"samples": samples, "device": device, "seed": seed}
        try:
            arms[s.name] = registry.make(
                s.name, "contraction", seed=seed, ctx=ctx
            )
        except Exception as e:
            per_arm[s.name] = {"make_error": str(e)}

    floors: dict[tuple, float] = {}
    floor_name = "our-greedy"
    for n in scales:
        for i in range(instances):
            tensors, sizes = random_bond_network(n, seed + i)
            e = _to_einsum(tensors, sizes)
            base_orders = _oe_orders(e, samples)
            key = (n, i)
            if "oe-greedy" in base_orders:
                floors[key] = cost_of_order(
                    tensors, sizes, base_orders["oe-greedy"]
                )
                floor_name = "oe-greedy"
            else:
                floors[key] = greedy(tensors, sizes)

            for name, order in base_orders.items():
                t0 = time.perf_counter()
                c = cost_of_order(tensors, sizes, order)
                secs = time.perf_counter() - t0
                acc = per_arm.setdefault(
                    name, {"costs": [], "floors": [], "wall": 0.0}
                )
                acc["costs"].append(c)
                acc["floors"].append(floors[key])
                acc["wall"] += secs

            for name, mk in arms.items():
                play = mk(i)
                t0 = time.perf_counter()
                try:
                    c = play(tensors, sizes)
                except Exception as exc:
                    c = float("inf")
                    note = f" failed: {exc}"
                else:
                    note = ""
                secs = time.perf_counter() - t0
                acc = per_arm.setdefault(
                    name,
                    {"costs": [], "floors": [], "wall": 0.0},
                )
                acc["costs"].append(c)
                acc["floors"].append(floors[key])
                acc["wall"] += secs
                if note:
                    acc.setdefault("errors", []).append(note.strip())

    rows: list[Row] = []
    kind_of = {s.name: s.kind for s in registry.specs().values()}
    for name, acc in per_arm.items():
        row = Row(
            board="contraction",
            player=name,
            kind=(
                "external"
                if name.startswith("oe-")
                else kind_of.get(name, "")
            ),
        )
        if acc.get("unsupported") or acc.get("make_error"):
            row.notes = (
                acc.get("make_error") or "does not play this board"
            )
            rows.append(row)
            continue
        costs = acc["costs"]
        fl = acc["floors"]
        finite = [c for c in costs if c < float("inf")]
        row.n = len(costs)
        row.wall_s = round(acc["wall"] / max(len(costs), 1), 3)
        row.cost = (
            round(statistics.fmean(finite), 1) if finite else None
        )
        ratios = [
            c / f
            for c, f in zip(costs, fl, strict=True)
            if f > 0 and c < float("inf")
        ]
        row.ratio = (
            round(statistics.fmean(ratios), 4) if ratios else None
        )
        notes = [_contraction_notes(costs, fl, floor_name)]
        if acc.get("errors"):
            notes.append(
                f"{len(acc['errors'])} board(s)" + acc["errors"][0]
            )
        row.notes = "; ".join(n for n in notes if n)
        rows.append(row)
    order = {name: i for i, name in enumerate(per_arm)}
    rows.sort(key=lambda r: order.get(r.player, 0))
    return rows


# ---------------------------------------------------------------------------
#  Dispatch
# ---------------------------------------------------------------------------


def run_board(
    board: str,
    players: list[str] | None,
    *,
    seed: int,
    meta_cases: int,
    meta_budget: int,
    open_pages: int,
    open_page: int,
    open_budget: int,
    neural_epochs: int,
    neural_hidden: int,
    scales: tuple[int, ...],
    instances: int,
    samples: int,
    device: str,
) -> tuple[list[Row], dict]:
    """Run one board; return ``(rows, meta)``."""
    if board == "contraction":
        return (
            run_contraction(
                players,
                seed=seed,
                scales=scales,
                instances=instances,
                samples=samples,
                device=device,
            ),
            {"scales": list(scales), "instances": instances},
        )
    if board == "open":
        return run_open(
            players,
            seed=seed,
            pages=open_pages,
            page=open_page,
            budget=open_budget,
            neural_epochs=neural_epochs,
            neural_hidden=neural_hidden,
            device=device,
        )
    if board not in DOMAINS:
        raise ValueError(
            f"unknown board {board!r} — known: "
            f"{', '.join([*sorted(DOMAINS), 'contraction'])}"
        )
    return (
        run_episode_board(
            board,
            players,
            seed=seed,
            cases=meta_cases,
            budget=meta_budget,
        ),
        {"cases": meta_cases, "budget": meta_budget},
    )

"""The player arena — the player itself as the rewritable term.

The two-tier meta-game this repository's game layer was built for:
an episode's *state* is a :class:`~mlopt_discovery.players.DomainSpec`
record — the object-level player's spec as data — and moves rewrite
it:

* ``toggle_feature(part, add)`` — drop or restore a composable
  feature part within the shipped vocabulary (the schema-leaderboard
  axis: ``bias``/``state``/``hist``/``onehot``/``action``);
* ``edit_weight(name, delta)`` — nudge the weight *prior* the child
  trains from;
* ``set_lr(lr)`` — set the child's REINFORCE learning rate;
* ``set_weights(table)`` — install a named prior (``cold`` /
  ``baked``);
* ``evaluate(k)`` — the oracle call: build the record's spec,
  train the child on the object board, score it on held-out cases.
  Non-terminal, bounded by the episode's eval budget — a *probe*;
* ``extract()`` — terminal: emit the best probed record, paying the
  held-out **delta vs the parent spec** it started from.

The nested evaluation is the leaderboard's own protocol
(``lab/schema_leaderboard.py`` — equal small budget, cold prior,
paired draws): the ``meta`` objective trains on
``meta_player.gen_cases(seed)`` and scores on ``gen_cases(seed +
7919)``; the ``open`` objective trains on even pages and scores on
the held-out odd pages under *memoized* per-candidate verdicts —
``proposal.evaluate`` draws its numeric tensors from torch's global
RNG, so each candidate is measured once under a stable per-candidate
seed inside a forked RNG (:func:`_forked_rng` — identical draws for
parent and every child, and the outer episode's RNG untouched).  The
regress is
bounded honestly: the *meta*-board's own spec — :data:`PLAYER_SPEC`,
the featurizer over spec records — stays fixed; this board rewrites
the object-level player, not its own rules.
"""

from __future__ import annotations

import argparse
import copy
import json
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from . import engine, lawdata
from . import meta_player as mp
from . import open_game as og
from . import proposal as lp
from .meta_arena import Action
from .players import (
    ActionFeats,
    Bias,
    DomainSpec,
    OpOneHot,
    PlayRates,
    StateBroadcast,
    register_legal,
    register_part,
    spec_from_record,
)

__all__ = [
    "OBJECTIVES",
    "PLAYER_ARENA_REWARD",
    "PLAYER_FEATURES",
    "PLAYER_SPEC",
    "GreedyEditor",
    "PlayerArena",
    "PlayerState",
    "gen_cases",
    "legal_actions",
    "main",
]

#: Held-out stream offset — ``play``'s convention (meta only).
_HELD_OUT_OFFSET = 7919

#: Eval-arm seed offsets — shared across children (paired draws).
_SAMPLED_SEED = 40_000
_GREEDY_SEED = 50_000

#: Per-candidate oracle seed base for the memoized verdicts.
_CAND_SEED = 977

#: The board's reward table — data, like the other arenas'.
#:
#: ``step`` prices every move cheaply (the open game's convention);
#: ``eval`` prices an ``evaluate`` probe — the oracle call is the
#: episode's real cost, so it pays visibly more than an edit;
#: ``delta`` scales the terminal payout ``delta·(child - parent)`` —
#: the raw held-out delta, unscaled.
PLAYER_ARENA_REWARD: dict[str, float] = {
    "step": -0.002,
    "eval": -0.01,
    "delta": 1.0,
}

#: The move vocabulary, in schema order.
_PLAYER_OPS: tuple[str, ...] = (
    "toggle_feature",
    "edit_weight",
    "set_lr",
    "set_weights",
    "evaluate",
    "extract",
)

#: The ``edit_weight`` delta grid and the ``set_lr`` grid — the move
#: parameter space is data, not a continuum.
_WEIGHT_DELTAS: tuple[float, ...] = (-0.5, 0.5)
_LR_GRID: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2)

#: Default oracle-call budget per episode.
_MAX_EVALS = 8

#: The memoized oracle — objective -> {candidate key: Outcome}.
#: Shared across boards and episodes: the same enumerated candidate
#: measures identically everywhere (the leaderboard's pairing), and
#: memoizing it keeps a meta-episode's probes priced at their first
#: measurement only.
_MEMOS: dict[str, dict] = {}


def _sig(record: Mapping) -> str:
    """Canonical record signature — dedup + probe identity."""
    return json.dumps(record, sort_keys=True, default=str)


def _can_drop(
    parts: tuple[str, ...],
    tag: str,
    requires: tuple[tuple[str, str], ...],
) -> bool:
    """Return False when a kept part still needs *tag*.

    *requires* are ``(writer, reader)`` pairs — the leaderboard's
    ``META_REQUIRES``: ``meta:action`` reads ``st:*`` keys the
    ``meta:state`` broadcast writes, so ``state`` may only drop once
    ``action`` is gone.
    """
    return not any(a == tag and b in parts for a, b in requires)


def _can_add(
    parts: tuple[str, ...],
    tag: str,
    requires: tuple[tuple[str, str], ...],
) -> bool:
    """Return False when *tag*'s own prerequisites are absent."""
    return all(a in parts for a, b in requires if b == tag)


def _toggled(
    parts: tuple[str, ...], tag: str, vocab: tuple[str, ...]
) -> tuple:
    """Toggle *tag* in *parts*, re-ordered to the canonical vocab."""
    kept = set(parts)
    if tag in kept:
        kept.discard(tag)
    else:
        kept.add(tag)
    return tuple(t for t in vocab if t in kept)


def _mean(rows: list, key: str) -> float:
    """Mean of row[*key*] over an ``engine.evaluate`` arm's rows."""
    n = max(len(rows), 1)
    return sum(r[key] for r in rows) / n


# ---------------------------------------------------------------------------
#  The nested evaluation — build the child spec, train, score held-out
# ---------------------------------------------------------------------------
#
#  Each objective's scorer is one function
#  ``(record, *, seed, cfg) -> {score, greedy, evals_ep}``: the record
#  builds through ``spec_from_record``, the child trains at the
#  objective's reduced budget (the leaderboard's), and *score* is the
#  sampled arm's mean held-out reward.


def _score_meta(record: Mapping, *, seed: int, cfg: dict) -> dict:
    """Nested meta-board eval — train cold, score ``gen_cases(+7919)``."""
    spec = spec_from_record(record)
    train = mp.gen_cases(seed, cfg.get("train_cases", 40))
    ev = mp.gen_cases(
        seed + _HELD_OUT_OFFSET, cfg.get("eval_cases", 16)
    )
    budget = cfg.get("budget", 12)
    player = mp.MetaLearnedPlayer(seed, spec=spec)
    for _ in range(cfg.get("epochs", 15)):
        mp.train_player(player, train, budget=budget)
    table = mp.evaluate(
        {
            "sampled": lambda i: player.frozen(
                seed + _SAMPLED_SEED + i
            ),
            "greedy": lambda i: player.frozen(
                seed + _GREEDY_SEED + i, greedy=True
            ),
        },
        ev,
        budget=budget,
    )
    return {
        "score": _mean(table["sampled"], "reward"),
        "greedy": _mean(table["greedy"], "reward"),
        "evals_ep": _mean(table["sampled"], "steps"),
    }


def _cand_key(cand: Any) -> tuple:
    """Stable candidate identity — same proposal, same verdict."""
    return (cand.label, repr(cand.lhs), repr(cand.rhs))


def _memoized_evaluate(memo: dict) -> Any:
    """Return an ``lp.evaluate`` replaying *memo*'s verdicts.

    A cache miss measures the candidate once under a stable
    per-candidate seed (``_CAND_SEED`` xor a crc of its key) — the
    verdict is deterministic per candidate regardless of ask order,
    so parent and every child see literally identical oracle draws.
    """
    import torch

    real_evaluate = lp.evaluate

    def evaluate(cands: Any, **kw: Any) -> list:
        out = []
        for c in cands:
            key = _cand_key(c)
            o = memo.get(key)
            if o is None:
                torch.manual_seed(
                    _CAND_SEED ^ zlib.crc32(repr(key).encode())
                )
                o = real_evaluate([c], **kw)[0]
                memo[key] = o
            out.append(o)
        return out

    return evaluate


def _forked_rng() -> Any:
    """Snapshot-and-restore torch's RNG — a portable ``fork_rng``.

    This torch build exposes ``get_rng_state``/``set_rng_state`` but
    not ``torch.fork_rng``; the manual pair is the same guarantee —
    the nested oracle draws never perturb the outer episode's RNG.
    """
    import contextlib

    import torch

    @contextlib.contextmanager
    def fork() -> Any:
        state = torch.get_rng_state()
        try:
            yield
        finally:
            torch.set_rng_state(state)

    return fork()


def _score_open(record: Mapping, *, seed: int, cfg: dict) -> dict:
    """Nested open-board eval — even-page train, odd-page held-out.

    Runs under a patched, memoized ``lp.evaluate`` (identical verdicts
    for parent and children) inside a forked torch RNG — the oracle's
    draws never perturb the outer episode's RNG.
    """
    spec = spec_from_record(record)
    page = cfg.get("page", 16)
    pages = cfg.get("pages", 36)
    budget = cfg.get("budget", page + 1)
    train = og.gen_cases(seed, pages, page=page, start=0, step=2)
    ev = og.gen_cases(seed, pages, page=page, start=1, step=2)
    orig = lp.evaluate
    lp.evaluate = _memoized_evaluate(cfg["memo"])
    try:
        import torch

        with _forked_rng():
            torch.manual_seed(seed)
            player = og.OpenLearnedPlayer(seed, spec=spec)
            for _ in range(cfg.get("epochs", 3)):
                engine.train_policy(
                    player, train, og.OpenArena, budget=budget
                )
            table = engine.evaluate(
                {
                    "sampled": lambda i: player.frozen(
                        seed + _SAMPLED_SEED + i
                    ),
                    "greedy": lambda i: player.frozen(
                        seed + _GREEDY_SEED + i, greedy=True
                    ),
                },
                ev,
                og.OpenArena,
                budget=budget,
            )
    finally:
        lp.evaluate = orig
    return {
        "score": _mean(table["sampled"], "reward"),
        "greedy": _mean(table["greedy"], "reward"),
        "evals_ep": _mean(table["sampled"], "steps"),
    }


def _open_parent() -> dict:
    """Return the incumbent open spec as a record — cold prior."""
    rec = og.OPEN_SPEC.to_record()
    rec["weights"] = None
    return rec


def _meta_parent() -> dict:
    """Return the incumbent meta spec as a record — cold prior."""
    rec = mp.META_SPEC.to_record()
    rec["weights"] = None
    return rec


@dataclass(frozen=True)
class Objective:
    """One object-level board a player spec is rewritten against.

    ``vocab`` is the togglable part vocabulary (canonical order —
    additions insert at their shipped position); ``requires`` the
    ``(writer, reader)`` order constraints; ``parent`` builds the
    incumbent record the episode starts from; ``tables`` the named
    weight priors ``set_weights`` installs; ``weight_names`` the
    prior features ``edit_weight`` may nudge; ``score`` the nested
    evaluation; ``cfg`` its protocol defaults.
    """

    vocab: tuple[str, ...]
    requires: tuple[tuple[str, str], ...]
    parent: Any  # () -> record dict
    tables: dict[str, Any]
    weight_names: tuple[str, ...]
    score: Any  # (record, *, seed, cfg) -> metrics dict
    cfg: dict[str, Any] = field(default_factory=dict)


#: The object-level objectives — domain data, like ``play.DOMAINS``.
OBJECTIVES: dict[str, Objective] = {
    "meta": Objective(
        vocab=mp.META_PART_TAGS,
        requires=(("meta:state", "meta:action"),),
        parent=_meta_parent,
        tables={
            "cold": None,
            "baked": lawdata.META_ARENA_PLAYER_WEIGHTS,
        },
        weight_names=(
            "bias",
            "op:extract",
            "op:saturate",
            "x:extract:improve",
            "st:improve",
            "a:unbounded",
        ),
        score=_score_meta,
        cfg={
            "epochs": 15,
            "train_cases": 40,
            "eval_cases": 16,
            "budget": 12,
        },
    ),
    "open": Objective(
        vocab=og.OPEN_PART_TAGS,
        requires=(),
        parent=_open_parent,
        tables={
            "cold": None,
            "baked": lawdata.OPEN_GAME_PLAYER_WEIGHTS,
        },
        weight_names=(
            "a:score",
            "a:depth",
            "a:metas",
            "a:ops",
            "bias",
            "op:evaluate",
            "op:stop",
            "h:evaluate",
            "st:useful_rate",
        ),
        score=_score_open,
        cfg={"epochs": 3, "pages": 36, "page": 16, "budget": 17},
    ),
}

#: Union vocabularies — the meta-board's own schema spans both
#: objectives' tags, weight names and table names.
_ALL_PART_TAGS: tuple[str, ...] = tuple(
    sorted({t for o in OBJECTIVES.values() for t in o.vocab})
)
_ALL_WEIGHT_NAMES: tuple[str, ...] = tuple(
    sorted({n for o in OBJECTIVES.values() for n in o.weight_names})
)
_ALL_TABLE_NAMES: tuple[str, ...] = tuple(
    sorted({t for o in OBJECTIVES.values() for t in o.tables})
)


# ---------------------------------------------------------------------------
#  The board — a spec record under rewrite
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerState:
    """The read-only snapshot a meta-player decides on.

    ``record`` is the term under rewrite (a spec record — pure data);
    ``parent`` the incumbent record the episode started from, and
    ``parent_score`` its held-out score (``None`` until the first
    probe measures it).  ``probes`` are the oracle's answers so far —
    ``(parts, score, delta, parent)`` — what a scripted meta-player
    reads to keep or revert an edit.
    """

    record: dict
    parent: dict
    parts: tuple[str, ...]
    droppable: tuple[str, ...]
    addable: tuple[str, ...]
    weight_names: tuple[str, ...]
    tables: tuple[str, ...]
    steps: int
    done: bool
    evals: int
    max_evals: int
    probed: bool
    probes: tuple[dict, ...] = ()
    best_delta: float | None = None
    parent_score: float | None = None


class PlayerArena:
    """One spec record, one eval budget, the rewrite-to-better game.

    *case* is ``{"objective": "meta"|"open", "seed": int,
    "max_evals": int, "cfg": {...}}`` — *objective* picks the
    object-level board family from :data:`OBJECTIVES`; *seed* drives
    the nested train/eval draws (paired across children); *cfg*
    overrides the objective's protocol (epochs, pages, budgets).
    The record starts as the incumbent spec's record with a cold
    prior — the leaderboard's equal-budget protocol.
    """

    def __init__(self, case: Mapping) -> None:
        """Bind the objective, the incumbent record and the budget."""
        self._objective = OBJECTIVES[str(case["objective"])]
        self._seed = int(case.get("seed", 0))
        self._cfg = dict(self._objective.cfg)
        self._cfg.update(dict(case.get("cfg", {})))
        self._cfg["memo"] = _MEMOS.setdefault(
            str(case["objective"]), {}
        )
        self._score_override = case.get("score_fn")
        self.record = self._objective.parent()
        self.parent = copy.deepcopy(self.record)
        self.parent_score: float | None = None
        self.steps = 0
        self.done = False
        self.evals = 0
        self.max_evals = int(case.get("max_evals", _MAX_EVALS))
        self.probes: list[dict] = []
        self._scores: dict[str, dict] = {}
        self.best: dict | None = None

    def observe(self) -> PlayerState:
        """Snapshot the record, the legal vocabulary and the probes."""
        parts = tuple(self.record["parts"])
        req = self._objective.requires
        return PlayerState(
            record=copy.deepcopy(self.record),
            parent=copy.deepcopy(self.parent),
            parts=parts,
            droppable=tuple(
                t for t in parts if _can_drop(parts, t, req)
            ),
            addable=tuple(
                t
                for t in self._objective.vocab
                if t not in parts and _can_add(parts, t, req)
            ),
            weight_names=self._objective.weight_names,
            tables=tuple(self._objective.tables),
            steps=self.steps,
            done=self.done,
            evals=self.evals,
            max_evals=self.max_evals,
            probed=_sig(self.record) in self._scores,
            probes=tuple(self.probes),
            best_delta=None
            if self.best is None
            else self.best["delta"],
            parent_score=self.parent_score,
        )

    def _score(self, record: dict) -> dict:
        """Nested-eval *record*, measuring the parent once alongside."""
        if self._score_override is not None:
            res = self._score_override(
                record, seed=self._seed, cfg=self._cfg
            )
            if self.parent_score is None:
                self.parent_score = self._score_override(
                    self.parent, seed=self._seed, cfg=self._cfg
                )["score"]
            return res
        if self.parent_score is None:
            pres = self._objective.score(
                self.parent, seed=self._seed, cfg=self._cfg
            )
            self.parent_score = pres["score"]
            self._scores[_sig(self.parent)] = pres
        sig = _sig(record)
        if sig not in self._scores:
            self._scores[sig] = self._objective.score(
                record, seed=self._seed, cfg=self._cfg
            )
        return self._scores[sig]

    def _apply_probe(self, res: dict) -> dict:
        """Record *res* as a probe of the current record; update best."""
        sig = _sig(self.record)
        delta = res["score"] - self.parent_score
        probe = {
            "parts": tuple(self.record["parts"]),
            "sig": sig,
            "score": res["score"],
            "parent": self.parent_score,
            "delta": delta,
        }
        self._scores[sig] = res
        self.probes.append(probe)
        if self.best is None or delta > self.best["delta"]:
            self.best = {
                "sig": sig,
                "record": copy.deepcopy(self.record),
                "score": res["score"],
                "delta": delta,
            }
        return probe

    # -- moves -------------------------------------------------------

    def _mv_toggle(self, p: dict) -> dict[str, Any]:
        """``toggle_feature(part)`` — drop/restore a part, requires-safe."""
        tag = str(p["part"])
        parts = tuple(self.record["parts"])
        req = self._objective.requires
        if tag in parts and not _can_drop(parts, tag, req):
            return {"applied": False, "note": f"required:{tag}"}
        if tag not in parts and not _can_add(parts, tag, req):
            return {"applied": False, "note": f"requires:{tag}"}
        if tag not in self._objective.vocab:
            return {"applied": False, "note": f"unknown-part:{tag}"}
        was = tag in parts
        rec = dict(self.record)
        rec["parts"] = list(_toggled(parts, tag, self._objective.vocab))
        self.record = rec
        return {
            "applied": True,
            "note": f"toggle:{'drop' if was else 'add'}:{tag}",
        }

    def _mv_edit_weight(self, p: dict) -> dict[str, Any]:
        """``edit_weight(name, delta)`` — nudge the weight prior."""
        name = str(p["name"])
        rec = dict(self.record)
        w = dict(rec.get("weights") or {})
        w[name] = w.get(name, 0.0) + float(p["delta"])
        rec["weights"] = w
        self.record = rec
        return {"applied": True, "note": f"w:{name}+={p['delta']}"}

    def _mv_set_lr(self, p: dict) -> dict[str, Any]:
        """``set_lr(lr)`` — set the child's REINFORCE rate."""
        rec = dict(self.record)
        rec["lr"] = float(p["lr"])
        self.record = rec
        return {"applied": True, "note": f"lr:{p['lr']}"}

    def _mv_set_weights(self, p: dict) -> dict[str, Any]:
        """``set_weights(table)`` — install a named prior table."""
        name = str(p["table"])
        if name not in self._objective.tables:
            return {"applied": False, "note": f"unknown-table:{name}"}
        table = self._objective.tables[name]
        rec = dict(self.record)
        rec["weights"] = None if table is None else dict(table)
        self.record = rec
        return {"applied": True, "note": f"weights:{name}"}

    def _mv_evaluate(self, p: dict) -> dict[str, Any]:
        """``evaluate(k)`` — the oracle probe: train, score held-out."""
        del p
        if self.evals >= self.max_evals:
            return {"applied": False, "note": "evals-exhausted"}
        sig = _sig(self.record)
        if sig in self._scores:
            return {"applied": False, "note": "already-measured"}
        res = self._score(self.record)
        self.evals += 1
        probe = self._apply_probe(res)
        return {
            "applied": True,
            "reward": PLAYER_ARENA_REWARD["eval"],
            "note": f"score:{res['score']:.4f} delta:{probe['delta']:+.4f}",
            "cost": res["score"],
            "detail": {"probe": probe, "metrics": res},
        }

    def _mv_extract(self, p: dict) -> dict[str, Any]:
        """``extract()`` — terminal: emit the best probed record.

        With no probe taken the terminal emission measures the record
        it emits (the deliverable gets its own measurement, off the
        probe budget — a bare extract resubmits the incumbent).
        """
        del p
        if self.best is None:
            res = self._score(self.record)
            self._apply_probe(res)
        best = self.best
        assert best is not None  # _apply_probe sets it
        delta = best["delta"]
        return {
            "applied": True,
            "terminal": True,
            "reward": PLAYER_ARENA_REWARD["delta"] * delta,
            "note": f"child:{best['score']:.4f} vs parent:{self.parent_score:.4f}",
            "cost": best["score"],
            "detail": {
                "child": best["record"],
                "child_score": best["score"],
                "parent_score": self.parent_score,
                "delta": delta,
                "probes": list(self.probes),
            },
        }

    def step(self, action: Action) -> tuple[PlayerState, engine.Report]:
        """Apply *action*; return ``(state', report)``.

        Edit moves pay ``step``; a declined move scores 0 (the
        honest-refusal convention); ``evaluate`` pays ``step + eval``
        and records the child-vs-parent delta; ``extract`` pays
        ``step + delta·(child - parent)`` and ends the episode.
        """
        if self.done:
            raise RuntimeError(
                "episode finished — extract() is terminal"
            )
        fn = _MOVES.get(action.op)
        out: dict[str, Any] = (
            {"applied": False, "note": f"unknown:{action.op}"}
            if fn is None
            else fn(self, action.params)
        )
        applied = bool(out.get("applied", False))
        terminal = bool(out.get("terminal", False))
        reward = PLAYER_ARENA_REWARD["step"] if applied else 0.0
        reward += float(out.get("reward", 0.0))
        self.steps += 1
        self.done = self.done or terminal
        return self.observe(), engine.Report(
            action=action,
            reward=reward,
            terminal=terminal,
            applied=applied,
            note=str(out.get("note", "")),
            cost=float(out.get("cost", float("nan"))),
            detail=dict(out.get("detail", {})),
        )


#: Move dispatch — op name -> handler ``(arena, params) -> outcome``.
_MOVES: dict[str, Any] = {
    "toggle_feature": PlayerArena._mv_toggle,
    "edit_weight": PlayerArena._mv_edit_weight,
    "set_lr": PlayerArena._mv_set_lr,
    "set_weights": PlayerArena._mv_set_weights,
    "evaluate": PlayerArena._mv_evaluate,
    "extract": PlayerArena._mv_extract,
}


def legal_actions(state: PlayerState) -> tuple[Action, ...]:
    """Enumerate the moves *state* affords, deterministically.

    ``toggle_feature`` per droppable part (``add=False``) and per
    addable one (``add=True``); ``edit_weight`` over the objective's
    weight names x deltas; ``set_lr`` over the grid minus the current
    rate; ``set_weights`` per named table; ``evaluate`` while the
    eval budget lasts and the record is unmeasured; ``extract`` last.
    """
    if state.done:
        return ()
    acts: list[Action] = [
        Action("toggle_feature", {"part": t, "add": False})
        for t in state.droppable
    ]
    acts += [
        Action("toggle_feature", {"part": t, "add": True})
        for t in state.addable
    ]
    for name in state.weight_names:
        for d in _WEIGHT_DELTAS:
            acts.append(
                Action("edit_weight", {"name": name, "delta": d})
            )
    for lr in _LR_GRID:
        if lr != state.record.get("lr"):
            acts.append(Action("set_lr", {"lr": lr}))
    for t in state.tables:
        acts.append(Action("set_weights", {"table": t}))
    if state.evals < state.max_evals and not state.probed:
        acts.append(Action("evaluate", {"k": state.evals}))
    acts.append(Action("extract", {}))
    return tuple(acts)


# ---------------------------------------------------------------------------
#  The scripted meta-player — greedy sequential part ablation
# ---------------------------------------------------------------------------


class GreedyEditor:
    """Leave-one-part-out ablation, then extract — the meta-player.

    The policy the schema leaderboard measured made human-legible:
    probe the record minus each droppable part (leave-one-out);
    a drop that does not cost held-out reward is *kept* — the part
    was dead weight — and the ablation continues from the smaller
    record (sequential elimination, so interactions between drops
    are measured, not assumed); a losing drop is reverted.  When
    every part has been tried the episode ``extract``s the best
    probed child.  *tol* is the keep-threshold — a drop costing at
    most *tol* reward is still a simplification.
    """

    def __init__(self, tol: float = 0.0) -> None:
        """Bind the keep-threshold and the ablation bookkeeping."""
        self._tol = tol
        self._tried: set[str] = set()
        self._awaiting: str | None = None
        self._base: float | None = None

    def __call__(self, state: PlayerState) -> Action | None:
        """Return the next ablation move, or ``extract`` when done."""
        if state.done:
            return None
        if self._awaiting is not None:
            move = self._resolve(state)
            if move is not None:
                return move
        for tag in state.droppable:
            if tag not in self._tried and state.evals < state.max_evals:
                self._tried.add(tag)
                self._awaiting = tag
                return Action(
                    "toggle_feature", {"part": tag, "add": False}
                )
        return Action("extract", {})

    def _resolve(self, state: PlayerState) -> Action | None:
        """Keep or revert the armed drop; ``None`` means probe it."""
        last = state.probes[-1] if state.probes else None
        if last is None or tuple(state.record["parts"]) != tuple(
            last["parts"]
        ):
            if state.evals < state.max_evals:
                return Action("evaluate", {"k": state.evals})
            bad, self._awaiting = self._awaiting, None
            return Action("toggle_feature", {"part": bad, "add": True})
        ref = self._base if self._base is not None else last["parent"]
        if last["score"] >= ref - self._tol:
            self._base = last["score"]
            self._awaiting = None
            return None
        bad, self._awaiting = self._awaiting, None
        return Action("toggle_feature", {"part": bad, "add": True})


# ---------------------------------------------------------------------------
#  The meta-board's own spec — the player-over-spec-records
# ---------------------------------------------------------------------------


def _p_state(state: PlayerState) -> dict[str, float]:
    """Return the ``st:*`` broadcast — record shape, budget, probes."""
    return {
        "st:steps": state.steps / 12.0,
        "st:n_parts": len(state.parts) / 8.0,
        "st:evals": state.evals / max(state.max_evals, 1),
        "st:probes": len(state.probes) / max(state.max_evals, 1),
        "st:lr": float(state.record.get("lr", 0.1)) / 0.2,
        "st:has_weights": 1.0 if state.record.get("weights") else 0.0,
        "st:best_delta": max(0.0, state.best_delta or 0.0),
        "st:edited": 0.0 if state.record == state.parent else 1.0,
        "st:parented": 0.0 if state.parent_score is None else 1.0,
    }


def _p_act(action: Action, _row: dict[str, float]) -> dict[str, float]:
    """Return the move's params — part/weight/table identity, values."""
    op, p = action.op, action.params
    out: dict[str, float] = {}
    if op == "toggle_feature":
        out[f"a:part:{p['part']}"] = 1.0
        out["a:dir:add" if p.get("add") else "a:dir:drop"] = 1.0
    elif op == "edit_weight":
        out[f"a:w:{p['name']}"] = 1.0
        out["a:delta"] = float(p["delta"])
    elif op == "set_lr":
        out["a:lr"] = float(p["lr"])
    elif op == "set_weights":
        out[f"a:table:{p['table']}"] = 1.0
    elif op == "evaluate":
        out["a:k"] = float(p.get("k", 0))
    return out


#: The player-board feature schema — ``st:*`` over the record, the
#: ``h:*``/``op:*`` move channels, and the ``a:*`` move params.
PLAYER_FEATURES: tuple[str, ...] = (
    "bias",
    "st:steps",
    "st:n_parts",
    "st:evals",
    "st:probes",
    "st:lr",
    "st:has_weights",
    "st:best_delta",
    "st:edited",
    "st:parented",
    *(f"h:{o}" for o in _PLAYER_OPS),
    *(f"op:{o}" for o in _PLAYER_OPS),
    "a:delta",
    "a:lr",
    "a:k",
    "a:dir:add",
    "a:dir:drop",
    *(f"a:part:{t}" for t in _ALL_PART_TAGS),
    *(f"a:w:{n}" for n in _ALL_WEIGHT_NAMES),
    *(f"a:table:{t}" for t in _ALL_TABLE_NAMES),
)

#: The player-board spec's own part tags — the meta-board's spec
#: reifies too (the regress stops here only because the meta-board's
#: rules stay fixed — see module docstring).
register_part("player:state", lambda: StateBroadcast(_p_state))
register_part("player:hist", lambda: PlayRates(_PLAYER_OPS))
register_part(
    "player:onehot", lambda: OpOneHot(frozenset(PLAYER_FEATURES))
)
register_part("player:action", lambda: ActionFeats(_p_act))
register_legal("player", legal_actions)

#: The learned meta-arm's spec — a ``LearnedPlayer`` over the
#: spec-record board, cold by default.
PLAYER_SPEC = DomainSpec.build(
    features=PLAYER_FEATURES,
    legal=legal_actions,
    parts=(
        Bias(),
        StateBroadcast(_p_state),
        PlayRates(_PLAYER_OPS),
        OpOneHot(frozenset(PLAYER_FEATURES)),
        ActionFeats(_p_act),
    ),
)

#: The composed featurizer — public for tests.
featurize = PLAYER_SPEC.featurize


# ---------------------------------------------------------------------------
#  Cases + driver — one meta-episode emits a child spec
# ---------------------------------------------------------------------------


def gen_cases(
    seed: int,
    n: int,
    *,
    objective: str = "open",
    max_evals: int = _MAX_EVALS,
) -> list[dict]:
    """Generate *n* player-board cases for *objective*.

    Each case is ``{objective, seed, max_evals}`` — the seed varies
    the nested draws (the open board's pages are fixed, so its cases
    differ only in sampler seeds; the meta board's case stream
    differs by construction).
    """
    return [
        {
            "name": f"player{i}:{objective}",
            "objective": objective,
            "seed": seed + i,
            "max_evals": max_evals,
        }
        for i in range(n)
    ]


def _report_table(reports: list) -> str:
    """Render one line per move — the meta-episode's trace."""
    lines = [f"{'move':<22} {'reward':>9} {'note':<40}", "-" * 73]
    for rep in reports:
        a = rep.action
        move = a.op
        if a.params:
            move += " " + ",".join(
                f"{k}={v}" for k, v in a.params.items()
            )
        lines.append(f"{move:<22} {rep.reward:>9.4f} {rep.note:<40}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Play one meta-episode; emit the best child spec record.

    The headline run: seed the incumbent spec, let
    :class:`GreedyEditor` ablate it under the oracle budget, extract
    the best probed child, and report the parent-vs-child held-out
    pair the terminal move measured.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--objective", choices=sorted(OBJECTIVES), default="open"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--max-evals", type=int, default=_MAX_EVALS)
    ap.add_argument(
        "--pages",
        type=int,
        default=0,
        help="open: pages per split (default 36, the leaderboard's)",
    )
    ap.add_argument(
        "--epochs", type=int, default=0, help="nested train epochs"
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="GreedyEditor keep-threshold (drop is free below it)",
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    cfg: dict[str, Any] = {}
    if args.pages:
        cfg["pages"] = args.pages
    if args.epochs:
        cfg["epochs"] = args.epochs
    case = {
        "objective": args.objective,
        "seed": args.seed,
        "max_evals": args.max_evals,
        "cfg": cfg,
    }
    board = PlayerArena(case)
    traj = engine.run_episode(
        board, GreedyEditor(tol=args.tol), args.budget
    )
    print(  # stdout-compat
        f"== player arena — {args.objective} spec under rewrite =="
    )
    print(_report_table(traj.reports))  # stdout-compat
    t = traj.terminal
    if t is None:
        print("episode ended without extract")  # stdout-compat
        return 1
    d = t.detail
    child = d["child"]
    print(  # stdout-compat
        f"parent: score={d['parent_score']:.4f} "
        f"parts={board.parent['parts']}"
    )
    print(  # stdout-compat
        f"child:  score={d['child_score']:.4f} "
        f"delta={d['delta']:+.4f} parts={child['parts']}"
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "objective": args.objective,
                    "seed": args.seed,
                    "parent": board.parent,
                    "parent_score": d["parent_score"],
                    "child": child,
                    "child_score": d["child_score"],
                    "delta": d["delta"],
                    "probes": d["probes"],
                },
                f,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

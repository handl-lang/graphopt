"""Fragment-application boards — plan 0033 phase 4 / ADR-0157.

The meta-arena's move space is a *Python rule bank*: ``fire`` names a
``Rewrite`` the bank already carries.  This module seats the same
referee/extraction machinery on the canonical ``vocab/`` fragment
directory instead: a legal move is *apply this vocabulary fragment to
this e-class/site*.

Move kinds
----------

* ``law`` / ``prim`` / ``schema`` — executable ``Rewrite`` fragments.
  ``law`` rows come from ``vocab/laws.yaml`` (the hand-written bank),
  ``prim`` rows are the minted ``prim:<op>`` folds the same file
  carries, and ``schema`` rows are the ``fold:*`` proposal schemas
  ``vocab/schemas.yaml`` ships as data.  Each move names one fragment
  and one matching e-class; ``step`` applies exactly that rule at
  exactly that site through ``EGraph.apply_rule(..., search={site})``.
* ``handler`` — a ``vocab/handlers.yaml`` interpretation row.  Its
  application is the meta-arena's own ``claim`` move: declare the
  handler-covered fold, fire it once, bind the kernel — still priced
  and certified by the same referee.
* ``saturate`` — the closure move over *all* executable rewrite
  fragments above, run through the board's engine.  Under
  ``engine="handl"`` the engine is :class:`HandlEngine`, so the
  Python saturation is replayed through the handl peer and a
  disagreement refuses the episode (the phase-1 differential path;
  native handl-egraph saturation remains phase 5).
* ``extract`` — terminal; delegates to ``MetaArena.step(Action.extract())``
  so the certificate verifier stays the referee.

``op`` and ``model`` rows are vocabulary *data* (AST shapes and
``.hnl`` denotations), not term rewrites or interpretations, so they
are counted in ``FragmentState.catalog`` but are not offered as
moves.  ``family`` rows are mint provenance, likewise catalogued.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gamekit import engine
from mlopt_core.egraph import Rewrite

from mlopt_discovery import vocabdir
from mlopt_discovery.meta_arena import Action, MetaArena, _reported

__all__ = [
    "FragmentArena",
    "FragmentMove",
    "FragmentState",
    "fragment_legal",
]


# ---------------------------------------------------------------------------
#  Moves and state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FragmentMove:
    """One fragment application — data only, no engine objects.

    ``site`` is the e-class id the fragment applies at; ``-1`` marks
    board-level moves (``saturate`` / ``extract`` / ``handler`` claims,
    which name their own site internally).
    """

    kind: str
    ref: str
    site: int = -1


@dataclass(frozen=True)
class FragmentState:
    """The board view a fragment player decides on."""

    moves: tuple[FragmentMove, ...]
    catalog: dict[str, int]
    n_enodes: int
    n_classes: int
    steps: int
    done: bool
    best_cost: float
    baseline_cost: float
    parity: dict[str, Any]


def fragment_legal(state: FragmentState) -> tuple[FragmentMove, ...]:
    """Legal moves are exactly the state's enumerated fragment moves."""
    return state.moves


# ---------------------------------------------------------------------------
#  Vocabulary → executable fragments
# ---------------------------------------------------------------------------


def _rewrite_kinds(v: vocabdir.Vocab) -> tuple[dict, dict, dict]:
    """Split the vocab into executable rewrite fragments by kind."""
    laws: dict[str, Rewrite] = {}
    prims: dict[str, Rewrite] = {}
    schemas: dict[str, Rewrite] = {}
    for rule in v.laws:
        if rule.name.startswith("prim:"):
            prims[rule.name] = rule
        else:
            laws[rule.name] = rule
    for rule in v.schemas:
        schemas[rule.name] = rule
    return laws, prims, schemas


def _catalog(v: vocabdir.Vocab) -> dict[str, int]:
    """Count every fragment row by kind — including non-move rows."""
    return {
        "law": len(v.laws),
        "prim": len(v.prims),
        "schema": len(v.schemas),
        "handler": len(v.handler_table()),
        "op": len(v.ops),
        "model": len(v.models),
        "family": len(v.families),
    }


# ---------------------------------------------------------------------------
#  The board
# ---------------------------------------------------------------------------


class FragmentArena:
    """A :class:`gamekit.engine.Board` over ``vocab/`` fragments.

    Parameters mirror :class:`MetaArena` where they matter; ``vocab_dir``
    selects the fragment directory (default: the repo ``vocab/`` or
    ``MLOPT_VOCAB_DIR``).  ``engine="handl"`` swaps the Python
    ``EGraph`` for :class:`HandlEngine` — saturation then runs the
    phase-1 differential parity path; it is *not* native handl-egraph
    saturation (that is phase 5 and gated on the perf-parity measure).
    """

    def __init__(
        self,
        term: Any,
        vocab_dir: Any = None,
        *,
        engine: str = "python",
        peer: Any = None,
        rules: Any = None,
        corpus: Any = (),
        cost_fn: Any = None,
        supported: Any = None,
        max_nodes: int = 20_000,
        max_moves: int = 512,
        weights: dict | None = None,
        handlers: dict | None = None,
    ) -> None:
        """Load the fragment vocabulary and open the referee board."""
        directory = (
            vocab_dir
            if vocab_dir is not None
            else vocabdir.default_vocab_dir()
        )
        if directory is None:
            raise ValueError(
                "no vocab directory — pass vocab_dir or set MLOPT_VOCAB_DIR"
            )
        self.vocab_dir = directory
        self.vocab = vocabdir.load_vocab(directory)
        self.laws, self.prims, self.schemas = _rewrite_kinds(self.vocab)
        self._rewrites: dict[tuple[str, str], Rewrite] = {
            **{("law", k): v for k, v in self.laws.items()},
            **{("prim", k): v for k, v in self.prims.items()},
            **{("schema", k): v for k, v in self.schemas.items()},
        }
        self.handlers = (
            self.vocab.handler_table() if handlers is None else handlers
        )
        exec_rules = (
            list(self._rewrites.values())
            if rules is None
            else list(rules)
        )
        self.arena = MetaArena(
            term,
            exec_rules,
            corpus=corpus,
            cost_fn=cost_fn,
            supported=supported,
            max_nodes=max_nodes,
            weights=weights,
            handlers=self.handlers,
        )
        if engine == "handl":
            from mlopt_discovery.handl_engine import HandlEngine

            he = HandlEngine(peer=peer, fragments=directory)
            self.arena.eg = he
            self.arena.root = he.add_term(term)
        elif engine != "python":
            raise ValueError(f"unknown fragment engine {engine!r}")
        self.engine_name = engine
        self.max_moves = int(max_moves)
        self.steps = 0
        self.done = False
        self._catalog = _catalog(self.vocab)

    # -- Board protocol ----------------------------------------------------

    def observe(self) -> FragmentState:
        """Enumerate currently-applicable fragment moves, deterministically."""
        meta = self.arena.observe()
        moves: list[FragmentMove] = []
        if not self.done:
            moves.extend(self._rewrite_moves())
            moves.extend(self._handler_moves())
            moves.append(FragmentMove("saturate", "*"))
            moves.append(FragmentMove("extract", "*"))
            if self.max_moves and len(moves) > self.max_moves:
                # Keep the terminal moves reachable under the cap.
                head = moves[: self.max_moves - 2]
                moves = [
                    *head,
                    FragmentMove("saturate", "*"),
                    FragmentMove("extract", "*"),
                ]
        parity_fn = getattr(self.arena.eg, "parity_stats", None)
        parity = parity_fn() if callable(parity_fn) else {}
        return FragmentState(
            moves=tuple(moves),
            catalog=dict(self._catalog),
            n_enodes=self.arena.eg.n_enodes,
            n_classes=self.arena.eg.n_classes,
            steps=self.steps,
            done=self.done,
            best_cost=meta.best_cost,
            baseline_cost=meta.baseline_cost,
            parity=parity,
        )

    def step(self, action: Any) -> tuple[FragmentState, engine.Report]:
        """Apply one fragment move; extraction stays referee-checked."""
        if self.done:
            raise RuntimeError(
                "episode finished — extract() is terminal"
            )
        move = _as_move(action)
        if move is None:
            raise KeyError(f"not a fragment move: {action!r}")
        before = self._best_cost()
        if move.kind == "extract":
            _state, rep = self.arena.step(Action.extract())
            self.done = True
            self.steps += 1
            return self.observe(), engine.Report(
                action=move,
                reward=rep.reward,
                terminal=True,
                applied=rep.applied,
                note=rep.note,
                cost=rep.cost,
                certificate_ok=rep.certificate_ok,
                detail=rep.detail,
            )
        if move.kind == "saturate":
            report = self._saturate()
        elif move.kind == "handler":
            report = self._claim(move.ref)
        else:
            report = self._apply(move)
        self.steps += 1
        after = self._best_cost()
        reward = _reward_delta(
            before, after, report.get("enodes_delta", 0)
        )
        return self.observe(), engine.Report(
            action=move,
            reward=reward,
            terminal=False,
            applied=report.get("applied", False),
            note=report.get("note", ""),
            cost=after,
            certificate_ok=None,
            detail=report.get("detail", {}),
        )

    # -- move enumeration ---------------------------------------------------

    def _rewrite_moves(self) -> list[FragmentMove]:
        """One move per (fragment, matching e-class), sorted."""
        out: list[FragmentMove] = []
        for (kind, ref), rule in sorted(self._rewrites.items()):
            for eid in self._match_sites(rule):
                out.append(FragmentMove(kind, ref, eid))
                if self.max_moves and len(out) >= self.max_moves:
                    return out
        return out

    def _match_sites(self, rule: Rewrite) -> list[int]:
        """E-classes where *rule*'s lhs has at least one match."""
        eg = self.arena.eg
        try:
            prog = eg._prog_for(rule.lhs)
            cands = eg._candidate_classes(prog, None)
        except Exception:
            cands = None
        if cands is None:
            cands = set(eg._classes)
        sites: list[int] = []
        for raw in sorted(cands):
            eid = eg.find(raw)
            if eid in sites:
                continue
            try:
                hit = next(
                    eg.matches(rule.lhs, eid, max_results=1), None
                )
            except Exception:
                hit = None
            if hit is not None:
                sites.append(eid)
        return sites

    def _handler_moves(self) -> list[FragmentMove]:
        """Return handler rows whose pattern occurs and is unclaimed."""
        out: list[FragmentMove] = []
        claimed = {
            op
            for op in self.arena.definitions
            if op.startswith("claim_")
        } | set(self.arena.interpretations)
        for tag, h in sorted(self.handlers.items()):
            if f"claim_{tag}_0" in claimed:
                continue
            try:
                occurs = self.arena._concrete_occurs(h["pattern"])
            except Exception:
                occurs = False
            if occurs:
                out.append(FragmentMove("handler", str(tag)))
        return out

    # -- move application ---------------------------------------------------

    def _apply(self, move: FragmentMove) -> dict[str, Any]:
        rule = self._rewrites.get((move.kind, move.ref))
        if rule is None:
            return {
                "applied": False,
                "note": f"{move.kind}: unknown fragment {move.ref!r}",
            }
        eg = self.arena.eg
        remaining = self.arena.max_nodes - eg.n_enodes
        if remaining <= 0:
            return {
                "applied": False,
                "note": "fragment: enode budget exhausted",
            }
        eid = eg.find(move.site)
        changed = eg.apply_rule(
            rule, self.arena.root, search={eid}, enode_budget=remaining
        )
        eg.rebuild()
        check_pending = getattr(eg, "check_pending", None)
        if callable(check_pending):
            check_pending()
        return {
            "applied": bool(changed),
            "note": f"{move.kind} {move.ref}@{move.site}: {'changed' if changed else 'no change'}",
            "enodes_delta": max(
                0, eg.n_enodes - (self.arena.max_nodes - remaining)
            ),
            "detail": {"site": move.site, "canonical_site": eid},
        }

    def _saturate(self) -> dict[str, Any]:
        """Run the executable fragment closure through the board engine."""
        eg = self.arena.eg
        rules = [
            r
            for (k, _), r in sorted(self._rewrites.items())
            if k in {"law", "prim", "schema"}
        ]
        stats = eg.run(
            rules,
            self.arena.root,
            max_iterations=8,
            max_nodes=self.arena.max_nodes,
            stop="improving",
            patience=3,
            cost_fn=self.arena.feasible_cost,
        )
        return {
            "applied": True,
            "note": f"saturate[{len(rules)}]: {stats.get('stop')} it={stats.get('iterations')}",
            "enodes_delta": 0,
            "detail": {
                "stop": stats.get("stop"),
                "iterations": stats.get("iterations"),
                "handl_parity": stats.get("handl_parity"),
            },
        }

    def _claim(self, tag: str) -> dict[str, Any]:
        """Apply a handler fragment through the referee's claim move."""
        _state, rep = self.arena.step(Action.claim(str(tag)))
        return {
            "applied": rep.applied,
            "note": rep.note,
            "enodes_delta": rep.enodes_delta,
            "detail": rep.detail,
        }

    def _best_cost(self) -> float:
        best = self.arena.eg.extract_best(
            self.arena.root, self.arena.feasible_cost
        )
        if best is None:
            return float("inf")
        from mlopt_core.cost import dag_cost

        return _reported(dag_cost(best, self.arena.feasible_cost))


def _as_move(action: Any) -> FragmentMove | None:
    if isinstance(action, FragmentMove):
        return action
    if isinstance(action, dict) and {"kind", "ref"} <= set(action):
        return FragmentMove(
            str(action["kind"]),
            str(action["ref"]),
            int(action.get("site", -1)),
        )
    return None


def _reward_delta(before: float, after: float, enodes: int) -> float:
    """Cost-improvement reward with a small enode tax."""
    delta = 0.0
    if before != float("inf") and after != float("inf") and before > 0:
        delta = (before - after) / before
    return delta - min(enodes, 1000) / 10000.0

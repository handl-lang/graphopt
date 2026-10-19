"""Feature-schema leaderboard — score candidate featurizer schemas held-out.

The capacity ablations (``docs/results_learned_ablations.md``) showed
*features* bind, not capacity — ``RuleValueNet`` h32 ≈ h96, while the
linear policy's whole world is its feature row.  This demo asks the
complement question: does **representation search** (which composable
feature parts ship in the schema, and in which row order) find a
better policy than the shipped schema re-trained at the same budget?
Every candidate is a subset x ordering of the shipped part vocabulary
(:data:`meta_player.META_PARTS` / :data:`open_game.OPEN_PARTS`), built
through the one :meth:`players.DomainSpec.build` path — no player code
changes, the schema *is* the candidate.

Enumeration (``--max-perms``, default 3):

* all ``2^5 - 1 = 31`` nonempty subsets of the five parts;
* orderings — ``Bias`` fixed first where present (the intercept is a
  convention, not a hypothesis), and on the **meta** board
  ``StateBroadcast`` must precede ``ActionFeats`` (``meta_player
  ._act_feats`` reads ``row["st:*"]`` — any other order raises
  ``KeyError``; subsets holding ``action`` without ``state`` are
  infeasible and skipped, counted in the summary);
* the canonical (shipped) relative order is always enumerated; further
  valid orderings up to *max_perms* are drawn seeded;
* orderings are **deduplicated by emitted-row probe** — two orderings
  that produce identical feature rows on the probe legal set are one
  schema.  In the shipped vocabulary only ``ActionFeats`` reads ``row``
  and only ``st:*`` keys, so every valid ordering of a subset emits the
  same row: the leaderboard's real axis is *which parts*, order is
  kept honest rather than multiplied for ceremony.

Per candidate, cold-trained at a **reduced budget** (this enumerates
many train+eval cycles — the point is schema ranking under one fixed
small budget, not best-possible weights):

* **meta** — ``mp.MetaLearnedPlayer(seed, spec=spec, weights={})``,
  ``--meta-epochs`` (15, vs the shipped bake's 30) x
  ``--meta-train-cases`` (40) of ``mp.train_player`` at budget 12;
  held-out ``mp.gen_cases(seed + 7919, --meta-eval-cases)``, sampled
  and greedy frozen arms (the ``train_meta_prior.py`` split).
* **open** — ``og.OpenLearnedPlayer(seed, spec=spec, weights={})``,
  ``--open-epochs`` (3, vs the shipped 5) x 36 even pages under the
  baseline's **memoized verdicts** (``policy_baseline._measure_pools``
  + a patched ``lp.evaluate`` — identical oracle draws for every
  candidate, training included); held-out odd pages, sampled and
  greedy frozen arms, plus AP of ``rank_score`` over the flat held-out
  pool.

Anchors: the **incumbent** (the shipped parts tuple, canonical order,
same reduced budget — it should *not* always win, and it does not
always) and eval-only reference rows (random / scripted / heuristic /
the shipped baked prior) so the numbers sit on the baseline's scale.

Runtime (measured, this box): meta ≈ 11 s/candidate (15 epochs x 40
cases ≈ 10.5 s + 0.6 s eval) → ~23 feasible candidates ≈ 5 min; open ≈
6 s/candidate + ~145 s one-time pool memoization → ~31 candidates ≈
5 min.  Each domain fits ``timeout 1800`` alone; both together do not
— run ``--domain meta`` and ``--domain open`` separately.

Usage::

    .venv/bin/python lab/schema_leaderboard.py --domain meta
    .venv/bin/python lab/schema_leaderboard.py --domain open \
        --json /tmp/schema_open.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from typing import Any

import policy_baseline as pb
import torch
from mlopt_core.cost.basic import count_cost
from mlopt_discovery import engine, play
from mlopt_discovery import meta_arena as ma
from mlopt_discovery import meta_player as mp
from mlopt_discovery import open_game as og
from mlopt_discovery import proposal as lp
from mlopt_discovery.players import DomainSpec

#: Defaults — the reduced per-candidate budget (see module docstring).
SEED = 0
MAX_PERMS = 3
META_TRAIN_CASES = 40
META_EVAL_CASES = 16
META_EPOCHS = 15  # the shipped bake used 30
META_BUDGET = 12
OPEN_PAGES = 36  # the even half trains, the odd half evaluates
PAGE = 16
OPEN_EPOCHS = 3  # the shipped bake used 5
OPEN_BUDGET = PAGE + 1

#: Part vocabulary, in shipped (canonical) order — the names the
#: leaderboard uses.  Order in these tables defines "canonical".
#: Vocab values are *bundles* of parts — an entry may expand to
#: several featurizer parts (the open board's ``ext`` toggle carries
#: the measured-AP winners ``a:score_rank``/``a:score_rel``/
#: ``x:score:*`` as one unit; per-feature importance is
#: ``lab/richer_features.py``'s axis, not this leaderboard's).
META_VOCAB: tuple[tuple[str, Any], ...] = tuple(
    zip(
        ("bias", "state", "hist", "onehot", "action"),
        ((p,) for p in mp.META_PARTS),
        strict=True,
    )
)
OPEN_VOCAB: tuple[tuple[str, Any], ...] = (
    *tuple(
        zip(
            ("bias", "state", "hist", "onehot", "action"),
            ((p,) for p in og.OPEN_PARTS[:5]),
            strict=True,
        )
    ),
    ("ext", tuple(og.OPEN_PARTS[5:])),
)
#: Order constraints: writer part → reader part that must follow it.
#: ``meta_player._act_feats`` reads ``row["st:*"]`` interaction inputs;
#: open's ``_act_feats`` ignores ``row`` entirely.
META_REQUIRES: tuple[tuple[str, str], ...] = (("state", "action"),)
OPEN_REQUIRES: tuple[tuple[str, str], ...] = ()

#: Eval arm seed offsets — shared across candidates (paired draws).
_SAMPLED_SEED = 40_000
_GREEDY_SEED = 50_000


# ---------------------------------------------------------------------------
#  Candidate enumeration — subsets x deduplicated orderings
# ---------------------------------------------------------------------------


def _valid_orderings(
    names: tuple[str, ...],
    canonical: tuple[str, ...],
    requires: tuple[tuple[str, str], ...],
) -> list[tuple[str, ...]]:
    """All orderings of *names* with ``bias`` first and *requires* met."""
    body = [n for n in names if n != "bias"]
    out: set[tuple[str, ...]] = set()
    for perm in itertools.permutations(body):
        seq = (("bias",) if "bias" in names else ()) + perm
        if any(
            a in seq and b in seq and seq.index(a) > seq.index(b)
            for a, b in requires
        ):
            continue
        out.add(seq)
    canon = tuple(n for n in canonical if n in names)
    rest = sorted(out - {canon})
    return ([canon] if canon in out else []) + rest


def _orderings_for(
    names: tuple[str, ...],
    canonical: tuple[str, ...],
    requires: tuple[tuple[str, str], ...],
    max_perms: int,
    rng: random.Random,
) -> list[tuple[str, ...]]:
    """Canonical order first, then a seeded sample up to *max_perms*."""
    valid = _valid_orderings(names, canonical, requires)
    if len(valid) <= max_perms:
        return valid
    head = valid[:1]  # canonical always kept
    return head + rng.sample(valid[1:], max_perms - len(head))


def _feasible(
    names: tuple[str, ...], requires: tuple[tuple[str, str], ...]
) -> bool:
    """Return False when a reader part's writer is absent."""
    return all(not (b in names and a not in names) for a, b in requires)


def enumerate_candidates(
    vocab: tuple[tuple[str, Any], ...],
    requires: tuple[tuple[str, str], ...],
    *,
    max_perms: int,
    seed: int,
) -> tuple[list[dict], int]:
    """Enumerate ``(subset, ordering)`` candidates over *vocab*.

    Returns ``(candidates, n_infeasible_subsets)``.  Each candidate is
    ``{"id", "names", "parts", "incumbent"}``; orderings are not yet
    deduplicated by probe (that needs a domain-specific probe state —
    the caller runs :func:`dedupe_by_probe`).
    """
    canonical = tuple(n for n, _ in vocab)
    by_name = dict(vocab)
    rng = random.Random(seed)
    cands: list[dict] = []
    infeasible = 0
    for r in range(1, len(vocab) + 1):
        for subset in itertools.combinations(canonical, r):
            if not _feasible(subset, requires):
                infeasible += 1
                continue
            for seq in _orderings_for(
                subset, canonical, requires, max_perms, rng
            ):
                cands.append(
                    {
                        "id": ">".join(seq),
                        "names": seq,
                        "parts": tuple(
                            p for n in seq for p in by_name[n]
                        ),
                        "incumbent": seq == canonical,
                    }
                )
    return cands, infeasible


def dedupe_by_probe(
    cands: list[dict], signature_of: Any
) -> tuple[list[dict], int]:
    """Drop orderings whose emitted rows are identical to a kept one.

    *signature_of(parts)* hashes the emitted feature rows on a probe
    legal set.  Within a subset, orderings that emit identical rows
    train identically — running them would measure noise, not order.
    """
    seen: set = set()
    out: list[dict] = []
    dropped = 0
    for c in cands:
        key = (frozenset(c["names"]), signature_of(c["parts"]))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(c)
    return out, dropped


# ---------------------------------------------------------------------------
#  Shared candidate machinery
# ---------------------------------------------------------------------------


def _n_features(spec: DomainSpec, state: Any, legal: Any) -> int:
    """Distinct feature names the schema emits on a probe state."""
    keys: set[str] = set()
    for a in legal(state):
        keys.update(spec.featurize(state, a))
    return len(keys)


def _sort_key(row: dict) -> tuple:
    """Leaderboard order: reward desc, then AP desc, then id."""
    ap = row["ap"] if row["ap"] is not None else float("-inf")
    return (-row["reward_ep"], -ap, row["id"])


def _ranked(rows: list[dict]) -> list[dict]:
    """Sort candidate rows; anchors are listed separately, unranked."""
    cands = [r for r in rows if r["kind"] == "candidate"]
    return sorted(cands, key=_sort_key)


def part_analysis(
    rows: list[dict], vocab_names: tuple[str, ...]
) -> dict:
    """Ablation-style part importance from the ranked rows.

    ``best_with``/``best_without`` — the best reward/ep achievable
    with the part present vs excluded (over candidate rows); a part
    matters when the best schema needs it.  ``loo`` — the canonical
    leave-one-out subset's delta vs the incumbent, or ``None`` when
    that subset is infeasible (meta: dropping ``state`` strands
    ``action``).
    """
    cands = [r for r in rows if r["kind"] == "candidate"]
    by_id = {r["id"]: r for r in cands}
    inc = next((r for r in cands if r["incumbent"]), None)
    out: dict[str, Any] = {"incumbent": inc["id"] if inc else None}
    for p in vocab_names:
        with_p = [r for r in cands if p in r["parts"]]
        without = [r for r in cands if p not in r["parts"]]
        loo_names = tuple(n for n in vocab_names if n != p)
        loo = by_id.get(">".join(loo_names))
        out[p] = {
            "best_with": max(
                (r["reward_ep"] for r in with_p), default=None
            ),
            "best_without": max(
                (r["reward_ep"] for r in without), default=None
            ),
            "loo_id": ">".join(loo_names),
            "loo_reward_ep": loo["reward_ep"] if loo else None,
            "loo_delta_vs_incumbent": (
                round(loo["reward_ep"] - inc["reward_ep"], 4)
                if loo and inc
                else None
            ),
        }
    return out


# ---------------------------------------------------------------------------
#  Meta domain — train + held-out eval per candidate
# ---------------------------------------------------------------------------


def _meta_spec(parts: tuple) -> DomainSpec:
    """Build the candidate spec — shipped schema names, candidate parts."""
    return DomainSpec.build(
        features=mp.lawdata.META_ARENA_FEATURES,
        legal=ma.legal_actions,
        parts=parts,
        weights=None,
        lr=mp._LR,
    )


def _meta_probe_state(cases: list) -> Any:
    """One initial board state for the emitted-row signature."""
    return mp._board(cases[0], count_cost, 20_000).observe()


def _meta_eval(
    player: Any, ev: list, budget: int, seed: int
) -> dict[str, Any]:
    """Held-out reward/steps for the sampled and greedy frozen arms."""
    arms = {
        "sampled": lambda i: player.frozen(seed + _SAMPLED_SEED + i),
        "greedy": lambda i: player.frozen(
            seed + _GREEDY_SEED + i, greedy=True
        ),
    }
    table = mp.evaluate(arms, ev, budget=budget)
    out: dict[str, Any] = {}
    for name, rows in table.items():
        k = max(len(rows), 1)
        out[name] = {
            "reward": round(sum(r["reward"] for r in rows) / k, 4),
            "steps": round(sum(r["steps"] for r in rows) / k, 3),
            "cert": sum(1 for r in rows if r["cert"]),
            "declared": sum(1 for r in rows if r["used_declared"]),
        }
    return out


def _meta_eval_fn(mk: Any, ev: list, budget: int) -> dict[str, Any]:
    """Eval an arm *factory* (non-learned anchors like random)."""
    table = mp.evaluate({"arm": mk}, ev, budget=budget)
    rows = table["arm"]
    k = max(len(rows), 1)
    return {
        "sampled": {
            "reward": round(sum(r["reward"] for r in rows) / k, 4),
            "steps": round(sum(r["steps"] for r in rows) / k, 3),
            "cert": sum(1 for r in rows if r["cert"]),
            "declared": sum(1 for r in rows if r["used_declared"]),
        }
    }


def _meta_row(
    cand: dict,
    res: dict[str, Any],
    n_feats: int,
    kind: str = "candidate",
) -> dict:
    """Shape the leaderboard row from the two eval arms."""
    g = res.get("greedy")
    return {
        "id": cand["id"],
        "kind": kind,
        "parts": list(cand["names"]),
        "n_parts": len(cand["names"]),
        "incumbent": bool(cand.get("incumbent")),
        "n_features": n_feats,
        "reward_ep": res["sampled"]["reward"],
        "reward_greedy_ep": g["reward"] if g else None,
        "useful_ep": None,  # the meta board has no useful verdict
        "evals_ep": res["sampled"]["steps"],
        "ap": None,  # META_SPEC ships no candidate ranker
        "cert_eps": res["sampled"]["cert"],
        "declared_eps": res["sampled"]["declared"],
    }


def run_meta(
    seed: int,
    train_cases_n: int,
    eval_cases_n: int,
    epochs: int,
    budget: int,
    max_perms: int,
) -> dict[str, Any]:
    """Enumerate the META_PARTS schemas; train and score each cold."""
    t0 = time.time()
    train_cases = mp.gen_cases(seed, train_cases_n)
    ev = mp.gen_cases(seed + pb.HELD_OUT_OFFSET, eval_cases_n)
    probe = _meta_probe_state(train_cases)

    cands, infeasible = enumerate_candidates(
        META_VOCAB, META_REQUIRES, max_perms=max_perms, seed=seed
    )

    def signature(parts: tuple) -> Any:
        spec = _meta_spec(parts)
        return tuple(
            tuple(sorted(spec.featurize(probe, a).items()))
            for a in ma.legal_actions(probe)
        )

    cands, dup_orderings = dedupe_by_probe(cands, signature)

    rows: list[dict] = []
    for i, c in enumerate(cands):
        spec = _meta_spec(c["parts"])
        torch.manual_seed(seed)
        # Same base seed for every candidate — the comparison is paired:
        # schemas diverge via features, not via sampler noise.
        player = mp.MetaLearnedPlayer(seed, spec=spec, weights={})
        for _ in range(epochs):
            mp.train_player(player, train_cases, budget=budget)
        torch.manual_seed(seed)
        res = _meta_eval(player, ev, budget, seed)
        rows.append(
            _meta_row(
                c, res, _n_features(spec, probe, ma.legal_actions)
            )
        )
        print(  # stdout-compat
            f"  [{i + 1}/{len(cands)}] {c['id']:<38} "
            f"reward={rows[-1]['reward_ep']:>8.4f}"
        )

    anchors: list[dict] = []
    # The shipped baked prior — eval only, the anchor on the old scale.
    baked = mp.MetaLearnedPlayer(seed)
    res = _meta_eval(baked, ev, budget, seed)
    anchors.append(
        _meta_row(
            {
                "id": "ANCHOR:shipped-prior(baked)",
                "names": tuple(n for n, _ in META_VOCAB),
            },
            res,
            _n_features(mp.META_SPEC, probe, ma.legal_actions),
            kind="anchor",
        )
    )
    # Scripted playbook and uniform random — the baseline's floors.
    res = _meta_eval_fn(lambda i: ma.ScriptedPlayer(), ev, budget)
    anchors.append(
        _meta_row(
            {"id": "ANCHOR:scripted", "names": ()},
            res,
            0,
            kind="anchor",
        )
    )
    rnd = pb._meta_random_arm(seed)
    res = _meta_eval_fn(rnd, ev, budget)
    anchors.append(
        _meta_row(
            {"id": "ANCHOR:random", "names": ()}, res, 0, kind="anchor"
        )
    )

    ranked = _ranked(rows)
    return {
        "rows": ranked + anchors,
        "summary": {
            "subsets_feasible": len(
                {tuple(sorted(c["names"])) for c in cands}
            ),
            "orderings_enumerated": len(cands) + dup_orderings,
            "orderings_deduped": dup_orderings,
            "subsets_infeasible": infeasible,
            "candidates_run": len(ranked),
            "incumbent_rank": next(
                (i + 1 for i, r in enumerate(ranked) if r["incumbent"]),
                None,
            ),
            "winner": ranked[0]["id"] if ranked else None,
            "elapsed_s": round(time.time() - t0, 1),
        },
        "parts": part_analysis(rows, tuple(n for n, _ in META_VOCAB)),
    }


# ---------------------------------------------------------------------------
#  Open domain — memoized verdicts, even-page train, odd-page eval
# ---------------------------------------------------------------------------


class _FixedArena(og.OpenArena):
    """An ``OpenArena`` over a pre-materialized pool (skip islice)."""

    def __init__(self, pool: list) -> None:
        self.pool = list(pool)
        self.verdicts: dict[int, str] = {}
        self.steps = 0
        self.done = not self.pool
        self._n_true = 0
        self._n_useful = 0


def _open_spec(parts: tuple) -> DomainSpec:
    """Build the candidate spec — shipped schema, parts, ranker."""
    return DomainSpec.build(
        features=og.OPEN_GAME_FEATURES,
        legal=og.legal_actions,
        parts=parts,
        weights=None,
        rank=og._cand_feats,
        lr=og.OPEN_SPEC.lr,
    )


def _play_open_pools(mk: Any, pools: list[list], budget: int) -> dict:
    """``pb._play_open`` over fixed pools — identical memoized draws."""
    found = evals = 0
    reward = 0.0
    for i, pool in enumerate(pools):
        traj = engine.run_episode(_FixedArena(pool), mk(i), budget)
        reward += traj.total
        for r in traj.reports:
            evals += 1
            if r.note and r.note.split(":", 1)[0] == "useful":
                found += 1
    n = len(pools)
    return {
        "reward": round(reward / n, 4),
        "useful_per_ep": round(found / n, 4),
        "useful_per_eval": round(found / max(evals, 1), 4),
        "evals_per_ep": round(evals / n, 3),
        "found": found,
    }


def _open_row(
    cand: dict,
    sampled: dict,
    greedy: dict | None,
    ap: float | None,
    n_feats: int,
    kind: str = "candidate",
) -> dict:
    """Shape the leaderboard row from eval results."""
    return {
        "id": cand["id"],
        "kind": kind,
        "parts": list(cand["names"]),
        "n_parts": len(cand["names"]),
        "incumbent": bool(cand.get("incumbent")),
        "n_features": n_feats,
        "reward_ep": sampled["reward"],
        "reward_greedy_ep": greedy["reward"] if greedy else None,
        "useful_ep": sampled["useful_per_ep"],
        "evals_ep": sampled["evals_per_ep"],
        "ap": ap,
    }


def run_open(
    seed: int,
    half: int,
    page: int,
    epochs: int,
    max_perms: int,
) -> dict[str, Any]:
    """Enumerate the OPEN_PARTS schemas under memoized verdicts."""
    t0 = time.time()
    budget = page + 1
    train_cases = og.gen_cases(seed, half, page=page, start=0, step=2)
    ev = og.gen_cases(seed, half, page=page, start=1, step=2)
    train_pools = [list(og.OpenArena(c).pool) for c in train_cases]
    ev_pools = [list(og.OpenArena(c).pool) for c in ev]
    eval_outcomes, by_key = pb._measure_pools(
        ev_pools, seed + pb.HELD_OUT_OFFSET
    )
    _t, train_keys = pb._measure_pools(train_pools, seed + 1_000_003)
    by_key.update(train_keys)
    del _t

    flat_pool = [c for pool in ev_pools for c in pool]
    flat_outcomes: dict[int, Any] = {}
    pos = 0
    for outs in eval_outcomes:
        for o in outs:
            flat_outcomes[pos] = o
            pos += 1

    probe = _FixedArena(ev_pools[0]).observe()

    def signature(parts: tuple) -> Any:
        spec = _open_spec(parts)
        return tuple(
            tuple(sorted(spec.featurize(probe, a).items()))
            for a in og.legal_actions(probe)
        )

    cands, infeasible = enumerate_candidates(
        OPEN_VOCAB, OPEN_REQUIRES, max_perms=max_perms, seed=seed
    )
    cands, dup_orderings = dedupe_by_probe(cands, signature)

    rows: list[dict] = []
    orig_evaluate = lp.evaluate
    lp.evaluate = pb._patched_evaluate(by_key)
    try:
        for i, c in enumerate(cands):
            spec = _open_spec(c["parts"])
            # Same base seed for every candidate — paired comparison.
            player = og.OpenLearnedPlayer(
                seed=seed, spec=spec, weights={}
            )
            for _ in range(epochs):
                engine.train_policy(
                    player, train_pools, _FixedArena, budget=budget
                )
            sampled = _play_open_pools(
                lambda i_, p=player: p.frozen(
                    seed + _SAMPLED_SEED + i_
                ),
                ev_pools,
                budget,
            )
            greedy = _play_open_pools(
                lambda i_, p=player: p.frozen(
                    seed + _GREEDY_SEED + i_, greedy=True
                ),
                ev_pools,
                budget,
            )
            ap = round(
                pb._ap(
                    [player.rank_score(c_) for c_ in flat_pool],
                    flat_outcomes,
                ),
                4,
            )
            rows.append(
                _open_row(
                    c,
                    sampled,
                    greedy,
                    ap,
                    _n_features(spec, probe, og.legal_actions),
                )
            )
            print(  # stdout-compat
                f"  [{i + 1}/{len(cands)}] {c['id']:<38} "
                f"reward={rows[-1]['reward_ep']:>8.4f} "
                f"ap={ap:.4f}"
            )

        anchors: list[dict] = []
        baked = og.OpenLearnedPlayer(seed)
        anchors.append(
            _open_row(
                {
                    "id": "ANCHOR:shipped-prior(baked)",
                    "names": tuple(n for n, _ in OPEN_VOCAB),
                },
                _play_open_pools(
                    lambda i_: baked.frozen(seed + 30_000 + i_),
                    ev_pools,
                    budget,
                ),
                _play_open_pools(
                    lambda i_: baked.frozen(
                        seed + 35_000 + i_, greedy=True
                    ),
                    ev_pools,
                    budget,
                ),
                round(
                    pb._ap(
                        [baked.rank_score(c_) for c_ in flat_pool],
                        flat_outcomes,
                    ),
                    4,
                ),
                _n_features(og.OPEN_SPEC, probe, og.legal_actions),
                kind="anchor",
            )
        )
        anchors.append(
            _open_row(
                {"id": "ANCHOR:heuristic", "names": ()},
                _play_open_pools(pb._heuristic_arm(), ev_pools, budget),
                None,
                round(
                    pb._ap(
                        [lp._heuristic_score(c_) for c_ in flat_pool],
                        flat_outcomes,
                    ),
                    4,
                ),
                0,
                kind="anchor",
            )
        )
        rnd = play._random_policy(og.legal_actions)
        anchors.append(
            _open_row(
                {"id": "ANCHOR:random", "names": ()},
                _play_open_pools(
                    lambda i_: rnd(seed + 10_000 + i_), ev_pools, budget
                ),
                None,
                round(
                    pb._ap([0.0] * len(flat_pool), flat_outcomes),
                    4,
                ),
                0,
                kind="anchor",
            )
        )
    finally:
        lp.evaluate = orig_evaluate

    ranked = _ranked(rows)
    n_useful = sum(1 for o in flat_outcomes.values() if o.useful)
    return {
        "rows": ranked + anchors,
        "summary": {
            "subsets_feasible": len(
                {tuple(sorted(c["names"])) for c in cands}
            ),
            "orderings_enumerated": len(cands) + dup_orderings,
            "orderings_deduped": dup_orderings,
            "subsets_infeasible": infeasible,
            "candidates_run": len(ranked),
            "incumbent_rank": next(
                (i + 1 for i, r in enumerate(ranked) if r["incumbent"]),
                None,
            ),
            "winner": ranked[0]["id"] if ranked else None,
            "pool": {"held_out": len(flat_pool), "useful": n_useful},
            "elapsed_s": round(time.time() - t0, 1),
        },
        "parts": part_analysis(rows, tuple(n for n, _ in OPEN_VOCAB)),
    }


# ---------------------------------------------------------------------------
#  Rendering + driver
# ---------------------------------------------------------------------------


def _print_leaderboard(
    name: str, res: dict[str, Any], top: int, bottom: int
) -> None:
    """Render the ranked table — top rows, bottom rows, anchors."""
    rows = res["rows"]
    cands = [r for r in rows if r["kind"] == "candidate"]
    anchors = [r for r in rows if r["kind"] == "anchor"]
    s = res["summary"]
    print(  # stdout-compat
        f"== {name} — {s['candidates_run']} candidates "
        f"({s['orderings_enumerated']} orderings enumerated, "
        f"{s['orderings_deduped']} deduped, "
        f"{s['subsets_infeasible']} infeasible subsets, "
        f"{s['elapsed_s']}s) =="
    )
    head = (
        f"{'rank':>4} {'schema':<38} {'reward':>8} {'greedy':>8}"
        f" {'use/ep':>7} {'evals/ep':>8} {'AP':>7} {'feats':>6}"
    )
    print(head)  # stdout-compat
    print("-" * len(head))  # stdout-compat

    def emit(rank: int, r: dict) -> None:
        ap = f"{r['ap']:7.4f}" if r["ap"] is not None else f"{'—':>7}"
        g = (
            f"{r['reward_greedy_ep']:8.4f}"
            if r["reward_greedy_ep"] is not None
            else f"{'—':>8}"
        )
        u = (
            f"{r['useful_ep']:7.4f}"
            if r["useful_ep"] is not None
            else f"{'—':>7}"
        )
        mark = " *incumbent*" if r["incumbent"] else ""
        print(  # stdout-compat
            f"{rank:>4} {r['id']:<38} {r['reward_ep']:8.4f} {g} "
            f"{u} {r['evals_ep']:8.3f} {ap} {r['n_features']:>6}{mark}"
        )

    for i, r in enumerate(cands[:top]):
        emit(i + 1, r)
    if len(cands) > top + bottom:
        print("  ...")  # stdout-compat
    start = max(len(cands) - bottom, top)
    for j in range(start, len(cands)):
        emit(j + 1, cands[j])
    inc_rank = s["incumbent_rank"]
    if inc_rank and (inc_rank > top and inc_rank <= start):
        emit(inc_rank, cands[inc_rank - 1])
    print("-- anchors --")  # stdout-compat
    for r in anchors:
        emit(0, r)
    print(  # stdout-compat
        f"winner: {s['winner']} | incumbent rank: {inc_rank}/"
        f"{len(cands)}"
    )
    print(
        "-- parts analysis (best reward/ep with vs without) --"
    )  # stdout-compat
    for p, a in res["parts"].items():
        if p == "incumbent":
            continue
        bw = a["best_with"]
        bwo = a["best_without"]
        loo = a["loo_delta_vs_incumbent"]
        print(  # stdout-compat
            f"  {p:<8} best_with={bw if bw is not None else '—':>9} "
            f"best_without={bwo if bwo is not None else '—':>9} "
            f"LOO delta={loo if loo is not None else 'infeasible'}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the schema leaderboard over the chosen domains."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--domain",
        choices=("meta", "open", "both"),
        default="both",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-perms", type=int, default=MAX_PERMS)
    ap.add_argument(
        "--meta-train-cases", type=int, default=META_TRAIN_CASES
    )
    ap.add_argument(
        "--meta-eval-cases", type=int, default=META_EVAL_CASES
    )
    ap.add_argument("--meta-epochs", type=int, default=META_EPOCHS)
    ap.add_argument("--meta-budget", type=int, default=META_BUDGET)
    ap.add_argument("--open-pages", type=int, default=OPEN_PAGES)
    ap.add_argument("--page", type=int, default=PAGE)
    ap.add_argument("--open-epochs", type=int, default=OPEN_EPOCHS)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--bottom", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    payload: dict[str, Any] = {
        "seed": args.seed,
        "protocol": {
            "max_perms": args.max_perms,
            "bias_first": True,
            "ordering_dedup": (
                "orderings emitting identical rows on the probe "
                "legal set are one schema"
            ),
            "meta": {
                "train_cases": args.meta_train_cases,
                "epochs": args.meta_epochs,
                "budget": args.meta_budget,
                "held_out": f"gen_cases(seed+{pb.HELD_OUT_OFFSET}, "
                f"{args.meta_eval_cases})",
            },
            "open": {
                "train": "even pages (start=0, step=2)",
                "held_out": "odd pages (start=1, step=2)",
                "pages": args.open_pages,
                "page": args.page,
                "epochs": args.open_epochs,
                "budget": args.page + 1,
                "oracle_pairing": "memoized per-candidate-seeded "
                "verdicts for train and eval alike",
            },
        },
    }
    if args.domain in ("meta", "both"):
        res = run_meta(
            args.seed,
            args.meta_train_cases,
            args.meta_eval_cases,
            args.meta_epochs,
            args.meta_budget,
            args.max_perms,
        )
        payload["meta"] = res
        _print_leaderboard("meta", res, args.top, args.bottom)
    if args.domain in ("open", "both"):
        res = run_open(
            args.seed,
            args.open_pages,
            args.page,
            args.open_epochs,
            args.max_perms,
        )
        payload["open"] = res
        _print_leaderboard("open", res, args.top, args.bottom)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

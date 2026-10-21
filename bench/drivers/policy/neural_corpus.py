"""Neural-arm corpus attack — grow the on-distribution training pool.

``docs/results_learned_ablations.md`` ablation 4 isolated the neural
deficit as *data, not capacity*: ``RuleValueNet`` hidden 32 -> 96 moved
AP by -0.0006, while 576 on-distribution rows beat 7,560 corpus
``rule_samples`` rows 0.041 vs 0.016.  The blocker is the size of the
on-distribution pool — the even pages of the enumerated candidate
stream.  This driver grows that pool three ways, keeping the held-out
side of the baseline protocol untouched:

* **tail** — the protocol's 36 even + 36 odd pages consume offsets
  0..1151 of the (exactly 1,160-candidate) default-``ops`` stream; the
  trailing 8 candidates are in neither split.  Free, tiny.
* **wide** — the same enumerator over the *full* grammar vocabulary
  (``lawdata.GRAMMAR_BINARY_OPS + GRAMMAR_UNARY_OPS`` — ``pow``,
  ``matmul``, ``rsqrt``, ``sigmoid``, ``silu``, ``tanh`` join the
  default eight): more pages from the same generator, the same
  featurization, the same ``lp.evaluate`` oracle.  Every candidate is
  deduped against the held-out pool by its instance pair
  ``(repr(lhs), repr(rhs))`` — the stream is a superset enumeration,
  so the protocol's candidates recur inside it.
* **sites** — real ``nn.Module``-derived candidates: the zoo
  (``mlopt_discovery.zoo.zoo()``) plus the new variant draw
  (:func:`mlopt_discovery.zoo.zoo_variants` — the same 23
  architectures re-parametrized) plus a capped prefix of the intake's
  ``real_workloads()`` are exported through the intake boundary, and
  ``lp.near_miss_candidates`` mines each saturated e-graph for
  structurally-close pairs the library cannot already reach —
  candidates with real-program leaf shapes and op mixes.

Held-out protocol is identical to ``policy_baseline`` /
``policy_ablations``: odd pages ``gen_cases(seed, 36, page=16,
start=1, step=2)``, every pool member measured once under a
per-candidate ``torch.manual_seed``, verdicts replayed through a
patched ``lp.evaluate``.  Training pools are measured with the same
per-candidate-seeded discipline (fresh verdicts — training rows are
never replayed inside episodes in this driver).  Every arm trains
``RuleValueNet`` (``train_rule_value``, BCE on ``delta > 0``, 300
epochs full-batch CPU, hidden 32) on a ``RuleSample`` per candidate
exactly as ``policy_ablations._ablation_capacity`` builds them; AP is
scored on the same 576-candidate held-out pool.  Two episode rows
(``neural-ondist-576`` / ``neural-ondist+all``) replay the trained
nets as drain-rankers — argmax over pending, drain, ``stop`` — so
reward/episode is comparable to the heuristic drain anchor.

Also measured, for the honest comparison: the off-distribution
``corpus`` arm (``rule_samples`` over ``families.mixture_programs`` —
ablation 4's losing recipe) re-fed with *real* zoo/variant/intake
programs, which asks whether real modules fix the corpus arm the way
they cannot fix the on-distribution one.

Usage::

    .venv/bin/python bench/drivers/policy/neural_corpus.py --json /tmp/nc.json
    .venv/bin/python bench/drivers/policy/neural_corpus.py \
        --wide 1728 --skip-episodes
"""

from __future__ import annotations

import argparse
import contextlib
import math
import multiprocessing as mp
import os
import resource
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any

import policy_baseline as pb
import torch
from mlopt_core.features import compute_features
from mlopt_core.laws import all_rules
from mlopt_core.trajectories import (
    RuleSample,
    rule_samples,
    rule_vector,
)
from mlopt_discovery import families, lawdata, play
from mlopt_discovery import intake as li
from mlopt_discovery import open_game as og
from mlopt_discovery import proposal as lp
from mlopt_discovery import zoo as zoo_mod
from mlopt_discovery.enumerate import (
    DEFAULT_OPS,
    enumerate_candidates,
)
from mlopt_discovery.meta_arena import Action
from mlopt_torch.adapters import TorchSink
from mlopt_torch.learned_policy import input_vector, train_rule_value

#: Defaults — the baseline protocol.
SEED = 0
PAGE = 16
OPEN_PAGES = 36
OPEN_BUDGET = PAGE + 1
NEURAL_EPOCHS = 300
NEURAL_HIDDEN = 32
CORPUS_PER_FAMILY = 30  # policy_ablations' halved CPU-budget choice

#: The protocol's last covered offset: even + odd pages of the
#: 1,160-candidate default stream end at 36*2*16 = 1152; the tail is
#: everything after.
TRAIN_SEED_OFFSET = 1_000_003

#: The widened enumeration vocabulary — every op the algebraic grammar
#: knows, instead of the enumerator's capped ``DEFAULT_OPS``.
WIDE_OPS: tuple = lawdata.GRAMMAR_BINARY_OPS + lawdata.GRAMMAR_UNARY_OPS

#: Near-miss mining parameters on real exported terms — relaxed vs
#: ``proposal.near_miss_candidates``'s defaults (dist 1, size 8 give
#: ~0 pairs on deep real terms; measured: 175 candidates on the zoo at
#: these settings).
SITE_MAX_DIST = 3
SITE_MAX_SIZE = 12
SITE_MAX_PAIRS = 600

#: Per-candidate evaluation guard.  ``lp.evaluate`` -> ``verify_law``
#: saturates a fresh e-graph under ``ALL_RULES`` with
#: ``max_nodes=200_000`` and no expansive-rule budgets; the
#: default-ops stream never trips it, but the widened vocabulary does
#: — ``lit-self:1907 matmul(A,A) => add(A,0)`` was measured OOM-killing
#: an 11 GB box at ~9.3 GB anon RSS (2026-10-20).  An in-process
#: ``MemoryError`` catch is NOT survivable — once the rlimit bites,
#: pymalloc arenas stay mapped and even the except-handler's
#: allocations fail (observed: cascading MemoryError past the catch).
#: Every ``lp.evaluate`` therefore runs in a spawned worker under an
#: address-space rlimit; a candidate that kills the worker (twice on a
#: fresh worker — a single death could be accumulated heap, not the
#: candidate) or stalls past :data:`EVAL_TIMEOUT_S` is recorded as
#: ``None``, dropped from its pool, reported as ``skipped`` — never
#: labeled, never trained on.  Every measured verdict is the
#: protocol's exact verdict: same call, same per-candidate seed.
EVAL_TIMEOUT_S = 180.0
EVAL_RLIMIT_AS = 6 * 1024**3  # ~6 GiB — baseline RSS is ~0.6 GB


def _eval_worker_main(conn: Any) -> None:
    """Worker loop: recv ``(seed, cand)`` -> send ``("ok", outcome)``.

    Death is the honest report for the unbounded-saturation case —
    after MemoryError the allocator cannot be trusted to even format
    an error, so the child exits without one.  A *regular* eval
    exception is reported before dying (it would have crashed the
    unguarded run; here it is a skipped row with a note).
    """
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        seed, cand = msg
        torch.manual_seed(seed)
        try:
            outcome = lp.evaluate([cand])[0]
        except MemoryError:
            os._exit(2)  # allocator wedged — do not allocate to report
        except BaseException as exc:
            with contextlib.suppress(Exception):
                conn.send(("err", f"{type(exc).__name__}: {exc}"[:300]))
            os._exit(3)
        try:
            conn.send(("ok", outcome))
        except Exception:
            os._exit(4)


class _EvalWorker:
    """One spawned ``lp.evaluate`` subprocess; respawned on death."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._proc: Any = None
        self._conn: Any = None
        self.evals = 0

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _spawn(self) -> None:
        self._conn, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_eval_worker_main, args=(child,), daemon=True
        )
        self._proc.start()
        child.close()
        self.evals = 0

    def kill(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.join()
            self._proc = None
            self._conn = None

    def evaluate(self, seed: int, cand: Any) -> tuple[Any, str]:
        """Send one candidate; ``(outcome, "ok")`` or ``(None, why)``.

        ``why`` is ``"died"`` (EOF/timeout — the bomb case) or
        ``"err:<detail>"`` (a reported eval exception).
        """
        if not self._alive():
            self._spawn()
        try:
            self._conn.send((seed, cand))
        except (BrokenPipeError, OSError):
            return None, "died"
        except Exception as exc:
            # Pickling failure — e.g. a candidate rule carrying a
            # lambda check.  Not a heap death: report, don't retry.
            return None, f"err:send:{type(exc).__name__}"
        if not self._conn.poll(EVAL_TIMEOUT_S):
            # Still busy — kill it, or every later eval queues behind
            # the stuck one and times out in turn.
            self.kill()
            return None, "timeout"
        try:
            status, payload = self._conn.recv()
        except (EOFError, OSError):
            self.kill()
            return None, "died"
        self.evals += 1
        if status == "ok":
            return payload, "ok"
        self.kill()
        return None, f"err:{payload}"

    def close(self) -> None:
        """Shut the worker down."""
        if self._alive():
            with contextlib.suppress(BrokenPipeError, OSError):
                self._conn.send(None)
        self.kill()


def _install_eval_limit() -> None:
    """Cap address space at ``EVAL_RLIMIT_AS`` (workers inherit it).

    A failed set is non-fatal (the guard degrades to a box-level OOM)
    so it is logged, not raised.
    """
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        lim = (
            EVAL_RLIMIT_AS
            if hard == resource.RLIM_INFINITY
            else min(hard, EVAL_RLIMIT_AS)
        )
        resource.setrlimit(resource.RLIMIT_AS, (lim, hard))
    except (ValueError, OSError) as exc:
        _log(f"rlimit not installed: {exc!r}")


# ---------------------------------------------------------------------------
#  Pool construction
# ---------------------------------------------------------------------------


def _instance_key(cand: Any) -> tuple[str, str]:
    """Leakage key — the evaluated instance pair, label-free.

    The label differs across streams (``form:n`` numbering) and
    memoization keys; the *instance* ``(lhs, rhs)`` is what
    ``lp.evaluate`` measures and what the featurizer reads, so any
    recurrence of a held-out pair inside a training pool is dropped.
    """
    return (repr(cand.lhs), repr(cand.rhs))


def _load_store(path: str | None) -> dict:
    """Load the measurement checkpoint store ([]-style dict on disk)."""
    if path and Path(path).exists():
        return torch.load(path, weights_only=False)
    return {}


def _save_store(store: dict, path: str | None) -> None:
    """Persist the checkpoint store (per-candidate, crash-resumable)."""
    if path:
        torch.save(store, path)


def _measure(
    cands: list, seed: int, store: dict, tag: str, path: str | None
) -> list:
    """Evaluate each candidate once under its own seeded draw.

    *store*/*path* checkpoint each verdict the moment it lands: the
    ``lp.evaluate`` pass is the run's dominant cost and the box's
    OOM killer does not wait for the table to print — a restart
    replays measured rows instead of re-paying them.

    A candidate whose isolated evaluation kills the worker twice or
    stalls past :data:`EVAL_TIMEOUT_S` is recorded as ``None`` —
    checkpointed like a verdict, so a restart never re-pays the
    blow-up.  ``None`` rows are dropped by :func:`_kept` downstream.
    """
    outs = []
    skipped = 0
    worker = _EvalWorker()
    try:
        for i, cand in enumerate(cands):
            k = (tag, i)
            if k in store:
                outs.append(store[k])
                continue
            o, why = worker.evaluate(seed + i, cand)
            if why in ("died", "timeout"):
                # A death mid-eval may be the candidate's bomb or the
                # worker's accumulated heap — retry once on a fresh
                # worker; two deaths on one candidate = unmeasurable.
                worker.kill()
                o, why = worker.evaluate(seed + i, cand)
            if why != "ok":
                skipped += 1
                _log(f"{tag}[{i}] skipped ({why}): {cand.label}")
                o = None
            store[k] = o
            _save_store(store, path)  # a kill loses at most one row
            outs.append(o)
    finally:
        worker.close()
    if skipped:
        _log(f"{tag}: {skipped}/{len(cands)} candidates unmeasurable")
    return outs


def _kept(cands: list, outcomes: list) -> tuple[list, list, int]:
    """Drop candidates whose evaluation was skipped (``None``)."""
    pairs = [
        (c, o)
        for c, o in zip(cands, outcomes, strict=True)
        if o is not None
    ]
    return (
        [c for c, _ in pairs],
        [o for _, o in pairs],
        len(cands) - len(pairs),
    )


def _stream_pool(cases: list[dict]) -> list:
    """Materialize the pools of generated page cases."""
    return [list(og.OpenArena(c).pool) for c in cases]


def _sample(cand: Any, outcome: Any) -> RuleSample:
    """One training row — the ablation-4 on-distribution recipe."""
    delta = outcome.base_cost - outcome.cand_cost
    return RuleSample(
        compute_features(cand.lhs),
        rule_vector(cand.as_rule()),
        cand.as_rule().name,
        delta if math.isfinite(delta) else 0.0,
        cand.label,
    )


def _samples(cands: list, outcomes: list) -> list[RuleSample]:
    """Zip a candidate pool with its outcomes into ``RuleSample``s."""
    return [_sample(c, o) for c, o in zip(cands, outcomes, strict=True)]


def _tail_candidates() -> list:
    """The default-stream candidates past both splits (offsets 1152+)."""
    covered = OPEN_PAGES * 2 * PAGE
    return list(islice(enumerate_candidates(), covered, None))


def _wide_candidates(n: int, banned: set) -> list:
    """First *n* widened-grammar candidates disjoint from *banned*."""
    out: list = []
    seen: set = set()
    for cand in enumerate_candidates(WIDE_OPS):
        k = _instance_key(cand)
        if k in banned or k in seen:
            continue
        seen.add(k)
        out.append(cand)
        if len(out) >= n:
            break
    return out


def _site_terms(
    intake_n: int, seed: int
) -> tuple[list, dict[str, int]]:
    """Export zoo + variants + a capped intake prefix to IR terms."""
    torch.manual_seed(seed)
    sink = TorchSink()
    ws = [*zoo_mod.zoo(), *zoo_mod.zoo_variants()]
    if intake_n:
        ws.extend(li.real_workloads()[:intake_n])
    cases, records, rejections = li.ingest(cands=ws, sink=sink)
    meta = {
        "workloads": len(ws),
        "exported": len(records),
        "rejections": len(rejections),
    }
    return [c.term for c in cases], meta


def _site_candidates(terms: list, max_pairs: int, banned: set) -> list:
    """Near-miss candidates mined off the real terms' e-graphs."""
    cands = lp.near_miss_candidates(
        terms,
        max_dist=SITE_MAX_DIST,
        max_size=SITE_MAX_SIZE,
        max_pairs=max_pairs,
    )
    out: list = []
    seen: set = set()
    for cand in cands:
        k = _instance_key(cand)
        if k in banned or k in seen:
            continue
        seen.add(k)
        out.append(cand)
    return out


def _corpus_rule_samples(
    per_family: int, seed: int, terms: list
) -> tuple[list, int]:
    """``rule_samples`` over the family mixture + real exported terms.

    The off-distribution control: program-level features and shipped
    rule vectors (ablation 4's losing recipe), re-fed with real
    ``nn.Module`` exports.
    """
    rules = all_rules()
    progs = families.mixture_programs(
        list(families.FAMILIES), per_family, seed
    )
    samples = [s for p in progs for s in rule_samples(p, rules)]
    n_fam = len(samples)
    for t in terms:
        samples.extend(rule_samples(t, rules))
    return samples, n_fam


# ---------------------------------------------------------------------------
#  Training + scoring
# ---------------------------------------------------------------------------


def _ap_of(
    samples: list[RuleSample],
    flat_pool: list,
    flat_outcomes: dict,
    *,
    epochs: int,
    hidden: int,
    seed: int,
) -> float:
    """Train ``RuleValueNet`` on *samples*; AP on the held-out pool."""
    model = train_rule_value(
        samples, epochs=epochs, hidden=hidden, device="cpu", seed=seed
    )
    x = torch.tensor(
        [
            input_vector(
                compute_features(c.lhs), rule_vector(c.as_rule())
            )
            for c in flat_pool
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        scores = [float(v) for v in model(x)]
    return round(pb._ap(scores, flat_outcomes), 4)


def _neural_ranker_arm(model: Any) -> Any:
    """Episode arm: evaluate the net's top pending candidate; drain.

    The ``RuleValueNet`` analogue of ``pb._heuristic_arm`` — same
    drain-then-stop behavior, so ``evals_per_ep`` is comparable.
    """

    def make(i: int) -> Any:
        del i

        def player(state: Any) -> Any:
            if not state.pending:
                return Action("stop", {})
            x = torch.tensor(
                [
                    input_vector(
                        compute_features(state.pool[j].lhs),
                        rule_vector(state.pool[j].as_rule()),
                    )
                    for j in state.pending
                ],
                dtype=torch.float32,
            )
            with torch.no_grad():
                scores = model(x)
            best = int(torch.argmax(scores).item())
            j = state.pending[best]
            return Action("evaluate", {"i": j, "cand": state.pool[j]})

        return player

    return make


def _pool_stats(cands: list, outcomes: list) -> dict[str, Any]:
    """Size/label census of one measured pool."""
    return {
        "n": len(cands),
        "positives": sum(
            1
            for o in outcomes
            if math.isfinite(o.base_cost - o.cand_cost)
            and o.base_cost > o.cand_cost
        ),
        "useful": sum(1 for o in outcomes if o.useful),
        "num_true": sum(1 for o in outcomes if o.num_true is True),
        "new": sum(1 for o in outcomes if o.new),
    }


# ---------------------------------------------------------------------------
#  The run
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Progress on stderr — the run's per-candidate evals are minutes."""
    print(f"[neural_corpus] {msg}", file=sys.stderr, flush=True)


def run(
    seed: int,
    half: int,
    page: int,
    *,
    wide: int,
    intake_seeds: int,
    site_max_pairs: int,
    corpus_per_family: int,
    neural_epochs: int,
    hidden: int,
    skip_episodes: bool,
    skip_corpus: bool,
    cache: str | None,
) -> dict[str, Any]:
    """Measure every pool once, then train each arm and score AP."""
    t0 = time.time()
    store = _load_store(cache)
    _install_eval_limit()
    # --- the held-out pool: odd pages, memoized verdicts (baseline) --
    ev = og.gen_cases(seed, half, page=page, start=1, step=2)
    ev_pools = _stream_pool(ev)
    raw_pool = [c for pool in ev_pools for c in pool]
    # Ban every held-out instance — measured or not — from training
    # pools; an unmeasurable verdict still marks the pair as held out.
    banned = {_instance_key(c) for c in raw_pool}
    _log("measuring held-out pool")
    flat_eval_raw = _measure(
        raw_pool, seed + pb.HELD_OUT_OFFSET, store, "heldout", cache
    )
    flat_pool, flat_eval, heldout_skipped = _kept(
        raw_pool, flat_eval_raw
    )
    by_key: dict = {}
    flat_outcomes = {i: o for i, o in enumerate(flat_eval)}
    for c, o in zip(flat_pool, flat_eval, strict=True):
        by_key.setdefault(pb._cand_key(c), o)
    n_useful = sum(1 for o in flat_outcomes.values() if o.useful)

    # --- the on-distribution base: even pages -----------------------
    _log("measuring even-page train pool")
    train_cases = og.gen_cases(seed, half, page=page, start=0, step=2)
    train_pools = _stream_pool(train_cases)
    raw_train = [c for pool in train_pools for c in pool]
    train_out_raw = _measure(
        raw_train,
        seed + TRAIN_SEED_OFFSET,
        store,
        "train",
        cache,
    )
    banned |= {_instance_key(c) for c in raw_train}
    base_cands, train_out, train_skipped = _kept(
        raw_train, train_out_raw
    )
    base_samples = _samples(base_cands, train_out)

    pools: dict[str, dict[str, Any]] = {
        "even-pages": {
            **_pool_stats(base_cands, train_out),
            "skipped": train_skipped,
        }
    }

    # --- tail: the default stream's uncovered offsets ---------------
    tail = _tail_candidates()
    tail_out_raw = _measure(
        tail, seed + 2_000_003, store, "tail", cache
    )
    banned |= {_instance_key(c) for c in tail}
    tail, tail_out, tail_skipped = _kept(tail, tail_out_raw)
    pools["stream-tail"] = {
        **_pool_stats(tail, tail_out),
        "skipped": tail_skipped,
    }

    # --- wide: the full-grammar stream ------------------------------
    wide_cands: list = []
    wide_out: list = []
    if wide:
        t = time.time()
        _log(f"enumerating + measuring {wide} wide-grammar candidates")
        wide_cands = _wide_candidates(wide, banned)
        wide_out_raw = _measure(
            wide_cands, seed + 2_100_001, store, "wide", cache
        )
        banned |= {_instance_key(c) for c in wide_cands}
        wide_cands, wide_out, wide_skipped = _kept(
            wide_cands, wide_out_raw
        )
        pools["wide-grammar"] = {
            **_pool_stats(wide_cands, wide_out),
            "skipped": wide_skipped,
            "ops": len(WIDE_OPS),
            "gen_s": round(time.time() - t, 1),
        }

    # --- sites: real nn.Module near-miss candidates -----------------
    site_cands: list = []
    site_out: list = []
    terms: list = []
    if site_max_pairs or not skip_corpus:
        t = time.time()
        _log("ingesting zoo + intake seeds")
        terms, meta = _site_terms(intake_seeds, seed)
        meta["ingest_s"] = round(time.time() - t, 1)
        if site_max_pairs:
            _log("mining + measuring real-program candidates")
            site_cands = _site_candidates(terms, site_max_pairs, banned)
            site_out_raw = _measure(
                site_cands, seed + 2_200_001, store, "sites", cache
            )
            site_cands, site_out, site_skipped = _kept(
                site_cands, site_out_raw
            )
            pools["zoo-sites"] = {
                **meta,
                **_pool_stats(site_cands, site_out),
                "skipped": site_skipped,
            }
        else:
            pools["zoo-sites"] = meta

    tail_samples = _samples(tail, tail_out)
    wide_samples = _samples(wide_cands, wide_out)
    site_samples = _samples(site_cands, site_out)
    all_on_dist = [
        *base_samples,
        *tail_samples,
        *wide_samples,
        *site_samples,
    ]

    arms: dict[str, list[RuleSample]] = {
        "ondist-576": base_samples,
        "extra-only": [*tail_samples, *wide_samples, *site_samples],
        "ondist+tail": [*base_samples, *tail_samples],
        "ondist+sites": [*base_samples, *site_samples],
        "ondist+wide": [*base_samples, *tail_samples, *wide_samples],
        "ondist+all": all_on_dist,
    }
    if not skip_corpus:
        _log("building corpus rule_samples")
        corpus, n_fam = _corpus_rule_samples(
            corpus_per_family, seed, terms
        )
        arms["corpus-mixture"] = [s for s in corpus[:n_fam]]
        arms["corpus+zoo"] = corpus
        pools["corpus"] = {
            "n": len(corpus),
            "mixture_rows": n_fam,
            "zoo_rows": len(corpus) - n_fam,
            "positives": sum(1 for s in corpus if s.delta_cost > 0),
            "mixture_positives": sum(
                1 for s in corpus[:n_fam] if s.delta_cost > 0
            ),
            "zoo_positives": sum(
                1 for s in corpus[n_fam:] if s.delta_cost > 0
            ),
        }

    aps: dict[str, float] = {}
    for name, samples in arms.items():
        if not samples:
            continue
        _log(f"training arm {name} (n={len(samples)})")
        aps[name] = _ap_of(
            samples,
            flat_pool,
            flat_outcomes,
            epochs=neural_epochs,
            hidden=hidden,
            seed=seed,
        )
    if all_on_dist:
        aps["ondist+all@h96"] = _ap_of(
            all_on_dist,
            flat_pool,
            flat_outcomes,
            epochs=neural_epochs,
            hidden=96,
            seed=seed,
        )

    anchors: dict[str, float] = {
        "heuristic": round(
            pb._ap(
                [lp._heuristic_score(c) for c in flat_pool],
                flat_outcomes,
            ),
            4,
        ),
        "trained-linear": round(
            pb._ap(
                [
                    og.OpenLearnedPlayer(seed).rank_score(c)
                    for c in flat_pool
                ],
                flat_outcomes,
            ),
            4,
        ),
        "random": round(n_useful / max(len(flat_pool), 1), 4),
    }

    episodes: dict[str, dict] = {}
    if not skip_episodes:
        orig_evaluate = lp.evaluate
        lp.evaluate = pb._patched_evaluate(by_key)
        try:
            rnd = play._random_policy(og.legal_actions)
            episodes["random"] = pb._play_open(
                lambda i: rnd(seed + 10_000 + i), ev, OPEN_BUDGET
            )
            episodes["heuristic"] = pb._play_open(
                pb._heuristic_arm(), ev, OPEN_BUDGET
            )
            baked = og.OpenLearnedPlayer(seed)
            episodes["trained-linear"] = pb._play_open(
                lambda i: baked.frozen(seed + 30_000 + i),
                ev,
                OPEN_BUDGET,
            )
            # Neural rankers as episode arms — the same seeded
            # ``train_rule_value`` calls as the AP rows above, so the
            # weights are identical to the ``ondist-576`` /
            # ``ondist+all`` rows.
            net = train_rule_value(
                base_samples,
                epochs=neural_epochs,
                hidden=hidden,
                device="cpu",
                seed=seed,
            )
            episodes["neural-ondist-576"] = pb._play_open(
                _neural_ranker_arm(net), ev, OPEN_BUDGET
            )
            if all_on_dist:
                net_all = train_rule_value(
                    all_on_dist,
                    epochs=neural_epochs,
                    hidden=hidden,
                    device="cpu",
                    seed=seed,
                )
                episodes["neural-ondist+all"] = pb._play_open(
                    _neural_ranker_arm(net_all), ev, OPEN_BUDGET
                )
        finally:
            lp.evaluate = orig_evaluate

    return {
        "held_out": {
            "pool": len(flat_pool),
            "useful": n_useful,
            "skipped": heldout_skipped,
        },
        "pools": pools,
        "ap": aps,
        "anchors": anchors,
        "episodes": episodes,
        "train_sizes": {k: len(v) for k, v in arms.items()},
        "wall_s": round(time.time() - t0, 1),
    }


def _print(res: dict[str, Any]) -> None:
    """Render the pool census and the AP table."""
    print("== pools ==")  # stdout-compat
    for name, st in res["pools"].items():
        pos = st.get("positives", "—")
        skipped = st.get("skipped", 0)
        print(  # stdout-compat
            f"  {name:<14} n={st.get('n', '—'):>6} pos={pos:>5} "
            f"useful={st.get('useful', '—')} skipped={skipped}"
        )
    print(  # stdout-compat
        f"== held-out AP (pool={res['held_out']['pool']}, "
        f"useful={res['held_out']['useful']}) =="
    )
    for name, ap in sorted(
        (kv for kv in res["ap"].items() if not kv[0].endswith("@h96")),
        key=lambda kv: -kv[1],
    ):
        print(  # stdout-compat
            f"  {name:<20} {ap:.4f}   (n={res['train_sizes'][name]})"
        )
    if "ondist+all@h96" in res["ap"]:
        print(  # stdout-compat
            f"  {'ondist+all@h96':<20} "
            f"{res['ap']['ondist+all@h96']:.4f}   "
            f"(n={res['train_sizes']['ondist+all']}, hidden=96)"
        )
    print("-- anchors --")  # stdout-compat
    for name, ap in res["anchors"].items():
        print(f"  {name:<20} {ap:.4f}")  # stdout-compat
    for name, r in res.get("episodes", {}).items():
        print(  # stdout-compat
            f"  ep {name:<16} reward={r['reward']:.4f} "
            f"use/eval={r['useful_per_eval']:.4f} "
            f"evals={r['evals_per_ep']:.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the corpus arms and print the AP table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--open-pages", type=int, default=OPEN_PAGES)
    ap.add_argument("--page", type=int, default=PAGE)
    ap.add_argument(
        "--wide",
        type=int,
        default=1728,
        help="widened-grammar candidates to add (0 disables)",
    )
    ap.add_argument(
        "--intake-seeds",
        type=int,
        default=48,
        help="intake real workloads exported as mining seeds",
    )
    ap.add_argument(
        "--site-max-pairs",
        type=int,
        default=SITE_MAX_PAIRS,
        help="cap on mined real-program candidates (0 disables)",
    )
    ap.add_argument(
        "--corpus-per-family", type=int, default=CORPUS_PER_FAMILY
    )
    ap.add_argument("--neural-epochs", type=int, default=NEURAL_EPOCHS)
    ap.add_argument("--hidden", type=int, default=NEURAL_HIDDEN)
    ap.add_argument(
        "--skip-episodes",
        action="store_true",
        help="skip the anchor episode rows (AP only)",
    )
    ap.add_argument(
        "--skip-corpus",
        action="store_true",
        help="skip the off-distribution rule_samples arms",
    )
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument(
        "--cache",
        type=str,
        default="/tmp/neural_corpus_cache.pt",
        help="per-candidate verdict checkpoint (survives a kill)",
    )
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    res = run(
        args.seed,
        args.open_pages,
        args.page,
        wide=args.wide,
        intake_seeds=args.intake_seeds,
        site_max_pairs=args.site_max_pairs,
        corpus_per_family=args.corpus_per_family,
        neural_epochs=args.neural_epochs,
        hidden=args.hidden,
        skip_episodes=args.skip_episodes,
        skip_corpus=args.skip_corpus,
        cache=args.cache,
    )
    _print(res)
    if args.json:
        import json

        with open(args.json, "w") as f:
            json.dump(
                {
                    "seed": args.seed,
                    "protocol": {
                        "open_pages": args.open_pages,
                        "open_page": args.page,
                        "open_held_out": "odd pages (start=1, step=2)",
                        "open_train": "even pages (start=0, step=2)",
                        "wide_ops": list(WIDE_OPS),
                        "default_ops": list(DEFAULT_OPS),
                        "neural_epochs": args.neural_epochs,
                        "hidden": args.hidden,
                        "oracle_pairing": (
                            "memoized per-candidate-seeded verdicts"
                        ),
                    },
                    "results": res,
                },
                f,
                indent=1,
                default=str,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

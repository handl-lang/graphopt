# The `open-gated` seat — shipped-player verification

**Date:** 2026-10-25 · **Script:** `bench/drivers/policy/open_gated_eval.py`
(new — the round-10 memoized-verdict protocol, replayed for the named
seat) · **Seat:** `gamekit.players.gated` `open-gated`
(`variant={"baked": True}` → `gamekit.gamedata.OPEN_GATED_PAYLOAD`,
gamekit@fd9137f, repinned `1b0723d`) · **Seed:** 0 · **Runner:**

```bash
.venv/bin/python bench/drivers/policy/open_gated_eval.py \
    --cache /tmp/gate_cache.pkl.censored-bak \
    --json /tmp/open_gated_seat.json
```

(reused round 10's memoized verdicts — the `(eval, train, by_key)`
tuple is `stop_gate.py`'s `--cache` format; the oracle pass is
skipped, so the seat faces literally the same per-candidate-seeded
verdicts the arm table did.)

> **Status:** complete (~2 s — cache hit; seat play + decode check +
> anchor).  **Verdict: the shipped seat reproduces the round-10
> measured arm exactly — every recorded digit.**

## What was pinned

The `open-gated` seat exists (verified: `registry.make("open-gated",
"open", ctx={})` builds) and its `PlayerSpec.record` quotes round
10 — **0.2893 reward/ep, 0.2101 AP, 26 stops** — but the seat was
never played through the shipped decode path in-tree.  This round
plays it under the identical protocol: held-out odd pages
(`gen_cases(seed, 36, page=16, start=1, step=2)` — 36 episodes,
`budget = page+1 = 17`), verdicts patched to the recorded oracle
table, `sp._play_pools` tallying.

Two cross-checks make the pin honest:

* `payload decode-check` — the *same* `OPEN_GATED_PAYLOAD` pushed
  through `stop_gate._play_payload` (the research decode path the
  arm table measured).  Seat ≡ payload is the proof that
  `gated.make`'s JSON decode is the measured decode;
* `anchor:baked-prior` — the ungated shipped-prior anchor replayed.
  It lands its recorded row (0.2758 / 13.056 evals / 0.0106
  use/eval) digit-for-digit, so the cached verdicts *are* the
  recorded oracle table — nothing drifted underneath the seat.

## The table — seat vs recorded

| row | reward | greedy | use/eval | evals/ep | stops | AP |
|---|---|---|---|---|---|---|
| **open-gated (seat)** | **0.2893** | — | 0.0115 | 14.556 | **26** | **0.2101** |
| payload decode-check | 0.2893 | 0.3524 | 0.0115 | 14.556 | 26 | 0.2101 |
| anchor:baked-prior | 0.2758 | — | 0.0106 | 13.056 | — | 0.2101 |
| *recorded `vhead:baked`* | *0.2893* | *0.3524* | *0.0115* | *14.556* | *26* | *0.2101* |

**Exact reproduction.**  The seat's 26 stops, 14.556 evals/ep and
0.2893 reward are the `vhead:baked` row verbatim; the decode-check
row is identical to the seat in every field (sampled *and* greedy —
the seat's inner seeds `seed + 40_000 + i` are the sampled arm's by
construction).  Even the rounding disclosure is a no-op: the shipped
`w_stop` literals are the round-10 head's 4-decimal `|w|>1e-4`
snapshot, and on this protocol the snapshot changes nothing.

## Disclosures — what this did and did not measure

* **Protocol** — 36 odd-page held-out episodes (the round-10 doc's
  "18" is a stale count; `pool.held_out` = 576 = 36 × 16 in both the
  recorded JSON and this replay).  Censored verdicts replayed as
  `relation="censored"` (3/1 152 — same candidates, same verdicts).
* **`--player open-gated` CLI ≠ this protocol.**  `play.py
  --domain open --player open-gated` does build and play the seat,
  but its eval stream is `d.cases(seed+7919, n)` =
  `gen_cases(seed, n)` — `page=48`, `start=0`, `step=1`, and
  `gen_cases` ignores the seed, so the "held-out" pages overlap the
  train stream and the oracle runs *live* (unmemoized
  `lp.evaluate`, reward on a different pool composition).  Measured
  anyway for the record: `--eval-cases 1 --budget 17` →
  `open-gated 1.67, random 2.51` — a build/smoke signal, not the
  bake number.  The named seat's measured record is the memoized
  protocol above.
* **`score_fn` AP = 0.1284** — the seat deposits
  `LearnedPlayer.rank_score` (candidate-only `w·_cand_feats`) into
  `ctx["score_fn"]`; priced here it under-reads the recorded 0.2101
  because the page-aware AP featurizes `evaluate(cand)` on the pool
  state (pool-relative `a:score_rank`/`a:score_rel` included).
  Expected, not a regression — but callers scoring the seat through
  `ctx["score_fn"]` get the candidate-only ordering, not the
  in-episode one.
* **Greedy has no shipped seat.**  `registry.make` exposes no
  greedy variant; the 0.3524/27 greedy column above is the payload
  through the round-10 decode path, recorded for completeness.

# The stopping policy, round 10 — the gate as *data*

**Date:** 2026-10-24 · **Script:** `bench/drivers/policy/stop_gate.py`
(new — same memoized-verdict protocol as `stop_isolated.py`, plus the
serialization machinery this round prices) · **Part:**
`mlopt_discovery.open_prior` (registered `open:prior`, unshipped) ·
**Seed:** 0 · **Runner:**

```bash
timeout 2400 .venv/bin/python -u bench/drivers/policy/stop_gate.py \
    --json /tmp/gate.json --cache /tmp/gate_cache.pkl --workers 8
```

> **Status:** the mechanism and the serializable form are landed and
> tested; the held-out arm table is being filled — the verdict
> memoization on this box now costs ~30–60 min serial (see *the
> infra finding* below), so the driver shards it over a process pool
> and checkpoints to a pickle cache.  This document reports the
> serialization result that does not depend on the pending numbers;
> the arm table lands with the finished run.

## The question round 9 left open

`docs/results_stop_isolated.md` landed the delivery: `GatedPlayer` +
the `forced_action` hook take the quit decision out of the ranking
softmax, and honest-κ arms clear both bake bars — but their payloads
ride a `score` *callable* (an injected value table, an in-process
calibrator), so "serializable" was asserted, not shown.  This round
prices exactly that: **can a stop decision outside the shared
softmax be expressed as pure data?**

## The serializable form — found

The `gated` seat (`gamekit.players.gated.make`) already accepts a
config whose every leaf has a JSON form:

```python
{
  "rank": {"spec": <DomainSpec.to_record()>, "weights": {f: w}},
  "stop": {"spec": <record|None>, "weights": {f: w}|None,
           "kappa": float, "scorer": <name>|None,
           "scorer_params": {...}},
}
```

* `spec` — `DomainSpec.to_record()`/`build_from_record` reify a
  stop-row spec to part tags; the registry idiom the codebase uses
  for every callable (parts/legals/ranks).
* `w_stop` + `kappa` — the pure linear path
  (`GatedPlayer(stop_feats=…, w_stop=…, kappa=…)`) implements
  `stop iff w_stop·φ_stop ≤ κ` with no callable at all; a
  single-column gate is `w_stop = {column: 1.0}` — three literals.
* `scorer` — callable-path arms (the calibrated head, the ranker's
  own margin) serialize through a *named* scorer — a `SCORERS`
  registry entry — the same registry idiom; the driver's
  `{"cal-head", "self-margin"}` table is the prototype, its params
  (weight dict, isotonic knot list) are data.
* `rank_weights` — an ordinary `{feature: weight}` table, the
  `OPEN_GAME_PLAYER_WEIGHTS` shape.

So the strict `OPEN_GAME_PLAYER_WEIGHTS`-only bake remains impossible
(the fold arm confirms the softmax still cannot carry an absolute
threshold — the recurring degenerate point), but the **gate-as-spec-field
form the engine supports is fully serializable**: `ctx["gated"]` is
data, and the driver's `_player_from_payload` rebuilds every arm from
the `json.dumps` round-trip of its payload, so a clearing arm *is*
the serialized artifact playing, not an argument that it could be.

## The evidence candidate — `open:prior`

Every priced stop column so far read a runtime seam (`set_value_preds`
behind `open:next-val`/`open:vlabel`, `set_self_scores` behind
`open:self`).  The new **`mlopt_discovery.open_prior`** part removes
the seam by construction: it prices the pending pool with the
*shipped prior's own linear model* — `w·_cand_feats` — computable at
decode from shipped tables alone:

`x:stop:prior_best` / `prior_top3` / `prior_pool` / `prior_frac`
(the rank_score view of what remains), `x:stop:prior_margin` (the
prior's *own* softmax margin — best eval logit − stop logit; κ=0 is
its literal indifference point, κ>0 a hysteresis) and
`x:stop:prior_p1` (its "keep going" conviction).  `open:next-value`'s
heuristic-priced `x:stop:best_score`/`topk_score` are the equally
seam-free comparison columns.

`tests/test_open_prior.py` pins the contract (registry, unshipped,
drained/unweighted fallbacks, exact math vs a manual `_dot` pass,
the spec-record round-trip) — 10 tests pass.

## The infra finding — the verdict memoization cost grew ~10×

The drivers' documented ~150–190 s pool memoization predates the
prim-vocabulary stream (`synop2_*` folds at depth-2/3+).  Mid-stream
`factor-*`/`pull-*` candidates now cost ~5–25 s+ each through
`lp.evaluate` (bounded-egraph `verify_law` over ~130 rules at
4 000-node saturation), so the 1 152-candidate memoization runs
~30–60 min serial on this box.  The driver therefore ships a
**parallel, checkpointed** memoizer (`--workers`, `--cache`):
per-candidate tasks preserve the seeded order bit-for-bit (verified
identical to `pb._measure_pools` on a 4-pool slice) and a partial
pickle survives a killed run.

## What the run is pricing (arm table pending)

* Port controls — incumbent / ac / oracle-chan / iso-drain / anchors.
* Non-serializable baselines — `ev-margin`/`ev-pool` on raw `delta`
  (round 7's 0.2821) and `useful`-isotonic @1.5κ (round 9's 0.2929 —
  needs the per-candidate table shipped).
* Column gates — `w_stop = {col: 1}`, κ fit on train drains by
  prefix-sum (bake-time only): `heur-best`/`top3`, `prior-best` /
  `top3` / `pool` / `frac` / `margin` / `p1`, each under the
  cold-retrained ranker (`:inc`) and the shipped prior (`:baked`),
  plus the analytic `prior-margin@0` (κ=0, no fit).
* Scorer gates — `self-margin` (the ranker's own `w·_cand_feats`
  margin as a named scorer) and `cal-head` (logistic head + isotonic
  knots + break-even p* — params are data).
* Heads as payloads — `vreg` (κ=0, pure `w_stop`) and `head-iso`
  over incumbent and `open:prior` schemas; `vhead-fold` is the
  shared-softmax A/B.
* Seat proof — `registry.make("gated","open",ctx)` must land
  identical to the direct-build row.

## Verdict so far

Serializable form: **yes** — `ctx["gated"]` is a pure-data payload
(`{spec record, w_stop, κ, scorer name+params, rank_weights}`), and
`open:prior` gives the first seam-free priced stop columns to fit a
single-column gate in three literals.  Whether any arm clears the
bake bars (reward ≥ 0.273 ∧ AP ≥ 0.210, honestly) is what the in-flight
table decides; the next lever it names, whatever the numbers, is the
same: the *evidence* — a per-candidate usefulness signal that
transfers held-out — now joined by the *infra* note that the grown
law bank repriced the oracle itself.

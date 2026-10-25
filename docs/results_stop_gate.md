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

> **Status:** complete — the held-out arm table landed 2026-10-25
> (~40 min wall: parallel+checkpointed memoization over 8 workers;
> 3/1 152 candidates censored at a 20-min per-candidate cap — see
> *the infra finding* below).  **Verdict: the serializable
> out-of-softmax gate exists as pure data AND clears the bake gate —
> by honestly-stopping arms, not the never-stop confound.**

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

## What the run priced (18 held-out episodes, 576 candidates)

* Port controls — incumbent / ac / oracle-chan / iso-drain / anchors.
* Non-serializable baselines — `ev-margin`/`ev-pool` on raw `delta`
  (round 7's 0.2821) and `useful`-isotonic @1.5κ (round 9's 0.2929 —
  needs the per-candidate table shipped; measured 0.0 here — it
  halted every episode at eval 1, an over-stop degenerate).
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
  shared-softmax A/B (degenerate — the recurring result).
* Seat proof — `registry.make("gated","open",ctx)` replays
  `vhead:inc` at 0.3261/26 stops — the registry path is proven.

## The held-out arm table

Bake bars: reward ≥ 0.273 ∧ AP ≥ 0.210, serializable, honestly
stopping (never-stop clears are the round-7 confound).

| arm | reward | greedy | use/eval | evals/ep | stops | AP | verdict |
|---|---|---|---|---|---|---|---|
| incumbent | 0.2768 | 0.3564 | 0.0091 | 15.28 | 15 | 0.2074 | AP bar missed |
| oracle-chan | 0.3476 | 0.3835 | 0.0414 | 4.03 | 35 | 0.2074 | reference only |
| iso-drain | 0.3217 | 0.3564 | 0.0104 | 16.00 | 0 | 0.2074 | non-serializable |
| ev-margin:delta | 0.3188 | 0.3371 | 0.0109 | 15.28 | 22 | 0.2074 | non-serializable |
| ev-pool:delta | 0.3137 | 0.3321 | 0.0111 | 15.06 | 22 | 0.2074 | non-serializable |
| vhead:inc | 0.3261 | 0.3499 | 0.0115 | 14.53 | 26 | 0.2074 | AP bar missed |
| **vhead:baked** | **0.2893** | 0.3524 | 0.0115 | 14.56 | **26** | **0.2101** | **clears — honest** |
| **head-iso:baked** | **0.2892** | — | — | 14.56 | **26** | **0.2101** | **clears — honest** |
| **gate:prior-frac:baked** | **0.2814** | 0.3503 | 0.0104 | 15.97 | **35** | **0.2101** | **clears — late-stopper** |
| gate:prior-best:baked | 0.2851 | 0.3595 | 0.0104 | 16.00 | 3 | 0.2101 | clears — near-never-stop |
| gate:prior-top3:baked | 0.2851 | 0.3595 | 0.0104 | 16.00 | 3 | 0.2101 | near-never-stop |
| gate:prior-pool:baked | 0.2851 | 0.3595 | 0.0104 | 16.00 | 3 | 0.2101 | near-never-stop |
| gate:prior-margin:baked | 0.2851 | 0.3595 | 0.0104 | 16.00 | 3 | 0.2101 | near-never-stop |
| gate:self-margin:baked | 0.2851 | 0.3595 | 0.0104 | 16.00 | 3 | 0.2101 | near-never-stop |
| gate:prior-p1:baked | 0.2850 | 0.3597 | 0.0104 | 16.00 | **0** | 0.2101 | never-stop — confound |
| gate:prior-margin@0:baked | 0.2850 | 0.3590 | 0.0104 | 16.00 | **0** | 0.2101 | never-stop — confound |
| vhead+prior:baked | 0.1849 | — | — | 6.24 | 35 | 0.2101 | over-stops — fails |
| head-iso+prior:baked | 0.2414 | — | — | 12.71 | 36 | 0.2101 | reward bar missed |

(`:inc` twin rows — incumbent rank weights + same gate — earn
0.318–0.326 on reward but hold AP at 0.2074: **the bake-worthy AP
0.2101 comes from the `baked` rank table**, the gate contributes the
reward headroom over it.)

## Verdict

Serializable form: **yes, and it clears.**  `ctx["gated"]` is a
pure-data payload (`{spec record, w_stop, κ, scorer name+params,
rank_weights}`); every arm above played as its own JSON round-trip,
and `registry.make("gated", "open", ctx)` reproduces the direct
build exactly.

The honest clears — arms that actually gate, not drain:

* **`vhead:baked` / `head-iso:baked` — 0.2893 / 0.2892 reward,
  0.2101 AP, 26 stops, 14.6 evals/ep.**  The fitted stop heads
  (`w_stop` literals / named `cal-head` scorer + knot list) stop in
  most episodes while holding incumbent-level reward.  This is the
  first serializable arm that both clears both bars AND has real
  gating behaviour (+0.0125 reward over incumbent while *saving*
  ~1.4 evals/ep).
* **`gate:prior-frac:baked` — 0.2814 / 0.2101, 35 stops.**  A
  three-literal gate (`w_stop={x:stop:prior_frac:1}, κ=0.067`) that
  also clears — but stops so late (15.97 evals/ep ≈ full drain)
  that its "gate" is nearly decorative.

The confound class: `prior-p1` and `prior-margin@0` clear numerically
at **stops=0** — the same never-stop artefact round 7 disqualified —
and the `prior-best/top3/pool/margin`/`self-margin` family at
stops=3 is the near-never-stop edge of it.  Their +0.008 over
incumbent is the baked rank table's gain, not gating.

**Censored disclosure:** 3/1 152 candidates (all held-out eval
pools) hit the 20-min per-candidate cap — they merge but their
post-merge certificate extraction runs unbounded (see *infra finding*
below).  They report `relation="censored"` ⇒ `derivable=False`: if
any were paying candidates, measured rewards *underestimate* — the
clears are conservative, not inflated.

**Bake candidate:** `vhead:baked`'s payload — `rank` =
`{spec: incumbent, weights: <baked rank table>}`, `stop` =
`{spec: incumbent stop rows, weights: w_v literals, kappa: 0.0}` —
is ship-able as a `ctx["gated"]` payload literally (the gamekit
`gated.make` seat already consumes the shape; binding it as a named
player seat needs the gamekit-repo change — flagged, not done here).

## The infra finding — follow-on: post-merge certificate extraction is unbounded

The verdict memoization now costs ~40 min wall ×8 workers (was
~150 s serial before the prim-vocabulary stream).  Worse, **3
candidates never finished**: `verify_law`'s saturation is bounded
(max_nodes 4 000, rule budgets), but when sides *merge*,
`eg.certificate` + `verify_certificate` replay on the dense e-graph
ran 2.5+ h at 100 % CPU with no progress marker — a pathological
(bug-looking) case in the substrate's proof machinery, not "slow
eval".  The driver gained `--cand-timeout` (SIGALRM per candidate →
`relation="censored"`) and `as_completed` checkpointing (ordered
`ex.map` had buffered finished results behind the outliers, so the
checkpoint couldn't see them).  **Upstream fix needed in eqsat's
certificate path** — the cap is a measurement workaround, not a cure.

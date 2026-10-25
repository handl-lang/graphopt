# The stopping policy, round 11 — verdict-visible evidence

**Date:** 2026-10-25 · **Script:** `bench/drivers/policy/stop_verdict.py`
(new — same memoized-verdict protocol as `stop_isolated.py`) ·
**Part:** `mlopt_discovery.open_verdict` (registered `open:verdict`,
unshipped) · **Seed:** 0 · **Runner:**

```bash
timeout 2400 .venv/bin/python bench/drivers/policy/stop_verdict.py \
    --json /tmp/vd.json --cache /tmp/gate_cache.pkl
```

(reused `stop_gate.py`'s memoized verdicts — the `(eval, train,
by_key)` tuple is format-identical; the oracle pass is skipped.)

> **Verdict: the signal transfers.**  `x:stop:vd_live` posts decision-row
> AP **0.631** — the first column past the ~0.40 ranking-view ceiling
> every family hit before.  The trained heads reach held-out usefulness
> AP 0.449–0.535.  The arm-level double-clear did not land yet:
> `head+vd-ep` clears reward (0.2866 ≥ 0.273) with real stops (32,
> 8.0 evals/ep) but holds the incumbent rank's AP floor (0.2074 <
> 0.210) — the bakeable pairing is with the `baked` rank table
> (0.2101) proven in `results_stop_gate.md`.

## What the part emits

`open:verdict` prices *stop* rows from certified evidence, not nets —
two sources the domain itself can vouch for:

* **episode verdicts** (`vd_*`): banked payers in the stream so far
  (`vd_banked`, `vd_use`), pending members sharing form/shape with a
  certified payer or a certified-dead sample (`vd_live`,
  `vd_dead`, `vd_above`), and the *dead ceiling* — the highest
  score an evaluated member carried while coming back non-paying
  (`vd_headroom`).
* **corpus verdict counts** (`cr_*`): pending candidates' hit-rates
  in a closed-form signature census (`cr_top`, `cr_pool`, `cr_use`),
  the table's certified-dead mass and coverage (`cr_dead`, `cr_cov`),
  and the coarse form-level fallback (`cr_ftop`, `cr_fpool`).
  Corpus: 439 signatures / 28 forms, 26 payers, 7 useful; eval-sig
  coverage 224/576.  Held-out candidates only ever read
  train-certified counts; train rows are jackknifed against their
  own page's contribution (the count-table OOF analogue).

## Decision-row APs — the transfer test (target > 0.40)

Head `head+verdict` (incumbent features + `vd_*`/`cr_*`, held-out
eval AP **0.5291**):

| column | solo | LOO | note |
|---|---|---|---|
| `x:stop:vd_live` | **0.631** | 0.383 | load-bearing — removing it drops the head hardest (d=−0.146) |
| `x:stop:vd_use` | 0.317 | 0.579 | — |
| `x:stop:vd_banked` | 0.245 | 0.542 | — |
| `x:stop:vd_headroom` | 0.227 | 0.530 | dead-ceiling — weak alone |
| `x:stop:vd_dead` | 0.131 | 0.532 | — |
| `x:stop:vd_above` | 0.218 | 0.543 | — |
| `x:stop:cr_ftop` | 0.467 | 0.506 | coarse form rate — second-best solo |
| `x:stop:cr_fpool` | 0.422 | 0.527 | — |
| `x:stop:cr_{top,pool,use,dead,cov}` | 0.300 | ~0.53 | all seven `cr_*`s solo-identical in the `head+vd-ep` head (0.4442) — pool-level constants collapse to the same signal |

Head `head+vd-ep`: eval AP 0.4486; `head+stop+verdict`: 0.5349.
Every trained head lands ≥0.45 — comfortably over the ceiling.

## The held-out arm table

| arm | reward | greedy | use/eval | evals/ep | stops | AP | verdict |
|---|---|---|---|---|---|---|---|
| incumbent | 0.2768 | 0.3564 | 0.0091 | 15.28 | 15 | 0.2074 | — |
| ac | 0.2415 | 0.3547 | 0.0096 | 14.50 | 18 | 0.0985 | AP collapse |
| verdict | 0.2393 | 0.2826 | 0.0091 | 15.28 | 15 | **0.0165** | **softmax fold collapses AP** |
| ac+verdict | 0.2234 | 0.2697 | 0.0103 | 13.44 | 21 | **0.0105** | same |
| head+verdict | 0.3217 | 0.3564 | 0.0104 | 16.00 | 0 | 0.2074 | never-stops |
| head+stop+verdict | 0.3218 | 0.3565 | 0.0104 | 16.00 | 1 | 0.2074 | never-stops |
| **head+vd-ep** | **0.2866** | 0.3295 | 0.0173 | **8.03** | **32** | 0.2074 | clears reward; AP floor |

Anchors: baked-prior 0.2758/0.2101 · heuristic 0.3595/0.0450 ·
random 0.1339/0.0123.

## Verdict

* **The evidence lever is real.**  `vd_live` (pending sharing form
  with a certified payer, minus the dead-ceiling view) is the first
  "pending has value" signal that survives held-out — exactly the
  family round 7 said was missing.
* **Folded into the softmax it still dies.**  `verdict` /
  `ac+verdict` collapse ranking AP to ~0.01 — the seventh-ish
  recurrence of the shared-softmax degeneracy.  These columns belong
  on the *gate* side (`ctx["gated"]`), not the rank softmax.
* **Arms miss the double-bar only on the rank side.**  `head+vd-ep`
  already clears reward while stopping 32 times (8.0 evals/ep, half
  the drain); its 0.2074 is the incumbent ranker's floor, and
  `stop_gate`'s `baked` rank table prices the missing +0.003.
* **Next (round 12):** `vhead`-class gate over the `vd_*`/`cr_*`
  stop spec × the `baked` rank table — i.e. the two proven halves in
  one payload: evidence features that transfer + the serializable
  gate seat that already cleared.  Corpus coverage (224/576 eval
  sigs) is the headroom knob — denser census signatures should widen
  the `cr_*` margin.

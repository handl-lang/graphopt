# The stopping policy, round 12 — evidence × the serializable gate

**Date:** 2026-10-25 · **Script:** `bench/drivers/policy/stop_evidence.py`
(new — reuses `stop_gate.py`'s payload/drain machinery and
`stop_verdict.py`'s corpus seam + jackknife) · **Parts:**
`mlopt_discovery.open_verdict` (registered `open:verdict`, unshipped)
· **Seed:** 0 · **Runner:**

```bash
timeout 2400 .venv/bin/python -u bench/drivers/policy/stop_evidence.py \
    --json /tmp/evidence.json --cache /tmp/gate_cache.pkl --cand-timeout 1200
```

(reused `stop_gate.py`'s memoized verdicts — identical
`gen_cases` ⇒ the whole oracle pass is skipped; the run cost **37 s**
wall.  The 3 previously-censored candidates stay `relation="censored"`.)

> **Verdict: the composition mostly does NOT clear — the evidence
> columns are the best stop predictors measured yet (head eval AP
> 0.627, `vd_live` solo 0.695 under the baked ordering) but every
> fitted verdict-spec gate over-stops on held-out pages.**  The one
> honest combined clear is a three-literal single-column gate,
> `gate:vd-headroom:baked` — 0.2857 / 0.2101, 8 stops — real but
> thin gating (fires in 8/36 episodes, saves ~0.14 evals/ep).  The
> substantive bake candidate stays `vhead:baked` / `head-iso:baked`
> (round 10), both reproduced in this run.

## The question

Round 10 proved the `gated` seat is pure data and that the `baked`
rank table is what carries the AP bar (0.2101 vs the incumbent's
0.2074).  Round 11 proved the `open:verdict` columns are the first
stop evidence that *transfers* (`vd_live` decision-row AP 0.631) —
but its best arm `head+vd-ep` (0.2866, 32 stops) sat on the
incumbent AP floor.  Round 12 prices the obvious composition:
**`open:verdict` stop spec × `baked` rank table, as serialized
`gated` payloads** — plus the honest controls (the `:inc` twins, and
same-run replays of `vhead:baked` and `head+vd-ep`).

Honesty plumbing is unchanged: train-side `+vd` drains read
*jackknifed* corpus tables (each pool minus its own page's
contribution), eval drains/plays read the full train tally, and the
`+vd-ep` variants run with the seam cleared — episode-only columns,
no corpus table at decode.  Corpus: 439 signatures / 28 forms, 26
payers, 7 useful; eval signature coverage 224/576.

## Decision-row APs — the fitted spec transfers better than round 11

The logistic head over the full verdict spec (`headk+vd:*` /
`head-iso+vd:*`), evaluated on the held-out greedy-drain decision
rows:

| fit | eval drain | head eval AP |
|---|---|---|
| `+vd` head | inc ranker | 0.5291 |
| `+vd` head | **baked ranker** | **0.6271** |
| `+vd-ep` head | inc ranker | 0.4486 |
| `+vd-ep` head | baked ranker | 0.5842 |

Under the *baked* ordering the same evidence predicts better — the
columns see cleaner decision states.  Per-column solo/LOO on the
`(vd, baked)` drain (`loo_delta` vs the 0.6271 head):

| column | solo | solo (:inc drain) | LOO | Δ |
|---|---|---|---|---|
| `x:stop:vd_live` | **0.695** | 0.631 | 0.376 | **−0.251** — load-bearing |
| `x:stop:cr_ftop` | 0.477 | 0.467 | 0.602 | −0.025 |
| `x:stop:cr_fpool` | 0.433 | 0.422 | 0.629 | +0.002 |
| `x:stop:vd_use` | 0.347 | 0.317 | 0.642 | +0.015 |
| `x:stop:cr_{top,pool,use}` | ~0.33 | ~0.30 | ~0.62–0.63 | ≈0 |
| `x:stop:vd_banked` | 0.282 | 0.245 | 0.626 | −0.001 |
| `x:stop:vd_{dead,above,headroom}` / `cr_{dead,cov}` | 0.12–0.24 | ~same | ~0.63 | ≈0 |

## The held-out arm table (18 odd pages, 576 candidates)

Bake bars: reward ≥ 0.273 ∧ AP ≥ 0.210 ∧ real gating (stops ≈ 0 is
the never-stop confound and does not count).

| arm | reward | greedy | use/eval | evals/ep | stops | AP | verdict |
|---|---|---|---|---|---|---|---|
| incumbent | 0.2768 | 0.3564 | 0.0091 | 15.28 | 15 | 0.2074 | — |
| oracle-chan | 0.3476 | 0.3835 | 0.0414 | 4.03 | 35 | 0.2074 | reference |
| iso-drain | 0.3217 | 0.3564 | 0.0104 | 16.00 | 0 | 0.2074 | never-stop bound |
| **vhead:baked** | **0.2893** | 0.3524 | 0.0115 | 14.56 | 26 | **0.2101** | **clears — round-10 arm, reproduces** |
| **head-iso:baked** | **0.2892** | 0.3523 | 0.0114 | 14.58 | 26 | **0.2101** | **clears — reproduces** |
| headk:baked | 0.2863 | 0.3604 | 0.0104 | 16.00 | 25 | 0.2101 | clears — boundary-thin (fires at the pool edge; saves ≤1 eval) |
| **gate:vd-headroom:baked** | **0.2857** | 0.3616 | 0.0105 | 15.86 | 8 | **0.2101** | **clears — thin (8/36 eps fire; the only combined clear)** |
| vhead+vd-ep:baked | 0.2512 | 0.3145 | 0.0138 | 10.06 | 28 | 0.2101 | over-stops — reward miss |
| vhead+vd:baked | 0.2151 | 0.2995 | 0.0154 | 7.22 | 27 | 0.2101 | over-stops — reward miss |
| head-iso+vd-ep:baked | 0.1875 | 0.3275 | 0.0093 | 8.97 | 32 | 0.2101 | over-stops |
| head-iso+vd:baked | 0.1660 | 0.2169 | 0.0142 | 5.86 | 33 | 0.2101 | over-stops |
| headk+vd:baked | 0.1654 | 0.2164 | 0.0135 | 6.17 | 33 | 0.2101 | over-stops |
| headk+vd-ep:baked | 0.1450 | 0.3308 | 0.0084 | 6.64 | 32 | 0.2101 | over-stops |
| gate:vd-above:baked | 0.1708 | 0.3254 | — | 6.36 | 36 | 0.2101 | misfires |
| gate:cr-ftop/-fpool:baked | 0.2045/0.2092 | — | — | ~7.4 | 34 | 0.2101 | over-stops |
| gate:cr-dead/-cov:baked | 0.0785/0.0886 | — | — | ~11.2 | 16–19 | 0.2101 | misfires |
| gate:vd_{banked,use,live,dead}:baked | 0.0000 | — | — | 1.00 | 36 | 0.2101 | **instant-stop degenerate** |
| gate:cr_{top,pool,use}:baked | 0.000–0.011 | — | — | ≤1.31 | 36 | 0.2101 | instant-stop degenerate |
| vhead+vd-ep:inc | **0.3270** | 0.3186 | 0.0147 | 11.33 | 28 | 0.2074 | best honest gating reward yet — AP floor |
| head-iso+vd-ep:inc | 0.2955 | 0.3251 | 0.0152 | 9.14 | 32 | 0.2074 | reward clears, AP floor |
| headk+vd-ep:inc | 0.2866 | 0.3295 | 0.0173 | 8.03 | 32 | 0.2074 | **== head+vd-ep, bit-for-bit** |
| vhead+vd:inc | 0.2606 | 0.3063 | 0.0141 | 7.86 | 24 | 0.2074 | over-stops |
| head-iso+vd:inc | 0.2247 | 0.2134 | — | 6.61 | 32 | 0.2074 | over-stops |
| headk+vd:inc | 0.2246 | 0.2130 | — | 6.67 | 32 | 0.2074 | over-stops |
| gate:vd-headroom:inc | 0.3224 | 0.3585 | 0.0105 | 15.86 | 8 | 0.2074 | clears reward, AP floor |
| head+vd (r11 replay) | 0.3217 | 0.3564 | 0.0104 | 16.00 | 0 | 0.2074 | never-stop — confound |
| head+vd-ep (r11 replay) | 0.2866 | 0.3295 | 0.0173 | 8.03 | 32 | 0.2074 | reproduces round 11 exactly |

Anchors: baked-prior 0.2758/0.2101 · heuristic 0.3595/0.0450 ·
random 0.1339/0.0123.  Corpus table: 439 sigs / 28 forms, 26 payers
/ 7 useful; eval sig coverage 224/576; sig pay-rate AP(pays) 0.098.

## Findings

* **The port controls reproduce exactly.**  `vhead:baked`
  0.2893/26 stops and `head-iso:baked` 0.2892/26 — identical to
  round 10.  `head+vd-ep` 0.2866/32/8.03 — identical to round 11.
* **The payload path is the τ-gate, literally.**  `headk+vd-ep:inc`
  — the logistic head as a `w_stop` literal with κ prefix-fit on the
  train drains — landed κ\* = −2.947749, the same threshold
  `sp._calibrate_tau` picked by replay (−2.9477), and posted the
  identical row (0.2866/32/8.03).  Round 11's mechanism is
  serializable data: `{w_stop: <w_head literals>, kappa: −2.947749}`.
* **The seat path reproduces.**  `registry.make("gated","open",ctx)`
  with `ctx["gated"] = {spec: vd_spec, weights: w_v, kappa: 0,
  rank_weights: w_baked}` replayed `vhead+vd:baked` at exactly
  0.2151/27.
* **The composition under-performs its halves.**  Every fitted
  verdict-spec gate on the baked ranker over-stops — `vhead+vd:baked`
  0.2151 (7.2 evals/ep), `headk+vd:baked` 0.1654, `vhead+vd-ep:baked`
  0.2512 — while the same heads under the incumbent ranker earn
  *more* reward (`vhead+vd-ep:inc` 0.3270 — the best honestly-gating
  number in the series) but sit on the 0.2074 AP floor.  The evidence
  predicts continue-labels *better* under baked ordering (0.627 vs
  0.529) yet the fitted operating point doesn't transfer to it — the
  drain-fit κ/heads are ranker-specific, and the baked drain's
  reward-optimal gate is not the held-out one.  Signal transfer ≠
  operating-point transfer.
* **Single-literal evidence gates are structurally degenerate for
  the banked-evidence columns.**  `vd_banked`/`vd_use`/`vd_live`/
  `vd_dead` (and `cr_top`/`cr_pool`/`cr_use`) are all ~0 at decision
  0 — nothing certified yet — so any κ ≥ 0 fires on the first state:
  36 stops at 1.00 evals/ep, reward 0.  A "certified evidence
  exists" column cannot carry an absolute threshold gate; it needs
  composition (the heads) or a decision-0 exemption.
* **The one combined clear is thin but real — and corpus-free.**
  `gate:vd-headroom:baked` (`w_stop={x:stop:vd_headroom:1}`,
  κ = −4.625, the "something pending still out-scores what already
  failed" margin): 0.2857/0.2101, 8 stops in 36 episodes —
  +0.0099 over the baked anchor while saving ~0.14 evals/ep.
  `vd_headroom` is an episode-only column, so this payload needs no
  corpus table at decode at all.

## Bake verdict

**No new honest double-bar clear from the fitted composition.**  The
combined arms that mattered (`vhead+vd:baked`, `headk+vd-ep:baked`)
miss the reward bar by 0.02–0.13 while gating genuinely — the
evidence spec is real signal with a non-transferring operating
point.  The standing bake candidate remains **round 10's
`vhead:baked` / `head-iso:baked`** payloads (reproduced here).

The marginal combined clear, if it is wanted anyway
(thin-gate caveat attached — it fires in 8/36 episodes):

```python
payload = {
  "rank": {"spec": <incumbent spec record: 9 parts, 45 features>,
           "weights": <the shipped `baked` rank table, 45 entries>},
  "stop": {"spec": <open:verdict spec record: incumbent parts +
                    "open:verdict", 61 features>,
           "weights": {"x:stop:vd_headroom": 1.0},
           "kappa": -4.625, "scorer": None, "scorer_params": None},
}
```

Corpus-free at decode (the `cr_*` columns emit 0.0 unfed and the
single weighted column ignores them).  Shipping any verdict-spec
payload needs the same two things as before: binding the `gated`
seat config (gamekit-side change, the separate in-flight task) and —
for arms that use `cr_*` columns — the corpus count-table as a
seat-side data artifact (`{signature: (payers, useful, seen)}`,
train-tallied).

**Censored disclosure:** the cache's 3 held-out candidates remain
`relation="censored"` (derivable=False) — if any were payers,
measured rewards *underestimate*; the clears are conservative.

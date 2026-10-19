# Feature-schema leaderboard

**Date:** 2026-10-19 · **Script:** `lab/schema_leaderboard.py` · **Seed:** 0
throughout · **Runner:**

```bash
timeout 1800 .venv/bin/python lab/schema_leaderboard.py --domain meta \
    --json /tmp/schema_meta.json    # ~161 s
timeout 1800 .venv/bin/python lab/schema_leaderboard.py --domain open \
    --json /tmp/schema_open.json    # ~167 s
```

## Question and method

The capacity ablations showed *features* bind, not capacity
(`RuleValueNet` h32 ≈ h96).  This demo asks the dual: does
**representation search over the feature schema** beat the shipped
schema's weight search?  Candidates are subsets × orderings of the
shipped part vocabulary — `META_PARTS` / `OPEN_PARTS` = `Bias`,
`StateBroadcast` (`st:*`), `PlayRates` (`h:*`), `OpOneHot` (`op:*`),
`ActionFeats` (`a:*`/`x:*`) — built through the one
`DomainSpec.build` path and trained **cold at a reduced budget**:

* **meta:** 15 epochs × 40 cases (the shipped bake used 30), budget 12;
  held-out `gen_cases(seed + 7919, 16)`.
* **open:** 3 epochs × 36 even pages (shipped: 5), budget 17; held-out
  odd pages under the baseline's memoized per-candidate verdicts
  (identical oracle draws for every candidate, train and eval alike);
  AP = `rank_score` ordering over the 576-candidate held-out pool
  (7 useful; random AP 0.0122).

**Enumeration honesty.**  `Bias` is fixed first where present; on meta
`state ≺ action` is required (`_act_feats` reads `row["st:*"]` — 8 of 31
subsets are infeasible: `action` without `state` raises `KeyError`).
Orderings are capped at 3 per subset and **deduplicated by emitted-row
probe**: meta enumerated 45 orderings → 22 deduped → 23 distinct
schemas; open 63 → 32 deduped → 31.  The ordering axis is *vacuous in
this vocabulary*: only `ActionFeats` reads `row`, only `st:*` keys, so
all legal orderings of a subset emit identical rows — the real axis is
*which parts*, and the leaderboard does not pretend otherwise.

## Meta arena — incumbent ties for first (rank 1/23)

| rank | schema | reward/ep | greedy | evals/ep | feats |
|---|---|---|---|---|---|
| 1 | `bias>state>hist>onehot>action` **(incumbent)** | −0.0971 | −0.050 | 2.88 | 46 |
| 1 | `bias>state>onehot>action` | −0.0971 | −0.050 | 2.88 | 41 |
| 1 | `state>hist>onehot>action` | −0.0971 | −0.050 | 2.88 | 45 |
| 1 | `state>onehot>action` | −0.0971 | −0.050 | 2.88 | 40 |
| 5 | `bias>hist>onehot` (+9 more `onehot`-bearing) | −0.1189 | −0.050 | 3.00 | ≤22 |
| … | every schema without `onehot` | −0.5752 | −0.600 | 11.50 | — |
| — | `state>action`, `state>hist>action` (sampled floor, **greedy +0.0517** — the only positive decode) | −0.5752 | **+0.0517** | 11.50 | 36–41 |
| — | ANCHOR shipped-prior (baked, 30 ep) | −0.0563 | −0.050 | 1.13 | 46 |
| — | ANCHOR scripted | −0.1000 | — | 2.00 | — |
| — | ANCHOR random | −0.5657 | — | 11.31 | — |

**Part importance (leave-one-out vs incumbent / best-without):**
`onehot` is the load-bearing part — LOO **−0.4781**, and *no* schema
without it escapes the random floor (−0.575): the policy cannot even
pick `extract` without the move-kind logits.  `action` LOO −0.0218
(small).  `state` cannot be LOO-dropped (structural: `action` reads its
row) but `best_without` −0.1189 shows it adds little on its own.
`bias`, `hist`: LOO 0.0 — free to drop.

## Open game — incumbent rank 11/31 (search *does* beat it)

| rank | schema | reward/ep | greedy | use/ep | evals/ep | AP |
|---|---|---|---|---|---|---|
| 1 | `action` alone **(winner)** | **0.2827** | 0.2939 | 0.139 | 14.7 | 0.0277 |
| 1 | `bias>action`, `…>hist>action`, `state>action`, … (8 tie) | 0.2827 | 0.2939 | 0.139 | 14.7 | 0.0277 |
| 9 | `bias>onehot>action` family (7 tie) | 0.2301 | **0.3476** | 0.111 | 14.5 | **0.0701** |
| 11 | `bias>state>hist>onehot>action` **(incumbent)** | 0.2301 | **0.3476** | 0.111 | 14.5 | **0.0701** |
| 25–31 | schemas without `action` | 0.0989–0.1024 | 0.2880 | ≤0.083 | ~9 | 0.0123 |
| — | ANCHOR shipped-prior (baked, 5 ep) | 0.3271 | 0.3576 | 0.167 | 13.8 | 0.1332 |
| — | ANCHOR heuristic drain | 0.3595 | — | 0.167 | 16.0 | 0.0450 |
| — | ANCHOR random | 0.1339 | — | 0.083 | 8.7 | 0.0123 |

Three clean tiers.  Without `action` (the `a:*`/`a:form:*` candidate
shape block) a policy is blind to *which* candidate it evaluates —
reward ~0.10, AP at random level.  Adding `action` jumps to 0.283.
Adding `onehot` *costs* sampled reward (0.283→0.230 — `op:stop` weight
learning slows the drain at 3 epochs) but buys the best greedy decode
(0.3476) and nearly 3× the AP (0.0701 vs 0.0277): the one-hot separates
"evaluate this shape" from "stop", which is what the ranker needs.

**Part importance:** `action` LOO **−0.1277** (the only part that can
see the candidate — drop it and AP collapses to random).  `onehot` LOO
**+0.0526** on sampled reward but −0.1175 on greedy (0.3476→0.2939) and
−0.042 on AP — a tradeoff, not a free drop.  `bias`/`state`/`hist`: LOO
0.0 — all dead weight on this board (state broadcasts cancel in the
softmax; the open `x:*` interaction block does not exist, unlike meta).

## Verdict

* **Does representation search beat weight search?** Qualified yes on
  open — `action` alone beats the incumbent's sampled reward at equal
  budget (0.283 vs 0.230); but the incumbent family wins greedy and AP,
  so search found a *simpler schema with a different reward/ranking
  tradeoff*, not a dominator.  On meta the incumbent is tied-optimal —
  the four winning schemas differ only in the provably-inert `bias`/
  `hist` parts.
* **Which parts matter:** meta lives on `onehot` (its whole policy is
  per-op logits; LOO −0.4781 → random floor).  Open lives on `action`
  (the only candidate-aware part; LOO −0.1277, AP→random without it),
  with `onehot` buying ranking quality.  `bias`, `hist`, `state` (open)
  never changed a score — dead weight at this budget.
* **Better than shipped?** The shipped schema is *not* strictly optimal
  — equivalent-or-better smaller schemas exist on both boards — but the
  baked prior's absolute numbers (open 0.327/AP 0.133) still exceed
  every reduced-budget candidate's, as expected at 3 vs 5 epochs.  The
  exercise's finding is that schema choice matters **more than** the
  parts suggest: one part carries each board, the rest are inertia.
* **Orderings:** a null axis — dedup proved every legal ordering emits
  identical rows; only `state ≺ action` is a hard (meta) constraint.

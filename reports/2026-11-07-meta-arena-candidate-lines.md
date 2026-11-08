# T-009 lever 4 — candidate-side oracle lines (2026-11-07)

> **Corpus caveat (added later):** the held-out numbers below were
> measured on the *five-kind* `gen_cases` stream.  `gen_cases` later
> grew to 10 kinds (see `2026-11-07-meta-arena-corpus-coverage.md`),
> so the same offsets now deal different boards — the figures are a
> historical record, not comparable to post-extension bakes.

Measurement report for the myco ledger's last open T-009 lever:
whether the meta-arena oracle's *candidate set* (not the policy)
bounds held-out reward.  Full write-up in
`project/retros/meta-arena-player.md` ("Lever 4"); this file is the
numbers table and the reproduction recipe.

## Change

`mlopt_discovery.meta_player._candidate_lines` (plus helpers
`_macro_line`, `_claim_candidates`, `_line_ops`, `_op_count`) now
enumerates, beyond the previous
{extract, saturate→extract, claim→extract, claim→saturate→extract,
all-claims→extract} set:

- every **offered macro** (`macro_moves.offered(state)` — the
  viable `lawdata.MACROS` compressions `legal_actions` itself
  enumerates), replayed as `[macro]` or `[macro, extract]`;
- `claim → saturate(b) → extract` per bounded arm;
- `saturate → claim → extract` (closure-first, both arms);
- bounded saturates after all-claims, saturate-first multi-claim,
  reversed claim order.

Only offered macros are candidates — an oracle pick must stay
imitable (`imitate` matches on the legal-set action key).  The
ordering/budget variants are spelled-legal moves the oracle had
never tried; they are enumerated for completeness and measured
below as never argmax.

`_plan_rank` / `evaluate` / the `spelled-pick` diagnostic flatten
macro seqs via `_line_ops`, so `macro[claim,extract]` still reads
as the tier-0 fused line and macro plays count their inner claims.

## Bake recipe (unchanged)

```
.venv/bin/python -m mlopt_discovery.meta_player \
    --seed <S> --train-cases 400 --eval-cases 40 --budget 24 \
    --temperature 1.2 --epsilon 0.25 \
    --imitate-epochs 3 --epochs 5 --json run.json
```

Held-out offsets: `S=0` → `gen_cases(7919, 40)`;
`S=23418` → `gen_cases(31337, 40)`.  The run warm-starts from the
shipped `meta_prior.META_ARENA_PLAYER_WEIGHTS` (the spec default),
same as any re-run of the recipe.

## Results — learned-greedy mean reward (stochastic arm in parens)

| arm | offset 7919 | offset 31337 |
|---|---:|---:|
| recorded prior bake (2026-11-07) | +0.17 (+0.17) | +0.22 (+0.23) |
| control: old candidates, same prior re-bake | +0.17 (+0.17) | +0.23 (+0.23) |
| **new candidates** | **+0.19 (+0.18)** | **+0.25 (+0.23)** |

Control runs isolate the candidate effect from the warm start —
the prior is a fixed point of its own recipe, so the delta is the
candidates'.  Imitation cloned 1446/1446 (seed 0) and 1413/1413
(seed 23418) demonstrated moves; `op:macro` weight moved −1.2 →
+2.7.  Oracle plan tiers unchanged (250 claim-line / 150 decline /
0 spelled-pick on seed 0).

## Verdict: MOVED, modestly — residual is the move space

~+0.02 mean at both offsets, matching the mechanism: the
`claim_silu_extract` macro is viable+argmax on the silu /
silu_nested boards (~25–30% of the stream) and pays +0.05 over the
spelled `claim → extract` (one step charge saved; a terminal macro
also escapes the enode term).  On swiglu boards the argmax remains
spelled `claim(swiglu) → extract` (+0.566) because `lawdata.MACROS`
ships no `claim_swiglu_extract` / all-claims / saturate
compressions — a synthesized macro is not in the legal set, so it
can be neither demonstrated nor played.  Closing that last gap is
a `lawdata.MACROS` table extension (new offers for every player),
i.e. arena move-space work, not oracle-candidate work.

Spelled multi-declare lines, fire-restricted variants and
claim-order permutations were probed directly: all tie or lose to
the existing argmax (the spelled line pays +0.30 vs claim +0.40 /
macro +0.45 on silu; fire/handle-order variants tie the spelled
line exactly).  Ambient `fire(rule) → extract` candidates were not
enumerated: `saturate → extract` already covers their reach at ~0
enode cost on this corpus (saturate spends ≈0 enodes here), so a
per-rule fire arm would multiply oracle simulation cost ~10× for
no measured difference.

## Files

- `packages/mlopt-discovery/src/mlopt_discovery/meta_player.py` —
  candidate enumeration + macro-aware plan/eval accounting.
- `tests/test_discovery_meta_player.py` — oracle-line pins updated
  to the macro argmax; new coverage for `_candidate_lines`
  shapes, `_macro_line`, `_line_ops`.
- Bake JSONs (this directory, uncommitted):
  `t009_candidates_s0.json`, `t009_candidates_s31337.json`;
  controls `t009_control_s0.json`, `t009_control_s31337.json`.

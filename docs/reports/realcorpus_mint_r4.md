# Real-corpus mint, round 4 — the fixed harness, judged for real

**Question.** r3's recorded `0/89` was produced under two harness bugs
(f0ee0c1 fixed both: `serialize_rule`/`rule_from_record` premise
round-trip; `FunctorchReplayError` named `verify_unverifiable` decline).
A scratch re-judge showed `rmint:075 prim:synop_f0576b5150` and
`rmint:084 prim:synop_0b304215b3` admit under faithful semantics — but
never through `admit_prim_record` → `AUTO_TIER`. r4 asks: **does the
shipped driver path, unmodified, reproduce the admits — and can they
be persisted?**

**Answer: yes — 2 of 89 admitted, and both are now in the store's
`AUTO_TIER`.** The r3 candidate ledger was resumed verbatim (mint
skipped; all 89 `admit` verdicts stripped so every candidate re-judged
under the fixed harness). All 89 judged, **zero `rerr`, zero
`resume_skip`**, and the two synops the r3 re-judge predicted admit —
`synop_f0576b5150` and `synop_0b304215b3` — cleared the full
oracle → pay → sweep → cert gauntlet with premise-anchored certs.
Both classified `machine-original`. `prim_mint.admit_prim` →
`admit_prim_record` persisted both into `AUTO_TIER`; a fresh process
shows all four names (`prim:*` folds + `*.expand` premises) in
`AUTO_ADMITTED` and `MACHINE_SHIPPED`, and `minted.prim_fold`
materializes the pairs.

## Setup

Same driver and flags as r3; the input ledger is r3's JSON with the
`admit` verdicts stripped (candidates, law records, prim meta
identical — r3-era records carry no `premise_laws`, so the depth-1
`synop*.expand` premise is re-derived by `_resume_rule`'s
`_synth_expand_premise` fallback, exactly as designed).

```sh
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 PYTHONPATH=/tmp/whisper_deps \
.venv/bin/python -u bench/drivers/mint/real_model_mint.py \
    --models whisper,t5 --pay-arm \
    --prim-models t5:decoder,whisper:decoder \
    --max-candidates 128 --eval-purge 4 \
    --resume docs/reports/realcorpus_mint_r4.json \
    --out docs/reports/realcorpus_mint_r4.json
```

(JSON evidence: `docs/reports/realcorpus_mint_r4.json`; the
AUTO_TIER write: scratch `/tmp/r4_admit.py`, not committed.)

- **Attempt 1** — resume, 89 to judge: exports + ctx build, seated
  24 prim denotations, judged `rmint:000`–`rmint:074` (75 candidates),
  then **earlyoom SIGTERM** — the driver's SIGTERM handler flushed the
  checkpoint cleanly (`killed: "SIGTERM (earlyoom)"` in the JSON).
- **Attempt 2** — resume on the checkpointed file: 14 to judge
  (`rmint:075`–`rmint:088`), 14 prim denotations seated, all judged,
  classify + fire + pay-arm ran, clean finish (`wall_s: 127.0` for the
  judging pass; first attempt ≈25 min).  Total r4 wall ≈ 50 min.

## Verdicts (89/89 judged)

**2 admitted, 87 refused.** Stage-at-refusal:

- **63 refused at `oracle`** — 57 `xanti-view`, 4 `census-mixed-view`,
  1 `fusion-schema`, 1 `layout`.  Same thin-real-env signature as r3
  (`0eq/0ne/Nrerr`); `rmint:000 reshape_transpose` false again
  (`real: 0eq/33ne` vs synth eqs).
- **24 refused at `pay`** — 22 `prim-fold` + `rmint:001`
  (`fold:sdpa_fold_addmul_drop`, `verify_fail=1` on the
  whisper:decoder case) + `rmint:007` (`grammar:sub_to_add_dup`,
  `paid=0` — third round running).  Every refused prim fold now reads
  `verify_fail=1 … verify_unverifiable=1` — the F1 functorch/vmap
  decline **named**, still counted as a verify failure (verification
  not weakened), still a refusal.
- **2 admitted** — the r3 re-judge's exact pair, now through the real
  driver path:

| id | synop | oracle | pay | sweep | cert |
|---|---|---|---|---|---|
| `rmint:075` | `synop_f0576b5150` | 33eq/0ne real | fires=121 changed=1 paid=1 verify_fail=0; count=1.054x exec=0.947x flops=0.947x | clean | `rhs->lhs 1 steps (synop_f0576b5150.expand) — premise-anchored` |
| `rmint:084` | `synop_0b304215b3` | 49eq/0ne real + 15eq synth | fires=53 changed=3 paid=3 verify_fail=0; count=0.500x exec=0.500x | clean | `rhs->lhs 1 steps (synop_0b304215b3.expand) — premise-anchored` |

F2 is confirmed fixed: `rmint:084` — refused `not derivable
(fixed_point)` in r3 precisely because resume stripped premises —
now clears cert with the premise-anchored derivation.  Harness health:
**0 `rerr` on any prim fold** (denotation re-seat works), 0
`resume_skip`, `eval_guard` on the judging pass: 1 835 eval calls,
458 purges, `env_skipped` 0.

## AUTO_TIER persistence

The driver judges but does not persist — the durable-admit step is
`prim_mint.admit_prim` (the same call `overnight._promote_prim` makes
for its own admits; it does not re-verify — the gauntlet clearance +
`machine-original` novelty is the asserted gate).  `/tmp/r4_admit.py`
rebuilt each admitted rule via the driver's own `_resume_rule`
(premise re-attached), set the shipped fold name `prim:<synop>` (the
`admit_prim_record` invariant — a fresh `as_rule()` mints that name;
the `rmint:NNN` was only the ledger label), and called
`admit_prim(meta, rule)`:

- `prim:synop_f0576b5150` + `synop_f0576b5150.expand` → AUTO_TIER
- `prim:synop_0b304215b3` + `synop_0b304215b3.expand` → AUTO_TIER

Fresh-process verification: `provenance.AUTO_ADMITTED` and
`MACHINE_SHIPPED` now contain all four names (joining the prior
`synop_6ed0a07c7d` pair); `minted.prim_fold` materializes both folds
with `premises=(<synop>.expand,)`; `store.query` shows both at tier 4.

**Durability caveat (unchanged from r3):** the write landed in the
venv's installed `mlopt_core/laws/data/minted.db` — a `uv sync` or
reinstall reverts it.  A durable artifact needs the vocab emit
(`vocabdir.emit_prim`/`prims.yaml` record) plus regenerating the eqsat
`minted.db`; and the synops' torch denotations still live only in the
minting process's `_IR_TO_TORCH` (`emit_vocabulary=False`), so a
fresh process can materialize the fold but cannot eagerly eval the
`synop_*` RHS until the binding is re-seated.

## New harness issue found

**`--pay-arm` is broken by signature drift** — `_pay_arm` calls
`mlopt_orchestrator.optimize.search(mod, feeds, rules=…, …)` without
the now-required keyword-only `source=` argument.  Both admitted laws'
pay rows record `TypeError: search() missing 1 required keyword-only
argument: 'source'` after the 300 s wall.  The gauntlet's own pay leg
(`bench.referees`) is unaffected — this is only the post-admit
search→lower→time arm.  Not fixed (driver-only constraint); flagged
for the next round.

## Honest negatives / caveats

- The 22 pay-refused prim folds are **not** evidence against the
  folds — every one is `verify_unverifiable=1` on the
  `whisper:decoder` case (F1's structural blind spot, now named).  The
  r3 counterfactual (probe minus whisper:decoder) admitted all 24;
  r4 confirms the two admits reproduce under the *real* ctx because
  their changed cases never touch the unverifiable module.
- e-graph extraction is not bit-stable: `rmint:075` measured
  `changed=1` this run vs `changed=0` in r3's resume pass — marginal
  pay verdicts are noisy; the admit survived because `paid=1`.
- r4 judged r3's candidate set, not a fresh mint — prim-pool
  nondeterminism means a fresh r5 would sample a different pool.
- The two admits are real but thin: `rmint:075` pays on a single
  changed case (count=1.054x — *more* enodes, exec 0.947x);
  `rmint:084` halves count/exec on 3 changed cases.  Both are
  gauntlet-green machine-originals, not headline speedups.

## Reproduce / inspect

```sh
.venv/bin/python -m py_compile bench/drivers/mint/real_model_mint.py
.venv/bin/ruff check bench/drivers/mint/real_model_mint.py
python3 -m json.tool docs/reports/realcorpus_mint_r4.json | less
# AUTO_TIER check (fresh process):
.venv/bin/python -c "from mlopt_core.laws import provenance; \
print(sorted(provenance.AUTO_ADMITTED))"
```

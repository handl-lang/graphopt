# 2026-11-12 — law auto-admission (Stage-1 lever 2)

Self-driving Stage 1, lever 2: **verified+paying minted laws promote
into the shipped bank automatically**, with honest provenance —
removing the human hand-integration step for the prim family.

Lever 1 (`corpus_cov`, same day) made corpus generation mechanical;
this lever closes the other half-open loop: the overnight driver's
mint → serialize → gauntlet → novelty pipeline already produced
admitted, machine-original laws that then sat in the JSONL ledger
waiting for a human to port the record into
`mlopt_core.laws.minted` by hand.

## What landed

**Substrate — `handl-lang/eqsat`, `packages/mlopt-core`** (uncommitted):

* `laws/store.py` — `AUTO_TIER = 4` joins `TIERS`; same `records`
  table, same canonical-JSON payloads.  Tier 4 is the
  auto-admission class: same record shape as tiers 1-3, distinguished
  by *who* promoted it.
* `laws/minted.py` — `_AUTO_RECORDS` (live dict, seeded from the
  store at import), `auto_records()`, `synop_shipped()` (the live
  shipped set), and **`admit_prim_record(name, record, db=None)`**:
  decodes the record's `fold`/`expand` rows, enforces the prim
  invariants (`prim:<name>` fold name, bare `<name>(args)` RHS, an
  embedded premise whose LHS is that call), refuses shadowing —
  identical payload idempotent, different payload raises — and
  `store.emit`s into `AUTO_TIER`.  It deliberately does **not**
  re-run the gauntlet: verification stays the discovery side's
  gate.  On the shipped store the live tables update immediately;
  `PRIM_FOLDS`/`ALL_RULES`/`DEFAULT` pick persisted admissions up at
  the next import — the same path the hand-promoted tiers take.
* `laws/provenance.py` — `AUTO_ADMITTED` ⊂ `MACHINE_SHIPPED` ⊂
  `MACHINE_LAWS`; `law_provenance` reports `"machine"` for live
  admissions; `pay_status` reports `"unmeasured"` until the
  slow-pay ledger measures them (honest — the gauntlet measured
  once, the standing ledger has not).
* `tools/build_minted_store.py` — `_ATTRS` carries tier 4.
* `tests/test_admit_prim.py` — 13 tests: materialization, embedded
  expand, idempotence, shadow refusal, shipped-name refusal,
  record-shape validation, the live shipped path, provenance/pay
  status, and a torch-free extraction-fires check.

**Domain — `graphopt`** (uncommitted):

* `mlopt_discovery.prim_mint.admit_prim(prim, rule, db=None)` —
  builds the `prims.yaml` record and calls `admit_prim_record`;
  degrades loudly while the pinned `mlopt-core` predates the API.
  `core_shipped()` reads the live `synop_shipped()` set so vocab
  pricing sees persisted admissions.
* `bench/drivers/overnight.py --auto-admit N` — promote pass after
  classify: candidate must be `admitted` (every referee stage
  green — the pay leg IS the paying-site check), carry `prim`
  meta, classify `machine-original`, and have a green `pay` stage.
  Each attempt is an `("admit", name)` checkpoint cell carrying the
  pay detail + measured metrics; the pass is resumable and
  protocol-pinned.  Depth-N records rebuild the live `Rewrite`
  from the deterministic pool (`law_from_data` cannot carry
  premises; depth-1 is the only honest fallback).

## The demonstrated admission

`overnight.py --corpus zoo-prim --prim-deep --auto-admit 1` →
`docs/reports/results_overnight_prim_auto_admit.{md,json}`:

* **`prim:synop_6ed0a07c7d`** — the same mint the
  `results_overnight_prim_{deep,vocab,depth3}` runs surfaced.
* Gauntlet: oracle `12eq/0ne/0rerr` (12 acc / 12 envs), sweep
  `12eq/0ne/0rerr`, cert 1 anchored step via
  `synop_6ed0a07c7d.expand`.
* **Pay: `fires=30 changed=6 paid=6 verify_fail=0 cases=72`** —
  `count=0.775x exec=0.775x flops=0.918x launch=0.917x
  params=1.000x`.
* Novelty: `machine-original` (underivable vs the widest bank).
* Persisted to `minted.db` tier `AUTO_TIER`; fresh-process
  verification: `auto_records()` carries it, `synop_shipped()`
  57 → 58, `prim_fold` materializes fold+expand,
  `missing_hooks == ()`, `law_provenance == "machine"`,
  `AUTO_ADMITTED` names both, `pay_status == "unmeasured"`.
* Bank effect at next import: `PRIM_FOLDS`/`MINTED_RULES` +1,
  `ALL_RULES`/`DEFAULT` 143 → **144**, `ALL_RULES_WITH_LAYOUT`
  220 → 221, `CATEGORICAL_RULES` 108 → 109, module census 598 →
  600, axioms 119 → 120, FUSION preset 78 → 79.  Eq sat fast lane
  green (1650 passed) with all count pins updated.

## Honest scope

* **AUTO_TIER is provenance, not quarantine** — persisted records
  join `DEFAULT` like every prior promotion; that is the point of
  the lever.  The tier marks *who* promoted (machine, not human).
* **Prim-only.**  `admit_prim_record` enforces the strict prim
  shape; composites with transported guards / non-empty
  `missing_hooks` still need the human audit (`conv_bn_fold`
  precedent).  No mass admission: exactly one record shipped; the
  other 47 machine-original finds stay ledger-only behind the
  `--auto-admit N` cap.
* **Pin gap.**  graphopt's `uv.lock` pins an eqsat `main` that
  predates `admit_prim_record`/`synop_shipped`; `prim_mint`
  degrades loudly (RuntimeError / `SYNOP_SHIPPED` fallback) until
  the eqsat change lands and the pin bumps.  The demonstrated run
  used `PYTHONPATH=eqsat/packages/mlopt-core/src`.
* **Bank blast radius.**  The admitted fold joins `DEFAULT` for
  every caller at next import — same as every prior promotion;
  eqsat's suite is green, graphopt's bank-touching slice should
  re-run once the pin carries the new store.

Retro: `project/retros/law-auto-admission.md` (+ lever section in
`meta-arena-player.md`).  Ledger:
`bench/results/checkpoints/overnight_prim_auto1.jsonl`.

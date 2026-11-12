---
title: 'Retro: law auto-admission — the gauntlet-cleared mint ships itself'
type: retro
date: 2026-11-12
---

# Retro: law auto-admission — the gauntlet-cleared mint ships itself

Context: self-driving Stage 1, lever 2 (lever 1 — coverage-driven
corpus generation — landed the same day, `corpus_cov`).  The loop
this closes: the overnight driver already minted → serialized →
gauntlet-admitted → novelty-classified prim laws into its JSONL
ledger, but promotion into the *shipped* law bank was a manual
step — hand-port the record into `mlopt_core.laws.minted`'s tables,
update provenance sets, re-run the bank-touching suite.  This retro
records the mechanism that removes the human step for the prim
family, and the one admission that proves it end-to-end.

## The path

Two halves, split along the existing package boundary:

* **Substrate (`handl-lang/eqsat`, `mlopt-core`)** — the store and
  the shipped object:
  * `laws/store.py` — a fifth record tier, `AUTO_TIER = 4`
    (`TIERS` extended; `emit`/`load`/`tiers`/`query` unchanged —
    the tier is data, not schema).
  * `laws/minted.py` — `_AUTO_RECORDS` loaded from the store at
    import; `admit_prim_record(name, record, db=None)` validates
    the record's structural invariants (fold decodes, named
    `prim:<name>`, RHS is the bare `<name>(args)` call, an embedded
    `expand` premise's LHS is that call), refuses name shadowing —
    identical payload is idempotent, a *different* record under a
    shipped name raises — and writes through `store.emit` into
    `AUTO_TIER`.  **It does not re-run the gauntlet**: verification
    stays the discovery side's gate; a record that reaches it is
    asserted to carry its admission.  On the shipped store the
    live `_AUTO_RECORDS`/`_PRIM_MATERIALIZED` tables update
    immediately (`prim_fold`/`prim_folds`/`prim_premises`/
    `auto_records`/`synop_shipped` see it this process); the
    `PRIM_FOLDS` → `MINTED_RULES` → `DEFAULT`/`ALL_RULES` lists —
    built at import — pick a *persisted* admission up on the next
    import, exactly the path the hand-promoted tiers take.
  * `laws/provenance.py` — `AUTO_ADMITTED` (the tier-4 name set,
    `prim:<name>` + `<name>.expand`), folded into
    `MACHINE_SHIPPED`/`MACHINE_LAWS`; `law_provenance` reports
    `"machine"` for live admissions; `pay_status` honestly reports
    `"unmeasured"` until the slow-pay ledger measures them.
  * `tools/build_minted_store.py` — `_ATTRS` carries tier 4 so the
    store rebuild tooling round-trips the new tier.
* **Domain (`graphopt`)** — the gate and the call:
  * `mlopt_discovery.prim_mint.admit_prim(prim, rule, db=None)` —
    builds the `prims.yaml`-shaped record via
    `minted_prim_record` and hands it to `admit_prim_record`.
    `core_shipped()` now reads the *live* `synop_shipped()` set so
    vocabulary pricing sees persisted admissions (with a
    `SYNOP_SHIPPED` fallback while the pinned `mlopt-core` predates
    the API).
  * `bench/drivers/overnight.py --auto-admit N` — a promote pass
    after the classify loop: a candidate promotes only when its
    ledger cell is `admitted` (every `bench.referees` stage green —
    the **pay leg is the paying-site check**, `fires/changed/paid`
    must be nonzero) AND its classify verdict is
    `machine-original`.  Each attempt lands an `("admit", name)`
    checkpoint cell (pay detail + measured metrics ride it), so a
    resume replays the verdict instead of re-writing the store;
    `--auto-admit` is protocol-pinned like the rest of the prim
    window.  Depth-N records rebuild the live `Rewrite` from the
    deterministic pool (law records cannot carry premises — a
    depth-1 `law_from_data` fallback is the only honest one).

## The demonstrated admission

`overnight.py --corpus zoo-prim --prim-deep --auto-admit 1`
(`docs/reports/results_overnight_prim_auto_admit.{md,json}`,
ledger `bench/results/checkpoints/overnight_prim_auto1.jsonl`):

* `prim:synop_6ed0a07c7d` — the same mint the earlier
  `results_overnight_prim_{deep,vocab,depth3}` runs surfaced —
  cleared the full gauntlet: oracle `12eq/0ne/0rerr`, sweep
  `12eq/0ne/0rerr`, cert anchored through
  `synop_6ed0a07c7d.expand`, and **pay `fires=30 changed=6 paid=6
  verify_fail=0 cases=72`** with `count=0.775x exec=0.775x
  flops=0.918x launch=0.917x params=1.000x`.
* Novelty: `machine-original` (underivable against the widest
  shipped bank).
* Promoted into `minted.db` tier `AUTO_TIER`; verified in a fresh
  process: `auto_records()` carries the record, `synop_shipped()`
  grew 57 → 58, `prim_fold` materializes
  `prim:synop_6ed0a07c7d` + `synop_6ed0a07c7d.expand`,
  `missing_hooks == ()` on both, `law_provenance == "machine"`,
  `pay_status == "unmeasured"` (honest — not yet in the slow-pay
  ledger), `AUTO_ADMITTED` names both.
* Bank effect at next import: `PRIM_FOLDS` 57 → 58,
  `MINTED_RULES` 58 → 59, `ALL_RULES`/`DEFAULT` 143 → **144**,
  `ALL_RULES_WITH_LAYOUT` 220 → 221, `CATEGORICAL_RULES` 108 →
  109, module rewrite census 598 → 600, axioms 119 → 120, FUSION
  preset 78 → 79.  All eqsat count pins updated; the fast lane is
  green (1650 passed).

## Design calls, honestly

* **AUTO_TIER is provenance, not quarantine.**  Persisted records
  join `DEFAULT`/`ALL_RULES` at next import through the same
  `prim_folds → MINTED_RULES` path the hand-promoted tiers take —
  that IS "the shipped law bank".  The tier exists so the *record*
  carries who promoted it (`AUTO_ADMITTED` ⊂ `MACHINE_SHIPPED`),
  not to keep the law out of saturation.  A runtime admit
  materializes for explicit composition but does not retroactively
  join the import-time rule lists.
* **The gauntlet stays the only verification gate.**  Core asserts
  shape and refuses shadowing; it never re-judges.  The driver's
  promote pass re-checks the ledger's own evidence (admitted +
  pay stage green + machine-original) rather than trusting a flag.
* **Prim-only, by construction.**  `admit_prim_record` enforces
  the strict prim record shape (bare-call RHS, `prim:` naming,
  optional expand anchor).  Composite records with transported
  guards/`missing_hooks` are NOT auto-admissible yet — they still
  need the human `missing_hooks` audit `conv_bn_fold` got.
* **No mass admission.**  Exactly one record shipped — the
  demonstrated `synop_6ed0a07c7d`.  The other 47 machine-original
  finds from the same run remain ledger-only; `--auto-admit N` is
  an explicit per-run cap.

## What remains open

* `pay_status` stays `"unmeasured"` until the slow-pay ledger
  measures the new fold at bound sites — the gauntlet's pay leg
  measured it once; the standing ledger has not.
* Composite/minted-macro auto-admission (guarded records,
  `missing_hooks` non-empty) is out of scope — the honest audit
  step is still human.
* The pinned `mlopt-core` in graphopt's `uv.lock` predates
  `admit_prim_record`/`synop_shipped`; `prim_mint` degrades
  loudly (RuntimeError / `SYNOP_SHIPPED` fallback) until the eqsat
  change lands on `main` and the pin bumps.  Until then
  `--auto-admit` on the pinned dep refuses at the admit call —
  the checkpoint cell records the refusal.
* The admitted fold joins `DEFAULT` for every caller at next
  import — the same blast radius every prior promotion had; the
  bank-touching suite ran green in eqsat, and graphopt's own
  suite should re-run once the pin carries the new store.

Report: `docs/reports/results_overnight_prim_auto_admit.md`
(measurement) — mechanism writeup in this retro; eqsat-side
coverage in `tests/test_admit_prim.py` (13 tests).

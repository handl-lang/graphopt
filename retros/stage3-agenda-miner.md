---
title: 'Retro: Stage-3 first cut — the agenda miner'
type: retro
date: 2026-11-13
---

# Retro: Stage-3 first cut — self-proposed research agenda

Stage 3's first cut: the system scans its own measurement surfaces
and files ranked research proposals.  Proposals only — no
auto-execution; the trust boundary holds.

## What landed

`bench/drivers/agenda/agenda_miner.py` — a standalone driver
(`.venv/bin/python bench/drivers/agenda/agenda_miner.py`) that
mines eight signal classes and emits `docs/reports/agenda.jsonl`
(machine-readable) + `docs/reports/agenda.md` (ranked human
agenda with per-signal evidence and `myco todo add` ingest
commands; `--myco` ingests directly with `--no-commit`).

Signals: overnight-ledger pay declines that changed the graph
(`changed>0, paid=0`, ranked by exec-ratio closeness to the keep
bar), per-family oracle gaps (families dying at oracle with 0 eq),
vacuous-novelty claims, frontier rows with uncaptured algebraic
headroom, sweep rows whose claim-bearing runs straddle parity
(referee noise decides keep/drop), delivered+verified losers,
`pay_measured` executor confounds, and a live `corpus_cov`
coverage census (ops/attrs/rules/handlers).

## First real agenda (2026-11-13, 23 proposals)

Top of the ranking:

1. **pay_measured confound** (0.83) — 6/14 marginal rows compare
   different executors (synop→IRModule vs noprim→BatchedScanModule);
   the pay table measures routing, not the fold.
2. **frontier delivery gaps** — ChunkedSSD carries ~1.5–1.75 ms of
   measured ceiling with captured ≈ 0 / negative; DecoupledMLA,
   DeltaRuleBlock, GatedDeltaRule similar shape.
3. **ledger pay declines** — `grammar:sub_add_factor` and four
   `syn_*` composites changed the graph but missed the keep bar
   with exec ratios 0.91–1.00 — near misses, not structural losses.
4. **sweep straddle/losers** — 18/64 and 52/64 claim-bearing rows
   respectively: referee noise and a global keep margin that
   admits below-parity deliveries.
5. **coverage residual** — 102 ops / 38 attrs / 112 rules still
   unexercised after the Stage-1 coverage arm.

## What it taught

- The measurement surfaces were already rich enough that the miner
  needed zero new instrumentation — the honest first cut is a
  *reader*, not a *probe*.  `PROFITABLE_LOG`/`TRITON_REFEREE_LOG`
  stay in-memory; persisted report JSONs carry enough.
- Claim-less sweep rows must be excluded from keep/drop signals —
  82 straddles collapse to 18 once rows without shipped claims
  are dropped; the rest is timing noise on identical work.
- `myco todo add` currently fails repo-wide: `project/TODO.md:19`
  (the human-written T-010 mega-line, ~9.7k chars with non-ASCII)
  fails myco's "item line must be plain text" parse, so `myco
  check` is already red on the ledger.  The miner emits correct
  commands and `--myco` degrades honestly (rc + stderr), but
  ledger repair is a prerequisite for programmatic ingest.

## Not done (the trust boundary)

No proposal auto-executes.  Next cuts: dedupe frontier rows across
dtype regimes, ingest accepted proposals as TODO items once the
ledger parses, and close the loop only behind an explicit human
approval gate.

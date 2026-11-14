# Agenda loop — first iteration: proposals → tracked work items

The self-driving loop is `agenda → mint → verify → measure → ship →
re-mine`.  This document closes the **first** arrow honestly:
`agenda_miner.py` proposals become *tracked myco work items* — with
the human audit gate kept explicit.  Nothing here executes research;
`--execute` only files ledger entries a human then picks up (or
deliberately doesn't).

## Pipeline

```
measurement surfaces                ledger (project/)
  overnight ledgers ─┐                ▲
  frontier*.json     ├─ agenda_miner  │  scripts/agenda_ingest.py
  zoo_chain_sweep*   │   .py          │   (validate → dedup → rank
  pay_measured*.json ┘                │    → cap → myco todo/issue)
  corpus_cov census ─┘                │
        │                             │
        ▼                             │
 docs/reports/agenda.jsonl ───────────┘
 docs/reports/agenda.md          (ranked table + per-signal evidence)
 docs/reports/agenda_ingest_state.json   (written on --execute only)
```

### `bench/drivers/agenda/agenda_miner.py` (producer, unchanged contract)

Emits `docs/reports/agenda.{jsonl,md}` — one proposal per JSONL line:
`{id, kind="agenda-proposal", title, target, signal, evidence, lever,
headroom, confidence, score, rank}`.  `--myco` on the miner still
exists but is the blunt instrument (files *every* proposal, no dedup,
no cap, no validation); `agenda_ingest.py` is the governed path.

One miner fix landed with this iteration: `_slug` truncated at 48
chars, so titles differing only past the cut (e.g. the
`...pow_reshape_l_w_6@`/`_w_10@RMSNormPerHead` laws) minted **the same
proposal id** — two id collisions in the current `agenda.jsonl`.  The
slug now appends a 6-hex sha1 of the full title when truncated;
ingest dedup made the collision harmless in the meantime (the second
proposal is skipped, not double-filed).

### `scripts/agenda_ingest.py` (consumer — new)

Per repo convention (`scripts/` = tooling that regenerates/audits
committed artifacts), a standalone script with a pure, testable core:

1. **Load + validate** (`load_proposals`/`_validate`) — strict schema
   check per line: `kind`, non-empty `id`/`title`/`signal`/`lever`,
   single-line title, `score`/`headroom` ∈ [0,1], `confidence` ∈
   {high,medium,low}, `rank` positive int.  Malformed lines are
   reported on stderr and skipped — never silently ingested.
2. **Dedup** — three layers: (a) proposal ids already in the state
   file, (b) proposal-id substrings in *live* `myco todo list` +
   `myco issue list` titles, (c) duplicate ids within the agenda file
   itself.  Ledger titles are minted as `agenda <short-id>: <title>`
   so the id is greppable; both the full `agenda-…` id and the
   stripped short id are tried as match needles.
3. **Rank + cap** — proposals re-sorted by `(-score, id)` regardless
   of file order; `--cap N` (default 5) bounds the batch.
4. **File** — `myco todo add` by default (`--pri high` when
   `score ≥ 0.5` else `med`, labels `agenda` + `low-confidence` when
   applicable, `--no-commit --json`); `--kind issue` switches to
   `myco issue new --body -` with a markdown body carrying the lever,
   target, and evidence metrics.
5. **State** — `--execute` records `{proposal-id: {item, kind, rank,
   ts}}` in `docs/reports/agenda_ingest_state.json`.

### The audit gate

- **Default is dry-run**: prints the exact plan (rank, priority,
  proposal id, title) and the skip reasons; shells out to `myco` only
  for *read-only* `todo list`/`issue list` dedup queries.
- **`--execute`** is the single mutating flag.  Even then it only
  files ledger items — no research runs, no laws admitted, no
  measurement surfaces touched.
- `--since RANK` gives incremental re-runs: only proposals ranked
  after the last ingested rank are candidates (state dedup still
  applies on top).

## Usage

```sh
# plan only (default) — change nothing
.venv/bin/python scripts/agenda_ingest.py                 # cap 5
.venv/bin/python scripts/agenda_ingest.py --cap 23        # whole agenda
.venv/bin/python scripts/agenda_ingest.py --kind issue --cap 3
.venv/bin/python scripts/agenda_ingest.py --since 12      # incremental

# human-approved execute — files todos/issues in project/
.venv/bin/python scripts/agenda_ingest.py --execute --cap 5
```

Tests: `tests/test_agenda_ingest.py` (20 tests — validation,
dedup/state/since/cap planning, argv/title shapes, fake-runner
execute path, dry-run-vs-execute `main()`).  Run:
`scripts/pytest-capped.sh .venv/bin/python -m pytest tests/test_agenda_ingest.py -q`.

## First ingest dry-run (2026-11-15, real `agenda.jsonl`)

`docs/reports/agenda.jsonl` holds 23 valid proposals (0 invalid).
Dry-run against the live ledger (`project/` has 4 todos, 0 issues;
none carry agenda proposal ids except the hand-filed T-034, whose
title predates this pipeline's id convention):

```
agenda_ingest [DRY-RUN] agenda=docs/reports/agenda.jsonl valid=23 file=5 skipped=0
  1. [high] rank=1  agenda-paymeasured_confound-6-14-pay-measured-rows-are-executor-confounded-t
     6/14 pay_measured rows are executor-confounded — the marginal measures routing, not the fold
  2. [high] rank=2  agenda-frontier_delivery_gap-uncaptured-headroom-on-zoo-chunkedssd-1543-317us
     Uncaptured headroom on zoo:ChunkedSSD: 1543.317us ceiling, captured=0.0584, vs_compile=None
  3. [high] rank=3  agenda-frontier_delivery_gap-uncaptured-headroom-on-zoo-chunkedssd-1751-576us
     Uncaptured headroom on zoo:ChunkedSSD: 1751.576us ceiling, captured=-0.0615, vs_compile=0.286
  4. [high] rank=4  agenda-frontier_delivery_gap-uncaptured-headroom-on-zoo-decoupledmla-140-084u
     Uncaptured headroom on zoo:DecoupledMLA: 140.084us ceiling, captured=-0.2396, vs_compile=0.6241
  5. [high] rank=5  agenda-frontier_delivery_gap-uncaptured-headroom-on-zoo-deltaruleblock-163-41
     Uncaptured headroom on zoo:DeltaRuleBlock: 163.416us ceiling, captured=0.0055, vs_compile=0.7611
dry-run only — re-run with --execute to file these in project/ (human approval gate).
```

At `--cap 23` the plan files 21 unique proposals and skips 2 as
in-file id duplicates (the slug-truncation collisions described
above — the rank 8/9 and 10/11 pairs share ids; the miner fix makes
this moot on the next `agenda_miner.py` run).

**Nothing was executed** — `docs/reports/agenda_ingest_state.json`
does not exist yet; the first `--execute` is a human decision.
Suggested first batch: `--execute --cap 5` files the five `[high]`
items above; `--kind issue --cap 3` is the better fit for the
measurement-integrity proposals (confound/straddle) whose evidence
bodies are worth keeping attached.

## Loop status

| arrow | status |
|---|---|
| agenda → tracked work items | **built** — `scripts/agenda_ingest.py`, dry-run verified, awaiting human `--execute` |
| tracked items → minted/verified/measured work | open — the ledger items are ordinary todos/issues; the existing overnight/mint drivers pick them up by hand |
| ship → re-mine | open — next `agenda_miner.py` run (now with unique ids) refreshes the agenda; `--since` ingests only the delta |

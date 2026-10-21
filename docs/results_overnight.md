# Overnight discovery — unattended episodes → mint → admit → classify

**Date:** _stamped by the run_ · **Script:**
`bench/drivers/overnight.py` · **Seed:** 0 · **Runner:**

```bash
.venv/bin/python bench/drivers/overnight.py \
    --episodes <N> --budget <seconds> --seed 0 \
    --player <name> --referee <name> \
    --resume --out docs/results_overnight.md
```

> Skeleton — committed ahead of the driver so its emitted document has
> a home.  The run stamps the date/seed, fills the arms table, and
> writes the verdict; `--dry` exercises the loop without emitting here.

## Motivation

The discovery drivers to date are single-shot probes — run, report,
done.  The overnight driver is the continuous form: play episodes under
a registered player/referee pair, mint candidate laws from what the
episodes surface, run each candidate through the `verify_law`
gauntlet, classify survivors against the prior-art bank for novelty,
and keep going until the episode count or the wall-clock budget is
spent.  The JSONL ledger makes the run resumable (`--resume`) — a
partial night is not lost work.

## Protocol

Per iteration:

1. **episodes** — `--episodes` games under `--player` / `--referee`,
   seeded by `--seed`.
2. **mint** — candidate laws synthesized from the episode trace.
3. **admit** — each candidate runs the `verify_law` gauntlet.
4. **classify** — admitted candidates are scored for novelty against
   the prior-art bank.
5. **ledger** — every step appends to a resumable JSONL ledger;
   `--budget` caps wall-clock, `--out` names this document.

## Results

| arm | episodes | minted | admitted | novel | notes |
|---|---|---|---|---|---|
| _run fills this_ | | | | | |

## Verdict

_Written by the run._

## Suite status

Deliberately **not** a `bench.suites` adapter.  The suite harness wants
`run_bench(args) -> Report` over `Case`/`Variant` cells off an
importable package — the `discovery` suite precedent wraps
`bench.discovery`, which *is* a package.  `bench/drivers/` is not one
(drivers are standalone scripts with script-local sibling imports),
the driver's output is its own ledger plus this document — not
`baselines/` rows — and a budget-bounded resumable loop has no
`--quick` / `run-all` reading.  It stays where the other research
drivers live: standalone under `bench/drivers/`.

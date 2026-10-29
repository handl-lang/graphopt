---
title: '`project/` — graphopt'
type: readme
date: 2026-10-28
---

# `project/` — graphopt

The rule is ADR-0162 (host repo); this file is the map. Core entries mean what
that ADR says; the two local entries below are this repo's declared additions.

| entry | answers | lifetime |
|---|---|---|
| `adrs/` | *why* we decided, and what we rejected | append-only |
| `notes/` | *what we found* | promoted or deleted |
| `reports/` | *what we measured* | dated (frontmatter), immutable |
| `retros/` | *what went wrong* in an experiment | dated (frontmatter), immutable |
| `tasks/`, `TODO.md` | *what next* | consumed |

Every file opens with ADR-0163 frontmatter (`title`, `type`, `date`): the date
lives there, never in the filename.

## Local entries

- `plans/` — per-workstream plans (`Plan NNNN — …`) recording what was attempted and whether it landed; superseded by `adrs/` once a plan becomes a decision.
- `GOALS.md` — the standing goal ledger: every goal carries a pass/fail criterion and the artifact that proves it.

Published documentation (explanations, tutorials, how-to, reference) is not
here — it lives in the org `docs` repo.

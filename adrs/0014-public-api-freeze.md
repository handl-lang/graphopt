---
title: public API freeze (plan 0032 phase 1)
type: adr
date: 2026-10-27
status: accepted
status_note: accepted, phase-1 deliverable. `project/` is gitignored —
---

# ADR 0014 — public API freeze (plan 0032 phase 1)

Status: accepted, phase-1 deliverable.  `project/` is gitignored —
the tracked manifest is `docs/public_api.md`; this note records the
decision.

## Decision

Freeze the cross-repo contract **before** any renames (phase 2) or
extraction (phase 4): the names a downstream domain may bind are
enumerated in `docs/public_api.md`, mechanically pinned by
`tests/test_public_api.py`, and versioned by
`mlopt_core.API_VERSION = "0.1.0"` (MINOR = additive surface change,
MAJOR = a frozen name's signature/meaning change; `0.x` until each
repo versions its own slice post-extraction).

Three classes of name:

* **frozen** — the manifest tables (substrate → `eqsat`, game →
  `gamekit`, domain → `graphopt`);
* **public, unfrozen** — other `__all__` members (public today, not
  promised across the split);
* **internal** — everything else; `_`-prefixed compat re-exports stay
  internal regardless of reachability.

The referee seam: the **value spec is gamekit's**
(`mlopt_discovery.scoring` — `ScoreGate{axis, threshold, direction}`,
`ScoringSpec{gates, weights, references}`, `score(spec, measured)`);
the **measurement is the gauntlet's** (a plain `{axis: float}`
mapping — the substrate never imports the spec, the spec never
imports `evidence`).  Phase-2 lane B lands the referee registry and
matrix runner as consumers.

The `gamekit ⊥ eqsat` edge is enforced now, not later: the
"game kernel is substrate-free" import-linter contract forbids
`mlopt_discovery.engine` / `.scoring` / `.players._base` /
`.players.registry` from importing any substrate package (the player
*folders* are exempt — lazy `make` binding is the registry contract).

## Boundary findings (for phase 2)

* `engine.imitate_policy` — a documented driver never added to
  `__all__`; deliberately **unfrozen**.  Promote it or mark it
  internal when the gamekit lane lands.
* `Trajectory` was returned by `run_episode` but missing from
  `__all__` — added (it is part of the de-facto contract).
* `mlopt_core.laws.provenance` was importable only as a submodule;
  now re-exported at `mlopt_core.laws` (in `__all__`) so the frozen
  registry has a canonical home.
* Registry *contents* (`DOMAINS`, `ALL_RULES`, `MACHINE_SHIPPED`
  rows, `lawdata` tables) are data — the names are frozen, the rows
  are each destination repo's business.
* `mlopt_native` is bound only through the `Engine` port — the
  optional maturin build must not become a frozen import path.

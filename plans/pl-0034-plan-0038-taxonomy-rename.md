+++
id = "PL-034"
title = "Plan 0038 — taxonomy rename: repos and packages"
status = "deferred"
priority = "med"
+++

---
title: 'Plan 0038 — taxonomy rename: repos and packages'
type: plan
date: 2026-11-19
---

# Plan 0038 — taxonomy rename: repos and packages

Origin: user direction (2026-11 session) — repo/package names should
communicate role (intuitiveness/comprehension), not merely be
cosmetic. Decision recorded: **full taxonomy rename — repos AND
packages**, executed only after the current fix queue drains and the
name mapping is signed off.

## Org taxonomy (documented convention)

| family | role | members today |
|---|---|---|
| `compile-*` | compilers/transpilers | compile-handl, compile-triton, … |
| `engine-*` | interpreter/rewrite engines | engine-parse, engine-exec-*, engine-typecheck* |
| `ext-*` | language plugins | ext-sdkgen, ext-lint, … |
| `kit-*` | frameworks/toolkits | kit-core, kit-kernel-ir, kit-ffi, gamekit* |
| `lib-*` | libraries (mostly .hnl) | lib-tensor, lib-math, lib-utils, … |
| `host*` | platform host/runtime | host, host-tensor, host-lsp, host-cli |
| `sdkgen-*` | code generators | sdkgen-c, sdkgen-python, … |
| `*-opt` | optimizer products/domains | graphopt, kernelopt (+torchopt, +kit-discovery pending) |
| bare | substrate/products | eqsat, discovery, assets, docs |

*gamekit predates strict prefixing; treated as `kit-*` family.

## Proposed mapping (NEEDS SIGN-OFF before execution)

### Repos

| current | proposed | family |
|---|---|---|
| `eqsat` | `engine-eqsat` | engine-* |
| `gamekit` | `kit-game` | kit-* |
| `graphopt` | `opt-graph` | *opt |
| `kernelopt` | `opt-kernel` | *opt |
| torch-facing packages → new repo | `opt-torch` (backend plugins: torch+carriers+cuda+onnx) | *opt |
| discovery core → new repo | `kit-discovery` | kit-* |
| `host-tensor`, `lib-tensor`, all `lib-*`/`host-*`/`compile-*`/`engine-*`/`ext-*`/`kit-*`/`sdkgen-*` | unchanged — already conformant | — |

### Packages (inside their repos)

| current | proposed | notes |
|---|---|---|
| `mlopt-core` | `eqsat` (or `eqsat.core`) | substrate engine |
| `mlopt-orchestrator` | `eqsat.orchestrator` | pipeline |
| `mlopt-native` | `eqsat.native` | Rust engine |
| `gamekit`, `benchkit` | unchanged | already conformant |
| `mlopt-torch` | `torchopt` | torch plugin |
| `mlopt-carriers` | `torchopt.carriers` or `carriers` | moves with torchopt |
| `mlopt-cuda` | `torchopt.cuda` or `cudart` | delivery layer |
| `mlopt-onnx` | `torchopt.onnx` or own `onnxopt` | backend plugin — decide repo scope |
| `mlopt-discovery` | `discokey`? `kit-discovery`? | split: generic → kit-discovery, torch-shaped stays |

## Scale / breakage inventory (measured)

- ~1,000 files reference `mlopt_*` (tests 464, packages 319, bench/scripts 216)
- `docs/public_api.md` is the frozen contract — every name in it changes
- `uv.lock` git+ssh subdir pins in graphopt, kernelopt, and cross-repo
- import-linter contracts, `ty.extra-paths`, coverage `source`/`omit`,
  `scripts/gates.sh` walls, AGENTS.md inventories, myco references,
  doc links — all carry names
- GitHub repo renames: clones redirect, but pins/locks/CI must relock
- Package renames break every `import mlopt_*` — compat shims
  (`mlopt_core = eqsat.core` style) are mandatory during transition

## Phases (execute only after sign-off + queue drains)

**Phase 0 — name sign-off.** User confirms/replaces every name in the
mapping tables. No code moves.

**Phase 1 — repo renames (mechanical).** GitHub rename + local remote
update + `uv.lock` relock sweep across all consumer repos + README/
docs link pass. Packages unchanged — zero import breakage. This phase
alone delivers the intuitiveness win for repo browsing.

**Phase 2 — package renames with shims.** New dist names land;
`mlopt_*` becomes a compat shim layer (one `__init__` per old name
re-exporting). Import-linter/coverage/ty configs updated. public_api.md
versioned bump documenting the rename.

**Phase 3 — shim removal (later, deliberate).** After downstream repos
migrate, delete compat shims in a breaking release.

## Hard gates

- Phase 1 must not break `uv sync` anywhere — relock verified per repo.
- Phase 2 shims must keep every existing import working (full suite).
- No phase starts until the previous one's verification is green.
- kernelopt/graphopt/eqsat/gamekit are public — renames are announced
  in commit messages + README badges, not silent.

## Dependencies / adjacency

- Plans 0034 (discoverykit) and the torchopt extraction SHOULD land
  first — they create the new repos this plan names; renaming
  graphopt's contents into `opt-torch`/`kit-discovery` simultaneously
  with a graphopt→`opt-graph` rename would double the churn.
- Order recommendation: 0034 extraction → torchopt extraction →
  THEN this rename (new repos get final names at creation; only the
  four originals need GitHub renames).

---
title: 'Plan 0031 — measurement infrastructure: discovery benchmark, players-as-modules, baseline registry'
type: plan
date: 2026-10-20
---

# Plan 0031 — measurement infrastructure: discovery benchmark, players-as-modules, baseline registry

Origin: Lotufo review (2026-10-19). Two questions exposed the gap the
suite must close — "how do you know the cost?" and "how do you know it
found something *better*?" — and the honest answer needs a machine
behind it, not a paragraph.

**Cost has three trust levels, and the suite exercises all three:**
extraction ranks on `dag_cost` (analytical: FLOPs + param bytes);
admission measures pay on real seeded fixtures (wall / peak memory);
delivery verifies the lowered module numerically against the source
(fp64). "Found a better algorithm" = certified-equal output at lower
measured cost. The scoreboard is analytical, the admission is
measured, the output is verified.

## Workstream 1 — the discovery benchmark

A versioned corpus of *named* optimizations sourced from literature and
real compilers — the suite the pitch implies exists.

```
bench/discovery/
  cases/<name>.toml     # spelled fixture + source citation + category + baselines
  refs/<name>/          # known-good impls (RetNet chunked, opt_einsum, …)
```

Each case: spelled "before" module · citation · category
(`memory | speed | structure | precision`) · registered baselines
(eager, torch.compile, known impl). Runner
`python -m mlopt_discovery.benchmark` emits per case:

- found? certified? (derivability + gauntlet outcome)
- cost vs each baseline (symbolic and/or measured)
- **beat-best-registered?** — a certified form below the registered
  floor on a category = a real find, not a rediscovery. This is the
  novelty detector.

Two lanes: `--fast` (symbolic only, seconds — PR gate) and `--full`
(measured pay — nightly). Emits `docs/scorecard.md` + JSON side-file.
First cases: conv_bn_fold, RetNet chunkwise, FlashAttention combine,
Blelloch scan, mul_unsq_nat, weight-chain fold, swiglu_mlp.

## Workstream 2 — players as modules

```
packages/mlopt-discovery/players/
  _base.py                  # protocol: feature schema + move enumerator + policy
  linear/  neural/  distilled/  playbook/  random/
    policy.py  train.py  <artifact>  README.md (measured record)
```

Registry maps name → spec; `--player <name>` on the play CLI. Adding a
player = adding a folder. Comparison runner
`python -m mlopt_discovery.compare --players … --boards …` replays the
memoized-verdict protocol from `lab/policy_baseline.py` and emits
`docs/leaderboard_players.md` — versioned rows (date, sha, seed) with
metrics *and* a notes column (drain artifacts, observability walls —
the interesting comparisons have been qualitative).

## Workstream 3 — baselines, same shape

`baselines/{name}/` mirroring players: opt_einsum, torch.compile, FLA,
hand-written carriers — adapter + measured record each. The
scorecard/leaderboard then read opponents from one registry, so the
"limits" tables in results docs are *generated*, not hand-copied.

## Order

1 → 2 → 3, independent landings, nothing breaks between. Benchmark
first because it is the referee for every future claim.

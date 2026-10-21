# Evaluation — the fourth dimension

<!-- provenance: graphopt:docs/evaluation.md@eea02f6 — summarized; the full account lives upstream -->

mlopt splits into four independent dimensions; this page is the
**evaluation** one — how a program is described, ranked, and measured
*without letting any of that change what is equivalent*.

```text
SEMANTICS ──▶ SEARCH ──▶ EVALUATION ──▶ EXECUTION
```

## The pieces

- **Static features** (`mlopt_core.features`) — `ProgramFeatures`
  describes a program without running it: FLOPs, bytes read/written,
  temporary bytes, depth, operations, reuse (arithmetic intensity),
  parallelism. Torch-free. Features describe a *program*; a
  `TargetProfile` describes a *measured target*.
- **Calibration** (`scripts/calibrate_profile.py`) — micro-benchmarks
  the machine it runs on (peak FLOPS, bandwidth, launch overheads, a
  per-op-class kernel table) and writes a loadable `TargetProfile`
  JSON, so `delivered_cost_for(profile, x=…)` prices extraction by
  *this machine's* numbers, not built-in constants.
- **The frontier** (`mlopt_core.pareto`) — keep the landscape, don't
  scalarize early: `pareto_frontier` keeps non-dominated candidates,
  `best` is the scalar view over them.
- **Policies** (`mlopt_core.policies`, `mlopt_torch.learned_policy`) —
  a `Policy` only *orders* the legal moves, so it can never change
  what is certified. The learned arm scores `(program features ⊕
  structural rule vector)` — a rule the model has never seen is scored
  by its shape, not a vocabulary index.
- **Performance prediction** (`mlopt_core.perf_model`) — a roofline
  `max(compute, memory) + launch` that *ranks*; measured to change the
  extracted program in 0/7 families, so it is a post-hoc ranking tool,
  not a target chooser.
- **Failure taxonomy** (`mlopt_core.failures`) — `OOM` / `TIMEOUT` /
  `KERNEL` / `NONFINITE` / `UNAVAILABLE` give measurement failures a
  vocabulary instead of an ambiguous crash.

## Screening — evaluation before search

Evaluation also decides where search is worth running at all. The
exposure screen scores each corpus workload by the FLOPs-weighted
share of ops any law's left-hand side can name; the distribution is
**bimodal** (most workloads expose everything or nothing), and
exposure predicts *reachability*, not *payoff*. The measured lesson:
run discovery where structure is exposed, and let the measured referee
— not the search budget — decide what ships.

## The rule

A profiler **observes**, a policy **orders**, a model **ranks** — none
of them decides equivalence. Feasibility (`supported_ops`, hard) and
performance (soft) stay distinct, which is what lets
[the three costs](./three-costs) compose honestly:
each level exists because the cheaper one can lie.

The full account is
[`docs/evaluation.md`](https://github.com/handl-lang/graphopt/blob/main/docs/evaluation.md)
upstream.

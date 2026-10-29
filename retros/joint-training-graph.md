---
title: 'Retro — the joint board: forward + backward on one e-graph'
type: retro
date: 2026-10-14
---

# Retro — the joint board: forward + backward on one e-graph

Plan 0021's meta-arena gave the board a `declare`/`handle` action
space; `training.py`'s reverse handler made gradient programs
ordinary terms.  `joint_probe` asks the natural follow-up: the
autograd pipeline is *two* programs (forward graph, backward graph)
that can never share work across the boundary — does one e-graph
over forward + every gradient root do better?

Reproduce:

    .venv/bin/python -m catopt_discovery.training --joint

## The measurement

Per `_demo_forwards` case: `backward` derives every leaf's
gradient term under a shaped all-ones cotangent.  The *separate*
arm saturates the forward in one `EGraph` and the whole gradient
program in a second, billing the two extracted programs
independently — the autograd comparison, where cross-program
sharing is impossible.  The *joint* arm adds every root to one
e-graph and saturates once; `dag_cost` dedups shared subtrees, so
`joint < separate` is measured cross-boundary sharing.

`count_cost`, `DEFAULT` rules, max_nodes=20k:

| forward    | separate | joint | shared e-classes | enodes sep/joint |
|---|---:|---:|---:|---|
| silu_fwd   | 10.0 | **10.0** | 2 | 17/15 |
| softmax_fwd| 13.0 | **13.0** | 3 | 21/18 |
| linear_fwd | 13.0 | **10.0** | 6 | 27/21 |

`linear_fwd` wins 23%: tanh's VJP spec spells `1 - tanh(a0)^2`,
so the *entire forward term* is a subterm of all three gradient
roots — in one e-graph it is one e-class, extracted once, billed
once.  In two pipelines it is paid twice (once as the forward,
once inside each gradient program).  Same for `matmul(x,w)`
shared between `dx = g·W^T`'s siblings.

The enodes column is the second win, quieter but unconditional:
one saturation over the union costs fewer enodes than two
saturations (17→15, 21→18, 27→21) because shared subterms have a
shared rewrite frontier.

## The honest wrinkle — fusion hides sharing

silu/softmax are *flat* despite 2–3 shared e-classes.  Cause:
`silu_fold`/`softmax_fold` fuse the forward into `silu(x)` /
`softmax(x)` — locally optimal (−1 op) — but the fused member no
longer *contains* `sigmoid(x)`/`exp(x)` as a visible subterm, so
the backward's spelled reuse can't dedup against it.  Greedy
per-class extraction picks the fused member because it prices
each class in isolation.

Without the folds (`rules=DEFAULT - {silu_fold, softmax_fold}`)
the joint arm wins both: silu 11→10, softmax 15→13.  So the
sharing is real — but see below: forcing the spelled member back
only ties, so the fold is a *wash*, not a failure.

## The named lever — measured: a wash

The codebase already has the mechanism: `extract_best`'s
`overrides` / `extract_paired` — built for the fused-GEMM-sharing
case where k members each selecting `split_i(fused)` must
coordinate to bill `fused` once.  The joint board needs the dual:
*unfold* the forward's fused member to expose a subterm the
backward pays for anyway.

**Measured, and it ties.**  Forcing the spelled `mul`/`div` member
of the forward root (`overrides={fwd_class: spelled_enode}`) leaves
silu at 10 and softmax at 13 — identical to greedy.  The math is
exact under `count_cost`: fused `silu(x)` saves precisely the one
op that sharing `sigmoid(x)` would have saved.  For *one-use*
sharing, fusion ≡ sharing and coordination buys nothing.

The structural win is *multi-use* sharing: `linear_fwd`'s `tanh`
term is consumed by the forward plus all three gradients (4
consumers), so billing it once saves 3 copies — and greedy
extraction already captures it because the shared form is the
class optimum.  The general rule the probe supports: **joint
saturations pay when a subterm has ≥3 consumers across the
boundary; folds and one-use shares are a wash.**  A cost model
that prices launches/memory rather than op count could break the
tie differently — `flops_cost` is the obvious next column.

## Status / caveats

- Extraction correctness is pinned, not assumed: joint-extracted
  gradient terms evaluate numerically equal to `torch.autograd`
  (`test_joint_extraction_stays_correct`, the linear case — the
  one that shares hardest).
- The separate arm is the *strong* baseline: it bills the
  gradient program as one DAG (all gradient roots dedup among
  themselves), which is more than torch actually does at the FX
  level.  Joint still wins or ties everywhere.
- `shared` counts e-classes reachable from both the forward and
  gradient roots — an upper bound on exploitable sharing, not all
  of it billable under greedy extraction (see the wrinkle).
- Demo corpus is three hand-shaped cases (silu, softmax spine,
  linear+tanh).  The claim needs a real module sweep next —
  `TorchSource` already produces such terms (the autograd
  end-to-end test does this for `backward` alone).

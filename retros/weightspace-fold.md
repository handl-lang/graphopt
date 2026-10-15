# Retro — the weight-space lane: the conv+BN fold, derived or refused

**Date**: 2026-10-14.  **Tool**: `tools/probes/weightspace_fold_probe.py`
(merged `f83928d`).  **Depends on**: plan 0025 (the missing law
families), plan 0012 (bounded rewrites), ADR 0003.

## Why this lane

The measured boundary from this cycle: models whose math is already
fused expose nothing to the op-algebra laws (BiRefNet delivered
FLOP-neutral).  Three families were named as the expansion frontier;
**weight space** is the one that tests the project's no-hardcoding
rule — the fold must be **derived** by the machinery or **refused**
with a named gap, never hand-written.

The fold is a pure identity on parameters:

    W'[o] = W[o]·γ[o]/√(σ²[o] + ε)      b' = (b − μ)·γ/√(σ² + ε) + β

In IR terms it is a **derived-leaf** rewrite: the RHS weights are
computed from the LHS's leaves.  Real BiRefNet's decoder carries **43
`nn.BatchNorm2d`** modules (`birefnet-bounded-lane.md`) and no conv+BN
law ships.

## The probe's gate-by-gate account (measured)

| gate | verdict |
|---|---|
| **1 · binding source** | ✅ the fold is a measured parameter identity — `Conv2d+BatchNorm2d` (eval): `max|y_ref − y_fold| = 1.9e-06`; `Linear+LayerNorm` affine half: `1.8e-07` |
| **2 · `dspec_synth` fit** | ⚠️ **raw fold: no fit** (`W'` None, `b'` None at depth 2–3).  **Pre-scaled: it fits** — `W' = mul(W, scale)`, and at depth 3 `b' = add(beta, mul(scale, sub(b, mu)))`.  So the fold's *structure* is derivable **given a `scale` leaf**; the missing piece is the derivation of `scale = γ/√(σ²+ε)` |
| **3 · cond DSL** | ❌ **no `sqrt`, no `rsqrt`**; float-exponent `pow` declines (`fdiv`/`recip` are fine) — so the DSL cannot spell `scale` |
| **4 · derive value domain** | ❌ the derive value domain is **`int | float`** — a tensor-valued leaf raises (`TypeError: unhashable type: 'numpy.ndarray'`).  Even with an `rsqrt`, a per-channel `scale` is not expressible as a derived value |
| **5 · e-graph visibility** | ✅/❌ the IR **does** carry the node (`batch_norm` × 2 in the census), but **no law LHS names it** across 426 rules — nothing can rewrite it |
| **6 · cost model** | ⚠️ `batch_norm` is **absent from the FLOPs table** (default price 256.0); `sqrt`/`rsqrt` *are* present |
| **7 · derived-leaf seam** | ✅ exists, is procedural, and registers source tensors |

## The verdict: refused, with the gap named precisely

The fold is **structurally derivable** — the synthesizer finds the
correct two-expression form once a `scale` leaf is supplied, which is
strong evidence the machinery understands the rewrite's shape.  It
**cannot be completed** for two independent, measured reasons:

1. **The cond/derive DSL has no square root** (`sqrt`/`rsqrt` absent;
   float `pow` declines) — `scale = γ/√(σ²+ε)` is unspellable.
2. **The derive value domain is scalar** (`int | float`) — a
   per-channel tensor `scale` cannot be computed as a derived value,
   even if the DSL gained an `rsqrt`.

Plus two gaps that are *content*, not capability, and are therefore
the parts a human may legitimately author:

3. **No law LHS names `batch_norm`** — the rewrite does not exist in
   the library (this is what the derived law would *be*).
4. **`batch_norm` is unpriced** in the FLOPs table, so the fold would
   not be ranked even if it existed.

## What this buys the project

- **The no-hardcoding principle held under test**: the machine derived
  the fold's *structure* and refused the rest with named causes —
  rather than a human writing `conv_bn_fold` and calling it done.
- **A concrete, ordered backlog**: (a) tensor-valued derive leaves
  (the `source_tensors` seam exists but the value domain does not
  admit them), (b) `rsqrt`/`sqrt` in the derive DSL, (c) `batch_norm`
  in the FLOPs table, (d) the law itself — derivable once (a)+(b) land.
- **An OOM class found and fixed**: the probe's depth-3 `dspec_synth`
  sweep over the fold's seven-attr bank is **quartic in leaf count**
  and OOM-killed the process (measured: `MemoryError` under a 4 GB cap
  after ~17 s; an OS OOM-kill without one).  Routing every fit through
  `dspec_audit.bounded_synth` (same candidate space, deepest level
  streamed with early bail under a wall-clock budget) makes it run end
  to end — the same class of unbounded-sweep risk the discovery work
  has hit before.

## Honest limits

- Measured on CPU, small shapes (`W(6,4,3,3)`, `W(6,8)`), fp64 — the
  identity is what is being verified, not a speedup.
- The probe *refuses* rather than *derives*: this retro records a
  negative, and the negative is the result.
- No pay was measured, because nothing could be admitted — the two
  blockers above come first.

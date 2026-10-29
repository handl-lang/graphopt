---
title: 'Retro — conv+BN weight fold: the synthesis attempt, measured'
type: retro
date: 2026-10-18
---

# Retro — conv+BN weight fold: the synthesis attempt, measured

**Follows**: `leaf-compose-seam.md` ("What stays
open": try `synthesize_rules`/`dspec_synth` on a real conv→batch_norm
seed rather than hand-writing the fold).  **Probe**:
`lab/probes/conv_bn_fold_synth_probe.py` — gate-by-gate, reproducible.

**Verdict**: `synthesize_rules` **refuses structurally** — the fold's
RHS `conv2d(x, W′, b′)` is unreachable by composition; `dspec_synth`
**mints** the fold's derive exprs in stages.  The named blockers are
below; none is the `$leaf:` seam itself (that works — the emitted
composites prove it).

## The seed

Real export through `TorchSource`, eval mode, perturbed running stats:

```
(batch_norm (conv2d x, p_conv_weight, p_conv_bias,
              stride=(1, 1), padding=(1, 1)),
            p_bn_weight, p_bn_bias, b_bn_running_mean,
            b_bn_running_var,
            training=False, momentum=0.1, eps=1e-05,
            cudnn_enabled=False)
```

## Gate A — bank census: `conv2d` is absent from the law space

Across all 86 rules (`ALL_RULES` 83 + `BATCH_NORM_RULES` 3):
**zero** rules name `conv2d` in their lhs, **zero** in their rhs;
`batch_norm` appears only in the three expand lhses.  Every composite
`synthesize_rules` can emit has rhs `r1.rhs` with `r2.rhs` spliced
inside — the rhs head set is closed under the parents' heads.  A
`conv2d`-headed rhs is therefore *structurally unreachable* regardless
of composition depth: not a depth problem, not a seam problem — a
**corpus gap**.  The missing piece is an anchor: one `conv2d`-headed
law.  The authored precedent is `linear_channel_scale` (opt-in) — the
conv twin `conv_channel_scale`
(`mul(conv2d(x,W,b), ŝ) ↔ conv2d(x, mul(W,ŝ′), mul(b,s))`) is what the
corpus lacks; alternatively an anchored mint path — antiunify over a
measured (seed, folded) pair — the ADR-0008 bridge direction.

## Gate B — `synthesize_rules` on the real seed

| bank | numeric_check | emitted | lhs has `conv2d` | rhs headed `conv2d` |
|---|---|---|---|---|
| DEFAULT+BN (74) | False | 5 | 1 (`syn_batch_norm_expand__sub_to_add_0`) | **0** |
| DEFAULT+BN | True | 4 | 0 | **0** |
| ALL+BN (86) | False | 32 | 6 | **0** |
| ALL+BN | True | 26 | 1 | **0** |

The seed-guided path *does* produce the fold's lhs — leaf-generalization
over the LCA region mints `batch_norm(conv2d(…))` patterns.  Inside the
expanded term only `mul_reshape_inert_l` and `sub_to_add` match under
DEFAULT (comm/assoc plumbing is opt-in, not in DEFAULT); under ALL+BN
the comm/assoc variants compose too.  All rhs heads stay `add`/`mul` —
the spelled subgraph.

## Gate B2 — `numeric_check` drop is a NaN-env artifact

`numeric_check=True` drops `syn_batch_norm_expand__sub_to_add_0`:
`_tensor_env` draws `randn` envs, so `rv < 0` makes **both** sides
evaluate `rsqrt` of a negative → `allclose(nan, nan) = False`.  Not an
unsound rewrite — a witness-env precondition (`var ≥ 0`) that shape
guards cannot express.  Noted so the next sweep doesn't misread it.

## Gates C/D — `dspec_synth` mints the fold's derives *in stages*

| target | tier | result |
|---|---|---|
| `scale = γ/√(σ²+ε)` | attr scalars | **depth 3**: `fdiv(float(float γ), sqrt(add(ε, σ²)))` |
| `b′ = (b−μ)·scale+β` | attr scalars | **depth 3**: `add(β, mul(scale, sub(b, μ)))` |
| `W′ = W·scale` | attr scalars | **depth 2** (given `scale`): `mul(attr W, attr scale)` |
| `W′` one-shot `tval` | tensors | **refuse** — depth ≥4 AND `(C,)×(C,i,k,k)` broadcast decline |
| `W′` staged `tval` | tensors | **depth 2**: `mul(tval W, tval s)` — needs pushed channel-broadcast `mul` |
| `b′` staged `tval` | tensors | budget-truncated (level-3 product ≈ 26 G pairs; 14 M checked in 60 s) |

So the fold's `dspec` content is mintable; the composition gap is the
anchor above.  Two *sub-gaps* named for the tensor tier: (a) the
one-shot tree is depth ≥4 — a **staged fit** (mint `scale` first,
then `W′`/`b′` over it) is required, which today's single-shot
`synth_dspec` API doesn't do; (b) full-tensor `W′` needs either a
**pushed channel-broadcast `mul`** convention (adapter-owned typing —
demonstrated in the probe) or `tapp`/reshape in the enumeration bank
(`tapp` evaluates; it is not enumerated).

## Lab bug fixed en route

`lab/audit/dspec_audit.py::bounded_synth` predated the ADR-0012 `_vkey`
boxing: `values = {v for _b, v in samples}` crashed `TypeError:
unhashable type: 'numpy.ndarray'` on tensor-valued samples, and its
sig/expected equality compared raw arrays (`v != expected[i]` →
ambiguous truthiness).  Ported `synth_dspec`'s split — `_vkey`
fingerprints for the constant check and `_box` for signature equality;
ints-only literal seeds unchanged.

## What stays open (named, ordered)

1. **The anchor** — a `conv2d`-headed law (channel-scale absorb) or an
   anchored mint path (`antiunify`/`atom_synth_loop` over a measured
   seed↔folded pair).  Until one exists, composition has no `conv2d`
   rhs material.  Once it does, `batch_norm_expand` + the spelled
   reassociation + the absorb compose — the `$leaf:` seam carries the
   minted weights.
2. **Staged dspec fitting** — the fold's derive mints only when
   `scale` is a bound leaf: a `synth_dspec` that threads a
   previously-minted derive as a leaf.
3. **`tapp`/reshape or broadcast-mul in the enumeration bank** — the
   `(C,)×(C,i,k,…)` broadcast declines under vanilla numpy `mul`.
4. `BATCH_NORM_RULES` `DEFAULT` admission — the guarded sweep (item 2
   of this work order).

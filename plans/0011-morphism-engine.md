---
title: Plan 0011 — The morphism engine (hierarchical block-level optimization)
type: plan
date: 2026-10-14
---

# Plan 0011 — The morphism engine (hierarchical block-level optimization)

Status: draft — queued after 0010 lands.

## Goal

Whole-model optimization is bounded by tensor-level eqsat — exponential
on real stacks.  The generalization of the carrier idea one level up:
**lift each block to a morphism signature and optimize the
morphism-level graph** — a tiny object (N blocks, not N×ops enodes)
where composition is polynomial algebra, not saturation.

search : model → block signatures → morphism-graph laws → chosen
         composition
lower  : reify the chosen composition to concrete terms → lower
         per-block as today → certify per law

## The compression

- `optimize_compositional` caps at adjacent pairs; whole-model eqsat
  hits the enode cliff at ~8 blocks (the measured cost cliff).
- At the morphism level a 48-layer model is a 48-node graph with
  ~5 law families — search is rewriting, not saturation.

## Stage 0 — Block signatures (the algebra)

`BlockSig = (in_projs: [Param], out_proj: Param|None, norm:
{kind,dims}, act, residual: bool, shape)` extracted from the module
DAG via the Composer port.  Prove the algebra on 2-block fixtures:
`A.out ∘ B.in` composition through norms (diagonal absorption),
residual `+` monoid laws, `embed ∘ head` tying.

## Stage 1 — The morphism engine

The lifted graph (blocks as objects, wires as arrows) + laws that
operate *on signatures* — never touching tensor-level terms until
reify.  Each law carries its certificate recipe so reified programs
remain verified.

## Stage 2 — Headline transforms (ranked)

1. **Cross-layer KV latent sharing** (MLA-style) — fold K/V
   projections across layers into a shared low-rank latent.  Decode
   KV traffic drops ~head-count factor.
2. **Residual-stream reassociation** — global `+` monoid over N
   blocks; batch projections globally.
3. **Norm-fusion cascades** — norm diagonals flow through block
   chains into the next weights.
4. **Inter-block out∘in windows** — extend cross-pairs to windows ≥3.
5. **TP-shard planning** — fused-vs-split pairing at model scale.

## Stage 3 — Where it pays

Decode-bound LLM serving (bandwidth regime), long-context attention,
MoE at scale.  Vision/conv stacks remain parity until conv signatures
land.

## Stage 4 — Bar

Honest target: **>1.2× E2E on a real-scale model** from morphism
transforms alone, with certificates.  Below that, document why.

## Dependencies

- 0006 `search`/`lower` seam (done)
- 0007 `Composer` port (done — the abstraction boundary already exists)
- 0009 `RuleSet` (done — morphism laws become a preset)
- 0010 perf stages (in flight — not required; morphism search is
  already polynomial)

## Risks

- Signature coverage: only structurally-recognized blocks lift
  (opaque = boundary, same as carriers) — honest partial coverage.
- Laws that *look* composable can fail numerics verification when
  reified — verify fp64 at each level, not just the signature level.

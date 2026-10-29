---
title: 'Plan 0014 — Certified re-expression: from proofs to delivery'
type: plan
date: 2026-10-14
---

# Plan 0014 — Certified re-expression: from proofs to delivery

Status: draft.  Depends on 0006/0007/0008/0009 (done), 0011
(morphism engine, landed), 0012 (bounded, landed), ADR 0002 (the
thesis).  Feeds 0013 (contraction search) as the discovery layer.

## Goal

Convert the categorical reach the engine already proves into
delivered performance and memory where structure exists.  Today the
project is theory-rich and conversion-poor: it *finds* the
structures (carriers, scans, streaming softmax, KV sharing) but the
carrier forms lose to Inductor's fused kernels (per-leaf dispatch).

The plan makes **architecture the variable** — search certified
structural variants (head/KV sharing, expert tying, adapter
merging, residual reassociation) and deliver the cheapest variant
within a proven `error_budget`.

## Delivery — two paths

Delivery is not one problem; it splits by whether the extracted
program expresses in Inductor's op set:

- **Standard forms** (fold, reassociation, pairing, KV share, head
  collapse, tying) are ordinary matmul/add/etc.  Hand Inductor the
  *better program* and it fuses and schedules it — **the win lands,
  no gap to close.**  Extraction already routes here:
  `backend_cost(cost_fn, sink.supported_ops)` only prices forms the
  backend can lower.
- **Carrier forms** (scan, streaming softmax, batched levels) are
  outside Inductor's reach, so each leaf dispatches separately and
  the carrier loses.  These need the fused executor (stage 1).

Most architecture-level wins are the first kind; the fused executor
is a carrier-only concern.

## Mechanism — structural interpretation

The model graph is *syntax* (ops = generators of the free monoidal
category, wires = objects).  A **structural interpretation** is a
functor from that free category into a category of structures, and
the structure supplies the laws: monoid → associativity (Blelloch),
comonoid → `Δ` (pairing), traced → feedback (resolvents), hypergraph
→ contraction order.  The interpretation adds no information — the
function is fixed — it adds **legality**: which transformations are
valid.

The interpretation is **not unique**.  The same graph reads as plain
composition (nothing to do), shared-input comonoid (pairing),
low-rank (bounded), or a contraction problem — each exposing
different rewrites.  So *discovery = choosing the richest
interpretation*, not applying laws inside a fixed one.  That is what
carrier recognition, the morphism engine, and 0013 do at successive
levels.

Reach is the **closure** of the generators, not the generators
themselves: the law library is the generating 2-cells, and the free
higher category they generate *is* the reachable set.  The e-graph
already computes this higher structure (1-cells programs, 2-cells
rewrites, 3-cells coherence via `_close_congruence`); budgets
**truncate** it, and a truncation can miss but never mis-prove.

## Stages

0. **Real-block lifting** (the gate).  `_sig_liftable` declines
   multi-activation and mutated-cache blocks, so a real KV-cache
   attention block `(x, kv_cache)` never enters the search.
   Classify the cache as read-only state and lift multi-input
   attention blocks.  Nothing downstream moves until this lands.

1. **Fused carrier executor** (carrier-only).  Measured carrier
   losses are per-leaf dispatch, not search.  Fuse/tile
   `scan_lower` / `om_lower`, or capture decode into a CUDA graph,
   so a proven carrier converts to wall-time and memory instead of
   losing to Inductor's fused kernels.  Standard-op wins need none
   of this — they ride Inductor (see Delivery).

2. **Certified variant search** (architecture as the variable).
   Extend the morphism engine from match-and-apply to a search over
   structural variants, priced by a diagram-level cost model, each
   accepted only under a certificate (exact or within
   `error_budget`).  Targets the bounded wins that already
   reproduce (tied-head collapse, 1.15–1.32×) and the named niche
   (decode KV, adapters, redundant heads).

3. **Structure zoo + production scale** (the validation).  A corpus
   of structured models (GQA, MoE, LoRA-merged, MLA-style KV, tied
   heads) and measurement at production dims (d≥1024, B≥8, real
   checkpoint, datacenter GPU).  Required for any paper claim; the
   (now-retired) research writeup listed it as blocking.

4. **Contraction search (0013)** — the discovery feeder.  Replace
   the hand-written morphism laws with contraction-order search so
   sharing is *found*, not hand-coded.  Sequenced *after* delivery
   works: discovering structure you cannot deliver is a proof, not
   a speedup.

## Bar

- **Delivery:** the fused carrier path beats Inductor on decode
  latency / peak memory in the structured regime — not raw prefill
  GEMM throughput.
- **Conversion:** a reified structural win converts to wall-time or
  memory at a rate that makes search worth its compile cost.
- **Scale:** ≥1.2× on a real checkpoint at production dims with
  certificates, or a documented reason why not.

## Risks

- **Carrier delivery gap is a different discipline.** Codegen/fusion
  for carriers is not categorical rewriting; closing it may need
  work outside the thesis.  Standard-op wins ride Inductor and are
  unaffected — but a carrier win without a competitive executor is a
  proof, not a speedup.
- **Structure is conditional.** The census says stock dense models
  are thin; the wins live in structured models, so the zoo and the
  honest scope are load-bearing.
- **Diagram cost model.** Search is blind without a cost that ranks
  variants before reify; measure rank correlation (as
  `cost_fidelity.py` does) before trusting the search.
- **Compile time.** 262s at T=2048 vs Inductor's 45s — construction
  over saturation is also a compile-time lever.

## Non-goals

- Weight-space compression (ADR 0001).
- Beating Inductor on raw prefill GEMM throughput.
- Reinforcement learning.  Directed search (bound + decompose +
  beam) handles the exponential joint move space — the Catalan
  bracketing search already does — because structure is what makes
  search beat RL.

## See also

- `docs/mechanism.md` — the syntax → structure → match → rewrite
  loop this plan implements.
- `project/adrs/0002-categorical-re-expression-thesis.md` — the
  thesis and its mandatory consistency follow-ups.

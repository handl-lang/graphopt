# T-032 re-scope — `zoo:DecoupledMLA` uncaptured headroom

**Date:** 2026-11-14 · **Supersedes:** the T-032 row of
`docs/reports/t032_034_frontier_delivery.md` (its "different gap"
verdict is confirmed and sharpened; the "needs a fused-attention/MLA
kernel tier" guess is *wrong* — the attention already folds).

## What the workload actually is

`_DecoupledMLA` (`packages/mlopt-discovery/src/mlopt_discovery/zoo.py:1834-1907`)
is the DeepSeek-V2/V3 / Kimi k2 MLA spelling, measured at
`_DecoupledMLA(d=64, heads=4, rank=8, rpe=2)` on `(b=8, t=32, d=64)`
(`zoo.py:2658`).  Structure per forward:

- **Latent projections** — `dq`/`uq` (q down/up through rank-8
  latent), `dkv`/`uk`/`uv` (kv down, k-up, v-up), `qr`/`kr` rope
  slices, `out` — **8 `nn.Linear` GEMMs**;
- **Decoupled rope** — `_rope_tables` (`arange → pow → outer →
  cos/sin`, *recomputed every call* — loop-invariant in `t, r, dtype`)
  and `_rot` rotate-half `cat`/`mul`/`sub` chains on q/k;
- **Attention** — `softmax((q @ kᵀ) / √dh, dim=-1) @ v` spelled
  manually, then `out` projection.  Broadcast `expand` of the shared
  `k_r` across heads is the MLA-specific glue.

## Does an existing tier cover it? — the attention half, yes

Verified live this session (CPU intake, `_zoo_case` → `_gen_board` →
`_probe_play`, fp32): the e-graph `saturate` fires the shipped law
**`sdpa_fold_div_nomask`**
(`eqsat:packages/mlopt-core/src/mlopt_core/laws/tensor.py:2518`) —
`matmul(softmax(div(matmul(Q, Kᵀ), S), -1), V) → sdpa(Q,K,V,scale=1/S)`
with `S=4.0` a Const — and `extract_best` returns a term containing
**`sdpa`**.  So the manual softmax attention is already one fused
kernel; no attention/flash tier is missing.  (This is *not* a
collector gap: the fold is a `DEFAULT` law riding saturation, not a
minted claim, which is why `claim_tags`/`gen_handlers` show nothing —
the earlier report's "`gen_handlers` empty" is about the *tier*
machinery, not the law bank.)

## What the mint tiers saw, and why nothing was kept

`_mint_tiers` on the raw IR mints exactly five candidates
(reproduced):

| tag | pattern | verdict |
|---|---|---|
| `gen_0` | `sub(mul(X1,X2), mul(X3,X4))` — rope rotate-half | declined by `profitable` |
| `gen_1` | `pow(X1, div(neg(X2), 2))` — rope freq table | declined |
| `gen_2` | `div(neg(X1), 2)` — freq-table fragment | declined |
| `gen_3` | `add(mul,mul)` — rope rotate-half (sin half) | declined |
| `red_0` | `softmax(div(X1, 4.0), dim=-1)` — fused softmax body over the materialized scores | declined |

Every candidate was *minted* — the collectors are not blind — and
every one was declined by the measured-cost referee
(`genkernel.profitable`, fold-vs-spelled on real values).  `red_0` is
moreover subsumed: extraction picks the `sdpa` member, which already
contains softmax+scale+both matmuls in one kernel, strictly better
than a softmax-only fusion.  **Verdict (c) is ruled out: no collector
or spec gap.**  The referee is doing its job — at `(8,32,64)` fp32 the
rope chains and a standalone softmax kernel do not beat their spelled
eval often enough to pay launch+compile overhead.

## Where the residual headroom actually lives

Post-fold the delivered module still dispatches **17 kernels**
(`frontier_fp32.json` `kernels: 17`): 8 projection GEMMs that no
pointwise/reduction tier can fuse, plus glue — the per-call rope-table
block (`arange/pow/outer/cos/sin`, loop-invariant but recomputed), the
`_rot`/`cat`/`expand`/`reshape`/`transpose` choreography around the
`sdpa`, and the `/4.0` scale.  `torch.compile` (Inductor) wins the
*lowering* column by fusing that glue into GEMM epilogues/prologues
and hoisting the tables — `vs_compile` 0.69 (fp32) / 0.80 (fp64).

But the frontier's own new column answers the product question: the
**same verified module under `CudaGraphRunner`** measures
`graph_vs_compiled` **1.77 (fp32) / 1.03 (fp64)** — `beats_graph_compile:
true` on both rows (`frontier_fp32.json`, `frontier.json`,
T-037 re-sweep).  The "uncaptured headroom" (`dispatch_share 0.28`,
`headroom_us ≈ 118–132 µs`) is *launch dispatch*, and the shipped
graph arm already captures it: fp32 `captured` on the lowering column
is 0.94 with the graph arm at 3.49× eager / 1.77× compile.

## Re-scoped verdict — honest boundary, with two named (small) residuals

**(b): the row is an honest boundary for the lowering path, and the
headroom is already captured by the graph delivery arm.**  No new
tier is needed for the attention itself — `sdpa_fold_*` owns it.  The
remaining eager-path gap vs Inductor is whole-graph glue fusion +
loop-invariant hoisting, worth ~78 µs fp32 (287→209 µs) — real but
already moot under `CudaGraphRunner`, which beats compile outright.

If a future agenda item wants the *non-graph* lowering to close it,
the concrete missing pieces, in ascending size:

1. **Loop-invariant / const hoisting** (small): the rope-table block
   depends only on `(t, r, dtype, device)` — all compile-time-known.
   No machinery exists (no const-fold or LICM tier in `TIERS` or the
   law bank).  Estimated win: a handful of launches per call — but the
   *graph arm already gets this for free*, so the marginal value is
   near zero unless graph capture is unavailable (CPU eager, dynamic
   shapes).
2. **Glue-body fusion across view ops** (medium): `_rot` + `cat` +
   `expand` + `transpose` around `sdpa` — the gen tier's spec grammar
   is elementwise-only and declines view ops by design
   (`reduction_spec`'s boundary rule, `genkernel.py:3531`); widening
   it is a grammar change, not a new tier.  Win bounded by the same
   dispatch share the graph arm already takes.
3. **Fused-MLA monolithic kernel** (large, *not* recommended): a
   FlashMLA-style Triton kernel fusing latent projections + rope +
   attention.  This is the only lever that would beat Inductor on the
   *eager lowered* path, and it is a research-grade kernel effort —
   while the shipped graph arm already delivers 1.77×/1.03× vs
   compile.  Poor value.

**Dtype caveat stands:** fp64 `speedup` is 0.99 (delivered ≈ eager)
vs fp32 1.36 on identical structure — the fold's sign is the backend
kernel's (T-021), and fp64 `graph_vs_compiled` is only 1.03.  Any
future measurement of this row must keep the dtype explicit.

## Suggested agenda text

> **T-032 (re-scoped, 2026-11-14):** `zoo:DecoupledMLA` — attention
> already folds to `sdpa` via `sdpa_fold_div_nomask`; all five minted
> tier candidates (`gen_0..3`, `red_0`) were honestly declined by
> `profitable`.  Residual vs compile is launch glue the `CudaGraphRunner`
> arm already beats (1.77× fp32 / 1.03× fp64 vs compile).  **Close as
> boundary** unless eager-path (non-graph) parity with Inductor becomes
> a goal — then the lever is loop-invariant hoisting + view-op glue
> fusion (medium), not a new attention tier.

## Evidence index

- Workload: `zoo.py:1834-1907` (module), `zoo.py:2658` (registry).
- sdpa fold fires: `eqsat …/laws/tensor.py:2334` (`_QK_SCORES`),
  `:2518` (`SDPA_FOLD_DIV_NOMASK`); verified by extraction containing
  `sdpa` (this session, `_gen_board`/`_probe_play`).
- Tier grammar boundary: `genkernel.py:3531-3553` (`reduction_spec`),
  `TIERS` at `genkernel.py:4454-4471`.
- Minted-then-declined candidates: `_mint_tiers` output reproduced
  above; referee = `genkernel.profitable` (`genkernel.py:768`),
  gated in `play.py:2104-2136` (`_zoo_case`).
- Frontier rows (post-T-037): `docs/reports/frontier_fp32.json` /
  `frontier.json`, case `zoo:DecoupledMLA` — `kernels: 17`,
  `beats_graph_compile: true` both dtypes.

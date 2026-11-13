# Zoo corpus expansion — workload-shape gap analysis + fifth family

**Date:** 2026-11-13 · **Registry:** `mlopt_discovery.zoo` grew 68 → **73**
workloads (plus the `zoo_variants` mirror, also 73).

## Why

The last several zoo surveys returned ~0 paying sites.  `corpus_cov`
 did a coverage census of the *meta-arena* op/attr/handler vocabulary —
but nobody had censused the *workload shapes* themselves against the
shipped tiers.  This report is that census, the five workloads it
motivated, and the honest per-workload verdict.

## The paying tiers and what they can see

| tier | collector (`play.TIER_SITES`) | what mints a site |
|---|---|---|
| `gen` (pointwise fusion) | `_fusion_sites` | ≥2-op elementwise chains (clamps/attrs included) |
| `decayed` | `_decayed_sites` | `matmul(mul(s,d),v)` / `om_decayed` contraction sites |
| `scan` | `_all_scan_sites` | saturated `applyd` carriers **+** raw `add(mul(a,h),x)` select-chains (`seq`) **+** state-feedback `h = h + f(h)` chains (`seqm`) |
| `reduction` | `_reduction_sites` | reduce-dim heads (`softmax`, `mean`, …) over elementwise bodies |

Adjacent pay paths outside `gen_probe`'s minted-claim table: sdpa-fold
delivery inside the torch pipeline, and CUDA-graph capture
(`gen_graph_us` in every `gen_probe` row — the dispatch-bound win the
`zoo_chain` series measured).

## Shape census — what the 68 covered vs what was missing

Covered: single-head diagonal selective scans (Mamba S6, GLA, RWKV,
RealGatedLRU), matrix-state delta rules (DeltaRuleBlock, GatedDeltaRule),
chunked dual forms (ChunkedRetention, ChunkedSSD — the cumsum *inside*
a chunk), full-attention softmax bodies (many variants), quant/cache
deployment shapes, conv stacks, routing, heads.

**Missing combinations** (the gaps this expansion targets):

1. **conv + recurrence hybrid** — every scan workload feeds the
   recurrence raw projections; the production Mamba block's causal
   depthwise `conv1d` + SiLU front-end and `silu(z)` output gate were
   absent.  Conv work is dispatch-bound; the seq fold cannot touch it.
2. **multi-head scan** — all scan states were `(b, …)`-headed; no
   workload put a head axis *inside* the recurrence's broadcast frame
   (`(b, h, dh, n)` state).
3. **cumsum carrier** — `ChunkedSSD`/`ChunkedRetention` use `cumsum`
   for *decay bookkeeping inside a chunk*, but no workload spells the
   scan ITSELF as a `cumsum` prefix carrier (linear attention's
   running sufficient statistic).  The seq collector only reads
   `select`-chains — a `cumsum` op is invisible to it by construction.
4. **bidirectional / reversed select-chains** — every recurrence ran
   forward over `x[:, i]`; no workload carried a second chain over a
   *flipped* operand.
5. **latent × decode** — `_DecoupledMLA` is MLA's *prefill* spelling
   (full sequence, `T_q = T`); the incremental decode shape (one query
   token against a stored latent KV cache + raw rope keys) was absent.

Also noted, already-covered-but-lossy: recurrences returning **all
states** (GLA, MambaSelectiveScan — `module_loss_gla_mamba.md` showed
the route folds only the final state while intermediates stay spelled).
The new scan workloads keep that honest shape deliberately.

## The five new workloads

All follow the `_Workload` conventions: `nn.Module` + `forward`,
`torch.zeros(..., dtype=x.dtype, device=x.device)` inside `forward`
(the fake-CUDA export test pins this), registered as `kind="zoo"`
thunks in both `zoo()` and the `zoo_variants()` `*V2` mirror.

| workload | shape | target tier |
|---|---|---|
| `MambaConvScan` | in_proj → split → causal depthwise conv1d+SiLU → S6 scan → `* silu(z)` gate → out_proj; returns all read-outs | `scan` (seq) + `gen` + conv dispatch |
| `HeadedSelectiveScan` | Mamba S6 with `(b,h,t,dh)` layout, state `(b,h,dh,n)`; returns all read-outs | `scan` (seq, head axis in frame) |
| `CumsumLinearAttention` | `S_t = cumsum(k_i ⊗ v_i)`, `o_t = q_tᵀ S_t` — linear attention as prefix carrier | negative control (all tiers) |
| `BidirectionalScan` | `h = a⊙h + x` forward on `x` AND on `flip(x)`, per-direction decay, outputs summed | `scan` (seq ×2 expected) |
| `LatentKVDecode` | MLA decode: latent KV cache `(b,T,rank)` + raw rope keys `(b,T,rpe)`, one query `(b,1,d)`, manual softmax | `reduction` (softmax body) + `gen` |

## Verification

Protocol: `play._zoo_case(wl)` (TorchSource export → `_mint_tiers` →
`genkernel.profitable` measured-cost referee) then `play.gen_probe`
(deliver → verify → time vs eager AND `torch.compile`, CUDA, medians).
Two shape points: the zoo default `d=16 t=8 b=2` fp64, and the paying
shape `d=64 t=32 b=8` fp32 (the `zoo_chain` convention).

### Sites minted (pre-`profitable`, fp64 default shapes)

| workload | gen | scan_sat | scan_seq | scan_seqm | reduction | decayed |
|---|---|---|---|---|---|---|
| MambaConvScan | 137 | 16 | **1** | 0 | 0 | 0 |
| HeadedSelectiveScan | 136 | 16 | **1** | 0 | 0 | 0 |
| CumsumLinearAttention | 0 | 0 | 0 | 0 | 0 | 0 |
| BidirectionalScan | 128 | 16 | **1** (of 2 chains) | 0 | 0 | 0 |
| LatentKVDecode | 36 | 0 | 0 | 0 | **1** | 0 |

The minted seq specs:

- `MambaConvScan`: `applyd(aff_diag(Q1, mul(Q2,Q3)), H, kind=seq,
  axis=1)` — the rank-1 write `x⊗B` abstracted as `mul(Q2,Q3)`; the
  conv front-end is invisible to the collector (correct — it is not
  part of the recurrence).
- `HeadedSelectiveScan`: identical spec shape, `axis=2` — the head
  axis normalises into the step axis cleanly; **the `(b,h,dh,n)`
  broadcast frame did NOT break the seq mint.**
- `BidirectionalScan`: `applyd(aff_diag(Q1,Q2), H, kind=seq, axis=1)`
  — **only one of the two chains minted.**  The backward chain's spec
  is the *identical tuple* (`aff_diag(Q1,Q2)` — metavar names are
  positional) and `_seq_scan_sites` dedups on `spec ==` without
  comparing `leaf_env`, so the flipped-operand site collapses into
  the forward one.  Honest collector boundary, now measured: two
  same-shaped recurrences in opposite directions mint ONE site.
- `LatentKVDecode`: `('softmax', ('div','X1',2.0), {'dim':-1})` — the
  score/body softmax minted a reduction spec; `profitable` gated it
  out (0 claims offered).
- `CumsumLinearAttention`: **zero sites of every tier** — the
  negative control behaved exactly as designed: `cumsum`, `einsum`,
  `unsqueeze`-outer and `matmul`-free bodies mint nothing the
  collectors can read.

### Does it pay? (`gen_probe`, verified deliveries only)

fp64, `d=16 t=8 b=2`, reps=50, CUDA:

| workload | claims offered | delivered | verified | gen÷eager | gen÷compile | graph÷eager |
|---|---|---|---|---|---|---|
| MambaConvScan | 38 | yes | yes (2.8e-17) | 0.57 | 0.26 | **2.72** |
| HeadedSelectiveScan | 0 | yes | yes | 0.81 | 0.36 | **4.72** |
| CumsumLinearAttention | 0 | yes | yes | 0.84 | **1.26** | 1.72 |
| BidirectionalScan | 7 | yes | yes | 0.42 | 0.19 | 1.97 |
| LatentKVDecode | 0 | yes | yes | 0.87 | 0.47 | **3.17** |

fp32, `d=64 t=32 b=8`, reps=50 (paying shapes):

| workload | claims | delivered | verified | gen÷eager | gen÷compile | graph÷eager |
|---|---|---|---|---|---|---|
| MambaConvScan | 40 | yes | yes | 0.48 | 0.13 | 2.12 |
| HeadedSelectiveScan | 0 | yes | yes | 0.93 | 0.32 | **6.97** |
| CumsumLinearAttention | 0 | yes | yes | 0.94 | 0.71 | 1.04 |
| BidirectionalScan | 60 | yes | yes | 0.19 | 0.03 | 0.18 |
| **LatentKVDecode** | 0 | yes | yes | **1.23** | 0.66 | **4.14** |

## Honest verdict

- **Corpus value: real.**  All five export cleanly through
  `TorchSource` → `ingest` (the `test_zoo_*` suite passes: 328 tests),
  and four of five mint collector sites the 68-workload census never
  produced — including the first `seq` mint with a head axis inside
  the frame and the first conv+scan hybrid.
- **Paying value: one payer, and not through a minted claim.**
  `LatentKVDecode` beats eager **1.23×** at the paying shape (fp32,
  d=64/t=32/b=8) with **zero claims offered** — the win is the
  saturate→extract delivery path (shipped-law simplification +
  dispatch), the same mechanism `zoo_chain` measured for the
  zero-claim dispatch payers, not a minted kernel.  Its graph arm
  hits 4.1×.  The other four lose at module level in both dtypes
  (BidirectionalScan catastrophically at fp32-large: 60 minted
  `gen_*` claims delivered a 6 ms term vs 1.1 ms eager — the
  pointwise-fold multiplication the GLA/Mamba retro already
  documented).  Graph-capture arms win broadly on the
  dispatch-bound members (HeadedSelectiveScan 7.0×).
- **Two measured collector boundaries** (the actual gap-analysis
  yield): (a) `cumsum`-spelled scans are invisible to every tier —
  `CumsumLinearAttention` mints zero sites despite being the same
  math as the paying delta rules; (b) same-shaped recurrences in
  opposite directions dedup to one seq site because spec identity
  ignores `leaf_env`.  Both are now pinned by corpus members, not
  just by argument.
- `HeadedSelectiveScan` offered **0 claims** despite minting a seq
  site + 16 saturated carriers — the board's offer gate declined
  them (the `aff_diag` spec's `b` operand `mul(Q2,Q3)` is the
  rank-1 write; `profitable`/offer-measurability filtered it).  It
  still delivered+verified through saturate→extract.

## Pins updated

- `tests/test_discovery_zoo.py::_EXPANSION_FAMILIES` += the five names
  (both draws pinned by `test_zoo_expansion_families_registered`).
- No numeric zoo-count pins exist in tests (`>= 10` only); the "68"
  figures in `docs/reports/*.md` are historical measurements of the
  old registry — this report is the new count (73).

## Reproduce

```bash
# sites minted per tier (pre-profitable)
.venv/bin/python - <<'EOF'   # see session notes: TIER_SITES walk over
                             # TorchSource().to_ir(model, feed).root
EOF
# end-to-end probe
.venv/bin/python -m mlopt_discovery.play --domain gen --zoo \
    --zoo-d 64 --zoo-t 32 --zoo-b 8     # or play.zoo_probe()/gen_probe
```

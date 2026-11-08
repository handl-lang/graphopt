# T-010 remainder levers — fp32 reduction gap at scale + attr-bearing bodies

Date: 2026-11-08.  Follow-up to `p33_reduction_triton_v2.md` (the
fp32-vs-Inductor gap) and the gen-kernels retro's "attr-bearing
bodies — occurrence-blocked" note.  Both levers measured; one
honest negative refined into a real positive, one boundary mapped.

Raw sweeps: `red_triton_probe_v2_fp32_128_all.json` (lever A),
`attr_elementwise_probe_{fp64,fp32_128,fp64_128b8}.json` (lever B).
Prior agent's leftover `red_triton_probe_v2_fp32_128.json`
(RWKVMixBlock-only fp32-128 cell) is subsumed by the full sweep.

## Lever A — fp32 reduction-vs-Inductor at 128×128×16

Full zoo sweep, `reduction_triton_probe --d 128 --t 128 --b 16
--dtype fp32 --probe` (RTX 2050, serial):

- **30 bound consults, 3 kept** — vs 1/30 at fp64-style 64×64×8
  (p33) and 0/30 at zoo 16×8×2.  Size is the whole story: at 128 the
  tiled kernels amortise launch+tile overhead and fp32 keeps appear.
- Kept consults (kernel/spelled): RWKVMixBlock `red_1` 0.535×,
  ALiBiSlopeAttention `red_0` 0.779×, SinkWindowAttention `red_0`
  0.899×/0.906×.  All other bound consults sit at 1.01–2.90× —
  dropped.

Delivered-module pricing (`gen_probe`, the number the lever is
priced on):

| workload | claims | verified | vs eager | vs compiled |
|---|---|---|---|---|
| ALiBiSlopeAttention | red_0 | yes (1.8e-6) | **1.828×** | **1.255×** |
| SinkWindowAttention | red_0 | yes (2.5e-7) | 1.043× | 0.899× |
| RWKVMixBlock | red_1 | **no** (NaN) | 0.837× | 0.057× |

**The fp32 gap is not absolute — it has a paying corner.**  The two
softmax-headed keeps are fused attention-logit bodies
(`softmax(add(mul(X1, s), X2), dim=-1)` — scale+mask inside the
reduce): ALiBiSlopeAttention's delivery beats whole-model
`torch.compile` by 25%, the first fp32 reduction-kernel win vs the
incumbent anywhere in the sweep.  SinkWindowAttention's keeps on
spelled-relative merit but loses to compiled (0.899×) — same
asymmetry p33 recorded.  RWKVMixBlock's `red_1`
(`sum(mul(exp(add(X1,X2)),X3), dim=(2,))`) keeps on kernel timing
yet the delivered module fails verification at fp32 — the same
accumulation-order tolerance issue p33 noted for `red_0`; the keep
is hollow and recorded as such.

### (b) Can the emitted kernel improve for fp32?

Read against the v2 emitter (`genkernel.py` `_red_*`): the kernels
are already dtype-agnostic (`out_ptr.dtype.element_ty`), int32
decode, `(row-blocks, chunks)` tiled with split-K partials, cached
`_red_ometa`, `pow(x,2)→x*x`.  The operand loads go through the
ometa gather offsets (`__o{i} = __ro + __co`) — there is no
contiguous dimension for `tl.max_contiguous`/`tl.multiple_of` to
exploit, and `BLOCK`/`BM`/`BR`/`NC` are already shape-tuned by
`_red_tile_dims`.  No cheap fp32-specific lever identified; a real
step would be layout-specialised (stride-1 fast path) codegen, a
project not a patch.

### (c) The honest boundary

- **fp32 wins exist but are narrow**: large (≥128-class) fused
  *softmax* reduce bodies inside attention modules.  Sum/mean/amax
  bodies at the same sizes keep on spelled-relative timing yet lose
  to Inductor at module level (SinkWindowAttention 0.899×) or fail
  fp32 verify (RWKVMixBlock).
- At 64×64×8 fp32 the picture is unchanged from p33: 1 keep,
  0.175× vs compiled — Inductor's whole-module fusion still wins
  the mid-size regime.
- At zoo sizes (16×8×2) nothing pays in either dtype.
- fp64 corner persists at scale: the attr probe's intake (below)
  shows RWKVMixBlock keeping `red_0`+`red_1` at fp64 128×128×8.
  (A dedicated fp64-128×128×16 full sweep was not run — fp64 at
  b=16 OOM-killed the probe process; b=8 completed for the attr
  subset.)

Measurement caveats: one dynamo `recompile_limit` warning fired
inside `genkernel.kerneled` during intake (compiled spelled kernels
can silently degrade to eager past 256 recompiles — keep votes
near the margin inherit the documented `profitable` noise); one
CUDA OOM allocator warning, run completed.

## Lever B — attr-bearing bodies: not occurrence-blocked, alphabet-blocked

**Census re-run** (all 68 zoo workloads, `TorchSource` IR walk):
the p32-era claim "zero attr-carrying elementwise ops" is confirmed
*within `genkernel.ELEMENTWISE`* — but the census was
alphabet-scoped, not corpus-scoped.  The IR carries **28
attr-bearing `clamp`/`clamp_min`/`clamp_max` sites across 11
workloads** (CosineAttention, GCNLayer, AffineCoupling,
CapsuleRouting, ChunkedRetention, RWKVMixBlock, STEQuantLinear,
RelPosTableAttention, SinkWindowAttention, TaylorLinearAttention,
TernaryWeightLinear) — elementwise ops with concrete `min`/`max`
attrs, excluded only because the alphabet is a deliberate narrow
claim.  (`softplus` is in the alphabet and attr-capable but the
zoo spells it attr-free; `elu`/`hardtanh` have zero sites.)

**Measured through the real spec path** — new driver
`bench/drivers/probes/attr_elementwise_probe.py` widens
`gk.ELEMENTWISE` **in-process** by the three clamp ops (no shipped
file touched; `_fusion_sites`/`_fusion_pattern`/`elementwise_spec`
all read the module global), then runs the unmodified intake:
`_mint_tiers` → attr-carrying `gen_*` specs → `profitable` spelled
pricing → `_claim_cost_gate` → `gen_probe`/`hybrid_sink` delivery.
`_triton_node_h` declines the attr dicts exactly as recorded, so
delivery is the `torch.compile`d spelled binding — the post-T-026
machinery (`_canon_taken`, attr matching, `spec_bind`) carries
everything else.

Results:

| cell | attr claims minted | kept | delivery |
|---|---|---|---|
| fp64 16×8×2 (full zoo) | 22 across 11 workloads | 0/22 | — |
| fp32 128×128×16 (11 workloads) | 22 | **1/22** | see below |
| fp64 128×128×8 (11 workloads) | 22 | 1/22 | see below |

- **fp32 128×128×16 — SinkWindowAttention `gen_0` KEPT and
  delivered**: spec `add(…, clamp_max(max=1.0), …)`-class body
  (two clamp_max attr sites inside the fusion).  Referee priced
  0.828×/0.833× kernel-over-spelled; `gen_probe` delivered a
  **verified** module (max_abs 2.4e-7) at **1.253× vs eager,
  1.056× vs whole-model `torch.compile`**.  Attribution caveat:
  the delivered module also carries `red_0` (the lever-A softmax
  keep) — the 1.056× is the *combined* module, not gen_0 alone.
  What is proven regardless: an attr-bearing spec minted from real
  corpus, survived the measured referee, delivered through
  `hybrid_sink`, and verified end-to-end.
- **fp64 128×128×8 — same `gen_0` kept**, delivered verified
  (max_abs 4.4e-16) but priced **0.86× eager / 0.85× compiled** —
  loses to Inductor at fp64 here (b=8 vs the fp32 run's b=16; the
  fp64 b=16 cell OOM-killed and is reported as not-run).
- At zoo sizes every attr claim drops (compiled-spelled loses
  1.05–2.9× to eager spelled — the same dynamo-overhead regime as
  every other tier).

**Verdict:** the "occurrence-blocked" note was wrong in an
interesting way — the corpus had the sites all along; the
*alphabet* excluded them.  With the three clamp ops admitted
in-process, 22 real attr-bearing claims mint and price correctly,
and at fp32-128 one delivers a verified module that beats
`torch.compile`.  The remaining gap is unchanged: Triton declines
attr nodes, so attr bodies ship as compiled spelled — they pay
only where spelled-fusion beats Inductor, the same narrow corner
as everything else.  **Whether `clamp`/`clamp_min`/`clamp_max`
belong in shipped `ELEMENTWISE` is a product call, not a
measurement gap** — flagging the blast radius: `ELEMENTWISE` is
shared vocabulary (`handler_synth`/`family_synth` mint *laws*
over it, `vocabdir` publishes it), so widening it changes the
discovery grammar, not just gen claims.  Recommend a scoped
decision, not a silent widening.

## Files

- `bench/drivers/probes/attr_elementwise_probe.py` — new driver
  (in-process `ELEMENTWISE` widening + `_mint_tiers` spy; ruff
  clean).
- `docs/reports/red_triton_probe_v2_fp32_128_all.json` — lever A
  full-zoo fp32 128×128×16 sweep.
- `docs/reports/attr_elementwise_probe_{fp64,fp32_128,fp64_128b8}.json`
  — lever B cells.
- `docs/reports/red_triton_probe_v2_fp32_128.json` — prior
  agent's RWKVMixBlock-only cell (subsumed, kept for provenance).
- No shipped package code changed; `tests/test_discovery_genkernel.py
  -k reduction` re-verified (11 passed).

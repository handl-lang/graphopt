# BiRefNet bounded lane — the precision lane on a real pretrained vision model

Question: does `search(..., error_budget=)` buy headroom on a real
pretrained model where the *exact* lane was already FLOP-neutral?
`bench/suites/bounded/bounded_e2e.py` delivered certified-bounded
members on stories15M (1.41× vs eager at 1e-2); the `34607e0`
BiRefNet trial exported the real checkpoint but found no exact-lane
win. The bounded lane had never been opened on it.

Probe: `lab/probes/birefnet_bounded_probe.py` (2026-10-14,
`feat/bounded-real-model` worktree, RTX 2050 3.68 GB, torch
2.14.0+cu130, `TORCHINDUCTOR_COMPILE_THREADS=2`).

Model: **the real `ZhengPeng7/BiRefNet` HF checkpoint** — Swin-v1-L
+ ASPPDeformable decoder, 220.18 M params fp32 (~886 MB
safetensors, snapshot `e2bf8e4`), loaded via
`AutoModelForImageSegmentation.from_pretrained(...,
trust_remote_code=True, dtype=torch.float32)` — transformers
5.19.0, timm 1.0.30, torchvision 0.29.1, kornia 0.8.3, einops —
scratch `--no-deps` installs into the worktree venv (documented in
the tool docstring). Input `(1,3,384,384)` (the swin_v1_l native
resolution; window 12 divides the 96×48 token grids, including the
`x/2` second backbone pass). Eager forward: ~2.8 s CPU / ~0.28 s
CUDA, 1.2 GB steady-state VRAM.

## Structural census — the decoder BN question (task item 3)

Exported `model.decoder` alone on its real captured input (forward
hook): **237 enodes — `batch_norm` ×35**, `conv2d` ×68, `relu_` ×31,
`mul` ×19, `sigmoid` ×19, `deform_conv2d` ×16, `upsample_bilinear2d`
×13, `reshape` ×10, `concat` ×9, `permute` ×5, `dropout` ×4,
`adaptive_avg_pool2d` ×4, `add` ×3, `zeros` ×1.

**Verdict: BN is present, not LN/conv-only.** The decoder
(`BasicDecBlk` ×5 + ASPPDeformable ×5 + `gdt_convs`) is a
conv→BN→relu graph — conv+BN fusion targets genuinely exist (43
`nn.BatchNorm2d` modules in the decoder/squeeze tree, all live in
eval mode — `config.batch_size=4 > 1` keeps them real). But no
conv+BN fold law exists in the rule set today, so they lower as
`batch_norm` nodes, not fusion sites. Side finding: `deform_conv2d`
*exports* into IR (16 nodes) but has **no torch lowering binding**
— leaf blocks containing it fail lower with
`ValueError: No torch binding for op 'deform_conv2d'`. The earlier
trial's "exports and lowers (no missing bindings)" never lowered a
deform site; the gap is real.

## Compositional plumbing — what the run needed to survive

Two model-shape traps had to be routed before the sweep ran:

* `mul_scl_ipt='cat'` calls `model.bb` **twice per forward** — once
  on `x`, once on `x/2`. A delivered block is *shape-specialized*
  for its first captured call (export bakes reshape constants), so
  any grafted `BasicLayer`/`SwinTransformerBlock` silently breaks
  the second (half-resolution) call — and the compositional
  e2e check only *records* the failure in `stats["end_to_end"]`,
  it does not roll back grafts. Separately the blocks read
  `self.H`/`self.W` mutated by the parent per call, so the
  captured-input replay sees the stale half-res values and the
  block's own `assert L == H*W` fires (all 24 failed in the first
  run). Fix: `_block_pred` declines `BasicLayer` /
  `SwinTransformerBlock` / `BasicDecBlk` /
  `_ASPPModuleDeformable`, so selection descends to the
  shape-generic leaves — and `_LeafComposer` restricts grafting to
  `nn.Linear` (the only leaf class the bounded specials scan:
  `offer_weight_specials` is 2-D `linear`/`matmul` only).
* Host-RAM: `capture_inputs` retains a detached clone of every
  selected block's input plus a live reference to its output —
  ~6 GB on the full 498-block selection at 384², which OOM-killed
  the first sweep (the same mechanism, smaller, is why
  `bounded_e2e` survives on a 15 M model). The Linear-only
  selection needs ~0.5 GB.

Sweep: `Optimizer(backend=TorchBackend(composer=_LeafComposer()))`
→ `optimize(search_model_cpu, x, strategy=Compositional(block_pred=
...)`, `detect_specials=True`, `error_budget=B)` — search on CPU,
delivered module re-verified and timed on CUDA. 99 Linear blocks
selected, all 99 optimized every cell (structural cache 80 hits /
19 misses). Timing: `_time_module` median of 30 reps under
`no_grad`; inductor baseline compiled once (10.5 s) and timed once.
Evidence JSON: `bench/results/birefnet_bounded.json`,
`birefnet_bounded_3e-2.json` (untracked).

## Per-budget results

| budget | opt s | bounded offers | delivered | bound max | fresh rel | mask IoU | catopt µs | vs eager | vs inductor |
|--------|-------|----------------|-----------|-----------|-----------|----------|-----------|----------|-------------|
| none   | 38.6  | 0  | 0 | —      | 0.0 | 1.0000 | 277 948 | 0.999× | 0.862× |
| 1e-4   | 205.6 | 0  | 0 | —      | 0.0 | 1.0000 | 277 005 | 0.997× | 0.865× |
| 1e-3   | 211.2 | 0  | 0 | —      | 0.0 | 1.0000 | 276 043 | 0.998× | 0.868× |
| 3e-3   | 221.9 | 0  | 0 | —      | 0.0 | 1.0000 | 277 953 | 0.997× | 0.862× |
| 1e-2   | 229.2 | 20 | 0 | —      | 0.0 | 1.0000 | 279 075 | 0.996× | 0.859× |
| 3e-2   | 364.1 | 56 | 0 | —      | 0.0 | 1.0000 | 282 116 | 0.993× | — |
| 1e-1   | —     | —  | — | OOM-killed mid-cell (dedup clustering host RAM; another agent was resident) |

Baselines: eager 277 551 µs (peak 2 032 MB), inductor 239 660 µs
(peak 1 974 MB) — compiled source is already **1.16× faster** than
eager on this model.

Offer frontier (all `elide_bounded`, `bound_norm=max_abs`):

* **1e-2** — 20 offers, exactly the `attn.qkv` projections of
  stages 2–3 (bounds 7.75e-3 stage 2, 9.95e-3 stage 3).
* **3e-2** — 56 offers: the same 20 qkv plus `mlp.fc1` and
  `mlp.fc2` of all 18 stage-3 blocks (bounds 1.17e-2 – 2.97e-2).

So the near-duplicate-row structure is *real and pervasive* in the
pretrained backbone — every stage-2/3 attention and MLP projection
carries rows within ~1e-2–3e-2 elementwise of an earlier
representative.

## Why nothing delivered — extraction economics, not the gate

`n_bounds_delivered == 0` at every budget and **no block was
declined by the bound gate** (no `accepted bound` error strings) —
the bounded members simply were never the cheapest extracted term.
Measured directly on `bb.layers.2.blocks.0.attn.qkv` (search →
`alternatives`):

* dense `linear(input, W, b)` — 832 773 backend-priced FLOPs;
* `elide_bounded` member — `index_select`-re-expanded shrunk weight
  `p_weight__bl2286x768` + bias add — **1 099 421**.

Only **18 of 2304 rows** dedup at ≤9.9e-3 (0.78% of the weight), so
the gather/re-expand/bias-add wrapper costs 1.32× the dense member.
Stories15M paid because its tied 32 000-row head is *dominated* by
near-duplicates; BiRefNet's structure is real but ~0.8% thin —
launch-aware pricing refuses it, correctly.

fp32 gate: every delivered module is bitwise identical to source
(`fresh_rel = 0.0`, e2e `max_abs_diff = 0.0`) — the extracted terms
were all exact members. Held-out mask proxy: `max_abs = 0`,
IoU = 1.0 on all cells.

## Verdict

**The precision lane does not pay on BiRefNet — not because the
bound machinery failed, but because the real structure is too thin
to price.** The bounded offers exist (20–56 sites, certified bounds
inside budget), the bound gate stayed honest, and the cost model
correctly declined a 1.32×-more-expensive member. Combined with the
earlier trial's FLOP-neutral exact lane, neither catopt lane buys
headroom on this architecture; `torch.compile` alone (1.16×) is the
bigger lever. New findings beyond the budget question: `deform_conv2d`
has no lowering binding (a real coverage gap — export emits it, the
sink can't lower it), compositional grafting is unsafe for modules
called at multiple resolutions (e2e reports but doesn't roll back),
and `capture_inputs`' retained clones scale to ~6 GB at 220 M
params — survivable only via filtered block selection on this host.

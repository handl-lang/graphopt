# Turbo bf16 delivery — the honest options, measured

Date: 2026-11-14.  Hardware: RTX 2050 4 GB (**shared GPU** — mint-r3
and another ~0.4–1.2 GiB process resident during parts of this; absolute
ms swing run to run).  torch 2.14.0+cu130, transformers 4.57.1 via
`/tmp/whisper_deps` + `PYTHONPATH`.  Builds on
`whisper_encoder_bench.md` (the 0.42-rel divergence) and
`whisper_production_bench.md` (decode-step numbers).  Probe:
`bench/drivers/probes/whisper_encoder_bench.py` — added an `optx`
mode (`opt <model> <iters> <exclude-substring>`) that filters
`DEFAULT_RULES` by rule name before `search(rules=...)`.

## The question

`attn_sdpa_*` / `sdpa_fold_*` fold spelled attention into `sdpa` ops.
On turbo's 32-layer bf16 encoder the delivered term diverges from the
verify reference by **max|diff| 4.129, rel 0.418** — `verify=True`
(correctly) fails.  fp32 turbo does not fit 4 GB.  What are the honest
delivery paths?

## Mechanism notes (read first)

* **Verify reference is spelled, by construction.**  `lower(verify=True)`
  gates delivered-vs-`sink.lower(result.ir)` — a fresh lowering of the
  *un-optimized exported IR*, no rewrite pass
  (`eqsat:mlopt_orchestrator/optimize.py:1044`).  The dumped turbo IR
  (`/tmp/enc_term_turbo.pkl`) contains **0 `sdpa` ops / 1066 ops**; the
  extracted term contains **32 `sdpa` ops / 906 ops**.  HF eager
  attention exports as `matmul/softmax/matmul`, so the reference is
  spelled attention end to end.
* **Ruleset plumbing has no gap.**  `Optimizer.search` accepts
  `rules=` as a `RuleSet` *or a plain iterable of `Rewrite`*; filtering
  `DEFAULT_RULES` (339 rules) by name substring is all Option A needs.
  `RuleSet.scoped`/`-`/`named` exist (`mlopt_core/laws/ruleset.py`) but
  were not required.  Excluding `*sdpa*` drops 46 rules:
  `attn_sdpa_*` ×8 (carriers `decode_laws.py`), `sdpa_fold_*` ×15 and
  `sdpa_cat_*` ×23 (core `laws/tensor.py` + carriers `om.py`).  The one
  non-`sdpa`-named sdpa-op consumer, `gqa_absorb_repeat`, was *kept* —
  its LHS requires an existing `sdpa` node, so it cannot mint one from
  spelled attention (verified: delivered term has 0 sdpa ops).
* **Caller-stated tolerance is a shipped knob.**  `lower(verify=True,
  rtol=, atol=, task=TaskMetric, task_tol=)`; the verdict and the
  measured `max_abs`/`max_rel` are always recorded in
  `LowerResult.verified`/`stats` — never silent.  Precedent:
  `mlopt_torch/region_fuse.py` ships caller `screen_rtol`/`output_rtol`
  with the measured fused-vs-unfused `max_rel` ledgered in
  `stats["region_fuse"]`.

## Option A — bf16-safe ruleset (exclude sdpa producers): **VERIFIED, SHIPPABLE**

`optx turbo 3 sdpa` → `rules=293`, search 163.3 s:

| measurement | sdpa-folded (default rules) | sdpa-excluded (Option A) |
|---|---|---|
| `lower(verify=True)` | **verified=False** (rel 0.418) | **verified=True** |
| delivered vs HF eager `max|diff|` | 4.129 (rel 0.418) | **0.000e+00 — bit-exact** |
| delivered+graph encoder | ~313 ms | **~450–457 ms** |
| vs HF eager (~500–507 ms) | −38 % (unverifiable) | **−10–11 % (verified)** |
| peak VRAM | 3.11 GiB | 2.93 GiB |

The `delivered` (non-graph) arm OOM'd inside the shared-GPU process
(another tenant held ~0.6–1.2 GiB); `delivered+graph` — the shipping
arm — ran clean.  Speed cost of the bf16-safe ruleset vs the
(unverifiable) folded path: **+~140 ms (+45 %)** on the encoder; still
~10 % ahead of HF eager.

## Option B — align the verify reference through sdpa: **not realizable**

The reference is `sink.lower(result.ir)` on the *exported* IR — spelled
attention, 0 sdpa ops (confirmed on the dumped artifact).  There is no
shipped knob to run the reference through the e-graph or a sdpa
backend; making one would be a substrate change
(`eqsat`/`mlopt-orchestrator`, read-only for this task) and would
change the verify contract for every caller.  Rejected as a delivery
path; noted only because it is the *semantically interesting* fix —
the folded module may be the more accurate one (see Option C).

## Option C — documented-tolerance delivery: **measured, rejected**

Is the 0.418 rel "just bf16 numerics"?  Decisive dtype probe on the
dumped sdpa-folded term (`termtest turbo default fp32`, params cast
in-place bf16→fp32 — the earlier naive cast OOM'd; fixed in the
probe):

| dtype | delivered-vs-IRref max rel |
|---|---|
| bf16 | **0.4181** |
| fp32 | **7.98e-05** |

Random-tensor control (whisper shapes, q prescaled, `scale=1.0`):
spelled-vs-`F.sdpa` agrees to **8.4e-07 rel in fp32**, ~9.7e-03 in
bf16.  So the fold is mathematically exact; the 0.418 is **bf16
kernel-level divergence compounding over 32 layers** (6L fp32 → 4e-5,
12L bf16 → 0.11, 32L bf16 → 0.42 — monotone in depth×dtype).  It is
*genuine* divergence vs the shipped reference, not a fold bug — but
that is exactly what `verify=True` exists to catch.

Would `lower(rtol≈0.5)` be defensible?  **No.**  The measured
`max_rel` would be ~4200× the fp32-equivalent divergence and ~7× the
0.0625 bf16 floor; it would certify a module that is 42 % off the
reference it claims to preserve, on the one model where the check
mattered.  Caller-stated rtol is for *bounded, ledgered* slack
(region_fuse precedent), not for absorbing a depth-compounded kernel
divergence.  (A `task=` metric — e.g. downstream token-id distance —
is the only honest version of this idea, and it redefines "correct"
rather than documenting a pointwise failure; not pursued.)

## Winning option — verified turbo delivery on 4 GB: **YES, via Option A**

| stage | arm | ms |
|---|---|---|
| encoder forward | HF eager bf16 | ~500–507 |
| encoder forward | **delivered+graph, sdpa-excluded ruleset, verify=True** | **~450–457** |
| encoder forward | delivered+graph, default rules (unverifiable) | ~313 |
| decode step | delivered `IRModule`, verify=True | 4.34–4.37 |
| decode step | delivered+`CudaGraphRunner`, verify=True | ~4.3–4.65 (noisy) |
| decode step | HF eager loop | ~5.1–5.4 |

Decode keeps the **default** ruleset — its sdpa folds verify at bf16
(4 layers, single-step verify input; prod-bench `verified=True`,
48/48 token-id match, fresh re-run this session).  Only the 32-layer
encoder needs the exclusion.

Derived e2e (48-token greedy, same methodology as
`whisper_production_bench.md`): **~452 ms encoder + ~211 ms decode +
~20–40 ms loop overhead ≈ 680–700 ms** vs HF eager `generate
(min_new_tokens=48)` ~810 ms — roughly **−14 %**, versus the
~530–560 ms the *unverifiable* folded encoder would have given.
A composed `e2e` probe run was not attempted: holding the delivered
encoder module plus decoder resident for the decode search does not
fit 4 GB alongside verify (the encoder-only verify already peaks
~2.9 GiB); the estimate reuses the prod-bench decode machinery
verbatim.

## Verdict

* **Verified turbo delivery on 4 GB is achievable** — exclude the
  `*sdpa*`-named rules for the encoder search (Option A), keep default
  rules for the decoder.  Cost: encoder ~452 ms vs ~313 ms folded;
  e2e ~680–700 ms vs ~810 ms HF eager.
* Option B is not realizable without a substrate change.
* Option C (rtol≈0.5) is rejected — 0.418 rel is a real, measured,
  depth-compounded bf16 divergence, ~4000× the fp32-equivalent error.
* Flag for law owners (unchanged from `whisper_encoder_bench.md`, now
  sharpened): the `attn_sdpa_*`/`sdpa_fold_*` folds are *exact* — fp32
  rel 8e-5 — but their bf16-vs-spelled divergence compounds past any
  sane tolerance by ~24–32 layers.  Either the folds need a bf16-aware
  admission gate (e.g. a depth/dtype `check` or an auto-cond on the
  verify dtype), or consumers must opt out per-model as done here.

## Reproduce

```sh
export PYTHONPATH=/tmp/whisper_deps \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Option A: encoder, sdpa rules excluded
.venv/bin/python bench/drivers/probes/whisper_encoder_bench.py optx turbo 3 sdpa
# fp32 divergence probe (needs the dumped /tmp/enc_term_turbo.pkl)
.venv/bin/python bench/drivers/probes/whisper_encoder_bench.py termtest turbo default fp32
# decode-step re-measurement (default rules)
.venv/bin/python bench/drivers/probes/whisper_prod_bench.py opt turbo 3 48 64
```

Logs: `/tmp/enc_optx_turbo_sdpa.log`, `/tmp/enc_termtest_fp32e.log`
(failed naive cast), `/tmp/enc_fp32_gpu.log` (fp32 result),
`/tmp/prod_opt_turbo_r2.log`.

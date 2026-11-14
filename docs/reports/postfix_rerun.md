# Post-fix probe re-run — t5 / whisper decode loop / whisper encoder

Date: 2026-11-14.  Re-runs of the round-1/2/3 pipeline probes after the
fix batch landed (`06a8f9b` torch_bridge: `min`/`max` elementwise
binding + `clamp` argpos, `_align_to` int-only + module-dtype keying;
`50b4e9e` `model_corpus` seq-scan `spec_cap` bound; `8589162`
`_GraphReplayModule` copy-direction; `verify_bf16_mem.md` param
sharing + bf16 rtol floor).  Same box as the originals: RTX 2050 4 GB,
torch 2.14.0+cu130, transformers 5.23.0.

**Headline: no regressions.  Every r1 workaround is now unnecessary —
the probes pass with both hacks removed.**  One honest delta: the
bare (int64-mask) bf16 encoder delivery is ~19 % slower than the
float-mask arm (still verified, still ~parity with eager).

## 1. `t5_probe.py` — re-run as-is (workarounds still in place)

`t5_r1.md` baselines in parentheses.

| arm | this run | t5_r1 |
|---|---|---|
| bench fp32 eager / compile (enc) | 4.90 / 4.25 ms | 4.94 / 4.28 |
| bench fp32 eager / compile (dec) | 7.41 / 6.49 ms | 7.52 / 6.56 |
| bench bf16 eager / compile (enc) | 2.09 / 1.22 ms | 2.14 / 1.22 |
| bench bf16 eager / compile (dec) | 4.40 / 1.86 ms | 4.52 / 1.70 |
| `opt encoder fp32` | search 5.3 s, **verified=True**, 4.47 ms, graph `fail=kernel` | 4.8 s, verified, 4.34 ms, `fail=kernel` |
| `opt encoder bf16` | search 5.2 s, **verified=True**, 1.89 ms | 5.1 s, verified, 1.90 ms |
| `opt decoder fp32` | search 6.2 s, **verified=True**, 7.63 ms | 6.5 s, verified, 7.49 ms |

Nothing regressed; timings are within run-to-run noise (the r1 report
itself noted ±30 % GPU variance on shared runs).

**Wall 4 closed:** `model_corpus --module ...:corpus_encoder
--mutants 0 --no-probe --corpus bench --dtype fp32` — which OOM'd
host RAM twice in r1 — now completes: 69 op-tuples / 121 shapes, **58
novel op-tuples / 121 novel shapes**, `site skips over
spec_cap: {seq_sites_over_cap: 10}` (the honest-skip accounting from
`model_corpus_t5_bound.md`), `mutants: 0/0`, verdicts all zero.

## 2. `t5_probe.py` — workarounds removed (`T5_PROBE_BARE=1`)

Added a probe-local flag (no package edits): `T5_PROBE_BARE=1` skips
`_patch_min_max` (wall 2) **and** restores the int64 `attention_mask`
(walls 1+3) in `_feeds`, `corpus_encoder`, `corpus_decoder`.

| arm | bare result | with workarounds |
|---|---|---|
| `opt encoder fp32` | search 5.2 s, **verified=True**, delivered **4.41 ms**, graph `fail=kernel` | 5.3 s, verified, 4.47 ms |
| `opt encoder bf16` | search 6.3 s ×2, **verified=True**, delivered **2.24 / 2.28 ms** | 5.2 s, verified, 1.89 ms |
| `opt decoder fp32` | search 6.6 s, **verified=True**, delivered **7.64 ms** | 6.2 s, verified, 7.63 ms |
| `bench bf16` (eager refs) | enc-eager 2.49, enc-compile 1.22, dec-eager 5.45, dec-compile 1.86 ms | 2.09 / 1.22 / 4.40 / 1.86 |
| `model_corpus` census (bare) | identical census, `seq_sites_over_cap: 10` | identical |

**Both workarounds are confirmed unnecessary:**

- Wall 2 (`torch.min(a,b)` → reduction binding, silent wrong answer →
  OOB gather): with the patch removed the export mints
  `minimum`/`maximum`, verify passes, no device-side assert.  The
  encoder — the arm that hit the INT64_MIN bucket collapse — verifies
  clean in both dtypes.
- Walls 1+3 (int-only feed skipped `_align_to`; fp32 mask upcast bf16
  weights): the all-int64 feed (`input_ids` + `attention_mask`, the
  dtype T5 actually ships) aligns correctly and the bf16 delivery
  stays bf16 — `verified=True` at 2.24 ms, i.e. genuinely bf16 speed,
  not the fp32-upcast 4.6 ms wall-3 signature.

**Honest delta:** bare bf16 encoder delivery is ~2.25 ms vs 1.89 ms
with the float mask (eager bf16 also read slower this run: 2.49 vs
2.09 — partly run variance, but the delivered/int-mask pair is
consistently ~parity with eager rather than the 1.13× the float-mask
arm showed).  The int64 mask likely takes a different
extended-attention-mask path (cast inside the delivered module vs a
pre-cast float operand).  Not a correctness issue — flagging it as a
small perf note, not a regression of the fixes.

## 3. `whisper_decode_loop.py loop` — decode numbers hold

`loop 3 32 40 cpuparams`, whisper-base fp32, 32-token greedy.
`whisper_decode_loop.md` values in parentheses.

| stage | this run | r3 report |
|---|---|---|
| eager loop warm / last-8 | 3.62 / 3.54 ms/tok | 3.59 / 3.56 |
| wrap sanity | max\|diff\| 6.7e-6, argmax equal | ~6e-6, equal |
| search | 7.0 s | 7.5 s |
| lower+verify | 0.1 s, **verified=True** | 0.4 s, verified |
| step: eager-wrap / delivered | 4.09 / 5.37 ms | 4.01 / 5.39 |
| `CudaGraphRunner` | `cuda_graph=True`, 6.06 ms | True, 6.10 ms |
| `torch.compile(wrap)` | 2.97 ms | 2.98 ms |
| loop delivered warm / last-8 | 5.81 / 5.44 ms/tok, **ids 32/32** | 5.99 / 5.55, 32/32 |
| loop delivered+graph | 6.10 ms/tok, **ids 32/32** | 6.16, 32/32 |
| loop torch.compile | 3.12 ms/tok, **ids 32/32** | 3.24, 32/32 |

The `_GraphReplayModule` fix holds: the graph arm replays *correctly*
inside the real loop (32/32 token match — pre-fix it silently replayed
capture inputs) and remains slower than plain delivery, exactly as
r3 measured.  Ordering unchanged: compile > eager > delivered > graph.

## 4. `whisper_probe.py` encoder — whisper-base fp32

`opt encoder base fp32` + `bench base fp32` (`whisper_experiment_r2.md`
values in parentheses).

| arm | this run | r2 |
|---|---|---|
| bench encoder eager / compile | 67.67 / 57.08 ms | 70.9 / 60.0 |
| `opt encoder` search | 3.4 s | 4.2 s |
| lower+verify | 0.3 s, **verified=True** | 0.2 s, verified |
| delivered | **58.28 ms** | 63.4 ms |
| `CudaGraphRunner` | `cuda_graph=True`, replay 58.32 ms | True, 63.4 ms |

Delivered is ~8 % faster than r2's 63.4 ms against a ~4 % faster eager
baseline — the 1.12×→**1.16×** vs-eager ratio is noise-level better,
and delivered ≈ compile (0.98× vs r2's 0.95×).  Graph replay still
ties plain delivery (GEMM-bound, as r2 predicted).

## Verdict

- **No regressions** on any re-run arm; several numbers improved
  slightly (whisper encoder delivered, decode-loop search).
- **t5 walls 1–3 workarounds are dead code in practice**: `T5_PROBE_BARE=1`
  runs the full encoder/decoder/corpus path with zero probe patches —
  int64 mask, unpatched `torch.min`/`max`.  The fixes in `torch_bridge`
  (`06a8f9b`) do what `t5_walls_fixed.md` claimed.
- **Wall 4 is closed**: `model_corpus` intake on t5-small completes in
  seconds with honest `spec_cap` skip accounting (was: earlyoom).
- One note worth keeping: bare-bf16 t5 encoder delivery is a touch
  slower than the float-mask arm (~2.25 vs ~1.9 ms) — the int-mask
  path costs a little inside the delivered module.  Cosmetic, not a
  fix regression.

## Reproduce

```sh
.venv/bin/python bench/drivers/probes/t5_probe.py bench fp32
.venv/bin/python bench/drivers/probes/t5_probe.py bench bf16
.venv/bin/python bench/drivers/probes/t5_probe.py opt encoder fp32 100000 cpuparams
.venv/bin/python bench/drivers/probes/t5_probe.py opt encoder bf16 100000 cpuparams
.venv/bin/python bench/drivers/probes/t5_probe.py opt decoder fp32 100000 cpuparams
T5_PROBE_BARE=1 .venv/bin/python bench/drivers/probes/t5_probe.py opt encoder fp32 100000 cpuparams
T5_PROBE_BARE=1 .venv/bin/python bench/drivers/probes/t5_probe.py opt encoder bf16 100000 cpuparams
T5_PROBE_BARE=1 .venv/bin/python bench/drivers/probes/t5_probe.py opt decoder fp32 100000 cpuparams
T5_PROBE_BARE=1 .venv/bin/python -m mlopt_discovery.model_corpus \
    --module bench.drivers.probes.t5_probe:corpus_encoder \
    --mutants 0 --no-probe --corpus bench --dtype fp32
.venv/bin/python bench/drivers/probes/whisper_decode_loop.py loop 3 32 40 cpuparams
.venv/bin/python bench/drivers/probes/whisper_probe.py bench base fp32
.venv/bin/python bench/drivers/probes/whisper_probe.py opt encoder base fp32
```

Files touched: `bench/drivers/probes/t5_probe.py` (`T5_PROBE_BARE`
env flag only — workarounds retained as the default path), this report.
No package source edited; nothing committed.

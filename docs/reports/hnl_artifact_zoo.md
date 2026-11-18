# The arch-zoo .hnl artifacts — correctness proven, speed blocked

**Question.** The wave-2 arch-zoo sweep exported runnable `.hnl`
programs + safetensors weight sidecars for 10 real architecture specs
(whisper encoder/decoder/full, bert, gpt2, t5, tinyllama) into
`/home/luis/tmp/arch-zoo` (~8.4 GB).  Do the artifacts *run*, and do
they *pay*?

**Answer.** Correctness: **8/10 pass `--verify-run`** against an f64
reference (max_abs ≤ 6.1e-12); the two giants export + `hnl-run`
check-clean but were never verify-run (memory gates).  Speed: **0/8
benchmarked specs are speed-promising** — the `.hnl` runtime is
~195–475× *slower* than torch eager CPU on identical weights and
inputs.  This is a correctness milestone, not a speed win; speed
promotion is **blocked on host-tensor perf work (PL-036)**.

Artifacts live outside git (ephemeral `/tmp`); the pinned inventory —
names, sizes, hashes, verify verdicts, bench folds — is
[`hnl_artifact_manifest.json`](hnl_artifact_manifest.json), generated
by `scripts/zoo_artifact_manifest.py`.  Raw bench numbers:
`/home/luis/tmp/zoo_speed_report.md` + `/home/luis/tmp/zoo_bench/`.

## Per-spec table

| spec | export wall | `--verify-run` | max_abs (values) | hnl ms/call | torch_ir ms | hnl ÷ torch_ir | verdict |
|---|---|---|---|---|---|---|---|
| whisper-tiny-encoder | 9 s | PASS | 3.3e-13 (576k) | 68 520 / 70 234 | 304 / 323 | **~220× slower** | correct, slow |
| whisper-tiny-decoder | 7 s | PASS | 1.6e-13 (1.66M) | 9 262 / 9 314 | 24.1 / 24.9 | **~375× slower** | correct, slow |
| bert-minilm-l6 | 9 s | PASS | 6.4e-16 (384) | 5 695 / 5 920 | 12.2 / 14.7 | **~475× slower** | correct, slow |
| whisper-base-encoder | 8 s | PASS | 1.0e-12 (768k) | 136 588 / 137 256 (n=3*) | 684 / 716 | **~195× slower** | correct, slow |
| t5-small | 57 s | PASS | 6.1e-12 (1.03M) | 7 749 / 8 056 | 33.9 / 34.3 | **~230× slower** | correct, slow |
| gpt2-small | 68 s | PASS | 1.8e-12 (3.22M) | 27 986 / 28 330 | 72.0 / 78.0 | **~370× slower** | correct, slow |
| whisper-tiny-full | 35 s | PASS | 1.6e-13 (1.66M) | 78 117 / 79 688 | 304 / 322 | **~260× slower** | correct, slow |
| whisper-small-encoder | 39 s | PASS | 1.3e-12 (1.15M) | _pending_ | — | — | correct, unbenched |
| whisper-large-v3-turbo-encoder | 286 s | **not run** (2.5 GB sidecar) | — | skipped | — | — | exported, unverified |
| tinyllama-1.1b | 572 s | **not run** (4.4 GB sidecar) | — | skipped | — | — | exported, unverified |

\* whisper-base-encoder: the 40-min per-spec bench timeout killed the
run at round 3/5 — hnl n=3 paired rounds (all arms interleaved),
torch arms n=3; numbers from the run log, no result JSON written.
whisper-small-encoder was never reached.  `torch_opt` and
`torch.compile` arms tracked `torch_ir` within noise on every spec —
the gap is the runtime, not the term.

Export failures outside the 10: `distilbert-base-uncased`,
`qwen2-0.5b-instruct`, `vit-base-p16-224` all exited rc=1 in the
sweep (`missing-status.tsv`); they produced no artifacts.

## Why the runtime is slow (measured, not guessed)

`hnl-run` evaluates `tensor.*` through **host-tensor — dense f64 on
CPU** (`handl-lang/host-tensor`, `src/ops/codec.rs`; no CUDA path
exists).  Two per-eval costs stack:

1. **Weight reload.** The shipped program re-executes every
   `tensor.load` on *each* eval — weights are re-read from the
   safetensors sidecar per call (ranged reads, no cache).  Cost
   decomposition on bert-minilm-l6: binding all 39 sidecar tensors and
   returning one — no compute — costs **~0.92 s of the ~5.9 s** per
   eval.
2. **Dense f64 compute.** The remaining ~5 s is interpreter f64 math
   with no SIMD/f32 path; torch runs the same math in f32 with
   resident weights.

Neither cost exists on the torch path.  The mlopt *term-level* win
measured on the TorchBackend path (1.0–1.2×) does not transfer: the
bottleneck is the interpreter backend, not the rewritten term.

## Verdict

- **Correctness-verified:** 8 specs — the `.hnl` artifact format,
  sidecar channel, and `hnl-run` evaluator reproduce the f64 reference
  within the F32-payload bound (512·eps(f32) ≈ 6.1e-05) on real
  architecture graphs up to whisper-small scale.
- **Speed-promising:** **zero.** Every benched spec is 2–3 orders of
  magnitude behind torch eager CPU.
- **Speed promotion is BLOCKED** on host-tensor perf work tracked
  under **PL-036** (`project/plans/pl-0036-…`): T-043 weight cache,
  T-044 f32 compute path, T-045 f32 verify tolerance, T-046 re-run
  `zoo_bench.py` per phase, T-047 product-proof round 2, T-048 this
  manifest, T-049 this report.

Re-running the bench after each perf phase is T-046; until then this
table is the honest floor.

## Reproducing the manifest

```bash
.venv/bin/python scripts/zoo_artifact_manifest.py \
    --zoo-dir /home/luis/tmp/arch-zoo \
    --bench-dir /home/luis/tmp/zoo_bench \
    --out docs/reports/hnl_artifact_manifest.json
```

Hash policy: full SHA-256 for every `.hnl` and every safetensors file
≤ 64 MiB (all `zoo-inputs-*` feeds); size + SHA-256 of first/last
1 MiB for the large sidecars.  Note the three whisper encoder
`zoo-inputs-*` feeds hash identically — the sweep reused the same
synthetic input tensors across specs.

_Generated 2026-11-18 from `summary.tsv`, `sweep-status.tsv`,
`logs/*-verify-*.log`, and `zoo_bench/result-*.json`.  Hand-check
against the manifest before quoting numbers._

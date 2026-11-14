# Whisper "lift any model" experiment — round 2

Follow-up to `whisper_experiment.md` (r1, kept as the dated record).
Same probe re-run after three walls fell: the bf16 law-bank bug
(eqsat retagged v0.1.0 at `d7e9a57`, graphopt relocked at `c08f34f`),
`_align_to` non-float leaf placement, and `IRModule` plan-liveness
freeing (`550177d`).  Wall 3 — decoder e-graph explosion — was still
open and is owned by a parallel investigation; it was probed here
only to confirm status.

Same hardware/software as r1: RTX 2050 4 GB, torch 2.14.0+cu130,
transformers 5.23.0, synthetic `(1, n_mels, 3000)` log-mel inputs,
forward-only timings (warmup 5 / reps 30).  `uv sync` was re-run
first; the scratch `transformers`/`soundfile` pip installs had to be
redone (`uv sync` prunes non-lockfile packages).

Probe: `bench/drivers/probes/whisper_probe.py` — extended to
`bench|opt <which> <turbo|small|base> <bf16|fp32> [max_enodes]
[cpuparams]`.  The r1 `_align_to` monkeypatch is **removed** — the
upstream fix carries the run.  Two probe-side memory measures were
added after measurement showed they matter on 4 GB:

- `res.eg`/`res.model`/`res.source` are nulled before `lower()` —
  `lower()` reads only `ir`/`term`/`param_values`/`stats`, and the
  provenance handles pin tensors.
- `cpuparams` parks `param_values` on CPU pre-lower; `_align_to`
  moves them back at first eval.  Without it even
  `lower(verify=False)` OOMs on turbo.

## Baselines (r2)

| arm | turbo bf16 | small fp32 | base fp32 |
|---|---|---|---|
| encoder eager | 508.5 ms | 251.1 ms | 70.9 ms |
| encoder `torch.compile` | 435.7 ms | 244.6 ms | 60.0 ms |
| decoder step eager (seq=1) | 9.5 ms | 31.2 ms | 7.8 ms |
| decoder step `torch.compile` | 7.7 ms | 28.4 ms | 6.5 ms |

Consistent with r1 (within noise); this GPU stays compute-bound and
`torch.compile` keeps its ≤15 % encoder edge.

## Pipeline results

### whisper-base fp32 — still the reference pass

- search 4.2 s; `lower(verify=True)` 0.2 s, **verified=True**.
- delivered **63.4 ms** vs eager 70.9 (**1.12×**) and vs compile 60.0
  (**0.95×** — still slower than Inductor).
- `CudaGraphRunner`: `cuda_graph=True`, replay **63.4 ms — no
  change**, as r1 predicted (GEMM-bound encoder; dispatch is already
  amortised).

### whisper-small fp32 — wall #4 fell; full verified pass

r1: `lower(verify=True)` OOM'd and the first forward OOM'd on the
retained eval tape.  r2:

- search 27–29 s; `lower(verify=True)` **0.5 s, verified=True** —
  fits 4 GB now.
- delivered **225–228 ms** vs eager 251.1 (**1.11×**) and vs compile
  244.6 (**1.08× — faster than Inductor**, the first checkpoint
  where delivery beats `torch.compile`).
- `CudaGraphRunner`: `cuda_graph=True`, replay ~227 ms — no change.

### whisper-turbo bf16 — wall #1 fell; verify still OOMs

- `opt.search` now **completes: 163–213 s** (r1 died with
  `TypeError: Got unsupported ScalarType BFloat16` in
  `share_duplicate_param_slices`).  The eqsat pairing fix widens
  bf16 through `.float()` for the dedup signature — exact, since
  every bf16 is representable in fp32.
- `lower(verify=True)` **still OOMs on 4 GB** — even with
  `param_values` parked on CPU.  Verify materialises a fresh
  reference lowering *and* the delivered module simultaneously:
  2 × ~1.6 GB bf16 weights + activations > 3.73 GiB usable.
- `lower(verify=False)` + `cpuparams` **fits and runs**:
  delivered **458.6 ms** vs eager 508.5 (**1.11×**) and vs compile
  435.7 (**0.95×**).  Unverified — the equivalence gate could not
  run on this card.
- `CudaGraphRunner`: `cuda_graph=False, fail=oom` — capture needs
  headroom the resident weights don't leave.

### Decoder (whisper-base fp32) — wall #3 stands

`opt.search(_DecWrap(whisper-base.decoder), (start, ehs),
max_iterations=3)` under `timeout 600`: **killed at 10 min with no
result** — same outcome as r1's >20 min kill.  Encoder search
saturates in seconds; the decoder's cross+self-attention IR does
not.  Status unchanged; investigation owned elsewhere (the untracked
`decoder_search_*.py` probes in `bench/drivers/probes/` are that
work).

## Wall scoreboard

| r1 wall | r2 status |
|---|---|
| bf16 `.numpy()` in law bank | **fixed** — turbo search completes |
| non-float leaf device placement | **fixed** — monkeypatch removed, CUDA delivery works |
| `IRModule` tape retention | **fixed** — small fp32 verifies+delivers in 4 GB |
| decoder e-graph explosion | **open** — >10 min, unchanged |
| verify-time weight duplication (turbo) | **open, narrowed** — only the verify gate OOMs; unverified delivery fits |
| `CudaGraphRunner` on encoder | measured, no gain (GEMM-bound — expected) |

## Verdict for "any model" (r2)

- "Any model" now means: any zoo-scale **or real HF encoder**,
  fp32 *or* bf16, single stateless submodule, ≤~250 M params fp32 /
  ≤~800 M bf16 on 4 GB — with the caveat that the **largest
  checkpoint delivers unverified** (the verify gate's second weight
  copy is the last memory wall).  whisper-small is the biggest
  *verified* delivery and it beats `torch.compile`.
- Speed story unchanged in shape: ~1.1× vs eager, parity-or-better
  vs Inductor on this weak-SM card.  No decoder win is reachable
  until wall #3 falls.

## Reproduce

```sh
uv sync
.venv/bin/pip install transformers soundfile   # scratch deps, pruned by uv sync
.venv/bin/python bench/drivers/probes/whisper_probe.py bench turbo bf16
.venv/bin/python bench/drivers/probes/whisper_probe.py bench small fp32
.venv/bin/python bench/drivers/probes/whisper_probe.py bench base fp32
.venv/bin/python bench/drivers/probes/whisper_probe.py opt encoder base fp32
.venv/bin/python bench/drivers/probes/whisper_probe.py opt encoder small fp32
.venv/bin/python bench/drivers/probes/whisper_probe.py opt encoder turbo bf16 100000 cpuparams
# decoder: timeout 600 .venv/bin/python bench/drivers/probes/whisper_probe.py opt decoder base fp32
```

Raw logs: `/tmp/whisper_r2/{bench_{turbo,small,base},opt_{base,small, turbo}_enc*,opt_base_dec}.log` (ephemeral).

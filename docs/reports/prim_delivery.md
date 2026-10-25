# prim delivery — do the shipped `synop*` folds pay on real wall time?

**Question.** The `prim:synop_*`/`synop2_*`/`synop3_*` folds ship in
core `DEFAULT`; `shipped_primitives.register()` seats each op's torch
denotation on the ambient table. The pay leg so far was cost-model
only — a folded `synop_*` prices as one dispatch against the manual
subgraph's N, and extraction reports the win. Does the delivered
module actually run faster? **Answer: no.** At the measured scales the
fold family loses or ties inside the executor, and every lowered arm
loses end-to-end to the spelled `nn.Module`. The one "delivers" cell
is an executor-routing artifact, not a dispatch win.

Measured with `bench/drivers/probes/prim_delivery_probe.py`
(fp64 CPU, `torch.inference_mode`, interleaved per-iteration samples —
each iteration calls every arm once in rotating order, so external
load spikes hit all arms inside the same window; warmup 60, 400
samples/arm, medians reported). Machine-readable dump:
`--json`; this table is the 2026-10-24 run.

**Measured surface.** The run caught the tree mid-promotion: the
installed core pin already carries the depth-3 tier
(`minted.SYNOP_SHIPPED` = 57 ops — 21 `synop_*` + 2 `synop2_*` + 34
`synop3_*`), so `search` ran the 57-fold bank. The task's bound set is
unchanged: every module still extracts `synop`-family ops and verifies
`pass` on `lower(verify=True)`.

## What a `synop_*` binding actually runs

There is no fused kernel. `register()` seats, per op, a Python closure
(`shipped_primitives._binding`) that re-interprets the definitional
premise's RHS through `genkernel._eval_spec` — a recursive spec-walk
calling the same ambient torch functions, one dict-lookup + Python
call per node, plus a `zip` env build and `_spec_bind` attr threading
per `synop` call. Inside `IRModule`, the trade is: N plan-tape steps
(`_EvalPlan.run_xs` dispatches) → 1 tape step + an N-node spec-walk.
The cost model scores that as N−1 dispatches saved; the wall clock
sees N−1 tape dispatches traded for a *deeper* interpreter (env dict +
attr bind + per-node table lookup). The spec-walk costs more than the
dispatches it removes at every measured size.

## Arms

- `spelled` — the zoo `nn.Module`, fp64, eager (the honest baseline);
- `tape` — `export_to_ir` → `ir_to_torch_module`, no search (pure
  `IRModule` interpreter overhead);
- `noprim` — `default_rules()` minus every `prim:synop*` fold →
  search + lower (the matched ablation — `synop/noprim` is the fold
  family's marginal inside the pipeline);
- `synop` — shipped `default_rules()` + `register()` → search +
  `lower(verify=True)` (the delivered module);
- `*_compiled` — `torch.compile` twins of all four.

`small` is the mint's own probe site (`zoo()` defaults: d=16, t=8,
b=2); `large` is `zoo(d=64, t=32, b=8)`. Verdict band on the marginal:
≤0.95 delivers / ≥1.05 loses / else tie.

## Eager results (median ms; marginal = synop/noprim, e2e = synop/spelled)

### small

| module | spelled | tape | noprim | synop | marginal | e2e | verify | verdict |
|---|---|---|---|---|---|---|---|---|
| GatedDeltaRule | 0.202 | 0.243 | 0.760 † | 0.259 | 0.34× | 1.28× | pass | delivers* |
| GatedLinearAttention | 0.312 | 0.361 | 0.350 | 0.528 | 1.51× | 1.69× | pass | loses |
| MambaSelectiveScan | 0.329 | 0.377 | 0.359 | 0.534 | 1.49× | 1.62× | pass | loses |
| RealGatedLRU | 0.278 | 0.276 | 0.227 | 0.442 | 1.95× | 1.59× | pass | loses |
| HighwayGate | 0.041 | 0.040 | 0.037 † | 0.044 | 1.20× | 1.07× | pass | loses |
| RotaryPair | 2.37 ‡ | 2.11 ‡ | 1.87 ‡ | 1.99 ‡ | 1.06× (min 1.20×) | ‡ | pass | loses (contended) |
| GRUFusedGate | 0.209 | 0.294 | 0.287 † | 0.298 | 1.04× | 1.43× | pass | tie, loses e2e |

### large

| module | spelled | tape | noprim | synop | marginal | e2e | verify | verdict |
|---|---|---|---|---|---|---|---|---|
| GatedDeltaRule | 0.693 | 0.797 | 4.79 † | 0.772 | 0.16× | 1.11× | pass | delivers* |
| GatedLinearAttention | 121.8 | 155.5 | 164.2 | 268.2 | 1.63× | 2.20× | pass | loses |
| MambaSelectiveScan | 22.9 | 22.2 | 24.6 | 27.5 | 1.12× | 1.20× | pass | loses |
| RealGatedLRU | 20.8 | 21.9 | 21.0 | 25.2 | 1.20× | 1.21× | pass | loses |
| HighwayGate | 0.157 | 0.149 | 0.146 † | 0.163 | 1.12× | 1.04× | pass | loses |
| RotaryPair | 4.42 ‡ | 3.82 ‡ | 4.09 ‡ | 3.90 ‡ | 0.96× (min 1.10×) | ‡ | pass | tie (contended) |
| GRUFusedGate | 0.082 | 0.110 | 0.104 † | 0.110 | 1.05× | 1.35× | pass | tie/lose |

† `noprim` routed to `BatchedScanModule`, `synop` to `IRModule` — the
marginal compares different executors, not just the fold.
‡ RotaryPair's medians sit inside multi-ms contention plateaus
(p25–p75 spans 0.19–4.16 ms at small); its marginal-by-min (1.20×/1.10×)
still says lose — the e2e sub-1.0 medians are contention artifacts, not
wins.

## Compiled twins (median ms)

| module @ size | spelled_c | tape_c | noprim_c | synop_c | synop_c / spelled_c |
|---|---|---|---|---|---|
| GatedDeltaRule small | 0.133 | 0.170 | 0.306 | 0.162 | 1.22× |
| GatedLinearAttention small | 0.137 | 0.207 | 0.196 | 0.243 | 1.78× |
| MambaSelectiveScan small | 0.148 | 0.223 | 0.215 | 0.251 | 1.69× |
| RealGatedLRU small | 0.131 | 0.162 | 0.147 | 0.148 | 1.13× |
| HighwayGate small | 0.063 | 0.082 | 0.074 | 0.075 | 1.19× |
| RotaryPair small | 2.05 ‡ | 2.01 ‡ | 2.01 ‡ | 2.20 ‡ | 1.07× ‡ |
| GRUFusedGate small | 0.191 | 0.219 | 0.205 | 0.210 | 1.10× |
| GatedDeltaRule large | 0.354 | 0.419 | 0.854 | 0.401 | 1.13× |
| GatedLinearAttention large | 60.0 | 54.8 | 54.5 | 165.2 | 2.75× |
| MambaSelectiveScan large | 108.3 | 105.2 | 105.0 | 18.9 | 0.17× ‡ |
| RealGatedLRU large | 33.8 | 33.8 | 35.0 | 36.6 | 1.08× |
| HighwayGate large | 0.185 | 0.231 | 0.220 | 0.223 | 1.21× |
| RotaryPair large | 3.81 ‡ | 2.71 ‡ | 3.05 ‡ | 2.78 ‡ | 0.73× ‡ |
| GRUFusedGate large | 0.096 | 0.110 | 0.101 | 0.105 | 1.10× |

## Findings

1. **The folds do not pay on wall clock.** The `synop/noprim` marginal
   is ≥1.04× (i.e. the fold makes the delivered module slower or tied)
   on every cleanly-measured case except the executor-confounded
   GatedDeltaRule. The spec-walk's per-call Python overhead exceeds the
   tape dispatches it saves — the cost model's "one dispatch vs N"
   credits a fused call that doesn't exist in the binding.

2. **End-to-end, every lowered arm loses to the spelled module**
   (1.03–2.2× where measurable). `tape` alone runs 0.95–1.4× spelled —
   `IRModule` interpretation overhead — so the whole lowered family
   starts behind at these scales.

3. **The only "delivers" is an artifact.** On GatedDeltaRule the
   fold-free extraction routes to `BatchedScanModule`, which is 2.9–6.2×
   *slower* than the generic IRModule on this workload (0.76 vs 0.24 ms
   small; 4.79 vs 0.80 ms large). The `prim:synop*` folds "win" there by
   keeping the term out of the carrier's `accepts` probe — preventing a
   bad executor choice, not by dispatch reduction. (Worth a separate
   look: a carrier executor 3–6× slower than the interpreter it
   replaces is a routing bug-shaped result on its own.)

4. **Compile neither needs nor recovers the fold.** `spelled_compiled`
   is the fastest arm in 10/14 cells and the floor everywhere clean —
   compile alone beats the best folded-lowered arm by ~2–5× at small
   size. The compiled twins converge to within ~1.1–1.8× of each other
   (dynamo inlines the spec-walk), but `synop_compiled` stays
   *behind* `spelled_compiled` wherever the trace isn't contended —
   the fold's runtime artifact never turns into compiled win. Large-size
   inductor numbers are contention-dominated (MambaSelectiveScan's
   0.17× is inductor variance: its `spelled_compiled` regressed to
   108 ms against 22.9 eager — compile *hurt* there).

5. **Saturation fires broadly, extraction commits narrowly.** The bank
   fires 20–60 `prim:*` times per SSM module, yet extracted terms carry
   1–8 synop ops; the folded term is strictly smaller (e.g.
   MambaSelectiveScan 609→429 ops small, 8577→5937 large) — the
   op-count reduction is real, the wall-time pay is not.

## Where the win would have to come from

A fold can only pay if the bound op executes *fewer* machine ops than
the subgraph it replaces: a codegen'd/minted handler kernel (the
`genkernel` path has `compile_kernels` — `torch.compile` per binding —
which `register()` does **not** use), or a carrier routing the whole
region to a batched executor. As shipped, the synop denotation is a
correctness-preserving term compression — it buys a smaller extracted
term and a provenance handle for downstream machinery, not a faster
forward pass. Reporting the 0.5–0.8× cost-model deltas as a
performance win is not supported by these measurements.

# The measured pay leg — the prim family does not pay

**Date:** 2026-10-25, re-run 2026-11-13 (pinned-executor arm) ·
**Script:** `bench/referees/pay_wall.py` ·
**Store:** `mlopt_core.laws.store` measurements (optional `--store`) ·
**Seed:** 0 · **Machine:** i5-12500H / RTX 2050, shared box (load 5–7)

```bash
.venv/bin/python -m bench.referees.pay_wall --iters 100 --warmup 30 \
    --json docs/reports/pay_measured.json
```

> **Verdict: measured, and it loses — with no confounded cells.** Of
> 14 module×size cells, **13 lose, 1 ties, 0 confounded** and **none
> delivers**. Every fold op that fires on this corpus — 9 shipped
> folds — loses its own marginal, worst 2.93×.  The 2026-10-25 run
> left 6 cells `confounded` (the two delivered arms routed to different
> executors); the 2026-11-13 re-run adds a pinned-executor arm
> (`noprim_ir`) that holds the executor fixed, and every one of those
> cells resolves to **loses** — the earlier GatedDeltaRule "delivers"
> was entirely the `BatchedScanModule` routing artifact
> (`docs/reports/t029_executor_confound.md`).
> `docs/reports/prim_delivery.md` said this from the probe; this is the
> **gate** that makes it admissible evidence.

## Why a second pay leg

The gauntlet's `pay` leg prices a claim in the extraction cost model: a
`synop_*` call costs one dispatch against the spelled subgraph's N. That
model is right about the *term* and wrong about the *executor* —
`shipped_primitives.register()` seats a Python spec-walk, so the runtime
trades N−1 tape dispatches for a recursive interpreter over N nodes.
`pay_wall` times the delivered module instead. Arms (eager, fp64,
median of 100 interleaved samples): `spelled` (the zoo `nn.Module`),
`tape` (export + `IRModule`, no search), `noprim` (shipped rules minus
the `prim:synop*` folds, delivered), `noprim_ir` (the *same* fold-free
term forced through `sink.lower` — the generic `IRModule` path —
bypassing carrier executor routing), and `synop` (shipped rules +
`register()`, delivered).

Three ratios: `marginal = synop/noprim` (as delivered — includes any
executor routing delta); `pay = synop/noprim_ir` (the fold's own
effect with the executor pinned — the verdict basis); `e2e =
synop/spelled` (what a caller receives). Verdict band on `pay`:
≤0.95 delivers / ≥1.05 loses / else ties. A cell is `confounded` only
when the delivered arms route to different executors AND no valid
pinned comparison exists — it then counts as pay evidence neither
way. `small` is the mint's own probe site (`zoo()` defaults: d=16,
t=8, b=2); `large` is `zoo(d=64, t=32, b=8)`.

## The cells — eager, fp64, median of 100 interleaved samples

| module | size | spelled | noprim | noprim_ir | synop | marginal | pay | e2e | verdict |
|---|---|---|---|---|---|---|---|---|---|
| GatedDeltaRule | small | 0.10 | 0.42 | 0.12 | 0.15 | 0.37 | **1.23** | 1.49 | loses |
| GatedDeltaRule | large | 0.26 | 0.92 | 0.28 | 0.30 | 0.33 | **1.08** | 1.15 | loses |
| GatedLinearAttention | small | 0.15 | 0.18 | 0.18 | 0.28 | 1.53 | **1.52** | 1.91 | loses |
| GatedLinearAttention | large | 2.31 | 7.91 | 7.77 | 8.12 | 1.03 | 1.04 | 3.51 | ties |
| MambaSelectiveScan | small | 0.18 | 0.22 | 0.22 | 0.33 | 1.55 | **1.54** | 1.89 | loses |
| MambaSelectiveScan | large | 2.62 | 3.17 | 3.10 | 4.08 | 1.29 | **1.31** | 1.56 | loses |
| RealGatedLRU | small | 0.15 | 0.13 | 0.13 | 0.27 | 2.09 | **2.07** | 1.81 | loses |
| RealGatedLRU | large | 0.77 | 0.71 | 0.71 | 2.09 | 2.95 | **2.93** | 2.71 | loses |
| HighwayGate | small | 0.02 | 0.02 | 0.02 | 0.02 | 1.22 | **1.33** | 1.37 | loses |
| HighwayGate | large | 0.02 | 0.02 | 0.02 | 0.03 | 1.20 | **1.30** | 1.34 | loses |
| RotaryPair | small | 0.08 | 0.05 | 0.05 | 0.07 | 1.24 | **1.24** | 0.91 | loses |
| RotaryPair | large | 0.24 | 0.20 | 0.20 | 0.22 | 1.09 | **1.11** | 0.91 | loses |
| GRUFusedGate | small | 0.03 | 0.05 | 0.05 | 0.05 | 1.10 | **1.14** | 1.85 | loses |
| GRUFusedGate | large | 0.04 | 0.05 | 0.05 | 0.06 | 1.10 | **1.13** | 1.63 | loses |

`marginal = synop / noprim` (as delivered — includes any executor
routing delta); `pay = synop / noprim_ir` (the fold's own effect with
the executor pinned to `IRModule` — the verdict basis);
`e2e = synop / spelled` (what a caller receives). On the six cells
where `noprim` delivered to `BatchedScanModule` (GatedDeltaRule,
HighwayGate, GRUFusedGate × both sizes) `marginal` and `pay` diverge
sharply — e.g. GatedDeltaRule small reads 0.37 delivered but 1.23
pinned: the "win" was the carrier executor, not the fold.

## Per fold

Nine distinct shipped ops fire on this corpus; all nine lose.

| fold op | worst | mean | verdict |
|---|---|---|---|
| `synop_4c657cb2c0` | 1.54 | 1.35 | loses |
| `synop_bb8c54c201` | 2.93 | 1.46 | loses |
| `synop_5e58f26875` | 2.93 | 2.50 | loses |
| `synop2_35bbd03301` | 2.93 | 2.50 | loses |
| `synop3_0ca31bf1e5` | 2.93 | 2.93 | loses |
| `synop3_0fa19d1ff8` | 2.93 | 2.50 | loses |
| `synop3_21b9729f32` | 2.93 | 2.50 | loses |
| `synop3_24c5153e5e` | 2.93 | 2.93 | loses |
| `synop3_4cdc06b88f` | 2.93 | 2.93 | loses |

## What this settles, and what it does not

- **Settled:** the shipped family's *pay* claim is false on measured
  wall clock. The cost-model number (0.5–0.8×) is not delivered, and no
  cell delivers under the pinned-executor rule — including the six
  cells the 2026-10-25 run could only mark `confounded`.
- **Settled:** the family's *originality* claim is untouched. These ops
  are still machine-original by construction
  (`docs/reports/primitives_finds.md`); they are a vocabulary result,
  not a speed result.
- **Not settled:** the remaining **49 shipped folds are unmeasured** —
  they do not fire on this corpus. Unmeasured is not evidence; a fold
  that never fires is a vocabulary entry with a price tag, not a pay
  claim.
- **Not attempted here:** a real fused body (triton via `genkernel`, or
  Inductor over a specialised body). The probe's compiled twins already
  lose to the spelled module compiled, which is the cheap version of
  that experiment and it says no at these sizes.

**Recommendation:** demote the family's pay claim — keep `MACHINE_SHIPPED`
as *provenance* (they are admitted, certified objects), and record
`pay_wall` beside each op so no future summary says "pays" without the
number. A demotion to `MACHINE_STORED` is a policy call, not a
measurement; the numbers to make it are in
`docs/reports/pay_measured.json`.

## G2b — a real fused body *does* beat Inductor (different mechanism)

The `synop` family cannot be fixed by re-binding it: the loss is the
spec-walk, and the probe's compiled twins already lose.  But the *other*
minting channel — `genkernel`, which emits `triton.jit` for a claimed
handler — answers the question the family raises.  Same box, same day:

```bash
.venv/bin/python -m mlopt_discovery.play --domain gen --probe \
    --train-cases 4 --eval-cases 2 --budget 8 --seed 0
```

| case | eager µs | `torch.compile` µs | generated kernel µs | vs eager | **vs compiled** | verified |
|---|---|---|---|---|---|---|
| `gen0:chainfuse` (6-op chain) | 317.3 | 244.7 | **85.2** | 3.72× | **2.87×** | yes (1.7e-5) |
| `gen1:mulsoftplus` (2-op) | 29.6 | 38.3 | **20.0** | 1.48× | **1.92×** | yes (9.5e-7) |

So the answer is not "folds don't pay" — it is **"a spec-walk doesn't
pay; a generated kernel does, by 2.87× over Inductor on a 6-op chain."**
The two arms are different mechanisms and must not be quoted as one:
`synop_*` is a *vocabulary* result (originality), `gen_*` is a
*delivery* result (speed).

Caveats, stated: 2 cases, `--train-cases 4 --eval-cases 2 --budget 8`,
one device (RTX 2050), `cost: nan` in the probe record.  The 2-op case
wins here where the earlier TODO note measured ~0.86× for 2-op
fusions — the probe's own `compiled_us` baseline is what moved.

## Recording the numbers beside the records

```bash
# copy the shipped store, then record (the shipped file stays read-only)
cp ../eqsat/packages/mlopt-core/src/mlopt_core/laws/data/minted.db /tmp/minted.db
.venv/bin/python -m bench.referees.pay_wall --modules RealGatedLRU \
    --sizes large --store /tmp/minted.db --run 2026-10-25
sqlite3 /tmp/minted.db "select name, arm, round(value,2) from measurements limit 5"
```

Each fold gets one row per module it binds:
`measurements(name=synop_…, metric='pay_wall', arm='RealGatedLRU@large',
value=2.93)` — the stored `value` is the pinned-executor `pay` ratio,
never a confounded marginal.  The store write path is `mlopt_core.laws.store.record_measurements`
(eqsat `1633ad1`), so graphopt needs its `mlopt-core` pin refreshed
(`uv lock --upgrade-package mlopt-core`) before `--store` runs without a
`PYTHONPATH` override.

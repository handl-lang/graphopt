# The measured pay leg — the prim family does not pay

**Date:** 2026-10-25 · **Script:** `bench/referees/pay_wall.py` (new) ·
**Store:** `mlopt_core.laws.store` measurements (optional `--store`) ·
**Seed:** 0 · **Machine:** i5-12500H / RTX 2050, shared box (load 5–7)

```bash
.venv/bin/python -m bench.referees.pay_wall --iters 100 --warmup 30 \
    --json docs/reports/pay_measured.json
```

> **Verdict: measured, and it loses.** Of 14 module×size cells, **7 lose,
> 1 ties, 6 are confounded** (the two arms route to different executors)
> and **none delivers honestly**. Every fold op that fires on this
> corpus — 9 of the 57 shipped — loses its own marginal, worst 2.66×.
> `docs/reports/prim_delivery.md` said this from the probe; this is the
> **gate** that makes it admissible evidence.

## Why a second pay leg

The gauntlet's `pay` leg prices a claim in the extraction cost model: a
`synop_*` call costs one dispatch against the spelled subgraph's N. That
model is right about the *term* and wrong about the *executor* —
`shipped_primitives.register()` seats a Python spec-walk, so the runtime
trades N−1 tape dispatches for a recursive interpreter over N nodes.
`pay_wall` times the delivered module instead. Verdict band:
`marginal ≤ 0.95` delivers, `≥ 1.05` loses, else ties; a cell whose two
arms route to different executors is **confounded** and cannot count as
pay either way.

## The cells — eager, fp64, median of 100 interleaved samples

| module | size | spelled | noprim | synop | marginal | e2e | verdict |
|---|---|---|---|---|---|---|---|
| GatedDeltaRule | small | 0.09 | 0.36 | 0.13 | 0.35 | 1.41 | confounded |
| GatedDeltaRule | large | 0.27 | 1.56 | 0.31 | 0.20 | 1.13 | confounded |
| GatedLinearAttention | small | 0.15 | 0.19 | 0.28 | **1.49** | 1.85 | loses |
| GatedLinearAttention | large | 2.04 | 6.11 | 6.06 | 0.99 | **2.97** | ties |
| MambaSelectiveScan | small | 0.15 | 0.19 | 0.28 | **1.48** | 1.84 | loses |
| MambaSelectiveScan | large | 1.19 | 1.72 | 1.98 | **1.15** | 1.66 | loses |
| RealGatedLRU | small | 0.13 | 0.12 | 0.23 | **1.98** | 1.70 | loses |
| RealGatedLRU | large | 2.38 | 1.99 | 5.29 | **2.66** | 2.22 | loses |
| HighwayGate | small | 0.02 | 0.02 | 0.02 | 1.16 | 1.27 | confounded |
| HighwayGate | large | 0.02 | 0.02 | 0.03 | 1.14 | 1.24 | confounded |
| RotaryPair | small | 0.08 | 0.06 | 0.07 | **1.21** | 0.90 | loses |
| RotaryPair | large | 0.23 | 0.20 | 0.21 | **1.06** | 0.91 | loses |
| GRUFusedGate | small | 0.03 | 0.05 | 0.05 | 1.07 | 1.80 | confounded |
| GRUFusedGate | large | 0.04 | 0.05 | 0.06 | 1.07 | 1.59 | confounded |

`marginal = synop / noprim` (the fold's own effect, same executor);
`e2e = synop / spelled` (what a caller receives). A `confounded` cell is
one where `synop` and `noprim` route to different executors — the old
probe's one "delivers" cell (GatedDeltaRule) is exactly this artifact,
not a dispatch win.

## Per fold

Nine distinct shipped ops fire on this corpus; all nine lose.

| fold op | worst | mean | verdict |
|---|---|---|---|
| `synop_4c657cb2c0` | 1.49 | 1.28 | loses |
| `synop_bb8c54c201` | 2.66 | 1.73 | loses |
| `synop_5e58f26875` | 2.66 | 2.32 | loses |
| `synop2_35bbd03301` | 2.66 | 2.32 | loses |
| `synop3_0ca31bf1e5` | 2.66 | 2.66 | loses |
| `synop3_0fa19d1ff8` | 2.66 | 2.32 | loses |
| `synop3_21b9729f32` | 2.66 | 2.32 | loses |
| `synop3_24c5153e5e` | 2.66 | 2.66 | loses |
| `synop3_4cdc06b88f` | 2.66 | 2.66 | loses |

## What this settles, and what it does not

- **Settled:** the shipped family's *pay* claim is false on measured
  wall clock. The cost-model number (0.5–0.8×) is not delivered, and no
  cell delivers honestly under the same-executor rule.
- **Settled:** the family's *originality* claim is untouched. These ops
  are still machine-original by construction
  (`docs/reports/primitives_finds.md`); they are a vocabulary result,
  not a speed result.
- **Not settled:** the remaining **48 shipped folds are unmeasured** —
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
value=2.73)`.  Re-measuring updates in place; nothing about the record
changes.  The store write path is `mlopt_core.laws.store.record_measurements`
(eqsat `1633ad1`), so graphopt needs its `mlopt-core` pin refreshed
(`uv lock --upgrade-package mlopt-core`) before `--store` runs without a
`PYTHONPATH` override.

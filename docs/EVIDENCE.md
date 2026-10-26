# Evidence table — one claim, one number, one artifact

Every claim this project makes, the measurement that supports it, and the
file it lives in. If a row is missing a number, the claim is not made.

**Date:** 2026-10-26 · all numbers on the dev box (i5-12500H / RTX 2050),
seeds recorded per artifact.

## The claims

| # | claim | number | artifact | re-run |
|---|---|---|---|---|
| 1 | the machine **invents new operations**, and they are original | **21/21** machine-original on the `zoo-prim` corpus, then 48/48 deeper; 57 shipped ops | `docs/reports/primitives_finds.md` | `.venv/bin/python -m bench.drivers.overnight --corpus zoo-prim` |
| 2 | **composition inside a fixed bank is not original** | **0** of 97 admitted codegen rules were machine-original | `kernelopt/docs/results_overnight_kernel_v2.md` (private) | kernelopt `bench/discovery/kernel.py` |
| 3 | **delivered as an interpreter, the invented rules lose** | 14 cells: **7 lose · 1 tie · 6 confounded · 0 wins**; worst **2.66×**; 48 folds unmeasured | `docs/reports/pay_measured.md` + `pay_coverage.md` | `.venv/bin/python -m bench.referees.pay_wall` |
| 4 | **a kernel body wins at op scale** | **0.365–0.392× the spec-walk — 2.6× faster**, fp32 *and* fp64, 6 ops verified, at **1 M elements** | `docs/reports/prim_kernel.md` (+ `prim_kernel_fp64.json`) | `.venv/bin/python -m bench.drivers.probes.prim_kernel_probe --dtype float64` |
| 4b | **and does not win at module scale, in any shape** | large (d64 t32 b8): **3.17** spec-walk vs **3.34** kernel; wide (d256 t256 b16, ~1 M elements): **3.23** vs **2.67** — the kernel body helps ~17 %, the fold still loses | `docs/reports/prim_kernel.md` §module level | `.venv/bin/python -m bench.drivers.probes.prim_kernel_module --size wide` |
| 5 | **a generated kernel beats `torch.compile` — on a synthetic chain** | 6-op chain **85 µs vs 245 µs (2.87×)**, 2-op 1.92× | `docs/reports/pay_measured.md` §G2b | `python -m mlopt_discovery.play --domain gen --probe` |
| 6 | **growth helps the same player, but only if it pays** | paying growth **+0.066** (5/5 seeds ≥ +0.036); unpaid growth **−0.023** (5/5 ≤ +0.008) | `docs/results_bank_growth.md` | `.venv/bin/python -m bench.drivers.policy.bank_growth --drop ...` |
| 7 | **minting during play on the toy board yields nothing** | 1 minted, **0 admitted** (no pay) | `docs/results_bank_growth.md` §Step 3 | `.venv/bin/python -m bench.drivers.policy.play_mint` |
| 8 | **the referee has teeth** | 212 bad accepts refused, 4 false twins rejected, 1 fabricated speedup caught, 1 real bug found | `README.md`, `docs/mechanism.md` | `docs/results_*` |
| 9 | **a chain-level kernel pays in a real module only where the module is dispatch-bound** | 5 zoo workloads at d64 t32 b8, 3 rounds each: `TimeCondConv` **1.2×–1.3× eager** (18 rounds, 1.15–1.42, never below 1; 22 vs 34 kernels), `FNOBlock` **1.77× `torch.compile`**, `GatedAttentionUnit` **0.95×**, `MLPMixerBlock` **0.98×**, `BlockSparseAttention` **0.96×** — all `max_abs 0.0` | `docs/reports/zoo_chain.md` (+ `zoo_chain.json`) | `.venv/bin/python -m bench.drivers.probes.zoo_chain --rounds 3` |

## The one gap, stated

| gap | state | why it matters |
|---|---|---|
| kernel win at module scale | **measured: yes, once, and narrowly** — `TimeCondConv` 1.2–1.3× eager (dispatch-bound: 22 vs 34 kernels), `FNOBlock` 1.77× `torch.compile`. Elsewhere 0.95–0.98× eager; the folds stay a per-module loss (2.67 marginal at ~1 M elements, best case). | the win appears where the module is **dispatch-bound**, not where it is kernel-bound — which is also why the fp64 folds lose |
| compile twins on CUDA | **unblocked 2026-10-26** — the cause was four zoo workloads minting a CPU tensor in `forward`, not the bridge. `GatedLinearAttention` and `MambaSelectiveScan` now measure at `large` fp32 CUDA: marginal **1.37–1.38×** (fold loses), verified equal | the family verdict now rests on three module families with kernels bound, not one |
| corpus coverage | **5 of 68 zoo workloads** at wide shapes — `torch.compile` as baseline makes each row cost tens of seconds, and `ConvNeXtStage` / `MBConvSEBlock` did not finish in budget | the table is a sample; say so |
| module-level timing protocol | **fixed** — one round is not evidence: a single sample of `TimeCondConv` read 0.42× while eighteen clean round-readings of the same code read 1.15–1.42×. The driver now runs 3 rounds and reports the median plus spread. | a ratio computed from two differently-timed arms can be wrong by 3×, in the direction that flatters whichever arm was measured first |

## What is deliberately *not* claimed

- not 127 laws — 127 ledger *names* (13 promoted + 57 folds + 57 premises)
- not a speed result for the folds as shipped (claim 3 is the opposite)
- not "the bank grows, therefore the player improves" — only *iff it pays*
- not scale: whole models run at ~parity; the 7–11× gap is codegen
  (`docs/results_wholemodel_gap.md`)

## How the ledger stays honest

- `provenance.pay_status()` answers per fold: `measured-slow` (9) /
  `unmeasured` (48) / `n/a` — pinned by
  `eqsat/tests/test_provenance_pay.py`, so no summary can call an
  unmeasured fold paying.
- the minted bank is a SQLite store (`laws/data/minted.db`) with a golden
  test on every record, so the bank cannot drift from what was measured.

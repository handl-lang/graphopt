# T-035 / T-036 — the two "pay-declined near the bar" agenda rows, judged

**Date:** 2026-11-12 · **Scope:** agenda proposals T-035
(`grammar:sub_add_factor`) and T-036
(`syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_{6,10}@RMSNormPerHead`)
from `docs/reports/agenda.jsonl` rows 7–9.

**Verdict up front: both declines are correct, and neither is actually
"near the profitability bar."** T-035 pays *literally nothing* (every
measured axis is exactly 1.000× — a pure reassociation). T-036 never
reached the price question: the pay leg refused it on **correctness**
(`verify_fail=4`), and its headline `exec=0.914×` is measured partly
over rewrites that do not verify. The agenda signal that produced both
rows has a filter bug — it does not check `verify_fail` or require
`paid=0`, contradicting its own docstring.

## The bar, as implemented

The overnight gauntlet's `pay` leg is
`bench/referees/components.py::_pay` (lines 474–523):

- `paid` counts changed cases where
  `dag_cost(best, cost_fn) < dag_cost(case.term, cost_fn)` — strict
  improvement under the extraction cost model (default
  `launch_aware_cost`, lines 357–359, 505).
- The leg passes iff `fires > 0 and paid > 0 and vfail == 0`
  (line 514). **There is no ratio threshold inside the leg** — the
  `ScoreGate("launch", 1.0)` in `bench/referees/pay/__init__.py` is a
  separate value-function gate on the measured `launch` axis.
- The `{axis}={v:.3f}x` ratios (`count/flops/launch/exec/params`)
  are collected over **every changed case, verified or not**
  (lines 507–510). A verify-failed rewrite still contributes its cost
  ratio to the headline numbers.

So "the bar" is binary: at least one paying, verifying changed case.
`exec=0.914×` is not "9% faster but below threshold" — the leg has no
percentage threshold at all; the decline came from `vfail != 0`.

## T-035 — `grammar:sub_add_factor`: parity, not near-miss

Ledger: `docs/reports/results_overnight_real_preview.json:757` (doc
`results_overnight_real_preview.md:61`).

```
pay: fires=4 changed=4 paid=0 verify_fail=0 cases=40
    | count=1.000x exec=1.000x flops=1.000x launch=1.000x
```

The law is `(x - y) - z → x - (y + z)`
(`packages/mlopt-discovery/src/mlopt_discovery/lawdata/laws/terms.py:184-198`;
probe module `_SubAddFactorHead`,
`packages/mlopt-discovery/src/mlopt_discovery/intake/models.py:2126-2137`).
Both sides are exactly two elementwise ops: LHS `sub(sub(x,y),z)`,
RHS `sub(x, add(y,z))`. Every cost axis counts ops, so all four ratios
are identically 1.000 — `paid=0` is structural, not noise. The
`arena` cell agrees (`base_cost == cand_cost == 2.0`, verdict false).

- **Is the bar calibrated right?** The question is moot — there is no
  delta to price. `exec=1.0×` is exact parity; the agenda's "near the
  bar" band `[0.7, 1.3]` (`agenda_miner.py:~318`) treats parity as
  proximity, which is a category error for a reassociation law.
- **Other shapes/corpus?** No. Node count is shape-independent; no
  corpus makes a 2-op → 2-op rewrite cheaper under this cost model.
- **Cheap lever?** Only as a *composition enabler*: the factored form
  `x - (y+z)` would pay if a companion law consumed it (e.g. an
  `add`-inside-`sub` factorization or a `neg`-distribution chain —
  `neg_factor`/`neg_distribute` exist in the same table,
  `terms.py:160-183`, but nothing targets `sub(x, add(y,z))`
  specifically; grep of the law tables finds no such LHS). In an
  e-graph both forms coexist at equal cost, so the law only changes
  the extracted optimum when composed with other rules — and the
  `--compose` arm that would test that runs on zoo-real module-bound
  candidates, not the grammar family. Speculative, low expected value.
  **The measured decline stands.**

## T-036 — `syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_{6,10}@RMSNormPerHead`: a correctness decline wearing a speed number

Ledgers: `docs/reports/results_overnight_deep.json` (the `_w_10`
row) and `docs/reports/results_overnight_real_full.json` (the `_w_6`
row; identical metrics — same rule re-minted under a dedup suffix,
see `overnight.py:805-811`). Earlier versions show the same signature:
`results_overnight_real_v3.md:126` and `results_overnight_real_v4.md:130`.

```
pay: fires=5 changed=5 paid=4 verify_fail=4 cases=57
    | count=1.000x exec=0.914x flops=0.900x launch=0.900x params=1.000x
```

The pay leg failed on `vfail == 0` (4 of 5 changed cases failed sink
verification), **not** on price — `paid=4` actually satisfies the
paying-case requirement. The `exec=0.914× / flops=0.900×` headline is
the mean over all 5 changed cases *including the 4 that do not
verify* (`components.py:507-510`), so even the apparent ~10% win is
partially measured on numerically wrong rewrites.

**Why it fails is structural, not noise.** The composite chains two
`mixed:` census-mixed-view laws (`pipeline.py:380-460`,
`_one_view_family`/`_two_view_family`; census driver
`learned_propose.py::_census_mixed_naturality`, wired at
`meta_game.py:1506,1901`):

- `mixed:mul_reshape_r_id` — strip a `reshape_r` view off `mul`'s
  right operand. Its own ledger row shows the underlying equality is
  **false on part of its accepted region**: oracle
  `real: 4eq/3ne/1rerr`, `synth: 60eq/84ne/63rerr`
  (`results_overnight_real_full.json`, candidates list).
- `mixed:pow_reshape_l_w` — push `pow` through a left `reshape`.
  Also oracle-refused: `synth: 155eq/14ne/73rerr`.

The composite's guard is the `_compose_guards` premise transport
(`overnight.py:765-860`), which evidently does not restrict the
accepted region enough: the pay probe's minted instances land on
sites where the composed rewrite computes different values. Verify is
`max_rel < 1e-4` in fp64 through `TorchSink`
(`mlopt_torch/report.py:143-165`); reshape/mul/pow are exact ops, so
a *correct* rewrite verifies at ~1e-15 — a 1e-4 failure is a real
numerical divergence, not a lowering artifact. The identical
`verify_fail=4` across independent runs (v3-era md, `deep`, `full`)
confirms determinism. This is unrelated to the executor-confound
issue in `docs/reports/pay_measured.md` — that confound is about
wall-clock routing in `bench.referees.pay_wall` (the synop fold
family); this leg is the symbolic cost-model probe, and its verify
sub-gate is doing exactly its job.

- **Is the bar calibrated right?** Yes — but the row never tested the
  bar. `verify_fail>0 ⇒ pay fails` is the correct refusal for a law
  whose accepted region contains counterexamples.
- **Would it pay on other shapes/corpus?** Only on the subdomain where
  the rewrite is actually true. The symbolic delta is real (~one view
  op removed: `flops/launch 0.9×`, `exec 0.914×`), and the single real
  match is `RMSNormPerHead` — thin coverage even if repaired. A
  *guard-fixed* version might pay on its true subdomain; the minted
  rule as admitted-region-defined does not.
- **Cheap lever?** Guard repair, not re-pricing: tighten the
  composite's `check` (or fix the parent `mixed:` view laws' missing
  shape/broadcast premises — both parents already carry measured
  counterexamples, so the whole `census-mixed-view` family's guard
  story is suspect) and re-run the pay leg. If the surviving accepted
  sites still show ~0.9×, it becomes a genuine admit candidate. That
  is a correctness fix upstream of pricing — the agenda's suggested
  lever ("re-price or re-codegen") points at the wrong stage.

## The miner bug behind both rows

`bench/drivers/agenda/agenda_miner.py`:

- The module docstring (lines 16–20) defines the signal as
  "`changed>0, paid=0` … declined at pay after changing the graph".
- The actual filter (lines ~299–305) is
  `pay is not None and not pay.passed and pay.changed > 0` — it never
  checks `paid` or `verify_fail`, both of which it already parses
  (lines 220–231).
- "Near the bar" is `0.7 <= exec <= 1.3` (line ~318): parity (1.0)
  and a verify-failed 0.914 both qualify.

Consequences: T-036 is a **verify-decline misfiled as a pricing
near-miss** (its `verify_fail` is parsed but dropped from the
proposal's `metrics`, so downstream readers can't see it), and T-035
is a parity rewrite that can never satisfy `paid>0` under any cost
axis — arguably it shouldn't be proposed at all, or only under a
"composition enabler" framing. Suggested fix (not applied): split the
signal — `ledger_pay_decline` should require `paid == 0 and
verify_fail == 0` per its docstring, and a new
`ledger_verify_decline` signal should catch `verify_fail > 0` rows as
guard-repair proposals with the verify count in evidence.

## Bottom line

| row | measured cause of decline | truly near a bar? | honest answer |
|---|---|---|---|
| T-035 `sub_add_factor` | `paid=0`: 2 ops → 2 ops, all axes 1.000× | no — exact parity | decline correct; only value would be a companion law, speculative |
| T-036 `syn_mixed:…pow_reshape_l_w_*@RMSNormPerHead` | `verify_fail=4`: accepted region contains numerically-false sites | no — refused on correctness before pricing | decline correct; real lever is guard repair on the `mixed:` parents, then re-measure |

No re-pricing, threshold change, or executor fix would flip either
row. The only live follow-up is upstream: the `census-mixed-view`
family's guards admit counterexample sites (both parents show oracle
`ne`/`rerr` mass), which is what actually killed T-036.

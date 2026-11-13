# T-037 — frontier re-sweep (graph arm) + the BatchedScanModule routing bug

**Date:** 2026-11-14 · **Scope:** the follow-up to the T-032/33/34
stale-frontier finding and the T-029 executor-confound residual.

Two work items: (1) re-sweep the frontier's affected workloads now
that `gen_probe` carries the CUDA-graph arm (`0fc704f`) and the
seq/seqm tiers have landed; (2) root-cause why `BatchedScanModule`
runs ~3.5× slower than `IRModule` on the fold-free GatedDeltaRule
term (T-029 report, "a routing bug-shaped result").

## 1. Frontier re-sweep — done, both dtypes, graph arm is first-class

### What was stale

`frontier.json`/`frontier_fp32.json` join `zoo_chain_sweep*.json`
run medians against `zoo_kcount.json` device shares
(`bench/suites/integration/frontier.py:load_rows`).  Both inputs
were last committed 2026-10-26 — before the `seq`/`seqm` tiers
(2026-11-13) and before the graph arm (`0fc704f`).  The T-032/33/34
report re-measured three rows live but never wrote them back.

### Commands run (this session, RTX 2050 4 GB, idle box)

```sh
# fp64 — the canonical frontier.json input (zoo_chain_sweep.json)
.venv/bin/python -u -m bench.drivers.probes.zoo_chain --isolate \
    --dtype fp64 --rounds 2 --reps 100 --budget 400 \
    --only ChunkedSSD,DeltaRuleBlock,GatedDeltaRule,DecoupledMLA \
    --out /tmp/zoo_t037_fp64.json

# fp32 — the shipped-dtype sweep (zoo_chain_sweep_fp32.json)
.venv/bin/python -u -m bench.drivers.probes.zoo_chain --isolate \
    --dtype fp32 --rounds 2 --reps 100 --budget 400 \
    --only ChunkedSSD,DeltaRuleBlock,GatedDeltaRule,DecoupledMLA \
    --out /tmp/zoo_t037_fp32.json
```

Reduced protocol vs the canonical sweep (3–5 rounds): **2 rounds**,
`--reps 100`, `--isolate` (one process per workload, 400 s kill
budget — none hit it; total wall ~4 min for all 8 child runs).  The
rows were merged into `docs/reports/zoo_chain_sweep.json` (fp64) and
`docs/reports/zoo_chain_sweep_fp32.json` (fp32) by case, and the
frontiers regenerated:

```sh
.venv/bin/python -m bench.suites.integration.frontier
.venv/bin/python -m bench.suites.integration.frontier \
    --sweep docs/reports/zoo_chain_sweep_fp32.json \
    --stem frontier_fp32 \
    --only zoo:RWKVMixBlock,zoo:SharedSwiGLUMoE,zoo:TopKRouterMoE,zoo:DecoupledMLA,zoo:ChunkedSSD
```

(`frontier_fp32` keeps its original 5-case `--only` scope — the
other three fp32 rows are unchanged pre-arm measurements.)

### New rows (medians of 2 rounds; `graph` = same verified module
under `CudaGraphRunner`)

| case | dtype | speedup (eager) | vs compile | graph µs | graph speedup | graph vs compile |
|---|---|---|---|---|---|---|
| ChunkedSSD | fp64 | 1.005 | 0.93 | 707.0 | **3.10×** | 0.92 |
| DeltaRuleBlock | fp64 | 0.973 | 0.78 | 143.6 | **2.03×** | **1.44** |
| GatedDeltaRule | fp64 | 0.703 | 0.75 | 206.7 | **1.95×** | **2.08** |
| DecoupledMLA | fp64 | 0.992 | 0.85 | 371.4 | **1.27×** | **1.03** |
| ChunkedSSD | fp32 | 0.859 | 0.25 | 493.6 | **4.38×** | **1.30** |
| DeltaRuleBlock | fp32 | 0.994 | 0.66 | 68.9 | **4.37×** | **2.75** |
| GatedDeltaRule | fp32 | 0.642 | 0.29 | 122.3 | **3.29×** | **1.43** |
| DecoupledMLA | fp32 | 1.365 | 0.69 | 120.2 | **3.49×** | **1.77** |

(`graph_max_abs` = 0.0 on every row — replay verified bit-equal to
the eager delivered module.)

### Reading

- **The per-leaf lowering (`gen_us`) still loses or treads water on
  all four rows** — consistent with T-032/33/34: the rewrite bank
  does not reach the launch gap, and on GatedDeltaRule the assembled
  module loses to eager outright (0.64–0.70×).
- **The graph arm changes the answer completely.**  Capture of the
  *same* verified module is 1.3–4.4× over eager and beats
  `torch.compile` on 7 of 8 rows (only fp64 ChunkedSSD trails it,
  0.92×).  The "uncaptured headroom" the frontier rows flagged is
  precisely the dispatch share `CudaGraphRunner` already ships a
  delivery path for.
- **`graph` is now a first-class frontier column.**  `Row.graph_us`
  (median of per-round `gen_graph_us`), `graph_speedup`,
  `graph_vs_compile`, `beats_graph_compile`, and
  `delivered_graph_us`/`graph_speedup`/`graph_vs_compile`/
  `beats_graph_compile` in `to_dict`; the markdown table gained
  `graph µs` / `graph speedup` columns; rows without arm data show
  `—`.  The lowering `speedup`/`captured` semantics are unchanged —
  the graph arm is reported beside `delivered`, not folded into it.
- Supporting change in `zoo_chain._measure_one`: `gen_graph_us` is
  now recorded **per round** inside `runs[]` (previously only the
  row-level copy from round 1 survived — a ratio/µs pairing across
  rounds, the exact mistake the `runs[]` convention exists to
  prevent), and row-level `gen_graph_us`/`graph_speedup`/
  `graph_vs_compiled`/`graph_max_abs` are medians/worst-case over
  the rounds that captured.

### Caveats

- 2 rounds, not the canonical 3–5 — medians are noisier; spread is
  kept in `spread`/`runs`.  ChunkedSSD fp32 flipped to 0.859 this
  run (claims `gen_13`/`gen_15` were kept by `profitable` and the
  delivered module lost) — claim sets also differ run-to-run
  (`synop_*` names change between identical invocations), so mint
  ordering is not deterministic; treat single-run claim lists as
  samples.
- `zoo_kcount.json` `device_share` was measured 2026-10-26 on the
  spelled modules — unchanged inputs, still valid for these cases.
- frontier.json verdicts after the merge: median capture −0.02 over
  35 rows (NEGATIVE, unchanged conclusion — the *lowering* still
  doesn't capture); the graph column is where the headroom shows up.

## 2. BatchedScanModule vs IRModule — root-caused and fixed

### Reproduction (T-029 numbers reproduced exactly)

Fold-free `GatedDeltaRule` term (`noprim` ruleset, small d16/t8/b2,
fp64 CPU — the probe's contract): `IRModule` 0.118–0.126 ms vs
`BatchedScanModule` 0.418–0.429 ms → **3.4× slower** (pay_measured
recorded 0.416 vs 0.125 ms — same ratio).

### The term's shape

`is_scan_apply_term` accepts it: root `applyd`, map tree a pure
`affd_compose`.  But `build_scan_plan` finds **2 leaves, 1 level** —
the whole "scan" is a *single* compose — while the leaf `(A, b)`
payloads carry **67 unique ops**, including a *nested* `applyd`
sub-scan inside a `b` argument (the recurrence's per-step gate math
plus an inner scan, all hidden inside opaque leaf payloads).

### Mechanism (three stacked costs, measured)

1. **Nothing to batch.**  One compose level means the schedule
   replaces a single `affd_compose` binding with leaf stacking,
   gathers, and one level of index_selects/cats — strictly more
   dispatched work in every regime.
2. **Forfeited DAG dedup.**  `IRModule` evaluates the whole term as
   ONE plan tape: shared subtrees are slots, evaluated once.
   `BatchedScanModule` evaluates each leaf's `(A, b)` (and `h`) as a
   *separate* `run_env` root via `_eval_fast` — memo-free inside —
   so subgraphs shared ACROSS leaf args are recomputed.  cProfile:
   608k vs 191k Python calls per 300 forwards (3.2×).
3. **Per-root overhead.**  Each `run_env` re-allocates the slot
   vector, re-fills var/param/const slots, and re-resolves bindings
   for its subplan; `broadcast_shapes`/`_slot_expand` add more per
   call.

Experiment: forcing leaf-arg eval through the memoized recursive
path (`eval_term` with a shared memo) recovered part of the gap
(3.39× → 2.15×) but not all — `eval_term` per-node cost plus the
per-root tape overhead remain.  The designed-for case confirms the
boundary: trivial leaves (`a` Param, `b=select`) on CUDA win big
(0.33–0.59× generic; fused 0.12–0.22×), while per-step *gate*
payloads are already ~1.0× — the spine is noise next to payload
compute.

### The fix (implemented)

`build_scan_plan` (`packages/mlopt-carriers/src/mlopt_carriers/
scan_lower.py:388`) now declines **`len(levels) < 2`** — extending
the existing `not levels` rationale one depth up: ≤2 leaves means
≤1 compose node, so the batched schedule is strictly more machinery
than the generic evaluator AND forfeits cross-payload dedup.  A
≥3-leaf spine (depth ≥2) still plans.

`is_scan_apply_term` is unchanged — it is a structural probe; the
plan is the cost gate (same division of labour as the existing
single-leaf decline).  `BatchedScanModule` on a declined term is a
transparent `IRModule` (`is_batched=False`), so worst case is
parity, never the 3.4× loss.

Verified: the GatedDeltaRule repro now measures **1.01×** vs
`IRModule` (was 3.40×); `is_batched=False`, output identical.

### Test updates (`tests/test_batched_scan_lift.py`)

Three tests pinned the old "two leaves → one real compose level →
still plans" contract; updated to the new gate (3-leaf spines where
the test's actual subject is shape-consistency/carrier-mix, plus an
explicit 2-leaf decline assertion in `test_lone_leaf_declines`).

**Pre-existing failures (NOT from this change — verified failing on
HEAD `218db2f` before the edit):**
`TestEndToEnd::test_batched_chunk_lowers_batched` and
`test_batched_chunk_fp64_decode`.  The chunked-decode modules now
extract a *single-leaf* `applyd` spine (the seq/seqm tiers absorbed
the recurrence), so `build_scan_plan` declines under the OLD `not
levels` gate already — `lowering: "generic"`, `is_batched: False`.
These tests need re-basing onto whatever now claims the chunk
spine (likely the seq tier) — left as a ticket note, not fixed here.

### Residual (documented, not fixed)

Even with ≥2 levels, leaf payloads are evaluated as separate
`run_env` roots — cross-leaf sharing is still forfeited vs the
single-tape `IRModule` (the memo experiment above).  On CUDA the
designed case still wins (launch-bound), but on CPU fp64 a deep
spine with gate payloads measured 1.4–1.9× *slower* than generic.
A payload-cost-aware plan gate (unique-op count under leaf args vs
spine size) is the principled next step if CPU-eager delivery of
scan terms matters; on CUDA the current gate is sufficient for the
observed regressions.

## Files touched

- `packages/mlopt-carriers/src/mlopt_carriers/scan_lower.py` —
  `len(levels) < 2` decline + rationale comment.
- `bench/drivers/probes/zoo_chain.py` — per-round `gen_graph_us`
  in `runs[]`; row-level graph fields as medians/worst-case.
- `bench/suites/integration/frontier.py` — `Row.graph_us` +
  graph properties, `to_dict` keys, markdown columns + note.
- `tests/test_batched_scan_lift.py` — 3 tests re-based to the new
  level gate.
- `tests/test_ports.py` — `_scan_ir` fixture re-based to a real
  3-leaf spine (it was a 1-unique-leaf/1-level spine that only
  passed because the old gate accepted it).
- `tests/test_frontier.py` — new graph-arm median/None test.
- `docs/reports/zoo_chain_sweep.json`, `zoo_chain_sweep_fp32.json` —
  4 rows each replaced with the T-037 re-measurement.
- `docs/reports/frontier.json`, `frontier.md`, `frontier_fp32.json`,
  `frontier_fp32.md` — regenerated.

## Verification

- `pytest tests/test_frontier.py tests/test_scan_lower_edges.py
  tests/test_batched_scan_lift.py tests/test_executor_base.py
  tests/test_scan_batched.py` — all pass except the 2 pre-existing
  EndToEnd failures above (failing on HEAD before this change).
- `ruff check` clean on all touched files; `ruff format` drift in
  `bench/`/`tests/` is pre-existing and outside the gated
  `packages scripts examples` scope.
- Sweep drivers run standalone (not pytest) per the memory rules —
  serial, one workload process at a time, ~3.8 GB host available
  throughout, no OOM.

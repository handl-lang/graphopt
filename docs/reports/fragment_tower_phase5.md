# Fragment tower — phase 5: handl-native saturation

Plan 0033 / host ADR-0157 / ledger T-017 (last phase).  Phase 5 lands
the plan's final seat: **the e-graph saturates natively on the handl
side** — `handl-egraph`'s `EGraph::run` drives saturation over the
fragment vocabulary's rewrite rows, reached through `mlopt-kit`'s
`egraph-run` binary (the same subprocess door
`bench/drivers/audit/egraph_differential.py` diffs whole runs
through).

## Delivered

`mlopt_discovery.handl_egraph` — the native-saturation module:

- `vocab_rewrites()` — splits the canonical `vocab/` rewrite bank
  (`laws.yaml` + `schemas.yaml` rows: `law`, minted `prim:*` folds,
  `fold:*` schemas) into the wire-expressible subset.  **84
  declarative rows saturate; 64 hooked rows (`check`/`derive` Python
  closures) are counted and named in `VocabRewrites.skipped`, never
  silently dropped; 0 epsilon rows.**  `op`/`model`/`family` rows are
  vocabulary data, not moves — unchanged from phase 4.
- `spec_term` / `build_spec` — the `{"mvar"}/``{"leaf"}/``{"op"}``
  wire encoder, pinned field-for-field equal to the audit driver's
  `_spec_term`/`spec_for` in tests.
- `run_native` / `saturate` — one `egraph-run` invocation per spec;
  returns the canonical engine-independent report (WL class
  signatures, member multisets, stats, provenance, extracted `best`,
  `check_terms` probes).  `HandlEgraphUnavailable` /
  `EgraphRunError` on a missing binary / failed run — loud, never a
  silent Python fallback.  Binary resolution: `MLOPT_EGRAPH_RUN` /
  `CATOPT_EGRAPH_RUN` env, then `target/debug` and
  `~/.cargo/shared-target/debug` (same order as the audit driver).

Verification is the differential contract itself: the Rust report is
compared against the Python `EGraph` run on the *same* spec, field by
field — divergence is a finding, not a fallback.

## Verified

`tests/test_handl_egraph.py` — **95 tests, all passing** (1.7 s):

- encoder parity (`spec_term`/`build_spec` vs the audit driver's,
  parametrized over concrete + pattern terms);
- the honest split (no hooked row leaks; skipped rows carry reasons;
  deterministic; `ValueError` with no vocab dir);
- binary resolution (env override order) and loud error paths;
- **per-rule parity: all 84 declarative vocab rows**, each saturated
  alone on a concrete instance of its own lhs — `run_native`'s report
  is byte-identical to `egraph-run`'s for the same spec, and the
  canonical `ed.compare` verdict is `agree`/`tie` on every row;
- a 13-rule compound fixture (algebra + silu/pow folds, comm/assoc
  budgeted) — `agree`;
- end-to-end extraction (`x*1 + 0` → leaf `x`, root-class membership
  confirmed on the Python oracle) and `check_terms` probes.

The binary-dependent tests are `e2e`-marked and skip honestly when
`egraph-run` is unbuilt (the `_build_bin` cargo fallback mirrors
`test_egraph_differential.py`).

`bench/drivers/audit/egraph_perf_parity.py` — the perf-parity
measurement driver (see below).

## The perf-parity gate — measured, and it does NOT pass

The plan's hard gate: *"`handl-egraph` saturation time/memory within
2× of `EGraph` on the 68-module corpus."*  Measured on the held-out
`mlopt_discovery.zoo` corpus — all 68 workloads lifted via
`TorchSource().to_ir`, saturated under the same 84 declarative vocab
rows, bounded by the audit fixtures' convention
(`max_iterations=40`, `max_nodes=20_000`, `patience=3`, comm/assoc
budgeted at 300).  Python timed untraced (a second tracemalloc pass
supplies peak heap); Rust timed per-invocation with the child's own
`VmHWM` polled from `/proc` (a `wait4`/`ru_maxrss` reading was tried
first and rejected — it momentarily counts the ~430 MB torch-loaded
parent at spawn; measured, not guessed).

**Debug `egraph-run` (`target/debug`, the binary the tests and the
audit driver use):**

| metric | value |
|---|---|
| modules measured | 68 / 68 (0 lift failures) |
| python total | 51.8 s |
| rust total | 225.1 s |
| **aggregate ratio** | **4.35× — gate FAILS** |
| worst module | RWKVMixBlock 46.5× |
| pathological | RealGatedLRU: Python 48.7 s / 3.4 GB tracemalloc peak → hit `max_nodes` at 925 k enodes; Rust **timed out at 120 s at ~4.9 GB RSS** |
| heavy firings | MambaSelectiveScan 37.5×, GatedLinearAttention 38.3×, GatedDeltaRule 45.9× (Rust 27–39 s vs Python 0.6–1.0 s on ~19 k-enode saturations) |
| where Rust wins | ChunkedRetention 0.22×, GRUFusedGate 0.15× — engine-vs-engine Rust is *faster* once real work dominates; the aggregate is inflated by ~10–15 ms fixed spawn+spec-parse cost on the ~50 trivial modules and the three blow-ups above |

**Parity on the corpus:** 66/68 modules produce identical stats.
Two divergences — `DeltaRuleBlock` and `ChunkedSSD` — are honest
findings: the saturated **e-graph class structure is identical**
(WL signatures, member multisets, root sig all equal), but the
*firing counters* differ (e.g. DeltaRuleBlock `n_enodes` 894 vs 836,
`assoc_add` applications 232 vs 180, `n_proof_edges` 794 vs 738) —
the Rust matcher records fewer applications yet reaches the same
saturated classes.  `ed.compare` verdicts on both: `diverge` on the
strict stats/provenance fields.  DeltaRuleBlock's extracted `best`
also differs — consistent with the documented member-order tie
class (identical classes, equal-cost members), not a correctness
break, but flagged rather than assumed.

**Release build — a second finding:** `cargo build --release -p
mlopt-kit --bin egraph-run` (handl-egraph @ `231172f5`, same rev as
the lock) produces a binary that **panics on every spec** —
`union_find.rs:25: index out of bounds: the len is 0 but the index is
0`, exit 101, even on a one-leaf term.  The debug build is unaffected.
This is an upstream `handl-lang/egraph` release-mode bug (or a
release-profile interaction in `mlopt-kit`), not a graphopt defect —
filed here as a finding for the egraph repo; the shipped door stays
the debug binary the whole toolchain already uses.

## Honest scope — what phase 5 is and is not

- **Native saturation exists and is parity-verified**: the fragment
  vocabulary's declarative rewrite bank saturates inside
  `handl-egraph`'s `EGraph::run`; Python never sees an intermediate
  e-class; verification is the canonical differential report.
- **Procedural hooks stay Python** — the 64 `check`/`derive` rows are
  counted in `skipped`, not hidden; migrating guards toward
  typecheck-nbe predicates remains the plan's dependent-surface work.
- **The perf-parity gate is not met.**  Per the plan's own wording —
  "`handl-egraph` becomes the saturation engine *when* the
  perf-parity gate passes" — the conditional does not trigger:
  `saturate()` is opt-in API, **not** wired as any default engine.
  Python `EGraph` remains the saturation engine of record; the Rust
  side is the verified second engine (and wins outright on the
  mid-weight modules once spawn overhead amortises).
- `Cargo.lock` churn from cargo re-resolution (patch-table
  reordering) was verified benign and restored.

## Files

- `packages/mlopt-discovery/src/mlopt_discovery/handl_egraph.py` —
  the module (pre-existing draft completed/verified; en-dash +
  formatting fixes only).
- `tests/test_handl_egraph.py` — 95 tests.
- `bench/drivers/audit/egraph_perf_parity.py` — the corpus
  perf-parity driver (`--json` emits the full row set; raw output
  captured at measurement time).
- Raw numbers: python 51.812 s / rust 225.136 s aggregate, per-module
  rows in the driver's `--json` output.

## Open findings for follow-up

1. **Perf gate residual** — aggregate 4.35× vs the 2× gate; the gap
   decomposes into (a) ~10–15 ms fixed per-invocation overhead that a
   library-boundary seat (PyO3/maturin, batch spec) would not pay,
   (b) debug-build matching throughput on heavy firings (27–46× on
   three modules), (c) one pathological case (RealGatedLRU) where
   Rust is slower *and* hungrier (4.9 GB vs Python's 3.4 GB
   tracemalloc peak before `max_nodes`).  A release-profile fix
   (blocked by finding 2) plus an in-process door are the obvious
   next measurements.
2. **Upstream release panic** — `handl-egraph` @ `231172f5` release
   build: `union_find.rs:25` OOB on every spec →
   `handl-lang/egraph` or `mlopt-kit` release profile.
3. **Stats/provenance counter drift** — 2/68 corpus modules saturate
   to identical classes with different application counts; worth a
   look in `handl-egraph`'s match enumeration vs Python's (benign
   direction — fewer recorded firings — but the audit driver's
   strict verdict says `diverge`).

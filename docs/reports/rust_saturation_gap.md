# Rust-vs-Python e-graph saturation gap — re-measured

**Date:** 2026-11-13 (measurement), repo `handl-lang/graphopt` @ `58df5dd`.
**Driver:** `bench/drivers/audit/egraph_perf_parity.py` (unchanged; it
already requests `report="stats"` since commit `218db2f`).
**Binary:** `target/release/egraph-run` — `cargo build --release -p
mlopt-kit --bin egraph-run`, handl-egraph @ lock rev, built *after*
the release-mode panic fix and the stats/timing instrumentation
(binary mtime 11:25 > commit `218db2f` 09:52; `cargo build` reports
fresh).  Selected via `MLOPT_EGRAPH_RUN` — note
`handl_egraph._BIN_CANDIDATES` still prefers `target/debug`, so an
un-overridden run measures the **debug** binary.

## Headline

| metric | phase-5 record (debug, full report) | this run (release, stats) |
|---|---|---|
| modules measured | 68 / 68 | 68 / 68, 0 lift failures |
| python total (`EGraph.run`) | 51.8 s | 56.1 s |
| rust total (wall, incl. spawn) | 225.1 s | **13.5 s** |
| aggregate ratio | **4.35× — gate FAILS** | **0.24× — Rust ~4× faster** |
| worst module | RWKVMixBlock 46.5× | SoftSlotMoE 4.02× (5 ms vs 1 ms — spawn floor) |
| heavy firings | MambaSelectiveScan 37.5×, GatedLinearAttention 38.3×, GatedDeltaRule 45.9× (Rust 27–39 s) | same modules **0.19–0.21×** (Rust 0.10–0.22 s vs Python 0.5–1.0 s) |
| RealGatedLRU | Rust **timed out at 120 s**, ~4.9 GB RSS | Rust **12.6 s**, ~3.9 GB RSS vs Python 53.3 s / 3.37 GB tracemalloc peak |
| stats divergences | DeltaRuleBlock, ChunkedSSD | same two — unchanged, known finding |

Raw rows: `/tmp/parity_full_release.json` (driver `--json` output).

## Where the old 4.35× came from

Three confounds, all now removed or quantified:

1. **Debug build.** The phase-5 table says so itself: "Debug
   `egraph-run` (`target/debug`, the binary the tests and the audit
   driver use)".  Release-mode codegen on the match/enumeration hot
   loop is the dominant term — the same three "pathological" modules
   went from Rust 27–39 s (debug) to 0.10–0.22 s (release), i.e. a
   ~150–200× swing on saturation-dominated work.
2. **Report generation inside the timed region.** The old timing
   covered the canonical report (WL labels, member renderings,
   extraction — measured up to ~50 s on heavy modules).  `report="
   "stats"` (commit `218db2f`) returns only the stats payload, matching
   what the Python timer actually covers (`EGraph.run` alone).
3. **Per-invocation overhead.** ~5 ms spawn+parse floor per module.
   On the ~50 trivial modules (Python 1–4 ms) this reads as a
   1.3–4× "ratio" but is ≤ ~5 ms absolute — noise, not a gap.

## `EGRAPH_RUN_TIMING=1` breakdown (release, stats report)

Ticks from `packages/mlopt-kit/src/egraph.rs` (`tick()` calls at
lines 330/358/420; stats mode returns before the report ticks at
455–491, which correctly never fire):

| module | decode_terms+add_term | decode_rules | saturate | driver wall |
|---|---|---|---|---|
| RealGatedLRU | 1.78 ms | 0.59 ms | **10.29 s** | 12.64 s |
| MambaSelectiveScan | 1.32 ms | 0.65 ms | **187.4 ms** | 0.213 s |
| GatedDeltaRule | 0.77 ms | 0.48 ms | **95.8 ms** | 0.096 s |

Saturation is ~99.9 % of in-process time; spec decode and rule decode
are sub-millisecond-to-2 ms.  The residual between `saturate` and the
driver's wall time (≈2.3 s on RealGatedLRU, ≈5 ms elsewhere) is
process spawn + stdin/stdout JSON (de)serialization outside
`run_spec` — the price of the subprocess door, not of the engine.

## Conclusion

**Saturation itself is not ~4× slower — it is ~4× faster.**  The
recorded gap was an artifact of (a) timing the debug binary, (b)
timing full canonical-report generation, and (c) a release-mode panic
that blocked the fast binary until its fix.  With the release
`egraph-run` and `report="stats"`, Rust wins every module where
saturation does real work (0.15–0.24×) and loses only on modules
whose entire Python run is a few milliseconds (spawn floor).

Memory is now comparable too: RealGatedLRU peaks at ~3.9 GB RSS
(Rust) vs 3.37 GB tracemalloc peak (Python) — the debug build's
4.9 GB / timeout is gone.

Against the plan-0033 gate ("within 2× of `EGraph`"), the release
numbers **pass comfortably**: aggregate 0.24×, and even the
spawn-floor worst case is 4.02× on a 4 ms module.  Caveats carried
forward unchanged: the two stats-counter divergences
(DeltaRuleBlock, ChunkedSSD — identical saturated classes, different
recorded firing counts) remain open findings for `handl-egraph`'s
match enumeration, and `_BIN_CANDIDATES` preferring `target/debug`
means the audit driver still measures the slow binary unless
`MLOPT_EGRAPH_RUN` points at the release build — worth fixing before
anyone re-runs the gate unmodified.

## Commands run

```sh
cargo build --release -p mlopt-kit --bin egraph-run   # fresh, ~0.25 s
MLOPT_EGRAPH_RUN=$PWD/target/release/egraph-run \
  .venv/bin/python bench/drivers/audit/egraph_perf_parity.py \
  --json /tmp/parity_full_release.json               # 68 modules, ~4.5 min wall
EGRAPH_RUN_TIMING=1 ./target/release/egraph-run < spec.json  # per-module ticks
```

# Trajectory / read-out scan tier — implementation

**Date:** 2026-11-13 · **Implements:** `trajectory_scan.md` §5 — the
collector / route / denotation / referee machinery that turns a
recurrence chain whose intermediate states are consumed per-step
(the GLA/Mamba shape) into a delivered `kind="seqs"` trajectory
kernel plus the spelled read-out consumers.

**Design chosen: trajectory + spelled read-outs (report §2a, minus
the fused single einsum).**  No new kernel kind was needed — the
shipped `seqs` fold already emits every prefix state
(`out_t = acc_t`) with an independently spelled denotation
(`seqs_denotation`).  What was missing was *machinery*: a collector
that recognises "this chain's states are read out", a route that
rewrites every recorded state e-class to `select(traj, t)`, and
handler metadata carrying the state map end-to-end.  The read-out
`einsum`s keep their spelled form — each reads `S_t` off the
trajectory exactly as it read the spelled chain — so semantics are
preserved by construction and the referee verifies the whole
composite against the spelled module as usual.  The fused
trajectory+read-out einsum of the prototype and the `seqt` kernel
of §2(b) remain future work (§Residual below).

## 1. What changed

### Collector — `play.py`

- `_op_parents(term)` / `_op_descendants(root)` — DAG parent census
  and subtree membership, the structural primitives the trajectory
  decision reads.
- `_seq_traj_states(term, root, steps)` — returns
  `((state_node, step_index), ...)` for every prefix state
  `S_0..S_{T-1}` in chronological order **iff** at least one
  non-final state has a parent outside the chain's own subtree
  (a read-out consumer).  The check is deliberately
  consumer-agnostic: interior consumers (the next step's `mul`)
  are descendants of `root`; anything else reading a state node is
  a read-out the final-state `seq` fold cannot serve.  Returns
  `None` for final-state-only chains, which keep the cheaper `seq`
  mint.
- `_seqs_chain_spec(spec)` — re-kinds the existing
  `applyd(aff_diag(a,b), H)` spec `seq` → `seqs`; the carrier,
  operands, and attrs are identical, only the declared semantics
  change.
- `_seq_scan_sites` — emits `kind="seqs"` records carrying
  `"states"` (and **no** `"chain"`: the kernel returns the whole
  `(T, ·)` trajectory, not the final state a root replacement
  would need) when `_seq_traj_states` fires; otherwise the
  unchanged `seq` record with `"chain"`.

Sites minted on the zoo: `MambaSelectiveScan`,
`GatedLinearAttention`, `RealGatedLRU`, `HeadedSelectiveScan`,
`BidirectionalScan` (two chains — forward and backward — two
sites).  `GatedDeltaRule`, `ChunkedSSD` &c. keep their existing
`seq`/`seqm` mints; `TaylorLinearAttention` and
`RelPosTableAttention` mint no scan sites at all (their sweep
losses are pre-existing and unrelated).

### Metadata — `genkernel.py`

- `_site_parts` forwards `"states"` into the handler entry beside
  the existing `"chain"`/`"axis"`/`"axes"` passthroughs, so the
  route sees the `(node, index)` map on the priced/delivered
  handler.

### Route — `meta_arena.py`

- `_route_scan_chains` handles `states`-carrying handlers: builds
  the trajectory op once (`Op.make(kernel, *leaf_env args)` — same
  construction as the `chain` route), maps every recorded state
  node's **e-class** (`eg._class_of_term`, with the shared
  `cls_memo`) to `(traj, step_index, axis)`, and rewrites each
  matching subterm to `select(traj, dim=axis, index=t)` during the
  recursive walk.  The trajectory op's own args still walk, so a
  routed chain can hide inside an operand subtree.
- Ambiguity is safe by construction: a class claimed by two
  different `(traj, index)` pairs is left spelled — any *subset*
  of the rewrites preserves semantics because `select(traj, t)`
  denotes `S_t` exactly.  Incomplete bindings (`KeyError` /
  `TypeError` on env lookup) fall back to spelled, same honest
  fallback as the `chain` route.
- The final state is covered too: `states` includes
  `(root, T-1)`, so consumers of the chain root read
  `select(traj, T-1)` — the trajectory's last row IS the final
  state.  No separate `"chain"` route is emitted for traj sites,
  avoiding a double rewrite.

### Denotation / referee — no new code needed

`seqs_denotation`, `scan_spec`, `handler_denotation`,
`scan_triton_bindings`, `_seqs_fold_source`, and the `_claim_measurable`
/`profitable` paths already handle `kind="seqs"` (minted until now
only from `cumsum`/`cumprod`).  The trajectory record flows through
`_site_parts` → `scan_handlers` → `handler_denotation` unchanged;
the spelled arm the referee prices against is the full
recurrence+read-out chain, so delivery only ships when the kernel
composite beats it *and* verifies `allclose`.

## 2. Verification

New tests in `tests/test_discovery_play.py`
(`TestSeqsTrajectorySites`): the collector mints a `seqs` spec with
the full `(node, index)` state map and no `chain` key; the handler
denotes a trajectory (`scan_mode="fold"`); the route rewrites each
state e-class to the right `select` and leaves non-state subtrees
alone; final-state-only chains still mint `seq` (no trajectory
write for a `(·,)` answer); ambiguous/partial cases stay spelled.

Gates run:

- `scripts/pytest-capped.sh` — two capped runs: the discovery
  play/genkernel slice (**323 passed, 1 skipped** in 137 s) and
  `tests/test_discovery_meta_arena.py tests/test_strassen_mcts.py`
  (**78 passed** in 12 s).
- `.venv/bin/ty check` on the three touched package files — clean.
- `ruff check` / `ruff format --check` on the touched package
  files — clean (the files were reformatted once; package lint is
  green.  `tests/` is outside the `packages scripts examples` gate
  scope and `test_discovery_play.py` was already non-conforming at
  HEAD; a repo-wide `ruff check tests/` shows pre-existing
  unrelated findings).
- `scripts/radon_ratchet.py` — **pre-existing red baseline**
  (unrelated regressions across `mlopt-carriers`, `domain_spec`,
  `genkernel`, `meta_arena`, `meta_player`, `play`); the new
  `_seq_traj_states` helper was decomposed (`_op_parents` /
  `_op_descendants`) to stay under the threshold itself.

## 3. Pay verdict — module level

`bench/drivers/probes/zoo_chain.py`, delivered+verified rows:

**fp32, zoo(16,8,2)** — full 65-workload sweep, every row
delivered+verified, no crashes:

| workload | speedup | gen µs | baseline µs |
|---|---|---|---|
| GatedLinearAttention | **1.269×** | 456.5 | 579.3 |
| MambaSelectiveScan | **1.099×** | 569.2 | 625.7 |
| RealGatedLRU | **4.027×** | 210.7 | 848.5 |
| HeadedSelectiveScan | **1.184×** | 613.8 | 726.4 |
| BidirectionalScan | **1.228×** | 312.3 | 383.5 |

Paying scans are unaffected: GatedDeltaRule 11.5×, ChunkedSSD
10.4×, DecoupledMLA 1.6×.

**fp64, zoo(64,32,8)** — targeted rerun of the seqs-minting set
(`--reps 60 --rounds 3`):

| workload | result |
|---|---|
| GatedLinearAttention | **delivered, verified, 1.651× vs eager** (median of 1.617–1.721; `gen_graph_us` 1105.7, `graph_speedup` 2.014, `graph_vs_compiled` 1.253, max_abs 2.2e-15) |
| MambaSelectiveScan | **delivered, verified, 1.613× vs eager** (median of 1.585–1.793; `gen_graph_us` 382.0, `graph_speedup` 5.752, `graph_vs_compiled` 1.462, max_abs 1.8e-15) — needed `--budget 900`; at the default 150 s the *search* phase alone timed out |
| RealGatedLRU | over the per-workload *search* budget even at 900 s — no row; delivery was not reached.  The fp32 row above (4.03×, delivered+verified) is the evidence; the fp64 search blowup is in saturation, not delivery |
| HeadedSelectiveScan | **delivered, verified, 1.589× vs eager** (graph arm 4.549×, `graph_vs_compiled` 1.570, max_abs 4.4e-16) |
| BidirectionalScan | **delivered, verified, 2.583× vs eager** (median of 2.47–2.90; `gen_graph_us` 278.3, `graph_speedup` 4.215, `graph_vs_compiled` 1.269, max_abs 1.1e-15) |

**Verdict: the trajectory tier pays at module level.**  Every
seqs-minting workload that completes search delivers, verifies
(`allclose`, max_abs ≤ ~2e-15 fp64), and beats eager — GLA 1.65×,
Mamba 1.61×, HeadedSelectiveScan 1.59×, BidirectionalScan 2.58×
(fp64); RealGatedLRU's fp64 row is search-budget-bound, its fp32
row pays 4.03×.

**Before → after on GLA:** `module_loss_gla_mamba.md` measured the
delivered module at **3208 µs vs 1861 µs eager — a loss**.  After
this change the same probe shape reports **~1294 µs gen vs ~2227 µs
eager — a 1.65× win**, with the CUDA-graph arm at 2.01×.  Mamba
went the same way: **3502 µs delivered vs 1814 µs eager (loss) →
~1300 µs gen vs ~2200 µs eager (1.61× win; graph arm 5.75×)**.  The
losses are gone because the trajectory route now replaces the
recurrence *and* its read-out consumers instead of folding only
the final state on top of the spelled loop.

## 4. Residual loss — honest accounting

The v1 delivery is **trajectory + T spelled read-outs**, not the
prototype's trajectory + ONE batched einsum, and not the `seqt`
fused kernel.  What still costs vs the prototype's ~479 µs (GLA) /
~93 µs (Mamba) recurrence-level numbers:

- **T separate `select`+`einsum` launches** remain where the
  prototype used one batched einsum (`"btij,bti->btj"`).  The
  collector records *that* states are consumed, not the read-out
  contraction structure (which operand contracts which state axis)
  — capturing that is the `seqt`/read-out-spec work of report §5.2.
- **The `(T, ·, state)` trajectory write** stays (4 MB for GLA) —
  design (b) would write only `(T, ·, out)`.
- **The ~151 µs `_seq_args` broadcast materialisation** in the
  seqs binding is unchanged (shared with the `seq`/`cumsum` tiers).
- **Host tape / per-op dispatch** of the remaining spelled
  read-outs is the same overhead class `module_loss_gla_mamba.md`
  flagged; it is now amortised over a delivered recurrence instead
  of a fully spelled one.

Net: the structural win (one scan launch vs ~200 spelled ops) is
real and measured at module level; the remaining gap to the
prototype floor is bounded, identified, and scoped — it is the
fused-read-out work, not a correctness or routing defect.

## 5. Files

- `packages/mlopt-discovery/src/mlopt_discovery/play.py` —
  `_op_parents`, `_op_descendants`, `_seq_traj_states`,
  `_seqs_chain_spec`, `_seq_scan_sites` upgrade.
- `packages/mlopt-discovery/src/mlopt_discovery/genkernel.py` —
  `_site_parts` `"states"` passthrough.
- `packages/mlopt-discovery/src/mlopt_discovery/meta_arena.py` —
  `_route_scan_chains` trajectory-state routing.
- `tests/test_discovery_play.py` — `TestSeqsTrajectorySites`.
- `docs/reports/trajectory_scan_impl.md` — this report.

No commits, no `myco`, no push.

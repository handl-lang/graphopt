# `seqt` — the fused trajectory + read-out scan tier

**Date:** 2026-11-14 · **Builds on:** `trajectory_scan.md` (the
prototype + scoping), `trajectory_scan_impl.md` (the `seqs`
trajectory tier, commit `736c81c`), `module_loss_gla_mamba.md`.

**Short answer: the fused read-out tier lands and pays.**  `seqt`
is `seqs` with the per-step contraction folded into the same
kernel: one launch computes `acc_t = a_t·acc + b_t`, contracts the
accumulator slab against the step's read-out operand
(`o_t[o] = Σ_c acc_t[o,c]·w_t[p(o),c]`), and writes ONLY the
`(T, ·, out)` read-out matrix — no `(T, ·, state)` trajectory
write, no T separate `select`+`einsum` launches.  The collector
only mints it when the read-out structure is *provably* the
two-operand einsum-over-`select` pattern feeding one `stack`; every
other trajectory site keeps the `seqs` offer as fallback.  At
module level (fp64, `zoo(64,32,8)`, delivered+verified through the
shipped `zoo_chain` path): **GatedLinearAttention 1.65× → 2.06×**,
**MambaSelectiveScan 1.61× → 6.29×**, **HeadedSelectiveScan
1.59× → 5.38×**, **BidirectionalScan 2.58× → 2.37×** (unchanged —
it honestly stays `seqs`; its read-out is not the einsum pattern).

## 1. What changed

### Collector — `play.py`

- `_seqt_readout(term, root, states)` (:1986) — given a trajectory
  site's `(node, step_index)` state map, proves the fused shape:
  * every recorded state has EXACTLY ONE consumer outside the
    chain's own subtree;
  * that consumer is a two-operand `einsum` (only an `equation`
    attr) with the state node as one argument and
    `select(child, dim, index)` — `index ==` the state's step
    index — as the other;
  * all steps share the equation, the state operand position, the
    read-out child, and the select dim;
  * `genkernel._seqt_layout` accepts the equation (below);
  * the einsums feed ONE `stack` whose args are the einsums in
    step order.
  Returns `{"equation", "state_pos", "ops": [{"meta": "W1",
  "axis"}], "stack", "stack_dim"}` plus a `leaf_env` extension
  binding `W1` to the whole read-out sequence tensor.  Any
  deviation → `None`, and `_seq_scan_sites` (:2173) emits only the
  `seqs` site — the fallback is structural, not a guess.
- `_seqt_chain_spec` (:1973) re-kinds the same `applyd` carrier
  `seq` → `seqt`; the fold skeleton is identical to `seqs`.

### Equation grammar — `genkernel.py`

- `_seqt_parse_equation` (:2977) — explicit `lhs->out` only; no
  `...`, no implicit equations, no repeated labels, alphabetic
  labels only.
- `_seqt_layout` (:3001) — the recognised contraction::

      out[kept…] = Σ_c state[kept…, c] · w[kept-prefix…, c]

  exactly ONE summed state axis shared with the read-out operand;
  the output is the kept labels in state-axis order; the read-out
  operand's subscripts are a PREFIX of the kept labels followed by
  the contract label LAST (so its per-step slice broadcasts over
  the suffix outer axes the kernel slab owns).  Anything else —
  permuted outputs, several summed labels, non-prefix read-outs —
  declines.

### Denotation — `genkernel.py`

- `seqt_denotation` (:3093) — independently spells the minted
  op's meaning: the same `acc = a_t·acc + b_t` walk as
  `seqs_denotation`'s fold arm, then per step evaluates the
  recorded einsum with `acc` in the state operand's position and
  the bound read-out tensor sliced at its own step axis, and
  `torch.stack`s at `stack_dim`.  Positional contract: spec
  metavars (`info["params"]`) then the read-out operand tensors —
  the same order `_handler_mvs` (:775) and the route keep.
  Incomplete/malformed `readout` records raise at call time —
  honest decline, never a guess.
- `handler_denotation` (:2367) dispatches `kind="seqt"`.

### Kernel — `genkernel.py`

- `_seqt_triton_source` (:3177) emits `_gst`: one program per
  `BLOCK_O=64` slab of the flattened OUTER state space owns the
  WHOLE contract axis (`_SEQT_MAX_C=64`), so the per-step
  contraction is a local `tl.sum` — the same program-decomposition
  trick `_gsm` uses for `Kᵀ·acc`.  Sequence operands arrive
  contiguous `(T, O, C)` (step axis front, state axes permuted
  outer-then-contract); the read-out operand arrives `(T, P, C)`;
  the kernel writes the `(T, O)` read-out matrix and the host
  reshapes/moves it into the stack's layout (:3258 `_seqt_binding`).
- `scan_triton_bindings` (:3856) binds `seqt` through
  `_seqt_binding`; `gen_bindings`/`hybrid_sink` fall back to the
  denotation where the Triton path declines (non-CUDA tensors,
  contract axis > 64, read-out rest ≠ `(outer-prefix…, C)`).

### Route — `meta_arena.py`

- `_route_scan_chains` (:1144): a `readout` site rewrites the
  recorded `stack` e-class to ONE `Op.make(kernel, *spec_args,
  *readout_ops)` — the recurrence, the T einsums, and the stack
  all leave the delivered term.  The `states`/`chain` routes are
  untouched; `seqs` remains the co-minted fallback.

## 2. Collector coverage on the zoo (`zoo(64,32,8)`, fp64 IR)

| workload | sites minted | `seqt` read-out |
|---|---|---|
| GatedLinearAttention | `seqt` + `seqs` | `einsum("bdv,bd->bv")`, state pos 0, `W1` axis 1, stack dim 1 |
| MambaSelectiveScan | `seqt` + `seqs` | `einsum("bdn,bn->bd")`, state pos 0, `W1` axis 1, stack dim 1 |
| HeadedSelectiveScan | `seqt` + `seqs` | `einsum("bhdn,bhn->bhd")`, state pos 0, `W1` axis 2, stack dim 2 |
| BidirectionalScan | `seqs` + `seqs` | **none** — two chains, no einsum-over-select read-out; honest decline |
| RealGatedLRU | `seqs` | **none** — returns the stacked trajectory itself; no contraction to fuse |

The delivered GLA term is literally one op — verified:
`(scan_64_k (unsqueeze (sigmoid (linear …))) … (linear x, p_q …))`
— no `einsum`, no `stack`, no `add`-chain survives.

## 3. Verification

- `seqt_denotation` vs the spelled recurrence+einsum+stack:
  **bit-exact** on the toy fixture; vs the Triton binding on CUDA
  fp32: max_abs 2.4e-7 (fp32 accumulation order).
- Module-level `max_abs` (fp64 probe): GLA 3.6e-15, Mamba
  1.8e-15, HeadedSelectiveScan 4.4e-16, BidirectionalScan
  1.3e-15 — all `verified` through `zoo_chain`'s deliver+verify.
- Tests: `TestSeqtReadoutSites` (play — mint record, handler arg
  order, denotation, route-to-one-op, value preservation, three
  decline shapes) and `TestSeqtSpec`/`TestSeqtEquationLayout`/
  `TestSeqtDenotation`/`TestSeqtBindings` (genkernel — grammar,
  layout classification, spelled denotation, CUDA binding vs
  denotation, contract-cap and CPU declines).  Existing
  `TestSeqsTrajectorySites` now filters to the `seqs` site
  explicitly — `seqt` co-mints on the same fixture.
- `pytest tests/test_discovery_play.py tests/test_discovery_genkernel.py`
  under `scripts/pytest-capped.sh`: **315 passed, 1 skipped, 1
  flaky failure** — `TestMeasuredReferee::test_paying_kernel_kept_
  losing_dropped` failed once while the GPU probe saturated the
  box and passed on re-run (timing-sensitive referee test,
  unrelated to this change; the whole class passes 11/11 idle).
- `ruff check` + `ruff format --check` clean on the three package
  files; `ty check packages/mlopt-discovery` clean.  (Repo-wide
  `ruff check tests/` shows pre-existing unrelated findings, same
  as the `seqs` report noted.)

## 4. Pay verdict — module level

`bench/drivers/probes/zoo_chain.py`, `--dtype fp64 --d 64 --t 32
--b 8 --reps 60`, medians of independent rounds, delivered+verified
rows.  v1 column is the `trajectory_scan_impl.md` §3 measurement of
the `seqs` delivery (same probe, same shapes).

| workload | v1 (`seqs`) speedup | `seqt` speedup | v1 `gen_graph_us` | `seqt` `gen_graph_us` | `seqt` minted? |
|---|---|---|---|---|---|
| GatedLinearAttention | 1.651× | **2.055×** (spread 1.64–2.60) | 1105.7 | **~1034** | yes |
| MambaSelectiveScan | 1.613× | **6.286×** (spread 6.17–10.5; eager-arm outliers) | 382.0 | **287.5** | yes |
| HeadedSelectiveScan | 1.589× | **5.383×** (spread 4.97–5.88) | — (graph arm 4.549×) | **420.2** (graph arm 5.823×, `graph_vs_compiled` 2.103) | yes |
| BidirectionalScan | 2.583× | 2.371× (spread 2.14–2.49) | 278.3 | 277.5 | **no — stays `seqs`** |

Honest reading:

- **GLA is the modest case.**  Its contract axis is `d=64` —
  exactly the `_SEQT_MAX_C` cap — so the kernel's `(64, 64)`
  accumulator tile is at the register-pressure edge; the fused
  read-out still wins (~7% graph-arm, ~24% on the per-leaf `gen_us`
  arm: median ~1034 vs v1's implied ~1294 µs) but nowhere near the
  prototype's ~479 µs recurrence-level floor.  The residual is the
  same `_seq_args` broadcast materialisation + launch overhead the
  `seqs` report flagged, now over a smaller write.
- **Mamba and HeadedSelectiveScan are the big wins.**  Contract
  axes `n`/`dh·n` are small, so the fused contraction is nearly
  free inside the fold: Mamba's graph arm went 382 → 287 µs
  (−25%), Headed's `gen_us` ~451 µs with graph arm 5.82× vs eager.
  The speedup ratios are partly eager-arm noise (Mamba's first
  round read eager at 15.9 ms); the kernel-level µs figures are
  the stable evidence.
- **BidirectionalScan is unchanged by design** — two recurrence
  chains whose outputs are combined, not per-step einsum
  read-outs; the collector declines and the `seqs` delivery stands
  (2.37× vs 2.58× is round-to-round noise on an identical
  delivered structure — `gen_graph_us` 277.5 vs 278.3).
- **RealGatedLRU stays `seqs`** (returns the trajectory itself) —
  see the fp32 table below for its rerun row.

**Verdict: `seqt` pays where it mints.**  Every module whose
read-out is the recognised einsum pattern improved or held; the
module that does not match keeps the verified `seqs` fallback.
The remaining gap to the prototype floor is the broadcast
materialisation + per-op host overhead class already documented —
not a correctness or routing defect.

### fp32, `zoo(16,8,2)` — same probe, `--reps 100 --rounds 3`

| workload | v1 (`seqs`) speedup | `seqt` speedup | `seqt` `gen_graph_us` | minted? |
|---|---|---|---|---|
| GatedLinearAttention | 1.269× | **2.386×** (1.71–2.39) | 38.6 | yes |
| MambaSelectiveScan | 1.099× | **2.251×** (2.24–2.26) | 37.2 | yes |
| HeadedSelectiveScan | 1.184× | **2.111×** (1.68–2.19) | 45.7 | yes |
| BidirectionalScan | 1.228× | 1.348× (1.07–1.73) | 40.2 | no — `seqs` |
| RealGatedLRU | 4.027× | 2.413× (2.40–3.31) | 40.6 | no — `seqs` |

The three `seqt` rows roughly **double** their v1 speedups at the
small fp32 shape.  BidirectionalScan's move is noise on an
identical delivered structure.  RealGatedLRU's 4.03× → 2.41× is
NOT a seqt effect — it stays `seqs` and its delivered structure is
unchanged; the v1 4.03× was a single sweep row and this rerun's
spread (2.40–3.31) plus different search outcomes (`gen_*` claim
mix differs run to run) put it in the same band as the other
scan modules.  Flagged honestly rather than explained away.

## 5. Files

- `packages/mlopt-discovery/src/mlopt_discovery/play.py` —
  `_seqt_chain_spec`, `_seqt_readout`, `_seq_scan_sites` seqt offer.
- `packages/mlopt-discovery/src/mlopt_discovery/genkernel.py` —
  `_seqt_parse_equation`, `_seqt_layout`, `_seqt_static`,
  `seqt_denotation`, `_seqt_triton_source`, `_seqt_kernel_for`,
  `_seqt_binding`, `_handler_mvs` read-out metas, dispatch in
  `handler_denotation` / `scan_triton_bindings` / `_site_parts`.
- `packages/mlopt-discovery/src/mlopt_discovery/meta_arena.py` —
  `_route_scan_chains` read-out route.
- `tests/test_discovery_play.py` — `TestSeqtReadoutSites` +
  `TestSeqsTrajectorySites._seqs_sites` filter.
- `tests/test_discovery_genkernel.py` — `TestSeqt*` classes.
- Probe data: `/tmp/seqt_zoo_fp64.json` (first pass, partially
  load-contaminated by a concurrent pytest run),
  `/tmp/seqt_zoo_fp64_rerun.json` (GLA+Mamba, 5 rounds on an idle
  box — the quoted fp64 medians), `/tmp/seqt_zoo_fp32.json`
  (fp32 `zoo(16,8,2)`, 3 rounds).

No commits, no `myco`, no push.

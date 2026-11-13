# `_seq_deindex` commute lift — the axis-safe-op boundary

Follow-up to `seq_deindex_params.md`.  That report shipped the
`Param`/projection lift and listed the next honest boundary:
"non-elementwise step ops other than `matmul`/`linear` — `softmax`,
`sum`, `layer_norm`, `transpose`, views … still decline: the `_gs`
seq kernel is elementwise-only."

**Verdict: implementable as a *hoist*, not as renderer grammar.** The
seq kernel (`_seq_triton_source`) stays a flat elementwise `[BLOCK]`
fold — no new Triton was written.  Instead the collector now pulls a
per-step `select` outward through every op whose denotation provably
commutes with the step-axis slice:

```
op(select(x, d, i), …)  ==  select(op(x, …), D, i)
```

whenever `op` does not contract, reduce, permute-away, or otherwise
destroy the selected axis — with `D` the output axis the step axis
lands on after the op's own axis bookkeeping.  The operand then binds
the **whole-tensor** `op(x, …)` evaluated once by `leaf_env` and
sliced per step by the fold — the same trick the projection lift
already used for `matmul`/`linear`, generalised.

This is deliberately *not* "add `layer_norm`/`concat` to the
renderer": those need row-wise reductions / multi-tensor joins that a
`[BLOCK]` elementwise fold cannot express.  Hoisting evaluates them
once on the full sequence where the existing whole-tensor path
handles them.

## What changed

### `packages/mlopt-discovery/src/mlopt_discovery/play.py`

- `_seq_commute_lift(t)` — the new pullout, consulted by
  `_seq_deindex` (with `pullout=True`, i.e. from `_seq_site` only;
  `_seqm_site` is unchanged) for any op node whose **direct first
  arg** is a bare per-step `select`.  Returns
  `(lifted_term, out_dim, index)` — fed through the ordinary
  operand path — or `None` to keep the generic decline.
- `_SeqLiftCtx` — frozen dataclass carrying the shared context:
  the select's child `c`, normalised step axis `d0`, step index,
  sliced/child ranks `rs`/`r`, the child's `shape_of` tuple, and
  `up`, the sliced-axis → child-axis map (skips `d0`).
- Per-family handlers, dispatched through `_SEQ_AXIS_LIFTS`:
  - `_seq_lift_transpose` / `_seq_lift_permute` — pure axis
    permutations: sliced dims are re-based into child coords
    (`up`) and the step axis is re-inserted (kept in place for
    transpose, appended last for permute so the outer `select`
    reproduces the sliced permute).
  - `_seq_lift_softmax` (`softmax`/`log_softmax`) — normalises along
    ONE axis; commutes iff that axis is not the step axis.
  - `_seq_lift_norm` (`layer_norm`/`rms_norm`) — normalises over the
    LAST `len(normalized_shape)` axes; commutes iff the step axis
    sits strictly before that trailing block **and** the block's
    sizes equal the child's trailing dims (checked against
    `shape_of`).
  - `_seq_lift_reduce` + `_seq_reduce_dims` (`sum`/`mean`/`amax`/
    `amin`) — reductions commute iff no reduced axis is the step
    axis; `keepdim` decides whether the output step axis shifts
    (`out_dim = d0 − #{reduced child axes < d0}` when not).
  - `_seq_lift_range` (`slice`/`narrow`/`chunk`/`split`) — range
    restriction along one non-step axis; rank and positions kept.
  - `_seq_lift_unbind` — removes one non-step axis; positions after
    it shift down.
  - `_SEQ_PASSTHROUGH_OPS` (`to`/`clone`/`contiguous`/`alias`/
    `dropout`/`broadcast`/`float`/`type_as`) — axis-preserving
    casts/views; commute unconditionally, no dim bookkeeping.
  - `_seq_concat_lift` — the multi-arg analogue: `concat` lifts iff
    **every** argument is a bare `select` on the SAME child axis
    with the SAME index (never the joined axis), all children share
    rank, and every non-joined child axis agrees in extent.  The
    lifted concat joins the children along the remapped concat axis.
    Mixed axes, mixed indices, a select ON the joined axis, or any
    non-select arg all decline — a single outer `select` cannot
    reproduce them (the both-args-selects boundary, generalised).
- `_seq_lift_finish` — the shared tail every handler feeds: the
  lifted term must be `select`-free (`_seq_select_free` — a surviving
  `select` would freeze one step's slice into the once-evaluated
  operand) and must type via `shape_of` with `out_dim` in range —
  the same contract the projection lift keeps.
- `_seq_argn_attrs` — positional `argN` attr spellings decline
  outright: they would alias the dims being remapped in child
  coordinates, so the lift refuses rather than guess.

### Commutation laws relied on (each verified, not assumed)

| op family | law | side condition |
|---|---|---|
| transpose/permute | `P(select(x,d,i)) = select(P'(x), D, i)` | step axis re-based through the permutation |
| softmax/log_softmax | mixes only along `dim` | `dim ≠ step axis` |
| layer_norm/rms_norm | normalises trailing `len(ns)` axes | step axis strictly before the block; block sizes match child's trailing dims |
| sum/mean/amax/amin | reduces listed axes | no reduced axis is the step axis |
| slice/narrow/chunk/split | ranges one axis | that axis ≠ step axis |
| unbind | drops one axis | dropped axis ≠ step axis |
| passthroughs | identity on axes | none |
| concat | per-arg slice then join = join then slice | all args selects, same child axis + index, joined axis ≠ step axis |

## Verified

- Eight new `TestSeqScanSites` tests in `tests/test_discovery_play.py`,
  each checking fold-equals-chain via `seq_scan_denotation` /
  `handler_denotation` against direct torch evaluation (fp64):
  - `test_seq_sites_lift_softmax` — softmax on a non-step axis.
  - `test_seq_sites_lift_layer_norm` — `layer_norm` over the
    feature axis while the step axis precedes the normalised block.
  - `test_seq_sites_lift_transpose_permutes` — transpose and permute
    remapping, including the step axis landing last.
  - `test_seq_sites_lift_reduction_and_ranges` — `sum`/`mean` with
    and without `keepdim`, plus `slice`/`narrow`-style ranges on
    non-step axes.
  - `test_seq_sites_lift_concat_all_selects` — concat of per-step
    selects on the same axis/index.
  - `test_seq_sites_lift_passthrough` — `clone`/`to`-style
    axis-preserving ops.
  - `test_seq_commute_lift_declines_unsafe` — reductions over the
    step axis, softmax over the step axis, `layer_norm` whose block
    covers the step axis, `unbind` of the step axis, mixed concat
    selects: all keep the generic decline.
  - `test_seq_sites_decline_unliftable_step_ops` — ops outside every
    law (e.g. `reshape`-class shape changes, selects nested deeper
    than a direct arg) still decline honestly.
- `tests/test_discovery_play.py -k seq`: **32 passed**;
  `test_discovery_play.py` + `test_discovery_genkernel.py`:
  **249 passed, 1 skipped** under `scripts/pytest-capped.sh`
  (serial, `-m "not e2e and not slow"`).
- `ruff check` / `ruff format --check` clean on
  `packages/mlopt-discovery/src/mlopt_discovery/play.py`; `ty check`
  clean on the touched file.
- Complexity: the lift is split into per-family handlers
  (`_seq_commute_lift` 12, `_seq_concat_lift` 31,
  `_seq_lift_*` ≤ ~10 each).  `scripts/radon_ratchet.py` was already
  red at HEAD (dozens of pre-existing `> 11` functions including the
  prior seq helpers); this change adds `_seq_commute_lift` (12) and
  `_seq_concat_lift` (31) to that list — a baseline `--update`
  decision for the author, not a new class of violation.

## Corpus evidence — honest census

A zoo re-census after the lift (all workloads, `to_ir` + seq
collectors + bare-leaf spine probes) shows the same principal
failure families as before: `deindex-None` for AdaLNBlock,
MLPMixerBlock, DeltaRuleBlock, TimeCondConv, RWKVMixBlock,
SkipInitNet, AdaGNConv; `shape/extent` and `skeleton-mismatch` for
MambaSelectiveScan, GatedLinearAttention, GatedDeltaRule,
RealGatedLRU.  The blocker ops still named in failures include
`zeros`, `mul`, `sub`, `unsqueeze`, `matmul`, `squeeze`, `linear`,
`concat`, `slice`, `sum`, `sqrt`, `square`, `sigmoid`, `div`, `to`.

**No new zoo site was minted.**  This mirrors the previous report's
finding and is expected: the complex spines carry *several*
independent unsupported ops (`zeros`/`arange` constants, `sqrt`,
`square`, nested non-direct selects, contracted-axis projections,
both-arg selects), so closing one boundary does not unblock the
whole spine.  The win is real at the collector/denotation level —
spellings that are elementwise-except-for-axis-safe-ops now mint —
but the zoo's recurrence modules either already minted via
hand-hoisted spellings or are blocked downstream by other gaps.
DeltaRuleBlock/GatedDeltaRule remain `seqm` territory, unchanged.

## Remaining honest boundaries

- **Ops with no commutation law** — `reshape`/`view`-class shape
  changes, `matmul`-family ops beyond the projection lift's cases
  (contracted-axis selects, rank-≥3 right operands, both-arg
  selects), `arange`/`zeros`-style constant producers, and anything
  not listed in `_SEQ_AXIS_LIFTS`/`_SEQ_PASSTHROUGH_OPS` keep the
  generic decline.
- **Selects not DIRECT first args** — a `select` nested inside an
  `add`/`sigmoid` under the lifted op still declines; the lift only
  fires on `t.args[0]` being a bare `select`.
- **`argN` positional attrs** decline rather than alias dims.
- **`concat` is all-or-nothing**: one non-select arg, mixed select
  axes/indices, or a select on the joined axis declines the whole
  concat.
- **Softmax/log_softmax over the step axis**, **reductions
  including the step axis**, **norm blocks covering the step
  axis**, **unbind of the step axis** — all verified declines, not
  guesses; lifting them would be a correctness bug.
- The renderer is untouched: lifted operands are whole-tensor terms
  the existing elementwise fold already knows how to slice.  If a
  future op cannot be expressed even as a hoisted whole-tensor
  term (needs new Triton grammar *inside* the step), that remains a
  renderer project, not a collector one.

## Files touched

- `packages/mlopt-discovery/src/mlopt_discovery/play.py` —
  `_seq_commute_lift` + `_SeqLiftCtx` + `_seq_argn_attrs` +
  `_seq_lift_finish` + `_seq_lift_attrs` + seven `_seq_lift_*`
  handlers + `_seq_reduce_dims` + `_SEQ_AXIS_LIFTS` +
  `_SEQ_PASSTHROUGH_OPS` + `_seq_concat_lift`; `_seq_deindex`
  consults it under `pullout`.
- `tests/test_discovery_play.py` — eight new
  `TestSeqScanSites` tests (softmax, layer_norm, transpose/permute,
  reductions+ranges, concat, passthrough, unsafe declines,
  unliftable declines).
- `docs/reports/seq_deindex_boundaries2.md` — this report.

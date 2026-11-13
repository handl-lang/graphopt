# `_seq_deindex` bare-`Param` leaves — resolved, not a blocker

Follow-up to the grammar-gap note in `t037_resweep_frontier.md`:
"`_seq_deindex`'s bare-`Param` decline means recurrences with
projections *inside* the step … are unreachable by the seq tiers —
a grammar gap, not a bug; the carrier lift is their current owner."

**Verdict: implementable, and implemented.** The decline was *two*
independent grammar gaps, neither a semantic blocker:

1. **Bare leaves.** `_seq_deindex` returned `None` for every leaf
   that was not a `Const` scalar — including `Param`/`Var`, which
   are just terms the fold can bind whole. They are loop-invariant:
   the same value serves every step, so they need no step axis.
2. **In-step projections.** `matmul`/`linear` over a per-step
   `select` cannot render in the seq kernel (a flat elementwise
   `[BLOCK]` fold — `_TRITON_BINARY`/`_TRITON_UNARY` have no matmul).
   But the projection *commutes with the slice* on non-contracted
   axes, so the collector can pull the `select` outward and bind the
   operand to the **hoisted** sequence `matmul(x, W)` — evaluated
   once by `leaf_env`, sliced per step by the fold.  That is exactly
   the spelling a hand-hoisted recurrence (`_SeqChunkMod`) already
   used; the collector now produces it mechanically.

## What changed

### `packages/mlopt-discovery/src/mlopt_discovery/play.py`

- `_seq_deindex` gains `consts`/`pullout` keyword arguments (both
  opt-in; `_seqm_site` calls without them and is byte-identical).
  - `consts` (a term→meta dict): `Param`/`Var` leaves mint
    `"P"-prefixed metas recorded in `consts`, never in
    `operands`/`events` — no step axis, no index-order obligation.
  - `pullout`: before the generic arg recursion, a `matmul`/`linear`
    node whose DIRECT arg is a bare `select` is rewritten by
    `_seq_projection_lift` into `(lifted_term, out_dim, index)` and
    mints an ordinary sequence operand on the lifted term.
- `_seq_projection_lift` — the commutation check, verified against
  `shape_of` + torch semantics:
  - `matmul(select(c,d,i), b)`, `b` rank ≤ 2, `d` not the contracted
    last axis of `c` → `select(matmul(c,b), d, i)` (`out_dim = d0`;
    batch and `m` dims keep position).
  - `matmul(a, select(c,d,i))`, `c` rank-2 `(k,n)`, `d` the `n` axis
    → `select(matmul(a,c), last, i)`.
  - `linear(select(x,d,i), W[, b])`, `d` not `x`'s last (contracted)
    axis → `select(linear(x,W,b), d, i)`.
  - The lifted term must be `select`-free (`_seq_select_free`) — a
    surviving `select` would freeze one step's slice into the
    once-evaluated operand.  Selects in *both* matmul args, selects
    nested deeper than a direct arg, rank-≥3 right operands, and
    every other op keep the generic decline.
- `_seq_site` passes `consts={}`/`pullout=True`, then checks every
  const's `shape_of` broadcasts into the fold's state shape
  (`_seq_bcast`, right-aligned) — a const that cannot is declined
  rather than emitted as a spec that raises on call.  Const metas
  are recorded in the `applyd` attr dict as `"const": (names…)` and
  bound in `leaf_env` like any other metavar.

### `packages/mlopt-discovery/src/mlopt_discovery/genkernel.py`

- `_scan_info`: for `kind="seq"` specs, metas named by the spec's
  `"const"` attr are partitioned out of `info["seq"]` into
  `info["const"]`; `info["params"]` still carries every meta + `H`
  in spec order, so handler `args`/binding signatures are unchanged.
- `seq_scan_denotation`: per-step `env` now also binds each const
  meta to its whole value (`vals[c]`) — the same tensor every step.
- `_seq_triton_source` / `_seq_args` / `_seq_binding`: const operands
  ride as extra pointer args after the sequence operands; the host
  broadcasts each into the state space once (`broadcast_to(state)`)
  and flattens it alongside `h`; the kernel loads them once before
  the `t` loop — never per row.  Arg order is `seq…, const…, h, out`.

## Verified

- `_BatchedChunkMod` (the T-037 module whose in-step `x @ w` /
  `x @ wr` projections + `sigmoid(log_decay)` decay declined) now
  mints exactly one `kind="seq"` site on its exported IR:
  `applyd(aff_diag(("sigmoid","P1"), ("mul",("sigmoid","Q1"),"Q2")),
  "H", {kind:seq, axis:0, const:("P1",)})` — the decay keeps its
  `sigmoid` op in the skeleton with `P1` the bare `Param` leaf, and
  `Q1`/`Q2` bind `matmul(x, wr)`/`matmul(x, w)`.  `handler_denotation` reproduces
  the module to 1.7e-16 (fp64; the hoisted batched matmul vs per-step
  matmul rounding — not bit-identity, hence `< 1e-13` in the test).
- Synthetic collector cases (`tests/test_discovery_play.py`,
  `TestSeqScanSites`): projection+const lift with fold-equals-chain
  (`assert_close`, fp64), `linear` lift, contracted-axis decline
  (site emitted, `scan_spec` → `None`, no handler), and
  non-broadcasting-const decline (`sites == []`).
- Zoo census (all 65 workloads, `to_ir` + both collectors + a
  bare-leaf spine probe): the new grammar mints **no additional
  zoo site** — every zoo spine carrying bare leaves
  (RWKVMixBlock, SkipInitNet, AdaLNBlock, MLPMixerBlock) also
  contains non-elementwise ops (`linear`, `layer_norm`, `transpose`,
  `concat`, `sum`) that `scan_spec`'s elementwise renderer still
  declines.  The zoo's recurrence modules already minted sites via
  hand-hoisted spellings.  The win is real but narrow: spellings
  that are elementwise-except-for-projections.
- `_seqm_scan_sites` unchanged (no `consts`/`pullout` args passed);
  `tests/test_discovery_play.py` seq+seqm (24), `test_discovery_
  genkernel.py` (158), `test_batched_scan_lift.py` (41), and the
  zoo/meta/player slice (456) all pass under `scripts/pytest-capped.sh`.
  `ruff check`/`ruff format --check` clean on both touched packages.

## Remaining honest boundaries

- **Non-elementwise step ops other than `matmul`/`linear`** —
  `softmax`, `sum`, `layer_norm`, `transpose`, views other than the
  unsqueeze/squeeze lift, `reshape` — still decline: the `_gs` seq
  kernel is elementwise-only.  `_FlatChunkMod`'s `reshape` and the
  zoo's `linear`-bearing spines stay out.
- **Selects not DIRECT args** of the projection (nested inside an
  `add`/`sigmoid` under the `matmul`) keep the generic path — the
  spec then carries a per-step `matmul` meta expression that
  `scan_spec` declines.
- **Both-args selects** in one `matmul`, **rank-≥3 right operands**,
  and **selects on contracted axes** all decline — verified cases,
  not guesses.
- The pipeline still delivers `_BatchedChunkMod` through the
  saturated carrier (`lowering == "generic"` unchanged — the e-graph
  extracts a single-leaf `applyd` spine before the seq tier's raw-IR
  collector is consulted).  The seq claim is real at the collector/
  denotation level; whether a given pipeline run routes it is the
  game layer's call, unchanged by this work.

## Files touched

- `packages/mlopt-discovery/src/mlopt_discovery/play.py` —
  `_seq_meta`/`_seq_const_meta`/`_seq_select_free`/
  `_seq_step_select`/`_seq_projection_lift`/`_seq_bcast` helpers;
  `_seq_deindex` `consts`/`pullout` params; `_seq_site` const
  partition + broadcast check + `"const"` attr.
- `packages/mlopt-discovery/src/mlopt_discovery/genkernel.py` —
  `_scan_info` const partition, `seq_scan_denotation` const env,
  `_seq_triton_source`/`_seq_args`/`_seq_binding` const operands.
- `tests/test_discovery_play.py` — 5 new `TestSeqScanSites` tests +
  `_proj_recurrence` fixture.
- `tests/test_batched_scan_lift.py` — `test_batched_chunk_seq_site_
  fold_matches`; refreshed the two stale "grammar cannot abstract"
  docstrings.

Note: the worktree carried unrelated uncommitted changes before this
work (`meta_arena.py`, the `_triton_expr_h` clamp grammar +
`test_discovery_genkernel.py` clamp tests, `docs/reports/
applyd_route.md`); this change is confined to the seq-deindex/scan
regions above.

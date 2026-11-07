# Ledger hygiene — T-011…T-015 (2026-11-07)

Five myco ledger items landed together.  Per-task record; the code
changes are uncommitted on each repo's `main` for the parent to
commit.

## T-011 — plan 0002 leftovers: **decided, no shared kernel**

Phase E asked whether the recognizer families share a kernel:
law-side e-class bindings vs lowerer-side extracted terms.

Audit (post-split names — the plan's `xcarrier`/`om_*` map to
`mlopt_carriers`):

- **Law side** — `xcarrier._check_chunk_apply*` /
  `_check_split_*` operate on e-graph `bound` dicts (pattern
  variables `a`/`b`/`h`, `$attr:D`) and answer *admission*: is this
  rewrite shape-true?  Output is bool/derived attrs.
- **Lowerer side** — `om_lower._slice_index`/`_sliced_gather`/
  `_qk_parts`, `omd_lower._select_index`/`_part_gather`,
  `scan_lower._select_index`, `decayed_lower` reuses
  `_sliced_gather`.  These decompose concrete `Op` terms into
  `(base, kind, dim, index, nparts, part_size)` tuples that become
  *plan rows* for the batched schedule.

The substrates genuinely differ: different input representation
(bound dict vs term tree), different output (proof vs decomposition),
different consumer (gauntlet vs executor plan).  No bit-identical
core modulo output mapping exists — **the parallel is intentional**,
exactly the plan's fallback ("document the intentional parallel").
Revisit only if a shared kernel emerges.

Plan-dict typing stays **dict-of-any**: executor plans
(`build_om_plan` &c.) are `dict` vocabularies read by `.get()` at
many sites; a dataclass migration buys nothing the ports layer
needs today.

## T-012 — vendored probe code: **fixed**

- `bench/drivers/probes/wholemodel_gap_probe.py` — deleted the
  "vendored fixtures" block (verbatim `_ChunkedRetention` +
  `_time_module` copies kept "while the tree moves" — the refactor
  landed long ago).  Both now import from `mlopt_discovery.zoo` /
  `mlopt_discovery.play`.
- `bench/drivers/probes/retnet_baseline_probe.py::_torchscale_arm` —
  the ~45-line `inner_mask` block was verbatim torchscale
  (`RetNetRelPos.forward`'s chunkwise branch), inlined to dodge the
  `timm` import.  Now calls the library itself:
  `RetNetRelPos(SimpleNamespace(decoder_embed_dim=2,
  decoder_retention_heads=1, recurrent_chunk_size=c))` with the
  decay buffer patched to gamma — verified the official mask tuple
  is **bit-identical** to the inlined math (max abs diff 0.0 on all
  four members).  Scratch-deps line gains `timm`.
- `bench/drivers/probes/awq/gen_manifest.py` "vendored" kernels are
  *our own* codegen output shipped into `lib-triton-qgemv` — a
  declared artifact mechanism, not third-party copying.  Left.
- kernelopt `bench/limits/registry.py` — was a full copy of
  graphopt's registry (name→spec index, lazy loaders, runner-facing
  `specs`/`names`/`for_case`/`availability`/`build`).  Nothing in
  kernelopt calls that machinery (`bench.discovery.kernel` uses
  `bench.limits.triton.kernel_builder` directly).  Trimmed to the
  contract slice the `triton` folder actually imports:
  `BaselineSpec`/`BaselineCtx`/`SkipBaseline` — the folder's
  `SPEC`/`build` stays a drop-in baseline if a runner lane lands.

## T-013 — import-time domain registration: **fixed, explicit**

- kernelopt `src/mlopt_kernel/play.py` — all module-level side
  effects (`register_part`/`register_legal`, `DOMAINS["kernel"] =
  spec_domain(...)`, five `register_player` seats) moved into
  `register()` — idempotent (gamekit seams overwrite by name).
  `main()` calls it; importing the module now mutates nothing.
  Call sites updated: `scripts/train_kernel_prior.py`,
  `bench/drivers/policy/transfer.py`, `tests/test_transfer.py`,
  `tests/test_kernel.py` (4 seats).  AGENTS.md updated.
- graphopt `mlopt_discovery/bilinear_game.py` — the same pattern
  (five module-level `register_player` calls fired by
  `import bilinear_game`).  Now `register_players()`, called by
  `mlopt_discovery.play` right after the `DOMAINS` table — the
  composition root seats explicitly.
- Boundary note: the `register_part`/`register_legal`/
  `register_rank` *feature-vocabulary* calls in `open_game.py`,
  `player_arena.py`, `meta_player.py`, `play.py` itself &c. stay
  module-level — they are each domain module's own parts vocabulary,
  consumed only through play's deliberate import of that module
  (the same composition-root relationship, one level down).  The
  ledger item named domain/player *seating*; converting ~34 part
  call sites is mechanical churn with no seam improvement — left,
  flagged here if a future pass wants uniformity.

Verified: `import mlopt_kernel.play` leaves `"kernel"` out of
`play.DOMAINS` until `register()`; `import bilinear_game` leaves the
bilinear seats unregistered until `mlopt_discovery.play` imports.

## T-014 — pinned test counts: **fixed where they pin the bank**

- kernelopt `tests/test_kernel.py::test_bank_shape` — was the exact
  ordered 28-name `KERNEL_RULES` census (every bank addition churned
  it).  Now invariant-based: unique names, no rule mints an
  undeclared kernel op, and every declared `FUSED_KERNEL_OPS` /
  `SCHEDULED_KERNEL_OPS` member is minted by some rule (vocab↔law
  cover both directions).
- graphopt `tests/test_law_provenance.py` — `== 70` promoted-rule
  census → derived `len(promoted.rules) == len(shipped_rule_names)
  > 0` (equality was already the real invariant; the literal only
  pinned the count).
- Already-derived pins left alone: `len(ALL_RULES)` comparisons in
  `test_discovery_{pipeline,gauntlet,experiments}`; the `== 11`
  pins are `lawdata.CANDIDATE_LAWS` — a fixed measurement fixture,
  not the shipped bank; `test_kernel_priorart.py` was already
  invariant-shaped (unique/tagged/disjoint).
- `test_discovery_experiments.py` `len(ALL_RULES) + 23` — the `+23`
  counts the grammar-synthesis family's own candidates (stable
  under bank growth); left.
- AGENTS.md counts ("13 machine-admitted laws", suite sizes) are
  prose snapshots of measured runs, not assertions — left as
  documentation, refreshed when the numbers are re-measured.

## T-015 — radon baseline: **regenerated**

`scripts/radon_ratchet.py` was failing on clean `main` — the baseline
predated the T-016/018/019–21/024/026 kernel-prim landings (the
ledger's `bilinear_game.py` flag was stale; current drift is
`genkernel.py`, `meta_player.py`, `play.py`).  Ran `--update`:
3,806 entries; ratchet green.  Largest single jump:
`genkernel._attr_match` cc 20 (new, over the 11 threshold —
attr-site gating from T-026) and `triton_bindings` 7→16,
`meta_player.main` 4→21.  All from landed, measured feature work —
baselined as the ratchet's documented "intentional change" path.
Worth an eyeball on `_attr_match` if a future refactor pass comes.

## Test/lint status

- kernelopt: `pytest tests/ -q` — **175 passed**; `ruff format
  --check` clean on touched files; `ruff check src scripts` shows 7
  pre-existing errors in untouched `src/mlopt_kernel/search.py`
  (present on clean main); `ty check` fails on pre-existing
  unresolved `lib_kernel_ir` import (sibling-repo path dep —
  environmental, untouched by this work).
- graphopt: `pytest -m "not slow"` on
  `test_law_provenance`/`test_discovery_bilinear_game`/
  `test_strassen_mcts`/`test_discovery_play`/`test_public_api` —
  **136 passed**; `ruff check`/`format`/`ty` clean on
  `packages scripts examples`.

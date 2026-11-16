# discoverykit Phase 2 — graphopt-side verification

**Date:** 2026-11-14 · **Plan:** 0034 (`project/plans/pl-0030`) ·
**Scope:** graphopt side only.  The discoverykit repo is done
(`handl-lang/discoverykit` @ `8016abc`, coverage floor
`fail_under=21`, measured 22.92% branch).  This report verifies the
consumer side: shim completeness, the `mlopt_discovery` coverage
ratchet, and Phase-3 readiness.  Nothing was committed.

## Dependency pin

`pyproject.toml` + `uv.lock` pin `discoverykit` as a git+tag dep:

```
discoverykit = { git = "https://github.com/handl-lang/discoverykit.git",
                 subdirectory = "packages/discoverykit", tag = "v0.1.0" }
```

Installed in `.venv` at **`b1a3996`** (= tag `v0.1.0`, the Phase-1+2
extraction commit).  discoverykit HEAD is `8016abc`, one commit ahead —
that commit touches only `pyproject.toml` coverage config + docs
(`git show 8016abc --stat`: AGENTS.md, README.md, pyproject.toml), so
the installed pin is functionally current for imports.  A relock to
`8016abc` is cosmetic but harmless to do at the next `uv sync`.

## 1. Shim completeness — verified

Every moved module resolves through its `sys.modules`-swap shim to the
`discoverykit` module (`mod.__name__ == "discoverykit.<mod>"`).  All
**29 top-level modules + `lawdata/` + `play/registry.py`** imported
cleanly in the graphopt venv:

```
antiunify bilinear_game bounds cellstore coherence coherence2 combinator
corpus_cov domain_spec dspec_synth evidence form_propose genkernel
handler_synth handl index_abstract law_order lemma_cert loop_propose
machine_pack macro_moves meta_arena meta_player meta_prior
model_domains object_synthesis pay_gate proposal_core verifier
lawdata  lawdata.laws  play.registry
```

Result: **0 failures**.  Submodule imports through the swapped parent
work (`mlopt_discovery.lawdata.laws` → `discoverykit.lawdata.laws`;
`mlopt_discovery.play.registry` → `discoverykit.play.registry`) because
the shim replaces the package object itself, so `__path__` resolves
against the discoverykit tree.

Spot-check: `mlopt_discovery.antiunify.__name__` is
`"discoverykit.antiunify"` — attribute access, `is` comparisons and
`__module__` strings behave as the shim contract intends.

### Deferred modules — still real, not broken shims

These remain full modules under
`packages/mlopt-discovery/src/mlopt_discovery/` (verified via
`__file__`); none contains a "Moved to" marker:

| Module | Reason deferred |
|---|---|
| `fragment_game`, `handl_egraph`, `handl_engine` | vocabdir dependency |
| `proposal`, `oracle`, `vocab`, `model_corpus`, `state_reindex` | torch-bound |

Also still real (torch/graphopt-bound stay set): `census`, `impact`,
`intake/`, `pipeline`, `play/{cli,domains,fragment,gen,meta_joint,
players,probes,search,sites,torch_arm}`, `player_arena`, `zoo`,
`training`, `open_*`, `meta_game`, `learned_propose`, `prim_mint`,
`shipped_primitives`, `vocabdir`, `emit`, `engine`/`players`/`scoring`/
`advantage` (gamekit shims from the earlier split — unrelated to this
phase).

## 2. Coverage ratchet — status: needs a green full run to re-measure

Config (`pyproject.toml` `[tool.coverage.run]`): `source` includes
`mlopt_discovery`; `discoverykit` is **not** a coverage source, so the
29 moved modules + `lawdata.laws.*` no longer count toward the
graphopt union at all — they are measured on the discoverykit side
(floor `fail_under=21`, 22.92% branch there).

What replaced them in `mlopt_discovery` are ~15-line shim files whose
only executable lines run at import; the `__main__` re-dispatch guard
is already in `exclude_also`.  Focused coverage run (54
discovery-focused test files, `-m "not slow"`, 19m24s): **every shim
file reports 100%** except `model_domains.py` at 0% — it is referenced
only by `scripts/verify_vocab_models.py` (outside `tests/`), so no test
imports it; its 4 uncovered lines are noise against a ~16k-statement
union, and the extraction *removed* its previously-uncovered bulk.

**Honest caveat:** the union floor could not be re-measured in this
session.  `scripts/gates.sh` `MLOPT_GATES_LANE=full` does not finish
inside 30 min on this host (serial or `-n 2`, per AGENTS.md), and the
suite is currently red for unrelated reasons (below).  The configured
floor is `--include="*/mlopt_discovery/*" --fail-under=94` (comment
records a 95% post-split measurement, 2026-10-21; the "99% ratchet"
wording in AGENTS.md predates the phase-4 re-baseline).

Direction of the shift is favourable: removed code was part of a ≥95%
union (mostly well-tested), stay-side code is untouched, and shims are
~100% covered on import — expect the re-measured union to be
preserved-or-better, but **the tighten-never-lower rule requires a
full `coverage run -m pytest tests/ -q` on a green tree before the
floor is touched**.  Do not raise `fail-under` off this focused run.

## 3. Sanity — all green

```
import discoverykit                        # site-packages, tag v0.1.0
import mlopt_discovery.antiunify           # -> discoverykit.antiunify
import mlopt_discovery.verifier            # -> discoverykit.verifier
import mlopt_discovery.play                # OK
python -m mlopt_discovery.play --help      # full CLI usage printed
```

## 4. Focused pytest — 22 failures, all attributable to concurrent dep drift, NOT the extraction

Run: 54 discovery-focused test files, `-m "not slow"`, under coverage:
**2018 passed, 22 failed, 6 skipped** (19m24s).

Failures cluster into three groups, none touching shim mechanics:

- **eqsat cost-model relock** (`b3ac192`/`762f85e`, landed *during*
  this session): eqsat `d51acaa` "cost: honest unpriced-op fallback +
  view-op zero-flop routing" changed `mlopt_core.cost` semantics
  (leaf enode 1.0→0.0 flops, `_UNPRICED_FLOPS`).  Victims:
  `test_discovery_proposal.py` (7 — `_cost_delta` now 16==16),
  `test_discovery_enumerate.py` (1 — `useful` now False),
  `test_learned_propose.py::test_default_seed_bank_is_machine_shipped`
  (MACHINE_SHIPPED/provenance drift), plus the `coherence`/`lemma_cert`
  verdict-note failures which depend on the same law/cost surface.
- **`weight_norm` vocabdir drift** (`67d6da8`, kit-torch
  `_weight_norm` bind): `test_vocabdir.py` (4),
  `test_overnight.py` prim-vocab pins (2), `test_prim_mint.py` (1) —
  `check_vocab` reports `ops.yaml:ops:weight_norm` stale; the committed
  vocabdir artifact needs regeneration by whoever owns that lane.
- **CUDA/batched-scan work** (`b8ccc72`, `1831217`):
  `test_batched_scan_lift.py` (2) — `IRModule` lowering contract
  changed under the new alias/KV-cache work.

**Why this is not an extraction regression:** I diffed the pre-move
sources (`git show 22789d8~1:...`) against the installed discoverykit
modules for `lemma_cert`, `coherence`, `proposal_core` — the only
differences are import-path rewrites (`mlopt_discovery.X` →
`discoverykit.X`, incl. lazy function-level imports per the seam
contract).  Moved code is verbatim; the sys.modules shim preserves
object identity, so behaviour is identical by construction.  The
failing assertions are cost/vocab/lowering semantics owned by eqsat,
kit-torch and mlopt-cuda — repos this task must not touch.

**Action for the parent agent:** these 22 failures need owners —
likely the agent that landed the eqsat relocks (`b3ac192`/`762f85e`)
and the weight_norm/CUDA lanes.  They will also block the full-lane
coverage re-measurement above.

## 5. Phase-3 readiness assessment

Ready, with caveats:

- **Seam is proven.**  graphopt consumes discoverykit purely through
  pinned git+tag dep + sys.modules shims; `play --help`, all 30 shim
  paths, and `lawdata.laws`/`play.registry` submodule resolution all
  work.  kernelopt can seat the same way (its imports already resolve
  through the shims per the phase-1 report).
- **Before Phase 3:** (a) relock `v0.1.0` → `8016abc` or cut `v0.1.1`
  so the pin carries the coverage-floor commit (cosmetic); (b) get a
  green `MLOPT_GATES_LANE=full` run to re-measure the
  `mlopt_discovery` union and re-baseline `fail-under` if it moved;
  (c) decide whether `model_domains`' 0% shim line merits an
  `exclude_also` or a test import — negligible either way.
- **Deferred set is stable.**  The 8 deferred modules
  (vocabdir/torch-bound) plus the torch play arms remain the honest
  graphopt-side residue; a second domain seating on discoverykit does
  not require moving them.

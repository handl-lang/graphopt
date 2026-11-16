# play.py → play/ package split

`packages/mlopt-discovery/src/mlopt_discovery/play.py` (5,002 lines)
is now the package `mlopt_discovery/play/` (5,735 lines across 12
files — the growth is per-module docstrings/import blocks and the
re-export surface in `__init__.py`).  The split is by section, not by
dependency guesswork: an AST pass over the old file produced a near
acyclic graph, and the only cycle (`gen` ⇄ `probes` over
`_time_module`/`_has_triton`) was broken by moving `_has_triton` into
`gen.py` next to `_time_module`.

## Layout

| module | lines | contents (old play.py lines) |
|---|---|---|
| `__init__.py` | 593 | module docstring (unchanged — it is the CLI `--help` text via `_pkg_doc`), explicit re-exports of all 141 top-level names, the import-alias surface (`play.pa`/`fg`/`mp`/`og`/`bg`/`ma`/`lawdata`/`training`/`engine`), `__all__` (unchanged: `DOMAINS`, `Domain`, `SearchBoard`, `play`, `spec_domain`), and `_PlayModule` |
| `__main__.py` | 6 | `python -m mlopt_discovery.play` entry |
| `registry.py` | 111 | `Domain`, `spec_domain`, `play`, `deliver`, empty `DOMAINS` (107–184) — the frozen kernelopt seam; imports only `gamekit` + `dataclasses` |
| `sites.py` | 2,306 | site collectors: fusion/reduction/decayed/scan + the whole seq/seqm/cumsum lift machinery + `TIER_SITES`, `_mint_tiers`, `_flat_ops` (576–2856) |
| `meta_joint.py` | 76 | `_meta_board`, `_joint_cases`, `_joint_board` (187–243) |
| `gen.py` | 1,186 | gen domain cases/board/probes, the zoo intake tail (`_case_tail`, `_zoo_case`, `zoo_probe`, …), `TRITON_REFEREE_LOG`, `_time_module`, `_has_triton` (2859–4015 + 4151–4155) |
| `probes.py` | 263 | `scan_probe`, `decay_probe`, spelled/carrier IR builders, shape tables (4018–4271 minus `_has_triton`) |
| `torch_arm.py` | 346 | torch module zoo, `_torch_cases`/`_torch_board`, `_torch_deliver` + runner arm (246–573) |
| `search.py` | 257 | `_SearchState`, `SearchBoard`, search pool/cases/board/legal/state/act, `SEARCH_SPEC` (4276–4501) |
| `fragment.py` | 170 | fragment pool/cases/board/state/act, `FRAGMENT_SPEC`, `_fragment_scripted` (4506–4653) |
| `players.py` | 56 | `_random_policy`, `_scripted_search` (4656–4694) |
| `domains.py` | 88 | the `DOMAINS.update({...})` literal + `bg.register_players()` — the composition root (4697–4753) |
| `cli.py` | 277 | `_arms`, `_table`, `_extras`, `_kernel_probes`, `main` (4756–4998) |

Dependency order (acyclic): `registry ← sites ← gen ← {probes,
torch_arm}`; `search ← players`; `domains` and `cli` sit on top.
`domains.py` populates `registry.DOMAINS` in place, so
`play.DOMAINS["kernel"] = …` (kernelopt's `register()`) mutates the
same dict object as before.

## Compatibility measures

- **`python -m mlopt_discovery.play`** works via `play/__main__.py`.
  `prog="play.py"` pins the usage line (a `-m` package would
  otherwise print `__main__.py`), and `description=_pkg_doc()` keeps
  the original module docstring as the `--help` text.
- **Namespace**: `__init__.py` explicitly re-exports every top-level
  name the monolith defined (141 names, `as`-aliased for ruff) and
  copies the remaining import aliases, so `play._seq_scan_sites`,
  `play._mint_tiers`, `play._time_module`, `from
  mlopt_discovery.play import _time_module`, `play.pa`, … all resolve
  to the *same objects* as before.
- **Monkeypatch semantics**: tests and bench drivers do
  `monkeypatch.setattr(play, "_claim_cost_gate", …)`,
  `play._mint_tiers = spy`, etc.  In a single module that repointed
  the one global every caller resolved; naively splitting would have
  silently broken all of them (the patch would land on the package
  while submodule callers kept the old binding — e.g.
  `tests/test_discovery_play.py:522,571,2211`,
  `tests/test_discovery_genkernel.py:271,351,462,2750,3288,3300`,
  `bench/drivers/probes/attr_elementwise_probe.py:107`).  `_PlayModule`
  overrides `__setattr__`/`__delattr__` on the package module object
  and forwards every write to *all* submodules holding that name —
  including names a submodule imported from a sibling (`_mint_tiers`
  is bound in both `sites` and `gen`), which reproduces the old
  single-namespace semantics exactly.  Verified: patching
  `play._mint_tiers`/`play._claim_cost_gate` reaches `gen`/`sites`
  globals and restores cleanly.
- **Relative imports**: the only intra-package relative import in
  function bodies (`from . import domain_spec` in `main`) became
  `from .. import domain_spec` in `cli.py`; all other lazy imports
  were already absolute `mlopt_discovery.*`.
- **Import weight**: unchanged — the same sibling modules
  (`meta_arena`, `meta_player`, `open_game`, `player_arena`,
  `bilinear_game`, `fragment_game`, `lawdata`, `training`,
  `bounds`) load at package import, and torch/onnx/cuda stay
  function-level lazy imports.  `registry.py` itself is light
  (gamekit + dataclass only), though `import
  mlopt_discovery.play` still pulls the whole package — same as the
  old module did.

## Gates

- `pytest tests/test_discovery_play.py -q -m "not slow"`: **115
  passed** (unchanged file).
- `pytest tests/test_discovery_genkernel.py test_discovery_bounds.py
  test_domain_spec.py test_discovery_player_arena.py
  test_batched_scan_lift.py test_discovery_pay_gate.py
  test_discovery_open_game.py test_public_api.py -m "not slow"`:
  **669 passed, 2 skipped**.
- `pytest tests/test_discovery_model_corpus.py
  test_om_decayed_kway_deliver.py test_discovery_open_prior.py
  test_discovery_open_learned.py -m "not slow"`: **44 passed**
  (22 deselected as slow).
- `python -m mlopt_discovery.play --help`: identical usage/options
  and docstring text; `--domain meta --train-cases 4 --eval-cases 2
  --budget 8` runs the full train/eval/table path.
- `ruff check` + `ruff format --check` on `play/`: clean.
- `ty check` (whole repo): clean — the explicit re-exports keep
  `domain_spec.py`/`model_corpus.py`'s `_play._x` accesses resolved.
- `vulture`: no findings in `play/`; `lint-imports`: 1 contract kept.

## complexity_baseline.json

`radon_ratchet.py` keys complexity by path, so the move orphaned all
62 `play.py::…` entries.  The baseline was **already failing at HEAD**
(113 pre-existing `rose to`/`new function` entries in
`kit-torch/torch_bridge.py`, `meta_player.py`, etc. — drift from
other in-flight work, untouched here).  For the moved code I dropped
the stale `play.py::` keys and recorded the *current* measured
complexity under the new `play/<mod>.py::` paths (139 entries) —
the code is byte-identical, so this only re-keys the measurement; it
also absorbs pre-existing un-baselined additions (`_seq_*` collectors,
`ProbeBounds` work) that were already failing as `play.py::` entries
before the move.  `play/` now contributes **zero** ratchet problems;
the remaining 113 are other files' drift for their owners.

## Notes for the parent agent

- `git status` before starting showed only
  `bench/drivers/mint/real_model_mint.py` modified (mint agent) —
  untouched.  The seam-sweep agent's module list was avoided; only
  `play.py` (clean) and new `play/` files were written, plus
  `scripts/complexity_baseline.json` (re-keyed play entries only).
- Nothing committed or pushed, per instructions.
- `kernelopt`'s `mlopt_kernel.play.register()` calls
  `play.spec_domain`/`DOMAINS` — both resolve to the same objects;
  `DOMAINS` is still a plain mutable dict.
- `play/__init__.py` is intentionally explicit-but-mechanical: the
  141-name re-export block exists because tests, `domain_spec.py`,
  `model_corpus.py`, bench drivers and scripts reach deep into
  `play._*` privates; shrinking it would break callers.

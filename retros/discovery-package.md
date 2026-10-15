# catopt-discovery — the law-discovery engine becomes a package

The `tools/law_*.py` scripts had quietly become a real subsystem: a
shape census, two proposal generators, a grammar enumerator, three
oracles, a ranking pipeline, a sqlite evidence store, a coherence
catalogue, an admission emitter, a meta-game RL prober, two corpus
generators and a real-`nn.Module` intake — ~20 kLOC across 19 modules,
cross-imported through `sys.path.insert(0, tools_dir)` hacks.  This
week the intake caught two genuine soundness bugs in shipped
lowerings, which settled the question of whether this is dev tooling
or a shipped engine: it is an engine, so it moved into the monorepo
as a sixth workspace package, `packages/catopt-discovery`
(`catopt_discovery`).

Reproduce / verify:

    .venv/bin/python -m catopt_discovery.pipeline --help
    .venv/bin/python -m catopt_discovery.coherence
    .venv/bin/python -m pytest tests/test_law_view_oracle.py \
        tests/test_law_pipeline_typed.py tests/test_law_serialize.py \
        tests/test_lemma_certificates.py tests/test_intake_defects.py

## 1. Module map

| old (`tools/`)                     | new (`catopt_discovery`) |
|------------------------------------|--------------------------|
| `law_pipeline.py`                  | `pipeline`               |
| `law_shape_census.py`              | `census`                 |
| `law_proposal.py`                  | `proposal`               |
| `law_shape_proposal.py`            | `shape_proposal`         |
| `grammar_proposal.py`              | `grammar`                |
| `law_evidence.py`                  | `evidence`               |
| `law_view_oracle.py`               | `oracle`                 |
| `law_impact.py`                    | `impact`                 |
| `law_vocab.py`                     | `vocab`                  |
| `law_lemma_cert.py`                | `lemma_cert`             |
| `law_intake.py`                    | `intake`                 |
| `law_coherence.py`                 | `coherence`              |
| `law_coherence2.py`                | `coherence2`             |
| `law_verifier.py`                  | `verifier`               |
| `law_emit.py`                      | `emit`                   |
| `law_meta_game.py`                 | `meta_game`              |
| `law_gap_targeted_gen.py`          | `gap_gen`                |
| `law_workload_gen.py`              | `workload_gen`           |
| `families.py`                      | `families`               |

Every module keeps its CLI: `python tools/law_X.py` becomes
`python -m catopt_discovery.<mod>`.  Thin `tools/law_*.py` shims were
kept **only** for the five entry points the README/retros invoke
(`law_pipeline`, `law_intake`, `law_shape_census`, `law_coherence`,
`law_evidence`) — each is a `main()` re-export; everything else moved
without a shim.

## 2. What the move surfaced

* **Dependencies.**  The package needs `catopt-core`, `catopt-torch`,
  `catopt-orchestrator` *and* torch — `impact` imports
  `catopt_orchestrator.optimize._lower_extracted` at top level, and
  `vocab`/`emit` reach orchestrator internals lazily.  The package was
  declared with all four deps.
* **The `bench` boundary.**  `impact`/`verifier` lazily import
  `benches.suites.correctness.law_bench.LAW_CASES` — `benches` is a
  repo-root package, not an installed distribution.  The engine now
  reaches it through `catopt_discovery.REPO_ROOT` (derived from the
  editable file location), keeping the same sys.path mechanism but
  computed honestly.
* **Artifacts stayed in `tools/`.**  `intake_corpus.json` and
  `intake_tensors.pt` did not move; `intake` resolves them through
  `catopt_discovery.TOOLS`.
* **Staying → moved coupling.**  `tools/train_rl_policy.py` and
  `tools/train_search_policy.py` (which stay) import `families` (which
  moved).  Their `import families` lines became
  `from catopt_discovery import families` — the only edits to staying
  files.  No moved module imports a stayed one.
* **ty debt.**  `tools/` was never type-checked; moving the code under
  `packages/` put it under `ty`, which reported 35 diagnostics — all
  fixed with real annotations/guards (a wrongly-typed `CorpusStats`
  field, `Op|Var` unions, `dict[str, Any]` kwargs bags, one
  `contextlib.suppress` replacing `except: pass`, and a `Callable`-hop
  for the verifier monkeypatch).  Zero `exclude`/`# ty: ignore`.
* **bandit.**  `packages/` is SAST-scanned; the moved code tripped six
  `B101` asserts and one `B110` try/except/pass.  All fixed in code
  (explicit raises / `cast` / `suppress`) — no config widening.
* **radon.**  The baseline only covered `packages/`; the move brought
  ~800 previously-unratcheted functions under the gate.  The baseline
  was regenerated (`--update`), pinning their current complexity —
  per the ratchet convention, not as an exception.

## 3. Config edits

* `pyproject.toml`: `catopt-discovery` added to `[project]`
  dependencies and `[tool.uv.sources]`; `packages/*` workspace glob
  picks the member up automatically.
* `[tool.ty.environment]`: `packages/catopt-discovery/src` added to
  `extra-paths`.
* `[tool.importlinter]`: `catopt_discovery` added to `root_packages`,
  to the core `forbidden` contract, and as the top `dependency layers`
  layer (it reads core/torch/orchestrator; nothing may depend back).
* `tools/complexity_baseline.json`: regenerated (+797 entries).
* Coverage **not** extended: `catopt_discovery` is not in
  `[tool.coverage.run] source` — the 100% floor is a separate staged
  effort (the engine is mostly CLI/reporting code; its test coverage
  is the law-focused subset, not the whole surface).
* `tools/runtime_types.sh` and `tools/mutmut.sh` keep their hardcoded
  package lists — discovery is deliberately not instrumented there
  yet, same staging decision as coverage.

## 4. What did not change

* The five shimmed CLIs still work as `python tools/law_X.py`.
* `tools/` keeps its dev-infra (`radon_ratchet.py`, `mutmut.sh`,
  `runtime_types.sh`, `complexity_baseline.json`), the benches/RL
  drivers (`contraction_*.py`, `train_*.py`, `calibrate_profile.py`,
  `executor_cost_probe.py`, `coordination_probe.py`,
  `law_wallclock.py`) and the artifacts (`evidence.db`,
  `intake_corpus.json`, `intake_tensors.pt`).
* `evidence.code_rev`'s `_CODE_PATHS` already covered `packages` +
  `tools`, so verdict cache keys still capture the engine's code —
  no schema or key change.

## 5. Gates after the move

`ruff check` / `ruff format --check` clean on `packages` + `tools`;
`ty check` 0 errors; `vulture` clean; `lint-imports` 4/4 contracts
kept; `bandit` 0 findings; `semgrep` (offline config) 0 findings;
`radon_ratchet` ok (2866 functions); the five focused law test files
pass (82 tests).  Full `pytest` and coverage runs were not part of
this change's verification (a background coverage run was in
progress; the suite exercises `catopt_discovery` transitively through
the focused tests).

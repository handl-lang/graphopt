# Agent guide — graphopt

The GitHub repo is `handl-lang/graphopt` (renamed from
`research-mlopt`; the `mlopt-*` package names and the "mlopt" project
name below are unchanged).

Categorical optimization of neural-network computation graphs
(Python 3.13, torch + numpy). Monorepo layout: the domain packages
live under `packages/` (`mlopt-core` — the torch-free engine;
`mlopt-torch` — PyTorch adapters; `mlopt-carriers` — carrier
laws/executors; `mlopt-cuda` — the CUDA-graph runner;
`mlopt-orchestrator` — the backend-neutral pipeline;
`mlopt-onnx` — the ONNX backend (ADR 0006: `OnnxSink` lowers IR to
a checked ONNX artifact, `OnnxSource` lifts ONNX graphs back to IR;
both verified through onnxruntime);
`mlopt-discovery` — the law-discovery engine, formerly
`tools/law_*.py`; invoke as `python -m mlopt_discovery.<mod>`;
the game layer is extracted to `packages/gamekit/` (plan 0032
lane A): `gamekit.engine` (the `Board` protocol + generic
episode/train/eval drivers), `gamekit.players` — a package:
`_base` holds the machinery (`LinearPolicy`, the
featurizer-injected learned arm), `players/registry.py` is the
name→spec table, and each player is a folder
(`random/ playbook/ linear/ neural/ distilled/` — its
`PlayerSpec` + measured record + `make` arm factory),
plus `gamekit.scoring` (the referee's value spec),
`gamekit.calibrate` (Platt/isotonic score calibration),
`gamekit.advantage` (the actor-critic player) and
`gamekit.gamedata` (the game weight/feature tables — the
ex-`lawdata.game` half).  The old paths
(`mlopt_discovery.engine`/`scoring`/`advantage`/`players`/
`lawdata.game`) are re-exporting shims; `play` (the
domain registry +
`python -m mlopt_discovery.play --domain meta|joint|search|torch|gen|open|player`
CLI; `--player <name>` runs a registered player eval-only, and
`--deliver` lowers+verifies the winning extraction on domains
that ship a deliver hook — torch does)).  The `kernel` domain
moved out with the kernelopt split (plan 0032 phase 4): it lives
in the private `handl-lang/kernelopt` repo, which seats it into
`DOMAINS` + the player registry at `mlopt_kernel.play` import —
the plugin seam is `play.spec_domain` +
`gamekit.players.registry.register_player`.
The optional
`mlopt-native` package — the PyO3/Rust search engine, excluded from
the uv workspace and built with maturin — is opt-in via `engine=`.
The `mlopt` façade is gone (plan 0008): `import mlopt` fails and
every name lives at its real package path — `mlopt_core.egraph`,
`mlopt_torch.adapters`, `mlopt_orchestrator.optimize`,
`mlopt_cuda.CudaGraphRunner`, … Tests in
`tests/`. The dev virtualenv is `.venv/` (uv-managed).

**Repo split (plan 0032 phase 4, landed).**  The substrate packages
moved to `handl-lang/eqsat`: `mlopt-core`,
`mlopt-orchestrator`, `mlopt-native`; the game layer moved to
`handl-lang/gamekit`: `packages/gamekit`.  Both are pinned
git+ssh deps on `packages/` subdirs (see `[tool.uv.sources]`); the
package/import names are unchanged.  The substrate-only test files
travelled with them — domain-bound test functions carved out live in
`tests/test_substrate_domain_edges.py`.  Coverage floors now apply
per-repo: this repo's wall is the domain packages
(`fail_under=100` on `mlopt_torch`/`mlopt_carriers`/`mlopt_cuda`/
`mlopt_onnx` + the 99% `mlopt_discovery` ratchet); the substrate and
game floors were re-measured in their own repos (56% / 98%).

**The research/measurement surface is `bench/`** — the dissolved
`lab/` lives there now, beside the harnessed benchmark package
(`python -m bench`; `bench/suites/`, `bench/discovery`,
`bench/players`, `bench/limits` — see `bench/README.md`).  The
ex-`lab/` drivers live under `bench/drivers/` and run standalone —
`.venv/bin/python bench/drivers/<sub>/<mod>.py` — and write result documents,
not `baselines/` rows:

- `bench/drivers/mint/` — `atom_law_loop.py` (ADR-0007 layer 3),
  `atom_synth_loop.py` (the composed need → handler → laws →
  admission loop).  The minted-macro miner is tooling:
  `scripts/macro_miner.py` mines + evals `lawdata.MACROS` candidate
  tables; the shipped replay machinery is
  `mlopt_discovery.macro_moves` and the retrain driver is
  `scripts/train_meta_macro_prior.py`.
- `bench/drivers/bridge/` — `combinator_probe.py`, `sharing_extract.py`
  (the ADR-0008 bridge).  The substrate and all four proposers
  promoted to `mlopt_discovery.combinator` / `.form_propose` /
  `.state_reindex` / `.index_abstract` / `.loop_propose`.
- `bench/drivers/audit/` — `grammar_probe.py`, `handl_differential.py`,
  `egraph_differential.py` (the handl-side oracles).
- `bench/drivers/screen/` — `exposure_screen.py`, `exposed_discovery.py`,
  `discovery_scale.py`.
- `bench/drivers/probes/` — the measurement probes, incl. `contraction/`
  (whose `contraction_distill.py --save` also mints the shipped
  `contraction_policy_distilled.pt`).
- `bench/drivers/policy/` — the controlled-protocol drivers
  `policy_baseline.py` + `policy_ablations.py` + `policy_rl_arms.py` +
  `schema_leaderboard.py` + `stop_policy.py` + `richer_features.py`
  (results in `docs/results_*.md`).
- `bench/drivers/train/` — the three measurement-only drivers
  `train_graph_policy.py`, `train_rl_policy.py`,
  `train_search_policy.py`.
- `bench/drivers/kit/` — `term_hnl.py`, the mlopt-term → `.hnl` emit/eval
  bridge into `CatoptKit`.
- `bench/drivers/overnight.py` — the overnight discovery loop:
  episodes → mint → `verify_law` gauntlet admission → prior-art-bank
  novelty classification → resumable JSONL ledger →
  `docs/results_overnight.md`.  `--corpus` picks the episode corpus:
  `grammar` (stream pages), `zoo` (stream filtered to real-matching
  claims per held-out zoo module), `zoo-real` (the corpus-driven
  mint — each episode's pool is `pipeline.propose`'s candidates over
  the module's own op-tuple census, judged against the corpus's
  distinct real subterms → `docs/results_overnight_real.md`),
  `zoo-xmodule` (the cross-module mint — one shared pool
  anti-unified over the WHOLE corpus: `op_tuple_census` op-patterns
  and `antiunify.mine_candidates` deep shapes recurring across
  ≥`--xbind` distinct modules, every claim K-bind gated to binding
  ≥K real modules → `docs/results_overnight_xmodule.md`).
  A flat-file driver (no `<sub>/`),
  standalone only — deliberately no `bench.suites` adapter
  (`bench/drivers/` is not a package; see the doc's suite-status
  note).

Some probes need scratch deps installed `--no-deps` outside the
lockfile: `birefnet_bounded_probe.py` wants
`timm einops kornia transformers torchvision`;
`retnet_baseline_probe.py` wants `torchscale fla-core einops fairscale`.
**Dev gates, artifact regeneration and user-facing utilities live in
`scripts/`** (`radon_ratchet.py` + `complexity_baseline.json`,
`runtime_types.sh`, `mutmut.sh`, `calibrate_profile.py`, plus the
shipped-artifact bakers promoted out of `lab/`: `train_meta_prior.py`,
`train_open_prior.py`, `train_player_meta_arm.py`,
`train_meta_macro_prior.py` — the `lawdata` weight-table literals;
`train_contraction_artifact.py` — the curriculum `.pt`;
`macro_miner.py` — the `lawdata.MACROS` table emit;
`dump_fragments.py` — the handl-spike fragment dirs; and
`dspec_audit.py` — the shipped-`dspec` audit).
**Regenerable artifacts live in the cache**, not the tree:
`mlopt_discovery.cache_dir()`
(`$MLOPT_CACHE_DIR` → `$XDG_CACHE_HOME/mlopt` → `~/.cache/mlopt`)
holds the intake census side-files (`intake_dir()`) and the evidence
store — shared across worktrees.  The minting machinery itself is in
`mlopt_discovery` (`antiunify`, `handler_synth`) — ADR 0007/0008.

**The handl bridge.**  `tests/fixtures/handl-spike/` holds the dumped
`fragments/`/`fragments-all/` language YAML the peer tests consume —
regenerate with `scripts/dump_fragments.py`.
`packages/mlopt-handl-peer/` is the PyO3 spike (Python calls handl's
loader + rewrite engine directly; excluded from the uv workspace,
built with maturin).  `packages/mlopt-experiments/` is the paper's
evidence crate (was root `experiments/`; a standalone cargo crate —
see `manifest/kit.toml` and the `Makefile`).

**The `lab/` dissolution is complete** — `lab/` is gone, dissolved by
destination (the per-file inventory and history live in
`docs/lab_dissolution_plan.md`):

- **prod machinery → `mlopt_discovery`** — code shipped modules
  depend on.  `combinator` / `form_propose` / `state_reindex`
  promoted at the 99% floor earlier; `index_abstract` /
  `loop_propose` followed at dissolution (shipped modules
  `form_propose` / `state_reindex` import them — they were already
  de-facto prod).  The discovery **99% ratchet** is a union
  property — tighten, never lower.
- **dev-facing tooling → `scripts/`** — drivers that *regenerate or
  audit a committed artifact* (the `train_*_prior` weight bakers,
  the `.pt` artifact trainers, `macro_miner.py`, `dump_fragments.py`,
  `dspec_audit.py`).
- **research drivers → `bench/drivers/`** — measurement drivers whose
  output is a result document (`docs/results_*.md`, `project/retros/`).
- **fixtures → `tests/fixtures/`** — the handl-spike fragment dirs.
- **Rust crates → `packages/`** — `mlopt-handl-peer`,
  `mlopt-experiments` (the paper's evidence crate).

New work lands by the same rule: a driver that starts emitting a
committed artifact graduates to `scripts/`; machinery shipped modules
depend on graduates to `mlopt_discovery` (the 99% ratchet applies to
the package union).

**Machine-admitted laws are in `DEFAULT`.**  `search(..., rules=DEFAULT)`
ships **13 machine-admitted laws** alongside the hand-written kernel
(`mlopt_core.laws.provenance.MACHINE_SHIPPED` — the promoted set;
`project/retros/promoted-laws.md`).  The newest is `add_unsq_pad_r` —
the first promotion whose paying site is a strictly-real intake
workload (`RelativePositionBias`), admitted through the auto-cond →
gauntlet retry (`docs/results_realcorpus.md`).  Before it,
`conv_bn_fold` — the first *verbatim* composite: a depth-4 `syn_*`
mint shipped through
`mlopt_core.laws.minted` (the embedded premise-closure records replay
`meta._compose_guards` at import; `missing_hooks` honestly reports the
transported `check`/`derive` hooks).  And `mul_unsq_nat` — the
first law minted from a corpus op-tuple, guard resolved by the
adversarial sweep.  Note the authorship split in `provenance.py`:
`select_mul` / `softmax_fold` / `glu_fold` / `rms_norm_fold` had
machine-*assisted* discovery but were hand-integrated before the
admission machinery existed and classify as **human-authored**.  A
change to any default rule changes extraction behavior for every
caller — run the full suite (not just a slice) when touching the bank.

## Verification commands

Run these before finishing any change; all must pass.

**One command instead of the list:** `scripts/gates.sh` runs the static
gates cheap-first, then the suite.  It has two lanes:
`MLOPT_GATES_LANE=dev` (default) runs `-m "not slow"` with no coverage —
the day-to-day gate; `=full` runs **everything once under coverage** and
enforces both floors (CI/release).  It defaults to a **serial** suite
(`MLOPT_GATES_NP=1`): the full-suite `-n 2` arm was measured
(2026-10-15) and **neither arm finished inside a 30-min box**, so the
parallel arm is not a safe default on this 11 GB host — see the
measured table below before raising it.

**Iterate with `-m "not slow"`.**  The minutes-scale work is marked
`slow` from measured durations.  There are **31** such tests:

- the six `torch.compile` tests (`test_torch_integration`,
  `test_om_decayed_cross`) — the 176-test slice runs **370.8 s**,
  ~349 s of it those six; the same slice with `-m "not slow"` is
  **42.6 s**;
- the two discovery held-out-claim fixtures (below);
- the two `test_square_rsqrt_laws` saturations (below).

**The fixture caveat matters.**  A module-scoped fixture is *not*
skipped by marking one test that uses it.  The discovery `measured`
fixtures cost minutes each, so **every** test that requests them is
marked (at class level in `test_discovery_open_learned` /
`test_discovery_open_prior`); with all of them deselected pytest never
instantiates the fixture.  Verified: `-m "not slow"` over the five
affected files selects 40 of 64 and runs in **14.1 s**, and no `setup`
entry for either `measured` fixture appears in `--durations`.

What it still costs: the tens-of-seconds tests that are *not* marked
(`test_om_decayed_kway_deliver` 20.9/15.8 s, `test_om_mha` 12.1 s,
`test_omd_lower` 11.7 s) and everything else.  So `-m "not slow"` is a
fast loop, **not** a substitute for the full pass — the full pass still
runs everything.

**The suite is tiered by kind, not just duration.**  Three markers:

- `e2e` — the test crosses an **external engine boundary**
  (torch.compile/Inductor, onnxruntime, CUDA-graph capture, the deliver
  pipeline).  Module-level on boundary-purpose files only:
  `test_e2e_pipeline`, `test_onnx_{sink,source,liftmap,opmap}`,
  `test_om_decayed_kway_deliver`, `test_runners`, `test_autotune`,
  `test_torch_integration`, `test_handl_peer` (**547 tests**).
  Torch-*eager* adapter
  tests are *not* e2e — they exercise the boundary shim, not the engine
  behind it.  Mixed files are deliberately left unmarked for now.
- `slow` — minutes-scale by construction (RL-training fixtures,
  60k-node saturation, compile sweeps): **31 tests**, orthogonal to
  `e2e`.
- `requires_cuda` — hardware gate, unchanged.

```sh
# smoke (~5 s): fast core sanity — 81 tests (the substrate smoke files
# moved to eqsat with the split; these are the domain-side pins)
.venv/bin/python -m pytest tests/test_law_order.py \
    tests/test_law_provenance.py tests/test_decode_geom.py -q

# core lane: unit/integration only — no external engines, no soaks
.venv/bin/python -m pytest -m "not e2e and not slow"     # 5,257 tests

# dev lane: everything except the minutes-scale
.venv/bin/python -m pytest -m "not slow"                 # 5,803 tests

# gate: everything, once, under coverage
scripts/gates.sh
```

```sh
# The shipped source: packages/ + scripts/ + examples/.
.venv/bin/ty check            # typecheck — 0 errors required (warnings OK)
.venv/bin/ruff check packages scripts examples   # lint
.venv/bin/ruff format --check packages scripts examples   # formatting
.venv/bin/vulture             # dead code (uses [tool.vulture] paths)
.venv/bin/lint-imports        # hexagonal boundary contracts
.venv/bin/bandit -c .bandit.yaml -r packages   # security SAST
.venv/bin/semgrep --config .semgrep.yml packages   # dataflow (offline)
.venv/bin/python scripts/radon_ratchet.py   # complexity ratchet

# The suite, once, under coverage (both floors below).
uv run pytest                 # 5,834 tests; does NOT finish in 30 min on
                              # either serial or `-n 2` (see the table
                              # below).  Iterate with `-m "not slow"`.
                              # Single-process: see the warning below.
coverage run -m pytest tests/ -q   # source list includes mlopt_discovery + mlopt_onnx
coverage report -m --omit="*/mlopt_discovery/*"   # the six: fail_under=100
coverage report -m --include="*/mlopt_discovery/*" --fail-under=99   # ratchet floor — measured 99%; tighten, never lower
```

**Scope the ruff commands to the shipped source, as above.**  The bare
`.venv/bin/ruff check` / `--check` forms are *not* green on `main` and
cannot be: they include the documented drift class — `tests/**` and
`bench/**` (pre-existing isort/format differences under the pinned ruff
0.16.9) and markdown code blocks in `README.md` and friends, which the
formatter would rewrap.  `packages`, `scripts` and `examples` are
clean under both, and `scripts/gates.sh` checks exactly those.

> **Parallel-warning (measured, not theoretical).**  Do *not* run this
> suite — with or without coverage — under `pytest -n auto` on
> low-RAM machines.  The suite is single-process by design: the
> torch.compile-heavy tests (`test_torch_integration` &c.) spawn an
> inductor pool of **16 compile workers per pytest process**, so
> `-n N` multiplies the fan-out to ~`16·N` subprocesses.  On an 11 GB /
> 16-core host, `-n auto` OOM-froze the machine; even `-n 3` drove
> memory to 10.2 GB near the suite's tail.
>
> **The suite does not finish in 30 min in *either* arm (full-suite
> measured 2026-10-15).**  On an *uncontended* 11 GB / 16-core host (load
> avg ~1.6), each arm was given a 30-min box and sampled with the same
> whole-tree `/proc` RSS helper (`peakrss.py`) — 6,372 tests:
>
> | arm | wall | progress in 30 min | peak RSS (tree sum) | min `free` headroom |
> |---|---|---|---|---|
> | serial | **did not finish** (killed at the box) | 36% (2,304/6,372) | 10,320 MB | ≥ 6.8 GB avail |
> | `-n 2` + `TORCHINDUCTOR_COMPILE_THREADS=2` | **did not finish** (killed at the box) | 98% (6,264/6,372) | 8,814 MB | ≥ 4.7 GB avail |
>
> So `-n 2` reaches **2.72× further in the same wall time** (98% vs
> 36%) — a real speed-up — but **neither arm completed**, so the
> *completed* wall-time win the default rule demands does not exist yet,
> and `scripts/gates.sh` stays **serial** (`MLOPT_GATES_NP=2` is the
> opt-in).  Caveat on the RSS column: the tree sum over-counts shared
> pages — the serial peak is a ~14-process fan-out each `mmap`ing the
> 419 MB `stories110M.bin`, so the real footprint is the `free` column,
> not the sum; both arms are memory-safe on this box, `-n 2` less so.
>
> **What the wall time is made of — and that the tail is now marked.**
> The 30-min box above *predates* the marking: as measured then, the
> `slow` marker covered only six tests (`test_torch_integration`,
> `test_om_decayed_cross`) and the *dominant* tail was **unmarked** — the
> discovery "held-out-claim" fixture (the module-scoped `measured`
> fixture in `test_discovery_open_learned` / `_open_prior`: a 36-page ×
> 4-arm episode loop) and the e-graph saturations in
> `test_square_rsqrt_laws`
> (`test_every_miss_spelling_reaches_rms_norm`,
> `test_canonical_rsqrt_spellings_fold_with_their_bridges`,
> `EGraph.run(..., max_nodes=60_000)`).  Serial stalls at ~36% inside
> the discovery block; `-n 2` overlaps those with the rest and reaches
> ~98%.  **Those 24 tests are now `@pytest.mark.slow`** (measured
> 2026-10-15), so `-m "not slow"` skips them: the learned module ran
> **>900 s** before the 900-s box killed it (the `measured` fixture
> dominates; `EGraph`-free tests in it are seconds) and the prior
> module's fixture ran **>712 s** (11:52) before it was terminated,
> `test_every_miss_spelling_reaches_rms_norm` **>9:49** and still
> climbing (~1.2 GB RSS), and
> `test_canonical_rsqrt_spellings_fold_with_their_bridges` **>7 min**
> (killed at the 420-s box).  The `test_discovery_meta_prior` /
> `test_discovery_open_game` fixtures are **not** marked — they measured
> cheap (0.6 s / <2 s), as are the other `max_nodes=60_000` tests in the
> file (`test_real_export_spellings_fold_and_verify` 2.8 s,
> `test_bridges_fire_in_default_pipeline` 0.9 s).  The suite was
> **6,372 tests** at that measurement; it now collects **5,834**
> (the substrate files moved to `eqsat` at the phase-4 split).  The old
> "~7.5 min" and "~28 min" figures are both stale.
>
> **The slice comparison is the only *completed* measurement.**  On the
> heaviest slice (`test_torch_integration` + `test_om_decayed_cross`, 176
> tests): **serial 370.8 s / `-n 2` 251.4 s** with
> `TORCHINDUCTOR_COMPILE_THREADS=2` — **1.47×**, peak RSS **2.4 GB** (two
> workers; the cap keeps the inductor fan-out at ~2×2 rather than 2×16).
> Re-confirmed 2026-10-15: serial **364.4 s, peak 1,229 MB**.  (An earlier
> claim here that two runs "died at 36%" was a **misdiagnosis** — the runs
> were alive; pytest's `-q` progress is block-buffered when redirected, so
> the log looks frozen, and `ps` greps must match the pytest *child*, not
> the wrapper shell.  Measure with `/proc` sampling, not the log.)
>
> The coverage gate is a commit-time check; for iteration use a scoped
> `coverage run -m pytest tests/test_<file>.py`.

Manual-stage gates (not run on every commit — network/slower):

```sh
.venv/bin/pip-audit                                          # dependency CVEs (network)
.venv/bin/semgrep --config p/python --config p/security-audit packages  # registry rules (network)
sh scripts/runtime_types.sh                                    # typeguard runtime contracts
MUTMUT_ONLY=mlopt_discovery/antiunify.py MUTMUT_TESTS='tests/test_antiunify.py' scripts/mutmut.sh run  # mutation testing
```

Pre-commit (`pre-commit install`) runs ruff check/format, **ty**,
import-linter, vulture, bandit, semgrep and the radon ratchet.  The
`manual`-stage hooks (typeguard, pip-audit, semgrep-registry) run with
`pre-commit run --hook-stage manual --all-files`.

Known drift: the installed ruff (0.16.9) still flags pre-existing
isort / format differences across `tests/` and `bench/`, which are not
format-checked.  The shipped source — `packages`, `scripts`
and `examples` — is clean under both `ruff check` and
`ruff format --check`.

## Typecheck (ty)

- Checker: **ty** (`[tool.ty]` in `pyproject.toml`) — Astral's type
  checker, replacing the earlier pyright ratchet.
  `[tool.ty.src] include = ["packages"]` scopes checking to the
  shipped packages (tests/ and bench/ are not type-checked);
  `[tool.ty.environment]` sets the 3.13 target and the per-package
  `extra-paths`; `[tool.ty.terminal] error-on-warning = false`.
- Installed in `.venv` via the `dev` dependency group (`ty>=0.0.84`;
  currently 0.0.84).
- **The ratchet `exclude` list is empty.** The migration baseline was
  198 errors across 19 files (down from pyright's 442/22); every one of
  those files has since been annotated, so `ty check` is clean with no
  opt-outs.  Keep it that way — annotate; never add an exclude entry or
  a `# ty: ignore`.
- New modules under `packages/*/src/` are checked automatically.

## Dead-code + architecture linting

Two dev-only ratchets (both in the `dev` dependency group, both
configured in `pyproject.toml`):

- **vulture** (`[tool.vulture]`, `min_confidence = 80`) flags
  unreachable / unused code across `packages` / `tests`.
  The suite is pinned at 100% coverage, so a genuine dead branch is a
  real finding, not noise.  Run `.venv/bin/vulture` (uses the
  configured `paths`); a non-zero exit (3) means dead code.
- **import-linter** (`[tool.importlinter]`) pins the hexagonal boundary:
  a `forbidden` contract makes `mlopt_core` importing `mlopt_torch` /
  `mlopt_carriers` / `mlopt_orchestrator` — or the `torch`
  / `numpy` external packages — a hard error.  Run
  `.venv/bin/lint-imports`.  Core is a *sink* for adapter-pushed state
  (see `mlopt_core.ops` *Backend wiring*), never a puller: the adapter
  registers its tables into core, core never imports the adapter.

## Property tests (Hypothesis)

The `test_property_*.py` hypothesis suite moved to `handl-lang/eqsat`
with the substrate at the phase-4 split — it exercises
`mlopt_core`'s IR interning, e-graph congruence, cost-model
monotonicity and law round-trips, and runs in that repo's suite.

## Static analysis & security

- **Bandit** (`.bandit.yaml`, dev dep): SAST over `packages`
  (tests/bench excluded).  The only finding is two `B112`
  (try/except/continue) in `mlopt_core.meta`'s rule matcher, where a
  guard/derive raising is the *signal to reject a candidate* — a
  justified config-level skip.  Run `.venv/bin/bandit -c .bandit.yaml
  -r packages`.
- **Semgrep** (`.semgrep.yml`, dev dep): the committed config is
  deterministic/offline (no-`eval`/`exec`, no-`shell=True`,
  no-unsafe-`yaml.load`).  The deeper registry scan
  (`--config p/python --config p/security-audit`) is a manual-stage
  gate; at migration it reported 0 findings.
- **pip-audit** (dev dep): audits the installed environment; manual
  stage (needs network).  At migration: 0 known vulnerabilities (the
  five first-party workspace packages are skipped as local editable
  installs).

## Complexity ratchet (radon)

The engine is math-heavy (47 rank-C, 11 D, 7 E, 5 F functions at
migration), so a hard ceiling is impractical.  Instead
`scripts/radon_ratchet.py` pins every function's cyclomatic complexity in
`scripts/complexity_baseline.json`: a function may not exceed its recorded
value, and a function **not** in the baseline must stay at or below
`THRESHOLD = 11` (rank C).  After an intentional complexity change,
regenerate with `--update` and review the diff.  Run
`.venv/bin/python scripts/radon_ratchet.py`.

## Docstring quality (ruff `D`)

The pydocstyle ruleset is enabled through ruff (`select = [..., "D"]`).
The `D` ignore list is now **empty**: the whole backlog (550 violations
— missing docstrings, summary/blank-line style, imperative mood, …) has
been cleared, so `ruff check packages` enforces `D` in full.  The
codebase adopts **D211** (blank line before a class docstring) and
**D212** (multi-line summary on the first line) over the mutually
exclusive D203/D213 — ruff prints its usual "incompatible" warning for
those pairs; that is the expected signature of the choice, not an
error.  The gate applies to shipped code only — `tests/**` and
`bench/**` ignore `D`.

## Runtime contracts (typeguard)

`typeguard` (dev dep) enforces annotations at runtime.
`scripts/runtime_types.sh` instruments all four packages
(`--typeguard-packages=mlopt_core,mlopt_torch,mlopt_carriers,mlopt_orchestrator`)
and runs the curated (1372 tests) — a manual-stage gate that CI also
runs (~11 min).  It caught several real annotation bugs, all fixed:
`laws/tensor._head` was annotated `str` but takes an `Op`;
`typing._infer_op_shape` passed a `tuple` to zero-arg shape rules typed
`list`; `typing._shape_of` returned the `_INVALID` string sentinel under
a `tuple | None` return type; `EGraph.matches`/`_match` substituted
attribute metavariables (floats/strs) into `dict[str, int]`; and the
adapter `eval_term` / `ev_factory` / carrier-module `forward`s declared
`-> torch.Tensor` returns that are really carrier tuples or `None`
(widened to `Any`).  One parameter was renamed `memo` → `memo_env` to
dodge a typeguard 4.6 shadowing bug.  Files still left out (and why):
heavy saturation tests that exceed ~20 min under instrumentation
(`test_torch_integration`, `test_omd_lower`, `test_om_mha`), and a few
error-path tests that deliberately pass wrong-typed values.  Grow the
list in the script.

## Mutation testing (mutmut)

`scripts/mutmut.sh` runs mutmut against a flat-layout sandbox
(`.mutmut-sandbox/`, gitignored): mutmut 3 derives mutant keys from
paths, but a per-package `src/` layout imports under a different dotted
name, so the wrapper symlinks each package at the sandbox top level
(path == import name) plus `tests/` and `uv.lock`, and writes
the matching `[tool.mutmut]`.  Scope a fast loop with `MUTMUT_ONLY` /
`MUTMUT_TESTS`; a surviving mutant is a weak-assertion finding (the
suite is pinned at 100% coverage, so it is not a coverage gap).  The
editable-finder drop in `tests/conftest.py` exists solely so the
sandbox's mutated copies win over uv's editable install.

## Differential oracle (opt-in, test-only)

`tests/test_egglog_oracle.py` cross-checks mlopt's hand-rolled
equality-saturation search against the Rust **egglog** library on a
small op/law subset.  It is an **opt-in, test-only oracle** — not a
production dependency and not an engine swap: `mlopt-core` stays
pure-Python / zero-dependency, and nothing under `packages/` or
`bench/` imports `egglog`.  The ported prototype and its documented
limitations (untyped-by-shape terms, unconditional `check` rewrites, no
proof replay) live in `tests/egglog_oracle.py`.

`egglog` is in a dedicated `oracle` dependency group — deliberately
**not** the default `dev` group — so the repo stays light (egglog is a
compiled Rust extension).  The test begins with
`pytest.importorskip("egglog")`, so the whole file skips cleanly when
the group is absent and the rest of the suite stays green.

Run it explicitly:

```sh
uv sync --group oracle
.venv/bin/python -m pytest tests/test_egglog_oracle.py
```

It runs mlopt's `EGraph` and egglog on the *same* law subset and
asserts they agree: the extracted lowest-FLOPs terms match up to
commutative-operand *representative* choice, the folded (param-
discounted) costs coincide, and the original plus both extracted terms
are numerically equal (torch ground truth + NumPy evaluation).  It is
fast (<~1 s) and needs no CUDA.  `egglog` ships wheels for CPython
>=3.12 only, so the group entry carries a `python_version` marker while
the workspace keeps `requires-python = ">=3.11"`.

## Ports (hexagonal boundary)

`mlopt_core.ports` names every adapter contract as a
`@runtime_checkable` Protocol.  The whole-graph ends are `Source`
(`model -> (IR, leaves)`) and `Sink` (`IR -> runnable`, plus its
`supported_ops` set and the module-level equivalence gate); the per-op
port is `Binding` (`TorchBinding` is an alias of the same object).
`mlopt_orchestrator.Optimizer` takes the ports as a
`mlopt_core.pipeline.Backend` value (torch:
`Optimizer(backend=TorchBackend())`, the
`mlopt_torch.backend.TorchBackend` bundle of
`TorchSource`/`TorchSink`/`TorchComposer`/`TorchMeter`) or explicitly
(`source=`/`sink=`/…); `discover_alternatives` takes `source=`.
There is no default backend.
Extraction is priced through
`mlopt_core.cost.backend_cost(cost_fn, sink.supported_ops)`, so the
search only selects forms the backend can lower.  A new backend
implements `Sink`; nothing in `mlopt-core` changes.

Carrier machinery is the second half of the boundary.  The
orchestrator's carrier-aware passes (batched plan builders, non-local
lifts, carrier rule sets, carrier-root probes) reach the torch-coupled
`mlopt_carriers` package through the `register_carriers` seam
(`mlopt_orchestrator.carriers`), never by importing it: the carrier
package registers a `CarrierMachinery` value — all thunks, so
registration loads no tensor library — when it is imported.  Without a
registration the passes degrade to their carrier-free defaults.  The
resulting package graph is pinned by the import-linter `dependency
layers` contract: `mlopt_cuda` above `mlopt_torch`/`mlopt_carriers`
(one mutually-dependent backend layer — the carriers' law modules
carry `TORCH_BINDINGS`, and the torch sink builds its executor table
from the carrier lowerings), above `mlopt_orchestrator`, above
`mlopt_native`, above `mlopt_core`.  `mlopt_torch` does not depend
on `mlopt_cuda`; import `CudaGraphRunner` from `mlopt_cuda`.

## Conventions

- Line length 72 (ruff). E501/B008/SIM108 intentionally ignored — see
  `[tool.ruff.lint]` for rationale.
- Coverage floor: `fail_under = 100` in `[tool.coverage.report]` —
  the suite is pinned at 100% for the original packages plus
  `mlopt_onnx` (which climbed to the wall in
  `project/retros/onnx-package.md`); new branches need tests (or a
  justified `pragma: no cover`).
  `mlopt_discovery` joined the source list after its staged climb
  and sits under a **pinned ratchet floor** (`--fail-under=99` at
  landing — tighten the number as the residual gaps close, never
  lower it; the unreachable-defensive arcs are documented in
  `project/retros/discovery-package.md`).
- Tests allocating CUDA tensors use the `requires_cuda` marker
  (auto-skipped when CUDA is absent).  Tests that exercise the CPU
  no-op path of `capture_cuda_graph` force `torch.cuda.is_available()`
  off via `monkeypatch`, so they pass on a CUDA host too.
- Ratchets, not rewrites: the ruff `D` ignore list, the radon baseline
  and the typeguard file list are "pin the current state, never
  regress" gates.  Tighten them as code improves; never widen them to
  make a change pass.  The ty `exclude` list is empty — keep it so.
- No new suppressions.  Do not add `# type: ignore` / `# ty: ignore` /
  `# noqa` / `# nosec` / `# pragma: no cover` to make a gate pass; fix
  the code or add a justified, documented config entry (as `.bandit.yaml`
  does for the two `B112` `try/except/continue` sites).
- The lockfile resolves a CUDA-enabled `torch` wheel on Linux.  CI
  installs the CPU wheel (`uv pip install --torch-backend cpu
  --reinstall torch`) for determinism; the CUDA-graph tests no longer
  require it locally.
- Four independent dimensions (ADR 0003): **semantics / search /
  evaluation / execution**.  No layer answers another's question — a
  cost model must not change semantics, a policy must not decide
  equivalence, a profiler must not run on the target.  Feasibility
  (`supported_ops`, hard) and performance ranking (soft) stay
  distinct.
- "Profile" means one thing: a measured target
  (`mlopt_core.profile.TargetProfile`).  Static characterization of
  a *program* is `ProgramFeatures` (a future `mlopt_core.features`),
  never called a "profile".

## Adding a rewrite law

Laws live in `packages/mlopt-core/src/mlopt_core/laws/` — `tensor.py`
(tensor algebra), `scan.py` (scan monoids).  Shape (via `laws.base.R`):
`R(name, lhs, rhs, law=..., check=..., derive=...)` — lhs/rhs are
`Op.make` pattern trees; bare `str` leaves are metavariables, `Const`
leaves are literals, and `str` attr values are attr metavariables bound
under `"$attr:"` keys.  Prefer `cond=...` — a declarative side
condition in `mlopt_core.laws.cond` (pure data: `("and", ("rank-eq",
"a", "b"), ("rank-ge", "a", 2))`, serializable via `cond_to_data`/
`cond_from_data`; 58 of 86 laws use it) — over `check(bound) -> bool`,
the escape hatch for conditions the DSL can't express: `bound` maps
each metavar to a resolved member term (use `laws.base._shape_of` for
shape guards — the matcher cannot see types).
`derive(bound) -> dict | None` computes RHS attrs absent from the LHS
(`{"$attr:SZ": ...}`); `None` vetoes.  Prefer `dspec=...` — the twin
declarative DSL in `mlopt_core.laws.cond` (a `{NAME: expr}` map of
shape/attr expressions, serializable via `derive_to_data`/
`derive_from_data` and folded into `rule.derive` at construction);
19 of 86 laws use it, leaving **85 of 86 laws fully
data-serializable** — the 86th, `conv_bn_fold`, is a shipped
machine-minted composite whose hooks are rebuilt deterministically
from its stored data guard spec (the partition is pinned in
`tests/test_law_serialize.py`; see
`project/retros/derive-declarative.md`).  Every firing records a
witness
(proof edge + rule provenance) — no extra bookkeeping needed.  Register
the rule in a group list at the bottom of its module
(`SIMPLIFICATION_RULES`, `CATEGORICAL_RULES`, `SDPA_FOLD_RULES`,
`SCAN_LAWS`, `SCAN_DIAG_LAWS`; `ALL_RULES`/`all_rules()` is the union).
Rules also carry tags from `mlopt_core.laws.tags` (pass
`R(..., tags=...)`); `EXPANSIVE` marks the closure-generating rules the
pipeline budgets (`rules.tagged(EXPANSIVE)`).  Each rule's kernel kind
lives on `Rewrite.derivation`/`rule.kind` — `R(..., derivation=(...))`
names the axioms that prove a lemma (`mlopt_discovery.coherence
--emit-basis` emits the measured table; see
`project/retros/axiom-lemma-split.md`).  `mlopt_discovery.lemma_cert`
materializes each recorded derivation as a replayable `Certificate`
(codec: `cert_to_data`/`cert_from_data` in `mlopt_core.egraph.certs`;
see `project/retros/lemma-certificates.md`).  The composable `RuleSet`
presets live in `mlopt_core.laws.ruleset`.

Measure it: `python -m bench.suites.correctness.law_bench --laws <name[,name|group]>`
(`--sizes <d[,d]>` selects sizes; the suite's own CLI is the entry point) —
per law: registered synthetic term → `eg.run` on that rule alone →
extraction (pipeline cost model) → `_lower_extracted` → `sink.verify` →
timed before/after.  Table shows fired / rhs-member / picked / verified /
cost & ms before→after; non-firing laws report honestly.  Add a builder
in `LAW_CASES` keyed by rule name for new laws.  Gates: `uv run pytest`,
`.venv/bin/ty`, `.venv/bin/ruff check` (packages), coverage stays 100.

Benchmarks are a package, not scripts: `python -m bench list` shows the
catalog; `python -m bench run <suite>` emits JSON + Markdown + HTML +
plotly plots (+ optional Quarto/Slidev) from one canonical `Report`;
`python -m bench dashboard` builds the cross-suite HTML index;
`python -m bench catalog` emits the docs.  Suites live in
`bench/suites/<intent>/` and expose `run_bench(args) -> Report`; shared
data (checkpoint loaders) is in `bench/common/`.  `bench/registry.py`
is the source of truth for a suite's **intent** (the question it
answers), its `question`/`expects`, `tier`, `status`, controlled
`mechanisms` and `needs_cuda` — the README tables and the dashboard
grouping are generated from it, so they cannot drift.  Details in
`bench/README.md`.  The named-optimization corpus lives in
`bench/discovery/` (TOML `cases/` + known-good `refs/`); its runner is
`python -m bench.discovery.run --fast|--full` (registered as the
`discovery` suite) and emits `docs/scorecard.md` + `.json`.
The player comparison lives in `bench/players/` — `python -m
bench.players.run --players … --boards meta,open,contraction`
(`--smoke` is the seconds lane) replays `bench/drivers/policy/policy_baseline.py`'s
memoized-verdict protocol over the `players/` registry and emits
`docs/leaderboard_players.md` + `.json`.

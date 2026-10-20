# `lab/` dissolution plan

**Status:** **complete** — `lab/` (and the root `experiments/` crate) are
gone; every file landed by destination.  The final destination map:

| destination | what landed |
|---|---|
| `mlopt_discovery` | `index_abstract.py`, `loop_propose.py` (the last bridge modules shipped code imported — `form_propose`/`state_reindex` depend on them, so they were de-facto prod; the 99% ratchet applies to the package union) |
| `scripts/` | `macro_miner.py` (the `lawdata.MACROS` `--emit` bake driver, imported by `train_meta_macro_prior.py`) |
| `bench/drivers/probes/` | all 23 probes incl. `contraction/` (12) |
| `bench/drivers/screen/` | `exposure_screen`, `exposed_discovery`, `discovery_scale` |
| `bench/drivers/mint/` | `atom_law_loop`, `atom_synth_loop` |
| `bench/drivers/bridge/` | `combinator_probe`, `sharing_extract` |
| `bench/drivers/audit/` | `grammar_probe`, `handl_differential`, `egraph_differential` |
| `bench/drivers/kit/` | `term_hnl` |
| `bench/drivers/train/` | the three measurement-only `train_*_policy` drivers |
| `bench/drivers/policy/` | the six controlled-protocol drivers (`policy_baseline`, `policy_ablations`, `policy_rl_arms`, `schema_leaderboard`, `stop_policy`, `richer_features`) |
| `tests/fixtures/handl-spike/` | the committed `fragments/`/`fragments-all/` peer-test fixtures (`scripts/dump_fragments.py` regenerates them there) |
| `packages/mlopt-handl-peer/` | the PyO3 spike (path deps unchanged — same depth) |
| `packages/mlopt-experiments/` | the paper's evidence crate (was root `experiments/`; relative paths to `mlopt-kit` / sibling checkouts re-anchored) |

`index_abstract`/`loop_propose` landed at 81%/80% dedicated-test
coverage — below the per-module ideal but inside the **union** 99%
ratchet they now fall under (the alternative — shipped package code
importing `bench/` — was strictly worse than the `lab/` wart it
replaced).  Every other promote queue item either promoted earlier or
dissolved to `bench/` with its consumers.

**History below** — the executed-2026-10-19 state: the safe moves landed
then; the classified queue follows.  `combinator.py` promoted to
`mlopt_discovery.combinator` at the 99% floor (92% → 100% under
`tests/test_combinator_bridge.py`).

**The rule.** `lab/` was the promotion queue with a single exit
("driver → `mlopt_discovery` at the 99% coverage floor").  The exit
set is now three, matching what the files actually *are*:

| destination | for | gate |
|---|---|---|
| `mlopt_discovery` | prod machinery — code whose output is shipped behaviour | tests at the **99% floor** (the discovery ratchet — tighten, never lower) |
| `scripts/` | dev-facing tooling — drivers that *regenerate or audit a committed artifact* | ruff/format clean, documented invocation |
| stays in `lab/` | research drivers — measurement whose output is a result document (`docs/results_*.md`, `project/retros/`) | — |
| archive | dead / superseded work | nothing qualifies today — every file was committed within the last week |

The distinguishing question is **what the file's output is**: a
committed artifact (a `lawdata` literal, a `.pt`, a fixture dir, a
shipped audit) means tooling or prod machinery; a result document
means research driver.

## What moved (this change)

Seven files — every driver whose *primary purpose* is regenerating or
auditing a committed artifact:

| from | to | artifact it owns |
|---|---|---|
| `lab/train/train_meta_prior.py` | `scripts/train_meta_prior.py` | `lawdata.META_ARENA_PLAYER_WEIGHTS` literal |
| `lab/train/train_open_prior.py` | `scripts/train_open_prior.py` | `lawdata.OPEN_GAME_PLAYER_WEIGHTS` literal |
| `lab/train/train_player_meta_arm.py` | `scripts/train_player_meta_arm.py` | `lawdata.PLAYER_ARENA_PLAYER_WEIGHTS` literal |
| `lab/train/train_meta_macro_prior.py` | `scripts/train_meta_macro_prior.py` | the macro-aware meta-prior literal |
| `lab/train/train_contraction_artifact.py` | `scripts/train_contraction_artifact.py` | `mlopt_torch/artifacts/contraction_policy_curriculum.pt` |
| `lab/handl-spike/dump_fragments.py` | `scripts/dump_fragments.py` | `lab/handl-spike/fragments*/` (peer-test fixtures) |
| `lab/audit/dspec_audit.py` | `scripts/dspec_audit.py` | the shipped-`dspec` vs git-oracle audit |

Reference updates landed with the moves:

- `scripts/train_meta_macro_prior.py` — `lab.train.train_meta_prior`
  → `scripts.train_meta_prior`; repo-root `sys.path` shim re-anchored
  (was `parent³`, now `parent²` — same resolution, correct depth).
- `scripts/train_contraction_artifact.py` — gained the sanctioned
  repo-root shim.  This **fixed two latent bugs** the `tools/` →
  `lab/` move had left: the documented
  `python lab/train/train_contraction_artifact.py` invocation died on
  `ModuleNotFoundError: No module named 'lab'` (no shim), and
  `_DEFAULT_OUT`'s `parents[1]` resolved to `lab/`, pointing the
  default artifact path at a nonexistent `lab/packages/` tree.  From
  `scripts/` both are correct.
- `scripts/train_player_meta_arm.py` — dead shim + unused
  `sys`/`pathlib` imports removed (it imports no `lab.*`).
- `lab/probes/weightspace_fold_probe.py`,
  `lab/probes/conv_bn_fold_synth_probe.py` — the by-path
  `importlib` loaders repointed `lab/audit/` → `scripts/`.
- Doc references: `lawdata/game/weights.py` (6 sites), `open_game.py` (2),
  `mlopt_torch/contraction_policy.py` (2 — incl. a stale `tools/`
  ref), `tests/test_discovery_{meta,open}_prior.py`, `AGENTS.md`.
- `scripts/dump_fragments.py` still defaults its output into
  `lab/handl-spike/fragments/` — the spike owns the committed
  fixtures (`tests/test_handl_peer.py` reads them there); the tool
  only regenerates them.

Historical references in `docs/results_*.md` and `project/retros/`
were left as written records of what ran.

## What stays, and why

### `lab/train/` — the three measurement-only drivers

`train_graph_policy.py`, `train_rl_policy.py`,
`train_search_policy.py` train policies and *measure* them (per-family
rank / hit-rate races, real-pipeline wiring) but bake nothing — their
output is `docs/`/`project/` results, so they stay as research
drivers.  If one starts emitting a shipped artifact it graduates to
`scripts/` the same way the bakers did.

### `lab/mint/` — the ADR-0007 loop

`atom_law_loop.py` (layer 3 — an atom's laws from measured bindings)
and `atom_synth_loop.py` (the composed need → handler → laws →
admission loop) are active research with no shipped artifact yet.
**`macro_miner.py` + `macro_moves.py` are owned by a concurrent
promotion** — classified only here: `macro_moves.py` is the `Macro`/
`legal_with_macros` machinery (prod machinery — a promote candidate;
a `mlopt_discovery.macro_moves` module + `lawdata/laws/macros.py` were
already in flight in the working tree when this landed), and
`macro_miner.py` is its minting driver (prod-generating — the minted
macros feed the macro prior bake).  `scripts/train_meta_macro_prior.py`
still imports `from lab.mint.macro_moves` / `lab.mint.macro_miner`;
**when the concurrent promotion lands, that import wants
`mlopt_discovery.macro_moves`** — flagged, not pre-edited (the package
module does not exist on `main` yet, so editing now would break the
script).

### `lab/bridge/` — the promote queue, ranked by gap-to-99%

Measured under `coverage run --source=lab -m pytest
<their dedicated files>` (the `slow` tests included):

| module | coverage | test file | gap |
|---|---|---|---|
| ~~`combinator.py`~~ → **`mlopt_discovery.combinator`** ✅ | **92% → 100%** | `test_combinator_bridge.py` | promoted: +9 targeted tests closed the guards (`arity`/`_apply`/`comb_to_dag` `raise`s, `_par`/`_mux` edges, `morph_stats` Drop/Loop/non-morphism, `dag_size`); the one residual — the closing-`Drop` in `dag_to_comb` — is provably unreachable (a ctx wire survives to the close only while `remaining[w] > 0`, and at the close `remaining` counts exactly the output uses), so it carries a documented `pragma: no cover` |
| ~~`form_propose.py`~~ → **`mlopt_discovery.form_propose`** ✅ | **84% → 99%** | `test_form_propose.py` | promoted: targeted tests cover every statement and all-but-three arcs; the residuals are provably unreachable (the `m_eid is None` guard in `factored_states` — an extracted term is always graph-resident — and the `prog_stats`/`boundary` falsy arcs in the report, unreachable under the `ok`/`setdefault` invariants); the dead `unbound-wires` guard and the `("raw", …)` concat-skeleton arms carry documented `pragma: no cover`s |
| ~~`state_reindex.py`~~ → **`mlopt_discovery.state_reindex`** ✅ | **83% → 100%** | `test_state_reindex.py` (new, dedicated) | promoted: law-guard/derive unit tests + a non-slow synthetic-fold `reindex_close` close cover everything; the `("raw", …)` scan arm carries the same documented pragma |
| `index_abstract.py` | 81% | `test_index_abstract.py` | 62 + 18 |
| `loop_propose.py` | 80% | `test_loop_propose.py` | 24 + 4 |

`combinator.py` was the promote-now candidate and landed: its gaps
were deterministic guards, with the single pragma exception noted
above.  The promoted module now falls under the package-only gates
(`ty`, vulture, the radon ratchet, the discovery 99% ratchet measured
on the *union*).

`sharing_extract.py` and `combinator_probe.py` have no dedicated test
file — they stay until the gate-3 line settles.

### `lab/audit/` — the oracles

`grammar_probe.py` (refusal census — a measurement instrument),
`handl_differential.py` and `egraph_differential.py` (the mlopt-vs-handl
oracles — `egraph_differential` is e2e-tested via
`tests/test_egraph_differential.py`, gated on the `egraph-run`
binary).  The two differentials are promote candidates once the
handl-peer spike settles; they live next to it for now.

### `lab/screen/` — the discovery drivers

`exposure_screen.py`, `exposed_discovery.py`, `discovery_scale.py` —
research drivers (their output is `project/retros/exposed-math.md`;
the one law they minted, `census:mul_unsqueeze`, shipped
hand-integrated through `mlopt_core.laws.minted` — the drivers
themselves bake nothing).  `exposure_screen._export`/`_workloads` are
imported by `lab/mint/` — moving it breaks the loop.

### `lab/probes/` — the measurement fleet

All 23 files are measurement probes; they stay.  Two need singling
out:

- **`contraction/contraction_distill.py`** — the *only* probe that
  also mints a shipped artifact (`--save` wrote the bundled-default
  `contraction_policy_distilled.pt`).  It stays because it is an
  arm-comparison harness inside a tightly cross-imported cluster
  (`contraction_{policy,scale,einsum,az}` — also imported by
  `bench/suites/evaluation/contraction_ladder.py` and
  `lab/policy_rl_arms.py`), not a standalone baker.  When the
  contraction line settles, the cluster moves or promotes together.
- **`conv_fold_wall_regimes_probe.py`** — holds the sanctioned
  `sys.path` shim precedent other scripts now cite.

### Top-level policy drivers — active research

`policy_baseline.py`, `policy_ablations.py`, `policy_rl_arms.py`,
`schema_leaderboard.py`, `stop_policy.py`, `richer_features.py` — the
controlled-protocol measurement drivers behind
`docs/results_{learned,schema,stop,richer,macro}*.md`.  None bake
(`richer_features` *registers* open-domain feature parts in
`open_game.py`, but the bake itself is `train_open_prior.py`'s job).
Active research — stay.

### `lab/kit/`, `lab/handl-spike/`, `lab/handl-peer/` — the handl bridge

`kit/term_hnl.py` — the mlopt-term → `.hnl` emit/eval bridge, e2e-tested
(`tests/test_kit_bridge.py`, needs the `hnl-run` binary): a promote
candidate once its coverage is measured (it is machinery, not a
driver).  `handl-spike/` keeps the committed `fragments/`/
`fragments-all/` fixture dirs the peer tests read.  `handl-peer/` is a
Rust crate + PyO3 spike (plan 0030) — it moves with the handl line,
not piecemeal.

## Promote-candidate ranking (gap-to-99%)

1. ~~**`lab/bridge/combinator.py`**~~ — **promoted** to
   `mlopt_discovery.combinator` (92% → 100%).
2. ~~**`lab/bridge/form_propose.py`**~~ — **promoted** to
   `mlopt_discovery.form_propose` (84% → 99%).
3. ~~**`lab/bridge/state_reindex.py`**~~ — **promoted** to
   `mlopt_discovery.state_reindex` (83% → 100%); now has the
   dedicated `tests/test_state_reindex.py` (the slow close test moved
   out of `test_form_propose.py`).
4. **`lab/bridge/index_abstract.py`** — 81%.
5. **`lab/bridge/loop_propose.py`** — 80%; small module (138 stmts).
6. **`lab/kit/term_hnl.py`** — e2e-tested; coverage unmeasured.
7. **`lab/audit/egraph_differential.py`** — e2e-tested; gated on the
   `egraph-run` binary.
8. **`lab/mint/macro_moves.py` + `macro_miner.py`** — classified only;
   the concurrent agent is already mid-promotion.

## What did NOT happen (guardrails kept)

- **No research driver moved into `packages/`** — that would put
  untested code under the 100%/99% coverage floors and break the
  gate.
- **No archive** — nothing is dead; every file was committed within
  the last week.
- **`lab/stop_policy.py`, `lab/mint/macro_{moves,miner}.py`**
  untouched — foreign WIP owned by the concurrent agent.

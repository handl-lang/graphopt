# `lab/` dissolution plan

**Status:** executed 2026-10-19 — the safe moves are landed; the rest is
the classified queue below.

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
- Doc references: `lawdata/weights.py` (6 sites), `open_game.py` (2),
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
a `mlopt_discovery.macro_moves` module + `lawdata/macros.py` were
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
| `combinator.py` | **92%** | `test_combinator_bridge.py` | 23 stmts + 12 partial branches — all deterministic guards and edge cases (`raise`s on bad morphisms, `_par`/`_mux` edges, `morph_stats` Drop/Loop, the distinct-leaf helper): plausibly closes to 99% with ~15 targeted tests |
| `form_propose.py` | 84% | `test_form_propose.py` | 71 stmts + 48 br — needs real test growth |
| `state_reindex.py` | 83% | via `test_form_propose.py` | 25 + 22 |
| `index_abstract.py` | 81% | `test_index_abstract.py` | 62 + 18 |
| `loop_propose.py` | 80% | `test_loop_propose.py` | 24 + 4 |

`combinator.py` is the promote-now candidate: its gaps are
deterministic, no unreachable-defensive arcs that would need pragmas.
Promotion cost is more than tests, though — the module then falls
under the package-only gates (`ty`, vulture, the radon ratchet, the
discovery 99% ratchet measured on the *union*), so it lands as its own
change, not bundled with tooling moves.

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

1. **`lab/bridge/combinator.py`** — 92% → 99% needs ~35 arcs of
   deterministic-guard tests; then ty/vulture/radon + the discovery
   ratchet.
2. **`lab/bridge/form_propose.py`** — 84%; the ADR-0008 capstone, but
   119 missing arcs.
3. **`lab/bridge/state_reindex.py`** — 83%; needs a dedicated file
   (currently rides `test_form_propose.py`).
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

# mlopt — a machine that writes its own rules, certifies them, and learns to play the game they create

> **Research artifact — proposed, not accepted.**  This repository and
> the accompanying draft in [`paper/`](paper/) are pre-review work:
> claims are measured and reproducible, but the manuscript has not had
> human revision or external review.

**Thesis.**  Compiler rule banks are *frozen*: every deployed rule was
written by a person, and even machine-generated banks (TASO, Isaria)
ship once and never re-enter.  mlopt is, to our knowledge, the first
*operational* substrate where programs, rules, and derivations are all
values in one calculus (n-morphisms) — so a minted rule re-enters the
bank as a legal move, and discovery closes into a **self-modifying
game** a learned policy can play under a certifying referee.

**The machine.**  Equality saturation over a torch-free IR + law
minting from measured need + an admission gauntlet (numeric oracle →
typed-pay gate → adversarial sweep of the guard's false region →
replayable certificate) + pluggable players (random · playbook ·
trained linear · neural).

**The claim is the loop.**  E-graphs (egg/TENSAT), replayable
derivations (CVC4/Lean), laws-as-values (Lisp) each exist elsewhere;
*mint → certify → re-enter as a legal move → learned play over the
growing bank* does not.  It buys **two-engine certification** (a wrong
law must fool e-graph *and* handl engine), **laws as values** (`Value`
construction, no codegen), **agnostic exits** (`Sink` → torch/ONNX/`.hnl`).

**What it found.**

- **`conv_bn_fold`** — the conv+BN weight fold every framework
  hand-codes, *composed by the machine*, certified, shipped — one of
  **13 machine-admitted laws** in the default bank.  Bitwise-equal
  through an independent runtime (the handl tree-walk engine) at ≈−5 %
  per-call wall in the deployed regime — a kernel-scale fp64 result,
  not a model-scale speedup claim.  Full account:
  [`bench/RESULTS.md`](bench/RESULTS.md).
- **Rediscovered, not encoded:** Blelloch's O(log T) scan,
  FlashAttention's combine, RetNet's chunkwise form — all fall out of
  generic algebraic laws, not a pattern library.
- **The referee has teeth:** 212 bad accepts refused, four false twins
  rejected, a fabricated speedup caught — and a real lowering bug no
  test covered.

**Learned play.**  A trained linear prior ranks law candidates at
**~4.7×** the tuned heuristic's average precision (AP 0.210 vs 0.045,
held-out); a distilled contraction policy beats `opt_einsum`'s
randomized greedy (0.91×).  The neural arm's residual gap is ordering
credit: a RankNet objective on the reward's label reaches AP 0.1941
(above the trained-linear 0.128, under the page-aware 0.210) and a
ListNet-graded drain earns heuristic-parity reward (0.3549 vs 0.3595)
— parity, not yet a win
([`docs/results_neural_rank.md`](docs/results_neural_rank.md)).

**The novelty — falsifiable.**  "Novel" is a closed-world claim:
cert ✓ + beats a registered baseline + underivable from
`known_bank()` (which now includes the prior-art preset) — *not*
"novel to humanity", which stays an open-world lit-review.  The sieve
ran 24 candidates over the 242-rule bank (the 31-case scorecard is
31/31 found+cert): **review queue empty** — the two prior review items
(`mul_unsq_nat`, `softsign_fold`) demoted to derivable once their
catalogued duals were registered as prior art; 6 derivable, 6 unpaid,
12 unmeasured — [`docs/results_novelty_sieve.md`](docs/results_novelty_sieve.md).
On the substrate side: TASO / Isaria / Ruler / Enumo mint + verify
offline; CompilerGym / Omelette / MCTS-GEB / Aurora learn over fixed
banks — none makes the minted bank itself an object of play.

**Honest scope.**  Whole models ≈1× (parity); on strictly-real
modules the yield is 0 so far — finds lean on purpose-built workloads;
the machine bank alone wins 3/22 vs human 4/22 (the union wins 6/22 —
complementary, not dominant; that bar measured the 22-model corpus —
the held-out zoo has since grown to 53 workloads); the learned arms cannot yet learn *when
to quit* — the blocker is observability, not the reward's price.  Wins
are structural and regime-bounded, inference-only, compile-time
search.

**Open — now measured (the Q-series).**  Unnamed structure on real
corpora: the rewrite-only mint saturates at **known-art** — 53
held-out zoo modules → 101 viable mints → **8 admitted, every one
classified known-art** (the paying folds it found — the norm kernels,
the sdpa/glu folds — were already catalogued)
([`docs/reports/results_overnight_real_v6.md`](docs/reports/results_overnight_real_v6.md)).
The **primitive arm breaks the ceiling**: `--corpus zoo-prim` mints
new `synop_*` ops for recurring manual shapes no bank op denotes, so
the fold is novel *by construction* — **21/24 admitted, all 21
classified machine-original**: certified, paying (launch 0.36–0.80x),
underivable.  What it found is honest: the shared
"sum-of-gated-products" skeleton of the SSM family (DeltaNet / GLA /
Mamba / RG-LRU — four published architectures, one discovered core)
plus the 14-module `a*b+c*d` shape — real cross-architecture
structure, named by the machine.  **Not** "novel to humanity":
elementwise fusers do this implicitly; what's new is the named
primitive + certified law
([`docs/reports/primitives_finds.md`](docs/reports/primitives_finds.md);
an earlier run surfaced and fixed an emit-time RHS-sanity substrate
bug, eqsat `2d6e4eb`).  New algorithms: **Strassen NO-GO** — the board
proves rank-7, every player stalls at 8
([`docs/results_strassen.md`](docs/results_strassen.md)).  Multi-step
credit: **yes, where chain density permits** — learned arms close
mint→apply chains until spontaneous chains thin out
([`docs/results_credit_chain.md`](docs/results_credit_chain.md)).
Transformer scale: **walls mapped** — matcher ~11k nodes,
verify-forward RAM, fp64 overflow
([`docs/reports/results_scale_ladder.md`](docs/reports/results_scale_ladder.md)).
Strassen fails where chains succeed → the frontier is **search depth
and density**, not machinery.  Still open: **can a player learn in a
game that grows under it?**

## Architecture

```text
lift (export → IR)  →  e-graph + gauntlet  →  lower (verified module)
                          ▲                        │
        laws as data: the shipped bank,            sinks: torch / onnx /
        minted composites, premise closures        handl kit / CUDA graph
```

- **Lift.**  Adapters export a model to the torch-free IR
  (`mlopt-core` imports no torch/numpy — enforced by import-linter).
  Torch and ONNX backends exist; ONNX landed with zero core changes.
- **E-graph + gauntlet.**  Saturation generates the equivalence class;
  extraction prices it.  Discovery runs *on the library itself*:
  census → propose → verify → measure → rank → emit → admit
  (`mlopt_discovery`), with per-artifact verification boundaries —
  numeric oracle, typed-pay gate, adversarial guard sweep, closure
  bound, certificate replay.
- **Lower.**  The extracted program lowers through a `Sink` port and is
  numerically verified against the source (fp64).
- **Handl composition.**  `packages/mlopt-kit` assembles the
  handl-language stack as dependencies — mlopt terms emit as `.hnl`
  programs (`bench/drivers/kit/term_hnl.py`) and evaluate through the kit's
  `hnl-run` door, so an optimization *decision* is priced by the same
  runtime that executes the term.

## Reproduce

```sh
uv sync                                                  # the dev environment

# the flagship, three surfaces
.venv/bin/python -m pytest tests/test_kit_bridge.py -q   # 8 e2e tests (~27 s; builds hnl-run if absent)
.venv/bin/python bench/drivers/probes/conv_fold_premise_pay_probe.py   # mint → transport → truth → pay → gauntlet (~13 s)
cargo test -p mlopt-kit                                 # the Rust consumer, 31 tests

# the day-to-day gate
.venv/bin/python -m pytest -m "not slow"                 # dev lane, 6,485 tests
scripts/gates.sh                                         # static gates + suite
```

`python examples/quickstart.py` runs the end-to-end pipeline on a toy
module (~15 s, CPU); `python -m bench list` catalogs the benchmark
harness.

## What's inside

| evidence | pointer |
|---|---|
| measured headline results + reproduction | [`bench/RESULTS.md`](bench/RESULTS.md) |
| the paper draft (quarto → PDF) | [`paper/`](paper/) |
| the mechanism — syntax → structure → search → certificate | [`docs/mechanism.md`](docs/mechanism.md) |
| the full results tables, wins *and* losses | [`docs/results.md`](docs/results.md) |
| controlled learned-policy baseline + ablations | [`docs/results_learned_baseline.md`](docs/results_learned_baseline.md), [`docs/results_learned_ablations.md`](docs/results_learned_ablations.md) |
| feature-schema leaderboard (representation vs weight search) | [`docs/results_schema_leaderboard.md`](docs/results_schema_leaderboard.md) |
| the working paper draft with the evidence table | [`docs/paper_draft.md`](docs/paper_draft.md) |
| the three costs — how the engine knows a form is *better* | [`docs/cost-model.md`](docs/cost-model.md) |
| the API walkthrough + the frozen public surface | [`docs/api.md`](docs/api.md), [`docs/public_api.md`](docs/public_api.md) |
| the evaluation dimension (fourth axis of ADR 0003) | [`docs/evaluation.md`](docs/evaluation.md) |
| discovery scorecard / player leaderboard / limits table | [`docs/scorecard.md`](docs/scorecard.md), [`docs/leaderboard_players.md`](docs/leaderboard_players.md), [`docs/limits_table.md`](docs/limits_table.md) |
| decision records with falsification paths | [`project/adrs/`](project/adrs/) |
| the measured retros each claim cites | [`project/retros/`](project/retros/) |
| repo map + verification gates | [`AGENTS.md`](AGENTS.md) |

## License

MIT.

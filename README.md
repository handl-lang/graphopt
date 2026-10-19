# mlopt — certified optimization of tensor programs, with machine-discovered laws

> **Research artifact — proposed, not accepted.**  This repository and
> the accompanying draft in [`paper/`](paper/) are pre-review work:
> claims are measured and reproducible, but the manuscript has not had
> human revision or external review.

**The problem.**  Every optimization a compiler knows was written by
a person — a hand-derived rule, encoded per framework.  The rule
library is the bottleneck: humans are slow, and the pairings nobody
names never get found.

**The idea.**  For a machine to *author* optimizations, four things
must exist: rules it can write (data, not code), a referee to check
them, a score to rank them, and a world — a runtime — that programs,
rules, and proofs all live inside.  A categorical substrate delivers
all four at once: a tensor program is a morphism of a category, the
category's equations are rewrite laws as data, derivations are
checkable certificates, and the handl runtime is the world the whole
loop executes in.  The consequence: discovery closes into a
well-defined *game* — moves, referee, score, board — and a game can be
played by learned policies (the AlphaGo Zero lineage).  On top of that
substrate, mlopt *mints* new laws from measured need — guard
synthesized, adversarially swept, admitted through a staged gauntlet —
and ships them verbatim in the default rule set.

**The contribution is the connection.**  Equality saturation
(Tate/egg/TENSAT), rule inference (Ruler), and learned play on
perfect-referee games (AlphaGo Zero; contrast AlphaTensor, which
searches one fixed structure) each existed separately.  mlopt connects
them: a machine that discovers, proves, prices, and ships its own
compiler rules — and can learn to play the discovery.

The flagship result: **`conv_bn_fold`, a machine-minted composite, is a
shipped law.**  The engine minted the eval-mode weight fold
`batch_norm(conv2d(x,W,b),…) → conv2d(x,W′,b′)` — the rewrite a hand
optimizer writes — as a composite over premise laws, transported its
guard out of the evidence store, proved it premise-derivable with a
replayable certificate, measured its pay, and shipped it.  The folded
form then evaluates **bitwise-equal** to the spelled program through an
independent runtime (the handl tree-walk engine, via `mlopt-kit`) at
**≈−5 % per-call wall** in the deployed regime (fold amortized to
load; the un-amortized/inline fold arm runs *slower* per call);
the cost model agrees: `dag_cost` 36,864 vs 55,296).  Full account:
[`bench/RESULTS.md`](bench/RESULTS.md).  **Scope, stated plainly:** this
is a kernel-scale result on fp64 fixtures — certified rewrite discovery
and runtime-priced deployment, not a model-scale speedup claim.

**Guidance, as a game.**  The same substrate that makes laws data also
makes discovery a well-defined game — and the deeper property is the
*closure*: programs, rules, and proofs are n-morphisms in one calculus,
so the system produces the values it consumes (minted rules re-enter
the bank; derivations replay as terms).  Rule applications are
well-formed moves, the gauntlet is the referee, cost is the score — so
learned policies can play the search.  Two shipped linear priors already beat
hand-tuned baselines on held-out boards: the open-game policy ranks
law-shapes with ~3× the heuristic's average precision, and a distilled
contraction policy beats `opt_einsum`'s randomized greedy (0.91× its
cost).  Neural arms currently lose to the linear/distilled players —
reported as measured, not tuned.

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
  programs (`lab/kit/term_hnl.py`) and evaluate through the kit's
  `hnl-run` door, so an optimization *decision* is priced by the same
  runtime that executes the term.

## Reproduce

```sh
uv sync                                                  # the dev environment

# the flagship, three surfaces
.venv/bin/python -m pytest tests/test_kit_bridge.py -q   # 8 e2e tests (~27 s; builds hnl-run if absent)
.venv/bin/python lab/probes/conv_fold_premise_pay_probe.py   # mint → transport → truth → pay → gauntlet (~13 s)
cargo test -p mlopt-kit                                 # the Rust consumer, 21 tests

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
| the working paper draft with the evidence table | [`docs/paper_draft.md`](docs/paper_draft.md) |
| decision records with falsification paths | [`project/adrs/`](project/adrs/) |
| the measured retros each claim cites | [`project/retros/`](project/retros/) |
| repo map + verification gates | [`AGENTS.md`](AGENTS.md) |

## Honest scope

- **Kernel-scale, not model-scale.**  Wins are structural (folds,
  recurrences, carrier lifts) and regime-bounded; fused-op workloads
  (`nn.Conv*`, `sdpa`) expose nothing and sit at parity.  Losses are
  kept in the same tables as wins.
- **Sparse discovery yield** — ~1.5 % of the exposed corpus class
  carries fresh paying structure; 12 machine-admitted laws ship beside
  the hand-written kernel.
- **Inference only; compile-time search** — seconds per block, minutes
  at large sequence lengths.

## License

MIT.

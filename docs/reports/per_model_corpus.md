# Per-model corpus — closing the "corpus ≠ arbitrary models" gap

Date: 2026-11 (worktree).  Module: `mlopt_discovery.model_corpus`.
Tests: `tests/test_discovery_model_corpus.py` (11 tests).

## The gap

The discovery measurement corpus is fixed: the bench law cases
(`impact._bench_cases`), the `mlopt_torch.models` registry
(`impact.model_cases`, ~40 exports), the intake side-file
(`intake.load_cases` → `intake_corpus.json` in the cache dir) and the
held-out `zoo` registry (73 `Workload` thunks).  `opt.optimize(model, x)`
never feeds the target model into any of it — the orchestrator is the
substrate repo and cannot import `mlopt_discovery` (hexagonal
boundary).  Pay claims were therefore only ever measured on zoo
*analogues*: `play._zoo_case` + `play.gen_probe` run the full
claim → measured-cost referee → deliver → verify → time pipeline, but
only over `zoo.zoo()`'s hand-picked list.  A model with novel
op-patterns got no coverage of its own shapes.

## What existed (the machinery map)

* **Census** — `census.corpus()` unions bench + models + intake
  side-file into `CorpusTerm`s; `op_tuple_census` /
  `shape_census` count `(op, child-op)` tuples and leaf-abstracted
  shapes.  `workload_gen.corpus_stats` mines the generative
  distribution (typed tuples, attr dicts, leaf slots, shape pools)
  off `TermCase`s.
* **Intake of a real model** — `TorchSource().to_ir(model, feed)`
  (`mlopt_torch.torch_bridge.export_to_ir`) lifts an `nn.Module` to
  `IR` + source tensors; `impact.model_cases` wraps the result as a
  `TermCase`.  `intake._ingest_one` does the same plus verify,
  writing `intake_corpus.json` + `intake_tensors.pt` — the
  *persistent* census seam.
* **Pay/referee** — `play._zoo_case(wl)` is the per-workload intake:
  export → `_mint_tiers` (the `TIER_SITES` collectors:
  `_fusion_sites`, `_decayed_sites`, `_all_scan_sites`,
  `_reduction_sites`) → `_claim_measurable` split →
  `genkernel.profitable` measured-cost referee (`reps=20`,
  `_triton_kernel_for` / `_claim_denotation_for` hooks) →
  `gen_sink`.  `play.gen_probe(case)` then boards the case
  (`_gen_board` → `MetaArena`), plays claim→saturate→extract,
  delivers through `hybrid_sink`, verifies numerically and times
  baseline vs `torch.compile` vs generated kernel — the pay/decline
  verdict row.
* **Generation** — `workload_gen.mutant_term` / `resampled_term` +
  `valid_term`'s five gates (mint contract, typing, sink-supported
  ops, `TorchConcreteEval` eval, novelty vs corpus keys) +
  `term_to_case` plumbing.  This is IR-term-level generation — no
  `nn.Module` needed.

`optimize()` itself lives in `handl-lang/eqsat`
(`mlopt_orchestrator.optimize.Optimizer`) — a pinned git dep that
never imports `mlopt_discovery`.  The wiring is therefore
caller-side (below).

## The wiring — `mlopt_discovery.model_corpus`

`model_corpus(name, model, feed, *, n_mutants, seed, corpus, dtype,
reps, gen_reps, probe, mutant_headroom)`:

1. **Export once** — `export_ir` runs `TorchSource().to_ir` at the
   intake dtype (`fp64` default, `fp32` knob like `_zoo_case`),
   returning `(IR, leaves, feed-tuple)`; `_term_case` wraps it as a
   `TermCase` (`inputs=ir.inputs`, `feed` positional, `param_vals`
   = source tensors — the same plumbing `model_cases` builds).
2. **Census** — `coverage_report` diffs the model's op-tuples and
   shapes against `default_corpus()` (`census.corpus()`:
   bench + models + intake side-file) on the census's own keys.
   Report: model totals + `novel_op_tuples` / `novel_shapes`.
3. **Generate** — `generate_mutants` mutates *the model's own term*
   (`workload_gen.mutant_term`) with donor stats mined over
   `corpus + model`, so the model's novel subterms join the
   graft/lift pools.  Every candidate passes `valid_term`'s five
   gates; `term_to_case` wraps survivors.  Node budget:
   `workload_gen._MAX_NODES` (80) is corpus-sized — mutants of a
   real model legitimately run larger, so `model_corpus` passes
   `max(model_nodes + mutant_headroom, 80)` capped at 240, applied
   as a scoped override inside `generate_mutants` (`finally`
   restored).  `_MAX_NODES` gained an `int` annotation in
   `workload_gen.py` (was an inferred `Literal[80]`) to allow this.
4. **Board** — `model_board` seats the model's *own* eager module
   as `ex["model"]` (the baseline `gen_probe` times — identical to
   `_zoo_case`); `mutant_board` seats `TorchSink().lower(ir,
   param_vals)` — the spelled `IRModule`, which *is* the denotation
   the delivered kernel is verified against.  Both funnel through
   `_case_tail`, which mirrors `play._zoo_case`'s intake tail call-
   for-call: `_mint_tiers` → `_claim_measurable` split →
   `gk.profitable(reps=gen_reps, kernel_for=_triton_kernel_for,
   denotation_for=_claim_denotation_for)` → `gk.gen_sink` → the
   `(name, term, {"gen": n}, tag, supported, ex)` tuple with
   `ex["_gen_claims_gated"]=True` so `_claim_cost_gate` doesn't
   re-pay the referee.
5. **Probe + verdict** — `play.gen_probe(case, reps=reps)` per
   board; `_verdict` classifies each row `pay` (delivered ∧
   verified ∧ speedup > 1) / `decline` / `verify-fail`.  Probe
   exceptions become honest `delivered: False` rows.

### The `_case_tail` mirror — hook for the pay-file owner

`_case_tail` duplicates ~50 lines of `play._zoo_case`'s tail
because `_zoo_case` is keyed on a `Workload.build()` thunk — a
generated IR term cannot enter it.  The clean fix is a
`play`-side extraction:

```python
def _case_tail(ir, leaves, feed, model, label, *, gen_reps=20): ...
```

which `_zoo_case` and `model_corpus.mutant_board` both call.  That
edit belongs to whoever owns `play.py`; until it lands the mirror
calls the *same* functions (`_mint_tiers`, `_claim_measurable`,
`profitable`, `gen_sink`) so behaviour cannot drift silently — but
it should be unified when the file is next touched.

## Measured result — synthetic novel model

`_GatedRecurrentPool` (test fixture): fixed-unroll gated recurrence
(`tanh(x@W_in + h@W_rec) * sigmoid(h@W_up)` per step, `torch.stack`)
→ single-head additive attention (`softmax(q@kᵀ/√d) @ v`) →
sigmoid-gated residual → squared bilinear mean-pool.  No zoo or
intake workload spells this arrangement.

Measured on the T=6 variant of the same architecture (the test
fixture ships T=4 for speed) vs the **bench-only** corpus (66
terms): **19 novel op-tuples / 67 novel shapes**,
including `mul(tanh, sigmoid)` (the gated update),
`matmul(select, ·)` (unrolled recurrence reads),
`matmul(softmax, matmul)` (attention pool), `mean(mul)` (bilinear
tail).  vs the full corpus the count shrinks (intake/zoo cover more
ops) but the *compositions* remain model-specific — the report
prints them per-run.

Pipeline run (`n_mutants=4, seed=1, corpus=bench, reps=2,
gen_reps=2`): **4/4 mutants accepted** (10 attempts, 6 gate
rejects), 5 boards total, all `delivered=True, verified=True`;
verdicts `pay=0, decline=5` — speedups 0.066–0.847 vs eager.  The
verdict is honest: on 2×6×16 fp64 launch-bound shapes the generated
kernels verify but don't pay; one mutant reached 0.847×.  Larger
dtypes/shapes is where the zoo tables show pays — the machinery is
shape-agnostic, the verdict is not.

## Wiring `opt.optimize` — the documented boundary

`Optimizer.optimize(model, x)` cannot call this module (substrate
repo, no `mlopt_discovery` import — by design).  Two honest
compositions:

* **Caller-side (today, zero eqsat changes):**

  ```python
  from mlopt_discovery.model_corpus import model_corpus
  rep = model_corpus("my_model", model, x, n_mutants=8, reps=50)
  opt_mod, stats = opt.optimize(model, x)
  ```

  The report's `coverage.novel_op_tuples` says what the corpus
  doesn't cover; `rows`/`verdicts` say what the generated kernels
  would do *on this model's shapes*.

* **Orchestrator hook (one line, eqsat repo):** an optional
  `corpus_probe: Callable[[str, Module, Any], dict] | None = None`
  parameter on `Optimizer.optimize`, invoked after export with
  `(name, model, example_inputs)` and its report merged into
  `stats["corpus_probe"]`.  `mlopt_discovery.model_corpus.model_corpus`
  is the drop-in callable.  Not implemented here — eqsat is a
  separate pinned repo; this is the documented seam.

## Boundaries — what cannot be auto-generated

* **Dynamic shapes.**  `torch.export` needs concrete example
  inputs; `None`/symbolic dims and data-dependent control flow
  reject at the boundary.  Like `intake`, a rejection is recorded
  (`export_ir` raises; the CLI surfaces it), never skipped.
* **Data-dependent structure.**  Mutants recombine *spelled* IR
  subterms; they cannot invent ops the sink doesn't lower
  (`valid_term` gates on `supported_ops`) or shapes no corpus/graft
  donor attests (`pool_by_shape`/`leaf_at` are corpus-keyed).
* **The 80→240 node window.**  `mutant_headroom` bounds mutant
  size; a model whose term exceeds ~240 nodes gets `n_mutants=0`
  effectively (all candidates gate-reject) — the honest outcome is
  "corpus coverage measured, generation declined", not padding.
* **Probes are not census members.**  Mutants inform this run's
  verdict table only; they are *not* appended to
  `intake_corpus.json` — a generated term has no `nn.Module`
  provenance to record.  Re-running `python -m
  mlopt_discovery.workload_gen` remains the path that feeds the
  standing census pipeline delta.
* **fp64 convention.**  Default intake dtype is fp64 like the zoo
  tables; `dtype="fp32"` is available but the IR is identical —
  only leaves/feeds move (same caveat `_zoo_case` documents: 12/65
  zoo modules flip sign-sensitive verdicts between dtypes).

## CLI

```sh
python -m mlopt_discovery.model_corpus \
    --module my_pkg.models:build --mutants 8 --reps 50
python -m mlopt_discovery.model_corpus \
    --workload zoo:GatedAttentionUnit --mutants 4 --corpus bench
python -m mlopt_discovery.model_corpus \
    --module my_pkg.models:build --input "2,8,16" --no-probe
```

`--workload` resolves `zoo:NAME`, `intake:NAME`, or a bare zoo
name; `--module` imports `pkg.mod:callable` returning
`(model, feed)` or a bare module plus `--input "d0,d1;…"`;
`--corpus bench` swaps the donor census for the bench cases (fast);
`--json` writes the machine-readable report.

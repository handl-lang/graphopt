# `corpus_probe=` — wiring per-model corpus coverage into `optimize`

Date: 2026-11 (worktree).  Companion to `per_model_corpus.md`, which
documented this seam as "one line, eqsat repo".  It has now landed in
both repos; this report records what was built, the contract, and the
verification.

## The seam (eqsat — `handl-lang/eqsat`)

`mlopt_orchestrator` stays backend-neutral: it cannot import
`mlopt_discovery`, so the hook is an injected callable.  In
`packages/mlopt-orchestrator/src/mlopt_orchestrator/optimize.py`:

* **`search(..., corpus_probe=None)`** — new keyword-only parameter.
  Invoked **once, right after `source.to_ir` export** (Phase 1), as
  `corpus_probe(name, model, x)` where `name` is the model's
  `__name__` or `type(model).__name__`.  Whatever it returns is
  recorded verbatim under `stats["corpus_probe"]` (set alongside
  `rule_fires`/`criteria`, ~line 2239); `lower`'s
  `stats = dict(result.stats)` carries it into the `LowerResult`, so
  `mod, stats = opt.optimize(...)` sees it.  Probe exceptions
  propagate — an opt-in diagnostic fails loudly, never silently.
* **`Optimizer.optimize` / `Optimizer.search` / `discover`** — no
  code needed: `Monolithic.run` partitions `**kw` through
  `_SEARCH_KW`, which is `inspect.signature(search)`-derived, so
  `corpus_probe` flows to `search` automatically.
  `Autotuned` forwards it through its inner `.optimize` call for the
  same reason.
* **`Compositional`** — `_optimize_compositional` gained the same
  parameter, scoped to the **whole model**: it runs once before the
  per-block loop (`corpus_probe(name, model, example_input)` — never
  per block, a block's captured input is not the model's feed) and
  lands in `stats["corpus_probe"]`.  Without this the kwarg would
  have hit the strategy's unknown-keyword `TypeError`.

Default path is **unchanged**: `corpus_probe=None` runs nothing and
records nothing — `stats` has no `"corpus_probe"` key (pinned by
`test_corpus_probe_default_is_inert` in eqsat's
`tests/test_backend_neutral.py`, which also covers the
recorded-report and whole-model-compositional cases).

## The adapter (graphopt — `mlopt_discovery.model_corpus`)

`model_corpus.corpus_probe(*, dtype=None, **kw)` returns the
`(name, model, feed) -> report` callable the hook expects, with
`model_corpus`'s knobs bound (`n_mutants`, `seed`, `corpus`, `reps`,
`gen_reps`, `probe`, `mutant_headroom`).  Usage:

```python
from mlopt_discovery import model_corpus as mc
from mlopt_orchestrator import Optimizer
from mlopt_torch.backend import TorchBackend

opt = Optimizer(backend=TorchBackend())
mod, stats = opt.optimize(
    model, x,
    corpus_probe=mc.corpus_probe(n_mutants=8, reps=50),
)
rep = stats["corpus_probe"]        # coverage + mutants + rows + verdicts
```

A bare `model_corpus` is also a valid `corpus_probe=` value — its
`(name, model, feed)` signature matches — when the defaults suit.

### The dtype hazard the adapter exists for

`model_corpus.export_ir` casts the module **in place**
(`model.to(td)`, the `_zoo_case` intake convention).  Wired naively
with the `"fp64"` default, an fp32 model would be re-dtyped *before*
`lower`'s verify runs `model(*x)` on the original fp32 `x` — a
dtype-mismatch failure, or worse, a silently-changed reference.
`corpus_probe()` therefore defaults `dtype` to `_model_dtype(model)`
— the model's own floating-point dtype (`"fp32"`/`"fp64"`) — so the
intake cast is a no-op on the caller's module.  Passing
`dtype="fp64"` explicitly opts back into the zoo-table convention
and owns the re-dtype side effect, exactly like a direct
`model_corpus` call.  Mixed/other dtypes (fp16, parameterless
modules) fall back to `"fp64"` — documented, not silently handled.

## `_case_tail` — the play.py extraction (done)

`per_model_corpus.md` flagged `model_corpus._case_tail` as a ~50-line
duplication of `play._zoo_case`'s tail awaiting a `play`-side
extraction.  It was mechanical and same-package, so it landed:

* `play._case_tail(ir, leaves, feed, model, case, tag, *,
  gen_reps=20)` now holds the shared tail — `_mint_tiers` →
  `_claim_measurable` split → `gk.profitable(reps=gen_reps,
  kernel_for=_triton_kernel_for,
  denotation_for=_claim_denotation_for)` → `gk.gen_sink` → the
  `(case, term, {"gen": n}, tag, supported, ex)` tuple with
  `ex["_gen_claims_gated"]=True`.  `case`/`tag` are the tuple's two
  label slots (the zoo passes `f"zoo:{wl.name}"`/`wl.name`; the
  historical `reps=20` is the `gen_reps` default).
* `_zoo_case` is now export + one call — behaviour identical
  (verified by the existing `_zoo_case`/`zoo_probe` tests in
  `tests/test_discovery_play.py`).
* `model_corpus._case_tail` is a thin wrapper passing its `label`
  for both slots — signature kept for `model_board`/`mutant_board`.

The mirror is gone: one tail, one behaviour, no drift surface.

## Verification

* **eqsat** — `pytest tests/ -q -m "not slow"`: **1653 passed, 15
  skipped** (includes the three new `corpus_probe` tests in
  `test_backend_neutral.py`).  `ruff check`/`format --check`, `ty
  check`, `lint-imports` (3 contracts kept — orchestrator still
  imports no backend), `vulture`: all clean.
* **graphopt** — `tests/test_discovery_model_corpus.py`: 15 tests.
  Run against the **local eqsat sources** (PYTHONPATH over
  `eqsat/packages/{mlopt-core,mlopt-orchestrator}/src`, since the
  venv pins eqsat `main`): **15 passed** — the wired-path tests
  exercised a real `optimize(..., corpus_probe=...)` end to end,
  including `mc.corpus_probe(...)` with board intake + `verify=True`
  and the fp32 no-re-dtype assertion.  Against the **pinned** eqsat
  (pre-hook): **13 passed, 2 skipped** — the two wired tests skip
  honestly via `_probe_supported()` until the eqsat pin carries the
  parameter; `test_default_path_records_no_probe` and the signature
  pin run either way.  `ruff`/`format`, `ty check`, `lint-imports`,
  `vulture`: clean.  `test_discovery_play.py -k "zoo_case or
  zoo_probe"`: 5 passed — the `_case_tail` extraction is
  behaviour-preserving.

## Caveats / follow-ups

* **Pin lag.**  graphopt's `[tool.uv.sources]` pins eqsat `main`;
  the wired tests skip until eqsat merges this and the pin bumps.
  Nothing else in graphopt depends on the hook — the default path
  never references it.
* **Cost.**  The probe runs synchronously inside `search`, before
  saturation — `model_corpus` defaults (`n_mutants=8, reps=50,
  gen_reps=20`) are a minutes-scale measurement, not a per-compile
  check.  Callers wanting coverage-only should pass
  `probe=False`/`n_mutants=0`/small `reps`; the adapter exposes all
  of `model_corpus`'s knobs.
* **Mutating probes are the caller's problem.**  eqsat documents
  that the probe owns its side effects; `mc.corpus_probe` solves the
  dtype one.  A future probe that mutates structure would need its
  own wrapper — the hook deliberately does not sandbox.
* **`Optimizer.discover`** inherits the hook through `.search`;
  `discover_alternatives` (module-level) does not take it — frontier
  enumeration isn't an optimize delivery.

# IRModule memory: liveness releases + param_values dedup

Follow-up to `docs/reports/whisper_experiment.md` — the whisper-small
fp32 encoder OOM'd on the 4 GB RTX 2050 in two places: `lower()`'s
verify/lower footprint (~3× the weight set) and the first delivered
forward (the eval tape pinned **every** intermediate until the root
returned).  Both are fixed in `packages/mlopt-torch/src/mlopt_torch/
torch_bridge.py`; measured end-to-end below.

## What retained memory (root cause)

Three independent retention sites, all inside `IRModule`:

1. **`_EvalPlan`'s `vals` tape kept every slot alive.**  The plan
   flattens the term DAG into `(out_slot, op_idx, arg_slots, attrs)`
   steps over a per-call `v: list` (`torch_bridge.py` ~line 2714).
   Slots dedup DAG-shared subterms correctly, but nothing ever
   dropped an entry — peak activation memory scaled with *graph
   size*, not the DAG's live set.  `eval_term`'s recursive path has
   the same property through `memo_env` (every `Op` result is written
   back and held until return), but the hot path — `forward`,
   `_eval` with an empty memo, `_eval_fast` — all run the tape.
2. **`_param_values` duplicated every weight ~1×.**  `__init__`
   stored the caller's `param_values` dict verbatim, then
   `_build_params` `.clone()`d each registered name into a
   `torch.nn.Parameter` in `_param_map`.  Post-init nothing *needs*
   the original tensor — eval resolves through `_param_map` — so the
   dict pinned a second full copy of the weight set for the module's
   lifetime — together with the `SearchResult`'s exported leaf
   tensors that is the ~3× weight blowup the report measured.
3. **Verify itself is cheap once (1) and (2) are fixed.**  The
   orchestrator's `verify_module` (`mlopt_torch/report.py:168`)
   runs ref and opt forwards *sequentially* under `no_grad` and
   retains only the two output tensors — the OOM was never the
   comparison, it was the delivered module's own footprint.
   Incremental/op-by-op verify was therefore **not** needed.

## The fix

### Liveness releases in `_EvalPlan` (`torch_bridge.py`)

- `build` (~line 2856) now computes per-slot use counts across the
  tape, then walks steps in order: a slot's count hits zero exactly
  at its **last consumer** (the tape is a topological post-order, so
  last-in-tape-order = last-to-run).  `releases[k]` is the tuple of
  arg slots safe to drop after step `k` executes; stored as a new
  `releases` field parallel to `steps`.
- `_run_tail` (~line 2929) iterates `enumerate(self.steps)` and, after
  each op call returns and `v[o]` is stored, sets `v[i] = None` for
  `i in releases[k]`.  The output is stored **before** any arg is
  released, so an aliasing binding (out *is* an arg tensor) keeps the
  storage alive through `v[o]`.

Correctness for shared subterms is structural, not special-cased: a
DAG-shared slot has use-count > 1 and survives until its final
consumer; `add(m, m)` counts the slot twice and releases once, after
both uses in the same step.  Leaf slots (Var/Param/Const) release the
same way — dropping the tape's reference never drops the Parameter
(`_param_map` holds it) or the caller's input.

### `_param_values` aliasing (`torch_bridge.py` ~line 3212)

After `_build_params()`, `self._param_values` is rebound to a fresh
dict where every *registered* name maps to its `_param_map`
**Parameter object itself** — same storage, zero extra bytes, key
still present (callers and `_align_to`'s table mirror may inspect it).
The pre-clone source tensors are released.  Unregistered names
(unshaped Params, caller extras) keep their tensors, and the
caller's dict is never mutated — the rebind also stops
`_align_to`'s moves from writing back into it.

Because the same object can now sit under two keys, `_align_to`
(~line 3444) dedups moves by object identity (`moved_for`: id →
(old, replacement), the tuple pinning the old object so its id can't
be recycled mid-loop): an aliased entry rebinds to the *same* moved
Parameter instead of materialising a second device copy on CUDA.
Fused `fused_*` params from `_fold_weight_chains` are registered
names, so their pre-clone tensors are freed too.

## Measured: whisper-small fp32 encoder, 4 GB RTX 2050

Probe: `/tmp/whisper_mem_probe.py` (scratch; same `_align_all`
non-float-leaf workaround as `whisper_probe.py`).  Searched IR:
**596 slots / 405 steps / 595 releases** — the encoder term is small;
the OOM was pure retention.

| stage | before | after |
|---|---|---|
| `lower(verify=True)` | **OOM** | **0.5 s, verified=True**, post-lower alloc 1.00 G, peak 1.54 G |
| `lower(verify=False)` | fit (~1.01 G) | fit, alloc 1.00 G |
| first `forward` peak (`max_memory_allocated`) | **OOM** — retain-all simulation died allocating at >3.43 G reserved | **1.21 G peak**, alloc steady 1.00 G |

The "before" forward number is measured on the *same delivered
module* with `plan.releases` emptied to `()` tuples — identical tape,
identical bindings, only the release pass disabled — so the delta is
attributable to liveness alone.  `verify=True` succeeding is the
compound effect: the verify forward runs the same tape, and the
module now holds ~1× weights instead of ~2×.

## What still retains (honest limits)

- **`SearchResult res` keeps the exported leaf tensors** — the third
  copy of the weight set lives in the orchestrator's result object
  (eqsat repo, `mlopt_orchestrator`), outside this fix.  `del res`
  after `lower` is still the caller's lever.
- **`eval_term`'s `memo_env` path retains all intermediates** by
  contract — callers may inspect the memo.  It only runs for
  seeded-memo evals (the omd carrier's batched seeds) and unplannable
  terms, not the hot forward.
- **Autograd**: releases drop the tape's Python refs; under
  `enable_grad`, saved tensors stay alive via the autograd graph
  regardless — the win is for `no_grad`/inference delivery, which is
  the verify/lower/CUDA-graph path that OOM'd.
- **CUDA graphs**: releases run during capture identically; freed
  blocks return to the graph memory pool and replay reuses them —
  strictly beneficial, and `fwd again` post-restore verified correct.
- **Plan tape itself** (`steps`/`releases`/`fills`) is O(nodes) of
  small ints — negligible next to activations.
- Non-float tensor leaves: `_align_to`/`_mint_const` were being fixed
  concurrently (device-only move for int/bool leaves — the whisper
  report's gap #3); orthogonal to this change, noted only so the
  probe's `_align_all` workaround isn't mistaken for part of it.

## Tests

`tests/test_eval_plan.py` (+4): weakref probe that intermediate slot
tensors are dead after forward; shared subterm computed once per
forward and live until its last consumer; duplicate-arg release
counting; `_param_values` alias semantics (registered names are the
Parameter object, extras keep tensors, caller dict unmutated).
`ruff`/`ruff format`/`ty` clean on `torch_bridge.py`.  Slice run
(`test_eval_plan` + `test_torch_bridge_edges` incl. the CUDA
`_align_to` integer-leaf test, `test_export` + `test_trace` +
`test_omd_lower` + `test_substrate_domain_edges`, `-m "not slow"`):
all pass except `test_share_duplicate_param_slices_bf16` — the known
bf16 `.numpy()` bug in `mlopt_core/laws/pairing.py` (eqsat repo;
owned elsewhere, unrelated to this change).  Concurrent edits to
`_mint_const`/`_align_to` (non-float leaf device moves) landed in the
same file from the bf16 workstream; the `moved_for` dedup above is
what keeps this change compatible with them.

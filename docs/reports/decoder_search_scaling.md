# Decoder-scale search scaling — whisper-base decoder post-mortem

Follow-up to `whisper_experiment.md` ("decoder search did not
terminate in >20 min").  Question: is the wall (a) a fixable
blowup, (b) a scheduling/extraction problem, or (c) a genuine
complexity boundary?

**Verdict: (a) — a fixable blowup, and it is FIXED in this repo.**
The wall was never e-graph growth: saturation stayed at ~615
enodes the whole time.  The minutes went into *rule side-condition*
shape inference — carrier-law `check`/`derive` hooks resolving
bound metavariables to `_min_term` representatives (heavily-shared
Op DAGs) and then calling `mlopt_core.typing._shape_of` **without a
memo**, an exponential tree walk over a DAG that unfolds to
~6·10⁸ paths on the decoder IR.  One memo dict per module removes
it: full decoder search now completes in **~9–10 s** (was >20 min
and killed), single decoder *layer* in **~1.1 s**.

A second, unrelated blocker surfaced downstream: `lower(verify)`
dies on a `where` node whose third operand rides the `arg2`
positional-attr spelling while the torch binding demands three
positionals — a `torch_bridge` eval-binding gap (owned elsewhere;
documented below, not fixed here).

## Setup

- `openai/whisper-base` fp32, **CPU** (the e-graph/search is
  host-side; GPU unnecessary and the 4 GB card is irrelevant here).
- Decoder step wrapped to return `last_hidden_state`
  (`_DecWrap`, same as `whisper_probe.py`); input `(start_ids,
  encoder_hidden_states)` with `encoder_hidden_states` from a CPU
  encoder forward over 3000-frame log-mel.
- `Optimizer(backend=TorchBackend()).search(..., max_iterations=3)`
  — the stock path: `default_rules()` = `core.DEFAULT` **+ carrier
  `CARRIERS` preset = 331 rules** once `mlopt_carriers` is
  registered (132 without it — carrier registration is import-time,
  so probe scripts that only import `mlopt_orchestrator` see the
  smaller set; `mlopt_torch` pulls carriers in).

## Where the time actually went

Instrumentation (`bench/drivers/probes/decoder_search_rules.py`)
wrapped `EGraph.run`/`apply_rule`/`_iter_matches`/`_instantiate`/
`_min_term`/`union` and every rule's `check`/`derive`:

- `EGraph.run` iteration 0 marched all 331 rules; enodes stayed
  **590 → 614** the entire time.  Nothing was exploding in the
  e-graph.
- Matching was trivial: `matmul_t_concat`'s LHS produced **12
  candidate classes / 12 substitutions total** in 0.0002 s.
- The cost was inside `check matmul_t_concat` itself, per
  substitution: **0.47 s → 0.60 s → 4.72 s → 9.85 s → 74.55 s**,
  all returning `False`.  `decay_state_weight_exit` (also a
  carrier rule) was likewise stuck >2 min when `matmul_t_concat`
  was skipped.
- Stack samples caught the process inside
  `mlopt_core.typing._arange_shape` — i.e. deep inside
  `_shape_of`'s recursive inference, not the matcher.

### Root cause

Carrier-law side conditions resolve bound e-classes through
`EGraph._any_term_cached` → `_min_term`, which returns the
class's minimum *tree-size* member as an **interned Op object
graph** — a DAG (shared children are the same object).  On the
decoder IR the root class's min-term reports `size=619,638,490`
tree paths over only ~600 distinct nodes (measured:
`_min_term(root)` → `(term, 619638490)` in 0.0 s — construction
is cheap because objects are shared).

`om._vshape` → `xcarrier._xshape` then computes value shapes.  Its
own `_memo` is `id()`-keyed **per call** (fine), but every fallback
dispatch — `_shape_of(f)`, `_shape_of(t)`, and the surrogate
`Op.make` re-dispatch — went through `xcarrier._shape_of`, which
called `typing._shape_of(t)` with **`memo=None`**: no caching at
all.  `typing._shape_of` is explicitly memo-aware ("memoising
turns an exponential tree walk into a linear DAG walk") — the
carrier wrapper simply never threaded one.  Each `check` call
re-walked hundreds of millions of paths; cost grew per
substitution because different bound classes resolve different
huge cones.

Core (`mlopt_core.laws`) checks were unaffected because
`laws.base._shape_of` already threads the module-level
`_SHAPE_MEMO` — which is why only *carrier* rules hung, and why
the encoder (whose checks mostly hit core laws / smaller cones)
saturated in 3 s.

## The fix (landed)

`packages/mlopt-carriers/src/mlopt_carriers/xcarrier.py` and
`trace.py`: each module-local `_shape_of` now passes a
module-level content-keyed `_SHAPE_MEMO` dict to
`typing._shape_of` — the same precedent as
`mlopt_core.laws.base._SHAPE_MEMO` (terms are interned, so the
cache is sound; keys hold terms alive, matching the existing
contract).

Measured after the fix, stock path, no monkeypatching:

| stage | before | after |
|---|---|---|
| `search` (sat + pairing + lifts + extract) | >20 min, killed | **8.8–10.4 s**, `stop=fixed_point` |
| e-nodes / classes | 615 / 601 | 615 / 601 (identical) |
| `rule_fires` | — | `naturality_scalar_rev`×12, `id_add`×1 |
| `pairing_groups` / `nonlocal_lifts` | — | 1 / 1, `paired_extract=True` |
| single decoder layer (`decoder.layers[0]`, hook-captured inputs) | untested (same mechanism) | **1.1 s**, 88 enodes, fixed_point |

Probes (scratch, uncommitted): `bench/drivers/probes/
decoder_search_{probe,stacks,rules,phases,fixed}.py`.

## Honest caveats — what the fix does NOT give you

1. **Near-zero rewrite yield on decoder IR.**  Saturation added
   ~25 enodes; extraction is essentially the lifted graph plus one
   pairing group.  The eager-attention decode decomposition
   (`linear → reshape/transpose → matmul → softmax → matmul →
   linear`, plus `split`/`concat` QKVKV plumbing, `where`/`le`
   mask construction, `embedding`/`index`/`repeat`/`arange`
   position machinery) is mostly *outside* the law bank's LHS
   vocabulary — the SDPA/OM fold families want the fused `sdpa`
   spelling, not the eager decomposition.  Search terminating ≠
   optimization happening.  Yield work is a law-coverage problem,
   orthogonal to this fix.
2. **`lower(verify=True)` still fails** — see below.
3. **Latent unmemoized `_shape_of` remains upstream.**  eqsat's
   `mlopt_core.laws.pairing` helpers (`_wshape`,
   `_uniform_input`, `_gates_uniform`, `_cover_check`,
   `_select_tile`) call `typing._shape_of(t)` with no memo, and
   the carrier *lowerers* (`om_lower`, `omd_lower`,
   `decayed_lower`) have the same pattern on single extracted
   terms.  On this graph pairing survived inside the 8.8 s
   (its bound terms were shallow enough), but the same
   exponential applies to any deeper cone — recommend the same
   one-line memo upstream in eqsat, plus auditing lower-phase
   shape walks.  I did not edit eqsat (sibling repo, pinned dep).
4. **`rule_budgets`/`symmetry_budget`/`policy`/`stop="improving"`
   were never the lever here** — they bound enode *growth*, and
   there was none.  The existing controls are fine; the bug was
   below them, in side-condition evaluation.

## Second blocker found (not fixed — torch_bridge is owned elsewhere)

`optz.lower(res, inp, verify=True)` →
`IRModule._run_tail` → `TypeError: <lambda>() missing 1 required
positional argument: 'y'`.

The extracted term contains `where(cond, zeros, arg2=-3.4028e38)`
— 2 children + the third operand in the `argN` positional-attr
spelling (the decoder self-attention mask construction:
`le(arange, arange)` → `where` → `expand`).  The torch binding
(`torch_bridge.py:2310`) is `lambda c, x, y, *a, **kw:
torch.where(c, x, y)` — it demands three positionals, and the
eval plan passes attrs through as `**kw`, so `arg2` arrives as a
keyword the lambda can't use → TypeError.  `where` is not in
`ATTR_SCHEMA`, so the attr was never canonicalised and the
binding never reads it via `attr_of(kw, ..., "arg2")`.

This is presumably why the original probe never got this far —
search never returned.  It is an export-faithful term (scalar
operands legitimately ride `argN`); the gap is binding-side.
Fix options for the owner: schema `where` (or read `arg2` in the
binding and mint the scalar as a const operand the way creator
ops do).  Same audit probably applies to any binding whose op
can carry tensor-valued `argN` operands (`nonzero`? `searchsorted`?
`index` variants) — worth a sweep of bindings vs `argN`-capable
ops.

## Compositional strategy — does sub-block splitting help?

Yes, structurally: `mlopt_torch.composer._default_block_pred`
selects children of `nn.ModuleList`/`nn.Sequential`, and
`WhisperDecoder.layers` is a `ModuleList` of 6
`WhisperDecoderLayer`s — `optimize(decoder, strategy=Compositional())`
would optimize each layer independently (plus leaf modules like
`embed_tokens`/`layer_norm` as trivial blocks).  Measured: a
single layer's `search` completes in **1.1 s** post-fix.

But note the same caveat as (1): per-block search terminates
easily *now*; whether it *pays* depends on law coverage of the
per-layer eager-attention IR, which is the same thin coverage as
the monolithic case.  Compositional is a viable scaling path for
delivery (smaller verify/lower units also dodge the 4 GB
verify-duplication OOM that killed whisper-small), not a
substitute for coverage work.  Untested end-to-end
(`Compositional` also needs the composer port + per-block input
capture; my probe drove `search` on a hook-captured layer
directly).

## Bottom line for the CudaGraphRunner decode win

- Search scaling: **fixed** (memoization, ~10 s for the whole
  decoder, ~1 s per layer).
- Decode delivery is still blocked by two things, neither a
  search-scaling problem: (i) the `where`/`arg2` eval-binding
  bug above (torch_bridge owner), and (ii) near-zero rewrite
  yield on eager-attention decode IR — the laws want `sdpa`,
  export gives the decomposed form.  A `sdpa`-canonicalising
  lift (or export with `attn_implementation="sdpa"`) is the
  highest-leverage next step for actual decode wins; the
  `sdpa`-fold/OM carrier families already exist and are exactly
  what they consume.

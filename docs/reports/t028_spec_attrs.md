# T-028 leftovers — `vocab.attr_values` lands; statistical footprints stay documented

2026-11-13.  Agenda item T-028's remaining Stage-2 domain-spec gaps
were "attr values, statistical footprints" (recorded in
`docs/reports/2026-11-12-domain-spec-second-cut.md` §"Still open" and
`project/retros/domain-spec-stage2.md`).  This cut ships the first —
an honest, per-case attribute-*value* constraint — and pins the
second's boundary precisely instead of faking it.

## Gap 1 — attr values: CLOSED (`vocab.attr_values`)

`vocab.attrs` whitelists `(op, attr-name)` spellings; it could not
say *which* values an attr may carry ("only `clamp(min=0)` sites",
"`softmax` over the trailing dim only").  The new field:

```json
"vocab": {
  "attr_values": {
    "softmax": {"dim": -1},
    "amax": {"dim": [-1, 1]},
    "clamp": {"min": 0.0, "max": 1.0}
  }
}
```

Semantics (`domain_spec._parse_attr_values` /
`_attr_values_ok`):

* `{op: {attr: scalar-or-list}}` — a scalar is one allowed value, a
  list is the allowed *set*.  Allowed values must be JSON scalars;
  anything nested deeper is a **load error**, not a silent no-match.
* For every `Op` node whose `op` is listed, each listed attr must be
  **present** and `==` an allowed value.  An absent attr is a
  violation — "only `clamp(min=0)` sites" does not admit attr-less
  clamps.  Unlisted ops and unlisted attrs stay unconstrained.
* Non-scalar attr values (the corpus does emit tuple/list attrs —
  `reshape.shape`, `permute.dims`, `split.sizes`) can never satisfy
  a listed constraint; constraining such an attr excludes those
  spellings.  Documented, not hidden.

Applied through the existing seams, no new machinery:

* `_case_ok` — the per-case post-filter every term board already
  runs (`meta-arena`/`torch`/`gen`/`joint`/`search`/`fragment`
  case streams all funnel through it), so the invariant is exact on
  every emitted term.
* `_eligible_kinds` — on `meta-arena` the 240-case footprint census
  marks a kind ineligible when *any* sampled case violates, the
  same drop-on-escape rule the name whitelists already apply.  The
  census stays statistical (a rare violating draw can escape the
  sample); the per-case filter remains the guarantee — unchanged
  honest edge.
* `_VOCAB_KEYS` — `attr_values` joins `ops`/`attrs`/`rules`/
  `handler_tags` on exactly the five term boards; `open`/`player`/
  `bilinear` still refuse `vocab.*` outright.  Unknown spellings
  load-error per the whitelist pattern.
* `spec_summary` reports the resolved `{op: {attr: [values]}}`
  table under `vocab.attr_values`.

Packaged demo: `domains/attr-bounded.json` — a pointwise+
reduction meta-arena spec bounding `softmax`/`log_softmax`/`sum`
to `dim=-1`, `amax` to `dim ∈ {-1, 1}` (the corpus's `dim=0`
spelling fails), `unsqueeze` to `dim ∈ {0, 1}` (drops the
`law_unsq` kind's dim-2 views), and `clamp` to the nonnegative
`min=0.0, max=1.0` spelling (the symmetric `±1.0` clamp the corpus
also emits fails).  Verified: 13 eligible kinds, 12/12 sampled
cases satisfy the invariant.

## Gap 2 — statistical footprints: boundary pinned, not shipped

"Statistical footprints" would be distributional constraints on
generated corpora — e.g. "≥30% of cases carry op X", "dim draws
uniform over {0,1}", quantile bounds on term size.  Assessed
honestly: **not implementable as a spec field today**, and the
reason is precise:

* The generators expose no distribution surface.
  `meta_player.gen_cases(seed, n, coverage, kinds)` is a seeded
  *stream*: kinds are drawn uniformly
  (`builders[rng.randrange(len(builders))]`), and every
  distributional choice lives *inside* builder closures as
  hardcoded `rng.random() < p` thresholds and `_pick(...)` calls
  (`corpus_cov.py` — e.g. `_law_softmax`'s `keep = rng.random() <
  0.7`, `_law_rsqrt`'s `0.6`).  There is no weights table, no
  per-kind probability, no exposed choice-point list for a spec to
  name or re-seat.
* `corpus.kinds` can *restrict* the kind set (hard constraint —
  shipped), and `vocab.attr_values` can now *filter* values (hard
  constraint — this cut).  Neither can *shape* frequencies: a spec
  that wants "half the corpus silu-flavored" has nothing to bind
  to — post-filtering can only delete, and deleting to hit a
  target rate is rejection sampling with no acceptance-rate
  guarantee (the `_MAX_ROUNDS` oversample already returns short
  honestly rather than padding).
* What *would* make it expressible: builders that take their
  internal choice points as data (a `build(rng, params)` seam —
  probabilities, dim sets, keep-flags as spec-visible kwargs), or
  a `gen_cases` `weights=`/`dist=` parameter.  That is generator-
  side work in `meta_player`/`corpus_cov` (and the torch/gen
  module zoos have the same shape), not a `domain_spec` schema
  addition — shipping a `corpus.dist` field now would be exactly
  the "silently ignored knob" the per-board whitelists exist to
  refuse.

So the honest state: **hard constraints on names and attribute
values are spec-able; distributional constraints are not, and the
boundary is the generators' lack of a parameter seam** — recorded
here so a future cut knows the prerequisite is upstream of
`domain_spec`.

## Files

* `packages/mlopt-discovery/src/mlopt_discovery/domain_spec.py` —
  `_parse_attr_values`, `_attr_values_ok`, `attr_values` field on
  `DomainSpecFile`, `_VOCAB_KEYS` rows for the five term boards,
  `_case_ok`/`_eligible_kinds` wiring, `spec_summary` census,
  module-docstring schema bullet.
* `packages/mlopt-discovery/src/mlopt_discovery/domains/
  attr-bounded.json` — packaged demo spec.
* `tests/test_domain_spec.py` — `TestAttrValues` (10 tests):
  parse/normalization, five malformed-shape refusals, non-term-
  board refusal, unit predicate (value/missing/unlisted/nested),
  eligibility kind-drop, per-case invariant, vacuous-constraint
  honesty, packaged-spec end-to-end, summary census.
* `docs/site/how-to/define-a-domain.md` — `attr_values` in the
  per-board `vocab.*` column, a seam bullet, and the packaged-spec
  count (four → five).

## Verification

* `scripts/pytest-capped.sh .venv/bin/python -m pytest
  tests/test_domain_spec.py -q` — **88 passed**.
* `.venv/bin/ruff check` + `ruff format --check` on
  `domain_spec.py` — clean (the test file carries pre-existing
  format drift; `tests/` is outside the lint gate's
  `packages scripts examples` paths).
* `.venv/bin/ty check` — 0 errors.

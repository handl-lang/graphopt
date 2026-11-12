# 2026-11-12 — domains-as-data: Stage-2 self-driving, first cut

Stage 2's first cut: a **domain-spec format** — given a vocabulary
subset, a corpus selection, a ruleset profile and an incumbent
recipe, emit a whole playable `play` domain (cases generator,
board factory, player spec, scripted arm) registered through the
existing `spec_domain` seam.  "Drive somewhere new" becomes "write
a spec" — for the meta-arena board family.

## What landed

* **`mlopt_discovery/domain_spec.py`** (new) — the spec loader,
  validator, generator and registry bridge:
  * `DomainSpecFile` — the parsed, validated spec record (frozen
    dataclass; `raw` keeps the source mapping).
  * `load_spec(source)` — JSON path, packaged name
    (`domains/<name>.json` inside the package), or a plain dict.
    JSON only in v1 — stdlib, no new dependency.
  * `register_file` / `register_domain` — build the `Domain` and
    seat it in a `DOMAINS` dict (explicit `domains=` parameter —
    see the `__main__` caveat below).
  * `BOARDS` / `PLAYERS` / `INCUMBENTS` — the v1 registries:
    `board="meta-arena"` only, `player="meta"` only
    (`meta_player.META_SPEC`), incumbent ∈
    {`scripted`, `greedy`, `none`}.
  * CLI: `python -m mlopt_discovery.domain_spec --spec pointwise`
    prints the resolved census (eligible kinds, vocab sizes,
    recorded-only fields) plus a case sample — no episodes.
* **`meta_player.gen_cases(..., kinds=None)`** — a builder-kind
  filter over the shared base+coverage namespace; unknown names
  raise.  This is the corpus-selection seam the spec drives.
* **`play.py`** — `--domain-spec PATH|NAME` (repeatable) loads and
  registers spec domains before the domain lookup; `--domain`
  lost its static `choices=` (spec names are data, unknown at
  parse time) and validates membership after registration,
  returning 2 with the known-domain list on a miss.
* **`domains/pointwise.json`** (packaged spec) — the demo domain:
  the pointwise+reduction vocabulary (all 28 `ELEMENTWISE` ops, all
  20 `reduce_heads`, the kernel spellings `glu`/`rms_norm`/
  `swiglu`/`log_softmax`), `handler_tags=["silu","square"]`
  (swiglu claims unoffered), `rules="default"` (scan-extended
  cases excluded), incumbent `scripted`.
* **`tests/test_domain_spec.py`** — 28 tests: load/validation
  matrix, eligibility census, per-case vocab/rules invariants,
  determinism, handler-subset boarding, `gen_cases(kinds=)`,
  registry + the `play.main --domain-spec` path.

## The spec schema (v1)

```json
{
  "name": "pointwise",            // registry key, ^[a-z][a-z0-9_-]*$
  "board": "meta-arena",          // v1: only "meta-arena"
  "description": "...",
  "corpus": {
    "coverage": true,             // gen_cases coverage arm
    "kinds": ["silu", "..."]      // optional explicit subset
  },
  "vocab": {
    "ops": ["add", "mul", "..."], // optional op whitelist
    "attrs": ["op.attr", "..."],  // optional attr whitelist
    "handler_tags": ["silu"],     // optional HANDLERS subset
    "rules": "all|default|scan"   // or [rule-name, ...]
  },
  "shapes":  {...},               // RECORDED ONLY in v1
  "hardware": {...},              // RECORDED ONLY in v1
  "player": "meta",               // learned-arm DomainSpec selector
  "incumbent": {"kind": "scripted"}
}
```

## How spec → domain works

1. **Eligibility census** — `_eligible_kinds` samples 240 cases
   through the spec's own corpus selection and unions each
   builder kind's op/attr *footprint*; a kind may draw iff its
   footprint fits `vocab.ops`/`vocab.attrs` and its name survives
   `corpus.kinds`.  For `pointwise`: **18 of 26 kinds eligible** —
   `law_mm`/`law_glu`/`law_split`/`law_unsq`/`view_gap`/`scan`/
   `control` fall out on matmul/view/carrier ops.
2. **Case stream** — `_make_cases` draws via
   `gen_cases(kinds=eligible)`, then enforces the same bounds
   *per case* (ops ⊆ vocab, attrs ⊆ vocab, ruleset names ⊆ the
   rules allowlist) with oversampling rounds — the footprint only
   decides what may draw; the post-filter guarantees the
   invariant on every emitted term.
3. **Board** — `MetaArena(term, rules, cost_fn=count_cost,
   supported=sup, handlers=<subset>, max_nodes=20_000)` — the
   handler subset is seated directly; unlisted tags are never
   claimed.
4. **Player + incumbent** — `META_SPEC` supplies legal/featurizer/
   features/weights; `scripted` maps to `ScriptedPlayer`
   (saturate→extract), `greedy` to `GreedyPlayer`, `none` to no
   scripted arm.

## Demo — `play --domain-spec pointwise`

``
.venv/bin/python -m mlopt_discovery.play \
    --domain-spec pointwise --domain pointwise \
    --train-cases 8 --eval-cases 4 --budget 12

== play — domain pointwise ==
player           cases    reward    cost  cert  steps
-----------------------------------------------------
scripted             4     -0.00    3.00     4    2.0
random               4     -0.60     nan     0   12.0
learned              4     -0.35    1.00     1    9.5
learned-greedy       4      0.14    3.00     4    1.0
train decade means: [-0.6, -0.6, -0.6, -0.6, 0.04, -0.6, -0.6, -0.6]
```

A spec-named domain trains the learned arm, evaluates all arms and
prints the paired table — the full `play` pipeline on a corpus
that did not exist before the spec did.  (4 eval cases is a smoke
run, not a measurement; learned-greedy's +0.14 vs scripted's −0.00
is noise-band, recorded honestly.)

## Gotchas found

* **`python -m mlopt_discovery.play` double-import.**  `-m` runs
  `play.py` as `__main__` — a *different module object* than the
  `mlopt_discovery.play` `domain_spec` imports, so mutating
  `play.DOMAINS` through the import seats the domain in the wrong
  registry.  `register_domain`/`register_file` take an explicit
  `domains=` dict; `play.main` passes its own `DOMAINS`.  The same
  hazard applies to any future spec-driven registration under `-m`.
* **`choices=` had to go.**  Spec domain names are data; argparse
  could never enumerate them.  `--domain` is now free-form with a
  post-registration membership check (exit 2 + known list).

## Honest gaps (what v1 does NOT do)

* **One board family.**  `board` accepts only `"meta-arena"` — the
  symbolic probe-tuple corpus.  The torch-backed boards (`torch`,
  `gen`) need module builders, sink bindings and example inputs a
  JSON spec cannot yet name; `board` is reserved for them.
* **`shapes`/`hardware`/`dtype`/`device` are recorded, not
  consumed.**  Meta-arena terms are symbolic — `gen_cases` spells
  fixed `(4, 4)` vars and no tensor is ever materialized, so there
  is nothing for a shape distribution or a CUDA target to bind.
  The fields exist so a torch-backed board honours them without a
  schema break.
* **`attrs` whitelist is opt-in and shallow** — `(op, attr-name)`
  pairs only; attr *values* (dims, shapes, eps) are unconstrained.
* **Footprint sampling is statistical** — 240 cases per build;
  rare ops inside a kind can escape the footprint (the per-case
  post-filter catches them, at the cost of yield).
* **Player spec is fixed to `meta`** — `META_SPEC`'s feature schema
  and baked weights were trained on the full corpus; a narrow-vocab
  domain reuses them unmodified (features that never fire are
  inert, but the prior is mismatched — retraining per spec is
  future work).
* **No deliver hook** — meta-arena boards have no lowering path;
  `deliver` stays `None`.
* **Rules profiles are name-sets** — `"default"`/`"scan"`/explicit
  lists; there is no "rules minus these" subtraction profile.
* **Pre-existing test drift noticed**: `test_discovery_corpus_cov`
  hardcodes 131/153 rule counts; the eqsat law bank now ships 132
  DEFAULT + 22 scan = 154 (`add_unsq_pad_r` et al.).  Four
  assertions fail on `main` independent of this change — the
  counts need re-measuring, not patching.

## Files

* `packages/mlopt-discovery/src/mlopt_discovery/domain_spec.py` (new)
* `packages/mlopt-discovery/src/mlopt_discovery/domains/pointwise.json` (new)
* `packages/mlopt-discovery/src/mlopt_discovery/meta_player.py` (`kinds=` param)
* `packages/mlopt-discovery/src/mlopt_discovery/play.py` (`--domain-spec`, free-form `--domain`)
* `tests/test_domain_spec.py` (new, 28 tests)

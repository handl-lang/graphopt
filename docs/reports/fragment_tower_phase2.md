# Fragment tower — phase 2 progress note

Plan: `project/plans/pl-0029-plan-0033-the-fragment-tower-mlopt-vocabularies-as-language-data.md`
(phase 2 — denotations and minted handlers as Handl Models)
Host ADR: `host/project/adrs/0157-adr-0157-mlopt-on-the-fragment-tower-domain-vocabularies-as-language-fragments.md`
Ledger: `T-017` (open — phases 3–5 remain)

## What phase 2 delivered

`mlopt_discovery.modelproj` — the projection that spells a minted
primitive's *meaning* as a Handl expression over `host-tensor`, plus
the fp64 gate that decides whether the spelling is shipped.  The
canonical fragment directory now carries a seventh file,
`vocab/models.yaml`: one Model row per `synop_*` denotation and per
minted handler record, each with its source semantic payload
(`spec`/`body`), the emitted `.hnl` text (or an honest `unsupported`
record), and a `verdicts` map written by the gate.

A found kernel's meaning now lives in the fragment tower, not only
its IR shape: `denotation:synop_*` rows carry the definitional
premise's RHS spec *and* its `tensor.*` rendering;
`handler:synop_*` rows carry the serialized `handler_body` descriptor
*and* its elementwise `tensor.*` rendering.  `TorchSink` remains the
delivery oracle — nothing here changes what ships to torch; the gate
only decides which Models may later be called "handl-shipped".

### `modelproj` — spec/body → `.hnl` over `tensor.*`

Table-driven, not term-specific:

- `spec_to_hnl(spec)` renders a genkernel spec tuple as a `.hnl`
  expression: binary ops → `tensor.<op>(a, b)`; `concat` →
  `tensor.concat([…], dim)`; shape ops (`reshape`, `transpose`,
  `flatten`, `unsqueeze`-free set, …) → `tensor.<op>(x, <attr
  slots>)`.  Attr dicts become positional literal slots via
  `_ATTR_OPS`; non-call lists are data and emit as literal lists.
- `handler_body_to_hnl(body)` accepts only the `ew` family — its body
  is a plain spec, projected by the same path.  Every other family
  tag (`gather`/`family`/`contract`/`fold`/`reduce`/`norm`/
  `fold_zip`/`fold_emit`) raises `ModelUnsupportedError` naming the
  tag: those semantics need ops or control flow `tensor.*` does not
  carry.
- Anything the surface cannot honour — unknown op, attr mismatch,
  bad arity, non-finite literal, metavar outside the name map —
  raises `ModelUnsupportedError` naming the gap.  The collector
  records `hnl: null` + `unsupported: <reason>`; **no fabricated
  model is ever emitted**.
- Metavar names are sanitized (`v_<name>`, collision-suffixed) so
  arbitrary spec metavariables bind as valid `.hnl` identifiers.
- `hnl_program(hnl, feed, mvars=…)` emits the runnable program:
  sorted `name :: tensor.reshape(tensor.from_list([data]), [shape])`
  bindings plus the model expression — the same marshaling
  `bench/drivers/kit/term_hnl.py` established for `hnl-run`.

### `verify_model` — the phase-2 fp64 gate

For each Model row the driver (`scripts/verify_vocab_models.py`)
reconstructs the torch-side reference — the denotation spec itself
for `denotation:` rows, the handler *pattern* for `handler:` rows —
and runs `handler_synth._domain_draws` for the measured bound-site
shapes plus adversarial redraws (the same paired fp64 draws every
candidate sees).  Each draw is marshaled through `hnl_program`,
evaluated by the real `hnl-run`/`CatoptKit` path (injected
`eval_hnl`), decoded from the `{"__tensor__": …}` wire, and compared
with `torch.allclose` at the shipped verify floor
(`rtol=1e-9`, `atol=1e-12`).

The verdict records `status` (`handl-shipped` /
`python-shipped`), `rtol`/`atol`, measured `max_err`/`max_rel`,
`sites`, `detail`, and the `hnl` text — the ADR-0157 provenance
record.  `handl-shipped` requires **every** bound site to hold;
anything else is `python-shipped` with the measured error and the
refusal named.  Verdicts are written by
`vocabdir.upsert_model_verdicts` into `models.yaml`'s `verdicts`
map — they are provenance, not owner rows, so re-emission
(`emit_all`) preserves them and `check_vocab` stays clean.

### `vocabdir` — seventh canonical file

`models.yaml` joined `CANONICAL_FILES` alongside `ops`/`laws`/
`handlers`/`schemas`/`prims`/`families`.  `collect_models` emits the
rows from `shipped_primitives.spec` + `minted.prim_records()`;
`load_vocab` exposes `vocab.models` and `vocab.model_verdicts`;
owner-drift checking covers model rows; `emit_all` round-trips them.
The shipped `lawdata.HANDLERS` interpretation entries
(`silu`/`square`/`swiglu`) are deliberately absent — their kernels
are named torch primitives with no spelled body to project.

## Coverage — what made it into Models

`vocab/models.yaml`: **109 rows** — 57 `denotation:` + 52 `handler:`.

| class | rows | `handl-shipped` | `python-shipped` |
|---|---|---|---|
| denotation (`synop_*` depth 1/2/3) | 57 | 52 | 5 |
| handler (minted `ew` bodies) | 52 | 52 | 0 |
| **total** | **109** | **104** | **5** |

Verification (`scripts/verify_vocab_models.py`, real `hnl-run`
evaluation, fp64 floor `rtol=1e-9`/`atol=1e-12`, adversarial redraws):

- **104 `handl-shipped`**, worst `max_rel = 0.0` across every bound
  site (max 8 sites/model); `max_err = 0.0`.
- **5 `python-shipped`**, all the same honest skip:
  `op 'unsqueeze' is not on the tensor.* model surface` —
  `denotation:synop_4c5296b3ba`, `synop_4c657cb2c0`,
  `synop_976f9836d7`, `synop_98a002d64b`, `synop_f08a53f9a0`.
  `host-tensor` has no `unsqueeze`; these denotations stay
  Python-side until the surface grows one (or the spec is respelled
  through `reshape`).

## Honest skips

- **`unsqueeze`** (5 denotations) — the only op gap hit in the
  shipped vocabulary.  Everything else in the minted space
  (add/sub/mul/div, reductions, matmul, conv, norm, pooling,
  reshape/transpose/concat families) is on `tensor.*`.
- **Non-`ew` handler families** — no shipped minted handler uses one
  today, but `handler_body_to_hnl` refuses them by name rather than
  guessing; when family synthesis (phase 3) mints `gather`/`fold`/
  `reduce` bodies, each needs either a `tensor.*` spelling or a
  recorded skip.
- **`lawdata.HANDLERS` kernel entries** — not Models at all (see
  above); they name primitives, they do not spell bodies.

## Tests

- `tests/test_modelproj.py` (31 tests): spec/body → `.hnl`
  conversion, literal/name-map edge cases, every refusal path,
  `hnl_program` marshaling, `verify_model` gate logic with an
  injected evaluator (shipped, eval failure, shape mismatch,
  numeric miss, undrawable domain), canonical `models.yaml`
  integration, verdict round-trip through `emit_all`,
  `check_vocab` drift on a hand-edited model row, and the
  unsupported-family collector path.
- `tests/test_vocab_model_parity.py` (2 e2e tests): real `hnl-run`
  parity — emitted `.hnl` evaluated by the actual Handl runtime
  matches the torch reference bit-for-bit on drawn fp64 sites.
- Scoped run: `276 passed` across the vocab/handl/discovery-pipeline
  selection; `modelproj` at 100% coverage, `vocabdir`'s new lines all
  covered (remaining misses are pre-existing edge branches —
  env-var/error paths — inside the discovery package's staged 99%
  ratchet, not phase-2 code).
- `scripts/emit_vocab_fragments.py --check`: vocabulary fresh, no
  drift; `vocabdir.check_vocab(vocab/)` clean.

## Gates

- `ruff check` / `ruff format --check` on `packages scripts examples`
  + the two new test files: clean.
- `ty check`: clean.  `vulture`: clean.  `lint-imports`: contracts
  kept.  `radon_ratchet`: ok (the attr-op emitter was split to stay
  under the complexity ceiling).

## Files changed

- `packages/mlopt-discovery/src/mlopt_discovery/modelproj.py` — new:
  projection + fp64 gate (~500 lines).
- `packages/mlopt-discovery/src/mlopt_discovery/vocabdir.py` —
  `models.yaml` as seventh canonical file; `Vocab.models` /
  `Vocab.model_verdicts`; `collect_models`; verdict merge/upsert;
  owner-drift coverage.
- `scripts/verify_vocab_models.py` — new: the verification driver
  (hnl-run eval, `_domain_draws`, verdict write).
- `tests/test_modelproj.py` — new: 31 unit/integration tests.
- `tests/test_vocab_model_parity.py` — new: 2 e2e parity tests.
- `vocab/models.yaml` — new canonical fragment file: 109 model rows
  + 109 verdicts.

## What phase 3 needs

Per the plan, phase 3 closes the loop the other direction — mint
*emits* fragments instead of Python registries growing by hand:

- `mint` writes op + denotation + handler-family + schema fragments
  (the `emit_prim` seam extended so a fresh mint lands `models.yaml`
  rows too — today `collect_models` picks it up on next emit, but the
  mint path should upsert directly).
- `zoo-prim` loop closure: a minted primitive discovered by the zoo
  re-enters discovery as vocabulary data.
- Depth-2 remint test: a `synop2_*` minted from `synop_*` parts must
  project to a `tensor.*` model and pass the same fp64 gate.
- Refused-spec analysis emitting handler-family fragments, and
  anti-unification emitting schema fragments — the two remaining
  hand-drawn surfaces.
- `unsqueeze` (or a documented respelling) if the five skipped
  denotations are to ship as Models.

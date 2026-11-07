# Fragment tower — phase 1 progress note

Plan: `project/plans/pl-0029-plan-0033-the-fragment-tower-mlopt-vocabularies-as-language-data.md`
(phase 1 — the `engine="handl"` seat)
Host ADR: `host/project/adrs/0157-adr-0157-mlopt-on-the-fragment-tower-domain-vocabularies-as-language-fragments.md`
Ledger: `T-017` (open — phases 2–5 remain)

## What phase 1 delivered

`mlopt_discovery.handl_engine` — the first-class differential backend.
The `mlopt-handl-peer` crate is no longer only a spike fixture: it is
the second engine every saturation can be checked against.

### `HandlPeer` — the peer client

One client for both peer surfaces (the resolution order
`tests/test_handl_peer.py` established):

1. the PyO3 extension `mlopt_handl_peer` (maturin build), preferred;
2. the CLI fallback `src/bin/rewrite.rs`, honoured via
   `MLOPT_HANDL_PEER_BIN` or `~/GitHub/handl-lang/host/target/debug/rewrite`.

`HandlPeer(fragments=...)` defaults to `vocabdir.default_vocab_dir()` —
the canonical `vocab/` projection, the same directory
`pipeline.propose` consumes (the phase-0 exit gate, now exercised
end-to-end).  `apply_rule(rule_id, nodes, at)` returns the parsed
result or `None` — a decline (load failure, unknown rule id, refused
native, nonzero exit, unparseable stdout) is `None`, never an
exception, so declines count separately from disagreements.
`transform_ids()` is extension-only (the CLI exposes rewrite only)
and raises `HandlPeerUnavailable` naming the gap.

Construction raises `HandlPeerUnavailable` — with the build command —
when neither surface exists or no fragment directory resolves: the
honest-skip pattern.  A differential engine with no differential
partner is not a differential engine.

### `HandlEngine` — the `engine="handl"` seat

An `EGraph` subclass (`engine_name = "handl"`), so it satisfies the
whole `mlopt_core.ports.Engine` port and
`mlopt_orchestrator.optimize.search(..., engine=...)` accepts it
directly — non-local passes and certificates keep working because it
*is* the reference engine underneath.

`run()` saturates normally, then `check_pending()` replays **every
rule application recorded since the last check** through
`transform.mlopt.<rule>` at the matched term's root:

- the fired substitution resolves to concrete terms through
  `_resolve_subst` + `_term_instantiate` — the same replay machinery
  certificates use;
- the instantiated lhs is flattened to the peer's arena format
  (`term_to_ast`), handl rewrites it, and the result is rendered back
  (`ast_result_term`) and compared structurally with the instantiated
  rhs (`render_term`);
- **a disagreement raises `HandlDisagreementError` before `run`
  returns** — the plan's "disagreement refuses loudly".  The exception
  carries the rule name plus both renderings.

Everything handl cannot express is *counted, not hidden*, in
`stats["handl_parity"]` (also `eg.parity_stats()`):

| bucket | meaning |
|---|---|
| `agree` / `declined` / `disagreed` | the three real outcomes (disagreed = distinct firings, with lhs/mlopt_rhs/handl_rhs detail) |
| `skipped` | leaf-carrying rhs (the `mlopt:const`/`var` bridge kinds are not comparable yet) **or** `derive` hooks that mint bindings the rule form cannot see; `check`/`cond`/`dspec` are pure vetoes mlopt already evaluated, so those laws stay checked |
| `unresolved` | applications with no `Rewrite` object or bindings that do not resolve to terms |
| `peer_calls` | actual peer invocations — identical `(rule, lhs)` firings memoize, so saturation's thousands of repeat applications cost one call each |
| `surface` / `tracked` | which peer surface ran; `tracked: false` honestly reports a `truncation_level=1` engine recorded nothing to check |

`resolve_engine("handl")` is the plan's `engine="handl"` spelling for
callers of `optimize.search` (whose `engine=` takes objects, not
strings — `"python"`/`None` resolve to the reference `EGraph`, any
other value passes through).  The literal string inside
`optimize.search` itself would need an eqsat-side registry — noted
below as a phase-2 seam.

### The `vocab/` projection fix (found by the e2e test)

`vocab/ast.yaml` declared an `AstShapeDef` for **every** op row —
including `arity: None` variadic ops (`linear`, `batch_norm`, …),
which got `children: []` (zero declared roles).  handl's transform
validator rejects `KeepPath`/`KindAt` indices into a zero-role shape,
so `language_from_dir(vocab/)` failed to load at all.  The spike
fixtures never hit this because they declare no op shapes.

`_projection_rows` now omits `AstShapeDef` rows for `arity: None`
ops — the honest projection (the op still rides in `ops.yaml`; the
canonical files are untouched).  `vocab/ast.yaml` regenerated;
`transform_ids(vocab/)` now returns all 143 law transforms and
`emit_vocab_fragments.py --check` stays fresh.

### The audit driver rides the same client

`bench/drivers/audit/handl_differential.py` now uses `HandlPeer` +
`term_to_ast`/`ast_result_term`/`render_term`/`is_structural_rhs`
instead of its own copies of the flatten/render/CLI-or-ext logic —
one peer client, one AST codec.  Measured output is unchanged:
24 laws compared, 24 agree, 0 disagree.

## Verified

- `pytest tests/test_handl_engine.py` — **20 passed** (incl. the e2e
  test against the real extension over `vocab/`)
- `pytest tests/test_handl_peer.py tests/test_vocabdir.py
  tests/test_discovery_pipeline.py tests/test_discovery_handl.py
  -m "not slow"` — 157 passed
- Real-peer smoke: `HandlEngine().run(ALL_RULES, …)` over six
  hand-built terms — **70 applications checked, 0 disagreements,
  0 declines** (plus guarded laws `transpose_noop`/`id_add`/`id_mul`/
  `double_neg` agreeing through `!Replace`)
- `bench/drivers/audit/handl_differential.py` — 24 agree / 0 disagree
- `scripts/emit_vocab_fragments.py --check` → `vocabulary fresh`
- `ruff check` / `ruff format --check`, `ty check`, `vulture`,
  `lint-imports`, `radon_ratchet` — clean

### The peer build on this box

`engine-transform` and `engine-transform-rewrite` were not checked
out locally (phase 0's honest-skip cause).  Both were cloned into
`/home/luis/GitHub/handl-lang/` and the crate built with maturin in
a shadow tree (`/tmp/handl-peer-shadow`) — building in place is
impossible because the uncommitted workspace
`/home/luis/GitHub/handl-lang/.cargo/config.toml` patches ~44
missing sibling checkouts and cargo validates every patch eagerly.
The shadow tree symlinks the five needed siblings at `/tmp/` and
reuses `~/GitHub/handl-lang/host/target` as `CARGO_TARGET_DIR`.
`mlopt_handl_peer` is now importable in `.venv`, so the e2e tests
run for real here; on a box without the build they skip with the
build instructions, exactly like `test_handl_peer.py`.

## What remains (phases 2–5)

- **Phase 2** — denotations and minted handlers as handl Models over
  `host-tensor`; fp64 parity gate vs `TorchSink`.
- **Phase 3** — the mint emits op + denotation + handler-family +
  schema fragments; the loop-closure depth-2 remint test.  The
  `emit_prim` seam from phase 0 is the landing strip.
- **Phase 4** — `Board` seat over fragment-application moves;
  `handl-egraph` (a crate that does not exist yet — the plan's
  saturation engine) behind the perf-parity gate; typed guards.
- **Phase 5 / saturation** — handl-native saturation; folds into the
  same arc.

### Phase-1 seams left open

- `optimize.search(engine="handl")` as a *literal string* needs an
  engine-name registry on the eqsat side (`_resolve_engine` lives in
  `mlopt_orchestrator`, a pinned `handl-lang/eqsat` dep — not editable
  from this repo).  `resolve_engine("handl")` / `engine=HandlEngine()`
  is the working spelling today.
- The overnight `--engine handl` arm from the plan text: overnight's
  e-graph use is anti-unification mining, not law saturation — there
  is no `optimize.search` call site to hang parity stats on yet.
  Wiring it means giving the mint/verify path an engine parameter;
  deferred to whoever needs the funnel-side numbers.
- Peer `apply_rule` reloads `language_from_dir` per call (Rust side);
  the Python memo bounds repeats but a long saturation still pays a
  language load per distinct firing.  Acceptable for a checker; a
  session-scoped `Language` handle is a peer-crate improvement for
  later phases.

T-017 stays open.

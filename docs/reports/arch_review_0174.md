# Architecture review — ADR-0174 compliance audit

Audited against `host/project/adrs/0174-adr-0174-the-end-state-architecture-one-term-many-interpretations-verified-delivery.md` (+0172, 0173). Read-only; no code changed. Import-linter contracts were verified by reading the contract definitions and grepping for violations rather than executing the linters.

## Verdict per invariant

### 1. Verify before ship — **PARTIAL**

The *product* path enforces the gate well; the *substrate* and *domain deliver-hooks* treat `VerifyReport.passed` as advisory data.

**Enforced:**
- `mlopt/src/mlopt/deliver.py:381-402` — a runner delivery with `verify=True` that fails verify **or produces no verify report** is declined; the verified eager lowering ships and `runner_name` stays `None` so the report can't imply a runner shipped.
- `mlopt/src/mlopt/deliver_game.py:475-489` — same mandatory-gate rule inside the game referee; `committed()` (lines 311-329) only ships a config that strictly beats the measured eager baseline, and `_evaluate` records declines with reasons.
- `mlopt/src/mlopt/cli.py:116-118, 153-158` — `optimize` exits 1 on `verified is False`; `export` hardcodes `verify=True` and refuses to emit on a failed gate; `--verify-run` (lines 202-305) is a real numeric gate through `hnl-run` vs the torch reference, declining loudly on every unsupported shape.
- `eqsat/.../compositional.py` / `taskgate.py:324-338` — `_block_decline` *raises* on per-block verify failure (the one place the substrate itself refuses).

**Not enforced (the gap):**
- `eqsat/packages/mlopt-orchestrator/src/mlopt_orchestrator/optimize.py:1136-1200` — `lower(verify=True)` computes `verified` and returns the module **regardless of `passed`**. `LowerResult` always carries the runnable; the verdict is a field, not a gate. Every downstream consumer must remember to check `.passed`. Nothing in the substrate prevents shipping an unverified artifact — the honesty contract lives entirely in callers.
- Domain deliver hooks report but don't gate: `graphopt/.../play/torch_arm.py:333-345` and `kernelopt/src/mlopt_kernel/play.py:443-452` both return `"delivered": True` with `"verified": rep.passed` even when verify fails. These are measurement/report arms (`--deliver`), not shipping paths, so it's arguably honest — but the ADR's wording ("`VerifyReport.passed` gates … every delivery candidate") is stronger than the code. A failed verify still produces a `delivered` artifact object.

**Recommendation:** either downgrade the ADR's claim to "verify verdict is mandatory data; enforcement is per-consumer" or add a substrate-level `require_verified` option / make `lower` raise on `passed=False` unless explicitly opted out.

### 2. Declines are named — **PASS (with one caveat)**

Every decline path sampled names its reason: `deliver.py` notes for every skip (`region_fuse:`/`cuda_graph:`/`deliver:` prefixed notes, lines 200-233, 350, 363-366, 392-394, 404); `deliver_game.py` records `{"reason": …}` per declined config; `cli.py` prints `declined — …` for every verify-run failure mode; `emit.py` raises `UnsupportedOpError` naming the op; `optimize.py:178-183` names the skipped game path; `optimize.py:212-217` names game-path fallback. Bare `except Exception` blocks all append notes rather than swallowing.

Caveat: `mlopt/src/mlopt/deliver.py:98,117,165,209` use `except Exception` for capability probes — each is followed by a named note or `None` return, so acceptable. `_shape_facts` (`deliver_game.py:440-447`) uses `contextlib.suppress(Exception)` for telemetry — shape facts silently zeroing is a minor honesty soft spot (a broken shape read yields `n_enodes=0` with no note).

### 3. Measured or it didn't happen — **PASS**

Delivery speedups are measured min/median over `bench_iters` with CUDA sync (`deliver.py:249-293`); the game commits only on measured medians (`deliver_game.py:311-329`); `speedup` is computed from measured values only (`optimize.py:249-250`). The docstring at `deliver_game.py:14-23` honestly explains why `gamekit.scoring`'s absolute `ScoreGate` doesn't fit and documents the custom referee. benchkit's `Runner` uses `torch.utils.benchmark` (`gamekit/packages/benchkit/src/benchkit/runner.py:17-19`).

### 4. One codec — **FAIL (current state; phase-consistent but worth stating bluntly)**

The end state is not here, and the *shipped product path depends on the duplicate*:

- `mlopt/src/mlopt/emit.py:1220` — `_OPS`, a ~199-entry hand-written `mlopt_core.ir.Op → tensor.*` builder table. It does **not** read `host_surface.json` and shares no code with `hnl_surface`. `mlopt export` (`cli.py:169`) emits through this table.
- `graphopt/packages/mlopt-discovery/src/mlopt_discovery/hnl_surface.py:140` — `_SPECS`, a *different* 12-op term→`.hnl` table with different coverage and different helpers.
- `graphopt/packages/kit-host-tensor/src/kit_host_tensor/bindings.py` — `HOST_BINDINGS`, ~137-op IR→host positional-arg table (the Sink lowering codec).
- `host_surface.json` (237 generated signatures) — drift-guarded metadata (`tests/test_host_surface.py`), consumed by `host_ops()`/`bindings` validation but **not** by either emitter.
- `mlopt_discovery.hnl_import` — the `.hnl`→term importer covers *more* ops than `_SPECS` emits (per `tests/test_hnl_import.py:146`), so the codec is asymmetric today.
- `mlopt_discovery.modelproj` — a third `.hnl` emitter (spec-side).
- `mlopt_torch.torch_bindings` — the IR→torch arg table (same role as `HOST_BINDINGS` for the torch sink).

ADR-0172's "Why" says the split "produced four divergent op tables"; ADR-0174 invariant 4 says "No fourth op table." Today there are **at least five** Python-side term↔surface tables plus the generated JSON. This is *phase-consistent* (P3 migration is pending, and ADR-0172 §4 explicitly keeps `mlopt_core.ir` canonical meanwhile) — but the audit should record that `mlopt export`, the product's flagship deliverable, is built on the table designated for retirement, and `emit._OPS` has **no drift guard** against `host_surface.json` (grep: no reference; `tests/test_emit.py` never mentions it). The 199-op table is validated only at authoring time against `hnl-run`.

Also note: `emit.py:125-128` declines non-finite literals outright — honest, but it means the product's export path cannot express what ADR-0173's `__nonfinite__` codec tag is meant to carry. Consistent with P0.5 being unlanded; flag so nobody reads `emit.py` as codec-conformant.

### 5. Substrate layering — **PASS (one soft spot)**

- `eqsat/pyproject.toml:85-118` — import-linter contracts: `mlopt_core` forbidden from `mlopt_orchestrator`/`mlopt_native`/`torch`/`numpy`; orchestrator forbidden from all backends; layers contract. Grep confirms zero `import torch|numpy` under `eqsat/packages`.
- `gamekit/pyproject.toml:61-84` — kernel modules (`engine`, `scoring`, `calibrate`, `advantage`, `gamedata`, `players._base`, `players.registry`) forbidden from all substrate/backend/domain packages + torch + numpy.
- Domain packages clean: `mlopt-torch`, `mlopt-carriers`, `mlopt-cuda`, `mlopt-onnx` contain only *comments* referencing `mlopt_discovery`/`gamekit` — no imports. `kit-host-tensor` imports only `mlopt_core` (the Sink port — correct direction).
- `mlopt` is composition: `src/mlopt` has zero module-level torch/numpy imports (all lazy); `optimize.py` delegates to `Optimizer`/`TorchBackend`; `crates/hnl-ext-mlopt` is a pure argv-forwarding shim (`lib.rs:37-78`).

Soft spot: `gamekit/packages/gamekit/src/gamekit/players/neural/__init__.py:76-78` lazily imports `torch` + `mlopt_core.features`/`trajectories` inside a factory — the import-linter contract *deliberately excludes* player folders for this reason, but ADR-0174's "gamekit ⊥ eqsat (stdlib-only)" is only true of the kernel modules, not the package. Also `benchkit` (same repo) hard-depends on `torch` and `mlopt-core` (`packages/benchkit/pyproject.toml:14-17`) — reasonable for a "measured ledger" but it means the gamekit *repo* is not stdlib-only; only the `gamekit` package's kernel is.

### 6. Ops are data — **PARTIAL (phase-consistent)**

Tensor ops are **not** yet a `Language` fragment — P1 is pending P0.5, and P0.5 (ADR-0173) is still `status: proposed` with nothing implemented: `engine/src/term.rs:45-70` `Term::Op` has **no `attrs` field** (fields: `dim`, `name`, `args`, `source`, `lit` only). Ops currently live as: `host-tensor/src/index.hnl` handwritten decls + generated `aten_generated.rs` (the host surface — good, ops-as-data on the host side), plus the five Python tables above (the not-yet-collapsed duplication). Nothing is *privileged* — host-tensor's own header says "nothing in the eval engine or language data knows what a tensor is" (`index.hnl:11-12`) — but the Language-fragment invariant is aspirational today.

## Phase honesty

- P0 codec "landed": mostly true — `hnl_import`/`hnl_surface` exist, `host_surface.json` is drift-guarded, `.hnl` Source works (`mlopt/adapters/hnl.py`). But "one codec" is not achieved (invariant 4 FAIL above) and `mlopt export` rides the duplicate table — the phase deliverable is real but narrower than the invariant implies.
- P0.5 kernel: correctly *not* claimed — ADR-0173 still `proposed`, `Term::Op.attrs` absent. Honest.
- Native path "census done": `graphopt/docs/reports/kernel_ir_coverage.md` says **0.12%** (5/4,289 nodes) — matches the ADR's "current honest number: 0.12%" exactly. Honest.
- `mlopt play.py:33-36` and AGENTS.md honestly state the `linear*` delivery players are **cold** (`DELIVERY_PLAYER_WEIGHTS = {}`); `deliver_train.py` is the trainer, not a shipped prior. Honest — though it means the "game is the decision procedure" invariant currently runs on `playbook`/`random` in production.

## Non-goal violations

- **No `Ref`/let/vars in Term kernel**: PASS — `engine/src/term.rs` enum variants are `Arr/Const/Op/Seq/First/Choice/Loop/App/Iter/Child/Handle/Rewrite/Src/Tgt/RecordProj/RecordWith/Tag/Cobind/Opaque` — no Ref/let/vars. The dag-sharing design (env leaves + engine hash-consing + Opaque record) is unimplemented but also un-violated.
- **No privileged VM traits**: PASS — tensor ops are host functions resolved through `OpRegistry`; engine stays domain-free.
- **Torch as foundation in substrate**: PASS — eqsat/gamekit-kernel clean; the torch denotation lives in `lib-kernel-ir` (`src/lib_kernel_ir/term.py:32`, `ops.py:30` `import torch`) which is *by design* the kernel-domain IR, not substrate.
- **Fourth IR**: `mlopt_core.ir` is the acknowledged temporary search IR. But the ADR's non-goal "No parallel IR proliferation" deserves a sharper note: there are currently **four** term representations in the org — `engine::Term` (arrow calculus), `egraph::Term` (`src/term.rs:25-46`, op/args/attrs triples), `mlopt_core.ir` (Op/Var/Param/Const), and `lib_kernel_ir.Term`. The ADR mentions three of these implicitly but **never names `egraph::Term`** — and that one already has the `(op, children, attrs)` shape ADR-0173 wants to add to the kernel Term. See drift.

## Per-repo layer conformance

| repo | claimed layer | conforms? | notes |
|---|---|---|---|
| `mlopt` | product — thin composition | MOSTLY | composition + report + verify gates are clean; but `emit.py` (1,926 lines) is a codec bridge duplicating `hnl_surface`, and `deliver_game.py`/`deliver_train.py`/`play.py` are real game wiring, not "thin". AGENTS.md's "zero optimization machinery" holds; "codec" is grayer. |
| `eqsat` | substrate (mlopt-core, orchestrator, native) | PASS | torch/numpy-free; import-linter contracts present; `lower` verify is advisory (invariant 1 caveat). |
| `graphopt` | domain packages + discovery shims + bench | PASS | domain packages don't import game/discovery; `mlopt_discovery` is now mostly re-export shims onto discoverykit. |
| `kernelopt` | codegen domain | PASS | `mlopt_kernel.play` is the explicit plugin seam; verify-first sink; lazy torch in deliver hook only. |
| `gamekit` (+`benchkit`) | game layer / measured ledger | MOSTLY | kernel stdlib-only per contract; `players/neural` lazily binds `mlopt_core`; `benchkit` is torch+mlopt-core dependent (documented as torch-shaped by design). |
| `discoverykit` | domain-free mining core | PASS | lazy `mlopt_discovery.*` seam preserved; no static import-linter contract but a dynamic poison-test (`tests/test_smoke.py`) — weaker enforcement, honestly documented. |
| `kernel` | Term/Value/Language kernel | PASS | no attrs field yet (P0.5 pending — consistent). |
| `engine`, `engine-exec-*`, `engine-typecheck-*`, `engine-transform*`, `engine-lex*`, `engine-parse*` | interpretations | PASS | tree-walk/bytecode/cranelift/NBE/rewrite — matches map. |
| `egraph` | (unlisted in ADR layer map) | DRIFT | Rust e-graph with its **own `Term` type** that already carries `attrs` — see below. |
| `host`, `host-cli`, `host-lsp`, `host-tensor` | handl platform / tensor host | PASS | host-tensor exposes `tensor.*` as data (`index.hnl` + `OpRegistry`); `hnl-ext-mlopt` lives in mlopt repo and forwards to `python -m mlopt` — composition-root shape holds. |
| `lib-kernel-ir` | native-path IR | PASS w/ note | torch-denoting by design; the "torch-free" non-goal is about *claims*, and none is made. |
| `compile-triton` (+`compile-js`/`llvm`/`dot`/`handl`/`term`/`sexp`) | native emit interpretations | PASS | `compile-triton` consumes lib-kernel-ir JSON → Triton source; serde/serde_json only. |
| `docs` | docs | n/a | — |

## Drift register (built, not on the map)

1. **`egraph` crate's own `Term`** — `egraph/src/term.rs` defines `Term::{Var, Leaf, Op{op,args,attrs}}` with `TermAttr::{Lit,Meta}`, *already first-class attrs* — the exact `(op, children, attrs)` triple ADR-0173 proposes adding to the kernel `Term`. The ADR never mentions this type. If engine-eqsat (P2) lands on `handl-egraph`, the "one Term" story gets a fifth representation unless the kernel Term and egraph Term are explicitly related (they are bridged: `egraph/src/term_bridge.rs`, and `handl-egraph` deps on `hnl_engine`/`handl_language`/`handl_value` per the build fingerprint). The ADR should say which Term is *the* Term.
2. **`mlopt.deliver_game` + `deliver_train` + `play.py` + `weights.py`** — a whole delivery-game stack (board, players, feature schema, prior trainer, baked weights) that post-dates the map's "mlopt = adapters·verify·deliver·report" sketch. It's *on-thesis* (measured delivery, named mechanism) but it's the largest un-mapped component in the product repo.
3. **`mlopt.adapters.hnl` Source** — `.hnl` as a first-class model source (`hnl_to_term` lift) landed; the phase map's P0 row says "mlopt Source in flight" — it's landed now, minor status drift in the ADR's own table.
4. **`verify_oracle` / `task=`/`task_tol=` contract** (`taskgate.py`) — a whole second verify dimension (task-metric distance gating) the ADR's "VerifyReport.passed" invariant doesn't mention. It's *more* honesty machinery, but the invariant as written ("`VerifyReport.passed` gates") is incomplete — the operative verdict can come from a task metric, a propagated bound, or pointwise tolerance (`accepted_by` labels which).
5. **`benchkit` inside gamekit repo** — torch-dependent benchmark harness co-located with the stdlib-only game kernel; the repo boundary doesn't match the layer boundary (kernel ⊥ benchkit deps differ).
6. **`discoverykit` has no static import-linter contract** — enforcement is a dynamic poison-test only; a regression could pass lint while breaking the seam.

## Top risks, ranked

1. **Codec fragmentation is the live risk, not a future one.** `mlopt export` — the product deliverable — ships through `emit._OPS`, a 199-op hand table with no drift guard against `host_surface.json`, while the canonical codec (`hnl_surface`, 12 ops) and importer cover different sets. Until P3, every new tensor op needs *two* (sometimes three) table updates, and nothing fails loudly when they diverge except per-builder `hnl-run` spot checks.
2. **Verify gate is caller-enforced, not substrate-enforced.** `Optimizer.lower` returns a runnable with `verified.passed=False` and no refusal; domain deliver hooks return `delivered: True` on failed verify. One careless consumer (a bench driver, a future sink) silently ships unverified output — exactly the failure class the invariant exists to prevent. The product path is careful; the *contract* is not.
3. **`egraph::Term` vs kernel `Term` convergence is unplanned in the ADR.** P2 "engine-eqsat" could land on either; the ADR's silence means two "Term"s with different attrs stories could both survive, deepening the proliferation the ADR warns about.
4. **`gamekit.players.neural` + `benchkit` blur the "stdlib-only gamekit" claim** — small today, but the import-linter contract's folder-exclusion means new player folders can accumulate lazy substrate binds without tripping any gate.
5. **ADR-0173 is still `proposed` while consumers already depend on its absence** — the `__mlopt_attrs__`/`lit` hack (`term_ir_spike.py:79`) and `emit.py`'s attr-reading conventions (`_attr` canonical→`argN` fallback, `emit.py:205-210`) are de-facto encoding decisions made *before* the attrs-slot ADR lands. When P0.5 lands, these spellings must migrate; nothing tracks that debt.

## What the ADR got wrong or missed

- **"One codec" is stated as an invariant but the phase map doesn't sequence the *product's* dependency on the duplicate.** `mlopt export` can't wait for P3 — either the ADR should bless `emit._OPS` as the interim canonical emitter (and retire `hnl_surface._SPECS` instead, since it's smaller and measurement-only), or it should fund the consolidation earlier. Right now the *smallest* table is called canonical and the *largest* is what users actually get.
- **`VerifyReport.passed` is not the only verdict.** `taskgate.py` composes pointwise/bound/task-metric gates and labels `accepted_by`; the invariant should say "the operative verify verdict" not "`VerifyReport.passed`" — as written, a task-gated delivery technically violates the letter of the invariant.
- **The egraph crate is missing from the layer map and the Term census.** It's the most likely P2 host and already has attrs — the ADR should decide now whether engine-eqsat *is* `handl-egraph` (making `engine::Term` the odd one out) or a new interpretation over kernel Terms (making `egraph::Term` the bridge's problem).
- **"gamekit ⊥ eqsat (stdlib-only)" is only true of a module list, not the package.** The contract enumerates kernel modules and excludes `players/*` folders; `benchkit` isn't covered at all. The invariant should name the boundary precisely or the next audit will "discover" the same soft spot.
- **Phase table is already stale**: P0's "mlopt Source in flight" has landed (`adapters/hnl.py` + `test_hnl_source.py`); the `linear*` delivery players being cold isn't reflected anywhere in the phase map — "the game is the decision procedure" currently means "the playbook is, until someone runs `train_delivery_prior.py` and pastes the literal."

## Actions the reviewer could not perform (for the parent agent)

- Write this report file — done by parent.
- Run `lint-imports` / pytest — contracts verified statically; a live run would confirm no indirect violations.
- Verify `host_surface.json` freshness — `tests/test_host_surface.py` is the drift guard.
- Exhaustive per-repo AGENTS-vs-layout audit of the ~100 lib/kit repos — out of scope.

# GOALS — the standing goal ledger

**Why this file exists**: so the work is *auditable* rather than
asserted.  Every goal carries a pass/fail criterion and the artifact
that proves it.  **If a claim is not in this ledger with an artifact,
treat it as not done.**  Commit messages name the goal they serve
(`G0-Q0`, `G1-ports`, …), so a night's work can be audited from
`git log` alone.

## The enduring goal (G\*)

A **semantic categorical language + learned search that discovers new
algorithms and semantics nobody wrote** — where discoveries are
*certified* (an equation with a replayable derivation), and where the
object language itself grows.

Not a night's deliverable.  Tonight only *unblocks* it.  **No artifact
below may be presented as G\* being met.**

## G0 — the handl ↔ catopt boundary spike (plan 0029)

Answer four questions with measurements and produce a decision.

| q | question | pass criterion | artifact |
|---|---|---|---|
| **Q0** | can catopt's **language** be expressed as handl data? | one catopt law encoded as a `TransformDef` + one shape as `AstShapeDef`; merges; handl's `RewriteEngine` fires it; result matches catopt's own `apply_rule` on the same input | retro + the fragment files |
| **Q1** | can a **tensor value** cross the boundary? | one catopt op registered as a handl **host function**; a small tensor term evaluates through handl; dtype/shape/device intact and fp64 agreement with catopt's `eval_term` | retro with the numbers |
| **Q3** | which integration direction? | one of (A) catopt embeds handl (PyO3 peer) or (B) handl drives catopt is buildable in catopt's CI **and** has a named **seam test** | retro + the seam-test sketch |
| **Q2** | what does the boundary cost? | host-fn dispatch does **not** dominate **rewrite throughput** (same order of magnitude vs in-process); exec timing is informational | retro with both timings |

**Decision the spike must produce**: `go (A)` · `go (B)` · `split`
(handl owns language+rewrite+typecheck; catopt owns the tensor exec
interpretation) · `no-go`.  A failed question with a **named gap** is a
successful result; a vague failure is not.

## G1 — tonight's supporting work

| goal | pass criterion | artifact |
|---|---|---|
| land the gate-speedups agent's work | `scripts/gates.sh` runs the suite once under coverage; `NP` default justified by a measured serial-vs-`-n 2` table (or set to 1); `slow` markers placed with measured durations; AGENTS.md's stale "~7.5 min" replaced | commits + the table |
| **ADR 0009 stage 1** — six core ports (`Calculus`, `Engine`, `Proposer`, `Environment`+`Reward`, `Prover`/`Bound`) | each lands **with a conformance test an existing implementation already satisfies**; no behaviour change; `ty`/`ruff`/`lint-imports` green | commits + tests |
| **ADR 0011** — canonical representation | the decision is written with a falsification path (op-graph IR vs arrow calculus) | the ADR |
| final gates on the merged tree | the full pass green **through the new `scripts/gates.sh`** | the run's output in the retro |

## G2 — the architectural chain

`0010 (handl is the kernel)` → **boundary spike (G0)** → `0011
(canonical representation)` → `0009 stage 4 (the calculus package)`.
Each is a prerequisite of the next; none is skipped.  **Proven by**: the
spike's decision, plus docs updated in **both** repositories.

## Invariants (violating one is a failure regardless of results)

1. **No force-push**; **no remote-branch deletion** without explicit
   consent.
2. **No `rm -rf`** outside regenerable caches (`~/.cache`, venvs for
   dead worktrees).
3. **No widening of any ratchet** (coverage floors, `ty` excludes, ruff
   ignores, radon baseline, the procedural-hook count).
4. **One heavy suite at a time**; check `free -m` before each; abort
   below 2.5 GB available.
5. **No invented numbers.**  A number that cannot be sourced is marked
   `UNSOURCED` or dropped with the reason.
6. **Nothing irreversible without asking** (destructive ops, remote
   state, other repos' history).
7. **Leave the tree committed and pushed**, with docs/md updated in both
   repos — so the next session starts from a clean, documented state.

## If something fails

The ledger shows **which** goal is unmet and **why** — no silent drops.
Salvage beats abandon: uncommitted work in a dying worktree gets
committed to its branch before anything else.

---

## Night status — 2026-10-14/15

### G0 — the handl ↔ catopt boundary spike: **DONE**

| q | result | artifact |
|---|---|---|
| **Q0** | **PASS, unqualified** — catopt's vocabulary assembles as handl data, and after closing two gaps (the leaf step; the `pass:node_kind:priority` rule collision) the **whole bank loads: 83 laws** in `lang.transforms`, with the three minted leaf kinds declared | catopt `8de8cb5` + `8d5b47d` + `5ca8aef`; handl `b20b978a` + `9372bd32`; retro `retros/handl-q0-language-as-data.md` |
| **Q1a** | **yes, by reference** — `TaggedValue::Resource(Rc<ForeignResource>)`: an opaque pointer + finalizer, no copy | retro `retros/handl-q1-tensor-crossing.md` |
| **Q1b** | **the seam is JSON-valued** — `Function::call(&[Value]) -> Result<Value>` with `Value = serde_json::Value`; a `Resource`'s JSON encoding is a *debug* process-local pointer (`{"__resource__": {"ptr": n}}`), dangling across processes | same retro |
| **Q3** | **(A) catopt embeds handl as an optional peer** (the `catopt-native` pattern); (B) sequenced, not rejected | retro `retros/handl-q3-integration-direction.md` |
| **Q2** | **not applicable as posed** — v1 has no host-fn dispatch; the measured cost is the fragment load, which is instant | retro `retros/handl-q2-boundary-cost.md` |

**Spike decision: `go (A)` — catopt embeds handl as an optional peer,
language-only in v1** (fragments in, terms out; no tensors across).
That is what the measurements support, and it is what ADR 0011 already
assumes.

**And the gate is met**: plan 0030's step 1 — *does a rewrite fire?* — is
**measured yes** (handl `82194954`): a catopt law fires through handl's
`RewriteEngine` and produces exactly what catopt's own engine produces.
So the peer is worth building, and "handl is the kernel" now rests on an
executed rewrite rather than a load.

Two representational requirements were found and fixed on the way, both
in catopt: `when` must fold into `And([...])` (handl's field is ONE
condition), and enum variants must be spelled as **YAML tags** under
`serde_yaml`, not JSON maps.  One residue is named: the leaf step
(`{"catopt:leaf": …}`) has no `RewriteStep` counterpart.

### G1 — supporting work

| item | state |
|---|---|
| `slow` markers | **done** — placed from measured durations; the heaviest slice goes 370.8 s → 42.6 s under `-m "not slow"` (`7463327`) |
| `scripts/gates.sh` | **landed** (`1bbe13d`); the **serial** default (`NP=1`, `ed86dd4`) is now *confirmed by measurement* — the `-n 2` arm did not finish a 30-min box (2.7× further, but no completed win) |
| **ADR 0011** (canonical representation) | **done** (`78ce874`) — the calculus for laws, the op-graph for execution |
| ADR 0009 corrected | **done** (`0446e73`) — most of the "ports to add" already exist (`Engine` in core; `Environment`/`Reward` as `Board`/`Report`; `Calculus` is data, not a port) |
| docs in both repos | **done** — handl `337b2c83` (`docs/explanation/kits-and-manifests.md`: the per-file data path + the catopt example); catopt `b9d0e7c` (`docs/mechanism.md`: the kernel, and the platform next door) |
| **full-suite / NP measurement** | **done** — both arms run to a 30-min box on an **uncontended** 11 GB / 16-core host (load ~1.6), whole-tree RSS sampled from `/proc`: **neither finished.**  Serial reached **36%** (2,304/6,372 tests; peak **10,320 MB** tree-summed); `-n 2` + `TORCHINDUCTOR_COMPILE_THREADS=2` reached **98%** (6,264/6,372; peak **8,814 MB**).  So `-n 2` is **2.72× further in the same wall time** — but no *completed* wall-time win, so `gates.sh` stays **serial** (`NP=1`); the table is in `AGENTS.md`.  The wall is dominated by **unmarked** tests, not the six `slow` ones: the discovery "held-out-claim" fixtures (module-scoped `measured` in `test_discovery_open_{learned,prior}` / `test_discovery_meta_prior` / `test_discovery_open_game`, ~15 min each) and the e-graph saturation tests (`test_square_rsqrt_laws::test_every_miss_spelling_reaches_rms_norm`, `max_nodes=60_000`).  The RSS column over-counts shared pages (the serial peak is a ~14-process fan-out each `mmap`ing the 419 MB `stories110M.bin`); real headroom stayed ≥ 4.7 GB.  (My first diagnosis was wrong: two runs "died at 36%" — they were alive; `-q` progress is block-buffered and `ps \| grep` matched the wrapper.  Corrected `5077e00`.) |

### G2 — the chain

`0010 (handl is the kernel)` → **spike (G0, done)** → `0011 (canonical
representation, written)` → `0009 stage 4 (the calculus package)`, now
**unblocked in principle**: the decision is `go (A)`, and the remaining
gate for stage 4 is ADR 0008's gate 3 (does search *reach* `loop`-headed
forms), which is a catopt-side measurement, not a boundary one.  **Gate 3
is now PASSED** (`lab/bridge/form_propose.py`, retro
`project/retros/gate3-form-level-proposer.md`): the e-graph-oracle
anti-unifier synthesises one step schema, emits `unroll(step,n)` per
state chain, `PROPOSE_LOOP` folds both to `Loop`, and the whole morphism
expands back into the winner's own root e-class — `loop` is reached on
the real extracted winner.  One named residual: the two chunk-width
families fold separately across `block3->block4` — re-measured, the
boundary states differ by γ^12 in *value* (a normalisation-convention
jump, not a reach gap), so no sound e-graph merge closes it; the
single-loop-with-reindexed-wires is numerically exact but its
boundary wires are never minted by the current bank (measured:
`boundary-xi-nonresident:block4`, `boundary-renorm-unbound:block3->
block4`).  **The residual is now CLOSED** — the state-reindex law
family (`lab/bridge/state_reindex.py`: the bare-`matmul` contraction
bisection, signed-offset `decay_vec` block-splitting, and the scalar
slide/merge laws) mints the steering wires and derives the steered
state into `M_4`'s e-class; a single `Loop(step,6)` lands in the
winner's root class with `whole_equal: True`
(`test_state_reindex_closes_boundary`).

### G\* — the enduring goal

**Untouched, as stated.**  Nothing above is evidence that the machine
discovers new algorithms; the spike only made the substrate real.

### Invariants — held, with one honest exception

No force-push, no remote-branch deletion, no `rm -rf` outside
regenerable caches, no ratchet widened, no invented numbers, nothing
irreversible without asking.  **Exception:** for a period the `-n 2`
slice arm and a full-suite run were both live (contention, and it
distorted the numbers); the duplicate was killed when noticed, and the
affected figures were discarded rather than reported.

### Late additions (same night)

| item | result |
|---|---|
| **a rewrite fires** (plan 0030 step 1 — the gate on the peer) | **PASS**: `assoc_add` executes through handl's `RewriteEngine` (one rewrite, result `add(add(a,b),c)`), and catopt's own `EGraph.apply_rule` produces the same member on the same term.  The two engines agree.  handl `82194954` |
| **the leaf + collision gaps** | **closed** — a leaf maps to `!Fresh` / `!Create` of a declared kind, and rules get distinct priorities per `(pass, node_kind)`; the **whole bank loads (83 laws)**.  catopt `5ca8aef`, handl `9372bd32` |
| **plan 0030** | written (`1382f19`); step 1 **DONE**, so the optional peer (step 2) is justified rather than speculative |
| **ADR 0008 gate 3** | given a **protocol** — one command, one criterion, with a negative result as valuable as a positive one (`337a839`).  Not run: the box was busy with a full suite |
| **coverage ratchet** | verified for the changed module: `catopt_discovery/handl.py` is **100%** (117 stmts, 50 branches, 0 missed) — the floor is safe |
| **docs** | both repos updated with the rewrite result (catopt `docs/mechanism.md`; handl `docs/explanation/kits-and-manifests.md`) |

| **the differential rewrite check** | **new asset**: catopt's engine vs handl's over the whole bank — **19 compared, 19 agree, 0 disagree** (83 accounted for; the rest reported with reasons).  Two findings: root-eliminating laws cannot be expressed (handl's `Declarative` always creates a node), and 19 guards are shape-sensitive enough to raise on a synthesized term.  `lab/audit/handl_differential.py` + a pinning test (`project/retros/handl-differential-rewrite-check.md`) |

**One caveat worth carrying forward**: the `priority` field is the first
place handl's schema demanded something catopt does not have (an order),
and it is currently supplied as a *load-time disambiguator* that means
nothing semantically.  Plan 0030 step 4 exists so it does not ossify
into an implied meaning.

## Session — 2026-10-17

### Derive-DSL root ops (plan 0025 §D3, partial)

The weight-space lane's measured refusal had two named blockers; the
op-set one is now **closed**:

- `sqrt`/`rsqrt` landed in the derive DSL
  (`catopt_core/laws/cond.py` `_DUNOPS`, strict: negative/non-positive
  declines) and in the synthesizer's search bank
  (`dspec_synth` `UNOPS` += `sqrt`/`rsqrt`, `BINOPS` += `fdiv`).
- Probe delta (`lab/probes/weightspace_fold_probe.py`, new gate 2c):
  gate 3 flips — the DSL spells `γ·rsqrt(σ²+ε)` — and the synth now
  **mints the scale subexpr** at depth 3 (`fdiv(γ, sqrt(add(ε,var)))`,
  exact on all 6 measured channels).  Retro: `retros/derive-root-ops.md`.
- **Still open (named)**: (a) the derive value domain is `int|float` —
  gate 4 unchanged, tensor-valued derives are the real gate on conv+BN;
  (b) the full-fold expr needs enumeration depth ≥4 over the 7-attr
  bank (streamed sweep is budget-bound); (c) content gaps — no law LHS
  names `batch_norm`, and `batch_norm` is absent from `_OP_FLOPS`.

### Handler grammar — `postnorm` (grammar-gaps residue, closed)

The `norm` template covered only a *bare* reduction operand; the
corpus's `log_softmax` (2 sites, `sub(X, log(sum(exp X)))`) needs the
reduction result wrapped in a unary.  `handler_synth` gains the
seed-only `postnorm` family (`op(lhs, post(red(rhs)))`,
`POST_UNARY = ("log",)`, grammar version → `handler-grammar-v2`) —
the log-softmax spec mints + verifies fp64, and agrees with
`F.log_softmax` to 1e-12.  Retro: `retros/handler-grammar-postnorm.md`.
The stabilized-scan residue stays open (non-affine two-state
recurrence — a semantics extension, not vocabulary).

## Session — 2026-10-18

### Tensor-valued derives — ADR 0012 (plan 0025 §D3, the last capability blocker)

The weight-space lane's final *capability* refusal is **closed**: the
derive value domain was `int | float`, and a per-channel tensor `scale`
raised `TypeError: unhashable` in the synthesizer (weightspace-fold
gate 4).  ADR `0012-tensor-derives.md` lands the seam:

- **`$leaf:` channel in `dspec`** — a spec key mints a *term* binding:
  `Const` for scalars, `Param` through the pushed `mint` for tensors.
  `apply_rule` interns non-`$attr:` derive values via `add_term`;
  term-level instantiate sites needed no change.
- **Pushed tensor algebra** — `register_tensor_algebra(source=, mint=,
  ops=, key=)`: core stays torch/numpy-free (sink direction, the
  `register_value_shape` precedent); `("tval", T)` reads a bound
  leaf's value, `("tapp", name, e…)` applies registered ops, and
  `sqrt`/`rsqrt` dispatch into `ops` on non-scalar operands.
- **`batch_norm` priced** in `_OP_FLOPS` (gate 6; `4·n_out`).
- Probe re-run: `scale_mints_as_leaf: True`, `tensor_synth_fit:
  {'Wn': mul(attr g, tval w)}`, `tensor_sample_error: None`.
  Retro: `retros/tensor-derives.md`.
- **Both named residuals closed (same day)**: the `batch_norm` LHS law
  landed as `BATCH_NORM_RULES` — three eval-mode expansion variants
  minting `γ·((x−μ)·rsqrt(σ²+ε)) + β` as data (cond + dspec; the
  per-channel broadcast `reshape` target derived from `x`'s rank,
  `Const(eps)` minted through the scalar `$leaf:` channel) — fp64-exact
  vs `F.batch_norm` (7.3e-13), held out of `DEFAULT` pending the
  admission floor; and the **`$leaf:` auto-compose seam** landed — the
  `@i:` placeholder machinery now covers *term* metavars, so a
  `$leaf:` rule composes in `synthesize_rules` (measured: refusal at
  `_synthesizable` → emitted composite firing `Const(2.0)` through
  `"@1:eps"`).  Retro: `retros/leaf-compose-seam.md`; addendum in
  ADR 0012.
- **Still open (named)**: the conv+BN *weight fold* law itself
  (`W'`/`b'` — now reachable: stats are terms, `$leaf:` parents
  compose), synth depth ≥4 over the 7-attr bank, `BATCH_NORM_RULES`
  `DEFAULT` admission (guarded sweep).

### conv+BN fold synthesis attempt — measured refusal named (2026-10-18)

`lab/probes/conv_bn_fold_synth_probe.py` runs both entries on a real
`batch_norm(conv2d(x,W,b), γ,β,μ,σ²)` export seed:

- `synthesize_rules` emits the fold's *LHS* (`batch_norm(conv2d(…))` —
  6 composites under ALL+BN) but **never the RHS**: `conv2d` occurs in
  no rule's lhs or rhs across the 86-rule bank, and a composite's rhs
  head only comes from a parent's rhs.  Named refusal: **missing
  anchor** — one `conv2d`-headed law (a `conv_channel_scale` absorb,
  the conv twin of opt-in `linear_channel_scale`) or an anchored path
  (antiunify over a measured seed/folded pair, the ADR-0008 bridge).
- `dspec_synth` mints the fold's derives **in stages**: `scale` and
  `b'` at depth 3, `W'` at depth 2 given `scale`.  The one-shot
  tensor-tier `W'` refuses on depth ≥4 + broadcasting (needs a pushed
  channel-broadcast `mul` or an enumerated `tapp`/reshape).
- Verifier-env artifact found: `numeric_check` drops the BN composite —
  `_tensor_env` gives `rv<0` → `rsqrt` NaN on both sides.
- Lab bug fixed: `dspec_audit.bounded_synth` crashed `TypeError:
  unhashable ndarray` on tensor samples — ported the `_vkey` boxing.
  Retro: `retros/conv-bn-fold-synth.md`.

### `BATCH_NORM_RULES` admission sweep — ran, refused at typed-pay

All three stored as objects and gauntleted on the real corpus
(364 terms / 245 probes; ~10 s corpus + ≤10 s per rule — no bounding
needed).  Truth sweep green everywhere (synth 16eq/0ne/0rerr each;
the full variant 10eq/0ne real).  **Stage 6 `typed-pay` refuses all
three**: `batch_norm_expand` fires 8 well-typed times but `paid=0`
(the fused member is always cheaper — the expose direction is
infrastructure, not an extraction win); the two short arities match
no corpus arity (`fires=0`).  `DEFAULT` unchanged.
Retro: `retros/bn-admission-sweep.md`.

### `conv_channel_scale` anchor landed — the fold mints by iterated composition (2026-10-19)

`CONV_FOLD_RULES` (60 generated rules, opt-in with `BATCH_NORM_RULES`)
closes gate A of the conv+BN fold: `conv_channel_scale{,_l,_rev}` +
`conv_channel_shift{,_l,_sub}` × 5 attr-prefixes × 2 bias arities —
the conv twin of `linear_channel_scale`, per-OUTPUT-channel scale
folded through W's axis 0 and the bias, per-channel shift folded into
the bias.  Guards are data (concrete rank-4, `sv == (1,W[0],1,1)`,
`x[1] == W[1]·groups`); views `WV`/`SV` are dspec-derived.  Verified
fp64 vs `F.conv2d` incl. `groups=2` (`tests/test_conv_fold_rules.py`).

Probe re-run (`lab/probes/conv_bn_fold_synth_probe.py`):

- **Pairwise `synthesize_rules` still refuses the bare fold** —
  derivation depth is exactly 2 over a fixed `usable`; no
  conv2d-headed composite in any bank.  But the anchor participates:
  `syn_batch_norm_expand__conv_channel_shift_sub_sp_6` absorbs the
  running mean into the conv bias in one composite.
- **Iterated composition mints it** (gate B3 — emitted composites
  re-enter the bank): `batch_norm(conv2d(x,W,b),…) → conv2d(x, W′, b′)`
  in **4 rounds**, verified fp64-exact (6.8e-13).  Named next step:
  a `synthesize_rules` fixpoint/`rounds=` knob, or the ADR-0008
  anchored mint — depth was the refusal, not material.
- Retro: `retros/conv-channel-scale-anchor.md`.

### `synthesize_rules(rounds=)` shipped; CONV sweep refuses at truth (2026-10-19)

The fixpoint is a real API (`18e82dd`): `rounds` feeds each emitted
stratum back into the bank (`usable = base + latest`), `max_emitted`
bounds the aggregate mint (validation cost dominates — the cap bounds
spend, not just list length), `on_round` observes/vetoes strata.
`rounds=1` is the unchanged single pass; the body moved verbatim to
`_synthesize_once` (radon re-keyed, 51→51).  The probe now calls the
API and reproduces the mint exactly — 26→61→242→944 raw strata, 974
unique after alpha-key dedup, fold fires + fp64 `allclose`
(`62dce40`).

The `CONV_FOLD_RULES` admission sweep ran
(`lab/probes/conv_fold_admission_probe.py`): **0/60 usable, refused at
stage 4 truth** — earlier than BN's typed-pay refusal.  Not a
counterexample: the guarded-region sweep is *vacuous* (0 accepted /
2360 envs incl. the `_GUARDED_CAP` window; the enumeration bank can't
spell the conv channel-view corner, the corpus has no spelled
`mul(conv2d, view)` sites, `derivable=False` with no conv premises).
The family is fp64-verified true in its unit tests — the refusal is
domain starvation + enablement-only value.  BN re-confirmed identical
typed-pay refusals.  `DEFAULT` unchanged; the "enablement-aware
floor" requirements are named in the retro (corner-reaching truth
route, premise-of-admitted-composite pay, guard transport for stored
composites — measured: `law_to_data` honestly flags guarded mints
`serializable: False`, `rulecache` loads only against a bank that
contains the parents).  Retro: `retros/fixpoint-api-and-conv-sweep.md`.

### Cached fixpoint + the enablement floor, measured (2026-10-19)

`synthesize_rules_cached` takes `rounds`/`max_emitted`/`on_round`
(`02e05f9`): the scalars join the cache key; `on_round` keys through
its `_hook_sig` proxy (a vetoing observer must not share the
unobserved entry); a warm hit calls no observer.  `RuleCache.load`
resolves `syn_*` parents payload-internally — fixpoint outputs
round-trip self-contained.  Guard transport landed (`ca42c16`):
`store_object` writes the `guard_pats`+`parents` spec; `admit_object`
rebuilds hooks transitively from stored parents; `full-data` counts
missing-minus-`transported`.  The premise-pay account ran
(`lab/probes/conv_fold_premise_pay_probe.py`): the conv+BN composite
admits, fires, truth-verifies (real site 1eq/0ne; premise-derivable,
2-step replayable cert) but **does not pay** — `dag_cost` cannot see
the lowering-time weight fold, so extraction keeps the spelled term.
Verdicts: the 4 conv anchors + `batch_norm_expand` **enabled-but-opt-in**;
the other 56 conv rules refused; `DEFAULT` unchanged.
Retro: `retros/enablement-floor.md`.

## Session — 2026-10-19 (b): catopt-kit — the modular handl composition

The `go (A)` direction ("catopt embeds handl as an optional peer") gains
its runtime seam: `packages/catopt-kit/` — a plain Rust library crate in
a new root cargo workspace (catopt-native and lab/handl-peer excluded,
standalone as before). It assembles the canonical `Language` via
`handl-kit-data-canonical`, installs it process-wide, binds the smallest
honest engine set (`data`/`pratt`/`tree-walk` — no sdkgen/compile/lint/
lsp), and composes `host-tensor`'s 18-op surface as a `HandlModule` over
`tensor_op_registry()` under the canonical fp host fns. A `ModuleLoader`
wires `import "path"` / `.handl.toml` `[dependencies]` name resolution.

**Verified** (`cargo test -p catopt-kit`, **12/12**): real `.hnl` matmul
→ `[[19,22],[43,50]]`; the result flows back as `TaggedValue::Tensor`
(both `run`'s wire form decoded and `run_tagged`'s direct variant — no
`Obj` fallback); conv2d + batch_norm smokes match the upstream spec
values; `from "host-tensor" import tensor` works on a fixture project.
Deps float on `main` via git + `handl_deps.env` insteadOf; the `[patch]`
local-override convention is documented in the root `Cargo.toml`.

**Named gap — closed (second pass, same day)**: platform `90853ec2`
landed the git-package→dir bridge — `[dependencies]` accepts
`{ git = …, rev|branch|tag }` (unpatched → the `~/.handl/pkg/<name>`
cache slot the ADR-0010 fetch hook will populate; a symlink is today's
stand-in) and `[patch]` dev-links the sibling checkout, winning when
its dir exists. The crate ships a committed `.handl.toml` +
**`hnl/conv_bn.hnl`** — the conv→batch_norm composite the e-graph
folds, importing `host-tensor` *by name* through the bridge — plus
`conv_bn_fold`, the minted rewrite's RHS evaluated (`s = γ·rsqrt(σ²+ε)`,
`W′ = W∘ŝ′`, `b′ = b∘s + β − s∘μ`), and `bn_adjoint`/`bn_grad`.
**Verified** (`cargo test -p catopt-kit`, **16/16**): `TaggedValue::
Tensor` results and *exact* folded-vs-unfolded agreement on a dyadic
fixture — the consumer proof of the premise-pay story below.
**Env note**: tree-walk eval needs a >2 MiB test-thread stack —
`.cargo/config.toml` sets `RUST_MIN_STACK = 32 MiB` (upstream carries
the same knob). Retro: `retros/catopt-kit.md`.

### `FOLD_OPS`: the premise-pay leg flips — cost model sees compile-time folds (2026-10-19)

The enablement-floor blocker was **alphabet drift, not a missing
model**: `dag_cost`/`extract_best`/`executor_overhead` already price
param-only subtrees at ~0 marginal, but `cost._folds_to_param`
consulted a 29-op table that had drifted ~130 names behind
`IRModule._fold_weight_chains`' 160-op `elif` gate (commit e6a618e
widened the lowerer, the cost side never re-synced) — and
extraction's `_FOLDABLE_OPS` was narrower still (no views).  The
conv+BN W′/b′ chains' `rsqrt` root therefore billed the whole fold
per-call.

`opmeta` now carries the `fold-at-lowering` tag + the 161-name
membership; `FOLD_OPS` is the one table the lowerer gate, the fold
predicate, and the extraction discount all read (drift structurally
impossible — pinned by `test_fold_alphabet_is_single_sourced`).  60
binding-only spellings register into the vocabulary through it.

**Measured** (`conv_fold_premise_pay_probe.py`): `fires=1
changed=False verified=same` → `changed=True verified=pass`;
`dag_cost` fused `34 874.337` → `17 438.7` vs spelled `34 874.3`.
Truth unchanged (real site 1eq/0ne; 2-step replayable cert).

**Guarded sweep verdict — DEFAULT unchanged**: the 4 conv anchors
refuse at **truth** again (`0 accepted / 2360 envs` — the conv
channel corner stays unreachable to the enumeration bank;
enablement-not-admission stands) and `batch_norm_expand` refuses at
typed-pay (`fires=8 paid=0` — expansion is the non-paying direction;
the composite is the payer).  Retro: `retros/fold-cost-single-source.md`.

### `_MAX_BINDING_ENVS`: the composite gauntlet terminates (2026-10-19)

The enablement floor's named scaling gap was **misdiagnosed**: not the
83-rule `_reach`/derivability saturation (measured instant on the
conv+BN seed — 9 enodes to `fixed_point`), but the binding-env
enumeration under `oracle.synthesize`.  `synthesize` /
`evidence._synth_sites` / `object_synthesis._oracle_sites` capped the
*sites they yield*, never the envs they consume — the composite's
composed `derive` vetoes every env (2000/2000 at ~8k envs/s on a
>10⁹ space), so the sweep never produced a site and never stopped.

`oracle._binding_envs` now stops after `_MAX_BINDING_ENVS = 250_000`
yielded envs — one bound every consumer inherits.  Sized on
measurement: `sdpa_fold_nomask`'s ~180k envs to natural exhaustion is
the largest legitimate sweep, so shipped sweeps are preserved; a
veto-everything domain degrades to a bounded corner of the Cantor
order (a partial sweep — the documented honesty of the site caps).

**Measured** (`conv_fold_premise_pay_probe.py`): `run_gauntlet` on the
one-case corpus completes in **4.9 s** (was >9 min, unbounded) with an
honest refusal — `truth` starved: `0 accepted / 2000 declined / 2360
envs` on the synth region, `1eq` on the real site, `derivable=False`
under `ALL_RULES` (the premise closure is outside the corpus rules).
Pay leg stands: `fires=1 changed=True verified=pass`, fused `17 438.7`
vs spelled `34 874.3`.  Retro: `retros/binding-env-cap.md`.

### Premise-anchored `derivable`: the minting proof counts, guarded (2026-10-19)

The named policy question in `binding-env-cap.md` — should a stored
composite's `derivable` leg see its premise closure — is **decided:
yes, with a site-verification guard**.  `run_gauntlet`'s measure stage
now evaluates derivability against `corpus.base_rules` + the record's
`guard.parents` resolved transitively through the store (the same
full-strength bar guard transport sets: stored row, `missing_hooks`
minus `transported` empty; a base-named premise resolves to the axiom
itself).  Anti-circularity: the anchored certificate must replay
standalone *and* site-verify every non-axiom step
(`eval_instance` on the recorded instance) — a false stored premise
merges the e-graph but measures `unequal` where it was applied, and
the credit revokes to the base-only verdict.

**Measured** (same probe): `derivable` flips `False` →
`True(premise-anchored)` (replayable 2-step cert, premise steps
site-verified); `truth` and `typed-pay` now pass (`paid=1`, fused
`17 438.7` vs `34 874.3`).  The composite's refusal moves to
**`closure`** — `enode=6.00x` vs the 2.0 cap: expansion cost on the
one-term graph, a different gate, left open as the next named policy
question.  Retro: `retros/premise-anchored-derivability.md`.

### `JOINT_SHAPES`: the guarded sweep reaches the conv corner (2026-10-19)

`binding-env-cap.md`'s named residual — "the binding banks have no
conv-weight-shaped inhabitants" — is closed.  `lawdata.JOINT_SHAPES`
tabulates correlated operand-shape rows keyed `(op, arity)`; a metavar
arg of a tabulated node leaves the independent banks and draws through
the joint row set (merged across nodes under Var agreement), and
`JOINT_VIEW_SHAPES` appends the `(1,C,1,1)` channel-view corner via
`extra_viewed` — joint-free enumerations are byte-identical.
`conv2d`'s `stride` is a measured `pos-int` override (every accepted
anchor site died on `non-positive stride`); `dilation`/`groups` are
positive by name default.

**Measured** (`conv_fold_premise_pay_probe.py`): the composite's
guarded synth region went `0 accepted / 2000 declined / 2360 envs` →
`105 accepted / 105 equal / 0 ne / 0 rerr` in the 360 window — truth
PASSES (`derivable=True` premise-anchored), novelty + typed-pay pass
(`paid=1`).  Every conv anchor sweeps a verified-equal region inside
the default window; `batch_norm_expand` keeps its corner.  The
composite's refusal moves to **closure** (`enode=6.00x` — the cost
gate named in `premise-anchored-derivability.md`, not a truth
question).  Retro: `retros/joint-shape-schemata.md`.

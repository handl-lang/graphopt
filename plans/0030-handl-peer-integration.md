# Plan 0030 — the handl peer: from "the bank loads" to "the kernel rewrites"

Depends on: **ADR 0010** (handl *is* the kernel — one language, many
interpretations), **ADR 0011** (the calculus is canonical for laws, the
op-graph for execution), **plan 0029** (the boundary spike — **done**),
and the spike retros (`handl-q0..q3`).
External: `handl-lang/platform` — `handl-kit-data-canonical`
(`language_from_dir`), `plugin-engine-transform-rewrite` (the
`RewriteEngine`), `plugin-engine-exec*` (the host-function seam).

## Where the spike left it

| measured | state |
|---|---|
| the **load** | ✅ the whole bank — **83 laws** — assembles into a `Language`; the three minted leaf kinds are declared (handl `9372bd32`, catopt `5ca8aef`) |
| the **bridge** | ✅ lossless both ways on the shipped bank (no silent drops; every row re-emits byte-identically) |
| the **channel** | ✅ exists (`TaggedValue::Resource` — an opaque pointer by reference), and deliberately **unused** in v1 |
| the **direction** | ✅ (A) catopt embeds handl as an optional peer (the `catopt-native` pattern) |
| **a rewrite firing** | ❌ **never observed.**  Loading a `Language` is not running its engine, and no catopt-derived term has been rewritten by handl |
| **the peer** | ❌ not built |

**The gap that matters is the fourth row.**  Everything so far proves
catopt's vocabulary can *sit* in handl; nothing proves handl's engine
can *do* anything with it.  That is step 1 below, and it is the step
that decides whether the rest is worth building.

## Steps, in order, each with its pass criterion

### 1 · Make a rewrite fire (hours — decisive) — ✅ **DONE**

**Measured 2026-10-14** (handl `82194954`): `assoc_add` fires through
handl's `RewriteEngine` — exactly one rewrite, and the result is
`add(add(a, b), c)`.  Cross-checked against catopt's own engine on the
same term: `EGraph.apply_rule` fires and the expected member lands in the
root's e-class.  **The two engines agree**, so step 2 is justified.

Build a handl-side test that takes a catopt-derived `Language`, an
`Ast` shaped like a catopt term, and runs the `RewriteEngine` with
catopt's law as the rule.

- **Pass**: the rule fires, and the rewritten term equals what catopt's
  own `apply_rule` produces for the same input (structurally, and
  numerically where leaves carry values).
- **Fail**: name *why* — a missing role, a step the engine does not
  interpret, an `Ast` shape the engine cannot traverse.  A named
  failure here is the honest end of the integration: **if a rewrite
  cannot fire, the peer is not worth building** (step 2), and the
  relationship stays "catopt can *emit* handl data".
- This is Q0's missing half, and the plan should not proceed past it
  without an answer.

### 2 · The peer (`catopt-handl`) — only if step 1 passes

A `PyO3`/`maturin` extension exposing `language_from_dir` + the rewrite
engine to Python, following `catopt-native` exactly: optional, excluded
from the uv workspace, opt-in.

- **Pass**: with the peer built, `import catopt_handl` loads the bank and
  rewrites a term; **catopt's default path does not require it** (no
  import from `catopt_core`, no CI toolchain); the seam test
  (round-trip identity on the shipped bank) still passes.
- **Fail**: if it cannot be optional — i.e. something in the default
  path needs it — the cost lands on every user and the design is wrong.

### 3 · Rule forms *in handl*: leaves **and** root-eliminating laws

Two gaps now measured, both of the same kind (a step handl's vocabulary
does not have), so they belong in one step:

* **a leaf** — today `!Create` of a declared `catopt:`-namespaced kind
  with the data in the payload (loadable and lossless via the `catopt:`
  record, but handl's engine sees an opaque created node);
* **a root-eliminating law** (5 in the bank: `id_add`, `id_mul`,
  `double_neg`, `transpose_noop`, `chunk_single`) — handl's
  `TransformRule::Declarative` always *creates* a node
  (`target_kind: String`), so a law that returns a subterm cannot be
  expressed faithfully; the differential probe
  (`project/retros/handl-differential-rewrite-check.md`) reports these
  separately rather than papering over the divergence.

**Pass**: a law with a **constant** leaf fires and produces the right
*value*, and a root-eliminating law fires and returns the *subterm*.
**Fail**: the workarounds stay, and they must be documented as
workarounds — with the note that the `catopt:` record is authoritative
and that handl's engine sees something coarser than the law means.

#### 3a · The root-eliminating gap, investigated (2026-10-16)

The gap is real, and the *first* thing to fix was a bug in catopt's own
bridge, not a missing handl form.

**What `target_kind: None` actually did (measured, through the peer).**
`TransformRule::Declarative.target_kind` is a Rust `String`, not an
`Option` — but `language_from_dir` reads `transforms.yaml` with
`serde_yaml::from_str::<Vec<TransformDef>>`, and serde_yaml coerces the
YAML `null` scalar into the **string `"null"`** for a `String` field.
So the row *loaded*, and the engine (`walk.rs::fire_untraced` →
`step.rs::build_rewrite`, which does `kind: target_kind.to_string()`)
**silently fabricated a node of kind `"null"`** whose children were the
kept child steps.  Firing the old `double_neg` row on `neg(neg(a))`:

```
root: {'kind': 'null', 'payload': {'children': [0]}}   # handl
       a                                                # catopt — the subterm
```

A silent lie, not a refusal — and handl's `validate_transforms` only
*Warns* ("targets unknown kind 'null'"), which does not fail assembly.

**The bridge now refuses explicitly** (catopt `8548243`): for a
bare-metavar rhs it emits `rule: !Native "catopt:eliminate_root"` — a
*declared* handl rule form naming the missing operation — instead of
`target_kind: None`.  handl's engine then rejects it loudly
(`"transform rule requires native rewrite 'catopt:eliminate_root' but
no native registry is wired"`), which is the honest outcome: the bank
still loads (83 transforms), the law still round-trips through
`catopt.law`, and the divergence is *louder*, not hidden.  The probe is
unchanged (it classifies root-eliminating laws before firing).

**Is there a faithful form in handl's *existing* vocabulary?**  No.
Checked against the engine source:

| candidate | verdict |
|---|---|
| `RewriteStep::Splice(i)` (`step.rs:240`) | **no** — lifts a child's *children* into the created node; the `Declarative` node still exists (`build_rewrite` always pushes `target_kind`) |
| `Keep`/`KeepPath` at the *rule* level | **no such thing** — `KeepPath` is a *child step* (`step.rs:186`); it produces a child, never the rewritten node |
| replace the node *by* a child | **no** — `Declarative` always creates; only `Native` can return an existing node |
| `TransformRule::Native(name)` | **the only mechanism, but not a faithful form here** — a native returns an `AstRef`, so `ctx.remap_child(i)` *can* return a subterm (`plugin-engine-transform/src/native.rs:118`); but (a) it needs a wired `NativeRewrite` registry — the peer wires none, so `fire_native` errors (`walk.rs:410`) — and (b) the implementation would be **catopt's** Rust, so handl's engine would be running catopt's code and the differential would stop being two *independent* engines (circular for exactly these 5 laws).  That is why the bridge emits it as a **refusal marker**, not as a fix |

**Option A — a rule form in handl (the honest fix) — TAKEN, see 3a-ii.**
handl needs one declarative rule shape that *returns an existing node*
instead of creating one.  The minimal exact shape:

```rust
// handl-language/src/fragments/transform_def.rs — TransformRule
Replace { path: Vec<usize> },   // the rewritten node IS the descendant at `path`
```

i.e. "the result is the node at index path `p` below the matched node"
— `Replace { path: [0, 0] }` is exactly `double_neg`.  It is a
`KeepPath` promoted to the rule level: the engine already navigates the
path and remaps the target (`step.rs:186` `KeepPath`), so
`fire_untraced` resolves the final index and returns it without
pushing a node.  Cost: a new `TransformRule` variant + one `walk.rs`
arm + validation — a **vocabulary addition**, which per handl's
`engine-genericity` rule is *an engine edit and handl's design
decision*, not something catopt may make.  With it, the bridge maps a
root-eliminating law to `!Replace { path }` and the 5 laws leave the
"root-eliminating" bucket and become *comparable*.

**Option B — a declared `catopt:alias` kind whose payload carries the
path.  This is *not* a fix.**  It loads and is lossless via the
`catopt:` record, but handl's engine then sees an opaque *created*
node — a plausible-looking `catopt:alias` wrapping the subterm — so the
semantic difference is **hidden from the very tool built to find
divergences**.  It is strictly worse than the loud `!Native` refusal:
same non-faithfulness, but silent.

#### 3a-ii · Option A implemented — handl ADR-0127 (2026-10-16)

**DONE.**  handl took option A: **handl ADR-0127**,
`project/adrs/0127-transform-rule-replace.md` (handl `c538b766`).  The
variant lives in
`crates/handl-language/src/fragments/transform_def.rs`; the engine arm in
`crates/plugin-engine-transform-rewrite/src/walk.rs::fire_untraced`
(one arm, sharing `step::descendant_index` with `KeepPath` — so `Replace`
is literally `KeepPath` promoted to the rule level); validation in
`crates/plugin-engine-transform/src/validate.rs` (empty path and
out-of-range first index are **Errors**, mirroring `KeepPath`).  An
unresolvable path is a hard error at runtime too — **never** a fabricated
node, which is the exact failure mode that motivated the ADR.

handl tests: 5 engine tests (`tests/engine.rs` — returns-the-subterm,
deep path, top-down traversal, out-of-range error, empty-path error), 2
validation tests (`plugin-engine-transform/src/lib.rs`), a serde
round-trip (`transform_def.rs`), and a conformance test that loads
catopt's emitted `!Replace` rows and fires `double_neg` through the real
engine (`crates/handl-kit-data-canonical/tests/catopt_fragments.rs`).

**The bridge now emits `!Replace { path }`** (catopt `8bf9911`): for a
bare-metavar rhs *bound in the lhs*, `_rule_body` returns the metavar's
lhs path — `double_neg` → `[0, 0]`, `id_add`/`id_mul`/`transpose_noop`/
`chunk_single` → `[0]`.  The `!Native` refusal is now only the residual
fallback for a bare-metavar rhs with *no* bound path (absent from the lhs,
or bound at the lhs root); no shipped law has one, so the whole bank maps
faithfully.

**Measured — the differential re-run** (`lab/audit/handl_differential.py`,
same peer, new fragment fixtures):

| bucket | before | after |
|---|---|---|
| compared | 19 | **24** |
| agree | 19 | **24** |
| **DISAGREE** | **0** | **0** |
| skipped (leaf rhs) | 3 | 1 |
| declined | 37 | 39 |
| guard raised (synth) | 19 | 19 |

All five root-eliminating laws are now **expressible**.  The probe reports
them per law:

```
  root-eliminating laws (bare-metavar rhs — handl !Replace, ADR-0127):
    - id_add: agree
    - id_mul: agree
    - double_neg: agree
    - transpose_noop: declined
    - chunk_single: declined
```

**Three agree in the probe** (`id_add`, `id_mul`, `double_neg`); the other
two are **declined by catopt's own engine** on the probe's synthesized
term — their guards (`axes-noop`, `attr-eq CK 1`) need concrete attrs the
generic synthesis cannot supply (retro Finding 2).  Given a
guard-satisfying term **both engines return the subterm**, so **all five
agree wherever both fire** — pinned by
`tests/test_handl_peer.py::test_root_eliminating_laws_agree_on_satisfying_terms`.

The probe was adjusted honestly, in two places, and **hides nothing**
(`DISAGREE` is still 0):
1. the obsolete `_eliminates_root` pre-classification is gone — the laws
   are fired and compared like any other structural law;
2. the "structural laws only" filter now filters the **rhs** only, which
   is its documented intent.  Filtering the lhs too had needlessly skipped
   the four laws whose *lhs* carries a leaf but whose rhs is structural
   (`id_add`, `id_mul`, `softsign_fold`, `pow_to_square`) — a leaf in the
   synthesized **input** is perfectly comparable, and all four agree.

So the 19→24 gain is the 5 root-eliminating laws minus the 2 the guards
decline, plus the 4 the lhs-filter hid.

### 4 · Decide `priority` (small, but say it out loud)

`priority` currently exists only because handl's rule identity is
`pass:node_kind:priority` and catopt has many laws per root kind.

- **Pass**: either catopt carries a *real* order (e.g. a policy-driven
  rule order, which the RL layer could learn) and it is documented as
  meaningful, or it stays a disambiguator and the docs say so.  No third
  state.

### 5 · ADR 0008 gate 3 — does search *reach* `loop`-headed forms?

The gate that decides ADR 0009's stage 4 (whether the calculus earns a
package).  Catopt-side, and independent of the peer.

- **Pass/fail**: measured yes or no.  A **no** means the calculus is a
  presentation win, stage 4 is not justified, and that should be written
  down rather than deferred again.
- **MEASURED 2026-10-16 — the answer is NO.**  The proposing rule was
  built (`lab/bridge/loop_propose.py`, catopt `b7ba9ff`): a sound
  rewrite `unroll(step, n) → Loop(step, n)` over the presentation.  It
  fires and reaches a `Loop`-headed winner on the presentation's
  *spelled* scan (the control), but on the real export the search's
  winner has **zero** `unroll(step, n≥2)` subterms — the recurrence is
  parametrically regular (per-block `split(index=i)`) but not
  *syntactically* regular — and the presentation's spelled form is not
  even in the search's root e-class.  So the rule fires 0× and no
  `loop`-headed winner is reached.  **Stage 4 is not justified by the
  presentation's discovery value**; the missing piece is a *form-level*
  proposer that abstracts the step, not a syntactic rewrite.  Written
  down here and in ADR 0008's gate-3 FAIL bullet; retro:
  `project/retros/gate3-loop-proposing-rule.md`.
- **MEASURED 2026-10-16 (later) — still NO, with the cause measured.**
  The form-level proposer that bullet named was built
  (`lab/bridge/index_abstract.py`, catopt `e4a193a`): a *parametric
  abstraction* that binds the per-block index (and the recurrence state)
  as parameters, wired as `--abstract-index`.  It collapses a *regular*
  fold to one shared step and the closure reaches a `Loop` (control
  PASS, `best_is_loop=True`), but on the real export the 7 blocks still
  do not collapse: they span **two chunk sizes** (4×size-4 + 3×size-16,
  from two `split` families) and extraction **mixed two state
  spellings** (the shared-state read `matmul(q, S)` and the intra
  re-expansion `om_decayed(matmul(q, k_prevᵀ), D, v_prev)`).  First
  difference reported by the move:
  `block1/block2/add[1]/mul[1]: op om_decayed vs add`.  Canonicalising
  every block modulo the varying integers leaves 7 distinct blocks, so
  no constant-parameter abstraction can unite them.  **Stage 4 is still
  not justified**; the missing piece is larger than the previous bullet
  implied.  Written down here and in ADR 0008's second gate-3 FAIL
  bullet; retro: `project/retros/gate3-parametric-abstraction.md`.

### 6 · Laws as data, exclusively (the standing principle)

With the calculus canonical for laws (ADR 0011), the bank's *source* can
become rows rather than Python constructors.

- **Pass**: the procedural-hook count (guards/derives the DSL cannot
  express) does **not** rise — it may only fall (ADR 0010's ratchet).
- **Fail**: the data format is too weak, and the count says by how much.

## Risks (measured where possible)

- **The second toolchain.**  handl's debug target is ~8.5 GB with default
  debuginfo; `CARGO_PROFILE_DEV_DEBUG=0` is the documented mitigation.
  Step 2 must keep this out of catopt's default path or the cost is real.
- **A load-only relationship.**  If step 1 fails, the honest outcome is
  "catopt can express its vocabulary as handl data, and handl can load
  it" — useful, but *not* a kernel.  Do not let the ADRs drift into
  implying otherwise.
- **The leaf workaround could ossify.**  It is lossless, which makes it
  easy to leave.  Step 3 is what keeps it honest.
- **Coverage floors** apply to any new Python surface; a Rust peer is
  verified by handl's suite plus the seam test, and that boundary must
  be stated, not assumed.

## Falsification of this plan

Step 1 is the gate.  If a rewrite cannot fire on a catopt term through
handl's engine, then **plan 0030 ends at step 1**, the peer is not
built, and ADR 0010's "handl is the kernel" is downgraded to "handl can
host catopt's vocabulary" — a smaller, truer claim.

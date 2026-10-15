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

### 1 · Make a rewrite fire (hours — decisive)

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

### 3 · Leaf semantics *in handl* (a real step, not a workaround)

Today a leaf maps to `Create` of a `catopt:`-namespaced kind with the
data in the payload: loadable and lossless, but handl's engine treats it
as an opaque created node.  Either handl gains a literal/variable step,
or the kinds are declared in a shared schema the engine resolves.

- **Pass**: a law with a **constant** leaf fires through handl and
  produces the right value (not merely a structurally-equal node).
- **Fail**: the workaround stays, and it must be documented as a
  workaround — with the note that the `catopt:` record is authoritative.

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

# Retro — `conv_bn_fold` ships: `DEFAULT` takes a machine-minted composite

**Date**: 2026-10-19.  **Follows**: `premise-anchored-derivability.md`
(the anchored `derivable` leg), `binding-env-cap.md` (the terminating
gauntlet), `fold-cost-single-source.md` (the pay leg), `0f5fc15` (the
closure gate's bounded-footprint forgiveness) — the conv+BN composite's
gauntlet is fully green: `usable=True` on the measured corpus.
**Commit**: `445d1c7` (the ship) + this commit (retro + ledger).

## The question

Does `DEFAULT` take a machine-minted `syn_*` composite, or is admission
for authored family members only?  The honest tension the task named:
the composite's premise family — `batch_norm_expand` + the four
`conv_channel_*` anchors — is itself **enabled-but-opt-in** (anchors
refused at truth-starvation, the expand at typed-pay).  If the composite
enters `DEFAULT` while its parents don't, is that coherent?

## The decision — yes, and the coherence was already settled

**Ship it, verbatim.**  Two precedents carry the coherence argument:

* `linear_channel_to_row_scale` sits in `DEFAULT` while both its
  premises are SYMMETRY-gated opt-ins — the shipped composite edge is
  self-contained: its guard is the transported conjunction evaluated on
  its own binding, and at fire time it does *not* replay its parents.
  The parents' exclusion reflects *their own* measured economics
  (closure cost, standalone pay) — a per-law account the composite does
  not inherit.  The conv+BN case is the same shape, one level deeper.
* The premise-anchored machinery was built for exactly this: the
  composite's `derivable` credit is premise-scoped *by construction* —
  "derivable under the premises I was minted from", each step
  site-verified.  Requiring the premises to also be DEFAULT members
  would re-conflate the two axes the anchored gate was built to
  separate: a premise's *truth* (unit-verified, on which the anchored
  proof rests) vs its *default-set economics* (the opt-in verdict).

The residual candidates checked and discharged:

* **runtime correctness** — the transported `check`/`derive` are
  closures over the *same* library parent objects the mint composed
  over; they resolve at fire time, not at storage.  Verified: the
  shipped `CONV_BN_FOLD` fires on the spelled eval-mode site producing
  the identical fused instance (`dag_cost` 34 874.3 → 17 438.7) and
  declines a channel-mismatched binding.
* **closure safety in `DEFAULT`** — the firing footprint is bounded
  (19 minted enodes + cascade ≤ 2×), measured, not a saturation
  blow-up; the rule fires once per guarded site — term-local, like
  every other shipped fold.
* **truth** — premise-anchored derivable (replayable cert,
  site-verified steps) + 105eq/0ne on the guarded synth region +
  1eq/0ne on the real site + `verify=pass` on the lowered modules.

## What shipping a composite means mechanically

`promoted-laws.md`'s class shipped by *transcription*: each admitted
object was re-written as an `R(...)` whose `cond` was hand-verified to
be "exactly what compose_objects minted".  That path does not scale to
a depth-4 interleaved check+derive composition — a hand-flattened
`cond` for it would be an *unmeasured twin*, not the admitted object.
So `conv_bn_fold` introduces a second shipping class:

**`catopt_core.laws.minted`** — the verbatim path.  The stored-closure
records (the four `syn_*` nodes' `law_json`, each carrying its
`"guard"` spec: parent names + `guard_pats` re-expression maps) embed
as data; `materialize()` replays `evidence._rebuild_stored`'s semantics
at import over the embedded table plus the shipped library's name
table.  Deliberately *louder* than the store: a parent that resolves
short of declared strength — absent, cyclic, or weaker-than-declared —
raises at import rather than admitting a degraded rule.  A shipped
object may never silently lose its guard.

The honest consequences, pinned:

* `missing_hooks(conv_bn_fold) == ("check", "derive")` — the first
  shipped law that is not bare-record full-data.  The census
  (`test_law_serialize.py`) gains a **transported** bucket keyed on
  `guard_pats` — the class is marked, not hidden: the guard's *spec* is
  data even though the hooks are code, and `store_object` re-encodes
  the spec faithfully on re-store (pinned round-trip test).
* `kind == "axiom"` by annotation: the `derivation` field can only name
  shipped rules, and the true derivation runs through three unshipped
  `syn_*` intermediates — and the `test_axiom_lemma_marker` premise
  check requires recorded names to resolve inside `ALL_RULES`, so a
  partial annotation would be a *worse* lie.  The lineage rides
  `rule.parents` + the embedded records instead.
* `emit-basis` re-measured: `ALL_RULES` 83→84, `conv_bn_fold` reads
  `no-instance` like every other shape-guarded promotion (the
  instancer builds no site); declared split 60 axioms / 22 lemmas /
  2 redundant; measured no-instance 19→20.

## The alternative considered — a `MACHINE_DEFAULT` adjunct set

Consulted-alongside-`DEFAULT` segregation was weighed and rejected on
honesty grounds: either every recommended caller composes it (then it
*is* DEFAULT membership with indirection, and the class boundary is
fiction) or nobody does (then the measured evidence is stranded —
"admitted but unreachable").  The class distinction belongs in
*provenance* (`MACHINE_SHIPPED`) and on the object (`guard_pats`,
`parents`), not in set membership.  The one real residual that would
have justified the adjunct — a guard the library's audit surfaces
cannot read — is addressed by marking the class (census bucket +
provenance), not by quarantining it.

## Second-order effects observed post-ship

* `conv_bn_fold` in the mint base changes what `synthesize_rules`
  finds: the probe's fixpoint now emits `syn_conv_bn_fold__comm_add_0`
  first — a trivial comm-compose whose guard spec names the *shipped
  composite* as a parent, and whose transport correctly refuses when
  the composite's own premise rows are absent from the store.  The
  probe pins its mint base to the pre-ship surface so it keeps
  measuring the admitted object.
* The same mint re-runs deterministically (974 emitted, identical
  `syn_*` names across runs) — the embedded records are the measured
  artifact, not a sample.
* Re-running the gauntlet post-ship refuses at **novelty**
  (`relation=duplicate`) — the documented signature of "already
  admitted", identical to the promoted-laws aftermath.
* The premise-anchored leg now resolves through the shipped law:
  `verify_law` under shipped axioms derives the mint in one step
  (`witness=('conv_bn_fold',)`) — the library law proving its own
  minted origin.
* The library-side guard sweep (`test_guard_sweep._guarded` keys on
  `cond or check or derive`) *does* cover the transported class —
  `conv_bn_fold` swept 218 sites: 1 true, 217 false, **0 violations**.
  Thin acceptance is the enumeration's conv-corner starvation the
  joint-shape work already names, not a coverage hole; the gauntlet's
  guarded-region sweep (105eq/0ne) remains the deeper measurement.

## What stays open

* A *flattened* `cond`/`dspec` transcription of the transported guard —
  the composite's accepted region is a conjunction of renamed parent
  conds over inline-derive attrs, which the DSL can express — would
  restore the full-data census property.  It needs a flattener (or a
  verified hand transcription), then a fresh gauntlet run on the twin.
* The LHS carries the seed's literal attrs (`stride=(1,1)`,
  `padding=(1,1)`, `eps=1e-05`, `training=False`): the law's region is
  the default-attr eval-mode fold — narrow, honest, and the common
  spelling; generalizing the attr metvars is a new measurement, not a
  transcription detail.

Gate account: `tests/test_minted_laws.py` (+9), the count pins
(`test_laws_structure`, `test_law_serialize` census,
`test_law_provenance` table) re-pinned at 84/285/12; focused slices
green; ruff/format/ty/vulture/lint-imports/bandit/semgrep/radon clean on
touched files; dev lane (`-m "not slow"`) run logged in GOALS.

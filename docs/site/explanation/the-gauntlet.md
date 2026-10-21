# The gauntlet — what a minted law must survive

<!-- provenance: graphopt:mlopt_discovery.verifier + docs/{public_api,mechanism}.md@eea02f6 -->

A candidate law is *proposed* cheaply and *admitted* expensively. The
admission gauntlet is the pipeline every minted law runs before it may
enter `DEFAULT` — and the reason the shipped bank can be trusted
without trusting the proposer.

```text
propose → numeric oracle → derivability check → typed-pay gate
        → adversarial sweep → certificate replay → MACHINE_SHIPPED
```

## The stages

1. **Numeric oracle.** Both sides of the candidate equality execute on
   real tensors (seeded fixtures, fp64) and must agree — equality is
   tested, not assumed.
2. **Derivability check.** `verify_law(lhs, rhs, rules)` puts both
   sides in a fresh small e-graph and saturates under the known laws —
   the e-graph *is* the decision procedure for e-class membership. On
   a merge it emits a positional certificate and replays it on real
   terms; the certificate is tried in both directions so a witness is
   a genuine derivation, never a trusted stub.
3. **The typed-pay gate.** The candidate must actually pay: measured
   wall/peak cost on the fixtures, under the
   [three-costs](./three-costs) discipline — `dag_cost` ranking is
   never enough.
4. **The adversarial sweep.** `mlopt_discovery.gap_gen` generates
   bindings the corpus cannot — witnesses inside the guard's *false
   region* — and tries to falsify the candidate's side conditions.
   This is where the teeth are: the sweep killed **212 bad accepts**
   on `mul_unsq_nat`'s first (plain broadcast-equality) guard, caught
   four false twins, and found a real lowering bug no test covered.
5. **Certificate + provenance.** The survivor ships in `DEFAULT` with
   `MACHINE_SHIPPED` provenance — replayable at import, auditable via
   `law_provenance(rule)`.

## The worked example — `mul_unsq_nat`

The first law minted from a *corpus op-tuple* rather than a
hand-declared need. The census saw `mul(unsqueeze, unsqueeze)` in a
real model's `β_t·k_t` write; the candidate
`mul(unsqueeze U d, unsqueeze V d) = unsqueeze(mul U V, d)` was
proposed; `auto_cond`'s first guard was **falsified by the sweep**
(212 bad accepts); the exact `flat-map-unsq` guard was synthesized
instead — 0 bad accepts, 0 misses; it pays −12.27% on its source
workload and ships in `DEFAULT`. Its apparent sibling
(`mul_unsqueeze_r_id`) was *correctly rejected* — its real sites are
semantically unequal, and the guard declines all of them.

## The referee contract

The gauntlet emits a plain `{axis: value}` measurement; the
[`ScoringSpec`](./the-game#the-referee-seam) on the game side turns it
into a verdict — `score(spec, measured)` returns gates cleared, the
weighted value, and the axes the measurement lacked. The two sides
meet only through that mapping: measurement code never imports the
scoring module, and the scoring module imports no measurement
machinery.

## The honest boundary

Admission is *regime-bounded*: fixtures are small, seeded, one
hardware box. The gauntlet's job is to make the claim *falsifiable and
bounded*, not universal — a law that pays on its fixtures and replays
its certificate ships; the scope of "pays" is part of the record.

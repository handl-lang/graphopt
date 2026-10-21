# Admit a law

<!-- provenance: graphopt:mlopt_discovery/{pipeline,verifier,emit}.py + eqsat:packages/mlopt-core/src/mlopt_core/laws/provenance.py@bac1a61 -->

Move a candidate equality through [the gauntlet](../explanation/the-gauntlet)
into `DEFAULT` with `MACHINE_SHIPPED` provenance. The discovery pipeline
proposes and ranks candidates; admission is what you do with a SHIP
verdict.

## 1. Get a candidate

```sh
.venv/bin/python -m mlopt_discovery.pipeline
.venv/bin/python -m mlopt_discovery.pipeline --holdout select_mul   # sanity: re-derive a shipped law
.venv/bin/python -m mlopt_discovery.pipeline --json /tmp/pipeline.json
```

One run does census → propose → verify → measure → rank over the real
corpus and emits a **ship / no-ship** verdict per candidate with the
reason. The `--holdout` arm removes a shipped law and asks the loop to
re-derive it — if the pipeline cannot rank a known winner at the top,
it is not trustworthy.

## 2. Check derivability standalone

```python
from mlopt_discovery.verifier import verify_law

res = verify_law(lhs, rhs, rules)   # -> LawResult
res.derivable        # the two roots share an e-class?
res.direction        # which way the witness ran
res.witness_rules    # the rules that fired the derivation
res.replayable       # the certificate replays standalone?
res.cert             # the certificate itself
```

`verify_law` interns both sides in a fresh e-graph and saturates under
the known laws — the e-graph *is* the decision procedure. Saturation is
bounded by default (expansive and non-library rules are capped), and a
bound can only ever *refuse*, never accept: resource exhaustion reports
`derivable=False`, never a fabricated merge. Or run the module CLI:

```sh
.venv/bin/python -m mlopt_discovery.verifier --json /tmp/law.json
```

## 3. Prove it pays — and that the guard is honest

The pipeline's measure stage already runs the pay audit per candidate:
the rule fires alone over every real model (fires, cost delta, lowered
`sink.verify`), every merged fire is audited for *well-typedness*, and a
closure-safety check flags rules that explode saturation. What remains
is the adversarial sweep — `mlopt_discovery.gap_gen` generates bindings
inside the guard's *false region* and tries to falsify the side
conditions. This is where candidates die: the sweep killed 212 bad
accepts on `mul_unsq_nat`'s first guard before the exact
`flat-map-unsq` guard was synthesized.

## 4. Emit the admission patch

```sh
.venv/bin/python -m mlopt_discovery.pipeline \
    --emit-admission <candidate> --out /tmp/admission
```

`--emit-admission` writes, for a SHIP candidate, the *admission
artifact*: the `R(...)` source with named `check`/`derive` hooks, a
generated `tests/test_admitted_<law>.py`, and
`admission_<law>.patch` — a `git apply`-able diff showing the exact
law-bank insertion. It is a report, never a mutation: review the patch
before it lands.

## 5. Land it

1. **`git apply` the patch.** The `R(...)` row lands in the law bank
   (`mlopt_core/laws/tensor.py`) — `cond`/`dspec` carry the declarative
   guard. A verbatim `syn_*` minted composite goes through
   `mlopt_core/laws/minted.py` instead (its premise-closure records
   replay at import).
2. **Record the provenance.** Add the law's name to `MACHINE_SHIPPED`
   in `mlopt_core/laws/provenance.py` — `law_provenance(rule)` then
   reports `"machine"`.
3. **Run the whole suite.** A change to any default rule changes
   extraction behavior for every caller — `scripts/gates.sh`, not a
   slice.

## Outcome

`search(..., rules=DEFAULT)` picks the law up for every caller, and the
shipped bank's claim holds: every machine-admitted law replayed its
certificate and paid on a measured fixture before it shipped. The
promotion table and per-law evidence live upstream in
[`project/retros/promoted-laws.md`](https://github.com/handl-lang/graphopt/blob/main/project/retros/promoted-laws.md).

## See also

- [Watch a law get minted](../tutorials/mint-a-law) — the tutorial version of the loop.
- [The gauntlet](../explanation/the-gauntlet) — why each stage exists.
- [The discovery scorecard](../reference/scorecard) — the shipped bank's public record.

# Watch a law get minted

<!-- provenance: graphopt:docs/mechanism.md@eea02f6 + mlopt_discovery.pipeline -->

The shipped bank is not fixed. `mlopt_discovery` runs a closed loop
over a corpus of real exported models:

```text
census → propose → verify → measure → rank → emit → admit
```

A candidate only ships after surviving [the gauntlet](../explanation/the-gauntlet)
— numeric oracle, typed-pay gate, adversarial sweep of the guard's
false region, replayable certificate.

## 1. Run the discovery loop

CPU-only, bounded to a few minutes:

```sh
.venv/bin/python -m mlopt_discovery.pipeline                     # the full loop
.venv/bin/python -m mlopt_discovery.pipeline --holdout select_mul  # rediscover a shipped law
.venv/bin/python -m mlopt_discovery.pipeline --json /tmp/pipeline.json
```

The `--holdout` arm removes a shipped law and asks the loop to
re-derive it — a falsifiable test of the "laws are emergent" claim
(the pipeline re-ranks the winner #1 and marks it SHIP, every run).

## 2. Emit an admission patch

```sh
.venv/bin/python -m mlopt_discovery.pipeline \
    --holdout softmax_fold \
    --emit-admission recognize:softmax --out /tmp/admission
```

`--emit-admission` writes the `R(...)` block plus generated tests as a
reviewable `git apply` patch — the emitted law is bytecode-identical
to a hand-written one in the held-out check.

## 3. Verify a law standalone

```sh
.venv/bin/python -m mlopt_discovery.verifier
.venv/bin/python -m mlopt_discovery.verifier --json /tmp/law.json
```

`verify_law(lhs, rhs, rules, ...) -> LawResult` puts both sides in a
fresh small e-graph, saturates under the known laws, and asks whether
they land in one e-class. On a merge it emits a positional certificate
and replays it on real terms — a witness is a genuine derivation,
never a trusted stub.

## 4. The minting stack

Laws over a fixed vocabulary are layer zero; ADR-0007 mints the
vocabulary itself:

- `bench/drivers/mint/atom_law_loop.py` — per-atom law synthesis
  (measured bindings → `auto_cond`/`dspec` → `verify_law`).
- `bench/drivers/mint/atom_synth_loop.py` — the composed
  need → handler → laws → admission → pay-probe loop.
- `bench/drivers/probes/conv_fold_premise_pay_probe.py` — the flagship:
  mint → transport → truth → pay → gauntlet (~13 s).

The shipped result: **36 machine-admitted laws** in `DEFAULT` — the
13 promoted laws plus 23 minted `prim:synop_*`/`synop2_*` folds — on a
**59-row** `MACHINE_SHIPPED` ledger once each fold's `.expand`
definitional premise is counted
(`mlopt_core.laws.provenance.MACHINE_SHIPPED`).  `add_unsq_pad_r` is
the first promotion whose paying site is a
strictly-real intake workload; `conv_bn_fold` before it is a depth-4
minted composite replayed at import and verified bitwise through an
independent runtime.

## Next

→ [Run the scorecard](./run-the-scorecard)

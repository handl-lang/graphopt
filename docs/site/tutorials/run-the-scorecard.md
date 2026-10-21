# Run the scorecard

<!-- provenance: graphopt:bench/discovery/run.py + docs/scorecard.md@eea02f6 -->

The **discovery scorecard** is the engine's public test of the
"found, not encoded" claim: each row is one named optimization from
the literature or a real compiler (a TOML case in
`bench/discovery/cases/`), and the suite asks whether the engine can
derive it, certify it, and *beat the registered floor* for that case.

## 1. Run it

```sh
.venv/bin/python -m bench.discovery.run --fast     # symbolic dag_cost — seconds; the gate lane
.venv/bin/python -m bench.discovery.run --full     # measured wall + verify — the claim lane
.venv/bin/python -m bench.discovery.run --fast --cases conv_bn_fold   # one case
```

It writes `docs/scorecard.md` + `docs/scorecard.json` in the repo
(commit only when you mean to refresh the published table — the
rendered copy lives at [reference/research/scorecard](../reference/scorecard)).

## 2. Read a row

| column | meaning |
|---|---|
| `found` | the case's expect spec held — the signal was observed and the extraction contract met |
| `cert` | the proof-carrying certificate replays standalone (`verify_certificate`) |
| `cost (axis)` | `dag_cost` before → after on the case's axis (`launch` / `exec` / `param_bytes` / …) |
| `best baseline` | the registered floor for the case (eager / Inductor / known-good impl) |
| `beat` | the certified form is *strictly below* that floor — a tie is a rediscovery, not a find |

Example row (`blelloch_scan`, measured lane): `found=yes cert=yes`,
`759 → 752 (−0.9%)` on the launch axis, best baseline `eager unrolled
759`, `beat=yes` — the engine re-derived Blelloch's O(log T) scan from
generic algebraic laws and delivered a form strictly cheaper than the
unrolled floor.

Honest losses stay in the table: a `beat=no` row (e.g.
`retnet_chunkwise` vs the paper's chunkwise reference) is a
rediscovery at parity or a loss, reported as measured.

## 3. Where the numbers come from

- `bench/discovery/cases/` — one TOML per named optimization (the
  corpus: 30+ cases incl. the `zoo_*` structural cases).
- `bench/discovery/refs/` — the registered hand-built floors each
  case measures against.
- `bench/limits/` — the baseline roster behind
  [the limits table](../reference/limits).
- `bench/players/` — the player-comparison runner behind
  [the leaderboard](../reference/leaderboard).

## What you just saw

Install → `search()` → mint → scorecard is the whole loop: syntax in,
certified program out, and a law bank that grows under measured play —
the [mechanism](../explanation/mechanism) page is the
conceptual account, and the [three costs](../explanation/three-costs)
page explains why `found` needs all of ranking, measurement, and proof.

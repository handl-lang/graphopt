# Retro — marking the real minutes-scale tail `slow`

Date: 2026-10-15
Context: `AGENTS.md` advertises `-m "not slow"` as the iteration
loop, and justified it with a slice measurement (the
`test_torch_integration` + `test_om_decayed_cross` slice:
370.8 s → 42.6 s).  That is true *for that slice* and misleading for
the suite: only **six** tests carried `@pytest.mark.slow`, and the
suite's dominant tail was **unmarked**.  A serial run stalled at ~36%
inside the discovery block; `-n 2` overlapped the stall and reached
98%.  So `-m "not slow"` did not actually give a fast loop.  This
retro records the measurement and the marking.

## The tail, measured (this host, 2026-10-15)

| target | cost | how measured |
|---|---|---|
| `test_discovery_open_learned.py` (module) | **>900 s** (killed at the box) | whole module; the module-scoped `measured` fixture dominates |
| `test_discovery_open_prior.py` (module) | **>712 s** (11:52, terminated) | `TestPreset` passed in seconds, then the `measured` fixture stalled past the 712-s mark |
| `test_square_rsqrt_laws.py::test_every_miss_spelling_reaches_rms_norm` | **>9:49**, still climbing (~1.2 GB RSS) | killed at the 1000-s box |
| `test_square_rsqrt_laws.py::test_canonical_rsqrt_spellings_fold_with_their_bridges` | **>7 min** (420-s box) | killed at the box |

The two discovery fixtures are the same shape: a 36-page × 4-arm
episode loop over `proposal.evaluate` (the learned module also trains
one epoch from the cold start).  The two `test_square_rsqrt_laws`
saturations run `EGraph.run(..., max_nodes=60_000)` over *every*
recorded-miss spelling of the manual RMSNorm — the canon laws expand
the graph to the node cap, which is the minutes.

## Measured **fast**, so deliberately not marked

- `test_discovery_meta_prior.py` — its `measured` fixture is 0.6 s
  (a held-out eval, no training).
- `test_discovery_open_game.py` — the whole module is 8.4 s (max
  single test 2.0 s).
- `test_square_rsqrt_laws.py::test_real_export_spellings_fold_and_verify`
  — 2.8 s, and `::test_bridges_fire_in_default_pipeline` — 0.9 s.
  Both use `max_nodes=60_000`/`max_enodes=100_000` but only on the
  canonical / real-export spellings, which fold before the graph
  explodes.
- the other module-scoped-fixture files (`test_regime`,
  `test_regime_edges`, `test_discovery_coherence*`,
  `test_discovery_impact`, `test_discovery_intake`,
  `test_contraction_policy_artifact`) — 229 tests in 21.7 s.
- the tens-of-seconds torch tests (`test_om_decayed_kway_deliver`
  20.9/15.8 s, `test_om_mha` 12.1 s, `test_omd_lower` 11.7 s) — kept
  unmarked; they are not minutes-scale.

## The fixture caveat (the thing to get right)

A **module-scoped fixture is not skipped by marking one test that uses
it.**  So the discovery markers go on the *classes* that request
`measured` (`TestHeldOutYield`, `TestOrderingSensitivity`,
`TestRanking` in the learned module; `TestHeldOutYield`, `TestRanking`
in the prior module) — every test that touches the fixture.  With all
of them deselected pytest never instantiates the fixture, because
fixtures are lazy.

**Verified.**  `-m "not slow"` over the five affected files selects
**40 of 64** and runs in **14.1 s**, and no `setup` entry for either
`measured` fixture appears in `--durations`.  (The same five files
without the marker run the learned module alone past 900 s.)

## Result

`slow` now covers **30** tests (the six `torch.compile` ones + 24
new).  `AGENTS.md`'s `-m "not slow"` paragraph now names what it
skips, what it still costs (the tens-of-seconds tests above), and the
fixture caveat.

**Note for the next reader:** these are minutes-scale *by design*, not
by accident — an episode loop over a real evaluation, and an e-graph
saturation run to the node cap.  They are regression pins for the
project's central claims (the learned arm beats random on held-out
pages; the miss spellings fold under `ALL_RULES`).  Do not shrink them
to make the loop faster; the marker is the right lever.

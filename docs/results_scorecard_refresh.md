# Discovery scorecard refresh — 2026-10-21

**Runner:** `.venv/bin/python -m bench.discovery.run --full` (~40 s wall,
cpu) · **Span:** `e104d90` → `eea02f6` (dirty tree; parallel-session WIP
in `packages/` + `tests/` unstaged).

## What moved

| case | before | after | note |
|---|---|---|---|
| `sdpa_decomposed` | — | found/cert/verified, tie 32,772 → 32,772 (launch), measured 0.041 ms vs eager-spelled 0.040 ms, beat=no | **new case row** — shipped in `09c5b59` (`sdpa_expand` family). Decompose fused SDPA → migrate the (T,T) score scale into a compile-time weight mul → refold. Honest parity row: the point is that the fused op is now *reachable* by the e-graph, not that it beats the kernel. Opt-in via `default_attention`. |
| `conv_bn_fold` | measured beat=yes (0.0111 vs 0.0112 ms) | measured beat=**no** (0.0099 vs 0.0095 ms) | noise — the extracted program *is* the hand-folded conv; the strict-< verdict jitters at the µs boundary run to run (re-run reproduced: 0.0106 vs 0.0105). Symbolic verdict unchanged (tie). |
| `flash_attention` | measured beat=no (0.0358 vs 0.0342 ms) | measured beat=**yes** (0.0286 vs 0.0289 ms) | same class — extraction calls `F.scaled_dot_product_attention` itself, so the "beat" is measurement jitter around parity (re-run: 0.0318 vs 0.0321). |

**Nothing else moved.** All 30 pre-existing rows keep identical
symbolic `cost before → after`, `found`, `cert`, `verified`, and
`beat_best`. Search/extraction details (`extracted_ops`, `cert_steps`,
`n_enodes`) are byte-identical per case; the only diffs are the
measured `median_s` columns.

## Honest negatives — unchanged, still honest

- `cumsum_scan`: 1,115,257 → 1,402,358 (**+25.7%** exec) — the
  derivation costs more than the baseline.
- `select_mul`: 15 → 15 (+0.0%) — rediscovery, not a find.
- `zoo_lora_param`: 400 → 400 (+0.0% param_bytes) — merge can't shrink
  the low-rank storage.
- `sdpa_decomposed` joins as a fourth honest-parity row (found, tie,
  beat=no) — recorded, not "fixed".

## Attribution

The only committed change that touched a case row is `09c5b59`
(`feat(laws): sdpa_expand`), which added `sdpa_decomposed.toml` and the
`ATTENTION` law pack. The dtype-contract fix (`2dec296`), generic-graph
`CudaGraphRunner` (`dacad50`), and the gamekit/kernel work changed no
symbolic cost or verdict on this suite.

<!-- provenance: graphopt:docs/limits_table.md@eea02f6 — generated upstream; do not hand-edit -->

# Limits table — the delivered form vs the registered baselines

Every registered opponent in `bench/limits/` crossed with the named-optimization corpus (`bench/discovery/cases/`).  **ratio** is `baseline ÷ mlopt` — the results-doc convention: >1× means the delivered form is ahead, ~1× is parity (a rediscovery of the known impl), <1× is an honest loss.  *caveat* carries what the number hides: measured-only arms, scratch-dep skips, verify failures.

- lane: `fast` (symbolic dag_cost)
- date: 2026-10-20T00:10:29+00:00 · git: `e7dc703` (dirty) · torch 2.14.0+cu130 on cpu

| case | category | baseline | kind | mlopt | baseline | ratio | verdict | caveat |
|---|---|---|---|---|---|---|---|---|
| `blelloch_scan` | speed | eager | eager | 752 | 759 | 1.01× | parity | no compilation — the unoptimized floor |
| `blelloch_scan` | speed | hand_carrier | carrier | 752 | 759 | 1.01× | parity | same extracted term — executor routing is the delta |
| `blelloch_scan` | speed | paper_impl | impl | 752 | 1,494 | 1.99× | WIN | from bench.discovery.refs — the hand-built floor |
| `blelloch_scan` | speed | torch_compile | compile | 752 | — | — | — | measured-only — no symbolic form |
| `conv_bn_fold` | speed | eager | eager | 27,649 | 29,186 | 1.06× | WIN | no compilation — the unoptimized floor |
| `conv_bn_fold` | speed | hand_carrier | carrier | 27,649 | — | — | — | no carrier executor claims the extraction — generic lowering would duplicate the mlopt arm |
| `conv_bn_fold` | speed | paper_impl | impl | 27,649 | 27,649 | 1.00× | parity | from bench.discovery.refs — the hand-built floor |
| `conv_bn_fold` | speed | torch_compile | compile | 27,649 | — | — | — | measured-only — no symbolic form |
| `flash_attention` | memory | eager | eager | 32,772 | 33,544 | 1.02× | WIN | no compilation — the unoptimized floor |
| `flash_attention` | memory | hand_carrier | carrier | 32,772 | — | — | — | no carrier executor claims the extraction — generic lowering would duplicate the mlopt arm |
| `flash_attention` | memory | paper_impl | impl | 32,772 | 32,772 | 1.00× | parity | from bench.discovery.refs — the hand-built floor |
| `flash_attention` | memory | torch_compile | compile | 32,772 | — | — | — | measured-only — no symbolic form |
| `mul_unsq_nat` | speed | eager | eager | 50 | 75 | 1.50× | WIN | no compilation — the unoptimized floor |
| `mul_unsq_nat` | speed | hand_carrier | carrier | 50 | — | — | — | no carrier executor claims the extraction — generic lowering would duplicate the mlopt arm |
| `mul_unsq_nat` | speed | paper_impl | impl | 50 | 50 | 1.00× | parity | from bench.discovery.refs — the hand-built floor |
| `mul_unsq_nat` | speed | torch_compile | compile | 50 | — | — | — | measured-only — no symbolic form |
| `retnet_chunkwise` | memory | eager | eager | 761,862 | 761,863 | 1.00× | parity | no compilation — the unoptimized floor |
| `retnet_chunkwise` | memory | fla | kernel | 761,862 | — | — | — | fla not installed (scratch dep) |
| `retnet_chunkwise` | memory | hand_carrier | carrier | 761,862 | — | — | — | no carrier executor claims the extraction — generic lowering would duplicate the mlopt arm |
| `retnet_chunkwise` | memory | paper_impl | impl | 761,862 | 491,656 | 0.65× | loss | from bench.discovery.refs — the hand-built floor |
| `retnet_chunkwise` | memory | torch_compile | compile | 761,862 | — | — | — | measured-only — no symbolic form |
| `swiglu_mlp` | speed | eager | eager | 4,355 | 4,356 | 1.00× | parity | no compilation — the unoptimized floor |
| `swiglu_mlp` | speed | hand_carrier | carrier | 4,355 | — | — | — | no carrier executor claims the extraction — generic lowering would duplicate the mlopt arm |
| `swiglu_mlp` | speed | paper_impl | impl | 4,355 | 4,485 | 1.03× | WIN | from bench.discovery.refs — the hand-built floor |
| `swiglu_mlp` | speed | torch_compile | compile | 4,355 | — | — | — | measured-only — no symbolic form |
| `weight_chain_fold` | speed | eager | eager | 2,049 | 4,098 | 2.00× | WIN | no compilation — the unoptimized floor |
| `weight_chain_fold` | speed | hand_carrier | carrier | 2,049 | — | — | — | no carrier executor claims the extraction — generic lowering would duplicate the mlopt arm |
| `weight_chain_fold` | speed | opt_einsum | order | 2,049 | — | — | — | ordering, not a graph — measured-only |
| `weight_chain_fold` | speed | paper_impl | impl | 2,049 | 2,049 | 1.00× | parity | from bench.discovery.refs — the hand-built floor |
| `weight_chain_fold` | speed | torch_compile | compile | 2,049 | — | — | — | measured-only — no symbolic form |

## The registered roster

- `eager` (eager) — PyTorch eager — the spelled module, as written · all cases · record: launch-bound crossover: carriers lose to eager below ~T=512 (0.51×, docs/results.md)
- `fla` (kernel) — FLA chunk_retention — the Triton-kernel floor · cases: retnet_chunkwise · record: ~7–11× ahead of the fused carrier — kernel-level: fp64 GEMMs + a materialized score tensor on our side (bench/drivers/probes/retnet_baseline_probe.py)
- `hand_carrier` (carrier) — carrier term, generic lowering (no batched schedule) · all cases · record: 0.51× vs eager below the T=512 crossover — the carrier without its level-batched schedule (docs/results.md)
- `opt_einsum` (order) — opt_einsum greedy — the contraction-ordering baseline · cases: weight_chain_fold · record: oe-greedy is the contraction board's floor — the engine's ordering players tie its cost under the shared pair model (bench/players leaderboard)
- `paper_impl` (impl) — known-good impl — the case's registered ref · all cases · record: paper chunkwise ties the fused carrier (2.08 vs 2.12 ms T=2048; faster under graph replay) — bench/drivers/probes/retnet_baseline_probe.py
- `torch_compile` (compile) — torch.compile — the Inductor pipeline · all cases · record: pointwise fusion 0.82–0.88× loss (Inductor wins); weight chain 3.96× mlopt (no fold without freezing) — docs/results.md

# Real-corpus funnel — why strictly-real yield was ~0, and the one fix that moved it

Date: 2026-10-21. Driver: `bench/drivers/screen/real_funnel.py`
(probe; results JSON `/tmp/real_funnel.json`). Context:
`project/retros/real-corpus-yield.md` measured `shippable: 0` on the
real half of the intake corpus and named the residual signal ("every
paying equality on real modules is conditional") without making the
funnel exact. This document is the exact funnel, plus the measured
answer to its named lever — the `auto_cond → run_gauntlet` retry —
run on a gauntlet corpus that actually contains the real modules.

**Headline: real-corpus yield moved 0 → 1.** The armed auto-cond
gauntlet admits `mixed:add_unsqueeze_r_id` — minted the declarative
`_COND_UNSQ_PAD_R` guard — **usable** on a corpus carrying the real
intake probe cases, with its paying site on `intake:RelativePositionBias`
(a strictly-real workload: `x + bias.unsqueeze(0)`, the T5/Swin
relative-position-bias spelling). The admitted law ships as
`add_unsq_pad_r` (`mlopt_core.laws.tensor`, `MACHINE_SHIPPED` 12→13).
On the case it fires once, the extracted winner drops dag-cost
52,296 → 34,873 (−33.3%) and the lowered module verifies fp64.

## Method

One intake ingestion (seed 0, `--skip-pipeline`) wrote
`/tmp/full_intake_corpus.json` + `intake_tensors.pt`: 254 exported
(206 ingested, 48 census-only), 2 export rejections (`nn.CTCLoss`,
`nn.GaussianNLLLoss`). The driver splits cases by `PURPOSE_BUILT`
(46 spellings) and ingests the held-out `mlopt_discovery.zoo`
registry through the same boundary. Four pipeline arms run
`workload_gen._run_pipeline` — the intake driver's own census →
propose → measure → rank mirror — with `pipeline._probe` wrapped so
every `paid` count is attributable to a named case:

| arm | corpus terms | probe cases | proposals | shipped |
|---|---|---|---|---|
| baseline (bench+models) | 110 | 44 | 54 | 0 |
| + real intake | 318 | 204 | 79 | **0** |
| + purpose-built | 156 | 90 | 61 | 13 |
| + zoo (held out) | 133→ | 66→ | 72 | 0 |

(zoo arm measured 22–35 ingested of 23–36 exported across seeds; the
funnel shape is seed-stable.)

## The funnel — first failed gate per candidate

Each candidate's death stage is its *first* failing gate in
`Evidence.shippable` order (truth → novelty → fires → typed → pay →
verify → cert → closure); `truth:*` sub-splits the oracle's verdict.

| first-failed gate | baseline | +real | +purpose | +zoo |
|---|---|---|---|---|
| truth: conditional | 20 | **42** | 23 | 36 |
| truth: false (numeric oracle) | 8 | 11 | 8 | 9 |
| truth: no real match | 11 | 11 | 0 | 11 |
| novelty (relation ≠ new) | 6 | 6 | 6 | 7 |
| fires (no probe firing) | 8 | 8 | 0 | 8 |
| ill-typed fires | 0 | 0 | 0 | 0 |
| pay (fires, never cheaper) | 1 | 1 | 10 | 1 |
| verify / cert / closure | 0 | 0 | 1 | 0 |
| **shipped** | **0** | **0** | **13** | **0** |

Three readings, now exact:

1. **Proposals generate on real structure** — the funnel's top is
   not the gap: 25 new proposals enter from the real census alone
   (79 vs 54), and 42 carry real matches the matcher can see. The
   plan-0025 "grammar dead space" hypothesis is *not* what eats the
   real corpus — generation is healthy.
2. **The loss concentrates at `truth: conditional`** — 42/79 (53%)
   of the real arm's candidates die as conditional equalities, vs
   23/61 purpose and 20/54 baseline. The *conditional class itself
   is larger on real code*: real modules genuinely spell the
   non-trivial corners (SwiGLU chunks, ALiBi unsqueezes, Conformer
   transposes) where the bare law is false.
3. **Nothing reaches the pay gate on real code unsound-but-true.** The
   single `pay`-stage death and zero verify/cert/closure deaths say
   the tail gates are not the bottleneck — the funnel is decided
   upstream, at truth.

## Attacking the blocker: the armed auto-cond retry on real terms

The truth stage's refusal is not the end of the story — it is a
*request* for the auto-cond path (`object_synthesis.auto_cond_object`
→ `run_gauntlet(auto_cond=True)`), which the intake pipeline never
invokes: `Evidence.shippable` has no guarded-retry leg, and the
recorded auto-cond measurement (`project/retros/auto-cond.md`) ran
`generator_pools`' static 66-candidate pool, which the census-driven
`mixed:add_unsqueeze_r_id` never entered — a proposal-provenance gap
on top of a wiring gap.

Retrying all **38 truth-refused candidates with real matches** on the
real arm (`synth_limit=360`, measured domain = synth windows ∪ real
matches):

| stage | count | what happened |
|---|---|---|
| truth-refused, real matches | 38 | the conditional+false+unproven set |
| guard minted | 31 | a declarable ≤3-clause cover exists |
| refused (no equal site / no declarable cover) | 7 | e.g. `mul_select_l_id` — 0 equal sites in domain |
| **guarded region inhabited on real terms** | **4** | `add_unsqueeze_r_id` (1 accepted/1 equal), `mul_unsqueeze_l_id` (1/1), `sub_unsqueeze_l_id` (3/3), `mul_transpose_l_w` (6/6) |
| region *empty* on real terms | 27 | the minted guard declines every real site — real code lives in the unequal corner |

Then the armed gauntlet (corpus = bench+models+real terms, probe =
models + the 160 probe-eligible real cases) on the four inhabited
candidates:

| candidate | minted guard | gauntlet verdict |
|---|---|---|
| `add_unsqueeze_r_id` | `bcast-eq(unsq-out U,V, U,V) ∧ bcast-into(U, unsq-out U)` | **usable** — fires 1, typed, paid 1 on `intake:RelativePositionBias`, verify-clean |
| `mul_unsqueeze_l_id` | same strip pair | refused at `novelty` — alpha-equal to shipped `mul_unsq_pad_l` |
| `sub_unsqueeze_l_id` | same strip pair | refused at `novelty` — alpha-equal to shipped `sub_unsq_pad_l` |
| `mul_transpose_l_w` | `bcast-eq(transpose-out …) ∧ (axes-noop(V) ∨ rank(V)≤1)` | refused at `typed-pay` — fires 6 real sites, `paid=0` (a wrap moves the transpose; nothing gets cheaper) |

The two `novelty` refusals are a *soundness check*, not a failure:
the auto-cond path re-derived the already-shipped pad strips
verbatim. The decisive datum is the middle row of the previous
table — **27/31 minted guards accept zero real sites**. That is the
exactly-measured version of the retro's "real code inhabits the false
corner": for the dominant conditional class the guarded region the
law needs is genuinely unoccupied on the corpus. No machinery fix
mints yield where no equal site exists; the residual loss is real
structure, not a bank gap.

Two honest near-misses worth naming:

- `mul_transpose_l_id` looked like the biggest live candidate —
  `paid=4` on real modules (`AxialAttention`, `ConformerBlock`,
  `PostNormBlock`, `PreNormBlock`, `nn.MultiheadAttention`) — but its
  minted `axes-noop` guard declines *all six* real sites. The bare
  rule "paid" by stripping transposes that are not no-ops: the
  measured pay was an unsound-rewrite artifact, and the honest
  guarded object is inert. The funnel's `paid` column alone would
  have overstated the residual signal.
- `mul_select_l_id` (`paid=3` on `DiagonalSSM`/`HybridBlock`/
  `TwoLayerHybrid`) is `num_true=False` — auto-cond measured 0 equal
  sites in 279; it refuses cleanly.

## What shipped

`add_unsq_pad_r` — `add(u, unsqueeze(v, d)) → add(u, v)` under the
already-shipped `_COND_UNSQ_PAD_R` — is the add sibling of
`mul_unsq_pad_r`, transcribed into `SIMPLIFICATION_RULES` /
`ALL_RULES` and `MACHINE_SHIPPED` (now 13). Verified on the real
case: `intake:RelativePositionBias` fires once, extracts the stripped
term, and verifies (`paid`, `verified=pass`, −33.3% dag-cost). Pins:
`tests/test_cond_laws.py::test_promoted_add_unsq_pad_right` (merge on
the RelativePositionBias spelling + both decline corners),
serialization/count pins in `test_laws_structure.py` /
`test_law_serialize.py` / `test_law_provenance.py`.

After the promotion the proposal `mixed:add_unsqueeze_r_id` reads
`duplicate` at novelty — the pipeline's `shippable` counter correctly
returns to 0; the yield is in the library.

## The verdict on "why ~0"

Not proposal generation (25 new proposals enter from real census
tuples), not the pay gate, not lowering verify — **`truth:
conditional` is the funnel's pinch, and the guarded regions it needs
are measured empty on real code for 27 of 31 mintable guards.** The
retro's guess is confirmed and sharpened: the next-yield lever is not
more spellings and not more guard vocabulary — the one real
inhabitant found by *widening the candidate pool to census-driven
proposals and running the admission retry on a corpus that contains
the real sites*. The remaining real-corpus upside lives in the
families auto-cond cannot yet express (`*_w` wrap forms needing
`w`-broadcast predicates; select/getitem regions with no declarable
≤3-clause separator) and in candidates whose guarded regions are
occupied but non-paying (`mul_transpose_l_w`).

## Caveats

- Funnel measured on the pre-promotion bank (85 rules); post-ship the
  candidate reads `duplicate`, as intended.
- `paid` is the executor-aware dag-cost model, not wall-clock — same
  caveat as every yield retro; the −33% figure is op-priced.
- Auto-cond guards are exact on the *measured* domain (synth windows
  + real matches); a counterexample outside both windows stands until
  a wider sweep reaches it — the standing caveat of every guarded
  law.
- Seeded draws: `torch.manual_seed(0)` per measurement; the
  `intake:SelectiveSSM`-class sites whose verdicts are draw-sensitive
  are classed `unstable` and must be declined (the auto-cond
  stability rule, unchanged).

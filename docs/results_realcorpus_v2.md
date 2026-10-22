# Real-corpus funnel v2 — the zoo grew, the compose arm landed; yield is still 1

Date: 2026-10-21 (evening run, measured under `2aa6ba5` + foreign
WIP in tree; sibling lands `b534bce`/`9121617` — scale-ladder probe,
bilinear coverage — arrived mid-run and do not touch the funnel
path).
Driver: `bench/drivers/screen/real_funnel.py`, same protocol as
`docs/results_realcorpus.md` (v1, measured at `737225c`): one
`--skip-pipeline` intake side-file, four pipeline arms, then the
`auto_cond → run_gauntlet` retry on truth-refused candidates.
Results JSON `/tmp/real_funnel_v2.json`; intake
`/tmp/full_intake_corpus.json` + `intake_tensors.pt`.

**Headline: yield did not move past 1.** The funnel's `shipped`
counter *did* move — 0→3 on the real arm — but every shipped row
carries `derivable=True`: they are theorems of the shipped bank
re-surfacing under spellings the structural `_relation` alpha-key
calls `new`, not new yield. The auto-cond gauntlet admits nothing
new (3 inhabited candidates, all refused), and the overnight
`--compose` arm on zoo-real admits 0 across both its pre- and
post-seed-gate runs. The only real-corpus law in the bank is still
`add_unsq_pad_r`.

## What changed since v1

- `f4ba235` — the held-out zoo grew 36→53 exported (52 ingested, 1
  census-only, 0 rejections), probe-eligible 52.
- `04201f2` / `1000ffd` — overnight `--compose` (syn_* composition on
  zoo-real) + the seed-eval gate that drops composites whose RHS
  cannot denote under a bound env before they spend a slot.
- `6eb949d` — attr-relaxed same-view families: `f(g,g)` census tuples
  now emit the two-view treatment (`wl`/`wr`/`id` with independent
  `A_*`/`B_*` attr namespaces), growing every arm's proposal pool.
- Intake itself is unchanged: 254 exported, 206 ingested, 48
  census-only, same 2 export rejections (`nn.CTCLoss`,
  `nn.GaussianNLLLoss`); 208 real + 46 purpose-built, 160 probe-eligible
  real.

## Arms — v1 → v2

| arm | corpus terms | probe cases | proposals (v1→v2) | shipped (v1→v2) |
|---|---|---|---|---|
| baseline (bench+models) | 110 | 44 | 54 → **57** | 0 → **2** |
| + real intake | 318 | 204 | 79 → **94** | 0 → **3** |
| + purpose-built | 156 | 90 | 61 → **73** | 13 → **15** |
| + zoo (held out) | 133 → **163** | 66 → **96** | 72 → **120** | 0 → **2** |

## The funnel — first failed gate per candidate (v2)

| first-failed gate | baseline | +real | +purpose | +zoo |
|---|---|---|---|---|
| truth: conditional | 20 | **49** | 29 | **74** |
| truth: false (numeric oracle) | 9 | 16 | 12 | 16 |
| truth: no real match | 11 | 11 | 0 | 10 |
| novelty (relation ≠ new) | 6 | 6 | 6 | 7 |
| fires (no probe firing) | 8 | 8 | 0 | 7 |
| ill-typed fires | 0 | 0 | 0 | 1 |
| pay (fires, never cheaper) | 1 | 1 | 10 | 3 |
| verify / cert / closure | 0 | 0 | 1 | 0 |
| **shipped** | **2** | **3** | **15** | **2** |
| **proposals** | 57 | 94 | 73 | 120 |

(v1 row for comparison: conditional 20/42/23/36, false 8/11/8/9,
no-match 11/11/0/11, novelty 6/6/6/7, fires 8/8/0/8, ill-typed 0,
pay 1/1/10/1, verify-cert-closure 0/0/1/0, shipped 0/0/13/0,
proposals 54/79/61/72.)

## What moved — and why it is not yield

The three real-arm `shipped` rows decompose as follows:

| candidate | fires / paid | pay sites | derivable | reading |
|---|---|---|---|---|
| `mixed:add_unsqueeze_r_id` | 1 / 1 | `intake:RelativePositionBias` | **True** | the shipped `add_unsq_pad_r`'s *commuted* spelling — `add(V, unsq(U,d)) → add(U,V)` vs `add(u, unsq(v,d)) → add(u,v)`. Alpha-key doesn't quotient by commutativity, so `relation=new`; `verify_law` derives it from the bank, so truth is waived and it ships. Same law, second sighting. |
| `mixed:mul_select_select_wl` | 24 / 5 | `DiagDenseSSM`, `DiagonalSSM`, `HybridBlock`, `SelectiveSSM`, `TwoLayerHybrid` — all baseline | **True** | new proposal (independent attr namespaces on `f(g,g)`); every paying instance has `A_sel == B_sel`, i.e. it is `select_mul` re-spelled — the bank derives the measured instance. |
| `mixed:mul_select_select_wr` | 24 / 5 | same five baseline cases | **True** | mirror of the above. |

Two honest consequences:

1. **No shipped row is new real-corpus yield.** The two genuinely-new
   proposals pay only on baseline model cases and are bank theorems
   on the measured instance; the one paying a real site is the law
   already shipped at v1. Novel, non-derivable, real-paying
   candidates through the pipeline: **0** — same as v1.
2. **`shipped` is no longer a clean novel-yield counter.** It counts
   derivable-but-structurally-new variants (commuted spellings,
   attr-bound instances). v1's post-ship check expected
   `add_unsqueeze_r_id` to read `duplicate`; it now reads
   `new`+`derivable`+`shippable`. The gap is `_relation`'s structural
   key vs the bank's actual equational content — worth a sieve pass
   if `shipped` is ever consumed as a headline metric.

The pinch itself is unchanged in kind and grew in volume:
`truth:conditional` is 49/94 (52%) on the real arm (v1: 42/79, 53%)
and **74/120 (62%) on the grown zoo arm** (v1: 36/72, 50%). More real
code still means more conditional equalities, not more unconditional
ones. The `ill-typed` bucket fires for the first time
(`mixed:mul_reshape_r_id`, zoo arm, 1).

## The auto-cond retry on real terms

`truth`-refused candidates with real matches: **50** (v1: 38) —
the attr-relaxed families minted ~15 new view-pair proposals.
Results:

| stage | v2 | v1 |
|---|---|---|
| truth-refused, real matches | 50 | 38 |
| guard minted (≤3-clause cover) | 30 | 31 |
| refused (no equal site / no declarable cover) | 20 | 7 |
| **guarded region inhabited on real terms** | **3** | **4** |
| guarded region inhabited on **zoo** terms | 5 | — (zoo not swept) |

The three real-inhabited candidates are the same trio as v1 minus
the now-derivable `add_unsqueeze_r_id`:

| candidate | real region | gauntlet verdict |
|---|---|---|
| `mixed:mul_unsqueeze_l_id` | 1 accepted / 1 equal | refused `novelty` — `relation=duplicate` (alpha-equal to shipped `mul_unsq_pad_l`) |
| `mixed:sub_unsqueeze_l_id` | 3 accepted / 3 equal | refused `novelty` — `relation=duplicate` (shipped `sub_unsq_pad_l`) |
| `mixed:mul_transpose_l_w` | 6 accepted / 6 equal | refused `typed-pay` — fires 6 real sites, `paid=0` (a wrap moves the transpose; nothing gets cheaper) |

**Admitted: none.** The armed gauntlet mints yield only where the
bank doesn't already reach — and the two clean real regions it found
are the shipped pad strips re-derived verbatim (a soundness check,
as v1 noted), plus the same honest inert wrap.

## The new paying family the DSL cannot cover

The attr-relaxed families surfaced one real-corpus payer class v1
didn't have: `sub_unsqueeze_unsqueeze_{wl,wr}` fire 3 and **pay 3**
each on real intake (`intake:ALiBiAttention`,
`intake:RelativePositionBias`, `intake:SlidingWindowMask` —
sub-over-two-unsqueezes sites the new independent-namespace family
now proposes), `sub_unsqueeze_unsqueeze_id` pays 2. All three die at
`truth:conditional`/`truth:false`, and auto-cond mints **no guard**
for any of them (`found=False`: 63–65 equal vs 129–131 bad measured
sites — no declarable ≤3-clause separator). Same for
`mul_unsqueeze_unsqueeze_{wl,wr}` on the zoo arm (26 fires, 2 paid).
This is the v1 verdict sharpening: the residual loss moved from
"guarded region unoccupied" to "region occupied but the condition
DSL cannot name its boundary".

## The zoo-inhabited regions — the untested admission path

A new datum this run (the v1 sweep didn't score zoo sites): five
minted guards accept *zoo* sites — `mul_unsqueeze_r_id` (3 equal),
`sub_unsqueeze_l_id` (2), `sub_unsqueeze_l_w` (1),
`mul_unsqueeze_l_id` (1), `mul_reshape_r_id` (4 equal / 3 unequal —
dirty). The driver's gauntlet corpus probes only the real-intake
cases, so zoo-inhabited candidates never reach `run_gauntlet`; the
zoo arm's 18 zoo-paying truth-refusals (`sub_unsqueeze_r_id` 4 paid,
`sub_unsqueeze_unsqueeze_{wl,wr}` 4 each, `add_unsqueeze_{l,r}_id` 2
each, …) have no admission path measured. A `GauntletCorpus` whose
probe includes the zoo cases is the concrete next lever.

## The overnight reports — compose arm, zoo-real corpus

`docs/reports/` holds the three v1/v2 ledgers the other session
landed:

| run | git | mint | minted | admitted | notes |
|---|---|---|---|---|---|
| preview | `863825c` | corpus-bind | 65 | 2 — `sub_neg`, `id_mul_lit`, **both known-art** | 107-scheduled/36-ran episodes |
| full | `04201f2` (dirty) | corpus-bind-compose | 318 (244 composites) | **0** | 310 oracle refusals — composites' RHS can't denote under the bound env (rhs_err) |
| v2 | `1000ffd` (dirty) | corpus-bind-compose-viable | 87 (13 composites) | **0** | seed-eval gate dropped non-denoting composites at emit; oracle 79 / pay 7 / cert 1 |

The compose arm's closest approach post-gate:
`syn_mixed:mul_reshape_r_id__mixed:pow_reshape_l_w_6@RMSNormPerHead`
reached `pay` — fires=5, changed=5, **paid=4** on the zoo probe —
then failed `verify_fail=4` (the lowered rewrite doesn't verify on
its own sites). Composition reached the pay gate once and still
produced no admission.

## Verdict

**Real-corpus yield: still 1 (`add_unsq_pad_r`).** Every leg of the
measurement agrees:

- pipeline `shippable` on real terms: 3 rows, all `derivable` bank
  theorems — 0 novel;
- armed auto-cond → gauntlet: 0 admitted (v1: 1 — the now-shipped
  law);
- overnight zoo-real mint (single-step + compose): 0
  machine-original, 0 admitted post-gate;
- zoo arm `shipped`: 2, both the same derivable select-select wraps
  paying baseline cases only.

**The pinch is still `truth: conditional`** — now measured 52% real /
62% zoo — and the residual signal is sharper: the corpus's occupied
equal regions are either already bank theorems or need a guard the
condition DSL can't declare (`*_unsqueeze_unsqueeze_*`, `*_select_*`,
`*_getitem_*` families). The two named levers standing: (a) a
gauntlet corpus that probes the zoo cases — five minted guards have
inhabited zoo regions no admission path has judged; (b) guard
vocabulary for the pointwise-over-two-views region boundary.
Growing the corpus grew the proposal pool and the conditional class
together — it did not mint a second yield.

## Caveats

- Measured at `2aa6ba5` with foreign WIP in tree (`play.py`,
  `player_arena.py`, `test_*` dirty) — the WIP touches the player
  arena, not the funnel path; numbers are the committed pipeline's.
  Sibling commits landed while the arms ran (bilinear domain, scale
  ladder, pin refresh); the process had already imported its
  modules, so the numbers are the `2aa6ba5` tree's.
- `shipped` inflation: the counter credits derivable variants the
  structural `_relation` calls `new`. Reported here decomposed; the
  counter itself is unchanged.
- `derivable` is per measured instance; a derivable instance doesn't
  prove the family's general form (the `mul_select_select` wraps are
  only true where `A_sel == B_sel` — which is `select_mul` again).
- Same seeded draws / `synth_limit=360` / executor-priced `paid`
  caveats as v1.
- The zoo arm is seed-sensitive at the margin (52 ingested this
  seed); the funnel shape — conditional-dominant, zero novel shipped —
  held across v1's seeds and this run.

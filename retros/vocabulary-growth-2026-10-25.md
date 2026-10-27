# Retro: the vocabulary week — originality found, delivery dropped

Date: 2026-10-25
Window: 2026-10-19 18:12 (pitch tag `v0.1.0`) → 2026-10-25
Context: six days after the audited positioning deck, the stack moved
further than the deck can express. This retro is the delta in one
place: what grew, what was *learned*, what measurably broke, and what
that forces next. Every claim names its artifact; anything not named
here is not done.

## 1. The ledger

| what | at `v0.1.0` | now | artifact |
|---|---|---|---|
| machine-shipped ledger names | 12 | **127** | `mlopt_core.laws.provenance` (`len(MACHINE_SHIPPED)`) |
| — promoted classic laws | 12 | 13 | same |
| — minted prim folds | 0 | **57** = 21 `synop_*` + 2 `synop2_*` + 34 `synop3_*` | `docs/reports/results_overnight_prim_depth3.md` |
| — their `.expand` premises | 0 | 57 | same |
| distinct folds minted across runs | — | **71** | `docs/reports/primitives_finds.md` |
| shipped bank | — | **131** `DEFAULT`, 143 `ALL_RULES`, 16 stored-unshipped | `mlopt_core.laws` |
| catalogue the sieve must beat | 242 | **265** | `docs/results_novelty_sieve.md` |

The bar rose with the bank: every new fold had to be underivable from
a catalogue that kept absorbing the previous ones.

## 2. The finding — originality is a function of the vocabulary

`docs/reports/primitives_finds.md` (2026-10-23, `zoo-prim`, 68 modules):

- **21/21** admitted prim folds classify **machine-original**, *because
  the op is new* — "novel by construction, since no catalog contains
  laws about an op that did not exist".
- `--prim-deep`: **48/48** admitted, all machine-original, disjoint
  names. `--prim-bootstrap` seats the shipped folds in the corpus
  e-graph as *vocabulary* (23 fired, +475 enodes) and 2 depth-2 folds
  become reachable; the depth-3 tier then lands 34 more.
- Content: DeltaNet, Gated Linear Attention, Mamba selective scan and
  Griffin's RG-LRU share one "sum of gated products" core; one
  primitive (`bb8c54c201`) binds 14 modules. Found by corpus
  antiunification, not by being told.

**The control, measured:** the codegen domain's 44-case deep run mints
117, admits 97, and classifies **0 machine-original** — composed rules
whose parents are both in the bank are theorems of the bank. So:

> Originality appears exactly where the vocabulary may grow, and
> vanishes where composition inside a fixed bank is all the mint can
> express.

That sentence is now the thesis, and it is two-sided. It also
**re-opens and re-answers Q1**: the honest zero on `zoo-real` (2026-10-21,
53 modules, 87 mints, 0 originals) was not search starvation — it was a
*vocabulary* wall. The lever was never "guarded composite parents".

## 3. The catch — the pay leg was priced, never measured

`docs/reports/prim_delivery.md` (2026-10-24). The `synop*` family ships
in `DEFAULT`; `shipped_primitives.register()` seats a Python closure
that walks the definitional premise per node.

- measured wall clock, fp64 eager, small and large: **13 of 14 cells
  lose or tie** (marginal 1.04–1.95×, e2e 1.07–2.20×); the compiled
  twins agree. The single "delivers" cell routes to a different
  executor — a routing artifact, not a dispatch win.
- mechanism: the spec-walk costs more than the N−1 dispatches it
  removes, at every measured size. There is no fused kernel.
- consequence for the ledger: the `pay` leg for this family was
  **cost-model only**. The referee *priced* the move and the price was
  wrong about the executor.

This is now tracked as tech debt (`TODO` → "synop* real delivery") and
it is the sharpest open problem in the domain: **a referee that prices
can be gamed by its own cost model.**

**Update, same night — the diagnosis was right.** `prim_kernel_probe`
gave six of the pointwise prims a generated `triton.jit` body and timed
it against the shipped binding on CUDA fp32 (1 M elements):
**0.371–0.392× the binding — 2.55–2.70× faster**, every op verified
(`graphopt/docs/reports/prim_kernel.md`). So the fold never was slow; the
*interpreter* was. The shipped path is fp64 on CPU, where Triton cannot
run, which is why the module-level delivery probe sees a loss.

## 4. A second domain, and the referee that measures

`kernelopt/` (private; numbers do not leave it) — `paper/paper.qmd`,
`docs/results_kernel_check.md`, `docs/results_overnight_kernel_v2.md`.

- 26 codegen laws; three-legged referee: certificate, verify-first
  numeric equality in the sink, **and measured wall time** against a
  hand-Triton field baseline.
- 11-case audit, CUDA fp32: verified-equal fused composites
  **1.1–5.0×** over spelled, at-or-near the Triton rows; `kernel_flat`
  control 1.00×; two mm-epilogue rows drifted, ordering holds.
- learned linear player beats the scripted playbook, held-out:
  **0.560 vs 0.200** reward/episode.
- headline: a 145-node carrier the sink could not lower at all → a
  learned schedule fold emits a fused chunked-scan plan at **0.271 ms**
  vs **0.454 ms** incumbent (~0.28 ms FLA, carried retro).
- caveats carried in the paper: realised under **CUDA-graph replay**
  (eager is Python-bound); parity on a launch-bound term.

Reading: the domain whose referee *times* produces delivery wins; the
domain whose referee *prices* produces them on paper only. Same game
layer, same seat, different score.

## 5. The boundary — priors do not move worlds

`kernelopt/docs/results_transfer.md` (Q5 remainder):

- **10 of 12** foreign policy/domain pairs cannot enumerate a single
  legal move; the enumerators are structurally typed to their own state.
- the `open` vocabulary shares **zero** feature names with any other
  domain.
- the one live channel is `a:r:*`, the rule-name hash bucket — same
  hash, different vocabularies; `search → kernel` reaches 96.8 % live
  mass and is semantically scrambled by construction.
- every transplant ≤ **0.39×** of the domain's own prior; native
  kernel greedy **+0.653** vs cold-greedy **−0.129**.

So Q5 is **NULL, sharpened**: what transfers is the *game* (world,
referee, score, seat); what does not is the *vocabulary and
featurization*. A prior is a function of a featurization.

## 6. The player — quit became a channel

`docs/results_stop_isolated.md` (10-20) → `results_stop_gate.md`
(10-24) → `results_stop_verdict.md` + `results_evidence_gate.md`
(10-25) → `results_open_gated_seat.md` (10-25).

| arm | reward | AP | stops | evals/ep |
|---|---|---|---|---|
| incumbent | 0.2768 | 0.2074 | 15 | 15.28 |
| **`vhead:baked` / shipped `open-gated` seat** | **0.2893** | **0.2101** | **26** | 14.56 |
| never-stop control | 0.2850 | 0.2101 | 0 | 16.00 |

- the stop decision is **pure data** and the shipped seat reproduces the
  measured arm digit for digit — gate as data is proven, not asserted.
- bake bars: reward ≥ 0.273 ∧ AP ≥ 0.210; never-stop arms are labelled
  confounds in the same table.
- **verdict-visible evidence transfers**: `vd_live` posts decision-row
  AP **0.631** (0.695 under the baked ordering) — the first column past
  the ~0.40 ceiling every earlier family hit; trained heads reach
  held-out usefulness AP 0.449–0.535.
- round 12: the composition **mostly fails** — fitted verdict-spec gates
  over-stop; one thin honest clear (`gate:vd-headroom:baked`, 0.2857 /
  0.2101, 8 stops).
- new: `mlopt_discovery.learned_propose` — 406 labelled candidate cells,
  14.5 % oracle-pass base rate, but 82–100 % for structured families vs
  **6.5 %** for blind enumeration. Shipped as a **prior, never a verdict**.

Unchanged and re-earned: 212 bad accepts refused, four false twins
rejected, the fabricated "1.98×" caught, one real lowering bug found.

## 7. Architecture — the two decisions the data forced

- **ADR-0157 / plan 0033 — the fragment tower.** Vocabularies become
  `Language` fragments; minting is fragment emission; growth is a
  fragment merge. Phases 0–5 in `project/plans/0033-fragment-tower.md`,
  each shipping standalone value. Phase 0 (canonical fragment dir +
  emit API) has no external dependency and is the generator fix.
  Stated falsifier: *if the flagship stack cannot live on the fragment
  tower, the language-as-data claim is unproven on its hardest corpus.*
- **ADR-0158 — effects resume in flight.** `perform` dispatches to the
  innermost covering handler and its return value *is* the result; the
  old trace semantics become the *unhandled* case; stateful handlers
  ride the calculus' own `loop`/trace; multi-shot `resume` deferred.
- Supporting: ADR-0154 (typecheck semantics on an NBE engine), ADR-0152
  (nominal inductive families), ADR-0153 (grades as arrow indices),
  `reify_ast` (Val→Ast quote half), CI off Dagger onto the bupha runner
  with smoke/verify layers.
- Artifacts: platform paper (`eqsat/paper`, 2026-10-21), tensor-domain
  paper (`paper/`), private codegen-domain paper.


## 8. Evidence table — one claim, one artifact

| claim | artifact |
|---|---|
| minted ops are original by construction | `docs/reports/primitives_finds.md` (21/21, 48/48) |
| composition inside a fixed bank cannot be original | `kernelopt/docs/results_overnight_kernel_v2.md` (97 admitted, 0 original) |
| vocabulary bootstraps depth | `results_overnight_prim_depth3.md`, `results_overnight_prim_bootstrap.md` |
| the new family does not pay in wall clock | `docs/reports/prim_delivery.md` (13/14 lose) |
| a measuring referee produces delivery | `kernelopt/docs/results_kernel_check.md` (1.1–5.0×) |
| priors do not transfer | `kernelopt/docs/results_transfer.md` (10/12, ≤0.39×) |
| the stop gate is data and ships | `docs/results_open_gated_seat.md` (digit-for-digit) |
| evidence-based stop transfers | `docs/results_stop_verdict.md` (0.631) |
| the vocabulary is still hand-drawn | `project/plans/0033-fragment-tower.md` (Phase 0 open) |

## 9. What we are *not* claiming

- **not** "127 laws": 127 ledger *names* = 13 promoted + 57 folds + 57
  premises.
- **not** a speed win for the minted family — the opposite, measured.
- **not** a reusable player — the transfer test says no.
- **not** that the self-modifying game has been shown to *improve play*.
  What is shown: the bank grows under play, and a player ranks
  candidates 4.7× better than the heuristic. The loop between them —
  play grows the bank, the grown bank makes the *same* player better
  without retraining — is still **unmeasured**. That is the G\* claim,
  and it remains open.

## 10. What next — ranked by risk removed per week

1. **Plan 0033 Phase 0 — canonical fragment vocabulary** (no external
   dependency). Emit `ops/laws/handlers/schemas/prims/families` from
   their five current owners behind one schema pin; registries become
   projections. *Pass:* an added fragment (human or machine) is visible
   to `pipeline.propose` and `mlopt-handl-peer.transform_ids` identically.
2. **Make `pay` measured, then re-admit.** Either build the fused-body
   codegen path for minted ops (`genkernel` already lowers `claim`ed
   handlers) or gate `MACHINE_SHIPPED` on measured pay. **Landed
   2026-10-25:** `bench.referees.pay_wall` (graphopt `523ab1d`) times the
   delivered module, with a same-executor rule. Verdict:
   **7 lose · 1 tie · 6 confounded · 0 honest delivers**, and every fold
   that fires (9 of 57) loses, worst 2.66× (`docs/reports/pay_measured.md`).
   So the pass criterion is met the honest way — the family is demoted as
   a *pay claim*, kept as provenance. Still open: the 48 unmeasured folds,
   and whether a real fused body can beat Inductor at these sizes (the
   compiled twins say no).
3. **The decisive experiment: does a grown bank improve the same
   player?** Freeze a player, grow the bank, replay the frozen player on
   the same held-out boards. *Pass:* a delta with a paired control.
   **Measured 2026-10-25** (`bench/drivers/policy/bank_growth.py`,
   `docs/results_bank_growth.md`, 5 seeds, frozen linear player):
   - minted prim growth 74 → 131: **−0.023** reward/episode (all seeds
     ≤ +0.008) — unpaid moves dilute a fixed policy;
   - control, paying growth 127 → 131 (`id_add`/`id_mul`/`double_neg`/
     `silu_fold`): **+0.066** (all seeds ≥ +0.036), ~9× the reward.
   So **growth helps iff it pays** — the same condition the measured pay
   gate enforces. G\* is not "the bank grows"; it is "the bank grows with
   moves that pay". Still to run: mint *during* play, then measure.
4. **Held-out replication of 21/21.** Mint on `zoo-prim`, admit on a
   model family never in the mint corpus. *Pass:* originals survive at a
   stated rate, or the corpus dependence is named.
5. **Close the tech debt that bank growth will break:** import-time
   domain registration, pinned rule-count tests, kernel-prim shape gate.
   Cheap now, expensive at 200 rules.
6. **Un-pin the delivery path from CPU.** `torch_bridge.Plan.run_xs`
   allocates intermediates from a CPU `like` tensor, so a generated
   Triton body cannot be measured end to end — the op-level 2.6× stands,
   the module-level number does not exist yet. `measure_module` now takes
   `device`/`dtype`; the harness (`prim_kernel_module`) is written and
   blocked on that one change. This is the highest-value refactor left:
   it turns "the folds pay" from an op-level result into a delivered one.

**Decision on the 48 unmeasured folds (2026-10-25):** accept them as
*vocabulary*, not as pay claims. They are labeled `unmeasured` in
`provenance.pay_status()` and in `docs/reports/pay_coverage.md`; measuring
each on its own bound site is a follow-up, not a blocker, because a fold
that never fires cannot pay on the corpus that matters.

Deferred on purpose: scale (Q4 already names the three walls —
matcher ~11k nodes, verify-forward RAM, fp64 overflow), and any new
domain until the pay leg is honest.

## 11. One-line summary

**The machine can now name new things and prove them — 127 names, 21 of
21 original — and the only two things still missing are a vocabulary
that lives as data and a referee that measures instead of prices.**

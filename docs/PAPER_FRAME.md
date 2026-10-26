# The paper, framed from the evidence

**Status:** frame, not draft. Every number here is measured and points at
an artifact (`docs/EVIDENCE.md` is the spine). Written 2026-10-25 after
the pay, kernel and growth measurements landed.

## The claim, in one paragraph

Compiler rule banks are frozen: every deployed rule was written by a
person, and the machine-generated ones ship once. We built a substrate
where rules, proofs and programs are values in one calculus, so a minted
rule re-enters the bank as a legal move and discovery closes into a game
a learned policy can play under a certifying referee. The measurement
that matters is not that the loop exists but **where in it knowledge
enters**: originality comes from *vocabulary growth* — 21 of 21 mints are
machine-original when the machine may name a new operation, and 0 of 97
when it may only compose known ones. And a discovered rule pays **only
when its deliverable is a kernel for a whole chain** (2.6–2.9×), never
when it becomes one more step inside an interpreted module (2.7–3.3×
slower). A frozen player improves under a grown bank **iff the grown
moves pay** (+0.066 vs −0.023 reward/episode). The contribution is the
loop *and* the three conditions that make it pay.

## What the paper must show, in order

1. **The substrate.** One calculus, three levels: programs (1-cells),
   rules (2-cells), derivations (3-cells); rules are serializable rows,
   certificates replay standalone, and a minted object re-enters as data.
   Artifact: the bank store, the golden test on every record.
2. **Originality is a vocabulary event.** `zoo-prim` (68 modules):
   21/21 machine-original, then 48/48 deeper; the anchor example is four
   published architectures sharing one "sum of gated products" core, found
   by antiunification. Artifact: `docs/reports/primitives_finds.md`.
3. **The control that makes it a result.** Composition inside a fixed
   bank cannot be original: 97 admitted, **0** original. Artifact:
   the codegen domain's overnight run.
4. **The referee must measure.** The cost model priced the fold at
   0.5–0.8×; the delivered module lost 7 of 14 cells, won 0. Then the
   gate: same-executor rule, 6 cells unusable rather than "wins".
   Artifact: `docs/reports/pay_measured.md`.
5. **The delivery condition.** Same folds: 2.6× as an isolated kernel
   body (fp32 *and* fp64), 2.67–3.34× slower as a module step, 2.87× as
   a whole-chain generated kernel. Artifact: `docs/reports/prim_kernel.md`.
6. **The growth condition.** Frozen player, two banks: +0.066 when the
   growth pays, −0.023 when it does not. And minting during play on the
   toy board admits nothing that pays. Artifact: `docs/results_bank_growth.md`.
7. **What this means for the field.** Learned play over rewriting has
   been measured with fixed banks (Omelette, MCTS-GEB, Aurora) or fixed
   structures (AlphaTensor). The variable that moves the ceiling is the
   one nobody lets the machine touch: the vocabulary. Rule-learning work
   (Ruler, Enumo, Isaria) grows the bank offline; this grows it *under
   play*, and pays only when the new move carries its own kernel.

## Where the paper must be humble

- 127 is ledger *names* (13 promoted + 57 folds + 57 premises), not 127
  independent theorems.
- The kernel result is at op scale and at chain scale; **not** at module
  scale. Say all three numbers.
- 48 shipped folds are unmeasured — vocabulary, not pay claims.
- Whole models run at parity; the 7–11× gap to hand-fused kernels is
  codegen, not semantics.
- One device, one policy class, five seeds; the growth deltas are within
  2× their own spread, and both directions clear the noise band.

## The next experiment the paper wants

A corpus whose folds bind ≥1 M elements **and** a delivery path that
emits one kernel per chain. Everything needed exists (`gen` codegen,
`genkernel.triton_bindings`, the gauntlet, the frozen-player harness);
what is missing is the mint producing *chain-level* kernel claims instead
of rule folds. That is the follow-up paper's experiment — run once here, on
the corpus's own modules, and it returned a negative with a shape: see the
result below. The recommendation this paper ends on is therefore the
conditional form, not the hope.

**Blocker, measured 2026-10-25 — fixed 2026-10-26.** `play --domain gen --zoo`
(the path that produces chain-level numbers on real modules) failed at every
zoo scaling. Four sample workloads baked auxiliary shapes sized for the
`d16 t8 b2` default; each now derives them from `(b, t, d)`, and the default
shapes are reproduced exactly, so no published number moves:

| workload | what was baked | now |
|---|---|---|
| `MLPMixerBlock` | token-mix linear at 8 tokens | `t` |
| `GCNLayer` | adjacency `(2, 6, 6)` — the default batch, by accident | `(b, 6, 6)` |
| `BlockSparseAttention` | `blocks = 4`, so `t = 8` | `t // 2` |
| `LearnedPosSlice` | table at 16 positions | `max(t, 16)` |

68/68 workloads now run at `(16, 8, 2)` **and** `(64, 32, 8)` (was 64/68
at wide).

**Result, measured 2026-10-26, on the whole corpus.** The chain-level question
is answerable and the answer is narrower than the follow-up hoped: over **all
68 zoo workloads** at `d64 t32 b8`, three rounds each, **65 deliver and verify
bit-equal** (`max_abs 0.0`) and **15 are faster than eager in every round** —
`TernaryWeightLinear` 1.94×, `RotaryPair` 1.52×, `PartialRotary` 1.47×,
`GroupedQuantLinear` 1.43×, `DeepEquilibrium` 1.30×, `LoRAAdapter` 1.28×,
`TimeCondConv` 1.24×, … — while 44 lose, median **0.972×** (31 by <10 %).
`FNOBlock` is **1.77× `torch.compile`**. The mechanism is dispatch: the
delivered form launches 22 kernels where eager launches 34, and the win rate
is 41 % where eager leaves more than half its wall time in dispatch against
10–17 % where the kernels already dominate (`zoo_kcount.py`). Kernel count
itself does **not** predict the sign (r = 0.01) — the delivered body's own
cost does, which is the referee's job and it misses at least once
(`RMSNormPerHead`, 0.208×). Full tables, profiler evidence and the timing
correction: `docs/reports/zoo_chain.md`, claim 9 in `EVIDENCE.md`.

A note on the timing, because it cost a wrong verdict: an earlier revision of
that report called `TimeCondConv` **unusable** (gen 130–431 µs across five
identical runs). The swings were mine — samples taken while other jobs shared
the box. Eighteen clean round-readings sit in 1.15–1.42×, none below 1. The driver now takes
`--rounds` (default 3), reports the median and keeps the spread, and a
module-level ratio from a single round is no longer publishable.

So the 2.87× stands for what it is — a whole-chain kernel over a synthetic
tape — and the opening claim is bounded by measurement rather than assertion:
a generated kernel pays **as an op** (2.6×), as the **whole chain** (2.87× on
a synthetic tape), or where the delivered module **launches fewer kernels than
eager** (15 of 65 zoo modules, 1.04–1.94×, win rate 41 % on launch-heavy rows
vs 10–17 % on kernel-heavy ones), and not as a step inside a module that is
already kernel-bound.

## Section skeleton

| § | content | artifact |
|---|---|---|
| 1 | problem: frozen banks | history table (TASO, Isaria, Ruler→Enumo, Aurora) |
| 2 | substrate: n-morphisms as values | substrate paper (`eqsat`) |
| 3 | the game | board/referee/score/players |
| 4 | originality requires vocabulary | `primitives_finds.md` + the 0/97 control |
| 5 | delivery requires a kernel | `pay_measured.md`, `prim_kernel.md` |
| 6 | growth requires pay | `results_bank_growth.md` |
| 7 | limits and the next experiment | this file |

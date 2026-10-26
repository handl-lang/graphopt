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
of rule folds. That is the follow-up paper's result, and it is the
recommendation this one ends on.

**Blocker, measured 2026-10-25:** `play --domain gen --zoo` — the path that
would produce chain-level numbers on real modules — fails on this tree,
at every zoo scaling tried:

```
RuntimeError: a and b must have same reduction dim,
              but got [512, 32] X [8, 8]
```

Three workloads hardcode auxiliary shapes instead of deriving them from
`(b, t, d)`, so at `d=64 t=32 b=8` they cannot run at all (found by
running each workload at those dims, 2026-10-25):

| workload | symptom | cause |
|---|---|---|
| `MLPMixerBlock` | `mat1 and mat2 shapes cannot be multiplied (512x32 and 8x8)` | mixer linear sized to a hardcoded 8 tokens, not `t`/`d` |
| `GCNLayer` | `size of tensor a (2) must match b (8)` at `zoo.py:357` | adjacency built at a fixed 2 nodes while the batch is 8 |
| `BlockSparseAttention` | `shape '[8, 4, 2, 64]' invalid for input of size 16384` | hardcoded heads/seq in the reshape |

None of these is in the discovery path — they are sample architectures
used by the zoo probe — but the chain-level measurement needs all of them
at wide shapes. The fix is to derive each auxiliary tensor from `(b, t,
d)`; the `--probe` path (fixed `chainfuse` / `mulsoftplus` cases) is
unaffected and is what the 2.87× / 1.92× numbers come from.

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

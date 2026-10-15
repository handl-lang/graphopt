# Retro — gate 3: the mixed state spelling is an extraction tie, and the fix is real but does not close the gate

**Date**: 2026-10-16.
**Commit**: `aa56e55` (catopt `main`).
**Decides**: ADR 0008 gate 3 → still **FAIL**; the third named cause
(*extraction mixes two state spellings*) is **measured**, is a real
extraction defect, and is **not sufficient** — fixing it recovers
**1.01 %** FLOPs (~~5.06 %~~ — see the correction note) and leaves the
gate negative.  ADR 0009 stage 4 stays unjustified.

> **Correction 2026-10-15** (`project/retros/saturation-determinism.md`,
> catopt `abf6114`).  The sharing-aware recovery this retro reports
> (**455,752 → 432,696, −5.06 %, 5 overrides**) is **superseded**.  The
> staged gate-3 saturation and its sharing-aware optimum were
> **hash-order dependent** — the matcher and the extraction scanned
> `EClass.nodes` (a `set` whose iteration order follows `ENode`'s hash) —
> so the numbers moved with `PYTHONHASHSEED`.  Both are now
> **deterministic** (canonical member order, `abf6114`).  The
> reproducible reading is **455,752 → 451,128 FLOPs (−1.01 %), 3
> overrides**, with the read sequence **unchanged** from the greedy term
> (`reexp|reexp|fold|fold|reexp|shared|shared`) and the e-graph at
> **9,784** enodes (was ~~9,733~~).  The **432,696 / −5.06 % / 5-override**
> figures below are kept visible struck through; 432,696 was a hash-order
> artifact of the pre-fix scan.  The retro's **mechanism** and its gate-3
> **FAIL** conclusion are unaffected: the extraction *is* still off the
> cost its own model assigns — by **1.01 %**, not 5.06 %.  Numbers
> reported before `abf6114` may be seeded/arbitrary; the reproducible
> runs use `PYTHONHASHSEED` 0–6 + unset, all identical.

## What the gate was, and what was asked

The parametric-abstraction retro
(`gate3-parametric-abstraction.md`) named three causes for the winner's
failure to collapse to one shared `step`: (1) two `split` families /
two chunk sizes, (2) **extraction mixed two state spellings**, (3) the
blocks are pairwise distinct modulo the varying integers.  This task
attacked cause 2 alone, as an **extraction/pricing** question:

> why does the extractor produce two spellings of the same state read,
> and does a *principled* fix unify it?

The answer, measured: **the two spellings are a cost tie**, broken by
`extract_best`'s member order (op name) — arbitrary with respect to
*sharing*.  A principled fix exists and was applied; it recovers
**1.01 %** (~~5.06 %~~ — see the correction note) of the winner's FLOPs
and **does not** close gate 3.

## What was built

`lab/probes/gate3_state_spelling_probe.py` — the measurement.  On the
staged-arm winner it reports, per block, the state-read subterm, its
e-class, **every member of that e-class priced**, the single-class
override cost of forcing the factored read, a cost-model sweep, the
per-chunk-family abstraction verdict, and a numeric greedy-vs-fix check.

`lab/bridge/sharing_extract.py` — the fix.  `extract_sharing_aware`:
begin at `eg.extract_best`, then sweep the extracted term's e-classes
and, for each member of each class, re-extract with that class forced to
that member; keep any override that **strictly lowers the whole term's
`dag_cost`**, to a fixed point.  It is *monotone* (a step is kept only
when the total falls), so it can never manufacture a form the cost model
did not already prefer — it only removes the arbitrary per-class choice.
Wired into the gate-3 protocol as
`lab/bridge/combinator_probe.py --sharing-extract`.

**Why in `lab/` and not in `catopt_core.egraph.extract`**: the same
tie-break inside `extract_best` would change extraction for **every**
caller, and the AGENTS rule is explicit that such a change requires the
full suite (which this host's memory budget forbids).  The measured
effect is one workload; the prototype is where that measurement belongs.

## The mechanism — measured, with numbers

Command (the probe; `PYTHONHASHSEED=0` as reported here — the seed is no
longer load-bearing, see the caveat at the end):

```sh
PYTHONHASHSEED=0 .venv/bin/python lab/probes/gate3_state_spelling_probe.py --t 64 --k 4
```

Staged arm, T=64 K=4, ~~9733~~ **9,784** enodes, winner dag=139, cost **455,752**,
read sequence (the winner's per-block state-read spelling)

```
read:reexp | read:reexp | read:fold | read:fold | read:reexp | read:shared | read:shared
```

i.e. blocks 0/1/4 spell the read as the *unfactored re-expansion*
(`om_decayed(matmul(q, kᵀ), D, v)` / a `div` wrapping it), blocks 2/3 as
the *nested fold* (`add(mul(γ, S_{i-1}), B_i)`), blocks 5/6 as the
*factored shared read* `matmul(q, S)`.  Three spellings, not two.

### 1. The cost model is indifferent — it is a tie

For each block, the state-read e-class has several members priced the
same, and `extract_best` breaks the tie by `nops` then by member sort
order `(op, children, repr(attrs))` — i.e. **by op name**:

| block | class | members | tied at min | chosen member | class min |
|---|---|---|---|---|---|
| 0 | 517 | 3 | **2** | `om_decayed(div(matmul(q,kᵀ)))` | 198,896 |
| 1 | 528 | 4 | **2** | `om_decayed(div(matmul(q,kᵀ)))` | 198,832 |
| 2 | 6409 | 4 | **2** | `applyd(aff_diag(…))` | 201,300 |
| 3 | 6417 | 11 | **2** | `add(mul(unsqueeze(decay_vec…)))` | 203,764 |
| 4 | 93 | 4 | **4** | `div(om_decayed(matmul(q,kᵀ)))` | 232,000 |
| 5 | 6345 | 4 | 1 | `matmul(div(split(q)), S)` | 249,924 |
| 6 | 6353 | 11 | 1 | `matmul(div(split(q)), S)` | 267,844 |

**5 of the 7** classes are tied at the minimum (2-way in blocks 0–3, a
**4-way tie at 232,000** in block 4); blocks 5–6 have a unique minimum
and already take the factored spelling, so no mix arises there.

### 2. Forcing the factored read per class moves the total by ≤ 0.03 %

Single-class override (`overrides={cls: factored_enode}`), whole-term
`dag_cost` against the 455,752 base:

| block | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Δ FLOPs | *n/a* | +144 | +16 | +16 | **−528** | 0 | 0 |

`n/a` = block 0 has no cross term (no factored read reachable).  So the
model is *indifferent* to within 0.03 % on blocks 1–3, exactly tied on
5–6, and **strictly prefers the factored read on block 4** — because
block 4's factored read *shares its state with block 5's read*, a
non-local saving the per-class greedy pass cannot see.

### 3. The arbitrary break is strictly suboptimal

**Corrected 2026-10-15** (`project/retros/saturation-determinism.md`):
the reproducible sharing-aware optimum is **451,128 FLOPs** — **4,624
FLOPs, 1.01 %** below the greedy winner — through **three** overrides
(`{355: split, 347: split, 93: matmul}`).  The superseded figures
(**432,696**, **−23,056**, **−5.06 %**, **5** overrides) were a
*hash-order artifact* of the pre-fix member scan; the pass now scans in
canonical member order, so it is deterministic across processes.  The
greedy arm is unchanged:

```
greedy           cost 455,752   dag=139   morph=1590   round-trip EQUAL
sharing-aware    cost 451,128   (was 432,696)         round-trip EQUAL
                 Δ = -4,624 (-1.01 %, was -23,056 / -5.06 %),
                 max|greedy - fix| = 2.220e-16
```

(The sharing-aware arm's `dag`/`morph` were reported for the superseded
hash-order term — `dag=133`, `morph=1509` — and were **not re-reported**
after `abf6114`; the reproducible term is numerically equivalent to the
greedy one, below.)

The numeric check lowers both terms through `TorchSink` on the same
weights and feeds and reads **2.2e-16** — the cheaper term is the same
program to machine epsilon (and the e-graph guarantees the equivalence
by construction).  Overrides kept: `{355: split, 347: split,
93: matmul}` (the superseded run kept two more, `{341: split, 6424:
applyd}`, as part of the artifact).

**This is a finding in its own right**: on the gate-3 workload the
per-class greedy extraction is **1.01 %** (~~5.06 %~~) off the cost its
own model assigns, and a bounded, monotone coordination recovers it —
though at ~1 % on a single workload the promotion of the tie-break into
`catopt_core.egraph.extract` is a **judgement call**, not the obvious win
the 5 % figure implied.  It is the same non-local blindness
`extract_paired` exists for, here on the tie-break rather than on a group
decision.

### 4. It is not a rule-budget artifact

The factored read **is reachable** in the small-chunk blocks (block 1's
class 528 contains `matmul(div(split(q, index=1, …)), S)` at 205,456).
So hypothesis 2 of the brief — "reachable in some blocks and not
others" — is false; hypothesis 1 (indifference) is what the numbers show.

## The gate-3 before/after

```sh
PYTHONHASHSEED=0 .venv/bin/python lab/bridge/combinator_probe.py \
    --gate2 --extracted --propose-loop --abstract-index                # before
PYTHONHASHSEED=0 .venv/bin/python lab/bridge/combinator_probe.py \
    --gate2 --extracted --propose-loop --abstract-index --sharing-extract  # after
```

| | before (greedy) | after (sharing-aware) |
|---|---|---|
| extracted cost (FLOPs) | 455,752 | **451,128** (−1.01 %; ~~432,696 / −5.06 %~~) |
| extracted dag | 139 | 133 (superseded term; not re-reported) |
| bridge morph | 1590 | 1509 (superseded term; not re-reported) |
| bridge round-trip | EQUAL | EQUAL |
| numeric max\|Δ\| (greedy vs fix) | — | 2.2e-16 |
| read sequence | reexp\|reexp\|fold\|fold\|reexp\|shared\|shared | reexp\|reexp\|fold\|fold\|reexp\|shared\|shared (**unchanged**; ~~reexp\|reexp\|fold\|fold\|**shared**\|shared\|shared~~) |
| `loop`-propose fires | 0 | 0 |
| `loop`-headed reachable | False | False |
| **abstraction verdict** | **IRREGULAR** | **IRREGULAR** |

Against the ADR's criterion — **PASS** = a `loop`-headed (or
loop-carrying) winner that round-trips equal; **FAIL** = still spelled —
the gate is **still FAIL**, with the **same first difference**:

```
block1/block2/add[1]/mul[1]: op om_decayed vs add
```

So the fix moves the winner's *cost* (by 1.01 %) — and the winner still
does not collapse.  (The superseded claim that it also *unified* the
large-chunk family's top-level read was part of the hash-order artifact:
the reproducible read sequence is **unchanged** from the greedy term.)

## Why fixing cause 2 does not close the gate

The mix is not one arbitrary choice; it is a **cascade** of per-class
ties, and one of them is a genuine cost preference.

1. **The small-chunk family genuinely prefers the unfactored spelling.**
   Forcing the factored `matmul(q, S)` on blocks 1/2/3 costs +144/+16/+16
   FLOPs.  For a chunk of width `C` and head dim `d`, the re-expansion
   contracts `C×C·d` while the factored read contracts `C×d·d`; with
   `C=4 < d=16` the unfactored form is genuinely cheaper.  So no
   tie-break can unify *that* family without paying — i.e. forcing it
   would be manufacturing a result.
2. **Unifying one level exposes the next level's tie.**  With the
   large-chunk family's top-level read unified to `matmul(q, S)`, the
   per-family abstraction's first difference becomes the *state's own*
   internal spelling — `op applyd vs add` (the documented cost-neutral
   respelling of `add(mul(a,h),x)`), then, on the greedy term, purely the
   `split` **index** (`index 1 vs index 2`).
3. **The abstraction's own state-binding is not stable across blocks.**
   Per family, `index_abstract._state_inputs` binds different structural
   positions for parallel blocks:

   ```
   family (4,4,4,4,16,16,16)  4 blocks  state_in = None | unsqueeze | unsqueeze | unsqueeze
   family (16,16,16,16)       3 blocks  state_in = None | matmul    | mul
   ```

   The move's "outermost own-split-free subterm" rule is exactly the
   structural heuristic its own retro flagged as a limitation; on the
   real winner it picks a different position per block, so the templates
   differ even where the spelling agrees.

## What it proves, and what it does not

**Proves:**

* **Cause 2 is real, and it is an extraction defect — but a *small,
  arbitrary* one.**  The state-read spelling is a cost tie in 5 of 7
  blocks (a 4-way tie at block 4), broken by op-name order, and the
  greedy pass is **1.01 %** (~~5.06 %~~) off the cost its own model
  assigns.
* **A principled fix exists and was applied.**  Sharing-aware
  coordinated extraction removes the arbitrary choice and adds the real
  (non-local) sharing signal; it is monotone, numerically verified
  equivalent (2.2e-16), and recovers **4,624** FLOPs (~~23,056~~).  At
  ~1 % on one workload, though, promoting the tie-break into
  `catopt_core.egraph.extract` is a **judgement call**, not an obvious
  win.
* **The fix does not close gate 3.**  The verdict is unchanged, and the
  remaining causes are named below.

**Does not prove:**

* That no extraction change could *ever* close the gate.  It rules out
  the *tie-break* class of fix — because the small-chunk family's
  preference for the unfactored spelling is a genuine cost difference,
  and because the residual differences are not spellings at all.
* Anything about `catopt_core.egraph.extract` as shipped: the
  sharing-aware pass is a **lab prototype**.  Whether the same
  coordination belongs in the package (as an opt-in arm beside
  `fusion_epsilon` / `extract_paired`) is a separate decision needing
  the full suite.

## The remaining cause, named

Cause 2 is **not** the blocker and is **not independent of cause 1**.
With the spelling mix removed as far as it can be principledly removed,
the winner still fails for:

1. **the two chunk widths** (`sizes=(16,16,16,16)` and
   `sizes=(4,4,4,4,16,16,16)`, both with indices `[0,1,2,3]`) — the
   abstraction's `INDEX_ATTRS` abstracts `index`/`t`/`offset`/`sign`
   but **not `sizes`**, so the two families can never be one step; and
   the small-chunk family's *unfactored* spelling is a genuine cost
   optimum (`C²d < Cd²` at `C<d`), so a cost-minimising extraction
   cannot be made to spell it uniformly for free; and
2. **the abstraction's structural state identification**, which binds
   `matmul` for one parallel block and `mul` for another
   (`_state_inputs`' "outermost own-split-free subterm" rule).

## Caveat — reproducibility (resolved)

**Corrected 2026-10-15** (`abf6114`): this caveat is **resolved**.  The
staged saturation used to consume `rule_budgets` in a set-iteration
order, so `PYTHONHASHSEED` changed the e-graph (measured **9733 / 9752 /
9756** enodes across runs) and the sharing-aware local optimum varied
between **432,696** (5 overrides) and **451,128** (3 overrides).  With
the canonical member order both are deterministic: the reproducible
staged arm is **9,784** enodes and **451,128** FLOPs in every process
(`PYTHONHASHSEED` 0–6 + unset, all identical).  Numbers reported before
`abf6114` — including this retro's headline 432,696 / −5.06 % — may be
seeded/arbitrary; pin the seed when re-measuring anyway, and prefer the
canonical reading.

## Cost

* The probe: one 15 s staged saturation + ~2 s of class enumeration +
  the sharing-aware pass (898 trials, ~46 s; ~~1307 trials, ~64 s~~ for
  the superseded hash-order run).  Peak RSS well under 1 GB.
* The gate-3 arms unchanged: one 15 s saturation + the 80 s
  sharing-aware arm.
* New surface: one lab module (`lab/bridge/sharing_extract.py`,
  ~165 lines), one lab probe (`lab/probes/gate3_state_spelling_probe.py`,
  ~560 lines), ~30 lines of probe wiring.  **No shipped-package change**,
  no law-bank change, no `catopt_core` change.

## Gates run

`.venv/bin/ruff check packages lab scripts examples` and
`ruff format --check` (clean), `.venv/bin/ty check` (clean),
`.venv/bin/vulture` (clean), `.venv/bin/lint-imports` (4 kept, 0 broken),
`.venv/bin/bandit -c .bandit.yaml -r packages` (0 findings),
`.venv/bin/semgrep --config .semgrep.yml packages` (0 findings),
`scripts/radon_ratchet.py` (ok, 4711 functions), and
`pytest tests/test_index_abstract.py tests/test_loop_propose.py
tests/test_combinator_bridge.py` (35 passed).  The full suite was **not**
run (memory/time budget; no shipped-package change).

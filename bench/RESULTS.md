# Headline measurements — what was measured, on what, and how to re-run it

Every row below points at a runnable probe or test.  The numbers in the
*fresh* column were re-measured on this dev box (i5-12500H, 11 GB,
debug-build `hnl-run`) on 2026-10-18; the *recorded* column is what the
retros pinned.  Where the two differ it is said plainly — wall-clock on
a debug tree-walk interpreter is directional, not a benchmark rig, and
at kernel scale the per-call total is dominated by program-parse +
leaf materialization, so reported deltas are ranges across repeated
runs, not single points.

**Scope, stated once:** these are **kernel-scale** results on fp64
fixtures — one conv+BN block, one decision.  The claim they support is
*"a certified rewrite a hand optimizer would write, discovered and
proven by the machine, and priced by the runtime itself"* — **not**
"we beat a production compiler end-to-end at model scale."

## The flagship — `conv_bn_fold`

The eval-mode weight fold `batch_norm(conv2d(x, W, b), γ, β, μ, σ²)
→ conv2d(x, W′, b′)` with `W′ = W ∘ ŝ′`, `b′ = b·s + (β − s·μ)`,
`s = γ·rsqrt(σ²+ε)` — minted by `meta.synthesize_rules` over
`BATCH_NORM_RULES` + `CONV_FOLD_RULES`, admitted through the gauntlet,
and shipped **verbatim** in `DEFAULT` (`catopt_core.laws.minted`;
`MACHINE_SHIPPED` = 12 laws).

### The decision, costed through the runtime itself

Spelled `batch_norm(conv2d(x,W,b),…)` vs deployed `conv2d(x,W2,b2)`
where `W2`/`b2` are catopt subterms materialized through the kit
(fold-once compile-time work).  Per-call wall measured by
`hnl-run --repeat 30` inside one `CatoptKit` process — the runtime
the e-graph's terms execute on, not an estimate.

**Three different pay numbers exist for this fold — −16 %, −5 %, and
+33 % — because `hnl-run --repeat` re-evaluates the *whole program*
per call, and each measurement puts the fold math in a different
place.**  The fold math is `channel_scales` (a `.hnl` `while`/`match`
loop over channels — the op surface has no elementwise `rsqrt`) plus
the `W′`/`b′` tensor updates: ~10–14 ms/call of tree-walk work on this
box's debug build.  Whether the per-call number shows a win, a loss,
or an inflated win depends entirely on whether that work is hoisted
out of the timed loop, inlined into it, or hiding inside the spelled
baseline's prelude:

| arm | what it actually measures | per-call (this box, 3 runs) | recorded |
|---|---|---|---|
| spelled, test feed | `emit_program(unfolded, feed)` — the prelude binds **every** `feed` key, incl. the `cs`/`s` marshal escape the spelled term never reads | 97–105 ms | 87.0 / 95.4 ms |
| spelled, trimmed prelude | the same term, prelude cut to its own leaves — the escape's ~10–14 ms/call removed | 87–91 ms | — |
| deployed `conv2d(x,W2,b2)` | `W2`/`b2` materialized once, *untimed*; per-call is a bare conv — the honest deployed regime | 82–89 ms | 72.8 / 82.1 ms |
| module `conv_bn` | the `.hnl` export called per eval — the scaffold's "spelled" shape through this binary | 80–91 ms | — |
| module `conv_bn_fold` | fold math inline, recomputed per call — the un-amortized regime | 86–99 ms | — |

| delta | this box | recorded | reading |
|---|---|---|---|
| deployed vs spelled (test feed) | −11 to −17 % | **−16.3 %** / −13.9 % | right regime, **inflated baseline** — the spelled program pays ~10–14 ms/call for marshal work it never uses |
| deployed vs spelled (trimmed) | −1 to −7 % | — | the op-level pay, prelude artifact isolated: low single digits at kernel scale |
| deployed vs module `conv_bn` | ±7 % (noise) | **−5 %** | the deployed regime through the module path — prelude-dominated, inside run noise here |
| `conv_bn_fold` vs `conv_bn` | +7 to +10 % | **+33 %** | the un-amortized regime — fold math per call is a net *loss* |

The scaffold's independent run (`handl-lang/research-mlopt`
`experiments/`, `conv-bn-fold --repeat 100`, in-process kit, its own
6×4×3×3-on-8×8 dyadic fixture) lands the same two regimes and confirms
`hnl-run` exercises the identical path — the scaffold composes the
renamed `hnl-engine-*` crates directly (the same engine component set
`CatoptKit` assembles; `manifest/kit.toml`'s gap row), just kept
in-process instead of behind the subprocess door:

| scaffold arm | this box (2 runs) | recorded |
|---|---|---|
| spelled `conv_bn` | 22.8 / 24.1 ms | 3,921 µs |
| folded-inline `conv_bn_fold` | 28.0 / 29.4 ms (**+22–23 %**) | 5,206 µs (**+33 %**) |
| deployed `conv2d` alone | 20.7 / 22.7 ms (**−6 to −9 %**) | 3,724 µs (**−5 %**) |

The honest reading:

* **`dag_cost` is the decision's stable headline** — 34,874.3 →
  17,438.7 (−50 %) on the premise-pay probe's 8×8 site, 55,296 →
  36,864 (−33 %) on this test's 24×24 term.  Box-independent.
* **The deployed regime is a real but small wall win at kernel
  scale** — ~−5 % is the best cross-harness estimate (the scaffold's
  deployed arm, reproduced twice at −6 to −9 %).  At this size
  per-call wall is dominated by program-parse + leaf materialization,
  so op-count savings land in low single digits; through `hnl-run`
  with a matched prelude the delta is inside run noise.
* **−16.3 % is the same deployed arm against a baseline carrying an
  unused marshal escape** — `emit_program` binds every `feed` key and
  the test's feed carries `cs`/`s` (they exist to *compute* the folded
  weights; the dict is reused for timing).  Direction right, magnitude
  artifact: trim the prelude and the same measurement reads −1 to −7 %.
* **+33 % is the wrong regime** — it prices the fold as if it were not
  amortized.  Reproduced at +7–10 % (this binary/fixture) and
  +22–23 % (scaffold).  It is the honest answer to "what would inline
  folding cost per call" — and precisely why the law's pay is a
  compile-time weight-space fold, which `dag_cost` prices correctly.

Output equality on the seeded fixture: **max abs diff 0.0** (bitwise);
the test's asserted bound is `<1e-12`.  The full `torch.export` IR of
an eval-mode `Conv2d → BatchNorm2d` module evaluates through the kit
to the torch-fp64 answer at **6.7e-16**.

- Reproduce the recorded arm:
  `pytest tests/test_kit_bridge.py` (8 tests, `e2e` — 8/8 green in
  26.7 s on this box; needs `target/debug/hnl-run`, built by
  `cargo build -p catopt-kit --bin hnl-run`).
- Reproduce the five-arm decomposition (the table above):
  `.venv/bin/python lab/probes/conv_fold_wall_regimes_probe.py`
  (~5 min — five programs × 31 subprocess evals, debug build).
- Reproduce the scaffold arm (sibling checkout):
  `cd ~/GitHub/handl-lang/research-mlopt &&
  cargo run -p mlopt-experiments --bin conv-bn-fold -- --repeat 100`
  (source on `origin/parallel/scaffold`; resolves this repo's
  `hnl/conv_bn.hnl` through its own ladder).
- Retro: `project/retros/kit-term-bridge.md`.

### The admission account, end to end

`lab/probes/conv_fold_premise_pay_probe.py` (~13 s on this box)
reproduces the whole arc on the real eval-mode ConvBN seed:

| leg | measured |
|---|---|
| mint | 974 rules emitted by `synthesize_rules` (fuel 30 000, 4 rounds) in **11.8 s**; the fold composite mints deterministically (identical `syn_*` names across runs) |
| premise closure | 8 rules — `batch_norm_expand` + four `conv_channel_*` anchors + three `syn_*` intermediates, parents-first |
| guard transport | `missing=('check','derive')` → both rebuilt from stored `"guard"` records, transitively |
| truth | real site **1eq/0ne**; guarded synth region **105eq/0ne** (255 declined, 360 envs) |
| derivability | premise-anchored: `replayable`, 1 step, `witness=('conv_bn_fold',)` — the shipped law proves its own minted origin; `derivable=True` under shipped axioms alone |
| pay | fires=1, `dag_cost` **34,874.3 → 17,438.7** on the 8×8 site, `verified=pass`, `typed=1/0` |
| gauntlet | refuses at **novelty** (`relation=duplicate`) in 0.4 s — the documented signature of "already admitted", not a regression |

- Reproduce: `.venv/bin/python lab/probes/conv_fold_premise_pay_probe.py`
- Retros: `project/retros/minted-composite-default.md`,
  `premise-anchored-derivability.md`, `fold-cost-single-source.md`.

### The shipped consumer — `cargo test -p catopt-kit` (21/21)

The Rust kit composes handl's language + engines + host-tensor op
surface as dependencies; `hnl/conv_bn.hnl` carries both spellings of
the fold, value-equal on dyadic fixtures.  The bridge test suite is
`tests/test_kit_bridge.py` (the Python↔kit seam via `hnl-run`).

## Supporting rows (full tables in `docs/results.md`)

| claim | measurement | artifact |
|---|---|---|
| `mul_unsq_nat` — the first corpus-minted law | −12.27 % on its source workload; `MACHINE_SHIPPED` | `project/retros/mul-unsq-nat.md` |
| Weight-chain fold vs Inductor | 14.92× at (k,d,B·T)=(16,512,4096) | `bench/baselines/reassoc_scale.json` |
| Spelled RetNet export, T=2048 | 5.0–5.9× eager, 6.2–7.4× compile; verified 9.8e-9 | `docs/results.md` |
| Held-out zoo | 11/18 beat `torch.compile` | `project/retros/gen-kernels.md` |
| ONNX round-trip | 1.11e-16 fp64, zero core changes | `project/retros/onnx-package.md` |
| Honest losses | fused-op vision model 0.80×; precision lane certified negative; carrier loses below T≈1024 | `docs/results.md` |

## Reproduction index

```sh
# the flagship decision — needs the hnl-run binary (cargo build -p catopt-kit --bin hnl-run)
.venv/bin/python -m pytest tests/test_kit_bridge.py -q          # 8 tests, ~27 s

# the fold's wall-clock regimes, decomposed (all five arms, one binary/fixture)
.venv/bin/python lab/probes/conv_fold_wall_regimes_probe.py     # ~5 min

# the admission arc on the real seed
.venv/bin/python lab/probes/conv_fold_premise_pay_probe.py      # ~13 s

# the scaffold's three arms (sibling checkout; source on origin/parallel/scaffold)
cd ~/GitHub/handl-lang/research-mlopt &&
    cargo run -p mlopt-experiments --bin conv-bn-fold -- --repeat 100

# the Rust consumer
cargo test -p catopt-kit                                        # 21 tests

# the dev lane (everything except the minutes-scale marked `slow`)
.venv/bin/python -m pytest -m "not slow"                        # 6,353 tests
```

Wall-clock numbers move with the box, the build profile, and the
program's prelude shape; `dag_cost` and the equality bounds do not.
The `--repeat` measurements are debug-build tree-walk timings — they
establish *direction* (the fold's value lands in the deployed program),
and the cost model's ordering agrees with them.

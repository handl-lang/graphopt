# Headline measurements — what was measured, on what, and how to re-run it

Every row below points at a runnable probe or test.  The numbers in the
*fresh* column were re-measured on this dev box (i5-12500H, 11 GB,
debug-build `hnl-run`) on 2026-10-20; the *recorded* column is what the
retros pinned.  Where the two differ it is said plainly — wall-clock on
a debug tree-walk interpreter is directional, not a benchmark rig.

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

| arm | per-call wall (fresh) | per-call wall (recorded) | `dag_cost` |
|---|---|---|---|
| spelled `batch_norm(conv2d)` | **87.0 ms** | 95.4 ms | 55,296 |
| deployed `conv2d(x,W2,b2)` | **72.8 ms** | 82.1 ms | 36,864 |
| delta | **−16.3 %** | −13.9 % | −33 % |

Output equality on the seeded fixture: **max abs diff 0.0** (bitwise);
the test's asserted bound is `<1e-12`.  The full `torch.export` IR of
an eval-mode `Conv2d → BatchNorm2d` module evaluates through the kit
to the torch-fp64 answer at **6.7e-16**.

- Reproduce: `pytest tests/test_kit_bridge.py` (8 tests, `e2e` —
  8/8 green in 26.7 s on this box; needs `target/debug/hnl-run`,
  built by `cargo build -p catopt-kit --bin hnl-run`).
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

# the admission arc on the real seed
.venv/bin/python lab/probes/conv_fold_premise_pay_probe.py      # ~13 s

# the Rust consumer
cargo test -p catopt-kit                                        # 21 tests

# the dev lane (everything except the minutes-scale marked `slow`)
.venv/bin/python -m pytest -m "not slow"                        # 6,353 tests
```

Wall-clock numbers move with the box and the build profile; `dag_cost`
and the equality bounds do not.  The `--repeat` measurements are
debug-build tree-walk timings — they establish *direction* (the fold's
value lands in the deployed program), and the cost model's ordering
agrees with them.

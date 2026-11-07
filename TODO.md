# TODO — graphopt

## title: TODO type: task date: 2026-09-26

- [ ] T-024 Scan-carrier claims mint unevaluable on CUDA — applyd specs embed host-pinned zeros op; spelled eval AND generated binding raise mixed-device bmm/shape (32 referee consults, both dtypes). Harvest-fidelity fix in \_scan\_sites: decline or re-base specs embedding device-pinned const ops. Surfaced by triton\_binding\_audit (T-018); masked today because profitable declines them. · created:2026-11-07
- [ ] T-025 fp64 pow codegen emits fp32 literal — triton\_bindings renders const leaves as (2.0) so libdevice.pow(f64 tile, f32 scalar) raises CompilationError (RWKVMixBlock gen\_12, fp64 only). Render dtype-correct literals or x\*x-style unfolds. Surfaced by triton\_binding\_audit (T-018). · pri:low · created:2026-11-07
- [ ] T-026 Fold-gate residuals — attr-metavar prim specs (5/57) never concrete-match so \_fold\_cost\_gate cannot site them (GatedDeltaRule fp64 still delivers synop\_4c5296b3ba+synop\_5e58f26875 at 0.54x); site-level referee is noisy at the margin (synop2\_35bbd03301 measured 0.64x and 2.2x across processes) and cannot see module-level loss; law folds to ambient ops (rms\_norm, attention folds — RMSNormPerHead fp64 0.21x, SigmoidAttention fp32) never pass through profitable at all. Docs: docs/reports/p27\_fold\_cost\_gate.md · created:2026-11-07

## HIGH — ADR-0159: dissolve the optimizer domains into ext tools

## Adopted on three triggers, not a date ( host/project/adrs/0159-… ):

- [ ] T-001 a caller that is not the researcher needs the capability
- [ ] T-002 the domain's public surface unchanged for four weeks
- [ ] T-003 the referee is shared ( bench/benchkit usable by any domain)

## HIGH — M1: the machine finds the kernel, not me (five items, ~5 days)

## Done — 2026-10-27 (kernel margin, valid)

## Active — HIGH PRIORITY (the open questions, post-plan-0032)

## Active

- [ ] T-009 Meta-arena Q3 ( project/plans/0021-meta-arena.md ) — measured, partially answered. Cold linear REINFORCE beats scripted on mean reward (−0.04 vs −0.10 at 2000 episodes) only via the degenerate extract-early policy; the composed declare→handle line was unreachable by sampling until the claim(tag) option move fused it (claim→extract wins silu 2→1 on held-out boards, +0.52 payouts) — reachability fixed, consolidation by the memoryless policy still open. Retro: project/retros/meta-arena-player.md . Next levers: curriculum boards, imitation warm-start, richer policy.
- [ ] T-010 Generated kernels ( genkernel ) — the novelty question, measured. Handler entries minted for elementwise subterms the shipped HANDLERS never named; torch.compile 'd bindings; gen sink supports+lowers them; claim binds them — the board's vocabulary grows with lowerings it built. The concrete-soundness fix this forced ( \_canon\_concrete , claim\_tags ) is a real correction: skeleton-level coverage could bind mismatched kernel semantics. play --domain gen --probe : bigfuse (6-op chain) → claim(gen\_0) → verified module at 2.87× vs eager (6.1× at 256×1024); 2-op fusions honestly lose (~0.86×, compiled-call overhead). Retro: project/retros/gen-kernels.md . Next: size-guarded claim offers, multi-claim composition, attr-bearing/reduction bodies.

## Deferred

- [ ] T-011 plan 0002 leftovers — Phase E recognizer consolidation (law-side e-class bindings vs lowerer-side extracted terms — substrates genuinely differ, revisit only if a shared kernel emerges); plan-dict typing stays dict-of-any.

## Tech debt to fix

- [ ] T-012 Vendored probe code — copied third-party code in bench/drivers/probes/ (and bench/limits/registry.py duplicated into kernelopt during the split). Replace with a declared dep/submodule or delete if dead.
- [ ] T-013 Import-time domain registration — domains seat themselves via import side-effects ( mlopt\_kernel.play mutating DOMAINS /registry on import). Replace with explicit registration at call site.
- [ ] T-014 Pinned test counts — tests asserting literal rule/law/bank counts ( test\_prior\_art , test\_laws\_structure totals, AGENTS.md counts). Replace with derived counts or invariant-based assertions so bank growth doesn't churn tests.
- [ ] T-015 radon baseline — bilinear\_game.py flagged on clean main; regenerate scripts/complexity\_baseline.json or justify.
- [ ] T-017 fragment-tower migration — local plan in project/plans/0033-fragment-tower.md + host ADR-0157; phases 0–5 (fragment dir → differential backend → denotations as models → fragment-emitting mint → fragment-move play → handl-native saturation).
- [ ] T-016 synop real delivery \* — docs/reports/prim\_delivery.md measured: folds compress terms (6993→4353 ops on GLA-large) but the spec-walk binding ( genkernel.\_eval\_spec ) loses to spelled eager wall-clock on every clean case; the cost-model 0.5–0.8× pay is not delivered. Real delivery needs a fused-body codegen path for minted ops, not a spec interpreter. First step, measured 2026-10-27: the mint does register its ops ( mlopt\_kernel.sink.\_table() 84→86, ksynop\_9e3cb9a03a , ksynop\_efc263c3dd ), but the zoo sweep's delivery path is play.gen\_probe → \_gen\_board(case) and that board's bank comes from genkernel — it never reads the sink table. Teach \_gen\_board to merge the minted ksynop\_\* folds into its bank, then re-sweep the 46 losing zoo rows (median 0.979×). · status:in-progress

## Recovered — session 1792950048869\_3wjyq (generated-kernel claims)

- [ ] T-020 9-round re-measure of the 5 straddlers — CapsuleRouting fp64; WaveNetGate , HyenaLongConv , SkipInitNet , DilatedBlockAttention fp32.
- [ ] T-021 Fold the fp32 sweep into the default driver path / CI table — so the quoted number can't drift back to the fp64 convention.

## Recovered — session 1792996176495\_5kxw5 (AWQ kernel margin)

# TODO — graphopt

## title: TODO type: task date: 2026-09-26

- [ ] T-034 agenda r6 s0.6: Uncaptured headroom on zoo:GatedDeltaRule: 260.755us ceiling, captured=-0.032, vs\_compile=0.9556 INVESTIGATED 2026-11-15: genuine residual — seqm kernel binds+verifies but whole-module delivery loses (0.33x compile); profitable prices claims in isolation, not assembled module. Lever = module-level costing/fusion. Report docs/reports/t032\_034\_frontier\_delivery.md · created:2026-11-13
- [ ] T-038 agenda paymeasured\_confound-6-14-pay-measured-rows-are-executor-confounded-t: 6/14 pay\_measured rows are executor-confounded — the marginal measures routing, not the fold · pri:high · created:2026-11-14 · #agenda
- [ ] T-039 agenda frontier\_delivery\_gap-uncaptured-headroom-on-zoo-chunkedssd-1543-317us: Uncaptured headroom on zoo:ChunkedSSD: 1543.317us ceiling, captured=0.0584, vs\_compile=None · pri:high · created:2026-11-14 · #agenda
- [ ] T-040 agenda frontier\_delivery\_gap-uncaptured-headroom-on-zoo-chunkedssd-1751-576us: Uncaptured headroom on zoo:ChunkedSSD: 1751.576us ceiling, captured=-0.0615, vs\_compile=0.286 · pri:high · created:2026-11-14 · #agenda
- [ ] T-041 agenda frontier\_delivery\_gap-uncaptured-headroom-on-zoo-decoupledmla-140-084u: Uncaptured headroom on zoo:DecoupledMLA: 140.084us ceiling, captured=-0.2396, vs\_compile=0.6241 · pri:high · created:2026-11-14 · #agenda
- [ ] T-042 agenda frontier\_delivery\_gap-uncaptured-headroom-on-zoo-deltaruleblock-163-41: Uncaptured headroom on zoo:DeltaRuleBlock: 163.416us ceiling, captured=0.0055, vs\_compile=0.7611 · pri:high · created:2026-11-14 · #agenda
- [ ] T-043 host-tensor weight cache: per-process safetensors tensor cache keyed by (path,name,dtype,offset); opt-in flag/env first · pri:high · created:2026-11-18 · #hnl-perf · links:PL-036
- [ ] T-044 host-tensor f32 compute path alongside f64; tensor.load decodes F32 sidecars without f64 upcast; f64 stays default · pri:high · created:2026-11-18 · #hnl-perf · links:PL-036
- [ ] T-045 f32 verify-run tolerance arm + re-verify promoted artifacts under f32 before shipping · pri:high · created:2026-11-18 · #hnl-perf · links:PL-036
- [ ] T-046 re-run zoo\_bench.py after each perf phase; update zoo\_speed\_report.md · created:2026-11-18 · #hnl-perf · links:PL-036
- [ ] T-047 product-proof round 2: re-bench whisper specs on fast path vs eager/compile CPU (gated on A) · created:2026-11-18 · #hnl-perf · links:PL-036
- [ ] T-048 permanent home + manifest for verified .hnl zoo artifacts (8.3GB in /tmp/arch-zoo is ephemeral) · created:2026-11-18 · #hnl-perf · links:PL-036
- [ ] T-049 docs/reports entry: honest .hnl speed table, correctness-verified vs speed-promising · created:2026-11-18 · #hnl-perf · links:PL-036
- [ ] T-050 agenda v3 frontier: ChunkedSSD/GatedDeltaRule/MambaSelectiveScan/DecoupledMLA/CapsuleRouting/RotaryPair carrier+lowering · created:2026-11-18 · #hnl-perf · links:PL-036

## HIGH — ADR-0159: dissolve the optimizer domains into ext tools

## Adopted on three triggers, not a date ( host/project/adrs/0159-… ):

- [ ] T-002 the domain's public surface unchanged for four weeks

## HIGH — M1: the machine finds the kernel, not me (five items, ~5 days)

## Done — 2026-10-27 (kernel margin, valid)

## Active — HIGH PRIORITY (the open questions, post-plan-0032)

## Active

- [ ] T-010 Generated kernels (genkernel) - novelty question measured; elementwise/decayed/reduction/seq/seqm tiers all priced through honest referees. Paying sites: pointwise folds corpus-bounded, reductions fp64+large-softmax only, seq scans 3 sites, seqm matrix-state 2 sites. Full lever history: retros/gen-kernels.md
- [ ] T-028 Stage-2 self-driving — domains-as-data, SECOND CUT LANDED 2026-11-12: board="torch" specs (corpus.modules selects play.\_torch\_modules zoo builders as data; shapes.widths/batch + hardware.device/dtype consumed into module/example tensors — cuda+float64 deliver verified; deliver reuses \_torch\_deliver); player={"kind","prior"} re-seats DomainSpec weights (priors/\*.json packaged, path, inline, or "none" cold) + bake\_prior() mints spec-matched tables via oracle-imitation on spec boards (~12s/200 cases) — MEASURED FLAT on pointwise (+0.173/+0.149 vs shipped +0.171/+0.180), kept opt-in; vocab.rules {"base","minus":\[names\]} subtraction profile (DEFAULT-minus empties meta-arena streams — documented); gen/other boards still code. Report: docs/reports/2026-11-12-domain-spec-second-cut.md . Retro: project/retros/domain-spec-second-cut.md . First cut: docs/reports/2026-11-12-domain-spec-domains-as-data.md . 2026-11-12 THIRD CUT: deliver is now a spec FIELD — "auto"|"none"|"torch" (bools spell auto/none; "torch" load-validated to board=torch; meta-arena auto stays hookless — symbolic terms have no lowering/verify target, documented); torch-silu.json declares "deliver":"torch", play --deliver verified (delivered+verified, max\_abs 1.19e-07). Report: docs/reports/p36\_seq\_view\_lift.md (Lever A rider) . Retro: project/retros/seq-view-lift.md . CODE-BOARDS CUT 2026-11-13 (docs/reports/2026-11-13-domain-spec-code-boards.md): ALL code-shaped boards spec-able — board=gen (corpus.builders + ambient handler\_tags subset + deliver auto->gen sink, verified 5.2e-06), joint (corpus.kinds=archetype digits, rules=DEFAULT-name filter), search (kinds=pool names, rules=ruleset filter, horizon/patience), fragment (kinds, rules=rewrite-name filter, handler\_tags=vocab table, max\_moves), open (corpus.ops/page/start/step), player (corpus.objective/max\_evals), bilinear (corpus.scramble); per-board \_CORPUS\_KEYS/\_VOCAB\_KEYS whitelists refuse unbindable fields at load; player/incumbent default+validate per board (\_BOARD\_DEFAULTS/\_BOARD\_INCUMBENTS); PLAYERS/INCUMBENTS gained the 5 non-meta specs + 4 playbooks lazily; play.py seams: \_gen\_cases(builders,ambient\_handlers), named \_search\_pool/\_fragment\_pool+kinds, parameterized \_search\_board/\_fragment\_board; packaged demos domains/gen-fusion.json + search-folds.json; bake\_prior refuses non-MetaArena boards. Remaining: attr values, statistical footprints; genuinely-new boards stay code (the 4 registries are the plugin seam). 2026-11-15 attr\_values SHIPPED (vocab.attr\_values bounds attr VALUES per op, load-error refusals, demo domains/attr-bounded.json, 88 spec tests green); statistical footprints documented as generator-side prerequisite (gen\_cases exposes no distribution surface) — report docs/reports/t028\_spec\_attrs.md.

## Deferred

## Tech debt to fix

## Recovered — session 1792950048869\_3wjyq (generated-kernel claims)

## Recovered — session 1792996176495\_5kxw5 (AWQ kernel margin)

## open

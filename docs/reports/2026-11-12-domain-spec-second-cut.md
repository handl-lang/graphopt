# 2026-11-12 — domains-as-data, second cut: torch board + spec priors

Follow-up to `2026-11-12-domain-spec-domains-as-data.md`.  The first
cut's honest gaps, re-measured and closed where closable:

## Gap 3 — per-spec player prior: CLOSED (machinery), measured FLAT

`player` now accepts `{"kind": "meta", "prior": <ref>}` where *ref* is

* `"none"` — the uniform cold start (`weights={}`);
* `"default"`/absent — the shipped `META_ARENA_PLAYER_WEIGHTS`;
* a packaged name — `mlopt_discovery/priors/<name>.json`
  (`priors/pointwise.json` ships as the worked example);
* a filesystem path to a `{feature: weight}` JSON, or an inline
  object in the spec itself.

`_resolve_player` rebuilds the `DomainSpec` over the same
parts/legal/features with only the weight table swapped — consumers
(`play --player linear`, `LearnedPlayer(spec=…)`, `frozen` arms)
see the spec's own prior through `Domain.spec.weights` as before.

`bake_prior(spec, cases=, seed=, budget=, imitate_epochs=,
rl_epochs=, init=)` mints a spec-matched table: `plan_oracles` +
`imitate_oracle` gained `board_of`/`handlers` parameters
(`meta_player.py`) so demonstrations are simulated on the spec
domain's *own* boards — handler subsets and supported bounds
included — and `eng.train_policy` adds optional REINFORCE epochs.
A 200-case / 3-imitate-epoch bake costs **~12 s** on this host —
auto-retrain at registration is *possible* but deliberately not
default: specs stay data, bakes stay artifacts.

**Measurement (honest: no win on pointwise).**  Held-out
`gen_cases`-equivalent stream (`seed+7919`, 40 cases, budget 12),
`LearnedPlayer` frozen arms:

| prior | learned | learned-greedy |
|---|---|---|
| shipped META (un-retrained) | +0.171 | +0.180 |
| `priors/pointwise.json` (spec-init imitation×3, 200 cases) | +0.173 | +0.149 |
| cold+RL bake (imitate×3 + REINFORCE×1) | +0.168 | +0.175 |
| `prior: "none"` (cold) | −0.509 | −0.600 |

The spec-matched bakes land inside the noise band of the shipped
prior — the mismatch the gap described is real in principle
(narrow vocab ⇒ different action distribution) but the baked
weights already transfer: features that can't fire are inert and
the `op:*`/`a:*` logits generalize.  Ship the machinery, keep
`pointwise.json` on the default prior, and treat per-spec bakes as
the remedy for specs whose eval actually diverges — the bake is
cheap enough to try per spec.

## Gap 1 — torch-backed boards: CLOSED for `board="torch"`

`play._torch_cases`' inline builders became
`play._torch_modules()` — a named zoo (`spelled_silu`, `mlp`,
`relu_lin`, `lin`, `softsign_mlp`) that `corpus.modules` selects
as data.  `_torch_cases` gained `modules`/`widths`/`batch`/
`device`/`dtype` keyword seams (the built-in `torch` domain passes
none — same stream, draw order preserved).  `domain_spec`
`BOARDS` now seats `"torch"`: cases are the lifted-module tuples,
`board_of` is `MetaArena` under `supported=TorchSink().supported_ops`
plus the spec's handler subset, and `deliver` reuses
`play._torch_deliver` — the game score lowers+verifies a real
module.  `corpus.kinds`/`corpus.modules` are validated per board
(each errors on the other's board).  Packaged demo:
`domains/torch-silu.json`.

Verified: `play --domain-spec torch-silu --deliver` →
`{'delivered': True, 'verified': True, 'max_abs': 1.19e-07}`.

**Still code, honestly:** `board="gen"` — its cases mint per-case
genkernel handlers (`_mint_tiers`, cost-gated claims, hybrid sink)
far beyond what JSON can name; `open`/`player`/`bilinear`/`search`/
`fragment`/`joint` boards likewise.  The zoo is five modules —
growing it is adding a builder to `_torch_modules`, which is code
by design (the spec names, it doesn't write modules).

## Gap 2 — shapes/hardware/dtype/device: CLOSED on torch, recorded on meta-arena

* `shapes.widths` (positive-int list) and `shapes.batch` size the
  module draw and example input; `hardware.device` /
  `hardware.dtype` (`float32|float64|float16|bfloat16`) place the
  model *and* input tensors before `TorchSource.to_ir`, so leaves,
  IR params and the deliver path all agree.
* Measured: `device="cuda"` and `dtype="float64"` spec domains
  build, play and **deliver verified** (max_abs 0.0 on the smoke
  cases) — the RTX 2050 is along for the ride.
* On `meta-arena` the fields stay recorded-only *by construction*:
  `gen_cases` spells fixed `(4, 4)` vars and no tensor is ever
  materialized — there is nothing for a shape distribution to
  bind.  `spec_summary` now reports `shapes`/`hardware` at top
  level instead of a `recorded_only` bucket (they're consumed
  where a board can consume them).

## Gap 4 — rules subtraction profile: CLOSED

`vocab.rules` accepts `{"base": <profile-or-names>, "minus":
[names]}` — the base allowlist minus the named rules ("all"
resolves against the 154-name scan universe).  Semantic honesty:
subtracting a DEFAULT law empties meta-arena streams (every case
carries the whole DEFAULT nameset) — minus is for carving scan
extensions or explicit subsets, and the torch board refuses
profiles that exclude DEFAULT up front rather than emitting an
empty stream.

## Still open (documented, not faked)

* `attrs` whitelist remains `(op, attr-name)` — attr *values*
  unconstrained.
* Footprint sampling still statistical (240 cases); per-case
  post-filter remains the guarantee.
* `gen` and the other code-shaped boards stay un-spec-able.
* No deliver hook on meta-arena (symbolic terms have no lowering).
* `corpus.coverage` is ignored on the torch board (no gen_cases
  seam) — recorded, harmless.
* Auto-retrain-at-registration exists (`bake_prior`) but is
  opt-in; no spec ships a bake because pointwise measured flat.

## Files

* `packages/mlopt-discovery/src/mlopt_discovery/domain_spec.py` —
  torch board, player.prior, rules-minus, `bake_prior`/`load_prior`,
  summary/CLI updates.
* `packages/mlopt-discovery/src/mlopt_discovery/play.py` —
  `_torch_modules()` zoo + `_torch_cases` spec seams.
* `packages/mlopt-discovery/src/mlopt_discovery/meta_player.py` —
  `board_of`/`handlers` params on `oracle_line`/`plan_oracles`/
  `imitate_oracle`.
* `packages/mlopt-discovery/src/mlopt_discovery/domains/torch-silu.json`,
  `packages/mlopt-discovery/src/mlopt_discovery/priors/pointwise.json`
  (new packaged artifacts).
* `tests/test_domain_spec.py` — 58 tests (was 28).

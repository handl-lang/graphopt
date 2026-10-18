# Retro — cached fixpoint + the enablement floor, measured

**Date**: 2026-10-19.  **Follows**: `fixpoint-api-and-conv-sweep.md`
(its two open items).  **Commits**: `02e05f9` (rulecache knobs +
payload-internal parents), `ca42c16` (guard transport for stored
composites), this commit (probe + retro + GOALS).

## Item 1 — `synthesize_rules_cached` takes the fixpoint API

`rounds` / `max_emitted` / `on_round` now pass through
`synthesize_rules_cached` into `meta.synthesize_rules`, and both scalar
knobs join `cache_key`'s `params` — a `rounds=3` mint is a different
problem instance from `rounds=1` and must not collide.  `on_round` is a
callable: it enters the key through its `_hook_sig` proxy (source text
when inspectable, else module+qualname) — a *vetoing* observer
truncates the fixpoint, so sharing the unobserved entry would be a
wrong hit; an uninspectable signature degrades to a miss.  Documented
hit semantics: the observer reports **live rounds only** — a warm load
calls nothing.

Also landed, from the same retro's "unbuilt" list: `RuleCache.load`
resolves parents *payload-internally* — records decode in stored order
and each decoded rule joins the resolver, so a round-≥2 composite whose
parent is itself a `syn_*` mint round-trips self-contained (emission
order guarantees parents precede dependents; all-or-nothing miss
semantics unchanged).

Tests (`tests/test_rulecache.py` §e, 4 tests): cached `rounds=3`
returns the fixpoint strata under a monkeypatched-exploding
synthesizer; `rounds=1`/`rounds=3`/`max_emitted=2` are distinct
entries; the observer fires on cold mints only and a vetoing callable
keys separately; a guarded `syn_*`-parented composite reloads its guard
from the payload.

## Item 2 — the enablement floor, measured

The named blocker is **fixed**: guard transport for stored composites.
`store_object` writes the minted rule's `guard_pats` + `parents` as a
`"guard"` record field (the `rulecache` binding wire form — additive;
older readers ignore it); `admit_object` resolves the named parents as
rows of the same store *transitively* and re-runs
`meta._compose_guards`.  `missing_hooks`/`serializable` keep their
meaning (what the bare record rebuilds); admit annotates the record
with `"transported"`, and `full-data` counts missing-minus-transported.
A parent that rebuilt weaker than it declared refuses the transport —
composing over it would smuggle a weakened guard.  5 tests in
`tests/test_discovery_gauntlet.py` cover spec-write, restore+fire,
transitivity, the honest partial admit, and the gate.

### The premise-pay measurement (`lab/probes/conv_fold_premise_pay_probe.py`)

The conv+BN fold composite mints in 4 rounds (974 emitted), premise
closure = 8 rules (`batch_norm_expand` + 4 `conv_channel_*` anchors + 3
syn intermediates).  Stored, admitted — `transported=['check','derive']`
— and measured on the real eval-mode `ConvBN` site:

| stage | measured |
|---|---|
| transport | hooks rebuilt transitively; composite fires on the seed |
| truth (real region) | accepted=1, **1eq/0ne** — fp64-equal on the real site |
| truth (premise-anchored) | `verify_law(seed, folded, ALL_RULES+closure)`: **derivable, replayable, 2-step cert** (`conv_channel_shift_sp` + the depth-3 syn composite); under `ALL_RULES` alone: not derivable |
| **pay** | `fires=1`, typed=1, ill=0, verified=`same` — and **changed=False**: extraction kept the spelled term (`spelled=34874.3` vs `fused=34874.337`) |

**The dependent mint does not pay in extraction.**  The composite's
fused member exists in the e-class but `dag_cost` prices its RHS's
spelled weight-fold trees (`mul`/`reshape`/`rsqrt`/`add` over the bound
params) as real ops — the constant-fold win materializes only at
*lowering* (`IRModule._fold_weight_chains`), a layer the term-level
cost model cannot see.  The premise-of-admitted-composite accounting is
now implemented and ran; the pay leg is measured **false**, not
assumed.

Caveat on the formal floor: the full `run_gauntlet` on the composite
did not terminate in `pl.measure` on this box (>9 min even on a
one-case corpus — the 83-rule `_reach`/derivability saturation is a
scaling gap, separate from starvation).  The per-stage numbers above
are the bounded measurement.

### Verdicts per family

- **conv anchors in the closure** (`conv_channel_shift_sub_sp`,
  `conv_channel_scale_sp`, `conv_channel_scale_l_sp`,
  `conv_channel_shift_sp`) + **`batch_norm_expand`**:
  **enabled-but-opt-in** — a dependent mint exists, fires, and
  truth-verifies (real region + premise-anchored replayable cert), but
  pays nothing in extraction → the floor's pay leg fails → **not
  DEFAULT**.  Enablement (the mint exists at all) is real; pay is the
  measured refusal.
- **the other 56 CONV_FOLD_RULES**: named refusal — truth-starved AND
  no paying dependent mint measured.
- **the composite itself**: now storable/admittable via transport and
  premise-derivable, but does not pay → not a DEFAULT candidate.
- **BATCH_NORM_RULES** (the other arities): unchanged — `fires=0`
  variants have no dependent mint; `batch_norm_expand` shares the
  closure's account.

## What stays open

- The corner-reaching truth route (enumeration vocabulary for
  `conv2d`/channel-views, or witness-driven sites) — requirement #1
  still stands; the premise-anchored `verify_law` replay is the honest
  partial substitute measured here.
- `run_gauntlet`/`pl.measure` on guarded minted composites does not
  terminate within the box — the reach/derivability saturation needs a
  bounded budget for composite-scale rules.
- A cost model that credits compile-time weight folds (the fused
  member's spelled weight trees are free after
  `_fold_weight_chains`) would flip the measured pay leg — that is a
  *cost-model* gap, not a law-truth gap.

`DEFAULT` unchanged.  Gate account: ruff/format/ty clean on touched
files; `tests/test_rulecache{,_codec}.py`, `test_meta_fixpoint.py`,
`tests/test_discovery_{gauntlet,evidence}.py` all pass (34 + 58 + 44;
the standalone `test_rulecache` fp64 tests still need the suite's
torch-import order — pre-existing at HEAD, verified).

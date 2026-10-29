---
title: Plan 0009 — Composable rule sets
type: plan
date: 2026-10-14
status_note: accepted
---

# Plan 0009 — Composable rule sets

Depends on **0006** (which puts `RuleSetLike` / `RuleLike` in
`catopt_core.ports`). Feeds **0010** (the scheduler consumes rule
priority + tags).

## Vocabulary

The codebase mixes "law" and "rule"; this plan fixes the meanings:

* a **law** is the mathematical identity (the provenance text —
  `Rewrite.law`);
* a **rule** is one rewrite direction/instance of a law
  (`Rewrite`);
* a **`RuleSet`** is a named, composable collection of rules.

`RuleSet` (concrete) satisfies the `RuleSetLike` protocol — the
renamed `LawSet` (see *Decisions*).

## Goal

Make the rewrite rules a **first-class, composable value** instead
of loose module-level lists plus a `ruleset: str` switch and two
hidden filters inside the pipeline. Same principles as the API work
— explicit, composable, protocol-based — with one deliberate
difference: rules are the library's *content*, so a **default preset
exists** (unlike the backend, which must be chosen explicitly).

## Principles

1. **Explicit.** No hidden filtering. Today `optimize_model` silently
   drops `_SUBSUMED` rules and applies `_EXPANSIVE_RULES` budgets
   (`optimize.py:742-769`). Under this plan the *preset* says what is
   in; budget is rule metadata and priority is `RuleSet` metadata.
2. **Composable.** Sets compose with an algebra, like criteria:
   `DEFAULT + SYMMETRY`, `FULL - SYMMETRY`, `CARRIERS & FUSION`.
3. **Protocol-based.** A `RuleSet` satisfies the `RuleSetLike`
   protocol in `catopt_core.ports` (an iterable of `RuleLike`) — the
   renamed `LawSet`. No new port is needed.
4. **Metadata lives with the rules.** Tags and budget are intrinsic
   rule attributes (a rule *is* symmetry/expansive/subsumed);
   priority is scheduling policy and rides on the `RuleSet`.
5. **A default preset.** `DEFAULT` is the recommended set; comm/assoc
   stay **available but opt-in**, never default.

## Design

### Rule metadata (`Rewrite`)

```python
@dataclass(frozen=True)
class Rewrite:
    ...
    tags: frozenset[str] = frozenset()
```

`Rewrite` stays **pure**: tags describe the rule's *nature*
(intrinsic); priority is scheduling policy and lives on the
`RuleSet` (below).

Tags are **constants**, never bare strings — so a typo cannot
silently miss a budget or a preset, and tags stay discoverable:

```python
# catopt_core.laws.tags
SIMPLIFICATION = "simplification"
CATEGORICAL    = "categorical"
FUSION         = "fusion"
CARRIER        = "carrier"
SYMMETRY       = "symmetry"
EXPANSIVE      = "expansive"
SUBSUMED       = "subsumed"
LAYOUT         = "layout"
SCAN           = "scan"
DECODE         = "decode"
```

* `SYMMETRY` — comm/assoc and the scale-hoist naturality rules:
  the Catalan-blowup closure generators.
* `EXPANSIVE` — anything whose closure is combinatorially
  explosive (superset of `SYMMETRY`; today's `_EXPANSIVE_RULES`).
* `SUBSUMED` — rules the non-local pairing pass replaces
  (`swiglu_fuse`, `qkv_fuse`, …; today's `_SUBSUMED`).

### `RuleSet`

```python
@dataclass(frozen=True)
class RuleSet:
    """A named, composable collection of rewrite rules."""
    name: str
    rules: tuple[Rewrite, ...]
    priorities: Mapping[str, int] = field(default_factory=dict)
    description: str = ""

    def __add__(self, other) -> RuleSet:   # union, by rule name
    def __sub__(self, other) -> RuleSet:   # subtract, by rule name
    def __and__(self, other) -> RuleSet:   # intersection, by name
    def named(self, *names) -> RuleSet:    # subset by rule name
    def tagged(self, *tags) -> RuleSet:    # subset by tag
    def priority_of(self, rule) -> int:    # default NORMAL
    def with_priorities(self, **prio) -> RuleSet:
    def __iter__(self) / __len__ / __contains__
```

Priority is a `rule name -> int` map on the `RuleSet`
(`EARLY = 0` / `NORMAL = 10` / `LATE = 20`); `Rewrite` does not
carry it, so the e-graph type stays free of scheduling concerns.

Composition is by **rule name** (stable identity), so
`(A + B) - B == A` for disjoint names and `A + A == A`. Duplicate
names raise (`override=` replaces deliberately).

### Known limit: scoped rules

A `RuleSet` is a flat list — there is no way to say "these rules
only fire at `linear` ops" or "only at the root". That is a real
extensibility limit for carrier- or region-specific rules. The
placeholder is a `scoped(op=..., rules=...)` combinator (membership
guarded by a head-op), but it is **not implemented in this plan** —
recorded here so the limit is explicit and the combinator has a
home.

### Presets

Core presets (in `catopt_core.laws`):

| preset | contains |
|---|---|
| `SIMPLIFICATION` | the simplification singletons |
| `CATEGORICAL` | the categorical laws |
| `FUSION` | structural folds (qkv / swiglu / sdpa) |
| `SYMMETRY` | comm/assoc + scale-hoist (opt-in) |
| `CARRIERS` | om / trace / xc / decode (from carrier packages) |
| `CARRIER_SEARCH` | today's `CARRIER_LAWS` — the regime search set |
| `WITH_LAYOUT` | today's `ALL_RULES_WITH_LAYOUT` |
| `DEFAULT` | `CATEGORICAL + SIMPLIFICATION + FUSION + CARRIERS`, minus `"subsumed"` and `"symmetry"` |
| `FULL` | `DEFAULT + SYMMETRY` |

`DEFAULT` is what the pipeline uses when none is given. `SYMMETRY`
is never in `DEFAULT` — it is `DEFAULT + SYMMETRY` when a caller
opts in.

### Pipeline surface

`search(..., rules: RuleSet = DEFAULT)`. The `ruleset: str` switch
is retired; a string may still resolve to a preset for convenience
(`RuleSet` lookup by name), but the value is first-class.

### Budgets and scheduling

The `_EXPANSIVE_RULES` set becomes the `EXPANSIVE` tag; the
pipeline applies the per-rule enode budget to **tagged** rules
(`rules.tagged(EXPANSIVE)`) rather than to a hardcoded name list.
Priority (`RuleSet.priorities`) is what 0010's scheduler orders by.

### Cross-package composition

Carrier packages own their rule sets (`OM_LAWS`, `TRACE_LAWS`,
`XC_LAWS`, `DECODE_LAWS` → `RuleSet` values). Core cannot import
them, so the **orchestrator** composes the effective default:

```python
# catopt_orchestrator
DEFAULT_RULES = (core.DEFAULT + carriers.CARRIERS)  # composed once
```

### Entry → preset

The two pipelines use different sets today: `optimize_model` runs
core `all_rules()` (minus the hidden `_SUBSUMED`), while `regime`'s
`build_egraph` runs `CARRIER_LAWS`. Make the choice explicit and
aligned:

| entry | preset |
|---|---|
| `search` (Monolithic / Compositional) | `DEFAULT` |
| `regime_dispatch` / `regime_frontier` | `CARRIER_SEARCH` |
| `Autotuned` | the preset of the search it reuses |
| explicit | whatever `RuleSet` the caller passes |

`CARRIER_SEARCH` and `WITH_LAYOUT` are the existing `CARRIER_LAWS` /
`ALL_RULES_WITH_LAYOUT` promoted to presets; the orchestrator
composes them.

## What changes from today

| today | after |
|---|---|
| `ALL_RULES` / `all_rules()` list | `RuleSet` values |
| `SIMPLIFICATION_RULES`, `CATEGORICAL_RULES`, `SDPA_FOLD_RULES`, `SCAN_LAWS`, `SCAN_DIAG_LAWS` | named presets |
| `ruleset="all" \| "simpl" \| "categorical"` | `rules: RuleSet` |
| `_SUBSUMED` hidden filter | `SUBSUMED` tag, excluded by `DEFAULT` |
| `_EXPANSIVE_RULES` hidden budget set | `EXPANSIVE` tag |
| `LawSet` / `RuleProvider` protocols | `RuleSetLike` / `RuleSetProvider` |
| carrier law lists | carrier `RuleSet`s, composed by the orchestrator |
| comm/assoc always on | opt-in via `DEFAULT + SYMMETRY` |

## Relationship to 0010

* 0010's **scheduler** orders rules by the `RuleSet`'s priority map
  and the rule set's tags — no engine-side name lists.
* 0010's **compiled matcher** compiles a `RuleSet` once; the set is
  the compilation unit.

## Execution stages

1. Add `tags` to `Rewrite`, the `priorities` map to `RuleSet`, and
   the `tags` constants module; rename `LawSet` → `RuleSetLike`
   (`RuleProvider` → `RuleSetProvider`); tag every existing rule
   (mechanical, name-keyed).
2. `RuleSet` dataclass + algebra + `named`/`tagged`.
3. Presets in `catopt_core.laws`; retire the loose group lists and
   `all_rules()` (keep thin aliases during migration).
4. Carrier packages expose `RuleSet` values.
5. `search(rules=...)`; remove the `ruleset` string and the hidden
   `_SUBSUMED` / `_EXPANSIVE_RULES` filters; budgets via
   `tagged("expansive")`.
6. Orchestrator composes `DEFAULT_RULES`; migrate call sites.

## Tests & gates

* Algebra laws: union/subtract/intersect by name; `A + A == A`;
  `(A + B) - B == A`; `tagged`/`named` subsets.
* `DEFAULT` excludes `"symmetry"` and `"subsumed"`; `FULL` includes
  everything; `DEFAULT + SYMMETRY` is `FULL` for the core set.
* Behaviour parity: the effective default reproduces today's
  `optimize_model` rule selection (minus the now-explicit filters).
* Coverage 100%; `ty` clean; ruff `D`; radon; vulture; import-linter
  (core defines presets, imports no carrier).

## Risks

* **Name collisions across packages** — composition is by rule name;
  a carrier rule sharing a name with a core rule must be detected
  (warn or error) rather than silently shadowed.
* **Parity** — `DEFAULT` must match today's behaviour; the pairing /
  subsumed relationship is subtle and needs an explicit test.
* **Preset sprawl** — keep the preset list small and documented; new
  presets should be compositions of the existing ones, not new
  hand-maintained lists.

## Decisions (resolved)

1. **Tags on `Rewrite`.** Rules genuinely belong to several groups (a
   scale-hoist rule is both `SYMMETRY` and `EXPANSIVE`;
   `swiglu_fuse` is both `FUSION` and `SUBSUMED`), so membership
   is by tag; presets are built with `tagged(...)` and composed with
   the `+`/`-`/`&` algebra. No parallel set-only mechanism.
2. **Priority is an `int` map on the `RuleSet`, with named
   constants.** `RuleSet.priorities: rule name -> int`, lower =
   earlier; most rules stay `NORMAL`. Ship `EARLY = 0` /
   `NORMAL = 10` / `LATE = 20` as module-level ints so the common
   cases are self-documenting while finer values stay available.
   `Rewrite` does not carry priority (no scheduler coupling).
3. **`core.DEFAULT` + orchestrator `DEFAULT_RULES`.** Core cannot
   import carriers (hexagon), so `core.DEFAULT` covers core rules and
   the orchestrator composes `DEFAULT_RULES = core.DEFAULT +
   carriers.CARRIERS` — the value `search` defaults to.
4. **Duplicate rule names in `+` raise.** Silent shadowing across
   packages is a real bug source and "explicit, no hidden behaviour"
   is the design's spine. An `override=` argument is the deliberate
   escape hatch.
5. **Port rename: `LawSet` → `RuleSetLike`** (and `RuleProvider` →
   `RuleSetProvider`), so the protocol reads symmetrically with
   `RuleLike` and the concrete `RuleSet`. The `law` / `rule`
   distinction is defined under *Vocabulary*.

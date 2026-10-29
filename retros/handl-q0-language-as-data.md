---
title: 'Retro — spike Q0: can catopt''s language be handl data?'
type: retro
date: 2026-10-14
---

# Retro — spike Q0: can catopt's language be handl data?

**Plan**: 0029 (the handl ↔ catopt boundary
spike), **Q0**.  **Verdict**: **PASS**, with two representational
requirements (both fixed) and one named residue.

**Evidence**: catopt `8d5b47d` (`lab/handl-spike/dump_fragments.py` +
the generated `fragments/`), the emitter fix in `8de8cb5`; handl
`b20b978a` (`crates/handl-kit-data-canonical/tests/catopt_fragments.rs`
+ `tests/data/catopt-fragments/`).

## The question

ADR 0010 claims handl *is* the kernel — catopt's vocabulary should be
handl data, not merely similar to it.  Q0 is the cheapest decisive test:
encode one catopt law as a `TransformDef` (+ the shapes it needs), load
it through handl's `language_from_dir`, and check it survives.

## Method

1. Compare catopt's emitted rows (`catopt_discovery.handl`) against
   handl's actual types, field by field — `TransformDef`, `TransformRule`,
   `RewriteStep`, `TransformCondition`, `AstShapeDef`.
2. Materialize the seam: `lab/handl-spike/dump_fragments.py` writes
   catopt's vocabulary into the **ten-file layout** `language_from_dir`
   reads (eight required shell files + `transforms.yaml`).
3. Load it from Rust (handl's test) and assert the merge + the fields.

## Result — PASS

```
test catopt_fragment_dir_assembles_a_language ... ok
test catopt_law_survives_as_data ... ok
```

A `Language` assembles from catopt's data; the law is retrievable by id
(`transform.catopt.assoc_add`), its `pass`/`node_kind` are intact, and
its folded `when` deserializes as `And([KindAt, ChildCount])`.

### Field-by-field: everything matched except two spellings

| catopt emits | handl expects | verdict |
|---|---|---|
| `id` · `pass` · `node_kind` | same | ✅ |
| `rule: {Declarative: {target_kind, children}}` | `TransformRule::Declarative` | ✅ |
| `KindAt{index,kind}` · `ChildCount(n)` | `TransformCondition` variants | ✅ |
| `KeepPath([...])` · `Create{kind, children}` | `RewriteStep` variants | ✅ |
| `children: [{role: arg0}, …]` · `payload_schema` | `AstShapeDef` fields | ✅ |
| the `catopt:` provenance block | no such field | ✅ (serde ignores unknown fields) |
| **`when: [clause, …]` (a list)** | `Option<TransformCondition>` — **one** | ❌ → fold into `And([...])` |
| **`{"Declarative": …}` (JSON map)** | **`!Declarative` (YAML tag)** | ❌ → emit tags |

### The two requirements, both now satisfied

1. **`when` is a single condition.**  handl's field is
   `Option<TransformCondition>`; a conjunction is spelled `And([...])`.
   catopt emitted a bare clause list, which does not deserialize.
   Fixed in the emitter (`8de8cb5`): omitted when empty, the clause
   itself when there is one, `And([...])` otherwise.
2. **Enum variants are YAML tags.**  Under `serde_yaml` a variant is
   `!Declarative`/`!Create`/`!KeepPath`/`!KindAt`/`!And`; under
   `serde_json` it is a map.  `language_from_dir` reads YAML, so the
   dump tool emits tags (measured failure: *"invalid type: map, expected
   a YAML tag starting with '!'"*).  Fixed in the dump tool.

### Named residue (not exercised here)

- **The leaf step has no counterpart.**  catopt renders a literal leaf
  as `{"catopt:leaf": …}`; handl's `RewriteStep` has no such variant.
  `assoc_add` is structural, so it never arises — but any law with a
  constant on the rhs would need a mapping (candidate: a `Create`d node
  carrying the value in its `payload`, or `Fresh(hint)` for variables).
- **The shell is a stub.**  catopt supplies `ast.yaml` (op shapes) and
  `transforms.yaml` (laws) of the ten files; the eight required shell
  files are empty today, because catopt's "syntax" is the IR, not a text
  language.  A *real* catopt language would need `eval`/`types` entries —
  which is precisely ADR 0010's "interpretations" question, not Q0's.

## What Q0 does **not** prove

- ~~**That a rewrite fires.**~~  **Measured later the same night** (plan
  0030 step 1): a catopt law *does* fire through handl's
  `RewriteEngine` — one rewrite, and the result is
  `add(add(a, b), c)`, which is exactly what catopt's own
  `EGraph.apply_rule` produces for the same term (handl `82194954`).
  What remains unproven is only the *peer*: nothing in catopt calls
  handl yet.
- **Anything about tensor values.**  That is Q1 (a host function
  carrying a tensor), and it is the question ADR 0010's "host functions"
  claim actually rests on.
- **That the load path is reachable from the CLI.**  It is not:
  `hnl` loads kits through registered *emitters*; a data directory is
  reachable only through the `language_from_dir` API (or a small host
  that calls it).  `FragmentSource::DataFile` is modeled but returns
  `LangError::NotImplemented` — that is the *runtime* half of the seam,
  and it is Q3's business.

## Cost

Hours, not days — because `hnl` is already installed and the load path
already exists.  No Rust code was written on the catopt side; the only
handl-side change is a test.

---

## Addendum (2026-10-14) — the FULL bank: the PASS is qualified

The verdict above rests on a **one-law slice** (`assoc_add`, purely
structural).  Loading the **whole shipped bank** — all 83 laws, 161 KB of
`transforms.yaml` — tells a different story:

```
transforms.yaml: .[14].rule.children[1]: invalid type: map,
expected a YAML tag starting with '!'
```

Row 14's rule has a **leaf** on its right-hand side.  catopt spells a
leaf `{"catopt:leaf": …}` (its own extension — handl's child vocabulary
has no literal-leaf step, which the first version of this retro recorded
as a *residue*); the bridge emits a map there, handl expects a tag, and
the load fails.

**So the honest statement is:**

> catopt's vocabulary is expressible as handl data **for structural
> laws**.  Leaf-bearing laws need a mapping first.

Candidates for that mapping, both using vocabulary handl already has:

| leaf | handl step |
|---|---|
| a **variable** | `Fresh(hint)` — "allocate a fresh `var` leaf with a unique, non-colliding name" |
| a **constant** | `Create { kind: <const-kind>, payload: <value>, children: [] }` — provided the shape table declares a const kind |

Both require the shape table to name the kinds involved, which is a
*data* change (the direction ADR 0010 already sets), not an engine one.

**The gap is pinned, not hidden**: handl's
`catopt_full_bank_leaf_gap_is_pinned` asserts the failure and its shape,
and panics with the transform count if the bank ever starts loading —
i.e. it flips when the mapping lands.  Fixture:
`crates/handl-kit-data-canonical/tests/data/catopt-fragments-all/`
(generated by `lab/handl-spike/dump_fragments.py --all`).

**Cost of the correction:** one extra measurement (the full-bank load),
and it changes the spike's read on Q0 from "expressible" to "expressible
for the structural majority; one named mapping stands between the bank
and the kernel".  That is a better answer than a PASS that only held for
the case that was easiest to pass.

---

## Closure (2026-10-14) — the gap is shut; the whole bank loads

The leaf mapping landed, and the bank revealed **one more** gap behind
it: handl keys a rule by `pass:node_kind:priority`, and catopt has many
laws per root kind, so they collided (`comm_add` vs `assoc_add`).  Both
are now closed on the producer's side:

| gap | mapping |
|---|---|
| leaf (`{"catopt:leaf": …}` — a map where serde wants a tag) | `!Fresh` for an unbound pattern variable; `!Create` of a declared `catopt:const` / `catopt:var` / `catopt:data` kind, with the data in the payload |
| rule collision (`pass:node_kind:priority`) | distinct priorities per `(pass, node_kind)` — a *load-time disambiguator*, since catopt's saturation is unordered |

handl's `catopt_full_bank_loads` (the pinned gap, flipped) asserts
**83 laws** in `lang.transforms` and that the three leaf kinds are
declared.  Catopt `5ca8aef`; handl `9372bd32`.

**So Q0's verdict is now unqualified:** catopt's whole law bank is
expressible as handl language data and loads into a `Language` — not
just the one-law slice the first measurement used.

Two lessons worth keeping, both about *where* a gap lives:

1. **The first gap hid the second.**  Fixing the leaf took one edit; the
   collision only appeared once the leaf stopped failing first.  A bank
   is not a slice, and a pinned-gap test is what makes the second one
   visible instead of mysterious.
2. **`priority` is the interesting one.**  It is the first place where
   handl's schema demanded something catopt does not *have* (an order),
   and the honest answer was to supply a disambiguator and say plainly
   that it means nothing semantically.  A future producer with a real
   ordering should map it to something that means something.

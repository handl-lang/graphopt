# Family synthesis — minting the grammar's families, not just bodies

`handler_synth` mints handler *bodies* inside a family; the families
themselves (`ew`, `gather`, `contract`, `fold*`, `reduce`, `norm`,
`postnorm`) are hand-authored Python. `family_synth.mint_families`
mints the family *descriptor* — plain data that parameterises the
generic `("family", desc, sub)` body arm of `eval_body`:

```
{gate, leaves, broadcast, requires, orders}
```

- `gate` — the spec ops the family re-expresses (a spec's claim must
  be `⊆ gate ∪ ELEMENTWISE ∪ broadcast`);
- `leaves` — leaf name → `{kind, from, binds}`: the family vocabulary.
  `kind: "read"` (an index-read carrying the spec's literal attrs)
  and `kind: "map"` (a metavar-args map); `binds` names the ambient
  op the leaf re-seats — resolved through the same bindings table as
  every descriptor (`_family_bindings`, the generalised
  `_gather_bindings`);
- `broadcast` — the view ops its trees may interleave;
- `requires` — the marker leaf a translated tree must carry (`"*"`
  = any leaf): the no-tautology gate, `_gather_marked`
  parameterised;
- `orders` — the read/map composition orders its seeds emit:
  `read_then_map` (the fused `aff(at(x), w, b)` — project only the
  gathered slice) first, `map_then_read` (the denotation-equal
  `at(aff(x, w, b))`) as fallback — gather's own discipline.

## The search — enumerated, not derived

The spec set's *movable* ops (`_spec_ops - ELEMENTWISE -
BROADCAST_OPS`) bound the enumeration. Per candidate: non-empty gate
subsets (smallest first) × per-op leaf-form assignment from
`LEAF_FORMS` (`at` re-seats `select`, `aff` re-seats `linear`) ×
`broadcast` ⊆ `BROADCAST_OPS` × `requires` ∈ the descriptor's leaves
∪ `{"*"}`. A gate op with no matching leaf form closes the gate.

Each candidate family translates every spec into `("family", desc,
sub)` bodies — coordinates *read off the spec* — and faces the
`mint_handlers` fp64 contract: same seeded draws, same `allclose`
gate, same adversarial re-draws. A family mints iff ≥1 spec verifies
— mint-a-label is refused.

## The measurement — the v3 refusal class

Input: the four `select`/`linear`-bearing specs the grammar-v3 prim
mint refused (the `prim_ops.*.handler` rows of
`results_overnight_prim_v3.json`, spec + bound-site domain embedded).

```
searched: 28 candidate families (budget 256) · movable: {linear, select}
minted: 10 · coverage: 4/4
```

| family | gate | broadcast | requires | covers |
|---|---|---|---|---|
| fam_628568cd | {select} | {unsqueeze} | `at` | spec 3 |
| fam_8c9a7ed8 | {linear, select} | {squeeze, unsqueeze} | `at` | **all 4 — gather-equal** |
| fam_f365bbe7 | {linear, select} | {unsqueeze} | `at` | all 4 |
| fam_eeca1c15 | {linear, select} | {unsqueeze} | `aff` | specs 0–2 |
| fam_9ad0ae6b, fam_65145d6f | {linear, select} | {unsqueeze} / both | `*` | all 4 |
| fam_3abec455 | {linear, select} | both | `aff` | specs 0–2 |
| fam_f1498f50, fam_566641e1 | {select} | {unsqueeze} / both | `*` | spec 3 |

(`*` = marker-any; the remaining 18 candidates were `no-claim` —
their gate/broadcast subset could not translate any spec.)

**Gather is recovered.** `fam_8c9a7ed8` is the gather family as a
descriptor: gate `{linear, select}`, leaves `at: select` (index-read)
+ `aff: linear` (per-position affine), broadcast `{squeeze,
unsqueeze}`, marker `at`, both orders — exactly what v4 hand-wrote,
and its translated bodies are *identical* to `_gather_seeds`' (the
parity is a test). `fam_f365bbe7` is the same family minus `squeeze`
(a view no spec here uses); `fam_628568cd` is a **distinct covering
family** — the `{select}`-gate sub-family that mints the
select×select spec alone (it leads by enumeration order: smaller
gate first). `requires="aff"` mints the strict sub-family covering
the three affine-bearing specs; `requires="*"` the same coverage with
a broader claim (it would also claim affine-only specs — none exist
here).

## The injection seam

Minted descriptors serve the specs the shipped families refuse with
no new machinery: `propose_bodies`/`mint_handlers` accept
`extra_families`. On the four refusals, `mint_handlers(items,
families=(), extra_families=fams)` mints all four at `tried=1`,
`family: family:fam_*`, fp64 `max_err` 0.0–5.3e-15 — the same bodies
v4's `gather` minted, evaluated through the generic arm.

## What's verified vs rename-equivalent — the honest line

- `read_then_map` is the measured claim: `aff(at(x), w, b)` is a
  genuinely different evaluation order — it projects only the
  gathered slice, equal to `select∘linear` exactly when the read axis
  is not the contracted feature axis. Specs 0–2 minted on the fused
  order (`fused=True`; max_err up to 5.3e-15, not construction-zero).
- `map_then_read` is rename-equivalent — a re-seated leaf evaluates
  identically to the spelled op, so it verifies by construction.
  Spec 3 (select×select, no affine to fuse) minted on `fused=False`
  — the same honesty boundary gather itself accepted at err 0.0.
  The per-spec `fused` flag and `orders_verified` provenance record
  which is which, so a family that mints only on renames reads as
  such.
- The marker gate is the other honesty bar: a leaf-less translation
  (a pure-elementwise spec) is never claimed — no tautology under a
  family label.

## Limits — what's enumerated vs what's derived

- **The leaf forms are the inventory, not the discovery.** The
  machine enumerates which of `LEAF_FORMS` (`at`, `aff`) form a
  covering family for the refused op set; it does not invent new leaf
  semantics. A movable op no leaf form re-seats (`matmul`, `slice`,
  `embedding`, …) is an honest `no-claim` — a meta-grammar extension,
  not a candidate. The next real extension would add a leaf *form*
  (e.g. a span-view leaf for `slice`), still as data.
- **The gate is enumerated over the spec set's own ops** — a family
  is minted per refusal class, not as a universal vocabulary.
- **The candidate space is small by construction** — 28 candidates
  for a two-op movable inventory; it grows as `2^|movable|` gate
  subsets × leaf assignments × `2^|broadcast|` × markers. The
  measured corpus walls stay shallow because refusal classes carry
  few movable ops; a wider inventory is a budget question, not a
  soundness one.
- **`orders` is fixed, not searched** — both orders ride one
  descriptor (the family's translation discipline), so the
  enumeration covers *which* ops/leaves/views/marker, while the
  per-spec candidate translations are the descriptor's own seeds.
- **Interior vocabulary is fixed** to `ELEMENTWISE` + the enumerated
  broadcast subset; a refused spec needing a non-ew interior op is a
  family the meta-grammar does not yet spell.

## Reproduce

```python
from mlopt_discovery import family_synth as fs
from mlopt_discovery import handler_synth as hs
fams, report = fs.mint_families(V3_REFUSAL_ITEMS)   # spec+domain rows
handlers, rep = hs.mint_handlers(V3_REFUSAL_ITEMS,
                               families=(),
                               extra_families=tuple(fams))
```

The refusal rows are embedded in `tests/test_family_synth.py`
(`V3_REFUSALS`), spec + domain verbatim from
`results_overnight_prim_v3.json`.

# Term-IR spike — tensor programs as handl `Term`s, e-graph as engine

Date: 2026-11-15. Scope: **proof of concept only** — feasibility
measurement, no migration. Scratch artifacts:

- `bench/drivers/probes/term_ir_spike.py` — encoder/decoder, fidelity
  corpus, vocab sweep, perf driver (standalone; nothing shipped imports
  it).
- `/tmp/term_spike_rs/` — scratch cargo bin (`hnl-engine` path dep +
  `[patch]` to `handl-lang/kernel/crates/handl-value`) that parses
  serde-shaped `Term` JSON and times parse → canonical serialize →
  hash. Not committed; rebuild recipe in §6.
- Outputs: `/tmp/term_ir_spike/result.json`,
  `lifted_terms.jsonl`, `terms_10k.jsonl`.

Versions measured against: graphopt venv's `mlopt-core` **v0.1.1**
(eqsat tag; `opmeta.REGISTRY` = 231 shipped ops),
`handl-lang/engine` `hnl-engine` @ `cfb1979` (`term.rs`, 917 lines).

## 0. Encoding under test

mlopt term → serde-shaped `Term` JSON (externally-tagged enum):

| mlopt | `Term` | payload |
|---|---|---|
| `Op(op, args, attrs)` | `Op{name=op, args=[…], source=Foreign("tensor"), lit={"__mlopt_attrs__": attrs}, dim=Const(1)}` | attrs via `attr_to_data` (JSON-safe: `__tuple__`/`__list__` wrappers) |
| `Var`/`Param(name, TensorType)` | `Arr{dim=1, value={"__mlopt_leaf__": kind, name, shape}}` | `shape=None` wildcard → `null` |
| `Const(v)` | `Const{value=v}` | int vs float spelling preserved by `serde_json::Number` |

`dim=Const(1)` (default 1-cell), `source=Foreign("tensor")` — the
host-function boundary whose name is the module-registry key matching
host-tensor's `tensor.*` surface. `lit` carries the attrs record: it is
the only `Value`-typed payload slot on `Term::Op`. This is a
**convention, not a standard** — see §5 gaps.

## 1. Round-trip fidelity — Q1

**Verdict: lossless on everything tested.**

| Corpus | Nodes | Result |
|---|---|---|
| Lifted torch module (conv2d→relu→mean→unsqueeze→expand→transpose→reshape→add→gelu→linear→layer_norm→linear→sdpa→pack, 2 outputs) | 35 (21 ops, 15 op names) | `dec(enc(root)) == root` **exactly**; 0 bad nodes; attrs `dim/dim0/dim1/eps/padding/shape/stride/cudnn_enabled` all survive |
| Every op in `opmeta.REGISTRY` minted once with best-effort canonical attrs (`_ATTR_PROBES` in the script, ~60 ops carry attrs incl. tuple/list/int/float/bool/None values) | 231 shipped (+1 declared op registers when `kit_torch` imports — 232 in the `all` run) | **all mint + round-trip equal**; 0 attr-encode failures |

What does *not* survive / needs a convention — the honest gap list:

- **Non-finite floats.** `json.dumps` emits bare `NaN`/`Infinity`;
  `serde_json` rejects them (verified: crafted `clamp(min=NaN)` term →
  `parse error: expected value`). mlopt `Const`/attrs can hold
  `float("nan")`/`inf` in principle (the boundary canonicalizes dtypes
  to strings, but nothing strips non-finite scalars). Fix exists:
  host-tensor's codec already wraps payloads as
  `{"__nonfinite__": "nan"|"inf"|"-inf"}` — the same wrapper would be
  needed in `attr_to_data`'s Term-JSON profile.
- **DAG sharing is not representable.** `Term` is a tree; mlopt terms
  are hash-consed DAGs with heavy sharing (residual streams). The
  encoder tree-expands shared subterms — same exponential-blowup caveat
  as `op_repr` (which is why `op_repr_dag` exists). The 10k-node perf
  program had ~no sharing so this didn't bite; a whisper decode step
  *will* share. A `let`/`{"Ref": id}` encoding or an interning sidecar
  is required for real programs — **this is the biggest structural
  mismatch**, bigger than attrs.
- **Multi-output ops**: already a non-issue — the torch boundary flattens
  them (`index` attr on `split/chunk/unbind/tensor_split` + `pack`
  root), all of which round-trip as ordinary ops.
- **Shapes/types**: `TensorType` lives only on `Var`/`Param` leaves and
  survives there; intermediate shapes are *derived* by mlopt's typing
  pass, not stored — nothing is lost, but a typed `tensor.*` fragment
  gets no shape information from the Term alone.
- Exotic attr values (sets, `torch.dtype` objects) can't appear — the
  export boundary already canonicalizes to JSON-safe spellings; attrs
  minted with `validate=False` still round-trip as long as they're
  `attr_to_data`-encodable.

## 2. Term-as-e-node viability — Q2

**Verdict: viable today with a ~40-line adapter; node identity does not
require mlopt's `Op`/`ENode` Python objects.**

`mlopt_core.egraph.types.ENode` is already exactly the flattened shape a
`Term::Op` carries: `(op: str, children: tuple[eclass_ids],
attrs: tuple[(k, repr(v))])` with precomputed strict signature hash.
The e-graph never sees term objects — `add_term` is itself just a
post-order walk minting `ENode`s. Demonstrated in the spike:
`termjson_to_eg()` walks the serde `Term` JSON and calls the public
`eg.add_enode(name, child_eids, attrs)` / `eg.add_leaf(key)` — **no
`Op`/`ENode` construction, no fork**. On the lifted program it
reproduced the identical e-graph: 35 enodes / 35 classes, same as
`eg.add_term(ir.root)`.

Minimal adapter sketch (already in the script):

```python
def walk(node):                      # serde Term JSON -> e-class id
    if "Op" in node:
        op = node["Op"]
        kids = tuple(walk(a) for a in op["args"])
        attrs = lit_attrs(op.get("lit"))        # __mlopt_attrs__ record
        return eg.add_enode(op["name"], kids, attrs)
    if "Arr" in node:                 # leaf record
        return eg.add_leaf(canonical_leaf_key(node["Arr"]["value"]))
    if "Const" in node:
        return eg.add_leaf(f"const:{node['Const']['value']!r}")
```

Caveats:

- The walk is over a **tree**, so shared subtrees are re-walked —
  `add_enode` dedups through `_node_to_class` so the *result* is
  correct, but insertion cost is exponential on sharing-heavy DAGs
  unless the adapter memoizes on a subtree key (serialize-hash) — the
  same fix `add_term`'s `_memo` already applies. With a `Ref`/let
  encoding the walk stays linear.
- Attr identity: `add_enode` runs `_norm_attr_value` + repr-keyed
  signatures — spelling-strict (`min=0` ≠ `min=0.0`), matching mlopt's
  `Const`/`Op` leaf-identity contract. `attr_from_data` restores
  tuples/lists so spellings survive.
- Matcher/law side is untouched: rules match on `(op, children,
  attrs)` — exactly what the adapter feeds. Nothing in the rewrite
  engine cares where the enodes came from.

So "can the existing Python e-graph be driven with Term-derived node
keys without forking it?" — **yes**, because the e-graph's node key was
never the mlopt term; it was always this triple. What Term buys is a
*serializable, handl-legal* spelling of that triple.

## 3. Perf smoke — Q3

10k-op random program (6,576 ops minted of 10k attempted — the rest hit
attr-schema misses and fell back to leaves; 6,592 deduped DAG nodes,
1.65 MB Term JSON / 1.47 MB canonical serde output):

| Stage | Time |
|---|---|
| Python `enc()` mlopt→Term-JSON dict | 11.5 ms |
| Python `json.dumps` (transport blob) | 11.7 ms |
| **Rust `serde_json::from_str::<Term>`** | **9.2 ms** |
| Rust `serde_json::to_string` (canonical — `Map` is BTreeMap) | 3.0 ms |
| Rust `DefaultHasher` over canonical bytes | 0.28 ms |
| Rust total (parse+canon+hash) | **~12.4 ms** |
| wall incl. subprocess spawn | 17.5 ms |
| Baseline: `term_to_data`+`json.dumps(sort_keys)`+`hash` | ~16.5 ms |
| Baseline: `EGraph.add_term(root)` (union-find + congruence) | ~63–185 ms (two runs, warmup variance) |

Reading:

- Hash-consing a whole 10k-op program through serde `Term` costs
  **~0.5 µs/node in Rust, ~3.5 µs/node end-to-end including the Python
  encoder** — one to two orders of magnitude *below* e-graph insertion
  cost. The Rust↔Python boundary is **viable for batch transport**;
  it is not viable (or needed) per-e-node inside the saturation loop —
  per-node crossing would cost more than the enode it carries.
- Keeping Terms in Python-land as JSON dicts is also fine at this
  scale: the Python encoder is the same order as the Rust parse. The
  honest conclusion: **neither side is the bottleneck; the boundary is
  a serialization format question, not a performance question.**
- `serde_json` canonical output is deterministic (BTreeMap-ordered
  object keys, enum-tag order fixed) — usable as a hash-cons key;
  `DefaultHasher::new()` is deterministic within a Rust version.
- `mlopt-native` (the existing PyO3 seam) was not exercised; nothing in
  these numbers argues for moving search to Rust *for speed* — the
  argument would have to be handl-side law execution, which is a
  different question.

## 4. What a `tensor.*` fragment must declare — Q4

Measured inventory (`opmeta.REGISTRY`, v0.1.1):

- **231 ops**; arity known for only 79 (`arity=None` on 152 —
  variadic/operand-symmetric ops the registry doesn't fix).
- **90 ops carry an `ATTR_SCHEMA`** entry (canonical attr names per
  positional slot); a subset carry `ATTR_REQUIRED` (e.g. `layer_norm`
  requires `dim`). Attr *values* are JSON-safe by boundary contract
  (dtypes/devices canonicalized to strings).
- **13 classification tags** (`pointwise`, `reduce-dim`, `viewish`,
  `relayout`, `commutative`, `activation`, `fold-at-lowering`, …) that
  projections — cost, typing, matchers — consume.
- Plus cost formulas, torch bindings, shape rules that stay code.

Cross-check against the existing host surface (`host-tensor`, per
`docs/reports/host_tensor_sink_phase0.md` + re-measured here): 238
`tensor.*` ops declared in `src/index.hnl` (`cap` decls), ~237 in the
Rust `TENSOR_OPS` table; **153 name overlap** with the IR vocabulary,
78 IR-only, ~85 host-only. Name overlap ≠ parity: host ops are
positional-JSON ATen ports, mlopt ops are attr-carrying IR nodes — e.g.
`tensor.max_pool` vs IR `max_pool2d`, `tensor.concat(tensors, axis)` vs
`concat(*tensors, dim=-2)`. IR-only ops are largely *structural* —
`pack`, `leaf`, `getitem`, `index`, `alias`, `copy`, `detach_`, `om`,
`lstm.input`, dtype casts (`float`/`half`/`bfloat16`/…) — which a
fragment would spell as `Op`s/`Arr`s/`Tag`s rather than `tensor.*`
calls.

So a `tensor.*` fragment needs, per op: qualified name, arity class
(fixed-N / variadic), attr names+types (the 90-schema subset), and the
13-tag classification projected into whatever the fragment's cost/
rewrite rules consume. The attr-schema JSON (`ATTR_SCHEMA` +
`ATTR_REQUIRED`) is already data and could generate declarations; the
`.hnl` `cap "tensor.x"` + doc-comment convention in `index.hnl` is the
existing declaration shape. Nothing in the spike found a metadata need
that mlopt doesn't already have as data — the gap is *coverage* (153/231
names, fewer semantically) and *spelling* (positional vs attrs), not
missing concepts.

## 5. Gaps and honest risks

1. **No attrs slot on `Term::Op`.** `lit` is documented for literal
   operands (`proj_env`/`proj_child`); overloading it for the whole
   attrs dict works but is a convention competing with any future lit
   use. Alternatives considered: attrs-as-extra-`Const`-args (changes
   arity/matching), `Opaque{data}` tagged record (children stop being
   `Term`s — breaks the e-node walk), extending `Term` (frozen kernel).
   The `lit` convention is the least-bad; it should be written down as
   a `tensor` module contract if this ever ships.
2. **Tree-only Terms vs DAG terms** — the real blocker for whole-program
   transport. Needs `{"Ref": n}`/let-bindings or an interning sidecar;
   until then, encoder must tree-expand (exponential on sharing) or
   refuse.
3. **Non-finite scalars** need the `__nonfinite__` wrapper (precedent
   exists in host-tensor's codec).
4. **`Foreign("tensor")` is a promise**: the name must resolve in the
   module registry at eval time and ops must exist as host functions —
   host-tensor covers ~2/3 of the vocabulary today.
5. Fidelity was measured on *mintable* terms; laws/patterns (metavar
   `"mvar"` strings, `term_to_data`'s pattern encoding) were not in
   scope — patterns would need their own Term spelling if handl-side
   rewriting is the goal.
6. Small print: `Const` spelling-strictness (`8` vs `8.0`) survives
   JSON because `serde_json::Number` keeps int/float distinct — but a
   *canonicalizer* that rounds-trip through `f64` would silently break
   it; the Rust probe's `to_string` preserves the parsed `Number`, so
   it's safe as measured.

## 6. Rebuild / reproduce

```sh
# Rust probe (scratch, not committed)
mkdir -p /tmp/term_spike_rs/src
# Cargo.toml: hnl-engine = { path = "/home/luis/GitHub/handl-lang/engine" }
#   + [patch."https://github.com/handl-lang/kernel"] handl-value = { path = "…/kernel/crates/handl-value" }
# main.rs: per JSONL line — from_str::<Term>, to_string, DefaultHasher; prints JSON timings
cd /tmp/term_spike_rs && cargo build --release

# Python driver (in repo)
.venv/bin/python bench/drivers/probes/term_ir_spike.py all   # fidelity+vocab+perf
.venv/bin/python bench/drivers/probes/term_ir_spike.py vocab # 231/231 round-trip
```

## 7. Recommendation

**Keep parallel IRs; ship the codec, not the convergence.** The three
options priced against the measurements:

- **Full convergence** (terms *are* `Term`s; search runs over them,
  ideally in Rust): fidelity is *not* the obstacle — it's perfect — but
  the costs are structural: no attrs slot (lit hack), no DAG/sharing
  construct (needs a new encoding anyway), the entire law/matcher/
  cost/extraction stack is Python and would have to be re-expressed as
  handl laws or ported to Rust, and the perf case for Rust search is
  unproven (the boundary itself is already noise-level). High effort,
  benefit unmeasured.
- **Typed-view-over-Term** (Term is the canonical term, mlopt types
  view it): inverts the dependency for no measured gain — mlopt's
  `TensorType`/attrs/e-class machinery is richer than what `Term`
  carries, and every consumer (laws, cost, sinks) already speaks mlopt
  terms. The view would be strictly lossy without the §5 additions.
- **Parallel IRs + thin codec** (status quo plus): mlopt IR stays
  canonical for search; the ~60-line `enc`/`dec` + `termjson_to_eg`
  adapter (both in the spike script) gives handl-side consumers —
  `.hnl` emission, CatoptKit eval, handl-authored laws — a legal,
  verified, lossless spelling of any mlopt program *today*. Cost:
  ~12 ms per 6.5k-op program at the boundary, zero changes to shipped
  packages.

Revisit convergence only if a concrete consumer needs handl-native
*rewriting* over tensor terms (laws authored in `.hnl` firing inside
the e-graph) — at that point the missing pieces are the attrs-slot
convention, a DAG encoding, and porting the matcher, in that order.
The spike's bottom line: **the boundary is better than expected on
fidelity and worse than expected on structure** — Terms carry the
information perfectly but not the *shape* (attrs slot, sharing) the IR
actually needs.

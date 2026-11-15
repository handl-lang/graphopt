# Handl law bridge — Phase B: can admitted laws be handl rewrite rules?

**Question.** Mint r3 proved prim-fold laws admit (`realcorpus_mint_r3.md`).
Phase A already compiles mlopt laws into handl `TransformDef` rows
(`discoverykit.handl.law_to_handl`) and replays recorded e-graph
applications through handl's `RewriteEngine` as a differential oracle
(`mlopt_discovery.handl_engine.HandlEngine`).  Phase B asks the sharper
question: **express the admitted laws themselves as handl rules, and
does the handl engine validate them — including where mlopt *declines*?**

**Answer.** Structurally yes for a minority of the bank; semantically
no for most of it, and the current oracle cannot see the failure mode.
Of 144 shipped laws + 58 synop expand premises, **89 (44%) translate
with no known fidelity gap on the attr-free surface** — but the
translated rules are *shape-directed rewrites*, not e-graph laws: they
carry no attrs, no guards, no repeated-metavar equality, and no
subtree-depth patterns.  On guard-violating inputs the handl rules
**over-fire and produce semantically wrong terms** — demonstrated live
on the shipped `vocab/` projection (`chunk_single` deletes any
`chunk`, `transpose_noop` deletes any `transpose`).  The existing
differential never observes this because it only replays applications
mlopt already accepted.

## 1. What `engine-transform-rewrite` consumes

The engine (`hnl-engine-transform-rewrite`, ADR-0094) interprets
`TransformDef` data — it is **not** a pattern-matching rewriter in the
e-graph sense:

- `TransformDef { id, pass, node_kind, priority, when, rule }`
  (`kernel/crates/handl-language/src/fragments/transform_def.rs:45`).
  Dispatch is `(pass, node_kind)`; `priority` orders same-kind rules;
  `when: Option<TransformCondition>` gates firing.
- `TransformRule` (`transform_def.rs:92`):
  - `Declarative { target_kind, children, payload? }` — rebuild the
    node as `target_kind` whose children are `RewriteStep`s
    (`Keep(i)`, `KeepRole`, `KeepPath`, `KeepRest`, `Splice`,
    `Create{kind,payload,children}` recursive, `Apply{rule_id,index}`,
    `Fresh(hint)` — `transform_def.rs:135`).  `payload` is a *literal*
    shallow patch — it cannot copy values from the matched node.
  - `Replace { path }` — the result IS the descendant at child-index
    `path` (ADR-0127; root-eliminating laws).
  - `Native(name)` — escape hatch into a wired `NativeRewrite`
    registry (`desugar.*`, or `handl:<mod>.<fn>` routing to `.hnl`
    functions).
- `TransformCondition` (`transform_def.rs:193`): `KindAt` (a child's
  *kind* at one positional index/role), `AnyChildKind`, `PayloadEq`,
  `ChildPayloadEq`, `ChildCount`, `And`/`Or`/`Not`.  **Depth-1 only**,
  and only kind/payload/arity — no attr tests (attrs are not part of
  the condition vocabulary at all), no subtree equality, no
  value/shape reasoning.
- Traversal: `BottomUp` (default), `TopDown`, `Fixpoint{max}`
  (`engine-transform-rewrite/src/lib.rs:71`) — iterate-until-stable,
  **not** e-graph saturation: no congruence closure, no e-classes, no
  cost model, no multi-directional search.  `apply_rule` is the
  surgical single-site door the peer uses; `run_pass` rewrites a whole
  arena bottom-up.

Relation to the rest of the stack: the engine is a pure `Ast → Ast`
rewriter, orthogonal to `.hnl` eval.  `CatoptKit` (`mlopt-kit`) is the
assembled eval-side kit; `hnl-run` evaluates `.hnl` programs.  Rewrite
rules live in fragment `transforms.yaml` files loaded by
`language_from_dir` (`handl-kit-prelude-lang`); the mlopt bridge emits
exactly that shape (`scripts/dump_fragments.py`,
`mlopt_discovery.vocabdir.emit_yaml_rows`).

**Expressibility verdict for e-graph laws.** A mlopt `Rewrite`
(`lhs → rhs` pattern terms with string metavars, `cond`/`check`
guards, `derive`/`dspec` computed attrs) maps onto this vocabulary
only after compilation to *shape-directed* form — which
`discoverykit.handl.law_to_handl` (`handl.py:286`) already does:
metavars bind to fixed child *paths* (`_metavar_paths`, `:86`),
depth-1 op children become `KindAt` checks (`_when_clauses`, `:102`),
the rhs rebuilds as nested `Create`/`KeepPath` (`_children_spec`,
`:211`), bare-metavar rhs becomes `Replace` (`_rule_body`, `:255`).
That compilation is faithful **iff** the pattern's non-metavar
structure is exactly "root kind + depth-1 child kinds + arity" and no
attrs/guards/equalities matter — the gap classes measured below.

## 2. The translation table (measured, not assumed)

Laws exercised (folds rebuilt from `realcorpus_mint_r3.json` via
`prim_mint.prim_fold_of` — they are judged-admitted but **not yet in
the shipped `minted.db`**; `AUTO_ADMITTED` still holds only
`synop_6ed0a07c7d`):

| law | lhs → rhs | handl row | what translated | what did NOT |
|---|---|---|---|---|
| `prim:synop_0b304215b3` (rmint:084, admitted) | `detach_(contiguous(v))` → `synop_0b304215b3(v)` | `detach_` node, `when: KindAt(0,"contiguous") ∧ ChildCount(1)`, `Declarative synop_0b304215b3 [KeepPath[0,0]]` | **everything** — no attrs, no guard, no leaves | nothing — the one fully clean case |
| `synop_0b304215b3.expand` | `synop(v)` → `detach_(contiguous(v))` | `Declarative detach_ [Create contiguous [KeepPath[0]]]` | full expand | nothing |
| `prim:synop_f0576b5150` (rmint:075, admitted) | `transpose(reshape(v,shape=a12),dim0=a5,dim1=a6)` → `synop(v,a5,a6,a12)` | `transpose` node, `when: KindAt(0,"reshape") ∧ ChildCount(1)`, `Declarative synop_f0576b5150 [KeepPath[0,0]]` | op spine + metavar path | **all three attrs** — rhs `Declarative` has no attr channel; the synop call is emitted parameterless |
| `synop_f0576b5150.expand` | `synop(v,a5,a6,a12)` → `transpose(reshape(v,shape=a12),dim0=a5,dim1=a6)` | `Declarative transpose [Create reshape [KeepPath[0]]]` | op spine | **all attrs** — expands to `transpose(reshape(v))` with no `shape`/`dim0`/`dim1`: a meaningless term |
| `add_unsq_pad_r` (hand-promoted, real paying site) | `add(u, unsqueeze(v,d=UD))` → `add(u,v)` | `add` node, `when: KindAt(1,"unsqueeze") ∧ ChildCount(2)`, `Declarative add [KeepPath[0], KeepPath[1,0]]` | structure + paths | **`cond` guard** (`bcast-eq`/`bcast-into` on broadcast grids — lives only in `mlopt.law` metadata) and the `dim` attr |
| `mul_unsq_nat` (machine-admitted) | `mul(unsq(u,UD), unsq(v,UD))` → `unsq(mul(u,v),UD)` | `mul` node, `when: KindAt(0,"unsqueeze") ∧ KindAt(1,"unsqueeze") ∧ ChildCount(2)` | structure | **`flat-map-unsq` guard**, shared-`UD` attr equality (5 bank laws carry repeated attr metavars — see §4), and the output `dim` attr |
| `transpose_noop` | `transpose(u,d0,d1)` → `u` (guarded) | `Replace { path: [0] }` — unconditional | the root-eliminating shape (ADR-0127 works) | **the entire guard** — fires on every `transpose` |
| `chunk_single` (in shipped `vocab/`) | `chunk(u,chunks=1,…)` → `u` (guarded `attr-eq CK 1`) | `Replace { path: [0] }` | shape | **the `chunks==1` attr guard AND the attr itself** |

## 3. Validation — does handl agree the rewrite is equivalence-preserving?

Peer: `packages/mlopt-handl-peer` CLI built this session
(`PYO3_PYTHON=.venv/bin/python cargo build --bin rewrite`, 11.7 s —
deps were already compiled; binary at
`packages/mlopt-handl-peer/target/debug/rewrite`, driven via
`MLOPT_HANDL_PEER_BIN`).  Fragment dir `/tmp/handl_phaseb/fragments`
(8 shell files + `ast.yaml` leaf shapes + 8 `transform.mlopt.*` rows
each carrying distinct `priority` — same-kind collisions are a load
time hard error, `handl.py:_disambiguate_priorities`).

Cases (`/tmp/handl_phaseb/run_cases.py`; mlopt side =
`EGraph().apply_rule`, handl side = `HandlPeer.apply_rule`, comparison
renders **with attrs** — handl's own surface drops them):

| case | mlopt | handl | verdict |
|---|---|---|---|
| `synop_0b304215b3` fold, `detach_(contiguous(x))` | fired → `synop(x)` | `synop_0b304215b3(x)` | **AGREE, fully** (incl. attrs — there are none) |
| `synop_0b304215b3.expand`, `synop(x)` | fired → `detach_(contiguous(x))` | same | **AGREE, fully** |
| `synop_f0576b5150` fold, `transpose(reshape(v50,S),1,2)` | fired → `synop(v50,a5=1,a6=2,a12=S)` | `synop_f0576b5150(v50)` **no attrs** | agree on `{kind,children}`; **attrs silently dropped** — the synop call is underdetermined (its `.expand` premise cannot reconstruct dims/shape) |
| `synop_f0576b5150.expand`, `synop(v50,attrs)` | fired → `transpose(reshape(v50,S),1,2)` | `transpose(reshape(v50))` **no attrs** | agree on shape; **semantically broken output** — reshape without `shape`, transpose without `dim0/dim1` |
| `add_unsq_pad_r`, guard HOLDS (`u=(2,4,8,8)`, `v=(4,8,8)`, dim 0 — the real `RelativePositionBias` spelling) | fired → `add(u,v)` | `add(u,v)` | **AGREE** |
| `add_unsq_pad_r`, guard FAILS (`u2=(8,8)`, `unsq(v2,dim=1)`) | **declined** | fired → `add(u2,v2)` | **DISAGREE — handl produces a wrong term** (the source broadcasts to `(8,8,8)`; the "rewrite" yields `(8,8)`) |
| `mul_unsq_nat`, guard HOLDS (`dim=-1` both, equal ranks) | fired → `unsq(mul(u,v),dim=-1)` | `unsq(mul(u,v))` **no dim** | agree on shape; output `unsqueeze` loses `dim` |
| `mul_unsq_nat`, mismatched dims (`-1` vs `0`) | **no match** (shared `UD` binds once) | fired → `unsq(mul(u,v))` | **DISAGREE — over-fire** |
| `transpose_noop`, `transpose(u4,0,0)` | fired → `u4` | `u4` | **AGREE** |
| `transpose_noop`, `transpose(u5,-2,-1)` real swap | **declined** | fired → `u5` | **DISAGREE — deletes a real transpose** |

Bank-wide, on the **current** `vocab/` projection (144
`transform.mlopt.*` rows — the live `vocabdir` output; the committed
`tests/fixtures/handl-spike/fragments-all` fixture is a stale 84-rule
snapshot): the existing audit driver
(`bench/drivers/audit/handl_differential.py`, peer pointed at `vocab/`)
reports **76 agree / 0 disagree / 48 declined / 19 guard-raised /
1 skipped** — wherever both engines fire on the same synthesized
term, they agree.  That is real but one-directional evidence: see the
blind spot below.

**The oracle's blind spot (measured, not hypothetical).**
`handl_differential._outcome` returns `"declined"` the moment mlopt
does not fire — handl is never consulted on guard-violating inputs.
`HandlEngine._check_application` likewise only replays *recorded*
applications (mlopt already vetted the guard) and buckets
`derive`/leaf-rhs laws as `skipped`
(`handl_engine.py:435`).  So the differential validates "handl agrees
where mlopt fires" and is **structurally incapable of detecting
handl firing where mlopt declines** — precisely the failure mode the
translated guarded laws have.  Confirmed on the shipped vocab:
`transform.mlopt.chunk_single` rewrites `chunk(u, chunks=2, …)` to `u`
(mlopt declines — `attr-eq CK 1`), and `transform.mlopt.transpose_noop`
rewrites `transpose(t, -2, -1)` to `t` (mlopt declines — real swap).
Both are unconditional `Replace` rules in `vocab/transforms.yaml`.

## 4. Coverage assessment — what fraction of the bank is expressible?

Classifier over `ALL_RULES` (144, which already includes the 58
`prim:synop_*` folds) + the 58 `.expand` premises
(`/tmp/handl_phaseb/coverage.py`, results `/tmp/handl_phaseb/coverage.json`).
A law is **faithful-structural** iff `law_to_handl` emits a real rule
(no `Native` refusal) and none of these gap classes apply:

- `guard_gap` — `cond`/`check` non-None (handl `when` can't see it)
- `derive_gap` — `derive`/`dspec` non-None (computed RHS attrs)
- `spine_deep` — an op node sits at depth ≥2 in the lhs (`when` pins
  only depth-1 child *kinds*; deeper structure is unconstrained)
- `attr_literal` — lhs carries a non-metavar attr value (attrs are not
  in the AST at all — `term_to_ast` emits `payload.children` only)
- `attr_rhs` — rhs carries attrs (no step copies bound attrs into a
  `Create` payload)
- `repeat_mvar` — a metavar occurs twice in the lhs (subtree-equality
  constraint; handl binds first path only)

| set | faithful | dominant gaps |
|---|---|---|
| `ALL_RULES` (144) | **37 (25.7%)** — of which 1 (`square_to_pow`) has a leaf rhs the differential skips → **36 fully clean** | `spine_deep` alone 35; `guard_gap` alone 21; attr_rhs/guard/derive combos ~40; `repeat_mvar` 14 across combos |
| `.expand` premises (58) | **52 (89.7%)** | the 6 non-faithful are exactly the attr-carrying synops (`attr_rhs`) |
| **combined** | **89 / 202 ≈ 44%** | |

Additional uncounted gap found while classifying: **5 laws carry a
repeated *attr* metavar** (`select_mul`, `mul_unsq_nat`, `glu_fold`,
`qkv_fuse`, `prim:synop_6ed0a07c7d`) — e.g. `mul_unsq_nat`'s shared
`dim="UD"` on both unsqueezes is an equality constraint the arg-level
`repeat_mvar` check misses; demonstrated live (mismatched dims:
mlopt cannot match, handl fires).

The 37 faithful laws are the expected core: `comm_*`, `assoc_*`,
`id_*`, `double_neg`, `sub_to_add`, `silu_*`, `square_*`,
`pow_to_square`, `recip_sqrt_to_rsqrt`, `mul_div_swap`,
`*_matmul`/`assoc_linear*` families — plus the shallow `prim:synop3_*`
folds whose manual spellings happen to be depth-1.

## 5. Honest negatives

- **The handl rules are not the laws.**  They are compilations that
  preserve the rewrite *on inputs where every dropped constraint
  happens to hold*.  For guarded laws that is a strict superset of the
  law's domain — over-firing is not a coverage gap, it is a
  correctness bug (measured: `add(u2,v2)` where the source denotes a
  different broadcast).
- **Attrs are the biggest single gap.**  The AST surface
  (`payload.children`) has no attr channel at all: lhs literal attrs
  become unchecked, rhs attrs cannot be emitted, attr-metavar
  bindings/equalities vanish.  Every `synop` fold whose primitive
  takes parameters (`f0576b5150`'s `a5/a6/a12`) produces an
  underdetermined call; its `.expand` produces a *meaningless* term.
- **Guards are the second gap.**  `cond` is pure serializable data —
  the vocabulary to *evaluate* it exists on the mlopt side
  (`laws.cond`), but handl's `TransformCondition` has no equivalent
  and `Native` is the only door.
- **The differential oracle is one-directional** (§3): it cannot
  detect the dominant failure mode.  Any Phase-B validation needs a
  decline-parity arm — feed guard-violating/mismatched instances and
  require handl to decline too.
- **`prim:synop_f0576b5150`/`0b304215b3` are not shipped laws** — they
  were admitted in the r3 re-judge; `minted.db` AUTO_TIER still holds
  only `synop_6ed0a07c7d`.  This spike rebuilt them from the r3 JSON
  candidates via `prim_fold_of`; the records above are faithful to
  what `as_rule()`/`prim_fold_of` produce.
- Coverage "faithful" is *necessary, not sufficient*: it means no
  known gap class fired; peer agreement was spot-checked (10 cases +
  the 76-law bank run), not exhaustively proven per law.

## 6. Phase-B implementation proposal (if warranted)

Warranted **narrowly**: the honest near-term win is not "run mlopt
laws in handl" but "handl validates mlopt laws it can already
express".  Concretely:

1. **Ship the faithful subset as real rules.**  The 36 clean laws +
   52 attr-free expands are already emitted into `vocab/transforms.yaml`
   — but the file also carries the unsound over-firing rows
   (`chunk_single`, `transpose_noop`, every `guard_gap` law).  Phase B
   should **gate emission**: `law_to_handl` (or a `vocabdir` filter)
   marks each row faithful/unfaithful using the §4 classifier, and
   unfaithful laws emit `Native("mlopt.law.<name>")` — the explicit
   refusal the bridge already uses for unbound root-eliminators —
   instead of a silently-wrong `Declarative`/`Replace`.
2. **Add a decline-parity arm to the oracle.**  Extend
   `handl_differential`/`HandlEngine` with negative cases: for each
   guarded law, instantiate guard-*violating* terms (the eqsat
   `test_cond_laws` cases are a ready corpus) and assert the handl
   rule declines.  Until this exists, "0 disagreements" is not evidence
   of equivalence-preservation.
3. **Attrs via payload + a `KeepAttr`/`AttrEq` vocabulary** (handl-side
   change, data-only per ADR-0044): carry `attrs` in
   `payload["attrs"]` through `term_to_ast`, add
   `RewriteStep::KeepAttr(name)` / `TransformCondition::AttrEq` —
   unblocks the 6 attr-carrying expands and every `attr_rhs` law
   (~30% of the bank).
4. **Guards via `Native` + the serialized record.**  `mlopt.law`
   already round-trips `cond`/`dspec` verbatim; a
   `NativeRewrite` that interprets `cond` over payload-carried
   attrs/shapes (reusing `mlopt_core.laws.cond`) makes guarded laws
   expressible *today* without new handl vocabulary — dispatch stays
   data-driven, semantics stay in the record.  This is the honest
   seat for the ~40 guard_gap/derive_gap laws.
5. **Deeper patterns via nested `when` or `Native`.**  `spine_deep`
   (35 laws) needs either recursive `KindAt` paths or the same
   record-interpreting native; e-graph congruence/saturation is out of
   scope for `RewriteEngine` by design — that remains
   `handl-egraph`'s later phase.

**Bottom line:** handl *can* express and validate the shallow,
attr-free, guard-free slice — and does so correctly (76/76 agreement
where both fire).  It cannot yet express guards, attrs, deep patterns,
or repeated-metavar equality, and the shipped `vocab/` rules
over-fire unsoundly on exactly those gaps.  Phase B should ship the
faithful subset gated + the decline-parity oracle, and treat
attr/guard/deep-pattern support as handl-vocabulary work, not a
bridge workaround.

## Reproduce

```sh
# peer binary (built this session; ~12 s, deps cached)
cd packages/mlopt-handl-peer && \
  PYO3_PYTHON=/home/luis/GitHub/handl-lang/graphopt/.venv/bin/python \
  cargo build --bin rewrite

# fragment dir + case runner + coverage classifier (scratch, /tmp):
#   /tmp/handl_phaseb/fragments/{tokens,grammar,eval,types,caps,options,lexer_rules,ast,transforms}.yaml
#   /tmp/handl_phaseb/run_cases.py     — the 10-case mlopt-vs-handl table
#   /tmp/handl_phaseb/coverage.py      — the §4 gap classifier
#   /tmp/handl_phaseb/coverage.json    — per-law verdicts

MLOPT_HANDL_PEER_BIN=$PWD/packages/mlopt-handl-peer/target/debug/rewrite \
  .venv/bin/python /tmp/handl_phaseb/run_cases.py

# bank-wide differential against the LIVE vocab projection
# (handl_differential hardcodes the stale fragments-all fixture;
#  point hd._FRAGMENTS at Path("vocab") and reset hd._peer — see §3)
```

# E-graph Rust/Python stats divergence — root cause and fix

**Date:** 2026-11-13. **Driver:** `bench/drivers/audit/egraph_perf_parity.py`
(+ `egraph_differential.py` for the deep diff). **Binary:**
`target/release/egraph-run` via `MLOPT_EGRAPH_RUN`.
**Resolves:** the "two known divergences (DeltaRuleBlock, ChunkedSSD)"
open finding in `docs/reports/rust_saturation_gap.md`.

## Verdict

**The Rust port was wrong — a candidate-scan ordering bug, now fixed.**
Neither engine's saturated partition was ever wrong (the e-graphs were
identical throughout); the divergence was in *bookkeeping order*, which
is correctness-relevant under `rule_budgets` (see below).

## Reproduction (before the fix)

`egraph_perf_parity.py --module DeltaRuleBlock` / `--module ChunkedSSD`
(both `stats=DIV`):

| module | field | Python | Rust |
|---|---|---|---|
| DeltaRuleBlock | n_enodes / n_classes / n_proof_edges | 894 / 98 / 794 | 836 / 98 / 738 |
| ChunkedSSD | n_enodes / n_classes / n_proof_edges | 163 / 108 / 55 | 164 / 108 / 56 |

A full-report run (`egraph_differential.compare` on the lifted zoo
term + the 85-rule vocab spec) showed `classes`, `signatures`,
`root_sig`, `stop`, `iterations`, `truncation_level` all **identical** —
the saturated e-graphs were the same partition.  Only the *counts*
differed: Python recorded 40 more `assoc_add` + 16 more `comm_add`
applications/proof-edges (DeltaRuleBlock); Rust recorded 1 more
`mul_div_swap` (ChunkedSSD).  `rule_budgets` (enodes-spent) differed
correspondingly.

## Mechanism

Merge/app counts are order-sensitive even at a fixed partition: every
successful `union` eliminates exactly one class, so the *number* of
recorded merges equals (classes ever created) − (final classes), and
the number of enodes interned depends on which firings ran before
their RHS hash-consed onto an existing enode.  Different rule/
candidate application order ⇒ different merge tree ⇒ different counts,
same final partition.

The order divergence was in `handl-egraph`'s
`EGraph::candidate_classes` (`egraph/src/egraph.rs`): for a
whole-graph scan (`search=None`, i.e. **every rule's first pass**)
it iterated `self.classes.keys()` — an `FxHashMap`, i.e. **hash
order** — while the Python spec iterates `self._classes.keys()`, a
dict whose insertion order is ascending class id.  The dirty-frontier
branch was already ordered (`BTreeSet` ≙ Python's `sorted(...)`), but
the first-pass order perturbation propagated: different merge order ⇒
different surviving union roots ⇒ different canonical ids ⇒ different
frontier order downstream ⇒ different firing counts.

This is not merely cosmetic: `rule_budgets` bound *new enodes per
rule*, and which matches land before suspension is decided by scan
order.  Under a binding budget the two engines could saturate to
*different partitions* — different extracted terms, different
certificates.  (Neither DeltaRuleBlock nor ChunkedSSD actually
suspended — `budget_suspended: []` — which is why only the counts
moved there.)

## The fix

`handl-egraph` `src/egraph.rs`, `candidate_classes`: sort the
candidate id list (`ids.sort_unstable()`) for both branches, matching
the spec's ascending-class-id scan order.  9 lines + comment; no
algorithmic change.  `cargo fmt --check` clean; `cargo test --release`
in the egraph repo: **26/26 pass** (incl.
`saturation_is_deterministic_across_runs`,
`rule_budgets_suspend_explosive_rules`).

Because `mlopt-kit`'s dep is `handl-egraph = { git …, branch = "main" }`
and git deps only see *committed* revs, `graphopt/Cargo.toml` gained a
`[patch."https://github.com/handl-lang/egraph"]` → `path = "../egraph"`
entry (the file's own documented local-override mechanism, same shape
as the existing `host-tensor` patch) so working-tree builds pick the
fix up.  **To land properly:** commit `egraph/src/egraph.rs` on egraph
`main`, then drop the `[patch]` + `cargo update -p handl-egraph`.

## Verification (after the fix)

* `egraph_perf_parity.py` full corpus: **0 stats divergences**.  The
  bulk run covered 55 modules clean; the remaining 18 were run
  per-module (`--module`) — all `stats divergences: 0`, including
  RealGatedLRU verified stats-only (the parity driver's tracemalloc
  second pass OOMs/sigterms on this 11 GB host — see caveat):
  Python `{max_nodes, it=1, enodes=925549, classes=124,
  proof_edges=925425, budgets {comm_add:9, comm_mul:26, assoc_add:0,
  assoc_mul:36}}` ≡ Rust identical.
* DeltaRuleBlock now `894/98/794`, ChunkedSSD `163/108/55` — both
  engines byte-identical stats.
* `egraph_differential.py`: **87/87 fixtures `agree`** (was already
  clean — its fixtures never hit the ordering-sensitive path).
* `pytest tests/test_egraph_differential.py
  tests/test_handl_egraph.py`: **183 passed**.
* `cargo build --release -p mlopt-kit --bin egraph-run`: clean.

## Residual known divergence (documented, not fixed)

Member *enumeration* order still differs by design: Python's
`_member_key` is `(op, children, repr(attrs))` — attrs compared as a
**repr string** — while Rust's `ENode: Ord` compares attrs
**structurally** (`AttrValue` rank-then-value, `value.rs`).  The two
orders agree except when same-`op`+same-`children` enodes carry attrs
whose repr order flips vs structural order (e.g. `Int(10)` vs `Int(2)`
— `'1'<'2'` as text but `2<10` numerically; `Str` vs `Int` — quote
`0x27` < digit `0x35` in repr, but Rust ranks `Int < Str`).  When such
a pair coexists in one class, match-enumeration order — and hence
merge order and the stats counters — can still diverge.  The saturated
*partition* is unaffected (confluence), and no corpus module triggered
it.  Aligning it would mean either porting Python `repr` ordering into
Rust (ugly, and the Rust side's structural order is the better
contract) or changing `_member_key` in eqsat (touches extraction
tie-breaks and the congruence drain — a substrate-wide behavioural
change).  Left as a named divergence; revisit only if a stats
counter divergence reappears.

## Caveats / notes for the parent agent

* `graphopt` working tree also contains **pre-existing unrelated
  changes** (`mlopt_discovery/zoo.py`, `tests/test_discovery_zoo.py`,
  `docs/reports/zoo_corpus_expansion.md`) — not mine, untouched.
* My changes: `egraph/src/egraph.rs` (+9/−1), `graphopt/Cargo.toml`
  (+6, the `[patch]`), `graphopt/Cargo.lock` (path-dep source update),
  this report.  **No commits made** per instructions.
* The parity driver could not complete RealGatedLRU end-to-end on this
  host (process killed during the ~4 GB module — likely memory
  pressure; it also died identically *before* the fix, so it is not a
  regression).  Stats were verified equivalence-class by direct
  spec replay instead (numbers above).
* `handl_egraph._BIN_CANDIDATES` / the audit driver's candidate list
  prefer `target/release` — the fix is only in the release binary
  built after the patch; a stale debug binary still carries the bug.

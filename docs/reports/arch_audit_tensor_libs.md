# Architecture audit — host-tensor + lib-tensor ATen port generators

Read-only audit (subagent). Scope: `aten_port_gen.rs` (1,395 lines),
`aten_port_gen.py` (839 lines), curated lists, generated surfaces,
`aten_kernels.rs` (4,923 lines), `internal/aten.hnl` (3,620 lines),
verify.hnl, math.* leaf surface across lib-utils/host/lib-math*.

## 1. Generator quality

- **host `--check` is real** (byte-compares generated file + all six
  marker regions; deterministic sort). NOT wired into CI — manual only.
- **Footgun — partial regeneration clobbers**: `--include`/`--exclude`
  shrink `resolved` but a non-check run rewrites ALL marker regions +
  `aten_generated.rs` with the subset. `-- --include 'tensor.relu'`
  would delete every other op. No guard. **Medium.** Fix: refuse write
  when filters active unless `--force`.
- Dead MAP entries: `unsqueeze`, `reshape` unreachable (handwritten in
  shape_ops.rs). Misleading.
- `seam` detection scans whole raw line incl. comments — false-positive
  risk. Trivial.
- **lib `--check` is NOT a check** (`aten_port_gen.py:738-745`): prints
  counts, returns; never regenerates/diffs. **No drift gate at all** on
  the .hnl side — generated files can silently rot. **High.** Fix: buffer
  emit output and diff (code already builds body strings).
- lib dead code in `gen_driver` (discarded first call_args loop, unused
  `real`, unreachable `stack_orig` block, duplicate assignments).
- lib IMPORTS lists `aten_ow`/`aten_owl` twice; ~100 names imported per
  family file regardless of use (source of unused-binding warnings).
- lib `resolve_schema` does no out/inplace filtering + silently falls
  back to `cands[0]` — provenance can cite wrong schema. Low-medium.
- Neither generator wired into CI (`ci.sh` → hnl-gate/cargo test only).

## 2. Duplication between generators

- Curated list + schema ARE single-source: lib reads
  `../host-tensor/tools/aten_schemas.json` + `aten_port_list.txt`
  read-only via hardcoded sibling path — fragile if repos not
  side-by-side (CI worktree symlinks recreate this; silent breakage
  since no CI runs the generator).
- **Real duplication = recipe tables**: host `MAP` (~224 rows) vs lib
  `MATH_UNARY`/`PRED`/`BIN`/`DRIVER`/`COMPOSE`/`STUBS`/`FAMILIES` — two
  hand-maintained op→semantics maps + kernels duplicated across
  languages (aten_kernels.rs 4,923 ↔ internal/aten.hnl 3,620). No
  cross-check that tables cover same ops — drift exists (§4).
- Recommendation: shared resolved-spec artifact — host's `--emit-spec`
  already emits it; lib could consume spec JSON instead of
  re-deriving classification. Medium-term.

## 3. Hardcoded vs generated — concrete list

| Item | Verdict |
|---|---|
| `aten_kernels.rs` ~120 `k_*` kernels | Cannot be generated — schema carries no semantics. This IS the port. |
| `internal/aten.hnl` 3,620 lines | Same — the point of the repo. |
| Per-op recipe tables (both gens) | Hand-maintained by necessity but **duplicated knowledge** — biggest drift risk. |
| `aten_unique`/`aten_nonzero_static` drivers | One-offs following emerging pattern: seam closure = handwritten driver + DRIVER entry. `einsum` next (~400-line port). Document as the workflow. |
| **`verify.hnl` (91 lines, 55 checks)** | **SHOULD BE GENERATED** — only covers original 31 handwritten ops; ~195 generated ops have ZERO differential coverage vs host. Generator emitting `eq(name, lt.op(fixture), tensor.op(fixture))` per port-list entry would close it. **Highest-value generation gap.** |
| Golden-file test for lib output | Missing entirely. |
| IMPORTS per family file | Could compute from actual usage. |
| `ltensor` record export splice | Generated but fragile regex (non-greedy to first `\n}`). |

## 4. Inconsistencies between op surfaces

- **9 ops host-only**: `exp2, expm1, log1p, logaddexp2, mish,
  softplus, copysign, nextafter` (math.* leaves) + `einsum` (host has
  real `k_einsum`; lib stub.hnl still dies).
- **`sort`/`where` missing top-level exports**: lib emits `sort_`/
  `where_` + record fields but NO `export sort :=`/`export where :=`
  in index.hnl — `from "lib-tensor" import sort` fails; test comment
  claiming re-export is FALSE. Medium-low.
- **`searchsorted` buggy on lib side** — see §5.
- Doc drift: `aten_port_generator.md` says "195 curated pins" (list
  has 207); `lib_tensor_port_gen.md` lists `unique` as seam though
  closed (self-contradicting).
- `ltensor` record missing comma before generated `abs:` field —
  parser tolerates, fragile.
- Semantic drift otherwise well-managed (documented divergences).

## 5. Bugs

1. **lib `searchsorted` — wrong arg binding + inverted compare.
   MEDIUM-HIGH.** Wrapper binds `rest[1]` (= `out_int32`) as `right`;
   real `right` is rest[2], `side`/`sorter` never read. `.hnl`
   comparator looks inverted for `right=true` vs host upper_bound.
   Fix binding + port `side` handling + recheck vs `k_searchsorted`.
2. lib `--check` verifies nothing — HIGH.
3. No CI drift gate either repo — medium.
4. host partial-regen clobber — medium.
5. `hnl_lit` latent edges: exponent-notation floats unsupported by
   dialect; tiny floats render "0". No current trigger — latent.
6. host `resolve`: `max_args` counts kwonly/dropped args — consistent
   with documented ignored-dtype-args limitation.

## 6. math.* gap — small extension, not deep

Spans 3 repos: declarations `lib-utils/src/math.hnl` (62 ops), Rust
impls `host/.../ops/math.rs` (unary_fn! macros ~1 line each), optional
backends lib-math-handl/lib-math-c. Adding exp2/expm1/log1p/copysign ≈
4 lines Rust + 4 lines .hnl + registry entries each. logaddexp2/mish/
softplus compose in .hnl recipes (report already spells them).
nextafter needs from_bits/to_bits (f64::next_up unstable) — still small.
**~5 leaves unblock 8 of 9 lib stubs.** einsum is the only genuinely
hard stub (~400-line equation parser).

## Recommended actions (priority)

1. Make lib `--check` a real diff gate; wire both `--check`s into ci.sh.
2. Fix `searchsorted` binding + comparator (or stub honestly).
3. Generate verify.hnl-style differential checks for the ~195 ported ops.
4. Add math.{exp2,expm1,log1p,copysign} (+recipes) across lib-utils +
   host plugin-lib-stdlib; flip 8 stubs to generated wrappers.
5. Guard host generator against --include/--exclude + write.
6. Housekeeping: dead MAP rows, sort/where re-export, dedupe IMPORTS,
   stale doc counts.

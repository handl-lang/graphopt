# `torch_port_gen` — schema-derived ATen classification for the torch bridge

`scripts/torch_port_gen.py` generates
`packages/mlopt-torch/src/mlopt_torch/aten_classify.py`, the
schema-derived base layer under the torch bridge's hand-maintained
classification tables.  It is the mlopt-side answer to the audit finding
in `docs/reports/arch_audit_graphopt_packages.md` (`torch_bridge.py`'s
`_ATEN_TO_IR` / `_LOCAL_ATTR_SCHEMA` / `_SCALAR_OPERAND_OPS` drifted
from ATen reality and caused argument-position bugs — `torch.min`
overload collapse, clamp positional mix-ups).

The design mirrors host-tensor's generator
(`handl-lang/host-tensor`: `tools/aten_extract.py` →
`tools/aten_schemas.json` + `tools/aten_port_list.txt` →
`src/bin/aten_port_gen.rs` → `src/aten_port.rs`, documented in
`docs/reports/aten_port_generator.md`).  The **same schema dump** that
drives host-tensor's port table now drives mlopt's export-boundary
classification, so a schema change is a `--check` failure instead of a
silent-wrong export.

## Inputs

| input | default | contents |
|---|---|---|
| `aten_schemas.json` | `../host-tensor/tools/aten_schemas.json` | offline `torch._C._jit_get_all_schemas()` dump — 3,757 schemas for torch 2.14.0+cu130, each `{name, overload, args[{name,type,default,kwonly,mutated}], ret, schema}` |
| `aten_port_list.txt` | `../host-tensor/tools/aten_port_list.txt` | the curated `tensor.<hnl> = aten::<query>` pins — 207 ported aten ops |

Both paths are CLI-overridable (`--schemas`, `--list`, `--out`).

## What is generated

`aten_classify.py` (1,642 lines, deterministic, `ruff format`-clean)
exports four tables:

- **`ATEN_TO_IR`** — 753 entries: every qualified (`min.other`,
  `pow.Tensor_Scalar`, `xlogy.OutTensor`) *and* bare (`min`, `pow`)
  aten name of every ported op → IR op name.  Overload suffixes are
  preserved where load-bearing.
- **`ATTR_SCHEMA`** — 79 IR ops → `{argpos: attr_name}` for
  config-typed positional args (`mean.dim` → `{1: "dim", 2: "keepdim"}`).
- **`SCALAR_OPERAND_OPS`** — 31 IR ops whose `Scalar`-typed positionals
  mint `Const` operands (`pow`, `where`, `eq.*`, `clamp`, `lerp`, …).
- **`CLASSIFY`** — per-qualified-name provenance records
  (`{ir, schema, overloads}`) for audit and `--emit-spec`.

## Derivation rules — what schemas CAN express

- **IR name**: bare aten name, unless `IR_NAME_OVERRIDES` says
  otherwise (`min.other → minimum`, `max.other → maximum`, `t`/`mT`/
  `swapaxes → transpose`, `view → reshape`, `ger → outer`,
  `moveaxis → movedim`, `trace → diag_sum`, `clone → contiguous`).
  The port list's host-tensor `hnl` names are deliberately **not**
  used — mlopt's IR vocabulary is the aten name space.
- **Scalar operand positions**: a `Scalar`/`Scalar?` positional at
  index ≥ 1 in a *Tensor-first* overload marks the IR op as
  scalar-operand-taking.  Only Tensor-first overloads contribute —
  scalar-first siblings (`pow.int`, `eq.float`) describe calls that
  never appear in an FX graph and would collide positionally.
- **Attr schema**: positional args of config type (`int`, `SymInt`,
  `float`, `bool`, `str`, and the `[]`/`[N]` list forms) map
  position → schema arg name.  `Scalar` positions become attrs only
  for ops *outside* the scalar-operand set.  `Tensor`,
  `ScalarType`, `Device`, `Layout`, `MemoryFormat`, `Generator`,
  `Dimname`, `complex` are excluded — the exporter has dedicated
  channels for them.
- Positions are ranks among **non-kwonly** args (kwonly args arrive via
  `node.kwargs`).

## What schemas CANNOT express — the override layer

A schema says *a `Scalar` is a `Scalar`*; it cannot say whether that
scalar is a tensor operand (`pow(x, 2)`) or a bound config attribute
(`celu.alpha`, `linspace.end`, `mean.dim` is `int` so fine, but
`hardtanh.min_val` is `Scalar` config).  `OP_OVERRIDES` in the
generator is the curated recipe for exactly this:

| override | why |
|---|---|
| `celu`, `softshrink`, `hardshrink` — `scalar_operands: false` | `alpha`/`lambd` are config Scalars the bindings read as attrs; schema type is identical to `pow.exponent` |
| `linspace`, `logspace` — `scalar_operands: false` | `Tensor_Scalar`/`Scalar_Tensor` sibling overloads would otherwise pull the op into the scalar set and strand `end`/`steps` in a `Const` operand the `*a`-ignoring binding never reads — bounds are attrs |
| `to` — `attrs: {}` | `prim_*` overloads carry dtype as a raw `int` code at colliding positions; the exporter's dedicated `torch.dtype`/`torch.device` channels own this op |
| `reshape`/`repeat`/`expand`/`view` — `attrs: {"1": "shape"}` | the `SymInt[]`/`int[]` size list must land on the canonical `shape` attr the bindings and shape rules already read, not a schema arg name |
| `IR_NAME_OVERRIDES` (14 renames) | `min.other→minimum` vs `min.dim→min` — same base name, different semantics; schemas give no IR vocabulary |

On the bridge side the hand tables remain, renamed to their override
role: `_ATEN_TO_IR_OVERRIDES` (52 entries — non-ported aten names like
`mul.Scalar`, `__or__`, `special_gammaln`, `deform_conv2d`, plus every
semantic rename), `_LOCAL_ATTR_SCHEMA_OVERRIDES` (13 ops),
`_SCALAR_OPERAND_OPS_OVERRIDES` (38 ops — the union can only *add* ops
the schema derivation missed).  The merged public tables are built
after `_IR_TO_TORCH` exists:

- `_ATEN_TO_IR` = generated entries filtered to **bound ∧
  opmeta-registered** IR ops (identity entries for registry-less ops
  are dropped — `_canon_aten_name`'s base fallback mints the same
  spelling, and `om.validate_aten_map` audits every table *value*)
  under the hand overrides.
- `_SCALAR_OPERAND_OPS` = generated ∪ hand.
- `_LOCAL_ATTR_SCHEMA` = per-op merge, hand entries winning position
  conflicts; substrate `ATTR_SCHEMA` still applies on top at the
  export site.

## Coverage

| table | before (hand) | generated | merged (live) |
|---|---|---|---|
| ATen→IR names | 52 + 23 (`_IR_TO_TORCH_EXTRA`) | 753 qualified+bare over 207 ported ops | 606 |
| attr schemas | 13 ops | 79 ops | 80 ops |
| scalar-operand ops | 38 | 31 | 51 |

Ported-op coverage: all 207 `aten_port_list.txt` pins now resolve a
schema-derived classification.  Of the 207, **55** port names map to IR
ops with no `_IR_TO_TORCH` binding today (`mm`, `bmm`, `mv`, `dot`,
`tensordot`, `corrcoef`, …) — they export as opaque leaves, same as
before; the generated table records their classification for when
bindings land.  `test_contracts.py` still pins every *table value* to a
lowering.

## Regenerate / drift gate

```sh
.venv/bin/python scripts/torch_port_gen.py          # write artifact
.venv/bin/python scripts/torch_port_gen.py --check  # byte-diff gate
.venv/bin/python scripts/torch_port_gen.py --emit-spec  # JSON dump
```

`--check` regenerates in memory and byte-diffs against the checked-in
`aten_classify.py`; any schema-dump or port-list change that was not
regenerated fails with `DRIFT`.  `tests/test_torch_port_gen.py` pins
both directions (clean check + tampered artifact) plus representative
classifications (`min.other→minimum`, `pow` scalar operand,
`celu`/`linspace` config-Scalar exclusions, `mean` attr schema).

The emitter pipes its output through `ruff format` (resolved as the
`.venv` sibling of `sys.executable`, then `PATH`) so the artifact
satisfies the repo's format gate byte-for-byte.

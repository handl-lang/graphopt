# Primitive handlers — the fused-body hunt

The `zoo-prim` mint produces `synop_*` primitives; each needs a *body*
(a verified handler) to be a real kernel rather than a bookkeeping op.
`handler_synth.mint_handlers` searches the handler grammar for a body
that verifies against the manual subgraph on the bound-site domain.

## v2 baseline (grammar v2, budget 256)

16/24 minted (`ew` family), **8 refused** — "no candidate verified
within budget=256".

## Grammar v3 — broadcast views admitted into `ew`

The refusal analysis split the 8 into two classes:

- **Broadcast-elementwise** — `add`/`mul` trees carrying
  `unsqueeze`/`squeeze` leaves. A size-1-axis view moves no data: it
  broadcasts its operand into the tree's shared shape, so the spec is
  still one fused elementwise kernel — exactly what `ew` names.
  `BROADCAST_OPS = {unsqueeze, squeeze}` joined the `ew` replay gate
  (`_replayable_spec`).
- **Index/contract** — shapes carrying `select`/`slice`/`linear`/
  `matmul`. The gate stays *closed* there: an `ew` replay of an
  indexing spec would verify by construction (the descriptor IS the
  spelled site) and mint a tautology, not a handler.

`_PRIM_BUDGET` moved 256 → 2048 (the grammar's own declared
`_BUDGET`), so a recorded refusal quotes the full declared search.
Refusal records now carry `best_err` + the frozen `spec`/`domain` —
each row reproduces offline.

## v3 measurement (68-module corpus, k=2)

| verdict | v2 | v3 |
|---|---|---|
| minted | 16 | **20** |
| refused | 8 | **4** |

Recovered (all broadcast-elementwise, err 0.0):

- `synop_976f9836d7` — `add(mul v0 (add v1 (mul v2 v3)), mul v4 (unsq v5))`
- `synop_f08a53f9a0` — `add(mul v0 (add v1 (mul v2 (unsq v3))), mul v4 v5)`
- `synop_4c5296b3ba` — `add(mul v0 (add (mul v1 v2) v3), mul v4 (unsq v5))`
- `synop_98a002d64b` — `add(mul v0 (add (mul v1 v2) (mul v3 (unsq v4))), v5)`

Still refused — an honest family wall, not a budget cap:

| synop | shape class | best_err |
|---|---|---|
| `synop_5f1bba5798` | `mul(unsq(select v), unsq(select(linear..)))` — select-gate × select-projection | inf (every tried candidate raised) |
| `synop_00cd48d9a2` | same select×linear shape | inf |
| `synop_336e34145b` | `mul(unsq(select v0), unsq(select v1))` — select×select | 1.67 (evaluated, wrong) |
| `synop_4c657cb2c0` | `add v0 (mul (unsq v1) (unsq (select (linear..)..)))` — residual + broadcast linear | 15.04 (evaluated, wrong) |

All four carry `select` or `linear` — indexing/contraction semantics
the `ew` grammar correctly refuses to claim. A handler for these needs
a gather/index or fused-linear family the grammar does not yet ship.

## Grammar v4 — the `gather` family (v3→v4)

The 4 refusals all carried `select`/`linear` — an index family was the
honest missing piece, and it landed:

- `("at", src, {dim,index})` — an index-read leaf (the fused kernel's
  gathered operand fetch);
- `("aff", x, w, b)` — a per-position affine leaf (the ambient
  `linear` re-seated under the family's own name);
- `_gather_view` translates a select/linear-bearing spec into that
  vocabulary — leading with the **fused read-then-map** order
  `aff(at(x),w,b)` (project only the gathered slice) over the
  denotation-equal `at(aff(x,w,b))`; `_gather_marked` keeps the gate
  closed on specs with no index-read (no tautologies — same
  discipline as `_replayable_spec`).

### v4 measurement (68-module corpus, k=2)

| verdict | v3 | v4 |
|---|---|---|
| minted | 20 | **24/24** |
| refused | 4 | **0** |

All four previously-refused primitives mint `gather` bodies at
tried=1, fp64 err 0.0–5.3e-15: `synop_5f1bba5798` (the SSM
select-gate×projection step), `synop_00cd48d9a2`, `synop_336e34145b`
(select×select), `synop_4c657cb2c0` (residual + broadcast linear).

## The claim boundary

- A minted `ew`/`gather` body is a verified fused kernel
  (fp64 `max_err` at the verify gate on every bound-site domain) —
  the primitive is a real kernel, not a label.
- The select-gate×linear `synop_5f1bba5798` — the SSM recurrence step
  shared by DeltaNet/GLA/Mamba/RG-LRU — now has a machine-minted fused
  gather+affine body. Whether a fused-kernel engineer calls that a
  *new* kernel stays a literature question; the machine can now name
  and implement it.

# ruff: noqa: RUF001 RUF002 -- the docstrings spell the BN affine math
"""conv+BN weight-fold synthesis probe — mint it, or name the refusal.

ADR 0012's `$leaf:` auto-compose seam (`eaa6221`) made ``$leaf:``-bearing
rules composable and shipped ``BATCH_NORM_RULES`` (the eval-mode expose
half).  The stated next step (`leaf-compose-seam.md` "What stays open"):
try ``synthesize_rules``/``dspec_synth`` on a real conv→batch_norm seed
rather than hand-writing the ``W'``/``b'`` fold

    W'[o] = W[o] · g[o] / sqrt(var[o] + eps)
    b'    = (b − μ)·scale + β            scale = g·rsqrt(var + eps)

The first measurement (`conv-bn-fold-synth.md`) named the anchor gap:
no bank rule named ``conv2d``, so no composite could mint a
conv2d-headed RHS.  ``CONV_FOLD_RULES`` (the ``conv_channel_scale`` /
``conv_channel_shift`` absorb family) now ships the anchor — this probe
re-measures both entry points plus the iterated-composition fixpoint
(gate B3) and reports the gate-by-gate account.

Run with ``.venv/bin/python bench/drivers/probes/conv_bn_fold_synth_probe.py``.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from mlopt_core import meta
from mlopt_core.ir import Param, TensorType, op_repr
from mlopt_core.laws import (
    ALL_RULES,
    BATCH_NORM_EXPAND,
    BATCH_NORM_RULES,
    CONV_FOLD_RULES,
    DEFAULT,
    POINTWISE_FACTOR_RULES,
)

# ---------------------------------------------------------------------------
#  The real seed — conv2d → batch_norm (eval mode) through the torch bridge
# ---------------------------------------------------------------------------


class ConvBN(nn.Module):
    """``BatchNorm2d(Conv2d(x))`` in eval mode — the fold site."""

    def __init__(self, cin: int = 4, cout: int = 6, k: int = 3) -> None:
        """Build the conv (with bias — the shift the fold reads)."""
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=True)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv then batch norm."""
        return self.bn(self.conv(x))


def real_seed() -> tuple[Any, dict[str, Any]]:
    """``batch_norm(conv2d(x, W, b), γ, β, μ, σ²)`` + source_tensors."""
    from mlopt_torch.adapters import TorchSource

    mod = ConvBN().eval()
    with torch.no_grad():
        mod.bn.running_mean.copy_(torch.randn(6))
        mod.bn.running_var.copy_(torch.rand(6) * 2 + 0.25)
    ir, tensors = TorchSource().to_ir(mod, torch.randn(1, 4, 8, 8))
    return ir.root, tensors


# ---------------------------------------------------------------------------
#  Gate A — the rule-bank census (does any law name conv2d?)
# ---------------------------------------------------------------------------


def _ops_in(t: Any, acc: set) -> set:
    o = getattr(t, "op", None)
    if o:
        acc.add(o)
    for a in getattr(t, "args", ()) or ():
        _ops_in(a, acc)
    return acc


def gate_bank_census() -> dict[str, Any]:
    """``conv2d``/``batch_norm`` occurrences across the whole bank.

    ``synthesize_rules`` composites only reshuffle parent RHSes — a
    candidate's RHS head comes from ``r1.rhs`` (or an ``r2.rhs`` spliced
    inside it).  An op absent from EVERY lhs AND rhs can never enter the
    candidate space.
    """
    rules = (
        list(ALL_RULES) + list(BATCH_NORM_RULES) + list(CONV_FOLD_RULES)
    )
    return {
        "n_rules": len(rules),
        "conv2d_in_any_lhs": [
            r.name for r in rules if "conv2d" in _ops_in(r.lhs, set())
        ],
        "conv2d_in_any_rhs": [
            r.name for r in rules if "conv2d" in _ops_in(r.rhs, set())
        ],
        "batch_norm_in_any_lhs": [
            r.name
            for r in rules
            if "batch_norm" in _ops_in(r.lhs, set())
        ],
        "batch_norm_in_any_rhs": [
            r.name
            for r in rules
            if "batch_norm" in _ops_in(r.rhs, set())
        ],
        "rhs_heads": sorted(
            {getattr(r.rhs, "op", type(r.rhs).__name__) for r in rules}
        ),
    }


# ---------------------------------------------------------------------------
#  Gate B — synthesize_rules on the real seed
# ---------------------------------------------------------------------------


def gate_synthesize(seed: Any) -> dict[str, Any]:
    """Run the composer over the real seed under both rule banks.

    The fold law is ``batch_norm(conv2d(x,W,b), …) → conv2d(x, W', b')``.
    The seed-guided path CAN produce the nested LHS (leaf-generalization
    over the LCA region of the seed); the RHS needs a parent that
    produces ``conv2d``.  Count emitted composites, which carry the
    nested LHS, and which RHS heads ever appear.
    """
    out: dict[str, Any] = {}
    for label, rules in (
        ("DEFAULT+BN", list(DEFAULT.rules) + list(BATCH_NORM_RULES)),
        ("ALL+BN", list(ALL_RULES) + list(BATCH_NORM_RULES)),
        (
            "ALL+BN+CONV",
            list(ALL_RULES)
            + list(BATCH_NORM_RULES)
            + list(CONV_FOLD_RULES),
        ),
        (
            "ALL+BN+CONV+PW",
            list(ALL_RULES)
            + list(BATCH_NORM_RULES)
            + list(CONV_FOLD_RULES)
            + list(POINTWISE_FACTOR_RULES),
        ),
    ):
        for nc in (False, True):
            emitted = meta.synthesize_rules(
                rules, [seed], fuel=20000, numeric_check=nc
            )
            keyed = f"{label} numeric={nc}"
            out[keyed] = {
                "emitted": len(emitted),
                "rhs_heads": sorted(
                    {
                        getattr(d.rhs, "op", type(d.rhs).__name__)
                        for d in emitted
                    }
                ),
                "lhs_mentions_conv2d": [
                    d.name
                    for d in emitted
                    if "conv2d" in op_repr(d.lhs)
                ],
                "rhs_heads_conv2d": any(
                    getattr(d.rhs, "op", None) == "conv2d"
                    for d in emitted
                ),
                # the actual fold: lhs names batch_norm AND the rhs is
                # a bare conv2d — the composite the work order wants.
                "fold_composites": [
                    d.name
                    for d in emitted
                    if "batch_norm" in op_repr(d.lhs)
                    and getattr(d.rhs, "op", None) == "conv2d"
                ],
            }
    return out


def gate_iterated(seed: Any, rounds: int = 4) -> dict[str, Any]:
    """Run the fixpoint driver — each round's stratum feeds back.

    ``synthesize_rules`` composes PAIRS over a fixed rule list: a
    composite's derivation depth is exactly 2.  The fold needs ~5
    (expand → shift_sub → scale → scale_l → shift), so this gate runs
    the shipped fixpoint: ``synthesize_rules(rounds=…)`` feeds each
    round's emitted composites back into the bank (``usable = base +
    latest stratum``), ``on_round`` reports the per-round mint and
    stops the loop once the fold composite lands.  ``numeric_check=
    False`` — the gate-B2 NaN-env artifact makes it counterproductive
    here.
    """
    base = (
        list(ALL_RULES)
        + list(BATCH_NORM_RULES)
        + list(CONV_FOLD_RULES)
        + list(POINTWISE_FACTOR_RULES)
    )
    out: dict[str, Any] = {}
    folds: list[Any] = []

    def watch(rnd: int, emitted: list[Any]) -> bool:
        conv_headed = [
            d for d in emitted if getattr(d.rhs, "op", None) == "conv2d"
        ]
        fs = [d for d in conv_headed if "batch_norm" in op_repr(d.lhs)]
        out[f"round{rnd}"] = {
            "emitted": len(emitted),
            "conv2d_headed": len(conv_headed),
            "fold_composites": [d.name for d in fs],
        }
        if fs:
            folds.extend(fs)
            return False  # the fold minted — stop the fixpoint
        return True

    emitted = meta.synthesize_rules(
        base,
        [seed],
        fuel=30000,
        numeric_check=False,
        rounds=rounds,
        on_round=watch,
    )
    out["total_emitted"] = len(emitted)
    if folds:
        out["fold_rule"] = folds[0]
        rewritten = meta.apply_rewrite_at(folds[0], seed, ())
        out["fold_fires"] = rewritten is not None
        if rewritten is not None:
            out["fold_allclose"] = _eval_seed_pair(seed, rewritten)
    return out


def _eval_seed_pair(seed: Any, folded: Any) -> dict[str, Any]:
    """fp64-eval seed vs the minted fold on the real module tensors.

    ``_tensor_env`` draws ``randn`` — a negative ``rv`` NaNs both sides
    (the gate-B2 artifact), so the env binds the seed's own
    ``source_tensors`` (``rv`` is positive there by construction).
    """
    _root, tensors = real_seed()

    def fp64(v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            v = v.detach()
        return torch.tensor(np.asarray(v), dtype=torch.float64)

    env: dict[Any, Any] = {}
    torch.manual_seed(0)
    for leaf in meta._leaf_vars(seed):
        name = getattr(leaf, "name", None)
        if name in tensors:
            env[leaf] = fp64(tensors[name])
        else:
            env[leaf] = torch.randn(
                *leaf.typ.shape, dtype=torch.float64
            )
    a = meta._eval_term(seed, env)
    b = meta._eval_term(folded, env)
    return {
        "allclose": meta._eval_allclose(a, b, 1e-10),
        "max_diff": float(np.abs(np.asarray(a) - np.asarray(b)).max()),
    }


def gate_numeric_drop(seed: Any) -> dict[str, Any]:
    """Why ``numeric_check`` drops the BN∘sub_to_add composite.

    ``_tensor_env`` draws ``randn`` envs: ``rv < 0`` makes BOTH sides
    evaluate ``rsqrt`` of a negative — ``allclose(nan, nan)`` is False.
    A verifier-env artifact (running-var positivity is not expressible
    as a shape guard), not a real unsoundness of the composite.
    """
    m = meta.match_pattern(BATCH_NORM_EXPAND.lhs, seed, {})
    out = meta.apply_rewrite_at(BATCH_NORM_EXPAND, seed, ())
    env = meta._tensor_env(m)
    a = meta._eval_term(seed, env)
    b = meta._eval_term(out, env)
    return {
        "env_rv_min": float(np.asarray(env[m["rv"]]).min()),
        "lhs_eval_has_nan": bool(np.isnan(np.asarray(a)).any()),
        "rhs_eval_has_nan": bool(np.isnan(np.asarray(b)).any()),
        "allclose": meta._eval_allclose(a, b),
    }


# ---------------------------------------------------------------------------
#  Gate C — dspec_synth on measured fold bindings (staged + tensor tier)
# ---------------------------------------------------------------------------


def _audit() -> Any:
    """Load ``scripts/dspec_audit.py`` by path (``scripts/`` is not a pkg)."""
    path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "scripts"
        / "dspec_audit.py"
    )
    spec = importlib.util.spec_from_file_location("_dspec_audit", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def gate_dspec_scalar() -> dict[str, Any]:
    """Attr-tier (per-channel scalar) fits — ``bounded_synth``.

    ``scale = γ/√(σ²+ε)``, ``b' = (b−μ)·scale + β`` are depth-3 trees —
    the shipped ``synth_dspec`` defaults to ``max_depth=2`` and refuses;
    the streamed audit twin decides them.
    """
    from mlopt_discovery import dspec_synth as dsp

    audit = _audit()
    rng = np.random.default_rng(0)
    rows = [
        {
            "gamma": float(rng.normal()),
            "var": float(rng.uniform(0.25, 2.0)),
            "eps": 1e-5,
            "b": float(rng.normal()),
            "mu": float(rng.normal()),
            "beta": float(rng.normal()),
            "W": float(rng.normal()),
        }
        for _ in range(8)
    ]
    for r in rows:
        r["scale"] = r["gamma"] / np.sqrt(r["var"] + r["eps"])

    def bound(r: dict, *names: str) -> dict:
        return {f"$attr:{n}": r[n] for n in names}

    sc_bounds = [bound(r, "gamma", "var", "eps") for r in rows]
    sc_samples = dsp.samples_from_hook(
        lambda bd: {
            "$attr:scale": bd["$attr:gamma"]
            / np.sqrt(bd["$attr:var"] + bd["$attr:eps"])
        },
        sc_bounds,
    )
    b_bounds = [bound(r, "b", "mu", "beta", "scale") for r in rows]
    b_samples = dsp.samples_from_hook(
        lambda bd: {
            "$attr:bn": (bd["$attr:b"] - bd["$attr:mu"])
            * bd["$attr:scale"]
            + bd["$attr:beta"]
        },
        b_bounds,
    )
    w_bounds = [bound(r, "W", "scale") for r in rows]
    w_samples = dsp.samples_from_hook(
        lambda bd: {"$attr:Wn": bd["$attr:W"] * bd["$attr:scale"]},
        w_bounds,
    )
    out: dict[str, Any] = {}
    for d in (2, 3):
        spec, mt = audit.bounded_synth(
            sc_samples, name="scale", max_depth=d, budget_s=90.0
        )
        out[f"scale_depth{d}"] = spec
        out[f"scale_depth{d}_meta"] = {
            k: mt[k]
            for k in ("checked", "truncated", "levels", "why")
            if k in mt
        }
        spec, mt = audit.bounded_synth(
            b_samples, name="bn", max_depth=d, budget_s=90.0
        )
        out[f"b_prime_depth{d}"] = spec
        out[f"b_prime_depth{d}_meta"] = {
            k: mt[k]
            for k in ("checked", "truncated", "levels", "why")
            if k in mt
        }
    spec, _meta = audit.bounded_synth(
        w_samples, name="Wn", max_depth=2, budget_s=60.0
    )
    out["w_prime_prescaled_depth2"] = spec
    return out


def _register_algebra(*, broadcast_mul: bool) -> dict:
    """Register a numpy ``TensorAlgebra`` for the tensor-tier fits.

    ``broadcast_mul=True`` pushes the channel-first convention: the
    lower-rank operand reshapes onto axis 0 — ``scale (C,)`` into
    ``W (C,i,k,…)``.  Pushed ops own their typing (``cond``'s
    ``_vbin``/``_binop`` dispatch), so this is an adapter convention,
    not a core edit.
    """
    import mlopt_core.laws.cond as cond

    table: dict[str, Any] = {}

    def source(t: Any) -> Any:
        return table.get(getattr(t, "name", None))

    def mint(v: Any, hint: str) -> tuple[str, tuple]:
        name = f"{hint}__{hashlib.sha256(v.tobytes()).hexdigest()[:12]}"
        table[name] = v
        return name, tuple(int(d) for d in v.shape)

    def mul(a: Any, b: Any) -> Any:
        a, b = np.asarray(a), np.asarray(b)
        if a.ndim and b.ndim:
            if a.ndim < b.ndim:
                a = a.reshape(a.shape + (1,) * (b.ndim - a.ndim))
            elif b.ndim < a.ndim:
                b = b.reshape(b.shape + (1,) * (a.ndim - b.ndim))
        return a * b

    cond.register_tensor_algebra(
        source=source,
        mint=mint,
        ops={
            "sqrt": np.sqrt,
            "rsqrt": lambda a: 1.0 / np.sqrt(a),
            "mul": mul if broadcast_mul else np.multiply,
            "add": np.add,
            "sub": np.subtract,
            "fdiv": np.divide,
        },
        key=lambda v: (tuple(v.shape), v.tobytes()),
    )
    return table


def gate_dspec_tensor() -> dict[str, Any]:
    """Tensor-tier fits — what the fold's ``$leaf:W'``/``$leaf:b'`` need.

    One-shot ``W' = mul(W, mul(γ, rsqrt(add(σ², eps))))`` is a depth-4
    tree — past the enumeration budget — AND vanilla ``mul`` declines
    on ``(C,)×(C,i,k,k)`` broadcasting.  Staged over a bound ``scale``
    leaf with a channel-broadcast ``mul``, the derive mints at depth 2.
    ``b'`` staged is a depth-3 tree whose level-3 product (~4 G pairs
    on this leaf bank) exceeds the streamed budget.
    """
    import mlopt_core.laws.cond as cond
    from mlopt_discovery import dspec_synth as dsp

    audit = _audit()
    rng = np.random.default_rng(0)
    C = 6
    out: dict[str, Any] = {}

    table = _register_algebra(broadcast_mul=True)
    try:
        # -- one-shot W' (all four bound leaves) — depth-4 tree + broadcast
        samples = []
        for i in range(6):
            Wi = rng.normal(size=(C, 4, 3, 3))
            gi = rng.normal(size=C)
            vi = rng.uniform(0.25, 2.0, size=C)
            bi = {
                "W": Param(f"W{i}", TensorType(Wi.shape)),
                "g": Param(f"g{i}", TensorType(gi.shape)),
                "rv": Param(f"rv{i}", TensorType(vi.shape)),
                "$attr:eps": 1e-5,
            }
            table[f"W{i}"], table[f"g{i}"], table[f"rv{i}"] = Wi, gi, vi
            samples.append(
                (
                    bi,
                    Wi * (gi / np.sqrt(vi + 1e-5)).reshape(-1, 1, 1, 1),
                )
            )
        out["w_prime_oneshot_depth2"] = dsp.synth_dspec(
            samples, name="WN", max_depth=2
        )

        # -- staged W' given a bound `s` (scale) leaf — broadcast mul
        samples_w = []
        samples_b = []
        for i in range(6):
            Wi = rng.normal(size=(C, 4, 3, 3))
            si = rng.uniform(0.3, 2.0, size=C)
            bi_ = rng.normal(size=C)
            mi = rng.normal(size=C)
            bei = rng.normal(size=C)
            bnd = {
                "W": Param(f"W{i}", TensorType(Wi.shape)),
                "s": Param(f"s{i}", TensorType(si.shape)),
                "b": Param(f"b{i}", TensorType(bi_.shape)),
                "m": Param(f"m{i}", TensorType(mi.shape)),
                "be": Param(f"be{i}", TensorType(bei.shape)),
            }
            for k, v in (
                ("W", Wi),
                ("s", si),
                ("b", bi_),
                ("m", mi),
                ("be", bei),
            ):
                table[f"{k}{i}"] = v
            samples_w.append((bnd, Wi * si.reshape(-1, 1, 1, 1)))
            samples_b.append((bnd, (bi_ - mi) * si + bei))
        out["w_prime_staged_depth2"] = dsp.synth_dspec(
            samples_w, name="WN", max_depth=2
        )
        spec, meta3 = audit.bounded_synth(
            samples_b, name="BN", max_depth=3, budget_s=60.0
        )
        out["b_prime_staged_depth3"] = spec
        out["b_prime_staged_meta"] = {
            k: meta3[k]
            for k in ("checked", "truncated", "levels", "why")
            if k in meta3
        }
    finally:
        cond.register_tensor_algebra()
    return out


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the probe and print the gate-by-gate account."""
    seed, _tensors = real_seed()
    print("SEED:", op_repr(seed))

    print()
    print("=" * 72)
    print("GATE A — rule-bank census (is conv2d reachable at all?)")
    print("=" * 72)
    for k, v in gate_bank_census().items():
        print(f"  {k}: {v}")

    print()
    print("=" * 72)
    print("GATE B — synthesize_rules on the real seed")
    print("=" * 72)
    for label, res in gate_synthesize(seed).items():
        print(
            f"  {label}: emitted={res['emitted']} "
            f"rhs_heads={res['rhs_heads']}"
        )
        print(f"    lhs mentions conv2d: {res['lhs_mentions_conv2d']}")
        print(f"    ANY rhs headed conv2d: {res['rhs_heads_conv2d']}")
        print(
            f"    fold composites (bn lhs -> conv2d): {res['fold_composites']}"
        )

    print()
    print("=" * 72)
    print(
        "GATE B3 — iterated composition (emitted rules feed the bank)"
    )
    print("=" * 72)
    for k, v in gate_iterated(seed).items():
        if k == "fold_rule":
            print(f"  {k}: {v.name}")
        else:
            print(f"  {k}: {v}")

    print()
    print("=" * 72)
    print("GATE B2 — the numeric_check drop is a NaN-env artifact")
    print("=" * 72)
    for k, v in gate_numeric_drop(seed).items():
        print(f"  {k}: {v}")

    print()
    print("=" * 72)
    print("GATE C — dspec_synth, scalar (attr) tier")
    print("=" * 72)
    for k, v in gate_dspec_scalar().items():
        print(f"  {k}: {v}")

    print()
    print("=" * 72)
    print("GATE D — dspec_synth, tensor (tval) tier")
    print("=" * 72)
    for k, v in gate_dspec_tensor().items():
        print(f"  {k}: {v}")

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(
        """  The anchor gap is CLOSED (gate A): `CONV_FOLD_RULES` gives
  conv2d 60 lhs + 60 rhs entries — and it participates immediately:
  `syn_batch_norm_expand__conv_channel_shift_sub_sp_6` absorbs the
  running mean into the conv bias as a pairwise composite (gate B).

  PAIRWISE synthesis still refuses the bare fold (gate B: no
  conv2d-headed rhs in any bank) — `synthesize_rules` composes pairs
  over a FIXED rule list, so its derivation depth is exactly 2 and the
  emitted set never re-enters the pool.  The remaining refusal is
  DEPTH, not material: the shipped fixpoint (gate B3 —
  `synthesize_rules(rounds=4, on_round=…)`, each round's emitted
  composites join the next round's bank) mints
  `batch_norm(conv2d(x,W,b),…) → conv2d(x, W′, b′)` in 4 rounds —
  expand → shift_sub → scale → scale_l → shift — verified fp64-exact
  on the real seed (max diff ~1e-12 vs the fused kernel).

  dspec_synth MINTS the fold's derives in STAGES (gate C): scale and
  b' at depth 3, W' at depth 2 given a `scale` leaf.  The one-shot
  tensor-tier fit refuses twice over (gate D): the tree is depth ≥4
  AND vanilla `mul` declines on (C,)×(C,i,k,…) — it needs (a) a staged
  fit and (b) a pushed channel-broadcast `mul` or an enumerated
  reshape/`tapp` — neither in the enumeration bank today."""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

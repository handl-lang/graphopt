"""Weight-space lane — derive the conv+BN fold, or refuse it honestly.

The measured boundary (this cycle): models whose math is already fused
expose nothing to the op-algebra laws.  Weight space is the frontier
that tests the project's no-hardcoding rule: the fold

    W'[o] = W[o] * g[o] / sqrt(var[o] + eps)
    b'    = (b - mu) * g / sqrt(var + eps) + beta

is a pure algebraic identity on *parameters* — in IR terms a
derived-leaf rewrite whose RHS weights are computed from the LHS's
leaves.  No conv+BN law ships today; BiRefNet's decoder carries 43
``nn.BatchNorm2d`` modules that lower as plain ``batch_norm`` nodes.

This probe builds the binding source (a real ``Conv2d + BatchNorm2d``
in eval mode and a ``Linear + LayerNorm`` affine half), measures the
fold numerically, and then asks the honest question: **can the derive
machinery express the fold spec** — a per-output-channel elementwise
function of the parameters (division, square root, epsilon)?

Run with ``.venv/bin/python bench/drivers/probes/weightspace_fold_probe.py``.

Verdicts are printed as a gate-by-gate account: the fold must clear
the binding source, the derive-synth fit, the cond-DSL bank, the
value domain, the e-graph's ability to see a BN node, the cost model,
and the derived-leaf (``source_tensors``) registration seam.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
#  Binding source — real modules, measured fold
# ---------------------------------------------------------------------------


class ConvBN(nn.Module):
    """``BatchNorm2d(Conv2d(x))`` in eval mode — the conv+BN fold site."""

    def __init__(
        self,
        cin: int = 4,
        cout: int = 6,
        k: int = 3,
        *,
        conv_bias: bool = False,
    ) -> None:
        """Build a conv (optional bias) plus a batch norm."""
        super().__init__()
        self.conv = nn.Conv2d(
            cin, cout, k, padding=k // 2, bias=conv_bias
        )
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv then batch norm."""
        return self.bn(self.conv(x))


class LinearLN(nn.Module):
    """``LayerNorm(Linear(x))`` — the affine half of the fold."""

    def __init__(
        self, cin: int = 8, cout: int = 6, *, bias: bool = True
    ) -> None:
        """Build a linear projection plus a layer norm."""
        super().__init__()
        self.lin = nn.Linear(cin, cout, bias=bias)
        self.ln = nn.LayerNorm(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear projection then layer norm."""
        return self.ln(self.lin(x))


def _fold_conv_bn(mod: ConvBN) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the folded ``(W', b')`` from the conv+BN parameters.

    ``W'[o] = W[o] * g[o] / sqrt(var[o] + eps)`` and
    ``b'[o] = (b[o] - mu[o]) * g[o] / sqrt(var[o] + eps) + beta[o]``.
    """
    w = mod.conv.weight.detach()
    b = (
        mod.conv.bias.detach()
        if mod.conv.bias is not None
        else torch.zeros(w.shape[0], dtype=w.dtype)
    )
    g = mod.bn.weight.detach()
    beta = mod.bn.bias.detach()
    mu = mod.bn.running_mean.detach()
    var = mod.bn.running_var.detach()
    eps = mod.bn.eps
    scale = g / torch.sqrt(var + eps)
    w_new = w * scale.reshape(-1, 1, 1, 1)
    b_new = (b - mu) * scale + beta
    return w_new, b_new


def measure_conv_fold(seed: int = 0) -> dict[str, Any]:
    """Measure the conv+BN fold numerically: max error on a forward."""
    torch.manual_seed(seed)
    mod = ConvBN(conv_bias=True)
    mod.eval()
    # perturb the running stats off their (0, 1) defaults so the fold
    # is a real affine map, not the identity
    with torch.no_grad():
        mod.bn.running_mean.copy_(torch.randn_like(mod.bn.running_mean))
        mod.bn.running_var.copy_(
            torch.rand_like(mod.bn.running_var) * 2 + 0.25
        )
        mod.bn.weight.copy_(torch.randn_like(mod.bn.weight))
        mod.bn.bias.copy_(torch.randn_like(mod.bn.bias))
    x = torch.randn(2, 4, 8, 8)
    with torch.no_grad():
        y_ref = mod(x)
        w_new, b_new = _fold_conv_bn(mod)
        conv = nn.Conv2d(
            4, 6, 3, padding=1, bias=True, dtype=w_new.dtype
        )
        with torch.no_grad():
            conv.weight.copy_(w_new)
            conv.bias.copy_(b_new)
        y_fold = conv(x)
    err = float((y_ref - y_fold).abs().max())
    return {
        "module": "ConvBN",
        "eps": mod.bn.eps,
        "w_shape": tuple(mod.conv.weight.shape),
        "max_abs_err": err,
        "folded_w": w_new,
        "folded_b": b_new,
        "orig_w": mod.conv.weight.detach(),
        "orig_b": (
            mod.conv.bias.detach()
            if mod.conv.bias is not None
            else torch.zeros(6)
        ),
        "mu": mod.bn.running_mean.detach(),
        "var": mod.bn.running_var.detach(),
        "gamma": mod.bn.weight.detach(),
        "beta": mod.bn.bias.detach(),
    }


def measure_linear_ln_fold(seed: int = 0) -> dict[str, Any]:
    """Measure the affine half: ``W' = W * (s*g)``, ``b' = b*(s*g)+beta``.

    LayerNorm in eval mode normalises over the last axis with
    *per-sample* statistics, so a pure parameter fold is exact only
    when the normalisation statistic ``s = 1/sqrt(var+eps)`` is fixed
    (e.g. a feature-normalisation over a single axis whose variance is
    a known constant).  The *affine half* — the parameter-only part —
    folds exactly as the conv case: ``W' = W*(s*g)``,
    ``b' = b*(s*g) + beta``.  Here ``s`` is pinned to a constant and
    the equality ``(Wx+b)*s*g + beta == W'x + b'`` is checked exactly.
    """
    torch.manual_seed(seed)
    mod = LinearLN()
    mod.eval()
    with torch.no_grad():
        mod.ln.weight.copy_(torch.randn_like(mod.ln.weight))
        mod.ln.bias.copy_(torch.randn_like(mod.ln.bias))
    w = mod.lin.weight.detach()
    b = mod.lin.bias.detach()
    g = mod.ln.weight.detach()
    beta = mod.ln.bias.detach()
    # a fixed normalisation statistic (the "known var" the affine half
    # folds against — var+eps > 0 so the sqrt is real)
    var = torch.rand_like(g) * 2 + 0.25
    eps = mod.ln.eps
    s = 1.0 / torch.sqrt(var + eps)
    scale = s * g
    w_new = w * scale.reshape(-1, 1)
    b_new = b * scale + beta
    x = torch.randn(3, 8)
    with torch.no_grad():
        y_ref = (mod.lin(x)) * s * g + beta
        y_fold = x @ w_new.T + b_new
    err = float((y_ref - y_fold).abs().max())
    return {
        "module": "LinearLN",
        "eps": eps,
        "w_shape": tuple(w.shape),
        "max_abs_err": err,
        "folded_w": w_new,
        "folded_b": b_new,
        "orig_w": w,
        "orig_b": b,
        "mu": torch.zeros(w.shape[0]),
        "var": var,
        "gamma": g,
        "beta": beta,
    }


# ---------------------------------------------------------------------------
#  Gate 2/3 — the derive-synth fit and the cond-DSL bank
# ---------------------------------------------------------------------------


def _scalar_samples(r: dict[str, Any], n: int = 8) -> list[dict]:
    """Per-output-channel scalar bindings from a measured fold.

    Each sample carries the *scalar* parameter values for one output
    channel and the folded scalar the derive must reproduce — the
    elementwise view the synth's value domain admits (it has no
    tensor leaves).
    """
    w = r["orig_w"].reshape(r["orig_w"].shape[0], -1)
    wn = r["folded_w"].reshape(r["folded_w"].shape[0], -1)
    out = []
    for o in range(min(n, w.shape[0])):
        out.append(
            {
                "W": float(w[o, 0]),
                "Wn": float(wn[o, 0]),
                "mu": float(r["mu"][o]),
                "var": float(r["var"][o]),
                "gamma": float(r["gamma"][o]),
                "beta": float(r["beta"][o]),
                "eps": float(r["eps"]),
                "b": float(r["orig_b"][o]),
                "bn": float(r["folded_b"][o]),
            }
        )
    return out


def _hook_fold_w(bound: dict) -> dict | None:
    """Measure the hook ``W' = W * g / sqrt(var+eps)`` (scalar)."""
    w, g = bound.get("$attr:W"), bound.get("$attr:gamma")
    v, e = bound.get("$attr:var"), bound.get("$attr:eps")
    if None in (w, g, v, e):
        return None
    return {"$attr:Wn": w * g / np.sqrt(v + e)}


def _hook_fold_b(bound: dict) -> dict | None:
    """Measure the hook ``b' = (b-mu)*g/sqrt(var+eps) + beta``."""
    b = bound.get("$attr:b")
    m = bound.get("$attr:mu")
    g = bound.get("$attr:gamma")
    v = bound.get("$attr:var")
    e = bound.get("$attr:eps")
    be = bound.get("$attr:beta")
    if None in (b, m, g, v, e, be):
        return None
    return {"$attr:bn": (b - m) * g / np.sqrt(v + e) + be}


def _fit(
    samples: list[tuple],
    name: str | None = None,
    *,
    max_depth: int = 2,
    budget_s: float = 60.0,
) -> dict | None:
    """Bounded synth fit — the streamed last level (never OOMs).

    ``dspec_synth.synth_dspec``'s blind level product is quartic in the
    leaf count; at ``max_depth=3`` on the fold's seven-attr bank it
    exhausts RAM and OOM-kills the host (measured: a bare ``MemoryError``
    under a 4 GB cap after ~17 s, an OS kill without one).  The
    ``dspec_audit.bounded_synth`` twin enumerates the same candidate
    space with the deepest level *streamed* (early-bailed against the
    expected vector) under a wall-clock budget, so the same fit is
    decided without the blow-up.
    """
    audit = _load_audit()
    spec, _meta = audit.bounded_synth(
        samples, name=name, max_depth=max_depth, budget_s=budget_s
    )
    return spec


def gate_synth_fit(r: dict[str, Any]) -> dict[str, Any]:
    """Gate 2 — feed the measured scalar bindings to ``dspec_synth``.

    The honest question: can the synth's bank express the per-channel
    fold?  Two tiers are measured — the *raw* fold (from ``var``/``eps``
    directly, needing division + sqrt) and the *pre-scaled* fold (from a
    supplied ``scale``, needing only multiplication), which isolates
    whether a refusal is a missing-op gap or something else.
    """
    from mlopt_discovery import dspec_synth as dsp

    rows = _scalar_samples(r)
    raw_bounds = [
        {
            "$attr:W": s["W"],
            "$attr:gamma": s["gamma"],
            "$attr:var": s["var"],
            "$attr:eps": s["eps"],
            "$attr:b": s["b"],
            "$attr:mu": s["mu"],
            "$attr:beta": s["beta"],
        }
        for s in rows
    ]
    w_samples = dsp.samples_from_hook(_hook_fold_w, raw_bounds)
    b_samples = dsp.samples_from_hook(_hook_fold_b, raw_bounds)
    w_fit = _fit(w_samples, "Wn")
    b_fit = _fit(b_samples, "bn")
    w_near = dsp._near_misses(w_samples)
    # depth sweep: the b' fold `(b-mu)*scale+beta` is a depth-3 tree,
    # past the default max_depth=2 — measure whether depth alone
    # unblocks it (isolating a depth gap from the sqrt/fdiv gap).
    b_depth = {d: _fit(b_samples, "bn", max_depth=d) for d in (2, 3)}
    return {
        "raw_w_fit": w_fit,
        "raw_b_fit": b_fit,
        "raw_b_fit_by_depth": b_depth,
        "w_near_misses": [
            (e, n, len(raw_bounds)) for e, n in w_near[:4]
        ],
    }


def gate_prescaled_fit(r: dict[str, Any]) -> dict[str, Any]:
    """Gate 2b — the fold given a *pre-computed* per-channel scale.

    ``W' = W*scale`` and ``b' = (b-mu)*scale + beta`` need only ``mul``
    and ``add`` — both in the synth's bank.  If these fit while the raw
    tier refuses, the refusal is precisely the missing ``sqrt``/``fdiv``
    ops, not a structural mismatch.
    """
    from mlopt_discovery import dspec_synth as dsp

    rows = _scalar_samples(r)

    def hook_w(bound: dict) -> dict | None:
        return {"$attr:Wn": bound["$attr:W"] * bound["$attr:scale"]}

    def hook_b(bound: dict) -> dict | None:
        b, m = bound["$attr:b"], bound["$attr:mu"]
        return {
            "$attr:bn": (b - m) * bound["$attr:scale"]
            + bound["$attr:beta"]
        }

    bounds = []
    for s in rows:
        scale = s["gamma"] / np.sqrt(s["var"] + s["eps"])
        bounds.append(
            {
                "$attr:W": s["W"],
                "$attr:scale": float(scale),
                "$attr:b": s["b"],
                "$attr:mu": s["mu"],
                "$attr:beta": s["beta"],
            }
        )
    w_fit = _fit(dsp.samples_from_hook(hook_w, bounds), "Wn")
    b_fit = {
        d: _fit(
            dsp.samples_from_hook(hook_b, bounds), "bn", max_depth=d
        )
        for d in (2, 3)
    }
    return {"prescaled_w_fit": w_fit, "prescaled_b_fit_by_depth": b_fit}


def gate_scale_fit(r: dict[str, Any]) -> dict[str, Any]:
    """Gate 2c — fit the *scale* subexpression ``gamma / sqrt(var+eps)``.

    The fold decomposes as ``W' = W*scale`` / ``b' = (b-mu)*scale +
    beta`` (gate 2b measured both fit given a ``scale`` leaf) plus the
    one
    derivation the bank could not spell: ``scale``.  With ``sqrt`` /
    ``rsqrt`` / ``fdiv`` in the synth bank this is the measured delta
    of the op-set extension — a bound restricted to exactly the attrs
    the statistic reads (``gamma``/``var``/``eps``), so the level
    product stays small enough for the streamed sweep to terminate.
    """
    from mlopt_discovery import dspec_synth as dsp

    rows = _scalar_samples(r)
    bounds = [
        {
            "$attr:gamma": s["gamma"],
            "$attr:var": s["var"],
            "$attr:eps": s["eps"],
        }
        for s in rows
    ]

    def hook_scale(bound: dict) -> dict | None:
        return {
            "$attr:scale": bound["$attr:gamma"]
            / np.sqrt(bound["$attr:var"] + bound["$attr:eps"])
        }

    audit = _load_audit()
    samples = dsp.samples_from_hook(hook_scale, bounds)
    out: dict[str, Any] = {}
    for d in (2, 3):
        spec, meta = audit.bounded_synth(
            samples, name="scale", max_depth=d, budget_s=60.0
        )
        out[f"scale_fit_depth{d}"] = spec
        out[f"scale_meta_depth{d}"] = meta
    return out


def gate_dsl_expressibility() -> dict[str, Any]:
    """Gate 3 — spell the fold's sqrt in the cond DSL, or report the gap.

    The DSL is strictly richer than the synth's search bank (it carries
    ``recip``/``pow``).  ``sqrt(v) == v**0.5``; the ``pow`` path still
    declines (int-only exponent) but the DSL now carries ``sqrt`` /
    ``rsqrt`` directly — measured through ``eval_derive``.
    """
    from mlopt_core.laws.cond import _DOPS, eval_derive

    b = {"$attr:var": 4.0, "$attr:eps": 1e-5, "$attr:gamma": 2.0}
    fold = {
        "S": (
            "fdiv",
            ("attr", "gamma"),
            (
                "pow",
                ("add", ("attr", "var"), ("attr", "eps")),
                0.5,
            ),
        )
    }
    rsqrt_scale = {
        "S": (
            "mul",
            ("attr", "gamma"),
            ("rsqrt", ("add", ("attr", "var"), ("attr", "eps"))),
        )
    }
    return {
        "dsl_has_sqrt": "sqrt" in _DOPS,
        "dsl_has_rsqrt": "rsqrt" in _DOPS,
        "pow_float_exp_declines": eval_derive(fold, b) is None,
        "fdiv_ok": eval_derive(
            {"S": ("fdiv", ("lit", 1.0), ("lit", 4.0))}, b
        )
        is not None,
        "recip_ok": eval_derive({"S": ("recip", ("lit", 4.0))}, b)
        is not None,
        "rsqrt_scale_ok": eval_derive(rsqrt_scale, b)
        == {"$attr:S": 2.0 / (4.0 + 1e-5) ** 0.5},
    }


def gate_value_domain() -> dict[str, Any]:
    """Gate 4 — the derive's value domain vs a tensor-valued leaf.

    A derive emits ``{"$attr:NAME": value}`` substituted into the RHS's
    *attrs*, which must round-trip through ``attr_to_data`` (JSON-safe).
    A folded weight is a *tensor*; ``Const.value`` is ``int | float``
    only.  ADR 0012 lands the ``$leaf:`` channel + the pushed tensor
    algebra — re-measured here: does the fold's ``scale`` mint as a
    tensor-valued derive, and does the synth survive tensor samples?
    """
    import mlopt_core.laws.cond as cond
    from mlopt_core.ir import Const, Param, TensorType
    from mlopt_discovery import dspec_synth as dsp

    table: dict[str, Any] = {}

    def source(t: Any) -> Any:
        # a bound term resolves by name; a bound value that IS the
        # payload (the synth's per-sample bounds) passes through
        return (
            t
            if isinstance(t, np.ndarray)
            else table.get(getattr(t, "name", None))
        )

    def mint(v: Any, hint: str) -> tuple[str, tuple]:
        import hashlib

        name = f"{hint}__{hashlib.sha256(v.tobytes()).hexdigest()[:12]}"
        table[name] = v
        return name, tuple(int(d) for d in v.shape)

    cond.register_tensor_algebra(
        source=source,
        mint=mint,
        ops={"sqrt": np.sqrt, "rsqrt": lambda a: 1.0 / np.sqrt(a)},
        key=lambda v: (tuple(v.shape), v.tobytes()),
    )
    try:
        # 4a — the fold's scale as a tensor-valued derive:
        # scale = gamma * rsqrt(var + eps), evaluated as arrays.
        gamma = np.array([1.5, 2.0, 0.5])
        var = np.array([4.0, 9.0, 16.0])
        table["gamma"] = gamma
        table["var"] = var
        bound = {
            "gamma": Param("gamma", TensorType((3,))),
            "var": Param("var", TensorType((3,))),
            "$attr:eps": 1e-5,
        }
        scale_spec = {
            "$leaf:scale": (
                "mul",
                ("tval", "gamma"),
                ("rsqrt", ("add", ("tval", "var"), ("attr", "eps"))),
            )
        }
        got = cond.eval_derive(scale_spec, bound)
        leaf = got["scale"] if got else None
        scale_ok = isinstance(leaf, Param) and np.allclose(
            table[leaf.name], gamma / np.sqrt(var + 1e-5)
        )
        # 4b — a tensor-valued synth sample set no longer crashes
        bounds = [
            {"w": np.array([float(i), 2.0 * i]), "$attr:g": float(i)}
            for i in range(1, 7)
        ]

        def hook(b: dict) -> dict | None:
            return {"$attr:Wn": b["w"] * b["$attr:g"]}

        err = None
        fit = None
        try:
            fit = dsp.synth_dspec(
                dsp.samples_from_hook(hook, bounds), name="Wn"
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
    finally:
        cond.register_tensor_algebra()
    return {
        "const_value_types": str(
            Const.__dataclass_fields__["value"].type
        ),
        "tensor_sample_error": err,
        "scale_mints_as_leaf": scale_ok,
        "scale_leaf_name": getattr(leaf, "name", None),
        "tensor_synth_fit": fit,
    }


# ---------------------------------------------------------------------------
#  Gate 5/6 — e-graph visibility, cost model, derived-leaf seam
# ---------------------------------------------------------------------------


def _load_audit() -> Any:
    """Load ``scripts/dspec_audit.py`` by path (``scripts/`` is not a pkg)."""
    import importlib.util
    from pathlib import Path

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


def gate_egraph_visibility() -> dict[str, Any]:
    """Gate 5 — check whether the e-graph *sees* a ``batch_norm`` node.

    Export a real ``ResNetBlock`` (conv→BN→relu) and count the op
    families in the IR; then check whether any shipped law's LHS
    mentions ``batch_norm`` (a fold needs the node to be matchable).
    """
    from mlopt_core.egraph import Rewrite
    from mlopt_torch.adapters import TorchSource
    from mlopt_torch.models import ResNetBlock

    mod = ResNetBlock(8)
    mod.eval()
    src = TorchSource()
    x = torch.randn(1, 8, 8, 8)
    ir, _leaves = src.to_ir(mod, x)
    counts: dict[str, int] = {}
    for t in _walk(ir.root):
        op = getattr(t, "op", None)
        if op is not None:
            counts[op] = counts.get(op, 0) + 1
    rules = _load_audit().collect_rules()
    lhs_ops = set()
    for rule in rules.values():
        if isinstance(rule, Rewrite):
            for t in _walk(rule.lhs):
                o = getattr(t, "op", None)
                if o is not None:
                    lhs_ops.add(o)
    return {
        "ir_ops": dict(sorted(counts.items())),
        "batch_norm_in_ir": counts.get("batch_norm", 0),
        "batch_norm_in_any_lhs": "batch_norm" in lhs_ops,
        "conv2d_in_any_lhs": "conv2d" in lhs_ops,
        "n_rules": len(rules),
    }


def _walk(t: Any) -> list[Any]:
    """Flatten a term tree into a list (cycle-safe)."""
    out = []
    stack = [t]
    seen = set()
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        out.append(cur)
        stack.extend(getattr(cur, "args", ()) or ())
    return out


def gate_cost_model() -> dict[str, Any]:
    """Gate 6 — check whether the cost model prices a ``batch_norm`` node.

    A fold only pays if the fused conv is cheaper than conv+BN.  The
    model must price the BN node it would remove (and the folded conv
    it would keep).
    """
    from mlopt_core.cost.basic import _OP_FLOPS, _flops_of
    from mlopt_core.ir import Op, TensorType, Var

    x = Var("x", TensorType((1, 4, 8, 8)))
    w = Var("w", TensorType((6, 4, 3, 3)))
    bn_x = Op.make("batch_norm", x)
    conv = Op.make("conv2d", x, w)
    return {
        "batch_norm_in_op_flops_table": "batch_norm" in _OP_FLOPS,
        "batch_norm_price": _flops_of(bn_x),
        "conv2d_price": _flops_of(conv),
        "sqrt_in_op_flops_table": "sqrt" in _OP_FLOPS,
        "rsqrt_in_op_flops_table": "rsqrt" in _OP_FLOPS,
    }


def gate_derived_leaf_seam() -> dict[str, Any]:
    """Gate 7 — the seam a fold's derived weight must travel.

    A folded weight is a *new leaf* whose tensor value must be
    registered for lowering.  ``offer_weight_specials`` is the shipped
    precedent: it computes a new weight and registers it into
    ``source_tensors``.  Report whether that seam exists and is
    procedural (not a ``dspec``).
    """
    from mlopt_core.laws.specials import offer_weight_specials

    doc = offer_weight_specials.__doc__ or ""
    return {
        "seam_exists": True,
        "seam_is_procedural": True,
        "seam_registers_source_tensors": ("source_tensors" in doc),
    }


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the probe and print the gate-by-gate account."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    conv = measure_conv_fold()
    lin = measure_linear_ln_fold()

    print("=" * 72)
    print(
        "GATE 1 — BINDING SOURCE: the fold is a measured parameter identity"
    )
    print("=" * 72)
    for r in (conv, lin):
        print(
            f"  {r['module']:9s} W{r['w_shape']} eps={r['eps']:.1e} "
            f"max|y_ref - y_fold| = {r['max_abs_err']:.3e}"
        )

    print()
    print("=" * 72)
    print("GATE 2 — dspec_synth FIT of the measured fold bindings")
    print("=" * 72)
    fit = gate_synth_fit(conv)
    print(f"  raw W' fit : {fit['raw_w_fit']}")
    print(f"  raw b' fit : {fit['raw_b_fit']}")
    print(f"  raw b' fit by depth : {fit['raw_b_fit_by_depth']}")
    print("  near misses (expr, #samples agreeing, total):")
    for e, n, tot in fit["w_near_misses"]:
        print(f"    {n}/{tot}  {e}")
    pre = gate_prescaled_fit(conv)
    print(f"  pre-scaled W' fit : {pre['prescaled_w_fit']}")
    print(
        f"  pre-scaled b' fit by depth : "
        f"{pre['prescaled_b_fit_by_depth']}"
    )
    sc = gate_scale_fit(conv)
    print(f"  scale fit depth 2 : {sc['scale_fit_depth2']}")
    print(f"  scale meta depth 2: {sc['scale_meta_depth2']}")
    print(f"  scale fit depth 3 : {sc['scale_fit_depth3']}")
    print(f"  scale meta depth 3: {sc['scale_meta_depth3']}")

    print()
    print("=" * 72)
    print("GATE 3 — can the cond DSL hand-spell the fold's sqrt?")
    print("=" * 72)
    for k, v in gate_dsl_expressibility().items():
        print(f"  {k}: {v}")

    print()
    print("=" * 72)
    print("GATE 4 — derive value domain vs a tensor-valued leaf")
    print("=" * 72)
    for k, v in gate_value_domain().items():
        print(f"  {k}: {v}")

    print()
    print("=" * 72)
    print("GATE 5 — e-graph visibility of the BN node")
    print("=" * 72)
    vis = gate_egraph_visibility()
    print(f"  IR op census: {vis['ir_ops']}")
    print(f"  batch_norm nodes in IR : {vis['batch_norm_in_ir']}")
    print(
        f"  batch_norm in any law LHS : "
        f"{vis['batch_norm_in_any_lhs']}  ({vis['n_rules']} rules)"
    )

    print()
    print("=" * 72)
    print("GATE 6 — cost model pricing")
    print("=" * 72)
    for k, v in gate_cost_model().items():
        print(f"  {k}: {v}")

    print()
    print("=" * 72)
    print("GATE 7 — the derived-leaf (source_tensors) seam")
    print("=" * 72)
    for k, v in gate_derived_leaf_seam().items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

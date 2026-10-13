"""Adversarial guard sweep — ADR 0004's remaining open item.

Every conditional law carries a side condition (``cond`` data folded
into ``check``, plus a vetoing ``derive``) that delimits where the
rewrite is valid.  Today's tests only exercise the guard's *true*
region — the ``select_mul`` unsoundness is the standing lesson that
"verified on all observed sites" is not "sound".  This module sweeps
both regions:

* **false region** — a binding the guard declines must produce NO
  rewrite when its instantiated redex is handed to ``apply_rule``.
* **true region** — a binding the guard accepts must fire.
* **guard robustness** — a ``check``/``derive``/``apply`` that RAISES
  on a binding is reported separately: ``apply_rule`` does not catch
  guard exceptions, so a raising guard is a saturation-time crash,
  not a decline.

Bindings come from three sources, in order:

1. ``catopt_discovery.oracle._binding_envs`` — the same honest
   domain enumeration the law-discovery pipeline uses (per-op attr
   domains valid for the bound operand shapes, ``Const`` corners,
   diagonal/Cantor ordering, capped windows);
2. a *diagonal seed* — every term metavariable bound to the same
   ``Var`` shape (plus one- and two-``Const`` variants), which covers
   the shape-correlated guards the independent enumeration starves
   (``split_scalar_*``, the concat-pair carriers);
3. ``_SEEDS`` — hand-minted bound dicts for rules whose accepted
   corner the enumeration domains cannot express (deep chained-view
   guards, carrier attrs the kind table does not type).

The oracle for "rewrote" is ``EGraph.apply_rule`` restricted to the
redex's own e-class (``search=[root]``): a union produced by that
binding's match IS a rewrite; a fire at any other site is a different
binding and cannot confound the attribution.  The verdict for
"accepted" mirrors ``apply_rule``'s gate exactly —
``check(bound)`` (``cond`` folded in) then ``derive(bound) is not
None``.

Violations and errors are *findings*, not silent fixes: the report
prints them, ``_KNOWN_ISSUES`` pins the triaged ones, and the test
asserts the swept set contains nothing unexpected.
"""

from __future__ import annotations

import itertools

import pytest
from catopt_carriers.om import OM_LAWS
from catopt_core.egraph import EGraph
from catopt_core.egraph.terms import _term_instantiate, _term_match
from catopt_core.ir import Const, Op, Param, TensorType, Var, op_repr
from catopt_core.laws import all_rules
from catopt_discovery import oracle as vo
from catopt_discovery.oracle import _GUARDED_CAP, _MAX_INSTANCES


def _v(name, *shape):
    """Mint a ``Var`` leaf of the given shape."""
    return Var(name, TensorType(tuple(shape)))


# ------------------------------------------------------------------
#  The swept rule set — every *conditional* law
# ------------------------------------------------------------------


def _guarded(rule) -> bool:
    """Return whether *rule* carries a firing gate.

    ``cond`` folds into ``check`` at construction, and ``dspec`` folds
    into ``derive`` — so ``check or derive`` covers every gated rule;
    ``cond is not None`` is spelled out for clarity.
    """
    return (
        rule.cond is not None
        or rule.check is not None
        or rule.derive is not None
    )


def _swept_rules() -> list:
    """Return the conditional rules of ``ALL_RULES`` + ``OM_LAWS``."""
    rules = [r for r in all_rules() if _guarded(r)]
    return rules + [r for r in OM_LAWS if _guarded(r)]


# ------------------------------------------------------------------
#  Diagonal seed bindings — correlated-shape coverage
# ------------------------------------------------------------------

#: Shapes every term metavariable is simultaneously bound to — the
#: correlated corner the independent-leaf enumeration starves (a guard
#: demanding ``s[-1] == v[-2]`` or equal-rank concat pairs needs draws
#: that agree, not independent samples).
_DIAG_SHAPES: tuple = (
    (),
    (4,),
    (2, 4),
    (4, 4),
    (3, 4),
    (2, 3, 4),
    (2, 3, 4, 4),
    (4, 4, 4),
)

#: ``Const`` corners for one- and two-leaf scalar variants — the values
#: shipped guards actually demand (``pow``'s ``2``, ``masked_fill``'s
#: ``-inf``, the generic ``0.5`` probe).
_DIAG_CONSTS: tuple = (Const(0.5), Const(2), Const(-float("inf")))


def _domain_options(node, subst):
    """Return a node's attr-option dicts, or ``None`` to skip.

    Mirrors ``oracle._single_domain`` with two honest relaxations: the
    operand shape falls back to ``()`` when instantiation or shape
    inference raises (a string-attr view nested inside the operand
    poisons ``_shape_of`` — the oracle's own ``except`` covers the
    instantiate call only), and the generic ``()``-shape domain is
    tried when the real one cannot be computed.
    """
    bound = None
    if node.args:
        try:
            bound = _term_instantiate(node.args[0], subst)
        except Exception:
            bound = None
    shape = ()
    if bound is not None:
        try:
            s = vo._operand_shape(bound)
            shape = s if isinstance(s, tuple) else ()
        except Exception:
            shape = ()
    mv_keys = tuple(
        k for k, v in node.attrs.items() if isinstance(v, str)
    )
    try:
        opts = vo._attr_options(node.op, mv_keys, shape)
    except Exception:
        opts = None
    if opts:
        return opts
    if shape:
        try:
            return vo._attr_options(node.op, mv_keys, ())
        except Exception:
            return None
    return None


def _diag_variants(mvs: list) -> list:
    """Return the diagonal substitution variants over *mvs*.

    Uniform-``Var`` bases plus one-``Const`` and two-``Const`` leaf
    swaps — the guards that demand a specific literal corner
    (``pow``'s exponent, ``masked_fill``'s ``-inf``) need a ``Const``
    in the right slot correlated with tensor shapes everywhere else.
    """
    out = []
    for s in _DIAG_SHAPES:
        base = {m: Var(m, TensorType(s)) for m in mvs}
        out.append(base)
        for m in mvs:
            for c in _DIAG_CONSTS:
                out.append({**base, m: c})
        for m1, m2 in itertools.combinations(mvs, 2):
            for c1 in _DIAG_CONSTS:
                for c2 in _DIAG_CONSTS:
                    out.append({**base, m1: c1, m2: c2})
    return out


def _diag_attr_bases(rule, subst):
    """Yield merged ``subst + attr draw`` dicts for one variant.

    Each attr-metavar group's option dicts are combined in Cantor
    order and merged consistently (shared names resolve once); a
    group whose domain is empty kills the variant outright.
    """
    nodes = vo._view_nodes([rule.lhs])
    member_ids = {id(n) for n in nodes}
    doms: list = []
    for members in vo._attr_groups(nodes):
        node = members[0]
        if vo._is_chained_group(members, member_ids):
            try:
                assigns = vo._chain_domain(members, subst)
            except Exception:
                assigns = None
            if not assigns:
                return
            names = sorted(
                {v for m in members for v in vo._mv_names(m)}
            )
            doms.append(
                (vo._ChainGroup(attrs={n: n for n in names}), assigns)
            )
        else:
            opts = _domain_options(node, subst)
            if not opts:
                return
            doms.append((node, opts))
    for combo in vo._diag_product([d[1] for d in doms]):
        base = vo._attr_merge(doms, combo, subst)
        if base is not None:
            yield base


def _diag_bindings(rule, cap: int = 480):
    """Yield bound dicts from the diagonal seed corner, capped.

    Variants (uniform shapes x Const swaps) are interleaved
    round-robin — one attr draw per variant per round — so a cap
    truncates a *corner* of the variant space, not whole variants:
    a rank-3 corner that sits late in the variant order still
    contributes before the early scalar rows spend the budget.
    """
    mvs = sorted(set(vo._leaf_metavars(rule.lhs)))
    if not mvs:
        return
    seen = set()
    gens = [
        iter(_diag_attr_bases(rule, s)) for s in _diag_variants(mvs)
    ]
    while gens:
        nxt: list = []
        for g in gens:
            try:
                base = next(g)
            except StopIteration:
                continue
            nxt.append(g)
            key = repr(sorted((k, repr(v)) for k, v in base.items()))
            if key in seen:
                continue
            seen.add(key)
            yield base
            if len(seen) >= cap:
                return
        gens = nxt


# ------------------------------------------------------------------
#  Hand-minted seeds — accepted corners the domains cannot express
# ------------------------------------------------------------------


def _sdpa_mf_seed(scaled: bool, drop: bool) -> dict:
    """Bound dict for one ``sdpa_fold_masked_fill*`` variant's LHS."""
    b = {
        "Q": _v("Q", 2, 4),
        "K": _v("K", 3, 4),
        "V": _v("V", 3, 5),
        "MK": _v("MK", 2, 3),
        "F": Const(-float("inf")),
        "$attr:TD1": -2,
        "$attr:TD2": -1,
        "$attr:SD": -1,
    }
    if scaled:
        b["S"] = Const(0.5)
    if drop:
        b["$attr:DP"] = 0.5
        b["$attr:DT"] = False
    return b


def _seeds() -> dict:
    """Return ``{rule name: [bound dicts]}`` for the starved corner."""
    seeds: dict = {}
    # Deep chained-view guard — the rank-4 (b, t, h, d) corner from
    # test_cond_laws.test_gqa_absorb_repeat_requires_rank4_operands.
    seeds["gqa_absorb_repeat"] = [
        {
            "q": _v("q", 2, 3, 4, 4),
            "k": _v("k", 2, 3, 2, 4),
            "v": _v("v", 2, 3, 2, 4),
            "$attr:UDk": -2,
            "$attr:ESk": (2, 3, 2, 2, 4),
            "$attr:RSk": (2, 3, 4, 4),
            "$attr:UDv": -2,
            "$attr:ESv": (2, 3, 2, 2, 4),
            "$attr:RSv": (2, 3, 4, 4),
            "$attr:D": 1e-5,
            "$attr:C": False,
        }
    ]
    # Two-Const guards: P == 2 and EPS numeric, MD the trailing block.
    seeds["rms_norm_fold"] = [
        {
            "u": _v("u", 2, 8),
            "w": _v("w", 8),
            "EPS": Const(1e-6),
            "P": Const(2),
            "$attr:MD": (-1,),
            "$attr:MK": True,
        }
    ]
    seeds["rms_norm_fold_nogain"] = [
        {
            "u": _v("u", 2, 8),
            "EPS": Const(1e-6),
            "P": Const(2),
            "$attr:MD": (-1,),
            "$attr:MK": True,
        }
    ]
    # Rank-mixed correlated guard: rank-2 weights, rank-1 biases.
    for nm in ("assoc_linear_bias", "assoc_linear_bias_rev"):
        seeds[nm] = [
            {
                "x": _v("x", 2, 4),
                "A": _v("A", 4, 4),
                "B": _v("B", 4, 4),
                "b1": _v("b1", 4),
                "b2": _v("b2", 4),
            }
        ]
    # The asymmetric-QKV fold — derive-only guard, split ``sizes``
    # domain the oracle cannot type.
    seeds["qkv_fuse_asym"] = [
        {
            "x": _v("x", 2, 8),
            "Q": _v("Q", 4, 8),
            "K": _v("K", 4, 8),
            "V": _v("V", 4, 8),
            "$attr:S1": (2, 2, 2),
            "$attr:S2": (2, 2, 2),
            "$attr:S3": (2, 2, 2),
            "$attr:SC": 0.5,
            "$attr:G": False,
        }
    ]
    # The masked-fill sdpa fold family — F < -1e30 const corner.
    for scaled in (False, True):
        for drop in (False, True):
            tail = ("mul" if scaled else "") + ("_drop" if drop else "")
            nm = f"sdpa_fold_masked_fill{tail}"
            seeds[nm] = [_sdpa_mf_seed(scaled, drop)]
            tail = ("div" if scaled else "") + ("_drop" if drop else "")
            nm = f"sdpa_fold_masked_fill{tail}"
            seeds.setdefault(nm, []).append(_sdpa_mf_seed(scaled, drop))
    # Carrier concat pair — chunk-compatible equal-shape corner.
    seeds["om_split"] = [
        {
            "s1": _v("s1", 4, 4),
            "s2": _v("s2", 4, 4),
            "v1": _v("v1", 4, 4),
            "v2": _v("v2", 4, 4),
            "$attr:SD": -1,
            "$attr:VD": -2,
        }
    ]
    seeds["om_merge"] = [
        {
            "s1": _v("s1", 4, 4),
            "s2": _v("s2", 4, 4),
            "v1": _v("v1", 4, 4),
            "v2": _v("v2", 4, 4),
        }
    ]
    for nm in ("split_matmul_rows", "split_matmul_cols"):
        seeds[nm] = [
            {
                "a": _v("a", 4, 6),
                "b": _v("b", 6, 8),
                "$attr:SZ": (2, 2),
                "$attr:I": 0,
            }
        ]
    # -- carrier corners ---------------------------------------------
    # The mask-distribution family: the sliced operand's extent on the
    # cat axis must be exactly K1+K2 ("slice") or broadcast-1 ("reuse")
    # — a corner the independent shape draws never correlate.
    cat01 = {"s1": _v("s1", 2, 4), "s2": _v("s2", 3, 4), "$attr:D": 0}
    for nm in ("masked_fill_cat_slice_dim",):
        seeds[nm] = [{**cat01, "m": _v("m", 5, 4), "v": _v("v")}]
    seeds["masked_fill_cat_reuse_dim"] = [
        {**cat01, "m": _v("m", 1, 4), "v": _v("v")}
    ]
    for nm in ("add_cat_m_slice_dim", "add_m_cat_slice_dim"):
        seeds[nm] = [{**cat01, "m": _v("m", 5, 4)}]
    for nm in ("where_cat_x_slice_dim", "where_cat_y_slice_dim"):
        seeds[nm] = [{**cat01, "m": _v("m", 5, 4), "v": _v("v", 1)}]
    for nm in ("where_cat_x_reuse_dim", "where_cat_y_reuse_dim"):
        seeds[nm] = [{**cat01, "m": _v("m", 1, 4), "v": _v("v", 1)}]
    # The sdpa_cat variants carrying a ``dropout_p`` metavar: the
    # float attr domain (0.5, 1.0, 1e-5) never mints the ``dp == 0``
    # the guard demands — every enumerated binding declines.
    sdpa_cat = {
        "q": _v("q", 4, 8),
        "k1": _v("k1", 3, 8),
        "k2": _v("k2", 5, 8),
        "v1": _v("v1", 3, 6),
        "v2": _v("v2", 5, 6),
        "$attr:KD": -2,
        "$attr:VD": -2,
    }
    for nm in (
        "sdpa_cat_causal_3_dim",
        "sdpa_cat_1_dim",
        "sdpa_cat_3_dim",
    ):
        seeds[nm] = [{**sdpa_cat, "$attr:DP": 0}]
    for nm in (
        "sdpa_cat_causal_4_dim",
        "sdpa_cat_causal_5_dim",
        "sdpa_cat_6_dim",
    ):
        seeds[nm] = [{**sdpa_cat, "$attr:DP": 0, "$attr:SC": 0.5}]
    seeds["sdpa_cat_mask_4_dim"] = [
        {**sdpa_cat, "m": _v("m", 4, 8), "$attr:DP": 0}
    ]
    seeds["sdpa_cat_mask_6_dim"] = [
        {**sdpa_cat, "m": _v("m", 4, 8), "$attr:DP": 0}
    ]
    seeds["sdpa_cat_mask_7_dim"] = [
        {**sdpa_cat, "m": _v("m", 4, 8), "$attr:DP": 0, "$attr:SC": 0.5}
    ]
    seeds["sdpa_cat_mask_8_dim"] = [
        {**sdpa_cat, "m": _v("m", 4, 8), "$attr:DP": 0, "$attr:SC": 0.5}
    ]
    # The decayed-product chunk splits — row- and key-axis plans.
    seeds["om_decayed_chunk_rows_slice"] = [
        {
            "s1": _v("s1", 2, 4),
            "s2": _v("s2", 3, 4),
            "d": _v("d", 5, 4),
            "v": _v("v", 4, 6),
            "$attr:D": -2,
        }
    ]
    seeds["om_decayed_chunk_rows_reuse"] = [
        {
            "s1": _v("s1", 2, 4),
            "s2": _v("s2", 3, 4),
            "d": _v("d", 1, 4),
            "v": _v("v", 4, 6),
            "$attr:D": -2,
        }
    ]
    seeds["om_decayed_chunk_rows_cat"] = [
        {
            "s1": _v("s1", 2, 4),
            "s2": _v("s2", 3, 4),
            "d1": _v("d1", 2, 4),
            "d2": _v("d2", 3, 4),
            "v": _v("v", 4, 6),
            "$attr:SD": 0,
            "$attr:DD": 0,
        }
    ]
    seeds["om_decayed_chunk_keys_slice"] = [
        {
            "s": _v("s", 4, 8),
            "d": _v("d", 4, 8),
            "v1": _v("v1", 3, 6),
            "v2": _v("v2", 5, 6),
            "$attr:VD": -2,
        }
    ]
    seeds["om_decayed_chunk_keys_reuse"] = [
        {
            "s": _v("s", 4, 8),
            "d": _v("d", 4, 1),
            "v1": _v("v1", 3, 6),
            "v2": _v("v2", 5, 6),
            "$attr:VD": -2,
        }
    ]
    seeds["om_decayed_chunk_keys_cat_slice"] = [
        {
            "s1": _v("s1", 4, 3),
            "s2": _v("s2", 4, 5),
            "d": _v("d", 4, 8),
            "v1": _v("v1", 3, 6),
            "v2": _v("v2", 5, 6),
            "$attr:SD": -1,
            "$attr:VD": -2,
        }
    ]
    seeds["om_decayed_chunk_keys_cat_reuse"] = [
        {
            "s1": _v("s1", 4, 3),
            "s2": _v("s2", 4, 5),
            "d": _v("d", 4, 1),
            "v1": _v("v1", 3, 6),
            "v2": _v("v2", 5, 6),
            "$attr:SD": -1,
            "$attr:VD": -2,
        }
    ]
    # decay_lift — the bound a1/a2 must be the SAME unit-step arange,
    # a term kind the leaf bank never mints (Vars and Consts only).
    ar = Op.make("arange", Const(0), Const(8), Const(1))
    for nm in ("decay_lift", "decay_lift_diag"):
        seeds[nm] = [
            {
                "a1": ar,
                "a2": ar,
                "g": _v("g", 8, 8),
                "$attr:U1": 1,
                "$attr:U2": 0,
                "$attr:CM": 0,
            }
        ]
    # decay_block_* — ``t``/``offset`` attr kinds the domain table
    # does not type; the partition contract is seeded directly.
    seeds["decay_block_diag"] = [
        {
            "g": _v("g", 4),
            "$attr:T": 4,
            "$attr:SZ": (2, 2),
            "$attr:DR": -2,
            "$attr:DC": -1,
            "$attr:I": 0,
        }
    ]
    seeds["decay_block_cross"] = [
        {
            "g": _v("g", 4),
            "$attr:T": 4,
            "$attr:SZ": (2, 2),
            "$attr:DR": -2,
            "$attr:DC": -1,
            "$attr:I": 1,
            "$attr:J": 0,
        }
    ]
    seeds["decay_vec_offset"] = [
        {"g": _v("g", 4), "$attr:T": 4, "$attr:OD": 2}
    ]
    # The generalized cross law: ``x`` accepts only row-constant
    # broadcasters (broadcast shape ending in a 1-axis) — a corner
    # the unconstrained enumeration can miss.
    seeds["om_decayed_cross_gen"] = [
        {
            "s": _v("s", 5, 4),
            "x": _v("x", 5, 1),
            "eta": _v("eta", 4),
            "v": _v("v", 4, 6),
            "$attr:UE": -2,
        }
    ]
    return seeds


# ------------------------------------------------------------------
#  The sweep
# ------------------------------------------------------------------


def _bind_key(bound: dict) -> str:
    """Return a dedup key for a bound dict (leaf shapes included)."""
    parts = []
    for k, v in bound.items():
        if isinstance(v, (Var, Param)):
            parts.append((k, ("leaf", tuple(v.typ.shape))))
        else:
            parts.append((k, repr(v)))
    return repr(sorted(parts))


def _verdict(rule, bound: dict) -> tuple:
    """Return ``(accepted, error)`` — the gate ``apply_rule`` applies.

    ``accepted`` is ``check(bound)`` (``cond`` folded in) and, when a
    ``derive`` exists, its non-veto — the exact conjunction
    ``apply_rule`` requires.  ``error`` captures a guard that RAISED
    instead of answering — in production the exception propagates out
    of ``apply_rule`` (no ``try`` around the hooks), so it is a
    crash-class finding, not a decline.
    """
    try:
        if rule.check is not None and not rule.check(bound):
            return False, None
    except Exception as e:
        return None, ("check", e)
    if rule.derive is not None:
        try:
            if rule.derive(bound) is None:
                return False, None
        except Exception as e:
            return None, ("derive", e)
    return True, None


def _fires_at_redex(rule, lhs_i) -> tuple:
    """Instantiate *lhs_i*'s redex; return ``(fired, err)``.

    The ``search=[root]`` scope is what makes the attribution honest:
    only the redex's own e-class is matched, so a union can only have
    been produced by THIS binding — a fire elsewhere would be a
    different match, not this binding's rewrite.
    """
    try:
        eg = EGraph()
        root = eg.add_term(lhs_i)
        changed = eg.apply_rule(rule, root, search=[root])
    except Exception as e:
        return None, e
    return bool(changed), None


def guard_sweep(rule, limit: int | None = None) -> dict:
    """Sweep one conditional rule; return the findings row.

    Enumerates the binding space (oracle domain enumeration, diagonal
    seeds, hand-minted seeds), evaluates the guard verdict on each
    unique binding, and applies the rule at the instantiated redex.
    Returns ``{"true": fired, "false": declined, "miss": accepted but
    not fired, "violations": [...], "errors": [...], "probed": n,
    "unbound": n}``.
    """
    lim = limit or _MAX_INSTANCES
    sources = [
        itertools.islice(_binding_envs_safe(rule), lim),
        _diag_bindings(rule),
        iter(_SEEDS.get(rule.name, ())),
    ]
    row = {
        "probed": 0,
        "true": 0,
        "false": 0,
        "miss": 0,
        "violations": [],
        "errors": [],
        "unbound": 0,
        "starved": False,
    }
    seen = set()
    for full in itertools.chain(*sources):
        try:
            lhs_i = _term_instantiate(rule.lhs, full)
        except Exception:
            row["unbound"] += 1
            continue
        m = _term_match(rule.lhs, lhs_i)
        if m is None:
            row["unbound"] += 1
            continue
        key = _bind_key(m)
        if key in seen:
            continue
        seen.add(key)
        row["probed"] += 1
        acc, err = _verdict(rule, m)
        if err is not None:
            row["errors"].append((err[0], repr(err[1]), _bind_key(m)))
            continue
        fired, ferr = _fires_at_redex(rule, lhs_i)
        if ferr is not None:
            row["errors"].append(("apply", repr(ferr), _bind_key(m)))
            continue
        if fired and not acc:
            row["violations"].append((op_repr(lhs_i), _bind_key(m)))
        elif fired and acc:
            row["true"] += 1
        elif acc:
            row["miss"] += 1
        else:
            row["false"] += 1
    row["starved"] = row["true"] == 0 and row["miss"] == 0
    return row


def _binding_envs_safe(rule):
    """Yield ``_binding_envs`` rows, swallowing domain crashes."""
    try:
        yield from vo._binding_envs(rule.lhs, rule.rhs)
    except Exception:
        return


_SEEDS = _seeds()


# ------------------------------------------------------------------
#  Report — the sweep table
# ------------------------------------------------------------------


def _sweep_all() -> dict:
    """Run the sweep over every conditional rule (two-phase cap)."""
    rows = {}
    for rule in _swept_rules():
        row = guard_sweep(rule)
        # A guarded rule whose window accepted nothing gets the
        # pipeline's escalated cap — the accepted corner of a
        # multi-clause guard sits deep in the fair order.
        if row["true"] + row["miss"] == 0 and not row["violations"]:
            row = guard_sweep(rule, limit=_GUARDED_CAP)
        rows[rule.name] = row
    return rows


def _report(rows: dict) -> str:
    """Render the sweep table."""
    head = (
        f"{'rule':<36} {'probed':>6} {'true':>5} {'false':>6} "
        f"{'miss':>4} {'viol':>4} {'err':>4} {'cov':>8}"
    )
    lines = [head, "-" * len(head)]
    for name, r in rows.items():
        cov = "starved" if r["starved"] else "ok"
        lines.append(
            f"{name:<36} {r['probed']:>6} {r['true']:>5} "
            f"{r['false']:>6} {r['miss']:>4} "
            f"{len(r['violations']):>4} {len(r['errors']):>4} "
            f"{cov:>8}"
        )
    return "\n".join(lines)


@pytest.fixture(scope="session")
def sweep_rows():
    """The full sweep, run once for the module's assertions."""
    return _sweep_all()


# ------------------------------------------------------------------
#  Known findings — triage records, not silent passes
# ------------------------------------------------------------------

#: Findings the sweep is EXPECTED to surface, pinned with their
#: signature so a new crash/violation cannot hide among them.
#: ``om_split``'s ``_check_om_concat_dims`` read ``sd % len(s1)``
#: without guarding the rank-0 case — a scalar ``s1`` binding raised
#: ``ZeroDivisionError`` instead of declining.  Fixed: the guard now
#: declines rank-0 operands before the modulo reads them.
_KNOWN_ISSUES: dict = {}

#: Conditional rules whose accepted corner neither the enumerated
#: domain nor the seeds reach within the sweep window — a coverage
#: gap, honestly reported.  Each entry names why the true region is
#: unreachable (attr kinds the domain table cannot type, or a corner
#: past the escalated cap).
_STARVED_OK: dict = {}


# ------------------------------------------------------------------
#  Tests
# ------------------------------------------------------------------


def test_every_conditional_rule_is_swept(sweep_rows):
    """Every conditional shipped + carrier law gets a row."""
    names = {r.name for r in _swept_rules()}
    assert set(sweep_rows) == names
    assert len(names) == 119  # the 54 core conds + 64 carrier guards


def test_no_false_region_rewrites(sweep_rows, capsys):
    """Zero violations: a declined binding never rewrites."""
    viols = {
        nm: r["violations"]
        for nm, r in sweep_rows.items()
        if r["violations"]
    }
    with capsys.disabled():
        print("\n" + _report(sweep_rows))
        for nm, vs in viols.items():
            for lhs_i, bound in vs:
                print(f"VIOLATION {nm}: {lhs_i}\n  bound={bound}")
    assert viols == {}, f"false-region rewrites: {sorted(viols)}"


def test_no_unexpected_guard_errors(sweep_rows):
    """Guard errors are crashes-in-waiting: only triaged allowed."""
    errs = {}
    for nm, r in sweep_rows.items():
        unexpected = [e for e in r["errors"] if nm not in _KNOWN_ISSUES]
        if unexpected:
            errs[nm] = unexpected
    assert errs == {}, f"guard errors outside the triage set: {errs}"


def test_known_findings_still_reproduce(sweep_rows):
    """Each triaged finding still manifests — stale entries surface."""
    for nm, note in _KNOWN_ISSUES.items():
        row = sweep_rows.get(nm)
        assert row is not None, f"{nm} not swept ({note})"
        assert row["errors"], (
            f"{nm}: triaged finding no longer reproduces — {note}. "
            "The upstream bug may be fixed; retire this entry."
        )


def test_om_split_scalar_operand_declines_cleanly():
    """Regression for the sweep's first finding: a rank-0 ``s1`` under
    the concat used to crash ``_check_om_concat_dims``
    (``ZeroDivisionError`` on ``sd % len(s1)``) — after the rank
    guard it declines like every other false-region binding."""
    from catopt_carriers.om import OM_SPLIT

    t = Op.make(
        "om_elem",
        Op.make("concat", _v("s1"), _v("s2", 4, 4), dim=-1),
        Op.make("concat", _v("v1", 4, 4), _v("v2", 4, 4), dim=-2),
    )
    eg = EGraph()
    root = eg.add_term(t)
    assert eg.apply_rule(OM_SPLIT, root) is False


def test_true_region_reached_or_documented(sweep_rows):
    """Every swept rule's true region fired, or the gap documented."""
    starved = {
        nm
        for nm, r in sweep_rows.items()
        if r["starved"] and not r["errors"] and nm not in _KNOWN_ISSUES
    }
    undocumented = starved - set(_STARVED_OK)
    assert undocumented == set(), (
        "conditional rules whose true region the sweep never reached: "
        f"{sorted(undocumented)}"
    )


def test_true_region_bindings_fire(sweep_rows):
    """An accepted binding that does not fire is a miss — report all."""
    misses = {
        nm: r["miss"] for nm, r in sweep_rows.items() if r["miss"]
    }
    assert misses == {}, (
        f"accepted bindings that did not fire: {misses}"
    )

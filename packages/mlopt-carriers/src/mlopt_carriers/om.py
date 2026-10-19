# ruff: noqa: RUF002, RUF003
"""Online-softmax monoid — the nonlinear analogue of the scan carrier.

The affine-map monoid (``SCAN_LAWS`` in :mod:`mlopt_core.laws`) showed that
lifting a recurrence into a monoid carrier lets plain associativity
discover parallel bracketings.  The same trick works for the softmax
recurrence — FlashAttention's running (max, sum, numerator) triple.

CARRIER
    ``om(m, l, a)`` is a triple value: running row-max ``m``, running
    denominator ``l = Σ exp(s − m)``, running numerator
    ``a = Σ exp(s − m) · v``.

COMPOSITION (associative *and* commutative)
    ``(m1,l1,a1) ⊕ (m2,l2,a2)``
        = (mx, l1·e^(m1−mx) + l2·e^(m2−mx), a1·e^(m1−mx) + a2·e^(m2−mx))
    with mx = max(m1, m2).  We deliberately do NOT add a commutativity
    rule — comm is the explosive law (permutation space), exactly as in
    SCAN_LAWS; OM_ASSOC/OM_ASSOC_REV suffice for reassociation.

ELEMENT
    ``om_elem(s, v)`` for score block ``s (…,T,K)`` and value block
    ``v (…,K,d)``: ``(rowmax(s), rowsum(exp(s−rowmax)), exp(s−rowmax)@v)``.

HOMOMORPHISM
    ``elem(cat(s1,s2), cat(v1,v2)) = elem(s1,v1) ⊕ elem(s2,v2)`` —
    and ``softmax(s) @ v = a / l``.  This is the law tensor algebra
    cannot state: softmax does not distribute over concat, but its
    monoid carrier does.

``om_apply(f)`` evaluates a carrier back to tensor-land: ``a / l``
(unclamped — a fully-masked row yields NaN exactly like dense softmax).

The rules:
* ``OM_LIFT`` / ``OM_UNLIFT``   — enter / leave the carrier.
* ``OM_SPLIT`` / ``OM_MERGE``   — the homomorphism, both directions.
* ``OM_ASSOC`` / ``OM_ASSOC_REV`` — reassociating the combine gives the
  chunked/blocked attention schedules (FlashAttention tiling is a
  bracketing of this monoid product).
* ``CONCAT_BINARIZE_*``         — n-ary concat → nested binary (the IR's
  concat is variadic; the homomorphism is binary).
* ``MATMUL_T_CONCAT``           — ``q @ cat(k1,k2).T = cat(q@k1.T, q@k2.T)``:
  matmul distributes over concat in the transposed operand, exposing
  per-block score tensors for OM_SPLIT.

Note on concat attrs: the export boundary canonicalises ``cat``'s axis
to ``dim`` (``ATTR_SCHEMA``), and rule-produced/hand-built terms use
``dim`` too — every LHS pattern and RHS here speaks the one canonical
spelling.

MASKED ATTENTION
    A causal/additive mask wraps the score concat in ``masked_fill`` /
    ``add`` / ``where`` and blocks OM_SPLIT.  The law that unblocks it
    is that masking distributes over concat (see the section below):
    each block takes the matching *slice* of the mask, and for causal
    masks the block's positional offset lives inside that slice.

DECAYED ATTENTION
    RetNet-style chunked retention writes ``(QKᵀ ⊙ D) V`` — a score
    matrix weighted elementwise by a position/decay matrix, then
    contracted by the values.  There is no softmax and no normaliser,
    so the om element cannot express it (``mul`` is not ``softmax`` —
    OM_LIFT declines the spelled term, and lifting it onto any
    existing carrier verifies FALSE).  ``om_decayed(s, d, v)`` is the
    vocabulary element naming it: see the section before the law set.

    The decay matrix itself is a value too — ``decay(γ, t=T)`` makes
    ``tril(γ^{i−j})`` a generator the block laws can see: the
    ``om_decayed_chunk_*`` splits decompose a decayed product over
    concat'd operands along both matmul axes, ``decay_lift`` folds
    the spelled arange/pow/tril subgraph into the generator, and
    ``decay_block_diag`` regenerates diagonal blocks in place.  The
    remaining structure — the strict-upper zero block and the cross
    block's ξ·ηᵀ factorisation — is documented where the laws live.
"""

from typing import Any, Literal

import torch
from mlopt_core.egraph import Rewrite, _LeafRegistry
from mlopt_core.ir import Const, Op, op_def
from mlopt_core.laws import R as _R
from mlopt_core.laws import RuleSet
from mlopt_core.laws import tags as _tags
from mlopt_core.laws.cond import as_derive

# Registers ``_xshape`` as the value-shape resolver
# (``typing.register_value_shape``) — the derive DSL's
# ``("vshape", T)`` falls back to the cost convention without it, and
# this module ships a ``vshape``-reading ``dspec``
# (:data:`_DSPEC_SCORE_CONCAT_DIM`).  One-way edge: xcarrier imports
# core only.
import mlopt_carriers.xcarrier as _xc  # noqa: F401


def R(name: str, lhs, rhs, **kw) -> Rewrite:
    """Module-local law constructor — om laws carry the ``CARRIER`` tag."""
    kw.setdefault("tags", (_tags.CARRIER,))
    return _R(name, lhs, rhs, **kw)


#: The canonical concat-axis attr KEY (see the module note on concat
#: attrs).  Spelled as a ``Literal`` so that ``Op.make(..., **{key: v})``
#: proves the keyword can never collide with ``Op.make``'s ``validate``
#: flag — a plain ``str`` key would admit that ambiguity.
_DimKey = Literal["dim"]


def _vshape(t: Any):
    """Return the *value* shape of a bound term — carrier-aware.

    Metavariable bindings resolve through ``EGraph``'s representative
    member (``_min_term``/``any_term``), which may be a CARRIER member:
    ``applyd``/``apply``/the om family report a *convention* shape
    under ``mlopt_core.typing.shape_of`` (the state slot, the map's linear
    part — ``()`` when the slot resolves to a scalar member), not the
    tensor value the e-class denotes.  Judging side conditions on
    convention shapes vetoed legal rewrites — observed: ``om_lift``
    on multi-head attention at T=32, where the score class's min-size
    member is an ``applyd`` reporting ``()`` (and the value class's
    an ``apply`` reporting the state shape) while both denote the
    correct ``(nh,T,K)``/``(nh,T,d)`` tensors.

    ``mlopt_carriers.xcarrier._xshape`` computes the true value shape for the
    carrier-application ops (map terms keep the cost convention — no
    check here inspects them) and re-dispatches the cost model over
    children carrying the corrected shapes, so every check below sees
    the shape the term would have as a plain tensor.  Terms whose
    value shape is genuinely unresolvable still veto — the checks are
    load-bearing, only now on truthful shapes.
    """
    from mlopt_carriers.xcarrier import _xshape

    return _xshape(t)


def _dim_eq(a: Any, b: Any) -> bool:
    """cat-compatibility on a non-concatenated axis (None = unknown)."""
    return a is None or b is None or a == b


def _broadcast_ok(a, b) -> bool:
    from mlopt_core.typing import INVALID, broadcast

    return broadcast(a, b) is not INVALID


# ---------------------------------------------------------------------------
#  Lift / unlift — enter and leave the carrier
# ---------------------------------------------------------------------------


def _check_om_lift(bound: dict) -> bool:
    """Require softmax over the scores' LAST dim (the key axis).

    The matmul must contract s[...,K] with v[...,K,d].  A softmax over
    any other axis is well-typed but a *different* program — the check
    is load-bearing.
    """
    sd = bound.get("$attr:SD", -1)
    ss, vs = _vshape(bound.get("s")), _vshape(bound.get("v"))
    if not (
        isinstance(ss, tuple)
        and isinstance(vs, tuple)
        and len(ss) >= 2
        and len(vs) >= 2
    ):
        return False
    if not isinstance(sd, int) or sd % len(ss) != len(ss) - 1:
        return False
    return _dim_eq(ss[-1], vs[-2])


def _om_lift(name: str, attr_key: _DimKey | None) -> Rewrite:
    sm = (
        Op.make("softmax", "s", **{attr_key: "SD"})
        if attr_key is not None
        else Op.make("softmax", "s")
    )
    return R(
        name,
        Op.make("matmul", sm, "v"),
        Op.make("om_apply", Op.make("om_elem", "s", "v")),
        law="matmul(softmax(s), v) lifts into the online-softmax monoid: "
        "softmax does not distribute over concat, but elem does — "
        "the carrier (m, l, a) is what distributes.",
        check=_check_om_lift,
    )


#: softmax's canonical ``dim`` spelling.
OM_LIFT = _om_lift("om_lift", "dim")
OM_LIFT_PLAIN = _om_lift("om_lift_plain", None)

OM_UNLIFT = R(
    "om_unlift",
    Op.make("om_apply", Op.make("om_elem", "s", "v")),
    Op.make("matmul", Op.make("softmax", "s", dim=-1), "v"),
    law="om_apply(elem(s,v)) unfolds back to dense softmax(s) @ v.",
)


# ---------------------------------------------------------------------------
#  The homomorphism — elem distributes over concat
# ---------------------------------------------------------------------------


def _chunks_compatible(s1, s2, v1, v2) -> bool:
    """Require chunk pair i to contract s_i[...,K_i] with v_i[...,K_i,d].

    The two chunks must be cat-compatible off the concatenated axis, and
    each chunk's batch dims must broadcast s_i against v_i.
    """
    ns, nv = len(s1), len(v1)
    if ns < 2 or nv < 2 or len(s2) != ns or len(v2) != nv:
        return False
    if not all(_dim_eq(s1[i], s2[i]) for i in range(ns - 1)):
        return False
    if not all(_dim_eq(v1[i], v2[i]) for i in range(nv) if i != nv - 2):
        return False
    if not (
        _dim_eq(s1[-1], v1[nv - 2]) and _dim_eq(s2[-1], v2[nv - 2])
    ):
        return False
    return _broadcast_ok(s1[:-2], v1[:-2]) and _broadcast_ok(
        s2[:-2], v2[:-2]
    )


def _check_om_concat_dims(bound: dict) -> bool:
    """OM_SPLIT fires only when scores concat on the LAST dim (keys).

    Values must concat on dim -2 (the same key axis, pre-contraction).
    A cat along any other axis is well-typed but WRONG.
    """
    sd, vd = bound.get("$attr:SD"), bound.get("$attr:VD")
    s1, s2 = _vshape(bound.get("s1")), _vshape(bound.get("s2"))
    v1, v2 = _vshape(bound.get("v1")), _vshape(bound.get("v2"))
    if not all(isinstance(x, tuple) for x in (s1, s2, v1, v2)):
        return False
    if not (isinstance(sd, int) and isinstance(vd, int)):
        return False
    # rank-0 operands carry no concat axis; ``max(len, 1)`` keeps the
    # modulo well-defined so they decline here instead of raising.
    n1, n2 = len(s1), len(v1)
    if sd % max(n1, 1) != n1 - 1:
        return False  # scores must concat along the key axis
    if vd % max(n2, 1) != n2 - 2:
        return False  # values concat along the key axis (-2 before @)
    return _chunks_compatible(s1, s2, v1, v2)


def _om_split(name: str, attr_key: _DimKey) -> Rewrite:
    return R(
        name,
        Op.make(
            "om_elem",
            Op.make("concat", "s1", "s2", **{attr_key: "SD"}),
            Op.make("concat", "v1", "v2", **{attr_key: "VD"}),
        ),
        Op.make(
            "om_compose",
            Op.make("om_elem", "s1", "v1"),
            Op.make("om_elem", "s2", "v2"),
        ),
        law="Homomorphism: elem(cat(s1,s2), cat(v1,v2)) = "
        "elem(s1,v1) ⊕ elem(s2,v2) — the law tensor algebra cannot "
        "state (softmax does not distribute over concat).",
        check=_check_om_concat_dims,
    )


OM_SPLIT = _om_split("om_split", "dim")


def _derive_om_concat_dims(bound: dict) -> dict | None:
    """Compute concat dims for the merged element from bound shapes.

    Scores join on their last dim, values on dim -2.  Vetoes (returns
    None) when the two chunks cannot form a well-typed cat.

    Stays procedural: reads the carrier-aware ``_vshape`` — a bound
    metavar may resolve to a carrier member whose *cost-convention*
    shape is not the value shape — and the ``_chunks_compatible``
    veto is shape logic the dspec grammar cannot express.
    """
    s1, s2 = _vshape(bound.get("s1")), _vshape(bound.get("s2"))
    v1, v2 = _vshape(bound.get("v1")), _vshape(bound.get("v2"))
    if not all(isinstance(x, tuple) for x in (s1, s2, v1, v2)):
        return None
    if not _chunks_compatible(s1, s2, v1, v2):
        return None
    return {"$attr:SD": len(s1) - 1, "$attr:VD": len(v1) - 2}


OM_MERGE = R(
    "om_merge",
    Op.make(
        "om_compose",
        Op.make("om_elem", "s1", "v1"),
        Op.make("om_elem", "s2", "v2"),
    ),
    Op.make(
        "om_elem",
        Op.make("concat", "s1", "s2", dim="SD"),
        Op.make("concat", "v1", "v2", dim="VD"),
    ),
    law="Reverse homomorphism: two block elements merge into the "
    "element of the concatenated block — chunking is reversible.",
    derive=_derive_om_concat_dims,
)


# ---------------------------------------------------------------------------
#  Associativity — the monoid law that generates blocked schedules
# ---------------------------------------------------------------------------

OM_ASSOC = R(
    "om_assoc",
    Op.make("om_compose", Op.make("om_compose", "f", "g"), "h"),
    Op.make("om_compose", "f", Op.make("om_compose", "g", "h")),
    law="⊕ is associative: (f ⊕ g) ⊕ h = f ⊕ (g ⊕ h).  Every "
    "bracketing of the block product is a legal attention schedule.",
)

OM_ASSOC_REV = R(
    "om_assoc_rev",
    Op.make("om_compose", "f", Op.make("om_compose", "g", "h")),
    Op.make("om_compose", Op.make("om_compose", "f", "g"), "h"),
    law="⊕ is associative: f ⊕ (g ⊕ h) = (f ⊕ g) ⊕ h.",
)


# ---------------------------------------------------------------------------
#  concat is variadic in the IR; the homomorphism is binary.
# ---------------------------------------------------------------------------


def _concat_binarize(n: int, attr_key: _DimKey) -> Rewrite:
    xs = [f"x{i}" for i in range(n)]
    rhs = Op.make("concat", xs[0], xs[1], dim="D")
    for x in xs[2:]:
        rhs = Op.make("concat", rhs, x, dim="D")
    return R(
        f"concat_binarize_{n}_{attr_key}",
        Op.make("concat", *xs, **{attr_key: "D"}),
        rhs,
        law="n-ary concat is iterated binary concat — cat is a binary "
        "tensor product applied repeatedly.",
    )


#: 3..6-ary concats, the canonical ``dim`` spelling.  Structural only —
#: needs no shape check (binarisation preserves well-typedness).
CONCAT_BINARIZE: list[Rewrite] = [
    _concat_binarize(n, "dim") for n in range(3, 7)
]


# ---------------------------------------------------------------------------
#  matmul distributes over concat in the transposed operand
# ---------------------------------------------------------------------------


def _check_matmul_t_concat(bound: dict) -> bool:
    """Q @ cat(k1,k2,dim).T splits only on a SEQUENCE-axis key concat.

    The key concat must be on dim -2 of k (keys) and the transpose
    exactly .T on the last two dims — so that after the transpose the
    concat lands on the scores' last dim.
    """
    kd, t1, t2 = (
        bound.get("$attr:KD"),
        bound.get("$attr:T1"),
        bound.get("$attr:T2"),
    )
    k1, k2 = _vshape(bound.get("k1")), _vshape(bound.get("k2"))
    q = _vshape(bound.get("q"))
    if not (
        isinstance(kd, int)
        and isinstance(t1, int)
        and isinstance(t2, int)
    ):
        return False
    if not all(isinstance(s, tuple) for s in (k1, k2, q)):
        return False
    nk = len(k1)
    if nk < 2 or len(k2) != nk or len(q) < 2:
        return False
    if kd % nk != nk - 2:
        return False  # keys concat on the sequence axis, not features
    if {t1 % nk, t2 % nk} != {nk - 2, nk - 1}:
        return False  # transpose must be exactly .T (last two dims)
    if not all(_dim_eq(k1[i], k2[i]) for i in range(nk) if i != nk - 2):
        return False
    # q[...,d] contracts with k.T[...,d,K] — k's LAST dim is d.
    if not _dim_eq(q[-1], k1[-1]):
        return False
    return _broadcast_ok(q[:-2], k1[:-2])


def _derive_score_concat_dim(bound: dict) -> dict | None:
    """Return the concat dim through the transpose.

    After .T the key axis is last in k.T and lands on the scores' last
    dim, at index rank(out)-1 = max(rank q, rank k) - 1.

    The procedural oracle for :data:`_DSPEC_SCORE_CONCAT_DIM` — the
    synthesized spec replaces the hook, and this twin stays as the
    test's oracle (kept here so the migration is auditable, not
    because the rule calls it).  Reads ``_vshape`` (carrier-aware); the
    spec must read ``("vshape", T)`` — the same value-shape source —
    not the cost-convention ``("shape", T)``: on a bound carrier
    member the two disagree (a deeper measurement domain over op-term
    bindings caught the original ``shape`` spelling minting the
    convention rank, e.g. SD=1 where the oracle's value rank gives 0).
    """
    q, k1 = _vshape(bound.get("q")), _vshape(bound.get("k1"))
    if not (isinstance(q, tuple) and isinstance(k1, tuple)):
        return None
    return {"$attr:SD": max(len(q), len(k1)) - 1}


#: ``SD`` := ``max(rank q, rank k1) - 1`` — the declarative twin of
#: :func:`_derive_score_concat_dim`, re-audited over the carrier
#: op-term domain: the rank readers go through ``("vshape", T)`` — the
#: derive DSL's value-shape source, resolved by the registered
#: ``_xshape`` — matching the hook's carrier-aware read on compound
#: members as well as leaves.  A rank reader that declines on an
#: unshaped operand vetoes the spec exactly as the hook's
#: ``isinstance`` guard did.
_DSPEC_SCORE_CONCAT_DIM = {
    "SD": (
        "sub",
        ("max", ("len", ("vshape", "q")), ("len", ("vshape", "k1"))),
        1,
    ),
}


def _matmul_t_concat(name: str, attr_key: _DimKey) -> Rewrite:
    return R(
        name,
        Op.make(
            "matmul",
            "q",
            Op.make(
                "transpose",
                Op.make("concat", "k1", "k2", **{attr_key: "KD"}),
                dim0="T1",
                dim1="T2",
            ),
        ),
        Op.make(
            "concat",
            Op.make(
                "matmul",
                "q",
                Op.make("transpose", "k1", dim0="T1", dim1="T2"),
            ),
            Op.make(
                "matmul",
                "q",
                Op.make("transpose", "k2", dim0="T1", dim1="T2"),
            ),
            dim="SD",
        ),
        law="matmul distributes over concat in the transposed operand: "
        "q @ cat(k1,k2).T = cat(q@k1.T, q@k2.T) — the score blocks "
        "surface as one concat so OM_SPLIT can chunk the softmax.",
        check=_check_matmul_t_concat,
        dspec=_DSPEC_SCORE_CONCAT_DIM,
    )


MATMUL_T_CONCAT = _matmul_t_concat("matmul_t_concat", "dim")


# ---------------------------------------------------------------------------
#  Masks distribute over concat — the law masked chunked attention needs.
#
#  OM_SPLIT needs the score operand to be a literal concat; a causal or
#  additive mask wraps that concat in ``masked_fill``/``add``/``where``
#  and the homomorphism cannot see through it.  The law that unblocks
#  it: masking COMMUTES with concatenation,
#
#      mask(cat(s1, s2), M) = cat(mask(s1, M1), mask(s2, M2))
#
#  where M_i is block i's slice of the full mask along the cat axis —
#  ``split(M, (K1, K2), d, i)`` (the IR's existing projection op; no new
#  generator is needed).  For a causal mask this slice IS the
#  positional dependency: block i's mask columns are the global mask's
#  columns [o_i, o_i + K_i), so the block offset lives inside the
#  slice — no index arithmetic appears in the rewrite.  When the mask
#  broadcasts along the cat axis (extent 1, or the axis absent from the
#  mask entirely) the SAME operand serves both blocks unsliced.
#
#  What this IR still cannot express (documented, not hacked):
#  * synthesising a block-local mask from positions — there are no
#    arange/iota/tril/ones generators, so the offset law
#    ``mask_i = lt(arange(K_i) + o_i, arange(T))`` cannot be written;
#    only slicing a MATERIALISED mask distributes;
#  * implicit masks — ``sdpa(..., is_causal=True)`` has no mask operand
#    to slice; the flag would need its own chunking law;
#  * masks that are not axis-aligned slices of the cat'd operand (e.g.
#    interleaved/block-diagonal layouts) — which don't arise from key
#    chunking anyway.
# ---------------------------------------------------------------------------


def _cat_axis_plan(
    bound: dict, sliced_key: str, fixed_keys: tuple = ()
) -> dict | None:
    """Plan operand distribution over ``concat(s1, s2, dim=D)``.

    The sliceable operand (``sliced_key`` — the mask/bias) is classified
    by its extent on the cat axis:

    * extent ``K1 + K2`` (the whole concatenated axis) → ``"slice"``:
      block i gets ``split(m, (K1, K2), md, i)`` — for causal masks the
      column slice is exactly the absolute-position offset;
    * extent 1, or the axis absent from the operand (lower rank /
      scalar) → ``"reuse"``: the operand broadcasts, unchanged, into
      both blocks;
    * anything else, or unknown extents → ``None`` (veto: the rule may
      not fire).

    ``fixed_keys`` are other operands that must broadcast along the cat
    axis to be reused (e.g. ``masked_fill``'s fill value, ``where``'s
    other branch): extent 1 — or exactly the block size when both
    blocks are equal — is fine; covering the whole axis would need its
    own slice, which these rules do not build → veto.

    Returns ``{"mode", "sizes", "md", "do"}``: ``md`` is the cat axis in
    the sliced operand's OWN coordinates (mask rank may differ from the
    score rank under broadcasting), ``do`` the cat axis of the
    broadcast result (output rank = max operand rank — a bigger-rank
    mask broadcasts the output up).
    """
    s1, s2 = _vshape(bound.get("s1")), _vshape(bound.get("s2"))
    D = bound.get("$attr:D")
    if not (
        isinstance(s1, tuple)
        and isinstance(s2, tuple)
        and s1
        and len(s1) == len(s2)
    ):
        return None
    if not isinstance(D, int):
        return None
    r, d = len(s1), D % len(s1)
    if not all(_dim_eq(s1[i], s2[i]) for i in range(r) if i != d):
        return None  # ill-typed cat
    k1, k2 = s1[d], s2[d]
    if not (isinstance(k1, int) and isinstance(k2, int)):
        return None  # unknown block extents

    shapes: dict[str, tuple] = {}
    out_rank = r
    for key in (sliced_key, *fixed_keys):
        sh = _vshape(bound.get(key))
        if not isinstance(sh, tuple):
            return None  # unknown shape: can't prove
        shapes[key] = sh
        out_rank = max(out_rank, len(sh))

    def _off_axis_ok(sh) -> bool:
        """Require every off-axis operand dim to broadcast against blocks.

        Extra leading dims are fine — they rank up the output
        identically on both sides.
        """
        for j, ext in enumerate(sh):
            dj = j + r - len(sh)  # operand dim → score dim
            if dj < 0 or dj == d:
                continue
            if not (ext is None or ext == 1 or _dim_eq(ext, s1[dj])):
                return False
        return True

    for key in fixed_keys:
        fs = shapes[key]
        if not _off_axis_ok(fs):
            return None
        fd = d + len(fs) - r  # cat axis in f's coords
        if fd >= 0:
            ext = fs[fd]
            if not (ext == 1 or (k1 == k2 and ext == k1)):
                return None  # needs its own slice

    ms = shapes[sliced_key]
    if not _off_axis_ok(ms):
        return None
    md = d + len(ms) - r  # cat axis in mask coords
    do = d + out_rank - r  # cat axis of the result
    if md < 0:
        return {"mode": "reuse", "sizes": None, "md": None, "do": do}
    ext = ms[md]
    if ext is None:
        return None
    if ext == k1 + k2:
        return {"mode": "slice", "sizes": (k1, k2), "md": md, "do": do}
    if ext == 1 or (k1 == k2 and ext == k1):
        return {"mode": "reuse", "sizes": None, "md": None, "do": do}
    return None


def _check_mask_cat(sliced_key: str, fixed_keys: tuple, mode: str):
    def check(bound: dict) -> bool:
        plan = _cat_axis_plan(bound, sliced_key, fixed_keys)
        return plan is not None and plan["mode"] == mode

    return check


def _derive_mask_cat(sliced_key: str, fixed_keys: tuple):
    """Compute the mask plan's attrs — stays procedural.

    Mints mode-dependent attrs from ``_cat_axis_plan`` — a
    carrier-aware shape analysis the dspec grammar cannot express.
    """

    def derive(bound: dict) -> dict | None:
        plan = _cat_axis_plan(bound, sliced_key, fixed_keys)
        if plan is None:
            return None
        out = {"$attr:DO": plan["do"]}
        if plan["mode"] == "slice":
            out["$attr:SZ"] = plan["sizes"]
            out["$attr:MD"] = plan["md"]
        return out

    return derive


def _mask_slice(key: str, i: int) -> Op:
    """Build block i's slice of a mask/bias along the cat axis.

    A projection of the full mask, which is exactly where the positional
    offset of block i lives (causal masks included).
    """
    return Op.make("split", key, sizes="SZ", dim="MD", index=i)


def _masked_fill_cat(
    name: str, attr_key: _DimKey, mode: str
) -> Rewrite:
    """masked_fill(cat(s1,s2,D), m, v) → cat(masked_fill(s_i, m_i, v), D).

    ``m_i`` is the mask's slice on the cat axis (mode "slice") or the
    mask itself when it broadcasts along that axis (mode "reuse").
    """
    m1 = _mask_slice("m", 0) if mode == "slice" else "m"
    m2 = _mask_slice("m", 1) if mode == "slice" else "m"
    return R(
        name,
        Op.make(
            "masked_fill",
            Op.make("concat", "s1", "s2", **{attr_key: "D"}),
            "m",
            "v",
        ),
        Op.make(
            "concat",
            Op.make("masked_fill", "s1", m1, "v"),
            Op.make("masked_fill", "s2", m2, "v"),
            dim="DO",
        ),
        check=_check_mask_cat("m", ("v",), mode),
        derive=_derive_mask_cat("m", ("v",)),
        law="masked_fill distributes over concat: masking a "
        "concatenated score matrix equals concatenating the masked "
        "blocks, each with its slice of the mask — a causal mask's "
        "block offset lives inside the slice.",
    )


def _add_cat(
    name: str, attr_key: _DimKey, mode: str, mask_first: bool
) -> Rewrite:
    """add(cat(s1,s2,D), m) / add(m, cat(s1,s2,D)) → cat of per-block adds.

    The additive-mask counterpart of masked_fill_cat.  add is
    commutative, but COMM_ADD is deliberately absent from OM_LAWS, so
    both operand orders get a rule.
    """
    m1 = _mask_slice("m", 0) if mode == "slice" else "m"
    m2 = _mask_slice("m", 1) if mode == "slice" else "m"
    cat = Op.make("concat", "s1", "s2", **{attr_key: "D"})
    lhs = (
        Op.make("add", "m", cat)
        if mask_first
        else Op.make("add", cat, "m")
    )
    b1 = (
        Op.make("add", m1, "s1")
        if mask_first
        else Op.make("add", "s1", m1)
    )
    b2 = (
        Op.make("add", m2, "s2")
        if mask_first
        else Op.make("add", "s2", m2)
    )
    return R(
        name,
        lhs,
        Op.make("concat", b1, b2, dim="DO"),
        check=_check_mask_cat("m", (), mode),
        derive=_derive_mask_cat("m", ()),
        law="An additive mask distributes over concat: add(cat s, m) = "
        "cat(add(s_i, m_i)) — concat is a homomorphism for "
        "elementwise ops, with the mask sliced on the cat axis.",
    )


def _where_cat(
    name: str, attr_key: _DimKey, mode: str, cat_in_x: bool
) -> Rewrite:
    """where(m, cat(s1,s2,D), v) / where(m, v, cat(s1,s2,D)).

    Splits to cat of per-block wheres — the torch.where masking idiom.
    """
    m1 = _mask_slice("m", 0) if mode == "slice" else "m"
    m2 = _mask_slice("m", 1) if mode == "slice" else "m"
    cat = Op.make("concat", "s1", "s2", **{attr_key: "D"})
    if cat_in_x:
        lhs = Op.make("where", "m", cat, "v")
        b1 = Op.make("where", m1, "s1", "v")
        b2 = Op.make("where", m2, "s2", "v")
    else:
        lhs = Op.make("where", "m", "v", cat)
        b1 = Op.make("where", m1, "v", "s1")
        b2 = Op.make("where", m2, "v", "s2")
    return R(
        name,
        lhs,
        Op.make("concat", b1, b2, dim="DO"),
        check=_check_mask_cat("m", ("v",), mode),
        derive=_derive_mask_cat("m", ("v",)),
        law="where distributes over concat in the masked operand: "
        "where(m, cat s, v) = cat(where(m_i, s_i, v)).",
    )


def _check_cat_pair(bound: dict) -> int | None:
    """For ``op(cat(a1,a2,DA), cat(b1,b2,DB)) → cat(op(a1,b1), op(a2,b2))``.

    Both cat axes must be the SAME axis of the broadcast result (ranks
    may differ — a lower-rank mask's cat axis is shifted), each operand
    pair must be cat-compatible on its own axis, and the per-block
    broadcast results must be cat-compatible on the shared axis.
    Returns the result cat dim, or None to veto.
    """
    a1, a2 = _vshape(bound.get("a1")), _vshape(bound.get("a2"))
    b1, b2 = _vshape(bound.get("b1")), _vshape(bound.get("b2"))
    DA, DB = bound.get("$attr:DA"), bound.get("$attr:DB")
    if not all(isinstance(s, tuple) and s for s in (a1, a2, b1, b2)):
        return None
    if not (isinstance(DA, int) and isinstance(DB, int)):
        return None
    ra, rb = len(a1), len(b1)
    if len(a2) != ra or len(b2) != rb:
        return None
    da, db = DA % ra, DB % rb
    ro = max(ra, rb)
    oa, ob = da + ro - ra, db + ro - rb
    if oa != ob:
        return None  # different axes — wrong
    if not all(_dim_eq(a1[i], a2[i]) for i in range(ra) if i != da):
        return None
    if not all(_dim_eq(b1[i], b2[i]) for i in range(rb) if i != db):
        return None
    from mlopt_core.typing import INVALID, broadcast

    ba, bb = broadcast(a1, b1), broadcast(a2, b2)
    if ba is INVALID or bb is INVALID:
        return None
    if not all(
        _dim_eq(ba[i], bb[i]) for i in range(ro) if i != oa
    ):  # pragma: no cover — off-axis equality forced by earlier guards
        return None
    return oa


def _cat_pair_check(bound: dict) -> bool:
    return _check_cat_pair(bound) is not None


def _cat_pair_derive(bound: dict) -> dict | None:
    """Derive ``DO`` — stays procedural.

    ``DO`` is the broadcast-grid axis the ``_check_cat_pair``
    analysis computes — carrier-aware shapes and broadcast joins the
    dspec grammar cannot express.
    """
    oa = _check_cat_pair(bound)
    return None if oa is None else {"$attr:DO": oa}


def _cat_hom_add(name: str, attr_key: _DimKey) -> Rewrite:
    """add(cat(a1,a2,D), cat(b1,b2,D)) = cat(add(a1,b1), add(a2,b2), D).

    Concat is a homomorphism for elementwise add.  This is the pure form
    of the additive-mask law when the mask is itself concat'd.
    """
    return R(
        name,
        Op.make(
            "add",
            Op.make("concat", "a1", "a2", **{attr_key: "DA"}),
            Op.make("concat", "b1", "b2", **{attr_key: "DB"}),
        ),
        Op.make(
            "concat",
            Op.make("add", "a1", "b1"),
            Op.make("add", "a2", "b2"),
            dim="DO",
        ),
        check=_cat_pair_check,
        derive=_cat_pair_derive,
        law="concat homomorphism over add: cat(a1,a2)+cat(b1,b2) = "
        "cat(a1+b1, a2+b2) — the free case of mask distribution.",
    )


def _cat_hom_masked_fill(name: str, attr_key: _DimKey) -> Rewrite:
    """masked_fill(cat(s1,s2,D), cat(m1,m2,DM), v) → per-block fills.

    The mask arrives already concat'd (e.g. chunked masks); block i
    pairs s_i with m_i directly, no split needed.
    """
    return R(
        name,
        Op.make(
            "masked_fill",
            Op.make("concat", "a1", "a2", **{attr_key: "DA"}),
            Op.make("concat", "b1", "b2", **{attr_key: "DB"}),
            "v",
        ),
        Op.make(
            "concat",
            Op.make("masked_fill", "a1", "b1", "v"),
            Op.make("masked_fill", "a2", "b2", "v"),
            dim="DO",
        ),
        check=_cat_pair_check,
        derive=_cat_pair_derive,
        law="masked_fill over two concat'd operands: the score concat "
        "and mask concat share an axis, so block i's mask is just "
        "m_i — the offset was already paid when the mask was "
        "concatenated.",
    )


#: Elementwise-mask ops pushed through a concat'd operand.  "slice"
#: variants emit ``split`` projections of the mask; "reuse" variants
#: fire when the mask broadcasts along the cat axis.  The canonical
#: ``dim`` spelling, both ``add`` operand orders, both ``where``
#: positions.
MASKED_FILL_CAT: list[Rewrite] = [
    _masked_fill_cat(f"masked_fill_cat_{mode}_{ak}", ak, mode)
    for mode in ("slice", "reuse")
    for ak in ("dim",)
]

ADD_MASK_CAT: list[Rewrite] = [
    _add_cat(
        f"add_{'m' if mask_first else 'cat'}_"
        f"{'cat' if mask_first else 'm'}_{mode}_{ak}",
        ak,
        mode,
        mask_first,
    )
    for mode in ("slice", "reuse")
    for mask_first in (False, True)
    for ak in ("dim",)
]

WHERE_CAT: list[Rewrite] = [
    _where_cat(
        f"where_cat_{'x' if cat_in_x else 'y'}_{mode}_{ak}",
        ak,
        mode,
        cat_in_x,
    )
    for mode in ("slice", "reuse")
    for cat_in_x in (True, False)
    for ak in ("dim",)
]

#: Concat homomorphism when BOTH operands arrive concat'd (a mask that
#: was itself built blockwise).  Same check as the single-operand
#: rules: the two cat axes must coincide on the broadcast result.
CAT_HOM: list[Rewrite] = [
    _cat_hom_add("cat_hom_add_dim", "dim"),
    _cat_hom_masked_fill("cat_hom_masked_fill_dim", "dim"),
]

#: The mask-distribution law set.  Together with OM_SPLIT these turn
#: ``softmax(mask(q @ cat kᵢ.T)) @ cat vᵢ`` — masked or causal — into
#: ``om_apply(⊕ᵢ om_elem(masked sᵢ, vᵢ))``.
OM_MASK_LAWS: list[Rewrite] = [
    *CAT_HOM,
    *MASKED_FILL_CAT,
    *ADD_MASK_CAT,
    *WHERE_CAT,
]


# ---------------------------------------------------------------------------
#  is_causal IS a mask — chunking the flag, not the mask.
#
#  ``sdpa(q, cat(k_i), cat(v_i), is_causal=True)`` carries no mask
#  operand, so OM_MASK_LAWS have nothing to slice.  But the flag is
#  only sugar for a materialised causal bad-mask over the concatenated
#  score matrix — unfold it into exactly that:
#
#      sdpa(q, cat(k_i), cat(v_i), is_causal)
#        = om_apply(om_elem(
#              masked_fill(cat(qs @ k_i.T), cmask(cat(qs @ k_i.T)), -inf),
#              cat(v_i)))        where qs = q · scale
#
#  and then the EXISTING laws do all the chunking: masked_fill_cat_slice
#  slices the materialised mask on the key axis (block i's positional
#  offset lives inside the slice — the same offset story as an
#  explicit causal mask), OM_SPLIT splits the carrier, and early rows
#  of late blocks (t < o_i, fully masked within that block) exercise
#  om_compose's isfinite guard exactly as in the materialised case.
#
#  The unfold needs two small generator ops, declared here via the
#  trace.py mechanism (op_def + a TORCH_BINDINGS export — phase 2c):
#
#  ``cmask(x, off=0)``
#      The causal bad-mask of an x-shaped score matrix: on x's last two
#      dims, out[...,t,j] = (j + off > t) — strict upper triangle
#      shifted ``off`` keys right, broadcast back to x's full shape.
#      Rank-preserving so ``typing.shape_of`` reads it through the default
#      case — which is what lets masked_fill_cat_slice derive split
#      sizes for n-ary concats.  Semantics follow torch's is_causal:
#      lower-LEFT triangular, i.e. j ≤ t, so a T_q=1 decode row sees
#      only key 0 (torch's documented footgun — kept, not patched).
#
#  ``fill(x, value=c)``
#      The constant map x ↦ c·1 (torch.full_like).  Needed because the
#      kernel's scale — 1/√E by default — is a DERIVED constant: derive
#      hooks can only fill attribute metavariables, never operand
#      positions, so a scalar cannot be materialised as a Const leaf.
#      fill puts it in an attr instead.  The scale is folded into q
#      once (qs = q·s ⇒ qs@k.T = s·qk.T), which keeps the score concat
#      visible to masked_fill_cat — scaling the concat'd scores would
#      bury them under a mul that no law distributes.
#
#  Also present: the unmasked companion — ``sdpa(q, cat k, cat v)``
#  with no mask at all chunks the same way minus the masked_fill.
#
#  Both spellings of the flag are covered: the ``is_causal``/``scale``
#  kwarg form and torch.export's positional ``arg4`` (dropout_p) /
#  ``arg5`` (is_causal) / ``arg6`` (scale) form.  A nonzero dropout_p
#  or a non-numeric scale vetoes the rewrite; sdpa with an explicit
#  attn_mask (a 4th operand) has its own law below — the mask slices
#  on the key axis in parallel with the k/v blocks.
#
#  EXPLICIT ATTN_MASK
#      ``sdpa(q, cat(k_i), cat(v_i), m, ...)`` chunks by composing the
#      mask into the score concat in ADDITIVE-BIAS form:
#
#          sdpa(q, cat k, cat v, m)
#            = om_apply(om_elem(
#                  add(cat(qs @ k_i.T), attnbias(m)), cat v))
#
#      ``attnbias`` is a one-input generator materialising torch's own
#      mask coercion — a float mask IS the additive bias, a bool mask
#      is a keep-mask torch internally rewrites to a 0/−inf bias:
#
#          attnbias(m) = m                     (m floating)
#          attnbias(m) = where(m, 0, −inf)     (m bool)
#
#      The term level carries no dtype (TensorType is shape-only), so
#      the dispatch lives in the lowering — this is what lets ONE law
#      cover both mask vocabularies soundly: add(s, attnbias(m)) is
#      exactly sdpa's masked-score semantics for either dtype.  Without
#      it the choice between add(s, m) and masked_fill(s, ~m, −inf)
#      would be a dtype guess the e-graph could not retract.  Once the
#      mask is in add-form the existing ADD_MASK_CAT laws slice it on
#      the key axis — ``split(attnbias(m), (K1,K2), -1, i)`` is block
#      i's mask columns, the same offset story as every other sliced
#      mask — and OM_SPLIT chunks the carrier.
#
#      Guard: the mask must broadcast against the (…,Tq,K1+K2) score
#      matrix with the key axis on its last dim — extent K1+K2 (sliced
#      per block) or 1/absent (broadcast reuse through ADD_MASK_CAT's
#      reuse mode).  The dropout veto is unchanged; is_causal=True
#      alongside a mask matches no pattern (torch forbids the combo).
# ---------------------------------------------------------------------------

op_def(
    "cmask",
    1,
    1,
    law="Causal-mask generator: cmask(x)[...,t,j] = (j + off > t) — "
    "the strict upper triangle of x's score plane, shifted off "
    "keys; materialises what sdpa's is_causal flag hides.",
)
op_def(
    "fill",
    1,
    1,
    law="Constant map x ↦ c·1 (full_like): the vehicle for derived "
    "scalar constants, which can only occupy attribute positions.",
)
op_def(
    "attnbias",
    1,
    1,
    law="sdpa's attn_mask in additive-bias form: a float mask passes "
    "through, a bool keep-mask becomes where(m, 0, -inf) — torch's "
    "own coercion, which the untyped (shape-only) term cannot "
    "spell.  Lets ONE sdpa-cat law serve both mask dtypes.",
)


def _cmask_torch(x: torch.Tensor, *a, **kw) -> torch.Tensor:
    off = int(kw.get("off", 0) or 0)
    t, k = x.shape[-2], x.shape[-1]
    keep = torch.ones(t, k, dtype=torch.bool, device=x.device).tril(
        -off
    )
    return keep.logical_not().expand(x.shape)


def _fill_torch(x: torch.Tensor, *a, **kw) -> torch.Tensor:
    return torch.full_like(x, float(kw.get("value", 0)))


def _attnbias_torch(m: torch.Tensor, *a, **kw) -> torch.Tensor:
    """Coerce an attn_mask the way Torch documents it.

    Float masks ARE the additive bias; bool keep-masks become the 0/-inf
    bias (positions with False are disallowed).  Shape-preserving and
    elementwise, so slicing commutes with it — split(attnbias(m)) is the
    bias of the slice.
    """
    if m.dtype == torch.bool:
        return torch.where(
            m,
            torch.zeros((), device=m.device),
            torch.full((), float("-inf"), device=m.device),
        )
    return m


def _om_decayed_torch(
    s: torch.Tensor, d: torch.Tensor, v: torch.Tensor, *a, **kw
) -> torch.Tensor:
    """Lower ``om_decayed`` — the spelled ``(s * d) @ v``, verbatim."""
    return (s * d) @ v


def _decay_torch(g: torch.Tensor, *a, **kw) -> torch.Tensor:
    """Lower ``decay`` — tril(γ^{i−j}) on the last two axes.

    γ's dims are LEADING (batch) dims: a scalar rate gives ``(T,T)``,
    a per-head ``(H,)`` rate ``(H,T,T)``.  ``torch.as_tensor``
    normalises Python-scalar leaves; their dtype is the ambient
    default — the term level carries no dtype.
    """
    t = int(kw.get("t", 0) or 0)
    g = torch.as_tensor(g)
    pos = torch.arange(t, dtype=g.dtype, device=g.device)
    diff = pos.unsqueeze(-1) - pos.unsqueeze(-2)
    return torch.tril(g[..., None, None] ** diff.clamp_min(0))


def _decay_vec_torch(g: torch.Tensor, *a, **kw) -> torch.Tensor:
    """Lower ``decay_vec`` — ``g^(sign·k + offset)``, ``k`` in ``[0,T)``.

    Same leading-batch-dims convention as ``decay``: a scalar rate
    gives ``(T,)``, a per-head ``(H,)`` rate ``(H,T)``.
    """
    t = int(kw.get("t", 0) or 0)
    off = int(kw.get("offset", 0) or 0)
    # ``or 1`` would swallow sign=0 — a legal exponent (the merge's
    # S1+S2 lands there): ``g^(0·k+o)`` is the constant g^o, not g^k.
    sign = int(kw["sign"]) if "sign" in kw else 1
    g = torch.as_tensor(g)
    pos = torch.arange(t, dtype=g.dtype, device=g.device)
    return g[..., None] ** (pos * sign + off)


#: Torch lowering bindings — an EXPORT, not an import-time mutation
#: (plan 0001 phase 2c): :class:`mlopt_core.ops.OpTable` folds this dict in
#: via ``register``/``full()``; importing this module registers nothing
#: into ``torch_bridge._IR_TO_TORCH``.
TORCH_BINDINGS: dict[str, Any] = {
    "cmask": _cmask_torch,
    "fill": _fill_torch,
    "attnbias": _attnbias_torch,
    # The decayed-attention element — declared with its laws at the
    # bottom of this module.
    "om_decayed": _om_decayed_torch,
    # The causal-decay matrix generator — same place.
    "decay": _decay_torch,
    # Its rank-1 vector factor — cross blocks of the chunked form.
    "decay_vec": _decay_vec_torch,
}

#: The masked_fill fill-value in rule RHSs.  _instantiate turns a
#: non-Op RHS leaf into a repr-keyed leaf enode; registering the Const
#: lets extraction decode "-inf" back to a Const instead of a bare
#: string (the registry is keyed by repr, and repr(Const) is str(value)).
_NEG_INF = Const(float("-inf"))
_LeafRegistry.register(_NEG_INF)


def _check_sdpa_cat(bound: dict) -> bool:
    """Check sdpa chunking over concatenated k/v blocks.

    The chunks are only valid when both cats are on the SEQUENCE axis
    (dim -2 of k and v — the key axis), the blocks are cat-compatible,
    q's head dim contracts k's, each key block pairs with its value
    block, batch dims broadcast per block, and the sdpa flags are benign
    (dropout_p == 0, numeric scale).
    """
    qs = _vshape(bound.get("q"))
    k1s, k2s = _vshape(bound.get("k1")), _vshape(bound.get("k2"))
    v1s, v2s = _vshape(bound.get("v1")), _vshape(bound.get("v2"))
    if not all(isinstance(s, tuple) for s in (qs, k1s, k2s, v1s, v2s)):
        return False
    if (
        len(qs) < 2
        or len(k1s) < 2
        or len(v1s) < 2
        or len(k2s) != len(k1s)
        or len(v2s) != len(v1s)
    ):
        return False
    kd, vd = bound.get("$attr:KD"), bound.get("$attr:VD")
    if not (isinstance(kd, int) and isinstance(vd, int)):
        return False
    nk, nv = len(k1s), len(v1s)
    if kd % nk != nk - 2:
        return False  # keys cat on seq axis
    if vd % nv != nv - 2:
        return False  # values cat on seq axis
    if not all(
        _dim_eq(k1s[i], k2s[i]) for i in range(nk) if i != nk - 2
    ):
        return False
    if not all(
        _dim_eq(v1s[i], v2s[i]) for i in range(nv) if i != nv - 2
    ):
        return False
    # q[...,d] contracts k[...,d]; k_i's key count is v_i's.
    if not (_dim_eq(qs[-1], k1s[-1]) and _dim_eq(qs[-1], k2s[-1])):
        return False
    if not (_dim_eq(k1s[-2], v1s[-2]) and _dim_eq(k2s[-2], v2s[-2])):
        return False
    # Per-block batch broadcast: q vs k_i, q vs v_i, k_i vs v_i.
    for ks, vs in ((k1s, v1s), (k2s, v2s)):
        if not (
            _broadcast_ok(qs[:-2], ks[:-2])
            and _broadcast_ok(qs[:-2], vs[:-2])
            and _broadcast_ok(ks[:-2], vs[:-2])
        ):
            return False
    dp = bound.get("$attr:DP")
    if dp is not None and dp != 0:
        return False  # dropout: not pure math
    sc = bound.get("$attr:SC")
    if sc is not None and (
        isinstance(sc, bool) or not isinstance(sc, (int, float))
    ):
        return False  # non-numeric scale
    # cannot derive 1/√E unless qs[-1] gives a numeric head dim
    return sc is not None or isinstance(qs[-1], int)


def _derive_sdpa_cat(bound: dict) -> dict | None:
    """Return the kernel's softmax scale.

    The bound ``scale`` attr if given, else the default 1/√E with
    E = q's head dim.  Lands in the RHS ``fill(q, value=SC)`` — the only
    way a derived number reaches an operand.

    Stays procedural: the fallback reads ``_vshape(q)`` (carrier-aware)
    and needs ``** -0.5`` — the dspec has neither the carrier-aware
    ``shape`` read nor a power op.
    """
    sc = bound.get("$attr:SC")
    if sc is None:
        e = _vshape(bound.get("q"))
        if not (isinstance(e, tuple) and e and isinstance(e[-1], int)):
            return None
        sc = float(e[-1]) ** -0.5
    return {"$attr:SC": float(sc)}


def _sdpa_cat_rhs(causal: bool) -> Op:
    """om_apply(om_elem(masked score-concat, v-concat)).

    The causal variant masks the concatenated scores with a materialised
    ``cmask``; the per-block chunking is left to OM_MASK_LAWS +
    OM_SPLIT.
    """
    qs = Op.make("mul", "q", Op.make("fill", "q", value="SC"))
    mm1 = Op.make(
        "matmul", qs, Op.make("transpose", "k1", dim0=-2, dim1=-1)
    )
    mm2 = Op.make(
        "matmul", qs, Op.make("transpose", "k2", dim0=-2, dim1=-1)
    )
    scores = Op.make("concat", mm1, mm2, dim=-1)
    if causal:
        scores = Op.make(
            "masked_fill", scores, Op.make("cmask", scores), _NEG_INF
        )
    return Op.make(
        "om_apply",
        Op.make(
            "om_elem", scores, Op.make("concat", "v1", "v2", dim="VD")
        ),
    )


def _sdpa_cat(
    name: str, attr_key: _DimKey, sdpa_attrs: dict, causal: bool
) -> Rewrite:
    return R(
        name,
        Op.make(
            "sdpa",
            "q",
            Op.make("concat", "k1", "k2", **{attr_key: "KD"}),
            Op.make("concat", "v1", "v2", **{attr_key: "VD"}),
            **sdpa_attrs,
        ),
        _sdpa_cat_rhs(causal),
        check=_check_sdpa_cat,
        derive=_derive_sdpa_cat,
        law=(
            "is_causal unfolds to a materialised causal mask: "
            "sdpa(q, cat k, cat v, is_causal) = om_apply(om_elem("
            "masked_fill(cat scaled-scores, cmask), cat v)) — the "
            "flag chunked, not the mask."
            if causal
            else "unmasked sdpa over concatenated keys/values is the plain "
            "om homomorphism on scaled scores: sdpa(q, cat k, cat v) "
            "= om_apply(om_elem(cat(qs@k_i.T), cat v_i))."
        ),
    )


#: torch.export emits sdpa positionally: arg4 = dropout_p,
#: arg5 = is_causal, arg6 = scale; hand-built terms use the kwarg
#: spellings is_causal=/scale=.  A literal ``True``/``False`` in the
#: pattern is an exact attr match — the flag, chunked.
_SDPA_CAUSAL_ATTRS: tuple[dict, ...] = (
    {"is_causal": True},
    {"is_causal": True, "scale": "SC"},
    {"arg5": True},
    {"arg4": "DP", "arg5": True},
    {"arg4": "DP", "arg5": True, "scale": "SC"},
    {"arg4": "DP", "arg5": True, "arg6": "SC"},
)

_SDPA_PLAIN_ATTRS: tuple[dict, ...] = (
    {},
    {"arg4": "DP"},
    {"arg5": False},
    {"arg4": "DP", "arg5": False},
    {"is_causal": False},
    {"scale": "SC"},
    {"arg4": "DP", "scale": "SC"},
)


def _check_sdpa_mask_cat(bound: dict) -> bool:
    """Check sdpa chunking with an explicit attn_mask.

    Everything ``_check_sdpa_cat`` requires (sequence-axis cats,
    per-block k_i/v_i pairing, batch broadcast, benign flags), plus:
    the explicit attn_mask must broadcast against the concatenated
    score matrix ``(…, Tq, K1+K2)`` with the key axis on its last dim —
    extent ``K1+K2`` (per-block slices) or 1 (broadcast reuse).  Any
    other extent, a broadcast-incompatible mask, or unknown key extents
    vetoes the rewrite — the split sizes could not be derived.
    """
    if not _check_sdpa_cat(bound):
        return False
    ms = _vshape(bound.get("m"))
    qs = _vshape(bound.get("q"))
    k1s, k2s = _vshape(bound.get("k1")), _vshape(bound.get("k2"))
    if not isinstance(ms, tuple):
        return False
    k1, k2 = k1s[-2], k2s[-2]
    if not (isinstance(k1, int) and isinstance(k2, int)):
        return False  # can't derive split sizes
    from mlopt_core.typing import INVALID, broadcast

    bb = broadcast(qs[:-2], k1s[:-2])
    if (
        bb is INVALID
    ):  # pragma: no cover — _check_sdpa_cat already proved this broadcast
        return False
    scores = (*tuple(bb), qs[-2], k1 + k2)
    b = broadcast(scores, ms)
    if b is INVALID or not isinstance(b, tuple) or not b:
        return False
    # Broadcast-compatible mask whose key-axis extent is Tk or 1.
    return b[-1] == k1 + k2


def _sdpa_cat_mask_rhs() -> Op:
    """om_apply(om_elem(add(cat scaled-scores, attnbias(m)), cat v)).

    The mask enters the score concat through ``attnbias`` — torch's own
    coercion to additive-bias form — so ONE law serves both mask dtypes
    and the EXISTING ADD_MASK_CAT laws do the slicing:
    ``split(attnbias(m), (K1,K2), -1, i)`` is block i's mask columns.
    Chunking then proceeds exactly like the cmask path: OM_SPLIT splits
    the carrier and fully-masked blocks exercise om_compose's isfinite
    guard.
    """
    qs = Op.make("mul", "q", Op.make("fill", "q", value="SC"))
    mm1 = Op.make(
        "matmul", qs, Op.make("transpose", "k1", dim0=-2, dim1=-1)
    )
    mm2 = Op.make(
        "matmul", qs, Op.make("transpose", "k2", dim0=-2, dim1=-1)
    )
    scores = Op.make("concat", mm1, mm2, dim=-1)
    masked = Op.make("add", scores, Op.make("attnbias", "m"))
    return Op.make(
        "om_apply",
        Op.make(
            "om_elem", masked, Op.make("concat", "v1", "v2", dim="VD")
        ),
    )


def _sdpa_cat_masked(
    name: str, attr_key: _DimKey, sdpa_attrs: dict
) -> Rewrite:
    return R(
        name,
        Op.make(
            "sdpa",
            "q",
            Op.make("concat", "k1", "k2", **{attr_key: "KD"}),
            Op.make("concat", "v1", "v2", **{attr_key: "VD"}),
            "m",
            **sdpa_attrs,
        ),
        _sdpa_cat_mask_rhs(),
        check=_check_sdpa_mask_cat,
        derive=_derive_sdpa_cat,
        law="explicit attn_mask over concatenated keys/values: "
        "sdpa(q, cat k, cat v, m) = om_apply(om_elem(add(cat "
        "scaled-scores, attnbias(m)), cat v)) — the mask slices on "
        "the key axis in parallel with the k/v blocks, dtype-"
        "agnostic through the attnbias coercion.",
    )


#: Attr spellings for the masked form: torch.export emits the mask as
#: the 4th positional operand and omits defaulted flags; hand-built
#: terms may spell dropout/is_causal positionally or as kwargs.  No
#: is_causal=True pattern — torch forbids mask+causal together.
_SDPA_MASK_ATTRS: tuple[dict, ...] = (
    {},
    {"scale": "SC"},
    {"is_causal": False},
    {"is_causal": False, "scale": "SC"},
    {"arg4": "DP"},
    {"arg5": False},
    {"arg4": "DP", "arg5": False},
    {"arg4": "DP", "scale": "SC"},
    {"arg4": "DP", "arg5": False, "arg6": "SC"},
    {"arg5": False, "arg6": "SC"},
)

#: sdpa-over-concat laws: the causal-flag unfold, the unmasked
#: companion, and the explicit-attn_mask form — the canonical concat
#: ``dim`` spelling.
SDPA_CAT_LAWS: list[Rewrite] = [
    *(
        _sdpa_cat(f"sdpa_cat_causal_{i}_{ak}", ak, dict(attrs), True)
        for i, attrs in enumerate(_SDPA_CAUSAL_ATTRS)
        for ak in ("dim",)
    ),
    *(
        _sdpa_cat(f"sdpa_cat_{i}_{ak}", ak, dict(attrs), False)
        for i, attrs in enumerate(_SDPA_PLAIN_ATTRS)
        for ak in ("dim",)
    ),
    *(
        _sdpa_cat_masked(f"sdpa_cat_mask_{i}_{ak}", ak, dict(attrs))
        for i, attrs in enumerate(_SDPA_MASK_ATTRS)
        for ak in ("dim",)
    ),
]


# ---------------------------------------------------------------------------
#  The decayed-attention element — ``matmul(mul(s, d), v)`` as one op.
#
#  The RetNet/GLA inner-chunk form ``(QKᵀ ⊙ D) V`` spells
#  ``matmul(mul(s, d), v)``: the score/prob matrix ``s`` (…,T,K) is
#  weighted ELEMENTWISE by a position/decay matrix ``d`` (a causal
#  ``γ^{t−j}`` triangle in RetNet, broadcastable against ``s`` under
#  torch.mul semantics — a scalar or lower-rank decay is legal), then
#  contracted on its last axis by the values ``v`` (…,K,dv) under
#  ordinary ``torch.matmul`` rules (batch dims broadcasting), giving
#  ``(…,T,dv)``.  Unlike softmax attention there is no normaliser:
#  ``om_elem``'s (m,l,a) triple has nothing to carry and OM_LIFT's
#  softmax guard declines the spelled term — which is why the form
#  verified FALSE against every existing carrier.
#
#  ``om_decayed`` is a VALUE-level element: it evaluates directly to
#  the applied output — no carrier triple, no separate apply op (the
#  decayed product's composition over key chunks is plain ``add`` —
#  a monoid already in the vocabulary).  Its executor is the spelled
#  body itself — a reference lowering, correct by construction; a
#  fused kernel is codegen's problem (``mlopt_discovery.genkernel``),
#  not this declaration's.
#
#  The lift accepts exactly what torch would compute: the mul must
#  broadcast and the contraction must be well-typed — the same gate
#  the op's shape rule (:func:`mlopt_core.typing._om_decayed_shape`)
#  and FLOP entry price.  The name marks the motivating reading
#  (decay-masked scores); the *semantics* is just the fused weighted
#  contraction, so any well-typed ``mul@matmul`` site lifts soundly.
# ---------------------------------------------------------------------------

op_def(
    "om_decayed",
    3,
    1,
    law="Decayed-attention element: om_decayed(s, d, v) = (s ⊙ d) @ v — "
    "score matrix decay-weighted elementwise, then contracted by the "
    "values (the RetNet inner-chunk form).  A value-level op: the "
    "element IS the applied result.",
)


def _check_om_decayed(bound: dict) -> bool:
    """Require the spelled ``mul(s, d) @ v`` to be well-typed.

    The mul broadcasts under torch semantics (a scalar decay is
    legal); the matmul contracts the broadcast result's last axis
    against ``v`` per :func:`mlopt_core.typing.matmul_shape`.  An
    unresolvable operand shape, an unbroadcastable ``d``, or a
    contraction the shapes do not admit vetoes the lift — the element
    only ever names a program torch itself would run.
    """
    ss, ds, vs = (
        _vshape(bound.get("s")),
        _vshape(bound.get("d")),
        _vshape(bound.get("v")),
    )
    if not all(isinstance(x, tuple) for x in (ss, ds, vs)):
        return False
    from mlopt_core.typing import INVALID, broadcast, matmul_shape

    m = broadcast(ss, ds)
    if m is INVALID:
        return False
    return matmul_shape(m, vs) is not None


OM_DECAYED_LIFT = R(
    "om_decayed_lift",
    Op.make("matmul", Op.make("mul", "s", "d"), "v"),
    Op.make("om_decayed", "s", "d", "v"),
    law="matmul(mul(s,d), v) lifts into the decayed-attention element: "
    "om_decayed(s,d,v) = (s ⊙ d) @ v — the RetNet inner-chunk form "
    "no other carrier element expresses (om_elem carries a softmax "
    "normaliser; the decayed form has none).",
    check=_check_om_decayed,
)

OM_DECAYED_UNLIFT = R(
    "om_decayed_unlift",
    Op.make("om_decayed", "s", "d", "v"),
    Op.make("matmul", Op.make("mul", "s", "d"), "v"),
    law="om_decayed(s,d,v) unfolds back to the spelled mul+matmul.",
)


# ---------------------------------------------------------------------------
#  The decay matrix — ``decay(γ, t=T)`` as a first-class value.
#
#  RetNet-style code builds the causal decay triangle in-graph:
#  ``tril(γ^{pos_i − pos_j})`` out of an arange pair (see the zoo's
#  ``_ChunkedRetention``).  That spelling is five ops the chunk laws
#  below cannot see through — an opaque ``mul`` operand.  ``decay``
#  names the matrix itself: γ is the operand (its dims are LEADING
#  batch dims — a scalar gives ``(T,T)``, a per-head ``(H,)`` rate
#  ``(H,T,T)``) and the extent rides in the ``t`` attr, the only place
#  a derived constant can land.
#
#  With the structure explicit, the block decomposition is
#  equational:
#
#      decay(γ, T1+T2)  =  [ decay(γ,T1)   0          ]
#                        [ cross(γ,T1,T2)  decay(γ,T2)]
#
#  — the diagonal blocks ARE smaller decays (``decay_block_diag``
#  regenerates them in place of sliced material), the strict-upper
#  block is identically zero (kept materialised for now: spelling the
#  zero needs a dtype-aware ``zeros`` the shape-only IR cannot mint —
#  see the note under ``decay_block_diag``), and the cross block is
#  the γ^{(o+r)−c} triangle the inter-chunk recurrence carries.
#
#  ``decay_lift`` recognises the spelled subgraph — ``tril(pow(γ,
#  clamp_min(sub(unsq(pos,1), unsq(pos,0)), 0)))`` — and folds it into
#  the generator.  The guards are load-bearing: the two aranges must
#  be the SAME unit-step sequence (``p_i − p_j = i − j`` fails for
#  step ≠ 1 or different bounds), the unsqueezes must put the column
#  operand first (swapped gives j−i), and the clamp bound must be 0.
# ---------------------------------------------------------------------------

op_def(
    "decay",
    1,
    1,
    law="Causal-decay generator: decay(g, t=T)[...,i,j] = g^(i-j) for "
    "i >= j, else 0 — the RetNet/GLA decay matrix as a value, so its "
    "block structure is equational rather than a spelled subgraph.",
)


def _positional_vals(node: Op) -> list:
    """Collect an op's bound scalars: Const operands or ``argN`` attrs."""
    vals = [a.value for a in node.args if isinstance(a, Const)]
    if vals:
        return vals
    from mlopt_core.attrs import is_positional_attr

    return [
        node.attrs[k]
        for k in sorted(
            (k for k in node.attrs if is_positional_attr(k)),
            key=lambda k: int(k[3:]),
        )
    ]


def _arange_params(node: Any) -> tuple | None:
    """``arange`` bounds as ``(start, end, step)``, or ``None``.

    Covers the Const-operand minting and the exported ``arg0``/
    ``arg1``/``arg2`` attrs — the same two spellings
    :func:`mlopt_core.typing._arange_shape` reads.  Defaults follow
    torch: start 0, step 1.
    """
    if not (isinstance(node, Op) and node.op == "arange"):
        return None
    vals = _positional_vals(node)
    if not all(isinstance(v, (int, float)) for v in vals):
        return None
    if len(vals) == 1:
        return (0, vals[0], 1)
    if len(vals) == 2:
        return (vals[0], vals[1], 1)
    if len(vals) == 3:
        return tuple(vals)
    return None


def _check_decay_lift(bound: dict) -> bool:
    """Require the spelled exponent to be a real position difference.

    The LHS already pins ``sub(unsq(a1), unsq(a2))`` inside
    ``clamp_min(·, 0)`` inside ``pow`` inside ``tril``; the guard
    proves a1/a2 are the SAME unit-step arange (equal bounds, step
    exactly 1 — else ``p_i − p_j ≠ i − j``), that the unsqueezes put
    the column operand first (a (T,1) minus a (1,T) is i−j; swapped
    is j−i — a different program), and the clamp bound is 0.
    ``tril`` then leaves exactly ``γ^{i−j}`` on the lower triangle.
    """
    a1, a2 = bound.get("a1"), bound.get("a2")
    u1, u2 = bound.get("$attr:U1"), bound.get("$attr:U2")
    cm = bound.get("$attr:CM")
    if not (isinstance(u1, int) and isinstance(u2, int)):
        return False
    if not (isinstance(cm, (int, float)) and cm == 0):
        return False
    p1, p2 = _arange_params(a1), _arange_params(a2)
    if p1 is None or p1 != p2 or p1[2] != 1:
        return False
    return u1 % 2 == 1 and u2 % 2 == 0


def _derive_decay_lift(bound: dict) -> dict | None:
    """``T`` := the shared arange's extent.

    Stays procedural: ``_arange_params`` reads the bound term's OWN op
    attrs (start/end/step spellings) — the dspec grammar has no
    term-structure read, only bound ``$attr:`` metavars and shapes.
    """
    p = _arange_params(bound.get("a1"))
    if p is None:
        return None
    import math

    t = math.ceil((p[1] - p[0]) / p[2]) if p[2] else 0
    return {"$attr:T": t} if t > 0 else None


def _decay_lift(name: str, diag: bool) -> Rewrite:
    """Spelled ``tril(γ^{clamp(i−j,0)})`` → ``decay(γ, t=T)``.

    ``diag`` covers the attributed ``tril(·, diagonal=0)`` spelling;
    a nonzero diagonal is a different mask and matches neither LHS.
    """
    diff = Op.make(
        "clamp_min",
        Op.make(
            "sub",
            Op.make("unsqueeze", "a1", dim="U1"),
            Op.make("unsqueeze", "a2", dim="U2"),
        ),
        min="CM",
    )
    powered = Op.make("pow", "g", diff)
    lhs = (
        Op.make("tril", powered, diagonal=0)
        if diag
        else Op.make("tril", powered)
    )
    return R(
        name,
        lhs,
        Op.make("decay", "g", t="T"),
        check=_check_decay_lift,
        derive=_derive_decay_lift,
        law="the spelled causal-decay subgraph is the decay matrix "
        "itself: tril(g^clamp_min(pos_i-pos_j,0)) = decay(g, T) — "
        "the structure made a value so the block laws can see it.",
    )


DECAY_LIFT = _decay_lift("decay_lift", False)
DECAY_LIFT_DIAG = _decay_lift("decay_lift_diag", True)


# ---------------------------------------------------------------------------
#  Chunking the decayed product — the block decomposition.
#
#  ``(S ⊙ D) @ V`` over concatenated operands splits along BOTH
#  matmul axes, and each half is exact — no approximation:
#
#  ROWS (the query axis, −2 of S):
#      om_decayed(cat_T(s1,s2), d, v)
#        = cat_T( om_decayed(s1, d_hi, v), om_decayed(s2, d_lo, v) )
#      where d_i is block i's slice of d on the row axis — or d
#      itself when it broadcasts along that axis (extent 1, a shared
#      per-chunk decay under equal block sizes, or no such axis).
#
#  KEYS (the contraction axis, −1 of S / −2 of V):
#      om_decayed(s, d, cat_K(v1,v2))
#        = om_decayed(s_L, d_L, v1) + om_decayed(s_R, d_R, v2)
#      — the block-column decomposition of the contraction, s/d
#      sliced on the key axis when they cover it.
#
#  Composed on ``om_decayed(cat_T(s1,s2), decay(γ,T), cat_K(v1,v2))``
#  the two splits produce the four-block chunked form: the diagonal
#  blocks' ``split(split(decay))`` operands fold back to
#  ``decay(γ, T_i)`` by ``decay_block_diag`` (the intra terms), the
#  (1,0) block is the cross-decay term the recurrence carries, and
#  the (0,1) block multiplies a provably-zero triangle — still
#  materialised (the zero-elision needs a typed ``zeros``; the IR's
#  leaves carry no dtype).  Slicing is exact but NOT free — the
#  chunked form pays extra launches for identical FLOPs, so
#  extraction keeps the fused element unless a cost model or a
#  deeper structure law (the ξ·ηᵀ factorisation of the cross block,
#  which needs the score's OWN q·kᵀ rank — see the module note) makes
#  it win.
# ---------------------------------------------------------------------------


def _sliced(sh: tuple, axis: int, ext: int) -> tuple:
    """Shape ``sh`` with ``axis``'s extent replaced by ``ext``."""
    return (*sh[:axis], ext, *sh[axis + 1 :])


def _block_out_ok(si, di, vi):
    """Output shape of one decayed block, or ``None`` when ill-typed.

    The block must satisfy the same gate the element itself does —
    the mul must broadcast and the contraction must form.
    """
    from mlopt_core.typing import INVALID, broadcast, matmul_shape

    m = broadcast(si, di)
    if m is INVALID:
        return None
    return matmul_shape(m, vi)


def _outs_cat_ok(o1, o2, axis: int) -> bool:
    """Two block outputs concat on ``axis``: same rank, equal off-axis."""
    return len(o1) == len(o2) and all(
        _dim_eq(o1[i], o2[i]) for i in range(len(o1)) if i != axis
    )


def _check_om_decayed_rows(mode: str):
    """Check the row-axis split — ``mode`` is "slice" or "reuse".

    The scores concat must be on the ROW axis (−2); ``d`` is the
    decay operand — sliced on its image of the row axis when it
    covers the whole concatenated extent (``_cat_axis_plan`` mode
    "slice"), reused verbatim when it broadcasts along that axis
    ("reuse").  Every admitted block must be a well-typed decayed
    contraction and the two outputs must concat on −2.
    """

    def check(bound: dict) -> bool:
        D = bound.get("$attr:D")
        s1, s2 = _vshape(bound.get("s1")), _vshape(bound.get("s2"))
        vs, ds = _vshape(bound.get("v")), _vshape(bound.get("d"))
        if not all(isinstance(x, tuple) for x in (s1, s2, vs, ds)):
            return False
        if not (isinstance(D, int) and len(s1) == len(s2)):
            return False
        r = len(s1)
        if r < 2 or D % r != r - 2:
            return False  # the cat axis must be the row axis
        plan = _cat_axis_plan(bound, "d")
        if plan is None or plan["mode"] != mode:
            return False
        outs = []
        for i, si in enumerate((s1, s2)):
            di = (
                ds
                if mode == "reuse"
                else _sliced(ds, plan["md"], plan["sizes"][i])
            )
            o = _block_out_ok(si, di, vs)
            if o is None:
                return False
            outs.append(o)
        return _outs_cat_ok(outs[0], outs[1], len(outs[0]) - 2)

    return check


def _derive_om_decayed_rows(bound: dict) -> dict | None:
    """Split sizes + the row axis in ``d``'s own coords (slice mode).

    Stays procedural: mints attrs from ``_cat_axis_plan`` — the
    carrier-aware shape analysis the dspec grammar cannot express.
    """
    plan = _cat_axis_plan(bound, "d")
    if plan is None or plan["mode"] != "slice":
        return None
    return {"$attr:SZ": plan["sizes"], "$attr:MD": plan["md"]}


def _om_decayed_chunk_rows(name: str, mode: str) -> Rewrite:
    """``om_decayed(cat(s1,s2,D), d, v)`` → cat of per-row-block calls."""
    d1, d2, derive = "d", "d", None
    if mode == "slice":
        d1 = Op.make("split", "d", sizes="SZ", dim="MD", index=0)
        d2 = Op.make("split", "d", sizes="SZ", dim="MD", index=1)
        derive = _derive_om_decayed_rows
    return R(
        name,
        Op.make(
            "om_decayed",
            Op.make("concat", "s1", "s2", dim="D"),
            "d",
            "v",
        ),
        Op.make(
            "concat",
            Op.make("om_decayed", "s1", d1, "v"),
            Op.make("om_decayed", "s2", d2, "v"),
            dim=-2,
        ),
        check=_check_om_decayed_rows(mode),
        derive=derive,
        law="row-block decomposition of the decayed product: "
        "(cat_T(s1,s2) ⊙ d) @ v = cat_T((s1⊙d1)v, (s2⊙d2)v) — the "
        "decay slices with the scores on the row axis, or broadcasts "
        "into both blocks.",
    )


def _check_om_decayed_rows_cat(bound: dict) -> bool:
    """Both operands concat'd on the row axis — pair the blocks.

    Reuses ``_check_cat_pair`` under renamed metvars: the two cats
    must land on the SAME axis of the broadcast score plane, and that
    axis must be the row axis (``ro − 2``); each block must then be a
    well-typed decayed contraction.
    """
    pair = {
        "a1": bound.get("s1"),
        "a2": bound.get("s2"),
        "b1": bound.get("d1"),
        "b2": bound.get("d2"),
        "$attr:DA": bound.get("$attr:SD"),
        "$attr:DB": bound.get("$attr:DD"),
    }
    oa = _check_cat_pair(pair)
    if oa is None:
        return False
    s1, s2 = _vshape(bound.get("s1")), _vshape(bound.get("s2"))
    d1, d2 = _vshape(bound.get("d1")), _vshape(bound.get("d2"))
    vs = _vshape(bound.get("v"))
    ro = max(len(s1), len(d1))
    if oa != ro - 2:
        return False
    o1 = _block_out_ok(s1, d1, vs)
    o2 = _block_out_ok(s2, d2, vs)
    if o1 is None or o2 is None:
        return False
    return _outs_cat_ok(o1, o2, len(o1) - 2)


def _v_cat_pair(bound: dict) -> tuple | None:
    """``concat(v1,v2,VD)`` on the key axis — return ``(K1, K2)``.

    The cat must be on dim −2 of v (keys, pre-contraction), the
    blocks cat-compatible off it, and the key extents concrete —
    the split sizes are derived from them.
    """
    VD = bound.get("$attr:VD")
    v1, v2 = _vshape(bound.get("v1")), _vshape(bound.get("v2"))
    if not (
        isinstance(v1, tuple)
        and isinstance(v2, tuple)
        and isinstance(VD, int)
        and len(v1) == len(v2)
        and len(v1) >= 2
    ):
        return None
    nv = len(v1)
    if VD % nv != nv - 2 or not _off_axis_eq(v1, v2, nv - 2):
        return None
    k1, k2 = v1[nv - 2], v2[nv - 2]
    return (
        (k1, k2)
        if isinstance(k1, int) and isinstance(k2, int)
        else None
    )


def _off_axis_eq(a: tuple, b: tuple, skip: int) -> bool:
    """All dims equal except axis ``skip`` — cat-compat off the cat axis."""
    return all(_dim_eq(a[i], b[i]) for i in range(len(a)) if i != skip)


def _s_cat_blocks(bound: dict, sizes: tuple):
    """Score blocks arriving already concat'd on the key axis (−1).

    The cat must sit on s's LAST axis, the blocks be cat-compatible
    off it, and each block's key extent pair with its value block.
    """
    s1, s2 = _vshape(bound.get("s1")), _vshape(bound.get("s2"))
    SD = bound.get("$attr:SD")
    if not (
        isinstance(s1, tuple)
        and isinstance(s2, tuple)
        and isinstance(SD, int)
        and s1
        and len(s1) == len(s2)
    ):
        return None
    if SD % len(s1) != len(s1) - 1:
        return None
    if not _off_axis_eq(s1, s2, len(s1) - 1):
        return None
    if not (_dim_eq(s1[-1], sizes[0]) and _dim_eq(s2[-1], sizes[1])):
        return None
    return (s1, s2)


def _s_key_blocks(bound: dict, s_cat: bool, sizes: tuple):
    """Per-block score shapes — split or concat'd on the key axis.

    ``s_cat=False``: the whole ``s`` splits at −1, so its last extent
    must equal K1+K2 concretely.  ``s_cat=True``: the score blocks
    arrive already concat'd on −1 — s_i pairs with v_i directly.
    """
    if s_cat:
        return _s_cat_blocks(bound, sizes)
    ss = _vshape(bound.get("s"))
    if not (
        isinstance(ss, tuple) and ss and ss[-1] == sizes[0] + sizes[1]
    ):
        return None
    return (
        _sliced(ss, len(ss) - 1, sizes[0]),
        _sliced(ss, len(ss) - 1, sizes[1]),
    )


def _d_key_mode(ds: tuple, sizes: tuple) -> str | None:
    """Classify ``d`` on the key axis: "slice" / "reuse" / veto.

    The key axis is ``d``'s LAST (it right-aligns with the score
    plane's contracted axis).  Extent K1+K2 slices; extent 1 or a
    scalar decay broadcasts into both blocks; anything else — an
    unproven (``None``) extent included — vetoes.
    """
    if not ds:
        return "reuse"
    if ds[-1] == sizes[0] + sizes[1]:
        return "slice"
    if ds[-1] == 1:
        return "reuse"
    return None


def _key_cat_plan(bound: dict, s_cat: bool) -> dict | None:
    """Plan the key-axis split of a decayed product.

    Returns ``{"mode", "sizes"}`` — the (K1,K2) split and how ``d``
    distributes — or ``None`` when the rewrite cannot be proven
    well-typed: values must cat on −2 with concrete key extents, the
    score blocks must pair with them, and every emitted block must be
    a valid decayed contraction.  The ``add`` needs no broadcast
    check — the key split leaves the row axis and batch dims shared
    and the value cat already pinned a common ``dv``.
    """
    pair = _v_cat_pair(bound)
    if pair is None:
        return None
    sblocks = _s_key_blocks(bound, s_cat, pair)
    if sblocks is None:
        return None
    ds = _vshape(bound.get("d"))
    if not isinstance(ds, tuple):
        return None
    mode = _d_key_mode(ds, pair)
    if mode is None:
        return None
    v1, v2 = _vshape(bound.get("v1")), _vshape(bound.get("v2"))
    dblocks = (
        (
            _sliced(ds, len(ds) - 1, pair[0]),
            _sliced(ds, len(ds) - 1, pair[1]),
        )
        if mode == "slice"
        else (ds, ds)
    )
    outs = [
        _block_out_ok(si, di, vi)
        for si, di, vi in zip(sblocks, dblocks, (v1, v2), strict=True)
    ]
    if any(o is None for o in outs):
        return None
    return {"mode": mode, "sizes": pair}


def _check_key_cat(s_cat: bool, mode: str):
    """Check the key-axis split — gate on the plan's ``d`` mode."""

    def check(bound: dict) -> bool:
        plan = _key_cat_plan(bound, s_cat)
        return plan is not None and plan["mode"] == mode

    return check


def _derive_key_cat(s_cat: bool):
    """Derive ``SZ`` = the (K1,K2) key extents from the v blocks.

    Stays procedural: the extents come from ``_key_cat_plan`` —
    carrier-aware ``_vshape`` reads plus the well-typedness veto.
    """

    def derive(bound: dict) -> dict | None:
        plan = _key_cat_plan(bound, s_cat)
        return None if plan is None else {"$attr:SZ": plan["sizes"]}

    return derive


def _om_decayed_chunk_keys(name: str, mode: str) -> Rewrite:
    """``om_decayed(s, d, cat(v1,v2,-2))`` → add of key-block calls."""
    s1 = Op.make("split", "s", sizes="SZ", dim=-1, index=0)
    s2 = Op.make("split", "s", sizes="SZ", dim=-1, index=1)
    d1, d2 = "d", "d"
    if mode == "slice":
        d1 = Op.make("split", "d", sizes="SZ", dim=-1, index=0)
        d2 = Op.make("split", "d", sizes="SZ", dim=-1, index=1)
    return R(
        name,
        Op.make(
            "om_decayed",
            "s",
            "d",
            Op.make("concat", "v1", "v2", dim="VD"),
        ),
        Op.make(
            "add",
            Op.make("om_decayed", s1, d1, "v1"),
            Op.make("om_decayed", s2, d2, "v2"),
        ),
        check=_check_key_cat(False, mode),
        derive=_derive_key_cat(False),
        law="key-block decomposition of the decayed product: "
        "(s ⊙ d) @ cat(v1,v2) = (s_L⊙d_L)v1 + (s_R⊙d_R)v2 — the "
        "contraction's own concat split; the cross term lands here.",
    )


def _om_decayed_chunk_keys_cat(name: str, mode: str) -> Rewrite:
    """Scores ALREADY concat'd on −1 — pair s_i with v_i directly."""
    d1, d2 = "d", "d"
    derive = None
    if mode == "slice":
        d1 = Op.make("split", "d", sizes="SZ", dim=-1, index=0)
        d2 = Op.make("split", "d", sizes="SZ", dim=-1, index=1)
        derive = _derive_key_cat(True)
    return R(
        name,
        Op.make(
            "om_decayed",
            Op.make("concat", "s1", "s2", dim="SD"),
            "d",
            Op.make("concat", "v1", "v2", dim="VD"),
        ),
        Op.make(
            "add",
            Op.make("om_decayed", "s1", d1, "v1"),
            Op.make("om_decayed", "s2", d2, "v2"),
        ),
        check=_check_key_cat(True, mode),
        derive=derive,
        law="both operands arrive key-blocked: (cat_K(s1,s2) ⊙ d) @ "
        "cat(v1,v2) = (s1⊙d_L)v1 + (s2⊙d_R)v2 — the chunked score "
        "form exported by blockwise QKᵀ.",
    )


#: Row- and key-axis chunk splits of ``om_decayed``.  The "slice"
#: variants emit ``split`` projections of ``d``; "reuse" variants
#: fire when the decay broadcasts along the split axis; the ``_cat``
#: rows law pairs operands that both arrive concat'd.
OM_DECAYED_CHUNK_LAWS: list[Rewrite] = [
    _om_decayed_chunk_rows("om_decayed_chunk_rows_slice", "slice"),
    _om_decayed_chunk_rows("om_decayed_chunk_rows_reuse", "reuse"),
    R(
        "om_decayed_chunk_rows_cat",
        Op.make(
            "om_decayed",
            Op.make("concat", "s1", "s2", dim="SD"),
            Op.make("concat", "d1", "d2", dim="DD"),
            "v",
        ),
        Op.make(
            "concat",
            Op.make("om_decayed", "s1", "d1", "v"),
            Op.make("om_decayed", "s2", "d2", "v"),
            dim=-2,
        ),
        check=_check_om_decayed_rows_cat,
        law="score and decay arrive concat'd on the row axis: pair "
        "the blocks — cat_T((s_i⊙d_i)v).",
    ),
    _om_decayed_chunk_keys("om_decayed_chunk_keys_slice", "slice"),
    _om_decayed_chunk_keys("om_decayed_chunk_keys_reuse", "reuse"),
    _om_decayed_chunk_keys_cat(
        "om_decayed_chunk_keys_cat_slice", "slice"
    ),
    _om_decayed_chunk_keys_cat(
        "om_decayed_chunk_keys_cat_reuse", "reuse"
    ),
]


# ---------------------------------------------------------------------------
#  The decay's own block structure.
#
#  The (i,i) diagonal block of ``decay(γ, T1+T2)`` is literally
#  ``decay(γ, T_i)``: equal row and column offsets cancel, leaving
#  ``γ^{r−c}`` on a smaller square.  ``split`` of a ``split`` with the
#  SAME sizes and index is the diagonal — the law regenerates the
#  block instead of slicing it out of the materialised matrix (one
#  small pow against a slice of a big one — and, more importantly,
#  the regenerated form is what later structure laws see).
#
#  What this still cannot spell (documented, not hacked):
#  * the strict-upper block is identically zero, but eliding it needs
#    a ``zeros`` whose DTYPE is the output's — the term level is
#    shape-only, so a minted ``zeros(shape=…)`` would carry the wrong
#    dtype and fail fp64 equality; the zero block stays a
#    materialised ``split`` until a dtype-aware factory exists;
#  * the cross block's ξ·ηᵀ rank-1 factorisation — γ^{(o+i)−j} splits
#    as a row scale times a column scale, but the vectors need a
#    decay-VECTOR generator this vocabulary does not yet have (a
#    ``decay(g, t=…)`` row/column form); until then the cross term is
#    the materialised slice — correct, just not linearised.
# ---------------------------------------------------------------------------


def _diag_attrs(bound: dict) -> tuple | None:
    """Read the bound ``(SZ, I)`` of a diagonal double-split.

    The sizes must be a pair of ints summing to the matrix extent
    ``t`` and the shared index must name one of the two blocks.
    """
    sz, i = bound.get("$attr:SZ"), bound.get("$attr:I")
    t = bound.get("$attr:T")
    if not (
        isinstance(sz, (tuple, list))
        and len(sz) == 2
        and all(isinstance(x, int) for x in sz)
        and isinstance(i, int)
        and 0 <= i < 2
        and isinstance(t, int)
        and t == sz[0] + sz[1]
    ):
        return None
    return (sz, i)


def _check_decay_block_diag(bound: dict) -> bool:
    """``split_i(split_i(decay(γ,T), SZ, −2), SZ, −1)`` is the (i,i) block.

    One sizes tuple and one index bind BOTH splits (equal row and
    column offsets — the diagonal, not an arbitrary block), the inner
    split must sit on the decay's row axis and the outer on its
    column axis (mod the decay rank — γ's dims are leading batch
    dims), and the block extent must be concrete so ``TC`` derives.
    """
    if _diag_attrs(bound) is None:
        return False
    dr, dc = bound.get("$attr:DR"), bound.get("$attr:DC")
    if not (isinstance(dr, int) and isinstance(dc, int)):
        return False
    gs = _vshape(bound.get("g"))
    if not isinstance(gs, tuple):
        return dr == -2 and dc == -1
    r = len(gs) + 2
    return dr % r == r - 2 and dc % r == r - 1


#: The ``_diag_attrs`` contract as data: ``SZ`` a 2-element int
#: partition of ``T``, ``I`` the block index in ``{0, 1}`` —
#: ``TC := SZ[I]``.
_DIAG_COND = (
    "and",
    ("attr-ints", "SZ"),
    ("attr-cmp", ("len", ("attr", "SZ")), "==", 2),
    ("attr-type", "I", "int"),
    ("attr-cmp", 0, "<=", "I"),
    ("attr-cmp", "I", "<", 2),
    ("attr-type", "T", "int"),
    (
        "attr-cmp",
        "T",
        "==",
        ("add", ("nth", ("attr", "SZ"), 0), ("nth", ("attr", "SZ"), 1)),
    ),
)
_DSPEC_DECAY_BLOCK_DIAG = {
    "TC": ("guard", _DIAG_COND, ("nth", ("attr", "SZ"), ("attr", "I"))),
}


DECAY_BLOCK_DIAG = R(
    "decay_block_diag",
    Op.make(
        "split",
        Op.make(
            "split",
            Op.make("decay", "g", t="T"),
            sizes="SZ",
            dim="DR",
            index="I",
        ),
        sizes="SZ",
        dim="DC",
        index="I",
    ),
    Op.make("decay", "g", t="TC"),
    check=_check_decay_block_diag,
    dspec=_DSPEC_DECAY_BLOCK_DIAG,
    law="the (i,i) block of the causal decay triangle IS the local "
    "decay of chunk i — g^((o+r)-(o+c)) = g^(r-c): diagonal blocks "
    "regenerate instead of slicing materialised decay.",
)


# ---------------------------------------------------------------------------
#  The decay vector — ``decay_vec(g, t=T, offset=o, sign=s)`` makes the
#  cross block rank-1:  ``X[r,c] = g^(o_i - o_j + r - c) = xi[r] * eta[c]``
#  with ``xi = decay_vec(g, T_i, offset=o_i-o_j)`` (g^{d+r}) and
#  ``eta = decay_vec(g, T_j, sign=-1)`` (g^{-c}).  The RetNet
#  inter-chunk factorization: the cross term is an outer product, so
#  the decayed product over it distributes —
#      (s (.) xi*eta^T) @ v  =  xi_row (.) ((s (.) eta_row) @ v)
#  — row and column scalings, no rank left.
#
#  ``decay_block_cross`` writes the strict-lower (i,j), i>j block of a
#  split decay as the factored spelling; ``om_decayed_cross`` then
#  pushes the outer product through the contraction.  Composed with
#  the chunk splits + ``decay_block_diag``, a decayed product over
#  concat'd chunks becomes the four-block form whose cross term is
#  the carried state — the algebraic shape the inter-chunk
#  recurrence needs (numerics note: ``eta = g^{-c}`` grows like
#  g^{-T_j}; fp64 carries it past T~2800, fp32 past ~830 — bounded
#  by CHUNK size, not total T, which is why chunked processing
#  exists).
# ---------------------------------------------------------------------------

op_def(
    "decay_vec",
    1,
    1,
    law="decay-vector generator: decay_vec(g, t=T, offset=o, sign=s)"
    "[...,k] = g^(s*k + o), k in [0,T) — g's dims leading batch dims."
    "  The rank-1 factor of the causal decay's cross blocks (the"
    " RetNet xi/eta vectors); decay_vec(g, T) is decay(g,T)'s"
    " diagonal.",
)


def _cross_attrs(bound: dict) -> tuple | None:
    """Read ``(SZ, I, J)`` of a strict-lower double-split — ``i > j``.

    Same ``split`` vocabulary as the diagonal: the sizes tuple
    partitions the matrix extent, ``I`` the row-block and ``J`` the
    column-block index.  ``j < i`` selects the block BELOW the
    diagonal — ``i <= j`` is either the diagonal itself
    (``decay_block_diag``) or the provably-zero strict upper.
    """
    sz, t = bound.get("$attr:SZ"), bound.get("$attr:T")
    i, j = bound.get("$attr:I"), bound.get("$attr:J")
    if not (
        isinstance(sz, (tuple, list))
        and all(isinstance(x, int) for x in sz)
        and isinstance(i, int)
        and isinstance(j, int)
        and 0 <= j < i < len(sz)
        and isinstance(t, int)
        and t == sum(sz)
    ):
        return None
    return sz, i, j


def _check_decay_block_cross(bound: dict) -> bool:
    """Strict-lower block, decay-aligned axes — see ``_cross_attrs``."""
    if _cross_attrs(bound) is None:
        return False
    dr, dc = bound.get("$attr:DR"), bound.get("$attr:DC")
    if not (isinstance(dr, int) and isinstance(dc, int)):
        return False
    gs = _vshape(bound.get("g"))
    if not isinstance(gs, tuple):
        return dr == -2 and dc == -1
    r = len(gs) + 2
    return dr % r == r - 2 and dc % r == r - 1


#: The ``_cross_attrs`` contract as data: ``SZ`` an int partition of
#: ``T``, ``0 <= J < I < len(SZ)`` — ``TI``/``TJ`` the two block
#: extents, ``OD`` the offset ``sum(sz[:i]) - sum(sz[:j])``.
_CROSS_COND = (
    "and",
    ("attr-ints", "SZ"),
    ("attr-type", "I", "int"),
    ("attr-type", "J", "int"),
    ("attr-type", "T", "int"),
    ("attr-cmp", 0, "<=", "J"),
    ("attr-cmp", "J", "<", "I"),
    ("attr-cmp", "I", "<", ("len", ("attr", "SZ"))),
    ("attr-cmp", "T", "==", ("sum", ("attr", "SZ"))),
)
_DSPEC_DECAY_BLOCK_CROSS = {
    "TI": (
        "guard",
        _CROSS_COND,
        ("nth", ("attr", "SZ"), ("attr", "I")),
    ),
    "TJ": (
        "guard",
        _CROSS_COND,
        ("nth", ("attr", "SZ"), ("attr", "J")),
    ),
    "OD": (
        "guard",
        _CROSS_COND,
        (
            "sub",
            ("sum", ("take", ("attr", "SZ"), ("attr", "I"))),
            ("sum", ("take", ("attr", "SZ"), ("attr", "J"))),
        ),
    ),
}


DECAY_BLOCK_CROSS = R(
    "decay_block_cross",
    Op.make(
        "split",
        Op.make(
            "split",
            Op.make("decay", "g", t="T"),
            sizes="SZ",
            dim="DR",
            index="I",
        ),
        sizes="SZ",
        dim="DC",
        index="J",
    ),
    Op.make(
        "mul",
        Op.make(
            "unsqueeze",
            Op.make("decay_vec", "g", t="TI", offset="OD"),
            dim=-1,
        ),
        Op.make(
            "unsqueeze",
            Op.make("decay_vec", "g", t="TJ", sign=-1),
            dim=-2,
        ),
    ),
    check=_check_decay_block_cross,
    dspec=_DSPEC_DECAY_BLOCK_CROSS,
    law="the strict-lower (i,j) block of the causal decay is rank-1: "
    "X[r,c] = g^(o_i-o_j+r-c) = g^(o_i-o_j+r) * g^-c — an outer "
    "product of two decay vectors (the RetNet xi/eta factorization), "
    "so the cross-chunk term needs no materialised block.",
)


def _partition_of(sz, t) -> tuple | None:
    """``sz`` as a positive-int partition of ``t``, or ``None``."""
    if not (
        isinstance(sz, (tuple, list))
        and all(isinstance(x, int) and x > 0 for x in sz)
        and isinstance(t, int)
        and sum(sz) == t
    ):
        return None
    return tuple(sz)


def _cross_flat_attrs(bound: dict) -> tuple | None:
    """Read ``(SZR, SZC, I, J)`` of a mixed-partition strict-lower.

    The recursive bisection leaves the row and column splits under
    DIFFERENT partitions (``(16,8,8)`` rows against ``(16,16)`` cols
    is the observed case), so block (I,J) is strict-lower iff its
    row range starts where the column range ends:
    ``sum(SZR[:I]) >= sum(SZC[:J+1])``.  For equal partitions that is
    exactly ``J < I``.
    """
    t = bound.get("$attr:T")
    szr = _partition_of(bound.get("$attr:SZR"), t)
    szc = _partition_of(bound.get("$attr:SZC"), t)
    i, j = bound.get("$attr:I"), bound.get("$attr:J")
    if not (
        szr is not None
        and szc is not None
        and isinstance(i, int)
        and isinstance(j, int)
        and 0 <= i < len(szr)
        and 0 <= j < len(szc)
    ):
        return None
    if sum(szr[:i]) < sum(szc[: j + 1]):
        return None
    return (szr, szc, i, j)


def _check_decay_block_cross_flat(bound: dict) -> bool:
    """Strict-lower block, mixed partitions, decay-aligned axes."""
    if _cross_flat_attrs(bound) is None:
        return False
    dr, dc = bound.get("$attr:DR"), bound.get("$attr:DC")
    if not (isinstance(dr, int) and isinstance(dc, int)):
        return False
    gs = _vshape(bound.get("g"))
    if not isinstance(gs, tuple):
        return dr == -2 and dc == -1
    r = len(gs) + 2
    return dr % r == r - 2 and dc % r == r - 1


def _derive_decay_block_cross_flat(bound: dict) -> dict | None:
    """``TI``/``TJ``/``OD`` from the two partitions' ranges."""
    attrs = _cross_flat_attrs(bound)
    if attrs is None:
        return None
    szr, szc, i, j = attrs
    return {
        "$attr:TI": szr[i],
        "$attr:TJ": szc[j],
        "$attr:OD": sum(szr[:i]) - sum(szc[:j]),
    }


DECAY_BLOCK_CROSS_FLAT = R(
    "decay_block_cross_flat",
    Op.make(
        "split",
        Op.make(
            "split",
            Op.make("decay", "g", t="T"),
            sizes="SZR",
            dim="DR",
            index="I",
        ),
        sizes="SZC",
        dim="DC",
        index="J",
    ),
    Op.make(
        "mul",
        Op.make(
            "unsqueeze",
            Op.make("decay_vec", "g", t="TI", offset="OD"),
            dim=-1,
        ),
        Op.make(
            "unsqueeze",
            Op.make("decay_vec", "g", t="TJ", sign=-1),
            dim=-2,
        ),
    ),
    check=_check_decay_block_cross_flat,
    derive=_derive_decay_block_cross_flat,
    law="the strict-lower block of the causal decay under MIXED "
    "partitions is still rank-1: X[r,c] = g^(o_r-o_c+r-c) factors "
    "into decay_vec(g, TI, offset=o_r-o_c) ⊗ decay_vec(g, TJ, "
    "sign=-1) for any row/column partitions whose block is "
    "provably below the diagonal — the flattened bisection's "
    "(16,8,8)-vs-(16,16) sites the strict law cannot see.",
    derivation=("decay_block_cross",),
)


def _check_om_decayed_cross(bound: dict) -> bool:
    """Require the factored-decay distribution to be well-typed.

    ``unsq(xi, -1)`` broadcasts against the mul's row axis and
    ``unsq(eta, -2)`` against its column axis; the rearranged rhs
    must broadcast identically and the inner matmul must contract.
    """
    ss, xs, es, vs = (
        _vshape(bound.get("s")),
        _vshape(bound.get("xi")),
        _vshape(bound.get("eta")),
        _vshape(bound.get("v")),
    )
    if not all(isinstance(x, tuple) for x in (ss, xs, es, vs)):
        return False
    if not (es and xs and len(ss) >= 1):
        return False
    from mlopt_core.typing import (
        INVALID,
        broadcast,
        matmul_shape,
    )

    # s ⊙ (1,…,1,TJ) must broadcast (eta on the column axis)
    eta_b = (*((1,) * max(len(es) - 1, 0)), 1, es[-1])
    m = broadcast(ss, eta_b)
    if m is INVALID:
        return False
    n = matmul_shape(m, vs)
    if n is None:
        return False
    # (TI,…,1) broadcasting the row axis of the product
    xi_b = (*((1,) * max(len(xs) - 1, 0)), xs[-1], 1)
    return broadcast(n, xi_b) is not INVALID


OM_DECAYED_CROSS = R(
    "om_decayed_cross",
    Op.make(
        "om_decayed",
        "s",
        Op.make(
            "mul",
            Op.make("unsqueeze", "xi", dim="UX"),
            Op.make("unsqueeze", "eta", dim="UE"),
        ),
        "v",
    ),
    Op.make(
        "mul",
        Op.make("unsqueeze", "xi", dim="UX"),
        Op.make(
            "matmul",
            Op.make("mul", "s", Op.make("unsqueeze", "eta", dim="UE")),
            "v",
        ),
    ),
    check=_check_om_decayed_cross,
    law="a decayed product over a rank-1 decay distributes into "
    "row/column scalings: (s (.) xi*eta^T) @ v = xi_row (.) ((s (.) "
    "eta_row) @ v) — the outer product never multiplies a matrix, "
    "the factored form is what the inter-chunk recurrence carries.",
)


def _check_om_decayed_cross_gen(bound: dict) -> bool:
    """Require the generalized factored-decay distribution to type.

    ``x`` absorbs the residual row-side decay factors, so it must be
    constant along the contracted (column) axis: right-aligned to the
    decay product's broadcast shape its last dim is 1.  That is what
    makes ``(s (.) x (.) e) @ v = x (.) ((s (.) e) @ v)`` exact — a
    row scaling factors out of the contraction pointwise.  ``e =
    unsq(eta, UE)`` stays inside, so it need only broadcast against
    ``s``; the inner matmul and the lhs product must both type.
    """
    ss, xs, es, vs = (
        _vshape(bound.get("s")),
        _vshape(bound.get("x")),
        _vshape(bound.get("eta")),
        _vshape(bound.get("v")),
    )
    if not all(isinstance(t, tuple) for t in (ss, xs, es, vs)):
        return False
    ue = bound.get("$attr:UE", -2)
    if not ss or not isinstance(ue, int):
        return False
    from mlopt_core.typing import (
        INVALID,
        broadcast,
        matmul_shape,
    )

    # unsq(eta, UE): a broadcast axis inserted into es at UE
    ue_n = ue % (len(es) + 1)
    eta_b = (*es[:ue_n], 1, *es[ue_n:])
    d = broadcast(xs, eta_b)
    if d is INVALID:
        return False
    p = broadcast(ss, d)
    q = broadcast(ss, eta_b)
    if p is INVALID or q is INVALID:
        return False
    # x constant along p's last (contracted) axis — right-aligned,
    # its last dim must broadcast as 1 (scalar () qualifies)
    xp = (1,) * max(len(p) - len(xs), 0) + tuple(xs)
    if xp[-1] != 1:
        return False
    m = matmul_shape(q, vs)
    if m is None or matmul_shape(p, vs) is None:
        return False
    return broadcast(m, xs) is not INVALID


OM_DECAYED_CROSS_GEN = R(
    "om_decayed_cross_gen",
    Op.make(
        "om_decayed",
        "s",
        Op.make(
            "mul",
            "x",
            Op.make("unsqueeze", "eta", dim="UE"),
        ),
        "v",
    ),
    Op.make(
        "mul",
        "x",
        Op.make(
            "matmul",
            Op.make("mul", "s", Op.make("unsqueeze", "eta", dim="UE")),
            "v",
        ),
    ),
    check=_check_om_decayed_cross_gen,
    law="the factored-decay distribution generalized over the row "
    "side: (s (.) x*e) @ v = x (.) ((s (.) e) @ v) for any "
    "row-constant x — the e-graph's decay chains arrive as deeper "
    "unsqueeze-products (every residual factor a row broadcaster), "
    "not the strict mul(unsq(xi), unsq(eta)) spelling, so the "
    "metavar absorbs them and the xi/eta split still happens.",
)


def _check_matmul_col_decay(bound: dict) -> bool:
    """Column-scaling must broadcast and the contraction must type.

    ``eta`` unsqueezed at ``-2`` becomes a row vector broadcasting
    over the product's columns — its last axis must match ``k``'s
    row axis so the scale lands where the transpose moves it.  The
    lhs's mul and the rhs's inner contraction must both type.
    """
    qs, ks, es = (
        _vshape(bound.get("q")),
        _vshape(bound.get("k")),
        _vshape(bound.get("eta")),
    )
    if not all(isinstance(x, tuple) for x in (qs, ks, es)):
        return False
    if len(ks) < 2 or not es:
        return False
    from mlopt_core.typing import (
        INVALID,
        broadcast,
        matmul_shape,
    )

    # k ⊙ (…,T_j,1) broadcasts eta over k's rows
    eta_k = (*((1,) * max(len(es) - 1, 0)), es[-1], 1)
    k_eta = broadcast(ks, eta_k)
    if k_eta is INVALID:
        return False
    kt = list(k_eta)
    kt[-2], kt[-1] = kt[-1], kt[-2]
    m = matmul_shape(qs, tuple(kt))
    if m is None:
        return False
    eta_s = (*((1,) * max(len(es) - 1, 0)), 1, es[-1])
    return broadcast(m, eta_s) is not INVALID


MATMUL_COL_DECAY = R(
    "matmul_col_decay",
    Op.make(
        "mul",
        Op.make(
            "matmul",
            "q",
            Op.make("transpose", "k", dim0="D0", dim1="D1"),
        ),
        Op.make("unsqueeze", "eta", dim="UE"),
    ),
    Op.make(
        "matmul",
        "q",
        Op.make(
            "transpose",
            Op.make("mul", "k", Op.make("unsqueeze", "eta", dim=-1)),
            dim0="D0",
            dim1="D1",
        ),
    ),
    check=_check_matmul_col_decay,
    law="column-scaling a q·kᵀ product by eta pulls inside the "
    "contraction as row-scaling the transposed factor: (q·kᵀ) ⊙ etaᵀ "
    "= q · (k ⊙ eta)ᵀ — the decay's eta column lands on the keys, "
    "which is what turns the cross term into q @ (carried state).",
)


def _check_decay_vec_offset(bound: dict) -> bool:
    """Fire only on a non-trivial offset."""
    o = bound.get("$attr:OD")
    return isinstance(o, int) and o != 0


DECAY_VEC_OFFSET = R(
    "decay_vec_offset",
    Op.make("decay_vec", "g", t="T", offset="OD"),
    Op.make(
        "mul",
        Op.make("decay_vec", "g", t=1, offset="OD"),
        Op.make("decay_vec", "g", t="T"),
    ),
    check=_check_decay_vec_offset,
    law="decay_vec(g, t, offset=o) = g^o ⊙ decay_vec(g, t) — the "
    "cross-block vector splits into a per-BLOCK scalar (a length-1 "
    "decay_vec carrying the offset) and the per-position base g^r, "
    "which is what exposes the per-chunk weights the inter-chunk "
    "fold carries.  The scalar stays in-generator: no Const minting.",
)


def _const_along_contraction(cs: tuple, bs: tuple) -> bool:
    """``c`` constant along ``b``'s contracted axis (``-2``).

    Right-aligned to ``b``, ``c``'s dim at ``-2`` must be 1 — the
    exactness condition of ``c ⊙ (q@B) = q @ (c⊙B)``: the lhs scales
    the product's output rows ``r``, the rhs scales ``B``'s
    contracted rows ``k``.  Scalars and per-column scales qualify;
    a ``(T,1)`` row vector does not.
    """
    if len(bs) < 2 or len(cs) > len(bs):
        return False
    ca = (1,) * (len(bs) - len(cs)) + tuple(cs)
    return ca[-2] == 1


def _check_matmul_scalar(bound: dict) -> bool:
    """Require the scalar to broadcast to the right operand.

    ``c ⊙ (q@B) = q @ (c⊙B)`` is exact only when ``c`` is constant
    along the contracted axis: the lhs scales the product's *output*
    rows ``r`` while the rhs scales ``B``'s *contracted* rows ``k``.
    A ``c`` varying there (a ``(T,1)`` row-decay vector is the
    observed case) gives ``c[r]·(q@B) ≠ q@(c[k]⊙B)`` — the guard
    right-aligns ``c`` to ``b`` and requires dim ``-2`` to be 1.
    """
    cs, bs = _vshape(bound.get("c")), _vshape(bound.get("b"))
    qs = _vshape(bound.get("q"))
    if not all(isinstance(x, tuple) for x in (cs, bs, qs)):
        return False
    if len(bs) < 2:
        return False
    from mlopt_core.typing import (
        INVALID,
        broadcast,
        matmul_shape,
    )

    m = matmul_shape(qs, bs)
    if m is None:
        return False
    # c must broadcast identically to BOTH m and b — the scalar
    # (or leading-broadcastable) case only.
    if broadcast(cs, bs) is INVALID:
        return False
    if broadcast(cs, m) is INVALID:
        return False
    cb = broadcast(cs, bs)
    if len(cb) > len(bs):
        return False
    # contraction-axis constancy: right-aligned to b, dim -2 is 1 —
    # scalars and per-column scales qualify, row vectors do not.
    return _const_along_contraction(cs, bs)


MATMUL_SCALAR_FACTOR = R(
    "matmul_scalar_factor",
    Op.make("mul", "c", Op.make("matmul", "q", "b")),
    Op.make("matmul", "q", Op.make("mul", "c", "b")),
    check=_check_matmul_scalar,
    law="a broadcast scale factors through the right operand of a "
    "matmul: c ⊙ (q·B) = q · (c ⊙ B) — the per-chunk decay weight "
    "g^o pulls off the cross contribution and onto the state term, "
    "which is the spelling the affine-scan lifts fold.",
)


def _check_state_weight_sink(bound: dict) -> bool:
    """Require ``a`` constant along the contracted axis of ``B``.

    ``a⊙b ⊙ (q@B) = b ⊙ (q @ (a⊙B))`` is exact iff ``a`` does not
    vary along the contraction: on the lhs ``a`` scales the product's
    output rows ``r``, on the rhs it scales ``B``'s contracted rows
    ``k`` — equal only when ``a``'s right-aligned dim ``-2`` is 1
    (scalars and per-column/batch scales qualify; a row-decay vector
    does not — same condition ``matmul_scalar_factor`` enforces).
    ``b`` is the residual row-side factor and need only broadcast
    over the rearranged product.
    """
    ss, xs, qs, bs = (
        _vshape(bound.get("a")),
        _vshape(bound.get("b")),
        _vshape(bound.get("q")),
        _vshape(bound.get("B")),
    )
    if not all(isinstance(t, tuple) for t in (ss, xs, qs, bs)):
        return False
    if not _const_along_contraction(ss, bs):
        return False
    from mlopt_core.typing import (
        INVALID,
        broadcast,
        matmul_shape,
    )

    ab = broadcast(ss, bs)
    if ab is INVALID:
        return False
    m = matmul_shape(qs, ab)
    if m is None:
        return False
    return broadcast(xs, m) is not INVALID


DECAY_STATE_WEIGHT_SINK = R(
    "decay_state_weight_sink",
    Op.make(
        "mul",
        Op.make("mul", "a", "b"),
        Op.make("matmul", "q", "B"),
    ),
    Op.make(
        "mul",
        "b",
        Op.make("matmul", "q", Op.make("mul", "a", "B")),
    ),
    check=_check_state_weight_sink,
    law="of a two-factor row scale on a matmul product, the "
    "contraction-constant factor sinks into the state: "
    "(a⊙b) ⊙ (q·B) = b ⊙ (q · (a⊙B)) when a is constant along B's "
    "rows — the per-block decay weight g^o detaches from the shared "
    "row vector g^r and lands on the state term, the weighted-sum "
    "spelling the inter-chunk accumulator folds.",
    derivation=("matmul_scalar_factor",),
)


def _check_state_weight_exit(bound: dict) -> bool:
    """Require ``a`` constant along the contracted axis of ``B``.

    The exactness condition is the sink's, read backward: ``a`` may
    only scale ``B``'s columns (right-aligned dim ``-2`` is 1), else
    the exit ``mul(a, q@B)`` would re-index the weight onto the
    output rows.
    """
    ss, qs, bs = (
        _vshape(bound.get("a")),
        _vshape(bound.get("q")),
        _vshape(bound.get("B")),
    )
    if not all(isinstance(t, tuple) for t in (ss, qs, bs)):
        return False
    if not _const_along_contraction(ss, bs):
        return False
    from mlopt_core.typing import (
        INVALID,
        broadcast,
        matmul_shape,
    )

    ab = broadcast(ss, bs)
    if ab is INVALID:
        return False
    m = matmul_shape(qs, ab)
    if m is None:
        return False
    return broadcast(ss, m) is not INVALID


DECAY_STATE_WEIGHT_EXIT = R(
    "decay_state_weight_exit",
    Op.make("matmul", "q", Op.make("mul", "a", "B")),
    Op.make("mul", "a", Op.make("matmul", "q", "B")),
    check=_check_state_weight_exit,
    law="a contraction-constant scale exits the state term: "
    "q · (a⊙B) = a ⊙ (q·B) when a is constant along B's rows — the "
    "per-block decay weight leaves mul(g^o, S_j), the addend "
    "spelling decay_state_lift's lhs folds onto applyd.",
    derivation=("matmul_scalar_factor",),
)


def _check_unsq_mul(bound: dict) -> bool:
    """``unsq(mul(a,b), d)`` splits when ``d`` is valid for both.

    The inserted axis is a size-1 broadcast pad in every operand —
    pushing the view inside the mul is exact provided ``d`` indexes
    each operand's (rank+1) window.  Unshapeable operands get the
    conservative -1..len window on the concrete side.
    """
    d = bound.get("$attr:UD")
    if not isinstance(d, int):
        return False
    a, b = _vshape(bound.get("a")), _vshape(bound.get("b"))
    for s in (a, b):
        if isinstance(s, tuple):
            if not (-len(s) - 1 <= d <= len(s)):
                return False
        elif s is not None:
            return False
    return True


UNSQ_MUL = R(
    "unsq_mul",
    Op.make(
        "unsqueeze",
        Op.make("mul", "a", "b"),
        dim="UD",
    ),
    Op.make(
        "mul",
        Op.make("unsqueeze", "a", dim="UD"),
        Op.make("unsqueeze", "b", dim="UD"),
    ),
    check=_check_unsq_mul,
    law="unsqueeze distributes over elementwise mul: the inserted "
    "axis is a size-1 broadcast pad in every operand, so pushing the "
    "view inside is exact — the per-block scalar (pow g, o) escapes "
    "its row-vector packaging and reaches the state-weight position.",
)


def _decay_state_like(bound: dict) -> bool:
    """``h`` must be a matrix state, ``a`` a scalar-ish weight.

    The diagonal ``affd_lift``'s state condition enumerates the
    *ops* a state may be spelled as (add/sub/apply/applyd/leaf) —
    right for the SSM alphabet where contributions are vectors.
    The decay-state alphabet's contributions are ``om_decayed`` /
    ``matmul`` products — op-enumeration would grow per carrier, so
    this checks the *shape* instead: ``h`` carries a ≥2-D state and
    ``a`` broadcasts onto it as a scalar or lower-rank weight.
    """
    hs, a_s = _vshape(bound.get("h")), _vshape(bound.get("a"))
    if not (isinstance(hs, tuple) and len(hs) >= 2):
        return False
    if a_s is None:
        return True
    if not isinstance(a_s, tuple):
        return False
    from mlopt_core.typing import INVALID, broadcast

    b = broadcast(a_s, hs)
    return b is not INVALID and len(a_s) <= len(hs)


def _derive_chunkify(bound: dict) -> dict | None:
    """``SZ`` := the row/key bisection ``(T//2, T−T//2)``.

    The split sizes come from the *score's* extent — the fused
    ``om_decayed`` carries no partition, so the introduction halves
    the axis; deeper chunking is saturation recursion, not a bigger
    law.

    Stays procedural: reads ``_vshape`` — the carrier-aware value
    shape; ``("dim", "s", -2)`` in a dspec would read core
    ``_shape_of``'s carrier-convention shape instead.
    """
    s = _vshape(bound.get("s"))
    if not (isinstance(s, tuple) and len(s) >= 2):
        return None
    t = s[-2]
    if not isinstance(t, int) or t < 2:
        return None
    return {"$attr:SZ": (t // 2, t - t // 2)}


def _check_chunkify_rows(bound: dict) -> bool:
    """Score and decay must cover the row axis at the same extent."""
    if _derive_chunkify(bound) is None:
        return False
    d = _vshape(bound.get("d"))
    s = _vshape(bound.get("s"))
    if not (isinstance(d, tuple) and len(d) >= 2):
        return False
    return d[-2] == s[-2]


def _derive_chunkify_keys(bound: dict) -> dict | None:
    """``SZ`` := the key-axis bisection — same halving on ``s[-1]``.

    Stays procedural: reads ``_vshape`` — see ``_derive_chunkify``.
    """
    s = _vshape(bound.get("s"))
    if not (isinstance(s, tuple) and len(s) >= 2):
        return None
    t = s[-1]
    if not isinstance(t, int) or t < 2:
        return None
    return {"$attr:SZ": (t // 2, t - t // 2)}


def _check_chunkify_keys(bound: dict) -> bool:
    """Score/decay cols and value rows must share the key extent."""
    s, d, v = (
        _vshape(bound.get("s")),
        _vshape(bound.get("d")),
        _vshape(bound.get("v")),
    )
    if not (
        isinstance(s, tuple)
        and isinstance(d, tuple)
        and isinstance(v, tuple)
        and len(s) >= 2
        and len(d) >= 2
        and len(v) >= 2
    ):
        return False
    kk = s[-1]
    if not isinstance(kk, int) or kk < 2:
        return False
    return d[-1] == kk and v[-2] == kk


OM_DECAYED_CHUNKIFY_ROWS = R(
    "om_decayed_chunkify_rows",
    Op.make("om_decayed", "s", "d", "v"),
    Op.make(
        "concat",
        Op.make(
            "om_decayed",
            Op.make("split", "s", sizes="SZ", dim=-2, index=0),
            Op.make("split", "d", sizes="SZ", dim=-2, index=0),
            "v",
        ),
        Op.make(
            "om_decayed",
            Op.make("split", "s", sizes="SZ", dim=-2, index=1),
            Op.make("split", "d", sizes="SZ", dim=-2, index=1),
            "v",
        ),
        dim=-2,
    ),
    check=_check_chunkify_rows,
    derive=_derive_chunkify,
    law="fused decayed product bisects along the row axis: "
    "om_decayed(s,d,v) = cat(om_decayed(s_hi,d_hi,v), "
    "om_decayed(s_lo,d_lo,v)) — the introduction direction the "
    "concat-side chunk laws can't reach (a real export carries no "
    "concat).  Halving is the data-free split; recursion deepens it.",
)


OM_DECAYED_CHUNKIFY_KEYS = R(
    "om_decayed_chunkify_keys",
    Op.make("om_decayed", "s", "d", "v"),
    Op.make(
        "add",
        Op.make(
            "om_decayed",
            Op.make("split", "s", sizes="SZ", dim=-1, index=0),
            Op.make("split", "d", sizes="SZ", dim=-1, index=0),
            Op.make("split", "v", sizes="SZ", dim=-2, index=0),
        ),
        Op.make(
            "om_decayed",
            Op.make("split", "s", sizes="SZ", dim=-1, index=1),
            Op.make("split", "d", sizes="SZ", dim=-1, index=1),
            Op.make("split", "v", sizes="SZ", dim=-2, index=1),
        ),
    ),
    check=_check_chunkify_keys,
    derive=_derive_chunkify_keys,
    law="fused decayed product bisects along the key axis: the "
    "contraction splits s/d's columns and v's rows into an add of "
    "two partial blocks — exact (block-column decomposition of a "
    "matmul), the shape the cross term needs.",
)


def _check_om_decayed_kway(k: int, bound: dict) -> bool:
    """Require a uniform K-partition of a fused decayed product.

    ``T`` must be a positive int divisible by ``K`` (the partition is
    ``(C,)*K`` with ``C = T//K``), the score plane square at ``T`` and
    the value rows at ``T`` — the causal block decomposition only
    types on the shared extent.
    """
    t = bound.get("$attr:T")
    if not isinstance(t, int) or t < k or t % k != 0:
        return False
    ss, vs = _vshape(bound.get("s")), _vshape(bound.get("v"))
    if not (isinstance(ss, tuple) and isinstance(vs, tuple)):
        return False
    if len(ss) < 2 or len(vs) < 2:
        return False
    return ss[-2] == t and ss[-1] == t and vs[-2] == t


def _derive_om_decayed_kway(k: int, bound: dict) -> dict | None:
    """``C``, ``SZ`` and the strict-lower blocks' decay offsets.

    ``C := T // K`` and ``SZ := (C,)*K``; block ``(i, j)`` sits
    ``(i-j)·C`` rows below the diagonal, so its row-decay vector's
    offset is ``OFF_{i}_{j} := (i-j)·C`` (see ``decay_block_cross``).
    """
    if not _check_om_decayed_kway(k, bound):
        return None
    c = bound["$attr:T"] // k
    out: dict = {"$attr:C": c, "$attr:SZ": (c,) * k}
    for i in range(k):
        for j in range(i):
            out[f"$attr:OFF_{i}_{j}"] = (i - j) * c
    return out


def _kway_score(i: int, j: int):
    """Score block ``(i, j)`` — split the rows, then the columns."""
    return Op.make(
        "split",
        Op.make("split", "s", sizes="SZ", dim=-2, index=i),
        sizes="SZ",
        dim=-1,
        index=j,
    )


def _kway_value(j: int):
    """Value block ``j`` — rows ``[jC:(j+1)C)`` of ``v``."""
    return Op.make("split", "v", sizes="SZ", dim=-2, index=j)


def _kway_diag(i: int):
    """Intra-chunk term: the ``(i, i)`` block over ``decay(γ, C)``.

    The diagonal block of ``decay(γ, T)`` is literally ``decay(γ, C)``
    — equal row and column offsets cancel — so the law regenerates
    the small generator rather than slicing the big one.
    """
    return Op.make(
        "om_decayed",
        _kway_score(i, i),
        Op.make("decay", "g", t="C"),
        _kway_value(i),
    )


def _kway_cross(i: int, j: int):
    """Strict-lower term: the factored rank-1 cross contribution.

    ``(s_ij ⊙ D_ij) @ v_j`` with ``D_ij`` the ``ξ_i·η_jᵀ`` outer
    product ``decay_vec(g, C, offset=(i-j)C) ⊗ decay_vec(g, C, -1)``.
    """
    return Op.make(
        "mul",
        Op.make(
            "unsqueeze",
            Op.make("decay_vec", "g", t="C", offset=f"OFF_{i}_{j}"),
            dim=-1,
        ),
        Op.make(
            "matmul",
            Op.make(
                "mul",
                _kway_score(i, j),
                Op.make(
                    "unsqueeze",
                    Op.make("decay_vec", "g", t="C", sign=-1),
                    dim=-2,
                ),
            ),
            _kway_value(j),
        ),
    )


def _kway_block(i: int):
    """Row block ``i``: the intra term plus its strict-lower crosses."""
    terms = [_kway_diag(i)]
    terms.extend(_kway_cross(i, j) for j in range(i))
    acc = terms[0]
    for term in terms[1:]:
        acc = Op.make("add", acc, term)
    return acc


def _kway_rhs(k: int):
    """``concat_i(block_i)`` — the K-way causal block decomposition."""
    blocks = [_kway_block(i) for i in range(k)]
    out = blocks[-1]
    for block in reversed(blocks[:-1]):
        out = Op.make("concat", block, out, dim=-2)
    return out


def om_decayed_kway_blocks(k: int) -> Rewrite:
    """K-way causal block decomposition of a fused decayed product.

    ``om_decayed(s, decay(γ,T), v)`` → ``concat_i(intra_i + Σ_{j<i}
    cross_ij)`` for the uniform ``(C,)*K`` partition (``C = T/K``):
    the diagonal blocks regenerate ``decay(γ, C)``, each strict-lower
    block factors as the rank-1 ``ξ_i·η_jᵀ`` outer product.

    ONE application reaches ``K`` chunks.  The pairwise bisection
    needs a split per level (and a budget per split), and the
    e-graph's dirty-frontier rescan makes each deeper level costlier
    — so the inter-chunk fold only ever sees one prior block per row
    (measured: depth 2 at 4 chunks, unchanged from 60 to 120 budget
    units).  The K-way form hands the fold laws all ``K-1`` prior
    blocks at once, the material the ``applyd`` chain needs.

    Opt-in and budgeted like ``OM_CHUNKIFY``: the law is a closure
    generator (its diagonal blocks re-match the lhs), so it stays out
    of ``OM_LAWS``.  ``k`` is validated eagerly so a bad arity fails
    at construction, not at saturation.
    """
    if not isinstance(k, int) or k < 2:
        raise ValueError(f"k must be an int >= 2, got {k!r}")

    def check(bound: dict) -> bool:
        return _check_om_decayed_kway(k, bound)

    def derive(bound: dict) -> dict | None:
        return _derive_om_decayed_kway(k, bound)

    return R(
        f"om_decayed_kway_blocks_{k}",
        Op.make(
            "om_decayed",
            "s",
            Op.make("decay", "g", t="T"),
            "v",
        ),
        _kway_rhs(k),
        check=check,
        derive=derive,
        law="the causal decay's block decomposition in one step: "
        "om_decayed(s, decay(\u03b3,T), v) = cat_i(intra_i + Σ_{j<i} "
        "ξ_i·η_jᵀ⊙(s_ij v_j)) for the uniform (C,)*K partition — "
        "the K-way split the fold needs, so a row block sees K-1 "
        "prior chunks instead of the bisection's one.",
    )


#: Opt-in K-way chunk laws — ``OM_LAWS + om_kway_laws(k)``.  Like
#: ``OM_CHUNKIFY`` these CREATE structure (a K-way expansion is a
#: closure generator), so they stay out of the everyday set and are
#: priced with ``rule_budgets``.
def om_kway_laws(k: int) -> list[Rewrite]:
    """Return the K-way causal block law as a one-element list."""
    return [om_decayed_kway_blocks(k)]


# ---------------------------------------------------------------------------
#  The shared-state reassociation — the sequential carrier's missing move
# ---------------------------------------------------------------------------
#  A fold is a *cost-neutral relabelling*: ``applyd(aff_diag(a,x),h)``
#  prices exactly like the ``add(mul(a,h),x)`` it re-spells, so a
#  fold-only member can never win an extraction.  The O(T·C) win needs
#  the state to be *shared* across row blocks, and the inter-chunk
#  cross sum is spelled as a FLAT, left-nested add —
#  ``Σ_{j<i} γ^{(i-j)C} b_j`` — so every row block re-spells its own
#  copy of the weighted contributions and the affine-scan lifts fold a
#  TREE, never a recurrence.
#
#  The move that makes the sum shared is the geometric reassociation
#
#      W1⊙X + W2⊙Y  =  W2 ⊙ (γ^{O1−O2} ⊙ X + Y)
#
#  for two weights from the SAME decay base at geometric-neighbour
#  offsets ``O1 > O2``.  ``decay_vec(g, t, O)`` is ``γ^{O+r}`` on its
#  axis, so ``W1 = γ^{O1−O2} ⊙ W2`` for the SAME ``g``/``t`` and the
#  identity is EXACT (not a broadcast approximation).  Applied
#  bottom-up over the flat sum it produces the nested
#
#      T_k = γ^C ⊙ T_{k−1} + b_{k−1}
#
#  whose inner ``add(mul(γ^C, ·), ·)`` is exactly the spelling
#  ``decay_state_lift`` folds onto ``applyd`` — and, crucially, is ONE
#  e-class consumed by row block ``k``'s output and by block ``k+1``'s
#  step, so the state is *shared*.  That is the recurrence the e-graph
#  otherwise never reaches.
#
#  The block sum carries its intra-chunk term as the FIRST addend
#  (``add(add(add(intra, c₀), c₁), …)``), so the cross terms are never
#  directly adjacent: the ``_PREFIX`` variant matches the two-level
#  nesting ``add(add(P, mul(W1,X)), mul(W2,Y))`` and leaves ``P``
#  untouched, which is what actually fires on a real export.  The
#  ``MAT`` variant additionally pins the shared left operand ``Q`` of
#  a contraction pair, so the reassociated state is ``Q @ (…)`` with
#  the q-free inner sum shared across blocks.
#
#  OPT-IN, like ``OM_CHUNKIFY`` / ``om_kway_laws``: the reassociation
#  ranges over the whole cross-sum space (a closure generator), so it
#  stays out of ``OM_LAWS`` and is priced with ``rule_budgets``.
# ---------------------------------------------------------------------------


def _shared_weight(offset: str, t: str | int):
    """``unsqueeze(decay_vec(g, t, offset), dim=D)`` — a decay row weight.

    ``g`` (the decay base) and ``D`` (the inserted broadcast axis) are
    shared metavariables, so the two weights a law matches are forced
    onto the SAME base and the SAME axis — exactly the condition under
    which ``decay_vec(g, t, O1) = γ^{O1−O2} ⊙ decay_vec(g, t, O2)``.
    ``t`` may itself be a metavariable: the identity is indifferent to
    the weight's period (a scalar ``t=1`` and a per-position ``t=C``
    vector telescope identically, both carrying the same ``γ^{O}``
    offset factor).
    """
    return Op.make(
        "unsqueeze",
        Op.make("decay_vec", "g", t=t, offset=offset),
        dim="D",
    )


def _shared_step(bound: dict) -> int | None:
    """Return the geometric step ``O1 − O2``, or ``None`` if not legal.

    The reassociation is exact for ANY two offsets sharing a base, so
    soundness needs only that ``OD`` is derivable.  The *alignment*
    clause (``OD`` a positive divisor of both offsets) is a search
    guard, not a soundness one: it confines the law to
    geometric-weighted cross sums (offsets that are multiples of a
    common chunk step) so the reassociation cannot chain away on
    unrelated sums.  A zero or negative step is rejected outright —
    the RHS would mint a non-advancing weight.
    """
    o1, o2 = bound.get("$attr:O1"), bound.get("$attr:O2")
    if not (isinstance(o1, int) and isinstance(o2, int)):
        return None
    step = o1 - o2
    if step <= 0 or o2 <= 0 or o2 % step or o1 % step:
        return None
    return step


def _check_shared_reassoc(bound: dict) -> bool:
    """Require a legal geometric step and a well-typed summand pair.

    The identity is unconditionally exact once ``OD = O1 − O2``; the
    shape clause only keeps the rule from minting an ill-typed
    ``add``/``mul`` on operands that cannot broadcast.
    """
    if _shared_step(bound) is None:
        return False
    xs, ys = _vshape(bound.get("X")), _vshape(bound.get("Y"))
    if not (isinstance(xs, tuple) and isinstance(ys, tuple)):
        return False
    from mlopt_core.typing import INVALID, broadcast

    return broadcast(xs, ys) is not INVALID


def _derive_shared_reassoc(bound: dict) -> dict | None:
    """``OD`` := ``O1 − O2`` — the geometric step the RHS spells."""
    step = _shared_step(bound)
    if step is None:
        return None
    return {"$attr:OD": step}


SHARED_STATE_REASSOC = R(
    "om_shared_reassoc",
    Op.make(
        "add",
        Op.make("mul", _shared_weight("O1", "T"), "X"),
        Op.make("mul", _shared_weight("O2", "T"), "Y"),
    ),
    Op.make(
        "mul",
        _shared_weight("O2", "T"),
        Op.make(
            "add",
            Op.make("mul", _shared_weight("OD", 1), "X"),
            "Y",
        ),
    ),
    check=_check_shared_reassoc,
    derive=_derive_shared_reassoc,
    law="geometric reassociation of a decay-weighted sum: W1⊙X + W2⊙Y "
    "= W2 ⊙ (gamma^{O1-O2}⊙X + Y) for two weights from one decay base "
    "at geometric-neighbour offsets.  Applied bottom-up it turns the "
    "flat cross sum Σ_{j<i} gamma^{(i-j)C} b_j into the nested T_k = "
    "gamma^C⊙T_{k-1} + b_{k-1}, whose inner add is ONE shared e-class "
    "— the recurrence the affine-scan lifts fold and the O(T·C) "
    "sequential carrier needs.",
)


SHARED_STATE_REASSOC_PREFIX = R(
    "om_shared_reassoc_prefix",
    Op.make(
        "add",
        Op.make(
            "add", "P", Op.make("mul", _shared_weight("O1", "T"), "X")
        ),
        Op.make("mul", _shared_weight("O2", "T"), "Y"),
    ),
    Op.make(
        "add",
        "P",
        Op.make(
            "mul",
            _shared_weight("O2", "T"),
            Op.make(
                "add",
                Op.make("mul", _shared_weight("OD", 1), "X"),
                "Y",
            ),
        ),
    ),
    check=_check_shared_reassoc,
    derive=_derive_shared_reassoc,
    law="the block-nested form of om_shared_reassoc: the cross sum "
    "carries its intra-chunk term P as the first addend, so the two "
    "weighted cross terms are only ever adjacent through "
    "add(add(P, W1⊙X), W2⊙Y).  Leaves P in place and reassociates the "
    "pair — the variant that actually fires on the chunked export and "
    "chains the shared state T_k up the row blocks.",
)


def _check_shared_mat_reassoc(bound: dict) -> bool:
    """Require a legal step, a typed contraction pair and a shared Q.

    ``Q`` is a repeated metavariable, so the matcher already forces the
    two contractions onto one left operand — the condition that makes
    the reassociated state ``Q @ (…)`` shared rather than duplicated.
    The γ^{step} factor is a scalar, hence constant along the
    contraction (``_const_along_contraction``): that is what lets it
    move inside the state, ``Q @ (c⊙A) = c⊙(Q@A)``.
    """
    if _shared_step(bound) is None:
        return False
    qs = _vshape(bound.get("Q"))
    as_, bs = _vshape(bound.get("A")), _vshape(bound.get("B"))
    if not all(isinstance(x, tuple) for x in (qs, as_, bs)):
        return False
    if len(as_) < 2 or len(bs) < 2:
        return False
    if not _const_along_contraction((1, 1), as_):
        return False
    from mlopt_core.typing import matmul_shape

    return (
        matmul_shape(qs, as_) is not None
        and matmul_shape(qs, bs) is not None
    )


SHARED_STATE_MAT_REASSOC = R(
    "om_shared_mat_reassoc",
    Op.make(
        "add",
        Op.make(
            "add",
            "P",
            Op.make(
                "mul",
                _shared_weight("O1", "T"),
                Op.make("matmul", "Q", "A"),
            ),
        ),
        Op.make(
            "mul",
            _shared_weight("O2", "T"),
            Op.make("matmul", "Q", "B"),
        ),
    ),
    Op.make(
        "add",
        "P",
        Op.make(
            "mul",
            _shared_weight("O2", "T"),
            Op.make(
                "matmul",
                "Q",
                Op.make(
                    "add",
                    Op.make("mul", _shared_weight("OD", 1), "A"),
                    "B",
                ),
            ),
        ),
    ),
    check=_check_shared_mat_reassoc,
    derive=_derive_shared_reassoc,
    law="the contraction-factoring form of the reassociation: when two "
    "weighted cross terms share one left operand Q, the geometric step "
    "pulls INSIDE the contraction — W1⊙(Q@A) + W2⊙(Q@B) = W2 ⊙ "
    "(Q @ (gamma^{O1-O2}⊙A + B)).  The reassociated state Q@(…) is now "
    "q-free inside, so the SAME inner add is consumed by two row "
    "blocks — the shared recurrence, not a tree.",
)


#: The shared-state reassociation family — ``OM_LAWS +
#: om_shared_state_laws()``.  Opt-in (a closure generator over the
#: cross-sum space) and priced with ``rule_budgets``.
def om_shared_state_laws() -> list[Rewrite]:
    """Return the shared-state reassociation laws as a list.

    ``om_shared_reassoc`` is the algebra-level move; the ``_prefix``
    and ``_mat`` variants are the spellings the chunked export
    actually presents (intra term first; a shared contraction left
    operand).
    """
    return [
        SHARED_STATE_REASSOC,
        SHARED_STATE_REASSOC_PREFIX,
        SHARED_STATE_MAT_REASSOC,
    ]


# ---------------------------------------------------------------------------
#  The div bridge — the scheduled twin of ``matmul_div_distribute``
# ---------------------------------------------------------------------------
#  The chunked export's score is ``q(x)@k(x)ᵀ/√t``, so every cross
#  contribution reaches the e-graph with the scalar denominator OUTSIDE
#  the contraction — ``div(matmul(q_i, B_j), c)``.  The reassociation's
#  ``_mat`` variant needs the contraction to be the *direct* operand of
#  the weight mul (``mul(W, matmul(Q, B))``), so the denominator has to
#  move inside first:
#
#      div(A@B, c)  =  (A/c) @ B      (c scalar)
#
#  — the naturality ``MATMUL_DIV_DISTRIBUTE`` (``mlopt_core.laws.tensor``)
#  spells in the OPPOSITE direction.  The two are exact inverses, so as
#  an e-graph rule pair they are a two-cycle: co-resident, every score
#  class merges with its distributed twin and the closure explodes
#  (measured: T=128/K=8, k=8 saturation did not finish inside 800 s,
#  against ~30 s for the reachability stage alone).
#
#  They must therefore be SCHEDULED — never co-resident.  The bridge is
#  a *reachability* law for the reassociation stage; see
#  :func:`om_shared_state_stages`, which returns the two stage lists.
# ---------------------------------------------------------------------------


DIV_INTO_MATMUL = R(
    "div_into_matmul",
    Op.make("div", Op.make("matmul", "A", "B"), "c"),
    Op.make("matmul", Op.make("div", "A", "c"), "B"),
    cond=("scalar", "c"),
    law="div(A@B, c) = (A/c)@B for a scalar c — the score's scalar "
    "denominator moves INSIDE the contraction, which is what lets the "
    "shared-state reassociation see mul(W, matmul(Q, B)).  Exact "
    "inverse of matmul_div_distribute; the pair must be scheduled, "
    "never co-resident (see om_shared_state_stages).",
)


def om_div_bridge_laws() -> list[Rewrite]:
    """Return the div-naturality bridge as a one-element list.

    The bridge is the *exact inverse* of ``matmul_div_distribute``, so
    it is only ever sound to run it in a stage from which that law is
    absent — :func:`om_shared_state_stages` does exactly that.
    """
    return [DIV_INTO_MATMUL]


def _fold_bridges() -> list[Rewrite]:
    """Return the naturality bridges the chunked carrier needs.

    The six the fold's last hops need (plan 0023): div/mul naturality,
    the contraction readouts, the associativity reversal and the
    shared-denominator factorings.
    """
    from mlopt_core.laws.tensor import (
        ASSOC_MATMUL_REV,
        DIV_ADD_FACTOR,
        FACTOR_MUL,
        FACTOR_MUL_ADD,
        MATMUL_DIV_DISTRIBUTE,
        MUL_DIV_SWAP,
    )

    return [
        MUL_DIV_SWAP,
        MATMUL_DIV_DISTRIBUTE,
        ASSOC_MATMUL_REV,
        FACTOR_MUL,
        FACTOR_MUL_ADD,
        DIV_ADD_FACTOR,
    ]


def om_shared_state_stages(
    k: int,
) -> tuple[list[Rewrite], list[Rewrite]]:
    """Return the two scheduled rule stages the shared fold needs.

    ``(reach, reassoc)``.  Saturate with ``reach`` to a fixed point,
    then continue with ``reassoc`` on the SAME e-graph — staging is
    successive :meth:`~mlopt_core.egraph.EGraph.run` calls, and
    ``rule_budgets`` persist across them, so the split costs nothing
    beyond the second pass.

    ``reach`` is the everyday chunked-carrier set: ``OM_LAWS`` +
    ``OM_CHUNKIFY`` + ``SPLIT_SCALAR_RULES`` + the six naturality
    bridges + ``om_kway_laws(k)``.  It reaches the chunked carrier and
    the folded *tree*, but the score's scalar denominator sits outside
    every cross contraction, so the reassociation never fires.

    ``reassoc`` is ``reach`` with ``matmul_div_distribute`` REPLACED by
    its exact inverse :data:`DIV_INTO_MATMUL`, plus
    ``om_shared_state_laws()``.  The bridge lifts the denominator
    inside the contraction — ``om_shared_mat_reassoc`` then sees
    ``mul(W, matmul(Q, B))`` and fires — and splitting the two div laws
    across the stages is what keeps the closure finite.

    Both stages are opt-in and priced with ``rule_budgets``: the K-way
    law and the bridge are closure generators.
    """
    from mlopt_core.laws.tensor import (
        MATMUL_DIV_DISTRIBUTE,
        SPLIT_SCALAR_RULES,
    )

    bridges = _fold_bridges()
    reach = (
        list(OM_LAWS)
        + list(OM_CHUNKIFY)
        + list(SPLIT_SCALAR_RULES)
        + bridges
        + om_kway_laws(k)
    )
    reassoc = (
        list(OM_LAWS)
        + list(OM_CHUNKIFY)
        + list(SPLIT_SCALAR_RULES)
        + [r for r in bridges if r is not MATMUL_DIV_DISTRIBUTE]
        + om_kway_laws(k)
        + om_div_bridge_laws()
        + om_shared_state_laws()
    )
    return reach, reassoc


def om_shared_state_budgets(
    k: int,
    *,
    kway: int = 200,
    bridge: int = 400,
    reassoc: int = 400,
    chunkify: int = 60,
) -> dict[str, int]:
    """Return the ``rule_budgets`` map :func:`om_shared_state_stages` wants.

    The closure generators — the K-way expansion, the div bridge and
    the reassociation family — are bounded; the everyday laws are
    unbudgeted and reach their exact fixed point.
    """
    out = {
        "om_decayed_chunkify_rows": chunkify,
        "om_decayed_chunkify_keys": chunkify,
        f"om_decayed_kway_blocks_{k}": kway,
        "div_into_matmul": bridge,
    }
    out.update({r.name: reassoc for r in om_shared_state_laws()})
    return out


def _check_split_matmul(bound: dict) -> bool:
    """Require a well-typed split: int sizes partitioning the extent."""
    sz, i = bound.get("$attr:SZ"), bound.get("$attr:I")
    return (
        isinstance(sz, (tuple, list))
        and all(isinstance(x, int) and x > 0 for x in sz)
        and isinstance(i, int)
        and 0 <= i < len(sz)
    )


SPLIT_MATMUL_ROWS = R(
    "split_matmul_rows",
    Op.make(
        "split",
        Op.make("matmul", "a", "b"),
        sizes="SZ",
        dim=-2,
        index="I",
    ),
    Op.make(
        "matmul",
        Op.make("split", "a", sizes="SZ", dim=-2, index="I"),
        "b",
    ),
    check=_check_split_matmul,
    law="rows of a matmul are rows of the left operand: "
    "split_{-2}(a@b, i) = split_{-2}(a, i) @ b — the row-block "
    "of q·kᵀ is q_i · kᵀ.",
)


SPLIT_MATMUL_COLS = R(
    "split_matmul_cols",
    Op.make(
        "split",
        Op.make("matmul", "a", "b"),
        sizes="SZ",
        dim=-1,
        index="I",
    ),
    Op.make(
        "matmul",
        "a",
        Op.make("split", "b", sizes="SZ", dim=-1, index="I"),
    ),
    check=_check_split_matmul,
    law="columns of a matmul are columns of the right operand: "
    "split_{-1}(a@b, j) = a @ split_{-1}(b, j) — the key-block "
    "of q·kᵀ is q · (kᵀ)_j.",
)


#: ``DP`` := the split's pre-transpose axis — the axis ``SD`` names:
#: ``D1`` when ``SD == D0``, ``D0`` when ``SD == D1``, else ``SD``
#: itself.  All three dims must be bound ints (the procedural hook's
#: ``isinstance`` veto, as data).
_DSPEC_SPLIT_TRANSPOSE = {
    "DP": (
        "guard",
        (
            "and",
            ("attr-type", "SD", "int"),
            ("attr-type", "D0", "int"),
            ("attr-type", "D1", "int"),
        ),
        (
            "if",
            ("attr-eq-attr", "SD", "D0"),
            ("attr", "D1"),
            (
                "if",
                ("attr-eq-attr", "SD", "D1"),
                ("attr", "D0"),
                ("attr", "SD"),
            ),
        ),
    ),
}

#: Compat alias — ``_check_split_transpose`` calls the hook shape.
_derive_split_transpose = as_derive(_DSPEC_SPLIT_TRANSPOSE)


def _check_split_transpose(bound: dict) -> bool:
    """Require a well-typed split and int transpose dims."""
    return (
        _check_split_matmul(bound)
        and _derive_split_transpose(bound) is not None
    )


def _split_attrs_ok(bound: dict, size_key: str, index_key: str) -> bool:
    """Require a well-typed ``sizes`` partition and ``index``."""
    sz = bound.get(size_key)
    i = bound.get(index_key)
    return (
        isinstance(sz, (tuple, list))
        and all(isinstance(x, int) and x > 0 for x in sz)
        and isinstance(i, int)
        and 0 <= i < len(sz)
    )


def _check_split_commute(bound: dict) -> bool:
    """Require well-typed, disjoint-axis nested splits."""
    da, db = bound.get("$attr:DA"), bound.get("$attr:DB")
    if not (isinstance(da, int) and isinstance(db, int)):
        return False
    t = _vshape(bound.get("t"))
    if isinstance(t, tuple) and t:
        rank = len(t)
        if da % rank == db % rank:
            return False
    elif da == db:
        return False
    return _split_attrs_ok(
        bound, "$attr:SA", "$attr:I"
    ) and _split_attrs_ok(bound, "$attr:SB", "$attr:J")


SPLIT_COMMUTE = R(
    "split_commute",
    Op.make(
        "split",
        Op.make("split", "t", sizes="SA", dim="DA", index="I"),
        sizes="SB",
        dim="DB",
        index="J",
    ),
    Op.make(
        "split",
        Op.make("split", "t", sizes="SB", dim="DB", index="J"),
        sizes="SA",
        dim="DA",
        index="I",
    ),
    check=_check_split_commute,
    law="disjoint-axis splits commute: split_{DB}(split_{DA}(t)) "
    "= split_{DA}(split_{DB}(t)) — the row/col order of a block "
    "decomposition is immaterial; puts double splits in the "
    "canonical (row-then-col) order the block laws consume.",
)


def _check_split_flatten(bound: dict) -> bool:
    """Require well-typed nested same-axis splits that partition.

    ``SZ2`` must sum to ``SZ[I]`` — the outer split carves the inner
    block into pieces, so its sizes must partition that block's
    extent exactly (an inconsistent pair would mint an ill-typed
    flat split).
    """
    return (
        _split_attrs_ok(bound, "$attr:SZ", "$attr:I")
        and _split_attrs_ok(bound, "$attr:SZ2", "$attr:J")
        and sum(bound["$attr:SZ2"])
        == bound["$attr:SZ"][bound["$attr:I"]]
    )


def _derive_split_flatten(bound: dict) -> dict | None:
    """``SZF`` := ``SZ[:I] + SZ2 + SZ[I+1:]``, ``IF`` := ``I + J``.

    The flat partition: inner blocks before ``I`` stay whole, block
    ``I`` is replaced by its ``SZ2`` sub-partition, so the flat index
    of sub-block ``J`` is ``I + J``.
    """
    if not _check_split_flatten(bound):
        return None
    sz, sz2 = bound["$attr:SZ"], bound["$attr:SZ2"]
    i, j = bound["$attr:I"], bound["$attr:J"]
    return {
        "$attr:SZF": (*sz[:i], *sz2, *sz[i + 1 :]),
        "$attr:IF": i + j,
    }


SPLIT_FLATTEN = R(
    "split_flatten",
    Op.make(
        "split",
        Op.make("split", "t", sizes="SZ", dim="D", index="I"),
        sizes="SZ2",
        dim="D",
        index="J",
    ),
    Op.make("split", "t", sizes="SZF", dim="D", index="IF"),
    check=_check_split_flatten,
    derive=_derive_split_flatten,
    law="nested splits on the same axis collapse to one flat "
    "partition: split(split_{SZ,I}(t), SZ2, j) = "
    "split_{SZ[:I]+SZ2+SZ[I+1:], I+J}(t) — the recursive bisection "
    "produces nested (16,16)→(8,8) splits, but the block laws need "
    "the flat (16,8,8)-style partition to see ``split(split(decay))`` "
    "sites at depth ≥2.",
)


SPLIT_TRANSPOSE = R(
    "split_transpose",
    Op.make(
        "split",
        Op.make("transpose", "t", dim0="D0", dim1="D1"),
        sizes="SZ",
        dim="SD",
        index="I",
    ),
    Op.make(
        "transpose",
        Op.make("split", "t", sizes="SZ", dim="DP", index="I"),
        dim0="D0",
        dim1="D1",
    ),
    check=_check_split_transpose,
    dspec=_DSPEC_SPLIT_TRANSPOSE,
    law="split commutes through transpose: splitting the transposed "
    "axis D is splitting the axis D names in the pre-image "
    "(d1 when D=d0, d0 when D=d1, D otherwise) — kᵀ's key blocks "
    "are k's row blocks transposed.",
)


def _decay_state_lift(name: str, mul_first: bool) -> Rewrite:
    """Build ``add(mul(a,h), x)``/``add(x, mul(a,h))`` → ``applyd``."""
    m = Op.make("mul", "a", "h")
    return R(
        name,
        Op.make("add", m, "x") if mul_first else Op.make("add", "x", m),
        Op.make("applyd", Op.make("aff_diag", "a", "x"), "h"),
        check=_decay_state_like,
        law="a weighted state update is a diagonal-affine step: "
        "a⊙S + x = (aff_diag(a,x))(S) — the inter-chunk fold's "
        "gamma^o·B + rest spelling, lifted onto the scan carrier.  "
        "State-ness is judged by SHAPE (≥2-D), not the affd cond's "
        "op list — om_decayed contributions are states too.",
    )


# ---------------------------------------------------------------------------
#  Law set
# ---------------------------------------------------------------------------

#: Minimal law set for chunked-attention discovery.  Deliberately
#: excludes a commutativity rule for om_compose — comm is the explosive
#: law, same as SCAN_LAWS; block order is preserved by the concat
#: structure itself.
OM_LAWS: list[Rewrite] = [
    OM_LIFT,
    OM_LIFT_PLAIN,
    OM_UNLIFT,
    OM_SPLIT,
    OM_MERGE,
    OM_ASSOC,
    OM_ASSOC_REV,
    *CONCAT_BINARIZE,
    MATMUL_T_CONCAT,
    *OM_MASK_LAWS,
    *SDPA_CAT_LAWS,
    OM_DECAYED_LIFT,
    OM_DECAYED_UNLIFT,
    DECAY_LIFT,
    DECAY_LIFT_DIAG,
    *OM_DECAYED_CHUNK_LAWS,
    DECAY_BLOCK_DIAG,
    DECAY_BLOCK_CROSS,
    DECAY_BLOCK_CROSS_FLAT,
    OM_DECAYED_CROSS,
    OM_DECAYED_CROSS_GEN,
    MATMUL_COL_DECAY,
    DECAY_VEC_OFFSET,
    MATMUL_SCALAR_FACTOR,
    DECAY_STATE_WEIGHT_SINK,
    DECAY_STATE_WEIGHT_EXIT,
    UNSQ_MUL,
    _decay_state_lift("decay_state_lift", True),
    _decay_state_lift("decay_state_lift_post", False),
    SPLIT_MATMUL_ROWS,
    SPLIT_MATMUL_COLS,
    SPLIT_TRANSPOSE,
    SPLIT_COMMUTE,
    SPLIT_FLATTEN,
]

#: The recursive bisection pair — split-introduction on a fused
#: ``om_decayed`` site.  These CREATE splits: an unbounded
#: saturation bisects every chunk again, so they stay out of the
#: everyday ``OM_LAWS`` set.  Sites that want the chunked space
#: opt in explicitly (``OM_LAWS + OM_CHUNKIFY``) and price the
#: search with ``rule_budgets``.
OM_CHUNKIFY: list[Rewrite] = [
    OM_DECAYED_CHUNKIFY_ROWS,
    OM_DECAYED_CHUNKIFY_KEYS,
]

#: The same set as a composable :class:`~mlopt_core.laws.RuleSet` —
#: the value the orchestrator's presets compose.
OM_RULES = RuleSet(
    "om",
    tuple(OM_LAWS),
    description="the online-softmax monoid laws (chunked attention)",
)

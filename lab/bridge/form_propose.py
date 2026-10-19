"""Form-level loop proposal — anti-unification with an e-graph oracle.

ADR-0008 gate 3, attempt 5.  The four prior moves each retired a cause:

* ``loop_propose`` — a syntactic ``unroll(step, n) → Loop(step, n)``
  rewrite: works on the authored scan, fires **0x** on the winner (no
  ``unroll`` subterm exists).
* ``index_abstract`` — parametric abstraction over the block index:
  works on the spelled fold, but the winner's blocks differ at the
  *cross* position in the **arity of the unfolded geometric sum**
  (``block1/block2/add[1]/mul[1]: op om_decayed vs add``) — no constant
  parameter absorbs arity.
* the state-spelling fix (``sharing_extract``) and the width
  abstraction (``sizes`` as ``PARAM_ATTRS``): real but not binding —
  the cross sum is *unfolded* and the state is re-derived per block
  with the query folded inside, so there is **no shared state subterm
  to bind**.

THE MOVE
    Use the saturated e-graph itself as the anti-unification oracle.
    Extraction *chose* the unfolded spelling, but the saturation had
    already **derived** the factored one: each block's state-read
    e-class contains a ``matmul(q̃, M)`` member whose right operand is
    the q-free accumulated matrix, and those state classes are linked
    by ``add(mul(w, M_prev), B)`` members — the recurrence, present in
    the graph, only not chosen.  So the step is not *found*
    syntactically and not *assumed* algebraically: it is **synthesised**
    from the family differences (index, width, decay constants all
    become wires) and **verified** by expanding the fold and checking
    every output lands back in the winner's own e-classes.

PROTOCOL
    1. Decompose the concat chain → blocks (scan order).
    2. Per block, enumerate the state-read e-class for a *q-factored*
       member ``matmul(q̃_i, M_i)``; the right operand's e-class is a
       state candidate.  None → obstruction ``no-factored-read``.
    3. Chain the candidates: ``M_i``'s class must hold an
       ``add(mul(w, M_{i-1}), B)``-shaped member (the ``applyd`` carrier
       spelling counts too).  Maximal linked runs are the fold chains;
       a gap is an obstruction ``state-chain-broken`` naming the pair.
    4. Bind per-block wires (own splits by source leaf, decay constants
       by attr signature) — the per-block *bindings*.  One step schema
       ``pack(S', o)`` serves every block: the family-varying content
       (split index, chunk width, decay constants) is all on wires.
    5. Emit ``unroll(step, n)`` per chain; ``saturate_comb`` with
       ``PROPOSE_LOOP`` folds it to ``Loop(step, n)`` — the loop-headed
       term, reached by proposal rather than recovered by the bridge.
    6. VERIFY: ``_apply`` the unrolled morphism on the real wire terms;
       each output block must land in the winner's own e-class (the
       e-graph's own certificate that the fold denotes the same
       computation), and the rebuilt concat must land in the root
       class.  The single-chain arm additionally tries to carry the
       state across the family boundary; its failure names the last
       obstruction (the e-graph never merged the two spellings of the
       boundary state — a saturation-reach gap, not a fold defect).

The measurement lives in ``lab/bridge/combinator_probe.py``
(``--form-propose``); the reading is in
``project/retros/gate3-form-level-proposer.md``.

THE BOUNDARY, MEASURED
    The named residual ``state-chain-broken:block3->block4`` is *not* a
    reach gap, as the first reading had it: the carried terminal state
    of the 4-wide chain is **g^12 * M_4** — a different value from the
    16-wide family's initial state M_4, not a different spelling of it
    (numeric: max|S_carried - g^12*M_4| <= 1.8e-15 vs max|d| = 12.4
    unscaled).  The fold invariant is ``S_i = g^phi_i * A_i`` with
    ``phi_i`` the start of the *previous own-partition chunk* and
    ``A_i`` the absolute g^{-p}-weighted accumulation; across the
    boundary ``phi`` jumps 8 -> 0, so the naive hand-off is
    over-weighted by g^12.  No sound rewrite can merge the two
    e-classes.

    The uniform step still works *at the value level* with two
    compensated wires: a reindexed read scale ``xi'_i = decay_vec(
    offset = start_i - off_i, t = C_i)`` and a steering update weight
    ``w'_i = decay_vec(t = 1, offset = phi_{i+1} - off_i)`` (with the
    matching eta offset ``phi_{i+1} - start_i`` on the new
    contribution).  :func:`compensated_loop_check` walks the covered
    blocks with those wires, binding each only when it is
    graph-resident — on the real winner the needed boundary wires
    (``decay_vec(t=1, offset=-8)`` and ``decay_vec(t=4, sign=-1,
    offset=-12)`` at the producing step, ``decay_vec(offset=4, t=16)``
    at block 4's read) are *not* in the graph, and the steered
    next-state lands in a fresh e-class (a true reach gap at block 5:
    same value, unmerged).  That is the measured obstruction, and it is
    what a "state-reindex" law family would have to derive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlopt_core.egraph.types import _member_key
from mlopt_core.ir import Const, Op, TensorType, Var

from lab.bridge import combinator as cb
from lab.bridge.index_abstract import _block_splits, block_outputs
from lab.bridge.loop_propose import (
    EXPAND_LOOP,
    PROPOSE_LOOP,
    loop_headed,
    saturate_comb,
    unroll,
)

__all__ = [
    "STEP_INPUTS",
    "ChainResult",
    "FormReport",
    "block_wires",
    "compensated_loop_check",
    "factored_states",
    "form_propose",
    "form_propose_report",
    "single_loop_check",
    "state_chains",
    "step_template",
]

_PH = TensorType(())

#: The step's wire parameters, in bundle order: the recurrence state
#: first (the ``Loop`` convention), then the per-block inputs — the
#: block's own q/k/v splits and every per-family constant (geometric
#: weight, within-chunk decay, intra mask decay, cross scale, score
#: denominator).  Because the constants are *wires*, one step schema
#: serves both chunk widths — the family difference lives in the
#: bindings, not the step.
S = Var("@S", _PH)
XQ = Var("@xq", _PH)
XK = Var("@xk", _PH)
XV = Var("@xv", _PH)
CW = Var("@cw", _PH)  # decay_vec(offset=C, t=1)  — the geometric weight
CE = Var("@ce", _PH)  # decay_vec(sign=-1, t=C)  — within-chunk decay
CD = Var("@cD", _PH)  # decay(t=C)               — intra mask decay
CX = Var("@cx", _PH)  # decay_vec(offset=C, t=C) — the cross scale
CC = Var("@cc", _PH)  # the score denominator (√T), a Const
STEP_INPUTS: tuple = (S, XQ, XK, XV, CW, CE, CD, CX, CC)

_Q_SRC = {"p_q_weight"}
_K_SRC = {"p_k_weight"}
_V_SRC = {"p_v_weight"}
_KV_LEAVES = {"p_k_weight", "p_k_bias", "p_v_weight", "p_v_bias"}
_Q_LEAVES = {"p_q_weight", "p_q_bias"}


# ---------------------------------------------------------------------------
#  The step schema — one template, all per-block content on wires
# ---------------------------------------------------------------------------


def step_template() -> Any:
    """Build the synthesised step ``pack(S', o)`` over :data:`STEP_INPUTS`.

    ``S' = w·S + (k ⊙ η)ᵀ v`` and
    ``o = om_decayed(qkᵀ/c, D, v) + xi ⊙ ((q/c) @ S)`` — the chunked-
    retention step, with the state's *arithmetically-unfolded* spelling
    (the arity-growing geometric sum the last retro named) replaced by
    the carried matrix the e-graph already proved equal.
    """
    b_term = Op.make(
        "matmul",
        Op.make("transpose", Op.make("mul", XK, CE), dim0=-1, dim1=-2),
        XV,
    )
    s_out = Op.make("add", Op.make("mul", CW, S), b_term)
    score = Op.make(
        "matmul", XQ, Op.make("transpose", XK, dim0=-1, dim1=-2)
    )
    intra = Op.make("om_decayed", Op.make("div", score, CC), CD, XV)
    read = Op.make("matmul", Op.make("div", XQ, CC), S)
    out = Op.make("add", intra, Op.make("mul", CX, read))
    return Op.make("pack", s_out, out)


# ---------------------------------------------------------------------------
#  Per-block wire bindings — derived from the block's own subterms
# ---------------------------------------------------------------------------


def _leaves_of(t: Any, acc: set | None = None) -> set:
    """Leaf reprs of a term."""
    acc = set() if acc is None else acc
    if not isinstance(t, Op):
        acc.add(repr(t))
        return acc
    for a in t.args:
        _leaves_of(a, acc)
    return acc


def _subterms(t: Any) -> list:
    """Distinct op subterms, parents before children (pre-order)."""
    seen: set = set()
    out: list = []
    stack = [t]
    while stack:
        x = stack.pop()
        if not isinstance(x, Op) or x in seen:
            continue
        seen.add(x)
        out.append(x)
        stack.extend(x.args)
    return out


def _const_role(t: Any) -> Var | None:
    """Classify a decay/scale subterm to its wire role (or ``None``).

    The signatures are the decay-vector attr patterns this workload's
    reassociation emits; an unrecognised subterm simply isn't bound —
    the verification stage catches a missing or wrong binding.
    """
    if isinstance(t, Const):
        return CC
    if not isinstance(t, Op):
        return None
    # a BARE ``decay_vec`` is not a wire: the step's roles carry the
    # ``unsqueeze(dim=-1)`` (column) spellings, bound by the wrapper
    # case below or reconstructed by ``_fill_wires``.
    if t.op == "decay_vec":
        return None
    if t.op == "decay":
        return CD
    # the vector/matrix spellings ride inside an unsqueeze; the step's
    # wires carry the ``dim=-1`` (column) broadcast — the ``dim=-2``
    # variant appears only inside re-expansion terms the step does
    # not read.
    if (
        t.op == "unsqueeze"
        and t.args
        and dict(t.attrs).get("dim") == -1
        and isinstance(t.args[0], Op)
        and t.args[0].op == "decay_vec"
    ):
        a = dict(t.args[0].attrs)
        if a.get("sign") == -1:
            return CE
        if a.get("offset") is not None and a.get("offset") == a.get(
            "t"
        ):
            return CX
        if a.get("t") == 1:
            return CW
        return None
    return None


def block_wires(block: Any, own: set) -> dict[Var, Any]:
    """Derive block ``i``'s wire bindings from its own subterms.

    ``own`` is the block's introduced splits (``_block_splits``); the
    ``dim=-2`` ones bind ``@xq``/``@xk``/``@xv`` by source leaf (the
    ``dim=-1`` split of transposed ``k`` is rebuilt inside the template
    from ``@xk``).  Closed decay/scale subterms bind the constant wires
    by attr signature (:func:`_const_role`).  A partial map is fine —
    :func:`_fill_wires` completes it from the e-graph.
    """
    wires: dict[Var, Any] = {}
    # the block's OWN index is the one its intra score reads — earlier
    # positions can be "introduced" here too (inside the state read), so
    # a bare membership test is not enough; the intra fixes the index.
    own_index = None
    intra = (
        block.args[0]
        if (isinstance(block, Op) and block.op == "add")
        else block
    )
    for s in _subterms(intra):
        if (
            s.op == "split"
            and dict(s.attrs).get("dim") == -2
            and (_leaves_of(s) & _Q_SRC)
        ):
            own_index = dict(s.attrs).get("index")
            break
    for s in own:
        if not isinstance(s, Op) or not s.args:
            continue
        a = dict(s.attrs)
        if a.get("dim") != -2 or (
            own_index is not None and a.get("index") != own_index
        ):
            continue
        src = _leaves_of(s.args[0])
        if src & _Q_SRC:
            wires[XQ] = s
        elif src & _K_SRC:
            wires[XK] = s
        elif src & _V_SRC:
            wires[XV] = s
    for st in _subterms(block):
        r = _const_role(st)
        if r is not None:
            wires.setdefault(r, st)
        for a in st.args:
            if isinstance(a, Const):
                wires.setdefault(CC, a)
    return wires


def _find_subterm(block: Any, pred: Any) -> Any:
    """First subterm matching ``pred`` (pre-order), else ``None``."""
    return next((s for s in _subterms(block) if pred(s)), None)


def _fill_wires(eg: Any, block: Any, wires: dict, own_ids: set) -> dict:
    """Fill missing wire roles with graph-resident terms.

    A block spelled without the factored read never *contains* the
    ``dim=-2`` k-split or the geometric weight — those live only in the
    state members the saturation derived.  Reconstruct them from the
    block's components (the ``linear`` leaves, the ``c_lifted`` constant
    base, the own split's ``sizes``/``index``) and keep only terms that
    are *present in the e-graph* (``_locate``) — a reconstruction that
    isn't is a named obstruction, not a binding.
    """
    own_splits = [
        s
        for s in _subterms(block)
        if s.op == "split"
        and dict(s.attrs).get("dim") == -2
        and _leaves_of(s) & _Q_SRC
    ]
    if not own_splits:
        return wires
    ref = own_splits[-1]  # the intra's q split (deepest-introduced)
    index = dict(ref.attrs).get("index")
    sizes = dict(ref.attrs).get("sizes")
    width = (
        sizes[index]
        if sizes is not None and index is not None
        else None
    )
    cbase = _find_subterm(block, lambda s: s.op == "detach_")
    if cbase is None:
        cbase = _find_subterm(block, lambda s: s.op == "contiguous")
    linear_of = {
        leaf: _find_subterm(
            block,
            lambda s, lf=leaf: s.op == "linear" and lf in _leaves_of(s),
        )
        for leaf in ("p_q_weight", "p_k_weight", "p_v_weight")
    }

    def located(t: Any) -> Any:
        eid, _ = eg._locate(t)
        return t if eid is not None else None

    if XK not in wires and linear_of["p_k_weight"] is not None:
        cand = Op.make(
            "split",
            linear_of["p_k_weight"],
            dim=-2,
            index=index,
            sizes=sizes,
        )
        got = located(cand)
        if got is not None:
            wires[XK] = got
    for var, mk in (
        (XQ, "p_q_weight"),
        (XV, "p_v_weight"),
    ):
        if var not in wires and linear_of[mk] is not None:
            cand = Op.make(
                "split",
                linear_of[mk],
                dim=-2,
                index=index,
                sizes=sizes,
            )
            got = located(cand)
            if got is not None:
                wires[var] = got
    if cbase is not None and width is not None:
        for var, t in (
            (
                CW,
                Op.make(
                    "unsqueeze",
                    Op.make("decay_vec", cbase, offset=width, t=1),
                    dim=-1,
                ),
            ),
            (
                CE,
                Op.make(
                    "unsqueeze",
                    Op.make("decay_vec", cbase, sign=-1, t=width),
                    dim=-1,
                ),
            ),
            (CD, Op.make("decay", cbase, t=width)),
            (
                CX,
                Op.make(
                    "unsqueeze",
                    Op.make("decay_vec", cbase, offset=width, t=width),
                    dim=-1,
                ),
            ),
        ):
            if var not in wires:
                got = located(t)
                if got is not None:
                    wires[var] = got
    return wires


# ---------------------------------------------------------------------------
#  The e-graph oracle: factored reads and the state chain
# ---------------------------------------------------------------------------


def _read_of(block: Any) -> Any:
    """Return the state-read subterm — ``add(intra, mul(xi, read))``'s payload."""
    if (
        isinstance(block, Op)
        and block.op == "add"
        and len(block.args) == 2
    ):
        tail = block.args[1]
        if (
            isinstance(tail, Op)
            and tail.op == "mul"
            and len(tail.args) == 2
        ):
            return tail.args[1]
        return tail
    return block


def _find_qm(t: Any) -> list[tuple[Any, Any]]:
    """Every ``matmul(q-side, M-side)`` in ``t`` — the factored reads.

    A factored read is a ``matmul`` whose left carries q-leaves and no
    k/v leaves and whose right carries both k and v leaves and no q —
    the q-free accumulated state.
    """
    out: list[tuple[Any, Any]] = []
    for x in _subterms(t):
        if x.op != "matmul" or len(x.args) != 2:
            continue
        a, b = x.args
        if not (isinstance(a, Op) and isinstance(b, Op)):
            continue
        la, lb = _leaves_of(a), _leaves_of(b)
        if (
            (la & _Q_LEAVES)
            and not (la & _KV_LEAVES)
            and not (lb & _Q_LEAVES)
            and ({"p_k_weight", "p_v_weight"} <= lb)
        ):
            out.append((a, b))
    return out


def factored_states(eg: Any, read_eid: int, cost_fn: Any) -> list[dict]:
    """Factored-read candidates of a block's read e-class.

    Enumerates the class's members (canonical order — the deterministic
    scan), extracts each as a term, and collects the ``matmul(q̃, M)``
    decompositions: ``(q-term, M-term, M's e-class)`` triples.
    """
    eid = eg.find(read_eid)
    out: list[dict] = []
    for node in sorted(eg.get_class(eid).nodes, key=_member_key):
        if node.op == "leaf":
            continue
        m = eg.extract_best(eid, cost_fn, overrides={eid: node})
        if m is None:
            continue
        for _q_t, m_t in _find_qm(m):
            m_eid, _ = eg._locate(m_t)
            if m_eid is not None:
                out.append({"m": m_t, "m_eid": eg.find(m_eid)})
    seen: set = set()
    uniq: list[dict] = []
    for c in out:
        if c["m_eid"] not in seen:
            seen.add(c["m_eid"])
            uniq.append(c)
    return uniq


def _chain_links(eg: Any, s_prev: int, s_cur: int) -> list[Any]:
    """Members of ``s_cur``'s class that read ``s_prev`` as update base.

    An ``add(X, B)`` member links when child ``X``'s class *is* the
    previous state or holds a ``mul``/``applyd`` member whose factor is
    the previous state — the ``w·S_{i-1}`` update under either spelling.
    """

    def wraps_prev(cid: int) -> bool:
        cc = eg.find(cid)
        if cc == s_prev:
            return True
        return any(
            (
                sub.op == "mul"
                and any(eg.find(x) == s_prev for x in sub.children)
            )
            or (
                sub.op == "applyd"
                and len(sub.children) == 2
                and eg.find(sub.children[1]) == s_prev
            )
            for sub in eg.get_class(cc).nodes
        )

    return [
        n
        for n in eg.get_class(s_cur).nodes
        if n.op == "add"
        and len(n.children) == 2
        and any(wraps_prev(c) for c in n.children)
    ]


def state_chains(
    eg: Any, blocks: list[Any], cost_fn: Any
) -> tuple[list[list[int]], list[list[dict]], list[str]]:
    """Chain the per-block state candidates in scan order.

    Returns ``(chains, candidates, obstructions)``: maximal runs of
    consecutive blocks whose read classes contain factored members whose
    state classes link by an ``add(mul(w,·),B)`` update; the candidate
    list per block; and named obstructions.  A block with no factored
    member is a *boundary* (peeled), not a break.
    """
    cands: list[list[dict]] = []
    for b in blocks:
        read = _read_of(b)
        eid, _ = eg._locate(read)
        cands.append(
            [] if eid is None else factored_states(eg, eid, cost_fn)
        )

    chains: list[list[int]] = []
    obs: list[str] = []
    i = 0
    n = len(blocks)
    while i < n:
        if not cands[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and cands[j + 1]:
            linked = any(
                _chain_links(eg, cj["m_eid"], cj1["m_eid"])
                for cj in cands[j]
                for cj1 in cands[j + 1]
            )
            if not linked:
                break
            j += 1
        chains.append(list(range(i, j + 1)))
        if j + 1 < n:
            if cands[j + 1]:
                obs.append(
                    f"state-chain-broken:block{j}->block{j + 1} — the "
                    "state classes hold no shared add(mul(w,·),B) update"
                )
            else:
                obs.append(f"no-factored-read:block{j + 1}")
        i = j + 1
    return chains, cands, obs


def _pick_chain_states(
    eg: Any, chain: list[int], cands: list[list[dict]]
) -> list[dict] | None:
    """Choose one candidate per block so consecutive M-classes link."""
    out = [cands[chain[0]][0]]
    for i in chain[1:]:
        nxt = next(
            (
                c
                for c in cands[i]
                if _chain_links(eg, out[-1]["m_eid"], c["m_eid"])
            ),
            None,
        )
        if nxt is None:
            return None
        out.append(nxt)
    return out


# ---------------------------------------------------------------------------
#  Expand / verify
# ---------------------------------------------------------------------------


def _subst(t: Any, env: dict) -> Any:
    if t in env:
        return env[t]
    if isinstance(t, Op):
        return Op.make(
            t.op, *(_subst(a, env) for a in t.args), **dict(t.attrs)
        )
    return t


def _eclass(eg: Any, t: Any) -> int:
    return eg.find(eg.add_term(t))


def _concat_skeleton(root: Any, blocks: list[Any]) -> Any:
    """Return the winner's concat tree with block leaves → their indices.

    A concat leaf that is not a block becomes ``("raw", term)`` — a
    verbatim program output the fold does not cover.
    """
    if isinstance(root, Op) and root.op == "concat":
        return tuple(_concat_skeleton(a, blocks) for a in root.args)
    for i, b in enumerate(blocks):
        if b == root:
            return i
    return ("raw", root)


def _skel_leaves(tree: Any) -> list[Any]:
    out: list[Any] = []
    if isinstance(tree, tuple) and tree and tree[0] == "raw":
        return [tree]
    if isinstance(tree, tuple):
        for x in tree:
            out.extend(_skel_leaves(x))
        return out
    return [tree]


def _concat_tree_morph(tree: Any, n_wires: int) -> Any:
    """Morphism ``n_wires → 1`` evaluating the concat tree in-place.

    Tree leaves are wire positions in the consumed bundle; concat nodes
    are ``arr(concat, k, dim=-2)`` applied to their children permuted to
    the bundle front, the result prepended.
    """
    bundle: list[Any] = list(range(n_wires))
    morphs: list = []

    def emit(t: Any) -> Any:
        if not isinstance(t, tuple):
            return t
        kids = [emit(x) for x in t]
        pos = [bundle.index(k) for k in kids]
        prm = cb._perm(
            pos + [i for i in range(len(bundle)) if i not in pos]
        )
        if not isinstance(prm, cb.Id):
            morphs.append(prm)
        morphs.append(
            cb._par(
                cb.Arr("concat", len(kids), (("dim", -2),)),
                cb.Id(len(bundle) - len(kids)),
            )
        )
        item = ("node", len(morphs))
        bundle[:] = [item] + [
            bundle[i] for i in range(len(bundle)) if i not in pos
        ]
        return item

    emit(tree)
    if len(bundle) > 1:
        morphs.append(
            cb._par(cb.Id(1), *(cb.Drop() for _ in bundle[1:]))
        )
    return cb._seq(*morphs)


def _positional(tree: Any, counter: list) -> Any:
    """Relabel a concat skeleton's leaves by DFS position (0, 1, …)."""
    if isinstance(tree, tuple) and tree and tree[0] == "raw":
        j = counter[0]
        counter[0] += 1
        return j
    if isinstance(tree, tuple):
        return tuple(_positional(x, counter) for x in tree)
    j = counter[0]
    counter[0] += 1
    return j


# ---------------------------------------------------------------------------
#  The report
# ---------------------------------------------------------------------------


@dataclass
class ChainResult:
    """One fold chain: the loop instance's measurement record."""

    blocks: list[int] = field(default_factory=list)
    seed_eid: int | None = None
    morph: Any = None  # the unroll(step, n) presentation term
    loop: Any = None  # the folded Loop(step, n) the closure selects
    verified: dict[int, bool] = field(default_factory=dict)
    ok: bool = False


@dataclass
class FormReport:
    """The form-level proposer's gate-3 record."""

    n_blocks: int = 0
    n_cands: list[int] = field(default_factory=list)
    chains: list[ChainResult] = field(default_factory=list)
    obstructions: list[str] = field(default_factory=list)
    step: Any = (
        None  # the step DAG template (pack(S', o) over the vars)
    )
    step_morph: Any = None  # the compiled step morphism
    bindings: dict[int, dict] = field(default_factory=dict)
    expanded: dict[int, Any] = field(default_factory=dict)
    rebuilt: Any = None
    prog: Any = None
    prog_stats: Any = None
    closure: dict = field(default_factory=dict)
    whole_equal: bool = False
    single_loop_verified: dict[int, bool] = field(default_factory=dict)
    max_abs_diff: float | None = None
    single_loop_compensated: dict[int, bool] = field(
        default_factory=dict
    )
    single_states: dict[int, Any] = field(
        default_factory=dict
    )  # the wire term entering each covered block (compensated arm)
    boundary: dict = field(
        default_factory=dict
    )  # the named boundary residual, measured (terms + e-classes +
    # the non-resident compensated wires)
    single_envs: dict[int, dict] = field(
        default_factory=dict
    )  # per-iteration effective wire envs (compensated arm)
    single_seed: Any = None  # the carried seed term (compensated arm)
    reindex: dict = field(
        default_factory=dict
    )  # the state-reindex close's measurement record
    single_prog: Any = None  # the assembled single-loop program morph
    single_prog_stats: Any = None
    single_loop: Any = (
        None  # the folded Loop(step, n) the closure selects
    )
    single_closure: dict = field(default_factory=dict)
    single_whole_equal: bool = (
        False  # single-loop DAG in the root class
    )


def form_propose(
    eg: Any,
    root_term: Any,
    cost_fn: Any,
    *,
    root_eid: int | None = None,
) -> FormReport:
    """Run the form-level proposer on an extracted term.

    Anti-unifies the block family through the saturated e-graph:
    factored reads give the state classes, the state classes give the
    chains, the chains give ``unroll(step, n)`` proposals — verified by
    expansion landing in the winner's own e-classes.
    """
    rep = FormReport()
    blocks = block_outputs(root_term)
    rep.n_blocks = len(blocks)
    block_cls = [_eclass(eg, b) for b in blocks]
    rep.step = step_template()

    chains, cands, obs = state_chains(eg, blocks, cost_fn)
    rep.obstructions.extend(obs)
    rep.n_cands = [len(c) for c in cands]

    own = _block_splits(blocks)
    raw_bindings = {
        i: _fill_wires(eg, b, block_wires(b, own[i]), own[i])
        for i, b in enumerate(blocks)
    }
    # a missing constant wire is borrowed from a sibling block of the
    # same chain — the constant is family-uniform (a wrong borrow fails
    # the e-class check, so this can only widen honestly).
    for ch in chains:
        merged: dict[Var, Any] = {}
        for i in ch:
            merged.update(raw_bindings[i])
        for i in ch:
            rep.bindings[i] = {
                v: raw_bindings[i].get(v, merged.get(v))
                for v in STEP_INPUTS[1:]
            }

    root_cls = (
        eg.find(root_eid)
        if root_eid is not None
        else _eclass(eg, root_term)
    )

    # --- expand each chain concretely; verify per block ---------------------
    ok_chains: list[tuple[ChainResult, list[int], list[dict]]] = []
    for ch in chains:
        res = ChainResult(blocks=ch)
        rep.chains.append(res)
        chosen = _pick_chain_states(eg, ch, cands)
        if chosen is None:
            rep.obstructions.append(
                f"chain-choice-failed:block{ch[0]}..block{ch[-1]}"
            )
            continue
        res.seed_eid = chosen[0]["m_eid"]
        s_cur = eg.extract_best(chosen[0]["m_eid"], cost_fn)
        ok = True
        for i in ch:
            env = dict(rep.bindings.get(i, {}))
            missing = [v.name for v in STEP_INPUTS[1:] if v not in env]
            if missing:
                rep.obstructions.append(
                    f"unbound-wires:block{i} {missing}"
                )
                ok = False
                res.verified[i] = False
                continue
            env[S] = s_cur
            inst = _subst(rep.step, env)
            s_cur, o_i = inst.args[0], inst.args[1]
            rep.expanded[i] = o_i
            hit = _eclass(eg, o_i) == block_cls[i]
            res.verified[i] = hit
            ok = ok and hit
            if not hit:
                rep.obstructions.append(
                    f"expand-mismatch:block{i} — the synthesised "
                    "step's output does not land in the block's "
                    "own e-class"
                )
        res.ok = ok
        if ok:
            ok_chains.append((res, ch, chosen))

    # --- present the folds as combinator terms ----------------------------
    covered = {i for res, ch, _c in ok_chains for i in ch}
    peeled = [i for i in range(rep.n_blocks) if i not in covered]
    if ok_chains:
        from lab.bridge.index_abstract import _compile_with_inputs

        step_morph = _compile_with_inputs(rep.step, list(STEP_INPUTS))
        rep.step_morph = step_morph
        nx = len(STEP_INPUTS) - 1
        wire_terms: list[Any] = []
        for res, ch, chosen in ok_chains:
            res.morph = unroll(step_morph, len(ch))
            res.loop = cb.Loop(step_morph, len(ch), nx=nx, ny=1)
            wire_terms.append(
                eg.extract_best(chosen[0]["m_eid"], cost_fn)
            )
            for i in ch:
                env = rep.bindings[i]
                wire_terms.extend(env[v] for v in STEP_INPUTS[1:])

        # --- assembled program morphism: seq(pre, mid, post) ---------------
        skel = _concat_skeleton(root_term, blocks)
        skel_leaves = _skel_leaves(skel)
        n_out = len(skel_leaves)
        raw_leaves = [x for x in skel_leaves if isinstance(x, tuple)]
        # wires: per chain [S_init, inputs…], then peeled block outputs,
        # then any raw (non-block) concat leaves — all verbatim.
        extras = [blocks[i] for i in peeled] + [
            r[1] for r in raw_leaves
        ]
        pre = cb._par(*(cb.Leaf(t) for t in wire_terms + extras))
        mid = cb._par(
            *(
                [res.morph for res, _c, _ch in ok_chains]
                + [cb.Id(len(extras))]
            )
        )
        # mid emits, per chain, [S'_end, o…] then the extras verbatim.
        pos_of: dict[Any, int] = {}
        pos = 0
        for _res, ch, _c in ok_chains:
            pos += 1  # the chain's final state wire (dropped)
            for i in ch:
                pos_of[i] = pos
                pos += 1
        for k in range(len(extras)):
            pos_of[("x", k)] = pos + k
        keep = sorted(pos_of.values())
        # a perm cannot drop wires — the final-state wires are dropped
        # positionally: par(Drop, Id(chain), …) over contiguous spans.
        drop_parts: list[Any] = []
        for _res, ch, _c in ok_chains:
            drop_parts += [cb.Drop(), cb.Id(len(ch))]
        drop_parts.append(cb.Id(len(extras)))
        drop_seg = cb._par(*drop_parts)
        shift = {old: new for new, old in enumerate(keep)}
        # scan order = the skeleton's leaf order; extras are indexed as
        # peeled blocks first, then raw leaves (matching ``extras``).
        peeled_set = set(peeled)
        scan: list[int] = []
        for leaf_ in skel_leaves:
            if isinstance(leaf_, tuple):
                scan.append(
                    pos_of[("x", len(peeled) + raw_leaves.index(leaf_))]
                )
            elif leaf_ in peeled_set:
                scan.append(pos_of[("x", peeled.index(leaf_))])
            else:
                scan.append(pos_of[leaf_])
        scan_perm = cb._perm([shift[s] for s in scan])
        post = cb._seq(
            drop_seg,
            scan_perm,
            _concat_tree_morph(_positional(skel, [0]), n_out),
        )
        rep.prog = cb._seq(pre, mid, post)
        rep.prog_stats = cb.morph_stats(rep.prog)

        # --- the proposing-rule closure on each unroll ---------------------
        fired: dict[str, int] = {}
        n_terms = 0
        costs: list[int] = []
        for res, _ch, _c in ok_chains:
            r = saturate_comb(res.morph, [PROPOSE_LOOP, EXPAND_LOOP])
            n_terms += r["n_terms"]
            for k_, v_ in r["fired"].items():
                fired[k_] = fired.get(k_, 0) + v_
            costs.append(r["cost"])
            if loop_headed(r["best"]):
                res.loop = r["best"]
        rep.closure = {
            "n_terms": n_terms,
            "fired": fired,
            "costs": costs,
        }

        # --- whole-term check: rebuilt concat must land in the root class --
        def _rebuild(t: Any) -> Any:
            if isinstance(t, tuple) and t and t[0] == "raw":
                return t[1]
            if isinstance(t, tuple):
                return Op.make(
                    "concat", *(_rebuild(x) for x in t), dim=-2
                )
            return rep.expanded.get(t, blocks[t])

        rep.rebuilt = _rebuild(skel)
        rep.whole_equal = _eclass(eg, rep.rebuilt) == root_cls
        rep.single_loop_verified = single_loop_check(
            eg, rep, root_term, cost_fn
        )
        rep.single_loop_compensated = compensated_loop_check(
            eg, rep, root_term, cost_fn
        )
    return rep


# ---------------------------------------------------------------------------
#  The single-chain arm: can one loop carry the whole scan?
# ---------------------------------------------------------------------------


def single_loop_check(
    eg: Any, rep: FormReport, root_term: Any, cost_fn: Any
) -> dict[int, bool]:
    """Try carrying ONE state through every covered block in scan order.

    Seeds the fold with the first covered block's state and lets the
    uniform step run across the family boundary.  The per-block e-class
    verdict shows exactly where the graph's equalities run out — the
    family-boundary state spellings are the same *value*, but the
    saturation never merged their classes, so the hand-off verifies
    numerically only.  That distinction is the honest report.
    """
    blocks = block_outputs(root_term)
    covered = sorted(i for r in rep.chains for i in r.blocks if r.ok)
    if not covered:
        return {}
    first = covered[0]
    eid, _ = eg._locate(_read_of(blocks[first]))
    if eid is None:
        return {}
    seed_cands = factored_states(eg, eid, cost_fn)
    if not seed_cands:
        return {}
    s_cur = eg.extract_best(seed_cands[0]["m_eid"], cost_fn)
    block_cls = [_eclass(eg, b) for b in blocks]
    out: dict[int, bool] = {}
    for i in covered:
        env = dict(rep.bindings[i])
        env[S] = s_cur
        inst = _subst(rep.step, env)
        s_cur, o_i = inst.args[0], inst.args[1]
        out[i] = _eclass(eg, o_i) == block_cls[i]
    return out


def _dv_attrs(t: Any) -> dict | None:
    """Attrs of an ``unsqueeze(decay_vec(...))`` or bare ``decay_vec``."""
    if (
        isinstance(t, Op)
        and t.op == "unsqueeze"
        and t.args
        and isinstance(t.args[0], Op)
    ):
        t = t.args[0]
    if isinstance(t, Op) and t.op == "decay_vec":
        return dict(t.attrs)
    return None


def _split_span(env: dict) -> tuple[int, int] | None:
    """``(start, width)`` of the block's own chunk, from its bound q split."""
    xq = env.get(XQ)
    if not isinstance(xq, Op) or xq.op != "split":
        return None
    a = dict(xq.attrs)
    sz, idx = a.get("sizes"), a.get("index")
    if not (
        isinstance(sz, (tuple, list))
        and isinstance(idx, int)
        and 0 <= idx < len(sz)
    ):
        return None
    return int(sum(sz[:idx])), int(sz[idx])


def _decay_base(env: dict) -> Any:
    """Return the decay-vector base term (the lifted gamma)."""
    for v in (CW, CE, CD, CX):
        t = env.get(v)
        if (
            isinstance(t, Op)
            and t.op == "unsqueeze"
            and t.args
            and isinstance(t.args[0], Op)
            and t.args[0].op == "decay_vec"
            and t.args[0].args
        ):
            return t.args[0].args[0]
        if isinstance(t, Op) and t.op == "decay" and t.args:
            return t.args[0]
    return None


def _usq_dv(cbase: Any, **attrs: Any) -> Any:
    """``unsqueeze(decay_vec(cbase, **attrs), dim=-1)`` — a wire term.

    ``offset=0`` is dropped: the bound spellings never carry the
    default explicitly, and attr dicts compare verbatim.
    """
    attrs = {
        k: v for k, v in attrs.items() if not (k == "offset" and v == 0)
    }
    return Op.make(
        "unsqueeze", Op.make("decay_vec", cbase, **attrs), dim=-1
    )


def compensated_loop_check(
    eg: Any,
    rep: FormReport,
    root_term: Any,
    cost_fn: Any,
    post_add: Any = None,
) -> dict[int, bool]:
    """Run the boundary-aware single-loop arm — the state-reindex probe.

    The fold's carried convention is ``S_i = g^phi_i * A_i``: the
    absolute accumulation ``A_i = sum_p g^{-p} k_p v_p`` rescaled by the
    start of the previous own-partition chunk.  A uniform step can
    still read block ``i`` correctly given *any* carried normalisation
    ``off_i`` — the read scale just needs the offset that compensates
    it: ``xi'_i = decay_vec(offset = start_i - off_i, t = C_i)``.  And
    the update can *steer* the next state into the next family's
    convention with ``w'_i = decay_vec(t=1, offset = phi_{i+1} -
    off_i)`` and the matching eta offset ``phi_{i+1} - start_i`` on the
    new contribution.

    Every compensated wire is bound **only if graph-resident**
    (``eg._locate``) — a missing wire is named in ``rep.obstructions``
    (``boundary-xi-nonresident`` / ``boundary-renorm-unbound``), and a
    produced-but-fresh e-class lands as the honest ``False`` verdict.
    ``rep.boundary`` records the measured pair (carried term/class vs
    the target's class/term) at the first broken boundary, plus the
    non-resident wire terms, so the numeric arm can measure the value
    gap without re-walking the chain.
    """
    blocks = block_outputs(root_term)
    covered = sorted(i for r in rep.chains for i in r.blocks if r.ok)
    if len(covered) < 2:
        return {}
    first = covered[0]
    eid, _ = eg._locate(_read_of(blocks[first]))
    if eid is None:
        return {}
    seed_cands = factored_states(eg, eid, cost_fn)
    if not seed_cands:
        return {}
    block_cls = [_eclass(eg, b) for b in blocks]
    # the seed must verify the FIRST covered block: post-saturation the
    # read class holds extra members, so the canonical order can put a
    # same-shape/wrong-value candidate first — probe each one.
    s_cur = None
    for c in seed_cands:
        cand = eg.extract_best(c["m_eid"], cost_fn)
        env_t = dict(rep.bindings.get(first, {}))
        env_t[S] = cand
        o_t = _subst(rep.step, env_t).args[1]
        if eg.find(eg.add_term(o_t)) == eg.find(block_cls[first]):
            s_cur = cand
            break
    if s_cur is None:
        s_cur = eg.extract_best(seed_cands[0]["m_eid"], cost_fn)
    rep.single_seed = s_cur
    out: dict[int, bool] = {}
    missing = rep.boundary.setdefault("missing_wires", {})

    # the seed is the family's own spelling: off = phi_first, read off
    # the bound xi (offset = start - phi).
    span0 = _split_span(rep.bindings.get(first, {}))
    cx0 = _dv_attrs(rep.bindings.get(first, {}).get(CX))
    off: int | None = (
        (span0[0] - cx0.get("offset", 0))
        if (span0 is not None and cx0 is not None)
        else None
    )

    for pos, i in enumerate(covered):
        env = dict(rep.bindings[i])
        span = _split_span(env)
        cbase = _decay_base(env)
        rep.single_states[i] = s_cur

        # --- the read wire, compensated for the carried convention ---
        ok = True
        if span is not None and cbase is not None and off is not None:
            need = span[0] - off
            a_cx = _dv_attrs(env.get(CX)) or {}
            if (a_cx.get("offset", 0), a_cx.get("t")) != (
                need,
                span[1],
            ):
                xi = _usq_dv(cbase, offset=need, t=span[1])
                if eg._locate(xi)[0] is not None:
                    env[CX] = xi
                else:
                    rep.obstructions.append(
                        f"boundary-xi-nonresident:block{i} — the "
                        f"reindexed read scale decay_vec(offset={need},"
                        f" t={span[1]}) is not in the e-graph"
                    )
                    missing[i] = xi
                    ok = False
        env[S] = s_cur
        inst = _subst(rep.step, env)
        s_default, o_i = inst.args[0], inst.args[1]
        oid = eg.add_term(o_i)
        if post_add is not None:
            post_add()
        hit = eg.find(oid) == eg.find(block_cls[i])
        out[i] = ok and hit
        if not out[i] and "carried_eid" not in rep.boundary:
            # the named residual, as data: the wire value the loop
            # carried in vs the state class the block actually reads.
            read = _read_of(blocks[i])
            rid, _ = eg._locate(read)
            cands_i = (
                factored_states(eg, rid, cost_fn)
                if rid is not None
                else []
            )
            rep.boundary.update(
                {
                    "at": i,
                    "carried_term": s_cur,
                    "carried_eid": eg.find(eg.add_term(s_cur)),
                    "target_eid": (
                        cands_i[0]["m_eid"] if cands_i else None
                    ),
                    "target_term": (
                        eg.extract_best(cands_i[0]["m_eid"], cost_fn)
                        if cands_i
                        else None
                    ),
                }
            )

        rep.single_envs[i] = {
            v: t for v, t in env.items() if v is not S
        }
        # --- the update wire, steered into the next family's norm ----
        if pos + 1 >= len(covered):
            eg.add_term(s_cur)
            if post_add is not None:
                post_add()
            break
        j = covered[pos + 1]
        env_j = rep.bindings.get(j, {})
        span_j = _split_span(env_j)
        cx_j = _dv_attrs(env_j.get(CX))
        a_cw = _dv_attrs(env.get(CW))
        a_ce = _dv_attrs(env.get(CE))
        steered = False
        if (
            span is not None
            and span_j is not None
            and cbase is not None
            and cx_j is not None
            and off is not None
        ):
            phi_j = span_j[0] - cx_j.get("offset", 0)
            need_cw = phi_j - off
            need_ce = phi_j - span[0]
            if (a_cw or {}).get("offset", 0) == need_cw and (
                a_ce or {}
            ).get("offset", 0) == need_ce:
                s_cur = s_default  # the bound wires already steer
                off = phi_j
                steered = True
            else:
                cw = _usq_dv(cbase, t=1, offset=need_cw)
                ce = _usq_dv(cbase, t=span[1], sign=-1, offset=need_ce)
                r_cw, r_ce = eg._locate(cw)[0], eg._locate(ce)[0]
                if r_cw is not None and r_ce is not None:
                    env2 = dict(rep.bindings[i])
                    env2[CW], env2[CE], env2[S] = cw, ce, s_cur
                    s_cur = _subst(rep.step, env2).args[0]
                    off = phi_j
                    steered = True
                    env[CW], env[CE] = cw, ce
                else:
                    rep.obstructions.append(
                        f"boundary-renorm-unbound:block{i}->block{j} "
                        f"— the steering pair decay_vec(t=1, offset="
                        f"{need_cw}) / decay_vec(t={span[1]}, sign=-1,"
                        f" offset={need_ce}) is not in the e-graph"
                    )
                    rep.boundary.setdefault("missing_steer", {})[i] = {
                        "cw": cw,
                        "ce": ce,
                    }
        if not steered:
            s_cur = s_default
            off = (
                off + (a_cw or {}).get("offset", 0)
                if off is not None
                else None
            )
        # re-record now that any steering overrides have landed in env
        rep.single_envs[i] = {
            v: t for v, t in env.items() if v is not S
        }
        eg.add_term(s_cur)
        if post_add is not None:
            post_add()
    return out


# ---------------------------------------------------------------------------
#  The measurement
# ---------------------------------------------------------------------------


def form_propose_report(
    eg: Any,
    root_term: Any,
    cost_fn: Any,
    *,
    root_eid: int | None = None,
    numeric: Any = None,
) -> FormReport:
    """Print the gate-3 form-level measurement for a term."""
    rep = form_propose(eg, root_term, cost_fn, root_eid=root_eid)
    print(
        f"   form-propose: blocks={rep.n_blocks} "
        f"factored-read-candidates={rep.n_cands}"
    )
    for res in rep.chains:
        print(
            f"      chain blocks={res.blocks} seed-cls={res.seed_eid} "
            f"verified={res.verified} ok={res.ok}"
        )
    for o in rep.obstructions:
        print(f"      obstruction: {o}")
    ok = [r for r in rep.chains if r.ok]
    if ok:
        st_step = (
            cb.morph_stats(rep.step_morph).nodes
            if rep.step_morph is not None
            else 0
        )
        print(
            f"      step: {st_step}-node morphism over "
            f"{len(STEP_INPUTS)} wires; chains folded: {len(ok)}"
        )
        print(
            f"      whole-term e-class check (rebuilt concat in the "
            f"winner's root class): {rep.whole_equal}"
        )
        if rep.prog_stats is not None:
            st = rep.prog_stats
            print(
                f"      assembled program: morph={st.nodes} "
                f"loops={st.loop} dup={st.dup} perm={st.perm}"
            )
        print(
            f"      closure: terms={rep.closure.get('n_terms')} "
            f"fired={rep.closure.get('fired')} "
            f"costs={rep.closure.get('costs')}"
        )
        print(
            f"      single-loop arm (one state across the family "
            f"boundary): {rep.single_loop_verified}"
        )
        if rep.single_loop_compensated:
            print(
                f"      compensated carry (reindexed boundary wires): "
                f"{rep.single_loop_compensated}"
            )
            if rep.boundary:
                print(
                    f"      boundary: block {rep.boundary.get('at')} "
                    f"carried-cls={rep.boundary.get('carried_eid')} "
                    f"vs target-cls={rep.boundary.get('target_eid')}"
                )
    else:
        print(
            "      NO loop proposed — the obstructions above are the "
            "measured cause"
        )
    if numeric is not None and ok:
        rep.max_abs_diff = numeric(rep)
        print(
            f"      numeric max|Δ| (rebuilt vs winner): "
            f"{rep.max_abs_diff:.3e}"
        )
    return rep

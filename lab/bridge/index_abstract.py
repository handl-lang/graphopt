"""Parametric abstraction over a per-block index (ADR-0008 gate 3).

Gate 3 measured FAIL with a *syntactic* proposing rule
(:mod:`lab.bridge.loop_propose`): the winner's projection has no
``unroll(step, n)`` subterm, because the export's unrolled recurrence is
*parametrically* regular — each block's step differs by the block index
(``split(…, index=i)``) — but not *syntactically* regular, so there is no
shared ``step`` subterm for the rule to bind.

This module is the piece that retro named as the missing move:
**abstraction over the step**.  A *parametric abstraction* binds the
varying per-block constant (the index) as a **parameter** rather than a
baked-in constant, turning ``K`` distinct subterms into **one shared
subterm applied ``K`` times** — the presentation's ``unroll(step, K)``
spelling, which :data:`~lab.bridge.loop_propose.PROPOSE_LOOP` then folds
to ``Loop(step, K)``.

THE MOVE
    Given a term whose output is a ``concat`` of ``K`` per-block parts
    (the natural per-block decomposition of a chunked scan):

    1. **Decompose** — flatten the ``concat`` chain into the ``K`` block
       outputs; find the ``split(…, index=i)`` family.
    2. **Abstract the index and the width** — in each block, replace its
       own indexed ``split`` with a fresh *block-input* generator, and
       replace every subterm shared with an earlier block (the recurrence
       state) with a fresh *state* generator.  The index and the state are
       now **parameters**, not constants.  A block's *own* split is
       replaced wholesale, so its ``sizes`` (the block **width**) rides on
       the input wire; the width of a node that *survives* into a template
       (an inherited read, a ``decay``/``decay_vec``) is abstracted by
       :data:`WIDTH_ATTRS`, so the move can also ask whether ``K`` blocks
       are the same step *up to the block width* (step 3b).
    3. **Test regularity** — all ``K`` abstracted templates must be the
       same term (hash-consed identity).  This is the whole question: does
       the abstraction actually collapse the ``K`` blocks to one shared
       ``step``?
    3b. **Test width-parametric regularity** — the strictly weaker
       question, and the one the gate-3 retro named: are the ``K`` blocks
       the same step once the block index **and** the block width are
       parameters (:func:`templates_agree_mod`)?  It can explain a
       negative; it is never promoted to a pass (the verdict still needs
       the exact collapse *and* the fold to reproduce).
    4. **Fold** — when regular, emit ``unroll(step, K)`` (the shared step
       applied ``K`` times); otherwise report the *first* difference, so
       the negative names its own cause.

Vocabulary: only ``split``/``Arr``/``Leaf``/``Loop`` — the abstraction
binds a constant as a **parameter wire**, so **no new IR node kind** is
needed (the presentation already has first-class wires and the ``Loop``
combinator; a parameter is just a wire).

The measurement lives in ``lab/bridge/combinator_probe.py``
(``--abstract-index``); the reading is in
``project/retros/gate3-parametric-abstraction.md``.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from mlopt_core.ir import Op, TensorType, Var
from mlopt_discovery import combinator as cb

from lab.bridge.loop_propose import (
    EXPAND_LOOP,
    PROPOSE_LOOP,
    loop_headed,
    saturate_comb,
    unroll,
)

__all__ = [
    "INDEX_ATTRS",
    "PARAM_ATTRS",
    "WIDTH_ATTRS",
    "AbstractionReport",
    "abstract_index",
    "abstract_index_report",
    "block_outputs",
    "index_family",
    "param_families",
    "templates_agree",
    "templates_agree_mod",
]


#: Integer attributes that vary per block in an unrolled chunked scan —
#: the *parametric constants* this move abstracts away.  ``index`` is the
#: split position; ``t``/``offset``/``sign`` are the per-block decay
#: spellings the reassociation re-writes (``xi_C`` outside, ``xi_1``
#: inside — the second confound the gate-3 retro named).
INDEX_ATTRS: tuple[str, ...] = ("index", "t", "offset", "sign")

#: The **block width**.  A chunked scan's ``split`` carries the per-block
#: sizes, and the two chunk-width families of the gate-3 winner differ
#: here (``(4, 4, 4, 4, 16, 16, 16)`` vs ``(16, 16, 16, 16)``).  The
#: gate-3 state-spelling retro named this as the first remaining cause,
#: because the abstraction abstracted the index but not the width.  A
#: *block's own* split already carries its width on the **input wire**
#: the move introduces (the split node is replaced wholesale, so its
#: ``sizes`` never reaches the template) — ``WIDTH_ATTRS`` is what makes
#: the width a parameter for the nodes that *survive* into a template
#: (an inherited read of an earlier block, a ``decay``/``decay_vec``).
WIDTH_ATTRS: tuple[str, ...] = ("sizes",)

#: Every attribute the move treats as a *parameter*: the per-block index
#: and the per-block width.  Two templates that differ only here are the
#: *same step up to its parameters* — the width-parametric reading the
#: gate-3 retro asked for.  See :func:`templates_agree_mod`.
PARAM_ATTRS: tuple[str, ...] = INDEX_ATTRS + WIDTH_ATTRS

#: Placeholder type for the parameter wires the abstraction introduces.
#: The move is structural — the parameters carry no value, so an empty
#: shape is honest; only the *names* (``@S``, ``@x0``, …) matter.
_PH = TensorType(())


def _state_var() -> Var:
    """Return the recurrence-state parameter (the ``Loop`` state wire)."""
    return Var("@S", _PH)


def _input_var(j: int) -> Var:
    """Return the ``j``-th abstracted block-input parameter."""
    return Var(f"@x{j}", _PH)


# ---------------------------------------------------------------------------
#  Decomposition
# ---------------------------------------------------------------------------


def _subterms(root: Any) -> list[Any]:
    """Distinct op subterms of a DAG.

    Keyed on the *term* (``Op`` is hash-consed, so content equality is
    identity) — never on ``id``, whose address reuse is a live hazard.
    """
    seen: set[Any] = set()
    out: list[Any] = []
    stack = [root]
    while stack:
        x = stack.pop()
        if not isinstance(x, Op) or x in seen:
            continue
        seen.add(x)
        out.append(x)
        stack.extend(x.args)
    return out


def block_outputs(root: Any) -> list[Any]:
    """Flatten the ``concat`` chain at the root into per-block outputs.

    A chunked scan's output is ``concat(b_0, …, b_{K-1})`` along the
    sequence axis; the leaves of that chain are the ``K`` block outputs.
    A non-``concat`` root is a single (degenerate) block.
    """
    out: list[Any] = []
    stack = [root]
    while stack:
        x = stack.pop()
        if isinstance(x, Op) and x.op == "concat":
            stack.extend(reversed(x.args))
        else:
            out.append(x)
    return out


def index_family(root: Any, dim: int = -2) -> dict[tuple, list[int]]:
    """Group the ``split`` nodes by ``(sizes, dim)`` → sorted indices.

    The per-block ``split(x, index=i)`` nodes of one scan share
    ``(sizes, dim)``; their indices are the block positions.  A family of
    ``K`` indices ``{0..K-1}`` is the signature of a ``K``-block scan.
    """
    fams: dict[tuple, set[int]] = {}
    for x in _subterms(root):
        if isinstance(x, Op) and x.op == "split":
            a = dict(x.attrs)
            if a.get("dim") != dim:
                continue
            fams.setdefault(
                (repr(a.get("sizes")), a.get("dim")), set()
            ).add(a.get("index"))
    return {k: sorted(v) for k, v in fams.items()}


def _is_split(x: Any) -> bool:
    """Return True for a ``split`` node — a per-block slice."""
    return isinstance(x, Op) and x.op == "split"


def _block_splits(blocks: list[Any]) -> list[set[Any]]:
    """``own[i]`` — the ``split`` nodes block ``i`` *introduces*.

    A block's subtree also contains the splits of every earlier block
    (threaded through the recurrence state); those are inherited, not this
    block's index carriers.  The own splits are the new ones.  Keyed on
    the term, not ``id``.
    """
    seen: set[Any] = set()
    own: list[set[Any]] = []
    for b in blocks:
        cur = {s for s in _subterms(b) if _is_split(s)}
        own.append(cur - seen)
        seen |= cur
    return own


def _shared_terms(blocks: list[Any]) -> set[Any]:
    """Subterms appearing in ``>=2`` blocks — the recurrence carrier.

    The state ``S_{i-1}`` is produced once and read by two blocks (the one
    that writes it and the one that reads it), so it is a *shared* term.
    A block-local transform of the state (e.g. the output's ``c·S``) is
    not.
    """
    counts: Counter = Counter()
    for b in blocks:
        counts.update(_subterms(b))
    return {s for s, c in counts.items() if c >= 2}


def _state_inputs(blocks: list[Any], own: list[set[Any]]) -> list[Any]:
    """``state_in[i]`` — the state subterm block ``i`` *consumes*.

    The state is the **outermost shared subterm of the block that carries
    none of the block's own splits** — it is built from earlier blocks
    alone (so it is the ``S_{i-1}`` the step reads), it is read by two
    blocks (the carrier the ``Loop`` threads), and it is outermost (the
    whole accumulated past, not a transform of it).  Block 0's state is
    the *boundary* (an external input, not shared) — ``None``, recovered
    later by alignment.
    """
    shared = _shared_terms(blocks)
    res: list[Any] = [None]
    for i in range(1, len(blocks)):
        subs = _subterms(blocks[i])
        sets = {s: set(_subterms(s)) for s in subs}
        cands = [s for s in subs if not (sets[s] & own[i])]
        outer = [
            s
            for s in cands
            if not any(t is not s and s in sets[t] for t in cands)
        ]
        # outermost wins (the whole accumulated past, not a transform of
        # it); a shared carrier breaks a size tie (the recurrence writes it
        # once and reads it twice).
        res.append(
            max(
                outer,
                key=lambda s: (len(sets[s]), s in shared),
            )
            if outer
            else None
        )
    return res


def _split_slots(
    blocks: list[Any], own: list[set[Any]]
) -> tuple[list[dict[Any, Any]], int]:
    """Map each block's own splits to a stable per-source input slot.

    Returns ``(own_of, n_slots)`` where ``own_of[i]`` maps a split node
    (in block ``i``) to the ``Var`` that carries its abstracted input.
    Splits that share a source across blocks share a slot — that is what
    makes the abstraction *parametric* (one parameter per input, not one
    per block).
    """
    slot_of_src: dict[Any, int] = {}
    own_of: list[dict[Any, Any]] = []
    for b, own_ids in zip(blocks, own, strict=True):
        m: dict[Any, Any] = {}
        for s in _subterms(b):
            if s in own_ids:
                j = slot_of_src.setdefault(s.args[0], len(slot_of_src))
                m[s] = _input_var(j)
        own_of.append(m)
    return own_of, len(slot_of_src)


# ---------------------------------------------------------------------------
#  Parametric abstraction of one block
# ---------------------------------------------------------------------------


def _abstract_block(
    block: Any,
    own: dict[Any, Any],
    state_in: Any,
    memo: dict[Any, Any],
) -> Any:
    """Rebuild ``block`` with the index and the state bound as parameters.

    ``own`` — split nodes → their input ``Var`` (the index abstracted into
    a wire).  ``state_in`` — the state subterm this block *consumes*
    (``None`` at the boundary); it becomes the single state generator
    ``S``, so the block reads a parameter instead of a baked subterm.
    """
    if state_in is not None and block == state_in:
        return _state_var()
    if block in own:
        return own[block]
    if not isinstance(block, Op):
        return block
    if block in memo:
        return memo[block]
    new = Op.make(
        block.op,
        *(_abstract_block(a, own, state_in, memo) for a in block.args),
        **dict(block.attrs),
    )
    memo[block] = new
    return new


def _align(ref: Any, blk: Any) -> Any:
    """Fill ``blk``'s state positions from ``ref`` (``Var("S")`` markers).

    Only the ``S`` positions are unified — a boundary fill for the initial
    state.  Any other difference (an op, an attr, an arity) is left in
    place, so it still shows up as a mismatch.
    """
    if isinstance(ref, Var) and ref.name == "@S":
        return _state_var()
    if not isinstance(ref, Op) or not isinstance(blk, Op):
        return blk
    if ref.op != blk.op or dict(ref.attrs) != dict(blk.attrs):
        return blk
    if len(ref.args) != len(blk.args):
        return blk
    return Op.make(
        ref.op,
        *(
            _align(a, b)
            for a, b in zip(ref.args, blk.args, strict=True)
        ),
        **dict(ref.attrs),
    )


def _leaf_desc(x: Any) -> str:
    """Describe a leaf node for a diff message (type + repr)."""
    return f"{type(x).__name__}({x!r})"


def _param_key(t: Any) -> Any:
    """Canonical structural key of ``t`` with the **param attrs erased**.

    This is the *width-parametric* identity of a template: two templates
    with the same key are the same step up to the block index and the
    block width (:data:`PARAM_ATTRS`).  Only the *values* of those attrs
    are dropped — every op, every child, and every other attribute is
    kept, so a genuine structural difference still separates two keys.
    The key is content-based (``Op`` is hash-consed, so identity and
    content agree) and never touches ``id``.
    """
    if not isinstance(t, Op):
        return t
    a = tuple(
        sorted(
            (k, repr(v))
            for k, v in dict(t.attrs).items()
            if k not in PARAM_ATTRS
        )
    )
    return (t.op, tuple(_param_key(x) for x in t.args), a)


def _first_diff(a: Any, b: Any, path: str) -> str:
    """Describe the first structural difference between two terms."""
    if not isinstance(a, Op) or not isinstance(b, Op):
        if a == b:
            return ""
        return f"{path}: {_leaf_desc(a)} vs {_leaf_desc(b)}"
    if a.op != b.op:
        return f"{path}: op {a.op} vs {b.op}"
    da, db = dict(a.attrs), dict(b.attrs)
    if da != db:
        return f"{path}: {a.op} attrs {da} vs {db}"
    if len(a.args) != len(b.args):
        return f"{path}: arity {len(a.args)} vs {len(b.args)}"
    for i, (ca, c_other) in enumerate(zip(a.args, b.args, strict=True)):
        d = _first_diff(ca, c_other, f"{path}/{a.op}[{i}]")
        if d:
            return d
    return ""


def _first_diff_mod(a: Any, b: Any, path: str) -> str:
    """First difference between two terms **modulo** the param attrs.

    :func:`_first_diff` with :data:`PARAM_ATTRS` erased, so the message
    names the first *non-parametric* difference — the reason the width
    and the index cannot be the whole story.
    """
    if not isinstance(a, Op) or not isinstance(b, Op):
        if a == b:
            return ""
        return f"{path}: {_leaf_desc(a)} vs {_leaf_desc(b)}"
    if a.op != b.op:
        return f"{path}: op {a.op} vs {b.op}"
    da = {
        k: v for k, v in dict(a.attrs).items() if k not in PARAM_ATTRS
    }
    db = {
        k: v for k, v in dict(b.attrs).items() if k not in PARAM_ATTRS
    }
    if da != db:
        return f"{path}: {a.op} attrs {da} vs {db}"
    if len(a.args) != len(b.args):
        return f"{path}: arity {len(a.args)} vs {len(b.args)}"
    for i, (ca, c_other) in enumerate(zip(a.args, b.args, strict=True)):
        d = _first_diff_mod(ca, c_other, f"{path}/{a.op}[{i}]")
        if d:
            return d
    return ""


def _abstract_blocks(
    root: Any,
) -> tuple[list[Any], list[Any], list[dict[Any, Any]], list[Any]]:
    """Abstract every block once; return the shared intermediate state.

    Returns ``(blocks, state_in, own_of, raw)`` — the decomposition, the
    per-block consumed state, the own-split→wire map, and the abstracted
    templates.  Factored out so the exact and the width-parametric
    agreement checks share one abstraction (they must not disagree about
    what a block *is*).
    """
    blocks = block_outputs(root)
    own_ids = _block_splits(blocks)
    state_in = _state_inputs(blocks, own_ids)
    own_of, _ = _split_slots(blocks, own_ids)
    raw = [
        _abstract_block(b, own, si, {})
        for b, own, si in zip(blocks, own_of, state_in, strict=True)
    ]
    return blocks, state_in, own_of, raw


def _agree(
    raw: list[Any],
    key: Any,
    diff: Any,
) -> tuple[bool, list[Any], str]:
    """Group ``raw`` templates by ``key``; return ``(agree, templates, diff)``.

    ``key`` decides what "the same template" means (exact identity, or
    identity modulo the param attrs); ``diff`` names the first difference
    the same way.  The interior mismatch is reported first — blocks
    ``1..K-1`` are the full (non-degenerate) steps — then the boundary
    block 0, whose state is an external input.
    """
    groups: dict[Any, list[int]] = {}
    for i, t in enumerate(raw):
        groups.setdefault(key(t), []).append(i)
    if len(groups) == 1:
        return True, raw, ""
    ref = max(
        groups, key=lambda k: (len(groups[k]), 0 not in groups[k])
    )
    ref_raw = next(t for t in raw if key(t) == ref)
    aligned = [t if key(t) == ref else _align(ref_raw, t) for t in raw]
    for i, t in enumerate(aligned[2:], start=2):
        if key(t) != key(aligned[1]):
            return (
                False,
                aligned,
                diff(aligned[1], t, f"block1/block{i}"),
            )
    for i, t in enumerate(aligned[1:], start=1):
        if key(t) != key(aligned[0]):
            return (
                False,
                aligned,
                diff(aligned[0], t, f"block0/block{i}"),
            )
    return True, aligned, ""


def templates_agree(
    root: Any, dim: int = -2
) -> tuple[bool, list[Any], str]:
    """Abstract every block; return ``(agree, templates, diff)``.

    ``agree`` is the move's whole question: after binding the index and the
    state as parameters, are all ``K`` block templates the *same* term?
    ``diff`` names the first structural difference when they are not
    (``""`` when they agree).

    The initial state is a *boundary*: block 0's state is an input, not a
    subterm shared with an earlier block, so it carries no ``S`` marker.
    It is aligned against the majority template (its state position is
    filled by ``S``) — a boundary fill, not a free pass: a genuine
    structural difference between block 0 and the rest survives the
    alignment and is reported.
    """
    blocks, _state_in, _own_of, raw = _abstract_blocks(root)
    if len(blocks) < 2:
        return False, blocks, "fewer than two blocks"
    return _agree(raw, key=lambda t: t, diff=_first_diff)


def templates_agree_mod(
    root: Any, dim: int = -2
) -> tuple[bool, list[Any], str]:
    """As :func:`templates_agree`, but **modulo the param attrs**.

    This is the *width-parametric* reading the gate-3 retro asked for:
    are the ``K`` blocks the same step once the block index **and** the
    block width are treated as parameters (:data:`PARAM_ATTRS`)?  It is
    strictly weaker than :func:`templates_agree` — an exact collapse
    implies a parametric one — so it can only ever *explain* an exact
    negative, never manufacture a positive: the verdict still requires
    the exact templates to collapse *and* the fold to reproduce.
    """
    blocks, _state_in, _own_of, raw = _abstract_blocks(root)
    if len(blocks) < 2:
        return False, blocks, "fewer than two blocks"
    return _agree(raw, key=_param_key, diff=_first_diff_mod)


def param_families(root: Any) -> dict[Any, list[int]]:
    """Return the distinct param-attr *values* per template, as text.

    Reports which block widths / indices actually appear on the
    abstracted templates, so the negative can name the parameter it is
    up against (e.g. ``sizes=(4, 4, 4, 4, 16, 16, 16)`` vs
    ``sizes=(16, 16, 16, 16)``).  Values are deduplicated — one entry per
    distinct ``(attr, value)`` pair, not one per occurrence.
    """
    _blocks, _state_in, _own_of, raw = _abstract_blocks(root)
    fams: dict[Any, list[int]] = {}
    for i, t in enumerate(raw):
        vals = tuple(
            sorted({(k, repr(v)) for k, v in _param_values(t)})
        )
        fams.setdefault(vals, []).append(i)
    return fams


def _param_values(t: Any) -> list[tuple[str, Any]]:
    """Every ``(attr, value)`` with a param-attr key anywhere in ``t``."""
    out: list[tuple[str, Any]] = []
    seen: set[Any] = set()
    stack = [t]
    while stack:
        x = stack.pop()
        if not isinstance(x, Op) or x in seen:
            continue
        seen.add(x)
        for k, v in dict(x.attrs).items():
            if k in PARAM_ATTRS:
                out.append((k, v))
        stack.extend(x.args)
    return out


# ---------------------------------------------------------------------------
#  Compile a template to a step morphism  S ⊗ X → S ⊗ Y
# ---------------------------------------------------------------------------


def _compile_with_inputs(root: Any, inputs: list[Any]) -> Any:
    """Compile a DAG to a morphism that *consumes* ``inputs`` as wires.

    The bridge's :func:`~mlopt_discovery.combinator.dag_to_comb` compiles
    ``I → outputs`` (every leaf is a generator).  A ``Loop`` step needs
    ``S ⊗ X → S ⊗ Y``: the state and the block inputs arrive as *wires*.
    This is the same postorder/mux compilation with the context seeded by
    the input wires; any *other* leaf is a closed generator inside the
    step (prepended to the context).  A ``pack`` root is the output
    bundle (``⊗``), exactly as ``dag_to_comb`` reads it.
    """
    order = cb._postorder(root)
    is_pack = isinstance(root, Op) and root.op == "pack"
    nodes = order[:-1] if is_pack else order
    outputs = list(root.args) if is_pack else [root]
    leaves: list[Any] = []
    seen: set[Any] = set(inputs)
    for n in nodes:
        for a in n.args:
            if not isinstance(a, Op) and a not in seen:
                seen.add(a)
                leaves.append(a)
    for a in outputs:
        if not isinstance(a, Op) and a not in seen:
            seen.add(a)
            leaves.append(a)
    total: Counter = Counter()
    for n in nodes:
        total.update(n.args)
    total.update(outputs)
    remaining = Counter(total)
    ctx = list(leaves) + list(inputs)
    morphs: list = [_par_leaves(leaves)]
    for node in nodes:
        args = list(node.args)
        use = Counter(args)
        maps: list = []
        slots: list[tuple[Any, str]] = []
        for w in ctx:
            u = use.get(w, 0)
            rem_after = remaining[w] - u
            if u and rem_after:
                maps.append(cb.Dup(u + 1))
            elif u > 1:
                maps.append(cb.Dup(u))
            else:
                maps.append(cb.Id(1))
            slots.extend((w, "arg") for _ in range(u))
            if rem_after or u == 0:
                slots.append((w, "live"))
        for w in args:
            remaining[w] -= 1
        arg_pos: dict[Any, deque] = {}
        for i, (w, role) in enumerate(slots):
            if role == "arg":
                arg_pos.setdefault(w, deque()).append(i)
        order_pi = [arg_pos[w].popleft() for w in args]
        order_pi += [i for i, (_, r) in enumerate(slots) if r == "live"]
        live = [w for w, r in slots if r == "live"]
        feed = cb._mux(maps, order_pi)
        step_parts = ([] if isinstance(feed, cb.Id) else [feed]) + [
            cb._par(cb._arr(node), cb.Id(len(live)))
        ]
        morphs.append(cb._seq(*step_parts))
        ctx = [node, *live]
    out_use = Counter(outputs)
    maps = []
    slots = []
    for w in ctx:
        u = out_use.get(w, 0)
        if u == 0:
            maps.append(cb.Drop())
            continue
        maps.append(cb.Dup(u) if u > 1 else cb.Id(1))
        slots.extend((w, "out") for _ in range(u))
    pos: dict[Any, deque] = {}
    for i, (w, _r) in enumerate(slots):
        pos.setdefault(w, deque()).append(i)
    order_pi = [pos[w].popleft() for w in outputs]
    close = cb._mux(maps, order_pi)
    if not (isinstance(close, cb.Id) and close.n == len(ctx)):
        morphs.append(close)
    return cb._seq(*morphs)


def _par_leaves(leaves: list[Any]) -> Any:
    """``par(Leaf(l)…)`` for the closed generators, or the unit map."""
    if not leaves:
        return cb.Id(0)
    return cb._par(*(cb.Leaf(lf) for lf in leaves))


# ---------------------------------------------------------------------------
#  The move
# ---------------------------------------------------------------------------


@dataclass
class AbstractionReport:
    """The result of the parametric-abstraction move on one term."""

    n_blocks: int
    families: dict = field(default_factory=dict)
    agree: bool = False
    diff: str = ""
    #: Agreement *modulo the param attrs* — the same step up to the block
    #: index and the block **width** (:data:`PARAM_ATTRS`).  Strictly
    #: weaker than :attr:`agree`; it explains a negative, it never
    #: replaces it (the verdict still needs :attr:`agree`).
    agree_mod: bool = False
    diff_mod: str = ""
    #: Which param-attr values actually appear per template (the widths
    #: and indices the negative is up against).
    param_families: dict = field(default_factory=dict)
    n_inputs: int = 0
    #: ``unroll(step, K)`` when the abstraction collapsed the blocks;
    #: ``None`` when it did not (the gate-3 negative).
    morph: Any = None
    step: Any = None
    roundtrip: bool = False
    loop_headed: bool = False

    @property
    def verdict(self) -> str:
        """``"regular"`` when the blocks collapse *and* the fold reproduces.

        Templates agreeing is necessary but not sufficient: the abstracted
        step must also expand back to the original blocks (``roundtrip``).
        A family that agrees but whose fold does not reproduce is reported
        ``irregular`` — the move did not reach the loop.
        """
        return (
            "regular"
            if (self.agree and self.roundtrip)
            else "irregular"
        )


def abstract_index(root: Any, dim: int = -2) -> AbstractionReport:
    """Run the parametric-abstraction move on a term.

    Binds the per-block index (and the recurrence state) as parameters; if
    the ``K`` blocks then collapse to one shared ``step``, returns the
    ``unroll(step, K)`` spelling (the loop-carrying form).  Otherwise the
    report names the first difference — the negative's cause — and, when
    the blocks are the same step *modulo the block width and index*
    (:func:`templates_agree_mod`), says so: that is the width-parametric
    reading, and it is recorded without ever being promoted to a pass.
    """
    blocks = block_outputs(root)
    rep = AbstractionReport(
        n_blocks=len(blocks), families=index_family(root, dim)
    )
    agree, templates, diff = templates_agree(root, dim)
    rep.agree, rep.diff = agree, diff
    agree_mod, _templates_mod, diff_mod = templates_agree_mod(root, dim)
    rep.agree_mod, rep.diff_mod = agree_mod, diff_mod
    rep.param_families = param_families(root)
    if not agree:
        return rep
    own_ids = _block_splits(blocks)
    state_in = _state_inputs(blocks, own_ids)
    own_of, n_slots = _split_slots(blocks, own_ids)
    rep.n_inputs = n_slots
    state_init = _state_input(templates[0], blocks[0], own_of[0])
    inputs = [_state_var()] + [_input_var(j) for j in range(n_slots)]
    # the step's state output S' is the state block 1 *consumes* (S_0),
    # with block 0's own index and the boundary state abstracted — so the
    # step reads S and returns S'
    if state_in[1] is not None:
        so = _abstract_block(state_in[1], own_of[0], state_init, {})
        root_tmpl = Op.make("pack", so, templates[0])
    else:
        root_tmpl = templates[0]
    step = _compile_with_inputs(root_tmpl, inputs)
    rep.step = step
    morph = unroll(step, len(blocks))
    rep.morph = morph
    rep.loop_headed = loop_headed(morph)
    rep.roundtrip = _roundtrips(morph, blocks, state_init, own_of)
    return rep


def _state_input(
    tmpl: Any,
    block: Any,
    own: dict[Any, Any],
) -> Any:
    """Return the initial state consumed by block 0 — the alignment fill.

    The aligned template carries ``Var("S")`` where block 0's state sits;
    walking template and block in parallel recovers the original subterm.
    """
    if isinstance(tmpl, Var) and tmpl.name == "@S":
        return block
    if block in own:
        return None
    if not isinstance(tmpl, Op) or not isinstance(block, Op):
        return None
    for a, b in zip(tmpl.args, block.args, strict=False):
        r = _state_input(a, b, own)
        if r is not None:
            return r
    return None


def _roundtrips(
    morph: Any,
    blocks: list[Any],
    state_init: Any,
    own_of: list[dict[Any, Any]],
) -> bool:
    """Apply the fold to the real inputs; compare to the original blocks.

    ``unroll(step, K)`` consumes ``[S, x_0…, x_{K-1}…]``; the wires are
    the original initial state followed by each block's own splits in slot
    order.  Its outputs ``[S_K, o_0…o_{K-1}]`` must reproduce the blocks.
    """
    if state_init is None:
        return False
    wires: list[Any] = [state_init]
    for i, block in enumerate(blocks):
        pairs = [
            (own_of[i][s], s)
            for s in _subterms(block)
            if s in own_of[i]
        ]
        pairs.sort(key=lambda kv: _var_slot(kv[0]))
        wires.extend(s for _v, s in pairs)
    try:
        res = cb._apply(morph, wires)
    except Exception:  # a failed expansion is simply not a match
        return False
    return res[1:] == blocks


def _var_slot(v: Any) -> int:
    """Numeric slot of an ``@x{j}`` parameter var (for stable ordering)."""
    name = v.name if isinstance(v, Var) else ""
    tail = name[2:] if name.startswith("@x") else ""
    return int(tail) if tail.isdigit() else 0


# ---------------------------------------------------------------------------
#  The measurement
# ---------------------------------------------------------------------------


def abstract_index_report(root: Any, control: Any = None) -> None:
    """Print the gate-3 parametric-abstraction measurement for a term.

    ``root`` is the search's extracted winner (the term gate 3 decides on).
    ``control`` — when given — is a *regular* ``K``-block fold, which shows
    the move is well-formed independently of whether the winner is regular.
    """
    rep = abstract_index(root)
    print(
        f"   abstract-index: blocks={rep.n_blocks} "
        f"split-families={len(rep.families)}"
    )
    for key, idxs in rep.families.items():
        print(f"      family sizes/dim={key} indices={idxs}")
    if rep.verdict == "regular":
        st = cb.morph_stats(rep.step)
        mt = cb.morph_stats(rep.morph)
        print(
            f"      verdict=REGULAR — the {rep.n_blocks} blocks collapse to "
            f"ONE shared step ({st.nodes} nodes, {rep.n_inputs} inputs)"
        )
        res = saturate_comb(rep.morph, [PROPOSE_LOOP, EXPAND_LOOP])
        print(
            f"      unroll(step,{rep.n_blocks})={mt.nodes} nodes  "
            f"round-trip={rep.roundtrip}  "
            f"closure: fired={res['fired']} "
            f"best_is_loop={loop_headed(res['best'])} "
            f"best_cost={res['cost']}"
        )
    else:
        print(
            f"      verdict=IRREGULAR — the {rep.n_blocks} blocks do NOT "
            "collapse to one shared step"
        )
        if rep.agree:
            print(
                "      the templates agree but the fold does not reproduce "
                "the term — the abstracted state is not the recurrence state"
            )
        print(f"      first difference: {rep.diff}")
        if rep.agree_mod:
            print(
                "      ... but they ARE the same step modulo the block "
                "width and index (sizes/index abstracted): the width is a "
                "parameter the fixed-shape IR cannot carry, so no fold is "
                "built"
            )
        else:
            print(
                "      ... and they are NOT the same step even modulo the "
                "block width and index — the difference is structural, not "
                "a width/index parameter"
            )
            print(
                f"      first difference mod width/index: {rep.diff_mod}"
            )
        for vals, idxs in rep.param_families.items():
            shown = ", ".join(f"{k}={v}" for k, v in vals) or "(none)"
            print(f"      param values blocks={idxs}: {shown}")
    if control is not None:
        crep = abstract_index(control)
        line = (
            f"   control (regular fold): blocks={crep.n_blocks} "
            f"verdict={crep.verdict} round-trip={crep.roundtrip}"
        )
        if crep.verdict == "regular":
            res = saturate_comb(crep.morph, [PROPOSE_LOOP, EXPAND_LOOP])
            line += (
                f"  closure: best_is_loop={loop_headed(res['best'])} "
                f"best_cost={res['cost']}"
            )
        print(line)

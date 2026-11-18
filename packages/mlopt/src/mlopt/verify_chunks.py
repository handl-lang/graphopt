"""Chunked ``export --verify-run`` — segmented ``hnl-run`` evaluation.

The monolithic verify program binds *every* ``Param`` leaf in one
``do`` block, so ``hnl-run`` holds the whole weight set live for the
whole eval — the arch-zoo sweep's two largest specs (tinyllama-1.1b
4.40 GB, whisper-large-v3-turbo-encoder 2.55 GB F32 sidecars) were
earlyoom-killed on an 11.6 GB host before reaching steady state
(ADR-0179 design note, option (c)).  The fix is scheduling, not wire:
split the computation DAG into contiguous topo-ordered segments under
a per-chunk bound-tensor byte budget, evaluate each segment as its own
``hnl-run`` invocation, and hand intermediate tensors between chunks
through F64 safetensors files — the same ``tensor.load`` channel the
weight sidecar already uses.  The final chunk's output is the program
result and goes through the *unchanged* torch-reference compare, so
the verified-or-declined contract is preserved: a wrong intermediate
cannot hide, it propagates into the logits.

Honesty rules:

* Chunk boundaries are a pure function of the (carrier-expanded) term
  — deterministic topo order, canonical leaf-name sanitization, never
  hash-order — so the same term always produces the same chunks.
* Inter-chunk handoffs are F64 safetensors: the ``__tensor__`` wire is
  already f64 and shortest-roundtrip JSON preserves it bit-exactly, so
  no unnamed quantization enters the chain (F32 handoffs would inject
  one the tolerance does not name — deliberately not done).
* A term that admits no budget-respecting partition raises
  :class:`ChunkPlanError`; the caller falls back to the monolithic
  eval (the status quo) rather than claiming a verification it did
  not run.
* ``{"hnl"}`` staged feed expressions may only reference binding names
  bound earlier *in the same chunk*; anything else declines.

Env knob: ``MLOPT_VERIFY_CHUNK_BYTES`` — the per-chunk bound-tensor
byte budget (default 1 GiB).  ``export --verify-run`` engages the
chunked arm when the sidecar payload (or, absent a sidecar file, the
estimated feed weight) exceeds the budget; ``0``/negative disables
chunking entirely.  ``MLOPT_HNL_RUN_TIMEOUT`` applies per chunk.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlopt_core.term import Op

from mlopt.carrier_expand import expand_carriers
from mlopt.emit import (
    _SHARED_MIN_REFS,
    _SIDECAR_MIN_ELEMS,
    _child_ops,
    _feed_expr,
    _leaf_key,
    _renderer,
    _sanitize,
    _Sidecar,
    leaf_kinds,
    write_safetensors,
)

__all__ = [
    "CHUNK_BYTES_DEFAULT",
    "CHUNK_BYTES_ENV",
    "ChunkPlanError",
    "ChunkSpec",
    "chunk_byte_budget",
    "emit_chunk_programs",
    "estimate_feed_bytes",
    "plan_verify_chunks",
    "run_chunk_programs",
]


class ChunkPlanError(ValueError):
    """The term admits no budget-respecting chunk partition.

    The caller treats this as "chunked verify unavailable" and falls
    back to the monolithic eval — an honest decline, never a silent
    partial verification.
    """


CHUNK_BYTES_ENV = "MLOPT_VERIFY_CHUNK_BYTES"
#: Default per-chunk bound-tensor budget: 1 GiB of payload bytes.  On
#: the 11.6 GB earlyoom host this leaves ~2-3x headroom for activations
#: and the f64 widen transients each op call makes (the llama-22L
#: prototype ran ~0.8 GB weight sets per chunk comfortably).
CHUNK_BYTES_DEFAULT = 1 << 30

_SAFETENSORS_ITEMSIZE = {"F64": 8, "F32": 4}


def chunk_byte_budget() -> int:
    """Per-chunk bound-tensor byte budget from ``$MLOPT_VERIFY_CHUNK_BYTES``.

    Default :data:`CHUNK_BYTES_DEFAULT` (1 GiB); ``0`` or a negative
    value disables chunking (monolithic eval always); an unparsable
    value is a named ``ValueError`` — never a silent default.
    """
    raw = os.environ.get(CHUNK_BYTES_ENV)
    if raw is None or not raw.strip():
        return CHUNK_BYTES_DEFAULT
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(f"{CHUNK_BYTES_ENV} must be an integer byte count, got {raw!r}") from None


# ---------------------------------------------------------------------------
#  DAG analysis — topo order, direct leaves, byte accounting
# ---------------------------------------------------------------------------


def _walk_leaves(v: Any):
    """Yield ``(name, kind)`` for ``Var``/``Param`` values inside *v``.

    Descends list/tuple/dict wrappers (shape specs and attr payloads
    can carry terms); an ``Op`` yields *nothing* — its leaves belong to
    that node's own accounting.  Mirrors ``mlopt.emit._op_subterms``
    coverage (args + attrs) without materializing anything.
    """
    k = _leaf_key(v)
    if k is not None:
        yield k
        return
    if isinstance(v, Op):
        return
    if isinstance(v, (list, tuple)):
        for x in v:
            yield from _walk_leaves(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _walk_leaves(x)


def _direct_leaves(t: Op) -> list[tuple[str, str]]:
    """``(name, kind)`` leaves attached to *t* itself (args + attrs), deduped."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for v in (*t.args, *t.attrs.values()):
        for k in _walk_leaves(v):
            if k[0] not in seen:
                seen.add(k[0])
                out.append(k)
    return out


def _topo(root: Any) -> tuple[list[Op], dict[Op, int]]:
    """Every ``Op`` node reachable from *root*, children before parents.

    Same iterative post-order shape as ``mlopt.emit._bind_order`` but
    over the whole DAG (no bind filter); deterministic because
    ``_child_ops`` iterates args then attrs in construction order and
    ``Op.make`` hash-conses structurally identical nodes to one object.
    """
    order: list[Op] = []
    seen: set[Op] = set()
    stack: list[tuple[Any, bool]] = [(root, False)]
    while stack:
        t, expanded = stack.pop()
        if not isinstance(t, Op):
            continue
        if expanded:
            order.append(t)
            continue
        if t in seen:
            continue
        seen.add(t)
        stack.append((t, True))
        for c in _child_ops(t):
            stack.append((c, False))
    return order, {t: i for i, t in enumerate(order)}


def _entry_numel(f: Any) -> int | None:
    """Element count of a feed value *without* materializing its data.

    ``{"hnl"}`` computed expressions and unrecognised values return
    ``None`` (counted as 0 bytes — they are typically small computed
    tensors or ``tensor.load`` probes; the weight channel dominates the
    budget either way).
    """
    if isinstance(f, Mapping):
        if "hnl" in f or "shape" not in f:
            return None
        try:
            n = 1
            for d in f["shape"]:
                n *= int(d)
            return n
        except (TypeError, ValueError):
            return None
    numel = getattr(f, "numel", None)  # torch.Tensor
    if callable(numel):
        try:
            return int(numel())
        except Exception:  # noqa: BLE001 — exotic tensor-like
            pass
    size = getattr(f, "size", None)  # numpy.ndarray
    if isinstance(size, int):
        return size
    shape = getattr(f, "shape", None)
    if shape is not None:
        try:
            n = 1
            for d in shape:
                n *= int(d)
            return n
        except (TypeError, ValueError):
            pass
    try:
        return len(f)  # flat sequence fallback (nested lists undercount)
    except TypeError:
        return None


def _leaf_bytes(name: str, kind: str, feed: Mapping[str, Any], param_itemsize: int) -> int:
    """Bound-tensor bytes one feed leaf contributes to a chunk's live set.

    *param_itemsize* is the sidecar payload itemsize (4 for ``"F32"``,
    8 for ``"F64"``/no sidecar).  ``Param`` leaves below
    ``_SIDECAR_MIN_ELEMS`` emit inline f64 ``from_list`` literals
    regardless of sidecar dtype, so they always count 8 B/elem; ``Var``
    leaves never sidecar (inline f64 or a ``{"hnl"}`` expression → 0).
    """
    f = feed.get(name)
    if f is None:
        return 0
    numel = _entry_numel(f)
    if numel is None:
        return 0
    if kind == "param" and numel >= _SIDECAR_MIN_ELEMS and param_itemsize < 8:
        return numel * param_itemsize
    return numel * 8


def estimate_feed_bytes(
    feed: Mapping[str, Any],
    kinds: list[tuple[str, str]],
    *,
    param_itemsize: int = 8,
) -> int:
    """Total bound-tensor bytes of *feed* over *kinds* — the engage estimate.

    Used when the sidecar file does not exist yet (or ``--sidecar`` was
    not passed): approximates what a monolithic ``hnl-run`` would hold
    live.  Best-effort — ``{"hnl"}`` feeds count 0.
    """
    return sum(_leaf_bytes(n, k, feed, param_itemsize) for n, k in kinds)


def _segment(
    order: list[Op],
    leaves_of: Any,
    leaf_bytes: Any,
    budget: int,
) -> list[list[Op]]:
    """Greedy contiguous topo partition under the per-chunk byte budget.

    Each segment's weight is the union of its nodes' *direct* leaf
    deps — leaves attached to earlier-segment nodes are not re-bound
    (they crossed the boundary as handoff tensors).  A single node whose
    own leaves exceed *budget* cannot be helped by splitting →
    :class:`ChunkPlanError`.
    """
    segs: list[list[Op]] = []
    cur: list[Op] = []
    cur_leaves: set[str] = set()
    cur_bytes = 0
    for t in order:
        tl = leaves_of(t)
        add = 0
        for n, _k in tl:
            if n not in cur_leaves:
                add += leaf_bytes(n)
        if cur and cur_bytes + add > budget:
            segs.append(cur)
            cur = []
            cur_leaves = set()
            cur_bytes = 0
            add = sum(leaf_bytes(n) for n, _k in tl)
        own = sum(leaf_bytes(n) for n, _k in tl)
        if not cur and own > budget:
            raise ChunkPlanError(
                f"single node {t.op!r} binds {own} bytes of feed tensors, "
                f"over the {budget}-byte chunk budget — no partition can "
                "respect it"
            )
        cur.append(t)
        for n, _k in tl:
            cur_leaves.add(n)
        cur_bytes += add
    if cur:
        segs.append(cur)
    return segs


def _analyze(
    term: Any,
    feed: Mapping[str, Any],
    *,
    budget: int,
    param_itemsize: int,
):
    """Shared plan state: ``(expanded, order, idx, segments)``."""
    expanded = expand_carriers(term)
    if not isinstance(expanded, Op):
        raise ChunkPlanError("term root is not an Op — nothing to segment")
    order, idx = _topo(expanded)
    if len(order) <= 1:
        raise ChunkPlanError("term has no segmentable Op DAG")
    _param_names = {n for n, k in leaf_kinds(expanded) if k == "param"}
    leaf_cache: dict[Op, list[tuple[str, str]]] = {}

    def leaves_of(t: Op) -> list[tuple[str, str]]:
        got = leaf_cache.get(t)
        if got is None:
            got = _direct_leaves(t)
            leaf_cache[t] = got
        return got

    byte_cache: dict[str, int] = {}

    def leaf_bytes(name: str) -> int:
        got = byte_cache.get(name)
        if got is None:
            # leaf names are unique across kinds in a term
            # (leaf_kinds dedups by name).
            kind = "param" if name in _param_names else "var"
            got = _leaf_bytes(name, kind, feed, param_itemsize)
            byte_cache[name] = got
        return got

    segments = _segment(order, leaves_of, leaf_bytes, budget)
    return expanded, order, idx, segments


def plan_verify_chunks(
    term: Any,
    feed: Mapping[str, Any],
    *,
    budget: int,
    param_itemsize: int = 8,
) -> list[list[Op]]:
    """Segment *term*'s carrier-expanded DAG into topo-ordered chunks.

    Returns the segments (each a list of ``Op`` nodes in topo order);
    raises :class:`ChunkPlanError` when no budget-respecting partition
    exists.  Pure function of the term + feed shapes — deterministic.
    """
    _expanded, _order, _idx, segments = _analyze(
        term, feed, budget=budget, param_itemsize=param_itemsize
    )
    return segments


# ---------------------------------------------------------------------------
#  Chunk program emission
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkSpec:
    """One chunk's runnable ``.hnl`` program + handoff contract.

    *out_names*/*out_file*: non-final chunks emit their live-out
    tensors as a safetensors handoff (``tensor.load`` names are
    ``o<topo-index>``); the final chunk's ``out_names`` is empty — its
    program result IS the verify output.
    """

    source: str
    out_names: tuple[str, ...]
    out_file: str = ""


def _chunk_out_name(k: int) -> str:
    return f"mlopt-verify-chunk{k}-out.safetensors"


def _handoff_tensor_name(topo_idx: int) -> str:
    return f"o{topo_idx}"


_STR_LIT = re.compile(r'"(?:[^"\\]|\\.)*"')
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _feed_refs(expr: str) -> set[str]:
    """Binding-name candidates referenced by a ``{"hnl"}`` feed expr.

    String literals are blanked first so safetensors filenames inside
    ``tensor.load("x.safetensors", "y")`` cannot false-positive; what
    remains is intersected with the canonical binding-name set by the
    caller.
    """
    return set(_IDENT.findall(_STR_LIT.sub('""', expr)))


def _canonical_names(feed: Mapping[str, Any], term: Any) -> dict[str, str]:
    """Leaf/feed-name → ``.hnl`` binding name, monolithic-prelude order.

    Replicates ``mlopt.emit._prelude``'s sanitize sequence exactly
    (``{"tensor"}`` taken, feed dict order, then first-use leaf order)
    so staged ``{"hnl"}`` expressions see the same names they were
    authored against.
    """
    taken: set[str] = {"tensor"}
    san: dict[str, str] = {}
    for n in feed:
        san[n] = _sanitize(n, taken)
    for n, _k in leaf_kinds(term):
        if n not in san:
            san[n] = _sanitize(n, taken)
    return san


def _select_entries(
    needed: set[str],
    feed: Mapping[str, Any],
    san: Mapping[str, str],
) -> list[tuple[str, Any]]:
    """Feed entries one chunk binds, in feed dict order.

    Starts from the chunk's needed leaf names and closes over
    ``{"hnl"}`` staged expressions that reference them (an extra feed
    entry is pulled in only when a required expression names its
    binding).  Every ``{"hnl"}`` expression's binding-name references
    must resolve to entries bound *earlier* in this chunk's prelude —
    the same contract ``emit_program`` gives the monolithic prelude —
    else :class:`ChunkPlanError`.
    """
    known = set(san.values())
    required = set(needed)
    while True:
        req_san = {san[n] for n in required if n in san}
        added = False
        for name, f in feed.items():
            if name in required or not (isinstance(f, Mapping) and "hnl" in f):
                continue
            if _feed_refs(str(f["hnl"])) & req_san:
                required.add(name)
                added = True
        if not added:
            break
    sel = [(n, f) for n, f in feed.items() if n in required]
    bound_so_far: set[str] = set()
    for name, f in sel:
        if isinstance(f, Mapping) and "hnl" in f:
            bad = (_feed_refs(str(f["hnl"])) & known) - bound_so_far
            if bad:
                raise ChunkPlanError(
                    f"feed entry {name!r} references binding names "
                    f"{sorted(bad)} not bound earlier in the chunk — "
                    "staged feed expressions cannot cross chunk boundaries"
                )
        bound_so_far.add(san[name])
    return sel


def _seg_bind_order(roots: list[Op], bind: set[Op], seg_set: set[Op]) -> list[Op]:
    """*bind* members in dependency order, restricted to *seg_set*.

    Post-order DFS from *roots* descending only through segment nodes
    — frontier (earlier-chunk) nodes are opaque handoff names and are
    never traversed, so each chunk's walk is O(segment), not O(prefix).
    """
    out: list[Op] = []
    pushed: set[Op] = set()
    stack: list[tuple[Any, bool]] = [(r, False) for r in reversed(roots)]
    while stack:
        t, expanded = stack.pop()
        if not isinstance(t, Op) or t not in seg_set:
            continue
        if expanded:
            if t in bind:
                out.append(t)
            continue
        if t in pushed:
            continue
        pushed.add(t)
        stack.append((t, True))
        for c in _child_ops(t):
            if c in seg_set:
                stack.append((c, False))
    return out


def _sidecar_expr_cached(
    sc: _Sidecar, name: str, f: Any, cache: dict[str, str | None]
) -> str | None:
    """``tensor.load``/``None`` for one Param leaf, memoized per emit.

    ``_Sidecar.expr`` re-runs ``_leaf_array`` (a dtype conversion over
    the whole tensor) on every call; a weight shared by several chunks
    would otherwise be converted once per chunk.  The safetensors entry
    is recorded on first call; later calls replay the same spelling.
    """
    if name not in cache:
        cache[name] = sc.expr(name, f)
    return cache[name]


def emit_chunk_programs(
    term: Any,
    feed: Mapping[str, Any],
    *,
    sidecar: str | Path | None,
    sidecar_dtype: str = "F64",
    budget: int,
) -> list[ChunkSpec]:
    """Emit the chunked verify programs for *term* over *feed*.

    Returns one :class:`ChunkSpec` per segment; the LAST spec's program
    result is the verify output (single tensor, same ``__tensor__``
    wire as the monolithic eval).  Non-final specs carry
    ``out_names``/``out_file`` — the safetensors handoff the next chunk
    consumes via ``tensor.load``.  Returns ``[]`` when the term needs
    only one segment (caller uses the monolithic path).  Raises
    :class:`ChunkPlanError` when no partition respects *budget*.

    *sidecar*/*sidecar_dtype* behave exactly as in
    :func:`mlopt.emit.emit_program`: eligible ``Param`` leaves emit
    ``tensor.load("<basename>", "<leaf>")`` and their values are
    written once, after all chunks are emitted, to the sidecar path.
    """
    param_itemsize = _SAFETENSORS_ITEMSIZE.get(sidecar_dtype, 8) if sidecar is not None else 8
    expanded, order, idx, segments = _analyze(
        term, feed, budget=budget, param_itemsize=param_itemsize
    )
    if len(segments) <= 1:
        return []

    seg_of: dict[Op, int] = {}
    for i, seg in enumerate(segments):
        for t in seg:
            seg_of[t] = i

    # Live-out per boundary: segment-k nodes with a parent in a later
    # segment.  Contiguous topo partition guarantees seg(child) <=
    # seg(parent) on every edge, so these are exactly the values that
    # must cross the handoff.
    live_sets: list[set[Op]] = [set() for _ in segments]
    for t in order:
        st = seg_of[t]
        for c in _child_ops(t):
            sc_ = seg_of[c]
            if sc_ < st:
                live_sets[sc_].add(c)
    last = len(segments) - 1
    for k in range(last):
        if not live_sets[k]:
            raise ChunkPlanError(
                f"segment {k} has no live-out tensors — partition is not a valid sequential cut"
            )
    live = [sorted(s, key=idx.get) for s in live_sets]

    kind_of = dict(leaf_kinds(expanded))
    san = _canonical_names(feed, expanded)
    sc = _Sidecar(sidecar, dtype=sidecar_dtype) if sidecar is not None else None
    expr_cache: dict[str, str | None] = {}
    specs: list[ChunkSpec] = []

    for k, seg_nodes in enumerate(segments):
        seg_set = set(seg_nodes)
        is_last = k == last
        roots: list[Op] = [expanded] if is_last else list(live[k])

        # Frontier: earlier-segment Op children of this segment's nodes.
        frontier: dict[Op, None] = {}
        for t in seg_nodes:
            for c in _child_ops(t):
                if isinstance(c, Op) and seg_of[c] < k:
                    frontier[c] = None
        frontier_sorted = sorted(frontier, key=idx.get)
        hnames = {c: f"__h{seg_of[c]}_{idx[c]}" for c in frontier_sorted}
        out_names = tuple(_handoff_tensor_name(idx[r]) for r in roots) if not is_last else ()
        out_file = "" if is_last else _chunk_out_name(k)

        # Share analysis restricted to the segment: refs = parent slots
        # inside the segment + one body occurrence per root.  A node
        # with refs >= _SHARED_MIN_REFS binds once as __sN; refs == 1
        # nodes have exactly one use site and render inline there —
        # no duplication, matching _emit_shared's invariant.
        refs: dict[Op, int] = {}
        for p in seg_nodes:
            for c in _child_ops(p):
                if c in seg_set:
                    refs[c] = refs.get(c, 0) + 1
        for r in roots:
            refs[r] = refs.get(r, 0) + 1
        bind = {t for t in seg_set if refs.get(t, 0) >= _SHARED_MIN_REFS}

        # Needed leaves = direct leaf deps of the segment's nodes.
        needed: set[str] = set()
        for t in seg_nodes:
            for n, _kd in _direct_leaves(t):
                needed.add(n)
        missing = [n for n in needed if n not in feed]
        if missing:
            raise ChunkPlanError(f"feed is missing leaves {missing}")
        sel = _select_entries(needed, feed, san)

        taken: set[str] = {"tensor"} | set(hnames.values())
        bound: dict[str, str] = {}
        lines = ['from "host-tensor" import tensor', "do {"]
        for name, f in sel:
            lname = san[name]
            if lname in taken:
                raise ChunkPlanError(
                    f"binding name {lname!r} collides with a chunk-internal "
                    "name — cannot emit chunk"
                )
            taken.add(lname)
            bound[name] = lname
            expr = None
            if sc is not None and kind_of.get(name) == "param":
                expr = _sidecar_expr_cached(sc, name, f, expr_cache)
            lines.append(f"  {lname} :: {expr if expr is not None else _feed_expr(name, f)}")

        for c in frontier_sorted:
            lines.append(
                f"  {hnames[c]} :: tensor.load("
                f"{json.dumps(_chunk_out_name(seg_of[c]))}, "
                f"{json.dumps(_handoff_tensor_name(idx[c]))})"
            )

        shared: dict[Op, str] = dict(hnames)
        go, render, _taken = _renderer(bound, shared)
        i = 0
        for t in _seg_bind_order(roots, bind, seg_set):
            while f"__s{i}" in taken:
                i += 1
            nm = f"__s{i}"
            i += 1
            shared[t] = nm
            taken.add(nm)
            lines.append(f"  {nm} :: {render(t)}")

        body = ", ".join(go(r) for r in roots)
        if len(roots) > 1:
            body = f"[{body}]"
        lines.append(f"  {body}")
        lines.append("}")
        specs.append(ChunkSpec("\n".join(lines), out_names, out_file))

    if sc is not None:
        sc.write()
    return specs


# ---------------------------------------------------------------------------
#  Chunk execution — ordered hnl-run invocations + safetensors handoffs
# ---------------------------------------------------------------------------


def run_chunk_programs(
    runner: Any,
    specs: list[ChunkSpec],
    workdir: str | Path,
    written: list[Path] | None = None,
) -> Any:
    """Evaluate chunk specs in order; return the final chunk's JSON value.

    Each non-final chunk's result (a ``__tensor__`` wire dict, or a
    list of them for multi-tensor live-outs) is written to its
    ``out_file`` safetensors under *workdir* — F64, the wire's own
    dtype, so handoffs are bit-exact.  *written* collects the handoff
    paths for caller cleanup.  The last spec's result is returned
    untouched — it is the verify output the caller compares.
    """
    from mlopt.emit import KitEvalError

    workdir = Path(workdir)
    n = len(specs)
    for i, spec in enumerate(specs):
        print(f"verify-run: chunk {i + 1}/{n} — hnl-run eval", file=sys.stderr)
        res = runner.run(spec.source)
        if i == n - 1:
            return res
        wires = res if isinstance(res, list) else [res]
        if len(wires) != len(spec.out_names):
            raise KitEvalError(
                f"chunk {i}: program emitted {len(wires)} tensors, expected {len(spec.out_names)}"
            )
        tensors: dict[str, tuple[tuple[int, ...], Any]] = {}
        for name, w in zip(spec.out_names, wires, strict=True):
            t = w.get("__tensor__") if isinstance(w, dict) else None
            if t is None:
                raise KitEvalError(f"chunk {i}: output {name!r} is not a tensor wire ({w!r})")
            tensors[name] = (
                tuple(int(d) for d in t.get("shape") or []),
                list(t.get("data") or []),
            )
        path = workdir / spec.out_file
        write_safetensors(path, tensors, dtype="F64")
        if written is not None:
            written.append(path)
    raise KitEvalError("chunked verify produced no final chunk")  # unreachable

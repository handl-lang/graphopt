"""Elementwise region fusion — ``torch.compile`` delivery for census regions.

The measurement path is ``docs/reports/kokoro_region_fuse.md``: maximal
weakly-connected components of the elementwise op set (the
``genkernel.ELEMENTWISE`` vocabulary, re-spelled locally as
:data:`ELEMENTWISE_OPS` — ``kit_discovery`` is the wrong direction
of the hexagonal boundary) are rewritten, *after* extraction, into
single ``fused_region`` ops whose boundary leaves become positional
runtime args.  Each distinct region *shape* (internal structure only —
leaf identity and ``Const`` values are runtime inputs) compiles to one
``torch.compile``-compiled unrolled body registered under a per-module
``fused_k`` id.  The rewrite is a term→term map proven **bitwise
identical** to the delivered term in spelled mode; compiled mode pays
inductor reassociation noise, which is exactly what the screening +
output gate below exist to bound.

Verify policy (report §5 option (i), strengthened): fusion NEVER
weakens the stock ``lower(verify=True)`` gate.  Three honest layers:

1. **Per-region screening** — before adoption, each distinct kernel's
   compiled body runs once against its spelled body on the REAL
   boundary tensors (the region leaves evaluated through the module's
   own plan tape, so Const minting/dtype semantics match the delivery
   exactly).  A kernel whose spelled-vs-compiled ``verify_equiv``
   report fails ``screen_rtol``/``screen_atol`` is permanently bound
   to its spelled body — bitwise identical to the unfused spelling.
2. **Module output gate** — after the rewrite, the fused module and
   the unfused module are both run on ``example_input`` (RNG streams
   realigned, like ``verify_module``'s seeded pass) and compared at
   ``output_rtol``/``output_atol``.  A miss reverts the module to its
   pre-fusion state entirely — fusion is all-or-nothing per delivery.
3. **Stock verify untouched** — ``sink.verify`` still compares the
   delivered module against a fresh lowering of the certified IR at
   the caller's ``rtol``.  ``output_rtol`` should be chosen ≤ that
   ``rtol``; the measured fused-vs-unfused ``max_rel`` is recorded in
   ``stats["region_fuse"]`` either way — never silent.

Opt-in surface: :class:`mlopt_torch.runners.RegionFusionRunner`
passed as ``lower(..., runner=...)`` (or chained ahead of
``TorchCompileRunner``).  ``IRModule`` itself is unchanged — the
runner reaches the term through ``module._root`` and rebuilds the
root plan, so the verify *reference* lowering stays unfused and the
delivered-vs-reference contract keeps its meaning.

Dynamo recompile limit: every exemplar body is a closure over the
same ``spelled`` code object, so dynamo counts ALL kernels' first
compiles against ONE recompile entry — the probe's shared-closure
bug (8 exemplars silently degrading to eager).  The production fix
(spec §5's alternative): :func:`fuse_module` raises
``torch._dynamo.config.recompile_limit`` to ``8 * n_kernels`` when
the census needs more, and records the before/after in the stats
ledger.  The knob is process-global; raising it only ever *allows*
more recompiles before dynamo falls back to eager — never a
correctness change.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from mlopt_core.ir import Const, Op
from mlopt_core.traversal import ops as term_ops

from mlopt_torch.report import _rng_restore, _rng_snapshot, verify_equiv
from mlopt_torch.torch_eval import IRModule

__all__ = [
    "ELEMENTWISE_OPS",
    "FusionPlan",
    "RegionFuseConfig",
    "collect_regions",
    "fuse_module",
    "region_census",
    "rewrite_term",
]

#: The elementwise vocabulary a region may contain — a local
#: re-spelling of ``kit_discovery.genkernel.ELEMENTWISE`` (28 ops).
#: Kept as a literal: importing the discovery package from the torch
#: backend would invert the hexagonal boundary.  If the discovery
#: table changes, this list must be re-synced deliberately.
ELEMENTWISE_OPS: frozenset[str] = frozenset(
    {
        "abs",
        "add",
        "clamp",
        "clamp_max",
        "clamp_min",
        "cos",
        "div",
        "exp",
        "gelu",
        "identity",
        "log",
        "mish",
        "mul",
        "neg",
        "pow",
        "reciprocal",
        "relu",
        "rsqrt",
        "sigmoid",
        "silu",
        "sin",
        "softplus",
        "softsign",
        "softsign_gate",
        "sqrt",
        "square",
        "sub",
        "tanh",
    }
)


@dataclass(frozen=True)
class RegionFuseConfig:
    """Knobs for :func:`fuse_module` / :class:`RegionFusionRunner`.

    Attributes
    ----------
    min_ops : int
        Minimum region size (member count) to fuse — the fusion-site
        floor; ``2`` matches the census convention (a singleton has
        nothing to fuse).
    compiled : bool
        ``True`` (default) builds ``torch.compile``-compiled bodies;
        ``False`` builds the spelled unrolled bodies only — the
        bitwise-identical control arm (useful for tests and for
        isolating rewrite correctness from inductor numerics).
    dynamic : bool
        Forwarded to ``torch.compile`` — ``False`` (default)
        specializes on first-seen shapes; variable-shape deliveries
        need ``True`` or per-bucket recompiles (documented caveat).
    screen_rtol, screen_atol
        Per-region screening gate: a distinct kernel's compiled body
        must match its spelled body on the real boundary tensors
        within these tolerances (``verify_equiv`` semantics) or the
        kernel binds spelled forever.  ``None`` atol disables the
        absolute leg.
    output_rtol, output_atol
        Whole-module gate: fused-vs-unfused on ``example_input`` must
        pass ``verify_equiv`` at these tolerances or fusion reverts.
        Choose ``output_rtol`` ≤ the ``lower(rtol=...)`` the delivery
        will verify at — the runner cannot see that knob, so the
        caller owns the consistency.

    """

    min_ops: int = 2
    compiled: bool = True
    dynamic: bool = False
    screen_rtol: float = 1e-4
    screen_atol: float | None = None
    output_rtol: float = 1e-4
    output_atol: float | None = None


# ---------------------------------------------------------------------------
#  Region census + eligibility (DAG view — shared subterms counted once)
# ---------------------------------------------------------------------------


def _subtree_ops(root: Any) -> int:
    """Count elementwise-Op descendants of *root* (DAG-deduped)."""
    seen: set[int] = set()
    stack = [root]
    count = 0
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        if isinstance(t, Op):
            if t.op in ELEMENTWISE_OPS:
                count += 1
            stack.extend(t.args)
    return count


def _parents_map(term: Any) -> dict[int, list[Any]]:
    """``id(child Op) -> [parent Ops]`` over the whole term DAG."""
    parents: dict[int, list[Any]] = {}
    for t in term_ops(term):
        for a in t.args:
            if isinstance(a, Op):
                parents.setdefault(id(a), []).append(t)
    return parents


def _components(
    nodes: list[Any], parents: dict[int, list[Any]]
) -> list[list[Any]]:
    """Weakly-connected components of *nodes* over elementwise edges.

    Union-find over ``(child, elementwise-parent)`` pairs — a shared
    subterm belongs to ONE component, not two (the DAG view).
    """
    uf = {id(t): id(t) for t in nodes}

    def find(x: int) -> int:
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            uf[rb] = ra

    for t in nodes:
        for p in parents.get(id(t), ()):
            if p.op in ELEMENTWISE_OPS:
                union(id(t), id(p))

    comps: dict[int, list[Any]] = {}
    for t in nodes:
        comps.setdefault(find(id(t)), []).append(t)
    return list(comps.values())


def _region_record(
    members: list[Any], parents: dict[int, list[Any]]
) -> tuple[Any, list[Any], list[Any]]:
    """``(root, members, boundary_leaf_nodes)`` for one component.

    The root is the member with no elementwise parent; a component
    with several (possible on shared-DAG corners) picks the largest
    ``_subtree_ops`` — eligibility re-derives roots itself and skips
    multi-root components, so this pick only orders the census.
    Boundary leaves are the non-member, non-``Const`` args — *leaf
    identity* for reporting only (canonicalisation re-derives the
    positional leaf list including Consts).
    """
    member_ids = {id(m) for m in members}
    roots = [
        m
        for m in members
        if not any(
            p.op in ELEMENTWISE_OPS for p in parents.get(id(m), ())
        )
    ]
    if len(roots) != 1:
        roots = sorted(roots, key=lambda r: -_subtree_ops(r))
    root = roots[0]
    leaves: dict[int, Any] = {}
    for m in members:
        for a in m.args:
            if id(a) in member_ids or isinstance(a, Const):
                continue
            leaves.setdefault(id(a), a)
    return root, members, list(leaves.values())


def region_census(
    term: Any,
) -> tuple[list[tuple[Any, list[Any], list[Any]]], int]:
    """Maximal elementwise-connex regions of *term*.

    Weakly-connected components over :data:`ELEMENTWISE_OPS` on the
    DAG view (a shared subterm belongs to one component, not two).
    Returns ``(regions, n_elementwise)`` sorted by size desc; each
    region is ``(root, members, boundary_leaf_nodes)``.
    """
    nodes = [t for t in term_ops(term) if t.op in ELEMENTWISE_OPS]
    parents = _parents_map(term)
    regions = [
        _region_record(members, parents)
        for members in _components(nodes, parents)
    ]
    regions.sort(key=lambda r: -len(r[1]))
    return regions, len(nodes)


def _canon_region(
    root: Any, members: list[Any]
) -> tuple[list[Any], list[list[tuple[str, int]]], list[Any]] | None:
    """Topo-order *members* from *root*; return ``(order, specs, leaves)``.

    ``order`` is the post-order member list (root last); ``specs`` is
    the parallel per-member arg spec — ``("m", i)`` reads member
    ``order[i]``'s memo slot, ``("x", i)`` reads positional leaf
    ``i`` — and ``leaves`` is every non-member arg in first-appearance
    order: ``Op``/``Var``/``Param``/``Const`` alike, all runtime tensor
    inputs of the fused op (Consts included — that is what keeps Const
    *tensor* leaves out of the compiled spec, the genkernel inductor
    blocker).  Returns ``None`` when a member is unreachable from the
    root.
    """
    member_ids = {id(m) for m in members}
    order: list[Any] = []
    seen: set[int] = set()

    def dfs(t: Any) -> None:
        if id(t) in seen or id(t) not in member_ids:
            return
        seen.add(id(t))
        for a in t.args:
            if isinstance(a, Op):
                dfs(a)
        order.append(t)

    dfs(root)
    if len(order) != len(members):
        return None
    pos = {id(t): i for i, t in enumerate(order)}
    leaf_pos: dict[int, int] = {}
    leaves: list[Any] = []
    specs: list[list[tuple[str, int]]] = []
    for m in order:
        ms = []
        for a in m.args:
            if isinstance(a, Op) and id(a) in pos:
                ms.append(("m", pos[id(a)]))
            else:
                if id(a) not in leaf_pos:
                    leaf_pos[id(a)] = len(leaves)
                    leaves.append(a)
                ms.append(("x", leaf_pos[id(a)]))
        specs.append(ms)
    return order, specs, leaves


def _freeze_attrs(attrs: Mapping[str, Any]) -> tuple:
    return tuple(sorted((k, repr(v)) for k, v in attrs.items()))


def _region_key(
    order: list[Any], specs: list[list[tuple[str, int]]]
) -> tuple:
    """Dedup key: internal structure only — leaves are positional."""
    return tuple(
        (m.op, _freeze_attrs(m.attrs), tuple(ms))
        for m, ms in zip(order, specs, strict=True)
    )


@dataclass
class FusionPlan:
    """The eligible-region census of one term.

    Attributes
    ----------
    sites : dict
        Region root term -> ``(kid, leaves)`` — the rewrite map.
        ``kid`` indexes the shared compiled-body registry; ``leaves``
        are the boundary nodes in positional order.
    exemplars : dict
        ``kid -> (order, specs)`` — one canonical member list per
        distinct region shape; the compiled body is built from the
        exemplar (any site with the same ``kid`` shares it).
    first_leaves : dict
        ``kid -> leaves`` of the first site seen with that shape —
        the tensors the per-region screen evaluates.
    stats : Counter
        Honest census: ``fused``/``ops_fused`` plus the skip reasons
        ``small``/``multi_root``/``multi_exit``/``unreachable``.

    """

    sites: dict[Any, tuple[int, tuple[Any, ...]]]
    exemplars: dict[int, tuple[list[Any], list[list[tuple[str, int]]]]]
    first_leaves: dict[int, tuple[Any, ...]]
    stats: Counter


def _elem_roots(
    members: list[Any], parents: dict[int, list[Any]]
) -> list[Any]:
    """Members with no elementwise parent — the region's exit candidates."""
    return [
        m
        for m in members
        if not any(
            p.op in ELEMENTWISE_OPS for p in parents.get(id(m), ())
        )
    ]


def _single_exit(
    members: list[Any],
    rt: Any,
    member_ids: set[int],
    parents: dict[int, list[Any]],
) -> bool:
    """Return True iff non-root members' consumers stay inside."""
    for m in members:
        if m is rt:
            continue
        for p in parents.get(id(m), ()):
            if id(p) not in member_ids:
                return False
    return True


def collect_regions(term: Any, *, min_ops: int = 2) -> FusionPlan:
    """Eligible single-exit regions of *term*.

    A region fuses iff it has a UNIQUE root (one member with no
    elementwise parent) and every non-root member's parents are all
    inside the member set — the root is the only exit, so no
    intermediate value is computed twice and no external consumer sees
    a stale spelling.  Everything else is counted and skipped.
    """
    regions, _n = region_census(term)
    parents = _parents_map(term)
    sites: dict[Any, tuple[int, tuple[Any, ...]]] = {}
    keys: dict[tuple, int] = {}
    exemplars: dict[
        int, tuple[list[Any], list[list[tuple[str, int]]]]
    ] = {}
    first_leaves: dict[int, tuple[Any, ...]] = {}
    stats: Counter = Counter()
    for _root, members, _lv in regions:
        if len(members) < min_ops:
            stats["small"] += 1
            continue
        roots = _elem_roots(members, parents)
        if len(roots) != 1:
            stats["multi_root"] += 1
            continue
        rt = roots[0]
        member_ids = {id(m) for m in members}
        if not _single_exit(members, rt, member_ids, parents):
            stats["multi_exit"] += 1
            continue
        canon = _canon_region(rt, members)
        if canon is None:
            stats["unreachable"] += 1
            continue
        order, specs, leaves = canon
        key = _region_key(order, specs)
        if key not in keys:
            keys[key] = len(keys)
            exemplars[keys[key]] = (order, specs)
        kid = keys[key]
        sites[rt] = (kid, tuple(leaves))
        first_leaves.setdefault(kid, tuple(leaves))
        stats["fused"] += 1
        stats["ops_fused"] += len(members)
    return FusionPlan(
        sites=sites,
        exemplars=exemplars,
        first_leaves=first_leaves,
        stats=stats,
    )


# ---------------------------------------------------------------------------
#  Region bodies — spelled closures, compiled per distinct kernel
# ---------------------------------------------------------------------------


def _make_body(
    order: list[Any],
    specs: list[list[tuple[str, int]]],
    bindings: Mapping[str, Callable[..., Any]],
) -> Callable[..., Any]:
    """Mint one unrolled region body — spelled, pre-``torch.compile``.

    ``("m", i)`` arg specs become member memo reads; ``("x", i)``
    become positional leaf reads.  Every boundary leaf — ``Const``s
    included — arrives as a runtime tensor arg, so the compiled graph
    is pure tensor ops (the genkernel inductor FakeTensorMode blocker
    never applies).  All exemplars share this closure's code object —
    see the module docstring's recompile-limit note.
    """
    meta = [(m.op, dict(m.attrs)) for m in order]
    rows = [
        tuple((1 if kind == "m" else 0, v) for kind, v in row)
        for row in specs
    ]
    root_i = len(order) - 1
    n = len(order)
    binds = bindings

    def spelled(*vals: Any) -> Any:
        memo: list[Any] = [None] * n
        for i in range(n):
            op, attrs = meta[i]
            args = [memo[v] if k else vals[v] for k, v in rows[i]]
            memo[i] = binds[op](*args, **attrs)
        return memo[root_i]

    return spelled


#: Shape/dtype specialization budget per distinct kernel when sizing
#: ``recompile_limit`` — one compile per kernel plus headroom for
#: guard variants (``dynamic=False`` specializes per first-seen
#: shape; a second shape is one more recompile).
_RECOMPILE_VARIANTS = 8


def _ensure_recompile_limit(needed: int, rep: dict[str, Any]) -> None:
    """Raise ``torch._dynamo.config.recompile_limit`` to *needed*.

    Every exemplar body shares one code object, so dynamo's per-object
    recompile budget must cover ALL kernels' first compiles (plus
    shape/dtype variants) — the default 8 silently degrades later
    regions to eager (the probe lost ~7.7 ms/call on decoder-nc until
    the limit was raised).  Records the before/after honestly; a
    missing/unsettable knob degrades to a recorded note, never an
    error — fusion still works, later kernels just run eager.
    """
    try:
        # torch's stubs pin ``recompile_limit`` to ``Literal[8]`` — the
        # runtime knob is a plain int; go through ``Any`` deliberately.
        cfg: Any = torch._dynamo.config
        cur = cfg.recompile_limit
        if cur is None or cur < needed:
            cfg.recompile_limit = needed
        rep["recompile_limit"] = {
            "before": cur,
            "after": cfg.recompile_limit,
            "needed": needed,
        }
    except Exception as exc:
        rep["recompile_limit"] = {
            "before": None,
            "after": None,
            "needed": needed,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _make_dispatch(
    registry: Mapping[int, Callable[..., Any]],
) -> Callable[..., Any]:
    """Dispatch ``fused_region`` calls on the ``fused_k`` attr."""

    def dispatch(*xs: Any, fused_k: int) -> Any:
        return registry[fused_k](*xs)

    return dispatch


class _FusedBindings(dict):
    """``dict`` view of the module's bindings plus ``fused_region``.

    ``IRModule._torch_bindings`` is normally the op table's LIVE dict —
    post-construction ``_IR_TO_TORCH[op] = fn`` overrides keep reaching
    already-built modules.  Fusing must not freeze that: this view
    delegates every lookup to the base dict (so overrides still land)
    and answers only ``"fused_region"`` itself, from the per-module
    registry.  ``isinstance(..., dict)`` holds so
    ``device.DevicePolicy.check_module``'s collective-op census still
    sees the real table; ``__bool__`` stays true despite the empty
    dict storage (``eval_term`` treats a falsy bindings mapping as
    ``{}``).
    """

    __slots__ = ("_base", "_fused")

    def __init__(
        self,
        base: Mapping[str, Callable[..., Any]],
        fused_fn: Callable[..., Any],
    ) -> None:
        super().__init__()
        self._base = base
        self._fused = fused_fn

    def __getitem__(self, key: str) -> Callable[..., Any]:
        if key == "fused_region":
            return self._fused
        return self._base[key]  # type: ignore[index]

    def get(self, key: object, default: Any = None) -> Any:
        if key == "fused_region":
            return self._fused
        return self._base.get(key, default)  # type: ignore[attr-defined]

    def __contains__(self, key: object) -> bool:
        return key == "fused_region" or key in self._base

    def __iter__(self):
        yield from self._base
        yield "fused_region"

    def __len__(self) -> int:
        return len(self._base) + 1

    def __bool__(self) -> bool:
        return True

    def keys(self):
        return [*self._base.keys(), "fused_region"]

    def items(self):
        return [(k, self[k]) for k in self]

    def values(self):
        return [self[k] for k in self._base] + [self._fused]


# ---------------------------------------------------------------------------
#  Screening + module rewrite
# ---------------------------------------------------------------------------


def _example_env(
    module: IRModule, example_input: Any
) -> tuple[tuple, dict[str, Any], Any]:
    """``(xs, env, x)`` mirroring ``IRModule.forward``'s input binding."""
    xs = (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )
    x = xs[0] if xs else None
    env: dict[str, Any] = {"self": x}
    for i, inp in enumerate(module._inputs):
        env[inp.name] = xs[i] if i < len(xs) else x
    return xs, env, x


def _screen_kernel(
    module: IRModule,
    env: dict[str, Any],
    x: Any,
    leaves: tuple[Any, ...],
    spelled: Callable[..., Any],
    compiled: Callable[..., Any],
    config: RegionFuseConfig,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Pick ``compiled`` or ``spelled`` for one kernel on real tensors.

    The boundary leaves are evaluated through the module's own plan
    tape (``_eval_fast`` — identical Const minting, dtype and liveness
    semantics to the delivery), then spelled and compiled bodies run
    on the same values.  Any failure — leaf eval, compiled call, or a
    ``verify_equiv`` report outside ``screen_rtol``/``screen_atol`` —
    binds spelled permanently: spelled is bitwise identical to the
    unfused spelling, so a declined kernel can never add error.
    """
    try:
        with torch.no_grad():
            vals = tuple(
                module._eval_fast(leaf, env, x) for leaf in leaves
            )
            ref = spelled(*vals)
            out = compiled(*vals)
        rep = verify_equiv(
            ref, out, rtol=config.screen_rtol, atol=config.screen_atol
        )
    except Exception as exc:
        return spelled, {
            "screened": False,
            "chosen": "spelled",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return (compiled if rep.passed else spelled), {
        "screened": True,
        "passed": rep.passed,
        "chosen": "compiled" if rep.passed else "spelled",
        "max_abs": rep.max_abs,
        "max_rel": rep.max_rel,
    }


def rewrite_term(
    term: Any, sites: Mapping[Any, tuple[int, tuple]]
) -> Any:
    """Rewrite region roots into ``fused_region`` ops (memoised DAG walk).

    Each site root becomes
    ``Op.make("fused_region", *leaves, fused_k=kid, validate=False)``;
    every other node is rebuilt only when a child changed, so shared
    subterms stay shared and untouched subtrees keep their interned
    identity.  Boundary leaves that contain inner regions are
    themselves rewritten — nesting composes.
    """
    memo: dict[Any, Any] = {}

    def rec(t: Any) -> Any:
        if not isinstance(t, Op):
            return t
        hit = memo.get(t)
        if hit is not None:
            return hit
        site = sites.get(t)
        if site is not None:
            kid, leaves = site
            new_leaves = tuple(
                rec(lv) if isinstance(lv, Op) else lv for lv in leaves
            )
            out = Op.make(
                "fused_region",
                *new_leaves,
                fused_k=kid,
                validate=False,
            )
        else:
            new_args = tuple(
                rec(a) if isinstance(a, Op) else a for a in t.args
            )
            if all(
                n is a for n, a in zip(new_args, t.args, strict=True)
            ):
                out = t
            else:
                try:
                    out = Op.make(
                        t.op, *new_args, validate=False, **t.attrs
                    )
                except TypeError:
                    # An attr literally named ``validate`` collides
                    # with the keyword — mint the dataclass directly.
                    out = Op(t.op, new_args, dict(t.attrs))
        memo[t] = out
        return out

    return rec(term)


def _restore(
    module: IRModule, root: Any, plan: Any, bindings: Any
) -> None:
    """Revert a module to its pre-fusion internals (decline path)."""
    module._root = root
    module._root_plan = plan
    module._torch_bindings = bindings


def fuse_module(
    module: Any,
    example_input: Any,
    *,
    config: RegionFuseConfig | None = None,
) -> dict[str, Any]:
    """Fuse *module*'s eligible elementwise regions in place.

    Returns the honest ledger recorded into ``stats["region_fuse"]``:
    census counts, per-kernel screening verdicts, the measured
    fused-vs-unfused ``max_abs``/``max_rel`` on ``example_input``, and
    ``applied`` — ``False`` with a ``reason`` whenever fusion declined
    (non-IRModule, no eligible regions, screening/gate failure, or an
    unexpected error; the module is then guaranteed untouched).

    The rewrite runs on ``module._root`` AFTER ``_fold_weight_chains``
    and ``_build_params`` — the folded/registered parameter set is
    unchanged because every region leaf is a node of the original
    term.  ``module._torch_bindings`` becomes a :class:`_FusedBindings`
    view so post-construction binding overrides still reach the
    module; ``OpTable`` itself is never mutated.
    """
    cfg = config if config is not None else RegionFuseConfig()
    t0 = time.time()
    rep: dict[str, Any] = {
        "applied": False,
        "mode": "compiled" if cfg.compiled else "spelled",
        "min_ops": cfg.min_ops,
        "output_rtol": cfg.output_rtol,
        "output_atol": cfg.output_atol,
        "screen_rtol": cfg.screen_rtol,
        "screen_atol": cfg.screen_atol,
        "reason": None,
        "elapsed_s": 0.0,
    }
    if not isinstance(module, IRModule):
        rep["reason"] = "not_ir_module"
        return rep
    root = module._root
    plan = collect_regions(root, min_ops=cfg.min_ops)
    rep["census"] = dict(plan.stats)
    if not plan.sites:
        rep["reason"] = "no_eligible_regions"
        rep["elapsed_s"] = time.time() - t0
        return rep

    bindings = module._torch_bindings
    xs, env, x = _example_env(module, example_input)
    old_root = module._root
    old_plan = module._root_plan
    old_bindings = module._torch_bindings
    swapped = False
    snap = _rng_snapshot()
    try:
        with torch.no_grad():
            # Warm-up forward: triggers ``_align_to`` so screening and
            # the gate evaluate on the delivery device/dtype, and
            # yields the unfused reference output for the gate.
            before = module(*xs)
        registry: dict[int, Callable[..., Any]] = {}
        kernel_reports: dict[str, Any] = {}
        if not cfg.compiled:
            for kid, (order, specs) in plan.exemplars.items():
                registry[kid] = _make_body(order, specs, bindings)
                kernel_reports[str(kid)] = {
                    "screened": False,
                    "chosen": "spelled",
                    "reason": "compiled_disabled",
                }
        else:
            _ensure_recompile_limit(
                _RECOMPILE_VARIANTS * max(1, len(plan.exemplars)), rep
            )
            for kid, (order, specs) in plan.exemplars.items():
                spelled = _make_body(order, specs, bindings)
                compiled = torch.compile(spelled, dynamic=cfg.dynamic)
                chosen, krep = _screen_kernel(
                    module,
                    env,
                    x,
                    plan.first_leaves[kid],
                    spelled,
                    compiled,
                    cfg,
                )
                registry[kid] = chosen
                kernel_reports[str(kid)] = krep
        new_root = rewrite_term(root, plan.sites)
        module._root = new_root
        module._root_plan = module._plan_for(new_root)
        module._torch_bindings = _FusedBindings(
            old_bindings, _make_dispatch(registry)
        )
        swapped = True
        # Aligned streams for the gate — same convention as
        # ``verify_module``'s seeded pass: restore the snapshot before
        # each forward so a stochastic module compares fairly.
        _rng_restore(snap)
        with torch.no_grad():
            after = module(*xs)
        _rng_restore(snap)
        eq = verify_equiv(
            before, after, rtol=cfg.output_rtol, atol=cfg.output_atol
        )
    except Exception as exc:
        if swapped:
            _restore(module, old_root, old_plan, old_bindings)
        rep["reason"] = f"{type(exc).__name__}: {exc}"
        rep["elapsed_s"] = time.time() - t0
        return rep
    finally:
        _rng_restore(snap)

    rep["fused_vs_unfused"] = {
        "max_abs": eq.max_abs,
        "max_rel": eq.max_rel,
        "passed": eq.passed,
        "rtol": eq.rtol,
    }
    rep["kernels"] = kernel_reports
    rep["regions_fused"] = int(plan.stats["fused"])
    rep["ops_fused"] = int(plan.stats["ops_fused"])
    rep["distinct_kernels"] = len(plan.exemplars)
    if not eq.passed:
        _restore(module, old_root, old_plan, old_bindings)
        rep["reason"] = "output_gate"
        rep["elapsed_s"] = time.time() - t0
        return rep
    rep["applied"] = True
    rep["elapsed_s"] = time.time() - t0
    return rep

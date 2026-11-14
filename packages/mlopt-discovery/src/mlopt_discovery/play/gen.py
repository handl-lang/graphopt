"""The ``gen`` domain — generated kernels over real torch modules.

The "discovery builds lowerings" story end to end: site collectors
(:mod:`.sites`) mint fusion patterns the ambient handler table has
no name for; the board mints a gen handler + kernel per occurring
elementwise subterm; ``claim`` binds it; ``deliver`` lowers it
through a ``TorchSink``.  Also hosts the shared measurement helpers
the probe arms use (:func:`_time_module`, :func:`_has_triton`).
"""

from __future__ import annotations

import random
from typing import Any

from mlopt_core.cost.basic import count_cost

from .. import lawdata
from .. import meta_arena as ma
from ..bounds import ProbeBounds, SkipStats
from .sites import _mint_tiers


def _gen_cases(
    seed: int,
    n: int,
    *,
    builders: Any = None,
    ambient_handlers: Any = None,
) -> list:
    """Torch modules over fusion-spelling and stock-model builders.

    The fusion builders spell pointwise patterns the ambient
    handler table has no fused op for — the candidates ``claim``
    can only reach through a *generated* kernel.  Stock pieces
    (mlp/attn chains) give the board real shape.

    *builders* (domain-spec seam) selects builder names from the
    table below; *ambient_handlers* substitutes the shipped
    ``lawdata.HANDLERS`` table the cases seat (``None`` keeps it).
    """
    import torch
    import torch.nn as nn
    from mlopt_torch.adapters import TorchSource

    from mlopt_discovery import genkernel as gk
    from mlopt_discovery import lawdata
    from mlopt_discovery.meta_arena import _canon_concrete

    class _Fuse(nn.Module):
        def __init__(self, kind: str) -> None:
            super().__init__()
            self.kind = kind

        def forward(self, x):
            """Spell the pointwise pattern — no fused op named."""
            if self.kind == "mulrelu":
                return x * torch.relu(x)
            if self.kind == "mulgelu":
                return x * torch.nn.functional.gelu(x)
            if self.kind == "mulsoftplus":
                return x * torch.nn.functional.softplus(x)
            if self.kind == "bigfuse":
                return (
                    x * torch.relu(x)
                    + torch.sigmoid(x)
                    - torch.tanh(x) * torch.abs(x)
                    + x * x
                )
            return x * torch.tanh(x)

    class _SwigluMLP(nn.Module):
        """Stock llama-MLP shape: ``down(silu(gate) * up)`` — spelled."""

        def __init__(self, d: int = 64) -> None:
            super().__init__()
            self.gate = nn.Linear(d, d * 2, bias=False)
            self.up = nn.Linear(d, d * 2, bias=False)
            self.down = nn.Linear(d * 2, d, bias=False)

        def forward(self, x):
            """Spell the swiglu — the site a claim can name."""
            g = self.gate(x)
            return self.down(g * torch.sigmoid(g) * self.up(x))

    class _ChainMLP(nn.Module):
        """A spelled matmul chain + pointwise tail — the reassoc story."""

        def __init__(self, d: int = 64) -> None:
            super().__init__()
            self.a = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.b = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.c = nn.Parameter(torch.randn(d, d) / d**0.5)

        def forward(self, x):
            """Three spelled mms, then a pointwise tail."""
            y = x @ self.a @ self.b @ self.c
            return y * torch.relu(y)

    class _TwoIslands(nn.Module):
        """Two fusion sites separated by matmuls — the composition case.

        No single elementwise-connex pattern covers both tails, so
        one extraction needs *two* minted kernels: ``claim`` each
        site and the delivered module calls both ``gen_*_k`` — and a
        claimed op can feed another claim's metavar (the inner
        fusion's e-class member is what the outer kernel receives).
        """

        def __init__(self, d: int = 128) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.v = nn.Parameter(torch.randn(d, d) / d**0.5)

        def forward(self, x):
            """Chain → fuse → chain → fuse, tails rejoined."""
            h = x @ self.w
            a = h * torch.relu(h)
            y = a @ self.v
            b = y * torch.tanh(y)
            return b + a

    class _ChainFuse(nn.Module):
        """Fold-able mm chain + a 6-op pointwise tail needing gen.

        The folded chain is a laws win; the tail's fused form exists
        only via ``claim(gen_*)`` — the win that needs the generated
        kernel.
        """

        def __init__(self, d: int = 512) -> None:
            super().__init__()
            self.a = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.b = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.c = nn.Parameter(torch.randn(d, d) / d**0.5)

        def forward(self, x):
            """Chain, then a long elementwise tail over the result."""
            y = x @ self.a @ self.b @ self.c
            return (
                y * torch.relu(y)
                - torch.nn.functional.softplus(y)
                + torch.tanh(y * y)
                * torch.nn.functional.softplus(y * torch.sigmoid(y))
            )

    rng = random.Random(seed)
    src = TorchSource()
    ambient = (
        lawdata.HANDLERS
        if ambient_handlers is None
        else ambient_handlers
    )
    covered = {_canon_concrete(h["pattern"]) for h in ambient.values()}
    table = [
        ("mulrelu", lambda: _Fuse("mulrelu"), (256, 512)),
        ("mulgelu", lambda: _Fuse("mulgelu"), (256, 512)),
        ("mulsoftplus", lambda: _Fuse("mulsoftplus"), (256, 512)),
        ("multanh", lambda: _Fuse("multanh"), (256, 512)),
        ("bigfuse", lambda: _Fuse("bigfuse"), (256, 512)),
        ("swiglu_mlp", _SwigluMLP, (16, 64)),
        ("chain_mlp", _ChainMLP, (16, 64)),
        ("chainfuse", _ChainFuse, (256, 512)),
        ("twosite", _TwoIslands, (256, 128)),
        (
            "mlp",
            lambda d=64: nn.Sequential(
                nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, d)
            ),
            (8, 64),
        ),
    ]
    if builders is not None:
        known = {b[0] for b in table}
        unknown = sorted(set(builders) - known)
        if unknown:
            raise ValueError(
                f"corpus.builders: unknown gen builder {unknown!r} "
                f"(known: {sorted(known)!r})"
            )
        wanted = set(builders)
        table = [b for b in table if b[0] in wanted]
    out = []
    for i in range(n):
        name, build, shape = table[rng.randrange(len(table))]
        model = build().eval()
        gen = torch.Generator().manual_seed(seed * 997 + i)
        x = torch.randn(*shape, generator=gen)
        ir, leaves = src.to_ir(model, x)
        genh = _mint_tiers(ir.root, covered)
        sink = gk.gen_sink(genh, compile_kernels=False)
        handlers = dict(ambient) | genh
        out.append(
            (
                f"gen{i}:{name}",
                ir.root,
                {"gen": len(genh)},
                name,
                sink.supported_ops,
                {
                    "ir": ir,
                    "leaves": leaves,
                    "model": model,
                    "input": x,
                    "sink": sink,
                    "handlers": handlers,
                    "gen_handlers": genh,
                    "ambient_handlers": ambient,
                },
            )
        )
    return out


def _var_env(ir: Any, leaves: dict, feed: Any) -> dict:
    """Build the metavar→value env ``profitable`` times against.

    Param leaves plus the input vars bound positionally to the
    feed — a param-only env makes every site touching the feed
    eval-fail and silently skip.
    """
    feeds = tuple(feed) if isinstance(feed, (tuple, list)) else (feed,)
    var_env = dict(leaves)
    var_env.update(
        {v.name: t for v, t in zip(ir.inputs, feeds, strict=False)}
    )
    return var_env


def _fold_cost_gate(case: Any, handlers: dict) -> dict:
    """Judge shipped prim folds by measured delivery, not op-count.

    T-019: the ``prim:synop*`` folds ride ``DEFAULT`` into saturation
    unconditionally — no claim offer, no ``profitable`` consult — so
    a folded member won extraction on the cost model while its
    delivered body lost wall-clock (the p26 re-sweep: 7 prim rows,
    median 0.943 → 0.744).  This is the same referee claims pass
    through, applied to the fold entries before they join the
    board's ``supported`` bound:

    * an entry whose pattern has **no concrete site** in the term
      cannot fire — it keeps its seat (it costs nothing);
    * an entry **with a site** is timed by ``profitable`` — the
      delivered binding (``kernel_for`` = the ``hybrid_sink`` codegen
      path) against the spelled denotation on the case's real
      values — and only a measured win stays feasible;
    * an entry the referee **cannot evaluate** — no concrete
      occurrence — keeps its seat: the gate refuses a *measured*
      loss, it does not invent one.  The attr-metavar specs (5 of
      57) DO get sited: ``genkernel.concretize_at_site`` binds the
      attr metavars from the matched site's real attrs and the
      referee times the frozen denotation — the same callable the
      delivery path's ``spec_bind`` produces for that site.  The
      *delivered* table keeps the polymorphic original; only the
      measurement sees the freeze.
    """
    from mlopt_discovery import genkernel as gk

    term = case[1]
    ex = case[5]
    ungated: dict[str, dict] = {}
    measured: dict[str, dict] = {}
    for name, h in handlers.items():
        if gk._handler_env(h, term) is not None:
            measured[name] = h
        elif h.get("attr_mvars"):
            concrete = gk.concretize_at_site(h, term)
            if concrete is None:
                ungated[name] = h
            else:
                measured[name] = concrete
        else:
            ungated[name] = h
    if not measured:
        return ungated
    bounds = _ex_bounds(ex)
    keep = gk.profitable(
        measured,
        term,
        _var_env(ex["ir"], ex["leaves"], ex["input"]),
        reps=bounds.reps or 20,
        kernel_for=_triton_kernel_for,
        bounds=bounds,
    )
    # ``keep`` holds the *concretized* copies — the delivered table
    # must carry the polymorphic originals (one site's frozen attrs
    # would mis-denote another site's call).
    return {
        **ungated,
        **{name: handlers[name] for name in keep},
    }


def _claim_measurable(tag: str, h: dict) -> bool:
    """Check a minted claim's spelled body is an honest denotation.

    The shared predicate behind both cost gates: ``gen_*``
    pointwise and ``decayed_*`` bodies evaluate through the
    ordinary bindings (except ``om_decayed``-headed specs, whose
    spelled side lives in ``_MEASURED_BINDINGS``, not
    ``_IR_TO_TORCH`` — unmeasurable stays unpriced); ``red_*``
    bodies evaluate through the ordinary bindings — the spelled
    spec IS the denotation (the reduction tier has no separate
    codegen; delivery is the compiled spelled body), so they are
    measurable like ``gen_*``/``decayed_*``.  ``scan_*`` claims
    are measurable in BOTH forms: *unrolled* compose-tree specs'
    spelled eval IS the carrier semantics, and *sequence*-form
    scans carry a registered spelled fold denotation
    (``genkernel.handler_denotation`` /
    ``seq_scan_denotation``) that ``profitable``'s
    ``denotation_for`` hook and ``gen_bindings`` both use — the
    ambient map application is never the referee's or the
    sink's stand-in for the fold.
    """
    from mlopt_discovery import genkernel as gk

    if tag.startswith(("gen_", "decayed_", "red_")):
        return "om_decayed" not in gk._spec_ops(h.get("pattern"))
    return tag.startswith("scan_")


def _claim_cost_gate(case: Any) -> None:
    """Drop minted ``gen_*`` claims that don't pay — the offer guard.

    ``_zoo_case`` already filters minted handlers through
    ``genkernel.profitable`` at intake; the synthetic ``_gen_cases``
    stream minted every fusion site *unpriced*, so ``claim`` offers
    included kernels that lose to their spelled body — the 0.38x /
    0.86x regimes the genkernel retro measured.  This applies the
    same measured-cost referee to the pointwise ``gen_*`` tier once
    per case, mutating ``ex["gen_handlers"]`` before the board's
    handler store is built: an unprofitable claim is never offered,
    so extraction cannot pick it and ``claim_tags`` never names it.

    Scope is every minted tier whose spelled body is an honest
    denotation of the claim:

    * ``gen_*`` pointwise fusions — the spelled spec-walk IS the
      unfused body;
    * ``decayed_*`` — the spelled ``matmul(mul(s, d), v)`` evaluates
      through the ordinary bindings, so kernel-vs-spelled is the
      same comparison the delivery ships (this is the offer that
      lost 0.47x on twosite before the gate existed).  ``om_decayed``-
      headed specs are excluded: that op lives in
      ``_MEASURED_BINDINGS``, not ``_IR_TO_TORCH``, so the referee's
      spelled side cannot evaluate them — unmeasurable stays
      unpriced rather than silently dropped;
    * ``scan_*`` — BOTH forms are measurable now.  *Unrolled*
      compose-tree specs' spelled eval IS the carrier semantics
      (``hybrid_sink``'s own rule).  *Sequence*-form scans carry a
      registered spelled fold denotation —
      ``genkernel.seq_scan_denotation`` spells the recurrence in
      plain torch and ``gen_bindings`` delivers it wherever the
      Triton kernel does not bind — so ``profitable``'s
      ``denotation_for`` hook times the generated fold kernel
      against the fold itself, never the ambient map application
      the ``("applyd", …)`` spec tuple evals to.  A seq claim
      with no bound codegen declines outright (delivery IS the
      denotation — there is no faster arm to beat);
    * ``red_*`` reduction-bodied fusions — the spelled
      ``mean``/``sum``/``softmax``-headed body evaluates through the
      ordinary bindings and IS the denotation (the tier ships the
      compiled spelled body, no separate codegen);
    * the shipped ``prim:*`` folds already pass
      :func:`_fold_cost_gate` on the same board build.

    ``_zoo_case`` sets ``ex["_gen_claims_gated"]`` — its intake
    ``profitable`` sweep already judged every minted tier, so the
    board does not pay the referee twice.
    """
    from mlopt_discovery import genkernel as gk

    ex = case[5]
    if ex.get("_gen_claims_gated"):
        return
    ex["_gen_claims_gated"] = True

    minted = {
        k: v
        for k, v in ex["gen_handlers"].items()
        if _claim_measurable(k, v)
    }
    if not minted:
        return
    bounds = _ex_bounds(ex)
    keep = gk.profitable(
        minted,
        case[1],
        _var_env(ex["ir"], ex["leaves"], ex["input"]),
        reps=bounds.reps or 20,
        kernel_for=_triton_kernel_for,
        denotation_for=_claim_denotation_for,
        bounds=bounds,
    )
    dropped = set(minted) - set(keep)
    if not dropped:
        return
    ex["gen_handlers"] = {
        k: v for k, v in ex["gen_handlers"].items() if k not in dropped
    }
    # reseat the delivery sink on the filtered table — the same
    # consistency the prim fold gate maintains
    if ex.get("sink") is not None:
        ex["sink"] = gk.gen_sink(
            ex["gen_handlers"], compile_kernels=False
        )


def _gen_board(case: Any) -> ma.MetaArena:
    """Board a gen case under its generated handler table.

    The bank merges the shipped ``synop*`` primitives' denotations as
    generated-kernel handler rows — one entry per minted op, kernel =
    the op itself.  The ``prim:synop*`` folds already ride ``DEFAULT``
    into saturation, but a folded ``synop_*`` member could never win
    extraction while no sink could lower it: the merged table is what
    makes the ops *feasible* (``supported``) and *deliverable* —
    ``gen_probe``'s ``hybrid_sink`` reads the same ``ex["gen_handlers"]``
    and binds each op to a fused-body codegen (``triton_bindings``
    where the denotation is elementwise, the ``torch.compile``d spelled
    body otherwise), not the spec-walk binding
    ``shipped_primitives.register()`` seats.  The merge mutates the
    case record so the delivery sink sees the identical table, and
    reseats ``ex["sink"]`` on it for the gen domain's deliver arm.
    """
    from mlopt_core.cells import Provenance
    from mlopt_core.laws import DEFAULT

    from mlopt_discovery import cellstore as cs
    from mlopt_discovery import genkernel as gk
    from mlopt_discovery import lawdata
    from mlopt_discovery import shipped_primitives as sp

    ex = case[5]
    _claim_cost_gate(case)

    def novel(op: str) -> bool:
        return op.startswith(("claim_", "foldabs_", "gen_"))

    prim = sp.handler_entries()
    if not any(tag in ex["gen_handlers"] for tag in prim):
        # T-019 fold cost gate: the ``prim:*`` folds fire
        # unconditionally through ``DEFAULT``, so the measured-cost
        # referee ``profitable`` applies to claims must also judge
        # the folds — a ``synop_*`` member stays feasible only when
        # its *delivered* binding beats the spelled site it replaces
        # on the case's real values (the 7 prim-delivering zoo rows
        # measured 0.744x median ungated — extraction preferred
        # members that lose wall-clock).
        ex["gen_handlers"] = {
            **ex["gen_handlers"],
            **_fold_cost_gate(case, prim),
        }
        if "sink" in ex:
            ex["sink"] = gk.gen_sink(
                ex["gen_handlers"], compile_kernels=False
            )
    kept = {k: v for k, v in prim.items() if k in ex["gen_handlers"]}
    supported = frozenset(case[4]) | {
        h["kernel"] for h in kept.values()
    }
    ambient = ex.get("ambient_handlers", lawdata.HANDLERS)
    store = cs.ambient_store(rules=DEFAULT, handlers=ambient).merge(
        cs.handlers_fragment(
            "gen",
            ex["gen_handlers"],
            provenance=Provenance("minted"),
        )
    )
    return ma.MetaArena(
        case[1],
        supported=supported,
        corpus=[case[1]],
        cost_fn=gk.novel_cost(count_cost, novel, 0.5),
        store=store,
        # ``spec_op_cap`` is the OP-EXPANSION metric the inventory
        # measures (``census.op_expansion``) — distinct from the
        # collectors' ``spec_tree_cap`` spelled tree-size
        spec_node_cap=_ex_bounds(ex).spec_op_cap,
    )


def _probe_play(board: Any) -> Any:
    """Play the maximal automated line; return the best extraction.

    Every claim offer, then ``saturate`` (the shipped laws —
    reassociation &c.), then ``extract``.
    """
    from mlopt_discovery import meta_arena as ma

    st = board.observe()
    for a in [a for a in ma.legal_actions(st) if a.op == "claim"]:
        board.step(a)
    board.step(ma.Action.saturate())
    board.step(ma.Action.extract())
    return board.eg.extract_best(board.root, board.feasible_cost)


def gen_probe(
    case: Any,
    *,
    budget: int = 12,
    reps: int = 200,
    pay_gate: bool = False,
) -> dict:
    """Measure the novelty claim: baseline vs generated-kernel module.

    For every claim the board offers: play it, extract, deliver
    through the case's gen sink (bindings compiled — the fused
    kernel is a real inductor graph), verify numerically against
    the source module, and time both sides.  The row answers the
    product question: did the search produce a *runnable module
    that is faster* than the spelled source, not just a new
    spelling.

    ``pay_gate=True`` runs :func:`mlopt_discovery.pay_gate.predict`
    on the case's root term *before* ``_gen_board`` pays the claim
    cost gate + board episode: a ``"skip"`` verdict returns a
    declined row immediately — scheduling only, the measured path
    for every other verdict is byte-identical.
    """
    if pay_gate:
        from mlopt_discovery import pay_gate as pg

        verdict = pg.predict(case[1])
        if not verdict.measure:
            return {
                "case": case[0],
                "delivered": False,
                "claims": [],
                "note": "pay-gate skip: " + "; ".join(verdict.reasons),
                "pay_gate": {
                    "class": verdict.cls,
                    "reasons": list(verdict.reasons),
                },
            }
    return _gen_probe_impl(case, budget=budget, reps=reps)


def _gen_probe_impl(case: Any, *, budget: int, reps: int) -> dict:
    """Run the measured claim → deliver → time path (``gen_probe`` body)."""
    import torch

    from mlopt_discovery import genkernel as gk

    # measurement integrity: each gen kernel is a distinct code
    # object recompiled under varying global state; dynamo's
    # default limit (8) would silently degrade late compiles —
    # and late *baseline* compiles — to eager.  Raise it so both
    # sides are measured as compiled, not corrupted by the cache.
    _dyno_cfg: Any = torch._dynamo.config
    _dyno_cfg.recompile_limit = max(int(_dyno_cfg.recompile_limit), 256)

    ex = case[5]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    board = _gen_board(case)
    tags = list(board.observe().claim_tags)
    best = _probe_play(board)
    rows: list[dict] = []
    base_model = ex["model"].to(device)
    xin = (
        tuple(t.to(device) for t in ex["input"])
        if isinstance(ex["input"], (tuple, list))
        else ex["input"].to(device)
    )
    try:
        t_base = _time_module(base_model, xin, reps)
    except Exception as exc:
        return {
            "case": case[0],
            "delivered": False,
            "note": f"baseline failed: {type(exc).__name__}: {exc}",
        }
    if best is None:
        return {"case": case[0], "delivered": False, "claims": tags}
    import torch as _torch

    compiled = _torch.compile(base_model)
    t_comp = _time_module(compiled, xin, reps)
    sink = gk.hybrid_sink(ex["gen_handlers"], compile_kernels=True)
    term = board.deliverable(best)
    from mlopt_core.ir import IR

    from mlopt_discovery import shipped_primitives as sp
    from mlopt_discovery.meta_arena import _ops_of

    prim_ops = sorted(_ops_of(term) & set(sp.SYNOP_SHIPPED))

    ir = IR(
        root=term,
        inputs=list(ex["ir"].inputs),
        input_names=set(ex["ir"].input_names),
        params=dict(ex["ir"].params),
    )
    try:
        rep, t_gen, module = _probe_delivery(
            sink, ir, ex, base_model, xin, device, reps
        )
    except Exception as exc:
        return {
            "case": case[0],
            "claims": tags,
            "delivered": False,
            "note": f"deliver failed: {type(exc).__name__}: {exc}",
            "baseline_us": t_base,
            "compiled_us": t_comp,
        }
    row = {
        "case": case[0],
        "claims": tags,
        "delivered": True,
        "verified": rep.passed,
        "prim_ops": prim_ops,
        "max_abs": rep.max_abs,
        "cost": getattr(board.observe(), "cost", float("nan")),
        "baseline_us": t_base,
        "compiled_us": t_comp,
        "gen_us": t_gen,
        "speedup": round(t_base / t_gen, 3) if t_gen else None,
        "vs_compiled": round(t_comp / t_gen, 3) if t_gen else None,
        "device": device,
    }
    # T-030/T-031 probe arm: the raw ``gen_us`` above is the per-leaf
    # lowering's wall — on launch-bound zoo rows that is ~75% dispatch
    # the rewrite bank cannot reach (the frontier ceiling counts it,
    # the delivered body doesn't take it).  The shipped
    # ``CudaGraphRunner`` is the delivery-side answer: capture the
    # *same* verified module into a CUDA graph and time the replay.
    # Additive keys only — ``gen_us``/``speedup`` stay the lowering
    # number; ``gen_graph_us`` is the runner-wrapped arm.  Optional
    # dep (``mlopt-cuda`` is not in this package's requirements) and
    # CUDA-only, so every failure path degrades to no keys.
    graph = _probe_graph_arm(module, xin, t_base, t_comp, reps)
    row.update(graph)
    rows.append(row)
    return rows[0]


def _probe_graph_arm(
    module: Any,
    xin: Any,
    t_base: float,
    t_comp: float,
    reps: int,
) -> dict:
    """Time the delivered module under ``CudaGraphRunner``; {} on decline.

    Returns ``{}`` unless the runner actually captured (``stats
    ["cuda_graph"]``) — an honest decline records nothing rather than
    a ``gen_graph_us`` that is just the eager path re-timed.  When it
    binds, the row also carries the replay-vs-eager verify delta so a
    capture that changed numerics could not pass silently.
    """
    import torch

    try:
        from mlopt_cuda import CudaGraphRunner
    except ImportError:  # pragma: no cover — optional dep absent
        return {}
    args = xin if isinstance(xin, (tuple, list)) else (xin,)
    if not (args and torch.is_tensor(args[0]) and args[0].is_cuda):
        return {}
    stats: dict = {}
    try:  # pragma: no cover — CUDA-only body
        wrapped = CudaGraphRunner().apply(module, xin, stats)
        if not stats.get("cuda_graph"):
            return {}
        t_graph = _time_module(wrapped, xin, reps)
        with torch.no_grad():
            out_eager = module(*args)
            out_graph = wrapped(*args)
        graph_max_abs = (
            (out_eager - out_graph).abs().max().item()
            if torch.is_tensor(out_eager)
            else float("nan")
        )
    except Exception:  # pragma: no cover — CUDA-only
        return {}
    return {
        "gen_graph_us": round(t_graph, 1),
        "graph_speedup": round(t_base / t_graph, 3)
        if t_graph
        else None,
        "graph_vs_compiled": (
            round(t_comp / t_graph, 3) if t_graph else None
        ),
        "graph_max_abs": graph_max_abs,
    }


#: Referee audit log — one record per ``_triton_kernel_for`` consult.
#: The ``kernel_for`` hook declines silently otherwise: a claim whose
#: Triton arm never binds is graded on the compiled spelled body and
#: nothing records why the generated kernel never ran (the "8 of 8 no
#: binding" row in ``docs/reports/zoo_chain.md``).  Each record carries
#: the claim tag, the kernel name, the bound-value dtype, the per-tier
#: attempt table and the final outcome; bench audit drivers
#: (``bench/drivers/probes/triton_binding_audit.py``) read it to answer
#: "did the generated kernel ever bind, and if not, why".  Appended
#: unconditionally — a claim sweep is tens of records, not a stream.
TRITON_REFEREE_LOG: list[dict] = []


def _binding_attempts(tag: str, h: dict) -> tuple[Any, list[dict]]:
    """Return the first binding any tier accepts, and the attempt trail.

    Returns ``(fn, attempts)``; each attempt is
    ``{"tier", "outcome", "reason"}`` — ``bound`` /
    ``declined`` / ``no_builder``.  A decline distinguishes a spec the
    tier's recogniser rejects (outside its grammar) from one it
    accepts whose generated module then fails to load — the same
    silence ``triton_bindings`` and friends return ``{}`` for.
    """
    from mlopt_discovery import genkernel as gk

    attempts: list[dict] = []
    for tier in gk.TIERS:
        if tier.triton is None:
            attempts.append(
                {
                    "tier": tier.name,
                    "outcome": "no_builder",
                    "reason": "tier.triton is None — the registry "
                    "has no generated-kernel builder for this tier",
                }
            )
            continue
        fn = tier.triton({tag: h}).get(h["kernel"])
        if fn is not None:
            attempts.append({"tier": tier.name, "outcome": "bound"})
            return fn, attempts
        # the tier's *codegen* recogniser first (``<tier>_triton_spec``
        # — e.g. ``reduction_triton_spec`` gates the subset the Triton
        # piece lowers, narrower than the mint grammar ``reduction_spec``)
        recognises = getattr(gk, f"{tier.name}_triton_spec", None)
        if recognises is None:
            recognises = getattr(gk, f"{tier.name}_spec", None)
        if (
            recognises is not None
            and recognises(h["pattern"]) is not None
        ):
            reason = "spec recognised; generated module failed to load"
        else:
            reason = "spec declined — outside the tier's grammar"
        attempts.append(
            {"tier": tier.name, "outcome": "declined", "reason": reason}
        )
    return None, attempts


def _triton_kernel_for(tag: str, h: dict, vals: dict) -> Any:
    """Return the delivery-path callable a claim would run.

    ``profitable``'s ``kernel_for``: consult each tier's generated
    binding builder — the callable the ``hybrid_sink`` delivery
    actually ships — on CUDA-bound values (the probe's device).
    Falls back to ``None`` (the compiled-spelled default) when
    Triton/CUDA or every tier's codegen declines.  Every consult is
    recorded in :data:`TRITON_REFEREE_LOG`.
    """
    import torch

    from mlopt_discovery import genkernel as gk

    record: dict = {
        "tag": tag,
        "kernel": h.get("kernel"),
        "pattern_head": (
            h["pattern"][0]
            if isinstance(h["pattern"], (tuple, list))
            else h["pattern"]
        ),
        "dtype": next(
            (
                str(v.dtype).removeprefix("torch.")
                for v in vals.values()
                if torch.is_tensor(v)
            ),
            None,
        ),
    }
    TRITON_REFEREE_LOG.append(record)
    if not (torch.cuda.is_available() and _has_triton()):
        record["outcome"] = "unavailable"
        record["reason"] = "no CUDA or no triton import"
        return None
    try:
        fn, attempts = _binding_attempts(tag, h)
    except Exception as exc:  # a builder's own failure is a decline too
        record["outcome"] = "declined"
        record["reason"] = (
            f"binding builder raised: {type(exc).__name__}: {exc}"
        )
        return None
    record["attempts"] = attempts
    if fn is None:
        record["outcome"] = "declined"
        return None
    try:
        mvs = gk._handler_mvs(h)
        cvals = {m: vals[m].cuda() for m in mvs}
        ctuple = tuple(cvals[m] for m in mvs)
    except (AttributeError, KeyError, RuntimeError, TypeError) as exc:
        record["outcome"] = "env_bind_failed"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        return None
    record["outcome"] = "bound"

    def kern(*a: Any) -> Any:
        return fn(*ctuple)

    return kern, cvals


def _claim_denotation_for(tag: str, h: dict, vals: dict) -> Any:
    """``profitable``'s ``denotation_for``: the seq fold, or ``None``.

    Returns the positional spelled-fold callable
    ``genkernel.handler_denotation`` registers for ``kind="seq"``
    scan specs — the denotation the minted op declares — so the
    referee times the generated fold kernel against the fold
    itself, never the ambient ``applyd`` map application the spec
    tuple evals to.  ``None`` for every other claim: their spelled
    spec eval already IS the denotation.
    """
    from mlopt_discovery import genkernel as gk

    return gk.handler_denotation(h)


def _probe_delivery(
    sink: Any,
    ir: Any,
    ex: dict,
    base_model: Any,
    xin: Any,
    device: str,
    reps: int,
) -> Any:
    """Lower → verify → time the delivered module; returns (rep, µs, mod)."""
    module = sink.lower(ir, params=ex["leaves"]).to(device)
    rep = sink.verify(
        base_model,
        module,
        xin if isinstance(xin, (tuple, list)) else (xin,),
    )
    return rep, _time_module(module, xin, reps), module


def _as_dtype(obj: Any, td: Any) -> Any:
    """Cast a feed (tensor, or a tuple/list of them) to ``td``.

    Feeds carry ints next to tensors, so a blind ``.to`` raises; only
    tensors move, everything else passes through untouched.
    """
    import torch

    if torch.is_tensor(obj):
        return obj.to(td)
    if isinstance(obj, (tuple, list)):
        return type(obj)(_as_dtype(x, td) for x in obj)
    return obj


def _zoo_export(wl: Any, *, dtype: str = "fp64") -> Any:
    """Build + export a zoo workload — the cheap half of ``_zoo_case``.

    Returns ``(model, feed, ir, leaves)``.  No mint, no referee —
    this is exactly the work a pay-gate verdict needs before the
    expensive intake sweep runs.
    """
    import torch
    from mlopt_torch.adapters import TorchSource

    td = {"fp32": torch.float32, "fp64": torch.float64}[dtype]
    model, feed = wl.build()
    model = model.to(td)  # the intake convention is fp64 feeds
    feed = _as_dtype(feed, td)
    ir, leaves = TorchSource().to_ir(model, feed)
    return model, feed, ir, leaves


def zoo_pay_verdict(wl: Any, *, dtype: str = "fp32") -> dict | None:
    """Cheap pre-search pay verdict for a zoo workload.

    Exports the module to IR (``_zoo_export`` — build + ``to_ir``,
    no mint, no ``profitable`` sweep) and runs
    :func:`mlopt_discovery.pay_gate.predict` on the root term.
    Returns ``{"class", "reasons", "features"}`` or ``None`` when the
    export itself fails — a workload that cannot be exported has no
    verdict and the caller should proceed exactly as today.
    """
    from mlopt_discovery import pay_gate as pg

    try:
        _model, _feed, ir, _leaves = _zoo_export(wl, dtype=dtype)
    except Exception:
        return None
    v = pg.predict(ir.root)
    return {
        "class": v.cls,
        "reasons": list(v.reasons),
        "features": dict(v.features),
    }


def _ex_bounds(ex: dict) -> ProbeBounds:
    """Return the case's probe bounds — the typed record or legacy keys.

    ``ex["bounds"]`` (a :class:`ProbeBounds`) is the current
    spelling; cases built before it existed carry
    ``ex["_referee"]``/``ex["_spec_node_cap"]`` and are reconstructed
    by :meth:`ProbeBounds.from_legacy_ex` — the legacy single cap
    fed BOTH spec metrics, so the fallback does the same (each side
    then measures its own unit).
    """
    b = ex.get("bounds")
    if isinstance(b, ProbeBounds):
        return b
    return ProbeBounds.from_legacy_ex(ex)


def _case_tail(
    ir: Any,
    leaves: dict,
    feed: Any,
    model: Any,
    case: str,
    tag: str,
    *,
    gen_reps: int = 20,
    bounds: ProbeBounds | None = None,
) -> tuple:
    """Run the shared gen-domain intake tail on an already-lifted term.

    ``_zoo_case``'s tail, factored out so a non-zoo term (a generated
    mutant's IR — ``model_corpus.mutant_board``) enters the *same*
    sequence of calls: ``_mint_tiers`` over the term's real sites →
    the ``_claim_measurable`` split → ``genkernel.profitable``'s
    measured-cost referee → ``gen_sink``.  ``case``/``tag`` are the
    two label slots the tuple carries (``case[0]`` the board name,
    ``case[3]`` the workload tag — the zoo passes
    ``f"zoo:{wl.name}"``/``wl.name``); ``gen_reps`` is the referee's
    timing reps (the zoo keeps the historical ``20``).

    *bounds* is the :class:`ProbeBounds` for model-scale terms
    (``model_corpus`` resolves it once per term via
    ``_probe_knobs``): ``spec_tree_cap`` bounds the minted site
    patterns' spelled tree-size, ``spec_op_cap`` the ``MetaArena``
    spec inventory's op-expansion, ``key_cap`` the novelty keys,
    and ``max_handlers``/``time_budget``/``reps`` the measured-cost
    referee sweep — ``max_handlers`` caps how many minted claims
    ``profitable`` prices (sorted, deterministic), ``time_budget``
    shrinks each claim's timing window, ``reps`` overrides the
    per-call-site default (``gen_reps`` here, ``20`` in the board
    gates).  Unpriced claims are *dropped*, never offered
    unmeasured.  The bounds are seated on ``ex["bounds"]`` (the
    legacy ``ex["_referee"]``/``ex["_spec_node_cap"]`` spellings are
    kept for older consumers) so ``_gen_board``'s fold/claim gates
    apply the same bounds instead of re-paying an uncapped sweep.

    ``None`` keeps the historical unbounded zoo behaviour.

    Returns the ``(name, term, decls, tag, supported, ex)`` tuple
    ``gen_probe`` reads — ``ex["_gen_claims_gated"]`` marks that
    ``profitable`` already ran so ``_gen_board``'s offer gate does
    not re-pay it, and ``ex["site_stats"]`` (a
    :class:`~mlopt_discovery.bounds.SkipStats`) carries EVERY
    bounded collector's over-cap skip counts.
    """
    from mlopt_discovery import genkernel as gk
    from mlopt_discovery.meta_arena import _canon_concrete

    bounds = bounds if bounds is not None else ProbeBounds()
    covered = {
        _canon_concrete(h["pattern"]) for h in lawdata.HANDLERS.values()
    }
    # ``site_stats`` records every bounded collector's honest
    # over-cap skip counts (``fusion_sites_over_cap``,
    # ``seq_sites_over_cap`` etc.) — sites the ``spec_tree_cap``
    # bound declined rather than spelled.
    site_stats: SkipStats = SkipStats()
    genh = _mint_tiers(
        ir.root, covered, bounds=bounds, stats=site_stats
    )
    # the measured-cost referee: mint only claims whose generated
    # kernel actually beats its spelled site on the real values;
    # the env must bind the input vars too — a param-only env makes
    # every site touching the feed eval-fail and silently skip.
    # Only *measurable* claims are priced (``_claim_measurable`` —
    # the same predicate ``_claim_cost_gate`` applies).  Sequence-
    # form scans ARE measurable: ``denotation_for`` hands the
    # referee the registered fold denotation
    # (``genkernel.handler_denotation``) as the spelled arm, so the
    # kernel is graded fold-vs-fold — and ``gen_sink`` below binds
    # the same denotation wherever the Triton kernel does not, so
    # delivery can never silently ship the ambient map
    # application.  Other unmeasurable claims (``om_decayed``-
    # headed decayed specs) keep their seat unpriced, exactly as
    # the board-build gate leaves them — an undeliverable binding
    # fails loudly there, never silently.
    var_env = _var_env(ir, leaves, feed)
    priced = {k: v for k, v in genh.items() if _claim_measurable(k, v)}
    unpriced = {
        k: v for k, v in genh.items() if not _claim_measurable(k, v)
    }
    priced_keep = gk.profitable(
        priced,
        ir.root,
        var_env,
        reps=bounds.reps or gen_reps,
        kernel_for=_triton_kernel_for,
        denotation_for=_claim_denotation_for,
        bounds=bounds,
    )
    genh = {**unpriced, **priced_keep}
    sink = gk.gen_sink(genh, compile_kernels=False)
    return (
        case,
        ir.root,
        {"gen": len(genh)},
        tag,
        sink.supported_ops,
        {
            "ir": ir,
            "leaves": leaves,
            "model": model,
            "input": feed,
            "sink": sink,
            "handlers": dict(lawdata.HANDLERS) | genh,
            "gen_handlers": genh,
            # intake already ran ``profitable`` over every minted
            # tier — ``_gen_board``'s offer gate must not re-pay it
            "_gen_claims_gated": True,
            # the typed bounds record — ``_ex_bounds`` prefers it;
            # the legacy spellings stay for older consumers
            "bounds": bounds,
            "_referee": bounds.referee_dict(),
            # legacy key name: this value is the OP-EXPANSION cap
            # (``ProbeBounds.spec_op_cap``) the MetaArena inventory
            # measures — NOT the collectors' tree-size cap
            "_spec_node_cap": bounds.spec_op_cap,
            "site_stats": site_stats,
            "claims_priced": len(priced),
            "claims_kept": len(priced_keep),
        },
    )


def _zoo_case(wl: Any, *, dtype: str = "fp64") -> Any:
    """Build a held-out zoo workload as a gen-domain case tuple.

    ``dtype`` is the **intake** dtype.  ``fp64`` is the convention every
    earlier table was measured at; ``fp32`` is what a caller ships, and
    the two disagree on 12 of 65 zoo modules — a folded body's sign is
    the backend kernel's, not the IR's.  The IR is the same either way;
    only the leaves, the feeds and the refinery's measurements move.
    """
    model, feed, ir, leaves = _zoo_export(wl, dtype=dtype)
    return _case_tail(
        ir, leaves, feed, model, f"zoo:{wl.name}", wl.name
    )


def _zoo_rows(args: Any) -> None:
    """Print the zoo probe rows — the ``--zoo`` CLI arm."""
    for row in _zoo_probe_for_args(args):
        print(f"zoo[{row['case']}]: {row}")  # stdout-compat


def _zoo_probe_for_args(args: Any) -> Any:
    """``zoo_probe`` bound to the CLI args (``--pay-gate`` opt-in)."""
    kwargs: dict = {}
    if getattr(args, "pay_gate", False):
        kwargs["pay_gate"] = True
    return zoo_probe(d=args.zoo_d, t=args.zoo_t, b=args.zoo_b, **kwargs)


def zoo_probe(
    *,
    reps: int = 100,
    d: int = 16,
    t: int = 8,
    b: int = 2,
    pay_gate: bool = False,
) -> list[dict]:
    """``gen_probe`` over the held-out model zoo — the real corpus.

    Untraceable workloads are reported honestly rather than dropped
    silently; each surviving row is the full claim → deliver →
    verify → time table vs eager *and* ``torch.compile``.  ``d``/``t``
    /``b`` scale the workloads — generated kernels pay at larger
    shapes.

    ``pay_gate=True`` routes each workload through
    :func:`zoo_pay_verdict` *before* ``_zoo_case`` spends the mint +
    ``profitable`` intake sweep: a ``"skip"`` verdict (the
    elementwise-bound signature — measured to tie the spelled body)
    records a skipped row and never pays for search or measurement.
    ``"pay"``/``"unknown"`` proceed exactly as today — the gate only
    ever removes budget, it never changes what a measured workload
    does.
    """
    if pay_gate:
        return _zoo_probe_gated(d=d, t=t, b=b, reps=reps)
    return _zoo_probe_impl(d=d, t=t, b=b, reps=reps)


def _zoo_probe_impl(*, reps: int, d: int, t: int, b: int) -> list[dict]:
    """Run the ungated per-workload loop (``zoo_probe``'s original body)."""
    from mlopt_discovery import zoo as _zoo

    return [
        _zoo_probe_workload(wl, reps=reps) for wl in _zoo.zoo(d, t, b)
    ]


def _zoo_probe_gated(
    *, reps: int, d: int, t: int, b: int
) -> list[dict]:
    """``zoo_probe`` with the pay gate: skip before the intake sweep."""
    from mlopt_discovery import zoo as _zoo

    rows: list[dict] = []
    for wl in _zoo.zoo(d, t, b):
        verdict = zoo_pay_verdict(wl)
        if verdict is not None and verdict["class"] == "skip":
            rows.append(
                {
                    "case": f"zoo:{wl.name}",
                    "delivered": False,
                    "note": "pay-gate skip: "
                    + "; ".join(verdict["reasons"]),
                    "pay_gate": verdict,
                }
            )
            continue
        rows.append(_zoo_probe_workload(wl, reps=reps))
    return rows


def _zoo_probe_workload(wl: Any, *, reps: int) -> dict:
    """One workload's ``_zoo_case`` + ``gen_probe`` row, or the decline."""
    try:
        case = _zoo_case(wl)
    except Exception as exc:
        return {
            "case": f"zoo:{wl.name}",
            "delivered": False,
            "note": f"export declined: {type(exc).__name__}: {exc}",
        }
    return gen_probe(case, reps=reps)


def _time_module(model: Any, x: Any, reps: int) -> float:
    """Median per-call latency (µs) of ``model(*x)`` on its device.

    ``x`` is a tuple of positional inputs — a single-tensor feed is
    ``(x,)``.
    """
    import torch

    args = x if isinstance(x, (tuple, list)) else (x,)
    dev = args[0].device
    for _ in range(10):
        model(*args)
    if dev.type == "cuda":  # pragma: no cover — CUDA-only host path
        torch.cuda.synchronize()
    import time

    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(*args)
        if dev.type == "cuda":  # pragma: no cover — CUDA-only host path
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def _has_triton() -> bool:
    """Check triton is importable (the codegen tiers' runtime)."""
    import importlib.util

    return importlib.util.find_spec("triton") is not None

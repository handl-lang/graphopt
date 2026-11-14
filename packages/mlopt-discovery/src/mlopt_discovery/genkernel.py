"""Generated kernels — lowerings the search itself builds.

The handler table (:data:`lawdata.HANDLERS`) is a human-authored
vocabulary: ``silu``/``square``/``swiglu`` are names someone wrote.
This module closes the loop the meta-arena's ``claim`` move implies:
given a *spec* — a canonical pattern tuple mined from a corpus — mint
a handler entry whose kernel is a NEW op name, plus a torch binding
that evaluates the spelled body.  The board's vocabulary grows with
lowerings the search built; nothing in the arena or the sink changes:

* ``gen_handlers(specs)`` mints ``{tag: {pattern, kernel, args}}``
  for each *elementwise* spec not already covered by the shipped
  handlers — the fused-op candidates nobody named.
* ``gen_bindings`` produces each kernel's :class:`~mlopt_core.ports.Binding`
  — ``fn(*vals) = eval(spelled body, leaves=vals)`` — optionally
  ``torch.compile``-compiled so the delivered kernel is a real fused
  implementation (one inductor graph), not a replay of the spelling.
* ``gen_sink`` folds the bindings into a private :class:`OpTable`
  copy and wraps it in a ``TorchSink`` — ``supported_ops`` reports
  the gen names (the feasibility bound the board prices against)
  and ``lower`` delivers modules that *call the kernel the board
  invented*.
* ``novel_cost`` — an extraction-priced novelty bonus so the search
  prefers a claimed/generated member when costs tie.

Feasibility stays honest: a gen op is only priced low when it is in
the board's ``supported`` set, and kernels are only *generated* for
elementwise bodies — the narrow claim this module makes (pointwise
fusion), not a general codegen story.

THE SCAN TIER
    ``triton_bindings`` covers pointwise bodies.  The second codegen
    tier lowers *carrier* terms — the diagonal-affine scan the
    ``affd_*_lift`` family mints — into ONE sequential Triton kernel.
    Two spec forms are recognised (:func:`scan_spec`):

    * ``("applyd", ("aff_diag", a, b), h)`` — the *sequence* form.
      The leaf operands are per-step tensors carrying a step axis
      (``axis``, default ``-2`` — the token axis of ``(B, T, d)``);
      the kernel folds ``h_t = a_t ⊙ h + b_t`` along it and returns
      the final state ``h_T``.  Operands may be metavars, constants,
      or elementwise sub-specs — every metavar inside is a sequence
      operand, so ``aff_diag(("sigmoid", "G"), ("mul", "B", "X"))``
      generates the fully-fused selective step.
    * ``("applyd", <affd_compose tree>, h)`` — the *unrolled* form.
      Each ``aff_diag`` leaf is one carrier step (state-shaped
      operands, no step axis); the kernel applies them in
      chronological order — in-order leaves are reverse-chronological
      (``affd_compose(f, g)`` applies ``g`` first) — in one pass.
      This is the carrier semantics *verbatim*.

    The generated binding is a NEW kernel name (``scan_<i>_k``), not
    ``applyd`` itself: the sequence interpretation is the minted op's
    declared semantics, never a silent override of the ambient
    carrier binding.  Call-time shape violations — no step axis, an
    ``h`` that cannot broadcast to the state shape, non-CUDA tensors
    — raise rather than fall back, so a wrong binding fails loudly.

    Delivery seam: ``scan_handlers`` entries merge into a handler
    table exactly as :func:`gen_handlers` output does — the arena's
    ``claim`` binds them with no new machinery, and
    :func:`hybrid_sink` (the sink ``gen_probe`` lowers through)
    consults both codegen tiers, so a minted scan op is deliverable
    end to end.  :func:`scan_triton_sink` is the scan-only sink.
    What is NOT wired is *minting*: raw exported IR carries no
    ``applyd`` sites — carrier members only appear once lifts run —
    so case builders do not call ``scan_handlers``; a caller that
    can vouch for a site mints the entry itself.  A claim's
    soundness is the minter's burden: the unrolled form is the
    ambient carrier semantics verbatim; the sequence form is a NEW
    declared semantics — the ambient
    ``applyd(aff_diag(a, b), h)`` is one map application
    (``a⊙h+b``), not a fold — so mint it only where the fold
    reading is what the site means.  The spelled fold denotation
    itself is registered: :func:`seq_scan_denotation` spells the
    recurrence in plain torch (per-step operand evals folded along
    the step axis), :func:`handler_denotation` resolves it per
    handler, ``gen_bindings`` delivers it wherever the Triton
    kernel does not bind, and ``profitable``'s ``denotation_for``
    hook grades claims against it — the map application is never
    again a silent stand-in for the fold.

THE DECAYED TIER
    The third codegen tier fuses the decayed-attention element —
    ``om_decayed(s, d, v) = (s ⊙ d) @ v``, the RetNet inner-chunk
    form — into ONE Triton matmul kernel.  Two spellings are
    recognised (:func:`decayed_spec`): the element itself and the
    spelled ``("matmul", ("mul", s, d), v)`` it lifts from.  The
    generated kernel is a tiled batched matmul whose inner loop
    folds the elementwise decay product into the contraction —
    the ``(…,T,K)`` intermediate is never materialised.  Broadcast
    is stride-driven: every operand meta carries an int64
    base-offset table plus row/col strides, so both broadcast
    regimes — a full-rank ``d`` (…,T,K) and a lower-rank/scalar
    decay read in place — run the same kernel, with per-meta
    ``FB`` constexprs specialising metas whose batch strides are
    all zero.  Same delivery seam and decline
    semantics as the scan tier: ``decayed_handlers`` mints
    ``decayed_<i>`` entries, :func:`decayed_triton_bindings`
    generates ``decayed_<i>_k`` kernels, :func:`hybrid_sink`
    consults the tier after scan, and a declined spec keeps its
    torch path — whose spelled eval IS the element's reference
    semantics (the ambient ``om_decayed`` binding evaluates
    ``(s·d)@v`` verbatim), so the fallback stays honest.

THE REDUCTION TIER
    The fourth codegen tier claims *reduction-bodied* fusions —
    ``mean(mul(x, relu(x)), dim=-1)``, ``sum(mul(s, d))``,
    ``softmax(gelu(x))`` — where the pointwise tier's boundary rule
    stops one level short: its elementwise walk abstracts every
    non-elementwise child to a metavar, so a claim over a reduction
    could never be minted at all.  :func:`reduction_spec` accepts a
    spec whose head is a reduce-dim op (:func:`reduce_heads` — the
    ``opmeta.REDUCE_DIM_OPS`` vocabulary plus ``softmax``) with an
    all-elementwise body of >=2 ops; :func:`reduction_handlers`
    mints ``red_<i>`` entries and ``Tier("reduction", …,
    reduction_triton_bindings)`` seats them.

    THE REDUCTION TRITON PIECE
        :func:`reduction_triton_spec`/``reduction_triton_bindings``
        lower the recognisable subset — ``sum``/``mean``/``amax``/
        ``amin``/``softmax`` over an arbitrary set of reduce dims
        (the index space permutes outer-then-reduced, so the reduce
        set need not be trailing), elementwise body, floating
        operands — into ONE generated
        Triton kernel: one program per outer index, the reduce axis
        chunked inside the program (``softmax`` runs the textbook
        max → sum-exp → normalise three-pass form in the same
        launch).  Broadcast and arbitrary strides ride an ``ometa``
        int64 table like the pointwise strided kernel; shapes
        outside the subset — non-trailing reduce dims, non-float
        operands, dims that cannot normalise against the broadcast
        rank — fall back to the spelled eval *inside* the binding,
        so delivery stays the denotation and the referee prices the
        Triton arm wherever it binds (and the compiled-spelled arm
        where it does not).
"""

from __future__ import annotations

import keyword
import math
from typing import Any, NamedTuple

__all__ = [
    "ELEMENTWISE",
    "TIERS",
    "Tier",
    "attr_site_env",
    "concretize_at_site",
    "decayed_handlers",
    "decayed_spec",
    "decayed_triton_bindings",
    "decayed_triton_sink",
    "elementwise_spec",
    "gen_bindings",
    "gen_handlers",
    "gen_sink",
    "handler_denotation",
    "novel_cost",
    "reduce_heads",
    "reduction_handlers",
    "reduction_spec",
    "reduction_triton_bindings",
    "reduction_triton_spec",
    "scan_handlers",
    "scan_spec",
    "scan_triton_bindings",
    "scan_triton_sink",
    "seq_scan_denotation",
    "seqm_denotation",
    "seqs_denotation",
    "seqt_denotation",
]

#: Ops whose torch bindings are pointwise (shape-preserving) — the
#: bodies a generated kernel may fuse.  Restricted to the plain
#: unary/binary elementwise vocabulary plus the attr-bearing clamps:
#: no reductions, no reshapes, no matmul — the honest narrow claim.
#: The clamps carry their bounds as attr dicts (``{"min": …,
#: "max": …}``), which every consumer already threads: ``_spec_ops``
#: skips dicts, ``_fusion_pattern`` appends ``t.attrs``, and the
#: Triton tiers decline attr-carrying nodes so those bodies deliver
#: through the compiled-spelled arm.
ELEMENTWISE: frozenset = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "abs",
        "square",
        "sigmoid",
        "tanh",
        "relu",
        "gelu",
        "softplus",
        "mish",
        "silu",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "sin",
        "cos",
        "identity",
        "pow",
        "reciprocal",
        "softsign",
        "softsign_gate",
        "clamp",
        "clamp_min",
        "clamp_max",
    }
)


def _spec_ops(spec: Any) -> set:
    """Collect op names in a canonical spec tuple (leaves are metas)."""
    if isinstance(spec, (tuple, list)) and spec:
        out = {spec[0]}
        for e in spec[1:]:
            if not isinstance(e, dict):
                out |= _spec_ops(e)
        return out
    return set()


def _spec_metas(spec: Any) -> list:
    """Ordered distinct metavar leaves of a canonical spec."""
    out: list = []

    def walk(s: Any) -> None:
        if isinstance(s, str):
            if s not in out:
                out.append(s)
        elif isinstance(s, (tuple, list)):
            for e in s[1:]:
                if not isinstance(e, dict):
                    walk(e)

    walk(spec)
    return out


def elementwise_spec(spec: Any) -> bool:
    """Check a spec is a pure elementwise body the gen path may fuse."""
    ops = _spec_ops(spec)
    return bool(ops) and ops <= ELEMENTWISE


def gen_handlers(terms: list, *, covered: set | None = None) -> dict:
    """Mint a handler table for concrete subterms or patterns.

    Each entry yields ``gen_<i>: {pattern, kernel, args}`` — the
    same entry shape ``lawdata.HANDLERS`` ships, so ``claim``
    binds them with no new machinery.  Entries whose *concrete*
    pattern already exists — in *covered* (e.g. the shipped
    handlers' canon-concrete patterns) or earlier in the list —
    are skipped: the generated vocabulary adds names only where
    the ambient table has none.
    """
    from mlopt_discovery.meta_arena import _spec_of

    skip = covered if covered is not None else set()
    out: dict[str, dict] = {}
    for term in terms:
        pattern = (
            term
            if isinstance(term, (tuple, list))
            else _spec_of(term, {})
        )
        # ``_canon_taken``, not ``canon in skip``: attr-carrying specs
        # canon to dicts — unhashable — and ``in`` raises where
        # equality does not (the ``_canon_taken`` contract).
        if not elementwise_spec(pattern):
            continue
        if _canon_taken(pattern, out, skip):
            continue
        tag = f"gen_{len(out)}"
        out[tag] = {
            "pattern": pattern,
            "kernel": f"{tag}_k",
            "args": tuple(_spec_metas(pattern)),
        }
    return out


#: Creator ops whose result *device* the spec does not pin — the
#: binding mints on the ambient default (CPU) unless told otherwise.
#: Under a device-bound env (the referee's CUDA ``cvals``, a delivery
#: on GPU) a literal ``zeros``/``ones``/… leaf would eval CPU while
#: its siblings are CUDA and the first op mixing them raises
#: mixed-device — the T-024 scan-tier audit residual.  These ops
#: follow the computation's device instead: the same rule
#: ``torch_bridge._randn_param``/``IRModule._align_to`` apply to
#: baked tensors.  The spec's dtype attr still wins; only placement
#: moves.
_DEVICE_FOLLOWERS: frozenset = frozenset(
    {"zeros", "ones", "empty", "randn", "rand", "full", "arange"}
)


def _spec_like(env: dict) -> Any:
    """First non-scalar tensor in *env* — the device context.

    Mirrors ``_const_like``/``_randn_param``: a 0-dim tensor is a
    weak scalar, not a device context, so only ``dim() > 0`` tensors
    qualify.  ``None`` when the env carries no tensor at all — the
    ambient default is then the only honest placement.
    """
    import torch

    for v in env.values():
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            return v
    return None


def _eval_spec(spec: Any, env: dict, bindings: dict) -> Any:
    """Evaluate a canonical spec tuple under a metavar environment."""
    return _eval_spec_dev(spec, env, bindings, _spec_like(env))


def _eval_spec_dev(
    spec: Any, env: dict, bindings: dict, like: Any
) -> Any:
    """Recursive worker for :func:`_eval_spec` carrying *like*."""
    import torch

    if isinstance(spec, str):
        return env[spec]
    if isinstance(spec, (tuple, list)):
        attrs = next((e for e in spec[1:] if isinstance(e, dict)), {})
        args = [
            _eval_spec_dev(e, env, bindings, like)
            for e in spec[1:]
            if not isinstance(e, dict)
        ]
        out = bindings[spec[0]](*args, **attrs)
        if (
            like is not None
            and spec[0] in _DEVICE_FOLLOWERS
            and isinstance(out, torch.Tensor)
            and out.device != like.device
        ):
            out = out.to(device=like.device)
        return out
    return spec  # Const leaf — the canonical form carries the value


def gen_bindings(
    handlers: dict, *, compile_kernels: bool = True
) -> dict:
    """Build each gen kernel's torch ``Binding``.

    The binding re-evaluates the handler's spelled body against its
    positional metavars — generated code, not a table entry.  With
    ``compile_kernels`` the callable is ``torch.compile``-compiled into a
    fused inductor graph: the delivered module genuinely runs one
    kernel where the source spelled several.

    A handler may declare ``"attr_mvars"`` — attr metavariables its
    pattern's attr dicts hold by *name* (the shipped ``synop*`` fold
    calls carry the bound values as real op attrs).  At call time the
    arriving kwargs are ``spec_bind``'d back into the spec before the
    spelled body evaluates; entries without the key evaluate exactly
    as before.
    """
    import mlopt_torch.torch_bridge as tb
    import torch

    from mlopt_discovery.prim_mint import spec_bind

    bindings = tb._IR_TO_TORCH
    out: dict[str, Any] = {}
    for h in handlers.values():
        spec = h["pattern"]
        amvars = tuple(h.get("attr_mvars", ()))
        den = handler_denotation(h)
        if den is not None:
            # seq-scan claim: the spelled spec eval is the ambient
            # map application, not the fold the minted op declares —
            # bind the registered fold denotation instead.  It stays
            # UNcompiled even under ``compile_kernels``: a T-step
            # python loop gains nothing from dynamo, and the Triton
            # tier is the fast delivery path where it binds.
            out[h["kernel"]] = den
            continue

        def fn(*args: Any, s=spec, mvars=amvars, **attrs: Any) -> Any:
            names = _spec_metas(s)
            env = dict(zip(names, args, strict=True))
            body = spec_bind(s, attrs, mvars) if mvars else s
            return _eval_spec(body, env, bindings)

        out[h["kernel"]] = torch.compile(fn) if compile_kernels else fn
    return out


#: Measured-dispatch cache for :func:`_rms_norm_measured` — keyed by
#: ``(dtype, device, normalized_shape, has_weight, input shape)``.
#: Bounded: a long-lived process sweeping many shapes must not grow
#: it without limit; a full cache resets (re-measurement is cheap
#: and rare).
_RMSNORM_PICK: dict = {}
_RMSNORM_CACHE_MAX = 64


def _rmsnorm_spelled(x: Any, ns: tuple, eps: float, w: Any) -> Any:
    """Spell the fold's LHS — ``x·rsqrt(mean(x²)+eps)·w``."""
    import torch

    dims = tuple(range(-len(ns), 0))
    y = x * torch.rsqrt(x.pow(2).mean(dims, keepdim=True) + eps)
    return y if w is None else y * w


def _rms_norm_measured(
    x: Any, w: Any = None, *a: Any, **kw: Any
) -> Any:
    """``rms_norm`` binding that picks its implementation by timing.

    T-026 law-fold residual: ``rms_norm_fold``/``_nogain`` are bank
    laws in ``DEFAULT`` — they fire in saturation and extraction
    prices the ``rms_norm`` member at one op, a cost-model win the
    measured referee never sees (it only gates handler *claims*).
    On this box ``F.rms_norm``'s fp64 kernel is ~7x slower than the
    spelled body at head-width shapes (222 us vs 30 us measured),
    so the fold's delivered module lost 0.15-0.21x wall-clock while
    verifying perfectly.  The op's *denotation* is dtype/shape
    polymorphic; its *implementation* is not — this binding measures
    both spellings once per ``(dtype, device, shape, weight)`` and
    caches the winner, so the law fold keeps its cost win where the
    fused kernel pays and silently delivers the spelled body where
    it does not.  Same function either way — verify is unaffected.
    """
    import torch
    import torch.nn.functional as F
    from mlopt_torch.torch_bridge import attr_of

    ns = tuple(attr_of(kw, "dim", "normalized_shape"))
    eps = float(attr_of(kw, "eps", default=1e-6))
    key = (
        x.dtype,
        str(x.device),
        ns,
        w is not None,
        tuple(x.shape),
    )
    pick = _RMSNORM_PICK.get(key)
    if pick is None:
        fused = lambda: F.rms_norm(  # noqa: E731
            x, ns, weight=w, eps=eps
        )
        spelled = lambda: _rmsnorm_spelled(x, ns, eps, w)  # noqa: E731
        with torch.no_grad():
            for _ in range(3):
                fused()
                spelled()
            t_fused, t_spelled = _paired_time(fused, spelled, 20)
        pick = t_fused <= t_spelled
        if len(_RMSNORM_PICK) >= _RMSNORM_CACHE_MAX:
            _RMSNORM_PICK.clear()
        _RMSNORM_PICK[key] = pick
    if pick:
        return F.rms_norm(x, ns, weight=w, eps=eps)
    return _rmsnorm_spelled(x, ns, eps, w)


#: Ambient-op overrides the generated-kernel sinks install: ops
#: whose shipped binding is measured per-site rather than fixed.
_MEASURED_BINDINGS: dict = {"rms_norm": _rms_norm_measured}


def _ambient_bindings() -> dict:
    """Return the ambient torch table plus measured-dispatch overrides."""
    import mlopt_torch.torch_bridge as tb

    return dict(tb._IR_TO_TORCH) | _MEASURED_BINDINGS


def gen_sink(handlers: dict, *, compile_kernels: bool = True) -> Any:
    """Build a ``TorchSink`` whose vocabulary includes the gen kernels.

    ``supported_ops`` reports the ambient table plus every gen name —
    the feasibility bound the board prices against.  Ambient ops in
    :data:`_MEASURED_BINDINGS` deliver through their measured
    dispatch (see :func:`_rms_norm_measured`).
    """
    from mlopt_core.ops import OpTable
    from mlopt_torch.adapters import TorchSink

    bindings = _ambient_bindings() | gen_bindings(
        handlers, compile_kernels=compile_kernels
    )
    table = OpTable.core().register(torch_bindings=bindings)
    return TorchSink(ops=table)


def _site_env(spec: Any, term: Any) -> dict | None:
    """Return the metavar bindings of the first concrete occurrence."""
    from mlopt_core.ir import Op

    from mlopt_discovery.meta_arena import _concrete_match

    env: dict = {}
    if _concrete_match(spec, term, env):
        return env
    seen: set[int] = set()

    def walk(t: Any) -> dict | None:
        # id-dedup: the match verdict for a node is env-independent
        # (each attempt binds fresh), so a shared subterm need only
        # be tried once — keeps the search DAG-linear on model IRs.
        # Pre-order (node, then children) preserves the original
        # first-occurrence semantics; leaves match too (a bare
        # metavar spec binds them).
        if id(t) in seen:
            return None
        seen.add(id(t))
        sub: dict = {}
        if _concrete_match(spec, t, sub):
            return sub
        if isinstance(t, Op):
            for a in t.args:
                found = walk(a)
                if found is not None:
                    return found
        return None

    if isinstance(term, Op):
        for a in term.args:
            found = walk(a)
            if found is not None:
                return found
    return None


def _bound_val(leaf: Any, var_env: dict, bindings: dict) -> Any:
    """Evaluate one bound metavar leaf under *var_env*.

    IR subterms evaluate through :func:`eval_term`; bound literals
    (a raw ``pow`` exponent, a shape int — non-term leaves a spec
    metavar may legally bind) pass through unchanged — the spelled
    body's binding consumes the scalar directly.
    """
    from mlopt_core.ir import Const, Op, Param, Var
    from mlopt_torch.torch_bridge import eval_term

    if isinstance(leaf, (Op, Var, Const, Param)):
        return eval_term(
            leaf,
            var_env=var_env,
            param_env=var_env,
            bindings=bindings,
            strict=True,
            # a bound metavar can be a whole shared-DAG chunk of a
            # model term — without the memo the eval tree-expands
            # it exponentially.  Fresh dict per leaf: the dedup
            # scope is this one evaluation.
            memo_env={},
        )
    return leaf


def _handler_env(h: dict, term: Any) -> dict | None:
    """Resolve the metavar→leaf map the referee evaluates.

    A ``"leaf_env"`` entry wins when present — handlers minted from
    saturated-graph sites carry the recorded binding because the
    occurrence exists only inside the collector's throwaway
    e-graph, so no raw-IR match can ever rebind it.  Every other
    handler falls back to the first concrete occurrence of its
    pattern in the term.
    """
    leaf = h.get("leaf_env")
    if leaf:
        return leaf
    return _site_env(h["pattern"], term)


def _attr_match(
    pattern: Any,
    term: Any,
    env: dict,
    attrs: dict,
    attr_mvars: tuple,
) -> bool:
    """``_concrete_match`` extended with attr-metavar binding.

    Attr-dict values equal to an attr-mvar *name* bind the site's
    real attr value into *attrs* (consistently — the same mvar must
    bind equal values at every occurrence); every other attr value
    must equal the term's literally, and the attr key sets must
    agree — the same strictness ``_match_node`` applies, with the
    metavar names treated as leaves.
    """
    from mlopt_core.ir import Const, Op

    from mlopt_discovery.meta_arena import _bind_meta

    if isinstance(pattern, str):
        return _bind_meta(pattern, term, env)
    if isinstance(pattern, (tuple, list)) and pattern:
        if not isinstance(term, Op) or term.op != pattern[0]:
            return False
        kids = [e for e in pattern[1:] if not isinstance(e, dict)]
        if len(kids) != len(term.args):
            return False
        pat_attrs = next(
            (e for e in pattern[1:] if isinstance(e, dict)), {}
        )
        if set(term.attrs) != set(pat_attrs):
            return False
        for k, v in pat_attrs.items():
            tv = term.attrs[k]
            if isinstance(v, str) and v in attr_mvars:
                bound = attrs.get(v)
                if bound is None:
                    attrs[v] = tv
                elif bound != tv:
                    return False
            elif tv != v:
                return False
        return all(
            _attr_match(p, t, env, attrs, attr_mvars)
            for p, t in zip(kids, term.args, strict=True)
        )
    return isinstance(term, Const) and term.value == pattern


def attr_site_env(h: dict, term: Any) -> tuple | None:
    """Site an attr-metavar handler; return ``(leaf_env, attrs)``.

    The 5 shipped prim specs whose denotation attr dicts hold
    metavar *names* (``{'dim': 'a97'}``) never concrete-match a
    program subterm — ``_concrete_match`` compares attrs literally.
    This matcher binds those names from the site's real attrs, so
    the fold-cost referee can measure them like any other spec.
    Returns the first occurrence's ``(metavar→leaf env,
    attr-mvar→value attrs)`` or ``None``.
    """
    from mlopt_core.ir import Op

    amvars = tuple(h.get("attr_mvars", ()))
    if not amvars:
        return None

    def walk(spec: Any, t: Any) -> tuple | None:
        env: dict = {}
        attrs: dict = {}
        if _attr_match(spec, t, env, attrs, amvars):
            return env, attrs
        if isinstance(t, Op):
            for a in t.args:
                found = walk(spec, a)
                if found is not None:
                    return found
        return None

    return walk(h["pattern"], term)


def concretize_at_site(h: dict, term: Any) -> dict | None:
    """Return a site-concretized copy of an attr-metavar handler.

    The copy's ``"pattern"`` has the bound attr values frozen in
    (``spec_bind``) and ``"attr_mvars"`` emptied, so
    ``_handler_env``/``profitable`` treat it like any concrete
    spec.  Measurement-only: callers that *deliver* must keep the
    original handler — the op is polymorphic over the attr values
    and one site's freeze would mis-denote another's.
    """
    from mlopt_discovery.prim_mint import spec_bind

    site = attr_site_env(h, term)
    if site is None:
        return None
    _env, attrs = site
    return {
        **h,
        "pattern": spec_bind(
            h["pattern"], attrs, tuple(h.get("attr_mvars", ()))
        ),
        "attr_mvars": (),
    }


def _referee_kernel(
    kernel_for: Any, name: str, h: dict, vals: dict, kerneled: Any
) -> tuple:
    """Resolve the callable AND the env the referee times.

    A ``kernel_for`` result is a callable — or a ``(callable,
    env)`` pair when the delivery path needs a different
    environment (a CUDA-bound kernel must be timed against a
    CUDA-bound spelled body, or the referee compares CPU BLAS to
    a GPU kernel — a race no measurement should rig).  Missing or
    ``None`` results fall back to the ``torch.compile``d spelled
    body — the pointwise tier's own delivery path — under the
    original env.

    The fallback is timed on the *delivery* device: the sinks this
    referee grades lower onto CUDA whenever it exists, and a
    ``torch.compile``d spec-walk that wins on CPU can lose there
    (dynamo graph-breaks on the recursive eval leave per-call
    guard overhead the eager spelled side never pays — the T-026
    ``synop_4c5296b3ba`` residual: kept at 0.64x on CPU values,
    delivered at 0.54x on CUDA).  When CUDA is available the
    fallback env moves with it; a move failure keeps the host env.
    """
    import torch

    if kernel_for is not None:
        k = kernel_for(name, h, vals)
        if isinstance(k, tuple):
            return k[0], k[1]
        if k is not None:
            return k, vals
    if torch.cuda.is_available():
        try:
            cvals = {
                m: (v.cuda() if torch.is_tensor(v) else v)
                for m, v in vals.items()
            }
        except (AttributeError, RuntimeError, TypeError):
            cvals = vals
        return torch.compile(kerneled), cvals
    return torch.compile(kerneled), vals


def _denotation_kernel(
    kernel_for: Any, name: str, h: dict, vals: dict
) -> tuple:
    """``kernel_for`` consult for denotation-backed claims.

    Same ``(fn, env)`` protocol as :func:`_referee_kernel` but NO
    spelled-eval fallback: a claim whose spelled arm is a registered
    denotation (the seq fold) delivers *that denotation* whenever no
    codegen tier binds, so ``None`` here means kernel ≡ spelled —
    the caller declines rather than timing a race it cannot win.
    """
    if kernel_for is None:
        return None, vals
    k = kernel_for(name, h, vals)
    if isinstance(k, tuple):
        return k[0], k[1]
    if k is not None:
        return k, vals
    return None, vals


def _host(x: Any) -> Any:
    """Move a tensor result host-side for same-device compares."""
    import torch

    return x.cpu() if isinstance(x, torch.Tensor) else x


def _referee_out(kernel: Any, vals: dict, mvs: list) -> Any:
    """Run the kernel on the bound values, host-side.

    Device-resident kernels (the generated Triton tiers) return
    CUDA tensors; the spelled body evaluates host-side — move the
    result so the referee's ``allclose`` compares same-device.
    """
    return _host(kernel(*[vals[m] for m in mvs]))


#: Per-handler referee timing log — appended by :func:`profitable`
#: with ``(name, t_spelled, t_kernel, window, keep_vote)`` per timing
#: sample.  A handler measured once contributes one row; a borderline
#: handler (see ``_REFEREE_BORDERLINE``) contributes one row per
#: confirmation window, consecutive in the log, and its keep/drop
#: decision is the majority of those rows' votes.  Measurement
#: surface only (the ``TRITON_REFEREE_LOG`` precedent in ``play``):
#: the flip-rate probe reads it to correlate keep/drop outcomes with
#: the measured ratios.
PROFITABLE_LOG: list = []

#: Borderline band around the keep threshold, as a ratio multiplier.
#: The referee's interleaved A/B windows still swing ±10-15% on this
#: host (the T-026 residual: identical-binary reruns flip keep-sets —
#: ChunkedSSD ``scan_2`` read 0.93x once against a ~1.05x typical, a
#: chainfuse ``gen_1`` read 1.03x against ~0.9x).  A measured ratio
#: inside ``margin / _REFEREE_BORDERLINE .. margin *
#: _REFEREE_BORDERLINE`` cannot be trusted from one window, so the
#: handler is re-timed (up to two extra windows, stopped early on
#: agreement) and decided by majority.  Handlers outside the band
#: keep the single-sample cost; the sweep's worst case is ~3x on the
#: borderline subset only, typically far less.
_REFEREE_BORDERLINE = 1.25


def _handler_mvs(h: dict) -> list:
    """Ordered metavar names a handler's callables take positionally.

    The spec metavars (``_spec_metas(pattern)`` — the ``params``
    order every scan denotation/binding shares) plus any read-out
    operand metas a ``kind="seqt"`` record binds: the fused
    read-out's per-step contraction operand is a real kernel arg,
    not a fold operand, so it rides after the params in both the
    Triton binding and the spelled denotation.
    """
    mvs = list(_spec_metas(h["pattern"]))
    for op in (h.get("readout") or {}).get("ops", ()):
        mvs.append(op["meta"])
    return mvs


def profitable(
    handlers: dict,
    term: Any,
    var_env: dict,
    *,
    reps: int = 40,
    margin: float = 1.0,
    kernel_for: Any = None,
    denotation_for: Any = None,
    max_handlers: int | None = None,
    time_budget: float | None = None,
) -> dict:
    """Keep only handlers whose compiled kernel beats the spelled site.

    The measured-cost referee: for each minted handler, locate its
    concrete occurrence, evaluate the bound subterms once, and time
    the spelled body against the generated compiled callable on those
    real values.  A claim whose kernel doesn't actually pay (≥
    ``margin`` x faster than its spelled body) is filtered out —
    op-count pricing can't see dynamo overhead, so measurement, not
    the cost model, decides which claims are worth offering.

    *kernel_for* overrides what gets timed: ``kernel_for(name, h,
    vals)`` returns the delivery-path callable — the generated
    Triton kernel for a non-elementwise tier — or ``None`` to keep
    the ``torch.compile``d spelled body (the pointwise path's own
    delivery).  The referee times and verifies *that* callable, so
    a tier is credited only what it would actually ship.

    *denotation_for* overrides the SPELLED arm for claims whose
    spec tuple does not denote the minted op's semantics:
    ``denotation_for(name, h, vals)`` returns a positional callable
    ``fn(*metavar_vals)`` implementing the declared denotation —
    today only the ``seq``-form scan fold
    (:func:`handler_denotation`/``seq_scan_denotation`), whose
    ``("applyd", …)`` spec evals as one map application.  When it
    returns a callable, the referee times ``kernel_for``'s result
    against THAT fold and — critically — has NO spelled-eval
    fallback: if no codegen tier binds, delivery IS the denotation
    itself, so the claim is declined outright rather than timed
    fold-vs-map or fold-vs-fold.

    *max_handlers* bounds the sweep for model-scale terms: only the
    first *max_handlers* entries of the sorted handler table are
    priced — the rest are dropped (a claim the referee never
    measured is never offered; capping is a probe-size bound, not
    a relaxation of the measurement).  *time_budget* (seconds,
    per claim) shrinks each claim's ``_paired_time`` window: after
    the verify warmup the referee estimates one call's cost and
    picks ``reps`` to fit the allowance (floor 3, ceiling *reps*).
    Borderline re-windows each pay the same budget again, so a
    claim's worst case is ~3x *time_budget*.

    Handlers minted from saturated-graph sites carry ``"leaf_env"``
    — the metavar→subterm map recorded at harvest; the referee
    evaluates it directly (:func:`_handler_env`), since the
    occurrence lives only in the collector's throwaway e-graph.
    """
    import mlopt_torch.torch_bridge as tb
    import torch

    # measurement integrity — see gen_probe: per-handler compiled
    # fns would exhaust dynamo's default recompile limit (8) and
    # silently degrade to eager mid-sweep.
    _dyno_cfg: Any = torch._dynamo.config
    _dyno_cfg.recompile_limit = max(int(_dyno_cfg.recompile_limit), 256)

    # measurement hygiene: drain in-flight inductor compile workers
    # before timing.  A backlog from earlier compiles in the same
    # process (the case build's own claim sweep, sink builds)
    # inflates wall-clock disproportionately on the spelled arm —
    # it issues N torch calls + a spec walk per invocation where the
    # fused kernel issues one — flipping measured losers into false
    # keeps (the T-016 residual: ``_fold_cost_gate`` kept 35 prim
    # folds on a contaminated window, 0 after the drain; a delivered
    # ``synop_2b8cfd1c90`` read 0.668x).  ``shutdown`` waits for
    # pending work; the per-handler verify compiles below then run
    # synchronously inside their own (untimed) warmup.
    try:
        from torch._inductor.async_compile import (
            shutdown_compile_workers,
        )

        shutdown_compile_workers()
    except Exception:
        pass

    bindings = tb._IR_TO_TORCH
    keep: dict[str, dict] = {}
    # sorted order: callers pass frozenset-derived dicts whose
    # iteration order varies per process (hash seed) — which handler
    # absorbs first-call warmup noise must not be a coin flip.  The
    # ``max_handlers`` cap takes the same deterministic prefix.
    items = sorted(handlers.items())
    if max_handlers is not None:
        items = items[: max(0, int(max_handlers))]
    for name, h in items:
        env = _handler_env(h, term)
        if env is None:
            continue
        try:
            vals = {
                k: _bound_val(v, var_env, bindings)
                for k, v in env.items()
            }
        except Exception:
            continue

        def kerneled(*a: Any, s=h["pattern"]) -> Any:
            return _eval_spec(
                s,
                dict(zip(_spec_metas(s), a, strict=True)),
                bindings,
            )

        mvs = _handler_mvs(h)
        den = (
            denotation_for(name, h, vals)
            if denotation_for is not None
            else None
        )
        if den is not None:
            kernel, vals = _denotation_kernel(kernel_for, name, h, vals)
            if kernel is None:
                # no codegen bound: delivery IS the denotation —
                # there is no faster arm to time, so the claim
                # cannot pay (fold-vs-fold would read ~1.0 anyway)
                continue
        else:
            kernel, vals = _referee_kernel(
                kernel_for, name, h, vals, kerneled
            )

        if den is not None:

            def spelled(v=vals, p=den, ms=mvs) -> Any:
                return p(*[v[m] for m in ms])

        else:

            def spelled(s=h["pattern"], v=vals) -> Any:
                return _eval_spec(s, v, bindings)

        # the verify-before-time gate (and the timing itself) live in
        # the helper; a kernel that raises on its bound values — a
        # site the codegen accepted but cannot lower — declines there
        # rather than aborting the whole sweep.
        timings = _kernel_pays(
            kernel, spelled, vals, mvs, reps, time_budget=time_budget
        )
        if timings is None:
            continue
        t_spelled, t_kernel, reps_eff = timings

        def kcall(
            kernel: Any = kernel, vals: dict = vals, mvs: list = mvs
        ) -> Any:
            return kernel(*[vals[m] for m in mvs])

        samples = [(t_spelled, t_kernel)]
        # borderline guard: a ratio inside the noise band gets up to
        # two confirmation windows — the second only when the first
        # disagrees — and the keep/drop decision is the majority.
        # ``_paired_time`` re-warms both arms, so each sample is an
        # independent interleaved window, not a continuation of the
        # first.
        ratio = t_kernel / t_spelled if t_spelled > 0 else float("inf")
        if (
            margin / _REFEREE_BORDERLINE
            <= ratio
            <= margin * _REFEREE_BORDERLINE
        ):
            # confirmation windows pay the SAME budget-derived rep
            # count — an unbounded re-window would let a borderline
            # claim spend ~3x the caller's allowance
            samples.append(_paired_time(spelled, kcall, reps_eff))
            # a third window only breaks a first/second disagreement;
            # two agreeing windows decide on their own
            if (samples[0][1] <= samples[0][0] * margin) != (
                samples[1][1] <= samples[1][0] * margin
            ):
                samples.append(_paired_time(spelled, kcall, reps_eff))
        for window, (ts_i, tk_i) in enumerate(samples):
            PROFITABLE_LOG.append(
                (name, ts_i, tk_i, window, tk_i <= ts_i * margin)
            )
        votes = sum(
            1 for ts_i, tk_i in samples if tk_i <= ts_i * margin
        )
        if votes * 2 > len(samples):
            keep[name] = h
    return keep


def _median_time(fn: Any, reps: int) -> float:
    """Median per-call seconds of ``fn()`` over *reps* runs.

    Five warmup calls precede timing (dynamo/inductor compilation and
    cudnn autotune must settle); CUDA is synchronised around each
    timed call so the measurement covers the work, not the enqueue.
    """
    import time

    import torch

    for _ in range(5):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def _paired_time(a: Any, b: Any, reps: int) -> tuple:
    """Median per-call seconds of ``a`` and ``b``, interleaved.

    Two sequential ``_median_time`` windows drift: clocks ramp, the
    scheduler moves, an unrelated process lands — and the ratio
    inherits the whole excursion (the T-026 residual: the same
    ``synop2_*`` site read 0.64x and ~2.2x across processes).  Here
    both sides warm up together, then alternate inside one loop —
    even rounds ``a``-then-``b``, odd rounds ``b``-then-``a`` — so a
    slow drift cancels instead of landing wholly inside one arm.
    """
    import time

    import torch

    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(8):
        a()
        b()
    _sync()
    ta: list[float] = []
    tb: list[float] = []
    for i in range(reps):
        if i % 2 == 0:
            t0 = time.perf_counter()
            a()
            _sync()
            ta.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            b()
            _sync()
            tb.append(time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            b()
            _sync()
            tb.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            a()
            _sync()
            ta.append(time.perf_counter() - t0)
    ta.sort()
    tb.sort()
    return ta[len(ta) // 2], tb[len(tb) // 2]


def _kernel_pays(
    kernel: Any,
    spelled: Any,
    vals: dict,
    mvs: list,
    reps: int,
    *,
    time_budget: float | None = None,
) -> tuple | None:
    """Time *kernel* against its spelled body; ``None`` declines.

    The referee verifies before it times: the kernel must reproduce
    the spelled body on the bound values — an inductor miscompile or
    dtype drift would otherwise win the timing race it should have
    lost.  A kernel that RAISES on its bound values — a site the
    codegen accepted but cannot lower, e.g. operands sharing no
    broadcast so the unrolled binding's own shape check fires —
    declines here exactly as a binding-builder decline does: the
    measured-cost filter must not abort the whole sweep on one
    unloweable site.

    *time_budget* (seconds) shrinks the paired window to fit a
    wall-clock allowance: one timed call per arm estimates the
    per-rep cost and ``reps`` is clamped into ``[3, reps]`` — floor
    3 keeps the median meaningful, ceiling preserves the caller's
    statistical request.  Both arms are already warm here (the
    verify ran each once), so the estimate is one extra call each.
    Returns ``(t_spelled, t_kernel, reps_eff)`` — the caller reuses
    ``reps_eff`` for borderline confirmation windows so they honour
    the same budget.
    """
    import time

    import torch

    try:
        out = _referee_out(kernel, vals, mvs)
        if not torch.allclose(
            out,
            _host(spelled()),
            rtol=1e-4,
            atol=1e-5,
        ):
            return None
        reps_eff = reps
        if time_budget is not None:
            kcall = lambda: kernel(*[vals[m] for m in mvs])  # noqa: E731
            t0 = time.perf_counter()
            spelled()
            t_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            kcall()
            t_k = time.perf_counter() - t0
            est = max(t_s + t_k, 1e-9)
            reps_eff = max(3, min(int(reps), int(time_budget / est)))
        t_spelled, t_kernel = _paired_time(
            spelled,
            lambda: kernel(*[vals[m] for m in mvs]),
            reps_eff,
        )
    except Exception:
        return None
    return t_spelled, t_kernel, reps_eff


def _kernel_cache() -> Any:
    """Return the generated-kernel module cache dir."""
    # real files, inspectable — the artifact is a module on disk
    import tempfile
    from pathlib import Path

    d = Path(tempfile.gettempdir()) / "mlopt_genkernels"
    d.mkdir(exist_ok=True)
    return d


def _load_generated_module(path: Any) -> Any:
    """Import a generated ``.py`` kernel module; ``None`` to decline.

    Same contract as :func:`_load_generated` but returns the module
    itself — the v2 reduction kernels ship several ``@triton.jit``
    functions per artifact (``_gr``/``_gc`` or the softmax family),
    not a single entry point.
    """
    import importlib.util

    spec_ = importlib.util.spec_from_file_location(path.stem, path)
    if spec_ is None or spec_.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


def _load_generated(path: Any, name: str) -> Any:
    """Import a generated ``.py`` kernel module; ``None`` to decline.

    The artifact is a real module file — inspectable on disk, not an
    exec'd string.  Loading through importlib keeps the repo's
    no-eval/exec invariant intact.  A ``None`` spec/loader means the
    import machinery declined the file; the caller keeps whatever
    fallback path it had.
    """
    mod = _load_generated_module(path)
    return getattr(mod, name) if mod is not None else None


_TRITON_UNARY = {
    "neg": "(-({a}))",
    "abs": "tl.abs({a})",
    "square": "(({a}) * ({a}))",
    "sigmoid": "(1.0 / (1.0 + tl.exp(-({a}))))",
    "tanh": "libdevice.tanh({a})",
    "relu": "tl.maximum({a}, 0.0)",
    "gelu": (
        "(0.5 * ({a}) * (1.0 + libdevice.erf("
        "({a}) * 0.70710678118654752)))"
    ),
    "softplus": "libdevice.log1p(tl.exp({a}))",
    "mish": "(({a}) * libdevice.tanh(libdevice.log1p(tl.exp({a}))))",
    "silu": "(({a}) / (1.0 + tl.exp(-({a}))))",
    "exp": "tl.exp({a})",
    "log": "tl.log({a})",
    "sqrt": "tl.sqrt({a})",
    "rsqrt": "(1.0 / tl.sqrt({a}))",
    "sin": "tl.sin({a})",
    "cos": "tl.cos({a})",
    "identity": "({a})",
    "reciprocal": "(1.0 / ({a}))",
    "softsign": "(({a}) / (1.0 + tl.abs({a})))",
    "softsign_gate": "((({a}) / (1.0 + tl.abs({a}))) * ({b}))",
    "atanh": "libdevice.atanh({a})",
}
_TRITON_BINARY = {
    "add": "(({a}) + ({b}))",
    "sub": "(({a}) - ({b}))",
    "mul": "(({a}) * ({b}))",
    "div": "(({a}) / ({b}))",
    "pow": "libdevice.pow({a}, {b})",
}


#: Const-leaf rendering for the 1-D kernels (``_gk``/``_gk_s`` and
#: both ``_gs`` scan kernels — all load ``[BLOCK]`` tiles and take
#: ``out_ptr``/``BLOCK`` params).  A bare ``2.0`` literal is an fp32
#: scalar in Triton: the semantic ops (``+``/``*``/``tl.maximum`` …)
#: promote it to the tile dtype, but ``libdevice.*`` calls require
#: both operands at the tile's dtype — ``libdevice.pow(f64_tile,
#: 2.0)`` is a hard compile-time type clash (the T-024 audit's one
#: ``raised`` consult, fp64 ``pow``).  ``tl.full`` typed by the
#: output pointer's element type follows the launch dtype, so the
#: same source compiles for fp32 and fp64.
_TRITON_CONST_1D = (
    "tl.full([BLOCK], {v}, dtype=out_ptr.dtype.element_ty)"
)


def _triton_expr(spec: Any, const: str | None = None) -> str | None:
    """Render a canonical spec as a Triton expression, or decline.

    The codegen map — each spec node becomes a ``tl``/libdevice
    expression over its metavar operands.  Constant leaves inline
    as literals; *const* is an optional ``{v}``-format template for
    how a leaf renders (the kernel emitters pass a dtype-typed
    ``tl.full`` so ``libdevice`` calls see matching operand dtypes —
    see :data:`_TRITON_CONST_1D`); ``None`` keeps the bare fp32
    scalar.  Attr dicts on ops this tier doesn't model make the
    whole spec decline honestly (``None``).
    """
    return _triton_expr_h(spec, None, const)


def _triton_expr_h(
    spec: Any, hoists: dict | None, const: str | None = None
) -> str | None:
    """Render *spec*; unrenderable subterms hoist into *hoists*.

    When *hoists* is given, a node outside the elementwise grammar
    that is not itself a metavar/constant — ``decay_vec(g, …)``, a
    view op, a compound like ``om_decayed`` — registers as a
    ``__c<i>`` computed operand instead of declining the whole
    spec: the binding evaluates the subterm on the host and passes
    it as another kernel argument.  ``None`` in *hoists* keeps the
    old decline behaviour.  *const* is forwarded to every leaf —
    see :func:`_triton_expr`.
    """
    if isinstance(spec, str):
        return spec
    if not isinstance(spec, (tuple, list)):
        v = float(spec)
        if const is not None:
            return const.format(v=repr(v))
        return f"({v})"
    return _triton_node_h(spec, hoists, const)


#: Attr-bearing elementwise ops the Triton renderer models inline —
#: the clamp family.  Their bounds ride the spec's attr dict
#: (``{"min": …, "max": …}``; either side may be absent or ``None``,
#: matching ``torch.clamp``'s partial-bound spelling).  Every other
#: attr-carrying op still hoists (a host-computed ``__c`` operand)
#: or declines, exactly as before.
_TRITON_ATTR_OPS = frozenset({"clamp", "clamp_min", "clamp_max"})


def _triton_clamp(
    op: str,
    attrs: dict,
    kid: Any,
    hoists: dict | None,
    const: str | None,
) -> str | None:
    """Render a clamp-family node; ``None`` leaves hoist/decline.

    Each present bound renders through the caller's *const* template
    — the same dtype-typed ``tl.full`` a const leaf gets, so the
    ``tl.maximum``/``tl.minimum`` semantic ops see the launch dtype
    on fp64 tiles and broadcast to whatever tile rank the tier
    emits.  A bound that is absent/``None`` applies no clamp on that
    side; a bound that is not a real number declines the inline
    render (the hoist path still evaluates it on the host).
    """
    want = {
        "clamp": ("min", "max"),
        "clamp_min": ("min",),
        "clamp_max": ("max",),
    }[op]
    a = _triton_expr_h(kid, hoists, const)
    if a is None:
        return None
    expr = a
    applied = 0
    for key in want:
        v = attrs.get(key)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        lit = (
            const.format(v=repr(float(v)))
            if const is not None
            else f"({float(v)})"
        )
        fn = "tl.maximum" if key == "min" else "tl.minimum"
        expr = f"{fn}({expr}, {lit})"
        applied += 1
    # a clamp with no applicable bound is not the spelled op —
    # decline rather than silently deliver the identity.
    return expr if applied else None


def _triton_node_h(
    spec: Any, hoists: dict | None, const: str | None = None
) -> str | None:
    """Render one op node — dispatch, or hoist the whole subterm."""
    kids = [e for e in spec[1:] if not isinstance(e, dict)]
    if len(spec[1:]) != len(kids):
        # attr-carrying node — the clamp grammar renders its bounds
        # inline; anything else hoists (a host-computed operand) or
        # declines, exactly as before.
        attrs = [e for e in spec[1:] if isinstance(e, dict)]
        if (
            spec[0] in _TRITON_ATTR_OPS
            and len(kids) == 1
            and len(attrs) == 1
            and set(attrs[0]) <= {"min", "max"}
        ):
            out = _triton_clamp(
                spec[0], attrs[0], kids[0], hoists, const
            )
            if out is not None:
                return out
        return _hoist(spec, hoists) if hoists is not None else None
    # ``pow(x, 2)`` lowers to ``libdevice.pow`` — a ~150µs fp64
    # penalty measured on the reduction probe (RMSNormPerHead) where
    # the spelled ``torch.pow`` costs ~20µs.  ``x * x`` is the same
    # value bit-for-bit on IEEE floats and what ``square`` already
    # spells; only the literal-2 exponent takes the shortcut.
    if (
        spec[0] == "pow"
        and len(kids) == 2
        and isinstance(kids[1], (int, float))
        and not isinstance(kids[1], bool)
        and kids[1] == 2
    ):
        a = _triton_expr_h(kids[0], hoists, const)
        if a is not None:
            return f"(({a}) * ({a}))"
    args = [_triton_expr_h(k, hoists, const) for k in kids]
    if any(a is None for a in args):
        return None
    out = _triton_dispatch(spec[0], args)
    if out is None and hoists is not None:
        return _hoist(spec, hoists)
    return out


def _hoist(spec: Any, hoists: dict) -> str:
    """Register *spec* as a host-computed operand; return its name."""
    name = f"__c{len(hoists)}"
    hoists[name] = spec
    return name


def _triton_dispatch(op: str, args: list) -> str | None:
    """Format the rendered kid expressions into one op expression."""
    if op in _TRITON_BINARY and len(args) == 2:
        return _TRITON_BINARY[op].format(a=args[0], b=args[1])
    if op == "softsign_gate" and len(args) == 2:
        return _TRITON_UNARY[op].format(a=args[0], b=args[1])
    if op in _TRITON_UNARY and len(args) == 1:
        return _TRITON_UNARY[op].format(a=args[0])
    return None


#: Kernel-local names a spec metavar must not reuse.  A meta named
#: ``off``/``mask``/``out`` would shadow the index/mask/store locals
#: (the value load would overwrite them before ``tl.store`` reads
#: them) — a silent miscompile the verify leg would only catch after
#: the fact.  Decline honestly at codegen instead.
_TRITON_LOCALS = frozenset(
    {
        "off",
        "mask",
        "out",
        "n",
        "BLOCK",
        "ometa_ptr",
        "off64",
        "__sz",
        "__stb",
        "__cd",
    }
)

#: Strided-kernel broadcast-metadata cache — ``ometa`` tensors are
#: keyed by ``(out_shape, per-operand (shape, strides, device))`` and
#: shared across every generated kernel (the layout is handler-
#: independent).  Bounded like ``_RMSNORM_PICK``: a long-lived sweep
#: must not grow it without limit.
_TRITON_OMETA: dict = {}
_TRITON_OMETA_MAX = 256


def _triton_ometa(xs: list, out_shape: tuple, device: Any) -> Any:
    """Pack the strided kernel's broadcast metadata; cached per layout.

    Layout: ``out_shape`` (K entries), the output's row-major strides
    (K), then each operand's K *effective* strides — the operand's
    real element stride on dims it spans, ``0`` on broadcast dims
    (size-1 dims read every output element from one lane).  The
    kernel reconstructs each element's source offset as
    ``sum(coord_d * eff_stride_d)`` over the flat output index.
    """
    import torch

    key = (
        out_shape,
        tuple(
            (tuple(x.shape), tuple(x.stride()), str(x.device))
            for x in xs
        ),
    )
    om = _TRITON_OMETA.get(key)
    if om is not None:
        return om
    k = len(out_shape)
    ostr: list = []
    acc = 1
    for s in reversed(out_shape):
        ostr.insert(0, acc)
        acc *= s
    buf = [int(s) for s in out_shape] + ostr
    for x in xs:
        pad = k - x.dim()
        eff = [0] * k
        for j in range(x.dim()):
            if x.shape[j] != 1:
                eff[j + pad] = x.stride(j)
        buf.extend(eff)
    om = torch.tensor(buf, dtype=torch.int64, device=device)
    if len(_TRITON_OMETA) >= _TRITON_OMETA_MAX:
        _TRITON_OMETA.clear()
    _TRITON_OMETA[key] = om
    return om


def triton_bindings(handlers: dict) -> dict:
    """Generate a Triton kernel per handler — one nobody wrote.

    Unlike :func:`gen_bindings` (whose callable wraps
    ``torch.compile`` — the incumbent's own codegen), this tier
    emits real ``triton.jit`` source for the fused body and
    compiles it directly.  Specs whose ops the codegen doesn't model
    are skipped — the handler keeps its torch path instead.

    Two launch paths share one generated module:

    * ``_gk`` — the flat kernel, for the common case where every
      operand already has the broadcast output shape contiguous
      (index ``off`` reads every operand directly);
    * ``_gk_s`` — the strided kernel, for broadcast or non-
      contiguous operands.  The old wrapper ``broadcast_tensors`` +
      ``.contiguous()`` *materialized* a full-size copy per
      broadcast operand — O(M·n) hidden traffic plus a copy launch
      each, which is why the "fused" kernel lost to spelled eager
      at exactly the broadcast-heavy sites the zoo folds (the T-016
      residual: 0.48-0.85x at every size tested, vs ~2x once the
      copies are gone).  ``_gk_s`` passes each operand's shape/
      strides through ``ometa`` instead and computes source offsets
      in-kernel — no host materialization.
    """
    import torch

    out: dict[str, Any] = {}
    for h in handlers.values():
        spec = h["pattern"]
        metas = _spec_metas(spec)
        body = _triton_expr(spec, _TRITON_CONST_1D)
        if (
            body is None
            or not metas
            or any(m in _TRITON_LOCALS for m in metas)
            or any(
                m.startswith("__o") and m[3:].isdigit() for m in metas
            )
        ):
            continue
        ptrs = ", ".join(f"{m}_ptr" for m in metas)
        loads = "".join(
            f"    {m} = tl.load({m}_ptr + off, mask=mask, other=0.0)\n"
            for m in metas
        )
        nmetas = len(metas)
        accs = "".join(
            f"    __o{i} = tl.zeros([BLOCK], dtype=tl.int64)\n"
            for i in range(nmetas)
        )
        dims = (
            "    for d in tl.static_range(NDIM):\n"
            "        __sz = tl.load(ometa_ptr + d)\n"
            "        __stb = tl.load(ometa_ptr + NDIM + d)\n"
            "        __cd = (off64 // __stb) % __sz\n"
        )
        dims += "".join(
            f"        __o{i} += __cd * tl.load("
            f"ometa_ptr + {2 + i} * NDIM + d)\n"
            for i in range(nmetas)
        )
        sloads = "".join(
            f"    {m} = tl.load({m}_ptr + __o{i}, mask=mask, "
            f"other=0.0)\n"
            for i, m in enumerate(metas)
        )
        src = (
            "import triton\n"
            "import triton.language as tl\n"
            "import triton.language.extra.libdevice as libdevice\n"
            "\n@triton.jit\n"
            f"def _gk({ptrs}, out_ptr, n, BLOCK: tl.constexpr):\n"
            "    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
            "    mask = off < n\n"
            f"{loads}"
            f"    out = {body}\n"
            "    tl.store(out_ptr + off, out, mask=mask)\n"
            "\n@triton.jit\n"
            f"def _gk_s({ptrs}, out_ptr, n, ometa_ptr, "
            "BLOCK: tl.constexpr, NDIM: tl.constexpr):\n"
            "    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
            "    mask = off < n\n"
            "    off64 = off.to(tl.int64)\n"
            f"{accs}"
            f"{dims}"
            f"{sloads}"
            f"    out = {body}\n"
            "    tl.store(out_ptr + off, out, mask=mask)\n"
        )
        # the artifact is a real module file — inspectable on disk,
        # not an exec'd string.  Loading through importlib keeps the
        # repo's no-eval/exec invariant intact.
        path = _kernel_cache() / f"_gk_{h['kernel']}.py"
        path.write_text(src)
        kern = _load_generated(path, "_gk")
        kern_s = _load_generated(path, "_gk_s")
        if kern is None or kern_s is None:
            continue  # module load declined — keep the torch path

        def fn(
            *args: Any,
            kern=kern,
            kern_s=kern_s,
            **_attrs: Any,
        ) -> Any:
            xs = list(args)
            out_shape = torch.broadcast_shapes(*[x.shape for x in xs])
            n = 1
            for s in out_shape:
                n *= s
            out = torch.empty(
                out_shape, dtype=xs[0].dtype, device=xs[0].device
            )
            if n == 0:
                return out
            block = 1024
            grid = ((n + block - 1) // block,)
            flat = out.reshape(-1)
            if all(
                x.is_contiguous() and x.shape == out_shape for x in xs
            ):
                kern[grid](
                    *[x.reshape(-1) for x in xs],
                    flat,
                    n,
                    BLOCK=block,
                )
            else:
                om = _triton_ometa(xs, out_shape, xs[0].device)
                kern_s[grid](
                    *xs,
                    flat,
                    n,
                    om,
                    BLOCK=block,
                    NDIM=len(out_shape),
                )
            return out

        out[h["kernel"]] = fn
    return out


def triton_sink(handlers: dict) -> Any:
    """Build a ``TorchSink`` binding gen ops to Triton kernels.

    Supported ops = ambient + handler kernels whose spec the
    codegen accepted — a claim on a declined spec stays
    unpriced-unlowerable via the ambient table, the same posture
    as ``gen_sink``.
    """
    from mlopt_core.ops import OpTable
    from mlopt_torch.adapters import TorchSink

    bindings = _ambient_bindings()
    bindings.update(triton_bindings(handlers))
    table = OpTable.core().register(torch_bindings=bindings)
    return TorchSink(ops=table)


def _seq_scan_spec(spec: Any) -> bool:
    """Check *spec* is a scan spec in a sequence (fold) form."""
    info = scan_spec(spec)
    return info is not None and info["kind"] in ("seq", "seqs", "seqt")


class Tier(NamedTuple):
    """One codegen tier — the data engines iterate, not branches.

    A tier declares *what it mints* (``handlers`` — spec list →
    handler dicts) and *what it lowers to* (``triton`` — handlers →
    generated bindings, or ``None`` when the tier's delivery is the
    compiled spelled body).  Site collection is play's own knowledge
    (the ``_spec_of`` traversal), mapped to tier names there; adding
    a tier is a registry entry + a site collector, never an engine
    branch.
    """

    name: str
    handlers: Any
    triton: Any  # handlers → generated bindings; None = compiled


def hybrid_sink(handlers: dict, *, compile_kernels: bool = True) -> Any:
    """Build a sink preferring Triton codegen, compile fallback.

    Every codegen tier is consulted.  Triton-accepted
    elementwise specs bind to the generated ``.py`` kernel — a
    kernel no incumbent compiler produces — ``applyd``-carrier
    specs the scan codegen accepts bind to their generated fold
    kernel, and ``om_decayed``/spelled ``matmul(mul(s,d),v)`` specs
    bind to the generated fused-contraction kernel.
    Specs the codegen declines fall back to the
    ``torch.compile``-wrapped body — including sequence-form scan
    specs, whose fallback is the registered fold denotation
    (:func:`seq_scan_denotation` via :func:`gen_bindings`): the
    same function the minted op declares, evaluated serially —
    slow but never the ambient map application ``_eval_spec``
    would silently deliver.  Unrolled compose-tree specs keep the
    ordinary spelled fallback — their spelled eval IS the carrier
    semantics, just serial.
    """
    from mlopt_core.ops import OpTable
    from mlopt_torch.adapters import TorchSink

    plain = {
        k: h
        for k, h in handlers.items()
        if not _seq_scan_spec(h["pattern"])
    }
    bindings = _ambient_bindings()
    # ``handlers``, not ``plain``: ``gen_bindings`` now binds seq
    # scans to their fold denotation — the spelled-eval map
    # application is never an option — and ``scan_triton_bindings``
    # below overrides it wherever the generated kernel binds.
    bindings.update(
        gen_bindings(handlers, compile_kernels=compile_kernels)
    )
    bindings.update(triton_bindings(plain))
    bindings.update(scan_triton_bindings(handlers))
    bindings.update(decayed_triton_bindings(handlers))
    bindings.update(reduction_triton_bindings(handlers))
    table = OpTable.core().register(torch_bindings=bindings)
    return TorchSink(ops=table)


def novel_cost(base: Any, is_novel: Any, bonus: float = 0.5) -> Any:
    """Discount members headed by a novel (declared/gen) op.

    Extraction-priced novelty: a member whose head satisfies
    *is_novel* — e.g. ``op.startswith(("claim_", "foldabs_", "gen_"))``
    — bills ``bonus`` less than its spelled equivalent: the search's
    reason to prefer the invented form when costs tie.  Feasibility
    is untouched: the member must already be lowerable.
    """
    from mlopt_core.ir import Op

    def cost(node: Any) -> float:
        c = base(node)
        if isinstance(node, Op) and is_novel(node.op):
            return max(c - bonus, 0.0)
        return c

    return cost


# ---------------------------------------------------------------------------
#  The scan tier — generated kernels for the diagonal-affine carrier
# ---------------------------------------------------------------------------

#: Tile width for generated scan kernels — one Triton program per
#: ``BLOCK`` columns of the flattened state space.
_SCAN_BLOCK = 1024

#: Names the generated scan source already binds — a metavar colliding
#: with one (or with the generated ``*_ptr`` parameter suffix) would
#: silently shadow a program variable, so such specs decline codegen.
_SCAN_RESERVED = frozenset(
    {
        "acc",
        "off",
        "mask",
        "row",
        "n",
        "N",
        "T",
        "t",
        "BLOCK",
        "out",
        "program_id",
    }
)


def _meta_name_ok(name: Any) -> bool:
    """Check *name* is safe as a generated-kernel variable."""
    return (
        isinstance(name, str)
        and name.isidentifier()
        and not keyword.iskeyword(name)
        and not name.endswith("_ptr")
        and name not in _SCAN_RESERVED
    )


def _affd_leaves(f_spec: Any) -> list | None:
    """In-order ``(a, b)`` operand pairs of a pure aff_diag map spec.

    ``None`` when *f_spec* is not an ``aff_diag`` leaf or an
    ``affd_compose`` tree over such leaves (attr-carrying or foreign
    nodes included — the map spec must be exactly the carrier's).
    """
    if not (
        isinstance(f_spec, (tuple, list))
        and len(f_spec) == 3
        and all(not isinstance(e, dict) for e in f_spec)
    ):
        return None
    if f_spec[0] == "aff_diag":
        return [(f_spec[1], f_spec[2])]
    if f_spec[0] != "affd_compose":
        return None
    left = _affd_leaves(f_spec[1])
    right = _affd_leaves(f_spec[2])
    if left is None or right is None:
        return None
    return [*left, *right]


def _leaf_exprs(
    leaves: list, h: str, hoists: dict | None
) -> tuple[list, list] | None:
    """Render leaf operands to Triton exprs, in chronological order.

    In-order leaves of an ``affd_compose`` tree are REVERSE
    chronological — ``affd_compose(f, g)`` applies ``g`` first — so
    the list is flipped here.  Returns ``(metas, steps)`` where
    *metas* are the operand metavars in spec order and *steps* the
    ``(a_expr, b_expr)`` rendered pairs.  ``None`` declines: an
    operand the elementwise renderer cannot express, a metavar that
    collides with kernel names, or the state meta rebound as a
    sequence operand.  With *hoists*, unrenderable non-leaf
    subterms become host-computed ``__c<i>`` operands rather than
    declines — the unrolled path only (a seq operand's step-axis
    indexing has no meaning for a state-shaped subterm).
    """
    metas: list = []
    for a_spec, b_spec in leaves:
        for m in _spec_metas(("aff_diag", a_spec, b_spec)):
            if m not in metas:
                metas.append(m)
    if (
        not metas
        or h in metas
        or any(not _meta_name_ok(m) for m in metas)
    ):
        return None
    steps = []
    for a_spec, b_spec in reversed(leaves):
        ea = _triton_expr_h(a_spec, hoists, _TRITON_CONST_1D)
        eb = _triton_expr_h(b_spec, hoists, _TRITON_CONST_1D)
        if ea is None or eb is None:
            return None
        steps.append((ea, eb))
    return metas, steps


def _applyd_operands(spec: Any) -> tuple | None:
    """Split an ``applyd`` spec into ``(map spec, state meta, attrs)``.

    ``None`` when the node is not a two-operand ``applyd`` over a
    metavar state — malformed arities, a non-metavar ``h``, or a
    kernel-name-colliding state all decline here.
    """
    if not (
        isinstance(spec, (tuple, list))
        and len(spec) >= 3
        and spec[0] == "applyd"
    ):
        return None
    kids = [e for e in spec[1:] if not isinstance(e, dict)]
    if len(kids) != 2 or not _meta_name_ok(kids[1]):
        return None
    attrs = next((e for e in spec[1:] if isinstance(e, dict)), None)
    return kids[0], kids[1], attrs


def scan_spec(spec: Any) -> dict | None:
    """Recognise an ``applyd``-carrier spec the scan codegen accepts.

    Returns a descriptor dict, or ``None`` when the spec is outside
    the tier's grammar (decline — the handler keeps its torch path):

    * ``{"kind": "seq", ...}`` for ``("applyd", ("aff_diag", a, b),
      h)`` — the sequence form.  ``"seq"`` lists the operand metavars
      (each a per-step tensor on the step axis), ``"a"``/``"b"`` the
      rendered per-step expressions, ``"h"`` the state metavar.
    * ``{"kind": "seqs", ...}`` — the EMITTED sequence form: the
      same one-leaf ``aff_diag`` fold, but the minted op denotes
      every prefix state (``out_t = acc_t``), the semantics
      ``cumsum``/``cumprod`` return.  ``info["scan_mode"]``
      classifies the leaf — ``"cumsum"``/``"cumprod"`` reach the
      ``tl`` scan primitives, anything else the serial emitted
      fold.
    * ``{"kind": "unrolled", ...}`` for ``("applyd",
      <affd_compose tree>, h)`` — ``"steps"`` the rendered ``(a, b)``
      pairs in chronological order.

    Both carry ``"params"`` — the spec's metavars in order, i.e. the
    generated binding's positional signature.  A trailing attr dict
    on the ``applyd`` node may declare ``{"axis": int}`` — surfaced
    as ``info["axis"]``; a ``None``/absent axis defers to the
    binding's default or its derive-from-``h`` rule.  It may also
    declare ``{"kind": "unrolled"}`` to pin the ambient single-step
    reading of a one-leaf ``aff_diag`` (the e-graph's folded
    carriers are that shape — ``a ⊙ h + x`` once, not a fold).
    ``info["computed"]`` lists the ``(name, subterm)`` pairs the
    renderer hoisted *and the step expressions reference* — dead
    hoists (kids of a node that itself hoisted) and metavars
    occurring only inside hoists never reach the kernel; the
    binding evaluates the live subterms host-side and passes them
    as extra kernel args.  ``info["kernel_args"]`` is the operand
    order — referenced metavars, the state, then live hoists.
    """
    parts = _applyd_operands(spec)
    if parts is None:
        return None
    f_spec, h_spec, attrs = parts
    if attrs is not None and attrs.get("kind") == "seqm":
        return _seqm_spec(spec)
    leaves = _affd_leaves(f_spec)
    kind = _scan_kind(leaves, attrs)
    if kind is None or leaves is None:
        return None
    hoists: dict | None = {} if kind == "unrolled" else None
    rendered = _leaf_exprs(leaves, h_spec, hoists)
    if rendered is None:
        return None
    return _scan_info(
        kind, rendered, h_spec, attrs, hoists, leaves=leaves
    )


def _scan_info(
    kind: str,
    rendered: tuple[list, list],
    h_spec: Any,
    attrs: dict | None,
    hoists: dict | None,
    leaves: Any = None,
) -> dict:
    """Assemble the ``scan_spec`` descriptor for a rendered carrier."""
    metas, steps = rendered
    # ``kind="seq"``/``"seqs"`` specs may declare ``"const"`` —
    # loop-invariant metavar names (``Param``/``Var`` leaves the
    # collector bound whole) that participate in the step exprs
    # WITHOUT a step axis: they are binding args like any meta but
    # broadcast into the state space rather than slicing along the
    # step axis.
    consts = (
        frozenset((attrs or {}).get("const", ()))
        if kind in ("seq", "seqs", "seqt")
        else frozenset()
    )
    seq = [m for m in metas if m not in consts]
    info: dict[str, Any] = {
        "h": h_spec,
        "seq": seq,
        "params": [*metas, h_spec],
    }
    if consts:
        info["const"] = [m for m in metas if m in consts]
    if kind in ("seq", "seqs", "seqt"):
        (ea, eb) = steps[0]
        info.update({"kind": kind, "a": ea, "b": eb})
        if kind in ("seqs", "seqt") and leaves:
            # the emitted scan's closed-form mode — ``cumsum`` /
            # ``cumprod`` spell the whole prefix tensor, so the
            # denotation and the Triton piece can use the real scan
            # primitive instead of the serial fold.  Detected from
            # the RAW leaf specs (the rendered exprs have already
            # lost the constant/meta distinction's meaning).
            info["scan_mode"] = _seqs_mode(leaves[0])
    else:
        _unrolled_info(info, steps, hoists)
    if attrs is not None and "axis" in attrs:
        info["axis"] = attrs["axis"]
    return info


def _seqs_mode(leaf: Any) -> str:
    """Classify a ``seqs`` leaf ``(a_spec, b_spec)`` for codegen.

    ``"cumsum"`` — ``a ≡ 1``, ``b`` the bare sequence meta (the
    prefix-sum recurrence ``acc + x_t``); ``"cumprod"`` — ``a``
    the bare meta, ``b ≡ 0`` (``x_t * acc``); anything else is the
    generic emitted fold (``"fold"``) — still an honest inclusive
    scan, just without a closed-form primitive.
    """
    a_spec, b_spec = leaf
    if a_spec == 1 and isinstance(b_spec, str):
        return "cumsum"
    if b_spec == 0 and isinstance(a_spec, str):
        return "cumprod"
    return "fold"


def _scan_kind(leaves: list | None, attrs: dict | None) -> str | None:
    """Resolve the spec's carrier kind — attr pin, then arity.

    ``kind="seq"`` is only meaningful for a ONE-leaf ``aff_diag`` —
    the sequence form folds a single per-step map over the step
    axis.  Pinning ``seq`` on a multi-leaf ``affd_compose`` tree
    would silently keep only ``steps[0]`` in ``_scan_info`` (the
    other leaves never reach the kernel), so it declines here
    rather than denote a different recurrence.
    """
    if leaves is None:
        return None
    kind = (attrs or {}).get("kind")
    if kind is None:
        return "seq" if len(leaves) == 1 else "unrolled"
    if kind in ("seq", "seqs", "seqt") and len(leaves) != 1:
        return None
    return kind if kind in ("seq", "seqs", "seqt", "unrolled") else None


def _name_used(name: str, steps: list) -> bool:
    """Check *name* survives into a rendered step expression.

    Only names appearing as bare tokens in the final Triton
    expressions reach the kernel — a metavar living wholly inside a
    hoisted subterm is evaluated on the host as part of that
    subterm, and a hoist registered for a kid of a node that itself
    hoisted is dead.  Neither may join the kernel's flat index
    space: their shapes belong to the host subterm, not the
    broadcast the flat loads require.  The ``(?<![.")`` guard keeps
    ``tl.exp``/``libdevice.tanh`` member names from counting.
    """
    import re

    pat = re.compile(rf"(?<![.\w]){re.escape(name)}(?![\w])")
    return any(pat.search(e) for ea, eb in steps for e in (ea, eb))


def _unrolled_info(
    info: dict, steps: list, hoists: dict | None
) -> None:
    """Fill *info*'s unrolled-kind fields from the rendered steps."""
    info.update({"kind": "unrolled", "steps": steps})
    # The kernel's flat operands are exactly the names its step
    # expressions reference, plus the state: ``flat`` metavars in
    # spec order, then ``h``, then the live hoists in registration
    # order — the binding broadcasts and loads in this order.
    used = {
        n
        for n in (*(hoists or {}), *info["seq"])
        if _name_used(n, steps)
    }
    computed = [(n, s) for n, s in (hoists or {}).items() if n in used]
    info["flat"] = [m for m in info["seq"] if m in used]
    info["kernel_args"] = [
        *info["flat"],
        info["h"],
        *(n for n, _ in computed),
    ]
    if computed:
        info["computed"] = computed


def seq_scan_denotation(spec: Any, *, axis: int | None = None) -> Any:
    """Return the spelled fold denotation of a ``seq``-form scan spec.

    The minted ``scan_*_k`` op's declared semantics is the fold
    ``h_{t+1} = a_t ⊙ h_t + b_t`` walked along the step axis — NOT
    the ambient ``applyd`` map application (``a ⊙ h + b``, once)
    that ``_eval_spec``/``_IR_TO_TORCH`` produces for the same spec
    tuple.  The spec tuple denotes the map; the fold is a property
    of the minted op, so the denotation cannot live in the spec —
    it lives here as a plain-torch callable, spelled once and
    shared by every consumer that must grade or deliver the claim:

    * ``profitable``'s ``denotation_for`` hook times the generated
      kernel against THIS fold, never the map;
    * ``gen_bindings`` binds the kernel name to it, so ``gen_sink``
      (and ``hybrid_sink``'s declined-codegen fallback) delivers
      the declared semantics on any device — slow but correct —
      instead of silently delivering the map.

    The callable takes the spec metavars positionally (the
    handler's ``args`` order, ``info["params"]``) and returns the
    final state.  It broadcasts the sequence operands exactly like
    :func:`_seq_args`, resolves the step axis through
    :func:`_seq_axis` (explicit *axis* — ``None`` derives it from
    the state shape per call), then walks the axis evaluating the
    ``aff_diag`` operand sub-specs per step through the ordinary
    bindings and folding ``acc = a_t * acc + b_t`` — the same
    recurrence, in the same order, the Triton kernel runs.

    Returns ``None`` unless *spec* is a recognised ``kind="seq"``
    scan spec.  Call-time shape violations (no step axis, an ``h``
    that cannot broadcast to the state shape) raise — the same
    contract the Triton binding keeps.
    """
    info = scan_spec(spec)
    if info is None or info["kind"] != "seq":
        return None
    parts = _applyd_operands(spec)
    if parts is None:
        return None
    leaves = _affd_leaves(parts[0])
    if leaves is None or len(leaves) != 1:
        return None
    a_spec, b_spec = leaves[0]
    seq_metas = list(info["seq"])
    h_meta = info["h"]
    params = list(info["params"])
    ax = axis if axis is not None else info.get("axis")
    bindings = _ambient_bindings()

    def fn(*args: Any, **_attrs: Any) -> Any:
        import torch

        vals = dict(zip(params, args, strict=True))
        xs = torch.broadcast_tensors(*(vals[m] for m in seq_metas))
        full = xs[0].shape
        if not full:
            raise ValueError("scan operands carry no step axis")
        step_ax = _seq_axis(full, vals[h_meta].shape, ax)
        t_len = full[step_ax]
        state = full[:step_ax] + full[step_ax + 1 :]
        moved = [x.movedim(step_ax, 0) for x in xs]
        acc = torch.broadcast_to(vals[h_meta], state)
        consts = info.get("const", ())
        for t in range(t_len):
            env = {
                m: mv[t] for m, mv in zip(seq_metas, moved, strict=True)
            }
            # loop-invariant operands bind whole — they carry no
            # step axis, so the same value serves every step
            for c in consts:
                env[c] = vals[c]
            a_t = _eval_spec(a_spec, env, bindings)
            b_t = _eval_spec(b_spec, env, bindings)
            acc = a_t * acc + b_t
        return acc

    return fn


def seqs_denotation(spec: Any, *, axis: int | None = None) -> Any:
    """Spelled denotation of a ``kind="seqs"`` emitted-scan spec.

    The emitted-scan analogue of :func:`seq_scan_denotation`: the
    same ``acc = a_t * acc + b_t`` recurrence, but the minted op
    denotes EVERY prefix state — ``out_t = acc_t`` stacked back
    along the step axis — because the spelled carrier (``cumsum``,
    ``cumprod``, an emitted selective scan) returns the whole
    prefix tensor, not the final state the ``seq`` fold keeps.

    Two spellings, picked by the spec's leaf shape
    (:func:`_seqs_mode`):

    * ``"cumsum"``/``"cumprod"`` — the recurrence IS the ambient
      ``torch.cumsum``/``torch.cumprod`` op, so the denotation
      calls it directly.  This is not a shortcut past honesty: the
      referee's spelled arm must be the *incumbent* the delivery
      replaces — timing the generated kernel against a python-loop
      fold would grade it against a body nobody ships, and the
      ``torch.*`` call IS the same function (up to summation
      order, inside the verify tolerance).
    * ``"fold"`` — any other one-leaf ``aff_diag``: the generic
      emitted fold, ``outs.append(acc)`` per step then
      ``torch.stack`` moved back to the step axis — the same
      order the Triton piece runs.

    Returns ``None`` unless *spec* is a recognised ``kind="seqs"``
    scan spec.  Call-time shape violations raise, matching the
    ``seq`` contract.
    """
    import torch

    info = scan_spec(spec)
    if info is None or info["kind"] != "seqs":
        return None
    parts = _applyd_operands(spec)
    if parts is None:
        return None
    leaves = _affd_leaves(parts[0])
    if leaves is None or len(leaves) != 1:
        return None
    a_spec, b_spec = leaves[0]
    seq_metas = list(info["seq"])
    h_meta = info["h"]
    params = list(info["params"])
    ax = axis if axis is not None else info.get("axis")
    bindings = _ambient_bindings()
    mode = info.get("scan_mode", "fold")

    if (
        mode in ("cumsum", "cumprod")
        and len(seq_metas) == 1
        and not info.get("const")
    ):
        m = seq_metas[0]
        op = torch.cumsum if mode == "cumsum" else torch.cumprod

        def fn(*args: Any, **_attrs: Any) -> Any:
            vals = dict(zip(params, args, strict=True))
            x = vals[m]
            step_ax = _seq_axis(x.shape, vals[h_meta].shape, ax)
            return op(x, dim=step_ax)

        return fn

    def fn(*args: Any, **_attrs: Any) -> Any:
        vals = dict(zip(params, args, strict=True))
        xs = torch.broadcast_tensors(*(vals[m] for m in seq_metas))
        full = xs[0].shape
        if not full:
            raise ValueError("scan operands carry no step axis")
        step_ax = _seq_axis(full, vals[h_meta].shape, ax)
        t_len = full[step_ax]
        state = full[:step_ax] + full[step_ax + 1 :]
        moved = [x.movedim(step_ax, 0) for x in xs]
        acc = torch.broadcast_to(vals[h_meta], state)
        consts = info.get("const", ())
        outs = []
        for t in range(t_len):
            env = {
                m: mv[t] for m, mv in zip(seq_metas, moved, strict=True)
            }
            for c in consts:
                env[c] = vals[c]
            a_t = _eval_spec(a_spec, env, bindings)
            b_t = _eval_spec(b_spec, env, bindings)
            acc = a_t * acc + b_t
            outs.append(acc)
        # ``(T, *state)`` moved so the emitted axis lands where the
        # operand's step axis sat — the same ``full`` the inputs
        # broadcast to, which is exactly the carrier's output shape.
        return torch.stack(outs, dim=0).movedim(0, step_ax)

    return fn


# ---------------------------------------------------------------------------
#  The seqm tier — matrix-state scans with state feedback in ``b``
# ---------------------------------------------------------------------------
#
# The diagonal ``seq`` fold is ``acc = a_t * acc + b_t`` with
# state-free operands.  The delta-rule family spells a *matrix*
# state whose update reads the accumulator back:
#
#   s' = a_t·s + C_t ⊗ (V_t - S_t·(K_tᵀ s))        (gated)
#   s' = s     + C_t ⊗ (V_t -      K_tᵀ s)        (plain)
#
# spelled elementwise as ``add(mul(a_t, s), write)`` — or
# ``add(s, write)`` for the unit map — with the previous state
# ``s`` occurring inside ``write`` through ONE ``matmul(K, s)``.
# ``play._seqm_site`` mints these as ``("applyd", ("aff_diag", a,
# b), "H", {"kind": "seqm"})`` specs where ``b`` contains the ``"H"``
# metavar at exactly that matmul; this section recognises the
# rank-1-feedback skeleton, spells the torch fold denotation, and
# emits a tiled Triton kernel (``acc`` a ``(dk, dv)`` tile per
# program, the ``Kᵀ·acc`` read a masked ``tl.sum`` matvec).


def _seqm_sub_ok(sub: Any, h_meta: str) -> bool:
    """Check a step sub-spec is state-free and elementwise-renderable.

    Every metavar it carries is a sequence operand; ``h_meta``
    (the accumulator) may only occur inside the recognised
    ``matmul(K, H)`` read — anywhere else the fold is not the
    rank-1-feedback form this tier denotes.  The render check keeps
    the kernel's expr language and the spelled eval in lockstep:
    a sub-spec the Triton elementwise renderer cannot express
    declines here rather than failing inside the binding.
    """
    if h_meta in _spec_metas(sub):
        return False
    return _triton_expr_h(sub, None, _TRITON_CONST_1D) is not None


def _seqm_spec_node(spec: Any, op: str, n: int) -> bool:
    """Check *spec* is a bare ``op`` node of *n* spec elements."""
    return (
        isinstance(spec, (tuple, list))
        and len(spec) == n
        and spec[0] == op
        and not any(isinstance(e, dict) for e in spec[1:])
    )


def _seqm_spec_attrs(spec: Any, dims: tuple) -> bool:
    """Check *spec*'s trailing attr dict pins ``dim`` into *dims*."""
    if not (
        isinstance(spec, (tuple, list))
        and len(spec) == 3
        and isinstance(spec[2], dict)
        and set(spec[2]) == {"dim"}
    ):
        return False
    return spec[2]["dim"] in dims


def _seqm_parse_p(p: Any, h_meta: str):
    """Parse the projected-read factor of a seqm ``b`` spec.

    Returns ``(s_spec, k_meta)`` — the optional scalar gate ``S``
    and the key operand ``K`` of ``matmul(K, H)`` — or ``None``.
    Accepted shapes::

        squeeze(matmul(K, H), dim in {1, -2})
        mul(S, <the above>)   (either operand order)

    ``S`` may itself be any elementwise spec; deeper nesting is
    refused by the single-``mul`` unwrap below.
    """
    if _seqm_spec_attrs(p, (1, -2)) and _seqm_spec_node(
        p[1], "matmul", 3
    ):
        m = p[1]
        if m[2] == h_meta and isinstance(m[1], str) and m[1] != h_meta:
            return None, m[1]
        return None
    if _seqm_spec_node(p, "mul", 3):
        for s0, p0 in ((p[1], p[2]), (p[2], p[1])):
            r = _seqm_parse_p(p0, h_meta)
            if r is not None:
                return s0, r[1]
    return None


def _seqm_parse_b(b: Any, h_meta: str):
    """Parse a seqm ``b`` spec into ``(c, v, s, k)`` or ``None``.

    The recognised skeleton is the rank-1-feedback write::

        mul(C, unsqueeze(sub(V, P), dim in {-2, 1}))

    with ``C`` the column factor, ``V`` the row residual target and
    ``P`` the projected read from :func:`_seqm_parse_p`.  Either
    ``mul`` operand order is accepted; anything else declines.
    """
    if not _seqm_spec_node(b, "mul", 3):
        return None
    for c, u in ((b[1], b[2]), (b[2], b[1])):
        if not _seqm_spec_attrs(u, (-2, 1)):
            continue
        r = u[1]
        if not _seqm_spec_node(r, "sub", 3):
            continue
        parsed = _seqm_parse_p(r[2], h_meta)
        if parsed is None:
            continue
        return c, r[1], parsed[0], parsed[1]
    return None


def _seqm_spec(spec: Any) -> dict | None:
    """Recognise a ``kind="seqm"`` scan spec; descriptor or ``None``.

    The descriptor carries what the flat ``seq`` info cannot: the
    parsed rank-1-feedback operands ``"c"``/``"v"``/``"s"``/``"k"``
    (the column factor, the row residual, the optional scalar gate
    and the matvec key), the full ``"a"``/``"b"`` step specs for the
    spelled denotation, and ``"params"`` — the spec metavars in
    ``_spec_metas`` order, which interleaves ``"H"`` exactly where
    the ``b`` spec first mentions it (the positional binding
    signature is the same order by construction).
    """
    parts = _applyd_operands(spec)
    if parts is None:
        return None
    f_spec, h_spec, attrs = parts
    if not (
        isinstance(f_spec, (tuple, list))
        and len(f_spec) == 3
        and f_spec[0] == "aff_diag"
        and not any(isinstance(e, dict) for e in f_spec[1:])
    ):
        return None
    parsed = _seqm_parse_b(f_spec[2], h_spec)
    if parsed is None:
        return None
    c_spec, v_spec, s_spec, k_meta = parsed
    a_spec = f_spec[1]
    for sub in (a_spec, c_spec, v_spec, s_spec):
        if sub is not None and not _seqm_sub_ok(sub, h_spec):
            return None
    params = list(_spec_metas(spec))
    seq = [m for m in params if m != h_spec]
    if k_meta not in seq:
        return None
    return {
        "kind": "seqm",
        "params": params,
        "seq": seq,
        "h": h_spec,
        "a": a_spec,
        "b": f_spec[2],
        "c": c_spec,
        "v": v_spec,
        "s": s_spec,
        "k": k_meta,
        "axis": (attrs or {}).get("axis"),
    }


def seqm_denotation(
    spec: Any, *, axis: int | None = None, axes: Any = None
) -> Any:
    """Spelled fold denotation of a ``kind="seqm"`` scan spec.

    ``acc = a_t * acc + b_t(acc)`` walked along each operand's OWN
    step axis — there is no shared broadcast frame, so *axes* maps
    every sequence metavar to its normalised step dim (the
    ``"axes"`` map the site collector records on the handler).
    Per step the operand specs evaluate through the ordinary
    bindings against ``{metas: per-step slices, "H": acc}`` —
    ``matmul(K, acc)`` included — exactly the recurrence the
    unrolled chain spells, in the same order the Triton kernel
    runs.  ``axis`` is accepted for signature parity with
    :func:`seq_scan_denotation`; the per-operand map supersedes it.
    """
    info = _seqm_spec(spec)
    if info is None:
        return None
    params = list(info["params"])
    seq = list(info["seq"])
    h_meta = info["h"]
    a_spec = info["a"]
    b_spec = info["b"]
    bindings = _ambient_bindings()

    def fn(*args: Any, **_attrs: Any) -> Any:
        vals = dict(zip(params, args, strict=True))
        if axes is None:
            raise ValueError(
                "seqm fold needs the handler's per-operand axes"
            )
        moved: dict = {}
        t_len = None
        for m in seq:
            x = vals[m]
            axm = axes[m] % x.dim()
            xs = x.movedim(axm, 0)
            if t_len is None:
                t_len = xs.shape[0]
            elif xs.shape[0] != t_len:
                raise ValueError(
                    "seqm operands disagree on the step count"
                )
            moved[m] = xs
        # ``seq`` is non-empty by construction (the key operand is
        # always a sequence metavar) — the assert narrows for ty.
        assert t_len is not None
        acc = vals[h_meta]
        for t in range(t_len):
            env = {m: moved[m][t] for m in seq}
            env[h_meta] = acc
            a_t = (
                1 if a_spec == 1 else _eval_spec(a_spec, env, bindings)
            )
            acc = a_t * acc + _eval_spec(b_spec, env, bindings)
        return acc

    return fn


def handler_denotation(h: dict) -> Any:
    """Positional spelled denotation for handlers that need one.

    A ``kind="seq"`` scan spec returns the fold callable
    :func:`seq_scan_denotation` builds at the handler's resolved
    axis (``h["axis"]`` → spec attr → the ``-2`` default — the same
    resolution :func:`scan_triton_bindings` applies, so the
    referee's spelled arm and the delivered kernel agree on the
    step axis).  A ``kind="seqm"`` spec returns
    :func:`seqm_denotation` at the handler's ``"axes"`` map — the
    per-operand step axes only the site record carries.  Every
    other handler returns ``None``: its spelled spec eval already
    IS the denotation and needs no override.
    """
    info = scan_spec(h.get("pattern"))
    if info is None:
        return None
    if info["kind"] == "seqm":
        return seqm_denotation(
            h["pattern"], axis=h.get("axis"), axes=h.get("axes")
        )
    ax = h.get("axis", info.get("axis", -2))
    if info["kind"] == "seq":
        return seq_scan_denotation(h["pattern"], axis=ax)
    if info["kind"] == "seqs":
        return seqs_denotation(h["pattern"], axis=ax)
    if info["kind"] == "seqt":
        return seqt_denotation(
            h["pattern"], readout=h.get("readout"), axis=ax
        )
    return None


def _seq_triton_source(info: dict) -> str:
    """Emit the sequential-scan kernel module for a ``seq`` spec.

    One program per ``BLOCK`` columns of the flattened state space;
    each walks the step axis serially — ``acc = a_t * acc + b_t``.
    Row offsets are computed in int64 so ``T * N`` may exceed int32.
    ``"const"`` operands — loop-invariant metas with no step axis —
    ride as extra pointer args after the sequence operands and load
    once, flat over the state space, before the step loop.
    """
    ptrs = [f"{m}_ptr" for m in info["seq"]]
    ptrs += [f"{m}_ptr" for m in info.get("const", ())]
    params = ", ".join([*ptrs, "h_ptr"])
    loads = "".join(
        f"        {m} = tl.load({m}_ptr + row + off,"
        " mask=mask, other=0.0)\n"
        for m in info["seq"]
    )
    cloads = "".join(
        f"    {m} = tl.load({m}_ptr + off, mask=mask, other=0.0)\n"
        for m in info.get("const", ())
    )
    return (
        "import triton\n"
        "import triton.language as tl\n"
        "import triton.language.extra.libdevice as libdevice\n"
        "\n\n@triton.jit\n"
        f"def _gs({params}, out_ptr, T, N, BLOCK: tl.constexpr):\n"
        "    off = tl.program_id(0).to(tl.int64) * BLOCK"
        " + tl.arange(0, BLOCK)\n"
        "    mask = off < N\n"
        f"    acc = tl.load(h_ptr + off,"
        " mask=mask, other=0.0)\n"
        f"{cloads}"
        "    for t in range(T):\n"
        "        row = t.to(tl.int64) * N\n"
        f"{loads}"
        f"        acc = ({info['a']}) * acc + ({info['b']})\n"
        "    tl.store(out_ptr + off, acc, mask=mask)\n"
    )


def _unrolled_sig(info: dict) -> list:
    """Build the kernel signature: the referenced flat operands.

    ``kernel_args`` — the expr-referenced metavars, the state, and
    the live hoists — is the operand list the step expressions
    actually load; spec metavars buried inside hoists stay
    positional binding args (the host subterm eval needs them) but
    never reach the kernel.
    """
    return list(info["kernel_args"])


def _flat_loads(names: Any) -> str:
    """Emit the flat-space load lines for operand names."""
    return "".join(
        f"    {m} = tl.load({m}_ptr + off, mask=mask, other=0.0)\n"
        for m in names
    )


def _unrolled_triton_source(info: dict) -> str:
    """Emit the unrolled carrier kernel for a compose-tree spec.

    One flat pass over the broadcast state space applying each leaf
    step in chronological order — ``acc = a_i * acc + b_i`` — exactly
    the ambient ``applyd``/``affd_compose`` semantics fused into one
    kernel.  Hoisted operands ride as extra pointer args after the
    caller's ``params`` — the binding evaluates them on the host.
    """
    sig = _unrolled_sig(info)
    params = ", ".join(f"{m}_ptr" for m in sig)
    loads = _flat_loads(sig)
    steps = "".join(
        f"    acc = ({ea}) * acc + ({eb})\n" for ea, eb in info["steps"]
    )
    return (
        "import triton\n"
        "import triton.language as tl\n"
        "import triton.language.extra.libdevice as libdevice\n"
        "\n\n@triton.jit\n"
        f"def _gs({params}, out_ptr, n, BLOCK: tl.constexpr):\n"
        "    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
        "    mask = off < n\n"
        f"{loads}"
        f"    acc = tl.load({info['h']}_ptr + off,"
        " mask=mask, other=0.0)\n"
        f"{steps}"
        "    tl.store(out_ptr + off, acc, mask=mask)\n"
    )


def _scan_triton_source(info: dict) -> str:
    """Emit the generated scan kernel module for a recognised spec.

    *info* is a :func:`scan_spec` descriptor; ``"seq"`` emits the
    sequential step-axis fold, ``"unrolled"`` the flat chronological
    pass.  The source is a real module — written to the kernel cache
    by the caller.
    """
    if info["kind"] == "seq":
        return _seq_triton_source(info)
    if info["kind"] == "seqs":
        return _seqs_triton_source(info)
    return _unrolled_triton_source(info)


def _broadcasts_into(src: Any, dst: tuple) -> bool:
    """Check a tensor shaped *src* broadcasts to *dst*."""
    import torch

    try:
        got = torch.broadcast_shapes(tuple(src), tuple(dst))
    except RuntimeError:
        return False
    return tuple(got) == tuple(dst)


def _seq_axis(full: tuple, h_shape: Any, axis: int | None) -> int:
    """Resolve the step axis of the common sequence shape.

    An explicit *axis* is normalised against ``len(full)``.  ``None``
    derives it: the step axis is the one whose removal leaves the
    state shape *h* broadcasts into — requiring exactly one
    candidate.  Zero candidates means *h* matches no contraction of
    the operands' shape; several means the shapes alone cannot tell
    (e.g. a square ambiguous state/step extent) — both are honest
    ``ValueError``s, not guesses.
    """
    if axis is not None:
        return axis % len(full)
    cand = [
        i
        for i in range(len(full))
        if _broadcasts_into(h_shape, full[:i] + full[i + 1 :])
    ]
    if len(cand) != 1:
        raise ValueError(
            f"scan axis is not determined by the shapes:"
            f" candidates={tuple(cand)} for full={tuple(full)}"
            f" h={tuple(h_shape)}"
        )
    return cand[0]


def _scan_dtype(vals: dict, metas: list) -> Any:
    """Promoted dtype over the bound meta values."""
    import torch

    dt = vals[metas[0]].dtype
    for m in metas[1:]:
        dt = torch.promote_types(dt, vals[m].dtype)
    return dt


def _seq_args(vals: dict, info: dict, axis: int | None) -> tuple:
    """Normalise bound sequence operands to ``(T, N)`` row tensors.

    Returns ``(rows, h_flat, state_shape, T, n, dtype)`` — the launch
    arguments in kernel order.  Broadcasting is right-aligned: each
    operand's trailing dims land on state axes, so an operand without
    a step axis (a shared ``(d,)`` decay under ``(T, d)`` inputs, or
    a scalar) broadcasts over steps automatically.  Every sequence
    operand is materialised contiguous — the price of generality
    over strides, stated plainly.
    """
    import torch

    xs = torch.broadcast_tensors(*(vals[m] for m in info["seq"]))
    full = xs[0].shape
    if not full:
        raise ValueError("scan operands carry no step axis")
    if not xs[0].is_cuda:
        raise RuntimeError("generated scan kernels need CUDA tensors")
    ax = _seq_axis(full, vals[info["h"]].shape, axis)
    t_len = full[ax]
    state = full[:ax] + full[ax + 1 :]
    dt = _scan_dtype(vals, info["params"])
    rows = {
        m: x.to(dt).movedim(ax, 0).contiguous().view(t_len, -1)
        for m, x in zip(info["seq"], xs, strict=True)
    }
    h_flat = (
        torch.broadcast_to(vals[info["h"]].to(dt), state)
        .contiguous()
        .view(-1)
    )
    # ``"const"`` operands carry no step axis: each broadcasts into
    # the state space once and flattens alongside ``h`` — the kernel
    # loads them before the step loop, never per row.
    cflat = {
        m: torch.broadcast_to(vals[m].to(dt), state)
        .contiguous()
        .view(-1)
        for m in info.get("const", ())
    }
    return rows, cflat, h_flat, state, t_len, h_flat.numel(), dt


def _seq_binding(kern: Any, info: dict, axis: int | None) -> Any:
    """Host wrapper for the sequential-scan kernel.

    ``fn(*vals)`` binds the spec metavars positionally (the handler's
    ``args`` order), normalises the sequence operands, and launches
    the fold.  ``axis`` is the declared step axis — ``None`` derives
    it from the state shape each call.
    """

    def fn(*args: Any, **_attrs: Any) -> Any:
        import torch

        vals = dict(zip(info["params"], args, strict=True))
        rows, cflat, h_flat, state, t_len, n, dt = _seq_args(
            vals, info, axis
        )
        out = torch.empty(n, dtype=dt, device=h_flat.device)
        kern[((n + _SCAN_BLOCK - 1) // _SCAN_BLOCK,)](
            *(rows[m] for m in info["seq"]),
            *(cflat[m] for m in info.get("const", ())),
            h_flat,
            out,
            t_len,
            n,
            BLOCK=_SCAN_BLOCK,
        )
        return out.view(state)

    return fn


# ---------------------------------------------------------------------------
#  seqs codegen — the emitted scan (every prefix state, not the last)
# ---------------------------------------------------------------------------
#
# ``kind="seqs"`` is the ``seq`` fold with the OUTPUT axis restored:
# ``out_t = acc_t`` for every ``t``, stacked back along the step
# axis — the semantics ``cumsum``/``cumprod`` (and any emitted
# selective scan) actually return.  Two kernel shapes:
#
# * ``scan_mode`` ``"cumsum"``/``"cumprod"`` — the closed-form
#   primitives: ``tl.cumsum``/``tl.cumprod`` along the step axis —
#   a real parallel scan (Blelloch-style inside the tile), NOT the
#   serial step loop.  Two launch layouts, picked per call:
#   ``_gs`` reads the operand STRIDED — viewed as ``(P, T, K)`` with
#   ``K`` the contiguous trailing block and ``P`` a rank-≤1 prefix —
#   so no transpose copy is paid; ``_gsc`` takes the dense ``(T, N)``
#   rows for shapes the strided view cannot express.  Honest bound:
#   ``T <= _SEQS_MAX_T`` (the tile's ``BT`` extent) — a longer
#   sequence raises, and the referee/denotation fallback decides.
# * ``"fold"`` — the generic emitted recurrence: the ``seq`` kernel
#   with a store inside the step loop (inclusive scan), any ``T``.

#: Column-block width and step-extent cap for the parallel ``seqs``
#: kernel — ``BT`` is ``next_pow2(T)`` per call, so the tile is at
#: most ``_SEQS_MAX_T x _SEQS_BN`` elements.
_SEQS_BN = 256
_SEQS_MAX_T = 256


def _seqs_triton_source(info: dict) -> str:
    """Emit the generated emitted-scan kernel module for a ``seqs`` spec.

    Dispatches on ``info["scan_mode"]``: the closed-form modes emit
    the ``tl.cumsum``/``tl.cumprod`` tile kernels (``_gs`` strided,
    ``_gsc`` dense), everything else the serial emitted fold.
    """
    mode = info.get("scan_mode", "fold")
    if mode in ("cumsum", "cumprod"):
        return _seqs_scan_source(info, mode)
    return _seqs_fold_source(info)


def _seqs_scan_source(info: dict, mode: str) -> str:
    """Emit the parallel-scan kernels for the closed-form modes.

    ``_gs`` is the STRIDED kernel: the operand is viewed as
    ``(P, T, K)`` — ``P`` a rank-≤1 prefix (grid axis 0), ``T`` the
    step axis, ``K`` the contiguous trailing block (grid axis 1 over
    ``BN``-wide column blocks) — so the scan reads the tensor in
    place, no transpose copy.  ``s_t``/``s_p`` are the step/prefix
    element strides; the output is a fresh contiguous ``(P, T, K)``
    the host reshapes to the operand's shape.  ``_gsc`` is the dense
    fallback: ``(T, N)`` rows for shapes the strided view cannot
    express.  Padding rows past ``T`` carry the scan's identity
    (``0.0`` cumsum / ``1.0`` cumprod — trailing identity rows never
    reach a stored prefix).  Single-sequence-operand specs only —
    enforced by :func:`_seqs_binding`.
    """
    prim = "tl.cumsum" if mode == "cumsum" else "tl.cumprod"
    pad = "0.0" if mode == "cumsum" else "1.0"
    return (
        "import triton\n"
        "import triton.language as tl\n"
        "import triton.language.extra.libdevice as libdevice\n"
        "\n\n@triton.jit\n"
        "def _gs(x_ptr, out_ptr, T, K, P, s_t, s_p,"
        " BT: tl.constexpr, BN: tl.constexpr):\n"
        "    p = tl.program_id(0)\n"
        "    kb = tl.program_id(1)\n"
        "    offs_t = tl.arange(0, BT)\n"
        "    offs_k = kb * BN + tl.arange(0, BN)\n"
        "    t_mask = offs_t < T\n"
        "    k_mask = offs_k < K\n"
        "    tile = tl.load(\n"
        "        x_ptr + p.to(tl.int64) * s_p"
        " + offs_t[:, None].to(tl.int64) * s_t\n"
        "        + offs_k[None, :].to(tl.int64),\n"
        f"        mask=t_mask[:, None] & k_mask[None, :], other={pad})\n"
        f"    tile = {prim}(tile, 0)\n"
        "    tl.store(\n"
        "        out_ptr + p.to(tl.int64) * (T.to(tl.int64) * K)\n"
        "        + offs_t[:, None].to(tl.int64) * K\n"
        "        + offs_k[None, :].to(tl.int64),\n"
        "        tile,\n"
        "        mask=t_mask[:, None] & k_mask[None, :])\n"
        "\n\n@triton.jit\n"
        "def _gsc(x_ptr, out_ptr, T, N, BT: tl.constexpr,"
        " BN: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    offs_t = tl.arange(0, BT)\n"
        "    offs_n = pid * BN + tl.arange(0, BN)\n"
        "    t_mask = offs_t < T\n"
        "    n_mask = offs_n < N\n"
        "    tile = tl.load(\n"
        "        x_ptr + offs_t[:, None] * N + offs_n[None, :],\n"
        f"        mask=t_mask[:, None] & n_mask[None, :], other={pad})\n"
        f"    tile = {prim}(tile, 0)\n"
        "    tl.store(\n"
        "        out_ptr + offs_t[:, None] * N + offs_n[None, :],\n"
        "        tile,\n"
        "        mask=t_mask[:, None] & n_mask[None, :])\n"
    )


def _seqs_fold_source(info: dict) -> str:
    """Emit the serial emitted-fold kernel for a generic ``seqs`` spec.

    The ``seq`` kernel's step loop with the accumulator stored after
    EVERY update — ``out[t] = acc_t`` — so the ``(T, N)`` output is
    the inclusive prefix tensor the carrier denotes.  ``"const"``
    operands ride exactly as in :func:`_seq_triton_source`.
    """
    ptrs = [f"{m}_ptr" for m in info["seq"]]
    ptrs += [f"{m}_ptr" for m in info.get("const", ())]
    params = ", ".join([*ptrs, "h_ptr"])
    loads = "".join(
        f"        {m} = tl.load({m}_ptr + row + off,"
        " mask=mask, other=0.0)\n"
        for m in info["seq"]
    )
    cloads = "".join(
        f"    {m} = tl.load({m}_ptr + off, mask=mask, other=0.0)\n"
        for m in info.get("const", ())
    )
    return (
        "import triton\n"
        "import triton.language as tl\n"
        "import triton.language.extra.libdevice as libdevice\n"
        "\n\n@triton.jit\n"
        f"def _gs({params}, out_ptr, T, N, BLOCK: tl.constexpr):\n"
        "    off = tl.program_id(0).to(tl.int64) * BLOCK"
        " + tl.arange(0, BLOCK)\n"
        "    mask = off < N\n"
        f"    acc = tl.load(h_ptr + off,"
        " mask=mask, other=0.0)\n"
        f"{cloads}"
        "    for t in range(T):\n"
        "        row = t.to(tl.int64) * N\n"
        f"{loads}"
        f"        acc = ({info['a']}) * acc + ({info['b']})\n"
        "        tl.store(out_ptr + row + off, acc, mask=mask)\n"
    )


def _seqs_strided_view(x: Any, ax: int):
    """``(P, T, K, s_t, s_p)`` view of *x* for the strided kernel.

    The operand must read as ``(P, T, K)`` in place: ``T`` the step
    axis, ``K`` the contiguous trailing block (every dim after the
    step axis C-contiguous, step stride == ``K``), and ``P`` a
    rank-≤1 prefix — dims *before* the step axis collapse to one
    program axis only when there is at most one of them (a
    multi-dim prefix has no single stride and declines).  Returns
    ``None`` when the view does not exist — the caller falls back
    to the dense ``(T, N)`` kernel, which pays a transpose copy but
    denotes the same scan.
    """
    shape = tuple(x.shape)
    r = len(shape)
    s = ax % r
    if s > 1:
        return None
    k_ext = 1
    for d in shape[s + 1 :]:
        k_ext *= d
    # trailing dims must be C-contiguous and the step stride the
    # block extent — the ``(P, T, K)`` flat read is then exact.
    for i in range(s + 1, r):
        want = 1
        for d in shape[i + 1 :]:
            want *= d
        if x.stride(i) != want:
            return None
    if x.stride(s) != k_ext:
        return None
    if s == 0:
        return 1, shape[0], k_ext, k_ext, 0
    return shape[0], shape[1], k_ext, k_ext, x.stride(0)


def _seqs_dense_args(vals: dict, info: dict, axis: int | None) -> tuple:
    """Normalise a closed-form ``seqs`` call to dense ``(T, N)`` rows.

    Single sequence operand (the mode contract): movedim the step
    axis to the front, flatten the state space to ``N`` — the same
    ``(T, N)`` layout every scan kernel consumes.
    """
    m = info["seq"][0]
    x = vals[m]
    ax = _seq_axis(x.shape, vals[info["h"]].shape, axis)
    xt = x.movedim(ax, 0)
    state = xt.shape[1:]
    dt = _scan_dtype(vals, info["params"])
    rows = xt.to(dt).contiguous().view(xt.shape[0], -1)
    return rows, state, rows.shape[0], rows.shape[1], dt, ax


def _seqs_binding(mod: Any, info: dict, axis: int | None) -> Any:
    """Host wrapper for an emitted-scan kernel module — mode dispatch.

    ``fn(*vals)`` binds the spec metavars positionally, runs the
    closed-form tile kernel for ``cumsum``/``cumprod`` specs and the
    serial emitted fold otherwise; both return the prefix tensor in
    the operand's own shape (step axis restored).  *mod* is the
    generated module — ``_gs`` everywhere, plus ``_gsc`` on the
    closed-form modes.
    """
    mode = info.get("scan_mode", "fold")
    if mode in ("cumsum", "cumprod"):
        if len(info["seq"]) != 1 or info.get("const"):
            raise ValueError(
                "closed-form seqs scan needs exactly one operand"
            )
        return _seqs_scan_binding(mod, info, axis)
    return _seqs_fold_binding(mod._gs, info, axis)


def _seqs_scan_binding(mod: Any, info: dict, axis: int | None) -> Any:
    """Host wrapper for the ``tl.cumsum``/``tl.cumprod`` tile kernels.

    Per call: the strided ``(P, T, K)`` view when the operand's
    strides allow it (no transpose paid), else the dense ``(T, N)``
    rows.  Both kernels denote the same inclusive scan; the output
    is always a fresh contiguous tensor in the operand's shape.
    """

    def fn(*args: Any, **_attrs: Any) -> Any:
        import torch

        vals = dict(zip(info["params"], args, strict=True))
        m = info["seq"][0]
        x = vals[m]
        ax = _seq_axis(x.shape, vals[info["h"]].shape, axis)
        dt = _scan_dtype(vals, info["params"])
        if dt != x.dtype:
            raise ValueError(
                f"seqs scan dtype {dt} != operand dtype {x.dtype}"
            )
        t_len = x.shape[ax]
        if t_len > _SEQS_MAX_T:
            raise ValueError(
                f"seqs scan step extent {t_len} > {_SEQS_MAX_T}"
            )
        if not x.is_cuda:
            raise RuntimeError(
                "generated seqs kernels need CUDA tensors"
            )
        bt = 1 << (t_len - 1).bit_length()
        view = _seqs_strided_view(x, ax)
        if view is not None:
            p_ext, _t, k_ext, s_t, s_p = view
            out = torch.empty(
                p_ext, t_len, k_ext, dtype=dt, device=x.device
            )
            mod._gs[(p_ext, (k_ext + _SEQS_BN - 1) // _SEQS_BN)](
                x,
                out,
                t_len,
                k_ext,
                p_ext,
                s_t,
                s_p,
                BT=bt,
                BN=_SEQS_BN,
            )
            return out.reshape(x.shape)
        rows, state, t_len, n, _dt, ax = _seqs_dense_args(
            vals, info, axis
        )
        out = torch.empty(t_len, n, dtype=dt, device=rows.device)
        mod._gsc[((n + _SEQS_BN - 1) // _SEQS_BN,)](
            rows,
            out,
            t_len,
            n,
            BT=bt,
            BN=_SEQS_BN,
        )
        return out.view((t_len, *state)).movedim(0, ax)

    return fn


def _seqs_fold_binding(kern: Any, info: dict, axis: int | None) -> Any:
    """Host wrapper for the serial emitted-fold kernel.

    Same operand normalisation as :func:`_seq_args` — the ``(T, N)``
    rows, the broadcast state space, the flattened carry — then a
    ``(T, N)`` output viewed back to ``(T, *state)`` and moved so
    the emitted axis lands where the step axis sat.
    """

    def fn(*args: Any, **_attrs: Any) -> Any:
        import torch

        vals = dict(zip(info["params"], args, strict=True))
        xs = torch.broadcast_tensors(*(vals[m] for m in info["seq"]))
        full = xs[0].shape
        if not full:
            raise ValueError("scan operands carry no step axis")
        ax = _seq_axis(full, vals[info["h"]].shape, axis)
        t_len = full[ax]
        state = full[:ax] + full[ax + 1 :]
        dt = _scan_dtype(vals, info["params"])
        rows = {
            m: x.to(dt).movedim(ax, 0).contiguous().view(t_len, -1)
            for m, x in zip(info["seq"], xs, strict=True)
        }
        h_flat = (
            torch.broadcast_to(vals[info["h"]].to(dt), state)
            .contiguous()
            .view(-1)
        )
        cflat = {
            m: torch.broadcast_to(vals[m].to(dt), state)
            .contiguous()
            .view(-1)
            for m in info.get("const", ())
        }
        if not rows[info["seq"][0]].is_cuda:
            raise RuntimeError(
                "generated seqs kernels need CUDA tensors"
            )
        n = h_flat.numel()
        out = torch.empty(t_len, n, dtype=dt, device=h_flat.device)
        kern[((n + _SCAN_BLOCK - 1) // _SCAN_BLOCK,)](
            *(rows[m] for m in info["seq"]),
            *(cflat[m] for m in info.get("const", ())),
            h_flat,
            out,
            t_len,
            n,
            BLOCK=_SCAN_BLOCK,
        )
        return out.view((t_len, *state)).movedim(0, ax)

    return fn


# ---------------------------------------------------------------------------
#  seqt codegen — the fused trajectory + read-out scan
# ---------------------------------------------------------------------------
#
# ``kind="seqt"`` is the ``seqs`` trajectory fold with the per-step
# read-out contraction FUSED into the same kernel: where ``seqs``
# writes the whole ``(T, ·, state)`` trajectory for T spelled
# ``einsum`` consumers to read back, ``seqt`` writes only the
# read-outs — ``o_t[o] = Σ_c acc_t[o, c] · w_t[p(o), c]`` — one
# launch for fold + contraction + stack.  The site record's
# ``"readout"`` carries the einsum equation, which operand position
# holds the state, the read-out operand's step axis, and the
# ``stack`` node + dim the minted op denotes — the collector
# (:func:`play._seqt_readout`) only emits the kind when EVERY
# prefix state feeds exactly one matching two-operand einsum whose
# outputs stack in step order.
#
# Kernel layout: state axes split into OUTER (the labels the
# equation's output keeps — batch/head dims) and CONTRACT (the one
# summed label).  One program owns a ``BLOCK_O``-wide slab of the
# flattened outer space times the WHOLE contract axis, so the
# per-step contraction is a local ``tl.sum`` — the same
# program-decomposition trick ``_gsm`` uses for ``Kᵀ·acc``.  The
# read-out operand must carry the contract axis LAST and a PREFIX
# of the outer axes (its remaining outer labels broadcast); its
# flat index is ``p(o)·C + c`` with ``p(o) = o // suffix``.

#: Contract-axis extent cap for the seqt kernel — the accumulator
#: tile is ``_SEQT_BLOCK_O x BLOCK_C`` registers per program.
_SEQT_MAX_C = 64
_SEQT_BLOCK_O = 64


def _seqt_parse_equation(equation: Any, n_ops: int):
    """Parse an einsum equation into ``(subs, out)`` or ``None``.

    Only the explicit ``lhs->out`` form with plain unique-letter
    subscripts is accepted — no ``...`` ellipsis, no implicit-mode
    equations, no repeated labels inside one operand (diagonal
    reads are outside this tier's grammar).
    """
    if not isinstance(equation, str) or "..." in equation:
        return None
    lhs, sep, out = equation.partition("->")
    if not sep:
        return None
    subs = lhs.split(",")
    if len(subs) != n_ops or not out:
        return None
    for s in (*subs, out):
        if not s or not s.isascii() or not s.isalpha():
            return None
        if len(set(s)) != len(s):
            return None
    return subs, out


def _seqt_layout(subs: Any, out: Any, state_pos: int):
    """Classify a two-operand einsum as a fused scan read-out.

    Returns ``{"contract", "outer", "w_k", "kept", "s_sub", "w_sub"}``
    or ``None``.  The recognised shape::

        out[kept…] = Σ_c state[kept…, c] · w[kept-prefix…, c]

    * the state operand's subscripts split into the labels the
      output keeps (``kept`` — the OUTER axes, in state-axis order)
      and exactly ONE summed label shared with the read-out operand
      (``contract`` — the axis the fold's state contracts over);
    * the output subscripts are exactly the kept labels in
      state-axis order (no permutation, no extra labels);
    * the read-out operand's subscripts are a PREFIX of the kept
      labels followed by the contract label LAST — so its per-step
      slice broadcasts over the suffix outer axes the program slab
      owns.

    Anything else — multi-operand equations, several summed labels,
    permuted outputs, non-prefix read-out operands — declines: the
    collector then keeps the ``seqs`` trajectory offer instead.
    """
    if len(subs) != 2 or state_pos not in (0, 1):
        return None
    s_sub, w_sub = subs[state_pos], subs[1 - state_pos]
    contract = [x for x in s_sub if x in w_sub and x not in out]
    if len(contract) != 1:
        return None
    c = contract[0]
    kept = [x for x in s_sub if x in out]
    if "".join(kept) != out or set(s_sub) != set(kept) | {c}:
        return None
    w_outer = [x for x in w_sub if x != c]
    if (
        not w_sub
        or w_sub[-1] != c
        or any(x not in kept for x in w_outer)
        or w_outer != kept[: len(w_outer)]
    ):
        return None
    return {
        "contract": s_sub.index(c),
        "outer": [i for i, x in enumerate(s_sub) if x != c],
        "w_k": len(w_outer),
        "kept": kept,
        "s_sub": s_sub,
        "w_sub": w_sub,
    }


def _seqt_static(readout: Any):
    """Parse + classify a ``"readout"`` record once per binding.

    Returns the static descriptor the denotation and the Triton
    binding share — equation, state-operand position, the
    :func:`_seqt_layout` classification, the read-out operand's step
    axis, the stack dim, and the state subscripts' rank — or
    ``None`` when the record is absent or outside the grammar (the
    callables then raise at call time, the honest-decline contract).
    """
    if not isinstance(readout, dict):
        return None
    spos = readout.get("state_pos")
    sd = readout.get("stack_dim")
    ops = readout.get("ops")
    if (
        type(spos) is not int
        or type(sd) is not int
        or not isinstance(ops, list)
        or len(ops) != 1
    ):
        return None
    op = ops[0]
    if not isinstance(op, dict) or type(op.get("axis")) is not int:
        return None
    parsed = _seqt_parse_equation(readout.get("equation"), 2)
    if parsed is None:
        return None
    lay = _seqt_layout(parsed[0], parsed[1], spos)
    if lay is None:
        return None
    return {
        "equation": readout["equation"],
        "state_pos": spos,
        "layout": lay,
        "w_axis": op["axis"],
        "stack_dim": sd,
        "s_rank": len(parsed[0][spos]),
    }


def seqt_denotation(
    spec: Any, *, readout: Any = None, axis: int | None = None
) -> Any:
    """Spelled denotation of a ``kind="seqt"`` fused read-out spec.

    The minted ``scan_<i>_k`` op denotes ``stack(einsum(S_t, w_t))``
    — the trajectory fold whose per-step states contract with the
    read-out operand and stack along the recorded dim — NOT the
    ``(T, ·, state)`` trajectory ``seqs`` emits and NOT the ambient
    ``applyd`` map application.  The callable walks the same
    ``acc = a_t * acc + b_t`` recurrence as
    :func:`seqs_denotation`'s fold arm, then per step evaluates the
    recorded einsum with ``acc`` in the state operand's position and
    the bound read-out tensor sliced at its own step axis, and
    ``torch.stack``s the results at ``stack_dim`` — the same
    composite the spelled module runs, independently checkable
    against it.

    Args are the spec metavars positionally (``info["params"]``)
    followed by the read-out operand tensors in ``readout["ops"]``
    order — the same positional contract the Triton binding and
    the delivery route keep.  An absent/ineligible ``readout``
    record raises at call time rather than denoting a guess.
    """
    import torch

    info = scan_spec(spec)
    if info is None or info["kind"] != "seqt":
        return None
    parts = _applyd_operands(spec)
    if parts is None:
        return None
    leaves = _affd_leaves(parts[0])
    if leaves is None or len(leaves) != 1:
        return None
    a_spec, b_spec = leaves[0]
    seq_metas = list(info["seq"])
    h_meta = info["h"]
    params = list(info["params"])
    ax = axis if axis is not None else info.get("axis")
    bindings = _ambient_bindings()
    st = _seqt_static(readout)

    def fn(*args: Any, **_attrs: Any) -> Any:
        vals = dict(zip(params, args[: len(params)], strict=True))
        ro_vals = args[len(params) :]
        if st is None or len(ro_vals) != 1:
            raise ValueError("seqt read-out spec is incomplete")
        xs = torch.broadcast_tensors(*(vals[m] for m in seq_metas))
        full = xs[0].shape
        if not full:
            raise ValueError("scan operands carry no step axis")
        step_ax = _seq_axis(full, vals[h_meta].shape, ax)
        t_len = full[step_ax]
        state = full[:step_ax] + full[step_ax + 1 :]
        if len(state) != st["s_rank"]:
            raise ValueError(
                "seqt state rank disagrees with the read-out"
            )
        moved = [x.movedim(step_ax, 0) for x in xs]
        acc = torch.broadcast_to(vals[h_meta], state)
        consts = info.get("const", ())
        w = ro_vals[0]
        wax = st["w_axis"] % w.dim()
        eq = st["equation"]
        spos = st["state_pos"]
        outs = []
        for t in range(t_len):
            env = {
                m: mv[t] for m, mv in zip(seq_metas, moved, strict=True)
            }
            for c in consts:
                env[c] = vals[c]
            a_t = _eval_spec(a_spec, env, bindings)
            b_t = _eval_spec(b_spec, env, bindings)
            acc = a_t * acc + b_t
            wt = w.select(wax, t)
            eargs = [acc if i == spos else wt for i in range(2)]
            outs.append(torch.einsum(eq, *eargs))
        return torch.stack(outs, dim=st["stack_dim"])

    return fn


def _seqt_triton_source(info: dict) -> str:
    """Emit the fused trajectory+read-out kernel module for a ``seqt``.

    One program per ``BLOCK_O`` outer-state elements owns the whole
    contract axis; the step loop folds ``acc = a_t·acc + b_t`` on its
    ``(BLOCK_O, BLOCK_C)`` slab exactly like the ``seqs`` fold, then
    contracts the slab against the step's read-out operand —
    ``o_t[o] = Σ_c acc[o,c]·w_t[p(o),c]`` — and stores ONLY the
    ``(T, O)`` read-out matrix (the host reshapes/moves it into the
    stack's layout).  Every sequence operand arrives contiguous
    ``(T, O, C)``, so all operand loads share one offset expression;
    ``"const"`` operands ride as ``(O, C)`` slabs loaded once.
    """
    ptrs = [f"{m}_ptr" for m in info["seq"]]
    ptrs += [f"{m}_ptr" for m in info.get("const", ())]
    params = ", ".join([*ptrs, "h_ptr", "w_ptr"])
    loads = "".join(
        f"        {m} = tl.load({m}_ptr + row + oc,\n"
        "            mask=m2, other=0.0)\n"
        for m in info["seq"]
    )
    cloads = "".join(
        f"    {m} = tl.load({m}_ptr + oc, mask=m2, other=0.0)\n"
        for m in info.get("const", ())
    )
    return (
        "import triton\n"
        "import triton.language as tl\n"
        "import triton.language.extra.libdevice as libdevice\n"
        "\n\n@triton.jit\n"
        f"def _gst({params}, out_ptr, T, O, C, P, Suf,"
        " BLOCK_O: tl.constexpr, BLOCK_C: tl.constexpr):\n"
        "    ob = tl.program_id(0)\n"
        "    o = ob.to(tl.int64) * BLOCK_O"
        " + tl.arange(0, BLOCK_O)\n"
        "    c = tl.arange(0, BLOCK_C)\n"
        "    om = o < O\n"
        "    m2 = om[:, None] & (c < C)[None, :]\n"
        "    oc = o[:, None] * C + c[None, :]\n"
        "    acc = tl.load(h_ptr + oc, mask=m2, other=0.0)\n"
        f"{cloads}"
        "    for t in range(T):\n"
        "        row = t.to(tl.int64) * O * C\n"
        f"{loads}"
        f"        acc = ({info['a']}) * acc + ({info['b']})\n"
        "        p = o // Suf\n"
        "        w = tl.load(w_ptr + t.to(tl.int64) * P * C"
        " + p[:, None] * C + c[None, :], mask=m2, other=0.0)\n"
        "        ot = tl.sum(acc * w, axis=1)\n"
        "        tl.store(out_ptr + t.to(tl.int64) * O + o,"
        " ot, mask=om)\n"
    )


#: Loaded seqt kernel modules keyed by cache path — the binding
#: resolves its layout per call but the ``.py`` artifact (and its
#: ``triton.jit`` compile) is loaded once per spec.
_SEQT_LOADED: dict = {}


def _seqt_kernel_for(info: dict, name: str):
    """Emit + load the seqt kernel module for a spec.

    Same cache contract as :func:`_seqm_kernel_for`: the source is
    spec-shaped (operand metas + rendered step exprs), so the module
    is cached by source hash under the kernel cache.
    """
    import hashlib

    src = _seqt_triton_source(info)
    digest = hashlib.sha256(src.encode()).hexdigest()[:16]
    path = _kernel_cache() / f"_gst_{name}_{digest}.py"
    kern = _SEQT_LOADED.get(path)
    if kern is None:
        if not path.exists():
            path.write_text(src)
        kern = _load_generated(path, "_gst")
        _SEQT_LOADED[path] = kern
    return kern


def _seqt_binding(
    name: str, info: dict, axis: int | None, readout: Any
) -> Any:
    """Host wrapper for the fused trajectory+read-out kernel.

    ``fn(*args)`` binds the spec metavars positionally (``params``
    order), then the read-out operand tensors in ``readout["ops"]``
    order — the same positional contract :func:`seqt_denotation`
    and the delivery route keep.  Sequence operands normalise to
    ``(T, O, C)`` slabs (step axis front, state axes permuted
    outer-then-contract, flattened), the read-out operand to
    ``(T, P, C)``, and the kernel writes the ``(T, O)`` read-out
    matrix the host reshapes into the stack's layout.  Shape
    ineligibility — a contract axis past ``_SEQT_MAX_C``, a
    read-out operand whose rest is not ``(outer-prefix…, C)``, a
    state rank disagreeing with the equation — raises, the same
    honest-decline contract the other scan bindings keep.
    """
    st = _seqt_static(readout)

    def fn(*args: Any, **_attrs: Any) -> Any:
        import torch

        vals = dict(
            zip(
                info["params"], args[: len(info["params"])], strict=True
            )
        )
        ro_vals = args[len(info["params"]) :]
        if st is None or len(ro_vals) != 1:
            raise ValueError("seqt read-out spec is incomplete")
        lay = st["layout"]
        xs = torch.broadcast_tensors(*(vals[m] for m in info["seq"]))
        full = xs[0].shape
        if not full:
            raise ValueError("scan operands carry no step axis")
        if not xs[0].is_cuda:
            raise RuntimeError(
                "generated seqt kernels need CUDA tensors"
            )
        ax = _seq_axis(full, vals[info["h"]].shape, axis)
        t_len = full[ax]
        state = full[:ax] + full[ax + 1 :]
        if len(state) != st["s_rank"]:
            raise ValueError(
                "seqt state rank disagrees with the read-out"
            )
        perm = (*lay["outer"], lay["contract"])
        outer_shape = tuple(state[i] for i in lay["outer"])
        c_ext = state[lay["contract"]]
        o_ext = math.prod(outer_shape)
        if c_ext < 1 or c_ext > _SEQT_MAX_C or o_ext < 1:
            raise ValueError(f"seqt layout ineligible: state={state}")
        dt = _scan_dtype(vals, info["params"])
        w = ro_vals[0]
        if torch.is_tensor(w):
            dt = torch.promote_types(dt, w.dtype)
        slabs = {}
        for m, x in zip(info["seq"], xs, strict=True):
            xm = x.movedim(ax, 0)
            xp = xm.permute(0, *[i + 1 for i in perm])
            slabs[m] = xp.to(dt).contiguous().view(t_len, o_ext, c_ext)
        cflat = {
            m: torch.broadcast_to(vals[m].to(dt), state)
            .permute(*perm)
            .contiguous()
            .view(o_ext, c_ext)
            for m in info.get("const", ())
        }
        h_sl = (
            torch.broadcast_to(vals[info["h"]].to(dt), state)
            .permute(*perm)
            .contiguous()
            .view(o_ext, c_ext)
        )
        # the read-out operand: step axis front, rest must be the
        # outer-prefix dims then the contract axis — (T, P, C)
        wax = st["w_axis"] % w.dim()
        wt = w.movedim(wax, 0)
        w_rest = tuple(wt.shape[1:])
        want = (
            *(state[i] for i in lay["outer"][: lay["w_k"]]),
            c_ext,
        )
        if w_rest != want or wt.shape[0] != t_len:
            raise ValueError(
                f"seqt read-out operand rest {w_rest} != {want}"
            )
        p_ext = math.prod(w_rest[:-1]) if w_rest[:-1] else 1
        suf = o_ext // p_ext
        wt = wt.to(dt).contiguous().view(t_len, p_ext, c_ext)
        kern = _seqt_kernel_for(info, name)
        if kern is None:
            raise RuntimeError("seqt kernel module failed to load")
        out = torch.empty(t_len, o_ext, dtype=dt, device=h_sl.device)
        bc = 1 << (c_ext - 1).bit_length()
        kern[((o_ext + _SEQT_BLOCK_O - 1) // _SEQT_BLOCK_O,)](
            *(slabs[m] for m in info["seq"]),
            *(cflat[m] for m in info.get("const", ())),
            h_sl,
            wt,
            out,
            t_len,
            o_ext,
            c_ext,
            p_ext,
            suf,
            BLOCK_O=_SEQT_BLOCK_O,
            BLOCK_C=bc,
        )
        r = len(outer_shape)
        return out.view(t_len, *outer_shape).movedim(
            0, st["stack_dim"] % (r + 1)
        )

    return fn


def _unrolled_binding(kern: Any, info: dict) -> Any:
    """Host wrapper for the unrolled carrier kernel.

    ``fn(*vals)`` binds the spec metavars positionally, broadcasts
    the kernel-referenced operands (and the state) to one flat index
    space, and applies each leaf step in chronological order inside
    the kernel.  Hoisted operands — the spec subterms the
    elementwise renderer declined — are evaluated on the host
    against the same env and broadcast in with the rest; metavars
    occurring only inside them stay binding args, not kernel
    operands.
    """
    computed = info.get("computed", [])

    def fn(*args: Any, **_attrs: Any) -> Any:
        import mlopt_torch.torch_bridge as tb
        import torch

        vals = dict(zip(info["params"], args, strict=True))
        extra = [
            _eval_spec(sub, vals, tb._IR_TO_TORCH)
            for _n, sub in computed
        ]
        # Broadcast exactly the operands the kernel loads — the
        # expr-referenced metavars, the state, and the live hoists
        # (``kernel_args`` order).  Spec metavars buried inside
        # hoists are bound for the host subterm eval but carry
        # shapes foreign to the flat index space; broadcasting them
        # in raised spurious shape errors on specs the kernel
        # denotes fine.
        xs = torch.broadcast_tensors(
            *(vals[m] for m in info["flat"]),
            vals[info["h"]],
            *extra,
        )
        if not xs[0].is_cuda:
            raise RuntimeError(
                "generated scan kernels need CUDA tensors"
            )
        shape = xs[0].shape
        dt = _scan_dtype(vals, info["params"])
        flats = [x.to(dt).contiguous().view(-1) for x in xs]
        n = flats[0].numel()
        out = torch.empty(n, dtype=dt, device=flats[0].device)
        kern[((n + _SCAN_BLOCK - 1) // _SCAN_BLOCK,)](
            *flats, out, n, BLOCK=_SCAN_BLOCK
        )
        return out.view(shape)

    return fn


# ---------------------------------------------------------------------------
#  seqm codegen — the tiled matrix-state scan kernel
# ---------------------------------------------------------------------------
#
# One program owns a ``(BLOCK_DK, BLOCK_DV)`` tile of the ``(dk, dv)``
# state for one batch row and one dv column block; the step loop is
# serial in ``t`` exactly like the flat ``seq`` kernel, but the
# ``b`` step is the rank-1-feedback write::
#
#   read_j = V_j - S * sum_i K_i * acc_ij      (the K^T acc matvec)
#   acc    = a * acc + C * read^T
#
# Every elementwise operand is loaded per step under its own
# broadcast role — ``frame`` operands normalise to the ``(DK, DV)``
# tile, ``row`` operands stay ``(BV,)`` vectors (or scalars), and
# the matvec key ``K`` loads as a ``(DK,)`` vector.  The role is
# fixed by WHERE the metavar sits in the parsed ``b`` skeleton, the
# load shape by the operand's own post-view rest shape; both are
# validated against ``dk``/``dv`` at call time and an ineligible
# shape raises (the referee declines the claim, delivery keeps the
# spelled fold).

#: ``dk`` extent cap for the seqm kernel — the accumulator tile is
#: ``BLOCK_DK x BLOCK_DV`` registers per program; 64 rows is the
#: honest ceiling before occupancy collapses.  ``dv`` tiles freely
#: over the program grid.
_SEQM_MAX_DK = 64
_SEQM_BV = 64


def _seqm_pad2(rest: tuple) -> tuple:
    """Left-pad a post-view rest shape to rank 2; decline rank > 2."""
    if len(rest) > 2:
        raise ValueError(f"seqm operand rest rank {len(rest)} > 2")
    r = list(rest)
    while len(r) < 2:
        r.insert(0, 1)
    return tuple(r)


def _seqm_metas(sub: Any) -> list:
    """Ordered distinct metavars of a seqm sub-spec (``None`` → [])."""
    if sub is None or sub == 1:
        return []
    return [m for m in _spec_metas(sub) if isinstance(m, str)]


def _seqm_rename(spec: Any, ren: dict) -> Any:
    """Return *spec* with metavar leaves renamed through *ren*."""
    if isinstance(spec, str):
        return ren.get(spec, spec)
    if isinstance(spec, (tuple, list)):
        return tuple(_seqm_rename(e, ren) for e in spec)
    return spec


def _seqm_role(rest2: tuple, dk: int, dv: int, kind: str) -> str:
    """Classify a padded rest shape into its kernel load tag.

    Tags: ``fscalar``/``fcol``/``frow``/``full`` for ``"frame"``
    operands (normalised to the ``(DK, DV)`` tile), ``rscalar``/
    ``rrow`` for ``"row"`` operands (kept ``(BV,)`` or scalar), and
    ``ccol`` for the ``"col"`` matvec key (``(DK,)``).  *kind* is
    the operand's skeleton role; the checks mirror torch's
    right-aligned broadcasting on the batch-stripped rest, so a
    shape the kernel accepts is one the spelled eval broadcasts
    identically.
    """
    r0, r1 = rest2
    if kind == "col":
        if rest2 != (1, dk):
            raise ValueError(f"seqm key rest {rest2} != (1, {dk})")
        return "ccol"
    if kind == "row":
        if r0 != 1 or r1 not in (1, dv):
            raise ValueError(f"seqm row rest {rest2} not in (1, dv)")
        return "rscalar" if r1 == 1 else "rrow"
    # frame
    if r0 not in (1, dk) or r1 not in (1, dv):
        raise ValueError(f"seqm frame rest {rest2} not in (dk, dv)")
    if r0 == 1 and r1 == 1:
        return "fscalar"
    if r0 == dk and r1 == 1:
        return "fcol"
    if r0 == 1 and r1 == dv:
        return "frow"
    return "full"


def _seqm_signature(info: dict, xs: list, dk: int, dv: int) -> dict:
    """Build the kernel signature for bound operand tensors.

    *xs* are the ``(T, B, *rest)`` contiguous operand tensors in
    ``info["seq"]`` order.  Returns the load list, rendered step
    exprs and block sizes; raises ``ValueError`` on any operand
    whose rest shape cannot play its skeleton role — an honest
    decline the referee turns into a dropped claim.
    """
    seq = list(info["seq"])
    rests = {
        m: _seqm_pad2(tuple(xs[i].shape[2:])) for i, m in enumerate(seq)
    }

    frame_ms: list = []
    for sub in (info["a"], info["c"]):
        for m in _seqm_metas(sub):
            if m not in frame_ms:
                frame_ms.append(m)
    row_ms: list = []
    for sub in (info["v"], info["s"]):
        for m in _seqm_metas(sub):
            if m not in row_ms:
                row_ms.append(m)
    k_meta = info["k"]

    # a metavar may sit in several skeleton positions — each gets
    # its own load and var (``m`` frame, ``m_r`` row, ``m_k`` key)
    loads: list = []
    frame_ren: dict = {}
    row_ren: dict = {}
    for m in frame_ms:
        frame_ren[m] = m
        loads.append(
            (
                m,
                seq.index(m),
                "frame",
                _seqm_role(rests[m], dk, dv, "frame"),
            )
        )
    for m in row_ms:
        row_ren[m] = f"{m}_r"
        loads.append(
            (
                f"{m}_r",
                seq.index(m),
                "row",
                _seqm_role(rests[m], dk, dv, "row"),
            )
        )
    k_var = f"{k_meta}_k"
    loads.append(
        (
            k_var,
            seq.index(k_meta),
            "col",
            _seqm_role(rests[k_meta], dk, dv, "col"),
        )
    )

    a_ren = dict(frame_ren)
    a_expr = (
        "1.0"
        if info["a"] == 1
        else _triton_expr_h(_seqm_rename(info["a"], a_ren), None, None)
    )
    c_expr = _triton_expr_h(
        _seqm_rename(info["c"], a_ren), None, _TRITON_CONST_1D
    )
    v_expr = _triton_expr_h(
        _seqm_rename(info["v"], row_ren), None, _TRITON_CONST_1D
    )
    s_expr = (
        None
        if info["s"] is None
        else _triton_expr_h(
            _seqm_rename(info["s"], row_ren), None, _TRITON_CONST_1D
        )
    )
    if (
        a_expr is None
        or c_expr is None
        or v_expr is None
        or (info["s"] is not None and s_expr is None)
    ):
        raise ValueError("seqm step expr failed to render")
    return {
        "loads": loads,
        "a": a_expr,
        "c": c_expr,
        "v": v_expr,
        "s": s_expr,
        "k": k_var,
    }


def _seqm_load_line(
    metas: list, load: tuple, indent: str = "        "
) -> str:
    """Emit one per-step operand load inside the ``t`` loop.

    *load* is ``(var, seq_index, kind, tag)``; the base offset is
    ``(t * B + b) * N<idx>`` over the ``(T, B, *rest)`` contiguous
    operand — ``N<idx>`` its rest numel, passed as a kernel arg.
    Frame tags normalise the loaded block to ``(BLOCK_DK,
    BLOCK_DV)``; row/key tags keep their native ``(BV,)``/``(DK,)``
    shape.
    """
    var, midx, _kind, tag = load
    ptr = f"{metas[midx]}_ptr"
    off = f"((t.to(tl.int64) * B + b) * N{midx})"
    if tag == "fscalar":
        return (
            f"{indent}{var} = tl.broadcast_to("
            f"tl.reshape(tl.load({ptr} + {off} + one), (1, 1)),"
            " (BLOCK_DK, BLOCK_DV))\n"
        )
    if tag == "fcol":
        return (
            f"{indent}{var} = tl.broadcast_to("
            f"tl.reshape(tl.load({ptr} + {off} + i,"
            " mask=i < dk, other=0.0), (BLOCK_DK, 1)),"
            " (BLOCK_DK, BLOCK_DV))\n"
        )
    if tag == "frow":
        return (
            f"{indent}{var} = tl.broadcast_to("
            f"tl.reshape(tl.load({ptr} + {off} + jg,"
            " mask=jg < dv, other=0.0), (1, BLOCK_DV)),"
            " (BLOCK_DK, BLOCK_DV))\n"
        )
    if tag == "full":
        return (
            f"{indent}{var} = tl.load({ptr} + {off}"
            " + im * dv + jm, mask=hmask, other=0.0)\n"
        )
    if tag == "rscalar":
        return f"{indent}{var} = tl.load({ptr} + {off} + one)\n"
    if tag == "rrow":
        return (
            f"{indent}{var} = tl.load({ptr} + {off} + jg,"
            " mask=jg < dv, other=0.0)\n"
        )
    # ccol — the matvec key, a (DK,) vector
    return (
        f"{indent}{var} = tl.load({ptr} + {off} + i,"
        " mask=i < dk, other=0.0)\n"
    )


def _seqm_triton_source(info: dict, sig: dict, n_seq: int) -> str:
    """Emit the seqm kernel module for a resolved signature.

    Signature: the ``info["seq"]`` operand pointers in order, then
    ``h_ptr``/``out_ptr``/``T``/``B``/``dk``/``dv``/``N0``…``Nk``
    (per-operand rest numels) and the ``BLOCK_DK``/``BLOCK_DV``
    constexprs.  One program per ``(batch, dv-block)`` owns the
    ``(dk, dv)`` state tile; the ``t`` loop is serial —
    ``read = V - S·(Kᵀ acc)`` via a masked ``tl.sum`` matvec, then
    ``acc = a·acc + C·readᵀ`` — the delta-rule fold.
    """
    metas = list(info["seq"])
    params = ", ".join(f"{m}_ptr" for m in metas)
    ns = ", ".join(f"N{i}" for i in range(n_seq))
    loads = "".join(
        _seqm_load_line(metas, load) for load in sig["loads"]
    )
    proj = (
        f"        proj = tl.sum("
        f"tl.reshape({sig['k']}, (BLOCK_DK, 1)) * acc, axis=0)\n"
    )
    if sig["s"] is None:
        read = f"        r = ({sig['v']}) - proj\n"
    else:
        read = f"        r = ({sig['v']}) - ({sig['s']}) * proj\n"
    read_bc = (
        "        rbc = tl.broadcast_to("
        "tl.reshape(r, (1, BLOCK_DV)), (BLOCK_DK, BLOCK_DV))\n"
    )
    update = f"        acc = ({sig['a']}) * acc + ({sig['c']}) * rbc\n"
    return (
        "import triton\n"
        "import triton.language as tl\n"
        "import triton.language.extra.libdevice as libdevice\n"
        "\n\n@triton.jit\n"
        f"def _gsm({params}, h_ptr, out_ptr, T, B, dk, dv, {ns},"
        " BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr):\n"
        "    b = tl.program_id(0)\n"
        "    jb = tl.program_id(1)\n"
        "    i = tl.arange(0, BLOCK_DK)\n"
        "    j = tl.arange(0, BLOCK_DV)\n"
        "    jg = jb.to(tl.int64) * BLOCK_DV + j\n"
        "    im = tl.reshape(i, (BLOCK_DK, 1))\n"
        "    jm = tl.reshape(jg, (1, BLOCK_DV))\n"
        "    one = tl.arange(0, 1)\n"
        "    hmask = (im < dk) & (jm < dv)\n"
        "    hoff = (b.to(tl.int64) * dk + im) * dv + jm\n"
        "    acc = tl.load(h_ptr + hoff, mask=hmask, other=0.0)\n"
        "    for t in range(T):\n"
        f"{loads}"
        f"{proj}"
        f"{read}"
        f"{read_bc}"
        f"{update}"
        "    tl.store(out_ptr + hoff, acc, mask=hmask)\n"
    )


#: Loaded seqm kernel modules keyed by cache path — the binding
#: resolves its signature per call, but the ``.py`` artifact (and
#: its ``triton.jit`` compile) is loaded once per signature.
_SEQM_LOADED: dict = {}


def _seqm_kernel_for(info: dict, sig: dict, name: str, n_seq: int):
    """Emit + load the seqm kernel for a resolved signature.

    The module is cached by source hash under the kernel cache —
    each distinct operand-shape signature gets its own ``.py``
    artifact, inspectable on disk like every generated tier — and
    the loaded ``triton.jit`` callable is memoised per path so the
    per-call signature resolution never re-imports the module.
    """
    import hashlib

    src = _seqm_triton_source(info, sig, n_seq)
    digest = hashlib.sha256(src.encode()).hexdigest()[:16]
    path = _kernel_cache() / f"_gsm_{name}_{digest}.py"
    kern = _SEQM_LOADED.get(path)
    if kern is None:
        if not path.exists():
            path.write_text(src)
        kern = _load_generated(path, "_gsm")
        _SEQM_LOADED[path] = kern
    return kern


def _seqm_binding(name: str, info: dict, axes: Any) -> Any:
    """Host wrapper for the matrix-state scan kernel.

    ``fn(*vals)`` binds the spec metavars positionally, moves each
    operand's own step axis (``axes`` — the site record's
    per-operand map) to the front, validates the ``(T, B, *rest)``
    layout against the ``(B, dk, dv)`` state, resolves the load
    signature, and launches the fold.  Shape ineligibility raises —
    the same honest-decline contract the flat seq binding keeps.
    """

    def fn(*args: Any, **_attrs: Any) -> Any:
        import torch

        vals = dict(zip(info["params"], args, strict=True))
        moved = []
        for m in info["seq"]:
            x = vals[m]
            if axes is None or m not in axes:
                raise ValueError(f"seqm operand {m} lacks a step axis")
            moved.append(x.movedim(axes[m] % x.dim(), 0))
        t_len = moved[0].shape[0]
        b_dim = moved[0].shape[1]
        for x in moved:
            if x.shape[0] != t_len or x.shape[1] != b_dim:
                raise ValueError("seqm operands disagree on (T, B)")
        hh = vals[info["h"]]
        if hh.dim() != 3 or hh.shape[0] != b_dim:
            raise ValueError(
                "seqm state must be (B, dk, dv) matching the operands"
            )
        dk, dv = int(hh.shape[1]), int(hh.shape[2])
        if dk < 1 or dk > _SEQM_MAX_DK:
            raise ValueError(
                f"seqm dk={dk} outside [1, {_SEQM_MAX_DK}]"
            )
        dt = _scan_dtype(vals, info["params"])
        xs = [x.to(dt).contiguous() for x in moved]
        hh = hh.to(dt).contiguous()
        if not (hh.is_cuda and xs[0].is_cuda):
            raise RuntimeError(
                "generated seqm kernels need CUDA tensors"
            )
        sig = _seqm_signature(info, xs, dk, dv)
        n_seq = len(info["seq"])
        kern = _seqm_kernel_for(info, sig, name, n_seq)
        if kern is None:
            raise RuntimeError("seqm kernel module failed to load")
        out = torch.empty_like(hh)
        bv = min(_SEQM_BV, 1 << (dv - 1).bit_length())
        bd = 1 << (dk - 1).bit_length()
        nrs = [int(x[0].numel() // b_dim) for x in xs]
        kern[(b_dim, (dv + bv - 1) // bv)](
            *xs,
            hh,
            out,
            t_len,
            b_dim,
            dk,
            dv,
            *nrs,
            BLOCK_DK=bd,
            BLOCK_DV=bv,
        )
        return out

    return fn


def scan_triton_bindings(
    handlers: dict, *, axis: int | None = -2
) -> dict:
    """Generate a Triton scan kernel per ``applyd``-carrier handler.

    The second codegen tier: each handler whose *pattern*
    :func:`scan_spec` recognises gets a binding whose callable runs a
    generated sequential-scan kernel — one launch for the whole
    recurrence, real ``triton.jit`` source on disk under the kernel
    cache.  The step axis defaults to ``-2`` (the token axis of
    ``(B, T, d)`` layouts); a handler may pin ``"axis"`` (or the spec
    may declare it as an ``applyd`` attr), and ``axis=None`` derives
    the axis from the state shape at call time.

    Unrecognised or un-generatable specs are skipped — the handler
    keeps its torch path, exactly as the elementwise tier declines.
    """
    out: dict[str, Any] = {}
    for h in handlers.values():
        info = scan_spec(h["pattern"])
        if info is None:
            continue
        ax = h.get("axis", info.get("axis", axis))
        if info["kind"] == "seqm":
            # the matrix-state fold: no flat ``_gs`` module exists —
            # the binding JITs a per-signature ``_gsm`` kernel at
            # call time from the handler's recorded ``axes`` map.
            out[h["kernel"]] = _seqm_binding(
                h["kernel"], info, h.get("axes")
            )
            continue
        if info["kind"] == "seqt":
            # the fused trajectory+read-out fold: the (O, C) layout
            # resolves per call like seqm — the binding JITs the
            # ``_gst`` kernel and raises on ineligible shapes.
            out[h["kernel"]] = _seqt_binding(
                h["kernel"], info, ax, h.get("readout")
            )
            continue
        src = _scan_triton_source(info)
        path = _kernel_cache() / f"_gs_{h['kernel']}.py"
        path.write_text(src)
        if info["kind"] == "seqs":
            # the emitted scan ships a MODULE — ``_gs`` plus the
            # dense ``_gsc`` fallback on closed-form modes — so the
            # binding gets the module, not a single entry point.
            mod = _load_generated_module(path)
            if mod is None:
                continue
            out[h["kernel"]] = _seqs_binding(mod, info, ax)
            continue
        kern = _load_generated(path, "_gs")
        if kern is None:
            continue  # module load declined — keep the torch path
        if info["kind"] == "seq":
            out[h["kernel"]] = _seq_binding(kern, info, ax)
        else:
            out[h["kernel"]] = _unrolled_binding(kern, info)
    return out


def _canon_taken(pattern: Any, out: dict, skip: set) -> bool:
    """Whether the spec's concrete canon is already covered.

    Equality over containment — canon specs carry attr dicts, so
    ``in``/set membership would raise on the unhashable elements.
    """
    from mlopt_discovery.meta_arena import _canon_concrete

    canon = _canon_concrete(pattern)
    return any(canon == c for c in skip) or any(
        _canon_concrete(h["pattern"]) == canon for h in out.values()
    )


def _site_parts(item: Any) -> tuple:
    """Split a site-collector item into ``(term_or_spec, extra)``.

    Collectors return plain specs/terms — or ``{"spec": ...,
    "leaf_env": ...}`` records when the site exists only post-
    saturation.  *extra* merges verbatim into the minted entry so
    the recorded metavar binding reaches the referee.
    """
    if isinstance(item, dict):
        extra = {"leaf_env": item.get("leaf_env")}
        if "chain" in item:
            # seq/seqm sites carry the unrolled chain root the fold
            # denotes — merged verbatim so delivery can rewrite the
            # extracted chain to the kernel call (the claim path's
            # ``applyd`` pattern never occurs in raw IR).
            extra["chain"] = item["chain"]
        if "axis" in item:
            # a site that resolved its own step axis (the seq
            # collector's shape-checked dim) overrides the builder's
            # default — merged after ``"axis": axis`` in the entry.
            extra["axis"] = item["axis"]
        if "axes" in item:
            # seq-matrix sites resolve a step axis PER OPERAND (no
            # shared broadcast frame exists) — the map reaches the
            # binding and denotation through the handler entry.
            extra["axes"] = item["axes"]
        if "states" in item:
            # seqs trajectory sites carry the chain's prefix-state
            # nodes + step indices — the deliver-time route's LHS
            # set: each state e-class rewrites to a ``select`` off
            # the emitted trajectory tensor.
            extra["states"] = item["states"]
        if "readout" in item:
            # seqt fused read-out sites carry the einsum/stack
            # record — the deliver-time route's LHS is the stack
            # e-class, replaced by the one kernel call denoting
            # ``stack(einsum(S_t, w_t))``.
            extra["readout"] = item["readout"]
        return item["spec"], extra
    return item, {}


def scan_handlers(
    terms: list, *, axis: int | None = -2, covered: set | None = None
) -> dict:
    """Mint handler entries for ``applyd``-carrier scan terms.

    The carrier analogue of :func:`gen_handlers`: ``scan_<i>`` entries
    ``{pattern, kernel, args, axis}`` for each term whose spec is a
    recognised scan — the *sequence* form (an ``applyd(aff_diag(a,
    b), h)`` site whose operands carry a step axis) or the *unrolled*
    form (an ``affd_compose`` tree, e.g. what ``affd_scan2_lift`` /
    ``affd_scan4_lift`` leave behind).  ``axis`` is recorded in the
    entry as the declared step axis; ``None`` defers to call-time
    derivation.  A ``{"spec", "leaf_env"}`` site record merges its
    ``leaf_env`` into the entry — the recorded metavar binding the
    referee evaluates, since the occurrence lives only in the
    collector's throwaway e-graph.

    The caller owns the semantics check for the sequence form: the
    minted ``scan_<i>_k`` op's declared meaning is the fold, so the
    handler belongs on sites whose ``a``/``b`` operands genuinely
    carry a step axis — the binding raises on operands that do not.
    """
    from mlopt_discovery.meta_arena import _canon_concrete, _spec_of

    skip = covered if covered is not None else set()
    out: dict[str, dict] = {}
    # (canon, site-record) pairs already minted — the canon alone is
    # NOT the dedup key for site records: two sites can share one
    # spec skeleton while ``leaf_env`` binds different operands and
    # ``chain`` routes a different recurrence — distinct folds that
    # must mint distinct ``scan_<i>`` kernels.  Plain spec/term
    # items carry ``extra == {}`` and dedup on canon exactly as
    # before.
    seen: list = []
    for term in terms:
        term, extra = _site_parts(term)
        pattern = (
            term
            if isinstance(term, (tuple, list))
            else _spec_of(term, {})
        )
        if scan_spec(pattern) is None:
            continue
        canon = _canon_concrete(pattern)
        # equality, not membership — canon specs carry attr dicts
        if any(canon == c for c in skip) or any(
            c == canon and e == extra for c, e in seen
        ):
            continue
        seen.append((canon, extra))
        tag = f"scan_{len(out)}"
        out[tag] = {
            "pattern": pattern,
            "kernel": f"{tag}_k",
            "args": tuple(_spec_metas(pattern)),
            "axis": axis,
            **extra,
        }
    return out


def scan_triton_sink(handlers: dict, *, axis: int | None = -2) -> Any:
    """Build a ``TorchSink`` binding minted scan ops to Triton kernels.

    Same posture as :func:`triton_sink`: ``supported_ops`` reports the
    ambient table plus every generated scan kernel — the feasibility
    bound the board prices the carrier claims against — and a claim
    on an unrecognised spec stays unpriced-unlowerable.
    """
    from mlopt_core.ops import OpTable
    from mlopt_torch.adapters import TorchSink

    bindings = _ambient_bindings()
    bindings.update(scan_triton_bindings(handlers, axis=axis))
    table = OpTable.core().register(torch_bindings=bindings)
    return TorchSink(ops=table)


# ---------------------------------------------------------------------------
#  The decayed tier — a fused ``(s ⊙ d) @ v`` kernel for ``om_decayed``
# ---------------------------------------------------------------------------

#: Output-tile and contraction-chunk extents for generated decayed
#: kernels — one Triton program per ``(batch, BM-by-BN)`` tile.
_DECAY_BM = 32
_DECAY_BN = 32
_DECAY_BK = 32


def _decayed_operands(spec: Any) -> tuple | None:
    """Split a decayed spec into its ``(s, d, v)`` operand specs.

    Two spellings name the same site: the element
    ``("om_decayed", s, d, v)`` and the spelled body
    ``("matmul", ("mul", s, d), v)`` it lifts from.  ``None`` on
    anything else — wrong heads, wrong arities, or attr-carrying op
    nodes (the element is attr-free; a spec carrying attrs this
    tier cannot model declines honestly).
    """
    if not (isinstance(spec, (tuple, list)) and len(spec) >= 2):
        return None
    kids = [e for e in spec[1:] if not isinstance(e, dict)]
    if len(kids) != len(spec) - 1:
        return None  # an attr dict this tier doesn't model
    if spec[0] == "om_decayed":
        return tuple(kids) if len(kids) == 3 else None
    if spec[0] != "matmul" or len(kids) != 2:
        return None
    return _mul_matmul_operands(kids[0], kids[1])


def _mul_matmul_operands(lhs: Any, v: Any) -> tuple | None:
    """Split a spelled ``matmul(mul(s, d), v)`` into ``(s, d, v)``.

    ``None`` when the left operand is not an attr-free two-operand
    ``mul`` — the spelled form must be exactly the decayed body.
    """
    if not (
        isinstance(lhs, (tuple, list))
        and len(lhs) == 3
        and lhs[0] == "mul"
        and all(not isinstance(e, dict) for e in lhs)
    ):
        return None
    return lhs[1], lhs[2], v


def _decayed_meta_ok(m: Any) -> bool:
    """Check a decayed-operand metavar is kernel-name safe."""
    return (
        isinstance(m, str)
        and m.isidentifier()
        and not keyword.iskeyword(m)
    )


def _rename_metas(spec: Any, prefix: str) -> Any:
    """Rename metavar leaves to ``{prefix}{meta}_t`` tile variables."""
    if isinstance(spec, str):
        return f"{prefix}{spec}_t"
    if isinstance(spec, (tuple, list)):
        return (
            spec[0],
            *(
                e if isinstance(e, dict) else _rename_metas(e, prefix)
                for e in spec[1:]
            ),
        )
    return spec  # Const leaf — the canonical form carries the value


def _decayed_const(i: int) -> str:
    """Const-leaf template for decayed operand role *i*.

    The ``_gd`` kernel's operand tiles are 2-D: roles 0/1 (the
    ``s``/``d`` mul pair) load ``(BM, BK)`` tiles and role 2 (``v``)
    loads ``(BK, BN)`` — a const leaf must render a ``tl.full`` of
    the matching shape so ``libdevice`` calls see matching operand
    dtypes *and* shapes (the same fp32-literal clash as the 1-D
    tier, one rank up).  ``BM``/``BN``/``BK`` are the kernel's
    ``tl.constexpr`` params; ``out_ptr.dtype.element_ty`` follows
    the launch dtype.
    """
    shape = "(BM, BK)" if i < 2 else "(BK, BN)"
    return f"tl.full({shape}, {{v}}, dtype=out_ptr.dtype.element_ty)"


def decayed_spec(spec: Any) -> dict | None:
    """Recognise a decayed-contraction spec the tier can lower.

    Returns a descriptor dict, or ``None`` when the spec is outside
    the tier's grammar (decline — the handler keeps its torch
    path).  Operands may be metavars, constants, or elementwise
    sub-specs: each is rendered through the same elementwise codegen
    as the pointwise tier, so ``om_decayed(("mul", "A", "B"), "D",
    "V")`` generates the fully fused kernel.  The descriptor carries
    ``"operands"`` — per role ``{"metas", "expr"}``, where *expr* is
    the rendered Triton expression over ``a{i}_{meta}_t`` tile
    variables — and ``"params"``, the spec's metavars in order (the
    generated binding's positional signature).
    """
    operands = _decayed_operands(spec)
    if operands is None:
        return None
    params: list = []
    roles: list[dict] = []
    for i, o in enumerate(operands):
        metas = _spec_metas(o)
        if any(not _decayed_meta_ok(m) for m in metas):
            return None
        expr = _triton_expr(
            _rename_metas(o, f"a{i}_"), _decayed_const(i)
        )
        if expr is None:
            return None
        for m in metas:
            if m not in params:
                params.append(m)
        roles.append({"metas": metas, "expr": expr})
    # the dot operands must be tiles: a constant v (or a constant
    # s·d product) renders a scalar where tl.dot needs a tile — a
    # scalar v fails the rank contract anyway, so decline honestly.
    if not roles[2]["metas"] or not (
        roles[0]["metas"] or roles[1]["metas"]
    ):
        return None
    return {"kind": "decayed", "operands": roles, "params": params}


def _dvar(i: int, m: str) -> str:
    """Kernel variable stem for operand *i*'s metavar *m*."""
    return f"a{i}_{m}"


def _decayed_decls(info: dict) -> str:
    """Kernel parameters — ptr/offsets/strides/FB-flag per meta."""
    return ", ".join(
        f"{v}_ptr, {v}_bo, {v}_rs, {v}_cs"
        for i, o in enumerate(info["operands"])
        for m in o["metas"]
        for v in [_dvar(i, m)]
    )


def _decayed_flags(info: dict) -> str:
    """Emit the per-meta ``FB`` constexprs — broadcast-regime markers."""
    return "".join(
        f", {v}_FB: tl.constexpr"
        for i, o in enumerate(info["operands"])
        for m in o["metas"]
        for v in [_dvar(i, m)]
    )


def _decayed_bases(info: dict) -> str:
    """Per-program batch-base loads, specialised by the FB flags."""
    return "".join(
        f"    if {v}_FB:\n"
        f"        {v}_base = 0\n"
        f"    else:\n"
        f"        {v}_base = tl.load({v}_bo + b)\n"
        for i, o in enumerate(info["operands"])
        for m in o["metas"]
        for v in [_dvar(i, m)]
    )


def _decayed_loads(info: dict) -> tuple:
    """Emit the mul-space ``(rm, rk)`` and v-space ``(rk, rn)`` loads."""
    mloads = "".join(
        f"        {v}_t = tl.load(\n"
        f"            {v}_ptr + {v}_base\n"
        f"            + rm[:, None] * {v}_rs\n"
        f"            + rk[None, :] * {v}_cs,\n"
        f"            mask=mk2, other=0.0)\n"
        for i in (0, 1)
        for m in info["operands"][i]["metas"]
        for v in [_dvar(i, m)]
    )
    vloads = "".join(
        f"        {v}_t = tl.load(\n"
        f"            {v}_ptr + {v}_base\n"
        f"            + rk[:, None] * {v}_rs\n"
        f"            + rn[None, :] * {v}_cs,\n"
        f"            mask=mkv, other=0.0)\n"
        for m in info["operands"][2]["metas"]
        for v in [_dvar(2, m)]
    )
    return mloads, vloads


def _decayed_triton_source(info: dict) -> str:
    """Emit the fused ``(s ⊙ d) @ v`` kernel module for a decayed spec.

    One program per ``(batch, BM-by-BN)`` output tile; each walks the
    K axis in ``BK`` chunks accumulating
    ``tl.dot((s·d)_tile, v_tile)`` — the elementwise decay product
    is folded into the contraction's inner loop, so the ``(…,T,K)``
    intermediate is never materialised.  Batch broadcast is
    stride-driven: every operand meta carries an int64 base-offset
    table plus row/col strides, so both broadcast regimes — a
    full-rank ``d`` and a lower-rank/scalar decay read in place —
    run the same kernel; the per-meta ``FB`` constexprs
    specialise metas whose batch strides are all zero (the
    broadcast regime).  The source is a real module — written to
    the kernel cache by the caller.
    """
    ops = info["operands"]
    mloads, vloads = _decayed_loads(info)
    return (
        "import triton\n"
        "import triton.language as tl\n"
        "import triton.language.extra.libdevice as libdevice\n"
        "\n\n@triton.jit\n"
        f"def _gd({_decayed_decls(info)}, out_ptr, T, K, DV"
        f"{_decayed_flags(info)},\n"
        "        BM: tl.constexpr, BN: tl.constexpr,"
        " BK: tl.constexpr):\n"
        "    b = tl.program_id(0).to(tl.int64)\n"
        "    rm = tl.program_id(1).to(tl.int64) * BM"
        " + tl.arange(0, BM)\n"
        "    rn = tl.program_id(2).to(tl.int64) * BN"
        " + tl.arange(0, BN)\n"
        "    mm = rm[:, None] < T\n"
        "    mn = rn[None, :] < DV\n"
        f"{_decayed_bases(info)}"
        "    acc = tl.zeros((BM, BN), dtype=out_ptr.dtype.element_ty)\n"
        "    for k0 in range(0, K, BK):\n"
        "        rk = (k0 + tl.arange(0, BK)).to(tl.int64)\n"
        "        mk2 = mm & (rk[None, :] < K)\n"
        f"{mloads}"
        f"        sd = ({ops[0]['expr']}) * ({ops[1]['expr']})\n"
        "        mkv = (rk[:, None] < K) & mn\n"
        f"{vloads}"
        f"        acc = tl.dot(sd, ({ops[2]['expr']}), acc=acc,\n"
        '                     input_precision="ieee")\n'
        "    tl.store(\n"
        "        out_ptr + b * (T * DV)\n"
        "        + rm[:, None] * DV + rn[None, :],\n"
        "        acc, mask=mm & mn)\n"
    )


def _bcast2(a: tuple, b: tuple) -> tuple:
    """Right-aligned broadcast of two shape tuples (torch rules).

    Pure-Python — this is host-side launch bookkeeping, so it must
    not pay a dispatcher round-trip per call.  Raises
    ``ValueError`` on unbroadcastable pairs, the same contract
    ``torch.broadcast_shapes`` enforces.
    """
    off = abs(len(a) - len(b))
    lo, hi = (a, b) if len(a) <= len(b) else (b, a)
    out = list(hi)
    for i, e in enumerate(lo):
        j = i + off
        if e == 1 or e == out[j]:
            continue
        if out[j] != 1:
            raise ValueError(f"unbroadcastable shapes: {a} vs {b}")
        out[j] = e
    return tuple(out)


def _metas_shape(vals: dict, metas: list) -> tuple:
    """Broadcast shape over an operand's bound meta values."""
    sh: tuple = ()
    for m in metas:
        sh = _bcast2(sh, tuple(vals[m].shape))
    return sh


def _exp_strides(shape: tuple, strides: tuple, target: tuple) -> tuple:
    """Strides of a tensor broadcast into *target* (torch rules).

    Right-aligned: leading target dims the source lacks and dims
    it broadcasts (extent 1) carry stride 0; the rest keep the
    source stride.  Raises ``ValueError`` when *shape* cannot
    broadcast into *target* — the contract ``torch.broadcast_to``
    enforces, checked here in Python so the hot path skips the
    dispatcher entirely (the kernel reads the tensor in place;
    no expanded view is ever materialised).
    """
    off = len(target) - len(shape)
    if off < 0:
        raise ValueError(f"{shape} cannot broadcast into {target}")
    out = []
    for i, e in enumerate(target):
        j = i - off
        if j < 0 or shape[j] == 1:
            out.append(0)
        elif shape[j] == e:
            out.append(strides[j])
        else:
            raise ValueError(f"{shape} cannot broadcast into {target}")
    return tuple(out)


def _decayed_shapes(vals: dict, ops: list) -> tuple:
    """Shape-check the mul/matmul contract; return launch dims.

    ``(m_sh, v_sh, ob, T, K, DV)`` — the broadcast mul shape, the
    v-operand shape, the output batch, and the tile extents.  The
    contraction requires ``v_sh[-2] == K`` and both ranks ≥ 2 — the
    same bound the op's shape rule declares; violations raise
    ``ValueError``, they never degrade.
    """
    m_sh = _bcast2(
        _metas_shape(vals, ops[0]["metas"]),
        _metas_shape(vals, ops[1]["metas"]),
    )
    v_sh = _metas_shape(vals, ops[2]["metas"])
    if len(m_sh) < 2 or len(v_sh) < 2 or v_sh[-2] != m_sh[-1]:
        raise ValueError(
            f"decayed operands out of contract:"
            f" (s·d)→{tuple(m_sh)}, v→{tuple(v_sh)}"
        )
    ob = _bcast2(m_sh[:-2], v_sh[:-2])
    return m_sh, v_sh, ob, m_sh[-2], m_sh[-1], v_sh[-1]


def _batch_offsets(strides: tuple, bshape: tuple, device: Any) -> Any:
    """int64 flat-batch base offsets for a non-uniform operand.

    The broadcast-expanded operand's batch strides index the flat
    batch axis: entry ``i`` is the element offset of the flat
    index's row-major digit decomposition over *bshape*.  Call only
    when some batch stride is nonzero — the caller serves uniform
    operands a cached dummy (the kernel's ``F`` constexpr makes the
    load dead code, so the contents are never read).
    """
    import torch

    nb = math.prod(bshape)
    idx = torch.arange(nb, dtype=torch.int64, device=device)
    offs = torch.zeros(nb, dtype=torch.int64, device=device)
    div = nb
    for sz, st in zip(bshape, strides, strict=True):
        div //= sz
        offs = offs + (idx // div % sz) * st
    return offs


#: Cap on the per-binding launch-plan memo — a binding sees a
#: handful of distinct call-shapes in practice; the bound keeps a
#: pathological shape-varying caller from growing the dict.
_DECAY_MEMO_CAP = 64


def _decayed_plan(
    vals: dict,
    ops: list,
    ob: tuple,
    m_sh: tuple,
    v_sh: tuple,
    dt: Any,
    zeros: Any,
) -> tuple:
    """Broadcast every meta to its role target; emit launch slots.

    Each meta contributes a ``(batch_offsets, row_stride,
    col_stride)`` slot — the tensor itself is bound per call:
    s/d metas broadcast into ``ob + (T, K)``, v metas into
    ``ob + (K, DV)``.  Broadcast dims carry stride 0 — a
    ``(T, K)`` decay against ``(B, T, K)`` scores is read in
    place, never materialised.  *flags* maps each meta's ``FB``
    constexpr — set when its batch strides are all zero (the
    broadcast regime); those metas bind the cached *zeros* dummy
    because the flag makes the kernel's offset load dead code.
    """
    slots: list = []
    flags: dict[str, bool] = {}
    for i, o in enumerate(ops):
        target = tuple(ob) + (m_sh[-2:] if i < 2 else v_sh[-2:])
        for m in o["metas"]:
            t = vals[m]
            if t.dtype != dt:
                t = t.to(dt)
            est = _exp_strides(
                tuple(t.shape), tuple(t.stride()), target
            )
            bst = est[: len(ob)]
            uni = all(st == 0 for st in bst)
            bo = (
                zeros(t.device)
                if uni
                else _batch_offsets(bst, tuple(ob), t.device)
            )
            flags[f"{_dvar(i, m)}_FB"] = uni
            slots.append((bo, est[-2], est[-1]))
    return slots, flags


def _decayed_binding(kern: Any, info: dict) -> Any:
    """Host wrapper for the fused decayed kernel.

    ``fn(*vals)`` binds the spec metavars positionally (the
    handler's ``args`` order), checks the broadcast/contraction
    contract, and launches one generated kernel — the spelled
    ``(s·d)@v`` never materialises its ``(…,T,K)`` intermediate.
    Shape/dtype-derived launch bookkeeping is memoised per call
    signature (the tensors are bound fresh); non-CUDA tensors and
    unsupported rank/contraction shapes raise, like the scan
    tier: a wrong binding fails loudly, never silently reverts
    to another function.
    """
    ops = info["operands"]
    params = info["params"]
    order = [(i, m) for i, o in enumerate(ops) for m in o["metas"]]
    dummies: dict = {}
    memo: dict = {}

    def _zeros(device: Any) -> Any:
        import torch

        t = dummies.get(device)
        if t is None:
            t = torch.zeros(1, dtype=torch.int64, device=device)
            dummies[device] = t
        return t

    def _plan(vals: dict, key: tuple) -> tuple:
        """Launch plan for a call signature, memoised by *key*."""
        plan = memo.get(key)
        if plan is not None:
            return plan
        m_sh, v_sh, ob, tt, kk, dv = _decayed_shapes(vals, ops)
        dt = vals[params[0]].dtype
        if any(vals[m].dtype != dt for m in params[1:]):
            dt = _scan_dtype(vals, params)
        out_sh = (*tuple(ob), tt, dv)
        nb = math.prod(ob)
        if nb * tt * dv == 0 or kk == 0:
            plan = (None, None, out_sh, nb, tt, kk, dv, dt)
        else:
            slots, flags = _decayed_plan(
                vals, ops, tuple(ob), m_sh, v_sh, dt, _zeros
            )
            plan = (slots, flags, out_sh, nb, tt, kk, dv, dt)
        if len(memo) >= _DECAY_MEMO_CAP:
            memo.clear()
        memo[key] = plan
        return plan

    def fn(*args: Any, **_attrs: Any) -> Any:
        import torch

        vals = dict(zip(params, args, strict=True))
        _decayed_check(vals, params)
        slots, flags, out_sh, nb, tt, kk, dv, dt = _plan(
            vals, _decayed_key(vals, params)
        )
        device = vals[params[0]].device
        if slots is None:
            return torch.zeros(out_sh, dtype=dt, device=device)
        launch = _decayed_tensors(vals, order, slots, dt)
        out = torch.empty((nb, tt, dv), dtype=dt, device=device)
        kern[(nb, -(-tt // _DECAY_BM), -(-dv // _DECAY_BN))](
            *launch,
            out,
            tt,
            kk,
            dv,
            **flags,
            BM=_DECAY_BM,
            BN=_DECAY_BN,
            BK=_DECAY_BK,
        )
        return out.view(out_sh)

    return fn


def _decayed_check(vals: dict, params: list) -> None:
    """Raise on non-CUDA operands — a wrong binding fails loudly."""
    if not all(vals[m].is_cuda for m in params):
        raise RuntimeError(
            "generated decayed kernels need CUDA tensors"
        )


def _decayed_key(vals: dict, params: list) -> tuple:
    """Call-signature key for the launch-plan memo."""
    return tuple(
        (
            tuple(vals[m].shape),
            tuple(vals[m].stride()),
            vals[m].dtype,
            vals[m].device,
        )
        for m in params
    )


def _decayed_tensors(
    vals: dict, order: list, slots: list, dt: Any
) -> list:
    """Interleave per-call tensors with their plan slots.

    Each slot's ``(batch_offsets, row_stride, col_stride)`` comes
    from the memoised plan; the tensor itself is bound fresh —
    cast to the plan dtype when the site mixes dtypes.
    """
    launch: list = []
    for (bo, rs, cs), (_, m) in zip(slots, order, strict=True):
        t = vals[m]
        if t.dtype != dt:
            t = t.to(dt)
        launch += [t, bo, rs, cs]
    return launch


def decayed_triton_bindings(handlers: dict) -> dict:
    """Generate a fused Triton kernel per decayed-contraction handler.

    The third codegen tier: each handler whose *pattern*
    :func:`decayed_spec` recognises gets a binding whose callable
    runs a generated fused ``(s·d)@v`` matmul — real ``triton.jit``
    source on disk under the kernel cache.  Unrecognised or
    un-generatable specs are skipped — the handler keeps its torch
    path, exactly as the other tiers decline.
    """
    out: dict[str, Any] = {}
    for h in handlers.values():
        info = decayed_spec(h["pattern"])
        if info is None:
            continue
        src = _decayed_triton_source(info)
        path = _kernel_cache() / f"_gd_{h['kernel']}.py"
        path.write_text(src)
        kern = _load_generated(path, "_gd")
        if kern is None:
            continue  # module load declined — keep the torch path
        out[h["kernel"]] = _decayed_binding(kern, info)
    return out


def decayed_handlers(
    terms: list, *, covered: set | None = None
) -> dict:
    """Mint handler entries for ``om_decayed``/spelled decay sites.

    The decayed-tier analogue of :func:`scan_handlers`:
    ``decayed_<i>`` entries ``{pattern, kernel, args}`` for each
    term whose spec the fused-contraction codegen recognises —
    ``("om_decayed", s, d, v)`` or the spelled
    ``("matmul", ("mul", s, d), v)`` it lifts from.  The minted
    ``decayed_<i>_k`` op's declared meaning is the element
    semantics verbatim — the same function the ambient
    ``om_decayed`` executor spells — so minting needs no new
    semantics check beyond the op's own well-typedness gate.
    """
    from mlopt_discovery.meta_arena import _spec_of

    skip = covered if covered is not None else set()
    out: dict[str, dict] = {}
    for term in terms:
        pattern = (
            term
            if isinstance(term, (tuple, list))
            else _spec_of(term, {})
        )
        if decayed_spec(pattern) is None:
            continue
        # equality, not ``in`` — attr dicts make canon unhashable
        if _canon_taken(pattern, out, skip):
            continue
        tag = f"decayed_{len(out)}"
        out[tag] = {
            "pattern": pattern,
            "kernel": f"{tag}_k",
            "args": tuple(_spec_metas(pattern)),
        }
    return out


#: ``mlopt_core.opmeta.REDUCE_DIM_OPS`` plus ``softmax`` — the
#: reduction-bodied heads the reduction tier may claim.  Lazily
#: resolved: the opmeta table is the single source for the
#: reduce-dim vocabulary, and ``softmax`` is the one reduction
#: spelled as an ACTIVATION there (its torch binding reduces over
#: ``dim`` exactly like the others).
_REDUCE_HEADS: frozenset | None = None


def reduce_heads() -> frozenset:
    """Return the reduction op heads the reduction tier recognises."""
    global _REDUCE_HEADS
    if _REDUCE_HEADS is None:
        from mlopt_core.opmeta import REDUCE_DIM_OPS

        _REDUCE_HEADS = REDUCE_DIM_OPS | {"softmax"}
    return _REDUCE_HEADS


def reduction_spec(spec: Any) -> bool:
    """Check a spec is a reduction-bodied claim the tier may mint.

    The grammar is deliberately narrow — the *pointwise* tier's own
    boundary rule one level up: the head is a reduction op
    (:func:`reduce_heads`), every OTHER op in the spec is elementwise
    (the fused pre-reduction body), and there are at least two ops
    (a bare ``sum(X)`` claims nothing the ambient op does not already
    spell).  Nested reductions, reshapes, contractions and view ops
    inside the body decline — the spelled binding evaluates them
    honestly but the claim would fuse nothing.
    """
    if not (
        isinstance(spec, (tuple, list))
        and spec
        and isinstance(spec[0], str)
        and spec[0] in reduce_heads()
    ):
        return False
    ops = _spec_ops(spec)
    if len(ops) < 2:
        return False
    return ops - {spec[0]} <= ELEMENTWISE


def reduction_handlers(
    terms: list, *, covered: set | None = None
) -> dict:
    """Mint handler entries for reduction-bodied fusion specs.

    The reduction-tier analogue of :func:`gen_handlers`:
    ``red_<i>`` entries ``{pattern, kernel, args}`` for each spec
    :func:`reduction_spec` accepts — ``("mean", ("mul", "X1",
    ("relu", "X1")), {"dim": (-1,), "keepdim": True})`` and kin.
    The minted ``red_<i>_k`` op's declared meaning is the spelled
    body verbatim — the same function the ambient reduction binding
    computes — so minting needs no new semantics check.  Delivery
    is the tier's Triton piece where :func:`reduction_triton_spec`
    recognises the spec (the ``_gr`` kernel over the elementwise
    body) and the compiled spelled body elsewhere; the binding
    itself falls back to spelled eval on shapes outside the
    call-time subset, so the referee prices exactly what ships.
    """
    from mlopt_discovery.meta_arena import _spec_of

    skip = covered if covered is not None else set()
    out: dict[str, dict] = {}
    for term in terms:
        pattern = (
            term
            if isinstance(term, (tuple, list))
            else _spec_of(term, {})
        )
        if not reduction_spec(pattern):
            continue
        # ``_canon_taken``, not ``canon in skip``: attr-carrying specs
        # canon to dicts — unhashable — and ``in`` raises where
        # equality does not (the ``_canon_taken`` contract).
        if _canon_taken(pattern, out, skip):
            continue
        tag = f"red_{len(out)}"
        out[tag] = {
            "pattern": pattern,
            "kernel": f"{tag}_k",
            "args": tuple(_spec_metas(pattern)),
        }
    return out


# ---------------------------------------------------------------------------
#  The reduction Triton piece — generated kernels for reduce-dim bodies
# ---------------------------------------------------------------------------

#: Reduce heads whose Triton lowering returns a plain tensor — the
#: honest subset of :func:`reduce_heads`.  ``max``/``min`` return
#: (values, indices) pairs, ``std``/``var``/``prod``/``median``/… need
#: semantics this kernel does not spell; they keep the compiled
#: spelled delivery.
_REDUCTION_TRITON_OPS: frozenset = frozenset(
    {"sum", "mean", "amax", "amin", "softmax"}
)

#: Kernel-local names a reduction-spec metavar must not reuse — the
#: generated source binds ``rows``/``pid*``/``cols``/``acc``/``mx``/
#: ``ss``/``__r*``/``__c*``/``__o*`` locals, so a colliding meta would
#: silently shadow them.
_REDUCTION_LOCALS: frozenset = _TRITON_LOCALS | frozenset(
    {
        "row",
        "rows",
        "pid0",
        "pid1",
        "ci",
        "c0",
        "s",
        "cols",
        "cols64",
        "cmask",
        "mask",
        "base",
        "mx",
        "mxv",
        "ss",
        "ssv",
        "acc",
        "out",
        "v",
        "vv",
        "e",
        "d",
        "j",
    }
)


def _red_meta_ok(name: Any) -> bool:
    """Check *name* is safe as a generated reduction-kernel variable."""
    return (
        isinstance(name, str)
        and name.isidentifier()
        and not keyword.iskeyword(name)
        and not name.endswith("_ptr")
        and not name.startswith("__")
        and name not in _REDUCTION_LOCALS
    )


def _red_permutation(rank: int, red: tuple) -> tuple:
    """Permute ``range(rank)`` to ``[non-red dims] + [red dims]``.

    The kernel's index space is outer-then-reduced; *perm* maps each
    permuted position to the broadcast dim it enumerates, so the
    reduce set need not be trailing — ``sum(dim=2)`` of a rank-4
    tensor permutes ``(0, 1, 3, 2)``.
    """
    rset = set(red)
    return tuple(d for d in range(rank) if d not in rset) + tuple(
        d for d in range(rank) if d in rset
    )


def reduction_triton_spec(spec: Any) -> dict | None:
    """Recognise a reduction spec the Triton piece can lower.

    The grammar is :func:`reduction_spec`'s, narrowed to what one
    generated kernel spells: head in :data:`_REDUCTION_TRITON_OPS`,
    exactly one elementwise body child (no hoisted subterms — every
    non-elementwise leaf stays a bound tensor operand), attr dict
    keys ⊆ ``{dim, axis, keepdim}``.  The *shape* subset — CUDA
    floating operands, a ``dim`` attr that normalises against the
    broadcast rank — is call-time knowledge the binding checks per
    call and falls back to spelled eval on; this recogniser only
    gates the spec grammar.  ``None`` declines.
    """
    if not (
        isinstance(spec, (tuple, list))
        and spec
        and isinstance(spec[0], str)
        and spec[0] in _REDUCTION_TRITON_OPS
    ):
        return None
    kids = [e for e in spec[1:] if not isinstance(e, dict)]
    attrs = [e for e in spec[1:] if isinstance(e, dict)]
    if len(kids) != 1 or len(attrs) > 1:
        return None
    attr = attrs[0] if attrs else {}
    if not set(attr) <= {"dim", "axis", "keepdim"}:
        return None
    child = kids[0]
    if not (
        isinstance(child, (tuple, list))
        and child
        and _spec_ops(child) <= ELEMENTWISE
    ):
        return None
    metas = _spec_metas(spec)
    if not metas or any(not _red_meta_ok(m) for m in metas):
        return None
    # scalar const leaves: the v2 kernels evaluate the body on
    # ``[BM, BR]`` tiles, and a ``()``-shaped ``tl.full`` broadcasts
    # to any tile rank while keeping the ``libdevice`` operand dtype
    # the ``[BLOCK]``-shaped template provided (see
    # ``_TRITON_CONST_1D``).
    const = "tl.full((), {v}, dtype=out_ptr.dtype.element_ty)"
    body = _triton_expr(child, const)
    if body is None:
        return None
    dim = attr.get("dim", attr.get("axis"))
    return {
        "op": spec[0],
        "body": body,
        "metas": metas,
        "params": metas,
        "dim": dim,
        "keepdim": bool(attr.get("keepdim", False)),
    }


def _red_col_offsets(nslots: int) -> str:
    """Emit per-slot column offsets ``__co{i}`` ([BR] int64 vectors).

    ``NRED`` is a launch-time ``tl.constexpr``, so the single-red-dim
    case constant-folds: the column coordinate *is* ``cols64`` and the
    slot's contribution is one stride multiply — the v1 generic path
    paid an int64 ``//``/``%`` reconstruction per red dim per chunk.
    Multi-dim reduces keep the stride-table decode (the column index
    enumerates the red dims row-major, exactly like v1).
    """
    out = "    if NRED == 1:\n"
    for i in range(nslots):
        out += (
            f"        __co{i} = cols64 * tl.load(ometa_ptr + "
            f"{2 + i} * NDIM + NOUT)\n"
        )
    out += "    else:\n"
    for i in range(nslots):
        out += f"        __co{i} = tl.zeros([BR], dtype=tl.int64)\n"
    out += (
        "        for j in tl.static_range(NRED):\n"
        "            __csz = tl.load(ometa_ptr + NOUT + j)\n"
        "            __cst = tl.load(ometa_ptr + NDIM + NOUT + j)\n"
        "            __cco = (cols64 // __cst) % __csz\n"
    )
    out += "".join(
        f"            __co{i} += __cco * tl.load(ometa_ptr + "
        f"{2 + i} * NDIM + NOUT + j)\n"
        for i in range(nslots)
    )
    return out


def _red_prologue(nslots: int) -> str:
    """Emit the tiled prologue every v2 kernel shares.

    ``rows`` enumerates outer-index blocks of ``BM`` and ``cols`` the
    program's reduce-axis chunk (``pid1 * BR``); ``__ro{i}`` /
    ``__co{i}`` are the per-slot row/column offset vectors and
    ``__o{i}`` their broadcast sum — the ``[BM, BR]`` element
    offsets.  The row decode runs once per *block* of rows (the v1
    kernel decoded one scalar row per program over a serial chunk
    loop), and the column decode runs once per chunk — neither is
    per-element.
    """
    out = (
        "    pid0 = tl.program_id(0)\n"
        "    pid1 = tl.program_id(1)\n"
        "    rows = (pid0 * BM + tl.arange(0, BM)).to(tl.int64)\n"
        "    rows32 = pid0 * BM + tl.arange(0, BM)\n"
    )
    # int32 decode: ``rows // __ost % __sz`` is a software division
    # sequence — int64 costs ~2x int32 and this runs per outer dim
    # per row-block-element.  The binding guarantees every shape dim,
    # index-space stride and operand stride fits int32 (total element
    # count and strides < 2**31), so int32 decode is exact.
    out += "".join(
        f"    __ro{i} = tl.zeros([BM], dtype=tl.int32)\n"
        for i in range(nslots)
    )
    out += (
        "    for d in tl.static_range(NOUT):\n"
        "        __sz = tl.load(ometa_ptr + d).to(tl.int32)\n"
        "        __ost = tl.load(ometa_ptr + NDIM + d).to(tl.int32)\n"
        "        __cd = (rows32 // __ost) % __sz\n"
    )
    out += "".join(
        f"        __ro{i} += __cd * tl.load("
        f"ometa_ptr + {2 + i} * NDIM + d).to(tl.int32)\n"
        for i in range(nslots)
    )
    out += (
        "    c0 = pid1 * BR\n"
        "    cols = c0 + tl.arange(0, BR)\n"
        "    cols64 = cols.to(tl.int64)\n"
        "    cmask = cols < R\n"
    )
    out += _red_col_offsets(nslots)
    out += "".join(
        f"    __o{i} = __ro{i}[:, None].to(tl.int64) + __co{i}[None, :]\n"
        for i in range(nslots)
    )
    out += "    mask = (rows < M)[:, None] & cmask[None, :]\n"
    return out


def _red_loads(metas: list) -> str:
    """Emit the masked operand loads for a ``[BM, BR]`` tile.

    Loads convert to the output dtype — the host's promote-types
    result — so mixed-dtype sites compile like the spelled broadcast
    does (the v1 contract, unchanged).
    """
    return "".join(
        f"    {m} = tl.load({m}_ptr + __o{i}, mask=mask, other=0.0)"
        f".to(out_ptr.dtype.element_ty)\n"
        for i, m in enumerate(metas)
    )


def _red_kernel_head() -> str:
    """Emit the import/``@triton.jit`` preamble shared by all kernels."""
    return (
        "import triton\n"
        "import triton.language as tl\n"
        "import triton.language.extra.libdevice as libdevice\n\n\n"
    )


#: Constexpr tail shared by every tile kernel — ``BM``/``BR`` are the
#: row/column tile extents, ``NRED`` the reduce-dim count.
_RED_CARGS = (
    " NDIM: tl.constexpr, NOUT: tl.constexpr, NRED: tl.constexpr,"
    " BM: tl.constexpr, BR: tl.constexpr"
)


def _red_reduce_source(info: dict) -> str:
    """Emit ``_gr`` (tile reduce) + ``_gc`` (partial combine).

    ``_gr`` is a ``(row-blocks, chunks)`` grid: each program reduces
    one ``[BM, BR]`` tile — ``BM`` outer rows by one ``BR`` column
    chunk — and either stores the final per-row result (``DIRECT``,
    the ``NC == 1`` case) or a per-chunk partial at
    ``rows * NC + pid1``.  ``_gc`` folds the ``NC`` partials per row
    (a ``mean`` divides by ``RF`` here, once).  Occupancy comes from
    tiling *both* axes: the v1 kernel ran one program per outer index
    with a serial chunk loop, so small outer counts left the SMs
    idle and tiny reduce axes paid full per-program cost per row.
    """
    metas = info["metas"]
    op = info["op"]
    body = info["body"]
    nops = len(metas)
    ptrs = ", ".join(f"{m}_ptr" for m in metas)
    dt = "out_ptr.dtype.element_ty"
    if op in ("sum", "mean"):
        init = f"tl.zeros([BM], dtype={dt})"
        initc = f"tl.zeros([BMC], dtype={dt})"
        ident = "0.0"
        identc = "0.0"
        accum = "acc += tl.sum(vv, axis=1)"
        comb = "acc += v"
        final = "acc" if op == "sum" else "acc / RF"
    elif op == "amax":
        init = f'tl.full([BM], -float("inf"), dtype={dt})'
        initc = f'tl.full([BMC], -float("inf"), dtype={dt})'
        ident = '-float("inf")'
        identc = '-float("inf")'
        accum = "acc = tl.maximum(acc, tl.max(vv, axis=1))"
        comb = "acc = tl.maximum(acc, v)"
        final = "acc"
    else:  # amin
        init = f'tl.full([BM], float("inf"), dtype={dt})'
        initc = f'tl.full([BMC], float("inf"), dtype={dt})'
        ident = 'float("inf")'
        identc = 'float("inf")'
        accum = "acc = tl.minimum(acc, tl.min(vv, axis=1))"
        comb = "acc = tl.minimum(acc, v)"
        final = "acc"
    src = _red_kernel_head()
    src += (
        "@triton.jit\n"
        f"def _gr({ptrs}, out_ptr, ometa_ptr, M, R, NC, RF,{_RED_CARGS}"
        ", DIRECT: tl.constexpr):\n"
        + _red_prologue(nops)
        + f"    acc = {init}\n"
        + _red_loads(metas)
        + f"    v = ({body})\n"
        + f"    vv = tl.where(mask, v, {ident})\n"
        + f"    {accum}\n"
        + "    if DIRECT:\n"
        + f"        tl.store(out_ptr + rows, {final}, mask=rows < M)\n"
        + "    else:\n"
        + "        tl.store(out_ptr + rows * NC + pid1, acc,"
        + " mask=rows < M)\n\n"
    )
    src += (
        "@triton.jit\n"
        "def _gc(out_ptr, p_ptr, M, NC, RF, BMC: tl.constexpr):\n"
        "    rows = (tl.program_id(0) * BMC + tl.arange(0, BMC))"
        ".to(tl.int64)\n"
        + f"    acc = {initc}\n"
        + "    for s in range(NC):\n"
        + "        v = tl.load(p_ptr + rows * NC + s, mask=rows < M,"
        + f" other={identc})\n"
        + f"        {comb}\n"
        + f"    tl.store(out_ptr + rows, {final}, mask=rows < M)\n"
    )
    return src


def _red_softmax_source(info: dict) -> str:
    """Emit the softmax kernel family.

    ``NC == 1`` (the whole reduce axis fits one ``BR`` chunk — the
    common case): ``_grms`` computes each row's max *and* sum-exp in
    a single pass over one loaded tile, ``_grn`` stores the
    normalised values — two launches, each fully tiled over rows and
    columns.  ``NC > 1``: ``_grm``/``_gcm`` (partial row max +
    combine), ``_grs``/``_gcs`` (partial sum-exp against the combined
    max + combine), ``_grn`` — the split-K form of the same three
    reductions, five launches.
    """
    metas = info["metas"]
    body = info["body"]
    nops = len(metas)
    ptrs = ", ".join(f"{m}_ptr" for m in metas)
    dt = "out_ptr.dtype.element_ty"
    src = _red_kernel_head()
    # fused stats pass — one tile covers the whole row (NC == 1)
    src += (
        "@triton.jit\n"
        f"def _grms({ptrs}, out_ptr, ss_ptr, ometa_ptr, M, R,{_RED_CARGS}"
        "):\n"
        + _red_prologue(nops)
        + _red_loads(metas)
        + f"    v = ({body})\n"
        + '    vv = tl.where(mask, v, -float("inf"))\n'
        + "    mx = tl.max(vv, axis=1)\n"
        + "    e = tl.exp(vv - mx[:, None])\n"
        + "    ss = tl.sum(e, axis=1)\n"
        + "    tl.store(out_ptr + rows, mx, mask=rows < M)\n"
        + "    tl.store(ss_ptr + rows, ss, mask=rows < M)\n\n"
    )
    # normalised store — grid spans every column chunk
    src += (
        "@triton.jit\n"
        f"def _grn({ptrs}, out_ptr, mx_ptr, ss_ptr, ometa_ptr, M, R,"
        f"{_RED_CARGS}):\n"
        + _red_prologue(nops + 1)
        + _red_loads(metas)
        + f"    v = ({body})\n"
        + '    vv = tl.where(mask, v, -float("inf"))\n'
        + "    mxv = tl.load(mx_ptr + rows, mask=rows < M,"
        + " other=0.0)[:, None]\n"
        + "    ssv = tl.load(ss_ptr + rows, mask=rows < M,"
        + " other=0.0)[:, None]\n"
        + "    e = tl.exp(vv - mxv) / ssv\n"
        + f"    tl.store(out_ptr + __o{nops}, e, mask=mask)\n\n"
    )
    # split-K stats: partial max per chunk …
    src += (
        "@triton.jit\n"
        f"def _grm({ptrs}, out_ptr, ometa_ptr, M, R, NC,{_RED_CARGS}"
        "):\n"
        + _red_prologue(nops)
        + f'    acc = tl.full([BM], -float("inf"), dtype={dt})\n'
        + _red_loads(metas)
        + f"    v = ({body})\n"
        + '    vv = tl.where(mask, v, -float("inf"))\n'
        + "    acc = tl.maximum(acc, tl.max(vv, axis=1))\n"
        + "    tl.store(out_ptr + rows * NC + pid1, acc,"
        + " mask=rows < M)\n\n"
    )
    src += (
        "@triton.jit\n"
        "def _gcm(out_ptr, p_ptr, M, NC, BMC: tl.constexpr):\n"
        "    rows = (tl.program_id(0) * BMC + tl.arange(0, BMC))"
        ".to(tl.int64)\n"
        f'    acc = tl.full([BMC], -float("inf"), dtype={dt})\n'
        + "    for s in range(NC):\n"
        + "        v = tl.load(p_ptr + rows * NC + s, mask=rows < M,"
        + ' other=-float("inf"))\n'
        + "        acc = tl.maximum(acc, v)\n"
        + "    tl.store(out_ptr + rows, acc, mask=rows < M)\n\n"
    )
    # … combine to the row max, then partial sum-exp per chunk …
    src += (
        "@triton.jit\n"
        f"def _grs({ptrs}, out_ptr, mx_ptr, ometa_ptr, M, R, NC,"
        f"{_RED_CARGS}):\n"
        + _red_prologue(nops)
        + "    mxv = tl.load(mx_ptr + rows, mask=rows < M,"
        + " other=0.0)[:, None]\n"
        + f"    acc = tl.zeros([BM], dtype={dt})\n"
        + _red_loads(metas)
        + f"    v = ({body})\n"
        + '    vv = tl.where(mask, v, -float("inf"))\n'
        + "    e = tl.exp(vv - mxv)\n"
        + "    acc += tl.sum(e, axis=1)\n"
        + "    tl.store(out_ptr + rows * NC + pid1, acc,"
        + " mask=rows < M)\n\n"
    )
    src += (
        "@triton.jit\n"
        "def _gcs(out_ptr, p_ptr, M, NC, BMC: tl.constexpr):\n"
        "    rows = (tl.program_id(0) * BMC + tl.arange(0, BMC))"
        ".to(tl.int64)\n"
        f"    acc = tl.zeros([BMC], dtype={dt})\n"
        + "    for s in range(NC):\n"
        + "        v = tl.load(p_ptr + rows * NC + s, mask=rows < M,"
        + " other=0.0)\n"
        + "        acc += v\n"
        + "    tl.store(out_ptr + rows, acc, mask=rows < M)\n"
    )
    return src


def _reduction_triton_source(info: dict) -> str:
    """Emit the generated reduction kernel module for a spec.

    v2: a ``(row-block, column-chunk)`` tiled grid with cross-program
    partials — see :func:`_red_reduce_source` /
    :func:`_red_softmax_source`.  Arbitrary reduce dims still ride
    the ``_red_permutation``/``ometa`` stride machinery; only the
    emitted kernel changed, not the spec grammar or the binding's
    honest spelled fallback.
    """
    if info["op"] == "softmax":
        return _red_softmax_source(info)
    return _red_reduce_source(info)


def _red_dims(info: dict, rank: int) -> tuple | None:
    """Normalise the spec's ``dim`` attr against broadcast *rank*.

    Returns the sorted reduce-dim tuple, or ``None`` when the attr
    cannot denote a reduction of this rank (out-of-range or
    duplicate dims — the spelled eval raises on those, so the
    binding defers to it).  ``None`` *dim* means the spelled
    dim-less form: a full reduce for the reduce ops, last-axis for
    ``softmax`` (its binding default).
    """
    dim = info["dim"]
    if dim is None:
        seq = (-1,) if info["op"] == "softmax" else None
    elif type(dim) is int:
        seq = (dim,)
    elif isinstance(dim, (tuple, list)):
        if not all(type(d) is int for d in dim):
            return None
        seq = tuple(dim)
    else:
        return None
    if seq is None:
        return tuple(range(rank)) if rank else None
    if not seq or rank == 0:
        return None
    out = []
    for d in seq:
        if d < 0:
            d += rank
        if not 0 <= d < rank:
            return None
        out.append(d)
    if len(set(out)) != len(out):
        return None
    return tuple(sorted(out))


def _red_dtype(tensors: list) -> Any:
    """Promoted dtype over the bound operand tensors."""
    import torch

    dt = tensors[0].dtype
    for t in tensors[1:]:
        dt = torch.promote_types(dt, t.dtype)
    return dt


#: Reduction ometa cache — same posture as ``_TRITON_OMETA``: the
#: buffer is a pure function of the broadcast layout, and rebuilding
#: it per call costs a pageable H2D copy (~14µs measured) that small
#: reductions cannot afford.  Bounded like the pointwise cache.
_RED_OMETA: dict = {}
_RED_OMETA_MAX = 256


def _red_ometa(
    tensors: list,
    shape: tuple,
    perm: tuple,
    nout: int,
    device: Any,
    store: bool = False,
) -> Any:
    """Pack the reduction kernel's stride/broadcast metadata.

    Everything is in *perm* order — ``[non-red dims] + [red dims]``,
    so the reduce set need not be trailing.  Layout: permuted
    ``shape`` sizes (K), permuted per-dim *index* strides (``prod``
    over later dims of the same index space — outer dims for
    ``p < nout``, red dims after), then one K-stride slot per
    operand (effective element strides, 0 on absent/broadcast dims)
    and — *store* — a final slot of the output's row-major strides
    for the softmax store offset.  Cached per layout — see
    ``_RED_OMETA``.
    """
    import torch

    key = (
        shape,
        perm,
        nout,
        store,
        str(device),
        tuple((tuple(t.shape), tuple(t.stride())) for t in tensors),
    )
    om = _RED_OMETA.get(key)
    if om is not None:
        return om
    k = len(shape)
    pshape = tuple(shape[d] for d in perm)
    # original-order row-major strides — the store slot's strides
    ss: list = []
    acc = 1
    for s in reversed(shape):
        ss.insert(0, acc)
        acc *= s
    # permuted-order index strides: within each index space
    pss: list = []
    acc = 1
    for s in reversed(pshape):
        pss.insert(0, acc)
        acc *= s
    buf = list(pshape)
    for p in range(k):
        if p < nout:
            q = 1
            for e in range(p + 1, nout):
                q *= pshape[e]
            buf.append(q)
        else:
            # every permuted dim after a red dim is red, so the
            # row-major stride IS the column-space stride
            buf.append(pss[p])
    for t in tensors:
        kt = t.dim()
        for g in perm:
            j = g - (k - kt)
            if j < 0 or t.shape[j] == 1:
                buf.append(0)
            else:
                buf.append(t.stride(j))
    if store:
        buf.extend(ss[g] for g in perm)
    om = torch.tensor(buf, dtype=torch.int64, device=device)
    if len(_RED_OMETA) >= _RED_OMETA_MAX:
        _RED_OMETA.clear()
    _RED_OMETA[key] = om
    return om


def _red_out_shape(info: dict, shape: tuple, red: tuple) -> tuple:
    """Return the spelled output shape for the call's broadcast.

    Non-trailing reduce dims remove/keepdim *in place* — the flat
    ``row`` index still enumerates the surviving dims in original
    order, so ``view`` lands correctly either way.
    """
    if info["op"] == "softmax":
        return tuple(shape)
    rset = set(red)
    if info["keepdim"]:
        return tuple(1 if d in rset else s for d, s in enumerate(shape))
    return tuple(s for d, s in enumerate(shape) if d not in rset)


def _red_spelled(spec: Any, vals: dict) -> Any:
    """Evaluate the spelled body — the binding's honest fallback."""
    import mlopt_torch.torch_bridge as tb

    return _eval_spec(spec, vals, tb._IR_TO_TORCH)


def _red_next_pow2(n: int, cap: int) -> int:
    """Smallest power of two ≥ *n*, capped at *cap*."""
    p = 1
    while p < n and p < cap:
        p *= 2
    return p


def _red_tile_dims(m_rows: int, r_cols: int) -> tuple:
    """Pick ``(BR, NC, BM)`` for the tiled reduction kernels.

    ``BR`` is the column tile — the smallest power of two covering
    the reduce axis, capped at 256 (``NC == 1`` selects the fused
    direct/stats paths; larger axes chunk into ``NC`` partials).
    ``BM`` is the row tile — a power of two sized so ``ceil(M/BM)``
    lands near ~512 row-blocks while ``BM * BR`` stays ≤ ~2048
    elements per program (register pressure, not occupancy, is the
    binding constraint at these tile sizes — measured on the p32
    probe shapes).
    """
    BR = _red_next_pow2(min(r_cols, 256), 256)
    NC = (r_cols + BR - 1) // BR
    cap = min(64, max(1, 2048 // BR))
    # a program's tile should fill its threads: BM * BR >= ~256
    # elems (a [1, 64] tile idles half of a 4-warp CTA — measured
    # 2x slower than BM >= 2 on the probe shapes)
    lo = 1
    while lo * BR < 256 and lo < cap:
        lo *= 2
    want = max(1, (m_rows + 255) // 256)
    BM = 1
    while min(cap, want) >= BM * 2:
        BM *= 2
    BM = max(BM, min(lo, cap))
    # never tile more rows than exist (masked lanes are free but
    # pointless — and a mostly-masked tile wastes the fill the `lo`
    # rule was buying)
    return BR, NC, min(BM, _red_next_pow2(m_rows, 64))


def _reduction_binding(mod: Any, info: dict, spec: Any) -> Any:
    """Host wrapper for a generated reduction kernel module.

    ``fn(*vals)`` binds the spec metavars positionally, checks the
    call-time subset — CUDA floating tensors, a ``dim`` attr that
    normalises against the broadcast rank — and launches the kernel
    family the module ships (one ``_gr``+``_gc`` pair for the
    pointwise-head ops, ``_grms``/``_grn`` or the split-K softmax
    chain).  Anything outside the subset — or a chunk count whose
    partial buffer would be unreasonable — evaluates the spelled
    body instead: the binding's denotation is the spec verbatim, the
    Triton arm is the implementation where it applies (the
    ``_rms_norm_measured`` posture).
    """
    params = info["params"]
    metas = info["metas"]
    op = info["op"]
    if op == "softmax":
        k_grms = mod._grms
        k_grn = mod._grn
        k_grm = mod._grm
        k_gcm = mod._gcm
        k_grs = mod._grs
        k_gcs = mod._gcs
    else:
        k_gr = mod._gr
        k_gc = mod._gc

    def fn(*args: Any, **_attrs: Any) -> Any:
        import torch

        vals = dict(zip(params, args, strict=True))
        xs = [vals[m] for m in metas]
        if not all(
            torch.is_tensor(t)
            and t.is_cuda
            and t.dtype.is_floating_point
            for t in xs
        ):
            return _red_spelled(spec, vals)
        try:
            shape = torch.broadcast_shapes(
                *[tuple(t.shape) for t in xs]
            )
        except RuntimeError:
            return _red_spelled(spec, vals)
        rank = len(shape)
        red = _red_dims(info, rank)
        if not red:
            return _red_spelled(spec, vals)
        nout = rank - len(red)
        perm = _red_permutation(rank, red)
        # outer index space = non-red dims in original order —
        # ``perm[:nout]`` IS that order, so M/R come from pshape
        pshape = tuple(shape[d] for d in perm)
        m_rows = 1
        for s in pshape[:nout]:
            m_rows *= s
        r_cols = 1
        for s in pshape[nout:]:
            r_cols *= s
        dt = _red_dtype(xs)
        if dt not in (torch.float32, torch.float64):
            return _red_spelled(spec, vals)
        out_shape = _red_out_shape(info, shape, red)
        device = xs[0].device
        if m_rows == 0 or r_cols == 0:
            return torch.empty(out_shape, dtype=dt, device=device)
        # the kernels decode row coordinates in int32 — the largest
        # element offset any operand can address must fit (a tensor
        # can legally carry strides past its own numel via
        # ``as_strided``; those sites keep the spelled arm)
        if m_rows * r_cols >= (1 << 31) or any(
            sum(
                abs((d - 1) * s)
                for d, s in zip(t.shape, t.stride(), strict=True)
            )
            >= (1 << 31)
            for t in xs
        ):
            return _red_spelled(spec, vals)
        br, nc, bm = _red_tile_dims(m_rows, r_cols)
        # the partial buffer is m_rows * nc elements — beyond this
        # the honest spelled eval is the better arm anyway
        if nc > 256 or m_rows * nc > (1 << 22):
            return _red_spelled(spec, vals)
        om = _red_ometa(
            xs, shape, perm, nout, device, store=op == "softmax"
        )
        rowblocks = (m_rows + bm - 1) // bm
        cargs = {
            "NDIM": rank,
            "NOUT": nout,
            "NRED": len(red),
            "BM": bm,
            "BR": br,
        }
        if op == "softmax":
            mx = torch.empty(m_rows, dtype=dt, device=device)
            ss = torch.empty(m_rows, dtype=dt, device=device)
            if nc == 1:
                # one tile covers the whole row — fused stats pass
                k_grms[(rowblocks, 1)](
                    *xs, mx, ss, om, m_rows, r_cols, **cargs
                )
            else:
                pmax = torch.empty(
                    (m_rows, nc), dtype=dt, device=device
                )
                psum = torch.empty(
                    (m_rows, nc), dtype=dt, device=device
                )
                k_grm[(rowblocks, nc)](
                    *xs, pmax, om, m_rows, r_cols, nc, **cargs
                )
                k_gcm[((m_rows + 127) // 128,)](
                    mx, pmax, m_rows, nc, BMC=128
                )
                k_grs[(rowblocks, nc)](
                    *xs, psum, mx, om, m_rows, r_cols, nc, **cargs
                )
                k_gcs[((m_rows + 127) // 128,)](
                    ss, psum, m_rows, nc, BMC=128
                )
            out = torch.empty(m_rows * r_cols, dtype=dt, device=device)
            k_grn[(rowblocks, nc)](
                *xs, out, mx, ss, om, m_rows, r_cols, **cargs
            )
            return out.view(out_shape)
        if nc == 1:
            out = torch.empty(m_rows, dtype=dt, device=device)
            k_gr[(rowblocks, 1)](
                *xs,
                out,
                om,
                m_rows,
                r_cols,
                nc,
                float(r_cols),
                **cargs,
                DIRECT=True,
            )
        else:
            part = torch.empty((m_rows, nc), dtype=dt, device=device)
            k_gr[(rowblocks, nc)](
                *xs,
                part,
                om,
                m_rows,
                r_cols,
                nc,
                float(r_cols),
                **cargs,
                DIRECT=False,
            )
            out = torch.empty(m_rows, dtype=dt, device=device)
            k_gc[((m_rows + 127) // 128,)](
                out, part, m_rows, nc, float(r_cols), BMC=128
            )
        return out.view(out_shape)

    return fn


def reduction_triton_bindings(handlers: dict) -> dict:
    """Generate a Triton kernel per reduction-bodied handler.

    The reduction tier's codegen: each handler whose *pattern*
    :func:`reduction_triton_spec` recognises gets a binding whose
    callable runs a generated fused reduce kernel — one launch for
    the elementwise body plus the reduction, real ``triton.jit``
    source on disk under the kernel cache.  Specs outside the
    grammar are skipped (the handler keeps its ``gen_bindings``
    compiled-spelled path); shapes outside the call-time subset
    fall back to spelled eval inside the binding.
    """
    out: dict[str, Any] = {}
    for h in handlers.values():
        info = reduction_triton_spec(h["pattern"])
        if info is None:
            continue
        src = _reduction_triton_source(info)
        path = _kernel_cache() / f"_gr_{h['kernel']}.py"
        path.write_text(src)
        mod = _load_generated_module(path)
        if mod is None:
            continue  # module load declined — keep the torch path
        need = (
            ("_grms", "_grn", "_grm", "_gcm", "_grs", "_gcs")
            if info["op"] == "softmax"
            else ("_gr", "_gc")
        )
        if any(getattr(mod, n, None) is None for n in need):
            continue
        out[h["kernel"]] = _reduction_binding(mod, info, h["pattern"])
    return out


def decayed_triton_sink(handlers: dict) -> Any:
    """Build a ``TorchSink`` binding minted decayed ops to Triton.

    Same posture as :func:`scan_triton_sink`: ``supported_ops``
    reports the ambient table plus every generated decayed kernel —
    the feasibility bound the board prices decayed claims against —
    and a claim on an unrecognised spec stays unpriced-unlowerable.
    """
    from mlopt_core.ops import OpTable
    from mlopt_torch.adapters import TorchSink

    bindings = _ambient_bindings()
    bindings.update(decayed_triton_bindings(handlers))
    table = OpTable.core().register(torch_bindings=bindings)
    return TorchSink(ops=table)


#: The codegen tiers, in mint order — the registry the claim/referee
#: machinery iterates rather than naming a tier.  ``triton=None``
#: tiers deliver through the compiled spelled body; the referee's
#: ``kernel_for`` hook consults each non-``None`` binding builder.
TIERS: tuple[Tier, ...] = (
    # the gen tier's delivery IS the pointwise Triton codegen —
    # ``hybrid_sink`` binds ``triton_bindings`` over every spec it
    # accepts ahead of the compiled spelled fallback, so the
    # referee must grade that kernel, not the fallback it replaces.
    # A ``None`` here once left the whole tier un-attempted: the
    # referee timed compiled-spelled while delivery shipped Triton.
    Tier("gen", gen_handlers, triton_bindings),
    Tier("scan", scan_handlers, scan_triton_bindings),
    Tier("decayed", decayed_handlers, decayed_triton_bindings),
    # the reduction tier's delivery is its Triton piece where the
    # spec is expressible (a reduce-dim head over an elementwise
    # body) and the compiled spelled body elsewhere — the binding
    # itself falls back to spelled eval on shapes outside the
    # subset, so ``reduction_triton_bindings`` is the honest
    # declaration either way.
    Tier("reduction", reduction_handlers, reduction_triton_bindings),
)

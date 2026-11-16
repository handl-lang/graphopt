"""Term evaluation + IR lowering.

Split from ``torch_bridge.py``.  ``eval_term`` (the single eval-term family), the flattened
``_EvalPlan`` slot tape, the const/dtype mint helpers and
``IRModule`` / ``ir_to_torch_module`` — the torch.nn.Module rebuilt
from a mlopt IR.  The binding tables it dispatches through are
parameters (``OpTable`` / the ambient ``_IR_TO_TORCH``); this module
imports neither the export nor the bindings side.  ``mlopt_torch.
torch_bridge`` re-exports this module — import from there for the
historical surface.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
from mlopt_core.ir import IR, Const, Op, Param, Var
from mlopt_core.opmeta import FOLD_OPS
from mlopt_core.ops import OpTable

from mlopt_torch import dtype_ctx

#: Sentinel for "no default supplied" — distinct from ``None`` so a
#: legitimate ``None`` default (e.g. a zero-arg forward's ``x``) still
#: resolves instead of reading as failure.
_MISS: Any = object()

#: Adapter-declared ops that join ``FOLD_OPS`` at lowering: parameter-
#: only subtrees of these evaluate ONCE at ``IRModule`` construction
#: and materialise as a fused ``Param``.  ``weight_norm`` is the
#: ``parametrizations.weight_norm`` reparameterisation — pure in its
#: (v, g) params, recomputed per-forward by the aten kernel (~5% of
#: Kokoro's decoder eager time across ~70 sites —
#: ``docs/reports/weight_norm_fold.md``).  It cannot join
#: ``mlopt_core.opmeta.FOLD_OPS`` — that frozenset is a snapshot taken
#: before adapter declarations exist — so the lowerer unions it here.
#: Data-dependent instances (a ``Var`` anywhere in the subtree) never
#: reach this fold: ``_uses_input`` gates it exactly like ``FOLD_OPS``.
_EXTRA_FOLD_OPS: frozenset[str] = frozenset({"weight_norm"})


def _randn_param(
    term: Param, like: torch.Tensor | None = None
) -> torch.Tensor:
    """``IRModule._eval``'s unregistered-Param fallback.

    A fresh randn of the declared shape (``None`` dims materialise as
    extent 1).
    """
    shape = tuple(d if d is not None else 1 for d in term.typ.shape)
    # A fallback param is a real operand: mint it where the computation
    # runs.  A 0-dim CPU scalar is not a device context (it is the
    # weak-scalar mint), so it is ignored on purpose.
    if (
        isinstance(like, torch.Tensor)
        and like.is_floating_point()
        and like.dim() > 0
    ):
        return torch.randn(*shape, dtype=like.dtype, device=like.device)
    return torch.randn(*shape)


#: ``_plan_cache`` sentinel for a term that cannot be plan-flattened
#: (a non-term root such as ``42`` — the strict-eval ``TypeError``
#: stays delegated to :func:`eval_term`).
_UNPLANNED: Any = object()


class _EvalPlan:
    """A flattened, slot-indexed eval tape for one term DAG.

    Plan 0003: the recursive :func:`eval_term` walk is the generic
    executor's dispatch floor — per node it pays an isinstance chain,
    a content-hash memo get/set, a ``bindings`` lookup, an args list
    build and a ``dict(term.attrs)`` copy.  The plan replaces all of
    it with a linear tape evaluated in one loop:

    * **Slots** — every distinct subterm (DAG-shared nodes once, keyed
      on the interned term object) owns one slot in a per-call
      ``vals`` list; operand references are slot indices, so no
      recursion and no memo dict ever runs.
    * **Leaves** — ``var_fills`` resolve ``Var`` nodes by name through
      the eval env (``env.get(name, x)``) or positionally from the
      forward args (``xs[pos] if pos < len(xs) else x`` — the same
      value, since ``forward``'s env only ever maps input names plus
      ``"self"``); ``param_fills`` re-read ``param_map.get(name)``
      EVERY call (``load_state_dict`` visibility) with the
      ``_randn_param`` fallback for unregistered Params;
      ``const_fills`` re-mint per call — float values at the call's
      operand dtype (the input, else a floating Param, else
      the ambient default) so a Const stays a weak scalar: an
      ambient-fp64 process default must not promote an fp32
      delivery.  Caching the tensor would freeze the build-time
      dtype; minting at the operand dtype keeps it a no-promotion
      scalar like spelled torch.
    * **Steps** — ``(out_slot, op_idx, arg_slots, attrs)``; ``op_idx``
      indexes ``op_names``, whose bindings are resolved through the
      module's op table per call — post-construction
      ``_IR_TO_TORCH[op] = fn`` overrides keep reaching already-built
      modules, and a deleted binding still raises the strict
      ``ValueError`` at eval.
    * **Releases** — ``releases[k]`` lists the arg slots whose LAST
      consumer is step ``k`` (use-count over the tape, topological
      order makes the last consumer deterministic).  After step ``k``
      runs, those ``vals`` entries are dropped: intermediates die at
      last use instead of pinning every slot tensor until the root
      returns — the difference between peak memory scaling with the
      DAG's *live set* and with its *size* (the whisper-small fp32
      encoder OOM in ``docs/reports/whisper_experiment.md``).  DAG
      sharing is handled structurally: a shared subterm's slot has
      use-count > 1 and survives until its final consumer.  Under
      ``enable_grad`` autograd's saved-tensor refs keep whatever
      backward needs regardless — the release only drops the tape's
      own reference.

    Plans are immutable once built and shared via the module's
    ``_plan_cache`` — safe under concurrent forwards (two racers
    build equal plans).
    """

    __slots__ = (
        "const_fills",
        "n_slots",
        "op_names",
        "param_fills",
        "releases",
        "root_slot",
        "steps",
        "var_fills",
    )

    def __init__(
        self,
        n_slots: int,
        var_fills: tuple,
        param_fills: tuple,
        const_fills: tuple,
        steps: tuple,
        op_names: tuple,
        root_slot: int,
        releases: tuple,
    ) -> None:
        self.n_slots = n_slots
        self.var_fills = var_fills
        self.param_fills = param_fills
        self.const_fills = const_fills
        self.steps = steps
        self.op_names = op_names
        self.root_slot = root_slot
        self.releases = releases

    @staticmethod
    def build(term: Any, input_pos: dict[str, int]) -> _EvalPlan | None:
        """Flatten ``term`` into a slot tape; ``None`` when unplannable.

        Iterative post-order (scan-domain prefix chains nest O(T)
        deep — past the recursion limit on long sequences).  A node
        that is not a ``Var``/``Param``/``Const``/``Op`` (a stray
        metavariable, a bare ``42`` root) makes the whole term
        unplannable — the caller falls back to ``eval_term``'s strict
        ``TypeError`` path.
        """
        slot_of: dict[Any, int] = {}
        var_fills: list[tuple[int, str, int]] = []
        param_fills: list[tuple[int, str, Any]] = []
        const_fills: list[tuple[int, Any]] = []
        steps: list[tuple[int, int, tuple, Any]] = []
        op_names: list[str] = []
        op_index: dict[str, int] = {}
        n = 0
        stack: list[tuple[Any, bool]] = [(term, False)]
        while stack:
            t, done = stack.pop()
            if done:
                s = n
                n += 1
                oi = op_index.get(t.op)
                if oi is None:
                    oi = len(op_names)
                    op_index[t.op] = oi
                    op_names.append(t.op)
                steps.append(
                    (
                        s,
                        oi,
                        tuple(slot_of[a] for a in t.args),
                        t.attrs or None,
                    )
                )
                slot_of[t] = s
                continue
            if t in slot_of:
                continue
            if isinstance(t, Op):
                stack.append((t, True))
                stack.extend((a, False) for a in t.args)
            elif isinstance(t, Var):
                slot_of[t] = n
                var_fills.append((n, t.name, input_pos.get(t.name, -1)))
                n += 1
            elif isinstance(t, Param):
                slot_of[t] = n
                param_fills.append((n, t.name, t))
                n += 1
            elif isinstance(t, Const):
                slot_of[t] = n
                const_fills.append((n, t.value))
                n += 1
            else:
                return None
        # Liveness: count each slot's consumers across the tape, then
        # walk steps in order — a slot's count hits zero exactly at its
        # LAST consumer (the tape is topological), so ``releases[k]``
        # is the set of arg slots safe to drop after step ``k`` runs.
        # The root never appears as an arg, so it is never released.
        uses = [0] * n
        for _o, _fi, a, _kw in steps:
            for i in a:
                uses[i] += 1
        rel: list[list[int]] = [[] for _ in steps]
        for k, (_o, _fi, a, _kw) in enumerate(steps):
            for i in a:
                uses[i] -= 1
                if uses[i] == 0:
                    rel[k].append(i)
        return _EvalPlan(
            n_slots=n,
            var_fills=tuple(var_fills),
            param_fills=tuple(param_fills),
            const_fills=tuple(const_fills),
            steps=tuple(steps),
            op_names=tuple(op_names),
            root_slot=slot_of[term],
            releases=tuple(tuple(r) for r in rel),
        )

    def run_xs(self, xs: tuple, pmap_get: Any, bget: Any) -> Any:
        """``forward`` entry: Vars resolve positionally from ``xs``.

        ``pmap_get``/``bget`` are the bound ``.get``s of the module's
        ``_param_map`` and ``_torch_bindings``.
        """
        v: list = [None] * self.n_slots
        n = len(xs)
        x = xs[0] if n else None
        for s, _name, pos in self.var_fills:
            v[s] = xs[pos] if 0 <= pos < n else x
        like = _sibling_like(list(xs))
        return self._run_tail(v, pmap_get, bget, like)

    def run_env(
        self, env: Any, x: Any, pmap_get: Any, bget: Any
    ) -> Any:
        """``_eval`` entry: Vars resolve through the caller's env dict."""
        v: list = [None] * self.n_slots
        eget = (env or {}).get
        for s, name, _pos in self.var_fills:
            v[s] = eget(name, x)
        like = _const_like(x, env, None)
        return self._run_tail(v, pmap_get, bget, like)

    def _run_tail(
        self, v: list, pmap_get: Any, bget: Any, like: Any = None
    ) -> Any:
        """Fill param/const slots, resolve bindings, run the tape."""
        for s, name, t in self.param_fills:
            p = pmap_get(name)
            v[s] = p if p is not None else _randn_param(t)
        if like is None:
            # No tensor input context (param-only root): a floating
            # Param supplies the const dtype — the same no-promotion
            # contract ``_const_like`` gives env callers.
            like = _sibling_like(
                [v[s] for s, _n, _t in self.param_fills]
            )
        for s, c in self.const_fills:
            v[s] = _mint_const(c, like)
        fns: list = []
        for nm in self.op_names:
            f = bget(nm)
            if f is None:
                raise ValueError(f"No torch binding for op '{nm}'")
            fns.append(f)
        releases = self.releases
        # Creator ops with no dtype attr (float ``arange``/``full``/
        # ``zeros``/…) mint at the computation's dtype — the same
        # *like* ``_mint_const`` uses — via ``dtype_ctx``; outside this
        # window they keep torch's ambient default.
        ctx_tok = dtype_ctx.LIKE_CTX.set(like)
        try:
            for k, (o, fi, a, kw) in enumerate(self.steps):
                f = fns[fi]
                if kw is None:
                    na = len(a)
                    if na == 1:
                        v[o] = f(v[a[0]])
                    elif na == 2:
                        v[o] = f(v[a[0]], v[a[1]])
                    elif na == 3:
                        v[o] = f(v[a[0]], v[a[1]], v[a[2]])
                    else:
                        v[o] = f(*[v[i] for i in a])
                else:
                    na = len(a)
                    if na == 0:
                        v[o] = f(**kw)
                    elif na == 1:
                        v[o] = f(v[a[0]], **kw)
                    elif na == 2:
                        v[o] = f(v[a[0]], v[a[1]], **kw)
                    elif na == 3:
                        v[o] = f(v[a[0]], v[a[1]], v[a[2]], **kw)
                    else:
                        v[o] = f(*[v[i] for i in a], **kw)
                # Liveness: drop arg slots whose last consumer just
                # ran.  The output is already stored under ``o`` before
                # any arg is released, so an aliasing binding (out IS
                # an arg tensor) keeps the storage alive through
                # ``v[o]``.
                for i in releases[k]:
                    v[i] = None
        finally:
            dtype_ctx.LIKE_CTX.reset(ctx_tok)
        return v[self.root_slot]


def _const_like(
    var_default: Any, var_env: dict | None, param_env: dict | None
) -> Any:
    """Return the floating/complex tensor a float ``Const`` mints against.

    A ``Const`` is a weak scalar operand (``x / 45.25``, ``pow(x, 2)``,
    a law-minted rate) — spelled torch never promotes it, so the
    materialised tensor must take the *computation's* dtype and
    device: the input (``var_default``/x) first, then the first
    floating tensor in the var/param envs.  ``None`` when no tensor
    context exists — the ambient default dtype is then the only
    honest fallback.  Minting at the ambient default unconditionally
    silently promoted fp32 deliveries to fp64 whenever the process
    default was flipped (``set_default_dtype(float64)`` — the verify
    convention) while the operands stayed fp32.
    """
    if isinstance(var_default, torch.Tensor) and (
        var_default.is_floating_point() or var_default.is_complex()
    ):
        return var_default
    for env in (var_env, param_env):
        for v in (env or {}).values():
            if isinstance(v, torch.Tensor) and (
                v.is_floating_point() or v.is_complex()
            ):
                return v
    return None


def _mint_const(value: Any, like: Any) -> torch.Tensor:
    """Materialise a ``Const`` — float scalars at *like*'s dtype.

    Int (and bool) values keep ``torch.tensor(v)`` — weak-int
    semantics (``x % 2`` must stay an integer operand).  A float with
    no tensor context falls back to the ambient default dtype.

    The mint stays on CPU even when *like* is a device tensor: a
    0-dim CPU scalar is legal against device operands (weak-scalar
    broadcast) AND survives CUDA-graph capture, while an on-device
    ``torch.tensor`` mint does a pageable H2D copy that capture
    rejects.
    """
    if isinstance(value, torch.Tensor):
        # A tensor-valued Const is a real operand (a baked weight or
        # index tensor), not a weak scalar: it follows the
        # computation's device — and a floating/complex one's dtype.
        # Non-float dtypes keep theirs: an integer index cast to the
        # float dtype would corrupt the gather it feeds.
        if like is None:
            return value
        if value.is_floating_point() or value.is_complex():
            return value.to(device=like.device, dtype=like.dtype)
        return value.to(device=like.device)
    if isinstance(value, float) and like is not None:
        return torch.tensor(value, dtype=like.dtype)
    return torch.tensor(value)


def _sibling_like(args: list) -> Any:
    """First floating/complex tensor among evaluated op args."""
    for a in args:
        if isinstance(a, torch.Tensor) and (
            a.is_floating_point() or a.is_complex()
        ):
            return a
    return None


def _device_like(args: list) -> Any:
    """Device reference for ``_align_to`` — dtype-agnostic.

    The first NON-CPU tensor input wins: integer/bool operands (token
    ids, index tensors, masks) pin the delivery device exactly like
    float activations do, and an all-int feed must still trigger the
    align (t5 wall #1).  Falls back to the first tensor input when
    every input is CPU — the caller's ``device.type == "cpu"`` early
    return then applies as before.
    """
    first = None
    for a in args:
        if not isinstance(a, torch.Tensor):
            continue
        if first is None:
            first = a
        if a.device.type != "cpu":
            return a
    return first


_COMPLEX_DTYPES = frozenset(
    d
    for d in (
        getattr(torch, "complex32", None),
        torch.complex64,
        torch.complex128,
    )
    if d is not None
)


def _cast_to_delivery(x: Any, dtype: torch.dtype) -> Any:
    """Cast one forward input to the module's delivery dtype.

    Real-floating tensors cast (``.to`` — a no-op view when the dtype
    already matches); integer/bool operands and non-tensors pass
    through untouched — an int64 index must never become float.  A
    complex input casts only onto a complex delivery dtype: dropping
    the imaginary part silently would be a wrong answer, so a
    complex→real mismatch passes through and lets torch's promotion
    (or its loud dtype error) decide.
    """
    if (
        not isinstance(x, torch.Tensor)
        or x.dtype == dtype
        or not (x.is_floating_point() or x.is_complex())
    ):
        return x
    if x.is_complex() and dtype not in _COMPLEX_DTYPES:
        return x
    return x.to(dtype=dtype)


def eval_term(
    term: Any,
    *,
    var_env: dict | None = None,
    param_env: dict | None = None,
    bindings: Any = None,
    memo_env: dict | None = None,
    strict: bool = False,
    var_default: Any = _MISS,
    param_default: Callable[[Param], Any] | None = None,
    tensor_only: bool = False,
) -> Any:
    """Evaluate an IR term against concrete environments.

    Single source for the eval-term family (plan 0002 phase D) —
    :meth:`IRModule._eval` (strict runtime eval; the hot path itself
    runs the :class:`_EvalPlan` slot tape and delegates here only for
    seeded-memo calls and unplannable terms), the permissive
    compile-time fold ``optimize._eval_const``, and
    ``_fold_weight_chains``' shallow fold
    all delegate here.  Leaf semantics:

    * ``Var``   → ``var_env[name]``; on a miss, ``var_default`` when
      the caller supplied one (``IRModule`` passes the forward's first
      input — the single-input "self" fallback), else failure.
    * ``Param`` → ``param_env[name]``; on a miss,
      ``param_default(term)`` when given (``IRModule`` passes
      :func:`_randn_param` — the unshaped-Param fallback), else
      failure.
    * ``Const`` → ``_mint_const(value, like)``: float values mint at
      the eval-context dtype (the operand sibling when one is a
      floating tensor, else :func:`_const_like`'s input/env pick,
      else the ambient default); ints stay ``torch.tensor(value)``.
      This keeps a Const a *weak* scalar — it never promotes its
      operands the way an ambient-fp64 mint would.
    * ``Op``    → ``bindings[op](*args, **attrs)``; ``memo_env`` (when
      given) dedups DAG-shared subtrees — interned terms are content
      keys.

    ``strict=True`` is the runtime contract: a missing binding raises
    ``ValueError``, binding errors propagate with grad enabled, and an
    unknown term type raises ``TypeError``.  ``strict=False`` is the
    fold contract: any un-evaluatable piece — missing leaf, missing
    binding, raising binding — yields ``None`` instead, and op calls
    run under ``torch.no_grad()``.  ``tensor_only`` (compile-time
    folds) additionally treats a non-Tensor op result (e.g. an ``aff``
    pair) as failure AT EVERY level — mirroring the recursive
    isinstance check of the fold sites it replaced, not a top-level
    filter.  The two env-miss raises under ``strict`` are nominal —
    every strict caller passes both defaults.
    """
    like = _const_like(var_default, var_env, param_env)

    def go(t: Any, rec: Callable[..., Any]) -> Any:
        # Recursion goes through the explicit ``rec`` parameter, not
        # the closure: a self-referencing ``go`` would sit in its own
        # ``__closure__``, leaving a reference cycle that keeps this
        # call's memo/env cells (every intermediate tensor) alive
        # until cyclic GC — an OOM source inside CUDA timing windows.

        if isinstance(t, Var):
            v = (var_env or {}).get(t.name, var_default)
            if v is _MISS:
                if (
                    strict
                ):  # pragma: no cover — strict callers pass var_default
                    raise KeyError(t.name)
                return None
            return v
        if isinstance(t, Const):
            return _mint_const(t.value, like)
        if isinstance(t, Param):
            v = (param_env or {}).get(t.name)
            if v is None:
                if param_default is not None:
                    return param_default(t)
                if strict:  # pragma: no cover — strict callers pass param_default
                    raise KeyError(t.name)
                return None
            return v
        if isinstance(t, Op):
            if memo_env is not None:
                hit = memo_env.get(t)
                if hit is not None:
                    return hit
            fn = (bindings or {}).get(t.op)
            if fn is None:
                if strict:
                    raise ValueError(
                        f"No torch binding for op '{t.op}'"
                    )
                return None
            args = [rec(a, rec) for a in t.args]
            if not strict and any(a is None for a in args):
                return None
            if any(isinstance(a, Const) for a in t.args):
                # A Const operand takes its *sibling*'s dtype — the
                # weak-scalar semantics the spelled ``x * 0.9`` has.
                # Matters when the env-wide ``like`` is absent or a
                # param-only fold mixes dtypes (``div(p_f32, c)``).
                ref = _sibling_like(args)
                if ref is not None:
                    args = [
                        _mint_const(t.args[i].value, ref)
                        if isinstance(t.args[i], Const)
                        else a
                        for i, a in enumerate(args)
                    ]
            if strict:
                out = fn(*args, **dict(t.attrs))
            else:
                try:
                    with torch.no_grad():
                        out = fn(*args, **dict(t.attrs))
                except Exception:
                    return None
            if tensor_only and not isinstance(out, torch.Tensor):
                if strict:  # pragma: no cover — no strict caller sets tensor_only
                    raise TypeError(
                        f"eval_term: op '{t.op}' returned "
                        f"{type(out).__name__}, not a tensor"
                    )
                return None
            if memo_env is not None:
                memo_env[t] = out
            return out
        if strict:
            raise TypeError(f"Cannot evaluate term: {t}")
        return None

    # Creator ops with no dtype attr mint at the computation's dtype
    # through ``dtype_ctx`` — the same *like* ``_mint_const`` uses.
    ctx_tok = dtype_ctx.LIKE_CTX.set(like)
    try:
        return go(term, go)
    finally:
        dtype_ctx.LIKE_CTX.reset(ctx_tok)


class IRModule(torch.nn.Module):
    """A torch module reconstructed from a mlopt IR term.

    The IR must contain `inputs` (Var list) and `params` (Param dict).

    Parameters
    ----------
    ir : IR
        The categorical IR to lower.
    param_values : dict[str, torch.Tensor] | None
        Optional mapping from IR param names (e.g. 'p_w1') to actual
        tensor values (e.g. the original model's W1).  Registered
        params SHARE STORAGE with these tensors (``detach``, not
        ``clone`` — eval is read-only and the verify pair lowers the
        same dict twice, so sharing collapses ~3 host copies of the
        weight set to 1; see ``docs/reports/verify_bf16_mem.md``).
        Callers needing an isolated copy clone the values before
        lowering.  If not provided, random parameters are generated.
    ops : OpTable | None
        The op table to lower through (plan 0001 phase 2c).  ``None``
        (default) resolves to ``OpTable.full()`` — the ambient table
        sharing ``_IR_TO_TORCH``, so post-construction binding
        overrides keep reaching ``_eval``.  An explicit table owns its
        own dict: an op absent from it fails loudly at eval.

    Ports layer: an ``IRModule`` is a minimal
    :class:`mlopt_core.ports.Executor` — ``forward(*xs) -> Tensor`` — and
    is itself the ``eval_mod`` the planned wrappers embed (see
    :class:`mlopt_core.ports.PlannedExecutor`); ``ops`` conforms to
    :class:`mlopt_core.ports.OpRegistry`.

    """

    def __init__(
        self,
        ir: IR,
        param_values: dict[str, torch.Tensor] | None = None,
        ops: OpTable | None = None,
    ) -> None:
        """Initialise the module, folding weight-only subtrees."""
        super().__init__()
        self._ops = ops if ops is not None else OpTable.full()
        self._torch_bindings = self._ops.torch_bindings
        self._inputs = ir.inputs
        self._param_map: dict[str, torch.nn.Parameter] = {}
        self._param_values = param_values or {}
        # Phase 3b: materialize weight-only subtrees (e.g. W1 @ (W2 @ W3))
        # at construction time, so runtime is a single matmul per fused chain.
        # Keyed by interned term OBJECTS (content-hashed), not ints.
        self._fold_memo: dict[Any, Any] = {}
        self._uses_memo: dict = {}
        self._root = self._fold_weight_chains(ir.root)
        self._build_params()
        # ``_param_values`` entries that registered as Parameters are
        # now aliases of the ``_param_map`` Parameter — the SAME
        # object, so no second copy of the weight set is pinned for
        # the module's lifetime (the ~3x weight-set blowup in the
        # whisper OOM — SearchResult leaves + this dict + the
        # Parameters).  The registered Parameters SHARE STORAGE with
        # the caller's ``param_values`` tensors (``_build_params``
        # detaches, never clones — see its comment), so the verify
        # pair's two lowers over one dict cost ONE host copy total.
        # The key stays present because ``_align_to`` mirrors both
        # tables and callers may inspect the dict; ``_align_to``
        # dedups by object identity so an aliased entry follows its
        # Parameter's move instead of materialising a second device
        # copy.  Unregistered
        # names (unshaped Params, caller extras) keep their tensors.
        # The caller's dict is never mutated — this rebinds.
        self._param_values = {
            k: self._param_map.get(k, v)
            for k, v in self._param_values.items()
        }
        # Plan-tape eval (plan 0003): last-wins name→position table so
        # the forward runner resolves Vars straight from ``xs`` without
        # building the env dict.  ``_plan_cache`` memoises per-term
        # plans for ``_eval``/``_eval_fast`` leaf evaluation.
        self._input_pos = {
            inp.name: i for i, inp in enumerate(self._inputs)
        }
        self._plan_cache: dict[Any, Any] = {}
        self._root_plan = self._plan_for(self._root)

    def _uses_input(self, term: Any) -> bool:
        """Return True if the term mentions a data-dependent leaf.

        Delegates to :func:`mlopt_core.typing.has_var_leaf` with the
        instance's content-keyed memo: extracted terms are
        shared-subterm DAGs, and an unmemoised walk is exponential in
        DAG depth.
        """
        from mlopt_core.typing import has_var_leaf

        return has_var_leaf(term, self._uses_memo)

    def _fold_weight_chains(self, term: Any) -> Any:
        """Bottom-up: replace weight-only subtrees with a single fused Param.

        A subtree with no Var leaves is parameter-only and can be evaluated
        once at construction time (using _param_values when available).
        Its result is stored as a new fused Param, so the runtime graph
        contains one matmul instead of a chain of them.  Matmul is handled
        with torch.matmul; elementwise weight-only chains (add/mul/neg/
        silu/sigmoid/square) are folded eagerly via _eval on _param_values.
        """
        from mlopt_core.ir import Op as _Op
        from mlopt_core.ir import Param as _Param

        # Memoize by identity of the CALLER's term object: the extracted
        # term may share a subtree object between several parents (e.g.
        # one fused GEMM under two chunk projections).  Returning the
        # same folded object preserves that sharing for _eval.  Must
        # capture the key before `term` is rebound to the rebuilt node.
        orig_key = term
        memo_hit = self._fold_memo.get(orig_key)
        if memo_hit is not None:
            return memo_hit

        if isinstance(term, _Op):
            folded_args = tuple(
                self._fold_weight_chains(a) for a in term.args
            )
            term = _Op.make(term.op, *folded_args, **dict(term.attrs))
            if not self._uses_input(term):
                if term.op == "matmul":
                    left = (
                        self._param_values.get(term.args[0].name)
                        if isinstance(term.args[0], _Param)
                        else None
                    )
                    right = (
                        self._param_values.get(term.args[1].name)
                        if isinstance(term.args[1], _Param)
                        else None
                    )
                    if left is not None and right is not None:
                        with torch.no_grad():
                            fused = torch.matmul(left, right)
                        fused_name = f"fused_{len(self._param_map) + len(self._param_values)}"
                        self._param_values[fused_name] = (
                            fused.detach().clone()
                        )
                        shape = tuple(int(d) for d in fused.shape)
                        from mlopt_core.ir import TensorType

                        result = _Param(
                            name=fused_name, typ=TensorType(shape)
                        )
                        self._fold_memo[orig_key] = result
                        return result
                elif term.op in FOLD_OPS or term.op in _EXTRA_FOLD_OPS:
                    # The widening semantic set — every single-tensor
                    # op folds identically (views, reductions, gathers
                    # and creators all evaluate to a fused Param).
                    # FOLD_OPS (mlopt_core.opmeta) is the one source:
                    # the cost model's _folds_to_param and the
                    # extraction param-only discount read the same
                    # set, so a subtree folded here is the subtree the
                    # cost model prices at ~0 marginal.  NOT folded:
                    # the slice/extraction/view family
                    # (index_select/gather/index/select/slice/narrow/
                    # unbind/getitem/tensor_split/expand/broadcast_to/
                    # repeat/expand_as/unfold/stack-excluded too).
                    # Those are exactly how
                    # share_duplicate_param_slices re-materialises a
                    # deduplicated weight — folding them would bake
                    # the reconstruction back into a full-size param
                    # and erase every dedup saving, for near-zero
                    # runtime gain (views and gathers are cheap).
                    # Try eager compile-time fold via the op table's
                    # torch bindings (ambient _IR_TO_TORCH for the
                    # default full table; the custom table's own dict
                    # for an explicit ``ops``).  SHALLOW by design:
                    # eval_term only fires when every arg already
                    # resolved to a leaf — a surviving Op arg means the
                    # fold list declined it below, and descending now
                    # would fold ops outside that list.
                    fused = None
                    if all(
                        isinstance(a, (_Param, Const))
                        for a in term.args
                    ):
                        fused = eval_term(
                            term,
                            param_env=self._param_values,
                            bindings=self._torch_bindings,
                            tensor_only=True,
                        )
                    if fused is not None:
                        fused_name = f"fused_{len(self._param_map) + len(self._param_values)}"
                        self._param_values[fused_name] = (
                            fused.detach().clone()
                        )
                        shape = tuple(int(d) for d in fused.shape)
                        from mlopt_core.ir import TensorType

                        result = _Param(
                            name=fused_name,
                            typ=TensorType(shape),
                        )
                        self._fold_memo[orig_key] = result
                        return result
        self._fold_memo[orig_key] = term
        return term

    def _build_params(self) -> None:
        from mlopt_core.ir import Op as _Op
        from mlopt_core.ir import Param as _Param

        param_shapes: dict[str, tuple] = {}

        # Interned term objects are content-hashed — safe set members
        # (no id()/GC hazards).
        seen: set[Any] = set()

        def collect(t: Any) -> None:
            if t in seen:
                return
            seen.add(t)
            if isinstance(t, _Param):
                if (
                    t.name not in param_shapes
                    and t.typ.size is not None
                ):
                    param_shapes[t.name] = tuple(
                        d if d is not None else 1 for d in t.typ.shape
                    )
            elif isinstance(t, _Op):
                for a in t.args:
                    collect(a)

        collect(self._root)
        for name, shape in param_shapes.items():
            if name in self._param_values:
                # Use the original model's parameter value.  Integer
                # tensors are registered without grad — Parameters
                # require a floating/complex dtype.  ``detach`` shares
                # STORAGE with the caller's tensor rather than cloning
                # (docs/reports/verify_bf16_mem.md): ``lower`` materialises
                # the verify reference and the delivery from the SAME
                # ``param_values`` dict, eval is read-only, and sharing
                # collapses ~3 host copies of the weight set to 1.  The
                # aliasing contract is documented on the class — callers
                # needing isolation clone before lowering.
                v = self._param_values[name].detach()
                p = torch.nn.Parameter(
                    v,
                    requires_grad=v.is_floating_point()
                    or v.is_complex(),
                )
            else:
                p = torch.nn.Parameter(torch.randn(*shape) * 0.02)
            # Use the IR name (e.g. p_w1) so _eval can find it
            setattr(self, name, p)
            self._param_map[name] = p

    def _plan_for(self, term: Any) -> _EvalPlan | None:
        """Fetch (or build and cache) the eval plan for ``term``.

        ``None`` when the term cannot be flattened — a non-term node
        anywhere in the DAG.  Cached under the (interned, content-keyed)
        term, so DAG-equal terms share one plan; builds are read-only
        afterwards.
        """
        plan = self._plan_cache.get(term, _MISS)
        if plan is _MISS:
            built = _EvalPlan.build(term, self._input_pos)
            plan = built if built is not None else _UNPLANNED
            self._plan_cache[term] = plan
        return None if plan is _UNPLANNED else plan

    def _align_to(self, xs: tuple) -> torch.dtype | None:
        """Move baked tensors to the delivery device/dtype, once.

        The exporter and the weight-chain fold can bake tensors in the
        process default context (CPU).  A delivery on another device must
        not inherit that — the first op mixing a real input with a baked
        weight raises.  The CPU path never enters this branch, so the fp64
        verify contract is untouched; the decision is cached per
        (device, dtype).

        Device reference: the first NON-CPU tensor input of ANY dtype —
        integer index tensors need the weight's device just as float
        activations do (``F.embedding`` requires index and weight on one
        device), and an all-int feed (T5's ``input_ids`` pipeline) must
        still trigger the align — the t5 wall #1 silent no-op.  With no
        tensor input at all the first parameter's device is the target,
        so CPU-baked ``fused_*`` tensors still reach CUDA params.

        Dtype reference: the MODULE's dtype — the majority floating
        dtype across the registered params — NOT the first float
        input's.  The old rule cast every weight to the caller's input
        dtype, so an fp32 mask silently flipped a bf16 delivery to fp32
        (t5 wall #3).  Weights keep the module dtype; ``forward`` casts
        float INPUTS to it instead.  Non-floating leaves move device
        only, dtype preserved (an int64 index must never become float).
        Scalar ``Const`` minting is unaffected — ``_mint_const``
        materialises those at eval time.

        Returns the delivery dtype (for ``forward``'s input cast) or
        ``None`` when no alignment applies (CPU / no tensors).
        """
        like = _device_like(list(xs))
        if like is None:
            # No tensor input: align baked tensors to the params' own
            # device (first tensor in the live tables).
            for table in (self._param_map, self._param_values):
                for t in table.values():
                    if isinstance(t, torch.Tensor):
                        like = t
                        break
                if like is not None:
                    break
        if like is None or like.device.type == "cpu":
            return None
        dtype_like = self._delivery_dtype(xs)
        key = (str(like.device), str(dtype_like))
        if getattr(self, "_dev_key", None) != key:
            tables: list[dict[str, Any]] = [
                self._param_map,
                self._param_values,
            ]
            # ``_param_values`` aliases registered names to their
            # ``_param_map`` Parameter (same object — see ``__init__``),
            # so the same tensor can appear under two keys.  ``moved_for``
            # maps id(old tensor) → (old, replacement): the tuple keeps
            # the old object alive so its id cannot be recycled mid-loop,
            # and an aliased entry rebinds to the SAME moved tensor
            # instead of paying a second device copy.
            moved_for: dict[int, tuple[Any, Any]] = {}
            for table in tables:
                for k, t in list(table.items()):
                    if not isinstance(t, torch.Tensor):
                        continue
                    prev = moved_for.get(id(t))
                    if prev is not None:
                        table[k] = prev[1]
                        continue
                    if t.is_floating_point() or t.is_complex():
                        target_dtype = (
                            dtype_like
                            if dtype_like is not None
                            else t.dtype
                        )
                        if (
                            t.device == like.device
                            and t.dtype == target_dtype
                        ):
                            continue
                        moved = t.detach().to(
                            device=like.device, dtype=target_dtype
                        )
                    else:
                        # Integer/bool leaves: dtype is semantic (index
                        # tensors must not be cast to the float dtype) —
                        # only the device must match.
                        if t.device == like.device:
                            continue
                        moved = t.detach().to(device=like.device)
                    new = (
                        torch.nn.Parameter(
                            moved, requires_grad=t.requires_grad
                        )
                        if isinstance(t, torch.nn.Parameter)
                        else moved
                    )
                    moved_for[id(t)] = (t, new)
                    table[k] = new
            self._dev_key = key
        return dtype_like

    def _delivery_dtype(self, xs: tuple) -> torch.dtype | None:
        """Return the dtype this delivery computes in — the module's.

        Majority floating dtype across the registered params (first-seen
        wins ties) so a stray differently-dtyped weight cannot claim the
        module; falls back to the first float INPUT's dtype only when the
        module has no floating param at all (a param-free float graph),
        and ``None`` when nothing floats — int-only deliveries then move
        device-only.
        """
        counts: dict[torch.dtype, int] = {}
        for t in self._param_map.values():
            if isinstance(t, torch.Tensor) and (
                t.is_floating_point() or t.is_complex()
            ):
                counts[t.dtype] = counts.get(t.dtype, 0) + 1
        if counts:
            return max(counts, key=counts.__getitem__)
        like = _sibling_like(list(xs))
        return like.dtype if like is not None else None

    def _park_params(self) -> None:
        """Move the live weight tables to CPU — the low-memory verify park.

        ``lower(verify=True)`` materialises a reference lowering next
        to the delivered module; on a memory-bound device two full
        weight sets do not fit (the whisper-turbo 4 GB OOM — see
        ``docs/reports/verify_lowmem.md``).
        :func:`mlopt_torch.report.verify_module` calls this on each
        side that offers it: the module's *live* tensors — the
        ``_param_map`` / ``_param_values`` tables ``_align_to``
        rewrites on first eval, NOT the stale registered
        ``_parameters`` entries, which are the pre-align CPU clones —
        drop to CPU around the other side's forward, so the verify
        pair never holds both weight sets on the compute device at
        once.  ``_dev_key`` resets so the next forward re-aligns
        exactly as a fresh module would: the park is invisible to
        callers beyond transient device residency, and the check
        itself is unchanged — same forwards, same comparison.

        The ``moved_for`` dedup mirrors ``_align_to``: ``_param_values``
        aliases registered names to their ``_param_map`` Parameter, so
        the same tensor appears under two keys and must rebind to ONE
        CPU tensor; the kept ``(old, new)`` tuples pin the old objects
        so their ids cannot be recycled mid-loop.

        The CPU replacement prefers ``self._parameters[k]`` — the
        registered attribute ``_build_params`` installed before any
        ``_align_to`` lift.  ``_align_to`` rewrites only the two live
        tables, never ``_parameters``, so the registered entry stays
        the canonical host-resident tensor — it SHARES STORAGE with
        the caller's ``param_values`` (``_build_params`` detaches,
        never clones): rebinding to it releases the device copy
        WITHOUT allocating a second host copy (the park trades VRAM
        for RAM — a fresh ``.to("cpu")`` per tensor would briefly
        hold both).  Registered entries are keyed by the same param
        name the tables use.  A live tensor with no CPU registered
        counterpart (a caller-supplied extra that arrived on-device)
        falls back to a real ``.to("cpu")``.
        """
        registered = self._parameters
        moved_for: dict[int, tuple[Any, Any]] = {}
        tables: list[dict[str, Any]] = [
            self._param_map,
            self._param_values,
        ]
        for table in tables:
            for k, t in list(table.items()):
                if not isinstance(t, torch.Tensor):
                    continue
                prev = moved_for.get(id(t))
                if prev is not None:
                    table[k] = prev[1]
                    continue
                if t.device.type == "cpu":
                    moved_for[id(t)] = (t, t)
                    continue
                reg = registered.get(k)
                if (
                    isinstance(reg, torch.Tensor)
                    and reg.device.type == "cpu"
                ):
                    moved = reg
                else:
                    moved = t.detach().to(device="cpu")
                    if isinstance(t, torch.nn.Parameter):
                        moved = torch.nn.Parameter(
                            moved, requires_grad=t.requires_grad
                        )
                moved_for[id(t)] = (t, moved)
                table[k] = moved
        self._dev_key = None

    def _apply(self, fn: Callable, recurse: bool = True) -> IRModule:
        """``Module.to``/``cuda``/``float``/``half`` — plus table resync.

        ``Module._apply`` rebinds every ``_parameters`` entry to a NEW
        tensor object, but ``forward`` reads the LIVE tables
        (``_param_map`` / ``_param_values``) — and after an
        ``_align_to`` lift those are deliberately different objects
        than the registered params (``_parameters`` stays the
        canonical host copy ``state_dict`` reports and ``_park_params``
        rebinds to).  A bare ``super()._apply`` would move the
        registration while eval kept using pre-move weights — e.g.
        ``.cpu()`` after a CUDA forward left CUDA tensors under CPU
        inputs and the next forward died on a device mismatch (the
        ``_dev_key`` cache would even suppress the re-align that
        could have healed it).

        Resync: registered names rebind to the moved ``_parameters``
        entries (restoring the ``__init__`` triple-alias
        ``_param_map[k] is _param_values[k] is _parameters[k]``);
        unregistered ``_param_values`` extras — which ``_apply``
        never sees — are moved by ``fn`` directly, deduped by object
        identity like ``_align_to``.  ``_dev_key`` resets so the next
        forward re-derives device/dtype alignment exactly as a fresh
        module would.
        """
        super()._apply(fn, recurse)
        param_map = getattr(self, "_param_map", None)
        if param_map is None:  # pragma: no cover — _apply is post-init
            return self
        registered = self._parameters
        for name in param_map:
            reg = registered.get(name)
            if reg is not None:
                param_map[name] = reg
        moved_for: dict[int, Any] = {}
        for k, t in list(self._param_values.items()):
            reg = registered.get(k)
            if reg is not None:
                self._param_values[k] = reg
                continue
            if not isinstance(t, torch.Tensor):
                continue
            prev = moved_for.get(id(t))
            if prev is not None:
                self._param_values[k] = prev
                continue
            with torch.no_grad():
                moved = fn(t)
            if isinstance(t, torch.nn.Parameter) and not isinstance(
                moved, torch.nn.Parameter
            ):
                moved = torch.nn.Parameter(
                    moved, requires_grad=t.requires_grad
                )
            moved_for[id(t)] = moved
            self._param_values[k] = moved
        self._dev_key = None
        return self

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Load weights, then propagate them into the LIVE tables.

        ``Module.load_state_dict`` writes ``_parameters`` only; after
        an ``_align_to`` lift the tables ``forward`` reads are
        different objects, so without this the module would keep
        evaluating the pre-load weights while ``state_dict()``
        reported the new ones.  For every registered name present in
        ``state_dict``, the live tensor gets ``copy_`` from the
        (already updated) registered param — ``copy_`` casts, so the
        aligned view keeps its device/dtype.  ``_param_values``
        aliases the same objects for registered names and follows;
        unregistered extras are not in ``state_dict`` and stay
        untouched.  ``_dev_key`` is value-independent (device/dtype
        only) and needs no reset.
        """
        result = super().load_state_dict(
            state_dict, strict=strict, assign=assign
        )
        param_map = getattr(self, "_param_map", None)
        if param_map is None:  # pragma: no cover — post-init only
            return result
        registered = self._parameters
        done: set[int] = set()
        with torch.no_grad():
            for table in (param_map, self._param_values):
                for name, live in table.items():
                    if name not in state_dict or not isinstance(
                        live, torch.Tensor
                    ):
                        continue
                    if id(live) in done:
                        continue
                    done.add(id(live))
                    src = registered.get(name)
                    if src is None or live is src:
                        continue
                    live.copy_(src)
        return result

    def forward(self, *xs: Any) -> Any:
        """Run the module: bind inputs and evaluate the root."""
        cast_to = self._align_to(xs)
        if cast_to is not None:
            # Inputs cast TO the module's dtype — never the reverse
            # (t5 wall #3: an fp32 mask used to flip every bf16 weight
            # to fp32).  Only real-floating inputs cast; integer/bool
            # operands keep their dtype (indices!), and a complex input
            # is never silently stripped to a real dtype.
            xs = tuple(_cast_to_delivery(x, cast_to) for x in xs)
        plan = self._root_plan
        if plan is not None:
            # The tape's positional Var fills compute exactly what the
            # env dict would: ``xs[i] if i < len(xs) else x`` per
            # input name, ``x`` for everything else.
            return plan.run_xs(
                xs, self._param_map.get, self._torch_bindings.get
            )
        # Unplannable root (e.g. a bare non-term): the strict recursive
        # path preserves eval_term's exact failure semantics.
        x = cast(torch.Tensor, xs[0] if xs else None)
        env: dict[str, Any] = {"self": x}
        # Map input placeholders positionally to forward args
        for i, inp in enumerate(self._inputs):
            env[inp.name] = xs[i] if i < len(xs) else x
        return self._eval(self._root, env, x, {})

    def _eval(
        self,
        term: Any,
        env: dict[str, Any],
        x: Any,
        memo: dict[Any, Any],
    ) -> Any:
        """Strict runtime evaluation — plan tape, or :func:`eval_term`.

        ``var_default=x`` is the single-input "self" fallback (``env``
        always carries ``"self"`` → ``x``, so a Var missing from ``env``
        resolves to the first input); ``param_default=_randn_param``
        materialises Params that never registered (unshaped
        ``typ.size is None``); shared subtrees dedup through ``memo``.

        A non-empty ``memo`` means caller-seeded short-circuits (the
        omd carrier seeds batched map/projection results INTO memo —
        they may hit at ANY depth), so the recursive evaluator runs
        verbatim.  An empty/absent memo gets the plan tape — internal
        dedup is structural (slots), and the top-level result is still
        written back under ``term`` when it is an ``Op``, matching
        ``eval_term``'s observable memo contract.
        """
        if memo:
            return eval_term(
                term,
                var_env=env,
                var_default=x,
                param_env=self._param_map,
                param_default=_randn_param,
                bindings=self._torch_bindings,
                memo_env=memo,
                strict=True,
            )
        plan = self._plan_for(term)
        if plan is None:
            return eval_term(
                term,
                var_env=env,
                var_default=x,
                param_env=self._param_map,
                param_default=_randn_param,
                bindings=self._torch_bindings,
                memo_env=memo,
                strict=True,
            )
        out = plan.run_env(
            env, x, self._param_map.get, self._torch_bindings.get
        )
        if memo is not None and isinstance(term, Op):
            memo[term] = out
        return out

    def _eval_fast(self, term: Any, env: dict[str, Any], x: Any) -> Any:
        """Memo-free strict eval for leaf terms — the tape, always.

        Same semantics as ``_eval`` without the caller memo: used by
        :meth:`BatchedExecutorBase.ev_factory` where the memo contract
        is discharged by the closure itself.
        """
        plan = self._plan_for(term)
        if plan is None:
            return eval_term(
                term,
                var_env=env,
                var_default=x,
                param_env=self._param_map,
                param_default=_randn_param,
                bindings=self._torch_bindings,
                strict=True,
            )
        return plan.run_env(
            env, x, self._param_map.get, self._torch_bindings.get
        )


def ir_to_torch_module(
    ir: IR,
    param_values: dict[str, torch.Tensor] | None = None,
    ops: OpTable | None = None,
) -> IRModule:
    """Convert a mlopt IR into a torch.nn.Module.

    ``ops`` selects the lowering table; ``None`` (default) uses
    ``OpTable.full()`` — the ambient ``_IR_TO_TORCH`` table.  The
    return type stays the concrete :class:`IRModule` rather than the
    :class:`mlopt_core.ports.Executor` port — callers use module attrs the
    port doesn't name (``state_dict``/``parameters``/``_eval``).
    """
    return IRModule(ir, param_values=param_values, ops=ops)

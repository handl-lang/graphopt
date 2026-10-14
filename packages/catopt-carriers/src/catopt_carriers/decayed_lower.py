"""Pairwise-batched lowering for the chunked ``om_decayed`` carrier.

After eqsat with the K-way chunk laws (``om_kway_laws``), decayed
retention extracts as a ``concat`` over per-chunk sums::

    out_i = om_decayed(s_i, decay, v_i) / S
          + Σ_j  decay_vec_ij ⊙ (q_i @ (A_j @ v_j)) / S

— the fully expanded pairwise-block form of the chunked decayed
scan: chunk *i*'s output gathers a masked intra-chunk attention
element plus one decayed cross term per earlier block *j*.

The generic :class:`~catopt_torch.torch_bridge.IRModule` evaluates
that term one op at a time: at K=8 chunks that is ~51 tiny batched
matmuls, ~190 kernel launches, and every ``decay``/``decay_vec``
generator re-materialised per forward.  The math is right — the
schedule is the waste.  This module lowers the same term
*pairwise-batched*:

* **Leaves** — every distinct operand subterm (``split``s of the
  shared q/k/v projections, ``A_j`` state halves) evaluates once
  through the embedded ``IRModule``'s memoised evaluator; the plan
  tape already dedups shared subterms.
* **Constants** — ``decay``/``decay_vec``/``unsqueeze`` wrappers
  carry only Param/Const leaves, so each is evaluated once per
  (device, input dtype) and cached: the ``arange``/``pow``/``tril``
  generators leave the steady-state loop entirely.
* **Intra** — the score matmuls batch into one ``matmul`` over the
  stacked chunk axis, and so do the decay product and the ``@v``
  contraction: 2 batched GEMMs for what was 2K serial ones.
* **Cross** — the ``A_j @ v_j`` state halves are content-identical
  across target chunks (interned terms dedup) and batch into one
  GEMM; every pair product ``q_i @ M_j`` gathers into a second.
  Contributions accumulate per target chunk by deterministic segment
  sums (pairs arrive in arm order — contiguous per chunk), never
  atomic scatter.

``to_chunked_decayed_module`` detects the pattern; any other IR —
or any arm that fails to parse (a missing ``om_decayed`` element, a
summand that is not a decay-gated pair product, non-uniform shapes)
— falls back to the ordinary IRModule evaluation (check
``mod.is_batched``).  The fallback is the embedded module itself, so
the wrapper is a drop-in replacement for ``ir_to_torch_module`` on
ANY IR — never a silent drop.

Usage::

    mod = to_chunked_decayed_module(opt_ir, param_values=source)
    out = mod(x)
    mod.capture_cuda_graph(x)      # optional, CUDA only
    out = mod(x)                   # graph replay
"""

from __future__ import annotations

from typing import Any, cast

import torch
from catopt_core.ir import IR, Const, Op
from catopt_core.typing import _shape_of
from catopt_torch.executors import BatchedExecutorBase
from catopt_torch.torch_bridge import IRModule, eval_term

from catopt_carriers.om_lower import _slice_index, _sliced_gather

__all__ = [
    "ChunkedDecayModule",
    "build_decayed_plan",
    "is_decayed_chunked_term",
    "to_chunked_decayed_module",
]


# ---------------------------------------------------------------------------
#  Term plumbing
# ---------------------------------------------------------------------------


def _flat_same(t: Any, op: str, dim: int | None = None) -> list:
    """Flatten a same-op tree (``concat`` on *dim*, or ``add``)."""
    if (
        isinstance(t, Op)
        and t.op == op
        and len(t.args) >= 2
        and (dim is None or t.attrs.get("dim") == dim)
    ):
        out: list = []
        for a in t.args:
            out += _flat_same(a, op, dim)
        return out
    return [t]


def _decay_row(t: Any) -> Op | None:
    """Match ``unsqueeze(decay_vec(...), dim)`` — a per-row decay gate."""
    if (
        isinstance(t, Op)
        and t.op == "unsqueeze"
        and len(t.args) == 1
        and isinstance(t.args[0], Op)
        and t.args[0].op == "decay_vec"
    ):
        return t
    return None


def _gated_factor(a: Any, b: Any) -> tuple | None:
    """Return the ``(gate, other)`` of a two-operand product, or ``None``.

    Exactly one operand must be a :func:`_decay_row`; zero or two
    gates is not a gated product.
    """
    u0, u1 = _decay_row(a), _decay_row(b)
    if (u0 is None) == (u1 is None):
        return None
    return (u0, b) if u0 is not None else (u1, a)


def _peel_summand(t: Any) -> tuple | None:
    """Peel ``mul(decay_row, ·)``/``div(·, const)`` off a summand.

    Returns ``(row_term | None, body, scale_leaf | None)`` — the row
    gate may sit inside or outside the ``div``, and either ``mul``
    operand order is accepted.  ``None`` when peeling lands on a
    non-``Op`` body or a second row gate appears.
    """
    row: Op | None = None
    scale: Any = None
    while isinstance(t, Op):  # every pass shrinks t or breaks
        if (
            t.op == "div"
            and len(t.args) == 2
            and not isinstance(t.args[1], Op)
            and scale is None
        ):
            scale = t.args[1]
            t = t.args[0]
            continue
        if t.op == "mul" and len(t.args) == 2:
            gate = _gated_factor(t.args[0], t.args[1])
            if gate is None:
                break  # no row gate — or two: the mul IS the body
            if row is not None:
                return None
            row, t = gate
            continue
        break
    if not isinstance(t, Op):
        return None
    return (row, t, scale)


def _classify_body(t: Op) -> tuple | None:
    """Classify a peeled summand body.

    ``("intra", (score, decay, v))`` for ``om_decayed(s, d, v)``;
    ``("cross", (q, a, v))`` for the right-associated pair product
    ``q @ (A @ v)`` — the extracted decayed-scan's cross term.
    """
    if t.op == "om_decayed" and len(t.args) == 3:
        return ("intra", tuple(t.args))
    if t.op == "matmul" and len(t.args) == 2:
        inner = t.args[1]
        if (
            isinstance(inner, Op)
            and inner.op == "matmul"
            and len(inner.args) == 2
        ):
            return ("cross", (t.args[0], inner))
    return None


def _parse_arm(arm: Any) -> dict | None:
    """Parse one concat arm into ``{intra, cross[]}`` or ``None``.

    Every arm must carry exactly one ``om_decayed`` intra element;
    every other summand must be a decay-gated pair product.
    """
    intra: tuple | None = None
    cross: list[tuple] = []
    for s in _flat_same(arm, "add"):
        p = _peel_summand(s)
        if p is None:
            return None
        row, body, scale = p
        cls = _classify_body(body)
        if cls is None or (cls[0] == "intra" and intra is not None):
            return None
        if cls[0] == "intra":
            intra = (*cls[1], scale, row)
        else:
            cross.append((row, *cls[1], scale))
    if intra is None:
        return None
    return {"intra": intra, "cross": cross}


def _uniform(shapes: list) -> tuple | None:
    """Return the common concrete shape, or ``None`` when non-uniform."""
    s0 = shapes[0]
    if not (
        isinstance(s0, tuple)
        and all(isinstance(d, int) for d in s0)
        and all(s == s0 for s in shapes[1:])
    ):
        return None
    return s0


def _score_pair(t: Any) -> tuple | None:
    """``matmul(a, b)`` → ``(a, b)``; anything else → ``None``."""
    if isinstance(t, Op) and t.op == "matmul" and len(t.args) == 2:
        return (t.args[0], t.args[1])
    return None


def _dedup(terms: list) -> tuple[list, dict]:
    """First-seen-order unique terms + their index map."""
    seen: dict[Any, int] = {}
    out: list = []
    for t in terms:
        if t not in seen:
            seen[t] = len(out)
            out.append(t)
    return out, seen


def _contig_chunk_range(parts: list) -> tuple | None:
    """Return a contiguous equal-chunk index range of one base.

    ``(base, dim, nparts, part, first_idx, count)`` — the contiguous
    analogue of :func:`_sliced_gather`: every part is a chunk of the
    same base along the same dim with equal part size, indices
    forming one contiguous run.  ``None`` otherwise.
    """
    if not parts or any(p is None for p in parts):
        return None
    base, kind, dim, _, n, part = parts[0]
    if kind != "chunk" or part is None or not _same_chunk_meta(parts):
        return None
    idxs = [p[3] for p in parts]
    if idxs != list(range(idxs[0], idxs[0] + len(idxs))):
        return None
    return (base, dim, n, part, idxs[0], len(idxs))


def _same_chunk_meta(parts: list) -> bool:
    """Return True when every part chunks the same base identically."""
    base, kind, dim, _, n, part = parts[0]
    return all(
        p[0] == base
        and p[1] == kind
        and p[2] == dim
        and p[4] == n
        and p[5] == part
        for p in parts
    )


def _state_half(inner: Op) -> tuple:
    """Decompose a cross term's inner ``matmul(A, v)`` for batching.

    Returns ``(k_or_a, u, v, d0, d1)``: when ``A`` is the extracted
    form ``transpose(mul(k, decay_row), d0, d1)``, the decay gate is
    lifted out so it const-caches (``u`` = the ``unsqueeze`` term);
    otherwise ``(A, None, v, None, None)`` and ``A`` evaluates through
    the generic evaluator each forward — still correct, just slower.
    """
    a, v = inner.args
    d0, d1 = (
        (a.attrs.get("dim0"), a.attrs.get("dim1"))
        if isinstance(a, Op)
        and a.op == "transpose"
        and len(a.args) == 1
        else (None, None)
    )
    if (
        isinstance(a, Op)
        and isinstance(d0, int)
        and isinstance(d1, int)
        and isinstance(a.args[0], Op)
        and a.args[0].op == "mul"
        and len(a.args[0].args) == 2
    ):
        gk = _gated_factor(a.args[0].args[0], a.args[0].args[1])
        if gk is not None:
            return (gk[1], gk[0], v, d0, d1)
    return (a, None, v, None, None)


def _cross_tables(arms: list[dict]) -> tuple:
    """Dedup the cross operands into q/state tables + pair indices.

    The inner ``matmul(a_j, v_j)`` halves dedup AS TERMS — interned,
    so the pair ``(A_j, v_j)`` is the term's own pairing and never
    re-pairs.
    """
    inners, in_idx = _dedup([c[2] for a in arms for c in a["cross"]])
    q_terms, q_idx = _dedup([c[1] for a in arms for c in a["cross"]])
    pairs = [
        (q_idx[c[1]], in_idx[c[2]], c[0], c[3])
        for a in arms
        for c in a["cross"]
    ]
    return inners, q_terms, pairs


def _uniform_shapes(arms: list[dict], inners: list, q_terms: list):
    """Validate uniform concrete shapes; return the chunk shape."""
    chunk = _uniform([_shape_of(a["intra"][2]) for a in arms])
    score = _uniform([_shape_of(a["intra"][0]) for a in arms])
    if chunk is None or score is None or len(chunk) < 2:
        return None
    if any(
        _uniform([_shape_of(t) for t in ts]) is None
        for ts in (q_terms, inners)
        if ts
    ):
        return None
    return chunk


def _score_operands(arms: list[dict]) -> list | None:
    """Score ``matmul`` operand pairs when every score is one."""
    scores = [_score_pair(a["intra"][0]) for a in arms]
    return scores if all(s is not None for s in scores) else None


def _assemble_plan(arms: list[dict], dim: int) -> dict | None:
    """Validate operand shapes and build the batched schedule."""
    inners, q_terms, pairs = _cross_tables(arms)
    chunk = _uniform_shapes(arms, inners, q_terms)
    if chunk is None:
        return None
    states = [_state_half(t) for t in inners]
    return {
        "dim": dim,
        "n": len(arms),
        "chunk": chunk,
        "score_ab": _score_operands(arms),
        "score_terms": [a["intra"][0] for a in arms],
        "decay_terms": [a["intra"][1] for a in arms],
        "v_intra": [a["intra"][2] for a in arms],
        "intra_scale": [a["intra"][3] for a in arms],
        "intra_row": [a["intra"][4] for a in arms],
        "states": states,
        "q_terms": q_terms,
        "pairs": pairs,
        "pair_counts": [len(a["cross"]) for a in arms],
    }


def build_decayed_plan(root: Any) -> dict | None:
    """Analyse a chunked-``om_decayed`` term into a batched schedule.

    ``None`` for anything that is not a ``concat`` over per-chunk
    decayed sums with uniform shapes — the caller then keeps the
    generic evaluation.
    """
    if not (isinstance(root, Op) and root.op == "concat"):
        return None
    dim = root.attrs.get("dim")
    if not isinstance(dim, int):
        return None
    parsed = [_parse_arm(a) for a in _flat_same(root, "concat", dim)]
    if any(p is None for p in parsed):
        return None
    return _assemble_plan([p for p in parsed if p is not None], dim)


def is_decayed_chunked_term(root: Any) -> bool:
    """Return True when *root* is the chunked decayed-scan pattern."""
    return build_decayed_plan(root) is not None


# ---------------------------------------------------------------------------
#  The module
# ---------------------------------------------------------------------------


class ChunkedDecayModule(BatchedExecutorBase, torch.nn.Module):
    """nn.Module running a chunked ``om_decayed`` term pairwise-batched.

    Parameters
    ----------
    ir : IR
        The (post-extraction) IR to lower.
    param_values : dict[str, torch.Tensor] | None
        Mapping from IR param names to tensor values, as for
        :func:`~catopt_torch.torch_bridge.ir_to_torch_module`.

    The module embeds a plain :class:`IRModule` (as submodule
    ``eval_mod``) used for operand evaluation and as the complete
    fallback when the root is not the chunked pattern — a drop-in
    replacement for ``ir_to_torch_module`` on ANY IR.

    Ports layer: conforms to
    :class:`catopt_core.ports.BatchedExecutor` (``forward`` +
    ``_plan`` + ``is_batched``).

    """

    def __init__(
        self,
        ir: IR,
        param_values: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Initialise the module, plan, and graph state."""
        super().__init__()
        self._inputs = ir.inputs
        self.eval_mod = IRModule(ir, param_values)
        # Plan from the FOLDED root: weight-only subtrees may have
        # been rewritten to fused Params during construction.
        self._plan = build_decayed_plan(self.eval_mod._root)
        self._const_cache: dict[tuple, torch.Tensor] = {}
        self._init_graph_state()

    @property
    def is_batched(self) -> bool:
        """True when the root matched the chunked decayed pattern."""
        return self._plan is not None

    @property
    def n_blocks(self) -> int:
        """Number of output chunks (0 when not batched)."""
        return 0 if self._plan is None else self._plan["n"]

    def forward(self, *xs: torch.Tensor) -> Any:
        """Run the batched (or serial-fallback) forward pass."""
        g = self._graph
        if (
            g is not None
            and len(xs) == len(self._graph_inputs)
            and all(
                t.shape == b.shape
                and t.dtype == b.dtype
                and t.device == b.device
                for t, b in zip(xs, self._graph_inputs, strict=True)
            )
        ):
            for buf, t in zip(self._graph_inputs, xs, strict=True):
                buf.copy_(t, non_blocking=True)
            g.replay()
            return cast(torch.Tensor, self._graph_out)
        return self._forward_impl(*xs)

    # -- eval helpers -------------------------------------------------

    def _const(self, term: Any, like: torch.Tensor) -> torch.Tensor:
        """Evaluate a const term once per (device, ambient dtype).

        ``decay``/``decay_vec``/``unsqueeze`` wrappers carry only
        Param/Const leaves; the evaluated tensor's own dtype is kept
        (a Const leaf may legitimately evaluate narrower than *like* —
        promotion happens at the mul, as in generic eval).
        """
        key = ("const", term, like.dtype)
        t = self._const_cache.get(key)
        if t is None or t.device != like.device:
            t = eval_term(
                term,
                var_env={"self": like},
                param_env=self.eval_mod._param_map,
                bindings=self.eval_mod._torch_bindings,
                strict=True,
            )
            self._const_cache[key] = t
        return t

    def _scalar(self, t: Any, env: dict, x: Any) -> float:
        """Evaluate a scale leaf (Const / Param / literal) to float."""
        if isinstance(t, Const):
            return float(t.value)
        if isinstance(t, (int, float)):
            return float(t)
        v = eval_term(
            t,
            var_env=env,
            var_default=x,
            param_env=self.eval_mod._param_map,
            bindings=self.eval_mod._torch_bindings,
            strict=True,
        )
        return float(v.detach() if isinstance(v, torch.Tensor) else v)

    def _scales(
        self,
        tag: str,
        terms: list,
        env: dict,
        x: Any,
        like: torch.Tensor,
    ) -> torch.Tensor:
        """Per-summand divisors as one cached ``(n,)`` tensor.

        Scale leaves are consts — evaluated once per (device, dtype)
        so steady-state forwards (and CUDA-graph capture) never build
        a host→device tensor.
        """
        key = ("scales", tag)
        t = self._const_cache.get(key)
        if (
            t is None
            or t.device != like.device
            or t.dtype != like.dtype
        ):
            vals = [self._scalar(s, env, x) for s in terms]
            t = torch.tensor(vals, dtype=like.dtype, device=like.device)
            self._const_cache[key] = t
        return t

    def _index(self, name: str, idxs: list[int], like: torch.Tensor):
        """Return a cached long index tensor for the pair gathers."""
        return self._cached(
            ("idx", name, tuple(idxs)),
            like,
            lambda t: torch.tensor(
                idxs, dtype=torch.long, device=t.device
            ),
            dtype=torch.long,
        )

    @staticmethod
    def _pad_row(t: torch.Tensor, rank: int) -> torch.Tensor:
        """Left-pad a per-row operand to a chunk's full rank."""
        pad = rank - t.dim()
        if pad > 0:
            t = t.reshape(*((1,) * pad), *t.shape)
        return t

    @staticmethod
    def _pad_stacked(t: torch.Tensor, rank: int) -> torch.Tensor:
        """Pad a ``(n, *row)`` stack so each row broadcasts at *rank*."""
        pad = rank - t.dim()
        if pad > 0:
            t = t.reshape(t.shape[0], *((1,) * pad), *t.shape[1:])
        return t

    def _eval_sliced(self, info: tuple, ev) -> torch.Tensor:
        """Stack ``(n, *leaf_shape)`` from a common-base slice gather.

        ``select`` slices: ``base.movedim(d, 0)``.  ``chunk``/equal
        ``split`` slices: split dim d into ``(n, part)`` by a view,
        then move the block axis front — a permute, zero compute.
        """
        base, kind, dim, n, part = info
        t = ev(base)
        d = dim % t.dim()
        if kind == "select":
            return t.movedim(d, 0)
        t = t.reshape(*t.shape[:d], n, part, *t.shape[d + 1 :])
        return t.movedim(d, 0)

    def _stack_terms(self, terms: list, ev) -> torch.Tensor:
        """``(n, *leaf)`` stack — a view when terms slice one base.

        Full coverage (the classic ``_sliced_gather`` case) is a
        reshape+permute with no kernel; a *contiguous* index range of
        equal chunks of one base is a reshape + narrow + permute —
        also a view.  Anything else falls back to per-term eval +
        ``torch.stack``.
        """
        parts = [_slice_index(t) for t in terms]
        g = _sliced_gather(parts)
        if g is not None:
            return self._eval_sliced(g, ev)
        rng = _contig_chunk_range(parts)
        if rng is not None:
            base, dim, n, part, i0, cnt = rng
            t = ev(base)
            d = dim % t.dim()
            v = t.reshape(*t.shape[:d], n, part, *t.shape[d + 1 :])
            return v.narrow(d, i0, cnt).movedim(d, 0)
        return torch.stack([ev(t) for t in terms])

    def _row_stack(
        self, terms: list, like: torch.Tensor, rank: int
    ) -> torch.Tensor:
        """Stack const rows, defaulting a missing gate to ``1``.

        Rows may broadcast differently against the chunk (a ``(c,1)``
        gate, a ``(c,c)`` mask): each pads to *rank*, all expand to
        their common broadcast shape, ``None`` gates contribute an
        exact multiplicative identity.  Callers guarantee ≥1 real
        term — the all-``None`` case never reaches the mul.
        """
        rows = [
            (
                self._pad_row(self._const(t, like), rank)
                if t is not None
                else None
            )
            for t in terms
        ]
        bt = torch.broadcast_shapes(
            *[r.shape for r in rows if r is not None]
        )
        one = self._cached(
            ("ones", bt),
            like,
            lambda t: torch.ones(bt, dtype=t.dtype, device=t.device),
        )
        return torch.stack(
            [r.expand(bt) if r is not None else one for r in rows]
        )

    # -- forward ------------------------------------------------------

    def _forward_intra(
        self, plan: dict, ev: Any, env: dict, x: Any, like: torch.Tensor
    ) -> torch.Tensor:
        """Compute the stacked ``(s ⊙ d) @ v / S`` intra-chunk product."""
        sab = plan["score_ab"]
        if sab is not None:
            s = torch.matmul(
                self._stack_terms([p[0] for p in sab], ev),
                self._stack_terms([p[1] for p in sab], ev),
            )
        else:
            s = self._stack_terms(plan["score_terms"], ev)
        dts = plan["decay_terms"]
        if all(t == dts[0] for t in dts):
            # A shared decay mask broadcasts — no K-way copy.
            d = self._pad_row(self._const(dts[0], like), s.dim() - 1)
            d = d.unsqueeze(0)
        else:
            d = self._pad_stacked(
                torch.stack([self._const(t, like) for t in dts]),
                s.dim(),
            )
        v = self._stack_terms(plan["v_intra"], ev)
        out = torch.matmul(s * d, v)  # (K, *batch, c, dv)
        if any(r is not None for r in plan["intra_row"]):
            rows = self._row_stack(
                plan["intra_row"], like, len(plan["chunk"])
            )
            out = rows * out
        sc = self._scales("intra", plan["intra_scale"], env, x, like)
        return out / sc.reshape(-1, *((1,) * (out.dim() - 1)))

    def _state_row(self, s: tuple, ev: Any, like: torch.Tensor):
        """One cross inner lhs — eval-mode or the parsed gated key."""
        kk = ev(s[0])
        if s[1] is None:
            return kk
        kd = kk * self._pad_row(self._const(s[1], like), kk.dim())
        return kd.transpose(s[3], s[4])

    def _state_stack(self, states: list, ev: Any, like: torch.Tensor):
        """Stack the ``(J, …)`` cross inner lhs terms.

        When every inner is the uniform ``transpose(k·U)`` form the
        decay product batches over the stack axis and the transpose
        applies once; otherwise per-term eval.
        """
        if all(s[1] is not None for s in states) and all(
            (s[3], s[4]) == (states[0][3], states[0][4]) for s in states
        ):
            ks = self._stack_terms([s[0] for s in states], ev)
            us = self._row_stack(
                [s[1] for s in states], like, ks.dim() - 1
            )
            kd = ks * us
            r = kd.dim() - 1
            d0, d1 = states[0][3] % r + 1, states[0][4] % r + 1
            return kd.transpose(d0, d1)
        return torch.stack(
            [self._state_row(s, ev, like) for s in states]
        )

    def _forward_cross(
        self, plan: dict, ev: Any, env: dict, x: Any, like: torch.Tensor
    ) -> torch.Tensor | None:
        """Compute the batched ``Σ_j row_ij ⊙ (q_i @ M_j) / S`` terms."""
        pairs = plan["pairs"]
        if not pairs:
            return None
        states = plan["states"]
        a = self._state_stack(states, ev, like)
        vc = self._stack_terms([s[2] for s in states], ev)
        m = torch.matmul(a, vc)  # (J, *batch, e, dv)
        q = self._stack_terms(plan["q_terms"], ev)
        qi = self._index("q", [p[0] for p in pairs], like)
        mi = self._index("m", [p[1] for p in pairs], like)
        prod = torch.matmul(
            q.index_select(0, qi), m.index_select(0, mi)
        )  # (P, *batch, c, dv)
        crank = len(plan["chunk"])
        psc = self._scales("cross", [p[3] for p in pairs], env, x, like)
        if any(p[2] is not None for p in pairs):
            prod = prod * (
                self._row_stack([p[2] for p in pairs], like, crank)
                / psc.reshape(-1, *((1,) * crank))
            )
        else:
            prod = prod / psc.reshape(-1, *((1,) * crank))
        # Deterministic accumulation: pairs are contiguous per target
        # chunk (arm order), so segment sums replace a scatter-add.
        parts = prod.split(plan["pair_counts"])
        return torch.stack([p.sum(0) for p in parts])

    def _forward_impl(self, *xs: torch.Tensor) -> Any:
        """Run the pairwise-batched schedule (or the fallback)."""
        plan = self._plan
        if plan is None:
            return self.eval_mod(*xs)
        x, env = self._input_env(xs)
        # A pre-seeded memo forces ``_eval``'s recursive path, so every
        # shared subterm (the q/k/v projections feeding all chunk
        # splits) evaluates exactly once per forward — the per-term
        # plan tapes would each recompute the shared spine.
        memo: dict[Any, Any] = {None: None}

        def ev(t: Any) -> Any:
            return self.eval_mod._eval(t, env, x, memo)

        like = xs[0]
        out = self._forward_intra(plan, ev, env, x, like)
        cross = self._forward_cross(plan, ev, env, x, like)
        if cross is not None:
            out = out + cross
        # Assemble: (K, *chunk) → concat along the term's own dim.
        dn = plan["dim"] % len(plan["chunk"])
        return out.movedim(0, dn).flatten(dn, dn + 1)


def to_chunked_decayed_module(
    ir: IR, param_values: dict[str, torch.Tensor] | None = None
) -> ChunkedDecayModule:
    """Lower ``ir`` through the chunked-decayed executor."""
    return ChunkedDecayModule(ir, param_values=param_values)

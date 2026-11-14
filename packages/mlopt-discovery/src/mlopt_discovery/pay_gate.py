"""Pay-prediction gate — route measurement budget by structure.

The honest boundary the gate encodes: **plain elementwise workloads
tie ``torch.compile``** — Inductor already fuses the pointwise tail,
so the search + measured-cost referee + delivery timing a probe pays
per workload buys nothing there.  The measured exceptions are
*carrier* structures the rewrite bank can actually move:

* **scan carriers** — the ``seq``/``seqm``/``seqt`` trajectory tiers
  (select-chain recurrences, einsum-over-select read-outs feeding one
  ``stack``).  Measured 1.6-6.3x on the scan zoo workloads
  (``docs/reports/seqt_readout.md``,
  ``docs/reports/module_loss_gla_mamba.md``);
* everything else is *unknown* — measured, not predicted away.

This module is the cheap pre-search predictor.  It reads ONLY the
exported IR term — an op census plus a handful of structural flags —
never shapes, values, or timings, so it costs microseconds next to the
mint + ``profitable`` sweep + board episode it gates.  It is a
heuristic lookup fit on the committed measurements
(``docs/reports/zoo_chain_sweep{,_fp32}.json``,
``docs/reports/frontier.json``, ``docs/reports/seqs_pay_sweep.md``),
not a learned model; ``docs/reports/pay_prediction_gate.md`` carries
the fitted rule, the confusion table, and the honest precision/recall.

Three verdicts:

* ``"pay"`` — structure with a *measured* paying carrier (scan
  recurrences).  These are the workloads budget should reach FIRST;
* ``"skip"`` — the elementwise-bound signature: small graphs made of
  pointwise/view ops and a handful of GEMMs, with no reduction,
  routing, quantisation, convolution, scan or einsum structure.  On
  the measured zoo every such workload lost to eager in at least one
  dtype except one documented straddler (``DynamicTanhBlock``);
* ``"unknown"`` — anything else.  The gate's default: measure it.
  A wrong ``unknown`` costs budget; a wrong ``skip`` loses a payer,
  so the skip rule is deliberately strict and asymmetric.

The gate changes *scheduling only* — it never touches verification,
extraction, or the referee.  Callers opt in explicitly
(``play --pay-gate``, ``zoo_chain --pay-gate``,
``gen_probe(..., pay_gate=True)``); the default everywhere is the
current uniform behaviour.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PAY",
    "SKIP",
    "UNKNOWN",
    "PayVerdict",
    "predict",
    "should_measure",
    "term_features",
]

#: Verdict classes.
PAY = "pay"
SKIP = "skip"
UNKNOWN = "unknown"

#: Ops that do real arithmetic off the pointwise path — GEMM-family,
#: convolutions, attention.  A graph with several of these is not
#: "elementwise-bound" no matter how many ``mul``/``add`` surround them.
_COMPUTE_OPS = frozenset(
    {
        "linear",
        "matmul",
        "mm",
        "mv",
        "bmm",
        "einsum",
        "addmm",
        "addbmm",
        "dot",
        "outer",
        "tensordot",
        "attention",
        "scaled_dot_product_attention",
        "sdpa",
    }
)

#: Ops whose presence means the graph is NOT a plain elementwise tail:
#: reductions (incl. the cumsum family — measured declining everywhere,
#: ``seqs_pay_sweep.md``), routing/top-k dispatch, and the quantised-
#: weight spellings (``round``/``clamp``/``detach`` — the deploy family
#: that measurably pays, e.g. ``STEQuantLinear`` 1.50x fp32).
_REDUCTION_OPS = frozenset(
    {
        "sum",
        "mean",
        "max",
        "min",
        "softmax",
        "logsumexp",
        "var",
        "std",
        "amax",
        "amin",
        "argmax",
        "argmin",
        "prod",
        "cumsum",
        "cumprod",
        "norm",
    }
)

_ROUTE_OPS = frozenset(
    {
        "topk",
        "argsort",
        "sort",
        "argpartition",
        "kthvalue",
        "one_hot",
        "nonzero",
        "searchsorted",
        "unique",
        "isin",
        "cummax",
        "cummin",
    }
)

_QUANT_OPS = frozenset(
    {
        "round",
        "floor",
        "ceil",
        "clamp",
        "clamp_min",
        "clamp_max",
        "detach",
        "detach_",
    }
)

#: The scan-carrier signature, read off the op census alone (no term
#: walk into collector internals): the trajectory tiers mint sites
#: only where per-step states are pulled out by ``select`` and either
#: re-stacked (``seqt`` read-out) or consumed without an einsum
#: read-out (``seqm`` matrix state).  ``select >= 8`` is the recurrence
#: unroll depth the zoo carriers spell; ``ChunkedSSD`` (select=15,
#: einsum=7, cumsum=1) is deliberately excluded — its chunked algebra
#: is a different contract and it measured a loss.
_SCAN_SELECT_MIN = 8

#: Skip-rule bounds, fit on the 67 measured zoo rows (see the report).
_SKIP_MAX_OPS = 24
_SKIP_MAX_COMPUTE = 5
#: Deep spelled-linear stacks (>=3 ``linear`` nodes, no norm/chunk
#: wrapper) are the LoRA / deep-equilibrium payers — exempt from skip.
_SKIP_MAX_LINEAR = 3


@dataclass(frozen=True)
class PayVerdict:
    """One workload's predicted pay class plus the evidence for it."""

    cls: str
    reasons: tuple[str, ...] = ()
    features: dict[str, int] = field(default_factory=dict)

    @property
    def measure(self) -> bool:
        """Whether the gate lets the workload spend search budget."""
        return self.cls != SKIP


def _subterms(term: Any) -> list[Any]:
    """Return every distinct subterm (deduped by identity)."""
    from mlopt_core.ir import Op

    seen: set[int] = set()
    out: list[Any] = []
    stack = [term]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        out.append(t)
        if isinstance(t, Op):
            stack.extend(t.args)
    return out


def _is_conv(op: str) -> bool:
    return op.startswith("conv")


def term_features(term: Any) -> dict[str, int]:
    """Cheap pre-search census of an IR term.

    Returns op counts plus the derived flags the verdict reads.  Works
    on any ``mlopt_core.ir`` term — torch-free, microseconds.
    """
    from mlopt_core.ir import Op

    counts: Counter[str] = Counter()
    for s in _subterms(term):
        if isinstance(s, Op):
            counts[s.op] += 1
    # bucket each op once — conv* counts as BOTH conv and compute
    groups = {"compute": 0, "conv": 0, "red": 0, "route": 0, "quant": 0}
    for op, v in counts.items():
        if op in _COMPUTE_OPS:
            groups["compute"] += v
        elif _is_conv(op):
            groups["conv"] += v
            groups["compute"] += v
        elif op in _REDUCTION_OPS:
            groups["red"] += v
        elif op in _ROUTE_OPS:
            groups["route"] += v
        elif op in _QUANT_OPS:
            groups["quant"] += v
    compute, conv = groups["compute"], groups["conv"]
    red, route = groups["red"], groups["route"]
    quant = groups["quant"]
    n = sum(counts.values())
    return {
        "n_ops": n,
        "compute_ops": compute,
        "conv_ops": conv,
        "reduction_ops": red,
        "route_ops": route,
        "quant_ops": quant,
        "linear_ops": counts.get("linear", 0),
        "matmul_ops": sum(
            counts.get(k, 0)
            for k in ("matmul", "mm", "bmm", "addmm", "addbmm")
        ),
        "einsum_ops": counts.get("einsum", 0),
        "select_ops": counts.get("select", 0),
        "stack_ops": counts.get("stack", 0),
        "scan_ops": counts.get("cumsum", 0) + counts.get("cumprod", 0),
        "transpose_ops": counts.get("transpose", 0)
        + counts.get("permute", 0),
        # everything not GEMM/conv, reduction, or routing — the
        # pointwise + view + quant residue Inductor already fuses
        "other_ops": n - compute - red - route,
    }


def predict(term: Any) -> PayVerdict:
    """Predict the pay class of the workload whose root is *term*.

    The rule order is deliberate: a measured-paying structure wins
    over the skip signature (a scan recurrence also contains
    pointwise ops), and anything ambiguous is ``unknown`` — the gate
    measures it exactly as today.
    """
    return _verdict(term_features(term))


def _pay_reasons(f: dict[str, int]) -> list[str]:
    """Return the measured-paying scan-carrier signatures, if any."""
    sel, stk = f["select_ops"], f["stack_ops"]
    es, cs = f["einsum_ops"], f["scan_ops"]
    out: list[str] = []
    if sel >= _SCAN_SELECT_MIN and stk >= 1:
        out.append(
            f"seqt trajectory: select x{sel} feeding stack x{stk}"
        )
    if sel >= _SCAN_SELECT_MIN and es == 0 and cs == 0 and stk == 0:
        out.append(
            f"seqm matrix-state recurrence: select x{sel}, "
            "no einsum read-out, no cumsum"
        )
    return out


#: Simple count objections: ``(feature, exclusive-upper-bound)`` — the
#: skip applies only when EVERY count stays at or under its bound.
#: ``linear_ops`` uses bound 2 because the rule is ``linear < 3``
#: (deep spelled-linear stacks are exempt payers).
_SKIP_COUNT_BOUNDS = (
    ("n_ops", _SKIP_MAX_OPS),
    ("compute_ops", _SKIP_MAX_COMPUTE),
    ("conv_ops", 0),
    ("reduction_ops", 0),
    ("route_ops", 0),
    ("quant_ops", 0),
    ("linear_ops", _SKIP_MAX_LINEAR - 1),
)


def _skip_objections(f: dict[str, int]) -> list[str]:
    """Reasons the elementwise-bound skip does NOT apply, if any."""
    out: list[str] = []
    for key, bound in _SKIP_COUNT_BOUNDS:
        if f[key] > bound:
            out.append(f"{key}={f[key]} > {bound}")
    # dequant-matmul exemption: one matmul flanked by transpose+mul
    # is the quantised-deployment spelling family (measured payers)
    if f["matmul_ops"] == 1 and f["transpose_ops"] > 0:
        out.append("dequant-matmul shape (matmul+transpose)")
    if (
        f["einsum_ops"]
        or f["select_ops"]
        or f["stack_ops"]
        or f["scan_ops"]
    ):
        out.append(
            f"einsum/select/stack/cumsum present "
            f"(es={f['einsum_ops']} sel={f['select_ops']} "
            f"stk={f['stack_ops']} cs={f['scan_ops']})"
        )
    return out


def _verdict(f: dict[str, int]) -> PayVerdict:
    """Apply the verdict rules to a :func:`term_features` row."""
    reasons = _pay_reasons(f)
    if reasons:
        return PayVerdict(PAY, tuple(reasons), f)
    objections = _skip_objections(f)
    if not objections:
        reasons = [
            f"elementwise-bound: n_ops={f['n_ops']}, "
            f"compute_ops={f['compute_ops']}, no reduction/route/"
            "quant/conv/scan structure"
        ]
        return PayVerdict(SKIP, tuple(reasons), f)
    return PayVerdict(UNKNOWN, tuple(objections), f)


def should_measure(term: Any) -> bool:
    """Whether the gate lets *term*'s workload spend search budget."""
    return predict(term).measure

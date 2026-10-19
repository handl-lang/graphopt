"""Machine-derived op vocabulary — classify corpus ops by property.

The law-discovery pipeline (``mlopt_discovery.pipeline``) derives the
*shapes* it proposes from the corpus (the census's frequent ``f(g, g)``
op-tuples), but takes the *op alphabet* — which ops are pointwise,
which are views — from two hand-written tuples (``_POINTWISE``,
``_VIEW_OPS``).  ``select_mul`` was found only because ``select``
happened to be listed; an op absent from the tables is invisible to
the generator.  This tool removes that last human ingredient: it mines
every op that actually occurs in the corpus (the 22 exported
``mlopt_torch.models`` blocks plus the bench law cases) and classifies
it from a *test*, not a lookup.

Properties (each decided by evaluating the op, not by membership):

* **view** — arity 1, shape-changing, value-preserving: the output's
  elements are a re-indexing (permutation / subset / broadcast) of the
  input's, so the op computes nothing.  Test: ``v(x)`` on a random
  tensor changes shape for some occurrence, and every output value is
  an input value (``set(v(x)) ⊆ set(x)``).
* **pointwise** — shape-preserving *and* commuting with views: the
  output has the input's (broadcast) shape and ``f(v(x), …) =
  v(f(x, …))`` for every view ``v``.  Test: the numeric oracle
  (``mlopt_discovery.proposal._allclose``) on both sides of the naturality.  The
  probe views are shape-agnostic (a flatten, an unsqueeze, a gather, a
  slice, a transpose), so the test applies to any op shape.
* **reduction** — arity ≥ 1 and the op combines elements: the output
  has fewer elements than the input and is not value-preserving.
* **attribute-carrying** — the op carries non-empty attrs in the
  corpus.

Validation (the honesty check): the derived ``pointwise`` / ``view``
sets are compared against the hand-written tables — the pipeline's
``_POINTWISE`` / ``_VIEW_OPS`` and the shipped
``mlopt_core.cost._VIEW_OPS``,
``mlopt_orchestrator.morphisms.signature._VIEW_OPS`` /
``_POINTWISE_OPS`` and ``mlopt_core.laws.layout._POINTWISE_*``.
Agreement on the shared ops and every disagreement are reported; a
disagreement is either a bug in the derivation or a stale entry in the
human table, and the tool says which it believes it is.  The test is
**not** tuned to the tables.

``derive_vocabulary()`` is the seam the pipeline reads
(``law_pipeline --vocab derived``).

The second half of the module is the *need* side of the same question
(plan 0024 part B): the op-classification half reads the vocabulary a
corpus *has*, and :func:`detect_need` / :func:`propose_atom` /
:func:`scan_needs` read the vocabulary the derive bank *lacks*.  Given
a procedural ``derive`` that ``dspec_synth`` refuses, :func:`detect_need`
probes each candidate bank op in a local bank copy and reports the one
that would unblock it (``agreement_gain``, ``rules_blocked``);
:func:`propose_atom` turns that into a ``declare``-shaped proposal — a
minted atom as data, **not** an auto-admission (ADR 0005: proposing is
machine work, admitting stays gated); :func:`scan_needs` runs the whole
carrier law set and ranks the missing ops by how many rules each would
unblock.

Run::

    .venv/bin/python -m mlopt_discovery.vocab
    .venv/bin/python -m mlopt_discovery.vocab --json /tmp/mlopt_discovery.vocab.json

CPU-only, bounded to seconds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any

import torch
from mlopt_core.ir import Op, TensorType, Var
from mlopt_core.laws.cond import _DOPS, _Decline
from mlopt_core.typing import _shape_of

from mlopt_discovery import dspec_synth as _dsp
from mlopt_discovery.census import _iter_subterms

# ``law_impact`` / ``law_shape_census`` own the corpus; reuse them so
# the vocabulary is derived over exactly the graphs the pipeline reads.
from mlopt_discovery.impact import _bench_cases, model_cases
from mlopt_discovery.proposal import _allclose

__all__ = [
    "CANDIDATE_OPS",
    "AtomNeed",
    "CandidateOp",
    "OpClass",
    "Vocabulary",
    "classify",
    "corpus_ops",
    "derive_vocabulary",
    "detect_need",
    "main",
    "propose_atom",
    "scan_needs",
    "validate",
]

#: Numeric-comparison tolerance (fp64).
_TOL = 1e-6

#: Sentinel for an op that could not be evaluated on a probe.
_ERR = object()


# ---------------------------------------------------------------------------
#  Evaluation helpers
# ---------------------------------------------------------------------------


def _rand(shape: tuple) -> Any:
    """Return a positive random fp64 tensor of *shape*.

    Positive values keep ``pow`` / ``rsqrt`` / ``log`` from producing
    ``nan`` (which would make a genuine pointwise op look false).
    """
    return torch.rand(tuple(shape), dtype=torch.float64) + 0.5


def _leaf(shape: tuple, name: str = "x") -> Var:
    """Return a fresh ``Var`` of *shape*."""
    return Var(name, TensorType(tuple(shape)))


def _numel(shape: tuple) -> int:
    """Return the element count of *shape*."""
    n = 1
    for d in shape:
        n *= d
    return n


def _eval(term: Any, env: dict) -> Any:
    """Evaluate *term* through the torch oracle, or ``_ERR``."""
    from mlopt_discovery.proposal import _eval_backend

    try:
        return _eval_backend().eval_term(term, env)
    except Exception:
        return _ERR


def _concrete(shape: Any) -> bool:
    """Return whether *shape* is a concrete all-int tuple.

    ``()`` (a scalar) is concrete: it is a real operand shape (the
    ``const`` exponent of ``pow``, the divisor of ``div``), and a
    pointwise op must still commute with a view of its tensor operand.

    Non-positive dims are NOT concrete: ``_shape_of`` uses ``-1`` as
    its unknown-dim sentinel (e.g. ``expand``'s keep-dim ``-1`` attr
    leaks into the inferred shape), and a ``-1`` fed to ``torch.rand``
    is a crash, not a probe.
    """
    return isinstance(shape, tuple) and all(
        isinstance(d, int) and d >= 0 for d in shape
    )


# ---------------------------------------------------------------------------
#  Corpus op mining
# ---------------------------------------------------------------------------


def corpus_ops(terms: list) -> tuple[dict, dict]:
    """Return ``(occurrences, arity)`` for every op in the corpus.

    ``occurrences[op]`` is the list of ``(attrs, arg_shapes)`` seen,
    deduped by node identity (a shared node counts once).  ``arity[op]``
    is the operand count.
    """
    occ: dict[str, list] = {}
    arity: dict[str, int] = {}
    for t in terms:
        for s in _iter_subterms(t):
            if not isinstance(s, Op):
                continue
            arity[s.op] = len(s.args)
            occ.setdefault(s.op, []).append(
                (dict(s.attrs), [_shape_of(a) for a in s.args])
            )
    return occ, arity


# ---------------------------------------------------------------------------
#  Property tests
# ---------------------------------------------------------------------------


def _value_preserving(x: Any, y: Any) -> bool:
    """Return whether every value of *y* is a value of *x*.

    A view (permutation / subset / broadcast) only re-arranges the
    input's values; a computing op produces values not in the input.
    """
    xs = set(torch.unique(x).tolist())
    ys = set(torch.unique(y).tolist())
    return ys.issubset(xs)


def _is_view(op: str, arity: int, occ: dict) -> bool | None:
    """Return whether *op* is a view, or ``None`` if untestable.

    A view is arity-1, value-preserving on every evaluable occurrence,
    and shape-changing on at least one.
    """
    if arity != 1:
        return False
    good = 0
    tested = 0
    changed = False
    for attrs, shapes in occ[op]:
        sh = shapes[0]
        if not _concrete(sh):
            continue
        v = _leaf(sh)
        x = _rand(sh)
        y = _eval(Op.make(op, v, **attrs), {v: x})
        if y is _ERR or not isinstance(y, torch.Tensor):
            continue
        tested += 1
        if _value_preserving(x, y):
            good += 1
        if tuple(y.shape) != tuple(sh):
            changed = True
    if tested == 0:
        return None
    return good == tested and changed


def _rep_occurrence(op: str, occ: dict) -> tuple | None:
    """Return one ``(attrs, shapes)`` with concrete operand shapes."""
    for attrs, shapes in occ[op]:
        if shapes and all(_concrete(s) for s in shapes):
            return attrs, shapes
    return None


def _battery(shape: tuple) -> list[tuple]:
    """Return shape-agnostic view probes valid on *shape*.

    A flatten (reinterprets the whole element order — kills a reduction
    over any single axis), an unsqueeze, a single-element gather, a
    contiguous subset, and (rank ≥ 2) a first/last transpose.  A
    pointwise op commutes with every one; a view op or a reduction does
    not.
    """
    out = [
        ("reshape", {"shape": (_numel(shape),)}),
        ("unsqueeze", {"dim": 0}),
        ("select", {"dim": 0, "index": 0}),
        ("slice", {"dim": 0, "start": 0, "end": 1}),
    ]
    if len(shape) >= 2:
        out.append(("transpose", {"dim0": 0, "dim1": -1}))
    return out


def _op_output(op: str, attrs: dict, shapes: list) -> Any:
    """Return the output tensor of ``op(*args, **attrs)``, or ``None``."""
    vs = [_leaf(s, f"x{i}") for i, s in enumerate(shapes)]
    env = {v: _rand(s) for v, s in zip(vs, shapes, strict=True)}
    y = _eval(Op.make(op, *vs, **attrs), env)
    if y is _ERR or not isinstance(y, torch.Tensor):
        return None
    return y


def _commutes(
    op: str, attrs: dict, shapes: list, probe: tuple
) -> bool | None:
    """Return whether *op* commutes with *probe* (a view instance).

    ``f(v(x0), …) == v(f(x0, …))`` on random tensors; the view is
    applied to the full-size operands only (a scalar/broadcast operand
    is left alone).  ``None`` when the probe cannot be composed on
    either side (shape-incompatible), which is not evidence either way;
    a one-sided failure is ``False`` (a pointwise op never fails to
    compose).
    """
    vop, vattrs = probe
    vs = [_leaf(s, f"x{i}") for i, s in enumerate(shapes)]
    env = {v: _rand(s) for v, s in zip(vs, shapes, strict=True)}
    big = max(_numel(s) for s in shapes)

    def arg(v: Any, s: tuple) -> Any:
        if _numel(s) == big:
            return Op.make(vop, v, **vattrs)
        return v

    lhs = Op.make(
        op,
        *[arg(v, s) for v, s in zip(vs, shapes, strict=True)],
        **attrs,
    )
    rhs = Op.make(vop, Op.make(op, *vs, **attrs), **vattrs)
    a = _eval(lhs, env)
    b = _eval(rhs, env)
    if a is _ERR and b is _ERR:
        return None
    if a is _ERR or b is _ERR:
        return False
    return _allclose(a, b, _TOL)


def _is_pointwise(op: str, arity: int, occ: dict) -> bool | None:
    """Return whether *op* is pointwise, or ``None`` if untestable.

    Pointwise means: shape-preserving, and commuting with every view
    probe.  Arity > 2 is left undecided (the generator only consumes
    unary / binary ops, and a scalar or mask operand breaks the probe's
    composition).
    """
    if arity not in (1, 2):
        return None
    rep = _rep_occurrence(op, occ)
    if rep is None:
        return None
    attrs, shapes = rep
    out = _op_output(op, attrs, shapes)
    if out is None:
        return None
    big = max(shapes, key=_numel)
    if tuple(out.shape) != tuple(big):
        return False
    results = [_commutes(op, attrs, shapes, p) for p in _battery(big)]
    definite = [r for r in results if r is not None]
    if any(r is False for r in definite):
        return False
    if any(r is True for r in definite):
        return True
    return None


def _is_reduction(op: str, arity: int, occ: dict) -> bool | None:
    """Return whether *op* combines elements (fewer, non-preserved)."""
    rep = _rep_occurrence(op, occ)
    if rep is None:
        return None
    attrs, shapes = rep
    vs = [_leaf(s, f"x{i}") for i, s in enumerate(shapes)]
    env = {v: _rand(s) for v, s in zip(vs, shapes, strict=True)}
    x = env[vs[0]]
    y = _eval(Op.make(op, *vs, **attrs), env)
    if y is _ERR or not isinstance(y, torch.Tensor):
        return None
    if y.numel() >= x.numel():
        return False
    return not _value_preserving(x, y)


# ---------------------------------------------------------------------------
#  Classification record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpClass:
    """One corpus op with its property-derived class."""

    op: str
    arity: int
    pointwise: bool | None
    view: bool | None
    reduction: bool | None
    attribute_carrying: bool
    evidence: str = ""


@dataclass
class Vocabulary:
    """The property-derived op vocabulary over the corpus."""

    classes: tuple[OpClass, ...] = ()
    #: Diagnostics: the shape-agnostic probes the pointwise test used.
    probes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def pointwise(self) -> tuple[str, ...]:
        """Arity-2 pointwise ops (the generator's ``_POINTWISE``)."""
        return tuple(
            c.op
            for c in self.classes
            if c.pointwise is True and c.arity == 2
        )

    @property
    def unary_pointwise(self) -> tuple[str, ...]:
        """Arity-1 pointwise ops."""
        return tuple(
            c.op
            for c in self.classes
            if c.pointwise is True and c.arity == 1
        )

    @property
    def views(self) -> tuple[str, ...]:
        """View ops (the generator's ``_VIEW_OPS``)."""
        return tuple(c.op for c in self.classes if c.view is True)


def classify(terms: list) -> Vocabulary:
    """Classify every op occurring in *terms* by property."""
    torch.manual_seed(0)
    occ, arity = corpus_ops(terms)
    classes: list[OpClass] = []
    for op in sorted(arity):
        n = arity[op]
        classes.append(
            OpClass(
                op=op,
                arity=n,
                pointwise=_is_pointwise(op, n, occ),
                view=_is_view(op, n, occ),
                reduction=_is_reduction(op, n, occ),
                attribute_carrying=any(a for a, _ in occ[op]),
            )
        )
    labels = tuple(
        f"{op}[{','.join(f'{k}={v}' for k, v in sorted(a.items()))}]"
        for op, a in _battery((2, 3, 4))
    )
    return Vocabulary(classes=tuple(classes), probes=labels)


def derive_vocabulary() -> Vocabulary:
    """Derive the vocabulary over the full corpus (the pipeline seam)."""
    bench, _be = _bench_cases()
    models, _me = model_cases()
    terms = [c.term for c in [*bench, *models]]
    return classify(terms)


# ---------------------------------------------------------------------------
#  Validation against the hand-written tables
# ---------------------------------------------------------------------------


def _hand_tables() -> list[tuple[str, str, frozenset]]:
    """Return ``(name, kind, ops)`` for every hand-written op table.

    ``kind`` is ``"view"``, ``"pointwise"`` (binary + unary),
    ``"binary"`` or ``"unary"`` — it selects which derived set the table
    is compared against.
    """
    from mlopt_core.cost import _VIEW_OPS as cost_views
    from mlopt_core.laws import layout
    from mlopt_orchestrator.morphisms import signature

    from mlopt_discovery import pipeline as lpl

    return [
        ("pipeline._POINTWISE", "binary", frozenset(lpl._POINTWISE)),
        ("pipeline._VIEW_OPS", "view", frozenset(lpl._VIEW_OPS)),
        ("cost._VIEW_OPS", "view", frozenset(cost_views)),
        (
            "signature._VIEW_OPS",
            "view",
            frozenset(signature._VIEW_OPS),
        ),
        (
            "signature._POINTWISE_OPS",
            "pointwise",
            frozenset(signature._POINTWISE_OPS),
        ),
        (
            "layout._POINTWISE_BINARY",
            "binary",
            frozenset(layout._POINTWISE_BINARY),
        ),
        (
            "layout._POINTWISE_UNARY",
            "unary",
            frozenset(layout._POINTWISE_UNARY),
        ),
    ]


def _derived_set(vocab: Vocabulary, kind: str) -> frozenset:
    """Return the derived op set matching a table *kind*."""
    if kind == "view":
        return frozenset(vocab.views)
    if kind == "binary":
        return frozenset(vocab.pointwise)
    if kind == "unary":
        return frozenset(vocab.unary_pointwise)
    return frozenset(vocab.pointwise) | frozenset(vocab.unary_pointwise)


def _compare(
    derived: frozenset, hand: frozenset, corpus: frozenset
) -> dict:
    """Compare a derived set with a hand table over the corpus ops."""
    d = derived & corpus
    h = hand & corpus
    return {
        "derived": sorted(d),
        "hand": sorted(h),
        "agree": sorted(d & h),
        "derived_only": sorted(d - h),
        "hand_only": sorted(h - d),
    }


def validate(vocab: Vocabulary) -> dict:
    """Compare the derived vocabulary against every hand table.

    Each derived-only op is tagged ``corroborated`` when it appears in
    some *other* hand table of the same kind — the machine-checkable
    signal that a "disagreement" is the hand table being incomplete
    rather than the derivation being wrong.
    """
    corpus = frozenset(c.op for c in vocab.classes)
    tables = _hand_tables()
    rows: list[dict] = []
    for name, kind, hand in tables:
        row = {"table": name, "kind": kind}
        row.update(_compare(_derived_set(vocab, kind), hand, corpus))
        corroborated = set()
        for other, other_kind, other_ops in tables:
            if other == name or other_kind != kind:
                continue
            corroborated |= other_ops
        row["corroborated"] = sorted(
            op for op in row["derived_only"] if op in corroborated
        )
        rows.append(row)
    return {"corpus_ops": sorted(corpus), "rows": rows}


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _yn(v: bool | None) -> str:
    """Render a tri-state verdict."""
    return {True: "yes", False: "no", None: "?"}[v]


def _class_table(vocab: Vocabulary) -> str:
    """Render the per-op classification table."""
    head = (
        f"{'op':<14} {'ar':>2} {'point':>5} {'view':>5} "
        f"{'reduc':>5} {'attr':>4}"
    )
    lines = [head, "-" * len(head)]
    for c in vocab.classes:
        lines.append(
            f"{c.op:<14} {c.arity:>2} {_yn(c.pointwise):>5} "
            f"{_yn(c.view):>5} {_yn(c.reduction):>5} "
            f"{'yes' if c.attribute_carrying else '-':>4}"
        )
    return "\n".join(lines)


def _validation_table(val: dict) -> str:
    """Render the derived-vs-hand agreement table."""
    head = (
        f"{'hand table':<28} {'kind':<9} {'agree':>5} "
        f"{'derived-only':>12} {'hand-only':>9}"
    )
    lines = [head, "-" * len(head)]
    for r in val["rows"]:
        lines.append(
            f"{r['table']:<28} {r['kind']:<9} {len(r['agree']):>5} "
            f"{len(r['derived_only']):>12} {len(r['hand_only']):>9}"
        )
    return "\n".join(lines)


def _print_report(vocab: Vocabulary, val: dict) -> None:
    """Print the full human-readable vocabulary report."""
    print(  # stdout-compat
        "== law_vocab — the op vocabulary by property =="
    )
    print(  # stdout-compat
        f"   corpus ops: {len(vocab.classes)}; "
        f"pointwise(binary)={len(vocab.pointwise)}, "
        f"pointwise(unary)={len(vocab.unary_pointwise)}, "
        f"views={len(vocab.views)}"
    )
    print()  # stdout-compat
    print("-- derived classification --")  # stdout-compat
    print(_class_table(vocab))  # stdout-compat
    print()  # stdout-compat
    print("-- derived sets --")  # stdout-compat
    print(  # stdout-compat
        f"   pointwise (arity 2): {', '.join(vocab.pointwise)}"
    )
    print(  # stdout-compat
        f"   pointwise (arity 1): {', '.join(vocab.unary_pointwise)}"
    )
    print(  # stdout-compat
        f"   views:               {', '.join(vocab.views)}"
    )
    print()  # stdout-compat
    print(  # stdout-compat
        "-- validation: derived vs hand tables (over corpus ops) --"
    )
    print(_validation_table(val))  # stdout-compat
    for r in val["rows"]:
        if r["derived_only"]:
            tag = (
                " (corroborated by another hand table)"
                if r["corroborated"]
                else ""
            )
            print(  # stdout-compat
                f"   {r['table']}: derived-only "
                f"{', '.join(r['derived_only'])}{tag}"
            )
        if r["hand_only"]:
            print(  # stdout-compat
                f"   {r['table']}: hand-only "
                f"{', '.join(r['hand_only'])}"
            )
    print()  # stdout-compat


def _dump_json(path: str, vocab: Vocabulary, val: dict) -> None:
    """Write the machine-readable vocabulary result."""
    payload = {
        "classes": [
            {
                "op": c.op,
                "arity": c.arity,
                "pointwise": c.pointwise,
                "view": c.view,
                "reduction": c.reduction,
                "attribute_carrying": c.attribute_carrying,
            }
            for c in vocab.classes
        ],
        "pointwise": list(vocab.pointwise),
        "unary_pointwise": list(vocab.unary_pointwise),
        "views": list(vocab.views),
        "probes": list(vocab.probes),
        "validation": val,
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Derive the vocabulary, validate it, and print (or dump) it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    vocab = derive_vocabulary()
    val = validate(vocab)
    _print_report(vocab, val)
    if args.json:
        _dump_json(args.json, vocab, val)
        print(f"wrote {args.json}")  # stdout-compat
    return 0


# ---------------------------------------------------------------------------
#  Need detection — the vocabulary the derive bank LACKS (plan 0024 B)
#
#  ``dspec_synth`` refuses a procedural derive when no expression in the
#  bank reproduces its measured outputs; the refusal names a gap.  This
#  half of the module turns that refusal into a *finding*: which bank op
#  would make the derive expressible (a targeted probe over a local bank
#  copy), and — for a whole rule set — the ranked backlog of missing
#  ops.  A proposal is data, never an admission (ADR 0005).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateOp:
    """A bank op the synthesizer could gain — the probe's hypothesis."""

    name: str
    arity: int
    #: The observable semantics a declared atom would carry.
    semantics: str
    #: When the op declines (its strictness contract).
    decline: str


#: The arity-2 int ops the derive bank could gain.  ``pow`` is pinned
#: by the refusal tests (``2 ** a``); ``mod`` is the axis-normalisation
#: ``dim % rank`` a carrier-aware derive spells — since ``mod`` landed
#: in the bank (``dspec_synth.BINOPS``) the probe finds no gap for it
#: (its agreement equals the base bank's), so the entry now documents
#: the proposal rather than an open need.  Both are ``_DOPS`` members,
#: so the probe's value function is the shipped handler — one
#: semantics, no drift.
CANDIDATE_OPS: tuple[CandidateOp, ...] = (
    CandidateOp(
        "pow",
        2,
        "base ** exp  (int ** int, exp >= 0)",
        "non-int operand / negative exponent / oversized result",
    ),
    CandidateOp(
        "mod",
        2,
        "a % b  (int % int)",
        "non-int operand / zero divisor",
    ),
)


@dataclass(frozen=True)
class AtomNeed:
    """A measured vocabulary gap: the bank op a refused derive needs.

    ``agreement_gain`` is how many more samples the probe reproduces
    with the op than the shipped bank reaches alone; ``rules_blocked``
    names the rules the op makes *fully* expressible (a valid, held-out
    fit) — empty when the op only narrows the gap.
    """

    op: str
    agreement_gain: int
    rules_blocked: tuple[str, ...]
    before: int = 0
    after: int = 0
    samples: int = 0

    def to_data(self) -> dict[str, Any]:
        """Return the finding as JSON-safe data."""
        return {
            "op": self.op,
            "agreement_gain": self.agreement_gain,
            "rules_blocked": list(self.rules_blocked),
            "before": self.before,
            "after": self.after,
            "samples": self.samples,
        }


def _candidate(op: str) -> CandidateOp:
    """Return the :class:`CandidateOp` named *op* (``ValueError`` else)."""
    for c in CANDIDATE_OPS:
        if c.name == op:
            return c
    raise ValueError(f"unknown candidate op: {op!r}")


def _candidate_value(op: str):
    """Return the value-level fn for bank op *op* — ``None`` on decline.

    The op is evaluated through its shipped ``_DOPS`` handler with the
    two operands bound as literals, so the probe's semantics is exactly
    the bank's (the ``_vbin`` value-level twin, for an op the
    synthesizer's ``BINOPS`` does not carry).
    """
    handler = _DOPS[op]

    def value(a: Any, b: Any) -> Any:
        try:
            return handler((("lit", a), ("lit", b)), {})
        except _Decline:
            return None

    return value


def _uses_op(expr: Any, op: str) -> bool:
    """Return whether *expr* mentions *op* anywhere in its tree."""
    if not isinstance(expr, tuple):
        return False
    return expr[0] == op or any(_uses_op(a, op) for a in expr[1:])


def _agreement(sig: tuple, samples: list[tuple]) -> int:
    """Count the samples *sig* reproduces exactly."""
    return sum(
        1 for v, (_b, e) in zip(sig, samples, strict=True) if v == e
    )


def _bin_level(level: list[tuple], op: str, apply: Any) -> list[tuple]:
    """One level of arity-2 compositions over *level* (plus *op*)."""
    out: list[tuple] = []
    for o in (*_dsp.BINOPS, op):
        for (ea, sa), (eb, sb) in product(level, level):
            out.append(
                (
                    (o, ea, eb),
                    tuple(
                        apply(o, a, b)
                        for a, b in zip(sa, sb, strict=True)
                    ),
                )
            )
    return out


def _unary_level(level: list[tuple]) -> list[tuple]:
    """One level of arity-1 and ``len`` compositions over *level*."""
    out: list[tuple] = []
    for o in _dsp.UNOPS:
        for ea, sa in level:
            out.append(((o, ea), tuple(_dsp._vun(o, a) for a in sa)))
    for ea, sa in level:
        out.append((("len", ea), tuple(_dsp._vlen(a) for a in sa)))
    return out


def _candidates_with(
    op: str, samples: list[tuple], *, max_depth: int
) -> list[tuple]:
    """Enumerate ``(expr, sig)`` over the bank plus one extra arity-2 op.

    The local bank copy the probe reads: ``dspec_synth``'s enumerator
    with *op* joined to its ``BINOPS``, so a candidate that needs the
    op at any depth ≤ *max_depth* is reachable.  Deduped by value
    fingerprint exactly as the shipped enumerator is.
    """
    value = _candidate_value(op)

    def apply(o: str, a: Any, b: Any) -> Any:
        return value(a, b) if o == op else _dsp._vbin(o, a, b)

    bound = samples[0][0]
    values = {v for _b, v in samples}
    level = _dsp._dedupe(
        [
            (e, _dsp._signature(e, samples))
            for e in _dsp._leaves(bound, values)
        ]
    )
    out = list(level)
    for _ in range(max_depth):
        nxt = _dsp._dedupe(
            _bin_level(level, op, apply) + _unary_level(level)
        )
        out.extend(nxt)
        level = nxt
    return out


def _probe_attr(
    op: str, samples: list[tuple], *, max_depth: int
) -> tuple[int, bool]:
    """Probe one candidate op against one attr's sample list.

    *fitted* is True iff some op-using candidate is a valid fit —
    reproduces every sample on both the even/odd halves and reads a
    bound value (``dspec_synth``'s own acceptance test), so a
    degenerate constant never counts as an unblock.  *best* is the
    agreement of the closest op-using candidate.
    """
    fit = range(0, len(samples), 2)
    holdout = range(1, len(samples), 2)
    best = 0
    fitted = False
    for expr, sig in _candidates_with(op, samples, max_depth=max_depth):
        if not _uses_op(expr, op):
            continue
        best = max(best, _agreement(sig, samples))
        if not fitted and _dsp._reproduces(
            expr, sig, samples, fit, holdout
        ):
            fitted = True
    return best, fitted


def _probe_op(
    op: str, by_attr: dict, *, max_depth: int
) -> tuple[int, bool]:
    """Probe one candidate op across every attr the hook returns.

    A multi-output hook is unblocked by *op* only when the op makes
    **every** returned attr expressible — a partial unblock is not a
    conversion — so *fitted* is the conjunction over attrs; an attr
    that already fits the shipped bank (``synth_dspec``) counts as
    fitted.  *best* sums each attr's closest op-using agreement.
    """
    total = 0
    fitted = True
    for samples in by_attr.values():
        best, ok = _probe_attr(op, samples, max_depth=max_depth)
        total += best
        if not ok:
            ok = (
                _dsp.synth_dspec(samples, max_depth=max_depth)
                is not None
            )
        fitted = fitted and ok
    return total, fitted


def _hook_and_name(rule_or_hook: Any) -> tuple[Any, str]:
    """Return ``(derive_hook, name)`` for a rule or a bare hook."""
    derive = getattr(rule_or_hook, "derive", None)
    if derive is not None and getattr(rule_or_hook, "name", None):
        return derive, rule_or_hook.name
    if callable(rule_or_hook):
        return rule_or_hook, getattr(rule_or_hook, "__name__", "hook")
    raise ValueError(f"not a rule or a hook: {rule_or_hook!r}")


def detect_need(
    rule_or_hook: Any, domain: list[dict], *, max_depth: int = 2
) -> AtomNeed | None:
    """Characterize the vocabulary gap that blocks a procedural derive.

    *rule_or_hook* is a :class:`Rewrite` (its ``derive`` and ``name``
    are read) or a bare ``derive``-shaped callable.  *domain* is the
    binding environments to measure over.  Returns the
    :class:`AtomNeed` naming the candidate op with the largest
    agreement gain — or ``None`` when the derive is already expressible
    (``synth_dspec`` finds a spec), too thinly sampled to fit, or no
    candidate op helps.  This *characterizes* a need; it admits nothing.

    The derive is measured per attr (:func:`dspec_synth.samples_by_attr`)
    and probed per attr (:func:`_probe_op`): a hook returning several
    attrs is blocked only when some attr lacks a fit, and an op unblocks
    it only when every attr becomes expressible — the same honesty the
    ``dspec`` map itself enforces.
    """
    hook, name = _hook_and_name(rule_or_hook)
    bounds = list(domain)
    by_attr = _dsp.samples_by_attr(hook, bounds)
    total = sum(len(s) for s in by_attr.values())
    if total < _dsp.MIN_SAMPLES:
        return None
    if (
        _dsp.synth_dspec_multi(hook, bounds, max_depth=max_depth)
        is not None
    ):
        return None
    before = _baseline(by_attr, max_depth=max_depth)
    found: list[AtomNeed] = []
    for cand in CANDIDATE_OPS:
        after, fitted = _probe_op(
            cand.name, by_attr, max_depth=max_depth
        )
        if after <= before:
            continue
        found.append(
            AtomNeed(
                op=cand.name,
                agreement_gain=after - before,
                rules_blocked=(name,) if fitted else (),
                before=before,
                after=after,
                samples=total,
            )
        )
    if not found:
        return None
    return max(found, key=lambda n: (n.agreement_gain, n.op))


def _baseline(by_attr: dict, *, max_depth: int) -> int:
    """Best agreement the shipped bank already reaches, summed per attr."""
    total = 0
    for samples in by_attr.values():
        near = _dsp._near_misses(samples, max_depth=max_depth, k=1)
        total += near[0][1] if near else 0
    return total


#: Input pairs the observable-semantics probe evaluates (small ints).
_PROBE_INPUTS: tuple[tuple[int, int], ...] = (
    (2, 3),
    (3, 2),
    (5, 0),
    (7, 3),
    (0, 4),
)


def _observations(op: str) -> list[dict[str, Any]]:
    """Return the op's ``{in, out}`` pairs — its observable semantics.

    Evaluated through the shipped ``_DOPS`` handler on real ints, so a
    proposal carries *what the atom computes*, not just a docstring
    (``None`` marks a decline — the strictness contract).
    """
    value = _candidate_value(op)
    return [
        {"in": [a, b], "out": value(a, b)} for a, b in _PROBE_INPUTS
    ]


def propose_atom(need: AtomNeed) -> dict[str, Any]:
    """Emit a ``declare``-shaped proposal for *need*'s missing op.

    The vocabulary-minting path as data: the op becomes a declared atom
    with an observable semantics, wrapped in the ``declare`` move's
    shape (``{"op": "declare", "params": {...}}``) and carrying the
    measured evidence.  ``admitted`` is always ``False`` — proposing is
    machine work, admitting stays gated (ADR 0005); a human or the
    gauntlet acts on the proposal, it does not auto-land.
    """
    decl = _candidate(need.op)
    return {
        "op": "declare",
        "params": {
            "construction": {
                "op": "atom",
                "params": {
                    "kind": "derive-bank-op",
                    "name": decl.name,
                    "arity": decl.arity,
                    "semantics": decl.semantics,
                    "decline": decl.decline,
                    "observable": _observations(decl.name),
                },
            },
            "unfold": False,
        },
        "evidence": need.to_data(),
        "admitted": False,
    }


def scan_needs(
    rules: Any,
    *,
    corpus: Any = (),
    synth_limit: int = 60,
    max_depth: int = 2,
    deadline: float | None = None,
    max_rules: int | None = None,
    measure: Any = None,
) -> list[AtomNeed]:
    """Rank the missing bank ops by how many rules each would unblock.

    *rules* is an iterable of :class:`Rewrite`; every residual
    procedural derive (``dspec is None`` with a ``derive`` hook) is
    measured over its domain (``object_synthesis._measure_domain`` with
    ``deep=True`` — the carrier op-term window, so a derive that reads
    the carrier-aware value shape is measurable — or an injected
    *measure* of the same shape) and probed.  The per-op
    needs are aggregated — ``rules_blocked`` unions the rules an op
    fully unblocks, ``agreement_gain`` sums the gains — and sorted by
    unblock count then gain.  *deadline* (seconds) and *max_rules*
    bound the scan; a partial result is honest.
    """
    import time

    if measure is None:
        from mlopt_discovery import object_synthesis as osyn

        measure = partial(osyn._measure_domain, deep=True)

    start = time.monotonic()
    found: list[AtomNeed] = []
    seen = 0
    for rule in rules:
        if max_rules is not None and seen >= max_rules:
            break
        if (
            deadline is not None
            and time.monotonic() - start >= deadline
        ):
            break
        if (
            getattr(rule, "dspec", None) is not None
            or getattr(rule, "derive", None) is None
        ):
            continue
        seen += 1
        dom = measure(rule, corpus, synth_limit)
        need = detect_need(
            rule, [env for env, _outcome in dom], max_depth=max_depth
        )
        if need is not None:
            found.append(need)
    return _rank_needs(found)


def _rank_needs(needs: list[AtomNeed]) -> list[AtomNeed]:
    """Aggregate per-rule needs into a per-op ranked backlog."""
    groups: dict[str, list[AtomNeed]] = {}
    for n in needs:
        groups.setdefault(n.op, []).append(n)
    ranked: list[AtomNeed] = []
    for op, group in groups.items():
        ranked.append(
            AtomNeed(
                op=op,
                agreement_gain=sum(n.agreement_gain for n in group),
                rules_blocked=tuple(
                    sorted({r for n in group for r in n.rules_blocked})
                ),
                before=sum(n.before for n in group),
                after=sum(n.after for n in group),
                samples=sum(n.samples for n in group),
            )
        )
    ranked.sort(
        key=lambda n: (-len(n.rules_blocked), -n.agreement_gain, n.op)
    )
    return ranked


if __name__ == "__main__":
    raise SystemExit(main())

"""``handl_egraph`` — plan 0033 phase 5: handl-native saturation.

Phases 0-4 landed the fragment tower's data and differential seats:
``vocab/`` is the canonical vocabulary, :class:`HandlEngine` replays
every Python-side firing through handl's ``RewriteEngine`` as a
*differential oracle*, and :class:`FragmentArena` seats fragment-
application moves on the board.  Phase 5 is the plan's last seat:
**the e-graph saturates natively on the handl side** — the
``handl-egraph`` crate's ``EGraph::run`` (reached through
``mlopt-kit``'s ``egraph-run`` binary, the same door
``bench/drivers/audit/egraph_differential.py`` diffs whole runs
through) drives saturation over the *fragment vocabulary's* rewrite
rows, instead of Python ``EGraph.run`` plus parity replay.

What "native" means here, honestly:

* the **rules are the fragments** — ``vocab/laws.yaml`` +
  ``vocab/schemas.yaml`` rows (``law``/``prim:*``/``fold:*``), the
  same :class:`~mlopt_core.egraph.Rewrite` objects phase 4 applies
  one-at-a-time, encoded once into the ``egraph-run`` JSON spec
  (``{"mvar"}/``{"leaf"}/``{"op"}`` wire forms — the codec
  ``handl-discovery``'s ``term_from_data`` reads, mirrored field for
  field by :func:`spec_term` below and pinned equal to the audit
  driver's encoder in tests);
* the **saturation is Rust** — matching, union, congruence closure,
  frontier scheduling, budgets, extraction all run inside
  ``handl_egraph::EGraph``; Python never sees an intermediate
  e-class;
* the **verification is the differential report** — the Rust side
  emits the canonical engine-independent report (WL class signatures,
  member multisets, stats, provenance multisets, extracted best) that
  ``bench/drivers/audit/egraph_differential.py`` compares against the
  Python ``EGraph`` run on the *same* spec, field by field.  A
  divergence is a loud finding, never a silent difference.

Boundaries, stated (the same honesty classes the audit driver and
phase 1 already carry):

* **Procedural hooks stay Python.**  ``check``/``derive`` are Python
  closures — inexpressible on the wire — so :func:`vocab_rewrites`
  splits the bank: declarative rows (no ``check``/``derive`` at all)
  saturate natively; hooked rows are *counted and named* in
  ``VocabRewrites.skipped``, never silently dropped.  ``cond``/
  ``dspec`` data folds into those hooks at ``Rewrite`` construction,
  so a data-carried guard lands in the skipped set too — migrating
  guards toward typecheck-nbe predicates is the plan's dependent-
  surface work, still open.
* **``op``/``model``/``family`` rows are not moves** — phase 4's
  catalogue boundary is unchanged: they are vocabulary data
  (AST shapes, ``.hnl`` denotations, mint provenance), not rewrite
  fragments, so they never enter a saturation spec.
* **Leaf identity is ``repr`` on both sides** — ``EGraph.add_term``
  keys leaves on ``repr(term)`` and the wire's ``{"leaf": key}`` is
  the same string, so typed ``Var``/``Param`` leaves stay consistent
  (``Const``'s spelling-strict identity is preserved verbatim).
* **ε-laws (``error_bound``)** ride the spec as metadata-free rules —
  the bound does not change firing; it is provenance the report does
  not compare.  Counted in :attr:`VocabRewrites.epsilon` for honesty.

Peer surface: the compiled ``egraph-run`` binary (``mlopt-kit``,
``src/bin/egraph-run.rs``) — env override ``MLOPT_EGRAPH_RUN`` (the
pre-rename ``CATOPT_EGRAPH_RUN`` honoured), then the repo
``target/debug`` and ``~/.cargo/shared-target/debug`` candidates, the
same resolution order as the differential driver and
``tests/test_egraph_differential.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlopt_core.egraph import Rewrite

from mlopt_discovery import vocabdir

__all__ = [
    "EgraphRunError",
    "HandlEgraphUnavailable",
    "VocabRewrites",
    "build_spec",
    "egraph_run_bin",
    "run_native",
    "saturate",
    "spec_term",
    "vocab_rewrites",
]

_REPO = Path(__file__).resolve().parents[4]

#: The ``egraph-run`` binary — env override first (``MLOPT_EGRAPH_RUN``,
#: ``CATOPT_EGRAPH_RUN`` the pre-rename spelling), then the two target
#: dirs a ``cargo build`` lands in on this host.  Same candidates as
#: ``bench/drivers/audit/egraph_differential.py::_bin`` — keep in
#: lockstep.
_BIN_CANDIDATES = [
    _REPO / "target" / "debug" / "egraph-run",
    Path.home() / ".cargo" / "shared-target" / "debug" / "egraph-run",
]


class HandlEgraphUnavailable(RuntimeError):
    """The ``egraph-run`` binary is not built / not found."""


class EgraphRunError(RuntimeError):
    """``egraph-run`` ran and failed (nonzero exit or bad JSON)."""


def egraph_run_bin() -> Path | None:
    """Locate the compiled ``egraph-run``; ``None`` when unbuilt."""
    env = os.environ.get("MLOPT_EGRAPH_RUN") or os.environ.get(
        "CATOPT_EGRAPH_RUN"
    )
    if env:
        return Path(env)
    for cand in _BIN_CANDIDATES:
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------------
#  Term -> spec encoding — mirrors bench/drivers/audit/egraph_differential.py's
#  ``_spec_term``/``_spec_attr`` field for field (the codec
#  ``handl-discovery``'s ``term_from_data`` reads).  Pinned equal in
#  tests; keep in lockstep with both.
# ---------------------------------------------------------------------------


def _spec_attr(v: Any, pattern: bool) -> Any:
    """Encode an attr value; in patterns a ``str`` is an attr metavar."""
    if isinstance(v, str) and pattern:
        return {"amvar": v}
    if isinstance(v, (tuple, list)):
        return {"__tuple__": [_spec_attr(x, pattern) for x in v]}
    return v


def spec_term(t: Any, *, pattern: bool = False) -> dict:
    """Encode a term (concrete or pattern) in the ``egraph-run`` wire form.

    ``{"mvar": name}`` for a pattern metavariable (a bare ``str``
    leaf), ``{"leaf": repr(t)}`` for any other leaf — ``repr`` is the
    Python e-graph's own leaf key (``EGraph.add_term``), so the wire
    form is faithful by construction — and ``{"op", "args", "attrs"}``
    for :class:`~mlopt_core.ir.Op` nodes.  In pattern position a
    ``str`` attr encodes ``{"amvar": name}`` and composite attrs ride
    ``{"__tuple__": [...]}`` (the Rust side's untuple normalisation
    canonicalises them to enode tuples, matching Python's list→tuple
    attr normalisation).
    """
    from mlopt_core.ir import Op

    if isinstance(t, Op):
        return {
            "op": t.op,
            "args": [spec_term(a, pattern=pattern) for a in t.args],
            "attrs": {
                k: _spec_attr(v, pattern) for k, v in t.attrs.items()
            },
        }
    if isinstance(t, str):
        return {"mvar": t}
    return {"leaf": repr(t)}


# ---------------------------------------------------------------------------
#  The fragment vocabulary -> the declarative rewrite subset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VocabRewrites:
    """The saturation-ready split of the ``vocab/`` rewrite bank.

    ``rules`` are the :class:`Rewrite` rows the wire can express —
    ``check``/``derive`` both ``None`` — spanning the ``law`` rows,
    the minted ``prim:*`` folds, and the ``fold:*``/schema rows (the
    same three kinds :mod:`mlopt_discovery.fragment_game` seats as
    moves).  ``skipped`` names every excluded row with its hook
    reasons — the honest boundary, counted not hidden.
    ``epsilon`` counts declarative rows carrying ``error_bound``
    (certified approximations — they saturate identically; the bound
    is provenance the wire does not carry).
    """

    rules: tuple[Rewrite, ...]
    skipped: tuple[tuple[str, tuple[str, ...]], ...]
    epsilon: int
    directory: Path

    @property
    def n_skipped(self) -> int:
        """Rows the wire cannot express."""
        return len(self.skipped)


def vocab_rewrites(
    directory: str | Path | None = None,
) -> VocabRewrites:
    """Split the canonical vocabulary's rewrite rows for native saturation.

    ``directory=None`` resolves :func:`vocabdir.default_vocab_dir`
    (the repo ``vocab/`` or ``MLOPT_VOCAB_DIR``).  Raises
    ``ValueError`` when no directory resolves — saturation over a
    vocabulary that does not exist is not saturation.
    """
    resolved = (
        Path(directory)
        if directory is not None
        else vocabdir.default_vocab_dir()
    )
    if resolved is None:
        raise ValueError(
            "no vocab directory — pass directory or set MLOPT_VOCAB_DIR"
        )
    vocab = vocabdir.load_vocab(resolved)
    # ``law``/``prim`` live in ``v.laws`` (``prim:``-prefixed names);
    # ``schema`` rows are ``v.schemas`` — fragment_game's split.
    rows: list[Rewrite] = [*vocab.laws, *vocab.schemas]
    rules: list[Rewrite] = []
    skipped: list[tuple[str, tuple[str, ...]]] = []
    epsilon = 0
    for rule in rows:
        reasons: list[str] = []
        if rule.check is not None:
            reasons.append("check")
        if rule.derive is not None:
            reasons.append("derive")
        if reasons:
            skipped.append((rule.name, tuple(reasons)))
            continue
        if rule.error_bound is not None:
            epsilon += 1
        rules.append(rule)
    return VocabRewrites(
        rules=tuple(rules),
        skipped=tuple(skipped),
        epsilon=epsilon,
        directory=vocab.directory,
    )


# ---------------------------------------------------------------------------
#  The run spec + the subprocess door
# ---------------------------------------------------------------------------


def build_spec(
    name: str,
    terms: list[Any],
    rules: list[Rewrite] | tuple[Rewrite, ...],
    *,
    root: int | None = None,
    options: dict | None = None,
    costs: dict | None = None,
    check_terms: list[Any] | None = None,
) -> dict:
    """Encode a saturation run as the ``egraph-run`` JSON spec.

    The spec shape is ``mlopt_kit::egraph::run_spec``'s contract —
    ``terms`` are the concrete terms to intern (all of them saturate
    together; ``root`` selects the extraction root, default the last
    interned term), ``rules`` the ``{"name", "lhs", "rhs"}`` pattern
    pairs, ``options`` the run knobs (``max_iterations`` /
    ``max_nodes`` / ``patience`` / ``stop`` / ``rule_budgets``),
    ``costs`` the per-op additive price table, and ``check_terms``
    post-run membership probes.  Field-for-field the same spec
    ``bench/drivers/audit/egraph_differential.py::spec_for`` emits.
    """
    spec: dict[str, Any] = {
        "name": name,
        "terms": [spec_term(t) for t in terms],
        "root": len(terms) - 1 if root is None else root,
        "rules": [
            {
                "name": r.name,
                "lhs": spec_term(r.lhs, pattern=True),
                "rhs": spec_term(r.rhs, pattern=True),
            }
            for r in rules
        ],
        "options": dict(options or {}),
        "costs": dict(costs or {}),
    }
    if check_terms:
        spec["check_terms"] = [spec_term(t) for t in check_terms]
    return spec


def run_native(spec: dict, *, timeout: float = 300.0) -> dict:
    """Drive ``egraph-run`` on *spec*; return the parsed report.

    Raises :class:`HandlEgraphUnavailable` when no binary resolves and
    :class:`EgraphRunError` on a nonzero exit, a timeout, or
    unparseable output — a native run that cannot be verified is a
    loud failure, never a silent fallback to Python.
    """
    bin_path = egraph_run_bin()
    if bin_path is None or not Path(bin_path).exists():
        raise HandlEgraphUnavailable(
            "egraph-run not available — build it:\n"
            "  cargo build -p mlopt-kit --bin egraph-run\n"
            "(or set MLOPT_EGRAPH_RUN to a built binary)"
        )
    try:
        out = subprocess.run(
            [str(bin_path)],
            input=json.dumps(spec),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise EgraphRunError(
            f"egraph-run timed out after {timeout}s"
        ) from e
    except OSError as e:
        raise EgraphRunError(f"egraph-run spawn failed: {e}") from e
    if out.returncode != 0:
        raise EgraphRunError(
            f"egraph-run failed ({out.returncode}): "
            f"{out.stderr.strip()[:400]}"
        )
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError as e:
        raise EgraphRunError(f"egraph-run bad JSON: {e}") from e


def saturate(
    terms: list[Any],
    rules: list[Rewrite] | tuple[Rewrite, ...],
    *,
    name: str = "saturation",
    root: int | None = None,
    options: dict | None = None,
    costs: dict | None = None,
    check_terms: list[Any] | None = None,
    timeout: float = 300.0,
) -> dict:
    """Saturate *terms* under *rules* natively; return the Rust report.

    One call = one ``egraph-run`` invocation: the handl-egraph crate
    interns every term, runs ``EGraph::run`` over the fragment
    rewrites, and reports the canonical engine-independent document
    (WL signatures, member multisets, stats, provenance, extracted
    ``best``).  Pair with the Python ``EGraph`` run on the same spec —
    ``bench/drivers/audit/egraph_differential.py``'s comparison is the
    parity oracle; divergence is a finding, not a fallback.
    """
    spec = build_spec(
        name,
        terms,
        rules,
        root=root,
        options=options,
        costs=costs,
        check_terms=check_terms,
    )
    return run_native(spec, timeout=timeout)

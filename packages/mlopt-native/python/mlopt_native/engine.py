"""The ``Engine``-conforming adapter over the Rust search core.

``mlopt_native._native.NativeEGraph`` is the compiled half — union-
find, enode storage, compiled matching, rebuild/congruence and the
dirty-frontier saturation loop, all a faithful port of
``mlopt_core.egraph.core.EGraph``'s *search* surface.  This module is
the thin Python half:

* :func:`_flatten` / :func:`_ser_pattern` — the serialization layer
  (terms and patterns cross the FFI as plain tuples);
* :func:`_build_term` — the ``build_term`` callback the Rust side
  calls to materialise a bound metavariable's resolved member (so
  ``check``/``derive`` see real ``Op``/leaf objects, exactly like the
  Python engine);
* :class:`NativeEngine` — the :class:`~mlopt_core.ports.Engine` the
  pipeline consumes: same ``add_term`` / ``find`` / ``run`` /
  ``extract_best`` surface, plus a lazily materialised ``_classes``
  view that lets the *reference* extraction mixin
  (:class:`mlopt_core.egraph.extract._ExtractMixin`) run unmodified —
  extraction semantics are the Python implementation's *by
  construction*, so the extracted term and cost are identical.

Scope boundary (plan 0010, lever 3): no proof machinery — ``union``
accepts and drops ``rule``/``subst``/``witness``/``note``; there is no
``merge_log``/``applications``/``certificate``.  A run that needs a
certificate uses the Python engine.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

from mlopt_core.cost import (
    _SOLVER_FACTOR,
    _SOLVER_OPS,
    _local_roofline,
    _memo_dispatch,
    _profile_constants,
    fusion_member_key,
)
from mlopt_core.egraph.extract import _ExtractMixin
from mlopt_core.egraph.types import (
    EClass,
    ENode,
    _LeafRegistry,
    _norm_attr_value,
)
from mlopt_core.ir import Const, Op, Param, Var
from mlopt_core.opmeta import FOLD_OPS

from mlopt_native._native import NativeEGraph as _RustEGraph
from mlopt_native._native import dag_cost as _rust_dag_cost

logger = logging.getLogger("mlopt_core.egraph.extract")

__all__ = ["NativeEngine", "dag_cost"]


def _flatten(term: Any, nodes: list, memo: dict) -> int:
    """Append *term*'s DAG to *nodes* post-order; return its index.

    Shared subtrees intern once through *memo* — exported IR terms are
    DAGs with heavy sharing, so walking without it re-expands shared
    cones exponentially.
    """
    if term in memo:
        return memo[term]
    if isinstance(term, Op):
        idxs = tuple(_flatten(a, nodes, memo) for a in term.args)
        attrs = tuple(
            sorted(
                (k, _norm_attr_value(v)) for k, v in term.attrs.items()
            )
        )
        nodes.append(("op", term.op, idxs, attrs))
    else:
        _LeafRegistry.register(term)
        nodes.append(("leaf", repr(term)))
    memo[term] = len(nodes) - 1
    return memo[term]


def _leaf_kind(term: Any) -> int:
    """Classification bits for ``dag_cost``'s flat leaf nodes."""
    if isinstance(term, Var):
        return 1
    if isinstance(term, Param):
        return 2
    if isinstance(term, Const):
        return 4
    return 0


def _flatten_dag(
    term: Any, nodes: list, terms: list, memo: dict
) -> int:
    """``dag_cost``'s flat serialization: post-order, terms alongside.

    Leaf nodes are ``("leaf", key, kind)`` — ``kind`` is
    :func:`_leaf_kind`'s Var/Param/Const bitfield — and op nodes
    ``("op", name, (child idxs), attrs)``.  ``terms[i]`` is the
    original Python object for node ``i`` — the pricer calls
    ``cfn(terms[i])``, the same ``c(t)`` calls the reference walk
    issues (interned terms hit the shared cost memo).
    """
    if term in memo:
        return memo[term]
    if isinstance(term, Op):
        idxs = tuple(
            _flatten_dag(a, nodes, terms, memo) for a in term.args
        )
        attrs = tuple(
            sorted(
                (k, _norm_attr_value(v)) for k, v in term.attrs.items()
            )
        )
        nodes.append(("op", term.op, idxs, attrs))
    else:
        _LeafRegistry.register(term)
        nodes.append(("leaf", repr(term), _leaf_kind(term)))
    terms.append(term)
    memo[term] = len(nodes) - 1
    return memo[term]


def dag_cost(
    term: Any, cost_fn: Any, memo: dict | None = None
) -> float:
    """Return the true-DAG cost of an extracted term — the walk in Rust.

    Identical contract to :func:`mlopt_core.cost.dag_cost`: each
    distinct node bills ``c(t) - Σ c(children)`` once, param-only
    foldable subtrees price at zero unless the model sets
    ``charges_param_only``, and ``dag_exact`` models return ``c(root)``
    verbatim.  ``cost_fn`` stays a plain Python callable — the Rust
    core drives the structural walk (``has_var_leaf`` /
    ``_folds_to_param`` / reachability) and calls back once per priced
    node.
    """
    unwrapped = getattr(cost_fn, "func", cost_fn)
    c = _memo_dispatch(cost_fn, memo)
    if getattr(unwrapped, "dag_exact", False):
        return c(term)
    nodes: list = []
    terms: list = []
    _flatten_dag(term, nodes, terms, {})
    return _rust_dag_cost(
        nodes,
        lambda i: c(terms[i]),
        getattr(unwrapped, "charges_param_only", False),
        FOLD_OPS,
    )


def _ser_pattern(p: Any) -> tuple:
    """Serialize a pattern subtree (metavariables → ``("var", n)``)."""
    if isinstance(p, str):
        return ("var", p)
    if isinstance(p, Op):
        return (
            "op",
            p.op,
            tuple(_ser_pattern(a) for a in p.args),
            tuple(
                (k, _norm_attr_value(v))
                for k, v in sorted(p.attrs.items())
            ),
        )
    _LeafRegistry.register(p)
    return ("leaf", repr(p))


def _build_term(ser: Any) -> Any:
    """Materialise a serialized term — the Rust ``build_term`` hook."""
    if ser[0] == "leaf":
        return _LeafRegistry.decode(ser[1])
    _tag, op, children, attrs = ser
    return Op.make(
        op, *(_build_term(c) for c in children), **dict(attrs)
    )


class NativeEngine(_ExtractMixin):
    """The native search engine — Rust core + reference extraction.

    Satisfies :class:`mlopt_core.ports.Engine` and (except proof
    machinery) the ``EGraph`` surface ``optimize.search`` consumes:
    ``add_term`` / ``find`` / ``rebuild`` / ``run`` / ``extract_best``
    / ``rule_fires`` / ``n_enodes`` / ``n_classes`` / ``_classes``.

    ``engine_name = "native"`` is what ``stats["engine"]`` records.
    """

    engine_name = "native"
    #: Honest reporting: the native core stores no proof witnesses.
    truncation_level = 1

    def __init__(self) -> None:
        """Create the Rust core with the ``build_term`` hook."""
        self._g = _RustEGraph(_build_term)
        self._view: dict[int, EClass] | None = None

    # -- internals -----------------------------------------------------

    def _invalidate(self) -> None:
        """Drop the materialised ``_classes`` view after a mutation."""
        self._view = None

    # -- engine surface --------------------------------------------------

    @property
    def n_enodes(self) -> int:
        """Return the number of enodes."""
        return self._g.n_enodes

    @property
    def n_classes(self) -> int:
        """Return the number of e-classes."""
        return self._g.n_classes

    @property
    def rule_fires(self) -> dict[str, int]:
        """Return ``{rule_name: merge_count}``."""
        return dict(self._g.rule_fires())

    def certificate(
        self,
        src_term: Any,
        dst_term: Any = None,
        *,
        root_eid: int | None = None,
        cost_fn: Any = None,
    ) -> Any:
        """Certificates are a Python-engine capability — raise.

        The native core records no merge log / applications
        (``truncation_level = 1``), so there is nothing to certify.
        A run needing a certificate uses the Python engine.
        """
        raise NotImplementedError(
            "certificates require the Python EGraph engine — "
            "the native core records no proof witnesses"
        )

    def find(self, eid: int) -> int:
        """Return the canonical e-class id of ``eid``."""
        return self._g.find(eid)

    def get_class(self, eid: int) -> EClass:
        """Return the e-class containing ``eid`` (from the view)."""
        return self._classes[self.find(eid)]

    def add_term(
        self,
        term: Any,
        _memo: dict | None = None,
        provenance: str = "input",
    ) -> int:
        """Add a term (Var/Const/Param/Op); return its e-class id."""
        nodes: list = []
        _flatten(term, nodes, {} if _memo is None else _memo)
        self._invalidate()
        return self._g.add_term(nodes)

    def add_leaf(self, key: str, provenance: str | None = None) -> int:
        """Add a leaf by registry key."""
        self._invalidate()
        return self._g.add_leaf(key)

    def union(
        self,
        a: int,
        b: int,
        rule: str | None = None,
        subst: dict | None = None,
        note: str = "",
        witness: Any = None,
    ) -> bool:
        """Merge two e-classes.

        Proof metadata (``rule``/``subst``/``note``/``witness``) is
        accepted for signature parity and dropped — the native core is
        search-only; certificates stay a Python-engine concern.
        """
        self._invalidate()
        return self._g.union(a, b)

    def rebuild(self, classes: Any = None) -> bool:
        """Canonicalize children and merge duplicates."""
        self._invalidate()
        ids = None if classes is None else [int(c) for c in classes]
        return self._g.rebuild(ids)

    def _register_rules(self, rules: Any) -> list[str]:
        """Compile every rule into the Rust core; return ordered names.

        A ``priority_of`` map (a ``RuleSet``) schedules each iteration
        exactly as the Python engine does (stable sort).
        """
        ordered = list(rules)
        priority_of = getattr(rules, "priority_of", None)
        if priority_of is not None:
            ordered.sort(key=priority_of)
        for r in ordered:
            self._g.add_rule(
                r.name,
                _ser_pattern(r.lhs),
                _ser_pattern(r.rhs),
                check=r.check,
                derive=r.derive,
            )
        return [r.name for r in ordered]

    def _improving_cost(self, root_eid: int, cost_fn: Any) -> float:
        """Re-extract under the mid-run graph and price it."""
        self._invalidate()
        term = self.extract_best(root_eid, cost_fn)
        return cost_fn(term) if term is not None else float("inf")

    def _run_loop(
        self,
        names: list[str],
        root_eid: int,
        max_iterations: int,
        cap: int,
        budgets: dict | None,
        stop: str,
        patience: int,
        cost_fn: Any,
    ) -> tuple[str, int, int]:
        """Drive ``run_iteration``; return ``(stop, iterations, improved)``.

        The loop runs here rather than inside the extension: each
        ``run_iteration`` is a complete FFI call, so the "improving"
        policy can re-extract between iterations through the reference
        mixin without tripping PyO3's reentrant-borrow guard.
        """
        stop_reason = "max_iterations"
        best_cost = float("inf")
        stall = 0
        improved = 0
        iterations = 0
        for _ in range(max_iterations):
            iterations += 1
            self._invalidate()
            reason = self._g.run_iteration(
                names,
                max_nodes=cap,
                rule_budgets=budgets,
            )
            if reason != "continue":
                stop_reason = reason
                break
            if stop == "improving":
                cost = self._improving_cost(root_eid, cost_fn)
                if cost < best_cost:
                    best_cost = cost
                    stall = 0
                    improved += 1
                else:
                    stall += 1
                    if stall >= patience:
                        stop_reason = "improving"
                        break
        # Postcondition: canonicalised graph (unrestricted rebuild).
        self._invalidate()
        self._g.finish()
        return stop_reason, iterations, improved

    def _run_stats(
        self,
        budgets: dict | None,
        stop_reason: str,
        iterations: int,
        improved: int,
        improving: bool,
    ) -> dict[str, Any]:
        """Assemble the ``EGraph.run`` stats record."""
        spent = dict(self._g.budget_spent())
        budget_keys = budgets or {}
        stats: dict[str, Any] = {
            "iterations": iterations,
            "n_enodes": self.n_enodes,
            "n_classes": self.n_classes,
            # Stats-shape parity with EGraph.run — the native core
            # records no proof witnesses (Python-engine concern).
            "n_proof_edges": 0,
            "truncation_level": 1,
            "rule_budgets": {n: spent.get(n, 0) for n in budget_keys},
            "budget_suspended": [
                n
                for n, b in budget_keys.items()
                if spent.get(n, 0) >= b
            ],
            "stop": stop_reason,
        }
        if improving:
            stats["improved"] = improved
        return stats

    def run(
        self,
        rules: Any,
        root_eid: int,
        max_iterations: int = 100,
        max_nodes: int = 100_000,
        rule_budgets: dict[str, int] | None = None,
        stop: str = "fixed_point",
        patience: int = 3,
        cost_fn: Any = None,
    ) -> dict[str, Any]:
        """Run equality saturation — mirrors ``EGraph.run``.

        ``rules`` is any ``RuleSetLike`` iterable of ``Rewrite``s;
        a ``priority_of`` map (a ``RuleSet``) schedules each iteration
        exactly as the Python engine does.  ``stop="improving"``
        extracts through the Python mixin between iterations (the
        reference extractor — identical terms and costs).
        """
        if stop not in ("fixed_point", "improving"):
            raise ValueError(
                f"stop must be 'fixed_point' or 'improving', "
                f"got {stop!r}"
            )
        if stop == "improving" and cost_fn is None:
            raise ValueError(
                "stop='improving' needs a cost_fn to extract with"
            )
        names = self._register_rules(rules)
        cap = max_nodes if max_nodes is not None else sys.maxsize
        budgets = dict(rule_budgets) if rule_budgets else None
        stop_reason, iterations, improved = self._run_loop(
            names,
            root_eid,
            max_iterations,
            cap,
            budgets,
            stop,
            patience,
            cost_fn,
        )
        return self._run_stats(
            budgets,
            stop_reason,
            iterations,
            improved,
            stop == "improving",
        )

    # -- extraction ---------------------------------------------------------

    @staticmethod
    def _ser_enode(n: Any) -> tuple:
        """Serialize an ``ENode`` to the ``(op, children, attrs)`` form."""
        return (
            n.op,
            tuple(int(c) for c in n.children),
            tuple((k, v) for k, v in n.attrs),
        )

    def _native_pricer(
        self, cfn: Any, m: dict, rc_parts: tuple | None
    ) -> Any:
        """Build the per-candidate pricing callback the Rust walk drives.

        ``table`` maps each ``term_id`` the walk hands out to
        ``(term, cost, eo, rc)`` — the same values the reference
        extractor's cache probes read out of the shared memo, plus the
        same ``("eo","generic",term)`` / ``("rc",...)`` pre-seeding, so
        ``cfn`` stays O(arity) per candidate for the additive
        sub-models.  Returns ``(term_id, cost, flags)``; leaf flags
        bit0 marks ``Var`` leaves.
        """
        table: list = []
        rc_pf = rc_bw = rc_ls = None
        if rc_parts is not None:
            rc_pf, rc_bw, rc_ls = rc_parts

        def price(tag: str, x: Any, children: tuple, attrs: tuple):
            if tag == "leaf":
                term = _LeafRegistry.decode(x)
                tid = len(table)
                ct = cfn(term)
                table.append(
                    (
                        term,
                        ct,
                        m.get(("eo", "generic", term)),
                        m.get(("rc", rc_pf, rc_bw, rc_ls, term))
                        if rc_parts is not None
                        else None,
                    )
                )
                return tid, ct, 1 if isinstance(term, Var) else 0
            kids = [table[i][0] for i in children]
            term = Op.make(x, *kids, **dict(attrs))
            eo_vals = [table[i][2] for i in children]
            if all(v is not None for v in eo_vals):
                ek = ("eo", "generic", term)
                if ek not in m:
                    m[ek] = sum(eo_vals) + (
                        _SOLVER_FACTOR if x in _SOLVER_OPS else 1.0
                    )
            if rc_parts is not None:
                rc_vals = [table[i][3] for i in children]
                if all(v is not None for v in rc_vals):
                    rk = ("rc", rc_pf, rc_bw, rc_ls, term)
                    if rk not in m:
                        rc = _local_roofline(
                            term,
                            m,
                            peak_flops=cast("float", rc_pf),
                            peak_bw=cast("float", rc_bw),
                            launch_s=cast("float", rc_ls),
                        )
                        for v in rc_vals:
                            rc += v
                        m[rk] = rc
            tid = len(table)
            ct = cfn(term)
            table.append(
                (
                    term,
                    ct,
                    m.get(("eo", "generic", term)),
                    m.get(("rc", rc_pf, rc_bw, rc_ls, term))
                    if rc_parts is not None
                    else None,
                )
            )
            return tid, ct, 0

        return price, table

    def extract_best(
        self,
        eid: int,
        cost_fn: Any,
        overrides: dict[int, Any] | None = None,
        bans: dict[int, set] | None = None,
        _cache_out: dict | None = None,
        fusion_epsilon: float = 0.0,
    ) -> Any:
        """Extract the minimum-cost term — the walk runs in Rust.

        Same semantics as the reference mixin (member order, DAG-once
        billing, param-only discount, the ``(total, nops)`` tie-break
        and the ``fusion_epsilon`` band): the Rust core drives the
        per-class DP and calls the ``price`` callback once per
        candidate term, so ``cost_fn`` stays a plain Python callable.
        ``overrides``/``bans``/``_cache_out`` carry the same contract —
        enodes cross serialized, cache entries come back per class.
        """
        root = self.find(eid)
        m = self._cost_memo_for(cost_fn)
        cfn = _memo_dispatch(cost_fn, m)
        unwrapped = getattr(cost_fn, "func", cost_fn)
        bill_params = getattr(unwrapped, "charges_param_only", False)
        rc_parts: tuple | None = None
        if hasattr(unwrapped, "profile"):
            try:
                rc_parts = _profile_constants(
                    getattr(unwrapped, "profile", None)
                )
            except (AttributeError, TypeError, KeyError):
                rc_parts = None
        price, table = self._native_pricer(cfn, m, rc_parts)

        def fuser(tid: int) -> tuple:
            return fusion_member_key(table[tid][0], m)

        over = (
            None
            if overrides is None
            else {
                self.find(k): self._ser_enode(v)
                for k, v in overrides.items()
            }
        )
        ban_map = (
            None
            if bans is None
            else {
                self.find(k): [self._ser_enode(n) for n in s]
                for k, s in bans.items()
            }
        )
        total, term_id, nops, dump = self._g.extract(
            root,
            price,
            fuser=fuser if fusion_epsilon else None,
            bill_params=bill_params,
            foldable=FOLD_OPS,
            fusion_epsilon=fusion_epsilon,
            overrides=over,
            bans=ban_map,
        )
        term = table[term_id][0] if term_id is not None else None
        if _cache_out is not None:
            for ceid, ctot, ctid, cused, cpo, cnops, cct in dump:
                ent = table[ctid] if ctid is not None else None
                _cache_out[ceid] = (
                    ctot,
                    ent[0] if ent is not None else None,
                    frozenset(cused),
                    cpo,
                    cnops,
                    cct,
                    ent[2] if ent is not None else None,
                    ent[3] if ent is not None else None,
                )
        if term is None:
            logger.warning(
                "extract_best: no extractable term in eclass %d", root
            )
        else:
            logger.info(
                "extract_best: eclass %d -> cost %.4g",
                root,
                total,
                extra={"cost": total, "nops": nops},
            )
        return term

    # -- the _classes view -------------------------------------------------

    @property
    def _classes(self) -> dict[int, EClass]:
        """Materialised ``eid -> EClass`` view for the extraction mixin.

        Rebuilt lazily and invalidated on every mutation — extraction
        always sees the post-saturation graph, and ``_ExtractMixin``
        (and ``_carrier_upgrade``, ``extract_alternatives``,
        ``extract_paired``, ``diverse_classes``) runs unmodified over
        real ``mlopt_core`` ``EClass``/``ENode`` objects.
        """
        view = self._view
        if view is None:
            view = {}
            for eid, nodes in self._g.classes():
                ec = EClass(id=eid)
                ec.nodes = {
                    ENode(op, tuple(children), tuple(attrs))
                    for op, children, attrs in nodes
                }
                view[eid] = ec
            self._view = view
        return view

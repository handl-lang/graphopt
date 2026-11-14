"""Candidate law proposal — compatibility shim over ``proposal_core``.

The whole candidate-proposal machinery moved to
:mod:`mlopt_discovery.proposal_core` at the plan-0034/0035 torch-seam
split: the records, generators, ranker and guarded second-chance move
are torch-free at import time there (torch/``mlopt_torch`` bind
lazily at call time, and ``evaluate``/``propose_guarded`` take an
injected ``numeric`` oracle).  Every name this module used to define
— public *and* private, including the ``lp._x`` reach-throughs
downstream modules and tests use — is re-exported unchanged, so
``mlopt_discovery.proposal`` bindings keep resolving; new code
should bind ``mlopt_discovery.proposal_core``.

Run::

    .venv/bin/python -m mlopt_discovery.proposal
    .venv/bin/python -m mlopt_discovery.proposal --json /tmp/mlopt_discovery.proposal.json
"""

from __future__ import annotations

from typing import Any

from mlopt_discovery import proposal_core as _proposal_core
from mlopt_discovery.proposal_core import (
    _AUTO_MAX_CLAUSES as _AUTO_MAX_CLAUSES,
)
from mlopt_discovery.proposal_core import (
    _D as _D,
)
from mlopt_discovery.proposal_core import (
    _EXPANSIVE_BUDGET as _EXPANSIVE_BUDGET,
)
from mlopt_discovery.proposal_core import (
    _GUARD_PAY_TERMS as _GUARD_PAY_TERMS,
)
from mlopt_discovery.proposal_core import (
    _GUARD_SYNTH_LIMIT as _GUARD_SYNTH_LIMIT,
)
from mlopt_discovery.proposal_core import (
    _MAX_ITERATIONS as _MAX_ITERATIONS,
)
from mlopt_discovery.proposal_core import (
    _MAX_NODES as _MAX_NODES,
)
from mlopt_discovery.proposal_core import (
    _MAX_PROGRAM_SIZE as _MAX_PROGRAM_SIZE,
)
from mlopt_discovery.proposal_core import (
    _SCAN_ITERATIONS as _SCAN_ITERATIONS,
)
from mlopt_discovery.proposal_core import (
    _SCAN_NODES as _SCAN_NODES,
)
from mlopt_discovery.proposal_core import (
    _TOL as _TOL,
)
from mlopt_discovery.proposal_core import (
    ALL_RULES as ALL_RULES,
)
from mlopt_discovery.proposal_core import (
    Action as Action,
)
from mlopt_discovery.proposal_core import (
    Candidate as Candidate,
)
from mlopt_discovery.proposal_core import (
    Const as Const,
)
from mlopt_discovery.proposal_core import (
    EGraph as EGraph,
)
from mlopt_discovery.proposal_core import (
    Iterable as Iterable,
)
from mlopt_discovery.proposal_core import (
    Op as Op,
)
from mlopt_discovery.proposal_core import (
    Outcome as Outcome,
)
from mlopt_discovery.proposal_core import (
    Param as Param,
)
from mlopt_discovery.proposal_core import (
    Path as Path,
)
from mlopt_discovery.proposal_core import (
    Rewrite as Rewrite,
)
from mlopt_discovery.proposal_core import (
    TensorType as TensorType,
)
from mlopt_discovery.proposal_core import (
    Var as Var,
)
from mlopt_discovery.proposal_core import (
    _abstract as _abstract,
)
from mlopt_discovery.proposal_core import (
    _allclose as _allclose,
)
from mlopt_discovery.proposal_core import (
    _apply as _apply,
)
from mlopt_discovery.proposal_core import (
    _average_precision as _average_precision,
)
from mlopt_discovery.proposal_core import (
    _canon as _canon,
)
from mlopt_discovery.proposal_core import (
    _class_reps as _class_reps,
)
from mlopt_discovery.proposal_core import (
    _cost_delta as _cost_delta,
)
from mlopt_discovery.proposal_core import (
    _det_rep as _det_rep,
)
from mlopt_discovery.proposal_core import (
    _dump_json as _dump_json,
)
from mlopt_discovery.proposal_core import (
    _eval_backend as _eval_backend,
)
from mlopt_discovery.proposal_core import (
    _eval_guarded as _eval_guarded,
)
from mlopt_discovery.proposal_core import (
    _fmt_useful as _fmt_useful,
)
from mlopt_discovery.proposal_core import (
    _fmt_yield as _fmt_yield,
)
from mlopt_discovery.proposal_core import (
    _grammar_candidate as _grammar_candidate,
)
from mlopt_discovery.proposal_core import (
    _guard_sites as _guard_sites,
)
from mlopt_discovery.proposal_core import (
    _heuristic_score as _heuristic_score,
)
from mlopt_discovery.proposal_core import (
    _key as _key,
)
from mlopt_discovery.proposal_core import (
    _LeafRegistry as _LeafRegistry,
)
from mlopt_discovery.proposal_core import (
    _learned_scores as _learned_scores,
)
from mlopt_discovery.proposal_core import (
    _leaves as _leaves,
)
from mlopt_discovery.proposal_core import (
    _library_keys as _library_keys,
)
from mlopt_discovery.proposal_core import (
    _numeric_true as _numeric_true,
)
from mlopt_discovery.proposal_core import (
    _outcome_row as _outcome_row,
)
from mlopt_discovery.proposal_core import (
    _pool_features as _pool_features,
)
from mlopt_discovery.proposal_core import (
    _positions as _positions,
)
from mlopt_discovery.proposal_core import (
    _probe_rule as _probe_rule,
)
from mlopt_discovery.proposal_core import (
    _proc_check as _proc_check,
)
from mlopt_discovery.proposal_core import (
    _proc_derive as _proc_derive,
)
from mlopt_discovery.proposal_core import (
    _rank_on as _rank_on,
)
from mlopt_discovery.proposal_core import (
    _relation as _relation,
)
from mlopt_discovery.proposal_core import (
    _rule_budgets as _rule_budgets,
)
from mlopt_discovery.proposal_core import (
    _sat_cost as _sat_cost,
)
from mlopt_discovery.proposal_core import (
    _size as _size,
)
from mlopt_discovery.proposal_core import (
    _spec_term as _spec_term,
)
from mlopt_discovery.proposal_core import (
    _tags as _tags,
)
from mlopt_discovery.proposal_core import (
    _term_expand as _term_expand,
)
from mlopt_discovery.proposal_core import (
    _term_leaves as _term_leaves,
)
from mlopt_discovery.proposal_core import (
    _term_match as _term_match,
)
from mlopt_discovery.proposal_core import (
    _train_scorer as _train_scorer,
)
from mlopt_discovery.proposal_core import (
    _tree_dist as _tree_dist,
)
from mlopt_discovery.proposal_core import (
    _v as _v,
)
from mlopt_discovery.proposal_core import (
    alpha_key as alpha_key,
)
from mlopt_discovery.proposal_core import (
    argparse as argparse,
)
from mlopt_discovery.proposal_core import (
    auto_cond_object as auto_cond_object,
)
from mlopt_discovery.proposal_core import (
    composite_candidates as composite_candidates,
)
from mlopt_discovery.proposal_core import (
    compute_features as compute_features,
)
from mlopt_discovery.proposal_core import (
    dag_cost as dag_cost,
)
from mlopt_discovery.proposal_core import (
    dataclass as dataclass,
)
from mlopt_discovery.proposal_core import (
    evaluate as evaluate,
)
from mlopt_discovery.proposal_core import (
    flops_cost as flops_cost,
)
from mlopt_discovery.proposal_core import (
    instance_of as instance_of,
)
from mlopt_discovery.proposal_core import (
    json as json,
)
from mlopt_discovery.proposal_core import (
    lawdata as lawdata,
)
from mlopt_discovery.proposal_core import (
    main as main,
)
from mlopt_discovery.proposal_core import (
    near_miss_candidates as near_miss_candidates,
)
from mlopt_discovery.proposal_core import (
    op_repr as op_repr,
)
from mlopt_discovery.proposal_core import (
    propose_guarded as propose_guarded,
)
from mlopt_discovery.proposal_core import (
    rank_pool as rank_pool,
)
from mlopt_discovery.proposal_core import (
    rule_samples as rule_samples,
)
from mlopt_discovery.proposal_core import (
    schema_candidates as schema_candidates,
)
from mlopt_discovery.proposal_core import (
    seed_terms as seed_terms,
)
from mlopt_discovery.proposal_core import (
    suppress as suppress,
)
from mlopt_discovery.proposal_core import (
    term_from_spec as term_from_spec,
)
from mlopt_discovery.proposal_core import (
    verify_law as verify_law,
)
from mlopt_discovery.proposal_core import (
    yield_table as yield_table,
)

__all__ = [
    "Candidate",
    "Outcome",
    "composite_candidates",
    "evaluate",
    "main",
    "near_miss_candidates",
    "propose_guarded",
    "rank_pool",
    "schema_candidates",
    "seed_terms",
    "yield_table",
]


def __getattr__(name: str) -> Any:
    """Reach-through for names bound only in ``proposal_core``.

    Covers compat lookups the explicit list does not name (e.g. the
    live ``_EVAL`` backend singleton) so ``mlopt_discovery.proposal``
    behaves exactly like the pre-split module.
    """
    return getattr(_proposal_core, name)


if __name__ == "__main__":
    raise SystemExit(main())

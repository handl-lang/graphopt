"""``play`` — the optimization game as a tool.

One registry of *domains* (a board family plus its player
vocabulary), one entrypoint: pick a domain, pick arms, get the
paired-boards table.  Adding a domain is data — a case generator, a
``case -> Board`` factory and a player *spec* (the feature schema,
the featurizer's composable parts and the legal-move enumerator) —
no driver code and no player subclass:

.. code-block:: text

    SPEC = DomainSpec.build(
        features=FEATURES,          # the learned arm's schema
        legal=legal_actions,        # state -> tuple[action, ...]
        parts=(Bias(), StateBroadcast(...), ...),  # the feature row
        weights=PLAYER_WEIGHTS,     # the baked prior (optional)
    )
    DOMAINS[name] = spec_domain(
        cases=gen_cases,            # seed, n -> [case, ...]
        board_of=...,               # case -> Board
        spec=SPEC,
        scripted=...,               # optional fixed-playbook arm
    )

Shipped domains:

* ``meta`` — the meta-arena: fire/saturate/declare/handle/extract
  over the mined-spec corpus (:func:`meta_player.gen_cases`);
* ``joint`` — training graphs: a forward term plus every derived
  gradient under one ``joint`` root, played on the same meta-arena
  board (the reverse handler's programs are ordinary terms);
* ``search`` — the core :class:`~mlopt_core.search_env.SearchEnv`:
  one rule per step under a horizon/patience bound, adapted to the
  :class:`~mlopt_discovery.engine.Board` contract;
* ``torch`` — real ``nn.Module``s lifted through
  ``mlopt_torch.adapters.TorchSource`` and played under the real
  sink's ``supported_ops`` bound; the ``--deliver`` flag lowers
  the winning extraction to a verified runnable module;
* ``open`` — a page of enumerated law candidates, one oracle
  evaluation per move (:mod:`open_game`);
* ``player`` — the two-tier meta-game (:mod:`player_arena`): the
  state is an object-level *player spec record* under rewrite, and
  ``evaluate`` runs a nested train+held-out-eval on the ``meta`` or
  ``open`` board — the player itself is the term being rewritten;
* ``bilinear`` — the ``<2,2,2>`` matmul tensor's decomposition
  space (:mod:`bilinear_game`): cut/pick/flip/drop over a
  rank-decomposition, cost = multiplication count — the Strassen
  rediscovery spike (Q2).

Downstream domain repos add their own: importing the domain module
registers its :class:`Domain` into ``DOMAINS`` (kernelopt's
``mlopt_kernel.play`` seats ``kernel`` — plan 0032 phase 4).

Run it::

    python -m mlopt_discovery.play --domain meta --train-cases 40 \
        --eval-cases 10
"""

from __future__ import annotations

import contextlib as _contextlib
import sys as _sys
import types as _types

from gamekit import engine as _engine

# Sibling-module aliases the monolithic ``play.py`` exposed —
# ``play.pa`` / ``play.fg`` / ``play.mp`` / ``play.og`` / ``play.bg``
# are read by ``domain_spec``/``model_corpus`` and bench drivers.
from .. import bilinear_game as _bilinear_game
from .. import fragment_game as _fragment_game
from .. import lawdata as _lawdata
from .. import meta_arena as _meta_arena
from .. import meta_player as _meta_player
from .. import open_game as _open_game
from .. import player_arena as _player_arena
from .. import training as _training

# Import order is the dependency order: registry is the frozen seam
# (import-light — kernelopt binds ``spec_domain``/``DOMAINS`` without
# pulling the domain arms' machinery beyond what the package already
# imported as a module), sites underpins gen, and domains is the
# composition root that populates ``DOMAINS``.
from . import cli as _cli
from . import domains as _domains
from . import fragment as _fragment
from . import gen as _gen
from . import meta_joint as _meta_joint
from . import players as _players
from . import probes as _probes
from . import registry as _registry
from . import search as _search
from . import sites as _sites
from . import torch_arm as _torch_arm
from .cli import (
    _arms as _arms,
)
from .cli import (
    _extras as _extras,
)
from .cli import (
    _kernel_probes as _kernel_probes,
)
from .cli import (
    _pkg_doc as _pkg_doc,
)
from .cli import (
    _table as _table,
)
from .cli import (
    main as main,
)
from .fragment import (
    _FRAGMENT_POOL as _FRAGMENT_POOL,
)
from .fragment import (
    FRAGMENT_FEATURES as FRAGMENT_FEATURES,
)
from .fragment import (
    FRAGMENT_SPEC as FRAGMENT_SPEC,
)
from .fragment import (
    _frag_act as _frag_act,
)
from .fragment import (
    _frag_state as _frag_state,
)
from .fragment import (
    _fragment_board as _fragment_board,
)
from .fragment import (
    _fragment_cases as _fragment_cases,
)
from .fragment import (
    _fragment_pool as _fragment_pool,
)
from .fragment import (
    _fragment_scripted as _fragment_scripted,
)
from .gen import (
    TRITON_REFEREE_LOG as TRITON_REFEREE_LOG,
)
from .gen import (
    _as_dtype as _as_dtype,
)
from .gen import (
    _binding_attempts as _binding_attempts,
)
from .gen import (
    _case_tail as _case_tail,
)
from .gen import (
    _claim_cost_gate as _claim_cost_gate,
)
from .gen import (
    _claim_denotation_for as _claim_denotation_for,
)
from .gen import (
    _claim_measurable as _claim_measurable,
)
from .gen import (
    _ex_bounds as _ex_bounds,
)
from .gen import (
    _fold_cost_gate as _fold_cost_gate,
)
from .gen import (
    _gen_board as _gen_board,
)
from .gen import (
    _gen_cases as _gen_cases,
)
from .gen import (
    _gen_probe_impl as _gen_probe_impl,
)
from .gen import (
    _has_triton as _has_triton,
)
from .gen import (
    _probe_delivery as _probe_delivery,
)
from .gen import (
    _probe_graph_arm as _probe_graph_arm,
)
from .gen import (
    _probe_play as _probe_play,
)
from .gen import (
    _time_module as _time_module,
)
from .gen import (
    _triton_kernel_for as _triton_kernel_for,
)
from .gen import (
    _var_env as _var_env,
)
from .gen import (
    _zoo_case as _zoo_case,
)
from .gen import (
    _zoo_export as _zoo_export,
)
from .gen import (
    _zoo_probe_for_args as _zoo_probe_for_args,
)
from .gen import (
    _zoo_probe_gated as _zoo_probe_gated,
)
from .gen import (
    _zoo_probe_impl as _zoo_probe_impl,
)
from .gen import (
    _zoo_probe_workload as _zoo_probe_workload,
)
from .gen import (
    _zoo_rows as _zoo_rows,
)
from .gen import (
    gen_probe as gen_probe,
)
from .gen import (
    zoo_pay_verdict as zoo_pay_verdict,
)
from .gen import (
    zoo_probe as zoo_probe,
)
from .meta_joint import (
    _joint_board as _joint_board,
)
from .meta_joint import (
    _joint_cases as _joint_cases,
)
from .meta_joint import (
    _meta_board as _meta_board,
)
from .players import (
    _random_policy as _random_policy,
)
from .players import (
    _scripted_search as _scripted_search,
)
from .probes import (
    _DECAY_PROBE_SHAPES as _DECAY_PROBE_SHAPES,
)
from .probes import (
    _SCAN_PROBE_SHAPES as _SCAN_PROBE_SHAPES,
)
from .probes import (
    _decay_carrier_ir as _decay_carrier_ir,
)
from .probes import (
    _scan_carrier_ir as _scan_carrier_ir,
)
from .probes import (
    _spelled_decay as _spelled_decay,
)
from .probes import (
    _spelled_scan as _spelled_scan,
)
from .probes import (
    decay_probe as decay_probe,
)
from .probes import (
    scan_probe as scan_probe,
)
from .registry import (
    DOMAINS as DOMAINS,
)
from .registry import (
    Domain as Domain,
)
from .registry import (
    deliver as deliver,
)
from .registry import (
    play as play,
)
from .registry import (
    spec_domain as spec_domain,
)
from .search import (
    _SEARCH_POOL as _SEARCH_POOL,
)
from .search import (
    SEARCH_FEATURES as SEARCH_FEATURES,
)
from .search import (
    SEARCH_SPEC as SEARCH_SPEC,
)
from .search import (
    SearchBoard as SearchBoard,
)
from .search import (
    _ops_walk as _ops_walk,
)
from .search import (
    _search_act as _search_act,
)
from .search import (
    _search_board as _search_board,
)
from .search import (
    _search_cases as _search_cases,
)
from .search import (
    _search_legal as _search_legal,
)
from .search import (
    _search_pool as _search_pool,
)
from .search import (
    _search_state as _search_state,
)
from .search import (
    _SearchState as _SearchState,
)
from .sites import (
    _SEQ_AXIS_LIFTS as _SEQ_AXIS_LIFTS,
)
from .sites import (
    _SEQ_PASSTHROUGH_OPS as _SEQ_PASSTHROUGH_OPS,
)
from .sites import (
    _SEQ_SITE_CAP as _SEQ_SITE_CAP,
)
from .sites import (
    _SEQ_SPINE_CAP as _SEQ_SPINE_CAP,
)
from .sites import (
    TIER_SITES as TIER_SITES,
)
from .sites import (
    _all_scan_sites as _all_scan_sites,
)
from .sites import (
    _carrier_member as _carrier_member,
)
from .sites import (
    _cumsum_sites as _cumsum_sites,
)
from .sites import (
    _decayed_sites as _decayed_sites,
)
from .sites import (
    _flat_ops as _flat_ops,
)
from .sites import (
    _fusion_pattern as _fusion_pattern,
)
from .sites import (
    _fusion_pattern_sz as _fusion_pattern_sz,
)
from .sites import (
    _fusion_sites as _fusion_sites,
)
from .sites import (
    _leaf_name as _leaf_name,
)
from .sites import (
    _mint_tiers as _mint_tiers,
)
from .sites import (
    _op_descendants as _op_descendants,
)
from .sites import (
    _op_parents as _op_parents,
)
from .sites import (
    _reduction_pattern as _reduction_pattern,
)
from .sites import (
    _reduction_sites as _reduction_sites,
)
from .sites import (
    _scan_sites as _scan_sites,
)
from .sites import (
    _seq_argn_attrs as _seq_argn_attrs,
)
from .sites import (
    _seq_axis_idx as _seq_axis_idx,
)
from .sites import (
    _seq_bcast as _seq_bcast,
)
from .sites import (
    _seq_commute_lift as _seq_commute_lift,
)
from .sites import (
    _seq_concat_lift as _seq_concat_lift,
)
from .sites import (
    _seq_const_meta as _seq_const_meta,
)
from .sites import (
    _seq_deindex as _seq_deindex,
)
from .sites import (
    _seq_expansion as _seq_expansion,
)
from .sites import (
    _seq_lift_attrs as _seq_lift_attrs,
)
from .sites import (
    _seq_lift_finish as _seq_lift_finish,
)
from .sites import (
    _seq_lift_norm as _seq_lift_norm,
)
from .sites import (
    _seq_lift_permute as _seq_lift_permute,
)
from .sites import (
    _seq_lift_range as _seq_lift_range,
)
from .sites import (
    _seq_lift_reduce as _seq_lift_reduce,
)
from .sites import (
    _seq_lift_softmax as _seq_lift_softmax,
)
from .sites import (
    _seq_lift_transpose as _seq_lift_transpose,
)
from .sites import (
    _seq_lift_unbind as _seq_lift_unbind,
)
from .sites import (
    _seq_meta as _seq_meta,
)
from .sites import (
    _seq_projection_lift as _seq_projection_lift,
)
from .sites import (
    _seq_reduce_dims as _seq_reduce_dims,
)
from .sites import (
    _seq_scan_sites as _seq_scan_sites,
)
from .sites import (
    _seq_select_free as _seq_select_free,
)
from .sites import (
    _seq_shape as _seq_shape,
)
from .sites import (
    _seq_shape_first as _seq_shape_first,
)
from .sites import (
    _seq_site as _seq_site,
)
from .sites import (
    _seq_spines as _seq_spines,
)
from .sites import (
    _seq_step_cands as _seq_step_cands,
)
from .sites import (
    _seq_step_select as _seq_step_select,
)
from .sites import (
    _seq_traj_states as _seq_traj_states,
)
from .sites import (
    _seq_view_lift as _seq_view_lift,
)
from .sites import (
    _SeqLiftCtx as _SeqLiftCtx,
)
from .sites import (
    _seqm_scan_sites as _seqm_scan_sites,
)
from .sites import (
    _seqm_site as _seqm_site,
)
from .sites import (
    _seqs_chain_spec as _seqs_chain_spec,
)
from .sites import (
    _SeqSpecOverflow as _SeqSpecOverflow,
)
from .sites import (
    _seqt_chain_spec as _seqt_chain_spec,
)
from .sites import (
    _seqt_readout as _seqt_readout,
)
from .sites import (
    _spec_has_meta as _spec_has_meta,
)
from .sites import (
    _spec_leaf_env as _spec_leaf_env,
)
from .torch_arm import (
    _deliver_runner_arm as _deliver_runner_arm,
)
from .torch_arm import (
    _is_plan_tape as _is_plan_tape,
)
from .torch_arm import (
    _max_abs_delta as _max_abs_delta,
)
from .torch_arm import (
    _torch_board as _torch_board,
)
from .torch_arm import (
    _torch_cases as _torch_cases,
)
from .torch_arm import (
    _torch_deliver as _torch_deliver,
)
from .torch_arm import (
    _torch_dtype as _torch_dtype,
)
from .torch_arm import (
    _torch_modules as _torch_modules,
)

bg = _bilinear_game
fg = _fragment_game
lawdata = _lawdata
ma = _meta_arena
mp = _meta_player
og = _open_game
pa = _player_arena
training = _training
engine = _engine

_SUBMODULES = (
    _registry,
    _sites,
    _meta_joint,
    _gen,
    _probes,
    _torch_arm,
    _search,
    _fragment,
    _players,
    _domains,
    _cli,
)

# Re-export the whole former module namespace: every public *and*
# private name the monolithic ``play.py`` exposed resolves here
# (``play._seq_scan_sites``, ``play._mint_tiers``, … are load-bearing
# for tests and bench drivers).
for _m in _SUBMODULES:
    for _k, _v in vars(_m).items():
        if not _k.startswith("__"):
            globals()[_k] = _v

__all__ = [
    "DOMAINS",
    "Domain",
    "SearchBoard",
    "play",
    "spec_domain",
]


class _PlayModule(_types.ModuleType):
    """Module ``__setattr__``/``__delattr__`` that forwards writes.

    The monolithic ``play.py`` had ONE namespace: ``play._mint_tiers =
    spy`` repointed the global every internal caller resolved.  Now
    that the code lives in submodules, a bare ``play.X = v`` would
    only shadow the package attribute while submodule callers kept
    the old binding.  Forwarding the write to every submodule that
    holds ``X`` preserves the single-namespace semantics — including
    names a submodule imported from a sibling (``_mint_tiers`` is
    bound in both ``sites`` and ``gen``).
    """

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name.startswith("__"):
            return
        for m in _SUBMODULES:
            if name in vars(m):
                setattr(m, name, value)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)
        if name.startswith("__"):
            return
        for m in _SUBMODULES:
            if name in vars(m):
                with _contextlib.suppress(AttributeError):
                    delattr(m, name)


_sys.modules[__name__].__class__ = _PlayModule

"""The learned meta-arena player — Q3's arm.

Plan 0021's open question: can a *learned* player beat the scripted
``saturate → extract`` playbook on the mixed board?  The probe's
baseline table says the headroom is real — every baseline ties at
baseline on the fold-site cases while a ``declare`` exists that
wins — but no baseline plays the line, because the paying sequence
is four moves deep: ``declare`` a fresh fold of a mined spec,
``saturate`` so the minted rule fires, ``handle`` to bind a
supported kernel, ``extract`` to collect.

``MetaLearnedPlayer`` is a linear softmax policy over the legal move
set, trained by episode-return REINFORCE — the shared
:class:`~mlopt_discovery.players.LearnedPlayer`, bound to the
:data:`META_SPEC` declared below (the feature schema, the featurizer's
composable parts and ``meta_arena.legal_actions``).  The board is
small (~dozens of legal moves vs the construction arena's ~50k), so
the softmax is *exact* over the unplayed set — no perceptual cap, no
played-mass correction; the gradient row is ``φ_chosen - E_π[φ]``
computed directly.

``gen_cases`` builds the training corpus: seeded variants of the
sites the handlers cover (a ``mul(x, sigmoid(x))`` under a ruleset
with ``silu_fold`` stripped pays exactly one line — the
declare→saturate→handle chain), controls where nothing pays (the
policy must learn *not* to burn steps), and a swiglu site whose
kernel enters the bound explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from gamekit.players import (
    ActionFeats,
    Bias,
    DomainSpec,
    LearnedPlayer,
    OpOneHot,
    PlayHist,
    PlayRates,
    StateBroadcast,
    register_legal,
    register_part,
)
from mlopt_core.cost.basic import count_cost
from mlopt_core.ir import Const, Op, TensorType, Var
from mlopt_core.laws import DEFAULT

from . import corpus_cov, lawdata, meta_prior
from . import meta_arena as ma

__all__ = [
    "META_FEATURES",
    "META_PART_TAGS",
    "META_SPEC",
    "META_SPEC_V2",
    "MetaHist",
    "MetaLearnedPlayer",
    "availability_order",
    "case_availability",
    "curriculum_order",
    "evaluate",
    "featurize",
    "gen_cases",
    "imitate_oracle",
    "oracle_line",
    "plan_oracles",
    "plan_spelled",
    "player_table",
    "spelled_candidates",
    "train_player",
]

#: REINFORCE default — the spec's learning rate.
_LR = 0.1

#: The pre-consolidation name, kept as an alias for callers/tests.
_Hist = PlayHist

#: Handler-identity bucket width for the meta schema — widened
#: from ``lawdata.META_ARENA_HASH_BUCKETS["handler"]`` (4) because
#: the shipped ``silu``/``square`` handler tags collide there
#: (both land in bucket 3): literal ``claim``/``handle`` rows and
#: the macro ``a:h:*`` legs could not tell them apart — a latent
#: blindness (square always declines today, so nothing paid the
#: difference, but a paying square claim would have been invisible
#: to the policy).  Width 8 separates all three shipped tags
#: (silu→7, square→3, swiglu→0) and leaves headroom for minted
#: handler tags.  ``_bucket`` reads this override for the
#: ``"handler"`` kind; the schema rows ``a:h:4..7``, ``x:mc:*``,
#: ``x:mh:*`` and V2's ``x:seqh:*:4..7`` are generated off it, so
#: the featurizer and the schema can never desync.
_META_HANDLER_BUCKETS = 8

_META_BUCKET_WIDTHS: dict[str, int] = {
    **lawdata.META_ARENA_HASH_BUCKETS,
    "handler": _META_HANDLER_BUCKETS,
}


# ---------------------------------------------------------------------------
#  Featurization — (MetaState, Action) -> feature row over the schema
# ---------------------------------------------------------------------------


def _bucket(name: Any, kind: str) -> int:
    """Stable-hash *name* into the kind's bucket width.

    Widths come from :data:`_META_BUCKET_WIDTHS` — the lawdata table
    with the ``"handler"`` kind widened (see
    :data:`_META_HANDLER_BUCKETS` for why).
    """
    width = _META_BUCKET_WIDTHS[kind]
    h = hashlib.sha256(str(name).encode("utf-8")).digest()
    return int.from_bytes(h[:4]) % width


def _state_feats(state: ma.MetaState) -> dict[str, float]:
    """Return the ``st:*`` broadcast block — the summary per move."""
    total = state.n_enodes + max(state.budget_left, 0)
    rel = (
        state.best_cost / state.baseline_cost
        if state.baseline_cost > 0
        else 1.0
    )
    return {
        "st:steps": state.steps / 32.0,
        "st:enodes_frac": state.n_enodes / max(total, 1),
        "st:classes_frac": state.n_classes / max(state.n_enodes, 1),
        "st:improve": max(0.0, 1.0 - rel),
        "st:declared": len(state.declared) / 8.0,
        "st:handled": len(state.handles) / 8.0,
        "st:unhandled": (
            len(state.declared_bodies) - len(state.handles)
        )
        / 8.0,
        "st:handleable": _handleable(state) / 8.0,
        "st:sites": len(state.sites) / 8.0,
        "st:claimable": state.claimable / 8.0,
        "st:specs": len(state.specs) / 32.0,
        "st:fires": sum(c for _, c in state.rule_fires) / 64.0,
    }


def _handleable(state: ma.MetaState) -> int:
    """Mined specs some handler's canonical pattern covers.

    The precondition of the paying line — ``declare`` a fresh fold
    of a covered spec, then ``handle`` binds it to the kernel.  A
    spec is coverable iff its alpha-canonical form equals a handler
    pattern's (``meta_arena._handle_actions``' own test).
    """
    pats = [p for _, p in state.handler_specs]
    return sum(
        1
        for s in state.specs
        if any(ma._canon_spec(s) == p for p in pats)
    )


def _spec_index(action: ma.Action) -> float:
    """Mined-spec position a ``declare`` targets (``foldabs_i``)."""
    cons = action.params.get("construction") or {}
    params = cons.get("params") or {}
    suffix = str(params.get("name", "")).rsplit("_", 1)[-1]
    return int(suffix) / 32.0 if suffix.isdigit() else 0.5


def _act_feats(
    action: ma.Action, row: dict[str, float]
) -> dict[str, float]:
    """Return the move's own params and interactions — ``a:*``/``x:*``.

    *row* is the partially-built feature row (the state broadcasts and
    play rates), so an interaction term can read a broadcast value.
    The move-kind one-hot is a separate part (:class:`OpOneHot`).
    """
    op = action.op
    out: dict[str, float] = {}
    if op == "fire":
        out[f"a:r:{_bucket(action.params.get('rule'), 'rule'):02d}"] = (
            1.0
        )
    elif op == "saturate":
        budget = action.params.get("budget")
        if budget is None:
            out["a:unbounded"] = 1.0
        else:
            out["a:budget"] = float(budget) / 512.0
        out["x:pressure:saturate"] = row["st:enodes_frac"]
    elif op == "declare":
        out["a:spec_i"] = _spec_index(action)
        out["x:specs:declare"] = row["st:specs"]
        out["x:handleable:declare"] = row["st:handleable"]
        out["x:claimable:declare"] = row["st:claimable"]
    elif op == "handle":
        out[
            f"a:h:{_bucket(action.params.get('handler'), 'handler')}"
        ] = 1.0
        out[f"a:o:{_bucket(action.params.get('object'), 'object')}"] = (
            1.0
        )
        out["x:declared:handle"] = row["st:declared"]
        out["x:unhandled:handle"] = row["st:unhandled"]
    elif op == "claim":
        out[
            f"a:h:{_bucket(action.params.get('handler'), 'handler')}"
        ] = 1.0
        out["x:handleable:claim"] = row["st:handleable"]
        out["x:claimable:claim"] = row["st:claimable"]
    elif op == "extract":
        out["x:extract:improve"] = row["st:improve"]
        out["x:late:extract"] = row["st:steps"]
    return out


def _macro_feats(
    action: ma.Action, row: dict[str, float]
) -> dict[str, float]:
    """Return the macro payload block — ``x:macro:*`` + identity reuse.

    ``docs/results_macro_moves.md``'s diagnosed blindness: every
    ``op == "macro"`` offer featurized to the same broadcast row, so
    the softmax could learn *when* a macro pays but never *which*.
    This part emits the per-offer signal:

    * ``x:macro:len`` — the compressed line's length (``seq``/8):
      a one-step ``claim→extract`` prices differently from the
      spelled four-move line;
    * ``x:macro:certified`` / ``x:macro:score`` — the miner's ranking
      evidence the offer carries on ``params["macro"]`` (the stat
      payload ``macro_moves.instantiate`` writes:
      ``certified`` = share of source episodes that certified;
      ``score`` = count x mean terminal reward, the miner's rank
      key, rescaled /8);
    * ``a:h:<bucket>`` — the *same* handler-identity channel a
      spelled ``claim``/``handle`` uses, emitted for each claim and
      handle leg in the seq.  This is what lets the policy prefer
      ``claim(swiglu)`` over the rank-1 ``claim(silu)`` on a swiglu
      board — the failure mode the macro doc's honest notes named
      ("viability is legality, not payoff");
    * ``x:claimable:claim`` / ``x:handleable:claim`` — the context
      terms a spelled claim gets, emitted when the seq *contains* a
      claim, so the macro row inherits the board-fit pricing;
    * the move-shape legs (``x:macro:spelled`` / ``saturating`` /
      ``nclaims`` / ``reach_*``) and the role-qualified handler
      buckets ``x:mc:*`` / ``x:mh:*`` — the spelled-vs-claim
      distinction the stat columns cannot carry.  On ``dual``/``tri``
      boards the oracle's argmax is the spelled
      ``declare→saturate→handle→extract`` macro while the
      ``claim_*_extract`` compressions differ only in
      ``x:macro:len``/``score`` and the merged ``a:h:*`` legs —
      winner anti-correlated with miner score, ~0.03 held-out
      residual.  ``spelled``/``saturating`` name the seq's
      structure; ``reach_*`` prices it against the board's site and
      claimable broadcasts (zero on non-spelled offers, so the
      product stays candidate-varying — the property every useful
      ``x:*`` term needs); ``x:mc:*``/``x:mh:*`` split the handler
      identity by *role* — a claim-leg bind and a handle-leg bind
      are different moves that ``a:h:*`` merges.

    ``op:macro`` itself comes free through :class:`OpOneHot` once the
    schema names it.  ``a:macro`` is deliberately absent — it would
    be the same flag twice.  A hand-spelled ``Action("macro",
    {"seq": ...})`` with no ``Macro`` record still gets the
    structural terms; the stat columns read as 0 (missing weight).
    """
    if action.op != "macro":
        return {}
    seq = tuple(action.params.get("seq") or ())
    out: dict[str, float] = {"x:macro:len": len(seq) / 8.0}
    m = action.params.get("macro")
    if m:
        if isinstance(m, dict):
            out["x:macro:certified"] = float(m.get("certified", 0.0))
            out["x:macro:score"] = float(m.get("score", 0.0)) / 8.0
        else:  # a Macro-like record duck-typed in by the caller
            out["x:macro:certified"] = float(
                getattr(m, "certified", 0.0)
            )
            out["x:macro:score"] = float(getattr(m, "score", 0.0)) / 8.0
    has_claim = False
    ops: list[str] = []
    claim_tags: list[str] = []
    handle_tags: list[str] = []
    for sub in seq:
        op, params = (
            (str(sub["op"]), dict(sub.get("params") or {}))
            if isinstance(sub, dict)
            else (str(sub.op), dict(getattr(sub, "params", {})))
        )
        ops.append(op)
        if op in ("claim", "handle"):
            tag = params.get("handler")
            if tag is not None:
                out[f"a:h:{_bucket(tag, 'handler')}"] = 1.0
                if op == "claim":
                    claim_tags.append(str(tag))
                else:
                    handle_tags.append(str(tag))
            has_claim = has_claim or op == "claim"
    spelled = "declare" in ops
    saturating = "saturate" in ops
    reach = 1.0 if spelled and saturating else 0.0
    out["x:macro:spelled"] = 1.0 if spelled else 0.0
    out["x:macro:saturating"] = 1.0 if saturating else 0.0
    out["x:macro:nclaims"] = ops.count("claim") / 4.0
    out["x:macro:reach_sites"] = reach * row.get("st:sites", 0.0)
    out["x:macro:reach_claimable"] = reach * row.get(
        "st:claimable", 0.0
    )
    for tag in claim_tags:
        out[f"x:mc:{_bucket(tag, 'handler')}"] = 1.0
    for tag in handle_tags:
        out[f"x:mh:{_bucket(tag, 'handler')}"] = 1.0
    if has_claim:
        out["x:claimable:claim"] = row.get("st:claimable", 0.0)
        out["x:handleable:claim"] = row.get("st:handleable", 0.0)
    return out


#: The shipped meta feature schema — ``lawdata.META_ARENA_FEATURES``
#: plus two extensions the lawdata table cannot express on its own:
#:
#: * ``a:h:4..7`` — the widened handler buckets
#:   (:data:`_META_HANDLER_BUCKETS`; lawdata's schema stops at its
#:   own width-4 ``a:h:0..3`` rows, which stay — the bucket
#:   *indices* are what changed, not the names);
#: * the macro move-shape legs — ``x:macro:spelled`` /
#:   ``saturating`` / ``nclaims`` / ``reach_sites`` /
#:   ``reach_claimable`` and the role-qualified handler buckets
#:   ``x:mc:*`` / ``x:mh:*`` (:func:`_macro_feats` emits them; the
#:   ``dual``/``tri`` spelled-vs-claim blindness they price is
#:   documented in that function's docstring).
#:
#: The extension lives here rather than in ``lawdata``/gamekit
#: because the *semantics* are domain machinery: gamekit's
#: ``gamedata.features`` owns the base schema both repos share,
#: while these legs name ``meta_player``'s own featurizer parts —
#: the same seam ``META_ARENA_FEATURES_V2`` already uses for its
#: ``x:seq*`` block.
META_FEATURES: tuple[str, ...] = (
    *lawdata.META_ARENA_FEATURES,
    *(
        f"a:h:{b}"
        for b in range(
            lawdata.META_ARENA_HASH_BUCKETS["handler"],
            _META_HANDLER_BUCKETS,
        )
    ),
    "x:macro:spelled",
    "x:macro:saturating",
    "x:macro:nclaims",
    "x:macro:reach_sites",
    "x:macro:reach_claimable",
    *(f"x:mc:{b}" for b in range(_META_HANDLER_BUCKETS)),
    *(f"x:mh:{b}" for b in range(_META_HANDLER_BUCKETS)),
)


#: The meta featurizer's composable parts, in row order: the intercept,
#: the ``st:*`` broadcast, the ``h:*`` play rates, the move-kind
#: one-hot, then the action's params and interactions — and the macro
#: payload last (it reads no ``st:*`` keys it doesn't guard, so the
#: part order only fixes row ordering).
META_PARTS: tuple = (
    Bias(),
    StateBroadcast(_state_feats),
    PlayRates(("fire", "saturate", "declare", "handle", "claim")),
    OpOneHot(frozenset(META_FEATURES)),
    ActionFeats(_act_feats),
    ActionFeats(_macro_feats),
)

#: The meta parts in the spec record's tag vocabulary — the names a
#: rewritable spec record carries (:meth:`DomainSpec.to_record`).
META_PART_TAGS: tuple[str, ...] = (
    "bias",
    "meta:state",
    "meta:hist",
    "meta:onehot",
    "meta:action",
    "meta:macro",
)

register_part("meta:state", lambda: StateBroadcast(_state_feats))
register_part(
    "meta:hist",
    lambda: PlayRates(
        ("fire", "saturate", "declare", "handle", "claim")
    ),
)
register_part(
    "meta:onehot",
    lambda: OpOneHot(frozenset(META_FEATURES)),
)
register_part("meta:action", lambda: ActionFeats(_act_feats))
register_part("meta:macro", lambda: ActionFeats(_macro_feats))
register_legal("meta", ma.legal_actions)

#: The meta-arena player spec — schema, parts, enumerator, prior.
META_SPEC = DomainSpec.build(
    features=META_FEATURES,
    legal=ma.legal_actions,
    parts=META_PARTS,
    weights=meta_prior.META_ARENA_PLAYER_WEIGHTS,
    lr=_LR,
)

#: The composed featurizer — public for tests.
featurize = META_SPEC.featurize


# ---------------------------------------------------------------------------
#  V2 — the 2nd-order feature arm (last move x board catalog)
# ---------------------------------------------------------------------------
#
#  The shipped schema is first-order: ``op:*`` names the candidate,
#  ``st:*``/``x:*`` price it against the board — but nothing says
#  *what just happened*.  The composed line's credit assignment is
#  exactly a second-order question: ``handle`` pays only *after* a
#  ``declare`` armed it, ``extract`` pays only *after* the payout
#  move landed.  :class:`MetaHist` carries the last played op into
#  the featurizer and the ``x:seq:*``/``x:seqh:*`` block emits the
#  pair terms — last-move x candidate-kind, and last-move x
#  handler-identity-bucket on the catalog-binding moves.  Broadcast
#  ``x:prev:*`` rows would cancel in the softmax, so only
#  candidate-varying products are emitted.

#: Move kinds the ``x:seq`` pair vocabulary spans — the schema's
#: ``op:*`` one-hot set plus ``macro`` and the ``start`` pseudo-op
#: (the episode's first decision has no predecessor).
_META_OPS_V2: tuple[str, ...] = (
    "fire",
    "saturate",
    "declare",
    "handle",
    "claim",
    "extract",
    "macro",
)
_PREV_OPS_V2: tuple[str, ...] = ("start", *_META_OPS_V2)

#: The V2 schema — the shipped rows plus the second-order block.
#: Built on :data:`META_FEATURES` (the widened handler buckets
#: included) so ``x:seqh:*`` spans the same ``_bucket`` width the
#: featurizer emits.
META_ARENA_FEATURES_V2: tuple[str, ...] = (
    *META_FEATURES,
    *(f"x:seq:{p}>{c}" for p in _PREV_OPS_V2 for c in _META_OPS_V2),
    *(
        f"x:seqh:{p}:{b}"
        for p in _PREV_OPS_V2
        for b in range(_META_HANDLER_BUCKETS)
    ),
)


@dataclass
class MetaHist(PlayHist):
    """``PlayHist`` plus the last played move kind.

    ``record`` is the player's post-play hook — it sees only the op
    name (``LinearPolicy._record``), which is exactly the
    predecessor signal the ``x:seq`` block needs.  ``None`` before
    the episode's first play reads as ``"start"`` in the featurizer.
    """

    last_op: str | None = None

    def record(self, op: str) -> None:
        """Tally the play and remember its kind as the predecessor."""
        self.last_op = op
        super().record(op)


def _seq_handler_tags(action: ma.Action) -> list[str]:
    """Collect the handler tags a candidate binds."""
    if action.op in ("claim", "handle"):
        tag = action.params.get("handler")
        return [tag] if tag is not None else []
    if action.op == "macro":
        tags: list[str] = []
        for sub in action.params.get("seq") or ():
            if isinstance(sub, dict):
                op, params = (
                    str(sub.get("op")),
                    dict(sub.get("params") or {}),
                )
            else:
                op = str(getattr(sub, "op", ""))
                params = dict(getattr(sub, "params", {}) or {})
            if op in ("claim", "handle") and params.get("handler"):
                tags.append(params["handler"])
        return tags
    return []


def _prev_move_feats(
    _state: ma.MetaState, action: ma.Action, hist: Any, _row: dict
) -> dict[str, float]:
    """Emit the ``x:seq:*``/``x:seqh:*`` second-order block.

    ``x:seq:<prev>><cand>`` — last-move x candidate-kind (the pair
    term: ``declare>handle``, ``claim>extract`` are the composed
    line's edges); ``x:seqh:<prev>:<bucket>`` — last-move x
    handler-identity on the catalog-binding moves, so "after a
    declare, prefer *this* handler's handle" is expressible.  Both
    vary across the candidate set, so neither cancels in the
    softmax — the property every useful ``x:*`` term needs.
    """
    prev = getattr(hist, "last_op", None) or "start"
    out: dict[str, float] = {f"x:seq:{prev}>{action.op}": 1.0}
    for tag in _seq_handler_tags(action):
        out[f"x:seqh:{prev}:{_bucket(tag, 'handler')}"] = 1.0
    return out


#: V2 parts — the shipped row plus the predecessor block last (it
#: reads no ``st:*`` keys, so part order only fixes row ordering).
META_PARTS_V2: tuple = (*META_PARTS, _prev_move_feats)

#: The V2 spec — same enumerator/prior/lr, extended schema.  A
#: measurement arm, not a shipped preset: ``META_SPEC`` and its
#: ``meta_prior`` bake stay untouched.
META_SPEC_V2 = DomainSpec.build(
    features=META_ARENA_FEATURES_V2,
    legal=ma.legal_actions,
    parts=META_PARTS_V2,
    weights=meta_prior.META_ARENA_PLAYER_WEIGHTS,
    lr=_LR,
)


# ---------------------------------------------------------------------------
#  The player
# ---------------------------------------------------------------------------


class MetaLearnedPlayer(LearnedPlayer):
    """The meta-arena preset — the meta :class:`DomainSpec` bound.

    A :class:`~mlopt_discovery.players.LearnedPlayer` over
    ``meta_arena.legal_actions`` scored by :data:`META_SPEC`'s
    featurizer; *weights* defaults to the shipped trained prior
    :data:`mlopt_discovery.meta_prior.META_ARENA_PLAYER_WEIGHTS`
    (the imitation+RL bake — pass ``weights={}`` for the uniform
    cold start, or ``lawdata.META_ARENA_PLAYER_WEIGHTS`` for the
    older macro-move prior).  ``learn=False`` freezes the policy
    for eval; ``frozen`` returns a non-learning copy.
    """

    SPEC = META_SPEC

    def reset_episode(self) -> None:
        """Clear per-episode bookkeeping before a fresh board.

        ``LinearPolicy._observe`` detects the episode boundary only
        when the previous episode was observed past step 0
        (``prev_steps > 0``).  An episode that ends on its first
        move — the degenerate ``extract``-early line this board
        rewards — leaves ``prev_steps == 0``, so the next board
        inherits the stale played set and the terminal ``extract``
        stays masked for a whole episode (training and imitation
        only; eval arms are fresh ``frozen`` copies).  The boundary
        cannot be inferred inside ``_observe`` — a ``steps == 0``
        re-observation is identical whether the board advanced or
        the same state was queried twice — so the drivers
        (:func:`train_player`, :func:`imitate_oracle`) reset
        explicitly.  The upstream fix has since landed in gamekit
        (``_observe`` treats a ``steps == 0`` re-observe with a
        non-empty played mask as a fresh board); this explicit reset
        stays as belt-and-braces for older pins and remains the
        honest boundary hook for callers.
        """
        self._prev_steps = -1  # forces _observe's boundary branch

    def fresh_hist(self) -> MetaHist:
        """Return the predecessor-carrying history (V2's channel).

        A :class:`MetaHist` is a :class:`PlayHist` plus ``last_op`` —
        the shipped spec's ``PlayRates`` parts read it unchanged, and
        the V2 ``x:seq`` block reads the predecessor.  Returning it
        unconditionally keeps one hist shape across specs.
        """
        return MetaHist()


def _key(action: ma.Action) -> tuple:
    """Stable dedup key — the played-move mask, replayable."""
    params = action.params
    return (
        action.op,
        repr(sorted(params.items(), key=lambda kv: kv[0])),
    )


# ---------------------------------------------------------------------------
#  Training and evaluation — disjoint case streams
# ---------------------------------------------------------------------------


def gen_cases(seed: int, n: int, *, coverage: bool = True) -> list:
    """Seeded case variants — the mixed board's paying structures.

    Kinds: ``silu`` sites under a ruleset with ``silu_fold``
    stripped (the declare→handle line is the only path back — silu
    stays in the ambient vocabulary through ``silu_expand`` /
    ``silu_mul_form``); ``swiglu`` sites whose kernel enters the
    bound explicitly; ``softsign`` sites under ``sans`` where the
    handler kernel is *out* of vocabulary — an honest negative, the
    line must decline; and ``control`` terms under ``DEFAULT`` where
    nothing pays and the best play is a cheap extract.  Case tuples
    follow ``meta_probe``'s arity — ``(name, term, rules, decls,
    supported)``.

    The corpus-side coverage arm (T-009) adds the handler tags and
    site multiplicities the five base kinds never produced:

    * ``square`` — ``mul(t, t)`` sites under ``nosq``
      (``sans`` minus ``mul_square``): the third shipped handler's
      concrete pattern occurs, so ``claim(square)`` is *offered* —
      but the fold is op-count neutral (``mul(X,X)`` and
      ``square(X)`` both price as one op over the shared ``X``), so
      the oracle declines.  The corpus's first claimable-but-
      unprofitable boards: every base-kind claim tag pays, so the
      policy never had to price a claim it must refuse.
    * ``square_mix`` / ``tri`` — a paying silu (and swiglu) site
      *alongside* a square site: ``claim(square)`` stays offered
      and must lose to the paying tag — per-tag selectivity the
      base corpus never exercised.
    * ``dual`` / ``dual_swiglu`` — two disjoint silu-family sites
      (``add(silu(x), swiglu(y,z))`` and the two-swiglu variant):
      one ``claim``/``saturate`` reaches both, so the oracle's argmax
      moves off the ``claim_*_extract`` compressions onto the
      spelled ``declare→saturate→handle→extract`` macro — the first
      corpus boards whose winning offer is a four-step seq.

    The coverage arm (``corpus_cov``, Stage-1 lever 1) appends the
    gap-targeting kinds when *coverage* is true (the default):
    ``law_*`` spelled forms arming the shipped apply-laws,
    ``scan`` carriers under the scan-extended ruleset, and the
    ``ew_gap``/``reduce_gap``/``view_gap``/``kernel_ops``/
    ``mixed_gap`` vocabulary holes the ten hand kinds never
    produced.  ``coverage=False`` reproduces the ten-kind stream
    exactly — the pre-extension corpus, kept as the A/B control.
    """
    rng = random.Random(seed)
    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    z = Var("z", TensorType((4, 4)))
    w = Var("w", TensorType((4, 4)))
    v = Var("v", TensorType((4, 4)))
    sans = DEFAULT - DEFAULT.named("silu_fold", "softsign_fold")
    nosq = sans - DEFAULT.named("mul_square")
    silu_spelled = Op.make("mul", x, Op.make("sigmoid", x))
    swiglu_sup = ma._supported_or(None, silu_spelled, sans) | {"swiglu"}
    silu_of = lambda a: Op.make("mul", a, Op.make("sigmoid", a))  # noqa: E731
    swiglu_of = lambda a, b: Op.make("mul", silu_of(a), b)  # noqa: E731
    square_of = lambda a: Op.make("mul", a, a)  # noqa: E731
    builders = [
        ("silu", lambda: (silu_spelled, sans, None)),
        (
            "silu_nested",
            lambda: (
                Op.make("tanh", Op.make("add", silu_spelled, y)),
                sans,
                None,
            ),
        ),
        (
            "swiglu",
            lambda: (
                Op.make("mul", silu_spelled, y),
                sans,
                swiglu_sup,
            ),
        ),
        (
            "softsign",
            lambda: (
                Op.make(
                    "div",
                    x,
                    Op.make("add", Op.make("abs", x), Const(1)),
                ),
                sans,
                None,
            ),
        ),
        (
            "control",
            lambda: (
                Op.make(
                    "add", Op.make("mul", x, y), Op.make("matmul", x, y)
                ),
                DEFAULT,
                None,
            ),
        ),
        # --- T-009 corpus coverage: the missing handler tags -------
        (
            "square",
            lambda: (
                square_of(Op.make("tanh", x))
                if rng.random() < 0.5
                else Op.make(
                    "add",
                    square_of(Op.make("add", x, y)),
                    z,
                ),
                nosq,
                None,
            ),
        ),
        (
            "square_mix",
            lambda: (
                Op.make("add", silu_spelled, square_of(y)),
                nosq,
                swiglu_sup,
            ),
        ),
        (
            "dual",
            lambda: (
                Op.make("add", silu_spelled, swiglu_of(y, z)),
                sans,
                swiglu_sup,
            ),
        ),
        (
            "dual_swiglu",
            lambda: (
                Op.make("add", swiglu_of(y, z), swiglu_of(w, v)),
                sans,
                swiglu_sup,
            ),
        ),
        (
            "tri",
            lambda: (
                Op.make(
                    "add",
                    Op.make("add", silu_spelled, swiglu_of(y, z)),
                    square_of(w),
                ),
                nosq,
                swiglu_sup,
            ),
        ),
    ]
    if coverage:
        builders += [
            (kind, (lambda b: lambda: b(rng))(build))
            for kind, build in corpus_cov.coverage_builders()
        ]
    out = []
    for i in range(n):
        kind, build = builders[rng.randrange(len(builders))]
        term, rules, sup = build()
        out.append((f"gen{i}:{kind}", term, rules, (), sup))
    return out


def _board(case: list | tuple, cf: Any, max_nodes: int) -> ma.MetaArena:
    """Build the arena for one case tuple (probe arity)."""
    rules = case[2] if len(case) > 2 else None
    sup = case[4] if len(case) > 4 else None
    return ma.MetaArena(
        case[1],
        rules,
        cost_fn=cf,
        supported=sup,
        max_nodes=max_nodes,
    )


def train_player(
    player: MetaLearnedPlayer,
    cases: Iterable,
    *,
    budget: int,
    cost_fn: Any = None,
    max_nodes: int = 20_000,
) -> list[float]:
    """Play one episode per case; REINFORCE on each trajectory total."""
    cf = cost_fn or count_cost
    totals: list[float] = []
    reset = getattr(player, "reset_episode", None)
    for case in cases:
        if reset is not None:
            reset()
        traj = ma.run_episode(
            _board(case, cf, max_nodes), player, budget
        )
        player.finish_episode(
            traj.total, rewards=[r.reward for r in traj.reports]
        )
        totals.append(traj.total)
    return totals


# ---------------------------------------------------------------------------
#  The oracle — the known-good line per board, found by simulation
# ---------------------------------------------------------------------------
#
#  Imitation warm-start needs demonstrations, and the meta-arena's
#  paying structure is *compositional*: no single move reveals
#  whether ``claim → extract`` beats ``extract`` until the claim has
#  fired and the extraction repriced.  The oracle is therefore a
#  small candidate set played out on fresh boards — the same
#  referee the players face, so a "demonstrated" line is one the
#  board itself scored best among the known shapes:
#
#  * ``extract`` — the degenerate stop;
#  * ``saturate(b) → extract`` per advertised budget arm;
#  * the *offered* macro compressions — ``macro_moves.offered``'s
#    viable ``lawdata.MACROS`` lines replayed as single refereed
#    steps (``[macro]`` when the seq ends terminal, else
#    ``macro → extract``).  A macro is the arena's own compression
#    of a spelled line — one ``step`` charge and one budget tick for
#    the whole seq — so the spelled ``claim → extract`` collects
#    0.05 less than the ``claim_silu_extract`` macro covering the
#    same rewrite.  Only *offered* macros are candidates: they
#    are the ones ``legal_actions`` actually enumerates, so an
#    oracle pick stays imitable (``imitate`` matches on the legal
#    set's action key);
#  * ``claim(tag) → extract`` per occurring handler tag — the
#    fused composed line;
#  * ``claim(tag) → saturate → extract`` — claim plus a closure
#    pass (the claim fires its minted rule once; saturate can
#    still find cross-rule reach) — unbounded and per budget arm;
#  * ``saturate → claim(tag) → extract`` — the closure-first
#    ordering, both saturate arms;
#  * every claim tag in sequence, then extract — with and without
#    a saturate (unbounded and bounded), saturate-first, and the
#    reversed claim order — the multi-claim compositions.
#
#  Ties resolve to the earlier candidate — cheapest line first —
#  so a claim that merely breaks even is not demonstrated as a win.
#  The ordering/budget variants are enumerated for completeness:
#  on the generated corpus they never beat the plain
#  ``claim → extract`` (the extra step costs 0.05 and the closure
#  finds no further reach), which is exactly the saturation
#  measurement this set exists to make.


def spelled_candidates(
    state: ma.MetaState, handlers: dict | None = None
) -> list[list[ma.Action]]:
    """Enumerate the *spelled* composed lines — declare→saturate→handle→extract.

    ``claim`` fuses this line into one refereed move; the spelled
    form is the primitive-move spelling it compresses.  One
    candidate per ``(occurring spec, covering handler)`` pair: the
    ``declare`` is the enumerator's own ``_fresh_fold`` (the same
    action ``_base_legal`` offers), the ``handle`` binds the fold
    to the handler whose concrete pattern equals the spec's —
    ``meta_arena._handle``'s own legality test, pre-filtered so a
    skeleton-collision can't fill the candidate set with declines.
    Where ``claim`` exists the spelled line is strictly dominated
    (one extra step of cost for the same payout); it is a
    *demonstration* candidate — imitation can replay it to teach
    the primitives the fused move hides — not an oracle contender.
    """
    table = lawdata.HANDLERS if handlers is None else handlers
    out: list[list[ma.Action]] = []
    for i in state.sites:
        spec = state.specs[i]
        canon = ma._canon_concrete(spec)
        for tag, h in table.items():
            if ma._canon_concrete(h["pattern"]) != canon:
                continue
            out.append(
                [
                    ma._fresh_fold(i, spec),
                    ma.Action.saturate(),
                    ma.Action.handle(f"foldabs_{i}", tag),
                    ma.Action.extract(),
                ]
            )
    return out


def _macro_line(action: ma.Action) -> list[ma.Action]:
    """Return the oracle candidate replaying one offered macro.

    A seq already ending in ``extract`` is terminal on its own —
    the candidate is the single macro step; anything shorter gets
    the terminal move appended so the line always reaches the
    referee's payout.
    """
    seq = action.params.get("seq") or ()
    last = seq[-1] if seq else {}
    last_op = (
        str(last.get("op"))
        if isinstance(last, dict)
        else str(getattr(last, "op", ""))
    )
    return (
        [action]
        if last_op == "extract"
        else [action, ma.Action.extract()]
    )


def _claim_candidates(
    tags: list[str], bounded: list
) -> list[list[ma.Action]]:
    """Enumerate the claim-side lines — per-tag and multi-claim.

    Per tag: ``claim → extract``, claim-plus-closure in both
    directions (``claim → saturate`` and ``saturate → claim``) over
    the unbounded arm and each bounded one.  Multi-tag boards add
    the all-claims compositions — plain, saturated (both
    directions, bounded arms) and the reversed claim order.
    """
    out: list[list[ma.Action]] = []
    for tag in tags:
        out.append([ma.Action.claim(tag), ma.Action.extract()])
        out.append(
            [
                ma.Action.claim(tag),
                ma.Action.saturate(),
                ma.Action.extract(),
            ]
        )
        for b in bounded:
            out.append(
                [
                    ma.Action.claim(tag),
                    ma.Action.saturate(budget=b),
                    ma.Action.extract(),
                ]
            )
            out.append(
                [
                    ma.Action.saturate(budget=b),
                    ma.Action.claim(tag),
                    ma.Action.extract(),
                ]
            )
        out.append(
            [
                ma.Action.saturate(),
                ma.Action.claim(tag),
                ma.Action.extract(),
            ]
        )
    if len(tags) > 1:
        claims = [ma.Action.claim(t) for t in tags]
        out.append([*claims, ma.Action.extract()])
        out.append([*claims, ma.Action.saturate(), ma.Action.extract()])
        out.append([ma.Action.saturate(), *claims, ma.Action.extract()])
        for b in bounded:
            out.append(
                [
                    *claims,
                    ma.Action.saturate(budget=b),
                    ma.Action.extract(),
                ]
            )
        out.append([*reversed(claims), ma.Action.extract()])
    return out


def _candidate_lines(
    state: ma.MetaState,
    handlers: dict | None = None,
    *,
    spelled: bool = False,
) -> list[list[ma.Action]]:
    """Enumerate the oracle's candidate lines for *state*."""
    from mlopt_discovery import macro_moves

    out: list[list[ma.Action]] = [[ma.Action.extract()]]
    for b in lawdata.META_SATURATE_BUDGETS:
        out.append([ma.Action.saturate(budget=b), ma.Action.extract()])
    # the offered macro compressions — the legal set's own one-step
    # spellings of the lines below (see the oracle comment above)
    for m in macro_moves.offered(state):
        out.append(_macro_line(m))
    out += _claim_candidates(
        list(state.claim_tags),
        [b for b in lawdata.META_SATURATE_BUDGETS if b is not None],
    )
    if spelled:
        out += spelled_candidates(state, handlers)
    return out


def oracle_line(
    case: list | tuple,
    *,
    cost_fn: Any = None,
    max_nodes: int = 20_000,
    budget: int = 24,
    handlers: dict | None = None,
    spelled: bool = False,
) -> tuple[float, list[ma.Action]]:
    """Return ``(reward, line)`` — the best candidate line simulated.

    Each candidate replays through :class:`ma.ScriptedPlayer` on a
    fresh board, so declined moves, spent enodes and the certified
    payout all count exactly as a player's episode would score them.
    *spelled* admits the primitive declare→saturate→handle→extract
    candidates (:func:`spelled_candidates`) into the argmax — they
    are dominated wherever ``claim`` exists, so this arm measures
    whether the spelled spelling ever wins on its own.
    """
    cf = cost_fn or count_cost
    st = _board(case, cf, max_nodes).observe()
    best_reward = float("-inf")
    best: list[ma.Action] = [ma.Action.extract()]
    for line in _candidate_lines(st, handlers, spelled=spelled):
        traj = ma.run_episode(
            _board(case, cf, max_nodes),
            ma.ScriptedPlayer(line),
            budget,
        )
        if traj.total > best_reward:
            best_reward = traj.total
            best = list(line)
    return best_reward, best


def plan_oracles(
    cases: Iterable,
    *,
    cost_fn: Any = None,
    max_nodes: int = 20_000,
    budget: int = 24,
    spelled: bool = False,
) -> list[tuple[float, list[ma.Action]]]:
    """Plan the oracle line for every case — ``(reward, line)`` pairs."""
    return [
        oracle_line(
            c,
            cost_fn=cost_fn,
            max_nodes=max_nodes,
            budget=budget,
            spelled=spelled,
        )
        for c in cases
    ]


def plan_spelled(
    cases: Iterable,
    *,
    cost_fn: Any = None,
    max_nodes: int = 20_000,
    budget: int = 24,
) -> list[tuple[float, list[ma.Action]] | None]:
    """Plan the best *spelled* line per case — ``None`` where absent.

    The demonstration source for the primitive-move curriculum:
    each entry is the best declare→saturate→handle→extract line the
    board scores (or ``None`` on boards with no coverable occurring
    spec), independent of whether ``claim`` would beat it.
    """
    cf = cost_fn or count_cost
    out: list[tuple[float, list[ma.Action]] | None] = []
    for case in cases:
        st = _board(case, cf, max_nodes).observe()
        cands = spelled_candidates(st)
        if not cands:
            out.append(None)
            continue
        best_reward = float("-inf")
        best: list[ma.Action] = []
        for line in cands:
            traj = ma.run_episode(
                _board(case, cf, max_nodes),
                ma.ScriptedPlayer(line),
                budget,
            )
            if traj.total > best_reward:
                best_reward = traj.total
                best = list(line)
        out.append((best_reward, best))
    return out


def _replay(line: list[ma.Action]) -> Any:
    """Return a state-blind player that walks *line* in order."""
    it = iter(line)

    def go(_state: ma.MetaState) -> ma.Action | None:
        return next(it, None)

    return go


def imitate_oracle(
    player: MetaLearnedPlayer,
    cases: list,
    plans: list[tuple[float, list[ma.Action]]],
    *,
    cost_fn: Any = None,
    max_nodes: int = 20_000,
    budget: int = 24,
) -> dict[str, int]:
    """Warm-start *player* by replaying each case's oracle line.

    One ``imitate_policy`` pass: the expert replays the simulated
    best line on the live board while ``player.imitate`` takes a
    cross-entropy step toward each demonstrated move — demonstrations
    of *both* sides of the decision (claim where it pays, cheap
    extract where it doesn't).  Returns the engine's
    ``{shown, imitated}`` counts.
    """
    from gamekit import engine as eng

    cf = cost_fn or count_cost
    lines = [line for _reward, line in plans]
    reset = getattr(player, "reset_episode", None)

    def expert(i: int) -> Any:
        # the factory runs at each episode's first decision — the
        # one hook that sees the board boundary before any pick
        if reset is not None:
            reset()
        return _replay(lines[i])

    def board_of(case: Any) -> ma.MetaArena:
        return _board(case, cf, max_nodes)

    return eng.imitate_policy(
        player, expert, list(cases), board_of, budget=budget
    )


def _line_ops(line: Iterable[ma.Action]) -> list[str]:
    """Flatten a plan's op sequence — macro seqs read as their steps."""
    ops: list[str] = []
    for a in line:
        if a.op == "macro":
            for sub in a.params.get("seq") or ():
                ops.append(
                    str(sub.get("op"))
                    if isinstance(sub, dict)
                    else str(getattr(sub, "op", ""))
                )
        else:
            ops.append(a.op)
    return ops


def _plan_rank(plan: tuple[float, list[ma.Action]]) -> int:
    """Curriculum tier of an oracle plan — easy composed lines first."""
    ops = _line_ops(plan[1])
    n_claims = sum(1 for o in ops if o == "claim")
    if n_claims == 1 and "saturate" not in ops:
        return 0  # claim → extract: the fused line, reachable early
    if n_claims:
        return 1  # multi-claim or claim+saturate: harder composition
    return 2  # nothing pays — the decline side, last


def curriculum_order(
    pairs: list[tuple[Any, tuple[float, list[ma.Action]]]],
) -> list[tuple[Any, tuple[float, list[ma.Action]]]]:
    """Stable-sort ``(case, plan)`` pairs easy→hard by plan tier.

    Tier 0 boards are exactly those where the compressed
    ``claim → extract`` line is the oracle's pick — the composed
    move is reachable before the stream spends episodes on boards
    where nothing pays.  Ties keep generation order (the seeded
    stream's own mixing).  Pairs stay aligned so the oracle plans
    can be replayed against the reordered stream.
    """
    return sorted(pairs, key=lambda cp: _plan_rank(cp[1]))


def case_availability(
    case: list | tuple,
    *,
    cost_fn: Any = None,
    max_nodes: int = 20_000,
) -> tuple[int, int]:
    """Return ``(claimable, spelled)`` — the board's line availability.

    *claimable* is ``MetaState.claimable`` — handler tags whose
    concrete pattern occurs (the ``claim`` precondition); *spelled*
    counts the primitive declare→saturate→handle→extract candidates
    :func:`spelled_candidates` finds — the spelled line's own
    precondition (an occurring spec a handler concretely covers).
    Both are *state* properties, not oracle outcomes: a board can
    offer the line and still not pay, which is exactly the
    distinction a claim-availability curriculum exploits.
    """
    st = _board(case, cost_fn or count_cost, max_nodes).observe()
    return st.claimable, len(spelled_candidates(st))


def availability_order(
    pairs: list[tuple[Any, tuple[float, list[ma.Action]]]],
    *,
    cost_fn: Any = None,
    max_nodes: int = 20_000,
) -> list[tuple[Any, tuple[float, list[ma.Action]]]]:
    """Stable-sort ``(case, plan)`` pairs by claim availability.

    Richest-first: boards offering the most composed lines (``claim``
    tags plus spelled-line candidates, lexicographic) lead the
    stream, decline boards trail — the claim-availability
    counterpart of :func:`curriculum_order`'s plan-tier sort.  The
    ordering reads board *state*, never the oracle's verdict, so it
    cannot leak which boards pay.
    """
    keyed = [
        (
            case_availability(
                cp[0], cost_fn=cost_fn, max_nodes=max_nodes
            ),
            cp,
        )
        for cp in pairs
    ]
    keyed.sort(key=lambda kv: kv[0], reverse=True)
    return [cp for _k, cp in keyed]


def _op_count(reports: Iterable, op: str) -> int:
    """Count *op* across reports — macro seqs read as their steps."""
    return sum(
        1 for r in reports for o in _line_ops([r.action]) if o == op
    )


def evaluate(
    arms: dict[str, Callable[[int], Any]],
    cases: list,
    *,
    budget: int,
    cost_fn: Any = None,
    max_nodes: int = 20_000,
) -> dict[str, list[dict]]:
    """Play every arm on the same cases; paired boards, probe rows."""
    cf = cost_fn or count_cost
    out: dict[str, list[dict]] = {}
    for name, mk in arms.items():
        rows: list[dict] = []
        for i, case in enumerate(cases):
            arena = _board(case, cf, max_nodes)
            traj = ma.run_episode(arena, mk(i), budget)
            t = traj.terminal
            rows.append(
                {
                    "case": case[0],
                    "steps": len(traj.reports),
                    "reward": round(traj.total, 3),
                    "cost": t.cost if t is not None else None,
                    "cert": t.certificate_ok if t is not None else None,
                    "used_declared": (
                        t.used_declared if t is not None else ()
                    ),
                    "claims": _op_count(traj.reports, "claim"),
                    "spelled": sum(
                        _op_count(traj.reports, o)
                        for o in ("declare", "handle")
                    ),
                }
            )
        out[name] = rows
    return out


def player_table(table: dict[str, list[dict]]) -> str:
    """Render per-arm means over the eval rows."""
    head = (
        f"{'player':<15} {'cases':>6} {'reward':>9} {'cost':>7}"
        f" {'cert':>5} {'decl':>5} {'steps':>6}"
    )
    lines = [head, "-" * len(head)]
    for name, rows in table.items():
        n = max(len(rows), 1)
        costs = [r["cost"] for r in rows if r["cost"] is not None]
        certs = sum(1 for r in rows if r["cert"])
        decls = sum(1 for r in rows if r["used_declared"])
        lines.append(
            f"{name:<15} {len(rows):>6} "
            f"{sum(r['reward'] for r in rows) / n:>9.2f} "
            f"{(sum(costs) / len(costs)) if costs else float('nan'):>7.2f} "
            f"{certs:>5} {decls:>5} "
            f"{sum(r['steps'] for r in rows) / n:>6.1f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Train on one case stream, evaluate on a disjoint one."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-cases", type=int, default=40)
    ap.add_argument("--eval-cases", type=int, default=10)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--max-nodes", type=int, default=20_000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=_LR)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="REINFORCE passes over the train stream",
    )
    ap.add_argument(
        "--imitate-epochs",
        type=int,
        default=0,
        help="oracle-replay warm-start passes before REINFORCE",
    )
    ap.add_argument(
        "--curriculum",
        action="store_true",
        help="order the train stream easy→hard by oracle plan",
    )
    ap.add_argument(
        "--availability",
        action="store_true",
        help="order the train stream by claim availability "
        "(state property, not oracle verdict)",
    )
    ap.add_argument(
        "--oracle-spelled",
        action="store_true",
        help="admit spelled declare→saturate→handle→extract lines "
        "into the oracle's argmax",
    )
    ap.add_argument(
        "--spelled-demo-epochs",
        type=int,
        default=0,
        help="imitate-epoch count replaying spelled lines before "
        "the oracle demonstrations",
    )
    ap.add_argument(
        "--prev-features",
        action="store_true",
        help="bind META_SPEC_V2 — the last-move x catalog "
        "second-order feature arm",
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    train = gen_cases(args.seed, args.train_cases)
    ev = gen_cases(args.seed + 7919, args.eval_cases)
    player = MetaLearnedPlayer(
        seed=args.seed,
        spec=META_SPEC_V2 if args.prev_features else None,
        temperature=args.temperature,
        lr=args.lr,
        epsilon=args.epsilon,
    )
    imitation: dict[str, int] | None = None
    plans: list | None = None
    if (
        args.imitate_epochs
        or args.curriculum
        or args.availability
        or args.spelled_demo_epochs
    ):
        plans = plan_oracles(
            train,
            max_nodes=args.max_nodes,
            budget=args.budget,
            spelled=args.oracle_spelled,
        )
    if args.curriculum and plans is not None:
        stream = curriculum_order(list(zip(train, plans, strict=True)))
        train = [c for c, _p in stream]
        plans = [p for _c, p in stream]
    if args.availability and plans is not None:
        stream = availability_order(
            list(zip(train, plans, strict=True)),
            max_nodes=args.max_nodes,
        )
        train = [c for c, _p in stream]
        plans = [p for _c, p in stream]
    spelled_plans = None
    if args.spelled_demo_epochs and plans is not None:
        spelled_plans = plan_spelled(
            train, max_nodes=args.max_nodes, budget=args.budget
        )
    if spelled_plans is not None and plans is not None:
        imitation = imitation or {"shown": 0, "imitated": 0}
        demo = [
            sp if sp is not None else p
            for sp, p in zip(spelled_plans, plans, strict=True)
        ]
        for _ in range(args.spelled_demo_epochs):
            r = imitate_oracle(
                player,
                train,
                demo,
                max_nodes=args.max_nodes,
                budget=args.budget,
            )
            imitation["shown"] += r["shown"]
            imitation["imitated"] += r["imitated"]
    if args.imitate_epochs and plans is not None:
        imitation = {"shown": 0, "imitated": 0}
        for _ in range(args.imitate_epochs):
            r = imitate_oracle(
                player,
                train,
                plans,
                max_nodes=args.max_nodes,
                budget=args.budget,
            )
            imitation["shown"] += r["shown"]
            imitation["imitated"] += r["imitated"]
    totals: list[float] = []
    for _ in range(args.epochs):
        totals.extend(
            train_player(
                player,
                train,
                budget=args.budget,
                max_nodes=args.max_nodes,
            )
        )
    arms: dict[str, Callable[[int], Any]] = {
        "scripted": lambda i: ma.ScriptedPlayer(),
        "greedy": lambda i: ma.GreedyPlayer(),
        "random": lambda i: ma.RandomPlayer(
            random.Random(args.seed + 10_000 + i)
        ),
        "learned": lambda i: player.frozen(args.seed + 20_000 + i),
        "learned-greedy": lambda i: player.frozen(
            args.seed + 30_000 + i, greedy=True
        ),
    }
    table = evaluate(
        arms, ev, budget=args.budget, max_nodes=args.max_nodes
    )
    print(  # stdout-compat
        "== meta-arena player — learned vs baselines =="
    )
    print(player_table(table))  # stdout-compat
    for name in ("learned", "learned-greedy"):
        rows = table[name]
        n = max(len(rows), 1)
        claims = sum(r["claims"] for r in rows)
        spelled = sum(r["spelled"] for r in rows)
        decl = sum(1 for r in rows if r["used_declared"])
        print(  # stdout-compat
            f"{name}: claims={claims} spelled-moves={spelled} "
            f"declared-use={decl}/{n}"
        )
    if imitation is not None:
        print(  # stdout-compat
            f"imitation: {imitation['imitated']}/"
            f"{imitation['shown']} demonstrated moves cloned"
        )
    if plans is not None:
        tiers = [_plan_rank(p) for p in plans]
        n_spelled = sum(
            1
            for _r, line in plans
            if "declare" in (ops := _line_ops(line)) and "handle" in ops
        )
        print(  # stdout-compat
            "oracle plans: "
            f"claim-line={tiers.count(0)} "
            f"hard={tiers.count(1)} decline={tiers.count(2)} "
            f"spelled-pick={n_spelled}"
        )
    decade = max(len(totals) // 10, 1)
    means = [
        round(sum(totals[i : i + decade]) / decade, 2)
        for i in range(0, len(totals) - decade + 1, decade)
    ]
    print(f"train decade means: {means}")  # stdout-compat
    if args.json:
        import json

        with open(args.json, "w") as f:
            json.dump(
                {
                    "table": table,
                    "train_totals": totals,
                    "weights": player.weights_dict(),
                    "imitation": imitation,
                },
                f,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

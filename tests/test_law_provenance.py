"""Law provenance — human-authored vs machine-admitted, and the ablation.

``mlopt_core.laws.provenance`` is the registry-level marker the
human-bar measurement needs (``project/retros/human-bar.md``): every
shipped law classifies as ``"human"`` or ``"machine"`` *without* a
per-law edit, and the machine sets name both the promoted objects now
in ``ALL_RULES`` (:data:`MACHINE_SHIPPED`) and the gauntlet-cleared
store objects that stayed unshipped (:data:`MACHINE_STORED`).

The behavioral pins bind the provenance to measured yield on the
held-out zoo (``mlopt_discovery.zoo`` — the disjointness property
``tests/test_discovery_zoo.py`` pins makes this a clean attribution
domain): the promoted pad strips carry the ``TimeCondConv`` win and
the machine arm loses the ``LoRAAdapter`` win, both exactly as the
provenance registry predicts.
"""

import mlopt_carriers  # noqa: F401 — register carriers before rules
import pytest
import torch
from mlopt_core.cost import dag_cost
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Op
from mlopt_core.laws import (
    ALL_RULES,
    MACHINE_LAWS,
    MACHINE_SHIPPED,
    MACHINE_STORED,
    RuleSet,
    law_provenance,
)
from mlopt_discovery import object_synthesis as synth
from mlopt_discovery import zoo as zoo_mod
from mlopt_orchestrator.optimize import Optimizer, default_rules
from mlopt_torch.backend import TorchBackend

_ZOO = {w.name: w for w in zoo_mod.zoo()}
_BY_NAME = {r.name: r for r in ALL_RULES}

#: The promotion table (``project/retros/promoted-laws.md`` plus the
#: ``mul_unsq_nat`` admission of ``project/retros/mul-unsq-nat.md`` plus
#: the verbatim-minted ``conv_bn_fold`` composite of
#: ``project/retros/minted-composite-default.md``) — the twelve
#: machine-admitted objects as they shipped into ``ALL_RULES``.
_PROMOTION_TABLE = frozenset(
    {
        "softsign_fold",
        "sdpa_fold_nomask",
        "sdpa_fold_div_nomask",
        "mul_unsq_nat",
        "mul_unsq_pad_l",
        "mul_unsq_pad_r",
        "sub_unsq_pad_l",
        "add_unsq_pad_r",
        "mul_reshape_inert_l",
        "transpose_noop",
        "chunk_single",
        "linear_channel_to_row_scale",
        "conv_bn_fold",
    }
)


def _p(op, *args, **attrs):
    return Op.make(op, *args, **attrs)


def _pad_cond_l() -> tuple:
    """The ``unsq-pad`` region as the auto-cond pair minted it."""
    return (
        "and",
        ("bcast-eq", ("unsq-out", "U", "A_dim"), "V", "U", "V"),
        ("bcast-into", "U", ("unsq-out", "U", "A_dim")),
    )


def _store_strips() -> list[Rewrite]:
    """The admitted view-strip store objects, spelled as recorded."""
    return [
        Rewrite(
            name="mul_unsqueeze_l_id",
            lhs=_p("mul", _p("unsqueeze", "U", dim="A_dim"), "V"),
            rhs=_p("mul", "U", "V"),
            cond=(
                "and",
                ("ones-before", "U", "A_dim"),
                ("bcast-eq", ("unsq-out", "U", "A_dim"), "V", "U", "V"),
            ),
        ),
        Rewrite(
            name="sub_unsqueeze_l_id",
            lhs=_p("sub", _p("unsqueeze", "U", dim="A_dim"), "V"),
            rhs=_p("sub", "U", "V"),
            cond=_pad_cond_l(),
        ),
        Rewrite(
            name="add_unsqueeze_l_id",
            lhs=_p("add", _p("unsqueeze", "U", dim="A_dim"), "V"),
            rhs=_p("add", "U", "V"),
            cond=_pad_cond_l(),
        ),
        Rewrite(
            name="mul_unsqueeze_r_id",
            lhs=_p("mul", "U", _p("unsqueeze", "V", dim="A_dim")),
            rhs=_p("mul", "V", "U"),
            cond=(
                "and",
                (
                    "bcast-eq",
                    ("unsq-out", "V", "A_dim"),
                    "U",
                    "V",
                    "U",
                ),
                ("bcast-into", "V", ("unsq-out", "V", "A_dim")),
            ),
        ),
        Rewrite(
            name="mul_transpose_l_id",
            lhs=_p(
                "mul",
                _p("transpose", "U", dim0="A_d0", dim1="A_d1"),
                "V",
            ),
            rhs=_p("mul", "U", "V"),
            cond=("axes-noop", "U", "A_d0", "A_d1"),
        ),
        Rewrite(
            name="add_transpose_l_id",
            lhs=_p(
                "add",
                _p("transpose", "U", dim0="A_d0", dim1="A_d1"),
                "V",
            ),
            rhs=_p("add", "U", "V"),
            cond=("axes-noop", "U", "A_d0", "A_d1"),
        ),
        Rewrite(
            name="mul_chunk_l_id",
            lhs=_p(
                "mul",
                _p(
                    "chunk",
                    "U",
                    chunks="A_chunks",
                    dim="A_dim",
                    index="A_index",
                ),
                "V",
            ),
            rhs=_p("mul", "U", "V"),
            cond=("attr-eq", "A_chunks", 1),
        ),
        Rewrite(
            name="mul_reshape_l_id",
            lhs=_p("mul", _p("reshape", "U", shape="A_shape"), "V"),
            rhs=_p("mul", "U", "V"),
            cond=(
                "bcast-eq",
                ("reshape-out", "U", "A_shape"),
                "V",
                "U",
                "V",
            ),
        ),
    ]


def _affd_step_lift() -> Rewrite:
    """The constructed diagonal-scan lift (``lift_object`` verbatim)."""
    cond_affd_state = (
        "or",
        ("op-in", "h", ("add", "sub", "apply", "applyd")),
        ("leaf", "h"),
    )
    return synth.lift_object(
        "affd_step_lift",
        ("add", ("mul", "a", "h"), "x"),
        ("aff_diag", "a", "x"),
        "applyd",
        state="h",
        cond=cond_affd_state,
    ).rule


def _search(w_name, rules):
    """Build the zoo model, run the search arm, return the result."""
    torch.manual_seed(0)
    model, x = _ZOO[w_name].build()
    model = model.eval().double()
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(model, x, rules=rules)
    return res, x


# ---------------------------------------------------------------------------
#  The marker itself — totality and the promotion table
# ---------------------------------------------------------------------------








def test_om_lift_collision_is_resolved_for_the_shipped_law():
    """``om_lift`` names two objects: the human-authored carrier law
    (``mlopt_carriers.om.OM_LIFT``, metavar-dim guard) and a
    constructed store object (literal ``dim=-1``).  The registry
    classifies *shipped* names, so the name resolves human; the store
    object's provenance rides its construction record."""
    assert law_provenance("om_lift") == "human"
    assert "om_lift" not in MACHINE_STORED
    carrier = {r.name for r in default_rules().rules} - {
        r.name for r in ALL_RULES
    }
    assert carrier  # carriers registered — the law is in the library
    assert "om_lift" in carrier




# ---------------------------------------------------------------------------
#  The ablation arms — the registry must actually split the ruleset
# ---------------------------------------------------------------------------


def _promoted_ruleset() -> RuleSet:
    """The promoted laws as a named subset of the composed default."""
    full = default_rules()
    return full & RuleSet(
        "promoted",
        [r for r in ALL_RULES if r.name in MACHINE_SHIPPED],
    )


def test_ablated_arms_partition_the_default_ruleset():
    """human = DEFAULT - machine-shipped; the two sets partition."""
    full = default_rules()
    promoted = _promoted_ruleset()
    human = full - promoted
    # the promoted *rules* — MACHINE_SHIPPED also names the 21
    # synop_*.expand premise axioms, which ship attached to the
    # prim folds but never as saturation rules
    shipped_rule_names = MACHINE_SHIPPED & {r.name for r in ALL_RULES}
    assert len(promoted.rules) == len(shipped_rule_names) == 34
    assert len(human.rules) + len(promoted.rules) == len(full.rules)
    assert {r.name for r in human.rules}.isdisjoint(MACHINE_SHIPPED)
    # machine-arm rules all classify machine under the registry.
    store = [*_store_strips(), _affd_step_lift()]
    machine_names = {r.name for r in promoted.rules} | {
        r.name for r in store
    }
    assert machine_names <= MACHINE_LAWS


# ---------------------------------------------------------------------------
#  Behavioral pins — the registry predicts measured yield
# ---------------------------------------------------------------------------


def test_machine_laws_deliver_the_timecondconv_win():
    """The promoted pad strips carry ``TimeCondConv`` — and the pure
    store spellings reproduce it: the machine arsenal alone lands the
    holdout's conditional-strip win where the human arm cannot."""
    store = RuleSet("store", [_store_strips()[0], _store_strips()[3]])
    res, _x = _search("TimeCondConv", store)
    fires = res.stats["rule_fires"]
    assert {"mul_unsqueeze_l_id", "mul_unsqueeze_r_id"} <= set(fires)
    cin = dag_cost(res.ir.root, res.cost_fn)
    cout = dag_cost(res.term, res.cost_fn)
    assert cout < cin - 1  # the pad drop is a real reduction


def test_human_arm_loses_the_machine_law_win():
    """Remove the promoted laws and ``TimeCondConv``'s win disappears —
    ``silu_expand`` still fires but nothing strips the pad."""
    human = default_rules() - _promoted_ruleset()
    res, _x = _search("TimeCondConv", human)
    fires = res.stats["rule_fires"]
    assert set(fires).isdisjoint(MACHINE_LAWS)
    cin = dag_cost(res.ir.root, res.cost_fn)
    cout = dag_cost(res.term, res.cost_fn)
    assert cout == pytest.approx(cin)


def test_store_lift_wins_where_the_default_library_misses():
    """``affd_step_lift`` — admitted, never shipped — folds AdaLN's
    ``ln(x)·(1+scale)+shift`` into one ``applyd`` and the verified
    extraction drops ~4 %: the machine store beats ``DEFAULT_RULES``
    on a held-out block."""
    store = RuleSet("store", [_affd_step_lift()])
    res, _x = _search("AdaLNBlock", store)
    assert "affd_step_lift" in res.stats["rule_fires"]
    cin = dag_cost(res.ir.root, res.cost_fn)
    cout = dag_cost(res.term, res.cost_fn)
    assert cout < cin
    # The shipped default now carries ``decay_state_lift`` — the
    # machine-derived carrier fold — so the DEFAULT arm reaches an
    # applyd member too.  ``affd_step_lift``'s historical edge is
    # subsumed, not lost: the store still fires it, and the default
    # win now belongs to the shipped carrier law.
    res2, _x2 = _search("AdaLNBlock", default_rules())
    assert not {n for n in res2.stats["rule_fires"]} & {
        "affd_step_lift",
        "affd_lift",
    }
    assert "decay_state_lift" in res2.stats["rule_fires"]
    cout2 = dag_cost(res2.term, res2.cost_fn)
    assert cout2 < dag_cost(res2.ir.root, res2.cost_fn)


def test_lora_win_is_human_law_yield():
    """``assoc_linear`` — the human-authored composition law — owns the
    biggest zoo win; the machine arm (promoted + store) cannot
    reproduce it."""
    human = default_rules() - _promoted_ruleset()
    res, _x = _search("LoRAAdapter", human)
    assert "assoc_linear" in res.stats["rule_fires"]
    cin = dag_cost(res.ir.root, res.cost_fn)
    cout = dag_cost(res.term, res.cost_fn)
    assert cout < cin
    machine = _promoted_ruleset().union(
        RuleSet("store", [*_store_strips(), _affd_step_lift()]),
        override=True,
    )
    res2, _x2 = _search("LoRAAdapter", machine)
    cout2 = dag_cost(res2.term, res2.cost_fn)
    assert cout2 == pytest.approx(dag_cost(res2.ir.root, res2.cost_fn))

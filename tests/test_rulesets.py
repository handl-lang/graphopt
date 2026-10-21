"""Composable rule sets — plan 0009.

Covers the ``RuleSet`` value (name-keyed +/−/& algebra, ``named`` /
``tagged`` subsets, the ``priorities`` scheduling map), the tag
assignments on the shipped rules, the named presets and their
compositions, the orchestrator's ``rules`` resolution, and the
parity contract: ``DEFAULT`` reproduces the retired
``ruleset="all"`` saturation behaviour on a fixture.
"""

import pytest
import torch
import torch.nn as nn
from mlopt_core import laws
from mlopt_core.laws import (
    COMM_ADD,
    DISTRIBUTE_MUL,
    ID_ADD,
    RuleSet,
    all_rules,
    tags,
)

SUBSUMED_NAMES = {
    "swiglu_fuse",
    "parallel_mul_fuse",
    "qkv_fuse",
    "qkv_fuse_asym",
}
SYMMETRY_NAMES = {
    "comm_add",
    "comm_mul",
    "assoc_add",
    "assoc_mul",
    "linear_row_scale",
    "linear_row_scale_rev",
    "linear_channel_scale",
    "linear_channel_scale_rev",
}


def _a() -> RuleSet:
    return RuleSet("a", (COMM_ADD, ID_ADD))


def _b() -> RuleSet:
    return RuleSet("b", (DISTRIBUTE_MUL,))


# ---------------------------------------------------------------------------
#  The value itself
# ---------------------------------------------------------------------------




















# ---------------------------------------------------------------------------
#  Tag assignments on the shipped rules
# ---------------------------------------------------------------------------




def test_carrier_package_tags():
    from mlopt_carriers.decode_laws import DECODE_LAWS
    from mlopt_carriers.om import OM_LAWS
    from mlopt_carriers.trace import TRACE_LAWS
    from mlopt_carriers.xcarrier import XC_LAWS

    assert all(
        r.tags == {tags.CARRIER, tags.DECODE} for r in DECODE_LAWS
    )
    assert all(r.tags == {tags.CARRIER} for r in OM_LAWS)
    assert all(r.tags == {tags.CARRIER} for r in TRACE_LAWS)
    assert all(r.tags == {tags.CARRIER} for r in XC_LAWS)




# ---------------------------------------------------------------------------
#  Presets
# ---------------------------------------------------------------------------












# ---------------------------------------------------------------------------
#  Orchestrator resolution + the composed default
# ---------------------------------------------------------------------------




def test_default_rules_composes_carriers():
    from mlopt_carriers.decode_laws import DECODE_LAWS
    from mlopt_carriers.om import OM_LAWS
    from mlopt_carriers.trace import TRACE_LAWS
    from mlopt_carriers.xcarrier import XC_LAWS
    from mlopt_orchestrator import DEFAULT_RULES, default_rules

    assert DEFAULT_RULES is default_rules()
    names = {r.name for r in DEFAULT_RULES}
    core = {r.name for r in laws.DEFAULT}
    assert core <= names
    # the carrier-package families (om / trace / xc / decode)
    # compose in on top of the core default.
    for fam in (OM_LAWS, TRACE_LAWS, XC_LAWS, DECODE_LAWS):
        assert {r.name for r in fam} <= names
    # lazy attrs on the package resolve.
    import mlopt_carriers

    assert mlopt_carriers.CARRIERS.name.startswith("om+")
    assert mlopt_carriers.OM_RULES.name == "om"
    assert mlopt_carriers.TRACE_RULES.name == "trace"
    assert mlopt_carriers.XC_RULES.name == "xc"
    assert mlopt_carriers.DECODE_RULES.name == "decode"
    with pytest.raises(AttributeError):
        _ = mlopt_carriers.bogus_name






def test_composer_joint_rules():
    from mlopt_orchestrator.optimize import default_rules
    from mlopt_torch.composer import _JOINT_DEFAULT, _joint_rules

    # the pipeline default widens to the layout-inclusive set.
    assert _joint_rules(None) is _JOINT_DEFAULT
    assert _joint_rules(default_rules()) is _JOINT_DEFAULT
    # a caller-chosen set is honoured as-is.
    assert _joint_rules(laws.CATEGORICAL) is laws.CATEGORICAL
    assert set(_JOINT_DEFAULT) >= set(laws.WITH_LAYOUT) - {
        r for r in laws.WITH_LAYOUT if tags.SUBSUMED in r.tags
    }


# ---------------------------------------------------------------------------
#  Parity — DEFAULT reproduces the retired ruleset="all" path
# ---------------------------------------------------------------------------


class _PairMLP(nn.Module):
    """Two parallel branches — exercises the pairing/fusion rules."""

    def __init__(self):
        super().__init__()
        self.g = nn.Linear(16, 32, bias=False)
        self.u = nn.Linear(16, 32, bias=False)
        self.o = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        import torch.nn.functional as F

        return self.o(F.silu(self.g(x)) * self.u(x))


def test_default_parity_with_legacy_all():
    """``DEFAULT`` reproduces the retired ``ruleset="all"`` behaviour.

    The legacy switch saturated with ``all_rules() - _SUBSUMED`` (53
    rules = today's ``SIMPLIFICATION + CATEGORICAL`` presets).
    ``DEFAULT`` drops the ``SYMMETRY`` generators and adds the core
    carrier laws by design; on this fixture (no scan sources, no
    scale hoists) saturation parity is the same extracted term and
    the same fires for every rule the two sets share — so the preset
    change cannot silently alter a default-pipeline result.
    """
    from mlopt_core.ir import op_repr
    from mlopt_orchestrator import search
    from mlopt_torch.adapters import TorchSource
    from mlopt_torch.torch_bridge import ir_to_torch_module

    torch.manual_seed(0)
    m = _PairMLP().eval().double()
    x = torch.randn(2, 16, dtype=torch.float64)

    legacy = RuleSet(
        "legacy-all",
        tuple(r for r in all_rules() if r.name not in SUBSUMED_NAMES),
    )
    new = RuleSet(
        "new-default",
        tuple(r for r in laws.DEFAULT if r.name in set(legacy)),
    )
    res_legacy = search(
        m, x, source=TorchSource(), rules=legacy, max_iterations=6
    )
    res_new = search(
        m, x, source=TorchSource(), rules=new, max_iterations=6
    )

    # Same extracted member and same fires on the shared rules.
    assert op_repr(res_new.term) == op_repr(res_legacy.term)
    shared = {r.name for r in new}
    assert res_new.stats["rule_fires"] == {
        k: v
        for k, v in res_legacy.stats["rule_fires"].items()
        if k in shared
    }

    # and the two terms genuinely agree on inputs.
    from mlopt_core.ir import IR

    tm_old = ir_to_torch_module(
        IR(root=res_legacy.term, inputs=res_legacy.ir.inputs),
        param_values=res_legacy.param_values,
    )
    tm_new = ir_to_torch_module(
        IR(root=res_new.term, inputs=res_new.ir.inputs),
        param_values=res_new.param_values,
    )
    with torch.no_grad():
        assert torch.allclose(tm_old(x), tm_new(x), atol=1e-12)
        assert torch.allclose(m(x), tm_new(x), atol=1e-12)


def test_full_ruleset_saturation_smoke():
    """The composed values drive ``EGraph.run`` / ``search`` directly."""
    from mlopt_orchestrator import search
    from mlopt_torch.adapters import TorchSource

    torch.manual_seed(0)
    m = nn.Linear(8, 8, bias=False).double()
    x = torch.randn(2, 8, dtype=torch.float64)
    res = search(
        m,
        x,
        source=TorchSource(),
        rules=laws.DEFAULT.with_priorities(id_add=laws.LATE),
        max_iterations=3,
    )
    # priorities exist in the data: a LATE rule sorts after NORMAL —
    # the scheduler that *consumes* the map is plan 0010.
    rs = laws.DEFAULT.with_priorities(id_add=laws.LATE)
    assert rs.priority_of("id_add") == laws.LATE
    assert rs.priority_of("assoc_matmul") == laws.NORMAL
    assert "rule_fires" in res.stats

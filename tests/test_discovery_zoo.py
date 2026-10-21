"""Tests for ``mlopt_discovery.zoo`` — the held-out model zoo.

The zoo is the holdout set for the honest yield question
(``project/retros/model-zoo-yield.md``): real ``nn.Module``
architecture families the intake corpus never contained, used to
measure whether the shipped laws and the discovery pipeline produce
anything on programs nobody wrote to spell them.  The tests pin the
two properties that make the measurement honest:

* **disjointness** — no zoo name collides with an intake candidate
  (so ``--real-only`` and the full-corpus runs can never have seen
  these modules), nor with the ``_PURPOSE_BUILT`` ledger, the model
  corpus names, the bench law-case names, or the persisted
  cached ``intake_corpus.json`` records;
* **viability** — every workload's thunk builds and the module runs
  fp64 to a finite output, and the intake's own ``ingest`` boundary
  consumes the zoo unchanged (the holdout is a provenance property,
  not a different path).
"""

import pytest
import torch
from mlopt_discovery import intake as li
from mlopt_discovery import zoo as zoo_mod
from mlopt_torch.adapters import TorchSink

_ZOO = zoo_mod.zoo()


def test_zoo_is_nonempty_and_wellformed():
    names = [w.name for w in _ZOO]
    assert len(names) == len(set(names)) >= 10
    assert all(w.kind == "zoo" for w in _ZOO)
    assert all(not w.purpose_built for w in _ZOO)


def test_zoo_variants_wellformed():
    """The second draw mirrors the registry, disjointly named."""
    variants = zoo_mod.zoo_variants()
    names = [w.name for w in variants]
    assert len(names) == len(set(names)) == len(_ZOO)
    assert all(w.kind == "zoo" for w in variants)
    assert all(not w.purpose_built for w in variants)
    # The "V2" suffix is the disjointness mechanism — a variant that
    # dropped it would collide with its holdout-registry namesake.
    assert all(n.endswith("V2") for n in names)
    zoo_names = {w.name for w in _ZOO}
    assert zoo_names.isdisjoint(names)
    intake_names = {w.name for w in li.candidates()}
    assert set(names).isdisjoint(intake_names)
    assert set(names).isdisjoint(li._PURPOSE_BUILT)


@pytest.mark.parametrize(
    "w",
    zoo_mod.zoo_variants(),
    ids=[w.name for w in zoo_mod.zoo_variants()],
)
def test_zoo_variant_runs_fp64(w):
    """Every variant builds and produces a finite fp64 output."""
    torch.manual_seed(0)
    model, x = w.build()
    feed = x if isinstance(x, tuple) else (x,)
    model = model.eval().double()
    with torch.no_grad():
        out = model(*feed)
    assert torch.isfinite(out).all()


def test_zoo_disjoint_from_intake_registry():
    """The holdout property: the intake has never seen these names."""
    zoo_names = {w.name for w in _ZOO}
    intake_names = {w.name for w in li.candidates()}
    assert zoo_names.isdisjoint(intake_names)
    # The ledger only marks intake workloads, but pin the overlap
    # explicitly — a name shared with _PURPOSE_BUILT would let a
    # future edit silently reclassify a zoo member.
    assert zoo_names.isdisjoint(li._PURPOSE_BUILT)


def test_zoo_disjoint_from_model_and_bench_corpus():
    """The corpus is bench + models + intake — check the other two."""
    from mlopt_discovery.impact import _model_cases

    from bench.suites.correctness.law_bench import LAW_CASES

    zoo_names = {w.name for w in _ZOO}
    model_names = {name for name, _m, _x in _model_cases()}
    assert zoo_names.isdisjoint(model_names)
    assert zoo_names.isdisjoint(set(LAW_CASES))


def test_zoo_not_persisted_in_intake_sidefile():
    """No zoo term may already sit in the appended census file."""
    zoo_names = {w.name for w in _ZOO}
    persisted = {
        r["name"].removeprefix("intake:") for r in li.load_records()
    }
    assert zoo_names.isdisjoint(persisted)


@pytest.mark.parametrize("w", _ZOO, ids=[w.name for w in _ZOO])
def test_zoo_model_runs_fp64(w):
    """Every zoo workload builds and produces a finite fp64 output."""
    torch.manual_seed(0)
    model, x = w.build()
    feed = x if isinstance(x, tuple) else (x,)
    model = model.eval().double()
    with torch.no_grad():
        out = model(*feed)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "name", ["HighwayGate", "LoRAAdapter", "FiLMHead"]
)
def test_zoo_ingests_through_the_real_boundary(name):
    """The intake's own ``ingest`` consumes the zoo unchanged — the
    holdout is provenance, not a different export path."""
    torch.manual_seed(0)
    picked = [w for w in _ZOO if w.name == name]
    cases, records, rejections = li.ingest(
        cands=picked, sink=TorchSink()
    )
    assert not rejections
    assert [r["status"] for r in records] == ["ingested"]
    assert cases and cases[0].source == "intake"


#: The under-catalogued families the corpus expansion added — the
#: SSM/linear-attention scans, deployment-shape routing, quantization
#: and cache spellings, and the modern block details.  Pinned as an
#: explicit list so a registry reorder cannot silently drop one from
#: the ingestion contract below.
_EXPANSION_FAMILIES = [
    "MambaSelectiveScan",
    "GatedLinearAttention",
    "RWKVMixBlock",
    "HyenaLongConv",
    "TopKRouterMoE",
    "DeepSeekSigMoE",
    "STEQuantLinear",
    "PerChannelQuantMatmul",
    "KVCacheAttention",
    "GEGLUBlock",
    "SkipInitNet",
    "SoftcapLogitHead",
    "RotaryPair",
    # --- second expansion — wider mint surface -------------------
    # patterned attention: in-graph masks, strided + gathered keys
    "SlidingWindowAttention",
    "DilatedBlockAttention",
    "BlockSparseAttention",
    # depthwise-conv stack: channels-last blocks, pooled gates
    "ConvNeXtStage",
    "SqueezeExcite",
    "MBConvSEBlock",
    # fused-projection gates: same-view chunk tuples
    "GRUFusedGate",
    "GatedConvUnit",
    # manual norm spellings + the norm-free dynamic tanh
    "ManualGroupNorm",
    "InstanceNormBroadcast",
    "RMSNormPerHead",
    "DynamicTanhBlock",
    # positional/table ops: in-graph bias construction
    "ALiBiSlopeAttention",
    "RelPosTableAttention",
    "LearnedPosSlice",
    # second quant/kv shapes: grouped scales, paged cache
    "GroupedQuantLinear",
    "PagedKVAttention",
]


def test_zoo_expansion_families_registered():
    """Every expansion family sits in BOTH draws — ``zoo()`` and the
    ``*V2`` mirror — so the enumeration paths pick them up together."""
    zoo_names = {w.name for w in _ZOO}
    variant_names = {w.name for w in zoo_mod.zoo_variants()}
    assert set(_EXPANSION_FAMILIES) <= zoo_names
    assert {n + "V2" for n in _EXPANSION_FAMILIES} <= variant_names


@pytest.mark.parametrize("name", _EXPANSION_FAMILIES)
def test_zoo_expansion_family_produces_candidates(name):
    """Each new family exports through the real boundary and yields a
    candidate term with at least one op node — no rejection, and the
    lowered module verifies fp64 (status ``ingested``)."""
    torch.manual_seed(0)
    picked = [w for w in _ZOO if w.name == name]
    cases, records, rejections = li.ingest(
        cands=picked, sink=TorchSink()
    )
    assert not rejections
    assert [r["status"] for r in records] == ["ingested"]
    assert len(cases) == 1
    assert li._n_nodes(cases[0].term) >= 1


@pytest.mark.parametrize("name", _EXPANSION_FAMILIES)
def test_zoo_expansion_family_deterministic_shapes(name):
    """Two seeded builds of the same workload produce identical
    output shapes and values — the census never sees a flaky feed."""
    w = next(w for w in _ZOO if w.name == name)
    outs = []
    for _ in range(2):
        torch.manual_seed(0)
        model, x = w.build()
        feed = x if isinstance(x, tuple) else (x,)
        model = model.eval().double()
        with torch.no_grad():
            outs.append(model(*feed))
    assert outs[0].shape == outs[1].shape
    assert torch.equal(outs[0], outs[1])

"""Coverage tests for mlopt_core.rulecache — the JSON codec (attrs, terms,
bindings, rule records incl. guarded re-expression), fingerprinting,
the filesystem store/load contract, and the cached-synthesis key
derivation.  Real codec paths only; the synthesis itself is stubbed
where the cache mechanics are what's under test."""
# ruff: noqa: E741 — test-idiom unpacking

import functools
import json

import pytest
import torch

import mlopt_core.meta as meta
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Const, Op, Param, TensorType, Var
from mlopt_core.rulecache import (
    CACHE_VERSION,
    RuleCache,
    _dec_attr,
    _dec_binding,
    _dec_rule,
    _dec_term,
    _enc_attr,
    _enc_binding,
    _enc_rule,
    _enc_term,
    _hook_sig,
    _sha,
    cache_key,
    ruleset_fingerprint,
    seed_fingerprint,
    synthesize_rules_cached,
)


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _p(name, *shape):
    return Param(name, _T(*shape))


def _rw(name, lhs, rhs, **kw):
    return Rewrite(name=name, lhs=lhs, rhs=rhs, law="l", **kw)


# ---------------------------------------------------------------------------
#  attr / term / binding codec
# ---------------------------------------------------------------------------


def test_attr_codec_roundtrips_and_rejections():
    for v in (3, 2.5, "mvar", True, None):
        assert _dec_attr(_enc_attr(v)) == v
    assert _dec_attr(_enc_attr((1, 2))) == (1, 2)
    assert _dec_attr(_enc_attr([1, "a", True])) == [1, "a", True]
    assert _dec_attr(_enc_attr((1, [2, 3]))) == (1, [2, 3])
    # unserializable value → TypeError at encode
    with pytest.raises(TypeError):
        _enc_attr(torch.zeros(2))
    with pytest.raises(TypeError):
        _enc_attr(object())
    # corrupt marker dict → ValueError at decode
    with pytest.raises(ValueError):
        _dec_attr({"__nope__": 1})
    # non-dict marker passes through
    assert _dec_attr("plain") == "plain"






# ---------------------------------------------------------------------------
#  rule records
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
#  fingerprints and cache key
# ---------------------------------------------------------------------------


def _om_rules():
    from mlopt_core.laws import all_rules

    return all_rules()










# ---------------------------------------------------------------------------
#  filesystem store/load
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
#  synthesize_rules_cached — hit/miss mechanics
# ---------------------------------------------------------------------------



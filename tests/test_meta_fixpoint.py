"""``synthesize_rules`` fixpoint driver — ``rounds`` / ``max_emitted`` / ``on_round``.

The pairwise composer mints depth-2 composites over a fixed bank; the
fold families (conv+BN is the measured one — ``bench/drivers/probes/
conv_bn_fold_synth_probe.py`` gate B3, 26→61→242→944 over 4 rounds)
need iterated composition, where each round's emitted set re-enters
the bank.  These tests pin the fixpoint's contract on a synthetic
chain bank — ``f→g→h→k`` — where every round's mint is exact:

* round 1: ``f→h``, ``g→k`` (the two base pairs)
* round 2: ``f→k`` alone — a ``syn_``-parented composite (depth 3,
  unreachable in one pass).  The base-pair re-mints are *subsumed*:
  round 1's stratum sits in round 2's bank, and an unguarded composite
  subsumes its own re-mint.
* round 3: ``f→h``, ``g→k`` again — the stratum rolled off
  (``usable = base + latest``), so the base pairs mint afresh and the
  aggregate dedup does real work.
* ``max_emitted`` bounds the aggregate; ``on_round`` observes each
  stratum and a ``False`` return stops the loop early.
"""

import pytest
from mlopt_core import meta
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Op, TensorType, Var


def _chain_bank() -> list[Rewrite]:
    """``f→g``, ``g→h``, ``h→k`` — composable pairs in one direction."""
    return [
        Rewrite(
            name=f"{a}_to_{b}",
            lhs=Op.make(a, "x"),
            rhs=Op.make(b, "x"),
        )
        for a, b in (("f", "g"), ("g", "h"), ("h", "k"))
    ]


def test_rounds_one_is_the_single_pass():
    """``rounds=1`` (and the default) is exactly today's behaviour."""
    bank = _chain_bank()
    d_default = meta.synthesize_rules(bank, fuel=200)
    d_one = meta.synthesize_rules(bank, fuel=200, rounds=1)
    assert [r.name for r in d_default] == [r.name for r in d_one]
    # the chain mints exactly the two pairwise composites
    assert {meta._alpha_key(r.lhs, r.rhs) for r in d_one} == {
        meta._alpha_key(Op.make("f", "x"), Op.make("h", "x")),
        meta._alpha_key(Op.make("g", "x"), Op.make("k", "x")),
    }


def test_fixpoint_feeds_strata_back():
    """Round 2 mints a composite whose parent is itself synthesized."""
    seen: list[int] = []
    out = meta.synthesize_rules(
        _chain_bank(),
        fuel=200,
        rounds=3,
        on_round=lambda i, em: seen.append(len(em)),
    )
    # strata: 2, 1, 2 — f→k joins at round 2 while the base-pair
    # re-mints are subsumed by round 1's stratum (still in the bank);
    # at round 3 it rolled off and they mint afresh.  (The probe's
    # guarded families behave differently — a guarded composite cannot
    # subsume, so its strata grow instead of alternating.)
    assert seen == [2, 1, 2]
    # the returned aggregate dedups the cross-round re-mints
    assert len(out) == 3
    deep = [
        r
        for r in out
        if any(p.startswith("syn_") for p in (r.parents or ()))
    ]
    assert len(deep) == 1
    assert meta._alpha_key(deep[0].lhs, deep[0].rhs) == meta._alpha_key(
        Op.make("f", "x"), Op.make("k", "x")
    )
    # ...and the deep composite actually rewrites: f(t) -> k(t)
    leaf = Var("v", TensorType((2, 2)))
    out_t = meta.apply_rewrite_at(deep[0], Op.make("f", leaf), ())
    assert out_t == Op.make("k", leaf)


def test_max_emitted_bounds_the_aggregate():
    """The cap truncates mid-round and stops the fixpoint."""
    # cap inside round 1's mint
    out = meta.synthesize_rules(_chain_bank(), fuel=200, max_emitted=1)
    assert len(out) == 1
    # cap reached exactly at a stratum boundary -> no further round runs
    calls: list[int] = []
    out = meta.synthesize_rules(
        _chain_bank(),
        fuel=200,
        rounds=3,
        max_emitted=2,
        on_round=lambda i, em: calls.append(i),
    )
    assert len(out) == 2
    assert calls == [1]
    # a zero cap emits nothing at all
    assert (
        meta.synthesize_rules(_chain_bank(), fuel=200, max_emitted=0)
        == []
    )
    # a cap the mint never reaches leaves the rounds untouched
    out = meta.synthesize_rules(
        _chain_bank(), fuel=200, rounds=2, max_emitted=50
    )
    assert len(out) == 3


def test_on_round_veto_stops_early():
    """Returning ``False`` halts after the observed stratum."""
    strata: list[int] = []

    def watch(i: int, em: list[Rewrite]) -> bool:
        strata.append(len(em))
        return False  # converged enough — stop

    out = meta.synthesize_rules(
        _chain_bank(), fuel=200, rounds=4, on_round=watch
    )
    assert strata == [2]
    assert len(out) == 2


def test_empty_stratum_ends_the_loop():
    """A bank with no composable pairs stops after one empty round."""
    lonely = [Rewrite("solo", Op.make("f", "x"), Op.make("g", "x"))]
    strata: list[int] = []
    out = meta.synthesize_rules(
        lonely,
        fuel=200,
        rounds=5,
        on_round=lambda i, em: strata.append(len(em)),
    )
    assert out == []
    assert strata == [0]  # one round, nothing minted, done


def test_rounds_must_be_positive():
    """``rounds=0`` is a caller bug, not an empty fixpoint."""
    with pytest.raises(ValueError, match="rounds"):
        meta.synthesize_rules(_chain_bank(), fuel=200, rounds=0)

"""Premise-of-admitted-composite pay accounting — the enablement floor.

``CONV_FOLD_RULES`` truth-verify in the unit suite but refuse the
gauntlet at stage 4 (the enumeration cannot spell the conv channel
corner — ``project/retros/fixpoint-api-and-conv-sweep.md``).  Their
measured value is *enablement*: the conv+BN fold composite needs them
as derivation premises.  This probe lands the bounded measurement the
retro named, in order:

1. **Guard transport** — store the composite plus its transitive
   premise closure; ``admit_object`` rebuilds the minted hooks through
   the record's ``"guard"`` field (parents resolve as store rows,
   transitively).
2. **Truth** — the composite's own guard accepts the real conv+BN
   site and the pair evaluates fp64-equal; ``verify_law`` under
   ``ALL_RULES + closure`` replays the derivation (a premise-anchored
   proof — honest *modulo* the premises' own truth).
3. **Pay** — ``_probe`` fires the composite alone on the real case:
   does the fused member win extraction over the spelled term?
4. **Gauntlet** — ``run_gauntlet`` on the one-case corpus, end to end.

The gauntlet leg used to hang: ``pl.measure``'s view oracle ran
``oracle.synthesize`` over ``_binding_envs`` with only an *instance*
cap — the composite's ``derive`` vetoes every enumerated env, so the
sweep never produced a site and never stopped (measured >9 min on a
one-case corpus; the retro misattributed it to the 83-rule
``_reach``/derivability saturation, which is actually instant on this
term — 9 enodes to a fixed point).  ``oracle._MAX_BINDING_ENVS`` now
caps the enumeration itself, so the gauntlet completes in seconds
with an honest stage-by-stage verdict.

Run with
``.venv/bin/python bench/probes/conv_fold_premise_pay_probe.py``.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from mlopt_core import meta
from mlopt_core.ir import op_repr
from mlopt_core.laws import (
    ALL_RULES,
    BATCH_NORM_RULES,
    CONV_FOLD_RULES,
    POINTWISE_FACTOR_RULES,
)


class ConvBN(nn.Module):
    """``BatchNorm2d(Conv2d(x))`` in eval mode — the fold site."""

    def __init__(self, cin: int = 4, cout: int = 6, k: int = 3) -> None:
        """Build the conv (with bias — the shift the fold reads)."""
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=True)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv then batch norm."""
        return self.bn(self.conv(x))


def real_seed() -> tuple[Any, Any, dict[str, Any], Any]:
    """``(root, ir, source_tensors, input)`` for an eval-mode ConvBN."""
    from mlopt_torch.adapters import TorchSource

    mod = ConvBN().eval().double()
    with torch.no_grad():
        mod.bn.running_mean.copy_(torch.randn(6))
        mod.bn.running_var.copy_(torch.rand(6) * 2 + 0.25)
    x = torch.randn(1, 4, 8, 8, dtype=torch.float64)
    ir, tensors = TorchSource().to_ir(mod, x)
    return ir.root, ir, dict(tensors), x


def mint_fold(seed: Any, base: list[Any]) -> tuple[Any, list[Any]]:
    """Mint the conv+BN fold composite; return ``(fold, emitted)``."""
    folds: list[Any] = []

    def watch(rnd: int, emitted: list[Any]) -> bool:
        fs = [
            d
            for d in emitted
            if getattr(d.rhs, "op", None) == "conv2d"
            and "batch_norm" in op_repr(d.lhs)
        ]
        if fs:
            folds.extend(fs)
            return False
        return True

    out = meta.synthesize_rules(
        base,
        [seed],
        fuel=30000,
        numeric_check=False,
        rounds=4,
        on_round=watch,
    )
    return (folds[0] if folds else None), out


def ancestor_closure(rule: Any, universe: dict[str, Any]) -> list[Any]:
    """Collect the transitive premise set of *rule*, parents-first."""
    seen: set[str] = set()
    order: list[Any] = []

    def walk(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        r = universe.get(name)
        if r is None:
            return
        for p in meta.provenance(r):
            walk(p)
        order.append(r)

    for p in meta.provenance(rule):
        walk(p)
    return order


def _family(name: str) -> str:
    if name.startswith("conv_"):
        return "CONV_FOLD_RULES"
    if name.startswith("batch_norm"):
        return "BATCH_NORM_RULES"
    if name.startswith("syn_"):
        return "syn (intermediate)"
    return "ALL_RULES"


def main() -> int:
    """Run the bounded premise-pay measurement; print the account."""
    from mlopt_core.cost.params import dag_cost
    from mlopt_core.egraph import EGraph
    from mlopt_core.egraph.terms import _term_match
    from mlopt_core.laws.serialize import alpha_key
    from mlopt_discovery import evidence as ev
    from mlopt_discovery import pipeline as pl
    from mlopt_discovery.impact import TermCase, _cost_fn, _probe
    from mlopt_discovery.pipeline import _typed_probe
    from mlopt_discovery.shape_proposal import _sink
    from mlopt_discovery.verifier import verify_law

    t0 = time.monotonic()
    seed, ir, tensors, x = real_seed()
    # The mint base pins the pre-ship surface: ``conv_bn_fold`` now
    # ships in ALL_RULES, and without the exclusion the fixpoint's
    # first fold is ``syn_conv_bn_fold__comm_add_0`` — a trivially
    # different comm compose whose guard spec names the shipped
    # composite as a parent (which then needs the composite's own
    # premise rows stored to transport).  Excluding it reproduces the
    # admitted object this probe measures; the interaction is the
    # honest semantics — store rows for the whole premise closure or
    # no transport.
    base = (
        [r for r in ALL_RULES if r.name != "conv_bn_fold"]
        + list(BATCH_NORM_RULES)
        + list(CONV_FOLD_RULES)
        + list(POINTWISE_FACTOR_RULES)
    )
    fold, emitted = mint_fold(seed, base)
    assert fold is not None, "the fold composite did not mint"
    universe = {r.name: r for r in [*base, *emitted]}
    closure = ancestor_closure(fold, universe)
    print(
        f"mint: {len(emitted)} emitted; fold={fold.name[:74]}… "
        f"({time.monotonic() - t0:.1f}s)"
    )
    print(f"premise closure ({len(closure)}):")
    for r in closure:
        print(f"  {_family(r.name):<20} {r.name[:64]}")

    # -- 1. guard transport ----------------------------------------------
    conn = ev.connect(":memory:")
    for r in [*closure, fold]:
        ev.store_object(conn, r, kind="abstraction", cert=None)
    key = repr(alpha_key(fold.lhs, fold.rhs))
    rule, rec = ev.admit_object(conn, key)
    print(
        f"\n[transport] missing={rec['missing_hooks']} "
        f"transported={rec.get('transported')} "
        f"check={rule.check is not None} derive={rule.derive is not None}"
    )

    # -- 2. truth: real-site guarded eval + premise-anchored replay -------
    subst = _term_match(rule.lhs, seed)
    region = ev._guarded_evals(rule, iter([(subst, seed)]))
    print(
        f"[truth] real site: {region.equal}eq/{region.unequal}ne "
        f"({region.accepted} accepted)"
    )
    rhs_i = meta.apply_rewrite_at(rule, seed, ())
    res = verify_law(
        seed,
        rhs_i,
        list(ALL_RULES) + closure,
        max_iterations=8,
        max_nodes=20_000,
    )
    print(
        f"[truth] premise-derivable: {res.derivable} "
        f"replayable={res.replayable} steps={res.witness_steps} "
        f"witness={res.witness_rules}"
    )
    res0 = verify_law(
        seed, rhs_i, list(ALL_RULES), max_iterations=8, max_nodes=20_000
    )
    print(
        f"[truth] under shipped axioms only: derivable={res0.derivable}"
    )

    # -- 3. pay: fires + typedness + extraction delta ----------------------
    sink = _sink()
    cost_fn = _cost_fn(sink)
    case = TermCase(
        source="probe",
        name="ConvBN",
        term=seed,
        inputs=tuple(ir.inputs),
        feed=(x,),
        param_vals=dict(tensors),
    )
    prop = pl.Proposal(
        name=rule.name,
        lhs=rule.lhs,
        rhs=rule.rhs,
        family="object:abstraction",
        check=rule.check,
        derive=rule.derive,
    )
    f = _probe(case, prop.as_rule(), sink, cost_fn)
    audit = _typed_probe(case, prop, cost_fn)
    print(
        f"[pay] fires={f.fires} base={f.base_cost:.1f} out={f.out_cost:.1f} "
        f"changed={f.changed} verified={f.verified} "
        f"typed={audit.typed} ill={audit.ill} "
        f"({f.note})"
    )
    eg = EGraph()
    root = eg.add_term(seed)
    eg.run([rule], root, max_iterations=6, max_nodes=20_000)
    best = eg.extract_best(root, cost_fn)
    print(
        f"[pay] extraction: spelled={dag_cost(seed, cost_fn):.1f} "
        f"fused={dag_cost(best, cost_fn) if best is not None else None}"
    )

    # -- 4. the gauntlet, bounded ------------------------------------------
    # Post-ship the gauntlet refuses at NOVELTY — ``relation ==
    # "duplicate"``: the admitted object IS a library law now (its
    # alpha pair ships as ``conv_bn_fold``), and the novelty gate's job
    # is exactly to refuse respellings of shipped rules.  ``usable=
    # False`` here is the signature of "already admitted", not a
    # regression.
    corpus = ev.GauntletCorpus(
        real_terms=(seed,),
        probe=(case,),
        base_rules=tuple(ALL_RULES),
        census_op={},
        sink=sink,
        cost_fn=cost_fn,
    )
    tg = time.monotonic()
    rep = ev.run_gauntlet(conn, key, corpus=corpus)
    print(
        f"\n[gauntlet] {time.monotonic() - tg:.1f}s usable={rep.usable}"
    )
    for s in rep.stages:
        print(
            f"  [{'pass' if s.passed else 'FAIL'}] {s.name}: {s.detail}"
        )

    # -- the premise account ------------------------------------------------
    pays = f.fires > 0 and f.changed and f.out_cost < f.base_cost
    print(f"\n== premise account (dependent mint pays? {pays}) ==")
    for r in closure:
        if r.name.startswith("syn_"):
            continue
        print(f"  {r.name:<40} {_family(r.name)}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

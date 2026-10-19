"""Saturation determinism: the result must not depend on ``PYTHONHASHSEED``.

``EClass.nodes`` is a ``set`` of :class:`~mlopt_core.egraph.ENode`, and
``ENode.__hash__`` folds in the ``op`` name (a ``str``) and the attr
reprs — so the set's *iteration order* changes per process under hash
randomisation.  Any site that reads members in an order that can reach
a decision makes the whole saturation (and hence the extraction) vary
across processes: measured on the gate-3 staged saturation, the same
inputs produced 9726..9761 enodes across seeds, and the sharing-aware
local optimum moved 451,128 → 432,696.

The contract these tests pin: member reads that reach a decision go
through the canonical order ``(op, children, repr(attrs))``
(:func:`mlopt_core.egraph.types._member_key`).  ``test_*`` below cover
the three leaks that made the staged saturation seed-dependent — the
matcher's enumeration order (which decides what a ``rule_budget`` /
``_MATCH_CAP`` admits before the rule is suspended), the representative
chosen for a ``check``/``derive`` binding, and the congruence merge
order — plus an end-to-end check that runs the same small budgeted
saturation in fresh processes under different seeds.
"""

import os
import subprocess
import sys
import textwrap

from mlopt_core.egraph import EGraph
from mlopt_core.egraph.types import _member_key
from mlopt_core.ir import Op
from mlopt_core.laws.base import R

#: The budgeted saturation whose result must be seed-independent.  One
#: e-class holds ``N`` distinct ``add`` members, so the rule's match
#: enumeration has ``N`` substitutions to admit; ``enode_budget=1``
#: admits exactly the first one, i.e. whichever member the matcher
#: yields first.  Pre-fix that was the set's hash order (a different
#: member per process); the fix makes it the canonical order, so the
#: single created ``mul`` is always the RHS of ``add(L0, L0)``.
_SEED_SCRIPT = textwrap.dedent(
    """
    from mlopt_core.egraph import EGraph
    from mlopt_core.ir import Op
    from mlopt_core.laws.base import R

    rule = R(
        "add_to_mul", Op.make("add", "x", "y"), Op.make("mul", "x", "y")
    )
    eg = EGraph()
    base = eg.add_leaf("L0")
    adds = [
        eg.add_enode("add", (eg.add_leaf(f"L{i}"), base)) for i in range(16)
    ]
    first = adds[0]
    for e in adds[1:]:
        eg.union(first, e)
    cls = eg.find(first)
    eg.apply_rule(rule, cls, enode_budget=1)
    muls = sorted(
        n.children for n in eg.get_class(cls).nodes if n.op == "mul"
    )
    print(f"{eg.n_enodes}|{muls}")
    """
)


def _budgeted_graph(n: int = 16) -> tuple[EGraph, int]:
    """Build the ``n``-member ``add`` e-class and return ``(eg, class)``."""
    eg = EGraph()
    base = eg.add_leaf("L0")
    adds = [
        eg.add_enode("add", (eg.add_leaf(f"L{i}"), base))
        for i in range(n)
    ]
    first = adds[0]
    for e in adds[1:]:
        eg.union(first, e)
    return eg, eg.find(first)


def test_member_reads_are_in_canonical_order():
    """The matcher's member buckets are canonically ordered, not set-ordered."""
    eg, cls = _budgeted_graph()
    ec = eg.get_class(cls)
    canonical = sorted(ec.nodes, key=_member_key)
    assert len(canonical) == 16
    assert eg._nodes_of(ec, "add") == canonical
    # The bucket is cached on the class; a mutation must rebuild it in
    # canonical order too (``union`` invalidates ``by_op``).
    eg.add_enode("add", (eg.add_leaf("L99"), eg.add_leaf("L0")))
    assert eg._nodes_of(ec, "add") == canonical


def test_budgeted_apply_rule_lands_on_the_canonical_member():
    """``enode_budget=1`` admits the canonically-first match, every run."""
    eg, cls = _budgeted_graph()
    rule = R(
        "add_to_mul", Op.make("add", "x", "y"), Op.make("mul", "x", "y")
    )
    eg.apply_rule(rule, cls, enode_budget=1)
    # 16 leaves + 16 adds + the one created mul.
    assert eg.n_enodes == 33
    muls = [n for n in eg.get_class(cls).nodes if n.op == "mul"]
    assert len(muls) == 1
    canonical_first = sorted(eg.get_class(cls).nodes, key=_member_key)[
        0
    ]
    # The RHS reuses the matched member's children, so the created
    # ``mul`` is pinned to the member the matcher admitted first.
    assert muls[0].children == canonical_first.children


def test_representative_choice_is_canonical():
    """``any_term``/``_min_term`` break equal-size ties canonically.

    Both walks take the first acceptable member (``_min_term`` with a
    strict ``<``), so an equal-size tie would otherwise resolve to
    whichever member the set yielded first — and that term is what a
    rule's ``check``/``derive`` sees.
    """
    eg = EGraph()
    arg = eg.add_leaf("a")
    members = [eg.add_enode(f"o{i}", (arg,)) for i in range(8)]
    first = members[0]
    for e in members[1:]:
        eg.union(first, e)
    cls = eg.find(first)
    assert len(eg.get_class(cls).nodes) == 8
    assert eg.any_term(cls).op == "o0"
    term, size = eg._min_term(cls, {})
    assert term.op == "o0"
    assert size == 2


def _run_seeded(seed: int | None) -> str:
    """Run the budgeted saturation in a fresh process under *seed*."""
    env = dict(os.environ)
    if seed is None:
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = str(seed)
    proc = subprocess.run(
        [sys.executable, "-c", _SEED_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return proc.stdout.strip()


def test_budgeted_saturation_is_hash_seed_independent():
    """The same saturation in fresh processes must agree, seed or not."""
    results = {_run_seeded(seed) for seed in (0, 1, 2, 3, None)}
    assert results == {"33|[(0, 0)]"}, results

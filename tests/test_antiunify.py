"""ADR-0007 layer 1 — definitional atoms via anti-unification.

Tests the whole pipeline at unit scale: mining (repeated motifs, attr
metavariables, shared columns, structure-only mode), minting (the
registered op + intro/expand pair + ConstructedObject provenance),
admission (intro adds the atom member, refuse-never-drop), the folded
rule variants, elaboration (``unfold_term``), honest pricing
(``unfolding_cost``), the declared-object store record, and the fp64
verification of an unfolded extract through ``TorchSink``.
"""

from __future__ import annotations

from typing import Any

import pytest

from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import EGraph, Rewrite
from catopt_core.ir import Op, TensorType, Var, generator, op_repr
from catopt_core.meta import match_pattern
from catopt_discovery.antiunify import (
    DEFINITIONAL,
    _expansion_body_cost,
    admit_atoms,
    atom_map,
    fold_rule,
    fold_ruleset,
    mint_atom,
    mint_atoms,
    mine_candidates,
    store_atoms,
    unfold_term,
    unfolding_cost,
)


def _v(name, *shape):
    return Var(name, TensorType(shape))


def _motif_graph():
    """add(mul(?, ?), ?) on three distinct instances under a pack."""
    eg = EGraph()
    x, w, a = _v("x", 4, 4), _v("W", 4, 4), _v("a", 4, 4)
    terms = [
        Op.make("add", Op.make("mul", x, w), a),
        Op.make("add", Op.make("mul", a, w), x),
        Op.make("add", Op.make("mul", x, x), w),
    ]
    root = eg.add_term(Op.make("pack", *terms))
    return eg, root, (x, w, a)


def _chunk_graph():
    """add(chunk(x, index=i), W) for i in 0..2 — attr-differing motif."""
    eg = EGraph()
    x, w = _v("x", 8, 4), _v("W", 4, 4)
    terms = [
        Op.make(
            "add",
            Op.make("chunk", x, chunks=4, dim=0, index=i),
            w,
        )
        for i in range(3)
    ]
    root = eg.add_term(Op.make("pack", *terms))
    return eg, root, (x, w)


class TestMining:
    """The anti-unification finds the shared structure."""

    def test_repeated_motif_mines(self):
        eg, _root, _leaves = _motif_graph()
        cands = mine_candidates(eg, min_ops=2, min_uses=2, max_ops=4)
        assert cands
        top = cands[0]
        assert top.pattern.op == "add"
        assert top.pattern.args[0].op == "mul"
        assert top.n_instances == 3
        assert len(top.leaf_mvars) == 3
        assert top.gain == 3 * (2 - 1) - 2

    def test_attr_differences_become_metavars(self):
        eg, _root, _leaves = _chunk_graph()
        cands = mine_candidates(eg, min_ops=2, min_uses=2, max_ops=3)
        top = cands[0]
        assert top.attr_mvars, "the differing index attr must generalise"
        assert len(top.attr_mvars) == 1
        # The pattern instantiates back at each site.
        for i in range(3):
            inst = match_pattern(
                top.pattern,
                Op.make(
                    "add",
                    Op.make(
                        "chunk",
                        _v("x", 8, 4),
                        chunks=4,
                        dim=0,
                        index=i,
                    ),
                    _v("W", 4, 4),
                ),
                {},
            )
            assert inst is not None

    def test_structure_only_mode_splits_on_attrs(self):
        eg, _root, _leaves = _chunk_graph()
        cands = mine_candidates(
            eg, min_ops=2, min_uses=2, max_ops=3, attr_mvars=False
        )
        assert cands == []

    def test_shared_columns_share_metavars(self):
        """x @ x + x sites: children equal => one metavar for both."""
        eg = EGraph()
        x, a = _v("x", 4, 4), _v("a", 4, 4)
        t1 = Op.make("add", Op.make("mul", x, x), a)
        t2 = Op.make("add", Op.make("mul", a, a), x)
        root = eg.add_term(Op.make("pack", t1, t2))
        cands = mine_candidates(eg, min_ops=2, min_uses=2, max_ops=3)
        sq = [
            c
            for c in cands
            if c.pattern.args[0].op == "mul"
            and c.pattern.args[0].args[0] == c.pattern.args[0].args[1]
        ]
        assert sq, "mul(v, v) must surface as a shared-metavar pattern"
        # And the matcher honours the equality constraint.
        for c in sq:
            hits = [
                s
                for cid in eg._classes
                for s in eg.matches(c.pattern, cid)
            ]
            assert hits

    def test_empty_graph_yields_nothing(self):
        eg = EGraph()
        assert mine_candidates(eg) == []

    def test_skip_ops_excludes(self):
        eg, _root, _leaves = _motif_graph()
        cands = mine_candidates(
            eg, min_ops=1, min_uses=2, max_ops=2, skip_ops=("add",)
        )
        assert all(
            "add" not in {n for n in _ops(c.pattern)} for c in cands
        )

    def test_site_cap_bounds_instances(self):
        eg, _root, _leaves = _motif_graph()
        cands = mine_candidates(
            eg, min_ops=2, min_uses=2, max_ops=3, site_cap=1
        )
        # cap=1 bounds each class's contribution to a single match.
        assert all(c.n_instances == len(c.sites) for c in cands)


def _ops(t):
    if isinstance(t, Op):
        yield t.op
        for a in t.args:
            yield from _ops(a)


def _ops_args(t):
    if isinstance(t, Op):
        yield t
        for a in t.args:
            yield from _ops_args(a)


class TestMinting:
    """The atom is a registered op + a definitional pair + provenance."""

    def test_mint_registers_op_and_rewrites(self):
        eg, _root, _leaves = _motif_graph()
        cand = mine_candidates(eg, min_ops=2, min_uses=2)[0]
        atom = mint_atom(cand, name="test_at0")
        assert generator("test_at0") is not None
        assert atom.arity == len(cand.leaf_mvars)
        # intro: expansion -> call; expand: call -> expansion.
        assert atom.intro.lhs is cand.pattern
        assert atom.expand.rhs is cand.pattern
        assert atom.intro.rhs is atom.call
        assert DEFINITIONAL in atom.intro.tags
        assert atom.object.kind == "abstraction"
        assert atom.object.construction == ("antiunify", "test_at0")
        assert atom.sites == cand.sites

    def test_min_gain_filters_trivial_patterns(self):
        eg = EGraph()
        x = _v("x", 4, 4)
        # One occurrence only of anything compound — nothing pays.
        eg.add_term(Op.make("add", Op.make("mul", x, x), x))
        assert mint_atoms(eg, min_uses=2) == []

    def test_names_advance_past_registered_ops(self):
        eg, _root, _leaves = _motif_graph()
        first = mint_atoms(eg, count=1, prefix="dup")
        second = mint_atoms(eg, count=1, prefix="dup")
        assert first and second
        assert first[0].name != second[0].name
        # Iterated minting can find atom-over-atom candidates.
        assert generator(second[0].name) is not None


class TestAdmission:
    """intro adds the compressed member; nothing is ever dropped."""

    def test_admit_adds_atom_members(self):
        eg, root, leaves = _motif_graph()
        x, w, a = leaves
        atoms = mint_atoms(eg, count=1)
        (atom,) = atoms
        before = eg.n_enodes
        fires = admit_atoms(eg, root, atoms)
        assert fires[atom.name] == 3
        assert eg.n_enodes == before + 3
        # Every minted site class holds an atom member now.
        atom_classes = {
            eg.find(cid)
            for cid in atom.sites
            if any(n.op == atom.name for n in eg.get_class(cid).nodes)
        }
        assert len(atom_classes) == 3
        # The spelled members are still there — refuse never drop.
        site = atom.sites[0]
        ops = {n.op for n in eg.get_class(site).nodes}
        assert "add" in ops and atom.name in ops

    def test_atom_member_denotes_the_expansion(self):
        eg, root, leaves = _motif_graph()
        atoms = mint_atoms(eg, count=1)
        atom = atoms[0]
        admit_atoms(eg, root, atoms)
        # The expand rule, applied at a site, re-materialises the body
        # — the definitional inverse replays in the e-graph.
        site = atom.sites[0]
        cid = eg.find(site)
        spelled_before = {
            n for n in eg.get_class(cid).nodes if n.op == "add"
        }
        eg.apply_rule(atom.expand, root)
        spelled_after = {
            n for n in eg.get_class(cid).nodes if n.op == "add"
        }
        assert spelled_after == spelled_before  # already present
        # An atom member instantiates back to a real expansion.
        assert any(
            n.op == atom.name for n in eg.get_class(cid).nodes
        )
        term = eg.any_term(cid)
        assert term is not None
        un = unfold_term(term, atoms)
        assert "defn" not in op_repr(un)


class TestFolding:
    """Folded rules mint atoms where parents spelled bodies."""

    def _atom(self):
        eg, root, _leaves = _chunk_graph()
        atoms = mint_atoms(eg, count=1)
        return eg, root, atoms[0], atoms

    def test_rhs_fold_mints_variant(self):
        _eg, _root, atom, _atoms = self._atom()
        lhs = Op.make(
            "mul",
            Op.make("chunk", "P", chunks=4, dim=0, index="I"),
            "Q",
        )
        r = Rewrite(
            "r_test",
            lhs=lhs,
            rhs=Op.make(
                "add",
                Op.make("chunk", "P", chunks=4, dim=0, index="I"),
                "Q",
            ),
            law="test",
        )
        vs = fold_rule(r, atom, side="rhs")
        assert len(vs) == 1
        assert vs[0].name == f"r_test~{atom.name}.r0"
        assert vs[0].rhs.op == atom.name
        assert DEFINITIONAL in vs[0].tags
        assert vs[0].derivation == ("r_test", atom.intro.name)
        # And the folded variant unfolds to the parent's rhs.
        assert op_repr(unfold_term(vs[0].rhs, [atom])) == op_repr(
            r.rhs
        )

    def test_rhs_fold_declines_free_metavars(self):
        _eg, _root, atom, _atoms = self._atom()
        r = Rewrite(
            "r_free",
            lhs=Op.make("mul", "P", "Q"),
            rhs=Op.make(
                "add",
                Op.make("chunk", "FREE", chunks=4, dim=0, index="I"),
                "W",
            ),
        )
        assert fold_rule(r, atom, side="rhs") == []

    def test_lhs_fold_mints_variant(self):
        _eg, _root, atom, _atoms = self._atom()
        lhs = Op.make(
            "add",
            Op.make("chunk", "P", chunks=4, dim=0, index="I"),
            "Q",
        )
        r = Rewrite("r_l", lhs=lhs, rhs=Op.make("mul", "P", "Q"))
        vs = fold_rule(r, atom, side="lhs")
        assert len(vs) == 1
        assert vs[0].lhs.op == atom.name

    def test_fold_ruleset_replace_drops_parent(self):
        _eg, _root, atom, atoms = self._atom()
        lhs = Op.make(
            "mul",
            Op.make("chunk", "P", chunks=4, dim=0, index="I"),
            "Q",
        )
        r = Rewrite(
            "r_test",
            lhs=lhs,
            rhs=Op.make(
                "add",
                Op.make("chunk", "P", chunks=4, dim=0, index="I"),
                "Q",
            ),
        )
        keep = Rewrite("r_keep", lhs="A", rhs="A")
        strict = fold_ruleset([r, keep], atoms)
        assert {x.name for x in strict} == {
            "r_keep",
            f"r_test~{atom.name}.r0",
        }
        mixed = fold_ruleset([r, keep], atoms, replace=False)
        assert {x.name for x in mixed} == {
            "r_test",
            "r_keep",
            f"r_test~{atom.name}.r0",
        }

    def test_fold_ruleset_lhs_side(self):
        _eg, _root, atom, atoms = self._atom()
        lhs = Op.make(
            "add",
            Op.make("chunk", "P", chunks=4, dim=0, index="I"),
            "Q",
        )
        r = Rewrite("r_l", lhs=lhs, rhs=Op.make("mul", "P", "Q"))
        vs = fold_ruleset([r], atoms, sides=("lhs",))
        assert any(x.name == f"r_l~{atom.name}.l0" for x in vs)

    def test_folded_rule_fires_and_stays_sound(self):
        eg, root, atom, atoms = self._atom()
        lhs = Op.make(
            "mul",
            Op.make("chunk", "P", chunks=4, dim=0, index="I"),
            "Q",
        )
        folded = fold_rule(
            Rewrite(
                "r_go",
                lhs=lhs,
                rhs=Op.make(
                    "add",
                    Op.make("chunk", "P", chunks=4, dim=0, index="I"),
                    "Q",
                ),
            ),
            atom,
            side="rhs",
        )[0]
        # A fresh mul site where the body is not yet spelled: the
        # folded rule mints just the atom member.
        x, w = _v("x", 8, 4), _v("W", 4, 4)
        site = eg.add_term(Op.make("mul", Op.make("chunk", x, chunks=4, dim=0, index=3), w))
        assert eg.apply_rule(folded, root)
        ops = {n.op for n in eg.get_class(site).nodes}
        assert atom.name in ops
        assert "add" not in ops  # the body stayed unspelled
        # Unfolding the atom member gives the spelled body back —
        # the atom's first arg is the chunk's OPERAND, not the chunk.
        ch = Op.make("chunk", x, chunks=4, dim=0, index=3)
        un = unfold_term(
            Op.make(atom.name, x, w, **{atom.attr_mvars[0]: 3}),
            atoms,
        )
        assert un == Op.make("add", ch, w)

    def test_bad_side_raises(self):
        _eg, _root, atom, _atoms = self._atom()
        with pytest.raises(ValueError, match="side"):
            fold_rule(Rewrite("r", lhs="A", rhs="A"), atom, side="mid")

    def test_wrapped_hooks_see_spelled_terms(self):
        """check/derive hooks get unfolded (library-op) bound terms."""
        eg, root, atom, atoms = self._atom()
        admit_atoms(eg, root, atoms)
        seen: list[Any] = []

        def check(bound):
            seen.append(bound["P"])
            return True

        lhs = Op.make(
            "mul",
            Op.make("chunk", "P", chunks=4, dim=0, index="I"),
            "Q",
        )
        hooked = Rewrite(
            "r_hook",
            lhs=lhs,
            rhs=Op.make(
                "add",
                Op.make("chunk", "P", chunks=4, dim=0, index="I"),
                "Q",
            ),
            check=check,
            derive=None,
        )
        folded = fold_ruleset([hooked], atoms)
        # A site whose class holds a defn member: any_term may hand
        # the hook a defn-headed term — the wrapper unfolds it first.
        x, w = _v("x", 8, 4), _v("W", 4, 4)
        eg.add_term(
            Op.make(
                "mul",
                Op.make("chunk", x, chunks=4, dim=0, index=1),
                w,
            )
        )
        eg.apply_rule(folded[-1], root)
        assert seen, "the wrapped check must have run"
        from catopt_core.ir import op_repr as _repr

        assert all("defn_" not in _repr(t) for t in seen)


class TestElaboration:
    """unfold restores the pure-library spelling."""

    def test_unfold_non_op_passthrough(self):
        v = _v("x", 2, 2)
        assert unfold_term(v, {}) is v

    def test_unfold_nested_atoms(self):
        eg, root, _leaves = _motif_graph()
        atoms = mint_atoms(eg, count=2)
        admit_atoms(eg, root, atoms)
        best = eg.extract_best(
            eg.find(root), unfolding_cost(flops_cost, atoms)
        )
        un = unfold_term(best, atoms)
        assert "defn_" not in op_repr(un)
        assert dag_cost(un, flops_cost) == pytest.approx(
            dag_cost(best, unfolding_cost(flops_cost, atoms))
        )

    def test_unfolding_cost_prices_semantics(self):
        eg, root, _leaves = _motif_graph()
        atoms = mint_atoms(eg, count=1)
        admit_atoms(eg, root, atoms)
        cfn = unfolding_cost(flops_cost, atoms)
        site = atoms[0].sites[0]
        spelled = eg.any_term(eg.find(site))
        # The wrapper prices the expansion, never the bare name.
        assert cfn(spelled) == pytest.approx(dag_cost(spelled, cfn))
        # Markers propagate like backend_cost's wrapper.
        assert getattr(cfn, "charges_param_only", None) == getattr(
            flops_cost, "charges_param_only", None
        )

    def test_atom_map_idempotent(self):
        eg, _root, _leaves = _motif_graph()
        atoms = mint_atoms(eg, count=1)
        assert atom_map(atom_map(atoms)) == atom_map(atoms)

    def test_unfolding_cost_accepts_memoless_fns(self):
        eg, _root, _leaves = _motif_graph()
        atoms = mint_atoms(eg, count=1)
        cfn = unfolding_cost(lambda t: 7.0, atoms)
        assert cfn(_v("x", 4, 4)) == 7.0

    def test_helper_leaf_edges(self):
        """Literal (non-metavar) leaves pass through the walk helpers."""
        from catopt_core.ir import Const

        from catopt_discovery.antiunify import (
            _alpha_norm,
            _attr_mvars_of,
            _leaf_mvars,
            _metavar_count,
            _swap_hole,
        )

        c = Const(3.0)
        assert _leaf_mvars(c) == ()
        assert _attr_mvars_of(c) == ()
        assert _metavar_count(c, "v0") == 0
        assert _swap_hole(c, "v0", Op.make("neg", "v0")) is c
        # A metavar-free term alpha-normalises to itself.
        t = Op.make("add", c, c)
        assert _alpha_norm(t) == _alpha_norm(t)
        # Non-Op unfold passthrough.
        assert unfold_term(c, {}) is c

    def test_min_ops_gated_early_return(self):
        eg, _root, _leaves = _motif_graph()
        # max_ops=1: every seed already sits at the cap — no growth.
        cands = mine_candidates(eg, min_ops=1, max_ops=1)
        assert cands and all(c.op_count == 1 for c in cands)

    def test_beam_trim(self):
        # The unrolled delta block offers many seeds; beam=2 forces
        # the worklist trim while candidates still grow.
        eg, _root, _leaves = _chunk_graph()
        x, w = _v("x", 8, 4), _v("W", 4, 4)
        for i in range(3):
            eg.add_term(
                Op.make(
                    "add",
                    Op.make("chunk", x, chunks=4, dim=0, index=i),
                    Op.make("neg", w),
                )
            )
        # add(chunk, neg) motif seeds add/neg/chunk groups; pushes
        # outlive the beam.
        assert mine_candidates(eg, beam=1, max_ops=3)

    def test_min_gain_skips_nonpaying(self):
        eg, _root, _leaves = _motif_graph()
        assert mint_atoms(eg, count=1, min_gain=99) == []

    def test_shared_attr_columns_share_metavars(self):
        """Two attrs equal-within-row but differing-across-rows share
        one attr metavar — the ``aenv`` reuse arc."""
        eg = EGraph()
        x, w = _v("x", 4, 4), _v("W", 4, 4)
        terms = [
            Op.make(
                "add",
                Op.make("mul", x, w, a=i, b=i),
                w,
            )
            for i in range(2)
        ]
        root = eg.add_term(Op.make("pack", *terms))
        cands = mine_candidates(eg, min_ops=1, min_uses=2, max_ops=2)
        shared = [
            c
            for c in cands
            if c.attr_mvars
            and any(
                isinstance(sub, Op)
                and len(c.attr_mvars) < len(sub.attrs)
                for sub in _ops_args(c.pattern)
            )
        ]
        assert shared, "a==b columns must share one attr metavar"

    def test_unfolding_cost_uninspectable_fn(self):
        from catopt_discovery.antiunify import unfolding_cost as uc

        cfn = uc(id, [])  # builtins raise in inspect.signature
        assert cfn(_v("x", 4, 4)) > 0
        assert cfn.__name__ == "id"

    def test_unfolding_cost_markers(self):
        def cm(t):
            return 1.0

        cm.charges_param_only = True
        cm.profile = ("pf", "bw", "ls")
        cfn = unfolding_cost(cm, [])
        assert cfn.charges_param_only and cfn.profile == ("pf", "bw", "ls")

    def test_unfolding_cost_nameless_callable(self):
        class _C:
            def __call__(self, t):
                return 1.0

        cfn = unfolding_cost(_C(), [])
        assert cfn.__name__ == "unfolding_cost"
        assert cfn(_v("x", 4, 4)) == 1.0

    def test_unfolding_cost_uninspectable_signature(self):
        class _C:
            @property
            def __signature__(self):
                raise ValueError("uninspectable")

            def __call__(self, t):
                return 1.0

        cfn = unfolding_cost(_C(), [])
        assert cfn(_v("x", 4, 4)) == 1.0

    def test_priced_prices_defn_members(self):
        """``priced`` on an atom term = body ops + per-position kids."""
        eg, _root, _leaves = _chunk_graph()
        atoms = mint_atoms(eg, count=1)
        (atom,) = atoms
        x, w = _v("x", 8, 4), _v("W", 4, 4)
        cfn = unfolding_cost(flops_cost, atoms)
        member = Op.make(
            atom.name, x, w, **{atom.attr_mvars[0]: 0}
        )
        body = unfold_term(member, atoms)
        # The member price is the body's own ops plus the children's
        # priced cost — ≥ the unfolded spelling's total.
        assert cfn(member) == pytest.approx(
            _expansion_body_cost(
                atom,
                member,
                lambda t: flops_cost(t),
            )
            + cfn(x)
            + cfn(w)
        )
        assert cfn(member) == pytest.approx(
            flops_cost(body)
        )

    def test_pick_skips_mismatched_keysets(self):
        """A class with same-(op,arity) members of different key-sets
        exercises the ``_pick`` miss-continue arc."""
        from catopt_discovery.antiunify import _member_index, _pick

        eg = EGraph()
        x, w = _v("x", 4, 4), _v("W", 4, 4)
        c1 = eg.add_term(Op.make("mul", x, w, a=1, b=1))
        c2 = eg.add_term(Op.make("mul", x, w))
        eg.union(c1, c2)
        cid = eg.find(c1)
        # Sorted member order puts the attr-carrying mul first: the
        # pick must skip it (key-set ('a','b') ≠ ()) before matching.
        midx = _member_index(eg)
        n = _pick(midx, cid, ("mul", 2, ()), True)
        assert n is not None and not n.attrs


class TestStorage:
    """The declared-object record carries the atom's provenance."""

    def test_store_atoms_round_trips(self, tmp_path):
        eg, _root, _leaves = _motif_graph()
        atoms = mint_atoms(eg, count=1)
        from catopt_discovery import evidence as ev

        conn = ev.connect(str(tmp_path / "s.db"))
        keys = store_atoms(conn, atoms, "corpus-x")
        assert len(keys) == 1
        row = conn.execute(
            "select name, law_json from lemmas where alpha_key = ?",
            (keys[0],),
        ).fetchone()
        assert row[0] == atoms[0].intro.name
        import json

        rec = json.loads(row[1])
        assert rec["kind"] == "abstraction"
        prov = rec["provenance"]
        assert prov["op"] == "antiunify"
        assert prov["atom"] == atoms[0].name
        assert prov["instances"] == atoms[0].n_instances
        # The embedded expand rule reconstructs.
        from catopt_core.laws.serialize import law_from_data

        expand = law_from_data(prov["expansion"])
        assert expand.rhs is not None


class TestEndToEnd:
    """A minted atom survives extract → unfold → lower → verify fp64."""

    def test_extract_unfold_lowers_and_verifies(self):
        torch = pytest.importorskip("torch")
        from catopt_core.ir import IR
        from catopt_torch.adapters import TorchSink, TorchSource

        class Block(torch.nn.Module):
            """add(chunk(x, i), w) three times — the mined motif."""

            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)

            def forward(self, x):
                y = self.lin(x)
                return (
                    torch.add(
                        torch.chunk(y, 4, dim=0)[0],
                        torch.chunk(y, 4, dim=0)[1],
                    )
                    + torch.chunk(y, 4, dim=0)[2]
                    + torch.chunk(y, 4, dim=0)[3]
                )

        torch.manual_seed(0)
        model = Block().double()
        x = torch.randn(8, 4, dtype=torch.float64)
        ir, leaves = TorchSource().to_ir(model, x)
        eg = EGraph()
        root = eg.add_term(ir.root)
        atoms = mint_atoms(eg, count=2, min_uses=2)
        assert atoms, "the repeated chunk-add motif must mint an atom"
        admit_atoms(eg, root, atoms)
        best = eg.extract_best(
            eg.find(root), unfolding_cost(flops_cost, atoms)
        )
        un = unfold_term(best, atoms)
        ops = set()
        stack = [un]
        while stack:
            t = stack.pop()
            if isinstance(t, Op):
                ops.add(t.op)
                stack.extend(t.args)
        sink = TorchSink()
        assert not (ops - sink.supported_ops)
        ex_ir = IR(
            root=un,
            inputs=list(ir.inputs),
            input_names=set(ir.input_names),
            params=dict(ir.params),
        )
        module = sink.lower(ex_ir, params=leaves)
        rep = sink.verify(model, module, (x,))
        assert rep.passed and rep.max_abs < 1e-9

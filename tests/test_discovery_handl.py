"""The handl-fragment schema bridge — cells as language data."""


from catopt_core.cells import (
    Fragment,
    Provenance,
    handle_cell,
    law_cell,
    merge_fragments,
    object_cell,
    op_cell,
)
from catopt_core.ir import Op
from catopt_core.laws.base import R
from catopt_core.opdata import OpDef
from catopt_discovery import handl
from catopt_discovery.cellstore import ambient_store

OM_LIFT = R(
    "om_decayed_lift",
    Op.make("matmul", Op.make("mul", "s", "d"), "v"),
    Op.make("om_decayed", "s", "d", "v"),
)


def _store():
    return merge_fragments(
        Fragment(
            "demo",
            (
                law_cell(OM_LIFT),
                op_cell(OpDef(name="om_decayed", arity=3)),
                handle_cell(
                    "decayed_0",
                    {
                        "pattern": ("matmul", ("mul", "S", "D"), "V"),
                        "kernel": "decayed_0_k",
                        "args": ("S", "D", "V"),
                    },
                    provenance=Provenance("minted"),
                ),
            ),
        )
    )


class TestLawToHandl:
    def test_root_kind_and_when(self):
        row = handl.law_to_handl(OM_LIFT, "om_decayed_lift")
        assert row["node_kind"] == "matmul"
        # handl's `when` is ONE TransformCondition; a conjunction folds
        # into And([...]) — a bare list does not deserialize.
        clauses = row["when"]["And"]
        kinds = [c.get("KindAt") for c in clauses]
        assert {"index": 0, "kind": "mul"} in kinds
        assert {"ChildCount": 2} in clauses

    def test_metavar_paths_rebuild_children(self):
        row = handl.law_to_handl(OM_LIFT, "om_decayed_lift")
        rule = row["rule"]["Declarative"]
        assert rule["target_kind"] == "om_decayed"
        assert rule["children"] == [
            {"KeepPath": [0, 0]},
            {"KeepPath": [0, 1]},
            {"KeepPath": [1]},
        ]

    def test_metavar_paths_in_extension(self):
        row = handl.law_to_handl(OM_LIFT, "om_decayed_lift")
        assert row["catopt"]["metavars"] == {
            "s": [0, 0],
            "d": [0, 1],
            "v": [1],
        }

    def test_unbound_metavar_uses_leaf_tag(self):
        r = R(
            "x",
            Op.make("mul", "a", "b"),
            Op.make("mul", "a", "c"),  # 'c' unbound — not in lhs
        )
        row = handl.law_to_handl(r, "x")
        last = row["rule"]["Declarative"]["children"][-1]
        assert last == {"Fresh": "c"}

    def test_rhs_op_children_create(self):
        r = R(
            "x",
            Op.make("mul", "a", "b"),
            Op.make("mul", Op.make("add", "a", "b"), "b"),
        )
        row = handl.law_to_handl(r, "x")
        first = row["rule"]["Declarative"]["children"][0]
        assert first == {
            "Create": {
                "kind": "add",
                "children": [
                    {"KeepPath": [0]},
                    {"KeepPath": [1]},
                ],
            }
        }

    def test_rhs_const_and_var_leaves(self):
        from catopt_core.ir import Const, TensorType, Var

        r = R(
            "x",
            Op.make("mul", "a", "b"),
            Op.make(
                "mul",
                Const(2.0),
                Var("w", TensorType((4, 4))),
            ),
        )
        row = handl.law_to_handl(r, "x")
        kids = row["rule"]["Declarative"]["children"]
        assert kids[0] == {"Create": {"kind": "catopt:const", "payload": {"value": 2.0}, "children": []}}
        assert kids[1] == {"Create": {"kind": "catopt:var", "payload": {"name": "w"}, "children": []}}

    def test_when_condition_folds_for_handl(self):
        """0 clauses omit the field; 1 is bare; 2+ become And([...])."""
        assert handl._when_condition("a") is None
        one = handl._when_condition(Op.make("eye", n=4))
        assert one == {"ChildCount": 0}
        many = handl._when_condition(OM_LIFT.lhs)
        assert set(many) == {"And"} and len(many["And"]) >= 2

    def test_bare_metavar_root(self):
        r = R("x", "a", "b")
        row = handl.law_to_handl(r, "x")
        assert row["node_kind"] is None and "when" not in row
        kids = row["rule"]["Declarative"]["children"]
        assert kids == [{"Fresh": "b"}]
        assert row["rule"]["Declarative"]["target_kind"] is None

    def test_zero_arity_root(self):
        r = R("x", Op.make("eye", n=4), Op.make("eye", n=4))
        row = handl.law_to_handl(r, "x")
        assert row["node_kind"] == "eye"
        assert row["when"] == {"ChildCount": 0}

    def test_op_parts_accepts_spec_tuples(self):
        assert handl._op_parts(("matmul", "s", "v")) == (
            "matmul",
            ("s", "v"),
        )
        assert handl._op_parts(("k", {"a": 1}, "s")) == ("k", ("s",))
        assert handl._op_parts("a") is None
        assert handl._op_parts(()) is None

    def test_var_leaf_in_lhs(self):
        from catopt_core.ir import TensorType, Var

        r = R(
            "x",
            Op.make("mul", Var("w", TensorType((4, 4))), "b"),
            Op.make("mul", "b", Var("w", TensorType((4, 4)))),
        )
        row = handl.law_to_handl(r, "x")
        kids = row["rule"]["Declarative"]["children"]
        assert kids[1] == {"Create": {"kind": "catopt:var", "payload": {"name": "w"}, "children": []}}

    def test_law_data_roundtrips(self):
        row = handl.law_to_handl(OM_LIFT, "om_decayed_lift")
        cell = handl.cell_from_handl(row)
        assert cell is not None
        assert cell.role == "law" and cell.name == "om_decayed_lift"
        assert cell.provenance.origin == "imported"
        assert cell.body.lhs == OM_LIFT.lhs


class TestOpDefToHandl:
    def test_arity_children_roles(self):
        row = handl.opdef_to_handl(OpDef(name="om_decayed", arity=3))
        assert row["node_kind"] == "om_decayed"
        assert row["children"] == [
            {"role": "arg0"},
            {"role": "arg1"},
            {"role": "arg2"},
        ]

    def test_undeclared_arity(self):
        row = handl.opdef_to_handl(OpDef(name="x"))
        assert row["children"] == []

    def test_opdef_roundtrip(self):
        row = handl.opdef_to_handl(OpDef(name="om_decayed", arity=3))
        cell = handl.cell_from_handl(row)
        assert cell.role == "op" and cell.name == "om_decayed"
        assert cell.body.arity == 3


class TestHandleToHandl:
    def test_native_tag_and_spec(self):
        row = handl.handle_to_handl(
            "decayed_0",
            {
                "pattern": ("matmul", ("mul", "S", "D"), "V"),
                "kernel": "decayed_0_k",
                "args": ("S", "D", "V"),
            },
        )
        assert row["native"] == "catopt:decayed_0_k"
        assert row["args"] == ["S", "D", "V"]

    def test_handler_roundtrip(self):
        row = handl.handle_to_handl(
            "decayed_0",
            {"pattern": ("om",), "kernel": "k0", "args": ()},
        )
        cell = handl.cell_from_handl(row)
        assert cell.role == "handle" and cell.body["kernel"] == "k0"


class TestStoreToHandl:
    def test_fragment_shape(self):
        frag = handl.store_to_handl(_store())
        assert len(frag["transforms"]) == 1
        # `ast` = the store's op cells + the minted leaf kinds the bridge
        # declares (catopt:const / catopt:var / catopt:data).
        assert len(frag["ast"]) == 1 + len(handl._LEAF_SHAPES)
        assert len(frag["natives"]) == 1
        assert frag["catopt"]["cells"] == 3
        assert set(frag["catopt"]["provenance"]) <= {
            "declared",
            "minted",
            "shipped",
        }

    def test_object_cell_maps_to_transform(self):
        store = merge_fragments(
            Fragment(
                "demo",
                (
                    object_cell(
                        OM_LIFT,
                        kind="abstraction",
                        provenance=Provenance("constructed"),
                    ),
                ),
            )
        )
        frag = handl.store_to_handl(store)
        assert frag["transforms"][0]["catopt"]["object"] is not None

    def test_cell_from_handl_declines_pure_rows(self):
        assert handl.cell_from_handl({"id": "x", "kind": "y"}) is None


class TestShippedBankRoundTrip:
    """The seam's acceptance test: the *whole* shipped bank, both ways.

    Per-cell round-trips are covered above; this is the bank-level
    identity ADR 0011 names as the seam test — every emitted row
    reconstructs to a cell, no cell is silently dropped, and re-emitting
    the reconstructed cell reproduces the row byte-for-byte (row
    idempotence), so the two representations cannot drift silently.
    """

    def test_no_cell_is_silently_dropped(self):
        store = ambient_store()
        frag = handl.store_to_handl(store)
        # `ast` also carries the minted leaf kinds (catopt:const/var/data),
        # which are bridge vocabulary, not store cells.
        op_rows = [r for r in frag["ast"] if not r["node_kind"].startswith("catopt:")]
        emitted = len(op_rows) + len(frag["transforms"]) + len(frag["natives"])
        assert emitted == len(list(store)), (
            f"store has {len(list(store))} cells, fragments emit {emitted} rows"
        )

    def test_every_row_round_trips_idempotently(self):
        store = ambient_store()
        frag = handl.store_to_handl(store)
        leaf_kinds = {s["node_kind"] for s in handl._LEAF_SHAPES}
        rows = frag["ast"] + frag["transforms"] + frag["natives"]
        assert rows, "the shipped bank must emit rows"
        # `natives` rows carry no `node_kind` (they are binding records).
        cells = [r for r in rows if r.get("node_kind") not in leaf_kinds]
        leaves = [r for r in rows if r.get("node_kind") in leaf_kinds]
        assert len(leaves) == len(handl._LEAF_SHAPES), "leaf shapes are minted"
        for row in cells:
            cell = handl.cell_from_handl(row)
            assert cell is not None, f"row did not reconstruct: {row.get('id')}"
            # `priority` is a store-level disambiguator (handl keys a rule
            # by pass:node_kind:priority), not part of the cell, so the
            # per-cell re-emission does not carry it.
            without_priority = {k: v for k, v in row.items() if k != "priority"}
            assert handl.cell_to_handl(cell) == without_priority, (
                f"row drifted on round-trip: {row.get('id')}"
            )

"""The handl-fragment schema bridge — cells as language data."""

import pytest

from catopt_core.cells import (
    Fragment,
    Provenance,
    handle_cell,
    law_cell,
    merge_fragments,
    object_cell,
    op_cell,
)
from catopt_core.egraph import Rewrite
from catopt_core.ir import Op
from catopt_core.laws.base import R
from catopt_core.opdata import OpDef
from catopt_discovery import handl

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
        kinds = [c.get("KindAt") for c in row["when"]]
        assert {"index": 0, "kind": "mul"} in kinds
        assert {"ChildCount": 2} in row["when"]

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
        assert last == {"catopt:leaf": "c"}

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
        assert kids[0] == {"catopt:leaf": {"const": 2.0}}
        assert kids[1] == {"catopt:leaf": {"var": "w"}}

    def test_bare_metavar_root(self):
        r = R("x", "a", "b")
        row = handl.law_to_handl(r, "x")
        assert row["node_kind"] is None and row["when"] == []
        kids = row["rule"]["Declarative"]["children"]
        assert kids == [{"catopt:leaf": "b"}]
        assert row["rule"]["Declarative"]["target_kind"] is None

    def test_zero_arity_root(self):
        r = R("x", Op.make("eye", n=4), Op.make("eye", n=4))
        row = handl.law_to_handl(r, "x")
        assert row["node_kind"] == "eye"
        assert row["when"] == [{"ChildCount": 0}]

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
        assert kids[1] == {"catopt:leaf": {"var": "w"}}

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
        assert len(frag["ast"]) == 1
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

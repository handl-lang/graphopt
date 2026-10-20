"""The proposal corpus — term specs and schema libraries.

The curated real terms ``proposal.seed_terms`` mines, the
algebraic-grammar and shape-aware schema libraries
(``proposal.schema_candidates`` / ``shape_proposal.schemas``), the
candidate laws ``impact.new_laws`` mints and the synthetic control
corpus ``impact.synthetic_cases`` reads.  Term specs are written in
the declarative language :func:`object_synthesis.term_from_spec`
reads — documented in the package docstring.
"""

from __future__ import annotations

__all__ = [
    "CANDIDATE_LAWS",
    "GRAMMAR_SCHEMAS",
    "SEED_TERMS",
    "SHAPE_SCHEMAS",
    "SYNTHETIC_CASES",
]

# ---------------------------------------------------------------------------
#  The proposal corpus — term specs (the spec language is documented
#  in the package docstring)
# ---------------------------------------------------------------------------

#: The curated real terms ``proposal.seed_terms`` mines: matmul
#: bracketing, elementwise duplication, weight-merge, stacked linear,
#: swiglu, an attention path, and the elementwise-algebra set.
SEED_TERMS: tuple = (
    # matmul bracketing (associativity)
    (
        "matmul",
        ("var", "a", (3, 5)),
        ("matmul", ("var", "b", (5, 2)), ("var", "c", (2, 3))),
    ),
    # elementwise duplication (CSE)
    (
        "add",
        ("square", ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        (
            "mul",
            ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
            ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
        ),
    ),
    # weight-merge / distribute
    (
        "add",
        ("matmul", ("var", "x", (4, 4)), ("param", "W1", (4, 4))),
        ("matmul", ("var", "x", (4, 4)), ("param", "W2", (4, 4))),
    ),
    (
        "matmul",
        ("param", "W", (4, 4)),
        ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
    ),
    # stacked linear
    (
        "linear",
        ("linear", ("var", "x", (4, 4)), ("param", "W1", (4, 4))),
        ("param", "W2", (4, 4)),
    ),
    # swiglu
    (
        "mul",
        (
            "silu",
            ("matmul", ("var", "x", (4, 4)), ("param", "W1", (4, 4))),
        ),
        ("matmul", ("var", "x", (4, 4)), ("param", "W2", (4, 4))),
    ),
    # attention path
    (
        "matmul",
        (
            "softmax",
            (
                "matmul",
                ("var", "q", (2, 8, 4)),
                (
                    "transpose",
                    ("var", "k", (2, 8, 4)),
                    {"dim0": -2, "dim1": -1},
                ),
            ),
            {"dim": -1},
        ),
        ("var", "v", (2, 8, 4)),
    ),
    # elementwise algebra
    (
        "add",
        ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
        ("mul", ("var", "x", (4, 4)), ("var", "z", (4, 4))),
    ),
    (
        "add",
        ("neg", ("var", "x", (4, 4))),
        ("neg", ("var", "y", (4, 4))),
    ),
    (
        "mul",
        ("exp", ("var", "x", (4, 4))),
        ("exp", ("var", "y", (4, 4))),
    ),
    ("sub", ("var", "x", (4, 4)), ("neg", ("var", "y", (4, 4)))),
    ("square", ("neg", ("var", "x", (4, 4)))),
    ("add", ("var", "x", (4, 4)), ("var", "x", (4, 4))),
    ("mul", ("var", "x", (4, 4)), 0),
    ("pow", ("var", "x", (4, 4)), 1),
    ("sub", ("var", "x", (4, 4)), ("var", "x", (4, 4))),
)

#: The algebraic-grammar schema library — ``(label, lhs, rhs, note)``
#: rows ``proposal.schema_candidates`` instantiates on the ``x``/``y``/
#: ``z`` leaves.  The set deliberately mixes *true* identities with
#: *false* ones (so the numeric oracle is exercised) and includes a
#: library duplicate (so the structural classifier is exercised).
GRAMMAR_SCHEMAS: tuple = (
    # -- true: elementwise distributivity / factoring -----------------
    (
        "mul_factor",
        (
            "add",
            ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
            ("mul", ("var", "x", (4, 4)), ("var", "z", (4, 4))),
        ),
        (
            "mul",
            ("var", "x", (4, 4)),
            ("add", ("var", "y", (4, 4)), ("var", "z", (4, 4))),
        ),
        "x*y + x*z = x*(y+z)  (fewer nodes)",
    ),
    (
        "mul_distribute",
        (
            "mul",
            ("var", "x", (4, 4)),
            ("add", ("var", "y", (4, 4)), ("var", "z", (4, 4))),
        ),
        (
            "add",
            ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
            ("mul", ("var", "x", (4, 4)), ("var", "z", (4, 4))),
        ),
        "reverse (expands)",
    ),
    (
        "mul_factor_right",
        (
            "add",
            ("mul", ("var", "y", (4, 4)), ("var", "x", (4, 4))),
            ("mul", ("var", "z", (4, 4)), ("var", "x", (4, 4))),
        ),
        (
            "mul",
            ("add", ("var", "y", (4, 4)), ("var", "z", (4, 4))),
            ("var", "x", (4, 4)),
        ),
        "right-slot variant",
    ),
    (
        "neg_factor",
        (
            "add",
            ("neg", ("var", "x", (4, 4))),
            ("neg", ("var", "y", (4, 4))),
        ),
        ("neg", ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        "-x + -y = -(x+y)",
    ),
    (
        "neg_distribute",
        ("neg", ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        (
            "add",
            ("neg", ("var", "x", (4, 4))),
            ("neg", ("var", "y", (4, 4))),
        ),
        "reverse (expands)",
    ),
    (
        "sub_add_factor",
        (
            "sub",
            ("sub", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
            ("var", "z", (4, 4)),
        ),
        (
            "sub",
            ("var", "x", (4, 4)),
            ("add", ("var", "y", (4, 4)), ("var", "z", (4, 4))),
        ),
        "(x-y)-z = x-(y+z)",
    ),
    (
        "div_add",
        (
            "div",
            ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
            ("var", "z", (4, 4)),
        ),
        (
            "add",
            ("div", ("var", "x", (4, 4)), ("var", "z", (4, 4))),
            ("div", ("var", "y", (4, 4)), ("var", "z", (4, 4))),
        ),
        "(x+y)/z = x/z + y/z",
    ),
    (
        "square_neg",
        ("square", ("neg", ("var", "x", (4, 4)))),
        ("square", ("var", "x", (4, 4))),
        "(-x)^2 = x^2",
    ),
    (
        "exp_factor",
        (
            "mul",
            ("exp", ("var", "x", (4, 4))),
            ("exp", ("var", "y", (4, 4))),
        ),
        ("exp", ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        "e^x e^y = e^(x+y)",
    ),
    (
        "exp_distribute",
        ("exp", ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        (
            "mul",
            ("exp", ("var", "x", (4, 4))),
            ("exp", ("var", "y", (4, 4))),
        ),
        "reverse (expands)",
    ),
    (
        "mul_neg",
        ("mul", ("neg", ("var", "x", (4, 4))), ("var", "y", (4, 4))),
        ("neg", ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        "(-x)y = -(xy)",
    ),
    (
        "square_mul",
        ("square", ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        (
            "mul",
            ("square", ("var", "x", (4, 4))),
            ("square", ("var", "y", (4, 4))),
        ),
        "(xy)^2 = x^2 y^2",
    ),
    (
        "sigmoid_neg",
        ("sigmoid", ("neg", ("var", "x", (4, 4)))),
        ("sub", 1, ("sigmoid", ("var", "x", (4, 4)))),
        "sigma(-x) = 1 - sigma(x)",
    ),
    # -- true: annihilators / identity elements -----------------------
    ("mul_zero", ("mul", ("var", "x", (4, 4)), 0), 0, "x*0 = 0"),
    ("mul_zero_left", ("mul", 0, ("var", "x", (4, 4))), 0, "0*x = 0"),
    (
        "pow_one",
        ("pow", ("var", "x", (4, 4)), 1),
        ("var", "x", (4, 4)),
        "x^1 = x",
    ),
    (
        "sub_self",
        ("sub", ("var", "x", (4, 4)), ("var", "x", (4, 4))),
        0,
        "x - x = 0",
    ),
    (
        "add_inv",
        ("add", ("var", "x", (4, 4)), ("neg", ("var", "x", (4, 4)))),
        0,
        "x + (-x) = 0",
    ),
    (
        "div_self",
        ("div", ("var", "x", (4, 4)), ("var", "x", (4, 4))),
        1,
        "x / x = 1",
    ),
    # -- duplicate of a library law (classifier control) --------------
    (
        "sub_to_add_dup",
        ("sub", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
        ("add", ("var", "x", (4, 4)), ("neg", ("var", "y", (4, 4)))),
        "already in the library",
    ),
    # -- false identities (numeric-oracle controls) -------------------
    (
        "FALSE_mul_factor",
        (
            "add",
            ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
            ("mul", ("var", "x", (4, 4)), ("var", "z", (4, 4))),
        ),
        (
            "mul",
            ("var", "x", (4, 4)),
            ("add", ("var", "x", (4, 4)), ("var", "z", (4, 4))),
        ),
        "false: x*y + x*z != x*(x+z)",
    ),
    (
        "FALSE_exp_add",
        ("exp", ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        (
            "add",
            ("exp", ("var", "x", (4, 4))),
            ("exp", ("var", "y", (4, 4))),
        ),
        "false: e^(x+y) != e^x + e^y",
    ),
    (
        "FALSE_square_add",
        ("square", ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        (
            "add",
            ("square", ("var", "x", (4, 4))),
            ("square", ("var", "y", (4, 4))),
        ),
        "false: (x+y)^2 != x^2 + y^2",
    ),
)

#: The shape-aware schema library — ``(name, lhs, rhs, note, family)``
#: rows ``shape_proposal.schemas`` returns as ``Schema`` records.
#: Metavariable leaves are the bare strings (``A``/``B``/``C``/``X``);
#: attr metavariables ride the trailing dict (``dim="D"`` …).  Some
#: are deliberate library duplicates (the ``*_factor`` controls) so
#: the duplicate detector is exercised.
SHAPE_SCHEMAS: tuple = (
    # -- elementwise factorization (targets add(mul,mul), sub(mul,mul)) --
    (
        "factor_left",
        ("add", ("mul", "A", "B"), ("mul", "A", "C")),
        ("mul", "A", ("add", "B", "C")),
        "x*y + x*z = x*(y+z) (shared left factor).",
        "elementwise-factor",
    ),
    (
        "factor_right",
        ("add", ("mul", "B", "A"), ("mul", "C", "A")),
        ("mul", ("add", "B", "C"), "A"),
        "y*x + z*x = (y+z)*x (shared right factor).",
        "elementwise-factor",
    ),
    (
        "factor_mid",
        ("add", ("mul", "A", "B"), ("mul", "C", "B")),
        ("mul", ("add", "A", "C"), "B"),
        "x*y + z*y = (x+z)*y (shared right factor, y).",
        "elementwise-factor",
    ),
    (
        "factor_sub_left",
        ("sub", ("mul", "A", "B"), ("mul", "A", "C")),
        ("mul", "A", ("sub", "B", "C")),
        "x*y - x*z = x*(y-z).",
        "elementwise-factor",
    ),
    (
        "factor_sub_right",
        ("sub", ("mul", "A", "B"), ("mul", "C", "B")),
        ("mul", ("sub", "A", "C"), "B"),
        "x*y - z*y = (x-z)*y.",
        "elementwise-factor",
    ),
    # -- select / slice naturality (targets mul(select,select)) --
    (
        "select_mul",
        (
            "mul",
            ("select", "A", {"dim": "D", "index": "I"}),
            ("select", "B", {"dim": "D", "index": "I"}),
        ),
        (
            "select",
            ("mul", "A", "B"),
            {"dim": "D", "index": "I"},
        ),
        "mul commutes with select: sel(x)*sel(y)=sel(x*y).",
        "select-naturality",
    ),
    (
        "select_add",
        (
            "add",
            ("select", "A", {"dim": "D", "index": "I"}),
            ("select", "B", {"dim": "D", "index": "I"}),
        ),
        (
            "select",
            ("add", "A", "B"),
            {"dim": "D", "index": "I"},
        ),
        "add commutes with select: sel(x)+sel(y)=sel(x+y).",
        "select-naturality",
    ),
    (
        "select_sub",
        (
            "sub",
            ("select", "A", {"dim": "D", "index": "I"}),
            ("select", "B", {"dim": "D", "index": "I"}),
        ),
        (
            "select",
            ("sub", "A", "B"),
            {"dim": "D", "index": "I"},
        ),
        "sub commutes with select: sel(x)-sel(y)=sel(x-y).",
        "select-naturality",
    ),
    (
        "slice_mul",
        (
            "mul",
            ("slice", "A", {"dim": "D", "start": "S0", "end": "E0"}),
            ("slice", "B", {"dim": "D", "start": "S0", "end": "E0"}),
        ),
        (
            "slice",
            ("mul", "A", "B"),
            {"dim": "D", "start": "S0", "end": "E0"},
        ),
        "mul commutes with slice (equal range).",
        "slice-naturality",
    ),
    # -- layout (targets transpose(reshape), reshape(reshape)) --
    (
        "reshape_reshape",
        ("reshape", ("reshape", "A", {"shape": "S1"}), {"shape": "S2"}),
        ("reshape", "A", {"shape": "S2"}),
        "consecutive reshapes fuse.",
        "layout",
    ),
    (
        "reshape_transpose",
        (
            "transpose",
            ("reshape", "A", {"shape": "S"}),
            {"dim0": "D", "dim1": "I"},
        ),
        (
            "reshape",
            ("transpose", "A", {"dim0": "D", "dim1": "I"}),
            {"shape": "S"},
        ),
        "conjecture — false in general (oracle must reject).",
        "layout",
    ),
    # -- elementwise algebra (targets mul(silu,linear), neg/dist) --
    (
        "neg_add",
        ("add", ("neg", "A"), ("neg", "B")),
        ("neg", ("add", "A", "B")),
        "-x + -y = -(x+y).",
        "elementwise-algebra",
    ),
    (
        "sub_neg",
        ("sub", "A", ("neg", "B")),
        ("add", "A", "B"),
        "x - (-y) = x + y.",
        "elementwise-algebra",
    ),
    (
        "mul_neg_left",
        ("mul", ("neg", "A"), "B"),
        ("neg", ("mul", "A", "B")),
        "(-x)*y = -(x*y).",
        "elementwise-algebra",
    ),
    (
        "exp_add",
        ("mul", ("exp", "A"), ("exp", "B")),
        ("exp", ("add", "A", "B")),
        "e^x * e^y = e^(x+y).",
        "elementwise-algebra",
    ),
    (
        "square_neg",
        ("square", ("neg", "A")),
        ("square", "A"),
        "(-x)^2 = x^2.",
        "elementwise-algebra",
    ),
    # -- linear/matmul factor (library controls; already shipped) --
    (
        "linear_factor",
        ("add", ("linear", "X", "A"), ("linear", "X", "B")),
        ("linear", "X", ("add", "A", "B")),
        "dup of weight_factor_linear (control).",
        "linear-control",
    ),
    (
        "matmul_factor",
        ("add", ("matmul", "X", "A"), ("matmul", "X", "B")),
        ("matmul", "X", ("add", "A", "B")),
        "dup of weight_factor_matmul (control).",
        "linear-control",
    ),
    # -- constants (annihilator / identity; targets are rare) --
    ("mul_zero", ("mul", "A", 0), 0, "x*0 = 0.", "annihilator"),
    (
        "id_mul_lit",
        ("mul", "A", 1),
        "A",
        "x*1 = x (dup of id_mul).",
        "annihilator",
    ),
    ("sub_self", ("sub", "A", "A"), 0, "x - x = 0.", "annihilator"),
)

#: The 11 proposed candidate laws — ``(name, lhs, rhs, law-note)``
#: rows ``impact.new_laws`` mints as ``Rewrite``s (names prefixed
#: ``cand_`` so a run can never shadow a shipped rule).  All are
#: unconditional; the fp caveats are noted in the proposal retro.
CANDIDATE_LAWS: tuple = (
    (
        "cand_mul_factor",
        ("add", ("mul", "x", "y"), ("mul", "x", "z")),
        ("mul", "x", ("add", "y", "z")),
        "x*y + x*z = x*(y+z)  (factoring).",
    ),
    (
        "cand_mul_factor_right",
        ("add", ("mul", "y", "x"), ("mul", "z", "x")),
        ("mul", ("add", "y", "z"), "x"),
        "y*x + z*x = (y+z)*x  (right-slot factoring).",
    ),
    (
        "cand_neg_factor",
        ("add", ("neg", "x"), ("neg", "y")),
        ("neg", ("add", "x", "y")),
        "-x + -y = -(x+y).",
    ),
    (
        "cand_square_neg",
        ("square", ("neg", "x")),
        ("square", "x"),
        "(-x)^2 = x^2.",
    ),
    (
        "cand_exp_factor",
        ("mul", ("exp", "x"), ("exp", "y")),
        ("exp", ("add", "x", "y")),
        "e^x * e^y = e^(x+y).",
    ),
    ("cand_mul_zero", ("mul", "x", 0), 0, "x * 0 = 0."),
    ("cand_mul_zero_left", ("mul", 0, "x"), 0, "0 * x = 0."),
    ("cand_pow_one", ("pow", "x", 1), "x", "x^1 = x."),
    ("cand_sub_self", ("sub", "x", "x"), 0, "x - x = 0."),
    (
        "cand_add_inv",
        ("add", "x", ("neg", "x")),
        0,
        "x + (-x) = 0.",
    ),
    ("cand_div_self", ("div", "x", "x"), 1, "x / x = 1."),
)

#: The synthetic control corpus — ``(name, term spec, var names)`` —
#: each candidate law's own LHS instantiated on small, nonzero fp64
#: leaves (the ``x``/``y``/``z`` specs are ``(4, 4)`` Vars).  The var
#: names order the case's ``inputs``/``feed``.
SYNTHETIC_CASES: tuple = (
    (
        "cand_mul_factor",
        (
            "add",
            ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
            ("mul", ("var", "x", (4, 4)), ("var", "z", (4, 4))),
        ),
        ("x", "y", "z"),
    ),
    (
        "cand_mul_factor_right",
        (
            "add",
            ("mul", ("var", "y", (4, 4)), ("var", "x", (4, 4))),
            ("mul", ("var", "z", (4, 4)), ("var", "x", (4, 4))),
        ),
        ("x", "y", "z"),
    ),
    (
        "cand_neg_factor",
        (
            "add",
            ("neg", ("var", "x", (4, 4))),
            ("neg", ("var", "y", (4, 4))),
        ),
        ("x", "y"),
    ),
    (
        "cand_square_neg",
        ("square", ("neg", ("var", "x", (4, 4)))),
        ("x",),
    ),
    (
        "cand_exp_factor",
        (
            "mul",
            ("exp", ("var", "x", (4, 4))),
            ("exp", ("var", "y", (4, 4))),
        ),
        ("x", "y"),
    ),
    ("cand_mul_zero", ("mul", ("var", "x", (4, 4)), 0), ("x",)),
    ("cand_mul_zero_left", ("mul", 0, ("var", "x", (4, 4))), ("x",)),
    ("cand_pow_one", ("pow", ("var", "x", (4, 4)), 1), ("x",)),
    (
        "cand_sub_self",
        ("sub", ("var", "x", (4, 4)), ("var", "x", (4, 4))),
        ("x",),
    ),
    (
        "cand_add_inv",
        ("add", ("var", "x", (4, 4)), ("neg", ("var", "x", (4, 4)))),
        ("x",),
    ),
    (
        "cand_div_self",
        ("div", ("var", "x", (4, 4)), ("var", "x", (4, 4))),
        ("x",),
    ),
)

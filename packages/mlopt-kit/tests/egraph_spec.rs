//! The egraph leg over the JSON wire — `mlopt_kit::egraph::run_spec`.
//!
//! What is proven here:
//!
//! * the spec interpreter end to end — terms/rules/options decode
//!   through the discovery codec, saturate, and report;
//! * a real closure — identities + involution + commutativity on
//!   `add(mul(a, 1), 0)` — reaching the fixed point and extracting the
//!   folded leaf;
//! * determinism — two runs of the same spec produce identical
//!   reports;
//! * the kernel bridge (`from_kernel`/`to_kernel`) round-tripping a
//!   kernel `Term` through the same `EGraph` — the substrate's
//!   advertised seam, exercised.

use mlopt_kit::egraph::run_spec;
use handl_egraph::{EGraph, Rewrite, RunOptions, StopReason, from_kernel, to_kernel};
use serde_json::{Value, json};

/// `add(mul(a, 1), 0)` under identities + involution + comm —
/// the same fixture the Python differential drives on
/// `mlopt_core.egraph.EGraph`.
fn identities_spec() -> Value {
    json!({
        "name": "identities",
        "terms": [
            {"op": "add", "args": [
                {"op": "mul", "args": [{"leaf": "a"}, {"leaf": "1"}], "attrs": {}},
                {"leaf": "0"}
            ], "attrs": {}}
        ],
        "root": 0,
        "rules": [
            {"name": "id_add", "lhs": {"op": "add", "args": [{"mvar": "a"}, {"leaf": "0"}], "attrs": {}}, "rhs": {"mvar": "a"}},
            {"name": "id_mul", "lhs": {"op": "mul", "args": [{"mvar": "a"}, {"leaf": "1"}], "attrs": {}}, "rhs": {"mvar": "a"}},
            {"name": "double_neg", "lhs": {"op": "neg", "args": [{"op": "neg", "args": [{"mvar": "a"}], "attrs": {}}], "attrs": {}}, "rhs": {"mvar": "a"}},
            {"name": "comm_add", "lhs": {"op": "add", "args": [{"mvar": "a"}, {"mvar": "b"}], "attrs": {}}, "rhs": {"op": "add", "args": [{"mvar": "b"}, {"mvar": "a"}], "attrs": {}}}
        ],
        "options": {"max_iterations": 20, "max_nodes": 10000},
        "costs": {"ops": {"add": 1.0, "mul": 1.0, "neg": 1.0}, "default": 1.0, "leaf": 0.0}
    })
}

#[test]
fn spec_runs_to_fixed_point_and_folds() {
    let report = run_spec(&identities_spec()).unwrap();
    assert_eq!(report["ok"], json!(true));
    assert_eq!(report["stop"], json!("fixed_point"));
    // Everything merges into the root class {a}: `mul(a,1)→a`,
    // `add(a,0)→a`, comm's `add(0,a)` — three classes survive
    // ({root}, {0}, {1}); enodes intern pre-canonical spellings too
    // (rekeyed members stay hash-consed — the spec's `n_enodes`
    // contract), so 8 interned over 3 live classes.
    assert_eq!(report["n_classes"], json!(3));
    assert_eq!(report["n_enodes"], json!(8));
    assert_eq!(report["iterations"], json!(2));
    // The folds recorded witnesses: id_mul + id_add + comm_add's
    // commuted member.
    assert_eq!(
        report["proof_rules"],
        json!(["comm_add", "id_add", "id_mul"])
    );
    // Extraction picks the leaf — the class's minimum member.
    assert_eq!(report["best"], json!({"leaf": "a"}));
    assert_eq!(report["best_cost"], json!(0.0));
    // Three classes: root {a}, {0}, {1}.
    assert_eq!(report["signatures"].as_array().unwrap().len(), 3);
}

#[test]
fn spec_is_deterministic_across_runs() {
    let r1 = run_spec(&identities_spec()).unwrap();
    let r2 = run_spec(&identities_spec()).unwrap();
    assert_eq!(r1, r2);
}

#[test]
fn bounded_saturation_suspends_the_budgeted_rule() {
    let mut spec = identities_spec();
    // A wide summation under comm_add with a small budget.
    spec["terms"] = json!([
        {"op": "add", "args": [
            {"leaf": "a"},
            {"op": "add", "args": [
                {"leaf": "b"},
                {"op": "add", "args": [{"leaf": "c"}, {"leaf": "d"}], "attrs": {}}
            ], "attrs": {}}
        ], "attrs": {}}
    ]);
    spec["options"] = json!({
        "max_iterations": 50,
        "max_nodes": 100000,
        "rule_budgets": {"comm_add": 2}
    });
    let report = run_spec(&spec).unwrap();
    assert_eq!(report["stop"], json!("fixed_point"));
    assert_eq!(report["budget_suspended"], json!(["comm_add"]));
    assert!(report["rule_budgets"]["comm_add"].as_u64().unwrap() >= 2);
}

#[test]
fn check_terms_report_class_membership() {
    let mut spec = identities_spec();
    spec["check_terms"] = json!([
        {"leaf": "a"},
        {"op": "mul", "args": [{"leaf": "a"}, {"leaf": "1"}], "attrs": {}},
        {"op": "not_interned", "args": [{"leaf": "z"}], "attrs": {}}
    ]);
    let report = run_spec(&spec).unwrap();
    let checks = report["checks"].as_array().unwrap();
    // `a` and `mul(a,1)` both live in the root class post-fold.
    assert_eq!(checks[0]["found"], json!(true));
    assert_eq!(checks[0]["in_root"], json!(true));
    assert_eq!(checks[1]["found"], json!(true));
    assert_eq!(checks[1]["in_root"], json!(true));
    // `not_interned(z)` was never interned.
    assert_eq!(checks[2]["found"], json!(false));
    assert_eq!(checks[2]["in_root"], json!(false));
}

/// The kernel bridge: a `hnl_engine::Term` lifted with `from_kernel`,
/// saturated, and lowered with `to_kernel` — `first(inner) -> inner`
/// is a real projection law over the kernel's own term space.
#[test]
fn kernel_terms_saturate_through_the_bridge() {
    use handl_language::term::{ArrSource, Term as KTerm};

    // First{ wrap(dim=1, args=[], source=native) } — `first` is a
    // real kernel op (record projection) over an `Op` payload.
    let kt = KTerm::Op {
        dim: Box::new(KTerm::Const {
            value: json!(1),
        }),
        name: "wrap".to_string(),
        args: vec![],
        source: ArrSource::Native,
        lit: None,
    };
    let proj = KTerm::First {
        inner: Box::new(kt.clone()),
    };

    let lifted_inner = from_kernel(&kt);
    let lifted = from_kernel(&proj);
    let mut g = EGraph::new();
    let root = g.add_term(&lifted, "input").unwrap();
    // first(?i) -> ?i — a bare-metavar RHS is the root-eliminating
    // form (the spec's `id_add` shape) exercised over kernel terms.
    let rule = Rewrite::new(
        "first_proj",
        first_pattern(),
        handl_egraph::Term::var("i"),
    );
    let stats = g
        .run(&[rule], root, &mut RunOptions::default())
        .unwrap();
    assert_eq!(stats.stop, StopReason::FixedPoint);
    let best = g
        .extract_best(root, &mut |t: &handl_egraph::Term| t.size() as f64, None, None)
        .unwrap();
    assert_eq!(best, lifted_inner);
    // …and the lowered form is the *same kernel term* — the bridge
    // round-trips through the saturated graph.
    assert_eq!(to_kernel(&best), kt);
}

/// `first(?i)` as a pattern Term.
fn first_pattern() -> handl_egraph::Term {
    handl_egraph::Term::opl("first", vec![handl_egraph::Term::var("i")])
}

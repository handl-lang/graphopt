//! `mlopt_kit::egraph` — the kit's equality-saturation leg.
//!
//! The runtime seam ([`crate::CatoptKit`]) composes handl's *eval*
//! engines; this module composes handl's *e-graph* substrate —
//! [`handl_egraph::EGraph`], the named Rust port of `mlopt_core`'s
//! `EGraph`/`Rewrite` spec — behind a JSON run-spec interpreter. The
//! wire forms are [`handl_discovery`]'s codec
//! (`term_from_data`/`term_to_data` — `{"mvar"}`/`{"leaf"}`/`{"op"}`),
//! the same scheme `mlopt_core.ir.term_to_data` canonized.
//!
//! The consumer is the differential driver `bench/audit/egraph_differential.py`
//! (and `tests/test_egraph_differential.py`): it hands the same
//! fixture — terms, declarative rules, run options, cost table — to the
//! Python `EGraph` and to `egraph-run`, then compares the reports
//! field by field. The report is therefore built to be *engine-
//! independent*: class ids are internal to each engine, so the
//! structure check compares per-class signatures — a Weisfeiler–Lehman
//! refinement over member enodes rendered with child *labels* — and
//! per-member forms rendered with canonical (tagged) attributes, not
//! raw ids.
//!
//! ## The spec
//!
//! ```json
//! {
//!   "name": "fixture-name",
//!   "terms": [ <term-data>, ... ],        // concrete terms to intern
//!   "root": 0,                          // index into `terms`
//!   "rules": [ {"name": "r", "lhs": <term>, "rhs": <term>} ],
//!   "options": {
//!     "max_iterations": 100, "max_nodes": 100000, "patience": 3,
//!     "stop": "fixed_point" | "improving",
//!     "rule_budgets": {"r": <cap>}
//!   },
//!   "costs": {"ops": {"op": f64}, "default": 1.0, "leaf": 0.0},
//!   "check_terms": [ <term-data>, ... ] // post-run membership probes
//! }
//! ```
//!
//! Attr values in spec terms are plain JSON scalars/arrays; the
//! mlopt tuple/list spellings (`{"__tuple__": …}` / `{"__list__": …}`
//! — `ir.attr_to_data`'s output for composite attrs) are normalized to
//! arrays before decode, matching the spec's list→tuple enode
//! normalisation. `{"amvar": name}` marks an attribute metavariable
//! in pattern position.
//!
//! ## The report
//!
//! `stats` fields mirror the Python `run` payload (`iterations`,
//! `n_enodes`, `n_classes`, `n_proof_edges`, `rule_budgets`,
//! `budget_suspended`, `stop`, `improved`); `signatures` is the sorted
//! WL-label multiset; `classes` lists `{sig, members, rep}` per class
//! (members rendered `[op, attrs, [child sigs]]`, sorted); `best` /
//! `best_cost` are the extraction result; `checks` reports
//! `class_of_term` for each `check_terms` entry. `rep` is the
//! `any_term` representative — reported for reading, not asserted:
//! equal-cost first-member picks can order divergently across the two
//! engines' (documented, different) member-order keys.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;

use handl_discovery::{term_from_data, term_to_data};
use handl_egraph::{
    AttrValue, ClassId, EGraph, ENode, Rewrite, RunOptions, SaturationStats, Stop, StopReason, Term,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// A spec decode / run failure — read as data by the caller.
#[derive(Debug)]
pub struct SpecError(pub String);

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "egraph spec: {}", self.0)
    }
}

impl std::error::Error for SpecError {}

fn bad<T>(msg: impl Into<String>) -> Result<T, SpecError> {
    Err(SpecError(msg.into()))
}

// ---------------------------------------------------------------------------
//  Spec decoding
// ---------------------------------------------------------------------------

/// Rewrite mlopt's composite-attr spellings to plain arrays, in place:
/// `{"__tuple__": […]}` and `{"__list__": […]}` (the sole-key objects
/// `ir.attr_to_data` emits) become their inner array — the form the
/// spec's list→tuple enode normalisation already canonicalizes to.
/// Applied to every term before [`term_from_data`] sees it.
fn untuple(v: &mut Value) {
    match v {
        Value::Array(items) => items.iter_mut().for_each(untuple),
        Value::Object(map) => {
            let lifted = if map.len() == 1 {
                map.get("__tuple__")
                    .or_else(|| map.get("__list__"))
                    .cloned()
            } else {
                None
            };
            if let Some(inner) = lifted {
                *v = inner;
                untuple(v);
            } else {
                map.values_mut().for_each(untuple);
            }
        }
        _ => {}
    }
}

/// Decode a spec term — the discovery codec over the untupled form.
fn decode_term(v: &Value) -> Result<Term, SpecError> {
    let mut v = v.clone();
    untuple(&mut v);
    term_from_data(&v).map_err(|e| SpecError(format!("term decode: {e}")))
}

// ---------------------------------------------------------------------------
//  Cost model — per-op additive pricing, same contract as the Python
//  driver's closure (cost_fn(term) prices the whole subtree).
// ---------------------------------------------------------------------------

/// The spec's `"costs"` table.
struct Costs {
    ops: BTreeMap<String, f64>,
    default: f64,
    leaf: f64,
}

fn cost_of(t: &Term, costs: &Costs) -> f64 {
    match t {
        Term::Var(_) | Term::Leaf(_) => costs.leaf,
        Term::Op(o) => {
            let head = costs
                .ops
                .get(o.op.as_ref())
                .copied()
                .unwrap_or(costs.default);
            head + o.args.iter().map(|a| cost_of(a, costs)).sum::<f64>()
        }
    }
}

// ---------------------------------------------------------------------------
//  Canonical renderings — engine-independent comparison forms
// ---------------------------------------------------------------------------

/// An attr value in the tagged canonical form — spelling-strict, so
/// `2`/`2.0`/`"2"` stay distinct and floats key on *bits* (the Rust
/// `AttrValue` identity contract; `0.0 != -0.0`, distinct NaN payloads
/// distinct). Form: `["z"]` null, `["b", bool]`, `["i", int]` (UInt
/// folds in — mlopt attrs never exceed i64 in spec fixtures),
/// `["f", f64::to_bits]`, `["s", str]`, `["a", […]]`,
/// `["o", [[k, v], …]]`.
fn canon_attr(v: &AttrValue) -> Value {
    match v {
        AttrValue::Null => json!(["z"]),
        AttrValue::Bool(b) => json!(["b", b]),
        AttrValue::Int(n) => json!(["i", n]),
        AttrValue::UInt(n) => json!(["i", n]),
        AttrValue::Float(f) => json!(["f", f.to_bits()]),
        AttrValue::Str(s) => json!(["s", s.as_ref()]),
        AttrValue::Array(a) => {
            json!(["a", a.iter().map(canon_attr).collect::<Vec<_>>()])
        }
        AttrValue::Object(o) => json!([
            "o",
            o.iter()
                .map(|(k, v)| json!([k.as_ref(), canon_attr(v)]))
                .collect::<Vec<_>>()
        ]),
    }
}

/// The canonical marshal — compact JSON, the same spelling the Python
/// side emits (`json.dumps(…, separators=(",", ":"), ensure_ascii=
/// False)`).
fn marshal(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_default()
}

/// `sha256` hex of a marshal — the fixed-size label content. The
/// labels get embedded inside each other at every refinement round;
/// digesting keeps each round's payload flat (and matches Python's
/// `hashlib.sha256(…).hexdigest()` byte-for-byte).
fn label_hash(marshal: &str) -> String {
    let mut h = Sha256::new();
    h.update(marshal.as_bytes());
    format!("{:x}", h.finalize())
}

/// One member enode rendered canonically: `[op, [[k, v]…], [child
/// labels…]]`, compact-serialized. Children carry the *label* of their
/// canonical class — the cross-engine-stable stand-in for the
/// internal class id.
fn member_sig(g: &EGraph, n: &ENode, labels: &BTreeMap<ClassId, String>) -> String {
    let attrs: Vec<Value> = n
        .attrs
        .iter()
        .map(|(k, v)| json!([k.as_ref(), canon_attr(v)]))
        .collect();
    let kids: Vec<Value> = n
        .children
        .iter()
        .map(|&c| {
            json!(labels
                .get(&g.find_ref(c))
                .cloned()
                .unwrap_or_default())
        })
        .collect();
    marshal(&json!([n.op.as_ref(), attrs, kids]))
}

/// The graph's canonical class labels — a Weisfeiler–Lehman
/// refinement: label₀ is the empty marshal, labelₖ₊₁ is the digest of
/// each member's marshal with child labels folded in. Runs a fixed
/// `n_classes + 1` rounds on both engines — deterministic whether or
/// not the refinement has converged (isomorphic graphs label
/// identically at every round).
///
/// Returns the labels keyed by canonical class id.
fn wl_labels(
    g: &EGraph,
    members: &BTreeMap<ClassId, Vec<ENode>>,
) -> BTreeMap<ClassId, String> {
    let mut labels: BTreeMap<ClassId, String> =
        members.keys().map(|&c| (c, String::new())).collect();
    for _ in 0..=members.len() {
        let next: BTreeMap<ClassId, String> = members
            .iter()
            .map(|(&c, ns)| {
                let mut ms: Vec<String> =
                    ns.iter().map(|n| member_sig(g, n, &labels)).collect();
                ms.sort();
                (c, label_hash(&marshal(&json!(ms))))
            })
            .collect();
        if next == labels {
            break;
        }
        labels = next;
    }
    labels
}

/// Every canonical class id in the cone of `roots`.
///
/// The public surface has no class enumeration (`classes` is
/// `pub(crate)`), and enode ids do *not* cover the union-find:
/// congruence re-canonicalisation interns re-keyed enodes without
/// allocating fresh ids, so `n_enodes` can exceed the id space.
/// Instead we walk member-enode children downward from the input
/// roots — every live class is in that cone (an instantiated RHS
/// merges into the matched class; nothing else mints unreachable
/// classes) — the same traversal the Python side performs.
fn cone_ids(g: &EGraph, roots: &[ClassId]) -> Vec<ClassId> {
    let mut seen: BTreeSet<ClassId> = BTreeSet::new();
    let mut stack: Vec<ClassId> =
        roots.iter().map(|&r| g.find_ref(r)).collect();
    while let Some(c) = stack.pop() {
        if !seen.insert(c) {
            continue;
        }
        if let Some(ec) = g.class(c) {
            for n in &ec.nodes {
                for &ch in n.children.iter() {
                    stack.push(g.find_ref(ch));
                }
            }
        }
    }
    seen.into_iter().collect()
}

fn stop_str(s: StopReason) -> &'static str {
    match s {
        StopReason::FixedPoint => "fixed_point",
        StopReason::Improving => "improving",
        StopReason::MaxNodes => "max_nodes",
        StopReason::MaxIterations => "max_iterations",
    }
}

// ---------------------------------------------------------------------------
//  The run
// ---------------------------------------------------------------------------

/// Execute a JSON run spec and return the canonical report.
///
/// # Errors
///
/// `SpecError` on a malformed spec (decode failure, missing fields,
/// out-of-range root, bad option types) — the caller renders it.
pub fn run_spec(spec: &Value) -> Result<Value, SpecError> {
    let name = spec.get("name").and_then(Value::as_str).unwrap_or("spec");
    let terms = spec
        .get("terms")
        .and_then(Value::as_array)
        .ok_or_else(|| SpecError("\"terms\" must be an array".into()))?;
    if terms.is_empty() {
        return bad("\"terms\" must hold at least one term");
    }

    let mut g = EGraph::new();
    let mut roots: Vec<ClassId> = Vec::with_capacity(terms.len());
    for t in terms {
        let term = decode_term(t)?;
        let eid = g
            .add_term(&term, "input")
            .map_err(|e| SpecError(format!("add_term: {e:?}")))?;
        roots.push(eid);
    }
    let root_idx = spec
        .get("root")
        .and_then(Value::as_u64)
        .map(|i| i as usize)
        .unwrap_or(roots.len() - 1);
    let root = *roots
        .get(root_idx)
        .ok_or_else(|| SpecError(format!("root index {root_idx} out of range")))?;

    let mut rules: Vec<Rewrite> = Vec::new();
    if let Some(rs) = spec.get("rules").and_then(Value::as_array) {
        for r in rs {
            let rname = r
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| SpecError("rule missing \"name\"".into()))?;
            let lhs = decode_term(
                r.get("lhs")
                    .ok_or_else(|| SpecError(format!("rule {rname}: missing \"lhs\"")))?,
            )?;
            let rhs = decode_term(
                r.get("rhs")
                    .ok_or_else(|| SpecError(format!("rule {rname}: missing \"rhs\"")))?,
            )?;
            rules.push(Rewrite::new(rname, lhs, rhs));
        }
    }

    let options = spec.get("options").cloned().unwrap_or(Value::Null);
    let opt_u64 = |key: &str, default: usize| -> Result<usize, SpecError> {
        match options.get(key) {
            None | Some(Value::Null) => Ok(default),
            Some(v) => v
                .as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| SpecError(format!("option {key:?} must be a nonneg int"))),
        }
    };
    let mut opts = RunOptions {
        max_iterations: opt_u64("max_iterations", 100)?,
        max_nodes: opt_u64("max_nodes", 100_000)?,
        patience: opt_u64("patience", 3)?,
        ..RunOptions::default()
    };
    if let Some(b) = options.get("rule_budgets").and_then(Value::as_object) {
        for (k, v) in b {
            let cap = v
                .as_u64()
                .ok_or_else(|| SpecError(format!("rule_budgets[{k:?}] must be an int")))?;
            opts.rule_budgets.insert(k.clone(), cap as usize);
        }
    }
    opts.stop = match options.get("stop").and_then(Value::as_str) {
        None | Some("fixed_point") => Stop::FixedPoint,
        Some("improving") => Stop::Improving,
        Some(other) => return bad(format!("unknown stop policy {other:?}")),
    };

    let costs = {
        let c = spec.get("costs").cloned().unwrap_or(Value::Null);
        let mut ops = BTreeMap::new();
        if let Some(m) = c.get("ops").and_then(Value::as_object) {
            for (k, v) in m {
                let Some(f) = v.as_f64() else {
                    return bad(format!("costs.ops[{k:?}] must be a number"));
                };
                ops.insert(k.clone(), f);
            }
        }
        Costs {
            ops,
            default: c
                .get("default")
                .and_then(Value::as_f64)
                .unwrap_or(1.0),
            leaf: c.get("leaf").and_then(Value::as_f64).unwrap_or(0.0),
        }
    };
    let costs = Rc::new(costs);
    let c2 = Rc::clone(&costs);
    let mut cost_fn = move |t: &Term| cost_of(t, &c2);
    if opts.stop == Stop::Improving {
        opts.cost_fn = Some(&mut cost_fn);
    }

    let stats: SaturationStats = g
        .run(&rules, root, &mut opts)
        .map_err(|e| SpecError(format!("run: {e:?}")))?;

    // -- the canonical report -----------------------------------------
    let canon = cone_ids(&g, &roots);
    let mut members: BTreeMap<ClassId, Vec<ENode>> = BTreeMap::new();
    for &c in &canon {
        if let Some(ec) = g.class(c) {
            members.insert(c, ec.nodes.iter().cloned().collect());
        }
    }
    let labels = wl_labels(&g, &members);

    let mut classes: Vec<Value> = Vec::with_capacity(canon.len());
    for &c in &canon {
        let mut ms: Vec<String> = members[&c]
            .iter()
            .map(|n| member_sig(&g, n, &labels))
            .collect();
        ms.sort();
        let rep = g
            .any_term(c)
            .as_ref()
            .map(term_to_data)
            .unwrap_or(Value::Null);
        classes.push(json!({
            "sig": labels[&c],
            "rep": rep,
            "members": ms,
        }));
    }
    classes.sort_by(|a, b| a["sig"].as_str().cmp(&b["sig"].as_str()));
    let mut signatures: Vec<String> = canon.iter().map(|c| labels[c].clone()).collect();
    signatures.sort();
    let root_sig = labels[&g.find(root)].clone();

    let best = g.extract_best(root, &mut cost_fn, None, None);
    let best_cost = best.as_ref().map(|t| cost_of(t, &costs));

    let mut checks: Vec<Value> = Vec::new();
    if let Some(ct) = spec.get("check_terms").and_then(Value::as_array) {
        for t in ct {
            let term = decode_term(t)?;
            let found = g.class_of_term(&term);
            checks.push(json!({
                "term": t,
                "found": found.is_some(),
                "in_root": found.is_some_and(|c| g.find(c) == g.find(root)),
            }));
        }
    }

    let mut proof_rules: Vec<String> = g
        .merge_log()
        .iter()
        .filter_map(|e| e.rule.as_deref().map(str::to_string))
        .collect();
    proof_rules.sort();
    let mut app_rules: Vec<String> = g
        .applications()
        .iter()
        .map(|a| a.rule.to_string())
        .collect();
    app_rules.sort();
    let mut suspended = stats.budget_suspended.clone();
    suspended.sort();

    Ok(json!({
        "name": name,
        "ok": true,
        "stop": stop_str(stats.stop),
        "iterations": stats.iterations,
        "n_enodes": stats.n_enodes,
        "n_classes": stats.n_classes,
        "n_proof_edges": stats.n_proof_edges,
        "truncation_level": stats.truncation_level,
        "rule_budgets": stats.rule_budgets,
        "budget_suspended": suspended,
        "improved": stats.improved,
        "proof_rules": proof_rules,
        "app_rules": app_rules,
        "root_sig": root_sig,
        "signatures": signatures,
        "classes": classes,
        "best": best.as_ref().map(term_to_data).unwrap_or(Value::Null),
        "best_cost": best_cost,
        "checks": checks,
    }))
}

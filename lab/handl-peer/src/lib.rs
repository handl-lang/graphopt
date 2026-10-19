//! Spike: the smallest useful handl peer.
//!
//! One function, data in / data out: load a mlopt fragment directory as
//! a handl `Language`, apply one named rewrite rule at one node, and
//! return the rewritten AST as JSON.  That is enough to prove the shape
//! the plan asks about (mlopt calls handl) without designing an API
//! before the integration is proven.
//!
//! Path dependencies point at a local handl checkout; a real package
//! needs a published or git dependency.  See `README.md`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::Path;

use handl_language::{Ast, AstNode, AstRef};
use handl_kit_data_canonical::language_from_dir;
use handl_plugin_engine_transform_rewrite::RewriteEngine;

/// Load `dir` as a handl language and return its transform ids.
pub fn transform_ids(dir: &str) -> Result<Vec<String>, String> {
    let lang = language_from_dir(Path::new(dir)).map_err(|e| format!("load {dir}: {e}"))?;
    Ok(lang.transforms.iter().map(|d| d.id.clone()).collect())
}

/// Rewrite `ast_json` at `at` with `rule_id`, returning the new AST JSON.
pub fn rewrite(dir: &str, ast_json: &str, at: usize, rule_id: &str) -> Result<String, String> {
    let lang = language_from_dir(Path::new(dir)).map_err(|e| format!("load {dir}: {e}"))?;
    let spec: serde_json::Value =
        serde_json::from_str(ast_json).map_err(|e| format!("ast json: {e}"))?;
    let nodes = spec
        .as_array()
        .ok_or_else(|| "ast json must be an array of nodes".to_string())?;
    let mut ast = Ast::new();
    for n in nodes {
        let kind = n
            .get("kind")
            .and_then(|k| k.as_str())
            .ok_or_else(|| "each node needs a kind".to_string())?;
        let payload = n.get("payload").cloned().unwrap_or(serde_json::json!({}));
        ast.push(AstNode::new(kind).with_payload(payload));
    }
    let out = RewriteEngine::new()
        .apply_rule(&lang, &ast, AstRef::new(at), rule_id)
        .map_err(|e| format!("apply {rule_id}: {e}"))?;

    let render = |i: AstRef| -> serde_json::Value {
        let n = out.ast.get(i).expect("node");
        serde_json::json!({"kind": n.kind, "payload": n.payload})
    };
    let root = out.root;
    Ok(serde_json::json!({
        "rewrites": out.trace.rewrites.len(),
        "root_index": root.index(),
        "root": render(root),
        "arena": (0..out.ast.len()).map(|i| render(AstRef::new(i))).collect::<Vec<_>>(),
    })
    .to_string())
}

#[pyfunction]
fn language_transform_ids(dir: &str) -> PyResult<Vec<String>> {
    transform_ids(dir).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (dir, ast_json, at, rule_id))]
fn apply_rule(dir: &str, ast_json: &str, at: usize, rule_id: &str) -> PyResult<String> {
    rewrite(dir, ast_json, at, rule_id).map_err(PyValueError::new_err)
}

#[pymodule]
fn mlopt_handl_peer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(language_transform_ids, m)?)?;
    m.add_function(wrap_pyfunction!(apply_rule, m)?)?;
    Ok(())
}

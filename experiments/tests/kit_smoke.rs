//! Kit composition smoke — the assembled runtime evaluates real tensor
//! work through the `data`/`pratt`/`tree-walk` pipeline, and the
//! `experiments/.handl.toml` dep-name resolution is honored end to end
//! (the same seams `conv-bn-fold` exercises, minus the catopt artifact).

use std::path::PathBuf;

use mlopt_experiments::Kit;

fn project_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `tensor.matmul` through the ambient dotted-call dispatch —
/// `field_access` → `"tensor.matmul"` → the module chain → the Rust
/// impl — returning the `{"__tensor__": …}` wire form.
#[test]
fn matmul_runs_through_the_composed_kit() {
    let kit = Kit::new_with_project(project_dir()).unwrap();
    let v = kit
        .run(
            "tensor.to_list(tensor.matmul(\
                tensor.from_list([[1, 2], [3, 4]]), \
                tensor.from_list([[5, 6], [7, 8]])))",
        )
        .unwrap();
    assert_eq!(v, serde_json::json!([[19.0, 22.0], [43.0, 50.0]]));
}

/// `run_tagged` surfaces the real `TaggedValue::Tensor`, not the `Obj`
/// fallback a wire decode would produce.
#[test]
fn tensor_result_is_the_real_tagged_variant() {
    let kit = Kit::new_with_project(project_dir()).unwrap();
    let v = kit.run_tagged("tensor.ones([2, 2])").unwrap();
    assert!(
        matches!(v, handl_value::TaggedValue::Tensor(_)),
        "expected TaggedValue::Tensor, got {v:?}"
    );
}

/// The `.handl.toml` dep-name arm: `from "host-tensor" import tensor`
/// inside an imported module resolves through the project manifest —
/// exercised indirectly by every conv_bn.hnl import in the experiment.
#[test]
fn dependency_names_resolve_through_the_manifest() {
    let deps = handl_platform::resolve_dependencies(&project_dir())
        .expect("experiments/.handl.toml must parse");
    let host_tensor = deps
        .get("host-tensor")
        .expect("the manifest declares host-tensor");
    assert!(
        host_tensor.join("index.hnl").is_file(),
        "host-tensor must resolve to a dir holding index.hnl, got {host_tensor:?}"
    );
}

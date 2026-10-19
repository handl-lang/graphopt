//! The composition story, end to end: a real `.hnl` program runs through
//! the assembled runtime — `tensor.*` calls dispatch through the module
//! chain to `host-tensor`'s Rust impls, and a tensor comes back as a
//! `TaggedValue::Tensor` (not the `Obj` fallback the `{"__tensor__": …}`
//! wire map would otherwise decode to).
//!
//! Also covered: the `.handl.toml` `[dependencies]` name-resolution seam
//! (`from "host-tensor" import tensor`), via a temp-dir project whose
//! dep path is written at test time — the seam resolves filesystem
//! package dirs, so the fixture vendors the embedded `TENSOR_HNL`
//! interface into a package-shaped directory.

use std::fs;

use mlopt_kit::CatoptKit;
use handl_host_tensor::TENSOR_HNL;
use handl_value::TaggedValue;
use serde_json::{Value, json};

/// The `{"__tensor__": …}` wire form, for assertions over `run` results.
fn tensor(shape: &[usize], data: &[f64]) -> Value {
    json!({
        "__tensor__": {
            "shape": shape,
            "data": data,
            "dtype": "f64",
        }
    })
}

/// `tensor.matmul` through the assembled runtime — the ambient
/// dotted-name call form (`tensor` unbound → `"tensor"` →
/// `field_access` → `"tensor.matmul"` → `call_host` → the op registry).
#[test]
fn matmul_runs_through_the_assembled_runtime() {
    let kit = CatoptKit::new().unwrap();
    let out = kit
        .run("tensor.to_list(tensor.matmul(tensor.from_list([[1, 2], [3, 4]]), tensor.from_list([[5, 6], [7, 8]])))")
        .unwrap();
    assert_eq!(out, json!([[19.0, 22.0], [43.0, 50.0]]));
}

/// The bare op result crosses the boundary as the `__tensor__` wire map.
#[test]
fn matmul_result_is_the_tensor_wire_form() {
    let kit = CatoptKit::new().unwrap();
    let out = kit
        .run("tensor.matmul(tensor.from_list([[1, 2], [3, 4]]), tensor.from_list([[5, 6], [7, 8]]))")
        .unwrap();
    assert_eq!(out, tensor(&[2, 2], &[19.0, 22.0, 43.0, 50.0]));
}

/// `TaggedValue::from` on the wire value decodes a real
/// `TaggedValue::Tensor` — the kernel-level variant, not a plain `Obj`.
#[test]
fn boundary_value_decodes_to_tagged_tensor() {
    let kit = CatoptKit::new().unwrap();
    let out = kit
        .run("tensor.matmul(tensor.from_list([[1, 2], [3, 4]]), tensor.from_list([[5, 6], [7, 8]]))")
        .unwrap();
    let tagged = TaggedValue::from(out);
    let t = tagged
        .as_tensor()
        .expect("matmul result should decode to TaggedValue::Tensor");
    assert_eq!(t.shape, vec![2, 2]);
    assert_eq!(*t.data, vec![19.0, 22.0, 43.0, 50.0]);
}

/// The stronger statement: inside the engine the value is *already* a
/// `TaggedValue::Tensor` — `run_tagged` skips the serde boundary
/// entirely, so an `Obj` here would prove the value never decodes.
#[test]
fn run_tagged_returns_the_tensor_variant() {
    let kit = CatoptKit::new().unwrap();
    let out = kit
        .run_tagged("tensor.matmul(tensor.from_list([[1, 2], [3, 4]]), tensor.from_list([[5, 6], [7, 8]]))")
        .unwrap();
    let t = out
        .as_tensor()
        .expect("tagged matmul result should be TaggedValue::Tensor");
    assert_eq!(t.shape, vec![2, 2]);
    assert_eq!(t.rank(), 2);
    assert_eq!(*t.data, vec![19.0, 22.0, 43.0, 50.0]);
}

/// A tensor threaded *between* ops inside one expression: `from_list`'s
/// tagged result feeds `matmul`'s host call, then `add`'s — the
/// Tensor variant round-trips through the `Function` boundary each hop.
#[test]
fn tensor_chains_between_ops_inside_one_program() {
    let kit = CatoptKit::new().unwrap();
    let out = kit
        .run_tagged(
            "tensor.add(tensor.matmul(tensor.from_list([[1, 2], [3, 4]]), tensor.from_list([[5, 6], [7, 8]])), tensor.from_list([[1, 1], [1, 1]]))",
        )
        .unwrap();
    let t = out.as_tensor().expect("add result should be a Tensor");
    assert_eq!(t.shape, vec![2, 2]);
    assert_eq!(*t.data, vec![20.0, 23.0, 44.0, 51.0]);
}

/// conv2d smoke — a 1×1 kernel per-pixel channel mix (NCHW:
/// `[1, 2, 1, 2]` input × `[2, 2, 1, 1]` weight → `[1, 2, 1, 2]`).
#[test]
fn conv2d_smoke() {
    let kit = CatoptKit::new().unwrap();
    let out = kit
        .run("tensor.to_list(tensor.conv2d(tensor.from_list([[[[1, 2]], [[3, 4]]]]), tensor.from_list([[[[1]], [[2]]], [[[-1]], [[0]]]])))")
        .unwrap();
    assert_eq!(out, json!([[[[7.0, 10.0]], [[-1.0, -2.0]]]]));
}

/// batch_norm smoke — eval mode `(x − μ)/√(σ²+ε)·γ+β` with `eps = 0`:
/// channel 0 normalizes to identity, channel 1 maps `x → x − 2`.
#[test]
fn batch_norm_smoke() {
    let kit = CatoptKit::new().unwrap();
    let out = kit
        .run("tensor.to_list(tensor.batch_norm(tensor.from_list([[[[1, 2]], [[3, 4]]]]), tensor.from_list([1, 2]), tensor.from_list([0, 1]), tensor.from_list([0, 3]), tensor.from_list([1, 4]), 0))")
        .unwrap();
    assert_eq!(out, json!([[[[1.0, 2.0]], [[1.0, 2.0]]]]));
}

/// `tensor.derive` — the registered-adjoint table a `.hnl`
/// optimization loop reads: `batch_norm` → `batch_norm_backward`, an
/// unpaired name → `null`.
#[test]
fn derive_returns_the_registered_adjoint_name() {
    let kit = CatoptKit::new().unwrap();
    assert_eq!(
        kit.run("tensor.derive(\"batch_norm\")").unwrap(),
        json!("tensor.batch_norm_backward")
    );
    assert_eq!(
        kit.run("tensor.derive(\"tensor.conv2d\")").unwrap(),
        Value::Null
    );
}

/// The `.handl.toml` name-resolution seam: a temp project declares
/// `host-tensor = { path = "host-tensor" }` pointing at a package dir
/// vendored from the embedded `TENSOR_HNL` interface, and
/// `from "host-tensor" import tensor` binds the `tensor` namespace
/// record — whose string fields dispatch through the same module chain.
#[test]
fn handl_toml_dependency_resolves_host_tensor_by_name() {
    let tmp = std::env::temp_dir().join(format!("mlopt-kit-proj-{}", std::process::id()));
    let pkg = tmp.join("host-tensor");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(pkg.join("index.hnl"), TENSOR_HNL).unwrap();
    fs::write(
        tmp.join(".handl.toml"),
        "[dependencies]\nhost-tensor = { path = \"host-tensor\" }\n",
    )
    .unwrap();

    let kit = CatoptKit::new_with_project(tmp.clone()).unwrap();
    let out = kit
        .run(
            "do {\nfrom \"host-tensor\" import tensor\ntensor.to_list(tensor.matmul(tensor.from_list([[1, 2], [3, 4]]), tensor.from_list([[5, 6], [7, 8]])))\n}",
        )
        .unwrap();
    assert_eq!(out, json!([[19.0, 22.0], [43.0, 50.0]]));

    let _ = fs::remove_dir_all(&tmp);
}

/// Same seam, the `* as` whole-module form the `host-tensor` spec test
/// uses (`from "./index" import * as tensor`) — here through the
/// dependency name rather than a relative path.
#[test]
fn handl_toml_dependency_star_import_binds_the_module_record() {
    let tmp = std::env::temp_dir().join(format!("mlopt-kit-star-{}", std::process::id()));
    let pkg = tmp.join("host-tensor");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(pkg.join("index.hnl"), TENSOR_HNL).unwrap();
    fs::write(
        tmp.join(".handl.toml"),
        "[dependencies]\nhost-tensor = { path = \"host-tensor\" }\n",
    )
    .unwrap();

    let kit = CatoptKit::new_with_project(tmp.clone()).unwrap();
    let out = kit
        .run(
            "do {\nfrom \"host-tensor\" import * as t\nt.to_list(t.add(t.ones([2]), t.ones([2])))\n}",
        )
        .unwrap();
    assert_eq!(out, json!([2.0, 2.0]));

    let _ = fs::remove_dir_all(&tmp);
}

/// A mlopt-side `.hnl` module of its own — a temp file defining a
/// pure-handl helper over the tensor namespace — proving a
/// `packages/mlopt-kit`-shipped `.hnl` module could compose the same
/// way (`import "./mod.hnl"` resolves under `module_root`).
#[test]
fn a_mlopt_hnl_module_composes_the_tensor_surface() {
    let tmp = std::env::temp_dir().join(format!("mlopt-kit-mod-{}", std::process::id()));
    let pkg = tmp.join("host-tensor");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(pkg.join("index.hnl"), TENSOR_HNL).unwrap();
    fs::write(
        tmp.join(".handl.toml"),
        "[dependencies]\nhost-tensor = { path = \"host-tensor\" }\n",
    )
    .unwrap();
    // A consumer-authored module: one arrow over the imported namespace.
    fs::write(
        tmp.join("linalg.hnl"),
        "from \"host-tensor\" import tensor\n\
         export double_matmul := (a, b) ~> tensor.matmul(tensor.add(a, a), b)\n",
    )
    .unwrap();

    let kit = CatoptKit::new_with_project(tmp.clone()).unwrap();
    let out = kit
        .run(
            "do {\nfrom \"./linalg.hnl\" import double_matmul\n\
             tensor.to_list(double_matmul(tensor.from_list([[1, 2], [3, 4]]), tensor.from_list([[5, 6], [7, 8]])))\n}",
        )
        .unwrap();
    // double_matmul doubles a then multiplies: (2·a)·b = 2·(a·b).
    assert_eq!(out, json!([[38.0, 44.0], [86.0, 100.0]]));

    let _ = fs::remove_dir_all(&tmp);
}

/// Regression guard for the env this crate claims: the ambient
/// `tensor.f` call form keeps working *alongside* an import (the
/// imported record and the unbound-identifier fallback resolve through
/// the same `ModuleLookup`).
#[test]
fn ambient_and_imported_forms_agree() {
    let kit = CatoptKit::new().unwrap();
    let src = "tensor.to_list(tensor.add(tensor.from_list([[1, 2]]), tensor.from_list([[3, 4]])))";
    assert_eq!(kit.run(src).unwrap(), json!([[4.0, 6.0]]));
    assert_eq!(kit.run(src).unwrap(), kit.run(src).unwrap());
}

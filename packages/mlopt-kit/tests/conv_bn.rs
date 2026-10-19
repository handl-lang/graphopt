//! The conv+BN fold, end to end through the consumer: the crate's
//! *committed* `.handl.toml` names `host-tensor` as a git dependency and
//! `[patch]` dev-links it to the sibling checkout (platform `90853ec2`'s
//! git-package→dir bridge), `hnl/conv_bn.hnl` imports the op surface
//! *by name* through that bridge, and the program CatoptKit runs is the
//! program mlopt's e-graph optimizes —
//! `batch_norm(conv2d(x, W, b), γ, β, μ, σ²)` — plus its folded form
//! `conv2d(x, W′, b′)`, proven value-equal through the actual runtime.

use std::path::PathBuf;

use mlopt_kit::CatoptKit;
use handl_value::TaggedValue;
use serde_json::{Value, json};

/// `packages/mlopt-kit` — the project root the committed
/// `.handl.toml` and `hnl/` tree hang off.
fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The fixture input/params — every intermediate is a dyadic fraction,
/// so folded-vs-unfolded agreement is *exact* f64 equality, not a
/// tolerance:
///
/// - `x`: NCHW `[1, 2, 1, 2]`; `w`: `[2, 2, 1, 1]` 1×1 channel mix
///   (out0 = x0 + 2·x1, out1 = −x0); `b = [5, 7]` →
///   conv out `[[[[12, 15]], [[6, 5]]]]`.
/// - BN params: γ = `[3, 2]`, β = `[1, 1]`, μ = `[0, 3]`, σ² = `[1, 4]`,
///   `eps = 0` → r = `[1, 0.5]`, s = γ∘r = `[3, 1]` → BN out
///   `[[[[37, 46]], [[4, 3]]]]`.
const X: &str = "tensor.from_list([[[[1, 2]], [[3, 4]]]])";
const W: &str = "tensor.from_list([[[[1]], [[2]]], [[[-1]], [[0]]]])";
const B: &str = "tensor.from_list([5, 7])";
const G: &str = "tensor.from_list([3, 2])";
const BE: &str = "tensor.from_list([1, 1])";
const M: &str = "tensor.from_list([0, 3])";
const V: &str = "tensor.from_list([1, 4])";

/// The source run for one exported arrow of `hnl/conv_bn.hnl` — the
/// `.handl.toml` `host-tensor` dep is what the module's own
/// `from "host-tensor" import tensor` resolves through.
fn call_export(export: &str, args: &str) -> String {
    format!("do {{\nfrom \"./hnl/conv_bn.hnl\" import {export}\n{export}({args})\n}}")
}

/// The bridge half: the committed manifest's git dep resolves to the
/// `[patch]` sibling checkout where one is linked, else to its
/// `~/.handl/pkg` package-cache slot (the dir the ADR-0010 fetch hook
/// will populate — a checkout/symlink there is the stand-in today).
/// Both arms are asserted, so the test is honest on any layout.
#[test]
fn manifest_git_dep_resolves_through_patch_or_cache_slot() {
    let deps = handl_platform::resolve_dependencies(&crate_root())
        .expect("the committed .handl.toml must parse");
    // research-mlopt sits inside the handl-lang checkout, so the
    // sibling convention is `../../../<repo>` — the same relative path
    // the committed `.handl.toml`'s [patch] declares.
    let sibling = crate_root().join("../../../host-tensor/src");
    let resolved = deps
        .get("host-tensor")
        .expect("the manifest declares host-tensor");
    if sibling.is_dir() {
        assert_eq!(
            *resolved,
            std::fs::canonicalize(&sibling).unwrap(),
            "[patch] must win over the git source where the sibling is linked"
        );
    } else {
        assert_eq!(
            *resolved,
            handl_platform::package_cache_dir()
                .join("host-tensor")
                .join("src"),
            "inert [patch] → the git source's package-cache slot \
             (subdir = \"src\" names the package root in the clone)"
        );
    }
}

/// The `.hnl` consumer: `conv_bn` — `tensor.batch_norm` over
/// `tensor.conv2d`, both dispatched through the module chain — returns
/// the real `TaggedValue::Tensor`, not the `Obj` fallback.
#[test]
fn conv_bn_runs_through_the_project_runtime() {
    let kit = CatoptKit::new_with_project(crate_root()).unwrap();
    let out = kit
        .run_tagged(&call_export(
            "conv_bn",
            &format!("{X}, {W}, {B}, {G}, {BE}, {M}, {V}, 0"),
        ))
        .expect("conv_bn must eval through the imported host-tensor dep");
    let t = out
        .as_tensor()
        .expect("the composite's result is TaggedValue::Tensor");
    assert_eq!(t.shape, vec![1, 2, 1, 2]);
    assert_eq!(*t.data, vec![37.0, 46.0, 4.0, 3.0]);
}

/// The minted rewrite, evaluated: `conv_bn_fold` computes
/// `s = γ·rsqrt(σ²+ε)`, folds it into `W′ = W∘ŝ′` (the (C,1,1,1) view)
/// and `b′ = b∘s + (β − s∘μ)`, and runs a single `conv2d` — value-equal
/// to the unfolded composite through the same runtime.
#[test]
fn conv_bn_fold_evaluates_to_the_same_values() {
    let kit = CatoptKit::new_with_project(crate_root()).unwrap();
    let args = format!("{X}, {W}, {B}, {G}, {BE}, {M}, {V}, 0");
    let unfolded = kit
        .run_tagged(&call_export("conv_bn", &args))
        .unwrap()
        .as_tensor()
        .unwrap()
        .data
        .clone();
    let folded = kit
        .run_tagged(&call_export("conv_bn_fold", &args))
        .expect("the folded form must eval");
    let t = folded.as_tensor().expect("fold result is a Tensor");
    assert_eq!(t.shape, vec![1, 2, 1, 2]);
    assert_eq!(*t.data, *unfolded, "folded conv2d == batch_norm(conv2d)");
    assert_eq!(*t.data, vec![37.0, 46.0, 4.0, 3.0]);
}

/// The registered-adjoint surface through the same import:
/// `bn_adjoint` is `"tensor.batch_norm_backward"`, and `bn_grad` runs
/// it — `dx = g·γ·inv_std`, `dγ = Σ g·x̂`, `dβ = Σ g` over batch+spatial
/// (here the BN input is the conv output `[[[[12,15]],[[6,5]]]]`,
/// upstream grad all-ones → dx `[3,3,1,1]`, dγ `[27, 2.5]`, dβ `[2,2]`).
#[test]
fn derive_and_backward_compose_through_the_module() {
    let kit = CatoptKit::new_with_project(crate_root()).unwrap();
    let adjoint = kit
        .run("do {\nfrom \"./hnl/conv_bn.hnl\" import bn_adjoint\nbn_adjoint\n}")
        .unwrap();
    assert_eq!(adjoint, json!("tensor.batch_norm_backward"));

    let conv_out = "tensor.from_list([[[[12, 15]], [[6, 5]]]])";
    let out = kit
        .run(&call_export(
            "bn_grad",
            &format!("{conv_out}, {G}, {BE}, {M}, {V}, tensor.ones([1, 2, 1, 2]), 0"),
        ))
        .expect("bn_grad must eval");
    let Value::Object(grads) = &out else {
        panic!("bn_grad returns the grads record, got {out}");
    };
    for (field, expected) in [
        ("input", vec![3.0, 3.0, 1.0, 1.0]),
        ("gamma", vec![27.0, 2.5]),
        ("beta", vec![2.0, 2.0]),
    ] {
        let tagged = TaggedValue::from(grads[field].clone());
        let t = tagged
            .as_tensor()
            .unwrap_or_else(|| panic!("grads[{field}] should decode to a Tensor"));
        assert_eq!(*t.data, expected, "grads[{field}]");
    }
}

/// The artifact over the crate API: `CONV_BN_HNL` is the committed
/// `hnl/conv_bn.hnl` byte for byte, and `hnl_module` resolves it by
/// name — consumers read the module through the crate, not the repo.
#[test]
fn conv_bn_hnl_is_exported_over_the_crate_api() {
    let on_disk = std::fs::read_to_string(crate_root().join("hnl/conv_bn.hnl"))
        .expect("hnl/conv_bn.hnl is committed");
    assert_eq!(
        mlopt_kit::CONV_BN_HNL,
        on_disk,
        "export == hnl/conv_bn.hnl"
    );
    assert_eq!(
        mlopt_kit::hnl_module("conv_bn"),
        Some(mlopt_kit::CONV_BN_HNL)
    );
    assert_eq!(
        mlopt_kit::hnl_module("conv_bn.hnl"),
        Some(mlopt_kit::CONV_BN_HNL),
        "the file spelling resolves too"
    );
    assert_eq!(mlopt_kit::hnl_module("nope"), None);
}

/// The embedded source is the runnable artifact: a downstream consumer
/// with no repo access materializes `CONV_BN_HNL` to a file and imports
/// it — the `from "host-tensor" import tensor` inside still resolves
/// through the project's `.handl.toml`.
#[test]
fn embedded_conv_bn_imports_and_evaluates() {
    let dir = std::env::temp_dir().join(format!("mlopt-kit-hnl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("conv_bn.hnl");
    std::fs::write(&path, mlopt_kit::CONV_BN_HNL).unwrap();

    let kit = CatoptKit::new_with_project(crate_root()).unwrap();
    let src = format!(
        "do {{\nfrom \"{}\" import conv_bn\nconv_bn({X}, {W}, {B}, {G}, {BE}, {M}, {V}, 0)\n}}",
        path.display()
    );
    let out = kit
        .run_tagged(&src)
        .expect("the embedded source imports and evals");
    let t = out.as_tensor().expect("conv_bn result is a Tensor");
    assert_eq!(t.shape, vec![1, 2, 1, 2]);
    assert_eq!(*t.data, vec![37.0, 46.0, 4.0, 3.0]);
}

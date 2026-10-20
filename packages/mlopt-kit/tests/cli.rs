//! The `hnl-run` subprocess surface — the seam the Python bridge
//! (`bench/drivers/kit/term_hnl.py`) drives. The binary evaluates a `.hnl`
//! program through the assembled kit and prints the
//! `serde_json::Value` boundary form (`{"__tensor__": …}` for a
//! tensor) on stdout; failures exit non-zero with the engine error on
//! stderr. The surface is program-generic — the bridge's emitted terms
//! and the hand-written `hnl/` modules go through the same door.

use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

/// Feed `src` to the binary on stdin; return `(status, stdout, stderr)`.
fn run_stdin(src: &str) -> (bool, String, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_hnl-run"))
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("hnl-run must spawn");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(src.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (
        out.status.success(),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

/// A tensor program in, the `{"__tensor__": …}` boundary value out —
/// the wire shape the Python side decodes.
#[test]
fn stdin_program_evaluates_to_the_tensor_wire_form() {
    let (ok, stdout, stderr) = run_stdin(
        "tensor.matmul(tensor.from_list([[1, 2], [3, 4]]), tensor.from_list([[5, 6], [7, 8]]))",
    );
    assert!(ok, "hnl-run failed: {stderr}");
    let value: Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(
        value,
        json!({
            "__tensor__": {
                "shape": [2, 2],
                "data": [19.0, 22.0, 43.0, 50.0],
                "dtype": "f64",
            }
        })
    );
}

/// `do`-block bindings — the shape the emitter produces
/// (`name :: expr` lines, then the term expression).
#[test]
fn do_block_bindings_evaluate() {
    let (ok, stdout, stderr) = run_stdin(
        "do {\n  a :: tensor.from_list([1, 2])\n  b :: tensor.from_list([10, 20])\n  tensor.add(a, b)\n}",
    );
    assert!(ok, "hnl-run failed: {stderr}");
    let value: Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(
        value["__tensor__"]["data"],
        json!([11.0, 22.0]),
        "bound tensors must feed the op"
    );
}

/// An eval failure is a non-zero exit with the error on stderr — the
/// bridge's honest refusal path (a program the op surface rejects must
/// fail loudly, not silently degrade).
#[test]
fn eval_failure_exits_nonzero() {
    let (ok, _stdout, stderr) = run_stdin("tensor.matmul(tensor.from_list([1]), nope)");
    assert!(!ok, "an unbound identifier must fail");
    assert!(!stderr.is_empty(), "the error belongs on stderr");
}

/// `--repeat n` evaluates the source n times in one kit and prints the
/// last result — the runtime-cost measurement arm.
#[test]
fn repeat_evaluates_and_returns_the_last_result() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_hnl-run"))
        .args(["--repeat", "3", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"tensor.shape(tensor.ones([4]))")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let value: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(value, json!([4]));
}

/// A file path runs the file — the `hnl-run <file>` arm.
#[test]
fn file_argument_runs_the_file() {
    let dir = std::env::temp_dir().join(format!("hnl-run-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("prog.hnl");
    std::fs::write(&path, "tensor.shape(tensor.ones([2, 3]))").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_hnl-run"))
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr: {:?}", out.stderr);
    let value: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(value, json!([2, 3]));
}

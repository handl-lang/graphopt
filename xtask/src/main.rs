//! Repo gate tasks — vendored per Rust repo as `xtask/`.
//!
//! Enable with `.cargo/config.toml`:
//!   [alias]
//!   xtask = "run --manifest-path xtask/Cargo.toml --"
//!
//! Tasks: fmt, fmt-fix, clippy, check, test, doc, gate (all of the above).

use std::process::Command;

fn cargo(args: &[&str]) -> bool {
    println!("$ cargo {}", args.join(" "));
    Command::new("cargo")
        .args(args)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn gate() -> bool {
    for (name, args) in [
        ("fmt", &["fmt", "--all", "--", "--check"][..]),
        ("clippy", &["clippy", "--workspace", "--all-targets", "--", "-D", "warnings"][..]),
        ("test", &["test", "--workspace"][..]),
        ("doc", &["doc", "--workspace", "--no-deps"][..]),
    ] {
        if !cargo(args) {
            eprintln!("xtask gate: {name} failed");
            return false;
        }
    }
    true
}

fn main() {
    let task = std::env::args().nth(1).unwrap_or_else(|| "gate".into());
    let ok = match task.as_str() {
        "fmt" => cargo(&["fmt", "--all", "--", "--check"]),
        "fmt-fix" => cargo(&["fmt", "--all"]),
        "clippy" => cargo(&["clippy", "--workspace", "--all-targets", "--", "-D", "warnings"]),
        "check" => cargo(&["check", "--workspace", "--all-targets"]),
        "test" => cargo(&["test", "--workspace"]),
        "doc" => cargo(&["doc", "--workspace", "--no-deps"]),
        "gate" => gate(),
        other => {
            eprintln!("usage: cargo xtask [fmt|fmt-fix|clippy|check|test|doc|gate] (got {other:?})");
            false
        }
    };
    std::process::exit(if ok { 0 } else { 1 });
}

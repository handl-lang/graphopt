//! `rewrite <fragment-dir> <rule-id> <at> <ast-json>` — the peer as a CLI.
//!
//! The PyO3 extension (`catopt_handl_peer`, built with maturin) is the
//! shipping surface; this binary is the fallback for a checkout without a
//! maturin build.  Both call the same library code, so they prove the same
//! *call path* (catopt -> handl's loader + rewrite engine) and are
//! interchangeable — the tests prefer the extension and fall back here.
//!
//! Build it against the venv interpreter: `PYO3_PYTHON=<venv>/bin/python`
//! (a bare `cargo build` binds to `python3` on `PATH`, which may be a
//! different minor version — see `README.md`, *The `Py_TYPE` trap*).

use catopt_handl_peer::rewrite;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        eprintln!("usage: rewrite <dir> <rule-id> <at> <ast-json>");
        return ExitCode::from(2);
    }
    let (dir, rule, at, ast_json) = (&args[0], &args[1], &args[2], &args[3]);
    let Ok(index) = at.parse::<usize>() else {
        eprintln!("bad index {at}");
        return ExitCode::from(2);
    };
    match rewrite(dir, ast_json, index, rule) {
        Ok(out) => {
            println!("{out}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

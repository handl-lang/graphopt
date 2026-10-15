//! `rewrite <fragment-dir> <rule-id> <at> <ast-json>` — the peer as a CLI.
//!
//! The PyO3 extension is the shipping shape, but its packaging needs
//! maturin (not installed here); this binary proves the *call path*
//! (catopt -> handl's loader + rewrite engine) with the same library
//! code, so the integration can be verified from Python today.

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

//! `egraph-run` — run a JSON e-graph spec through [`mlopt_kit::egraph`];
//! print the canonical report.
//!
//! The subprocess seam the differential driver
//! (`bench/audit/egraph_differential.py`,
//! `tests/test_egraph_differential.py`) drives: a run spec goes in on
//! stdin (or a file path), the report comes out on stdout. Spec and
//! report shapes are documented on [`mlopt_kit::egraph::run_spec`].
//!
//! usage: `egraph-run [<spec.json> | -]`
//!
//! Exit codes: 0 success; 1 spec/run failure (message on stderr);
//! 2 usage/IO failure.

use std::io::Read;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() > 1 {
        eprintln!("usage: egraph-run [<spec.json> | -]");
        return ExitCode::from(2);
    }
    let src = match args.first().map(String::as_str) {
        None | Some("-") => {
            let mut s = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut s) {
                eprintln!("egraph-run: stdin: {e}");
                return ExitCode::from(2);
            }
            s
        }
        Some(path) => match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("egraph-run: {path}: {e}");
                return ExitCode::from(2);
            }
        },
    };
    let spec: serde_json::Value = match serde_json::from_str(&src) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("egraph-run: spec JSON: {e}");
            return ExitCode::from(2);
        }
    };
    match mlopt_kit::egraph::run_spec(&spec) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("egraph-run: {e}");
            ExitCode::from(1)
        }
    }
}

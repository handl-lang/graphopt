//! `hnl-run` — run a `.hnl` program through [`CatoptKit`]; print the value.
//!
//! The subprocess seam the Python bridge (`lab/kit/term_hnl.py`)
//! drives: a `.hnl` source goes in on stdin (or a file path), the
//! `serde_json::Value` boundary form comes out on stdout —
//! `{"__tensor__": …}` for a tensor result. Nothing catopt-term-aware:
//! any program the kit's op surface evaluates runs through it, which is
//! what makes it the honest Python↔kit boundary (the bridge emits a
//! *program*, the binary just runs programs).
//!
//! usage: `hnl-run [--project <dir>] [<file.hnl>]`
//!
//! - `--project <dir>` — assemble with `<dir>/.handl.toml`'s
//!   `[dependencies]` wired (`from "host-tensor" import tensor`
//!   resolves by name); without it `import "./f.hnl"` resolves against
//!   the process working directory and the ambient `tensor.*` dotted
//!   calls still dispatch.
//! - `--repeat <n>` — evaluate the source `n` times in the same kit,
//!   printing the last result. The measurement hook: per-call runtime
//!   cost is `(t(n) − t(1)) / (n − 1)` wall — how an optimization
//!   *decision* is costed through the actual kit runtime rather than
//!   the term-level `dag_cost`.
//! - `<file.hnl>` — the program path; `-` or absent reads stdin.
//!
//! Exit codes: 0 success; 1 eval/assembly failure (message on stderr);
//! 2 usage/IO failure.

use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;

use catopt_kit::CatoptKit;

fn usage() -> ExitCode {
    eprintln!("usage: hnl-run [--project <dir>] [--repeat <n>] [<file.hnl> | -]");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut project: Option<PathBuf> = None;
    let mut file: Option<String> = None;
    let mut repeat = 1usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--project" => match args.next() {
                Some(dir) => project = Some(PathBuf::from(dir)),
                None => return usage(),
            },
            "--repeat" => match args.next().and_then(|n| n.parse().ok()) {
                Some(n) if n >= 1 => repeat = n,
                _ => return usage(),
            },
            _ if file.is_none() => file = Some(arg),
            _ => return usage(),
        }
    }

    let src = match file.as_deref() {
        None | Some("-") => {
            let mut s = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut s) {
                eprintln!("hnl-run: stdin: {e}");
                return ExitCode::from(2);
            }
            s
        }
        Some(path) => match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("hnl-run: {path}: {e}");
                return ExitCode::from(2);
            }
        },
    };

    let kit = match &project {
        Some(dir) => CatoptKit::new_with_project(dir.clone()),
        None => CatoptKit::new(),
    };
    let kit = match kit {
        Ok(kit) => kit,
        Err(e) => {
            eprintln!("hnl-run: kit assembly: {e:?}");
            return ExitCode::from(1);
        }
    };

    let mut value = None;
    for _ in 0..repeat {
        match kit.run(&src) {
            Ok(v) => value = Some(v),
            Err(e) => {
                eprintln!("hnl-run: {e:?}");
                return ExitCode::from(1);
            }
        }
    }
    match value {
        Some(v) => {
            println!("{v}");
            ExitCode::SUCCESS
        }
        None => unreachable!("repeat >= 1"),
    }
}

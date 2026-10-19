//! `conv-bn-fold` — the founding-result experiment.
//!
//! Runs mlopt's conv+BN weight-space fold **end to end through the
//! handl runtime** and produces the two numbers the paper's headline
//! section cites:
//!
//! 1. **Certification** — the spelled program
//!    `conv_bn(x, W, b, γ, β, μ, σ², ε)` (`tensor.batch_norm` over
//!    `tensor.conv2d`) and the minted law's folded RHS `conv_bn_fold`
//!    (`conv2d(x, W′, b′)` with `W′ = W∘ŝ′`, `b′ = b∘s + (β − s∘μ)`)
//!    evaluate to *value-equal* tensors through the kit's real
//!    `lex → parse → eval` pipeline.
//! 2. **Runtime pay** — per-call wall time of each arm evaluated N
//!    times in one kit (`hnl-run --repeat`'s measurement shape, in
//!    process): wall/N per arm, reported as the e2e delta.
//!
//! Nothing is vendored: the `.hnl` program under measurement is
//! mlopt-kit's own `hnl/conv_bn.hnl` — in this merged tree it resolves
//! in-repo at `packages/mlopt-kit/hnl/conv_bn.hnl`; see
//! [`resolve_conv_bn`] for the full resolution ladder (the sibling /
//! pkg-slot / fetched arms keep a standalone checkout working).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use mlopt_experiments::CatoptKit;

/// The crate dir — `experiments/`, whose `.handl.toml` anchors
/// `from "host-tensor" import tensor` inside the imported program.
fn project_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Resolve mlopt-kit's `hnl/conv_bn.hnl` — the artifact under
/// measurement — without copying it into this repo's tree.
///
/// The ladder, first hit wins:
///
/// 1. `--conv-bn <path>` flag / `CONV_BN_HNL` env — explicit override.
/// 2. `experiments/../packages/mlopt-kit/hnl/conv_bn.hnl` — in this
///    merged tree mlopt-kit is an in-repo workspace member, so the
///    artifact is always present in-tree.
/// 3. `$MLOPT_DIR/packages/mlopt-kit/hnl/conv_bn.hnl` — a mlopt
///    checkout the caller names (standalone-checkout arm).
/// 4. The `.handl.toml` `[patch]` arm — the sibling-checkout dev-link:
///    `<repo>/../mlopt/packages/mlopt-kit` (exists wherever mlopt is
///    an org sibling of a standalone research-mlopt checkout).
/// 5. The package-cache slot the `[dependencies]` git arm resolves to:
///    `$HANDL_PKG_DIR/mlopt-kit` else `~/.handl/pkg/mlopt-kit` — the
///    directory the ADR-0010 fetch hook will populate (a checkout or
///    symlink there is today's stand-in).
/// 6. `experiments/.cache/mlopt-kit/hnl/conv_bn.hnl` — the file
///    `make artifact` fetches from the declared git source
///    (`raw.githubusercontent.com/handl-lang/research-mlopt/main/…`) when no
///    local arm resolves. `.cache/` is ignored — the file is a
///    *fetched dependency artifact*, not a committed copy.
fn resolve_conv_bn(overrides: &cli::Args) -> Result<PathBuf, String> {
    let candidates: Vec<PathBuf> = [
        overrides.conv_bn.clone(),
        std::env::var("CONV_BN_HNL").ok().map(PathBuf::from),
        Some(project_dir().join("../packages/mlopt-kit/hnl/conv_bn.hnl")),
        std::env::var("MLOPT_DIR")
            .ok()
            .map(|d| PathBuf::from(d).join("packages/mlopt-kit/hnl/conv_bn.hnl")),
        Some(project_dir().join("../../mlopt/packages/mlopt-kit/hnl/conv_bn.hnl")),
        Some(pkg_dir().join("mlopt-kit/hnl/conv_bn.hnl")),
        Some(project_dir().join(".cache/mlopt-kit/hnl/conv_bn.hnl")),
    ]
    .into_iter()
    .flatten()
    .collect();
    for c in &candidates {
        if c.is_file() {
            return Ok(c.clone());
        }
    }
    Err(format!(
        "could not resolve mlopt-kit's hnl/conv_bn.hnl — tried:\n  {}\n\
         provide it explicitly with --conv-bn <path> or CONV_BN_HNL, or\n\
         fetch it from the declared source: make artifact\n\
         (Gap, documented in manifest/kit.toml: `.handl.toml` git deps \
         resolve to a package-cache slot nothing yet populates — the \
         ADR-0010 fetch hook is upstream's follow-up.)",
        candidates
            .iter()
            .map(|c| c.display().to_string())
            .collect::<Vec<_>>()
            .join("\n  ")
    ))
}

/// `$HANDL_PKG_DIR` else `~/.handl/pkg` — the package-cache root the
/// `.handl.toml` git-dep arm resolves to.
fn pkg_dir() -> PathBuf {
    if let Ok(d) = std::env::var("HANDL_PKG_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".handl/pkg")
}

/// A `.hnl` argument expression for a nested number array.
mod hnl {
    /// Render a flat list of integers as a `.hnl` list literal. All
    /// fixture values are integers, so the emitted arrays are exactly
    /// representable and folded-vs-spelled agreement is *bitwise* f64
    /// equality, not a tolerance.
    pub fn list(v: &[i64]) -> String {
        format!(
            "[{}]",
            v.iter().map(i64::to_string).collect::<Vec<_>>().join(", ")
        )
    }

    /// `tensor.from_list(<literal>)`.
    pub fn tensor(lit: &str) -> String {
        format!("tensor.from_list({lit})")
    }
}

/// The dyadic fixture — the same shape regime as mlopt's
/// `ConvBN(cin=4, cout=6, k=3)` premise-pay seed, with every value a
/// small integer so the fold agrees bitwise.
///
/// `x`: NCHW `[1,4,8,8]`; `w`: `[6,4,3,3]`; `b`: `[6]`.
/// BN: `γ=[3,2,1,4,2,1]`, `β=[1,1,0,2,1,0]`, `μ=[0,3,1,2,0,1]`,
/// `σ²=[1,4,1,4,1,4]` (→ `rsqrt = [1,.5,1,.5,1,.5]`), `eps=0`.
struct Fixture {
    x: String,
    w: String,
    b: String,
    g: String,
    be: String,
    m: String,
    v: String,
    eps: String,
}

impl Fixture {
    fn dyadic() -> Self {
        // x[n=0][c][h][w] — small integers from a fixed mix.
        let mut xc = Vec::new();
        for c in 0..4i64 {
            let mut ch = Vec::new();
            for h in 0..8i64 {
                let row: Vec<i64> = (0..8i64)
                    .map(|w| (c * 11 + h * 5 + w * 3) % 9 - 4)
                    .collect();
                ch.push(hnl::list(&row));
            }
            xc.push(format!("[{}]", ch.join(", ")));
        }
        let x = hnl::tensor(&format!("[[{}]]", xc.join(", ")));

        // w[o][i][kh][kw] — small integers.
        let mut wo = Vec::new();
        for o in 0..6i64 {
            let mut wi = Vec::new();
            for i in 0..4i64 {
                let mut wk = Vec::new();
                for kh in 0..3i64 {
                    let row: Vec<i64> = (0..3i64)
                        .map(|kw| (o * 13 + i * 7 + kh * 3 + kw) % 5 - 2)
                        .collect();
                    wk.push(hnl::list(&row));
                }
                wi.push(format!("[{}]", wk.join(", ")));
            }
            wo.push(format!("[{}]", wi.join(", ")));
        }
        let w = hnl::tensor(&format!("[{}]", wo.join(", ")));

        Self {
            x,
            w,
            b: hnl::tensor(&hnl::list(&[5, -3, 7, 1, -2, 4])),
            g: hnl::tensor(&hnl::list(&[3, 2, 1, 4, 2, 1])),
            be: hnl::tensor(&hnl::list(&[1, 1, 0, 2, 1, 0])),
            m: hnl::tensor(&hnl::list(&[0, 3, 1, 2, 0, 1])),
            v: hnl::tensor(&hnl::list(&[1, 4, 1, 4, 1, 4])),
            eps: "0".into(),
        }
    }

    /// The positional argument tuple both exports take.
    fn args(&self) -> String {
        format!(
            "{}, {}, {}, {}, {}, {}, {}, {}",
            self.x, self.w, self.b, self.g, self.be, self.m, self.v, self.eps
        )
    }
}

/// Build the evaluation program: import one export from the resolved
/// mlopt-kit module and apply it to the fixture. The import names an
/// absolute path (the resolved artifact); the program's own
/// `from "host-tensor" import tensor` resolves through the project
/// `.handl.toml`.
fn program(conv_bn_hnl: &Path, export: &str, f: &Fixture) -> String {
    format!(
        "do {{\nfrom \"{}\" import {export}\n{export}({})\n}}",
        conv_bn_hnl.display(),
        f.args()
    )
}

/// Decode the `{"__tensor__": {shape, data, dtype}}` wire form for
/// reporting; returns `(shape, data)` or `None` for non-tensor values.
fn decode_wire(v: &serde_json::Value) -> Option<(Vec<i64>, Vec<f64>)> {
    let t = v.get("__tensor__")?;
    let shape = t
        .get("shape")?
        .as_array()?
        .iter()
        .map(|n| n.as_i64())
        .collect::<Option<Vec<_>>>()?;
    let data = t
        .get("data")?
        .as_array()?
        .iter()
        .map(|n| n.as_f64())
        .collect::<Option<Vec<_>>>()?;
    Some((shape, data))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: conv-bn-fold [--repeat <n>] [--conv-bn <path.hnl>] [--json]\n\
         \n\
         evaluate mlopt's conv_bn.hnl (spelled + folded arms) through\n\
         the composed kit; verify value-equality, then time N calls per arm.\n\
         \n\
         --repeat <n>   timed evals per arm (default 200; 0 = verify only)\n\
         --conv-bn <p>  explicit path to conv_bn.hnl (env: CONV_BN_HNL)\n\
         --json         emit the report as one JSON line last"
    );
    ExitCode::from(2)
}

mod cli {
    use std::path::PathBuf;

    pub struct Args {
        pub repeat: usize,
        pub conv_bn: Option<PathBuf>,
        pub json: bool,
    }

    pub fn parse() -> Result<Args, ()> {
        let mut repeat = 200usize;
        let mut conv_bn = None;
        let mut json = false;
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--repeat" => match args.next().and_then(|n| n.parse().ok()) {
                    Some(n) => repeat = n,
                    None => return Err(()),
                },
                "--conv-bn" => match args.next() {
                    Some(p) => conv_bn = Some(PathBuf::from(p)),
                    None => return Err(()),
                },
                "--json" => json = true,
                _ => return Err(()),
            }
        }
        Ok(Args {
            repeat,
            conv_bn,
            json,
        })
    }
}

fn run() -> Result<(), String> {
    let args = cli::parse().map_err(|_| "bad arguments".to_string())?;
    let conv_bn_hnl = resolve_conv_bn(&args)?;
    let kit = CatoptKit::new_with_project(project_dir()).map_err(|e| format!("kit assembly: {e:?}"))?;
    let f = Fixture::dyadic();

    let spelled = program(&conv_bn_hnl, "conv_bn", &f);
    let folded = program(&conv_bn_hnl, "conv_bn_fold", &f);
    // The *deployed* arm: a weight-space fold's real pay is amortized —
    // `W′`/`b′` are computed once at load time and the deployed program
    // is a single `conv2d(x, W′, b′)`. `W′`/`b′` have the same shapes as
    // `W`/`b` (`W∘ŝ′` is elementwise, `b∘s + (β − s∘μ)` rank-1), so
    // `conv2d` over the fixture weights is the honest per-call cost —
    // op cost here is shape-driven, not value-driven.
    let deployed = format!("tensor.conv2d({}, {}, {})", f.x, f.w, f.b);

    // -- 1. certification: value-equality through the real runtime ----
    let v_spelled = kit
        .run(&spelled)
        .map_err(|e| format!("eval conv_bn: {e:?}"))?;
    let v_folded = kit
        .run(&folded)
        .map_err(|e| format!("eval conv_bn_fold: {e:?}"))?;
    let (s_shape, s_data) =
        decode_wire(&v_spelled).ok_or("conv_bn did not return a tensor wire value")?;
    let (f_shape, f_data) =
        decode_wire(&v_folded).ok_or("conv_bn_fold did not return a tensor wire value")?;
    let equal = v_spelled == v_folded;
    let diff = max_abs_diff(&s_data, &f_data);
    println!("== conv-bn-fold experiment ==");
    println!("artifact: {}", conv_bn_hnl.display());
    println!(
        "[certify] spelled shape={s_shape:?} folded shape={f_shape:?} \
         wire_equal={equal} max_abs_diff={diff:e}"
    );
    if !equal {
        return Err("certification failed: folded != spelled".into());
    }

    // -- 2. runtime pay: per-call wall through the kit ----------------
    let n = args.repeat;
    let mut per: [Option<f64>; 3] = [None; 3];
    if n > 0 {
        let arms = [
            ("spelled (conv+BN)", &spelled),
            ("folded-inline", &folded),
            ("deployed (conv only)", &deployed),
        ];
        for (i, (_, prog)) in arms.iter().enumerate() {
            for _ in 0..5.min(n) {
                let _ = kit.run(prog);
            }
            let t0 = Instant::now();
            for _ in 0..n {
                let _ = kit.run(prog).map_err(|e| format!("{e:?}"))?;
            }
            per[i] = Some(t0.elapsed().as_secs_f64() / n as f64);
        }
        let [ps, pf, pd] = per;
        println!("[pay] per-call wall (n={n}):");
        for ((name, _), v) in arms.iter().zip(per.iter()) {
            println!("    {name:<24} {:.1}µs", v.unwrap() * 1e6);
        }
        if let (Some(s), Some(f), Some(d)) = (ps, pf, pd) {
            println!(
                "    e2e delta: inline={:+.1}%  deployed={:+.1}%  \
                 (fold math is a one-time compile-time cost in the deployed arm)",
                (f / s - 1.0) * 100.0,
                (d / s - 1.0) * 100.0
            );
        }
    }

    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "experiment": "conv-bn-fold",
                "artifact": conv_bn_hnl.display().to_string(),
                "certified": equal,
                "max_abs_diff": diff,
                "shape": s_shape,
                "repeat": n,
                "per_call_us": {
                    "spelled": per[0].map(|t| t * 1e6),
                    "folded_inline": per[1].map(|t| t * 1e6),
                    "deployed": per[2].map(|t| t * 1e6),
                },
                "e2e_delta_pct": {
                    "inline": match (per[0], per[1]) {
                        (Some(s), Some(f)) => Some((f / s - 1.0) * 100.0),
                        _ => None,
                    },
                    "deployed": match (per[0], per[2]) {
                        (Some(s), Some(d)) => Some((d / s - 1.0) * 100.0),
                        _ => None,
                    },
                },
            })
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    // Re-parse for usage errors (parse is idempotent).
    if cli::parse().is_err() {
        return usage();
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("conv-bn-fold: {e}");
            ExitCode::from(1)
        }
    }
}

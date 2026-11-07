"""M3 breadth arm: generate ``lib-triton-qgemv``'s manifest from a
real checkpoint — the routing table is *searched*, not hand-written.

For every quantized operator in the checkpoint (enumerated from the
``.qweight``/``.qzeros``/``.scales`` triples, not a hand list) this
driver:

1. groups operators by contraction/output shape ``(k, n)`` — the
   kernel's term fixes ``k`` and ``split``; ``n`` and the launch
   config are runtime arguments, so one vendored module serves every
   operator sharing a ``(k, split)`` pair;
2. sweeps ``split_k ∈ {1,2,4,8}`` (the term space) x a launch grid
   ``(block_n, block_k, warps)``, gating every candidate against the
   real incumbent ``vllm._custom_ops.awq_gemm`` on the operator's own
   tensors — the same order ``mlopt_kernel.search`` enforces;
3. times the winner per operator (CUDA-graph referee,
   ``timeit_graph``) and records ``us`` / cosine / speedup;
4. vendors the winning ``(k, split)`` modules as source into
   ``lib-triton-qgemv/src/lib_triton_qgemv/kernels/`` and rewrites
   ``manifest.py`` — every row carries its provenance.

    PYTHONPATH=$PWD:<kernelopt>/src:<lib-kernel-ir>/src \
      python -m bench.drivers.probes.awq.gen_manifest \
        --lib /home/luis/GitHub/handl-lang/lib-triton-qgemv \
        --out docs/reports/p27_manifest_breadth.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import torch
from lib_kernel_ir import qgemv
from mlopt_kernel.search import Candidate, search
from safetensors import safe_open

from .awq_packing import to_standard_order
from .irtriton import cached_module, compile_source
from .kernel_v2 import BW_GBPS, roofline_us, timeit_graph

CKPT = (
    "/home/luis/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-"
    "Instruct-AWQ/snapshots/db09cd27ead7fee40cdee309693cf83601b9c899/"
    "model.safetensors"
)

SPLITS = (1, 2, 4, 8)
#: launch grid — bn must stay a multiple of 8 (the kernel packs 8
#: outputs per word); bk <= GROUP=128 (the hoisted dequant needs it)
BN_GRID = (32, 64, 128)
BK_GRID = (64, 128)
WARP_GRID = (2, 4)

_GROUP = 128  # AWQ group size along k


@dataclass
class OpSpec:
    """One quantized operator, enumerated from the checkpoint."""

    name: (
        str  # last path component, e.g. "mlp.down_proj" -> "down_proj"
    )
    path: str  # full tensor prefix inside the safetensors file
    n: int
    k: int
    layer: int


@dataclass
class ShapeResult:
    """The search outcome for one (k, n) shape group."""

    k: int
    n: int
    winner: dict | None = None
    rows: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)


def enumerate_operators(path: str, layer: int) -> list[OpSpec]:
    """Every ``{prefix}.{qweight,qzeros,scales}`` triple at *layer*."""
    pat = re.compile(rf"^model\.layers\.{layer}\.(.+)\.qweight$")
    ops: list[OpSpec] = []
    with safe_open(path, "pt") as f:
        keys = set(f.keys())
        for key in sorted(keys):
            m = pat.match(key)
            if not m:
                continue
            prefix = f"model.layers.{layer}.{m.group(1)}"
            if (
                f"{prefix}.qzeros" not in keys
                or f"{prefix}.scales" not in keys
            ):
                continue
            qw = f.get_slice(f"{prefix}.qweight").get_shape()
            sc = f.get_slice(f"{prefix}.scales").get_shape()
            k, packed_n = qw
            n = sc[1]
            if packed_n * 8 != n:
                raise ValueError(
                    f"{prefix}: qweight packs {packed_n * 8} cols "
                    f"but scales says n={n}"
                )
            ops.append(
                OpSpec(
                    name=m.group(1).split(".")[-1],
                    path=prefix,
                    n=n,
                    k=k,
                    layer=layer,
                )
            )
    return ops


def _run_with(mod, x, qw, qz, sc, n, k, bn, bk, warps):
    return mod.run(x, qw, qz, sc, n, k, bn, bk, warps)


def sweep_shape(
    *,
    k: int,
    n: int,
    qw_std: torch.Tensor,
    qz_std: torch.Tensor,
    sc: torch.Tensor,
    x: torch.Tensor,
    incumbent_fn,
    incumbent_us: float,
) -> tuple[dict | None, list[dict]]:
    """Term sweep (split_k) x launch sweep, gated vs the incumbent.

    Returns ``(winner_row, rejected_rows)`` where the winner row
    carries the full launch config the manifest will ship.
    """
    xf = x.reshape(-1)
    expected = incumbent_fn()

    per_split: dict[int, dict] = {}
    rejected: list[dict] = []

    for split in SPLITS:
        term = qgemv(k=k, split=split)
        mod = cached_module(term)
        # a mid config for the gate: any valid launch denotes the same
        # term, so the cosine is a property of the term, not the config
        gate_bn = min(128, n - n % 8)
        try:
            out = _run_with(
                mod,
                xf,
                qw_std,
                qz_std,
                sc,
                n,
                k,
                gate_bn,
                64,
                2,
            )
        except Exception as exc:
            rejected.append(
                {
                    "label": f"split{split}",
                    "reason": "failed to run",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        cos = torch.nn.functional.cosine_similarity(
            out.detach().flatten().float(),
            expected.detach().flatten().float(),
            dim=0,
        ).item()
        if cos < 0.9999:
            rejected.append(
                {
                    "label": f"split{split}",
                    "reason": "gate",
                    "detail": f"cosine {cos:.6f} < 0.9999 vs awq_gemm",
                }
            )
            continue

        # launch sweep — the term is fixed; only when/how it runs moves
        best = None
        for bn in BN_GRID:
            if bn > n or bn % 8:
                continue
            for bk in BK_GRID:
                if bk > _GROUP:
                    continue
                for warps in WARP_GRID:

                    def run(mod=mod, bn=bn, bk=bk, warps=warps):
                        return _run_with(
                            mod,
                            xf,
                            qw_std,
                            qz_std,
                            sc,
                            n,
                            k,
                            bn,
                            bk,
                            warps,
                        )

                    us = timeit_graph(run, calls=10, replays=3)
                    if us is None:
                        continue
                    if best is None or us < best[0]:
                        best = (us, bn, bk, warps, cos)
        if best is None:
            rejected.append(
                {
                    "label": f"split{split}",
                    "reason": "unmeasurable",
                    "detail": "no launch config captured",
                }
            )
            continue
        us, bn, bk, warps, _ = best
        per_split[split] = {
            "term": term,
            "label": f"split{split}",
            "bn": bn,
            "bk": bk,
            "warps": warps,
            "us": us,
            "cos": cos,
        }

    if not per_split:
        return None, rejected

    candidates = []
    for _split, rec in per_split.items():

        def build(rec=rec):
            return _run_with(
                cached_module(rec["term"]),
                xf,
                qw_std,
                qz_std,
                sc,
                n,
                k,
                rec["bn"],
                rec["bk"],
                rec["warps"],
            )

        candidates.append(
            Candidate(
                term=rec["term"],
                label=f"{rec['label']} bn{rec['bn']} bk{rec['bk']} "
                f"w{rec['warps']}",
                build=build,
                measure=lambda rec=rec: rec["us"],
            )
        )
    result = search(
        operator=f"qgemv/k{k}n{n}",
        candidates=candidates,
        incumbent=incumbent_fn,
        bytes_moved=n * k * 0.5 + 2 * k + 2 * n,
        bandwidth_gbps=BW_GBPS,
        measure_incumbent=lambda: incumbent_us,
    )
    rejected.extend(result.to_dict()["rejected"])

    if result.chosen is None:
        return None, rejected

    # recover the swept record for the winning label's split
    win_label = result.chosen.label
    win_split = int(re.search(r"split(\d+)", win_label).group(1))
    rec = per_split[win_split]
    # search's cosine is on the winning config; keep the swept one —
    # identical term, identical denotation
    winner = {
        "k": k,
        "n": n,
        "split": win_split,
        "block_n": rec["bn"],
        "block_k": rec["bk"],
        "warps": rec["warps"],
        "us": rec["us"],
        "cosine": rec["cos"],
        "roofline_us": round(roofline_us(n, k), 2),
        "speedup_vs_incumbent": round(incumbent_us / rec["us"], 2),
        "term": win_label,
        "all_splits": {
            s: {kk: vv for kk, vv in r.items() if kk != "term"}
            for s, r in per_split.items()
        },
    }
    return winner, rejected


def _manifest_py(rows: list[dict], meta: dict) -> str:
    """Render manifest.py — generated, provenance-carrying."""
    today = date.today().isoformat()
    lines = [
        '"""The shipped routing table — GENERATED, do not hand-edit.',
        "",
        f"Regenerate with ``bench.drivers.probes.awq.gen_manifest`` "
        f"({meta['ckpt_name']}, layer {meta['layer']}, {today}).",
        "Each row records the term the search chose, the launch config,",
        "and the measured row that earned it the seat — gate was",
        "``vllm._custom_ops.awq_gemm`` on the checkpoint's own tensors,",
        "clock was ``timeit_graph`` (CUDA-graph referee).",
        "",
        "The kernels themselves are the ``compile-triton`` output,",
        "vendored verbatim — a found kernel is shipped as source, not",
        "regenerated at call time.  Operators sharing ``(k, split)``",
        "share a module: the term fixes k and split; n and the launch",
        "config are runtime arguments.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        '__all__ = ["ROUTES", "Route"]',
        "",
        "from dataclasses import dataclass",
        "",
        "",
        "@dataclass(frozen=True)",
        "class Route:",
        '    """One operator\'s shipped kernel row."""',
        "",
        "    module: str  # vendored kernels/ module name",
        "    k: int  # contraction dim the term was lowered for",
        "    n: int  # output dim — runtime arg, recorded for hosts",
        "    split: int  # split_k the search chose",
        "    block_n: int",
        "    block_k: int",
        "    warps: int",
        "    us: float  # measured on the referee, CUDA graph",
        "    cosine: float  # vs awq_gemm, inside the timing path",
        "    pct_of_roofline: float",
        "    speedup_vs_incumbent: float",
        '    provenance: str = ""  # ckpt / layer / shape-group note',
        "",
        "",
        "ROUTES: dict[str, Route] = {",
    ]
    for r in rows:
        pct = r["pct_of_roofline"]
        prov = r.get("provenance", "").replace('"', "'")
        lines.append(f'    "{r["operator"]}": Route(')
        lines.append(f'        module="{r["module"]}",')
        lines.append(f"        k={r['k']},")
        lines.append(f"        n={r['n']},")
        lines.append(f"        split={r['split']},")
        lines.append(f"        block_n={r['block_n']},")
        lines.append(f"        block_k={r['block_k']},")
        lines.append(f"        warps={r['warps']},")
        lines.append(f"        us={r['us']},")
        lines.append(f"        cosine={r['cosine']:.7f},")
        lines.append(f"        pct_of_roofline={pct:.1f},")
        lines.append(
            f"        speedup_vs_incumbent={r['speedup_vs_incumbent']},"
        )
        lines.append(f'        provenance="{prov}",')
        lines.append("    ),")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument(
        "--lib",
        type=Path,
        default=Path("/home/luis/GitHub/handl-lang/lib-triton-qgemv"),
        help="lib-triton-qgemv checkout to write kernels/ + manifest.py into",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/home/luis/GitHub/handl-lang/graphopt/docs/reports/"
            "p27_manifest_breadth.json"
        ),
    )
    ap.add_argument(
        "--ops",
        default="",
        help="comma-separated operator names to include (default: all)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="measure and report only — do not vendor kernels or "
        "rewrite manifest.py (use with --ops for smoke runs)",
    )
    ap.add_argument(
        "--verify-layers",
        type=int,
        nargs="*",
        default=[23],
        help="extra layers to gate the winning config on (correctness only)",
    )
    args = ap.parse_args()

    from vllm import _custom_ops as cops

    ops = enumerate_operators(args.ckpt, args.layer)
    if args.ops:
        wanted = set(args.ops.split(","))
        ops = [o for o in ops if o.name in wanted]
        if not args.dry_run:
            print(
                "refusing to rewrite the manifest from a partial "
                "--ops sweep — pass --dry-run for smoke runs",
                file=sys.stderr,
            )
            return 2
    if not ops:
        print("no quantized operators found", file=sys.stderr)
        return 1
    print(f"{len(ops)} operators at layer {args.layer}:")
    for op in ops:
        print(f"  {op.name:>16}  k={op.k:<5} n={op.n}")

    # group by shape — one term sweep per (k, n); every operator in the
    # group still gets its own gate + timed row on its own tensors
    groups: dict[tuple[int, int], list[OpSpec]] = {}
    for op in ops:
        groups.setdefault((op.k, op.n), []).append(op)

    report: dict = {
        "ckpt": args.ckpt,
        "layer": args.layer,
        "generated": date.today().isoformat(),
        "shapes": {},
        "operators": {},
    }
    manifest_rows: list[dict] = []
    needed_modules: dict[tuple[int, int], tuple[int, int]] = {}

    with safe_open(args.ckpt, "pt") as f:
        for (k, n), group in sorted(groups.items()):
            rep = group[0]
            qw = f.get_tensor(f"{rep.path}.qweight").cuda()
            qz = f.get_tensor(f"{rep.path}.qzeros").cuda()
            sc = f.get_tensor(f"{rep.path}.scales").cuda()
            qw_std, qz_std = (
                to_standard_order(qw),
                to_standard_order(qz),
            )
            x = torch.randn(1, k, device="cuda", dtype=torch.float16)

            def incumbent_fn(x=x, qw=qw, sc=sc, qz=qz):
                return cops.awq_gemm(x, qw, sc, qz, 1)

            incumbent_us = timeit_graph(
                incumbent_fn, calls=10, replays=3
            )

            winner, rejected = sweep_shape(
                k=k,
                n=n,
                qw_std=qw_std,
                qz_std=qz_std,
                sc=sc,
                x=x,
                incumbent_fn=incumbent_fn,
                incumbent_us=incumbent_us,
            )
            report["shapes"][f"k{k}n{n}"] = {
                "operators": [o.name for o in group],
                "winner": winner,
                "rejected": rejected,
                "incumbent_us": incumbent_us,
            }
            if winner is None:
                print(
                    f"!! k={k} n={n}: no candidate survived", flush=True
                )
                continue
            print(
                f"k={k:<5} n={n:<5} -> split{winner['split']} "
                f"bn{winner['block_n']} bk{winner['block_k']} "
                f"w{winner['warps']}: {winner['us']:.1f} us "
                f"({100 * winner['roofline_us'] / winner['us']:.0f}% "
                f"roof), {winner['speedup_vs_incumbent']}x vs awq_gemm",
                flush=True,
            )
            needed_modules[(k, winner["split"])] = (
                winner["block_n"],
                winner["block_k"],
            )

            # per-operator rows: gate + timed verify on the op's own
            # tensors (same shape => same bytes; values differ per layer)
            for op in group:
                qw_o = f.get_tensor(f"{op.path}.qweight").cuda()
                qz_o = f.get_tensor(f"{op.path}.qzeros").cuda()
                sc_o = f.get_tensor(f"{op.path}.scales").cuda()
                if op is rep:
                    qw_s, qz_s, sc_o2, x_o = (qw_std, qz_std, sc, x)
                else:
                    qw_s = to_standard_order(qw_o)
                    qz_s = to_standard_order(qz_o)
                    sc_o2 = sc_o
                    x_o = torch.randn(
                        1, k, device="cuda", dtype=torch.float16
                    )
                mod = cached_module(qgemv(k=k, split=winner["split"]))
                xf_o = x_o.reshape(-1)

                def run_o(
                    mod=mod,
                    xf_o=xf_o,
                    qw_s=qw_s,
                    qz_s=qz_s,
                    sc_o2=sc_o2,
                    n=n,
                    k=k,
                    w=winner,
                ):
                    return _run_with(
                        mod,
                        xf_o,
                        qw_s,
                        qz_s,
                        sc_o2,
                        n,
                        k,
                        w["block_n"],
                        w["block_k"],
                        w["warps"],
                    )

                def inc_o(x_o=x_o, qw_o=qw_o, sc_o=sc_o, qz_o=qz_o):
                    return cops.awq_gemm(x_o, qw_o, sc_o, qz_o, 1)

                inc_us_o = timeit_graph(inc_o, calls=10, replays=3)
                out_o = run_o()
                cos_o = torch.nn.functional.cosine_similarity(
                    out_o.detach().flatten().float(),
                    inc_o().detach().flatten().float(),
                    dim=0,
                ).item()
                us_o = timeit_graph(run_o, calls=10, replays=3)
                if us_o is None:
                    print(f"    {op.name}: unmeasurable — skipped")
                    continue
                roof = roofline_us(n, k)
                pct = 100.0 * roof / us_o if us_o else 0.0
                row = {
                    "operator": op.name,
                    "path": op.path,
                    "module": f"qgemv_k{k}_split{winner['split']}",
                    "k": k,
                    "n": n,
                    "split": winner["split"],
                    "block_n": winner["block_n"],
                    "block_k": winner["block_k"],
                    "warps": winner["warps"],
                    "us": round(us_o, 1),
                    "cosine": round(cos_o, 7),
                    "pct_of_roofline": round(pct, 1),
                    "speedup_vs_incumbent": round(inc_us_o / us_o, 2)
                    if inc_us_o
                    else -1.0,
                    "incumbent_us": round(inc_us_o, 1)
                    if inc_us_o
                    else None,
                    "provenance": (
                        f"{Path(args.ckpt).parent.name} layer {op.layer}; "
                        f"shape group k{k}n{n}"
                        + (
                            " (timed on own tensors)"
                            if op is not rep
                            else ""
                        )
                    ),
                }
                manifest_rows.append(row)
                report["operators"][op.name] = row
                print(
                    f"    {op.name:>16} n={n:<5} {us_o:6.1f} us "
                    f"cos {cos_o:.6f} "
                    f"{row['speedup_vs_incumbent']}x",
                    flush=True,
                )

        # correctness spot-check on other layers, winning config only
        for layer in args.verify_layers:
            if layer == args.layer:
                continue
            with safe_open(args.ckpt, "pt") as f2:
                for op0 in ops:
                    prefix = op0.path.replace(
                        f"model.layers.{args.layer}.",
                        f"model.layers.{layer}.",
                    )
                    try:
                        qw_l = f2.get_tensor(f"{prefix}.qweight").cuda()
                        qz_l = f2.get_tensor(f"{prefix}.qzeros").cuda()
                        sc_l = f2.get_tensor(f"{prefix}.scales").cuda()
                    except Exception:
                        continue
                    row = report["operators"].get(op0.name)
                    if row is None:
                        continue
                    qw_s = to_standard_order(qw_l)
                    qz_s = to_standard_order(qz_l)
                    x_l = torch.randn(
                        1, row["k"], device="cuda", dtype=torch.float16
                    )
                    mod = cached_module(
                        qgemv(k=row["k"], split=row["split"])
                    )
                    out_l = _run_with(
                        mod,
                        x_l.reshape(-1),
                        qw_s,
                        qz_s,
                        sc_l,
                        row["n"],
                        row["k"],
                        row["block_n"],
                        row["block_k"],
                        row["warps"],
                    )
                    inc_l = cops.awq_gemm(x_l, qw_l, sc_l, qz_l, 1)
                    cos_l = torch.nn.functional.cosine_similarity(
                        out_l.detach().flatten().float(),
                        inc_l.detach().flatten().float(),
                        dim=0,
                    ).item()
                    report["operators"][op0.name].setdefault(
                        "verify_cosine", {}
                    )[f"layer{layer}"] = round(cos_l, 7)

    # persist measurements before any artifact writes — a vendoring
    # failure must not lose the sweep
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    # vendor the winning modules that aren't already shipped
    if args.dry_run:
        report["vendored"] = "dry-run: skipped"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out} (dry-run: manifest untouched)")
        return 0

    kernels_dir = args.lib / "src" / "lib_triton_qgemv" / "kernels"
    vendored = []
    for (k, split), (bn, bk) in sorted(needed_modules.items()):
        name = f"qgemv_k{k}_split{split}"
        dest = kernels_dir / f"{name}.py"
        if dest.exists():
            vendored.append({"module": name, "action": "exists"})
            continue
        src = compile_source(
            qgemv(k=k, split=split), block_n=bn, block_k=bk
        )
        dest.write_text(src)
        vendored.append({"module": name, "action": "vendored"})
        print(f"vendored {dest}")
    report["vendored"] = vendored

    # rewrite the manifest from the measured rows
    ckpt_name = Path(args.ckpt).parent.name
    manifest_rows.sort(key=lambda r: r["operator"])
    text = _manifest_py(
        manifest_rows, {"ckpt_name": ckpt_name, "layer": args.layer}
    )
    manifest_path = (
        args.lib / "src" / "lib_triton_qgemv" / "manifest.py"
    )
    manifest_path.write_text(text)
    # keep the generated file inside the repo's format gate when ruff
    # is reachable; the emit is already close, this is belt-and-braces
    import shutil
    import subprocess

    ruff = shutil.which("ruff")
    if ruff:
        subprocess.run(
            [ruff, "format", str(manifest_path)], check=False
        )
    print(f"wrote {manifest_path}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    ok = all(r.get("cosine", 0) >= 0.9999 for r in manifest_rows)
    print(f"all rows gated: {ok} ({len(manifest_rows)} operators)")
    return 0 if ok and manifest_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

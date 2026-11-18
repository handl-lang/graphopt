"""Emit the arch-zoo .hnl artifact manifest.

The wave-2 arch-zoo sweep exported ``arch-<spec>.hnl`` programs plus
safetensors weight sidecars into an ephemeral directory
(``/home/luis/tmp/arch-zoo``, ~8.4 GB).  The binaries cannot live in
git; what *can* is a manifest that pins each artifact by name, size,
and hash so a re-export or a relocated store can be checked against it.

This script inventories a zoo directory (the ``.hnl`` + sidecar pairs,
the ``zoo-inputs-*``/``*.verify-inputs`` safetensors feeds), joins it
against the sweep's ``summary.tsv``/``sweep-status.tsv`` and the
``--verify-run`` verdicts, folds in the paired speed-bench results
(``result-<spec>.json`` from the zoo bench), and writes
``docs/reports/hnl_artifact_manifest.json``.

Hashing policy (documented in the manifest itself):

- ``.hnl`` programs are small (≤ ~7 MB) — full SHA-256.
- safetensors files ≤ ``FULL_HASH_LIMIT`` (64 MiB) — full SHA-256
  (covers every ``zoo-inputs-*`` feed).
- Larger sidecars (up to 4.4 GB) — size + SHA-256 of the first and
  last 1 MiB only; a full pass over 8 GB of weights buys nothing the
  paired verify-run verdict doesn't already cover.

Usage:
    .venv/bin/python scripts/zoo_artifact_manifest.py \
        [--zoo-dir DIR] [--bench-dir DIR] [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

FULL_HASH_LIMIT = 64 * 1024 * 1024
PARTIAL_CHUNK = 1024 * 1024

# Specs the speed bench deliberately skipped or never finished — the
# JSON arm table only exists where a result-<spec>.json was written.
BENCH_STATUS = {
    "whisper-large-v3-turbo-encoder": (
        "skipped",
        "2.5 GB sidecar, verify_run=False — memory gates",
    ),
    "tinyllama-1.1b": (
        "skipped",
        "4.4 GB sidecar, verify_run=False — memory gates",
    ),
    "whisper-small-encoder": (
        "pending",
        "bench run did not reach this spec",
    ),
    "whisper-base-encoder": (
        "partial",
        "40-min timeout killed the run at round 3/5; "
        "hnl n=3 paired rounds, JSON not written",
    ),
}


def sha256_file(path: Path) -> str:
    """Full-file SHA-256."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_head_tail(path: Path, size: int) -> dict:
    """SHA-256 of the first and last ``PARTIAL_CHUNK`` bytes."""
    head = hashlib.sha256()
    with path.open("rb") as f:
        head.update(f.read(PARTIAL_CHUNK))
    tail = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(max(0, size - PARTIAL_CHUNK))
        tail.update(f.read())
    return {
        "sha256_head_1mib": head.hexdigest(),
        "sha256_tail_1mib": tail.hexdigest(),
    }


def hash_entry(path: Path) -> dict:
    """Size + hash record for one artifact file."""
    size = path.stat().st_size
    entry = {"bytes": size}
    if size <= FULL_HASH_LIMIT:
        entry["sha256"] = sha256_file(path)
    else:
        entry.update(sha256_head_tail(path, size))
        entry["hash"] = "head_tail_1mib"
    return entry


def read_tsv(path: Path) -> list[dict]:
    """Read a headered TSV into a list of dicts."""
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_tsv_fields(path: Path, fields: list[str]) -> list[dict]:
    """Read a headerless TSV using explicit column names."""
    if not path.exists():
        return []
    with path.open() as f:
        return list(
            csv.DictReader(f, delimiter="\t", fieldnames=fields)
        )


def bench_entry(bench_dir: Path, spec: str) -> dict | None:
    """Fold one ``result-<spec>.json`` into a compact arm table."""
    path = bench_dir / f"result-{spec}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    arms = {}
    for name, arm in data.get("arms", {}).items():
        arms[name] = {
            "min_of_mins_ms": arm.get("min_of_mins"),
            "med_of_meds_ms": arm.get("med_of_meds"),
        }
    hnl = arms.get("hnl", {}).get("med_of_meds_ms")
    ref = arms.get("torch_ir", {}).get("med_of_meds_ms")
    ratio = hnl / ref if hnl and ref else None
    return {
        "result_json": str(path),
        "rounds": data.get("rounds"),
        "iters": data.get("iters"),
        "wall_s": data.get("wall_s"),
        "arms_ms": arms,
        "hnl_over_torch_ir": ratio,
    }


def main() -> int:
    """Build the manifest and write it."""
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--zoo-dir",
        type=Path,
        default=Path("/home/luis/tmp/arch-zoo"),
    )
    ap.add_argument(
        "--bench-dir",
        type=Path,
        default=Path("/home/luis/tmp/zoo_bench"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=repo / "docs/reports/hnl_artifact_manifest.json",
    )
    args = ap.parse_args()

    zoo = args.zoo_dir
    summary = {r["spec"]: r for r in read_tsv(zoo / "summary.tsv")}
    sweep = {
        r["spec"]: r
        for r in read_tsv_fields(
            zoo / "sweep-status.tsv",
            [
                "spec",
                "rc",
                "wall_s",
                "hnl_bytes",
                "sidecar_bytes",
                "log",
            ],
        )
    }

    specs = []
    for hnl in sorted(zoo.glob("arch-*.hnl")):
        spec = hnl.name[len("arch-") : -len(".hnl")]
        sidecar = zoo / f"arch-{spec}.safetensors"
        srow = summary.get(spec, {})
        entry = {
            "spec": spec,
            "export_rc": int(
                srow.get("export_rc", sweep.get(spec, {}).get("rc", -1))
            ),
            "export_wall_s": int(
                srow.get(
                    "wall_s", sweep.get(spec, {}).get("wall_s", -1)
                )
            ),
            "search_s": float(srow.get("search_s", "nan")),
            "lower_s": float(srow.get("lower_s", "nan")),
            "lower_verify": srow.get("lower_verify") == "True",
            "hnl_check": srow.get("hnl_check") == "True",
            "verify_run": srow.get("verify_run") == "True",
            "verify_max_abs": srow.get("verify_max_abs") or None,
            "verify_values": int(srow.get("verify_values", "0") or 0),
            "artifacts": {
                "hnl": {"path": str(hnl), **hash_entry(hnl)},
            },
        }
        if sidecar.exists():
            entry["artifacts"]["sidecar"] = {
                "path": str(sidecar),
                **hash_entry(sidecar),
            }
        bench = bench_entry(args.bench_dir, spec)
        status, note = BENCH_STATUS.get(spec, ("benchmarked", ""))
        entry["speed_bench"] = {
            "status": status,
            "note": note,
            "result": bench,
        }
        specs.append(entry)

    extras = []
    sidecars = {
        s["artifacts"].get("sidecar", {}).get("path") for s in specs
    }
    for path in sorted(zoo.glob("*.safetensors")):
        if str(path) in sidecars:
            continue
        extras.append(
            {"path": str(path), "role": path.name, **hash_entry(path)}
        )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generator": "scripts/zoo_artifact_manifest.py",
        "zoo_dir": str(zoo),
        "note": (
            "Binaries live outside git (ephemeral /tmp store). "
            "This manifest pins name+size+hash for re-verification. "
            "Sidecars >64 MiB carry head/tail 1 MiB SHA-256, not full."
        ),
        "hash_policy": {
            "full_sha256_max_bytes": FULL_HASH_LIMIT,
            "partial": f"sha256 of first+last {PARTIAL_CHUNK} bytes",
        },
        "specs": specs,
        "auxiliary_inputs": extras,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {args.out} ({len(specs)} specs, {len(extras)} aux inputs)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

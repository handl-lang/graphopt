#!/usr/bin/env python3
"""Regenerate ``mlopt_host_tensor/host_surface.json``.

Reads the host-tensor checkout *read-only* — the handwritten
``op "tensor.X"`` decls in ``src/index.hnl`` plus
``aten_port_gen --emit-spec`` (build ``target/debug/aten_port_gen``
first if absent: ``cargo build --bin aten_port_gen`` in host-tensor).

Usage::

    .venv/bin/python scripts/dump_host_surface.py [--check]

``HOST_TENSOR_DIR`` overrides the sibling-checkout default.  ``--check``
exits non-zero when the artifact is stale — the drift gate the
``test_host_surface`` suite runs when the checkout is present.
"""

from __future__ import annotations

import argparse
import sys

from mlopt_host_tensor.surface import (
    collect_host_surface,
    render_surface,
    surface_json_path,
)


def main() -> int:
    """Collect the surface; write or check the artifact."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare against the artifact without writing",
    )
    ap.add_argument(
        "--host-tensor-dir",
        default=None,
        help="host-tensor checkout (default: $HOST_TENSOR_DIR or sibling)",
    )
    args = ap.parse_args()

    ops = collect_host_surface(args.host_tensor_dir)
    text = render_surface(ops)
    path = surface_json_path()

    if args.check:
        current = path.read_text() if path.is_file() else ""
        if current != text:
            print(
                f"{path} is stale — run scripts/dump_host_surface.py",
                file=sys.stderr,
            )
            return 1
        print(f"{path} is fresh ({len(ops)} ops)")
        return 0

    path.write_text(text)
    print(f"wrote {path} ({len(ops)} ops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

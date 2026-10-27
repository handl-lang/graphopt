"""pip-audit the resolution, not the mutated CI environment.

`pip-audit` with no arguments audits whatever is installed.  Three
kinds of distribution in this environment are un-auditable, and each
one makes pip-audit exit non-zero before it reports an advisory:

* the workspace members, installed editable;
* the `mlopt-core` / `mlopt-orchestrator` git deps (not on PyPI);
* `torch`, installed with a `+cpu` local version label.

That is why CI sat red on a clean lock.  This script audits the
*resolution* instead: `uv export` the lock, drop the first-party
entries, strip local version labels, and audit the rest.  A real
advisory still fails the build; a distribution pip-audit cannot
resolve is reported as a skip, because dropping it silently would be
worse than naming it.  First-party code is covered by bandit,
semgrep, vulture and `ty`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

#: A local version label (`2.14.1+cpu`); PyPI only knows the base.
LOCAL_LABEL = re.compile(r"^(?P<base>[^=]+==[^-+]+)\+[^-]+$")

#: Advisories that cannot be remediated from this lock, with the reason.
#: Both are pinned by `semgrep` (a dev-only gate tool, not a shipped
#: package), so the vulnerable code never reaches the product; the pin
#: lifts when semgrep does.  Re-check at the next semgrep bump.
ACCEPTED: dict[str, str] = {
    "CVE-2026-59951": "mcp 1.29.0 — semgrep's pin; dev-only",
    "CVE-2026-114603": "opentelemetry-api 1.37.0 — semgrep's pin; dev-only",
}


def export_lock() -> str:
    """Return the exported requirements for the dependency closure."""
    proc = subprocess.run(
        [
            "uv",
            "export",
            "--no-emit-project",
            "--no-hashes",
            "--format",
            "requirements-txt",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def auditable(lines: list[str]) -> list[str]:
    """Drop first-party and direct-URL lines; normalise local tags."""
    kept: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-e ", "--")):
            continue
        if " @ " in line:
            continue
        kept.add(LOCAL_LABEL.sub(r"\g<base>", line))
    return sorted(kept)


def _pip_audit() -> str:
    """Return the `pip-audit` beside the running interpreter."""
    return str(Path(sys.executable).with_name("pip-audit"))


def main(argv: list[str]) -> int:
    """Audit the exported resolution; fail only on real advisories."""
    requirements = auditable(export_lock().splitlines())
    fd, name = tempfile.mkstemp(suffix=".txt")
    path = Path(name)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(requirements) + "\n")
        proc = subprocess.run(
            [
                _pip_audit(),
                "-r",
                str(path),
                "-f",
                "json",
                *(f"--ignore-vuln={v}" for v in ACCEPTED),
                *argv,
            ],
            capture_output=True,
            text=True,
        )
    finally:
        path.unlink(missing_ok=True)

    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        # pip-audit itself failed: network, a bad requirement, a missing
        # advisory database.  That is a gate failure, not a skip.
        sys.stderr.write(proc.stdout + proc.stderr)
        return proc.returncode or 1

    deps = report.get("dependencies", [])
    vulns = [d for d in deps if d.get("vulns")]
    skipped = [d for d in deps if d.get("skip_reason")]
    print(
        f"pip-audit: {len(deps)} distributions, "
        f"{len(vulns)} vulnerable, {len(skipped)} skipped"
    )
    for dep in skipped:
        print(
            f"  skip {dep['name']} {dep['version']}: "
            f"{dep['skip_reason']}"
        )
    for vuln_id, why in ACCEPTED.items():
        print(f"  accept {vuln_id}: {why}")
    for dep in vulns:
        for vuln in dep["vulns"]:
            summary = (vuln.get("description") or "").split("\n")[0]
            print(
                f"  VULN {dep['name']} {dep['version']}: "
                f"{vuln['id']} {summary[:110]}"
            )
    return 1 if vulns else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env bash
# dev-link.sh — regenerate the `[patch."https://github.com/handl-lang/*"]`
# tables in a repo's root Cargo.toml from the sibling checkouts
# (ADR-0128 §Resolution: git deps + `[patch]`, local-only).
#
#   scripts/dev-link.sh            # patch this repo (default: cwd)
#   scripts/dev-link.sh ../host-cli     # patch a specific sibling checkout
#
# The generated tables replace the region between
#
#   # dev-link: begin
#   # dev-link: end
#
# in <repo>/Cargo.toml (markers are created on first run, appended at EOF).
# For every sibling repo R referenced by a `git = "…/handl-lang/R"` dep in
# the target's manifests, each crate of R is patched to its sibling path:
# the workspace members under `R/{crates,modules}/`, plus R's own root
# `[package]` when it is a single-crate repo (the ADR-0128 `lib-*` ports).
# CI strips/ignores the tables and resolves the pinned revs.
set -euo pipefail

REPO="${1:-.}"
ORG="$(cd "$REPO/.." && pwd)"
TARGET="$ORG/$(basename "$(cd "$REPO" && pwd)")"

python3 - "$TARGET" "$ORG" <<'PY'
import re, sys, pathlib

target = pathlib.Path(sys.argv[1])
org = pathlib.Path(sys.argv[2])
root_toml = target / "Cargo.toml"
base = "https://github.com/handl-lang/"
BEGIN = "# dev-link: begin"
END = "# dev-link: end"

# Repos this repo references via git deps (any Cargo.toml, any depth) —
# plus one hop through the referenced repos' own manifests: patched-in
# sibling crates git-dep further repos (e.g. host's own
# `[patch."…/host"]` table exists because kit-canonical's manifest
# git-deps host crates).
pat = re.compile(r'git = "' + re.escape(base) + r'([A-Za-z0-9._-]+)"')

def refs_of(d: pathlib.Path):
    out = set()
    for toml in d.rglob("Cargo.toml"):
        if "target" in toml.parts:
            continue
        out.update(pat.findall(toml.read_text()))
    return out

refd = refs_of(target)
for repo in list(refd):
    d = org / repo
    if d.is_dir():
        refd.update(refs_of(d))

# Map repo -> [(package name, sibling relative path)] for referenced repos.
#
# Two layouts are supported: a workspace of `crates/*` + `modules/*`
# members, and the single-crate-at-root layout (`lib-*` ports after the
# ADR-0128 split), which has no `crates/`/`modules/` directory at all.
def packages(repo_dir: pathlib.Path):
    out = []
    root_toml = repo_dir / "Cargo.toml"
    if root_toml.is_file():
        text = root_toml.read_text()
        m = re.search(r'^name = "([^"]+)"', text, re.M)
        if m and re.search(r"^\[package\]", text, re.M):
            out.append((m.group(1), repo_dir.relative_to(org).as_posix()))
    for group in ("crates", "modules"):
        g = repo_dir / group
        if not g.is_dir():
            continue
        for toml in sorted(g.glob("*/Cargo.toml")):
            m = re.search(r'^name = "([^"]+)"', toml.read_text(), re.M)
            if m:
                out.append((m.group(1), toml.parent.relative_to(org).as_posix()))
    return out

blocks = []
missing = []
for repo in sorted(refd):
    d = org / repo
    if not d.is_dir():
        missing.append(repo)
        continue
    lines = [f'[patch."{base}{repo}"]']
    for name, rel in packages(d):
        lines.append(f"{name} = {{ path = \"../{rel}\" }}")
    blocks.append("\n".join(lines))

for repo in missing:
    print(f"dev-link: warning: referenced repo ../{repo} not checked out", file=sys.stderr)

text = root_toml.read_text()
region = "\n".join([BEGIN] + blocks + [END])
if BEGIN in text and END in text:
    text = re.sub(
        re.escape(BEGIN) + r".*?" + re.escape(END),
        lambda _: region,
        text,
        count=1,
        flags=re.S,
    )
else:
    text = text.rstrip() + "\n\n" + region + "\n"
root_toml.write_text(text)
print(f"dev-link: wrote {len(blocks)} patch table(s) to {root_toml}")
PY

#!/usr/bin/env bash
# Public-closure gate — fail if any handl-lang git dependency source
# reachable from this repo's lockfiles resolves to a non-public repo.
#
# A repo must be public iff it sits in the dep closure of a public repo —
# visibility is derived from the edge graph, and this script is the check
# that keeps that derived set honest. Run after touching Cargo.toml deps:
#
#   scripts/check_public_closure.sh
#
# Checks every `github.com/handl-lang/<repo>` source recorded in the
# workspace and standalone lockfiles via `gh repo view` (visibility field —
# authenticated; a private repo is reported as "private").

set -euo pipefail
cd "$(dirname "$0")/.."

mapfile -t repos < <(
    grep -oh 'github\.com/handl-lang/[a-zA-Z0-9_-]*' Cargo.lock packages/*/Cargo.lock 2>/dev/null \
        | sed 's|.*handl-lang/||' | sort -u
)

fail=0
for r in "${repos[@]}"; do
    vis=$(gh repo view "handl-lang/$r" --json visibility --jq .visibility 2>/dev/null || echo "MISSING")
    if [ "$vis" != "PUBLIC" ]; then
        echo "NOT-PUBLIC: handl-lang/$r ($vis)"
        fail=1
    fi
done

if [ "$fail" -eq 0 ]; then
    echo "closure fully public: ${#repos[@]} repos"
else
    echo "closure has private members — external cargo builds will fail" >&2
    exit 1
fi

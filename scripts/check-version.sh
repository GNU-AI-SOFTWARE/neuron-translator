#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# check-version.sh — CI gate of the project versioning system
# (docs/VERSIONING.md):
#   1. VERSION exists at the repo root and is strict semver X.Y.Z
#   2. if CHANGELOG.md exists, its newest released ## [X.Y.Z] section
#      must equal VERSION
#   3. if any v* git tag exists, the newest one must equal VERSION
set -u
cd "$(dirname "$0")/.."
fail() { echo "check-version: FAIL: $*" >&2; exit 1; }
[ -f VERSION ] || fail "no VERSION file at the repo root"
V=$(cat VERSION)
echo "$V" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' || fail "VERSION '$V' is not semver X.Y.Z"
if [ -f CHANGELOG.md ]; then
  C=$(grep -m1 -E '^## \[[0-9]+\.[0-9]+\.[0-9]+\]' CHANGELOG.md | sed -E 's/^## \[([^]]+)\].*/\1/')
  if [ -n "$C" ] && [ "$C" != "$V" ]; then
    fail "CHANGELOG newest release '$C' != VERSION '$V' (cut the release with scripts/release.sh)"
  fi
fi
if git rev-parse -q --is-inside-work-tree >/dev/null 2>&1 && git tag -l 'v*' | grep -q .; then
  T=$(git tag -l 'v*' | sort -V | tail -1 | sed 's/^v//')
  [ "$T" = "$V" ] || fail "newest git tag v$T != VERSION $V (tag with scripts/release.sh)"
fi
echo "check-version: OK $V"

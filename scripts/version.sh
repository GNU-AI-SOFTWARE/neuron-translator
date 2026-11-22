#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# version.sh — query the project version (see docs/VERSIONING.md)
#   scripts/version.sh           print the current VERSION
#   scripts/version.sh --next    print the next iterative version (patch + 1)
#   scripts/version.sh --check   run the CI gate (scripts/check-version.sh)
set -eu
cd "$(dirname "$0")/.."
[ -f VERSION ] || { echo "version: no VERSION file at the repo root" >&2; exit 1; }
V=$(cat VERSION)
case ${1:-} in
  "") echo "$V" ;;
  --next|-n)
    maj=${V%%.*}; rest=${V#*.}; min=${rest%%.*}; pat=${rest#*.}
    echo "$maj.$min.$((pat+1))"
    ;;
  --check|-c) exec sh scripts/check-version.sh ;;
  *) echo "usage: $0 [--next|--check]" >&2; exit 2 ;;
esac

#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# release.sh — cut a release: bump VERSION, stamp CHANGELOG.md, commit,
# tag vX.Y.Z (immutable). Whole-org stack release with --all.
# 'iterate' = iterative versioning: one automatic patch bump per
# iteration (0.1.0 -> 0.1.1 -> 0.1.2 ...).
# Usage:
#   scripts/release.sh patch|minor|major|iterate|X.Y.Z
#   scripts/release.sh X.Y.Z --all /path/to/org-checkouts
set -eu
usage() { echo "usage: $0 patch|minor|major|iterate|X.Y.Z [--all DIR]" >&2; exit 2; }
[ $# -ge 1 ] || usage
KIND=$1
ALL=""
if [ "${2:-}" = "--all" ]; then ALL=$3; [ -n "$ALL" ] || usage; fi

bump() {  # $1=current X.Y.Z  $2=kind -> prints the new version
  old=$1; k=$2
  maj=${old%%.*}; rest=${old#*.}; min=${rest%%.*}; pat=${rest#*.}
  case $k in
    major) echo "$((maj+1)).0.0" ;;
    minor) echo "$maj.$((min+1)).0" ;;
    patch|iterate) echo "$maj.$min.$((pat+1))" ;;
    *) echo "$k" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' && echo "$k" \
         || { echo "release: bad version/kind '$k'" >&2; exit 2; } ;;
  esac
}

release_one() {
  d=$1
  cd "$d"
  [ -f VERSION ] || { echo "release: skip $d (no VERSION file)"; return 0; }
  V=$(cat VERSION)
  NEW=$(bump "$V" "$KIND")
  [ "$NEW" = "$V" ] && { echo "release: $d already at $V"; return 0; }
  echo "$NEW" > VERSION
  if [ -f CHANGELOG.md ]; then
    today=$(date +%F)
    awk -v hdr="## [$NEW] — $today" '
      /^## \[[0-9]+\.[0-9]+\.[0-9]+\]/ { rel=1 }
      /^## \[[0-9]+\.[0-9]+\.[0-9]+\]/ && !done { print hdr; print ""; done=1 }
      !done && !rel && /^## \[Unreleased\]/ { print hdr; print ""; done=1 }
      { print }
      END { if (!done) { print ""; print hdr } }
    ' CHANGELOG.md > CHANGELOG.md.new && mv CHANGELOG.md.new CHANGELOG.md
  fi
  git add VERSION CHANGELOG.md 2>/dev/null || git add VERSION
  git -c commit.gpgsign=false commit -qm "release: v$NEW" || true
  git tag -a "v$NEW" -m "v$NEW" 2>/dev/null || echo "release: tag v$NEW already exists in $d"
  echo "release: $d $V -> v$NEW  (push with: git push origin main v$NEW)"
}

if [ -n "$ALL" ]; then
  for d in "$ALL"/*/; do release_one "$d"; done
else
  release_one "$(pwd)"
fi

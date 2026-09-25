#!/usr/bin/env bash
# Fetch the pinned NVD snapshot (fkie-cad/nvd-json-data-feeds) used by the paper.
#
#   bash code/cve/fetch_nvd.sh [TARGET_DIR]        # default: data/cve/nvd
#
# The commit SHA is read from data/cve/SNAPSHOT.txt. If that file does not exist
# yet, the current HEAD of the mirror is resolved and recorded there (this is how
# the paper's snapshot was pinned). All CVE-YYYY directories are checked out,
# because the analysis bins CVEs by *publish date*, and IDs reserved in earlier
# years are routinely published later.
set -euo pipefail

REPO_URL="https://github.com/fkie-cad/nvd-json-data-feeds.git"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SNAP="$ROOT/data/cve/SNAPSHOT.txt"
TARGET="${1:-$ROOT/data/cve/nvd}"

if [[ ! -f "$SNAP" ]]; then
    sha="$(git ls-remote "$REPO_URL" HEAD | cut -f1)"
    {
        echo "repo=$REPO_URL"
        echo "commit=$sha"
        echo "retrieved=$(date -u +%Y-%m-%d)"
    } > "$SNAP"
    echo "Pinned new snapshot $sha -> $SNAP"
fi
SHA="$(grep '^commit=' "$SNAP" | cut -d= -f2)"

if [[ -d "$TARGET/.git" ]] && [[ "$(git -C "$TARGET" rev-parse HEAD)" == "$SHA" ]]; then
    echo "NVD snapshot $SHA already present at $TARGET"
    exit 0
fi

mkdir -p "$TARGET"
git -C "$TARGET" init -q
git -C "$TARGET" remote add origin "$REPO_URL" 2>/dev/null || true
git -C "$TARGET" fetch --depth 1 origin "$SHA"
git -C "$TARGET" -c advice.detachedHead=false checkout -q FETCH_HEAD

# Record the commit date of the snapshot (the data cut-off) once.
if ! grep -q '^commit_date=' "$SNAP"; then
    echo "commit_date=$(git -C "$TARGET" log -1 --format=%cI)" >> "$SNAP"
fi
echo "Checked out $SHA into $TARGET"
cat "$SNAP"

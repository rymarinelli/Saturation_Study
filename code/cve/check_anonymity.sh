#!/usr/bin/env bash
# Fail if any tracked file (or, with --export DIR, an exported tree) contains an
# author-identifying string. Run before syncing the anonymous artifact mirror.
#
#   bash code/cve/check_anonymity.sh              # scan tracked files at HEAD
#   bash code/cve/check_anonymity.sh --export DIR # scan an exported tree
#
# Base64 image payloads inside notebooks are skipped (random matches). Git commit
# metadata is not part of the anonymous mirror, but is reported for information.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATTERN='rymarinelli|marinelli|kopp|cedric|mail\.cedrickopp|chetwyn|adepoju|sara ?hobe|uio\.no|university of oslo|mediafutures|frankfurt ai safety'

if [[ "${1:-}" == "--export" ]]; then
    hits="$(grep -rIniE "$PATTERN" "$2" | grep -v '"image/png"' | grep -vE '^[^:]+:[0-9]+:\s*"[A-Za-z0-9+/=]{40,}' || true)"
else
    cd "$ROOT"
    hits="$(git grep -IniE "$PATTERN" HEAD -- . | grep -v '"image/png"' | grep -vE ':[0-9]+:\s*"[A-Za-z0-9+/=]{40,}' || true)"
fi

if [[ -n "$hits" ]]; then
    echo "Identifying strings found:"
    echo "$hits" | cut -c1-200
    exit 1
fi
echo "anonymity: no identifying strings in files."
if [[ "${1:-}" != "--export" ]]; then
    echo "note: commit authors in history (not shown by the anonymous mirror):"
    git -C "$ROOT" log --format='  %an' | sort -u
fi

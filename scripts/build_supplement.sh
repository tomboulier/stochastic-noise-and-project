#!/usr/bin/env bash
# Build the anonymized code archive of the supplementary material.
#
#     scripts/build_supplement.sh [commit]        # -> dist/snap_code.zip  (default: HEAD)
#
# The archive holds the files tracked by git at that commit, without the git history (which carries
# the authors' names) and without the paths marked `export-ignore` in .gitattributes. Weights and
# data are not tracked, hence not in the archive.
#
# The script then fails if the archive contains an e-mail address, a home-directory path, or a match
# of the extended regular expression $ANONYMITY_PATTERNS (authors, institutions, user names: e.g.
# ANONYMITY_PATTERNS='smith|example-university'). That expression is read from the environment so
# that the names never enter the repository.
set -euo pipefail

commit="${1:-HEAD}"
root="$(git rev-parse --show-toplevel)"
out="$root/dist/snap_code.zip"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$root/dist"
rm -f "$out"
git -C "$root" archive --format=zip --prefix=snap_code/ -o "$out" "$commit"
unzip -q "$out" -d "$tmp"

status=0
check() {  # check <label> <grep options> <pattern>
    if grep -rIn "$2" -e "$3" "$tmp"; then
        echo "error: $1 found in the archive (see above)" >&2
        status=1
    fi
}
check "an e-mail address" -E '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]'
check "a home-directory path" -E '(/home/|/Users/|C:\\Users\\)[A-Za-z0-9._-]+'
if [ -n "${ANONYMITY_PATTERNS:-}" ]; then
    check "an identifying name" -iE "$ANONYMITY_PATTERNS"
else
    echo "warning: ANONYMITY_PATTERNS is not set, names were not checked" >&2
fi
[ "$status" -eq 0 ] || { rm -f "$out"; exit 1; }

echo "$out ($(du -h "$out" | cut -f1), $(unzip -Z1 "$out" | grep -vc '/$') files)"

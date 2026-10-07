#!/usr/bin/env bash
# Build the versioned documentation site for https://docs.isaacnet.zifanzhang.com/.
#
#   scripts/build_docs.sh [OUT_DIR]          # default OUT_DIR: site
#
# Layout of OUT_DIR:
#   /               latest docs, built from the current checkout (main in CI)
#   /<version>/     the docs at tag v<version>, for each version in DOCS_VERSIONS
#   /versions.json  read by the mkdocs-material version selector (provider: mike)
#   /_redirects     Cloudflare Pages rules: /latest/* -> /*
#
# Every build runs `mkdocs build --strict` through uvx, so nothing needs to be installed
# besides uv and git. Add a released version to DOCS_VERSIONS (newest first) at release.
set -euo pipefail

DOCS_VERSIONS="${DOCS_VERSIONS:-0.2.0}"
SITE_URL="${SITE_URL:-https://docs.isaacnet.zifanzhang.com/}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-site}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

MKDOCS=(uvx --from 'mkdocs>=1.6,<2' --with 'mkdocs-material>=9.5'
        --with 'mkdocstrings[python]>=0.26' --with 'mkdocs-jupyter>=0.25' mkdocs)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# mkdocs build empties OUT_DIR first (clean build), so stale versions do not survive.
echo "==> latest (current checkout) -> $OUT"
(cd "$ROOT" && "${MKDOCS[@]}" build --strict -d "$OUT")

for v in $DOCS_VERSIONS; do
  tag="v$v"
  echo "==> $v (tag $tag) -> $OUT/$v"
  src="$TMP/$v"
  mkdir -p "$src"
  git -C "$ROOT" archive "$tag" | tar -x -C "$src"
  # Older tags carry their own mkdocs.yml; override only the site URL and the version selector.
  cat > "$src/mkdocs.versioned.yml" <<YML
INHERIT: mkdocs.yml
site_url: ${SITE_URL}${v}/
extra:
  version:
    provider: mike
YML
  (cd "$src" && "${MKDOCS[@]}" build --strict -f mkdocs.versioned.yml -d "$OUT/$v")
done

echo "==> versions.json, _redirects"
{
  printf '[{"version": "latest", "title": "latest (main)", "aliases": []}'
  for v in $DOCS_VERSIONS; do
    printf ', {"version": "%s", "title": "%s", "aliases": []}' "$v" "$v"
  done
  printf ']\n'
} > "$OUT/versions.json"

cat > "$OUT/_redirects" <<'RED'
/latest / 301
/latest/* /:splat 301
RED

echo "Built $OUT"

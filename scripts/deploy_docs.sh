#!/usr/bin/env bash
# Deploy a built docs site (scripts/build_docs.sh) to the Cloudflare Pages project isaacnet-docs,
# served at https://docs.isaacnet.zifanzhang.com/.
#
#   scripts/deploy_docs.sh [SITE_DIR]        # default SITE_DIR: site
#
# Authentication: CLOUDFLARE_API_TOKEN (+ CLOUDFLARE_ACCOUNT_ID) in the environment, or a
# `wrangler login` session. Never `wrangler deploy` / `wrangler init`: those create Workers.
set -euo pipefail

SITE="${1:-site}"
PROJECT="${DOCS_PAGES_PROJECT:-isaacnet-docs}"

if [ ! -f "$SITE/index.html" ] || [ ! -f "$SITE/versions.json" ]; then
  echo "error: $SITE does not look like a build of scripts/build_docs.sh" >&2
  exit 1
fi

WRANGLER_SEND_METRICS=false npx --yes wrangler@4 pages deploy "$SITE" \
  --project-name "$PROJECT" --branch main --commit-dirty=true

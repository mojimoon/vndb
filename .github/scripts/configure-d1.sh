#!/usr/bin/env bash
# Writes the D1 database id into web/wrangler.jsonc for CI.
# Takes it from the D1_DATABASE_ID environment variable (a GitHub repository
# variable or secret); a real id already committed in the file also works.
set -euo pipefail
cfg="web/wrangler.jsonc"
placeholder="00000000-0000-0000-0000-000000000000"
id="${D1_DATABASE_ID:-}"
if [ -n "$id" ]; then
  if ! [[ "$id" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
    echo "::error::D1_DATABASE_ID does not look like a database id (expected a UUID such as 1b2c3d4e-...)."
    exit 1
  fi
  sed -i "s/\"database_id\": \"[^\"]*\"/\"database_id\": \"$id\"/" "$cfg"
fi
if grep -q "$placeholder" "$cfg"; then
  echo "::error::No D1 database id. Run 'npx wrangler d1 create vndb' (or 'npx wrangler d1 list'), then add the id as a repository variable named D1_DATABASE_ID (Settings > Secrets and variables > Actions > Variables)."
  exit 1
fi
grep -o '"database_id": "[^"]*"' "$cfg"

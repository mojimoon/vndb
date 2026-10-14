#!/usr/bin/env bash
# Fetch the rank history from D1 for the pipeline. Run from web/ (wrangler).
#
# Writes $OUT/history.json (all history_day rows), or, once, before history_day
# exists, $OUT/legacy-history.json (the old per-VN column) to migrate from.
# Any other failure stops the refresh: a snapshot built on a missing history
# would reset every title's chart, so no data beats wrong data.
set -euo pipefail
OUT=${1:?usage: fetch-history.sh OUT_DIR}
ERR=$(mktemp)

# q SQL -> JSON on stdout; retries transient failures, not a missing table.
q() {
  local attempt
  for attempt in 1 2 3; do
    if npx wrangler d1 execute vndb --remote --json --command "$1" 2>"$ERR" >"$OUT/q.json"; then
      return 0
    fi
    if grep -qs "no such table" "$ERR" "$OUT/q.json"; then return 2; fi
    echo "attempt $attempt failed:" >&2; cat "$ERR" "$OUT/q.json" >&2
    sleep $((attempt * 10))
  done
  return 1
}

status=0
q "SELECT day, COUNT(*) AS parts FROM history_day GROUP BY day ORDER BY day" || status=$?
if [ "$status" = 2 ]; then
  echo "history_day does not exist yet: migrating from vn.history"
  status=0
  q "SELECT id, history FROM vn" || status=$?
  if [ "$status" = 2 ]; then echo "no vn table either (first import): starting a new history"; exit 0; fi
  [ "$status" = 0 ] || exit 1
  mv "$OUT/q.json" "$OUT/legacy-history.json"
  echo "legacy history of $(jq '[.[].results | length] | add // 0' "$OUT/legacy-history.json") VNs"
  exit 0
fi
[ "$status" = 0 ] || exit 1

mapfile -t days < <(jq -r '.[0].results[].day' "$OUT/q.json")
expected=$(jq '[.[0].results[].parts] | add // 0' "$OUT/q.json")
echo '[]' >"$OUT/history.json"
# A few days per query keeps each response small (~150 KB per day).
for ((i = 0; i < ${#days[@]}; i += 10)); do
  j=$((i + 9 < ${#days[@]} - 1 ? i + 9 : ${#days[@]} - 1))
  q "SELECT day, part, data FROM history_day WHERE day BETWEEN ${days[i]} AND ${days[j]} ORDER BY day, part" || exit 1
  jq -s '.[0] + [.[1][].results[]]' "$OUT/history.json" "$OUT/q.json" >"$OUT/h.tmp" && mv "$OUT/h.tmp" "$OUT/history.json"
done
rm -f "$OUT/q.json"
got=$(jq length "$OUT/history.json")
if [ "$got" != "$expected" ]; then
  echo "fetched $got history rows, expected $expected" >&2
  exit 1
fi
echo "history: ${#days[@]} days, $got rows"

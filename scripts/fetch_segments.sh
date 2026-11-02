#!/bin/bash
# Download the 21 public eligible segments (zarr surface volume + tifxyz mesh) from the public bucket.
set -u
B=s3://vesuvius-challenge-open-data
W=${VIM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=$W/data/segments
grep -E '^\| PHerc(0800|1447) \|' $W/inventory/eligible_segments.md | awk -F'|' '{gsub(/ /,"",$2);gsub(/ /,"",$3);print $2" "$3}' | while read sc seg; do
  echo "$(date -u +%FT%TZ) start $sc $seg"
  aws s3 sync --no-sign-request --only-show-errors "$B/$sc/segments/$seg/" "$OUT/$sc/$seg/" --exclude "*" --include "mesh/*" --include "*.tifxyz/*" --include "surface-volumes/*" --include "*.json" 
  echo "$(date -u +%FT%TZ) done $sc $seg rc=$? $(du -sb $OUT/$sc/$seg | cut -f1)"
done
echo ALLDONE

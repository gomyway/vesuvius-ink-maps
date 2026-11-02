#!/bin/bash
# Ink + surface benchmark on one shared card, nice 10. Run dirs logs/runs/<name>$RUN_SUFFIX; host/card/free memory per run in logs/bench_free.log.
RUN_SUFFIX=${RUN_SUFFIX:-}
W=${VIM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd $W
export PYTHONPATH=$W/data/src/villa/ink-detection:$W/data/src/villa/vesuvius/src:$W/data/pylib_extra
PY=${PYTHON:-python}
SV=data/segments/PHerc1447/20250702235910-auto_grown_20250702235910292/surface-volumes/8.64um-1.2m-116keV-volume-20250521151220.zarr
CK=data/models/reader-v2-step040000.pth
need(){ while :; do f=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F, '{print $1-$2}'); [ "$f" -ge "$1" ] && break; sleep 60; done
  echo "$(date -u +%FT%TZ) $2 host=$(hostname) card=$(nvidia-smi --query-gpu=name --format=csv,noheader) free_MiB=$f" >> logs/bench_free.log; }
run(){ name=$1; mib=$2; shift 2; for t in 0 1 2; do need $mib $name; scripts/measure_run.sh ${name}_hostA "$@"; rc=$?
  [ $rc -eq 0 ] && break; grep -q "out of memory" logs/runs/${name}_hostA/time.log || break; sleep 60; done; echo "$(date -u +%FT%TZ) $name rc=$rc" >> logs/bench.log; }
RV="$PY -m koine_machines.inference.infer $SV $CK"
run readerv2_default_first 2048 $RV data/out/bench_rv2_default_first.tif --overlap 0.5 --blend-mode hann --direction forward
run readerv2_default 2048 $RV data/out/bench_rv2_default.tif --overlap 0.5 --blend-mode hann --direction forward
run readerv2_b8_nc 2048 $RV data/out/bench_rv2_b8_nc.tif --overlap 0.5 --blend-mode hann --direction forward --batch-size 8 --no-compile
run readerv2_b8_nc_rep 2048 $RV data/out/bench_rv2_b8_nc_rep.tif --overlap 0.5 --blend-mode hann --direction forward --batch-size 8 --no-compile
HC="$PY data/models/hecate/hecate.py --checkpoint data/models/hecate/hecate_9.6um.pth --input data/sample/sv96.npy --spacing-um 9.6 --device cuda --precision bf16 --batch-size 8"
run hecate96_bf16_b8 5120 $HC --output data/out/bench_hec_b8.png
run hecate96_bf16_b8_rep 5120 $HC --output data/out/bench_hec_b8_rep.png
run surface_m7_640_p128 8704 $PY scripts/run_surface_m7.py data/models/surface_m7 data/sample/vol_640.npy data/out/bench_surf_640_p128.npy --patch 128
run surface_m7_640_p192 9728 $PY scripts/run_surface_m7.py data/models/surface_m7 data/sample/vol_640.npy data/out/bench_surf_640_p192.npy --patch 192
$PY scripts/bench_upstream.py >> logs/bench.log 2>&1
echo "$(date -u +%FT%TZ) BENCH DONE" >> logs/bench.log

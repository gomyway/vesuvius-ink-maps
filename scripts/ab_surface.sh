#!/bin/bash
# Same-card speed A/B: original run_surface_m7.py vs run_surface_m7_lowmem.py, same input (data/sample/vol_384.npy, 384^3 L0
# block of PHerc1447), patch 192 step 0.5 fp16, interleaved, 3 reps each. Needs ~8 GiB free on the card.
W=${VIM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd $W; PY=${PYTHON:-python}; export PYTHONPATH=$W/data/pylib_extra
L=logs/ab_surface.log; 
log(){ echo "$(date -u +%FT%TZ) $*" >> $L; }
log "watcher start pid $$"
log "start; free_MiB=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{print $2-$1}') card=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
for r in 0 1 2; do
  for v in orig lowmem; do
    if [ $v = orig ]; then scr=scripts/run_surface_m7.py; else scr=scripts/run_surface_m7_lowmem.py; fi
    for t in 0 1 2; do
      f=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{print $2-$1}'); log "ab_${v}_384_p192_r$r try$t free_MiB=$f"
      nice -n 10 scripts/measure_run.sh ab_${v}_384_p192_r$r $PY $scr data/models/surface_m7 data/sample/vol_384.npy data/out/ab_${v}_384_p192_r$r.npy --patch 192; rc=$?
      log "ab_${v}_384_p192_r$r rc=$rc"; [ $rc -eq 0 ] && break; grep -q "out of memory" logs/runs/ab_${v}_384_p192_r$r/time.log || break; sleep 60
    done
  done
done
$PY - <<'PY' >> $L 2>&1
import numpy as np, json, glob, re, os
ref=np.load('data/out/surf_384_p192.npy')
rows=[]
for d in sorted(glob.glob('logs/runs/ab_*_384_p192_r?')):
    n=os.path.basename(d); out=f'data/out/{n}.npy'
    try: j=json.loads([l for l in open(f'{d}/stdout.log') if l.startswith('{')][-1])
    except Exception as e: rows.append((n,'no json',str(e))); continue
    m=open(f'{d}/measure.txt').read(); pk=re.search(r'run_peak_MiB (\S+)',m); wall=re.search(r'Elapsed \(wall clock\) = (\S+)',m)
    a=np.load(out); A=a>127; B=ref>127; dice=2*(A&B).sum()/(A.sum()+B.sum()); ident=(a==ref).mean(); mx=int(np.abs(a.astype(int)-ref.astype(int)).max())
    rows.append((n,wall.group(1) if wall else None,j['loop_s'],j['Mvox_per_s'],j['torch_max_allocated_MiB'],pk.group(1) if pk else None,round(float(dice),4),mx,round(float(ident),4)))
print('AB_SUMMARY run wall loop_s Mvox/s torch_max_alloc_MiB nvsmi_peak_MiB dice_vs_orig_oct31 maxdiff frac_identical')
for r in rows: print('AB_ROW',*r)
PY
log "A/B DONE"

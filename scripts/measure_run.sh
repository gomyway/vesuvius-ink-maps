#!/bin/bash
# usage: measure_run.sh <run_name> <cmd...>
# Writes logs/runs/<run_name>/{cmd.txt,stdout.log,time.log,gpu_trace.csv,gpu_apps.csv,measure.txt}
# gpu_apps.csv = per-process VRAM (nvidia-smi --query-compute-apps) so other tenants on the shared card don't pollute the peak.
set -u
W=${VIM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; R=$W/logs/runs/$1; shift; mkdir -p $R
nvidia-smi --query-compute-apps=pid --format=csv,noheader > $R/pids_before.txt; echo "$@" > $R/cmd.txt; date -u +%FT%TZ > $R/start_utc.txt; hostname >> $R/start_utc.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $R/start_utc.txt
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -lms 500 > $R/gpu_trace.csv & G1=$!
( while true; do nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits 2>/dev/null; sleep 0.5; done ) > $R/gpu_apps.csv & G2=$!
/usr/bin/time -v "$@" > $R/stdout.log 2> $R/time.log; RC=$?
kill $G1 $G2 2>/dev/null; wait 2>/dev/null
date -u +%FT%TZ >> $R/start_utc.txt
python3 - $R <<'PY' > $R/measure.txt
import csv,sys,os,re,subprocess
R=sys.argv[1]
r=list(csv.reader(open(f'{R}/gpu_trace.csv')))[1:]
m=[int(x[1].split()[0]) for x in r if len(x)>1]
print('card_peak_MiB',max(m),'card_min_MiB',min(m),'card_delta_MiB',max(m)-min(m),'samples',len(m))
# per-pid: pids that appear only during our run and are our descendants is hard post-hoc; take pids whose cmdline we logged
pids={}
for line in open(f'{R}/gpu_apps.csv'):
    p=[s.strip() for s in line.split(',')]
    if len(p)==3 and p[2].isdigit(): pids.setdefault(p[1],[]).append(int(p[2]))
pre=set(open(f'{R}/pids_before.txt').read().split())
new={k:max(v) for k,v in pids.items() if k not in pre}
print('new_pid_peaks_MiB',new,'run_peak_MiB',max(new.values()) if new else None)
t=open(f'{R}/time.log').read()
for k in ['Elapsed (wall clock)','Maximum resident','Exit status','Percent of CPU']:
    mm=re.search(re.escape(k)+r'.*?: (.*)',t); print(k,'=',mm.group(1) if mm else None)
PY
cat $R/measure.txt; exit $RC

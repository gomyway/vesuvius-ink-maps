#!/usr/bin/env python3
"""GPU stage for one public segment: reader-v2 and Hecate ink maps, forward and reverse.

Gated on free card memory (and optionally on VIM_GATE_FILE reading 'held'); retry-on-OOM with
smaller batches. Every GPU call is wrapped by scripts/measure_run.sh, so each run leaves
logs/runs/<name>/{cmd.txt,time.log,gpu_trace.csv,gpu_apps.csv,measure.txt} (benchmark rows).
Outputs (idempotent, skipped if present) in data/ink/<scroll>/<segment>/:
  readerv2_fwd.tif readerv2_rev.tif  (uint8, native surface-volume canvas)
  hecate_fwd.png  hecate_rev.png    (uint8, 9.6 um canvas; export_ink.py resizes back)
"""
import argparse, datetime as dt, glob, json, os, socket, subprocess, sys, time
HOST = socket.gethostname()

W = os.environ.get('VIM_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.environ.get('PYTHON', sys.executable)
STOP_AT = dt.datetime.fromisoformat(os.environ['VIM_STOP_AT']) if os.environ.get('VIM_STOP_AT') else None
ENV = dict(os.environ,
           PYTHONPATH=f'{W}/data/src/villa/ink-detection:{W}/data/src/villa/vesuvius/src:{W}/data/pylib_extra',
           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
OOM_MARKERS = ('out of memory', 'OutOfMemoryError', 'CUBLAS_STATUS_ALLOC_FAILED', 'CUDNN_STATUS_ALLOC_FAILED')


def log(*a):
    print(dt.datetime.now(dt.timezone.utc).strftime('%FT%TZ'), *a, flush=True)


def held():
    # optional external gate: a file whose content is 'held' (e.g. a scheduler's reservation state)
    g = os.environ.get('VIM_GATE_FILE')
    return (not g) or open(g).read().strip() == 'held'


def free_mib():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout.strip().split(',')
        return int(out[0]) - int(out[1])
    except Exception:
        return 0


def gate(need_mib, max_wait_s=None):
    """Wait for held + free memory. Returns False if max_wait_s elapses first (caller steps down the batch)."""
    last = 0; t0 = time.time()
    while True:
        if dt.datetime.now(dt.timezone.utc) > STOP_AT:
            log('STOP_AT reached, exiting'); sys.exit(3)
        h, f = held(), free_mib()
        if h and f >= need_mib:
            return True
        if max_wait_s is not None and time.time() - t0 > max_wait_s:
            log(f'gate timeout after {max_wait_s}s: free_MiB={f} need_MiB={need_mib}; stepping down batch'); return False
        if time.time() - last > 1800:
            log(f'gate wait: held={h} free_MiB={f} need_MiB={need_mib}'); last = time.time()
        time.sleep(60)


def run_measured(name, cmd):
    r = subprocess.run([f'{W}/scripts/measure_run.sh', name, *cmd], env=ENV, cwd=W,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rd = f'{W}/logs/runs/{name}'
    txt = ''
    for f in ('time.log', 'stdout.log'):
        try:
            txt += open(f'{rd}/{f}', errors='ignore').read()
        except OSError:
            pass
    return r.returncode, any(m in txt for m in OOM_MARKERS)


def with_retry(tag, make_cmd, out, batches, need_mib):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        log('skip (exists)', out); return True
    frac = {8: 1.0, 4: 0.6, 2: 0.4, 1: 0.3}  # memory need relative to b8 (Hecate measured: b8 3.7 GiB, b1 fp32 1.5 GiB)
    for i, b in enumerate(batches):
        last = i == len(batches) - 1
        if not gate(int(need_mib * frac.get(b, 1.0)), None if last else 900):
            continue
        name = f'{tag}_b{b}_try{i}' + os.environ.get('RUN_SUFFIX', '')
        log(f'launch {name} host={HOST} card_free_MiB={free_mib()}')
        t = time.time()
        rc, oom = run_measured(name, make_cmd(b))
        ok = rc == 0 and os.path.exists(out)
        log(f'{name}: rc={rc} oom={oom} ok={ok} wall={time.time()-t:.1f}s')
        if ok:
            return True
        if not oom:
            return False
        time.sleep(60)  # card filled up from outside our reservation: back off, smaller batch
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scroll'); ap.add_argument('segment')
    a = ap.parse_args()
    seg_dir = f'{W}/data/segments/{a.scroll}/{a.segment}'
    zs = sorted(glob.glob(f'{seg_dir}/surface-volumes/*.zarr'))
    if not zs:
        log('no surface volume zarr in', seg_dir); sys.exit(2)
    sv = zs[0]
    um = 8.64 if '8.64um' in os.path.basename(sv) else float(os.path.basename(sv).split('um')[0])
    out = f'{W}/data/ink/{a.scroll}/{a.segment}'
    os.makedirs(out, exist_ok=True)
    short = a.segment.split('-')[0]
    ok = True
    for d in ('fwd', 'rev'):
        direction = 'forward' if d == 'fwd' else 'reverse'
        o = f'{out}/readerv2_{d}.tif'
        ok &= with_retry(f'ink_{short}_readerv2_{d}',
                         lambda b, o=o, direction=direction: [PY, '-m', 'koine_machines.inference.infer', sv,
                                                              f'{W}/data/models/reader-v2-step040000.pth', o,
                                                              '--overlap', '0.5', '--blend-mode', 'hann',
                                                              '--direction', direction, '--batch-size', str(b),
                                                              '--no-compile'],
                         o, [8, 4, 2, 1], need_mib=2048)
    npy = f'{out}/sv96.npy'
    need_h = not all(os.path.exists(f'{out}/hecate_{d}.png') for d in ('fwd', 'rev'))
    if need_h and not os.path.exists(npy):
        subprocess.run([PY, f'{W}/scripts/resample96.py', sv, str(um), npy], env=ENV, check=True)
    for d in ('fwd', 'rev'):
        o = f'{out}/hecate_{d}.png'
        ok &= with_retry(f'ink_{short}_hecate_{d}',
                         lambda b, o=o, d=d: [PY, f'{W}/data/models/hecate/hecate.py', '--checkpoint',
                                              f'{W}/data/models/hecate/hecate_9.6um.pth', '--input', npy,
                                              '--spacing-um', '9.6', '--device', 'cuda', '--output', o,
                                              '--precision', 'bf16', '--batch-size', str(b)]
                         + (['--reverse'] if d == 'rev' else []),
                         o, [8, 4, 2, 1], need_mib=5120)
    if ok and os.path.exists(npy):
        os.remove(npy)
    log('GPU stage', 'OK' if ok else 'FAILED', a.scroll, a.segment)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

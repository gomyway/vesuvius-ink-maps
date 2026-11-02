#!/usr/bin/env python3
"""Re-baseline the reader-v2 inference benchmark on upstream villa main (vesuvius.ink_detection, clone
data/src/villa_main d57c28e) with the same sample segment as docs/BENCHMARK.md. Runs only after the ink driver has
finished (never shares our slice with it); each run via measure_run.sh -> logs/runs/up_*."""
import os, subprocess, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
import ink_gpu as G
W = G.W
SEG = f'{W}/data/segments/PHerc1447/20250702235910-auto_grown_20250702235910292'
SV = f'{SEG}/surface-volumes/8.64um-1.2m-116keV-volume-20250521151220.zarr'
CK = f'{W}/data/models/reader-v2-step040000.pth'
G.ENV['PYTHONPATH'] = f'{W}/data/src/villa_main/vesuvius/src:{W}/data/pylib_extra'
BASE = [G.PY, '-m', 'vesuvius.ink_detection.inference.infer', SV, CK]
RUNS = [  # name, extra args
    ('up_default_first', ['--overlap', '0.5', '--blend-mode', 'hann']),
    ('up_default', ['--overlap', '0.5', '--blend-mode', 'hann']),
    ('up_b4_compile', ['--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '4']),
    ('up_b8_nc', ['--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '8', '--no-compile']),
    ('up_b8_nc_rep', ['--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '8', '--no-compile']),
    ('up_b8_nc_lt8_cc64', ['--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '8', '--no-compile',
                           '--loader-threads', '8', '--chunk-cache-size', '64']),
    ('up_b1_nc', ['--overlap', '0.5', '--blend-mode', 'hann', '--no-compile']),
]

def driver_running():
    try:
        os.kill(int(open(f'{W}/logs/run_ink_all.pid').read()), 0); return True
    except Exception:
        return False

def main():
    while driver_running():
        if time.time() > G.STOP_AT.timestamp(): sys.exit(3)
        time.sleep(300)
    os.makedirs(f'{W}/data/out/upstream', exist_ok=True)
    res = {}
    for name, extra in RUNS:
        out = f'{W}/data/out/upstream/{name}.tif'
        for i in range(3):
            G.gate(1600)  # reader-v2 peaks at about 1.2 GiB (BENCHMARK.md)
            t = time.time()
            sfx = os.environ.get('RUN_SUFFIX', '')
            rc, oom = G.run_measured((name if i == 0 else f'{name}_try{i}') + sfx, BASE + [out] + extra)
            G.log(name, 'rc', rc, 'oom', oom, 'wall', round(time.time() - t, 1))
            if rc == 0 or not oom: break
            time.sleep(60)
        res[name] = rc
    json.dump(res, open(f'{W}/data/out/upstream/rc.json', 'w'))
    G.log('BENCH DONE', res)

if __name__ == '__main__':
    main()

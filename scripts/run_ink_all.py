#!/usr/bin/env python3
"""Driver: for each of the 21 public PHerc0800/PHerc1447 segments, GPU stage (retry-on-OOM) -> export -> VC3D check. Status in data/ink/status.json. Detached; single instance."""
import datetime as dt, json, os, re, subprocess, sys, time
W = os.environ.get('VIM_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.environ.get('PYTHON', sys.executable)
ENV = dict(os.environ, PYTHONPATH=f'{W}/data/pylib_extra')
ST = f'{W}/data/ink/status.json'

def log(*a): print(dt.datetime.now(dt.timezone.utc).strftime('%FT%TZ'), *a, flush=True)

def segments():
    out = []
    for line in open(f'{W}/inventory/eligible_segments.md'):
        m = re.match(r'\| (PHerc0800|PHerc1447) \| (\S+) \|', line)
        if m: out.append((m.group(1), m.group(2)))
    return out

def downloaded(sc, seg):
    try: return f'done {sc} {seg} rc=0' in open(f'{W}/logs/fetch_segments.log').read()
    except OSError: return False

def save(st):
    st['finished'] = sum(1 for v in st['segments'].values() if v.get('vc3d') == 'pass')
    st['updated'] = dt.datetime.now(dt.timezone.utc).strftime('%FT%TZ')
    json.dump(st, open(ST + '.tmp', 'w'), indent=1); os.replace(ST + '.tmp', ST)

def main():
    os.makedirs(f'{W}/data/ink', exist_ok=True)
    st = json.load(open(ST)) if os.path.exists(ST) else {'segments': {}}
    todo = segments(); log('segments', len(todo))
    for rnd in range(2):
        for sc, seg in todo:
            key = f'{sc}/{seg}'; s = st['segments'].setdefault(key, {})
            if s.get('vc3d') == 'pass': continue
            while not downloaded(sc, seg):
                log('waiting for download', key); time.sleep(60)
            t = time.time()
            r = subprocess.run([PY, f'{W}/scripts/ink_gpu.py', sc, seg], env=ENV)
            s['gpu'] = 'ok' if r.returncode == 0 else f'rc{r.returncode}'; s['gpu_s'] = round(time.time() - t)
            if r.returncode == 3: save(st); log('stop time'); return
            if r.returncode: save(st); continue
            t = time.time()
            r = subprocess.run([PY, f'{W}/scripts/export_ink.py', sc, seg], env=ENV)
            s['export'] = 'ok' if r.returncode == 0 else f'rc{r.returncode}'; s['export_s'] = round(time.time() - t)
            if r.returncode: save(st); continue
            r = subprocess.run([PY, f'{W}/scripts/check_vc3d.py', sc, seg], env=ENV)
            s['vc3d'] = 'pass' if r.returncode == 0 else 'fail'; s['at'] = dt.datetime.now(dt.timezone.utc).strftime('%FT%TZ')
            save(st); log(key, s)
    log('ALLDONE finished', st.get('finished'))

if __name__ == '__main__':
    main()

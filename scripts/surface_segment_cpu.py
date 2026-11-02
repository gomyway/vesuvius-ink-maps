#!/usr/bin/env python3
"""surface_m7 over one public segment's bounding box, without a GPU (or with one via SURFACE_DEVICE), written as a
scroll-space VC3D overlay beside the ink overlays, then checked.
Steps: tifxyz surface points -> bbox (+margin, 128-aligned) -> stream raw L0 chunks from the public bucket -> block .npy
-> scripts/run_surface_m7_lowmem.py (patch 192, step 0.75 by default) -> sparse 6-level OME-Zarr
data/ink/<scroll>/vc3d_overlays/<scroll>_surface_m7.zarr (same shape/chunks as the scroll volume, VC3D meta.json)
-> check (meta/level shapes as VC3D's Volume.cpp, readback at 2000 surface points vs random bbox voxels)
-> data/ink/<scroll>/<seg>/surface_check.json and status entry in data/ink/surface_status.json.
usage: surface_segment_cpu.py <scroll> <segment_id> [--patch 192] [--step 0.75] [--margin 32] [--threads 16]"""
import sys, os, json, time, glob, argparse, random, subprocess
import numpy as np, zarr, boto3
from concurrent.futures import ThreadPoolExecutor
from botocore import UNSIGNED; from botocore.config import Config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import export_ink as E
ap = argparse.ArgumentParser(); ap.add_argument('scroll'); ap.add_argument('seg'); ap.add_argument('--patch', type=int, default=192)
ap.add_argument('--step', type=float, default=0.75); ap.add_argument('--margin', type=int, default=32); ap.add_argument('--threads', type=int, default=16)
a = ap.parse_args(); W = E.W; t_all = time.time()
shape, vol_id = E.VOLS[a.scroll]; B = 'vesuvius-challenge-open-data'; C = 128
vol_key = f'{a.scroll}/volumes/{vol_id}-8.640um-1.2m-116keV-masked.zarr'
seg_dir = sorted(glob.glob(f'{W}/data/segments/{a.scroll}/{a.seg}*'))[0]
ink_dir = sorted(glob.glob(f'{W}/data/ink/{a.scroll}/{a.seg}*'))[0]
meta = json.load(open(f'{ink_dir}/ink_meta.json')); H, Wd = meta['registration']['canvas_hw']
_, _, _, mask, Zc, Yc, Xc = E.surface_points(seg_dir, H, Wd)
inb = mask & (Zc >= 0) & (Yc >= 0) & (Xc >= 0) & (Zc < shape[0] - 1) & (Yc < shape[1] - 1) & (Xc < shape[2] - 1)
lo = [max(0, int(np.floor(c[inb].min())) - a.margin) // C * C for c in (Zc, Yc, Xc)]
hi = [min(n, -(-(int(np.ceil(c[inb].max())) + a.margin) // C) * C) for c, n in zip((Zc, Yc, Xc), shape)]
blk = [h - l for l, h in zip(lo, hi)]
print(json.dumps(dict(seg=a.seg, bbox_lo=lo, bbox_hi=hi, block=blk, gvox=round(np.prod(blk) / 1e9, 3), surface_pts=int(inb.sum()))), flush=True)
out_dir = f'{W}/data/surface/{a.scroll}/{a.seg}'; os.makedirs(out_dir, exist_ok=True)
blk_npy = f'{out_dir}/block_{lo[0]}_{lo[1]}_{lo[2]}.npy'; prob_npy = f'{out_dir}/surface_m7_p{a.patch}_s{a.step}.npy'
# 1. fetch raw L0 chunks (compressor null, 128^3 uint8)
if not os.path.exists(blk_npy):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, max_pool_connections=a.threads * 2))
    vol = np.zeros(blk, np.uint8); t0 = time.time(); nb = [0]; miss = [0]
    def get(ijk):
        i, j, k = ijk
        try: b = s3.get_object(Bucket=B, Key=f'{vol_key}/0/{lo[0]//C+i}/{lo[1]//C+j}/{lo[2]//C+k}')['Body'].read()
        except s3.exceptions.NoSuchKey: miss[0] += 1; return
        nb[0] += len(b); vol[i*C:(i+1)*C, j*C:(j+1)*C, k*C:(k+1)*C] = np.frombuffer(b, np.uint8).reshape(C, C, C)
    idx = [(i, j, k) for i in range(blk[0] // C) for j in range(blk[1] // C) for k in range(blk[2] // C)]
    with ThreadPoolExecutor(a.threads) as ex: list(ex.map(get, idx))
    dt = time.time() - t0; np.save(blk_npy, vol)
    print(json.dumps(dict(fetch_chunks=len(idx), missing=miss[0], MB=round(nb[0] / 1e6, 1), fetch_s=round(dt, 1), MB_per_s=round(nb[0] / 1e6 / dt, 1), nonzero_frac=round(float((vol > 0).mean()), 3))), flush=True)
    del vol
# 2. surface_m7 (low-VRAM runner; SURFACE_DEVICE from the environment, default cuda if available else cpu)
if not os.path.exists(prob_npy):
    env = dict(os.environ, PYTHONPATH=f'{W}/data/pylib_extra'); env.setdefault('SURFACE_DEVICE', 'cpu')
    cmd = [sys.executable, f'{W}/scripts/run_surface_m7_lowmem.py', f'{W}/data/models/surface_m7', blk_npy, prob_npy, '--patch', str(a.patch), '--step', str(a.step)]
    print('run', ' '.join(cmd), 'SURFACE_DEVICE=' + env['SURFACE_DEVICE'], flush=True)
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True); print(r.stdout[-2000:], flush=True)
    if r.returncode: sys.exit(r.returncode)
    json.dump(json.loads([l for l in r.stdout.splitlines() if l.startswith('{')][-1]), open(f'{out_dir}/run.json', 'w'), indent=1)
prob = np.load(prob_npy)
# 3. write into the scroll-space overlay (sparse, 6 levels, VC3D meta.json), block region only
p = f'{W}/data/ink/{a.scroll}/vc3d_overlays/{a.scroll}_surface_m7.zarr'
if not os.path.exists(f'{p}/meta.json'):
    os.makedirs(p, exist_ok=True); g = zarr.open_group(p, mode='a'); ds = []
    for l in range(E.NLEV):
        shp = tuple(E.ceil_div_pow2(v, l) for v in shape)
        g.create_dataset(str(l), shape=shp, chunks=(C, C, C), dtype='u1', compressor=E.COMP, fill_value=0, dimension_separator='/', write_empty_chunks=False, overwrite=True)
        ds.append({'path': str(l), 'coordinateTransformations': [{'type': 'scale', 'scale': [E.UM * 2 ** l] * 3}]})
    g.attrs['multiscales'] = [{'version': '0.4', 'name': f'{a.scroll}_surface_m7', 'axes': [{'name': ax, 'type': 'space', 'unit': 'micrometer'} for ax in 'zyx'], 'datasets': ds}]
    g.attrs['vesuvius_surface'] = {'source_volume': f's3://{B}/{vol_key}', 'model': 'scrollprize/surface_m7_nnunet fold_0 (Apache-2.0)', 'runner': 'scripts/run_surface_m7_lowmem.py',
                                   'note': 'surface probability x255 inside the bounding boxes of the processed public segments; 0 = not processed', 'segments': []}
    json.dump({'type': 'vol', 'uuid': f'{a.scroll}_surface_m7', 'name': f'{a.scroll} surface_m7', 'format': 'zarr', 'width': shape[2], 'height': shape[1], 'slices': shape[0], 'voxelsize': E.UM, 'min': 0.0, 'max': 255.0}, open(f'{p}/meta.json', 'w'), indent=1)
g = zarr.open_group(p, mode='a'); cur = prob
for l in range(E.NLEV):
    arr = zarr.open_array(f'{p}/{l}', mode='r+', write_empty_chunks=False); s = 2 ** l
    arr[lo[0]//s:lo[0]//s + cur.shape[0], lo[1]//s:lo[1]//s + cur.shape[1], lo[2]//s:lo[2]//s + cur.shape[2]] = cur
    if l < E.NLEV - 1: cur = cur.reshape(cur.shape[0]//2, 2, cur.shape[1]//2, 2, cur.shape[2]//2, 2).max(axis=(1, 3, 5))
segs = g.attrs['vesuvius_surface']; segs['segments'] = sorted(set(segs['segments'] + [a.seg])); g.attrs['vesuvius_surface'] = segs
# 4. check
res = {'checks': [], 'pass': True}
def ck(name, ok, info=''): res['checks'].append({'check': name, 'ok': bool(ok), 'info': str(info)}); res['pass'] &= bool(ok)
m = json.load(open(f'{p}/meta.json'))
ck('meta.json type/fields', m.get('type') == 'vol' and all(k in m for k in ('uuid', 'width', 'height', 'slices')) and m.get('format') == 'zarr')
ck('meta dims == scroll volume', (m['slices'], m['height'], m['width']) == shape, (m['slices'], m['height'], m['width']))
for l in range(E.NLEV):
    za = json.load(open(f'{p}/{l}/.zarray')); exp = [E.ceil_div_pow2(v, l) for v in shape]
    ck(f'level {l} shape', za['shape'] == exp and za['dtype'] == '|u1', (za['shape'], exp))
ov = zarr.open_array(f'{p}/0', mode='r'); ii, jj = np.nonzero(inb); rng = random.Random(0)
idx = [rng.randrange(len(ii)) for _ in range(2000)]
surf = np.array([int(ov[int(Zc[ii[k], jj[k]]), int(Yc[ii[k], jj[k]]), int(Xc[ii[k], jj[k]])]) for k in idx])
rnd = np.array([int(ov[rng.randrange(lo[0], hi[0]), rng.randrange(lo[1], hi[1]), rng.randrange(lo[2], hi[2])]) for _ in range(2000)])
ck('surface points lie inside the written block', all(lo[d] <= int(c[inb].min()) and int(c[inb].max()) < hi[d] for d, c in enumerate((Zc, Yc, Xc))))
ck('readback at 2000 tifxyz surface points: mean prob > 0.5', surf.mean() > 127.5, f'mean {surf.mean()/255:.3f}, frac>=0.5 {(surf>=128).mean():.3f}')
ck('surface points score higher than random bbox voxels (ratio > 2)', surf.mean() > 2 * max(rnd.mean(), 1), f'random mean {rnd.mean()/255:.3f}, frac>=0.5 {(rnd>=128).mean():.3f}')
res.update(seg=a.seg, bbox_lo=lo, bbox_hi=hi, block=blk, overlay=os.path.relpath(p, W), prob_npy=os.path.relpath(prob_npy, W), total_s=round(time.time() - t_all, 1))
json.dump(res, open(f'{ink_dir}/surface_check.json', 'w'), indent=1)
sp = f'{W}/data/ink/surface_status.json'; st = json.load(open(sp)) if os.path.exists(sp) else {}
st[f'{a.scroll}/{a.seg}'] = {'pass': res['pass'], 'checks': f"{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}", 'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'run': json.load(open(f'{out_dir}/run.json'))}
json.dump(st, open(sp, 'w'), indent=1)
print('PASS' if res['pass'] else 'FAIL', sum(c['ok'] for c in res['checks']), '/', len(res['checks']), json.dumps(res['checks'][-3:]))
sys.exit(0 if res['pass'] else 1)

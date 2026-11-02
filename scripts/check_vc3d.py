#!/usr/bin/env python3
"""VC3D-loadability check for one segment's outputs, mirroring volume-cartographer core/src/Volume.cpp:
meta.json type=vol with uuid/width/height/slices; every zarr level shape == ceil(dim / 2^level) as [z,y,x];
uint8; level 0 equal to the scroll volume shape (so the overlay registers to the scroll in VC3D).
Plus a registration self-test: overlay values read back at 2000 random valid surface points must be
>= the 2D map value there (max-merge with other segments can only raise them) and equal for >= 95%.
Writes data/ink/<scroll>/<segment>/vc3d_check.json; exit 0 = pass."""
import json, os, sys, random
import numpy as np, zarr
sys.path.insert(0, os.path.dirname(__file__))
import export_ink as E

def main():
    scroll, seg = sys.argv[1], sys.argv[2]
    d = f'{E.W}/data/ink/{scroll}/{seg}'
    meta = json.load(open(f'{d}/ink_meta.json'))
    res = {'checks': [], 'pass': True}
    def ck(name, ok, info=''):
        res['checks'].append({'check': name, 'ok': bool(ok), 'info': str(info)}); res['pass'] &= bool(ok)
    for n, rel in meta['overlays'].items():
        p = f'{E.W}/{rel}'
        m = json.load(open(f'{p}/meta.json'))
        ck(f'{n} meta.json type/fields', m.get('type') == 'vol' and all(k in m for k in ('uuid', 'width', 'height', 'slices')) and m.get('format') == 'zarr')
        shp = E.VOLS[scroll][0]
        ck(f'{n} meta dims == scroll volume', (m['slices'], m['height'], m['width']) == shp, (m['slices'], m['height'], m['width']))
        for l in range(E.NLEV):
            za = json.load(open(f'{p}/{l}/.zarray'))
            exp = [E.ceil_div_pow2(v, l) for v in shp]
            ck(f'{n} level {l} shape', za['shape'] == exp and za['dtype'] == '|u1', (za['shape'], exp))
    H, Wd = meta['registration']['canvas_hw']
    for n in meta['maps']:
        za = json.load(open(f'{d}/{n}.ome.zarr/0/.zarray'))
        ck(f'2D {n} canvas == surface volume', za['shape'] == [H, Wd], za['shape'])
    g = meta['registration']['grid_times_inv_scale_hw']
    ck('tifxyz grid / scale ~= canvas', abs(g[0] - H) <= 1 / meta['registration']['tifxyz_scale_xy'][1] and abs(g[1] - Wd) <= 1 / meta['registration']['tifxyz_scale_xy'][0], (g, H, Wd))
    seg_dir = f'{E.W}/data/segments/{scroll}/{seg}'
    _, _, _, mask, Zc, Yc, Xc = E.surface_points(seg_dir, H, Wd)
    shp0 = E.VOLS[scroll][0]
    inb = mask & (Zc >= 0) & (Yc >= 0) & (Xc >= 0) & (Zc < shp0[0] - 1) & (Yc < shp0[1] - 1) & (Xc < shp0[2] - 1)
    ck('surface points inside the scroll volume (out-of-bounds ones are not splatted)', inb.sum() > 0, f'{int(inb.sum())} of {int(mask.sum())} in bounds')
    ii, jj = np.nonzero(inb)
    rng = random.Random(0); idx = [rng.randrange(len(ii)) for _ in range(2000)]
    for n, rel in meta['overlays'].items():
        mp = zarr.open_array(f'{d}/{n}.ome.zarr/0', mode='r')[:]
        ov = zarr.open_array(f'{E.W}/{rel}/0', mode='r')
        ge = le = 0
        for k in idx:
            i, j = ii[k], jj[k]
            z, y, x = (int(np.floor(c[i, j])) for c in (Zc, Yc, Xc))
            a, b = int(mp[i, j]), int(ov[z, y, x])
            nb = int(mp[max(0, i - 2):i + 3, max(0, j - 2):j + 3].max())
            ge += b >= a; le += b <= nb
        # negative control: random voxels in the segment's bounding box are mostly off-surface (0)
        lo = [max(0, int(np.floor(c[inb].min()))) for c in (Zc, Yc, Xc)]; hi = [min(n, int(np.ceil(c[inb].max()))) for c, n in zip((Zc, Yc, Xc), shp0)]
        nz_ctrl = sum(int(ov[rng.randrange(lo[0], hi[0]), rng.randrange(lo[1], hi[1]), rng.randrange(lo[2], hi[2])]) > 0 for _ in range(500))
        ck(f'{n} overlay readback: map <= overlay (100%) and overlay <= 5x5 map max (>=95%)', ge == 2000 and le >= 1900, f'ge={ge} le={le} of 2000')
        ck(f'{n} negative control: random bbox voxels nonzero < 20%', nz_ctrl < 100, f'{nz_ctrl}/500 nonzero')
    json.dump(res, open(f'{d}/vc3d_check.json', 'w'), indent=1)
    print('PASS' if res['pass'] else 'FAIL', sum(c['ok'] for c in res['checks']), '/', len(res['checks']))
    sys.exit(0 if res['pass'] else 1)

if __name__ == '__main__':
    main()

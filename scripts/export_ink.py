#!/usr/bin/env python3
"""CPU stage for one segment: GPU ink maps -> registered, VC3D-loadable outputs.

Per segment (data/ink/<scroll>/<segment>/):
  <map>.ome.zarr   2D OME-Zarr (y,x), 5 levels, on the segment's surface-volume canvas, which is the
                   tifxyz grid divided by its meta.json scale (pixel (i,j) == tifxyz grid (i*sy, j*sx)).
  contact.png      CT mid layer | reader-v2 | Hecate | ensemble, one row per direction, 1 cm scale bar.
  ink_meta.json    shapes, registration, per-map stats, timings.
Per scroll (data/ink/<scroll>/vc3d_overlays/<scroll>_ink_<map>.zarr):
  sparse uint8 OME-Zarr in scroll voxel space (same shape/chunks as the scroll volume, 6 levels, meta.json),
  each map value written at its surface point (floor voxel + 1-voxel dilation = all trilinear corners),
  levels 1-5 by 2x max pooling. VC3D loads it as an overlay volume on the scroll.
"""
import argparse, glob, json, os, time
import numpy as np, cv2, tifffile, zarr
from numcodecs import Blosc

W = os.environ.get('VIM_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMP = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
VOLS = {'PHerc0800': ((24298, 9867, 9867), '20250521135224'),
        'PHerc1447': ((24297, 8343, 8343), '20250521151220')}
UM = 8.64
C = 128
NLEV = 6


def ceil_div_pow2(v, l):
    return -(-v // (1 << l))


def load_maps(d, H, Wd, dirs):
    m = {}
    for k in dirs:
        r = tifffile.imread(f'{d}/readerv2_{k}.tif').astype(np.float32)
        assert r.shape == (H, Wd), (r.shape, H, Wd)
        h = cv2.imread(f'{d}/hecate_{k}.png', cv2.IMREAD_UNCHANGED)
        h = (h[..., 0] if h.ndim == 3 else h).astype(np.float32)
        h = cv2.resize(h, (Wd, H), interpolation=cv2.INTER_LINEAR)
        m[f'readerv2_{k}'] = r.astype(np.uint8)
        m[f'hecate_{k}'] = np.round(h).clip(0, 255).astype(np.uint8)
        m[f'ensemble_{k}'] = np.round((r + h) / 2).clip(0, 255).astype(np.uint8)
    return m


def write_2d(path, img, attrs, levels=5):
    g = zarr.open_group(path, mode='w')
    cur = img
    ds = []
    for l in range(levels):
        if l:
            h, w = cur.shape
            cur = cv2.resize(cur, (-(-w // 2), -(-h // 2)), interpolation=cv2.INTER_AREA)
        g.create_dataset(str(l), data=cur, chunks=(512, 512), compressor=COMP, dimension_separator='/', overwrite=True)
        ds.append({'path': str(l), 'coordinateTransformations': [{'type': 'scale', 'scale': [UM * 2 ** l] * 2}]})
    g.attrs['multiscales'] = [{'version': '0.4', 'name': os.path.basename(path),
                               'axes': [{'name': 'y', 'type': 'space', 'unit': 'micrometer'},
                                        {'name': 'x', 'type': 'space', 'unit': 'micrometer'}], 'datasets': ds}]
    g.attrs['vesuvius_ink'] = attrs


def contact_sheet(path, ct, maps, dirs):
    H, Wd = ct.shape
    s = min(1.0, 700.0 / H)
    size = (max(1, int(Wd * s)), max(1, int(H * s)))
    rows = []
    for k in dirs:
        panels = [ct] + [maps[f'{n}_{k}'] for n in ('readerv2', 'hecate', 'ensemble')]
        ps = []
        for p, lab in zip(panels, ['CT layer mid', f'reader-v2 {k}', f'Hecate {k}', f'ensemble {k}']):
            q = cv2.cvtColor(cv2.resize(p, size, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
            cv2.putText(q, lab, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            bar = int(round(10000.0 / UM * s))  # 1 cm
            y = q.shape[0] - 20
            cv2.rectangle(q, (10, y - 8), (10 + bar, y), (255, 255, 255), -1)
            cv2.putText(q, '1 cm', (10, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ps.append(q)
        rows.append(np.concatenate(ps, axis=1))
    cv2.imwrite(path, np.concatenate(rows, axis=0))


def surface_points(seg_dir, H, Wd):
    t = sorted(glob.glob(f'{seg_dir}/mesh/*.tifxyz'))[0]
    meta = json.load(open(f'{t}/meta.json'))
    sx, sy = meta['scale']
    X, Y, Z = (tifffile.imread(f'{t}/{a}.tif').astype(np.float32) for a in 'xyz')
    gh, gw = X.shape
    valid = ((X >= 0) & (Y >= 0) & (Z >= 0)).astype(np.float32)
    gi = np.arange(H, dtype=np.float64) * sy
    gj = np.arange(Wd, dtype=np.float64) * sx
    i0 = np.clip(np.floor(gi).astype(int), 0, gh - 2); wi = (gi - i0).astype(np.float32)[:, None]
    j0 = np.clip(np.floor(gj).astype(int), 0, gw - 2); wj = (gj - j0).astype(np.float32)[None, :]

    def interp(A):
        a0, a1 = A[i0], A[i0 + 1]
        top = a0[:, j0] * (1 - wj) + a0[:, j0 + 1] * wj
        bot = a1[:, j0] * (1 - wj) + a1[:, j0 + 1] * wj
        return top * (1 - wi) + bot * wi

    v0, v1 = valid[i0], valid[i0 + 1]
    mask = (v0[:, j0] * v0[:, j0 + 1] * v1[:, j0] * v1[:, j0 + 1]) > 0
    return t, (sx, sy), (gh, gw), mask, interp(Z), interp(Y), interp(X)


def open_overlay(scroll, name):
    shape, vol_id = VOLS[scroll]
    p = f'{W}/data/ink/{scroll}/vc3d_overlays/{scroll}_ink_{name}.zarr'
    if not os.path.exists(f'{p}/meta.json'):
        os.makedirs(p, exist_ok=True)
        g = zarr.open_group(p, mode='a')
        ds = []
        for l in range(NLEV):
            shp = tuple(ceil_div_pow2(v, l) for v in shape)
            g.create_dataset(str(l), shape=shp, chunks=(C, C, C), dtype='u1', compressor=COMP, fill_value=0,
                             dimension_separator='/', write_empty_chunks=False, overwrite=True)
            ds.append({'path': str(l), 'coordinateTransformations': [{'type': 'scale', 'scale': [UM * 2 ** l] * 3}]})
        g.attrs['multiscales'] = [{'version': '0.4', 'name': f'{scroll}_ink_{name}',
                                   'axes': [{'name': a, 'type': 'space', 'unit': 'micrometer'} for a in 'zyx'],
                                   'datasets': ds}]
        g.attrs['vesuvius_ink'] = {'source_volume': f's3://vesuvius-challenge-open-data/{scroll}/volumes/{vol_id}-8.640um-1.2m-116keV-masked.zarr',
                                   'map': name, 'note': 'ink probability x255 written on public segment surfaces; 0 = no segment'}
        json.dump({'type': 'vol', 'uuid': f'{scroll}_ink_{name}', 'name': f'{scroll} ink {name}', 'format': 'zarr',
                   'width': shape[2], 'height': shape[1], 'slices': shape[0], 'voxelsize': UM, 'min': 0.0, 'max': 255.0},
                  open(f'{p}/meta.json', 'w'), indent=1)
    return p, [zarr.open_array(f'{p}/{l}', mode='r+', write_empty_chunks=False) for l in range(NLEV)]


def splat(scroll, names, maps, mask, Zc, Yc, Xc):
    shape, _ = VOLS[scroll]
    zi = np.floor(Zc[mask]).astype(np.int64); yi = np.floor(Yc[mask]).astype(np.int64); xi = np.floor(Xc[mask]).astype(np.int64)
    keep = (zi >= 0) & (yi >= 0) & (xi >= 0) & (zi < shape[0] - 1) & (yi < shape[1] - 1) & (xi < shape[2] - 1)
    zi, yi, xi = zi[keep], yi[keep], xi[keep]
    vals = {n: maps[n][mask][keep] for n in names}
    ny, nx = -(-shape[1] // C), -(-shape[2] // C)
    cid = ((zi // C) * ny + yi // C) * nx + xi // C
    loc = ((zi % C) * C + yi % C) * C + xi % C
    o = np.argsort(cid, kind='stable')
    cid, loc = cid[o], loc[o]
    vals = {n: v[o] for n, v in vals.items()}
    cuts = np.flatnonzero(np.diff(cid)) + 1
    starts = np.r_[0, cuts]; ends = np.r_[cuts, len(cid)]
    touched = []
    arrs = {n: open_overlay(scroll, n)[1] for n in names}
    for s, e in zip(starts, ends):
        c = int(cid[s]); cx = c % nx; cy = (c // nx) % ny; cz = c // (nx * ny)
        sl = tuple(slice(k * C, min((k + 1) * C, d)) for k, d in zip((cz, cy, cx), shape))
        touched.append((cz, cy, cx))
        for n in names:
            a0 = arrs[n][0]
            buf = np.zeros((C, C, C), np.uint8)
            ex = a0[sl]
            buf[:ex.shape[0], :ex.shape[1], :ex.shape[2]] = ex
            pts = np.zeros(C * C * C, np.uint8)
            np.maximum.at(pts, loc[s:e], vals[n][s:e])
            pts = pts.reshape(C, C, C)
            d = pts.copy()  # +1 dilation in z, y, x (covers the 8 trilinear corners)
            d[1:] = np.maximum(d[1:], pts[:-1])
            d[:, 1:] = np.maximum(d[:, 1:], d[:, :-1])
            d[:, :, 1:] = np.maximum(d[:, :, 1:], d[:, :, :-1])
            np.maximum(buf, d, out=buf)
            a0[sl] = buf[:ex.shape[0], :ex.shape[1], :ex.shape[2]]
    # pyramid: 2x max pooling for the parents of touched chunks
    cur = set(touched)
    for l in range(1, NLEV):
        par = {(z // 2, y // 2, x // 2) for z, y, x in cur}
        for n in names:
            src, dst = arrs[n][l - 1], arrs[n][l]
            for (pz, py, px) in par:
                ssl = tuple(slice(k * 2 * C, min((k + 1) * 2 * C, d)) for k, d in zip((pz, py, px), src.shape))
                blk = src[ssl]
                pad = [(0, (-s) % 2) for s in blk.shape]
                blk = np.pad(blk, pad)
                b = blk.reshape(blk.shape[0] // 2, 2, blk.shape[1] // 2, 2, blk.shape[2] // 2, 2).max(axis=(1, 3, 5))
                dsl = tuple(slice(k * C, k * C + b.shape[i]) for i, k in enumerate((pz, py, px)))
                dst[dsl] = b
        cur = par
    return len(zi), len(touched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scroll'); ap.add_argument('segment')
    ap.add_argument('--dirs', default='fwd,rev')
    ap.add_argument('--overlays', default='ensemble_fwd,ensemble_rev,readerv2_fwd')
    a = ap.parse_args()
    t0 = time.time()
    dirs = a.dirs.split(',')
    seg_dir = f'{W}/data/segments/{a.scroll}/{a.segment}'
    d = f'{W}/data/ink/{a.scroll}/{a.segment}'
    sv = sorted(glob.glob(f'{seg_dir}/surface-volumes/*.zarr'))[0]
    v0 = zarr.open_array(f'{sv}/0', mode='r')
    nz, H, Wd = v0.shape
    ct = v0[nz // 2]
    maps = load_maps(d, H, Wd, dirs)
    tif, scale, grid, mask, Zc, Yc, Xc = surface_points(seg_dir, H, Wd)
    reg = {'tifxyz': os.path.relpath(tif, W), 'tifxyz_scale_xy': scale, 'tifxyz_grid_hw': grid,
           'canvas_hw': [H, Wd], 'surface_volume': os.path.relpath(sv, W),
           'relation': 'canvas pixel (row i, col j) <-> tifxyz grid (i*scale_y, j*scale_x); same canvas as the segment surface volume',
           'grid_times_inv_scale_hw': [round(grid[0] / scale[1], 1), round(grid[1] / scale[0], 1)]}
    stats = {}
    for n, m in maps.items():
        write_2d(f'{d}/{n}.ome.zarr', m, dict(reg, map=n, voxel_um=UM))
        mm = m[mask]
        stats[n] = {'mean': round(float(mm.mean()), 2), 'p99': int(np.percentile(mm, 99)),
                    'frac_gt_127': round(float((mm > 127).mean()), 5)}
    contact_sheet(f'{d}/contact.png', ct, maps, dirs)
    names = [n for n in a.overlays.split(',') if n in maps]
    npts, nch = splat(a.scroll, names, maps, mask, Zc, Yc, Xc)
    meta = {'scroll': a.scroll, 'segment': a.segment, 'registration': reg, 'maps': sorted(maps),
            'overlays': {n: f'data/ink/{a.scroll}/vc3d_overlays/{a.scroll}_ink_{n}.zarr' for n in names},
            'surface_pixels_valid': int(mask.sum()), 'papyrus_cm2_valid': round(float(mask.sum()) * (UM * 1e-4) ** 2, 3),
            'splatted_points': npts, 'touched_l0_chunks': nch, 'stats': stats,
            'export_seconds': round(time.time() - t0, 1), 'status': 'exported'}
    json.dump(meta, open(f'{d}/ink_meta.json', 'w'), indent=1)
    print(json.dumps({k: meta[k] for k in ('segment', 'surface_pixels_valid', 'splatted_points', 'touched_l0_chunks', 'export_seconds')}))


if __name__ == '__main__':
    main()

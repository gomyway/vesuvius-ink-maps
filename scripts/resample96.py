# Resample a native surface volume to 9.6 um (copy of reader-v2 scripts/infer_with_hecate.py resample_96, MIT) -> .npy
import sys, time, numpy as np, zarr
import os; sys.path.insert(0, os.path.join(os.environ.get('VIM_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data/src/reader-v2/scripts'))
from infer_with_hecate import resample_96
src, um, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
t=time.time(); r=zarr.open(src, mode='r'); v=r['0'] if not isinstance(r, zarr.Array) else r
a=resample_96(v[:], um); np.save(out, a); print('in', v.shape, 'out', a.shape, 'sec', round(time.time()-t,2))

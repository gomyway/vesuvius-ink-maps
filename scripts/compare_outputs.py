# Sanity: stats and pairwise Pearson r between ink maps (resized to the reader-v2 canvas); writes a preview PNG.
import sys, numpy as np, tifffile, cv2, json
O='data/out/'
ref=tifffile.imread(O+'readerv2_default.tif').astype(np.float32); H,W=ref.shape
def load(f):
    a=tifffile.imread(O+f) if f.endswith('.tif') else cv2.imread(O+f,cv2.IMREAD_UNCHANGED)
    a=(a[...,0] if a.ndim==3 else a).astype(np.float32)
    return cv2.resize(a,(W,H),interpolation=cv2.INTER_LINEAR) if a.shape!=(H,W) else a
fs=['readerv2_default.tif','readerv2_b8_rerun.tif','readerv2_b8_ov0.tif','readerv2_p64_b8_nc.tif','readerv2_p256_b8_nc.tif','readerv2_b8_nc_rep.tif','hecate96_bf16_b8.png','hecate96_fp32_b1.png','hecate96_bf16_b8_stride64.png']
m=ref>0; res={}
for f in fs:
    a=load(f); r=np.corrcoef(ref[m],a[m])[0,1]; h=load('hecate96_bf16_b8.png'); rh=np.corrcoef(h[m],a[m])[0,1]
    res[f]=dict(mean=round(float(a[m].mean()),2),p99=float(np.percentile(a[m],99)),frac_gt128=round(float((a[m]>128).mean()),4),r_vs_readerv2_default=round(float(r),4),r_vs_hecate_bf16_b8=round(float(rh),4))
    print(f,res[f])
json.dump(res,open('logs/output_agreement.json','w'),indent=1)
sv=tifffile.imread('data/sample/seg/surface-volumes/8.64um-1.2m-116keV-volume-20250521151220.tifs/15.tif') if False else None
import zarr; z=zarr.open('data/sample/seg/surface-volumes/8.64um-1.2m-116keV-volume-20250521151220.zarr',mode='r')['0'][15]
s=4; tiles=[cv2.resize(x,(W//s,H//s)) for x in [z.astype(np.float32), ref, load('hecate96_bf16_b8.png')]]
cv2.imwrite('analysis/sample_preview_ct15_readerv2_hecate.png', np.hstack([cv2.normalize(t,None,0,255,cv2.NORM_MINMAX) for t in tiles]).astype(np.uint8))

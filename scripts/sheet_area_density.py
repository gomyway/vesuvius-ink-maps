# Sheet area per unit volume from the surface_m7 prediction: marching-cubes iso-surface area at p=0.5, /2 (a thin sheet label has two faces).
import numpy as np, sys
from skimage.measure import marching_cubes, mesh_surface_area
um=float(sys.argv[2]); s=np.load(sys.argv[1]).astype(np.float32)/255; m=8; s=s[m:-m,m:-m,m:-m]  # drop borders
v,f,_,_=marching_cubes(s,0.5); A=mesh_surface_area(v,f)/2  # voxel^2
vol=s.size; d=A/vol  # 1/voxel
pitch_vox=1/d
cm=um*1e-4
print(f'block {s.shape} midsurface_area_vox2 {A:.0f} density_per_vox {d:.5f} mean_sheet_pitch_vox {pitch_vox:.1f} = {pitch_vox*um:.0f} um; sheet_cm2_per_cm3 {d/cm:.1f}')

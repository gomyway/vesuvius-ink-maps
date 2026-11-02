# GPU-hours and cloud $ per scroll/segment from measured runs. Inputs: inventory/envelope_volumes.json, logs/sheet_area_density.txt, logs/runs/*.
import json, re
E=json.load(open('inventory/envelope_volumes.json'))
DENS=72.1  # sheet cm2 per cm3, logs/sheet_area_density.txt (640^3 p192 block, PHerc1447); 384^3 block gave 74.3
PAP=3.787  # papyrus cm2 of the sample segment, logs/sample_area.txt
# measured seconds for the sample (papyrus 3.787 cm2) on the shared RTX 5090
INK={'reader-v2 b8 (loop, steady)':5.30,          # logs/runs/readerv2_b8_nc_rep (loop_s via scripts/loop_time.py)
     'reader-v2 default cmd (wall)':18.59,        # logs/runs/readerv2_default
     'Hecate bf16 b8 (wall)':54.20,               # logs/runs/hecate96_bf16_b8_rep
     'Hecate bf16 b8 stride64 (wall)':17.57,      # logs/runs/hecate96_bf16_b8_stride64
     'reader-v2+Hecate ensemble (sum)':5.30+54.20}
SURF_MVOX=11.03  # logs/runs/surface_m7_640_p192 Mvox/s (fp16, no TTA, step 0.5)
PRICE={'own RTX5090':0,'RunPod 4090 community':0.34,'RunPod 5090 community':0.69,'RunPod A100 80GB community':1.19,'RunPod H100 PCIe community':1.99,'Lambda H100 SXM 1x':4.29}
rows=[]
for s,e in E.items():
    area=e['envelope_cm3']*DENS; vox=e['envelope_cm3']/(e['um']*1e-4)**3
    r=dict(scroll=s,um=e['um'],envelope_cm3=e['envelope_cm3'],height_cm=e['height_cm'],sheet_area_cm2=round(area),envelope_voxels_e9=round(vox/1e9,1))
    for k,sec in INK.items(): r['ink_h '+k]=round(area/PAP*sec/3600,2)
    r['surface_m7_h (no TTA)']=round(vox/SURF_MVOX/1e6/3600,1); r['surface_m7_h (8x mirror TTA)']=round(8*vox/SURF_MVOX/1e6/3600,1)
    rows.append(r)
json.dump(dict(method='sheet_area = envelope_cm3 (nonzero L5 voxels) x 72.1 cm2/cm3 (marching-cubes midsurface of surface_m7 on 640^3 block); ink_h = sheet_area/3.787 cm2 x measured seconds; surface_h = envelope voxels / 11.03 Mvox/s',prices_per_h=PRICE,rows=rows),open('analysis/extrapolation.json','w'),indent=1)
ks=[k for k in rows[0] if k.startswith('ink_h') or k.startswith('surface')]
print('| scroll | um | envelope cm3 | sheet area cm2 | L0 vox (1e9) | '+' | '.join(ks)+' |'); print('|'+'---|'*(5+len(ks)))
for r in rows: print(f"| {r['scroll']} | {r['um']} | {r['envelope_cm3']} | {r['sheet_area_cm2']:,} | {r['envelope_voxels_e9']} | "+' | '.join(str(r[k]) for k in ks)+' |')
T={k:round(sum(r[k] for r in rows),1) for k in ks}; GP="PHerc0125 PHerc0191 PHerc0211 PHerc0257 PHerc0268 PHerc0358 PHerc0800 PHerc0813 PHerc0826 PHerc1203 PHerc1218 PHerc1447 PHerc1545".split()
TG={k:round(sum(r[k] for r in rows if r['scroll'] in GP),1) for k in ks}
print('\nTOTAL all 22:',T,'\nTOTAL 13 GP:',TG,'\nGP sheet area cm2', sum(r['sheet_area_cm2'] for r in rows if r['scroll'] in GP))

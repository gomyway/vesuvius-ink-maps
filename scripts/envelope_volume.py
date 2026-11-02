# Scroll envelope volume from the masked OME-Zarr level 5 (32x): count nonzero voxels (masked-out = 0 / absent chunk).
import boto3, json, numpy as np, sys, itertools
from botocore import UNSIGNED; from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
B='vesuvius-challenge-open-data'; s3=boto3.client('s3',config=Config(signature_version=UNSIGNED,max_pool_connections=32))
SC="PHerc0125 PHerc0191 PHerc0211 PHerc0257 PHerc0268 PHerc0358 PHerc0800 PHerc0813 PHerc0826 PHerc1203 PHerc1218 PHerc1447 PHerc1545 PHerc0175A PHerc0175B PHerc0306B PHerc0483A PHerc0483B PHerc0490A PHerc0490B PHerc0846A PHerc0846B".split()
LV=5; out={}
for s in SC:
    vols=[c['Prefix'] for c in s3.list_objects_v2(Bucket=B,Prefix=f'{s}/volumes/',Delimiter='/').get('CommonPrefixes',[])]
    vols=[v for v in vols if '2.40' not in v]  # native ~9 um volume only
    v=sorted(vols)[-1]
    za=json.loads(s3.get_object(Bucket=B,Key=f'{v}{LV}/.zarray')['Body'].read())
    um=float(v.split('-')[1].replace('um',''))
    sh,ch,sep=za['shape'],za['chunks'],za.get('dimension_separator','.')
    assert za['compressor'] is None, za['compressor']
    nb=[ -(-a//c) for a,c in zip(sh,ch)]
    def get(ijk):
        try: return ijk, s3.get_object(Bucket=B,Key=f'{v}{LV}/'+sep.join(map(str,ijk)))['Body'].read()
        except s3.exceptions.NoSuchKey: return ijk, None
    nz=0; nvox=0
    with ThreadPoolExecutor(16) as ex:
        for ijk,b in ex.map(get, itertools.product(*[range(n) for n in nb])):
            if b is None: continue
            a=np.frombuffer(b,np.uint8); nz+=int((a>0).sum())
    vox_um=um*2**LV; cm3=nz*(vox_um*1e-4)**3
    out[s]=dict(volume=v,um=um,L5_shape=sh,nonzero_vox=nz,envelope_cm3=round(cm3,2),height_cm=round(sh[0]*vox_um*1e-4,2))
    print(s,out[s],flush=True)
json.dump(out,open('inventory/envelope_volumes.json','w'),indent=1)

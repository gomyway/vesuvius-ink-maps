# Stream a block of raw L0 chunks (compressor: null, 128^3 uint8) from the public bucket into a .npy; reports MB/s.
import boto3, numpy as np, sys, time, itertools
from botocore import UNSIGNED; from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
B='vesuvius-challenge-open-data'; vol, z0,y0,x0, n, out = sys.argv[1], *map(int,sys.argv[2:6]), sys.argv[6]
s3=boto3.client('s3',config=Config(signature_version=UNSIGNED,max_pool_connections=32))
C=128; a=np.zeros((n*C,)*3,np.uint8); nbytes=0; missing=0
def get(ijk):
    i,j,k=ijk; key=f'{vol}/0/{z0//C+i}/{y0//C+j}/{x0//C+k}'
    try: b=s3.get_object(Bucket=B,Key=key)['Body'].read()
    except s3.exceptions.NoSuchKey: return ijk,None
    return ijk,b
t=time.time()
with ThreadPoolExecutor(16) as ex:
    for (i,j,k),b in ex.map(get, itertools.product(range(n),repeat=3)):
        if b is None: missing+=1; continue
        nbytes+=len(b); a[i*C:(i+1)*C,j*C:(j+1)*C,k*C:(k+1)*C]=np.frombuffer(b,np.uint8).reshape(C,C,C)
dt=time.time()-t; np.save(out,a)
print(f'chunks {n**3} missing {missing} bytes {nbytes} sec {dt:.2f} MB_per_s {nbytes/dt/1e6:.1f} nonzero_frac {(a>0).mean():.3f} mean {a.mean():.1f}')

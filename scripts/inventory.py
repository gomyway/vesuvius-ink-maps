# Full anonymous S3 listing of eligible scrolls: per volume per OME-Zarr level, per segment, per representation.
import boto3, json, sys, time, collections
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
B='vesuvius-challenge-open-data'
s3=boto3.client('s3',config=Config(signature_version=UNSIGNED,max_pool_connections=64,retries={'max_attempts':10}))
SC="PHerc0125 PHerc0191 PHerc0211 PHerc0257 PHerc0268 PHerc0358 PHerc0800 PHerc0813 PHerc0826 PHerc1203 PHerc1218 PHerc1447 PHerc1545 PHerc0175A PHerc0175B PHerc0306B PHerc0483A PHerc0483B PHerc0490A PHerc0490B PHerc0846A PHerc0846B".split()
def ls_dirs(p):
    out=[];pg=s3.get_paginator('list_objects_v2')
    for r in pg.paginate(Bucket=B,Prefix=p,Delimiter='/'):
        out+= [c['Prefix'] for c in r.get('CommonPrefixes',[])]
    return out
def sumprefix(p, groupdepth=None):
    n=0;b=0;g=collections.Counter();gn=collections.Counter();pg=s3.get_paginator('list_objects_v2')
    for r in pg.paginate(Bucket=B,Prefix=p,PaginationConfig={'PageSize':1000}):
        for o in r.get('Contents',[]):
            n+=1;b+=o['Size']
            if groupdepth:
                k='/'.join(o['Key'][len(p):].split('/')[:groupdepth]); g[k]+=o['Size']; gn[k]+=1
    return p,n,b,dict(g),dict(gn)
t0=time.time();jobs=[]
for s in SC:
    for v in ls_dirs(f'{s}/volumes/'):
        for l in ls_dirs(v):
            jobs.append(('vol',s,l,None))
        jobs.append(('volroot',s,v,None))
    jobs.append(('seg',s,f'{s}/segments/',2))
    jobs.append(('rep',s,f'{s}/representations/',3))
print('jobs',len(jobs),file=sys.stderr)
res=[]
def run(j):
    kind,s,p,gd=j
    if kind=='volroot':  # only top-level files (.zattrs etc)
        return None
    r=sumprefix(p,gd); print(kind,p,r[1],r[2],round(time.time()-t0),file=sys.stderr,flush=True)
    return dict(kind=kind,scroll=s,prefix=p,objects=r[1],bytes=r[2],groups=r[3],group_objects=r[4])
with ThreadPoolExecutor(24) as ex:
    for r in ex.map(run,jobs):
        if r: res.append(r)
json.dump(dict(listed_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),bucket=B,seconds=round(time.time()-t0),results=res),open(sys.argv[1],'w'),indent=0)

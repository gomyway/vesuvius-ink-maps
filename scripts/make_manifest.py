# Manifest of the publishable outputs: per top-level output directory (segment dir, overlay zarr), file count, total bytes
# and a tree sha256 = sha256 over the sorted lines "<relpath> <sha256(file)>\n". Also lists single files with their sha256.
# usage: python make_manifest.py <out.json> <root1> [<root2> ...]
import sys, os, hashlib, json, time
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
out=sys.argv[1]; res={'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'entries':[]}
for root in sys.argv[2:]:
    units=[]
    for scroll in sorted(os.listdir(root)):
        p=os.path.join(root,scroll)
        if os.path.isfile(p): units.append(p); continue
        for seg in sorted(os.listdir(p)):
            q=os.path.join(p,seg)
            if seg=='vc3d_overlays':
                for z in sorted(os.listdir(q)): units.append(os.path.join(q,z))
            else: units.append(q)
    for u in units:
        if os.path.isfile(u):
            res['entries'].append(dict(path=os.path.relpath(u,os.path.dirname(root)),kind='file',files=1,bytes=os.path.getsize(u),sha256=sha(u))); continue
        lines=[]; n=0; b=0
        for d,_,fs in os.walk(u):
            for f in fs:
                fp=os.path.join(d,f); n+=1; b+=os.path.getsize(fp); lines.append(f'{os.path.relpath(fp,u)} {sha(fp)}\n')
        lines.sort(); th=hashlib.sha256(''.join(lines).encode()).hexdigest()
        res['entries'].append(dict(path=os.path.relpath(u,os.path.dirname(root)),kind='tree',files=n,bytes=b,tree_sha256=th))
        print(u,n,b,th,flush=True)
res['total_files']=sum(e['files'] for e in res['entries']); res['total_bytes']=sum(e['bytes'] for e in res['entries'])
json.dump(res,open(out,'w'),indent=1); print('wrote',out,res['total_files'],res['total_bytes'])

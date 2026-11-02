# Markdown tables from the full anonymous S3 listing (logs/inventory_listing.log lines: kind prefix objects bytes t; or inventory/inventory_full.json).
import json, os, collections
GP="PHerc0125 PHerc0191 PHerc0211 PHerc0257 PHerc0268 PHerc0358 PHerc0800 PHerc0813 PHerc0826 PHerc1203 PHerc1218 PHerc1447 PHerc1545".split()
FL="PHerc0175A PHerc0175B PHerc0306B PHerc0483A PHerc0483B PHerc0490A PHerc0490B PHerc0846A PHerc0846B".split()
rows=[]
if os.path.exists('inventory/inventory_full.json'):
    for r in json.load(open('inventory/inventory_full.json'))['results']: rows.append((r['kind'],r['prefix'],r['objects'],r['bytes'],r.get('groups',{})))
else:
    for l in open('logs/inventory_listing.log'):
        p=l.split()
        if len(p)==5 and p[0] in('vol','seg','rep'): rows.append((p[0],p[1],int(p[2]),int(p[3]),{}))
G=lambda b: b/1e9
vol=collections.defaultdict(dict); seg={}; rep={}; segg={}
for k,p,n,b,g in rows:
    s=p.split('/')[0]
    if k=='vol': v=p.split('/')[2]; vol[(s,v)][int(p.split('/')[3])]=(n,b)
    elif k=='seg': seg[s]=(n,b); segg[s]=g
    else: rep[s]=(n,b)
print('### Volumes: GB per OME-Zarr level (decimal GB, present chunks only; full listing)\n')
print('| scroll | prize | volume | L0 | L1 | L2 | L3 | L4 | L5 | total | L0 objects |'); print('|'+'---|'*11)
tot=collections.Counter()
for (s,v),L in sorted(vol.items(), key=lambda x:(x[0][0] not in GP, x[0])):
    pr='GP+FL' if s in GP and s!='PHerc1447' else ('GP' if s=='PHerc1447' else 'FL')
    cells=[f"{G(L[i][1]):.2f}" if i in L else 'listing' for i in range(6)]; t=sum(L[i][1] for i in L)
    print(f"| {s} | {pr} | {v.replace('-masked.zarr','')} | "+' | '.join(cells)+f" | {G(t):.1f} | {L.get(0,(0,0))[0]:,} |")
    key='2.4um' if '2.40' in v else 'native'
    for i in L: tot[(key,i,'GP' if s in GP else 'FLonly')]+=L[i][1]
print('\nTotals (GB):')
for key in ['native','2.4um']:
    for grp in ['GP','FLonly']:
        print(f"- {key} {grp}: "+', '.join(f"L{i} {G(tot[(key,i,grp)]):.1f}" for i in range(6))+f"; all levels {G(sum(tot[(key,i,grp)] for i in range(6))):.1f}")
print('\n### Segments and representations per scroll (GB)\n'); print('| scroll | segments (objects / GB) | representations (objects / GB) |'); print('|---|---|---|')
for s in GP+FL:
    a=seg.get(s); b=rep.get(s)
    print(f"| {s} | {(f'{a[0]:,} / {G(a[1]):.2f}') if a else 'listing'} | {(f'{b[0]:,} / {G(b[1]):.1f}') if b else 'listing'} |")
for s,g in segg.items():
    if g:
        print(f'\n{s} segment dirs (GB):', ', '.join(f"{k} {G(v):.3f}" for k,v in sorted(g.items())))

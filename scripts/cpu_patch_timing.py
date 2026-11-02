# How fast is surface_m7 on CPU? Times forward passes of one 128^3 and one 192^3 patch with all cores, fp32, to decide
# whether CPU inference over a whole segment bbox is feasible without a GPU. Reuses the model build from run_surface_m7_lowmem.py.
import sys, json, time, torch, numpy as np, pydoc
torch.set_num_threads(torch.get_num_threads()); md=sys.argv[1]
plans=json.load(open(f'{md}/plans.json')); conf=plans['configurations']['3d_fullres']; arch=conf['architecture']
cls=pydoc.locate(arch['network_class_name']); kw=dict(arch['arch_kwargs'])
for k in arch.get('_kw_requires_import',[]): kw[k]=pydoc.locate(kw[k]) if kw[k] else None
net=cls(input_channels=1,num_classes=plans.get('num_classes',2) if 'num_classes' in plans else 2,**kw) if False else None
# fall back to importing the runner's builder by exec of its top part is brittle; instead replicate minimal build via the runner module
sys.argv=[sys.argv[0], md, 'x', 'y']  # satisfy its argparse
src=open('scripts/run_surface_m7_lowmem.py').read().split('\nfp=P[')[0]  # everything up to the volume load (model build)
src=src.replace("dev='cuda'","dev='cpu'")
g={}; exec(compile(src,'lowmem_head','exec'),g); net=g['net']
print('threads',torch.get_num_threads(),flush=True)
for p in (128,192):
    x=torch.randn(1,1,p,p,p)
    with torch.inference_mode():
        net(x); t=time.time(); n=2
        for _ in range(n): net(x)
        dt=(time.time()-t)/n
    print(json.dumps(dict(patch=p,sec_per_patch=round(dt,2),Mvox_per_s=round(p**3/dt/1e6,3))),flush=True)

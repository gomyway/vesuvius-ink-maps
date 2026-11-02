# Minimal nnU-Net-v2-equivalent sliding-window inference for scrollprize/surface_m7_nnunet (Apache-2.0) on a uint8 ZYX .npy.
# Builds ResidualEncoderUNet from plans.json, CTNormalization from plans, gaussian-weighted tiles at step 0.5, fp16 autocast.
# No mirroring TTA (nnU-Net default would use 8x mirroring; the comparison baseline used the same setting). Reports torch peak memory and timing.
import sys, json, time, argparse, numpy as np, torch, pydoc
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
ap=argparse.ArgumentParser(); ap.add_argument('model_dir'); ap.add_argument('inp'); ap.add_argument('out')
ap.add_argument('--patch',type=int,default=192); ap.add_argument('--step',type=float,default=0.5); ap.add_argument('--batch',type=int,default=1)
a=ap.parse_args(); dev='cuda'
P=json.load(open(f'{a.model_dir}/plans.json')); c=P['configurations']['3d_fullres']; ak=dict(c['architecture']['arch_kwargs'])
for k in c['architecture']['_kw_requires_import']:
    if ak.get(k): ak[k]=pydoc.locate(ak[k])
net=ResidualEncoderUNet(input_channels=1,num_classes=2,deep_supervision=False,**ak)
ck=torch.load(f'{a.model_dir}/fold_0/checkpoint_best.pth',map_location='cpu',weights_only=False)
sd={k.replace('_orig_mod.',''):v for k,v in ck['network_weights'].items()}
print('load',net.load_state_dict(sd,strict=False)); net=net.to(dev).eval()
fp=P['foreground_intensity_properties_per_channel']['0']
x=np.load(a.inp).astype(np.float32); x=np.clip(x,fp['percentile_00_5'],fp['percentile_99_5']); x=(x-fp['mean'])/max(fp['std'],1e-8)
D=x.shape; p=a.patch; st=max(1,int(p*a.step))
pos=lambda L: sorted(set(list(range(0,max(1,L-p+1),st))+[max(0,L-p)]))
tiles=[(z,y,xx) for z in pos(D[0]) for y in pos(D[1]) for xx in pos(D[2])]
g=np.exp(-((np.arange(p)-(p-1)/2)**2)/(2*(p/8)**2)); g=np.einsum('i,j,k->ijk',g,g,g); g=torch.tensor(g/g.max(),dtype=torch.float32,device=dev).clamp_min(1e-4)
xt=torch.from_numpy(x).to(dev); acc=torch.zeros((2,)+D,device=dev); w=torch.zeros(D,device=dev)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t=time.time()
with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
    for i in range(0,len(tiles),a.batch):
        tb=tiles[i:i+a.batch]; inp=torch.stack([xt[z:z+p,y:y+p,q:q+p] for z,y,q in tb])[:,None]
        o=torch.softmax(net(inp).float(),1)
        for (z,y,q),oo in zip(tb,o):
            acc[:,z:z+p,y:y+p,q:q+p]+=oo*g; w[z:z+p,y:y+p,q:q+p]+=g
torch.cuda.synchronize(); dt=time.time()-t
prob=(acc[1]/w).clamp(0,1); np.save(a.out,(prob*255).round().byte().cpu().numpy())
vox=np.prod(D); r=dict(shape=D,patch=p,step=a.step,batch=a.batch,tiles=len(tiles),loop_s=round(dt,2),voxels=int(vox),Mvox_per_s=round(vox/dt/1e6,3),
  torch_max_allocated_MiB=round(torch.cuda.max_memory_allocated()/2**20),torch_max_reserved_MiB=round(torch.cuda.max_memory_reserved()/2**20),surface_frac=None)
r["surface_frac"]=float((prob>0.5).float().mean()); print(json.dumps(r))

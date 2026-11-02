# Copy a koine_machines checkpoint with a different inference crop (patch) size in its config. Weights unchanged.
import sys, torch
src, dst, hw = sys.argv[1], sys.argv[2], int(sys.argv[3])
ck = torch.load(src, map_location='cpu', weights_only=False)
def walk(d, path=''):
    if isinstance(d, dict):
        for k, v in d.items():
            if k in ('crop_size', 'patch_size') and isinstance(v, (list, tuple)) and len(v) == 3:
                print(path + k, v, '->', [v[0], hw, hw]); d[k] = [v[0], hw, hw]
            elif k not in ('model', 'state_dict', 'model_state_dict', 'ema'): walk(v, path + k + '.')
walk(ck); torch.save(ck, dst)

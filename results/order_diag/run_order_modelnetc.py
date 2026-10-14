"""External wrapper: run the unchanged ModelNet-C runner with every DataLoader's dataset permuted.
Permutation = np.random.default_rng([seed, len(dataset)]).permutation(len) (so train and every test set get a fixed random order).
seed 0 = original order. No change to runner source."""
import sys, os, numpy as np, torch
code_dir, seed = sys.argv[1], int(sys.argv[2]); rest = sys.argv[3:]
sys.path.insert(0, code_dir); os.chdir(code_dir); sys.argv = ['run_cls_modelnetcv2.py'] + rest
import run_cls_modelnetcv2 as R
_DL = R.DataLoader
def DL(ds, *a, **k):
    if seed != 0:
        p = np.random.default_rng([seed, len(ds)]).permutation(len(ds))
        ds = torch.utils.data.Subset(ds, p.tolist())
    return _DL(ds, *a, **k)
R.DataLoader = DL
torch.manual_seed(3407)
R.main()

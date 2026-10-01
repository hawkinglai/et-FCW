# Environment

## Verified information from completed work

- The recorded interpreter was a Conda `pytorch` environment. At archival audit time that interpreter reported Python **3.12.2**; the exact Python patch version at every historical run was not logged.
- The RISP result summary recorded PyTorch **2.7.1+cu118**, CUDA build **11.8**, and an **NVIDIA RTX A5000** selected as GPU 0.
- RISP used transforms from a cached PyTorch3D **0.7.8** package exposed through `PYTHONPATH`. The cache name indicates a specific build; compatibility with another PyTorch build is not guaranteed.

## Dependencies inferred from imports; exact versions unavailable

Historical code imports NumPy, h5py, tqdm, scikit-learn, pandas, `pointnet2_ops`, and (for RISP) `pytorch3d.transforms`. The corrected local source imports PyTorch, NumPy, h5py, tqdm, scikit-learn and `pointnet2_ops`. No complete historical environment lock exists for these standalone reproduction runs. Do not interpret this list as a tested installation recipe.

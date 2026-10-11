# RISP rotation — Table V

Use the CUDA environment in [ENVIRONMENT.md](../../ENVIRONMENT.md); run commands from the release root.

```bash
export TFCW_MODELNET_NORMAL_ROOT=/path/to/modelnet40_normal_resampled
bash experiments/risp_rotation/run.sh
```

Requires installed PyTorch3D 0.7.8 and pointnet2_ops.
Data: shape-name/train/test lists, category xyz/normal text files, and `modelnet40_{train,test}_1024pts_fps.dat` caches under the root. Missing caches trigger NumPy FPS preprocessing; regeneration can change results.
Parameters: batch 32; points [1024,512,256,128]; K [95,80,100,95]; metric 2; Torch seed 3407. The runner executes all four conditions internally; xyz and normals rotate together.
γ protocol: select separately on each labeled test condition over 0,2,…,9998.
RISP ignores surface and rescale; split is unused. Training drops the last batch (9824 shapes); testing uses all 2468. Batch-wide knnxyz standard deviation is not used on the RISP path.

| Condition | Accuracy (%) | γ |
|---|---:|---:|
| z/z | 73.9870 | 8628 |
| z/SO(3) | 74.0276 | 7192 |
| SO(3)/SO(3) | 73.9465 | 7186 |
| none/none | 74.0681 | 7456 |

Paper: 73.9% for each condition.
[Summary](../../results/risp_rotation/summary.json) · [Installed-dependency acceptance](../../results/risp_rotation/port_check_installed.json).

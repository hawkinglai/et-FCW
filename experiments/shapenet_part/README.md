# ShapeNet-Part plain t-FCW — Table IV

Use the CUDA environment in [ENVIRONMENT.md](../../ENVIRONMENT.md); run commands from the release root.

```bash
export TFCW_SHAPENET_DATALOADERS_ROOT=/path/to/dataloaders
bash experiments/shapenet_part/run.sh
```

Data: `$TFCW_SHAPENET_DATALOADERS_ROOT/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`, containing `synsetoffset2category.txt`, `train_test_split/shuffled_{train,val,test}_file_list.json`, and category text files with xyz, normals and part labels.
Parameters: trainval → test; batch 128; points [1024,512,256,128]; encoder K [105,105,110,120]; decoder K3; metric 2; rescale 0.2; knnxyz; Torch seed 3407.
γ protocol: fixed γ230.
Per-sample std in `pcs_knnxyz` ([knnxyz correction](../../docs/knnxyz_std_correction.md)). NumPy sampling uses replacement without an explicit top-level NumPy seed; retain eight DataLoader workers. Normals do not enter the model.
Result (per-sample std, pre-fix `c8dd640`): **70.35% mIoU** (bz 32: 70.21; bz 1: 70.26); paper 70.50%. Batch-wide std (`51686e4`, pre-fix): 70.18% in each of three acceptance runs.

**Note (2026-11, stage-0 feature ordering):** Fixed stage-0 feature-coordinate misalignment (`models/surfaces.py`, both `pcs_knnxyz` and `pcs_geobp`). With the fix (`bank = trainval`, script metric): per-sample std (`c8dd640` base) gives instance mIoU **70.33 / 70.37** and class mIoU **67.07 / 67.07** (batch 1 / 128), vs. 70.26 / 70.35 and 67.65 / 67.70 before the fix; legacy batch-wide std (`51686e4` base) gives instance mIoU 70.33 / 70.10 and class mIoU 67.07 / 66.29 (batch 1 / 128), vs. 70.26 / 70.18 and 67.65 / 67.32 before the fix. Under the standard PointNet++ per-category mIoU metric on the same post-fix `c8dd640` predictions: instance mIoU 61.77 / 61.85 and class mIoU 63.45 / 63.67 (batch 1 / 128).
[Summary](../../results/shapenet_part/summary.json) · [Acceptance](../../results/shapenet_part/port_check.json).

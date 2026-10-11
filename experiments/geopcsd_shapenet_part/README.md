# Corrected GeoPCSD — ShapeNet-Part

Use the CUDA environment in [ENVIRONMENT.md](../../ENVIRONMENT.md); run commands from the release root.

```bash
export TFCW_SHAPENET_PART_ROOT=/path/to/shapenetcore_partanno_segmentation_benchmark_v0_normal
bash experiments/geopcsd_shapenet_part/run.sh
```

Data: category xyz/normal/part-label text files, `synsetoffset2category.txt`, and `train_test_split/shuffled_{train,val,test}_file_list.json` directly under the root.
Parameters: trainval → test; batch 128; points [1024,512,256,128]; encoder K [105,105,110,120]; decoder K3; metric 2; rescale 0.2; geobp; Torch seed 3407.
γ protocol: fixed γ230.
NumPy sampling uses replacement without an explicit top-level NumPy seed; keep eight DataLoader workers and batch 128. Batch-wide knnxyz standard deviation is not used by this surface.
Result: **68.68% mIoU**, corrected local reconstruction; [neighbor correction](../../docs/geopcsd_correction.md).
[Summary](../../results/geopcsd_shapenet_part/summary.json) · [Saved-result consistency check](../../results/geopcsd_shapenet_part/port_check.json).

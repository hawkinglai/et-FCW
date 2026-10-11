# ShapeNet-C plain t-FCW — Table VIII

Use the CUDA environment in [ENVIRONMENT.md](../../ENVIRONMENT.md); run commands from the release root.

```bash
export TFCW_SHAPENET_DATALOADERS_ROOT=/path/to/dataloaders
export TFCW_SHAPENETC_ROOT=/path/to/shapenet_c
bash experiments/shapenetc/run.sh
```

Clean training data: `$TFCW_SHAPENET_DATALOADERS_ROOT/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`, with category text files, `synsetoffset2category.txt` and `train_test_split/shuffled_{train,val,test}_file_list.json`.
Corruption data: `$TFCW_SHAPENETC_ROOT/clean.h5` and `{jitter,add_global}_{0,1,2,3,4}.h5`; keys `data`, `label`, `pid` (or `part_label`). Missing part labels become zeros in the original loader. File indices 0–4 correspond to S1–S5.
Parameters: batch 128; points [1024,512,256,128]; encoder K [105,105,110,120]; decoder K3; metric 2; rescale 0.2; Torch seed 3407.
γ protocol: fixed γ230.
Batch-wide standard deviation makes features depend on batch composition. Non-1024 clouds are sampled with replacement with matching xyz/part-label indices; no explicit top-level NumPy seed. Test category labels select the part memory bank.
The original runner also evaluates a second surface (`geobp`), whose output is not part of the reported results and should be ignored.

| knnxyz corruption | Clean | S1 | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|---:|---:|
| jitter | 62.78 | 62.49 | 61.74 | 60.83 | 59.34 | 57.57 |
| add_global | 62.78 | 61.63 | 60.67 | 59.74 | 59.02 | 58.20 |

The clean value is the ShapeNet-C `clean.h5` baseline (62.78% mIoU), evaluated with a different loader from ShapeNet-Part’s 70.18% mIoU. Ten corrupted values and both clean baselines match to two decimals. `add_global` corresponds numerically to paper Global.
[Summary](../../results/shapenetc/summary.json) · [Acceptance](../../results/shapenetc/port_check.json).

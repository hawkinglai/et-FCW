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
Per-sample std in `pcs_knnxyz` ([knnxyz correction](../../docs/knnxyz_std_correction.md)). The table below is the **batch-wide** result that matched the paper; per-sample knnxyz values (bz 128) are 64.00, 63.79, 62.90, 61.91, 60.26, 58.35 (jitter) and 64.00, 62.56, 61.35, 60.23, 59.34, 58.43 (add_global). Non-1024 clouds are sampled with replacement with matching xyz/part-label indices; no explicit top-level NumPy seed. Test category labels select the part memory bank.
The original runner also evaluates a second surface (`geobp`), whose output is not part of the reported results and should be ignored.

| knnxyz corruption | Clean | S1 | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|---:|---:|
| jitter | 64.00 | 63.79 | 62.90 | 61.91 | 60.26 | 58.35 |
| add_global | 64.00 | 62.56 | 61.35 | 60.23 | 59.34 | 58.43 |

The clean value is the ShapeNet-C `clean.h5` baseline (62.78% mIoU), evaluated with a different loader from ShapeNet-Part (70.35% mIoU here). `add_global` corresponds numerically to paper Global. Per-sample values are 0.23–1.30 pp above paper Table VIII (the clean value differs by 1.22).

**Legacy (batch-wide std, exact paper match):** status verified at `51686e4`; to reproduce Table VIII exactly, `git checkout 51686e4`. There the ten corrupted values and both clean baselines matched to two decimals:

| knnxyz corruption | Clean | S1 | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|---:|---:|
| jitter | 62.78 | 62.49 | 61.74 | 60.83 | 59.34 | 57.57 |
| add_global | 62.78 | 61.63 | 60.67 | 59.74 | 59.02 | 58.20 |
[Summary](../../results/shapenetc/summary.json) · [Acceptance](../../results/shapenetc/port_check.json).

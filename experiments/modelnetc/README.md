# ModelNet-C plain t-FCW — Table VII

Use the CUDA environment in [ENVIRONMENT.md](../../ENVIRONMENT.md); run commands from the release root.

```bash
export TFCW_MODELNET40_ROOT=/path/to/modelnet-parent
export TFCW_MODELNETC_ROOT=/path/to/modelnet_c
bash experiments/modelnetc/run.sh
```

Clean data: `$TFCW_MODELNET40_ROOT/data/modelnet40_ply_hdf5_2048/ply_data_{train,test}*.h5` with `data`/`label`. Corruptions: `$TFCW_MODELNETC_ROOT/{jitter,add_global}_{0,1,2,3,4}.h5` with the same keys. File indices 0–4 are S1–S5; clean evaluation uses ModelNet40 test data.
Parameters: batch 32; points [1024,512,256,128]; K [110,110,110,110]; metric 2; rescale 0.8; knnxyz; Torch seed 3407.
γ protocol: select separately on each labeled test set over 0,10,…,9990; the original runner does not print γ.
Per-sample std in `pcs_knnxyz` ([knnxyz correction](../../docs/knnxyz_std_correction.md)). The table below is the **batch-wide** result that matched the paper; per-sample values are in [summary.json](../../results/modelnetc/summary.json) and do not match. Non-1024 clouds are resampled with replacement from the full cloud (including appended points); no explicit top-level NumPy seed. Exceptions print skipped severities; these are not accepted results.

| Mode / corruption | Clean | S1 | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|---:|---:|
| global / jitter | 81.69 | 80.92 | 77.84 | 68.03 | 58.63 | 51.42 |
| local / jitter | 83.31 | 83.06 | 82.29 | 78.69 | 74.11 | 68.23 |
| united / jitter | 84.36 | 84.12 | 82.90 | 78.65 | 72.16 | 65.40 |
| global / add_global | 81.69 | 75.49 | 67.18 | 58.71 | 51.82 | 45.75 |
| local / add_global | 83.31 | 47.29 | 35.17 | 25.53 | 21.64 | 19.29 |
| united / add_global | 84.36 | 59.56 | 48.99 | 41.09 | 34.76 | 29.21 |

`add_global` corresponds numerically to paper Global. Per-sample values differ from paper Table VII by up to 5.35 pp.

**Legacy (batch-wide std, exact paper match):** status verified at `51686e4`; to reproduce Tables VII exactly, `git checkout 51686e4`. There all 30 corruptions and three unique clean accuracies matched to two decimals:

| Mode / corruption | Clean | S1 | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|---:|---:|
| global / jitter | 81.40 | 80.27 | 75.41 | 64.51 | 53.93 | 46.07 |
| local / jitter | 83.39 | 82.70 | 82.05 | 78.48 | 73.14 | 66.65 |
| united / jitter | 84.72 | 84.40 | 82.54 | 78.28 | 69.69 | 60.90 |
| global / add_global | 81.40 | 73.50 | 66.17 | 58.51 | 51.26 | 47.16 |
| local / add_global | 83.39 | 44.69 | 33.43 | 23.58 | 20.14 | 17.42 |
| united / add_global | 84.72 | 58.02 | 45.18 | 36.51 | 29.70 | 24.84 |
[Summary](../../results/modelnetc/summary.json) · [Acceptance](../../results/modelnetc/port_check.json).

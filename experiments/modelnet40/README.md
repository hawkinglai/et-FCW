# ModelNet40 plain t-FCW — Table I

Use the CUDA environment in [ENVIRONMENT.md](../../ENVIRONMENT.md); run commands from the release root.

```bash
export TFCW_MODELNET40_ROOT=/path/to/modelnet-parent
bash experiments/modelnet40/run.sh
```

Data: `$TFCW_MODELNET40_ROOT/data/modelnet40_ply_hdf5_2048/ply_data_{train,test}*.h5`, with `data` and `label` datasets.
Parameters: batch 32; points [1024,512,256,128]; K [66,66,66,66]; metric 2; rescale 0.8; knnxyz; Torch seed 3407.
γ protocol: select the first best value on labeled test data over 0,2,…,9998.
`pcs_knnxyz` uses a per-sample standard deviation ([knnxyz correction](../../docs/knnxyz_std_correction.md)); features no longer depend on batch composition. First 1024 points are retained; the training shuffle is a seeded copy (seed 3407).
`--split` is ignored for mn40; unknown CLI arguments are ignored. The second printed zero accuracy is an unused accumulator.
Result (per-sample std): **85.0486%**, γ5790 (bz 1: 85.0486%, γ5792); paper 84.8%. Batch-wide std regression (51686e4): 85.1297%, γ6090.

ScanObjectNN xyz uses the same runner: `bash experiments/modelnet40/run_scanobjectnn.sh obj_bg` (also `obj_only`, `pb_t50_rs`; needs `TFCW_SCANOBJECTNN_DATALOADERS_ROOT`). Batch 64 / 64 / 32, K 90, metric 2, rescale 0.8: OBJ-BG **65.92** (γ4446), OBJ-ONLY **72.29** (γ2276), PB-T50-RS **55.14** (γ3572); paper 66.3 / 71.9 / 55.6. [Summary](../../results/scanobjectnn_xyz/summary.json).
[Summary](../../results/modelnet40/summary.json) · [Acceptance](../../results/modelnet40/port_check.json).

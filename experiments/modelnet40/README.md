# ModelNet40 plain t-FCW — Table I

Use the CUDA environment in [ENVIRONMENT.md](../../ENVIRONMENT.md); run commands from the release root.

```bash
export TFCW_MODELNET40_ROOT=/path/to/modelnet-parent
bash experiments/modelnet40/run.sh
```

Data: `$TFCW_MODELNET40_ROOT/data/modelnet40_ply_hdf5_2048/ply_data_{train,test}*.h5`, with `data` and `label` datasets.
Parameters: batch 32; points [1024,512,256,128]; K [66,66,66,66]; metric 2; rescale 0.8; knnxyz; Torch seed 3407.
γ protocol: select the first best value on labeled test data over 0,2,…,9998.
Batch-wide standard deviation makes features depend on batch size/composition; keep batch 32. Original unsorted file enumeration, first 1024 points, and training-point shuffling are retained.
`--split` is ignored for mn40; unknown CLI arguments are ignored. The second printed zero accuracy is an unused accumulator.
Result: **85.1297%**, γ6090; paper 84.8%.
[Summary](../../results/modelnet40/summary.json) · [Acceptance](../../results/modelnet40/port_check.json).

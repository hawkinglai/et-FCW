# Corrected GeoPCSD — ScanObjectNN

Use the CUDA environment in [ENVIRONMENT.md](../../ENVIRONMENT.md); run commands from the release root.

```bash
export TFCW_SCANOBJECTNN_DATALOADERS_ROOT=/path/to/dataloaders
bash experiments/geopcsd_scanobjectnn/run.sh obj_bg
bash experiments/geopcsd_scanobjectnn/run.sh obj_only
bash experiments/geopcsd_scanobjectnn/run.sh pb_t50_rs
```

Data: `$TFCW_SCANOBJECTNN_DATALOADERS_ROOT/data/h5_files/`, with the OBJ-BG, OBJ-ONLY and PB-T50-RS HDF5 splits; see loader filenames in `dataloaders/ScanObjectNN.py`.
Parameters: points [1024,512,256,128]; K [90,90,90,90]; metric 2; rescale 0.8; geobp; Torch seed 3407. Batch 64 for OBJ-BG/OBJ-ONLY, 32 for PB-T50-RS.
γ protocol: select on labeled test data over 0,2,…,9998.
Keep the recorded batch sizes and original loader ordering/sampling. Batch-wide knnxyz standard deviation is not used by this surface.

| Split | Accuracy (%) | γ |
|---|---:|---:|
| OBJ-BG | 69.7074 | 5434 |
| OBJ-ONLY | 72.9776 | 4152 |
| PB-T50-RS | 58.0500 | 3550 |

Status: corrected local reconstruction; [neighbor correction](../../docs/geopcsd_correction.md).
[Summary](../../results/geopcsd_scanobjectnn/summary.json) · [Saved-result consistency check](../../results/geopcsd_scanobjectnn/port_check.json).

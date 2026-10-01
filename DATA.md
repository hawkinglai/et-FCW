# Data

No raw dataset or generated point-cloud cache is included in this source-only release. The saved results were produced using local inputs whose acquisition and redistribution terms were not fully documented. Do not infer dataset permissions from the code. The external `reconstruction/` companion, if made available under a suitable license, also requires separately obtained data.

| Dataset | Public experiment | Expected placement or path variable | Acquisition note |
|---|---|---|---|
| ModelNet40 HDF5 | Plain classification and ModelNet-C memory bank | Historical `reconstruction/dataloaders/data/modelnet40_ply_hdf5_2048/` | Exact acquisition provenance unresolved. |
| ShapeNet-Part normal benchmark | Plain/GeoPCSD part segmentation and ShapeNet-C | Historical dataset path; for corrected local source set `TFCW_SHAPENET_PART_ROOT` to the benchmark directory | Original train/val/test split files are required. Exact acquisition provenance unresolved. |
| ModelNet40 normals/cached FPS | RISP rotation | Historical `reconstruction/dataloaders/data/modelnet40_normal_resampled/` | Preprocessing provenance incomplete. |
| ModelNet-C | Table VII | Historical `reconstruction/dataloaders/data/modelnet_c/` | `add_global` maps functionally to paper Global; filename is not in the paper. |
| ShapeNet-C | Table VIII | Historical `reconstruction/dataloaders/data/shapenet_c/` | Same `add_global` caveat. |
| ScanObjectNN HDF5 | Corrected GeoPCSD classification | Set `TFCW_SCANOBJECTNN_DATALOADERS_ROOT` to the directory containing `data/h5_files/` | OBJ-BG, OBJ-ONLY and PB-T50-RS files are required. Acquisition and redistribution terms need review. |
| S3DIS and 3D adversarial inputs | Documented partial or pilot work only | Not needed for main public result commands | Not shipped; protocol and mapping gaps remain. |

The separate forensic workspace contains tens of gigabytes of dataset and episode-cache files. They were not copied into this release. See [protocol notes](docs/protocol_notes.md) for the historical source dependency.

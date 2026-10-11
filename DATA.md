# Dataset layout

Datasets and generated point-cloud caches are not included. Set these variables before running the experiment commands; obtain datasets under their respective terms.

| Environment variable | Directory contents | Experiments |
|---|---|---|
| `TFCW_MODELNET40_ROOT` | `data/modelnet40_ply_hdf5_2048/ply_data_{train,test}*.h5`; keys `data`, `label` | ModelNet40, ModelNet-C |
| `TFCW_MODELNET_NORMAL_ROOT` | `modelnet40_shape_names.txt`, train/test lists, category xyz/normal files, `modelnet40_{train,test}_1024pts_fps.dat` | RISP |
| `TFCW_SHAPENET_DATALOADERS_ROOT` | `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/` | ShapeNet-Part, ShapeNet-C |
| `TFCW_SHAPENET_PART_ROOT` | ShapeNet-Part benchmark directory directly | Corrected GeoPCSD ShapeNet-Part |
| `TFCW_MODELNETC_ROOT` | `{jitter,add_global}_{0,1,2,3,4}.h5`; keys `data`, `label` | ModelNet-C |
| `TFCW_SHAPENETC_ROOT` | `clean.h5`, `{jitter,add_global}_{0,1,2,3,4}.h5`; keys `data`, `label`, `pid` or `part_label` | ShapeNet-C |
| `TFCW_SCANOBJECTNN_DATALOADERS_ROOT` | `data/h5_files/` with OBJ-BG, OBJ-ONLY, PB-T50-RS splits | Corrected GeoPCSD ScanObjectNN |

ShapeNet-Part directories contain `synsetoffset2category.txt`, `train_test_split/shuffled_{train,val,test}_file_list.json`, and category text files with xyz, normals and part labels. ModelNet-C uses clean ModelNet40 training/test data; ShapeNet-C uses clean text training data and its HDF5 files for evaluation.

Corruption file indices 0–4 correspond to S1–S5; `add_global` corresponds numerically to paper Global. Preserve the recorded RISP FPS caches: regeneration may change results. Dataset acquisition and redistribution permissions are separate from the code license.

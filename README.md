# t-FCW reproduction

Commands run from this directory with the dataset variables and CUDA environment configured. Accuracies and mIoUs are percentages; corruption lists follow S1–S5.

| Experiment | Paper table | Reproduced number(s) | Command |
|---|---|---|---|
| ModelNet40 plain | I | 85.0486; γ5790 | `bash experiments/modelnet40/run.sh` |
| ScanObjectNN plain xyz | II | OBJ-BG 65.92; OBJ-ONLY 72.29; PB-T50-RS 55.14 | `bash experiments/modelnet40/run_scanobjectnn.sh obj_bg` (also `obj_only`, `pb_t50_rs`) |
| ShapeNet-Part plain | IV | 70.35 mIoU; γ230 | `bash experiments/shapenet_part/run.sh` |
| RISP rotation | V | z/z 73.9870; z/SO(3) 74.0276; SO(3)/SO(3) 73.9465; none/none 74.0681 | `bash experiments/risp_rotation/run.sh` |
| ModelNet-C global / jitter | VII | Clean 81.69; 80.92, 77.84, 68.03, 58.63, 51.42 | `bash experiments/modelnetc/run.sh` |
| ModelNet-C local / jitter | VII | Clean 83.31; 83.06, 82.29, 78.69, 74.11, 68.23 | Same command |
| ModelNet-C united / jitter | VII | Clean 84.36; 84.12, 82.90, 78.65, 72.16, 65.40 | Same command |
| ModelNet-C global / add_global | VII | Clean 81.69; 75.49, 67.18, 58.71, 51.82, 45.75 | Same command |
| ModelNet-C local / add_global | VII | Clean 83.31; 47.29, 35.17, 25.53, 21.64, 19.29 | Same command |
| ModelNet-C united / add_global | VII | Clean 84.36; 59.56, 48.99, 41.09, 34.76, 29.21 | Same command |
| ShapeNet-C knnxyz / jitter | VIII | Clean 64.00; 63.79, 62.90, 61.91, 60.26, 58.35 | `bash experiments/shapenetc/run.sh` |
| ShapeNet-C knnxyz / add_global | VIII | Clean 64.00; 62.56, 61.35, 60.23, 59.34, 58.43 | Same command |
| Corrected GeoPCSD / ScanObjectNN | II, corrected local reconstruction | OBJ-BG 69.7074; OBJ-ONLY 72.9776; PB-T50-RS 58.0500 | `bash experiments/geopcsd_scanobjectnn/run.sh obj_bg` (also `obj_only`, `pb_t50_rs`) |
| Corrected GeoPCSD / ShapeNet-Part | IV, corrected local reconstruction | 68.68 mIoU | `bash experiments/geopcsd_shapenet_part/run.sh` |

**knnxyz std correction:** `pcs_knnxyz` uses a per-sample std on this branch, and every row below is what its command produces here. ModelNet-C and ShapeNet-C (Tables VII/VIII) matched the paper exactly with the previous batch-wide std; see the legacy note below. Details and all before/after numbers: [docs/knnxyz_std_correction.md](docs/knnxyz_std_correction.md).

**Legacy (batch-wide std; `git checkout 51686e4`).** Verified at `51686e4`: ModelNet40 85.1297 (γ6090, batch-wide std regression), ShapeNet-Part 70.18 mIoU, ModelNet-C global / local / united jitter and add_global (all 30 corrupted values and three clean values matched Table VII to two decimals) and ShapeNet-C knnxyz jitter / add_global (ten corrupted values and both clean baselines matched Table VIII to two decimals). Exact numbers are in the per-experiment READMEs and `results/*/summary.json` under `legacy_batch_wide_std`. To reproduce Tables VII/VIII exactly, `git checkout 51686e4`.

Data: [DATA.md](DATA.md) specifies dataset layouts and environment variables. Datasets and point-cloud caches are not included. `add_global` corresponds numerically to paper Global.

Environment: [ENVIRONMENT.md](ENVIRONMENT.md) lists tested versions and installation commands. Each experiment README records batch size, γ selection and sampling behavior; [protocol notes](docs/protocol_notes.md) and [result status](docs/reproduction_status.md) cover all seven experiments.

License: owner code is MIT under [LICENSE.txt](LICENSE.txt); [LICENSE_STATUS.md](LICENSE_STATUS.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and bundled license texts specify third-party attribution.

Citation: [CITATION.cff](CITATION.cff), *A Unified Non-Parametric and Interpretable Point Cloud Analysis Via t-FCW Graph Representation*, IEEE Transactions on Multimedia (2026), DOI 10.1109/TMM.2026.3716075.

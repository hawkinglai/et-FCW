# t-FCW reproduction

Commands run from this directory with the dataset variables and CUDA environment configured. Accuracies and mIoUs are percentages; corruption lists follow S1–S5.

| Experiment | Paper table | Reproduced number(s) | Command |
|---|---|---|---|
| ModelNet40 plain | I | 85.1297; γ6090 | `bash experiments/modelnet40/run.sh` |
| ShapeNet-Part plain | IV | 70.18 mIoU; γ230 | `bash experiments/shapenet_part/run.sh` |
| RISP rotation | V | z/z 73.9870; z/SO(3) 74.0276; SO(3)/SO(3) 73.9465; none/none 74.0681 | `bash experiments/risp_rotation/run.sh` |
| ModelNet-C global / jitter | VII | Clean 81.40; 80.27, 75.41, 64.51, 53.93, 46.07 | `bash experiments/modelnetc/run.sh` |
| ModelNet-C local / jitter | VII | Clean 83.39; 82.70, 82.05, 78.48, 73.14, 66.65 | Same command |
| ModelNet-C united / jitter | VII | Clean 84.72; 84.40, 82.54, 78.28, 69.69, 60.90 | Same command |
| ModelNet-C global / add_global | VII | Clean 81.40; 73.50, 66.17, 58.51, 51.26, 47.16 | Same command |
| ModelNet-C local / add_global | VII | Clean 83.39; 44.69, 33.43, 23.58, 20.14, 17.42 | Same command |
| ModelNet-C united / add_global | VII | Clean 84.72; 58.02, 45.18, 36.51, 29.70, 24.84 | Same command |
| ShapeNet-C knnxyz / jitter | VIII | 62.49, 61.74, 60.83, 59.34, 57.57 | `bash experiments/shapenetc/run.sh` |
| ShapeNet-C knnxyz / add_global | VIII | 61.63, 60.67, 59.74, 59.02, 58.20 | Same command |
| Corrected GeoPCSD / ScanObjectNN | II, corrected local reconstruction | OBJ-BG 69.7074; OBJ-ONLY 72.9776; PB-T50-RS 58.0500 | `bash experiments/geopcsd_scanobjectnn/run.sh obj_bg` (also `obj_only`, `pb_t50_rs`) |
| Corrected GeoPCSD / ShapeNet-Part | IV, corrected local reconstruction | 68.68 mIoU | `bash experiments/geopcsd_shapenet_part/run.sh` |
| [ModelNet-C with a single train-selected γ (thesis protocol)](../mnc_fixedgamma_v1/summary.md) — global | NEW EXPERIMENT — NOT A REPRODUCTION | γ9500; mean(S1–S5): jitter 63.1848, add_global 55.9643 | `cd ../mnc_fixedgamma_v1 && python extract_gate.py && python cv_evaluate.py` |
| ↳ local | NEW EXPERIMENT — NOT A REPRODUCTION | γ5260; mean(S1–S5): jitter 76.2723, add_global 27.0989 | Same external command |
| ↳ united | NEW EXPERIMENT — NOT A REPRODUCTION | γ5440; mean(S1–S5): jitter 74.9190, add_global 38.0470 | Same external command |

Data: [DATA.md](DATA.md) specifies dataset layouts and environment variables. Datasets and point-cloud caches are not included. `add_global` corresponds numerically to paper Global.

Environment: [ENVIRONMENT.md](ENVIRONMENT.md) lists tested versions and installation commands. Each experiment README records batch size, γ selection and sampling behavior; [protocol notes](docs/protocol_notes.md) and [result status](docs/reproduction_status.md) cover all seven experiments.

License: owner code is MIT under [LICENSE.txt](LICENSE.txt); [LICENSE_STATUS.md](LICENSE_STATUS.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and bundled license texts specify third-party attribution.

Citation: [CITATION.cff](CITATION.cff), *A Unified Non-Parametric and Interpretable Point Cloud Analysis Via t-FCW Graph Representation*, IEEE Transactions on Multimedia (2026), DOI 10.1109/TMM.2026.3716075.

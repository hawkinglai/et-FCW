# Result status

| Experiment | Status | Accepted or saved numbers (%) |
|---|---|---|
| ModelNet40, Table I | Corrected reproduction | 85.0486; γ5790; paper 84.8 |
| ScanObjectNN xyz, Table II | Verified reproduction | OBJ-BG 65.92; OBJ-ONLY 72.29; PB-T50-RS 55.14; paper 66.3 / 71.9 / 55.6, all within 0.5 pp |
| ShapeNet-Part, Table IV | Corrected rerun | 70.35 mIoU; paper 70.50 |
| RISP rotation, Table V | Verified reproduction | z/z 73.9870; z/SO(3) 74.0276; SO(3)/SO(3) 73.9465; none/none 74.0681; paper 73.9 each |
| ModelNet-C, Table VII | Verified at 51686e4; per-sample re-run on this branch | This branch: per-sample values differ from the paper by up to 5.35 pp (global jitter clean 81.69; S1–S5 80.92, 77.84, 68.03, 58.63, 51.42); full tables in `experiments/modelnetc/README.md` |
| ShapeNet-C, Table VIII | Verified at 51686e4; per-sample re-run on this branch | This branch: knnxyz jitter 64.00; 63.79, 62.90, 61.91, 60.26, 58.35 and add_global 64.00; 62.56, 61.35, 60.23, 59.34, 58.43, i.e. 0.23–1.30 pp above the paper |
| GeoPCSD ScanObjectNN | Corrected local reconstruction | OBJ-BG 69.7074; OBJ-ONLY 72.9776; PB-T50-RS 58.0500 |
| GeoPCSD ShapeNet-Part | Corrected local reconstruction | 68.68 mIoU |

The five numerical acceptance records and corresponding logs are under [results](../results/). RISP acceptance used properly installed PyTorch3D 0.7.8. GeoPCSD current records check saved-log consistency and unchanged source packaging; no numerical rerun was performed during release cleanup.

[Protocols](protocol_notes.md) specify γ selection and data behaviors. **Legacy (exact paper match):** `git checkout 51686e4` reproduces ModelNet40 85.1297 (γ6090), ShapeNet-Part 70.18, and Tables VII/VIII to two decimals. [knnxyz std correction](knnxyz_std_correction.md) lists what was and was not re-run (RISP and S3DIS few-shot: not re-run). [Correction details](geopcsd_correction.md) apply to the two corrected GeoPCSD experiments.

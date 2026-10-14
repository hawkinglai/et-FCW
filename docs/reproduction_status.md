# Result status

| Experiment | Status | Accepted or saved numbers (%) |
|---|---|---|
| ModelNet40, Table I | Corrected reproduction (per-sample std) | 85.0486; γ5790; paper 84.8. Batch-wide std regression: 85.1297; γ6090 |
| ScanObjectNN xyz, Table II | Verified reproduction (per-sample std) | OBJ-BG 65.92; OBJ-ONLY 72.29; PB-T50-RS 55.14; paper 66.3 / 71.9 / 55.6, all within 0.5 pp |
| ShapeNet-Part, Table IV | Corrected rerun (per-sample std) | 70.35 mIoU; paper 70.50; batch-wide std: 70.18 in three acceptance runs |
| RISP rotation, Table V | Verified reproduction | z/z 73.9870; z/SO(3) 74.0276; SO(3)/SO(3) 73.9465; none/none 74.0681; paper 73.9 each |
| ModelNet-C, Table VII | Batch-wide std: verified reproduction. Per-sample std: does not match | Batch-wide: all 30 corrupted and three unique clean values match to two decimals. Per-sample differs by up to 5.35 pp |
| ShapeNet-C, Table VIII | Batch-wide std: verified reproduction. Per-sample std: does not match | Batch-wide: all ten knnxyz corrupted values match to two decimals. Per-sample is 0.23–1.30 pp higher |
| GeoPCSD ScanObjectNN | Corrected local reconstruction | OBJ-BG 69.7074; OBJ-ONLY 72.9776; PB-T50-RS 58.0500 |
| GeoPCSD ShapeNet-Part | Corrected local reconstruction | 68.68 mIoU |

The five numerical acceptance records and corresponding logs are under [results](../results/). RISP acceptance used properly installed PyTorch3D 0.7.8. GeoPCSD current records check saved-log consistency and unchanged source packaging; no numerical rerun was performed during release cleanup.

[Protocols](protocol_notes.md) specify γ selection and data behaviors. [knnxyz std correction](knnxyz_std_correction.md) lists what was and was not re-run (RISP and S3DIS few-shot: not re-run). [Correction details](geopcsd_correction.md) apply to the two corrected GeoPCSD experiments.

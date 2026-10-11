# Result status

| Experiment | Status | Accepted or saved numbers (%) |
|---|---|---|
| ModelNet40, Table I | Verified reproduction | 85.1297; γ6090; paper 84.8 |
| ShapeNet-Part, Table IV | Verified reproduction | 70.18 mIoU in three acceptance runs; paper 70.50 |
| RISP rotation, Table V | Verified reproduction | z/z 73.9870; z/SO(3) 74.0276; SO(3)/SO(3) 73.9465; none/none 74.0681; paper 73.9 each |
| ModelNet-C, Table VII | Verified reproduction | All 30 corrupted and three unique clean values match to two decimals; clean checked twice |
| ShapeNet-C, Table VIII | Verified reproduction | All ten knnxyz corrupted values match to two decimals |
| GeoPCSD ScanObjectNN | Corrected local reconstruction | OBJ-BG 69.7074; OBJ-ONLY 72.9776; PB-T50-RS 58.0500 |
| GeoPCSD ShapeNet-Part | Corrected local reconstruction | 68.68 mIoU |

The five numerical acceptance records and corresponding logs are under [results](../results/). RISP acceptance used properly installed PyTorch3D 0.7.8. GeoPCSD current records check saved-log consistency and unchanged source packaging; no numerical rerun was performed during release cleanup.

[Protocols](protocol_notes.md) specify γ selection and data behaviors. [Correction details](geopcsd_correction.md) apply to the two corrected GeoPCSD experiments.

[ModelNet-C with a single train-selected γ (thesis protocol)](../../mnc_fixedgamma_v1/summary.md): NEW EXPERIMENT — NOT A REPRODUCTION. Sanity gate: 36/36 matched; determinism: 36/36 feature and label sets bit-identical. Selected γ: global 9500, local 5260, united 5440.

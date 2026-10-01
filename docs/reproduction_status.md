# Reproduction status

The public claim set is deliberately narrow. The historical `reconstruction/` code is the protocol authority, but it is an external companion and has not been relicensed for this release. Saved values and logs are in [results](../results/); source and command constraints are in [protocol notes](protocol_notes.md).

| Paper item | Public status | Supported result or limitation |
|---|---|---|
| Table I plain ModelNet40 | VERIFIED_REPRODUCTION | Paper 84.8%; measured 85.1297%; γ6090 selected using labeled test data. |
| Table IV plain ShapeNet-Part | VERIFIED_REPRODUCTION | Paper 70.50% mIoU; measured 70.18%; γ230. |
| Table V RISP rotation | CLOSE_REPRODUCTION | Paper 73.9% for all four settings; measured 74.0681, 73.9870, 74.0276, 73.9465%. |
| Table VII plain ModelNet-C | VERIFIED_REPRODUCTION | 30/30 printed values match to two decimals. The `add_global` filename mapping is functional rather than explicit in the paper. |
| Table VIII plain ShapeNet-C | VERIFIED_REPRODUCTION | 10/10 printed values match to two decimals; same filename caveat. |
| GeoPCSD ScanObjectNN | CORRECTED_RECONSTRUCTION | Corrected local implementation: 69.7074, 72.9776, 58.0500%. |
| GeoPCSD ShapeNet-Part | CORRECTED_RECONSTRUCTION | Corrected local implementation: 68.68% mIoU. |
| Table II plain ScanObjectNN | STRONG_PARTIAL_RECOVERY | Historical loaders and scoring known; exact plain K, rescale, batch size and commands unavailable. No faithful run. |
| Table III plain S3DIS | STRONG_PARTIAL_RECOVERY | Mechanics known; exact plain K, γ, decoder settings, full commands and cache identity unavailable. No faithful run. |
| Table VI adversarial | PARTIAL_REPRODUCTION | Eligible denominators differ and `pert_l2` mapping is uncertain; excluded from main public results. |
| Table V xyz and historical GeoPCSD rotation | HISTORICAL_PROTOCOL_UNAVAILABLE | Exact historical executables unavailable; rewritten diagnostics are not reproduction. |
| Tables IX/X ablations | STRONG_PARTIAL_RECOVERY | Formulations partially recoverable; exact historical executable variants unavailable. |
| Figures 5–7 and speed | UNVERIFIED / PROTOCOL-LIMITED | Stability protocol incomplete, K-curve normalization unresolved, and volume/speed are machine-specific. |

The paired GeoPCSD source comparison and one-setting S3DIS comparison remain in the separate forensic workspace. They do not establish what source was used for the published GeoPCSD numbers.

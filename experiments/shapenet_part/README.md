# ShapeNet-Part plain t-FCW

Faithful historical entry point: `bash scripts/shapenet.sh` from the external `reconstruction/` source directory. Trainval → test; surface `knnxyz`; 1024 input points; encoder K `[105,105,110,120]`; decoder K3; γ230; Torch seed 3407. Historical NumPy point sampling had no explicit top-level seed. The [result](../../results/shapenet_part/summary.json) is 70.18% mIoU versus paper 70.50%. Historical source is not duplicated here because its redistribution license is unresolved.

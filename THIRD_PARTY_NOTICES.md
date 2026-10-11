# Third-party notices

Owner code is MIT. Third-party portions retain their upstream licenses, source notices and modification headers. The following inventory covers the five plain/RISP ports; notices in the corrected GeoPCSD source are retained.

| Release file | Upstream origin | Retained license(s) |
|---|---|---|
| `experiments/modelnet40/run_cls.py` | [Point-NN](https://github.com/ZrrSkywalker/Point-NN) | MIT |
| `experiments/modelnet40/models/surfaces.py` | [Point-NN](https://github.com/ZrrSkywalker/Point-NN); [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch); [pointMLP](https://github.com/ma-xu/pointMLP-pytorch); [GBNet](https://github.com/ShiQiu0419/GBNet); [DGCNN](https://github.com/WangYueFt/dgcnn) | MIT; Apache-2.0 |
| `experiments/modelnet40/dataloaders/ModelNet.py` | [Point-NN](https://github.com/ZrrSkywalker/Point-NN) | MIT |
| `experiments/modelnet40/dataloaders/ScanObjectNN.py` | [Point-NN](https://github.com/ZrrSkywalker/Point-NN) | MIT |
| `experiments/shapenet_part/run_part.py` | [Point-NN](https://github.com/ZrrSkywalker/Point-NN) | MIT |
| `experiments/shapenet_part/models/parts.py` | [Point-NN](https://github.com/ZrrSkywalker/Point-NN); [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) | MIT |
| `experiments/shapenet_part/models/surfaces.py` | [Point-NN](https://github.com/ZrrSkywalker/Point-NN); [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch); [pointMLP](https://github.com/ma-xu/pointMLP-pytorch); [GBNet](https://github.com/ShiQiu0419/GBNet); [DGCNN](https://github.com/WangYueFt/dgcnn) | MIT; Apache-2.0 |
| `experiments/shapenet_part/dataloaders/ShapeNet.py` | [Point-NN](https://github.com/ZrrSkywalker/Point-NN); [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) | MIT |
| `experiments/risp_rotation/run_cls_ri.py` | [Point-NN](https://github.com/ZrrSkywalker/Point-NN) | MIT |
| `experiments/risp_rotation/models/surfaces.py` | [Point-NN](https://github.com/ZrrSkywalker/Point-NN); [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch); [pointMLP](https://github.com/ma-xu/pointMLP-pytorch); [GBNet](https://github.com/ShiQiu0419/GBNet); [DGCNN](https://github.com/WangYueFt/dgcnn) | MIT; Apache-2.0 |
| `experiments/risp_rotation/models/risurconv_utils.py` | [RISurConv](https://github.com/cszyzhang/RISurConv); [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) | MIT |
| `experiments/risp_rotation/dataloaders/ModelNet_ri.py` | [RISurConv](https://github.com/cszyzhang/RISurConv); [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) | MIT |
| `experiments/risp_rotation/dataloaders/risurconv_utils.py` | [RISurConv](https://github.com/cszyzhang/RISurConv); [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) | MIT |

## Added corruption-port notices

| Release file | Upstream origin | Retained license(s) |
|---|---|---|
| `experiments/modelnetc/models/surfaces.py` | [Point-NN](https://github.com/ZrrSkywalker/Point-NN); [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch); [pointMLP](https://github.com/ma-xu/pointMLP-pytorch); [GBNet](https://github.com/ShiQiu0419/GBNet); [DGCNN](https://github.com/WangYueFt/dgcnn) | MIT; Apache-2.0 |
| `experiments/modelnetc/dataloaders/ModelNet.py` | [Point-NN](https://github.com/ZrrSkywalker/Point-NN) | MIT |
| `experiments/shapenetc/models/parts.py` | [Point-NN](https://github.com/ZrrSkywalker/Point-NN); [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) | MIT |
| `experiments/shapenetc/models/surfaces.py` | [Point-NN](https://github.com/ZrrSkywalker/Point-NN); [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch); [pointMLP](https://github.com/ma-xu/pointMLP-pytorch); [GBNet](https://github.com/ShiQiu0419/GBNet); [DGCNN](https://github.com/WangYueFt/dgcnn) | MIT; Apache-2.0 |
| `experiments/shapenetc/dataloaders/ShapeNet.py` | [Point-NN](https://github.com/ZrrSkywalker/Point-NN); [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) | MIT |
| `experiments/modelnetc/run_cls_modelnetcv2.py` (notice: `run_cls_modelnetcv2.py.license`) | et-FCW; related evaluation material from [Point-NN](https://github.com/ZrrSkywalker/Point-NN) | MIT |
| `experiments/shapenetc/run_part_cv2.py` (notice: `run_part_cv2.py.license`) | et-FCW; related evaluation material from [Point-NN](https://github.com/ZrrSkywalker/Point-NN) | MIT |


Copyright notices: Point-NN © 2023 Renrui Zhang; Pointnet_Pointnet2_pytorch © 2019 benny; RISurConv © 2024 Zhiyuan Zhang; GBNet © 2019 ShiQiu0419; DGCNN © Massachusetts Institute of Technology and its affiliates. The full license texts are in `third_party/licenses/`.

PointMLP FPS/helper portions retain Apache-2.0; its upstream source/license supplies no filled copyright line or NOTICE file. Modified files are marked. Shared portions credit each identified upstream listed above.

The corruption runners retain attribution in adjacent `.license` files. Their mode-specific model and corruption-loader code are et-FCW code covered by owner authorization.

Pointnet2_PyTorch (Unlicense; pointnet2_ops 3.0.0) and PyTorch3D (BSD-3-Clause; 0.7.8) are external installed dependencies. Their license texts are included as notices; no implementation or wheel is bundled.
